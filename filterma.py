import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import os
from glob import glob
import pandas as pd
from scipy.ndimage import generic_filter

# --- Configuración ---
IMG_HEIGHT = 128
IMG_WIDTH = 128
NUM_CLASSES = 6
BATCH_SIZE = 16
MODEL_PATH = "efficient_weed_model (1).keras"

# --- Rutas de todos los conjuntos de datos ---
BASE_PATH = "./Balanced"
TRAIN_IMAGES_PATH = os.path.join(BASE_PATH, "train/images")
TRAIN_MASKS_PATH = os.path.join(BASE_PATH, "train/masks")
VAL_IMAGES_PATH = os.path.join(BASE_PATH, "val/images")
VAL_MASKS_PATH = os.path.join(BASE_PATH, "val/masks")
TEST_IMAGES_PATH = os.path.join(BASE_PATH, "test/images")
TEST_MASKS_PATH = os.path.join(BASE_PATH, "test/masks")

# --------------------------------------------------------------------------
# Clases y funciones personalizadas
# --------------------------------------------------------------------------


class ASPPModule(layers.Layer):
    def __init__(self, filters=192, **kwargs):
        super(ASPPModule, self).__init__(**kwargs)
        self.filters = filters
        self.conv_1x1 = layers.Conv2D(filters, 1, padding="same", use_bias=False)
        self.bn_1x1 = layers.BatchNormalization()
        self.relu_1x1 = layers.ReLU()
        self.conv_3x3_6 = layers.Conv2D(
            filters, 3, padding="same", dilation_rate=6, use_bias=False
        )
        self.bn_3x3_6 = layers.BatchNormalization()
        self.relu_3x3_6 = layers.ReLU()
        self.conv_3x3_12 = layers.Conv2D(
            filters, 3, padding="same", dilation_rate=12, use_bias=False
        )
        self.bn_3x3_12 = layers.BatchNormalization()
        self.relu_3x3_12 = layers.ReLU()
        self.conv_3x3_18 = layers.Conv2D(
            filters, 3, padding="same", dilation_rate=18, use_bias=False
        )
        self.bn_3x3_18 = layers.BatchNormalization()
        self.relu_3x3_18 = layers.ReLU()
        self.global_avg_pool = layers.GlobalAveragePooling2D(keepdims=True)
        self.conv_1x1_gap = layers.Conv2D(filters, 1, padding="same", use_bias=False)
        self.bn_1x1_gap = layers.BatchNormalization()
        self.relu_1x1_gap = layers.ReLU()
        self.conv_final = layers.Conv2D(filters, 1, padding="same", use_bias=False)
        self.bn_final = layers.BatchNormalization()
        self.relu_final = layers.ReLU()
        self.dropout = layers.Dropout(0.2)

    def call(self, inputs, training=None):
        input_shape = tf.shape(inputs)
        conv_1x1 = self.relu_1x1(self.bn_1x1(self.conv_1x1(inputs), training=training))
        conv_3x3_6 = self.relu_3x3_6(
            self.bn_3x3_6(self.conv_3x3_6(inputs), training=training)
        )
        conv_3x3_12 = self.relu_3x3_12(
            self.bn_3x3_12(self.conv_3x3_12(inputs), training=training)
        )
        conv_3x3_18 = self.relu_3x3_18(
            self.bn_3x3_18(self.conv_3x3_18(inputs), training=training)
        )
        gap = self.global_avg_pool(inputs)
        gap = self.relu_1x1_gap(
            self.bn_1x1_gap(self.conv_1x1_gap(gap), training=training)
        )
        gap = tf.image.resize(gap, [input_shape[1], input_shape[2]], method="bilinear")
        concat = layers.Concatenate()(
            [conv_1x1, conv_3x3_6, conv_3x3_12, conv_3x3_18, gap]
        )
        output = self.relu_final(
            self.bn_final(self.conv_final(concat), training=training)
        )
        return self.dropout(output, training=training)


class DeformableAttention(layers.Layer):
    def __init__(self, filters, **kwargs):
        super(DeformableAttention, self).__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        self.attention_conv = layers.Conv2D(
            self.filters, 1, padding="same", activation="sigmoid", use_bias=False
        )
        self.bn_attention = layers.BatchNormalization()
        self.feature_conv = layers.SeparableConv2D(
            self.filters, 3, padding="same", use_bias=False
        )
        self.bn_feature = layers.BatchNormalization()
        self.relu_feature = layers.ReLU()

    def call(self, inputs, training=None):
        attention_weights = self.bn_attention(
            self.attention_conv(inputs), training=training
        )
        features = self.relu_feature(
            self.bn_feature(self.feature_conv(inputs), training=training)
        )
        return features * attention_weights


# --- Métricas y Filtro ---
def dice_coefficient_v1(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1, NUM_CLASSES])
    y_pred_f = tf.reshape(y_pred, [-1, NUM_CLASSES])
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    dice = (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f, axis=0) + tf.reduce_sum(y_pred_f, axis=0) + smooth
    )
    return tf.reduce_mean(dice)


def iou_metric_v1(y_true, y_pred):
    # Convertir y_true a float32 para que coincida con y_pred
    y_true = tf.cast(y_true, tf.float32)  # <-- CORRECCIÓN

    y_pred = tf.one_hot(tf.argmax(y_pred, axis=-1), depth=NUM_CLASSES)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = (
        tf.reduce_sum(y_true, axis=[1, 2, 3])
        + tf.reduce_sum(y_pred, axis=[1, 2, 3])
        - intersection
    )
    iou = tf.where(tf.equal(union, 0), 1.0, intersection / union)
    return tf.reduce_mean(iou)


def dice_coefficient_v2(y_true, y_pred, num_classes=NUM_CLASSES):
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.cast(y_pred, tf.int32)
    y_true_oh = tf.one_hot(y_true, num_classes, axis=-1)
    y_pred_oh = tf.one_hot(y_pred, num_classes, axis=-1)
    intersection = tf.reduce_sum(
        tf.cast(y_true_oh * y_pred_oh, tf.float32), axis=[0, 1, 2]
    )
    union = tf.reduce_sum(
        tf.cast(y_true_oh, tf.float32), axis=[0, 1, 2]
    ) + tf.reduce_sum(tf.cast(y_pred_oh, tf.float32), axis=[0, 1, 2])
    dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
    return tf.reduce_mean(dice)


def iou_metric_v2(y_true, y_pred, num_classes=NUM_CLASSES):
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.cast(y_pred, tf.int32)
    y_true_oh = tf.one_hot(y_true, num_classes, axis=-1)
    y_pred_oh = tf.one_hot(y_pred, num_classes, axis=-1)
    intersection = tf.reduce_sum(
        tf.cast(y_true_oh * y_pred_oh, tf.float32), axis=[0, 1, 2]
    )
    union = (
        tf.reduce_sum(tf.cast(y_true_oh, tf.float32), axis=[0, 1, 2])
        + tf.reduce_sum(tf.cast(y_pred_oh, tf.float32), axis=[0, 1, 2])
        - intersection
    )
    iou = (intersection + 1e-6) / (union + 1e-6)
    return tf.reduce_mean(iou)


def filtro_mayoria(mask, kernel_size=5):
    def filtro_local(values):
        values = values.astype(int)
        counts = np.bincount(values, minlength=NUM_CLASSES)
        return np.argmax(counts)

    return generic_filter(mask, filtro_local, size=kernel_size, mode="nearest")


# --- Carga de Datos ---
def create_dataset(image_path, mask_path, one_hot_mask=True):
    image_files = sorted(glob(os.path.join(image_path, "*.jpg")))
    mask_files = sorted(glob(os.path.join(mask_path, "*.png")))
    if not image_files or not mask_files:
        return None

    def load_image(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH], method="bilinear")
        return tf.cast(img, tf.float32)

    def load_mask(path):
        msk = tf.io.read_file(path)
        msk = tf.image.decode_png(msk, channels=1)
        msk = tf.image.resize(msk, [IMG_HEIGHT, IMG_WIDTH], method="nearest")
        if one_hot_mask:
            msk = tf.one_hot(tf.squeeze(msk), depth=NUM_CLASSES)
            # En V1 necesitamos que sea float para las métricas
            return tf.cast(msk, tf.float32)
        else:
            msk = tf.squeeze(msk, axis=-1)
            # En V2 necesitamos que sea int para el argmax y one-hot manual
            return tf.cast(msk, tf.int32)

    img_ds = tf.data.Dataset.from_tensor_slices(image_files).map(load_image)
    mask_ds = tf.data.Dataset.from_tensor_slices(mask_files).map(load_mask)
    dataset = tf.data.Dataset.zip((img_ds, mask_ds))
    dataset = dataset.map(
        lambda img, msk: (
            tf.keras.applications.efficientnet_v2.preprocess_input(img),
            msk,
        )
    )
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    print(
        f"Dataset cargado desde '{image_path}' para {'evaluación por lotes' if one_hot_mask else 'evaluación global'}."
    )
    return dataset


# --- Función de Evaluación Global (para Tabla 2) ---
def evaluate_globally_for_filter(model, dataset):
    all_true_masks, all_pred_masks_no_filter = [], []
    for images, true_masks in dataset:
        pred_masks_no_filter = np.argmax(model.predict(images, verbose=0), axis=-1)
        all_true_masks.append(true_masks.numpy())
        all_pred_masks_no_filter.append(pred_masks_no_filter)

    y_true = np.concatenate(all_true_masks, axis=0)
    y_pred_no_filter = np.concatenate(all_pred_masks_no_filter, axis=0)
    y_pred_filtered = np.array([filtro_mayoria(p) for p in y_pred_no_filter])

    return {
        "Dice (Sin Filtro)": dice_coefficient_v2(y_true, y_pred_no_filter).numpy(),
        "IoU (Sin Filtro)": iou_metric_v2(y_true, y_pred_no_filter).numpy(),
        "Dice (Con Filtro)": dice_coefficient_v2(y_true, y_pred_filtered).numpy(),
        "IoU (Con Filtro)": iou_metric_v2(y_true, y_pred_filtered).numpy(),
    }


# --- Ejecución Principal ---
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(
            f"❌ Error: No se encontró el archivo del modelo en la ruta: '{MODEL_PATH}'"
        )
        exit()

    # --- 1. CÁLCULO DE MÉTRICAS ORIGINALES (POR LOTES) ---
    print("--- Cargando Modelo para Evaluación Original (por lotes) ---")
    try:
        model_v1 = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={
                "ASPPModule": ASPPModule,
                "DeformableAttention": DeformableAttention,
            },
            compile=False,
        )  # Se carga sin compilar

        # Se compila aquí con las métricas deseadas
        model_v1.compile(
            loss=lambda y_true, y_pred: 0.0,  # Loss no se usa, pero es necesaria
            metrics=[dice_coefficient_v1, iou_metric_v1, "accuracy"],
        )
        print("✅ ¡Modelo cargado y compilado para evaluación por lotes!")
    except Exception as e:
        print(f"❌ Error al cargar el modelo para v1: {e}")
        exit()

    results_v1 = {}
    datasets_v1 = {
        "Entrenamiento": create_dataset(
            TRAIN_IMAGES_PATH, TRAIN_MASKS_PATH, one_hot_mask=True
        ),
        "Validación": create_dataset(
            VAL_IMAGES_PATH, VAL_MASKS_PATH, one_hot_mask=True
        ),
        "Test": create_dataset(TEST_IMAGES_PATH, TEST_MASKS_PATH, one_hot_mask=True),
    }

    for name, ds in datasets_v1.items():
        if ds:
            print(f"--- Evaluando (por lotes) en {name} ---")
            metrics = model_v1.evaluate(ds, verbose=1)
            results_v1[name] = {
                "Loss": metrics[0],
                "Dice Coefficient": metrics[1],
                "IoU (Mean)": metrics[2],
                "Accuracy": metrics[3],
            }

    print("\n\n--- 📋 Tabla 1: Resumen de Métricas Originales (Cálculo por Lotes) ---")
    if results_v1:
        df_results_v1 = pd.DataFrame.from_dict(results_v1, orient="index")
        print(df_results_v1.round(4))
    else:
        print("No se pudieron calcular resultados.")
    print("---------------------------------------------------------------------\n")

    # --- 2. CÁLCULO DE MÉTRICAS GLOBALES (CON/SIN FILTRO) ---
    print("--- Re-cargando Modelo para Evaluación Global ---")
    try:
        model_v2 = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={
                "ASPPModule": ASPPModule,
                "DeformableAttention": DeformableAttention,
            },
            compile=False,
        )
        print("✅ ¡Modelo cargado para evaluación global!")
    except Exception as e:
        print(f"❌ Error al cargar el modelo para v2: {e}")
        exit()

    results_v2 = {}
    datasets_v2 = {
        "Entrenamiento": create_dataset(
            TRAIN_IMAGES_PATH, TRAIN_MASKS_PATH, one_hot_mask=False
        ),
        "Validación": create_dataset(
            VAL_IMAGES_PATH, VAL_MASKS_PATH, one_hot_mask=False
        ),
        "Test": create_dataset(TEST_IMAGES_PATH, TEST_MASKS_PATH, one_hot_mask=False),
    }

    for name, ds in datasets_v2.items():
        if ds:
            print(f"--- Evaluando (globalmente) en {name}... (Esto puede tardar) ---")
            metrics = evaluate_globally_for_filter(model_v2, ds)
            results_v2[name] = metrics

    print("\n\n--- 📊 Tabla 2: Análisis de Filtro (Cálculo Global) ---")
    if results_v2:
        df_results_v2 = pd.DataFrame.from_dict(results_v2, orient="index")
        print(df_results_v2.round(4))
    else:
        print("No se pudieron calcular resultados.")
    print("--------------------------------------------------------\n")

    print("✅ Proceso de evaluación finalizado.")
