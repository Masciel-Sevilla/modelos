import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import os
from glob import glob
import pandas as pd

# --- Configuración ---
IMG_HEIGHT = 128
IMG_WIDTH = 128
NUM_CLASSES = 6
BATCH_SIZE = 16
MODEL_PATH = "efficient_weed_model (1).keras"  # Ruta a tu modelo entrenado

# --- Rutas de todos los conjuntos de datos ---
BASE_PATH = "./Balanced"
TRAIN_IMAGES_PATH = os.path.join(BASE_PATH, "train/images")
TRAIN_MASKS_PATH = os.path.join(BASE_PATH, "train/masks")
VAL_IMAGES_PATH = os.path.join(BASE_PATH, "val/images")
VAL_MASKS_PATH = os.path.join(BASE_PATH, "val/masks")
TEST_IMAGES_PATH = os.path.join(BASE_PATH, "test/images")
TEST_MASKS_PATH = os.path.join(BASE_PATH, "test/masks")

# --------------------------------------------------------------------------
# Es necesario definir todas las clases y funciones personalizadas para que
# Keras pueda cargar el modelo correctamente.
# --------------------------------------------------------------------------


# --- Módulos del modelo ---
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
        output = self.dropout(output, training=training)
        return output


class DeformableAttention(layers.Layer):
    def __init__(self, filters, **kwargs):
        super(DeformableAttention, self).__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        self.attention_conv = layers.Conv2D(
            self.filters,
            1,
            padding="same",
            activation="sigmoid",
            name="attention_weights_conv",
            use_bias=False,
        )
        self.bn_attention = layers.BatchNormalization()
        self.feature_conv = layers.SeparableConv2D(
            self.filters,
            3,
            padding="same",
            name="feature_processing_conv",
            use_bias=False,
        )
        self.bn_feature = layers.BatchNormalization()
        self.relu_feature = layers.ReLU()
        super(DeformableAttention, self).build(input_shape)

    def call(self, inputs, training=None):
        attention_weights = self.bn_attention(
            self.attention_conv(inputs), training=training
        )
        features = self.relu_feature(
            self.bn_feature(self.feature_conv(inputs), training=training)
        )
        attended_features = features * attention_weights
        return attended_features


# --- Métricas y Funciones de Pérdida ---
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1, NUM_CLASSES])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1, NUM_CLASSES])
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    dice_scores = (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f, axis=0) + tf.reduce_sum(y_pred_f, axis=0) + smooth
    )
    return tf.reduce_mean(dice_scores)


def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        loss = alpha * tf.pow(1 - y_pred, gamma) * cross_entropy
        return loss

    return focal_loss_fixed


def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)


def combined_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    class_weights = tf.constant([0.4, 4.0, 3.5, 1.0, 2.0, 4.0])
    f_loss_per_pixel_per_class = focal_loss(gamma=2.0, alpha=0.75)(y_true, y_pred)
    weighted_f_loss = f_loss_per_pixel_per_class * class_weights
    f_loss_reduced = tf.reduce_sum(weighted_f_loss, axis=-1)
    f_loss_mean = tf.reduce_mean(f_loss_reduced)
    d_loss = dice_loss(y_true, y_pred)
    return f_loss_mean + (1.5 * d_loss)


def iou_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.one_hot(tf.argmax(y_pred, axis=-1), depth=NUM_CLASSES)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = (
        tf.reduce_sum(y_true, axis=[1, 2, 3])
        + tf.reduce_sum(y_pred, axis=[1, 2, 3])
        - intersection
    )
    iou = tf.where(tf.equal(union, 0), 1.0, intersection / union)
    return tf.reduce_mean(iou)


def iou_per_class(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.one_hot(tf.argmax(y_pred, axis=-1), depth=NUM_CLASSES)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    union = (
        tf.reduce_sum(y_true, axis=[1, 2])
        + tf.reduce_sum(y_pred, axis=[1, 2])
        - intersection
    )
    iou = tf.where(tf.equal(union, 0), 1.0, intersection / union)
    return tf.reduce_mean(iou, axis=0)


# --- Función para Cargar un Dataset Específico ---
def create_dataset(image_path, mask_path):
    """Carga imágenes y máscaras, y crea un tf.data.Dataset."""
    image_files = sorted(glob(os.path.join(image_path, "*.jpg")))
    mask_files = sorted(glob(os.path.join(mask_path, "*.png")))

    if not image_files or not mask_files:
        return None

    def load_image_only(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH], method="bilinear")
        return tf.cast(image, tf.float32)

    def load_mask_only(path):
        mask = tf.io.read_file(path)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.image.resize(mask, [IMG_HEIGHT, IMG_WIDTH], method="nearest")
        mask = tf.cast(mask, tf.int32)
        mask = tf.squeeze(mask, axis=-1)
        mask = tf.one_hot(mask, NUM_CLASSES)
        return mask

    img_ds = tf.data.Dataset.from_tensor_slices(image_files).map(
        load_image_only, num_parallel_calls=tf.data.AUTOTUNE
    )
    mask_ds = tf.data.Dataset.from_tensor_slices(mask_files).map(
        load_mask_only, num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = tf.data.Dataset.zip((img_ds, mask_ds))
    # Para evaluación, solo preprocesamos la imagen, sin aumentos de datos
    dataset = dataset.map(
        lambda img, msk: (
            tf.keras.applications.efficientnet_v2.preprocess_input(img),
            msk,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    print(f"Dataset cargado desde '{image_path}' con {len(image_files)} imágenes.")
    return dataset


# --- Función Principal de Ejecución ---
if __name__ == "__main__":
    # 1. Verificar si el modelo existe
    if not os.path.exists(MODEL_PATH):
        print(
            f"❌ Error: No se encontró el archivo del modelo en la ruta: '{MODEL_PATH}'"
        )
        exit()

    # 2. Cargar todos los datasets
    print("--- Cargando Datasets ---")
    train_ds = create_dataset(TRAIN_IMAGES_PATH, TRAIN_MASKS_PATH)
    val_ds = create_dataset(VAL_IMAGES_PATH, VAL_MASKS_PATH)
    test_ds = create_dataset(TEST_IMAGES_PATH, TEST_MASKS_PATH)
    print("-------------------------\n")

    # 3. Cargar el modelo entrenado
    print(f"--- Cargando Modelo Entrenado desde '{MODEL_PATH}' ---")
    try:
        custom_objects = {
            "combined_loss": combined_loss,
            "dice_coefficient": dice_coefficient,
            "iou_metric": iou_metric,
            "ASPPModule": ASPPModule,
            "DeformableAttention": DeformableAttention,
        }
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
        print("✅ ¡Modelo cargado exitosamente!")
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        exit()
    print("------------------------------------------------\n")

    # 4. Evaluar en todos los conjuntos y almacenar resultados
    results = {}
    datasets = {"Entrenamiento": train_ds, "Validación": val_ds, "Test": test_ds}

    for name, ds in datasets.items():
        if ds:
            print(f"--- Evaluando en el conjunto de {name} ---")
            metrics = model.evaluate(ds, verbose=1)
            results[name] = {
                "Loss": metrics[0],
                "Dice Coefficient": metrics[1],
                "IoU (Mean)": metrics[2],
                "Accuracy": metrics[3],
            }
        else:
            print(
                f"⚠️  Saltando evaluación para el conjunto de {name} (no se encontraron datos)."
            )

    # 5. Mostrar tabla de métricas generales
    print("\n\n--- 📊 Resumen de Métricas Generales ---")
    if results:
        df_results = pd.DataFrame.from_dict(results, orient="index")
        df_results = df_results.round(4)  # Redondear a 4 decimales
        print(df_results)
    else:
        print("No se pudieron calcular resultados.")
    print("--------------------------------------\n")

    # 6. Calcular y mostrar tabla de IoU por clase
    iou_results = {}
    class_names = ["Background", "Cow-tongue", "Dandelion", "Kikuyo", "Other", "Potato"]

    for name, ds in datasets.items():
        if ds:
            print(f"--- Calculando IoU por clase para {name} ---")
            all_iou_values = []
            for images_batch, masks_batch in ds:
                predictions_batch = model.predict(images_batch, verbose=0)
                all_iou_values.append(iou_per_class(masks_batch, predictions_batch))

            mean_iou_per_class = tf.reduce_mean(
                tf.stack(all_iou_values), axis=0
            ).numpy()
            iou_results[name] = mean_iou_per_class

    print("\n--- 📈 Resumen de IoU por Clase ---")
    if iou_results:
        df_iou = pd.DataFrame.from_dict(
            iou_results, orient="index", columns=class_names
        )
        df_iou = df_iou.round(4)  # Redondear a 4 decimales
        print(df_iou)
    else:
        print("No se pudieron calcular los valores de IoU.")
    print("----------------------------------\n")

    print("✅ Proceso de evaluación finalizado.")
