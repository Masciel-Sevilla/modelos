import tensorflow as tf
from tensorflow.keras import layers, Model
import os
from glob import glob
import time

# --- Configuración ---
IMG_HEIGHT = 128
IMG_WIDTH = 128
NUM_CLASSES = 6
# Modifica el BATCH_SIZE para probar diferentes rendimientos.
# Un batch más grande suele dar más FPS si la GPU lo permite.
BATCH_SIZE = 16
MODEL_PATH = "efficient_weed_model (1).keras"

# --- Rutas del conjunto de Test ---
BASE_PATH = "./Balanced"
TEST_IMAGES_PATH = os.path.join(BASE_PATH, "test/images")
TEST_MASKS_PATH = os.path.join(
    BASE_PATH, "test/masks"
)  # Necesario para la función, aunque las máscaras no se usan

# --------------------------------------------------------------------------
# Clases personalizadas (necesarias para cargar el modelo)
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


# --- Funciones de Carga de Datos y Cálculo de FPS ---
def create_dataset_for_fps(image_path, mask_path):
    image_files = sorted(glob(os.path.join(image_path, "*.jpg")))
    mask_files = sorted(glob(os.path.join(mask_path, "*.png")))
    if not image_files or not mask_files:
        return None

    def load_image(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH], method="bilinear")
        return tf.cast(img, tf.float32)

    def load_dummy_mask(path):
        # Creamos una máscara vacía, ya que no la necesitamos para predecir
        return tf.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=tf.int32)

    img_ds = tf.data.Dataset.from_tensor_slices(image_files).map(load_image)
    mask_ds = tf.data.Dataset.from_tensor_slices(mask_files).map(
        load_dummy_mask
    )  # Usamos máscaras dummy

    dataset = tf.data.Dataset.zip((img_ds, mask_ds))
    dataset = dataset.map(
        lambda img, msk: (
            tf.keras.applications.efficientnet_v2.preprocess_input(img),
            msk,
        )
    )
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    print(
        f"Dataset cargado desde '{image_path}' con {len(image_files)} imágenes para cálculo de FPS."
    )
    return dataset


def calculate_fps(model, dataset):
    num_images = 0
    inference_times = []

    print("\n -> Realizando un ciclo de calentamiento (warm-up)...")
    for images, _ in dataset.take(1):
        _ = model.predict(images, verbose=0)

    print(" -> Midiendo el tiempo de inferencia en todo el dataset...")
    for images, _ in dataset:
        start_time = time.perf_counter()
        _ = model.predict(images, verbose=0)
        end_time = time.perf_counter()

        inference_times.append(end_time - start_time)
        num_images += images.shape[0]

    total_time = sum(inference_times)

    if total_time == 0:
        return 0

    fps = num_images / total_time
    return fps


# --- Ejecución Principal ---
if __name__ == "__main__":
    # 1. Verificar si el modelo y los datos existen
    if not os.path.exists(MODEL_PATH):
        print(
            f"❌ Error: No se encontró el archivo del modelo en la ruta: '{MODEL_PATH}'"
        )
        exit()
    if not os.path.exists(TEST_IMAGES_PATH):
        print(
            f"❌ Error: No se encontró la carpeta de imágenes de test en: '{TEST_IMAGES_PATH}'"
        )
        exit()

    # 2. Cargar el modelo
    print(f"--- Cargando Modelo desde '{MODEL_PATH}' ---")
    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={
                "ASPPModule": ASPPModule,
                "DeformableAttention": DeformableAttention,
            },
            compile=False,
        )
        print("✅ ¡Modelo cargado exitosamente!")
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        exit()

    # 3. Cargar el dataset de test
    dataset_test = create_dataset_for_fps(TEST_IMAGES_PATH, TEST_MASKS_PATH)
    if not dataset_test:
        print("❌ No se pudieron cargar los datos de test. Revisa las rutas.")
        exit()

    # 4. Calcular y mostrar los FPS
    fps_result = calculate_fps(model, dataset_test)

    print("\n------------------------------------------")
    print(f"🚀 Velocidad de Inferencia del Modelo 🚀")
    print("------------------------------------------")
    print(f"   Imágenes por Segundo (FPS): {fps_result:.2f}")
    print(f"   (Calculado con un tamaño de lote de {BATCH_SIZE})")
    print("------------------------------------------\n")
