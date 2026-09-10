import os

# ============================================================
# RENDER CPU OPTIMIZATION
# ============================================================
# Render Free uses CPU only. Limit TensorFlow threads so it
# doesn't consume too many CPU resources during prediction.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"

import time

from flask import Flask, render_template, request
import tensorflow as tf
from PIL import Image
import numpy as np


# ============================================================
# TENSORFLOW THREAD CONFIGURATION
# ============================================================
try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except RuntimeError:
    # Ignore if TensorFlow has already initialized its runtime.
    pass


# ============================================================
# FLASK APP
# ============================================================
app = Flask(__name__)


# ============================================================
# MODEL
# ============================================================
MODEL_PATH = "model/agrovision_mobilenetv2.keras"

print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully!")


# ============================================================
# CLASS NAMES
# ============================================================
class_names = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

print("Number of classes:", len(class_names))


# ============================================================
# CHECK MODEL OUTPUT
# ============================================================
if len(class_names) != model.output_shape[-1]:
    raise ValueError(
        f"Class mismatch! "
        f"Model has {model.output_shape[-1]} outputs, "
        f"but class_names contains {len(class_names)} classes."
    )


# ============================================================
# SOLUTIONS
# ============================================================
solutions = {
    "Apple___Apple_scab":
        "Remove affected leaves and fallen plant material. Improve air circulation and avoid wetting the leaves.",

    "Apple___Black_rot":
        "Remove infected leaves and fruit. Keep the area clean and improve air circulation around the plant.",

    "Apple___Cedar_apple_rust":
        "Remove affected leaves and keep the area around the plant clean. Follow local agricultural guidance for treatment.",

    "Cherry_(including_sour)___Powdery_mildew":
        "Remove severely affected leaves and improve sunlight and air circulation. Avoid excessive moisture around the foliage.",

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot":
        "Remove heavily affected plant material and improve field air circulation. Follow local crop-management recommendations.",

    "Corn_(maize)___Common_rust_":
        "Monitor the crop regularly and remove severely affected leaves. Follow local agricultural guidance if the disease spreads.",

    "Corn_(maize)___Northern_Leaf_Blight":
        "Remove severely affected leaves and improve field hygiene. Follow local agricultural recommendations for disease management.",

    "Grape___Black_rot":
        "Remove infected leaves and fruit and keep the area clean. Improve air circulation around the vines.",

    "Grape___Esca_(Black_Measles)":
        "Remove severely affected plant material and maintain good vineyard hygiene. Seek local agricultural guidance for management.",

    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)":
        "Remove affected leaves and improve air circulation. Avoid prolonged leaf wetness.",

    "Orange___Haunglongbing_(Citrus_greening)":
        "This condition requires professional agricultural assessment. Follow local agricultural guidance for management.",

    "Peach___Bacterial_spot":
        "Remove severely affected leaves and maintain good plant hygiene. Avoid overhead watering and improve air circulation.",

    "Pepper,_bell___Bacterial_spot":
        "Remove affected leaves and maintain good plant hygiene. Avoid overhead watering and keep foliage as dry as practical.",

    "Potato___Early_blight":
        "Remove severely affected leaves and maintain field hygiene. Avoid prolonged leaf wetness and follow local agricultural guidance.",

    "Potato___Late_blight":
        "Remove affected plant material and avoid overhead irrigation. Because this disease can spread quickly, follow local agricultural guidance.",

    "Squash___Powdery_mildew":
        "Remove severely affected leaves and improve sunlight and air circulation. Avoid excessive moisture on foliage.",

    "Strawberry___Leaf_scorch":
        "Remove severely affected leaves and maintain good plant hygiene. Improve air circulation around the plants.",

    "Tomato___Bacterial_spot":
        "Remove affected leaves and maintain good plant hygiene. Avoid overhead watering and improve air circulation.",

    "Tomato___Early_blight":
        "Remove affected leaves and keep the area clean. Avoid prolonged leaf wetness and maintain good plant spacing.",

    "Tomato___Late_blight":
        "Remove affected plant material and avoid overhead watering. Monitor the crop closely and follow local agricultural guidance.",

    "Tomato___Leaf_Mold":
        "Remove affected leaves and improve ventilation and air circulation. Avoid excessive humidity around the foliage.",

    "Tomato___Septoria_leaf_spot":
        "Remove affected leaves and keep the soil and plant area clean. Avoid overhead watering.",

    "Tomato___Spider_mites Two-spotted_spider_mite":
        "Inspect the undersides of leaves and remove heavily affected leaves. Maintain plant health and seek local guidance if infestation increases.",

    "Tomato___Target_Spot":
        "Remove affected leaves and improve air circulation. Avoid prolonged leaf wetness and maintain good plant hygiene.",

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":
        "Remove severely affected plants according to local guidance and control insect vectors such as whiteflies.",

    "Tomato___Tomato_mosaic_virus":
        "Remove severely affected plants and maintain good hygiene. Clean tools and avoid spreading plant sap between plants."
}


# ============================================================
# MODEL WARM-UP
# ============================================================
print("Starting model warm-up...")

dummy_image = np.zeros(
    (1, 224, 224, 3),
    dtype=np.float32
)

try:
    model(dummy_image, training=False)
    print("Model warm-up completed!")
except Exception as e:
    print("Warm-up error:", e)


# ============================================================
# HOME PAGE
# ============================================================
@app.route("/")
def home():
    return render_template("index.html")


# ============================================================
# PREDICTION
# ============================================================
@app.route("/predict", methods=["POST"])
def predict():

    print("---------------------------------------")
    print("Starting prediction...")

    if "leaf_image" not in request.files:
        print("ERROR: No image uploaded.")
        return render_template(
            "index.html",
            error="No image uploaded."
        )

    file = request.files["leaf_image"]

    if file.filename == "":
        print("ERROR: Empty filename.")
        return render_template(
            "index.html",
            error="Please select an image."
        )

    try:

        # ----------------------------------------------------
        # READ IMAGE
        # ----------------------------------------------------
        print("Reading image...")

        image = Image.open(file).convert("RGB")

        print("Original image size:", image.size)

        # ----------------------------------------------------
        # RESIZE
        # ----------------------------------------------------
        image = image.resize((224, 224))

        # ----------------------------------------------------
        # CONVERT TO NUMPY
        # ----------------------------------------------------
        image_array = np.array(
            image,
            dtype=np.float32
        )

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        print("Image prepared for model.")

        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------
        print("Running TensorFlow prediction...")

        start_time = time.time()

        # IMPORTANT:
        # The model already contains:
        # Rescaling(1.0 / 127.5, offset=-1)
        #
        # Therefore we DO NOT use preprocess_input here.