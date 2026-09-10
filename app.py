from flask import Flask, render_template, request
import tensorflow as tf
from PIL import Image
import numpy as np


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

MODEL_PATH = "model/agrovision_mobilenetv2.keras"

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully!")


# =========================================================
# PLANTVILLAGE CLASS NAMES
# IMPORTANT:
# This order must match the training dataset class order.
# image_dataset_from_directory() sorts folders alphabetically.
# =========================================================

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


# =========================================================
# CHECK NUMBER OF CLASSES
# =========================================================

print("Number of classes:", len(class_names))

if len(class_names) != model.output_shape[-1]:
    raise ValueError(
        f"Class mismatch! "
        f"Model has {model.output_shape[-1]} outputs, "
        f"but class_names contains {len(class_names)} classes."
    )


# =========================================================
# DISEASE GUIDANCE
# =========================================================

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


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():
    return render_template("index.html")


# =========================================================
# PREDICTION
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    # -----------------------------------------------------
    # CHECK IMAGE
    # -----------------------------------------------------

    if "leaf_image" not in request.files:
        return render_template(
            "index.html",
            error="No image uploaded."
        )

    file = request.files["leaf_image"]

    if file.filename == "":
        return render_template(
            "index.html",
            error="Please select an image."
        )

    try:

        # -------------------------------------------------
        # OPEN IMAGE
        # -------------------------------------------------

        image = Image.open(file).convert("RGB")

        # Resize to model input size
        image = image.resize((224, 224))

        # Convert to NumPy
        image_array = np.array(image, dtype=np.float32)

        # Add batch dimension
        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # -------------------------------------------------
        # IMPORTANT
        # -------------------------------------------------
        # DO NOT use:
        #
        # tf.keras.applications.mobilenet_v2.preprocess_input()
        #
        # here.
        #
        # Your trained model already contains:
        #
        # Rescaling(1.0 / 127.5, offset=-1)
        #
        # Therefore the model performs preprocessing itself.
        # -------------------------------------------------

        # -------------------------------------------------
        # AI PREDICTION
        # -------------------------------------------------

        predictions = model.predict(
            image_array,
            verbose=0
        )

        # Get prediction index
        predicted_index = int(
            np.argmax(predictions[0])
        )

        # Get predicted class
        predicted_class = class_names[predicted_index]

        # Get confidence
        confidence = float(
            predictions[0][predicted_index]
        ) * 100


        # -------------------------------------------------
        # SPLIT CROP AND CONDITION
        # -------------------------------------------------

        parts = predicted_class.split("___", 1)

        if len(parts) != 2:
            raise ValueError(
                "Invalid class name format: "
                + predicted_class
            )

        crop = parts[0]
        condition = parts[1]


        # -------------------------------------------------
        # FRIENDLY CROP NAME
        # -------------------------------------------------

        crop = crop.replace("_", " ")
        crop = crop.replace("(maize)", "")
        crop = crop.replace("(including sour)", "")
        crop = crop.replace(", bell", "")

        crop = " ".join(crop.split())


        # -------------------------------------------------
        # HEALTHY PLANT
        # -------------------------------------------------

        if condition.lower() == "healthy":

            status = "Healthy"

            disease = "No disease detected"

            solution = (
                "Your plant appears healthy. "
                "Continue regular watering, proper nutrition, "
                "good sunlight, and regular monitoring."
            )


        # -------------------------------------------------
        # DISEASED PLANT
        # -------------------------------------------------

        else:

            status = "Disease Detected"

            disease = condition.replace("_", " ")

            disease = " ".join(
                disease.split()
            )

            solution = solutions.get(
                predicted_class,
                "Maintain good plant hygiene, remove severely "
                "affected plant material, improve air circulation, "
                "and consult local agricultural guidance if symptoms continue."
            )


        # -------------------------------------------------
        # PRINT RESULT IN TERMINAL
        # -------------------------------------------------

        print("---------------------------------------")
        print("Predicted class :", predicted_class)
        print("Crop            :", crop)
        print("Condition       :", disease)
        print("Confidence      :", round(confidence, 2), "%")
        print("---------------------------------------")


        # -------------------------------------------------
        # SEND RESULT TO HTML
        # -------------------------------------------------

        return render_template(
            "index.html",

            prediction=predicted_class,

            crop=crop,

            status=status,

            disease=disease,

            confidence=round(confidence, 2),

            solution=solution
        )


    # =====================================================
    # ERROR HANDLING
    # =====================================================

    except Exception as e:

        print("ERROR:", e)

        return render_template(
            "index.html",
            error=f"Error processing image: {e}"
        )


# =========================================================
# RUN FLASK
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=False
    )