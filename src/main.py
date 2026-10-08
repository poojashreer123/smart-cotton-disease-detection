import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ==========================================
# SETTINGS
# ==========================================

MODEL_PATH = "model/efficientnetb0_best.keras"
IMG_SIZE = (224, 224)

# ==========================================
# CLASS NAMES
# ==========================================

CLASS_NAMES = [
    "Bacterial Blight",
    "Curl Virus",
    "Healthy Leaf",
    "Herbicide Growth Damage",
    "Leaf Hopper Jassids",
    "Leaf Redding",
    "Leaf Variegation"
]

# ==========================================
# ADVISORY INFORMATION
# ==========================================

ADVISORIES = {

    "Bacterial Blight": [
        "Monitor affected plants regularly.",
        "Maintain good field sanitation.",
        "Avoid unnecessary overhead irrigation and prolonged leaf wetness.",
        "Follow recommended disease-management practices for cotton."
    ],

    "Curl Virus": [
        "Monitor plants regularly for disease symptoms.",
        "Remove severely affected plants where recommended.",
        "Monitor and manage insect vectors according to agricultural guidance.",
        "Maintain proper field sanitation."
    ],

    "Healthy Leaf": [
        "Continue regular monitoring of the crop.",
        "Maintain proper irrigation and nutrition.",
        "Continue normal crop-management practices.",
        "Inspect plants regularly for early signs of disease or pests."
    ],

    "Herbicide Growth Damage": [
        "Review recent herbicide application and dosage.",
        "Avoid unintended herbicide drift onto cotton plants.",
        "Follow recommended herbicide application practices.",
        "Consult an agricultural expert if severe damage is observed."
    ],

    "Leaf Hopper Jassids": [
        "Monitor the crop regularly for jassid activity.",
        "Use appropriate integrated pest-management practices.",
        "Monitor heavily affected plants carefully.",
        "Use recommended pest-control measures according to local agricultural guidance."
    ],

    "Leaf Redding": [
        "Check plant nutrition and soil conditions.",
        "Maintain appropriate irrigation.",
        "Monitor plants for additional disease or pest symptoms.",
        "Consult an agricultural expert if the condition persists or spreads."
    ],

    "Leaf Variegation": [
        "Monitor affected plants regularly.",
        "Check for associated pest, disease or nutritional problems.",
        "Maintain proper crop nutrition and irrigation.",
        "Consult an agricultural expert if symptoms increase."
    ]
}

# ==========================================
# LOAD MODEL
# ==========================================

print("Loading EfficientNetB0 model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_leaf(image_path):

    if not os.path.exists(image_path):
        print("Image not found:", image_path)
        return

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Resize
    image = image.resize(IMG_SIZE)

    # Convert to array
    image_array = np.array(image)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    predictions = model.predict(image_array, verbose=0)

    predicted_index = np.argmax(predictions[0])

    predicted_condition = CLASS_NAMES[predicted_index]

    confidence = predictions[0][predicted_index] * 100

    # ==========================================
    # DISPLAY RESULT
    # ==========================================

    print("\n==========================================")
    print("       COTTON LEAF ANALYSIS")
    print("==========================================")

    print("Predicted Condition :", predicted_condition)
    print(f"Confidence          : {confidence:.2f}%")

    print("\nRecommended Advisory:")

    for i, advice in enumerate(
        ADVISORIES[predicted_condition], 1
    ):
        print(f"{i}. {advice}")

    print("==========================================")

    return predicted_condition, confidence


# ==========================================
# RUN PROGRAM
# ==========================================

if __name__ == "__main__":

    image_path = "sample_images/CV00002.jpg"

    predict_leaf(image_path)