import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ==============================
# Settings
# ==============================

MODEL_PATH = "model/efficientnetb0_best.keras"
IMAGE_PATH = "sample_images/CV00002.jpg"

IMG_SIZE = (224, 224)

# ==============================
# Class names
# ==============================

class_names = [
    "Bacterial Blight",
    "Curl Virus",
    "Healthy Leaf",
    "Herbicide Growth Damage",
    "Leaf Hopper Jassids",
    "Leaf Redding",
    "Leaf Variegation"
]

# ==============================
# Load model
# ==============================

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# ==============================
# Load image
# ==============================

if not os.path.exists(IMAGE_PATH):
    print("\nImage not found!")
    print("Please check the image path:")
    print(IMAGE_PATH)
    exit()

image = Image.open(IMAGE_PATH).convert("RGB")

# Resize image
image = image.resize(IMG_SIZE)

# Convert image to NumPy array
image_array = np.array(image)

# Add batch dimension
image_array = np.expand_dims(image_array, axis=0)

# ==============================
# Prediction
# ==============================

predictions = model.predict(image_array, verbose=0)

predicted_index = np.argmax(predictions[0])

predicted_class = class_names[predicted_index]

confidence = predictions[0][predicted_index] * 100

# ==============================
# Display result
# ==============================

print("\n==============================")
print("COTTON LEAF PREDICTION")
print("==============================")

print("Predicted Condition:", predicted_class)
print(f"Confidence: {confidence:.2f}%")

print("==============================")