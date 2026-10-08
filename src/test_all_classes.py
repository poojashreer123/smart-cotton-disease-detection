import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# 1. SETTINGS
# ==========================================

TEST_DIR = "dataset/split/test"

# Use the BEST fine-tuned MobileNetV2 model
MODEL_PATH = "model/mobilenetv2_finetuned_best.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ==========================================
# 2. CLASS NAMES
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
# 3. LOAD TEST DATASET
# ==========================================

print("\n==========================================")
print("        LOADING TEST DATASET")
print("==========================================")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nTest dataset loaded successfully!")

print("\nClasses found:")

for i, name in enumerate(test_dataset.class_names):
    print(i, ":", name)

# ==========================================
# 4. CHECK CLASS ORDER
# ==========================================

if test_dataset.class_names != CLASS_NAMES:
    print("\nWARNING: Dataset class order is different!")
    print("Dataset classes:", test_dataset.class_names)
    print("Expected classes:", CLASS_NAMES)

# ==========================================
# 5. LOAD MODEL
# ==========================================

print("\n==========================================")
print("      LOADING FINE-TUNED MOBILENETV2")
print("==========================================")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("\nFine-tuned MobileNetV2 loaded successfully!")

# ==========================================
# 6. MODEL SUMMARY
# ==========================================

print("\n==========================================")
print("             MODEL INFORMATION")
print("==========================================")

print("Model:", MODEL_PATH)

print("Input Size:", IMG_SIZE)

print("Number of Classes:", len(CLASS_NAMES))

# ==========================================
# 7. MODEL EVALUATION
# ==========================================

print("\n==========================================")
print("          MODEL EVALUATION")
print("==========================================")

test_loss, test_accuracy = model.evaluate(
    test_dataset,
    verbose=1
)

print("\nTest Loss     :", f"{test_loss:.4f}")

print(
    "Test Accuracy :",
    f"{test_accuracy * 100:.2f}%"
)

# ==========================================
# 8. GENERATE PREDICTIONS
# ==========================================

print("\n==========================================")
print("       GENERATING PREDICTIONS")
print("==========================================")

y_true = []
y_pred = []
confidences = []

for images, labels in test_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    prediction_confidence = np.max(
        predictions,
        axis=1
    )

    y_true.extend(
        labels.numpy()
    )

    y_pred.extend(
        predicted_classes
    )

    confidences.extend(
        prediction_confidence
    )

# Convert to NumPy arrays

y_true = np.array(y_true)

y_pred = np.array(y_pred)

confidences = np.array(confidences)

print("Predictions generated successfully!")

# ==========================================
# 9. CHECK NUMBER OF PREDICTIONS
# ==========================================

print("\nTotal actual labels      :", len(y_true))

print("Total predicted labels   :", len(y_pred))

print("Total confidence values  :", len(confidences))

# ==========================================
# 10. CLASSIFICATION REPORT
# ==========================================

print("\n==========================================")
print("        CLASSIFICATION REPORT")
print("==========================================")

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4
)

print(report)

# ==========================================
# 11. CONFUSION MATRIX
# ==========================================

print("\n==========================================")
print("          CONFUSION MATRIX")
print("==========================================")

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nRows    = Actual Class")

print("Columns = Predicted Class\n")

print(cm)

# ==========================================
# 12. CLASS-WISE ACCURACY
# ==========================================

print("\n==========================================")
print("        CLASS-WISE ACCURACY")
print("==========================================")

for i, class_name in enumerate(CLASS_NAMES):

    total_samples = np.sum(
        y_true == i
    )

    correct_samples = np.sum(
        (y_true == i) &
        (y_pred == i)
    )

    if total_samples > 0:

        class_accuracy = (
            correct_samples /
            total_samples
        ) * 100

        print(
            f"{class_name}: "
            f"{class_accuracy:.2f}% "
            f"({correct_samples}/{total_samples})"
        )

# ==========================================
# 13. CONFIDENCE ANALYSIS
# ==========================================

average_confidence = (
    np.mean(confidences) * 100
)

print("\n==========================================")
print("        CONFIDENCE ANALYSIS")
print("==========================================")

print(
    "Average Prediction Confidence:",
    f"{average_confidence:.2f}%"
)

# ==========================================
# 14. FINAL TEST SUMMARY
# ==========================================

total_images = len(y_true)

correct_predictions = int(
    np.sum(y_true == y_pred)
)

incorrect_predictions = int(
    np.sum(y_true != y_pred)
)

print("\n==========================================")
print("          FINAL TEST SUMMARY")
print("==========================================")

print(
    "Total Test Images     :",
    total_images
)

print(
    "Correct Predictions   :",
    correct_predictions
)

print(
    "Incorrect Predictions :",
    incorrect_predictions
)

print(
    "Overall Accuracy      :",
    f"{test_accuracy * 100:.2f}%"
)

0  ,                                                                                                                                                                                                                                                                                                                                                                                                                             

print(
    "Average Confidence    :",
    f"{average_confidence:.2f}%"
)

# ==========================================
# 15. VERIFICATION
# ==========================================

print("\n==========================================")
print("             VERIFICATION")
print("==========================================")

if (
    correct_predictions +
    incorrect_predictions
) == total_images:

    print(
        "Prediction count verified successfully!"
    )

else:

    print(
        "WARNING: Prediction count mismatch!"
    )

# ==========================================
# 16. FINAL MESSAGE
# ==========================================

print("\n==========================================")
print("       TESTING COMPLETED SUCCESSFULLY")
print("==========================================")