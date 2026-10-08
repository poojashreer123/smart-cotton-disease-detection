import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ==============================
# 1. Settings
# ==============================

TEST_DIR = "dataset/split/test"
MODEL_PATH = "model/efficientnetb0_finetuned_best.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ==============================
# 2. Load test dataset
# ==============================

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_dataset.class_names

print("\nClasses:")
for i, name in enumerate(class_names):
    print(i, ":", name)

# ==============================
# 3. Load trained model
# ==============================

print("\nLoading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# ==============================
# 4. Evaluate test accuracy
# ==============================

print("\nEvaluating model...")

test_loss, test_accuracy = model.evaluate(test_dataset)

print("\n==============================")
print("TEST RESULTS")
print("==============================")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# ==============================
# 5. Get predictions
# ==============================

y_true = []
y_pred = []

for images, labels in test_dataset:

    predictions = model.predict(images, verbose=0)

    predicted_classes = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ==============================
# 6. Classification report
# ==============================

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)

print(report)

# ==============================
# 7. Confusion matrix
# ==============================

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("EfficientNetB0 Confusion Matrix")

plt.tight_layout()

# Save confusion matrix
plt.savefig("results/confusion_matrix.png", dpi=300)

plt.show()

print("\nConfusion matrix saved to:")
print("results/confusion_matrix.png")