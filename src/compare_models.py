
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Project folders
ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
TEST_DIR = ROOT / "dataset" / "split" / "test"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

# Load test images in a fixed order
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="int"
)

class_names = test_ds.class_names
y_true = np.concatenate([labels.numpy() for _, labels in test_ds])

print("\nClasses:", class_names)
print("Test images:", len(y_true))

# EfficientNetB0 and MobileNetV2 saved models
model_files = {
    "MobileNetV2": MODEL_DIR / "mobilenetv2_finetuned_best.keras",
    "EfficientNetB0": MODEL_DIR / "efficientnetb0_finetuned_best.keras",
}

summary = []

for model_name, model_path in model_files.items():
    print(f"\n{'=' * 50}")
    print("Evaluating:", model_name)

    if not model_path.exists():
        print("Model file not found:", model_path)
        continue

    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as exc:
        print("Could not load model:", exc)
        continue

    print("Model size (MB):", round(model_path.stat().st_size / (1024**2), 2))

    # Measure inference time over the same test images
    start = time.perf_counter()
    predictions = model.predict(test_ds, verbose=1)
    elapsed = time.perf_counter() - start

    y_pred = np.argmax(predictions, axis=1)

    if len(y_pred) != len(y_true):
        print("Prediction count does not match test labels. Skipping.")
        continue

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    avg_time = elapsed / len(y_true) * 1000

    print(f"\n{model_name} results")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Macro precision: {precision * 100:.2f}%")
    print(f"Macro recall: {recall * 100:.2f}%")
    print(f"Macro F1-score: {f1 * 100:.2f}%")
    print(f"Average inference time: {avg_time:.2f} ms/image")

    print("\nPer-class report:")
    report = classification_report(
        y_true, y_pred, labels=range(len(class_names)),
        target_names=class_names, zero_division=0
    )
    print(report)

    (RESULTS_DIR / f"{model_name.lower()}_classification_report.txt").write_text(
        report, encoding="utf-8"
    )

    # Save confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=class_names
    )
    fig, ax = plt.subplots(figsize=(11, 9))
    display.plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(f"{model_name} Confusion Matrix")
    fig.tight_layout()
    fig.savefig(
        RESULTS_DIR / f"{model_name.lower()}_confusion_matrix.png",
        dpi=160
    )
    plt.close(fig)

    summary.append({
        "Model": model_name,
        "Accuracy (%)": round(accuracy * 100, 2),
        "Macro Precision (%)": round(precision * 100, 2),
        "Macro Recall (%)": round(recall * 100, 2),
        "Macro F1 (%)": round(f1 * 100, 2),
        "Model Size (MB)": round(model_path.stat().st_size / (1024**2), 2),
        "Avg Inference (ms/image)": round(avg_time, 2),
        "Test Images": len(y_true)
    })

    del model
    tf.keras.backend.clear_session()

if summary:
    results_df = pd.DataFrame(summary)
    output_csv = RESULTS_DIR / "model_comparison.csv"
    results_df.to_csv(output_csv, index=False)

    print("\n\nFINAL MODEL COMPARISON")
    print(results_df.to_string(index=False))
    print("\nSaved comparison:", output_csv)
    print("Saved reports and confusion matrices in:", RESULTS_DIR)
else:
    print("No models could be evaluated. Check the model filenames and errors.")
