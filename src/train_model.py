import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

import matplotlib.pyplot as plt
import json
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# 1. SETTINGS
# ============================================================

TRAIN_DIR = "dataset/split/train"
VALIDATION_DIR = "dataset/split/validation"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

NUM_CLASSES = 7

EPOCHS = 20

SEED = 42


# ============================================================
# 2. CREATE DIRECTORIES
# ============================================================

os.makedirs("model", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ============================================================
# 3. PRINT TENSORFLOW INFORMATION
# ============================================================

print("\n==========================================")
print("       MOBILE NET V2 TRAINING")
print("==========================================")

print("TensorFlow version:", tf.__version__)


# ============================================================
# 4. LOAD TRAINING DATASET
# ============================================================

print("\n==========================================")
print("       LOADING TRAINING DATASET")
print("==========================================")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

# ============================================================
# 5. LOAD VALIDATION DATASET
# ============================================================

print("\n==========================================")
print("      LOADING VALIDATION DATASET")
print("==========================================")

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ============================================================
# 6. CLASS NAMES
# ============================================================

class_names = train_dataset.class_names

print("\n==========================================")
print("             CLASS NAMES")
print("==========================================")

for i, name in enumerate(class_names):
    print(i, ":", name)


# ============================================================
# 7. CHECK NUMBER OF CLASSES
# ============================================================

if len(class_names) != NUM_CLASSES:

    raise ValueError(
        f"Expected {NUM_CLASSES} classes, "
        f"but found {len(class_names)} classes."
    )


# ============================================================
# 8. CALCULATE CLASS WEIGHTS
# ============================================================

print("\n==========================================")
print("          CALCULATING CLASS WEIGHTS")
print("==========================================")


# Collect training labels
train_labels = []

for _, labels in train_dataset:
    train_labels.extend(labels.numpy())


train_labels = np.array(train_labels)


class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels
)


class_weights = {
    int(i): float(weight)
    for i, weight in enumerate(class_weights_array)
}


print("\nClass weights:")

for i, weight in class_weights.items():

    print(
        f"{i} - {class_names[i]}: "
        f"{weight:.4f}"
    )


# ============================================================
# 9. DATA AUGMENTATION
# ============================================================

print("\n==========================================")
print("          CREATING AUGMENTATION")
print("==========================================")

data_augmentation = tf.keras.Sequential([

    layers.RandomFlip(
        "horizontal"
    ),

    layers.RandomRotation(
        0.08
    ),

    layers.RandomZoom(
        0.10
    ),

    layers.RandomContrast(
        0.10
    )

], name="data_augmentation")


# ============================================================
# 10. BUILD MOBILENETV2
# ============================================================

print("\n==========================================")
print("          LOADING MOBILENETV2")
print("==========================================")

base_model = MobileNetV2(

    include_top=False,

    weights="imagenet",

    input_shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    )
)


# Freeze pretrained layers
base_model.trainable = False


print(
    "MobileNetV2 loaded successfully!"
)


# ============================================================
# 11. BUILD COMPLETE MODEL
# ============================================================

print("\n==========================================")
print("             BUILDING MODEL")
print("==========================================")

inputs = layers.Input(
    shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    )
)


# Data augmentation
x = data_augmentation(inputs)


# MobileNetV2 preprocessing
x = preprocess_input(x)


# Feature extraction
x = base_model(
    x,
    training=False
)


# Global average pooling
x = layers.GlobalAveragePooling2D()(x)


# Dropout
x = layers.Dropout(
    0.35
)(x)


# Classification layer
outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)


model = models.Model(
    inputs,
    outputs
)


# ============================================================
# 12. COMPILE MODEL
# ============================================================

print("\n==========================================")
print("             COMPILING MODEL")
print("==========================================")

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# 13. MODEL SUMMARY
# ============================================================

model.summary()


# ============================================================
# 14. CALLBACKS
# ============================================================

checkpoint = tf.keras.callbacks.ModelCheckpoint(

    "model/mobilenetv2_best.keras",

    monitor="val_accuracy",

    mode="max",

    save_best_only=True,

    verbose=1
)


early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor="val_loss",

    patience=6,

    restore_best_weights=True,

    verbose=1
)


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.3,

    patience=2,

    min_lr=1e-6,

    verbose=1
)


# ============================================================
# 15. START TRAINING
# ============================================================

print("\n==========================================")
print("       STARTING MOBILENETV2 TRAINING")
print("==========================================")

history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=EPOCHS,

    class_weight=class_weights,

    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ],

    verbose=1
)


# ============================================================
# 16. SAVE FINAL MODEL
# ============================================================

model.save(
    "model/mobilenetv2_final.keras"
)


print("\n==========================================")
print("          TRAINING COMPLETED")
print("==========================================")

print(
    "Final model saved:"
)

print(
    "model/mobilenetv2_final.keras"
)


print(
    "Best model saved:"
)

print(
    "model/mobilenetv2_best.keras"
)


# ============================================================
# 17. SAVE CLASS NAMES
# ============================================================

class_names_path = (
    "results/mobilenetv2_class_names.json"
)


with open(
    class_names_path,
    "w"
) as file:

    json.dump(
        class_names,
        file,
        indent=4
    )


print(
    "\nClass names saved to:",
    class_names_path
)


# ============================================================
# 18. SAVE TRAINING HISTORY
# ============================================================

history_path = (
    "results/mobilenetv2_training_history.json"
)


with open(
    history_path,
    "w"
) as file:

    json.dump(
        history.history,
        file,
        indent=4
    )


print(
    "Training history saved to:",
    history_path
)


# ============================================================
# 19. TRAINING ACCURACY GRAPH
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)


plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)


plt.title(
    "MobileNetV2 Training and Validation Accuracy"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Accuracy"
)


plt.legend()


plt.grid(
    True
)


plt.tight_layout()


accuracy_path = (
    "results/mobilenetv2_accuracy_graph.png"
)


plt.savefig(
    accuracy_path,
    dpi=300
)


plt.close()


print(
    "Accuracy graph saved to:",
    accuracy_path
)


# ============================================================
# 20. TRAINING LOSS GRAPH
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    history.history["loss"],
    label="Training Loss"
)


plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)


plt.title(
    "MobileNetV2 Training and Validation Loss"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Loss"
)


plt.legend()


plt.grid(
    True
)


plt.tight_layout()


loss_path = (
    "results/mobilenetv2_loss_graph.png"
)


plt.savefig(
    loss_path,
    dpi=300
)


plt.close()


print(
    "Loss graph saved to:",
    loss_path
)


# ============================================================
# 21. TRAINING SUMMARY
# ============================================================

best_train_accuracy = max(
    history.history["accuracy"]
)


best_validation_accuracy = max(
    history.history["val_accuracy"]
)


best_train_loss = min(
    history.history["loss"]
)


best_validation_loss = min(
    history.history["val_loss"]
)


epochs_completed = len(
    history.history["accuracy"]
)


print("\n==========================================")
print("          TRAINING SUMMARY")
print("==========================================")

print(
    "Best Training Accuracy   :",
    f"{best_train_accuracy * 100:.2f}%"
)


print(
    "Best Validation Accuracy :",
    f"{best_validation_accuracy * 100:.2f}%"
)


print(
    "Best Training Loss       :",
    f"{best_train_loss:.4f}"
)


print(
    "Best Validation Loss     :",
    f"{best_validation_loss:.4f}"
)


print(
    "Epochs Completed         :",
    epochs_completed
)


print("\n==========================================")
print("       TRAINING OUTPUTS SAVED")
print("==========================================")

print(
    "1. model/mobilenetv2_best.keras"
)

print(
    "2. model/mobilenetv2_final.keras"
)

print(
    "3. results/mobilenetv2_accuracy_graph.png"
)

print(
    "4. results/mobilenetv2_loss_graph.png"
)

print(
    "5. results/mobilenetv2_training_history.json"
)

print(
    "6. results/mobilenetv2_class_names.json"
)

print("\n==========================================")
print("             DONE")
print("==========================================")