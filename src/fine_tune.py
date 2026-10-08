import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
import json
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# 1. SETTINGS
# ============================================================

TRAIN_DIR = "dataset/split/train"
VALIDATION_DIR = "dataset/split/validation"

MODEL_PATH = "model/mobilenetv2_best.keras"

IMG_SIZE = (224, 224)

BATCH_SIZE = 32

EPOCHS = 15

SEED = 42


# ============================================================
# 2. CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(
    "model",
    exist_ok=True
)

os.makedirs(
    "results",
    exist_ok=True
)


# ============================================================
# 3. LOAD TRAINING DATASET
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
# 4. LOAD VALIDATION DATASET
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
# 5. CLASS NAMES
# ============================================================

class_names = train_dataset.class_names


print("\n==========================================")
print("             CLASS NAMES")
print("==========================================")

for i, name in enumerate(class_names):

    print(
        i,
        ":",
        name
    )


# ============================================================
# 6. CALCULATE CLASS WEIGHTS
# ============================================================

print("\n==========================================")
print("          CALCULATING CLASS WEIGHTS")
print("==========================================")

train_labels = []


for _, labels in train_dataset:

    train_labels.extend(
        labels.numpy()
    )


train_labels = np.array(
    train_labels
)


class_weights_array = compute_class_weight(

    class_weight="balanced",

    classes=np.unique(
        train_labels
    ),

    y=train_labels
)


class_weights = {

    int(i): float(weight)

    for i, weight
    in enumerate(class_weights_array)

}


print("\nClass weights:")


for i, weight in class_weights.items():

    print(
        f"{i} - {class_names[i]}: "
        f"{weight:.4f}"
    )


# ============================================================
# 7. LOAD TRAINED MODEL
# ============================================================

print("\n==========================================")
print("       LOADING TRAINED MOBILENETV2")
print("==========================================")

model = tf.keras.models.load_model(
    MODEL_PATH
)


print(
    "MobileNetV2 model loaded successfully!"
)


# ============================================================
# 8. FIND MOBILENETV2 BASE MODEL
# ============================================================

print("\n==========================================")
print("       FINDING BASE MODEL")
print("==========================================")


base_model = None


for layer in model.layers:

    if isinstance(
        layer,
        tf.keras.Model
    ):

        base_model = layer

        break


if base_model is None:

    raise ValueError(
        "MobileNetV2 base model "
        "was not found."
    )


print(
    "Base model found:",
    base_model.name
)


# ============================================================
# 9. ENABLE FINE-TUNING
# ============================================================

print("\n==========================================")
print("       CONFIGURING FINE-TUNING")
print("==========================================")


base_model.trainable = True


# ------------------------------------------------------------
# Freeze most of MobileNetV2
# ------------------------------------------------------------

TOTAL_LAYERS = len(
    base_model.layers
)


# Fine-tune only the last 40 layers
fine_tune_from = max(
    0,
    TOTAL_LAYERS - 40
)


print(
    "Total MobileNetV2 layers:",
    TOTAL_LAYERS
)


print(
    "Fine-tuning from layer:",
    fine_tune_from
)


for layer in base_model.layers:

    layer.trainable = False


for layer in base_model.layers[
    fine_tune_from:
]:

    layer.trainable = True


# ------------------------------------------------------------
# Keep BatchNormalization frozen
# ------------------------------------------------------------

for layer in base_model.layers:

    if isinstance(
        layer,
        layers.BatchNormalization
    ):

        layer.trainable = False


# ============================================================
# 10. COUNT TRAINABLE PARAMETERS
# ============================================================

trainable_layers = sum(

    1

    for layer
    in base_model.layers

    if layer.trainable

)


print(
    "Trainable base-model layers:",
    trainable_layers
)


# ============================================================
# 11. COMPILE FOR FINE-TUNING
# ============================================================

print("\n==========================================")
print("          COMPILING MODEL")
print("==========================================")


model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=1e-5

    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]

)


# ============================================================
# 12. CALLBACKS
# ============================================================

checkpoint = tf.keras.callbacks.ModelCheckpoint(

    "model/mobilenetv2_finetuned_best.keras",

    monitor="val_accuracy",

    mode="max",

    save_best_only=True,

    verbose=1
)


early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True,

    verbose=1
)


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.3,

    patience=2,

    min_lr=1e-7,

    verbose=1
)


# ============================================================
# 13. START FINE-TUNING
# ============================================================

print("\n==========================================")
print("       STARTING MOBILENETV2 FINE-TUNING")
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
# 14. SAVE FINAL FINE-TUNED MODEL
# ============================================================

model.save(

    "model/mobilenetv2_finetuned_final.keras"

)


# ============================================================
# 15. SAVE FINE-TUNING HISTORY
# ============================================================

history_path = (

    "results/"
    "mobilenetv2_finetuning_history.json"

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


# ============================================================
# 16. TRAINING SUMMARY
# ============================================================

best_train_accuracy = max(

    history.history[
        "accuracy"
    ]

)


best_validation_accuracy = max(

    history.history[
        "val_accuracy"
    ]

)


epochs_completed = len(

    history.history[
        "accuracy"
    ]

)


# ============================================================
# 17. FINAL OUTPUT
# ============================================================

print("\n==========================================")
print("       FINE-TUNING COMPLETED")
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

    "Epochs Completed         :",

    epochs_completed

)


print("\nModels saved:")


print(

    "Best model:"

)


print(

    "model/"
    "mobilenetv2_finetuned_best.keras"

)


print(

    "Final model:"

)


print(

    "model/"
    "mobilenetv2_finetuned_final.keras"

)


print("\n==========================================")
print("       FINE-TUNING FINISHED")
print("==========================================")