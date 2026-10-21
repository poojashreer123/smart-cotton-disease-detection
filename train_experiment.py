import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# 1. Dataset Configuration
DATA_DIR = "dataset/train_augmented/"
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
EPOCHS = 3  # Keep epochs short for testing (e.g., 3 to 5)

print("Loading dataset for multi-model experiment...")
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR, target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE, class_mode='categorical', subset='training'
)

val_generator = train_datagen.flow_from_directory(
    DATA_DIR, target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE, class_mode='categorical', subset='validation'
)

num_classes = len(train_generator.class_indices)

# 2. Define the Three Model Architectures
def get_simpler_model():
    return Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

def get_base_model():
    return Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

def get_deeper_model():
    return Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

# 3. Run Training for All Models Automatically
models_to_test = {
    "Simpler Model (Removed Layers)": get_simpler_model(),
    "Base Model (Standard)": get_base_model(),
    "Deeper Model (Added Layers)": get_deeper_model()
}

results = {}

for name, model in models_to_test.items() if hasattr(models_to_test, 'items') else models_to_test.items():
    print(f"\n==============================================")
    print(f"🚀 Training: {name}")
    print(f"==============================================")
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS
    )
    
    # Save final validation accuracy
    final_val_acc = history.history['val_accuracy'][-1] * 100
    results[name] = final_val_acc

# 4. Print Final Comparison Table
print("\n\n📊 ==============================================")
print("          ARCHITECTURAL ABLATION RESULTS          ")
print("================================================")
for name, acc in results.items():
    print(f"  * {name}: {acc:.2f}% Validation Accuracy")
print("================================================")