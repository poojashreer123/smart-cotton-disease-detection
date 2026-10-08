import os
import shutil
import random

# Original dataset location
SOURCE_DIR = "dataset/Original Dataset"

# New split dataset location
OUTPUT_DIR = "dataset/split"

# Split percentages
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# For reproducible splitting
random.seed(42)

# Supported image formats
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Create train, validation and test folders
for split in ["train", "validation", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

# Get class folders
classes = [
    folder for folder in os.listdir(SOURCE_DIR)
    if os.path.isdir(os.path.join(SOURCE_DIR, folder))
]

print("Starting dataset split...")
print("=" * 50)

for class_name in sorted(classes):

    source_class_path = os.path.join(SOURCE_DIR, class_name)

    # Get images
    images = [
        file for file in os.listdir(source_class_path)
        if file.lower().endswith(IMAGE_EXTENSIONS)
    ]

    # Shuffle images
    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    # Create class folders
    for split in ["train", "validation", "test"]:
        os.makedirs(
            os.path.join(OUTPUT_DIR, split, class_name),
            exist_ok=True
        )

    # Copy images
    for image in train_images:
        shutil.copy2(
            os.path.join(source_class_path, image),
            os.path.join(OUTPUT_DIR, "train", class_name, image)
        )

    for image in val_images:
        shutil.copy2(
            os.path.join(source_class_path, image),
            os.path.join(OUTPUT_DIR, "validation", class_name, image)
        )

    for image in test_images:
        shutil.copy2(
            os.path.join(source_class_path, image),
            os.path.join(OUTPUT_DIR, "test", class_name, image)
        )

    print(
        f"{class_name}: "
        f"Train={len(train_images)}, "
        f"Validation={len(val_images)}, "
        f"Test={len(test_images)}"
    )

print("=" * 50)
print("Dataset splitting completed!")