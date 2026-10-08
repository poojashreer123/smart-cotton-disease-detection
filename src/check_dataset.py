import os

# Path to the dataset
dataset_path = "dataset/Original Dataset"

# Get all class folders
classes = [
    folder for folder in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, folder))
]

print("Cotton Disease Dataset")
print("=" * 40)

total_images = 0

for class_name in sorted(classes):
    class_path = os.path.join(dataset_path, class_name)

    images = [
        file for file in os.listdir(class_path)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    count = len(images)
    total_images += count

    print(f"{class_name}: {count} images")

print("=" * 40)
print(f"Total images: {total_images}")
print(f"Total classes: {len(classes)}")