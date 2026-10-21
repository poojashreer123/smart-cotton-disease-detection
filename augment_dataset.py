import os
import cv2
import numpy as np
import albumentations as A
from glob import glob

# --- CONFIGURATION ---
INPUT_TRAIN_DIR = "dataset/Original Dataset"           # Where your current raw images are organized by class folders
OUTPUT_AUG_DIR = "dataset/train_augmented"    # Where the expanded 16k+ images will be saved
TARGET_IMAGES_PER_CLASS = 2250              # E.g., 8 classes * 2250 = 18,000 total images

# Robust augmentation pipeline for plant leaves (simulates field lighting, angles, and camera variations)
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, p=0.6, border_mode=cv2.BORDER_CONSTANT),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2)
])

os.makedirs(OUTPUT_AUG_DIR, exist_ok=True)

# Get all class subfolders
class_folders = [f for f in os.listdir(INPUT_TRAIN_DIR) if os.path.isdir(os.path.join(INPUT_TRAIN_DIR, f))]

if not class_folders:
    print(f"❌ Error: No class subfolders found inside '{INPUT_TRAIN_DIR}'. Please check your path!")
else:
    print(f"🌿 Found {len(class_folders)} classes: {class_folders}\nStarting bulk augmentation...")

for class_name in class_folders:
    input_class_path = os.path.join(INPUT_TRAIN_DIR, class_name)
    output_class_path = os.path.join(OUTPUT_AUG_DIR, class_name)
    os.makedirs(output_class_path, exist_ok=True)
    
    # Load all original images in this class
    original_images = glob(os.path.join(input_class_path, "*.jpg")) + \
                      glob(os.path.join(input_class_path, "*.png")) + \
                      glob(os.path.join(input_class_path, "*.jpeg"))
                      
    print(f"-> Processing class '{class_name}': Found {len(original_images)} original images.")
    
    if len(original_images) == 0:
        continue

    # Step A: Copy original images into output folder first
    for idx, img_path in enumerate(original_images):
        img = cv2.imread(img_path)
        if img is not None:
            cv2.imwrite(os.path.join(output_class_path, f"orig_{idx}.jpg"), img)
            
    # Step B: Loop and generate augmented images until the target count is reached
    current_count = len(os.listdir(output_class_path))
    loop_idx = 0
    
    while current_count < TARGET_IMAGES_PER_CLASS:
        for img_path in original_images:
            if current_count >= TARGET_IMAGES_PER_CLASS:
                break
                
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Convert BGR to RGB for Albumentations
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Apply augmentations
            augmented = transform(image=img)['image']
            
            # Convert back to BGR for OpenCV saving
            augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
            
            # Save the new file
            save_path = os.path.join(output_class_path, f"aug_{loop_idx}.jpg")
            cv2.imwrite(save_path, augmented)
            
            loop_idx += 1
            current_count = len(os.listdir(output_class_path))
            
    print(f"   Successfully scaled '{class_name}' to {current_count} images.")

print("\n✨ Bulk Augmentation Complete! Your 16,000+ image dataset is ready in 'dataset/train_augmented'.")