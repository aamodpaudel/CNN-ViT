import os
import random
from glob import glob
from PIL import Image
import numpy as np
import torchvision.transforms as T


input_dir = "classified_images"  
output_dir = "augmented_data"      


for emotion in os.listdir(input_dir):
    os.makedirs(os.path.join(output_dir, emotion), exist_ok=True)

# augmentation transforms applied

augment_transforms = T.Compose([
    T.RandomRotation(degrees=30),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    T.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
])

# augmenting images until there are exactly 7000 images.
target_count = 7000

for emotion in os.listdir(input_dir):
    class_input_dir = os.path.join(input_dir, emotion)
    class_output_dir = os.path.join(output_dir, emotion)
    
    
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(glob(os.path.join(class_input_dir, ext)))
    
    current_count = len(image_files)
    print(f"Emotion: {emotion} - Original count: {current_count}")

    for idx, file in enumerate(image_files):
        try:
            img = Image.open(file).convert("RGB")
        except Exception as e:
            print(f"Skipping {file}: {e}")
            continue
        save_path = os.path.join(class_output_dir, f"{emotion}_orig_{idx}.jpg")
        img.save(save_path)
    
    
    current_count = len(os.listdir(class_output_dir))
    
    
    needed = target_count - current_count
    print(f"For emotion '{emotion}', augmenting {needed} images.")
    
    
    saved_images = glob(os.path.join(class_output_dir, "*.jpg"))
    
    for i in range(needed):
        source_file = random.choice(saved_images)
        try:
            img = Image.open(source_file).convert("RGB")
        except Exception as e:
            print(f"Error opening {source_file}: {e}")
            continue
        
        
        img_aug = augment_transforms(img)
        
        save_path = os.path.join(class_output_dir, f"{emotion}_aug_{i}.jpg")
        img_aug.save(save_path)
    
    final_count = len(os.listdir(class_output_dir))
    print(f"Final count for emotion '{emotion}': {final_count}")

print("Augmentation complete. Each category now has 7000 images in the 'augmented_data' folder.")