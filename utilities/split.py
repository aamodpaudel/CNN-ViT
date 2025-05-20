import os
import shutil
import random


input_dir = "augmented_data"

output_dir = "augmented_data_split"



train_ratio = 75 / 105    
test_ratio  = 15 / 105    
val_ratio   = 15 / 105    


emotions = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]


for split in ['train', 'test', 'val']:
    for emotion in emotions:
        os.makedirs(os.path.join(output_dir, split, emotion), exist_ok=True)


for emotion in emotions:
    emotion_dir = os.path.join(input_dir, emotion)
    image_files = [os.path.join(emotion_dir, f) for f in os.listdir(emotion_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    random.shuffle(image_files)
    
    total = len(image_files)
    train_count = int(total * train_ratio)
    test_count = int(total * test_ratio)
    val_count = total - train_count - test_count  

    train_files = image_files[:train_count]
    test_files  = image_files[train_count:train_count+test_count]
    val_files   = image_files[train_count+test_count:]
    
    
    for f in train_files:
        shutil.copy(f, os.path.join(output_dir, "train", emotion))
    for f in test_files:
        shutil.copy(f, os.path.join(output_dir, "test", emotion))
    for f in val_files:
        shutil.copy(f, os.path.join(output_dir, "val", emotion))
    
    print(f"Emotion '{emotion}' split: Total={total}, Train={len(train_files)}, Test={len(test_files)}, Val={len(val_files)}")

print("Data splitting is complete. Folder 'augmented_data_split' is seen for train, test and validation sets.")