import os
import shutil
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image






MODEL_PATH = r'path\to\my\model.pth'  
IMAGES_FOLDER = r'path\to\my\images'  
OUTPUT_FOLDER = r'path\to\my\output'         


EMOTIONS = ['angry', 'happy', 'relaxed', 'sad']





transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_model(model_path, device):
    
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 4)  
    
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    
    model = load_model(MODEL_PATH, device)

    
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    for emotion in EMOTIONS:
        emotion_dir = os.path.join(OUTPUT_FOLDER, emotion)
        if not os.path.exists(emotion_dir):
            os.makedirs(emotion_dir)

    
    for filename in os.listdir(IMAGES_FOLDER):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(IMAGES_FOLDER, filename)
            try:
                
                image = Image.open(image_path).convert('RGB')
                input_tensor = transform(image)
                input_tensor = input_tensor.unsqueeze(0)  

                
                with torch.no_grad():
                    outputs = model(input_tensor.to(device))
                    
                    _, pred_index = torch.max(outputs, 1)
                    pred_index = pred_index.item()

                
                emotion_label = EMOTIONS[pred_index]

                
                dest_path = os.path.join(OUTPUT_FOLDER, emotion_label, filename)
                shutil.copy2(image_path, dest_path)
                print(f"Image '{filename}' classified as '{emotion_label}'.")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print("Classification plus file organization is complete.")

if __name__ == '__main__':
    main()