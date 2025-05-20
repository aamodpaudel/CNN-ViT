import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from scipy.spatial import distance



class HybridCNNViT(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        
        self.cnn = resnet50(weights=None)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])
        
        self.feature_projection = nn.Conv2d(
            in_channels=2048,
            out_channels=768,
            kernel_size=1
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, 768))
        
        self.positional_embedding = nn.Parameter(torch.randn(1, 50, 768))
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=768,
                nhead=8,
                dim_feedforward=3072,
                activation="gelu"
            ),
            num_layers=4
        )
        
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        features = self.cnn(x)
        features = self.feature_projection(features)
        
        features = features.flatten(2).permute(0, 2, 1)
        
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        features = torch.cat((cls_tokens, features), dim=1)
        features += self.positional_embedding
        
        features = features.permute(1, 0, 2)
        features = self.transformer(features)
        
        cls_output = features[0]
        return self.classifier(cls_output)






def analyze_facial_features(image):
   
    image_np = np.array(image)
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    
    try:
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    except:
        
        st.warning("Using comparable direct image analysis.")
        return analyze_image_directly(image_rgb)
    
    
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        st.warning("Using approximate values for image analysis.")
        return analyze_image_directly(image_rgb)
    
   
    x, y, w, h = faces[0]
    face_region = gray[y:y+h, x:x+w]
    
    
    
    face_height, face_width = face_region.shape
    
   
    eye_region = face_region[0:int(face_height/3), :]
    
   
    _, eye_binary = cv2.threshold(eye_region, 70, 255, cv2.THRESH_BINARY)
    eye_white_ratio = np.sum(eye_binary == 255) / (eye_region.size + 1e-10)
    
    
    mouth_region = face_region[int(face_height/2):, :]
    
    
    _, mouth_binary = cv2.threshold(mouth_region, 50, 255, cv2.THRESH_BINARY_INV)
    mouth_dark_ratio = np.sum(mouth_binary == 255) / (mouth_region.size + 1e-10)
    
   
    ear = eye_white_ratio * 2  
    mar = mouth_dark_ratio * 3  
    
    
    ear = max(0, min(1, ear))
    mar = max(0, min(1, mar))
    
    return ear, mar

def analyze_image_directly(image):
    
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
   
    height, width = gray.shape
    
    
    eye_region = gray[0:int(height/3), :]
    
   
    mouth_region = gray[int(height/2):, :]
    
   
    eye_std = np.std(eye_region)
    mouth_std = np.std(mouth_region)
    
    
    eye_edges = cv2.Canny(eye_region, 100, 200)
    mouth_edges = cv2.Canny(mouth_region, 100, 200)
    
    eye_edge_density = np.sum(eye_edges == 255) / (eye_region.size + 1e-10)
    mouth_edge_density = np.sum(mouth_edges == 255) / (mouth_region.size + 1e-10)
    
    
    ear = (eye_std / 128) * 0.5 + eye_edge_density * 2
    mar = (mouth_std / 128) * 0.5 + mouth_edge_density * 2
    
   
    ear = max(0, min(1, ear))
    mar = max(0, min(1, mar))
    
    return ear, mar



def process_image(image):
    
    ear, mar = analyze_facial_features(image)
    
    
    st.write(f"Extracted Eye Feature (EAR): {ear:.3f}")
    st.write(f"Extracted Mouth Feature (MAR): {mar:.3f}")
    
    return ear, mar



def plot_emotion(x, y, emotion_name, specific_emotion):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    
    circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='-')
    ax.add_patch(circle)
    
   
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel('Valence', fontsize=12, fontweight='bold')
    ax.set_ylabel('Arousal', fontsize=12, fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
    
   
    ax.text(0.7, 0.7, 'I (Happy)', fontsize=12, ha='center', va='center', fontweight='bold')
    ax.text(-0.7, 0.7, 'II (Angry)', fontsize=12, ha='center', va='center', fontweight='bold')
    ax.text(-0.7, -0.7, 'III (Sad)', fontsize=12, ha='center', va='center', fontweight='bold')
    ax.text(0.7, -0.7, 'IV (Relaxed)', fontsize=12, ha='center', va='center', fontweight='bold')
    
    
    for angle in range(0, 360, 30):
        rad = math.radians(angle)
        x_line = math.cos(rad)
        y_line = math.sin(rad)
        ax.plot([0, x_line], [0, y_line], 'k--', alpha=0.3)
        
        
        if angle % 30 == 0:
            label_x = 1.1 * math.cos(rad)
            label_y = 1.1 * math.sin(rad)
            ax.text(label_x, label_y, f"{angle}°", fontsize=8, ha='center', va='center')
    
    
    emotion_labels = [
        
        (15, "Pleased"),
        (45, "Happy"),
        (75, "Excited"),
        
        (105, "Annoyed"),
        (135, "Angry"),
        (165, "Enraged"),
        
        (195, "Bored"),
        (225, "Sad"),
        (255, "Depressed"),
        
        (285, "Calm"),
        (315, "Relaxed"),
        (345, "Content")
    ]
    
    for angle_deg, label in emotion_labels:
        angle_rad = math.radians(angle_deg)
        label_x = 0.85 * math.cos(angle_rad)
        label_y = 0.85 * math.sin(angle_rad)
        ax.text(label_x, label_y, label, fontsize=9, ha='center', va='center')
    
    
    ax.scatter(x, y, s=150, color='red', zorder=5)
    
    
    ax.plot([0, x], [0, y], color='red', linestyle='-', alpha=0.7)
    
    
    angle = math.degrees(math.atan2(y, x))
    if angle < 0:
        angle += 360
    
    
    ax.set_title(f'Emotion: {emotion_name} ({specific_emotion})\nCoordinates: ({x:.2f}, {y:.2f})\nAngle: {angle:.1f}°', fontsize=14)
    
    
    ax.grid(True, alpha=0.3)
    
    return fig



def main():
    st.set_page_config(page_title="Pet Sentiment Analyzer", layout="wide")
    st.title("Pet Sentiment Analyzer")
    st.write("Upload a pet face image to analyze its emotion on the 2D valence-arousal space.")
    
    st.sidebar.title("About")
    st.sidebar.info(
        """
        This app analyzes pet facial emotions using a pre-trained deep learning model.
        It plots the results on a 2D valence-arousal circumplex with 12 emotion subdivisions.
        """
    )
    
    st.sidebar.title("Emotion Classification")
    st.sidebar.markdown(
        """
        ### Emotion Quadrants:
        - **Quadrant I (0–90°)**: Pleased, Happy, Excited
        - **Quadrant II (90–180°)**: Annoyed, Angry, Enraged
        - **Quadrant III (180–270°)**: Bored, Sad, Depressed
        - **Quadrant IV (270–360°)**: Calm, Relaxed, Content
        """
    )
    
    
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor = transform(image).unsqueeze(0)
        
       
        try:
            model = HybridCNNViT(num_classes=4)
            model.load_state_dict(torch.load(r"C:\Users\aamod\Downloads\Qualia\pet_sentiment_model.pth", map_location=torch.device('cpu')))
            model.eval()
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return
        
        
        with torch.no_grad():
            output = model(input_tensor)
            _, prediction = torch.max(output, 1)
            emotion_class = prediction.item()
        
        
        emotion_mapping = {
            0: ("Angry", 2),  
            1: ("Happy", 1),  
            2: ("Relaxed", 4),  
            3: ("Sad", 3)  
        }
        
        emotion_name, quadrant = emotion_mapping[emotion_class]
        
        
        ear, mar = process_image(image)
        
        
        magnitude = math.sqrt(ear**2 + mar**2)
        if magnitude == 0:  
            x_normalized = 0.7071  
            y_normalized = 0.7071
        else:
            x_normalized = ear / magnitude
            y_normalized = mar / magnitude
        
        
        if quadrant == 1:  # Happy (Q1)
            x, y = abs(x_normalized), abs(y_normalized)
        elif quadrant == 2:  # Angry (Q2)
            x, y = -abs(x_normalized), abs(y_normalized)
        elif quadrant == 3:  # Sad (Q3)
            x, y = -abs(x_normalized), -abs(y_normalized)
        else:  # Relaxed (Q4)
            x, y = abs(x_normalized), -abs(y_normalized)
        
        
        angle = math.degrees(math.atan2(y, x))
        if angle < 0:
            angle += 360
        
        
        specific_emotions = {
            
            (0, 30): "Pleased",
            (30, 60): "Happy",
            (60, 90): "Excited",
           
            (90, 120): "Annoyed",
            (120, 150): "Angry",
            (150, 180): "Enraged",
           
            (180, 210): "Bored",
            (210, 240): "Sad",
            (240, 270): "Depressed",
            
            (270, 300): "Calm",
            (300, 330): "Relaxed",
            (330, 360): "Content",
        }
        
        
        specific_emotion = None
        for (lower, upper), emotion in specific_emotions.items():
            if lower <= angle < upper:
                specific_emotion = emotion
                break
                
        
        if angle == 0 or angle == 360:
            specific_emotion = "Pleased"
        
     
        st.write(f"### Results:")
        st.write(f"Detected Emotion: **{emotion_name}**")
        st.write(f"Specific Emotion: **{specific_emotion}**")
        st.write(f"Quadrant: **{quadrant}**")
        st.write(f"Coordinates: **(x={x:.2f}, y={y:.2f})**")
        st.write(f"Angle: **{angle:.1f}°**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Facial Metrics:")
            st.write(f"Eye Aspect Ratio (EAR): **{ear:.3f}**")
            st.write(f"Mouth Aspect Ratio (MAR): **{mar:.3f}**")
            st.write(f"Normalized Values for Unit Circle: **({x:.3f}, {y:.3f})**")
        
        
        fig = plot_emotion(x, y, emotion_name, specific_emotion)
        st.pyplot(fig)


if __name__ == "__main__":
    main()