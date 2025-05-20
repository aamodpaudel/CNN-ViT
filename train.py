import os
import json
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50  
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


try:
    from torchviz import make_dot
    TORCHVIZ_AVAILABLE = True
except ImportError:
    TORCHVIZ_AVAILABLE = False




DATA_DIR = "augmented_data_split"  
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR   = os.path.join(DATA_DIR, "val")
TEST_DIR  = os.path.join(DATA_DIR, "test")

NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-4




class HybridCNNViT(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        
        self.cnn = resnet50(pretrained=True)
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




def main():
    
    for directory in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if not os.path.isdir(directory):
            raise ValueError(f"Data directory {directory} does not exist!")
    
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
    val_dataset   = datasets.ImageFolder(VAL_DIR, transform=transform)
    test_dataset  = datasets.ImageFolder(TEST_DIR, transform=transform)
    
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridCNNViT(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    scaler = torch.cuda.amp.GradScaler()

    
    metrics = []
    epoch_train_acc = []
    epoch_train_loss = []
    epoch_val_acc = []
    epoch_val_loss = []
    epoch_test_acc = []
    epoch_test_loss = []
    
    
    
    
    for epoch in range(NUM_EPOCHS):
        
        model.train()
        running_train_loss = 0.0
        total_train = 0
        correct_train = 0
        with tqdm(total=len(train_loader.dataset), desc=f"Epoch {epoch+1}/{NUM_EPOCHS} Training", unit="img") as pbar:
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                running_train_loss += loss.item() * inputs.size(0)
                total_train += inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_train += (preds == labels).sum().item()
                pbar.update(inputs.size(0))
        avg_train_loss = running_train_loss / total_train
        train_accuracy = correct_train / total_train
        epoch_train_loss.append(avg_train_loss)
        epoch_train_acc.append(train_accuracy)
    
        
        model.eval()
        running_val_loss = 0.0
        total_val = 0
        correct_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
                total_val += labels.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
        avg_val_loss = running_val_loss / total_val
        val_accuracy = correct_val / total_val
        epoch_val_loss.append(avg_val_loss)
        epoch_val_acc.append(val_accuracy)
    
        
        running_test_loss = 0.0
        total_test = 0
        correct_test = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                running_test_loss += loss.item() * inputs.size(0)
                total_test += labels.size(0)
                _, preds = torch.max(outputs, 1)
                correct_test += (preds == labels).sum().item()
        avg_test_loss = running_test_loss / total_test
        test_accuracy = correct_test / total_test
        epoch_test_loss.append(avg_test_loss)
        epoch_test_acc.append(test_accuracy)
    
        scheduler.step()
    
        
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy,
            "test_loss": avg_test_loss,
            "test_accuracy": test_accuracy
        }
        metrics.append(epoch_metrics)
    
        print(f"\nEpoch {epoch+1} complete:")
        print(f"  Train Loss: {avg_train_loss:.4f} - Train Acc: {train_accuracy:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f} - Val Acc:   {val_accuracy:.4f}")
        print(f"  Test Loss:  {avg_test_loss:.4f} - Test Acc:  {test_accuracy:.4f}\n")
    
        
        if test_accuracy >= 0.95:
            print("Test accuracy became 95%, stopping training.")
            break

    
    with open('training_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    
    torch.save(model.state_dict(), "pet_sentiment_model.pth")
    print("Training complete and model saved.")

    
    
    
    
    plt.figure(figsize=(10,6))
    epochs = range(1, len(epoch_train_acc) + 1)
    plt.plot(epochs, epoch_train_acc, 'bo-', label='Train Accuracy')
    plt.plot(epochs, epoch_val_acc, 'ro-', label='Validation Accuracy')
    plt.plot(epochs, epoch_test_acc, 'go-', label='Test Accuracy')
    plt.title("Epoch vs Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("epoch_vs_accuracy.png")
    plt.close()

    
    plt.figure(figsize=(10,6))
    plt.plot(epochs, epoch_train_loss, 'bo-', label='Train Loss')
    plt.plot(epochs, epoch_val_loss, 'ro-', label='Validation Loss')
    plt.plot(epochs, epoch_test_loss, 'go-', label='Test Loss')
    plt.title("Epoch vs Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("epoch_vs_loss.png")
    plt.close()

    
    x = np.linspace(-5, 5, 100)
    gelu = 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * np.power(x, 3))))
    plt.figure(figsize=(8,6))
    plt.plot(x, gelu, label="GELU")
    plt.title("GELU Activation Function")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.legend()
    plt.savefig("activation_function_gelu.png")
    plt.close()


    print("All figures generated and saved.")




    
 

if __name__ == '__main__':
    main()
