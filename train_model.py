#!/usr/bin/env python3
"""
Model Training Script for GI Abnormality Detection
Trains the CNN model using the uploaded dataset
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from datetime import datetime
import argparse
import json

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the model class directly to avoid circular imports
import torch
import torch.nn as nn
import torch.nn.functional as F

class GICNN(nn.Module):
    def __init__(self, num_classes=4):
        super(GICNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        # Calculate the correct size after 3 pooling layers: 256 -> 128 -> 64 -> 32
        self.fc1 = nn.Linear(128 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 32 * 32)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# EWT preprocessing function
def apply_ewt_preprocessing(image):
    """Apply Empirical Wavelet Transform preprocessing"""
    import cv2
    # Convert to grayscale for EWT
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(blurred)
    
    # Convert back to RGB
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    
    return enhanced_rgb

class GIImageDataset(Dataset):
    """Custom dataset for GI images"""
    
    def __init__(self, image_paths, labels, transform=None, apply_ewt=True):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.apply_ewt = apply_ewt
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a placeholder image
            image = Image.new('RGB', (256, 256), color='gray')
        
        # Apply EWT preprocessing if enabled
        if self.apply_ewt:
            image_array = np.array(image)
            processed_array = apply_ewt_preprocessing(image_array)
            image = Image.fromarray(processed_array)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label

def load_dataset(data_dir, csv_file=None):
    """Load dataset from the data directory"""
    print("📁 Loading dataset...")
    
    image_paths = []
    labels = []
    class_mapping = {}
    class_counter = 0
    
    # Define the classes based on the actual dataset structure
    # Separate normal anatomical landmarks from pathological findings
    target_classes = {
        # Normal Anatomical Landmarks (Healthy structures)
        'z-line': 'Normal_Z_Line',
        'retroflex-stomach': 'Normal_Retroflex_Stomach',
        'pylorus': 'Normal_Pylorus',
        
        # Pathological Findings (Abnormal conditions)
        'esophagitis-a': 'Abnormal_Esophagitis_A',
        'esophagitis-b-d': 'Abnormal_Esophagitis_BD',
        'barretts': 'Abnormal_Barretts',
        'barretts-short-segment': 'Abnormal_Barretts_Short',
        'ulcerative-colitis-grade-0-1': 'Abnormal_UC_Grade_0_1',
        'ulcerative-colitis-grade-1': 'Abnormal_UC_Grade_1',
        'ulcerative-colitis-grade-1-2': 'Abnormal_UC_Grade_1_2',
        'ulcerative-colitis-grade-2': 'Abnormal_UC_Grade_2',
        'ulcerative-colitis-grade-2-3': 'Abnormal_UC_Grade_2_3',
        'ulcerative-colitis-grade-3': 'Abnormal_UC_Grade_3',
        'polyps': 'Abnormal_Polyps',
        'hemorrhoids': 'Abnormal_Hemorrhoids'
    }
    
    # Walk through the data directory
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_path = os.path.join(root, file)
                
                # Extract class from directory structure
                path_parts = root.split(os.sep)
                for part in path_parts:
                    part_lower = part.lower()
                    for key, value in target_classes.items():
                        if key in part_lower:
                            if value not in class_mapping:
                                class_mapping[value] = class_counter
                                class_counter += 1
                            
                            image_paths.append(image_path)
                            labels.append(class_mapping[value])
                            break
                    else:
                        continue
                    break
    
    print(f"✓ Loaded {len(image_paths)} images")
    print(f"✓ Found {len(class_mapping)} classes: {list(class_mapping.keys())}")
    
    return image_paths, labels, class_mapping

def create_data_loaders(image_paths, labels, class_mapping, batch_size=32, test_size=0.2, val_size=0.1):
    """Create train, validation, and test data loaders"""
    print("🔄 Creating data loaders...")
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        image_paths, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
    )
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = GIImageDataset(X_train, y_train, transform=train_transform, apply_ewt=True)
    val_dataset = GIImageDataset(X_val, y_val, transform=val_transform, apply_ewt=True)
    test_dataset = GIImageDataset(X_test, y_test, transform=val_transform, apply_ewt=True)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")
    print(f"✓ Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, class_mapping

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, device='cuda'):
    """Train the model"""
    print("🚀 Starting model training...")
    
    # Move model to device
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training history
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] '
                      f'Loss: {loss.item():.4f}')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate metrics
        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        # Store history
        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_loss_avg)
        
        # Save best model
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_model_state = model.state_dict().copy()
        
        print(f'Epoch [{epoch+1}/{num_epochs}] - '
              f'Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%')
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }

def evaluate_model(model, test_loader, class_mapping, device='cuda'):
    """Evaluate the trained model"""
    print("📊 Evaluating model...")
    
    model.eval()
    all_predictions = []
    all_labels = []
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    test_accuracy = 100 * test_correct / test_total
    print(f"✓ Test Accuracy: {test_accuracy:.2f}%")
    
    # Generate classification report
    class_names = list(class_mapping.keys())
    report = classification_report(all_labels, all_predictions, target_names=class_names, output_dict=True)
    
    # Print detailed report
    print("\n📋 Classification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return test_accuracy, report

def plot_training_history(history):
    """Plot training history"""
    print("📈 Plotting training history...")
    
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['val_losses'], label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(history['train_accuracies'], label='Train Accuracy')
    ax2.plot(history['val_accuracies'], label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Training history plots saved")

def save_model_info(model, class_mapping, history, test_accuracy, report):
    """Save model information and metadata"""
    print("💾 Saving model information...")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), 'models/gi_cnn_model.pth')
    
    # Save metadata
    metadata = {
        'model_architecture': 'GICNN',
        'num_classes': len(class_mapping),
        'class_mapping': class_mapping,
        'input_size': (256, 256),
        'training_date': datetime.now().isoformat(),
        'test_accuracy': test_accuracy,
        'classification_report': report,
        'final_train_loss': history['train_losses'][-1],
        'final_val_loss': history['val_losses'][-1],
        'final_train_acc': history['train_accuracies'][-1],
        'final_val_acc': history['val_accuracies'][-1]
    }
    
    with open('models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("✓ Model and metadata saved")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train GI Abnormality Detection Model')
    parser.add_argument('--data_dir', type=str, default='data/labeled-images',
                       help='Path to the data directory')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"🔧 Using device: {device}")
    
    # Load dataset
    image_paths, labels, class_mapping = load_dataset(args.data_dir)
    
    if len(image_paths) == 0:
        print("❌ No images found in the dataset!")
        return
    
    # Create data loaders
    train_loader, val_loader, test_loader, class_mapping = create_data_loaders(
        image_paths, labels, class_mapping, batch_size=args.batch_size
    )
    
    # Initialize model with correct number of classes
    num_classes = len(class_mapping)
    model = GICNN(num_classes=num_classes)
    print(f"✓ Model initialized with {num_classes} classes: {list(class_mapping.keys())}")
    
    # Train model
    model, history = train_model(
        model, train_loader, val_loader, 
        num_epochs=args.epochs, learning_rate=args.lr, device=device
    )
    
    # Evaluate model
    test_accuracy, report = evaluate_model(model, test_loader, class_mapping, device)
    
    # Plot training history
    plot_training_history(history)
    
    # Save model and metadata
    save_model_info(model, class_mapping, history, test_accuracy, report)
    
    print("\n🎉 Training completed successfully!")
    print(f"📊 Final Test Accuracy: {test_accuracy:.2f}%")
    print("📁 Model saved to: models/gi_cnn_model.pth")
    print("📊 Training plots saved to: models/")
    print("📋 Model metadata saved to: models/model_metadata.json")

if __name__ == '__main__':
    main() 