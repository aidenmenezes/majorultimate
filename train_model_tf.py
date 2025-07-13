#!/usr/bin/env python3
"""
Model Training Script for GI Abnormality Detection using TensorFlow
Trains the CNN model using the uploaded dataset
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import argparse
import json
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input

# Check GPU availability
print("🔧 TensorFlow GPU Setup:")
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
print(f"CUDA Available: {tf.test.is_built_with_cuda()}")

# Enable memory growth to avoid GPU memory issues
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✓ GPU memory growth enabled")
    except RuntimeError as e:
        print(f"⚠️ GPU memory growth error: {e}")

class GIImageDataset:
    """Custom dataset for GI images using TensorFlow"""
    
    def __init__(self, image_paths, labels, apply_ewt=True):
        self.image_paths = image_paths
        self.labels = labels
        self.apply_ewt = apply_ewt
        
    def __len__(self):
        return len(self.image_paths)
    
    def load_and_preprocess_image(self, image_path):
        """Load and preprocess a single image"""
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            image = image.resize((256, 256))
            image_array = np.array(image)
            
            # Apply EWT preprocessing if enabled
            if self.apply_ewt:
                image_array = apply_ewt_preprocessing(image_array)
            
            # Normalize to [0, 1]
            image_array = image_array.astype(np.float32) / 255.0
            
            return image_array
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a placeholder image
            return np.zeros((256, 256, 3), dtype=np.float32)
    
    def get_data(self):
        """Get all images and labels as numpy arrays"""
        images = []
        for image_path in self.image_paths:
            image = self.load_and_preprocess_image(image_path)
            images.append(image)
        
        return np.array(images), np.array(self.labels)

def apply_ewt_preprocessing(image):
    """Apply Empirical Wavelet Transform preprocessing"""
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

def load_dataset(data_dir):
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

def create_model(num_classes, input_shape=(256, 256, 3)):
    """Create a CNN model for GI abnormality detection"""
    print("🏗️ Creating model architecture...")
    
    # Use ResNet50V2 as base model with transfer learning
    base_model = ResNet50V2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Create the full model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✓ Model created with {num_classes} output classes")
    return model

def train_model(model, train_data, val_data, num_epochs=30, batch_size=32):
    """Train the model"""
    print("🚀 Starting model training...")
    
    # Unpack data
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    # Data augmentation for training
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        fill_mode='nearest'
    )
    
    # Callbacks
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        ),
        callbacks.ModelCheckpoint(
            'models/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train the model
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(X_train) // batch_size,
        epochs=num_epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks_list,
        verbose=1
    )
    
    return model, history

def evaluate_model(model, test_data, class_mapping):
    """Evaluate the trained model"""
    print("📊 Evaluating model...")
    
    X_test, y_test = test_data
    
    # Make predictions
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Calculate accuracy
    test_accuracy = np.mean(predicted_classes == y_test)
    print(f"✓ Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Generate classification report
    class_names = list(class_mapping.keys())
    report = classification_report(y_test, predicted_classes, target_names=class_names, output_dict=True)
    
    # Print detailed report
    print("\n📋 Classification Report:")
    print(classification_report(y_test, predicted_classes, target_names=class_names))
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, predicted_classes)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return test_accuracy, report

def plot_training_history(history):
    """Plot training history"""
    print("📈 Plotting training history...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
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
    
    # Save the model
    model.save('models/gi_cnn_model_tf.h5')
    
    # Save metadata
    metadata = {
        'model_architecture': 'ResNet50V2_Transfer_Learning',
        'framework': 'TensorFlow',
        'num_classes': len(class_mapping),
        'class_mapping': class_mapping,
        'input_size': (256, 256, 3),
        'training_date': datetime.now().isoformat(),
        'test_accuracy': test_accuracy,
        'classification_report': report,
        'final_train_accuracy': history.history['accuracy'][-1],
        'final_val_accuracy': history.history['val_accuracy'][-1],
        'final_train_loss': history.history['loss'][-1],
        'final_val_loss': history.history['val_loss'][-1]
    }
    
    with open('models/model_metadata_tf.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("✓ Model and metadata saved")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train GI Abnormality Detection Model with TensorFlow')
    parser.add_argument('--data_dir', type=str, default='data/labeled-images',
                       help='Path to the data directory')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    print("🚀 Starting GI Abnormality Detection Model Training with TensorFlow")
    print("=" * 70)
    
    # Load dataset
    image_paths, labels, class_mapping = load_dataset(args.data_dir)
    
    if len(image_paths) == 0:
        print("❌ No images found in the dataset!")
        return
    
    # Create dataset objects
    dataset = GIImageDataset(image_paths, labels, apply_ewt=True)
    X, y = dataset.get_data()
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1, random_state=42, stratify=y_temp
    )
    
    print(f"✓ Train samples: {len(X_train)}")
    print(f"✓ Validation samples: {len(X_val)}")
    print(f"✓ Test samples: {len(X_test)}")
    
    # Create model
    model = create_model(len(class_mapping))
    
    # Train model
    model, history = train_model(
        model, 
        (X_train, y_train), 
        (X_val, y_val),
        num_epochs=args.epochs, 
        batch_size=args.batch_size
    )
    
    # Evaluate model
    test_accuracy, report = evaluate_model(model, (X_test, y_test), class_mapping)
    
    # Plot training history
    plot_training_history(history)
    
    # Save model and metadata
    save_model_info(model, class_mapping, history, test_accuracy, report)
    
    print("\n🎉 Training completed successfully!")
    print(f"📊 Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print("📁 Model saved to: models/gi_cnn_model_tf.h5")
    print("📊 Training plots saved to: models/")
    print("📋 Model metadata saved to: models/model_metadata_tf.json")

if __name__ == '__main__':
    main() 