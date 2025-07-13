#!/usr/bin/env python3
"""
Simple script to run model training
"""

import subprocess
import sys
import os

def main():
    print("🚀 Starting GI Abnormality Detection Model Training")
    print("=" * 60)
    
    # Check if data directory exists
    if not os.path.exists('data/labeled-images'):
        print("❌ Error: data/labeled-images directory not found!")
        print("Please make sure you have uploaded your dataset to the data/labeled-images folder.")
        return
    
    # Check if we have images
    image_count = 0
    for root, dirs, files in os.walk('data/labeled-images'):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_count += 1
    
    if image_count == 0:
        print("❌ Error: No images found in the dataset!")
        print("Please make sure you have uploaded images to the data/labeled-images folder.")
        return
    
    print(f"✓ Found {image_count} images in the dataset")
    
    # Run training with default parameters
    print("\n📋 Training Parameters:")
    print("- Data directory: data/labeled-images")
    print("- Batch size: 32")
    print("- Epochs: 50")
    print("- Learning rate: 0.001")
    print("- Device: Auto-detect (CUDA if available, else CPU)")
    
    print("\n🔄 Starting training...")
    print("=" * 60)
    
    try:
        # Run the training script
        result = subprocess.run([
            sys.executable, 'train_model.py',
            '--data_dir', 'data/labeled-images',
            '--batch_size', '32',
            '--epochs', '50',
            '--lr', '0.001',
            '--device', 'auto'
        ], check=True)
        
        print("\n" + "=" * 60)
        print("🎉 Training completed successfully!")
        print("\n📁 Output files:")
        print("- models/gi_cnn_model.pth (trained model)")
        print("- models/model_metadata.json (model information)")
        print("- models/training_history.png (training plots)")
        print("- models/confusion_matrix.png (evaluation results)")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with error code: {e.returncode}")
        print("Please check the error messages above and try again.")
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == '__main__':
    main() 