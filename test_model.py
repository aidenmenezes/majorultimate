#!/usr/bin/env python3
"""
Test script for GI Detection Model
"""

import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import json

def load_model():
    """Load the trained model"""
    try:
        model = tf.keras.models.load_model('models/gi_cnn_model_tf.h5')
        print("✅ Model loaded successfully")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def load_metadata():
    """Load model metadata"""
    try:
        with open('models/model_metadata_tf.json', 'r') as f:
            metadata = json.load(f)
        print("✅ Metadata loaded successfully")
        return metadata
    except Exception as e:
        print(f"❌ Error loading metadata: {e}")
        return None

def preprocess_image(image_path):
    """Preprocess image for model input"""
    try:
        # Load and resize image
        image = Image.open(image_path)
        image = image.resize((256, 256))
        image_array = np.array(image)
        
        # Convert to float and normalize
        image_array = image_array.astype(np.float32) / 255.0
        
        # Add batch dimension
        image_tensor = np.expand_dims(image_array, axis=0)
        
        print("✅ Image preprocessed successfully")
        return image_tensor
    except Exception as e:
        print(f"❌ Error preprocessing image: {e}")
        return None

def test_prediction(model, image_tensor, class_mapping):
    """Test model prediction"""
    try:
        # Make prediction
        predictions = model.predict(image_tensor, verbose=0)
        confidence_scores = predictions[0]
        
        # Get predicted class
        predicted_class_idx = np.argmax(confidence_scores)
        confidence = confidence_scores[predicted_class_idx]
        
        # Get class name
        class_names = list(class_mapping.keys())
        predicted_class = class_names[predicted_class_idx]
        
        print(f"✅ Prediction successful")
        print(f"   Predicted class: {predicted_class}")
        print(f"   Confidence: {confidence:.4f}")
        
        # Show top 3 predictions
        top_indices = np.argsort(confidence_scores)[-3:][::-1]
        print("\nTop 3 predictions:")
        for i, idx in enumerate(top_indices):
            print(f"   {i+1}. {class_names[idx]}: {confidence_scores[idx]:.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Error making prediction: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Testing GI Detection Model")
    print("=" * 40)
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Load metadata
    metadata = load_metadata()
    if metadata is None:
        return
    
    class_mapping = metadata.get('class_mapping', {})
    print(f"📊 Model info:")
    print(f"   Classes: {len(class_mapping)}")
    print(f"   Test accuracy: {metadata.get('test_accuracy', 'N/A')}")
    print(f"   Framework: {metadata.get('framework', 'N/A')}")
    
    # Test with a sample image if available
    sample_images = [
        'data/labeled-images/lower-gi-tract/anatomical-landmarks/sample.jpg',
        'data/labeled-images/upper-gi-tract/anatomical-landmarks/sample.jpg',
        'uploads/sample.jpg'
    ]
    
    test_image = None
    for img_path in sample_images:
        if os.path.exists(img_path):
            test_image = img_path
            break
    
    if test_image:
        print(f"\n🖼️ Testing with image: {test_image}")
        image_tensor = preprocess_image(test_image)
        if image_tensor is not None:
            test_prediction(model, image_tensor, class_mapping)
    else:
        print("\n⚠️ No test image found, creating dummy input")
        # Create dummy image for testing
        dummy_image = np.random.rand(1, 256, 256, 3).astype(np.float32)
        test_prediction(model, dummy_image, class_mapping)
    
    print("\n🎉 Model test completed!")

if __name__ == "__main__":
    main() 