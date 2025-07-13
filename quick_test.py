#!/usr/bin/env python3
"""
Quick test script to verify the application works
"""

import os
import sys
import numpy as np
from PIL import Image
import torch

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_model():
    """Test the CNN model"""
    print("Testing CNN model...")
    try:
        from app import GICNN
        model = GICNN()
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 256, 256)
        
        # Test forward pass
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✓ Model output shape: {output.shape}")
        print(f"✓ Model test passed!")
        return True
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False

def test_ewt():
    """Test EWT preprocessing"""
    print("Testing EWT preprocessing...")
    try:
        from app import apply_ewt_preprocessing
        
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Apply EWT
        processed = apply_ewt_preprocessing(dummy_image)
        
        print(f"✓ EWT output shape: {processed.shape}")
        print(f"✓ EWT test passed!")
        return True
    except Exception as e:
        print(f"✗ EWT test failed: {e}")
        return False

def test_grad_cam():
    """Test Grad-CAM function"""
    print("Testing Grad-CAM...")
    try:
        from app import grad_cam, GICNN
        
        model = GICNN()
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(3, 256, 256)
        
        # Test Grad-CAM
        heatmap = grad_cam(model, dummy_input, 0)
        
        print(f"✓ Grad-CAM output shape: {heatmap.shape}")
        print(f"✓ Grad-CAM test passed!")
        return True
    except Exception as e:
        print(f"✗ Grad-CAM test failed: {e}")
        return False

def test_database():
    """Test database models"""
    print("Testing database models...")
    try:
        from app import app, db, User, Patient
        
        app.config['TESTING'] = True
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        
        with app.app_context():
            db.create_all()
            
            # Test user creation
            user = User(name='Test User', email='test@example.com', 
                       password_hash='test', role='doctor')
            db.session.add(user)
            db.session.commit()
            
            # Test patient creation
            patient = Patient(patient_id='P001', name='Test Patient')
            db.session.add(patient)
            db.session.commit()
            
            print(f"✓ Database test passed!")
            return True
    except Exception as e:
        print(f"✗ Database test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Running Quick Tests...")
    print("=" * 50)
    
    tests = [
        test_model,
        test_ewt,
        test_grad_cam,
        test_database
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! Application should work correctly.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 