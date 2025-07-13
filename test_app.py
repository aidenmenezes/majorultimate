#!/usr/bin/env python3
"""
Test script for GI Abnormality Detection Application
"""

import os
import sys
import unittest
from io import BytesIO
from PIL import Image
import numpy as np

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, db, User, Patient, EndoscopicImage, Prediction, GICNN, apply_ewt_preprocessing

class TestGIAbnormalityDetection(unittest.TestCase):
    """Test cases for the GI Abnormality Detection application"""
    
    def setUp(self):
        """Set up test environment"""
        app.config['TESTING'] = True
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        self.app = app.test_client()
        self.app_context = app.app_context()
        self.app_context.push()
        db.create_all()
    
    def tearDown(self):
        """Clean up after tests"""
        db.session.remove()
        db.drop_all()
        self.app_context.pop()
    
    def test_home_page(self):
        """Test if home page loads correctly"""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'GI Abnormality Detection', response.data)
    
    def test_login_page(self):
        """Test if login page loads correctly"""
        response = self.app.get('/login')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Sign In', response.data)
    
    def test_database_models(self):
        """Test database model creation"""
        # Test User model
        user = User(name='Test Doctor', email='test@example.com', 
                   password_hash='hashed_password', role='doctor')
        db.session.add(user)
        db.session.commit()
        self.assertIsNotNone(user.id)
        
        # Test Patient model
        patient = Patient(patient_id='P001', name='John Doe', age=45, gender='Male')
        db.session.add(patient)
        db.session.commit()
        self.assertIsNotNone(patient.id)
        
        # Test EndoscopicImage model
        image = EndoscopicImage(patient_id=patient.id, image_path='test/path.jpg', 
                               image_type='endoscopic', original_filename='test.jpg')
        db.session.add(image)
        db.session.commit()
        self.assertIsNotNone(image.id)
        
        # Test Prediction model
        prediction = Prediction(image_id=image.id, abnormality_type='Normal', 
                              confidence_score=0.85)
        db.session.add(prediction)
        db.session.commit()
        self.assertIsNotNone(prediction.id)
    
    def test_cnn_model(self):
        """Test CNN model initialization and forward pass"""
        model = GICNN()
        model.eval()
        
        # Create dummy input tensor
        dummy_input = np.random.randn(1, 3, 256, 256).astype(np.float32)
        import torch
        input_tensor = torch.FloatTensor(dummy_input)
        
        # Test forward pass
        with torch.no_grad():
            output = model(input_tensor)
        
        # Check output shape
        self.assertEqual(output.shape, (1, 4))  # 4 classes
    
    def test_ewt_preprocessing(self):
        """Test EWT preprocessing function"""
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Apply EWT preprocessing
        processed_image = apply_ewt_preprocessing(dummy_image)
        
        # Check output shape and type
        self.assertEqual(processed_image.shape, (256, 256, 3))
        self.assertEqual(processed_image.dtype, np.uint8)
    
    def test_image_upload_validation(self):
        """Test image upload validation"""
        # Create a dummy image
        img = Image.new('RGB', (256, 256), color='red')
        img_io = BytesIO()
        img.save(img_io, 'JPEG')
        img_io.seek(0)
        
        # Test with valid image
        response = self.app.post('/upload', 
                               data={'image': (img_io, 'test.jpg'),
                                    'patient_id': 'P001',
                                    'patient_name': 'Test Patient'},
                               content_type='multipart/form-data',
                               follow_redirects=True)
        
        # Should redirect to login since not authenticated
        self.assertIn(response.status_code, [302, 200])
    
    def test_admin_user_creation(self):
        """Test admin user creation"""
        # Check if admin user exists after app initialization
        admin = User.query.filter_by(email='admin@gi-detection.com').first()
        if admin:
            self.assertEqual(admin.role, 'admin')
            self.assertEqual(admin.name, 'Administrator')

def run_tests():
    """Run all tests"""
    print("Running GI Abnormality Detection Tests...")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGIAbnormalityDetection)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1) 