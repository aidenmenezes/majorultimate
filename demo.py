#!/usr/bin/env python3
"""
Demo script for GI Abnormality Detection Application
This script demonstrates the key features and creates sample data
"""

import os
import sys
import numpy as np
from PIL import Image
import uuid
from datetime import datetime, timedelta

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, db, User, Patient, EndoscopicImage, Prediction, ExplainabilityResult, Feedback
from werkzeug.security import generate_password_hash

def create_sample_image(filename, size=(256, 256), color=(100, 150, 200)):
    """Create a sample endoscopic image"""
    img = Image.new('RGB', size, color)
    
    # Add some random patterns to simulate endoscopic features
    pixels = np.array(img)
    
    # Add some random spots (simulating abnormalities)
    for _ in range(10):
        x = np.random.randint(0, size[0])
        y = np.random.randint(0, size[1])
        radius = np.random.randint(5, 15)
        color_variation = np.random.randint(-50, 50, 3)
        
        for i in range(max(0, x-radius), min(size[0], x+radius)):
            for j in range(max(0, y-radius), min(size[1], y+radius)):
                if (i-x)**2 + (j-y)**2 <= radius**2:
                    pixels[j, i] = np.clip(pixels[j, i] + color_variation, 0, 255)
    
    img = Image.fromarray(pixels)
    img.save(filename)
    return filename

def create_sample_heatmap(filename, size=(256, 256)):
    """Create a sample Grad-CAM heatmap"""
    # Create a heatmap with random hotspots
    heatmap = np.zeros(size)
    
    # Add random hotspots
    for _ in range(5):
        x = np.random.randint(0, size[0])
        y = np.random.randint(0, size[1])
        intensity = np.random.uniform(0.3, 1.0)
        radius = np.random.randint(20, 60)
        
        for i in range(max(0, x-radius), min(size[0], x+radius)):
            for j in range(max(0, y-radius), min(size[1], y+radius)):
                distance = np.sqrt((i-x)**2 + (j-y)**2)
                if distance <= radius:
                    heatmap[j, i] = max(heatmap[j, i], intensity * (1 - distance/radius))
    
    # Convert to image and save
    heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
    heatmap_img.save(filename)
    return filename

def setup_demo_data():
    """Create demo data for the application"""
    print("🎭 Setting up demo data...")
    
    with app.app_context():
        # Create directories
        os.makedirs('uploads', exist_ok=True)
        os.makedirs('static/heatmaps', exist_ok=True)
        
        # Create sample users
        users_data = [
            {'name': 'Dr. Sarah Johnson', 'email': 'sarah.johnson@hospital.com', 'role': 'doctor'},
            {'name': 'Dr. Michael Chen', 'email': 'michael.chen@hospital.com', 'role': 'doctor'},
            {'name': 'Dr. Emily Rodriguez', 'email': 'emily.rodriguez@hospital.com', 'role': 'doctor'},
            {'name': 'Tech Specialist', 'email': 'tech@hospital.com', 'role': 'technician'},
        ]
        
        users = []
        for user_data in users_data:
            user = User(
                name=user_data['name'],
                email=user_data['email'],
                password_hash=generate_password_hash('password123'),
                role=user_data['role']
            )
            db.session.add(user)
            users.append(user)
        
        # Create sample patients
        patients_data = [
            {'patient_id': 'P001', 'name': 'John Smith', 'age': 45, 'gender': 'Male'},
            {'patient_id': 'P002', 'name': 'Mary Johnson', 'age': 52, 'gender': 'Female'},
            {'patient_id': 'P003', 'name': 'Robert Davis', 'age': 38, 'gender': 'Male'},
            {'patient_id': 'P004', 'name': 'Lisa Wilson', 'age': 61, 'gender': 'Female'},
            {'patient_id': 'P005', 'name': 'David Brown', 'age': 49, 'gender': 'Male'},
        ]
        
        patients = []
        for patient_data in patients_data:
            patient = Patient(**patient_data)
            db.session.add(patient)
            patients.append(patient)
        
        db.session.commit()
        
        # Create sample images and predictions
        abnormality_types = ['Normal', 'Ulcer', 'Polyp', 'Bleeding']
        image_types = ['endoscopic', 'colonoscopy', 'gastroscopy']
        
        for i, patient in enumerate(patients):
            # Create 2-4 images per patient
            num_images = np.random.randint(2, 5)
            
            for j in range(num_images):
                # Create sample image
                image_filename = f"sample_image_{patient.patient_id}_{j+1}.jpg"
                image_path = os.path.join('uploads', image_filename)
                create_sample_image(image_path)
                
                # Create image record
                image = EndoscopicImage(
                    patient_id=patient.id,
                    image_path=image_path,
                    image_type=np.random.choice(image_types),
                    original_filename=image_filename,
                    upload_date=datetime.now() - timedelta(days=np.random.randint(1, 30))
                )
                db.session.add(image)
                db.session.commit()
                
                # Create prediction
                abnormality = np.random.choice(abnormality_types, p=[0.4, 0.2, 0.2, 0.2])
                confidence = np.random.uniform(0.6, 0.95)
                
                prediction = Prediction(
                    image_id=image.id,
                    abnormality_type=abnormality,
                    confidence_score=confidence,
                    is_uncertain=confidence < 0.7,
                    prediction_date=image.upload_date + timedelta(minutes=np.random.randint(5, 30))
                )
                db.session.add(prediction)
                db.session.commit()
                
                # Create heatmap
                heatmap_filename = f"heatmap_{prediction.id}_{uuid.uuid4()}.png"
                heatmap_path = os.path.join('static/heatmaps', heatmap_filename)
                create_sample_heatmap(heatmap_path)
                
                # Create explainability result
                explainability = ExplainabilityResult(
                    prediction_id=prediction.id,
                    grad_cam_path=heatmap_path,
                    heatmap_confidence=confidence
                )
                db.session.add(explainability)
                
                # Create some feedback (for some predictions)
                if np.random.random() < 0.3:  # 30% chance of feedback
                    feedback = Feedback(
                        user_id=np.random.choice(users).id,
                        prediction_id=prediction.id,
                        feedback_text=np.random.choice([
                            "Prediction appears accurate based on visual assessment.",
                            "Slight disagreement with the classification.",
                            "Excellent detection of the abnormality.",
                            "Need to review this case more carefully.",
                            "Agree with the AI assessment."
                        ]),
                        is_correct=np.random.choice([True, False], p=[0.8, 0.2]),
                        manual_label=abnormality if np.random.random() < 0.9 else np.random.choice(abnormality_types),
                        feedback_date=prediction.prediction_date + timedelta(hours=np.random.randint(1, 24))
                    )
                    db.session.add(feedback)
        
        db.session.commit()
        
        print("✅ Demo data created successfully!")
        print(f"   • {len(users)} users created")
        print(f"   • {len(patients)} patients created")
        print(f"   • {len(patients) * 3} sample images created")
        print(f"   • Sample predictions and heatmaps generated")

def show_demo_info():
    """Display demo information"""
    print("🎭 GI Abnormality Detection - Demo Mode")
    print("=" * 60)
    print("📋 Demo Credentials:")
    print("   • Admin: admin@gi-detection.com / admin123")
    print("   • Doctor: sarah.johnson@hospital.com / password123")
    print("   • Doctor: michael.chen@hospital.com / password123")
    print("   • Doctor: emily.rodriguez@hospital.com / password123")
    print("   • Technician: tech@hospital.com / password123")
    print()
    print("🔍 Demo Features:")
    print("   • Sample patients with multiple cases")
    print("   • AI predictions with confidence scores")
    print("   • Grad-CAM heatmaps for explainability")
    print("   • Expert feedback on predictions")
    print("   • Patient history and case management")
    print("   • Admin dashboard with analytics")
    print()
    print("🚀 To start the demo:")
    print("   python run.py")
    print("   Then visit: http://localhost:5000")
    print("=" * 60)

def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == '--setup':
        setup_demo_data()
    else:
        show_demo_info()

if __name__ == '__main__':
    main() 