#!/usr/bin/env python3
"""
Deployment script for GI Detection System
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'flask', 'tensorflow', 'numpy', 'opencv-python', 
        'pillow', 'matplotlib', 'pandas', 'reportlab'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies installed")
    return True

def check_model_files():
    """Check if trained model files exist"""
    print("🔍 Checking model files...")
    
    model_files = [
        'models/gi_cnn_model_tf.h5',
        'models/model_metadata_tf.json'
    ]
    
    missing_files = []
    for file_path in model_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing model files: {', '.join(missing_files)}")
        return False
    
    print("✅ Model files found")
    return True

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = [
        'uploads',
        'static/heatmaps',
        'logs',
        'instance'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created {directory}")

def setup_environment():
    """Setup environment variables"""
    print("⚙️ Setting up environment...")
    
    env_file = '.env'
    if not os.path.exists(env_file):
        with open(env_file, 'w') as f:
            f.write("SECRET_KEY=your-secret-key-change-this-in-production\n")
            f.write("FLASK_ENV=production\n")
        print("✅ Created .env file")
    else:
        print("✅ .env file already exists")

def run_tests():
    """Run basic tests"""
    print("🧪 Running tests...")
    
    try:
        # Test database creation
        from app import app, db
        with app.app_context():
            db.create_all()
        print("✅ Database test passed")
        
        # Test model loading
        import tensorflow as tf
        model = tf.keras.models.load_model('models/gi_cnn_model_tf.h5')
        print("✅ Model loading test passed")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return True

def main():
    """Main deployment function"""
    print("🚀 Starting deployment...")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check model files
    if not check_model_files():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Setup environment
    setup_environment()
    
    # Run tests
    if not run_tests():
        sys.exit(1)
    
    print("\n🎉 Deployment completed successfully!")
    print("\nNext steps:")
    print("1. Update .env file with your secret key")
    print("2. Run: python app.py")
    print("3. Access the application at: http://localhost:5000")
    print("4. Login with: admin@gi-detection.com / admin123")

if __name__ == "__main__":
    main() 