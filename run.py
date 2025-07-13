#!/usr/bin/env python3
"""
Startup script for GI Abnormality Detection Application
"""

import os
import sys
import argparse
from app import app, db, User
from werkzeug.security import generate_password_hash

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = ['uploads', 'static/heatmaps', 'logs', 'models']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def setup_database():
    """Initialize database and create admin user"""
    with app.app_context():
        # Create all tables
        db.create_all()
        print("✓ Database tables created")
        
        # Create admin user if not exists
        admin = User.query.filter_by(email='admin@gi-detection.com').first()
        if not admin:
            admin = User(
                name='Administrator',
                email='admin@gi-detection.com',
                password_hash=generate_password_hash('admin123'),
                role='admin'
            )
            db.session.add(admin)
            db.session.commit()
            print("✓ Admin user created")
        else:
            print("✓ Admin user already exists")

def run_development():
    """Run the application in development mode"""
    print("🚀 Starting GI Abnormality Detection Application...")
    print("=" * 60)
    print("📋 Application Information:")
    print("   • URL: http://localhost:5000")
    print("   • Admin Login: admin@gi-detection.com / admin123")
    print("   • Environment: Development")
    print("=" * 60)
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

def run_production():
    """Run the application in production mode"""
    print("🚀 Starting GI Abnormality Detection Application (Production)...")
    print("=" * 60)
    print("📋 Application Information:")
    print("   • URL: http://localhost:5000")
    print("   • Environment: Production")
    print("   • Debug: Disabled")
    print("=" * 60)
    
    try:
        app.run(debug=False, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

def run_tests():
    """Run the test suite"""
    print("🧪 Running tests...")
    try:
        from test_app import run_tests
        success = run_tests()
        if success:
            print("✅ All tests passed!")
        else:
            print("❌ Some tests failed!")
            sys.exit(1)
    except ImportError:
        print("❌ Test module not found")
        sys.exit(1)

def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(
        description='GI Abnormality Detection Application',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                    # Run in development mode
  python run.py --production       # Run in production mode
  python run.py --setup            # Setup database and directories
  python run.py --test             # Run tests
        """
    )
    
    parser.add_argument(
        '--production', 
        action='store_true',
        help='Run in production mode'
    )
    
    parser.add_argument(
        '--setup', 
        action='store_true',
        help='Setup database and create directories'
    )
    
    parser.add_argument(
        '--test', 
        action='store_true',
        help='Run test suite'
    )
    
    parser.add_argument(
        '--port', 
        type=int, 
        default=5000,
        help='Port to run the application on (default: 5000)'
    )
    
    args = parser.parse_args()
    
    # Setup directories
    create_directories()
    
    # Setup database if requested
    if args.setup:
        setup_database()
        print("✅ Setup completed successfully!")
        return
    
    # Run tests if requested
    if args.test:
        run_tests()
        return
    
    # Setup database (always needed for running)
    setup_database()
    
    # Run application
    if args.production:
        run_production()
    else:
        run_development()

if __name__ == '__main__':
    main() 