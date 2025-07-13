#!/usr/bin/env python3
"""
Reset database script to fix relationship issues
"""

import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, db

def reset_database():
    """Reset the database and recreate all tables"""
    print("🗄️  Resetting database...")
    
    with app.app_context():
        # Drop all tables
        db.drop_all()
        print("✓ Dropped all tables")
        
        # Create all tables
        db.create_all()
        print("✓ Created all tables with proper relationships")
        
        # Create admin user
        from werkzeug.security import generate_password_hash
        from app import User
        
        admin = User(
            name='Administrator',
            email='admin@gi-detection.com',
            password_hash=generate_password_hash('admin123'),
            role='admin'
        )
        db.session.add(admin)
        db.session.commit()
        print("✓ Created admin user")
        
        print("✅ Database reset completed successfully!")

if __name__ == '__main__':
    reset_database() 