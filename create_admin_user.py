#!/usr/bin/env python3
"""
Script to create an admin user for FinScope application.
This script creates an admin user with the specified email and details.
"""

import sys
import os
from datetime import datetime, timedelta
import uuid
import bcrypt
import importlib.util

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

try:
    # Import from the backend directory
    backend_path = os.path.join(os.path.dirname(__file__), 'backend')
    sys.path.insert(0, backend_path)
    
    # Import the specific models.py file (not the models package)
    import importlib.util
    models_path = os.path.join(backend_path, 'models.py')
    spec = importlib.util.spec_from_file_location("models_module", models_path)
    models_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(models_module)
    
    User = models_module.User
    Base = models_module.Base
    
    from database import SessionLocal, init_db
except ImportError as e:
    print(f"Error importing backend modules: {e}")
    print("Make sure you're running this script from the FinScope root directory")
    sys.exit(1)
except Exception as e:
    print(f"Error setting up imports: {e}")
    sys.exit(1)

def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def create_admin_user():
    """Create an admin user with the specified details."""
    
    # Initialize database
    init_db()
    
    # Create database session
    db = SessionLocal()
    
    try:
        # Admin user details
        admin_email = "yaragandla0904@gmail.com"
        admin_password = "AdminPass123!"
        admin_first_name = "Richie"
        admin_last_name = "Yaragandla"
        admin_username = "richie_yaragandla"
        
        print(f"Creating admin user with email: {admin_email}")
        
        # Check if user already exists
        existing_user = db.query(User).filter(User.email == admin_email).first()
        if existing_user:
            print(f"\n✅ User with email {admin_email} already exists!")
            print(f"User ID: {existing_user.id}")
            print(f"Username: {existing_user.username}")
            print(f"Full Name: {existing_user.full_name}")
            print(f"Is Verified: {existing_user.is_verified}")
            print(f"Subscription: {existing_user.subscription_plan}")
            
            # Update existing user to admin privileges
            existing_user.is_verified = True
            existing_user.is_active = True
            existing_user.subscription_plan = "PREMIUM"
            existing_user.trial_used = True
            existing_user.is_trial_active = False
            existing_user.updated_at = datetime.utcnow()
            existing_user.full_name = f"{admin_first_name} {admin_last_name}"
            
            # Update password
            hashed_password = hash_password(admin_password)
            existing_user.hashed_password = hashed_password
            
            db.commit()
            print("\n🔄 Updated existing user with admin privileges and premium access!")
            print(f"\n🔑 Login credentials:")
            print(f"Email: {admin_email}")
            print(f"Password: {admin_password}")
            return existing_user.id
        
        # Hash the password
        hashed_password = hash_password(admin_password)
        
        # Create new admin user
        admin_user = User(
            id=str(uuid.uuid4()),
            email=admin_email,
            username=admin_username,
            full_name=f"{admin_first_name} {admin_last_name}",
            hashed_password=hashed_password,
            is_active=True,
            is_verified=True,
            subscription_plan="PREMIUM",
            trial_used=True,
            is_trial_active=False,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            risk_tolerance="medium",
            trading_experience="advanced"
        )
        
        db.add(admin_user)
        db.commit()
        db.refresh(admin_user)
        
        print("\n✅ Admin user created successfully!")
        print(f"Email: {admin_email}")
        print(f"Name: {admin_first_name} {admin_last_name}")
        print(f"Username: {admin_username}")
        print(f"User ID: {admin_user.id}")
        print(f"Subscription: {admin_user.subscription_plan}")
        print(f"Verified: {admin_user.is_verified}")
        print("\n🔑 Login credentials:")
        print(f"Email: {admin_email}")
        print(f"Password: {admin_password}")
        
        return admin_user.id
        
    except Exception as e:
        print(f"Error creating admin user: {e}")
        db.rollback()
        raise
    finally:
        db.close()

if __name__ == "__main__":
    print("Creating admin user for FinScope...")
    try:
        user_id = create_admin_user()
        print(f"\n🎉 Admin user setup complete! User ID: {user_id}")
        print("\nYou can now log in to the application with the admin credentials.")
        print("\n📝 Next steps:")
        print("1. Open the application in your browser (http://localhost:3000)")
        print("2. Go to the login page (/auth/login)")
        print("3. Use the email and password shown above")
        print("4. You should have full access to all premium features")
    except Exception as e:
        print(f"\n❌ Failed to create admin user: {e}")
        sys.exit(1)