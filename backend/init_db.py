#!/usr/bin/env python3
"""
Database initialization script for FinScope
This script creates the PostgreSQL database and initializes all tables
"""

import os
import sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import Base, engine
from db_models import *  # Import all models to ensure they're registered

load_dotenv()

def create_database():
    """
    Create the PostgreSQL database if it doesn't exist
    """
    # Database connection parameters
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '5432')
    db_user = os.getenv('DB_USER', 'postgres')
    db_password = os.getenv('DB_PASSWORD', 'KevinDurant1!')
    db_name = os.getenv('DB_NAME', 'finscope_db')
    
    try:
        # Connect to PostgreSQL server (not to a specific database)
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_password,
            database='postgres'  # Connect to default postgres database
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'")
        exists = cursor.fetchone()
        
        if not exists:
            # Create the database
            cursor.execute(f'CREATE DATABASE "{db_name}"')
            print(f"✅ Database '{db_name}' created successfully")
        else:
            print(f"ℹ️  Database '{db_name}' already exists")
        
        cursor.close()
        conn.close()
        
    except psycopg2.Error as e:
        print(f"❌ Error creating database: {e}")
        return False
    
    return True

def create_tables():
    """
    Create all tables using SQLAlchemy models
    """
    try:
        # Create all tables from imported models
        Base.metadata.create_all(bind=engine)
        print("✅ All tables created successfully")
        return True
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        return False

def main():
    """
    Main initialization function
    """
    print("🚀 Starting FinScope database initialization...")
    
    # Step 1: Create database
    if not create_database():
        print("❌ Failed to create database. Exiting.")
        sys.exit(1)
    
    # Step 2: Create tables
    if not create_tables():
        print("❌ Failed to create tables. Exiting.")
        sys.exit(1)
    
    print("🎉 Database initialization completed successfully!")
    print("\n📋 Next steps:")
    print("1. Install PostgreSQL if not already installed")
    print("2. Update the .env file with your database credentials")
    print("3. Run the backend server: python main.py")

if __name__ == "__main__":
    main()