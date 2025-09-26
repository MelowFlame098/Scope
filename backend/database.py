from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import os
from dotenv import load_dotenv

load_dotenv()

# Database configuration
# For development, using SQLite instead of PostgreSQL to avoid dependency issues
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./finscope.db"
)

# For production, you can use PostgreSQL
# DATABASE_URL = "postgresql://postgres:KevinDurant1!@localhost:5432/finscope_db"

# Create engine
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
else:
    engine = create_engine(DATABASE_URL)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Database initialization
def init_db():
    """Initialize database with tables"""
    # Import Base from models.py to ensure we use the same Base as the models
    import importlib.util
    import os
    models_path = os.path.join(os.path.dirname(__file__), 'models.py')
    spec = importlib.util.spec_from_file_location("models_module", models_path)
    models_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(models_module)
    models_base = models_module.Base
    
    models_base.metadata.create_all(bind=engine)
    print("Database initialized successfully")

# Database connection test
def test_connection():
    """Test database connection"""
    try:
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        print("Database connection successful")
        return True
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False

if __name__ == "__main__":
    test_connection()
    init_db()