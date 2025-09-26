import importlib.util
import os
import sys

# Add backend to path
sys.path.append('backend')

# Import models
models_path = os.path.join('backend', 'models.py')
spec = importlib.util.spec_from_file_location('models_module', models_path)
models_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(models_module)
User = models_module.User

# Import database
from backend.database import SessionLocal

db = SessionLocal()
try:
    users = db.query(User).all()
    print(f'Found {len(users)} users:')
    for user in users:
        print(f'- {user.email} (ID: {user.id})')
finally:
    db.close()