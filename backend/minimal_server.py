from fastapi import FastAPI, HTTPException, Depends, Header, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn
import uuid
import random
import string
import jwt
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Set
from enum import Enum
import operator
import json
import asyncio
import importlib.util
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
from database import engine, SessionLocal, init_db
import sys
import os
# Import directly from models.py file to avoid package conflict
models_path = os.path.join(os.path.dirname(__file__), 'models.py')
spec = importlib.util.spec_from_file_location("models_module", models_path)
models_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(models_module)
User = models_module.User
Base = models_module.Base
import bcrypt

app = FastAPI(title="FinScope Minimal Backend")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CaptchaRequest(BaseModel):
    captcha_type: str = "math"

class CaptchaResponse(BaseModel):
    challenge_id: str
    challenge_data: dict
    expires_at: str

# Simple in-memory storage for CAPTCHA challenges
captcha_store = {}

# Simple WebSocket manager
class SimpleWebSocketManager:
    def __init__(self):
        self.connections: Set[WebSocket] = set()
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.connections.add(websocket)
        print(f"WebSocket connected. Total connections: {len(self.connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.connections.discard(websocket)
        print(f"WebSocket disconnected. Total connections: {len(self.connections)}")
    
    async def send_message(self, websocket: WebSocket, message: dict):
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            print(f"Error sending WebSocket message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: dict):
        if self.connections:
            disconnected = set()
            for websocket in self.connections:
                try:
                    await websocket.send_text(json.dumps(message))
                except Exception as e:
                    print(f"Error broadcasting to WebSocket: {e}")
                    disconnected.add(websocket)
            
            # Remove disconnected websockets
            for websocket in disconnected:
                self.disconnect(websocket)

ws_manager = SimpleWebSocketManager()

def generate_math_captcha():
    """Generate a simple math CAPTCHA"""
    a = random.randint(1, 10)
    b = random.randint(1, 10)
    operation = random.choice(["+", "-", "*"])
    
    if operation == "+":
        answer = a + b
        question = f"{a} + {b}"
    elif operation == "-":
        answer = a - b
        question = f"{a} - {b}"
    else:  # multiplication
        answer = a * b
        question = f"{a} × {b}"
    
    return {
        "question": question,
        "answer": str(answer),
        "type": "math"
    }

@app.get("/")
async def root():
    return {"message": "FinScope Minimal Backend", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "minimal-backend"
    }

@app.post("/auth/captcha/challenge")
async def create_captcha_challenge(request: CaptchaRequest):
    try:
        challenge_id = str(uuid.uuid4())
        
        if request.captcha_type == "math":
            challenge_data = generate_math_captcha()
        else:
            # Default to math if unknown type
            challenge_data = generate_math_captcha()
        
        # Store the challenge (in production, use Redis or database)
        expires_at = datetime.now() + timedelta(minutes=5)
        captcha_store[challenge_id] = {
            "challenge_data": challenge_data,
            "expires_at": expires_at,
            "used": False
        }
        
        return {
            "challenge_id": challenge_id,
            "challenge_data": {
                "question": challenge_data["question"],
                "type": challenge_data["type"]
            },
            "expires_at": expires_at.isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create CAPTCHA: {str(e)}")

@app.post("/auth/captcha/verify")
async def verify_captcha(challenge_id: str, answer: str):
    try:
        if challenge_id not in captcha_store:
            raise HTTPException(status_code=400, detail="Invalid challenge ID")
        
        challenge = captcha_store[challenge_id]
        
        if challenge["used"]:
            raise HTTPException(status_code=400, detail="Challenge already used")
        
        if datetime.now() > challenge["expires_at"]:
            raise HTTPException(status_code=400, detail="Challenge expired")
        
        correct_answer = challenge["challenge_data"]["answer"]
        is_valid = answer.strip() == correct_answer
        
        # Mark as used
        challenge["used"] = True
        
        return {
            "valid": is_valid,
            "message": "Correct!" if is_valid else "Incorrect answer"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")

# Authentication models
class LoginRequest(BaseModel):
    email: str
    password: str
    captcha_id: Optional[str] = None
    captcha_answer: Optional[str] = None

class RegisterRequest(BaseModel):
    email: str
    password: str
    first_name: str
    last_name: str
    captcha_id: str
    captcha_answer: str

class UserResponse(BaseModel):
    id: str
    email: str
    first_name: str
    last_name: str
    subscription_plan: str = "free"
    subscription_status: str = "active"
    trial_ends_at: Optional[str] = None

# Subscription models
class SubscriptionPlan(str, Enum):
    FREE = "free"  # $0
    BASIC = "basic"  # $4.99
    PREMIUM = "premium"  # $9.99

class SubscriptionStatus(str, Enum):
    ACTIVE = "active"
    TRIAL = "trial"
    EXPIRED = "expired"
    CANCELLED = "cancelled"

class SubscriptionInfo(BaseModel):
    plan: SubscriptionPlan
    status: SubscriptionStatus
    trial_ends_at: Optional[datetime] = None
    subscription_ends_at: Optional[datetime] = None
    auto_renew: bool = True
    created_at: datetime
    updated_at: datetime

class SubscriptionRequest(BaseModel):
    plan: SubscriptionPlan
    trial: bool = False

class TrialInfo(BaseModel):
    email: str
    started_at: datetime
    expires_at: datetime
    used: bool = True
    created_at: datetime

class PlanFeatures(BaseModel):
    name: str
    price: float
    features: List[str]
    max_portfolios: int
    max_watchlists: int
    advanced_analytics: bool
    ai_insights: bool
    social_trading: bool
    institutional_tools: bool
    priority_support: bool

# Plan configurations
PLAN_FEATURES = {
    SubscriptionPlan.FREE: PlanFeatures(
        name="Free",
        price=0.00,
        features=["Basic market data", "Simple portfolio tracking", "3 watchlists", "Basic insights", "Community access", "Email support"],
        max_portfolios=1,
        max_watchlists=3,
        advanced_analytics=False,
        ai_insights=False,
        social_trading=False,
        institutional_tools=False,
        priority_support=False
    ),
    SubscriptionPlan.BASIC: PlanFeatures(
        name="Basic",
        price=4.99,
        features=["Real-time market data", "Advanced portfolio tracking", "10 watchlists", "AI insights", "Social trading", "Email support", "Mobile app access"],
        max_portfolios=5,
        max_watchlists=10,
        advanced_analytics=True,
        ai_insights=True,
        social_trading=True,
        institutional_tools=False,
        priority_support=False
    ),
    SubscriptionPlan.PREMIUM: PlanFeatures(
        name="Premium",
        price=9.99,
        features=["Everything in Basic", "Unlimited watchlists", "Institutional tools", "Priority support", "Advanced AI analytics", "Custom alerts", "Advanced charting tools"],
        max_portfolios=999,
        max_watchlists=999,
        advanced_analytics=True,
        ai_insights=True,
        social_trading=True,
        institutional_tools=True,
        priority_support=True
    )
}

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize database
init_db()

# Database-based subscription management (no more in-memory storage)
security = HTTPBearer()
SECRET_KEY = "your-secret-key-here"

def create_access_token(user_id: str):
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + timedelta(hours=24)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload.get("user_id")
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

@app.post("/auth/login")
async def login(request: LoginRequest, db: SessionLocal = Depends(get_db)):
    # Verify CAPTCHA if provided
    if request.captcha_id and request.captcha_answer:
        if request.captcha_id not in captcha_store:
            raise HTTPException(status_code=400, detail="Invalid CAPTCHA")
        
        challenge = captcha_store[request.captcha_id]
        if challenge["used"] or datetime.now() > challenge["expires_at"]:
            raise HTTPException(status_code=400, detail="CAPTCHA expired or used")
        
        if request.captcha_answer.strip() != challenge["challenge_data"]["answer"]:
            raise HTTPException(status_code=400, detail="Invalid CAPTCHA answer")
        
        # Mark CAPTCHA as used
        challenge["used"] = True
    
    # Database authentication
    user = db.query(User).filter(User.email == request.email.lower()).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Verify password
    if not bcrypt.checkpw(request.password.encode('utf-8'), user.hashed_password.encode('utf-8')):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token = create_access_token(user.id)
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "email": user.email,
            "username": user.username,
            "full_name": user.full_name,
            "subscription_plan": user.subscription_plan,
            "is_trial_active": user.is_trial_active
        }
    }

@app.post("/auth/register")
async def register(request: RegisterRequest, db: SessionLocal = Depends(get_db)):
    # Verify CAPTCHA first
    if request.captcha_id not in captcha_store:
        raise HTTPException(status_code=400, detail="Invalid CAPTCHA")
    
    challenge = captcha_store[request.captcha_id]
    if challenge["used"] or datetime.now() > challenge["expires_at"]:
        raise HTTPException(status_code=400, detail="CAPTCHA expired or used")
    
    if request.captcha_answer.strip() != challenge["challenge_data"]["answer"]:
        raise HTTPException(status_code=400, detail="Invalid CAPTCHA answer")
    
    # Mark CAPTCHA as used
    challenge["used"] = True
    
    # Check if user already exists
    existing_user = db.query(User).filter(User.email == request.email.lower()).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")
    
    # Check if email has already used a trial
    trial_used_user = db.query(User).filter(User.email == request.email.lower(), User.trial_used == True).first()
    if trial_used_user:
        raise HTTPException(status_code=400, detail="This email has already used a free trial")
    
    # Hash password
    hashed_password = bcrypt.hashpw(request.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    # Create trial dates
    now = datetime.now()
    trial_expires = now + timedelta(days=7)
    
    # Create new user with trial
    new_user = User(
        email=request.email.lower(),
        username=request.email.lower(),  # Use email as username for now
        full_name=f"{request.first_name} {request.last_name}",
        hashed_password=hashed_password,
        subscription_plan="BASIC",  # Default to BASIC plan
        trial_used=True,
        trial_start_date=now,
        trial_end_date=trial_expires,
        is_trial_active=True
    )
    
    try:
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="User already exists")
    
    return {
        "message": "User registered successfully with 7-day free trial",
        "user_id": new_user.id,
        "trial_expires_at": trial_expires.isoformat()
    }

@app.get("/auth/me")
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: SessionLocal = Depends(get_db)):
    user_id = verify_token(credentials.credentials)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    # Find user by ID in database
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Check trial status
    trial_expired = False
    if user.is_trial_active and user.trial_end_date:
        if datetime.now() > user.trial_end_date:
            # Update trial status to expired
            user.is_trial_active = False
            db.commit()
            trial_expired = True
    
    return {
        "id": user.id,
        "email": user.email,
        "username": user.username,
        "full_name": user.full_name,
        "subscription_plan": user.subscription_plan,
        "is_trial_active": user.is_trial_active,
        "trial_end_date": user.trial_end_date.isoformat() if user.trial_end_date else None,
        "trial_expired": trial_expired
    }

@app.get("/auth/trial-status")
async def get_trial_status(credentials: HTTPAuthorizationCredentials = Depends(security), db: SessionLocal = Depends(get_db)):
    user_id = verify_token(credentials.credentials)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Check if trial is expired
    trial_expired = False
    if user.is_trial_active and user.trial_end_date:
        if datetime.now() > user.trial_end_date:
            user.is_trial_active = False
            db.commit()
            trial_expired = True
    
    days_remaining = 0
    if user.trial_end_date and user.is_trial_active:
        days_remaining = max(0, (user.trial_end_date - datetime.now()).days)
    
    return {
        "plan": user.subscription_plan,
        "is_trial_active": user.is_trial_active,
        "trial_ends_at": user.trial_end_date.isoformat() if user.trial_end_date else None,
        "trial_expired": trial_expired,
        "days_remaining": days_remaining
    }

# Subscription endpoints
@app.get("/subscription/plans")
async def get_subscription_plans():
    """Get all available subscription plans"""
    return {
        "plans": [
            {
                "id": plan.value,
                "name": features.name,
                "price": features.price,
                "features": features.features,
                "limits": {
                    "portfolios": features.max_portfolios,
                    "watchlists": features.max_watchlists
                },
                "capabilities": {
                    "advanced_analytics": features.advanced_analytics,
                    "ai_insights": features.ai_insights,
                    "social_trading": features.social_trading,
                    "institutional_tools": features.institutional_tools,
                    "priority_support": features.priority_support
                }
            }
            for plan, features in PLAN_FEATURES.items()
        ]
    }

@app.get("/subscription/current")
async def get_current_subscription(credentials: HTTPAuthorizationCredentials = Depends(security), db: SessionLocal = Depends(get_db)):
    """Get current user's subscription info"""
    user_id = verify_token(credentials.credentials)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Check trial status
    trial_expired = False
    if user.is_trial_active and user.trial_end_date:
        if datetime.now() > user.trial_end_date:
            user.is_trial_active = False
            db.commit()
            trial_expired = True
    
    subscription = {
        "plan": user.subscription_plan,
        "status": "trial" if user.is_trial_active else "active",
        "trial_ends_at": user.trial_end_date.isoformat() if user.trial_end_date else None,
        "subscription_ends_at": None,
        "auto_renew": True,
        "created_at": user.created_at.isoformat() if user.created_at else None,
        "updated_at": user.updated_at.isoformat() if user.updated_at else None
    }
    
    plan_enum = SubscriptionPlan(user.subscription_plan.lower())
    return {
        "subscription": subscription,
        "features": PLAN_FEATURES[plan_enum].dict()
    }

@app.post("/subscription/subscribe")
async def subscribe_to_plan(
    request: SubscriptionRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: SessionLocal = Depends(get_db)
):
    """Subscribe to a plan or start a trial"""
    user_id = verify_token(credentials.credentials)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    current_time = datetime.now()
    
    # Check if user is already on a paid plan or active trial
    if user.subscription_plan in ["BASIC", "PREMIUM"] and user.is_trial_active:
        raise HTTPException(status_code=400, detail="Already subscribed to a plan")
    
    # Update user subscription
    user.subscription_plan = request.plan.value.upper()
    
    if request.trial:
        # Start trial (note: this is different from registration trial)
        trial_ends_at = current_time + timedelta(days=30)
        user.trial_start_date = current_time
        user.trial_end_date = trial_ends_at
        user.is_trial_active = True
        status = "trial"
    else:
        # Direct subscription (would integrate with payment processor)
        user.is_trial_active = False
        status = "active"
    
    user.updated_at = current_time
    db.commit()
    
    subscription = {
        "plan": request.plan.value,
        "status": status,
        "trial_ends_at": user.trial_end_date.isoformat() if user.trial_end_date else None,
        "subscription_ends_at": None,
        "auto_renew": True,
        "created_at": user.created_at.isoformat() if user.created_at else None,
        "updated_at": user.updated_at.isoformat()
    }
    
    return {
        "message": "Subscription successful",
        "subscription": subscription,
        "features": PLAN_FEATURES[request.plan].dict()
    }

@app.post("/subscription/cancel")
async def cancel_subscription(credentials: HTTPAuthorizationCredentials = Depends(security), db: SessionLocal = Depends(get_db)):
    """Cancel current subscription"""
    user_id = verify_token(credentials.credentials)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if user.subscription_plan not in ["BASIC", "PREMIUM"]:
        raise HTTPException(status_code=404, detail="No active subscription found")
    
    # Reset to basic plan and deactivate trial
    user.subscription_plan = "BASIC"
    user.is_trial_active = False
    user.updated_at = datetime.now()
    db.commit()
    
    return {"message": "Subscription cancelled successfully"}

@app.get("/subscription/trial-status")
async def get_trial_status_subscription(credentials: HTTPAuthorizationCredentials = Depends(security), db: SessionLocal = Depends(get_db)):
    """Get trial status and handle auto-downgrade"""
    user_id = verify_token(credentials.credentials)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    current_time = datetime.now()
    
    # Check if trial has expired and auto-downgrade
    trial_expired = False
    if user.is_trial_active and user.trial_end_date and current_time > user.trial_end_date:
        user.is_trial_active = False
        user.updated_at = current_time
        db.commit()
        trial_expired = True
        
        return {
            "is_trial": False,
            "trial_expired": True,
            "days_remaining": 0,
            "can_start_trial": False,
            "message": "Trial expired"
        }
    
    is_trial = user.is_trial_active
    days_remaining = 0
    
    if is_trial and user.trial_end_date:
        time_remaining = user.trial_end_date - current_time
        days_remaining = max(0, time_remaining.days)
    
    # Can start trial if not currently on trial and hasn't used trial before
    can_start_trial = not user.is_trial_active and not user.trial_used
    
    return {
        "is_trial": is_trial,
        "trial_expired": trial_expired,
        "days_remaining": days_remaining,
        "can_start_trial": can_start_trial,
        "message": f"{days_remaining} days remaining" if is_trial else "Not on trial"
    }

@app.get("/subscription/check-access/{feature}")
async def check_feature_access(
    feature: str,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: SessionLocal = Depends(get_db)
):
    """Check if user has access to a specific feature"""
    user_id = verify_token(credentials.credentials)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Check if trial has expired
    current_time = datetime.now()
    if user.is_trial_active and user.trial_end_date and current_time > user.trial_end_date:
        user.is_trial_active = False
        user.updated_at = current_time
        db.commit()
    
    # Get plan features
    plan_enum = SubscriptionPlan(user.subscription_plan.lower())
    plan_features = PLAN_FEATURES[plan_enum]
    
    # Feature access mapping
    feature_access = {
        "advanced_analytics": plan_features.advanced_analytics,
        "ai_insights": plan_features.ai_insights,
        "social_trading": plan_features.social_trading,
        "institutional_tools": plan_features.institutional_tools,
        "priority_support": plan_features.priority_support
    }
    
    has_access = feature_access.get(feature, False)
    
    return {
        "has_access": has_access,
        "plan": user.subscription_plan,
        "status": "trial" if user.is_trial_active else "active"
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await ws_manager.connect(websocket)
    try:
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                message_type = message.get('type', 'unknown')
                
                # Handle different message types
                if message_type == 'subscribe':
                    # Send acknowledgment
                    await ws_manager.send_message(websocket, {
                        'type': 'subscribed',
                        'data': {'channels': message.get('data', {}).get('channels', [])},
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # Start sending periodic updates
                    asyncio.create_task(send_periodic_updates(websocket))
                    
                elif message_type == 'ping':
                    # Respond to ping with pong
                    await ws_manager.send_message(websocket, {
                        'type': 'pong',
                        'timestamp': datetime.now().isoformat()
                    })
                    
                else:
                    # Echo unknown messages
                    await ws_manager.send_message(websocket, {
                        'type': 'echo',
                        'data': message,
                        'timestamp': datetime.now().isoformat()
                    })
                    
            except json.JSONDecodeError:
                await ws_manager.send_message(websocket, {
                    'type': 'error',
                    'data': {'message': 'Invalid JSON format'},
                    'timestamp': datetime.now().isoformat()
                })
                
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        ws_manager.disconnect(websocket)

async def send_periodic_updates(websocket: WebSocket):
    """Send periodic market data updates"""
    try:
        while websocket in ws_manager.connections:
            # Send mock market data
            await ws_manager.send_message(websocket, {
                'type': 'market_update',
                'data': {
                    'symbol': 'AAPL',
                    'price': 150 + random.uniform(-5, 5),
                    'change': random.uniform(-2, 2),
                    'volume': random.randint(1000000, 5000000)
                },
                'timestamp': datetime.now().isoformat()
            })
            
            # Send mock news update occasionally
            if random.random() < 0.3:
                await ws_manager.send_message(websocket, {
                    'type': 'news_update',
                    'data': {
                        'title': 'Market Update',
                        'content': 'Sample news content for testing',
                        'sentiment': random.choice(['positive', 'negative', 'neutral'])
                    },
                    'timestamp': datetime.now().isoformat()
                })
            
            await asyncio.sleep(5)  # Update every 5 seconds
            
    except Exception as e:
        print(f"Error in periodic updates: {e}")

if __name__ == "__main__":
    uvicorn.run(
        "minimal_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )