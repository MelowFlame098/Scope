from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
from database import get_db
from auth import get_current_user

router = APIRouter(prefix="/mobile", tags=["mobile"])

# Mobile-specific models
class MobileDeviceInfo(BaseModel):
    device_id: str
    device_type: str  # ios, android
    app_version: str
    os_version: str
    push_token: Optional[str] = None

class MobileUserPreferences(BaseModel):
    notifications_enabled: bool = True
    biometric_auth: bool = False
    dark_mode: bool = False
    quick_actions: List[str] = []
    watchlist_sync: bool = True

class MobilePortfolioSummary(BaseModel):
    total_value: float
    daily_change: float
    daily_change_percent: float
    top_performers: List[Dict[str, Any]]
    alerts_count: int

class MobileTradingQuote(BaseModel):
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    market_cap: Optional[float] = None

class MobileNotification(BaseModel):
    id: str
    title: str
    message: str
    type: str  # alert, trade, news, system
    timestamp: datetime
    read: bool = False
    action_url: Optional[str] = None

class MobileAPIService:
    def __init__(self):
        self.active_devices = {}
        self.user_preferences = {}
        
    def register_device(self, user_id: str, device_info: MobileDeviceInfo) -> Dict[str, Any]:
        """Register a mobile device for a user"""
        if user_id not in self.active_devices:
            self.active_devices[user_id] = []
        
        # Remove existing device with same ID
        self.active_devices[user_id] = [
            d for d in self.active_devices[user_id] 
            if d.get('device_id') != device_info.device_id
        ]
        
        device_data = device_info.dict()
        device_data['registered_at'] = datetime.now()
        device_data['last_active'] = datetime.now()
        
        self.active_devices[user_id].append(device_data)
        
        return {
            "status": "registered",
            "device_id": device_info.device_id,
            "features_enabled": [
                "push_notifications",
                "biometric_auth",
                "offline_mode",
                "quick_trade"
            ]
        }
    
    def get_user_preferences(self, user_id: str) -> MobileUserPreferences:
        """Get mobile preferences for a user"""
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = MobileUserPreferences()
        return self.user_preferences[user_id]
    
    def update_user_preferences(self, user_id: str, preferences: MobileUserPreferences) -> bool:
        """Update mobile preferences for a user"""
        self.user_preferences[user_id] = preferences
        return True
    
    def get_portfolio_summary(self, user_id: str) -> MobilePortfolioSummary:
        """Get mobile-optimized portfolio summary"""
        # Mock data for demonstration
        return MobilePortfolioSummary(
            total_value=125000.50,
            daily_change=2500.75,
            daily_change_percent=2.04,
            top_performers=[
                {"symbol": "AAPL", "change_percent": 3.2, "value": 15000},
                {"symbol": "GOOGL", "change_percent": 2.8, "value": 12000},
                {"symbol": "MSFT", "change_percent": 1.9, "value": 18000}
            ],
            alerts_count=3
        )
    
    def get_quick_quotes(self, symbols: List[str]) -> List[MobileTradingQuote]:
        """Get quick quotes for mobile trading"""
        # Mock data for demonstration
        quotes = []
        for symbol in symbols[:10]:  # Limit to 10 for mobile
            quotes.append(MobileTradingQuote(
                symbol=symbol,
                price=150.25 + hash(symbol) % 100,
                change=2.50 + hash(symbol) % 10,
                change_percent=1.69 + hash(symbol) % 5,
                volume=1000000 + hash(symbol) % 500000,
                market_cap=50000000000 + hash(symbol) % 10000000000
            ))
        return quotes
    
    def get_mobile_notifications(self, user_id: str, limit: int = 20) -> List[MobileNotification]:
        """Get mobile-optimized notifications"""
        # Mock data for demonstration
        notifications = []
        for i in range(min(limit, 20)):
            notifications.append(MobileNotification(
                id=f"notif_{i}",
                title=f"Price Alert: AAPL",
                message=f"AAPL has reached your target price of $150.00",
                type="alert",
                timestamp=datetime.now(),
                read=i % 3 == 0,
                action_url=f"/trading/AAPL"
            ))
        return notifications
    
    def send_push_notification(self, user_id: str, notification: MobileNotification) -> bool:
        """Send push notification to user's devices"""
        if user_id not in self.active_devices:
            return False
        
        # In a real implementation, this would send to FCM/APNS
        for device in self.active_devices[user_id]:
            if device.get('push_token'):
                # Mock sending notification
                print(f"Sending push to {device['device_id']}: {notification.title}")
        
        return True

# Initialize service
mobile_api_service = MobileAPIService()

# API Endpoints
@router.post("/register-device")
async def register_device(
    device_info: MobileDeviceInfo,
    current_user = Depends(get_current_user)
):
    """Register a mobile device"""
    result = mobile_api_service.register_device(current_user.id, device_info)
    return result

@router.get("/preferences")
async def get_preferences(current_user = Depends(get_current_user)):
    """Get user's mobile preferences"""
    preferences = mobile_api_service.get_user_preferences(current_user.id)
    return preferences

@router.put("/preferences")
async def update_preferences(
    preferences: MobileUserPreferences,
    current_user = Depends(get_current_user)
):
    """Update user's mobile preferences"""
    success = mobile_api_service.update_user_preferences(current_user.id, preferences)
    if success:
        return {"status": "updated"}
    raise HTTPException(status_code=400, detail="Failed to update preferences")

@router.get("/portfolio/summary")
async def get_portfolio_summary(current_user = Depends(get_current_user)):
    """Get mobile-optimized portfolio summary"""
    summary = mobile_api_service.get_portfolio_summary(current_user.id)
    return summary

@router.post("/quotes")
async def get_quick_quotes(
    symbols: List[str],
    current_user = Depends(get_current_user)
):
    """Get quick quotes for mobile"""
    quotes = mobile_api_service.get_quick_quotes(symbols)
    return quotes

@router.get("/notifications")
async def get_notifications(
    limit: int = 20,
    current_user = Depends(get_current_user)
):
    """Get mobile notifications"""
    notifications = mobile_api_service.get_mobile_notifications(current_user.id, limit)
    return notifications

@router.post("/notifications/{notification_id}/read")
async def mark_notification_read(
    notification_id: str,
    current_user = Depends(get_current_user)
):
    """Mark notification as read"""
    # In a real implementation, this would update the database
    return {"status": "marked_read", "notification_id": notification_id}

@router.get("/health")
async def mobile_health_check():
    """Mobile API health check"""
    return {
        "status": "healthy",
        "service": "mobile_api",
        "timestamp": datetime.now(),
        "features": [
            "device_registration",
            "push_notifications",
            "portfolio_summary",
            "quick_quotes",
            "mobile_notifications"
        ]
    }

# Export service and router
__all__ = ["MobileAPIService", "mobile_api_service", "router"]