"""Notification System for FinScope - Phase 7 Implementation

Provides comprehensive notification capabilities including real-time alerts,
email notifications, push notifications, and user preference management.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json

import pandas as pd
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc

from database import get_db
from db_models import User, Notification, NotificationPreference, PriceAlert
from market_data import MarketDataService
from portfolio_manager import PortfolioManager

logger = logging.getLogger(__name__)

class NotificationType(str, Enum):
    """Types of notifications"""
    PRICE_ALERT = "price_alert"
    PORTFOLIO_UPDATE = "portfolio_update"
    TRADE_EXECUTION = "trade_execution"
    MARKET_NEWS = "market_news"
    COMMUNITY_ACTIVITY = "community_activity"
    SYSTEM_ANNOUNCEMENT = "system_announcement"
    RISK_WARNING = "risk_warning"
    EARNINGS_ALERT = "earnings_alert"
    DIVIDEND_ALERT = "dividend_alert"
    TECHNICAL_SIGNAL = "technical_signal"
    FUNDAMENTAL_CHANGE = "fundamental_change"
    SOCIAL_SENTIMENT = "social_sentiment"

class NotificationPriority(str, Enum):
    """Notification priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"

class NotificationStatus(str, Enum):
    """Notification status"""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"
    FAILED = "failed"
    CANCELLED = "cancelled"

class DeliveryChannel(str, Enum):
    """Notification delivery channels"""
    IN_APP = "in_app"
    EMAIL = "email"
    PUSH = "push"
    SMS = "sms"
    WEBHOOK = "webhook"
    SLACK = "slack"
    DISCORD = "discord"

class AlertCondition(str, Enum):
    """Price alert conditions"""
    ABOVE = "above"
    BELOW = "below"
    CROSSES_ABOVE = "crosses_above"
    CROSSES_BELOW = "crosses_below"
    PERCENT_CHANGE = "percent_change"
    VOLUME_SPIKE = "volume_spike"
    VOLATILITY_SPIKE = "volatility_spike"

class FrequencyType(str, Enum):
    """Notification frequency"""
    IMMEDIATE = "immediate"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"

@dataclass
class NotificationContent:
    """Notification content structure"""
    title: str
    message: str
    action_url: Optional[str] = None
    action_text: Optional[str] = None
    image_url: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class NotificationTemplate:
    """Notification template"""
    template_id: str
    notification_type: NotificationType
    title_template: str
    message_template: str
    default_priority: NotificationPriority
    default_channels: List[DeliveryChannel]
    variables: List[str]

class NotificationRequest(BaseModel):
    """Request for sending notifications"""
    user_id: str
    notification_type: NotificationType
    priority: NotificationPriority = NotificationPriority.MEDIUM
    channels: List[DeliveryChannel] = [DeliveryChannel.IN_APP]
    content: Dict[str, Any]
    scheduled_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = {}

class NotificationResponse(BaseModel):
    """Response for notification operations"""
    notification_id: str
    user_id: str
    notification_type: NotificationType
    priority: NotificationPriority
    status: NotificationStatus
    channels: List[DeliveryChannel]
    content: NotificationContent
    created_at: datetime
    sent_at: Optional[datetime]
    read_at: Optional[datetime]
    expires_at: Optional[datetime]

class PriceAlertRequest(BaseModel):
    """Request for creating price alerts"""
    symbol: str
    condition: AlertCondition
    target_value: float
    comparison_value: Optional[float] = None  # For percent change alerts
    is_active: bool = True
    expires_at: Optional[datetime] = None
    notes: Optional[str] = None

class PriceAlertResponse(BaseModel):
    """Response for price alert operations"""
    alert_id: str
    user_id: str
    symbol: str
    condition: AlertCondition
    target_value: float
    comparison_value: Optional[float]
    current_value: float
    is_active: bool
    triggered_at: Optional[datetime]
    created_at: datetime
    expires_at: Optional[datetime]
    notes: Optional[str]

class NotificationPreferenceRequest(BaseModel):
    """Request for updating notification preferences"""
    notification_type: NotificationType
    enabled: bool = True
    channels: List[DeliveryChannel] = [DeliveryChannel.IN_APP]
    frequency: FrequencyType = FrequencyType.IMMEDIATE
    quiet_hours_start: Optional[str] = None  # HH:MM format
    quiet_hours_end: Optional[str] = None    # HH:MM format
    min_priority: NotificationPriority = NotificationPriority.LOW

class NotificationSystem:
    """Advanced notification and alerting system"""
    
    def __init__(self):
        self.market_service = MarketDataService()
        self.portfolio_manager = PortfolioManager()
        
        # System configuration
        self.max_notifications_per_hour = 50
        self.max_alerts_per_user = 100
        self.default_expiry_hours = 24
        
        # Notification templates
        self.templates = self._initialize_templates()
        
        # Active price alerts monitoring
        self.active_alerts: Dict[str, PriceAlertResponse] = {}
        self.alert_check_interval = 60  # seconds
        
        # Delivery channels configuration
        self.channel_config = {
            DeliveryChannel.EMAIL: {
                "enabled": True,
                "rate_limit": 10,  # per hour
                "template_format": "html"
            },
            DeliveryChannel.PUSH: {
                "enabled": True,
                "rate_limit": 20,  # per hour
                "template_format": "text"
            },
            DeliveryChannel.SMS: {
                "enabled": False,  # Requires SMS service setup
                "rate_limit": 5,   # per hour
                "template_format": "text"
            }
        }
        
        # Background tasks
        self._monitoring_task = None
        self._cleanup_task = None
    
    async def send_notification(
        self,
        request: NotificationRequest,
        db: Session
    ) -> NotificationResponse:
        """Send a notification to a user"""
        try:
            # Validate user exists
            user = db.query(User).filter(User.id == request.user_id).first()
            if not user:
                raise ValueError(f"User {request.user_id} not found")
            
            # Check rate limits
            if not await self._check_rate_limits(request.user_id, db):
                raise ValueError("Rate limit exceeded for notifications")
            
            # Check user preferences
            filtered_channels = await self._filter_channels_by_preferences(
                request.user_id, request.notification_type, request.channels, db
            )
            
            if not filtered_channels:
                logger.info(f"No enabled channels for user {request.user_id}, notification type {request.notification_type}")
                return None
            
            # Generate notification ID
            notification_id = self._generate_notification_id()
            
            # Process content using template if available
            content = await self._process_notification_content(
                request.notification_type, request.content
            )
            
            # Set expiry if not provided
            expires_at = request.expires_at or (
                datetime.utcnow() + timedelta(hours=self.default_expiry_hours)
            )
            
            # Create notification record
            notification = Notification(
                id=notification_id,
                user_id=request.user_id,
                notification_type=request.notification_type.value,
                priority=request.priority.value,
                title=content.title,
                message=content.message,
                action_url=content.action_url,
                action_text=content.action_text,
                image_url=content.image_url,
                channels=json.dumps([c.value for c in filtered_channels]),
                status=NotificationStatus.PENDING.value,
                notification_metadata=json.dumps(request.metadata),
                created_at=datetime.utcnow(),
                scheduled_at=request.scheduled_at or datetime.utcnow(),
                expires_at=expires_at
            )
            
            db.add(notification)
            db.commit()
            
            # Send through channels
            delivery_results = await self._deliver_notification(
                notification, filtered_channels, content
            )
            
            # Update status based on delivery results
            if any(delivery_results.values()):
                notification.status = NotificationStatus.SENT.value
                notification.sent_at = datetime.utcnow()
            else:
                notification.status = NotificationStatus.FAILED.value
            
            db.commit()
            
            # Create response
            response = NotificationResponse(
                notification_id=notification_id,
                user_id=request.user_id,
                notification_type=request.notification_type,
                priority=request.priority,
                status=NotificationStatus(notification.status),
                channels=filtered_channels,
                content=content,
                created_at=notification.created_at,
                sent_at=notification.sent_at,
                read_at=notification.read_at,
                expires_at=notification.expires_at
            )
            
            logger.info(
                f"Notification {notification_id} sent to user {request.user_id} "
                f"via channels: {[c.value for c in filtered_channels]}"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error sending notification: {str(e)}")
            db.rollback()
            raise
    
    async def create_price_alert(
        self,
        request: PriceAlertRequest,
        user_id: str,
        db: Session
    ) -> PriceAlertResponse:
        """Create a price alert"""
        try:
            # Check user alert limits
            existing_alerts = db.query(PriceAlert).filter(
                PriceAlert.user_id == user_id,
                PriceAlert.is_active == True
            ).count()
            
            if existing_alerts >= self.max_alerts_per_user:
                raise ValueError(f"Maximum number of alerts ({self.max_alerts_per_user}) reached")
            
            # Validate symbol and get current price
            try:
                current_price = await self.market_service.get_current_price(request.symbol)
            except Exception:
                raise ValueError(f"Invalid symbol: {request.symbol}")
            
            # Generate alert ID
            alert_id = self._generate_alert_id()
            
            # Create alert record
            alert = PriceAlert(
                id=alert_id,
                user_id=user_id,
                symbol=request.symbol,
                condition=request.condition.value,
                target_value=request.target_value,
                comparison_value=request.comparison_value,
                is_active=request.is_active,
                created_at=datetime.utcnow(),
                expires_at=request.expires_at,
                notes=request.notes
            )
            
            db.add(alert)
            db.commit()
            
            # Add to active monitoring if enabled
            if request.is_active:
                alert_response = PriceAlertResponse(
                    alert_id=alert_id,
                    user_id=user_id,
                    symbol=request.symbol,
                    condition=request.condition,
                    target_value=request.target_value,
                    comparison_value=request.comparison_value,
                    current_value=current_price,
                    is_active=request.is_active,
                    triggered_at=None,
                    created_at=alert.created_at,
                    expires_at=request.expires_at,
                    notes=request.notes
                )
                
                self.active_alerts[alert_id] = alert_response
            
            logger.info(f"Price alert {alert_id} created for {request.symbol} by user {user_id}")
            
            return PriceAlertResponse(
                alert_id=alert_id,
                user_id=user_id,
                symbol=request.symbol,
                condition=request.condition,
                target_value=request.target_value,
                comparison_value=request.comparison_value,
                current_value=current_price,
                is_active=request.is_active,
                triggered_at=None,
                created_at=alert.created_at,
                expires_at=request.expires_at,
                notes=request.notes
            )
            
        except Exception as e:
            logger.error(f"Error creating price alert: {str(e)}")
            db.rollback()
            raise
    
    async def update_notification_preferences(
        self,
        request: NotificationPreferenceRequest,
        user_id: str,
        db: Session
    ) -> Dict[str, Any]:
        """Update user notification preferences"""
        try:
            # Get or create preference record
            preference = db.query(NotificationPreference).filter(
                NotificationPreference.user_id == user_id,
                NotificationPreference.notification_type == request.notification_type.value
            ).first()
            
            if not preference:
                preference = NotificationPreference(
                    user_id=user_id,
                    notification_type=request.notification_type.value
                )
                db.add(preference)
            
            # Update preference fields
            preference.enabled = request.enabled
            preference.channels = json.dumps([c.value for c in request.channels])
            preference.frequency = request.frequency.value
            preference.quiet_hours_start = request.quiet_hours_start
            preference.quiet_hours_end = request.quiet_hours_end
            preference.min_priority = request.min_priority.value
            preference.updated_at = datetime.utcnow()
            
            db.commit()
            
            logger.info(
                f"Notification preferences updated for user {user_id}, "
                f"type {request.notification_type.value}"
            )
            
            return {
                "user_id": user_id,
                "notification_type": request.notification_type.value,
                "status": "updated",
                "preferences": {
                    "enabled": request.enabled,
                    "channels": [c.value for c in request.channels],
                    "frequency": request.frequency.value,
                    "min_priority": request.min_priority.value
                }
            }
            
        except Exception as e:
            logger.error(f"Error updating notification preferences: {str(e)}")
            db.rollback()
            raise
    
    async def get_user_notifications(
        self,
        user_id: str,
        db: Session,
        unread_only: bool = False,
        limit: int = 50,
        offset: int = 0
    ) -> List[NotificationResponse]:
        """Get user notifications"""
        try:
            query = db.query(Notification).filter(
                Notification.user_id == user_id,
                Notification.expires_at > datetime.utcnow()
            )
            
            if unread_only:
                query = query.filter(Notification.read_at.is_(None))
            
            notifications = query.order_by(
                desc(Notification.created_at)
            ).offset(offset).limit(limit).all()
            
            responses = []
            for notification in notifications:
                content = NotificationContent(
                    title=notification.title,
                    message=notification.message,
                    action_url=notification.action_url,
                    action_text=notification.action_text,
                    image_url=notification.image_url,
                    metadata=json.loads(notification.notification_metadata or "{}")
                )
                
                channels = [DeliveryChannel(c) for c in json.loads(notification.channels or "[]")]
                
                response = NotificationResponse(
                    notification_id=notification.id,
                    user_id=notification.user_id,
                    notification_type=NotificationType(notification.notification_type),
                    priority=NotificationPriority(notification.priority),
                    status=NotificationStatus(notification.status),
                    channels=channels,
                    content=content,
                    created_at=notification.created_at,
                    sent_at=notification.sent_at,
                    read_at=notification.read_at,
                    expires_at=notification.expires_at
                )
                
                responses.append(response)
            
            return responses
            
        except Exception as e:
            logger.error(f"Error getting user notifications: {str(e)}")
            return []
    
    async def mark_notification_read(
        self,
        notification_id: str,
        user_id: str,
        db: Session
    ) -> Dict[str, Any]:
        """Mark a notification as read"""
        try:
            notification = db.query(Notification).filter(
                Notification.id == notification_id,
                Notification.user_id == user_id
            ).first()
            
            if not notification:
                raise ValueError(f"Notification {notification_id} not found")
            
            if not notification.read_at:
                notification.read_at = datetime.utcnow()
                db.commit()
            
            return {
                "notification_id": notification_id,
                "status": "read",
                "read_at": notification.read_at
            }
            
        except Exception as e:
            logger.error(f"Error marking notification as read: {str(e)}")
            db.rollback()
            raise
    
    async def get_user_alerts(
        self,
        user_id: str,
        db: Session,
        active_only: bool = True
    ) -> List[PriceAlertResponse]:
        """Get user price alerts"""
        try:
            query = db.query(PriceAlert).filter(
                PriceAlert.user_id == user_id
            )
            
            if active_only:
                query = query.filter(
                    PriceAlert.is_active == True,
                    or_(
                        PriceAlert.expires_at.is_(None),
                        PriceAlert.expires_at > datetime.utcnow()
                    )
                )
            
            alerts = query.order_by(desc(PriceAlert.created_at)).all()
            
            responses = []
            for alert in alerts:
                # Get current price
                try:
                    current_price = await self.market_service.get_current_price(alert.symbol)
                except Exception:
                    current_price = 0
                
                response = PriceAlertResponse(
                    alert_id=alert.id,
                    user_id=alert.user_id,
                    symbol=alert.symbol,
                    condition=AlertCondition(alert.condition),
                    target_value=alert.target_value,
                    comparison_value=alert.comparison_value,
                    current_value=current_price,
                    is_active=alert.is_active,
                    triggered_at=alert.triggered_at,
                    created_at=alert.created_at,
                    expires_at=alert.expires_at,
                    notes=alert.notes
                )
                
                responses.append(response)
            
            return responses
            
        except Exception as e:
            logger.error(f"Error getting user alerts: {str(e)}")
            return []
    
    async def start_monitoring(self):
        """Start background monitoring tasks"""
        try:
            if not self._monitoring_task or self._monitoring_task.done():
                self._monitoring_task = asyncio.create_task(self._monitor_price_alerts())
            
            if not self._cleanup_task or self._cleanup_task.done():
                self._cleanup_task = asyncio.create_task(self._cleanup_expired_notifications())
            
            logger.info("Notification monitoring started")
            
        except Exception as e:
            logger.error(f"Error starting monitoring: {str(e)}")
    
    async def stop_monitoring(self):
        """Stop background monitoring tasks"""
        try:
            if self._monitoring_task and not self._monitoring_task.done():
                self._monitoring_task.cancel()
            
            if self._cleanup_task and not self._cleanup_task.done():
                self._cleanup_task.cancel()
            
            logger.info("Notification monitoring stopped")
            
        except Exception as e:
            logger.error(f"Error stopping monitoring: {str(e)}")
    
    async def _monitor_price_alerts(self):
        """Monitor price alerts and trigger notifications"""
        while True:
            try:
                # Get database session
                db = next(get_db())
                
                # Check each active alert
                triggered_alerts = []
                
                for alert_id, alert in self.active_alerts.items():
                    try:
                        # Get current price
                        current_price = await self.market_service.get_current_price(alert.symbol)
                        alert.current_value = current_price
                        
                        # Check if alert condition is met
                        if await self._check_alert_condition(alert, current_price):
                            # Trigger alert
                            await self._trigger_price_alert(alert, current_price, db)
                            triggered_alerts.append(alert_id)
                    
                    except Exception as e:
                        logger.error(f"Error checking alert {alert_id}: {str(e)}")
                
                # Remove triggered alerts
                for alert_id in triggered_alerts:
                    if alert_id in self.active_alerts:
                        del self.active_alerts[alert_id]
                
                db.close()
                
                # Wait before next check
                await asyncio.sleep(self.alert_check_interval)
                
            except Exception as e:
                logger.error(f"Error in price alert monitoring: {str(e)}")
                await asyncio.sleep(self.alert_check_interval)
    
    async def _cleanup_expired_notifications(self):
        """Clean up expired notifications"""
        while True:
            try:
                # Run cleanup every hour
                await asyncio.sleep(3600)
                
                db = next(get_db())
                
                # Delete expired notifications
                expired_count = db.query(Notification).filter(
                    Notification.expires_at < datetime.utcnow()
                ).delete()
                
                if expired_count > 0:
                    db.commit()
                    logger.info(f"Cleaned up {expired_count} expired notifications")
                
                db.close()
                
            except Exception as e:
                logger.error(f"Error in notification cleanup: {str(e)}")
    
    async def _check_alert_condition(
        self,
        alert: PriceAlertResponse,
        current_price: float
    ) -> bool:
        """Check if alert condition is met"""
        try:
            if alert.condition == AlertCondition.ABOVE:
                return current_price > alert.target_value
            
            elif alert.condition == AlertCondition.BELOW:
                return current_price < alert.target_value
            
            elif alert.condition == AlertCondition.CROSSES_ABOVE:
                # Would need previous price to implement properly
                return current_price > alert.target_value
            
            elif alert.condition == AlertCondition.CROSSES_BELOW:
                # Would need previous price to implement properly
                return current_price < alert.target_value
            
            elif alert.condition == AlertCondition.PERCENT_CHANGE:
                if alert.comparison_value:
                    change_percent = ((current_price - alert.comparison_value) / alert.comparison_value) * 100
                    return abs(change_percent) >= alert.target_value
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking alert condition: {str(e)}")
            return False
    
    async def _trigger_price_alert(
        self,
        alert: PriceAlertResponse,
        current_price: float,
        db: Session
    ):
        """Trigger a price alert notification"""
        try:
            # Update alert in database
            db_alert = db.query(PriceAlert).filter(
                PriceAlert.id == alert.alert_id
            ).first()
            
            if db_alert:
                db_alert.triggered_at = datetime.utcnow()
                db_alert.is_active = False  # Deactivate after triggering
                db.commit()
            
            # Create notification
            notification_request = NotificationRequest(
                user_id=alert.user_id,
                notification_type=NotificationType.PRICE_ALERT,
                priority=NotificationPriority.HIGH,
                channels=[DeliveryChannel.IN_APP, DeliveryChannel.PUSH],
                content={
                    "symbol": alert.symbol,
                    "condition": alert.condition.value,
                    "target_value": alert.target_value,
                    "current_price": current_price,
                    "alert_id": alert.alert_id
                }
            )
            
            await self.send_notification(notification_request, db)
            
            logger.info(
                f"Price alert triggered for {alert.symbol}: "
                f"{alert.condition.value} {alert.target_value}, current: {current_price}"
            )
            
        except Exception as e:
            logger.error(f"Error triggering price alert: {str(e)}")
    
    async def _check_rate_limits(self, user_id: str, db: Session) -> bool:
        """Check if user has exceeded notification rate limits"""
        try:
            hour_ago = datetime.utcnow() - timedelta(hours=1)
            
            recent_notifications = db.query(Notification).filter(
                Notification.user_id == user_id,
                Notification.created_at >= hour_ago
            ).count()
            
            return recent_notifications < self.max_notifications_per_hour
            
        except Exception as e:
            logger.error(f"Error checking rate limits: {str(e)}")
            return True  # Allow on error
    
    async def _filter_channels_by_preferences(
        self,
        user_id: str,
        notification_type: NotificationType,
        requested_channels: List[DeliveryChannel],
        db: Session
    ) -> List[DeliveryChannel]:
        """Filter channels based on user preferences"""
        try:
            # Get user preferences
            preference = db.query(NotificationPreference).filter(
                NotificationPreference.user_id == user_id,
                NotificationPreference.notification_type == notification_type.value
            ).first()
            
            if not preference or not preference.enabled:
                return []
            
            # Get preferred channels
            preferred_channels = [DeliveryChannel(c) for c in json.loads(preference.channels or "[]")]
            
            # Return intersection of requested and preferred channels
            return [c for c in requested_channels if c in preferred_channels]
            
        except Exception as e:
            logger.error(f"Error filtering channels by preferences: {str(e)}")
            return requested_channels  # Return all requested on error
    
    async def _process_notification_content(
        self,
        notification_type: NotificationType,
        content_data: Dict[str, Any]
    ) -> NotificationContent:
        """Process notification content using templates"""
        try:
            template = self.templates.get(notification_type)
            
            if template:
                # Replace template variables
                title = template.title_template
                message = template.message_template
                
                for var in template.variables:
                    if var in content_data:
                        title = title.replace(f"{{{var}}}", str(content_data[var]))
                        message = message.replace(f"{{{var}}}", str(content_data[var]))
                
                return NotificationContent(
                    title=title,
                    message=message,
                    action_url=content_data.get("action_url"),
                    action_text=content_data.get("action_text"),
                    image_url=content_data.get("image_url"),
                    metadata=content_data.get("metadata", {})
                )
            
            else:
                # Use raw content if no template
                return NotificationContent(
                    title=content_data.get("title", "Notification"),
                    message=content_data.get("message", ""),
                    action_url=content_data.get("action_url"),
                    action_text=content_data.get("action_text"),
                    image_url=content_data.get("image_url"),
                    metadata=content_data.get("metadata", {})
                )
            
        except Exception as e:
            logger.error(f"Error processing notification content: {str(e)}")
            return NotificationContent(
                title="Notification",
                message="An error occurred processing this notification"
            )
    
    async def _deliver_notification(
        self,
        notification: Notification,
        channels: List[DeliveryChannel],
        content: NotificationContent
    ) -> Dict[DeliveryChannel, bool]:
        """Deliver notification through specified channels"""
        results = {}
        
        for channel in channels:
            try:
                if channel == DeliveryChannel.IN_APP:
                    # In-app notifications are stored in database (already done)
                    results[channel] = True
                
                elif channel == DeliveryChannel.EMAIL:
                    # Email delivery (would integrate with email service)
                    results[channel] = await self._send_email_notification(
                        notification, content
                    )
                
                elif channel == DeliveryChannel.PUSH:
                    # Push notification delivery (would integrate with push service)
                    results[channel] = await self._send_push_notification(
                        notification, content
                    )
                
                elif channel == DeliveryChannel.SMS:
                    # SMS delivery (would integrate with SMS service)
                    results[channel] = await self._send_sms_notification(
                        notification, content
                    )
                
                else:
                    results[channel] = False
                    
            except Exception as e:
                logger.error(f"Error delivering notification via {channel.value}: {str(e)}")
                results[channel] = False
        
        return results
    
    async def _send_email_notification(
        self,
        notification: Notification,
        content: NotificationContent
    ) -> bool:
        """Send email notification (placeholder)"""
        # Would integrate with email service (SendGrid, AWS SES, etc.)
        logger.info(f"Email notification sent: {content.title}")
        return True
    
    async def _send_push_notification(
        self,
        notification: Notification,
        content: NotificationContent
    ) -> bool:
        """Send push notification (placeholder)"""
        # Would integrate with push service (Firebase, APNs, etc.)
        logger.info(f"Push notification sent: {content.title}")
        return True
    
    async def _send_sms_notification(
        self,
        notification: Notification,
        content: NotificationContent
    ) -> bool:
        """Send SMS notification (placeholder)"""
        # Would integrate with SMS service (Twilio, AWS SNS, etc.)
        logger.info(f"SMS notification sent: {content.title}")
        return True
    
    def _initialize_templates(self) -> Dict[NotificationType, NotificationTemplate]:
        """Initialize notification templates"""
        return {
            NotificationType.PRICE_ALERT: NotificationTemplate(
                template_id="price_alert",
                notification_type=NotificationType.PRICE_ALERT,
                title_template="Price Alert: {symbol}",
                message_template="{symbol} has {condition} {target_value}. Current price: {current_price}",
                default_priority=NotificationPriority.HIGH,
                default_channels=[DeliveryChannel.IN_APP, DeliveryChannel.PUSH],
                variables=["symbol", "condition", "target_value", "current_price"]
            ),
            NotificationType.TRADE_EXECUTION: NotificationTemplate(
                template_id="trade_execution",
                notification_type=NotificationType.TRADE_EXECUTION,
                title_template="Trade Executed: {symbol}",
                message_template="{side} {quantity} shares of {symbol} at {price}",
                default_priority=NotificationPriority.HIGH,
                default_channels=[DeliveryChannel.IN_APP, DeliveryChannel.EMAIL],
                variables=["symbol", "side", "quantity", "price"]
            ),
            NotificationType.PORTFOLIO_UPDATE: NotificationTemplate(
                template_id="portfolio_update",
                notification_type=NotificationType.PORTFOLIO_UPDATE,
                title_template="Portfolio Update",
                message_template="Your portfolio value has changed by {change_percent}% to {total_value}",
                default_priority=NotificationPriority.MEDIUM,
                default_channels=[DeliveryChannel.IN_APP],
                variables=["change_percent", "total_value"]
            )
        }
    
    def _generate_notification_id(self) -> str:
        """Generate unique notification ID"""
        import uuid
        return f"NOTIF_{uuid.uuid4().hex[:8].upper()}"
    
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID"""
        import uuid
        return f"ALERT_{uuid.uuid4().hex[:8].upper()}"

# Global notification system instance
notification_system = NotificationSystem()

def get_notification_system() -> NotificationSystem:
    """Get notification system instance"""
    return notification_system