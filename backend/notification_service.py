import asyncio
import smtplib
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
import os
from dotenv import load_dotenv
import json
from dataclasses import dataclass
from enum import Enum

from database import get_db, SessionLocal
from db_models import User, Watchlist, Asset
from schemas import AssetResponse
from market_data import MarketDataService

load_dotenv()
logger = logging.getLogger(__name__)

class NotificationType(Enum):
    PRICE_ALERT = "price_alert"
    NEWS_ALERT = "news_alert"
    AI_INSIGHT = "ai_insight"
    PORTFOLIO_ALERT = "portfolio_alert"
    SYSTEM_ALERT = "system_alert"
    WELCOME = "welcome"
    VERIFICATION = "verification"

class NotificationChannel(Enum):
    EMAIL = "email"
    WEBSOCKET = "websocket"
    PUSH = "push"
    SMS = "sms"

@dataclass
class NotificationTemplate:
    subject: str
    html_body: str
    text_body: str
    notification_type: NotificationType

class NotificationService:
    def __init__(self):
        # Email configuration
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.smtp_username = os.getenv('SMTP_USERNAME')
        self.smtp_password = os.getenv('SMTP_PASSWORD')
        self.from_email = os.getenv('FROM_EMAIL', 'noreply@finscope.com')
        
        # Services
        self.market_service = MarketDataService()
        
        # Notification templates
        self.templates = self._load_templates()
        
        # Background task for checking alerts
        self.alert_check_interval = 60  # seconds
        self.background_task = None
        
    def _load_templates(self) -> Dict[NotificationType, NotificationTemplate]:
        """Load notification templates."""
        return {
            NotificationType.PRICE_ALERT: NotificationTemplate(
                subject="Price Alert: {symbol} - {alert_type}",
                html_body="""
                <html>
                <body>
                    <h2>Price Alert for {symbol}</h2>
                    <p>Your price alert has been triggered:</p>
                    <ul>
                        <li><strong>Symbol:</strong> {symbol}</li>
                        <li><strong>Current Price:</strong> ${current_price}</li>
                        <li><strong>Alert Type:</strong> {alert_type}</li>
                        <li><strong>Target Price:</strong> ${target_price}</li>
                        <li><strong>Time:</strong> {timestamp}</li>
                    </ul>
                    <p>Visit <a href="https://finscope.com">FinScope</a> to view more details.</p>
                </body>
                </html>
                """,
                text_body="""
                Price Alert for {symbol}
                
                Your price alert has been triggered:
                - Symbol: {symbol}
                - Current Price: ${current_price}
                - Alert Type: {alert_type}
                - Target Price: ${target_price}
                - Time: {timestamp}
                
                Visit https://finscope.com to view more details.
                """,
                notification_type=NotificationType.PRICE_ALERT
            ),
            NotificationType.AI_INSIGHT: NotificationTemplate(
                subject="AI Insight: {symbol} - {insight_type}",
                html_body="""
                <html>
                <body>
                    <h2>New AI Insight for {symbol}</h2>
                    <p><strong>Insight Type:</strong> {insight_type}</p>
                    <p><strong>Title:</strong> {title}</p>
                    <div>
                        <h3>Analysis:</h3>
                        <p>{content}</p>
                    </div>
                    <p><strong>Confidence Score:</strong> {confidence_score}%</p>
                    <p><strong>Risk Level:</strong> {risk_level}</p>
                    <p>Visit <a href="https://finscope.com">FinScope</a> to view the full analysis.</p>
                </body>
                </html>
                """,
                text_body="""
                New AI Insight for {symbol}
                
                Insight Type: {insight_type}
                Title: {title}
                
                Analysis:
                {content}
                
                Confidence Score: {confidence_score}%
                Risk Level: {risk_level}
                
                Visit https://finscope.com to view the full analysis.
                """,
                notification_type=NotificationType.AI_INSIGHT
            ),
            NotificationType.WELCOME: NotificationTemplate(
                subject="Welcome to FinScope!",
                html_body="""
                <html>
                <body>
                    <h2>Welcome to FinScope, {username}!</h2>
                    <p>Thank you for joining our AI-powered financial analysis platform.</p>
                    <h3>Getting Started:</h3>
                    <ul>
                        <li>Add assets to your watchlist</li>
                        <li>Set up price alerts</li>
                        <li>Explore AI-powered insights</li>
                        <li>Join our community forum</li>
                    </ul>
                    <p>If you have any questions, feel free to reach out to our support team.</p>
                    <p>Happy trading!</p>
                    <p>The FinScope Team</p>
                </body>
                </html>
                """,
                text_body="""
                Welcome to FinScope, {username}!
                
                Thank you for joining our AI-powered financial analysis platform.
                
                Getting Started:
                - Add assets to your watchlist
                - Set up price alerts
                - Explore AI-powered insights
                - Join our community forum
                
                If you have any questions, feel free to reach out to our support team.
                
                Happy trading!
                The FinScope Team
                """,
                notification_type=NotificationType.WELCOME
            )
        }
    
    async def send_notification(
        self,
        user_id: str,
        notification_type: NotificationType,
        channels: List[NotificationChannel],
        data: Dict[str, Any],
        db: Session = None
    ) -> bool:
        """Send notification through specified channels."""
        if db is None:
            db = SessionLocal()
            
        try:
            # Get user information
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                logger.error(f"User not found: {user_id}")
                return False
            
            success = True
            
            # Send through each channel
            for channel in channels:
                try:
                    if channel == NotificationChannel.EMAIL:
                        await self._send_email_notification(user, notification_type, data)
                    elif channel == NotificationChannel.WEBSOCKET:
                        await self._send_websocket_notification(user_id, notification_type, data)
                    elif channel == NotificationChannel.PUSH:
                        await self._send_push_notification(user_id, notification_type, data)
                    elif channel == NotificationChannel.SMS:
                        await self._send_sms_notification(user, notification_type, data)
                except Exception as e:
                    logger.error(f"Failed to send {channel.value} notification to {user_id}: {e}")
                    success = False
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending notification to {user_id}: {e}")
            return False
        finally:
            if db:
                db.close()
    
    async def _send_email_notification(
        self,
        user: User,
        notification_type: NotificationType,
        data: Dict[str, Any]
    ):
        """Send email notification."""
        if not self.smtp_username or not self.smtp_password:
            logger.warning("SMTP credentials not configured")
            return
            
        template = self.templates.get(notification_type)
        if not template:
            logger.error(f"Template not found for {notification_type}")
            return
        
        try:
            # Format template with data
            subject = template.subject.format(**data)
            html_body = template.html_body.format(**data)
            text_body = template.text_body.format(**data)
            
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.from_email
            msg['To'] = user.email
            
            # Add text and HTML parts
            text_part = MIMEText(text_body, 'plain')
            html_part = MIMEText(html_body, 'html')
            
            msg.attach(text_part)
            msg.attach(html_part)
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)
            
            logger.info(f"Email sent to {user.email} for {notification_type}")
            
        except Exception as e:
            logger.error(f"Failed to send email to {user.email}: {e}")
            raise
    
    async def _send_websocket_notification(
        self,
        user_id: str,
        notification_type: NotificationType,
        data: Dict[str, Any]
    ):
        """Send WebSocket notification."""
        try:
            # Import here to avoid circular imports
            from websocket_manager import websocket_manager
            
            message = {
                "type": "notification",
                "notification_type": notification_type.value,
                "data": data,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await websocket_manager.broadcast_to_user(user_id, message)
            logger.info(f"WebSocket notification sent to {user_id} for {notification_type}")
            
        except Exception as e:
            logger.error(f"Failed to send WebSocket notification to {user_id}: {e}")
            raise
    
    async def _send_push_notification(
        self,
        user_id: str,
        notification_type: NotificationType,
        data: Dict[str, Any]
    ):
        """Send push notification (placeholder for future implementation)."""
        # TODO: Implement push notifications using FCM or similar service
        logger.info(f"Push notification would be sent to {user_id} for {notification_type}")
    
    async def _send_sms_notification(
        self,
        user: User,
        notification_type: NotificationType,
        data: Dict[str, Any]
    ):
        """Send SMS notification (placeholder for future implementation)."""
        # TODO: Implement SMS notifications using Twilio or similar service
        logger.info(f"SMS notification would be sent to {user.email} for {notification_type}")
    
    async def check_price_alerts(self):
        """Check and trigger price alerts."""
        db = SessionLocal()
        try:
            # Get all active watchlists with price alerts
            watchlists = db.query(Watchlist).filter(
                Watchlist.price_alert_enabled == True,
                or_(
                    Watchlist.price_alert_above.isnot(None),
                    Watchlist.price_alert_below.isnot(None)
                )
            ).all()
            
            for watchlist in watchlists:
                try:
                    # Get current asset price
                    asset_data = await self.market_service.get_asset_details(watchlist.symbol)
                    if not asset_data or not asset_data.current_price:
                        continue
                    
                    current_price = asset_data.current_price
                    alert_triggered = False
                    alert_type = ""
                    target_price = 0
                    
                    # Check price above alert
                    if (watchlist.price_alert_above and 
                        current_price >= watchlist.price_alert_above):
                        alert_triggered = True
                        alert_type = "Price Above Target"
                        target_price = watchlist.price_alert_above
                    
                    # Check price below alert
                    elif (watchlist.price_alert_below and 
                          current_price <= watchlist.price_alert_below):
                        alert_triggered = True
                        alert_type = "Price Below Target"
                        target_price = watchlist.price_alert_below
                    
                    if alert_triggered:
                        # Send notification
                        notification_data = {
                            "symbol": watchlist.symbol,
                            "current_price": current_price,
                            "alert_type": alert_type,
                            "target_price": target_price,
                            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
                        }
                        
                        await self.send_notification(
                            user_id=watchlist.user_id,
                            notification_type=NotificationType.PRICE_ALERT,
                            channels=[NotificationChannel.EMAIL, NotificationChannel.WEBSOCKET],
                            data=notification_data,
                            db=db
                        )
                        
                        # Disable the alert to prevent spam (user can re-enable)
                        if alert_type == "Price Above Target":
                            watchlist.price_alert_above = None
                        else:
                            watchlist.price_alert_below = None
                        
                        db.commit()
                        
                        logger.info(f"Price alert triggered for {watchlist.symbol} - {alert_type}")
                
                except Exception as e:
                    logger.error(f"Error checking price alert for watchlist {watchlist.id}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error in price alerts check: {e}")
        finally:
            db.close()
    
    async def send_welcome_notification(self, user_id: str):
        """Send welcome notification to new user."""
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if user:
                notification_data = {
                    "username": user.username or user.full_name or "User"
                }
                
                await self.send_notification(
                    user_id=user_id,
                    notification_type=NotificationType.WELCOME,
                    channels=[NotificationChannel.EMAIL],
                    data=notification_data,
                    db=db
                )
        finally:
            db.close()
    
    async def send_ai_insight_notification(
        self,
        user_ids: List[str],
        insight_data: Dict[str, Any]
    ):
        """Send AI insight notification to users."""
        for user_id in user_ids:
            try:
                await self.send_notification(
                    user_id=user_id,
                    notification_type=NotificationType.AI_INSIGHT,
                    channels=[NotificationChannel.WEBSOCKET, NotificationChannel.EMAIL],
                    data=insight_data
                )
            except Exception as e:
                logger.error(f"Failed to send AI insight notification to {user_id}: {e}")
    
    def start_background_tasks(self):
        """Start background tasks for checking alerts."""
        if self.background_task is None or self.background_task.done():
            self.background_task = asyncio.create_task(self._alert_check_loop())
            logger.info("Notification service background tasks started")
    
    async def _alert_check_loop(self):
        """Background loop for checking alerts."""
        while True:
            try:
                await self.check_price_alerts()
                await asyncio.sleep(self.alert_check_interval)
            except Exception as e:
                logger.error(f"Error in alert check loop: {e}")
                await asyncio.sleep(60)  # Wait a minute before retrying
    
    def stop_background_tasks(self):
        """Stop background tasks."""
        if self.background_task and not self.background_task.done():
            self.background_task.cancel()
            logger.info("Notification service background tasks stopped")

# Global notification service instance
notification_service = NotificationService()