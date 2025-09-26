"""Email service for FinScope with verification and notification capabilities.

This module provides:
- Email verification for user registration
- Password reset emails
- Trading notifications
- Portfolio alerts
- Multi-provider support (SMTP, SendGrid)
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import smtplib
import ssl
import secrets
import hashlib
from jinja2 import Environment, FileSystemLoader
from pathlib import Path
import aioredis
import asyncio
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from app.config.settings import get_settings
from app.core.logging_config import get_logger
from app.core.exceptions import EmailException
from app.models.user import User

logger = get_logger("email_service")
settings = get_settings()


class EmailTemplate(BaseModel):
    """Email template model."""
    subject: str
    html_content: str
    text_content: Optional[str] = None
    attachments: Optional[List[Dict[str, Any]]] = None


class EmailVerificationToken(BaseModel):
    """Email verification token model."""
    user_id: str
    email: EmailStr
    token: str
    expires_at: datetime
    created_at: datetime


class EmailService:
    """Enhanced email service with verification and notifications."""
    
    def __init__(self):
        self.settings = get_settings()
        self.redis_client = None
        self.template_env = Environment(
            loader=FileSystemLoader(Path(__file__).parent.parent / "templates" / "email")
        )
        
    async def initialize(self):
        """Initialize email service."""
        try:
            self.redis_client = aioredis.from_url(
                self.settings.redis.url,
                password=self.settings.redis.password,
                db=self.settings.redis.db
            )
            logger.info("Email service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize email service: {e}")
            raise EmailException(f"Email service initialization failed: {e}")
    
    def generate_verification_token(self, user_id: str, email: str) -> str:
        """Generate email verification token."""
        timestamp = str(int(datetime.utcnow().timestamp()))
        data = f"{user_id}:{email}:{timestamp}:{secrets.token_urlsafe(32)}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    async def store_verification_token(self, user_id: str, email: str, token: str, expires_in: int = 3600):
        """Store verification token in Redis."""
        try:
            key = f"email_verification:{token}"
            value = {
                "user_id": user_id,
                "email": email,
                "created_at": datetime.utcnow().isoformat()
            }
            await self.redis_client.setex(key, expires_in, str(value))
            logger.info(f"Verification token stored for user {user_id}")
        except Exception as e:
            logger.error(f"Failed to store verification token: {e}")
            raise EmailException(f"Token storage failed: {e}")
    
    async def verify_email_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify email verification token."""
        try:
            key = f"email_verification:{token}"
            value = await self.redis_client.get(key)
            if value:
                await self.redis_client.delete(key)  # One-time use
                return eval(value.decode())  # Convert string back to dict
            return None
        except Exception as e:
            logger.error(f"Failed to verify email token: {e}")
            return None
    
    def _get_smtp_connection(self):
        """Get SMTP connection."""
        try:
            context = ssl.create_default_context()
            server = smtplib.SMTP(self.settings.notifications.smtp_host, self.settings.notifications.smtp_port)
            server.starttls(context=context)
            server.login(self.settings.notifications.smtp_username, self.settings.notifications.smtp_password)
            return server
        except Exception as e:
            logger.error(f"Failed to create SMTP connection: {e}")
            raise EmailException(f"SMTP connection failed: {e}")
    
    def _render_template(self, template_name: str, context: Dict[str, Any]) -> EmailTemplate:
        """Render email template."""
        try:
            template = self.template_env.get_template(f"{template_name}.html")
            html_content = template.render(**context)
            
            # Try to get text version
            text_content = None
            try:
                text_template = self.template_env.get_template(f"{template_name}.txt")
                text_content = text_template.render(**context)
            except:
                pass
            
            return EmailTemplate(
                subject=context.get("subject", "FinScope Notification"),
                html_content=html_content,
                text_content=text_content
            )
        except Exception as e:
            logger.error(f"Failed to render template {template_name}: {e}")
            raise EmailException(f"Template rendering failed: {e}")
    
    async def send_email(self, to_email: str, template: EmailTemplate, from_email: Optional[str] = None):
        """Send email using configured provider."""
        try:
            if self.settings.notifications.email_backend == "sendgrid":
                await self._send_via_sendgrid(to_email, template, from_email)
            else:
                await self._send_via_smtp(to_email, template, from_email)
            
            logger.info(f"Email sent successfully to {to_email}")
        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {e}")
            raise EmailException(f"Email sending failed: {e}")
    
    async def _send_via_smtp(self, to_email: str, template: EmailTemplate, from_email: Optional[str] = None):
        """Send email via SMTP."""
        from_email = from_email or self.settings.notifications.smtp_username
        
        msg = MIMEMultipart("alternative")
        msg["Subject"] = template.subject
        msg["From"] = from_email
        msg["To"] = to_email
        
        # Add text and HTML parts
        if template.text_content:
            text_part = MIMEText(template.text_content, "plain")
            msg.attach(text_part)
        
        html_part = MIMEText(template.html_content, "html")
        msg.attach(html_part)
        
        # Add attachments if any
        if template.attachments:
            for attachment in template.attachments:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment["content"])
                encoders.encode_base64(part)
                part.add_header(
                    "Content-Disposition",
                    f"attachment; filename= {attachment['filename']}"
                )
                msg.attach(part)
        
        # Send email
        server = self._get_smtp_connection()
        server.send_message(msg)
        server.quit()
    
    async def _send_via_sendgrid(self, to_email: str, template: EmailTemplate, from_email: Optional[str] = None):
        """Send email via SendGrid."""
        # Implementation for SendGrid API
        # This would use the SendGrid Python SDK
        pass
    
    async def send_verification_email(self, user: User, verification_url: str):
        """Send email verification email."""
        try:
            context = {
                "subject": "Verify Your FinScope Account",
                "user_name": f"{user.first_name} {user.last_name}",
                "verification_url": verification_url,
                "app_name": self.settings.app_name,
                "support_email": "support@finscope.com"
            }
            
            template = self._render_template("verification", context)
            await self.send_email(user.email, template)
            
            logger.info(f"Verification email sent to {user.email}")
        except Exception as e:
            logger.error(f"Failed to send verification email: {e}")
            raise
    
    async def send_password_reset_email(self, user: User, reset_url: str):
        """Send password reset email."""
        try:
            context = {
                "subject": "Reset Your FinScope Password",
                "user_name": f"{user.first_name} {user.last_name}",
                "reset_url": reset_url,
                "app_name": self.settings.app_name,
                "support_email": "support@finscope.com"
            }
            
            template = self._render_template("password_reset", context)
            await self.send_email(user.email, template)
            
            logger.info(f"Password reset email sent to {user.email}")
        except Exception as e:
            logger.error(f"Failed to send password reset email: {e}")
            raise
    
    async def send_trading_alert(self, user: User, alert_data: Dict[str, Any]):
        """Send trading alert email."""
        try:
            context = {
                "subject": f"Trading Alert: {alert_data.get('title', 'Market Update')}",
                "user_name": f"{user.first_name} {user.last_name}",
                "alert_data": alert_data,
                "app_name": self.settings.app_name
            }
            
            template = self._render_template("trading_alert", context)
            await self.send_email(user.email, template)
            
            logger.info(f"Trading alert sent to {user.email}")
        except Exception as e:
            logger.error(f"Failed to send trading alert: {e}")
            raise
    
    async def send_portfolio_report(self, user: User, report_data: Dict[str, Any]):
        """Send portfolio performance report."""
        try:
            context = {
                "subject": "Your FinScope Portfolio Report",
                "user_name": f"{user.first_name} {user.last_name}",
                "report_data": report_data,
                "app_name": self.settings.app_name
            }
            
            template = self._render_template("portfolio_report", context)
            await self.send_email(user.email, template)
            
            logger.info(f"Portfolio report sent to {user.email}")
        except Exception as e:
            logger.error(f"Failed to send portfolio report: {e}")
            raise


# Global email service instance
email_service = EmailService()


async def get_email_service() -> EmailService:
    """Get email service instance."""
    if not email_service.redis_client:
        await email_service.initialize()
    return email_service