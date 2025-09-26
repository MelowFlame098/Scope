"""Security utilities for FinScope application.

This module provides:
- Password hashing and verification
- JWT token generation and validation
- API key management
- Rate limiting
- Security headers
- Input sanitization
- CSRF protection
"""

import hashlib
import secrets
import hmac
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, timedelta
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi import HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import re
import bleach
import base64
import json
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import logging

from app.config.settings import get_settings
from app.core.exceptions import (
    AuthenticationException,
    AuthorizationException,
    SecurityException
)

logger = logging.getLogger(__name__)


class PasswordManager:
    """Password hashing and verification manager."""
    
    def __init__(self):
        """Initialize password manager."""
        self.pwd_context = CryptContext(
            schemes=["bcrypt"],
            deprecated="auto",
            bcrypt__rounds=12
        )
    
    def hash_password(self, password: str) -> str:
        """Hash a password.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password
        """
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash.
        
        Args:
            plain_password: Plain text password
            hashed_password: Hashed password
            
        Returns:
            True if password matches
        """
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def needs_update(self, hashed_password: str) -> bool:
        """Check if password hash needs update.
        
        Args:
            hashed_password: Hashed password
            
        Returns:
            True if hash needs update
        """
        return self.pwd_context.needs_update(hashed_password)
    
    def generate_password(self, length: int = 16) -> str:
        """Generate a secure random password.
        
        Args:
            length: Password length
            
        Returns:
            Generated password
        """
        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    def validate_password_strength(self, password: str) -> Dict[str, Any]:
        """Validate password strength.
        
        Args:
            password: Password to validate
            
        Returns:
            Validation result with score and feedback
        """
        score = 0
        feedback = []
        
        # Length check
        if len(password) >= 12:
            score += 2
        elif len(password) >= 8:
            score += 1
        else:
            feedback.append("Password should be at least 8 characters long")
        
        # Character variety checks
        if re.search(r'[a-z]', password):
            score += 1
        else:
            feedback.append("Password should contain lowercase letters")
        
        if re.search(r'[A-Z]', password):
            score += 1
        else:
            feedback.append("Password should contain uppercase letters")
        
        if re.search(r'\d', password):
            score += 1
        else:
            feedback.append("Password should contain numbers")
        
        if re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            score += 1
        else:
            feedback.append("Password should contain special characters")
        
        # Common patterns check
        common_patterns = ['123', 'abc', 'password', 'qwerty']
        if any(pattern in password.lower() for pattern in common_patterns):
            score -= 1
            feedback.append("Password should not contain common patterns")
        
        # Determine strength
        if score >= 5:
            strength = "strong"
        elif score >= 3:
            strength = "medium"
        else:
            strength = "weak"
        
        return {
            "score": max(0, score),
            "strength": strength,
            "feedback": feedback,
            "is_valid": score >= 3
        }


class JWTManager:
    """JWT token management."""
    
    def __init__(self):
        """Initialize JWT manager."""
        self.settings = get_settings()
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30
        self.refresh_token_expire_days = 7
    
    def create_access_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create access token.
        
        Args:
            data: Token payload data
            expires_delta: Custom expiration time
            
        Returns:
            JWT access token
        """
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        })
        
        return jwt.encode(to_encode, self.settings.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create refresh token.
        
        Args:
            data: Token payload data
            expires_delta: Custom expiration time
            
        Returns:
            JWT refresh token
        """
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        })
        
        return jwt.encode(to_encode, self.settings.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str, token_type: str = "access") -> Dict[str, Any]:
        """Verify and decode JWT token.
        
        Args:
            token: JWT token
            token_type: Expected token type
            
        Returns:
            Token payload
            
        Raises:
            AuthenticationException: If token is invalid
        """
        try:
            payload = jwt.decode(
                token,
                self.settings.secret_key,
                algorithms=[self.algorithm]
            )
            
            # Verify token type
            if payload.get("type") != token_type:
                raise AuthenticationException(f"Invalid token type. Expected {token_type}")
            
            return payload
        
        except JWTError as e:
            raise AuthenticationException(f"Invalid token: {str(e)}")
    
    def get_token_payload(self, token: str) -> Optional[Dict[str, Any]]:
        """Get token payload without verification (for debugging).
        
        Args:
            token: JWT token
            
        Returns:
            Token payload or None if invalid
        """
        try:
            return jwt.get_unverified_claims(token)
        except JWTError:
            return None
    
    def is_token_expired(self, token: str) -> bool:
        """Check if token is expired.
        
        Args:
            token: JWT token
            
        Returns:
            True if token is expired
        """
        try:
            payload = jwt.get_unverified_claims(token)
            exp = payload.get("exp")
            if exp:
                return datetime.utcnow() > datetime.fromtimestamp(exp)
            return True
        except JWTError:
            return True


class APIKeyManager:
    """API key management."""
    
    def __init__(self):
        """Initialize API key manager."""
        self.settings = get_settings()
    
    def generate_api_key(self, prefix: str = "fsk") -> str:
        """Generate a new API key.
        
        Args:
            prefix: Key prefix
            
        Returns:
            Generated API key
        """
        # Generate 32 bytes of random data
        key_bytes = secrets.token_bytes(32)
        # Encode as base64 and remove padding
        key_b64 = base64.urlsafe_b64encode(key_bytes).decode('utf-8').rstrip('=')
        return f"{prefix}_{key_b64}"
    
    def hash_api_key(self, api_key: str) -> str:
        """Hash an API key for storage.
        
        Args:
            api_key: API key to hash
            
        Returns:
            Hashed API key
        """
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    def verify_api_key(self, api_key: str, hashed_key: str) -> bool:
        """Verify an API key against its hash.
        
        Args:
            api_key: API key to verify
            hashed_key: Stored hash
            
        Returns:
            True if key matches
        """
        return hmac.compare_digest(
            self.hash_api_key(api_key),
            hashed_key
        )


class EncryptionManager:
    """Data encryption and decryption."""
    
    def __init__(self):
        """Initialize encryption manager."""
        self.settings = get_settings()
        self._fernet = None
    
    @property
    def fernet(self) -> Fernet:
        """Get Fernet encryption instance.
        
        Returns:
            Fernet instance
        """
        if self._fernet is None:
            # Derive key from secret
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'finscope_salt',  # In production, use a random salt
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(
                kdf.derive(self.settings.secret_key.encode())
            )
            self._fernet = Fernet(key)
        return self._fernet
    
    def encrypt(self, data: str) -> str:
        """Encrypt string data.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data (base64 encoded)
        """
        encrypted = self.fernet.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt string data.
        
        Args:
            encrypted_data: Encrypted data (base64 encoded)
            
        Returns:
            Decrypted data
            
        Raises:
            SecurityException: If decryption fails
        """
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted = self.fernet.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            raise SecurityException(f"Decryption failed: {str(e)}")
    
    def encrypt_dict(self, data: Dict[str, Any]) -> str:
        """Encrypt dictionary data.
        
        Args:
            data: Dictionary to encrypt
            
        Returns:
            Encrypted JSON string
        """
        json_str = json.dumps(data, sort_keys=True)
        return self.encrypt(json_str)
    
    def decrypt_dict(self, encrypted_data: str) -> Dict[str, Any]:
        """Decrypt dictionary data.
        
        Args:
            encrypted_data: Encrypted JSON string
            
        Returns:
            Decrypted dictionary
            
        Raises:
            SecurityException: If decryption fails
        """
        try:
            json_str = self.decrypt(encrypted_data)
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise SecurityException(f"Invalid encrypted data format: {str(e)}")


class InputSanitizer:
    """Input sanitization and validation."""
    
    def __init__(self):
        """Initialize input sanitizer."""
        self.allowed_tags = [
            'p', 'br', 'strong', 'em', 'u', 'ol', 'ul', 'li',
            'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote'
        ]
        self.allowed_attributes = {
            '*': ['class'],
            'a': ['href', 'title'],
            'img': ['src', 'alt', 'width', 'height']
        }
    
    def sanitize_html(self, html: str) -> str:
        """Sanitize HTML content.
        
        Args:
            html: HTML content to sanitize
            
        Returns:
            Sanitized HTML
        """
        return bleach.clean(
            html,
            tags=self.allowed_tags,
            attributes=self.allowed_attributes,
            strip=True
        )
    
    def sanitize_string(self, text: str, max_length: Optional[int] = None) -> str:
        """Sanitize string input.
        
        Args:
            text: Text to sanitize
            max_length: Maximum allowed length
            
        Returns:
            Sanitized text
        """
        # Remove null bytes and control characters
        sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Trim whitespace
        sanitized = sanitized.strip()
        
        # Limit length
        if max_length and len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized
    
    def validate_email(self, email: str) -> bool:
        """Validate email format.
        
        Args:
            email: Email address to validate
            
        Returns:
            True if email is valid
        """
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    def validate_username(self, username: str) -> bool:
        """Validate username format.
        
        Args:
            username: Username to validate
            
        Returns:
            True if username is valid
        """
        # Allow alphanumeric, underscore, hyphen, 3-30 characters
        pattern = r'^[a-zA-Z0-9_-]{3,30}$'
        return bool(re.match(pattern, username))
    
    def validate_phone(self, phone: str) -> bool:
        """Validate phone number format.
        
        Args:
            phone: Phone number to validate
            
        Returns:
            True if phone is valid
        """
        # Remove all non-digit characters
        digits_only = re.sub(r'\D', '', phone)
        # Check if it's 10-15 digits
        return 10 <= len(digits_only) <= 15


class CSRFProtection:
    """CSRF protection utilities."""
    
    def __init__(self):
        """Initialize CSRF protection."""
        self.settings = get_settings()
    
    def generate_csrf_token(self, session_id: str) -> str:
        """Generate CSRF token.
        
        Args:
            session_id: Session identifier
            
        Returns:
            CSRF token
        """
        # Create token with timestamp
        timestamp = str(int(datetime.utcnow().timestamp()))
        data = f"{session_id}:{timestamp}"
        
        # Create HMAC signature
        signature = hmac.new(
            self.settings.secret_key.encode(),
            data.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Combine data and signature
        token = f"{data}:{signature}"
        return base64.urlsafe_b64encode(token.encode()).decode()
    
    def verify_csrf_token(self, token: str, session_id: str, max_age: int = 3600) -> bool:
        """Verify CSRF token.
        
        Args:
            token: CSRF token to verify
            session_id: Session identifier
            max_age: Maximum token age in seconds
            
        Returns:
            True if token is valid
        """
        try:
            # Decode token
            decoded = base64.urlsafe_b64decode(token.encode()).decode()
            parts = decoded.split(':')
            
            if len(parts) != 3:
                return False
            
            token_session_id, timestamp, signature = parts
            
            # Verify session ID
            if token_session_id != session_id:
                return False
            
            # Verify timestamp
            token_time = int(timestamp)
            current_time = int(datetime.utcnow().timestamp())
            
            if current_time - token_time > max_age:
                return False
            
            # Verify signature
            data = f"{token_session_id}:{timestamp}"
            expected_signature = hmac.new(
                self.settings.secret_key.encode(),
                data.encode(),
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(signature, expected_signature)
        
        except Exception:
            return False


class SecurityHeaders:
    """Security headers management."""
    
    @staticmethod
    def get_security_headers() -> Dict[str, str]:
        """Get recommended security headers.
        
        Returns:
            Dictionary of security headers
        """
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "font-src 'self'; "
                "connect-src 'self'; "
                "frame-ancestors 'none';"
            ),
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": (
                "geolocation=(), "
                "microphone=(), "
                "camera=()"
            )
        }


# Global instances
password_manager = PasswordManager()
jwt_manager = JWTManager()
api_key_manager = APIKeyManager()
encryption_manager = EncryptionManager()
input_sanitizer = InputSanitizer()
csrf_protection = CSRFProtection()
security_headers = SecurityHeaders()


# Utility functions
def generate_secure_token(length: int = 32) -> str:
    """Generate a secure random token.
    
    Args:
        length: Token length in bytes
        
    Returns:
        Secure random token
    """
    return secrets.token_urlsafe(length)


def constant_time_compare(a: str, b: str) -> bool:
    """Compare two strings in constant time.
    
    Args:
        a: First string
        b: Second string
        
    Returns:
        True if strings are equal
    """
    return hmac.compare_digest(a, b)


def hash_data(data: str, salt: Optional[str] = None) -> str:
    """Hash data with optional salt.
    
    Args:
        data: Data to hash
        salt: Optional salt
        
    Returns:
        Hashed data
    """
    if salt:
        data = f"{data}{salt}"
    return hashlib.sha256(data.encode()).hexdigest()