"""CAPTCHA service for FinScope with multiple provider support.

This module provides:
- reCAPTCHA v2 and v3 integration
- hCaptcha support
- Custom image CAPTCHA generation
- Rate limiting and abuse prevention
"""

from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import httpx
import secrets
import string
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import random
import math
from pydantic import BaseModel
import aioredis
from sqlalchemy.orm import Session

from app.config.settings import get_settings
from app.core.logging_config import get_logger
from app.core.exceptions import CaptchaException

logger = get_logger("captcha_service")
settings = get_settings()


class CaptchaType(str):
    """CAPTCHA types."""
    RECAPTCHA_V2 = "recaptcha_v2"
    RECAPTCHA_V3 = "recaptcha_v3"
    HCAPTCHA = "hcaptcha"
    IMAGE = "image"
    MATH = "math"


class CaptchaChallenge(BaseModel):
    """CAPTCHA challenge model."""
    challenge_id: str
    challenge_type: str
    challenge_data: Dict[str, Any]
    expires_at: datetime
    attempts: int = 0
    max_attempts: int = 3


class CaptchaResponse(BaseModel):
    """CAPTCHA response model."""
    success: bool
    challenge_id: Optional[str] = None
    challenge_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    score: Optional[float] = None  # For reCAPTCHA v3


class CaptchaService:
    """Enhanced CAPTCHA service with multiple providers."""
    
    def __init__(self):
        self.settings = get_settings()
        self.redis_client = None
        self.recaptcha_secret = None
        self.hcaptcha_secret = None
        
    async def initialize(self):
        """Initialize CAPTCHA service."""
        try:
            self.redis_client = aioredis.from_url(
                self.settings.redis.url,
                password=self.settings.redis.password,
                db=self.settings.redis.db
            )
            
            # Load CAPTCHA secrets from environment
            import os
            self.recaptcha_secret = os.getenv("RECAPTCHA_SECRET_KEY")
            self.hcaptcha_secret = os.getenv("HCAPTCHA_SECRET_KEY")
            
            logger.info("CAPTCHA service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize CAPTCHA service: {e}")
            raise CaptchaException(f"CAPTCHA service initialization failed: {e}")
    
    def generate_challenge_id(self) -> str:
        """Generate unique challenge ID."""
        return secrets.token_urlsafe(32)
    
    async def store_challenge(self, challenge: CaptchaChallenge, expires_in: int = 300):
        """Store CAPTCHA challenge in Redis."""
        try:
            key = f"captcha_challenge:{challenge.challenge_id}"
            value = challenge.json()
            await self.redis_client.setex(key, expires_in, value)
            logger.debug(f"CAPTCHA challenge stored: {challenge.challenge_id}")
        except Exception as e:
            logger.error(f"Failed to store CAPTCHA challenge: {e}")
            raise CaptchaException(f"Challenge storage failed: {e}")
    
    async def get_challenge(self, challenge_id: str) -> Optional[CaptchaChallenge]:
        """Get CAPTCHA challenge from Redis."""
        try:
            key = f"captcha_challenge:{challenge_id}"
            value = await self.redis_client.get(key)
            if value:
                return CaptchaChallenge.parse_raw(value)
            return None
        except Exception as e:
            logger.error(f"Failed to get CAPTCHA challenge: {e}")
            return None
    
    async def delete_challenge(self, challenge_id: str):
        """Delete CAPTCHA challenge from Redis."""
        try:
            key = f"captcha_challenge:{challenge_id}"
            await self.redis_client.delete(key)
        except Exception as e:
            logger.error(f"Failed to delete CAPTCHA challenge: {e}")
    
    async def verify_recaptcha_v2(self, response_token: str, remote_ip: Optional[str] = None) -> CaptchaResponse:
        """Verify reCAPTCHA v2 response."""
        if not self.recaptcha_secret:
            return CaptchaResponse(success=False, error_message="reCAPTCHA not configured")
        
        try:
            async with httpx.AsyncClient() as client:
                data = {
                    "secret": self.recaptcha_secret,
                    "response": response_token
                }
                if remote_ip:
                    data["remoteip"] = remote_ip
                
                response = await client.post(
                    "https://www.google.com/recaptcha/api/siteverify",
                    data=data
                )
                result = response.json()
                
                return CaptchaResponse(
                    success=result.get("success", False),
                    error_message=None if result.get("success") else "reCAPTCHA verification failed"
                )
        except Exception as e:
            logger.error(f"reCAPTCHA v2 verification failed: {e}")
            return CaptchaResponse(success=False, error_message="Verification service unavailable")
    
    async def verify_recaptcha_v3(self, response_token: str, action: str, remote_ip: Optional[str] = None) -> CaptchaResponse:
        """Verify reCAPTCHA v3 response."""
        if not self.recaptcha_secret:
            return CaptchaResponse(success=False, error_message="reCAPTCHA not configured")
        
        try:
            async with httpx.AsyncClient() as client:
                data = {
                    "secret": self.recaptcha_secret,
                    "response": response_token
                }
                if remote_ip:
                    data["remoteip"] = remote_ip
                
                response = await client.post(
                    "https://www.google.com/recaptcha/api/siteverify",
                    data=data
                )
                result = response.json()
                
                success = result.get("success", False)
                score = result.get("score", 0.0)
                expected_action = result.get("action", "")
                
                # Check if action matches and score is above threshold
                if success and expected_action == action and score >= 0.5:
                    return CaptchaResponse(success=True, score=score)
                else:
                    return CaptchaResponse(
                        success=False,
                        score=score,
                        error_message="reCAPTCHA verification failed or low score"
                    )
        except Exception as e:
            logger.error(f"reCAPTCHA v3 verification failed: {e}")
            return CaptchaResponse(success=False, error_message="Verification service unavailable")
    
    async def verify_hcaptcha(self, response_token: str, remote_ip: Optional[str] = None) -> CaptchaResponse:
        """Verify hCaptcha response."""
        if not self.hcaptcha_secret:
            return CaptchaResponse(success=False, error_message="hCaptcha not configured")
        
        try:
            async with httpx.AsyncClient() as client:
                data = {
                    "secret": self.hcaptcha_secret,
                    "response": response_token
                }
                if remote_ip:
                    data["remoteip"] = remote_ip
                
                response = await client.post(
                    "https://hcaptcha.com/siteverify",
                    data=data
                )
                result = response.json()
                
                return CaptchaResponse(
                    success=result.get("success", False),
                    error_message=None if result.get("success") else "hCaptcha verification failed"
                )
        except Exception as e:
            logger.error(f"hCaptcha verification failed: {e}")
            return CaptchaResponse(success=False, error_message="Verification service unavailable")
    
    def generate_image_captcha(self, length: int = 6) -> Tuple[str, str]:
        """Generate image CAPTCHA."""
        try:
            # Generate random text
            characters = string.ascii_uppercase + string.digits
            text = ''.join(random.choice(characters) for _ in range(length))
            
            # Create image
            width, height = 200, 80
            image = Image.new('RGB', (width, height), color='white')
            draw = ImageDraw.Draw(image)
            
            # Try to load a font, fallback to default if not available
            try:
                font = ImageFont.truetype("arial.ttf", 36)
            except:
                font = ImageFont.load_default()
            
            # Add noise lines
            for _ in range(5):
                x1, y1 = random.randint(0, width), random.randint(0, height)
                x2, y2 = random.randint(0, width), random.randint(0, height)
                draw.line([(x1, y1), (x2, y2)], fill='gray', width=2)
            
            # Add text with distortion
            for i, char in enumerate(text):
                x = 20 + i * 25 + random.randint(-5, 5)
                y = 20 + random.randint(-10, 10)
                angle = random.randint(-30, 30)
                
                # Create temporary image for character rotation
                char_img = Image.new('RGBA', (40, 60), (255, 255, 255, 0))
                char_draw = ImageDraw.Draw(char_img)
                char_draw.text((10, 10), char, font=font, fill='black')
                
                # Rotate character
                rotated = char_img.rotate(angle, expand=1)
                image.paste(rotated, (x, y), rotated)
            
            # Add noise dots
            for _ in range(100):
                x, y = random.randint(0, width), random.randint(0, height)
                draw.point((x, y), fill='gray')
            
            # Convert to base64
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            image_data = base64.b64encode(buffer.getvalue()).decode()
            
            return text, f"data:image/png;base64,{image_data}"
        except Exception as e:
            logger.error(f"Failed to generate image CAPTCHA: {e}")
            raise CaptchaException(f"Image CAPTCHA generation failed: {e}")
    
    def generate_math_captcha(self) -> Tuple[str, str]:
        """Generate math CAPTCHA."""
        try:
            # Generate simple math problem
            num1 = random.randint(1, 20)
            num2 = random.randint(1, 20)
            operation = random.choice(['+', '-', '*'])
            
            if operation == '+':
                answer = num1 + num2
                question = f"{num1} + {num2} = ?"
            elif operation == '-':
                if num1 < num2:
                    num1, num2 = num2, num1
                answer = num1 - num2
                question = f"{num1} - {num2} = ?"
            else:  # multiplication
                num1 = random.randint(1, 10)
                num2 = random.randint(1, 10)
                answer = num1 * num2
                question = f"{num1} × {num2} = ?"
            
            return str(answer), question
        except Exception as e:
            logger.error(f"Failed to generate math CAPTCHA: {e}")
            raise CaptchaException(f"Math CAPTCHA generation failed: {e}")
    
    async def create_challenge(self, challenge_type: str) -> CaptchaResponse:
        """Create new CAPTCHA challenge."""
        try:
            challenge_id = self.generate_challenge_id()
            
            if challenge_type == CaptchaType.IMAGE:
                answer, image_data = self.generate_image_captcha()
                challenge_data = {
                    "image": image_data,
                    "answer": answer.lower()
                }
            elif challenge_type == CaptchaType.MATH:
                answer, question = self.generate_math_captcha()
                challenge_data = {
                    "question": question,
                    "answer": answer
                }
            else:
                return CaptchaResponse(
                    success=False,
                    error_message="Unsupported challenge type"
                )
            
            challenge = CaptchaChallenge(
                challenge_id=challenge_id,
                challenge_type=challenge_type,
                challenge_data=challenge_data,
                expires_at=datetime.utcnow() + timedelta(minutes=5)
            )
            
            await self.store_challenge(challenge)
            
            # Remove answer from response data
            response_data = challenge_data.copy()
            response_data.pop("answer", None)
            
            return CaptchaResponse(
                success=True,
                challenge_id=challenge_id,
                challenge_data=response_data
            )
        except Exception as e:
            logger.error(f"Failed to create CAPTCHA challenge: {e}")
            return CaptchaResponse(
                success=False,
                error_message="Failed to create challenge"
            )
    
    async def verify_challenge(self, challenge_id: str, user_response: str) -> CaptchaResponse:
        """Verify CAPTCHA challenge response."""
        try:
            challenge = await self.get_challenge(challenge_id)
            if not challenge:
                return CaptchaResponse(
                    success=False,
                    error_message="Challenge not found or expired"
                )
            
            # Check if challenge has expired
            if datetime.utcnow() > challenge.expires_at:
                await self.delete_challenge(challenge_id)
                return CaptchaResponse(
                    success=False,
                    error_message="Challenge expired"
                )
            
            # Check attempt limit
            if challenge.attempts >= challenge.max_attempts:
                await self.delete_challenge(challenge_id)
                return CaptchaResponse(
                    success=False,
                    error_message="Too many attempts"
                )
            
            # Verify response
            correct_answer = challenge.challenge_data.get("answer", "")
            user_answer = user_response.strip().lower()
            
            if user_answer == correct_answer.lower():
                await self.delete_challenge(challenge_id)
                return CaptchaResponse(success=True)
            else:
                # Increment attempts
                challenge.attempts += 1
                await self.store_challenge(challenge)
                
                return CaptchaResponse(
                    success=False,
                    error_message="Incorrect answer"
                )
        except Exception as e:
            logger.error(f"Failed to verify CAPTCHA challenge: {e}")
            return CaptchaResponse(
                success=False,
                error_message="Verification failed"
            )
    
    async def verify_captcha(self, captcha_type: str, **kwargs) -> CaptchaResponse:
        """Verify CAPTCHA based on type."""
        try:
            if captcha_type == CaptchaType.RECAPTCHA_V2:
                return await self.verify_recaptcha_v2(
                    kwargs.get("response_token"),
                    kwargs.get("remote_ip")
                )
            elif captcha_type == CaptchaType.RECAPTCHA_V3:
                return await self.verify_recaptcha_v3(
                    kwargs.get("response_token"),
                    kwargs.get("action", "submit"),
                    kwargs.get("remote_ip")
                )
            elif captcha_type == CaptchaType.HCAPTCHA:
                return await self.verify_hcaptcha(
                    kwargs.get("response_token"),
                    kwargs.get("remote_ip")
                )
            elif captcha_type in [CaptchaType.IMAGE, CaptchaType.MATH]:
                return await self.verify_challenge(
                    kwargs.get("challenge_id"),
                    kwargs.get("user_response")
                )
            else:
                return CaptchaResponse(
                    success=False,
                    error_message="Unsupported CAPTCHA type"
                )
        except Exception as e:
            logger.error(f"CAPTCHA verification failed: {e}")
            return CaptchaResponse(
                success=False,
                error_message="Verification service error"
            )


# Global CAPTCHA service instance
captcha_service = CaptchaService()


async def get_captcha_service() -> CaptchaService:
    """Get CAPTCHA service instance."""
    if not captcha_service.redis_client:
        await captcha_service.initialize()
    return captcha_service