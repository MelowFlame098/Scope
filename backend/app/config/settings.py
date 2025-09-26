"""Centralized configuration management for FinScope.

This module provides a comprehensive configuration system that supports:
- Environment-based configuration
- Feature flags for optional components
- Service-specific settings
- Security configurations
- Database and caching settings
"""

from typing import Optional, List, Dict, Any
from pydantic import Field, validator
from pydantic_settings import BaseSettings
from enum import Enum
import os
from pathlib import Path


class Environment(str, Enum):
    """Application environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    url: str = Field(..., env="DATABASE_URL")
    echo: bool = Field(False, env="DATABASE_ECHO")
    pool_size: int = Field(10, env="DATABASE_POOL_SIZE")
    max_overflow: int = Field(20, env="DATABASE_MAX_OVERFLOW")
    pool_timeout: int = Field(30, env="DATABASE_POOL_TIMEOUT")
    pool_recycle: int = Field(3600, env="DATABASE_POOL_RECYCLE")
    
    class Config:
        env_prefix = "DATABASE_"


class RedisSettings(BaseSettings):
    """Redis configuration settings."""
    url: str = Field("redis://localhost:6379/0", env="REDIS_URL")
    password: Optional[str] = Field(None, env="REDIS_PASSWORD")
    db: int = Field(0, env="REDIS_DB")
    max_connections: int = Field(20, env="REDIS_MAX_CONNECTIONS")
    socket_timeout: int = Field(5, env="REDIS_SOCKET_TIMEOUT")
    socket_connect_timeout: int = Field(5, env="REDIS_SOCKET_CONNECT_TIMEOUT")
    
    class Config:
        env_prefix = "REDIS_"


class SecuritySettings(BaseSettings):
    """Security configuration settings."""
    secret_key: str = Field(..., env="SECRET_KEY")
    algorithm: str = Field("HS256", env="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    refresh_token_expire_days: int = Field(7, env="REFRESH_TOKEN_EXPIRE_DAYS")
    password_min_length: int = Field(8, env="PASSWORD_MIN_LENGTH")
    max_login_attempts: int = Field(5, env="MAX_LOGIN_ATTEMPTS")
    lockout_duration_minutes: int = Field(15, env="LOCKOUT_DURATION_MINUTES")
    
    class Config:
        env_prefix = "SECURITY_"


class AISettings(BaseSettings):
    """AI/ML configuration settings."""
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4", env="OPENAI_MODEL")
    max_tokens: int = Field(2048, env="AI_MAX_TOKENS")
    temperature: float = Field(0.7, env="AI_TEMPERATURE")
    enable_langchain: bool = Field(False, env="ENABLE_LANGCHAIN")
    enable_autonomous_trading: bool = Field(False, env="ENABLE_AUTONOMOUS_TRADING")
    risk_tolerance: float = Field(0.05, env="AI_RISK_TOLERANCE")
    max_position_size: float = Field(0.1, env="AI_MAX_POSITION_SIZE")
    
    class Config:
        env_prefix = "AI_"


class DeFiSettings(BaseSettings):
    """DeFi configuration settings."""
    ethereum_rpc_url: Optional[str] = Field(None, env="ETHEREUM_RPC_URL")
    polygon_rpc_url: Optional[str] = Field(None, env="POLYGON_RPC_URL")
    bsc_rpc_url: Optional[str] = Field(None, env="BSC_RPC_URL")
    private_key: Optional[str] = Field(None, env="WALLET_PRIVATE_KEY")
    gas_limit: int = Field(200000, env="GAS_LIMIT")
    gas_price_gwei: int = Field(20, env="GAS_PRICE_GWEI")
    slippage_tolerance: float = Field(0.01, env="SLIPPAGE_TOLERANCE")
    enable_cross_chain: bool = Field(False, env="ENABLE_CROSS_CHAIN")
    
    class Config:
        env_prefix = "DEFI_"


class MarketDataSettings(BaseSettings):
    """Market data configuration settings."""
    alpha_vantage_api_key: Optional[str] = Field(None, env="ALPHA_VANTAGE_API_KEY")
    finnhub_api_key: Optional[str] = Field(None, env="FINNHUB_API_KEY")
    polygon_api_key: Optional[str] = Field(None, env="POLYGON_API_KEY")
    iex_api_key: Optional[str] = Field(None, env="IEX_API_KEY")
    cache_ttl_seconds: int = Field(300, env="MARKET_DATA_CACHE_TTL")
    rate_limit_per_minute: int = Field(60, env="MARKET_DATA_RATE_LIMIT")
    enable_real_time: bool = Field(True, env="ENABLE_REAL_TIME_DATA")
    
    class Config:
        env_prefix = "MARKET_DATA_"


class NotificationSettings(BaseSettings):
    """Notification configuration settings."""
    email_backend: str = Field("smtp", env="EMAIL_BACKEND")
    smtp_host: Optional[str] = Field(None, env="SMTP_HOST")
    smtp_port: int = Field(587, env="SMTP_PORT")
    smtp_username: Optional[str] = Field(None, env="SMTP_USERNAME")
    smtp_password: Optional[str] = Field(None, env="SMTP_PASSWORD")
    smtp_use_tls: bool = Field(True, env="SMTP_USE_TLS")
    sendgrid_api_key: Optional[str] = Field(None, env="SENDGRID_API_KEY")
    slack_webhook_url: Optional[str] = Field(None, env="SLACK_WEBHOOK_URL")
    enable_push_notifications: bool = Field(True, env="ENABLE_PUSH_NOTIFICATIONS")
    
    class Config:
        env_prefix = "NOTIFICATION_"


class MonitoringSettings(BaseSettings):
    """Monitoring and observability settings."""
    enable_metrics: bool = Field(True, env="ENABLE_METRICS")
    enable_tracing: bool = Field(False, env="ENABLE_TRACING")
    sentry_dsn: Optional[str] = Field(None, env="SENTRY_DSN")
    prometheus_port: int = Field(8001, env="PROMETHEUS_PORT")
    jaeger_endpoint: Optional[str] = Field(None, env="JAEGER_ENDPOINT")
    log_level: LogLevel = Field(LogLevel.INFO, env="LOG_LEVEL")
    
    class Config:
        env_prefix = "MONITORING_"


class FeatureFlags(BaseSettings):
    """Feature flags for optional components."""
    enable_ai_features: bool = Field(True, env="ENABLE_AI_FEATURES")
    enable_defi_features: bool = Field(False, env="ENABLE_DEFI_FEATURES")
    enable_enterprise_features: bool = Field(False, env="ENABLE_ENTERPRISE_FEATURES")
    enable_social_trading: bool = Field(True, env="ENABLE_SOCIAL_TRADING")
    enable_paper_trading: bool = Field(True, env="ENABLE_PAPER_TRADING")
    enable_mobile_api: bool = Field(True, env="ENABLE_MOBILE_API")
    enable_advanced_charting: bool = Field(True, env="ENABLE_ADVANCED_CHARTING")
    enable_news_sentiment: bool = Field(True, env="ENABLE_NEWS_SENTIMENT")
    enable_portfolio_analytics: bool = Field(True, env="ENABLE_PORTFOLIO_ANALYTICS")
    enable_risk_management: bool = Field(True, env="ENABLE_RISK_MANAGEMENT")
    
    class Config:
        env_prefix = "FEATURE_"


class Settings(BaseSettings):
    """Main application settings."""
    
    # Core application settings
    app_name: str = Field("FinScope", env="APP_NAME")
    app_version: str = Field("1.0.0", env="APP_VERSION")
    environment: Environment = Field(Environment.DEVELOPMENT, env="ENVIRONMENT")
    debug: bool = Field(False, env="DEBUG")
    
    # API settings
    api_v1_prefix: str = Field("/api/v1", env="API_V1_PREFIX")
    api_v2_prefix: str = Field("/api/v2", env="API_V2_PREFIX")
    docs_url: str = Field("/docs", env="DOCS_URL")
    redoc_url: str = Field("/redoc", env="REDOC_URL")
    
    # Server settings
    host: str = Field("0.0.0.0", env="HOST")
    port: int = Field(8000, env="PORT")
    workers: int = Field(1, env="WORKERS")
    
    # CORS settings
    allowed_origins: List[str] = Field(["*"], env="ALLOWED_ORIGINS")
    allowed_methods: List[str] = Field(["*"], env="ALLOWED_METHODS")
    allowed_headers: List[str] = Field(["*"], env="ALLOWED_HEADERS")
    
    # Rate limiting
    rate_limit_per_minute: int = Field(100, env="RATE_LIMIT_PER_MINUTE")
    
    # Component settings
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    security: SecuritySettings = SecuritySettings()
    ai: AISettings = AISettings()
    defi: DeFiSettings = DeFiSettings()
    market_data: MarketDataSettings = MarketDataSettings()
    notifications: NotificationSettings = NotificationSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    features: FeatureFlags = FeatureFlags()
    
    @validator("environment", pre=True)
    def validate_environment(cls, v):
        if isinstance(v, str):
            return Environment(v.lower())
        return v
    
    @validator("allowed_origins", pre=True)
    def validate_allowed_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("allowed_methods", pre=True)
    def validate_allowed_methods(cls, v):
        if isinstance(v, str):
            return [method.strip() for method in v.split(",")]
        return v
    
    @validator("allowed_headers", pre=True)
    def validate_allowed_headers(cls, v):
        if isinstance(v, str):
            return [header.strip() for header in v.split(",")]
        return v
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == Environment.DEVELOPMENT
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == Environment.PRODUCTION
    
    @property
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.environment == Environment.TESTING
    
    def get_feature_config(self) -> Dict[str, Any]:
        """Get feature configuration as dictionary."""
        return {
            "ai_features": self.features.enable_ai_features,
            "defi_features": self.features.enable_defi_features,
            "enterprise_features": self.features.enable_enterprise_features,
            "social_trading": self.features.enable_social_trading,
            "paper_trading": self.features.enable_paper_trading,
            "mobile_api": self.features.enable_mobile_api,
            "advanced_charting": self.features.enable_advanced_charting,
            "news_sentiment": self.features.enable_news_sentiment,
            "portfolio_analytics": self.features.enable_portfolio_analytics,
            "risk_management": self.features.enable_risk_management,
        }
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings instance."""
    return settings