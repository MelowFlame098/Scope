import os
from typing import Any, Dict, List, Optional, Union

from pydantic import AnyHttpUrl, EmailStr, HttpUrl, PostgresDsn, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"extra": "allow", "env_file": ".env", "case_sensitive": True}
    
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "FinScope"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "Advanced Financial Analysis and Portfolio Management Platform"
    
    # Environment Configuration
    ENVIRONMENT: str = "development"
    DATABASE_URL: Optional[str] = None
    CORS_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000"
    
    # Server Configuration
    SERVER_NAME: str = "localhost"
    SERVER_HOST: AnyHttpUrl = "http://localhost"
    SERVER_PORT: int = 8000
    
    # CORS Configuration
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = [
        "http://localhost:3000",  # React development server
        "http://localhost:8080",  # Alternative frontend port
        "http://localhost:5173",  # Vite development server
    ]
    
    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    # Database Configuration
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "finscope"
    POSTGRES_PASSWORD: str = "finscope123"
    POSTGRES_DB: str = "finscope"
    POSTGRES_PORT: str = "5432"
    
    SQLALCHEMY_DATABASE_URI: Optional[PostgresDsn] = None
    
    @validator("SQLALCHEMY_DATABASE_URI", pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        if isinstance(v, str):
            return v
        user = values.get("POSTGRES_USER")
        password = values.get("POSTGRES_PASSWORD")
        host = values.get("POSTGRES_SERVER")
        port = values.get("POSTGRES_PORT")
        db = values.get("POSTGRES_DB") or ""
        
        # Build the URL manually for compatibility with newer Pydantic versions
        if user and password:
            return f"postgresql://{user}:{password}@{host}:{port}/{db}"
        elif user:
            return f"postgresql://{user}@{host}:{port}/{db}"
        else:
            return f"postgresql://{host}:{port}/{db}"
    
    # Security Configuration
    SECRET_KEY: str = "your-secret-key-change-this-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Email Configuration
    SMTP_TLS: bool = True
    SMTP_PORT: Optional[int] = None
    SMTP_HOST: Optional[str] = None
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    EMAILS_FROM_EMAIL: Optional[EmailStr] = None
    EMAILS_FROM_NAME: Optional[str] = None
    
    @validator("EMAILS_FROM_NAME")
    def get_project_name(cls, v: Optional[str], values: Dict[str, Any]) -> str:
        if not v:
            return values["PROJECT_NAME"]
        return v
    
    EMAIL_RESET_TOKEN_EXPIRE_HOURS: int = 48
    EMAIL_TEMPLATES_DIR: str = "backend/email-templates/build"
    EMAILS_ENABLED: bool = False
    
    @validator("EMAILS_ENABLED", pre=True)
    def get_emails_enabled(cls, v: bool, values: Dict[str, Any]) -> bool:
        return bool(
            values.get("SMTP_HOST")
            and values.get("SMTP_PORT")
            and values.get("EMAILS_FROM_EMAIL")
        )
    
    # External API Configuration
    # CoinGecko API (Free tier)
    COINGECKO_API_URL: str = "https://api.coingecko.com/api/v3"
    COINGECKO_API_KEY: Optional[str] = None  # Optional for free tier
    
    # Alpha Vantage API (Free tier: 5 calls per minute, 500 calls per day)
    ALPHA_VANTAGE_API_KEY: Optional[str] = None
    ALPHA_VANTAGE_API_URL: str = "https://www.alphavantage.co/query"
    
    # News API (Free tier: 1000 requests per month)
    NEWS_API_KEY: Optional[str] = None
    NEWS_API_URL: str = "https://newsapi.org/v2"
    
    # Financial Modeling Prep API (Free tier: 250 calls per day)
    FMP_API_KEY: Optional[str] = None
    FMP_API_URL: str = "https://financialmodelingprep.com/api/v3"
    
    # Polygon.io API (Free tier: 5 calls per minute)
    POLYGON_API_KEY: Optional[str] = None
    POLYGON_API_URL: str = "https://api.polygon.io"
    
    # Redis Configuration (for caching)
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_EXPIRE_SECONDS: int = 300  # 5 minutes
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_CALLS: int = 100
    RATE_LIMIT_PERIOD: int = 60  # seconds
    
    # Data Refresh Configuration
    AUTO_REFRESH_ENABLED: bool = True
    CRYPTO_REFRESH_INTERVAL: int = 300  # 5 minutes
    STOCK_REFRESH_INTERVAL: int = 900   # 15 minutes
    NEWS_REFRESH_INTERVAL: int = 1800   # 30 minutes
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # File Upload Configuration
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".gif", ".pdf"]
    UPLOAD_DIR: str = "uploads"
    
    # Pagination Configuration
    DEFAULT_PAGE_SIZE: int = 20
    MAX_PAGE_SIZE: int = 100
    
    # AI/ML Configuration
    AI_INSIGHTS_ENABLED: bool = True
    MIN_CONFIDENCE_THRESHOLD: float = 0.7
    
    # WebSocket Configuration
    WEBSOCKET_ENABLED: bool = True
    WEBSOCKET_HEARTBEAT_INTERVAL: int = 30
    
    # Development Configuration
    DEBUG: bool = False
    TESTING: bool = False
    
    # First Superuser
    FIRST_SUPERUSER: EmailStr = "admin@finscope.com"
    FIRST_SUPERUSER_PASSWORD: str = "admin123"
    



# Create settings instance
settings = Settings()


# Helper functions
def get_database_url() -> str:
    """Get the database URL for SQLAlchemy."""
    return str(settings.SQLALCHEMY_DATABASE_URI)


def is_development() -> bool:
    """Check if the application is running in development mode."""
    return settings.DEBUG or settings.TESTING


def get_cors_origins() -> List[str]:
    """Get CORS origins as a list of strings."""
    return [str(origin) for origin in settings.BACKEND_CORS_ORIGINS]


def get_api_keys() -> Dict[str, Optional[str]]:
    """Get all API keys for external services."""
    return {
        "coingecko": settings.COINGECKO_API_KEY,
        "alpha_vantage": settings.ALPHA_VANTAGE_API_KEY,
        "news_api": settings.NEWS_API_KEY,
        "fmp": settings.FMP_API_KEY,
        "polygon": settings.POLYGON_API_KEY,
    }


def validate_required_settings() -> None:
    """Validate that required settings are present."""
    required_settings = [
        "SECRET_KEY",
        "SQLALCHEMY_DATABASE_URI",
    ]
    
    missing_settings = []
    for setting in required_settings:
        if not getattr(settings, setting):
            missing_settings.append(setting)
    
    if missing_settings:
        raise ValueError(
            f"Missing required settings: {', '.join(missing_settings)}"
        )


def get_environment_info() -> Dict[str, Any]:
    """Get environment information for debugging."""
    return {
        "project_name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "debug": settings.DEBUG,
        "testing": settings.TESTING,
        "database_configured": bool(settings.SQLALCHEMY_DATABASE_URI),
        "emails_enabled": settings.EMAILS_ENABLED,
        "redis_url": settings.REDIS_URL,
        "api_keys_configured": {
            key: bool(value) for key, value in get_api_keys().items()
        },
    }