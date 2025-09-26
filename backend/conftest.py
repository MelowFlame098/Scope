import pytest
import asyncio
from httpx import AsyncClient
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock
import os
import sys

# Add the backend directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    mock_redis = Mock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.set = AsyncMock(return_value=True)
    mock_redis.delete = AsyncMock(return_value=1)
    mock_redis.exists = AsyncMock(return_value=False)
    mock_redis.expire = AsyncMock(return_value=True)
    mock_redis.hget = AsyncMock(return_value=None)
    mock_redis.hset = AsyncMock(return_value=True)
    mock_redis.hgetall = AsyncMock(return_value={})
    mock_redis.publish = AsyncMock(return_value=1)
    mock_redis.subscribe = AsyncMock()
    return mock_redis

@pytest.fixture
def mock_database():
    """Mock database session for testing."""
    mock_db = Mock()
    mock_db.query = Mock()
    mock_db.add = Mock()
    mock_db.commit = Mock()
    mock_db.rollback = Mock()
    mock_db.close = Mock()
    return mock_db

@pytest.fixture
def mock_market_data():
    """Mock market data for testing."""
    return {
        "symbol": "AAPL",
        "price": 150.00,
        "change": 2.50,
        "change_percent": 1.69,
        "volume": 1000000,
        "timestamp": "2024-01-01T10:00:00Z"
    }

@pytest.fixture
def mock_user_data():
    """Mock user data for testing."""
    return {
        "id": 1,
        "email": "test@example.com",
        "username": "testuser",
        "is_active": True,
        "subscription_tier": "premium"
    }

@pytest.fixture
def mock_portfolio_data():
    """Mock portfolio data for testing."""
    return {
        "id": 1,
        "user_id": 1,
        "name": "Test Portfolio",
        "total_value": 10000.00,
        "cash_balance": 5000.00,
        "positions": [
            {
                "symbol": "AAPL",
                "quantity": 10,
                "avg_price": 145.00,
                "current_price": 150.00
            }
        ]
    }

@pytest.fixture
def mock_news_data():
    """Mock news data for testing."""
    return [
        {
            "id": 1,
            "title": "Test News Article",
            "content": "This is a test news article content.",
            "source": "Test Source",
            "published_at": "2024-01-01T10:00:00Z",
            "symbols": ["AAPL", "MSFT"],
            "sentiment": 0.5
        }
    ]

@pytest.fixture
def mock_ai_prediction():
    """Mock AI prediction data for testing."""
    return {
        "symbol": "AAPL",
        "prediction": "bullish",
        "confidence": 0.85,
        "price_target": 160.00,
        "timeframe": "1d",
        "factors": ["technical_analysis", "sentiment", "volume"]
    }

@pytest.fixture
async def async_client():
    """Create async HTTP client for testing."""
    async with AsyncClient() as client:
        yield client

@pytest.fixture
def test_client():
    """Create FastAPI test client."""
    try:
        from main import app
        return TestClient(app)
    except ImportError:
        # Return a mock client if main app is not available
        return Mock()

# Environment setup for testing
@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Set up test environment variables."""
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///test.db")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/1")
    monkeypatch.setenv("SECRET_KEY", "test-secret-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

# Mock external API calls
@pytest.fixture
def mock_external_apis(monkeypatch):
    """Mock external API calls."""
    # Mock yfinance
    mock_yf = Mock()
    mock_yf.download = Mock(return_value=Mock())
    monkeypatch.setattr("yfinance.download", mock_yf.download)
    
    # Mock OpenAI
    mock_openai = Mock()
    mock_openai.chat.completions.create = AsyncMock(return_value=Mock(
        choices=[Mock(message=Mock(content="Test AI response"))]
    ))
    monkeypatch.setattr("openai.chat.completions.create", mock_openai.chat.completions.create)
    
    return {
        "yfinance": mock_yf,
        "openai": mock_openai
    }