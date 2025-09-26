import pytest
from unittest.mock import Mock, patch, AsyncMock
import json
import redis
from datetime import datetime, timedelta

class TestRedisService:
    """Test Redis service functionality."""
    
    @pytest.fixture
    def redis_service(self, mock_redis):
        """Create Redis service instance with mocked Redis."""
        from services.redis_service import RedisService
        return RedisService(redis_client=mock_redis)
    
    @pytest.mark.asyncio
    async def test_cache_market_data(self, redis_service, mock_redis):
        """Test caching market data in Redis."""
        symbol = "AAPL"
        market_data = {
            "price": 150.25,
            "change": 2.50,
            "change_percent": 1.69,
            "timestamp": datetime.now().isoformat()
        }
        
        await redis_service.cache_market_data(symbol, market_data)
        
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        assert call_args[0][0] == f"market:{symbol}"
        assert call_args[0][2] == json.dumps(market_data)
    
    @pytest.mark.asyncio
    async def test_get_cached_market_data(self, redis_service, mock_redis):
        """Test retrieving cached market data from Redis."""
        symbol = "AAPL"
        cached_data = {
            "price": 150.25,
            "change": 2.50,
            "change_percent": 1.69
        }
        
        mock_redis.get.return_value = json.dumps(cached_data)
        
        result = await redis_service.get_cached_market_data(symbol)
        
        mock_redis.get.assert_called_once_with(f"market:{symbol}")
        assert result == cached_data
    
    @pytest.mark.asyncio
    async def test_publish_price_update(self, redis_service, mock_redis):
        """Test publishing price updates via Redis pub/sub."""
        symbol = "AAPL"
        price_data = {
            "symbol": symbol,
            "price": 150.25,
            "timestamp": datetime.now().isoformat()
        }
        
        await redis_service.publish_price_update(symbol, price_data)
        
        mock_redis.publish.assert_called_once_with(
            f"price_updates:{symbol}",
            json.dumps(price_data)
        )
    
    @pytest.mark.asyncio
    async def test_cache_user_session(self, redis_service, mock_redis):
        """Test caching user session data."""
        user_id = "user123"
        session_data = {
            "user_id": user_id,
            "username": "testuser",
            "email": "test@example.com",
            "login_time": datetime.now().isoformat()
        }
        
        await redis_service.cache_user_session(user_id, session_data, ttl=3600)
        
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        assert call_args[0][0] == f"session:{user_id}"
        assert call_args[0][1] == 3600
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, redis_service, mock_redis):
        """Test rate limiting functionality."""
        user_id = "user123"
        action = "api_call"
        limit = 100
        window = 3600
        
        # Mock current count
        mock_redis.get.return_value = "50"
        mock_redis.incr.return_value = 51
        mock_redis.expire.return_value = True
        
        result = await redis_service.check_rate_limit(user_id, action, limit, window)
        
        assert result is True
        mock_redis.get.assert_called_once_with(f"rate_limit:{user_id}:{action}")
        mock_redis.incr.assert_called_once_with(f"rate_limit:{user_id}:{action}")
    
    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self, redis_service, mock_redis):
        """Test rate limit exceeded scenario."""
        user_id = "user123"
        action = "api_call"
        limit = 100
        window = 3600
        
        # Mock exceeded count
        mock_redis.get.return_value = "100"
        
        result = await redis_service.check_rate_limit(user_id, action, limit, window)
        
        assert result is False
        mock_redis.incr.assert_not_called()

class TestDatabaseService:
    """Test database service functionality."""
    
    @pytest.fixture
    def db_service(self, mock_database):
        """Create database service instance with mocked database."""
        from services.database_service import DatabaseService
        return DatabaseService(db=mock_database)
    
    @pytest.mark.asyncio
    async def test_create_user(self, db_service, mock_database):
        """Test creating a new user."""
        user_data = {
            "email": "test@example.com",
            "username": "testuser",
            "password_hash": "hashed_password"
        }
        
        mock_database.execute.return_value = Mock(lastrowid=1)
        
        result = await db_service.create_user(user_data)
        
        assert result["id"] == 1
        assert result["email"] == user_data["email"]
        mock_database.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_user_by_email(self, db_service, mock_database):
        """Test retrieving user by email."""
        email = "test@example.com"
        mock_user = {
            "id": 1,
            "email": email,
            "username": "testuser",
            "created_at": datetime.now()
        }
        
        mock_database.fetch_one.return_value = mock_user
        
        result = await db_service.get_user_by_email(email)
        
        assert result == mock_user
        mock_database.fetch_one.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_save_trade_record(self, db_service, mock_database):
        """Test saving trade record to database."""
        trade_data = {
            "user_id": 1,
            "symbol": "AAPL",
            "action": "buy",
            "quantity": 100,
            "price": 150.25,
            "timestamp": datetime.now()
        }
        
        mock_database.execute.return_value = Mock(lastrowid=1)
        
        result = await db_service.save_trade_record(trade_data)
        
        assert result["id"] == 1
        mock_database.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_user_portfolio(self, db_service, mock_database, mock_portfolio_data):
        """Test retrieving user portfolio."""
        user_id = 1
        
        mock_database.fetch_all.return_value = mock_portfolio_data
        
        result = await db_service.get_user_portfolio(user_id)
        
        assert result == mock_portfolio_data
        mock_database.fetch_all.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_portfolio_position(self, db_service, mock_database):
        """Test updating portfolio position."""
        position_data = {
            "user_id": 1,
            "symbol": "AAPL",
            "quantity": 150,
            "average_price": 148.75
        }
        
        mock_database.execute.return_value = Mock(rowcount=1)
        
        result = await db_service.update_portfolio_position(position_data)
        
        assert result is True
        mock_database.execute.assert_called_once()

class TestAIService:
    """Test AI service functionality."""
    
    @pytest.fixture
    def ai_service(self):
        """Create AI service instance."""
        from services.ai_service import AIService
        return AIService()
    
    @pytest.mark.asyncio
    async def test_generate_market_prediction(self, ai_service, mock_ai_predictions):
        """Test generating market predictions."""
        symbol = "AAPL"
        market_data = {
            "price": 150.25,
            "volume": 50000000,
            "indicators": {
                "rsi": 65.4,
                "sma_20": 148.50,
                "sma_50": 145.75
            }
        }
        
        with patch('ai_service.predict_price_movement') as mock_predict:
            mock_predict.return_value = mock_ai_predictions
            
            result = await ai_service.generate_market_prediction(symbol, market_data)
            
            assert "prediction" in result
            assert "confidence" in result
            assert "reasoning" in result
            mock_predict.assert_called_once_with(symbol, market_data)
    
    @pytest.mark.asyncio
    async def test_analyze_sentiment(self, ai_service):
        """Test sentiment analysis of news/social media."""
        text_data = [
            "Apple stock is performing exceptionally well this quarter",
            "Concerns about supply chain issues affecting tech stocks",
            "Strong earnings report boosts investor confidence"
        ]
        
        with patch('ai_service.analyze_text_sentiment') as mock_sentiment:
            mock_sentiment.return_value = {
                "overall_sentiment": "positive",
                "sentiment_score": 0.75,
                "individual_scores": [0.8, -0.3, 0.9]
            }
            
            result = await ai_service.analyze_sentiment(text_data)
            
            assert result["overall_sentiment"] == "positive"
            assert result["sentiment_score"] == 0.75
            mock_sentiment.assert_called_once_with(text_data)
    
    @pytest.mark.asyncio
    async def test_generate_trading_signals(self, ai_service):
        """Test generating trading signals."""
        symbol = "AAPL"
        technical_data = {
            "price": 150.25,
            "sma_20": 148.50,
            "sma_50": 145.75,
            "rsi": 65.4,
            "macd": 1.25,
            "volume": 50000000
        }
        
        with patch('ai_service.generate_signals') as mock_signals:
            mock_signals.return_value = {
                "signal": "buy",
                "strength": "moderate",
                "entry_price": 150.00,
                "stop_loss": 145.00,
                "take_profit": 158.00,
                "reasoning": "Bullish crossover with strong volume"
            }
            
            result = await ai_service.generate_trading_signals(symbol, technical_data)
            
            assert result["signal"] == "buy"
            assert result["strength"] == "moderate"
            mock_signals.assert_called_once_with(symbol, technical_data)