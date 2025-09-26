import pytest
from httpx import AsyncClient
from unittest.mock import Mock, patch, AsyncMock
import json
from datetime import datetime, timedelta

class TestMarketAPI:
    """Test market data API endpoints."""
    
    @pytest.mark.asyncio
    async def test_get_stock_price(self, async_client, mock_market_data):
        """Test getting current stock price."""
        symbol = "AAPL"
        
        with patch('market_data.get_current_price') as mock_price:
            mock_price.return_value = mock_market_data["price_data"]
            
            response = await async_client.get(f"/api/market/price/{symbol}")
            
            assert response.status_code == 200
            data = response.json()
            assert data["symbol"] == symbol
            assert "price" in data
            assert "change" in data
            assert "change_percent" in data
    
    @pytest.mark.asyncio
    async def test_get_stock_price_invalid_symbol(self, async_client):
        """Test getting price for invalid symbol."""
        symbol = "INVALID"
        
        with patch('market_data.get_current_price') as mock_price:
            mock_price.return_value = None
            
            response = await async_client.get(f"/api/market/price/{symbol}")
            
            assert response.status_code == 404
            assert "Symbol not found" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_get_historical_data(self, async_client, mock_market_data):
        """Test getting historical market data."""
        symbol = "AAPL"
        params = {
            "period": "1d",
            "interval": "1h"
        }
        
        with patch('market_data.get_historical_data') as mock_historical:
            mock_historical.return_value = mock_market_data["historical_data"]
            
            response = await async_client.get(
                f"/api/market/historical/{symbol}",
                params=params
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "data" in data
            assert len(data["data"]) > 0
            assert "timestamp" in data["data"][0]
            assert "open" in data["data"][0]
            assert "high" in data["data"][0]
            assert "low" in data["data"][0]
            assert "close" in data["data"][0]
            assert "volume" in data["data"][0]
    
    @pytest.mark.asyncio
    async def test_get_market_news(self, async_client, mock_news_data):
        """Test getting market news."""
        params = {
            "symbol": "AAPL",
            "limit": 10
        }
        
        with patch('news.get_market_news') as mock_news:
            mock_news.return_value = mock_news_data
            
            response = await async_client.get(
                "/api/market/news",
                params=params
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "articles" in data
            assert len(data["articles"]) > 0
            assert "title" in data["articles"][0]
            assert "url" in data["articles"][0]
            assert "published_at" in data["articles"][0]
    
    @pytest.mark.asyncio
    async def test_get_watchlist(self, async_client, mock_user_data):
        """Test getting user watchlist."""
        with patch('auth.get_current_user') as mock_current_user:
            mock_current_user.return_value = mock_user_data
            
            with patch('watchlist.get_user_watchlist') as mock_watchlist:
                mock_watchlist.return_value = [
                    {"symbol": "AAPL", "name": "Apple Inc."},
                    {"symbol": "GOOGL", "name": "Alphabet Inc."},
                    {"symbol": "MSFT", "name": "Microsoft Corporation"}
                ]
                
                headers = {"Authorization": "Bearer test-token"}
                response = await async_client.get(
                    "/api/market/watchlist",
                    headers=headers
                )
                
                assert response.status_code == 200
                data = response.json()
                assert "symbols" in data
                assert len(data["symbols"]) == 3
    
    @pytest.mark.asyncio
    async def test_add_to_watchlist(self, async_client, mock_user_data):
        """Test adding symbol to watchlist."""
        symbol_data = {
            "symbol": "TSLA",
            "name": "Tesla, Inc."
        }
        
        with patch('auth.get_current_user') as mock_current_user:
            mock_current_user.return_value = mock_user_data
            
            with patch('watchlist.add_to_watchlist') as mock_add:
                mock_add.return_value = True
                
                headers = {"Authorization": "Bearer test-token"}
                response = await async_client.post(
                    "/api/market/watchlist",
                    json=symbol_data,
                    headers=headers
                )
                
                assert response.status_code == 201
                assert "added to watchlist" in response.json()["message"]
    
    @pytest.mark.asyncio
    async def test_remove_from_watchlist(self, async_client, mock_user_data):
        """Test removing symbol from watchlist."""
        symbol = "AAPL"
        
        with patch('auth.get_current_user') as mock_current_user:
            mock_current_user.return_value = mock_user_data
            
            with patch('watchlist.remove_from_watchlist') as mock_remove:
                mock_remove.return_value = True
                
                headers = {"Authorization": "Bearer test-token"}
                response = await async_client.delete(
                    f"/api/market/watchlist/{symbol}",
                    headers=headers
                )
                
                assert response.status_code == 200
                assert "removed from watchlist" in response.json()["message"]
    
    @pytest.mark.asyncio
    async def test_get_market_indicators(self, async_client, mock_market_data):
        """Test getting technical indicators."""
        symbol = "AAPL"
        params = {
            "indicators": "sma,rsi,macd",
            "period": "1d"
        }
        
        with patch('indicators.calculate_indicators') as mock_indicators:
            mock_indicators.return_value = {
                "sma": [150.25, 151.30, 152.15],
                "rsi": [65.4, 67.2, 69.1],
                "macd": {
                    "macd": [1.25, 1.30, 1.35],
                    "signal": [1.20, 1.25, 1.30],
                    "histogram": [0.05, 0.05, 0.05]
                }
            }
            
            response = await async_client.get(
                f"/api/market/indicators/{symbol}",
                params=params
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "indicators" in data
            assert "sma" in data["indicators"]
            assert "rsi" in data["indicators"]
            assert "macd" in data["indicators"]
    
    @pytest.mark.asyncio
    async def test_get_crypto_price(self, async_client):
        """Test getting cryptocurrency price."""
        symbol = "BTC"
        
        with patch('crypto.get_crypto_price') as mock_crypto:
            mock_crypto.return_value = {
                "symbol": "BTC",
                "price": 45000.50,
                "change": 1250.75,
                "change_percent": 2.85,
                "volume": 28500000000
            }
            
            response = await async_client.get(f"/api/market/crypto/{symbol}")
            
            assert response.status_code == 200
            data = response.json()
            assert data["symbol"] == symbol
            assert data["price"] == 45000.50
            assert data["change_percent"] == 2.85
    
    @pytest.mark.asyncio
    async def test_search_symbols(self, async_client):
        """Test searching for symbols."""
        params = {
            "query": "apple",
            "limit": 5
        }
        
        with patch('search.search_symbols') as mock_search:
            mock_search.return_value = [
                {"symbol": "AAPL", "name": "Apple Inc.", "type": "stock"},
                {"symbol": "APLE", "name": "Apple Hospitality REIT", "type": "reit"}
            ]
            
            response = await async_client.get(
                "/api/market/search",
                params=params
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "results" in data
            assert len(data["results"]) == 2
            assert data["results"][0]["symbol"] == "AAPL"