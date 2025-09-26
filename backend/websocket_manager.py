import asyncio
import json
import logging
from typing import Dict, List, Set, Any, Optional
from datetime import datetime, timedelta
from fastapi import WebSocket, WebSocketDisconnect
import uuid
from dataclasses import dataclass, asdict
from enum import Enum

from market_data import MarketDataService
from ai_service import AIService
from news_service import NewsService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MessageType(Enum):
    """WebSocket message types."""
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    MARKET_DATA = "market_data"
    PRICE_UPDATE = "price_update"
    NEWS_UPDATE = "news_update"
    AI_INSIGHT = "ai_insight"
    PORTFOLIO_UPDATE = "portfolio_update"
    ALERT = "alert"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    AUTH = "auth"
    STATUS = "status"

@dataclass
class WebSocketMessage:
    """WebSocket message structure."""
    type: str
    data: Any
    timestamp: datetime = None
    user_id: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result

@dataclass
class ConnectionInfo:
    """Information about a WebSocket connection."""
    websocket: WebSocket
    user_id: Optional[str]
    subscriptions: Set[str]
    last_heartbeat: datetime
    connection_time: datetime
    
    def __post_init__(self):
        if self.connection_time is None:
            self.connection_time = datetime.now()
        if self.last_heartbeat is None:
            self.last_heartbeat = datetime.now()

class WebSocketManager:
    """Manages WebSocket connections and real-time data streaming."""
    
    def __init__(self):
        # Active connections
        self.connections: Dict[str, ConnectionInfo] = {}
        
        # Subscription management
        self.symbol_subscribers: Dict[str, Set[str]] = {}  # symbol -> connection_ids
        self.user_subscribers: Dict[str, Set[str]] = {}    # user_id -> connection_ids
        
        # Services
        self.market_service = MarketDataService()
        self.ai_service = AIService()
        self.news_service = NewsService()
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        
        # Configuration
        self.heartbeat_interval = 30  # seconds
        self.max_connections_per_user = 5
        self.price_update_interval = 1  # seconds
        self.news_update_interval = 60  # seconds
        
        # Data cache
        self.price_cache: Dict[str, Dict[str, Any]] = {}
        self.last_price_update = datetime.now()
        
        # Background tasks will be started when needed
        self.tasks_started = False
    
    async def connect(self, websocket: WebSocket, user_id: Optional[str] = None) -> str:
        """Accept a new WebSocket connection."""
        # Start background tasks if not already started
        if not self.tasks_started:
            self._start_background_tasks()
            
        await websocket.accept()
        
        # Generate unique connection ID
        connection_id = str(uuid.uuid4())
        
        # Check connection limits
        if user_id and self._get_user_connection_count(user_id) >= self.max_connections_per_user:
            await self._send_error(websocket, "Maximum connections exceeded for user")
            await websocket.close()
            return None
        
        # Store connection info
        self.connections[connection_id] = ConnectionInfo(
            websocket=websocket,
            user_id=user_id,
            subscriptions=set(),
            last_heartbeat=datetime.now(),
            connection_time=datetime.now()
        )
        
        # Track user connections
        if user_id:
            if user_id not in self.user_subscribers:
                self.user_subscribers[user_id] = set()
            self.user_subscribers[user_id].add(connection_id)
        
        logger.info(f"WebSocket connected: {connection_id} (user: {user_id})")
        
        # Send welcome message
        await self._send_message(connection_id, WebSocketMessage(
            type=MessageType.STATUS.value,
            data={
                "status": "connected",
                "connection_id": connection_id,
                "server_time": datetime.now().isoformat()
            }
        ))
        
        return connection_id
    
    async def disconnect(self, connection_id: str):
        """Handle WebSocket disconnection."""
        if connection_id not in self.connections:
            return
        
        connection = self.connections[connection_id]
        
        # Remove from symbol subscriptions
        for symbol in connection.subscriptions:
            if symbol in self.symbol_subscribers:
                self.symbol_subscribers[symbol].discard(connection_id)
                if not self.symbol_subscribers[symbol]:
                    del self.symbol_subscribers[symbol]
        
        # Remove from user subscriptions
        if connection.user_id and connection.user_id in self.user_subscribers:
            self.user_subscribers[connection.user_id].discard(connection_id)
            if not self.user_subscribers[connection.user_id]:
                del self.user_subscribers[connection.user_id]
        
        # Remove connection
        del self.connections[connection_id]
        
        logger.info(f"WebSocket disconnected: {connection_id}")
    
    async def handle_message(self, connection_id: str, message: str):
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            message_data = data.get('data', {})
            
            if message_type == MessageType.SUBSCRIBE.value:
                await self._handle_subscribe(connection_id, message_data)
            elif message_type == MessageType.UNSUBSCRIBE.value:
                await self._handle_unsubscribe(connection_id, message_data)
            elif message_type == MessageType.HEARTBEAT.value:
                await self._handle_heartbeat(connection_id)
            elif message_type == MessageType.AUTH.value:
                await self._handle_auth(connection_id, message_data)
            else:
                await self._send_error_to_connection(connection_id, f"Unknown message type: {message_type}")
                
        except json.JSONDecodeError:
            await self._send_error_to_connection(connection_id, "Invalid JSON message")
        except Exception as e:
            logger.error(f"Error handling message from {connection_id}: {e}")
            await self._send_error_to_connection(connection_id, "Internal server error")
    
    async def _handle_subscribe(self, connection_id: str, data: Dict[str, Any]):
        """Handle subscription request."""
        symbols = data.get('symbols', [])
        
        if not isinstance(symbols, list):
            await self._send_error_to_connection(connection_id, "Symbols must be a list")
            return
        
        connection = self.connections.get(connection_id)
        if not connection:
            return
        
        # Add subscriptions
        for symbol in symbols:
            symbol = symbol.upper()
            connection.subscriptions.add(symbol)
            
            if symbol not in self.symbol_subscribers:
                self.symbol_subscribers[symbol] = set()
            self.symbol_subscribers[symbol].add(connection_id)
        
        # Send confirmation
        await self._send_message(connection_id, WebSocketMessage(
            type=MessageType.STATUS.value,
            data={
                "status": "subscribed",
                "symbols": symbols,
                "total_subscriptions": len(connection.subscriptions)
            }
        ))
        
        # Send current prices for subscribed symbols
        for symbol in symbols:
            if symbol in self.price_cache:
                await self._send_message(connection_id, WebSocketMessage(
                    type=MessageType.PRICE_UPDATE.value,
                    data=self.price_cache[symbol]
                ))
    
    async def _handle_unsubscribe(self, connection_id: str, data: Dict[str, Any]):
        """Handle unsubscription request."""
        symbols = data.get('symbols', [])
        
        if not isinstance(symbols, list):
            await self._send_error_to_connection(connection_id, "Symbols must be a list")
            return
        
        connection = self.connections.get(connection_id)
        if not connection:
            return
        
        # Remove subscriptions
        for symbol in symbols:
            symbol = symbol.upper()
            connection.subscriptions.discard(symbol)
            
            if symbol in self.symbol_subscribers:
                self.symbol_subscribers[symbol].discard(connection_id)
                if not self.symbol_subscribers[symbol]:
                    del self.symbol_subscribers[symbol]
        
        # Send confirmation
        await self._send_message(connection_id, WebSocketMessage(
            type=MessageType.STATUS.value,
            data={
                "status": "unsubscribed",
                "symbols": symbols,
                "total_subscriptions": len(connection.subscriptions)
            }
        ))
    
    async def _handle_heartbeat(self, connection_id: str):
        """Handle heartbeat message."""
        connection = self.connections.get(connection_id)
        if connection:
            connection.last_heartbeat = datetime.now()
            
            await self._send_message(connection_id, WebSocketMessage(
                type=MessageType.HEARTBEAT.value,
                data={"server_time": datetime.now().isoformat()}
            ))
    
    async def _handle_auth(self, connection_id: str, data: Dict[str, Any]):
        """Handle authentication message."""
        # In a real implementation, you would validate the token
        token = data.get('token')
        user_id = data.get('user_id')
        
        if token and user_id:
            connection = self.connections.get(connection_id)
            if connection:
                connection.user_id = user_id
                
                # Update user subscribers
                if user_id not in self.user_subscribers:
                    self.user_subscribers[user_id] = set()
                self.user_subscribers[user_id].add(connection_id)
                
                await self._send_message(connection_id, WebSocketMessage(
                    type=MessageType.STATUS.value,
                    data={"status": "authenticated", "user_id": user_id}
                ))
        else:
            await self._send_error_to_connection(connection_id, "Invalid authentication data")
    
    async def broadcast_price_update(self, symbol: str, price_data: Dict[str, Any]):
        """Broadcast price update to all subscribers of a symbol."""
        symbol = symbol.upper()
        
        # Update cache
        self.price_cache[symbol] = {
            "symbol": symbol,
            "price": price_data.get('price'),
            "change": price_data.get('change'),
            "change_percent": price_data.get('change_percent'),
            "volume": price_data.get('volume'),
            "timestamp": datetime.now().isoformat()
        }
        
        # Broadcast to subscribers
        if symbol in self.symbol_subscribers:
            message = WebSocketMessage(
                type=MessageType.PRICE_UPDATE.value,
                data=self.price_cache[symbol]
            )
            
            await self._broadcast_to_connections(
                self.symbol_subscribers[symbol],
                message
            )
    
    async def broadcast_news_update(self, news_data: Dict[str, Any]):
        """Broadcast news update to all connected clients."""
        message = WebSocketMessage(
            type=MessageType.NEWS_UPDATE.value,
            data=news_data
        )
        
        # Broadcast to all connections
        await self._broadcast_to_all(message)
    
    async def broadcast_ai_insight(self, insight_data: Dict[str, Any], target_symbols: List[str] = None):
        """Broadcast AI insight to relevant subscribers."""
        message = WebSocketMessage(
            type=MessageType.AI_INSIGHT.value,
            data=insight_data
        )
        
        if target_symbols:
            # Broadcast to subscribers of specific symbols
            target_connections = set()
            for symbol in target_symbols:
                symbol = symbol.upper()
                if symbol in self.symbol_subscribers:
                    target_connections.update(self.symbol_subscribers[symbol])
            
            await self._broadcast_to_connections(target_connections, message)
        else:
            # Broadcast to all connections
            await self._broadcast_to_all(message)
    
    async def send_user_alert(self, user_id: str, alert_data: Dict[str, Any]):
        """Send alert to specific user."""
        if user_id in self.user_subscribers:
            message = WebSocketMessage(
                type=MessageType.ALERT.value,
                data=alert_data,
                user_id=user_id
            )
            
            await self._broadcast_to_connections(
                self.user_subscribers[user_id],
                message
            )
    
    async def _send_message(self, connection_id: str, message: WebSocketMessage):
        """Send message to a specific connection."""
        connection = self.connections.get(connection_id)
        if not connection:
            return
        
        try:
            await connection.websocket.send_text(json.dumps(message.to_dict()))
        except Exception as e:
            logger.error(f"Error sending message to {connection_id}: {e}")
            await self.disconnect(connection_id)
    
    async def _send_error_to_connection(self, connection_id: str, error_message: str):
        """Send error message to a specific connection."""
        await self._send_message(connection_id, WebSocketMessage(
            type=MessageType.ERROR.value,
            data={"error": error_message}
        ))
    
    async def _send_error(self, websocket: WebSocket, error_message: str):
        """Send error message to a WebSocket."""
        try:
            message = WebSocketMessage(
                type=MessageType.ERROR.value,
                data={"error": error_message}
            )
            await websocket.send_text(json.dumps(message.to_dict()))
        except Exception as e:
            logger.error(f"Error sending error message: {e}")
    
    async def _broadcast_to_connections(self, connection_ids: Set[str], message: WebSocketMessage):
        """Broadcast message to specific connections."""
        if not connection_ids:
            return
        
        # Create tasks for concurrent sending
        tasks = []
        for connection_id in connection_ids.copy():  # Copy to avoid modification during iteration
            if connection_id in self.connections:
                tasks.append(self._send_message(connection_id, message))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _broadcast_to_all(self, message: WebSocketMessage):
        """Broadcast message to all connections."""
        await self._broadcast_to_connections(set(self.connections.keys()), message)
    
    def _get_user_connection_count(self, user_id: str) -> int:
        """Get number of connections for a user."""
        return len(self.user_subscribers.get(user_id, set()))
    
    def _start_background_tasks(self):
        """Start background tasks for data updates and maintenance."""
        if self.tasks_started:
            return
            
        try:
            # Price update task
            task = asyncio.create_task(self._price_update_loop())
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)
            
            # News update task
            task = asyncio.create_task(self._news_update_loop())
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)
            
            # Heartbeat check task
            task = asyncio.create_task(self._heartbeat_check_loop())
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)
            
            # AI insights task
            task = asyncio.create_task(self._ai_insights_loop())
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)
            
            self.tasks_started = True
        except RuntimeError:
            # No event loop running, tasks will be started later
            pass
    
    async def _price_update_loop(self):
        """Background task for updating prices."""
        while True:
            try:
                # Get all subscribed symbols
                symbols = set(self.symbol_subscribers.keys())
                
                if symbols:
                    # Fetch real-time data for subscribed symbols
                    for symbol in symbols:
                        try:
                            # Get real-time data
                            real_time_data = await self.market_service.get_real_time_data([symbol])
                            
                            if real_time_data and symbol in real_time_data:
                                await self.broadcast_price_update(symbol, real_time_data[symbol])
                                
                        except Exception as e:
                            logger.error(f"Error updating price for {symbol}: {e}")
                
                await asyncio.sleep(self.price_update_interval)
                
            except Exception as e:
                logger.error(f"Error in price update loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _news_update_loop(self):
        """Background task for news updates."""
        while True:
            try:
                # Get latest news
                news_articles = await self.news_service.get_news(limit=5)
                
                if news_articles:
                    # Broadcast latest news
                    for article in news_articles:
                        news_data = {
                            "id": article.id,
                            "title": article.title,
                            "summary": article.summary,
                            "source": article.source,
                            "category": article.category,
                            "sentiment": article.sentiment_label,
                            "related_symbols": article.related_symbols,
                            "published_at": article.published_at.isoformat() if article.published_at else None
                        }
                        
                        await self.broadcast_news_update(news_data)
                
                await asyncio.sleep(self.news_update_interval)
                
            except Exception as e:
                logger.error(f"Error in news update loop: {e}")
                await asyncio.sleep(30)  # Wait before retrying
    
    async def _heartbeat_check_loop(self):
        """Background task for checking connection health."""
        while True:
            try:
                current_time = datetime.now()
                stale_connections = []
                
                # Check for stale connections
                for connection_id, connection in self.connections.items():
                    time_since_heartbeat = current_time - connection.last_heartbeat
                    if time_since_heartbeat.total_seconds() > self.heartbeat_interval * 3:
                        stale_connections.append(connection_id)
                
                # Disconnect stale connections
                for connection_id in stale_connections:
                    logger.info(f"Disconnecting stale connection: {connection_id}")
                    await self.disconnect(connection_id)
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in heartbeat check loop: {e}")
                await asyncio.sleep(10)
    
    async def _ai_insights_loop(self):
        """Background task for generating and broadcasting AI insights."""
        while True:
            try:
                # Generate AI insights periodically
                insights = await self.ai_service.get_insights(limit=3)
                
                for insight in insights:
                    insight_data = {
                        "id": insight.id,
                        "title": insight.title,
                        "summary": insight.summary,
                        "category": insight.category,
                        "sentiment": insight.sentiment,
                        "confidence": insight.confidence,
                        "related_symbols": insight.related_symbols,
                        "created_at": insight.created_at.isoformat()
                    }
                    
                    await self.broadcast_ai_insight(insight_data, insight.related_symbols)
                
                # Wait 5 minutes before generating new insights
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in AI insights loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def cleanup(self):
        """Clean up resources and cancel background tasks."""
        # Cancel all background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Close all connections
        for connection_id in list(self.connections.keys()):
            await self.disconnect(connection_id)
        
        logger.info("WebSocket manager cleaned up")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get WebSocket manager statistics."""
        return {
            "total_connections": len(self.connections),
            "unique_users": len(self.user_subscribers),
            "subscribed_symbols": len(self.symbol_subscribers),
            "background_tasks": len(self.background_tasks),
            "cached_prices": len(self.price_cache),
            "connections_by_user": {
                user_id: len(connections) 
                for user_id, connections in self.user_subscribers.items()
            },
            "symbol_subscribers": {
                symbol: len(subscribers)
                for symbol, subscribers in self.symbol_subscribers.items()
            }
        }

# Global WebSocket manager instance
websocket_manager = WebSocketManager()