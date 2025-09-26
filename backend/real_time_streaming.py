import asyncio
import aiohttp
import aioredis
import websockets
import json
import logging
from typing import Dict, List, Set, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from contextlib import asynccontextmanager
from fastapi import WebSocket, WebSocketDisconnect
import yfinance as yf
import ccxt
from enhanced_market_data import enhanced_market_data_service

logger = logging.getLogger(__name__)

class StreamType(Enum):
    PRICE_UPDATE = "price_update"
    NEWS_UPDATE = "news_update"
    MARKET_OVERVIEW = "market_overview"
    PORTFOLIO_UPDATE = "portfolio_update"
    AI_INSIGHT = "ai_insight"
    SYSTEM_ALERT = "system_alert"

@dataclass
class StreamMessage:
    """Real-time stream message structure."""
    type: StreamType
    data: Dict[str, Any]
    timestamp: datetime
    id: str = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'type': self.type.value,
            'data': self.data,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class Subscription:
    """Client subscription configuration."""
    client_id: str
    websocket: WebSocket
    symbols: Set[str]
    stream_types: Set[StreamType]
    user_id: Optional[str] = None
    portfolio_id: Optional[str] = None
    last_heartbeat: datetime = None
    
    def __post_init__(self):
        if self.last_heartbeat is None:
            self.last_heartbeat = datetime.now()

class RealTimeStreamingService:
    """Enhanced real-time data streaming service with WebSocket management."""
    
    def __init__(self):
        # Client management
        self.subscriptions: Dict[str, Subscription] = {}
        self.symbol_subscribers: Dict[str, Set[str]] = {}  # symbol -> client_ids
        self.user_subscribers: Dict[str, Set[str]] = {}   # user_id -> client_ids
        
        # Redis for pub/sub
        self.redis_client = None
        self.redis_subscriber = None
        
        # Data sources
        self.binance_ws = None
        self.external_feeds: Dict[str, Any] = {}
        
        # Configuration
        self.heartbeat_interval = 30  # seconds
        self.price_update_interval = 1  # seconds
        self.news_update_interval = 60  # seconds
        self.market_overview_interval = 300  # seconds
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        
        # Initialize when needed
        self.initialized = False
    
    async def _initialize(self):
        """Initialize the streaming service."""
        if self.initialized:
            return
            
        try:
            await self._init_redis()
            await self._start_background_tasks()
            logger.info("Real-time streaming service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize streaming service: {e}")
        finally:
            self.initialized = True
    
    async def _init_redis(self):
        """Initialize Redis for pub/sub messaging."""
        try:
            self.redis_client = aioredis.from_url(
                "redis://localhost:6379/1", 
                decode_responses=True
            )
            await self.redis_client.ping()
            
            # Create subscriber for internal messaging
            self.redis_subscriber = self.redis_client.pubsub()
            await self.redis_subscriber.subscribe(
                "price_updates",
                "news_updates", 
                "market_overview",
                "portfolio_updates",
                "ai_insights",
                "system_alerts"
            )
            
            logger.info("Redis pub/sub initialized")
        except Exception as e:
            logger.warning(f"Redis initialization failed: {e}")
    
    async def _start_background_tasks(self):
        """Start background data streaming tasks."""
        tasks = [
            self._price_streaming_task(),
            self._news_streaming_task(),
            self._market_overview_task(),
            self._heartbeat_task(),
            self._redis_message_handler(),
            self._external_feed_manager()
        ]
        
        for task_coro in tasks:
            task = asyncio.create_task(task_coro)
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)
    
    async def subscribe_client(self, websocket: WebSocket, client_id: str, 
                             symbols: List[str] = None, 
                             stream_types: List[str] = None,
                             user_id: str = None,
                             portfolio_id: str = None):
        """Subscribe a client to real-time streams."""
        # Initialize if not already done
        if not self.initialized:
            await self._initialize()
            
        try:
            await websocket.accept()
            
            # Convert stream types
            stream_type_enums = set()
            if stream_types:
                for st in stream_types:
                    try:
                        stream_type_enums.add(StreamType(st))
                    except ValueError:
                        logger.warning(f"Invalid stream type: {st}")
            else:
                stream_type_enums = {StreamType.PRICE_UPDATE, StreamType.NEWS_UPDATE}
            
            # Create subscription
            subscription = Subscription(
                client_id=client_id,
                websocket=websocket,
                symbols=set(symbols) if symbols else set(),
                stream_types=stream_type_enums,
                user_id=user_id,
                portfolio_id=portfolio_id
            )
            
            self.subscriptions[client_id] = subscription
            
            # Update symbol subscribers
            for symbol in subscription.symbols:
                if symbol not in self.symbol_subscribers:
                    self.symbol_subscribers[symbol] = set()
                self.symbol_subscribers[symbol].add(client_id)
            
            # Update user subscribers
            if user_id:
                if user_id not in self.user_subscribers:
                    self.user_subscribers[user_id] = set()
                self.user_subscribers[user_id].add(client_id)
            
            logger.info(f"Client {client_id} subscribed to {len(subscription.symbols)} symbols")
            
            # Send initial data
            await self._send_initial_data(subscription)
            
            # Handle client messages
            await self._handle_client_messages(subscription)
            
        except WebSocketDisconnect:
            await self.unsubscribe_client(client_id)
        except Exception as e:
            logger.error(f"Error in client subscription: {e}")
            await self.unsubscribe_client(client_id)
    
    async def unsubscribe_client(self, client_id: str):
        """Unsubscribe a client from all streams."""
        if client_id not in self.subscriptions:
            return
        
        subscription = self.subscriptions[client_id]
        
        # Remove from symbol subscribers
        for symbol in subscription.symbols:
            if symbol in self.symbol_subscribers:
                self.symbol_subscribers[symbol].discard(client_id)
                if not self.symbol_subscribers[symbol]:
                    del self.symbol_subscribers[symbol]
        
        # Remove from user subscribers
        if subscription.user_id and subscription.user_id in self.user_subscribers:
            self.user_subscribers[subscription.user_id].discard(client_id)
            if not self.user_subscribers[subscription.user_id]:
                del self.user_subscribers[subscription.user_id]
        
        # Close websocket if still open
        try:
            await subscription.websocket.close()
        except:
            pass
        
        del self.subscriptions[client_id]
        logger.info(f"Client {client_id} unsubscribed")
    
    async def _send_initial_data(self, subscription: Subscription):
        """Send initial data to newly subscribed client."""
        try:
            # Send current prices for subscribed symbols
            if subscription.symbols and StreamType.PRICE_UPDATE in subscription.stream_types:
                prices = await enhanced_market_data_service.get_real_time_prices(
                    list(subscription.symbols)
                )
                
                message = StreamMessage(
                    type=StreamType.PRICE_UPDATE,
                    data={
                        'prices': prices,
                        'initial': True
                    },
                    timestamp=datetime.now()
                )
                
                await self._send_to_client(subscription, message)
            
            # Send market overview if subscribed
            if StreamType.MARKET_OVERVIEW in subscription.stream_types:
                market_overview = await self._get_market_overview()
                
                message = StreamMessage(
                    type=StreamType.MARKET_OVERVIEW,
                    data=market_overview,
                    timestamp=datetime.now()
                )
                
                await self._send_to_client(subscription, message)
                
        except Exception as e:
            logger.error(f"Error sending initial data: {e}")
    
    async def _handle_client_messages(self, subscription: Subscription):
        """Handle incoming messages from client."""
        try:
            while True:
                message = await subscription.websocket.receive_text()
                data = json.loads(message)
                
                message_type = data.get('type')
                
                if message_type == 'heartbeat':
                    subscription.last_heartbeat = datetime.now()
                    await subscription.websocket.send_text(json.dumps({
                        'type': 'heartbeat_ack',
                        'timestamp': datetime.now().isoformat()
                    }))
                
                elif message_type == 'subscribe_symbols':
                    new_symbols = set(data.get('symbols', []))
                    await self._update_symbol_subscription(subscription, new_symbols)
                
                elif message_type == 'unsubscribe_symbols':
                    remove_symbols = set(data.get('symbols', []))
                    await self._remove_symbol_subscription(subscription, remove_symbols)
                
                elif message_type == 'update_stream_types':
                    new_stream_types = set()
                    for st in data.get('stream_types', []):
                        try:
                            new_stream_types.add(StreamType(st))
                        except ValueError:
                            pass
                    subscription.stream_types = new_stream_types
                
        except WebSocketDisconnect:
            raise
        except Exception as e:
            logger.error(f"Error handling client message: {e}")
    
    async def _update_symbol_subscription(self, subscription: Subscription, new_symbols: Set[str]):
        """Update symbol subscription for a client."""
        # Remove old symbols
        for symbol in subscription.symbols - new_symbols:
            if symbol in self.symbol_subscribers:
                self.symbol_subscribers[symbol].discard(subscription.client_id)
                if not self.symbol_subscribers[symbol]:
                    del self.symbol_subscribers[symbol]
        
        # Add new symbols
        for symbol in new_symbols - subscription.symbols:
            if symbol not in self.symbol_subscribers:
                self.symbol_subscribers[symbol] = set()
            self.symbol_subscribers[symbol].add(subscription.client_id)
        
        subscription.symbols = new_symbols
        
        # Send initial prices for new symbols
        if new_symbols:
            prices = await enhanced_market_data_service.get_real_time_prices(
                list(new_symbols)
            )
            
            message = StreamMessage(
                type=StreamType.PRICE_UPDATE,
                data={'prices': prices},
                timestamp=datetime.now()
            )
            
            await self._send_to_client(subscription, message)
    
    async def _remove_symbol_subscription(self, subscription: Subscription, remove_symbols: Set[str]):
        """Remove symbols from client subscription."""
        for symbol in remove_symbols:
            subscription.symbols.discard(symbol)
            if symbol in self.symbol_subscribers:
                self.symbol_subscribers[symbol].discard(subscription.client_id)
                if not self.symbol_subscribers[symbol]:
                    del self.symbol_subscribers[symbol]
    
    async def _price_streaming_task(self):
        """Background task for streaming price updates."""
        while True:
            try:
                if self.symbol_subscribers:
                    # Get all subscribed symbols
                    all_symbols = set(self.symbol_subscribers.keys())
                    
                    if all_symbols:
                        # Fetch current prices
                        prices = await enhanced_market_data_service.get_real_time_prices(
                            list(all_symbols)
                        )
                        
                        # Send updates to subscribers
                        for symbol, price_data in prices.items():
                            if symbol in self.symbol_subscribers:
                                message = StreamMessage(
                                    type=StreamType.PRICE_UPDATE,
                                    data={
                                        'symbol': symbol,
                                        'price_data': price_data
                                    },
                                    timestamp=datetime.now()
                                )
                                
                                await self._broadcast_to_symbol_subscribers(symbol, message)
                
                await asyncio.sleep(self.price_update_interval)
                
            except Exception as e:
                logger.error(f"Error in price streaming task: {e}")
                await asyncio.sleep(5)
    
    async def _news_streaming_task(self):
        """Background task for streaming news updates."""
        while True:
            try:
                # This would integrate with news service
                # For now, we'll create a placeholder
                
                if self.subscriptions:
                    # Get latest news (placeholder)
                    news_data = {
                        'headline': 'Market Update',
                        'summary': 'Latest market developments...',
                        'timestamp': datetime.now().isoformat(),
                        'source': 'FinScope News'
                    }
                    
                    message = StreamMessage(
                        type=StreamType.NEWS_UPDATE,
                        data=news_data,
                        timestamp=datetime.now()
                    )
                    
                    await self._broadcast_to_all_subscribers(message, StreamType.NEWS_UPDATE)
                
                await asyncio.sleep(self.news_update_interval)
                
            except Exception as e:
                logger.error(f"Error in news streaming task: {e}")
                await asyncio.sleep(30)
    
    async def _market_overview_task(self):
        """Background task for streaming market overview updates."""
        while True:
            try:
                if self.subscriptions:
                    market_overview = await self._get_market_overview()
                    
                    message = StreamMessage(
                        type=StreamType.MARKET_OVERVIEW,
                        data=market_overview,
                        timestamp=datetime.now()
                    )
                    
                    await self._broadcast_to_all_subscribers(message, StreamType.MARKET_OVERVIEW)
                
                await asyncio.sleep(self.market_overview_interval)
                
            except Exception as e:
                logger.error(f"Error in market overview task: {e}")
                await asyncio.sleep(60)
    
    async def _heartbeat_task(self):
        """Background task for client heartbeat monitoring."""
        while True:
            try:
                current_time = datetime.now()
                stale_clients = []
                
                for client_id, subscription in self.subscriptions.items():
                    if (current_time - subscription.last_heartbeat).seconds > self.heartbeat_interval * 2:
                        stale_clients.append(client_id)
                
                # Remove stale clients
                for client_id in stale_clients:
                    logger.info(f"Removing stale client: {client_id}")
                    await self.unsubscribe_client(client_id)
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in heartbeat task: {e}")
                await asyncio.sleep(30)
    
    async def _redis_message_handler(self):
        """Handle messages from Redis pub/sub."""
        if not self.redis_subscriber:
            return
        
        try:
            async for message in self.redis_subscriber.listen():
                if message['type'] == 'message':
                    channel = message['channel']
                    data = json.loads(message['data'])
                    
                    # Route message based on channel
                    if channel == 'price_updates':
                        await self._handle_redis_price_update(data)
                    elif channel == 'news_updates':
                        await self._handle_redis_news_update(data)
                    elif channel == 'portfolio_updates':
                        await self._handle_redis_portfolio_update(data)
                    elif channel == 'ai_insights':
                        await self._handle_redis_ai_insight(data)
                    elif channel == 'system_alerts':
                        await self._handle_redis_system_alert(data)
                        
        except Exception as e:
            logger.error(f"Error in Redis message handler: {e}")
    
    async def _external_feed_manager(self):
        """Manage external data feed connections."""
        while True:
            try:
                # This would manage connections to external WebSocket feeds
                # like Binance, Alpha Vantage, etc.
                
                # For now, just a placeholder
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in external feed manager: {e}")
                await asyncio.sleep(30)
    
    async def _get_market_overview(self) -> Dict[str, Any]:
        """Get current market overview data."""
        try:
            # Get major indices and market stats
            major_symbols = ['BTC', 'ETH', 'AAPL', 'MSFT', 'GOOGL']
            prices = await enhanced_market_data_service.get_real_time_prices(major_symbols)
            
            return {
                'major_assets': prices,
                'market_status': 'open',  # This would be calculated
                'total_market_cap': 2500000000000,  # Placeholder
                'fear_greed_index': 65,  # Placeholder
                'trending_assets': major_symbols[:3]
            }
        except Exception as e:
            logger.error(f"Error getting market overview: {e}")
            return {}
    
    async def _send_to_client(self, subscription: Subscription, message: StreamMessage):
        """Send message to a specific client."""
        try:
            if message.type in subscription.stream_types:
                await subscription.websocket.send_text(json.dumps(message.to_dict()))
        except Exception as e:
            logger.error(f"Error sending to client {subscription.client_id}: {e}")
            await self.unsubscribe_client(subscription.client_id)
    
    async def _broadcast_to_symbol_subscribers(self, symbol: str, message: StreamMessage):
        """Broadcast message to all subscribers of a symbol."""
        if symbol not in self.symbol_subscribers:
            return
        
        for client_id in list(self.symbol_subscribers[symbol]):
            if client_id in self.subscriptions:
                await self._send_to_client(self.subscriptions[client_id], message)
    
    async def _broadcast_to_all_subscribers(self, message: StreamMessage, stream_type: StreamType):
        """Broadcast message to all subscribers of a stream type."""
        for subscription in list(self.subscriptions.values()):
            if stream_type in subscription.stream_types:
                await self._send_to_client(subscription, message)
    
    async def _broadcast_to_user_subscribers(self, user_id: str, message: StreamMessage):
        """Broadcast message to all clients of a specific user."""
        if user_id not in self.user_subscribers:
            return
        
        for client_id in list(self.user_subscribers[user_id]):
            if client_id in self.subscriptions:
                await self._send_to_client(self.subscriptions[client_id], message)
    
    # Redis message handlers
    async def _handle_redis_price_update(self, data: Dict[str, Any]):
        """Handle price update from Redis."""
        symbol = data.get('symbol')
        if symbol and symbol in self.symbol_subscribers:
            message = StreamMessage(
                type=StreamType.PRICE_UPDATE,
                data=data,
                timestamp=datetime.now()
            )
            await self._broadcast_to_symbol_subscribers(symbol, message)
    
    async def _handle_redis_news_update(self, data: Dict[str, Any]):
        """Handle news update from Redis."""
        message = StreamMessage(
            type=StreamType.NEWS_UPDATE,
            data=data,
            timestamp=datetime.now()
        )
        await self._broadcast_to_all_subscribers(message, StreamType.NEWS_UPDATE)
    
    async def _handle_redis_portfolio_update(self, data: Dict[str, Any]):
        """Handle portfolio update from Redis."""
        user_id = data.get('user_id')
        if user_id:
            message = StreamMessage(
                type=StreamType.PORTFOLIO_UPDATE,
                data=data,
                timestamp=datetime.now()
            )
            await self._broadcast_to_user_subscribers(user_id, message)
    
    async def _handle_redis_ai_insight(self, data: Dict[str, Any]):
        """Handle AI insight from Redis."""
        message = StreamMessage(
            type=StreamType.AI_INSIGHT,
            data=data,
            timestamp=datetime.now()
        )
        await self._broadcast_to_all_subscribers(message, StreamType.AI_INSIGHT)
    
    async def _handle_redis_system_alert(self, data: Dict[str, Any]):
        """Handle system alert from Redis."""
        message = StreamMessage(
            type=StreamType.SYSTEM_ALERT,
            data=data,
            timestamp=datetime.now()
        )
        await self._broadcast_to_all_subscribers(message, StreamType.SYSTEM_ALERT)
    
    # Public API methods
    async def publish_price_update(self, symbol: str, price_data: Dict[str, Any]):
        """Publish price update to Redis."""
        if self.redis_client:
            await self.redis_client.publish(
                'price_updates',
                json.dumps({'symbol': symbol, **price_data})
            )
    
    async def publish_news_update(self, news_data: Dict[str, Any]):
        """Publish news update to Redis."""
        if self.redis_client:
            await self.redis_client.publish('news_updates', json.dumps(news_data))
    
    async def publish_portfolio_update(self, user_id: str, portfolio_data: Dict[str, Any]):
        """Publish portfolio update to Redis."""
        if self.redis_client:
            await self.redis_client.publish(
                'portfolio_updates',
                json.dumps({'user_id': user_id, **portfolio_data})
            )
    
    async def publish_ai_insight(self, insight_data: Dict[str, Any]):
        """Publish AI insight to Redis."""
        if self.redis_client:
            await self.redis_client.publish('ai_insights', json.dumps(insight_data))
    
    async def publish_system_alert(self, alert_data: Dict[str, Any]):
        """Publish system alert to Redis."""
        if self.redis_client:
            await self.redis_client.publish('system_alerts', json.dumps(alert_data))
    
    async def get_active_subscriptions(self) -> Dict[str, Any]:
        """Get information about active subscriptions."""
        return {
            'total_clients': len(self.subscriptions),
            'total_symbols': len(self.symbol_subscribers),
            'clients_by_symbol': {symbol: len(clients) for symbol, clients in self.symbol_subscribers.items()},
            'clients_by_user': {user_id: len(clients) for user_id, clients in self.user_subscribers.items()}
        }
    
    async def cleanup(self):
        """Cleanup resources."""
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Close all client connections
        for client_id in list(self.subscriptions.keys()):
            await self.unsubscribe_client(client_id)
        
        # Close Redis connections
        if self.redis_subscriber:
            await self.redis_subscriber.unsubscribe()
            await self.redis_subscriber.close()
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Real-time streaming service cleaned up")

# Create global instance
real_time_streaming_service = RealTimeStreamingService()