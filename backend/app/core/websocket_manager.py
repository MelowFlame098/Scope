"""WebSocket connection manager for real-time communication.

This module provides:
- WebSocket connection management
- Real-time data broadcasting
- User-specific messaging
- Connection lifecycle management
- Message routing and filtering
- Authentication integration
"""

from typing import Dict, List, Set, Optional, Any, Callable, Union
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
import json
import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import uuid
from collections import defaultdict
import weakref

from app.core.logging_config import get_logger
from app.core.exceptions import FinScopeException

logger = get_logger("websocket")


class MessageType(str, Enum):
    """WebSocket message types."""
    # System messages
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    PING = "ping"
    PONG = "pong"
    ERROR = "error"
    
    # Market data
    MARKET_DATA = "market_data"
    PRICE_UPDATE = "price_update"
    VOLUME_UPDATE = "volume_update"
    
    # Portfolio updates
    PORTFOLIO_UPDATE = "portfolio_update"
    POSITION_UPDATE = "position_update"
    BALANCE_UPDATE = "balance_update"
    
    # Trading
    ORDER_UPDATE = "order_update"
    TRADE_EXECUTION = "trade_execution"
    
    # Notifications
    ALERT = "alert"
    NOTIFICATION = "notification"
    
    # AI/Analysis
    ANALYSIS_UPDATE = "analysis_update"
    PREDICTION_UPDATE = "prediction_update"
    
    # DeFi
    DEFI_UPDATE = "defi_update"
    YIELD_UPDATE = "yield_update"
    
    # Custom
    CUSTOM = "custom"


class ConnectionStatus(str, Enum):
    """WebSocket connection status."""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"


class WebSocketMessage(BaseModel):
    """WebSocket message model."""
    type: MessageType
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    channel: Optional[str] = None
    
    def to_json(self) -> str:
        """Convert message to JSON string.
        
        Returns:
            JSON string representation
        """
        return self.model_dump_json()
    
    @classmethod
    def from_json(cls, json_str: str) -> "WebSocketMessage":
        """Create message from JSON string.
        
        Args:
            json_str: JSON string
            
        Returns:
            WebSocket message instance
        """
        return cls.model_validate_json(json_str)


@dataclass
class ConnectionInfo:
    """WebSocket connection information."""
    connection_id: str
    websocket: WebSocket
    user_id: Optional[str] = None
    channels: Set[str] = field(default_factory=set)
    connected_at: datetime = field(default_factory=datetime.utcnow)
    last_ping: Optional[datetime] = None
    status: ConnectionStatus = ConnectionStatus.CONNECTING
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization setup."""
        if isinstance(self.channels, list):
            self.channels = set(self.channels)
    
    @property
    def is_active(self) -> bool:
        """Check if connection is active.
        
        Returns:
            True if connection is active
        """
        return self.status == ConnectionStatus.CONNECTED
    
    @property
    def connection_duration(self) -> timedelta:
        """Get connection duration.
        
        Returns:
            Duration since connection established
        """
        return datetime.utcnow() - self.connected_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "connection_id": self.connection_id,
            "user_id": self.user_id,
            "channels": list(self.channels),
            "connected_at": self.connected_at.isoformat(),
            "last_ping": self.last_ping.isoformat() if self.last_ping else None,
            "status": self.status.value,
            "connection_duration_seconds": self.connection_duration.total_seconds(),
            "metadata": self.metadata
        }


class WebSocketManager:
    """WebSocket connection manager."""
    
    def __init__(self):
        """Initialize WebSocket manager."""
        # Connection storage
        self._connections: Dict[str, ConnectionInfo] = {}
        self._user_connections: Dict[str, Set[str]] = defaultdict(set)
        self._channel_connections: Dict[str, Set[str]] = defaultdict(set)
        
        # Message handlers
        self._message_handlers: Dict[MessageType, List[Callable]] = defaultdict(list)
        
        # Configuration
        self._ping_interval = 30  # seconds
        self._ping_timeout = 10   # seconds
        self._max_connections_per_user = 5
        self._max_message_size = 1024 * 1024  # 1MB
        
        # Statistics
        self._total_connections = 0
        self._total_messages_sent = 0
        self._total_messages_received = 0
        
        # Background tasks
        self._ping_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        logger.info("WebSocket manager initialized")
    
    async def connect(
        self,
        websocket: WebSocket,
        user_id: Optional[str] = None,
        channels: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Accept a new WebSocket connection.
        
        Args:
            websocket: WebSocket instance
            user_id: User ID for authenticated connections
            channels: Initial channels to subscribe to
            metadata: Additional connection metadata
            
        Returns:
            Connection ID
            
        Raises:
            FinScopeException: If connection limit exceeded
        """
        # Check connection limits
        if user_id and len(self._user_connections[user_id]) >= self._max_connections_per_user:
            raise FinScopeException(
                f"Maximum connections per user ({self._max_connections_per_user}) exceeded"
            )
        
        # Accept WebSocket connection
        await websocket.accept()
        
        # Create connection info
        connection_id = str(uuid.uuid4())
        connection_info = ConnectionInfo(
            connection_id=connection_id,
            websocket=websocket,
            user_id=user_id,
            channels=set(channels or []),
            metadata=metadata or {},
            status=ConnectionStatus.CONNECTED
        )
        
        # Store connection
        self._connections[connection_id] = connection_info
        
        # Update user connections
        if user_id:
            self._user_connections[user_id].add(connection_id)
        
        # Subscribe to channels
        for channel in connection_info.channels:
            self._channel_connections[channel].add(connection_id)
        
        # Update statistics
        self._total_connections += 1
        
        # Send connection confirmation
        await self._send_to_connection(
            connection_id,
            WebSocketMessage(
                type=MessageType.CONNECT,
                data={
                    "connection_id": connection_id,
                    "status": "connected",
                    "channels": list(connection_info.channels)
                }
            )
        )
        
        logger.info(
            f"WebSocket connected: {connection_id} "
            f"(user: {user_id}, channels: {connection_info.channels})"
        )
        
        # Start background tasks if not running
        if not self._ping_task:
            self._ping_task = asyncio.create_task(self._ping_connections())
        if not self._cleanup_task:
            self._cleanup_task = asyncio.create_task(self._cleanup_connections())
        
        return connection_id
    
    async def disconnect(self, connection_id: str, code: int = 1000) -> None:
        """Disconnect a WebSocket connection.
        
        Args:
            connection_id: Connection ID
            code: WebSocket close code
        """
        connection_info = self._connections.get(connection_id)
        if not connection_info:
            return
        
        # Update status
        connection_info.status = ConnectionStatus.DISCONNECTING
        
        try:
            # Send disconnect message
            await self._send_to_connection(
                connection_id,
                WebSocketMessage(
                    type=MessageType.DISCONNECT,
                    data={"reason": "Server initiated disconnect"}
                )
            )
            
            # Close WebSocket
            await connection_info.websocket.close(code=code)
            
        except Exception as e:
            logger.warning(f"Error during disconnect: {e}")
        
        finally:
            # Clean up connection
            await self._cleanup_connection(connection_id)
    
    async def _cleanup_connection(self, connection_id: str) -> None:
        """Clean up a connection.
        
        Args:
            connection_id: Connection ID
        """
        connection_info = self._connections.get(connection_id)
        if not connection_info:
            return
        
        # Remove from user connections
        if connection_info.user_id:
            self._user_connections[connection_info.user_id].discard(connection_id)
            if not self._user_connections[connection_info.user_id]:
                del self._user_connections[connection_info.user_id]
        
        # Remove from channel connections
        for channel in connection_info.channels:
            self._channel_connections[channel].discard(connection_id)
            if not self._channel_connections[channel]:
                del self._channel_connections[channel]
        
        # Remove connection
        del self._connections[connection_id]
        
        logger.info(
            f"WebSocket disconnected: {connection_id} "
            f"(user: {connection_info.user_id})"
        )
    
    async def send_to_user(
        self,
        user_id: str,
        message: WebSocketMessage
    ) -> int:
        """Send message to all connections of a user.
        
        Args:
            user_id: User ID
            message: Message to send
            
        Returns:
            Number of connections message was sent to
        """
        connection_ids = self._user_connections.get(user_id, set())
        sent_count = 0
        
        for connection_id in connection_ids.copy():
            if await self._send_to_connection(connection_id, message):
                sent_count += 1
        
        return sent_count
    
    async def send_to_channel(
        self,
        channel: str,
        message: WebSocketMessage,
        exclude_user: Optional[str] = None
    ) -> int:
        """Send message to all connections in a channel.
        
        Args:
            channel: Channel name
            message: Message to send
            exclude_user: User ID to exclude from broadcast
            
        Returns:
            Number of connections message was sent to
        """
        connection_ids = self._channel_connections.get(channel, set())
        sent_count = 0
        
        for connection_id in connection_ids.copy():
            connection_info = self._connections.get(connection_id)
            if connection_info and connection_info.user_id != exclude_user:
                if await self._send_to_connection(connection_id, message):
                    sent_count += 1
        
        return sent_count
    
    async def broadcast(
        self,
        message: WebSocketMessage,
        exclude_user: Optional[str] = None
    ) -> int:
        """Broadcast message to all connections.
        
        Args:
            message: Message to send
            exclude_user: User ID to exclude from broadcast
            
        Returns:
            Number of connections message was sent to
        """
        sent_count = 0
        
        for connection_id, connection_info in self._connections.items():
            if connection_info.user_id != exclude_user:
                if await self._send_to_connection(connection_id, message):
                    sent_count += 1
        
        return sent_count
    
    async def _send_to_connection(
        self,
        connection_id: str,
        message: WebSocketMessage
    ) -> bool:
        """Send message to a specific connection.
        
        Args:
            connection_id: Connection ID
            message: Message to send
            
        Returns:
            True if message was sent successfully
        """
        connection_info = self._connections.get(connection_id)
        if not connection_info or not connection_info.is_active:
            return False
        
        try:
            # Set user ID in message if not set
            if not message.user_id and connection_info.user_id:
                message.user_id = connection_info.user_id
            
            # Send message
            message_json = message.to_json()
            
            # Check message size
            if len(message_json.encode('utf-8')) > self._max_message_size:
                logger.warning(f"Message too large for connection {connection_id}")
                return False
            
            await connection_info.websocket.send_text(message_json)
            self._total_messages_sent += 1
            
            return True
            
        except Exception as e:
            logger.warning(f"Failed to send message to {connection_id}: {e}")
            # Mark connection for cleanup
            connection_info.status = ConnectionStatus.ERROR
            return False
    
    async def subscribe_to_channel(
        self,
        connection_id: str,
        channel: str
    ) -> bool:
        """Subscribe connection to a channel.
        
        Args:
            connection_id: Connection ID
            channel: Channel name
            
        Returns:
            True if subscription was successful
        """
        connection_info = self._connections.get(connection_id)
        if not connection_info:
            return False
        
        connection_info.channels.add(channel)
        self._channel_connections[channel].add(connection_id)
        
        logger.debug(f"Connection {connection_id} subscribed to channel {channel}")
        return True
    
    async def unsubscribe_from_channel(
        self,
        connection_id: str,
        channel: str
    ) -> bool:
        """Unsubscribe connection from a channel.
        
        Args:
            connection_id: Connection ID
            channel: Channel name
            
        Returns:
            True if unsubscription was successful
        """
        connection_info = self._connections.get(connection_id)
        if not connection_info:
            return False
        
        connection_info.channels.discard(channel)
        self._channel_connections[channel].discard(connection_id)
        
        # Clean up empty channel
        if not self._channel_connections[channel]:
            del self._channel_connections[channel]
        
        logger.debug(f"Connection {connection_id} unsubscribed from channel {channel}")
        return True
    
    async def handle_message(
        self,
        connection_id: str,
        message_data: str
    ) -> None:
        """Handle incoming WebSocket message.
        
        Args:
            connection_id: Connection ID
            message_data: Raw message data
        """
        try:
            # Parse message
            message = WebSocketMessage.from_json(message_data)
            self._total_messages_received += 1
            
            # Update connection info
            connection_info = self._connections.get(connection_id)
            if connection_info:
                if message.type == MessageType.PING:
                    connection_info.last_ping = datetime.utcnow()
                    # Send pong response
                    await self._send_to_connection(
                        connection_id,
                        WebSocketMessage(type=MessageType.PONG)
                    )
                    return
            
            # Call message handlers
            handlers = self._message_handlers.get(message.type, [])
            for handler in handlers:
                try:
                    await handler(connection_id, message)
                except Exception as e:
                    logger.error(f"Message handler error: {e}")
            
        except Exception as e:
            logger.error(f"Error handling message from {connection_id}: {e}")
            # Send error response
            await self._send_to_connection(
                connection_id,
                WebSocketMessage(
                    type=MessageType.ERROR,
                    data={"error": "Invalid message format"}
                )
            )
    
    def add_message_handler(
        self,
        message_type: MessageType,
        handler: Callable[[str, WebSocketMessage], None]
    ) -> None:
        """Add message handler.
        
        Args:
            message_type: Message type to handle
            handler: Handler function
        """
        self._message_handlers[message_type].append(handler)
        logger.debug(f"Added handler for message type: {message_type}")
    
    def remove_message_handler(
        self,
        message_type: MessageType,
        handler: Callable[[str, WebSocketMessage], None]
    ) -> None:
        """Remove message handler.
        
        Args:
            message_type: Message type
            handler: Handler function to remove
        """
        if handler in self._message_handlers[message_type]:
            self._message_handlers[message_type].remove(handler)
            logger.debug(f"Removed handler for message type: {message_type}")
    
    async def _ping_connections(self) -> None:
        """Background task to ping connections."""
        while True:
            try:
                await asyncio.sleep(self._ping_interval)
                
                current_time = datetime.utcnow()
                ping_message = WebSocketMessage(type=MessageType.PING)
                
                for connection_id, connection_info in list(self._connections.items()):
                    if connection_info.is_active:
                        # Check if connection is responsive
                        if (connection_info.last_ping and 
                            current_time - connection_info.last_ping > timedelta(seconds=self._ping_timeout * 2)):
                            logger.warning(f"Connection {connection_id} appears unresponsive")
                            connection_info.status = ConnectionStatus.ERROR
                        else:
                            await self._send_to_connection(connection_id, ping_message)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in ping task: {e}")
    
    async def _cleanup_connections(self) -> None:
        """Background task to clean up dead connections."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                dead_connections = []
                for connection_id, connection_info in self._connections.items():
                    if connection_info.status == ConnectionStatus.ERROR:
                        dead_connections.append(connection_id)
                
                for connection_id in dead_connections:
                    await self._cleanup_connection(connection_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
    
    def get_connection_info(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Get connection information.
        
        Args:
            connection_id: Connection ID
            
        Returns:
            Connection information or None
        """
        connection_info = self._connections.get(connection_id)
        return connection_info.to_dict() if connection_info else None
    
    def get_user_connections(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all connections for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of connection information
        """
        connection_ids = self._user_connections.get(user_id, set())
        return [
            self._connections[conn_id].to_dict()
            for conn_id in connection_ids
            if conn_id in self._connections
        ]
    
    def get_channel_connections(self, channel: str) -> List[Dict[str, Any]]:
        """Get all connections in a channel.
        
        Args:
            channel: Channel name
            
        Returns:
            List of connection information
        """
        connection_ids = self._channel_connections.get(channel, set())
        return [
            self._connections[conn_id].to_dict()
            for conn_id in connection_ids
            if conn_id in self._connections
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get WebSocket manager statistics.
        
        Returns:
            Statistics dictionary
        """
        active_connections = sum(
            1 for conn in self._connections.values()
            if conn.is_active
        )
        
        return {
            "total_connections": self._total_connections,
            "active_connections": active_connections,
            "total_users": len(self._user_connections),
            "total_channels": len(self._channel_connections),
            "messages_sent": self._total_messages_sent,
            "messages_received": self._total_messages_received,
            "connections_by_status": {
                status.value: sum(
                    1 for conn in self._connections.values()
                    if conn.status == status
                )
                for status in ConnectionStatus
            }
        }
    
    async def shutdown(self) -> None:
        """Shutdown WebSocket manager."""
        logger.info("Shutting down WebSocket manager")
        
        # Cancel background tasks
        if self._ping_task:
            self._ping_task.cancel()
            try:
                await self._ping_task
            except asyncio.CancelledError:
                pass
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Disconnect all connections
        for connection_id in list(self._connections.keys()):
            await self.disconnect(connection_id)
        
        logger.info("WebSocket manager shutdown complete")


# Global WebSocket manager instance
websocket_manager = WebSocketManager()


# Utility functions
async def handle_websocket_connection(
    websocket: WebSocket,
    user_id: Optional[str] = None,
    channels: Optional[List[str]] = None
) -> None:
    """Handle WebSocket connection lifecycle.
    
    Args:
        websocket: WebSocket instance
        user_id: User ID for authenticated connections
        channels: Initial channels to subscribe to
    """
    connection_id = None
    try:
        # Connect
        connection_id = await websocket_manager.connect(
            websocket, user_id, channels
        )
        
        # Handle messages
        while True:
            try:
                data = await websocket.receive_text()
                await websocket_manager.handle_message(connection_id, data)
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
                break
    
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    
    finally:
        # Clean up connection
        if connection_id:
            await websocket_manager._cleanup_connection(connection_id)


def get_websocket_manager() -> WebSocketManager:
    """Get WebSocket manager instance.
    
    Returns:
        WebSocket manager
    """
    return websocket_manager