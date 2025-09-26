"""Conversation Manager for FinScope - Phase 6 Implementation

Manages conversation history, context, and memory for LLM interactions.
Provides persistent storage and retrieval of chat sessions.
"""

import json
import logging
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

import aioredis
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class ConversationStatus(str, Enum):
    """Conversation status types"""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"

class MessageRole(str, Enum):
    """Message role types"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

@dataclass
class ConversationMessage:
    """Individual message in a conversation"""
    id: str
    role: MessageRole
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = None
    token_count: Optional[int] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)

@dataclass
class ConversationContext:
    """Conversation context and metadata"""
    user_id: str
    session_id: str
    topic: Optional[str] = None
    complexity_level: str = "intermediate"
    portfolio_context: Dict[str, Any] = None
    market_context: Dict[str, Any] = None
    preferences: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.portfolio_context is None:
            self.portfolio_context = {}
        if self.market_context is None:
            self.market_context = {}
        if self.preferences is None:
            self.preferences = {}

@dataclass
class Conversation:
    """Complete conversation object"""
    id: str
    context: ConversationContext
    messages: List[ConversationMessage]
    status: ConversationStatus
    created_at: datetime
    updated_at: datetime
    summary: Optional[str] = None
    total_tokens: int = 0
    
    def __post_init__(self):
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
        if isinstance(self.updated_at, str):
            self.updated_at = datetime.fromisoformat(self.updated_at)

class ConversationRequest(BaseModel):
    """Request model for conversation operations"""
    user_id: str
    message: str
    conversation_id: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    complexity_level: str = "intermediate"
    topic: Optional[str] = None

class ConversationResponse(BaseModel):
    """Response model for conversation operations"""
    conversation_id: str
    message_id: str
    response: str
    context_updated: bool = False
    suggestions: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ConversationManager:
    """Manages conversation history and context for LLM interactions"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client: Optional[aioredis.Redis] = None
        self.conversations: Dict[str, Conversation] = {}  # In-memory cache
        self.max_conversation_length = 50  # Maximum messages per conversation
        self.context_window_size = 10  # Messages to include in context
        self.conversation_ttl = timedelta(days=30)  # Conversation expiry
    
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("Connected to Redis for conversation management")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using in-memory storage.")
            self.redis_client = None
    
    async def create_conversation(
        self,
        user_id: str,
        initial_context: Optional[Dict[str, Any]] = None,
        topic: Optional[str] = None
    ) -> str:
        """Create a new conversation"""
        conversation_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        
        context = ConversationContext(
            user_id=user_id,
            session_id=session_id,
            topic=topic,
            complexity_level="intermediate"
        )
        
        if initial_context:
            context.portfolio_context.update(
                initial_context.get("portfolio", {})
            )
            context.market_context.update(
                initial_context.get("market", {})
            )
            context.preferences.update(
                initial_context.get("preferences", {})
            )
            if "complexity_level" in initial_context:
                context.complexity_level = initial_context["complexity_level"]
        
        conversation = Conversation(
            id=conversation_id,
            context=context,
            messages=[],
            status=ConversationStatus.ACTIVE,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        await self._save_conversation(conversation)
        return conversation_id
    
    async def add_message(
        self,
        conversation_id: str,
        role: MessageRole,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        token_count: Optional[int] = None
    ) -> str:
        """Add a message to a conversation"""
        conversation = await self.get_conversation(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        message_id = str(uuid.uuid4())
        message = ConversationMessage(
            id=message_id,
            role=role,
            content=content,
            timestamp=datetime.utcnow(),
            metadata=metadata or {},
            token_count=token_count
        )
        
        conversation.messages.append(message)
        conversation.updated_at = datetime.utcnow()
        
        if token_count:
            conversation.total_tokens += token_count
        
        # Trim conversation if it gets too long
        if len(conversation.messages) > self.max_conversation_length:
            # Keep system messages and recent messages
            system_messages = [
                msg for msg in conversation.messages
                if msg.role == MessageRole.SYSTEM
            ]
            recent_messages = conversation.messages[-(self.max_conversation_length - len(system_messages)):]
            conversation.messages = system_messages + recent_messages
        
        await self._save_conversation(conversation)
        return message_id
    
    async def add_interaction(
        self,
        conversation_id: str,
        user_message: str,
        assistant_response: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """Add a complete user-assistant interaction"""
        user_msg_id = await self.add_message(
            conversation_id,
            MessageRole.USER,
            user_message,
            metadata=context
        )
        
        assistant_msg_id = await self.add_message(
            conversation_id,
            MessageRole.ASSISTANT,
            assistant_response
        )
        
        return {
            "user_message_id": user_msg_id,
            "assistant_message_id": assistant_msg_id
        }
    
    async def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Retrieve a conversation by ID"""
        # Check in-memory cache first
        if conversation_id in self.conversations:
            return self.conversations[conversation_id]
        
        # Try Redis if available
        if self.redis_client:
            try:
                data = await self.redis_client.get(f"conversation:{conversation_id}")
                if data:
                    conv_dict = json.loads(data)
                    conversation = self._dict_to_conversation(conv_dict)
                    self.conversations[conversation_id] = conversation
                    return conversation
            except Exception as e:
                logger.error(f"Error retrieving conversation from Redis: {e}")
        
        return None
    
    async def get_context(
        self,
        conversation_id: str,
        include_recent_messages: bool = True
    ) -> Dict[str, Any]:
        """Get conversation context for prompt generation"""
        conversation = await self.get_conversation(conversation_id)
        if not conversation:
            return {}
        
        context = {
            "user_id": conversation.context.user_id,
            "topic": conversation.context.topic,
            "complexity_level": conversation.context.complexity_level,
            "portfolio_context": conversation.context.portfolio_context,
            "market_context": conversation.context.market_context,
            "preferences": conversation.context.preferences
        }
        
        if include_recent_messages and conversation.messages:
            # Get recent messages for context
            recent_messages = conversation.messages[-self.context_window_size:]
            context["recent_messages"] = [
                {
                    "role": msg.role.value,
                    "content": msg.content[:200] + "..." if len(msg.content) > 200 else msg.content,
                    "timestamp": msg.timestamp.isoformat()
                }
                for msg in recent_messages
            ]
            
            # Generate conversation summary
            context["summary"] = await self._generate_conversation_summary(conversation)
        
        return context
    
    async def update_context(
        self,
        conversation_id: str,
        context_updates: Dict[str, Any]
    ) -> bool:
        """Update conversation context"""
        conversation = await self.get_conversation(conversation_id)
        if not conversation:
            return False
        
        # Update context fields
        if "complexity_level" in context_updates:
            conversation.context.complexity_level = context_updates["complexity_level"]
        
        if "topic" in context_updates:
            conversation.context.topic = context_updates["topic"]
        
        if "portfolio" in context_updates:
            conversation.context.portfolio_context.update(context_updates["portfolio"])
        
        if "market" in context_updates:
            conversation.context.market_context.update(context_updates["market"])
        
        if "preferences" in context_updates:
            conversation.context.preferences.update(context_updates["preferences"])
        
        conversation.updated_at = datetime.utcnow()
        await self._save_conversation(conversation)
        return True
    
    async def get_user_conversations(
        self,
        user_id: str,
        status: Optional[ConversationStatus] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get conversations for a user"""
        conversations = []
        
        if self.redis_client:
            try:
                # Get conversation IDs for user
                pattern = f"conversation:*"
                keys = await self.redis_client.keys(pattern)
                
                for key in keys[:limit]:  # Limit for performance
                    data = await self.redis_client.get(key)
                    if data:
                        conv_dict = json.loads(data)
                        if conv_dict["context"]["user_id"] == user_id:
                            if not status or conv_dict["status"] == status.value:
                                conversations.append({
                                    "id": conv_dict["id"],
                                    "topic": conv_dict["context"].get("topic"),
                                    "status": conv_dict["status"],
                                    "created_at": conv_dict["created_at"],
                                    "updated_at": conv_dict["updated_at"],
                                    "message_count": len(conv_dict["messages"]),
                                    "total_tokens": conv_dict.get("total_tokens", 0)
                                })
            except Exception as e:
                logger.error(f"Error retrieving user conversations: {e}")
        else:
            # Use in-memory cache
            for conv in self.conversations.values():
                if conv.context.user_id == user_id:
                    if not status or conv.status == status:
                        conversations.append({
                            "id": conv.id,
                            "topic": conv.context.topic,
                            "status": conv.status.value,
                            "created_at": conv.created_at.isoformat(),
                            "updated_at": conv.updated_at.isoformat(),
                            "message_count": len(conv.messages),
                            "total_tokens": conv.total_tokens
                        })
        
        # Sort by updated_at descending
        conversations.sort(key=lambda x: x["updated_at"], reverse=True)
        return conversations[:limit]
    
    async def archive_conversation(self, conversation_id: str) -> bool:
        """Archive a conversation"""
        conversation = await self.get_conversation(conversation_id)
        if not conversation:
            return False
        
        conversation.status = ConversationStatus.ARCHIVED
        conversation.updated_at = datetime.utcnow()
        await self._save_conversation(conversation)
        return True
    
    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation"""
        # Remove from cache
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
        
        # Remove from Redis
        if self.redis_client:
            try:
                await self.redis_client.delete(f"conversation:{conversation_id}")
                return True
            except Exception as e:
                logger.error(f"Error deleting conversation from Redis: {e}")
                return False
        
        return True
    
    async def get_active_count(self) -> int:
        """Get count of active conversations"""
        if self.redis_client:
            try:
                keys = await self.redis_client.keys("conversation:*")
                active_count = 0
                for key in keys:
                    data = await self.redis_client.get(key)
                    if data:
                        conv_dict = json.loads(data)
                        if conv_dict["status"] == ConversationStatus.ACTIVE.value:
                            active_count += 1
                return active_count
            except Exception as e:
                logger.error(f"Error counting active conversations: {e}")
        
        # Fallback to in-memory count
        return len([
            conv for conv in self.conversations.values()
            if conv.status == ConversationStatus.ACTIVE
        ])
    
    async def cleanup_expired_conversations(self) -> int:
        """Clean up expired conversations"""
        cleaned_count = 0
        cutoff_date = datetime.utcnow() - self.conversation_ttl
        
        if self.redis_client:
            try:
                keys = await self.redis_client.keys("conversation:*")
                for key in keys:
                    data = await self.redis_client.get(key)
                    if data:
                        conv_dict = json.loads(data)
                        updated_at = datetime.fromisoformat(conv_dict["updated_at"])
                        if updated_at < cutoff_date:
                            await self.redis_client.delete(key)
                            cleaned_count += 1
            except Exception as e:
                logger.error(f"Error cleaning up conversations: {e}")
        
        # Clean up in-memory cache
        expired_ids = [
            conv_id for conv_id, conv in self.conversations.items()
            if conv.updated_at < cutoff_date
        ]
        for conv_id in expired_ids:
            del self.conversations[conv_id]
            cleaned_count += 1
        
        return cleaned_count
    
    async def _save_conversation(self, conversation: Conversation):
        """Save conversation to storage"""
        # Save to in-memory cache
        self.conversations[conversation.id] = conversation
        
        # Save to Redis if available
        if self.redis_client:
            try:
                conv_dict = self._conversation_to_dict(conversation)
                await self.redis_client.set(
                    f"conversation:{conversation.id}",
                    json.dumps(conv_dict, default=str),
                    ex=int(self.conversation_ttl.total_seconds())
                )
            except Exception as e:
                logger.error(f"Error saving conversation to Redis: {e}")
    
    def _conversation_to_dict(self, conversation: Conversation) -> Dict[str, Any]:
        """Convert conversation to dictionary for serialization"""
        return {
            "id": conversation.id,
            "context": asdict(conversation.context),
            "messages": [
                {
                    "id": msg.id,
                    "role": msg.role.value,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "metadata": msg.metadata,
                    "token_count": msg.token_count
                }
                for msg in conversation.messages
            ],
            "status": conversation.status.value,
            "created_at": conversation.created_at.isoformat(),
            "updated_at": conversation.updated_at.isoformat(),
            "summary": conversation.summary,
            "total_tokens": conversation.total_tokens
        }
    
    def _dict_to_conversation(self, conv_dict: Dict[str, Any]) -> Conversation:
        """Convert dictionary to conversation object"""
        context = ConversationContext(**conv_dict["context"])
        
        messages = [
            ConversationMessage(
                id=msg["id"],
                role=MessageRole(msg["role"]),
                content=msg["content"],
                timestamp=datetime.fromisoformat(msg["timestamp"]),
                metadata=msg.get("metadata", {}),
                token_count=msg.get("token_count")
            )
            for msg in conv_dict["messages"]
        ]
        
        return Conversation(
            id=conv_dict["id"],
            context=context,
            messages=messages,
            status=ConversationStatus(conv_dict["status"]),
            created_at=datetime.fromisoformat(conv_dict["created_at"]),
            updated_at=datetime.fromisoformat(conv_dict["updated_at"]),
            summary=conv_dict.get("summary"),
            total_tokens=conv_dict.get("total_tokens", 0)
        )
    
    async def _generate_conversation_summary(self, conversation: Conversation) -> str:
        """Generate a summary of the conversation for context"""
        if not conversation.messages:
            return ""
        
        # Simple summary based on recent messages
        recent_messages = conversation.messages[-5:]  # Last 5 messages
        
        topics = []
        if conversation.context.topic:
            topics.append(conversation.context.topic)
        
        # Extract key topics from messages (simple keyword extraction)
        keywords = ["price", "analysis", "risk", "portfolio", "market", "trading", "investment"]
        for msg in recent_messages:
            for keyword in keywords:
                if keyword.lower() in msg.content.lower() and keyword not in topics:
                    topics.append(keyword)
        
        summary_parts = []
        if topics:
            summary_parts.append(f"Discussion topics: {', '.join(topics[:3])}")
        
        summary_parts.append(f"Messages: {len(conversation.messages)}")
        summary_parts.append(f"Complexity: {conversation.context.complexity_level}")
        
        return "; ".join(summary_parts)

# Global conversation manager instance
conversation_manager = ConversationManager()