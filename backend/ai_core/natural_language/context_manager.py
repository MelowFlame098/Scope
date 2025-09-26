# Context Manager
# Phase 9: AI-First Platform Implementation

import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
from collections import deque

logger = logging.getLogger(__name__)

class ConversationState(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    WAITING_CONFIRMATION = "waiting_confirmation"
    EXECUTING_TRADE = "executing_trade"
    ANALYZING = "analyzing"
    ERROR = "error"
    ENDED = "ended"

class UserRole(Enum):
    BASIC = "basic"
    PREMIUM = "premium"
    PROFESSIONAL = "professional"
    INSTITUTIONAL = "institutional"
    ADMIN = "admin"

class ContextScope(Enum):
    SESSION = "session"
    CONVERSATION = "conversation"
    TURN = "turn"
    GLOBAL = "global"

@dataclass
class ConversationTurn:
    turn_id: str
    timestamp: datetime
    user_input: str
    processed_query: Optional[Any] = None
    intent: Optional[str] = None
    entities: List[Dict[str, Any]] = field(default_factory=list)
    response: Optional[str] = None
    actions_taken: List[Dict[str, Any]] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class UserProfile:
    user_id: str
    role: UserRole
    preferences: Dict[str, Any] = field(default_factory=dict)
    trading_experience: str = "beginner"
    risk_tolerance: str = "moderate"
    investment_goals: List[str] = field(default_factory=list)
    communication_style: str = "formal"
    language: str = "en"
    timezone: str = "UTC"
    notification_preferences: Dict[str, bool] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)

@dataclass
class SessionContext:
    session_id: str
    user_profile: UserProfile
    start_time: datetime
    last_activity: datetime
    state: ConversationState = ConversationState.IDLE
    conversation_history: deque = field(default_factory=lambda: deque(maxlen=50))
    pending_actions: List[Dict[str, Any]] = field(default_factory=list)
    active_topics: List[str] = field(default_factory=list)
    context_variables: Dict[str, Any] = field(default_factory=dict)
    error_count: int = 0
    total_turns: int = 0
    successful_actions: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ContextMemory:
    short_term: Dict[str, Any] = field(default_factory=dict)  # Current conversation
    medium_term: Dict[str, Any] = field(default_factory=dict)  # Current session
    long_term: Dict[str, Any] = field(default_factory=dict)   # Persistent user data
    working_memory: Dict[str, Any] = field(default_factory=dict)  # Temporary processing

class ContextManager:
    """Advanced context manager for conversational AI"""
    
    def __init__(self, max_sessions: int = 1000, session_timeout: int = 3600):
        self.max_sessions = max_sessions
        self.session_timeout = session_timeout  # seconds
        
        # Active sessions
        self.sessions: Dict[str, SessionContext] = {}
        
        # Context memory storage
        self.memory_store: Dict[str, ContextMemory] = {}
        
        # Topic tracking
        self.topic_transitions: Dict[str, List[str]] = {}
        self.topic_weights: Dict[str, float] = {}
        
        # Entity resolution cache
        self.entity_cache: Dict[str, Dict[str, Any]] = {}
        
        # Conversation patterns
        self.conversation_patterns: Dict[str, List[str]] = {}
        
        # Background cleanup task
        asyncio.create_task(self._cleanup_expired_sessions())
        
        logger.info("Context manager initialized")
    
    async def create_session(self, user_id: str, user_profile: Optional[UserProfile] = None) -> str:
        """Create a new conversation session"""
        try:
            session_id = str(uuid.uuid4())
            
            # Create or use provided user profile
            if not user_profile:
                user_profile = await self._get_or_create_user_profile(user_id)
            
            # Create session context
            session = SessionContext(
                session_id=session_id,
                user_profile=user_profile,
                start_time=datetime.now(),
                last_activity=datetime.now()
            )
            
            # Initialize context memory
            self.memory_store[session_id] = ContextMemory()
            
            # Load long-term memory for user
            await self._load_user_memory(session_id, user_id)
            
            # Store session
            self.sessions[session_id] = session
            
            # Cleanup old sessions if needed
            if len(self.sessions) > self.max_sessions:
                await self._cleanup_oldest_sessions()
            
            logger.info(f"Created session {session_id} for user {user_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            raise
    
    async def update_context(self, session_id: str, turn: ConversationTurn) -> Dict[str, Any]:
        """Update conversation context with new turn"""
        try:
            session = self.sessions.get(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")
            
            # Update session activity
            session.last_activity = datetime.now()
            session.total_turns += 1
            
            # Add turn to conversation history
            session.conversation_history.append(turn)
            
            # Update context variables
            await self._update_context_variables(session, turn)
            
            # Update active topics
            await self._update_active_topics(session, turn)
            
            # Update entity cache
            await self._update_entity_cache(session_id, turn.entities)
            
            # Update memory
            await self._update_memory(session_id, turn)
            
            # Analyze conversation patterns
            await self._analyze_conversation_patterns(session, turn)
            
            # Generate context summary
            context_summary = await self._generate_context_summary(session)
            
            logger.debug(f"Updated context for session {session_id}")
            return context_summary
            
        except Exception as e:
            logger.error(f"Error updating context: {e}")
            return {}
    
    async def get_context(self, session_id: str, scope: ContextScope = ContextScope.CONVERSATION) -> Dict[str, Any]:
        """Get context information for a session"""
        try:
            session = self.sessions.get(session_id)
            if not session:
                return {}
            
            memory = self.memory_store.get(session_id, ContextMemory())
            
            context = {
                'session_id': session_id,
                'user_profile': session.user_profile,
                'state': session.state,
                'active_topics': session.active_topics,
                'pending_actions': session.pending_actions,
                'context_variables': session.context_variables,
                'last_activity': session.last_activity,
                'total_turns': session.total_turns
            }
            
            if scope == ContextScope.TURN:
                # Only current turn context
                if session.conversation_history:
                    context['current_turn'] = session.conversation_history[-1]
                    
            elif scope == ContextScope.CONVERSATION:
                # Recent conversation history
                context['conversation_history'] = list(session.conversation_history)[-10:]
                context['short_term_memory'] = memory.short_term
                
            elif scope == ContextScope.SESSION:
                # Full session context
                context['conversation_history'] = list(session.conversation_history)
                context['short_term_memory'] = memory.short_term
                context['medium_term_memory'] = memory.medium_term
                context['session_metadata'] = session.metadata
                
            elif scope == ContextScope.GLOBAL:
                # All available context
                context['conversation_history'] = list(session.conversation_history)
                context['memory'] = {
                    'short_term': memory.short_term,
                    'medium_term': memory.medium_term,
                    'long_term': memory.long_term,
                    'working': memory.working_memory
                }
                context['entity_cache'] = self.entity_cache.get(session_id, {})
                context['conversation_patterns'] = self.conversation_patterns.get(session_id, [])
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting context: {e}")
            return {}
    
    async def set_state(self, session_id: str, state: ConversationState, metadata: Optional[Dict[str, Any]] = None):
        """Set conversation state"""
        try:
            session = self.sessions.get(session_id)
            if not session:
                return
            
            old_state = session.state
            session.state = state
            session.last_activity = datetime.now()
            
            if metadata:
                session.metadata.update(metadata)
            
            # Log state transition
            logger.info(f"Session {session_id} state: {old_state.value} -> {state.value}")
            
            # Handle state-specific logic
            await self._handle_state_transition(session, old_state, state)
            
        except Exception as e:
            logger.error(f"Error setting state: {e}")
    
    async def add_pending_action(self, session_id: str, action: Dict[str, Any]):
        """Add a pending action to the session"""
        try:
            session = self.sessions.get(session_id)
            if not session:
                return
            
            action['id'] = str(uuid.uuid4())
            action['created_at'] = datetime.now().isoformat()
            action['status'] = 'pending'
            
            session.pending_actions.append(action)
            session.last_activity = datetime.now()
            
            logger.debug(f"Added pending action to session {session_id}: {action['type']}")
            
        except Exception as e:
            logger.error(f"Error adding pending action: {e}")
    
    async def complete_action(self, session_id: str, action_id: str, result: Dict[str, Any]):
        """Mark an action as completed"""
        try:
            session = self.sessions.get(session_id)
            if not session:
                return
            
            for action in session.pending_actions:
                if action.get('id') == action_id:
                    action['status'] = 'completed'
                    action['completed_at'] = datetime.now().isoformat()
                    action['result'] = result
                    session.successful_actions += 1
                    break
            
            session.last_activity = datetime.now()
            
            logger.debug(f"Completed action {action_id} in session {session_id}")
            
        except Exception as e:
            logger.error(f"Error completing action: {e}")
    
    async def resolve_entity(self, session_id: str, entity_text: str, entity_type: str) -> Optional[Dict[str, Any]]:
        """Resolve entity using context and cache"""
        try:
            # Check entity cache first
            cache_key = f"{entity_type}:{entity_text.lower()}"
            session_cache = self.entity_cache.get(session_id, {})
            
            if cache_key in session_cache:
                return session_cache[cache_key]
            
            # Get session context
            session = self.sessions.get(session_id)
            if not session:
                return None
            
            # Resolve based on context
            resolved_entity = await self._resolve_entity_with_context(session, entity_text, entity_type)
            
            # Cache the result
            if resolved_entity:
                if session_id not in self.entity_cache:
                    self.entity_cache[session_id] = {}
                self.entity_cache[session_id][cache_key] = resolved_entity
            
            return resolved_entity
            
        except Exception as e:
            logger.error(f"Error resolving entity: {e}")
            return None
    
    async def get_conversation_summary(self, session_id: str, max_turns: int = 10) -> str:
        """Generate a summary of recent conversation"""
        try:
            session = self.sessions.get(session_id)
            if not session or not session.conversation_history:
                return "No conversation history available."
            
            recent_turns = list(session.conversation_history)[-max_turns:]
            
            summary_parts = []
            for turn in recent_turns:
                if turn.user_input and turn.response:
                    summary_parts.append(f"User: {turn.user_input[:100]}...")
                    summary_parts.append(f"Assistant: {turn.response[:100]}...")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error generating conversation summary: {e}")
            return "Error generating summary."
    
    async def end_session(self, session_id: str):
        """End a conversation session"""
        try:
            session = self.sessions.get(session_id)
            if not session:
                return
            
            # Set final state
            session.state = ConversationState.ENDED
            
            # Save long-term memory
            await self._save_user_memory(session_id, session.user_profile.user_id)
            
            # Clean up
            del self.sessions[session_id]
            if session_id in self.memory_store:
                del self.memory_store[session_id]
            if session_id in self.entity_cache:
                del self.entity_cache[session_id]
            
            logger.info(f"Ended session {session_id}")
            
        except Exception as e:
            logger.error(f"Error ending session: {e}")
    
    async def _get_or_create_user_profile(self, user_id: str) -> UserProfile:
        """Get or create user profile"""
        # This would typically load from a database
        # For now, create a default profile
        return UserProfile(
            user_id=user_id,
            role=UserRole.BASIC,
            preferences={
                'response_length': 'medium',
                'technical_level': 'intermediate',
                'chart_preferences': 'candlestick',
                'notification_frequency': 'normal'
            },
            trading_experience='intermediate',
            risk_tolerance='moderate',
            investment_goals=['growth', 'income'],
            communication_style='friendly',
            language='en',
            timezone='UTC'
        )
    
    async def _load_user_memory(self, session_id: str, user_id: str):
        """Load user's long-term memory"""
        try:
            # This would typically load from a database
            # For now, initialize with empty memory
            memory = self.memory_store.get(session_id, ContextMemory())
            memory.long_term = {
                'user_preferences': {},
                'trading_history': [],
                'frequent_queries': [],
                'learned_patterns': {}
            }
            
        except Exception as e:
            logger.error(f"Error loading user memory: {e}")
    
    async def _save_user_memory(self, session_id: str, user_id: str):
        """Save user's long-term memory"""
        try:
            memory = self.memory_store.get(session_id)
            if not memory:
                return
            
            # This would typically save to a database
            logger.debug(f"Saved long-term memory for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error saving user memory: {e}")
    
    async def _update_context_variables(self, session: SessionContext, turn: ConversationTurn):
        """Update context variables based on the turn"""
        try:
            # Extract relevant information from entities
            for entity in turn.entities:
                entity_type = entity.get('label', '')
                entity_text = entity.get('text', '')
                
                if entity_type == 'STOCK_SYMBOL':
                    session.context_variables['last_symbol'] = entity_text
                elif entity_type == 'QUANTITY':
                    session.context_variables['last_quantity'] = entity.get('normalized_value', entity_text)
                elif entity_type == 'MONEY':
                    session.context_variables['last_amount'] = entity.get('normalized_value', entity_text)
                elif entity_type == 'ACTION':
                    session.context_variables['last_action'] = entity_text
            
            # Update based on intent
            if turn.intent:
                session.context_variables['last_intent'] = turn.intent
                
                # Intent-specific context updates
                if turn.intent == 'trade_execution':
                    session.context_variables['in_trading_flow'] = True
                elif turn.intent == 'portfolio_query':
                    session.context_variables['viewing_portfolio'] = True
                elif turn.intent == 'market_analysis':
                    session.context_variables['analyzing_market'] = True
            
            # Update timestamp
            session.context_variables['last_update'] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"Error updating context variables: {e}")
    
    async def _update_active_topics(self, session: SessionContext, turn: ConversationTurn):
        """Update active topics based on the turn"""
        try:
            # Determine topics from intent and entities
            topics = set()
            
            if turn.intent:
                # Map intents to topics
                intent_topic_map = {
                    'portfolio_query': 'portfolio',
                    'trade_execution': 'trading',
                    'market_analysis': 'market_analysis',
                    'risk_assessment': 'risk_management',
                    'performance_review': 'performance'
                }
                
                topic = intent_topic_map.get(turn.intent)
                if topic:
                    topics.add(topic)
            
            # Add topics from entities
            for entity in turn.entities:
                entity_type = entity.get('label', '')
                if entity_type == 'STOCK_SYMBOL':
                    topics.add('stocks')
                elif entity_type == 'CURRENCY':
                    topics.add('forex')
                elif entity_type == 'RISK_METRIC':
                    topics.add('risk_analysis')
                elif entity_type == 'TECHNICAL_INDICATOR':
                    topics.add('technical_analysis')
            
            # Update active topics (keep recent ones)
            for topic in topics:
                if topic in session.active_topics:
                    session.active_topics.remove(topic)
                session.active_topics.insert(0, topic)
            
            # Keep only recent topics (max 5)
            session.active_topics = session.active_topics[:5]
            
        except Exception as e:
            logger.error(f"Error updating active topics: {e}")
    
    async def _update_entity_cache(self, session_id: str, entities: List[Dict[str, Any]]):
        """Update entity cache with new entities"""
        try:
            if session_id not in self.entity_cache:
                self.entity_cache[session_id] = {}
            
            cache = self.entity_cache[session_id]
            
            for entity in entities:
                entity_type = entity.get('label', '')
                entity_text = entity.get('text', '')
                cache_key = f"{entity_type}:{entity_text.lower()}"
                
                # Store entity with metadata
                cache[cache_key] = {
                    'text': entity_text,
                    'type': entity_type,
                    'normalized_value': entity.get('normalized_value'),
                    'confidence': entity.get('confidence', 0.0),
                    'last_seen': datetime.now().isoformat(),
                    'frequency': cache.get(cache_key, {}).get('frequency', 0) + 1
                }
            
        except Exception as e:
            logger.error(f"Error updating entity cache: {e}")
    
    async def _update_memory(self, session_id: str, turn: ConversationTurn):
        """Update memory with information from the turn"""
        try:
            memory = self.memory_store.get(session_id)
            if not memory:
                return
            
            # Update short-term memory (current conversation)
            if turn.intent:
                memory.short_term['recent_intents'] = memory.short_term.get('recent_intents', [])
                memory.short_term['recent_intents'].append(turn.intent)
                memory.short_term['recent_intents'] = memory.short_term['recent_intents'][-5:]  # Keep last 5
            
            # Update working memory (temporary processing)
            memory.working_memory['current_turn'] = {
                'intent': turn.intent,
                'entities': turn.entities,
                'confidence': turn.confidence_scores,
                'timestamp': turn.timestamp.isoformat()
            }
            
            # Update medium-term memory (session-level)
            if 'session_stats' not in memory.medium_term:
                memory.medium_term['session_stats'] = {
                    'total_turns': 0,
                    'intent_counts': {},
                    'entity_counts': {},
                    'error_count': 0
                }
            
            stats = memory.medium_term['session_stats']
            stats['total_turns'] += 1
            
            if turn.intent:
                stats['intent_counts'][turn.intent] = stats['intent_counts'].get(turn.intent, 0) + 1
            
            for entity in turn.entities:
                entity_type = entity.get('label', '')
                stats['entity_counts'][entity_type] = stats['entity_counts'].get(entity_type, 0) + 1
            
        except Exception as e:
            logger.error(f"Error updating memory: {e}")
    
    async def _analyze_conversation_patterns(self, session: SessionContext, turn: ConversationTurn):
        """Analyze conversation patterns"""
        try:
            session_id = session.session_id
            
            if session_id not in self.conversation_patterns:
                self.conversation_patterns[session_id] = []
            
            patterns = self.conversation_patterns[session_id]
            
            # Add current turn pattern
            pattern = {
                'intent': turn.intent,
                'entity_types': [entity.get('label') for entity in turn.entities],
                'timestamp': turn.timestamp.isoformat(),
                'confidence': turn.confidence_scores.get('intent', 0.0)
            }
            
            patterns.append(pattern)
            
            # Keep only recent patterns
            self.conversation_patterns[session_id] = patterns[-20:]
            
        except Exception as e:
            logger.error(f"Error analyzing conversation patterns: {e}")
    
    async def _generate_context_summary(self, session: SessionContext) -> Dict[str, Any]:
        """Generate a summary of current context"""
        try:
            return {
                'session_id': session.session_id,
                'state': session.state.value,
                'active_topics': session.active_topics,
                'pending_actions_count': len(session.pending_actions),
                'conversation_length': len(session.conversation_history),
                'last_intent': session.context_variables.get('last_intent'),
                'user_role': session.user_profile.role.value,
                'session_duration': (datetime.now() - session.start_time).total_seconds(),
                'error_rate': session.error_count / max(session.total_turns, 1)
            }
            
        except Exception as e:
            logger.error(f"Error generating context summary: {e}")
            return {}
    
    async def _handle_state_transition(self, session: SessionContext, old_state: ConversationState, new_state: ConversationState):
        """Handle state transition logic"""
        try:
            # State-specific handling
            if new_state == ConversationState.ERROR:
                session.error_count += 1
            elif new_state == ConversationState.EXECUTING_TRADE:
                # Clear any conflicting pending actions
                session.pending_actions = [action for action in session.pending_actions if action.get('type') != 'trade']
            elif new_state == ConversationState.IDLE:
                # Clear working memory
                memory = self.memory_store.get(session.session_id)
                if memory:
                    memory.working_memory.clear()
            
        except Exception as e:
            logger.error(f"Error handling state transition: {e}")
    
    async def _resolve_entity_with_context(self, session: SessionContext, entity_text: str, entity_type: str) -> Optional[Dict[str, Any]]:
        """Resolve entity using conversation context"""
        try:
            # Basic resolution based on type
            if entity_type == 'STOCK_SYMBOL':
                # Validate and normalize stock symbol
                symbol = entity_text.upper()
                return {
                    'text': symbol,
                    'type': entity_type,
                    'normalized_value': symbol,
                    'confidence': 0.9 if len(symbol) <= 5 and symbol.isalpha() else 0.5
                }
            
            elif entity_type == 'CURRENCY':
                # Validate currency code
                currency = entity_text.upper()
                return {
                    'text': currency,
                    'type': entity_type,
                    'normalized_value': currency,
                    'confidence': 0.8 if len(currency) == 3 and currency.isalpha() else 0.3
                }
            
            # Default resolution
            return {
                'text': entity_text,
                'type': entity_type,
                'normalized_value': entity_text,
                'confidence': 0.5
            }
            
        except Exception as e:
            logger.error(f"Error resolving entity with context: {e}")
            return None
    
    async def _cleanup_expired_sessions(self):
        """Background task to cleanup expired sessions"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                current_time = datetime.now()
                expired_sessions = []
                
                for session_id, session in self.sessions.items():
                    if (current_time - session.last_activity).total_seconds() > self.session_timeout:
                        expired_sessions.append(session_id)
                
                for session_id in expired_sessions:
                    await self.end_session(session_id)
                    logger.info(f"Cleaned up expired session {session_id}")
                
            except Exception as e:
                logger.error(f"Error in session cleanup: {e}")
    
    async def _cleanup_oldest_sessions(self):
        """Cleanup oldest sessions when limit is reached"""
        try:
            # Sort sessions by last activity
            sorted_sessions = sorted(
                self.sessions.items(),
                key=lambda x: x[1].last_activity
            )
            
            # Remove oldest 10% of sessions
            sessions_to_remove = int(len(sorted_sessions) * 0.1)
            
            for i in range(sessions_to_remove):
                session_id = sorted_sessions[i][0]
                await self.end_session(session_id)
                logger.info(f"Cleaned up old session {session_id}")
                
        except Exception as e:
            logger.error(f"Error cleaning up oldest sessions: {e}")