# Conversational AI Engine
# Phase 9: AI-First Platform Implementation

import asyncio
import json
import logging
import re
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

from .query_processor import QueryProcessor
from .response_generator import ResponseGenerator
from .intent_classifier import IntentClassifier
from .entity_extractor import EntityExtractor
from .context_manager import ContextManager

logger = logging.getLogger(__name__)

class ConversationState(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    WAITING_CONFIRMATION = "waiting_confirmation"
    EXECUTING_TRADE = "executing_trade"
    ERROR = "error"

class UserRole(Enum):
    BASIC = "basic"
    PREMIUM = "premium"
    INSTITUTIONAL = "institutional"
    ADMIN = "admin"

@dataclass
class ConversationContext:
    user_id: str
    session_id: str
    user_role: UserRole
    conversation_history: List[Dict[str, Any]]
    current_intent: Optional[str] = None
    extracted_entities: Dict[str, Any] = None
    pending_actions: List[Dict[str, Any]] = None
    user_preferences: Dict[str, Any] = None
    portfolio_context: Dict[str, Any] = None
    market_context: Dict[str, Any] = None
    last_interaction: datetime = None
    state: ConversationState = ConversationState.IDLE

@dataclass
class ConversationResponse:
    text: str
    intent: str
    confidence: float
    entities: Dict[str, Any]
    actions: List[Dict[str, Any]]
    requires_confirmation: bool = False
    data: Optional[Dict[str, Any]] = None
    suggestions: List[str] = None
    timestamp: datetime = None

class ConversationalAI:
    """Advanced conversational AI for financial interactions"""
    
    def __init__(self, redis_client=None, autonomous_system=None):
        self.redis_client = redis_client
        self.autonomous_system = autonomous_system
        
        # Core NLP components
        self.query_processor = QueryProcessor()
        self.response_generator = ResponseGenerator()
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.context_manager = ContextManager(redis_client)
        
        # Active conversations
        self.active_conversations = {}
        
        # Supported intents and their handlers
        self.intent_handlers = {
            "portfolio_query": self._handle_portfolio_query,
            "market_analysis": self._handle_market_analysis,
            "trade_execution": self._handle_trade_execution,
            "risk_assessment": self._handle_risk_assessment,
            "performance_review": self._handle_performance_review,
            "strategy_discussion": self._handle_strategy_discussion,
            "news_analysis": self._handle_news_analysis,
            "educational_query": self._handle_educational_query,
            "system_control": self._handle_system_control,
            "greeting": self._handle_greeting,
            "goodbye": self._handle_goodbye,
            "help": self._handle_help,
            "confirmation": self._handle_confirmation,
            "cancellation": self._handle_cancellation
        }
        
        # Financial knowledge base
        self.knowledge_base = self._initialize_knowledge_base()
        
        logger.info("Conversational AI initialized")
    
    async def process_message(self, user_id: str, message: str, 
                            session_id: Optional[str] = None) -> ConversationResponse:
        """Process user message and generate response"""
        try:
            # Get or create conversation context
            context = await self._get_conversation_context(user_id, session_id)
            context.state = ConversationState.PROCESSING
            
            # Process the query
            processed_query = await self.query_processor.process(message, context)
            
            # Classify intent
            intent_result = await self.intent_classifier.classify(processed_query, context)
            
            # Extract entities
            entities = await self.entity_extractor.extract(processed_query, intent_result)
            
            # Update context
            context.current_intent = intent_result["intent"]
            context.extracted_entities = entities
            context.last_interaction = datetime.now()
            
            # Add to conversation history
            context.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "user_message": message,
                "intent": intent_result["intent"],
                "confidence": intent_result["confidence"],
                "entities": entities
            })
            
            # Handle the intent
            response = await self._handle_intent(
                intent_result["intent"], 
                entities, 
                context
            )
            
            # Update conversation history with response
            context.conversation_history[-1]["ai_response"] = response.text
            context.conversation_history[-1]["actions"] = response.actions
            
            # Update context state
            if response.requires_confirmation:
                context.state = ConversationState.WAITING_CONFIRMATION
                context.pending_actions = response.actions
            else:
                context.state = ConversationState.IDLE
            
            # Save context
            await self.context_manager.save_context(context)
            
            # Update active conversations
            self.active_conversations[context.session_id] = context
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return ConversationResponse(
                text="I apologize, but I encountered an error processing your request. Please try again.",
                intent="error",
                confidence=0.0,
                entities={},
                actions=[],
                timestamp=datetime.now()
            )
    
    async def _get_conversation_context(self, user_id: str, 
                                      session_id: Optional[str] = None) -> ConversationContext:
        """Get or create conversation context"""
        try:
            if not session_id:
                session_id = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Try to load existing context
            context = await self.context_manager.load_context(user_id, session_id)
            
            if not context:
                # Create new context
                context = ConversationContext(
                    user_id=user_id,
                    session_id=session_id,
                    user_role=UserRole.BASIC,  # Default role
                    conversation_history=[],
                    extracted_entities={},
                    pending_actions=[],
                    user_preferences={},
                    portfolio_context={},
                    market_context={},
                    last_interaction=datetime.now()
                )
                
                # Load user preferences and portfolio context
                await self._load_user_context(context)
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting conversation context: {e}")
            # Return minimal context on error
            return ConversationContext(
                user_id=user_id,
                session_id=session_id or f"{user_id}_emergency",
                user_role=UserRole.BASIC,
                conversation_history=[],
                last_interaction=datetime.now()
            )
    
    async def _load_user_context(self, context: ConversationContext):
        """Load user-specific context data"""
        try:
            # Load user preferences
            # This would integrate with user management system
            context.user_preferences = {
                "risk_tolerance": "moderate",
                "investment_horizon": "long_term",
                "preferred_assets": ["stocks", "etfs"],
                "notification_preferences": ["email", "push"]
            }
            
            # Load portfolio context
            if self.autonomous_system:
                portfolio_status = await self.autonomous_system.get_system_status()
                context.portfolio_context = portfolio_status.get("portfolio_summary", {})
            
            # Load market context
            context.market_context = {
                "market_hours": self._is_market_open(),
                "market_condition": "normal",  # Would be fetched from market analyzer
                "volatility_level": "moderate"
            }
            
        except Exception as e:
            logger.error(f"Error loading user context: {e}")
    
    async def _handle_intent(self, intent: str, entities: Dict[str, Any], 
                           context: ConversationContext) -> ConversationResponse:
        """Handle classified intent"""
        try:
            handler = self.intent_handlers.get(intent, self._handle_unknown_intent)
            return await handler(entities, context)
            
        except Exception as e:
            logger.error(f"Error handling intent {intent}: {e}")
            return ConversationResponse(
                text="I'm having trouble understanding your request. Could you please rephrase it?",
                intent=intent,
                confidence=0.0,
                entities=entities,
                actions=[],
                timestamp=datetime.now()
            )
    
    async def _handle_portfolio_query(self, entities: Dict[str, Any], 
                                    context: ConversationContext) -> ConversationResponse:
        """Handle portfolio-related queries"""
        try:
            query_type = entities.get("query_type", "overview")
            
            if query_type == "overview":
                portfolio_data = context.portfolio_context
                
                response_text = f"""Here's your portfolio overview:
                
💰 **Total Value**: ${portfolio_data.get('total_value', 0):,.2f}
💵 **Cash Balance**: ${portfolio_data.get('cash_balance', 0):,.2f}
📈 **Daily P&L**: ${portfolio_data.get('daily_pnl', 0):,.2f}
📊 **Positions**: {portfolio_data.get('positions', 0)} active positions
🎯 **Trades Today**: {portfolio_data.get('trades_today', 0)}
                
                Would you like me to provide more details about any specific aspect?"""
                
                return ConversationResponse(
                    text=response_text,
                    intent="portfolio_query",
                    confidence=0.95,
                    entities=entities,
                    actions=[],
                    data=portfolio_data,
                    suggestions=[
                        "Show me my top performing positions",
                        "What's my risk exposure?",
                        "Analyze my portfolio allocation"
                    ],
                    timestamp=datetime.now()
                )
            
            elif query_type == "performance":
                return await self._handle_performance_query(entities, context)
            
            elif query_type == "allocation":
                return await self._handle_allocation_query(entities, context)
            
            else:
                return ConversationResponse(
                    text="I can help you with portfolio overview, performance, or allocation. What would you like to know?",
                    intent="portfolio_query",
                    confidence=0.8,
                    entities=entities,
                    actions=[],
                    timestamp=datetime.now()
                )
                
        except Exception as e:
            logger.error(f"Error handling portfolio query: {e}")
            return ConversationResponse(
                text="I'm having trouble accessing your portfolio data. Please try again in a moment.",
                intent="portfolio_query",
                confidence=0.0,
                entities=entities,
                actions=[],
                timestamp=datetime.now()
            )
    
    async def _handle_market_analysis(self, entities: Dict[str, Any], 
                                    context: ConversationContext) -> ConversationResponse:
        """Handle market analysis requests"""
        try:
            symbol = entities.get("symbol", "SPY")
            analysis_type = entities.get("analysis_type", "overview")
            
            # This would integrate with real market data and analysis
            market_data = {
                "symbol": symbol,
                "current_price": 450.25,
                "change": 2.15,
                "change_percent": 0.48,
                "volume": 1250000,
                "market_cap": "45.2B",
                "pe_ratio": 18.5
            }
            
            response_text = f"""📊 **Market Analysis for {symbol}**
            
💲 **Current Price**: ${market_data['current_price']}
📈 **Change**: ${market_data['change']} ({market_data['change_percent']:+.2f}%)
📊 **Volume**: {market_data['volume']:,}
🏢 **Market Cap**: {market_data['market_cap']}
📉 **P/E Ratio**: {market_data['pe_ratio']}
            
Based on current market conditions, {symbol} is showing {'positive' if market_data['change'] > 0 else 'negative'} momentum.
            
Would you like me to provide technical analysis or fundamental insights?"""
            
            return ConversationResponse(
                text=response_text,
                intent="market_analysis",
                confidence=0.9,
                entities=entities,
                actions=[],
                data=market_data,
                suggestions=[
                    f"Show technical analysis for {symbol}",
                    f"Compare {symbol} with sector peers",
                    f"What's the sentiment for {symbol}?"
                ],
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error handling market analysis: {e}")
            return ConversationResponse(
                text="I'm having trouble accessing market data. Please try again.",
                intent="market_analysis",
                confidence=0.0,
                entities=entities,
                actions=[],
                timestamp=datetime.now()
            )
    
    async def _handle_trade_execution(self, entities: Dict[str, Any], 
                                    context: ConversationContext) -> ConversationResponse:
        """Handle trade execution requests"""
        try:
            action = entities.get("action", "")
            symbol = entities.get("symbol", "")
            quantity = entities.get("quantity", 0)
            order_type = entities.get("order_type", "market")
            
            if not symbol or not action:
                return ConversationResponse(
                    text="I need more information to execute a trade. Please specify the action (buy/sell) and symbol.",
                    intent="trade_execution",
                    confidence=0.5,
                    entities=entities,
                    actions=[],
                    timestamp=datetime.now()
                )
            
            # Validate trade parameters
            validation_result = await self._validate_trade_request(action, symbol, quantity, context)
            
            if not validation_result["valid"]:
                return ConversationResponse(
                    text=f"I cannot execute this trade: {validation_result['reason']}",
                    intent="trade_execution",
                    confidence=0.8,
                    entities=entities,
                    actions=[],
                    timestamp=datetime.now()
                )
            
            # Create trade confirmation
            estimated_cost = quantity * 450.25  # Simulated price
            
            response_text = f"""🔄 **Trade Confirmation Required**
            
📋 **Action**: {action.upper()}
🏷️ **Symbol**: {symbol}
📊 **Quantity**: {quantity:,} shares
💰 **Estimated Cost**: ${estimated_cost:,.2f}
📈 **Order Type**: {order_type.title()}
            
Do you want me to execute this trade? Reply 'yes' to confirm or 'no' to cancel."""
            
            trade_action = {
                "type": "trade_execution",
                "action": action,
                "symbol": symbol,
                "quantity": quantity,
                "order_type": order_type,
                "estimated_cost": estimated_cost
            }
            
            return ConversationResponse(
                text=response_text,
                intent="trade_execution",
                confidence=0.95,
                entities=entities,
                actions=[trade_action],
                requires_confirmation=True,
                data={"trade_details": trade_action},
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error handling trade execution: {e}")
            return ConversationResponse(
                text="I encountered an error processing your trade request. Please try again.",
                intent="trade_execution",
                confidence=0.0,
                entities=entities,
                actions=[],
                timestamp=datetime.now()
            )
    
    async def _handle_confirmation(self, entities: Dict[str, Any], 
                                 context: ConversationContext) -> ConversationResponse:
        """Handle user confirmations"""
        try:
            if not context.pending_actions:
                return ConversationResponse(
                    text="I don't have any pending actions to confirm. How can I help you?",
                    intent="confirmation",
                    confidence=0.8,
                    entities=entities,
                    actions=[],
                    timestamp=datetime.now()
                )
            
            confirmation = entities.get("confirmation", "yes").lower()
            
            if confirmation in ["yes", "confirm", "proceed", "execute"]:
                # Execute pending actions
                results = []
                for action in context.pending_actions:
                    if action["type"] == "trade_execution":
                        result = await self._execute_trade(action, context)
                        results.append(result)
                
                context.pending_actions = []
                context.state = ConversationState.IDLE
                
                return ConversationResponse(
                    text="✅ Trade executed successfully! You can check your portfolio for updates.",
                    intent="confirmation",
                    confidence=0.95,
                    entities=entities,
                    actions=results,
                    data={"execution_results": results},
                    timestamp=datetime.now()
                )
            
            else:
                # Cancel pending actions
                context.pending_actions = []
                context.state = ConversationState.IDLE
                
                return ConversationResponse(
                    text="❌ Trade cancelled. Is there anything else I can help you with?",
                    intent="confirmation",
                    confidence=0.95,
                    entities=entities,
                    actions=[],
                    timestamp=datetime.now()
                )
                
        except Exception as e:
            logger.error(f"Error handling confirmation: {e}")
            return ConversationResponse(
                text="I encountered an error processing your confirmation. Please try again.",
                intent="confirmation",
                confidence=0.0,
                entities=entities,
                actions=[],
                timestamp=datetime.now()
            )
    
    async def _handle_greeting(self, entities: Dict[str, Any], 
                             context: ConversationContext) -> ConversationResponse:
        """Handle greeting messages"""
        time_of_day = self._get_time_of_day()
        
        greeting_text = f"""Good {time_of_day}! 👋 I'm your AI trading assistant.
        
I can help you with:
• 📊 Portfolio analysis and performance
• 📈 Market insights and analysis  
• 🔄 Trade execution and management
• 🎯 Risk assessment and strategy
• 📚 Financial education and guidance
        
What would you like to explore today?"""
        
        return ConversationResponse(
            text=greeting_text,
            intent="greeting",
            confidence=0.95,
            entities=entities,
            actions=[],
            suggestions=[
                "Show my portfolio",
                "What's the market doing today?",
                "Help me analyze a stock"
            ],
            timestamp=datetime.now()
        )
    
    async def _handle_help(self, entities: Dict[str, Any], 
                         context: ConversationContext) -> ConversationResponse:
        """Handle help requests"""
        help_text = """🤖 **AI Trading Assistant Help**
        
**Portfolio Commands:**
• "Show my portfolio" - View portfolio overview
• "What's my performance?" - Performance analysis
• "Check my risk exposure" - Risk assessment
        
**Market Analysis:**
• "Analyze AAPL" - Stock analysis
• "What's the market doing?" - Market overview
• "Show me tech stocks" - Sector analysis
        
**Trading:**
• "Buy 100 shares of AAPL" - Execute trades
• "Sell my TSLA position" - Position management
• "Set a stop loss on MSFT" - Risk management
        
**System Control:**
• "Pause trading" - System control
• "Show system status" - System information
        
Just ask me naturally - I understand conversational language!"""
        
        return ConversationResponse(
            text=help_text,
            intent="help",
            confidence=0.95,
            entities=entities,
            actions=[],
            timestamp=datetime.now()
        )
    
    async def _handle_unknown_intent(self, entities: Dict[str, Any], 
                                   context: ConversationContext) -> ConversationResponse:
        """Handle unknown or unclear intents"""
        return ConversationResponse(
            text="I'm not sure I understand. Could you please rephrase your question? You can also say 'help' to see what I can do.",
            intent="unknown",
            confidence=0.3,
            entities=entities,
            actions=[],
            suggestions=[
                "Show my portfolio",
                "Help",
                "What can you do?"
            ],
            timestamp=datetime.now()
        )
    
    async def _validate_trade_request(self, action: str, symbol: str, quantity: int, 
                                    context: ConversationContext) -> Dict[str, Any]:
        """Validate trade request parameters"""
        try:
            # Check if market is open
            if not self._is_market_open() and context.user_role != UserRole.INSTITUTIONAL:
                return {"valid": False, "reason": "Market is currently closed"}
            
            # Check quantity limits
            if quantity <= 0:
                return {"valid": False, "reason": "Quantity must be positive"}
            
            if quantity > 10000:  # Max 10k shares per trade
                return {"valid": False, "reason": "Quantity exceeds maximum limit (10,000 shares)"}
            
            # Check symbol format
            if not re.match(r'^[A-Z]{1,5}$', symbol.upper()):
                return {"valid": False, "reason": "Invalid symbol format"}
            
            # Check user permissions
            if action.upper() == "SELL" and context.user_role == UserRole.BASIC:
                # Check if user has position to sell
                portfolio = context.portfolio_context
                # Simplified check - in practice would verify actual holdings
                
            return {"valid": True, "reason": "Trade request is valid"}
            
        except Exception as e:
            logger.error(f"Error validating trade request: {e}")
            return {"valid": False, "reason": "Validation error occurred"}
    
    async def _execute_trade(self, trade_action: Dict[str, Any], 
                           context: ConversationContext) -> Dict[str, Any]:
        """Execute a validated trade"""
        try:
            # This would integrate with the autonomous trading system
            # For now, simulate trade execution
            
            execution_result = {
                "trade_id": f"trade_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "status": "executed",
                "symbol": trade_action["symbol"],
                "action": trade_action["action"],
                "quantity": trade_action["quantity"],
                "price": 450.25,  # Simulated execution price
                "timestamp": datetime.now().isoformat(),
                "commission": 0.50
            }
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return {
                "trade_id": None,
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_time_of_day(self) -> str:
        """Get appropriate greeting based on time"""
        hour = datetime.now().hour
        if hour < 12:
            return "morning"
        elif hour < 17:
            return "afternoon"
        else:
            return "evening"
    
    def _is_market_open(self) -> bool:
        """Check if market is currently open"""
        now = datetime.now()
        # Simplified market hours check (9:30 AM - 4:00 PM EST, weekdays)
        if now.weekday() >= 5:  # Weekend
            return False
        
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    def _initialize_knowledge_base(self) -> Dict[str, Any]:
        """Initialize financial knowledge base"""
        return {
            "financial_terms": {
                "pe_ratio": "Price-to-Earnings ratio measures a company's valuation",
                "market_cap": "Total value of a company's shares",
                "volatility": "Measure of price fluctuation over time",
                "beta": "Measure of stock's correlation with market movements"
            },
            "trading_strategies": {
                "buy_and_hold": "Long-term investment strategy",
                "day_trading": "Short-term trading within a single day",
                "swing_trading": "Medium-term trading over days to weeks"
            },
            "risk_concepts": {
                "diversification": "Spreading investments across different assets",
                "stop_loss": "Order to sell when price falls to certain level",
                "position_sizing": "Determining appropriate investment amount"
            }
        }
    
    # Additional handler methods would be implemented here
    async def _handle_risk_assessment(self, entities: Dict[str, Any], 
                                    context: ConversationContext) -> ConversationResponse:
        """Handle risk assessment requests"""
        # Implementation for risk assessment
        pass
    
    async def _handle_performance_review(self, entities: Dict[str, Any], 
                                       context: ConversationContext) -> ConversationResponse:
        """Handle performance review requests"""
        # Implementation for performance review
        pass
    
    async def _handle_strategy_discussion(self, entities: Dict[str, Any], 
                                        context: ConversationContext) -> ConversationResponse:
        """Handle strategy discussion"""
        # Implementation for strategy discussion
        pass
    
    async def _handle_news_analysis(self, entities: Dict[str, Any], 
                                  context: ConversationContext) -> ConversationResponse:
        """Handle news analysis requests"""
        # Implementation for news analysis
        pass
    
    async def _handle_educational_query(self, entities: Dict[str, Any], 
                                       context: ConversationContext) -> ConversationResponse:
        """Handle educational queries"""
        # Implementation for educational content
        pass
    
    async def _handle_system_control(self, entities: Dict[str, Any], 
                                   context: ConversationContext) -> ConversationResponse:
        """Handle system control commands"""
        # Implementation for system control
        pass
    
    async def _handle_goodbye(self, entities: Dict[str, Any], 
                            context: ConversationContext) -> ConversationResponse:
        """Handle goodbye messages"""
        return ConversationResponse(
            text="Goodbye! Feel free to return anytime for trading assistance. Have a great day! 👋",
            intent="goodbye",
            confidence=0.95,
            entities=entities,
            actions=[],
            timestamp=datetime.now()
        )
    
    async def _handle_cancellation(self, entities: Dict[str, Any], 
                                 context: ConversationContext) -> ConversationResponse:
        """Handle cancellation requests"""
        context.pending_actions = []
        context.state = ConversationState.IDLE
        
        return ConversationResponse(
            text="All pending actions have been cancelled. How else can I help you?",
            intent="cancellation",
            confidence=0.95,
            entities=entities,
            actions=[],
            timestamp=datetime.now()
        )
    
    async def _handle_performance_query(self, entities: Dict[str, Any], 
                                      context: ConversationContext) -> ConversationResponse:
        """Handle performance-specific queries"""
        # Implementation for detailed performance analysis
        pass
    
    async def _handle_allocation_query(self, entities: Dict[str, Any], 
                                     context: ConversationContext) -> ConversationResponse:
        """Handle allocation-specific queries"""
        # Implementation for portfolio allocation analysis
        pass