# Response Generator
# Phase 9: AI-First Platform Implementation

import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import random
import re

logger = logging.getLogger(__name__)

class ResponseType(Enum):
    INFORMATIONAL = "informational"
    ACTIONABLE = "actionable"
    CONFIRMATIONAL = "confirmational"
    EDUCATIONAL = "educational"
    ERROR = "error"
    GREETING = "greeting"
    FAREWELL = "farewell"

class ResponseTone(Enum):
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    CASUAL = "casual"
    FORMAL = "formal"
    URGENT = "urgent"

@dataclass
class ResponseTemplate:
    template_id: str
    intent: str
    template_text: str
    placeholders: List[str]
    tone: ResponseTone
    response_type: ResponseType
    requires_data: bool = False
    fallback_text: Optional[str] = None

@dataclass
class GeneratedResponse:
    text: str
    response_type: ResponseType
    tone: ResponseTone
    confidence: float
    suggestions: List[str]
    data_requirements: List[str]
    personalization_applied: bool
    generation_time: float
    template_used: Optional[str] = None

class ResponseGenerator:
    """Intelligent response generator for financial conversations"""
    
    def __init__(self):
        self.templates = self._load_response_templates()
        self.personalization_rules = self._load_personalization_rules()
        self.tone_adjustments = self._load_tone_adjustments()
        
        # Response enhancement patterns
        self.enhancement_patterns = {
            'emoji_financial': {
                'portfolio': '📊', 'profit': '💰', 'loss': '📉', 'gain': '📈',
                'trade': '🔄', 'buy': '🟢', 'sell': '🔴', 'analysis': '🔍',
                'risk': '⚠️', 'success': '✅', 'warning': '⚠️', 'error': '❌'
            },
            'formatting': {
                'currency': lambda x: f"${x:,.2f}" if isinstance(x, (int, float)) else x,
                'percentage': lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else x,
                'large_numbers': lambda x: f"{x:,}" if isinstance(x, int) and x >= 1000 else x
            }
        }
        
        # Context-aware response modifiers
        self.context_modifiers = {
            'market_hours': {
                'open': "The market is currently open.",
                'closed': "Please note that the market is currently closed.",
                'pre_market': "We're in pre-market hours.",
                'after_hours': "We're in after-hours trading."
            },
            'user_experience': {
                'beginner': "Let me explain this in simple terms.",
                'intermediate': "Here's a detailed breakdown.",
                'expert': "Here's the technical analysis."
            }
        }
        
        logger.info("Response generator initialized")
    
    async def generate_response(self, intent: str, entities: Dict[str, Any], 
                              context: Any, data: Optional[Dict[str, Any]] = None) -> GeneratedResponse:
        """Generate intelligent response based on intent and context"""
        start_time = datetime.now()
        
        try:
            # Select appropriate template
            template = self._select_template(intent, entities, context)
            
            # Determine response tone
            tone = self._determine_tone(intent, context)
            
            # Generate base response
            base_response = await self._generate_base_response(template, entities, context, data)
            
            # Apply personalization
            personalized_response = self._apply_personalization(base_response, context)
            
            # Enhance with formatting and emojis
            enhanced_response = self._enhance_response(personalized_response, intent, entities)
            
            # Add context-aware modifiers
            final_response = self._add_context_modifiers(enhanced_response, context)
            
            # Generate suggestions
            suggestions = self._generate_suggestions(intent, entities, context)
            
            # Calculate confidence
            confidence = self._calculate_generation_confidence(template, entities, data)
            
            # Determine data requirements
            data_requirements = self._identify_data_requirements(intent, entities)
            
            generation_time = (datetime.now() - start_time).total_seconds()
            
            return GeneratedResponse(
                text=final_response,
                response_type=template.response_type if template else ResponseType.INFORMATIONAL,
                tone=tone,
                confidence=confidence,
                suggestions=suggestions,
                data_requirements=data_requirements,
                personalization_applied=True,
                generation_time=generation_time,
                template_used=template.template_id if template else None
            )
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            generation_time = (datetime.now() - start_time).total_seconds()
            
            return GeneratedResponse(
                text="I apologize, but I'm having trouble generating a response right now. Please try again.",
                response_type=ResponseType.ERROR,
                tone=ResponseTone.PROFESSIONAL,
                confidence=0.0,
                suggestions=["Try rephrasing your question", "Ask for help"],
                data_requirements=[],
                personalization_applied=False,
                generation_time=generation_time
            )
    
    def _select_template(self, intent: str, entities: Dict[str, Any], context: Any) -> Optional[ResponseTemplate]:
        """Select the most appropriate response template"""
        try:
            # Filter templates by intent
            intent_templates = [t for t in self.templates if t.intent == intent]
            
            if not intent_templates:
                # Fallback to generic templates
                intent_templates = [t for t in self.templates if t.intent == "generic"]
            
            if not intent_templates:
                return None
            
            # Score templates based on context and entities
            scored_templates = []
            for template in intent_templates:
                score = self._score_template(template, entities, context)
                scored_templates.append((template, score))
            
            # Select highest scoring template
            scored_templates.sort(key=lambda x: x[1], reverse=True)
            return scored_templates[0][0] if scored_templates else None
            
        except Exception as e:
            logger.error(f"Error selecting template: {e}")
            return None
    
    def _score_template(self, template: ResponseTemplate, entities: Dict[str, Any], context: Any) -> float:
        """Score template based on relevance to context and entities"""
        try:
            score = 0.5  # Base score
            
            # Check if template requires data and data is available
            if template.requires_data:
                # This would check if required data is available in context
                score += 0.2
            
            # Check placeholder coverage
            entity_keys = set(entities.keys())
            template_placeholders = set(template.placeholders)
            
            if template_placeholders:
                coverage = len(entity_keys.intersection(template_placeholders)) / len(template_placeholders)
                score += coverage * 0.3
            
            # Context-based scoring
            if hasattr(context, 'user_role'):
                if template.tone == ResponseTone.PROFESSIONAL and context.user_role.value in ['institutional', 'premium']:
                    score += 0.2
                elif template.tone == ResponseTone.FRIENDLY and context.user_role.value == 'basic':
                    score += 0.2
            
            return min(1.0, score)
            
        except Exception as e:
            logger.error(f"Error scoring template: {e}")
            return 0.5
    
    def _determine_tone(self, intent: str, context: Any) -> ResponseTone:
        """Determine appropriate response tone"""
        try:
            # Default tone mapping
            tone_mapping = {
                'greeting': ResponseTone.FRIENDLY,
                'goodbye': ResponseTone.FRIENDLY,
                'trade_execution': ResponseTone.PROFESSIONAL,
                'risk_assessment': ResponseTone.PROFESSIONAL,
                'portfolio_query': ResponseTone.PROFESSIONAL,
                'market_analysis': ResponseTone.PROFESSIONAL,
                'educational_query': ResponseTone.FRIENDLY,
                'help': ResponseTone.FRIENDLY,
                'error': ResponseTone.PROFESSIONAL
            }
            
            base_tone = tone_mapping.get(intent, ResponseTone.PROFESSIONAL)
            
            # Adjust based on user context
            if hasattr(context, 'user_role'):
                if context.user_role.value == 'institutional':
                    return ResponseTone.FORMAL
                elif context.user_role.value == 'basic':
                    return ResponseTone.FRIENDLY
            
            # Adjust based on urgency
            if hasattr(context, 'extracted_entities'):
                entities = context.extracted_entities or {}
                if any(word in str(entities).lower() for word in ['urgent', 'emergency', 'immediately', 'asap']):
                    return ResponseTone.URGENT
            
            return base_tone
            
        except Exception as e:
            logger.error(f"Error determining tone: {e}")
            return ResponseTone.PROFESSIONAL
    
    async def _generate_base_response(self, template: Optional[ResponseTemplate], 
                                    entities: Dict[str, Any], context: Any, 
                                    data: Optional[Dict[str, Any]]) -> str:
        """Generate base response from template"""
        try:
            if not template:
                return "I understand your request, but I need more information to provide a helpful response."
            
            response_text = template.template_text
            
            # Replace placeholders with actual values
            for placeholder in template.placeholders:
                value = self._get_placeholder_value(placeholder, entities, context, data)
                response_text = response_text.replace(f"{{{placeholder}}}", str(value))
            
            # Handle conditional sections
            response_text = self._process_conditional_sections(response_text, entities, context, data)
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error generating base response: {e}")
            return template.fallback_text if template and template.fallback_text else "I'm having trouble generating a response."
    
    def _get_placeholder_value(self, placeholder: str, entities: Dict[str, Any], 
                             context: Any, data: Optional[Dict[str, Any]]) -> str:
        """Get value for template placeholder"""
        try:
            # Check entities first
            if placeholder in entities:
                return str(entities[placeholder])
            
            # Check context
            if hasattr(context, placeholder):
                return str(getattr(context, placeholder))
            
            # Check data
            if data and placeholder in data:
                value = data[placeholder]
                # Format based on placeholder type
                if 'price' in placeholder.lower() or 'value' in placeholder.lower():
                    if isinstance(value, (int, float)):
                        return f"${value:,.2f}"
                elif 'percent' in placeholder.lower():
                    if isinstance(value, (int, float)):
                        return f"{value:.2f}%"
                return str(value)
            
            # Default values for common placeholders
            defaults = {
                'user_name': 'there',
                'symbol': 'the security',
                'quantity': 'some',
                'price': 'market price',
                'time': datetime.now().strftime('%H:%M'),
                'date': datetime.now().strftime('%Y-%m-%d')
            }
            
            return defaults.get(placeholder, f"[{placeholder}]")
            
        except Exception as e:
            logger.error(f"Error getting placeholder value for {placeholder}: {e}")
            return f"[{placeholder}]"
    
    def _process_conditional_sections(self, text: str, entities: Dict[str, Any], 
                                    context: Any, data: Optional[Dict[str, Any]]) -> str:
        """Process conditional sections in template text"""
        try:
            # Pattern for conditional sections: [if condition]content[/if]
            pattern = r'\[if\s+([^\]]+)\]([^\[]+)\[/if\]'
            
            def evaluate_condition(match):
                condition = match.group(1).strip()
                content = match.group(2)
                
                # Simple condition evaluation
                if self._evaluate_condition(condition, entities, context, data):
                    return content
                else:
                    return ""
            
            return re.sub(pattern, evaluate_condition, text)
            
        except Exception as e:
            logger.error(f"Error processing conditional sections: {e}")
            return text
    
    def _evaluate_condition(self, condition: str, entities: Dict[str, Any], 
                          context: Any, data: Optional[Dict[str, Any]]) -> bool:
        """Evaluate a simple condition"""
        try:
            # Simple condition evaluation
            if '=' in condition:
                key, value = condition.split('=', 1)
                key, value = key.strip(), value.strip().strip('"\'')
                
                # Check in entities
                if key in entities:
                    return str(entities[key]).lower() == value.lower()
                
                # Check in context
                if hasattr(context, key):
                    return str(getattr(context, key)).lower() == value.lower()
                
                # Check in data
                if data and key in data:
                    return str(data[key]).lower() == value.lower()
            
            elif condition in entities:
                return bool(entities[condition])
            
            elif hasattr(context, condition):
                return bool(getattr(context, condition))
            
            return False
            
        except Exception as e:
            logger.error(f"Error evaluating condition {condition}: {e}")
            return False
    
    def _apply_personalization(self, response: str, context: Any) -> str:
        """Apply personalization rules to response"""
        try:
            personalized_response = response
            
            # Apply user role-based personalization
            if hasattr(context, 'user_role'):
                role_rules = self.personalization_rules.get(context.user_role.value, {})
                
                for pattern, replacement in role_rules.items():
                    personalized_response = personalized_response.replace(pattern, replacement)
            
            # Apply user preference-based personalization
            if hasattr(context, 'user_preferences'):
                preferences = context.user_preferences or {}
                
                # Adjust complexity based on user experience
                experience_level = preferences.get('experience_level', 'intermediate')
                if experience_level == 'beginner':
                    personalized_response = self._simplify_language(personalized_response)
                elif experience_level == 'expert':
                    personalized_response = self._add_technical_details(personalized_response)
            
            return personalized_response
            
        except Exception as e:
            logger.error(f"Error applying personalization: {e}")
            return response
    
    def _enhance_response(self, response: str, intent: str, entities: Dict[str, Any]) -> str:
        """Enhance response with formatting and emojis"""
        try:
            enhanced_response = response
            
            # Add appropriate emojis
            emoji_map = self.enhancement_patterns['emoji_financial']
            for term, emoji in emoji_map.items():
                if term in response.lower() and emoji not in response:
                    # Add emoji at the beginning of relevant sentences
                    pattern = rf'\b{re.escape(term)}\b'
                    enhanced_response = re.sub(pattern, f"{emoji} {term}", enhanced_response, count=1, flags=re.IGNORECASE)
            
            # Format numbers
            enhanced_response = self._format_numbers_in_text(enhanced_response)
            
            # Add structure for long responses
            if len(enhanced_response) > 200:
                enhanced_response = self._add_structure(enhanced_response)
            
            return enhanced_response
            
        except Exception as e:
            logger.error(f"Error enhancing response: {e}")
            return response
    
    def _format_numbers_in_text(self, text: str) -> str:
        """Format numbers in text for better readability"""
        try:
            # Format currency
            text = re.sub(r'\$(\d+(?:\.\d{2})?)', lambda m: f"${float(m.group(1)):,.2f}", text)
            
            # Format large numbers
            text = re.sub(r'\b(\d{4,})\b', lambda m: f"{int(m.group(1)):,}", text)
            
            return text
            
        except Exception as e:
            logger.error(f"Error formatting numbers: {e}")
            return text
    
    def _add_structure(self, response: str) -> str:
        """Add structure to long responses"""
        try:
            # Split into sentences
            sentences = response.split('. ')
            
            if len(sentences) > 3:
                # Group related sentences
                structured_response = ""
                for i, sentence in enumerate(sentences):
                    if i > 0 and i % 2 == 0:
                        structured_response += "\n\n"
                    structured_response += sentence
                    if i < len(sentences) - 1:
                        structured_response += ". "
                
                return structured_response
            
            return response
            
        except Exception as e:
            logger.error(f"Error adding structure: {e}")
            return response
    
    def _add_context_modifiers(self, response: str, context: Any) -> str:
        """Add context-aware modifiers to response"""
        try:
            modified_response = response
            
            # Add market hours context
            if hasattr(context, 'market_context'):
                market_context = context.market_context or {}
                market_hours = market_context.get('market_hours', 'unknown')
                
                if market_hours in self.context_modifiers['market_hours']:
                    modifier = self.context_modifiers['market_hours'][market_hours]
                    modified_response += f"\n\n{modifier}"
            
            return modified_response
            
        except Exception as e:
            logger.error(f"Error adding context modifiers: {e}")
            return response
    
    def _generate_suggestions(self, intent: str, entities: Dict[str, Any], context: Any) -> List[str]:
        """Generate contextual suggestions"""
        try:
            suggestions = []
            
            # Intent-based suggestions
            suggestion_map = {
                'portfolio_query': [
                    "Show me my risk exposure",
                    "Analyze my portfolio performance",
                    "What's my asset allocation?"
                ],
                'market_analysis': [
                    "Compare with sector peers",
                    "Show technical indicators",
                    "What's the market sentiment?"
                ],
                'trade_execution': [
                    "Set a stop loss",
                    "Check my buying power",
                    "Review similar trades"
                ],
                'greeting': [
                    "Show my portfolio",
                    "What's the market doing today?",
                    "Help me analyze a stock"
                ]
            }
            
            base_suggestions = suggestion_map.get(intent, [])
            suggestions.extend(base_suggestions[:3])  # Limit to 3 suggestions
            
            # Entity-based suggestions
            if 'symbol' in entities:
                symbol = entities['symbol']
                suggestions.append(f"Analyze {symbol} fundamentals")
                suggestions.append(f"Show {symbol} price history")
            
            # Remove duplicates and limit
            suggestions = list(dict.fromkeys(suggestions))[:4]
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
            return ["Ask me anything about your portfolio", "Get market analysis", "Execute a trade"]
    
    def _calculate_generation_confidence(self, template: Optional[ResponseTemplate], 
                                       entities: Dict[str, Any], data: Optional[Dict[str, Any]]) -> float:
        """Calculate confidence in generated response"""
        try:
            confidence = 0.5  # Base confidence
            
            # Template quality factor
            if template:
                confidence += 0.3
                
                # Placeholder coverage
                if template.placeholders:
                    covered = sum(1 for p in template.placeholders if p in entities or (data and p in data))
                    coverage = covered / len(template.placeholders)
                    confidence += coverage * 0.2
            
            # Data availability factor
            if data:
                confidence += 0.2
            
            # Entity richness factor
            if entities:
                confidence += min(0.2, len(entities) * 0.05)
            
            return min(1.0, confidence)
            
        except Exception as e:
            logger.error(f"Error calculating generation confidence: {e}")
            return 0.5
    
    def _identify_data_requirements(self, intent: str, entities: Dict[str, Any]) -> List[str]:
        """Identify what data is needed for better responses"""
        try:
            requirements = []
            
            # Intent-based requirements
            requirement_map = {
                'portfolio_query': ['portfolio_data', 'position_data', 'performance_data'],
                'market_analysis': ['market_data', 'price_data', 'volume_data'],
                'trade_execution': ['account_data', 'buying_power', 'position_data'],
                'risk_assessment': ['portfolio_data', 'market_data', 'volatility_data']
            }
            
            base_requirements = requirement_map.get(intent, [])
            requirements.extend(base_requirements)
            
            # Entity-based requirements
            if 'symbol' in entities:
                requirements.extend(['price_data', 'company_data', 'financial_data'])
            
            if 'date' in entities or 'time_period' in entities:
                requirements.append('historical_data')
            
            return list(set(requirements))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error identifying data requirements: {e}")
            return []
    
    def _simplify_language(self, text: str) -> str:
        """Simplify language for beginner users"""
        try:
            # Replace complex terms with simpler ones
            simplifications = {
                'volatility': 'price swings',
                'liquidity': 'how easily you can buy/sell',
                'market capitalization': 'company size',
                'price-to-earnings ratio': 'P/E ratio (valuation measure)',
                'diversification': 'spreading your investments',
                'correlation': 'how things move together'
            }
            
            simplified_text = text
            for complex_term, simple_term in simplifications.items():
                simplified_text = simplified_text.replace(complex_term, simple_term)
            
            return simplified_text
            
        except Exception as e:
            logger.error(f"Error simplifying language: {e}")
            return text
    
    def _add_technical_details(self, text: str) -> str:
        """Add technical details for expert users"""
        try:
            # This would add more sophisticated technical analysis
            # For now, just return the original text
            return text
            
        except Exception as e:
            logger.error(f"Error adding technical details: {e}")
            return text
    
    def _load_response_templates(self) -> List[ResponseTemplate]:
        """Load response templates"""
        return [
            # Portfolio query templates
            ResponseTemplate(
                template_id="portfolio_overview",
                intent="portfolio_query",
                template_text="Here's your portfolio overview:\n\n💰 **Total Value**: {total_value}\n💵 **Cash Balance**: {cash_balance}\n📈 **Daily P&L**: {daily_pnl}\n📊 **Positions**: {position_count} active positions\n\nWould you like more details about any specific aspect?",
                placeholders=["total_value", "cash_balance", "daily_pnl", "position_count"],
                tone=ResponseTone.PROFESSIONAL,
                response_type=ResponseType.INFORMATIONAL,
                requires_data=True
            ),
            
            # Market analysis templates
            ResponseTemplate(
                template_id="market_analysis",
                intent="market_analysis",
                template_text="📊 **Market Analysis for {symbol}**\n\n💲 **Current Price**: {current_price}\n📈 **Change**: {price_change} ({change_percent})\n📊 **Volume**: {volume}\n\nBased on current conditions, {symbol} is showing {trend} momentum.",
                placeholders=["symbol", "current_price", "price_change", "change_percent", "volume", "trend"],
                tone=ResponseTone.PROFESSIONAL,
                response_type=ResponseType.INFORMATIONAL,
                requires_data=True
            ),
            
            # Trade execution templates
            ResponseTemplate(
                template_id="trade_confirmation",
                intent="trade_execution",
                template_text="🔄 **Trade Confirmation Required**\n\n📋 **Action**: {action}\n🏷️ **Symbol**: {symbol}\n📊 **Quantity**: {quantity} shares\n💰 **Estimated Cost**: {estimated_cost}\n\nDo you want me to execute this trade? Reply 'yes' to confirm or 'no' to cancel.",
                placeholders=["action", "symbol", "quantity", "estimated_cost"],
                tone=ResponseTone.PROFESSIONAL,
                response_type=ResponseType.CONFIRMATIONAL,
                requires_data=False
            ),
            
            # Greeting templates
            ResponseTemplate(
                template_id="greeting_general",
                intent="greeting",
                template_text="Good {time_of_day}! 👋 I'm your AI trading assistant.\n\nI can help you with:\n• 📊 Portfolio analysis and performance\n• 📈 Market insights and analysis\n• 🔄 Trade execution and management\n• 🎯 Risk assessment and strategy\n\nWhat would you like to explore today?",
                placeholders=["time_of_day"],
                tone=ResponseTone.FRIENDLY,
                response_type=ResponseType.GREETING,
                requires_data=False
            ),
            
            # Error templates
            ResponseTemplate(
                template_id="error_general",
                intent="error",
                template_text="I apologize, but I encountered an issue processing your request. Please try again or rephrase your question.",
                placeholders=[],
                tone=ResponseTone.PROFESSIONAL,
                response_type=ResponseType.ERROR,
                requires_data=False,
                fallback_text="Something went wrong. Please try again."
            ),
            
            # Generic fallback template
            ResponseTemplate(
                template_id="generic_fallback",
                intent="generic",
                template_text="I understand you're asking about {topic}. Let me help you with that. [if data_available]Here's what I found:[/if] [if no_data]I need more information to provide a complete answer.[/if]",
                placeholders=["topic"],
                tone=ResponseTone.PROFESSIONAL,
                response_type=ResponseType.INFORMATIONAL,
                requires_data=False,
                fallback_text="I'm here to help with your financial questions. Could you please provide more details?"
            )
        ]
    
    def _load_personalization_rules(self) -> Dict[str, Dict[str, str]]:
        """Load personalization rules for different user types"""
        return {
            'basic': {
                'technical analysis': 'chart analysis',
                'fundamental analysis': 'company research',
                'derivatives': 'options and futures'
            },
            'premium': {
                'execute': 'place',
                'position': 'holding'
            },
            'institutional': {
                'trade': 'transaction',
                'portfolio': 'investment portfolio',
                'analysis': 'comprehensive analysis'
            }
        }
    
    def _load_tone_adjustments(self) -> Dict[str, Dict[str, str]]:
        """Load tone adjustment patterns"""
        return {
            'formal': {
                'Hi': 'Good day',
                'Thanks': 'Thank you',
                'OK': 'Very well'
            },
            'casual': {
                'Good day': 'Hi',
                'Thank you': 'Thanks',
                'Very well': 'OK'
            }
        }