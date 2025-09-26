# Intent Classifier
# Phase 9: AI-First Platform Implementation

import logging
import asyncio
import json
import pickle
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import joblib

logger = logging.getLogger(__name__)

class IntentCategory(Enum):
    PORTFOLIO = "portfolio"
    TRADING = "trading"
    MARKET = "market"
    ANALYSIS = "analysis"
    SYSTEM = "system"
    SOCIAL = "social"
    EDUCATIONAL = "educational"
    GENERAL = "general"

@dataclass
class IntentDefinition:
    intent_name: str
    category: IntentCategory
    description: str
    keywords: List[str]
    patterns: List[str]
    confidence_threshold: float = 0.7
    requires_entities: List[str] = None

@dataclass
class ClassificationResult:
    intent: str
    confidence: float
    category: IntentCategory
    alternative_intents: List[Tuple[str, float]]
    features_used: List[str]
    classification_time: float
    model_used: str

class IntentClassifier:
    """Advanced intent classifier for financial conversations"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.vectorizer = None
        self.models = {}
        self.intent_definitions = self._load_intent_definitions()
        self.training_data = self._load_training_data()
        
        # Feature extractors
        self.keyword_weights = self._calculate_keyword_weights()
        self.pattern_matchers = self._compile_pattern_matchers()
        
        # Model ensemble weights
        self.model_weights = {
            'tfidf_nb': 0.3,
            'tfidf_lr': 0.3,
            'tfidf_rf': 0.2,
            'keyword_match': 0.1,
            'pattern_match': 0.1
        }
        
        # Initialize models
        asyncio.create_task(self._initialize_models())
        
        logger.info("Intent classifier initialized")
    
    async def classify(self, processed_query: Any, context: Optional[Any] = None) -> Dict[str, Any]:
        """Classify intent from processed query"""
        start_time = datetime.now()
        
        try:
            text = processed_query.cleaned_text
            tokens = processed_query.tokens
            entities = processed_query.entities
            
            # Extract features
            features = await self._extract_features(text, tokens, entities, context)
            
            # Get predictions from all models
            predictions = await self._get_ensemble_predictions(features, text)
            
            # Combine predictions
            final_prediction = self._combine_predictions(predictions)
            
            # Apply context-based adjustments
            adjusted_prediction = self._apply_context_adjustments(final_prediction, context)
            
            # Validate prediction
            validated_prediction = self._validate_prediction(adjusted_prediction, entities)
            
            classification_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'intent': validated_prediction['intent'],
                'confidence': validated_prediction['confidence'],
                'category': self._get_intent_category(validated_prediction['intent']),
                'alternative_intents': validated_prediction.get('alternatives', []),
                'features_used': list(features.keys()),
                'classification_time': classification_time,
                'model_used': 'ensemble'
            }
            
        except Exception as e:
            logger.error(f"Error classifying intent: {e}")
            classification_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'intent': 'unknown',
                'confidence': 0.0,
                'category': IntentCategory.GENERAL,
                'alternative_intents': [],
                'features_used': [],
                'classification_time': classification_time,
                'model_used': 'fallback'
            }
    
    async def _initialize_models(self):
        """Initialize and train classification models"""
        try:
            if self.model_path and self._load_saved_models():
                logger.info("Loaded saved models")
                return
            
            logger.info("Training new models...")
            await self._train_models()
            
            if self.model_path:
                self._save_models()
                
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            # Initialize with basic rule-based classification
            self._initialize_fallback_classifier()
    
    async def _train_models(self):
        """Train classification models"""
        try:
            # Prepare training data
            X, y = self._prepare_training_data()
            
            if len(X) == 0:
                logger.warning("No training data available, using rule-based classification")
                self._initialize_fallback_classifier()
                return
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Initialize and fit vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 3),
                stop_words='english',
                lowercase=True
            )
            
            X_train_vec = self.vectorizer.fit_transform(X_train)
            X_test_vec = self.vectorizer.transform(X_test)
            
            # Train multiple models
            models_to_train = {
                'naive_bayes': MultinomialNB(alpha=0.1),
                'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
            }
            
            for name, model in models_to_train.items():
                logger.info(f"Training {name}...")
                model.fit(X_train_vec, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test_vec)
                accuracy = accuracy_score(y_test, y_pred)
                logger.info(f"{name} accuracy: {accuracy:.3f}")
                
                self.models[name] = model
            
            logger.info("Model training completed")
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            self._initialize_fallback_classifier()
    
    def _prepare_training_data(self) -> Tuple[List[str], List[str]]:
        """Prepare training data from intent definitions and examples"""
        X, y = [], []
        
        try:
            # Use predefined training examples
            for intent_name, examples in self.training_data.items():
                for example in examples:
                    X.append(example)
                    y.append(intent_name)
            
            # Generate synthetic examples from patterns
            for intent_def in self.intent_definitions:
                for pattern in intent_def.patterns:
                    # Simple pattern to example conversion
                    example = pattern.replace('{symbol}', 'AAPL').replace('{quantity}', '100')
                    example = example.replace('{action}', 'buy').replace('{price}', '$150')
                    X.append(example)
                    y.append(intent_def.intent_name)
            
            logger.info(f"Prepared {len(X)} training examples for {len(set(y))} intents")
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return [], []
    
    async def _extract_features(self, text: str, tokens: List[str], 
                              entities: List[Dict[str, Any]], context: Optional[Any]) -> Dict[str, Any]:
        """Extract features for classification"""
        features = {}
        
        try:
            # Text-based features
            features['text_length'] = len(text)
            features['token_count'] = len(tokens)
            features['question_words'] = sum(1 for token in tokens if token.lower() in ['what', 'how', 'when', 'where', 'why', 'which', 'who'])
            features['command_words'] = sum(1 for token in tokens if token.lower() in ['buy', 'sell', 'execute', 'place', 'cancel', 'show', 'display'])
            
            # Entity-based features
            features['entity_count'] = len(entities)
            features['has_symbol'] = any(entity.get('label') == 'STOCK_SYMBOL' for entity in entities)
            features['has_currency'] = any(entity.get('label') == 'CURRENCY' for entity in entities)
            features['has_percentage'] = any(entity.get('label') == 'PERCENTAGE' for entity in entities)
            features['has_date'] = any(entity.get('label') in ['DATE', 'TIME'] for entity in entities)
            
            # Keyword-based features
            for intent_def in self.intent_definitions:
                keyword_score = sum(1 for keyword in intent_def.keywords if keyword.lower() in text.lower())
                features[f'keyword_score_{intent_def.intent_name}'] = keyword_score
            
            # Pattern-based features
            features['has_question_mark'] = '?' in text
            features['has_exclamation'] = '!' in text
            features['starts_with_question'] = any(text.lower().startswith(word) for word in ['what', 'how', 'when', 'where', 'why', 'which', 'who', 'can', 'should', 'would', 'could'])
            features['starts_with_command'] = any(text.lower().startswith(word) for word in ['buy', 'sell', 'execute', 'place', 'cancel', 'show', 'display', 'get', 'find'])
            
            # Context-based features
            if context:
                features['has_pending_actions'] = bool(getattr(context, 'pending_actions', []))
                features['conversation_length'] = len(getattr(context, 'conversation_history', []))
                features['user_role'] = getattr(context, 'user_role', 'basic').value if hasattr(getattr(context, 'user_role', None), 'value') else 'basic'
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return {}
    
    async def _get_ensemble_predictions(self, features: Dict[str, Any], text: str) -> Dict[str, Dict[str, float]]:
        """Get predictions from all models in the ensemble"""
        predictions = {}
        
        try:
            # TF-IDF based models
            if self.vectorizer and self.models:
                text_vec = self.vectorizer.transform([text])
                
                for model_name, model in self.models.items():
                    try:
                        if hasattr(model, 'predict_proba'):
                            proba = model.predict_proba(text_vec)[0]
                            classes = model.classes_
                            predictions[f'tfidf_{model_name}'] = dict(zip(classes, proba))
                        else:
                            pred = model.predict(text_vec)[0]
                            predictions[f'tfidf_{model_name}'] = {pred: 1.0}
                    except Exception as e:
                        logger.error(f"Error getting prediction from {model_name}: {e}")
            
            # Keyword-based prediction
            keyword_pred = self._get_keyword_prediction(text, features)
            predictions['keyword_match'] = keyword_pred
            
            # Pattern-based prediction
            pattern_pred = self._get_pattern_prediction(text, features)
            predictions['pattern_match'] = pattern_pred
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error getting ensemble predictions: {e}")
            return {'fallback': {'unknown': 1.0}}
    
    def _get_keyword_prediction(self, text: str, features: Dict[str, Any]) -> Dict[str, float]:
        """Get prediction based on keyword matching"""
        try:
            scores = {}
            text_lower = text.lower()
            
            for intent_def in self.intent_definitions:
                score = 0.0
                for keyword in intent_def.keywords:
                    if keyword.lower() in text_lower:
                        weight = self.keyword_weights.get(keyword, 1.0)
                        score += weight
                
                if score > 0:
                    scores[intent_def.intent_name] = score
            
            # Normalize scores
            if scores:
                total_score = sum(scores.values())
                scores = {intent: score / total_score for intent, score in scores.items()}
            else:
                scores = {'unknown': 1.0}
            
            return scores
            
        except Exception as e:
            logger.error(f"Error getting keyword prediction: {e}")
            return {'unknown': 1.0}
    
    def _get_pattern_prediction(self, text: str, features: Dict[str, Any]) -> Dict[str, float]:
        """Get prediction based on pattern matching"""
        try:
            scores = {}
            
            # Simple pattern-based rules
            text_lower = text.lower()
            
            # Portfolio queries
            if any(word in text_lower for word in ['portfolio', 'holdings', 'positions', 'balance']):
                scores['portfolio_query'] = 0.8
            
            # Trading commands
            if any(word in text_lower for word in ['buy', 'sell', 'trade', 'execute', 'order']):
                scores['trade_execution'] = 0.8
            
            # Market analysis
            if any(word in text_lower for word in ['analyze', 'analysis', 'market', 'price', 'chart']):
                scores['market_analysis'] = 0.7
            
            # Risk assessment
            if any(word in text_lower for word in ['risk', 'volatility', 'exposure', 'var']):
                scores['risk_assessment'] = 0.7
            
            # Greetings
            if any(word in text_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
                scores['greeting'] = 0.9
            
            # Help requests
            if any(word in text_lower for word in ['help', 'assist', 'support', 'how to']):
                scores['help'] = 0.8
            
            # Confirmations
            if text_lower.strip() in ['yes', 'no', 'confirm', 'cancel', 'ok', 'okay', 'proceed']:
                scores['confirmation'] = 0.9
            
            # Questions
            if features.get('has_question_mark') or features.get('starts_with_question'):
                if 'portfolio_query' not in scores and 'market_analysis' not in scores:
                    scores['educational_query'] = 0.6
            
            # Commands
            if features.get('starts_with_command'):
                if 'trade_execution' not in scores:
                    scores['system_control'] = 0.6
            
            # Normalize scores
            if scores:
                total_score = sum(scores.values())
                scores = {intent: score / total_score for intent, score in scores.items()}
            else:
                scores = {'unknown': 1.0}
            
            return scores
            
        except Exception as e:
            logger.error(f"Error getting pattern prediction: {e}")
            return {'unknown': 1.0}
    
    def _combine_predictions(self, predictions: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Combine predictions from all models using weighted ensemble"""
        try:
            combined_scores = {}
            
            # Combine weighted predictions
            for model_name, model_predictions in predictions.items():
                weight = self.model_weights.get(model_name, 0.1)
                
                for intent, score in model_predictions.items():
                    if intent not in combined_scores:
                        combined_scores[intent] = 0.0
                    combined_scores[intent] += weight * score
            
            # Sort by score
            sorted_scores = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            
            if not sorted_scores:
                return {'intent': 'unknown', 'confidence': 0.0, 'alternatives': []}
            
            # Get top prediction
            top_intent, top_score = sorted_scores[0]
            
            # Get alternatives
            alternatives = [(intent, score) for intent, score in sorted_scores[1:4]]
            
            return {
                'intent': top_intent,
                'confidence': top_score,
                'alternatives': alternatives
            }
            
        except Exception as e:
            logger.error(f"Error combining predictions: {e}")
            return {'intent': 'unknown', 'confidence': 0.0, 'alternatives': []}
    
    def _apply_context_adjustments(self, prediction: Dict[str, Any], context: Optional[Any]) -> Dict[str, Any]:
        """Apply context-based adjustments to prediction"""
        try:
            if not context:
                return prediction
            
            intent = prediction['intent']
            confidence = prediction['confidence']
            
            # Adjust based on conversation state
            if hasattr(context, 'state'):
                state = context.state
                
                # If waiting for confirmation, boost confirmation intent
                if state.value == 'waiting_confirmation' and intent in ['confirmation', 'cancellation']:
                    confidence = min(1.0, confidence + 0.2)
                
                # If in trading mode, boost trading-related intents
                elif state.value == 'executing_trade' and intent in ['trade_execution', 'confirmation']:
                    confidence = min(1.0, confidence + 0.1)
            
            # Adjust based on recent conversation history
            if hasattr(context, 'conversation_history'):
                history = context.conversation_history or []
                
                if len(history) > 0:
                    last_intent = history[-1].get('intent')
                    
                    # If last intent was trade_execution and current is confirmation
                    if last_intent == 'trade_execution' and intent == 'confirmation':
                        confidence = min(1.0, confidence + 0.3)
                    
                    # If user is asking follow-up questions
                    if last_intent in ['portfolio_query', 'market_analysis'] and intent in ['portfolio_query', 'market_analysis']:
                        confidence = min(1.0, confidence + 0.1)
            
            prediction['confidence'] = confidence
            return prediction
            
        except Exception as e:
            logger.error(f"Error applying context adjustments: {e}")
            return prediction
    
    def _validate_prediction(self, prediction: Dict[str, Any], entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate prediction against required entities"""
        try:
            intent = prediction['intent']
            confidence = prediction['confidence']
            
            # Get intent definition
            intent_def = next((idef for idef in self.intent_definitions if idef.intent_name == intent), None)
            
            if intent_def and intent_def.requires_entities:
                entity_labels = [entity.get('label', '') for entity in entities]
                
                # Check if required entities are present
                missing_entities = [req for req in intent_def.requires_entities if req not in entity_labels]
                
                if missing_entities:
                    # Reduce confidence if required entities are missing
                    confidence *= 0.7
                    
                    # If confidence drops too low, consider alternatives
                    if confidence < intent_def.confidence_threshold:
                        alternatives = prediction.get('alternatives', [])
                        if alternatives:
                            # Try the next best alternative
                            next_intent, next_confidence = alternatives[0]
                            next_def = next((idef for idef in self.intent_definitions if idef.intent_name == next_intent), None)
                            
                            if not next_def or not next_def.requires_entities or all(req in entity_labels for req in next_def.requires_entities):
                                intent = next_intent
                                confidence = next_confidence
            
            # Apply confidence threshold
            if intent_def and confidence < intent_def.confidence_threshold:
                intent = 'unknown'
                confidence = 0.0
            
            prediction['intent'] = intent
            prediction['confidence'] = confidence
            return prediction
            
        except Exception as e:
            logger.error(f"Error validating prediction: {e}")
            return prediction
    
    def _get_intent_category(self, intent: str) -> IntentCategory:
        """Get category for intent"""
        try:
            intent_def = next((idef for idef in self.intent_definitions if idef.intent_name == intent), None)
            return intent_def.category if intent_def else IntentCategory.GENERAL
            
        except Exception as e:
            logger.error(f"Error getting intent category: {e}")
            return IntentCategory.GENERAL
    
    def _calculate_keyword_weights(self) -> Dict[str, float]:
        """Calculate weights for keywords based on specificity"""
        keyword_counts = {}
        
        # Count keyword occurrences across intents
        for intent_def in self.intent_definitions:
            for keyword in intent_def.keywords:
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        # Calculate weights (inverse frequency)
        total_intents = len(self.intent_definitions)
        weights = {}
        for keyword, count in keyword_counts.items():
            weights[keyword] = 1.0 / count  # More specific keywords get higher weights
        
        return weights
    
    def _compile_pattern_matchers(self) -> Dict[str, Any]:
        """Compile pattern matchers for intents"""
        # This would compile regex patterns for more sophisticated matching
        return {}
    
    def _initialize_fallback_classifier(self):
        """Initialize basic rule-based classifier as fallback"""
        logger.info("Initializing fallback rule-based classifier")
        # The pattern-based prediction method serves as the fallback
    
    def _load_saved_models(self) -> bool:
        """Load saved models from disk"""
        try:
            if not self.model_path:
                return False
            
            # Load vectorizer
            vectorizer_path = f"{self.model_path}/vectorizer.pkl"
            self.vectorizer = joblib.load(vectorizer_path)
            
            # Load models
            model_names = ['naive_bayes', 'logistic_regression', 'random_forest']
            for name in model_names:
                model_path = f"{self.model_path}/{name}.pkl"
                self.models[name] = joblib.load(model_path)
            
            logger.info("Successfully loaded saved models")
            return True
            
        except Exception as e:
            logger.error(f"Error loading saved models: {e}")
            return False
    
    def _save_models(self):
        """Save trained models to disk"""
        try:
            if not self.model_path:
                return
            
            # Save vectorizer
            vectorizer_path = f"{self.model_path}/vectorizer.pkl"
            joblib.dump(self.vectorizer, vectorizer_path)
            
            # Save models
            for name, model in self.models.items():
                model_path = f"{self.model_path}/{name}.pkl"
                joblib.dump(model, model_path)
            
            logger.info("Successfully saved models")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def _load_intent_definitions(self) -> List[IntentDefinition]:
        """Load intent definitions"""
        return [
            IntentDefinition(
                intent_name="portfolio_query",
                category=IntentCategory.PORTFOLIO,
                description="User asking about their portfolio",
                keywords=["portfolio", "holdings", "positions", "balance", "assets", "investments"],
                patterns=["show my portfolio", "what are my holdings", "portfolio overview"],
                confidence_threshold=0.6
            ),
            IntentDefinition(
                intent_name="trade_execution",
                category=IntentCategory.TRADING,
                description="User wants to execute a trade",
                keywords=["buy", "sell", "trade", "execute", "order", "purchase"],
                patterns=["buy {quantity} shares of {symbol}", "sell {symbol}", "place order"],
                confidence_threshold=0.7,
                requires_entities=["STOCK_SYMBOL"]
            ),
            IntentDefinition(
                intent_name="market_analysis",
                category=IntentCategory.MARKET,
                description="User requesting market analysis",
                keywords=["analyze", "analysis", "market", "price", "chart", "performance"],
                patterns=["analyze {symbol}", "market analysis", "how is {symbol} doing"],
                confidence_threshold=0.6
            ),
            IntentDefinition(
                intent_name="risk_assessment",
                category=IntentCategory.ANALYSIS,
                description="User asking about risk",
                keywords=["risk", "volatility", "exposure", "var", "beta", "correlation"],
                patterns=["what's my risk", "risk assessment", "portfolio risk"],
                confidence_threshold=0.7
            ),
            IntentDefinition(
                intent_name="performance_review",
                category=IntentCategory.PORTFOLIO,
                description="User asking about performance",
                keywords=["performance", "returns", "profit", "loss", "gains", "pnl"],
                patterns=["how am I performing", "show returns", "portfolio performance"],
                confidence_threshold=0.6
            ),
            IntentDefinition(
                intent_name="greeting",
                category=IntentCategory.GENERAL,
                description="User greeting",
                keywords=["hello", "hi", "hey", "good morning", "good afternoon", "good evening"],
                patterns=["hello", "hi there", "good morning"],
                confidence_threshold=0.8
            ),
            IntentDefinition(
                intent_name="help",
                category=IntentCategory.GENERAL,
                description="User requesting help",
                keywords=["help", "assist", "support", "how to", "guide", "tutorial"],
                patterns=["help me", "how do I", "can you help"],
                confidence_threshold=0.7
            ),
            IntentDefinition(
                intent_name="confirmation",
                category=IntentCategory.GENERAL,
                description="User confirming or denying",
                keywords=["yes", "no", "confirm", "cancel", "ok", "okay", "proceed"],
                patterns=["yes", "no", "confirm", "cancel"],
                confidence_threshold=0.8
            ),
            IntentDefinition(
                intent_name="educational_query",
                category=IntentCategory.EDUCATIONAL,
                description="User asking educational questions",
                keywords=["what is", "explain", "define", "meaning", "learn", "understand"],
                patterns=["what is {term}", "explain {concept}", "how does {thing} work"],
                confidence_threshold=0.6
            ),
            IntentDefinition(
                intent_name="system_control",
                category=IntentCategory.SYSTEM,
                description="User controlling system settings",
                keywords=["settings", "configure", "setup", "preferences", "options"],
                patterns=["change settings", "configure system", "update preferences"],
                confidence_threshold=0.7
            )
        ]
    
    def _load_training_data(self) -> Dict[str, List[str]]:
        """Load training examples for each intent"""
        return {
            "portfolio_query": [
                "show me my portfolio",
                "what are my current holdings",
                "portfolio overview",
                "what stocks do I own",
                "my investment summary",
                "current positions",
                "account balance",
                "what's in my portfolio"
            ],
            "trade_execution": [
                "buy 100 shares of AAPL",
                "sell my TSLA position",
                "place a buy order for MSFT",
                "execute trade for GOOGL",
                "purchase 50 shares of AMZN",
                "sell 200 shares of NVDA",
                "buy AAPL at market price",
                "place limit order"
            ],
            "market_analysis": [
                "analyze AAPL stock",
                "how is the market doing",
                "TSLA price analysis",
                "market overview today",
                "analyze tech sector",
                "what's happening with MSFT",
                "market trends",
                "stock performance"
            ],
            "risk_assessment": [
                "what's my portfolio risk",
                "risk analysis",
                "how volatile is my portfolio",
                "portfolio risk exposure",
                "calculate VaR",
                "risk metrics",
                "portfolio beta",
                "diversification analysis"
            ],
            "performance_review": [
                "how am I performing",
                "portfolio returns",
                "show my gains and losses",
                "performance summary",
                "investment returns",
                "profit and loss",
                "portfolio performance",
                "how much have I made"
            ],
            "greeting": [
                "hello",
                "hi there",
                "good morning",
                "good afternoon",
                "hey",
                "hi",
                "hello there",
                "good evening"
            ],
            "help": [
                "help me",
                "I need assistance",
                "how do I use this",
                "can you help",
                "what can you do",
                "help",
                "support",
                "guide me"
            ],
            "confirmation": [
                "yes",
                "no",
                "confirm",
                "cancel",
                "ok",
                "okay",
                "proceed",
                "go ahead",
                "stop",
                "abort"
            ],
            "educational_query": [
                "what is a P/E ratio",
                "explain market cap",
                "what does beta mean",
                "define volatility",
                "how do options work",
                "what is diversification",
                "explain technical analysis",
                "what are dividends"
            ],
            "system_control": [
                "change my settings",
                "update preferences",
                "configure notifications",
                "system settings",
                "pause trading",
                "resume trading",
                "system status",
                "update configuration"
            ]
        }