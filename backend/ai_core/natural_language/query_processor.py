# Query Processor
# Phase 9: AI-First Platform Implementation

import re
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
import spacy

logger = logging.getLogger(__name__)

class QueryType(Enum):
    QUESTION = "question"
    COMMAND = "command"
    STATEMENT = "statement"
    GREETING = "greeting"
    CONFIRMATION = "confirmation"

class QueryComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"

@dataclass
class ProcessedQuery:
    original_text: str
    cleaned_text: str
    tokens: List[str]
    lemmatized_tokens: List[str]
    pos_tags: List[Tuple[str, str]]
    entities: List[Dict[str, Any]]
    query_type: QueryType
    complexity: QueryComplexity
    financial_terms: List[str]
    numbers: List[Dict[str, Any]]
    dates: List[Dict[str, Any]]
    symbols: List[str]
    sentiment: float
    confidence: float
    processing_time: float

class QueryProcessor:
    """Advanced natural language query processor for financial conversations"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Load spaCy model for advanced NLP
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Using basic processing.")
            self.nlp = None
        
        # Financial terms dictionary
        self.financial_terms = self._load_financial_terms()
        
        # Stock symbol patterns
        self.symbol_pattern = re.compile(r'\b[A-Z]{1,5}\b')
        
        # Number patterns
        self.number_patterns = {
            'currency': re.compile(r'\$[\d,]+(?:\.\d{2})?'),
            'percentage': re.compile(r'\d+(?:\.\d+)?%'),
            'integer': re.compile(r'\b\d+\b'),
            'decimal': re.compile(r'\b\d+\.\d+\b')
        }
        
        # Date patterns
        self.date_patterns = {
            'relative': re.compile(r'\b(?:today|yesterday|tomorrow|last week|next week|last month|next month)\b', re.IGNORECASE),
            'specific': re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'),
            'named': re.compile(r'\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march|april|may|june|july|august|september|october|november|december)\b', re.IGNORECASE)
        }
        
        # Query type indicators
        self.query_indicators = {
            'question': ['what', 'how', 'when', 'where', 'why', 'which', 'who', 'can', 'should', 'would', 'could', 'is', 'are', 'do', 'does', 'did'],
            'command': ['buy', 'sell', 'execute', 'place', 'cancel', 'modify', 'set', 'create', 'delete', 'update', 'show', 'display'],
            'greeting': ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening'],
            'confirmation': ['yes', 'no', 'confirm', 'cancel', 'proceed', 'stop', 'continue', 'ok', 'okay']
        }
        
        logger.info("Query processor initialized")
    
    async def process(self, query: str, context: Optional[Any] = None) -> ProcessedQuery:
        """Process natural language query"""
        start_time = datetime.now()
        
        try:
            # Clean and normalize the query
            cleaned_text = self._clean_text(query)
            
            # Tokenize
            tokens = self._tokenize(cleaned_text)
            
            # Lemmatize
            lemmatized_tokens = self._lemmatize(tokens)
            
            # POS tagging
            pos_tags = self._pos_tag(tokens)
            
            # Extract entities
            entities = await self._extract_entities(cleaned_text)
            
            # Determine query type
            query_type = self._classify_query_type(cleaned_text, tokens)
            
            # Assess complexity
            complexity = self._assess_complexity(cleaned_text, tokens, entities)
            
            # Extract financial terms
            financial_terms = self._extract_financial_terms(cleaned_text)
            
            # Extract numbers
            numbers = self._extract_numbers(cleaned_text)
            
            # Extract dates
            dates = self._extract_dates(cleaned_text)
            
            # Extract stock symbols
            symbols = self._extract_symbols(cleaned_text)
            
            # Analyze sentiment
            sentiment = self._analyze_sentiment(cleaned_text)
            
            # Calculate confidence
            confidence = self._calculate_confidence(cleaned_text, tokens, entities)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ProcessedQuery(
                original_text=query,
                cleaned_text=cleaned_text,
                tokens=tokens,
                lemmatized_tokens=lemmatized_tokens,
                pos_tags=pos_tags,
                entities=entities,
                query_type=query_type,
                complexity=complexity,
                financial_terms=financial_terms,
                numbers=numbers,
                dates=dates,
                symbols=symbols,
                sentiment=sentiment,
                confidence=confidence,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Return minimal processed query on error
            return ProcessedQuery(
                original_text=query,
                cleaned_text=query.lower().strip(),
                tokens=query.split(),
                lemmatized_tokens=query.split(),
                pos_tags=[],
                entities=[],
                query_type=QueryType.STATEMENT,
                complexity=QueryComplexity.SIMPLE,
                financial_terms=[],
                numbers=[],
                dates=[],
                symbols=[],
                sentiment=0.0,
                confidence=0.5,
                processing_time=processing_time
            )
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        try:
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text.strip())
            
            # Handle contractions
            contractions = {
                "won't": "will not",
                "can't": "cannot",
                "n't": " not",
                "'re": " are",
                "'ve": " have",
                "'ll": " will",
                "'d": " would",
                "'m": " am"
            }
            
            for contraction, expansion in contractions.items():
                text = text.replace(contraction, expansion)
            
            # Normalize currency symbols
            text = re.sub(r'\$([\d,]+(?:\.\d{2})?)', r'\1 dollars', text)
            
            # Normalize percentages
            text = re.sub(r'(\d+(?:\.\d+)?)%', r'\1 percent', text)
            
            return text
            
        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            return text.lower().strip()
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        try:
            tokens = word_tokenize(text.lower())
            # Filter out punctuation and empty tokens
            tokens = [token for token in tokens if token.isalnum() or token in ['$', '%']]
            return tokens
            
        except Exception as e:
            logger.error(f"Error tokenizing text: {e}")
            return text.lower().split()
    
    def _lemmatize(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens"""
        try:
            return [self.lemmatizer.lemmatize(token) for token in tokens]
            
        except Exception as e:
            logger.error(f"Error lemmatizing tokens: {e}")
            return tokens
    
    def _pos_tag(self, tokens: List[str]) -> List[Tuple[str, str]]:
        """Part-of-speech tagging"""
        try:
            return pos_tag(tokens)
            
        except Exception as e:
            logger.error(f"Error in POS tagging: {e}")
            return [(token, 'NN') for token in tokens]
    
    async def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text"""
        entities = []
        
        try:
            if self.nlp:
                doc = self.nlp(text)
                for ent in doc.ents:
                    entities.append({
                        'text': ent.text,
                        'label': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'confidence': 0.8  # spaCy doesn't provide confidence scores
                    })
            
            # Add custom financial entity extraction
            financial_entities = self._extract_financial_entities(text)
            entities.extend(financial_entities)
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []
    
    def _extract_financial_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract financial-specific entities"""
        entities = []
        
        try:
            # Extract stock symbols
            symbols = self.symbol_pattern.findall(text)
            for symbol in symbols:
                if len(symbol) <= 5 and symbol.isupper():
                    entities.append({
                        'text': symbol,
                        'label': 'STOCK_SYMBOL',
                        'start': text.find(symbol),
                        'end': text.find(symbol) + len(symbol),
                        'confidence': 0.9
                    })
            
            # Extract currency amounts
            currency_matches = self.number_patterns['currency'].finditer(text)
            for match in currency_matches:
                entities.append({
                    'text': match.group(),
                    'label': 'CURRENCY',
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.95
                })
            
            # Extract percentages
            percentage_matches = self.number_patterns['percentage'].finditer(text)
            for match in percentage_matches:
                entities.append({
                    'text': match.group(),
                    'label': 'PERCENTAGE',
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.95
                })
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting financial entities: {e}")
            return []
    
    def _classify_query_type(self, text: str, tokens: List[str]) -> QueryType:
        """Classify the type of query"""
        try:
            text_lower = text.lower()
            
            # Check for question indicators
            if any(indicator in tokens for indicator in self.query_indicators['question']):
                return QueryType.QUESTION
            
            # Check for command indicators
            if any(indicator in tokens for indicator in self.query_indicators['command']):
                return QueryType.COMMAND
            
            # Check for greeting indicators
            if any(greeting in text_lower for greeting in self.query_indicators['greeting']):
                return QueryType.GREETING
            
            # Check for confirmation indicators
            if any(conf in tokens for conf in self.query_indicators['confirmation']):
                return QueryType.CONFIRMATION
            
            # Check for question marks
            if '?' in text:
                return QueryType.QUESTION
            
            # Check for imperative mood (commands)
            if text.endswith('!') or any(word in tokens[:2] for word in ['please', 'can', 'could']):
                return QueryType.COMMAND
            
            return QueryType.STATEMENT
            
        except Exception as e:
            logger.error(f"Error classifying query type: {e}")
            return QueryType.STATEMENT
    
    def _assess_complexity(self, text: str, tokens: List[str], entities: List[Dict[str, Any]]) -> QueryComplexity:
        """Assess query complexity"""
        try:
            complexity_score = 0
            
            # Length factor
            if len(tokens) > 20:
                complexity_score += 2
            elif len(tokens) > 10:
                complexity_score += 1
            
            # Entity factor
            if len(entities) > 5:
                complexity_score += 2
            elif len(entities) > 2:
                complexity_score += 1
            
            # Financial terms factor
            financial_term_count = sum(1 for token in tokens if token.lower() in self.financial_terms)
            if financial_term_count > 3:
                complexity_score += 2
            elif financial_term_count > 1:
                complexity_score += 1
            
            # Multiple clauses factor
            if any(word in text.lower() for word in ['and', 'or', 'but', 'however', 'also', 'additionally']):
                complexity_score += 1
            
            # Question complexity
            if any(word in tokens for word in ['why', 'how', 'explain', 'analyze', 'compare']):
                complexity_score += 1
            
            if complexity_score >= 4:
                return QueryComplexity.COMPLEX
            elif complexity_score >= 2:
                return QueryComplexity.MODERATE
            else:
                return QueryComplexity.SIMPLE
                
        except Exception as e:
            logger.error(f"Error assessing complexity: {e}")
            return QueryComplexity.SIMPLE
    
    def _extract_financial_terms(self, text: str) -> List[str]:
        """Extract financial terms from text"""
        try:
            text_lower = text.lower()
            found_terms = []
            
            for term in self.financial_terms:
                if term in text_lower:
                    found_terms.append(term)
            
            return found_terms
            
        except Exception as e:
            logger.error(f"Error extracting financial terms: {e}")
            return []
    
    def _extract_numbers(self, text: str) -> List[Dict[str, Any]]:
        """Extract numbers and their types from text"""
        numbers = []
        
        try:
            for number_type, pattern in self.number_patterns.items():
                matches = pattern.finditer(text)
                for match in matches:
                    numbers.append({
                        'text': match.group(),
                        'type': number_type,
                        'start': match.start(),
                        'end': match.end(),
                        'value': self._parse_number_value(match.group(), number_type)
                    })
            
            return numbers
            
        except Exception as e:
            logger.error(f"Error extracting numbers: {e}")
            return []
    
    def _parse_number_value(self, text: str, number_type: str) -> float:
        """Parse number value from text"""
        try:
            if number_type == 'currency':
                return float(text.replace('$', '').replace(',', ''))
            elif number_type == 'percentage':
                return float(text.replace('%', '')) / 100
            elif number_type in ['integer', 'decimal']:
                return float(text)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error parsing number value: {e}")
            return 0.0
    
    def _extract_dates(self, text: str) -> List[Dict[str, Any]]:
        """Extract dates from text"""
        dates = []
        
        try:
            for date_type, pattern in self.date_patterns.items():
                matches = pattern.finditer(text)
                for match in matches:
                    dates.append({
                        'text': match.group(),
                        'type': date_type,
                        'start': match.start(),
                        'end': match.end(),
                        'parsed_date': self._parse_date(match.group(), date_type)
                    })
            
            return dates
            
        except Exception as e:
            logger.error(f"Error extracting dates: {e}")
            return []
    
    def _parse_date(self, text: str, date_type: str) -> Optional[datetime]:
        """Parse date from text"""
        try:
            text_lower = text.lower()
            now = datetime.now()
            
            if date_type == 'relative':
                if text_lower == 'today':
                    return now.date()
                elif text_lower == 'yesterday':
                    return (now - timedelta(days=1)).date()
                elif text_lower == 'tomorrow':
                    return (now + timedelta(days=1)).date()
                elif 'last week' in text_lower:
                    return (now - timedelta(weeks=1)).date()
                elif 'next week' in text_lower:
                    return (now + timedelta(weeks=1)).date()
                elif 'last month' in text_lower:
                    return (now - timedelta(days=30)).date()
                elif 'next month' in text_lower:
                    return (now + timedelta(days=30)).date()
            
            # For specific and named dates, would need more sophisticated parsing
            return None
            
        except Exception as e:
            logger.error(f"Error parsing date: {e}")
            return None
    
    def _extract_symbols(self, text: str) -> List[str]:
        """Extract stock symbols from text"""
        try:
            symbols = self.symbol_pattern.findall(text)
            # Filter to likely stock symbols (1-5 uppercase letters)
            valid_symbols = [s for s in symbols if 1 <= len(s) <= 5 and s.isupper()]
            
            # Remove common false positives
            false_positives = {'I', 'A', 'AM', 'PM', 'US', 'UK', 'EU', 'AI', 'IT', 'OR', 'AND', 'THE'}
            valid_symbols = [s for s in valid_symbols if s not in false_positives]
            
            return list(set(valid_symbols))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error extracting symbols: {e}")
            return []
    
    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of the text"""
        try:
            # Simple sentiment analysis based on keywords
            positive_words = ['good', 'great', 'excellent', 'positive', 'bullish', 'up', 'gain', 'profit', 'buy', 'strong']
            negative_words = ['bad', 'terrible', 'negative', 'bearish', 'down', 'loss', 'sell', 'weak', 'decline', 'drop']
            
            text_lower = text.lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            total_words = len(text.split())
            if total_words == 0:
                return 0.0
            
            sentiment_score = (positive_count - negative_count) / total_words
            return max(-1.0, min(1.0, sentiment_score))  # Clamp between -1 and 1
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return 0.0
    
    def _calculate_confidence(self, text: str, tokens: List[str], entities: List[Dict[str, Any]]) -> float:
        """Calculate processing confidence"""
        try:
            confidence = 0.5  # Base confidence
            
            # Length factor
            if 5 <= len(tokens) <= 20:
                confidence += 0.2
            elif len(tokens) < 5:
                confidence -= 0.1
            
            # Entity factor
            if entities:
                confidence += min(0.3, len(entities) * 0.1)
            
            # Financial terms factor
            financial_term_count = sum(1 for token in tokens if token.lower() in self.financial_terms)
            if financial_term_count > 0:
                confidence += min(0.2, financial_term_count * 0.05)
            
            # Grammar factor (simple check)
            if text.strip().endswith(('.', '?', '!')):
                confidence += 0.1
            
            return max(0.0, min(1.0, confidence))  # Clamp between 0 and 1
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def _load_financial_terms(self) -> set:
        """Load financial terms dictionary"""
        return {
            # Basic financial terms
            'stock', 'stocks', 'share', 'shares', 'equity', 'equities',
            'bond', 'bonds', 'etf', 'etfs', 'mutual fund', 'index',
            'portfolio', 'position', 'positions', 'holding', 'holdings',
            'trade', 'trading', 'buy', 'sell', 'purchase', 'sale',
            'price', 'value', 'market', 'markets', 'exchange',
            'dividend', 'dividends', 'yield', 'return', 'returns',
            'profit', 'loss', 'gain', 'gains', 'pnl', 'p&l',
            'risk', 'volatility', 'beta', 'alpha', 'correlation',
            'volume', 'liquidity', 'spread', 'bid', 'ask',
            'bull', 'bullish', 'bear', 'bearish', 'trend', 'trending',
            'support', 'resistance', 'breakout', 'breakdown',
            'analysis', 'technical', 'fundamental', 'chart', 'indicator',
            'moving average', 'rsi', 'macd', 'bollinger', 'fibonacci',
            'earnings', 'revenue', 'eps', 'pe ratio', 'market cap',
            'sector', 'industry', 'growth', 'value', 'momentum',
            'diversification', 'allocation', 'rebalance', 'hedge',
            'option', 'options', 'call', 'put', 'strike', 'expiration',
            'futures', 'commodity', 'forex', 'currency', 'crypto',
            'ipo', 'merger', 'acquisition', 'split', 'spinoff',
            'insider', 'institutional', 'retail', 'analyst', 'rating',
            'upgrade', 'downgrade', 'target', 'recommendation',
            'news', 'announcement', 'filing', 'sec', 'regulation',
            'margin', 'leverage', 'short', 'long', 'stop loss',
            'limit order', 'market order', 'execution', 'fill'
        }

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except Exception as e:
    logger.warning(f"Could not download NLTK data: {e}")