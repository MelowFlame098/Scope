import asyncio
import spacy
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import re
import numpy as np
from dataclasses import dataclass
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor
import aioredis
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentLabel(str, Enum):
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"

class MarketSentiment(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    VOLATILE = "volatile"

@dataclass
class SentimentResult:
    score: float  # -1 to 1
    label: SentimentLabel
    confidence: float  # 0 to 1
    market_sentiment: MarketSentiment
    emotions: Dict[str, float]
    entities: List[Dict[str, Any]]
    keywords: List[str]
    relevance_score: float

@dataclass
class NewsAnalysis:
    sentiment: SentimentResult
    summary: str
    key_points: List[str]
    impact_score: float  # 0 to 1
    urgency: str  # low, medium, high
    related_symbols: List[str]
    category: str
    tags: List[str]

class AdvancedSentimentEngine:
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url
        self.redis_client = None
        self.cache_ttl = 3600  # 1 hour cache
        
        # Initialize models
        self.nlp = None
        self.financial_sentiment_model = None
        self.emotion_model = None
        self.summarization_model = None
        
        # Thread pool for CPU-intensive tasks
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Financial keywords and patterns
        self.financial_keywords = {
            'bullish': ['bull', 'bullish', 'rally', 'surge', 'soar', 'climb', 'rise', 'gain', 'up', 'positive', 'growth', 'increase'],
            'bearish': ['bear', 'bearish', 'crash', 'plunge', 'fall', 'drop', 'decline', 'down', 'negative', 'loss', 'decrease'],
            'volatile': ['volatile', 'volatility', 'swing', 'fluctuate', 'unstable', 'erratic', 'choppy'],
            'neutral': ['stable', 'steady', 'flat', 'unchanged', 'sideways', 'consolidate']
        }
        
        # Market impact indicators
        self.impact_indicators = {
            'high': ['breaking', 'urgent', 'alert', 'major', 'significant', 'massive', 'huge', 'unprecedented'],
            'medium': ['important', 'notable', 'considerable', 'substantial', 'moderate'],
            'low': ['minor', 'slight', 'small', 'limited', 'marginal']
        }
        
        # Symbol patterns
        self.symbol_patterns = [
            r'\b[A-Z]{1,5}\b',  # Stock symbols
            r'\b(?:BTC|ETH|ADA|DOT|SOL|MATIC|LINK|UNI|AAVE|COMP)\b',  # Crypto symbols
            r'\$[A-Z]{1,5}\b',  # $SYMBOL format
        ]
    
    async def initialize(self):
        """Initialize all models and connections."""
        try:
            # Initialize Redis connection
            if self.redis_url:
                self.redis_client = await aioredis.from_url(self.redis_url)
            
            # Load spaCy model
            logger.info("Loading spaCy model...")
            self.nlp = spacy.load("en_core_web_sm")
            
            # Load financial sentiment model (FinBERT)
            logger.info("Loading financial sentiment model...")
            self.financial_sentiment_model = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Load emotion analysis model
            logger.info("Loading emotion analysis model...")
            self.emotion_model = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Load summarization model
            logger.info("Loading summarization model...")
            self.summarization_model = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error initializing sentiment engine: {e}")
            raise
    
    async def analyze_news_article(self, title: str, content: str, url: str = "") -> NewsAnalysis:
        """Comprehensive analysis of a news article."""
        try:
            # Check cache first
            cache_key = f"news_analysis:{hash(title + content)}"
            if self.redis_client:
                cached_result = await self.redis_client.get(cache_key)
                if cached_result:
                    return NewsAnalysis(**json.loads(cached_result))
            
            # Combine title and content for analysis
            full_text = f"{title}. {content}"
            
            # Run analysis tasks concurrently
            sentiment_task = self.analyze_sentiment(full_text)
            summary_task = self.generate_summary(content)
            entities_task = self.extract_entities(full_text)
            
            sentiment, summary, entities = await asyncio.gather(
                sentiment_task, summary_task, entities_task
            )
            
            # Extract additional information
            key_points = await self.extract_key_points(content)
            impact_score = self.calculate_impact_score(full_text)
            urgency = self.determine_urgency(full_text)
            related_symbols = self.extract_symbols(full_text)
            category = self.categorize_content(full_text)
            tags = self.extract_tags(full_text)
            
            # Create analysis result
            analysis = NewsAnalysis(
                sentiment=sentiment,
                summary=summary,
                key_points=key_points,
                impact_score=impact_score,
                urgency=urgency,
                related_symbols=related_symbols,
                category=category,
                tags=tags
            )
            
            # Cache the result
            if self.redis_client:
                await self.redis_client.setex(
                    cache_key,
                    self.cache_ttl,
                    json.dumps(analysis.__dict__, default=str)
                )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing news article: {e}")
            # Return basic analysis on error
            return self._create_fallback_analysis(title, content)
    
    async def analyze_sentiment(self, text: str) -> SentimentResult:
        """Advanced sentiment analysis using multiple models."""
        try:
            # Run models in parallel
            financial_sentiment_task = self._run_financial_sentiment(text)
            emotion_analysis_task = self._run_emotion_analysis(text)
            spacy_analysis_task = self._run_spacy_analysis(text)
            
            financial_result, emotions, spacy_result = await asyncio.gather(
                financial_sentiment_task, emotion_analysis_task, spacy_analysis_task
            )
            
            # Combine results
            combined_score = self._combine_sentiment_scores(financial_result, spacy_result)
            sentiment_label = self._determine_sentiment_label(combined_score)
            market_sentiment = self._determine_market_sentiment(text, combined_score)
            confidence = financial_result.get('score', 0.5)
            
            # Extract entities and keywords
            entities = spacy_result['entities']
            keywords = spacy_result['keywords']
            relevance_score = self._calculate_relevance_score(text)
            
            return SentimentResult(
                score=combined_score,
                label=sentiment_label,
                confidence=confidence,
                market_sentiment=market_sentiment,
                emotions=emotions,
                entities=entities,
                keywords=keywords,
                relevance_score=relevance_score
            )
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return self._create_fallback_sentiment()
    
    async def _run_financial_sentiment(self, text: str) -> Dict[str, Any]:
        """Run financial sentiment analysis."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            lambda: self.financial_sentiment_model(text[:512])[0]
        )
    
    async def _run_emotion_analysis(self, text: str) -> Dict[str, float]:
        """Run emotion analysis."""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            lambda: self.emotion_model(text[:512])
        )
        
        # Convert to emotion scores
        emotions = {}
        for item in result:
            emotions[item['label'].lower()] = item['score']
        
        return emotions
    
    async def _run_spacy_analysis(self, text: str) -> Dict[str, Any]:
        """Run spaCy NLP analysis."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._process_with_spacy,
            text
        )
    
    def _process_with_spacy(self, text: str) -> Dict[str, Any]:
        """Process text with spaCy (CPU-bound operation)."""
        doc = self.nlp(text)
        
        # Extract entities
        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        
        # Extract keywords (nouns and adjectives)
        keywords = []
        for token in doc:
            if (token.pos_ in ['NOUN', 'ADJ'] and 
                not token.is_stop and 
                not token.is_punct and 
                len(token.text) > 2):
                keywords.append(token.lemma_.lower())
        
        # Calculate basic sentiment
        sentiment_score = 0
        for token in doc:
            if token.sentiment:
                sentiment_score += token.sentiment
        
        sentiment_score = sentiment_score / len(doc) if len(doc) > 0 else 0
        
        return {
            'entities': entities,
            'keywords': list(set(keywords))[:10],  # Top 10 unique keywords
            'sentiment_score': sentiment_score
        }
    
    async def generate_summary(self, text: str) -> str:
        """Generate article summary."""
        try:
            if len(text) < 100:
                return text
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                lambda: self.summarization_model(
                    text[:1024],
                    max_length=150,
                    min_length=50,
                    do_sample=False
                )
            )
            
            return result[0]['summary_text']
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            # Return first 200 characters as fallback
            return text[:200] + "..." if len(text) > 200 else text
    
    async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text."""
        spacy_result = await self._run_spacy_analysis(text)
        return spacy_result['entities']
    
    async def extract_key_points(self, text: str) -> List[str]:
        """Extract key points from article."""
        try:
            # Split into sentences
            sentences = re.split(r'[.!?]+', text)
            
            # Score sentences based on financial keywords
            scored_sentences = []
            for sentence in sentences:
                if len(sentence.strip()) > 20:
                    score = self._score_sentence_importance(sentence)
                    scored_sentences.append((sentence.strip(), score))
            
            # Sort by score and return top 3-5
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            return [sent[0] for sent in scored_sentences[:5]]
            
        except Exception as e:
            logger.error(f"Error extracting key points: {e}")
            return []
    
    def _score_sentence_importance(self, sentence: str) -> float:
        """Score sentence importance based on financial keywords."""
        score = 0
        sentence_lower = sentence.lower()
        
        # Check for financial keywords
        for category, keywords in self.financial_keywords.items():
            for keyword in keywords:
                if keyword in sentence_lower:
                    score += 1
        
        # Check for impact indicators
        for level, indicators in self.impact_indicators.items():
            for indicator in indicators:
                if indicator in sentence_lower:
                    if level == 'high':
                        score += 3
                    elif level == 'medium':
                        score += 2
                    else:
                        score += 1
        
        # Check for numbers (often important in financial news)
        if re.search(r'\d+', sentence):
            score += 1
        
        return score
    
    def calculate_impact_score(self, text: str) -> float:
        """Calculate potential market impact score."""
        score = 0
        text_lower = text.lower()
        
        # High impact indicators
        high_impact = ['breaking', 'urgent', 'major', 'significant', 'massive', 'unprecedented']
        for indicator in high_impact:
            if indicator in text_lower:
                score += 0.3
        
        # Medium impact indicators
        medium_impact = ['important', 'notable', 'considerable', 'substantial']
        for indicator in medium_impact:
            if indicator in text_lower:
                score += 0.2
        
        # Financial action words
        action_words = ['merger', 'acquisition', 'ipo', 'earnings', 'dividend', 'buyback']
        for word in action_words:
            if word in text_lower:
                score += 0.15
        
        return min(score, 1.0)  # Cap at 1.0
    
    def determine_urgency(self, text: str) -> str:
        """Determine news urgency level."""
        text_lower = text.lower()
        
        high_urgency = ['breaking', 'urgent', 'alert', 'immediate', 'emergency']
        medium_urgency = ['important', 'significant', 'major', 'notable']
        
        for word in high_urgency:
            if word in text_lower:
                return 'high'
        
        for word in medium_urgency:
            if word in text_lower:
                return 'medium'
        
        return 'low'
    
    def extract_symbols(self, text: str) -> List[str]:
        """Extract stock and crypto symbols from text."""
        symbols = set()
        
        for pattern in self.symbol_patterns:
            matches = re.findall(pattern, text)
            symbols.update(matches)
        
        # Filter out common false positives
        false_positives = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'HAD', 'BUT', 'HAS'}
        symbols = symbols - false_positives
        
        return list(symbols)[:10]  # Limit to 10 symbols
    
    def categorize_content(self, text: str) -> str:
        """Categorize content based on keywords."""
        text_lower = text.lower()
        
        categories = {
            'crypto': ['bitcoin', 'ethereum', 'cryptocurrency', 'blockchain', 'defi', 'nft'],
            'stocks': ['stock', 'equity', 'shares', 'earnings', 'dividend', 'nasdaq', 'nyse'],
            'forex': ['forex', 'currency', 'exchange rate', 'dollar', 'euro', 'yen'],
            'commodities': ['gold', 'silver', 'oil', 'crude', 'commodity'],
            'economics': ['gdp', 'inflation', 'interest rate', 'federal reserve', 'economy']
        }
        
        category_scores = {}
        for category, keywords in categories.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            return max(category_scores, key=category_scores.get)
        
        return 'general'
    
    def extract_tags(self, text: str) -> List[str]:
        """Extract relevant tags from text."""
        tags = set()
        text_lower = text.lower()
        
        # Financial terms
        financial_terms = [
            'earnings', 'revenue', 'profit', 'loss', 'ipo', 'merger', 'acquisition',
            'dividend', 'buyback', 'guidance', 'forecast', 'rally', 'crash',
            'volatility', 'bull', 'bear', 'correction', 'recession'
        ]
        
        for term in financial_terms:
            if term in text_lower:
                tags.add(term)
        
        # Market sentiment tags
        for sentiment_type, keywords in self.financial_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    tags.add(sentiment_type)
                    break
        
        return list(tags)[:8]  # Limit to 8 tags
    
    def _combine_sentiment_scores(self, financial_result: Dict, spacy_result: Dict) -> float:
        """Combine sentiment scores from different models."""
        financial_score = financial_result.get('score', 0.5)
        
        # Convert FinBERT labels to scores
        label = financial_result.get('label', 'neutral').lower()
        if label == 'positive':
            financial_normalized = financial_score
        elif label == 'negative':
            financial_normalized = -financial_score
        else:
            financial_normalized = 0
        
        spacy_score = spacy_result.get('sentiment_score', 0)
        
        # Weighted combination (FinBERT gets more weight for financial content)
        combined = (financial_normalized * 0.7) + (spacy_score * 0.3)
        
        return max(-1, min(1, combined))  # Clamp to [-1, 1]
    
    def _determine_sentiment_label(self, score: float) -> SentimentLabel:
        """Determine sentiment label from score."""
        if score > 0.6:
            return SentimentLabel.VERY_POSITIVE
        elif score > 0.2:
            return SentimentLabel.POSITIVE
        elif score > -0.2:
            return SentimentLabel.NEUTRAL
        elif score > -0.6:
            return SentimentLabel.NEGATIVE
        else:
            return SentimentLabel.VERY_NEGATIVE
    
    def _determine_market_sentiment(self, text: str, score: float) -> MarketSentiment:
        """Determine market sentiment from text and score."""
        text_lower = text.lower()
        
        # Check for volatility indicators
        volatility_words = ['volatile', 'volatility', 'swing', 'fluctuate', 'erratic']
        if any(word in text_lower for word in volatility_words):
            return MarketSentiment.VOLATILE
        
        # Check for directional sentiment
        if score > 0.3:
            return MarketSentiment.BULLISH
        elif score < -0.3:
            return MarketSentiment.BEARISH
        else:
            return MarketSentiment.NEUTRAL
    
    def _calculate_relevance_score(self, text: str) -> float:
        """Calculate how relevant the text is to financial markets."""
        text_lower = text.lower()
        financial_terms_count = 0
        
        all_financial_terms = []
        for keywords in self.financial_keywords.values():
            all_financial_terms.extend(keywords)
        
        for term in all_financial_terms:
            if term in text_lower:
                financial_terms_count += 1
        
        # Normalize by text length
        words = len(text.split())
        relevance = financial_terms_count / max(words / 100, 1)  # Per 100 words
        
        return min(relevance, 1.0)
    
    def _create_fallback_sentiment(self) -> SentimentResult:
        """Create fallback sentiment result."""
        return SentimentResult(
            score=0.0,
            label=SentimentLabel.NEUTRAL,
            confidence=0.5,
            market_sentiment=MarketSentiment.NEUTRAL,
            emotions={},
            entities=[],
            keywords=[],
            relevance_score=0.5
        )
    
    def _create_fallback_analysis(self, title: str, content: str) -> NewsAnalysis:
        """Create fallback analysis result."""
        return NewsAnalysis(
            sentiment=self._create_fallback_sentiment(),
            summary=content[:200] + "..." if len(content) > 200 else content,
            key_points=[],
            impact_score=0.5,
            urgency='low',
            related_symbols=[],
            category='general',
            tags=[]
        )
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.redis_client:
            await self.redis_client.close()
        
        if self.executor:
            self.executor.shutdown(wait=True)

# Global instance
sentiment_engine = None

async def get_sentiment_engine() -> AdvancedSentimentEngine:
    """Get or create sentiment engine instance."""
    global sentiment_engine
    if sentiment_engine is None:
        sentiment_engine = AdvancedSentimentEngine()
        await sentiment_engine.initialize()
    return sentiment_engine