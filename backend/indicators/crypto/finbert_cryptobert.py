"""FinBERT/CryptoBERT NLP Sentiment Analysis Models for Cryptocurrency

This module implements advanced NLP sentiment analysis models for cryptocurrency:
- FinBERT for financial sentiment analysis
- CryptoBERT for crypto-specific sentiment
- News sentiment aggregation
- Social media sentiment tracking
- Market sentiment indicators
- Fear & Greed Index calculation
- Sentiment-based trading signals
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import re
import json
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# For production use, these would be actual transformers models
# For this implementation, we'll simulate the behavior
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers library not available. Using simulated sentiment analysis.")

logger = logging.getLogger(__name__)

@dataclass
class SentimentScore:
    """Individual sentiment score result"""
    text: str
    positive: float
    negative: float
    neutral: float
    compound: float
    confidence: float
    timestamp: datetime
    source: str

@dataclass
class AggregatedSentiment:
    """Aggregated sentiment analysis result"""
    overall_sentiment: float
    positive_ratio: float
    negative_ratio: float
    neutral_ratio: float
    sentiment_trend: str
    confidence_score: float
    volume_weighted_sentiment: float
    fear_greed_index: float
    market_mood: str
    timestamps: List[datetime]
    sentiment_history: List[float]

@dataclass
class NewsAnalysisResult:
    """News sentiment analysis result"""
    headline_sentiment: float
    content_sentiment: float
    source_credibility: float
    impact_score: float
    key_topics: List[str]
    entity_sentiment: Dict[str, float]
    news_volume: int
    sentiment_distribution: Dict[str, int]

@dataclass
class SocialMediaResult:
    """Social media sentiment analysis result"""
    twitter_sentiment: float
    reddit_sentiment: float
    telegram_sentiment: float
    discord_sentiment: float
    influencer_sentiment: float
    viral_content_sentiment: float
    engagement_weighted_sentiment: float
    social_volume: int
    trending_topics: List[str]

@dataclass
class SentimentSignalsResult:
    """Sentiment-based trading signals result"""
    buy_signal_strength: float
    sell_signal_strength: float
    hold_signal_strength: float
    sentiment_momentum: float
    contrarian_signal: float
    crowd_psychology_phase: str
    market_sentiment_cycle: str
    signal_confidence: float

@dataclass
class FinBERTCryptoBERTResult:
    """Combined FinBERT/CryptoBERT analysis result"""
    aggregated_sentiment: AggregatedSentiment
    news_analysis: NewsAnalysisResult
    social_media_analysis: SocialMediaResult
    sentiment_signals: SentimentSignalsResult
    overall_market_sentiment: float
    sentiment_based_price_prediction: float
    risk_sentiment_assessment: str
    confidence_score: float

class FinBERTModel:
    """FinBERT Model for Financial Sentiment Analysis"""
    
    def __init__(self):
        self.model_name = "ProsusAI/finbert"
        self.tokenizer = None
        self.model = None
        self.sentiment_pipeline = None
        self._initialize_model()
        
        # Fallback sentiment lexicon for when transformers is not available
        self.positive_words = {
            'bullish', 'moon', 'pump', 'rally', 'surge', 'breakout', 'gains', 'profit',
            'buy', 'accumulate', 'hodl', 'diamond', 'hands', 'rocket', 'lambo',
            'adoption', 'institutional', 'mainstream', 'breakthrough', 'innovation'
        }
        
        self.negative_words = {
            'bearish', 'dump', 'crash', 'correction', 'sell', 'panic', 'fear',
            'bubble', 'scam', 'rug', 'pull', 'liquidation', 'margin', 'call',
            'regulation', 'ban', 'hack', 'exploit', 'vulnerability', 'decline'
        }
    
    def _initialize_model(self):
        """Initialize FinBERT model"""
        if TRANSFORMERS_AVAILABLE:
            try:
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model=self.model_name,
                    tokenizer=self.model_name
                )
                logger.info("FinBERT model initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize FinBERT model: {e}. Using fallback.")
                self.sentiment_pipeline = None
        else:
            logger.info("Using fallback sentiment analysis")
    
    def _fallback_sentiment_analysis(self, text: str) -> Dict[str, float]:
        """Fallback sentiment analysis using lexicon-based approach"""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            return {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
        
        positive_ratio = positive_count / total_sentiment_words
        negative_ratio = negative_count / total_sentiment_words
        
        # Adjust for text length and sentiment word density
        sentiment_density = total_sentiment_words / len(words) if words else 0
        confidence_multiplier = min(sentiment_density * 2, 1.0)
        
        if positive_ratio > negative_ratio:
            positive_score = 0.5 + (positive_ratio * 0.5 * confidence_multiplier)
            negative_score = 0.25 - (positive_ratio * 0.25 * confidence_multiplier)
        else:
            negative_score = 0.5 + (negative_ratio * 0.5 * confidence_multiplier)
            positive_score = 0.25 - (negative_ratio * 0.25 * confidence_multiplier)
        
        neutral_score = 1.0 - positive_score - negative_score
        
        return {
            'positive': max(0, positive_score),
            'negative': max(0, negative_score),
            'neutral': max(0, neutral_score)
        }
    
    def analyze_sentiment(self, text: str, source: str = "unknown") -> SentimentScore:
        """Analyze sentiment of given text"""
        try:
            if self.sentiment_pipeline:
                # Use actual FinBERT model
                result = self.sentiment_pipeline(text[:512])  # Truncate to model limit
                
                # Convert to our format
                if isinstance(result, list):
                    result = result[0]
                
                label = result['label'].lower()
                score = result['score']
                
                if 'positive' in label:
                    positive, negative, neutral = score, (1-score)/2, (1-score)/2
                elif 'negative' in label:
                    positive, negative, neutral = (1-score)/2, score, (1-score)/2
                else:
                    positive, negative, neutral = (1-score)/2, (1-score)/2, score
                
                confidence = score
            else:
                # Use fallback method
                sentiment_scores = self._fallback_sentiment_analysis(text)
                positive = sentiment_scores['positive']
                negative = sentiment_scores['negative']
                neutral = sentiment_scores['neutral']
                confidence = max(positive, negative, neutral)
            
            # Calculate compound score
            compound = positive - negative
            
            return SentimentScore(
                text=text[:100] + "..." if len(text) > 100 else text,
                positive=positive,
                negative=negative,
                neutral=neutral,
                compound=compound,
                confidence=confidence,
                timestamp=datetime.now(),
                source=source
            )
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            # Return neutral sentiment on error
            return SentimentScore(
                text=text[:100] + "..." if len(text) > 100 else text,
                positive=0.33,
                negative=0.33,
                neutral=0.34,
                compound=0.0,
                confidence=0.1,
                timestamp=datetime.now(),
                source=source
            )

class CryptoBERTModel:
    """CryptoBERT Model for Crypto-specific Sentiment Analysis"""
    
    def __init__(self):
        self.model_name = "ElKulako/cryptobert"  # Hypothetical crypto-specific model
        
        # Crypto-specific sentiment lexicon
        self.crypto_positive = {
            'moon', 'lambo', 'diamond', 'hands', 'hodl', 'btfd', 'ath', 'bullrun',
            'adoption', 'institutional', 'whale', 'accumulation', 'breakout', 'pump',
            'defi', 'nft', 'web3', 'metaverse', 'blockchain', 'decentralized',
            'staking', 'yield', 'farming', 'liquidity', 'mining', 'halving'
        }
        
        self.crypto_negative = {
            'dump', 'rug', 'pull', 'scam', 'ponzi', 'shitcoin', 'rekt', 'liquidated',
            'bear', 'market', 'crash', 'correction', 'fud', 'panic', 'selling',
            'regulation', 'ban', 'hack', 'exploit', 'vulnerability', 'centralized',
            'inflation', 'dilution', 'unlock', 'vesting', 'dump', 'exit'
        }
        
        self.crypto_multipliers = {
            'moon': 2.0, 'lambo': 1.8, 'diamond': 1.5, 'hodl': 1.3,
            'rug': -2.0, 'scam': -2.5, 'rekt': -1.8, 'dump': -1.5
        }
    
    def analyze_crypto_sentiment(self, text: str, source: str = "crypto") -> SentimentScore:
        """Analyze crypto-specific sentiment"""
        try:
            text_lower = text.lower()
            words = re.findall(r'\b\w+\b', text_lower)
            
            positive_score = 0
            negative_score = 0
            
            for word in words:
                if word in self.crypto_positive:
                    multiplier = self.crypto_multipliers.get(word, 1.0)
                    positive_score += multiplier
                elif word in self.crypto_negative:
                    multiplier = abs(self.crypto_multipliers.get(word, -1.0))
                    negative_score += multiplier
            
            # Normalize scores
            total_score = positive_score + negative_score
            if total_score > 0:
                positive_norm = positive_score / total_score
                negative_norm = negative_score / total_score
            else:
                positive_norm = 0.33
                negative_norm = 0.33
            
            neutral_norm = 1.0 - positive_norm - negative_norm
            compound = positive_norm - negative_norm
            
            # Calculate confidence based on crypto-specific word density
            crypto_words = sum(1 for word in words if word in self.crypto_positive or word in self.crypto_negative)
            confidence = min(crypto_words / len(words) if words else 0, 1.0)
            
            return SentimentScore(
                text=text[:100] + "..." if len(text) > 100 else text,
                positive=positive_norm,
                negative=negative_norm,
                neutral=max(0, neutral_norm),
                compound=compound,
                confidence=max(confidence, 0.1),
                timestamp=datetime.now(),
                source=source
            )
            
        except Exception as e:
            logger.error(f"Error in crypto sentiment analysis: {str(e)}")
            return SentimentScore(
                text=text[:100] + "..." if len(text) > 100 else text,
                positive=0.33,
                negative=0.33,
                neutral=0.34,
                compound=0.0,
                confidence=0.1,
                timestamp=datetime.now(),
                source=source
            )

class SentimentAggregator:
    """Aggregates sentiment from multiple sources"""
    
    def __init__(self):
        self.finbert = FinBERTModel()
        self.cryptobert = CryptoBERTModel()
    
    def aggregate_sentiment_scores(self, sentiment_scores: List[SentimentScore]) -> AggregatedSentiment:
        """Aggregate multiple sentiment scores"""
        if not sentiment_scores:
            return self._create_neutral_sentiment()
        
        # Weight scores by confidence and recency
        weighted_positive = 0
        weighted_negative = 0
        weighted_neutral = 0
        total_weight = 0
        
        current_time = datetime.now()
        sentiment_history = []
        timestamps = []
        
        for score in sentiment_scores:
            # Time decay factor (more recent = higher weight)
            time_diff = (current_time - score.timestamp).total_seconds() / 3600  # hours
            time_weight = np.exp(-time_diff / 24)  # Decay over 24 hours
            
            # Combined weight
            weight = score.confidence * time_weight
            
            weighted_positive += score.positive * weight
            weighted_negative += score.negative * weight
            weighted_neutral += score.neutral * weight
            total_weight += weight
            
            sentiment_history.append(score.compound)
            timestamps.append(score.timestamp)
        
        if total_weight == 0:
            return self._create_neutral_sentiment()
        
        # Normalize
        avg_positive = weighted_positive / total_weight
        avg_negative = weighted_negative / total_weight
        avg_neutral = weighted_neutral / total_weight
        
        overall_sentiment = avg_positive - avg_negative
        
        # Calculate ratios
        total_sentiment = avg_positive + avg_negative + avg_neutral
        positive_ratio = avg_positive / total_sentiment if total_sentiment > 0 else 0.33
        negative_ratio = avg_negative / total_sentiment if total_sentiment > 0 else 0.33
        neutral_ratio = avg_neutral / total_sentiment if total_sentiment > 0 else 0.34
        
        # Determine trend
        if len(sentiment_history) >= 2:
            recent_avg = np.mean(sentiment_history[-5:]) if len(sentiment_history) >= 5 else sentiment_history[-1]
            earlier_avg = np.mean(sentiment_history[:-5]) if len(sentiment_history) >= 10 else sentiment_history[0]
            
            if recent_avg > earlier_avg + 0.1:
                sentiment_trend = "Improving"
            elif recent_avg < earlier_avg - 0.1:
                sentiment_trend = "Deteriorating"
            else:
                sentiment_trend = "Stable"
        else:
            sentiment_trend = "Insufficient Data"
        
        # Calculate confidence
        confidence_score = np.mean([score.confidence for score in sentiment_scores])
        
        # Volume weighted sentiment (assuming equal volume for now)
        volume_weighted_sentiment = overall_sentiment
        
        # Fear & Greed Index (0-100)
        fear_greed_index = (overall_sentiment + 1) * 50  # Convert from [-1,1] to [0,100]
        
        # Market mood
        if fear_greed_index >= 75:
            market_mood = "Extreme Greed"
        elif fear_greed_index >= 55:
            market_mood = "Greed"
        elif fear_greed_index >= 45:
            market_mood = "Neutral"
        elif fear_greed_index >= 25:
            market_mood = "Fear"
        else:
            market_mood = "Extreme Fear"
        
        return AggregatedSentiment(
            overall_sentiment=overall_sentiment,
            positive_ratio=positive_ratio,
            negative_ratio=negative_ratio,
            neutral_ratio=neutral_ratio,
            sentiment_trend=sentiment_trend,
            confidence_score=confidence_score,
            volume_weighted_sentiment=volume_weighted_sentiment,
            fear_greed_index=fear_greed_index,
            market_mood=market_mood,
            timestamps=timestamps,
            sentiment_history=sentiment_history
        )
    
    def _create_neutral_sentiment(self) -> AggregatedSentiment:
        """Create neutral sentiment for edge cases"""
        return AggregatedSentiment(
            overall_sentiment=0.0,
            positive_ratio=0.33,
            negative_ratio=0.33,
            neutral_ratio=0.34,
            sentiment_trend="Neutral",
            confidence_score=0.1,
            volume_weighted_sentiment=0.0,
            fear_greed_index=50.0,
            market_mood="Neutral",
            timestamps=[datetime.now()],
            sentiment_history=[0.0]
        )

class NewsAnalyzer:
    """Analyzes news sentiment and impact"""
    
    def __init__(self):
        self.finbert = FinBERTModel()
        self.source_credibility = {
            'reuters': 0.95, 'bloomberg': 0.95, 'wsj': 0.90, 'ft': 0.90,
            'coindesk': 0.85, 'cointelegraph': 0.80, 'decrypt': 0.75,
            'twitter': 0.40, 'reddit': 0.35, 'telegram': 0.30
        }
    
    def analyze_news(self, news_data: List[Dict]) -> NewsAnalysisResult:
        """Analyze news sentiment and impact
        
        Args:
            news_data: List of news items with 'headline', 'content', 'source', 'timestamp'
        """
        if not news_data:
            return self._create_neutral_news_analysis()
        
        headline_sentiments = []
        content_sentiments = []
        impact_scores = []
        all_topics = []
        entity_sentiments = defaultdict(list)
        source_credibilities = []
        
        for news_item in news_data:
            headline = news_item.get('headline', '')
            content = news_item.get('content', '')
            source = news_item.get('source', 'unknown').lower()
            
            # Analyze headline sentiment
            if headline:
                headline_sentiment = self.finbert.analyze_sentiment(headline, 'news_headline')
                headline_sentiments.append(headline_sentiment.compound)
            
            # Analyze content sentiment
            if content:
                content_sentiment = self.finbert.analyze_sentiment(content[:1000], 'news_content')
                content_sentiments.append(content_sentiment.compound)
            
            # Calculate impact score
            credibility = self.source_credibility.get(source, 0.5)
            source_credibilities.append(credibility)
            
            # Simple impact calculation based on sentiment strength and credibility
            sentiment_strength = abs(headline_sentiment.compound if headline else 0)
            impact_score = sentiment_strength * credibility
            impact_scores.append(impact_score)
            
            # Extract topics (simplified)
            topics = self._extract_topics(headline + ' ' + content)
            all_topics.extend(topics)
            
            # Entity sentiment (simplified)
            entities = self._extract_entities(headline + ' ' + content)
            for entity in entities:
                entity_sentiments[entity].append(headline_sentiment.compound if headline else 0)
        
        # Aggregate results
        avg_headline_sentiment = np.mean(headline_sentiments) if headline_sentiments else 0
        avg_content_sentiment = np.mean(content_sentiments) if content_sentiments else 0
        avg_source_credibility = np.mean(source_credibilities) if source_credibilities else 0.5
        avg_impact_score = np.mean(impact_scores) if impact_scores else 0
        
        # Key topics
        topic_counts = Counter(all_topics)
        key_topics = [topic for topic, count in topic_counts.most_common(10)]
        
        # Entity sentiment aggregation
        entity_sentiment_avg = {}
        for entity, sentiments in entity_sentiments.items():
            entity_sentiment_avg[entity] = np.mean(sentiments)
        
        # Sentiment distribution
        all_sentiments = headline_sentiments + content_sentiments
        sentiment_distribution = {
            'positive': sum(1 for s in all_sentiments if s > 0.1),
            'negative': sum(1 for s in all_sentiments if s < -0.1),
            'neutral': sum(1 for s in all_sentiments if -0.1 <= s <= 0.1)
        }
        
        return NewsAnalysisResult(
            headline_sentiment=avg_headline_sentiment,
            content_sentiment=avg_content_sentiment,
            source_credibility=avg_source_credibility,
            impact_score=avg_impact_score,
            key_topics=key_topics,
            entity_sentiment=entity_sentiment_avg,
            news_volume=len(news_data),
            sentiment_distribution=sentiment_distribution
        )
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from text (simplified)"""
        crypto_topics = [
            'bitcoin', 'ethereum', 'defi', 'nft', 'regulation', 'adoption',
            'institutional', 'mining', 'staking', 'halving', 'upgrade', 'fork'
        ]
        
        text_lower = text.lower()
        found_topics = [topic for topic in crypto_topics if topic in text_lower]
        return found_topics
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities from text (simplified)"""
        entities = [
            'bitcoin', 'btc', 'ethereum', 'eth', 'binance', 'coinbase',
            'tesla', 'microstrategy', 'sec', 'fed', 'china', 'usa'
        ]
        
        text_lower = text.lower()
        found_entities = [entity for entity in entities if entity in text_lower]
        return found_entities
    
    def _create_neutral_news_analysis(self) -> NewsAnalysisResult:
        """Create neutral news analysis for edge cases"""
        return NewsAnalysisResult(
            headline_sentiment=0.0,
            content_sentiment=0.0,
            source_credibility=0.5,
            impact_score=0.0,
            key_topics=[],
            entity_sentiment={},
            news_volume=0,
            sentiment_distribution={'positive': 0, 'negative': 0, 'neutral': 0}
        )

class SocialMediaAnalyzer:
    """Analyzes social media sentiment"""
    
    def __init__(self):
        self.cryptobert = CryptoBERTModel()
        
        # Platform weights
        self.platform_weights = {
            'twitter': 0.35,
            'reddit': 0.25,
            'telegram': 0.20,
            'discord': 0.15,
            'other': 0.05
        }
    
    def analyze_social_media(self, social_data: List[Dict]) -> SocialMediaResult:
        """Analyze social media sentiment
        
        Args:
            social_data: List of social media posts with 'text', 'platform', 'engagement', 'timestamp'
        """
        if not social_data:
            return self._create_neutral_social_analysis()
        
        platform_sentiments = defaultdict(list)
        engagement_weights = []
        all_topics = []
        
        for post in social_data:
            text = post.get('text', '')
            platform = post.get('platform', 'other').lower()
            engagement = post.get('engagement', 1)
            
            if text:
                sentiment = self.cryptobert.analyze_crypto_sentiment(text, f'social_{platform}')
                platform_sentiments[platform].append(sentiment.compound)
                engagement_weights.append((sentiment.compound, engagement))
                
                # Extract trending topics
                topics = self._extract_trending_topics(text)
                all_topics.extend(topics)
        
        # Calculate platform-specific sentiments
        twitter_sentiment = np.mean(platform_sentiments.get('twitter', [0]))
        reddit_sentiment = np.mean(platform_sentiments.get('reddit', [0]))
        telegram_sentiment = np.mean(platform_sentiments.get('telegram', [0]))
        discord_sentiment = np.mean(platform_sentiments.get('discord', [0]))
        
        # Influencer sentiment (simplified - assume high engagement posts are from influencers)
        high_engagement_posts = [sentiment for sentiment, engagement in engagement_weights if engagement > 100]
        influencer_sentiment = np.mean(high_engagement_posts) if high_engagement_posts else 0
        
        # Viral content sentiment (very high engagement)
        viral_posts = [sentiment for sentiment, engagement in engagement_weights if engagement > 1000]
        viral_content_sentiment = np.mean(viral_posts) if viral_posts else 0
        
        # Engagement weighted sentiment
        if engagement_weights:
            total_weighted_sentiment = sum(sentiment * engagement for sentiment, engagement in engagement_weights)
            total_engagement = sum(engagement for _, engagement in engagement_weights)
            engagement_weighted_sentiment = total_weighted_sentiment / total_engagement if total_engagement > 0 else 0
        else:
            engagement_weighted_sentiment = 0
        
        # Trending topics
        topic_counts = Counter(all_topics)
        trending_topics = [topic for topic, count in topic_counts.most_common(10)]
        
        return SocialMediaResult(
            twitter_sentiment=twitter_sentiment,
            reddit_sentiment=reddit_sentiment,
            telegram_sentiment=telegram_sentiment,
            discord_sentiment=discord_sentiment,
            influencer_sentiment=influencer_sentiment,
            viral_content_sentiment=viral_content_sentiment,
            engagement_weighted_sentiment=engagement_weighted_sentiment,
            social_volume=len(social_data),
            trending_topics=trending_topics
        )
    
    def _extract_trending_topics(self, text: str) -> List[str]:
        """Extract trending topics from social media text"""
        # Extract hashtags
        hashtags = re.findall(r'#(\w+)', text.lower())
        
        # Extract mentions of popular crypto terms
        crypto_terms = [
            'bitcoin', 'btc', 'ethereum', 'eth', 'altcoin', 'defi', 'nft',
            'moon', 'lambo', 'hodl', 'diamond', 'hands', 'ape', 'degen'
        ]
        
        text_lower = text.lower()
        found_terms = [term for term in crypto_terms if term in text_lower]
        
        return hashtags + found_terms
    
    def _create_neutral_social_analysis(self) -> SocialMediaResult:
        """Create neutral social media analysis for edge cases"""
        return SocialMediaResult(
            twitter_sentiment=0.0,
            reddit_sentiment=0.0,
            telegram_sentiment=0.0,
            discord_sentiment=0.0,
            influencer_sentiment=0.0,
            viral_content_sentiment=0.0,
            engagement_weighted_sentiment=0.0,
            social_volume=0,
            trending_topics=[]
        )

class SentimentSignalGenerator:
    """Generates trading signals based on sentiment analysis"""
    
    def __init__(self):
        pass
    
    def generate_signals(self, 
                        aggregated_sentiment: AggregatedSentiment,
                        news_analysis: NewsAnalysisResult,
                        social_analysis: SocialMediaResult) -> SentimentSignalsResult:
        """Generate trading signals from sentiment data"""
        
        # Calculate individual signal components
        sentiment_signal = self._calculate_sentiment_signal(aggregated_sentiment)
        news_signal = self._calculate_news_signal(news_analysis)
        social_signal = self._calculate_social_signal(social_analysis)
        contrarian_signal = self._calculate_contrarian_signal(aggregated_sentiment)
        
        # Combine signals with weights
        buy_signal_strength = (
            sentiment_signal['buy'] * 0.4 +
            news_signal['buy'] * 0.35 +
            social_signal['buy'] * 0.25
        )
        
        sell_signal_strength = (
            sentiment_signal['sell'] * 0.4 +
            news_signal['sell'] * 0.35 +
            social_signal['sell'] * 0.25
        )
        
        hold_signal_strength = 1.0 - buy_signal_strength - sell_signal_strength
        
        # Sentiment momentum
        sentiment_momentum = self._calculate_momentum(aggregated_sentiment)
        
        # Crowd psychology phase
        crowd_psychology_phase = self._determine_crowd_psychology(aggregated_sentiment)
        
        # Market sentiment cycle
        market_sentiment_cycle = self._determine_sentiment_cycle(aggregated_sentiment)
        
        # Signal confidence
        signal_confidence = (
            aggregated_sentiment.confidence_score * 0.5 +
            news_analysis.source_credibility * 0.3 +
            min(social_analysis.social_volume / 1000, 1.0) * 0.2
        )
        
        return SentimentSignalsResult(
            buy_signal_strength=max(0, min(1, buy_signal_strength)),
            sell_signal_strength=max(0, min(1, sell_signal_strength)),
            hold_signal_strength=max(0, min(1, hold_signal_strength)),
            sentiment_momentum=sentiment_momentum,
            contrarian_signal=contrarian_signal,
            crowd_psychology_phase=crowd_psychology_phase,
            market_sentiment_cycle=market_sentiment_cycle,
            signal_confidence=signal_confidence
        )
    
    def _calculate_sentiment_signal(self, sentiment: AggregatedSentiment) -> Dict[str, float]:
        """Calculate signal from aggregated sentiment"""
        if sentiment.overall_sentiment > 0.3:
            return {'buy': 0.8, 'sell': 0.1}
        elif sentiment.overall_sentiment < -0.3:
            return {'buy': 0.1, 'sell': 0.8}
        else:
            return {'buy': 0.4, 'sell': 0.4}
    
    def _calculate_news_signal(self, news: NewsAnalysisResult) -> Dict[str, float]:
        """Calculate signal from news analysis"""
        weighted_sentiment = (news.headline_sentiment + news.content_sentiment) / 2
        impact_factor = news.impact_score
        
        signal_strength = weighted_sentiment * impact_factor
        
        if signal_strength > 0.2:
            return {'buy': 0.7, 'sell': 0.2}
        elif signal_strength < -0.2:
            return {'buy': 0.2, 'sell': 0.7}
        else:
            return {'buy': 0.4, 'sell': 0.4}
    
    def _calculate_social_signal(self, social: SocialMediaResult) -> Dict[str, float]:
        """Calculate signal from social media analysis"""
        # Weight different platforms
        weighted_sentiment = (
            social.twitter_sentiment * 0.4 +
            social.reddit_sentiment * 0.3 +
            social.influencer_sentiment * 0.3
        )
        
        if weighted_sentiment > 0.3:
            return {'buy': 0.6, 'sell': 0.2}
        elif weighted_sentiment < -0.3:
            return {'buy': 0.2, 'sell': 0.6}
        else:
            return {'buy': 0.4, 'sell': 0.4}
    
    def _calculate_contrarian_signal(self, sentiment: AggregatedSentiment) -> float:
        """Calculate contrarian signal (opposite of crowd sentiment)"""
        # Contrarian signal is stronger at extremes
        if sentiment.fear_greed_index > 80:  # Extreme greed
            return 0.8  # Strong contrarian sell signal
        elif sentiment.fear_greed_index < 20:  # Extreme fear
            return -0.8  # Strong contrarian buy signal
        else:
            return 0.0  # No contrarian signal
    
    def _calculate_momentum(self, sentiment: AggregatedSentiment) -> float:
        """Calculate sentiment momentum"""
        if len(sentiment.sentiment_history) < 2:
            return 0.0
        
        recent_sentiment = np.mean(sentiment.sentiment_history[-3:]) if len(sentiment.sentiment_history) >= 3 else sentiment.sentiment_history[-1]
        earlier_sentiment = np.mean(sentiment.sentiment_history[:-3]) if len(sentiment.sentiment_history) >= 6 else sentiment.sentiment_history[0]
        
        momentum = recent_sentiment - earlier_sentiment
        return np.clip(momentum, -1.0, 1.0)
    
    def _determine_crowd_psychology(self, sentiment: AggregatedSentiment) -> str:
        """Determine current crowd psychology phase"""
        fear_greed = sentiment.fear_greed_index
        
        if fear_greed >= 90:
            return "Euphoria - Maximum Optimism"
        elif fear_greed >= 75:
            return "Greed - Optimism Building"
        elif fear_greed >= 60:
            return "Optimism - Positive Sentiment"
        elif fear_greed >= 40:
            return "Neutral - Balanced Sentiment"
        elif fear_greed >= 25:
            return "Pessimism - Negative Sentiment"
        elif fear_greed >= 10:
            return "Fear - Pessimism Building"
        else:
            return "Panic - Maximum Pessimism"
    
    def _determine_sentiment_cycle(self, sentiment: AggregatedSentiment) -> str:
        """Determine market sentiment cycle phase"""
        if sentiment.sentiment_trend == "Improving":
            if sentiment.overall_sentiment > 0:
                return "Bull Market - Sentiment Improving"
            else:
                return "Recovery - Sentiment Bottoming"
        elif sentiment.sentiment_trend == "Deteriorating":
            if sentiment.overall_sentiment > 0:
                return "Distribution - Sentiment Topping"
            else:
                return "Bear Market - Sentiment Declining"
        else:
            return "Consolidation - Sentiment Stable"

class FinBERTCryptoBERTModel:
    """Combined FinBERT/CryptoBERT Analysis Model"""
    
    def __init__(self, asset: str = "BTC"):
        self.asset = asset.upper()
        self.sentiment_aggregator = SentimentAggregator()
        self.news_analyzer = NewsAnalyzer()
        self.social_analyzer = SocialMediaAnalyzer()
        self.signal_generator = SentimentSignalGenerator()
    
    def analyze(self, 
               news_data: List[Dict] = None,
               social_data: List[Dict] = None,
               additional_text_data: List[str] = None) -> FinBERTCryptoBERTResult:
        """Perform comprehensive sentiment analysis
        
        Args:
            news_data: List of news items
            social_data: List of social media posts
            additional_text_data: Additional text data for analysis
        """
        try:
            # Collect all sentiment scores
            all_sentiment_scores = []
            
            # Analyze additional text data
            if additional_text_data:
                for text in additional_text_data:
                    finbert_score = self.sentiment_aggregator.finbert.analyze_sentiment(text, 'general')
                    crypto_score = self.sentiment_aggregator.cryptobert.analyze_crypto_sentiment(text, 'crypto')
                    all_sentiment_scores.extend([finbert_score, crypto_score])
            
            # Analyze news data
            news_analysis = self.news_analyzer.analyze_news(news_data or [])
            
            # Add news sentiment scores
            if news_data:
                for news_item in news_data:
                    headline = news_item.get('headline', '')
                    if headline:
                        score = self.sentiment_aggregator.finbert.analyze_sentiment(headline, 'news')
                        all_sentiment_scores.append(score)
            
            # Analyze social media data
            social_analysis = self.social_analyzer.analyze_social_media(social_data or [])
            
            # Add social sentiment scores
            if social_data:
                for post in social_data:
                    text = post.get('text', '')
                    if text:
                        score = self.sentiment_aggregator.cryptobert.analyze_crypto_sentiment(text, 'social')
                        all_sentiment_scores.append(score)
            
            # Aggregate all sentiment scores
            aggregated_sentiment = self.sentiment_aggregator.aggregate_sentiment_scores(all_sentiment_scores)
            
            # Generate trading signals
            sentiment_signals = self.signal_generator.generate_signals(
                aggregated_sentiment, news_analysis, social_analysis
            )
            
            # Calculate overall metrics
            overall_market_sentiment = self._calculate_overall_sentiment(
                aggregated_sentiment, news_analysis, social_analysis
            )
            
            sentiment_based_price_prediction = self._predict_price_movement(overall_market_sentiment)
            
            risk_sentiment_assessment = self._assess_sentiment_risk(
                aggregated_sentiment, sentiment_signals
            )
            
            confidence_score = self._calculate_overall_confidence(
                aggregated_sentiment, news_analysis, social_analysis
            )
            
            return FinBERTCryptoBERTResult(
                aggregated_sentiment=aggregated_sentiment,
                news_analysis=news_analysis,
                social_media_analysis=social_analysis,
                sentiment_signals=sentiment_signals,
                overall_market_sentiment=overall_market_sentiment,
                sentiment_based_price_prediction=sentiment_based_price_prediction,
                risk_sentiment_assessment=risk_sentiment_assessment,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            logger.error(f"Error in FinBERT/CryptoBERT analysis: {str(e)}")
            raise
    
    def _calculate_overall_sentiment(self, 
                                   aggregated: AggregatedSentiment,
                                   news: NewsAnalysisResult,
                                   social: SocialMediaResult) -> float:
        """Calculate overall market sentiment score"""
        # Weight different sources
        overall = (
            aggregated.overall_sentiment * 0.4 +
            (news.headline_sentiment + news.content_sentiment) / 2 * 0.35 +
            social.engagement_weighted_sentiment * 0.25
        )
        
        return np.clip(overall, -1.0, 1.0)
    
    def _predict_price_movement(self, overall_sentiment: float) -> float:
        """Predict price movement based on sentiment (simplified)"""
        # This is a simplified model - in practice, you'd use more sophisticated methods
        # Returns expected price change percentage
        
        if overall_sentiment > 0.5:
            return 15.0  # Expect 15% increase
        elif overall_sentiment > 0.2:
            return 5.0   # Expect 5% increase
        elif overall_sentiment > -0.2:
            return 0.0   # Expect no change
        elif overall_sentiment > -0.5:
            return -5.0  # Expect 5% decrease
        else:
            return -15.0 # Expect 15% decrease
    
    def _assess_sentiment_risk(self, 
                             aggregated: AggregatedSentiment,
                             signals: SentimentSignalsResult) -> str:
        """Assess risk based on sentiment analysis"""
        risk_factors = 0
        
        # Extreme sentiment risk
        if aggregated.fear_greed_index > 90 or aggregated.fear_greed_index < 10:
            risk_factors += 2
        
        # Low confidence risk
        if aggregated.confidence_score < 0.3:
            risk_factors += 1
        
        # Contrarian signal risk
        if abs(signals.contrarian_signal) > 0.6:
            risk_factors += 1
        
        # Signal conflict risk
        signal_conflict = abs(signals.buy_signal_strength - signals.sell_signal_strength)
        if signal_conflict < 0.2:
            risk_factors += 1
        
        if risk_factors >= 4:
            return "Very High Risk - Multiple sentiment warning signals"
        elif risk_factors >= 2:
            return "High Risk - Some sentiment concerns"
        elif risk_factors >= 1:
            return "Medium Risk - Minor sentiment issues"
        else:
            return "Low Risk - Sentiment indicators stable"
    
    def _calculate_overall_confidence(self, 
                                    aggregated: AggregatedSentiment,
                                    news: NewsAnalysisResult,
                                    social: SocialMediaResult) -> float:
        """Calculate overall confidence in sentiment analysis"""
        confidence_factors = [
            aggregated.confidence_score,
            news.source_credibility,
            min(news.news_volume / 50, 1.0),  # More news = higher confidence
            min(social.social_volume / 1000, 1.0),  # More social data = higher confidence
        ]
        
        return np.mean(confidence_factors)
    
    def get_sentiment_insights(self, result: FinBERTCryptoBERTResult) -> Dict[str, str]:
        """Generate comprehensive sentiment insights"""
        insights = {}
        
        # Overall sentiment
        insights['overall_sentiment'] = f"Market Sentiment: {result.overall_market_sentiment:.2f} ({result.aggregated_sentiment.market_mood})"
        
        # Fear & Greed
        insights['fear_greed'] = f"Fear & Greed Index: {result.aggregated_sentiment.fear_greed_index:.0f}/100"
        
        # Trading signals
        insights['trading_signals'] = f"Buy: {result.sentiment_signals.buy_signal_strength:.1%}, Sell: {result.sentiment_signals.sell_signal_strength:.1%}, Hold: {result.sentiment_signals.hold_signal_strength:.1%}"
        
        # News sentiment
        insights['news_sentiment'] = f"News Impact: {result.news_analysis.impact_score:.2f} (Headlines: {result.news_analysis.headline_sentiment:+.2f})"
        
        # Social sentiment
        insights['social_sentiment'] = f"Social Sentiment: {result.social_media_analysis.engagement_weighted_sentiment:+.2f} (Volume: {result.social_media_analysis.social_volume})"
        
        # Price prediction
        insights['price_prediction'] = f"Sentiment-based Price Prediction: {result.sentiment_based_price_prediction:+.1f}%"
        
        # Risk assessment
        insights['risk_assessment'] = result.risk_sentiment_assessment
        
        # Confidence
        insights['confidence'] = f"Analysis Confidence: {result.confidence_score:.1%}"
        
        return insights

# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    sample_news = [
        {
            'headline': 'Bitcoin reaches new all-time high as institutional adoption grows',
            'content': 'Major corporations continue to add Bitcoin to their treasury reserves...',
            'source': 'reuters',
            'timestamp': datetime.now()
        },
        {
            'headline': 'Regulatory concerns weigh on cryptocurrency markets',
            'content': 'Government officials express concerns about crypto regulation...',
            'source': 'bloomberg',
            'timestamp': datetime.now() - timedelta(hours=2)
        }
    ]
    
    sample_social = [
        {
            'text': 'Bitcoin to the moon! 🚀 Diamond hands! #HODL #BTC',
            'platform': 'twitter',
            'engagement': 150,
            'timestamp': datetime.now()
        },
        {
            'text': 'Thinking about selling my crypto, market looks scary',
            'platform': 'reddit',
            'engagement': 25,
            'timestamp': datetime.now() - timedelta(minutes=30)
        }
    ]
    
    sample_texts = [
        "The cryptocurrency market is showing strong bullish momentum",
        "Investors are becoming increasingly bearish on digital assets",
        "Bitcoin adoption by institutions continues to accelerate"
    ]
    
    # Test the model
    sentiment_model = FinBERTCryptoBERTModel("BTC")
    result = sentiment_model.analyze(
        news_data=sample_news,
        social_data=sample_social,
        additional_text_data=sample_texts
    )
    
    insights = sentiment_model.get_sentiment_insights(result)
    
    print("=== FinBERT/CryptoBERT Sentiment Analysis ===")
    print(f"Overall Market Sentiment: {result.overall_market_sentiment:.2f}")
    print(f"Fear & Greed Index: {result.aggregated_sentiment.fear_greed_index:.0f}/100")
    print(f"Market Mood: {result.aggregated_sentiment.market_mood}")
    print(f"Price Prediction: {result.sentiment_based_price_prediction:+.1f}%")
    print(f"Risk Assessment: {result.risk_sentiment_assessment}")
    print(f"Confidence: {result.confidence_score:.1%}")
    print()
    
    print("=== Detailed Insights ===")
    for key, insight in insights.items():
        print(f"{key}: {insight}")