# Sentiment Predictor
# Phase 9: AI-First Platform Implementation

import logging
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class SentimentType(Enum):
    MARKET_SENTIMENT = "market_sentiment"
    NEWS_SENTIMENT = "news_sentiment"
    SOCIAL_SENTIMENT = "social_sentiment"
    ANALYST_SENTIMENT = "analyst_sentiment"
    EARNINGS_SENTIMENT = "earnings_sentiment"
    ECONOMIC_SENTIMENT = "economic_sentiment"
    SECTOR_SENTIMENT = "sector_sentiment"
    CRYPTO_SENTIMENT = "crypto_sentiment"

class SentimentPolarity(Enum):
    VERY_NEGATIVE = "very_negative"    # -1.0 to -0.6
    NEGATIVE = "negative"              # -0.6 to -0.2
    NEUTRAL = "neutral"                # -0.2 to 0.2
    POSITIVE = "positive"              # 0.2 to 0.6
    VERY_POSITIVE = "very_positive"    # 0.6 to 1.0

class SentimentSource(Enum):
    NEWS_ARTICLES = "news_articles"
    SOCIAL_MEDIA = "social_media"
    ANALYST_REPORTS = "analyst_reports"
    EARNINGS_CALLS = "earnings_calls"
    REGULATORY_FILINGS = "regulatory_filings"
    MARKET_DATA = "market_data"
    ECONOMIC_REPORTS = "economic_reports"
    FORUM_DISCUSSIONS = "forum_discussions"

class SentimentConfidence(Enum):
    LOW = "low"                        # 0-50%
    MODERATE = "moderate"              # 50-75%
    HIGH = "high"                      # 75-90%
    VERY_HIGH = "very_high"            # 90%+

@dataclass
class SentimentData:
    text: str
    source: SentimentSource
    timestamp: datetime
    author: Optional[str]
    reach: int  # Number of views/shares
    relevance_score: float  # 0-1
    language: str
    metadata: Dict[str, Any]

@dataclass
class SentimentScore:
    polarity: SentimentPolarity
    score: float  # -1 to 1
    confidence: SentimentConfidence
    confidence_score: float  # 0-1
    keywords: List[str]
    emotions: Dict[str, float]  # fear, greed, optimism, etc.
    analysis_timestamp: datetime

@dataclass
class SentimentAnalysis:
    sentiment_type: SentimentType
    overall_sentiment: SentimentScore
    source_breakdown: Dict[SentimentSource, SentimentScore]
    temporal_trend: List[Tuple[datetime, float]]  # Time series of sentiment
    key_themes: List[str]
    influential_content: List[SentimentData]
    market_impact_prediction: float  # -1 to 1
    volatility_prediction: float  # 0 to 1
    analysis_period: Tuple[datetime, datetime]
    sample_size: int

@dataclass
class SentimentAlert:
    alert_type: str
    severity: str  # low, medium, high, critical
    message: str
    sentiment_change: float
    time_window: str
    affected_assets: List[str]
    recommended_actions: List[str]
    alert_timestamp: datetime

class SentimentPredictor:
    """Advanced sentiment analysis and prediction engine"""
    
    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        self.sentiment_cache = {}
        self.alert_thresholds = {
            'sentiment_change': 0.3,
            'volatility_spike': 0.5,
            'volume_surge': 2.0
        }
        
        # Initialize sentiment lexicons
        self.financial_lexicon = self._initialize_financial_lexicon()
        self.emotion_lexicon = self._initialize_emotion_lexicon()
        
        # Initialize models
        self._initialize_models()
        
        logger.info("Sentiment predictor initialized")
    
    async def analyze_sentiment(self, sentiment_type: SentimentType,
                              data_sources: List[SentimentSource],
                              time_window: timedelta = timedelta(days=7)) -> Optional[SentimentAnalysis]:
        """Analyze sentiment from multiple sources"""
        try:
            # Collect sentiment data
            sentiment_data = await self._collect_sentiment_data(
                sentiment_type, data_sources, time_window
            )
            
            if not sentiment_data:
                logger.warning(f"No sentiment data found for {sentiment_type.value}")
                return None
            
            # Analyze overall sentiment
            overall_sentiment = await self._analyze_overall_sentiment(sentiment_data)
            
            # Analyze by source
            source_breakdown = await self._analyze_by_source(sentiment_data)
            
            # Generate temporal trend
            temporal_trend = await self._generate_temporal_trend(sentiment_data)
            
            # Extract key themes
            key_themes = await self._extract_key_themes(sentiment_data)
            
            # Identify influential content
            influential_content = await self._identify_influential_content(sentiment_data)
            
            # Predict market impact
            market_impact = await self._predict_market_impact(overall_sentiment, sentiment_data)
            
            # Predict volatility
            volatility_prediction = await self._predict_volatility(overall_sentiment, sentiment_data)
            
            # Determine analysis period
            timestamps = [data.timestamp for data in sentiment_data]
            analysis_period = (min(timestamps), max(timestamps))
            
            return SentimentAnalysis(
                sentiment_type=sentiment_type,
                overall_sentiment=overall_sentiment,
                source_breakdown=source_breakdown,
                temporal_trend=temporal_trend,
                key_themes=key_themes,
                influential_content=influential_content,
                market_impact_prediction=market_impact,
                volatility_prediction=volatility_prediction,
                analysis_period=analysis_period,
                sample_size=len(sentiment_data)
            )
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return None
    
    async def predict_sentiment_impact(self, sentiment_analysis: SentimentAnalysis,
                                     asset_symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """Predict sentiment impact on specific assets"""
        try:
            impact_predictions = {}
            
            for symbol in asset_symbols:
                try:
                    # Get asset-specific sentiment factors
                    asset_factors = await self._get_asset_sentiment_factors(symbol)
                    
                    # Calculate base impact from overall sentiment
                    base_impact = sentiment_analysis.market_impact_prediction
                    
                    # Adjust for asset-specific factors
                    adjusted_impact = base_impact * asset_factors.get('sensitivity', 1.0)
                    
                    # Calculate volatility impact
                    volatility_impact = sentiment_analysis.volatility_prediction * asset_factors.get('volatility_sensitivity', 1.0)
                    
                    # Calculate directional bias
                    directional_bias = self._calculate_directional_bias(
                        sentiment_analysis.overall_sentiment, asset_factors
                    )
                    
                    # Calculate time decay
                    time_decay = self._calculate_time_decay(sentiment_analysis.analysis_period)
                    
                    impact_predictions[symbol] = {
                        'price_impact': adjusted_impact * time_decay,
                        'volatility_impact': volatility_impact * time_decay,
                        'directional_bias': directional_bias,
                        'confidence': sentiment_analysis.overall_sentiment.confidence_score,
                        'time_horizon_hours': 24 * time_decay
                    }
                    
                except Exception as e:
                    logger.error(f"Error predicting impact for {symbol}: {e}")
                    continue
            
            return impact_predictions
            
        except Exception as e:
            logger.error(f"Error predicting sentiment impact: {e}")
            return {}
    
    async def monitor_sentiment_alerts(self, sentiment_types: List[SentimentType],
                                     monitoring_window: timedelta = timedelta(hours=1)) -> List[SentimentAlert]:
        """Monitor for sentiment-based alerts"""
        try:
            alerts = []
            
            for sentiment_type in sentiment_types:
                try:
                    # Get recent sentiment data
                    recent_analysis = await self.analyze_sentiment(
                        sentiment_type, 
                        list(SentimentSource), 
                        monitoring_window
                    )
                    
                    if not recent_analysis:
                        continue
                    
                    # Check for sentiment alerts
                    sentiment_alerts = await self._check_sentiment_alerts(
                        recent_analysis, sentiment_type
                    )
                    
                    alerts.extend(sentiment_alerts)
                    
                except Exception as e:
                    logger.error(f"Error monitoring {sentiment_type.value}: {e}")
                    continue
            
            # Sort alerts by severity
            severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
            alerts.sort(key=lambda x: severity_order.get(x.severity, 4))
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error monitoring sentiment alerts: {e}")
            return []
    
    async def generate_sentiment_report(self, sentiment_analyses: List[SentimentAnalysis]) -> Dict[str, Any]:
        """Generate comprehensive sentiment report"""
        try:
            if not sentiment_analyses:
                return {}
            
            # Calculate aggregate metrics
            overall_sentiment = np.mean([analysis.overall_sentiment.score for analysis in sentiment_analyses])
            sentiment_volatility = np.std([analysis.overall_sentiment.score for analysis in sentiment_analyses])
            
            # Identify dominant themes
            all_themes = []
            for analysis in sentiment_analyses:
                all_themes.extend(analysis.key_themes)
            theme_counts = Counter(all_themes)
            dominant_themes = [theme for theme, count in theme_counts.most_common(10)]
            
            # Calculate source reliability
            source_reliability = await self._calculate_source_reliability(sentiment_analyses)
            
            # Generate trend analysis
            trend_analysis = await self._analyze_sentiment_trends(sentiment_analyses)
            
            # Calculate market correlation
            market_correlation = await self._calculate_market_correlation(sentiment_analyses)
            
            # Generate recommendations
            recommendations = await self._generate_sentiment_recommendations(sentiment_analyses)
            
            report = {
                'summary': {
                    'overall_sentiment': overall_sentiment,
                    'sentiment_volatility': sentiment_volatility,
                    'dominant_polarity': self._score_to_polarity(overall_sentiment).value,
                    'analysis_count': len(sentiment_analyses),
                    'time_range': {
                        'start': min(analysis.analysis_period[0] for analysis in sentiment_analyses),
                        'end': max(analysis.analysis_period[1] for analysis in sentiment_analyses)
                    }
                },
                'themes': {
                    'dominant_themes': dominant_themes,
                    'theme_distribution': dict(theme_counts.most_common(20))
                },
                'sources': {
                    'reliability_scores': source_reliability,
                    'coverage': self._calculate_source_coverage(sentiment_analyses)
                },
                'trends': trend_analysis,
                'market_impact': {
                    'correlation_score': market_correlation,
                    'predicted_volatility': np.mean([analysis.volatility_prediction for analysis in sentiment_analyses]),
                    'market_direction_bias': np.mean([analysis.market_impact_prediction for analysis in sentiment_analyses])
                },
                'recommendations': recommendations,
                'report_timestamp': datetime.now()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating sentiment report: {e}")
            return {}
    
    def _initialize_models(self):
        """Initialize ML models for sentiment analysis"""
        try:
            # Text vectorizer
            self.vectorizers['tfidf'] = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words='english',
                lowercase=True
            )
            
            # Classification models
            self.models['random_forest'] = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
            
            self.models['logistic_regression'] = LogisticRegression(
                random_state=42,
                max_iter=1000
            )
            
            self.models['naive_bayes'] = MultinomialNB()
            
            # Feature scaler
            self.models['scaler'] = StandardScaler()
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    def _initialize_financial_lexicon(self) -> Dict[str, float]:
        """Initialize financial sentiment lexicon"""
        try:
            # Financial sentiment words with scores (-1 to 1)
            lexicon = {
                # Positive financial terms
                'profit': 0.8, 'growth': 0.7, 'bullish': 0.9, 'rally': 0.8,
                'surge': 0.7, 'boom': 0.8, 'gains': 0.6, 'uptrend': 0.7,
                'outperform': 0.6, 'beat': 0.5, 'strong': 0.6, 'robust': 0.7,
                'recovery': 0.5, 'expansion': 0.6, 'momentum': 0.5,
                
                # Negative financial terms
                'loss': -0.8, 'decline': -0.6, 'bearish': -0.9, 'crash': -1.0,
                'plunge': -0.8, 'recession': -0.9, 'downturn': -0.7, 'fall': -0.5,
                'underperform': -0.6, 'miss': -0.5, 'weak': -0.6, 'volatile': -0.4,
                'uncertainty': -0.5, 'risk': -0.4, 'concern': -0.5, 'worry': -0.6,
                
                # Neutral but important terms
                'earnings': 0.0, 'revenue': 0.0, 'guidance': 0.0, 'forecast': 0.0,
                'analyst': 0.0, 'estimate': 0.0, 'target': 0.0, 'rating': 0.0
            }
            
            return lexicon
            
        except Exception as e:
            logger.error(f"Error initializing financial lexicon: {e}")
            return {}
    
    def _initialize_emotion_lexicon(self) -> Dict[str, Dict[str, float]]:
        """Initialize emotion detection lexicon"""
        try:
            # Emotion categories with associated words
            emotions = {
                'fear': {
                    'panic': 0.9, 'scared': 0.7, 'worried': 0.6, 'anxious': 0.6,
                    'nervous': 0.5, 'uncertain': 0.4, 'doubt': 0.5, 'concern': 0.6
                },
                'greed': {
                    'fomo': 0.8, 'euphoria': 0.9, 'bubble': 0.7, 'speculation': 0.6,
                    'hype': 0.7, 'mania': 0.8, 'overvalued': 0.5, 'irrational': 0.6
                },
                'optimism': {
                    'confident': 0.7, 'positive': 0.6, 'hopeful': 0.6, 'bullish': 0.8,
                    'optimistic': 0.8, 'encouraging': 0.6, 'promising': 0.7
                },
                'pessimism': {
                    'negative': 0.6, 'bearish': 0.8, 'pessimistic': 0.8, 'gloomy': 0.7,
                    'discouraging': 0.6, 'disappointing': 0.6, 'concerning': 0.5
                }
            }
            
            return emotions
            
        except Exception as e:
            logger.error(f"Error initializing emotion lexicon: {e}")
            return {}
    
    async def _collect_sentiment_data(self, sentiment_type: SentimentType,
                                    data_sources: List[SentimentSource],
                                    time_window: timedelta) -> List[SentimentData]:
        """Collect sentiment data from various sources"""
        try:
            # In a real implementation, this would fetch from APIs, databases, etc.
            # For now, we'll generate synthetic but realistic data
            
            sentiment_data = []
            end_time = datetime.now()
            start_time = end_time - time_window
            
            # Generate synthetic data for each source
            for source in data_sources:
                source_data = await self._generate_synthetic_sentiment_data(
                    sentiment_type, source, start_time, end_time
                )
                sentiment_data.extend(source_data)
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Error collecting sentiment data: {e}")
            return []
    
    async def _generate_synthetic_sentiment_data(self, sentiment_type: SentimentType,
                                               source: SentimentSource,
                                               start_time: datetime,
                                               end_time: datetime) -> List[SentimentData]:
        """Generate synthetic sentiment data for testing"""
        try:
            data = []
            
            # Number of data points based on source type
            source_volumes = {
                SentimentSource.SOCIAL_MEDIA: 100,
                SentimentSource.NEWS_ARTICLES: 20,
                SentimentSource.ANALYST_REPORTS: 5,
                SentimentSource.FORUM_DISCUSSIONS: 50,
                SentimentSource.EARNINGS_CALLS: 2
            }
            
            num_points = source_volumes.get(source, 10)
            
            # Sample texts based on sentiment type and source
            sample_texts = self._get_sample_texts(sentiment_type, source)
            
            for i in range(num_points):
                # Random timestamp within window
                time_delta = (end_time - start_time) * np.random.random()
                timestamp = start_time + time_delta
                
                # Random text from samples
                text = np.random.choice(sample_texts)
                
                # Random reach based on source
                reach_ranges = {
                    SentimentSource.SOCIAL_MEDIA: (10, 10000),
                    SentimentSource.NEWS_ARTICLES: (1000, 100000),
                    SentimentSource.ANALYST_REPORTS: (500, 5000),
                    SentimentSource.FORUM_DISCUSSIONS: (50, 5000)
                }
                
                reach_range = reach_ranges.get(source, (10, 1000))
                reach = np.random.randint(reach_range[0], reach_range[1])
                
                # Random relevance score
                relevance_score = np.random.beta(2, 2)  # Tends toward middle values
                
                data.append(SentimentData(
                    text=text,
                    source=source,
                    timestamp=timestamp,
                    author=f"user_{i}",
                    reach=reach,
                    relevance_score=relevance_score,
                    language="en",
                    metadata={"synthetic": True}
                ))
            
            return data
            
        except Exception as e:
            logger.error(f"Error generating synthetic data: {e}")
            return []
    
    def _get_sample_texts(self, sentiment_type: SentimentType, source: SentimentSource) -> List[str]:
        """Get sample texts for synthetic data generation"""
        try:
            # Sample texts based on sentiment type and source
            if sentiment_type == SentimentType.MARKET_SENTIMENT:
                if source == SentimentSource.SOCIAL_MEDIA:
                    return [
                        "Market looking bullish today! 📈",
                        "This volatility is making me nervous",
                        "Great earnings report, stock should rally",
                        "Fed policy uncertainty weighing on markets",
                        "Technical analysis suggests uptrend continuation",
                        "Bearish divergence forming, time to be cautious",
                        "Strong volume surge indicates institutional buying",
                        "Market sentiment seems overly optimistic"
                    ]
                elif source == SentimentSource.NEWS_ARTICLES:
                    return [
                        "Markets rally on positive economic data",
                        "Concerns over inflation impact market sentiment",
                        "Strong earnings drive sector outperformance",
                        "Geopolitical tensions weigh on investor confidence",
                        "Central bank policy supports market optimism",
                        "Analysts upgrade sector outlook on growth prospects"
                    ]
            
            # Default generic texts
            return [
                "Market conditions remain uncertain",
                "Economic indicators show mixed signals",
                "Investor sentiment appears cautious",
                "Trading volumes suggest moderate interest",
                "Technical patterns indicate consolidation"
            ]
            
        except Exception as e:
            logger.error(f"Error getting sample texts: {e}")
            return ["Market update"]
    
    async def _analyze_overall_sentiment(self, sentiment_data: List[SentimentData]) -> SentimentScore:
        """Analyze overall sentiment from collected data"""
        try:
            if not sentiment_data:
                return self._create_neutral_sentiment()
            
            # Extract text for analysis
            texts = [data.text for data in sentiment_data]
            
            # Calculate lexicon-based sentiment
            lexicon_scores = [self._calculate_lexicon_sentiment(text) for text in texts]
            
            # Weight by relevance and reach
            weights = []
            for data in sentiment_data:
                weight = data.relevance_score * np.log(1 + data.reach)
                weights.append(weight)
            
            # Calculate weighted average
            if sum(weights) > 0:
                weighted_score = np.average(lexicon_scores, weights=weights)
            else:
                weighted_score = np.mean(lexicon_scores)
            
            # Determine polarity
            polarity = self._score_to_polarity(weighted_score)
            
            # Calculate confidence based on agreement and sample size
            score_std = np.std(lexicon_scores)
            sample_size_factor = min(1.0, len(sentiment_data) / 100)
            agreement_factor = max(0.1, 1.0 - score_std)
            confidence_score = sample_size_factor * agreement_factor
            confidence_level = self._score_to_confidence(confidence_score)
            
            # Extract keywords
            keywords = self._extract_keywords(texts)
            
            # Analyze emotions
            emotions = self._analyze_emotions(texts)
            
            return SentimentScore(
                polarity=polarity,
                score=weighted_score,
                confidence=confidence_level,
                confidence_score=confidence_score,
                keywords=keywords,
                emotions=emotions,
                analysis_timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error analyzing overall sentiment: {e}")
            return self._create_neutral_sentiment()
    
    def _calculate_lexicon_sentiment(self, text: str) -> float:
        """Calculate sentiment score using financial lexicon"""
        try:
            # Clean and tokenize text
            words = re.findall(r'\b\w+\b', text.lower())
            
            # Calculate sentiment score
            scores = []
            for word in words:
                if word in self.financial_lexicon:
                    scores.append(self.financial_lexicon[word])
            
            if not scores:
                return 0.0
            
            # Return average score, clamped to [-1, 1]
            avg_score = np.mean(scores)
            return max(-1.0, min(1.0, avg_score))
            
        except Exception as e:
            logger.error(f"Error calculating lexicon sentiment: {e}")
            return 0.0
    
    def _score_to_polarity(self, score: float) -> SentimentPolarity:
        """Convert sentiment score to polarity enum"""
        if score <= -0.6:
            return SentimentPolarity.VERY_NEGATIVE
        elif score <= -0.2:
            return SentimentPolarity.NEGATIVE
        elif score <= 0.2:
            return SentimentPolarity.NEUTRAL
        elif score <= 0.6:
            return SentimentPolarity.POSITIVE
        else:
            return SentimentPolarity.VERY_POSITIVE
    
    def _score_to_confidence(self, score: float) -> SentimentConfidence:
        """Convert confidence score to confidence enum"""
        if score >= 0.9:
            return SentimentConfidence.VERY_HIGH
        elif score >= 0.75:
            return SentimentConfidence.HIGH
        elif score >= 0.5:
            return SentimentConfidence.MODERATE
        else:
            return SentimentConfidence.LOW
    
    def _extract_keywords(self, texts: List[str], top_k: int = 10) -> List[str]:
        """Extract key terms from texts"""
        try:
            # Combine all texts
            combined_text = ' '.join(texts).lower()
            
            # Extract words
            words = re.findall(r'\b\w+\b', combined_text)
            
            # Filter for financial terms and meaningful words
            financial_words = [word for word in words if word in self.financial_lexicon]
            
            # Count frequencies
            word_counts = Counter(financial_words)
            
            # Return top keywords
            return [word for word, count in word_counts.most_common(top_k)]
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    def _analyze_emotions(self, texts: List[str]) -> Dict[str, float]:
        """Analyze emotional content of texts"""
        try:
            emotion_scores = {emotion: 0.0 for emotion in self.emotion_lexicon.keys()}
            
            for text in texts:
                words = re.findall(r'\b\w+\b', text.lower())
                
                for emotion, emotion_words in self.emotion_lexicon.items():
                    emotion_score = 0.0
                    for word in words:
                        if word in emotion_words:
                            emotion_score += emotion_words[word]
                    
                    if len(words) > 0:
                        emotion_scores[emotion] += emotion_score / len(words)
            
            # Normalize by number of texts
            if len(texts) > 0:
                for emotion in emotion_scores:
                    emotion_scores[emotion] /= len(texts)
            
            return emotion_scores
            
        except Exception as e:
            logger.error(f"Error analyzing emotions: {e}")
            return {}
    
    def _create_neutral_sentiment(self) -> SentimentScore:
        """Create neutral sentiment score"""
        return SentimentScore(
            polarity=SentimentPolarity.NEUTRAL,
            score=0.0,
            confidence=SentimentConfidence.LOW,
            confidence_score=0.1,
            keywords=[],
            emotions={},
            analysis_timestamp=datetime.now()
        )
    
    async def _analyze_by_source(self, sentiment_data: List[SentimentData]) -> Dict[SentimentSource, SentimentScore]:
        """Analyze sentiment breakdown by source"""
        try:
            source_breakdown = {}
            
            # Group data by source
            source_groups = {}
            for data in sentiment_data:
                if data.source not in source_groups:
                    source_groups[data.source] = []
                source_groups[data.source].append(data)
            
            # Analyze each source
            for source, source_data in source_groups.items():
                source_sentiment = await self._analyze_overall_sentiment(source_data)
                source_breakdown[source] = source_sentiment
            
            return source_breakdown
            
        except Exception as e:
            logger.error(f"Error analyzing by source: {e}")
            return {}
    
    async def _generate_temporal_trend(self, sentiment_data: List[SentimentData]) -> List[Tuple[datetime, float]]:
        """Generate temporal sentiment trend"""
        try:
            # Sort data by timestamp
            sorted_data = sorted(sentiment_data, key=lambda x: x.timestamp)
            
            if not sorted_data:
                return []
            
            # Group by time buckets (hourly)
            time_buckets = {}
            for data in sorted_data:
                # Round to nearest hour
                bucket_time = data.timestamp.replace(minute=0, second=0, microsecond=0)
                
                if bucket_time not in time_buckets:
                    time_buckets[bucket_time] = []
                time_buckets[bucket_time].append(data)
            
            # Calculate sentiment for each bucket
            trend_points = []
            for bucket_time, bucket_data in sorted(time_buckets.items()):
                bucket_sentiment = await self._analyze_overall_sentiment(bucket_data)
                trend_points.append((bucket_time, bucket_sentiment.score))
            
            return trend_points
            
        except Exception as e:
            logger.error(f"Error generating temporal trend: {e}")
            return []
    
    async def _extract_key_themes(self, sentiment_data: List[SentimentData]) -> List[str]:
        """Extract key themes from sentiment data"""
        try:
            # Combine all texts
            all_texts = [data.text for data in sentiment_data]
            
            # Extract keywords
            keywords = self._extract_keywords(all_texts, top_k=20)
            
            # Group related keywords into themes
            themes = []
            
            # Financial performance themes
            performance_words = ['earnings', 'revenue', 'profit', 'growth', 'performance']
            if any(word in keywords for word in performance_words):
                themes.append('Financial Performance')
            
            # Market direction themes
            direction_words = ['bullish', 'bearish', 'rally', 'decline', 'trend']
            if any(word in keywords for word in direction_words):
                themes.append('Market Direction')
            
            # Risk and volatility themes
            risk_words = ['risk', 'volatility', 'uncertainty', 'concern']
            if any(word in keywords for word in risk_words):
                themes.append('Risk and Volatility')
            
            # Policy and regulation themes
            policy_words = ['fed', 'policy', 'regulation', 'government']
            if any(word in keywords for word in policy_words):
                themes.append('Policy and Regulation')
            
            return themes[:10]  # Return top 10 themes
            
        except Exception as e:
            logger.error(f"Error extracting themes: {e}")
            return []
    
    async def _identify_influential_content(self, sentiment_data: List[SentimentData]) -> List[SentimentData]:
        """Identify most influential content"""
        try:
            # Score content by influence (reach * relevance)
            scored_content = []
            for data in sentiment_data:
                influence_score = data.reach * data.relevance_score
                scored_content.append((influence_score, data))
            
            # Sort by influence score
            scored_content.sort(key=lambda x: x[0], reverse=True)
            
            # Return top 10 most influential
            return [data for score, data in scored_content[:10]]
            
        except Exception as e:
            logger.error(f"Error identifying influential content: {e}")
            return []
    
    async def _predict_market_impact(self, overall_sentiment: SentimentScore,
                                   sentiment_data: List[SentimentData]) -> float:
        """Predict market impact of sentiment"""
        try:
            # Base impact from sentiment score
            base_impact = overall_sentiment.score * 0.5
            
            # Adjust for confidence
            confidence_multiplier = overall_sentiment.confidence_score
            
            # Adjust for volume of sentiment data
            volume_factor = min(1.0, len(sentiment_data) / 100)
            
            # Adjust for source diversity
            unique_sources = len(set(data.source for data in sentiment_data))
            source_diversity = min(1.0, unique_sources / len(SentimentSource))
            
            # Calculate final impact
            market_impact = base_impact * confidence_multiplier * volume_factor * source_diversity
            
            return max(-1.0, min(1.0, market_impact))
            
        except Exception as e:
            logger.error(f"Error predicting market impact: {e}")
            return 0.0
    
    async def _predict_volatility(self, overall_sentiment: SentimentScore,
                                sentiment_data: List[SentimentData]) -> float:
        """Predict volatility impact of sentiment"""
        try:
            # Base volatility from sentiment extremity
            sentiment_extremity = abs(overall_sentiment.score)
            
            # Volatility from emotion analysis
            fear_level = overall_sentiment.emotions.get('fear', 0.0)
            greed_level = overall_sentiment.emotions.get('greed', 0.0)
            emotion_volatility = (fear_level + greed_level) * 0.5
            
            # Volatility from sentiment disagreement
            sentiment_scores = [self._calculate_lexicon_sentiment(data.text) for data in sentiment_data]
            disagreement = np.std(sentiment_scores) if sentiment_scores else 0.0
            
            # Combine factors
            volatility_prediction = (sentiment_extremity + emotion_volatility + disagreement) / 3
            
            return max(0.0, min(1.0, volatility_prediction))
            
        except Exception as e:
            logger.error(f"Error predicting volatility: {e}")
            return 0.0
    
    async def _get_asset_sentiment_factors(self, symbol: str) -> Dict[str, float]:
        """Get asset-specific sentiment factors"""
        try:
            # Default factors
            factors = {
                'sensitivity': 1.0,
                'volatility_sensitivity': 1.0,
                'sector_correlation': 0.5
            }
            
            # Adjust based on asset type
            if symbol in ['TSLA', 'NVDA', 'AAPL']:  # High-sentiment stocks
                factors['sensitivity'] = 1.5
                factors['volatility_sensitivity'] = 1.3
            elif symbol in ['TLT', 'IEF']:  # Bonds
                factors['sensitivity'] = 0.3
                factors['volatility_sensitivity'] = 0.5
            elif symbol in ['GLD', 'SLV']:  # Safe havens
                factors['sensitivity'] = -0.5  # Inverse correlation
                factors['volatility_sensitivity'] = 0.8
            
            return factors
            
        except Exception as e:
            logger.error(f"Error getting asset factors: {e}")
            return {'sensitivity': 1.0, 'volatility_sensitivity': 1.0}
    
    def _calculate_directional_bias(self, sentiment: SentimentScore,
                                  asset_factors: Dict[str, float]) -> float:
        """Calculate directional bias for asset"""
        try:
            base_bias = sentiment.score
            sensitivity = asset_factors.get('sensitivity', 1.0)
            
            return base_bias * sensitivity
            
        except Exception as e:
            logger.error(f"Error calculating directional bias: {e}")
            return 0.0
    
    def _calculate_time_decay(self, analysis_period: Tuple[datetime, datetime]) -> float:
        """Calculate time decay factor for sentiment impact"""
        try:
            # Calculate how recent the analysis is
            end_time = analysis_period[1]
            hours_since = (datetime.now() - end_time).total_seconds() / 3600
            
            # Exponential decay with 24-hour half-life
            decay_factor = np.exp(-hours_since / 24)
            
            return max(0.1, min(1.0, decay_factor))
            
        except Exception as e:
            logger.error(f"Error calculating time decay: {e}")
            return 0.5
    
    async def _check_sentiment_alerts(self, analysis: SentimentAnalysis,
                                    sentiment_type: SentimentType) -> List[SentimentAlert]:
        """Check for sentiment-based alerts"""
        try:
            alerts = []
            
            # Check for extreme sentiment
            if abs(analysis.overall_sentiment.score) > 0.7:
                severity = 'high' if abs(analysis.overall_sentiment.score) > 0.8 else 'medium'
                alerts.append(SentimentAlert(
                    alert_type='extreme_sentiment',
                    severity=severity,
                    message=f'Extreme {analysis.overall_sentiment.polarity.value} sentiment detected',
                    sentiment_change=analysis.overall_sentiment.score,
                    time_window='1h',
                    affected_assets=['MARKET'],
                    recommended_actions=['Monitor closely', 'Consider position adjustments'],
                    alert_timestamp=datetime.now()
                ))
            
            # Check for high volatility prediction
            if analysis.volatility_prediction > 0.6:
                alerts.append(SentimentAlert(
                    alert_type='volatility_warning',
                    severity='medium',
                    message='High volatility predicted based on sentiment analysis',
                    sentiment_change=analysis.volatility_prediction,
                    time_window='24h',
                    affected_assets=['MARKET'],
                    recommended_actions=['Reduce position sizes', 'Increase hedging'],
                    alert_timestamp=datetime.now()
                ))
            
            # Check for rapid sentiment change
            if len(analysis.temporal_trend) > 2:
                recent_change = analysis.temporal_trend[-1][1] - analysis.temporal_trend[-2][1]
                if abs(recent_change) > self.alert_thresholds['sentiment_change']:
                    alerts.append(SentimentAlert(
                        alert_type='sentiment_shift',
                        severity='medium',
                        message=f'Rapid sentiment shift detected: {recent_change:.2f}',
                        sentiment_change=recent_change,
                        time_window='1h',
                        affected_assets=['MARKET'],
                        recommended_actions=['Investigate cause', 'Review positions'],
                        alert_timestamp=datetime.now()
                    ))
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error checking sentiment alerts: {e}")
            return []
    
    async def _calculate_source_reliability(self, analyses: List[SentimentAnalysis]) -> Dict[str, float]:
        """Calculate reliability scores for different sources"""
        try:
            source_scores = {}
            
            for source in SentimentSource:
                scores = []
                for analysis in analyses:
                    if source in analysis.source_breakdown:
                        source_sentiment = analysis.source_breakdown[source]
                        scores.append(source_sentiment.confidence_score)
                
                if scores:
                    source_scores[source.value] = np.mean(scores)
                else:
                    source_scores[source.value] = 0.0
            
            return source_scores
            
        except Exception as e:
            logger.error(f"Error calculating source reliability: {e}")
            return {}
    
    def _calculate_source_coverage(self, analyses: List[SentimentAnalysis]) -> Dict[str, int]:
        """Calculate source coverage statistics"""
        try:
            coverage = {}
            
            for source in SentimentSource:
                count = sum(1 for analysis in analyses if source in analysis.source_breakdown)
                coverage[source.value] = count
            
            return coverage
            
        except Exception as e:
            logger.error(f"Error calculating source coverage: {e}")
            return {}
    
    async def _analyze_sentiment_trends(self, analyses: List[SentimentAnalysis]) -> Dict[str, Any]:
        """Analyze sentiment trends across multiple analyses"""
        try:
            # Combine all temporal trends
            all_trends = []
            for analysis in analyses:
                all_trends.extend(analysis.temporal_trend)
            
            if not all_trends:
                return {}
            
            # Sort by time
            all_trends.sort(key=lambda x: x[0])
            
            # Calculate trend statistics
            sentiment_values = [trend[1] for trend in all_trends]
            
            trend_analysis = {
                'overall_direction': 'rising' if sentiment_values[-1] > sentiment_values[0] else 'falling',
                'volatility': np.std(sentiment_values),
                'momentum': sentiment_values[-1] - sentiment_values[0] if len(sentiment_values) > 1 else 0,
                'trend_strength': abs(sentiment_values[-1] - sentiment_values[0]) / len(sentiment_values) if len(sentiment_values) > 1 else 0
            }
            
            return trend_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment trends: {e}")
            return {}
    
    async def _calculate_market_correlation(self, analyses: List[SentimentAnalysis]) -> float:
        """Calculate correlation between sentiment and market impact"""
        try:
            sentiment_scores = [analysis.overall_sentiment.score for analysis in analyses]
            market_impacts = [analysis.market_impact_prediction for analysis in analyses]
            
            if len(sentiment_scores) < 2:
                return 0.0
            
            correlation = np.corrcoef(sentiment_scores, market_impacts)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating market correlation: {e}")
            return 0.0
    
    async def _generate_sentiment_recommendations(self, analyses: List[SentimentAnalysis]) -> List[str]:
        """Generate recommendations based on sentiment analysis"""
        try:
            recommendations = []
            
            # Calculate average sentiment
            avg_sentiment = np.mean([analysis.overall_sentiment.score for analysis in analyses])
            avg_volatility = np.mean([analysis.volatility_prediction for analysis in analyses])
            
            # Sentiment-based recommendations
            if avg_sentiment > 0.5:
                recommendations.append("Consider taking profits on overextended positions")
                recommendations.append("Monitor for signs of excessive optimism")
            elif avg_sentiment < -0.5:
                recommendations.append("Look for oversold opportunities")
                recommendations.append("Consider defensive positioning")
            
            # Volatility-based recommendations
            if avg_volatility > 0.6:
                recommendations.append("Reduce position sizes due to high volatility expectations")
                recommendations.append("Implement volatility hedging strategies")
            
            # General recommendations
            recommendations.append("Continue monitoring sentiment trends")
            recommendations.append("Diversify across sentiment-uncorrelated assets")
            
            return recommendations[:5]
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []

# Export main classes
__all__ = ['SentimentPredictor', 'SentimentAnalysis', 'SentimentType', 'SentimentPolarity',
           'SentimentSource', 'SentimentScore', 'SentimentAlert', 'SentimentData']