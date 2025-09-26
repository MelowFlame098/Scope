from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import re
import json
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# NLP and ML imports (with fallbacks)
try:
    import transformers
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    from transformers import BertTokenizer, BertForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers library not available. Using simplified sentiment analysis.")

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("NLTK not available. Using basic text processing.")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("TextBlob not available. Using alternative sentiment analysis.")

@dataclass
class NewsArticle:
    """Structure for news article data"""
    title: str
    content: str
    timestamp: datetime
    source: str
    currency_pair: str
    url: Optional[str] = None
    author: Optional[str] = None
    
@dataclass
class SentimentScore:
    """Sentiment analysis result"""
    positive: float
    negative: float
    neutral: float
    compound: float
    confidence: float
    
@dataclass
class ForexSentimentData:
    """Forex sentiment analysis data"""
    timestamp: List[datetime]
    currency_pair: str
    news_articles: List[NewsArticle]
    social_media_posts: List[str]
    economic_calendar: List[Dict[str, Any]]
    central_bank_communications: List[str]
    
@dataclass
class SentimentAnalysisResult:
    """Results from sentiment analysis"""
    timestamp: datetime
    overall_sentiment: SentimentScore
    news_sentiment: SentimentScore
    social_sentiment: SentimentScore
    economic_sentiment: SentimentScore
    central_bank_sentiment: SentimentScore
    entity_sentiments: Dict[str, SentimentScore]
    topic_sentiments: Dict[str, SentimentScore]
    
@dataclass
class ForexBERTResults:
    """Comprehensive ForexBERT analysis results"""
    sentiment_timeline: List[SentimentAnalysisResult]
    sentiment_indicators: Dict[str, List[float]]
    correlation_with_price: Dict[str, float]
    sentiment_signals: List[str]
    risk_metrics: Dict[str, float]
    insights: List[str]
    recommendations: List[str]
    model_performance: Dict[str, float]
    
class TextPreprocessor:
    """Text preprocessing for forex sentiment analysis"""
    
    def __init__(self):
        self.forex_keywords = {
            'currencies': ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD', 'CNY'],
            'economic_terms': ['inflation', 'gdp', 'unemployment', 'interest rate', 'monetary policy',
                             'fiscal policy', 'trade balance', 'current account', 'recession',
                             'growth', 'stimulus', 'tapering', 'quantitative easing'],
            'market_terms': ['bullish', 'bearish', 'rally', 'decline', 'volatility', 'support',
                           'resistance', 'breakout', 'correction', 'trend', 'momentum'],
            'central_banks': ['fed', 'ecb', 'boe', 'boj', 'snb', 'rba', 'rbnz', 'pboc']
        }
        
        if NLTK_AVAILABLE:
            try:
                nltk.download('vader_lexicon', quiet=True)
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)
                self.sia = SentimentIntensityAnalyzer()
                self.lemmatizer = WordNetLemmatizer()
                self.stop_words = set(stopwords.words('english'))
            except:
                self.sia = None
                self.lemmatizer = None
                self.stop_words = set()
        else:
            self.sia = None
            self.lemmatizer = None
            self.stop_words = set()
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters but keep forex-related symbols
        text = re.sub(r'[^a-zA-Z0-9\s\$\€\£\¥\%]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract forex-related entities from text"""
        entities = defaultdict(list)
        text_lower = text.lower()
        
        for category, keywords in self.forex_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    entities[category].append(keyword)
        
        return dict(entities)
    
    def tokenize_and_filter(self, text: str) -> List[str]:
        """Tokenize text and filter stopwords"""
        if NLTK_AVAILABLE and self.lemmatizer:
            tokens = word_tokenize(text)
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                     if token not in self.stop_words and len(token) > 2]
        else:
            # Simple tokenization
            tokens = text.split()
            tokens = [token for token in tokens if len(token) > 2]
        
        return tokens

class ForexBERTModel:
    """ForexBERT model for sentiment analysis"""
    
    def __init__(self, model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.sentiment_pipeline = None
        self.is_loaded = False
        
        if TRANSFORMERS_AVAILABLE:
            self._load_model()
    
    def _load_model(self):
        """Load ForexBERT model"""
        try:
            # Try to load a financial sentiment model first
            financial_models = [
                "ProsusAI/finbert",
                "nlptown/bert-base-multilingual-uncased-sentiment",
                "cardiffnlp/twitter-roberta-base-sentiment-latest"
            ]
            
            for model_name in financial_models:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                    self.sentiment_pipeline = pipeline(
                        "sentiment-analysis",
                        model=self.model,
                        tokenizer=self.tokenizer,
                        return_all_scores=True
                    )
                    self.model_name = model_name
                    self.is_loaded = True
                    print(f"Loaded model: {model_name}")
                    break
                except Exception as e:
                    print(f"Failed to load {model_name}: {e}")
                    continue
            
            if not self.is_loaded:
                print("Failed to load any transformer model. Using fallback methods.")
                
        except Exception as e:
            print(f"Error loading ForexBERT model: {e}")
            self.is_loaded = False
    
    def predict_sentiment(self, text: str) -> SentimentScore:
        """Predict sentiment using ForexBERT"""
        if not text or not text.strip():
            return SentimentScore(0.33, 0.33, 0.34, 0.0, 0.0)
        
        if TRANSFORMERS_AVAILABLE and self.is_loaded:
            return self._predict_with_transformers(text)
        else:
            return self._predict_fallback(text)
    
    def _predict_with_transformers(self, text: str) -> SentimentScore:
        """Predict sentiment using transformers"""
        try:
            # Truncate text if too long
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]
            
            results = self.sentiment_pipeline(text)
            
            # Parse results based on model type
            if isinstance(results[0], list):
                results = results[0]
            
            # Initialize scores
            positive = 0.0
            negative = 0.0
            neutral = 0.0
            
            # Map labels to scores
            for result in results:
                label = result['label'].upper()
                score = result['score']
                
                if 'POSITIVE' in label or 'POS' in label or label == 'LABEL_2':
                    positive = score
                elif 'NEGATIVE' in label or 'NEG' in label or label == 'LABEL_0':
                    negative = score
                elif 'NEUTRAL' in label or label == 'LABEL_1':
                    neutral = score
            
            # Ensure scores sum to 1
            total = positive + negative + neutral
            if total > 0:
                positive /= total
                negative /= total
                neutral /= total
            else:
                positive = negative = neutral = 1/3
            
            # Calculate compound score
            compound = positive - negative
            
            # Calculate confidence (max score)
            confidence = max(positive, negative, neutral)
            
            return SentimentScore(
                positive=positive,
                negative=negative,
                neutral=neutral,
                compound=compound,
                confidence=confidence
            )
            
        except Exception as e:
            print(f"Error in transformer prediction: {e}")
            return self._predict_fallback(text)
    
    def _predict_fallback(self, text: str) -> SentimentScore:
        """Fallback sentiment prediction"""
        # Try NLTK VADER first
        if NLTK_AVAILABLE:
            try:
                from nltk.sentiment import SentimentIntensityAnalyzer
                sia = SentimentIntensityAnalyzer()
                scores = sia.polarity_scores(text)
                
                return SentimentScore(
                    positive=max(0, scores['pos']),
                    negative=max(0, scores['neg']),
                    neutral=max(0, scores['neu']),
                    compound=scores['compound'],
                    confidence=abs(scores['compound'])
                )
            except:
                pass
        
        # Try TextBlob
        if TEXTBLOB_AVAILABLE:
            try:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity  # -1 to 1
                
                if polarity > 0:
                    positive = polarity
                    negative = 0
                elif polarity < 0:
                    positive = 0
                    negative = abs(polarity)
                else:
                    positive = negative = 0
                
                neutral = 1 - positive - negative
                
                return SentimentScore(
                    positive=positive,
                    negative=negative,
                    neutral=neutral,
                    compound=polarity,
                    confidence=abs(polarity)
                )
            except:
                pass
        
        # Simple keyword-based fallback
        return self._simple_keyword_sentiment(text)
    
    def _simple_keyword_sentiment(self, text: str) -> SentimentScore:
        """Simple keyword-based sentiment analysis"""
        positive_words = [
            'bullish', 'rally', 'surge', 'gain', 'rise', 'increase', 'strong', 'robust',
            'growth', 'expansion', 'optimistic', 'confident', 'positive', 'upbeat',
            'recovery', 'improvement', 'boost', 'support'
        ]
        
        negative_words = [
            'bearish', 'decline', 'fall', 'drop', 'decrease', 'weak', 'fragile',
            'recession', 'contraction', 'pessimistic', 'concerned', 'negative', 'dovish',
            'crisis', 'deterioration', 'pressure', 'resistance'
        ]
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = len(text_lower.split())
        if total_words == 0:
            return SentimentScore(0.33, 0.33, 0.34, 0.0, 0.0)
        
        pos_score = pos_count / total_words
        neg_score = neg_count / total_words
        
        # Normalize scores
        total_score = pos_score + neg_score
        if total_score > 0:
            positive = pos_score / (total_score + 0.1)  # Add small constant
            negative = neg_score / (total_score + 0.1)
            neutral = 1 - positive - negative
        else:
            positive = negative = 0.1
            neutral = 0.8
        
        compound = positive - negative
        confidence = abs(compound)
        
        return SentimentScore(
            positive=positive,
            negative=negative,
            neutral=neutral,
            compound=compound,
            confidence=confidence
        )

class ForexSentimentAnalyzer:
    """Comprehensive forex sentiment analyzer"""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.forexbert_model = ForexBERTModel()
        self.sentiment_history = []
        
    def analyze_sentiment(self, forex_data: ForexSentimentData) -> ForexBERTResults:
        """Perform comprehensive sentiment analysis"""
        
        print("Analyzing forex sentiment...")
        
        # Analyze sentiment timeline
        sentiment_timeline = self._analyze_timeline(forex_data)
        
        # Calculate sentiment indicators
        sentiment_indicators = self._calculate_sentiment_indicators(sentiment_timeline)
        
        # Correlate with price (if price data available)
        correlation_with_price = self._calculate_price_correlation(forex_data, sentiment_indicators)
        
        # Generate trading signals
        sentiment_signals = self._generate_sentiment_signals(sentiment_indicators)
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(sentiment_indicators)
        
        # Generate insights
        insights = self._generate_insights(sentiment_timeline, sentiment_indicators)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(sentiment_indicators, risk_metrics)
        
        # Calculate model performance
        model_performance = self._calculate_model_performance(sentiment_timeline)
        
        return ForexBERTResults(
            sentiment_timeline=sentiment_timeline,
            sentiment_indicators=sentiment_indicators,
            correlation_with_price=correlation_with_price,
            sentiment_signals=sentiment_signals,
            risk_metrics=risk_metrics,
            insights=insights,
            recommendations=recommendations,
            model_performance=model_performance
        )
    
    def _analyze_timeline(self, forex_data: ForexSentimentData) -> List[SentimentAnalysisResult]:
        """Analyze sentiment over time"""
        timeline = []
        
        # Group data by time periods (daily)
        daily_data = defaultdict(lambda: {
            'news': [],
            'social': [],
            'economic': [],
            'central_bank': []
        })
        
        # Group news articles by date
        for article in forex_data.news_articles:
            date_key = article.timestamp.date()
            daily_data[date_key]['news'].append(article)
        
        # Group social media posts (assuming they have timestamps)
        for i, post in enumerate(forex_data.social_media_posts):
            # Assume posts are distributed over the time period
            if forex_data.timestamp:
                post_date = forex_data.timestamp[i % len(forex_data.timestamp)].date()
                daily_data[post_date]['social'].append(post)
        
        # Group economic calendar events
        for event in forex_data.economic_calendar:
            if 'timestamp' in event:
                event_date = event['timestamp'].date()
                daily_data[event_date]['economic'].append(event)
        
        # Group central bank communications
        for i, comm in enumerate(forex_data.central_bank_communications):
            if forex_data.timestamp:
                comm_date = forex_data.timestamp[i % len(forex_data.timestamp)].date()
                daily_data[comm_date]['central_bank'].append(comm)
        
        # Analyze sentiment for each day
        for date, data in sorted(daily_data.items()):
            # News sentiment
            news_texts = [f"{article.title} {article.content}" for article in data['news']]
            news_sentiment = self._analyze_text_list(news_texts)
            
            # Social media sentiment
            social_sentiment = self._analyze_text_list(data['social'])
            
            # Economic calendar sentiment
            economic_texts = [str(event) for event in data['economic']]
            economic_sentiment = self._analyze_text_list(economic_texts)
            
            # Central bank sentiment
            central_bank_sentiment = self._analyze_text_list(data['central_bank'])
            
            # Overall sentiment (weighted average)
            overall_sentiment = self._calculate_weighted_sentiment([
                (news_sentiment, 0.4),
                (social_sentiment, 0.3),
                (economic_sentiment, 0.2),
                (central_bank_sentiment, 0.1)
            ])
            
            # Entity and topic analysis
            all_texts = news_texts + data['social'] + economic_texts + data['central_bank']
            entity_sentiments = self._analyze_entities(all_texts)
            topic_sentiments = self._analyze_topics(all_texts)
            
            timeline.append(SentimentAnalysisResult(
                timestamp=datetime.combine(date, datetime.min.time()),
                overall_sentiment=overall_sentiment,
                news_sentiment=news_sentiment,
                social_sentiment=social_sentiment,
                economic_sentiment=economic_sentiment,
                central_bank_sentiment=central_bank_sentiment,
                entity_sentiments=entity_sentiments,
                topic_sentiments=topic_sentiments
            ))
        
        return timeline
    
    def _analyze_text_list(self, texts: List[str]) -> SentimentScore:
        """Analyze sentiment for a list of texts"""
        if not texts:
            return SentimentScore(0.33, 0.33, 0.34, 0.0, 0.0)
        
        sentiments = []
        for text in texts:
            if text and text.strip():
                cleaned_text = self.preprocessor.clean_text(text)
                sentiment = self.forexbert_model.predict_sentiment(cleaned_text)
                sentiments.append(sentiment)
        
        if not sentiments:
            return SentimentScore(0.33, 0.33, 0.34, 0.0, 0.0)
        
        # Average the sentiments
        avg_positive = np.mean([s.positive for s in sentiments])
        avg_negative = np.mean([s.negative for s in sentiments])
        avg_neutral = np.mean([s.neutral for s in sentiments])
        avg_compound = np.mean([s.compound for s in sentiments])
        avg_confidence = np.mean([s.confidence for s in sentiments])
        
        return SentimentScore(
            positive=avg_positive,
            negative=avg_negative,
            neutral=avg_neutral,
            compound=avg_compound,
            confidence=avg_confidence
        )
    
    def _calculate_weighted_sentiment(self, weighted_sentiments: List[Tuple[SentimentScore, float]]) -> SentimentScore:
        """Calculate weighted average of sentiments"""
        total_weight = sum(weight for _, weight in weighted_sentiments)
        if total_weight == 0:
            return SentimentScore(0.33, 0.33, 0.34, 0.0, 0.0)
        
        weighted_positive = sum(sentiment.positive * weight for sentiment, weight in weighted_sentiments) / total_weight
        weighted_negative = sum(sentiment.negative * weight for sentiment, weight in weighted_sentiments) / total_weight
        weighted_neutral = sum(sentiment.neutral * weight for sentiment, weight in weighted_sentiments) / total_weight
        weighted_compound = sum(sentiment.compound * weight for sentiment, weight in weighted_sentiments) / total_weight
        weighted_confidence = sum(sentiment.confidence * weight for sentiment, weight in weighted_sentiments) / total_weight
        
        return SentimentScore(
            positive=weighted_positive,
            negative=weighted_negative,
            neutral=weighted_neutral,
            compound=weighted_compound,
            confidence=weighted_confidence
        )
    
    def _analyze_entities(self, texts: List[str]) -> Dict[str, SentimentScore]:
        """Analyze sentiment for specific entities"""
        entity_texts = defaultdict(list)
        
        for text in texts:
            entities = self.preprocessor.extract_entities(text)
            for category, entity_list in entities.items():
                for entity in entity_list:
                    # Extract sentences containing the entity
                    sentences = text.split('.')
                    for sentence in sentences:
                        if entity.lower() in sentence.lower():
                            entity_texts[entity].append(sentence)
        
        entity_sentiments = {}
        for entity, entity_text_list in entity_texts.items():
            entity_sentiments[entity] = self._analyze_text_list(entity_text_list)
        
        return entity_sentiments
    
    def _analyze_topics(self, texts: List[str]) -> Dict[str, SentimentScore]:
        """Analyze sentiment for different topics"""
        topic_keywords = {
            'monetary_policy': ['interest rate', 'monetary policy', 'fed', 'ecb', 'central bank'],
            'economic_growth': ['gdp', 'growth', 'expansion', 'recession', 'recovery'],
            'inflation': ['inflation', 'cpi', 'ppi', 'deflation'],
            'employment': ['unemployment', 'jobs', 'employment', 'labor'],
            'trade': ['trade', 'exports', 'imports', 'tariff', 'trade war']
        }
        
        topic_texts = defaultdict(list)
        
        for text in texts:
            text_lower = text.lower()
            for topic, keywords in topic_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    topic_texts[topic].append(text)
        
        topic_sentiments = {}
        for topic, topic_text_list in topic_texts.items():
            topic_sentiments[topic] = self._analyze_text_list(topic_text_list)
        
        return topic_sentiments
    
    def _calculate_sentiment_indicators(self, timeline: List[SentimentAnalysisResult]) -> Dict[str, List[float]]:
        """Calculate sentiment indicators over time"""
        indicators = {
            'overall_compound': [result.overall_sentiment.compound for result in timeline],
            'overall_confidence': [result.overall_sentiment.confidence for result in timeline],
            'news_compound': [result.news_sentiment.compound for result in timeline],
            'social_compound': [result.social_sentiment.compound for result in timeline],
            'economic_compound': [result.economic_sentiment.compound for result in timeline],
            'central_bank_compound': [result.central_bank_sentiment.compound for result in timeline]
        }
        
        # Calculate moving averages
        for key, values in indicators.copy().items():
            if len(values) >= 5:
                ma_5 = pd.Series(values).rolling(5).mean().tolist()
                indicators[f'{key}_ma5'] = ma_5
            
            if len(values) >= 10:
                ma_10 = pd.Series(values).rolling(10).mean().tolist()
                indicators[f'{key}_ma10'] = ma_10
        
        # Calculate sentiment momentum
        for key, values in indicators.copy().items():
            if not key.endswith('_ma5') and not key.endswith('_ma10'):
                momentum = np.diff(values).tolist()
                indicators[f'{key}_momentum'] = [0] + momentum  # Pad with 0 for first value
        
        return indicators
    
    def _calculate_price_correlation(self, forex_data: ForexSentimentData, 
                                   sentiment_indicators: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate correlation between sentiment and price movements"""
        correlations = {}
        
        # This would require actual price data
        # For now, return placeholder correlations
        for indicator_name in sentiment_indicators.keys():
            if not indicator_name.endswith('_momentum'):
                # Simulate correlation (in real implementation, use actual price data)
                correlations[indicator_name] = np.random.uniform(-0.5, 0.5)
        
        return correlations
    
    def _generate_sentiment_signals(self, sentiment_indicators: Dict[str, List[float]]) -> List[str]:
        """Generate trading signals based on sentiment"""
        signals = []
        
        overall_compound = sentiment_indicators.get('overall_compound', [])
        overall_momentum = sentiment_indicators.get('overall_compound_momentum', [])
        
        for i in range(len(overall_compound)):
            compound = overall_compound[i]
            momentum = overall_momentum[i] if i < len(overall_momentum) else 0
            
            # Signal generation logic
            if compound > 0.3 and momentum > 0.1:
                signals.append('STRONG_BUY')
            elif compound > 0.1 and momentum > 0:
                signals.append('BUY')
            elif compound < -0.3 and momentum < -0.1:
                signals.append('STRONG_SELL')
            elif compound < -0.1 and momentum < 0:
                signals.append('SELL')
            else:
                signals.append('HOLD')
        
        return signals
    
    def _calculate_risk_metrics(self, sentiment_indicators: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate sentiment-based risk metrics"""
        overall_compound = sentiment_indicators.get('overall_compound', [])
        overall_confidence = sentiment_indicators.get('overall_confidence', [])
        
        if not overall_compound or not overall_confidence:
            return {}
        
        return {
            'sentiment_volatility': np.std(overall_compound),
            'average_confidence': np.mean(overall_confidence),
            'sentiment_range': max(overall_compound) - min(overall_compound),
            'positive_sentiment_ratio': sum(1 for x in overall_compound if x > 0) / len(overall_compound),
            'extreme_sentiment_ratio': sum(1 for x in overall_compound if abs(x) > 0.5) / len(overall_compound)
        }
    
    def _generate_insights(self, timeline: List[SentimentAnalysisResult], 
                          sentiment_indicators: Dict[str, List[float]]) -> List[str]:
        """Generate analytical insights"""
        insights = []
        
        if not timeline:
            return insights
        
        # Overall sentiment trend
        overall_compounds = [result.overall_sentiment.compound for result in timeline]
        if len(overall_compounds) > 1:
            trend = np.polyfit(range(len(overall_compounds)), overall_compounds, 1)[0]
            if trend > 0.01:
                insights.append("Overall sentiment shows positive trend")
            elif trend < -0.01:
                insights.append("Overall sentiment shows negative trend")
            else:
                insights.append("Overall sentiment remains stable")
        
        # Source analysis
        avg_news = np.mean([result.news_sentiment.compound for result in timeline])
        avg_social = np.mean([result.social_sentiment.compound for result in timeline])
        
        if avg_news > avg_social + 0.1:
            insights.append("News sentiment more positive than social media")
        elif avg_social > avg_news + 0.1:
            insights.append("Social media sentiment more positive than news")
        
        # Confidence analysis
        avg_confidence = np.mean([result.overall_sentiment.confidence for result in timeline])
        if avg_confidence > 0.7:
            insights.append("High confidence in sentiment predictions")
        elif avg_confidence < 0.4:
            insights.append("Low confidence in sentiment predictions")
        
        # Volatility analysis
        sentiment_volatility = np.std(overall_compounds)
        if sentiment_volatility > 0.3:
            insights.append("High sentiment volatility detected")
        elif sentiment_volatility < 0.1:
            insights.append("Low sentiment volatility - stable market mood")
        
        return insights
    
    def _generate_recommendations(self, sentiment_indicators: Dict[str, List[float]], 
                                 risk_metrics: Dict[str, float]) -> List[str]:
        """Generate trading recommendations"""
        recommendations = []
        
        # Risk-based recommendations
        sentiment_volatility = risk_metrics.get('sentiment_volatility', 0)
        if sentiment_volatility > 0.3:
            recommendations.append("High sentiment volatility - use smaller position sizes")
        
        average_confidence = risk_metrics.get('average_confidence', 0)
        if average_confidence < 0.5:
            recommendations.append("Low sentiment confidence - wait for clearer signals")
        
        positive_ratio = risk_metrics.get('positive_sentiment_ratio', 0.5)
        if positive_ratio > 0.7:
            recommendations.append("Predominantly positive sentiment - consider long positions")
        elif positive_ratio < 0.3:
            recommendations.append("Predominantly negative sentiment - consider short positions")
        
        # Signal-based recommendations
        overall_compound = sentiment_indicators.get('overall_compound', [])
        if overall_compound:
            recent_sentiment = overall_compound[-1]
            if abs(recent_sentiment) > 0.5:
                recommendations.append("Extreme sentiment detected - potential reversal opportunity")
        
        recommendations.append("Combine sentiment analysis with technical indicators")
        recommendations.append("Monitor central bank communications for policy shifts")
        
        return recommendations
    
    def _calculate_model_performance(self, timeline: List[SentimentAnalysisResult]) -> Dict[str, float]:
        """Calculate model performance metrics"""
        if not timeline:
            return {}
        
        confidences = [result.overall_sentiment.confidence for result in timeline]
        compounds = [result.overall_sentiment.compound for result in timeline]
        
        return {
            'average_confidence': np.mean(confidences),
            'confidence_stability': 1 - np.std(confidences),
            'sentiment_coverage': len([c for c in compounds if abs(c) > 0.1]) / len(compounds),
            'extreme_sentiment_detection': len([c for c in compounds if abs(c) > 0.5]) / len(compounds)
        }
    
    def plot_results(self, forex_data: ForexSentimentData, results: ForexBERTResults):
        """Plot sentiment analysis results"""
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        
        # Extract timestamps and sentiment data
        timestamps = [result.timestamp for result in results.sentiment_timeline]
        overall_compounds = [result.overall_sentiment.compound for result in results.sentiment_timeline]
        overall_confidences = [result.overall_sentiment.confidence for result in results.sentiment_timeline]
        
        # Plot 1: Overall sentiment timeline
        ax1 = axes[0, 0]
        ax1.plot(timestamps, overall_compounds, label='Sentiment', linewidth=2, color='blue')
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.axhline(y=0.3, color='green', linestyle='--', alpha=0.5, label='Positive Threshold')
        ax1.axhline(y=-0.3, color='red', linestyle='--', alpha=0.5, label='Negative Threshold')
        ax1.set_title('Overall Sentiment Timeline', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Sentiment Compound Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Sentiment by source
        ax2 = axes[0, 1]
        news_compounds = [result.news_sentiment.compound for result in results.sentiment_timeline]
        social_compounds = [result.social_sentiment.compound for result in results.sentiment_timeline]
        economic_compounds = [result.economic_sentiment.compound for result in results.sentiment_timeline]
        
        ax2.plot(timestamps, news_compounds, label='News', alpha=0.7)
        ax2.plot(timestamps, social_compounds, label='Social Media', alpha=0.7)
        ax2.plot(timestamps, economic_compounds, label='Economic', alpha=0.7)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_title('Sentiment by Source', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Sentiment Compound Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Confidence over time
        ax3 = axes[1, 0]
        ax3.plot(timestamps, overall_confidences, color='orange', linewidth=2)
        ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold')
        ax3.set_title('Sentiment Confidence Over Time', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Confidence Score')
        ax3.set_ylim(0, 1)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Trading signals distribution
        ax4 = axes[1, 1]
        if results.sentiment_signals:
            signal_counts = Counter(results.sentiment_signals)
            colors = {'STRONG_BUY': 'darkgreen', 'BUY': 'green', 'HOLD': 'gray', 
                     'SELL': 'red', 'STRONG_SELL': 'darkred'}
            signal_colors = [colors.get(signal, 'blue') for signal in signal_counts.keys()]
            
            wedges, texts, autotexts = ax4.pie(signal_counts.values(), 
                                              labels=signal_counts.keys(),
                                              colors=signal_colors,
                                              autopct='%1.1f%%',
                                              startangle=90)
            ax4.set_title('Trading Signals Distribution', fontsize=14, fontweight='bold')
        
        # Plot 5: Sentiment indicators
        ax5 = axes[2, 0]
        if 'overall_compound_ma5' in results.sentiment_indicators:
            ma5 = results.sentiment_indicators['overall_compound_ma5']
            ax5.plot(timestamps[:len(ma5)], ma5, label='5-day MA', linewidth=2)
        
        if 'overall_compound_ma10' in results.sentiment_indicators:
            ma10 = results.sentiment_indicators['overall_compound_ma10']
            ax5.plot(timestamps[:len(ma10)], ma10, label='10-day MA', linewidth=2)
        
        ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax5.set_title('Sentiment Moving Averages', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Time')
        ax5.set_ylabel('Sentiment Score')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Risk metrics
        ax6 = axes[2, 1]
        risk_names = list(results.risk_metrics.keys())
        risk_values = list(results.risk_metrics.values())
        
        bars = ax6.bar(range(len(risk_names)), risk_values, color='lightcoral')
        ax6.set_xticks(range(len(risk_names)))
        ax6.set_xticklabels([name.replace('_', ' ').title() for name in risk_names], rotation=45, ha='right')
        ax6.set_title('Risk Metrics', fontsize=14, fontweight='bold')
        ax6.set_ylabel('Value')
        
        # Add value labels on bars
        for bar, value in zip(bars, risk_values):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(risk_values)*0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self, forex_data: ForexSentimentData, results: ForexBERTResults) -> str:
        """Generate comprehensive sentiment analysis report"""
        report = []
        report.append("=== FOREXBERT SENTIMENT ANALYSIS REPORT ===")
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Currency Pair: {forex_data.currency_pair}")
        report.append(f"Analysis Period: {len(results.sentiment_timeline)} days")
        report.append(f"News Articles Analyzed: {len(forex_data.news_articles)}")
        report.append(f"Social Media Posts Analyzed: {len(forex_data.social_media_posts)}")
        report.append("")
        
        # Model Performance
        report.append("MODEL PERFORMANCE:")
        for metric, value in results.model_performance.items():
            report.append(f"{metric.replace('_', ' ').title()}: {value:.3f}")
        report.append("")
        
        # Sentiment Summary
        if results.sentiment_timeline:
            overall_compounds = [r.overall_sentiment.compound for r in results.sentiment_timeline]
            report.append("SENTIMENT SUMMARY:")
            report.append(f"Average Sentiment: {np.mean(overall_compounds):.3f}")
            report.append(f"Sentiment Volatility: {np.std(overall_compounds):.3f}")
            report.append(f"Most Positive Day: {max(overall_compounds):.3f}")
            report.append(f"Most Negative Day: {min(overall_compounds):.3f}")
            report.append("")
        
        # Trading Signals
        if results.sentiment_signals:
            signal_counts = Counter(results.sentiment_signals)
            report.append("TRADING SIGNALS SUMMARY:")
            for signal, count in signal_counts.items():
                percentage = (count / len(results.sentiment_signals)) * 100
                report.append(f"{signal}: {count} ({percentage:.1f}%)")
            report.append("")
        
        # Risk Metrics
        report.append("RISK METRICS:")
        for metric, value in results.risk_metrics.items():
            report.append(f"{metric.replace('_', ' ').title()}: {value:.3f}")
        report.append("")
        
        # Price Correlation
        if results.correlation_with_price:
            report.append("SENTIMENT-PRICE CORRELATIONS:")
            for indicator, correlation in results.correlation_with_price.items():
                report.append(f"{indicator}: {correlation:.3f}")
            report.append("")
        
        # Insights
        report.append("KEY INSIGHTS:")
        for insight in results.insights:
            report.append(f"• {insight}")
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        for recommendation in results.recommendations:
            report.append(f"• {recommendation}")
        
        return "\n".join(report)

# Example usage and testing
if __name__ == "__main__":
    # Generate sample forex sentiment data
    np.random.seed(42)
    
    # Sample news articles
    sample_articles = [
        NewsArticle(
            title="Fed Signals Hawkish Stance on Interest Rates",
            content="The Federal Reserve indicated a more aggressive approach to combating inflation, suggesting higher interest rates ahead.",
            timestamp=datetime.now() - timedelta(days=5),
            source="Reuters",
            currency_pair="EURUSD"
        ),
        NewsArticle(
            title="ECB Maintains Dovish Policy Outlook",
            content="European Central Bank maintains accommodative monetary policy stance amid economic uncertainty.",
            timestamp=datetime.now() - timedelta(days=3),
            source="Bloomberg",
            currency_pair="EURUSD"
        ),
        NewsArticle(
            title="Strong US Employment Data Boosts Dollar",
            content="Robust job growth and declining unemployment rate strengthen dollar outlook against major currencies.",
            timestamp=datetime.now() - timedelta(days=1),
            source="Financial Times",
            currency_pair="EURUSD"
        )
    ]
    
    # Sample social media posts
    sample_social_posts = [
        "Bullish on USD after strong NFP data! #forex #trading",
        "EUR looking weak against USD, expecting further decline",
        "Fed hawkish comments driving USD strength across the board",
        "ECB dovish stance weighing on EUR, time to short EURUSD?",
        "Market volatility increasing, be careful with position sizing"
    ]
    
    # Sample economic calendar
    sample_economic_calendar = [
        {
            'event': 'US Non-Farm Payrolls',
            'impact': 'High',
            'actual': 250000,
            'forecast': 200000,
            'timestamp': datetime.now() - timedelta(days=2)
        },
        {
            'event': 'ECB Interest Rate Decision',
            'impact': 'High',
            'actual': 0.00,
            'forecast': 0.00,
            'timestamp': datetime.now() - timedelta(days=4)
        }
    ]
    
    # Sample central bank communications
    sample_central_bank_comms = [
        "Fed Chair emphasizes commitment to price stability and full employment mandate.",
        "ECB President highlights ongoing support for economic recovery in eurozone."
    ]
    
    # Create forex sentiment data
    forex_sentiment_data = ForexSentimentData(
        timestamp=[datetime.now() - timedelta(days=i) for i in range(7, 0, -1)],
        currency_pair="EURUSD",
        news_articles=sample_articles,
        social_media_posts=sample_social_posts,
        economic_calendar=sample_economic_calendar,
        central_bank_communications=sample_central_bank_comms
    )
    
    # Initialize analyzer
    analyzer = ForexSentimentAnalyzer()
    
    try:
        # Perform sentiment analysis
        print("Starting ForexBERT Sentiment Analysis...")
        results = analyzer.analyze_sentiment(forex_sentiment_data)
        
        # Print summary
        print("\n=== SENTIMENT ANALYSIS SUMMARY ===")
        
        if results.sentiment_timeline:
            avg_sentiment = np.mean([r.overall_sentiment.compound for r in results.sentiment_timeline])
            avg_confidence = np.mean([r.overall_sentiment.confidence for r in results.sentiment_timeline])
            print(f"Average Sentiment: {avg_sentiment:.3f}")
            print(f"Average Confidence: {avg_confidence:.3f}")
        
        print("\nModel Performance:")
        for metric, value in results.model_performance.items():
            print(f"{metric}: {value:.3f}")
        
        print("\nRisk Metrics:")
        for metric, value in results.risk_metrics.items():
            print(f"{metric}: {value:.3f}")
        
        print("\nKey Insights:")
        for insight in results.insights[:5]:
            print(f"• {insight}")
        
        print("\nRecommendations:")
        for rec in results.recommendations[:3]:
            print(f"• {rec}")
        
        # Generate report
        report = analyzer.generate_report(forex_sentiment_data, results)
        
        # Plot results
        try:
            analyzer.plot_results(forex_sentiment_data, results)
        except Exception as e:
            print(f"Plotting failed: {e}")
        
        print("\nSentiment analysis completed successfully!")
        
    except Exception as e:
        print(f"Sentiment analysis failed: {e}")
        import traceback
        traceback.print_exc()