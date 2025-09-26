from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd

# Conditional imports with fallbacks
try:
    from transformers import AutoTokenizer, AutoModel, pipeline
except ImportError:
    AutoTokenizer = None
    AutoModel = None
    pipeline = None

try:
    import torch
except ImportError:
    torch = None

try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    SentimentIntensityAnalyzer = None


@dataclass
class CrossAssetData:
    """Data structure for cross-asset analysis"""
    asset_prices: Dict[str, List[float]]
    asset_returns: Dict[str, List[float]]
    timestamps: List[str]
    correlations: Optional[Dict[str, Dict[str, float]]] = None
    volatilities: Optional[Dict[str, float]] = None


@dataclass
class NLPResults:
    """Results from NLP sentiment analysis"""
    sentiment_scores: List[float]
    sentiment_classification: List[str]
    finbert_embeddings: Optional[np.ndarray]
    cryptobert_embeddings: Optional[np.ndarray]
    forexbert_embeddings: Optional[np.ndarray]
    news_impact_scores: List[float]
    sentiment_momentum: List[float]
    keyword_analysis: Dict[str, float]
    entity_sentiment: Dict[str, float]
    topic_sentiment: Dict[str, float]


class NLPAnalyzer:
    """NLP analysis for financial sentiment and news impact"""
    
    def __init__(self):
        self.finbert_model = None
        self.cryptobert_model = None
        self.forexbert_model = None
        self.sentiment_analyzer = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize NLP models with fallbacks"""
        try:
            if AutoTokenizer and AutoModel:
                # Try to load FinBERT
                self.finbert_tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
                self.finbert_model = AutoModel.from_pretrained('ProsusAI/finbert')
        except Exception:
            print("FinBERT not available, using fallback sentiment analysis")
        
        try:
            if SentimentIntensityAnalyzer:
                self.sentiment_analyzer = SentimentIntensityAnalyzer()
        except Exception:
            print("VADER sentiment analyzer not available")
    
    def analyze_sentiment_mock(self, text: str) -> Dict[str, float]:
        """Mock sentiment analysis when advanced models are unavailable"""
        # Simple keyword-based sentiment
        positive_words = ['good', 'great', 'excellent', 'positive', 'bullish', 'up', 'gain', 'profit', 'strong', 'growth']
        negative_words = ['bad', 'terrible', 'negative', 'bearish', 'down', 'loss', 'weak', 'decline', 'fall', 'crash']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0}
        
        pos_score = positive_count / total_words
        neg_score = negative_count / total_words
        neu_score = max(0, 1 - pos_score - neg_score)
        
        compound = pos_score - neg_score
        
        return {
            'compound': compound,
            'pos': pos_score,
            'neu': neu_score,
            'neg': neg_score
        }
    
    def extract_finbert_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """Extract FinBERT embeddings with fallback"""
        if not texts:
            return None
        
        if self.finbert_model and self.finbert_tokenizer and torch:
            try:
                embeddings = []
                for text in texts[:100]:  # Limit to avoid memory issues
                    inputs = self.finbert_tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
                    with torch.no_grad():
                        outputs = self.finbert_model(**inputs)
                        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                        embeddings.append(embedding)
                return np.array(embeddings)
            except Exception as e:
                print(f"FinBERT embedding extraction failed: {e}")
        
        # Fallback: simple TF-IDF-like embeddings
        return self._create_simple_embeddings(texts)
    
    def extract_cryptobert_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """Extract CryptoBERT embeddings with fallback"""
        # Mock implementation - in practice would use actual CryptoBERT
        return self._create_simple_embeddings(texts, domain='crypto')
    
    def extract_forexbert_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """Extract ForexBERT embeddings with fallback"""
        # Mock implementation - in practice would use actual ForexBERT
        return self._create_simple_embeddings(texts, domain='forex')
    
    def _create_simple_embeddings(self, texts: List[str], domain: str = 'general') -> np.ndarray:
        """Create simple embeddings based on keyword presence"""
        if not texts:
            return np.array([])
        
        # Domain-specific keywords
        if domain == 'crypto':
            keywords = ['bitcoin', 'ethereum', 'blockchain', 'crypto', 'defi', 'nft', 'mining', 'wallet', 'exchange', 'altcoin']
        elif domain == 'forex':
            keywords = ['currency', 'dollar', 'euro', 'yen', 'pound', 'exchange', 'rate', 'central', 'bank', 'inflation']
        else:
            keywords = ['market', 'stock', 'price', 'trade', 'invest', 'profit', 'loss', 'bull', 'bear', 'volume']
        
        embeddings = []
        for text in texts:
            text_lower = text.lower()
            embedding = [1.0 if keyword in text_lower else 0.0 for keyword in keywords]
            # Add some random noise to make embeddings more realistic
            embedding = np.array(embedding) + np.random.normal(0, 0.1, len(embedding))
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def calculate_news_impact(self, news_texts: List[str], prices: List[float]) -> List[float]:
        """Calculate news impact on price movements"""
        if not news_texts or not prices or len(prices) < 2:
            return [0.0] * len(news_texts)
        
        impact_scores = []
        
        for i, text in enumerate(news_texts):
            # Get sentiment for this news item
            sentiment = self.analyze_sentiment_mock(text)
            sentiment_score = sentiment['compound']
            
            # Calculate price change around this news (if we have enough data)
            if i < len(prices) - 1:
                price_change = (prices[i + 1] - prices[i]) / prices[i]
            else:
                price_change = 0.0
            
            # Impact is correlation between sentiment and price change
            # Simplified: just multiply sentiment by absolute price change
            impact = abs(sentiment_score * price_change * 100)  # Scale for visibility
            impact_scores.append(impact)
        
        return impact_scores
    
    def calculate_sentiment_momentum(self, sentiment_scores: List[float], window: int = 5) -> List[float]:
        """Calculate sentiment momentum using moving averages"""
        if not sentiment_scores or len(sentiment_scores) < window:
            return [0.0] * len(sentiment_scores)
        
        momentum = []
        
        for i in range(len(sentiment_scores)):
            if i < window:
                # Not enough data for full window
                momentum.append(0.0)
            else:
                # Calculate momentum as difference between current and past average
                current_avg = np.mean(sentiment_scores[i-window//2:i+window//2+1])
                past_avg = np.mean(sentiment_scores[i-window:i-window//2])
                momentum.append(current_avg - past_avg)
        
        return momentum
    
    def extract_keywords(self, texts: List[str]) -> Dict[str, float]:
        """Extract and score important keywords"""
        if not texts:
            return {}
        
        # Financial keywords to look for
        financial_keywords = [
            'earnings', 'revenue', 'profit', 'loss', 'growth', 'decline',
            'bullish', 'bearish', 'rally', 'crash', 'volatility', 'risk',
            'fed', 'interest', 'rate', 'inflation', 'gdp', 'unemployment',
            'merger', 'acquisition', 'ipo', 'dividend', 'buyback', 'split'
        ]
        
        keyword_scores = {}
        total_texts = len(texts)
        
        for keyword in financial_keywords:
            count = 0
            sentiment_sum = 0.0
            
            for text in texts:
                if keyword.lower() in text.lower():
                    count += 1
                    # Get sentiment of text containing this keyword
                    sentiment = self.analyze_sentiment_mock(text)
                    sentiment_sum += sentiment['compound']
            
            if count > 0:
                # Score is frequency * average sentiment
                frequency = count / total_texts
                avg_sentiment = sentiment_sum / count
                keyword_scores[keyword] = frequency * avg_sentiment
        
        return keyword_scores
    
    def analyze_entity_sentiment(self, texts: List[str]) -> Dict[str, float]:
        """Analyze sentiment for specific entities (companies, currencies, etc.)"""
        if not texts:
            return {}
        
        # Common financial entities
        entities = [
            'apple', 'microsoft', 'google', 'amazon', 'tesla',
            'bitcoin', 'ethereum', 'dollar', 'euro', 'gold',
            'oil', 'nasdaq', 'sp500', 'dow'
        ]
        
        entity_sentiment = {}
        
        for entity in entities:
            relevant_texts = [text for text in texts if entity.lower() in text.lower()]
            
            if relevant_texts:
                sentiments = [self.analyze_sentiment_mock(text)['compound'] for text in relevant_texts]
                entity_sentiment[entity] = np.mean(sentiments)
        
        return entity_sentiment
    
    def analyze_topic_sentiment(self, texts: List[str]) -> Dict[str, float]:
        """Analyze sentiment by financial topics"""
        if not texts:
            return {}
        
        topics = {
            'monetary_policy': ['fed', 'interest', 'rate', 'policy', 'central', 'bank'],
            'earnings': ['earnings', 'revenue', 'profit', 'eps', 'guidance'],
            'market_structure': ['volatility', 'liquidity', 'volume', 'trading'],
            'economic_data': ['gdp', 'inflation', 'unemployment', 'cpi', 'ppi'],
            'geopolitical': ['war', 'trade', 'tariff', 'sanction', 'political']
        }
        
        topic_sentiment = {}
        
        for topic, keywords in topics.items():
            relevant_texts = []
            for text in texts:
                text_lower = text.lower()
                if any(keyword in text_lower for keyword in keywords):
                    relevant_texts.append(text)
            
            if relevant_texts:
                sentiments = [self.analyze_sentiment_mock(text)['compound'] for text in relevant_texts]
                topic_sentiment[topic] = np.mean(sentiments)
        
        return topic_sentiment
    
    def analyze_news_sentiment(self, news_texts: List[str], prices: List[float]) -> NLPResults:
        """Comprehensive NLP analysis of news sentiment"""
        print("Performing NLP sentiment analysis...")
        
        if not news_texts:
            return NLPResults(
                sentiment_scores=[],
                sentiment_classification=[],
                finbert_embeddings=None,
                cryptobert_embeddings=None,
                forexbert_embeddings=None,
                news_impact_scores=[],
                sentiment_momentum=[],
                keyword_analysis={},
                entity_sentiment={},
                topic_sentiment={}
            )
        
        # Analyze sentiment for each text
        sentiment_scores = []
        sentiment_classifications = []
        
        for text in news_texts:
            if self.sentiment_analyzer:
                try:
                    scores = self.sentiment_analyzer.polarity_scores(text)
                    sentiment_scores.append(scores['compound'])
                except Exception:
                    scores = self.analyze_sentiment_mock(text)
                    sentiment_scores.append(scores['compound'])
            else:
                scores = self.analyze_sentiment_mock(text)
                sentiment_scores.append(scores['compound'])
            
            # Classify sentiment
            if scores['compound'] >= 0.05:
                sentiment_classifications.append('positive')
            elif scores['compound'] <= -0.05:
                sentiment_classifications.append('negative')
            else:
                sentiment_classifications.append('neutral')
        
        # Extract embeddings
        finbert_embeddings = self.extract_finbert_embeddings(news_texts)
        cryptobert_embeddings = self.extract_cryptobert_embeddings(news_texts)
        forexbert_embeddings = self.extract_forexbert_embeddings(news_texts)
        
        # Calculate news impact
        news_impact_scores = self.calculate_news_impact(news_texts, prices)
        
        # Calculate sentiment momentum
        sentiment_momentum = self.calculate_sentiment_momentum(sentiment_scores)
        
        # Keyword analysis
        keyword_analysis = self.extract_keywords(news_texts)
        
        # Entity sentiment
        entity_sentiment = self.analyze_entity_sentiment(news_texts)
        
        # Topic sentiment
        topic_sentiment = self.analyze_topic_sentiment(news_texts)
        
        return NLPResults(
            sentiment_scores=sentiment_scores,
            sentiment_classification=sentiment_classifications,
            finbert_embeddings=finbert_embeddings,
            cryptobert_embeddings=cryptobert_embeddings,
            forexbert_embeddings=forexbert_embeddings,
            news_impact_scores=news_impact_scores,
            sentiment_momentum=sentiment_momentum,
            keyword_analysis=keyword_analysis,
            entity_sentiment=entity_sentiment,
            topic_sentiment=topic_sentiment
        )
    
    def get_sentiment_summary(self, results: NLPResults) -> Dict[str, Any]:
        """Generate summary statistics for sentiment analysis"""
        if not results.sentiment_scores:
            return {}
        
        sentiment_scores = results.sentiment_scores
        
        return {
            'average_sentiment': np.mean(sentiment_scores),
            'sentiment_volatility': np.std(sentiment_scores),
            'positive_ratio': len([s for s in results.sentiment_classification if s == 'positive']) / len(results.sentiment_classification),
            'negative_ratio': len([s for s in results.sentiment_classification if s == 'negative']) / len(results.sentiment_classification),
            'neutral_ratio': len([s for s in results.sentiment_classification if s == 'neutral']) / len(results.sentiment_classification),
            'sentiment_trend': 'bullish' if sentiment_scores[-1] > sentiment_scores[0] else 'bearish',
            'max_sentiment': max(sentiment_scores),
            'min_sentiment': min(sentiment_scores),
            'sentiment_range': max(sentiment_scores) - min(sentiment_scores)
        }
    
    def detect_sentiment_anomalies(self, results: NLPResults, threshold: float = 2.0) -> List[Dict[str, Any]]:
        """Detect anomalous sentiment readings"""
        if not results.sentiment_scores or len(results.sentiment_scores) < 10:
            return []
        
        sentiment_scores = np.array(results.sentiment_scores)
        mean_sentiment = np.mean(sentiment_scores)
        std_sentiment = np.std(sentiment_scores)
        
        anomalies = []
        
        for i, score in enumerate(sentiment_scores):
            z_score = abs(score - mean_sentiment) / std_sentiment if std_sentiment > 0 else 0
            
            if z_score > threshold:
                anomalies.append({
                    'index': i,
                    'sentiment_score': score,
                    'z_score': z_score,
                    'type': 'extremely_positive' if score > mean_sentiment else 'extremely_negative',
                    'impact_score': results.news_impact_scores[i] if i < len(results.news_impact_scores) else 0.0
                })
        
        return sorted(anomalies, key=lambda x: x['z_score'], reverse=True)


# Example usage
if __name__ == "__main__":
    # Sample data
    sample_news = [
        "Apple reports strong quarterly earnings with record iPhone sales",
        "Federal Reserve hints at potential interest rate cuts amid economic uncertainty",
        "Tech stocks rally as investors show renewed confidence in growth prospects",
        "Oil prices surge following geopolitical tensions in the Middle East",
        "Bitcoin reaches new all-time high as institutional adoption accelerates",
        "Market volatility increases as traders react to mixed economic signals",
        "Amazon announces major expansion plans, stock price jumps 5%",
        "Inflation concerns weigh on consumer sentiment and spending patterns",
        "Gold prices decline as dollar strengthens against major currencies",
        "Cryptocurrency market shows signs of consolidation after recent gains"
    ]
    
    sample_prices = [100.0, 102.5, 101.8, 105.2, 108.7, 106.3, 109.1, 107.8, 111.2, 113.5]
    
    # Initialize analyzer
    nlp_analyzer = NLPAnalyzer()
    
    # Perform analysis
    results = nlp_analyzer.analyze_news_sentiment(sample_news, sample_prices)
    
    print("NLP Analysis Results:")
    print(f"Analyzed {len(sample_news)} news items")
    print(f"Average Sentiment: {np.mean(results.sentiment_scores):.3f}")
    print(f"Sentiment Classifications: {dict(zip(*np.unique(results.sentiment_classification, return_counts=True)))}")
    
    # Get summary
    summary = nlp_analyzer.get_sentiment_summary(results)
    print(f"\nSentiment Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Detect anomalies
    anomalies = nlp_analyzer.detect_sentiment_anomalies(results)
    if anomalies:
        print(f"\nDetected {len(anomalies)} sentiment anomalies:")
        for anomaly in anomalies[:3]:  # Show top 3
            print(f"  Index {anomaly['index']}: {anomaly['type']} (z-score: {anomaly['z_score']:.2f})")
    
    print(f"\nKeyword Analysis:")
    for keyword, score in list(results.keyword_analysis.items())[:5]:
        print(f"  {keyword}: {score:.3f}")
    
    print(f"\nEntity Sentiment:")
    for entity, sentiment in list(results.entity_sentiment.items())[:5]:
        print(f"  {entity}: {sentiment:.3f}")
    
    print(f"\nTopic Sentiment:")
    for topic, sentiment in results.topic_sentiment.items():
        print(f"  {topic}: {sentiment:.3f}")