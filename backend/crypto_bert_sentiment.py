import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    
logger = logging.getLogger(__name__)

@dataclass
class CryptoIndicatorResult:
    """Result container for crypto indicator calculations"""
    indicator_name: str
    value: float
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any]
    signal: str  # 'buy', 'sell', 'hold'
    strength: float  # 0-1 signal strength

class CryptoBERTSentiment:
    """FinBERT/CryptoBERT sentiment analysis"""
    
    def __init__(self):
        self.model_name = "ElKulako/cryptobert"
        self.tokenizer = None
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the CryptoBERT model"""
        try:
            if TRANSFORMERS_AVAILABLE:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            else:
                logger.warning("Transformers library not available, using mock sentiment")
        except Exception as e:
            logger.warning(f"Could not load CryptoBERT model: {e}")
            # Fallback to mock sentiment
            self.tokenizer = None
            self.model = None
    
    def analyze_sentiment(self, 
                         texts: List[str],
                         sources: Optional[List[str]] = None) -> CryptoIndicatorResult:
        """Analyze sentiment from crypto-related texts"""
        try:
            if not texts:
                raise ValueError("No texts provided for sentiment analysis")
                
            if self.model is None or self.tokenizer is None or not TRANSFORMERS_AVAILABLE:
                # Mock sentiment analysis
                sentiment_score = np.random.uniform(-1, 1)  # Random sentiment for demo
                confidence = 0.5
            else:
                # Real sentiment analysis
                sentiments = []
                for text in texts:
                    inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                        # Assuming binary classification: negative (0), positive (1)
                        sentiment = predictions[0][1].item() - predictions[0][0].item()  # Range: -1 to 1
                        sentiments.append(sentiment)
                
                sentiment_score = np.mean(sentiments)
                confidence = 1.0 - np.std(sentiments)  # Lower std = higher confidence
            
            # Generate trading signals based on sentiment
            if sentiment_score > 0.3:  # Positive sentiment
                signal = 'buy'
                strength = min(sentiment_score, 1.0)
            elif sentiment_score < -0.3:  # Negative sentiment
                signal = 'sell'
                strength = min(abs(sentiment_score), 1.0)
            else:
                signal = 'hold'
                strength = 0.5
            
            return CryptoIndicatorResult(
                indicator_name='CryptoBERT Sentiment',
                value=sentiment_score,
                confidence=confidence,
                timestamp=datetime.now(),
                metadata={
                    'text_count': len(texts),
                    'sources': sources or ['unknown'],
                    'model_used': self.model_name if self.model else 'mock'
                },
                signal=signal,
                strength=strength
            )
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return self._error_result('CryptoBERT Sentiment', str(e))
    
    def _error_result(self, indicator_name: str, error_msg: str) -> CryptoIndicatorResult:
        return CryptoIndicatorResult(
            indicator_name=indicator_name,
            value=0.0,
            confidence=0.0,
            timestamp=datetime.now(),
            metadata={'error': error_msg},
            signal='hold',
            strength=0.0
        )