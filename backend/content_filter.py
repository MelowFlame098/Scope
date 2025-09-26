import asyncio
import re
import json
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import logging
import aioredis
from concurrent.futures import ThreadPoolExecutor
import hashlib
from urllib.parse import urlparse
import spacy
from transformers import pipeline
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentQuality(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    SPAM = "spam"

class ContentCategory(str, Enum):
    FINANCIAL_NEWS = "financial_news"
    MARKET_ANALYSIS = "market_analysis"
    COMPANY_NEWS = "company_news"
    CRYPTO_NEWS = "crypto_news"
    ECONOMIC_DATA = "economic_data"
    OPINION = "opinion"
    ADVERTISEMENT = "advertisement"
    SPAM = "spam"
    OTHER = "other"

class FilterAction(str, Enum):
    APPROVE = "approve"
    REVIEW = "review"
    REJECT = "reject"
    FLAG = "flag"

@dataclass
class ContentScore:
    quality_score: float  # 0-1
    relevance_score: float  # 0-1
    credibility_score: float  # 0-1
    readability_score: float  # 0-1
    engagement_score: float  # 0-1
    spam_score: float  # 0-1 (higher = more likely spam)
    overall_score: float  # 0-1

@dataclass
class FilterResult:
    action: FilterAction
    category: ContentCategory
    quality: ContentQuality
    scores: ContentScore
    flags: List[str]
    reasons: List[str]
    confidence: float
    processing_time: float

@dataclass
class ContentMetrics:
    word_count: int
    sentence_count: int
    paragraph_count: int
    reading_time: int  # minutes
    complexity_score: float
    keyword_density: Dict[str, float]
    entity_count: int
    link_count: int
    image_count: int

class ContentFilterService:
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url
        self.redis_client = None
        self.nlp = None
        self.spam_classifier = None
        self.quality_classifier = None
        
        # Thread pool for CPU-intensive tasks
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Cache settings
        self.cache_ttl = 3600  # 1 hour
        
        # Spam detection patterns
        self.spam_patterns = [
            r'\b(?:click here|buy now|limited time|act now|urgent|free money)\b',
            r'\b(?:guaranteed|100% profit|risk-free|instant)\b',
            r'\b(?:pump|dump|moon|lambo|diamond hands)\b',  # Crypto spam
            r'(?:https?://)?(?:bit\.ly|tinyurl|t\.co)/\w+',  # Shortened URLs
            r'[A-Z]{3,}\s+[A-Z]{3,}',  # Excessive caps
            r'[!]{3,}|[?]{3,}',  # Excessive punctuation
        ]
        
        # Quality indicators
        self.quality_indicators = {
            'high': [
                'analysis', 'research', 'data', 'study', 'report', 'findings',
                'according to', 'statistics', 'evidence', 'methodology'
            ],
            'medium': [
                'news', 'update', 'announcement', 'statement', 'release',
                'information', 'details', 'sources'
            ],
            'low': [
                'rumor', 'speculation', 'unconfirmed', 'allegedly', 'reportedly',
                'might', 'could', 'possibly', 'perhaps'
            ]
        }
        
        # Financial relevance keywords
        self.financial_keywords = {
            'stocks': [
                'stock', 'share', 'equity', 'dividend', 'earnings', 'revenue',
                'profit', 'loss', 'ipo', 'merger', 'acquisition', 'nasdaq', 'nyse'
            ],
            'crypto': [
                'bitcoin', 'ethereum', 'cryptocurrency', 'blockchain', 'defi',
                'nft', 'mining', 'wallet', 'exchange', 'altcoin'
            ],
            'forex': [
                'forex', 'currency', 'exchange rate', 'dollar', 'euro', 'yen',
                'pound', 'central bank', 'monetary policy'
            ],
            'economics': [
                'gdp', 'inflation', 'unemployment', 'interest rate', 'fed',
                'economy', 'recession', 'growth', 'trade', 'tariff'
            ],
            'markets': [
                'market', 'trading', 'investor', 'portfolio', 'bull', 'bear',
                'volatility', 'index', 'futures', 'options'
            ]
        }
        
        # Credible sources
        self.credible_sources = {
            'tier1': [  # Highest credibility
                'reuters.com', 'bloomberg.com', 'wsj.com', 'ft.com',
                'cnbc.com', 'marketwatch.com', 'yahoo.com/finance'
            ],
            'tier2': [  # High credibility
                'cnn.com/business', 'bbc.com/business', 'forbes.com',
                'economist.com', 'investopedia.com'
            ],
            'tier3': [  # Medium credibility
                'coindesk.com', 'cointelegraph.com', 'techcrunch.com',
                'seekingalpha.com', 'fool.com'
            ]
        }
        
        # Blacklisted domains
        self.blacklisted_domains = {
            'spam-news.com', 'fake-crypto.net', 'pump-dump.org',
            'get-rich-quick.com', 'scam-alert.info'
        }
        
        # Content length thresholds
        self.length_thresholds = {
            'min_words': 50,
            'max_words': 10000,
            'min_sentences': 3,
            'optimal_words': (200, 2000)
        }
    
    async def initialize(self):
        """Initialize the content filter service."""
        try:
            # Initialize Redis connection
            if self.redis_url:
                self.redis_client = await aioredis.from_url(self.redis_url)
            
            # Load spaCy model
            logger.info("Loading spaCy model for content filtering...")
            self.nlp = spacy.load("en_core_web_sm")
            
            # Load spam classifier
            logger.info("Loading spam classification model...")
            self.spam_classifier = pipeline(
                "text-classification",
                model="unitary/toxic-bert",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Load quality classifier (using a general text classification model)
            logger.info("Loading quality classification model...")
            self.quality_classifier = pipeline(
                "text-classification",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("Content filter service initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing content filter service: {e}")
            raise
    
    async def filter_content(
        self,
        title: str,
        content: str,
        url: str = "",
        source: str = "",
        author: str = ""
    ) -> FilterResult:
        """Filter and analyze content quality."""
        start_time = datetime.now()
        
        try:
            # Check cache first
            cache_key = f"content_filter:{hashlib.md5(f'{title}{content}'.encode()).hexdigest()}"
            if self.redis_client:
                cached_result = await self.redis_client.get(cache_key)
                if cached_result:
                    result_dict = json.loads(cached_result)
                    return FilterResult(**result_dict)
            
            # Combine title and content for analysis
            full_text = f"{title}. {content}"
            
            # Run analysis tasks concurrently
            tasks = [
                self._calculate_content_scores(full_text, url, source),
                self._detect_spam(full_text),
                self._categorize_content(full_text),
                self._assess_quality(full_text),
                self._extract_flags(full_text, url, source, author)
            ]
            
            scores, spam_result, category, quality, flags = await asyncio.gather(*tasks)
            
            # Determine action based on scores and flags
            action, reasons, confidence = self._determine_action(scores, flags, spam_result)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = FilterResult(
                action=action,
                category=category,
                quality=quality,
                scores=scores,
                flags=flags,
                reasons=reasons,
                confidence=confidence,
                processing_time=processing_time
            )
            
            # Cache the result
            if self.redis_client:
                result_dict = {
                    'action': result.action.value,
                    'category': result.category.value,
                    'quality': result.quality.value,
                    'scores': {
                        'quality_score': result.scores.quality_score,
                        'relevance_score': result.scores.relevance_score,
                        'credibility_score': result.scores.credibility_score,
                        'readability_score': result.scores.readability_score,
                        'engagement_score': result.scores.engagement_score,
                        'spam_score': result.scores.spam_score,
                        'overall_score': result.scores.overall_score
                    },
                    'flags': result.flags,
                    'reasons': result.reasons,
                    'confidence': result.confidence,
                    'processing_time': result.processing_time
                }
                
                await self.redis_client.setex(
                    cache_key,
                    self.cache_ttl,
                    json.dumps(result_dict)
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error filtering content: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            return self._create_fallback_result(processing_time)
    
    async def _calculate_content_scores(
        self,
        text: str,
        url: str = "",
        source: str = ""
    ) -> ContentScore:
        """Calculate comprehensive content scores."""
        try:
            # Run scoring tasks concurrently
            tasks = [
                self._calculate_quality_score(text),
                self._calculate_relevance_score(text),
                self._calculate_credibility_score(url, source),
                self._calculate_readability_score(text),
                self._calculate_engagement_score(text),
                self._calculate_spam_score(text)
            ]
            
            scores = await asyncio.gather(*tasks)
            
            quality_score = scores[0]
            relevance_score = scores[1]
            credibility_score = scores[2]
            readability_score = scores[3]
            engagement_score = scores[4]
            spam_score = scores[5]
            
            # Calculate overall score (weighted average)
            overall_score = (
                quality_score * 0.25 +
                relevance_score * 0.25 +
                credibility_score * 0.20 +
                readability_score * 0.15 +
                engagement_score * 0.10 +
                (1 - spam_score) * 0.05  # Invert spam score
            )
            
            return ContentScore(
                quality_score=quality_score,
                relevance_score=relevance_score,
                credibility_score=credibility_score,
                readability_score=readability_score,
                engagement_score=engagement_score,
                spam_score=spam_score,
                overall_score=overall_score
            )
            
        except Exception as e:
            logger.error(f"Error calculating content scores: {e}")
            return ContentScore(
                quality_score=0.5,
                relevance_score=0.5,
                credibility_score=0.5,
                readability_score=0.5,
                engagement_score=0.5,
                spam_score=0.5,
                overall_score=0.5
            )
    
    async def _calculate_quality_score(self, text: str) -> float:
        """Calculate content quality score."""
        try:
            score = 0.0
            text_lower = text.lower()
            
            # Check for quality indicators
            for quality_level, indicators in self.quality_indicators.items():
                for indicator in indicators:
                    if indicator in text_lower:
                        if quality_level == 'high':
                            score += 0.1
                        elif quality_level == 'medium':
                            score += 0.05
                        else:  # low
                            score -= 0.05
            
            # Check content length
            word_count = len(text.split())
            if self.length_thresholds['optimal_words'][0] <= word_count <= self.length_thresholds['optimal_words'][1]:
                score += 0.2
            elif word_count < self.length_thresholds['min_words']:
                score -= 0.3
            
            # Check for proper structure (sentences, paragraphs)
            sentence_count = len(re.split(r'[.!?]+', text))
            if sentence_count >= self.length_thresholds['min_sentences']:
                score += 0.1
            
            # Check for numbers and data (often indicates quality)
            if re.search(r'\d+(?:\.\d+)?%', text):  # Percentages
                score += 0.1
            if re.search(r'\$\d+(?:,\d{3})*(?:\.\d{2})?', text):  # Money amounts
                score += 0.1
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 0.5
    
    async def _calculate_relevance_score(self, text: str) -> float:
        """Calculate financial relevance score."""
        try:
            score = 0.0
            text_lower = text.lower()
            word_count = len(text.split())
            
            # Count financial keywords
            total_keywords = 0
            for category, keywords in self.financial_keywords.items():
                category_count = 0
                for keyword in keywords:
                    if keyword in text_lower:
                        category_count += 1
                        total_keywords += 1
                
                # Bonus for having keywords from multiple categories
                if category_count > 0:
                    score += 0.1
            
            # Calculate keyword density
            if word_count > 0:
                keyword_density = total_keywords / word_count
                score += min(keyword_density * 10, 0.5)  # Cap at 0.5
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating relevance score: {e}")
            return 0.5
    
    async def _calculate_credibility_score(self, url: str, source: str) -> float:
        """Calculate source credibility score."""
        try:
            score = 0.5  # Default score
            
            if url:
                domain = urlparse(url).netloc.lower()
                
                # Check against credible sources
                for tier, domains in self.credible_sources.items():
                    if any(domain.endswith(d) for d in domains):
                        if tier == 'tier1':
                            score = 0.95
                        elif tier == 'tier2':
                            score = 0.85
                        elif tier == 'tier3':
                            score = 0.75
                        break
                
                # Check against blacklisted domains
                if any(domain.endswith(d) for d in self.blacklisted_domains):
                    score = 0.1
            
            # Adjust based on source name if provided
            if source:
                source_lower = source.lower()
                if any(credible in source_lower for tier_domains in self.credible_sources.values() for credible in tier_domains):
                    score = max(score, 0.8)
            
            return score
            
        except Exception as e:
            logger.error(f"Error calculating credibility score: {e}")
            return 0.5
    
    async def _calculate_readability_score(self, text: str) -> float:
        """Calculate readability score."""
        try:
            # Simple readability metrics
            words = text.split()
            sentences = re.split(r'[.!?]+', text)
            
            if not words or not sentences:
                return 0.0
            
            # Average words per sentence
            avg_words_per_sentence = len(words) / len(sentences)
            
            # Average syllables per word (approximation)
            total_syllables = sum(self._count_syllables(word) for word in words)
            avg_syllables_per_word = total_syllables / len(words)
            
            # Flesch Reading Ease approximation
            flesch_score = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)
            
            # Normalize to 0-1 scale
            normalized_score = max(0, min(100, flesch_score)) / 100
            
            return normalized_score
            
        except Exception as e:
            logger.error(f"Error calculating readability score: {e}")
            return 0.5
    
    def _count_syllables(self, word: str) -> int:
        """Approximate syllable count for a word."""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        # Handle silent 'e'
        if word.endswith('e'):
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    async def _calculate_engagement_score(self, text: str) -> float:
        """Calculate potential engagement score."""
        try:
            score = 0.0
            text_lower = text.lower()
            
            # Engagement indicators
            engagement_words = [
                'breaking', 'urgent', 'exclusive', 'revealed', 'shocking',
                'surprising', 'important', 'major', 'significant', 'huge'
            ]
            
            for word in engagement_words:
                if word in text_lower:
                    score += 0.1
            
            # Question marks (indicate engagement)
            question_count = text.count('?')
            score += min(question_count * 0.05, 0.2)
            
            # Numbers and statistics
            if re.search(r'\d+', text):
                score += 0.1
            
            # Quotes (indicate human interest)
            quote_count = text.count('"') + text.count("'")
            score += min(quote_count * 0.02, 0.1)
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating engagement score: {e}")
            return 0.5
    
    async def _calculate_spam_score(self, text: str) -> float:
        """Calculate spam probability score."""
        try:
            score = 0.0
            
            # Check spam patterns
            for pattern in self.spam_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    score += 0.2
            
            # Check for excessive capitalization
            caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
            if caps_ratio > 0.3:
                score += 0.3
            
            # Check for excessive punctuation
            punct_ratio = sum(1 for c in text if c in '!?.,;:') / len(text) if text else 0
            if punct_ratio > 0.1:
                score += 0.2
            
            # Check for repetitive content
            words = text.lower().split()
            if len(words) > 0:
                unique_words = set(words)
                repetition_ratio = 1 - (len(unique_words) / len(words))
                if repetition_ratio > 0.5:
                    score += 0.3
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating spam score: {e}")
            return 0.5
    
    async def _detect_spam(self, text: str) -> Dict[str, Any]:
        """Detect spam using ML model."""
        try:
            # Use the toxic classifier as a proxy for spam detection
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                lambda: self.spam_classifier(text[:512])
            )
            
            # Extract spam probability
            spam_prob = 0.0
            for item in result:
                if item['label'] in ['TOXIC', 'SEVERE_TOXIC', 'OBSCENE', 'THREAT', 'INSULT']:
                    spam_prob = max(spam_prob, item['score'])
            
            return {
                'is_spam': spam_prob > 0.7,
                'spam_probability': spam_prob,
                'details': result
            }
            
        except Exception as e:
            logger.error(f"Error detecting spam: {e}")
            return {'is_spam': False, 'spam_probability': 0.0, 'details': []}
    
    async def _categorize_content(self, text: str) -> ContentCategory:
        """Categorize content based on keywords and patterns."""
        try:
            text_lower = text.lower()
            category_scores = {}
            
            # Score each category based on keyword presence
            for category, keywords in self.financial_keywords.items():
                score = sum(1 for keyword in keywords if keyword in text_lower)
                if score > 0:
                    category_scores[category] = score
            
            # Map to content categories
            category_mapping = {
                'stocks': ContentCategory.COMPANY_NEWS,
                'crypto': ContentCategory.CRYPTO_NEWS,
                'forex': ContentCategory.FINANCIAL_NEWS,
                'economics': ContentCategory.ECONOMIC_DATA,
                'markets': ContentCategory.MARKET_ANALYSIS
            }
            
            if category_scores:
                top_category = max(category_scores, key=category_scores.get)
                return category_mapping.get(top_category, ContentCategory.FINANCIAL_NEWS)
            
            # Check for opinion indicators
            opinion_words = ['opinion', 'think', 'believe', 'feel', 'perspective', 'view']
            if any(word in text_lower for word in opinion_words):
                return ContentCategory.OPINION
            
            # Check for advertisement indicators
            ad_words = ['buy', 'sell', 'offer', 'deal', 'discount', 'promotion']
            if any(word in text_lower for word in ad_words):
                return ContentCategory.ADVERTISEMENT
            
            return ContentCategory.OTHER
            
        except Exception as e:
            logger.error(f"Error categorizing content: {e}")
            return ContentCategory.OTHER
    
    async def _assess_quality(self, text: str) -> ContentQuality:
        """Assess overall content quality."""
        try:
            # Calculate quality metrics
            word_count = len(text.split())
            sentence_count = len(re.split(r'[.!?]+', text))
            
            # Base quality assessment
            if word_count < self.length_thresholds['min_words']:
                return ContentQuality.LOW
            
            if sentence_count < self.length_thresholds['min_sentences']:
                return ContentQuality.LOW
            
            # Check for quality indicators
            quality_score = 0
            text_lower = text.lower()
            
            for level, indicators in self.quality_indicators.items():
                for indicator in indicators:
                    if indicator in text_lower:
                        if level == 'high':
                            quality_score += 2
                        elif level == 'medium':
                            quality_score += 1
                        else:  # low
                            quality_score -= 1
            
            # Check for spam patterns
            spam_count = sum(1 for pattern in self.spam_patterns 
                           if re.search(pattern, text, re.IGNORECASE))
            
            if spam_count > 2:
                return ContentQuality.SPAM
            
            # Determine quality level
            if quality_score >= 5:
                return ContentQuality.HIGH
            elif quality_score >= 2:
                return ContentQuality.MEDIUM
            elif spam_count > 0:
                return ContentQuality.SPAM
            else:
                return ContentQuality.LOW
            
        except Exception as e:
            logger.error(f"Error assessing quality: {e}")
            return ContentQuality.MEDIUM
    
    async def _extract_flags(self, text: str, url: str, source: str, author: str) -> List[str]:
        """Extract content flags for review."""
        flags = []
        text_lower = text.lower()
        
        try:
            # Spam flags
            for pattern in self.spam_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    flags.append('potential_spam')
                    break
            
            # Quality flags
            word_count = len(text.split())
            if word_count < self.length_thresholds['min_words']:
                flags.append('too_short')
            elif word_count > self.length_thresholds['max_words']:
                flags.append('too_long')
            
            # Credibility flags
            if url:
                domain = urlparse(url).netloc.lower()
                if any(domain.endswith(d) for d in self.blacklisted_domains):
                    flags.append('blacklisted_source')
            
            # Content flags
            if 'unconfirmed' in text_lower or 'rumor' in text_lower:
                flags.append('unverified_information')
            
            if text.count('!') > 10:
                flags.append('excessive_punctuation')
            
            caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
            if caps_ratio > 0.3:
                flags.append('excessive_caps')
            
            # Financial flags
            risky_words = ['guaranteed', 'risk-free', 'sure thing', 'can\'t lose']
            if any(word in text_lower for word in risky_words):
                flags.append('misleading_claims')
            
            return flags
            
        except Exception as e:
            logger.error(f"Error extracting flags: {e}")
            return flags
    
    def _determine_action(
        self,
        scores: ContentScore,
        flags: List[str],
        spam_result: Dict[str, Any]
    ) -> Tuple[FilterAction, List[str], float]:
        """Determine the appropriate action for the content."""
        reasons = []
        confidence = 0.8
        
        # Check for immediate rejection criteria
        if spam_result.get('is_spam', False) or 'potential_spam' in flags:
            return FilterAction.REJECT, ['Content identified as spam'], 0.9
        
        if 'blacklisted_source' in flags:
            return FilterAction.REJECT, ['Source is blacklisted'], 0.95
        
        if scores.overall_score < 0.3:
            return FilterAction.REJECT, ['Overall quality score too low'], 0.8
        
        # Check for review criteria
        review_flags = ['unverified_information', 'misleading_claims', 'excessive_caps']
        if any(flag in flags for flag in review_flags):
            reasons.append('Content flagged for manual review')
            return FilterAction.REVIEW, reasons, 0.7
        
        if scores.credibility_score < 0.5:
            reasons.append('Low credibility score')
            return FilterAction.REVIEW, reasons, 0.6
        
        # Check for flagging criteria
        if scores.relevance_score < 0.4:
            reasons.append('Low relevance to financial topics')
            return FilterAction.FLAG, reasons, 0.7
        
        if 'too_short' in flags or 'too_long' in flags:
            reasons.append('Content length outside optimal range')
            return FilterAction.FLAG, reasons, 0.6
        
        # Approve high-quality content
        if scores.overall_score >= 0.7:
            reasons.append('High-quality content approved')
            return FilterAction.APPROVE, reasons, 0.9
        
        # Default to approval for moderate quality
        reasons.append('Content meets minimum quality standards')
        return FilterAction.APPROVE, reasons, 0.7
    
    def _create_fallback_result(self, processing_time: float) -> FilterResult:
        """Create a fallback result when processing fails."""
        return FilterResult(
            action=FilterAction.REVIEW,
            category=ContentCategory.OTHER,
            quality=ContentQuality.MEDIUM,
            scores=ContentScore(
                quality_score=0.5,
                relevance_score=0.5,
                credibility_score=0.5,
                readability_score=0.5,
                engagement_score=0.5,
                spam_score=0.5,
                overall_score=0.5
            ),
            flags=['processing_error'],
            reasons=['Error occurred during content analysis'],
            confidence=0.3,
            processing_time=processing_time
        )
    
    async def get_content_metrics(self, text: str) -> ContentMetrics:
        """Get detailed content metrics."""
        try:
            # Basic metrics
            words = text.split()
            sentences = re.split(r'[.!?]+', text)
            paragraphs = text.split('\n\n')
            
            word_count = len(words)
            sentence_count = len([s for s in sentences if s.strip()])
            paragraph_count = len([p for p in paragraphs if p.strip()])
            reading_time = max(1, word_count // 200)  # ~200 WPM
            
            # Complexity score (based on average sentence length and syllables)
            avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
            avg_syllables = sum(self._count_syllables(word) for word in words) / word_count if word_count > 0 else 0
            complexity_score = (avg_sentence_length * 0.5) + (avg_syllables * 0.5)
            
            # Keyword density
            keyword_density = {}
            text_lower = text.lower()
            for category, keywords in self.financial_keywords.items():
                density = sum(1 for keyword in keywords if keyword in text_lower) / word_count if word_count > 0 else 0
                keyword_density[category] = density
            
            # Entity count (using spaCy if available)
            entity_count = 0
            if self.nlp:
                doc = self.nlp(text[:1000])  # Limit for performance
                entity_count = len(doc.ents)
            
            # Link and image count
            link_count = len(re.findall(r'https?://\S+', text))
            image_count = text.lower().count('<img') + text.lower().count('[image')
            
            return ContentMetrics(
                word_count=word_count,
                sentence_count=sentence_count,
                paragraph_count=paragraph_count,
                reading_time=reading_time,
                complexity_score=complexity_score,
                keyword_density=keyword_density,
                entity_count=entity_count,
                link_count=link_count,
                image_count=image_count
            )
            
        except Exception as e:
            logger.error(f"Error calculating content metrics: {e}")
            return ContentMetrics(
                word_count=0,
                sentence_count=0,
                paragraph_count=0,
                reading_time=0,
                complexity_score=0.0,
                keyword_density={},
                entity_count=0,
                link_count=0,
                image_count=0
            )
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.redis_client:
            await self.redis_client.close()
        
        if self.executor:
            self.executor.shutdown(wait=True)

# Global instance
content_filter_service = None

async def get_content_filter_service() -> ContentFilterService:
    """Get or create content filter service instance."""
    global content_filter_service
    if content_filter_service is None:
        content_filter_service = ContentFilterService()
        await content_filter_service.initialize()
    return content_filter_service