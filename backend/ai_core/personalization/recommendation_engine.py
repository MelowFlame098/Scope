# Recommendation Engine
# Phase 9: AI-First Platform Implementation

import logging
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class RecommendationType(Enum):
    INVESTMENT = "investment"
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization"
    TRADING_STRATEGY = "trading_strategy"
    EDUCATIONAL_CONTENT = "educational_content"
    MARKET_ANALYSIS = "market_analysis"
    RISK_MANAGEMENT = "risk_management"
    FEATURE_SUGGESTION = "feature_suggestion"
    SOCIAL_CONNECTION = "social_connection"
    NEWS_ARTICLE = "news_article"
    RESEARCH_REPORT = "research_report"

class RecommendationPriority(Enum):
    CRITICAL = "critical"      # Immediate attention required
    HIGH = "high"              # Important but not urgent
    MEDIUM = "medium"          # Moderate importance
    LOW = "low"                # Nice to have
    BACKGROUND = "background"  # Passive suggestions

class RecommendationSource(Enum):
    COLLABORATIVE_FILTERING = "collaborative_filtering"
    CONTENT_BASED = "content_based"
    HYBRID = "hybrid"
    RULE_BASED = "rule_based"
    MARKET_DRIVEN = "market_driven"
    BEHAVIORAL = "behavioral"
    SOCIAL = "social"
    EXPERT_SYSTEM = "expert_system"

class RecommendationContext(Enum):
    DASHBOARD = "dashboard"
    PORTFOLIO_VIEW = "portfolio_view"
    TRADING_INTERFACE = "trading_interface"
    RESEARCH_CENTER = "research_center"
    EDUCATION_HUB = "education_hub"
    MARKET_OVERVIEW = "market_overview"
    NOTIFICATION = "notification"
    EMAIL_DIGEST = "email_digest"

@dataclass
class RecommendationItem:
    item_id: str
    item_type: str  # asset, strategy, content, feature, etc.
    title: str
    description: str
    category: str
    tags: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    popularity_score: float = 0.0
    quality_score: float = 0.0
    relevance_features: Dict[str, float] = field(default_factory=dict)

@dataclass
class Recommendation:
    recommendation_id: str
    user_id: str
    item: RecommendationItem
    recommendation_type: RecommendationType
    priority: RecommendationPriority
    source: RecommendationSource
    context: RecommendationContext
    score: float  # 0-1 confidence score
    reasoning: List[str]  # Explanation for recommendation
    expected_impact: str  # Expected benefit
    risk_level: str  # Associated risk
    time_sensitivity: str  # How urgent is this
    personalization_factors: Dict[str, float]
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RecommendationFeedback:
    recommendation_id: str
    user_id: str
    feedback_type: str  # clicked, dismissed, saved, acted_upon, rated
    feedback_value: Optional[float] = None  # Rating or engagement score
    timestamp: datetime = field(default_factory=datetime.now)
    context: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RecommendationSet:
    user_id: str
    context: RecommendationContext
    recommendations: List[Recommendation]
    total_score: float
    diversity_score: float
    novelty_score: float
    generated_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(hours=24))
    metadata: Dict[str, Any] = field(default_factory=dict)

class RecommendationEngine:
    """Advanced personalized recommendation engine"""
    
    def __init__(self):
        # Recommendation data
        self.recommendation_items = {}  # item_id -> RecommendationItem
        self.user_interactions = defaultdict(list)  # user_id -> [interactions]
        self.recommendation_history = defaultdict(list)  # user_id -> [recommendations]
        self.feedback_history = defaultdict(list)  # user_id -> [feedback]
        
        # ML models
        self.collaborative_model = None
        self.content_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.content_features = None
        self.nmf_model = NMF(n_components=50, random_state=42)
        self.knn_model = NearestNeighbors(n_neighbors=10, metric='cosine')
        
        # Recommendation strategies
        self.strategy_weights = {
            RecommendationSource.COLLABORATIVE_FILTERING: 0.3,
            RecommendationSource.CONTENT_BASED: 0.25,
            RecommendationSource.BEHAVIORAL: 0.2,
            RecommendationSource.RULE_BASED: 0.15,
            RecommendationSource.MARKET_DRIVEN: 0.1
        }
        
        # Context-specific configurations
        self.context_configs = {
            RecommendationContext.DASHBOARD: {
                'max_recommendations': 5,
                'diversity_weight': 0.3,
                'novelty_weight': 0.2,
                'preferred_types': [RecommendationType.INVESTMENT, RecommendationType.MARKET_ANALYSIS]
            },
            RecommendationContext.PORTFOLIO_VIEW: {
                'max_recommendations': 8,
                'diversity_weight': 0.4,
                'novelty_weight': 0.1,
                'preferred_types': [RecommendationType.PORTFOLIO_OPTIMIZATION, RecommendationType.RISK_MANAGEMENT]
            },
            RecommendationContext.TRADING_INTERFACE: {
                'max_recommendations': 6,
                'diversity_weight': 0.2,
                'novelty_weight': 0.3,
                'preferred_types': [RecommendationType.TRADING_STRATEGY, RecommendationType.MARKET_ANALYSIS]
            }
        }
        
        # Performance tracking
        self.recommendation_metrics = defaultdict(dict)
        
        logger.info("Recommendation engine initialized")
    
    async def add_recommendation_item(self, item: RecommendationItem):
        """Add a new item to the recommendation catalog"""
        try:
            self.recommendation_items[item.item_id] = item
            
            # Update content features if needed
            await self._update_content_features()
            
            logger.info(f"Added recommendation item: {item.item_id}")
            
        except Exception as e:
            logger.error(f"Error adding recommendation item: {e}")
    
    async def generate_recommendations(self, user_id: str, 
                                     context: RecommendationContext,
                                     user_profile: Optional[Dict[str, Any]] = None,
                                     max_recommendations: Optional[int] = None) -> RecommendationSet:
        """Generate personalized recommendations for a user"""
        try:
            # Get context configuration
            config = self.context_configs.get(context, {
                'max_recommendations': 5,
                'diversity_weight': 0.3,
                'novelty_weight': 0.2,
                'preferred_types': []
            })
            
            if max_recommendations is None:
                max_recommendations = config['max_recommendations']
            
            # Generate recommendations from different sources
            collaborative_recs = await self._generate_collaborative_recommendations(user_id, user_profile)
            content_recs = await self._generate_content_based_recommendations(user_id, user_profile)
            behavioral_recs = await self._generate_behavioral_recommendations(user_id, user_profile)
            rule_recs = await self._generate_rule_based_recommendations(user_id, user_profile, context)
            market_recs = await self._generate_market_driven_recommendations(user_id, user_profile)
            
            # Combine and rank recommendations
            all_recommendations = {
                RecommendationSource.COLLABORATIVE_FILTERING: collaborative_recs,
                RecommendationSource.CONTENT_BASED: content_recs,
                RecommendationSource.BEHAVIORAL: behavioral_recs,
                RecommendationSource.RULE_BASED: rule_recs,
                RecommendationSource.MARKET_DRIVEN: market_recs
            }
            
            # Hybrid ranking
            ranked_recommendations = await self._hybrid_ranking(
                all_recommendations, user_profile, context
            )
            
            # Apply diversity and novelty
            final_recommendations = await self._apply_diversity_and_novelty(
                ranked_recommendations, 
                max_recommendations,
                config['diversity_weight'],
                config['novelty_weight'],
                user_id
            )
            
            # Filter by context preferences
            if config['preferred_types']:
                final_recommendations = [
                    rec for rec in final_recommendations
                    if rec.recommendation_type in config['preferred_types']
                ][:max_recommendations]
            
            # Calculate set metrics
            total_score = np.mean([rec.score for rec in final_recommendations]) if final_recommendations else 0.0
            diversity_score = await self._calculate_diversity_score(final_recommendations)
            novelty_score = await self._calculate_novelty_score(final_recommendations, user_id)
            
            recommendation_set = RecommendationSet(
                user_id=user_id,
                context=context,
                recommendations=final_recommendations,
                total_score=total_score,
                diversity_score=diversity_score,
                novelty_score=novelty_score
            )
            
            # Store recommendation history
            self.recommendation_history[user_id].append(recommendation_set)
            
            # Update metrics
            await self._update_recommendation_metrics(recommendation_set)
            
            logger.info(f"Generated {len(final_recommendations)} recommendations for user {user_id}")
            return recommendation_set
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return RecommendationSet(
                user_id=user_id,
                context=context,
                recommendations=[],
                total_score=0.0,
                diversity_score=0.0,
                novelty_score=0.0
            )
    
    async def record_feedback(self, feedback: RecommendationFeedback):
        """Record user feedback on recommendations"""
        try:
            self.feedback_history[feedback.user_id].append(feedback)
            
            # Update recommendation item scores based on feedback
            await self._update_item_scores_from_feedback(feedback)
            
            # Update user interaction history
            interaction = {
                'type': 'feedback',
                'recommendation_id': feedback.recommendation_id,
                'feedback_type': feedback.feedback_type,
                'feedback_value': feedback.feedback_value,
                'timestamp': feedback.timestamp
            }
            self.user_interactions[feedback.user_id].append(interaction)
            
            # Retrain models if enough feedback accumulated
            await self._check_and_retrain_models()
            
            logger.info(f"Recorded feedback for recommendation {feedback.recommendation_id}")
            
        except Exception as e:
            logger.error(f"Error recording feedback: {e}")
    
    async def get_recommendation_explanation(self, recommendation_id: str) -> Dict[str, Any]:
        """Get detailed explanation for a recommendation"""
        try:
            # Find the recommendation
            recommendation = None
            for user_recs in self.recommendation_history.values():
                for rec_set in user_recs:
                    for rec in rec_set.recommendations:
                        if rec.recommendation_id == recommendation_id:
                            recommendation = rec
                            break
                    if recommendation:
                        break
                if recommendation:
                    break
            
            if not recommendation:
                return {'error': 'Recommendation not found'}
            
            explanation = {
                'recommendation_id': recommendation_id,
                'item_title': recommendation.item.title,
                'recommendation_type': recommendation.recommendation_type.value,
                'score': recommendation.score,
                'source': recommendation.source.value,
                'reasoning': recommendation.reasoning,
                'expected_impact': recommendation.expected_impact,
                'risk_level': recommendation.risk_level,
                'personalization_factors': recommendation.personalization_factors,
                'similar_items': await self._find_similar_items(recommendation.item.item_id),
                'user_context': await self._get_user_context_for_recommendation(recommendation),
                'confidence_breakdown': await self._get_confidence_breakdown(recommendation)
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error getting recommendation explanation: {e}")
            return {'error': 'Failed to generate explanation'}
    
    async def get_user_recommendation_stats(self, user_id: str, 
                                          time_window: timedelta = timedelta(days=30)) -> Dict[str, Any]:
        """Get recommendation statistics for a user"""
        try:
            cutoff_time = datetime.now() - time_window
            
            # Filter recent recommendations
            recent_recommendations = []
            for rec_set in self.recommendation_history[user_id]:
                if rec_set.generated_at >= cutoff_time:
                    recent_recommendations.extend(rec_set.recommendations)
            
            # Filter recent feedback
            recent_feedback = [
                feedback for feedback in self.feedback_history[user_id]
                if feedback.timestamp >= cutoff_time
            ]
            
            if not recent_recommendations:
                return {'no_data': True}
            
            # Calculate statistics
            total_recommendations = len(recent_recommendations)
            avg_score = np.mean([rec.score for rec in recent_recommendations])
            
            # Recommendation type distribution
            type_distribution = Counter([rec.recommendation_type.value for rec in recent_recommendations])
            
            # Source distribution
            source_distribution = Counter([rec.source.value for rec in recent_recommendations])
            
            # Priority distribution
            priority_distribution = Counter([rec.priority.value for rec in recent_recommendations])
            
            # Feedback statistics
            feedback_stats = {}
            if recent_feedback:
                feedback_types = Counter([fb.feedback_type for fb in recent_feedback])
                avg_rating = np.mean([
                    fb.feedback_value for fb in recent_feedback 
                    if fb.feedback_value is not None
                ])
                feedback_stats = {
                    'total_feedback': len(recent_feedback),
                    'feedback_types': dict(feedback_types),
                    'average_rating': avg_rating if not np.isnan(avg_rating) else None,
                    'engagement_rate': len(recent_feedback) / total_recommendations
                }
            
            # Performance metrics
            clicked_recs = [fb.recommendation_id for fb in recent_feedback if fb.feedback_type == 'clicked']
            click_through_rate = len(clicked_recs) / total_recommendations if total_recommendations > 0 else 0
            
            acted_upon_recs = [fb.recommendation_id for fb in recent_feedback if fb.feedback_type == 'acted_upon']
            conversion_rate = len(acted_upon_recs) / total_recommendations if total_recommendations > 0 else 0
            
            stats = {
                'user_id': user_id,
                'time_window_days': time_window.days,
                'total_recommendations': total_recommendations,
                'average_score': avg_score,
                'type_distribution': dict(type_distribution),
                'source_distribution': dict(source_distribution),
                'priority_distribution': dict(priority_distribution),
                'feedback_stats': feedback_stats,
                'click_through_rate': click_through_rate,
                'conversion_rate': conversion_rate,
                'analysis_timestamp': datetime.now()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting user recommendation stats: {e}")
            return {'error': 'Failed to generate statistics'}
    
    async def optimize_recommendations(self, user_id: str, 
                                     optimization_target: str = 'engagement') -> Dict[str, Any]:
        """Optimize recommendation strategy for a user"""
        try:
            # Analyze user's historical feedback
            user_feedback = self.feedback_history[user_id]
            if len(user_feedback) < 10:  # Need minimum feedback for optimization
                return {'insufficient_data': True}
            
            # Analyze which recommendation sources perform best
            source_performance = defaultdict(list)
            for feedback in user_feedback:
                # Find the corresponding recommendation
                rec = await self._find_recommendation_by_id(feedback.recommendation_id)
                if rec and feedback.feedback_value is not None:
                    source_performance[rec.source].append(feedback.feedback_value)
            
            # Calculate optimal weights
            optimal_weights = {}
            total_weight = 0
            
            for source, scores in source_performance.items():
                if scores:
                    avg_score = np.mean(scores)
                    weight = max(0.05, avg_score)  # Minimum weight of 5%
                    optimal_weights[source] = weight
                    total_weight += weight
            
            # Normalize weights
            if total_weight > 0:
                for source in optimal_weights:
                    optimal_weights[source] /= total_weight
            
            # Analyze preferred recommendation types
            type_preferences = defaultdict(list)
            for feedback in user_feedback:
                rec = await self._find_recommendation_by_id(feedback.recommendation_id)
                if rec and feedback.feedback_value is not None:
                    type_preferences[rec.recommendation_type].append(feedback.feedback_value)
            
            preferred_types = sorted(
                type_preferences.items(),
                key=lambda x: np.mean(x[1]) if x[1] else 0,
                reverse=True
            )[:3]  # Top 3 preferred types
            
            # Analyze optimal timing
            feedback_hours = [fb.timestamp.hour for fb in user_feedback if fb.feedback_type == 'clicked']
            optimal_hours = Counter(feedback_hours).most_common(3)
            
            optimization_result = {
                'user_id': user_id,
                'optimization_target': optimization_target,
                'optimal_source_weights': optimal_weights,
                'preferred_types': [pref[0].value for pref in preferred_types],
                'optimal_timing_hours': [hour[0] for hour in optimal_hours],
                'confidence_score': min(1.0, len(user_feedback) / 50.0),  # More feedback = higher confidence
                'recommendations': await self._generate_optimization_recommendations(optimal_weights, preferred_types),
                'analysis_timestamp': datetime.now()
            }
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"Error optimizing recommendations: {e}")
            return {'error': 'Failed to optimize recommendations'}
    
    async def _generate_collaborative_recommendations(self, user_id: str, 
                                                   user_profile: Optional[Dict[str, Any]]) -> List[Recommendation]:
        """Generate recommendations using collaborative filtering"""
        try:
            recommendations = []
            
            # Find similar users based on interaction patterns
            similar_users = await self._find_similar_users(user_id)
            
            if not similar_users:
                return recommendations
            
            # Get items liked by similar users
            similar_user_items = set()
            for similar_user_id in similar_users:
                user_feedback = self.feedback_history[similar_user_id]
                positive_feedback = [
                    fb for fb in user_feedback 
                    if fb.feedback_type in ['clicked', 'acted_upon', 'saved'] or 
                    (fb.feedback_value is not None and fb.feedback_value > 0.6)
                ]
                
                for feedback in positive_feedback:
                    rec = await self._find_recommendation_by_id(feedback.recommendation_id)
                    if rec:
                        similar_user_items.add(rec.item.item_id)
            
            # Get user's already seen items
            user_seen_items = set()
            for rec_set in self.recommendation_history[user_id]:
                for rec in rec_set.recommendations:
                    user_seen_items.add(rec.item.item_id)
            
            # Recommend items not yet seen by user
            candidate_items = similar_user_items - user_seen_items
            
            for item_id in list(candidate_items)[:10]:  # Top 10 candidates
                if item_id in self.recommendation_items:
                    item = self.recommendation_items[item_id]
                    
                    # Calculate collaborative score
                    score = await self._calculate_collaborative_score(user_id, item_id, similar_users)
                    
                    recommendation = Recommendation(
                        recommendation_id=f"collab_{user_id}_{item_id}_{datetime.now().timestamp()}",
                        user_id=user_id,
                        item=item,
                        recommendation_type=self._infer_recommendation_type(item),
                        priority=RecommendationPriority.MEDIUM,
                        source=RecommendationSource.COLLABORATIVE_FILTERING,
                        context=RecommendationContext.DASHBOARD,
                        score=score,
                        reasoning=[f"Users with similar preferences also liked this {item.item_type}"],
                        expected_impact="Based on similar user behavior",
                        risk_level="Medium",
                        time_sensitivity="Not urgent",
                        personalization_factors={'collaborative_similarity': score}
                    )
                    
                    recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating collaborative recommendations: {e}")
            return []
    
    async def _generate_content_based_recommendations(self, user_id: str,
                                                    user_profile: Optional[Dict[str, Any]]) -> List[Recommendation]:
        """Generate recommendations using content-based filtering"""
        try:
            recommendations = []
            
            # Get user's interaction history
            user_feedback = self.feedback_history[user_id]
            positive_items = []
            
            for feedback in user_feedback:
                if (feedback.feedback_type in ['clicked', 'acted_upon', 'saved'] or 
                    (feedback.feedback_value is not None and feedback.feedback_value > 0.6)):
                    rec = await self._find_recommendation_by_id(feedback.recommendation_id)
                    if rec:
                        positive_items.append(rec.item)
            
            if not positive_items:
                return recommendations
            
            # Create user preference profile
            user_tags = []
            user_categories = []
            
            for item in positive_items:
                user_tags.extend(item.tags)
                user_categories.append(item.category)
            
            # Find items with similar content
            candidate_items = []
            for item_id, item in self.recommendation_items.items():
                # Check if user has already seen this item
                if await self._user_has_seen_item(user_id, item_id):
                    continue
                
                # Calculate content similarity
                content_score = await self._calculate_content_similarity(item, user_tags, user_categories)
                
                if content_score > 0.3:  # Minimum similarity threshold
                    candidate_items.append((item, content_score))
            
            # Sort by similarity and take top candidates
            candidate_items.sort(key=lambda x: x[1], reverse=True)
            
            for item, score in candidate_items[:10]:
                recommendation = Recommendation(
                    recommendation_id=f"content_{user_id}_{item.item_id}_{datetime.now().timestamp()}",
                    user_id=user_id,
                    item=item,
                    recommendation_type=self._infer_recommendation_type(item),
                    priority=RecommendationPriority.MEDIUM,
                    source=RecommendationSource.CONTENT_BASED,
                    context=RecommendationContext.DASHBOARD,
                    score=score,
                    reasoning=[f"Similar to {item.item_type}s you've shown interest in"],
                    expected_impact="Based on your content preferences",
                    risk_level="Low",
                    time_sensitivity="Not urgent",
                    personalization_factors={'content_similarity': score}
                )
                
                recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating content-based recommendations: {e}")
            return []
    
    async def _generate_behavioral_recommendations(self, user_id: str,
                                                 user_profile: Optional[Dict[str, Any]]) -> List[Recommendation]:
        """Generate recommendations based on user behavior patterns"""
        try:
            recommendations = []
            
            if not user_profile:
                return recommendations
            
            # Analyze user behavior patterns
            behavior_patterns = user_profile.get('behavior_patterns', [])
            
            for pattern in behavior_patterns:
                if pattern.get('pattern_type') == 'preferred_trading_time':
                    # Recommend time-sensitive opportunities
                    preferred_hour = pattern.get('metadata', {}).get('preferred_hour')
                    if preferred_hour is not None:
                        time_sensitive_items = await self._get_time_sensitive_items(preferred_hour)
                        
                        for item in time_sensitive_items[:3]:
                            recommendation = Recommendation(
                                recommendation_id=f"behavioral_{user_id}_{item.item_id}_{datetime.now().timestamp()}",
                                user_id=user_id,
                                item=item,
                                recommendation_type=RecommendationType.TRADING_STRATEGY,
                                priority=RecommendationPriority.HIGH,
                                source=RecommendationSource.BEHAVIORAL,
                                context=RecommendationContext.TRADING_INTERFACE,
                                score=0.8,
                                reasoning=[f"Matches your preferred trading time around {preferred_hour}:00"],
                                expected_impact="Optimized for your trading schedule",
                                risk_level="Medium",
                                time_sensitivity="High",
                                personalization_factors={'time_preference_match': 0.9}
                            )
                            
                            recommendations.append(recommendation)
                
                elif pattern.get('pattern_type') == 'volatility_preference':
                    # Recommend assets matching volatility preference
                    volatility_pref = pattern.get('metadata', {}).get('avg_volatility_preference', 0.5)
                    volatility_items = await self._get_volatility_matched_items(volatility_pref)
                    
                    for item in volatility_items[:3]:
                        recommendation = Recommendation(
                            recommendation_id=f"behavioral_{user_id}_{item.item_id}_{datetime.now().timestamp()}",
                            user_id=user_id,
                            item=item,
                            recommendation_type=RecommendationType.INVESTMENT,
                            priority=RecommendationPriority.MEDIUM,
                            source=RecommendationSource.BEHAVIORAL,
                            context=RecommendationContext.PORTFOLIO_VIEW,
                            score=0.7,
                            reasoning=[f"Matches your volatility preference ({volatility_pref:.1f})"],
                            expected_impact="Aligned with your risk comfort level",
                            risk_level="Medium",
                            time_sensitivity="Medium",
                            personalization_factors={'volatility_match': 0.8}
                        )
                        
                        recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating behavioral recommendations: {e}")
            return []
    
    async def _generate_rule_based_recommendations(self, user_id: str,
                                                 user_profile: Optional[Dict[str, Any]],
                                                 context: RecommendationContext) -> List[Recommendation]:
        """Generate recommendations using rule-based logic"""
        try:
            recommendations = []
            
            if not user_profile:
                return recommendations
            
            # Rule 1: Risk tolerance based recommendations
            risk_tolerance = user_profile.get('risk_tolerance', 'moderate')
            
            if risk_tolerance == 'very_conservative':
                conservative_items = await self._get_conservative_investment_items()
                for item in conservative_items[:2]:
                    recommendation = self._create_rule_based_recommendation(
                        user_id, item, "Conservative investment matching your risk profile",
                        RecommendationType.INVESTMENT, 0.8
                    )
                    recommendations.append(recommendation)
            
            elif risk_tolerance == 'very_aggressive':
                aggressive_items = await self._get_aggressive_investment_items()
                for item in aggressive_items[:2]:
                    recommendation = self._create_rule_based_recommendation(
                        user_id, item, "High-growth opportunity matching your risk appetite",
                        RecommendationType.INVESTMENT, 0.7
                    )
                    recommendations.append(recommendation)
            
            # Rule 2: Technical sophistication based recommendations
            tech_sophistication = user_profile.get('technical_sophistication', 'beginner')
            
            if tech_sophistication == 'beginner':
                educational_items = await self._get_educational_content_items('basic')
                for item in educational_items[:2]:
                    recommendation = self._create_rule_based_recommendation(
                        user_id, item, "Educational content to improve your trading skills",
                        RecommendationType.EDUCATIONAL_CONTENT, 0.9
                    )
                    recommendations.append(recommendation)
            
            elif tech_sophistication in ['expert', 'quantitative']:
                advanced_items = await self._get_advanced_strategy_items()
                for item in advanced_items[:2]:
                    recommendation = self._create_rule_based_recommendation(
                        user_id, item, "Advanced strategy for experienced traders",
                        RecommendationType.TRADING_STRATEGY, 0.8
                    )
                    recommendations.append(recommendation)
            
            # Rule 3: Trading frequency based recommendations
            trading_frequency = user_profile.get('trading_frequency', 'long_term')
            
            if trading_frequency == 'high_frequency':
                hft_items = await self._get_high_frequency_tools()
                for item in hft_items[:2]:
                    recommendation = self._create_rule_based_recommendation(
                        user_id, item, "Tool optimized for high-frequency trading",
                        RecommendationType.FEATURE_SUGGESTION, 0.8
                    )
                    recommendations.append(recommendation)
            
            # Rule 4: Context-specific recommendations
            if context == RecommendationContext.PORTFOLIO_VIEW:
                portfolio_items = await self._get_portfolio_optimization_items()
                for item in portfolio_items[:2]:
                    recommendation = self._create_rule_based_recommendation(
                        user_id, item, "Portfolio optimization opportunity",
                        RecommendationType.PORTFOLIO_OPTIMIZATION, 0.7
                    )
                    recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating rule-based recommendations: {e}")
            return []
    
    async def _generate_market_driven_recommendations(self, user_id: str,
                                                    user_profile: Optional[Dict[str, Any]]) -> List[Recommendation]:
        """Generate recommendations based on current market conditions"""
        try:
            recommendations = []
            
            # Get current market conditions (placeholder - would integrate with market data)
            market_conditions = await self._get_current_market_conditions()
            
            if market_conditions.get('volatility', 'medium') == 'high':
                # High volatility - recommend risk management
                risk_mgmt_items = await self._get_risk_management_items()
                for item in risk_mgmt_items[:2]:
                    recommendation = Recommendation(
                        recommendation_id=f"market_{user_id}_{item.item_id}_{datetime.now().timestamp()}",
                        user_id=user_id,
                        item=item,
                        recommendation_type=RecommendationType.RISK_MANAGEMENT,
                        priority=RecommendationPriority.HIGH,
                        source=RecommendationSource.MARKET_DRIVEN,
                        context=RecommendationContext.DASHBOARD,
                        score=0.9,
                        reasoning=["High market volatility detected - consider risk management"],
                        expected_impact="Protect portfolio during volatile conditions",
                        risk_level="High",
                        time_sensitivity="Urgent",
                        personalization_factors={'market_volatility': 0.9}
                    )
                    recommendations.append(recommendation)
            
            if market_conditions.get('trend', 'neutral') == 'bullish':
                # Bullish market - recommend growth opportunities
                growth_items = await self._get_growth_opportunity_items()
                for item in growth_items[:2]:
                    recommendation = Recommendation(
                        recommendation_id=f"market_{user_id}_{item.item_id}_{datetime.now().timestamp()}",
                        user_id=user_id,
                        item=item,
                        recommendation_type=RecommendationType.INVESTMENT,
                        priority=RecommendationPriority.MEDIUM,
                        source=RecommendationSource.MARKET_DRIVEN,
                        context=RecommendationContext.DASHBOARD,
                        score=0.7,
                        reasoning=["Bullish market trend - growth opportunity"],
                        expected_impact="Capitalize on positive market momentum",
                        risk_level="Medium",
                        time_sensitivity="Medium",
                        personalization_factors={'market_trend': 0.8}
                    )
                    recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating market-driven recommendations: {e}")
            return []
    
    def _create_rule_based_recommendation(self, user_id: str, item: RecommendationItem,
                                        reasoning: str, rec_type: RecommendationType,
                                        score: float) -> Recommendation:
        """Helper to create rule-based recommendations"""
        return Recommendation(
            recommendation_id=f"rule_{user_id}_{item.item_id}_{datetime.now().timestamp()}",
            user_id=user_id,
            item=item,
            recommendation_type=rec_type,
            priority=RecommendationPriority.MEDIUM,
            source=RecommendationSource.RULE_BASED,
            context=RecommendationContext.DASHBOARD,
            score=score,
            reasoning=[reasoning],
            expected_impact="Tailored to your profile",
            risk_level="Medium",
            time_sensitivity="Not urgent",
            personalization_factors={'rule_match': score}
        )
    
    async def _hybrid_ranking(self, all_recommendations: Dict[RecommendationSource, List[Recommendation]],
                            user_profile: Optional[Dict[str, Any]],
                            context: RecommendationContext) -> List[Recommendation]:
        """Combine and rank recommendations from different sources"""
        try:
            combined_recommendations = []
            
            # Combine all recommendations with weighted scores
            for source, recommendations in all_recommendations.items():
                weight = self.strategy_weights.get(source, 0.1)
                
                for rec in recommendations:
                    # Apply source weight to score
                    rec.score = rec.score * weight
                    combined_recommendations.append(rec)
            
            # Remove duplicates (same item recommended by multiple sources)
            unique_recommendations = {}
            for rec in combined_recommendations:
                item_id = rec.item.item_id
                if item_id not in unique_recommendations or rec.score > unique_recommendations[item_id].score:
                    unique_recommendations[item_id] = rec
            
            # Convert back to list and sort by score
            final_recommendations = list(unique_recommendations.values())
            final_recommendations.sort(key=lambda x: x.score, reverse=True)
            
            return final_recommendations
            
        except Exception as e:
            logger.error(f"Error in hybrid ranking: {e}")
            return []
    
    async def _apply_diversity_and_novelty(self, recommendations: List[Recommendation],
                                         max_recommendations: int,
                                         diversity_weight: float,
                                         novelty_weight: float,
                                         user_id: str) -> List[Recommendation]:
        """Apply diversity and novelty constraints to recommendations"""
        try:
            if len(recommendations) <= max_recommendations:
                return recommendations
            
            selected_recommendations = []
            remaining_recommendations = recommendations.copy()
            
            # Always include the top recommendation
            if remaining_recommendations:
                selected_recommendations.append(remaining_recommendations.pop(0))
            
            # Select remaining recommendations balancing score, diversity, and novelty
            while len(selected_recommendations) < max_recommendations and remaining_recommendations:
                best_rec = None
                best_score = -1
                
                for rec in remaining_recommendations:
                    # Calculate diversity score
                    diversity_score = await self._calculate_diversity_with_selected(rec, selected_recommendations)
                    
                    # Calculate novelty score
                    novelty_score = await self._calculate_novelty_for_user(rec, user_id)
                    
                    # Combined score
                    combined_score = (rec.score * (1 - diversity_weight - novelty_weight) +
                                    diversity_score * diversity_weight +
                                    novelty_score * novelty_weight)
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_rec = rec
                
                if best_rec:
                    selected_recommendations.append(best_rec)
                    remaining_recommendations.remove(best_rec)
                else:
                    break
            
            return selected_recommendations
            
        except Exception as e:
            logger.error(f"Error applying diversity and novelty: {e}")
            return recommendations[:max_recommendations]
    
    # Helper methods for recommendation generation
    async def _find_similar_users(self, user_id: str, top_k: int = 5) -> List[str]:
        """Find users with similar interaction patterns"""
        try:
            # Simplified similarity based on feedback patterns
            user_feedback = self.feedback_history[user_id]
            if not user_feedback:
                return []
            
            user_items = set()
            for feedback in user_feedback:
                if feedback.feedback_type in ['clicked', 'acted_upon', 'saved']:
                    rec = await self._find_recommendation_by_id(feedback.recommendation_id)
                    if rec:
                        user_items.add(rec.item.item_id)
            
            if not user_items:
                return []
            
            # Find users with overlapping items
            similar_users = []
            for other_user_id, other_feedback in self.feedback_history.items():
                if other_user_id == user_id:
                    continue
                
                other_items = set()
                for feedback in other_feedback:
                    if feedback.feedback_type in ['clicked', 'acted_upon', 'saved']:
                        rec = await self._find_recommendation_by_id(feedback.recommendation_id)
                        if rec:
                            other_items.add(rec.item.item_id)
                
                if other_items:
                    # Calculate Jaccard similarity
                    intersection = len(user_items & other_items)
                    union = len(user_items | other_items)
                    similarity = intersection / union if union > 0 else 0
                    
                    if similarity > 0.1:  # Minimum similarity threshold
                        similar_users.append((other_user_id, similarity))
            
            # Sort by similarity and return top k
            similar_users.sort(key=lambda x: x[1], reverse=True)
            return [user_id for user_id, sim in similar_users[:top_k]]
            
        except Exception as e:
            logger.error(f"Error finding similar users: {e}")
            return []
    
    async def _calculate_collaborative_score(self, user_id: str, item_id: str, similar_users: List[str]) -> float:
        """Calculate collaborative filtering score"""
        try:
            total_score = 0.0
            total_weight = 0.0
            
            for similar_user_id in similar_users:
                # Find feedback for this item from similar user
                user_feedback = self.feedback_history[similar_user_id]
                item_feedback = None
                
                for feedback in user_feedback:
                    rec = await self._find_recommendation_by_id(feedback.recommendation_id)
                    if rec and rec.item.item_id == item_id:
                        item_feedback = feedback
                        break
                
                if item_feedback:
                    # Convert feedback to score
                    if item_feedback.feedback_type == 'acted_upon':
                        score = 1.0
                    elif item_feedback.feedback_type == 'saved':
                        score = 0.8
                    elif item_feedback.feedback_type == 'clicked':
                        score = 0.6
                    elif item_feedback.feedback_value is not None:
                        score = item_feedback.feedback_value
                    else:
                        score = 0.5
                    
                    # Weight by user similarity (simplified as equal weight)
                    weight = 1.0
                    total_score += score * weight
                    total_weight += weight
            
            return total_score / total_weight if total_weight > 0 else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating collaborative score: {e}")
            return 0.5
    
    async def _calculate_content_similarity(self, item: RecommendationItem, 
                                          user_tags: List[str], user_categories: List[str]) -> float:
        """Calculate content-based similarity score"""
        try:
            # Tag similarity
            item_tags_set = set(item.tags)
            user_tags_set = set(user_tags)
            
            tag_intersection = len(item_tags_set & user_tags_set)
            tag_union = len(item_tags_set | user_tags_set)
            tag_similarity = tag_intersection / tag_union if tag_union > 0 else 0
            
            # Category similarity
            category_similarity = 1.0 if item.category in user_categories else 0.0
            
            # Combined similarity
            similarity = 0.7 * tag_similarity + 0.3 * category_similarity
            
            return similarity
            
        except Exception as e:
            logger.error(f"Error calculating content similarity: {e}")
            return 0.0
    
    async def _user_has_seen_item(self, user_id: str, item_id: str) -> bool:
        """Check if user has already seen an item"""
        try:
            for rec_set in self.recommendation_history[user_id]:
                for rec in rec_set.recommendations:
                    if rec.item.item_id == item_id:
                        return True
            return False
        except Exception as e:
            logger.error(f"Error checking if user has seen item: {e}")
            return False
    
    def _infer_recommendation_type(self, item: RecommendationItem) -> RecommendationType:
        """Infer recommendation type from item"""
        try:
            item_type = item.item_type.lower()
            
            if 'stock' in item_type or 'etf' in item_type or 'crypto' in item_type:
                return RecommendationType.INVESTMENT
            elif 'strategy' in item_type:
                return RecommendationType.TRADING_STRATEGY
            elif 'education' in item_type or 'tutorial' in item_type:
                return RecommendationType.EDUCATIONAL_CONTENT
            elif 'analysis' in item_type or 'report' in item_type:
                return RecommendationType.MARKET_ANALYSIS
            elif 'feature' in item_type or 'tool' in item_type:
                return RecommendationType.FEATURE_SUGGESTION
            else:
                return RecommendationType.INVESTMENT
                
        except Exception as e:
            logger.error(f"Error inferring recommendation type: {e}")
            return RecommendationType.INVESTMENT
    
    # Placeholder methods for item retrieval (would integrate with actual data sources)
    async def _get_time_sensitive_items(self, preferred_hour: int) -> List[RecommendationItem]:
        """Get items relevant to specific trading hours"""
        # Placeholder implementation
        return []
    
    async def _get_volatility_matched_items(self, volatility_pref: float) -> List[RecommendationItem]:
        """Get items matching volatility preference"""
        # Placeholder implementation
        return []
    
    async def _get_conservative_investment_items(self) -> List[RecommendationItem]:
        """Get conservative investment options"""
        # Placeholder implementation
        return []
    
    async def _get_aggressive_investment_items(self) -> List[RecommendationItem]:
        """Get aggressive investment options"""
        # Placeholder implementation
        return []
    
    async def _get_educational_content_items(self, level: str) -> List[RecommendationItem]:
        """Get educational content for specific level"""
        # Placeholder implementation
        return []
    
    async def _get_advanced_strategy_items(self) -> List[RecommendationItem]:
        """Get advanced trading strategies"""
        # Placeholder implementation
        return []
    
    async def _get_high_frequency_tools(self) -> List[RecommendationItem]:
        """Get tools for high-frequency trading"""
        # Placeholder implementation
        return []
    
    async def _get_portfolio_optimization_items(self) -> List[RecommendationItem]:
        """Get portfolio optimization suggestions"""
        # Placeholder implementation
        return []
    
    async def _get_current_market_conditions(self) -> Dict[str, Any]:
        """Get current market conditions"""
        # Placeholder implementation
        return {'volatility': 'medium', 'trend': 'neutral'}
    
    async def _get_risk_management_items(self) -> List[RecommendationItem]:
        """Get risk management tools and strategies"""
        # Placeholder implementation
        return []
    
    async def _get_growth_opportunity_items(self) -> List[RecommendationItem]:
        """Get growth investment opportunities"""
        # Placeholder implementation
        return []
    
    async def _find_recommendation_by_id(self, recommendation_id: str) -> Optional[Recommendation]:
        """Find recommendation by ID"""
        try:
            for user_recs in self.recommendation_history.values():
                for rec_set in user_recs:
                    for rec in rec_set.recommendations:
                        if rec.recommendation_id == recommendation_id:
                            return rec
            return None
        except Exception as e:
            logger.error(f"Error finding recommendation by ID: {e}")
            return None
    
    async def _update_content_features(self):
        """Update content-based features"""
        try:
            # Extract text features from items
            texts = []
            for item in self.recommendation_items.values():
                text = f"{item.title} {item.description} {' '.join(item.tags)}"
                texts.append(text)
            
            if texts:
                self.content_features = self.content_vectorizer.fit_transform(texts)
                
        except Exception as e:
            logger.error(f"Error updating content features: {e}")
    
    async def _update_item_scores_from_feedback(self, feedback: RecommendationFeedback):
        """Update item scores based on user feedback"""
        try:
            rec = await self._find_recommendation_by_id(feedback.recommendation_id)
            if not rec:
                return
            
            item = rec.item
            
            # Update popularity score
            if feedback.feedback_type in ['clicked', 'acted_upon', 'saved']:
                item.popularity_score = min(1.0, item.popularity_score + 0.01)
            
            # Update quality score based on ratings
            if feedback.feedback_value is not None:
                # Simple moving average update
                alpha = 0.1  # Learning rate
                item.quality_score = (1 - alpha) * item.quality_score + alpha * feedback.feedback_value
                
        except Exception as e:
            logger.error(f"Error updating item scores from feedback: {e}")
    
    async def _check_and_retrain_models(self):
        """Check if models need retraining based on feedback volume"""
        try:
            total_feedback = sum(len(feedback_list) for feedback_list in self.feedback_history.values())
            
            # Retrain every 1000 feedback points
            if total_feedback % 1000 == 0 and total_feedback > 0:
                logger.info("Retraining recommendation models...")
                # Placeholder for model retraining
                
        except Exception as e:
            logger.error(f"Error checking model retraining: {e}")
    
    async def _calculate_diversity_score(self, recommendations: List[Recommendation]) -> float:
        """Calculate diversity score for a set of recommendations"""
        try:
            if len(recommendations) <= 1:
                return 1.0
            
            # Calculate diversity based on recommendation types and categories
            types = [rec.recommendation_type for rec in recommendations]
            categories = [rec.item.category for rec in recommendations]
            
            type_diversity = len(set(types)) / len(types)
            category_diversity = len(set(categories)) / len(categories)
            
            return (type_diversity + category_diversity) / 2
            
        except Exception as e:
            logger.error(f"Error calculating diversity score: {e}")
            return 0.5
    
    async def _calculate_novelty_score(self, recommendations: List[Recommendation], user_id: str) -> float:
        """Calculate novelty score for recommendations"""
        try:
            if not recommendations:
                return 0.0
            
            # Get user's historical items
            user_items = set()
            for rec_set in self.recommendation_history[user_id]:
                for rec in rec_set.recommendations:
                    user_items.add(rec.item.item_id)
            
            # Calculate novelty as percentage of new items
            new_items = sum(1 for rec in recommendations if rec.item.item_id not in user_items)
            novelty = new_items / len(recommendations) if recommendations else 0
            
            return novelty
            
        except Exception as e:
            logger.error(f"Error calculating novelty score: {e}")
            return 0.5
    
    async def _calculate_diversity_with_selected(self, candidate: Recommendation, 
                                               selected: List[Recommendation]) -> float:
        """Calculate diversity of candidate with already selected recommendations"""
        try:
            if not selected:
                return 1.0
            
            # Check type diversity
            selected_types = set(rec.recommendation_type for rec in selected)
            type_diversity = 1.0 if candidate.recommendation_type not in selected_types else 0.5
            
            # Check category diversity
            selected_categories = set(rec.item.category for rec in selected)
            category_diversity = 1.0 if candidate.item.category not in selected_categories else 0.5
            
            return (type_diversity + category_diversity) / 2
            
        except Exception as e:
            logger.error(f"Error calculating diversity with selected: {e}")
            return 0.5
    
    async def _calculate_novelty_for_user(self, recommendation: Recommendation, user_id: str) -> float:
        """Calculate novelty of a recommendation for a specific user"""
        try:
            # Check if user has seen this item before
            has_seen = await self._user_has_seen_item(user_id, recommendation.item.item_id)
            
            if has_seen:
                return 0.0
            
            # Check if user has seen similar items
            similar_items = await self._find_similar_items(recommendation.item.item_id)
            user_items = set()
            
            for rec_set in self.recommendation_history[user_id]:
                for rec in rec_set.recommendations:
                    user_items.add(rec.item.item_id)
            
            similar_seen = sum(1 for item_id in similar_items if item_id in user_items)
            novelty = 1.0 - (similar_seen / len(similar_items)) if similar_items else 1.0
            
            return novelty
            
        except Exception as e:
            logger.error(f"Error calculating novelty for user: {e}")
            return 0.5
    
    async def _find_similar_items(self, item_id: str, top_k: int = 5) -> List[str]:
        """Find items similar to the given item"""
        try:
            if item_id not in self.recommendation_items:
                return []
            
            target_item = self.recommendation_items[item_id]
            similarities = []
            
            for other_id, other_item in self.recommendation_items.items():
                if other_id == item_id:
                    continue
                
                # Calculate similarity based on tags and category
                similarity = await self._calculate_content_similarity(
                    other_item, target_item.tags, [target_item.category]
                )
                
                if similarity > 0.3:
                    similarities.append((other_id, similarity))
            
            # Sort by similarity and return top k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return [item_id for item_id, sim in similarities[:top_k]]
            
        except Exception as e:
            logger.error(f"Error finding similar items: {e}")
            return []
    
    async def _get_user_context_for_recommendation(self, recommendation: Recommendation) -> Dict[str, Any]:
        """Get user context that influenced the recommendation"""
        try:
            user_id = recommendation.user_id
            
            # Get recent user activity
            recent_feedback = [
                fb for fb in self.feedback_history[user_id]
                if fb.timestamp >= datetime.now() - timedelta(days=7)
            ]
            
            # Get recent recommendations
            recent_recs = []
            for rec_set in self.recommendation_history[user_id]:
                if rec_set.generated_at >= datetime.now() - timedelta(days=7):
                    recent_recs.extend(rec_set.recommendations)
            
            context = {
                'recent_feedback_count': len(recent_feedback),
                'recent_recommendations_count': len(recent_recs),
                'feedback_types': Counter([fb.feedback_type for fb in recent_feedback]),
                'recommendation_types': Counter([rec.recommendation_type.value for rec in recent_recs]),
                'avg_recent_score': np.mean([rec.score for rec in recent_recs]) if recent_recs else 0
            }
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting user context: {e}")
            return {}
    
    async def _get_confidence_breakdown(self, recommendation: Recommendation) -> Dict[str, float]:
        """Get detailed confidence breakdown for a recommendation"""
        try:
            breakdown = {
                'base_score': recommendation.score,
                'item_quality': recommendation.item.quality_score,
                'item_popularity': recommendation.item.popularity_score,
                'personalization_strength': sum(recommendation.personalization_factors.values()),
                'source_reliability': self.strategy_weights.get(recommendation.source, 0.1)
            }
            
            # Normalize to 0-1 range
            total = sum(breakdown.values())
            if total > 0:
                breakdown = {k: v/total for k, v in breakdown.items()}
            
            return breakdown
            
        except Exception as e:
            logger.error(f"Error getting confidence breakdown: {e}")
            return {}
    
    async def _generate_optimization_recommendations(self, optimal_weights: Dict[RecommendationSource, float],
                                                   preferred_types: List[Tuple]) -> List[str]:
        """Generate recommendations for optimizing the recommendation strategy"""
        try:
            recommendations = []
            
            # Analyze optimal weights
            best_source = max(optimal_weights.items(), key=lambda x: x[1])
            recommendations.append(f"Focus on {best_source[0].value} recommendations (weight: {best_source[1]:.2f})")
            
            # Analyze preferred types
            if preferred_types:
                top_type = preferred_types[0][0].value
                recommendations.append(f"Prioritize {top_type} recommendations based on your preferences")
            
            # General optimization suggestions
            if len(optimal_weights) < 3:
                recommendations.append("Consider diversifying recommendation sources for better coverage")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating optimization recommendations: {e}")
            return []
    
    async def _update_recommendation_metrics(self, recommendation_set: RecommendationSet):
        """Update performance metrics for recommendations"""
        try:
            user_id = recommendation_set.user_id
            context = recommendation_set.context.value
            
            if user_id not in self.recommendation_metrics:
                self.recommendation_metrics[user_id] = {}
            
            if context not in self.recommendation_metrics[user_id]:
                self.recommendation_metrics[user_id][context] = {
                    'total_recommendations': 0,
                    'total_score': 0.0,
                    'diversity_scores': [],
                    'novelty_scores': []
                }
            
            metrics = self.recommendation_metrics[user_id][context]
            metrics['total_recommendations'] += len(recommendation_set.recommendations)
            metrics['total_score'] += recommendation_set.total_score
            metrics['diversity_scores'].append(recommendation_set.diversity_score)
            metrics['novelty_scores'].append(recommendation_set.novelty_score)
            
        except Exception as e:
            logger.error(f"Error updating recommendation metrics: {e}")
    
    async def get_recommendation_performance(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get recommendation performance metrics"""
        try:
            if user_id:
                # User-specific metrics
                if user_id not in self.recommendation_metrics:
                    return {'no_data': True}
                
                user_metrics = self.recommendation_metrics[user_id]
                performance = {
                    'user_id': user_id,
                    'contexts': {}
                }
                
                for context, metrics in user_metrics.items():
                    performance['contexts'][context] = {
                        'total_recommendations': metrics['total_recommendations'],
                        'average_score': metrics['total_score'] / max(1, metrics['total_recommendations']),
                        'average_diversity': np.mean(metrics['diversity_scores']) if metrics['diversity_scores'] else 0,
                        'average_novelty': np.mean(metrics['novelty_scores']) if metrics['novelty_scores'] else 0
                    }
                
                return performance
            
            else:
                # Global metrics
                total_users = len(self.recommendation_metrics)
                total_recommendations = 0
                total_score = 0.0
                all_diversity_scores = []
                all_novelty_scores = []
                
                for user_metrics in self.recommendation_metrics.values():
                    for context_metrics in user_metrics.values():
                        total_recommendations += context_metrics['total_recommendations']
                        total_score += context_metrics['total_score']
                        all_diversity_scores.extend(context_metrics['diversity_scores'])
                        all_novelty_scores.extend(context_metrics['novelty_scores'])
                
                performance = {
                    'total_users': total_users,
                    'total_recommendations': total_recommendations,
                    'average_score': total_score / max(1, total_recommendations),
                    'average_diversity': np.mean(all_diversity_scores) if all_diversity_scores else 0,
                    'average_novelty': np.mean(all_novelty_scores) if all_novelty_scores else 0
                }
                
                return performance
            
        except Exception as e:
            logger.error(f"Error getting recommendation performance: {e}")
            return {'error': 'Failed to get performance metrics'}
    
    async def export_recommendations(self, user_id: str, 
                                   time_window: timedelta = timedelta(days=30)) -> Dict[str, Any]:
        """Export user's recommendation history"""
        try:
            cutoff_time = datetime.now() - time_window
            
            # Filter recent recommendations
            recent_recommendation_sets = [
                rec_set for rec_set in self.recommendation_history[user_id]
                if rec_set.generated_at >= cutoff_time
            ]
            
            export_data = {
                'user_id': user_id,
                'export_timestamp': datetime.now(),
                'time_window_days': time_window.days,
                'total_recommendation_sets': len(recent_recommendation_sets),
                'recommendation_sets': []
            }
            
            for rec_set in recent_recommendation_sets:
                set_data = {
                    'generated_at': rec_set.generated_at,
                    'context': rec_set.context.value,
                    'total_score': rec_set.total_score,
                    'diversity_score': rec_set.diversity_score,
                    'novelty_score': rec_set.novelty_score,
                    'recommendations': []
                }
                
                for rec in rec_set.recommendations:
                    rec_data = {
                        'recommendation_id': rec.recommendation_id,
                        'item_title': rec.item.title,
                        'item_type': rec.item.item_type,
                        'recommendation_type': rec.recommendation_type.value,
                        'priority': rec.priority.value,
                        'source': rec.source.value,
                        'score': rec.score,
                        'reasoning': rec.reasoning,
                        'expected_impact': rec.expected_impact,
                        'risk_level': rec.risk_level,
                        'time_sensitivity': rec.time_sensitivity
                    }
                    set_data['recommendations'].append(rec_data)
                
                export_data['recommendation_sets'].append(set_data)
            
            return export_data
            
        except Exception as e:
            logger.error(f"Error exporting recommendations: {e}")
            return {'error': 'Failed to export recommendations'}
    
    async def clear_user_data(self, user_id: str):
        """Clear all data for a specific user"""
        try:
            # Clear recommendation history
            if user_id in self.recommendation_history:
                del self.recommendation_history[user_id]
            
            # Clear feedback history
            if user_id in self.feedback_history:
                del self.feedback_history[user_id]
            
            # Clear user interactions
            if user_id in self.user_interactions:
                del self.user_interactions[user_id]
            
            # Clear metrics
            if user_id in self.recommendation_metrics:
                del self.recommendation_metrics[user_id]
            
            logger.info(f"Cleared all data for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error clearing user data: {e}")

# Export classes and functions
__all__ = [
    'RecommendationType',
    'RecommendationPriority', 
    'RecommendationSource',
    'RecommendationContext',
    'RecommendationItem',
    'Recommendation',
    'RecommendationFeedback',
    'RecommendationSet',
    'RecommendationEngine'
]