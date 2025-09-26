# User Profiler
# Phase 9: AI-First Platform Implementation

import logging
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ProfileDimension(Enum):
    RISK_TOLERANCE = "risk_tolerance"
    INVESTMENT_STYLE = "investment_style"
    TRADING_FREQUENCY = "trading_frequency"
    ASSET_PREFERENCE = "asset_preference"
    TIME_HORIZON = "time_horizon"
    MARKET_SENTIMENT = "market_sentiment"
    TECHNICAL_SOPHISTICATION = "technical_sophistication"
    INFORMATION_CONSUMPTION = "information_consumption"
    DECISION_MAKING_STYLE = "decision_making_style"
    FINANCIAL_GOALS = "financial_goals"

class RiskTolerance(Enum):
    VERY_CONSERVATIVE = "very_conservative"  # 0-20%
    CONSERVATIVE = "conservative"            # 20-40%
    MODERATE = "moderate"                    # 40-60%
    AGGRESSIVE = "aggressive"                # 60-80%
    VERY_AGGRESSIVE = "very_aggressive"      # 80-100%

class InvestmentStyle(Enum):
    VALUE_INVESTOR = "value_investor"
    GROWTH_INVESTOR = "growth_investor"
    MOMENTUM_TRADER = "momentum_trader"
    CONTRARIAN = "contrarian"
    DIVIDEND_FOCUSED = "dividend_focused"
    INDEX_INVESTOR = "index_investor"
    SWING_TRADER = "swing_trader"
    DAY_TRADER = "day_trader"
    ALGORITHMIC = "algorithmic"

class TradingFrequency(Enum):
    HODLER = "hodler"                        # < 1 trade/month
    LONG_TERM = "long_term"                  # 1-5 trades/month
    MEDIUM_TERM = "medium_term"              # 5-20 trades/month
    ACTIVE = "active"                        # 20-100 trades/month
    HIGH_FREQUENCY = "high_frequency"        # > 100 trades/month

class TechnicalSophistication(Enum):
    BEGINNER = "beginner"                    # Basic understanding
    INTERMEDIATE = "intermediate"            # Some technical knowledge
    ADVANCED = "advanced"                    # Strong technical skills
    EXPERT = "expert"                        # Professional level
    QUANTITATIVE = "quantitative"            # Mathematical/algorithmic

@dataclass
class BehaviorPattern:
    pattern_type: str
    frequency: float  # 0-1
    strength: float   # 0-1
    consistency: float  # 0-1
    trend: str  # increasing, decreasing, stable
    last_observed: datetime
    confidence: float  # 0-1
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProfileScore:
    dimension: ProfileDimension
    score: float  # 0-1 normalized score
    confidence: float  # 0-1
    evidence_count: int
    last_updated: datetime
    trend: str  # increasing, decreasing, stable
    percentile: float  # 0-100, compared to other users

@dataclass
class UserProfile:
    user_id: str
    profile_scores: Dict[ProfileDimension, ProfileScore]
    risk_tolerance: RiskTolerance
    investment_style: InvestmentStyle
    trading_frequency: TradingFrequency
    technical_sophistication: TechnicalSophistication
    behavior_patterns: List[BehaviorPattern]
    preferences: Dict[str, Any]
    demographics: Dict[str, Any]
    financial_situation: Dict[str, Any]
    goals: List[str]
    constraints: List[str]
    profile_completeness: float  # 0-1
    last_updated: datetime
    profile_version: str

@dataclass
class ProfileAnalysis:
    user_profile: UserProfile
    similar_users: List[str]  # User IDs of similar profiles
    cluster_id: int
    anomalies: List[str]  # Detected anomalies in behavior
    recommendations: List[str]
    confidence_score: float  # 0-1
    analysis_timestamp: datetime
    key_insights: List[str]
    risk_factors: List[str]
    opportunities: List[str]

class UserProfiler:
    """Advanced user profiling and behavior analysis engine"""
    
    def __init__(self):
        self.user_profiles = {}  # Cache of user profiles
        self.behavior_history = defaultdict(list)
        self.profile_clusters = {}
        self.similarity_matrix = {}
        
        # ML models for profiling
        self.clustering_model = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=10)
        
        # Behavior tracking
        self.behavior_weights = {
            'trade_execution': 0.3,
            'portfolio_changes': 0.25,
            'research_activity': 0.2,
            'platform_usage': 0.15,
            'social_interaction': 0.1
        }
        
        # Profile evolution tracking
        self.profile_history = defaultdict(list)
        
        logger.info("User profiler initialized")
    
    async def create_user_profile(self, user_id: str, 
                                initial_data: Dict[str, Any] = None) -> UserProfile:
        """Create initial user profile"""
        try:
            # Initialize profile scores
            profile_scores = {}
            for dimension in ProfileDimension:
                profile_scores[dimension] = ProfileScore(
                    dimension=dimension,
                    score=0.5,  # Neutral starting point
                    confidence=0.1,  # Low initial confidence
                    evidence_count=0,
                    last_updated=datetime.now(),
                    trend="stable",
                    percentile=50.0
                )
            
            # Set initial values from provided data
            if initial_data:
                await self._apply_initial_data(profile_scores, initial_data)
            
            # Create user profile
            user_profile = UserProfile(
                user_id=user_id,
                profile_scores=profile_scores,
                risk_tolerance=RiskTolerance.MODERATE,
                investment_style=InvestmentStyle.INDEX_INVESTOR,
                trading_frequency=TradingFrequency.LONG_TERM,
                technical_sophistication=TechnicalSophistication.BEGINNER,
                behavior_patterns=[],
                preferences={},
                demographics=initial_data.get('demographics', {}),
                financial_situation=initial_data.get('financial_situation', {}),
                goals=initial_data.get('goals', []),
                constraints=initial_data.get('constraints', []),
                profile_completeness=0.1,
                last_updated=datetime.now(),
                profile_version="1.0"
            )
            
            # Cache the profile
            self.user_profiles[user_id] = user_profile
            
            logger.info(f"Created user profile for {user_id}")
            return user_profile
            
        except Exception as e:
            logger.error(f"Error creating user profile: {e}")
            raise
    
    async def update_user_profile(self, user_id: str, 
                                behavior_data: Dict[str, Any]) -> Optional[UserProfile]:
        """Update user profile based on new behavior data"""
        try:
            # Get existing profile or create new one
            if user_id not in self.user_profiles:
                await self.create_user_profile(user_id)
            
            profile = self.user_profiles[user_id]
            
            # Store behavior data
            self.behavior_history[user_id].append({
                'timestamp': datetime.now(),
                'data': behavior_data
            })
            
            # Update profile scores based on behavior
            await self._update_profile_scores(profile, behavior_data)
            
            # Update derived attributes
            await self._update_derived_attributes(profile)
            
            # Detect behavior patterns
            await self._detect_behavior_patterns(profile)
            
            # Update profile completeness
            profile.profile_completeness = await self._calculate_completeness(profile)
            
            # Update timestamp and version
            profile.last_updated = datetime.now()
            profile.profile_version = self._increment_version(profile.profile_version)
            
            # Store profile history
            self.profile_history[user_id].append({
                'timestamp': datetime.now(),
                'profile_snapshot': self._serialize_profile(profile)
            })
            
            logger.info(f"Updated user profile for {user_id}")
            return profile
            
        except Exception as e:
            logger.error(f"Error updating user profile: {e}")
            return None
    
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile by ID"""
        try:
            if user_id in self.user_profiles:
                return self.user_profiles[user_id]
            
            # Try to load from persistent storage (placeholder)
            # In real implementation, this would load from database
            logger.warning(f"Profile not found for user {user_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting user profile: {e}")
            return None
    
    async def analyze_user_profile(self, user_id: str) -> Optional[ProfileAnalysis]:
        """Perform comprehensive profile analysis"""
        try:
            profile = await self.get_user_profile(user_id)
            if not profile:
                return None
            
            # Find similar users
            similar_users = await self._find_similar_users(profile)
            
            # Determine cluster
            cluster_id = await self._get_user_cluster(profile)
            
            # Detect anomalies
            anomalies = await self._detect_anomalies(profile)
            
            # Generate recommendations
            recommendations = await self._generate_profile_recommendations(profile)
            
            # Calculate confidence score
            confidence_score = await self._calculate_analysis_confidence(profile)
            
            # Extract key insights
            key_insights = await self._extract_key_insights(profile)
            
            # Identify risk factors
            risk_factors = await self._identify_risk_factors(profile)
            
            # Find opportunities
            opportunities = await self._identify_opportunities(profile)
            
            analysis = ProfileAnalysis(
                user_profile=profile,
                similar_users=similar_users,
                cluster_id=cluster_id,
                anomalies=anomalies,
                recommendations=recommendations,
                confidence_score=confidence_score,
                analysis_timestamp=datetime.now(),
                key_insights=key_insights,
                risk_factors=risk_factors,
                opportunities=opportunities
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing user profile: {e}")
            return None
    
    async def cluster_users(self, user_ids: List[str] = None) -> Dict[int, List[str]]:
        """Cluster users based on their profiles"""
        try:
            # Use all users if none specified
            if user_ids is None:
                user_ids = list(self.user_profiles.keys())
            
            if len(user_ids) < 2:
                return {0: user_ids}
            
            # Extract features for clustering
            features = []
            valid_users = []
            
            for user_id in user_ids:
                profile = self.user_profiles.get(user_id)
                if profile:
                    feature_vector = await self._extract_feature_vector(profile)
                    features.append(feature_vector)
                    valid_users.append(user_id)
            
            if len(features) < 2:
                return {0: valid_users}
            
            # Normalize features
            features_array = np.array(features)
            features_scaled = self.scaler.fit_transform(features_array)
            
            # Determine optimal number of clusters
            n_clusters = min(5, max(2, len(valid_users) // 10))
            
            # Perform clustering
            self.clustering_model = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = self.clustering_model.fit_predict(features_scaled)
            
            # Group users by cluster
            clusters = defaultdict(list)
            for user_id, cluster_id in zip(valid_users, cluster_labels):
                clusters[cluster_id].append(user_id)
                self.profile_clusters[user_id] = cluster_id
            
            logger.info(f"Clustered {len(valid_users)} users into {n_clusters} clusters")
            return dict(clusters)
            
        except Exception as e:
            logger.error(f"Error clustering users: {e}")
            return {}
    
    async def get_profile_evolution(self, user_id: str, 
                                  time_window: timedelta = timedelta(days=30)) -> Dict[str, Any]:
        """Analyze how user profile has evolved over time"""
        try:
            if user_id not in self.profile_history:
                return {}
            
            # Filter history by time window
            cutoff_time = datetime.now() - time_window
            relevant_history = [
                entry for entry in self.profile_history[user_id]
                if entry['timestamp'] >= cutoff_time
            ]
            
            if len(relevant_history) < 2:
                return {'insufficient_data': True}
            
            # Analyze changes in profile dimensions
            dimension_changes = {}
            for dimension in ProfileDimension:
                changes = []
                for entry in relevant_history:
                    profile_data = entry['profile_snapshot']
                    if dimension.value in profile_data.get('profile_scores', {}):
                        score_data = profile_data['profile_scores'][dimension.value]
                        changes.append({
                            'timestamp': entry['timestamp'],
                            'score': score_data['score'],
                            'confidence': score_data['confidence']
                        })
                
                if len(changes) > 1:
                    # Calculate trend
                    scores = [change['score'] for change in changes]
                    trend_slope = np.polyfit(range(len(scores)), scores, 1)[0]
                    
                    dimension_changes[dimension.value] = {
                        'trend_slope': trend_slope,
                        'trend_direction': 'increasing' if trend_slope > 0.01 else 'decreasing' if trend_slope < -0.01 else 'stable',
                        'volatility': np.std(scores),
                        'total_change': scores[-1] - scores[0],
                        'data_points': len(changes)
                    }
            
            # Analyze behavior pattern evolution
            behavior_evolution = await self._analyze_behavior_evolution(user_id, time_window)
            
            evolution_analysis = {
                'user_id': user_id,
                'time_window_days': time_window.days,
                'dimension_changes': dimension_changes,
                'behavior_evolution': behavior_evolution,
                'overall_stability': np.mean([change['volatility'] for change in dimension_changes.values()]),
                'analysis_timestamp': datetime.now()
            }
            
            return evolution_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing profile evolution: {e}")
            return {}
    
    async def compare_profiles(self, user_id1: str, user_id2: str) -> Dict[str, Any]:
        """Compare two user profiles"""
        try:
            profile1 = await self.get_user_profile(user_id1)
            profile2 = await self.get_user_profile(user_id2)
            
            if not profile1 or not profile2:
                return {'error': 'One or both profiles not found'}
            
            # Compare profile dimensions
            dimension_comparison = {}
            for dimension in ProfileDimension:
                score1 = profile1.profile_scores[dimension].score
                score2 = profile2.profile_scores[dimension].score
                
                dimension_comparison[dimension.value] = {
                    'user1_score': score1,
                    'user2_score': score2,
                    'difference': abs(score1 - score2),
                    'similarity': 1 - abs(score1 - score2),
                    'direction': 'user1_higher' if score1 > score2 else 'user2_higher' if score2 > score1 else 'equal'
                }
            
            # Calculate overall similarity
            similarities = [comp['similarity'] for comp in dimension_comparison.values()]
            overall_similarity = np.mean(similarities)
            
            # Compare categorical attributes
            categorical_comparison = {
                'risk_tolerance': {
                    'user1': profile1.risk_tolerance.value,
                    'user2': profile2.risk_tolerance.value,
                    'match': profile1.risk_tolerance == profile2.risk_tolerance
                },
                'investment_style': {
                    'user1': profile1.investment_style.value,
                    'user2': profile2.investment_style.value,
                    'match': profile1.investment_style == profile2.investment_style
                },
                'trading_frequency': {
                    'user1': profile1.trading_frequency.value,
                    'user2': profile2.trading_frequency.value,
                    'match': profile1.trading_frequency == profile2.trading_frequency
                }
            }
            
            # Identify key differences
            key_differences = []
            for dimension, comp in dimension_comparison.items():
                if comp['difference'] > 0.3:  # Significant difference
                    key_differences.append({
                        'dimension': dimension,
                        'difference': comp['difference'],
                        'description': f"Significant difference in {dimension.replace('_', ' ')}"
                    })
            
            comparison_result = {
                'user1_id': user_id1,
                'user2_id': user_id2,
                'overall_similarity': overall_similarity,
                'dimension_comparison': dimension_comparison,
                'categorical_comparison': categorical_comparison,
                'key_differences': key_differences,
                'compatibility_score': overall_similarity,
                'comparison_timestamp': datetime.now()
            }
            
            return comparison_result
            
        except Exception as e:
            logger.error(f"Error comparing profiles: {e}")
            return {}
    
    async def _apply_initial_data(self, profile_scores: Dict[ProfileDimension, ProfileScore],
                                initial_data: Dict[str, Any]):
        """Apply initial data to profile scores"""
        try:
            # Map initial data to profile dimensions
            if 'risk_tolerance' in initial_data:
                risk_score = self._risk_tolerance_to_score(initial_data['risk_tolerance'])
                profile_scores[ProfileDimension.RISK_TOLERANCE].score = risk_score
                profile_scores[ProfileDimension.RISK_TOLERANCE].confidence = 0.7
            
            if 'investment_experience' in initial_data:
                exp_score = initial_data['investment_experience'] / 10.0  # Assume 0-10 scale
                profile_scores[ProfileDimension.TECHNICAL_SOPHISTICATION].score = exp_score
                profile_scores[ProfileDimension.TECHNICAL_SOPHISTICATION].confidence = 0.6
            
            if 'trading_frequency' in initial_data:
                freq_score = self._trading_frequency_to_score(initial_data['trading_frequency'])
                profile_scores[ProfileDimension.TRADING_FREQUENCY].score = freq_score
                profile_scores[ProfileDimension.TRADING_FREQUENCY].confidence = 0.8
            
        except Exception as e:
            logger.error(f"Error applying initial data: {e}")
    
    def _risk_tolerance_to_score(self, risk_tolerance: str) -> float:
        """Convert risk tolerance string to score"""
        mapping = {
            'very_conservative': 0.1,
            'conservative': 0.3,
            'moderate': 0.5,
            'aggressive': 0.7,
            'very_aggressive': 0.9
        }
        return mapping.get(risk_tolerance.lower(), 0.5)
    
    def _trading_frequency_to_score(self, frequency: str) -> float:
        """Convert trading frequency to score"""
        mapping = {
            'hodler': 0.1,
            'long_term': 0.3,
            'medium_term': 0.5,
            'active': 0.7,
            'high_frequency': 0.9
        }
        return mapping.get(frequency.lower(), 0.3)
    
    async def _update_profile_scores(self, profile: UserProfile, behavior_data: Dict[str, Any]):
        """Update profile scores based on behavior data"""
        try:
            # Update based on trading behavior
            if 'trades' in behavior_data:
                await self._update_from_trading_behavior(profile, behavior_data['trades'])
            
            # Update based on portfolio changes
            if 'portfolio_changes' in behavior_data:
                await self._update_from_portfolio_behavior(profile, behavior_data['portfolio_changes'])
            
            # Update based on research activity
            if 'research_activity' in behavior_data:
                await self._update_from_research_behavior(profile, behavior_data['research_activity'])
            
            # Update based on platform usage
            if 'platform_usage' in behavior_data:
                await self._update_from_platform_behavior(profile, behavior_data['platform_usage'])
            
        except Exception as e:
            logger.error(f"Error updating profile scores: {e}")
    
    async def _update_from_trading_behavior(self, profile: UserProfile, trades: List[Dict[str, Any]]):
        """Update profile based on trading behavior"""
        try:
            if not trades:
                return
            
            # Analyze trading frequency
            trade_count = len(trades)
            time_span = (datetime.now() - datetime.fromisoformat(trades[0]['timestamp'])).days
            if time_span > 0:
                daily_trade_rate = trade_count / time_span
                freq_score = min(1.0, daily_trade_rate / 5.0)  # Normalize to 0-1
                
                self._update_score(profile, ProfileDimension.TRADING_FREQUENCY, freq_score, 0.1)
            
            # Analyze risk tolerance from position sizes
            position_sizes = [trade.get('position_size_pct', 0) for trade in trades]
            if position_sizes:
                avg_position_size = np.mean(position_sizes)
                risk_score = min(1.0, avg_position_size / 20.0)  # Normalize assuming 20% is high risk
                
                self._update_score(profile, ProfileDimension.RISK_TOLERANCE, risk_score, 0.1)
            
            # Analyze investment style from asset types
            asset_types = [trade.get('asset_type', '') for trade in trades]
            style_indicators = Counter(asset_types)
            
            if 'crypto' in style_indicators and style_indicators['crypto'] > len(trades) * 0.5:
                self._update_score(profile, ProfileDimension.RISK_TOLERANCE, 0.8, 0.05)
            
        except Exception as e:
            logger.error(f"Error updating from trading behavior: {e}")
    
    async def _update_from_portfolio_behavior(self, profile: UserProfile, portfolio_changes: List[Dict[str, Any]]):
        """Update profile based on portfolio changes"""
        try:
            if not portfolio_changes:
                return
            
            # Analyze diversification
            asset_count = len(set(change.get('asset', '') for change in portfolio_changes))
            diversification_score = min(1.0, asset_count / 20.0)  # Normalize to 0-1
            
            self._update_score(profile, ProfileDimension.RISK_TOLERANCE, 
                             1 - diversification_score * 0.3, 0.05)  # More diversification = lower risk
            
            # Analyze rebalancing frequency
            rebalance_count = sum(1 for change in portfolio_changes if change.get('type') == 'rebalance')
            if len(portfolio_changes) > 0:
                rebalance_ratio = rebalance_count / len(portfolio_changes)
                sophistication_score = min(1.0, rebalance_ratio * 2.0)
                
                self._update_score(profile, ProfileDimension.TECHNICAL_SOPHISTICATION, 
                                 sophistication_score, 0.05)
            
        except Exception as e:
            logger.error(f"Error updating from portfolio behavior: {e}")
    
    async def _update_from_research_behavior(self, profile: UserProfile, research_activity: Dict[str, Any]):
        """Update profile based on research activity"""
        try:
            # Analyze research depth
            research_time = research_activity.get('total_time_minutes', 0)
            research_score = min(1.0, research_time / 120.0)  # Normalize to 2 hours
            
            self._update_score(profile, ProfileDimension.TECHNICAL_SOPHISTICATION, research_score, 0.05)
            
            # Analyze information sources
            sources = research_activity.get('sources', [])
            if 'technical_analysis' in sources:
                self._update_score(profile, ProfileDimension.TECHNICAL_SOPHISTICATION, 0.8, 0.05)
            
            if 'fundamental_analysis' in sources:
                self._update_score(profile, ProfileDimension.INVESTMENT_STYLE, 0.3, 0.05)  # Value-oriented
            
        except Exception as e:
            logger.error(f"Error updating from research behavior: {e}")
    
    async def _update_from_platform_behavior(self, profile: UserProfile, platform_usage: Dict[str, Any]):
        """Update profile based on platform usage"""
        try:
            # Analyze feature usage
            features_used = platform_usage.get('features_used', [])
            
            if 'advanced_charts' in features_used:
                self._update_score(profile, ProfileDimension.TECHNICAL_SOPHISTICATION, 0.7, 0.03)
            
            if 'options_trading' in features_used:
                self._update_score(profile, ProfileDimension.RISK_TOLERANCE, 0.8, 0.05)
                self._update_score(profile, ProfileDimension.TECHNICAL_SOPHISTICATION, 0.8, 0.05)
            
            if 'social_features' in features_used:
                self._update_score(profile, ProfileDimension.INFORMATION_CONSUMPTION, 0.7, 0.03)
            
            # Analyze session patterns
            session_duration = platform_usage.get('avg_session_duration_minutes', 0)
            engagement_score = min(1.0, session_duration / 60.0)  # Normalize to 1 hour
            
            self._update_score(profile, ProfileDimension.TECHNICAL_SOPHISTICATION, engagement_score, 0.02)
            
        except Exception as e:
            logger.error(f"Error updating from platform behavior: {e}")
    
    def _update_score(self, profile: UserProfile, dimension: ProfileDimension, 
                     new_evidence: float, learning_rate: float):
        """Update a profile score with new evidence"""
        try:
            current_score = profile.profile_scores[dimension]
            
            # Weighted update based on confidence
            weight = learning_rate * (1 - current_score.confidence)
            updated_score = current_score.score * (1 - weight) + new_evidence * weight
            
            # Update confidence (increases with more evidence)
            updated_confidence = min(0.95, current_score.confidence + 0.01)
            
            # Update the score
            current_score.score = max(0.0, min(1.0, updated_score))
            current_score.confidence = updated_confidence
            current_score.evidence_count += 1
            current_score.last_updated = datetime.now()
            
            # Determine trend
            if updated_score > current_score.score + 0.05:
                current_score.trend = "increasing"
            elif updated_score < current_score.score - 0.05:
                current_score.trend = "decreasing"
            else:
                current_score.trend = "stable"
            
        except Exception as e:
            logger.error(f"Error updating score: {e}")
    
    async def _update_derived_attributes(self, profile: UserProfile):
        """Update derived attributes based on profile scores"""
        try:
            # Update risk tolerance
            risk_score = profile.profile_scores[ProfileDimension.RISK_TOLERANCE].score
            if risk_score < 0.2:
                profile.risk_tolerance = RiskTolerance.VERY_CONSERVATIVE
            elif risk_score < 0.4:
                profile.risk_tolerance = RiskTolerance.CONSERVATIVE
            elif risk_score < 0.6:
                profile.risk_tolerance = RiskTolerance.MODERATE
            elif risk_score < 0.8:
                profile.risk_tolerance = RiskTolerance.AGGRESSIVE
            else:
                profile.risk_tolerance = RiskTolerance.VERY_AGGRESSIVE
            
            # Update trading frequency
            freq_score = profile.profile_scores[ProfileDimension.TRADING_FREQUENCY].score
            if freq_score < 0.2:
                profile.trading_frequency = TradingFrequency.HODLER
            elif freq_score < 0.4:
                profile.trading_frequency = TradingFrequency.LONG_TERM
            elif freq_score < 0.6:
                profile.trading_frequency = TradingFrequency.MEDIUM_TERM
            elif freq_score < 0.8:
                profile.trading_frequency = TradingFrequency.ACTIVE
            else:
                profile.trading_frequency = TradingFrequency.HIGH_FREQUENCY
            
            # Update technical sophistication
            tech_score = profile.profile_scores[ProfileDimension.TECHNICAL_SOPHISTICATION].score
            if tech_score < 0.25:
                profile.technical_sophistication = TechnicalSophistication.BEGINNER
            elif tech_score < 0.5:
                profile.technical_sophistication = TechnicalSophistication.INTERMEDIATE
            elif tech_score < 0.75:
                profile.technical_sophistication = TechnicalSophistication.ADVANCED
            elif tech_score < 0.9:
                profile.technical_sophistication = TechnicalSophistication.EXPERT
            else:
                profile.technical_sophistication = TechnicalSophistication.QUANTITATIVE
            
            # Update investment style (simplified logic)
            style_score = profile.profile_scores[ProfileDimension.INVESTMENT_STYLE].score
            if freq_score > 0.7:  # High frequency trader
                if tech_score > 0.8:
                    profile.investment_style = InvestmentStyle.ALGORITHMIC
                else:
                    profile.investment_style = InvestmentStyle.DAY_TRADER
            elif freq_score > 0.5:
                profile.investment_style = InvestmentStyle.SWING_TRADER
            elif style_score < 0.3:
                profile.investment_style = InvestmentStyle.VALUE_INVESTOR
            elif style_score > 0.7:
                profile.investment_style = InvestmentStyle.GROWTH_INVESTOR
            else:
                profile.investment_style = InvestmentStyle.INDEX_INVESTOR
            
        except Exception as e:
            logger.error(f"Error updating derived attributes: {e}")
    
    async def _detect_behavior_patterns(self, profile: UserProfile):
        """Detect behavior patterns from user history"""
        try:
            user_id = profile.user_id
            if user_id not in self.behavior_history:
                return
            
            behavior_data = self.behavior_history[user_id]
            if len(behavior_data) < 5:  # Need minimum data
                return
            
            # Detect trading time patterns
            trading_times = []
            for entry in behavior_data:
                if 'trades' in entry['data']:
                    for trade in entry['data']['trades']:
                        trade_time = datetime.fromisoformat(trade['timestamp'])
                        trading_times.append(trade_time.hour)
            
            if trading_times:
                # Find most common trading hours
                hour_counts = Counter(trading_times)
                most_common_hour = hour_counts.most_common(1)[0][0]
                
                pattern = BehaviorPattern(
                    pattern_type="preferred_trading_time",
                    frequency=hour_counts[most_common_hour] / len(trading_times),
                    strength=0.7,
                    consistency=0.8,
                    trend="stable",
                    last_observed=datetime.now(),
                    confidence=0.6,
                    metadata={"preferred_hour": most_common_hour}
                )
                
                # Update or add pattern
                existing_pattern = next(
                    (p for p in profile.behavior_patterns if p.pattern_type == "preferred_trading_time"),
                    None
                )
                
                if existing_pattern:
                    existing_pattern.frequency = pattern.frequency
                    existing_pattern.last_observed = pattern.last_observed
                else:
                    profile.behavior_patterns.append(pattern)
            
            # Detect volatility preference patterns
            volatility_preferences = []
            for entry in behavior_data:
                if 'portfolio_changes' in entry['data']:
                    for change in entry['data']['portfolio_changes']:
                        if 'volatility' in change:
                            volatility_preferences.append(change['volatility'])
            
            if volatility_preferences:
                avg_volatility_pref = np.mean(volatility_preferences)
                volatility_pattern = BehaviorPattern(
                    pattern_type="volatility_preference",
                    frequency=1.0,
                    strength=min(1.0, avg_volatility_pref),
                    consistency=1 - np.std(volatility_preferences),
                    trend="stable",
                    last_observed=datetime.now(),
                    confidence=0.7,
                    metadata={"avg_volatility_preference": avg_volatility_pref}
                )
                
                # Update or add pattern
                existing_pattern = next(
                    (p for p in profile.behavior_patterns if p.pattern_type == "volatility_preference"),
                    None
                )
                
                if existing_pattern:
                    existing_pattern.strength = volatility_pattern.strength
                    existing_pattern.consistency = volatility_pattern.consistency
                    existing_pattern.last_observed = volatility_pattern.last_observed
                else:
                    profile.behavior_patterns.append(volatility_pattern)
            
        except Exception as e:
            logger.error(f"Error detecting behavior patterns: {e}")
    
    async def _calculate_completeness(self, profile: UserProfile) -> float:
        """Calculate profile completeness score"""
        try:
            completeness_factors = {
                'profile_scores': 0.3,
                'demographics': 0.15,
                'financial_situation': 0.15,
                'goals': 0.1,
                'behavior_patterns': 0.2,
                'evidence_count': 0.1
            }
            
            total_score = 0.0
            
            # Profile scores completeness
            confident_scores = sum(1 for score in profile.profile_scores.values() if score.confidence > 0.5)
            score_completeness = confident_scores / len(ProfileDimension)
            total_score += score_completeness * completeness_factors['profile_scores']
            
            # Demographics completeness
            demo_fields = ['age', 'income', 'experience', 'education']
            demo_completeness = sum(1 for field in demo_fields if field in profile.demographics) / len(demo_fields)
            total_score += demo_completeness * completeness_factors['demographics']
            
            # Financial situation completeness
            fin_fields = ['net_worth', 'investment_capital', 'debt', 'income_stability']
            fin_completeness = sum(1 for field in fin_fields if field in profile.financial_situation) / len(fin_fields)
            total_score += fin_completeness * completeness_factors['financial_situation']
            
            # Goals completeness
            goals_completeness = min(1.0, len(profile.goals) / 3)  # Assume 3 goals is complete
            total_score += goals_completeness * completeness_factors['goals']
            
            # Behavior patterns completeness
            patterns_completeness = min(1.0, len(profile.behavior_patterns) / 5)  # Assume 5 patterns is complete
            total_score += patterns_completeness * completeness_factors['behavior_patterns']
            
            # Evidence count completeness
            total_evidence = sum(score.evidence_count for score in profile.profile_scores.values())
            evidence_completeness = min(1.0, total_evidence / 100)  # Assume 100 evidence points is complete
            total_score += evidence_completeness * completeness_factors['evidence_count']
            
            return min(1.0, total_score)
            
        except Exception as e:
            logger.error(f"Error calculating completeness: {e}")
            return 0.0
    
    def _increment_version(self, current_version: str) -> str:
        """Increment profile version"""
        try:
            parts = current_version.split('.')
            if len(parts) >= 2:
                major, minor = int(parts[0]), int(parts[1])
                return f"{major}.{minor + 1}"
            else:
                return "1.1"
        except:
            return "1.1"
    
    def _serialize_profile(self, profile: UserProfile) -> Dict[str, Any]:
        """Serialize profile for storage"""
        try:
            return {
                'user_id': profile.user_id,
                'profile_scores': {
                    dim.value: {
                        'score': score.score,
                        'confidence': score.confidence,
                        'evidence_count': score.evidence_count,
                        'trend': score.trend
                    }
                    for dim, score in profile.profile_scores.items()
                },
                'risk_tolerance': profile.risk_tolerance.value,
                'investment_style': profile.investment_style.value,
                'trading_frequency': profile.trading_frequency.value,
                'technical_sophistication': profile.technical_sophistication.value,
                'profile_completeness': profile.profile_completeness,
                'last_updated': profile.last_updated.isoformat(),
                'profile_version': profile.profile_version
            }
        except Exception as e:
            logger.error(f"Error serializing profile: {e}")
            return {}
    
    async def _extract_feature_vector(self, profile: UserProfile) -> List[float]:
        """Extract feature vector for ML operations"""
        try:
            features = []
            
            # Profile dimension scores
            for dimension in ProfileDimension:
                score = profile.profile_scores[dimension].score
                confidence = profile.profile_scores[dimension].confidence
                features.extend([score, confidence])
            
            # Categorical features (one-hot encoded)
            risk_features = [0.0] * len(RiskTolerance)
            risk_features[list(RiskTolerance).index(profile.risk_tolerance)] = 1.0
            features.extend(risk_features)
            
            style_features = [0.0] * len(InvestmentStyle)
            style_features[list(InvestmentStyle).index(profile.investment_style)] = 1.0
            features.extend(style_features)
            
            freq_features = [0.0] * len(TradingFrequency)
            freq_features[list(TradingFrequency).index(profile.trading_frequency)] = 1.0
            features.extend(freq_features)
            
            tech_features = [0.0] * len(TechnicalSophistication)
            tech_features[list(TechnicalSophistication).index(profile.technical_sophistication)] = 1.0
            features.extend(tech_features)
            
            # Additional features
            features.append(profile.profile_completeness)
            features.append(len(profile.behavior_patterns))
            features.append(len(profile.goals))
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting feature vector: {e}")
            return [0.0] * 50  # Return default vector
    
    async def _find_similar_users(self, profile: UserProfile, top_k: int = 5) -> List[str]:
        """Find users with similar profiles"""
        try:
            if len(self.user_profiles) < 2:
                return []
            
            target_features = await self._extract_feature_vector(profile)
            similarities = []
            
            for user_id, other_profile in self.user_profiles.items():
                if user_id == profile.user_id:
                    continue
                
                other_features = await self._extract_feature_vector(other_profile)
                
                # Calculate cosine similarity
                similarity = cosine_similarity([target_features], [other_features])[0][0]
                similarities.append((user_id, similarity))
            
            # Sort by similarity and return top k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return [user_id for user_id, sim in similarities[:top_k]]
            
        except Exception as e:
            logger.error(f"Error finding similar users: {e}")
            return []
    
    async def _get_user_cluster(self, profile: UserProfile) -> int:
        """Get cluster ID for user"""
        try:
            return self.profile_clusters.get(profile.user_id, 0)
        except Exception as e:
            logger.error(f"Error getting user cluster: {e}")
            return 0
    
    async def _detect_anomalies(self, profile: UserProfile) -> List[str]:
        """Detect anomalies in user profile"""
        try:
            anomalies = []
            
            # Check for inconsistent risk tolerance
            risk_score = profile.profile_scores[ProfileDimension.RISK_TOLERANCE].score
            freq_score = profile.profile_scores[ProfileDimension.TRADING_FREQUENCY].score
            
            if risk_score < 0.3 and freq_score > 0.7:
                anomalies.append("Low risk tolerance but high trading frequency")
            
            # Check for unusual technical sophistication vs behavior
            tech_score = profile.profile_scores[ProfileDimension.TECHNICAL_SOPHISTICATION].score
            if tech_score > 0.8 and profile.investment_style == InvestmentStyle.INDEX_INVESTOR:
                anomalies.append("High technical sophistication but passive investment style")
            
            # Check for profile completeness vs confidence
            if profile.profile_completeness < 0.3:
                high_confidence_scores = sum(
                    1 for score in profile.profile_scores.values() if score.confidence > 0.8
                )
                if high_confidence_scores > len(ProfileDimension) * 0.5:
                    anomalies.append("High confidence scores despite low profile completeness")
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return []
    
    async def _generate_profile_recommendations(self, profile: UserProfile) -> List[str]:
        """Generate recommendations for profile improvement"""
        try:
            recommendations = []
            
            # Completeness recommendations
            if profile.profile_completeness < 0.5:
                recommendations.append("Complete your profile by providing more demographic and financial information")
            
            # Low confidence recommendations
            low_confidence_dimensions = [
                dim.value for dim, score in profile.profile_scores.items() if score.confidence < 0.5
            ]
            
            if low_confidence_dimensions:
                recommendations.append(
                    f"Increase activity to improve confidence in: {', '.join(low_confidence_dimensions)}"
                )
            
            # Behavior pattern recommendations
            if len(profile.behavior_patterns) < 3:
                recommendations.append("Continue using the platform to establish clearer behavior patterns")
            
            # Risk-specific recommendations
            if profile.risk_tolerance == RiskTolerance.VERY_AGGRESSIVE:
                recommendations.append("Consider diversification strategies to manage high-risk exposure")
            elif profile.risk_tolerance == RiskTolerance.VERY_CONSERVATIVE:
                recommendations.append("Explore moderate-risk opportunities for potential growth")
            
            # Technical sophistication recommendations
            if profile.technical_sophistication == TechnicalSophistication.BEGINNER:
                recommendations.append("Consider educational resources to improve technical analysis skills")
            
            return recommendations[:5]  # Return top 5 recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []
    
    async def _calculate_analysis_confidence(self, profile: UserProfile) -> float:
        """Calculate confidence in profile analysis"""
        try:
            # Base confidence from profile completeness
            base_confidence = profile.profile_completeness
            
            # Adjust for evidence count
            total_evidence = sum(score.evidence_count for score in profile.profile_scores.values())
            evidence_factor = min(1.0, total_evidence / 50.0)
            
            # Adjust for score confidence
            avg_score_confidence = np.mean([score.confidence for score in profile.profile_scores.values()])
            
            # Adjust for behavior patterns
            pattern_factor = min(1.0, len(profile.behavior_patterns) / 5.0)
            
            # Combined confidence
            confidence = (base_confidence * 0.4 + 
                         evidence_factor * 0.3 + 
                         avg_score_confidence * 0.2 + 
                         pattern_factor * 0.1)
            
            return min(1.0, confidence)
            
        except Exception as e:
            logger.error(f"Error calculating analysis confidence: {e}")
            return 0.5
    
    async def _extract_key_insights(self, profile: UserProfile) -> List[str]:
        """Extract key insights from profile"""
        try:
            insights = []
            
            # Risk tolerance insights
            risk_score = profile.profile_scores[ProfileDimension.RISK_TOLERANCE].score
            if risk_score > 0.8:
                insights.append("High risk tolerance suggests comfort with volatile investments")
            elif risk_score < 0.2:
                insights.append("Conservative approach indicates preference for stable investments")
            
            # Trading frequency insights
            freq_score = profile.profile_scores[ProfileDimension.TRADING_FREQUENCY].score
            if freq_score > 0.7:
                insights.append("Active trading pattern suggests short-term investment focus")
            elif freq_score < 0.3:
                insights.append("Long-term holding pattern indicates buy-and-hold strategy")
            
            # Technical sophistication insights
            tech_score = profile.profile_scores[ProfileDimension.TECHNICAL_SOPHISTICATION].score
            if tech_score > 0.8:
                insights.append("High technical sophistication enables advanced trading strategies")
            elif tech_score < 0.3:
                insights.append("Basic technical knowledge suggests need for simplified tools")
            
            # Behavior pattern insights
            for pattern in profile.behavior_patterns:
                if pattern.pattern_type == "preferred_trading_time" and pattern.strength > 0.7:
                    preferred_hour = pattern.metadata.get('preferred_hour', 'unknown')
                    insights.append(f"Consistent trading pattern around {preferred_hour}:00")
            
            return insights[:5]
            
        except Exception as e:
            logger.error(f"Error extracting insights: {e}")
            return []
    
    async def _identify_risk_factors(self, profile: UserProfile) -> List[str]:
        """Identify risk factors in user profile"""
        try:
            risk_factors = []
            
            # High risk tolerance without sophistication
            risk_score = profile.profile_scores[ProfileDimension.RISK_TOLERANCE].score
            tech_score = profile.profile_scores[ProfileDimension.TECHNICAL_SOPHISTICATION].score
            
            if risk_score > 0.7 and tech_score < 0.4:
                risk_factors.append("High risk appetite with limited technical knowledge")
            
            # Frequent trading with low sophistication
            freq_score = profile.profile_scores[ProfileDimension.TRADING_FREQUENCY].score
            if freq_score > 0.7 and tech_score < 0.5:
                risk_factors.append("High trading frequency may lead to overtrading")
            
            # Inconsistent behavior patterns
            inconsistent_patterns = [
                pattern for pattern in profile.behavior_patterns 
                if pattern.consistency < 0.5
            ]
            if len(inconsistent_patterns) > 2:
                risk_factors.append("Inconsistent behavior patterns may indicate emotional trading")
            
            # Low profile completeness with high confidence
            if profile.profile_completeness < 0.4:
                high_confidence_count = sum(
                    1 for score in profile.profile_scores.values() if score.confidence > 0.7
                )
                if high_confidence_count > len(ProfileDimension) * 0.3:
                    risk_factors.append("Overconfidence despite limited profile data")
            
            return risk_factors
            
        except Exception as e:
            logger.error(f"Error identifying risk factors: {e}")
            return []
    
    async def _identify_opportunities(self, profile: UserProfile) -> List[str]:
        """Identify opportunities for user"""
        try:
            opportunities = []
            
            # High sophistication opportunities
            tech_score = profile.profile_scores[ProfileDimension.TECHNICAL_SOPHISTICATION].score
            if tech_score > 0.7:
                opportunities.append("Advanced trading features and strategies")
                opportunities.append("Algorithmic trading tools")
            
            # Conservative investor opportunities
            risk_score = profile.profile_scores[ProfileDimension.RISK_TOLERANCE].score
            if risk_score < 0.4:
                opportunities.append("Stable income-generating investments")
                opportunities.append("Diversified index funds")
            
            # Active trader opportunities
            freq_score = profile.profile_scores[ProfileDimension.TRADING_FREQUENCY].score
            if freq_score > 0.6:
                opportunities.append("Real-time market data and alerts")
                opportunities.append("Advanced order types")
            
            # Learning opportunities
            if tech_score < 0.5:
                opportunities.append("Educational content and tutorials")
                opportunities.append("Paper trading for skill development")
            
            # Social opportunities
            info_score = profile.profile_scores[ProfileDimension.INFORMATION_CONSUMPTION].score
            if info_score > 0.6:
                opportunities.append("Community features and social trading")
                opportunities.append("Expert insights and analysis")
            
            return opportunities[:5]
            
        except Exception as e:
            logger.error(f"Error identifying opportunities: {e}")
            return []
    
    async def _analyze_behavior_evolution(self, user_id: str, time_window: timedelta) -> Dict[str, Any]:
        """Analyze how user behavior has evolved"""
        try:
            if user_id not in self.behavior_history:
                return {}
            
            cutoff_time = datetime.now() - time_window
            relevant_history = [
                entry for entry in self.behavior_history[user_id]
                if entry['timestamp'] >= cutoff_time
            ]
            
            if len(relevant_history) < 2:
                return {'insufficient_data': True}
            
            # Analyze trading frequency evolution
            trading_frequencies = []
            for entry in relevant_history:
                if 'trades' in entry['data']:
                    trade_count = len(entry['data']['trades'])
                    trading_frequencies.append(trade_count)
            
            evolution_analysis = {}
            
            if trading_frequencies:
                freq_trend = np.polyfit(range(len(trading_frequencies)), trading_frequencies, 1)[0]
                evolution_analysis['trading_frequency'] = {
                    'trend': 'increasing' if freq_trend > 0.1 else 'decreasing' if freq_trend < -0.1 else 'stable',
                    'volatility': np.std(trading_frequencies),
                    'average': np.mean(trading_frequencies)
                }
            
            # Analyze research activity evolution
            research_times = []
            for entry in relevant_history:
                if 'research_activity' in entry['data']:
                    research_time = entry['data']['research_activity'].get('total_time_minutes', 0)
                    research_times.append(research_time)
            
            if research_times:
                research_trend = np.polyfit(range(len(research_times)), research_times, 1)[0]
                evolution_analysis['research_activity'] = {
                    'trend': 'increasing' if research_trend > 1 else 'decreasing' if research_trend < -1 else 'stable',
                    'volatility': np.std(research_times),
                    'average': np.mean(research_times)
                }
            
            return evolution_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing behavior evolution: {e}")
            return {}

# Export main classes
__all__ = ['UserProfiler', 'UserProfile', 'ProfileDimension', 'ProfileAnalysis', 'BehaviorPattern',
           'RiskTolerance', 'InvestmentStyle', 'TradingFrequency', 'TechnicalSophistication', 'ProfileScore']