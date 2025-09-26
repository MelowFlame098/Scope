# Behavior Analyzer
# Phase 9: AI-First Platform Implementation

import logging
import asyncio
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque, Counter
import sqlite3
import aiosqlite
from pathlib import Path
import hashlib
import statistics
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import pandas as pd

logger = logging.getLogger(__name__)

class BehaviorType(Enum):
    NAVIGATION = "navigation"          # Page/screen navigation
    INTERACTION = "interaction"        # UI element interactions
    TRADING = "trading"                # Trading activities
    ANALYSIS = "analysis"              # Chart/data analysis
    SEARCH = "search"                  # Search behaviors
    CUSTOMIZATION = "customization"    # UI/preference changes
    LEARNING = "learning"              # Educational content engagement
    SOCIAL = "social"                  # Social features usage
    NOTIFICATION = "notification"      # Notification interactions
    ERROR = "error"                    # Error encounters

class BehaviorContext(Enum):
    DESKTOP = "desktop"                # Desktop application
    MOBILE = "mobile"                  # Mobile application
    WEB = "web"                        # Web browser
    API = "api"                        # API usage
    BACKGROUND = "background"          # Background processes

class SessionType(Enum):
    ACTIVE = "active"                  # Active user session
    PASSIVE = "passive"                # Passive monitoring
    AUTOMATED = "automated"            # Automated/bot activity
    GUEST = "guest"                    # Guest user session

class BehaviorPattern(Enum):
    FREQUENT_USER = "frequent_user"    # High activity user
    CASUAL_USER = "casual_user"        # Low activity user
    POWER_USER = "power_user"          # Advanced feature user
    EXPLORER = "explorer"              # Tries new features
    ROUTINE_USER = "routine_user"      # Consistent patterns
    STRUGGLING_USER = "struggling_user" # High error rates
    LEARNING_USER = "learning_user"    # Educational focus
    SOCIAL_USER = "social_user"        # Social features focus

class EngagementLevel(Enum):
    VERY_LOW = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    VERY_HIGH = 5

class RiskLevel(Enum):
    MINIMAL = 1
    LOW = 2
    MODERATE = 3
    HIGH = 4
    CRITICAL = 5

@dataclass
class BehaviorEvent:
    event_id: str
    user_id: str
    session_id: str
    behavior_type: BehaviorType
    action: str
    context: BehaviorContext
    timestamp: datetime
    duration: Optional[float] = None  # Duration in seconds
    metadata: Dict[str, Any] = field(default_factory=dict)
    location: Optional[str] = None    # Page/screen location
    element: Optional[str] = None     # UI element identifier
    value: Optional[Any] = None       # Associated value
    success: bool = True              # Whether action was successful
    error_code: Optional[str] = None  # Error code if failed
    device_info: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

@dataclass
class UserSession:
    session_id: str
    user_id: str
    session_type: SessionType
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    context: BehaviorContext = BehaviorContext.WEB
    events_count: int = 0
    unique_actions: Set[str] = field(default_factory=set)
    pages_visited: Set[str] = field(default_factory=set)
    features_used: Set[str] = field(default_factory=set)
    errors_encountered: int = 0
    successful_actions: int = 0
    device_info: Dict[str, Any] = field(default_factory=dict)
    location_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BehaviorMetrics:
    user_id: str
    period_start: datetime
    period_end: datetime
    total_sessions: int = 0
    total_events: int = 0
    total_duration: float = 0.0
    avg_session_duration: float = 0.0
    unique_actions: int = 0
    unique_pages: int = 0
    unique_features: int = 0
    error_rate: float = 0.0
    success_rate: float = 0.0
    engagement_score: float = 0.0
    activity_score: float = 0.0
    consistency_score: float = 0.0
    learning_score: float = 0.0
    social_score: float = 0.0
    behavior_patterns: List[BehaviorPattern] = field(default_factory=list)
    engagement_level: EngagementLevel = EngagementLevel.MEDIUM
    risk_indicators: Dict[str, float] = field(default_factory=dict)
    anomaly_score: float = 0.0
    trend_indicators: Dict[str, float] = field(default_factory=dict)

@dataclass
class BehaviorInsight:
    insight_id: str
    user_id: str
    insight_type: str
    title: str
    description: str
    confidence: float
    importance: float
    category: str
    recommendations: List[str] = field(default_factory=list)
    supporting_data: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    acted_upon: bool = False
    feedback_score: Optional[float] = None

@dataclass
class BehaviorPrediction:
    prediction_id: str
    user_id: str
    prediction_type: str
    predicted_behavior: str
    probability: float
    confidence_interval: Tuple[float, float]
    time_horizon: timedelta
    factors: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(days=7))
    actual_outcome: Optional[bool] = None
    accuracy_score: Optional[float] = None

@dataclass
class BehaviorCluster:
    cluster_id: str
    cluster_name: str
    description: str
    user_count: int
    characteristics: Dict[str, Any]
    representative_users: List[str] = field(default_factory=list)
    behavior_patterns: List[BehaviorPattern] = field(default_factory=list)
    engagement_profile: Dict[str, float] = field(default_factory=dict)
    risk_profile: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

class BehaviorAnalyzer:
    """Advanced behavior analysis and pattern recognition system"""
    
    def __init__(self, db_path: str = "behavior.db"):
        self.db_path = db_path
        
        # Event storage and processing
        self.event_buffer = deque(maxlen=10000)  # Recent events buffer
        self.session_cache = {}  # session_id -> UserSession
        self.user_metrics_cache = {}  # user_id -> BehaviorMetrics
        
        # Pattern recognition
        self.behavior_patterns = {}  # user_id -> List[BehaviorPattern]
        self.pattern_models = {}  # pattern_type -> ML model
        self.anomaly_detectors = {}  # context -> anomaly detection model
        
        # Clustering and segmentation
        self.user_clusters = {}  # cluster_id -> BehaviorCluster
        self.user_cluster_assignments = {}  # user_id -> cluster_id
        self.clustering_model = None
        
        # Insights and predictions
        self.insights_cache = defaultdict(list)  # user_id -> List[BehaviorInsight]
        self.predictions_cache = defaultdict(list)  # user_id -> List[BehaviorPrediction]
        self.insight_generators = []  # List of insight generation functions
        
        # Real-time processing
        self.event_processors = []  # List of event processing functions
        self.real_time_alerts = defaultdict(list)  # user_id -> List[alert]
        self.processing_queue = asyncio.Queue()
        
        # Analytics and reporting
        self.analytics_cache = {}  # Cached analytics results
        self.report_generators = {}  # report_type -> generator function
        
        # Configuration
        self.config = {
            'session_timeout': 1800,  # 30 minutes
            'anomaly_threshold': 0.1,
            'insight_retention_days': 30,
            'prediction_retention_days': 7,
            'clustering_update_interval': 86400,  # 24 hours
            'real_time_processing': True
        }
        
        logger.info("Behavior analyzer initialized")
    
    async def initialize(self):
        """Initialize the behavior analyzer"""
        try:
            await self._create_database_schema()
            await self._load_existing_data()
            await self._initialize_models()
            
            # Start real-time processing if enabled
            if self.config['real_time_processing']:
                asyncio.create_task(self._process_events_continuously())
            
            logger.info("Behavior analyzer initialization completed")
            
        except Exception as e:
            logger.error(f"Error initializing behavior analyzer: {e}")
            raise
    
    async def track_event(self, event: BehaviorEvent) -> bool:
        """Track a user behavior event"""
        try:
            # Validate event
            if not await self._validate_event(event):
                logger.warning(f"Invalid event: {event.event_id}")
                return False
            
            # Add to buffer
            self.event_buffer.append(event)
            
            # Add to processing queue for real-time analysis
            if self.config['real_time_processing']:
                await self.processing_queue.put(event)
            
            # Update session
            await self._update_session(event)
            
            # Store in database
            await self._store_event(event)
            
            # Process event through registered processors
            for processor in self.event_processors:
                try:
                    await processor(event)
                except Exception as e:
                    logger.error(f"Error in event processor: {e}")
            
            logger.debug(f"Tracked event: {event.action} for user {event.user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error tracking event: {e}")
            return False
    
    async def analyze_user_behavior(self, user_id: str, 
                                  period_days: int = 30) -> BehaviorMetrics:
        """Analyze user behavior over a specified period"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=period_days)
            
            # Check cache first
            cache_key = f"{user_id}_{period_days}_{end_time.date()}"
            if cache_key in self.user_metrics_cache:
                return self.user_metrics_cache[cache_key]
            
            # Get user events
            events = await self._get_user_events(user_id, start_time, end_time)
            sessions = await self._get_user_sessions(user_id, start_time, end_time)
            
            # Calculate metrics
            metrics = await self._calculate_behavior_metrics(user_id, events, sessions, start_time, end_time)
            
            # Detect patterns
            patterns = await self._detect_behavior_patterns(user_id, events, sessions)
            metrics.behavior_patterns = patterns
            
            # Calculate engagement level
            metrics.engagement_level = await self._calculate_engagement_level(metrics)
            
            # Detect anomalies
            metrics.anomaly_score = await self._detect_anomalies(user_id, events)
            
            # Calculate trend indicators
            metrics.trend_indicators = await self._calculate_trend_indicators(user_id, events)
            
            # Cache results
            self.user_metrics_cache[cache_key] = metrics
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing user behavior for {user_id}: {e}")
            return BehaviorMetrics(user_id=user_id, period_start=start_time, period_end=end_time)
    
    async def generate_insights(self, user_id: str) -> List[BehaviorInsight]:
        """Generate behavioral insights for a user"""
        try:
            # Check cache first
            if user_id in self.insights_cache:
                # Filter out expired insights
                current_time = datetime.now()
                valid_insights = [
                    insight for insight in self.insights_cache[user_id]
                    if not insight.expires_at or insight.expires_at > current_time
                ]
                if valid_insights:
                    return valid_insights
            
            insights = []
            
            # Get user behavior metrics
            metrics = await self.analyze_user_behavior(user_id)
            
            # Generate insights using registered generators
            for generator in self.insight_generators:
                try:
                    generated_insights = await generator(user_id, metrics)
                    insights.extend(generated_insights)
                except Exception as e:
                    logger.error(f"Error in insight generator: {e}")
            
            # Built-in insight generation
            builtin_insights = await self._generate_builtin_insights(user_id, metrics)
            insights.extend(builtin_insights)
            
            # Sort by importance and confidence
            insights.sort(key=lambda x: (x.importance * x.confidence), reverse=True)
            
            # Cache insights
            self.insights_cache[user_id] = insights
            
            # Store in database
            for insight in insights:
                await self._store_insight(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights for user {user_id}: {e}")
            return []
    
    async def _store_insight(self, insight: BehaviorInsight):
        """Store insight in database"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO behavior_insights (
                        insight_id, user_id, insight_type, title, description,
                        confidence, importance, category, recommendations,
                        supporting_data, created_at, expires_at, acted_upon, feedback_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    insight.insight_id,
                    insight.user_id,
                    insight.insight_type,
                    insight.title,
                    insight.description,
                    insight.confidence,
                    insight.importance,
                    insight.category,
                    json.dumps(insight.recommendations),
                    json.dumps(insight.supporting_data),
                    insight.created_at.isoformat(),
                    insight.expires_at.isoformat() if insight.expires_at else None,
                    insight.acted_upon,
                    insight.feedback_score
                ))
                await db.commit()
        except Exception as e:
            logger.error(f"Error storing insight {insight.insight_id}: {e}")
    
    async def _get_recent_events(self, user_id: str, hours: int = 24) -> List[BehaviorEvent]:
        """Get recent events for a user"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            return await self._get_user_events(user_id, cutoff_time, datetime.now())
        except Exception as e:
            logger.error(f"Error getting recent events: {e}")
            return []
    
    # Additional helper methods for clustering and analytics
    async def _get_active_users(self, min_activity_days: int) -> List[str]:
        """Get list of active users"""
        try:
            cutoff_time = datetime.now() - timedelta(days=min_activity_days)
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute("""
                    SELECT DISTINCT user_id FROM behavior_events 
                    WHERE timestamp > ?
                """, (cutoff_time.isoformat(),)) as cursor:
                    return [row[0] async for row in cursor]
        except Exception as e:
            logger.error(f"Error getting active users: {e}")
            return []
    
    async def _extract_clustering_features(self, metrics: BehaviorMetrics) -> List[float]:
        """Extract features for user clustering"""
        try:
            features = [
                metrics.engagement_score,
                metrics.activity_score,
                metrics.consistency_score,
                metrics.learning_score,
                metrics.social_score,
                metrics.error_rate,
                metrics.avg_session_duration / 3600,  # Normalize to hours
                len(metrics.behavior_patterns),
                metrics.unique_features / 100,  # Normalize
                metrics.total_events / 1000  # Normalize
            ]
            return features
        except Exception as e:
            logger.error(f"Error extracting clustering features: {e}")
            return [0.0] * 10
    
    async def _create_behavior_cluster(self, cluster_id: int, cluster_users: List[str]) -> BehaviorCluster:
        """Create a behavior cluster from user list"""
        try:
            # Analyze cluster characteristics
            user_metrics = []
            for user_id in cluster_users[:10]:  # Sample for performance
                metrics = await self.analyze_user_behavior(user_id)
                user_metrics.append(metrics)
            
            if not user_metrics:
                raise ValueError("No user metrics available")
            
            # Calculate cluster characteristics
            avg_engagement = statistics.mean(m.engagement_score for m in user_metrics)
            avg_activity = statistics.mean(m.activity_score for m in user_metrics)
            common_patterns = Counter()
            for metrics in user_metrics:
                common_patterns.update(metrics.behavior_patterns)
            
            # Determine cluster name and description
            if avg_engagement > 0.7:
                cluster_name = "High Engagement Users"
                description = "Users with high engagement and activity levels"
            elif avg_activity > 0.6:
                cluster_name = "Active Users"
                description = "Users with moderate to high activity"
            else:
                cluster_name = "Casual Users"
                description = "Users with lower engagement and activity"
            
            cluster = BehaviorCluster(
                cluster_id=f"cluster_{cluster_id}",
                cluster_name=cluster_name,
                description=description,
                user_count=len(cluster_users),
                characteristics={
                    "avg_engagement": avg_engagement,
                    "avg_activity": avg_activity,
                    "common_patterns": [p.value for p, _ in common_patterns.most_common(3)]
                },
                representative_users=cluster_users[:5],
                behavior_patterns=[p for p, _ in common_patterns.most_common(3)],
                engagement_profile={"average": avg_engagement, "distribution": "normal"},
                risk_profile={"churn_risk": 1 - avg_engagement}
            )
            
            return cluster
            
        except Exception as e:
            logger.error(f"Error creating behavior cluster: {e}")
            return BehaviorCluster(
                cluster_id=f"cluster_{cluster_id}",
                cluster_name="Unknown Cluster",
                description="Cluster analysis failed",
                user_count=len(cluster_users),
                characteristics={}
            )
    
    async def _get_clustering_feature_names(self) -> List[str]:
        """Get feature names for clustering"""
        return [
            "engagement_score", "activity_score", "consistency_score",
            "learning_score", "social_score", "error_rate",
            "avg_session_hours", "pattern_count", "feature_diversity", "event_volume"
        ]
    
    async def _store_clusters(self, clusters: Dict[str, BehaviorCluster]):
        """Store clusters in database"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                for cluster in clusters.values():
                    await db.execute("""
                        INSERT OR REPLACE INTO behavior_clusters (
                            cluster_id, cluster_name, description, user_count,
                            characteristics, representative_users, behavior_patterns,
                            engagement_profile, risk_profile, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        cluster.cluster_id,
                        cluster.cluster_name,
                        cluster.description,
                        cluster.user_count,
                        json.dumps(cluster.characteristics),
                        json.dumps(cluster.representative_users),
                        json.dumps([p.value for p in cluster.behavior_patterns]),
                        json.dumps(cluster.engagement_profile),
                        json.dumps(cluster.risk_profile),
                        cluster.created_at.isoformat(),
                        cluster.updated_at.isoformat()
                    ))
                await db.commit()
        except Exception as e:
            logger.error(f"Error storing clusters: {e}")
    
    async def _calculate_action_consistency(self, events: List[BehaviorEvent]) -> float:
        """Calculate consistency of user actions"""
        try:
            if not events:
                return 0.0
            
            action_counts = Counter(e.action for e in events)
            total_actions = len(events)
            
            # Calculate entropy (lower entropy = higher consistency)
            entropy = 0
            for count in action_counts.values():
                probability = count / total_actions
                if probability > 0:
                    entropy -= probability * np.log2(probability)
            
            # Normalize entropy to 0-1 scale (1 = most consistent)
            max_entropy = np.log2(len(action_counts)) if len(action_counts) > 1 else 1
            consistency = 1 - (entropy / max_entropy) if max_entropy > 0 else 1
            
            return consistency
            
        except Exception as e:
            logger.error(f"Error calculating action consistency: {e}")
            return 0.0


# Export classes and functions
__all__ = [
    'BehaviorType',
    'BehaviorContext', 
    'SessionType',
    'BehaviorPattern',
    'EngagementLevel',
    'RiskLevel',
    'BehaviorEvent',
    'UserSession',
    'BehaviorMetrics',
    'BehaviorInsight',
    'BehaviorPrediction',
    'BehaviorCluster',
    'BehaviorAnalyzer'
]