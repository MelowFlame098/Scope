# Interface Customizer
# Phase 9: AI-First Platform Implementation

import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import numpy as np

logger = logging.getLogger(__name__)

class InterfaceElement(Enum):
    DASHBOARD_LAYOUT = "dashboard_layout"
    CHART_TYPE = "chart_type"
    COLOR_SCHEME = "color_scheme"
    NAVIGATION_STYLE = "navigation_style"
    WIDGET_PLACEMENT = "widget_placement"
    DATA_DENSITY = "data_density"
    NOTIFICATION_STYLE = "notification_style"
    TOOLBAR_CONFIGURATION = "toolbar_configuration"
    MENU_ORGANIZATION = "menu_organization"
    FONT_SIZE = "font_size"
    ANIMATION_SPEED = "animation_speed"
    KEYBOARD_SHORTCUTS = "keyboard_shortcuts"

class CustomizationScope(Enum):
    GLOBAL = "global"          # Affects entire application
    MODULE = "module"          # Affects specific module
    VIEW = "view"              # Affects specific view/page
    WIDGET = "widget"          # Affects specific widget
    ELEMENT = "element"        # Affects specific UI element

class AdaptationTrigger(Enum):
    USER_BEHAVIOR = "user_behavior"        # Based on usage patterns
    EXPLICIT_PREFERENCE = "explicit_preference"  # User explicitly sets
    PERFORMANCE_OPTIMIZATION = "performance_optimization"  # System optimization
    ACCESSIBILITY_NEED = "accessibility_need"  # Accessibility requirements
    CONTEXT_CHANGE = "context_change"      # Context-aware adaptation
    A_B_TEST = "a_b_test"                  # A/B testing results
    MACHINE_LEARNING = "machine_learning"  # ML-driven adaptation

class CustomizationPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class InterfacePreference:
    element: InterfaceElement
    value: Any
    scope: CustomizationScope
    priority: CustomizationPriority
    trigger: AdaptationTrigger
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CustomizationRule:
    rule_id: str
    condition: str  # JSON-serializable condition
    action: Dict[str, Any]  # Customization to apply
    priority: CustomizationPriority
    scope: CustomizationScope
    active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_applied: Optional[datetime] = None
    application_count: int = 0
    success_rate: float = 1.0

@dataclass
class InterfaceContext:
    user_id: str
    session_id: str
    device_type: str  # desktop, mobile, tablet
    screen_resolution: Tuple[int, int]
    browser_info: Dict[str, str]
    current_module: str
    current_view: str
    time_of_day: str
    user_role: str
    accessibility_needs: List[str] = field(default_factory=list)
    performance_constraints: Dict[str, Any] = field(default_factory=dict)
    network_conditions: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CustomizationResult:
    customization_id: str
    applied_preferences: List[InterfacePreference]
    interface_config: Dict[str, Any]
    performance_impact: Dict[str, float]
    user_satisfaction_score: Optional[float] = None
    application_timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class UsagePattern:
    pattern_id: str
    user_id: str
    element: InterfaceElement
    interaction_type: str  # click, hover, scroll, etc.
    frequency: int
    duration: float  # seconds
    success_rate: float
    context: InterfaceContext
    timestamp: datetime = field(default_factory=datetime.now)

class InterfaceCustomizer:
    """Advanced interface customization and adaptation engine"""
    
    def __init__(self):
        # User preferences storage
        self.user_preferences = defaultdict(list)  # user_id -> [InterfacePreference]
        self.global_preferences = {}  # element -> default_value
        
        # Customization rules
        self.customization_rules = {}  # rule_id -> CustomizationRule
        self.active_rules = defaultdict(list)  # scope -> [rule_id]
        
        # Usage tracking
        self.usage_patterns = defaultdict(list)  # user_id -> [UsagePattern]
        self.interaction_history = defaultdict(deque)  # user_id -> interaction_queue
        
        # A/B testing
        self.ab_tests = {}  # test_id -> test_config
        self.ab_assignments = {}  # user_id -> {test_id: variant}
        
        # Performance tracking
        self.customization_performance = {}  # customization_id -> performance_metrics
        self.adaptation_history = defaultdict(list)  # user_id -> [CustomizationResult]
        
        # Machine learning models for prediction
        self.preference_predictors = {}  # element -> ML_model
        self.satisfaction_predictors = {}  # context -> ML_model
        
        # Default interface configurations
        self.default_configs = self._initialize_default_configs()
        
        logger.info("Interface customizer initialized")
    
    async def customize_interface(self, user_id: str, context: InterfaceContext) -> CustomizationResult:
        """Generate customized interface configuration for user"""
        try:
            customization_id = f"custom_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Get user preferences
            user_prefs = await self._get_user_preferences(user_id, context)
            
            # Apply customization rules
            rule_based_prefs = await self._apply_customization_rules(user_id, context)
            
            # Predict optimal preferences using ML
            predicted_prefs = await self._predict_preferences(user_id, context)
            
            # Merge all preferences with priority resolution
            merged_prefs = await self._merge_preferences(user_prefs, rule_based_prefs, predicted_prefs)
            
            # Generate interface configuration
            interface_config = await self._generate_interface_config(merged_prefs, context)
            
            # Estimate performance impact
            performance_impact = await self._estimate_performance_impact(interface_config, context)
            
            # Create customization result
            result = CustomizationResult(
                customization_id=customization_id,
                applied_preferences=merged_prefs,
                interface_config=interface_config,
                performance_impact=performance_impact
            )
            
            # Store adaptation history
            self.adaptation_history[user_id].append(result)
            
            logger.info(f"Generated interface customization for user {user_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error customizing interface for user {user_id}: {e}")
            return CustomizationResult(
                customization_id="error",
                applied_preferences=[],
                interface_config=self.default_configs.copy(),
                performance_impact={}
            )
    
    async def track_user_interaction(self, user_id: str, element: InterfaceElement,
                                   interaction_type: str, context: InterfaceContext,
                                   duration: float = 0.0, success: bool = True):
        """Track user interaction with interface elements"""
        try:
            # Create usage pattern
            pattern = UsagePattern(
                pattern_id=f"pattern_{user_id}_{len(self.usage_patterns[user_id])}",
                user_id=user_id,
                element=element,
                interaction_type=interaction_type,
                frequency=1,
                duration=duration,
                success_rate=1.0 if success else 0.0,
                context=context
            )
            
            # Store pattern
            self.usage_patterns[user_id].append(pattern)
            
            # Add to interaction history (limited size)
            self.interaction_history[user_id].append({
                'element': element,
                'interaction': interaction_type,
                'timestamp': datetime.now(),
                'success': success,
                'duration': duration
            })
            
            # Analyze patterns for adaptive learning
            await self._analyze_usage_patterns(user_id)
            
            # Check if adaptation is needed
            await self._check_adaptation_triggers(user_id, element, context)
            
        except Exception as e:
            logger.error(f"Error tracking interaction for user {user_id}: {e}")
    
    async def set_user_preference(self, user_id: str, element: InterfaceElement,
                                value: Any, scope: CustomizationScope = CustomizationScope.GLOBAL,
                                priority: CustomizationPriority = CustomizationPriority.HIGH):
        """Set explicit user preference"""
        try:
            preference = InterfacePreference(
                element=element,
                value=value,
                scope=scope,
                priority=priority,
                trigger=AdaptationTrigger.EXPLICIT_PREFERENCE,
                confidence=1.0
            )
            
            # Remove existing preferences for same element and scope
            existing_prefs = self.user_preferences[user_id]
            self.user_preferences[user_id] = [
                p for p in existing_prefs 
                if not (p.element == element and p.scope == scope)
            ]
            
            # Add new preference
            self.user_preferences[user_id].append(preference)
            
            logger.info(f"Set preference for user {user_id}: {element.value} = {value}")
            
        except Exception as e:
            logger.error(f"Error setting user preference: {e}")
    
    async def create_customization_rule(self, rule_id: str, condition: str,
                                      action: Dict[str, Any], priority: CustomizationPriority,
                                      scope: CustomizationScope) -> bool:
        """Create a new customization rule"""
        try:
            rule = CustomizationRule(
                rule_id=rule_id,
                condition=condition,
                action=action,
                priority=priority,
                scope=scope
            )
            
            self.customization_rules[rule_id] = rule
            self.active_rules[scope].append(rule_id)
            
            logger.info(f"Created customization rule: {rule_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating customization rule {rule_id}: {e}")
            return False
    
    async def start_ab_test(self, test_id: str, element: InterfaceElement,
                          variants: Dict[str, Any], traffic_split: Dict[str, float],
                          success_metric: str) -> bool:
        """Start A/B test for interface element"""
        try:
            test_config = {
                'test_id': test_id,
                'element': element,
                'variants': variants,
                'traffic_split': traffic_split,
                'success_metric': success_metric,
                'start_time': datetime.now(),
                'active': True,
                'results': defaultdict(list)
            }
            
            self.ab_tests[test_id] = test_config
            
            logger.info(f"Started A/B test: {test_id} for {element.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting A/B test {test_id}: {e}")
            return False
    
    async def get_ab_variant(self, user_id: str, test_id: str) -> Optional[str]:
        """Get A/B test variant for user"""
        try:
            if test_id not in self.ab_tests:
                return None
            
            test_config = self.ab_tests[test_id]
            if not test_config['active']:
                return None
            
            # Check if user already assigned
            if user_id in self.ab_assignments and test_id in self.ab_assignments[user_id]:
                return self.ab_assignments[user_id][test_id]
            
            # Assign user to variant based on traffic split
            import random
            random.seed(hash(user_id + test_id))  # Consistent assignment
            
            rand_val = random.random()
            cumulative = 0.0
            
            for variant, split in test_config['traffic_split'].items():
                cumulative += split
                if rand_val <= cumulative:
                    # Assign user to variant
                    if user_id not in self.ab_assignments:
                        self.ab_assignments[user_id] = {}
                    self.ab_assignments[user_id][test_id] = variant
                    return variant
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting A/B variant for user {user_id}, test {test_id}: {e}")
            return None
    
    async def record_ab_result(self, user_id: str, test_id: str, metric_value: float):
        """Record A/B test result"""
        try:
            if test_id not in self.ab_tests:
                return
            
            variant = await self.get_ab_variant(user_id, test_id)
            if variant:
                self.ab_tests[test_id]['results'][variant].append({
                    'user_id': user_id,
                    'metric_value': metric_value,
                    'timestamp': datetime.now()
                })
            
        except Exception as e:
            logger.error(f"Error recording A/B result: {e}")
    
    async def analyze_ab_test(self, test_id: str) -> Dict[str, Any]:
        """Analyze A/B test results"""
        try:
            if test_id not in self.ab_tests:
                return {'error': 'Test not found'}
            
            test_config = self.ab_tests[test_id]
            results = test_config['results']
            
            analysis = {
                'test_id': test_id,
                'element': test_config['element'].value,
                'start_time': test_config['start_time'].isoformat(),
                'variants': {}
            }
            
            for variant, data in results.items():
                if data:
                    values = [d['metric_value'] for d in data]
                    analysis['variants'][variant] = {
                        'sample_size': len(values),
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
            
            # Statistical significance test (simplified)
            if len(analysis['variants']) == 2:
                variants = list(analysis['variants'].keys())
                if (analysis['variants'][variants[0]]['sample_size'] > 30 and 
                    analysis['variants'][variants[1]]['sample_size'] > 30):
                    
                    mean_diff = (analysis['variants'][variants[0]]['mean'] - 
                               analysis['variants'][variants[1]]['mean'])
                    analysis['mean_difference'] = mean_diff
                    analysis['statistical_significance'] = abs(mean_diff) > 0.05  # Simplified
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing A/B test {test_id}: {e}")
            return {'error': 'Analysis failed'}
    
    async def get_personalization_insights(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive personalization insights for user"""
        try:
            insights = {
                'user_id': user_id,
                'total_preferences': len(self.user_preferences[user_id]),
                'total_interactions': len(self.usage_patterns[user_id]),
                'adaptation_history': len(self.adaptation_history[user_id]),
                'preference_breakdown': {},
                'usage_statistics': {},
                'adaptation_effectiveness': {},
                'recommendations': []
            }
            
            # Preference breakdown
            prefs_by_element = defaultdict(int)
            for pref in self.user_preferences[user_id]:
                prefs_by_element[pref.element.value] += 1
            insights['preference_breakdown'] = dict(prefs_by_element)
            
            # Usage statistics
            if self.usage_patterns[user_id]:
                patterns = self.usage_patterns[user_id]
                total_duration = sum(p.duration for p in patterns)
                avg_success_rate = np.mean([p.success_rate for p in patterns])
                
                insights['usage_statistics'] = {
                    'total_interactions': len(patterns),
                    'total_time_spent': total_duration,
                    'average_success_rate': avg_success_rate,
                    'most_used_elements': await self._get_most_used_elements(user_id)
                }
            
            # Adaptation effectiveness
            if self.adaptation_history[user_id]:
                recent_adaptations = self.adaptation_history[user_id][-5:]
                avg_satisfaction = np.mean([
                    a.user_satisfaction_score for a in recent_adaptations 
                    if a.user_satisfaction_score is not None
                ])
                
                insights['adaptation_effectiveness'] = {
                    'total_adaptations': len(self.adaptation_history[user_id]),
                    'recent_satisfaction': avg_satisfaction if not np.isnan(avg_satisfaction) else None,
                    'adaptation_frequency': len(recent_adaptations) / 30  # per day
                }
            
            # Generate recommendations
            insights['recommendations'] = await self._generate_personalization_recommendations(user_id)
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting personalization insights for user {user_id}: {e}")
            return {'error': 'Failed to generate insights'}
    
    async def export_user_customizations(self, user_id: str) -> Dict[str, Any]:
        """Export user customizations for backup or transfer"""
        try:
            export_data = {
                'user_id': user_id,
                'export_timestamp': datetime.now().isoformat(),
                'preferences': [],
                'usage_patterns': [],
                'ab_assignments': self.ab_assignments.get(user_id, {})
            }
            
            # Export preferences
            for pref in self.user_preferences[user_id]:
                export_data['preferences'].append({
                    'element': pref.element.value,
                    'value': pref.value,
                    'scope': pref.scope.value,
                    'priority': pref.priority.value,
                    'trigger': pref.trigger.value,
                    'confidence': pref.confidence,
                    'timestamp': pref.timestamp.isoformat(),
                    'context': pref.context,
                    'metadata': pref.metadata
                })
            
            # Export recent usage patterns (last 100)
            recent_patterns = self.usage_patterns[user_id][-100:]
            for pattern in recent_patterns:
                export_data['usage_patterns'].append({
                    'pattern_id': pattern.pattern_id,
                    'element': pattern.element.value,
                    'interaction_type': pattern.interaction_type,
                    'frequency': pattern.frequency,
                    'duration': pattern.duration,
                    'success_rate': pattern.success_rate,
                    'timestamp': pattern.timestamp.isoformat()
                })
            
            return export_data
            
        except Exception as e:
            logger.error(f"Error exporting customizations for user {user_id}: {e}")
            return {'error': 'Export failed'}
    
    async def import_user_customizations(self, user_id: str, import_data: Dict[str, Any]) -> bool:
        """Import user customizations from backup"""
        try:
            # Import preferences
            if 'preferences' in import_data:
                imported_prefs = []
                for pref_data in import_data['preferences']:
                    pref = InterfacePreference(
                        element=InterfaceElement(pref_data['element']),
                        value=pref_data['value'],
                        scope=CustomizationScope(pref_data['scope']),
                        priority=CustomizationPriority(pref_data['priority']),
                        trigger=AdaptationTrigger(pref_data['trigger']),
                        confidence=pref_data['confidence'],
                        timestamp=datetime.fromisoformat(pref_data['timestamp']),
                        context=pref_data.get('context', {}),
                        metadata=pref_data.get('metadata', {})
                    )
                    imported_prefs.append(pref)
                
                self.user_preferences[user_id] = imported_prefs
            
            # Import A/B assignments
            if 'ab_assignments' in import_data:
                self.ab_assignments[user_id] = import_data['ab_assignments']
            
            logger.info(f"Imported customizations for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing customizations for user {user_id}: {e}")
            return False
    
    # Helper methods
    async def _get_user_preferences(self, user_id: str, context: InterfaceContext) -> List[InterfacePreference]:
        """Get user preferences filtered by context"""
        try:
            all_prefs = self.user_preferences[user_id]
            
            # Filter preferences by context relevance
            relevant_prefs = []
            for pref in all_prefs:
                if await self._is_preference_relevant(pref, context):
                    relevant_prefs.append(pref)
            
            return relevant_prefs
            
        except Exception as e:
            logger.error(f"Error getting user preferences: {e}")
            return []
    
    async def _apply_customization_rules(self, user_id: str, context: InterfaceContext) -> List[InterfacePreference]:
        """Apply customization rules to generate preferences"""
        try:
            rule_prefs = []
            
            # Check rules for each scope
            for scope in [context.current_module, context.current_view, 'global']:
                scope_enum = CustomizationScope.GLOBAL
                if scope == context.current_module:
                    scope_enum = CustomizationScope.MODULE
                elif scope == context.current_view:
                    scope_enum = CustomizationScope.VIEW
                
                for rule_id in self.active_rules.get(scope_enum, []):
                    rule = self.customization_rules[rule_id]
                    if rule.active and await self._evaluate_rule_condition(rule, user_id, context):
                        # Convert rule action to preference
                        pref = await self._rule_action_to_preference(rule, context)
                        if pref:
                            rule_prefs.append(pref)
                            
                            # Update rule statistics
                            rule.last_applied = datetime.now()
                            rule.application_count += 1
            
            return rule_prefs
            
        except Exception as e:
            logger.error(f"Error applying customization rules: {e}")
            return []
    
    async def _predict_preferences(self, user_id: str, context: InterfaceContext) -> List[InterfacePreference]:
        """Use ML to predict optimal preferences"""
        try:
            predicted_prefs = []
            
            # For each interface element, predict optimal value
            for element in InterfaceElement:
                if element in self.preference_predictors:
                    predictor = self.preference_predictors[element]
                    
                    # Create feature vector from context and user history
                    features = await self._create_prediction_features(user_id, element, context)
                    
                    if features is not None:
                        # Make prediction
                        predicted_value = predictor.predict([features])[0]
                        confidence = 0.7  # Default confidence for ML predictions
                        
                        pref = InterfacePreference(
                            element=element,
                            value=predicted_value,
                            scope=CustomizationScope.GLOBAL,
                            priority=CustomizationPriority.MEDIUM,
                            trigger=AdaptationTrigger.MACHINE_LEARNING,
                            confidence=confidence
                        )
                        predicted_prefs.append(pref)
            
            return predicted_prefs
            
        except Exception as e:
            logger.error(f"Error predicting preferences: {e}")
            return []
    
    async def _merge_preferences(self, *preference_lists: List[InterfacePreference]) -> List[InterfacePreference]:
        """Merge multiple preference lists with priority resolution"""
        try:
            # Flatten all preferences
            all_prefs = []
            for pref_list in preference_lists:
                all_prefs.extend(pref_list)
            
            # Group by element and scope
            grouped_prefs = defaultdict(list)
            for pref in all_prefs:
                key = (pref.element, pref.scope)
                grouped_prefs[key].append(pref)
            
            # Resolve conflicts by priority and confidence
            merged_prefs = []
            for (element, scope), prefs in grouped_prefs.items():
                if len(prefs) == 1:
                    merged_prefs.append(prefs[0])
                else:
                    # Sort by priority (higher first) then confidence (higher first)
                    priority_order = {CustomizationPriority.CRITICAL: 4, CustomizationPriority.HIGH: 3, 
                                    CustomizationPriority.MEDIUM: 2, CustomizationPriority.LOW: 1}
                    
                    sorted_prefs = sorted(prefs, 
                                        key=lambda p: (priority_order[p.priority], p.confidence), 
                                        reverse=True)
                    merged_prefs.append(sorted_prefs[0])
            
            return merged_prefs
            
        except Exception as e:
            logger.error(f"Error merging preferences: {e}")
            return []
    
    async def _generate_interface_config(self, preferences: List[InterfacePreference], 
                                       context: InterfaceContext) -> Dict[str, Any]:
        """Generate interface configuration from preferences"""
        try:
            config = self.default_configs.copy()
            
            # Apply preferences to configuration
            for pref in preferences:
                config_key = self._element_to_config_key(pref.element)
                if config_key:
                    # Apply scope-specific configuration
                    if pref.scope == CustomizationScope.GLOBAL:
                        config[config_key] = pref.value
                    elif pref.scope == CustomizationScope.MODULE:
                        if 'modules' not in config:
                            config['modules'] = {}
                        if context.current_module not in config['modules']:
                            config['modules'][context.current_module] = {}
                        config['modules'][context.current_module][config_key] = pref.value
                    elif pref.scope == CustomizationScope.VIEW:
                        if 'views' not in config:
                            config['views'] = {}
                        if context.current_view not in config['views']:
                            config['views'][context.current_view] = {}
                        config['views'][context.current_view][config_key] = pref.value
            
            # Apply context-specific optimizations
            config = await self._apply_context_optimizations(config, context)
            
            return config
            
        except Exception as e:
            logger.error(f"Error generating interface config: {e}")
            return self.default_configs.copy()
    
    async def _estimate_performance_impact(self, config: Dict[str, Any], 
                                         context: InterfaceContext) -> Dict[str, float]:
        """Estimate performance impact of configuration"""
        try:
            impact = {
                'load_time_impact': 0.0,
                'memory_usage_impact': 0.0,
                'cpu_usage_impact': 0.0,
                'network_usage_impact': 0.0,
                'battery_impact': 0.0
            }
            
            # Estimate based on configuration changes
            if config.get('animation_speed', 'normal') == 'fast':
                impact['cpu_usage_impact'] += 0.1
            elif config.get('animation_speed', 'normal') == 'slow':
                impact['cpu_usage_impact'] -= 0.05
            
            if config.get('data_density', 'normal') == 'high':
                impact['memory_usage_impact'] += 0.2
                impact['load_time_impact'] += 0.1
            elif config.get('data_density', 'normal') == 'low':
                impact['memory_usage_impact'] -= 0.1
                impact['load_time_impact'] -= 0.05
            
            # Device-specific adjustments
            if context.device_type == 'mobile':
                # Mobile devices are more sensitive to performance
                for key in impact:
                    impact[key] *= 1.5
            
            return impact
            
        except Exception as e:
            logger.error(f"Error estimating performance impact: {e}")
            return {}
    
    async def _analyze_usage_patterns(self, user_id: str):
        """Analyze usage patterns to identify optimization opportunities"""
        try:
            patterns = self.usage_patterns[user_id]
            if len(patterns) < 10:  # Need sufficient data
                return
            
            # Analyze recent patterns (last 50)
            recent_patterns = patterns[-50:]
            
            # Group by element
            element_stats = defaultdict(list)
            for pattern in recent_patterns:
                element_stats[pattern.element].append(pattern)
            
            # Identify problematic elements (low success rate or high duration)
            for element, element_patterns in element_stats.items():
                avg_success = np.mean([p.success_rate for p in element_patterns])
                avg_duration = np.mean([p.duration for p in element_patterns])
                
                if avg_success < 0.7 or avg_duration > 10.0:  # Thresholds
                    # Generate adaptive preference
                    await self._generate_adaptive_preference(user_id, element, avg_success, avg_duration)
            
        except Exception as e:
            logger.error(f"Error analyzing usage patterns: {e}")
    
    async def _check_adaptation_triggers(self, user_id: str, element: InterfaceElement, 
                                       context: InterfaceContext):
        """Check if adaptation should be triggered"""
        try:
            recent_interactions = list(self.interaction_history[user_id])[-10:]
            
            # Check for repeated failures
            element_failures = [
                interaction for interaction in recent_interactions
                if interaction['element'] == element and not interaction['success']
            ]
            
            if len(element_failures) >= 3:  # 3 failures in last 10 interactions
                # Trigger adaptation
                await self._trigger_adaptive_customization(user_id, element, context, 'repeated_failures')
            
            # Check for performance issues
            element_durations = [
                interaction['duration'] for interaction in recent_interactions
                if interaction['element'] == element and interaction['duration'] > 0
            ]
            
            if element_durations and np.mean(element_durations) > 15.0:  # Average > 15 seconds
                await self._trigger_adaptive_customization(user_id, element, context, 'performance_issue')
            
        except Exception as e:
            logger.error(f"Error checking adaptation triggers: {e}")
    
    def _initialize_default_configs(self) -> Dict[str, Any]:
        """Initialize default interface configurations"""
        return {
            'dashboard_layout': 'grid',
            'chart_type': 'line',
            'color_scheme': 'light',
            'navigation_style': 'sidebar',
            'widget_placement': 'auto',
            'data_density': 'normal',
            'notification_style': 'toast',
            'toolbar_configuration': 'standard',
            'menu_organization': 'hierarchical',
            'font_size': 'medium',
            'animation_speed': 'normal',
            'keyboard_shortcuts': 'enabled'
        }
    
    def _element_to_config_key(self, element: InterfaceElement) -> Optional[str]:
        """Convert interface element to configuration key"""
        element_mapping = {
            InterfaceElement.DASHBOARD_LAYOUT: 'dashboard_layout',
            InterfaceElement.CHART_TYPE: 'chart_type',
            InterfaceElement.COLOR_SCHEME: 'color_scheme',
            InterfaceElement.NAVIGATION_STYLE: 'navigation_style',
            InterfaceElement.WIDGET_PLACEMENT: 'widget_placement',
            InterfaceElement.DATA_DENSITY: 'data_density',
            InterfaceElement.NOTIFICATION_STYLE: 'notification_style',
            InterfaceElement.TOOLBAR_CONFIGURATION: 'toolbar_configuration',
            InterfaceElement.MENU_ORGANIZATION: 'menu_organization',
            InterfaceElement.FONT_SIZE: 'font_size',
            InterfaceElement.ANIMATION_SPEED: 'animation_speed',
            InterfaceElement.KEYBOARD_SHORTCUTS: 'keyboard_shortcuts'
        }
        return element_mapping.get(element)
    
    async def _is_preference_relevant(self, preference: InterfacePreference, 
                                    context: InterfaceContext) -> bool:
        """Check if preference is relevant to current context"""
        try:
            # Always include global preferences
            if preference.scope == CustomizationScope.GLOBAL:
                return True
            
            # Check module-specific preferences
            if (preference.scope == CustomizationScope.MODULE and 
                preference.context.get('module') == context.current_module):
                return True
            
            # Check view-specific preferences
            if (preference.scope == CustomizationScope.VIEW and 
                preference.context.get('view') == context.current_view):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking preference relevance: {e}")
            return True  # Default to including preference
    
    async def _evaluate_rule_condition(self, rule: CustomizationRule, user_id: str, 
                                     context: InterfaceContext) -> bool:
        """Evaluate if rule condition is met"""
        try:
            # Parse condition (simplified JSON-based conditions)
            condition = json.loads(rule.condition)
            
            # Evaluate different condition types
            if condition.get('type') == 'device_type':
                return context.device_type == condition.get('value')
            
            elif condition.get('type') == 'time_of_day':
                return context.time_of_day == condition.get('value')
            
            elif condition.get('type') == 'user_role':
                return context.user_role == condition.get('value')
            
            elif condition.get('type') == 'screen_resolution':
                min_width = condition.get('min_width', 0)
                max_width = condition.get('max_width', float('inf'))
                return min_width <= context.screen_resolution[0] <= max_width
            
            elif condition.get('type') == 'usage_frequency':
                # Check user's usage frequency for specific element
                element = InterfaceElement(condition.get('element'))
                recent_patterns = [p for p in self.usage_patterns[user_id][-50:] if p.element == element]
                frequency = len(recent_patterns)
                return frequency >= condition.get('min_frequency', 0)
            
            return False
            
        except Exception as e:
            logger.error(f"Error evaluating rule condition: {e}")
            return False
    
    async def _rule_action_to_preference(self, rule: CustomizationRule, 
                                       context: InterfaceContext) -> Optional[InterfacePreference]:
        """Convert rule action to interface preference"""
        try:
            action = rule.action
            
            if 'element' in action and 'value' in action:
                element = InterfaceElement(action['element'])
                value = action['value']
                
                preference = InterfacePreference(
                    element=element,
                    value=value,
                    scope=rule.scope,
                    priority=rule.priority,
                    trigger=AdaptationTrigger.PERFORMANCE_OPTIMIZATION,
                    confidence=0.8,
                    context={'rule_id': rule.rule_id}
                )
                
                return preference
            
            return None
            
        except Exception as e:
            logger.error(f"Error converting rule action to preference: {e}")
            return None
    
    async def _create_prediction_features(self, user_id: str, element: InterfaceElement, 
                                        context: InterfaceContext) -> Optional[np.ndarray]:
        """Create feature vector for ML prediction"""
        try:
            features = []
            
            # Context features
            features.append(1.0 if context.device_type == 'mobile' else 0.0)
            features.append(1.0 if context.device_type == 'tablet' else 0.0)
            features.append(context.screen_resolution[0] / 1920.0)  # Normalized width
            features.append(context.screen_resolution[1] / 1080.0)  # Normalized height
            
            # Time features
            hour = datetime.now().hour
            features.append(hour / 24.0)  # Normalized hour
            
            # User behavior features
            user_patterns = self.usage_patterns[user_id]
            element_patterns = [p for p in user_patterns if p.element == element]
            
            if element_patterns:
                avg_duration = np.mean([p.duration for p in element_patterns[-10:]])
                avg_success = np.mean([p.success_rate for p in element_patterns[-10:]])
                features.append(avg_duration / 60.0)  # Normalized duration
                features.append(avg_success)
            else:
                features.extend([0.0, 1.0])  # Default values
            
            # Preference history features
            element_prefs = [p for p in self.user_preferences[user_id] if p.element == element]
            features.append(len(element_prefs) / 10.0)  # Normalized preference count
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error creating prediction features: {e}")
            return None
    
    async def _apply_context_optimizations(self, config: Dict[str, Any], 
                                         context: InterfaceContext) -> Dict[str, Any]:
        """Apply context-specific optimizations to configuration"""
        try:
            optimized_config = config.copy()
            
            # Mobile optimizations
            if context.device_type == 'mobile':
                optimized_config['data_density'] = 'low'
                optimized_config['font_size'] = 'large'
                optimized_config['animation_speed'] = 'fast'
            
            # Low resolution optimizations
            if context.screen_resolution[0] < 1366:
                optimized_config['dashboard_layout'] = 'single_column'
                optimized_config['widget_placement'] = 'compact'
            
            # Performance constraint optimizations
            if context.performance_constraints.get('low_memory', False):
                optimized_config['data_density'] = 'low'
                optimized_config['animation_speed'] = 'disabled'
            
            # Accessibility optimizations
            if 'high_contrast' in context.accessibility_needs:
                optimized_config['color_scheme'] = 'high_contrast'
            
            if 'large_text' in context.accessibility_needs:
                optimized_config['font_size'] = 'extra_large'
            
            return optimized_config
            
        except Exception as e:
            logger.error(f"Error applying context optimizations: {e}")
            return config
    
    async def _get_most_used_elements(self, user_id: str) -> List[Dict[str, Any]]:
        """Get most frequently used interface elements"""
        try:
            patterns = self.usage_patterns[user_id]
            
            # Count usage by element
            element_counts = defaultdict(int)
            element_durations = defaultdict(list)
            
            for pattern in patterns[-100:]:  # Last 100 patterns
                element_counts[pattern.element.value] += 1
                element_durations[pattern.element.value].append(pattern.duration)
            
            # Create sorted list
            most_used = []
            for element, count in element_counts.items():
                avg_duration = np.mean(element_durations[element]) if element_durations[element] else 0.0
                most_used.append({
                    'element': element,
                    'usage_count': count,
                    'average_duration': avg_duration
                })
            
            # Sort by usage count
            most_used.sort(key=lambda x: x['usage_count'], reverse=True)
            
            return most_used[:10]  # Top 10
            
        except Exception as e:
            logger.error(f"Error getting most used elements: {e}")
            return []
    
    async def _generate_personalization_recommendations(self, user_id: str) -> List[str]:
        """Generate personalization recommendations for user"""
        try:
            recommendations = []
            
            # Analyze usage patterns
            patterns = self.usage_patterns[user_id]
            if patterns:
                # Check for low success rates
                recent_patterns = patterns[-50:]
                avg_success = np.mean([p.success_rate for p in recent_patterns])
                
                if avg_success < 0.8:
                    recommendations.append("Consider customizing interface elements with low success rates")
                
                # Check for long durations
                avg_duration = np.mean([p.duration for p in recent_patterns if p.duration > 0])
                if avg_duration > 10.0:
                    recommendations.append("Enable keyboard shortcuts to improve efficiency")
            
            # Check preference diversity
            prefs = self.user_preferences[user_id]
            if len(prefs) < 5:
                recommendations.append("Explore more customization options to improve your experience")
            
            # Check adaptation history
            adaptations = self.adaptation_history[user_id]
            if len(adaptations) < 3:
                recommendations.append("Allow more time for the system to learn your preferences")
            
            if not recommendations:
                recommendations.append("Your interface is well-optimized for your usage patterns")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Unable to generate recommendations"]
    
    async def _generate_adaptive_preference(self, user_id: str, element: InterfaceElement,
                                          success_rate: float, avg_duration: float):
        """Generate adaptive preference based on usage analysis"""
        try:
            # Determine adaptive value based on performance issues
            adaptive_value = None
            
            if element == InterfaceElement.DATA_DENSITY and avg_duration > 15.0:
                adaptive_value = 'low'  # Reduce data density for better performance
            
            elif element == InterfaceElement.ANIMATION_SPEED and success_rate < 0.7:
                adaptive_value = 'disabled'  # Disable animations if causing issues
            
            elif element == InterfaceElement.NAVIGATION_STYLE and success_rate < 0.6:
                adaptive_value = 'breadcrumb'  # Simpler navigation
            
            if adaptive_value:
                preference = InterfacePreference(
                    element=element,
                    value=adaptive_value,
                    scope=CustomizationScope.GLOBAL,
                    priority=CustomizationPriority.MEDIUM,
                    trigger=AdaptationTrigger.USER_BEHAVIOR,
                    confidence=0.6,
                    metadata={
                        'generated_from': 'usage_analysis',
                        'success_rate': success_rate,
                        'avg_duration': avg_duration
                    }
                )
                
                self.user_preferences[user_id].append(preference)
                logger.info(f"Generated adaptive preference for user {user_id}: {element.value} = {adaptive_value}")
            
        except Exception as e:
            logger.error(f"Error generating adaptive preference: {e}")
    
    async def _trigger_adaptive_customization(self, user_id: str, element: InterfaceElement,
                                            context: InterfaceContext, trigger_reason: str):
        """Trigger adaptive customization based on detected issues"""
        try:
            logger.info(f"Triggering adaptive customization for user {user_id}, element {element.value}, reason: {trigger_reason}")
            
            # Generate new customization based on trigger reason
            if trigger_reason == 'repeated_failures':
                # Simplify the problematic element
                if element == InterfaceElement.NAVIGATION_STYLE:
                    await self.set_user_preference(user_id, element, 'simple', 
                                                  CustomizationScope.GLOBAL, CustomizationPriority.HIGH)
                
                elif element == InterfaceElement.DASHBOARD_LAYOUT:
                    await self.set_user_preference(user_id, element, 'list', 
                                                  CustomizationScope.GLOBAL, CustomizationPriority.HIGH)
            
            elif trigger_reason == 'performance_issue':
                # Optimize for performance
                if element == InterfaceElement.ANIMATION_SPEED:
                    await self.set_user_preference(user_id, element, 'disabled', 
                                                  CustomizationScope.GLOBAL, CustomizationPriority.HIGH)
                
                elif element == InterfaceElement.DATA_DENSITY:
                    await self.set_user_preference(user_id, element, 'low', 
                                                  CustomizationScope.GLOBAL, CustomizationPriority.HIGH)
            
        except Exception as e:
            logger.error(f"Error triggering adaptive customization: {e}")

# Export classes and functions
__all__ = [
    'InterfaceElement',
    'CustomizationScope',
    'AdaptationTrigger',
    'CustomizationPriority',
    'InterfacePreference',
    'CustomizationRule',
    'InterfaceContext',
    'CustomizationResult',
    'UsagePattern',
    'InterfaceCustomizer'
]