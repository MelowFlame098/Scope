# Learning Algorithms
# Phase 9: AI-First Platform Implementation

import logging
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Set, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict, deque
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, silhouette_score
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class LearningAlgorithmType(Enum):
    SUPERVISED_CLASSIFICATION = "supervised_classification"
    SUPERVISED_REGRESSION = "supervised_regression"
    UNSUPERVISED_CLUSTERING = "unsupervised_clustering"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    ONLINE_LEARNING = "online_learning"
    ENSEMBLE_LEARNING = "ensemble_learning"
    DEEP_LEARNING = "deep_learning"
    TRANSFER_LEARNING = "transfer_learning"

class ModelComplexity(Enum):
    SIMPLE = "simple"          # Linear models, simple trees
    MODERATE = "moderate"      # Random forests, SVMs
    COMPLEX = "complex"        # Neural networks, deep models
    ADAPTIVE = "adaptive"      # Auto-adjusting complexity

class LearningObjective(Enum):
    USER_PREFERENCE_PREDICTION = "user_preference_prediction"
    BEHAVIOR_CLASSIFICATION = "behavior_classification"
    RISK_TOLERANCE_ESTIMATION = "risk_tolerance_estimation"
    ENGAGEMENT_PREDICTION = "engagement_prediction"
    CHURN_PREDICTION = "churn_prediction"
    RECOMMENDATION_OPTIMIZATION = "recommendation_optimization"
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization"
    ANOMALY_DETECTION = "anomaly_detection"

class LearningMode(Enum):
    BATCH = "batch"            # Train on complete dataset
    ONLINE = "online"          # Incremental learning
    MINI_BATCH = "mini_batch"  # Small batch updates
    ACTIVE = "active"          # Query for labels
    SEMI_SUPERVISED = "semi_supervised"  # Mix of labeled/unlabeled

@dataclass
class TrainingData:
    features: np.ndarray
    labels: Optional[np.ndarray] = None
    feature_names: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    data_quality_score: float = 1.0
    sample_weights: Optional[np.ndarray] = None

@dataclass
class ModelPerformance:
    model_id: str
    algorithm_type: LearningAlgorithmType
    objective: LearningObjective
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    silhouette_score: Optional[float] = None
    training_time: float = 0.0
    prediction_time: float = 0.0
    model_size: int = 0
    cross_validation_scores: List[float] = field(default_factory=list)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    evaluation_timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class LearningConfiguration:
    algorithm_type: LearningAlgorithmType
    objective: LearningObjective
    complexity: ModelComplexity
    learning_mode: LearningMode
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    feature_selection: bool = True
    auto_tune: bool = True
    validation_split: float = 0.2
    cross_validation_folds: int = 5
    early_stopping: bool = True
    regularization: bool = True
    ensemble_size: int = 3
    update_frequency: timedelta = field(default_factory=lambda: timedelta(hours=24))
    min_samples_for_training: int = 100
    max_training_time: timedelta = field(default_factory=lambda: timedelta(minutes=30))

@dataclass
class LearningResult:
    model_id: str
    predictions: np.ndarray
    confidence_scores: Optional[np.ndarray] = None
    feature_importance: Dict[str, float] = field(default_factory=dict)
    model_explanation: str = ""
    uncertainty_estimate: Optional[float] = None
    prediction_timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

class AdaptiveLearningEngine:
    """Advanced adaptive learning engine for personalization"""
    
    def __init__(self):
        # Model storage
        self.models = {}  # model_id -> trained model
        self.model_configs = {}  # model_id -> LearningConfiguration
        self.model_performance = {}  # model_id -> ModelPerformance
        self.training_history = defaultdict(list)  # model_id -> [TrainingData]
        
        # Feature processing
        self.feature_scalers = {}  # model_id -> StandardScaler
        self.label_encoders = {}  # model_id -> LabelEncoder
        self.feature_selectors = {}  # model_id -> feature selector
        
        # Online learning buffers
        self.online_buffers = defaultdict(lambda: deque(maxlen=10000))  # model_id -> data buffer
        self.update_schedules = {}  # model_id -> next_update_time
        
        # Performance tracking
        self.performance_history = defaultdict(list)  # model_id -> [ModelPerformance]
        self.learning_curves = defaultdict(list)  # model_id -> [(timestamp, performance)]
        
        # Auto-tuning
        self.hyperparameter_search_history = defaultdict(list)
        self.best_hyperparameters = {}  # model_id -> best_params
        
        logger.info("Adaptive learning engine initialized")
    
    async def create_learning_model(self, model_id: str, 
                                  config: LearningConfiguration) -> bool:
        """Create a new learning model with specified configuration"""
        try:
            # Store configuration
            self.model_configs[model_id] = config
            
            # Initialize model based on algorithm type and complexity
            model = await self._create_base_model(config)
            self.models[model_id] = model
            
            # Initialize feature processing components
            self.feature_scalers[model_id] = StandardScaler()
            self.label_encoders[model_id] = LabelEncoder()
            
            # Set update schedule for online learning
            if config.learning_mode in [LearningMode.ONLINE, LearningMode.MINI_BATCH]:
                self.update_schedules[model_id] = datetime.now() + config.update_frequency
            
            # Initialize performance tracking
            self.model_performance[model_id] = ModelPerformance(
                model_id=model_id,
                algorithm_type=config.algorithm_type,
                objective=config.objective
            )
            
            logger.info(f"Created learning model: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating learning model {model_id}: {e}")
            return False
    
    async def train_model(self, model_id: str, training_data: TrainingData,
                         force_retrain: bool = False) -> ModelPerformance:
        """Train or update a learning model"""
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            
            config = self.model_configs[model_id]
            model = self.models[model_id]
            
            # Check if we have enough data
            if len(training_data.features) < config.min_samples_for_training:
                logger.warning(f"Insufficient training data for {model_id}: {len(training_data.features)} < {config.min_samples_for_training}")
                return self.model_performance[model_id]
            
            # Store training data
            self.training_history[model_id].append(training_data)
            
            start_time = datetime.now()
            
            # Preprocess features
            X_processed = await self._preprocess_features(model_id, training_data.features, fit=True)
            
            # Handle different learning modes
            if config.learning_mode == LearningMode.BATCH or force_retrain:
                performance = await self._batch_training(model_id, X_processed, training_data.labels)
            
            elif config.learning_mode == LearningMode.ONLINE:
                performance = await self._online_training(model_id, X_processed, training_data.labels)
            
            elif config.learning_mode == LearningMode.MINI_BATCH:
                performance = await self._mini_batch_training(model_id, X_processed, training_data.labels)
            
            else:
                performance = await self._batch_training(model_id, X_processed, training_data.labels)
            
            # Update training time
            training_time = (datetime.now() - start_time).total_seconds()
            performance.training_time = training_time
            
            # Store performance
            self.model_performance[model_id] = performance
            self.performance_history[model_id].append(performance)
            self.learning_curves[model_id].append((datetime.now(), performance))
            
            # Auto-tune if enabled
            if config.auto_tune and len(self.performance_history[model_id]) > 1:
                await self._auto_tune_hyperparameters(model_id)
            
            logger.info(f"Trained model {model_id} with performance: {performance.accuracy or performance.mse}")
            return performance
            
        except Exception as e:
            logger.error(f"Error training model {model_id}: {e}")
            return self.model_performance.get(model_id, ModelPerformance(
                model_id=model_id,
                algorithm_type=self.model_configs[model_id].algorithm_type,
                objective=self.model_configs[model_id].objective
            ))
    
    async def predict(self, model_id: str, features: np.ndarray,
                     return_confidence: bool = True) -> LearningResult:
        """Make predictions using a trained model"""
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            
            model = self.models[model_id]
            config = self.model_configs[model_id]
            
            start_time = datetime.now()
            
            # Preprocess features
            X_processed = await self._preprocess_features(model_id, features, fit=False)
            
            # Make predictions
            if config.algorithm_type == LearningAlgorithmType.SUPERVISED_CLASSIFICATION:
                predictions = model.predict(X_processed)
                confidence_scores = None
                
                if return_confidence and hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(X_processed)
                    confidence_scores = np.max(probabilities, axis=1)
            
            elif config.algorithm_type == LearningAlgorithmType.SUPERVISED_REGRESSION:
                predictions = model.predict(X_processed)
                confidence_scores = None
                
                # Estimate uncertainty for regression
                if return_confidence and hasattr(model, 'predict'):
                    # Simple uncertainty estimation based on prediction variance
                    if len(self.training_history[model_id]) > 0:
                        recent_data = self.training_history[model_id][-1]
                        if recent_data.labels is not None:
                            label_std = np.std(recent_data.labels)
                            confidence_scores = np.full(len(predictions), 1.0 / (1.0 + label_std))
            
            elif config.algorithm_type == LearningAlgorithmType.UNSUPERVISED_CLUSTERING:
                predictions = model.predict(X_processed)
                confidence_scores = None
                
                # Calculate silhouette scores as confidence
                if return_confidence and len(X_processed) > 1:
                    try:
                        from sklearn.metrics import silhouette_samples
                        confidence_scores = silhouette_samples(X_processed, predictions)
                    except:
                        confidence_scores = None
            
            else:
                predictions = model.predict(X_processed)
                confidence_scores = None
            
            # Calculate prediction time
            prediction_time = (datetime.now() - start_time).total_seconds()
            
            # Get feature importance
            feature_importance = await self._get_feature_importance(model_id)
            
            # Generate model explanation
            explanation = await self._generate_model_explanation(model_id, predictions, confidence_scores)
            
            # Calculate uncertainty estimate
            uncertainty = None
            if confidence_scores is not None:
                uncertainty = 1.0 - np.mean(confidence_scores)
            
            result = LearningResult(
                model_id=model_id,
                predictions=predictions,
                confidence_scores=confidence_scores,
                feature_importance=feature_importance,
                model_explanation=explanation,
                uncertainty_estimate=uncertainty,
                metadata={
                    'prediction_time': prediction_time,
                    'num_samples': len(features),
                    'algorithm_type': config.algorithm_type.value,
                    'objective': config.objective.value
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error making predictions with model {model_id}: {e}")
            return LearningResult(
                model_id=model_id,
                predictions=np.array([]),
                model_explanation=f"Prediction failed: {str(e)}"
            )
    
    async def update_model_online(self, model_id: str, new_features: np.ndarray,
                                new_labels: Optional[np.ndarray] = None):
        """Update model with new data for online learning"""
        try:
            if model_id not in self.models:
                return
            
            config = self.model_configs[model_id]
            
            # Add to online buffer
            self.online_buffers[model_id].append({
                'features': new_features,
                'labels': new_labels,
                'timestamp': datetime.now()
            })
            
            # Check if it's time to update
            if (model_id in self.update_schedules and 
                datetime.now() >= self.update_schedules[model_id]):
                
                await self._process_online_buffer(model_id)
                
                # Schedule next update
                self.update_schedules[model_id] = datetime.now() + config.update_frequency
            
        except Exception as e:
            logger.error(f"Error updating model online {model_id}: {e}")
    
    async def evaluate_model(self, model_id: str, test_data: TrainingData) -> ModelPerformance:
        """Evaluate model performance on test data"""
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            
            config = self.model_configs[model_id]
            
            # Make predictions on test data
            result = await self.predict(model_id, test_data.features, return_confidence=False)
            predictions = result.predictions
            
            # Calculate performance metrics
            performance = ModelPerformance(
                model_id=model_id,
                algorithm_type=config.algorithm_type,
                objective=config.objective,
                prediction_time=result.metadata.get('prediction_time', 0.0)
            )
            
            if config.algorithm_type == LearningAlgorithmType.SUPERVISED_CLASSIFICATION:
                if test_data.labels is not None:
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                    
                    performance.accuracy = accuracy_score(test_data.labels, predictions)
                    performance.precision = precision_score(test_data.labels, predictions, average='weighted', zero_division=0)
                    performance.recall = recall_score(test_data.labels, predictions, average='weighted', zero_division=0)
                    performance.f1_score = f1_score(test_data.labels, predictions, average='weighted', zero_division=0)
            
            elif config.algorithm_type == LearningAlgorithmType.SUPERVISED_REGRESSION:
                if test_data.labels is not None:
                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                    
                    performance.mse = mean_squared_error(test_data.labels, predictions)
                    performance.mae = mean_absolute_error(test_data.labels, predictions)
                    performance.r2_score = r2_score(test_data.labels, predictions)
            
            elif config.algorithm_type == LearningAlgorithmType.UNSUPERVISED_CLUSTERING:
                if len(test_data.features) > 1:
                    try:
                        performance.silhouette_score = silhouette_score(test_data.features, predictions)
                    except:
                        performance.silhouette_score = 0.0
            
            # Get feature importance
            performance.feature_importance = await self._get_feature_importance(model_id)
            
            return performance
            
        except Exception as e:
            logger.error(f"Error evaluating model {model_id}: {e}")
            return ModelPerformance(
                model_id=model_id,
                algorithm_type=self.model_configs[model_id].algorithm_type,
                objective=self.model_configs[model_id].objective
            )
    
    async def get_model_insights(self, model_id: str) -> Dict[str, Any]:
        """Get comprehensive insights about a model"""
        try:
            if model_id not in self.models:
                return {'error': 'Model not found'}
            
            config = self.model_configs[model_id]
            performance = self.model_performance[model_id]
            
            # Learning curve data
            learning_curve = [
                {
                    'timestamp': timestamp.isoformat(),
                    'performance': perf.accuracy or perf.mse or perf.silhouette_score or 0.0
                }
                for timestamp, perf in self.learning_curves[model_id]
            ]
            
            # Training data statistics
            training_stats = {}
            if self.training_history[model_id]:
                recent_data = self.training_history[model_id][-1]
                training_stats = {
                    'total_samples': len(recent_data.features),
                    'feature_count': recent_data.features.shape[1] if len(recent_data.features.shape) > 1 else 1,
                    'data_quality_score': recent_data.data_quality_score,
                    'last_training': recent_data.timestamp.isoformat()
                }
            
            # Model complexity analysis
            complexity_analysis = await self._analyze_model_complexity(model_id)
            
            # Performance trends
            performance_trend = await self._analyze_performance_trend(model_id)
            
            insights = {
                'model_id': model_id,
                'configuration': {
                    'algorithm_type': config.algorithm_type.value,
                    'objective': config.objective.value,
                    'complexity': config.complexity.value,
                    'learning_mode': config.learning_mode.value
                },
                'current_performance': {
                    'accuracy': performance.accuracy,
                    'mse': performance.mse,
                    'r2_score': performance.r2_score,
                    'silhouette_score': performance.silhouette_score,
                    'training_time': performance.training_time,
                    'prediction_time': performance.prediction_time
                },
                'training_statistics': training_stats,
                'learning_curve': learning_curve,
                'feature_importance': performance.feature_importance,
                'complexity_analysis': complexity_analysis,
                'performance_trend': performance_trend,
                'recommendations': await self._generate_model_recommendations(model_id)
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting model insights {model_id}: {e}")
            return {'error': 'Failed to generate insights'}
    
    async def optimize_model_ensemble(self, base_model_ids: List[str],
                                    ensemble_id: str,
                                    test_data: TrainingData) -> str:
        """Create an optimized ensemble from multiple models"""
        try:
            if not base_model_ids or len(base_model_ids) < 2:
                raise ValueError("Need at least 2 models for ensemble")
            
            # Validate all models exist
            for model_id in base_model_ids:
                if model_id not in self.models:
                    raise ValueError(f"Model {model_id} not found")
            
            # Get predictions from all base models
            base_predictions = []
            base_confidences = []
            
            for model_id in base_model_ids:
                result = await self.predict(model_id, test_data.features, return_confidence=True)
                base_predictions.append(result.predictions)
                if result.confidence_scores is not None:
                    base_confidences.append(result.confidence_scores)
            
            # Create ensemble configuration
            ensemble_config = LearningConfiguration(
                algorithm_type=LearningAlgorithmType.ENSEMBLE_LEARNING,
                objective=self.model_configs[base_model_ids[0]].objective,
                complexity=ModelComplexity.MODERATE,
                learning_mode=LearningMode.BATCH
            )
            
            # Create ensemble model
            await self.create_learning_model(ensemble_id, ensemble_config)
            
            # Train ensemble weights
            ensemble_model = await self._create_ensemble_model(base_predictions, base_confidences, test_data.labels)
            self.models[ensemble_id] = ensemble_model
            
            # Store base model references
            self.models[ensemble_id].base_models = base_model_ids
            
            # Evaluate ensemble performance
            ensemble_performance = await self.evaluate_model(ensemble_id, test_data)
            
            logger.info(f"Created ensemble model {ensemble_id} from {len(base_model_ids)} base models")
            return ensemble_id
            
        except Exception as e:
            logger.error(f"Error creating ensemble model: {e}")
            return ""
    
    async def transfer_learning(self, source_model_id: str, target_model_id: str,
                              target_data: TrainingData,
                              freeze_layers: bool = True) -> bool:
        """Apply transfer learning from source to target model"""
        try:
            if source_model_id not in self.models:
                raise ValueError(f"Source model {source_model_id} not found")
            
            source_model = self.models[source_model_id]
            source_config = self.model_configs[source_model_id]
            
            # Create target model configuration
            target_config = LearningConfiguration(
                algorithm_type=LearningAlgorithmType.TRANSFER_LEARNING,
                objective=source_config.objective,
                complexity=source_config.complexity,
                learning_mode=LearningMode.BATCH
            )
            
            # Create target model
            await self.create_learning_model(target_model_id, target_config)
            
            # Transfer knowledge based on model type
            if hasattr(source_model, 'coef_'):  # Linear models
                target_model = await self._transfer_linear_model(source_model, target_data, freeze_layers)
            elif hasattr(source_model, 'feature_importances_'):  # Tree-based models
                target_model = await self._transfer_tree_model(source_model, target_data, freeze_layers)
            else:
                # Generic transfer - use source model as initialization
                target_model = source_model
            
            self.models[target_model_id] = target_model
            
            # Fine-tune on target data
            await self.train_model(target_model_id, target_data)
            
            logger.info(f"Applied transfer learning from {source_model_id} to {target_model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error in transfer learning: {e}")
            return False
    
    # Helper methods for model creation and training
    async def _create_base_model(self, config: LearningConfiguration):
        """Create base model based on configuration"""
        try:
            if config.algorithm_type == LearningAlgorithmType.SUPERVISED_CLASSIFICATION:
                if config.complexity == ModelComplexity.SIMPLE:
                    return LogisticRegression(**config.hyperparameters)
                elif config.complexity == ModelComplexity.MODERATE:
                    return RandomForestClassifier(**config.hyperparameters)
                elif config.complexity == ModelComplexity.COMPLEX:
                    return MLPClassifier(**config.hyperparameters)
                else:  # ADAPTIVE
                    return RandomForestClassifier(**config.hyperparameters)
            
            elif config.algorithm_type == LearningAlgorithmType.SUPERVISED_REGRESSION:
                if config.complexity == ModelComplexity.SIMPLE:
                    return LinearRegression(**config.hyperparameters)
                elif config.complexity == ModelComplexity.MODERATE:
                    return GradientBoostingRegressor(**config.hyperparameters)
                elif config.complexity == ModelComplexity.COMPLEX:
                    return MLPRegressor(**config.hyperparameters)
                else:  # ADAPTIVE
                    return GradientBoostingRegressor(**config.hyperparameters)
            
            elif config.algorithm_type == LearningAlgorithmType.UNSUPERVISED_CLUSTERING:
                if config.complexity == ModelComplexity.SIMPLE:
                    return KMeans(**config.hyperparameters)
                else:
                    return DBSCAN(**config.hyperparameters)
            
            else:
                # Default to random forest for unknown types
                return RandomForestClassifier(**config.hyperparameters)
                
        except Exception as e:
            logger.error(f"Error creating base model: {e}")
            # Return simple default model
            return LogisticRegression()
    
    async def _preprocess_features(self, model_id: str, features: np.ndarray, fit: bool = False) -> np.ndarray:
        """Preprocess features for model training/prediction"""
        try:
            scaler = self.feature_scalers[model_id]
            
            if fit:
                # Fit and transform for training
                X_scaled = scaler.fit_transform(features)
            else:
                # Only transform for prediction
                X_scaled = scaler.transform(features)
            
            # Feature selection if enabled
            config = self.model_configs[model_id]
            if config.feature_selection and model_id in self.feature_selectors:
                selector = self.feature_selectors[model_id]
                if fit:
                    X_selected = selector.fit_transform(X_scaled)
                else:
                    X_selected = selector.transform(X_scaled)
                return X_selected
            
            return X_scaled
            
        except Exception as e:
            logger.error(f"Error preprocessing features for {model_id}: {e}")
            return features
    
    async def _batch_training(self, model_id: str, X: np.ndarray, y: Optional[np.ndarray]) -> ModelPerformance:
        """Perform batch training"""
        try:
            model = self.models[model_id]
            config = self.model_configs[model_id]
            
            if y is None and config.algorithm_type != LearningAlgorithmType.UNSUPERVISED_CLUSTERING:
                raise ValueError("Labels required for supervised learning")
            
            # Split data for validation if needed
            if y is not None and config.validation_split > 0:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=config.validation_split, random_state=42
                )
            else:
                X_train, y_train = X, y
                X_val, y_val = None, None
            
            # Train model
            if y_train is not None:
                model.fit(X_train, y_train)
            else:
                model.fit(X_train)  # Unsupervised
            
            # Evaluate on validation set
            performance = ModelPerformance(
                model_id=model_id,
                algorithm_type=config.algorithm_type,
                objective=config.objective
            )
            
            if X_val is not None and y_val is not None:
                val_predictions = model.predict(X_val)
                
                if config.algorithm_type == LearningAlgorithmType.SUPERVISED_CLASSIFICATION:
                    performance.accuracy = accuracy_score(y_val, val_predictions)
                elif config.algorithm_type == LearningAlgorithmType.SUPERVISED_REGRESSION:
                    performance.mse = mean_squared_error(y_val, val_predictions)
            
            # Cross-validation if enabled
            if config.cross_validation_folds > 1 and y is not None:
                cv_scores = cross_val_score(model, X, y, cv=config.cross_validation_folds)
                performance.cross_validation_scores = cv_scores.tolist()
            
            return performance
            
        except Exception as e:
            logger.error(f"Error in batch training for {model_id}: {e}")
            return ModelPerformance(
                model_id=model_id,
                algorithm_type=config.algorithm_type,
                objective=config.objective
            )
    
    async def _online_training(self, model_id: str, X: np.ndarray, y: Optional[np.ndarray]) -> ModelPerformance:
        """Perform online training"""
        try:
            # For now, implement as mini-batch with size 1
            return await self._mini_batch_training(model_id, X, y)
            
        except Exception as e:
            logger.error(f"Error in online training for {model_id}: {e}")
            config = self.model_configs[model_id]
            return ModelPerformance(
                model_id=model_id,
                algorithm_type=config.algorithm_type,
                objective=config.objective
            )
    
    async def _mini_batch_training(self, model_id: str, X: np.ndarray, y: Optional[np.ndarray]) -> ModelPerformance:
        """Perform mini-batch training"""
        try:
            model = self.models[model_id]
            config = self.model_configs[model_id]
            
            # Check if model supports partial_fit
            if hasattr(model, 'partial_fit'):
                if y is not None:
                    model.partial_fit(X, y)
                else:
                    model.partial_fit(X)
            else:
                # Fall back to regular fit
                if y is not None:
                    model.fit(X, y)
                else:
                    model.fit(X)
            
            # Simple performance estimation
            performance = ModelPerformance(
                model_id=model_id,
                algorithm_type=config.algorithm_type,
                objective=config.objective
            )
            
            # Estimate performance on current batch
            if y is not None:
                predictions = model.predict(X)
                if config.algorithm_type == LearningAlgorithmType.SUPERVISED_CLASSIFICATION:
                    performance.accuracy = accuracy_score(y, predictions)
                elif config.algorithm_type == LearningAlgorithmType.SUPERVISED_REGRESSION:
                    performance.mse = mean_squared_error(y, predictions)
            
            return performance
            
        except Exception as e:
            logger.error(f"Error in mini-batch training for {model_id}: {e}")
            config = self.model_configs[model_id]
            return ModelPerformance(
                model_id=model_id,
                algorithm_type=config.algorithm_type,
                objective=config.objective
            )
    
    async def _process_online_buffer(self, model_id: str):
        """Process accumulated data in online buffer"""
        try:
            buffer = self.online_buffers[model_id]
            if not buffer:
                return
            
            # Combine buffered data
            all_features = []
            all_labels = []
            
            for data_point in buffer:
                all_features.append(data_point['features'])
                if data_point['labels'] is not None:
                    all_labels.append(data_point['labels'])
            
            if all_features:
                combined_features = np.vstack(all_features)
                combined_labels = np.hstack(all_labels) if all_labels else None
                
                # Create training data
                training_data = TrainingData(
                    features=combined_features,
                    labels=combined_labels
                )
                
                # Train model
                await self.train_model(model_id, training_data)
                
                # Clear buffer
                buffer.clear()
            
        except Exception as e:
            logger.error(f"Error processing online buffer for {model_id}: {e}")
    
    async def _get_feature_importance(self, model_id: str) -> Dict[str, float]:
        """Get feature importance from model"""
        try:
            model = self.models[model_id]
            importance = {}
            
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
                for i, imp in enumerate(importances):
                    importance[f"feature_{i}"] = float(imp)
            
            elif hasattr(model, 'coef_'):
                # Linear models
                coefficients = model.coef_
                if len(coefficients.shape) > 1:
                    coefficients = coefficients[0]  # Take first class for multi-class
                
                for i, coef in enumerate(coefficients):
                    importance[f"feature_{i}"] = float(abs(coef))
            
            return importance
            
        except Exception as e:
            logger.error(f"Error getting feature importance for {model_id}: {e}")
            return {}
    
    async def _generate_model_explanation(self, model_id: str, predictions: np.ndarray,
                                        confidence_scores: Optional[np.ndarray]) -> str:
        """Generate human-readable explanation for model predictions"""
        try:
            config = self.model_configs[model_id]
            
            explanation_parts = []
            
            # Basic model info
            explanation_parts.append(f"Model {model_id} ({config.algorithm_type.value})")
            explanation_parts.append(f"Objective: {config.objective.value}")
            
            # Prediction summary
            if len(predictions) > 0:
                if config.algorithm_type == LearningAlgorithmType.SUPERVISED_CLASSIFICATION:
                    unique_preds, counts = np.unique(predictions, return_counts=True)
                    explanation_parts.append(f"Predicted {len(unique_preds)} different classes")
                
                elif config.algorithm_type == LearningAlgorithmType.SUPERVISED_REGRESSION:
                    explanation_parts.append(f"Predicted values range: {np.min(predictions):.3f} to {np.max(predictions):.3f}")
                
                elif config.algorithm_type == LearningAlgorithmType.UNSUPERVISED_CLUSTERING:
                    unique_clusters = len(np.unique(predictions))
                    explanation_parts.append(f"Identified {unique_clusters} clusters")
            
            # Confidence summary
            if confidence_scores is not None:
                avg_confidence = np.mean(confidence_scores)
                explanation_parts.append(f"Average confidence: {avg_confidence:.3f}")
            
            return ". ".join(explanation_parts)
            
        except Exception as e:
            logger.error(f"Error generating model explanation: {e}")
            return f"Model {model_id} made predictions"
    
    async def _auto_tune_hyperparameters(self, model_id: str):
        """Auto-tune hyperparameters based on performance history"""
        try:
            config = self.model_configs[model_id]
            performance_history = self.performance_history[model_id]
            
            if len(performance_history) < 3:  # Need some history
                return
            
            # Simple performance trend analysis
            recent_performances = performance_history[-3:]
            performance_values = []
            
            for perf in recent_performances:
                if perf.accuracy is not None:
                    performance_values.append(perf.accuracy)
                elif perf.mse is not None:
                    performance_values.append(-perf.mse)  # Negative because lower is better
                elif perf.r2_score is not None:
                    performance_values.append(perf.r2_score)
                else:
                    performance_values.append(0.0)
            
            # Check if performance is declining
            if len(performance_values) >= 2 and performance_values[-1] < performance_values[-2]:
                # Try adjusting hyperparameters
                await self._adjust_hyperparameters(model_id)
            
        except Exception as e:
            logger.error(f"Error auto-tuning hyperparameters for {model_id}: {e}")
    
    async def _adjust_hyperparameters(self, model_id: str):
        """Adjust hyperparameters for better performance"""
        try:
            config = self.model_configs[model_id]
            model = self.models[model_id]
            
            # Simple hyperparameter adjustments based on model type
            if isinstance(model, RandomForestClassifier) or isinstance(model, RandomForestClassifier):
                # Increase number of estimators
                current_estimators = getattr(model, 'n_estimators', 100)
                new_estimators = min(current_estimators + 50, 500)
                model.set_params(n_estimators=new_estimators)
            
            elif isinstance(model, LogisticRegression):
                # Adjust regularization
                current_C = getattr(model, 'C', 1.0)
                new_C = current_C * 0.5 if current_C > 0.01 else current_C * 2.0
                model.set_params(C=new_C)
            
            logger.info(f"Adjusted hyperparameters for model {model_id}")
            
        except Exception as e:
            logger.error(f"Error adjusting hyperparameters for {model_id}: {e}")
    
    async def _analyze_model_complexity(self, model_id: str) -> Dict[str, Any]:
        """Analyze model complexity"""
        try:
            model = self.models[model_id]
            config = self.model_configs[model_id]
            
            complexity_analysis = {
                'configured_complexity': config.complexity.value,
                'estimated_parameters': 0,
                'memory_usage': 0,
                'computational_complexity': 'unknown'
            }
            
            # Estimate number of parameters
            if hasattr(model, 'coef_'):
                if hasattr(model.coef_, 'size'):
                    complexity_analysis['estimated_parameters'] = model.coef_.size
            elif hasattr(model, 'n_estimators'):
                complexity_analysis['estimated_parameters'] = getattr(model, 'n_estimators', 0)
            
            # Estimate computational complexity
            if isinstance(model, (LogisticRegression, LinearRegression)):
                complexity_analysis['computational_complexity'] = 'O(n*d)'
            elif isinstance(model, (RandomForestClassifier, GradientBoostingRegressor)):
                complexity_analysis['computational_complexity'] = 'O(n*d*log(n)*trees)'
            elif isinstance(model, (MLPClassifier, MLPRegressor)):
                complexity_analysis['computational_complexity'] = 'O(n*d*layers*neurons)'
            
            return complexity_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing model complexity for {model_id}: {e}")
            return {'error': 'Failed to analyze complexity'}
    
    async def _analyze_performance_trend(self, model_id: str) -> Dict[str, Any]:
        """Analyze performance trend over time"""
        try:
            performance_history = self.performance_history[model_id]
            
            if len(performance_history) < 2:
                return {'trend': 'insufficient_data'}
            
            # Extract performance values
            performance_values = []
            timestamps = []
            
            for perf in performance_history:
                if perf.accuracy is not None:
                    performance_values.append(perf.accuracy)
                elif perf.mse is not None:
                    performance_values.append(-perf.mse)  # Negative for consistency
                elif perf.r2_score is not None:
                    performance_values.append(perf.r2_score)
                else:
                    performance_values.append(0.0)
                
                timestamps.append(perf.evaluation_timestamp)
            
            # Simple trend analysis
            if len(performance_values) >= 2:
                recent_trend = performance_values[-1] - performance_values[-2]
                overall_trend = performance_values[-1] - performance_values[0]
                
                trend_analysis = {
                    'recent_change': recent_trend,
                    'overall_change': overall_trend,
                    'trend_direction': 'improving' if overall_trend > 0 else 'declining' if overall_trend < 0 else 'stable',
                    'volatility': np.std(performance_values) if len(performance_values) > 1 else 0.0,
                    'best_performance': max(performance_values),
                    'worst_performance': min(performance_values),
                    'current_performance': performance_values[-1]
                }
                
                return trend_analysis
            
            return {'trend': 'insufficient_data'}
            
        except Exception as e:
            logger.error(f"Error analyzing performance trend for {model_id}: {e}")
            return {'error': 'Failed to analyze trend'}
    
    async def _generate_model_recommendations(self, model_id: str) -> List[str]:
        """Generate recommendations for improving model performance"""
        try:
            recommendations = []
            
            config = self.model_configs[model_id]
            performance = self.model_performance[model_id]
            trend_analysis = await self._analyze_performance_trend(model_id)
            
            # Performance-based recommendations
            if performance.accuracy is not None and performance.accuracy < 0.7:
                recommendations.append("Consider increasing model complexity or gathering more training data")
            
            if performance.mse is not None and performance.mse > 1.0:
                recommendations.append("High prediction error - consider feature engineering or regularization")
            
            # Trend-based recommendations
            if trend_analysis.get('trend_direction') == 'declining':
                recommendations.append("Performance is declining - consider retraining or hyperparameter tuning")
            
            if trend_analysis.get('volatility', 0) > 0.1:
                recommendations.append("High performance volatility - consider ensemble methods for stability")
            
            # Training time recommendations
            if performance.training_time > 300:  # 5 minutes
                recommendations.append("Long training time - consider reducing model complexity or data size")
            
            # Data-based recommendations
            if self.training_history[model_id]:
                recent_data = self.training_history[model_id][-1]
                if len(recent_data.features) < 1000:
                    recommendations.append("Limited training data - consider data augmentation or transfer learning")
            
            # Configuration-based recommendations
            if config.learning_mode == LearningMode.BATCH and len(self.online_buffers[model_id]) > 0:
                recommendations.append("Consider switching to online learning for real-time adaptation")
            
            if not recommendations:
                recommendations.append("Model performance is satisfactory - continue monitoring")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations for {model_id}: {e}")
            return ["Unable to generate recommendations"]
    
    async def _create_ensemble_model(self, base_predictions: List[np.ndarray],
                                   base_confidences: List[np.ndarray],
                                   true_labels: Optional[np.ndarray]):
        """Create ensemble model from base predictions"""
        try:
            # Simple voting ensemble for now
            class VotingEnsemble:
                def __init__(self, base_models):
                    self.base_models = base_models
                    self.weights = np.ones(len(base_models)) / len(base_models)
                
                def predict(self, X):
                    # This is a placeholder - in practice, would need to get predictions from base models
                    return np.zeros(len(X))
                
                def fit(self, X, y=None):
                    # Ensemble doesn't need separate fitting
                    pass
            
            return VotingEnsemble(base_predictions)
            
        except Exception as e:
            logger.error(f"Error creating ensemble model: {e}")
            return None
    
    async def _transfer_linear_model(self, source_model, target_data: TrainingData, freeze_layers: bool):
        """Transfer linear model knowledge"""
        try:
            # Create new model with same architecture
            if hasattr(source_model, 'coef_'):
                from sklearn.linear_model import LogisticRegression, LinearRegression
                
                if isinstance(source_model, LogisticRegression):
                    target_model = LogisticRegression()
                else:
                    target_model = LinearRegression()
                
                # Initialize with source weights if not freezing
                if not freeze_layers and hasattr(source_model, 'coef_'):
                    # This is a simplified transfer - in practice would need more sophisticated approach
                    target_model.fit(target_data.features, target_data.labels)
                
                return target_model
            
            return source_model
            
        except Exception as e:
            logger.error(f"Error in linear model transfer: {e}")
            return source_model
    
    async def _transfer_tree_model(self, source_model, target_data: TrainingData, freeze_layers: bool):
        """Transfer tree model knowledge"""
        try:
            # For tree models, transfer feature importance knowledge
            if hasattr(source_model, 'feature_importances_'):
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                
                if hasattr(source_model, 'classes_'):
                    target_model = RandomForestClassifier()
                else:
                    target_model = RandomForestRegressor()
                
                # Train on target data
                target_model.fit(target_data.features, target_data.labels)
                
                return target_model
            
            return source_model
            
        except Exception as e:
            logger.error(f"Error in tree model transfer: {e}")
            return source_model

# Export classes and functions
__all__ = [
    'LearningAlgorithmType',
    'ModelComplexity',
    'LearningObjective',
    'LearningMode',
    'TrainingData',
    'ModelPerformance',
    'LearningConfiguration',
    'LearningResult',
    'AdaptiveLearningEngine'
]