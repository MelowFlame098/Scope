import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SelectKBest, f_regression, RFE, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns

# Advanced ML libraries (with fallbacks)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    
try:
    from sklearn.experimental import enable_halving_search_cv
    from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
    HALVING_SEARCH_AVAILABLE = True
except ImportError:
    HALVING_SEARCH_AVAILABLE = False
    
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    
try:
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore')

@dataclass
class FeatureImportance:
    """Feature importance information."""
    feature_name: str
    importance: float
    rank: int
    method: str
    
@dataclass
class ModelPerformance:
    """Model performance metrics."""
    model_name: str
    mse: float
    mae: float
    r2: float
    rmse: float
    mape: float
    directional_accuracy: float
    training_time: float
    prediction_time: float
    cross_val_score: float
    cross_val_std: float
    
@dataclass
class HyperparameterResult:
    """Hyperparameter optimization result."""
    model_name: str
    best_params: Dict[str, Any]
    best_score: float
    optimization_method: str
    n_trials: int
    optimization_time: float
    param_importance: Optional[Dict[str, float]] = None
    
@dataclass
class AutoMLResults:
    """Comprehensive AutoML results."""
    best_model: Any
    best_model_name: str
    model_performances: List[ModelPerformance]
    feature_importance: List[FeatureImportance]
    hyperparameter_results: List[HyperparameterResult]
    ensemble_performance: Optional[ModelPerformance]
    feature_engineering_results: Dict[str, Any]
    model_selection_summary: Dict[str, Any]
    predictions: np.ndarray
    actual_values: np.ndarray
    prediction_intervals: Optional[Dict[str, np.ndarray]]
    insights: Dict[str, Any]
    
class AutoFeatureEngineer(BaseEstimator, TransformerMixin):
    """Automated feature engineering for time series data."""
    
    def __init__(self, max_features: int = 100, include_interactions: bool = True,
                 include_polynomials: bool = True, polynomial_degree: int = 2,
                 include_lag_features: bool = True, max_lags: int = 10,
                 include_rolling_features: bool = True, rolling_windows: List[int] = None):
        self.max_features = max_features
        self.include_interactions = include_interactions
        self.include_polynomials = include_polynomials
        self.polynomial_degree = polynomial_degree
        self.include_lag_features = include_lag_features
        self.max_lags = max_lags
        self.include_rolling_features = include_rolling_features
        self.rolling_windows = rolling_windows or [5, 10, 20, 50]
        self.feature_names_ = []
        self.selected_features_ = []
        
    def fit(self, X, y=None):
        """Fit the feature engineer."""
        return self
        
    def transform(self, X):
        """Transform the data with automated feature engineering."""
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X)
            
        original_features = list(df.columns)
        
        # Basic statistical features
        df = self._add_statistical_features(df)
        
        # Lag features
        if self.include_lag_features:
            df = self._add_lag_features(df, original_features)
            
        # Rolling window features
        if self.include_rolling_features:
            df = self._add_rolling_features(df, original_features)
            
        # Polynomial features
        if self.include_polynomials:
            df = self._add_polynomial_features(df, original_features)
            
        # Interaction features
        if self.include_interactions:
            df = self._add_interaction_features(df, original_features)
            
        # Technical indicators
        df = self._add_technical_indicators(df, original_features)
        
        # Remove infinite and NaN values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Feature selection if too many features
        if len(df.columns) > self.max_features:
            df = self._select_top_features(df)
            
        self.feature_names_ = list(df.columns)
        return df.values
        
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic statistical features."""
        for col in df.select_dtypes(include=[np.number]).columns:
            # Percentile features
            df[f'{col}_pct_rank'] = df[col].rank(pct=True)
            
            # Z-score
            df[f'{col}_zscore'] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
            
            # Log transformation (for positive values)
            if (df[col] > 0).all():
                df[f'{col}_log'] = np.log(df[col] + 1)
                
            # Square root transformation
            if (df[col] >= 0).all():
                df[f'{col}_sqrt'] = np.sqrt(df[col])
                
        return df
        
    def _add_lag_features(self, df: pd.DataFrame, original_features: List[str]) -> pd.DataFrame:
        """Add lag features."""
        for col in original_features:
            if df[col].dtype in [np.float64, np.int64]:
                for lag in range(1, min(self.max_lags + 1, len(df) // 4)):
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                    
        return df
        
    def _add_rolling_features(self, df: pd.DataFrame, original_features: List[str]) -> pd.DataFrame:
        """Add rolling window features."""
        for col in original_features:
            if df[col].dtype in [np.float64, np.int64]:
                for window in self.rolling_windows:
                    if window < len(df):
                        # Rolling statistics
                        df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
                        df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()
                        df[f'{col}_rolling_min_{window}'] = df[col].rolling(window=window).min()
                        df[f'{col}_rolling_max_{window}'] = df[col].rolling(window=window).max()
                        df[f'{col}_rolling_median_{window}'] = df[col].rolling(window=window).median()
                        
                        # Rolling ratios
                        rolling_mean = df[col].rolling(window=window).mean()
                        df[f'{col}_ratio_to_rolling_mean_{window}'] = df[col] / (rolling_mean + 1e-8)
                        
        return df
        
    def _add_polynomial_features(self, df: pd.DataFrame, original_features: List[str]) -> pd.DataFrame:
        """Add polynomial features."""
        numeric_features = [col for col in original_features if df[col].dtype in [np.float64, np.int64]]
        
        # Limit to prevent explosion of features
        selected_features = numeric_features[:min(5, len(numeric_features))]
        
        for col in selected_features:
            for degree in range(2, self.polynomial_degree + 1):
                df[f'{col}_poly_{degree}'] = df[col] ** degree
                
        return df
        
    def _add_interaction_features(self, df: pd.DataFrame, original_features: List[str]) -> pd.DataFrame:
        """Add interaction features."""
        numeric_features = [col for col in original_features if df[col].dtype in [np.float64, np.int64]]
        
        # Limit interactions to prevent feature explosion
        selected_features = numeric_features[:min(5, len(numeric_features))]
        
        for i, col1 in enumerate(selected_features):
            for col2 in selected_features[i+1:]:
                # Multiplication
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                
                # Division (avoid division by zero)
                df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-8)
                
        return df
        
    def _add_technical_indicators(self, df: pd.DataFrame, original_features: List[str]) -> pd.DataFrame:
        """Add technical indicators if price-like data is available."""
        # Look for price-like columns
        price_cols = [col for col in original_features if any(keyword in col.lower() 
                     for keyword in ['price', 'close', 'open', 'high', 'low'])]
        
        for col in price_cols:
            # Simple moving averages
            for window in [5, 10, 20]:
                if window < len(df):
                    sma = df[col].rolling(window=window).mean()
                    df[f'{col}_sma_{window}'] = sma
                    df[f'{col}_sma_ratio_{window}'] = df[col] / (sma + 1e-8)
                    
            # Exponential moving averages
            for span in [5, 10, 20]:
                ema = df[col].ewm(span=span).mean()
                df[f'{col}_ema_{span}'] = ema
                df[f'{col}_ema_ratio_{span}'] = df[col] / (ema + 1e-8)
                
            # RSI-like indicator
            delta = df[col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-8)
            df[f'{col}_rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            sma_20 = df[col].rolling(window=20).mean()
            std_20 = df[col].rolling(window=20).std()
            df[f'{col}_bb_upper'] = sma_20 + (std_20 * 2)
            df[f'{col}_bb_lower'] = sma_20 - (std_20 * 2)
            df[f'{col}_bb_width'] = (df[f'{col}_bb_upper'] - df[f'{col}_bb_lower']) / (sma_20 + 1e-8)
            df[f'{col}_bb_position'] = (df[col] - df[f'{col}_bb_lower']) / (df[f'{col}_bb_upper'] - df[f'{col}_bb_lower'] + 1e-8)
            
        return df
        
    def _select_top_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select top features based on variance."""
        # Calculate variance for each feature
        variances = df.var().sort_values(ascending=False)
        
        # Select top features by variance
        top_features = variances.head(self.max_features).index.tolist()
        self.selected_features_ = top_features
        
        return df[top_features]
        
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        return self.feature_names_

class AutoModelSelector:
    """Automated model selection and evaluation."""
    
    def __init__(self, task_type: str = 'regression', cv_folds: int = 5,
                 scoring: str = 'neg_mean_squared_error', random_state: int = 42):
        self.task_type = task_type
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.random_state = random_state
        self.models = {}
        self.model_performances = []
        
    def get_default_models(self) -> Dict[str, Any]:
        """Get default models for evaluation."""
        models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(random_state=self.random_state),
            'lasso': Lasso(random_state=self.random_state),
            'elastic_net': ElasticNet(random_state=self.random_state),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=self.random_state),
            'extra_trees': ExtraTreesRegressor(n_estimators=100, random_state=self.random_state),
            'gradient_boosting': GradientBoostingRegressor(random_state=self.random_state),
            'svr': SVR(),
            'knn': KNeighborsRegressor(),
            'decision_tree': DecisionTreeRegressor(random_state=self.random_state),
            'mlp': MLPRegressor(random_state=self.random_state, max_iter=500)
        }
        
        # Add advanced models if available
        if XGBOOST_AVAILABLE:
            models['xgboost'] = xgb.XGBRegressor(random_state=self.random_state)
            
        if LIGHTGBM_AVAILABLE:
            models['lightgbm'] = lgb.LGBMRegressor(random_state=self.random_state, verbose=-1)
            
        if CATBOOST_AVAILABLE:
            models['catboost'] = cb.CatBoostRegressor(random_state=self.random_state, verbose=False)
            
        return models
        
    def evaluate_models(self, X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       models: Dict[str, Any] = None) -> List[ModelPerformance]:
        """Evaluate multiple models."""
        if models is None:
            models = self.get_default_models()
            
        self.models = models
        performances = []
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        
        for name, model in models.items():
            try:
                print(f"Evaluating {name}...")
                
                # Training time
                start_time = datetime.now()
                model.fit(X_train, y_train)
                training_time = (datetime.now() - start_time).total_seconds()
                
                # Prediction time
                start_time = datetime.now()
                y_pred = model.predict(X_test)
                prediction_time = (datetime.now() - start_time).total_seconds()
                
                # Metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mse)
                mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
                
                # Directional accuracy
                if len(y_test) > 1:
                    actual_direction = np.sign(np.diff(y_test))
                    pred_direction = np.sign(np.diff(y_pred))
                    directional_accuracy = np.mean(actual_direction == pred_direction)
                else:
                    directional_accuracy = 0.5
                    
                # Cross-validation
                try:
                    cv_scores = []
                    for train_idx, val_idx in tscv.split(X_train):
                        X_train_cv, X_val_cv = X_train[train_idx], X_train[val_idx]
                        y_train_cv, y_val_cv = y_train[train_idx], y_train[val_idx]
                        
                        model_cv = type(model)(**model.get_params())
                        model_cv.fit(X_train_cv, y_train_cv)
                        y_pred_cv = model_cv.predict(X_val_cv)
                        
                        if self.scoring == 'neg_mean_squared_error':
                            score = -mean_squared_error(y_val_cv, y_pred_cv)
                        elif self.scoring == 'r2':
                            score = r2_score(y_val_cv, y_pred_cv)
                        else:
                            score = r2_score(y_val_cv, y_pred_cv)
                            
                        cv_scores.append(score)
                        
                    cv_mean = np.mean(cv_scores)
                    cv_std = np.std(cv_scores)
                except Exception as e:
                    print(f"Cross-validation failed for {name}: {e}")
                    cv_mean = r2
                    cv_std = 0
                    
                performance = ModelPerformance(
                    model_name=name,
                    mse=mse,
                    mae=mae,
                    r2=r2,
                    rmse=rmse,
                    mape=mape,
                    directional_accuracy=directional_accuracy,
                    training_time=training_time,
                    prediction_time=prediction_time,
                    cross_val_score=cv_mean,
                    cross_val_std=cv_std
                )
                
                performances.append(performance)
                
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
                continue
                
        self.model_performances = performances
        return performances
        
    def get_best_model(self, metric: str = 'r2') -> Tuple[str, Any]:
        """Get the best performing model."""
        if not self.model_performances:
            raise ValueError("No models have been evaluated")
            
        if metric == 'r2':
            best_performance = max(self.model_performances, key=lambda x: x.r2)
        elif metric == 'mse':
            best_performance = min(self.model_performances, key=lambda x: x.mse)
        elif metric == 'mae':
            best_performance = min(self.model_performances, key=lambda x: x.mae)
        elif metric == 'cross_val_score':
            best_performance = max(self.model_performances, key=lambda x: x.cross_val_score)
        else:
            best_performance = max(self.model_performances, key=lambda x: x.r2)
            
        best_model_name = best_performance.model_name
        best_model = self.models[best_model_name]
        
        return best_model_name, best_model

class AutoHyperparameterOptimizer:
    """Automated hyperparameter optimization."""
    
    def __init__(self, optimization_method: str = 'random_search',
                 n_trials: int = 100, cv_folds: int = 3,
                 scoring: str = 'neg_mean_squared_error', random_state: int = 42):
        self.optimization_method = optimization_method
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.random_state = random_state
        
    def get_param_grids(self) -> Dict[str, Dict[str, Any]]:
        """Get parameter grids for different models."""
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2']
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            },
            'svr': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'linear', 'poly']
            },
            'ridge': {
                'alpha': [0.1, 1, 10, 100, 1000]
            },
            'lasso': {
                'alpha': [0.001, 0.01, 0.1, 1, 10]
            },
            'elastic_net': {
                'alpha': [0.001, 0.01, 0.1, 1],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            },
            'knn': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree']
            },
            'mlp': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            }
        }
        
        # Add advanced model parameters if available
        if XGBOOST_AVAILABLE:
            param_grids['xgboost'] = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            
        if LIGHTGBM_AVAILABLE:
            param_grids['lightgbm'] = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'num_leaves': [31, 50, 100]
            }
            
        return param_grids
        
    def optimize_hyperparameters(self, model_name: str, model: Any,
                                X_train: np.ndarray, y_train: np.ndarray) -> HyperparameterResult:
        """Optimize hyperparameters for a specific model."""
        param_grids = self.get_param_grids()
        
        if model_name not in param_grids:
            print(f"No parameter grid available for {model_name}")
            return HyperparameterResult(
                model_name=model_name,
                best_params=model.get_params(),
                best_score=0,
                optimization_method='none',
                n_trials=0,
                optimization_time=0
            )
            
        param_grid = param_grids[model_name]
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        
        start_time = datetime.now()
        
        try:
            if self.optimization_method == 'grid_search':
                search = GridSearchCV(
                    model, param_grid, cv=tscv, scoring=self.scoring,
                    n_jobs=-1, verbose=0
                )
            elif self.optimization_method == 'random_search':
                search = RandomizedSearchCV(
                    model, param_grid, n_iter=self.n_trials, cv=tscv,
                    scoring=self.scoring, n_jobs=-1, verbose=0,
                    random_state=self.random_state
                )
            elif self.optimization_method == 'halving_search' and HALVING_SEARCH_AVAILABLE:
                search = HalvingRandomSearchCV(
                    model, param_grid, cv=tscv, scoring=self.scoring,
                    n_jobs=-1, verbose=0, random_state=self.random_state
                )
            else:
                # Fallback to random search
                search = RandomizedSearchCV(
                    model, param_grid, n_iter=self.n_trials, cv=tscv,
                    scoring=self.scoring, n_jobs=-1, verbose=0,
                    random_state=self.random_state
                )
                
            search.fit(X_train, y_train)
            
            optimization_time = (datetime.now() - start_time).total_seconds()
            
            return HyperparameterResult(
                model_name=model_name,
                best_params=search.best_params_,
                best_score=search.best_score_,
                optimization_method=self.optimization_method,
                n_trials=len(search.cv_results_['params']),
                optimization_time=optimization_time
            )
            
        except Exception as e:
            print(f"Hyperparameter optimization failed for {model_name}: {e}")
            optimization_time = (datetime.now() - start_time).total_seconds()
            
            return HyperparameterResult(
                model_name=model_name,
                best_params=model.get_params(),
                best_score=0,
                optimization_method='failed',
                n_trials=0,
                optimization_time=optimization_time
            )
            
    def optimize_with_optuna(self, model_name: str, model: Any,
                           X_train: np.ndarray, y_train: np.ndarray) -> HyperparameterResult:
        """Optimize hyperparameters using Optuna."""
        if not OPTUNA_AVAILABLE:
            print("Optuna not available, falling back to random search")
            return self.optimize_hyperparameters(model_name, model, X_train, y_train)
            
        def objective(trial):
            # Define parameter search space based on model
            if model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
                }
            elif model_name == 'xgboost' and XGBOOST_AVAILABLE:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
                }
            else:
                # Default parameters for other models
                return 0
                
            # Create model with suggested parameters
            model_with_params = type(model)(**{**model.get_params(), **params})
            
            # Cross-validation
            tscv = TimeSeriesSplit(n_splits=self.cv_folds)
            scores = []
            
            for train_idx, val_idx in tscv.split(X_train):
                X_train_cv, X_val_cv = X_train[train_idx], X_train[val_idx]
                y_train_cv, y_val_cv = y_train[train_idx], y_train[val_idx]
                
                model_with_params.fit(X_train_cv, y_train_cv)
                y_pred_cv = model_with_params.predict(X_val_cv)
                
                if self.scoring == 'neg_mean_squared_error':
                    score = -mean_squared_error(y_val_cv, y_pred_cv)
                else:
                    score = r2_score(y_val_cv, y_pred_cv)
                    
                scores.append(score)
                
            return np.mean(scores)
            
        start_time = datetime.now()
        
        try:
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
            
            optimization_time = (datetime.now() - start_time).total_seconds()
            
            # Get parameter importance
            param_importance = None
            try:
                param_importance = optuna.importance.get_param_importances(study)
            except:
                pass
                
            return HyperparameterResult(
                model_name=model_name,
                best_params=study.best_params,
                best_score=study.best_value,
                optimization_method='optuna',
                n_trials=len(study.trials),
                optimization_time=optimization_time,
                param_importance=param_importance
            )
            
        except Exception as e:
            print(f"Optuna optimization failed for {model_name}: {e}")
            optimization_time = (datetime.now() - start_time).total_seconds()
            
            return HyperparameterResult(
                model_name=model_name,
                best_params=model.get_params(),
                best_score=0,
                optimization_method='failed',
                n_trials=0,
                optimization_time=optimization_time
            )

class AutoEnsemble:
    """Automated ensemble creation and optimization."""
    
    def __init__(self, ensemble_method: str = 'voting', meta_learner: str = 'linear'):
        self.ensemble_method = ensemble_method
        self.meta_learner = meta_learner
        self.base_models = []
        self.ensemble_model = None
        self.weights = None
        
    def create_ensemble(self, models: Dict[str, Any], performances: List[ModelPerformance],
                       X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """Create ensemble from best performing models."""
        # Select top models based on performance
        sorted_performances = sorted(performances, key=lambda x: x.r2, reverse=True)
        top_models = sorted_performances[:min(5, len(sorted_performances))]
        
        self.base_models = [(perf.model_name, models[perf.model_name]) for perf in top_models]
        
        if self.ensemble_method == 'voting':
            return self._create_voting_ensemble(X_train, y_train)
        elif self.ensemble_method == 'stacking':
            return self._create_stacking_ensemble(X_train, y_train)
        elif self.ensemble_method == 'weighted':
            return self._create_weighted_ensemble(performances, X_train, y_train)
        else:
            return self._create_voting_ensemble(X_train, y_train)
            
    def _create_voting_ensemble(self, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """Create simple voting ensemble."""
        from sklearn.ensemble import VotingRegressor
        
        estimators = [(name, model) for name, model in self.base_models]
        
        self.ensemble_model = VotingRegressor(estimators=estimators)
        self.ensemble_model.fit(X_train, y_train)
        
        return self.ensemble_model
        
    def _create_stacking_ensemble(self, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """Create stacking ensemble."""
        try:
            from sklearn.ensemble import StackingRegressor
            
            estimators = [(name, model) for name, model in self.base_models]
            
            # Meta-learner
            if self.meta_learner == 'linear':
                final_estimator = LinearRegression()
            elif self.meta_learner == 'ridge':
                final_estimator = Ridge()
            else:
                final_estimator = LinearRegression()
                
            self.ensemble_model = StackingRegressor(
                estimators=estimators,
                final_estimator=final_estimator,
                cv=3
            )
            self.ensemble_model.fit(X_train, y_train)
            
            return self.ensemble_model
            
        except ImportError:
            print("StackingRegressor not available, using voting ensemble")
            return self._create_voting_ensemble(X_train, y_train)
            
    def _create_weighted_ensemble(self, performances: List[ModelPerformance],
                                 X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """Create weighted ensemble based on performance."""
        # Calculate weights based on R² scores
        model_r2 = {perf.model_name: perf.r2 for perf in performances}
        
        # Normalize weights
        total_r2 = sum(model_r2[name] for name, _ in self.base_models if model_r2[name] > 0)
        
        if total_r2 > 0:
            self.weights = {name: model_r2[name] / total_r2 
                          for name, _ in self.base_models if model_r2[name] > 0}
        else:
            # Equal weights if all models perform poorly
            self.weights = {name: 1.0 / len(self.base_models) for name, _ in self.base_models}
            
        # Train all base models
        for name, model in self.base_models:
            model.fit(X_train, y_train)
            
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions."""
        if self.ensemble_model is not None:
            return self.ensemble_model.predict(X)
        elif self.weights is not None:
            # Weighted prediction
            predictions = np.zeros(len(X))
            
            for name, model in self.base_models:
                if name in self.weights:
                    pred = model.predict(X)
                    predictions += self.weights[name] * pred
                    
            return predictions
        else:
            raise ValueError("Ensemble not trained")

class AutoMLStockAnalyzer:
    """Main AutoML class for comprehensive stock analysis."""
    
    def __init__(self, max_features: int = 100, cv_folds: int = 5,
                 optimization_method: str = 'random_search', n_trials: int = 50,
                 ensemble_method: str = 'voting', random_state: int = 42):
        self.max_features = max_features
        self.cv_folds = cv_folds
        self.optimization_method = optimization_method
        self.n_trials = n_trials
        self.ensemble_method = ensemble_method
        self.random_state = random_state
        
        # Components
        self.feature_engineer = AutoFeatureEngineer(max_features=max_features)
        self.model_selector = AutoModelSelector(cv_folds=cv_folds, random_state=random_state)
        self.hyperparameter_optimizer = AutoHyperparameterOptimizer(
            optimization_method=optimization_method, n_trials=n_trials,
            cv_folds=cv_folds, random_state=random_state
        )
        self.ensemble = AutoEnsemble(ensemble_method=ensemble_method)
        
        # Results
        self.results = None
        
    def analyze(self, data: Union[pd.DataFrame, np.ndarray], target_column: str = 'close',
               test_size: float = 0.2, optimize_hyperparameters: bool = True,
               create_ensemble: bool = True) -> AutoMLResults:
        """Run complete AutoML analysis."""
        print("Starting AutoML Stock Analysis...")
        
        # Data preparation
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=[target_column])
            
        print("1. Feature Engineering...")
        # Feature engineering
        featured_data = self._prepare_features(data, target_column)
        
        # Prepare target and features
        target = featured_data[target_column].values
        features = featured_data.drop(columns=[target_column])
        
        # Apply automated feature engineering
        X = self.feature_engineer.fit_transform(features)
        y = target
        
        # Train-test split (time series aware)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"   - Generated {X.shape[1]} features from {features.shape[1]} original features")
        
        print("2. Model Selection and Evaluation...")
        # Model evaluation
        model_performances = self.model_selector.evaluate_models(
            X_train, y_train, X_test, y_test
        )
        
        print(f"   - Evaluated {len(model_performances)} models")
        
        # Get best model
        best_model_name, best_model = self.model_selector.get_best_model()
        print(f"   - Best model: {best_model_name}")
        
        hyperparameter_results = []
        
        if optimize_hyperparameters:
            print("3. Hyperparameter Optimization...")
            # Optimize top 3 models
            top_models = sorted(model_performances, key=lambda x: x.r2, reverse=True)[:3]
            
            for perf in top_models:
                print(f"   - Optimizing {perf.model_name}...")
                model = self.model_selector.models[perf.model_name]
                
                if OPTUNA_AVAILABLE and perf.model_name in ['random_forest', 'xgboost']:
                    result = self.hyperparameter_optimizer.optimize_with_optuna(
                        perf.model_name, model, X_train, y_train
                    )
                else:
                    result = self.hyperparameter_optimizer.optimize_hyperparameters(
                        perf.model_name, model, X_train, y_train
                    )
                    
                hyperparameter_results.append(result)
                
                # Update model with best parameters
                if result.best_score > 0:
                    optimized_model = type(model)(**{**model.get_params(), **result.best_params})
                    optimized_model.fit(X_train, y_train)
                    self.model_selector.models[perf.model_name] = optimized_model
                    
        # Re-evaluate models after optimization
        if optimize_hyperparameters and hyperparameter_results:
            print("4. Re-evaluating optimized models...")
            model_performances = self.model_selector.evaluate_models(
                X_train, y_train, X_test, y_test, self.model_selector.models
            )
            best_model_name, best_model = self.model_selector.get_best_model()
            
        ensemble_performance = None
        
        if create_ensemble:
            print("5. Creating Ensemble...")
            try:
                ensemble_model = self.ensemble.create_ensemble(
                    self.model_selector.models, model_performances, X_train, y_train
                )
                
                # Evaluate ensemble
                y_pred_ensemble = ensemble_model.predict(X_test)
                
                ensemble_performance = ModelPerformance(
                    model_name='ensemble',
                    mse=mean_squared_error(y_test, y_pred_ensemble),
                    mae=mean_absolute_error(y_test, y_pred_ensemble),
                    r2=r2_score(y_test, y_pred_ensemble),
                    rmse=np.sqrt(mean_squared_error(y_test, y_pred_ensemble)),
                    mape=np.mean(np.abs((y_test - y_pred_ensemble) / (y_test + 1e-8))) * 100,
                    directional_accuracy=np.mean(np.sign(np.diff(y_test)) == np.sign(np.diff(y_pred_ensemble))) if len(y_test) > 1 else 0.5,
                    training_time=0,
                    prediction_time=0,
                    cross_val_score=0,
                    cross_val_std=0
                )
                
                print(f"   - Ensemble R²: {ensemble_performance.r2:.4f}")
                
            except Exception as e:
                print(f"   - Ensemble creation failed: {e}")
                
        # Feature importance analysis
        feature_importance = self._analyze_feature_importance()
        
        # Final predictions
        if ensemble_performance and ensemble_performance.r2 > max(perf.r2 for perf in model_performances):
            final_predictions = ensemble_model.predict(X_test)
            final_model = ensemble_model
            final_model_name = 'ensemble'
        else:
            final_predictions = best_model.predict(X_test)
            final_model = best_model
            final_model_name = best_model_name
            
        # Generate insights
        insights = self._generate_insights(model_performances, hyperparameter_results, 
                                         ensemble_performance, feature_importance)
        
        # Model selection summary
        model_selection_summary = {
            'total_models_evaluated': len(model_performances),
            'best_single_model': best_model_name,
            'best_single_model_r2': max(perf.r2 for perf in model_performances),
            'ensemble_r2': ensemble_performance.r2 if ensemble_performance else None,
            'final_model': final_model_name,
            'hyperparameter_optimization': optimize_hyperparameters,
            'ensemble_created': create_ensemble
        }
        
        # Feature engineering results
        feature_engineering_results = {
            'original_features': features.shape[1],
            'generated_features': X.shape[1],
            'feature_engineering_methods': [
                'statistical_features', 'lag_features', 'rolling_features',
                'polynomial_features', 'interaction_features', 'technical_indicators'
            ],
            'selected_features': len(self.feature_engineer.selected_features_) if self.feature_engineer.selected_features_ else X.shape[1]
        }
        
        self.results = AutoMLResults(
            best_model=final_model,
            best_model_name=final_model_name,
            model_performances=model_performances,
            feature_importance=feature_importance,
            hyperparameter_results=hyperparameter_results,
            ensemble_performance=ensemble_performance,
            feature_engineering_results=feature_engineering_results,
            model_selection_summary=model_selection_summary,
            predictions=final_predictions,
            actual_values=y_test,
            prediction_intervals=None,
            insights=insights
        )
        
        print("\nAutoML Analysis Complete!")
        print(f"Final Model: {final_model_name}")
        print(f"Final R²: {r2_score(y_test, final_predictions):.4f}")
        
        return self.results
        
    def _prepare_features(self, data: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Prepare basic features before automated feature engineering."""
        df = data.copy()
        
        # Basic price features if target is price-like
        if target_column in df.columns:
            df['returns'] = df[target_column].pct_change()
            
            # Simple moving averages
            for window in [5, 10, 20]:
                if len(df) > window:
                    df[f'{target_column}_ma_{window}'] = df[target_column].rolling(window=window).mean()
                    
            # Volatility
            df['volatility_10'] = df['returns'].rolling(window=10).std()
            
        # Remove NaN values
        df = df.dropna()
        
        return df
        
    def _analyze_feature_importance(self) -> List[FeatureImportance]:
        """Analyze feature importance from different models."""
        feature_importance = []
        
        # Get feature names
        feature_names = self.feature_engineer.get_feature_names_out()
        
        # Collect importance from tree-based models
        for name, model in self.model_selector.models.items():
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                for i, importance in enumerate(importances):
                    if i < len(feature_names):
                        feature_importance.append(FeatureImportance(
                            feature_name=feature_names[i],
                            importance=float(importance),
                            rank=i + 1,
                            method=name
                        ))
                        
        # Sort by importance
        feature_importance.sort(key=lambda x: x.importance, reverse=True)
        
        return feature_importance
        
    def _generate_insights(self, model_performances: List[ModelPerformance],
                          hyperparameter_results: List[HyperparameterResult],
                          ensemble_performance: Optional[ModelPerformance],
                          feature_importance: List[FeatureImportance]) -> Dict[str, Any]:
        """Generate insights from AutoML analysis."""
        insights = {
            'model_insights': {},
            'feature_insights': {},
            'performance_insights': {},
            'recommendations': []
        }
        
        # Model insights
        best_model = max(model_performances, key=lambda x: x.r2)
        worst_model = min(model_performances, key=lambda x: x.r2)
        
        insights['model_insights'] = {
            'best_performing_model': best_model.model_name,
            'best_r2_score': best_model.r2,
            'worst_performing_model': worst_model.model_name,
            'performance_spread': best_model.r2 - worst_model.r2,
            'ensemble_improvement': (ensemble_performance.r2 - best_model.r2) if ensemble_performance else 0
        }
        
        # Feature insights
        if feature_importance:
            top_features = feature_importance[:5]
            insights['feature_insights'] = {
                'most_important_feature': top_features[0].feature_name,
                'top_5_features': [f.feature_name for f in top_features],
                'feature_importance_concentration': sum(f.importance for f in top_features) / sum(f.importance for f in feature_importance)
            }
            
        # Performance insights
        avg_r2 = np.mean([p.r2 for p in model_performances])
        insights['performance_insights'] = {
            'average_r2': avg_r2,
            'model_consistency': 'high' if np.std([p.r2 for p in model_performances]) < 0.1 else 'low',
            'prediction_quality': 'excellent' if best_model.r2 > 0.8 else 'good' if best_model.r2 > 0.6 else 'moderate' if best_model.r2 > 0.4 else 'poor'
        }
        
        # Recommendations
        if best_model.r2 > 0.7:
            insights['recommendations'].append('Model performance is good for production use')
        elif best_model.r2 > 0.5:
            insights['recommendations'].append('Model performance is moderate - consider more data or features')
        else:
            insights['recommendations'].append('Model performance is poor - review data quality and feature engineering')
            
        if ensemble_performance and ensemble_performance.r2 > best_model.r2:
            insights['recommendations'].append('Ensemble model provides better performance than individual models')
            
        if len(hyperparameter_results) > 0:
            avg_improvement = np.mean([hr.best_score for hr in hyperparameter_results if hr.best_score > 0])
            if avg_improvement > 0.1:
                insights['recommendations'].append('Hyperparameter optimization significantly improved model performance')
                
        return insights
        
    def plot_results(self, save_path: str = None) -> None:
        """Plot comprehensive AutoML results."""
        if self.results is None:
            raise ValueError("No results to plot. Run analyze() first.")
            
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('AutoML Stock Analysis Results', fontsize=16)
        
        # Model performance comparison
        model_names = [p.model_name for p in self.results.model_performances]
        r2_scores = [p.r2 for p in self.results.model_performances]
        
        axes[0, 0].bar(model_names, r2_scores)
        axes[0, 0].set_title('Model R² Comparison')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Feature importance
        if self.results.feature_importance:
            top_features = self.results.feature_importance[:10]
            feature_names = [f.feature_name for f in top_features]
            importances = [f.importance for f in top_features]
            
            axes[0, 1].barh(feature_names, importances)
            axes[0, 1].set_title('Top 10 Feature Importance')
            axes[0, 1].set_xlabel('Importance')
            
        # Predictions vs Actual
        axes[0, 2].scatter(self.results.actual_values, self.results.predictions, alpha=0.6)
        axes[0, 2].plot([self.results.actual_values.min(), self.results.actual_values.max()],
                       [self.results.actual_values.min(), self.results.actual_values.max()], 'r--')
        axes[0, 2].set_title('Predictions vs Actual')
        axes[0, 2].set_xlabel('Actual Values')
        axes[0, 2].set_ylabel('Predicted Values')
        
        # Training time comparison
        training_times = [p.training_time for p in self.results.model_performances]
        axes[1, 0].bar(model_names, training_times)
        axes[1, 0].set_title('Training Time Comparison')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Cross-validation scores
        cv_scores = [p.cross_val_score for p in self.results.model_performances]
        cv_stds = [p.cross_val_std for p in self.results.model_performances]
        
        axes[1, 1].bar(model_names, cv_scores, yerr=cv_stds, capsize=5)
        axes[1, 1].set_title('Cross-Validation Scores')
        axes[1, 1].set_ylabel('CV Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Residuals plot
        residuals = self.results.actual_values - self.results.predictions
        axes[1, 2].scatter(self.results.predictions, residuals, alpha=0.6)
        axes[1, 2].axhline(y=0, color='r', linestyle='--')
        axes[1, 2].set_title('Residuals Plot')
        axes[1, 2].set_xlabel('Predicted Values')
        axes[1, 2].set_ylabel('Residuals')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def predict(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions on new data."""
        if self.results is None:
            raise ValueError("Model not trained. Run analyze() first.")
            
        # Prepare features
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
            
        # Apply same feature engineering
        X = self.feature_engineer.transform(data)
        
        # Make predictions
        return self.results.best_model.predict(X)
        
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        if self.results is None:
            raise ValueError("No results available. Run analyze() first.")
            
        return {
            'best_model': self.results.best_model_name,
            'performance_metrics': {
                'r2': r2_score(self.results.actual_values, self.results.predictions),
                'mse': mean_squared_error(self.results.actual_values, self.results.predictions),
                'mae': mean_absolute_error(self.results.actual_values, self.results.predictions)
            },
            'model_selection_summary': self.results.model_selection_summary,
            'feature_engineering_summary': self.results.feature_engineering_results,
            'insights': self.results.insights,
            'recommendations': self.results.insights.get('recommendations', [])
        }

# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    
    # Generate synthetic stock data
    n_days = len(dates)
    returns = np.random.normal(0.001, 0.02, n_days)
    prices = [100]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
        
    # Add some volume and high/low data
    volumes = np.random.lognormal(10, 0.5, n_days)
    highs = np.array(prices) * (1 + np.random.uniform(0, 0.05, n_days))
    lows = np.array(prices) * (1 - np.random.uniform(0, 0.05, n_days))
    
    sample_data = pd.DataFrame({
        'close': prices,
        'high': highs,
        'low': lows,
        'volume': volumes
    }, index=dates)
    
    # Initialize AutoML analyzer
    automl = AutoMLStockAnalyzer(
        max_features=50,
        optimization_method='random_search',
        n_trials=20,
        ensemble_method='voting'
    )
    
    print("Running AutoML analysis on sample stock data...")
    
    # Run analysis
    results = automl.analyze(
        data=sample_data,
        target_column='close',
        test_size=0.2,
        optimize_hyperparameters=True,
        create_ensemble=True
    )
    
    # Print results summary
    print("\n" + "="*50)
    print("AUTOML ANALYSIS SUMMARY")
    print("="*50)
    
    summary = automl.get_model_summary()
    print(f"Best Model: {summary['best_model']}")
    print(f"R² Score: {summary['performance_metrics']['r2']:.4f}")
    print(f"MSE: {summary['performance_metrics']['mse']:.4f}")
    print(f"MAE: {summary['performance_metrics']['mae']:.4f}")
    
    print(f"\nFeature Engineering:")
    print(f"  - Original features: {summary['feature_engineering_summary']['original_features']}")
    print(f"  - Generated features: {summary['feature_engineering_summary']['generated_features']}")
    
    print(f"\nModel Selection:")
    print(f"  - Models evaluated: {summary['model_selection_summary']['total_models_evaluated']}")
    print(f"  - Best single model R²: {summary['model_selection_summary']['best_single_model_r2']:.4f}")
    
    if summary['model_selection_summary']['ensemble_r2']:
        print(f"  - Ensemble R²: {summary['model_selection_summary']['ensemble_r2']:.4f}")
    
    print(f"\nRecommendations:")
    for rec in summary['recommendations']:
        print(f"  - {rec}")
    
    # Plot results
    try:
        automl.plot_results()
    except Exception as e:
        print(f"Plotting failed: {e}")
    
    print("\nAutoML analysis completed successfully!")