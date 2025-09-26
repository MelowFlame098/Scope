"""XGBoost Model for Stock Price Prediction

This module implements XGBoost regression for stock price prediction
with feature importance analysis, cross-validation, and uncertainty estimation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# XGBoost import (with fallback)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Using LightGBM or RandomForest as fallback.")
    
# Alternative ML libraries
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Suppress warnings
warnings.filterwarnings('ignore')

@dataclass
class ModelPrediction:
    """Prediction with confidence intervals."""
    prediction: float
    lower_bound: float
    upper_bound: float
    confidence: float

@dataclass
class XGBoostResults:
    """Results from XGBoost model."""
    predictions: np.ndarray
    actual_values: np.ndarray
    feature_importance: Dict[str, float]
    cross_val_scores: np.ndarray
    prediction_intervals: List[ModelPrediction]
    mse: float
    mae: float
    r2: float
    directional_accuracy: float
    model_params: Dict[str, Any]
    cv_mean: float
    cv_std: float

class XGBoostModel:
    """XGBoost model for stock price prediction."""
    
    def __init__(self, params: Dict[str, Any] = None):
        self.params = params or {
            'objective': 'reg:squarederror',
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        self.model = None
        self.feature_names = None
        self.scaler = StandardScaler()
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray,
             feature_names: List[str] = None) -> XGBoostResults:
        """Train XGBoost model with cross-validation and uncertainty estimation."""
        
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X_train.shape[1])]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        if XGBOOST_AVAILABLE:
            self.model = xgb.XGBRegressor(**self.params)
        elif LIGHTGBM_AVAILABLE:
            # Convert XGBoost params to LightGBM params
            lgb_params = {
                'objective': 'regression',
                'n_estimators': self.params.get('n_estimators', 100),
                'max_depth': self.params.get('max_depth', 6),
                'learning_rate': self.params.get('learning_rate', 0.1),
                'subsample': self.params.get('subsample', 0.8),
                'colsample_bytree': self.params.get('colsample_bytree', 0.8),
                'random_state': self.params.get('random_state', 42),
                'verbose': -1
            }
            self.model = lgb.LGBMRegressor(**lgb_params)
        else:
            # Fallback to RandomForest
            rf_params = {
                'n_estimators': self.params.get('n_estimators', 100),
                'max_depth': self.params.get('max_depth', 6),
                'random_state': self.params.get('random_state', 42)
            }
            self.model = RandomForestRegressor(**rf_params)
            
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Cross-validation
        cv_scores = self._cross_validate(X_train_scaled, y_train)
        
        # Predictions
        val_pred = self.model.predict(X_val_scaled)
        
        # Feature importance
        feature_importance = self._get_feature_importance()
        
        # Prediction intervals using quantile regression
        prediction_intervals = self._estimate_prediction_intervals(X_val_scaled, y_val)
        
        # Metrics
        mse = mean_squared_error(y_val, val_pred)
        mae = mean_absolute_error(y_val, val_pred)
        r2 = r2_score(y_val, val_pred)
        
        # Directional accuracy
        val_direction = np.sign(np.diff(y_val))
        pred_direction = np.sign(np.diff(val_pred))
        directional_accuracy = np.mean(val_direction == pred_direction) if len(val_direction) > 0 else 0.0
        
        return XGBoostResults(
            predictions=val_pred,
            actual_values=y_val,
            feature_importance=feature_importance,
            cross_val_scores=cv_scores,
            prediction_intervals=prediction_intervals,
            mse=mse,
            mae=mae,
            r2=r2,
            directional_accuracy=directional_accuracy,
            model_params=self.params,
            cv_mean=np.mean(cv_scores),
            cv_std=np.std(cv_scores)
        )
        
    def _cross_validate(self, X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> np.ndarray:
        """Perform time series cross-validation."""
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        if XGBOOST_AVAILABLE:
            model = xgb.XGBRegressor(**self.params)
        elif LIGHTGBM_AVAILABLE:
            lgb_params = {
                'objective': 'regression',
                'n_estimators': self.params.get('n_estimators', 100),
                'max_depth': self.params.get('max_depth', 6),
                'learning_rate': self.params.get('learning_rate', 0.1),
                'subsample': self.params.get('subsample', 0.8),
                'colsample_bytree': self.params.get('colsample_bytree', 0.8),
                'random_state': self.params.get('random_state', 42),
                'verbose': -1
            }
            model = lgb.LGBMRegressor(**lgb_params)
        else:
            rf_params = {
                'n_estimators': self.params.get('n_estimators', 100),
                'max_depth': self.params.get('max_depth', 6),
                'random_state': self.params.get('random_state', 42)
            }
            model = RandomForestRegressor(**rf_params)
            
        scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
        return -scores  # Convert back to positive MSE
        
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model."""
        if hasattr(self.model, 'feature_importances_'):
            importance_values = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance_values = np.abs(self.model.coef_)
        else:
            # Fallback: all features equally important
            importance_values = np.ones(len(self.feature_names)) / len(self.feature_names)
            
        # Normalize importance values
        importance_values = importance_values / np.sum(importance_values)
        
        return dict(zip(self.feature_names, importance_values))
        
    def _estimate_prediction_intervals(self, X: np.ndarray, y_true: np.ndarray,
                                     confidence_levels: List[float] = None) -> List[ModelPrediction]:
        """Estimate prediction intervals using quantile regression."""
        if confidence_levels is None:
            confidence_levels = [0.8, 0.9, 0.95]
            
        predictions = []
        base_pred = self.model.predict(X)
        
        for conf_level in confidence_levels:
            alpha = 1 - conf_level
            lower_quantile = alpha / 2
            upper_quantile = 1 - alpha / 2
            
            # Train quantile regressors for bounds
            try:
                if XGBOOST_AVAILABLE:
                    # Lower bound model
                    lower_params = self.params.copy()
                    lower_params['objective'] = f'reg:quantileerror'
                    lower_params['quantile_alpha'] = lower_quantile
                    lower_model = xgb.XGBRegressor(**lower_params)
                    
                    # Upper bound model
                    upper_params = self.params.copy()
                    upper_params['objective'] = f'reg:quantileerror'
                    upper_params['quantile_alpha'] = upper_quantile
                    upper_model = xgb.XGBRegressor(**upper_params)
                    
                    # Fit models (using training data from the main model)
                    # Note: This is a simplified approach; in practice, you'd want to retrain
                    lower_model.fit(X, y_true)
                    upper_model.fit(X, y_true)
                    
                    lower_bound = lower_model.predict(X)
                    upper_bound = upper_model.predict(X)
                    
                else:
                    # Fallback: use prediction residuals to estimate intervals
                    residuals = y_true - base_pred
                    residual_std = np.std(residuals)
                    
                    # Use normal distribution approximation
                    from scipy import stats
                    z_score = stats.norm.ppf(upper_quantile)
                    
                    lower_bound = base_pred - z_score * residual_std
                    upper_bound = base_pred + z_score * residual_std
                    
            except Exception:
                # Simple fallback using residual standard deviation
                residuals = y_true - base_pred
                residual_std = np.std(residuals)
                
                # Use t-distribution for small samples
                from scipy import stats
                df = len(y_true) - 1
                t_score = stats.t.ppf(upper_quantile, df)
                
                lower_bound = base_pred - t_score * residual_std
                upper_bound = base_pred + t_score * residual_std
                
            # Create prediction objects
            for i, pred in enumerate(base_pred):
                predictions.append(ModelPrediction(
                    prediction=pred,
                    lower_bound=lower_bound[i],
                    upper_bound=upper_bound[i],
                    confidence=conf_level
                ))
                
        return predictions
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
            
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
        
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """Get top N most important features."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
            
        importance = self._get_feature_importance()
        
        # Sort by importance and return top N
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_importance[:top_n])
        
    def plot_feature_importance(self, top_n: int = 20):
        """Plot feature importance."""
        try:
            import matplotlib.pyplot as plt
            
            importance = self.get_feature_importance(top_n)
            
            features = list(importance.keys())
            values = list(importance.values())
            
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(features)), values)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Feature Importance')
            plt.title(f'Top {top_n} Feature Importance')
            plt.gca().invert_yaxis()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for plotting")
            
    def plot_predictions(self, results: XGBoostResults):
        """Plot predictions vs actual values."""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(15, 10))
            
            # Predictions vs Actual
            plt.subplot(2, 2, 1)
            plt.scatter(results.actual_values, results.predictions, alpha=0.6)
            plt.plot([results.actual_values.min(), results.actual_values.max()],
                    [results.actual_values.min(), results.actual_values.max()], 'r--')
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title('Predictions vs Actual')
            plt.grid(True, alpha=0.3)
            
            # Residuals
            plt.subplot(2, 2, 2)
            residuals = results.actual_values - results.predictions
            plt.scatter(results.predictions, residuals, alpha=0.6)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title('Residual Plot')
            plt.grid(True, alpha=0.3)
            
            # Cross-validation scores
            plt.subplot(2, 2, 3)
            plt.bar(range(len(results.cross_val_scores)), results.cross_val_scores)
            plt.xlabel('CV Fold')
            plt.ylabel('MSE Score')
            plt.title('Cross-Validation Scores')
            plt.grid(True, alpha=0.3)
            
            # Feature importance (top 10)
            plt.subplot(2, 2, 4)
            importance = dict(list(results.feature_importance.items())[:10])
            features = list(importance.keys())
            values = list(importance.values())
            
            plt.barh(range(len(features)), values)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Importance')
            plt.title('Top 10 Feature Importance')
            plt.gca().invert_yaxis()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for plotting")
            
    def plot_prediction_intervals(self, results: XGBoostResults, sample_size: int = 100):
        """Plot prediction intervals."""
        try:
            import matplotlib.pyplot as plt
            
            # Sample data for visualization
            if len(results.predictions) > sample_size:
                indices = np.random.choice(len(results.predictions), sample_size, replace=False)
                indices = np.sort(indices)
            else:
                indices = np.arange(len(results.predictions))
                
            actual_sample = results.actual_values[indices]
            pred_sample = results.predictions[indices]
            
            # Get prediction intervals for the sample
            intervals_sample = [results.prediction_intervals[i] for i in indices]
            
            plt.figure(figsize=(12, 8))
            
            # Plot actual vs predicted
            plt.scatter(range(len(indices)), actual_sample, label='Actual', alpha=0.7, color='blue')
            plt.scatter(range(len(indices)), pred_sample, label='Predicted', alpha=0.7, color='red')
            
            # Plot confidence intervals
            if intervals_sample:
                confidence_levels = list(set([interval.confidence for interval in intervals_sample]))
                colors = ['lightblue', 'lightgreen', 'lightyellow']
                
                for i, conf_level in enumerate(sorted(confidence_levels, reverse=True)):
                    conf_intervals = [interval for interval in intervals_sample if interval.confidence == conf_level]
                    if conf_intervals:
                        lower_bounds = [interval.lower_bound for interval in conf_intervals]
                        upper_bounds = [interval.upper_bound for interval in conf_intervals]
                        
                        plt.fill_between(range(len(indices)), lower_bounds, upper_bounds,
                                       alpha=0.3, color=colors[i % len(colors)],
                                       label=f'{conf_level*100:.0f}% Confidence Interval')
            
            plt.xlabel('Sample Index')
            plt.ylabel('Value')
            plt.title('Predictions with Confidence Intervals')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for plotting")

# Example usage
if __name__ == "__main__":
    # Generate sample stock data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    # Create synthetic features
    X = np.random.randn(n_samples, n_features)
    
    # Create target with some relationship to features
    true_coeffs = np.random.randn(n_features) * 0.5
    y = X @ true_coeffs + np.random.randn(n_samples) * 0.1
    
    # Add trend
    trend = np.linspace(0, 10, n_samples)
    y += trend
    
    # Feature names
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    print(f"Generated dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Train-test split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train XGBoost model
    xgb_model = XGBoostModel()
    
    try:
        print("Training XGBoost model...")
        results = xgb_model.train(X_train, y_train, X_test, y_test, feature_names)
        
        print("\n=== XGBoost Results ===")
        print(f"MSE: {results.mse:.6f}")
        print(f"MAE: {results.mae:.6f}")
        print(f"R²: {results.r2:.6f}")
        print(f"Directional Accuracy: {results.directional_accuracy:.6f}")
        print(f"Cross-validation Mean: {results.cv_mean:.6f} ± {results.cv_std:.6f}")
        
        # Show top features
        print("\nTop 10 Most Important Features:")
        top_features = xgb_model.get_feature_importance(10)
        for feature, importance in top_features.items():
            print(f"{feature}: {importance:.4f}")
            
        # Plot results
        xgb_model.plot_predictions(results)
        xgb_model.plot_feature_importance()
        xgb_model.plot_prediction_intervals(results)
        
        print("\nXGBoost model training completed successfully!")
        
    except Exception as e:
        print(f"XGBoost model training failed: {e}")
        import traceback
        traceback.print_exc()