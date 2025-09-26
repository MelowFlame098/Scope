"""XGBoost Model for Forex Prediction

This module implements XGBoost (Extreme Gradient Boosting) for forex price prediction,
including quantile regression for uncertainty estimation and cross-validation.
"""

import numpy as np
import pandas as pd
from typing import List, Dict
from dataclasses import dataclass
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

@dataclass
class MLPrediction:
    """Machine learning prediction with uncertainty bounds"""
    value: float
    confidence: float
    lower_bound: float
    upper_bound: float
    probability_up: float
    probability_down: float

@dataclass
class XGBoostResults:
    """XGBoost model results"""
    predictions: List[MLPrediction]
    model_performance: Dict[str, float]
    feature_importance: Dict[str, float]
    shap_values: np.ndarray
    model_parameters: Dict[str, any]
    cross_validation_scores: List[float]

class XGBoostForexModel:
    """XGBoost model for forex prediction"""
    
    def __init__(self):
        self.model = None
        self.feature_names = []
        
    def fit(self, features: pd.DataFrame, target: pd.Series) -> XGBoostResults:
        """Train XGBoost model"""
        # Prepare data
        X = features.fillna(method='ffill').fillna(0)
        y = target.fillna(method='ffill')
        
        self.feature_names = X.columns.tolist()
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # XGBoost parameters
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        # Train model
        self.model = xgb.XGBRegressor(**params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        # Make predictions
        predictions = self.model.predict(X_test)
        
        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X_train):
            X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            cv_model = xgb.XGBRegressor(**params)
            cv_model.fit(X_cv_train, y_cv_train, verbose=False)
            cv_pred = cv_model.predict(X_cv_val)
            cv_scores.append(mean_squared_error(y_cv_val, cv_pred))
        
        # Create prediction objects with uncertainty estimation
        ml_predictions = []
        
        # Estimate prediction uncertainty using quantile regression
        quantile_models = {}
        for alpha in [0.1, 0.9]:  # 80% prediction interval
            quantile_params = params.copy()
            quantile_params['objective'] = f'reg:quantileerror'
            quantile_params['quantile_alpha'] = alpha
            
            q_model = xgb.XGBRegressor(**quantile_params)
            q_model.fit(X_train, y_train, verbose=False)
            quantile_models[alpha] = q_model
        
        lower_bounds = quantile_models[0.1].predict(X_test)
        upper_bounds = quantile_models[0.9].predict(X_test)
        
        for i, pred in enumerate(predictions):
            # Calculate confidence based on prediction interval width
            interval_width = upper_bounds[i] - lower_bounds[i]
            confidence = max(0.1, 1.0 - (interval_width / abs(pred + 1e-8)))
            confidence = min(confidence, 0.95)
            
            # Direction probability based on recent trend
            if i > 0:
                recent_change = pred - predictions[i-1]
                prob_up = 0.5 + np.tanh(recent_change * 10) * 0.3
            else:
                prob_up = 0.5
            
            ml_predictions.append(MLPrediction(
                value=pred,
                confidence=confidence,
                lower_bound=lower_bounds[i],
                upper_bound=upper_bounds[i],
                probability_up=prob_up,
                probability_down=1 - prob_up
            ))
        
        # Performance metrics
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        
        # Direction accuracy
        actual_directions = (y_test.diff() > 0).astype(int)[1:]
        pred_directions = (np.diff(predictions) > 0).astype(int)
        direction_accuracy = accuracy_score(actual_directions, pred_directions)
        
        performance = {
            'mse': mse,
            'mae': mae,
            'rmse': np.sqrt(mse),
            'direction_accuracy': direction_accuracy,
            'r2_score': self.model.score(X_test, y_test),
            'cv_mean_mse': np.mean(cv_scores),
            'cv_std_mse': np.std(cv_scores)
        }
        
        # Feature importance
        feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        
        return XGBoostResults(
            predictions=ml_predictions,
            model_performance=performance,
            feature_importance=feature_importance,
            shap_values=None,  # Would need SHAP library
            model_parameters=params,
            cross_validation_scores=cv_scores
        )
    
    def predict(self, features: pd.DataFrame) -> List[MLPrediction]:
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        X = features.fillna(method='ffill').fillna(0)
        predictions = self.model.predict(X)
        
        # Simple uncertainty estimation for new predictions
        ml_predictions = []
        for i, pred in enumerate(predictions):
            # Basic confidence estimation
            confidence = 0.7  # Default confidence
            
            # Simple bounds estimation
            std_error = abs(pred) * 0.05  # 5% of prediction as error estimate
            lower_bound = pred - 1.96 * std_error
            upper_bound = pred + 1.96 * std_error
            
            # Direction probability
            prob_up = 0.6 if i > 0 and pred > predictions[i-1] else 0.4
            
            ml_predictions.append(MLPrediction(
                value=pred,
                confidence=confidence,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                probability_up=prob_up,
                probability_down=1 - prob_up
            ))
        
        return ml_predictions
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model"""
        if self.model is None:
            raise ValueError("Model must be trained before getting feature importance")
        
        return dict(zip(self.feature_names, self.model.feature_importances_))
    
    def plot_feature_importance(self, top_n: int = 10):
        """Plot feature importance"""
        try:
            import matplotlib.pyplot as plt
            
            importance = self.get_feature_importance()
            
            # Sort by importance
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
            features, importances = zip(*sorted_features)
            
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(features)), importances)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Feature Importance')
            plt.title(f'Top {top_n} Feature Importance (XGBoost)')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for plotting")

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n_points = 200
    
    # Create sample features
    features = pd.DataFrame({
        'price_lag1': np.random.randn(n_points),
        'volume': np.random.exponential(1, n_points),
        'volatility': np.random.exponential(0.1, n_points),
        'rsi': np.random.uniform(20, 80, n_points),
        'macd': np.random.randn(n_points) * 0.01,
        'interest_rate_diff': np.random.randn(n_points) * 0.005,
        'inflation_diff': np.random.randn(n_points) * 0.01,
        'gdp_growth': np.random.randn(n_points) * 0.02
    })
    
    # Create sample target (exchange rate)
    target = pd.Series(1.2 + np.cumsum(np.random.randn(n_points) * 0.01))
    
    # Initialize and train model
    xgb_model = XGBoostForexModel()
    
    try:
        print("Training XGBoost model...")
        results = xgb_model.fit(features, target)
        
        print("\n=== XGBoost Model Results ===")
        print(f"MSE: {results.model_performance['mse']:.6f}")
        print(f"MAE: {results.model_performance['mae']:.6f}")
        print(f"RMSE: {results.model_performance['rmse']:.6f}")
        print(f"Direction Accuracy: {results.model_performance['direction_accuracy']:.3f}")
        print(f"R² Score: {results.model_performance['r2_score']:.3f}")
        print(f"CV Mean MSE: {results.model_performance['cv_mean_mse']:.6f}")
        print(f"CV Std MSE: {results.model_performance['cv_std_mse']:.6f}")
        
        print(f"\nNumber of predictions: {len(results.predictions)}")
        if results.predictions:
            avg_confidence = np.mean([p.confidence for p in results.predictions])
            print(f"Average confidence: {avg_confidence:.3f}")
        
        # Show top features
        print("\nTop 5 Most Important Features:")
        top_features = sorted(results.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        for feature, importance in top_features:
            print(f"  {feature}: {importance:.4f}")
        
        print("\nModel training completed successfully!")
        
        # Test prediction on new data
        new_features = features.tail(10)  # Use last 10 rows as new data
        new_predictions = xgb_model.predict(new_features)
        print(f"\nMade {len(new_predictions)} new predictions")
        
    except Exception as e:
        print(f"Model training failed: {e}")
        import traceback
        traceback.print_exc()