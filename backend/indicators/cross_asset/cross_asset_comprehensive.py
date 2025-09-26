from typing import Optional, Dict, Any, List, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from enum import Enum
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AssetType(Enum):
    STOCK = "stock"
    CRYPTO = "crypto"
    FOREX = "forex"
    FUTURES = "futures"
    INDEX = "index"
    CROSS_ASSET = "cross_asset"

class IndicatorCategory(Enum):
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    MACHINE_LEARNING = "machine_learning"
    STATISTICAL = "statistical"

@dataclass
class IndicatorResult:
    name: str
    values: pd.DataFrame
    metadata: Dict[str, Any]
    confidence: float
    timestamp: datetime
    asset_type: AssetType
    category: IndicatorCategory

class CrossAssetComprehensiveIndicators:
    """Comprehensive cross-asset indicators using advanced ML techniques"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
    
    def lstm_price_prediction(self, data: pd.DataFrame, lookback_window: int = 60,
                            prediction_horizon: int = 5, features: Optional[List[str]] = None) -> IndicatorResult:
        """LSTM-based price prediction (simplified implementation using statistical methods)"""
        try:
            # Since we're avoiding heavy ML dependencies, we'll use a statistical approach
            # that mimics LSTM behavior with moving averages and trend analysis
            
            if features is None:
                features = ['close', 'volume', 'high', 'low']
            
            # Ensure we have the required features
            available_features = [f for f in features if f in data.columns]
            if not available_features:
                available_features = ['close']
            
            # Create lagged features (simulating LSTM memory)
            feature_data = pd.DataFrame(index=data.index)
            
            for feature in available_features:
                for lag in range(1, lookback_window + 1, 5):  # Sample every 5 lags to reduce dimensionality
                    feature_data[f'{feature}_lag_{lag}'] = data[feature].shift(lag)
            
            # Add technical indicators as features
            feature_data['sma_20'] = data['close'].rolling(20).mean()
            feature_data['sma_50'] = data['close'].rolling(50).mean()
            feature_data['rsi'] = self._calculate_rsi(data['close'])
            feature_data['volatility'] = data['close'].rolling(20).std()
            
            # Price momentum features
            feature_data['momentum_5'] = data['close'].pct_change(5)
            feature_data['momentum_10'] = data['close'].pct_change(10)
            feature_data['momentum_20'] = data['close'].pct_change(20)
            
            # Target variable (future returns)
            target = data['close'].shift(-prediction_horizon).pct_change(prediction_horizon)
            
            # Remove NaN values
            feature_data = feature_data.dropna()
            target = target.loc[feature_data.index].dropna()
            
            # Align data
            common_index = feature_data.index.intersection(target.index)
            feature_data = feature_data.loc[common_index]
            target = target.loc[common_index]
            
            if len(feature_data) < 100:  # Need sufficient data
                raise ValueError("Insufficient data for LSTM prediction")
            
            # Split data
            split_idx = int(len(feature_data) * 0.8)
            X_train = feature_data.iloc[:split_idx]
            X_test = feature_data.iloc[split_idx:]
            y_train = target.iloc[:split_idx]
            y_test = target.iloc[split_idx:]
            
            # Scale features
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            
            # Use Random Forest as LSTM substitute (ensemble method)
            model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            train_pred = model.predict(X_train_scaled)
            test_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            
            # Create full prediction series
            predictions = pd.Series(index=data.index, dtype=float)
            predictions.loc[X_train.index] = train_pred
            predictions.loc[X_test.index] = test_pred
            
            # Convert predictions to price levels
            predicted_prices = data['close'] * (1 + predictions)
            
            # Prediction confidence based on model performance
            confidence_score = max(0, min(1, (test_r2 + 1) / 2))  # Normalize R² to 0-1
            
            # Feature importance
            feature_importance = dict(zip(X_train.columns, model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Prediction intervals (simplified)
            prediction_std = np.std(test_pred - y_test)
            upper_bound = predicted_prices + 2 * prediction_std * data['close']
            lower_bound = predicted_prices - 2 * prediction_std * data['close']
            
            result_df = pd.DataFrame({
                'actual_price': data['close'],
                'predicted_price': predicted_prices,
                'prediction_return': predictions,
                'upper_bound': upper_bound,
                'lower_bound': lower_bound,
                'prediction_error': data['close'] - predicted_prices
            }, index=data.index)
            
            return IndicatorResult(
                name="LSTM Price Prediction",
                values=result_df,
                metadata={
                    'lookback_window': lookback_window,
                    'prediction_horizon': prediction_horizon,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'features_used': available_features,
                    'top_features': top_features,
                    'model_type': 'Random Forest (LSTM substitute)',
                    'confidence_score': confidence_score,
                    'interpretation': f'Model explains {test_r2:.2%} of price variation with RMSE of {test_rmse:.4f}'
                },
                confidence=confidence_score,
                timestamp=datetime.now(),
                asset_type=AssetType.CROSS_ASSET,
                category=IndicatorCategory.MACHINE_LEARNING
            )
            
        except Exception as e:
            logger.error(f"Error in LSTM price prediction: {e}")
            return self._empty_result("LSTM Price Prediction", AssetType.CROSS_ASSET)
    
    def xgboost_ensemble(self, data: pd.DataFrame, target_column: str = 'close',
                        feature_engineering: bool = True, cross_validation: bool = True) -> IndicatorResult:
        """XGBoost ensemble model (simplified using Random Forest ensemble)"""
        try:
            # Feature engineering
            features_df = pd.DataFrame(index=data.index)
            
            # Price-based features
            features_df['price'] = data[target_column]
            features_df['returns'] = data[target_column].pct_change()
            features_df['log_returns'] = np.log(data[target_column] / data[target_column].shift(1))
            
            # Moving averages
            for window in [5, 10, 20, 50]:
                features_df[f'sma_{window}'] = data[target_column].rolling(window).mean()
                features_df[f'ema_{window}'] = data[target_column].ewm(span=window).mean()
                features_df[f'price_to_sma_{window}'] = data[target_column] / features_df[f'sma_{window}']
            
            # Volatility features
            for window in [10, 20, 30]:
                features_df[f'volatility_{window}'] = data[target_column].rolling(window).std()
                features_df[f'volatility_ratio_{window}'] = (features_df[f'volatility_{window}'] / 
                                                           features_df[f'volatility_{window}'].rolling(50).mean())
            
            # Technical indicators
            features_df['rsi'] = self._calculate_rsi(data[target_column])
            bb_upper, bb_lower = self._calculate_bollinger_bands(data[target_column])
            features_df['bb_position'] = (data[target_column] - bb_lower) / (bb_upper - bb_lower)
            features_df['bb_width'] = (bb_upper - bb_lower) / data[target_column]
            
            # Momentum features
            for period in [1, 5, 10, 20]:
                features_df[f'momentum_{period}'] = data[target_column].pct_change(period)
                features_df[f'momentum_rank_{period}'] = features_df[f'momentum_{period}'].rolling(50).rank(pct=True)
            
            # Volume features (if available)
            if 'volume' in data.columns:
                features_df['volume'] = data['volume']
                features_df['volume_sma'] = data['volume'].rolling(20).mean()
                features_df['volume_ratio'] = data['volume'] / features_df['volume_sma']
                features_df['price_volume'] = data[target_column] * data['volume']
            
            # Lagged features
            for lag in [1, 2, 3, 5, 10]:
                features_df[f'price_lag_{lag}'] = data[target_column].shift(lag)
                features_df[f'returns_lag_{lag}'] = features_df['returns'].shift(lag)
            
            # Target variable (next period return)
            target = data[target_column].shift(-1).pct_change()
            
            # Remove NaN values
            features_df = features_df.dropna()
            target = target.loc[features_df.index].dropna()
            
            # Align data
            common_index = features_df.index.intersection(target.index)
            features_df = features_df.loc[common_index]
            target = target.loc[common_index]
            
            if len(features_df) < 100:
                raise ValueError("Insufficient data for XGBoost ensemble")
            
            # Split data
            split_idx = int(len(features_df) * 0.8)
            X_train = features_df.iloc[:split_idx]
            X_test = features_df.iloc[split_idx:]
            y_train = target.iloc[:split_idx]
            y_test = target.iloc[split_idx:]
            
            # Scale features
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            
            # Ensemble of Random Forest models (XGBoost substitute)
            models = []
            predictions_train = []
            predictions_test = []
            
            # Create ensemble with different hyperparameters
            model_configs = [
                {'n_estimators': 100, 'max_depth': 8, 'random_state': 42},
                {'n_estimators': 150, 'max_depth': 10, 'random_state': 43},
                {'n_estimators': 200, 'max_depth': 6, 'random_state': 44}
            ]
            
            for config in model_configs:
                model = RandomForestRegressor(**config)
                model.fit(X_train_scaled, y_train)
                models.append(model)
                
                predictions_train.append(model.predict(X_train_scaled))
                predictions_test.append(model.predict(X_test_scaled))
            
            # Ensemble predictions (average)
            ensemble_train_pred = np.mean(predictions_train, axis=0)
            ensemble_test_pred = np.mean(predictions_test, axis=0)
            
            # Calculate metrics
            train_r2 = r2_score(y_train, ensemble_train_pred)
            test_r2 = r2_score(y_test, ensemble_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, ensemble_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, ensemble_test_pred))
            
            # Feature importance (average across models)
            feature_importance = {}
            for feature in X_train.columns:
                importance_scores = [model.feature_importances_[list(X_train.columns).index(feature)] 
                                   for model in models]
                feature_importance[feature] = np.mean(importance_scores)
            
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:15]
            
            # Create full prediction series
            predictions = pd.Series(index=data.index, dtype=float)
            predictions.loc[X_train.index] = ensemble_train_pred
            predictions.loc[X_test.index] = ensemble_test_pred
            
            # Convert to price predictions
            predicted_prices = data[target_column] * (1 + predictions)
            
            # Confidence score
            confidence_score = max(0, min(1, (test_r2 + 1) / 2))
            
            # Prediction intervals
            prediction_std = np.std(ensemble_test_pred - y_test)
            upper_bound = predicted_prices + 2 * prediction_std * data[target_column]
            lower_bound = predicted_prices - 2 * prediction_std * data[target_column]
            
            result_df = pd.DataFrame({
                'actual_price': data[target_column],
                'predicted_return': predictions,
                'predicted_price': predicted_prices,
                'upper_bound': upper_bound,
                'lower_bound': lower_bound,
                'prediction_error': predictions - target
            }, index=data.index)
            
            return IndicatorResult(
                name="XGBoost Ensemble",
                values=result_df,
                metadata={
                    'ensemble_size': len(models),
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'top_features': top_features,
                    'feature_count': len(features_df.columns),
                    'model_type': 'Random Forest Ensemble (XGBoost substitute)',
                    'confidence_score': confidence_score,
                    'interpretation': f'Ensemble model with {test_r2:.2%} accuracy and {test_rmse:.4f} RMSE'
                },
                confidence=confidence_score,
                timestamp=datetime.now(),
                asset_type=AssetType.CROSS_ASSET,
                category=IndicatorCategory.MACHINE_LEARNING
            )
            
        except Exception as e:
            logger.error(f"Error in XGBoost ensemble: {e}")
            return self._empty_result("XGBoost Ensemble", AssetType.CROSS_ASSET)
    
    def multi_asset_correlation_analysis(self, asset_data: Dict[str, pd.DataFrame],
                                       window: int = 252, min_periods: int = 50) -> IndicatorResult:
        """Multi-asset correlation analysis across different asset classes"""
        try:
            # Extract price series for each asset
            price_series = {}
            for asset_name, data in asset_data.items():
                if 'close' in data.columns:
                    price_series[asset_name] = data['close']
                else:
                    # Use first numeric column if 'close' not available
                    numeric_cols = data.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        price_series[asset_name] = data[numeric_cols[0]]
            
            if len(price_series) < 2:
                raise ValueError("Need at least 2 assets for correlation analysis")
            
            # Combine into single DataFrame
            combined_df = pd.DataFrame(price_series)
            
            # Calculate returns
            returns_df = combined_df.pct_change().dropna()
            
            # Rolling correlation matrix
            rolling_corr = {}
            asset_names = list(returns_df.columns)
            
            for i, asset1 in enumerate(asset_names):
                for j, asset2 in enumerate(asset_names):
                    if i < j:  # Avoid duplicate pairs
                        pair_name = f"{asset1}_{asset2}"
                        rolling_corr[pair_name] = returns_df[asset1].rolling(
                            window=window, min_periods=min_periods
                        ).corr(returns_df[asset2])
            
            # Static correlation matrix (full period)
            static_corr = returns_df.corr()
            
            # Correlation stability analysis
            corr_stability = {}
            for pair, corr_series in rolling_corr.items():
                corr_stability[pair] = {
                    'mean': corr_series.mean(),
                    'std': corr_series.std(),
                    'min': corr_series.min(),
                    'max': corr_series.max(),
                    'current': corr_series.iloc[-1] if not corr_series.empty else np.nan
                }
            
            # Diversification ratio
            portfolio_weights = np.ones(len(asset_names)) / len(asset_names)  # Equal weights
            portfolio_variance = np.dot(portfolio_weights, np.dot(static_corr, portfolio_weights))
            avg_individual_variance = np.mean([returns_df[asset].var() for asset in asset_names])
            diversification_ratio = avg_individual_variance / portfolio_variance if portfolio_variance > 0 else 1
            
            # Create result DataFrame with rolling correlations
            result_df = pd.DataFrame(rolling_corr, index=returns_df.index)
            
            # Add portfolio metrics
            result_df['diversification_ratio'] = diversification_ratio
            result_df['portfolio_volatility'] = returns_df.dot(portfolio_weights).rolling(window).std() * np.sqrt(252)
            
            # Correlation regime detection (simplified)
            avg_correlation = result_df[list(rolling_corr.keys())].mean(axis=1)
            high_corr_threshold = avg_correlation.quantile(0.75)
            low_corr_threshold = avg_correlation.quantile(0.25)
            
            correlation_regime = pd.Series(index=avg_correlation.index, dtype=str)
            correlation_regime[avg_correlation >= high_corr_threshold] = 'High Correlation'
            correlation_regime[avg_correlation <= low_corr_threshold] = 'Low Correlation'
            correlation_regime[(avg_correlation > low_corr_threshold) & 
                             (avg_correlation < high_corr_threshold)] = 'Medium Correlation'
            
            result_df['avg_correlation'] = avg_correlation
            result_df['correlation_regime'] = correlation_regime
            
            return IndicatorResult(
                name="Multi-Asset Correlation Analysis",
                values=result_df,
                metadata={
                    'assets_analyzed': asset_names,
                    'correlation_window': window,
                    'static_correlation_matrix': static_corr.to_dict(),
                    'correlation_stability': corr_stability,
                    'diversification_ratio': diversification_ratio,
                    'current_avg_correlation': avg_correlation.iloc[-1] if not avg_correlation.empty else np.nan,
                    'current_regime': correlation_regime.iloc[-1] if not correlation_regime.empty else 'Unknown',
                    'interpretation': f'Diversification ratio of {diversification_ratio:.2f} with current correlation regime'
                },
                confidence=0.80,
                timestamp=datetime.now(),
                asset_type=AssetType.CROSS_ASSET,
                category=IndicatorCategory.STATISTICAL
            )
            
        except Exception as e:
            logger.error(f"Error in multi-asset correlation analysis: {e}")
            return self._empty_result("Multi-Asset Correlation Analysis", AssetType.CROSS_ASSET)
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, lower_band
    
    def _empty_result(self, name: str, asset_type: AssetType) -> IndicatorResult:
        """Return empty result for error cases"""
        return IndicatorResult(
            name=name,
            values=pd.DataFrame(),
            metadata={'error': 'Calculation failed'},
            confidence=0.0,
            timestamp=datetime.now(),
            asset_type=asset_type,
            category=IndicatorCategory.TECHNICAL
        )
    
    async def calculate_indicator(self, indicator_name: str, data: pd.DataFrame, **kwargs) -> IndicatorResult:
        """Calculate specific cross-asset indicator based on name"""
        indicator_map = {
            'lstm_prediction': self.lstm_price_prediction,
            'xgboost_ensemble': self.xgboost_ensemble,
            'correlation_analysis': self.multi_asset_correlation_analysis,
        }
        
        if indicator_name not in indicator_map:
            raise ValueError(f"Unknown cross-asset indicator: {indicator_name}")
        
        return indicator_map[indicator_name](data, **kwargs)