import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import talib
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TechnicalSignal:
    """Data class for technical analysis signals"""
    indicator: str
    signal: str  # 'BUY', 'SELL', 'HOLD'
    strength: float  # 0-1 confidence score
    value: float
    timestamp: datetime
    metadata: Dict[str, Any] = None

@dataclass
class MLPrediction:
    """Data class for ML predictions"""
    symbol: str
    prediction_type: str  # 'price', 'direction', 'volatility'
    predicted_value: float
    confidence: float
    horizon: str  # '1h', '1d', '1w', etc.
    features_used: List[str]
    model_name: str
    timestamp: datetime

class EnhancedTechnicalIndicators:
    """Advanced technical indicators with machine learning enhancement"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = {}
        
    def calculate_traditional_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate traditional technical indicators"""
        try:
            # Ensure we have required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            # Convert to numpy arrays for talib
            open_prices = df['open'].values
            high_prices = df['high'].values
            low_prices = df['low'].values
            close_prices = df['close'].values
            volume = df['volume'].values
            
            # Moving Averages
            df['sma_20'] = talib.SMA(close_prices, timeperiod=20)
            df['sma_50'] = talib.SMA(close_prices, timeperiod=50)
            df['sma_200'] = talib.SMA(close_prices, timeperiod=200)
            df['ema_12'] = talib.EMA(close_prices, timeperiod=12)
            df['ema_26'] = talib.EMA(close_prices, timeperiod=26)
            
            # MACD
            df['macd'], df['macd_signal'], df['macd_histogram'] = talib.MACD(close_prices)
            
            # RSI
            df['rsi'] = talib.RSI(close_prices, timeperiod=14)
            
            # Bollinger Bands
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(close_prices, timeperiod=20)
            
            # Stochastic
            df['stoch_k'], df['stoch_d'] = talib.STOCH(high_prices, low_prices, close_prices)
            
            # Williams %R
            df['williams_r'] = talib.WILLR(high_prices, low_prices, close_prices, timeperiod=14)
            
            # Average True Range
            df['atr'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
            
            # Commodity Channel Index
            df['cci'] = talib.CCI(high_prices, low_prices, close_prices, timeperiod=14)
            
            # Money Flow Index
            df['mfi'] = talib.MFI(high_prices, low_prices, close_prices, volume, timeperiod=14)
            
            # On Balance Volume
            df['obv'] = talib.OBV(close_prices, volume)
            
            # Parabolic SAR
            df['sar'] = talib.SAR(high_prices, low_prices)
            
            # Aroon
            df['aroon_up'], df['aroon_down'] = talib.AROON(high_prices, low_prices, timeperiod=14)
            
            # ADX
            df['adx'] = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating traditional indicators: {e}")
            return df

    def calculate_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced custom indicators"""
        try:
            # Price momentum indicators
            df['price_momentum_5'] = df['close'].pct_change(5)
            df['price_momentum_10'] = df['close'].pct_change(10)
            df['price_momentum_20'] = df['close'].pct_change(20)
            
            # Volume indicators
            df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
            df['price_volume'] = df['close'] * df['volume']
            
            # Volatility indicators
            df['volatility_10'] = df['close'].rolling(window=10).std()
            df['volatility_20'] = df['close'].rolling(window=20).std()
            df['volatility_ratio'] = df['volatility_10'] / df['volatility_20']
            
            # Support and Resistance levels
            df['support_20'] = df['low'].rolling(window=20).min()
            df['resistance_20'] = df['high'].rolling(window=20).max()
            df['support_distance'] = (df['close'] - df['support_20']) / df['close']
            df['resistance_distance'] = (df['resistance_20'] - df['close']) / df['close']
            
            # Trend strength
            df['trend_strength'] = np.where(
                df['close'] > df['sma_20'], 1,
                np.where(df['close'] < df['sma_20'], -1, 0)
            )
            
            # Gap analysis
            df['gap'] = df['open'] - df['close'].shift(1)
            df['gap_percent'] = df['gap'] / df['close'].shift(1)
            
            # Candlestick patterns (simplified)
            df['doji'] = np.where(
                abs(df['close'] - df['open']) / (df['high'] - df['low']) < 0.1, 1, 0
            )
            df['hammer'] = np.where(
                (df['close'] > df['open']) & 
                ((df['open'] - df['low']) > 2 * (df['close'] - df['open'])) &
                ((df['high'] - df['close']) < 0.1 * (df['close'] - df['open'])), 1, 0
            )
            
            # Market structure
            df['higher_high'] = np.where(
                (df['high'] > df['high'].shift(1)) & (df['high'].shift(1) > df['high'].shift(2)), 1, 0
            )
            df['lower_low'] = np.where(
                (df['low'] < df['low'].shift(1)) & (df['low'].shift(1) < df['low'].shift(2)), 1, 0
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating advanced indicators: {e}")
            return df

    def prepare_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for machine learning models"""
        try:
            # Create lagged features
            for lag in [1, 2, 3, 5, 10]:
                df[f'close_lag_{lag}'] = df['close'].shift(lag)
                df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
                df[f'rsi_lag_{lag}'] = df['rsi'].shift(lag)
            
            # Create rolling statistics
            for window in [5, 10, 20]:
                df[f'close_mean_{window}'] = df['close'].rolling(window=window).mean()
                df[f'close_std_{window}'] = df['close'].rolling(window=window).std()
                df[f'volume_mean_{window}'] = df['volume'].rolling(window=window).mean()
                df[f'high_max_{window}'] = df['high'].rolling(window=window).max()
                df[f'low_min_{window}'] = df['low'].rolling(window=window).min()
            
            # Create interaction features
            df['rsi_macd'] = df['rsi'] * df['macd']
            df['volume_price'] = df['volume'] * df['close']
            df['volatility_volume'] = df['volatility_20'] * df['volume_ratio']
            
            # Time-based features
            if 'timestamp' in df.columns:
                df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
                df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
                df['month'] = pd.to_datetime(df['timestamp']).dt.month
            
            # Target variables for prediction
            df['future_return_1'] = df['close'].shift(-1) / df['close'] - 1
            df['future_return_5'] = df['close'].shift(-5) / df['close'] - 1
            df['future_direction'] = np.where(df['future_return_1'] > 0, 1, 0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparing ML features: {e}")
            return df

    def train_price_prediction_model(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Train machine learning model for price prediction"""
        try:
            # Prepare features
            feature_columns = [
                'sma_20', 'sma_50', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal',
                'bb_upper', 'bb_lower', 'stoch_k', 'stoch_d', 'williams_r', 'atr',
                'cci', 'mfi', 'adx', 'price_momentum_5', 'price_momentum_10',
                'volume_ratio', 'volatility_20', 'support_distance', 'resistance_distance'
            ]
            
            # Filter available features
            available_features = [col for col in feature_columns if col in df.columns]
            
            if len(available_features) < 5:
                logger.warning(f"Insufficient features for training: {len(available_features)}")
                return {}
            
            # Prepare data
            df_clean = df[available_features + ['future_return_1']].dropna()
            
            if len(df_clean) < 100:
                logger.warning(f"Insufficient data for training: {len(df_clean)}")
                return {}
            
            X = df_clean[available_features]
            y = df_clean['future_return_1']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train multiple models
            models = {
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'linear_regression': LinearRegression()
            }
            
            model_results = {}
            
            for model_name, model in models.items():
                try:
                    # Train model
                    model.fit(X_train_scaled, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test_scaled)
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    # Feature importance (if available)
                    feature_importance = None
                    if hasattr(model, 'feature_importances_'):
                        feature_importance = dict(zip(available_features, model.feature_importances_))
                    
                    model_results[model_name] = {
                        'model': model,
                        'mse': mse,
                        'r2': r2,
                        'feature_importance': feature_importance
                    }
                    
                    logger.info(f"Trained {model_name} for {symbol}: MSE={mse:.6f}, R2={r2:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error training {model_name}: {e}")
                    continue
            
            # Select best model
            if model_results:
                best_model_name = min(model_results.keys(), key=lambda k: model_results[k]['mse'])
                best_model_info = model_results[best_model_name]
                
                # Store model and scaler
                self.models[symbol] = {
                    'model': best_model_info['model'],
                    'model_name': best_model_name,
                    'features': available_features,
                    'performance': {
                        'mse': best_model_info['mse'],
                        'r2': best_model_info['r2']
                    },
                    'trained_at': datetime.now()
                }
                self.scalers[symbol] = scaler
                self.feature_importance[symbol] = best_model_info['feature_importance']
                
                return self.models[symbol]
            
            return {}
            
        except Exception as e:
            logger.error(f"Error training price prediction model: {e}")
            return {}

    def predict_price_movement(self, df: pd.DataFrame, symbol: str) -> Optional[MLPrediction]:
        """Predict future price movement using trained model"""
        try:
            if symbol not in self.models or symbol not in self.scalers:
                logger.warning(f"No trained model found for {symbol}")
                return None
            
            model_info = self.models[symbol]
            model = model_info['model']
            scaler = self.scalers[symbol]
            features = model_info['features']
            
            # Get latest data point
            latest_data = df[features].iloc[-1:].values
            
            if np.any(np.isnan(latest_data)):
                logger.warning("NaN values in latest data, cannot make prediction")
                return None
            
            # Scale features
            latest_scaled = scaler.transform(latest_data)
            
            # Make prediction
            prediction = model.predict(latest_scaled)[0]
            
            # Calculate confidence based on model performance
            confidence = max(0.1, min(0.9, model_info['performance']['r2']))
            
            return MLPrediction(
                symbol=symbol,
                prediction_type='price_return',
                predicted_value=prediction,
                confidence=confidence,
                horizon='1d',
                features_used=features,
                model_name=model_info['model_name'],
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error predicting price movement: {e}")
            return None

    def generate_enhanced_signals(self, df: pd.DataFrame, symbol: str) -> List[TechnicalSignal]:
        """Generate enhanced trading signals combining traditional and ML indicators"""
        try:
            signals = []
            
            if len(df) < 50:
                logger.warning("Insufficient data for signal generation")
                return signals
            
            latest = df.iloc[-1]
            
            # RSI signals
            if 'rsi' in df.columns and not np.isnan(latest['rsi']):
                if latest['rsi'] < 30:
                    signals.append(TechnicalSignal(
                        indicator='RSI',
                        signal='BUY',
                        strength=min(0.9, (30 - latest['rsi']) / 30),
                        value=latest['rsi'],
                        timestamp=datetime.now(),
                        metadata={'threshold': 30, 'condition': 'oversold'}
                    ))
                elif latest['rsi'] > 70:
                    signals.append(TechnicalSignal(
                        indicator='RSI',
                        signal='SELL',
                        strength=min(0.9, (latest['rsi'] - 70) / 30),
                        value=latest['rsi'],
                        timestamp=datetime.now(),
                        metadata={'threshold': 70, 'condition': 'overbought'}
                    ))
            
            # MACD signals
            if all(col in df.columns for col in ['macd', 'macd_signal']):
                if not (np.isnan(latest['macd']) or np.isnan(latest['macd_signal'])):
                    macd_diff = latest['macd'] - latest['macd_signal']
                    prev_macd_diff = df.iloc[-2]['macd'] - df.iloc[-2]['macd_signal']
                    
                    if macd_diff > 0 and prev_macd_diff <= 0:  # Bullish crossover
                        signals.append(TechnicalSignal(
                            indicator='MACD',
                            signal='BUY',
                            strength=min(0.8, abs(macd_diff) * 100),
                            value=macd_diff,
                            timestamp=datetime.now(),
                            metadata={'type': 'bullish_crossover'}
                        ))
                    elif macd_diff < 0 and prev_macd_diff >= 0:  # Bearish crossover
                        signals.append(TechnicalSignal(
                            indicator='MACD',
                            signal='SELL',
                            strength=min(0.8, abs(macd_diff) * 100),
                            value=macd_diff,
                            timestamp=datetime.now(),
                            metadata={'type': 'bearish_crossover'}
                        ))
            
            # Bollinger Bands signals
            if all(col in df.columns for col in ['bb_upper', 'bb_lower', 'close']):
                if not any(np.isnan([latest['bb_upper'], latest['bb_lower'], latest['close']])):
                    bb_position = (latest['close'] - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower'])
                    
                    if bb_position < 0.1:  # Near lower band
                        signals.append(TechnicalSignal(
                            indicator='Bollinger_Bands',
                            signal='BUY',
                            strength=0.6,
                            value=bb_position,
                            timestamp=datetime.now(),
                            metadata={'position': 'near_lower_band'}
                        ))
                    elif bb_position > 0.9:  # Near upper band
                        signals.append(TechnicalSignal(
                            indicator='Bollinger_Bands',
                            signal='SELL',
                            strength=0.6,
                            value=bb_position,
                            timestamp=datetime.now(),
                            metadata={'position': 'near_upper_band'}
                        ))
            
            # Moving Average signals
            if all(col in df.columns for col in ['sma_20', 'sma_50', 'close']):
                if not any(np.isnan([latest['sma_20'], latest['sma_50'], latest['close']])):
                    if latest['close'] > latest['sma_20'] > latest['sma_50']:
                        signals.append(TechnicalSignal(
                            indicator='Moving_Average',
                            signal='BUY',
                            strength=0.7,
                            value=latest['close'] / latest['sma_20'],
                            timestamp=datetime.now(),
                            metadata={'pattern': 'bullish_alignment'}
                        ))
                    elif latest['close'] < latest['sma_20'] < latest['sma_50']:
                        signals.append(TechnicalSignal(
                            indicator='Moving_Average',
                            signal='SELL',
                            strength=0.7,
                            value=latest['close'] / latest['sma_20'],
                            timestamp=datetime.now(),
                            metadata={'pattern': 'bearish_alignment'}
                        ))
            
            # ML-based signal
            ml_prediction = self.predict_price_movement(df, symbol)
            if ml_prediction:
                if ml_prediction.predicted_value > 0.02:  # Expecting >2% return
                    signals.append(TechnicalSignal(
                        indicator='ML_Prediction',
                        signal='BUY',
                        strength=ml_prediction.confidence,
                        value=ml_prediction.predicted_value,
                        timestamp=datetime.now(),
                        metadata={
                            'model': ml_prediction.model_name,
                            'predicted_return': ml_prediction.predicted_value
                        }
                    ))
                elif ml_prediction.predicted_value < -0.02:  # Expecting <-2% return
                    signals.append(TechnicalSignal(
                        indicator='ML_Prediction',
                        signal='SELL',
                        strength=ml_prediction.confidence,
                        value=ml_prediction.predicted_value,
                        timestamp=datetime.now(),
                        metadata={
                            'model': ml_prediction.model_name,
                            'predicted_return': ml_prediction.predicted_value
                        }
                    ))
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating enhanced signals: {e}")
            return []

    def calculate_composite_score(self, signals: List[TechnicalSignal]) -> Dict[str, float]:
        """Calculate composite trading score from multiple signals"""
        try:
            if not signals:
                return {'buy_score': 0.0, 'sell_score': 0.0, 'hold_score': 1.0}
            
            buy_signals = [s for s in signals if s.signal == 'BUY']
            sell_signals = [s for s in signals if s.signal == 'SELL']
            
            # Weight signals by strength and indicator importance
            indicator_weights = {
                'RSI': 0.8,
                'MACD': 0.9,
                'Bollinger_Bands': 0.7,
                'Moving_Average': 0.8,
                'ML_Prediction': 1.0,
                'default': 0.6
            }
            
            buy_score = sum(
                s.strength * indicator_weights.get(s.indicator, indicator_weights['default'])
                for s in buy_signals
            )
            
            sell_score = sum(
                s.strength * indicator_weights.get(s.indicator, indicator_weights['default'])
                for s in sell_signals
            )
            
            # Normalize scores
            total_score = buy_score + sell_score
            if total_score > 0:
                buy_score = buy_score / total_score
                sell_score = sell_score / total_score
                hold_score = max(0, 1 - buy_score - sell_score)
            else:
                buy_score = sell_score = 0.0
                hold_score = 1.0
            
            return {
                'buy_score': buy_score,
                'sell_score': sell_score,
                'hold_score': hold_score
            }
            
        except Exception as e:
            logger.error(f"Error calculating composite score: {e}")
            return {'buy_score': 0.0, 'sell_score': 0.0, 'hold_score': 1.0}

    async def analyze_symbol(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Complete analysis of a symbol with enhanced indicators"""
        try:
            # Calculate all indicators
            df = self.calculate_traditional_indicators(df)
            df = self.calculate_advanced_indicators(df)
            df = self.prepare_ml_features(df)
            
            # Train ML model if not exists
            if symbol not in self.models:
                self.train_price_prediction_model(df, symbol)
            
            # Generate signals
            signals = self.generate_enhanced_signals(df, symbol)
            
            # Calculate composite score
            composite_score = self.calculate_composite_score(signals)
            
            # Get ML prediction
            ml_prediction = self.predict_price_movement(df, symbol)
            
            # Feature importance
            feature_importance = self.feature_importance.get(symbol, {})
            
            analysis = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'signals': [
                    {
                        'indicator': s.indicator,
                        'signal': s.signal,
                        'strength': s.strength,
                        'value': s.value,
                        'metadata': s.metadata
                    } for s in signals
                ],
                'composite_score': composite_score,
                'ml_prediction': {
                    'predicted_return': ml_prediction.predicted_value if ml_prediction else None,
                    'confidence': ml_prediction.confidence if ml_prediction else None,
                    'model': ml_prediction.model_name if ml_prediction else None
                } if ml_prediction else None,
                'feature_importance': feature_importance,
                'recommendation': self._get_recommendation(composite_score),
                'risk_level': self._assess_risk_level(df, signals)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing symbol {symbol}: {e}")
            return {}

    def _get_recommendation(self, composite_score: Dict[str, float]) -> str:
        """Get trading recommendation based on composite score"""
        buy_score = composite_score.get('buy_score', 0)
        sell_score = composite_score.get('sell_score', 0)
        
        if buy_score > 0.6:
            return 'STRONG_BUY'
        elif buy_score > 0.4:
            return 'BUY'
        elif sell_score > 0.6:
            return 'STRONG_SELL'
        elif sell_score > 0.4:
            return 'SELL'
        else:
            return 'HOLD'

    def _assess_risk_level(self, df: pd.DataFrame, signals: List[TechnicalSignal]) -> str:
        """Assess risk level based on volatility and signal consensus"""
        try:
            if 'volatility_20' in df.columns:
                latest_volatility = df['volatility_20'].iloc[-1]
                avg_volatility = df['volatility_20'].mean()
                
                volatility_ratio = latest_volatility / avg_volatility if avg_volatility > 0 else 1
                
                # Signal consensus
                buy_signals = len([s for s in signals if s.signal == 'BUY'])
                sell_signals = len([s for s in signals if s.signal == 'SELL'])
                total_signals = buy_signals + sell_signals
                
                consensus = abs(buy_signals - sell_signals) / total_signals if total_signals > 0 else 0
                
                if volatility_ratio > 1.5 or consensus < 0.3:
                    return 'HIGH'
                elif volatility_ratio > 1.2 or consensus < 0.5:
                    return 'MEDIUM'
                else:
                    return 'LOW'
            
            return 'MEDIUM'
            
        except Exception as e:
            logger.error(f"Error assessing risk level: {e}")
            return 'MEDIUM'

# Example usage
async def main():
    """Example usage of EnhancedTechnicalIndicators"""
    # Create sample data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    np.random.seed(42)
    
    # Generate realistic price data
    price = 100
    prices = []
    volumes = []
    
    for _ in range(len(dates)):
        price += np.random.normal(0, 2)
        prices.append(max(price, 1))  # Ensure positive prices
        volumes.append(np.random.randint(1000000, 10000000))
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + np.random.uniform(0, 0.05)) for p in prices],
        'low': [p * (1 - np.random.uniform(0, 0.05)) for p in prices],
        'close': prices,
        'volume': volumes
    })
    
    # Initialize indicators
    indicators = EnhancedTechnicalIndicators()
    
    # Analyze symbol
    analysis = await indicators.analyze_symbol(df, 'AAPL')
    
    print(f"Analysis for AAPL:")
    print(f"Recommendation: {analysis.get('recommendation', 'N/A')}")
    print(f"Risk Level: {analysis.get('risk_level', 'N/A')}")
    print(f"Composite Score: {analysis.get('composite_score', {})}")
    print(f"Number of Signals: {len(analysis.get('signals', []))}")

if __name__ == "__main__":
    asyncio.run(main())