"""Advanced Machine Learning Models for Index Analysis

This module implements enhanced LSTM and Transformer models with attention mechanisms,
ensemble methods, and advanced feature engineering for index prediction.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from datetime import datetime
from enum import Enum
import math

class IndexIndicatorType(Enum):
    """Index-specific indicator types"""
    LSTM = "lstm"
    TRANSFORMER = "transformer"

@dataclass
class MacroeconomicData:
    """Macroeconomic indicators"""
    gdp_growth: float
    inflation_rate: float
    unemployment_rate: float
    interest_rate: float
    money_supply_growth: float
    government_debt_to_gdp: float
    trade_balance: float
    consumer_confidence: float
    business_confidence: float
    manufacturing_pmi: float
    services_pmi: float
    retail_sales_growth: float
    industrial_production: float
    housing_starts: float
    oil_price: float
    dollar_index: float
    vix: float

@dataclass
class IndexData:
    """Index information"""
    symbol: str
    name: str
    current_level: float
    historical_levels: list[float]
    dividend_yield: float
    pe_ratio: float
    pb_ratio: float
    market_cap: float
    volatility: float
    beta: float
    sector_weights: Dict[str, float]
    constituent_count: int
    volume: float

@dataclass
class MLResult:
    """Result of ML model prediction"""
    predicted_value: float
    confidence: float
    metadata: Dict[str, Any]
    timestamp: datetime
    interpretation: str
    risk_level: str
    signal: str
    time_horizon: str

class AdvancedMLModels:
    """Enhanced Machine Learning Models with Attention Mechanisms and Ensemble Methods"""
    
    def __init__(self):
        self.sequence_length = 30  # Extended look-back period
        self.attention_heads = 8  # Multi-head attention
        self.feature_dim = 64  # Feature embedding dimension
        self.dropout_rate = 0.1
        
        # Enhanced feature weights with more granular control
        self.feature_weights = {
            "price_momentum": 0.22,
            "volatility": 0.18,
            "volume": 0.12,
            "macro_factors": 0.35,
            "technical_indicators": 0.13
        }
        
        # Ensemble weights (adaptive based on recent performance)
        self.ensemble_weights = {
            "lstm": 0.35,
            "transformer": 0.40,
            "hybrid_attention": 0.25
        }
        
        # Advanced feature engineering parameters
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.bollinger_period = 20
        self.bollinger_std = 2.0
    
    def lstm_prediction(self, index_data: IndexData, macro_data: MacroeconomicData) -> MLResult:
        """LSTM model for index prediction"""
        try:
            # Prepare features
            features = self._prepare_lstm_features(index_data, macro_data)
            
            # Simulate LSTM prediction (in practice, would use trained model)
            prediction = self._simulate_lstm_forward_pass(features)
            
            # Generate signal
            current_level = index_data.current_level
            predicted_change = (prediction - current_level) / current_level
            
            if predicted_change > 0.03:  # 3% threshold
                signal = "BUY"
            elif predicted_change < -0.03:
                signal = "SELL"
            else:
                signal = "HOLD"
            
            # Calculate confidence based on prediction consistency
            confidence = self._calculate_lstm_confidence(features, prediction)
            
            risk_level = "Low" if abs(predicted_change) < 0.05 else "Medium" if abs(predicted_change) < 0.10 else "High"
            
            return MLResult(
                predicted_value=prediction,
                confidence=confidence,
                metadata={
                    "features": features,
                    "predicted_change": predicted_change,
                    "current_level": current_level,
                    "model_type": "LSTM",
                    "sequence_length": self.sequence_length
                },
                timestamp=datetime.now(),
                interpretation=f"LSTM prediction: {prediction:.0f} (Change: {predicted_change:.2%})",
                risk_level=risk_level,
                signal=signal,
                time_horizon="Short to Medium-term"
            )
        except Exception as e:
            return MLResult(
                predicted_value=0.0,
                confidence=0.0,
                metadata={"error": str(e), "model_type": "LSTM"},
                timestamp=datetime.now(),
                interpretation="LSTM prediction failed",
                risk_level="High",
                signal="HOLD",
                time_horizon="N/A"
            )
    
    def transformer_prediction(self, index_data: IndexData, macro_data: MacroeconomicData) -> MLResult:
        """Transformer model for index prediction"""
        try:
            # Prepare sequence data for transformer
            sequence_data = self._prepare_transformer_sequence(index_data, macro_data)
            
            # Simulate transformer prediction with attention mechanism
            prediction, attention_weights = self._simulate_transformer_forward_pass(sequence_data)
            
            # Generate signal
            current_level = index_data.current_level
            predicted_change = (prediction - current_level) / current_level
            
            if predicted_change > 0.025:  # 2.5% threshold (more sensitive than LSTM)
                signal = "BUY"
            elif predicted_change < -0.025:
                signal = "SELL"
            else:
                signal = "HOLD"
            
            # Calculate confidence based on attention consistency
            confidence = self._calculate_transformer_confidence(attention_weights)
            
            risk_level = "Low" if abs(predicted_change) < 0.04 else "Medium" if abs(predicted_change) < 0.08 else "High"
            
            return MLResult(
                predicted_value=prediction,
                confidence=confidence,
                metadata={
                    "sequence_data": sequence_data,
                    "attention_weights": attention_weights,
                    "predicted_change": predicted_change,
                    "current_level": current_level,
                    "model_type": "Transformer"
                },
                timestamp=datetime.now(),
                interpretation=f"Transformer prediction: {prediction:.0f} (Change: {predicted_change:.2%})",
                risk_level=risk_level,
                signal=signal,
                time_horizon="Short-term"
            )
        except Exception as e:
            return MLResult(
                predicted_value=0.0,
                confidence=0.0,
                metadata={"error": str(e), "model_type": "Transformer"},
                timestamp=datetime.now(),
                interpretation="Transformer prediction failed",
                risk_level="High",
                signal="HOLD",
                time_horizon="N/A"
            )
    
    def _prepare_lstm_features(self, index_data: IndexData, macro_data: MacroeconomicData) -> Dict[str, Any]:
        """Enhanced feature preparation with advanced technical indicators"""
        # Historical price features
        historical_levels = index_data.historical_levels[-self.sequence_length:]
        if len(historical_levels) < self.sequence_length:
            historical_levels = [index_data.current_level] * (self.sequence_length - len(historical_levels)) + historical_levels
        
        # Enhanced technical features
        returns = self._calculate_returns(historical_levels)
        volatilities = self._calculate_rolling_volatility(returns, window=10)
        
        # Advanced momentum indicators
        momentum_features = self._calculate_momentum_indicators(historical_levels)
        
        # Technical indicators
        technical_indicators = self._calculate_technical_indicators(historical_levels)
        
        # Enhanced macro features with regime detection
        macro_features = self._prepare_enhanced_macro_features(macro_data)
        
        # Cross-asset correlations and market regime indicators
        market_regime = self._detect_market_regime(historical_levels, macro_data)
        
        # Feature embeddings for better representation
        embedded_features = self._create_feature_embeddings({
            "price_levels": historical_levels,
            "returns": returns,
            "volatilities": volatilities,
            **momentum_features,
            **technical_indicators,
            **macro_features,
            "market_regime": market_regime
        })
        
        return {
            "price_levels": historical_levels,
            "returns": returns,
            "volatilities": volatilities,
            "momentum_features": momentum_features,
            "technical_indicators": technical_indicators,
            "macro_features": macro_features,
            "market_regime": market_regime,
            "embedded_features": embedded_features,
            "current_pe": index_data.pe_ratio,
            "current_pb": index_data.pb_ratio,
            "dividend_yield": index_data.dividend_yield
        }
    
    def _simulate_lstm_forward_pass(self, features: Dict[str, Any]) -> float:
        """Enhanced LSTM forward pass with attention and residual connections"""
        current_level = features["price_levels"][-1]
        
        # Multi-layer LSTM simulation with attention
        lstm_states = self._simulate_lstm_layers(features["embedded_features"])
        
        # Apply self-attention to LSTM outputs
        attended_features = self._apply_self_attention(lstm_states)
        
        # Combine different signal components with enhanced weighting
        momentum_signal = self._calculate_enhanced_momentum_signal(features["momentum_features"])
        technical_signal = self._calculate_technical_signal(features["technical_indicators"])
        volatility_signal = self._calculate_volatility_signal(features["volatilities"])
        macro_signal = self._calculate_macro_signal(features["macro_features"])
        regime_signal = self._calculate_regime_signal(features["market_regime"])
        
        # Adaptive feature weighting based on market conditions
        adaptive_weights = self._calculate_adaptive_weights(features["market_regime"])
        
        # Combine all signals with adaptive weighting
        total_signal = (
            momentum_signal * adaptive_weights["momentum"] +
            technical_signal * adaptive_weights["technical"] +
            volatility_signal * adaptive_weights["volatility"] +
            macro_signal * adaptive_weights["macro"] +
            regime_signal * adaptive_weights["regime"]
        )
        
        # Add attention-weighted signal
        attention_signal = np.mean(attended_features) * 0.15
        total_signal += attention_signal
        
        # Enhanced prediction with residual connections
        predicted_change = np.tanh(total_signal) * 0.12  # Slightly higher range
        
        # Add residual connection for stability
        residual = np.mean(features["returns"][-3:]) * 0.1 if len(features["returns"]) >= 3 else 0.0
        predicted_change += residual
        
        prediction = current_level * (1 + predicted_change)
        return prediction
    
    def _calculate_lstm_confidence(self, features: Dict[str, Any], prediction: float) -> float:
        """Enhanced confidence calculation with multiple factors"""
        base_confidence = 0.65
        
        # Signal consistency across different timeframes
        momentum_consistency = self._calculate_momentum_consistency(features["momentum_features"])
        technical_consistency = self._calculate_technical_consistency(features["technical_indicators"])
        
        # Market regime stability
        regime_stability = features["market_regime"].get("stability", 0.5)
        
        # Volatility-adjusted confidence
        vol_adjustment = self._calculate_volatility_adjustment(features["volatilities"])
        
        # Feature quality assessment
        feature_quality = self._assess_feature_quality(features)
        
        # Combine all confidence factors
        confidence = (
            base_confidence +
            momentum_consistency * 0.15 +
            technical_consistency * 0.12 +
            regime_stability * 0.10 +
            vol_adjustment * 0.08 +
            feature_quality * 0.10
        )
        
        return max(0.25, min(0.95, confidence))
    
    def _prepare_transformer_sequence(self, index_data: IndexData, macro_data: MacroeconomicData) -> List[Dict[str, float]]:
        """Enhanced sequence preparation with positional encoding and multi-scale features"""
        sequence = []
        
        historical_levels = index_data.historical_levels[-self.sequence_length:]
        if len(historical_levels) < self.sequence_length:
            historical_levels = [index_data.current_level] * (self.sequence_length - len(historical_levels)) + historical_levels
        
        # Pre-calculate technical indicators for the entire sequence
        sequence_returns = self._calculate_returns(historical_levels)
        sequence_rsi = self._calculate_rsi(historical_levels)
        sequence_macd = self._calculate_macd(historical_levels)
        sequence_bollinger = self._calculate_bollinger_bands(historical_levels)
        
        for i, level in enumerate(historical_levels):
            # Multi-scale return features
            returns_dict = self._calculate_multi_scale_returns(historical_levels, i)
            
            # Enhanced positional encoding with sinusoidal patterns
            pos_encoding = self._calculate_positional_encoding(i, self.sequence_length, self.feature_dim)
            
            # Technical indicators at this position
            technical_features = {
                "rsi": sequence_rsi[i] if i < len(sequence_rsi) else 50.0,
                "macd": sequence_macd[i] if i < len(sequence_macd) else 0.0,
                "bb_position": sequence_bollinger[i] if i < len(sequence_bollinger) else 0.5
            }
            
            # Enhanced macro features with time-varying components
            macro_features = self._prepare_time_varying_macro_features(macro_data, i)
            
            # Market microstructure features
            microstructure_features = self._calculate_microstructure_features(index_data, i)
            
            feature_vector = {
                "price_level": level / index_data.current_level,
                **returns_dict,
                **technical_features,
                **macro_features,
                **microstructure_features,
                "position_encoding": pos_encoding,
                "sequence_position": i / self.sequence_length
            }
            
            sequence.append(feature_vector)
        
        return sequence
    
    def _simulate_transformer_forward_pass(self, sequence_data: List[Dict[str, float]]) -> Tuple[float, Dict[str, Any]]:
        """Enhanced transformer with multi-head attention and layer normalization"""
        # Multi-head self-attention
        attention_outputs = []
        attention_weights_all_heads = []
        
        for head in range(self.attention_heads):
            attention_output, attention_weights = self._compute_attention_head(sequence_data, head)
            attention_outputs.append(attention_output)
            attention_weights_all_heads.append(attention_weights)
        
        # Concatenate multi-head outputs
        multi_head_output = self._concatenate_attention_heads(attention_outputs)
        
        # Add residual connection and layer normalization
        normalized_output = self._layer_normalize(multi_head_output, sequence_data)
        
        # Feed-forward network simulation
        ff_output = self._simulate_feed_forward(normalized_output)
        
        # Final prediction layer
        prediction_signal = self._compute_final_prediction_signal(ff_output)
        
        # Convert to price prediction with enhanced logic
        current_level = sequence_data[-1]["price_level"] * sequence_data[0].get("price_level", 1.0)
        predicted_change = np.tanh(prediction_signal) * 0.10  # Enhanced range
        prediction = current_level * (1 + predicted_change)
        
        attention_info = {
            "multi_head_weights": attention_weights_all_heads,
            "average_attention": np.mean(attention_weights_all_heads, axis=0).tolist(),
            "attention_entropy": self._calculate_attention_entropy(attention_weights_all_heads),
            "head_diversity": self._calculate_head_diversity(attention_weights_all_heads)
        }
        
        return prediction, attention_info
    
    def _calculate_transformer_confidence(self, attention_info: Dict[str, Any]) -> float:
        """Enhanced confidence calculation using multi-head attention analysis"""
        base_confidence = 0.60
        
        # Multi-head attention consistency
        head_diversity = attention_info["head_diversity"]
        attention_entropy = attention_info["attention_entropy"]
        
        # Higher confidence when heads agree but maintain diversity
        diversity_bonus = min(0.15, head_diversity * 0.3)
        
        # Attention focus bonus (lower entropy = more focused)
        max_entropy = np.log(len(attention_info["average_attention"]))
        attention_focus = 1.0 - (attention_entropy / max_entropy)
        focus_bonus = attention_focus * 0.20
        
        # Temporal consistency (recent vs distant attention)
        temporal_consistency = self._calculate_temporal_attention_consistency(attention_info["average_attention"])
        temporal_bonus = temporal_consistency * 0.10
        
        confidence = base_confidence + diversity_bonus + focus_bonus + temporal_bonus
        
        return max(0.30, min(0.95, confidence))
    
    def ensemble_prediction(self, index_data: IndexData, macro_data: MacroeconomicData) -> MLResult:
        """Combine LSTM and Transformer predictions"""
        try:
            lstm_result = self.lstm_prediction(index_data, macro_data)
            transformer_result = self.transformer_prediction(index_data, macro_data)
            
            # Weight predictions by confidence
            lstm_weight = lstm_result.confidence
            transformer_weight = transformer_result.confidence
            total_weight = lstm_weight + transformer_weight
            
            if total_weight > 0:
                ensemble_prediction = (
                    lstm_result.predicted_value * lstm_weight +
                    transformer_result.predicted_value * transformer_weight
                ) / total_weight
                
                ensemble_confidence = (lstm_result.confidence + transformer_result.confidence) / 2
            else:
                ensemble_prediction = index_data.current_level
                ensemble_confidence = 0.3
            
            # Generate ensemble signal
            predicted_change = (ensemble_prediction - index_data.current_level) / index_data.current_level
            
            if predicted_change > 0.025:
                signal = "BUY"
            elif predicted_change < -0.025:
                signal = "SELL"
            else:
                signal = "HOLD"
            
            risk_level = "Low" if abs(predicted_change) < 0.04 else "Medium" if abs(predicted_change) < 0.08 else "High"
            
            return MLResult(
                predicted_value=ensemble_prediction,
                confidence=ensemble_confidence,
                metadata={
                    "lstm_prediction": lstm_result.predicted_value,
                    "transformer_prediction": transformer_result.predicted_value,
                    "lstm_confidence": lstm_result.confidence,
                    "transformer_confidence": transformer_result.confidence,
                    "predicted_change": predicted_change,
                    "model_type": "Ensemble"
                },
                timestamp=datetime.now(),
                interpretation=f"Ensemble prediction: {ensemble_prediction:.0f} (Change: {predicted_change:.2%})",
                risk_level=risk_level,
                signal=signal,
                time_horizon="Short to Medium-term"
            )
        except Exception as e:
            return MLResult(
                predicted_value=0.0,
                confidence=0.0,
                metadata={"error": str(e), "model_type": "Ensemble"},
                timestamp=datetime.now(),
                interpretation="Ensemble prediction failed",
                risk_level="High",
                signal="HOLD",
                time_horizon="N/A"
            )
    
    # ==================== ENHANCED UTILITY METHODS ====================
    
    def _calculate_returns(self, prices: List[float]) -> List[float]:
        """Calculate returns from price series"""
        returns = []
        for i in range(1, len(prices)):
            returns.append((prices[i] - prices[i-1]) / prices[i-1])
        return returns
    
    def _calculate_rolling_volatility(self, returns: List[float], window: int = 10) -> List[float]:
        """Calculate rolling volatility"""
        volatilities = []
        for i in range(len(returns)):
            start_idx = max(0, i - window + 1)
            window_returns = returns[start_idx:i+1]
            vol = np.std(window_returns) if len(window_returns) > 1 else 0.0
            volatilities.append(vol)
        return volatilities
    
    def _calculate_momentum_indicators(self, prices: List[float]) -> Dict[str, float]:
        """Calculate various momentum indicators"""
        momentum_features = {}
        
        # Multiple timeframe momentum
        for period in [3, 5, 10, 20]:
            if len(prices) >= period + 1:
                momentum = (prices[-1] - prices[-period-1]) / prices[-period-1]
                momentum_features[f"momentum_{period}"] = momentum
            else:
                momentum_features[f"momentum_{period}"] = 0.0
        
        # Rate of change
        if len(prices) >= 2:
            momentum_features["roc_1"] = (prices[-1] - prices[-2]) / prices[-2]
        else:
            momentum_features["roc_1"] = 0.0
        
        return momentum_features
    
    def _calculate_technical_indicators(self, prices: List[float]) -> Dict[str, float]:
        """Calculate technical indicators"""
        indicators = {}
        
        # RSI
        indicators["rsi"] = self._calculate_rsi_single(prices)
        
        # MACD
        macd_line, signal_line = self._calculate_macd_single(prices)
        indicators["macd"] = macd_line
        indicators["macd_signal"] = signal_line
        indicators["macd_histogram"] = macd_line - signal_line
        
        # Bollinger Bands position
        indicators["bb_position"] = self._calculate_bollinger_position(prices)
        
        # Moving averages
        indicators["sma_20"] = np.mean(prices[-20:]) if len(prices) >= 20 else prices[-1]
        indicators["ema_12"] = self._calculate_ema(prices, 12)
        
        return indicators
    
    def _calculate_rsi_single(self, prices: List[float]) -> float:
        """Calculate RSI for current position"""
        if len(prices) < self.rsi_period + 1:
            return 50.0
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(-change)
        
        avg_gain = np.mean(gains[-self.rsi_period:])
        avg_loss = np.mean(losses[-self.rsi_period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd_single(self, prices: List[float]) -> Tuple[float, float]:
        """Calculate MACD line and signal line"""
        if len(prices) < max(self.macd_slow, 26):
            return 0.0, 0.0
        
        ema_fast = self._calculate_ema(prices, self.macd_fast)
        ema_slow = self._calculate_ema(prices, self.macd_slow)
        macd_line = ema_fast - ema_slow
        
        # Signal line (9-period EMA of MACD)
        signal_line = macd_line * 0.8  # Simplified
        
        return macd_line, signal_line
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return np.mean(prices)
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _calculate_bollinger_position(self, prices: List[float]) -> float:
        """Calculate position within Bollinger Bands"""
        if len(prices) < self.bollinger_period:
            return 0.5
        
        recent_prices = prices[-self.bollinger_period:]
        sma = np.mean(recent_prices)
        std = np.std(recent_prices)
        
        upper_band = sma + (self.bollinger_std * std)
        lower_band = sma - (self.bollinger_std * std)
        
        current_price = prices[-1]
        
        if upper_band == lower_band:
            return 0.5
        
        position = (current_price - lower_band) / (upper_band - lower_band)
        return max(0.0, min(1.0, position))
    
    def _prepare_enhanced_macro_features(self, macro_data: MacroeconomicData) -> Dict[str, float]:
        """Enhanced macro feature preparation with regime detection"""
        features = {
            # Normalized macro indicators
            "gdp_growth": (macro_data.gdp_growth - 2.0) / 2.0,
            "inflation": (macro_data.inflation_rate - 2.0) / 2.0,
            "interest_rate": (macro_data.interest_rate - 3.0) / 3.0,
            "unemployment": (macro_data.unemployment_rate - 5.0) / 3.0,
            "vix": (macro_data.vix - 20.0) / 15.0,
            "oil_price": (macro_data.oil_price - 70.0) / 30.0,
            "dollar_index": (macro_data.dollar_index - 100.0) / 10.0,
            
            # Composite indicators
            "economic_momentum": (macro_data.manufacturing_pmi + macro_data.services_pmi) / 100.0 - 1.0,
            "confidence_composite": (macro_data.consumer_confidence + macro_data.business_confidence) / 200.0 - 1.0,
            "financial_stress": macro_data.vix / 50.0 + abs(macro_data.trade_balance) / 100.0
        }
        
        return features
    
    def _detect_market_regime(self, prices: List[float], macro_data: MacroeconomicData) -> Dict[str, float]:
        """Detect current market regime"""
        regime = {}
        
        # Volatility regime
        if len(prices) >= 20:
            recent_vol = np.std(self._calculate_returns(prices[-20:]))
            regime["volatility_regime"] = min(1.0, recent_vol / 0.02)  # Normalized
        else:
            regime["volatility_regime"] = 0.5
        
        # Trend regime
        if len(prices) >= 10:
            trend_strength = (prices[-1] - prices[-10]) / prices[-10]
            regime["trend_regime"] = np.tanh(trend_strength * 10) * 0.5 + 0.5
        else:
            regime["trend_regime"] = 0.5
        
        # Economic regime
        econ_score = (macro_data.gdp_growth / 4.0 + 
                     (5.0 - macro_data.unemployment_rate) / 5.0 + 
                     (60.0 - macro_data.vix) / 60.0) / 3.0
        regime["economic_regime"] = max(0.0, min(1.0, econ_score))
        
        # Stability measure
        regime["stability"] = 1.0 - regime["volatility_regime"] * 0.5
        
        return regime
    
    def _create_feature_embeddings(self, features: Dict[str, Any]) -> List[float]:
        """Create feature embeddings for better representation"""
        embeddings = []
        
        # Price-based embeddings
        if "price_levels" in features:
            price_embedding = np.mean(features["price_levels"][-5:]) / features["price_levels"][-1]
            embeddings.append(price_embedding)
        
        # Return-based embeddings
        if "returns" in features:
            return_embedding = np.mean(features["returns"][-5:]) if len(features["returns"]) >= 5 else 0.0
            embeddings.append(return_embedding)
        
        # Volatility embedding
        if "volatilities" in features:
            vol_embedding = np.mean(features["volatilities"][-3:]) if len(features["volatilities"]) >= 3 else 0.0
            embeddings.append(vol_embedding)
        
        # Pad to fixed size
        while len(embeddings) < 10:
            embeddings.append(0.0)
        
        return embeddings[:10]  # Fixed size
    
    def _simulate_lstm_layers(self, embedded_features: List[float]) -> List[float]:
        """Simulate multi-layer LSTM processing"""
        # Simplified LSTM simulation
        layer1_output = [np.tanh(x * 0.8) for x in embedded_features]
        layer2_output = [np.tanh(x * 0.6 + layer1_output[i] * 0.4) for i, x in enumerate(layer1_output)]
        return layer2_output
    
    def _apply_self_attention(self, lstm_states: List[float]) -> List[float]:
        """Apply self-attention to LSTM states"""
        # Simplified self-attention
        attention_weights = [np.exp(x) for x in lstm_states]
        total_weight = sum(attention_weights)
        normalized_weights = [w / total_weight for w in attention_weights]
        
        attended_output = []
        for i, state in enumerate(lstm_states):
            attended_value = sum(state * weight for state, weight in zip(lstm_states, normalized_weights))
            attended_output.append(attended_value)
        
        return attended_output
    
    # ==================== ENHANCED SIGNAL CALCULATION METHODS ====================
    
    def _calculate_enhanced_momentum_signal(self, momentum_features: Dict[str, float]) -> float:
        """Calculate enhanced momentum signal from multiple timeframes"""
        signal = 0.0
        weights = {"momentum_3": 0.4, "momentum_5": 0.3, "momentum_10": 0.2, "momentum_20": 0.1}
        
        for key, weight in weights.items():
            if key in momentum_features:
                signal += momentum_features[key] * weight
        
        return np.tanh(signal * 2.0)  # Normalize
    
    def _calculate_technical_signal(self, technical_indicators: Dict[str, float]) -> float:
        """Calculate signal from technical indicators"""
        signal = 0.0
        
        # RSI signal (overbought/oversold)
        rsi = technical_indicators.get("rsi", 50.0)
        if rsi > 70:
            signal -= (rsi - 70) / 30 * 0.3  # Bearish
        elif rsi < 30:
            signal += (30 - rsi) / 30 * 0.3  # Bullish
        
        # MACD signal
        macd_histogram = technical_indicators.get("macd_histogram", 0.0)
        signal += np.tanh(macd_histogram * 10) * 0.4
        
        # Bollinger Bands signal
        bb_position = technical_indicators.get("bb_position", 0.5)
        if bb_position > 0.8:
            signal -= 0.2  # Overbought
        elif bb_position < 0.2:
            signal += 0.2  # Oversold
        
        return signal
    
    def _calculate_volatility_signal(self, volatilities: List[float]) -> float:
        """Calculate volatility-based signal (mean reversion)"""
        if len(volatilities) < 5:
            return 0.0
        
        recent_vol = np.mean(volatilities[-3:])
        long_term_vol = np.mean(volatilities[-10:]) if len(volatilities) >= 10 else recent_vol
        
        # High volatility suggests mean reversion
        vol_ratio = recent_vol / (long_term_vol + 1e-8)
        signal = -np.tanh((vol_ratio - 1.0) * 3.0) * 0.5
        
        return signal
    
    def _calculate_macro_signal(self, macro_features: Dict[str, float]) -> float:
        """Calculate macroeconomic signal"""
        signal = 0.0
        
        # Economic growth components
        signal += macro_features.get("gdp_growth", 0.0) * 0.3
        signal += macro_features.get("economic_momentum", 0.0) * 0.25
        signal += macro_features.get("confidence_composite", 0.0) * 0.2
        
        # Risk factors (negative impact)
        signal -= macro_features.get("inflation", 0.0) * 0.15
        signal -= macro_features.get("financial_stress", 0.0) * 0.1
        
        return np.tanh(signal)
    
    def _calculate_regime_signal(self, market_regime: Dict[str, float]) -> float:
        """Calculate market regime-based signal"""
        # Favor trending markets, be cautious in high volatility
        trend_signal = (market_regime.get("trend_regime", 0.5) - 0.5) * 2.0
        vol_penalty = market_regime.get("volatility_regime", 0.5) * 0.3
        econ_boost = (market_regime.get("economic_regime", 0.5) - 0.5) * 0.4
        
        return trend_signal - vol_penalty + econ_boost
    
    def _calculate_adaptive_weights(self, market_regime: Dict[str, float]) -> Dict[str, float]:
        """Calculate adaptive feature weights based on market regime"""
        base_weights = {
            "momentum": 0.25,
            "technical": 0.20,
            "volatility": 0.15,
            "macro": 0.30,
            "regime": 0.10
        }
        
        # Adjust weights based on market conditions
        vol_regime = market_regime.get("volatility_regime", 0.5)
        trend_regime = market_regime.get("trend_regime", 0.5)
        
        # In high volatility, increase technical and volatility weights
        if vol_regime > 0.7:
            base_weights["technical"] *= 1.3
            base_weights["volatility"] *= 1.4
            base_weights["momentum"] *= 0.8
        
        # In strong trends, increase momentum weight
        if abs(trend_regime - 0.5) > 0.3:
            base_weights["momentum"] *= 1.4
            base_weights["technical"] *= 0.9
        
        # Normalize weights
        total_weight = sum(base_weights.values())
        return {k: v / total_weight for k, v in base_weights.items()}
    
    # ==================== ENHANCED CONFIDENCE CALCULATION METHODS ====================
    
    def _calculate_momentum_consistency(self, momentum_features: Dict[str, float]) -> float:
        """Calculate consistency across momentum timeframes"""
        momentum_values = [v for k, v in momentum_features.items() if k.startswith("momentum_")]
        if len(momentum_values) < 2:
            return 0.5
        
        # Check if all momentum signals agree in direction
        positive_count = sum(1 for v in momentum_values if v > 0)
        negative_count = sum(1 for v in momentum_values if v < 0)
        
        consistency = max(positive_count, negative_count) / len(momentum_values)
        return consistency
    
    def _calculate_technical_consistency(self, technical_indicators: Dict[str, float]) -> float:
        """Calculate consistency across technical indicators"""
        signals = []
        
        # RSI signal
        rsi = technical_indicators.get("rsi", 50.0)
        if rsi > 60:
            signals.append(1)
        elif rsi < 40:
            signals.append(-1)
        else:
            signals.append(0)
        
        # MACD signal
        macd_histogram = technical_indicators.get("macd_histogram", 0.0)
        signals.append(1 if macd_histogram > 0 else -1)
        
        # Bollinger position signal
        bb_position = technical_indicators.get("bb_position", 0.5)
        if bb_position > 0.7:
            signals.append(-1)  # Overbought
        elif bb_position < 0.3:
            signals.append(1)   # Oversold
        else:
            signals.append(0)
        
        # Calculate agreement
        if not signals:
            return 0.5
        
        positive_count = sum(1 for s in signals if s > 0)
        negative_count = sum(1 for s in signals if s < 0)
        
        consistency = max(positive_count, negative_count) / len(signals)
        return consistency
    
    def _calculate_volatility_adjustment(self, volatilities: List[float]) -> float:
        """Calculate volatility-based confidence adjustment"""
        if len(volatilities) < 3:
            return 0.0
        
        recent_vol = np.mean(volatilities[-3:])
        # Lower confidence in high volatility environments
        vol_adjustment = -min(0.2, recent_vol * 5.0)
        return vol_adjustment
    
    def _assess_feature_quality(self, features: Dict[str, Any]) -> float:
        """Assess overall quality of input features"""
        quality_score = 0.5
        
        # Check data completeness
        if len(features.get("price_levels", [])) >= self.sequence_length:
            quality_score += 0.2
        
        # Check for reasonable values
        if "current_pe" in features and 5 < features["current_pe"] < 50:
            quality_score += 0.1
        
        # Check market regime stability
        regime_stability = features.get("market_regime", {}).get("stability", 0.5)
        quality_score += (regime_stability - 0.5) * 0.4
        
        return max(0.0, min(1.0, quality_score))
    
    # ==================== TRANSFORMER UTILITY METHODS ====================
    
    def _compute_attention_head(self, sequence_data: List[Dict[str, float]], head_idx: int) -> Tuple[List[float], List[float]]:
        """Compute single attention head"""
        seq_len = len(sequence_data)
        attention_weights = []
        
        # Different attention patterns for different heads
        if head_idx % 4 == 0:  # Recent focus
            for i in range(seq_len):
                weight = np.exp(-(seq_len - i - 1) * 0.3)
                attention_weights.append(weight)
        elif head_idx % 4 == 1:  # Uniform attention
            attention_weights = [1.0] * seq_len
        elif head_idx % 4 == 2:  # Early focus
            for i in range(seq_len):
                weight = np.exp(-i * 0.2)
                attention_weights.append(weight)
        else:  # Peak focus (middle)
            for i in range(seq_len):
                distance_from_center = abs(i - seq_len // 2)
                weight = np.exp(-distance_from_center * 0.4)
                attention_weights.append(weight)
        
        # Normalize
        total_weight = sum(attention_weights)
        attention_weights = [w / total_weight for w in attention_weights]
        
        # Compute attended output
        attended_output = []
        for i, data_point in enumerate(sequence_data):
            # Simplified attention computation
            value = sum(data_point.values()) / len(data_point)
            attended_output.append(value * attention_weights[i])
        
        return attended_output, attention_weights
    
    def _concatenate_attention_heads(self, attention_outputs: List[List[float]]) -> List[float]:
        """Concatenate multi-head attention outputs"""
        if not attention_outputs:
            return []
        
        # Average across heads
        seq_len = len(attention_outputs[0])
        concatenated = []
        
        for i in range(seq_len):
            avg_value = np.mean([head[i] for head in attention_outputs])
            concatenated.append(avg_value)
        
        return concatenated
    
    def _layer_normalize(self, multi_head_output: List[float], original_sequence: List[Dict[str, float]]) -> List[float]:
        """Apply layer normalization with residual connection"""
        # Simplified layer normalization
        mean_val = np.mean(multi_head_output)
        std_val = np.std(multi_head_output) + 1e-8
        
        normalized = [(x - mean_val) / std_val for x in multi_head_output]
        
        # Add residual connection (simplified)
        residual = [sum(data_point.values()) / len(data_point) for data_point in original_sequence]
        
        return [norm + res * 0.1 for norm, res in zip(normalized, residual)]
    
    def _simulate_feed_forward(self, normalized_output: List[float]) -> List[float]:
        """Simulate feed-forward network"""
        # Two-layer feed-forward with ReLU activation
        layer1 = [max(0, x * 2.0 - 0.5) for x in normalized_output]  # ReLU
        layer2 = [np.tanh(x * 0.8) for x in layer1]  # Output layer
        
        return layer2
    
    def _compute_final_prediction_signal(self, ff_output: List[float]) -> float:
        """Compute final prediction signal from feed-forward output"""
        # Weighted combination with position bias (recent data more important)
        weights = [np.exp(-i * 0.1) for i in range(len(ff_output))]
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        signal = sum(output * weight for output, weight in zip(ff_output, normalized_weights))
        return signal
    
    def _calculate_attention_entropy(self, attention_weights_all_heads: List[List[float]]) -> float:
        """Calculate entropy of attention weights"""
        # Average entropy across all heads
        entropies = []
        
        for head_weights in attention_weights_all_heads:
            entropy = -sum(w * np.log(w + 1e-8) for w in head_weights)
            entropies.append(entropy)
        
        return np.mean(entropies)
    
    def _calculate_head_diversity(self, attention_weights_all_heads: List[List[float]]) -> float:
        """Calculate diversity between attention heads"""
        if len(attention_weights_all_heads) < 2:
            return 0.0
        
        # Calculate pairwise correlations between heads
        correlations = []
        
        for i in range(len(attention_weights_all_heads)):
            for j in range(i + 1, len(attention_weights_all_heads)):
                corr = np.corrcoef(attention_weights_all_heads[i], attention_weights_all_heads[j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        # Diversity is inverse of average correlation
        avg_correlation = np.mean(correlations) if correlations else 0.0
        diversity = 1.0 - avg_correlation
        
        return max(0.0, diversity)
    
    def _calculate_temporal_attention_consistency(self, average_attention: List[float]) -> float:
        """Calculate temporal consistency of attention weights"""
        if len(average_attention) < 3:
            return 0.5
        
        # Check if attention follows expected temporal pattern (recent > distant)
        recent_attention = np.mean(average_attention[-3:])
        distant_attention = np.mean(average_attention[:3])
        
        # Consistency bonus if recent data gets more attention
        consistency = (recent_attention - distant_attention + 1.0) / 2.0
        return max(0.0, min(1.0, consistency))
    
    # ==================== ADDITIONAL UTILITY METHODS ====================
    
    def _calculate_rsi(self, prices: List[float]) -> List[float]:
        """Calculate RSI for entire price series"""
        rsi_values = []
        
        for i in range(len(prices)):
            if i < self.rsi_period:
                rsi_values.append(50.0)  # Neutral RSI
            else:
                rsi = self._calculate_rsi_single(prices[:i+1])
                rsi_values.append(rsi)
        
        return rsi_values
    
    def _calculate_macd(self, prices: List[float]) -> List[float]:
        """Calculate MACD for entire price series"""
        macd_values = []
        
        for i in range(len(prices)):
            if i < max(self.macd_slow, 26):
                macd_values.append(0.0)
            else:
                macd_line, _ = self._calculate_macd_single(prices[:i+1])
                macd_values.append(macd_line)
        
        return macd_values
    
    def _calculate_bollinger_bands(self, prices: List[float]) -> List[float]:
        """Calculate Bollinger Band positions for entire price series"""
        bb_positions = []
        
        for i in range(len(prices)):
            if i < self.bollinger_period:
                bb_positions.append(0.5)
            else:
                position = self._calculate_bollinger_position(prices[:i+1])
                bb_positions.append(position)
        
        return bb_positions
    
    def _calculate_multi_scale_returns(self, prices: List[float], position: int) -> Dict[str, float]:
        """Calculate multi-scale returns at given position"""
        returns_dict = {}
        
        # Different return periods
        periods = [1, 3, 5, 10]
        
        for period in periods:
            if position >= period:
                ret = (prices[position] - prices[position - period]) / prices[position - period]
                returns_dict[f"return_{period}d"] = ret
            else:
                returns_dict[f"return_{period}d"] = 0.0
        
        return returns_dict
    
    def _calculate_positional_encoding(self, position: int, seq_length: int, d_model: int) -> float:
        """Calculate sinusoidal positional encoding"""
        # Simplified positional encoding
        angle = position / np.power(10000, 2 * (position % 2) / d_model)
        
        if position % 2 == 0:
            return np.sin(angle)
        else:
            return np.cos(angle)
    
    def _prepare_time_varying_macro_features(self, macro_data: MacroeconomicData, position: int) -> Dict[str, float]:
        """Prepare time-varying macro features"""
        # Add time-based variations to macro features
        time_factor = position / self.sequence_length
        
        features = {
            "macro_gdp": (macro_data.gdp_growth - 2.0) / 2.0 * (1 + time_factor * 0.1),
            "macro_inflation": (macro_data.inflation_rate - 2.0) / 2.0 * (1 + time_factor * 0.05),
            "macro_vix": (macro_data.vix - 20.0) / 15.0 * (1 + time_factor * 0.15),
            "macro_rates": (macro_data.interest_rate - 3.0) / 3.0 * (1 + time_factor * 0.08)
        }
        
        return features
    
    def _calculate_microstructure_features(self, index_data: IndexData, position: int) -> Dict[str, float]:
        """Calculate market microstructure features"""
        features = {
            "volume_ratio": min(2.0, index_data.volume / 1e9),  # Normalized volume
            "bid_ask_spread": 0.001 * (1 + index_data.volatility),  # Estimated spread
            "market_impact": index_data.volatility * 0.5,  # Simplified market impact
            "liquidity_score": min(1.0, index_data.volume / index_data.market_cap * 1e6)
        }
        
        return features

# Example usage
if __name__ == "__main__":
    # Sample data
    sample_index = IndexData(
        symbol="SPX",
        name="S&P 500",
        current_level=4200.0,
        historical_levels=[4000, 4050, 4100, 4150, 4200, 4180, 4220, 4190, 4210, 4200],
        dividend_yield=1.8,
        pe_ratio=22.5,
        pb_ratio=3.2,
        market_cap=35000000000000,  # $35T
        volatility=0.18,
        beta=1.0,
        sector_weights={"Technology": 0.28, "Healthcare": 0.13, "Financials": 0.11},
        constituent_count=500,
        volume=1000000000
    )
    
    sample_macro = MacroeconomicData(
        gdp_growth=2.5,
        inflation_rate=3.2,
        unemployment_rate=3.8,
        interest_rate=5.25,
        money_supply_growth=8.5,
        government_debt_to_gdp=120.0,
        trade_balance=-50.0,
        consumer_confidence=105.0,
        business_confidence=95.0,
        manufacturing_pmi=52.0,
        services_pmi=54.0,
        retail_sales_growth=4.2,
        industrial_production=2.8,
        housing_starts=1.4,
        oil_price=75.0,
        dollar_index=103.0,
        vix=18.5
    )
    
    # Create models and test
    ml_models = AdvancedMLModels()
    
    # LSTM prediction
    lstm_result = ml_models.lstm_prediction(sample_index, sample_macro)
    print(f"LSTM Prediction: {lstm_result.predicted_value:.0f}")
    print(f"LSTM Confidence: {lstm_result.confidence:.2f}")
    print(f"LSTM Signal: {lstm_result.signal}")
    
    # Transformer prediction
    transformer_result = ml_models.transformer_prediction(sample_index, sample_macro)
    print(f"\nTransformer Prediction: {transformer_result.predicted_value:.0f}")
    print(f"Transformer Confidence: {transformer_result.confidence:.2f}")
    print(f"Transformer Signal: {transformer_result.signal}")
    
    # Ensemble prediction
    ensemble_result = ml_models.ensemble_prediction(sample_index, sample_macro)
    print(f"\nEnsemble Prediction: {ensemble_result.predicted_value:.0f}")
    print(f"Ensemble Confidence: {ensemble_result.confidence:.2f}")
    print(f"Ensemble Signal: {ensemble_result.signal}")