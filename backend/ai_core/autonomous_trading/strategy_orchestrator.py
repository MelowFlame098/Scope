# Strategy Orchestrator
# Phase 9: AI-First Platform Implementation

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    CRISIS = "crisis"

class StrategyType(Enum):
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    ARBITRAGE = "arbitrage"
    TREND_FOLLOWING = "trend_following"
    PAIRS_TRADING = "pairs_trading"
    VOLATILITY = "volatility"
    ML_ENSEMBLE = "ml_ensemble"

@dataclass
class TradingSignal:
    symbol: str
    signal_type: str  # BUY, SELL, HOLD
    confidence: float
    strength: float
    strategy_source: StrategyType
    timestamp: datetime
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class StrategyPerformance:
    strategy_type: StrategyType
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_trade_duration: timedelta
    last_updated: datetime
    trades_count: int
    current_weight: float

class StrategyOrchestrator:
    """Coordinates multiple AI trading strategies with ensemble methods"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.strategies = {}
        self.strategy_weights = {}
        self.performance_history = {}
        self.market_regime_detector = MarketRegimeDetector()
        self.ensemble_model = EnsembleSignalProcessor()
        self.current_regime = MarketRegime.SIDEWAYS
        self.regime_confidence = 0.5
        
        # Initialize strategy weights
        self._initialize_strategy_weights()
        
        logger.info("Strategy Orchestrator initialized")
    
    def _initialize_strategy_weights(self):
        """Initialize equal weights for all strategies"""
        strategy_count = len(StrategyType)
        equal_weight = 1.0 / strategy_count
        
        for strategy_type in StrategyType:
            self.strategy_weights[strategy_type] = equal_weight
            self.performance_history[strategy_type] = []
    
    async def generate_signals(self, market_data: Dict[str, Any], 
                             portfolio: Dict[str, Any]) -> List[TradingSignal]:
        """Generate ensemble trading signals from multiple strategies"""
        try:
            # Detect current market regime
            await self._update_market_regime(market_data)
            
            # Generate signals from each strategy
            all_signals = []
            
            for strategy_type in StrategyType:
                if self.strategy_weights[strategy_type] > 0.01:  # Only active strategies
                    signals = await self._generate_strategy_signals(
                        strategy_type, market_data, portfolio
                    )
                    all_signals.extend(signals)
            
            # Process signals through ensemble model
            ensemble_signals = await self.ensemble_model.process_signals(
                all_signals, self.current_regime, self.strategy_weights
            )
            
            # Apply regime-specific adjustments
            adjusted_signals = await self._apply_regime_adjustments(
                ensemble_signals
            )
            
            logger.info(f"Generated {len(adjusted_signals)} ensemble signals")
            return adjusted_signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return []
    
    async def _update_market_regime(self, market_data: Dict[str, Any]):
        """Update current market regime detection"""
        try:
            regime_data = await self.market_regime_detector.detect_regime(
                market_data
            )
            
            self.current_regime = regime_data['regime']
            self.regime_confidence = regime_data['confidence']
            
            # Cache regime data
            if self.redis_client:
                await self.redis_client.setex(
                    "market_regime",
                    300,  # 5 minutes
                    f"{self.current_regime.value}:{self.regime_confidence}"
                )
                
        except Exception as e:
            logger.error(f"Error updating market regime: {e}")
    
    async def _generate_strategy_signals(self, strategy_type: StrategyType,
                                       market_data: Dict[str, Any],
                                       portfolio: Dict[str, Any]) -> List[TradingSignal]:
        """Generate signals for a specific strategy type"""
        try:
            if strategy_type == StrategyType.MOMENTUM:
                return await self._momentum_strategy(market_data, portfolio)
            elif strategy_type == StrategyType.MEAN_REVERSION:
                return await self._mean_reversion_strategy(market_data, portfolio)
            elif strategy_type == StrategyType.ARBITRAGE:
                return await self._arbitrage_strategy(market_data, portfolio)
            elif strategy_type == StrategyType.TREND_FOLLOWING:
                return await self._trend_following_strategy(market_data, portfolio)
            elif strategy_type == StrategyType.PAIRS_TRADING:
                return await self._pairs_trading_strategy(market_data, portfolio)
            elif strategy_type == StrategyType.VOLATILITY:
                return await self._volatility_strategy(market_data, portfolio)
            elif strategy_type == StrategyType.ML_ENSEMBLE:
                return await self._ml_ensemble_strategy(market_data, portfolio)
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error in {strategy_type.value} strategy: {e}")
            return []
    
    async def _momentum_strategy(self, market_data: Dict[str, Any],
                               portfolio: Dict[str, Any]) -> List[TradingSignal]:
        """Momentum-based trading strategy"""
        signals = []
        
        for symbol, data in market_data.items():
            if isinstance(data, dict) and 'prices' in data:
                prices = np.array(data['prices'][-20:])  # Last 20 periods
                
                if len(prices) >= 10:
                    # Calculate momentum indicators
                    short_ma = np.mean(prices[-5:])
                    long_ma = np.mean(prices[-10:])
                    roc = (prices[-1] - prices[-10]) / prices[-10] * 100
                    
                    # Generate signal
                    if short_ma > long_ma and roc > 2.0:
                        signal = TradingSignal(
                            symbol=symbol,
                            signal_type="BUY",
                            confidence=min(abs(roc) / 10, 1.0),
                            strength=0.7,
                            strategy_source=StrategyType.MOMENTUM,
                            timestamp=datetime.now(),
                            target_price=prices[-1] * 1.05,
                            stop_loss=prices[-1] * 0.95
                        )
                        signals.append(signal)
                    elif short_ma < long_ma and roc < -2.0:
                        signal = TradingSignal(
                            symbol=symbol,
                            signal_type="SELL",
                            confidence=min(abs(roc) / 10, 1.0),
                            strength=0.7,
                            strategy_source=StrategyType.MOMENTUM,
                            timestamp=datetime.now(),
                            target_price=prices[-1] * 0.95,
                            stop_loss=prices[-1] * 1.05
                        )
                        signals.append(signal)
        
        return signals
    
    async def _mean_reversion_strategy(self, market_data: Dict[str, Any],
                                     portfolio: Dict[str, Any]) -> List[TradingSignal]:
        """Mean reversion trading strategy"""
        signals = []
        
        for symbol, data in market_data.items():
            if isinstance(data, dict) and 'prices' in data:
                prices = np.array(data['prices'][-30:])  # Last 30 periods
                
                if len(prices) >= 20:
                    # Calculate mean reversion indicators
                    mean_price = np.mean(prices)
                    std_price = np.std(prices)
                    current_price = prices[-1]
                    z_score = (current_price - mean_price) / std_price
                    
                    # Generate signal based on z-score
                    if z_score < -2.0:  # Oversold
                        signal = TradingSignal(
                            symbol=symbol,
                            signal_type="BUY",
                            confidence=min(abs(z_score) / 3, 1.0),
                            strength=0.6,
                            strategy_source=StrategyType.MEAN_REVERSION,
                            timestamp=datetime.now(),
                            target_price=mean_price,
                            stop_loss=current_price * 0.95
                        )
                        signals.append(signal)
                    elif z_score > 2.0:  # Overbought
                        signal = TradingSignal(
                            symbol=symbol,
                            signal_type="SELL",
                            confidence=min(abs(z_score) / 3, 1.0),
                            strength=0.6,
                            strategy_source=StrategyType.MEAN_REVERSION,
                            timestamp=datetime.now(),
                            target_price=mean_price,
                            stop_loss=current_price * 1.05
                        )
                        signals.append(signal)
        
        return signals
    
    async def _arbitrage_strategy(self, market_data: Dict[str, Any],
                                portfolio: Dict[str, Any]) -> List[TradingSignal]:
        """Cross-asset arbitrage strategy"""
        signals = []
        
        # Simple pairs arbitrage example
        symbols = list(market_data.keys())
        
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                symbol1, symbol2 = symbols[i], symbols[j]
                
                if (isinstance(market_data[symbol1], dict) and 
                    isinstance(market_data[symbol2], dict) and
                    'prices' in market_data[symbol1] and 
                    'prices' in market_data[symbol2]):
                    
                    prices1 = np.array(market_data[symbol1]['prices'][-20:])
                    prices2 = np.array(market_data[symbol2]['prices'][-20:])
                    
                    if len(prices1) >= 10 and len(prices2) >= 10:
                        # Calculate price ratio
                        ratio = prices1 / prices2
                        mean_ratio = np.mean(ratio)
                        std_ratio = np.std(ratio)
                        current_ratio = ratio[-1]
                        
                        if std_ratio > 0:
                            z_score = (current_ratio - mean_ratio) / std_ratio
                            
                            if abs(z_score) > 2.0:
                                # Arbitrage opportunity detected
                                if z_score > 2.0:  # Symbol1 overpriced relative to Symbol2
                                    signals.extend([
                                        TradingSignal(
                                            symbol=symbol1,
                                            signal_type="SELL",
                                            confidence=min(abs(z_score) / 3, 1.0),
                                            strength=0.8,
                                            strategy_source=StrategyType.ARBITRAGE,
                                            timestamp=datetime.now()
                                        ),
                                        TradingSignal(
                                            symbol=symbol2,
                                            signal_type="BUY",
                                            confidence=min(abs(z_score) / 3, 1.0),
                                            strength=0.8,
                                            strategy_source=StrategyType.ARBITRAGE,
                                            timestamp=datetime.now()
                                        )
                                    ])
        
        return signals
    
    async def _trend_following_strategy(self, market_data: Dict[str, Any],
                                      portfolio: Dict[str, Any]) -> List[TradingSignal]:
        """Trend following strategy with multiple timeframes"""
        signals = []
        
        for symbol, data in market_data.items():
            if isinstance(data, dict) and 'prices' in data:
                prices = np.array(data['prices'][-50:])  # Last 50 periods
                
                if len(prices) >= 30:
                    # Multiple moving averages
                    ma_short = np.mean(prices[-10:])
                    ma_medium = np.mean(prices[-20:])
                    ma_long = np.mean(prices[-30:])
                    
                    # Trend strength
                    trend_strength = 0
                    if ma_short > ma_medium > ma_long:
                        trend_strength = 1  # Strong uptrend
                    elif ma_short < ma_medium < ma_long:
                        trend_strength = -1  # Strong downtrend
                    
                    if abs(trend_strength) > 0:
                        signal_type = "BUY" if trend_strength > 0 else "SELL"
                        
                        signal = TradingSignal(
                            symbol=symbol,
                            signal_type=signal_type,
                            confidence=0.8,
                            strength=abs(trend_strength),
                            strategy_source=StrategyType.TREND_FOLLOWING,
                            timestamp=datetime.now(),
                            target_price=prices[-1] * (1.1 if trend_strength > 0 else 0.9),
                            stop_loss=prices[-1] * (0.95 if trend_strength > 0 else 1.05)
                        )
                        signals.append(signal)
        
        return signals
    
    async def _pairs_trading_strategy(self, market_data: Dict[str, Any],
                                    portfolio: Dict[str, Any]) -> List[TradingSignal]:
        """Statistical pairs trading strategy"""
        # Implementation similar to arbitrage but with statistical correlation
        return await self._arbitrage_strategy(market_data, portfolio)
    
    async def _volatility_strategy(self, market_data: Dict[str, Any],
                                 portfolio: Dict[str, Any]) -> List[TradingSignal]:
        """Volatility-based trading strategy"""
        signals = []
        
        for symbol, data in market_data.items():
            if isinstance(data, dict) and 'prices' in data:
                prices = np.array(data['prices'][-30:])  # Last 30 periods
                
                if len(prices) >= 20:
                    # Calculate volatility indicators
                    returns = np.diff(prices) / prices[:-1]
                    volatility = np.std(returns) * np.sqrt(252)  # Annualized
                    
                    # Historical volatility percentile
                    hist_vol = []
                    for i in range(10, len(prices)):
                        period_returns = np.diff(prices[i-10:i]) / prices[i-10:i-1]
                        hist_vol.append(np.std(period_returns) * np.sqrt(252))
                    
                    if hist_vol:
                        vol_percentile = np.percentile(hist_vol, 50)
                        
                        # Generate signals based on volatility regime
                        if volatility > vol_percentile * 1.5:  # High volatility
                            # Volatility contraction expected
                            signal = TradingSignal(
                                symbol=symbol,
                                signal_type="HOLD",
                                confidence=0.6,
                                strength=0.5,
                                strategy_source=StrategyType.VOLATILITY,
                                timestamp=datetime.now(),
                                metadata={"volatility_regime": "high", "volatility": volatility}
                            )
                            signals.append(signal)
        
        return signals
    
    async def _ml_ensemble_strategy(self, market_data: Dict[str, Any],
                                  portfolio: Dict[str, Any]) -> List[TradingSignal]:
        """Machine learning ensemble strategy"""
        signals = []
        
        # This would integrate with existing ML models
        # For now, return placeholder signals
        for symbol in list(market_data.keys())[:3]:  # Limit to 3 symbols
            signal = TradingSignal(
                symbol=symbol,
                signal_type="HOLD",
                confidence=0.5,
                strength=0.5,
                strategy_source=StrategyType.ML_ENSEMBLE,
                timestamp=datetime.now(),
                metadata={"model_version": "v1.0"}
            )
            signals.append(signal)
        
        return signals
    
    async def _apply_regime_adjustments(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Apply market regime-specific adjustments to signals"""
        adjusted_signals = []
        
        for signal in signals:
            adjusted_signal = signal
            
            # Adjust based on current market regime
            if self.current_regime == MarketRegime.CRISIS:
                # Reduce position sizes and increase stops in crisis
                adjusted_signal.strength *= 0.5
                adjusted_signal.confidence *= 0.7
            elif self.current_regime == MarketRegime.VOLATILE:
                # Tighter stops in volatile markets
                adjusted_signal.strength *= 0.8
            elif self.current_regime == MarketRegime.BULL:
                # More aggressive in bull markets
                if signal.signal_type == "BUY":
                    adjusted_signal.strength *= 1.2
            elif self.current_regime == MarketRegime.BEAR:
                # More conservative in bear markets
                if signal.signal_type == "SELL":
                    adjusted_signal.strength *= 1.1
            
            adjusted_signals.append(adjusted_signal)
        
        return adjusted_signals
    
    async def update_strategy_performance(self, strategy_type: StrategyType,
                                        performance_data: StrategyPerformance):
        """Update strategy performance and adjust weights"""
        try:
            self.performance_history[strategy_type].append(performance_data)
            
            # Keep only last 100 performance records
            if len(self.performance_history[strategy_type]) > 100:
                self.performance_history[strategy_type] = \
                    self.performance_history[strategy_type][-100:]
            
            # Recalculate strategy weights based on performance
            await self._recalculate_weights()
            
            logger.info(f"Updated performance for {strategy_type.value}")
            
        except Exception as e:
            logger.error(f"Error updating strategy performance: {e}")
    
    async def _recalculate_weights(self):
        """Recalculate strategy weights based on recent performance"""
        try:
            strategy_scores = {}
            
            for strategy_type in StrategyType:
                if self.performance_history[strategy_type]:
                    recent_performance = self.performance_history[strategy_type][-10:]
                    
                    # Calculate composite score
                    avg_return = np.mean([p.total_return for p in recent_performance])
                    avg_sharpe = np.mean([p.sharpe_ratio for p in recent_performance])
                    avg_win_rate = np.mean([p.win_rate for p in recent_performance])
                    
                    # Composite score with equal weights
                    score = (avg_return * 0.4 + avg_sharpe * 0.4 + avg_win_rate * 0.2)
                    strategy_scores[strategy_type] = max(score, 0.01)  # Minimum score
                else:
                    strategy_scores[strategy_type] = 0.1  # Default for new strategies
            
            # Normalize weights
            total_score = sum(strategy_scores.values())
            for strategy_type in StrategyType:
                self.strategy_weights[strategy_type] = \
                    strategy_scores[strategy_type] / total_score
            
            logger.info("Strategy weights recalculated")
            
        except Exception as e:
            logger.error(f"Error recalculating weights: {e}")

class MarketRegimeDetector:
    """Detects current market regime using multiple indicators"""
    
    def __init__(self):
        self.regime_model = None
        self._load_or_create_model()
    
    def _load_or_create_model(self):
        """Load existing model or create new one"""
        try:
            self.regime_model = joblib.load('market_regime_model.pkl')
        except:
            # Create simple model if none exists
            self.regime_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.scaler = StandardScaler()
    
    async def detect_regime(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect current market regime"""
        try:
            # Extract features for regime detection
            features = self._extract_regime_features(market_data)
            
            if features is not None:
                # Simple rule-based regime detection for now
                volatility = features.get('volatility', 0)
                trend_strength = features.get('trend_strength', 0)
                volume_ratio = features.get('volume_ratio', 1)
                
                if volatility > 0.3:
                    if trend_strength < -0.5:
                        regime = MarketRegime.CRISIS
                        confidence = 0.8
                    else:
                        regime = MarketRegime.VOLATILE
                        confidence = 0.7
                elif trend_strength > 0.3:
                    regime = MarketRegime.BULL
                    confidence = 0.8
                elif trend_strength < -0.3:
                    regime = MarketRegime.BEAR
                    confidence = 0.8
                else:
                    regime = MarketRegime.SIDEWAYS
                    confidence = 0.6
            else:
                regime = MarketRegime.SIDEWAYS
                confidence = 0.5
            
            return {
                'regime': regime,
                'confidence': confidence,
                'features': features
            }
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return {
                'regime': MarketRegime.SIDEWAYS,
                'confidence': 0.5,
                'features': None
            }
    
    def _extract_regime_features(self, market_data: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Extract features for regime detection"""
        try:
            all_prices = []
            all_volumes = []
            
            for symbol, data in market_data.items():
                if isinstance(data, dict) and 'prices' in data:
                    prices = data['prices'][-30:]  # Last 30 periods
                    volumes = data.get('volumes', [1] * len(prices))[-30:]
                    
                    all_prices.extend(prices)
                    all_volumes.extend(volumes)
            
            if len(all_prices) < 10:
                return None
            
            prices = np.array(all_prices)
            volumes = np.array(all_volumes)
            
            # Calculate features
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns)
            
            # Trend strength (simple linear regression slope)
            x = np.arange(len(prices))
            trend_strength = np.corrcoef(x, prices)[0, 1]
            
            # Volume analysis
            avg_volume = np.mean(volumes)
            recent_volume = np.mean(volumes[-5:])
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
            
            return {
                'volatility': volatility,
                'trend_strength': trend_strength,
                'volume_ratio': volume_ratio,
                'price_momentum': (prices[-1] - prices[0]) / prices[0]
            }
            
        except Exception as e:
            logger.error(f"Error extracting regime features: {e}")
            return None

class EnsembleSignalProcessor:
    """Processes and combines signals from multiple strategies"""
    
    def __init__(self):
        self.signal_history = []
        self.consensus_threshold = 0.6
    
    async def process_signals(self, signals: List[TradingSignal],
                            regime: MarketRegime,
                            strategy_weights: Dict[StrategyType, float]) -> List[TradingSignal]:
        """Process and combine signals using ensemble methods"""
        try:
            # Group signals by symbol
            symbol_signals = {}
            for signal in signals:
                if signal.symbol not in symbol_signals:
                    symbol_signals[signal.symbol] = []
                symbol_signals[signal.symbol].append(signal)
            
            ensemble_signals = []
            
            for symbol, symbol_signal_list in symbol_signals.items():
                ensemble_signal = await self._create_ensemble_signal(
                    symbol, symbol_signal_list, strategy_weights
                )
                
                if ensemble_signal:
                    ensemble_signals.append(ensemble_signal)
            
            return ensemble_signals
            
        except Exception as e:
            logger.error(f"Error processing ensemble signals: {e}")
            return signals  # Return original signals on error
    
    async def _create_ensemble_signal(self, symbol: str,
                                    signals: List[TradingSignal],
                                    strategy_weights: Dict[StrategyType, float]) -> Optional[TradingSignal]:
        """Create ensemble signal from multiple strategy signals"""
        try:
            if not signals:
                return None
            
            # Calculate weighted consensus
            buy_weight = 0
            sell_weight = 0
            hold_weight = 0
            total_confidence = 0
            total_strength = 0
            
            for signal in signals:
                weight = strategy_weights.get(signal.strategy_source, 0.1)
                signal_weight = weight * signal.confidence * signal.strength
                
                if signal.signal_type == "BUY":
                    buy_weight += signal_weight
                elif signal.signal_type == "SELL":
                    sell_weight += signal_weight
                else:  # HOLD
                    hold_weight += signal_weight
                
                total_confidence += signal.confidence * weight
                total_strength += signal.strength * weight
            
            # Determine ensemble signal
            max_weight = max(buy_weight, sell_weight, hold_weight)
            
            if max_weight < self.consensus_threshold:
                signal_type = "HOLD"
            elif buy_weight == max_weight:
                signal_type = "BUY"
            elif sell_weight == max_weight:
                signal_type = "SELL"
            else:
                signal_type = "HOLD"
            
            # Create ensemble signal
            ensemble_signal = TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=min(total_confidence, 1.0),
                strength=min(total_strength, 1.0),
                strategy_source=StrategyType.ML_ENSEMBLE,  # Mark as ensemble
                timestamp=datetime.now(),
                metadata={
                    "ensemble_weights": {
                        "buy": buy_weight,
                        "sell": sell_weight,
                        "hold": hold_weight
                    },
                    "contributing_strategies": len(signals)
                }
            )
            
            return ensemble_signal
            
        except Exception as e:
            logger.error(f"Error creating ensemble signal for {symbol}: {e}")
            return None