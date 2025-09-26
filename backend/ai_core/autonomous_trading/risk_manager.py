# AI Risk Manager
# Phase 9: AI-First Platform Implementation

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from scipy import stats
import json

from .strategy_orchestrator import TradingSignal, MarketRegime, StrategyType

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EXTREME = "extreme"

class RiskMetric(Enum):
    VAR = "value_at_risk"
    CVAR = "conditional_var"
    MAX_DRAWDOWN = "max_drawdown"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    BETA = "beta"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"

@dataclass
class RiskAssessment:
    symbol: str
    risk_level: RiskLevel
    risk_score: float  # 0-100
    var_95: float  # 95% Value at Risk
    cvar_95: float  # 95% Conditional VaR
    max_drawdown: float
    volatility: float
    beta: float
    correlation_to_market: float
    liquidity_risk: float
    concentration_risk: float
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class PortfolioRisk:
    total_var: float
    portfolio_beta: float
    diversification_ratio: float
    concentration_risk: float
    liquidity_risk: float
    leverage_ratio: float
    risk_budget_utilization: float
    stress_test_results: Dict[str, float]
    timestamp: datetime

@dataclass
class RiskAdjustedSignal:
    original_signal: TradingSignal
    adjusted_signal: TradingSignal
    risk_assessment: RiskAssessment
    position_size_adjustment: float
    risk_justification: str
    timestamp: datetime

class AIRiskManager:
    """Advanced AI-powered risk management system"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.risk_limits = self._initialize_risk_limits()
        self.portfolio_risk_budget = 0.02  # 2% daily VaR limit
        self.max_position_size = 0.05  # 5% max position size
        self.max_sector_exposure = 0.20  # 20% max sector exposure
        self.max_leverage = 2.0  # 2x max leverage
        
        # Risk models
        self.var_model = VaRModel()
        self.stress_tester = StressTester()
        self.correlation_monitor = CorrelationMonitor()
        
        # Historical data for risk calculations
        self.price_history = {}
        self.return_history = {}
        
        logger.info("AI Risk Manager initialized")
    
    def _initialize_risk_limits(self) -> Dict[str, float]:
        """Initialize risk limits for different asset classes"""
        return {
            "equity_var_limit": 0.03,  # 3% daily VaR for equities
            "bond_var_limit": 0.01,   # 1% daily VaR for bonds
            "commodity_var_limit": 0.05,  # 5% daily VaR for commodities
            "crypto_var_limit": 0.10,  # 10% daily VaR for crypto
            "max_correlation": 0.8,   # Max correlation between positions
            "min_liquidity_score": 0.3,  # Minimum liquidity requirement
            "max_concentration": 0.15  # Max single position concentration
        }
    
    async def adjust_signals(self, signals: List[TradingSignal],
                           portfolio: Dict[str, Any]) -> List[RiskAdjustedSignal]:
        """Apply risk management to trading signals"""
        try:
            # Update market data and risk models
            await self._update_risk_models(portfolio)
            
            # Assess portfolio-level risk
            portfolio_risk = await self._assess_portfolio_risk(portfolio)
            
            adjusted_signals = []
            
            for signal in signals:
                # Assess individual signal risk
                risk_assessment = await self._assess_signal_risk(
                    signal, portfolio, portfolio_risk
                )
                
                # Apply risk adjustments
                adjusted_signal = await self._apply_risk_adjustments(
                    signal, risk_assessment, portfolio_risk
                )
                
                # Create risk-adjusted signal
                risk_adjusted = RiskAdjustedSignal(
                    original_signal=signal,
                    adjusted_signal=adjusted_signal,
                    risk_assessment=risk_assessment,
                    position_size_adjustment=self._calculate_position_adjustment(
                        signal, adjusted_signal
                    ),
                    risk_justification=self._generate_risk_justification(
                        risk_assessment
                    ),
                    timestamp=datetime.now()
                )
                
                adjusted_signals.append(risk_adjusted)
            
            # Portfolio-level risk checks
            final_signals = await self._portfolio_risk_check(
                adjusted_signals, portfolio_risk
            )
            
            logger.info(f"Risk-adjusted {len(final_signals)} signals")
            return final_signals
            
        except Exception as e:
            logger.error(f"Error adjusting signals for risk: {e}")
            # Return original signals with minimal risk adjustment
            return [self._create_minimal_adjustment(signal) for signal in signals]
    
    async def _update_risk_models(self, portfolio: Dict[str, Any]):
        """Update risk models with latest market data"""
        try:
            # Update price and return history
            for position in portfolio.get('positions', []):
                symbol = position.get('symbol')
                if symbol:
                    # This would fetch latest price data
                    # For now, simulate with random data
                    if symbol not in self.price_history:
                        self.price_history[symbol] = []
                        self.return_history[symbol] = []
                    
                    # Add latest price (simulated)
                    latest_price = position.get('current_price', 100)
                    self.price_history[symbol].append(latest_price)
                    
                    # Calculate return
                    if len(self.price_history[symbol]) > 1:
                        prev_price = self.price_history[symbol][-2]
                        return_val = (latest_price - prev_price) / prev_price
                        self.return_history[symbol].append(return_val)
                    
                    # Keep only last 252 days (1 year)
                    if len(self.price_history[symbol]) > 252:
                        self.price_history[symbol] = self.price_history[symbol][-252:]
                        self.return_history[symbol] = self.return_history[symbol][-252:]
            
            # Update correlation matrix
            await self.correlation_monitor.update_correlations(self.return_history)
            
        except Exception as e:
            logger.error(f"Error updating risk models: {e}")
    
    async def _assess_portfolio_risk(self, portfolio: Dict[str, Any]) -> PortfolioRisk:
        """Assess overall portfolio risk"""
        try:
            positions = portfolio.get('positions', [])
            total_value = portfolio.get('total_value', 1000000)  # Default 1M
            
            if not positions:
                return PortfolioRisk(
                    total_var=0.0,
                    portfolio_beta=1.0,
                    diversification_ratio=1.0,
                    concentration_risk=0.0,
                    liquidity_risk=0.0,
                    leverage_ratio=1.0,
                    risk_budget_utilization=0.0,
                    stress_test_results={},
                    timestamp=datetime.now()
                )
            
            # Calculate portfolio VaR
            portfolio_returns = self._calculate_portfolio_returns(positions)
            total_var = self.var_model.calculate_portfolio_var(portfolio_returns)
            
            # Calculate portfolio beta
            portfolio_beta = self._calculate_portfolio_beta(positions)
            
            # Calculate concentration risk
            position_weights = [pos.get('weight', 0) for pos in positions]
            concentration_risk = max(position_weights) if position_weights else 0
            
            # Calculate diversification ratio
            diversification_ratio = self._calculate_diversification_ratio(positions)
            
            # Calculate liquidity risk
            liquidity_risk = self._calculate_liquidity_risk(positions)
            
            # Calculate leverage
            leverage_ratio = self._calculate_leverage_ratio(portfolio)
            
            # Risk budget utilization
            risk_budget_utilization = total_var / self.portfolio_risk_budget
            
            # Stress test results
            stress_test_results = await self.stress_tester.run_stress_tests(
                positions, self.return_history
            )
            
            return PortfolioRisk(
                total_var=total_var,
                portfolio_beta=portfolio_beta,
                diversification_ratio=diversification_ratio,
                concentration_risk=concentration_risk,
                liquidity_risk=liquidity_risk,
                leverage_ratio=leverage_ratio,
                risk_budget_utilization=risk_budget_utilization,
                stress_test_results=stress_test_results,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error assessing portfolio risk: {e}")
            return PortfolioRisk(
                total_var=0.02,
                portfolio_beta=1.0,
                diversification_ratio=0.8,
                concentration_risk=0.1,
                liquidity_risk=0.1,
                leverage_ratio=1.0,
                risk_budget_utilization=0.5,
                stress_test_results={},
                timestamp=datetime.now()
            )
    
    async def _assess_signal_risk(self, signal: TradingSignal,
                                portfolio: Dict[str, Any],
                                portfolio_risk: PortfolioRisk) -> RiskAssessment:
        """Assess risk for individual trading signal"""
        try:
            symbol = signal.symbol
            
            # Get historical returns for the symbol
            returns = self.return_history.get(symbol, [])
            
            if len(returns) < 30:  # Need minimum data
                # Use default risk assessment for new symbols
                return self._default_risk_assessment(symbol)
            
            returns_array = np.array(returns)
            
            # Calculate VaR and CVaR
            var_95 = self.var_model.calculate_var(returns_array, confidence=0.95)
            cvar_95 = self.var_model.calculate_cvar(returns_array, confidence=0.95)
            
            # Calculate volatility
            volatility = np.std(returns_array) * np.sqrt(252)  # Annualized
            
            # Calculate beta (simplified - against portfolio)
            portfolio_returns = self._calculate_portfolio_returns(
                portfolio.get('positions', [])
            )
            if len(portfolio_returns) > 0:
                beta = self._calculate_beta(returns_array, portfolio_returns)
            else:
                beta = 1.0
            
            # Calculate max drawdown
            prices = self.price_history.get(symbol, [])
            max_drawdown = self._calculate_max_drawdown(prices)
            
            # Calculate correlation to market
            correlation_to_market = self.correlation_monitor.get_market_correlation(
                symbol
            )
            
            # Calculate liquidity risk
            liquidity_risk = self._calculate_asset_liquidity_risk(symbol)
            
            # Calculate concentration risk
            concentration_risk = self._calculate_asset_concentration_risk(
                symbol, portfolio
            )
            
            # Calculate overall risk score
            risk_score = self._calculate_risk_score(
                var_95, volatility, beta, max_drawdown, liquidity_risk
            )
            
            # Determine risk level
            risk_level = self._determine_risk_level(risk_score)
            
            return RiskAssessment(
                symbol=symbol,
                risk_level=risk_level,
                risk_score=risk_score,
                var_95=var_95,
                cvar_95=cvar_95,
                max_drawdown=max_drawdown,
                volatility=volatility,
                beta=beta,
                correlation_to_market=correlation_to_market,
                liquidity_risk=liquidity_risk,
                concentration_risk=concentration_risk,
                timestamp=datetime.now(),
                metadata={
                    "data_points": len(returns),
                    "signal_strength": signal.strength,
                    "signal_confidence": signal.confidence
                }
            )
            
        except Exception as e:
            logger.error(f"Error assessing signal risk for {signal.symbol}: {e}")
            return self._default_risk_assessment(signal.symbol)
    
    def _default_risk_assessment(self, symbol: str) -> RiskAssessment:
        """Create default risk assessment for new symbols"""
        return RiskAssessment(
            symbol=symbol,
            risk_level=RiskLevel.MEDIUM,
            risk_score=50.0,
            var_95=0.02,
            cvar_95=0.03,
            max_drawdown=0.15,
            volatility=0.20,
            beta=1.0,
            correlation_to_market=0.5,
            liquidity_risk=0.3,
            concentration_risk=0.1,
            timestamp=datetime.now(),
            metadata={"default_assessment": True}
        )
    
    async def _apply_risk_adjustments(self, signal: TradingSignal,
                                    risk_assessment: RiskAssessment,
                                    portfolio_risk: PortfolioRisk) -> TradingSignal:
        """Apply risk-based adjustments to trading signal"""
        try:
            adjusted_signal = TradingSignal(
                symbol=signal.symbol,
                signal_type=signal.signal_type,
                confidence=signal.confidence,
                strength=signal.strength,
                strategy_source=signal.strategy_source,
                timestamp=signal.timestamp,
                target_price=signal.target_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                position_size=signal.position_size,
                metadata=signal.metadata or {}
            )
            
            # Adjust based on risk level
            if risk_assessment.risk_level == RiskLevel.VERY_HIGH:
                adjusted_signal.strength *= 0.3  # Reduce position size significantly
                adjusted_signal.confidence *= 0.5
            elif risk_assessment.risk_level == RiskLevel.HIGH:
                adjusted_signal.strength *= 0.5
                adjusted_signal.confidence *= 0.7
            elif risk_assessment.risk_level == RiskLevel.MEDIUM:
                adjusted_signal.strength *= 0.8
            elif risk_assessment.risk_level == RiskLevel.LOW:
                adjusted_signal.strength *= 1.1  # Slightly increase
            elif risk_assessment.risk_level == RiskLevel.VERY_LOW:
                adjusted_signal.strength *= 1.2
            
            # Adjust stop loss based on volatility
            if signal.stop_loss and risk_assessment.volatility > 0.3:
                # Wider stops for high volatility assets
                if signal.signal_type == "BUY":
                    adjusted_signal.stop_loss = signal.stop_loss * 0.95
                else:
                    adjusted_signal.stop_loss = signal.stop_loss * 1.05
            
            # Portfolio-level adjustments
            if portfolio_risk.risk_budget_utilization > 0.8:
                # Reduce all positions if near risk budget limit
                adjusted_signal.strength *= 0.6
            
            if portfolio_risk.concentration_risk > 0.15:
                # Reduce position if concentration is high
                adjusted_signal.strength *= 0.7
            
            # Liquidity adjustments
            if risk_assessment.liquidity_risk > 0.7:
                # Reduce position for illiquid assets
                adjusted_signal.strength *= 0.5
            
            # Ensure minimum thresholds
            adjusted_signal.strength = max(adjusted_signal.strength, 0.1)
            adjusted_signal.confidence = max(adjusted_signal.confidence, 0.1)
            
            # Add risk metadata
            adjusted_signal.metadata.update({
                "risk_adjusted": True,
                "original_strength": signal.strength,
                "original_confidence": signal.confidence,
                "risk_level": risk_assessment.risk_level.value,
                "risk_score": risk_assessment.risk_score
            })
            
            return adjusted_signal
            
        except Exception as e:
            logger.error(f"Error applying risk adjustments: {e}")
            return signal  # Return original signal on error
    
    def _calculate_position_adjustment(self, original: TradingSignal,
                                     adjusted: TradingSignal) -> float:
        """Calculate position size adjustment ratio"""
        try:
            original_size = original.strength * original.confidence
            adjusted_size = adjusted.strength * adjusted.confidence
            
            if original_size > 0:
                return adjusted_size / original_size
            else:
                return 1.0
                
        except:
            return 1.0
    
    def _generate_risk_justification(self, risk_assessment: RiskAssessment) -> str:
        """Generate human-readable risk justification"""
        justifications = []
        
        if risk_assessment.risk_level == RiskLevel.VERY_HIGH:
            justifications.append("Very high risk asset")
        elif risk_assessment.risk_level == RiskLevel.HIGH:
            justifications.append("High risk asset")
        
        if risk_assessment.volatility > 0.3:
            justifications.append("High volatility")
        
        if risk_assessment.liquidity_risk > 0.7:
            justifications.append("Low liquidity")
        
        if risk_assessment.concentration_risk > 0.15:
            justifications.append("High concentration risk")
        
        if risk_assessment.var_95 > 0.05:
            justifications.append("High VaR")
        
        if not justifications:
            return "Standard risk adjustment applied"
        
        return "; ".join(justifications)
    
    async def _portfolio_risk_check(self, adjusted_signals: List[RiskAdjustedSignal],
                                  portfolio_risk: PortfolioRisk) -> List[RiskAdjustedSignal]:
        """Final portfolio-level risk checks"""
        try:
            # If portfolio risk is too high, reduce all signals
            if portfolio_risk.risk_budget_utilization > 1.0:
                reduction_factor = 0.5
                
                for risk_adjusted in adjusted_signals:
                    risk_adjusted.adjusted_signal.strength *= reduction_factor
                    risk_adjusted.position_size_adjustment *= reduction_factor
                    risk_adjusted.risk_justification += "; Portfolio risk limit exceeded"
            
            # Remove signals that would exceed concentration limits
            final_signals = []
            for risk_adjusted in adjusted_signals:
                if risk_adjusted.risk_assessment.concentration_risk < 0.2:
                    final_signals.append(risk_adjusted)
                else:
                    logger.warning(
                        f"Rejected signal for {risk_adjusted.original_signal.symbol} "
                        f"due to concentration risk"
                    )
            
            return final_signals
            
        except Exception as e:
            logger.error(f"Error in portfolio risk check: {e}")
            return adjusted_signals
    
    def _create_minimal_adjustment(self, signal: TradingSignal) -> RiskAdjustedSignal:
        """Create minimal risk adjustment for error cases"""
        return RiskAdjustedSignal(
            original_signal=signal,
            adjusted_signal=signal,
            risk_assessment=self._default_risk_assessment(signal.symbol),
            position_size_adjustment=1.0,
            risk_justification="Minimal adjustment due to risk calculation error",
            timestamp=datetime.now()
        )
    
    # Helper methods for risk calculations
    def _calculate_portfolio_returns(self, positions: List[Dict]) -> List[float]:
        """Calculate portfolio returns from positions"""
        if not positions:
            return []
        
        # Simplified portfolio return calculation
        portfolio_returns = []
        for i in range(min(50, len(self.return_history.get(positions[0].get('symbol', ''), [])))):
            weighted_return = 0
            total_weight = 0
            
            for position in positions:
                symbol = position.get('symbol')
                weight = position.get('weight', 0)
                
                if symbol in self.return_history and i < len(self.return_history[symbol]):
                    weighted_return += self.return_history[symbol][i] * weight
                    total_weight += weight
            
            if total_weight > 0:
                portfolio_returns.append(weighted_return / total_weight)
        
        return portfolio_returns
    
    def _calculate_portfolio_beta(self, positions: List[Dict]) -> float:
        """Calculate portfolio beta"""
        # Simplified beta calculation
        weighted_beta = 0
        total_weight = 0
        
        for position in positions:
            weight = position.get('weight', 0)
            # Assume beta of 1.0 for simplicity
            beta = 1.0
            
            weighted_beta += beta * weight
            total_weight += weight
        
        return weighted_beta / total_weight if total_weight > 0 else 1.0
    
    def _calculate_diversification_ratio(self, positions: List[Dict]) -> float:
        """Calculate diversification ratio"""
        if len(positions) <= 1:
            return 1.0
        
        # Simplified diversification calculation
        return min(1.0, len(positions) / 10)  # Assume optimal at 10 positions
    
    def _calculate_liquidity_risk(self, positions: List[Dict]) -> float:
        """Calculate portfolio liquidity risk"""
        if not positions:
            return 0.0
        
        # Simplified liquidity risk calculation
        total_liquidity_risk = 0
        for position in positions:
            weight = position.get('weight', 0)
            # Assume moderate liquidity risk for all assets
            liquidity_risk = 0.3
            total_liquidity_risk += liquidity_risk * weight
        
        return total_liquidity_risk
    
    def _calculate_leverage_ratio(self, portfolio: Dict[str, Any]) -> float:
        """Calculate portfolio leverage ratio"""
        total_value = portfolio.get('total_value', 1000000)
        cash = portfolio.get('cash', total_value)
        
        if cash > 0:
            return total_value / cash
        else:
            return 1.0
    
    def _calculate_beta(self, asset_returns: np.ndarray, market_returns: List[float]) -> float:
        """Calculate asset beta against market"""
        try:
            if len(market_returns) == 0 or len(asset_returns) == 0:
                return 1.0
            
            min_length = min(len(asset_returns), len(market_returns))
            asset_ret = asset_returns[-min_length:]
            market_ret = np.array(market_returns[-min_length:])
            
            if len(asset_ret) < 10:  # Need minimum data
                return 1.0
            
            covariance = np.cov(asset_ret, market_ret)[0, 1]
            market_variance = np.var(market_ret)
            
            if market_variance > 0:
                return covariance / market_variance
            else:
                return 1.0
                
        except:
            return 1.0
    
    def _calculate_max_drawdown(self, prices: List[float]) -> float:
        """Calculate maximum drawdown"""
        try:
            if len(prices) < 2:
                return 0.0
            
            prices_array = np.array(prices)
            peak = np.maximum.accumulate(prices_array)
            drawdown = (prices_array - peak) / peak
            
            return abs(np.min(drawdown))
            
        except:
            return 0.15  # Default 15% max drawdown
    
    def _calculate_asset_liquidity_risk(self, symbol: str) -> float:
        """Calculate liquidity risk for individual asset"""
        # Simplified liquidity risk calculation
        # In practice, this would use bid-ask spreads, volume, etc.
        return 0.3  # Default moderate liquidity risk
    
    def _calculate_asset_concentration_risk(self, symbol: str,
                                          portfolio: Dict[str, Any]) -> float:
        """Calculate concentration risk for asset in portfolio"""
        positions = portfolio.get('positions', [])
        
        for position in positions:
            if position.get('symbol') == symbol:
                return position.get('weight', 0)
        
        return 0.0
    
    def _calculate_risk_score(self, var_95: float, volatility: float,
                            beta: float, max_drawdown: float,
                            liquidity_risk: float) -> float:
        """Calculate overall risk score (0-100)"""
        try:
            # Weighted risk score calculation
            var_score = min(var_95 * 1000, 100)  # Scale VaR
            vol_score = min(volatility * 100, 100)  # Scale volatility
            beta_score = min(abs(beta - 1) * 50, 100)  # Beta deviation from 1
            drawdown_score = min(max_drawdown * 100, 100)  # Scale drawdown
            liquidity_score = liquidity_risk * 100  # Scale liquidity risk
            
            # Weighted average
            risk_score = (
                var_score * 0.3 +
                vol_score * 0.25 +
                beta_score * 0.15 +
                drawdown_score * 0.15 +
                liquidity_score * 0.15
            )
            
            return min(max(risk_score, 0), 100)
            
        except:
            return 50.0  # Default medium risk
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level from risk score"""
        if risk_score >= 80:
            return RiskLevel.VERY_HIGH
        elif risk_score >= 60:
            return RiskLevel.HIGH
        elif risk_score >= 40:
            return RiskLevel.MEDIUM
        elif risk_score >= 20:
            return RiskLevel.LOW
        else:
            return RiskLevel.VERY_LOW

class VaRModel:
    """Value at Risk calculation model"""
    
    def calculate_var(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        try:
            if len(returns) == 0:
                return 0.02  # Default 2% VaR
            
            return abs(np.percentile(returns, (1 - confidence) * 100))
            
        except:
            return 0.02
    
    def calculate_cvar(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk"""
        try:
            if len(returns) == 0:
                return 0.03  # Default 3% CVaR
            
            var_threshold = np.percentile(returns, (1 - confidence) * 100)
            tail_returns = returns[returns <= var_threshold]
            
            if len(tail_returns) > 0:
                return abs(np.mean(tail_returns))
            else:
                return abs(var_threshold)
                
        except:
            return 0.03
    
    def calculate_portfolio_var(self, portfolio_returns: List[float]) -> float:
        """Calculate portfolio VaR"""
        try:
            if not portfolio_returns:
                return 0.02
            
            returns_array = np.array(portfolio_returns)
            return self.calculate_var(returns_array)
            
        except:
            return 0.02

class StressTester:
    """Portfolio stress testing"""
    
    async def run_stress_tests(self, positions: List[Dict],
                             return_history: Dict[str, List[float]]) -> Dict[str, float]:
        """Run various stress test scenarios"""
        try:
            stress_results = {}
            
            # Market crash scenario (-20% market drop)
            stress_results['market_crash'] = await self._market_crash_test(
                positions, return_history
            )
            
            # Volatility spike scenario
            stress_results['volatility_spike'] = await self._volatility_spike_test(
                positions, return_history
            )
            
            # Correlation breakdown scenario
            stress_results['correlation_breakdown'] = await self._correlation_breakdown_test(
                positions, return_history
            )
            
            # Liquidity crisis scenario
            stress_results['liquidity_crisis'] = await self._liquidity_crisis_test(
                positions, return_history
            )
            
            return stress_results
            
        except Exception as e:
            logger.error(f"Error running stress tests: {e}")
            return {
                'market_crash': -0.15,
                'volatility_spike': -0.10,
                'correlation_breakdown': -0.08,
                'liquidity_crisis': -0.12
            }
    
    async def _market_crash_test(self, positions: List[Dict],
                               return_history: Dict[str, List[float]]) -> float:
        """Simulate market crash scenario"""
        # Simplified: assume all positions drop by 20%
        return -0.20
    
    async def _volatility_spike_test(self, positions: List[Dict],
                                   return_history: Dict[str, List[float]]) -> float:
        """Simulate volatility spike scenario"""
        # Simplified: assume 10% loss due to volatility
        return -0.10
    
    async def _correlation_breakdown_test(self, positions: List[Dict],
                                        return_history: Dict[str, List[float]]) -> float:
        """Simulate correlation breakdown scenario"""
        # Simplified: assume 8% loss due to correlation changes
        return -0.08
    
    async def _liquidity_crisis_test(self, positions: List[Dict],
                                   return_history: Dict[str, List[float]]) -> float:
        """Simulate liquidity crisis scenario"""
        # Simplified: assume 12% loss due to liquidity issues
        return -0.12

class CorrelationMonitor:
    """Monitor asset correlations"""
    
    def __init__(self):
        self.correlation_matrix = {}
        self.market_correlations = {}
    
    async def update_correlations(self, return_history: Dict[str, List[float]]):
        """Update correlation matrix"""
        try:
            symbols = list(return_history.keys())
            
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols[i+1:], i+1):
                    returns1 = return_history[symbol1]
                    returns2 = return_history[symbol2]
                    
                    if len(returns1) > 10 and len(returns2) > 10:
                        min_length = min(len(returns1), len(returns2))
                        corr = np.corrcoef(
                            returns1[-min_length:],
                            returns2[-min_length:]
                        )[0, 1]
                        
                        if not np.isnan(corr):
                            self.correlation_matrix[f"{symbol1}_{symbol2}"] = corr
            
            # Update market correlations (simplified)
            for symbol in symbols:
                self.market_correlations[symbol] = 0.5  # Default correlation
                
        except Exception as e:
            logger.error(f"Error updating correlations: {e}")
    
    def get_market_correlation(self, symbol: str) -> float:
        """Get correlation to market for a symbol"""
        return self.market_correlations.get(symbol, 0.5)
    
    def get_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation between two symbols"""
        key1 = f"{symbol1}_{symbol2}"
        key2 = f"{symbol2}_{symbol1}"
        
        return self.correlation_matrix.get(key1, 
               self.correlation_matrix.get(key2, 0.0))