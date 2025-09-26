from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import math
from enum import Enum

class FuturesIndicatorType(Enum):
    """Futures-specific indicator types"""
    COST_OF_CARRY = "cost_of_carry"  # Cost-of-Carry Model
    CONVENIENCE_YIELD = "convenience_yield"  # Convenience Yield
    SAMUELSON_EFFECT = "samuelson_effect"  # Samuelson Effect
    BACKWARDATION_CONTANGO = "backwardation_contango"  # Market Structure Analysis
    VAR = "var"  # Value at Risk
    GARCH = "garch"  # GARCH Volatility Models
    SEASONAL_ARIMA = "seasonal_arima"  # Seasonal ARIMA
    MOMENTUM = "momentum"  # Momentum Strategy
    MEAN_REVERSION = "mean_reversion"  # Mean Reversion Strategy
    RL_PPO = "rl_ppo"  # Proximal Policy Optimization
    RL_SAC = "rl_sac"  # Soft Actor-Critic
    RL_DDPG = "rl_ddpg"  # Deep Deterministic Policy Gradient
    TERM_STRUCTURE = "term_structure"  # Term Structure Analysis
    BASIS_ANALYSIS = "basis_analysis"  # Basis Analysis
    ROLL_YIELD = "roll_yield"  # Roll Yield Analysis
    VOLATILITY_SURFACE = "volatility_surface"  # Volatility Surface
    CALENDAR_SPREAD = "calendar_spread"  # Calendar Spread Analysis
    STORAGE_COSTS = "storage_costs"  # Storage Cost Analysis
    SUPPLY_DEMAND = "supply_demand"  # Supply/Demand Fundamentals

@dataclass
class FuturesContract:
    """Futures contract information"""
    symbol: str
    underlying_asset: str
    contract_type: str  # "commodity", "financial", "currency", "energy"
    expiration_date: datetime
    current_price: float
    spot_price: float
    risk_free_rate: float
    dividend_yield: float
    storage_cost: float
    convenience_yield: float
    historical_prices: List[float]
    volume: float
    open_interest: float
    volatility: float
    days_to_expiration: int

@dataclass
class MarketData:
    """Market data for futures analysis"""
    supply_data: Dict[str, float]
    demand_data: Dict[str, float]
    inventory_levels: float
    seasonal_factors: Dict[str, float]
    weather_data: Optional[Dict[str, Any]]
    economic_indicators: Dict[str, float]
    geopolitical_risk: float

@dataclass
class FuturesIndicatorResult:
    """Result of futures indicator calculation"""
    indicator_type: FuturesIndicatorType
    value: float
    confidence: float
    metadata: Dict[str, Any]
    timestamp: datetime
    interpretation: str
    risk_level: str
    signal: str  # "BUY", "SELL", "HOLD"
    time_horizon: str

class CostOfCarryModel:
    """Cost-of-Carry Model for Futures Pricing"""
    
    @staticmethod
    def calculate(contract: FuturesContract) -> FuturesIndicatorResult:
        """Calculate theoretical futures price using cost-of-carry model"""
        try:
            S = contract.spot_price
            r = contract.risk_free_rate
            q = contract.dividend_yield
            c = contract.storage_cost
            y = contract.convenience_yield
            T = contract.days_to_expiration / 365.0
            
            # F = S * e^((r - q + c - y) * T)
            carry_cost = (r - q + c - y) * T
            theoretical_price = S * math.exp(carry_cost)
            
            current_price = contract.current_price
            price_difference = current_price - theoretical_price
            percentage_difference = price_difference / theoretical_price
            
            # Generate signal based on mispricing
            if percentage_difference > 0.02:  # Overpriced
                signal = "SELL"
            elif percentage_difference < -0.02:  # Underpriced
                signal = "BUY"
            else:
                signal = "HOLD"
            
            confidence = min(0.9, max(0.5, 1 - abs(percentage_difference) * 5))
            risk_level = "Low" if abs(percentage_difference) < 0.05 else "Medium" if abs(percentage_difference) < 0.1 else "High"
            
            return FuturesIndicatorResult(
                indicator_type=FuturesIndicatorType.COST_OF_CARRY,
                value=theoretical_price,
                confidence=confidence,
                metadata={
                    "spot_price": S,
                    "risk_free_rate": r,
                    "dividend_yield": q,
                    "storage_cost": c,
                    "convenience_yield": y,
                    "time_to_expiration": T,
                    "carry_cost": carry_cost,
                    "current_price": current_price,
                    "price_difference": price_difference,
                    "percentage_difference": percentage_difference
                },
                timestamp=datetime.now(),
                interpretation=f"Theoretical price: {theoretical_price:.2f} vs Current: {current_price:.2f} ({percentage_difference:.2%})",
                risk_level=risk_level,
                signal=signal,
                time_horizon="Until Expiration"
            )
        except Exception as e:
            return FuturesIndicatorResult(
                indicator_type=FuturesIndicatorType.COST_OF_CARRY,
                value=0.0,
                confidence=0.0,
                metadata={"error": str(e)},
                timestamp=datetime.now(),
                interpretation="Cost-of-carry calculation failed",
                risk_level="High",
                signal="HOLD",
                time_horizon="N/A"
            )

class ConvenienceYieldModel:
    """Convenience Yield Analysis"""
    
    @staticmethod
    def calculate(contract: FuturesContract) -> FuturesIndicatorResult:
        """Calculate implied convenience yield"""
        try:
            S = contract.spot_price
            F = contract.current_price
            r = contract.risk_free_rate
            q = contract.dividend_yield
            c = contract.storage_cost
            T = contract.days_to_expiration / 365.0
            
            if T <= 0 or S <= 0:
                raise ValueError("Invalid time to expiration or spot price")
            
            # Solve for convenience yield: F = S * e^((r - q + c - y) * T)
            # y = r - q + c - ln(F/S) / T
            implied_convenience_yield = r - q + c - math.log(F / S) / T
            
            # Compare with historical convenience yield
            historical_cy = contract.convenience_yield
            cy_difference = implied_convenience_yield - historical_cy
            
            # Convenience yield interpretation
            if implied_convenience_yield > 0.05:  # High convenience yield
                interpretation = "High convenience yield suggests tight supply"
                signal = "BUY"  # Bullish for spot
            elif implied_convenience_yield < -0.02:  # Negative convenience yield
                interpretation = "Negative convenience yield suggests oversupply"
                signal = "SELL"  # Bearish for spot
            else:
                interpretation = "Normal convenience yield levels"
                signal = "HOLD"
            
            # Confidence based on contract type and liquidity
            base_confidence = 0.7
            if contract.contract_type in ["commodity", "energy"]:
                base_confidence = 0.8  # Higher confidence for physical commodities
            
            confidence = min(0.9, max(0.4, base_confidence - abs(cy_difference) * 2))
            risk_level = "Low" if abs(implied_convenience_yield) < 0.1 else "Medium" if abs(implied_convenience_yield) < 0.2 else "High"
            
            return FuturesIndicatorResult(
                indicator_type=FuturesIndicatorType.CONVENIENCE_YIELD,
                value=implied_convenience_yield,
                confidence=confidence,
                metadata={
                    "spot_price": S,
                    "futures_price": F,
                    "risk_free_rate": r,
                    "dividend_yield": q,
                    "storage_cost": c,
                    "time_to_expiration": T,
                    "historical_convenience_yield": historical_cy,
                    "cy_difference": cy_difference,
                    "contract_type": contract.contract_type
                },
                timestamp=datetime.now(),
                interpretation=f"{interpretation} (Implied CY: {implied_convenience_yield:.2%})",
                risk_level=risk_level,
                signal=signal,
                time_horizon="Short-term"
            )
        except Exception as e:
            return FuturesIndicatorResult(
                indicator_type=FuturesIndicatorType.CONVENIENCE_YIELD,
                value=0.0,
                confidence=0.0,
                metadata={"error": str(e)},
                timestamp=datetime.now(),
                interpretation="Convenience yield calculation failed",
                risk_level="High",
                signal="HOLD",
                time_horizon="N/A"
            )

class SamuelsonEffect:
    """Samuelson Effect Analysis"""
    
    @staticmethod
    def calculate(contract: FuturesContract, near_contract: Optional[FuturesContract] = None) -> FuturesIndicatorResult:
        """Analyze Samuelson Effect (volatility increases as expiration approaches)"""
        try:
            current_volatility = contract.volatility
            days_to_expiration = contract.days_to_expiration
            
            # Theoretical Samuelson Effect: volatility increases as expiration approaches
            # σ(T) = σ_base * e^(-λ * T)
            # where λ is the decay parameter
            
            lambda_decay = 0.1  # Decay parameter (can be calibrated)
            T = days_to_expiration / 365.0
            
            # Base volatility (long-term)
            base_volatility = 0.2  # 20% annual volatility as baseline
            
            # Expected volatility based on Samuelson Effect
            expected_volatility = base_volatility * math.exp(-lambda_decay * T)
            
            # Compare actual vs expected
            volatility_ratio = current_volatility / expected_volatility
            
            # If we have a near contract, compare volatilities
            if near_contract:
                near_volatility = near_contract.volatility
                near_days = near_contract.days_to_expiration
                
                if near_days < days_to_expiration:
                    volatility_gradient = (near_volatility - current_volatility) / (days_to_expiration - near_days)
                else:
                    volatility_gradient = 0
            else:
                volatility_gradient = 0
            
            # Generate signal based on volatility analysis
            if volatility_ratio > 1.2:  # Higher than expected volatility
                signal = "SELL"  # High volatility might indicate overreaction
                interpretation = "Volatility higher than Samuelson Effect predicts"
            elif volatility_ratio < 0.8:  # Lower than expected volatility
                signal = "BUY"  # Low volatility might indicate underreaction
                interpretation = "Volatility lower than Samuelson Effect predicts"
            else:
                signal = "HOLD"
                interpretation = "Volatility consistent with Samuelson Effect"
            
            confidence = min(0.8, max(0.4, 1 - abs(volatility_ratio - 1) * 2))
            risk_level = "Low" if days_to_expiration > 90 else "Medium" if days_to_expiration > 30 else "High"
            
            return FuturesIndicatorResult(
                indicator_type=FuturesIndicatorType.SAMUELSON_EFFECT,
                value=volatility_ratio,
                confidence=confidence,
                metadata={
                    "current_volatility": current_volatility,
                    "expected_volatility": expected_volatility,
                    "days_to_expiration": days_to_expiration,
                    "lambda_decay": lambda_decay,
                    "volatility_gradient": volatility_gradient,
                    "base_volatility": base_volatility
                },
                timestamp=datetime.now(),
                interpretation=f"{interpretation} (Ratio: {volatility_ratio:.2f})",
                risk_level=risk_level,
                signal=signal,
                time_horizon="Near-term"
            )
        except Exception as e:
            return FuturesIndicatorResult(
                indicator_type=FuturesIndicatorType.SAMUELSON_EFFECT,
                value=0.0,
                confidence=0.0,
                metadata={"error": str(e)},
                timestamp=datetime.now(),
                interpretation="Samuelson Effect calculation failed",
                risk_level="High",
                signal="HOLD",
                time_horizon="N/A"
            )

class BackwardationContangoAnalysis:
    """Backwardation and Contango Analysis"""
    
    @staticmethod
    def calculate(contract: FuturesContract, contracts_curve: List[FuturesContract]) -> FuturesIndicatorResult:
        """Analyze market structure (backwardation vs contango)"""
        try:
            spot_price = contract.spot_price
            current_price = contract.current_price
            
            # Basic backwardation/contango
            if current_price < spot_price:
                market_structure = "Backwardation"
                structure_value = (spot_price - current_price) / spot_price
            else:
                market_structure = "Contango"
                structure_value = (current_price - spot_price) / spot_price
            
            # Analyze term structure if multiple contracts available
            if len(contracts_curve) > 1:
                # Sort contracts by expiration
                sorted_contracts = sorted(contracts_curve, key=lambda x: x.days_to_expiration)
                
                # Calculate slope of futures curve
                prices = [c.current_price for c in sorted_contracts]
                days = [c.days_to_expiration for c in sorted_contracts]
                
                if len(prices) >= 2:
                    # Linear regression for curve slope
                    slope = np.polyfit(days, prices, 1)[0]
                    curve_steepness = abs(slope) * 365  # Annualized slope
                else:
                    slope = 0
                    curve_steepness = 0
            else:
                slope = 0
                curve_steepness = 0
            
            # Generate trading signal
            if market_structure == "Backwardation" and structure_value > 0.05:
                signal = "BUY"  # Strong backwardation often bullish
                interpretation = "Strong backwardation suggests supply tightness"
            elif market_structure == "Contango" and structure_value > 0.05:
                signal = "SELL"  # Strong contango often bearish
                interpretation = "Strong contango suggests oversupply"
            else:
                signal = "HOLD"
                interpretation = f"Mild {market_structure.lower()}"
            
            # Confidence based on structure strength and curve consistency
            confidence = min(0.8, max(0.4, structure_value * 10 + 0.3))
            
            # Risk assessment
            if structure_value > 0.1:
                risk_level = "High"  # Extreme market structures are risky
            elif structure_value > 0.05:
                risk_level = "Medium"
            else:
                risk_level = "Low"
            
            return FuturesIndicatorResult(
                indicator_type=FuturesIndicatorType.BACKWARDATION_CONTANGO,
                value=structure_value,
                confidence=confidence,
                metadata={
                    "market_structure": market_structure,
                    "spot_price": spot_price,
                    "futures_price": current_price,
                    "curve_slope": slope,
                    "curve_steepness": curve_steepness,
                    "contracts_analyzed": len(contracts_curve)
                },
                timestamp=datetime.now(),
                interpretation=f"{interpretation} ({structure_value:.2%})",
                risk_level=risk_level,
                signal=signal,
                time_horizon="Medium-term"
            )
        except Exception as e:
            return FuturesIndicatorResult(
                indicator_type=FuturesIndicatorType.BACKWARDATION_CONTANGO,
                value=0.0,
                confidence=0.0,
                metadata={"error": str(e)},
                timestamp=datetime.now(),
                interpretation="Backwardation/Contango analysis failed",
                risk_level="High",
                signal="HOLD",
                time_horizon="N/A"
            )

class SeasonalARIMA:
    """Seasonal ARIMA Model for Futures"""
    
    @staticmethod
    def calculate(contract: FuturesContract, market_data: MarketData) -> FuturesIndicatorResult:
        """Calculate seasonal ARIMA forecast"""
        try:
            historical_prices = contract.historical_prices
            if len(historical_prices) < 52:  # Need at least 1 year of data
                raise ValueError("Insufficient historical data for seasonal analysis")
            
            # Convert to numpy array for analysis
            prices = np.array(historical_prices)
            
            # Detect seasonality (simplified)
            seasonal_factors = market_data.seasonal_factors
            current_month = datetime.now().month
            
            # Get seasonal factor for current month
            seasonal_factor = seasonal_factors.get(str(current_month), 1.0)
            
            # Simple trend analysis
            recent_prices = prices[-20:]  # Last 20 periods
            trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
            
            # Seasonal adjustment
            current_price = contract.current_price
            seasonally_adjusted_price = current_price / seasonal_factor
            
            # ARIMA components (simplified)
            # AR component: weighted average of recent prices
            ar_component = np.mean(recent_prices[-5:]) * 0.6 + np.mean(recent_prices[-10:-5]) * 0.3 + np.mean(recent_prices[-15:-10]) * 0.1
            
            # MA component: moving average of residuals (simplified)
            ma_component = np.mean(prices[-10:]) - ar_component
            
            # Forecast (simplified SARIMA)
            forecast = ar_component + ma_component * 0.5 + trend * 5  # 5-period ahead
            seasonal_forecast = forecast * seasonal_factor
            
            # Generate signal
            price_change = (seasonal_forecast - current_price) / current_price
            
            if price_change > 0.03:
                signal = "BUY"
            elif price_change < -0.03:
                signal = "SELL"
            else:
                signal = "HOLD"
            
            # Confidence based on seasonal strength and trend consistency
            seasonal_strength = abs(seasonal_factor - 1.0)
            trend_consistency = 1 / (1 + abs(trend) * 100)  # Normalize trend impact
            confidence = min(0.8, max(0.3, seasonal_strength * 2 + trend_consistency * 0.5))
            
            risk_level = "Low" if seasonal_strength > 0.1 else "Medium" if seasonal_strength > 0.05 else "High"
            
            return FuturesIndicatorResult(
                indicator_type=FuturesIndicatorType.SEASONAL_ARIMA,
                value=seasonal_forecast,
                confidence=confidence,
                metadata={
                    "current_price": current_price,
                    "seasonal_factor": seasonal_factor,
                    "seasonally_adjusted_price": seasonally_adjusted_price,
                    "trend": trend,
                    "ar_component": ar_component,
                    "ma_component": ma_component,
                    "price_change": price_change,
                    "forecast_horizon": "5 periods"
                },
                timestamp=datetime.now(),
                interpretation=f"Seasonal ARIMA forecast: {seasonal_forecast:.2f} ({price_change:.2%})",
                risk_level=risk_level,
                signal=signal,
                time_horizon="Short to Medium-term"
            )
        except Exception as e:
            return FuturesIndicatorResult(
                indicator_type=FuturesIndicatorType.SEASONAL_ARIMA,
                value=0.0,
                confidence=0.0,
                metadata={"error": str(e)},
                timestamp=datetime.now(),
                interpretation="Seasonal ARIMA calculation failed",
                risk_level="High",
                signal="HOLD",
                time_horizon="N/A"
            )

class ReinforcementLearningAgents:
    """Reinforcement Learning Agents for Futures Trading"""
    
    @staticmethod
    def ppo_agent(contract: FuturesContract, market_data: MarketData) -> FuturesIndicatorResult:
        """Proximal Policy Optimization Agent"""
        try:
            # State representation
            state_features = {
                "price_momentum": (contract.current_price - np.mean(contract.historical_prices[-10:])) / contract.current_price,
                "volatility": contract.volatility,
                "time_decay": contract.days_to_expiration / 365.0,
                "volume_ratio": contract.volume / (np.mean([contract.volume]) + 1e-6),
                "open_interest_ratio": contract.open_interest / (np.mean([contract.open_interest]) + 1e-6),
                "basis": (contract.current_price - contract.spot_price) / contract.spot_price,
                "inventory_level": market_data.inventory_levels,
                "geopolitical_risk": market_data.geopolitical_risk
            }
            
            # PPO policy network simulation (simplified)
            # Action probabilities: [BUY, HOLD, SELL]
            action_logits = np.array([0.0, 0.0, 0.0])
            
            # Reward calculation based on features
            if state_features["price_momentum"] > 0.02 and state_features["volume_ratio"] > 1.2:
                action_logits[0] += 0.3  # BUY
            if state_features["price_momentum"] < -0.02 and state_features["volume_ratio"] > 1.2:
                action_logits[2] += 0.3  # SELL
            if state_features["volatility"] > 0.3:
                action_logits[1] += 0.2  # HOLD (high volatility)
            if state_features["time_decay"] < 0.1:  # Near expiration
                action_logits[1] += 0.2  # HOLD
            
            # Softmax to get probabilities
            action_probs = np.exp(action_logits) / np.sum(np.exp(action_logits))
            actions = ["BUY", "HOLD", "SELL"]
            best_action = actions[np.argmax(action_probs)]
            action_confidence = np.max(action_probs)
            
            # Expected return calculation
            expected_return = (action_probs[0] - action_probs[2]) * 0.05  # Max 5% expected return
            
            confidence = min(0.8, max(0.3, action_confidence))
            risk_level = "Low" if state_features["volatility"] < 0.2 else "Medium" if state_features["volatility"] < 0.4 else "High"
            
            return FuturesIndicatorResult(
                indicator_type=FuturesIndicatorType.RL_PPO,
                value=expected_return,
                confidence=confidence,
                metadata={
                    "state_features": state_features,
                    "action_probabilities": dict(zip(actions, action_probs)),
                    "best_action": best_action,
                    "action_confidence": action_confidence
                },
                timestamp=datetime.now(),
                interpretation=f"PPO Agent: {best_action} (Expected return: {expected_return:.2%})",
                risk_level=risk_level,
                signal=best_action,
                time_horizon="Short-term"
            )
        except Exception as e:
            return FuturesIndicatorResult(
                indicator_type=FuturesIndicatorType.RL_PPO,
                value=0.0,
                confidence=0.0,
                metadata={"error": str(e)},
                timestamp=datetime.now(),
                interpretation="PPO agent calculation failed",
                risk_level="High",
                signal="HOLD",
                time_horizon="N/A"
            )
    
    @staticmethod
    def sac_agent(contract: FuturesContract, market_data: MarketData) -> FuturesIndicatorResult:
        """Soft Actor-Critic Agent"""
        try:
            # SAC focuses on entropy-regularized policy learning
            state_features = {
                "price_level": contract.current_price / np.mean(contract.historical_prices[-50:]),
                "volatility_regime": min(2.0, contract.volatility / 0.2),  # Normalized volatility
                "liquidity": min(2.0, contract.volume / np.mean([contract.volume])),
                "term_structure": (contract.current_price - contract.spot_price) / contract.spot_price,
                "market_stress": market_data.geopolitical_risk
            }
            
            # SAC Q-values for continuous action space (simplified to discrete)
            q_values = {
                "BUY": 0.0,
                "HOLD": 0.0,
                "SELL": 0.0
            }
            
            # Q-value updates based on state
            if state_features["price_level"] < 0.95 and state_features["liquidity"] > 1.0:
                q_values["BUY"] += 0.4
            if state_features["price_level"] > 1.05 and state_features["liquidity"] > 1.0:
                q_values["SELL"] += 0.4
            if state_features["volatility_regime"] > 1.5:
                q_values["HOLD"] += 0.3
            if state_features["market_stress"] > 0.7:
                q_values["HOLD"] += 0.2
            
            # Entropy regularization (exploration bonus)
            entropy_bonus = 0.1
            for action in q_values:
                q_values[action] += np.random.normal(0, entropy_bonus)
            
            best_action = max(q_values, key=q_values.get)
            best_q_value = q_values[best_action]
            
            confidence = min(0.8, max(0.3, (best_q_value + 1) / 2))  # Normalize Q-value to confidence
            risk_level = "Low" if state_features["volatility_regime"] < 1.2 else "Medium" if state_features["volatility_regime"] < 1.8 else "High"
            
            return FuturesIndicatorResult(
                indicator_type=FuturesIndicatorType.RL_SAC,
                value=best_q_value,
                confidence=confidence,
                metadata={
                    "state_features": state_features,
                    "q_values": q_values,
                    "best_action": best_action,
                    "entropy_bonus": entropy_bonus
                },
                timestamp=datetime.now(),
                interpretation=f"SAC Agent: {best_action} (Q-value: {best_q_value:.3f})",
                risk_level=risk_level,
                signal=best_action,
                time_horizon="Medium-term"
            )
        except Exception as e:
            return FuturesIndicatorResult(
                indicator_type=FuturesIndicatorType.RL_SAC,
                value=0.0,
                confidence=0.0,
                metadata={"error": str(e)},
                timestamp=datetime.now(),
                interpretation="SAC agent calculation failed",
                risk_level="High",
                signal="HOLD",
                time_horizon="N/A"
            )

class FuturesIndicatorEngine:
    """Main engine for calculating futures indicators"""
    
    def __init__(self):
        self.cost_of_carry = CostOfCarryModel()
        self.convenience_yield = ConvenienceYieldModel()
        self.samuelson_effect = SamuelsonEffect()
        self.backwardation_contango = BackwardationContangoAnalysis()
        self.seasonal_arima = SeasonalARIMA()
        self.rl_agents = ReinforcementLearningAgents()
    
    def calculate_all_indicators(self, contract: FuturesContract, 
                               market_data: MarketData,
                               contracts_curve: Optional[List[FuturesContract]] = None,
                               near_contract: Optional[FuturesContract] = None) -> Dict[FuturesIndicatorType, FuturesIndicatorResult]:
        """Calculate all futures indicators"""
        results = {}
        
        # Core pricing models
        results[FuturesIndicatorType.COST_OF_CARRY] = self.cost_of_carry.calculate(contract)
        results[FuturesIndicatorType.CONVENIENCE_YIELD] = self.convenience_yield.calculate(contract)
        results[FuturesIndicatorType.SAMUELSON_EFFECT] = self.samuelson_effect.calculate(contract, near_contract)
        
        # Market structure analysis
        if contracts_curve:
            results[FuturesIndicatorType.BACKWARDATION_CONTANGO] = self.backwardation_contango.calculate(contract, contracts_curve)
        
        # Time series models
        results[FuturesIndicatorType.SEASONAL_ARIMA] = self.seasonal_arima.calculate(contract, market_data)
        
        # Reinforcement learning agents
        results[FuturesIndicatorType.RL_PPO] = self.rl_agents.ppo_agent(contract, market_data)
        results[FuturesIndicatorType.RL_SAC] = self.rl_agents.sac_agent(contract, market_data)
        
        # Additional technical indicators
        results[FuturesIndicatorType.MOMENTUM] = self._calculate_momentum(contract)
        results[FuturesIndicatorType.MEAN_REVERSION] = self._calculate_mean_reversion(contract)
        results[FuturesIndicatorType.ROLL_YIELD] = self._calculate_roll_yield(contract, contracts_curve)
        
        return results
    
    def _calculate_momentum(self, contract: FuturesContract) -> FuturesIndicatorResult:
        """Calculate momentum indicator"""
        try:
            historical_prices = contract.historical_prices
            if len(historical_prices) < 20:
                raise ValueError("Insufficient data for momentum calculation")
            
            current_price = contract.current_price
            price_5d = historical_prices[-5] if len(historical_prices) >= 5 else current_price
            price_20d = historical_prices[-20] if len(historical_prices) >= 20 else current_price
            
            momentum_5d = (current_price - price_5d) / price_5d
            momentum_20d = (current_price - price_20d) / price_20d
            
            # Combined momentum score
            momentum_score = momentum_5d * 0.6 + momentum_20d * 0.4
            
            signal = "BUY" if momentum_score > 0.03 else "SELL" if momentum_score < -0.03 else "HOLD"
            confidence = min(0.7, max(0.3, 1 - contract.volatility))
            risk_level = "Medium"  # Momentum strategies are inherently risky
            
            return FuturesIndicatorResult(
                indicator_type=FuturesIndicatorType.MOMENTUM,
                value=momentum_score,
                confidence=confidence,
                metadata={
                    "momentum_5d": momentum_5d,
                    "momentum_20d": momentum_20d,
                    "current_price": current_price,
                    "price_5d": price_5d,
                    "price_20d": price_20d
                },
                timestamp=datetime.now(),
                interpretation=f"Momentum score: {momentum_score:.2%}",
                risk_level=risk_level,
                signal=signal,
                time_horizon="Short-term"
            )
        except Exception as e:
            return FuturesIndicatorResult(
                indicator_type=FuturesIndicatorType.MOMENTUM,
                value=0.0,
                confidence=0.0,
                metadata={"error": str(e)},
                timestamp=datetime.now(),
                interpretation="Momentum calculation failed",
                risk_level="High",
                signal="HOLD",
                time_horizon="N/A"
            )
    
    def _calculate_mean_reversion(self, contract: FuturesContract) -> FuturesIndicatorResult:
        """Calculate mean reversion indicator"""
        try:
            historical_prices = contract.historical_prices
            if len(historical_prices) < 50:
                raise ValueError("Insufficient data for mean reversion calculation")
            
            long_term_mean = np.mean(historical_prices)
            current_price = contract.current_price
            
            deviation = (current_price - long_term_mean) / long_term_mean
            std_dev = np.std(historical_prices)
            z_score = (current_price - long_term_mean) / std_dev
            
            # Mean reversion signal
            if abs(z_score) > 2:
                signal = "SELL" if z_score > 0 else "BUY"
            elif abs(z_score) > 1:
                signal = "SELL" if z_score > 0 else "BUY"
            else:
                signal = "HOLD"
            
            confidence = min(0.8, max(0.3, abs(z_score) / 3))
            risk_level = "Low" if abs(z_score) > 2 else "Medium" if abs(z_score) > 1 else "High"
            
            return FuturesIndicatorResult(
                indicator_type=FuturesIndicatorType.MEAN_REVERSION,
                value=long_term_mean,
                confidence=confidence,
                metadata={
                    "current_price": current_price,
                    "long_term_mean": long_term_mean,
                    "deviation": deviation,
                    "z_score": z_score,
                    "std_dev": std_dev
                },
                timestamp=datetime.now(),
                interpretation=f"Mean reversion target: {long_term_mean:.2f} (Z-score: {z_score:.2f})",
                risk_level=risk_level,
                signal=signal,
                time_horizon="Medium-term"
            )
        except Exception as e:
            return FuturesIndicatorResult(
                indicator_type=FuturesIndicatorType.MEAN_REVERSION,
                value=0.0,
                confidence=0.0,
                metadata={"error": str(e)},
                timestamp=datetime.now(),
                interpretation="Mean reversion calculation failed",
                risk_level="High",
                signal="HOLD",
                time_horizon="N/A"
            )
    
    def _calculate_roll_yield(self, contract: FuturesContract, 
                            contracts_curve: Optional[List[FuturesContract]] = None) -> FuturesIndicatorResult:
        """Calculate roll yield"""
        try:
            if not contracts_curve or len(contracts_curve) < 2:
                raise ValueError("Need multiple contracts for roll yield calculation")
            
            # Sort by expiration
            sorted_contracts = sorted(contracts_curve, key=lambda x: x.days_to_expiration)
            
            # Find current and next contract
            current_contract = sorted_contracts[0]
            next_contract = sorted_contracts[1] if len(sorted_contracts) > 1 else None
            
            if not next_contract:
                raise ValueError("Need next contract for roll yield calculation")
            
            # Roll yield calculation
            price_diff = next_contract.current_price - current_contract.current_price
            time_diff = (next_contract.days_to_expiration - current_contract.days_to_expiration) / 365.0
            
            if time_diff <= 0:
                raise ValueError("Invalid time difference for roll yield")
            
            roll_yield = (price_diff / current_contract.current_price) / time_diff
            
            # Annualized roll yield
            annualized_roll_yield = roll_yield
            
            signal = "BUY" if annualized_roll_yield < -0.05 else "SELL" if annualized_roll_yield > 0.05 else "HOLD"
            confidence = min(0.7, max(0.4, abs(annualized_roll_yield) * 10))
            risk_level = "Low" if abs(annualized_roll_yield) < 0.1 else "Medium" if abs(annualized_roll_yield) < 0.2 else "High"
            
            return FuturesIndicatorResult(
                indicator_type=FuturesIndicatorType.ROLL_YIELD,
                value=annualized_roll_yield,
                confidence=confidence,
                metadata={
                    "current_price": current_contract.current_price,
                    "next_price": next_contract.current_price,
                    "price_diff": price_diff,
                    "time_diff": time_diff,
                    "current_expiration": current_contract.days_to_expiration,
                    "next_expiration": next_contract.days_to_expiration
                },
                timestamp=datetime.now(),
                interpretation=f"Roll yield: {annualized_roll_yield:.2%} (annualized)",
                risk_level=risk_level,
                signal=signal,
                time_horizon="Roll period"
            )
        except Exception as e:
            return FuturesIndicatorResult(
                indicator_type=FuturesIndicatorType.ROLL_YIELD,
                value=0.0,
                confidence=0.0,
                metadata={"error": str(e)},
                timestamp=datetime.now(),
                interpretation="Roll yield calculation failed",
                risk_level="High",
                signal="HOLD",
                time_horizon="N/A"
            )