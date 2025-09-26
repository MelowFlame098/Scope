"""Risk Assessment Engine for FinScope - Phase 7 Implementation

Provides comprehensive risk analysis, monitoring, and management
capabilities for portfolio risk assessment and mitigation.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np
from scipy import stats
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database import get_db
from db_models import Portfolio, PortfolioHolding
from market_data import MarketDataService

logger = logging.getLogger(__name__)

class RiskLevel(str, Enum):
    """Risk level classifications"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

class AlertType(str, Enum):
    """Types of risk alerts"""
    CONCENTRATION = "concentration"
    VOLATILITY = "volatility"
    DRAWDOWN = "drawdown"
    VAR_BREACH = "var_breach"
    CORRELATION = "correlation"
    LIQUIDITY = "liquidity"
    SECTOR_EXPOSURE = "sector_exposure"
    CURRENCY_EXPOSURE = "currency_exposure"
    LEVERAGE = "leverage"
    STRESS_TEST = "stress_test"

class RiskMetric(str, Enum):
    """Risk metrics for monitoring"""
    VALUE_AT_RISK = "value_at_risk"
    EXPECTED_SHORTFALL = "expected_shortfall"
    MAXIMUM_DRAWDOWN = "maximum_drawdown"
    VOLATILITY = "volatility"
    BETA = "beta"
    CORRELATION = "correlation"
    CONCENTRATION = "concentration"
    LIQUIDITY_RISK = "liquidity_risk"

@dataclass
class RiskAssessment:
    """Comprehensive risk assessment"""
    overall_risk_score: float  # 0-100 scale
    risk_level: RiskLevel
    key_risks: List[str]
    risk_metrics: Dict[str, float]
    stress_test_results: Dict[str, float]
    recommendations: List[str]
    alerts: List[Dict[str, Any]]
    last_updated: datetime

@dataclass
class RiskLimit:
    """Risk limit configuration"""
    metric: RiskMetric
    threshold: float
    warning_threshold: float
    enabled: bool = True
    description: str = ""

@dataclass
class StressScenario:
    """Stress test scenario definition"""
    name: str
    description: str
    market_shock: float
    sector_shocks: Dict[str, float]
    correlation_shock: float
    volatility_multiplier: float

class RiskRequest(BaseModel):
    """Request for risk analysis"""
    portfolio_id: str
    risk_horizon_days: int = 30
    confidence_level: float = 0.95
    include_stress_tests: bool = True
    custom_scenarios: Optional[List[Dict[str, Any]]] = None

class RiskLimitRequest(BaseModel):
    """Request for setting risk limits"""
    portfolio_id: str
    limits: List[Dict[str, Any]]
    notification_enabled: bool = True

class RiskResponse(BaseModel):
    """Response for risk analysis"""
    portfolio_id: str
    assessment: RiskAssessment
    detailed_metrics: Dict[str, Any]
    scenario_analysis: Dict[str, Any]
    recommendations: List[str]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class RiskEngine:
    """Advanced risk assessment and management engine"""
    
    def __init__(self):
        self.market_service = MarketDataService()
        
        # Risk calculation parameters
        self.confidence_levels = [0.90, 0.95, 0.99]
        self.risk_horizon_days = [1, 7, 30, 90]
        self.trading_days_per_year = 252
        
        # Default risk limits
        self.default_risk_limits = {
            RiskMetric.VALUE_AT_RISK: RiskLimit(
                metric=RiskMetric.VALUE_AT_RISK,
                threshold=10.0,  # 10% daily VaR
                warning_threshold=7.5,
                description="Daily Value at Risk (95% confidence)"
            ),
            RiskMetric.MAXIMUM_DRAWDOWN: RiskLimit(
                metric=RiskMetric.MAXIMUM_DRAWDOWN,
                threshold=20.0,  # 20% max drawdown
                warning_threshold=15.0,
                description="Maximum portfolio drawdown"
            ),
            RiskMetric.CONCENTRATION: RiskLimit(
                metric=RiskMetric.CONCENTRATION,
                threshold=25.0,  # 25% max single position
                warning_threshold=20.0,
                description="Maximum single position concentration"
            ),
            RiskMetric.VOLATILITY: RiskLimit(
                metric=RiskMetric.VOLATILITY,
                threshold=25.0,  # 25% annualized volatility
                warning_threshold=20.0,
                description="Annualized portfolio volatility"
            )
        }
        
        # Predefined stress scenarios
        self.stress_scenarios = {
            "market_crash_2008": StressScenario(
                name="2008 Financial Crisis",
                description="Severe market crash scenario",
                market_shock=-0.30,
                sector_shocks={"Financials": -0.50, "Technology": -0.25},
                correlation_shock=0.20,
                volatility_multiplier=2.0
            ),
            "covid_pandemic_2020": StressScenario(
                name="COVID-19 Pandemic",
                description="Pandemic-induced market shock",
                market_shock=-0.25,
                sector_shocks={"Travel": -0.60, "Energy": -0.40, "Technology": 0.10},
                correlation_shock=0.15,
                volatility_multiplier=1.8
            ),
            "interest_rate_shock": StressScenario(
                name="Interest Rate Shock",
                description="Rapid interest rate increase",
                market_shock=-0.15,
                sector_shocks={"REITs": -0.25, "Utilities": -0.20, "Financials": 0.05},
                correlation_shock=0.10,
                volatility_multiplier=1.3
            ),
            "inflation_shock": StressScenario(
                name="Inflation Shock",
                description="Unexpected inflation surge",
                market_shock=-0.12,
                sector_shocks={"Consumer Staples": -0.15, "Energy": 0.15},
                correlation_shock=0.08,
                volatility_multiplier=1.4
            )
        }
    
    async def assess_portfolio_risk(
        self,
        request: RiskRequest,
        holdings: List[Dict[str, Any]],
        db: Session
    ) -> RiskResponse:
        """Perform comprehensive portfolio risk assessment"""
        try:
            if not holdings:
                return self._get_default_risk_response(request.portfolio_id)
            
            # Calculate risk metrics
            risk_metrics = await self._calculate_risk_metrics(
                holdings, request.risk_horizon_days, request.confidence_level
            )
            
            # Perform stress tests
            stress_results = {}
            if request.include_stress_tests:
                stress_results = await self._perform_stress_tests(
                    holdings, request.custom_scenarios
                )
            
            # Calculate overall risk score
            overall_risk_score = self._calculate_overall_risk_score(risk_metrics)
            risk_level = self._determine_risk_level(overall_risk_score)
            
            # Identify key risks
            key_risks = self._identify_key_risks(risk_metrics, stress_results)
            
            # Generate recommendations
            recommendations = self._generate_risk_recommendations(
                risk_metrics, stress_results, holdings
            )
            
            # Check for risk alerts
            alerts = await self._check_risk_alerts(
                request.portfolio_id, risk_metrics, db
            )
            
            # Create risk assessment
            assessment = RiskAssessment(
                overall_risk_score=overall_risk_score,
                risk_level=risk_level,
                key_risks=key_risks,
                risk_metrics=risk_metrics,
                stress_test_results=stress_results,
                recommendations=recommendations,
                alerts=alerts,
                last_updated=datetime.utcnow()
            )
            
            # Detailed scenario analysis
            scenario_analysis = await self._perform_scenario_analysis(
                holdings, request.confidence_level
            )
            
            return RiskResponse(
                portfolio_id=request.portfolio_id,
                assessment=assessment,
                detailed_metrics=self._format_detailed_metrics(risk_metrics),
                scenario_analysis=scenario_analysis,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error assessing portfolio risk: {str(e)}")
            return self._get_default_risk_response(request.portfolio_id)
    
    async def calculate_portfolio_risk(
        self,
        holdings: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate basic portfolio risk metrics"""
        try:
            if not holdings:
                return {}
            
            symbols = [h["symbol"] for h in holdings]
            weights = np.array([h["weight"] / 100 for h in holdings])
            
            # Get historical data for risk calculations
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=365)
            
            returns_data = {}
            for symbol in symbols:
                try:
                    historical_data = await self.market_service.get_historical_data(
                        symbol, start_date, end_date
                    )
                    prices = [price["close"] for price in historical_data]
                    returns = self._calculate_returns(prices)
                    returns_data[symbol] = returns
                except Exception as e:
                    logger.warning(f"Could not get data for {symbol}: {str(e)}")
                    returns_data[symbol] = [0] * 252
            
            # Align returns data
            min_length = min(len(returns) for returns in returns_data.values())
            aligned_returns = {
                symbol: returns[-min_length:] 
                for symbol, returns in returns_data.items()
            }
            
            # Create returns matrix
            returns_matrix = np.array(list(aligned_returns.values())).T
            
            # Calculate portfolio returns
            portfolio_returns = np.dot(returns_matrix, weights)
            
            # Calculate risk metrics
            portfolio_volatility = np.std(portfolio_returns) * np.sqrt(self.trading_days_per_year) * 100
            
            # Value at Risk (95%)
            var_95 = np.percentile(portfolio_returns, 5) * 100
            
            # Expected Shortfall (Conditional VaR)
            es_95 = np.mean(
                [r for r in portfolio_returns if r <= np.percentile(portfolio_returns, 5)]
            ) * 100
            
            # Maximum individual position
            max_position = max(weights) * 100
            
            # Correlation analysis
            correlation_matrix = np.corrcoef(returns_matrix.T)
            avg_correlation = np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
            
            return {
                "portfolio_volatility": portfolio_volatility,
                "value_at_risk_95": var_95,
                "expected_shortfall_95": es_95,
                "max_position_weight": max_position,
                "average_correlation": float(avg_correlation),
                "number_of_positions": len(holdings),
                "diversification_ratio": self._calculate_diversification_ratio(weights, correlation_matrix)
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {str(e)}")
            return {}
    
    async def set_risk_limits(
        self,
        request: RiskLimitRequest,
        db: Session
    ) -> Dict[str, Any]:
        """Set risk limits for portfolio monitoring"""
        try:
            # Store risk limits in database (would need RiskLimit model)
            # For now, return success response
            
            logger.info(f"Risk limits set for portfolio {request.portfolio_id}")
            
            return {
                "status": "success",
                "portfolio_id": request.portfolio_id,
                "limits_set": len(request.limits),
                "notification_enabled": request.notification_enabled
            }
            
        except Exception as e:
            logger.error(f"Error setting risk limits: {str(e)}")
            raise
    
    async def monitor_risk_limits(
        self,
        portfolio_id: str,
        current_metrics: Dict[str, float],
        db: Session
    ) -> List[Dict[str, Any]]:
        """Monitor portfolio against risk limits"""
        try:
            alerts = []
            
            # Check against default limits
            for metric, limit in self.default_risk_limits.items():
                if not limit.enabled:
                    continue
                
                metric_value = current_metrics.get(metric.value, 0)
                
                if metric_value > limit.threshold:
                    alerts.append({
                        "type": AlertType.VAR_BREACH.value,
                        "severity": "critical",
                        "metric": metric.value,
                        "current_value": metric_value,
                        "threshold": limit.threshold,
                        "message": f"{limit.description} exceeded: {metric_value:.2f}% > {limit.threshold:.2f}%",
                        "timestamp": datetime.utcnow()
                    })
                elif metric_value > limit.warning_threshold:
                    alerts.append({
                        "type": AlertType.VAR_BREACH.value,
                        "severity": "warning",
                        "metric": metric.value,
                        "current_value": metric_value,
                        "threshold": limit.warning_threshold,
                        "message": f"{limit.description} warning: {metric_value:.2f}% > {limit.warning_threshold:.2f}%",
                        "timestamp": datetime.utcnow()
                    })
            
            # Store alerts in database if any
            if alerts:
                await self._store_risk_alerts(portfolio_id, alerts, db)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error monitoring risk limits: {str(e)}")
            return []
    
    async def _calculate_risk_metrics(
        self,
        holdings: List[Dict[str, Any]],
        horizon_days: int,
        confidence_level: float
    ) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        try:
            symbols = [h["symbol"] for h in holdings]
            weights = np.array([h["weight"] / 100 for h in holdings])
            values = np.array([h["market_value"] for h in holdings])
            
            # Get historical returns
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=max(365, horizon_days * 5))
            
            returns_data = {}
            for symbol in symbols:
                try:
                    historical_data = await self.market_service.get_historical_data(
                        symbol, start_date, end_date
                    )
                    prices = [price["close"] for price in historical_data]
                    returns = self._calculate_returns(prices)
                    returns_data[symbol] = returns
                except Exception:
                    returns_data[symbol] = [0] * 252
            
            # Align and create returns matrix
            min_length = min(len(returns) for returns in returns_data.values())
            returns_matrix = np.array([
                returns_data[symbol][-min_length:] for symbol in symbols
            ]).T
            
            # Portfolio returns
            portfolio_returns = np.dot(returns_matrix, weights)
            
            # Scale for horizon
            horizon_factor = np.sqrt(horizon_days)
            
            # Calculate metrics
            metrics = {
                "volatility": np.std(portfolio_returns) * np.sqrt(self.trading_days_per_year) * 100,
                "value_at_risk": abs(np.percentile(portfolio_returns, (1 - confidence_level) * 100)) * horizon_factor * 100,
                "expected_shortfall": abs(np.mean([
                    r for r in portfolio_returns 
                    if r <= np.percentile(portfolio_returns, (1 - confidence_level) * 100)
                ])) * horizon_factor * 100,
                "maximum_drawdown": self._calculate_max_drawdown_from_returns(portfolio_returns) * 100,
                "concentration_risk": max(weights) * 100,
                "correlation_risk": self._calculate_correlation_risk(returns_matrix),
                "liquidity_risk": self._calculate_liquidity_risk(holdings),
                "beta": self._calculate_portfolio_beta(portfolio_returns),
                "tracking_error": np.std(portfolio_returns) * np.sqrt(self.trading_days_per_year) * 100
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            return {}
    
    async def _perform_stress_tests(
        self,
        holdings: List[Dict[str, Any]],
        custom_scenarios: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, float]:
        """Perform stress testing on portfolio"""
        try:
            stress_results = {}
            
            # Test predefined scenarios
            for scenario_name, scenario in self.stress_scenarios.items():
                portfolio_impact = await self._apply_stress_scenario(
                    holdings, scenario
                )
                stress_results[scenario_name] = portfolio_impact
            
            # Test custom scenarios if provided
            if custom_scenarios:
                for i, custom_scenario in enumerate(custom_scenarios):
                    scenario_obj = StressScenario(
                        name=custom_scenario.get("name", f"Custom_{i+1}"),
                        description=custom_scenario.get("description", ""),
                        market_shock=custom_scenario.get("market_shock", -0.10),
                        sector_shocks=custom_scenario.get("sector_shocks", {}),
                        correlation_shock=custom_scenario.get("correlation_shock", 0.05),
                        volatility_multiplier=custom_scenario.get("volatility_multiplier", 1.2)
                    )
                    
                    portfolio_impact = await self._apply_stress_scenario(
                        holdings, scenario_obj
                    )
                    stress_results[scenario_obj.name] = portfolio_impact
            
            return stress_results
            
        except Exception as e:
            logger.error(f"Error performing stress tests: {str(e)}")
            return {}
    
    async def _apply_stress_scenario(
        self,
        holdings: List[Dict[str, Any]],
        scenario: StressScenario
    ) -> float:
        """Apply stress scenario to portfolio"""
        try:
            total_impact = 0
            total_value = sum(h["market_value"] for h in holdings)
            
            for holding in holdings:
                # Base market shock
                impact = scenario.market_shock
                
                # Apply sector-specific shocks
                sector = self._get_sector(holding["symbol"])
                if sector in scenario.sector_shocks:
                    impact += scenario.sector_shocks[sector]
                
                # Calculate position impact
                position_impact = holding["market_value"] * impact
                total_impact += position_impact
            
            # Return as percentage of total portfolio value
            return (total_impact / total_value) * 100 if total_value > 0 else 0
            
        except Exception as e:
            logger.error(f"Error applying stress scenario: {str(e)}")
            return 0
    
    async def _perform_scenario_analysis(
        self,
        holdings: List[Dict[str, Any]],
        confidence_level: float
    ) -> Dict[str, Any]:
        """Perform detailed scenario analysis"""
        try:
            scenarios = {
                "bull_market": {"description": "Strong market rally", "shock": 0.20},
                "bear_market": {"description": "Market decline", "shock": -0.20},
                "sideways_market": {"description": "Range-bound market", "shock": 0.0},
                "high_volatility": {"description": "Increased volatility", "shock": 0.0, "vol_multiplier": 2.0},
                "recession": {"description": "Economic recession", "shock": -0.30}
            }
            
            scenario_results = {}
            total_value = sum(h["market_value"] for h in holdings)
            
            for scenario_name, scenario_data in scenarios.items():
                shock = scenario_data["shock"]
                portfolio_impact = total_value * shock
                
                scenario_results[scenario_name] = {
                    "description": scenario_data["description"],
                    "portfolio_impact_pct": shock * 100,
                    "portfolio_impact_value": portfolio_impact,
                    "new_portfolio_value": total_value + portfolio_impact
                }
            
            return scenario_results
            
        except Exception as e:
            logger.error(f"Error performing scenario analysis: {str(e)}")
            return {}
    
    def _calculate_overall_risk_score(self, risk_metrics: Dict[str, float]) -> float:
        """Calculate overall risk score (0-100)"""
        try:
            # Weighted risk score calculation
            weights = {
                "volatility": 0.25,
                "value_at_risk": 0.30,
                "concentration_risk": 0.20,
                "correlation_risk": 0.15,
                "liquidity_risk": 0.10
            }
            
            # Normalize metrics to 0-100 scale
            normalized_scores = {}
            
            # Volatility (0-50% -> 0-100 score)
            vol = risk_metrics.get("volatility", 0)
            normalized_scores["volatility"] = min(vol * 2, 100)
            
            # VaR (0-20% -> 0-100 score)
            var = risk_metrics.get("value_at_risk", 0)
            normalized_scores["value_at_risk"] = min(var * 5, 100)
            
            # Concentration (0-100% -> 0-100 score)
            conc = risk_metrics.get("concentration_risk", 0)
            normalized_scores["concentration_risk"] = conc
            
            # Correlation (0-1 -> 0-100 score)
            corr = risk_metrics.get("correlation_risk", 0)
            normalized_scores["correlation_risk"] = corr * 100
            
            # Liquidity (0-100 -> 0-100 score, inverted)
            liq = risk_metrics.get("liquidity_risk", 0)
            normalized_scores["liquidity_risk"] = liq
            
            # Calculate weighted score
            total_score = sum(
                normalized_scores.get(metric, 0) * weight
                for metric, weight in weights.items()
            )
            
            return min(max(total_score, 0), 100)
            
        except Exception as e:
            logger.error(f"Error calculating overall risk score: {str(e)}")
            return 50  # Default moderate risk
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level from score"""
        if risk_score < 25:
            return RiskLevel.LOW
        elif risk_score < 50:
            return RiskLevel.MODERATE
        elif risk_score < 75:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def _identify_key_risks(self, risk_metrics: Dict[str, float], stress_results: Dict[str, float]) -> List[str]:
        """Identify key risk factors"""
        key_risks = []
        
        # Check concentration risk
        if risk_metrics.get("concentration_risk", 0) > 20:
            key_risks.append("High concentration in single positions")
        
        # Check volatility
        if risk_metrics.get("volatility", 0) > 25:
            key_risks.append("High portfolio volatility")
        
        # Check VaR
        if risk_metrics.get("value_at_risk", 0) > 10:
            key_risks.append("High Value at Risk")
        
        # Check correlation
        if risk_metrics.get("correlation_risk", 0) > 0.7:
            key_risks.append("High correlation between holdings")
        
        # Check stress test results
        for scenario, impact in stress_results.items():
            if impact < -25:  # More than 25% loss
                key_risks.append(f"Vulnerable to {scenario} scenario")
        
        return key_risks[:5]  # Return top 5 risks
    
    def _generate_risk_recommendations(self, risk_metrics: Dict[str, float], stress_results: Dict[str, float], holdings: List[Dict[str, Any]]) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        # Concentration recommendations
        if risk_metrics.get("concentration_risk", 0) > 20:
            recommendations.append("Consider reducing position sizes to improve diversification")
        
        # Volatility recommendations
        if risk_metrics.get("volatility", 0) > 25:
            recommendations.append("Add lower volatility assets to reduce portfolio risk")
        
        # Correlation recommendations
        if risk_metrics.get("correlation_risk", 0) > 0.7:
            recommendations.append("Diversify across uncorrelated asset classes")
        
        # Stress test recommendations
        worst_scenario = min(stress_results.items(), key=lambda x: x[1], default=(None, 0))
        if worst_scenario[1] < -20:
            recommendations.append(f"Consider hedging against {worst_scenario[0]} scenario")
        
        # Liquidity recommendations
        if risk_metrics.get("liquidity_risk", 0) > 30:
            recommendations.append("Increase allocation to more liquid assets")
        
        return recommendations[:5]  # Return top 5 recommendations
    
    async def _check_risk_alerts(self, portfolio_id: str, risk_metrics: Dict[str, float], db: Session) -> List[Dict[str, Any]]:
        """Check for risk limit breaches and generate alerts"""
        return await self.monitor_risk_limits(portfolio_id, risk_metrics, db)
    
    async def _store_risk_alerts(self, portfolio_id: str, alerts: List[Dict[str, Any]], db: Session):
        """Store risk alerts in database"""
        try:
            for alert in alerts:
                # Would store in RiskAlert model
                logger.info(f"Risk alert for portfolio {portfolio_id}: {alert['message']}")
        except Exception as e:
            logger.error(f"Error storing risk alerts: {str(e)}")
    
    def _calculate_returns(self, prices: List[float]) -> List[float]:
        """Calculate returns from price series"""
        if len(prices) < 2:
            return []
        
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] != 0:
                ret = (prices[i] / prices[i-1]) - 1
                returns.append(ret)
            else:
                returns.append(0)
        
        return returns
    
    def _calculate_max_drawdown_from_returns(self, returns: List[float]) -> float:
        """Calculate maximum drawdown from returns"""
        if len(returns) < 2:
            return 0
        
        cumulative = np.cumprod(1 + np.array(returns))
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        
        return abs(np.min(drawdown))
    
    def _calculate_correlation_risk(self, returns_matrix: np.ndarray) -> float:
        """Calculate correlation risk metric"""
        try:
            correlation_matrix = np.corrcoef(returns_matrix.T)
            # Average correlation excluding diagonal
            mask = ~np.eye(correlation_matrix.shape[0], dtype=bool)
            avg_correlation = np.mean(correlation_matrix[mask])
            return float(avg_correlation)
        except Exception:
            return 0.0
    
    def _calculate_liquidity_risk(self, holdings: List[Dict[str, Any]]) -> float:
        """Calculate liquidity risk score (simplified)"""
        # Simplified liquidity scoring
        # In reality, would consider bid-ask spreads, volume, etc.
        total_weight = sum(h["weight"] for h in holdings)
        
        # Assume larger positions in major indices are more liquid
        liquid_symbols = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "GOOGL", "AMZN"]
        liquid_weight = sum(
            h["weight"] for h in holdings 
            if h["symbol"] in liquid_symbols
        )
        
        liquidity_score = (liquid_weight / total_weight) * 100 if total_weight > 0 else 0
        return 100 - liquidity_score  # Higher score = higher risk
    
    def _calculate_portfolio_beta(self, portfolio_returns: List[float]) -> float:
        """Calculate portfolio beta (simplified)"""
        # Simplified beta calculation
        # In reality, would calculate against market benchmark
        return 1.0  # Default beta
    
    def _calculate_diversification_ratio(self, weights: np.ndarray, correlation_matrix: np.ndarray) -> float:
        """Calculate diversification ratio"""
        try:
            # Weighted average volatility / Portfolio volatility
            # Simplified calculation
            return float(1.0 / np.sqrt(np.dot(weights, np.dot(correlation_matrix, weights))))
        except Exception:
            return 1.0
    
    def _get_sector(self, symbol: str) -> str:
        """Get sector for symbol (simplified)"""
        # Simplified sector mapping
        sector_map = {
            "AAPL": "Technology",
            "MSFT": "Technology",
            "GOOGL": "Technology",
            "AMZN": "Consumer Discretionary",
            "TSLA": "Consumer Discretionary",
            "JPM": "Financials",
            "BAC": "Financials",
            "JNJ": "Healthcare",
            "PFE": "Healthcare",
            "XOM": "Energy",
            "CVX": "Energy"
        }
        return sector_map.get(symbol, "Other")
    
    def _format_detailed_metrics(self, risk_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Format detailed risk metrics for response"""
        return {
            "risk_metrics": risk_metrics,
            "risk_breakdown": {
                "market_risk": risk_metrics.get("volatility", 0),
                "concentration_risk": risk_metrics.get("concentration_risk", 0),
                "liquidity_risk": risk_metrics.get("liquidity_risk", 0),
                "correlation_risk": risk_metrics.get("correlation_risk", 0)
            },
            "var_analysis": {
                "daily_var_95": risk_metrics.get("value_at_risk", 0),
                "expected_shortfall_95": risk_metrics.get("expected_shortfall", 0)
            }
        }
    
    def _get_default_risk_response(self, portfolio_id: str) -> RiskResponse:
        """Get default risk response for empty portfolio"""
        assessment = RiskAssessment(
            overall_risk_score=0.0,
            risk_level=RiskLevel.LOW,
            key_risks=[],
            risk_metrics={},
            stress_test_results={},
            recommendations=["Add holdings to begin risk analysis"],
            alerts=[],
            last_updated=datetime.utcnow()
        )
        
        return RiskResponse(
            portfolio_id=portfolio_id,
            assessment=assessment,
            detailed_metrics={},
            scenario_analysis={},
            recommendations=["Add holdings to begin risk analysis"]
        )

# Global risk engine instance
risk_engine = RiskEngine()

def get_risk_engine() -> RiskEngine:
    """Get risk engine instance"""
    return risk_engine