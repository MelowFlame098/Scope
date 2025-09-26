# Scenario Analyzer
# Phase 9: AI-First Platform Implementation

import logging
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ScenarioType(Enum):
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    RECESSION = "recession"
    RECOVERY = "recovery"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    INTEREST_RATE_RISE = "interest_rate_rise"
    INTEREST_RATE_CUT = "interest_rate_cut"
    INFLATION_SPIKE = "inflation_spike"
    DEFLATION = "deflation"
    GEOPOLITICAL_CRISIS = "geopolitical_crisis"
    TECH_BUBBLE = "tech_bubble"
    ENERGY_CRISIS = "energy_crisis"
    CURRENCY_CRISIS = "currency_crisis"
    PANDEMIC = "pandemic"

class ScenarioProbability(Enum):
    VERY_LOW = "very_low"      # 0-10%
    LOW = "low"                # 10-25%
    MODERATE = "moderate"      # 25-50%
    HIGH = "high"              # 50-75%
    VERY_HIGH = "very_high"    # 75-100%

class ImpactSeverity(Enum):
    MINIMAL = "minimal"        # 0-5% impact
    LOW = "low"                # 5-15% impact
    MODERATE = "moderate"      # 15-30% impact
    HIGH = "high"              # 30-50% impact
    SEVERE = "severe"          # 50%+ impact

@dataclass
class EconomicIndicator:
    name: str
    current_value: float
    historical_mean: float
    historical_std: float
    trend: str  # 'rising', 'falling', 'stable'
    importance_weight: float  # 0-1
    last_updated: datetime

@dataclass
class ScenarioImpact:
    asset_class: str
    expected_return: float
    volatility_change: float
    correlation_change: float
    liquidity_impact: float
    time_horizon: int  # days
    confidence_interval: Tuple[float, float]

@dataclass
class ScenarioAnalysis:
    scenario_type: ScenarioType
    probability: ScenarioProbability
    probability_score: float  # 0-1
    time_horizon: int  # days
    triggers: List[str]
    impacts: Dict[str, ScenarioImpact]
    portfolio_impact: Dict[str, float]
    risk_metrics: Dict[str, float]
    hedging_strategies: List[str]
    description: str
    confidence: float
    analysis_timestamp: datetime

@dataclass
class StressTestResult:
    scenario_name: str
    portfolio_value_change: float
    max_drawdown: float
    var_95: float
    var_99: float
    expected_shortfall: float
    recovery_time: int  # days
    worst_performing_assets: List[Tuple[str, float]]
    best_performing_assets: List[Tuple[str, float]]
    stress_timestamp: datetime

class ScenarioAnalyzer:
    """Advanced scenario analysis and stress testing engine"""
    
    def __init__(self):
        self.economic_indicators = {}
        self.scenario_models = {}
        self.correlation_matrices = {}
        
        # Initialize economic indicators
        self._initialize_economic_indicators()
        
        # Scenario probability thresholds
        self.probability_thresholds = {
            ScenarioProbability.VERY_LOW: (0.0, 0.1),
            ScenarioProbability.LOW: (0.1, 0.25),
            ScenarioProbability.MODERATE: (0.25, 0.5),
            ScenarioProbability.HIGH: (0.5, 0.75),
            ScenarioProbability.VERY_HIGH: (0.75, 1.0)
        }
        
        logger.info("Scenario analyzer initialized")
    
    async def analyze_scenarios(self, portfolio: Dict[str, float], 
                               time_horizon: int = 252) -> List[ScenarioAnalysis]:
        """Analyze multiple market scenarios and their impacts"""
        try:
            scenarios = []
            
            # Update economic indicators
            await self._update_economic_indicators()
            
            # Analyze each scenario type
            for scenario_type in ScenarioType:
                try:
                    analysis = await self._analyze_single_scenario(
                        scenario_type, portfolio, time_horizon
                    )
                    if analysis:
                        scenarios.append(analysis)
                except Exception as e:
                    logger.error(f"Error analyzing scenario {scenario_type}: {e}")
                    continue
            
            # Sort by probability score (descending)
            scenarios.sort(key=lambda x: x.probability_score, reverse=True)
            
            return scenarios[:10]  # Return top 10 most likely scenarios
            
        except Exception as e:
            logger.error(f"Error analyzing scenarios: {e}")
            return []
    
    async def stress_test_portfolio(self, portfolio: Dict[str, float],
                                   scenarios: Optional[List[str]] = None) -> List[StressTestResult]:
        """Perform comprehensive stress testing on portfolio"""
        try:
            results = []
            
            # Default stress test scenarios
            if scenarios is None:
                scenarios = [
                    "2008_financial_crisis",
                    "2020_covid_crash",
                    "2000_dot_com_bubble",
                    "1987_black_monday",
                    "2018_volatility_spike",
                    "interest_rate_shock",
                    "inflation_shock",
                    "geopolitical_crisis",
                    "liquidity_crisis",
                    "currency_devaluation"
                ]
            
            for scenario_name in scenarios:
                try:
                    result = await self._run_stress_test(portfolio, scenario_name)
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error in stress test {scenario_name}: {e}")
                    continue
            
            # Sort by severity (worst first)
            results.sort(key=lambda x: x.portfolio_value_change)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in stress testing: {e}")
            return []
    
    async def monte_carlo_simulation(self, portfolio: Dict[str, float],
                                   num_simulations: int = 1000,
                                   time_horizon: int = 252) -> Dict[str, Any]:
        """Run Monte Carlo simulation for portfolio scenarios"""
        try:
            # Get historical data for portfolio assets
            asset_data = await self._get_portfolio_data(portfolio, time_horizon * 2)
            
            if not asset_data:
                return {}
            
            # Calculate returns and correlations
            returns = asset_data.pct_change().dropna()
            mean_returns = returns.mean()
            cov_matrix = returns.cov()
            
            # Portfolio weights
            weights = np.array([portfolio.get(asset, 0) for asset in returns.columns])
            weights = weights / weights.sum()  # Normalize
            
            # Run simulations
            portfolio_returns = []
            
            for _ in range(num_simulations):
                # Generate random returns
                random_returns = np.random.multivariate_normal(
                    mean_returns * time_horizon,
                    cov_matrix * time_horizon
                )
                
                # Calculate portfolio return
                portfolio_return = np.dot(weights, random_returns)
                portfolio_returns.append(portfolio_return)
            
            portfolio_returns = np.array(portfolio_returns)
            
            # Calculate statistics
            results = {
                'mean_return': float(np.mean(portfolio_returns)),
                'std_return': float(np.std(portfolio_returns)),
                'var_95': float(np.percentile(portfolio_returns, 5)),
                'var_99': float(np.percentile(portfolio_returns, 1)),
                'var_99_9': float(np.percentile(portfolio_returns, 0.1)),
                'expected_shortfall_95': float(np.mean(portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)])),
                'expected_shortfall_99': float(np.mean(portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 1)])),
                'max_loss': float(np.min(portfolio_returns)),
                'max_gain': float(np.max(portfolio_returns)),
                'probability_of_loss': float(np.sum(portfolio_returns < 0) / num_simulations),
                'probability_of_large_loss': float(np.sum(portfolio_returns < -0.1) / num_simulations),
                'sharpe_ratio': float(np.mean(portfolio_returns) / np.std(portfolio_returns)) if np.std(portfolio_returns) > 0 else 0,
                'skewness': float(stats.skew(portfolio_returns)),
                'kurtosis': float(stats.kurtosis(portfolio_returns)),
                'simulation_timestamp': datetime.now()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo simulation: {e}")
            return {}
    
    async def scenario_optimization(self, portfolio: Dict[str, float],
                                  target_scenarios: List[ScenarioType]) -> Dict[str, Any]:
        """Optimize portfolio for specific scenarios"""
        try:
            optimization_results = {}
            
            for scenario in target_scenarios:
                try:
                    # Analyze scenario
                    scenario_analysis = await self._analyze_single_scenario(
                        scenario, portfolio, 252
                    )
                    
                    if not scenario_analysis:
                        continue
                    
                    # Calculate optimal weights for this scenario
                    optimal_weights = await self._optimize_for_scenario(
                        portfolio, scenario_analysis
                    )
                    
                    optimization_results[scenario.value] = {
                        'optimal_weights': optimal_weights,
                        'expected_return': scenario_analysis.portfolio_impact.get('expected_return', 0),
                        'risk_reduction': scenario_analysis.portfolio_impact.get('risk_reduction', 0),
                        'hedging_strategies': scenario_analysis.hedging_strategies
                    }
                    
                except Exception as e:
                    logger.error(f"Error optimizing for scenario {scenario}: {e}")
                    continue
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Error in scenario optimization: {e}")
            return {}
    
    def _initialize_economic_indicators(self):
        """Initialize economic indicators with default values"""
        try:
            self.economic_indicators = {
                'gdp_growth': EconomicIndicator(
                    name='GDP Growth Rate',
                    current_value=2.5,
                    historical_mean=2.8,
                    historical_std=1.2,
                    trend='stable',
                    importance_weight=0.9,
                    last_updated=datetime.now()
                ),
                'inflation_rate': EconomicIndicator(
                    name='Inflation Rate',
                    current_value=3.2,
                    historical_mean=2.1,
                    historical_std=1.5,
                    trend='rising',
                    importance_weight=0.8,
                    last_updated=datetime.now()
                ),
                'unemployment_rate': EconomicIndicator(
                    name='Unemployment Rate',
                    current_value=4.1,
                    historical_mean=5.5,
                    historical_std=2.1,
                    trend='falling',
                    importance_weight=0.7,
                    last_updated=datetime.now()
                ),
                'interest_rate': EconomicIndicator(
                    name='Federal Funds Rate',
                    current_value=5.25,
                    historical_mean=3.8,
                    historical_std=2.5,
                    trend='stable',
                    importance_weight=0.9,
                    last_updated=datetime.now()
                ),
                'vix': EconomicIndicator(
                    name='VIX Volatility Index',
                    current_value=18.5,
                    historical_mean=19.2,
                    historical_std=8.4,
                    trend='stable',
                    importance_weight=0.6,
                    last_updated=datetime.now()
                ),
                'yield_curve_spread': EconomicIndicator(
                    name='10Y-2Y Yield Spread',
                    current_value=0.8,
                    historical_mean=1.2,
                    historical_std=1.1,
                    trend='falling',
                    importance_weight=0.8,
                    last_updated=datetime.now()
                )
            }
            
        except Exception as e:
            logger.error(f"Error initializing economic indicators: {e}")
    
    async def _update_economic_indicators(self):
        """Update economic indicators with latest data"""
        try:
            # In a real implementation, this would fetch live data
            # For now, we'll simulate some updates
            
            for indicator_name, indicator in self.economic_indicators.items():
                # Simulate small random changes
                change = np.random.normal(0, indicator.historical_std * 0.1)
                indicator.current_value += change
                indicator.last_updated = datetime.now()
                
                # Update trend based on recent changes
                if change > indicator.historical_std * 0.05:
                    indicator.trend = 'rising'
                elif change < -indicator.historical_std * 0.05:
                    indicator.trend = 'falling'
                else:
                    indicator.trend = 'stable'
            
        except Exception as e:
            logger.error(f"Error updating economic indicators: {e}")
    
    async def _analyze_single_scenario(self, scenario_type: ScenarioType,
                                     portfolio: Dict[str, float],
                                     time_horizon: int) -> Optional[ScenarioAnalysis]:
        """Analyze a single scenario"""
        try:
            # Calculate scenario probability
            probability_score = await self._calculate_scenario_probability(scenario_type)
            probability_enum = self._score_to_probability_enum(probability_score)
            
            # Identify triggers
            triggers = await self._identify_scenario_triggers(scenario_type)
            
            # Calculate impacts
            impacts = await self._calculate_scenario_impacts(scenario_type, time_horizon)
            
            # Calculate portfolio impact
            portfolio_impact = await self._calculate_portfolio_impact(
                portfolio, impacts, scenario_type
            )
            
            # Calculate risk metrics
            risk_metrics = await self._calculate_scenario_risk_metrics(
                portfolio, scenario_type
            )
            
            # Generate hedging strategies
            hedging_strategies = await self._generate_hedging_strategies(
                scenario_type, portfolio_impact
            )
            
            # Generate description
            description = self._generate_scenario_description(scenario_type, triggers)
            
            # Calculate confidence
            confidence = min(1.0, probability_score + 0.3)
            
            return ScenarioAnalysis(
                scenario_type=scenario_type,
                probability=probability_enum,
                probability_score=probability_score,
                time_horizon=time_horizon,
                triggers=triggers,
                impacts=impacts,
                portfolio_impact=portfolio_impact,
                risk_metrics=risk_metrics,
                hedging_strategies=hedging_strategies,
                description=description,
                confidence=confidence,
                analysis_timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error analyzing scenario {scenario_type}: {e}")
            return None
    
    async def _calculate_scenario_probability(self, scenario_type: ScenarioType) -> float:
        """Calculate probability of scenario occurring"""
        try:
            # Base probabilities for different scenarios
            base_probabilities = {
                ScenarioType.BULL_MARKET: 0.3,
                ScenarioType.BEAR_MARKET: 0.2,
                ScenarioType.RECESSION: 0.15,
                ScenarioType.RECOVERY: 0.25,
                ScenarioType.HIGH_VOLATILITY: 0.4,
                ScenarioType.LOW_VOLATILITY: 0.3,
                ScenarioType.INTEREST_RATE_RISE: 0.6,
                ScenarioType.INTEREST_RATE_CUT: 0.2,
                ScenarioType.INFLATION_SPIKE: 0.3,
                ScenarioType.DEFLATION: 0.1,
                ScenarioType.GEOPOLITICAL_CRISIS: 0.25,
                ScenarioType.TECH_BUBBLE: 0.15,
                ScenarioType.ENERGY_CRISIS: 0.2,
                ScenarioType.CURRENCY_CRISIS: 0.1,
                ScenarioType.PANDEMIC: 0.05
            }
            
            base_prob = base_probabilities.get(scenario_type, 0.2)
            
            # Adjust based on current economic indicators
            adjustments = 0.0
            
            if scenario_type == ScenarioType.RECESSION:
                # Higher probability if yield curve is inverted
                if self.economic_indicators['yield_curve_spread'].current_value < 0:
                    adjustments += 0.3
                # Higher probability if unemployment is rising
                if self.economic_indicators['unemployment_rate'].trend == 'rising':
                    adjustments += 0.2
            
            elif scenario_type == ScenarioType.INFLATION_SPIKE:
                # Higher probability if inflation is already elevated
                current_inflation = self.economic_indicators['inflation_rate'].current_value
                if current_inflation > 4.0:
                    adjustments += 0.2
                if self.economic_indicators['inflation_rate'].trend == 'rising':
                    adjustments += 0.15
            
            elif scenario_type == ScenarioType.HIGH_VOLATILITY:
                # Higher probability if VIX is elevated
                current_vix = self.economic_indicators['vix'].current_value
                if current_vix > 25:
                    adjustments += 0.2
            
            elif scenario_type == ScenarioType.INTEREST_RATE_RISE:
                # Higher probability if inflation is high
                current_inflation = self.economic_indicators['inflation_rate'].current_value
                if current_inflation > 3.0:
                    adjustments += 0.1
            
            # Ensure probability stays within bounds
            final_probability = max(0.0, min(1.0, base_prob + adjustments))
            
            return final_probability
            
        except Exception as e:
            logger.error(f"Error calculating scenario probability: {e}")
            return 0.2
    
    def _score_to_probability_enum(self, score: float) -> ScenarioProbability:
        """Convert probability score to enum"""
        for prob_enum, (min_val, max_val) in self.probability_thresholds.items():
            if min_val <= score < max_val:
                return prob_enum
        return ScenarioProbability.MODERATE
    
    async def _identify_scenario_triggers(self, scenario_type: ScenarioType) -> List[str]:
        """Identify potential triggers for scenario"""
        try:
            triggers = {
                ScenarioType.RECESSION: [
                    "Inverted yield curve",
                    "Rising unemployment",
                    "Declining consumer confidence",
                    "Tightening monetary policy"
                ],
                ScenarioType.INFLATION_SPIKE: [
                    "Supply chain disruptions",
                    "Energy price surge",
                    "Excessive monetary stimulus",
                    "Wage-price spiral"
                ],
                ScenarioType.GEOPOLITICAL_CRISIS: [
                    "Military conflicts",
                    "Trade wars",
                    "Sanctions",
                    "Political instability"
                ],
                ScenarioType.HIGH_VOLATILITY: [
                    "Market uncertainty",
                    "Economic data surprises",
                    "Central bank policy changes",
                    "Geopolitical tensions"
                ],
                ScenarioType.TECH_BUBBLE: [
                    "Excessive valuations",
                    "Speculative trading",
                    "Easy monetary policy",
                    "FOMO investing"
                ]
            }
            
            return triggers.get(scenario_type, ["Economic uncertainty", "Market volatility"])
            
        except Exception as e:
            logger.error(f"Error identifying triggers: {e}")
            return []
    
    async def _calculate_scenario_impacts(self, scenario_type: ScenarioType,
                                        time_horizon: int) -> Dict[str, ScenarioImpact]:
        """Calculate impacts on different asset classes"""
        try:
            # Define impact templates for different scenarios
            impact_templates = {
                ScenarioType.RECESSION: {
                    'stocks': ScenarioImpact('stocks', -0.25, 0.5, 0.8, -0.3, time_horizon, (-0.4, -0.1)),
                    'bonds': ScenarioImpact('bonds', 0.15, -0.2, -0.3, 0.2, time_horizon, (0.05, 0.25)),
                    'commodities': ScenarioImpact('commodities', -0.15, 0.3, 0.4, -0.2, time_horizon, (-0.3, 0.0)),
                    'real_estate': ScenarioImpact('real_estate', -0.2, 0.2, 0.3, -0.4, time_horizon, (-0.35, -0.05)),
                    'cash': ScenarioImpact('cash', 0.02, 0.0, 0.0, 0.5, time_horizon, (0.01, 0.03))
                },
                ScenarioType.INFLATION_SPIKE: {
                    'stocks': ScenarioImpact('stocks', -0.1, 0.3, 0.2, -0.1, time_horizon, (-0.2, 0.0)),
                    'bonds': ScenarioImpact('bonds', -0.2, 0.4, 0.1, -0.2, time_horizon, (-0.3, -0.1)),
                    'commodities': ScenarioImpact('commodities', 0.25, 0.2, -0.1, 0.1, time_horizon, (0.1, 0.4)),
                    'real_estate': ScenarioImpact('real_estate', 0.1, 0.1, 0.0, -0.1, time_horizon, (0.0, 0.2)),
                    'cash': ScenarioImpact('cash', -0.05, 0.0, 0.0, 0.2, time_horizon, (-0.08, -0.02))
                },
                ScenarioType.HIGH_VOLATILITY: {
                    'stocks': ScenarioImpact('stocks', -0.05, 0.8, 0.3, -0.2, time_horizon, (-0.15, 0.05)),
                    'bonds': ScenarioImpact('bonds', 0.05, 0.3, -0.2, 0.1, time_horizon, (-0.02, 0.12)),
                    'commodities': ScenarioImpact('commodities', 0.0, 0.5, 0.2, -0.1, time_horizon, (-0.1, 0.1)),
                    'real_estate': ScenarioImpact('real_estate', -0.02, 0.2, 0.1, -0.1, time_horizon, (-0.08, 0.04)),
                    'cash': ScenarioImpact('cash', 0.01, 0.0, 0.0, 0.3, time_horizon, (0.005, 0.015))
                }
            }
            
            # Get template or create default
            if scenario_type in impact_templates:
                return impact_templates[scenario_type]
            else:
                # Create default impacts
                return {
                    'stocks': ScenarioImpact('stocks', 0.0, 0.2, 0.1, 0.0, time_horizon, (-0.1, 0.1)),
                    'bonds': ScenarioImpact('bonds', 0.0, 0.1, 0.0, 0.0, time_horizon, (-0.05, 0.05)),
                    'commodities': ScenarioImpact('commodities', 0.0, 0.3, 0.1, 0.0, time_horizon, (-0.15, 0.15)),
                    'real_estate': ScenarioImpact('real_estate', 0.0, 0.1, 0.0, 0.0, time_horizon, (-0.05, 0.05)),
                    'cash': ScenarioImpact('cash', 0.01, 0.0, 0.0, 0.1, time_horizon, (0.005, 0.015))
                }
            
        except Exception as e:
            logger.error(f"Error calculating scenario impacts: {e}")
            return {}
    
    async def _calculate_portfolio_impact(self, portfolio: Dict[str, float],
                                        impacts: Dict[str, ScenarioImpact],
                                        scenario_type: ScenarioType) -> Dict[str, float]:
        """Calculate overall portfolio impact"""
        try:
            # Map portfolio assets to asset classes
            asset_class_mapping = {
                'AAPL': 'stocks', 'GOOGL': 'stocks', 'MSFT': 'stocks', 'TSLA': 'stocks',
                'SPY': 'stocks', 'QQQ': 'stocks', 'IWM': 'stocks',
                'TLT': 'bonds', 'IEF': 'bonds', 'SHY': 'bonds',
                'GLD': 'commodities', 'SLV': 'commodities', 'USO': 'commodities',
                'VNQ': 'real_estate', 'IYR': 'real_estate',
                'SHV': 'cash', 'BIL': 'cash'
            }
            
            total_return = 0.0
            total_volatility_change = 0.0
            total_weight = 0.0
            
            for asset, weight in portfolio.items():
                asset_class = asset_class_mapping.get(asset, 'stocks')  # Default to stocks
                
                if asset_class in impacts:
                    impact = impacts[asset_class]
                    total_return += weight * impact.expected_return
                    total_volatility_change += weight * impact.volatility_change
                    total_weight += weight
            
            # Normalize if weights don't sum to 1
            if total_weight > 0:
                total_return /= total_weight
                total_volatility_change /= total_weight
            
            # Calculate additional metrics
            max_drawdown = min(-0.05, total_return * 1.5)  # Estimate max drawdown
            recovery_time = max(30, int(abs(total_return) * 365))  # Estimate recovery time
            
            return {
                'expected_return': total_return,
                'volatility_change': total_volatility_change,
                'max_drawdown': max_drawdown,
                'recovery_time_days': recovery_time,
                'risk_reduction': max(0, -total_return * 0.5)  # Potential risk reduction
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio impact: {e}")
            return {'expected_return': 0.0, 'volatility_change': 0.0}
    
    async def _calculate_scenario_risk_metrics(self, portfolio: Dict[str, float],
                                             scenario_type: ScenarioType) -> Dict[str, float]:
        """Calculate risk metrics for scenario"""
        try:
            # Simulate portfolio data for risk calculations
            portfolio_data = await self._simulate_portfolio_data(portfolio, scenario_type)
            
            if portfolio_data is None or len(portfolio_data) < 30:
                return {'var_95': -0.05, 'var_99': -0.1, 'expected_shortfall': -0.12}
            
            returns = portfolio_data.pct_change().dropna()
            
            # Calculate VaR and Expected Shortfall
            var_95 = float(np.percentile(returns, 5))
            var_99 = float(np.percentile(returns, 1))
            
            # Expected Shortfall (Conditional VaR)
            es_95 = float(np.mean(returns[returns <= var_95]))
            es_99 = float(np.mean(returns[returns <= var_99]))
            
            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = float(drawdown.min())
            
            return {
                'var_95': var_95,
                'var_99': var_99,
                'expected_shortfall_95': es_95,
                'expected_shortfall_99': es_99,
                'max_drawdown': max_drawdown,
                'volatility': float(returns.std()),
                'skewness': float(stats.skew(returns)),
                'kurtosis': float(stats.kurtosis(returns))
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {'var_95': -0.05, 'var_99': -0.1, 'expected_shortfall': -0.12}
    
    async def _generate_hedging_strategies(self, scenario_type: ScenarioType,
                                         portfolio_impact: Dict[str, float]) -> List[str]:
        """Generate hedging strategies for scenario"""
        try:
            strategies = {
                ScenarioType.RECESSION: [
                    "Increase allocation to government bonds",
                    "Add defensive stocks (utilities, consumer staples)",
                    "Consider gold allocation",
                    "Reduce leverage and increase cash position",
                    "Use put options for downside protection"
                ],
                ScenarioType.INFLATION_SPIKE: [
                    "Increase commodity exposure",
                    "Add inflation-protected bonds (TIPS)",
                    "Consider real estate investment",
                    "Reduce duration in bond portfolio",
                    "Add energy and materials stocks"
                ],
                ScenarioType.HIGH_VOLATILITY: [
                    "Implement volatility targeting strategy",
                    "Use options strategies for protection",
                    "Increase diversification across asset classes",
                    "Consider volatility ETFs as hedge",
                    "Reduce position sizes"
                ],
                ScenarioType.GEOPOLITICAL_CRISIS: [
                    "Increase safe haven assets (gold, treasuries)",
                    "Reduce exposure to affected regions",
                    "Add currency hedging",
                    "Consider defense and energy stocks",
                    "Maintain higher cash reserves"
                ]
            }
            
            base_strategies = strategies.get(scenario_type, [
                "Diversify across asset classes",
                "Monitor risk metrics closely",
                "Consider defensive positioning"
            ])
            
            # Add specific strategies based on portfolio impact
            expected_return = portfolio_impact.get('expected_return', 0)
            if expected_return < -0.1:
                base_strategies.append("Implement stop-loss orders")
                base_strategies.append("Consider inverse ETFs")
            
            return base_strategies[:5]  # Return top 5 strategies
            
        except Exception as e:
            logger.error(f"Error generating hedging strategies: {e}")
            return ["Maintain diversified portfolio", "Monitor market conditions"]
    
    def _generate_scenario_description(self, scenario_type: ScenarioType,
                                     triggers: List[str]) -> str:
        """Generate human-readable scenario description"""
        try:
            descriptions = {
                ScenarioType.RECESSION: "Economic downturn characterized by declining GDP, rising unemployment, and reduced consumer spending.",
                ScenarioType.INFLATION_SPIKE: "Rapid increase in price levels across the economy, eroding purchasing power and affecting asset valuations.",
                ScenarioType.HIGH_VOLATILITY: "Period of increased market uncertainty and price swings across asset classes.",
                ScenarioType.GEOPOLITICAL_CRISIS: "International tensions or conflicts affecting global markets and trade relationships.",
                ScenarioType.TECH_BUBBLE: "Overvaluation in technology sector driven by speculation and excessive optimism."
            }
            
            base_description = descriptions.get(scenario_type, "Market scenario with potential impacts on portfolio performance.")
            
            if triggers:
                trigger_text = ", ".join(triggers[:3])  # First 3 triggers
                return f"{base_description} Key triggers include: {trigger_text}."
            
            return base_description
            
        except Exception as e:
            logger.error(f"Error generating description: {e}")
            return "Market scenario requiring monitoring and potential portfolio adjustments."
    
    async def _simulate_portfolio_data(self, portfolio: Dict[str, float],
                                     scenario_type: ScenarioType,
                                     periods: int = 252) -> Optional[pd.Series]:
        """Simulate portfolio performance data for scenario"""
        try:
            # Generate synthetic portfolio returns based on scenario
            np.random.seed(hash(str(scenario_type)) % 2**32)
            
            # Base parameters
            if scenario_type == ScenarioType.RECESSION:
                mean_return = -0.0008  # -20% annualized
                volatility = 0.025     # 40% annualized
            elif scenario_type == ScenarioType.HIGH_VOLATILITY:
                mean_return = 0.0002   # 5% annualized
                volatility = 0.035     # 55% annualized
            elif scenario_type == ScenarioType.INFLATION_SPIKE:
                mean_return = -0.0004  # -10% annualized
                volatility = 0.02      # 32% annualized
            else:
                mean_return = 0.0004   # 10% annualized
                volatility = 0.015     # 24% annualized
            
            # Generate returns
            returns = np.random.normal(mean_return, volatility, periods)
            
            # Add some autocorrelation for realism
            for i in range(1, len(returns)):
                returns[i] += 0.1 * returns[i-1]
            
            # Convert to price series
            dates = pd.date_range(end=datetime.now(), periods=periods, freq='D')
            prices = pd.Series(100 * np.exp(np.cumsum(returns)), index=dates)
            
            return prices
            
        except Exception as e:
            logger.error(f"Error simulating portfolio data: {e}")
            return None
    
    async def _get_portfolio_data(self, portfolio: Dict[str, float],
                                periods: int) -> Optional[pd.DataFrame]:
        """Get historical data for portfolio assets"""
        try:
            # Generate synthetic data for portfolio assets
            dates = pd.date_range(end=datetime.now(), periods=periods, freq='D')
            data = {}
            
            for asset, weight in portfolio.items():
                # Generate realistic price data
                np.random.seed(hash(asset) % 2**32)
                
                # Different assets have different characteristics
                if asset in ['TLT', 'IEF', 'SHY']:  # Bonds
                    mean_return = 0.0002
                    volatility = 0.008
                elif asset in ['GLD', 'SLV']:  # Commodities
                    mean_return = 0.0003
                    volatility = 0.02
                else:  # Stocks
                    mean_return = 0.0005
                    volatility = 0.018
                
                returns = np.random.normal(mean_return, volatility, periods)
                prices = 100 * np.exp(np.cumsum(returns))
                data[asset] = prices
            
            return pd.DataFrame(data, index=dates)
            
        except Exception as e:
            logger.error(f"Error getting portfolio data: {e}")
            return None
    
    async def _run_stress_test(self, portfolio: Dict[str, float],
                             scenario_name: str) -> Optional[StressTestResult]:
        """Run individual stress test"""
        try:
            # Define stress test parameters
            stress_scenarios = {
                "2008_financial_crisis": {
                    'stocks': -0.37, 'bonds': 0.05, 'commodities': -0.25, 'real_estate': -0.31, 'cash': 0.01
                },
                "2020_covid_crash": {
                    'stocks': -0.34, 'bonds': 0.08, 'commodities': -0.21, 'real_estate': -0.15, 'cash': 0.01
                },
                "2000_dot_com_bubble": {
                    'stocks': -0.49, 'bonds': 0.12, 'commodities': -0.10, 'real_estate': 0.05, 'cash': 0.02
                },
                "interest_rate_shock": {
                    'stocks': -0.15, 'bonds': -0.20, 'commodities': 0.05, 'real_estate': -0.10, 'cash': 0.03
                },
                "inflation_shock": {
                    'stocks': -0.10, 'bonds': -0.15, 'commodities': 0.25, 'real_estate': 0.10, 'cash': -0.05
                }
            }
            
            if scenario_name not in stress_scenarios:
                return None
            
            scenario_returns = stress_scenarios[scenario_name]
            
            # Map portfolio to asset classes and calculate impact
            asset_class_mapping = {
                'AAPL': 'stocks', 'GOOGL': 'stocks', 'MSFT': 'stocks', 'TSLA': 'stocks',
                'SPY': 'stocks', 'QQQ': 'stocks', 'IWM': 'stocks',
                'TLT': 'bonds', 'IEF': 'bonds', 'SHY': 'bonds',
                'GLD': 'commodities', 'SLV': 'commodities', 'USO': 'commodities',
                'VNQ': 'real_estate', 'IYR': 'real_estate',
                'SHV': 'cash', 'BIL': 'cash'
            }
            
            portfolio_return = 0.0
            asset_returns = []
            
            for asset, weight in portfolio.items():
                asset_class = asset_class_mapping.get(asset, 'stocks')
                asset_return = scenario_returns.get(asset_class, -0.1)
                portfolio_return += weight * asset_return
                asset_returns.append((asset, asset_return))
            
            # Sort assets by performance
            asset_returns.sort(key=lambda x: x[1])
            worst_performing = asset_returns[:3]
            best_performing = asset_returns[-3:]
            
            # Calculate risk metrics
            var_95 = portfolio_return * 1.2  # Estimate
            var_99 = portfolio_return * 1.5
            expected_shortfall = portfolio_return * 1.8
            max_drawdown = min(portfolio_return * 1.3, -0.05)
            recovery_time = max(30, int(abs(portfolio_return) * 500))
            
            return StressTestResult(
                scenario_name=scenario_name,
                portfolio_value_change=portfolio_return,
                max_drawdown=max_drawdown,
                var_95=var_95,
                var_99=var_99,
                expected_shortfall=expected_shortfall,
                recovery_time=recovery_time,
                worst_performing_assets=worst_performing,
                best_performing_assets=best_performing,
                stress_timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error running stress test {scenario_name}: {e}")
            return None
    
    async def _optimize_for_scenario(self, portfolio: Dict[str, float],
                                   scenario_analysis: ScenarioAnalysis) -> Dict[str, float]:
        """Optimize portfolio weights for specific scenario"""
        try:
            # Simple optimization based on scenario impacts
            current_weights = portfolio.copy()
            optimized_weights = {}
            
            # Asset class preferences for different scenarios
            scenario_preferences = {
                ScenarioType.RECESSION: {
                    'bonds': 1.2, 'cash': 1.1, 'stocks': 0.7, 'commodities': 0.8, 'real_estate': 0.6
                },
                ScenarioType.INFLATION_SPIKE: {
                    'commodities': 1.3, 'real_estate': 1.1, 'stocks': 0.9, 'bonds': 0.6, 'cash': 0.5
                },
                ScenarioType.HIGH_VOLATILITY: {
                    'bonds': 1.1, 'cash': 1.2, 'stocks': 0.8, 'commodities': 0.9, 'real_estate': 0.9
                }
            }
            
            preferences = scenario_preferences.get(scenario_analysis.scenario_type, {})
            
            # Apply preferences to current weights
            total_adjusted_weight = 0.0
            for asset, weight in current_weights.items():
                # Map asset to asset class (simplified)
                if asset in ['TLT', 'IEF', 'SHY']:
                    asset_class = 'bonds'
                elif asset in ['GLD', 'SLV']:
                    asset_class = 'commodities'
                elif asset in ['SHV', 'BIL']:
                    asset_class = 'cash'
                elif asset in ['VNQ', 'IYR']:
                    asset_class = 'real_estate'
                else:
                    asset_class = 'stocks'
                
                preference = preferences.get(asset_class, 1.0)
                adjusted_weight = weight * preference
                optimized_weights[asset] = adjusted_weight
                total_adjusted_weight += adjusted_weight
            
            # Normalize weights
            if total_adjusted_weight > 0:
                for asset in optimized_weights:
                    optimized_weights[asset] /= total_adjusted_weight
            
            return optimized_weights
            
        except Exception as e:
            logger.error(f"Error optimizing for scenario: {e}")
            return portfolio

# Export main class
__all__ = ['ScenarioAnalyzer', 'ScenarioAnalysis', 'ScenarioType', 'ScenarioProbability',
           'StressTestResult', 'ScenarioImpact', 'EconomicIndicator']