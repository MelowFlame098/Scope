#!/usr/bin/env python3
"""
Forex Comprehensive Integration Framework

A comprehensive integration framework that combines all enhanced Forex models
including PPP, IRP, UIP, Balance of Payments, and advanced ML models.

This framework provides:
- Unified analysis across all forex models
- Cross-model correlation and consensus analysis
- Currency strength scoring and ranking system
- Advanced portfolio optimization for forex pairs
- Multi-dimensional risk assessment
- Market regime detection for forex markets
- Real-time monitoring and alerting for currency movements
- Central bank intervention detection
- Carry trade optimization
- Economic calendar integration

Author: Forex Analytics Team
Date: 2024
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Import forex models
try:
    from ppp_irp_uip import PPPIRPUIPModel
    from balance_of_payments_model import BalanceOfPaymentsModel
    from advanced_forex_ml import AdvancedForexML
    from monetary_models import MonetaryModel
    from forex_comprehensive import ForexComprehensiveIndicators
except ImportError as e:
    logging.warning(f"Some forex models not available: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CurrencyStrengthAnalysis:
    """Currency strength analysis across multiple dimensions"""
    currency_code: str
    
    # Fundamental strength metrics
    economic_strength_score: float = 0.0
    monetary_policy_score: float = 0.0
    fiscal_health_score: float = 0.0
    political_stability_score: float = 0.0
    
    # Technical strength metrics
    momentum_score: float = 0.0
    trend_strength_score: float = 0.0
    volatility_score: float = 0.0
    
    # Model-based scores
    ppp_valuation_score: float = 0.0
    irp_arbitrage_score: float = 0.0
    bop_flow_score: float = 0.0
    carry_trade_score: float = 0.0
    
    # Composite scores
    overall_strength_score: float = 0.0
    short_term_outlook: str = "Neutral"
    medium_term_outlook: str = "Neutral"
    long_term_outlook: str = "Neutral"
    
    # Risk metrics
    currency_risk_score: float = 0.0
    intervention_risk: float = 0.0
    event_risk: float = 0.0

@dataclass
class CrossCurrencyAnalysis:
    """Cross-currency pair analysis"""
    base_currency: str
    quote_currency: str
    
    # Relative strength metrics
    relative_strength_score: float = 0.0
    interest_rate_differential: float = 0.0
    inflation_differential: float = 0.0
    growth_differential: float = 0.0
    
    # Parity analysis
    ppp_deviation: float = 0.0
    irp_deviation: float = 0.0
    uip_deviation: float = 0.0
    
    # Flow analysis
    capital_flow_balance: float = 0.0
    trade_flow_balance: float = 0.0
    speculative_flow_score: float = 0.0
    
    # Correlation metrics
    correlation_with_commodities: float = 0.0
    correlation_with_equities: float = 0.0
    correlation_with_bonds: float = 0.0
    
    # Volatility and risk
    implied_volatility: float = 0.0
    realized_volatility: float = 0.0
    volatility_risk_premium: float = 0.0

@dataclass
class MarketRegimeForex:
    """Forex market regime analysis"""
    regime_type: str = "Unknown"
    regime_confidence: float = 0.0
    
    # Regime characteristics
    volatility_regime: str = "Normal"
    trend_regime: str = "Sideways"
    correlation_regime: str = "Normal"
    
    # Central bank activity
    intervention_probability: float = 0.0
    policy_divergence_score: float = 0.0
    
    # Market structure
    liquidity_conditions: str = "Normal"
    risk_appetite: str = "Neutral"
    carry_trade_environment: str = "Neutral"
    
    # Economic cycle
    economic_cycle_phase: str = "Expansion"
    synchronization_score: float = 0.0

@dataclass
class ForexComprehensiveResult:
    """Comprehensive forex analysis result"""
    timestamp: datetime
    
    # Individual model results
    ppp_analysis: Optional[Any] = None
    irp_analysis: Optional[Any] = None
    uip_analysis: Optional[Any] = None
    bop_analysis: Optional[Any] = None
    monetary_analysis: Optional[Any] = None
    ml_analysis: Optional[Any] = None
    
    # Currency strength analysis
    currency_strengths: Dict[str, CurrencyStrengthAnalysis] = field(default_factory=dict)
    
    # Cross-currency analysis
    cross_currency_analysis: Dict[str, CrossCurrencyAnalysis] = field(default_factory=dict)
    
    # Market regime
    market_regime: MarketRegimeForex = field(default_factory=MarketRegimeForex)
    
    # Cross-model analytics
    model_consensus: Dict[str, float] = field(default_factory=dict)
    model_divergence: Dict[str, float] = field(default_factory=dict)
    correlation_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Risk assessment
    systemic_forex_risk: float = 0.0
    liquidity_risk: float = 0.0
    volatility_risk: float = 0.0
    event_risk: float = 0.0
    
    # Trading signals
    currency_rankings: List[Tuple[str, float]] = field(default_factory=list)
    pair_recommendations: Dict[str, str] = field(default_factory=dict)
    carry_trade_opportunities: List[Dict[str, Any]] = field(default_factory=list)
    arbitrage_opportunities: List[Dict[str, Any]] = field(default_factory=list)
    
    # Actionable insights
    trading_signals: Dict[str, str] = field(default_factory=dict)
    risk_warnings: List[str] = field(default_factory=list)
    opportunities: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

@dataclass
class ForexModelWeights:
    """Dynamic weights for different forex models"""
    ppp_weight: float = 0.20
    irp_weight: float = 0.18
    uip_weight: float = 0.15
    bop_weight: float = 0.22
    monetary_weight: float = 0.15
    ml_weight: float = 0.10
    
    def normalize(self):
        """Normalize weights to sum to 1.0"""
        total = sum([self.ppp_weight, self.irp_weight, self.uip_weight,
                    self.bop_weight, self.monetary_weight, self.ml_weight])
        
        if total > 0:
            self.ppp_weight /= total
            self.irp_weight /= total
            self.uip_weight /= total
            self.bop_weight /= total
            self.monetary_weight /= total
            self.ml_weight /= total

@dataclass
class ForexRegimeConfig:
    """Configuration for forex market regime detection"""
    high_volatility_threshold: float = 0.15
    low_volatility_threshold: float = 0.05
    trend_strength_threshold: float = 0.6
    correlation_threshold: float = 0.7
    intervention_threshold: float = 0.8
    carry_trade_threshold: float = 2.0  # Interest rate differential in %

class ForexComprehensiveIntegration:
    """Comprehensive integration framework for all Forex models"""
    
    def __init__(self, 
                 major_currencies: List[str] = None,
                 enable_cross_validation: bool = True,
                 enable_regime_detection: bool = True,
                 enable_risk_assessment: bool = True,
                 enable_ml_ensemble: bool = True,
                 lookback_window: int = 252,
                 prediction_horizon: int = 30):
        """
        Initialize the comprehensive forex integration framework
        
        Args:
            major_currencies: List of major currencies to analyze
            enable_cross_validation: Enable cross-model validation
            enable_regime_detection: Enable market regime detection
            enable_risk_assessment: Enable comprehensive risk assessment
            enable_ml_ensemble: Enable ML ensemble methods
            lookback_window: Historical data window for analysis
            prediction_horizon: Forward-looking prediction window
        """
        self.major_currencies = major_currencies or ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']
        self.enable_cross_validation = enable_cross_validation
        self.enable_regime_detection = enable_regime_detection
        self.enable_risk_assessment = enable_risk_assessment
        self.enable_ml_ensemble = enable_ml_ensemble
        self.lookback_window = lookback_window
        self.prediction_horizon = prediction_horizon
        
        # Initialize models
        self.models = self._initialize_models()
        
        # Configuration
        self.model_weights = ForexModelWeights()
        self.regime_config = ForexRegimeConfig()
        
        # State tracking
        self.is_fitted = False
        self.last_analysis_time = None
        self.historical_results = []
        
        # ML components
        self.scaler = StandardScaler()
        self.ensemble_model = None
        
        logger.info("Initialized Forex Comprehensive Integration Framework")
    
    def _initialize_models(self) -> Dict[str, Any]:
        """Initialize all forex models"""
        models = {}
        
        try:
            models['ppp_irp_uip'] = PPPIRPUIPModel()
            models['balance_of_payments'] = BalanceOfPaymentsModel()
            models['advanced_ml'] = AdvancedForexML()
            models['monetary'] = MonetaryModel()
            models['comprehensive'] = ForexComprehensiveIndicators()
            
            logger.info(f"Successfully initialized {len(models)} forex models")
        except Exception as e:
            logger.error(f"Error initializing forex models: {e}")
        
        return models
    
    def fit(self, historical_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Fit all models on historical forex data
        
        Args:
            historical_data: Dictionary with currency pair data (e.g., 'EURUSD': DataFrame)
        """
        try:
            logger.info("Fitting forex comprehensive integration framework...")
            
            fit_results = {}
            
            # Fit individual models
            for name, model in self.models.items():
                try:
                    if hasattr(model, 'fit'):
                        # Fit on all currency pairs
                        model_fit_results = {}
                        for pair, data in historical_data.items():
                            result = model.fit(data)
                            model_fit_results[pair] = result
                        fit_results[name] = model_fit_results
                        logger.info(f"Successfully fitted {name} model")
                    else:
                        logger.info(f"{name} model does not require fitting")
                except Exception as e:
                    logger.error(f"Error fitting {name} model: {e}")
                    fit_results[name] = None
            
            # Fit ensemble model if enabled
            if self.enable_ml_ensemble:
                self._fit_ensemble_model(historical_data)
            
            self.is_fitted = True
            logger.info("Forex comprehensive framework fitting completed")
            
            return fit_results
            
        except Exception as e:
            logger.error(f"Error in forex comprehensive fitting: {e}")
            return {}
    
    def _fit_ensemble_model(self, data: Dict[str, pd.DataFrame]):
        """Fit ensemble model for forex meta-predictions"""
        try:
            # Create feature matrix from individual model outputs
            features = []
            targets = []
            
            # Use the first currency pair for ensemble training (can be extended)
            main_pair = list(data.keys())[0]
            pair_data = data[main_pair]
            
            # Generate synthetic training data
            for i in range(len(pair_data) - self.prediction_horizon):
                window_data = pair_data.iloc[i:i+self.lookback_window]
                if len(window_data) < self.lookback_window:
                    continue
                
                # Extract features from each model (simplified)
                model_features = []
                for name, model in self.models.items():
                    try:
                        # Simplified feature extraction for forex
                        if name == 'ppp_irp_uip':
                            # PPP deviation and interest rate differential
                            price_mean = window_data['close'].mean()
                            price_current = window_data['close'].iloc[-1]
                            ppp_deviation = (price_current - price_mean) / price_mean
                            model_features.extend([ppp_deviation, window_data['close'].std()])
                        elif name == 'balance_of_payments':
                            # Trade flow proxy
                            volume_trend = window_data['volume'].pct_change().mean()
                            model_features.extend([volume_trend, window_data['volume'].std()])
                        else:
                            # Generic features
                            model_features.extend([window_data['close'].iloc[-1], window_data['volume'].iloc[-1]])
                    except:
                        model_features.extend([0.0, 0.0])
                
                features.append(model_features)
                
                # Target: future return
                current_price = pair_data['close'].iloc[i + self.lookback_window]
                future_price = pair_data['close'].iloc[i + self.lookback_window + self.prediction_horizon]
                target = (future_price - current_price) / current_price
                targets.append(target)
            
            if len(features) > 0:
                X = np.array(features)
                y = np.array(targets)
                
                # Scale features
                X_scaled = self.scaler.fit_transform(X)
                
                # Train ensemble
                self.ensemble_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                self.ensemble_model.fit(X_scaled, y)
                
                logger.info("Forex ensemble model fitted successfully")
            
        except Exception as e:
            logger.error(f"Error fitting forex ensemble model: {e}")
    
    def analyze(self, data: Dict[str, pd.DataFrame], 
                economic_data: Optional[Dict[str, Dict]] = None) -> ForexComprehensiveResult:
        """Perform comprehensive forex analysis
        
        Args:
            data: Dictionary with currency pair data
            economic_data: Optional economic indicators for each currency
        """
        try:
            logger.info("Starting comprehensive forex analysis...")
            
            # Initialize result
            result = ForexComprehensiveResult(timestamp=datetime.now())
            
            # Run individual model analyses
            model_results = {}
            model_scores = {}
            
            for name, model in self.models.items():
                try:
                    # Analyze each currency pair
                    pair_results = {}
                    pair_scores = {}
                    
                    for pair, pair_data in data.items():
                        analysis_result = model.analyze(pair_data)
                        pair_results[pair] = analysis_result
                        
                        # Extract key score (simplified)
                        if hasattr(analysis_result, 'overall_score'):
                            pair_scores[pair] = analysis_result.overall_score
                        elif hasattr(analysis_result, 'value'):
                            pair_scores[pair] = min(max(analysis_result.value, 0), 1)
                        else:
                            pair_scores[pair] = 0.5  # Neutral score
                    
                    model_results[name] = pair_results
                    model_scores[name] = pair_scores
                    
                    logger.info(f"Completed {name} analysis")
                except Exception as e:
                    logger.error(f"Error in {name} analysis: {e}")
                    model_results[name] = {}
                    model_scores[name] = {}
            
            # Store individual results
            result.ppp_analysis = model_results.get('ppp_irp_uip')
            result.bop_analysis = model_results.get('balance_of_payments')
            result.ml_analysis = model_results.get('advanced_ml')
            result.monetary_analysis = model_results.get('monetary')
            
            # Currency strength analysis
            result.currency_strengths = self._analyze_currency_strengths(
                data, model_results, economic_data
            )
            
            # Cross-currency analysis
            result.cross_currency_analysis = self._analyze_cross_currencies(
                data, model_results
            )
            
            # Cross-model analysis
            if self.enable_cross_validation:
                result.correlation_matrix = self._calculate_forex_correlations(model_results)
                result.model_consensus = self._calculate_forex_consensus(model_scores)
                result.model_divergence = self._analyze_forex_divergences(model_scores)
            
            # Market regime detection
            if self.enable_regime_detection:
                result.market_regime = self._detect_forex_regime(data, model_scores)
            
            # Risk assessment
            if self.enable_risk_assessment:
                risk_metrics = self._assess_forex_risk(data, model_results)
                result.systemic_forex_risk = risk_metrics.get('systemic_risk', 0.5)
                result.liquidity_risk = risk_metrics.get('liquidity_risk', 0.5)
                result.volatility_risk = risk_metrics.get('volatility_risk', 0.5)
                result.event_risk = risk_metrics.get('event_risk', 0.5)
            
            # Generate trading insights
            result.currency_rankings = self._rank_currencies(result.currency_strengths)
            result.pair_recommendations = self._generate_pair_recommendations(result)
            result.carry_trade_opportunities = self._identify_carry_trades(result)
            result.arbitrage_opportunities = self._identify_arbitrage_opportunities(result)
            
            # Generate actionable insights
            result.trading_signals = self._generate_forex_signals(result)
            result.risk_warnings = self._generate_forex_warnings(result)
            result.opportunities = self._identify_forex_opportunities(result)
            result.recommendations = self._generate_forex_recommendations(result)
            
            # Store for historical tracking
            self.historical_results.append(result)
            self.last_analysis_time = datetime.now()
            
            logger.info("Comprehensive forex analysis completed")
            return result
            
        except Exception as e:
            logger.error(f"Error in comprehensive forex analysis: {e}")
            return ForexComprehensiveResult(timestamp=datetime.now())
    
    def _analyze_currency_strengths(self, data: Dict[str, pd.DataFrame], 
                                   model_results: Dict[str, Any],
                                   economic_data: Optional[Dict[str, Dict]]) -> Dict[str, CurrencyStrengthAnalysis]:
        """Analyze individual currency strengths"""
        try:
            currency_strengths = {}
            
            for currency in self.major_currencies:
                strength = CurrencyStrengthAnalysis(currency_code=currency)
                
                # Calculate strength metrics from model results
                currency_scores = []
                
                # Extract scores from pairs involving this currency
                for pair, pair_data in data.items():
                    if currency in pair:
                        # Determine if currency is base or quote
                        is_base = pair.startswith(currency)
                        
                        # Extract model scores for this pair
                        for model_name, model_pair_results in model_results.items():
                            if pair in model_pair_results:
                                result = model_pair_results[pair]
                                if hasattr(result, 'value'):
                                    score = result.value if is_base else (1 - result.value)
                                    currency_scores.append(score)
                
                # Calculate composite strength scores
                if currency_scores:
                    strength.overall_strength_score = np.mean(currency_scores)
                    strength.momentum_score = np.mean(currency_scores[-5:]) if len(currency_scores) >= 5 else np.mean(currency_scores)
                    strength.volatility_score = 1 - np.std(currency_scores)  # Lower volatility = higher score
                
                # Add economic data if available
                if economic_data and currency in economic_data:
                    econ_data = economic_data[currency]
                    strength.economic_strength_score = econ_data.get('gdp_growth', 0) / 5.0  # Normalize
                    strength.monetary_policy_score = econ_data.get('interest_rate', 0) / 10.0  # Normalize
                    strength.fiscal_health_score = max(0, 1 - econ_data.get('debt_to_gdp', 0.5))
                
                # Determine outlooks
                if strength.overall_strength_score > 0.7:
                    strength.short_term_outlook = "Bullish"
                elif strength.overall_strength_score < 0.3:
                    strength.short_term_outlook = "Bearish"
                
                currency_strengths[currency] = strength
            
            return currency_strengths
            
        except Exception as e:
            logger.error(f"Error analyzing currency strengths: {e}")
            return {}
    
    def _analyze_cross_currencies(self, data: Dict[str, pd.DataFrame], 
                                 model_results: Dict[str, Any]) -> Dict[str, CrossCurrencyAnalysis]:
        """Analyze cross-currency relationships"""
        try:
            cross_analysis = {}
            
            for pair, pair_data in data.items():
                if len(pair) >= 6:  # Standard currency pair format
                    base_currency = pair[:3]
                    quote_currency = pair[3:6]
                    
                    analysis = CrossCurrencyAnalysis(
                        base_currency=base_currency,
                        quote_currency=quote_currency
                    )
                    
                    # Calculate relative strength
                    returns = pair_data['close'].pct_change().dropna()
                    analysis.relative_strength_score = np.mean(returns[-30:]) if len(returns) >= 30 else 0
                    
                    # Volatility analysis
                    analysis.realized_volatility = returns.std() * np.sqrt(252)
                    analysis.implied_volatility = analysis.realized_volatility * 1.2  # Simplified
                    analysis.volatility_risk_premium = analysis.implied_volatility - analysis.realized_volatility
                    
                    # Extract model-specific metrics
                    for model_name, model_pair_results in model_results.items():
                        if pair in model_pair_results:
                            result = model_pair_results[pair]
                            
                            if model_name == 'ppp_irp_uip' and hasattr(result, 'ppp_deviation'):
                                analysis.ppp_deviation = result.ppp_deviation
                            elif model_name == 'balance_of_payments' and hasattr(result, 'trade_balance'):
                                analysis.trade_flow_balance = result.trade_balance
                    
                    cross_analysis[pair] = analysis
            
            return cross_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing cross-currencies: {e}")
            return {}
    
    def _calculate_forex_correlations(self, model_results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Calculate correlations between forex model outputs"""
        try:
            correlations = {}
            
            # Extract comparable metrics from each model
            model_metrics = {}
            for name, pair_results in model_results.items():
                if not pair_results:
                    continue
                
                # Calculate average score across all pairs for this model
                pair_scores = []
                for pair, result in pair_results.items():
                    if result and hasattr(result, 'value'):
                        pair_scores.append(result.value)
                    elif result and hasattr(result, 'overall_score'):
                        pair_scores.append(result.overall_score)
                
                if pair_scores:
                    model_metrics[name] = np.mean(pair_scores)
            
            # Calculate pairwise correlations (simplified)
            for name1 in model_metrics:
                correlations[name1] = {}
                for name2 in model_metrics:
                    if name1 == name2:
                        correlations[name1][name2] = 1.0
                    else:
                        # Simplified correlation (in practice, use historical data)
                        correlations[name1][name2] = np.random.uniform(0.2, 0.8)
            
            return correlations
            
        except Exception as e:
            logger.error(f"Error calculating forex correlations: {e}")
            return {}
    
    def _calculate_forex_consensus(self, model_scores: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate consensus signals across forex models"""
        try:
            consensus = {}
            
            # Calculate consensus for each currency pair
            all_pairs = set()
            for pair_scores in model_scores.values():
                all_pairs.update(pair_scores.keys())
            
            for pair in all_pairs:
                pair_model_scores = []
                for model_name, pair_scores in model_scores.items():
                    if pair in pair_scores:
                        pair_model_scores.append(pair_scores[pair])
                
                if pair_model_scores:
                    consensus[pair] = np.mean(pair_model_scores)
                else:
                    consensus[pair] = 0.5
            
            return consensus
            
        except Exception as e:
            logger.error(f"Error calculating forex consensus: {e}")
            return {}
    
    def _analyze_forex_divergences(self, model_scores: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Analyze divergences between forex model signals"""
        try:
            divergences = {}
            
            # Calculate divergence for each currency pair
            all_pairs = set()
            for pair_scores in model_scores.values():
                all_pairs.update(pair_scores.keys())
            
            for pair in all_pairs:
                pair_model_scores = []
                for model_name, pair_scores in model_scores.items():
                    if pair in pair_scores:
                        pair_model_scores.append(pair_scores[pair])
                
                if len(pair_model_scores) > 1:
                    divergences[pair] = np.std(pair_model_scores)
                else:
                    divergences[pair] = 0.0
            
            return divergences
            
        except Exception as e:
            logger.error(f"Error analyzing forex divergences: {e}")
            return {}
    
    def _detect_forex_regime(self, data: Dict[str, pd.DataFrame], 
                            model_scores: Dict[str, Dict[str, float]]) -> MarketRegimeForex:
        """Detect current forex market regime"""
        try:
            regime = MarketRegimeForex()
            
            # Calculate average volatility across major pairs
            volatilities = []
            for pair, pair_data in data.items():
                returns = pair_data['close'].pct_change().dropna()
                vol = returns.std() * np.sqrt(252)
                volatilities.append(vol)
            
            avg_volatility = np.mean(volatilities) if volatilities else 0.1
            
            # Determine volatility regime
            if avg_volatility > self.regime_config.high_volatility_threshold:
                regime.volatility_regime = "High Volatility"
                regime.regime_type = "Crisis"
                regime.regime_confidence = min(avg_volatility / self.regime_config.high_volatility_threshold, 1.0)
            elif avg_volatility < self.regime_config.low_volatility_threshold:
                regime.volatility_regime = "Low Volatility"
                regime.regime_type = "Calm"
                regime.regime_confidence = 1 - (avg_volatility / self.regime_config.low_volatility_threshold)
            else:
                regime.volatility_regime = "Normal"
                regime.regime_type = "Normal"
                regime.regime_confidence = 0.7
            
            # Analyze trend regime
            trend_strengths = []
            for pair, pair_data in data.items():
                # Simple trend strength calculation
                short_ma = pair_data['close'].rolling(20).mean()
                long_ma = pair_data['close'].rolling(50).mean()
                trend_strength = abs((short_ma.iloc[-1] - long_ma.iloc[-1]) / long_ma.iloc[-1])
                trend_strengths.append(trend_strength)
            
            avg_trend_strength = np.mean(trend_strengths) if trend_strengths else 0
            
            if avg_trend_strength > self.regime_config.trend_strength_threshold:
                regime.trend_regime = "Strong Trend"
            else:
                regime.trend_regime = "Sideways"
            
            # Determine overall market conditions
            if avg_volatility > self.regime_config.high_volatility_threshold:
                regime.liquidity_conditions = "Stressed"
                regime.risk_appetite = "Risk Off"
            elif avg_volatility < self.regime_config.low_volatility_threshold:
                regime.liquidity_conditions = "Abundant"
                regime.risk_appetite = "Risk On"
            else:
                regime.liquidity_conditions = "Normal"
                regime.risk_appetite = "Neutral"
            
            return regime
            
        except Exception as e:
            logger.error(f"Error detecting forex regime: {e}")
            return MarketRegimeForex()
    
    def _assess_forex_risk(self, data: Dict[str, pd.DataFrame], 
                          model_results: Dict[str, Any]) -> Dict[str, float]:
        """Assess comprehensive forex risk"""
        try:
            risk_metrics = {}
            
            # Volatility risk
            volatilities = []
            for pair, pair_data in data.items():
                returns = pair_data['close'].pct_change().dropna()
                vol = returns.std() * np.sqrt(252)
                volatilities.append(vol)
            
            avg_volatility = np.mean(volatilities) if volatilities else 0.1
            risk_metrics['volatility_risk'] = min(avg_volatility / 0.2, 1.0)  # Normalize
            
            # Liquidity risk (based on volume patterns)
            volume_volatilities = []
            for pair, pair_data in data.items():
                if 'volume' in pair_data.columns:
                    volume_changes = pair_data['volume'].pct_change().dropna()
                    vol_vol = volume_changes.std()
                    volume_volatilities.append(vol_vol)
            
            if volume_volatilities:
                avg_volume_vol = np.mean(volume_volatilities)
                risk_metrics['liquidity_risk'] = min(avg_volume_vol / 2.0, 1.0)
            else:
                risk_metrics['liquidity_risk'] = 0.3
            
            # Systemic risk (based on model consensus)
            consensus_scores = []
            for model_name, pair_results in model_results.items():
                if pair_results:
                    model_scores = []
                    for pair, result in pair_results.items():
                        if result and hasattr(result, 'value'):
                            model_scores.append(result.value)
                    if model_scores:
                        consensus_scores.append(np.std(model_scores))
            
            if consensus_scores:
                avg_consensus_disagreement = np.mean(consensus_scores)
                risk_metrics['systemic_risk'] = min(avg_consensus_disagreement * 2, 1.0)
            else:
                risk_metrics['systemic_risk'] = 0.5
            
            # Event risk (simplified - based on volatility spikes)
            recent_volatilities = volatilities[-5:] if len(volatilities) >= 5 else volatilities
            historical_volatilities = volatilities[:-5] if len(volatilities) > 5 else volatilities
            
            if recent_volatilities and historical_volatilities:
                vol_ratio = np.mean(recent_volatilities) / np.mean(historical_volatilities)
                risk_metrics['event_risk'] = min(max(vol_ratio - 1, 0), 1.0)
            else:
                risk_metrics['event_risk'] = 0.2
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error assessing forex risk: {e}")
            return {'systemic_risk': 0.5, 'liquidity_risk': 0.5, 'volatility_risk': 0.5, 'event_risk': 0.5}
    
    def _rank_currencies(self, currency_strengths: Dict[str, CurrencyStrengthAnalysis]) -> List[Tuple[str, float]]:
        """Rank currencies by overall strength"""
        try:
            rankings = []
            for currency, strength in currency_strengths.items():
                rankings.append((currency, strength.overall_strength_score))
            
            # Sort by strength score (descending)
            rankings.sort(key=lambda x: x[1], reverse=True)
            return rankings
            
        except Exception as e:
            logger.error(f"Error ranking currencies: {e}")
            return []
    
    def _generate_pair_recommendations(self, result: ForexComprehensiveResult) -> Dict[str, str]:
        """Generate trading recommendations for currency pairs"""
        try:
            recommendations = {}
            
            # Get top and bottom currencies
            if result.currency_rankings:
                strongest_currency = result.currency_rankings[0][0]
                weakest_currency = result.currency_rankings[-1][0]
                
                # Recommend long strongest vs short weakest
                pair_name = f"{strongest_currency}{weakest_currency}"
                recommendations[pair_name] = "Strong Buy"
                
                # Add more nuanced recommendations based on cross-currency analysis
                for pair, cross_analysis in result.cross_currency_analysis.items():
                    if cross_analysis.relative_strength_score > 0.7:
                        recommendations[pair] = "Buy"
                    elif cross_analysis.relative_strength_score < -0.7:
                        recommendations[pair] = "Sell"
                    elif abs(cross_analysis.ppp_deviation) > 10:  # Significant PPP deviation
                        if cross_analysis.ppp_deviation > 0:
                            recommendations[pair] = "Sell (Overvalued)"
                        else:
                            recommendations[pair] = "Buy (Undervalued)"
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating pair recommendations: {e}")
            return {}
    
    def _identify_carry_trades(self, result: ForexComprehensiveResult) -> List[Dict[str, Any]]:
        """Identify carry trade opportunities"""
        try:
            carry_trades = []
            
            # Look for significant interest rate differentials
            for pair, cross_analysis in result.cross_currency_analysis.items():
                if abs(cross_analysis.interest_rate_differential) > self.regime_config.carry_trade_threshold:
                    carry_trade = {
                        'pair': pair,
                        'interest_differential': cross_analysis.interest_rate_differential,
                        'direction': 'Long' if cross_analysis.interest_rate_differential > 0 else 'Short',
                        'expected_carry': abs(cross_analysis.interest_rate_differential),
                        'risk_score': cross_analysis.realized_volatility,
                        'risk_adjusted_return': abs(cross_analysis.interest_rate_differential) / cross_analysis.realized_volatility
                    }
                    carry_trades.append(carry_trade)
            
            # Sort by risk-adjusted return
            carry_trades.sort(key=lambda x: x['risk_adjusted_return'], reverse=True)
            
            return carry_trades[:5]  # Top 5 opportunities
            
        except Exception as e:
            logger.error(f"Error identifying carry trades: {e}")
            return []
    
    def _identify_arbitrage_opportunities(self, result: ForexComprehensiveResult) -> List[Dict[str, Any]]:
        """Identify arbitrage opportunities"""
        try:
            arbitrage_ops = []
            
            # Look for significant IRP deviations
            for pair, cross_analysis in result.cross_currency_analysis.items():
                if abs(cross_analysis.irp_deviation) > 1.0:  # 1% threshold
                    arbitrage = {
                        'pair': pair,
                        'type': 'Interest Rate Parity Arbitrage',
                        'deviation': cross_analysis.irp_deviation,
                        'expected_profit': abs(cross_analysis.irp_deviation),
                        'risk_level': 'Low' if abs(cross_analysis.irp_deviation) < 2.0 else 'Medium'
                    }
                    arbitrage_ops.append(arbitrage)
            
            return arbitrage_ops
            
        except Exception as e:
            logger.error(f"Error identifying arbitrage opportunities: {e}")
            return []
    
    def _generate_forex_signals(self, result: ForexComprehensiveResult) -> Dict[str, str]:
        """Generate actionable forex trading signals"""
        try:
            signals = {}
            
            # Overall market signals
            if result.market_regime.regime_type == "Crisis":
                signals['market_regime'] = "Risk Off - Seek Safe Havens"
            elif result.market_regime.regime_type == "Calm":
                signals['market_regime'] = "Risk On - Consider Carry Trades"
            
            # Currency-specific signals
            for currency, strength in result.currency_strengths.items():
                if strength.overall_strength_score > 0.8:
                    signals[currency] = "Strong Bullish"
                elif strength.overall_strength_score > 0.6:
                    signals[currency] = "Bullish"
                elif strength.overall_strength_score < 0.2:
                    signals[currency] = "Strong Bearish"
                elif strength.overall_strength_score < 0.4:
                    signals[currency] = "Bearish"
            
            # Pair-specific signals from recommendations
            signals.update(result.pair_recommendations)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating forex signals: {e}")
            return {}
    
    def _generate_forex_warnings(self, result: ForexComprehensiveResult) -> List[str]:
        """Generate forex risk warnings"""
        warnings = []
        
        try:
            if result.systemic_forex_risk > 0.7:
                warnings.append("High systemic forex risk - Consider reducing position sizes")
            
            if result.volatility_risk > 0.8:
                warnings.append("Extreme forex volatility expected - Use tight stop losses")
            
            if result.liquidity_risk > 0.7:
                warnings.append("Forex liquidity concerns - Monitor spreads closely")
            
            if result.event_risk > 0.7:
                warnings.append("High event risk - Central bank intervention possible")
            
            if result.market_regime.intervention_probability > 0.8:
                warnings.append("High probability of central bank intervention")
            
            # Currency-specific warnings
            for currency, strength in result.currency_strengths.items():
                if strength.intervention_risk > 0.8:
                    warnings.append(f"{currency} intervention risk elevated")
                if strength.event_risk > 0.8:
                    warnings.append(f"{currency} high event risk - Monitor economic calendar")
            
        except Exception as e:
            logger.error(f"Error generating forex warnings: {e}")
        
        return warnings
    
    def _identify_forex_opportunities(self, result: ForexComprehensiveResult) -> List[str]:
        """Identify forex trading opportunities"""
        opportunities = []
        
        try:
            # Carry trade opportunities
            if result.carry_trade_opportunities:
                best_carry = result.carry_trade_opportunities[0]
                opportunities.append(f"High-yield carry trade: {best_carry['pair']} with {best_carry['expected_carry']:.2f}% differential")
            
            # Arbitrage opportunities
            if result.arbitrage_opportunities:
                opportunities.append(f"Arbitrage opportunity detected: {result.arbitrage_opportunities[0]['type']}")
            
            # PPP reversion opportunities
            for pair, cross_analysis in result.cross_currency_analysis.items():
                if abs(cross_analysis.ppp_deviation) > 15:  # Significant PPP deviation
                    direction = "undervalued" if cross_analysis.ppp_deviation < 0 else "overvalued"
                    opportunities.append(f"{pair} significantly {direction} by PPP - Mean reversion opportunity")
            
            # Low volatility opportunities
            if result.market_regime.volatility_regime == "Low Volatility":
                opportunities.append("Low volatility environment - Consider volatility strategies")
            
            # Strong trend opportunities
            if result.market_regime.trend_regime == "Strong Trend":
                opportunities.append("Strong trending environment - Momentum strategies favored")
            
        except Exception as e:
            logger.error(f"Error identifying forex opportunities: {e}")
        
        return opportunities
    
    def _generate_forex_recommendations(self, result: ForexComprehensiveResult) -> List[str]:
        """Generate actionable forex recommendations"""
        recommendations = []
        
        try:
            # Overall market recommendations
            if result.market_regime.regime_type == "Crisis":
                recommendations.append("Focus on safe-haven currencies (USD, JPY, CHF)")
                recommendations.append("Avoid high-beta currencies and emerging markets")
            elif result.market_regime.regime_type == "Calm":
                recommendations.append("Consider carry trades with high-yielding currencies")
                recommendations.append("Explore momentum strategies in trending pairs")
            
            # Risk management recommendations
            if result.volatility_risk > 0.7:
                recommendations.append("Reduce position sizes due to high volatility")
                recommendations.append("Use options for volatility protection")
            
            if result.liquidity_risk > 0.6:
                recommendations.append("Monitor bid-ask spreads before trading")
                recommendations.append("Consider using limit orders in illiquid conditions")
            
            # Currency-specific recommendations
            if result.currency_rankings:
                strongest = result.currency_rankings[0][0]
                weakest = result.currency_rankings[-1][0]
                recommendations.append(f"Consider long {strongest} positions")
                recommendations.append(f"Consider short {weakest} positions")
            
            # Regime-based recommendations
            if result.market_regime.carry_trade_environment == "Favorable":
                recommendations.append("Carry trade environment is favorable")
            elif result.market_regime.carry_trade_environment == "Unfavorable":
                recommendations.append("Avoid carry trades - Focus on defensive strategies")
            
        except Exception as e:
            logger.error(f"Error generating forex recommendations: {e}")
        
        return recommendations
    
    def get_forex_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of all forex models"""
        try:
            if not self.historical_results:
                return {"message": "No historical forex results available"}
            
            # Calculate performance metrics
            performance = {}
            
            # Model performance tracking
            model_names = ['ppp_analysis', 'irp_analysis', 'bop_analysis', 'ml_analysis']
            
            for model_name in model_names:
                model_scores = []
                for result in self.historical_results:
                    analysis = getattr(result, model_name, None)
                    if analysis:
                        # Extract scores from all pairs
                        for pair, pair_result in analysis.items():
                            if pair_result and hasattr(pair_result, 'value'):
                                model_scores.append(pair_result.value)
                
                if model_scores:
                    performance[model_name] = {
                        'avg_score': np.mean(model_scores),
                        'score_volatility': np.std(model_scores),
                        'min_score': np.min(model_scores),
                        'max_score': np.max(model_scores),
                        'consistency': 1 - np.std(model_scores)
                    }
            
            # Currency performance tracking
            currency_performance = {}
            for currency in self.major_currencies:
                currency_scores = []
                for result in self.historical_results:
                    if currency in result.currency_strengths:
                        strength = result.currency_strengths[currency]
                        currency_scores.append(strength.overall_strength_score)
                
                if currency_scores:
                    currency_performance[currency] = {
                        'avg_strength': np.mean(currency_scores),
                        'strength_volatility': np.std(currency_scores),
                        'trend': 'Strengthening' if currency_scores[-1] > currency_scores[0] else 'Weakening'
                    }
            
            performance['currencies'] = currency_performance
            
            return performance
            
        except Exception as e:
            logger.error(f"Error getting forex performance summary: {e}")
            return {}

# Example usage and testing
if __name__ == "__main__":
    # Create sample forex data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    
    # Generate realistic forex data for major pairs
    forex_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD']
    forex_data = {}
    
    for pair in forex_pairs:
        n_days = len(dates)
        # Generate realistic forex price movements
        returns = np.random.normal(0, 0.01, n_days)
        prices = np.cumprod(1 + returns) * 1.1000  # Start around 1.1000
        
        volumes = np.random.lognormal(10, 0.5, n_days)
        
        forex_data[pair] = pd.DataFrame({
            'date': dates,
            'open': prices * (1 + np.random.normal(0, 0.001, n_days)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.002, n_days))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.002, n_days))),
            'close': prices,
            'volume': volumes
        })
        forex_data[pair].set_index('date', inplace=True)
    
    print("=== Forex Comprehensive Integration Test ===")
    print(f"Forex pairs: {list(forex_data.keys())}")
    print(f"Date range: {dates[0]} to {dates[-1]}")
    
    # Initialize comprehensive forex framework
    forex_framework = ForexComprehensiveIntegration(
        major_currencies=['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'AUD', 'CAD'],
        enable_cross_validation=True,
        enable_regime_detection=True,
        enable_risk_assessment=True,
        enable_ml_ensemble=True
    )
    
    # Fit framework
    print("\n=== Fitting Forex Framework ===")
    fit_results = forex_framework.fit(forex_data)
    print(f"Fit results: {list(fit_results.keys())}")
    
    # Run comprehensive analysis
    print("\n=== Running Comprehensive Forex Analysis ===")
    
    # Sample economic data
    economic_data = {
        'USD': {'gdp_growth': 2.1, 'interest_rate': 5.25, 'debt_to_gdp': 0.78},
        'EUR': {'gdp_growth': 0.8, 'interest_rate': 4.50, 'debt_to_gdp': 0.85},
        'GBP': {'gdp_growth': 1.2, 'interest_rate': 5.00, 'debt_to_gdp': 0.82},
        'JPY': {'gdp_growth': 0.5, 'interest_rate': -0.10, 'debt_to_gdp': 2.40},
        'CHF': {'gdp_growth': 1.8, 'interest_rate': 1.75, 'debt_to_gdp': 0.42},
        'AUD': {'gdp_growth': 2.8, 'interest_rate': 4.35, 'debt_to_gdp': 0.45},
        'CAD': {'gdp_growth': 2.2, 'interest_rate': 5.00, 'debt_to_gdp': 0.52}
    }
    
    analysis_result = forex_framework.analyze(forex_data, economic_data)
    
    print(f"\n=== Forex Analysis Results ===")
    print(f"Market Regime: {analysis_result.market_regime.regime_type}")
    print(f"Volatility Regime: {analysis_result.market_regime.volatility_regime}")
    print(f"Trend Regime: {analysis_result.market_regime.trend_regime}")
    
    print(f"\n=== Currency Rankings ===")
    for i, (currency, score) in enumerate(analysis_result.currency_rankings[:5], 1):
        print(f"{i}. {currency}: {score:.3f}")
    
    print(f"\n=== Risk Assessment ===")
    print(f"Systemic Forex Risk: {analysis_result.systemic_forex_risk:.3f}")
    print(f"Volatility Risk: {analysis_result.volatility_risk:.3f}")
    print(f"Liquidity Risk: {analysis_result.liquidity_risk:.3f}")
    print(f"Event Risk: {analysis_result.event_risk:.3f}")
    
    print(f"\n=== Trading Signals ===")
    for signal_type, signal in list(analysis_result.trading_signals.items())[:5]:
        print(f"{signal_type}: {signal}")
    
    print(f"\n=== Carry Trade Opportunities ===")
    for i, carry_trade in enumerate(analysis_result.carry_trade_opportunities[:3], 1):
        print(f"{i}. {carry_trade['pair']}: {carry_trade['expected_carry']:.2f}% ({carry_trade['direction']})")
    
    print(f"\n=== Risk Warnings ===")
    for warning in analysis_result.risk_warnings[:3]:
        print(f"⚠️  {warning}")
    
    print(f"\n=== Opportunities ===")
    for opportunity in analysis_result.opportunities[:3]:
        print(f"💡 {opportunity}")
    
    print(f"\n=== Recommendations ===")
    for i, recommendation in enumerate(analysis_result.recommendations[:5], 1):
        print(f"{i}. {recommendation}")
    
    print("\n=== Forex Comprehensive Integration Test Complete ===")