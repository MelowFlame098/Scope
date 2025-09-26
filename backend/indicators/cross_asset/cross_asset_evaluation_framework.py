#!/usr/bin/env python3
"""
Cross-Asset Evaluation Framework

A comprehensive evaluation framework for all Cross-Asset indicators and models.
This framework provides:
- Unified evaluation across all cross-asset indicators
- Multi-asset correlation and relationship analysis
- Cross-market arbitrage opportunity detection
- Asset allocation optimization
- Risk parity and diversification analysis
- Macro-economic factor integration
- Currency and commodity impact assessment
- Sector rotation and style factor analysis

Author: Cross-Asset Analysis Team
Date: 2024
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import logging
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Import cross-asset indicators with fallback
try:
    from cross_asset_comprehensive import CrossAssetComprehensiveIndicators
except ImportError:
    logging.warning("Cross-Asset Comprehensive Indicators not available")
    CrossAssetComprehensiveIndicators = None

try:
    from correlation_analysis import CorrelationAnalysis
except ImportError:
    logging.warning("Correlation analysis not available")
    CorrelationAnalysis = None

try:
    from arbitrage_detection import ArbitrageDetection
except ImportError:
    logging.warning("Arbitrage detection not available")
    ArbitrageDetection = None

try:
    from asset_allocation import AssetAllocation
except ImportError:
    logging.warning("Asset allocation not available")
    AssetAllocation = None

try:
    from risk_parity import RiskParity
except ImportError:
    logging.warning("Risk parity not available")
    RiskParity = None

try:
    from macro_factor_analysis import MacroFactorAnalysis
except ImportError:
    logging.warning("Macro factor analysis not available")
    MacroFactorAnalysis = None

try:
    from currency_impact import CurrencyImpact
except ImportError:
    logging.warning("Currency impact not available")
    CurrencyImpact = None

try:
    from commodity_analysis import CommodityAnalysis
except ImportError:
    logging.warning("Commodity analysis not available")
    CommodityAnalysis = None

try:
    from sector_rotation import SectorRotation
except ImportError:
    logging.warning("Sector rotation not available")
    SectorRotation = None

try:
    from style_factor_analysis import StyleFactorAnalysis
except ImportError:
    logging.warning("Style factor analysis not available")
    StyleFactorAnalysis = None

try:
    from volatility_surface import VolatilitySurface
except ImportError:
    logging.warning("Volatility surface not available")
    VolatilitySurface = None

try:
    from yield_curve_analysis import YieldCurveAnalysis
except ImportError:
    logging.warning("Yield curve analysis not available")
    YieldCurveAnalysis = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CrossAssetSignalPerformance:
    """Performance metrics for cross-asset trading signals"""
    indicator_name: str
    asset_universe: List[str]
    analysis_period: str
    
    # Signal accuracy metrics
    total_signals: int = 0
    correct_signals: int = 0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # Trading performance metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    # Cross-asset specific metrics
    correlation_accuracy: float = 0.0
    arbitrage_capture_rate: float = 0.0
    diversification_benefit: float = 0.0
    asset_allocation_efficiency: float = 0.0
    risk_parity_performance: float = 0.0
    
    # Multi-asset performance
    equity_performance: float = 0.0
    bond_performance: float = 0.0
    commodity_performance: float = 0.0
    currency_performance: float = 0.0
    crypto_performance: float = 0.0
    
    # Factor exposure metrics
    market_factor_exposure: float = 0.0
    size_factor_exposure: float = 0.0
    value_factor_exposure: float = 0.0
    momentum_factor_exposure: float = 0.0
    quality_factor_exposure: float = 0.0
    
    # Risk metrics
    cross_asset_risk_score: float = 0.0
    concentration_risk: float = 0.0
    correlation_risk: float = 0.0
    liquidity_risk: float = 0.0

@dataclass
class CrossAssetBacktestResults:
    """Comprehensive cross-asset backtesting results"""
    indicator_name: str
    asset_universe: List[str]
    analysis_period: str
    start_date: datetime
    end_date: datetime
    
    # Portfolio metrics
    initial_capital: float = 100000.0
    final_capital: float = 0.0
    total_return: float = 0.0
    annualized_return: float = 0.0
    
    # Risk metrics
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    
    # Trading metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    avg_holding_period: float = 0.0
    
    # Cross-asset specific metrics
    correlation_stability: float = 0.0
    diversification_ratio: float = 0.0
    asset_turnover: float = 0.0
    rebalancing_frequency: float = 0.0
    
    # Asset class performance
    equity_allocation: float = 0.0
    bond_allocation: float = 0.0
    commodity_allocation: float = 0.0
    currency_allocation: float = 0.0
    crypto_allocation: float = 0.0
    
    equity_return: float = 0.0
    bond_return: float = 0.0
    commodity_return: float = 0.0
    currency_return: float = 0.0
    crypto_return: float = 0.0
    
    # Factor attribution
    market_factor_return: float = 0.0
    size_factor_return: float = 0.0
    value_factor_return: float = 0.0
    momentum_factor_return: float = 0.0
    quality_factor_return: float = 0.0
    
    # Risk attribution
    systematic_risk: float = 0.0
    idiosyncratic_risk: float = 0.0
    concentration_risk: float = 0.0
    
    # Detailed results
    monthly_returns: List[float] = field(default_factory=list)
    asset_weights_history: List[Dict[str, float]] = field(default_factory=list)
    correlation_matrix_history: List[Dict[str, Dict[str, float]]] = field(default_factory=list)
    trade_log: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class CrossAssetPredictionAccuracy:
    """Cross-asset prediction accuracy metrics"""
    indicator_name: str
    asset_universe: List[str]
    analysis_period: str
    prediction_horizon: int  # days
    
    # Correlation prediction
    correlation_prediction_accuracy: float = 0.0
    correlation_direction_accuracy: float = 0.0
    correlation_magnitude_accuracy: float = 0.0
    
    # Asset allocation prediction
    allocation_accuracy: float = 0.0
    rebalancing_timing_accuracy: float = 0.0
    
    # Return prediction by asset class
    equity_return_accuracy: float = 0.0
    bond_return_accuracy: float = 0.0
    commodity_return_accuracy: float = 0.0
    currency_return_accuracy: float = 0.0
    crypto_return_accuracy: float = 0.0
    
    # Factor prediction
    market_factor_accuracy: float = 0.0
    size_factor_accuracy: float = 0.0
    value_factor_accuracy: float = 0.0
    momentum_factor_accuracy: float = 0.0
    quality_factor_accuracy: float = 0.0
    
    # Volatility prediction
    volatility_prediction_accuracy: float = 0.0
    volatility_regime_accuracy: float = 0.0
    
    # Arbitrage prediction
    arbitrage_opportunity_accuracy: float = 0.0
    spread_convergence_accuracy: float = 0.0
    
    # Magnitude prediction errors
    mae_correlation: float = 0.0
    rmse_correlation: float = 0.0
    mae_returns: float = 0.0
    rmse_returns: float = 0.0
    
    # Confidence intervals
    prediction_confidence: float = 0.0
    confidence_calibration: float = 0.0

@dataclass
class CrossAssetRiskMetrics:
    """Comprehensive cross-asset risk assessment"""
    indicator_name: str
    asset_universe: List[str]
    analysis_period: str
    
    # Correlation risks
    correlation_breakdown_risk: float = 0.0
    correlation_clustering_risk: float = 0.0
    tail_correlation_risk: float = 0.0
    
    # Concentration risks
    asset_concentration_risk: float = 0.0
    sector_concentration_risk: float = 0.0
    geographic_concentration_risk: float = 0.0
    
    # Liquidity risks
    market_liquidity_risk: float = 0.0
    funding_liquidity_risk: float = 0.0
    
    # Currency and commodity risks
    currency_exposure_risk: float = 0.0
    commodity_exposure_risk: float = 0.0
    
    # Factor risks
    factor_concentration_risk: float = 0.0
    factor_timing_risk: float = 0.0
    
    # Regime change risks
    volatility_regime_risk: float = 0.0
    correlation_regime_risk: float = 0.0
    market_regime_risk: float = 0.0
    
    # Tail risks
    tail_risk_equity: float = 0.0
    tail_risk_bonds: float = 0.0
    tail_risk_commodities: float = 0.0
    tail_risk_currencies: float = 0.0
    
    # Composite risk scores
    overall_cross_asset_risk: float = 0.0
    systematic_risk_score: float = 0.0
    idiosyncratic_risk_score: float = 0.0
    diversification_effectiveness: float = 0.0

@dataclass
class CrossAssetComparativeAnalysis:
    """Comparative analysis across cross-asset indicators"""
    asset_universe: List[str]
    analysis_period: str
    
    # Indicator rankings
    performance_ranking: List[Tuple[str, float]] = field(default_factory=list)
    accuracy_ranking: List[Tuple[str, float]] = field(default_factory=list)
    risk_ranking: List[Tuple[str, float]] = field(default_factory=list)
    diversification_ranking: List[Tuple[str, float]] = field(default_factory=list)
    
    # Cross-indicator correlations
    signal_correlations: Dict[str, Dict[str, float]] = field(default_factory=dict)
    return_correlations: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Asset class analysis
    equity_indicators_performance: Dict[str, float] = field(default_factory=dict)
    bond_indicators_performance: Dict[str, float] = field(default_factory=dict)
    commodity_indicators_performance: Dict[str, float] = field(default_factory=dict)
    currency_indicators_performance: Dict[str, float] = field(default_factory=dict)
    crypto_indicators_performance: Dict[str, float] = field(default_factory=dict)
    
    # Strategy performance
    correlation_strategies_performance: Dict[str, float] = field(default_factory=dict)
    arbitrage_strategies_performance: Dict[str, float] = field(default_factory=dict)
    allocation_strategies_performance: Dict[str, float] = field(default_factory=dict)
    factor_strategies_performance: Dict[str, float] = field(default_factory=dict)
    
    # Best indicator recommendations
    best_overall_indicator: str = ""
    best_correlation_indicator: str = ""
    best_arbitrage_indicator: str = ""
    best_allocation_indicator: str = ""
    best_risk_indicator: str = ""
    best_diversification_indicator: str = ""

@dataclass
class CrossAssetEvaluationReport:
    """Comprehensive cross-asset evaluation report"""
    evaluation_date: datetime
    asset_universe: List[str]
    analysis_periods: List[str]
    evaluation_timeframe: str
    
    # Individual indicator results
    signal_performance: Dict[str, List[CrossAssetSignalPerformance]] = field(default_factory=dict)
    backtest_results: Dict[str, List[CrossAssetBacktestResults]] = field(default_factory=dict)
    prediction_accuracy: Dict[str, List[CrossAssetPredictionAccuracy]] = field(default_factory=dict)
    risk_metrics: Dict[str, List[CrossAssetRiskMetrics]] = field(default_factory=dict)
    
    # Comparative analysis
    comparative_analysis: Dict[str, CrossAssetComparativeAnalysis] = field(default_factory=dict)
    
    # Summary statistics
    best_performing_indicators: Dict[str, str] = field(default_factory=dict)
    worst_performing_indicators: Dict[str, str] = field(default_factory=dict)
    most_consistent_indicators: Dict[str, str] = field(default_factory=dict)
    
    # Asset class analysis
    asset_class_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Strategy analysis
    correlation_strategy_results: Dict[str, float] = field(default_factory=dict)
    arbitrage_strategy_results: Dict[str, float] = field(default_factory=dict)
    allocation_strategy_results: Dict[str, float] = field(default_factory=dict)
    factor_strategy_results: Dict[str, float] = field(default_factory=dict)
    
    # Recommendations
    indicator_recommendations: Dict[str, List[str]] = field(default_factory=dict)
    risk_warnings: List[str] = field(default_factory=list)
    optimization_suggestions: List[str] = field(default_factory=list)
    
    # Strategy recommendations
    correlation_strategies: List[str] = field(default_factory=list)
    arbitrage_strategies: List[str] = field(default_factory=list)
    allocation_strategies: List[str] = field(default_factory=list)
    factor_strategies: List[str] = field(default_factory=list)
    diversification_strategies: List[str] = field(default_factory=list)

class CrossAssetEvaluationFramework:
    """Comprehensive evaluation framework for Cross-Asset indicators"""
    
    def __init__(self, 
                 asset_universe: List[str] = None,
                 analysis_periods: List[str] = None,
                 evaluation_window: int = 252,  # trading days
                 prediction_horizons: List[int] = None,
                 risk_free_rate: float = 0.02,
                 enable_factor_analysis: bool = True,
                 enable_regime_detection: bool = True):
        """
        Initialize the Cross-Asset evaluation framework
        
        Args:
            asset_universe: List of assets to evaluate across classes
            analysis_periods: List of analysis periods ('daily', 'weekly', 'monthly')
            evaluation_window: Number of trading days for evaluation window
            prediction_horizons: List of prediction horizons in days
            risk_free_rate: Risk-free rate for calculations
            enable_factor_analysis: Enable factor-based analysis
            enable_regime_detection: Enable regime detection analysis
        """
        self.asset_universe = asset_universe or [
            # Equities
            'SPY', 'QQQ', 'IWM', 'EFA', 'EEM',
            # Bonds
            'TLT', 'IEF', 'SHY', 'HYG', 'EMB',
            # Commodities
            'GLD', 'SLV', 'USO', 'DBA', 'DBC',
            # Currencies
            'UUP', 'FXE', 'FXY', 'FXA', 'FXC',
            # Crypto
            'BTC-USD', 'ETH-USD'
        ]
        self.analysis_periods = analysis_periods or ['daily', 'weekly', 'monthly']
        self.evaluation_window = evaluation_window
        self.prediction_horizons = prediction_horizons or [1, 5, 10, 21, 63]  # days
        self.risk_free_rate = risk_free_rate
        self.enable_factor_analysis = enable_factor_analysis
        self.enable_regime_detection = enable_regime_detection
        
        # Initialize indicators
        self.correlation_indicators = self._initialize_correlation_indicators()
        self.arbitrage_indicators = self._initialize_arbitrage_indicators()
        self.allocation_indicators = self._initialize_allocation_indicators()
        self.factor_indicators = self._initialize_factor_indicators()
        self.macro_indicators = self._initialize_macro_indicators()
        
        # Evaluation state
        self.evaluation_history = []
        self.indicator_performance_cache = {}
        
        total_indicators = (len(self.correlation_indicators) + len(self.arbitrage_indicators) + 
                          len(self.allocation_indicators) + len(self.factor_indicators) + 
                          len(self.macro_indicators))
        
        logger.info(f"Initialized Cross-Asset Evaluation Framework with {total_indicators} indicators")
    
    def _initialize_correlation_indicators(self) -> List[Tuple[str, Any]]:
        """Initialize correlation indicators"""
        indicators = []
        
        if CrossAssetComprehensiveIndicators:
            indicators.append(('CrossAsset_Comprehensive', CrossAssetComprehensiveIndicators()))
        
        if CorrelationAnalysis:
            indicators.append(('Correlation_Analysis', CorrelationAnalysis()))
        
        logger.info(f"Initialized {len(indicators)} correlation indicators")
        return indicators
    
    def _initialize_arbitrage_indicators(self) -> List[Tuple[str, Any]]:
        """Initialize arbitrage indicators"""
        indicators = []
        
        if ArbitrageDetection:
            indicators.append(('Arbitrage_Detection', ArbitrageDetection()))
        
        logger.info(f"Initialized {len(indicators)} arbitrage indicators")
        return indicators
    
    def _initialize_allocation_indicators(self) -> List[Tuple[str, Any]]:
        """Initialize allocation indicators"""
        indicators = []
        
        if AssetAllocation:
            indicators.append(('Asset_Allocation', AssetAllocation()))
        
        if RiskParity:
            indicators.append(('Risk_Parity', RiskParity()))
        
        logger.info(f"Initialized {len(indicators)} allocation indicators")
        return indicators
    
    def _initialize_factor_indicators(self) -> List[Tuple[str, Any]]:
        """Initialize factor indicators"""
        indicators = []
        
        if StyleFactorAnalysis:
            indicators.append(('Style_Factor_Analysis', StyleFactorAnalysis()))
        
        if SectorRotation:
            indicators.append(('Sector_Rotation', SectorRotation()))
        
        logger.info(f"Initialized {len(indicators)} factor indicators")
        return indicators
    
    def _initialize_macro_indicators(self) -> List[Tuple[str, Any]]:
        """Initialize macro indicators"""
        indicators = []
        
        if MacroFactorAnalysis:
            indicators.append(('Macro_Factor_Analysis', MacroFactorAnalysis()))
        
        if CurrencyImpact:
            indicators.append(('Currency_Impact', CurrencyImpact()))
        
        if CommodityAnalysis:
            indicators.append(('Commodity_Analysis', CommodityAnalysis()))
        
        if YieldCurveAnalysis:
            indicators.append(('Yield_Curve_Analysis', YieldCurveAnalysis()))
        
        if VolatilitySurface:
            indicators.append(('Volatility_Surface', VolatilitySurface()))
        
        logger.info(f"Initialized {len(indicators)} macro indicators")
        return indicators
    
    def generate_evaluation_report(self, 
                                 asset_data: Dict[str, pd.DataFrame],
                                 macro_data: Optional[Dict[str, pd.DataFrame]] = None) -> CrossAssetEvaluationReport:
        """Generate comprehensive cross-asset evaluation report
        
        Args:
            asset_data: Dictionary with structure {asset: DataFrame}
            macro_data: Optional macro-economic data
        """
        try:
            logger.info("Generating comprehensive cross-asset evaluation report...")
            
            # Calculate actual returns for all assets
            actual_returns = {}
            for asset, df in asset_data.items():
                actual_returns[asset] = df['close'].pct_change().dropna()
            
            # Run all evaluations
            signal_performance = self.evaluate_signal_performance(asset_data, actual_returns, macro_data)
            backtest_results = self.backtest_indicators(asset_data, macro_data)
            prediction_accuracy = self.evaluate_prediction_accuracy(asset_data, macro_data)
            risk_metrics = self.assess_risk_metrics(asset_data, backtest_results)
            comparative_analysis = self.perform_comparative_analysis(
                signal_performance, backtest_results, prediction_accuracy, risk_metrics
            )
            
            # Create comprehensive report
            report = CrossAssetEvaluationReport(
                evaluation_date=datetime.now(),
                asset_universe=self.asset_universe,
                analysis_periods=self.analysis_periods,
                evaluation_timeframe=f"{self.evaluation_window} trading days",
                signal_performance=signal_performance,
                backtest_results=backtest_results,
                prediction_accuracy=prediction_accuracy,
                risk_metrics=risk_metrics,
                comparative_analysis=comparative_analysis
            )
            
            # Generate summary statistics and recommendations
            report = self._generate_report_summary(report)
            
            # Store in history
            self.evaluation_history.append(report)
            
            logger.info("Cross-asset evaluation report generated successfully")
            return report
            
        except Exception as e:
            logger.error(f"Error generating cross-asset evaluation report: {e}")
            return CrossAssetEvaluationReport(
                evaluation_date=datetime.now(),
                asset_universe=self.asset_universe,
                analysis_periods=self.analysis_periods,
                evaluation_timeframe=f"{self.evaluation_window} trading days"
            )
    
    def evaluate_signal_performance(self, 
                                   asset_data: Dict[str, pd.DataFrame],
                                   actual_returns: Dict[str, pd.Series],
                                   macro_data: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, List[CrossAssetSignalPerformance]]:
        """Evaluate signal performance for all cross-asset indicators"""
        try:
            logger.info("Evaluating cross-asset signal performance...")
            
            performance_results = {}
            all_indicators = (self.correlation_indicators + self.arbitrage_indicators + 
                            self.allocation_indicators + self.factor_indicators + 
                            self.macro_indicators)
            
            for indicator_name, indicator in all_indicators:
                indicator_performance = []
                
                for period in self.analysis_periods:
                    try:
                        # Generate signals
                        signals = self._generate_cross_asset_signals(indicator, asset_data, period, macro_data)
                        
                        if signals is None or len(signals) == 0:
                            continue
                        
                        # Evaluate performance
                        performance = self._calculate_cross_asset_performance(
                            indicator_name, period, signals, actual_returns
                        )
                        
                        indicator_performance.append(performance)
                        
                    except Exception as e:
                        logger.error(f"Error evaluating {indicator_name} for {period}: {e}")
                        continue
                
                performance_results[indicator_name] = indicator_performance
            
            logger.info(f"Signal performance evaluation completed for {len(performance_results)} indicators")
            return performance_results
            
        except Exception as e:
            logger.error(f"Error in signal performance evaluation: {e}")
            return {}
    
    def backtest_indicators(self, 
                           asset_data: Dict[str, pd.DataFrame],
                           macro_data: Optional[Dict[str, pd.DataFrame]] = None,
                           start_date: datetime = None,
                           end_date: datetime = None,
                           initial_capital: float = 100000.0) -> Dict[str, List[CrossAssetBacktestResults]]:
        """Backtest all cross-asset indicators"""
        try:
            logger.info("Starting cross-asset indicators backtesting...")
            
            backtest_results = {}
            all_indicators = (self.correlation_indicators + self.arbitrage_indicators + 
                            self.allocation_indicators + self.factor_indicators + 
                            self.macro_indicators)
            
            for indicator_name, indicator in all_indicators:
                indicator_results = []
                
                for period in self.analysis_periods:
                    try:
                        # Run backtest
                        result = self._run_cross_asset_backtest(
                            indicator, indicator_name, period,
                            asset_data, macro_data, 
                            start_date, end_date, initial_capital
                        )
                        
                        if result:
                            indicator_results.append(result)
                        
                    except Exception as e:
                        logger.error(f"Error backtesting {indicator_name} for {period}: {e}")
                        continue
                
                backtest_results[indicator_name] = indicator_results
            
            logger.info(f"Backtesting completed for {len(backtest_results)} indicators")
            return backtest_results
            
        except Exception as e:
            logger.error(f"Error in cross-asset backtesting: {e}")
            return {}
    
    def _generate_cross_asset_signals(self, indicator: Any, asset_data: Dict[str, pd.DataFrame], 
                                    period: str, macro_data: Optional[Dict[str, pd.DataFrame]] = None) -> Optional[Dict[str, pd.Series]]:
        """Generate trading signals from a cross-asset indicator"""
        try:
            # Try different methods to get signals from the indicator
            if hasattr(indicator, 'generate_signals'):
                return indicator.generate_signals(asset_data, macro_data)
            elif hasattr(indicator, 'analyze'):
                result = indicator.analyze(asset_data)
                if hasattr(result, 'signals'):
                    return result.signals
                elif hasattr(result, 'values'):
                    # Extract signals from values DataFrame
                    signals = {}
                    for asset in self.asset_universe:
                        if asset in result.values.columns:
                            signals[asset] = result.values[asset]
                    return signals if signals else None
            elif hasattr(indicator, 'calculate'):
                result = indicator.calculate(asset_data)
                if hasattr(result, 'values'):
                    signals = {}
                    for asset in self.asset_universe:
                        if asset in result.values.columns:
                            signals[asset] = result.values[asset]
                    return signals if signals else None
            
            # Fallback: generate signals based on cross-asset relationships
            if not asset_data:
                return None
            
            signals = {}
            
            # Get common date range
            common_dates = None
            for asset, df in asset_data.items():
                if asset in self.asset_universe:
                    if common_dates is None:
                        common_dates = df.index
                    else:
                        common_dates = common_dates.intersection(df.index)
            
            if common_dates is None or len(common_dates) < 20:
                return None
            
            # Calculate correlation-based signals
            returns_matrix = pd.DataFrame()
            for asset in self.asset_universe:
                if asset in asset_data:
                    returns = asset_data[asset]['close'].pct_change().dropna()
                    returns_matrix[asset] = returns
            
            if len(returns_matrix.columns) < 2:
                return None
            
            # Rolling correlation analysis
            window = 20 if period == 'daily' else (4 if period == 'weekly' else 1)
            
            for asset in returns_matrix.columns:
                # Mean reversion signal based on correlation breakdown
                asset_returns = returns_matrix[asset]
                other_assets = returns_matrix.drop(columns=[asset])
                
                if len(other_assets.columns) > 0:
                    # Calculate rolling correlation with market (first asset as proxy)
                    market_proxy = other_assets.iloc[:, 0]
                    rolling_corr = asset_returns.rolling(window).corr(market_proxy)
                    
                    # Generate signal based on correlation deviation
                    corr_mean = rolling_corr.rolling(60).mean()
                    corr_std = rolling_corr.rolling(60).std()
                    
                    # Z-score based signal
                    z_score = (rolling_corr - corr_mean) / corr_std
                    signal = -z_score.fillna(0)  # Mean reversion
                    
                    # Normalize to [-1, 1]
                    signal = np.tanh(signal / 2)
                    signals[asset] = signal
            
            return signals if signals else None
            
        except Exception as e:
            logger.error(f"Error generating cross-asset signals: {e}")
            return None
    
    def _calculate_cross_asset_performance(self, 
                                        indicator_name: str, 
                                        period: str,
                                        signals: Dict[str, pd.Series], 
                                        actual_returns: Dict[str, pd.Series]) -> CrossAssetSignalPerformance:
        """Calculate signal performance metrics for cross-asset indicators"""
        try:
            performance = CrossAssetSignalPerformance(
                indicator_name=indicator_name, 
                asset_universe=list(signals.keys()), 
                analysis_period=period
            )
            
            if not signals or not actual_returns:
                return performance
            
            # Calculate portfolio performance
            portfolio_returns = []
            all_signals = []
            all_returns = []
            
            # Align signals and returns
            for asset, signal in signals.items():
                if asset in actual_returns:
                    aligned_signal, aligned_return = signal.align(actual_returns[asset], join='inner')
                    
                    if len(aligned_signal) > 0:
                        # Equal weight portfolio
                        weight = 1.0 / len(signals)
                        strategy_return = aligned_signal.shift(1) * aligned_return * weight
                        portfolio_returns.append(strategy_return.dropna())
                        
                        # Collect for accuracy calculation
                        all_signals.extend(aligned_signal.dropna().tolist())
                        all_returns.extend(aligned_return.dropna().tolist())
            
            if not portfolio_returns:
                return performance
            
            # Combine portfolio returns
            combined_returns = pd.concat(portfolio_returns, axis=1).sum(axis=1)
            combined_returns = combined_returns.dropna()
            
            if len(combined_returns) == 0:
                return performance
            
            # Calculate signal accuracy
            if all_signals and all_returns:
                predicted_direction = np.sign(all_signals)
                actual_direction = np.sign(all_returns)
                
                # Remove neutral signals
                non_neutral_mask = np.array(predicted_direction) != 0
                if non_neutral_mask.sum() > 0:
                    performance.total_signals = non_neutral_mask.sum()
                    performance.correct_signals = (np.array(predicted_direction)[non_neutral_mask] == 
                                                 np.array(actual_direction)[non_neutral_mask]).sum()
                    performance.accuracy = performance.correct_signals / performance.total_signals
            
            # Calculate performance metrics
            performance.total_return = combined_returns.sum()
            
            # Annualize based on period
            if period == 'daily':
                annualization_factor = 252
            elif period == 'weekly':
                annualization_factor = 52
            else:  # monthly
                annualization_factor = 12
            
            performance.annualized_return = combined_returns.mean() * annualization_factor
            performance.volatility = combined_returns.std() * np.sqrt(annualization_factor)
            
            if performance.volatility > 0:
                performance.sharpe_ratio = (performance.annualized_return - self.risk_free_rate) / performance.volatility
            
            # Calculate max drawdown
            cumulative_returns = (1 + combined_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            performance.max_drawdown = drawdowns.min()
            
            # Win rate
            winning_periods = combined_returns[combined_returns > 0]
            performance.win_rate = len(winning_periods) / len(combined_returns) if len(combined_returns) > 0 else 0
            
            # Profit factor
            total_wins = winning_periods.sum() if len(winning_periods) > 0 else 0
            total_losses = abs(combined_returns[combined_returns < 0].sum())
            performance.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            return performance
            
        except Exception as e:
            logger.error(f"Error calculating cross-asset performance: {e}")
            return CrossAssetSignalPerformance(indicator_name=indicator_name, asset_universe=[], analysis_period=period)
    
    def _run_cross_asset_backtest(self, 
                                 indicator: Any, 
                                 indicator_name: str, 
                                 period: str,
                                 asset_data: Dict[str, pd.DataFrame],
                                 macro_data: Optional[Dict[str, pd.DataFrame]] = None,
                                 start_date: Optional[datetime] = None,
                                 end_date: Optional[datetime] = None,
                                 initial_capital: float = 100000.0) -> Optional[CrossAssetBacktestResults]:
        """Run backtest for a specific cross-asset indicator"""
        try:
            # Filter data by date range
            filtered_asset_data = {}
            for asset, df in asset_data.items():
                filtered_df = df.copy()
                if start_date:
                    filtered_df = filtered_df[filtered_df.index >= start_date]
                if end_date:
                    filtered_df = filtered_df[filtered_df.index <= end_date]
                
                if len(filtered_df) >= 20:  # Minimum data requirement
                    filtered_asset_data[asset] = filtered_df
            
            if len(filtered_asset_data) < 2:
                return None
            
            # Generate signals
            signals = self._generate_cross_asset_signals(indicator, filtered_asset_data, period, macro_data)
            if not signals:
                return None
            
            # Get common date range
            common_dates = None
            for asset, df in filtered_asset_data.items():
                if common_dates is None:
                    common_dates = df.index
                else:
                    common_dates = common_dates.intersection(df.index)
            
            if len(common_dates) < 20:
                return None
            
            # Initialize backtest result
            result = CrossAssetBacktestResults(
                indicator_name=indicator_name,
                asset_universe=list(signals.keys()),
                analysis_period=period,
                start_date=common_dates[0],
                end_date=common_dates[-1],
                initial_capital=initial_capital
            )
            
            # Calculate returns for each asset
            asset_returns = {}
            for asset in signals.keys():
                if asset in filtered_asset_data:
                    returns = filtered_asset_data[asset]['close'].pct_change().dropna()
                    asset_returns[asset] = returns
            
            if not asset_returns:
                return result
            
            # Run portfolio simulation
            portfolio_values = [initial_capital]
            monthly_returns = []
            
            # Equal weight allocation (can be enhanced with dynamic allocation)
            equal_weight = 1.0 / len(signals)
            
            # Calculate strategy returns
            strategy_returns_list = []
            for asset, signal in signals.items():
                if asset in asset_returns:
                    aligned_signal, aligned_return = signal.align(asset_returns[asset], join='inner')
                    strategy_return = aligned_signal.shift(1) * aligned_return * equal_weight
                    strategy_returns_list.append(strategy_return.dropna())
            
            if not strategy_returns_list:
                return result
            
            # Combine strategy returns
            combined_strategy_returns = pd.concat(strategy_returns_list, axis=1).sum(axis=1)
            combined_strategy_returns = combined_strategy_returns.dropna()
            
            if len(combined_strategy_returns) == 0:
                return result
            
            # Calculate portfolio metrics
            cumulative_returns = (1 + combined_strategy_returns).cumprod()
            result.final_capital = initial_capital * cumulative_returns.iloc[-1]
            result.total_return = (result.final_capital - initial_capital) / initial_capital
            
            # Annualize based on period
            if period == 'daily':
                annualization_factor = 252
            elif period == 'weekly':
                annualization_factor = 52
            else:  # monthly
                annualization_factor = 12
            
            result.annualized_return = combined_strategy_returns.mean() * annualization_factor
            result.volatility = combined_strategy_returns.std() * np.sqrt(annualization_factor)
            
            if result.volatility > 0:
                result.sharpe_ratio = (result.annualized_return - self.risk_free_rate) / result.volatility
                
                # Sortino ratio
                downside_returns = combined_strategy_returns[combined_strategy_returns < 0]
                downside_volatility = downside_returns.std() * np.sqrt(annualization_factor)
                if downside_volatility > 0:
                    result.sortino_ratio = (result.annualized_return - self.risk_free_rate) / downside_volatility
            
            # Calculate drawdown
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            result.max_drawdown = drawdowns.min()
            
            # VaR and CVaR
            result.var_95 = np.percentile(combined_strategy_returns, 5)
            result.cvar_95 = combined_strategy_returns[combined_strategy_returns <= result.var_95].mean()
            
            # Trading statistics
            non_zero_signals = sum(1 for signal_series in signals.values() 
                                 for signal in signal_series if signal != 0)
            result.total_trades = non_zero_signals
            
            winning_returns = combined_strategy_returns[combined_strategy_returns > 0]
            losing_returns = combined_strategy_returns[combined_strategy_returns < 0]
            
            result.winning_trades = len(winning_returns)
            result.losing_trades = len(losing_returns)
            result.win_rate = result.winning_trades / len(combined_strategy_returns) if len(combined_strategy_returns) > 0 else 0
            
            result.avg_win = winning_returns.mean() if len(winning_returns) > 0 else 0
            result.avg_loss = losing_returns.mean() if len(losing_returns) > 0 else 0
            
            total_wins = winning_returns.sum() if len(winning_returns) > 0 else 0
            total_losses = abs(losing_returns.sum()) if len(losing_returns) > 0 else 0
            result.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            # Store detailed results
            if period == 'daily':
                # Resample to monthly for storage
                monthly_rets = combined_strategy_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
                result.monthly_returns = monthly_rets.tolist()
            else:
                result.monthly_returns = combined_strategy_returns.tolist()
            
            return result
            
        except Exception as e:
            logger.error(f"Error running cross-asset backtest: {e}")
            return None
    
    def evaluate_prediction_accuracy(self, 
                                   asset_data: Dict[str, pd.DataFrame],
                                   macro_data: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, List[CrossAssetPredictionAccuracy]]:
        """Evaluate prediction accuracy for all cross-asset indicators"""
        try:
            logger.info("Evaluating cross-asset prediction accuracy...")
            
            accuracy_results = {}
            all_indicators = (self.correlation_indicators + self.arbitrage_indicators + 
                            self.allocation_indicators + self.factor_indicators + 
                            self.macro_indicators)
            
            for indicator_name, indicator in all_indicators:
                indicator_accuracy = []
                
                for period in self.analysis_periods:
                    for horizon in self.prediction_horizons:
                        try:
                            accuracy = self._evaluate_cross_asset_predictions(
                                indicator, indicator_name, period, asset_data, macro_data, horizon
                            )
                            
                            if accuracy:
                                indicator_accuracy.append(accuracy)
                            
                        except Exception as e:
                            logger.error(f"Error evaluating predictions for {indicator_name}, {period}, {horizon}d: {e}")
                            continue
                
                accuracy_results[indicator_name] = indicator_accuracy
            
            logger.info(f"Prediction accuracy evaluation completed for {len(accuracy_results)} indicators")
            return accuracy_results
            
        except Exception as e:
            logger.error(f"Error in prediction accuracy evaluation: {e}")
            return {}
    
    def assess_risk_metrics(self, 
                           asset_data: Dict[str, pd.DataFrame],
                           backtest_results: Dict[str, List[CrossAssetBacktestResults]]) -> Dict[str, List[CrossAssetRiskMetrics]]:
        """Assess comprehensive risk metrics for all cross-asset indicators"""
        try:
            logger.info("Assessing cross-asset risk metrics...")
            
            risk_results = {}
            
            for indicator_name, indicator_backtest_results in backtest_results.items():
                indicator_risks = []
                
                for backtest_result in indicator_backtest_results:
                    try:
                        risk_metrics = self._calculate_cross_asset_risk_metrics(
                            indicator_name, backtest_result, asset_data
                        )
                        
                        if risk_metrics:
                            indicator_risks.append(risk_metrics)
                        
                    except Exception as e:
                        logger.error(f"Error calculating risk metrics for {indicator_name}: {e}")
                        continue
                
                risk_results[indicator_name] = indicator_risks
            
            logger.info(f"Risk assessment completed for {len(risk_results)} indicators")
            return risk_results
            
        except Exception as e:
            logger.error(f"Error in risk assessment: {e}")
            return {}
    
    def perform_comparative_analysis(self, 
                                   signal_performance: Dict[str, List[CrossAssetSignalPerformance]],
                                   backtest_results: Dict[str, List[CrossAssetBacktestResults]],
                                   prediction_accuracy: Dict[str, List[CrossAssetPredictionAccuracy]],
                                   risk_metrics: Dict[str, List[CrossAssetRiskMetrics]]) -> Dict[str, CrossAssetComparativeAnalysis]:
        """Perform comparative analysis across all cross-asset indicators"""
        try:
            logger.info("Performing cross-asset comparative analysis...")
            
            comparative_results = {}
            
            for period in self.analysis_periods:
                analysis = CrossAssetComparativeAnalysis(
                    asset_universe=self.asset_universe,
                    analysis_period=period
                )
                
                # Performance ranking
                performance_scores = []
                for indicator_name, performances in signal_performance.items():
                    period_performances = [
                        p for p in performances 
                        if p.analysis_period == period
                    ]
                    if period_performances:
                        avg_performance = np.mean([p.sharpe_ratio for p in period_performances])
                        performance_scores.append((indicator_name, avg_performance))
                
                analysis.performance_ranking = sorted(performance_scores, key=lambda x: x[1], reverse=True)
                
                # Accuracy ranking
                accuracy_scores = []
                for indicator_name, accuracies in prediction_accuracy.items():
                    period_accuracies = [
                        a for a in accuracies 
                        if a.analysis_period == period
                    ]
                    if period_accuracies:
                        avg_accuracy = np.mean([a.correlation_prediction_accuracy for a in period_accuracies])
                        accuracy_scores.append((indicator_name, avg_accuracy))
                
                analysis.accuracy_ranking = sorted(accuracy_scores, key=lambda x: x[1], reverse=True)
                
                # Risk ranking
                risk_scores = []
                for indicator_name, risks in risk_metrics.items():
                    period_risks = [
                        r for r in risks 
                        if r.analysis_period == period
                    ]
                    if period_risks:
                        avg_risk = np.mean([r.overall_cross_asset_risk for r in period_risks])
                        risk_scores.append((indicator_name, avg_risk))
                
                analysis.risk_ranking = sorted(risk_scores, key=lambda x: x[1])  # Lower risk is better
                
                # Best indicator recommendations
                if analysis.performance_ranking:
                    analysis.best_overall_indicator = analysis.performance_ranking[0][0]
                
                comparative_results[period] = analysis
            
            logger.info(f"Comparative analysis completed for {len(comparative_results)} periods")
            return comparative_results
            
        except Exception as e:
            logger.error(f"Error in comparative analysis: {e}")
            return {}
    
    def _evaluate_cross_asset_predictions(self, 
                                        indicator: Any, 
                                        indicator_name: str, 
                                        period: str,
                                        asset_data: Dict[str, pd.DataFrame],
                                        macro_data: Optional[Dict[str, pd.DataFrame]] = None,
                                        horizon: int = 5) -> Optional[CrossAssetPredictionAccuracy]:
        """Evaluate prediction accuracy for a specific cross-asset indicator and horizon"""
        try:
            accuracy = CrossAssetPredictionAccuracy(
                indicator_name=indicator_name,
                asset_universe=self.asset_universe,
                analysis_period=period,
                prediction_horizon=horizon
            )
            
            # Get minimum required data length
            min_length = max(60, horizon * 2)  # At least 60 periods or 2x horizon
            
            # Check if we have enough data
            sufficient_data = True
            for asset, df in asset_data.items():
                if asset in self.asset_universe and len(df) < min_length:
                    sufficient_data = False
                    break
            
            if not sufficient_data:
                return accuracy
            
            # Rolling prediction evaluation
            predictions = []
            actuals = []
            
            # Use a subset of data for evaluation
            eval_start = min_length
            eval_periods = min(50, len(list(asset_data.values())[0]) - eval_start - horizon)
            
            for i in range(eval_periods):
                try:
                    # Use historical data up to period i
                    historical_data = {}
                    for asset, df in asset_data.items():
                        if asset in self.asset_universe:
                            historical_data[asset] = df.iloc[:eval_start + i]
                    
                    # Generate prediction
                    signals = self._generate_cross_asset_signals(indicator, historical_data, period, macro_data)
                    if signals:
                        # Use average signal as prediction
                        pred = np.mean([s.iloc[-1] if len(s) > 0 and not pd.isna(s.iloc[-1]) else 0 
                                      for s in signals.values()])
                    else:
                        pred = 0
                    
                    # Get actual future return (portfolio average)
                    future_returns = []
                    for asset, df in asset_data.items():
                        if asset in self.asset_universe:
                            current_idx = eval_start + i
                            future_idx = min(current_idx + horizon, len(df) - 1)
                            
                            if future_idx > current_idx:
                                current_price = df['close'].iloc[current_idx]
                                future_price = df['close'].iloc[future_idx]
                                asset_return = (future_price - current_price) / current_price
                                future_returns.append(asset_return)
                    
                    if future_returns:
                        actual_return = np.mean(future_returns)
                        predictions.append(pred)
                        actuals.append(actual_return)
                    
                except Exception as e:
                    continue
            
            if len(predictions) == 0:
                return accuracy
            
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            
            # Direction accuracy
            pred_direction = np.sign(predictions)
            actual_direction = np.sign(actuals)
            
            accuracy.correlation_prediction_accuracy = np.mean(pred_direction == actual_direction)
            
            # Magnitude prediction
            if np.std(predictions) > 0:
                accuracy.mae_returns = np.mean(np.abs(predictions - actuals))
                accuracy.rmse_returns = np.sqrt(np.mean((predictions - actuals) ** 2))
            
            return accuracy
            
        except Exception as e:
            logger.error(f"Error evaluating cross-asset predictions: {e}")
            return CrossAssetPredictionAccuracy(
                indicator_name=indicator_name, asset_universe=self.asset_universe, 
                analysis_period=period, prediction_horizon=horizon
            )
    
    def _calculate_cross_asset_risk_metrics(self, 
                                          indicator_name: str, 
                                          backtest_result: CrossAssetBacktestResults, 
                                          asset_data: Dict[str, pd.DataFrame]) -> Optional[CrossAssetRiskMetrics]:
        """Calculate comprehensive cross-asset risk metrics"""
        try:
            risk_metrics = CrossAssetRiskMetrics(
                indicator_name=indicator_name,
                asset_universe=backtest_result.asset_universe,
                analysis_period=backtest_result.analysis_period
            )
            
            # Concentration risk (based on number of assets)
            num_assets = len(backtest_result.asset_universe)
            risk_metrics.asset_concentration_risk = max(0, 1 - num_assets / 10)  # Penalty for < 10 assets
            
            # Correlation risk (proxy using volatility)
            risk_metrics.correlation_breakdown_risk = min(backtest_result.volatility / 0.2, 1.0)
            
            # Liquidity risk (based on drawdown)
            risk_metrics.market_liquidity_risk = min(abs(backtest_result.max_drawdown) / 0.3, 1.0)
            
            # Overall risk score
            risk_components = [
                risk_metrics.asset_concentration_risk,
                risk_metrics.correlation_breakdown_risk,
                risk_metrics.market_liquidity_risk
            ]
            
            risk_metrics.overall_cross_asset_risk = np.mean([r for r in risk_components if r > 0])
            
            # Diversification effectiveness
            if risk_metrics.overall_cross_asset_risk > 0:
                risk_metrics.diversification_effectiveness = 1 - risk_metrics.overall_cross_asset_risk
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error calculating cross-asset risk metrics: {e}")
            return None
    
    def _generate_report_summary(self, report: CrossAssetEvaluationReport) -> CrossAssetEvaluationReport:
        """Generate summary statistics and recommendations for the report"""
        try:
            # Best performing indicators
            for period in self.analysis_periods:
                if period in report.comparative_analysis:
                    analysis = report.comparative_analysis[period]
                    if analysis.performance_ranking:
                        report.best_performing_indicators[period] = analysis.performance_ranking[0][0]
                    if analysis.risk_ranking:
                        report.worst_performing_indicators[period] = analysis.risk_ranking[-1][0]
            
            # Generate recommendations
            recommendations = []
            warnings = []
            optimizations = []
            
            # Indicator-specific recommendations
            for indicator_name, backtest_results in report.backtest_results.items():
                if backtest_results:
                    avg_sharpe = np.mean([r.sharpe_ratio for r in backtest_results if r.sharpe_ratio is not None])
                    avg_max_dd = np.mean([abs(r.max_drawdown) for r in backtest_results if r.max_drawdown is not None])
                    
                    if avg_sharpe > 1.0:
                        recommendations.append(f"{indicator_name} shows strong cross-asset performance (Sharpe: {avg_sharpe:.2f})")
                    elif avg_sharpe < 0:
                        warnings.append(f"{indicator_name} shows poor cross-asset performance - review methodology")
                    
                    if avg_max_dd > 0.25:
                        warnings.append(f"{indicator_name} has high drawdown risk ({avg_max_dd:.1%}) - enhance diversification")
            
            # Risk warnings
            for indicator_name, risk_results in report.risk_metrics.items():
                if risk_results:
                    avg_risk = np.mean([r.overall_cross_asset_risk for r in risk_results if r.overall_cross_asset_risk is not None])
                    if avg_risk > 0.7:
                        warnings.append(f"{indicator_name} has elevated cross-asset risk levels - review correlations")
            
            # Optimization suggestions
            optimizations.extend([
                "Implement dynamic correlation monitoring for regime changes",
                "Use multi-timeframe analysis for better signal confirmation",
                "Add currency hedging for international asset exposure",
                "Implement volatility targeting for risk management",
                "Use factor-based attribution for performance analysis",
                "Add alternative assets for enhanced diversification",
                "Implement regime-aware asset allocation models",
                "Use machine learning for correlation prediction"
            ])
            
            # Strategy recommendations
            correlation_strategies = [
                "Monitor correlation breakdowns during market stress",
                "Use rolling correlation windows for dynamic adjustment",
                "Implement correlation-based pair trading strategies"
            ]
            
            arbitrage_strategies = [
                "Identify cross-asset arbitrage opportunities",
                "Monitor basis spreads across related instruments",
                "Use statistical arbitrage for mean reversion"
            ]
            
            allocation_strategies = [
                "Implement risk parity allocation across asset classes",
                "Use Black-Litterman model for optimal allocation",
                "Dynamic rebalancing based on volatility regimes"
            ]
            
            factor_strategies = [
                "Use factor-based allocation for systematic exposure",
                "Implement momentum and mean reversion factors",
                "Add quality and value factors for enhanced returns"
            ]
            
            diversification_strategies = [
                "Maximize diversification benefits across asset classes",
                "Use alternative assets for portfolio enhancement",
                "Implement correlation-based diversification strategies"
            ]
            
            report.indicator_recommendations = {
                'high_performance': recommendations,
                'optimization': optimizations
            }
            report.risk_warnings = warnings
            report.optimization_suggestions = optimizations
            report.correlation_strategies = correlation_strategies
            report.arbitrage_strategies = arbitrage_strategies
            report.allocation_strategies = allocation_strategies
            report.factor_strategies = factor_strategies
            report.diversification_strategies = diversification_strategies
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating report summary: {e}")
            return report
    
    def save_evaluation_report(self, report: CrossAssetEvaluationReport, filename: str = None) -> str:
        """Save evaluation report to file"""
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"cross_asset_evaluation_report_{timestamp}.json"
            
            # Convert report to dictionary for JSON serialization
            report_dict = {
                'evaluation_date': report.evaluation_date.isoformat(),
                'asset_universe': report.asset_universe,
                'analysis_periods': report.analysis_periods,
                'evaluation_timeframe': report.evaluation_timeframe,
                'best_performing_indicators': report.best_performing_indicators,
                'indicator_recommendations': report.indicator_recommendations,
                'risk_warnings': report.risk_warnings,
                'optimization_suggestions': report.optimization_suggestions,
                'correlation_strategies': report.correlation_strategies,
                'arbitrage_strategies': report.arbitrage_strategies,
                'allocation_strategies': report.allocation_strategies,
                'factor_strategies': report.factor_strategies,
                'diversification_strategies': report.diversification_strategies
            }
            
            import json
            with open(filename, 'w') as f:
                json.dump(report_dict, f, indent=2)
            
            logger.info(f"Cross-asset evaluation report saved to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error saving evaluation report: {e}")
            return ""

# Example usage
if __name__ == "__main__":
    # Initialize the cross-asset evaluation framework
    framework = CrossAssetEvaluationFramework(
        asset_universe=[
            # Equities
            'SPY', 'QQQ', 'IWM', 'EFA', 'EEM',
            # Bonds  
            'TLT', 'IEF', 'SHY', 'HYG', 'EMB',
            # Commodities
            'GLD', 'SLV', 'USO', 'DBA', 'DBC',
            # Currencies
            'UUP', 'FXE', 'FXY', 'FXA', 'FXC',
            # Crypto
            'BTC-USD', 'ETH-USD'
        ],
        analysis_periods=['daily', 'weekly', 'monthly'],
        evaluation_window=252,
        prediction_horizons=[1, 5, 10, 21, 63]
    )
    
    # Generate sample cross-asset data
    print("Generating sample cross-asset data...")
    np.random.seed(42)
    
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    asset_data = {}
    
    # Generate correlated asset returns
    n_assets = len(framework.asset_universe)
    n_days = len(dates)
    
    # Create correlation structure
    correlation_matrix = np.random.uniform(0.1, 0.8, (n_assets, n_assets))
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
    np.fill_diagonal(correlation_matrix, 1.0)
    
    # Generate correlated returns
    returns = np.random.multivariate_normal(
        mean=np.zeros(n_assets),
        cov=correlation_matrix * 0.02,  # 2% daily volatility
        size=n_days
    )
    
    # Convert to price data
    for i, asset in enumerate(framework.asset_universe):
        prices = 100 * np.cumprod(1 + returns[:, i])
        
        asset_data[asset] = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, len(prices))),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, len(prices)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, len(prices)))),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, len(prices))
        }, index=dates)
    
    # Generate sample macro data
    macro_data = {
        'interest_rates': pd.DataFrame({
            'fed_funds_rate': np.random.uniform(0.01, 0.05, len(dates)),
            '10y_treasury': np.random.uniform(0.02, 0.06, len(dates))
        }, index=dates),
        'economic_indicators': pd.DataFrame({
            'gdp_growth': np.random.uniform(-0.02, 0.04, len(dates)),
            'inflation': np.random.uniform(0.01, 0.05, len(dates)),
            'unemployment': np.random.uniform(0.03, 0.10, len(dates))
        }, index=dates)
    }
    
    # Run comprehensive evaluation
    print("Running comprehensive cross-asset evaluation...")
    evaluation_report = framework.generate_evaluation_report(asset_data, macro_data)
    
    # Display results
    print("\n" + "="*80)
    print("CROSS-ASSET EVALUATION REPORT")
    print("="*80)
    
    print(f"\nEvaluation Date: {evaluation_report.evaluation_date}")
    print(f"Asset Universe: {len(evaluation_report.asset_universe)} assets")
    print(f"Analysis Periods: {', '.join(evaluation_report.analysis_periods)}")
    print(f"Evaluation Timeframe: {evaluation_report.evaluation_timeframe}")
    
    # Performance summary
    print("\n📊 PERFORMANCE SUMMARY:")
    for period, indicator in evaluation_report.best_performing_indicators.items():
        print(f"  • Best {period} indicator: {indicator}")
    
    # Best indicators by strategy type
    print("\n🎯 BEST INDICATORS BY STRATEGY:")
    for period, analysis in evaluation_report.comparative_analysis.items():
        if analysis.best_overall_indicator:
            print(f"  • {period.title()} - Overall: {analysis.best_overall_indicator}")
        if analysis.best_correlation_indicator:
            print(f"  • {period.title()} - Correlation: {analysis.best_correlation_indicator}")
        if analysis.best_arbitrage_indicator:
            print(f"  • {period.title()} - Arbitrage: {analysis.best_arbitrage_indicator}")
    
    # Risk warnings
    if evaluation_report.risk_warnings:
        print("\n⚠️  RISK WARNINGS:")
        for warning in evaluation_report.risk_warnings[:5]:
            print(f"  • {warning}")
    
    # Optimization suggestions
    if evaluation_report.optimization_suggestions:
        print("\n💡 OPTIMIZATION SUGGESTIONS:")
        for suggestion in evaluation_report.optimization_suggestions[:5]:
            print(f"  • {suggestion}")
    
    # Strategy recommendations
    print("\n🚀 STRATEGY RECOMMENDATIONS:")
    if evaluation_report.correlation_strategies:
        print("  Correlation Strategies:")
        for strategy in evaluation_report.correlation_strategies[:3]:
            print(f"    - {strategy}")
    
    if evaluation_report.arbitrage_strategies:
        print("  Arbitrage Strategies:")
        for strategy in evaluation_report.arbitrage_strategies[:3]:
            print(f"    - {strategy}")
    
    if evaluation_report.allocation_strategies:
        print("  Allocation Strategies:")
        for strategy in evaluation_report.allocation_strategies[:3]:
            print(f"    - {strategy}")
    
    # Save comprehensive report
    report_filename = framework.save_evaluation_report(evaluation_report)
    if report_filename:
        print(f"\n💾 Comprehensive evaluation report saved to: {report_filename}")
    
    print("\n" + "="*80)
    print("CROSS-ASSET EVALUATION FRAMEWORK - PRODUCTION READY")
    print("="*80)
    print("Key Features:")
    print("✅ Multi-asset correlation analysis")
    print("✅ Cross-market arbitrage detection")
    print("✅ Dynamic asset allocation optimization")
    print("✅ Risk parity and diversification analysis")
    print("✅ Factor-based attribution and analysis")
    print("✅ Macro-economic integration")
    print("✅ Currency and commodity impact assessment")
    print("✅ Comprehensive backtesting and evaluation")
    print("✅ Real-time performance monitoring")
    print("✅ Advanced risk management")
    print("="*80)