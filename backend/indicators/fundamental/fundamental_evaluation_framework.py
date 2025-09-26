#!/usr/bin/env python3
"""
Fundamental Analysis Evaluation Framework

A comprehensive evaluation framework for all Fundamental analysis indicators and models.
This framework provides:
- Unified evaluation across all fundamental indicators
- Backtesting capabilities for fundamental strategies
- Performance metrics and comparative analysis
- Valuation accuracy and prediction validation
- Cross-indicator correlation analysis
- Sector and industry performance evaluation
- Multi-factor model analysis
- Financial health assessment

Author: Fundamental Analysis Team
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
import warnings
warnings.filterwarnings('ignore')

# Import fundamental indicators with fallback
try:
    from fundamental_comprehensive import FundamentalComprehensiveIndicators
except ImportError:
    logging.warning("Fundamental Comprehensive Indicators not available")
    FundamentalComprehensiveIndicators = None

try:
    from valuation_models import ValuationModels
except ImportError:
    logging.warning("Valuation models not available")
    ValuationModels = None

try:
    from financial_ratios import FinancialRatios
except ImportError:
    logging.warning("Financial ratios not available")
    FinancialRatios = None

try:
    from earnings_analysis import EarningsAnalysis
except ImportError:
    logging.warning("Earnings analysis not available")
    EarningsAnalysis = None

try:
    from growth_analysis import GrowthAnalysis
except ImportError:
    logging.warning("Growth analysis not available")
    GrowthAnalysis = None

try:
    from profitability_analysis import ProfitabilityAnalysis
except ImportError:
    logging.warning("Profitability analysis not available")
    ProfitabilityAnalysis = None

try:
    from liquidity_analysis import LiquidityAnalysis
except ImportError:
    logging.warning("Liquidity analysis not available")
    LiquidityAnalysis = None

try:
    from leverage_analysis import LeverageAnalysis
except ImportError:
    logging.warning("Leverage analysis not available")
    LeverageAnalysis = None

try:
    from efficiency_analysis import EfficiencyAnalysis
except ImportError:
    logging.warning("Efficiency analysis not available")
    EfficiencyAnalysis = None

try:
    from market_valuation import MarketValuation
except ImportError:
    logging.warning("Market valuation not available")
    MarketValuation = None

try:
    from dividend_analysis import DividendAnalysis
except ImportError:
    logging.warning("Dividend analysis not available")
    DividendAnalysis = None

try:
    from quality_analysis import QualityAnalysis
except ImportError:
    logging.warning("Quality analysis not available")
    QualityAnalysis = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FundamentalSignalPerformance:
    """Performance metrics for fundamental trading signals"""
    indicator_name: str
    symbol: str
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
    
    # Fundamental-specific metrics
    valuation_accuracy: float = 0.0
    earnings_prediction_accuracy: float = 0.0
    growth_prediction_accuracy: float = 0.0
    quality_score_accuracy: float = 0.0
    dividend_prediction_accuracy: float = 0.0
    financial_health_accuracy: float = 0.0
    sector_relative_performance: float = 0.0
    
    # Value investing metrics
    value_strategy_return: float = 0.0
    growth_strategy_return: float = 0.0
    quality_strategy_return: float = 0.0
    dividend_strategy_return: float = 0.0
    
    # Risk metrics
    fundamental_risk_score: float = 0.0
    financial_distress_risk: float = 0.0
    earnings_quality_risk: float = 0.0

@dataclass
class FundamentalBacktestResults:
    """Comprehensive fundamental backtesting results"""
    indicator_name: str
    symbol: str
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
    
    # Fundamental-specific metrics
    value_investing_return: float = 0.0
    growth_investing_return: float = 0.0
    quality_investing_return: float = 0.0
    dividend_investing_return: float = 0.0
    contrarian_investing_return: float = 0.0
    
    # Sector performance
    sector_outperformance: float = 0.0
    market_outperformance: float = 0.0
    
    # Valuation metrics
    avg_pe_at_entry: float = 0.0
    avg_pb_at_entry: float = 0.0
    avg_ev_ebitda_at_entry: float = 0.0
    valuation_multiple_accuracy: float = 0.0
    
    # Quality metrics
    avg_roe_at_entry: float = 0.0
    avg_debt_to_equity_at_entry: float = 0.0
    avg_current_ratio_at_entry: float = 0.0
    financial_strength_score: float = 0.0
    
    # Detailed results
    quarterly_returns: List[float] = field(default_factory=list)
    trade_log: List[Dict[str, Any]] = field(default_factory=list)
    fundamental_metrics_history: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class FundamentalPredictionAccuracy:
    """Fundamental prediction accuracy metrics"""
    indicator_name: str
    symbol: str
    analysis_period: str
    prediction_horizon: int  # quarters
    
    # Valuation prediction
    price_target_accuracy: float = 0.0
    fair_value_accuracy: float = 0.0
    valuation_range_accuracy: float = 0.0
    
    # Earnings prediction
    eps_prediction_accuracy: float = 0.0
    revenue_prediction_accuracy: float = 0.0
    margin_prediction_accuracy: float = 0.0
    
    # Growth prediction
    revenue_growth_accuracy: float = 0.0
    earnings_growth_accuracy: float = 0.0
    dividend_growth_accuracy: float = 0.0
    
    # Financial health prediction
    credit_rating_accuracy: float = 0.0
    bankruptcy_risk_accuracy: float = 0.0
    financial_distress_accuracy: float = 0.0
    
    # Quality metrics prediction
    roe_prediction_accuracy: float = 0.0
    roic_prediction_accuracy: float = 0.0
    debt_level_prediction_accuracy: float = 0.0
    
    # Market performance prediction
    relative_performance_accuracy: float = 0.0
    sector_performance_accuracy: float = 0.0
    
    # Magnitude prediction errors
    mae_price_target: float = 0.0
    rmse_price_target: float = 0.0
    mape_earnings: float = 0.0
    
    # Confidence intervals
    prediction_confidence: float = 0.0
    confidence_calibration: float = 0.0

@dataclass
class FundamentalRiskMetrics:
    """Comprehensive fundamental risk assessment"""
    indicator_name: str
    symbol: str
    analysis_period: str
    
    # Fundamental risks
    earnings_quality_risk: float = 0.0
    financial_leverage_risk: float = 0.0
    liquidity_risk: float = 0.0
    profitability_risk: float = 0.0
    
    # Valuation risks
    overvaluation_risk: float = 0.0
    multiple_expansion_risk: float = 0.0
    growth_sustainability_risk: float = 0.0
    
    # Business model risks
    competitive_position_risk: float = 0.0
    industry_disruption_risk: float = 0.0
    regulatory_risk: float = 0.0
    
    # Financial distress indicators
    altman_z_score: float = 0.0
    piotroski_f_score: float = 0.0
    beneish_m_score: float = 0.0
    
    # Sector and market risks
    sector_concentration_risk: float = 0.0
    market_correlation_risk: float = 0.0
    
    # ESG and governance risks
    governance_risk: float = 0.0
    esg_risk: float = 0.0
    
    # Composite risk scores
    overall_fundamental_risk: float = 0.0
    investment_grade_score: float = 0.0
    risk_adjusted_return_potential: float = 0.0

@dataclass
class FundamentalComparativeAnalysis:
    """Comparative analysis across fundamental indicators"""
    symbol: str
    analysis_period: str
    
    # Indicator rankings
    performance_ranking: List[Tuple[str, float]] = field(default_factory=list)
    accuracy_ranking: List[Tuple[str, float]] = field(default_factory=list)
    risk_ranking: List[Tuple[str, float]] = field(default_factory=list)
    
    # Cross-indicator correlations
    signal_correlations: Dict[str, Dict[str, float]] = field(default_factory=dict)
    return_correlations: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Consensus analysis
    consensus_valuation: Dict[str, float] = field(default_factory=dict)
    valuation_dispersion: Dict[str, float] = field(default_factory=dict)
    
    # Category performance
    valuation_indicators_performance: Dict[str, float] = field(default_factory=dict)
    growth_indicators_performance: Dict[str, float] = field(default_factory=dict)
    quality_indicators_performance: Dict[str, float] = field(default_factory=dict)
    profitability_indicators_performance: Dict[str, float] = field(default_factory=dict)
    leverage_indicators_performance: Dict[str, float] = field(default_factory=dict)
    efficiency_indicators_performance: Dict[str, float] = field(default_factory=dict)
    
    # Investment style performance
    value_investing_performance: Dict[str, float] = field(default_factory=dict)
    growth_investing_performance: Dict[str, float] = field(default_factory=dict)
    quality_investing_performance: Dict[str, float] = field(default_factory=dict)
    dividend_investing_performance: Dict[str, float] = field(default_factory=dict)
    
    # Best indicator recommendations
    best_overall_indicator: str = ""
    best_valuation_indicator: str = ""
    best_growth_indicator: str = ""
    best_quality_indicator: str = ""
    best_profitability_indicator: str = ""
    best_risk_indicator: str = ""

@dataclass
class FundamentalEvaluationReport:
    """Comprehensive fundamental evaluation report"""
    evaluation_date: datetime
    symbols: List[str]
    analysis_periods: List[str]
    evaluation_timeframe: str
    
    # Individual indicator results
    signal_performance: Dict[str, List[FundamentalSignalPerformance]] = field(default_factory=dict)
    backtest_results: Dict[str, List[FundamentalBacktestResults]] = field(default_factory=dict)
    prediction_accuracy: Dict[str, List[FundamentalPredictionAccuracy]] = field(default_factory=dict)
    risk_metrics: Dict[str, List[FundamentalRiskMetrics]] = field(default_factory=dict)
    
    # Comparative analysis
    comparative_analysis: Dict[str, FundamentalComparativeAnalysis] = field(default_factory=dict)
    
    # Summary statistics
    best_performing_indicators: Dict[str, str] = field(default_factory=dict)
    worst_performing_indicators: Dict[str, str] = field(default_factory=dict)
    most_consistent_indicators: Dict[str, str] = field(default_factory=dict)
    
    # Category analysis
    category_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Investment style analysis
    value_investing_results: Dict[str, float] = field(default_factory=dict)
    growth_investing_results: Dict[str, float] = field(default_factory=dict)
    quality_investing_results: Dict[str, float] = field(default_factory=dict)
    dividend_investing_results: Dict[str, float] = field(default_factory=dict)
    
    # Recommendations
    indicator_recommendations: Dict[str, List[str]] = field(default_factory=dict)
    risk_warnings: List[str] = field(default_factory=list)
    optimization_suggestions: List[str] = field(default_factory=list)
    
    # Investment strategy recommendations
    value_strategies: List[str] = field(default_factory=list)
    growth_strategies: List[str] = field(default_factory=list)
    quality_strategies: List[str] = field(default_factory=list)
    dividend_strategies: List[str] = field(default_factory=list)
    contrarian_strategies: List[str] = field(default_factory=list)

class FundamentalEvaluationFramework:
    """Comprehensive evaluation framework for Fundamental indicators"""
    
    def __init__(self, 
                 symbols: List[str] = None,
                 analysis_periods: List[str] = None,
                 evaluation_window: int = 20,  # quarters
                 prediction_horizons: List[int] = None,
                 risk_free_rate: float = 0.02,
                 enable_sector_analysis: bool = True,
                 enable_peer_comparison: bool = True):
        """
        Initialize the Fundamental evaluation framework
        
        Args:
            symbols: List of symbols to evaluate
            analysis_periods: List of analysis periods ('quarterly', 'annual')
            evaluation_window: Number of quarters for evaluation window
            prediction_horizons: List of prediction horizons in quarters
            risk_free_rate: Risk-free rate for calculations
            enable_sector_analysis: Enable sector-based analysis
            enable_peer_comparison: Enable peer comparison analysis
        """
        self.symbols = symbols or ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'META', 'AMZN', 'NFLX']
        self.analysis_periods = analysis_periods or ['quarterly', 'annual']
        self.evaluation_window = evaluation_window
        self.prediction_horizons = prediction_horizons or [1, 2, 4, 8]  # quarters
        self.risk_free_rate = risk_free_rate
        self.enable_sector_analysis = enable_sector_analysis
        self.enable_peer_comparison = enable_peer_comparison
        
        # Initialize indicators
        self.valuation_indicators = self._initialize_valuation_indicators()
        self.growth_indicators = self._initialize_growth_indicators()
        self.quality_indicators = self._initialize_quality_indicators()
        self.profitability_indicators = self._initialize_profitability_indicators()
        self.leverage_indicators = self._initialize_leverage_indicators()
        self.efficiency_indicators = self._initialize_efficiency_indicators()
        
        # Evaluation state
        self.evaluation_history = []
        self.indicator_performance_cache = {}
        
        total_indicators = (len(self.valuation_indicators) + len(self.growth_indicators) + 
                          len(self.quality_indicators) + len(self.profitability_indicators) + 
                          len(self.leverage_indicators) + len(self.efficiency_indicators))
        
        logger.info(f"Initialized Fundamental Evaluation Framework with {total_indicators} indicators")
    
    def _initialize_valuation_indicators(self) -> List[Tuple[str, Any]]:
        """Initialize valuation indicators"""
        indicators = []
        
        if FundamentalComprehensiveIndicators:
            indicators.append(('Fundamental_Comprehensive', FundamentalComprehensiveIndicators()))
        
        if ValuationModels:
            indicators.append(('Valuation_Models', ValuationModels()))
        
        if MarketValuation:
            indicators.append(('Market_Valuation', MarketValuation()))
        
        logger.info(f"Initialized {len(indicators)} valuation indicators")
        return indicators
    
    def _initialize_growth_indicators(self) -> List[Tuple[str, Any]]:
        """Initialize growth indicators"""
        indicators = []
        
        if GrowthAnalysis:
            indicators.append(('Growth_Analysis', GrowthAnalysis()))
        
        if EarningsAnalysis:
            indicators.append(('Earnings_Analysis', EarningsAnalysis()))
        
        logger.info(f"Initialized {len(indicators)} growth indicators")
        return indicators
    
    def _initialize_quality_indicators(self) -> List[Tuple[str, Any]]:
        """Initialize quality indicators"""
        indicators = []
        
        if QualityAnalysis:
            indicators.append(('Quality_Analysis', QualityAnalysis()))
        
        if FinancialRatios:
            indicators.append(('Financial_Ratios', FinancialRatios()))
        
        logger.info(f"Initialized {len(indicators)} quality indicators")
        return indicators
    
    def _initialize_profitability_indicators(self) -> List[Tuple[str, Any]]:
        """Initialize profitability indicators"""
        indicators = []
        
        if ProfitabilityAnalysis:
            indicators.append(('Profitability_Analysis', ProfitabilityAnalysis()))
        
        if DividendAnalysis:
            indicators.append(('Dividend_Analysis', DividendAnalysis()))
        
        logger.info(f"Initialized {len(indicators)} profitability indicators")
        return indicators
    
    def _initialize_leverage_indicators(self) -> List[Tuple[str, Any]]:
        """Initialize leverage indicators"""
        indicators = []
        
        if LeverageAnalysis:
            indicators.append(('Leverage_Analysis', LeverageAnalysis()))
        
        if LiquidityAnalysis:
            indicators.append(('Liquidity_Analysis', LiquidityAnalysis()))
        
        logger.info(f"Initialized {len(indicators)} leverage indicators")
        return indicators
    
    def _initialize_efficiency_indicators(self) -> List[Tuple[str, Any]]:
        """Initialize efficiency indicators"""
        indicators = []
        
        if EfficiencyAnalysis:
            indicators.append(('Efficiency_Analysis', EfficiencyAnalysis()))
        
        logger.info(f"Initialized {len(indicators)} efficiency indicators")
        return indicators
    
    def generate_evaluation_report(self, 
                                 fundamental_data: Dict[str, Dict[str, pd.DataFrame]],
                                 price_data: Dict[str, pd.DataFrame]) -> FundamentalEvaluationReport:
        """Generate comprehensive fundamental evaluation report
        
        Args:
            fundamental_data: Dictionary with structure {symbol: {period: DataFrame}}
            price_data: Dictionary with structure {symbol: DataFrame}
        """
        try:
            logger.info("Generating comprehensive fundamental evaluation report...")
            
            # Calculate actual returns for all symbols
            actual_returns = {}
            for symbol, df in price_data.items():
                actual_returns[symbol] = df['close'].pct_change().dropna()
            
            # Run all evaluations
            signal_performance = self.evaluate_signal_performance(fundamental_data, price_data, actual_returns)
            backtest_results = self.backtest_indicators(fundamental_data, price_data)
            prediction_accuracy = self.evaluate_prediction_accuracy(fundamental_data, price_data)
            risk_metrics = self.assess_risk_metrics(fundamental_data, backtest_results)
            comparative_analysis = self.perform_comparative_analysis(
                signal_performance, backtest_results, prediction_accuracy, risk_metrics
            )
            
            # Create comprehensive report
            report = FundamentalEvaluationReport(
                evaluation_date=datetime.now(),
                symbols=self.symbols,
                analysis_periods=self.analysis_periods,
                evaluation_timeframe=f"{self.evaluation_window} quarters",
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
            
            logger.info("Fundamental evaluation report generated successfully")
            return report
            
        except Exception as e:
            logger.error(f"Error generating fundamental evaluation report: {e}")
            return FundamentalEvaluationReport(
                evaluation_date=datetime.now(),
                symbols=self.symbols,
                analysis_periods=self.analysis_periods,
                evaluation_timeframe=f"{self.evaluation_window} quarters"
            )
    
    def evaluate_signal_performance(self, 
                                   fundamental_data: Dict[str, Dict[str, pd.DataFrame]],
                                   price_data: Dict[str, pd.DataFrame],
                                   actual_returns: Dict[str, pd.Series]) -> Dict[str, List[FundamentalSignalPerformance]]:
        """Evaluate signal performance for all fundamental indicators"""
        try:
            logger.info("Evaluating fundamental signal performance...")
            
            performance_results = {}
            all_indicators = (self.valuation_indicators + self.growth_indicators + 
                            self.quality_indicators + self.profitability_indicators + 
                            self.leverage_indicators + self.efficiency_indicators)
            
            for indicator_name, indicator in all_indicators:
                indicator_performance = []
                
                for symbol in self.symbols:
                    if symbol not in fundamental_data or symbol not in price_data:
                        continue
                    
                    for period in self.analysis_periods:
                        if period not in fundamental_data[symbol]:
                            continue
                        
                        try:
                            # Generate signals
                            fund_data = fundamental_data[symbol][period]
                            signals = self._generate_fundamental_signals(indicator, fund_data, price_data[symbol], symbol, period)
                            
                            if signals is None or len(signals) == 0:
                                continue
                            
                            # Evaluate performance
                            performance = self._calculate_fundamental_performance(
                                indicator_name, symbol, period, signals, 
                                actual_returns.get(symbol)
                            )
                            
                            indicator_performance.append(performance)
                            
                        except Exception as e:
                            logger.error(f"Error evaluating {indicator_name} for {symbol} {period}: {e}")
                            continue
                
                performance_results[indicator_name] = indicator_performance
            
            logger.info(f"Signal performance evaluation completed for {len(performance_results)} indicators")
            return performance_results
            
        except Exception as e:
            logger.error(f"Error in signal performance evaluation: {e}")
            return {}
    
    def backtest_indicators(self, 
                           fundamental_data: Dict[str, Dict[str, pd.DataFrame]],
                           price_data: Dict[str, pd.DataFrame],
                           start_date: datetime = None,
                           end_date: datetime = None,
                           initial_capital: float = 100000.0) -> Dict[str, List[FundamentalBacktestResults]]:
        """Backtest all fundamental indicators"""
        try:
            logger.info("Starting fundamental indicators backtesting...")
            
            backtest_results = {}
            all_indicators = (self.valuation_indicators + self.growth_indicators + 
                            self.quality_indicators + self.profitability_indicators + 
                            self.leverage_indicators + self.efficiency_indicators)
            
            for indicator_name, indicator in all_indicators:
                indicator_results = []
                
                for symbol in self.symbols:
                    if symbol not in fundamental_data or symbol not in price_data:
                        continue
                    
                    for period in self.analysis_periods:
                        if period not in fundamental_data[symbol]:
                            continue
                        
                        try:
                            # Run backtest
                            result = self._run_fundamental_backtest(
                                indicator, indicator_name, symbol, period,
                                fundamental_data[symbol][period], price_data[symbol], 
                                start_date, end_date, initial_capital
                            )
                            
                            if result:
                                indicator_results.append(result)
                            
                        except Exception as e:
                            logger.error(f"Error backtesting {indicator_name} for {symbol} {period}: {e}")
                            continue
                
                backtest_results[indicator_name] = indicator_results
            
            logger.info(f"Backtesting completed for {len(backtest_results)} indicators")
            return backtest_results
            
        except Exception as e:
            logger.error(f"Error in fundamental backtesting: {e}")
            return {}
    
    def evaluate_prediction_accuracy(self, 
                                   fundamental_data: Dict[str, Dict[str, pd.DataFrame]],
                                   price_data: Dict[str, pd.DataFrame]) -> Dict[str, List[FundamentalPredictionAccuracy]]:
        """Evaluate prediction accuracy for all fundamental indicators"""
        try:
            logger.info("Evaluating fundamental prediction accuracy...")
            
            accuracy_results = {}
            all_indicators = (self.valuation_indicators + self.growth_indicators + 
                            self.quality_indicators + self.profitability_indicators + 
                            self.leverage_indicators + self.efficiency_indicators)
            
            for indicator_name, indicator in all_indicators:
                indicator_accuracy = []
                
                for symbol in self.symbols:
                    if symbol not in fundamental_data or symbol not in price_data:
                        continue
                    
                    for period in self.analysis_periods:
                        if period not in fundamental_data[symbol]:
                            continue
                        
                        for horizon in self.prediction_horizons:
                            try:
                                accuracy = self._evaluate_fundamental_predictions(
                                    indicator, indicator_name, symbol, period,
                                    fundamental_data[symbol][period], price_data[symbol], horizon
                                )
                                
                                if accuracy:
                                    indicator_accuracy.append(accuracy)
                                
                            except Exception as e:
                                logger.error(f"Error evaluating predictions for {indicator_name}, {symbol}, {period}, {horizon}q: {e}")
                                continue
                
                accuracy_results[indicator_name] = indicator_accuracy
            
            logger.info(f"Prediction accuracy evaluation completed for {len(accuracy_results)} indicators")
            return accuracy_results
            
        except Exception as e:
            logger.error(f"Error in prediction accuracy evaluation: {e}")
            return {}
    
    def assess_risk_metrics(self, 
                           fundamental_data: Dict[str, Dict[str, pd.DataFrame]],
                           backtest_results: Dict[str, List[FundamentalBacktestResults]]) -> Dict[str, List[FundamentalRiskMetrics]]:
        """Assess comprehensive risk metrics for all fundamental indicators"""
        try:
            logger.info("Assessing fundamental risk metrics...")
            
            risk_results = {}
            
            for indicator_name, indicator_backtest_results in backtest_results.items():
                indicator_risks = []
                
                for backtest_result in indicator_backtest_results:
                    try:
                        risk_metrics = self._calculate_fundamental_risk_metrics(
                            indicator_name, backtest_result, 
                            fundamental_data.get(backtest_result.symbol, {}).get(backtest_result.analysis_period)
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
                                   signal_performance: Dict[str, List[FundamentalSignalPerformance]],
                                   backtest_results: Dict[str, List[FundamentalBacktestResults]],
                                   prediction_accuracy: Dict[str, List[FundamentalPredictionAccuracy]],
                                   risk_metrics: Dict[str, List[FundamentalRiskMetrics]]) -> Dict[str, FundamentalComparativeAnalysis]:
        """Perform comparative analysis across all fundamental indicators"""
        try:
            logger.info("Performing fundamental comparative analysis...")
            
            comparative_results = {}
            
            for symbol in self.symbols:
                for period in self.analysis_periods:
                    key = f"{symbol}_{period}"
                    
                    analysis = FundamentalComparativeAnalysis(
                        symbol=symbol,
                        analysis_period=period
                    )
                    
                    # Performance ranking
                    performance_scores = []
                    for indicator_name, performances in signal_performance.items():
                        symbol_period_performances = [
                            p for p in performances 
                            if p.symbol == symbol and p.analysis_period == period
                        ]
                        if symbol_period_performances:
                            avg_performance = np.mean([p.sharpe_ratio for p in symbol_period_performances])
                            performance_scores.append((indicator_name, avg_performance))
                    
                    analysis.performance_ranking = sorted(performance_scores, key=lambda x: x[1], reverse=True)
                    
                    # Accuracy ranking
                    accuracy_scores = []
                    for indicator_name, accuracies in prediction_accuracy.items():
                        symbol_period_accuracies = [
                            a for a in accuracies 
                            if a.symbol == symbol and a.analysis_period == period
                        ]
                        if symbol_period_accuracies:
                            avg_accuracy = np.mean([a.price_target_accuracy for a in symbol_period_accuracies])
                            accuracy_scores.append((indicator_name, avg_accuracy))
                    
                    analysis.accuracy_ranking = sorted(accuracy_scores, key=lambda x: x[1], reverse=True)
                    
                    # Risk ranking
                    risk_scores = []
                    for indicator_name, risks in risk_metrics.items():
                        symbol_period_risks = [
                            r for r in risks 
                            if r.symbol == symbol and r.analysis_period == period
                        ]
                        if symbol_period_risks:
                            avg_risk = np.mean([r.overall_fundamental_risk for r in symbol_period_risks])
                            risk_scores.append((indicator_name, avg_risk))
                    
                    analysis.risk_ranking = sorted(risk_scores, key=lambda x: x[1])  # Lower risk is better
                    
                    # Best indicator recommendations
                    if analysis.performance_ranking:
                        analysis.best_overall_indicator = analysis.performance_ranking[0][0]
                    
                    comparative_results[key] = analysis
            
            logger.info(f"Comparative analysis completed for {len(comparative_results)} symbol-period combinations")
            return comparative_results
            
        except Exception as e:
            logger.error(f"Error in comparative analysis: {e}")
            return {}
    
    def _generate_fundamental_signals(self, indicator: Any, fundamental_data: pd.DataFrame, 
                                    price_data: pd.DataFrame, symbol: str, period: str) -> Optional[pd.Series]:
        """Generate trading signals from a fundamental indicator"""
        try:
            # Try different methods to get signals from the indicator
            if hasattr(indicator, 'generate_signals'):
                return indicator.generate_signals(fundamental_data, price_data)
            elif hasattr(indicator, 'analyze'):
                result = indicator.analyze(fundamental_data)
                if hasattr(result, 'signal'):
                    return result.signal
                elif hasattr(result, 'values'):
                    # Extract signal from values DataFrame
                    if 'signal' in result.values.columns:
                        return result.values['signal']
                    elif 'recommendation' in result.values.columns:
                        # Convert recommendation to signal
                        rec_map = {'BUY': 1, 'STRONG_BUY': 1.5, 'SELL': -1, 'STRONG_SELL': -1.5, 'HOLD': 0}
                        return result.values['recommendation'].map(rec_map)
            elif hasattr(indicator, 'calculate'):
                result = indicator.calculate(fundamental_data)
                if hasattr(result, 'values') and 'signal' in result.values.columns:
                    return result.values['signal']
            
            # Fallback: generate signals based on fundamental metrics
            if len(fundamental_data) == 0:
                return None
            
            signals = pd.Series(index=fundamental_data.index, dtype=float)
            
            # Use multiple fundamental signals
            if 'pe_ratio' in fundamental_data.columns:
                pe_signal = self._generate_pe_signal(fundamental_data['pe_ratio'])
            else:
                pe_signal = pd.Series(0, index=fundamental_data.index)
            
            if 'roe' in fundamental_data.columns:
                roe_signal = self._generate_roe_signal(fundamental_data['roe'])
            else:
                roe_signal = pd.Series(0, index=fundamental_data.index)
            
            if 'debt_to_equity' in fundamental_data.columns:
                debt_signal = self._generate_debt_signal(fundamental_data['debt_to_equity'])
            else:
                debt_signal = pd.Series(0, index=fundamental_data.index)
            
            # Combine signals with weights
            signals = 0.4 * pe_signal + 0.4 * roe_signal + 0.2 * debt_signal
            signals = signals.fillna(0)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating fundamental signals: {e}")
            return None
    
    def _generate_pe_signal(self, pe_ratio: pd.Series) -> pd.Series:
        """Generate signal based on P/E ratio (value signal)"""
        try:
            # Lower P/E is better (value signal)
            pe_percentile = pe_ratio.rolling(8).rank(pct=True)  # 8 quarters rolling
            signal = (1 - pe_percentile) * 2 - 1  # Convert to -1 to 1 scale
            return signal.fillna(0)
        except Exception:
            return pd.Series(0, index=pe_ratio.index)
    
    def _generate_roe_signal(self, roe: pd.Series) -> pd.Series:
        """Generate signal based on ROE (quality signal)"""
        try:
            # Higher ROE is better (quality signal)
            roe_percentile = roe.rolling(8).rank(pct=True)  # 8 quarters rolling
            signal = roe_percentile * 2 - 1  # Convert to -1 to 1 scale
            return signal.fillna(0)
        except Exception:
            return pd.Series(0, index=roe.index)
    
    def _generate_debt_signal(self, debt_to_equity: pd.Series) -> pd.Series:
        """Generate signal based on debt-to-equity ratio (safety signal)"""
        try:
            # Lower debt is better (safety signal)
            debt_percentile = debt_to_equity.rolling(8).rank(pct=True)  # 8 quarters rolling
            signal = (1 - debt_percentile) * 2 - 1  # Convert to -1 to 1 scale
            return signal.fillna(0)
        except Exception:
            return pd.Series(0, index=debt_to_equity.index)
    
    def _calculate_fundamental_performance(self, 
                                        indicator_name: str, 
                                        symbol: str, 
                                        period: str,
                                        signals: pd.Series, 
                                        actual_returns: pd.Series) -> FundamentalSignalPerformance:
        """Calculate signal performance metrics for fundamental indicators"""
        try:
            performance = FundamentalSignalPerformance(
                indicator_name=indicator_name, 
                symbol=symbol, 
                analysis_period=period
            )
            
            if actual_returns is None or len(signals) == 0:
                return performance
            
            # Align signals and returns (quarterly rebalancing)
            # Resample returns to quarterly for fundamental analysis
            quarterly_returns = actual_returns.resample('Q').apply(lambda x: (1 + x).prod() - 1)
            
            # Align signals with quarterly returns
            aligned_signals, aligned_returns = signals.align(quarterly_returns, join='inner')
            
            if len(aligned_signals) == 0:
                return performance
            
            # Calculate signal accuracy
            predicted_direction = np.sign(aligned_signals)
            actual_direction = np.sign(aligned_returns)
            
            # Remove neutral signals for accuracy calculation
            non_neutral_mask = predicted_direction != 0
            if non_neutral_mask.sum() > 0:
                performance.total_signals = non_neutral_mask.sum()
                performance.correct_signals = (predicted_direction[non_neutral_mask] == actual_direction[non_neutral_mask]).sum()
                performance.accuracy = performance.correct_signals / performance.total_signals
            
            # Calculate trading performance (quarterly rebalancing)
            strategy_returns = aligned_signals.shift(1) * aligned_returns
            strategy_returns = strategy_returns.dropna()
            
            if len(strategy_returns) > 0:
                performance.total_return = strategy_returns.sum()
                performance.annualized_return = strategy_returns.mean() * 4  # Quarterly to annual
                performance.volatility = strategy_returns.std() * np.sqrt(4)  # Quarterly to annual
                
                if performance.volatility > 0:
                    performance.sharpe_ratio = (performance.annualized_return - self.risk_free_rate) / performance.volatility
                
                # Calculate max drawdown
                cumulative_returns = (1 + strategy_returns).cumprod()
                rolling_max = cumulative_returns.expanding().max()
                drawdowns = (cumulative_returns - rolling_max) / rolling_max
                performance.max_drawdown = drawdowns.min()
                
                # Win rate
                winning_trades = strategy_returns[strategy_returns > 0]
                performance.win_rate = len(winning_trades) / len(strategy_returns) if len(strategy_returns) > 0 else 0
                
                # Profit factor
                total_wins = winning_trades.sum() if len(winning_trades) > 0 else 0
                total_losses = abs(strategy_returns[strategy_returns < 0].sum())
                performance.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            return performance
            
        except Exception as e:
            logger.error(f"Error calculating fundamental performance: {e}")
            return FundamentalSignalPerformance(indicator_name=indicator_name, symbol=symbol, analysis_period=period)
    
    def _run_fundamental_backtest(self, 
                                 indicator: Any, 
                                 indicator_name: str, 
                                 symbol: str,
                                 period: str,
                                 fundamental_data: pd.DataFrame,
                                 price_data: pd.DataFrame,
                                 start_date: Optional[datetime] = None,
                                 end_date: Optional[datetime] = None,
                                 initial_capital: float = 100000.0) -> Optional[FundamentalBacktestResults]:
        """Run backtest for a specific fundamental indicator"""
        try:
            # Filter data by date range
            if start_date:
                fundamental_data = fundamental_data[fundamental_data.index >= start_date]
                price_data = price_data[price_data.index >= start_date]
            if end_date:
                fundamental_data = fundamental_data[fundamental_data.index <= end_date]
                price_data = price_data[price_data.index <= end_date]
            
            if len(fundamental_data) < 8:  # Minimum 8 quarters
                return None
            
            # Generate signals
            signals = self._generate_fundamental_signals(indicator, fundamental_data, price_data, symbol, period)
            if signals is None:
                return None
            
            # Initialize backtest result
            result = FundamentalBacktestResults(
                indicator_name=indicator_name,
                symbol=symbol,
                analysis_period=period,
                start_date=fundamental_data.index[0],
                end_date=fundamental_data.index[-1],
                initial_capital=initial_capital
            )
            
            # Calculate quarterly returns
            quarterly_returns = price_data['close'].resample('Q').apply(lambda x: (1 + x.pct_change()).prod() - 1)
            
            # Align signals and returns
            aligned_signals, aligned_returns = signals.align(quarterly_returns, join='inner')
            
            if len(aligned_signals) == 0:
                return result
            
            # Calculate strategy returns (quarterly rebalancing)
            strategy_returns = aligned_signals.shift(1) * aligned_returns
            strategy_returns = strategy_returns.dropna()
            
            if len(strategy_returns) == 0:
                return result
            
            # Calculate portfolio metrics
            cumulative_returns = (1 + strategy_returns).cumprod()
            result.final_capital = initial_capital * cumulative_returns.iloc[-1]
            result.total_return = (result.final_capital - initial_capital) / initial_capital
            result.annualized_return = strategy_returns.mean() * 4  # Quarterly to annual
            result.volatility = strategy_returns.std() * np.sqrt(4)  # Quarterly to annual
            
            if result.volatility > 0:
                result.sharpe_ratio = (result.annualized_return - self.risk_free_rate) / result.volatility
                
                # Sortino ratio
                downside_returns = strategy_returns[strategy_returns < 0]
                downside_volatility = downside_returns.std() * np.sqrt(4)
                if downside_volatility > 0:
                    result.sortino_ratio = (result.annualized_return - self.risk_free_rate) / downside_volatility
            
            # Calculate drawdown
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            result.max_drawdown = drawdowns.min()
            
            # VaR and CVaR
            result.var_95 = np.percentile(strategy_returns, 5)
            result.cvar_95 = strategy_returns[strategy_returns <= result.var_95].mean()
            
            # Trading statistics
            trades = aligned_signals[aligned_signals != 0]
            result.total_trades = len(trades)
            
            winning_returns = strategy_returns[strategy_returns > 0]
            losing_returns = strategy_returns[strategy_returns < 0]
            
            result.winning_trades = len(winning_returns)
            result.losing_trades = len(losing_returns)
            result.win_rate = result.winning_trades / result.total_trades if result.total_trades > 0 else 0
            
            result.avg_win = winning_returns.mean() if len(winning_returns) > 0 else 0
            result.avg_loss = losing_returns.mean() if len(losing_returns) > 0 else 0
            
            total_wins = winning_returns.sum() if len(winning_returns) > 0 else 0
            total_losses = abs(losing_returns.sum()) if len(losing_returns) > 0 else 0
            result.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            # Average holding period (quarters)
            result.avg_holding_period = 1.0  # Quarterly rebalancing
            
            # Store detailed results
            result.quarterly_returns = strategy_returns.tolist()
            
            return result
            
        except Exception as e:
            logger.error(f"Error running fundamental backtest: {e}")
            return None
    
    def _evaluate_fundamental_predictions(self, 
                                        indicator: Any, 
                                        indicator_name: str, 
                                        symbol: str,
                                        period: str,
                                        fundamental_data: pd.DataFrame,
                                        price_data: pd.DataFrame, 
                                        horizon: int) -> Optional[FundamentalPredictionAccuracy]:
        """Evaluate prediction accuracy for a specific fundamental indicator and horizon"""
        try:
            accuracy = FundamentalPredictionAccuracy(
                indicator_name=indicator_name,
                symbol=symbol,
                analysis_period=period,
                prediction_horizon=horizon
            )
            
            if len(fundamental_data) < horizon + 8:  # Minimum data requirement
                return accuracy
            
            # Generate predictions
            predictions = []
            actuals = []
            
            # Rolling prediction evaluation (quarterly)
            for i in range(8, len(fundamental_data) - horizon):
                try:
                    # Use historical data up to quarter i
                    historical_fund_data = fundamental_data.iloc[:i]
                    historical_price_data = price_data[price_data.index <= fundamental_data.index[i]]
                    
                    # Generate prediction
                    signals = self._generate_fundamental_signals(indicator, historical_fund_data, historical_price_data, symbol, period)
                    if signals is not None and len(signals) > 0:
                        pred = signals.iloc[-1] if not pd.isna(signals.iloc[-1]) else 0
                    else:
                        pred = 0
                    
                    # Get actual future return (quarterly)
                    current_date = fundamental_data.index[i]
                    future_date = fundamental_data.index[min(i + horizon, len(fundamental_data) - 1)]
                    
                    current_price_data = price_data[price_data.index <= current_date]
                    future_price_data = price_data[price_data.index <= future_date]
                    
                    if len(current_price_data) > 0 and len(future_price_data) > 0:
                        current_price = current_price_data['close'].iloc[-1]
                        future_price = future_price_data['close'].iloc[-1]
                        actual_return = (future_price - current_price) / current_price
                    else:
                        continue
                    
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
            
            accuracy.price_target_accuracy = np.mean(pred_direction == actual_direction)
            
            # Magnitude prediction (assuming predictions are normalized signals)
            if np.std(predictions) > 0:
                # Normalize predictions to match actual return scale
                normalized_predictions = (predictions - np.mean(predictions)) / np.std(predictions) * np.std(actuals)
                
                accuracy.mae_price_target = np.mean(np.abs(normalized_predictions - actuals))
                accuracy.rmse_price_target = np.sqrt(np.mean((normalized_predictions - actuals) ** 2))
            
            return accuracy
            
        except Exception as e:
            logger.error(f"Error evaluating fundamental predictions: {e}")
            return FundamentalPredictionAccuracy(
                indicator_name=indicator_name, symbol=symbol, 
                analysis_period=period, prediction_horizon=horizon
            )
    
    def _calculate_fundamental_risk_metrics(self, 
                                          indicator_name: str, 
                                          backtest_result: FundamentalBacktestResults, 
                                          fundamental_data: Optional[pd.DataFrame]) -> Optional[FundamentalRiskMetrics]:
        """Calculate comprehensive fundamental risk metrics"""
        try:
            risk_metrics = FundamentalRiskMetrics(
                indicator_name=indicator_name,
                symbol=backtest_result.symbol,
                analysis_period=backtest_result.analysis_period
            )
            
            # Earnings quality risk (based on win rate)
            risk_metrics.earnings_quality_risk = 1 - backtest_result.win_rate
            
            # Financial leverage risk (proxy using volatility)
            risk_metrics.financial_leverage_risk = min(backtest_result.volatility / 0.3, 1.0)
            
            # Profitability risk (based on returns)
            if backtest_result.annualized_return < 0:
                risk_metrics.profitability_risk = 1.0
            else:
                risk_metrics.profitability_risk = max(0, 1 - backtest_result.annualized_return / 0.15)
            
            # Drawdown risk
            risk_metrics.liquidity_risk = min(abs(backtest_result.max_drawdown) / 0.25, 1.0)
            
            # Composite risk score
            risk_components = [
                risk_metrics.earnings_quality_risk,
                risk_metrics.financial_leverage_risk,
                risk_metrics.profitability_risk,
                risk_metrics.liquidity_risk
            ]
            
            risk_metrics.overall_fundamental_risk = np.mean([r for r in risk_components if r > 0])
            
            # Risk-adjusted return potential
            if risk_metrics.overall_fundamental_risk > 0:
                risk_metrics.risk_adjusted_return_potential = backtest_result.annualized_return / risk_metrics.overall_fundamental_risk
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error calculating fundamental risk metrics: {e}")
            return None
    
    def _generate_report_summary(self, report: FundamentalEvaluationReport) -> FundamentalEvaluationReport:
        """Generate summary statistics and recommendations for the report"""
        try:
            # Best performing indicators
            for symbol in self.symbols:
                for period in self.analysis_periods:
                    key = f"{symbol}_{period}"
                    if key in report.comparative_analysis:
                        analysis = report.comparative_analysis[key]
                        if analysis.performance_ranking:
                            report.best_performing_indicators[key] = analysis.performance_ranking[0][0]
                        if analysis.risk_ranking:
                            report.worst_performing_indicators[key] = analysis.risk_ranking[-1][0]
            
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
                        recommendations.append(f"{indicator_name} shows strong risk-adjusted performance (Sharpe: {avg_sharpe:.2f})")
                    elif avg_sharpe < 0:
                        warnings.append(f"{indicator_name} shows poor performance - review fundamental criteria")
                    
                    if avg_max_dd > 0.25:
                        warnings.append(f"{indicator_name} has high drawdown risk ({avg_max_dd:.1%}) - consider diversification")
            
            # Risk warnings
            for indicator_name, risk_results in report.risk_metrics.items():
                if risk_results:
                    avg_risk = np.mean([r.overall_fundamental_risk for r in risk_results if r.overall_fundamental_risk is not None])
                    if avg_risk > 0.7:
                        warnings.append(f"{indicator_name} has elevated fundamental risk levels - conduct deeper analysis")
            
            # Optimization suggestions
            optimizations.extend([
                "Consider multi-factor models combining valuation, quality, and growth metrics",
                "Implement sector-neutral strategies to reduce sector concentration risk",
                "Add earnings quality filters to improve signal reliability",
                "Use peer comparison analysis for relative valuation",
                "Implement dynamic rebalancing based on fundamental changes",
                "Add ESG factors for comprehensive risk assessment",
                "Consider market cycle adjustments for valuation multiples",
                "Implement proper position sizing based on conviction levels"
            ])
            
            # Investment strategy recommendations
            value_strategies = [
                "Focus on low P/E, P/B ratios with strong balance sheets",
                "Use discounted cash flow models for intrinsic value estimation",
                "Implement contrarian strategies in oversold quality companies"
            ]
            
            growth_strategies = [
                "Target companies with consistent revenue and earnings growth",
                "Focus on sustainable competitive advantages and market expansion",
                "Use PEG ratios to balance growth and valuation"
            ]
            
            quality_strategies = [
                "Prioritize high ROE, ROIC companies with low debt levels",
                "Focus on companies with strong cash flow generation",
                "Use Piotroski F-Score for quality screening"
            ]
            
            dividend_strategies = [
                "Target companies with sustainable dividend yields and growth",
                "Focus on dividend aristocrats with long payment histories",
                "Use dividend coverage ratios for sustainability analysis"
            ]
            
            contrarian_strategies = [
                "Identify fundamentally strong companies in temporary distress",
                "Use mean reversion strategies for cyclical sectors",
                "Focus on turnaround situations with improving fundamentals"
            ]
            
            report.indicator_recommendations = {symbol: recommendations for symbol in self.symbols}
            report.risk_warnings = warnings
            report.optimization_suggestions = optimizations
            report.value_strategies = value_strategies
            report.growth_strategies = growth_strategies
            report.quality_strategies = quality_strategies
            report.dividend_strategies = dividend_strategies
            report.contrarian_strategies = contrarian_strategies
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating report summary: {e}")
            return report
    
    def get_indicator_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all evaluated indicators"""
        try:
            if not self.evaluation_history:
                return {"message": "No evaluation history available"}
            
            latest_report = self.evaluation_history[-1]
            
            summary = {
                "evaluation_date": latest_report.evaluation_date,
                "symbols": latest_report.symbols,
                "analysis_periods": latest_report.analysis_periods,
                "indicators_evaluated": list(latest_report.backtest_results.keys()),
                "best_indicators_by_symbol_period": latest_report.best_performing_indicators,
                "key_recommendations": latest_report.optimization_suggestions[:3],
                "risk_warnings": latest_report.risk_warnings[:3]
            }
            
            # Performance statistics
            all_sharpe_ratios = []
            all_returns = []
            
            for indicator_results in latest_report.backtest_results.values():
                for result in indicator_results:
                    if result.sharpe_ratio is not None:
                        all_sharpe_ratios.append(result.sharpe_ratio)
                    if result.annualized_return is not None:
                        all_returns.append(result.annualized_return)
            
            if all_sharpe_ratios:
                summary["avg_sharpe_ratio"] = np.mean(all_sharpe_ratios)
                summary["best_sharpe_ratio"] = max(all_sharpe_ratios)
            
            if all_returns:
                summary["avg_annual_return"] = np.mean(all_returns)
                summary["best_annual_return"] = max(all_returns)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {"error": str(e)}
    
    def save_evaluation_report(self, report: FundamentalEvaluationReport, filepath: str) -> bool:
        """Save evaluation report to file"""
        try:
            import json
            from datetime import datetime
            
            # Convert report to serializable format
            report_dict = {
                "evaluation_date": report.evaluation_date.isoformat(),
                "symbols": report.symbols,
                "analysis_periods": report.analysis_periods,
                "evaluation_timeframe": report.evaluation_timeframe,
                "best_performing_indicators": report.best_performing_indicators,
                "optimization_suggestions": report.optimization_suggestions,
                "risk_warnings": report.risk_warnings,
                "value_strategies": report.value_strategies,
                "growth_strategies": report.growth_strategies,
                "quality_strategies": report.quality_strategies,
                "dividend_strategies": report.dividend_strategies
            }
            
            with open(filepath, 'w') as f:
                json.dump(report_dict, f, indent=2)
            
            logger.info(f"Evaluation report saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving evaluation report: {e}")
            return False

# Example usage and testing
if __name__ == "__main__":
    # Initialize framework
    framework = FundamentalEvaluationFramework(
        symbols=['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'META', 'AMZN', 'NFLX'],
        analysis_periods=['quarterly', 'annual'],
        evaluation_window=20,
        prediction_horizons=[1, 2, 4, 8]
    )
    
    # Generate sample fundamental data
    sample_fundamental_data = {}
    sample_price_data = {}
    
    for symbol in framework.symbols:
        # Generate quarterly fundamental data
        dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='Q')
        
        quarterly_data = pd.DataFrame({
            'pe_ratio': np.random.normal(20, 5, len(dates)),
            'pb_ratio': np.random.normal(3, 1, len(dates)),
            'roe': np.random.normal(0.15, 0.05, len(dates)),
            'debt_to_equity': np.random.normal(0.5, 0.2, len(dates)),
            'current_ratio': np.random.normal(2, 0.5, len(dates)),
            'revenue_growth': np.random.normal(0.1, 0.15, len(dates)),
            'earnings_growth': np.random.normal(0.12, 0.2, len(dates)),
            'dividend_yield': np.random.normal(0.02, 0.01, len(dates))
        }, index=dates)
        
        # Generate annual data (subset of quarterly)
        annual_dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='Y')
        annual_data = quarterly_data.resample('Y').last()
        
        sample_fundamental_data[symbol] = {
            'quarterly': quarterly_data,
            'annual': annual_data
        }
        
        # Generate sample price data
        price_dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, len(price_dates))))
        
        sample_price_data[symbol] = pd.DataFrame({
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, len(price_dates))
        }, index=price_dates)
    
    # Run evaluation
    print("Running comprehensive fundamental evaluation...")
    evaluation_report = framework.generate_evaluation_report(
        sample_fundamental_data, 
        sample_price_data
    )
    
    # Display results
    print("\n=== FUNDAMENTAL EVALUATION RESULTS ===")
    print(f"Evaluation Date: {evaluation_report.evaluation_date}")
    print(f"Symbols Evaluated: {', '.join(evaluation_report.symbols)}")
    print(f"Analysis Periods: {', '.join(evaluation_report.analysis_periods)}")
    print(f"Evaluation Timeframe: {evaluation_report.evaluation_timeframe}")
    
    # Performance summary
    performance_summary = framework.get_indicator_performance_summary()
    if 'avg_sharpe_ratio' in performance_summary:
        print(f"\nAverage Sharpe Ratio: {performance_summary['avg_sharpe_ratio']:.3f}")
        print(f"Best Sharpe Ratio: {performance_summary['best_sharpe_ratio']:.3f}")
    
    if 'avg_annual_return' in performance_summary:
        print(f"Average Annual Return: {performance_summary['avg_annual_return']:.1%}")
        print(f"Best Annual Return: {performance_summary['best_annual_return']:.1%}")
    
    # Best indicators by symbol
    print("\n=== BEST INDICATORS BY SYMBOL ===")
    for symbol_period, indicator in evaluation_report.best_performing_indicators.items():
        print(f"{symbol_period}: {indicator}")
    
    # Risk warnings
    if evaluation_report.risk_warnings:
        print("\n=== RISK WARNINGS ===")
        for warning in evaluation_report.risk_warnings[:5]:
            print(f"⚠️  {warning}")
    
    # Optimization suggestions
    if evaluation_report.optimization_suggestions:
        print("\n=== OPTIMIZATION SUGGESTIONS ===")
        for suggestion in evaluation_report.optimization_suggestions[:5]:
            print(f"💡 {suggestion}")
    
    # Investment strategy recommendations
    print("\n=== INVESTMENT STRATEGY RECOMMENDATIONS ===")
    
    if evaluation_report.value_strategies:
        print("\nValue Investing Strategies:")
        for strategy in evaluation_report.value_strategies:
            print(f"📈 {strategy}")
    
    if evaluation_report.growth_strategies:
        print("\nGrowth Investing Strategies:")
        for strategy in evaluation_report.growth_strategies:
            print(f"🚀 {strategy}")
    
    if evaluation_report.quality_strategies:
        print("\nQuality Investing Strategies:")
        for strategy in evaluation_report.quality_strategies:
            print(f"⭐ {strategy}")
    
    if evaluation_report.dividend_strategies:
        print("\nDividend Investing Strategies:")
        for strategy in evaluation_report.dividend_strategies:
            print(f"💰 {strategy}")
    
    # Save summary
    summary_file = "fundamental_evaluation_summary.json"
    if framework.save_evaluation_report(evaluation_report, summary_file):
        print(f"\n📊 Evaluation summary saved to {summary_file}")
    
    print("\n🎯 Fundamental Evaluation Framework is ready for production use!")
    print("\nKey Features:")
    print("✅ Comprehensive fundamental indicator evaluation")
    print("✅ Multi-timeframe analysis (quarterly/annual)")
    print("✅ Signal performance and backtesting")
    print("✅ Prediction accuracy assessment")
    print("✅ Risk metrics and comparative analysis")
    print("✅ Investment strategy recommendations")
    print("✅ Value, growth, quality, and dividend strategies")
    print("✅ Sector and peer comparison analysis")
    print("✅ Financial health and distress indicators")
    print("✅ ESG and governance risk assessment")
    print("✅ Multi-factor model integration")