from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd

# Import individual analyzers
from .technical_indicators import TechnicalIndicatorCalculator, TechnicalIndicators
from .time_series_models import TimeSeriesAnalyzer, TimeSeriesResults
from .ml_models import MLAnalyzer, MLResults
from .rl_models import RLAnalyzer, RLResults
from .portfolio_theory import PortfolioOptimizer, PortfolioResults
from .nlp_models import NLPAnalyzer, NLPResults
from .state_models import StateModelAnalyzer, StateModelResults

# Conditional imports for plotting
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None
    sns = None


@dataclass
class CrossAssetData:
    """Data structure for cross-asset analysis"""
    asset_prices: Dict[str, List[float]]
    asset_returns: Dict[str, List[float]]
    timestamps: List[str]
    correlations: Optional[Dict[str, Dict[str, float]]] = None
    volatilities: Optional[Dict[str, float]] = None
    # Legacy support for single asset analysis
    prices: Optional[List[float]] = None
    volumes: Optional[List[float]] = None
    returns: Optional[List[float]] = None
    asset_names: Optional[List[str]] = None


@dataclass
class CrossAssetResult:
    """Comprehensive results from cross-asset analysis"""
    technical_indicators: TechnicalIndicators
    time_series_results: TimeSeriesResults
    ml_results: MLResults
    rl_results: RLResults
    portfolio_results: PortfolioResults
    nlp_results: NLPResults
    state_model_results: StateModelResults
    trading_signals: List[str]
    risk_metrics: Dict[str, float]
    insights: List[str]
    recommendations: List[str]


class CrossAssetAnalyzer:
    """Comprehensive cross-asset analysis orchestrator"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.technical_calculator = TechnicalIndicatorCalculator()
        self.ts_analyzer = TimeSeriesAnalyzer()
        self.ml_analyzer = MLAnalyzer()
        self.rl_analyzer = RLAnalyzer()
        self.portfolio_optimizer = PortfolioOptimizer(risk_free_rate)
        self.nlp_analyzer = NLPAnalyzer()
        self.state_analyzer = StateModelAnalyzer()
    
    def _prepare_data(self, data: CrossAssetData) -> CrossAssetData:
        """Prepare and validate data for analysis"""
        # Handle legacy single asset format
        if data.prices is not None and not data.asset_prices:
            data.asset_prices = {'main_asset': data.prices}
        
        if data.returns is not None and not data.asset_returns:
            data.asset_returns = {'main_asset': data.returns}
        
        # Ensure we have at least one asset
        if not data.asset_prices:
            raise ValueError("No asset price data provided")
        
        # Calculate returns if not provided
        for asset, prices in data.asset_prices.items():
            if asset not in data.asset_returns or not data.asset_returns[asset]:
                if len(prices) > 1:
                    returns = [0.0] + [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
                    data.asset_returns[asset] = returns
                else:
                    data.asset_returns[asset] = [0.0]
        
        return data
    
    def analyze(self, data: CrossAssetData, news_data: Optional[List[str]] = None) -> CrossAssetResult:
        """Comprehensive cross-asset analysis"""
        print("Starting comprehensive cross-asset analysis...")
        
        try:
            # Prepare data
            data = self._prepare_data(data)
            
            # Get main asset for single-asset analyses
            main_asset = list(data.asset_prices.keys())[0]
            main_prices = data.asset_prices[main_asset]
            main_returns = data.asset_returns[main_asset]
            
            # Calculate technical indicators
            technical_indicators = self.technical_calculator.calculate_all_indicators(data)
            
            # Time series analysis
            ts_results = self.ts_analyzer.analyze_all_assets(data)
            
            # Machine learning analysis
            ml_results = self.ml_analyzer.analyze_all_assets(data)
            
            # Reinforcement learning analysis
            rl_results = self.rl_analyzer.analyze_all_assets(data)
            
            # Portfolio optimization
            portfolio_results = self.portfolio_optimizer.analyze_portfolio(data)
            
            # NLP analysis
            if news_data:
                nlp_results = self.nlp_analyzer.analyze_news_sentiment(news_data, main_prices)
            else:
                nlp_results = self.nlp_analyzer.analyze_news_sentiment([], main_prices)
            
            # State model analysis
            returns_array = np.array(main_returns)
            state_results = self.state_analyzer.analyze_regime_switching(returns_array)
            
            # Generate combined trading signals
            trading_signals = self._generate_combined_signals(
                technical_indicators, ts_results, ml_results, rl_results, nlp_results, state_results
            )
            
            # Calculate risk metrics
            risk_metrics = self._calculate_comprehensive_risk_metrics(
                data, ts_results, ml_results, portfolio_results, state_results
            )
            
            # Generate insights and recommendations
            insights = self._generate_insights(
                technical_indicators, ts_results, ml_results, rl_results, 
                portfolio_results, nlp_results, state_results, risk_metrics
            )
            
            recommendations = self._generate_recommendations(
                trading_signals, risk_metrics, insights, portfolio_results
            )
            
            return CrossAssetResult(
                technical_indicators=technical_indicators,
                time_series_results=ts_results,
                ml_results=ml_results,
                rl_results=rl_results,
                portfolio_results=portfolio_results,
                nlp_results=nlp_results,
                state_model_results=state_results,
                trading_signals=trading_signals,
                risk_metrics=risk_metrics,
                insights=insights,
                recommendations=recommendations
            )
            
        except Exception as e:
            print(f"Analysis failed: {e}")
            return self._create_default_result(data)
    
    def _generate_combined_signals(self, technical: TechnicalIndicators, ts: TimeSeriesResults, 
                                 ml: MLResults, rl: RLResults, nlp: NLPResults, 
                                 state: StateModelResults) -> List[str]:
        """Generate combined trading signals from all models"""
        signals = []
        
        # Technical signals
        if technical.rsi and len(technical.rsi) > 0:
            if technical.rsi[-1] < 30:
                signals.append('BUY_RSI_OVERSOLD')
            elif technical.rsi[-1] > 70:
                signals.append('SELL_RSI_OVERBOUGHT')
        
        if technical.macd_signal and len(technical.macd_signal) > 0:
            if technical.macd_signal[-1] > 0:
                signals.append('BUY_MACD_BULLISH')
            elif technical.macd_signal[-1] < 0:
                signals.append('SELL_MACD_BEARISH')
        
        # Time series signals
        if ts.arima_forecast and len(ts.arima_forecast) > 1:
            if ts.arima_forecast[-1] > ts.arima_forecast[-2]:
                signals.append('BUY_ARIMA_UPTREND')
            else:
                signals.append('SELL_ARIMA_DOWNTREND')
        
        # ML signals
        if ml.lstm_predictions and len(ml.lstm_predictions) > 1:
            if ml.lstm_predictions[-1] > ml.lstm_predictions[-2]:
                signals.append('BUY_LSTM_PREDICTION')
            else:
                signals.append('SELL_LSTM_PREDICTION')
        
        # RL signals
        if rl.ppo_actions and len(rl.ppo_actions) > 0:
            if rl.ppo_actions[-1] == 1:  # Assuming 1 = buy, 0 = hold, -1 = sell
                signals.append('BUY_RL_PPO')
            elif rl.ppo_actions[-1] == -1:
                signals.append('SELL_RL_PPO')
        
        # Sentiment signals
        if nlp.sentiment_scores and len(nlp.sentiment_scores) > 0:
            if nlp.sentiment_scores[-1] > 0.6:
                signals.append('BUY_SENTIMENT_POSITIVE')
            elif nlp.sentiment_scores[-1] < 0.4:
                signals.append('SELL_SENTIMENT_NEGATIVE')
        
        # State model signals
        if len(state.hmm_states) > 0:
            current_state = state.hmm_states[-1]
            if current_state in state.regime_statistics:
                regime_return = state.regime_statistics[current_state]['mean_return']
                if regime_return > 0.001:  # Positive regime
                    signals.append('BUY_REGIME_POSITIVE')
                elif regime_return < -0.001:  # Negative regime
                    signals.append('SELL_REGIME_NEGATIVE')
        
        return signals if signals else ['HOLD_NO_CLEAR_SIGNAL']
    
    def _calculate_comprehensive_risk_metrics(self, data: CrossAssetData, ts: TimeSeriesResults,
                                            ml: MLResults, portfolio: PortfolioResults,
                                            state: StateModelResults) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        main_asset = list(data.asset_prices.keys())[0]
        prices = data.asset_prices[main_asset]
        returns = np.array(data.asset_returns[main_asset])
        
        # Basic risk metrics
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.1
        var_95 = np.percentile(returns, 5) if len(returns) > 0 else -0.02
        cvar_95 = np.mean(returns[returns <= var_95]) if len(returns[returns <= var_95]) > 0 else -0.03
        
        # Model-specific risks
        garch_vol = np.mean(ts.garch_volatility) if ts.garch_volatility else volatility
        ml_prediction_error = np.std(np.array(ml.lstm_predictions) - np.array(prices[-len(ml.lstm_predictions):])) if len(ml.lstm_predictions) > 0 and len(ml.lstm_predictions) <= len(prices) else 0.1
        
        # Portfolio risk
        portfolio_vol = portfolio.portfolio_metrics.get('volatility', volatility) if portfolio.portfolio_metrics else volatility
        
        # Regime risk
        regime_vol = np.mean([stats['volatility'] for stats in state.regime_statistics.values()]) if state.regime_statistics else volatility
        
        return {
            'volatility': volatility,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'garch_volatility': garch_vol,
            'ml_prediction_error': ml_prediction_error,
            'portfolio_volatility': portfolio_vol,
            'regime_volatility': regime_vol,
            'sharpe_ratio': np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(prices)
        }
    
    def _calculate_max_drawdown(self, prices: List[float]) -> float:
        """Calculate maximum drawdown"""
        if len(prices) < 2:
            return 0.0
        
        peak = prices[0]
        max_dd = 0.0
        
        for price in prices:
            if price > peak:
                peak = price
            drawdown = (peak - price) / peak if peak > 0 else 0
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _generate_insights(self, technical: TechnicalIndicators, ts: TimeSeriesResults,
                         ml: MLResults, rl: RLResults, portfolio: PortfolioResults,
                         nlp: NLPResults, state: StateModelResults, 
                         risk_metrics: Dict[str, float]) -> List[str]:
        """Generate key insights from analysis"""
        insights = []
        
        # Technical insights
        if technical.rsi and len(technical.rsi) > 0:
            if technical.rsi[-1] < 30:
                insights.append("Technical analysis indicates oversold conditions (RSI < 30)")
            elif technical.rsi[-1] > 70:
                insights.append("Technical analysis indicates overbought conditions (RSI > 70)")
        
        # Volatility insights
        if risk_metrics['volatility'] > 0.3:
            insights.append("High volatility environment detected - increased risk management required")
        elif risk_metrics['volatility'] < 0.1:
            insights.append("Low volatility environment - potential for volatility expansion")
        
        # ML model insights
        if ml.model_performance and ml.model_performance.get('lstm', 0) > 0.7:
            insights.append("LSTM model shows strong predictive performance (R² > 0.7)")
        
        # Sentiment insights
        if nlp.sentiment_scores:
            avg_sentiment = np.mean(nlp.sentiment_scores)
            if avg_sentiment > 0.6:
                insights.append("Market sentiment is predominantly positive")
            elif avg_sentiment < 0.4:
                insights.append("Market sentiment is predominantly negative")
        
        # Regime insights
        if state.regime_statistics and len(state.hmm_states) > 0:
            current_state = state.hmm_states[-1]
            if current_state in state.regime_statistics:
                regime_vol = state.regime_statistics[current_state]['volatility']
                if regime_vol > risk_metrics['volatility'] * 1.5:
                    insights.append("Currently in high-volatility regime")
                elif regime_vol < risk_metrics['volatility'] * 0.5:
                    insights.append("Currently in low-volatility regime")
        
        # Portfolio insights
        if portfolio.portfolio_metrics and portfolio.portfolio_metrics.get('sharpe_ratio', 0) > 1.0:
            insights.append("Portfolio optimization suggests strong risk-adjusted returns potential")
        
        return insights if insights else ["No significant insights detected from current analysis"]
    
    def _generate_recommendations(self, signals: List[str], risk_metrics: Dict[str, float],
                                insights: List[str], portfolio: PortfolioResults) -> List[str]:
        """Generate investment recommendations"""
        recommendations = []
        
        # Signal-based recommendations
        buy_signals = [s for s in signals if s.startswith('BUY')]
        sell_signals = [s for s in signals if s.startswith('SELL')]
        
        if len(buy_signals) > len(sell_signals):
            recommendations.append("Consider LONG position - multiple buy signals detected")
        elif len(sell_signals) > len(buy_signals):
            recommendations.append("Consider SHORT position - multiple sell signals detected")
        else:
            recommendations.append("HOLD position - mixed signals suggest caution")
        
        # Risk-based recommendations
        if risk_metrics['volatility'] > 0.3:
            recommendations.append("Reduce position size due to high volatility")
            recommendations.append("Implement tight stop-loss orders")
        
        if risk_metrics['max_drawdown'] > 0.2:
            recommendations.append("Consider diversification to reduce drawdown risk")
        
        # Portfolio recommendations
        if portfolio.optimal_weights:
            max_weight_asset = max(portfolio.optimal_weights, key=portfolio.optimal_weights.get)
            recommendations.append(f"Portfolio optimization suggests overweighting {max_weight_asset}")
        
        return recommendations if recommendations else ["Maintain current allocation pending clearer signals"]
    
    def _create_default_result(self, data: CrossAssetData) -> CrossAssetResult:
        """Create default result for error cases"""
        main_asset = list(data.asset_prices.keys())[0] if data.asset_prices else 'default'
        main_prices = data.asset_prices.get(main_asset, [100.0])
        
        default_technical = TechnicalIndicators(
            rsi=[50.0] * len(main_prices),
            macd=[0.0] * len(main_prices),
            macd_signal=[0.0] * len(main_prices),
            ichimoku_cloud_top=[main_prices[-1]] * len(main_prices),
            ichimoku_cloud_bottom=[main_prices[-1]] * len(main_prices)
        )
        
        return CrossAssetResult(
            technical_indicators=default_technical,
            time_series_results=TimeSeriesResults(
                arima_forecast=[main_prices[-1]] * 10,
                sarima_forecast=[main_prices[-1]] * 10,
                garch_volatility=[0.1] * len(main_prices),
                var_forecast=[main_prices[-1]] * 10
            ),
            ml_results=MLResults(
                lstm_predictions=[main_prices[-1]] * 10,
                gru_predictions=[main_prices[-1]] * 10,
                transformer_predictions=[main_prices[-1]] * 10,
                xgboost_predictions=[main_prices[-1]] * 10,
                lightgbm_predictions=[main_prices[-1]] * 10,
                svm_predictions=[main_prices[-1]] * 10,
                model_performance={'lstm': 0.5, 'gru': 0.5, 'transformer': 0.5, 'xgboost': 0.5, 'lightgbm': 0.5, 'svm': 0.5},
                feature_importance={}
            ),
            rl_results=RLResults(
                ppo_actions=[0] * len(main_prices),
                sac_actions=[0] * len(main_prices),
                ddpg_actions=[0] * len(main_prices),
                cumulative_rewards=[0.0] * len(main_prices),
                policy_performance={'ppo': 0.0, 'sac': 0.0, 'ddpg': 0.0}
            ),
            portfolio_results=PortfolioResults(
                optimal_weights={},
                expected_returns={},
                covariance_matrix=np.array([]),
                efficient_frontier=[],
                monte_carlo_simulations=[],
                portfolio_metrics={'expected_return': 0.0, 'volatility': 0.1, 'sharpe_ratio': 0.0},
                risk_attribution={}
            ),
            nlp_results=NLPResults(
                sentiment_scores=[],
                sentiment_classification=[],
                finbert_embeddings=None,
                cryptobert_embeddings=None,
                forexbert_embeddings=None,
                news_impact_scores=[],
                sentiment_momentum=[],
                keyword_analysis={},
                entity_sentiment={},
                topic_sentiment={}
            ),
            state_model_results=StateModelResults(
                hmm_states=np.array([0] * len(main_prices)),
                hmm_transition_matrix=np.array([[0.5, 0.5], [0.5, 0.5]]),
                hmm_means=np.array([0.0, 0.0]),
                hmm_covariances=np.array([0.1, 0.1]),
                change_points=[],
                regime_statistics={},
                model_likelihood=-1000.0,
                regime_probabilities=None,
                state_durations={},
                regime_transitions={}
            ),
            trading_signals=['HOLD_DEFAULT'],
            risk_metrics={'volatility': 0.1, 'var_95': -0.02, 'cvar_95': -0.03},
            insights=['Analysis failed - using default values'],
            recommendations=['Unable to generate recommendations - please check data quality']
        )
    
    def plot_results(self, result: CrossAssetResult, data: CrossAssetData) -> None:
        """Generate comprehensive visualization plots"""
        if plt is None:
            print("Matplotlib not available. Skipping plots.")
            return
        
        try:
            main_asset = list(data.asset_prices.keys())[0]
            main_prices = data.asset_prices[main_asset]
            
            fig, axes = plt.subplots(3, 3, figsize=(20, 15))
            fig.suptitle('Comprehensive Cross-Asset Analysis Results', fontsize=16)
            
            # 1. Price and Technical Indicators
            axes[0, 0].plot(main_prices, label='Price', color='blue')
            if result.technical_indicators.rsi:
                axes[0, 0].plot(result.technical_indicators.rsi, label='RSI', color='orange', alpha=0.7)
            axes[0, 0].set_title('Price and RSI')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # 2. MACD
            if result.technical_indicators.macd and result.technical_indicators.macd_signal:
                axes[0, 1].plot(result.technical_indicators.macd, label='MACD', color='blue')
                axes[0, 1].plot(result.technical_indicators.macd_signal, label='Signal', color='red')
            axes[0, 1].set_title('MACD Analysis')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # 3. Time Series Forecasts
            forecast_len = min(len(result.time_series_results.arima_forecast), 50)
            axes[0, 2].plot(main_prices[-forecast_len:], label='Actual', color='blue')
            axes[0, 2].plot(result.time_series_results.arima_forecast[:forecast_len], label='ARIMA', color='red', linestyle='--')
            axes[0, 2].set_title('Time Series Forecasts')
            axes[0, 2].legend()
            axes[0, 2].grid(True)
            
            # 4. ML Model Predictions
            pred_len = min(len(result.ml_results.lstm_predictions), len(main_prices))
            axes[1, 0].plot(main_prices[-pred_len:], label='Actual', color='blue')
            axes[1, 0].plot(result.ml_results.lstm_predictions[:pred_len], label='LSTM', color='green', alpha=0.7)
            axes[1, 0].plot(result.ml_results.xgboost_predictions[:pred_len], label='XGBoost', color='purple', alpha=0.7)
            axes[1, 0].set_title('ML Model Predictions')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # 5. Model Performance Comparison
            if result.ml_results.model_performance:
                models = list(result.ml_results.model_performance.keys())
                performance = list(result.ml_results.model_performance.values())
                axes[1, 1].bar(models, performance, color=['blue', 'green', 'red', 'purple', 'orange', 'brown'])
                axes[1, 1].set_title('Model Performance (R²)')
                axes[1, 1].set_ylabel('R² Score')
                axes[1, 1].tick_params(axis='x', rotation=45)
                axes[1, 1].grid(True, alpha=0.3)
            
            # 6. Sentiment Analysis
            if result.nlp_results.sentiment_scores:
                axes[1, 2].plot(result.nlp_results.sentiment_scores, label='Sentiment', color='green')
                axes[1, 2].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
                axes[1, 2].set_title('Sentiment Analysis')
                axes[1, 2].set_ylabel('Sentiment Score')
                axes[1, 2].legend()
                axes[1, 2].grid(True)
            else:
                axes[1, 2].text(0.5, 0.5, 'No Sentiment Data', ha='center', va='center', transform=axes[1, 2].transAxes)
                axes[1, 2].set_title('Sentiment Analysis')
            
            # 7. HMM States
            if len(result.state_model_results.hmm_states) > 0:
                axes[2, 0].plot(result.state_model_results.hmm_states, label='HMM States', color='red', marker='o', markersize=2)
                axes[2, 0].set_title('Hidden Markov Model States')
                axes[2, 0].set_ylabel('State')
                axes[2, 0].legend()
                axes[2, 0].grid(True)
            else:
                axes[2, 0].text(0.5, 0.5, 'No HMM Data', ha='center', va='center', transform=axes[2, 0].transAxes)
                axes[2, 0].set_title('Hidden Markov Model States')
            
            # 8. Risk Metrics
            risk_names = list(result.risk_metrics.keys())[:6]  # Show top 6 metrics
            risk_values = [result.risk_metrics[name] for name in risk_names]
            axes[2, 1].bar(risk_names, risk_values, color='red', alpha=0.7)
            axes[2, 1].set_title('Risk Metrics')
            axes[2, 1].tick_params(axis='x', rotation=45)
            axes[2, 1].grid(True, alpha=0.3)
            
            # 9. Trading Signals
            signal_counts = {}
            for signal in result.trading_signals:
                signal_type = signal.split('_')[0]  # BUY, SELL, HOLD
                signal_counts[signal_type] = signal_counts.get(signal_type, 0) + 1
            
            if signal_counts:
                axes[2, 2].pie(signal_counts.values(), labels=signal_counts.keys(), autopct='%1.1f%%')
                axes[2, 2].set_title('Trading Signals Distribution')
            else:
                axes[2, 2].text(0.5, 0.5, 'No Trading Signals', ha='center', va='center', transform=axes[2, 2].transAxes)
                axes[2, 2].set_title('Trading Signals Distribution')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Plotting failed: {e}")
    
    def generate_report(self, result: CrossAssetResult, data: CrossAssetData) -> str:
        """Generate comprehensive analysis report"""
        main_asset = list(data.asset_prices.keys())[0]
        main_prices = data.asset_prices[main_asset]
        
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE CROSS-ASSET ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 20)
        report.append(f"Analysis Period: {len(main_prices)} data points")
        report.append(f"Asset Classes: {', '.join(data.asset_prices.keys())}")
        report.append(f"Current Price: ${main_prices[-1]:.2f}")
        report.append(f"Overall Volatility: {result.risk_metrics.get('volatility', 0):.2%}")
        report.append(f"Sharpe Ratio: {result.risk_metrics.get('sharpe_ratio', 0):.2f}")
        report.append("")
        
        # Technical Analysis
        report.append("TECHNICAL ANALYSIS")
        report.append("-" * 20)
        if result.technical_indicators.rsi:
            report.append(f"Current RSI: {result.technical_indicators.rsi[-1]:.1f}")
            if result.technical_indicators.rsi[-1] < 30:
                report.append("  → Oversold condition detected")
            elif result.technical_indicators.rsi[-1] > 70:
                report.append("  → Overbought condition detected")
            else:
                report.append("  → Neutral RSI levels")
        
        if result.technical_indicators.macd_signal:
            report.append(f"MACD Signal: {result.technical_indicators.macd_signal[-1]:.4f}")
            if result.technical_indicators.macd_signal[-1] > 0:
                report.append("  → Bullish MACD crossover")
            else:
                report.append("  → Bearish MACD crossover")
        report.append("")
        
        # Key Insights
        report.append("KEY INSIGHTS")
        report.append("-" * 20)
        for i, insight in enumerate(result.insights[:5], 1):  # Show top 5 insights
            report.append(f"{i}. {insight}")
        report.append("")
        
        # Investment Recommendations
        report.append("INVESTMENT RECOMMENDATIONS")
        report.append("-" * 20)
        for i, recommendation in enumerate(result.recommendations[:5], 1):  # Show top 5 recommendations
            report.append(f"{i}. {recommendation}")
        report.append("")
        
        # Trading Signals
        report.append("TRADING SIGNALS")
        report.append("-" * 20)
        buy_signals = [s for s in result.trading_signals if s.startswith('BUY')]
        sell_signals = [s for s in result.trading_signals if s.startswith('SELL')]
        hold_signals = [s for s in result.trading_signals if s.startswith('HOLD')]
        
        report.append(f"Buy Signals: {len(buy_signals)}")
        for signal in buy_signals[:3]:  # Show top 3
            report.append(f"  • {signal.replace('_', ' ')}")
        
        report.append(f"Sell Signals: {len(sell_signals)}")
        for signal in sell_signals[:3]:  # Show top 3
            report.append(f"  • {signal.replace('_', ' ')}")
        
        if hold_signals:
            report.append(f"Hold Signals: {len(hold_signals)}")
        report.append("")
        
        # Current Trading Signal
        report.append("CURRENT TRADING SIGNAL")
        report.append("-" * 20)
        if len(buy_signals) > len(sell_signals):
            report.append("🟢 BUY - Multiple bullish indicators detected")
        elif len(sell_signals) > len(buy_signals):
            report.append("🔴 SELL - Multiple bearish indicators detected")
        else:
            report.append("🟡 HOLD - Mixed signals, maintain current position")
        report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)


# Example usage
if __name__ == "__main__":
    # Create sample cross-asset data
    np.random.seed(42)
    n_periods = 252  # One year of daily data
    
    # Generate synthetic price data with trends and volatility
    base_price = 100
    returns = np.random.normal(0.0005, 0.02, n_periods)  # Daily returns
    prices = [base_price]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    # Create sample data for multiple assets
    sample_data = CrossAssetData(
        asset_prices={
            'STOCK_A': prices,
            'STOCK_B': [p * 1.1 + np.random.normal(0, 1) for p in prices],
            'BOND_C': [p * 0.8 + np.random.normal(0, 0.5) for p in prices]
        },
        asset_returns={
            'STOCK_A': returns.tolist(),
            'STOCK_B': (returns * 1.2 + np.random.normal(0, 0.005, len(returns))).tolist(),
            'BOND_C': (returns * 0.6 + np.random.normal(0, 0.002, len(returns))).tolist()
        },
        timestamps=[f"2024-01-01T{i:02d}:00:00" for i in range(len(prices))]
    )
    
    # Sample news data
    sample_news = [
        "Market shows positive momentum with strong earnings",
        "Economic indicators suggest continued growth",
        "Central bank maintains dovish stance on rates",
        "Geopolitical tensions create market uncertainty",
        "Technology sector leads market gains"
    ] * (len(prices) // 5)  # Repeat to match data length
    
    # Initialize analyzer
    analyzer = CrossAssetAnalyzer()
    
    print("Starting comprehensive cross-asset analysis...")
    print(f"Analyzing {len(prices)} data points across {len(sample_data.asset_prices)} assets")
    
    # Perform analysis
    result = analyzer.analyze(sample_data, sample_news)
    
    # Generate and display report
    report = analyzer.generate_report(result, sample_data)
    print(report)
    
    # Generate plots
    analyzer.plot_results(result, sample_data)
    
    print("\nAnalysis completed successfully!")
    print(f"Generated {len(result.trading_signals)} trading signals")
    print(f"Identified {len(result.insights)} key insights")
    print(f"Provided {len(result.recommendations)} investment recommendations")