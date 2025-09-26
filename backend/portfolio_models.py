from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from datetime import datetime
from enum import Enum

class CrossAssetIndicatorType(Enum):
    """Cross-asset indicator types"""
    ARIMA = "arima"  # ARIMA Models
    SARIMA = "sarima"  # Seasonal ARIMA
    GARCH = "garch"  # GARCH Models
    EGARCH = "egarch"  # Exponential GARCH
    TGARCH = "tgarch"  # Threshold GARCH
    LSTM = "lstm"  # Long Short-Term Memory
    GRU = "gru"  # Gated Recurrent Unit
    TRANSFORMER = "transformer"  # Transformer Models
    XGBOOST = "xgboost"  # XGBoost
    LIGHTGBM = "lightgbm"  # LightGBM
    SVM = "svm"  # Support Vector Machine
    RSI = "rsi"  # Relative Strength Index
    MACD = "macd"  # Moving Average Convergence Divergence
    ICHIMOKU = "ichimoku"  # Ichimoku Cloud
    BOLLINGER_BANDS = "bollinger_bands"  # Bollinger Bands
    STOCHASTIC = "stochastic"  # Stochastic Oscillator
    PPO = "ppo"  # Proximal Policy Optimization (RL)
    SAC = "sac"  # Soft Actor-Critic (RL)
    DDPG = "ddpg"  # Deep Deterministic Policy Gradient (RL)
    MARKOWITZ_MPT = "markowitz_mpt"  # Modern Portfolio Theory
    MONTE_CARLO = "monte_carlo"  # Monte Carlo Simulation
    FINBERT = "finbert"  # Financial BERT
    CRYPTOBERT = "cryptobert"  # Crypto BERT
    FOREXBERT = "forexbert"  # Forex BERT
    HMM = "hmm"  # Hidden Markov Model
    BAYESIAN_CHANGE_POINT = "bayesian_change_point"  # Bayesian Change Point Detection
    CORRELATION_ANALYSIS = "correlation_analysis"  # Cross-Asset Correlation
    COINTEGRATION = "cointegration"  # Cointegration Analysis
    PAIRS_TRADING = "pairs_trading"  # Pairs Trading Strategy
    REGIME_SWITCHING = "regime_switching"  # Regime Switching Models

@dataclass
class AssetData:
    """Generic asset data structure"""
    symbol: str
    asset_type: str  # "crypto", "stock", "forex", "futures", "index"
    current_price: float
    historical_prices: List[float]
    volume: float
    market_cap: Optional[float] = None
    volatility: float = 0.0
    beta: Optional[float] = None
    correlation_matrix: Optional[Dict[str, float]] = None
    fundamental_data: Optional[Dict[str, Any]] = None

@dataclass
class CrossAssetIndicatorResult:
    """Result of cross-asset indicator calculation"""
    indicator_type: CrossAssetIndicatorType
    value: Union[float, Dict[str, float], List[float]]
    confidence: float
    metadata: Dict[str, Any]
    timestamp: datetime
    interpretation: str
    risk_level: str
    signal: str  # "BUY", "SELL", "HOLD"
    time_horizon: str
    asset_symbols: List[str]

class PortfolioModels:
    """Portfolio optimization models for cross-asset analysis"""
    
    @staticmethod
    def markowitz_optimization(assets_data: List[AssetData], risk_free_rate: float = 0.02) -> CrossAssetIndicatorResult:
        """Markowitz Mean-Variance Portfolio Optimization"""
        try:
            if len(assets_data) < 2:
                raise ValueError("Need at least 2 assets for portfolio optimization")
            
            # Calculate returns for each asset
            returns_matrix = []
            asset_symbols = []
            
            for asset in assets_data:
                prices = np.array(asset.historical_prices)
                if len(prices) < 30:
                    continue  # Skip assets with insufficient data
                
                returns = np.diff(np.log(prices))
                returns_matrix.append(returns[-min(252, len(returns)):])  # Use last year or available data
                asset_symbols.append(asset.symbol)
            
            if len(returns_matrix) < 2:
                raise ValueError("Insufficient data for portfolio optimization")
            
            # Align returns to same length
            min_length = min(len(r) for r in returns_matrix)
            returns_matrix = np.array([r[-min_length:] for r in returns_matrix]).T
            
            # Calculate expected returns and covariance matrix
            expected_returns = np.mean(returns_matrix, axis=0) * 252  # Annualized
            cov_matrix = np.cov(returns_matrix.T) * 252  # Annualized
            
            n_assets = len(asset_symbols)
            
            # Simplified portfolio optimization (equal risk contribution as baseline)
            # In practice, this would use scipy.optimize or cvxpy
            
            # Method 1: Equal Risk Contribution (Risk Parity)
            def calculate_portfolio_risk(weights, cov_matrix):
                return np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            
            def calculate_risk_contributions(weights, cov_matrix):
                portfolio_vol = calculate_portfolio_risk(weights, cov_matrix)
                marginal_contrib = np.dot(cov_matrix, weights)
                contrib = weights * marginal_contrib / (portfolio_vol ** 2)
                return contrib
            
            # Start with equal weights
            equal_weights = np.ones(n_assets) / n_assets
            
            # Iterative approach to risk parity
            weights = equal_weights.copy()
            for iteration in range(50):  # Simple iterative optimization
                risk_contrib = calculate_risk_contributions(weights, cov_matrix)
                target_contrib = 1.0 / n_assets
                
                # Adjust weights based on risk contribution deviation
                adjustment = (target_contrib - risk_contrib) * 0.1
                weights += adjustment
                
                # Normalize weights
                weights = np.maximum(weights, 0.01)  # Minimum 1% allocation
                weights = weights / np.sum(weights)
                
                # Check convergence
                if np.max(np.abs(risk_contrib - target_contrib)) < 0.01:
                    break
            
            # Method 2: Maximum Sharpe Ratio (simplified)
            excess_returns = expected_returns - risk_free_rate
            
            try:
                # Simplified optimization: inverse volatility weighting with return adjustment
                inv_vol = 1.0 / np.sqrt(np.diag(cov_matrix))
                vol_weights = inv_vol / np.sum(inv_vol)
                
                # Adjust for returns
                return_adjustment = np.maximum(excess_returns, 0)
                if np.sum(return_adjustment) > 0:
                    return_weights = return_adjustment / np.sum(return_adjustment)
                    sharpe_weights = 0.7 * vol_weights + 0.3 * return_weights
                else:
                    sharpe_weights = vol_weights
                
                sharpe_weights = sharpe_weights / np.sum(sharpe_weights)
            except:
                sharpe_weights = equal_weights
            
            # Calculate portfolio metrics for both approaches
            portfolios = {
                "equal_weight": equal_weights,
                "risk_parity": weights,
                "max_sharpe": sharpe_weights
            }
            
            portfolio_metrics = {}
            
            for name, portfolio_weights in portfolios.items():
                portfolio_return = np.dot(portfolio_weights, expected_returns)
                portfolio_vol = calculate_portfolio_risk(portfolio_weights, cov_matrix)
                sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
                
                portfolio_metrics[name] = {
                    "weights": portfolio_weights.tolist(),
                    "expected_return": portfolio_return,
                    "volatility": portfolio_vol,
                    "sharpe_ratio": sharpe_ratio
                }
            
            # Select best portfolio based on Sharpe ratio
            best_portfolio = max(portfolio_metrics.keys(), key=lambda x: portfolio_metrics[x]["sharpe_ratio"])
            optimal_weights = portfolio_metrics[best_portfolio]["weights"]
            optimal_metrics = portfolio_metrics[best_portfolio]
            
            # Generate signals based on portfolio composition
            weight_dict = dict(zip(asset_symbols, optimal_weights))
            
            # Determine overall signal
            if optimal_metrics["sharpe_ratio"] > 1.0:
                signal = "BUY"
                risk_level = "Low"
            elif optimal_metrics["sharpe_ratio"] > 0.5:
                signal = "HOLD"
                risk_level = "Medium"
            else:
                signal = "SELL"
                risk_level = "High"
            
            # Calculate confidence based on Sharpe ratio and diversification
            diversification_ratio = 1.0 / np.sum(np.array(optimal_weights) ** 2)  # Effective number of assets
            confidence = min(0.8, max(0.3, (optimal_metrics["sharpe_ratio"] + diversification_ratio / n_assets) / 2))
            
            return CrossAssetIndicatorResult(
                indicator_type=CrossAssetIndicatorType.MARKOWITZ_MPT,
                value=weight_dict,
                confidence=confidence,
                metadata={
                    "all_portfolios": portfolio_metrics,
                    "best_portfolio_type": best_portfolio,
                    "expected_return": optimal_metrics["expected_return"],
                    "volatility": optimal_metrics["volatility"],
                    "sharpe_ratio": optimal_metrics["sharpe_ratio"],
                    "diversification_ratio": diversification_ratio,
                    "correlation_matrix": cov_matrix.tolist(),
                    "risk_free_rate": risk_free_rate
                },
                timestamp=datetime.now(),
                interpretation=f"Optimal portfolio ({best_portfolio}): Return {optimal_metrics['expected_return']:.1%}, Vol {optimal_metrics['volatility']:.1%}, Sharpe {optimal_metrics['sharpe_ratio']:.2f}",
                risk_level=risk_level,
                signal=signal,
                time_horizon="Long-term",
                asset_symbols=asset_symbols
            )
            
        except Exception as e:
            return CrossAssetIndicatorResult(
                indicator_type=CrossAssetIndicatorType.MARKOWITZ_MPT,
                value={},
                confidence=0.0,
                metadata={"error": str(e)},
                timestamp=datetime.now(),
                interpretation="Portfolio optimization failed",
                risk_level="High",
                signal="HOLD",
                time_horizon="N/A",
                asset_symbols=[asset.symbol for asset in assets_data]
            )
    
    @staticmethod
    def monte_carlo_simulation(assets_data: List[AssetData], num_simulations: int = 1000, time_horizon: int = 252) -> CrossAssetIndicatorResult:
        """Monte Carlo portfolio simulation"""
        try:
            if len(assets_data) < 1:
                raise ValueError("Need at least 1 asset for Monte Carlo simulation")
            
            # Prepare data
            returns_data = []
            asset_symbols = []
            current_prices = []
            
            for asset in assets_data:
                prices = np.array(asset.historical_prices)
                if len(prices) < 30:
                    continue
                
                returns = np.diff(np.log(prices))
                returns_data.append(returns[-min(252, len(returns)):])  # Last year of data
                asset_symbols.append(asset.symbol)
                current_prices.append(asset.current_price)
            
            if len(returns_data) == 0:
                raise ValueError("No valid asset data for simulation")
            
            # Align returns
            min_length = min(len(r) for r in returns_data)
            returns_matrix = np.array([r[-min_length:] for r in returns_data])
            
            # Calculate statistics
            mean_returns = np.mean(returns_matrix, axis=1)
            cov_matrix = np.cov(returns_matrix)
            
            # Generate random portfolio weights for simulation
            np.random.seed(42)  # For reproducibility
            
            portfolio_results = []
            
            for _ in range(num_simulations):
                # Random weights
                weights = np.random.random(len(asset_symbols))
                weights = weights / np.sum(weights)
                
                # Portfolio statistics
                portfolio_return = np.dot(weights, mean_returns)
                portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
                portfolio_vol = np.sqrt(portfolio_variance)
                
                # Simulate portfolio path
                portfolio_values = [1.0]  # Start with $1
                
                for day in range(time_horizon):
                    # Generate correlated random returns
                    random_returns = np.random.multivariate_normal(mean_returns, cov_matrix)
                    portfolio_daily_return = np.dot(weights, random_returns)
                    
                    new_value = portfolio_values[-1] * (1 + portfolio_daily_return)
                    portfolio_values.append(new_value)
                
                final_value = portfolio_values[-1]
                total_return = final_value - 1.0
                annualized_return = (final_value ** (252 / time_horizon)) - 1
                
                # Calculate maximum drawdown
                running_max = np.maximum.accumulate(portfolio_values)
                drawdowns = (np.array(portfolio_values) - running_max) / running_max
                max_drawdown = np.min(drawdowns)
                
                portfolio_results.append({
                    "weights": weights,
                    "final_value": final_value,
                    "total_return": total_return,
                    "annualized_return": annualized_return,
                    "volatility": portfolio_vol * np.sqrt(252),
                    "max_drawdown": max_drawdown,
                    "sharpe_ratio": annualized_return / (portfolio_vol * np.sqrt(252)) if portfolio_vol > 0 else 0
                })
            
            # Analyze results
            final_values = [r["final_value"] for r in portfolio_results]
            returns = [r["total_return"] for r in portfolio_results]
            sharpe_ratios = [r["sharpe_ratio"] for r in portfolio_results]
            
            # Statistics
            mean_final_value = np.mean(final_values)
            median_final_value = np.median(final_values)
            std_final_value = np.std(final_values)
            
            # Risk metrics
            var_95 = np.percentile(returns, 5)  # 95% VaR
            var_99 = np.percentile(returns, 1)  # 99% VaR
            prob_loss = np.mean(np.array(returns) < 0)
            prob_large_loss = np.mean(np.array(returns) < -0.2)
            
            # Best portfolio (highest Sharpe ratio)
            best_idx = np.argmax(sharpe_ratios)
            best_portfolio = portfolio_results[best_idx]
            
            # Generate signal based on simulation results
            if mean_final_value > 1.1 and prob_loss < 0.3:
                signal = "BUY"
                risk_level = "Low" if var_95 > -0.1 else "Medium"
            elif mean_final_value > 1.0 and prob_loss < 0.5:
                signal = "HOLD"
                risk_level = "Medium"
            else:
                signal = "SELL"
                risk_level = "High"
            
            # Confidence based on consistency of results
            confidence_score = 1 - std_final_value / mean_final_value if mean_final_value > 0 else 0
            confidence = min(0.8, max(0.3, confidence_score))
            
            return CrossAssetIndicatorResult(
                indicator_type=CrossAssetIndicatorType.MONTE_CARLO,
                value={
                    "mean_final_value": mean_final_value,
                    "median_final_value": median_final_value,
                    "probability_of_loss": prob_loss,
                    "var_95": var_95,
                    "var_99": var_99,
                    "best_portfolio_weights": dict(zip(asset_symbols, best_portfolio["weights"]))
                },
                confidence=confidence,
                metadata={
                    "num_simulations": num_simulations,
                    "time_horizon_days": time_horizon,
                    "mean_return": np.mean(returns),
                    "std_return": np.std(returns),
                    "best_sharpe_ratio": best_portfolio["sharpe_ratio"],
                    "prob_large_loss": prob_large_loss,
                    "percentiles": {
                        "5th": np.percentile(final_values, 5),
                        "25th": np.percentile(final_values, 25),
                        "75th": np.percentile(final_values, 75),
                        "95th": np.percentile(final_values, 95)
                    }
                },
                timestamp=datetime.now(),
                interpretation=f"Monte Carlo ({num_simulations} sims): Mean return {np.mean(returns):.1%}, Loss prob {prob_loss:.1%}, VaR95 {var_95:.1%}",
                risk_level=risk_level,
                signal=signal,
                time_horizon="Medium-term",
                asset_symbols=asset_symbols
            )
            
        except Exception as e:
            return CrossAssetIndicatorResult(
                indicator_type=CrossAssetIndicatorType.MONTE_CARLO,
                value={"mean_final_value": 1.0, "probability_of_loss": 0.5},
                confidence=0.0,
                metadata={"error": str(e)},
                timestamp=datetime.now(),
                interpretation="Monte Carlo simulation failed",
                risk_level="High",
                signal="HOLD",
                time_horizon="N/A",
                asset_symbols=[asset.symbol for asset in assets_data]
            )