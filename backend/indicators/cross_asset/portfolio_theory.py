from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd

# Conditional imports with fallbacks
try:
    import cvxpy as cp
except ImportError:
    cp = None


@dataclass
class CrossAssetData:
    """Data structure for cross-asset analysis"""
    asset_prices: Dict[str, List[float]]
    asset_returns: Dict[str, List[float]]
    timestamps: List[str]
    correlations: Optional[Dict[str, Dict[str, float]]] = None
    volatilities: Optional[Dict[str, float]] = None


@dataclass
class PortfolioResults:
    """Results from portfolio optimization"""
    optimal_weights: Dict[str, float]
    expected_returns: Dict[str, float]
    covariance_matrix: np.ndarray
    efficient_frontier: List[Tuple[float, float]]
    monte_carlo_simulations: List[Dict[str, float]]
    portfolio_metrics: Dict[str, float]
    risk_attribution: Dict[str, float]


class PortfolioOptimizer:
    """Portfolio optimization using Markowitz and Monte Carlo"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
    
    def calculate_returns_covariance(self, data: CrossAssetData) -> Tuple[Dict[str, float], np.ndarray]:
        """Calculate expected returns and covariance matrix"""
        assets = list(data.asset_returns.keys())
        returns_matrix = []
        
        for asset in assets:
            returns = data.asset_returns[asset]
            if returns:
                returns_matrix.append(returns)
            else:
                # Calculate returns from prices
                prices = data.asset_prices[asset]
                asset_returns = [0.0] + [prices[i]/prices[i-1] - 1 for i in range(1, len(prices))]
                returns_matrix.append(asset_returns)
        
        # Ensure all return series have the same length
        min_length = min(len(r) for r in returns_matrix)
        returns_matrix = [r[:min_length] for r in returns_matrix]
        
        returns_df = pd.DataFrame(returns_matrix).T
        returns_df.columns = assets
        
        # Expected returns (annualized)
        expected_returns = {}
        for asset in assets:
            mean_return = returns_df[asset].mean() * 252  # Annualized
            expected_returns[asset] = mean_return
        
        # Covariance matrix (annualized)
        cov_matrix = returns_df.cov().values * 252
        
        return expected_returns, cov_matrix
    
    def optimize_portfolio(self, expected_returns: Dict[str, float], cov_matrix: np.ndarray, target_return: Optional[float] = None) -> Dict[str, Any]:
        """Optimize portfolio using Markowitz theory"""
        assets = list(expected_returns.keys())
        n_assets = len(assets)
        
        if n_assets == 0:
            return {
                'weights': {},
                'expected_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0
            }
        
        # Use cvxpy if available, otherwise use simple optimization
        if cp is not None:
            return self._optimize_with_cvxpy(assets, expected_returns, cov_matrix, target_return)
        else:
            return self._optimize_simple(assets, expected_returns, cov_matrix)
    
    def _optimize_with_cvxpy(self, assets: List[str], expected_returns: Dict[str, float], cov_matrix: np.ndarray, target_return: Optional[float]) -> Dict[str, Any]:
        """Optimize using cvxpy"""
        n_assets = len(assets)
        weights = cp.Variable(n_assets)
        
        # Expected returns vector
        mu = np.array([expected_returns[asset] for asset in assets])
        
        # Objective: minimize portfolio variance
        portfolio_variance = cp.quad_form(weights, cov_matrix)
        objective = cp.Minimize(portfolio_variance)
        
        # Constraints
        constraints = [
            cp.sum(weights) == 1,  # Weights sum to 1
            weights >= 0  # Long-only
        ]
        
        # Target return constraint
        if target_return is not None:
            constraints.append(mu.T @ weights >= target_return)
        
        # Solve
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve()
            
            if weights.value is not None:
                optimal_weights = {assets[i]: float(weights.value[i]) for i in range(n_assets)}
                portfolio_return = sum(optimal_weights[asset] * expected_returns[asset] for asset in assets)
                portfolio_variance = float(portfolio_variance.value)
                portfolio_volatility = np.sqrt(portfolio_variance)
                sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
                
                return {
                    'weights': optimal_weights,
                    'expected_return': portfolio_return,
                    'volatility': portfolio_volatility,
                    'sharpe_ratio': sharpe_ratio
                }
        except Exception as e:
            print(f"CVXPY optimization failed: {e}")
        
        # Fallback to simple optimization
        return self._optimize_simple(assets, expected_returns, cov_matrix)
    
    def _optimize_simple(self, assets: List[str], expected_returns: Dict[str, float], cov_matrix: np.ndarray) -> Dict[str, Any]:
        """Simple equal-weight or return-weighted optimization"""
        n_assets = len(assets)
        
        # Equal weights as fallback
        equal_weights = {asset: 1.0/n_assets for asset in assets}
        
        # Calculate portfolio metrics
        portfolio_return = sum(equal_weights[asset] * expected_returns[asset] for asset in assets)
        
        # Portfolio variance
        weights_array = np.array([equal_weights[asset] for asset in assets])
        portfolio_variance = np.dot(weights_array, np.dot(cov_matrix, weights_array))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
        
        return {
            'weights': equal_weights,
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio
        }
    
    def generate_efficient_frontier(self, expected_returns: Dict[str, float], cov_matrix: np.ndarray, n_points: int = 50) -> List[Tuple[float, float]]:
        """Generate efficient frontier points"""
        if not expected_returns:
            return []
        
        assets = list(expected_returns.keys())
        min_return = min(expected_returns.values())
        max_return = max(expected_returns.values())
        
        target_returns = np.linspace(min_return, max_return, n_points)
        efficient_frontier = []
        
        for target_return in target_returns:
            result = self.optimize_portfolio(expected_returns, cov_matrix, target_return)
            if result['volatility'] > 0:
                efficient_frontier.append((result['volatility'], result['expected_return']))
        
        return efficient_frontier
    
    def monte_carlo_simulation(self, expected_returns: Dict[str, float], cov_matrix: np.ndarray, n_simulations: int = 10000) -> List[Dict[str, float]]:
        """Run Monte Carlo simulation for portfolio optimization"""
        if not expected_returns:
            return []
        
        assets = list(expected_returns.keys())
        n_assets = len(assets)
        
        simulations = []
        
        for _ in range(n_simulations):
            # Generate random weights
            weights = np.random.random(n_assets)
            weights /= np.sum(weights)  # Normalize to sum to 1
            
            # Calculate portfolio metrics
            portfolio_return = sum(weights[i] * expected_returns[assets[i]] for i in range(n_assets))
            
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
            
            simulation = {
                'return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'weights': {assets[i]: weights[i] for i in range(n_assets)}
            }
            simulations.append(simulation)
        
        return simulations
    
    def analyze_portfolio(self, data: CrossAssetData) -> PortfolioResults:
        """Comprehensive portfolio analysis"""
        print("Performing portfolio optimization...")
        
        # Calculate returns and covariance
        expected_returns, cov_matrix = self.calculate_returns_covariance(data)
        
        if not expected_returns:
            return PortfolioResults(
                optimal_weights={},
                expected_returns={},
                covariance_matrix=np.array([]),
                efficient_frontier=[],
                monte_carlo_simulations=[],
                portfolio_metrics={},
                risk_attribution={}
            )
        
        # Optimize portfolio
        optimal_portfolio = self.optimize_portfolio(expected_returns, cov_matrix)
        
        # Generate efficient frontier
        efficient_frontier = self.generate_efficient_frontier(expected_returns, cov_matrix)
        
        # Monte Carlo simulation
        mc_simulations = self.monte_carlo_simulation(expected_returns, cov_matrix, 1000)
        
        # Risk attribution
        risk_attribution = self._calculate_risk_attribution(optimal_portfolio['weights'], cov_matrix, list(expected_returns.keys()))
        
        return PortfolioResults(
            optimal_weights=optimal_portfolio['weights'],
            expected_returns=expected_returns,
            covariance_matrix=cov_matrix,
            efficient_frontier=efficient_frontier,
            monte_carlo_simulations=mc_simulations,
            portfolio_metrics={
                'expected_return': optimal_portfolio['expected_return'],
                'volatility': optimal_portfolio['volatility'],
                'sharpe_ratio': optimal_portfolio['sharpe_ratio']
            },
            risk_attribution=risk_attribution
        )
    
    def _calculate_risk_attribution(self, weights: Dict[str, float], cov_matrix: np.ndarray, assets: List[str]) -> Dict[str, float]:
        """Calculate risk attribution for each asset"""
        if not weights or len(cov_matrix) == 0:
            return {}
        
        weights_array = np.array([weights.get(asset, 0.0) for asset in assets])
        portfolio_variance = np.dot(weights_array, np.dot(cov_matrix, weights_array))
        
        risk_attribution = {}
        for i, asset in enumerate(assets):
            # Marginal contribution to risk
            marginal_contrib = np.dot(cov_matrix[i], weights_array)
            risk_contrib = weights_array[i] * marginal_contrib / portfolio_variance if portfolio_variance > 0 else 0
            risk_attribution[asset] = risk_contrib
        
        return risk_attribution
    
    def calculate_var(self, portfolio_returns: List[float], confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk (VaR)"""
        if not portfolio_returns:
            return 0.0
        
        sorted_returns = sorted(portfolio_returns)
        var_index = int(confidence_level * len(sorted_returns))
        return abs(sorted_returns[var_index]) if var_index < len(sorted_returns) else 0.0
    
    def calculate_cvar(self, portfolio_returns: List[float], confidence_level: float = 0.05) -> float:
        """Calculate Conditional Value at Risk (CVaR)"""
        if not portfolio_returns:
            return 0.0
        
        sorted_returns = sorted(portfolio_returns)
        var_index = int(confidence_level * len(sorted_returns))
        tail_returns = sorted_returns[:var_index] if var_index > 0 else [sorted_returns[0]]
        return abs(np.mean(tail_returns)) if tail_returns else 0.0
    
    def calculate_maximum_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not portfolio_values or len(portfolio_values) < 2:
            return 0.0
        
        peak = portfolio_values[0]
        max_drawdown = 0.0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return max_drawdown
    
    def calculate_sortino_ratio(self, portfolio_returns: List[float], target_return: float = 0.0) -> float:
        """Calculate Sortino ratio"""
        if not portfolio_returns:
            return 0.0
        
        excess_returns = [r - target_return for r in portfolio_returns]
        downside_returns = [r for r in excess_returns if r < 0]
        
        if not downside_returns:
            return float('inf') if np.mean(excess_returns) > 0 else 0.0
        
        downside_deviation = np.sqrt(np.mean([r**2 for r in downside_returns]))
        return np.mean(excess_returns) / downside_deviation if downside_deviation > 0 else 0.0


# Example usage
if __name__ == "__main__":
    # Sample data
    sample_data = CrossAssetData(
        asset_prices={
            'AAPL': [150.0, 152.0, 148.0, 155.0, 160.0, 158.0, 162.0, 165.0, 163.0, 168.0] * 10,
            'GOOGL': [2800.0, 2820.0, 2790.0, 2850.0, 2900.0, 2880.0, 2920.0, 2950.0, 2930.0, 2980.0] * 10,
            'MSFT': [300.0, 305.0, 298.0, 310.0, 315.0, 312.0, 318.0, 322.0, 320.0, 325.0] * 10
        },
        asset_returns={
            'AAPL': [0.01, -0.02, 0.03, 0.02, -0.01, 0.025, 0.018, -0.012, 0.03, -0.005] * 10,
            'GOOGL': [0.007, -0.01, 0.02, 0.018, -0.005, 0.014, 0.01, -0.007, 0.017, -0.003] * 10,
            'MSFT': [0.015, -0.008, 0.025, 0.012, -0.003, 0.02, 0.013, -0.006, 0.015, -0.002] * 10
        },
        timestamps=[f'2023-01-{i:02d}' for i in range(1, 101)]
    )
    
    # Initialize optimizer
    portfolio_optimizer = PortfolioOptimizer(risk_free_rate=0.02)
    
    # Perform analysis
    results = portfolio_optimizer.analyze_portfolio(sample_data)
    
    print("Portfolio Optimization Results:")
    print(f"Optimal Weights: {results.optimal_weights}")
    print(f"Expected Return: {results.portfolio_metrics['expected_return']:.4f}")
    print(f"Volatility: {results.portfolio_metrics['volatility']:.4f}")
    print(f"Sharpe Ratio: {results.portfolio_metrics['sharpe_ratio']:.4f}")
    print(f"Efficient Frontier Points: {len(results.efficient_frontier)}")
    print(f"Monte Carlo Simulations: {len(results.monte_carlo_simulations)}")
    print(f"Risk Attribution: {results.risk_attribution}")