"""Portfolio Analytics Engine for FinScope - Phase 7 Implementation

Provides comprehensive portfolio performance analysis, risk metrics,
and advanced analytics for investment decision making.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

import pandas as pd
import numpy as np
from scipy import stats
from sqlalchemy.orm import Session

from database import get_db
from db_models import Portfolio, PortfolioHolding
from market_data import MarketDataService

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Portfolio performance metrics"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    alpha: float
    beta: float
    information_ratio: float
    tracking_error: float
    win_rate: float
    profit_factor: float
    var_95: float  # Value at Risk (95%)
    cvar_95: float  # Conditional Value at Risk (95%)

@dataclass
class RiskMetrics:
    """Portfolio risk analysis metrics"""
    portfolio_var: float
    component_var: Dict[str, float]
    marginal_var: Dict[str, float]
    correlation_matrix: Dict[str, Dict[str, float]]
    concentration_risk: float
    sector_exposure: Dict[str, float]
    geographic_exposure: Dict[str, float]
    currency_exposure: Dict[str, float]
    liquidity_score: float
    stress_test_results: Dict[str, float]

@dataclass
class AttributionAnalysis:
    """Performance attribution analysis"""
    asset_allocation_effect: float
    security_selection_effect: float
    interaction_effect: float
    total_active_return: float
    sector_attribution: Dict[str, Dict[str, float]]
    security_attribution: Dict[str, Dict[str, float]]

class PortfolioAnalytics:
    """Advanced portfolio analytics engine"""
    
    def __init__(self):
        self.market_service = MarketDataService()
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        
        # Performance calculation settings
        self.trading_days_per_year = 252
        self.confidence_levels = [0.95, 0.99]
        
    async def calculate_performance_metrics(
        self,
        portfolio_id: str,
        db: Session,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        benchmark_symbol: str = "SPY"
    ) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        try:
            # Get portfolio data
            portfolio = db.query(Portfolio).filter(
                Portfolio.id == portfolio_id
            ).first()
            
            if not portfolio:
                raise ValueError(f"Portfolio {portfolio_id} not found")
            
            # Set default date range
            if not end_date:
                end_date = datetime.utcnow()
            if not start_date:
                start_date = portfolio.created_at
            
            # Get daily portfolio values
            daily_values = await self.calculate_daily_portfolio_values(
                portfolio_id, start_date, end_date, db
            )
            
            if len(daily_values) < 2:
                return self._get_default_metrics()
            
            # Calculate returns
            returns = self._calculate_returns(daily_values)
            
            # Get benchmark data
            benchmark_data = await self.market_service.get_historical_data(
                benchmark_symbol, start_date, end_date
            )
            benchmark_returns = self._calculate_returns(
                [price["close"] for price in benchmark_data]
            )
            
            # Calculate metrics
            total_return = (daily_values[-1] / daily_values[0] - 1) * 100
            
            # Annualized return
            days = len(daily_values)
            annualized_return = (
                (daily_values[-1] / daily_values[0]) ** (365.25 / days) - 1
            ) * 100
            
            # Volatility (annualized)
            volatility = np.std(returns) * np.sqrt(self.trading_days_per_year) * 100
            
            # Sharpe ratio
            excess_returns = np.array(returns) - (self.risk_free_rate / self.trading_days_per_year)
            sharpe_ratio = (
                np.mean(excess_returns) / np.std(returns) * np.sqrt(self.trading_days_per_year)
                if np.std(returns) > 0 else 0
            )
            
            # Sortino ratio
            downside_returns = [r for r in returns if r < 0]
            downside_deviation = np.std(downside_returns) if downside_returns else 0
            sortino_ratio = (
                np.mean(excess_returns) / downside_deviation * np.sqrt(self.trading_days_per_year)
                if downside_deviation > 0 else 0
            )
            
            # Maximum drawdown
            max_drawdown = self._calculate_max_drawdown(daily_values)
            
            # Calmar ratio
            calmar_ratio = (
                annualized_return / abs(max_drawdown)
                if max_drawdown != 0 else 0
            )
            
            # Alpha and Beta (vs benchmark)
            alpha, beta = self._calculate_alpha_beta(returns, benchmark_returns)
            
            # Information ratio and tracking error
            active_returns = np.array(returns) - np.array(benchmark_returns[:len(returns)])
            tracking_error = np.std(active_returns) * np.sqrt(self.trading_days_per_year) * 100
            information_ratio = (
                np.mean(active_returns) / np.std(active_returns) * np.sqrt(self.trading_days_per_year)
                if np.std(active_returns) > 0 else 0
            )
            
            # Win rate
            positive_returns = [r for r in returns if r > 0]
            win_rate = len(positive_returns) / len(returns) * 100 if returns else 0
            
            # Profit factor
            total_gains = sum(positive_returns)
            total_losses = abs(sum([r for r in returns if r < 0]))
            profit_factor = total_gains / total_losses if total_losses > 0 else float('inf')
            
            # Value at Risk (VaR) and Conditional VaR
            var_95 = np.percentile(returns, 5) * 100
            cvar_95 = np.mean([r for r in returns if r <= np.percentile(returns, 5)]) * 100
            
            return PerformanceMetrics(
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                calmar_ratio=calmar_ratio,
                alpha=alpha,
                beta=beta,
                information_ratio=information_ratio,
                tracking_error=tracking_error,
                win_rate=win_rate,
                profit_factor=profit_factor,
                var_95=var_95,
                cvar_95=cvar_95
            )
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return self._get_default_metrics()
    
    async def calculate_risk_metrics(
        self,
        portfolio_id: str,
        holdings: List[Dict[str, Any]],
        db: Session
    ) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        try:
            if not holdings:
                return self._get_default_risk_metrics()
            
            symbols = [h["symbol"] for h in holdings]
            weights = np.array([h["weight"] / 100 for h in holdings])
            
            # Get historical data for correlation analysis
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=365)  # 1 year of data
            
            returns_data = {}
            for symbol in symbols:
                try:
                    historical_data = await self.market_service.get_historical_data(
                        symbol, start_date, end_date
                    )
                    prices = [price["close"] for price in historical_data]
                    returns_data[symbol] = self._calculate_returns(prices)
                except Exception as e:
                    logger.warning(f"Could not get data for {symbol}: {str(e)}")
                    returns_data[symbol] = [0] * 252  # Default to zero returns
            
            # Align returns data
            min_length = min(len(returns) for returns in returns_data.values())
            aligned_returns = {
                symbol: returns[-min_length:] 
                for symbol, returns in returns_data.items()
            }
            
            # Create returns matrix
            returns_matrix = np.array(list(aligned_returns.values())).T
            
            # Calculate covariance matrix
            cov_matrix = np.cov(returns_matrix.T) * self.trading_days_per_year
            
            # Portfolio variance
            portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
            
            # Component VaR (contribution to portfolio risk)
            marginal_var = np.dot(cov_matrix, weights)
            component_var = weights * marginal_var
            component_var_dict = {
                symbols[i]: float(component_var[i]) 
                for i in range(len(symbols))
            }
            marginal_var_dict = {
                symbols[i]: float(marginal_var[i]) 
                for i in range(len(symbols))
            }
            
            # Correlation matrix
            correlation_matrix = np.corrcoef(returns_matrix.T)
            correlation_dict = {
                symbols[i]: {
                    symbols[j]: float(correlation_matrix[i][j]) 
                    for j in range(len(symbols))
                }
                for i in range(len(symbols))
            }
            
            # Concentration risk (Herfindahl index)
            concentration_risk = sum(w**2 for w in weights)
            
            # Sector/Geographic/Currency exposure (simplified)
            sector_exposure = self._calculate_sector_exposure(holdings)
            geographic_exposure = self._calculate_geographic_exposure(holdings)
            currency_exposure = self._calculate_currency_exposure(holdings)
            
            # Liquidity score (simplified)
            liquidity_score = self._calculate_liquidity_score(holdings)
            
            # Stress test scenarios
            stress_test_results = await self._perform_stress_tests(
                holdings, returns_matrix, weights
            )
            
            return RiskMetrics(
                portfolio_var=float(portfolio_var),
                component_var=component_var_dict,
                marginal_var=marginal_var_dict,
                correlation_matrix=correlation_dict,
                concentration_risk=float(concentration_risk),
                sector_exposure=sector_exposure,
                geographic_exposure=geographic_exposure,
                currency_exposure=currency_exposure,
                liquidity_score=liquidity_score,
                stress_test_results=stress_test_results
            )
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            return self._get_default_risk_metrics()
    
    async def calculate_attribution_analysis(
        self,
        portfolio_id: str,
        benchmark_symbol: str,
        db: Session,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> AttributionAnalysis:
        """Calculate performance attribution analysis"""
        try:
            # Get portfolio and benchmark data
            portfolio_returns = await self._get_portfolio_returns(
                portfolio_id, start_date, end_date, db
            )
            
            benchmark_data = await self.market_service.get_historical_data(
                benchmark_symbol, start_date or datetime.utcnow() - timedelta(days=365), 
                end_date or datetime.utcnow()
            )
            benchmark_returns = self._calculate_returns(
                [price["close"] for price in benchmark_data]
            )
            
            # Calculate attribution effects (simplified Brinson model)
            asset_allocation_effect = 0.0  # Would need sector/asset class data
            security_selection_effect = 0.0  # Would need individual security performance
            interaction_effect = 0.0
            
            # Total active return
            portfolio_return = np.mean(portfolio_returns) if portfolio_returns else 0
            benchmark_return = np.mean(benchmark_returns) if benchmark_returns else 0
            total_active_return = portfolio_return - benchmark_return
            
            return AttributionAnalysis(
                asset_allocation_effect=asset_allocation_effect,
                security_selection_effect=security_selection_effect,
                interaction_effect=interaction_effect,
                total_active_return=total_active_return * 100,
                sector_attribution={},  # Would be populated with real data
                security_attribution={}  # Would be populated with real data
            )
            
        except Exception as e:
            logger.error(f"Error calculating attribution analysis: {str(e)}")
            return AttributionAnalysis(
                asset_allocation_effect=0.0,
                security_selection_effect=0.0,
                interaction_effect=0.0,
                total_active_return=0.0,
                sector_attribution={},
                security_attribution={}
            )
    
    async def calculate_daily_portfolio_values(
        self,
        portfolio_id: str,
        start_date: datetime,
        end_date: datetime,
        db: Session
    ) -> List[float]:
        """Calculate daily portfolio values over time period"""
        try:
            # Get all holdings for the portfolio
            portfolio_holdings = db.query(PortfolioHolding).filter(
                PortfolioHolding.portfolio_id == portfolio_id,
                PortfolioHolding.last_transaction_date <= end_date
            ).all()
            
            # Get unique symbols
            symbols = list(set(h.symbol for h in portfolio_holdings if h.symbol != "CASH"))
            
            # Get historical price data for all symbols
            price_data = {}
            for symbol in symbols:
                try:
                    historical_data = await self.market_service.get_historical_data(
                        symbol, start_date, end_date
                    )
                    price_data[symbol] = {
                        price["date"]: price["close"] 
                        for price in historical_data
                    }
                except Exception as e:
                    logger.warning(f"Could not get price data for {symbol}: {str(e)}")
                    price_data[symbol] = {}
            
            # Calculate daily values
            daily_values = []
            current_date = start_date
            
            while current_date <= end_date:
                # Calculate holdings as of this date
                holdings = self._calculate_holdings_as_of_date(
                    portfolio_holdings, current_date
                )
                
                # Calculate portfolio value
                portfolio_value = holdings.get("CASH", 0)
                
                for symbol, quantity in holdings.items():
                    if symbol != "CASH" and quantity > 0:
                        # Get price for this date
                        date_str = current_date.strftime("%Y-%m-%d")
                        if symbol in price_data and date_str in price_data[symbol]:
                            price = price_data[symbol][date_str]
                            portfolio_value += quantity * price
                        elif symbol in price_data and price_data[symbol]:
                            # Use last available price
                            available_dates = sorted(price_data[symbol].keys())
                            last_date = max([d for d in available_dates if d <= date_str], default=None)
                            if last_date:
                                price = price_data[symbol][last_date]
                                portfolio_value += quantity * price
                
                daily_values.append(portfolio_value)
                current_date += timedelta(days=1)
            
            return daily_values
            
        except Exception as e:
            logger.error(f"Error calculating daily portfolio values: {str(e)}")
            return []
    
    async def calculate_period_performance(
        self,
        daily_values: List[float],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Calculate performance metrics for a specific period"""
        try:
            if len(daily_values) < 2:
                return {}
            
            returns = self._calculate_returns(daily_values)
            
            # Period return
            period_return = (daily_values[-1] / daily_values[0] - 1) * 100
            
            # Volatility
            volatility = np.std(returns) * 100
            
            # Maximum drawdown
            max_drawdown = self._calculate_max_drawdown(daily_values)
            
            # Best and worst days
            best_day = max(returns) * 100 if returns else 0
            worst_day = min(returns) * 100 if returns else 0
            
            # Positive days percentage
            positive_days = len([r for r in returns if r > 0])
            positive_days_pct = positive_days / len(returns) * 100 if returns else 0
            
            return {
                "period_return": period_return,
                "volatility": volatility,
                "max_drawdown": max_drawdown,
                "best_day": best_day,
                "worst_day": worst_day,
                "positive_days_percentage": positive_days_pct,
                "total_days": len(daily_values),
                "start_value": daily_values[0],
                "end_value": daily_values[-1]
            }
            
        except Exception as e:
            logger.error(f"Error calculating period performance: {str(e)}")
            return {}
    
    def _calculate_returns(self, values: List[float]) -> List[float]:
        """Calculate returns from price series"""
        if len(values) < 2:
            return []
        
        returns = []
        for i in range(1, len(values)):
            if values[i-1] != 0:
                ret = (values[i] / values[i-1]) - 1
                returns.append(ret)
            else:
                returns.append(0)
        
        return returns
    
    def _calculate_max_drawdown(self, values: List[float]) -> float:
        """Calculate maximum drawdown"""
        if len(values) < 2:
            return 0
        
        peak = values[0]
        max_dd = 0
        
        for value in values[1:]:
            if value > peak:
                peak = value
            else:
                drawdown = (peak - value) / peak
                max_dd = max(max_dd, drawdown)
        
        return max_dd * 100
    
    def _calculate_alpha_beta(self, portfolio_returns: List[float], benchmark_returns: List[float]) -> Tuple[float, float]:
        """Calculate alpha and beta vs benchmark"""
        try:
            if len(portfolio_returns) < 2 or len(benchmark_returns) < 2:
                return 0.0, 1.0
            
            # Align returns
            min_length = min(len(portfolio_returns), len(benchmark_returns))
            port_returns = np.array(portfolio_returns[-min_length:])
            bench_returns = np.array(benchmark_returns[-min_length:])
            
            # Calculate beta
            covariance = np.cov(port_returns, bench_returns)[0][1]
            benchmark_variance = np.var(bench_returns)
            
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
            
            # Calculate alpha
            port_mean = np.mean(port_returns)
            bench_mean = np.mean(bench_returns)
            alpha = (port_mean - self.risk_free_rate / self.trading_days_per_year) - \
                   beta * (bench_mean - self.risk_free_rate / self.trading_days_per_year)
            
            # Annualize alpha
            alpha = alpha * self.trading_days_per_year * 100
            
            return float(alpha), float(beta)
            
        except Exception as e:
            logger.error(f"Error calculating alpha/beta: {str(e)}")
            return 0.0, 1.0
    
    def _calculate_holdings_as_of_date(
        self,
        portfolio_holdings: List[PortfolioHolding],
        as_of_date: datetime
    ) -> Dict[str, float]:
        """Calculate holdings as of a specific date"""
        holdings = {"CASH": 0}
        
        for holding in portfolio_holdings:
            if holding.last_transaction_date <= as_of_date:
                symbol = holding.symbol
                holdings[symbol] = holding.quantity
        
        # Remove zero holdings
        return {k: v for k, v in holdings.items() if v != 0}
    
    def _calculate_sector_exposure(self, holdings: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate sector exposure (simplified)"""
        # This would use real sector classification data
        sector_map = {
            "AAPL": "Technology",
            "MSFT": "Technology",
            "GOOGL": "Technology",
            "AMZN": "Consumer Discretionary",
            "TSLA": "Consumer Discretionary",
            "JPM": "Financials",
            "JNJ": "Healthcare",
            "PG": "Consumer Staples",
            "SPY": "Diversified",
            "QQQ": "Technology"
        }
        
        sector_exposure = {}
        for holding in holdings:
            sector = sector_map.get(holding["symbol"], "Other")
            if sector not in sector_exposure:
                sector_exposure[sector] = 0
            sector_exposure[sector] += holding["weight"]
        
        return sector_exposure
    
    def _calculate_geographic_exposure(self, holdings: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate geographic exposure (simplified)"""
        # Simplified geographic mapping
        total_weight = 0
        for holding in holdings:
            total_weight += holding["weight"]
        
        return {"United States": total_weight}  # Simplified
    
    def _calculate_currency_exposure(self, holdings: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate currency exposure (simplified)"""
        # Simplified currency mapping
        total_weight = 0
        for holding in holdings:
            total_weight += holding["weight"]
        
        return {"USD": total_weight}  # Simplified
    
    def _calculate_liquidity_score(self, holdings: List[Dict[str, Any]]) -> float:
        """Calculate portfolio liquidity score (simplified)"""
        # Simplified liquidity scoring (0-100)
        # In reality, this would consider average daily volume, bid-ask spreads, etc.
        return 85.0  # Assume good liquidity
    
    async def _perform_stress_tests(
        self,
        holdings: List[Dict[str, Any]],
        returns_matrix: np.ndarray,
        weights: np.ndarray
    ) -> Dict[str, float]:
        """Perform stress test scenarios"""
        try:
            # Define stress scenarios
            scenarios = {
                "market_crash_2008": -0.20,  # -20% market shock
                "covid_crash_2020": -0.15,   # -15% market shock
                "interest_rate_shock": -0.10, # -10% for rate sensitive assets
                "inflation_shock": -0.08,     # -8% for inflation sensitive assets
                "liquidity_crisis": -0.12     # -12% liquidity shock
            }
            
            stress_results = {}
            
            for scenario_name, shock in scenarios.items():
                # Apply shock to portfolio
                shocked_returns = returns_matrix.mean(axis=0) + shock
                portfolio_impact = np.dot(weights, shocked_returns) * 100
                stress_results[scenario_name] = float(portfolio_impact)
            
            return stress_results
            
        except Exception as e:
            logger.error(f"Error performing stress tests: {str(e)}")
            return {}
    
    async def _get_portfolio_returns(
        self,
        portfolio_id: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        db: Session
    ) -> List[float]:
        """Get portfolio returns for attribution analysis"""
        try:
            daily_values = await self.calculate_daily_portfolio_values(
                portfolio_id, 
                start_date or datetime.utcnow() - timedelta(days=365),
                end_date or datetime.utcnow(),
                db
            )
            return self._calculate_returns(daily_values)
        except Exception as e:
            logger.error(f"Error getting portfolio returns: {str(e)}")
            return []
    
    def _get_default_metrics(self) -> PerformanceMetrics:
        """Get default performance metrics"""
        return PerformanceMetrics(
            total_return=0.0,
            annualized_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            calmar_ratio=0.0,
            alpha=0.0,
            beta=1.0,
            information_ratio=0.0,
            tracking_error=0.0,
            win_rate=0.0,
            profit_factor=1.0,
            var_95=0.0,
            cvar_95=0.0
        )
    
    def _get_default_risk_metrics(self) -> RiskMetrics:
        """Get default risk metrics"""
        return RiskMetrics(
            portfolio_var=0.0,
            component_var={},
            marginal_var={},
            correlation_matrix={},
            concentration_risk=0.0,
            sector_exposure={},
            geographic_exposure={},
            currency_exposure={},
            liquidity_score=100.0,
            stress_test_results={}
        )

# Global analytics instance
portfolio_analytics = PortfolioAnalytics()

def get_portfolio_analytics() -> PortfolioAnalytics:
    """Get portfolio analytics instance"""
    return portfolio_analytics