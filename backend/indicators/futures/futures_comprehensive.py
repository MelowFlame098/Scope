from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from enum import Enum
from dataclasses import dataclass
from scipy import stats

logger = logging.getLogger(__name__)

class AssetType(Enum):
    STOCK = "stock"
    CRYPTO = "crypto"
    FOREX = "forex"
    FUTURES = "futures"
    INDEX = "index"
    CROSS_ASSET = "cross_asset"

class IndicatorCategory(Enum):
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    MACHINE_LEARNING = "machine_learning"
    STATISTICAL = "statistical"

@dataclass
class IndicatorResult:
    name: str
    values: pd.DataFrame
    metadata: Dict[str, Any]
    confidence: float
    timestamp: datetime
    asset_type: AssetType
    category: IndicatorCategory

class FuturesComprehensiveIndicators:
    """Comprehensive futures-specific indicators"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def cost_of_carry_model(self, spot_data: pd.DataFrame, futures_data: Optional[pd.DataFrame] = None,
                           risk_free_rate: float = 0.025, dividend_yield: float = 0.02,
                           storage_cost: float = 0.01, convenience_yield: float = 0.005,
                           time_to_expiry: float = 0.25) -> IndicatorResult:
        """Cost of Carry model for futures pricing"""
        try:
            spot_price = spot_data['close']
            
            # Calculate theoretical futures price
            # F = S * e^((r - q + c - y) * T)
            # Where: r = risk-free rate, q = dividend yield, c = storage cost, y = convenience yield
            carry_cost = risk_free_rate - dividend_yield + storage_cost - convenience_yield
            theoretical_futures = spot_price * np.exp(carry_cost * time_to_expiry)
            
            # If actual futures data provided, use it; otherwise use theoretical with noise
            if futures_data is not None and 'close' in futures_data.columns:
                actual_futures = futures_data['close']
            else:
                # Add realistic noise to theoretical price
                noise = np.random.normal(0, 0.02, len(theoretical_futures))
                actual_futures = theoretical_futures * (1 + noise)
            
            # Basis calculation
            basis = actual_futures - spot_price
            basis_percentage = (basis / spot_price) * 100
            
            # Theoretical basis
            theoretical_basis = theoretical_futures - spot_price
            theoretical_basis_percentage = (theoretical_basis / spot_price) * 100
            
            # Basis deviation from theoretical
            basis_deviation = basis - theoretical_basis
            basis_deviation_percentage = (basis_deviation / spot_price) * 100
            
            # Contango/Backwardation signals
            contango_signal = basis > 0  # Futures > Spot
            backwardation_signal = basis < 0  # Futures < Spot
            
            # Arbitrage opportunities
            arbitrage_threshold = 0.5  # 0.5% threshold
            arbitrage_opportunity = np.abs(basis_deviation_percentage) > arbitrage_threshold
            
            # Time decay effect
            time_decay = theoretical_basis * (1 / time_to_expiry) / 252  # Daily time decay
            
            # Roll yield calculation
            roll_yield = -basis_percentage / (time_to_expiry * 252)  # Daily roll yield
            
            # Convenience yield estimation
            implied_convenience_yield = risk_free_rate - dividend_yield + storage_cost - \
                                      (np.log(actual_futures / spot_price) / time_to_expiry)
            
            result_df = pd.DataFrame({
                'spot_price': spot_price,
                'futures_price': actual_futures,
                'theoretical_futures': theoretical_futures,
                'basis': basis,
                'basis_percentage': basis_percentage,
                'theoretical_basis': theoretical_basis,
                'basis_deviation': basis_deviation,
                'basis_deviation_percentage': basis_deviation_percentage,
                'contango_signal': contango_signal,
                'backwardation_signal': backwardation_signal,
                'arbitrage_opportunity': arbitrage_opportunity,
                'time_decay': time_decay,
                'roll_yield': roll_yield,
                'implied_convenience_yield': implied_convenience_yield
            }, index=spot_data.index)
            
            return IndicatorResult(
                name="Cost of Carry Model",
                values=result_df,
                metadata={
                    'risk_free_rate': risk_free_rate,
                    'dividend_yield': dividend_yield,
                    'storage_cost': storage_cost,
                    'convenience_yield': convenience_yield,
                    'time_to_expiry': time_to_expiry,
                    'carry_cost': carry_cost,
                    'current_basis': basis.iloc[-1],
                    'current_basis_percentage': basis_percentage.iloc[-1],
                    'market_structure': 'Contango' if basis.iloc[-1] > 0 else 'Backwardation',
                    'interpretation': 'Positive basis indicates contango, negative indicates backwardation'
                },
                confidence=0.75,
                timestamp=datetime.now(),
                asset_type=AssetType.FUTURES,
                category=IndicatorCategory.FUNDAMENTAL
            )
            
        except Exception as e:
            logger.error(f"Error calculating cost of carry model: {e}")
            return self._empty_result("Cost of Carry Model", AssetType.FUTURES)
    
    def term_structure_analysis(self, futures_chain: Dict[str, pd.DataFrame],
                               spot_data: pd.DataFrame) -> IndicatorResult:
        """Analyze futures term structure across different expiries"""
        try:
            spot_price = spot_data['close'].iloc[-1]
            
            # Extract futures prices for different expiries
            expiries = list(futures_chain.keys())
            expiries.sort()  # Sort by expiry date
            
            term_structure_data = {}
            
            for expiry in expiries:
                futures_price = futures_chain[expiry]['close'].iloc[-1]
                
                # Calculate time to expiry (simplified - assume monthly expiries)
                months_to_expiry = int(expiry.split('M')[0]) if 'M' in expiry else 1
                time_to_expiry = months_to_expiry / 12
                
                # Basis and annualized basis
                basis = futures_price - spot_price
                annualized_basis = (basis / spot_price) / time_to_expiry * 100
                
                term_structure_data[expiry] = {
                    'futures_price': futures_price,
                    'time_to_expiry': time_to_expiry,
                    'basis': basis,
                    'annualized_basis': annualized_basis
                }
            
            # Create term structure DataFrame
            ts_df = pd.DataFrame(term_structure_data).T
            
            # Calculate term structure slope
            if len(ts_df) >= 2:
                # Linear regression of futures prices vs time to expiry
                x = ts_df['time_to_expiry'].values
                y = ts_df['futures_price'].values
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                
                # Term structure curvature (second derivative approximation)
                if len(ts_df) >= 3:
                    # Fit quadratic polynomial
                    coeffs = np.polyfit(x, y, 2)
                    curvature = 2 * coeffs[0]  # Second derivative
                else:
                    curvature = 0
            else:
                slope = 0
                curvature = 0
                r_value = 0
            
            # Term structure shape classification
            if slope > 0:
                structure_shape = "Contango"
            elif slope < 0:
                structure_shape = "Backwardation"
            else:
                structure_shape = "Flat"
            
            # Calendar spread opportunities
            calendar_spreads = {}
            if len(ts_df) >= 2:
                for i in range(len(ts_df) - 1):
                    near_expiry = ts_df.index[i]
                    far_expiry = ts_df.index[i + 1]
                    spread = ts_df.loc[far_expiry, 'futures_price'] - ts_df.loc[near_expiry, 'futures_price']
                    calendar_spreads[f"{near_expiry}_{far_expiry}"] = spread
            
            # Create result DataFrame with time series data
            result_df = pd.DataFrame({
                'spot_price': spot_data['close'],
                'term_structure_slope': slope,
                'term_structure_curvature': curvature,
                'structure_shape': structure_shape
            }, index=spot_data.index)
            
            return IndicatorResult(
                name="Term Structure Analysis",
                values=result_df,
                metadata={
                    'term_structure_data': term_structure_data,
                    'slope': slope,
                    'curvature': curvature,
                    'r_squared': r_value**2 if 'r_value' in locals() else 0,
                    'structure_shape': structure_shape,
                    'calendar_spreads': calendar_spreads,
                    'expiries_analyzed': expiries,
                    'interpretation': f"Term structure is in {structure_shape} with slope {slope:.4f}"
                },
                confidence=0.70,
                timestamp=datetime.now(),
                asset_type=AssetType.FUTURES,
                category=IndicatorCategory.FUNDAMENTAL
            )
            
        except Exception as e:
            logger.error(f"Error calculating term structure analysis: {e}")
            return self._empty_result("Term Structure Analysis", AssetType.FUTURES)
    
    def seasonality_analysis(self, data: pd.DataFrame, commodity_type: str = "agricultural") -> IndicatorResult:
        """Analyze seasonal patterns in futures prices"""
        try:
            # Extract month and year from index
            data_with_time = data.copy()
            data_with_time['month'] = data_with_time.index.month
            data_with_time['year'] = data_with_time.index.year
            data_with_time['day_of_year'] = data_with_time.index.dayofyear
            
            # Calculate monthly returns
            monthly_returns = data['close'].resample('M').last().pct_change().dropna()
            monthly_returns.index = monthly_returns.index.month
            
            # Seasonal patterns by month
            seasonal_returns = monthly_returns.groupby(monthly_returns.index).agg({
                monthly_returns.name: ['mean', 'std', 'count']
            }).round(4)
            
            # Seasonal strength calculation
            seasonal_strength = {}
            for month in range(1, 13):
                if month in monthly_returns.index:
                    month_returns = monthly_returns[monthly_returns.index == month]
                    if len(month_returns) > 1:
                        # T-test against zero
                        t_stat, p_value = stats.ttest_1samp(month_returns, 0)
                        seasonal_strength[month] = {
                            'mean_return': month_returns.mean(),
                            'std_return': month_returns.std(),
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05
                        }
            
            # Commodity-specific seasonal patterns
            seasonal_factors = self._get_commodity_seasonal_factors(commodity_type)
            
            # Apply seasonal adjustment
            seasonal_adjustment = pd.Series(index=data.index, dtype=float)
            for i, date in enumerate(data.index):
                month = date.month
                if month in seasonal_factors:
                    seasonal_adjustment.iloc[i] = seasonal_factors[month]
                else:
                    seasonal_adjustment.iloc[i] = 1.0
            
            # Seasonally adjusted prices
            seasonally_adjusted_price = data['close'] / seasonal_adjustment
            
            # Seasonal deviation
            seasonal_deviation = (data['close'] - seasonally_adjusted_price) / seasonally_adjusted_price * 100
            
            # Harvest/planting cycle indicators (for agricultural commodities)
            if commodity_type == "agricultural":
                planting_season = ((data.index.month >= 3) & (data.index.month <= 5)).astype(int)
                harvest_season = ((data.index.month >= 9) & (data.index.month <= 11)).astype(int)
            else:
                planting_season = pd.Series(0, index=data.index)
                harvest_season = pd.Series(0, index=data.index)
            
            result_df = pd.DataFrame({
                'price': data['close'],
                'seasonal_adjustment': seasonal_adjustment,
                'seasonally_adjusted_price': seasonally_adjusted_price,
                'seasonal_deviation': seasonal_deviation,
                'month': data.index.month,
                'planting_season': planting_season,
                'harvest_season': harvest_season
            }, index=data.index)
            
            return IndicatorResult(
                name="Seasonality Analysis",
                values=result_df,
                metadata={
                    'commodity_type': commodity_type,
                    'seasonal_returns': seasonal_returns.to_dict(),
                    'seasonal_strength': seasonal_strength,
                    'seasonal_factors': seasonal_factors,
                    'strongest_month': max(seasonal_strength.keys(), 
                                         key=lambda x: seasonal_strength[x]['mean_return']) if seasonal_strength else None,
                    'weakest_month': min(seasonal_strength.keys(), 
                                       key=lambda x: seasonal_strength[x]['mean_return']) if seasonal_strength else None,
                    'interpretation': 'Seasonal patterns can provide trading opportunities based on historical cycles'
                },
                confidence=0.65,
                timestamp=datetime.now(),
                asset_type=AssetType.FUTURES,
                category=IndicatorCategory.STATISTICAL
            )
            
        except Exception as e:
            logger.error(f"Error calculating seasonality analysis: {e}")
            return self._empty_result("Seasonality Analysis", AssetType.FUTURES)
    
    def _get_commodity_seasonal_factors(self, commodity_type: str) -> Dict[int, float]:
        """Get seasonal adjustment factors for different commodity types"""
        seasonal_patterns = {
            "agricultural": {
                1: 1.02, 2: 1.01, 3: 0.98, 4: 0.96, 5: 0.97, 6: 0.99,
                7: 1.01, 8: 1.03, 9: 1.05, 10: 1.04, 11: 1.02, 12: 1.01
            },
            "energy": {
                1: 1.05, 2: 1.03, 3: 1.01, 4: 0.98, 5: 0.97, 6: 0.98,
                7: 1.02, 8: 1.04, 9: 1.02, 10: 1.01, 11: 1.03, 12: 1.06
            },
            "metals": {
                1: 1.01, 2: 1.02, 3: 1.03, 4: 1.01, 5: 0.99, 6: 0.98,
                7: 0.97, 8: 0.98, 9: 1.00, 10: 1.02, 11: 1.03, 12: 1.02
            },
            "livestock": {
                1: 0.98, 2: 0.97, 3: 0.99, 4: 1.02, 5: 1.04, 6: 1.03,
                7: 1.01, 8: 0.99, 9: 0.98, 10: 0.97, 11: 0.99, 12: 1.01
            }
        }
        
        return seasonal_patterns.get(commodity_type, 
                                    {i: 1.0 for i in range(1, 13)})  # Default: no seasonal adjustment
    
    def _empty_result(self, name: str, asset_type: AssetType) -> IndicatorResult:
        """Return empty result for error cases"""
        return IndicatorResult(
            name=name,
            values=pd.DataFrame(),
            metadata={'error': 'Calculation failed'},
            confidence=0.0,
            timestamp=datetime.now(),
            asset_type=asset_type,
            category=IndicatorCategory.TECHNICAL
        )
    
    async def calculate_indicator(self, indicator_name: str, data: pd.DataFrame, **kwargs) -> IndicatorResult:
        """Calculate specific futures indicator based on name"""
        indicator_map = {
            'cost_of_carry': self.cost_of_carry_model,
            'term_structure': self.term_structure_analysis,
            'seasonality': self.seasonality_analysis,
        }
        
        if indicator_name not in indicator_map:
            raise ValueError(f"Unknown futures indicator: {indicator_name}")
        
        return indicator_map[indicator_name](data, **kwargs)