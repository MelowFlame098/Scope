from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Try to import advanced libraries
try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import coint, adfuller
    from statsmodels.stats.diagnostic import het_breuschpagan
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Statsmodels not available. Using simplified statistical methods.")

@dataclass
class IndexData:
    """Structure for index market data"""
    prices: List[float]
    returns: List[float]
    volume: List[float]
    market_cap: List[float]
    timestamps: List[datetime]
    index_symbol: str
    constituent_weights: Optional[Dict[str, float]] = None
    sector_weights: Optional[Dict[str, float]] = None

@dataclass
class MacroeconomicFactors:
    """Structure for macroeconomic factors"""
    gdp_growth: List[float]
    inflation_rate: List[float]
    interest_rates: List[float]
    unemployment_rate: List[float]
    industrial_production: List[float]
    consumer_confidence: List[float]
    oil_prices: List[float]
    exchange_rates: List[float]
    vix_index: List[float]
    timestamps: List[datetime]

@dataclass
class FactorModelResult:
    """Results from factor model analysis"""
    factor_loadings: Dict[str, float]
    factor_returns: Dict[str, List[float]]
    explained_variance: float
    residual_variance: float
    alpha: float
    beta_coefficients: Dict[str, float]
    r_squared: float
    information_ratio: float
    tracking_error: float
    factor_exposures: List[Dict[str, float]]

class MacroeconomicFactorModel:
    """Macroeconomic factor model for index analysis"""
    
    def __init__(self, regularization: str = 'ridge', alpha: float = 1.0):
        self.regularization = regularization
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.model = None
        
    def fit_factor_model(self, index_data: IndexData, 
                        macro_factors: MacroeconomicFactors) -> FactorModelResult:
        """Fit macroeconomic factor model to index returns"""
        try:
            # Align data
            aligned_data = self._align_data(index_data, macro_factors)
            
            returns = aligned_data['returns']
            factors = aligned_data['factors']
            factor_names = aligned_data['factor_names']
            
            # Scale factors
            factors_scaled = self.scaler.fit_transform(factors)
            
            # Choose model based on regularization
            if self.regularization == 'ridge':
                model = Ridge(alpha=self.alpha)
            elif self.regularization == 'lasso':
                model = Lasso(alpha=self.alpha)
            else:
                model = LinearRegression()
            
            # Fit model
            model.fit(factors_scaled, returns)
            self.model = model
            
            # Calculate predictions and residuals
            predictions = model.predict(factors_scaled)
            residuals = returns - predictions
            
            # Calculate metrics
            r_squared = r2_score(returns, predictions)
            explained_variance = r_squared
            residual_variance = np.var(residuals)
            
            # Factor loadings (coefficients)
            factor_loadings = dict(zip(factor_names, model.coef_))
            
            # Alpha (intercept)
            alpha = model.intercept_
            
            # Beta coefficients (same as factor loadings for this model)
            beta_coefficients = factor_loadings.copy()
            
            # Information ratio and tracking error
            tracking_error = np.std(residuals) * np.sqrt(252)  # Annualized
            excess_return = np.mean(returns) - 0.02/252  # Assuming 2% risk-free rate
            information_ratio = excess_return / (tracking_error / np.sqrt(252)) if tracking_error > 0 else 0
            
            # Factor returns (for analysis)
            factor_returns = {}
            for i, name in enumerate(factor_names):
                factor_returns[name] = factors[:, i].tolist()
            
            # Factor exposures over time
            factor_exposures = []
            for t in range(len(returns)):
                exposure = {}
                for i, name in enumerate(factor_names):
                    exposure[name] = factors_scaled[t, i] * model.coef_[i]
                factor_exposures.append(exposure)
            
            return FactorModelResult(
                factor_loadings=factor_loadings,
                factor_returns=factor_returns,
                explained_variance=explained_variance,
                residual_variance=residual_variance,
                alpha=alpha,
                beta_coefficients=beta_coefficients,
                r_squared=r_squared,
                information_ratio=information_ratio,
                tracking_error=tracking_error,
                factor_exposures=factor_exposures
            )
            
        except Exception as e:
            print(f"Error in factor model fitting: {e}")
            # Return empty result
            return FactorModelResult(
                factor_loadings={},
                factor_returns={},
                explained_variance=0.0,
                residual_variance=1.0,
                alpha=0.0,
                beta_coefficients={},
                r_squared=0.0,
                information_ratio=0.0,
                tracking_error=1.0,
                factor_exposures=[]
            )
    
    def _align_data(self, index_data: IndexData, 
                   macro_factors: MacroeconomicFactors) -> Dict[str, Any]:
        """Align index and macroeconomic data by timestamps"""
        # Convert to pandas for easier alignment
        index_df = pd.DataFrame({
            'timestamp': index_data.timestamps,
            'returns': index_data.returns
        })
        
        macro_df = pd.DataFrame({
            'timestamp': macro_factors.timestamps,
            'gdp_growth': macro_factors.gdp_growth,
            'inflation_rate': macro_factors.inflation_rate,
            'interest_rates': macro_factors.interest_rates,
            'unemployment_rate': macro_factors.unemployment_rate,
            'industrial_production': macro_factors.industrial_production,
            'consumer_confidence': macro_factors.consumer_confidence,
            'oil_prices': macro_factors.oil_prices,
            'exchange_rates': macro_factors.exchange_rates,
            'vix_index': macro_factors.vix_index
        })
        
        # Merge on timestamp
        merged_df = pd.merge(index_df, macro_df, on='timestamp', how='inner')
        
        # Extract aligned data
        returns = merged_df['returns'].values
        
        factor_names = ['gdp_growth', 'inflation_rate', 'interest_rates', 
                       'unemployment_rate', 'industrial_production', 
                       'consumer_confidence', 'oil_prices', 'exchange_rates', 'vix_index']
        
        factors = merged_df[factor_names].values
        
        return {
            'returns': returns,
            'factors': factors,
            'factor_names': factor_names,
            'timestamps': merged_df['timestamp'].tolist()
        }