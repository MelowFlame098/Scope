from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA, FactorAnalysis
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
class APTResult:
    """Results from Arbitrage Pricing Theory analysis"""
    factor_premiums: Dict[str, float]
    expected_returns: List[float]
    systematic_risk: List[float]
    idiosyncratic_risk: List[float]
    arbitrage_opportunities: List[Dict[str, Any]]
    model_fit: Dict[str, float]
    factor_significance: Dict[str, float]
    residual_analysis: Dict[str, Any]

class ArbitragePricingTheory:
    """Arbitrage Pricing Theory implementation for multi-factor analysis"""
    
    def __init__(self, n_factors: int = 5):
        self.n_factors = n_factors
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_factors)
        
    def analyze_apt(self, index_data: IndexData, 
                   macro_factors: MacroeconomicFactors) -> APTResult:
        """Perform APT analysis on index data"""
        try:
            # Align data
            aligned_data = self._align_data(index_data, macro_factors)
            
            returns = aligned_data['returns']
            factors = aligned_data['factors']
            factor_names = aligned_data['factor_names']
            
            # Apply PCA to reduce dimensionality and identify key factors
            factors_scaled = self.scaler.fit_transform(factors)
            principal_factors = self.pca.fit_transform(factors_scaled)
            
            # Fit APT model
            model = LinearRegression()
            model.fit(principal_factors, returns)
            
            # Calculate predictions and residuals
            predictions = model.predict(principal_factors)
            residuals = returns - predictions
            
            # Calculate factor premiums (risk premiums)
            factor_premiums = {}
            for i in range(self.n_factors):
                factor_premiums[f'Factor_{i+1}'] = model.coef_[i]
            
            # Expected returns based on factor loadings
            expected_returns = predictions.tolist()
            
            # Risk decomposition
            systematic_risk = np.var(predictions, ddof=1)
            idiosyncratic_risk = np.var(residuals, ddof=1)
            total_risk = systematic_risk + idiosyncratic_risk
            
            systematic_risk_list = [systematic_risk] * len(returns)
            idiosyncratic_risk_list = [idiosyncratic_risk] * len(returns)
            
            # Model fit metrics
            r_squared = r2_score(returns, predictions)
            mse = mean_squared_error(returns, predictions)
            
            model_fit = {
                'r_squared': r_squared,
                'mse': mse,
                'systematic_risk_ratio': systematic_risk / total_risk,
                'idiosyncratic_risk_ratio': idiosyncratic_risk / total_risk
            }
            
            # Identify arbitrage opportunities
            arbitrage_opportunities = self._identify_arbitrage_opportunities(
                returns, expected_returns, residuals
            )
            
            # Factor significance testing
            factor_significance = self._calculate_factor_significance(
                principal_factors, returns, model
            )
            
            # Residual analysis
            residual_analysis = self._analyze_residuals(residuals)
            
            return APTResult(
                factor_premiums=factor_premiums,
                expected_returns=expected_returns,
                systematic_risk=systematic_risk_list,
                idiosyncratic_risk=idiosyncratic_risk_list,
                arbitrage_opportunities=arbitrage_opportunities,
                model_fit=model_fit,
                factor_significance=factor_significance,
                residual_analysis=residual_analysis
            )
            
        except Exception as e:
            print(f"Error in APT analysis: {e}")
            # Return empty result
            return APTResult(
                factor_premiums={},
                expected_returns=[],
                systematic_risk=[],
                idiosyncratic_risk=[],
                arbitrage_opportunities=[],
                model_fit={'r_squared': 0.0, 'mse': 1.0},
                factor_significance={},
                residual_analysis={}
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
    
    def _identify_arbitrage_opportunities(self, actual_returns: np.ndarray,
                                        expected_returns: List[float],
                                        residuals: np.ndarray) -> List[Dict[str, Any]]:
        """Identify potential arbitrage opportunities"""
        opportunities = []
        
        # Look for periods with large residuals (mispricing)
        threshold = 2 * np.std(residuals)
        
        for i, (actual, expected, residual) in enumerate(zip(actual_returns, expected_returns, residuals)):
            if abs(residual) > threshold:
                opportunities.append({
                    'period': i,
                    'actual_return': actual,
                    'expected_return': expected,
                    'mispricing': residual,
                    'opportunity_type': 'overvalued' if residual < 0 else 'undervalued',
                    'confidence': min(abs(residual) / threshold, 3.0)  # Cap at 3x threshold
                })
        
        return opportunities
    
    def _calculate_factor_significance(self, factors: np.ndarray, 
                                     returns: np.ndarray, 
                                     model: LinearRegression) -> Dict[str, float]:
        """Calculate statistical significance of factors"""
        significance = {}
        
        try:
            # Calculate t-statistics for each factor
            n = len(returns)
            k = factors.shape[1]
            
            # Residual sum of squares
            predictions = model.predict(factors)
            residuals = returns - predictions
            rss = np.sum(residuals**2)
            
            # Standard errors
            mse = rss / (n - k - 1)
            var_covar_matrix = mse * np.linalg.inv(factors.T @ factors)
            standard_errors = np.sqrt(np.diag(var_covar_matrix))
            
            # T-statistics and p-values
            t_stats = model.coef_ / standard_errors
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k - 1))
            
            for i in range(len(t_stats)):
                significance[f'Factor_{i+1}'] = p_values[i]
                
        except Exception as e:
            print(f"Error calculating factor significance: {e}")
            for i in range(self.n_factors):
                significance[f'Factor_{i+1}'] = 1.0  # No significance
        
        return significance
    
    def _analyze_residuals(self, residuals: np.ndarray) -> Dict[str, Any]:
        """Analyze model residuals for diagnostics"""
        analysis = {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'skewness': stats.skew(residuals),
            'kurtosis': stats.kurtosis(residuals),
            'jarque_bera_pvalue': stats.jarque_bera(residuals)[1],
            'autocorrelation': self._calculate_autocorrelation(residuals)
        }
        
        return analysis
    
    def _calculate_autocorrelation(self, residuals: np.ndarray, max_lags: int = 5) -> Dict[str, float]:
        """Calculate autocorrelation of residuals"""
        autocorr = {}
        
        for lag in range(1, max_lags + 1):
            if len(residuals) > lag:
                corr = np.corrcoef(residuals[:-lag], residuals[lag:])[0, 1]
                autocorr[f'lag_{lag}'] = corr if not np.isnan(corr) else 0.0
            else:
                autocorr[f'lag_{lag}'] = 0.0
        
        return autocorr