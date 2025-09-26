import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Suppress warnings
warnings.filterwarnings('ignore')

@dataclass
class CapitalAccountData:
    """Capital and financial account data."""
    capital_account: pd.Series  # Capital transfers
    direct_investment: pd.Series  # FDI flows
    portfolio_investment: pd.Series  # Portfolio flows
    other_investment: pd.Series  # Bank flows, trade credits
    reserve_assets: pd.Series  # Central bank reserves
    financial_account_balance: pd.Series  # Total financial account

class FinancialAccountAnalyzer:
    """Financial account analysis implementation."""
    
    def __init__(self):
        self.data = None
        
    def analyze_financial_account(self, fa_data: CapitalAccountData,
                                 exchange_rates: pd.Series = None) -> Dict[str, Any]:
        """Analyze financial account flows and patterns."""
        self.data = fa_data
        
        # Capital flows analysis
        capital_flows = self._analyze_capital_flows()
        
        # Flow volatility analysis
        flow_volatility = self._analyze_flow_volatility()
        
        # Flow composition analysis
        flow_composition = self._analyze_flow_composition()
        
        # Hot money flows analysis
        hot_money = self._analyze_hot_money_flows()
        
        # Reserve changes analysis
        reserve_analysis = self._analyze_reserve_changes()
        
        # Crisis indicators
        crisis_indicators = self._detect_crisis_indicators()
        
        return {
            'capital_flows': capital_flows,
            'flow_volatility': flow_volatility,
            'flow_composition': flow_composition,
            'hot_money_flows': hot_money,
            'reserve_analysis': reserve_analysis,
            'crisis_indicators': crisis_indicators
        }
        
    def _analyze_capital_flows(self) -> Dict[str, Any]:
        """Analyze capital flow patterns."""
        return {
            'fdi_analysis': {
                'trend': self._calculate_trend(self.data.direct_investment),
                'volatility': self.data.direct_investment.std(),
                'average_flow': self.data.direct_investment.mean()
            },
            'portfolio_analysis': {
                'trend': self._calculate_trend(self.data.portfolio_investment),
                'volatility': self.data.portfolio_investment.std(),
                'average_flow': self.data.portfolio_investment.mean()
            },
            'other_investment_analysis': {
                'trend': self._calculate_trend(self.data.other_investment),
                'volatility': self.data.other_investment.std(),
                'average_flow': self.data.other_investment.mean()
            },
            'total_flows': {
                'trend': self._calculate_trend(self.data.financial_account_balance),
                'volatility': self.data.financial_account_balance.std(),
                'average_flow': self.data.financial_account_balance.mean()
            }
        }
        
    def _analyze_flow_volatility(self) -> Dict[str, Any]:
        """Analyze volatility patterns in capital flows."""
        return {
            'fdi_volatility': self._classify_flow_volatility(self.data.direct_investment.pct_change()),
            'portfolio_volatility': self._classify_flow_volatility(self.data.portfolio_investment.pct_change()),
            'other_investment_volatility': self._classify_flow_volatility(self.data.other_investment.pct_change()),
            'overall_volatility': self._classify_flow_volatility(self.data.financial_account_balance.pct_change()),
            'volatility_components': {
                'fdi': self.data.direct_investment.pct_change().std(),
                'portfolio': self.data.portfolio_investment.pct_change().std(),
                'other_investment': self.data.other_investment.pct_change().std(),
                'reserves': self.data.reserve_assets.pct_change().std()
            }
        }
        
    def _classify_flow_volatility(self, changes: pd.Series) -> str:
        """Classify flow volatility level."""
        try:
            changes_clean = changes.dropna()
            if len(changes_clean) < 5:
                return 'insufficient_data'
                
            volatility = changes_clean.std()
            mean_abs_change = changes_clean.abs().mean()
            
            # Classification thresholds
            if volatility > 3 * mean_abs_change:
                return 'very_high'
            elif volatility > 2 * mean_abs_change:
                return 'high'
            elif volatility > mean_abs_change:
                return 'moderate'
            else:
                return 'low'
        except:
            return 'unknown'
            
    def _analyze_flow_composition(self) -> Dict[str, Any]:
        """Analyze composition of capital flows."""
        try:
            # Calculate average composition
            total_inflows = (self.data.direct_investment + 
                           self.data.portfolio_investment + 
                           self.data.other_investment).abs()
            
            if total_inflows.sum() == 0:
                return {'composition_analysis': 'no_flows'}
                
            fdi_share = (self.data.direct_investment.abs() / total_inflows).mean()
            portfolio_share = (self.data.portfolio_investment.abs() / total_inflows).mean()
            other_share = (self.data.other_investment.abs() / total_inflows).mean()
            
            # Determine dominant flow type
            shares = {'fdi': fdi_share, 'portfolio': portfolio_share, 'other': other_share}
            dominant_flow = max(shares, key=shares.get)
            
            return {
                'composition': {
                    'fdi_share': fdi_share,
                    'portfolio_share': portfolio_share,
                    'other_investment_share': other_share
                },
                'dominant_flow_type': dominant_flow,
                'flow_stability': {
                    'fdi_stability': 'stable' if fdi_share > 0.4 else 'volatile',
                    'portfolio_stability': 'stable' if portfolio_share < 0.6 else 'volatile',
                    'overall_stability': 'stable' if max(shares.values()) < 0.7 else 'concentrated'
                }
            }
        except:
            return {'composition_analysis': 'failed'}
            
    def _analyze_hot_money_flows(self) -> Dict[str, Any]:
        """Analyze hot money (volatile short-term) flows."""
        try:
            # Portfolio investment is typically more volatile (hot money)
            portfolio_flows = self.data.portfolio_investment
            
            # Calculate hot money indicators
            hot_money_indicator = self._calculate_hot_money_indicator(portfolio_flows)
            sudden_stops = self._detect_sudden_stops(portfolio_flows)
            
            return {
                'hot_money_indicator': hot_money_indicator,
                'sudden_stops': sudden_stops,
                'portfolio_flow_trend': self._calculate_trend(portfolio_flows),
                'flow_reversals': self._count_flow_reversals(portfolio_flows)
            }
        except:
            return {'hot_money_analysis': 'failed'}
            
    def _calculate_hot_money_indicator(self, flows: pd.Series) -> float:
        """Calculate hot money flow indicator."""
        try:
            # Measure of flow volatility relative to average flow
            flow_changes = flows.pct_change().dropna()
            
            if len(flow_changes) < 5:
                return 0.0
                
            volatility = flow_changes.std()
            mean_abs_flow = flows.abs().mean()
            
            if mean_abs_flow == 0:
                return 0.0
                
            # Normalize indicator (0-100 scale)
            indicator = min(100, (volatility / (mean_abs_flow / 100)) * 10)
            return indicator
        except:
            return 0.0
            
    def _detect_sudden_stops(self, flows: pd.Series) -> Dict[str, Any]:
        """Detect sudden stops in capital flows."""
        try:
            flow_changes = flows.pct_change().dropna()
            
            # Define sudden stop as large negative change
            threshold = flow_changes.quantile(0.1)  # Bottom 10%
            sudden_stops = flow_changes[flow_changes < threshold]
            
            return {
                'sudden_stop_count': len(sudden_stops),
                'sudden_stop_dates': sudden_stops.index.tolist(),
                'average_stop_magnitude': sudden_stops.mean() if len(sudden_stops) > 0 else 0.0
            }
        except:
            return {'sudden_stops': 'detection_failed'}
            
    def _count_flow_reversals(self, flows: pd.Series) -> int:
        """Count flow direction reversals."""
        try:
            flow_signs = np.sign(flows)
            reversals = (flow_signs.diff() != 0).sum()
            return int(reversals)
        except:
            return 0
            
    def _analyze_reserve_changes(self) -> Dict[str, Any]:
        """Analyze central bank reserve changes."""
        try:
            reserves = self.data.reserve_assets
            reserve_changes = reserves.diff()
            
            return {
                'reserve_trend': self._calculate_trend(reserves),
                'reserve_volatility': reserve_changes.std(),
                'reserve_accumulation_periods': len(reserve_changes[reserve_changes > 0]),
                'reserve_depletion_periods': len(reserve_changes[reserve_changes < 0]),
                'largest_accumulation': reserve_changes.max(),
                'largest_depletion': reserve_changes.min(),
                'intervention_intensity': {
                    'high_intervention_periods': len(reserve_changes[reserve_changes.abs() > reserve_changes.std() * 2]),
                    'average_intervention': reserve_changes.abs().mean()
                }
            }
        except:
            return {'reserve_analysis': 'failed'}
            
    def _detect_crisis_indicators(self) -> Dict[str, Any]:
        """Detect financial crisis indicators."""
        try:
            # Multiple crisis indicators
            indicators = {}
            
            # Sudden stop indicator
            portfolio_changes = self.data.portfolio_investment.pct_change().dropna()
            sudden_stop_threshold = portfolio_changes.quantile(0.05)
            indicators['sudden_stops'] = len(portfolio_changes[portfolio_changes < sudden_stop_threshold])
            
            # Reserve depletion indicator
            reserve_changes = self.data.reserve_assets.diff()
            large_depletion_threshold = reserve_changes.quantile(0.1)
            indicators['reserve_depletions'] = len(reserve_changes[reserve_changes < large_depletion_threshold])
            
            # Flow reversal indicator
            total_flows = self.data.financial_account_balance
            flow_reversals = self._count_flow_reversals(total_flows)
            indicators['flow_reversals'] = flow_reversals
            
            # Overall crisis risk
            crisis_score = (indicators['sudden_stops'] * 0.4 + 
                          indicators['reserve_depletions'] * 0.3 + 
                          min(flow_reversals, 10) * 0.3)
            
            risk_level = 'low'
            if crisis_score > 5:
                risk_level = 'high'
            elif crisis_score > 2:
                risk_level = 'moderate'
                
            return {
                'crisis_indicators': indicators,
                'crisis_score': crisis_score,
                'risk_level': risk_level
            }
        except:
            return {'crisis_indicators': 'detection_failed'}
            
    def _calculate_trend(self, series: pd.Series) -> float:
        """Calculate trend using linear regression."""
        try:
            if len(series.dropna()) < 2:
                return 0.0
            
            x = np.arange(len(series))
            y = series.values
            
            # Remove NaN values
            mask = ~np.isnan(y)
            if np.sum(mask) < 2:
                return 0.0
                
            slope, _, _, _, _ = stats.linregress(x[mask], y[mask])
            return slope
        except:
            return 0.0