from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, optimize
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

@dataclass
class FuturesContractData:
    """Structure for individual futures contract data"""
    contract_symbol: str
    expiry_date: datetime
    prices: List[float]
    timestamps: List[datetime]
    volume: List[float]
    open_interest: List[float]
    time_to_maturity: List[float]  # in days
    
@dataclass
class SamuelsonResult:
    """Results from Samuelson Effect analysis"""
    volatility_profile: List[float]
    time_to_maturity: List[float]
    samuelson_coefficient: float
    volatility_increase_rate: float
    r_squared: float
    statistical_significance: bool
    maturity_effect_strength: str
    
@dataclass
class BackwardationContangoResult:
    """Results from backwardation/contango analysis"""
    market_structure: List[str]  # 'Backwardation', 'Contango', 'Neutral'
    basis_values: List[float]  # Futures - Spot
    term_structure_slope: List[float]
    convenience_yield_proxy: List[float]
    storage_cost_proxy: List[float]
    supply_demand_indicator: List[str]
    seasonal_patterns: Dict[str, float]
    
@dataclass
class TermStructureResult:
    """Results from term structure analysis"""
    curve_shape: str  # 'Normal', 'Inverted', 'Humped', 'Flat'
    slope_coefficient: float
    curvature_coefficient: float
    butterfly_spread: List[float]
    calendar_spreads: List[float]
    roll_yield: List[float]
    carry_return: List[float]
    
@dataclass
class FuturesStructureResult:
    """Comprehensive futures structure analysis results"""
    samuelson_results: SamuelsonResult
    backwardation_contango_results: BackwardationContangoResult
    term_structure_results: TermStructureResult
    trading_signals: List[str]
    risk_metrics: Dict[str, float]
    insights: List[str]
    recommendations: List[str]
    model_performance: Dict[str, float]

class SamuelsonEffectAnalyzer:
    """Analyzer for Samuelson Effect - volatility increases as maturity approaches"""
    
    def __init__(self):
        self.model_name = "Samuelson Effect Analyzer"
        
    def calculate_volatility_profile(self, contract_data: FuturesContractData, 
                                   window: int = 20) -> Tuple[List[float], List[float]]:
        """Calculate rolling volatility profile against time to maturity"""
        if len(contract_data.prices) < window:
            return [], []
        
        volatilities = []
        time_to_maturity = []
        
        # Calculate rolling volatility
        for i in range(window, len(contract_data.prices)):
            price_window = contract_data.prices[i-window:i]
            returns = [np.log(price_window[j]/price_window[j-1]) for j in range(1, len(price_window))]
            
            if returns:
                vol = np.std(returns) * np.sqrt(252)  # Annualized volatility
                volatilities.append(vol)
                time_to_maturity.append(contract_data.time_to_maturity[i])
        
        return volatilities, time_to_maturity
    
    def analyze_samuelson_effect(self, contract_data: FuturesContractData) -> SamuelsonResult:
        """Analyze Samuelson Effect in futures volatility"""
        volatilities, time_to_maturity = self.calculate_volatility_profile(contract_data)
        
        if len(volatilities) < 10:
            return SamuelsonResult(
                volatility_profile=volatilities,
                time_to_maturity=time_to_maturity,
                samuelson_coefficient=0.0,
                volatility_increase_rate=0.0,
                r_squared=0.0,
                statistical_significance=False,
                maturity_effect_strength="Insufficient Data"
            )
        
        # Fit exponential decay model: volatility = a * exp(-b * time_to_maturity) + c
        # Or linear model: volatility = a + b * (1/time_to_maturity)
        
        # Prepare data for regression
        X = np.array([[1/max(ttm, 1)] for ttm in time_to_maturity])  # Inverse time to maturity
        y = np.array(volatilities)
        
        # Fit linear regression
        reg = LinearRegression()
        reg.fit(X, y)
        
        samuelson_coefficient = reg.coef_[0]
        r_squared = reg.score(X, y)
        
        # Statistical significance test
        n = len(volatilities)
        if n > 2:
            t_stat = samuelson_coefficient / (np.std(y) / np.sqrt(n-2))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n-2))
            statistical_significance = p_value < 0.05
        else:
            statistical_significance = False
        
        # Calculate volatility increase rate
        if len(volatilities) > 1:
            early_vol = np.mean(volatilities[:len(volatilities)//3])
            late_vol = np.mean(volatilities[-len(volatilities)//3:])
            volatility_increase_rate = (late_vol - early_vol) / early_vol if early_vol > 0 else 0
        else:
            volatility_increase_rate = 0
        
        # Determine effect strength
        if abs(samuelson_coefficient) > 0.1 and statistical_significance:
            if samuelson_coefficient > 0:
                strength = "Strong Positive"
            else:
                strength = "Strong Negative"
        elif abs(samuelson_coefficient) > 0.05:
            strength = "Moderate"
        else:
            strength = "Weak"
        
        return SamuelsonResult(
            volatility_profile=volatilities,
            time_to_maturity=time_to_maturity,
            samuelson_coefficient=samuelson_coefficient,
            volatility_increase_rate=volatility_increase_rate,
            r_squared=r_squared,
            statistical_significance=statistical_significance,
            maturity_effect_strength=strength
        )

class BackwardationContangoAnalyzer:
    """Analyzer for backwardation and contango patterns"""
    
    def __init__(self):
        self.model_name = "Backwardation/Contango Analyzer"
        
    def analyze_market_structure(self, spot_prices: List[float], 
                               futures_contracts: List[FuturesContractData]) -> BackwardationContangoResult:
        """Analyze backwardation and contango patterns"""
        
        if not futures_contracts or len(spot_prices) == 0:
            return BackwardationContangoResult(
                market_structure=[],
                basis_values=[],
                term_structure_slope=[],
                convenience_yield_proxy=[],
                storage_cost_proxy=[],
                supply_demand_indicator=[],
                seasonal_patterns={}
            )
        
        # Analyze each time period
        market_structure = []
        basis_values = []
        term_structure_slopes = []
        convenience_yield_proxies = []
        storage_cost_proxies = []
        supply_demand_indicators = []
        
        # Get the shortest contract for consistent analysis
        min_length = min(len(contract.prices) for contract in futures_contracts)
        min_length = min(min_length, len(spot_prices))
        
        for i in range(min_length):
            spot_price = spot_prices[i]
            
            # Get futures prices for this time period
            futures_prices = []
            maturities = []
            
            for contract in futures_contracts:
                if i < len(contract.prices) and i < len(contract.time_to_maturity):
                    futures_prices.append(contract.prices[i])
                    maturities.append(contract.time_to_maturity[i])
            
            if not futures_prices:
                continue
            
            # Calculate basis (futures - spot)
            nearest_futures = min(futures_prices) if futures_prices else spot_price
            basis = nearest_futures - spot_price
            basis_values.append(basis)
            
            # Determine market structure
            if basis < -0.01 * spot_price:  # Threshold: 1% of spot price
                structure = "Backwardation"
                supply_demand = "Supply Shortage"
            elif basis > 0.01 * spot_price:
                structure = "Contango"
                supply_demand = "Supply Abundance"
            else:
                structure = "Neutral"
                supply_demand = "Balanced"
            
            market_structure.append(structure)
            supply_demand_indicators.append(supply_demand)
            
            # Calculate term structure slope
            if len(futures_prices) >= 2 and len(maturities) >= 2:
                # Sort by maturity
                sorted_data = sorted(zip(maturities, futures_prices))
                maturities_sorted = [x[0] for x in sorted_data]
                prices_sorted = [x[1] for x in sorted_data]
                
                # Calculate slope (price change per unit time)
                if len(prices_sorted) >= 2:
                    slope = (prices_sorted[-1] - prices_sorted[0]) / max(maturities_sorted[-1] - maturities_sorted[0], 1)
                    term_structure_slopes.append(slope)
                else:
                    term_structure_slopes.append(0)
            else:
                term_structure_slopes.append(0)
            
            # Proxy for convenience yield (negative basis normalized by spot price)
            convenience_yield_proxy = -basis / spot_price if spot_price > 0 else 0
            convenience_yield_proxies.append(convenience_yield_proxy)
            
            # Proxy for storage cost (positive basis normalized by spot price)
            storage_cost_proxy = max(basis, 0) / spot_price if spot_price > 0 else 0
            storage_cost_proxies.append(storage_cost_proxy)
        
        # Analyze seasonal patterns
        seasonal_patterns = self._analyze_seasonal_patterns(
            [contract.timestamps for contract in futures_contracts if contract.timestamps],
            basis_values
        )
        
        return BackwardationContangoResult(
            market_structure=market_structure,
            basis_values=basis_values,
            term_structure_slope=term_structure_slopes,
            convenience_yield_proxy=convenience_yield_proxies,
            storage_cost_proxy=storage_cost_proxies,
            supply_demand_indicator=supply_demand_indicators,
            seasonal_patterns=seasonal_patterns
        )
    
    def _analyze_seasonal_patterns(self, timestamps_list: List[List[datetime]], 
                                 basis_values: List[float]) -> Dict[str, float]:
        """Analyze seasonal patterns in basis"""
        if not timestamps_list or not basis_values:
            return {}
        
        # Use the first timestamp list that has data
        timestamps = None
        for ts_list in timestamps_list:
            if ts_list and len(ts_list) >= len(basis_values):
                timestamps = ts_list[:len(basis_values)]
                break
        
        if not timestamps:
            return {}
        
        # Group by month
        monthly_basis = {}
        for timestamp, basis in zip(timestamps, basis_values):
            month = timestamp.month
            if month not in monthly_basis:
                monthly_basis[month] = []
            monthly_basis[month].append(basis)
        
        # Calculate average basis by month
        seasonal_patterns = {}
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        for month, basis_list in monthly_basis.items():
            if basis_list:
                seasonal_patterns[month_names[month-1]] = np.mean(basis_list)
        
        return seasonal_patterns

class TermStructureAnalyzer:
    """Analyzer for futures term structure"""
    
    def __init__(self):
        self.model_name = "Term Structure Analyzer"
        
    def analyze_term_structure(self, futures_contracts: List[FuturesContractData]) -> TermStructureResult:
        """Analyze futures term structure characteristics"""
        
        if len(futures_contracts) < 2:
            return TermStructureResult(
                curve_shape="Insufficient Data",
                slope_coefficient=0.0,
                curvature_coefficient=0.0,
                butterfly_spread=[],
                calendar_spreads=[],
                roll_yield=[],
                carry_return=[]
            )
        
        # Sort contracts by time to maturity
        contracts_sorted = sorted(futures_contracts, 
                                key=lambda x: np.mean(x.time_to_maturity) if x.time_to_maturity else 0)
        
        # Analyze term structure for each time period
        min_length = min(len(contract.prices) for contract in contracts_sorted)
        
        curve_shapes = []
        butterfly_spreads = []
        calendar_spreads = []
        roll_yields = []
        carry_returns = []
        
        for i in range(min_length):
            # Get prices and maturities for this time period
            prices = []
            maturities = []
            
            for contract in contracts_sorted:
                if i < len(contract.prices) and i < len(contract.time_to_maturity):
                    prices.append(contract.prices[i])
                    maturities.append(contract.time_to_maturity[i])
            
            if len(prices) < 3:
                continue
            
            # Analyze curve shape
            curve_shape = self._determine_curve_shape(prices, maturities)
            curve_shapes.append(curve_shape)
            
            # Calculate butterfly spread (if we have at least 3 contracts)
            if len(prices) >= 3:
                butterfly = prices[0] + prices[2] - 2 * prices[1]
                butterfly_spreads.append(butterfly)
            
            # Calculate calendar spreads
            if len(prices) >= 2:
                calendar_spread = prices[1] - prices[0]  # Far - Near
                calendar_spreads.append(calendar_spread)
            
            # Calculate roll yield (return from rolling contracts)
            if len(prices) >= 2 and maturities[1] > maturities[0]:
                time_diff = max(maturities[1] - maturities[0], 1)
                roll_yield = (prices[0] - prices[1]) / prices[1] * (365 / time_diff)
                roll_yields.append(roll_yield)
            
            # Calculate carry return
            if len(prices) >= 2:
                carry_return = (prices[1] - prices[0]) / prices[0]
                carry_returns.append(carry_return)
        
        # Determine overall curve characteristics
        if curve_shapes:
            from collections import Counter
            most_common_shape = Counter(curve_shapes).most_common(1)[0][0]
        else:
            most_common_shape = "Unknown"
        
        # Calculate slope and curvature coefficients
        slope_coeff, curvature_coeff = self._calculate_curve_coefficients(contracts_sorted)
        
        return TermStructureResult(
            curve_shape=most_common_shape,
            slope_coefficient=slope_coeff,
            curvature_coefficient=curvature_coeff,
            butterfly_spread=butterfly_spreads,
            calendar_spreads=calendar_spreads,
            roll_yield=roll_yields,
            carry_return=carry_returns
        )
    
    def _determine_curve_shape(self, prices: List[float], maturities: List[float]) -> str:
        """Determine the shape of the term structure curve"""
        if len(prices) < 3:
            return "Unknown"
        
        # Calculate first and second derivatives
        first_diffs = [prices[i+1] - prices[i] for i in range(len(prices)-1)]
        
        if len(first_diffs) < 2:
            return "Unknown"
        
        second_diffs = [first_diffs[i+1] - first_diffs[i] for i in range(len(first_diffs)-1)]
        
        # Analyze slope
        avg_first_diff = np.mean(first_diffs)
        avg_second_diff = np.mean(second_diffs) if second_diffs else 0
        
        # Determine shape
        if avg_first_diff > 0.1:
            if abs(avg_second_diff) < 0.05:
                return "Normal"  # Upward sloping
            elif avg_second_diff < -0.05:
                return "Humped"  # Upward then downward
            else:
                return "Normal"
        elif avg_first_diff < -0.1:
            return "Inverted"  # Downward sloping
        else:
            return "Flat"  # Relatively flat
    
    def _calculate_curve_coefficients(self, contracts: List[FuturesContractData]) -> Tuple[float, float]:
        """Calculate slope and curvature coefficients using polynomial fitting"""
        if len(contracts) < 3:
            return 0.0, 0.0
        
        try:
            # Use average prices and maturities
            avg_prices = []
            avg_maturities = []
            
            for contract in contracts:
                if contract.prices and contract.time_to_maturity:
                    avg_prices.append(np.mean(contract.prices))
                    avg_maturities.append(np.mean(contract.time_to_maturity))
            
            if len(avg_prices) < 3:
                return 0.0, 0.0
            
            # Fit polynomial: price = a + b*maturity + c*maturity^2
            X = np.array(avg_maturities).reshape(-1, 1)
            poly_features = PolynomialFeatures(degree=2)
            X_poly = poly_features.fit_transform(X)
            
            reg = LinearRegression()
            reg.fit(X_poly, avg_prices)
            
            # Coefficients: [intercept, linear, quadratic]
            coeffs = reg.coef_
            
            slope_coeff = coeffs[1] if len(coeffs) > 1 else 0.0
            curvature_coeff = coeffs[2] if len(coeffs) > 2 else 0.0
            
            return slope_coeff, curvature_coeff
            
        except Exception:
            return 0.0, 0.0

class FuturesStructureAnalyzer:
    """Comprehensive analyzer for futures market structure"""
    
    def __init__(self):
        self.samuelson_analyzer = SamuelsonEffectAnalyzer()
        self.backwardation_analyzer = BackwardationContangoAnalyzer()
        self.term_structure_analyzer = TermStructureAnalyzer()
        
    def analyze(self, spot_prices: List[float], 
               futures_contracts: List[FuturesContractData]) -> FuturesStructureResult:
        """Perform comprehensive futures structure analysis"""
        
        print("Analyzing futures market structure...")
        
        # Samuelson Effect analysis (use the contract with most data)
        if futures_contracts:
            main_contract = max(futures_contracts, key=lambda x: len(x.prices))
            samuelson_results = self.samuelson_analyzer.analyze_samuelson_effect(main_contract)
        else:
            samuelson_results = SamuelsonResult(
                volatility_profile=[], time_to_maturity=[], samuelson_coefficient=0.0,
                volatility_increase_rate=0.0, r_squared=0.0, statistical_significance=False,
                maturity_effect_strength="No Data"
            )
        
        # Backwardation/Contango analysis
        backwardation_results = self.backwardation_analyzer.analyze_market_structure(
            spot_prices, futures_contracts
        )
        
        # Term structure analysis
        term_structure_results = self.term_structure_analyzer.analyze_term_structure(futures_contracts)
        
        # Generate trading signals
        trading_signals = self._generate_trading_signals(
            samuelson_results, backwardation_results, term_structure_results
        )
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(
            samuelson_results, backwardation_results, term_structure_results
        )
        
        # Generate insights
        insights = self._generate_insights(
            samuelson_results, backwardation_results, term_structure_results
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            samuelson_results, backwardation_results, term_structure_results, risk_metrics
        )
        
        # Calculate model performance
        model_performance = self._calculate_model_performance(
            samuelson_results, backwardation_results, term_structure_results
        )
        
        return FuturesStructureResult(
            samuelson_results=samuelson_results,
            backwardation_contango_results=backwardation_results,
            term_structure_results=term_structure_results,
            trading_signals=trading_signals,
            risk_metrics=risk_metrics,
            insights=insights,
            recommendations=recommendations,
            model_performance=model_performance
        )
    
    def _generate_trading_signals(self, samuelson: SamuelsonResult,
                                backwardation: BackwardationContangoResult,
                                term_structure: TermStructureResult) -> List[str]:
        """Generate trading signals based on structure analysis"""
        signals = []
        
        # Signals based on market structure
        for i, structure in enumerate(backwardation.market_structure):
            signal = "HOLD"
            
            # Backwardation signals
            if structure == "Backwardation":
                if i < len(backwardation.convenience_yield_proxy) and backwardation.convenience_yield_proxy[i] > 0.02:
                    signal = "BUY_SPOT_SELL_FUTURES"  # Strong convenience yield
                else:
                    signal = "BUY_FUTURES"  # Expect convergence
            
            # Contango signals
            elif structure == "Contango":
                if i < len(term_structure.roll_yield) and term_structure.roll_yield and term_structure.roll_yield[i] > 0.05:
                    signal = "SELL_FUTURES"  # Negative roll yield
                else:
                    signal = "BUY_SPOT"  # Storage arbitrage
            
            # Samuelson effect signals
            if samuelson.maturity_effect_strength in ["Strong Positive", "Strong Negative"]:
                if len(samuelson.time_to_maturity) > i:
                    ttm = samuelson.time_to_maturity[i] if i < len(samuelson.time_to_maturity) else 30
                    if ttm < 30:  # Near expiry
                        signal = "REDUCE_POSITION"  # High volatility risk
            
            signals.append(signal)
        
        return signals
    
    def _calculate_risk_metrics(self, samuelson: SamuelsonResult,
                              backwardation: BackwardationContangoResult,
                              term_structure: TermStructureResult) -> Dict[str, float]:
        """Calculate risk metrics"""
        metrics = {}
        
        # Samuelson effect metrics
        if samuelson.volatility_profile:
            metrics['volatility_mean'] = np.mean(samuelson.volatility_profile)
            metrics['volatility_std'] = np.std(samuelson.volatility_profile)
            metrics['samuelson_strength'] = abs(samuelson.samuelson_coefficient)
        
        # Basis risk metrics
        if backwardation.basis_values:
            metrics['basis_volatility'] = np.std(backwardation.basis_values)
            metrics['basis_mean'] = np.mean(backwardation.basis_values)
            
            # Market structure stability
            structure_changes = sum(1 for i in range(1, len(backwardation.market_structure))
                                  if backwardation.market_structure[i] != backwardation.market_structure[i-1])
            metrics['structure_stability'] = 1 - (structure_changes / max(len(backwardation.market_structure) - 1, 1))
        
        # Term structure metrics
        if term_structure.calendar_spreads:
            metrics['calendar_spread_volatility'] = np.std(term_structure.calendar_spreads)
            
        if term_structure.roll_yield:
            metrics['roll_yield_mean'] = np.mean(term_structure.roll_yield)
            metrics['roll_yield_volatility'] = np.std(term_structure.roll_yield)
        
        return metrics
    
    def _generate_insights(self, samuelson: SamuelsonResult,
                         backwardation: BackwardationContangoResult,
                         term_structure: TermStructureResult) -> List[str]:
        """Generate analytical insights"""
        insights = []
        
        # Samuelson effect insights
        if samuelson.statistical_significance:
            insights.append(f"Samuelson effect detected: {samuelson.maturity_effect_strength} "
                          f"(coefficient: {samuelson.samuelson_coefficient:.4f})")
            
            if samuelson.volatility_increase_rate > 0.2:
                insights.append(f"High volatility increase near expiry: {samuelson.volatility_increase_rate:.1%}")
        
        # Market structure insights
        if backwardation.market_structure:
            backwardation_pct = (backwardation.market_structure.count('Backwardation') / 
                               len(backwardation.market_structure)) * 100
            
            if backwardation_pct > 60:
                insights.append(f"Market predominantly in backwardation ({backwardation_pct:.1f}%) - "
                              "supply constraints likely")
            elif backwardation_pct < 40:
                insights.append(f"Market predominantly in contango ({100-backwardation_pct:.1f}%) - "
                              "ample supply conditions")
        
        # Term structure insights
        if term_structure.curve_shape != "Unknown":
            insights.append(f"Term structure shape: {term_structure.curve_shape}")
            
            if term_structure.curve_shape == "Inverted":
                insights.append("Inverted curve suggests immediate supply shortage or high convenience yield")
            elif term_structure.curve_shape == "Humped":
                insights.append("Humped curve indicates seasonal or temporary supply/demand imbalances")
        
        # Roll yield insights
        if term_structure.roll_yield:
            avg_roll_yield = np.mean(term_structure.roll_yield)
            if avg_roll_yield > 0.05:
                insights.append(f"Positive average roll yield ({avg_roll_yield:.2%}) favors long positions")
            elif avg_roll_yield < -0.05:
                insights.append(f"Negative average roll yield ({avg_roll_yield:.2%}) favors short positions")
        
        # Seasonal patterns
        if backwardation.seasonal_patterns:
            max_month = max(backwardation.seasonal_patterns.items(), key=lambda x: x[1])
            min_month = min(backwardation.seasonal_patterns.items(), key=lambda x: x[1])
            insights.append(f"Seasonal pattern: Highest basis in {max_month[0]}, lowest in {min_month[0]}")
        
        return insights
    
    def _generate_recommendations(self, samuelson: SamuelsonResult,
                                backwardation: BackwardationContangoResult,
                                term_structure: TermStructureResult,
                                risk_metrics: Dict[str, float]) -> List[str]:
        """Generate trading recommendations"""
        recommendations = []
        
        # Samuelson effect recommendations
        if samuelson.maturity_effect_strength in ["Strong Positive", "Strong Negative"]:
            recommendations.append("Adjust position sizes as contracts approach expiry due to Samuelson effect")
            recommendations.append("Consider volatility trading strategies near contract expiration")
        
        # Market structure recommendations
        structure_stability = risk_metrics.get('structure_stability', 0.5)
        if structure_stability < 0.7:
            recommendations.append("Market structure is unstable - use shorter holding periods")
        
        # Basis risk recommendations
        basis_vol = risk_metrics.get('basis_volatility', 0)
        if basis_vol > 1.0:
            recommendations.append("High basis volatility - consider basis trading strategies")
        
        # Roll yield recommendations
        roll_yield_mean = risk_metrics.get('roll_yield_mean', 0)
        if roll_yield_mean > 0.03:
            recommendations.append("Positive roll yield environment - favor calendar spread strategies")
        elif roll_yield_mean < -0.03:
            recommendations.append("Negative roll yield environment - avoid long-only strategies")
        
        # Term structure recommendations
        if term_structure.curve_shape == "Inverted":
            recommendations.append("Inverted curve - consider buying near-term, selling far-term contracts")
        elif term_structure.curve_shape == "Normal":
            recommendations.append("Normal curve - storage arbitrage opportunities may exist")
        
        # Seasonal recommendations
        if backwardation.seasonal_patterns:
            recommendations.append("Incorporate seasonal patterns into trading calendar")
        
        # Risk management
        volatility_mean = risk_metrics.get('volatility_mean', 0)
        if volatility_mean > 0.3:
            recommendations.append("High volatility environment - use appropriate risk management")
        
        recommendations.append("Monitor convenience yield and storage cost changes for arbitrage opportunities")
        recommendations.append("Consider cross-commodity spread opportunities based on term structure differences")
        
        return recommendations
    
    def _calculate_model_performance(self, samuelson: SamuelsonResult,
                                   backwardation: BackwardationContangoResult,
                                   term_structure: TermStructureResult) -> Dict[str, float]:
        """Calculate model performance metrics"""
        performance = {}
        
        # Samuelson model performance
        performance['samuelson_r_squared'] = samuelson.r_squared
        performance['samuelson_significance'] = 1.0 if samuelson.statistical_significance else 0.0
        
        # Structure prediction accuracy
        if backwardation.market_structure:
            # Simple accuracy based on structure consistency
            structure_consistency = 0
            for i in range(1, len(backwardation.market_structure)):
                if backwardation.market_structure[i] == backwardation.market_structure[i-1]:
                    structure_consistency += 1
            
            performance['structure_consistency'] = structure_consistency / max(len(backwardation.market_structure) - 1, 1)
        
        # Term structure model fit
        if term_structure.calendar_spreads:
            # Measure of term structure predictability
            spreads = term_structure.calendar_spreads
            if len(spreads) > 1:
                spread_autocorr = np.corrcoef(spreads[:-1], spreads[1:])[0, 1] if len(spreads) > 2 else 0
                performance['term_structure_predictability'] = abs(spread_autocorr)
        
        # Overall model confidence
        confidence_scores = [v for v in performance.values() if not np.isnan(v)]
        performance['overall_confidence'] = np.mean(confidence_scores) if confidence_scores else 0.0
        
        return performance
    
    def plot_results(self, spot_prices: List[float], 
                    futures_contracts: List[FuturesContractData],
                    results: FuturesStructureResult):
        """Plot comprehensive futures structure analysis results"""
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        
        # Plot 1: Samuelson Effect - Volatility vs Time to Maturity
        ax1 = axes[0, 0]
        if results.samuelson_results.volatility_profile and results.samuelson_results.time_to_maturity:
            ax1.scatter(results.samuelson_results.time_to_maturity, 
                       results.samuelson_results.volatility_profile, alpha=0.6)
            
            # Fit line
            if len(results.samuelson_results.time_to_maturity) > 1:
                z = np.polyfit(results.samuelson_results.time_to_maturity, 
                              results.samuelson_results.volatility_profile, 1)
                p = np.poly1d(z)
                ax1.plot(results.samuelson_results.time_to_maturity, 
                        p(results.samuelson_results.time_to_maturity), "r--", alpha=0.8)
        
        ax1.set_title(f'Samuelson Effect\n(Strength: {results.samuelson_results.maturity_effect_strength})', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time to Maturity (days)')
        ax1.set_ylabel('Volatility')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Market Structure Over Time
        ax2 = axes[0, 1]
        if results.backwardation_contango_results.basis_values:
            time_periods = range(len(results.backwardation_contango_results.basis_values))
            basis_values = results.backwardation_contango_results.basis_values
            
            colors = ['red' if structure == 'Backwardation' else 'blue' if structure == 'Contango' else 'gray' 
                     for structure in results.backwardation_contango_results.market_structure]
            
            ax2.bar(time_periods, basis_values, color=colors, alpha=0.7)
            ax2.axhline(y=0, color='black', linewidth=1)
            ax2.set_title('Market Structure (Basis = Futures - Spot)', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Time Period')
            ax2.set_ylabel('Basis')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='red', alpha=0.7, label='Backwardation'),
                              Patch(facecolor='blue', alpha=0.7, label='Contango'),
                              Patch(facecolor='gray', alpha=0.7, label='Neutral')]
            ax2.legend(handles=legend_elements)
        
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Term Structure Shape
        ax3 = axes[1, 0]
        if futures_contracts and len(futures_contracts) >= 2:
            # Plot average term structure
            avg_prices = []
            avg_maturities = []
            
            for contract in futures_contracts:
                if contract.prices and contract.time_to_maturity:
                    avg_prices.append(np.mean(contract.prices))
                    avg_maturities.append(np.mean(contract.time_to_maturity))
            
            if len(avg_prices) >= 2:
                # Sort by maturity
                sorted_data = sorted(zip(avg_maturities, avg_prices))
                maturities_sorted = [x[0] for x in sorted_data]
                prices_sorted = [x[1] for x in sorted_data]
                
                ax3.plot(maturities_sorted, prices_sorted, 'bo-', linewidth=2, markersize=8)
                ax3.set_title(f'Term Structure\n(Shape: {results.term_structure_results.curve_shape})', 
                             fontsize=14, fontweight='bold')
                ax3.set_xlabel('Time to Maturity (days)')
                ax3.set_ylabel('Futures Price')
        
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Calendar Spreads
        ax4 = axes[1, 1]
        if results.term_structure_results.calendar_spreads:
            time_periods = range(len(results.term_structure_results.calendar_spreads))
            ax4.plot(time_periods, results.term_structure_results.calendar_spreads, 
                    'g-', linewidth=2, label='Calendar Spread')
            ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax4.set_title('Calendar Spreads (Far - Near)', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Time Period')
            ax4.set_ylabel('Spread Value')
            ax4.legend()
        
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Roll Yield
        ax5 = axes[2, 0]
        if results.term_structure_results.roll_yield:
            time_periods = range(len(results.term_structure_results.roll_yield))
            colors = ['green' if ry > 0 else 'red' for ry in results.term_structure_results.roll_yield]
            
            ax5.bar(time_periods, results.term_structure_results.roll_yield, 
                   color=colors, alpha=0.7)
            ax5.axhline(y=0, color='black', linewidth=1)
            ax5.set_title('Roll Yield', fontsize=14, fontweight='bold')
            ax5.set_xlabel('Time Period')
            ax5.set_ylabel('Roll Yield')
            
            # Add legend
            legend_elements = [Patch(facecolor='green', alpha=0.7, label='Positive'),
                              Patch(facecolor='red', alpha=0.7, label='Negative')]
            ax5.legend(handles=legend_elements)
        
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Risk Metrics
        ax6 = axes[2, 1]
        if results.risk_metrics:
            risk_names = list(results.risk_metrics.keys())
            risk_values = list(results.risk_metrics.values())
            
            bars = ax6.bar(range(len(risk_names)), risk_values, color='lightcoral')
            ax6.set_xticks(range(len(risk_names)))
            ax6.set_xticklabels([name.replace('_', ' ').title() for name in risk_names], 
                               rotation=45, ha='right')
            ax6.set_title('Risk Metrics', fontsize=14, fontweight='bold')
            ax6.set_ylabel('Value')
            
            # Add value labels on bars
            for bar, value in zip(bars, risk_values):
                ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(risk_values)*0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self, spot_prices: List[float],
                       futures_contracts: List[FuturesContractData],
                       results: FuturesStructureResult) -> str:
        """Generate comprehensive futures structure analysis report"""
        report = []
        report.append("=== FUTURES STRUCTURE ANALYSIS REPORT ===")
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Contracts Analyzed: {len(futures_contracts)}")
        report.append(f"Spot Price Observations: {len(spot_prices)}")
        report.append("")
        
        # Samuelson Effect Analysis
        report.append("SAMUELSON EFFECT ANALYSIS:")
        report.append(f"Effect Strength: {results.samuelson_results.maturity_effect_strength}")
        report.append(f"Samuelson Coefficient: {results.samuelson_results.samuelson_coefficient:.6f}")
        report.append(f"R-squared: {results.samuelson_results.r_squared:.4f}")
        report.append(f"Statistical Significance: {results.samuelson_results.statistical_significance}")
        report.append(f"Volatility Increase Rate: {results.samuelson_results.volatility_increase_rate:.2%}")
        report.append("")
        
        # Market Structure Analysis
        report.append("MARKET STRUCTURE ANALYSIS:")
        if results.backwardation_contango_results.market_structure:
            structure_counts = {}
            for structure in results.backwardation_contango_results.market_structure:
                structure_counts[structure] = structure_counts.get(structure, 0) + 1
            
            total_periods = len(results.backwardation_contango_results.market_structure)
            for structure, count in structure_counts.items():
                percentage = (count / total_periods) * 100
                report.append(f"{structure}: {count} periods ({percentage:.1f}%)")
        
        if results.backwardation_contango_results.basis_values:
            avg_basis = np.mean(results.backwardation_contango_results.basis_values)
            basis_vol = np.std(results.backwardation_contango_results.basis_values)
            report.append(f"Average Basis: {avg_basis:.4f}")
            report.append(f"Basis Volatility: {basis_vol:.4f}")
        report.append("")
        
        # Term Structure Analysis
        report.append("TERM STRUCTURE ANALYSIS:")
        report.append(f"Curve Shape: {results.term_structure_results.curve_shape}")
        report.append(f"Slope Coefficient: {results.term_structure_results.slope_coefficient:.6f}")
        report.append(f"Curvature Coefficient: {results.term_structure_results.curvature_coefficient:.6f}")
        
        if results.term_structure_results.roll_yield:
            avg_roll_yield = np.mean(results.term_structure_results.roll_yield)
            roll_yield_vol = np.std(results.term_structure_results.roll_yield)
            report.append(f"Average Roll Yield: {avg_roll_yield:.4f}")
            report.append(f"Roll Yield Volatility: {roll_yield_vol:.4f}")
        report.append("")
        
        # Model Performance
        report.append("MODEL PERFORMANCE:")
        for metric, value in results.model_performance.items():
            report.append(f"{metric.replace('_', ' ').title()}: {value:.4f}")
        report.append("")
        
        # Risk Metrics
        report.append("RISK METRICS:")
        for metric, value in results.risk_metrics.items():
            report.append(f"{metric.replace('_', ' ').title()}: {value:.4f}")
        report.append("")
        
        # Trading Signals Summary
        if results.trading_signals:
            from collections import Counter
            signal_counts = Counter(results.trading_signals)
            report.append("TRADING SIGNALS SUMMARY:")
            for signal, count in signal_counts.items():
                percentage = (count / len(results.trading_signals)) * 100
                report.append(f"{signal}: {count} ({percentage:.1f}%)")
            report.append("")
        
        # Seasonal Patterns
        if results.backwardation_contango_results.seasonal_patterns:
            report.append("SEASONAL PATTERNS (Average Basis by Month):")
            for month, avg_basis in results.backwardation_contango_results.seasonal_patterns.items():
                report.append(f"{month}: {avg_basis:.4f}")
            report.append("")
        
        # Key Insights
        report.append("KEY INSIGHTS:")
        for insight in results.insights:
            report.append(f"• {insight}")
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        for recommendation in results.recommendations:
            report.append(f"• {recommendation}")
        
        return "\n".join(report)

# Example usage and testing
if __name__ == "__main__":
    # Generate sample futures data
    np.random.seed(42)
    
    # Create sample spot prices
    n_periods = 100
    base_spot_price = 70.0
    
    timestamps = [datetime.now() - timedelta(days=i) for i in range(n_periods, 0, -1)]
    
    # Generate spot price series
    spot_returns = np.random.normal(0, 0.02, n_periods)
    spot_prices = [base_spot_price]
    for ret in spot_returns[1:]:
        spot_prices.append(spot_prices[-1] * (1 + ret))
    
    # Create multiple futures contracts with different maturities
    futures_contracts = []
    
    for contract_num in range(3):  # 3 different contract maturities
        base_maturity = 30 + contract_num * 30  # 30, 60, 90 days
        
        # Generate futures prices with basis relationship
        futures_prices = []
        time_to_maturity = []
        volumes = []
        open_interests = []
        
        for i, spot in enumerate(spot_prices):
            # Time to maturity decreases over time
            ttm = max(base_maturity - i * 0.5, 1)
            time_to_maturity.append(ttm)
            
            # Futures price with basis (contango/backwardation)
            if contract_num == 0:  # Near contract - more backwardation
                basis = -0.5 + np.random.normal(0, 0.3)
            else:  # Far contracts - more contango
                basis = 1.0 + contract_num * 0.5 + np.random.normal(0, 0.2)
            
            futures_price = spot + basis
            futures_prices.append(futures_price)
            
            # Generate volume and open interest
            base_volume = 1000 * (4 - contract_num)  # Near contracts have higher volume
            volume = max(base_volume + np.random.normal(0, base_volume * 0.2), 100)
            volumes.append(volume)
            
            base_oi = 5000 * (4 - contract_num)
            oi = max(base_oi + np.random.normal(0, base_oi * 0.1), 500)
            open_interests.append(oi)
        
        contract = FuturesContractData(
            contract_symbol=f"CLZ{23+contract_num}",
            expiry_date=datetime.now() + timedelta(days=base_maturity),
            prices=futures_prices,
            timestamps=timestamps,
            volume=volumes,
            open_interest=open_interests,
            time_to_maturity=time_to_maturity
        )
        
        futures_contracts.append(contract)
    
    # Initialize analyzer
    analyzer = FuturesStructureAnalyzer()
    
    try:
        # Perform analysis
        print("Starting Futures Structure Analysis...")
        results = analyzer.analyze(spot_prices, futures_contracts)
        
        # Print summary
        print("\n=== ANALYSIS SUMMARY ===")
        print(f"Contracts Analyzed: {len(futures_contracts)}")
        print(f"Spot Price Observations: {len(spot_prices)}")
        
        print("\nSamuelson Effect:")
        print(f"Effect Strength: {results.samuelson_results.maturity_effect_strength}")
        print(f"Coefficient: {results.samuelson_results.samuelson_coefficient:.6f}")
        print(f"R-squared: {results.samuelson_results.r_squared:.4f}")
        
        print("\nMarket Structure:")
        if results.backwardation_contango_results.market_structure:
            from collections import Counter
            structure_counts = Counter(results.backwardation_contango_results.market_structure)
            for structure, count in structure_counts.items():
                pct = (count / len(results.backwardation_contango_results.market_structure)) * 100
                print(f"{structure}: {count} ({pct:.1f}%)")
        
        print(f"\nTerm Structure Shape: {results.term_structure_results.curve_shape}")
        
        print("\nModel Performance:")
        for metric, value in results.model_performance.items():
            print(f"{metric}: {value:.4f}")
        
        print("\nKey Insights:")
        for insight in results.insights[:5]:
            print(f"• {insight}")
        
        print("\nRecommendations:")
        for rec in results.recommendations[:3]:
            print(f"• {rec}")
        
        # Generate report
        report = analyzer.generate_report(spot_prices, futures_contracts, results)
        
        # Plot results
        try:
            analyzer.plot_results(spot_prices, futures_contracts, results)
        except Exception as e:
            print(f"Plotting failed: {e}")
        
        print("\nFutures structure analysis completed successfully!")
        
    except Exception as e:
        print(f"Futures structure analysis failed: {e}")
        import traceback
        traceback.print_exc()