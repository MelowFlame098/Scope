from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, optimize
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Import modularized classes
from .macroeconomic_factor_model import MacroeconomicFactorModel, IndexData, MacroeconomicFactors, FactorModelResult
from .arbitrage_pricing_theory import ArbitragePricingTheory, APTResult
from .capm_analyzer import CAPMAnalyzer, CAPMResult

# Try to import advanced libraries
try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import coint, adfuller
    from statsmodels.stats.diagnostic import het_breuschpagan
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Statsmodels not available. Using simplified statistical methods.")

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    print("ARCH library not available. Using simplified volatility models.")

@dataclass
class IndexFactorAnalysisResult:
    """Comprehensive index factor analysis results"""
    factor_model_results: FactorModelResult
    apt_results: APTResult
    capm_results: CAPMResult
    risk_attribution: Dict[str, Dict[str, float]]
    performance_attribution: Dict[str, Dict[str, float]]
    factor_timing_analysis: Dict[str, Any]
    insights: List[str]
    recommendations: List[str]
    model_diagnostics: Dict[str, Any]

class IndexFactorAnalyzer:
    """Comprehensive index factor analysis combining multiple models"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.factor_model = MacroeconomicFactorModel()
        self.apt_model = ArbitragePricingTheory()
        self.capm_model = CAPMAnalyzer(risk_free_rate)
        
    def analyze(self, index_data: IndexData, 
               macro_factors: MacroeconomicFactors,
               market_data: IndexData) -> IndexFactorAnalysisResult:
        """Perform comprehensive factor analysis"""
        
        # Run individual models
        factor_results = self.factor_model.fit_factor_model(index_data, macro_factors)
        apt_results = self.apt_model.analyze_apt(index_data, macro_factors)
        capm_results = self.capm_model.analyze_capm(index_data, market_data)
        
        # Calculate risk attribution
        risk_attribution = self._calculate_risk_attribution(factor_results, capm_results)
        
        # Calculate performance attribution
        performance_attribution = self._calculate_performance_attribution(
            factor_results, apt_results, capm_results
        )
        
        # Analyze factor timing
        factor_timing_analysis = self._analyze_factor_timing(factor_results, index_data)
        
        # Generate insights and recommendations
        insights = self._generate_insights(factor_results, apt_results, capm_results)
        recommendations = self._generate_recommendations(
            factor_results, apt_results, capm_results, risk_attribution
        )
        
        # Calculate model diagnostics
        model_diagnostics = self._calculate_model_diagnostics(
            factor_results, apt_results, capm_results
        )
        
        return IndexFactorAnalysisResult(
            factor_model_results=factor_results,
            apt_results=apt_results,
            capm_results=capm_results,
            risk_attribution=risk_attribution,
            performance_attribution=performance_attribution,
            factor_timing_analysis=factor_timing_analysis,
            insights=insights,
            recommendations=recommendations,
            model_diagnostics=model_diagnostics
        )
    
    def _calculate_risk_attribution(self, factor_results: FactorModelResult,
                                  capm_results: CAPMResult) -> Dict[str, Dict[str, float]]:
        """Calculate risk attribution across models"""
        return {
            'factor_model': {
                'explained': factor_results.explained_variance,
                'residual': factor_results.residual_variance
            },
            'capm': {
                'systematic': capm_results.systematic_risk,
                'unsystematic': capm_results.unsystematic_risk
            }
        }
    
    def _calculate_performance_attribution(self, factor_results: FactorModelResult,
                                         apt_results: APTResult,
                                         capm_results: CAPMResult) -> Dict[str, Dict[str, float]]:
        """Calculate performance attribution"""
        return {
            'factor_contributions': factor_results.factor_loadings,
            'apt_premiums': apt_results.factor_premiums,
            'capm_metrics': {
                'alpha': capm_results.alpha,
                'beta': capm_results.beta,
                'market_premium': capm_results.market_risk_premium
            }
        }
    
    def _analyze_factor_timing(self, factor_results: FactorModelResult,
                             index_data: IndexData) -> Dict[str, Any]:
        """Analyze factor timing and exposures"""
        return {
            'factor_stability': self._calculate_factor_stability(factor_results),
            'exposure_changes': self._calculate_exposure_changes(factor_results),
            'timing_metrics': self._calculate_timing_metrics(factor_results, index_data)
        }
    
    def _calculate_factor_stability(self, factor_results: FactorModelResult) -> Dict[str, float]:
        """Calculate factor loading stability over time"""
        stability_metrics = {}
        
        if factor_results.factor_exposures:
            for factor_name in factor_results.factor_loadings.keys():
                exposures = [exp.get(factor_name, 0) for exp in factor_results.factor_exposures]
                if exposures:
                    stability_metrics[factor_name] = np.std(exposures)
        
        return stability_metrics
    
    def _calculate_exposure_changes(self, factor_results: FactorModelResult) -> Dict[str, Any]:
        """Calculate changes in factor exposures"""
        if not factor_results.factor_exposures or len(factor_results.factor_exposures) < 2:
            return {}
        
        first_half = factor_results.factor_exposures[:len(factor_results.factor_exposures)//2]
        second_half = factor_results.factor_exposures[len(factor_results.factor_exposures)//2:]
        
        changes = {}
        for factor_name in factor_results.factor_loadings.keys():
            first_avg = np.mean([exp.get(factor_name, 0) for exp in first_half])
            second_avg = np.mean([exp.get(factor_name, 0) for exp in second_half])
            changes[factor_name] = second_avg - first_avg
        
        return {
            'factor_changes': changes,
            'significant_changes': {k: v for k, v in changes.items() if abs(v) > 0.1}
        }
    
    def _calculate_timing_metrics(self, factor_results: FactorModelResult,
                                index_data: IndexData) -> Dict[str, float]:
        """Calculate factor timing metrics"""
        return {
            'tracking_error': factor_results.tracking_error,
            'information_ratio': factor_results.information_ratio,
            'r_squared': factor_results.r_squared
        }
    
    def _generate_insights(self, factor_results: FactorModelResult,
                         apt_results: APTResult,
                         capm_results: CAPMResult) -> List[str]:
        """Generate analytical insights"""
        insights = []
        
        # Factor model insights
        if factor_results.r_squared > 0.7:
            insights.append(f"Strong factor model fit (R² = {factor_results.r_squared:.1%})")
        elif factor_results.r_squared > 0.4:
            insights.append(f"Moderate factor model fit (R² = {factor_results.r_squared:.1%})")
        else:
            insights.append(f"Weak factor model fit (R² = {factor_results.r_squared:.1%})")
        
        # Top factors
        sorted_factors = sorted(factor_results.factor_loadings.items(), 
                              key=lambda x: abs(x[1]), reverse=True)
        if sorted_factors:
            top_factor = sorted_factors[0]
            insights.append(f"Most influential factor: {top_factor[0]} (loading: {top_factor[1]:.3f})")
        
        # CAPM insights
        if capm_results.beta > 1.2:
            insights.append(f"High systematic risk (β = {capm_results.beta:.2f})")
        elif capm_results.beta < 0.8:
            insights.append(f"Low systematic risk (β = {capm_results.beta:.2f})")
        
        if abs(camp_results.alpha) > 0.02:
            direction = "positive" if capm_results.alpha > 0 else "negative"
            insights.append(f"Significant {direction} alpha ({capm_results.alpha:.2%})")
        
        return insights
    
    def _generate_recommendations(self, factor_results: FactorModelResult,
                                apt_results: APTResult,
                                capm_results: CAPMResult,
                                risk_attribution: Dict[str, Dict[str, float]]) -> List[str]:
        """Generate investment recommendations"""
        recommendations = []
        
        # Risk management recommendations
        if capm_results.beta > 1.5:
            recommendations.append("Consider hedging strategies due to high market sensitivity")
        
        if factor_results.r_squared < 0.5:
            recommendations.append("Diversify across additional factors to improve risk management")
        
        # Factor exposure recommendations
        sorted_factors = sorted(factor_results.factor_loadings.items(),
                              key=lambda x: abs(x[1]), reverse=True)
        
        if sorted_factors and abs(sorted_factors[0][1]) > 0.5:
            factor_name = sorted_factors[0][0]
            recommendations.append(f"Monitor {factor_name} exposure closely due to high sensitivity")
        
        # Performance recommendations
        if capm_results.alpha > 0.01:
            recommendations.append("Strong alpha generation - consider increasing allocation")
        elif capm_results.alpha < -0.01:
            recommendations.append("Negative alpha - review investment strategy")
        
        return recommendations
    
    def _calculate_model_diagnostics(self, factor_results: FactorModelResult,
                                   apt_results: APTResult,
                                   capm_results: CAPMResult) -> Dict[str, Any]:
        """Calculate comprehensive model diagnostics"""
        return {
            'model_comparison': {
                'factor_model_r2': factor_results.r_squared,
                'apt_model_r2': apt_results.model_fit.get('r_squared', 0),
                'capm_r2': capm_results.r_squared,
                'best_model': self._determine_best_model(factor_results, apt_results, capm_results)
            },
            'risk_metrics': {
                'tracking_error': factor_results.tracking_error,
                'information_ratio': factor_results.information_ratio,
                'systematic_risk': capm_results.systematic_risk,
                'total_risk': capm_results.systematic_risk + capm_results.unsystematic_risk
            }
        }
    
    def _determine_best_model(self, factor_results: FactorModelResult,
                            apt_results: APTResult,
                            capm_results: CAPMResult) -> str:
        """Determine the best performing model"""
        r2_scores = {
            'Factor Model': factor_results.r_squared,
            'APT Model': apt_results.model_fit.get('r_squared', 0),
            'CAPM': capm_results.r_squared
        }
        return max(r2_scores, key=r2_scores.get)
    
    def plot_results(self, index_data: IndexData, 
                    results: IndexFactorAnalysisResult):
        """Generate comprehensive analysis plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Index Factor Analysis Results', fontsize=16)
        
        # Factor loadings
        factors = list(results.factor_model_results.factor_loadings.keys())
        loadings = list(results.factor_model_results.factor_loadings.values())
        
        axes[0, 0].bar(factors, loadings)
        axes[0, 0].set_title('Factor Loadings')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Risk attribution
        risk_data = results.risk_attribution['capm']
        axes[0, 1].pie([risk_data['systematic'], risk_data['unsystematic']], 
                      labels=['Systematic', 'Unsystematic'], autopct='%1.1f%%')
        axes[0, 1].set_title('Risk Attribution (CAPM)')
        
        # Model comparison
        models = ['Factor Model', 'APT', 'CAPM']
        r2_values = [
            results.factor_model_results.r_squared,
            results.apt_results.model_fit.get('r_squared', 0),
            results.capm_results.r_squared
        ]
        
        axes[0, 2].bar(models, r2_values)
        axes[0, 2].set_title('Model R² Comparison')
        axes[0, 2].set_ylabel('R²')
        
        # Factor exposures over time (if available)
        if results.factor_model_results.factor_exposures:
            exposure_data = results.factor_model_results.factor_exposures
            if len(exposure_data) > 1:
                # Plot top 3 factors over time
                top_factors = sorted(results.factor_model_results.factor_loadings.items(),
                                   key=lambda x: abs(x[1]), reverse=True)[:3]
                
                for i, (factor_name, _) in enumerate(top_factors):
                    exposures = [exp.get(factor_name, 0) for exp in exposure_data]
                    axes[1, 0].plot(exposures, label=factor_name)
                
                axes[1, 0].set_title('Factor Exposures Over Time')
                axes[1, 0].legend()
                axes[1, 0].set_xlabel('Time Period')
                axes[1, 0].set_ylabel('Exposure')
        
        # Performance metrics
        metrics = ['Alpha', 'Beta', 'Sharpe Ratio', 'Information Ratio']
        values = [
            results.capm_results.alpha,
            results.capm_results.beta,
            results.capm_results.sharpe_ratio,
            results.factor_model_results.information_ratio
        ]
        
        axes[1, 1].bar(metrics, values)
        axes[1, 1].set_title('Performance Metrics')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Factor contributions
        if results.factor_model_results.factor_returns:
            factor_names = list(results.factor_model_results.factor_returns.keys())[:5]
            contributions = []
            
            for name in factor_names:
                factor_return = results.factor_model_results.factor_returns[name]
                if factor_return:
                    contributions.append(np.mean(factor_return))
                else:
                    contributions.append(0)
            
            axes[1, 2].bar(factor_names, contributions)
            axes[1, 2].set_title('Average Factor Contributions')
            axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self, index_data: IndexData, 
                       results: IndexFactorAnalysisResult) -> str:
        """Generate comprehensive analysis report"""
        
        report = f"""
# INDEX FACTOR ANALYSIS REPORT

## Executive Summary

Index: {index_data.index_symbol}
Analysis Period: {len(index_data.prices)} periods
Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Performance Summary

### Factor Model Results
- R-squared: {results.factor_model_results.r_squared:.2%}
- Explained Variance: {results.factor_model_results.explained_variance:.2%}
- Tracking Error: {results.factor_model_results.tracking_error:.4f}
- Information Ratio: {results.factor_model_results.information_ratio:.4f}

### CAPM Analysis Results
- Beta: {results.capm_results.beta:.3f}
- Alpha: {results.capm_results.alpha:.4f} ({results.capm_results.alpha:.2%})
- R-squared: {results.capm_results.r_squared:.2%}
- Sharpe Ratio: {results.capm_results.sharpe_ratio:.3f}
- Systematic Risk: {results.capm_results.systematic_risk:.4f}
- Unsystematic Risk: {results.capm_results.unsystematic_risk:.4f}

### APT Model Results
- Model R-squared: {results.apt_results.model_fit.get('r_squared', 0):.2%}
- Number of Arbitrage Opportunities: {len(results.apt_results.arbitrage_opportunities)}

## Factor Analysis

### Top Factor Loadings
"""
        
        # Add factor loadings
        sorted_factors = sorted(results.factor_model_results.factor_loadings.items(),
                              key=lambda x: abs(x[1]), reverse=True)
        
        for i, (factor, loading) in enumerate(sorted_factors[:10], 1):
            report += f"{i:2d}. {factor:25s}: {loading:8.4f}\n"
        
        report += f"""

### Factor Premiums (APT)
"""
        
        # Add APT factor premiums
        for factor, premium in results.apt_results.factor_premiums.items():
            report += f"- {factor:25s}: {premium:8.4f}\n"
        
        report += f"""

## Risk Attribution

### CAPM Risk Breakdown
- Systematic Risk: {results.risk_attribution['capm']['systematic']:.2%}
- Unsystematic Risk: {results.risk_attribution['capm']['unsystematic']:.2%}

### Factor Model Risk Breakdown
- Explained Variance: {results.risk_attribution['factor_model']['explained']:.2%}
- Residual Variance: {results.risk_attribution['factor_model']['residual']:.2%}

## Performance Attribution

### Factor Contributions
"""
        
        # Add performance attribution
        for factor, contribution in results.performance_attribution['factor_contributions'].items():
            report += f"- {factor:25s}: {contribution:8.4f}\n"
        
        report += f"""

## Model Diagnostics

### Model Comparison
- Best Performing Model: {results.model_diagnostics['model_comparison']['best_model']}
- Factor Model R²: {results.model_diagnostics['model_comparison']['factor_model_r2']:.2%}
- APT Model R²: {results.model_diagnostics['model_comparison']['apt_model_r2']:.2%}
- CAPM R²: {results.model_diagnostics['model_comparison']['capm_r2']:.2%}

### Risk Metrics Summary
- Tracking Error: {results.model_diagnostics['risk_metrics']['tracking_error']:.4f}
- Information Ratio: {results.model_diagnostics['risk_metrics']['information_ratio']:.4f}
- Total Risk: {results.model_diagnostics['risk_metrics']['total_risk']:.4f}

## Key Insights

"""
        
        # Add insights
        for i, insight in enumerate(results.insights, 1):
            report += f"{i}. {insight}\n"
        
        report += f"""

## Recommendations

"""
        
        # Add recommendations
        for i, recommendation in enumerate(results.recommendations, 1):
            report += f"{i}. {recommendation}\n"
        
        report += f"""

## Factor Timing Analysis

### Factor Stability Metrics
"""
        
        # Add factor timing analysis
        if 'factor_stability' in results.factor_timing_analysis:
            for factor, stability in results.factor_timing_analysis['factor_stability'].items():
                report += f"- {factor:25s}: {stability:8.4f}\n"
        
        report += f"""

### Significant Exposure Changes
"""
        
        # Add exposure changes
        if 'exposure_changes' in results.factor_timing_analysis:
            changes = results.factor_timing_analysis['exposure_changes'].get('significant_changes', {})
            if changes:
                for factor, change in changes.items():
                    direction = "increased" if change > 0 else "decreased"
                    report += f"- {factor} exposure {direction} by {abs(change):.4f}\n"
            else:
                report += "- No significant exposure changes detected\n"
        
        report += f"""

---
Report generated by FinScope Index Factor Analyzer
"""
        
        return report

if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)
    n_periods = 252  # One year of daily data
    
    # Generate synthetic index data
    base_price = 1000.0
    returns = np.random.normal(0.0008, 0.015, n_periods)  # Daily returns
    prices = [base_price]
    
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    # Generate synthetic volume and market cap data
    base_volume = 1000000
    volumes = [base_volume * (1 + np.random.normal(0, 0.2)) for _ in range(len(prices))]
    volumes = [max(100000, v) for v in volumes]
    
    market_caps = [p * 1000000 for p in prices]  # Simple market cap proxy
    
    # Generate timestamps
    start_date = datetime.now() - timedelta(days=n_periods)
    timestamps = [start_date + timedelta(days=i) for i in range(len(prices))]
    
    # Create IndexData object
    index_data = IndexData(
        prices=prices,
        returns=returns.tolist(),
        volume=volumes,
        market_cap=market_caps,
        timestamps=timestamps,
        index_symbol="SPX",
        constituent_weights={"AAPL": 0.07, "MSFT": 0.06, "GOOGL": 0.04},
        sector_weights={"Technology": 0.28, "Healthcare": 0.13, "Financials": 0.11}
    )
    
    # Generate synthetic macroeconomic data
    macro_factors = MacroeconomicFactors(
        gdp_growth=np.random.normal(0.02, 0.005, len(timestamps)).tolist(),
        inflation_rate=np.random.normal(0.025, 0.003, len(timestamps)).tolist(),
        interest_rates=np.random.normal(0.03, 0.01, len(timestamps)).tolist(),
        unemployment_rate=np.random.normal(0.05, 0.01, len(timestamps)).tolist(),
        industrial_production=np.random.normal(0.01, 0.02, len(timestamps)).tolist(),
        consumer_confidence=np.random.normal(100, 10, len(timestamps)).tolist(),
        oil_prices=np.random.normal(70, 15, len(timestamps)).tolist(),
        exchange_rates=np.random.normal(1.1, 0.1, len(timestamps)).tolist(),
        vix_index=np.random.normal(20, 8, len(timestamps)).tolist(),
        timestamps=timestamps
    )
    
    # Generate synthetic market data
    market_returns = np.random.normal(0.0007, 0.012, n_periods)
    market_prices = [1000.0]
    for ret in market_returns:
        market_prices.append(market_prices[-1] * (1 + ret))
    
    market_data = IndexData(
        prices=market_prices,
        returns=market_returns.tolist(),
        volume=[v * 1.5 for v in volumes],  # Higher volume for market
        market_cap=[mc * 10 for mc in market_caps],  # Larger market cap
        timestamps=timestamps,
        index_symbol="MARKET"
    )
    
    # Create analyzer and run analysis
    analyzer = IndexFactorAnalyzer(risk_free_rate=0.02)
    
    # Perform analysis
    print("Performing comprehensive index factor analysis...")
    results = analyzer.analyze(index_data, macro_factors, market_data)
    
    # Print summary
    print("\n" + "="*80)
    print("INDEX FACTOR ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\nIndex: {index_data.index_symbol}")
    print(f"Analysis Period: {len(index_data.prices)} periods")
    
    print(f"\nModel Performance:")
    print(f"- Factor Model R²: {results.factor_model_results.r_squared:.2%}")
    print(f"- APT Model R²: {results.apt_results.model_fit['r_squared']:.2%}")
    print(f"- CAPM R²: {results.capm_results.r_squared:.2%}")
    print(f"- Best Model: {results.model_diagnostics['model_comparison']['best_model']}")
    
    print(f"\nCAPM Metrics:")
    print(f"- Beta: {results.capm_results.beta:.3f}")
    print(f"- Alpha: {results.capm_results.alpha:.4f}")
    print(f"- Jensen's Alpha: {results.capm_results.jensen_alpha:.2%}")
    print(f"- Sharpe Ratio: {results.capm_results.sharpe_ratio:.2f}")
    
    print(f"\nTop Factor Loadings:")
    sorted_factors = sorted(results.factor_model_results.factor_loadings.items(), 
                          key=lambda x: abs(x[1]), reverse=True)
    for factor, loading in sorted_factors[:5]:
        print(f"- {factor}: {loading:.3f}")
    
    print(f"\nRisk Attribution (CAPM):")
    print(f"- Systematic Risk: {results.risk_attribution['capm']['systematic']:.1%}")
    print(f"- Unsystematic Risk: {results.risk_attribution['capm']['unsystematic']:.1%}")
    
    print(f"\nKey Insights:")
    for i, insight in enumerate(results.insights[:5], 1):
        print(f"{i}. {insight}")
    
    print(f"\nTop Recommendations:")
    for i, recommendation in enumerate(results.recommendations[:3], 1):
        print(f"{i}. {recommendation}")
    
    # Generate and save report
    report = analyzer.generate_report(index_data, results)
    
    try:
        with open("index_factor_analysis_report.txt", "w") as f:
            f.write(report)
        print(f"\nDetailed report saved to: index_factor_analysis_report.txt")
    except Exception as e:
        print(f"\nCould not save report: {e}")
    
    # Generate plots
    try:
        print("\nGenerating plots...")
        analyzer.plot_results(index_data, results)
    except Exception as e:
        print(f"Could not generate plots: {e}")
    
    print("\nAnalysis completed successfully!")