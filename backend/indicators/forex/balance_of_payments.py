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
import matplotlib.pyplot as plt
import seaborn as sns

# Import modular components
from current_account_analyzer import CurrentAccountAnalyzer, CurrentAccountData
from financial_account_analyzer import FinancialAccountAnalyzer, CapitalAccountData
from exchange_rate_pressure_analyzer import ExchangeRatePressureAnalyzer, ExchangeRatePressureResult
from bop_sustainability_analyzer import BOPSustainabilityAnalyzer, SustainabilityResult

# Suppress warnings
warnings.filterwarnings('ignore')

@dataclass
class BOPAnalysisResult:
    """Balance of Payments analysis result."""
    current_account_analysis: Dict[str, Any]
    financial_account_analysis: Dict[str, Any]
    sustainability_metrics: Dict[str, Any]
    exchange_rate_pressure: Dict[str, Any]
    flow_volatility: Dict[str, Any]
    crisis_indicators: Dict[str, Any]
    forecasts: Dict[str, Any]
    policy_implications: Dict[str, Any]

class BalanceOfPaymentsAnalyzer:
    """Main Balance of Payments analyzer that coordinates all sub-analyzers."""
    
    def __init__(self):
        self.ca_analyzer = CurrentAccountAnalyzer()
        self.fa_analyzer = FinancialAccountAnalyzer()
        self.pressure_analyzer = ExchangeRatePressureAnalyzer()
        self.sustainability_analyzer = BOPSustainabilityAnalyzer()
        
    def analyze(self, bop_data: Dict[str, Any]) -> BOPAnalysisResult:
        """Perform comprehensive Balance of Payments analysis."""
        try:
            # Extract data components
            ca_data = bop_data.get('current_account_data')
            fa_data = bop_data.get('financial_account_data')
            gdp_data = bop_data.get('gdp_data')
            exchange_rates = bop_data.get('exchange_rates')
            interest_rates = bop_data.get('interest_rates')
            external_debt = bop_data.get('external_debt')
            
            # Current account analysis
            ca_analysis = {}
            if ca_data:
                ca_analysis = self.ca_analyzer.analyze_current_account(
                    ca_data, gdp_data, exchange_rates
                )
            
            # Financial account analysis
            fa_analysis = {}
            if fa_data:
                fa_analysis = self.fa_analyzer.analyze_financial_account(
                    fa_data, exchange_rates
                )
            
            # Exchange rate pressure analysis
            pressure_analysis = {}
            if exchange_rates is not None and interest_rates is not None and fa_data:
                pressure_result = self.pressure_analyzer.calculate_pressure_index(
                    exchange_rates, interest_rates, fa_data.reserve_assets
                )
                pressure_analysis = {
                    'pressure_index': pressure_result.pressure_index,
                    'pressure_components': pressure_result.pressure_components,
                    'crisis_probability': pressure_result.crisis_probability,
                    'pressure_threshold': pressure_result.pressure_threshold,
                    'crisis_periods': pressure_result.crisis_periods,
                    'early_warning_signals': pressure_result.early_warning_signals
                }
            
            # Sustainability analysis
            sustainability_metrics = {}
            if ca_data and fa_data and gdp_data is not None:
                sustainability_result = self.sustainability_analyzer.analyze_sustainability(
                    ca_data, fa_data, gdp_data, external_debt
                )
                sustainability_metrics = {
                    'current_account_sustainability': sustainability_result.current_account_sustainability,
                    'external_debt_sustainability': sustainability_result.external_debt_sustainability,
                    'reserve_adequacy': sustainability_result.reserve_adequacy,
                    'vulnerability_indicators': sustainability_result.vulnerability_indicators,
                    'sustainability_score': sustainability_result.sustainability_score,
                    'risk_level': sustainability_result.risk_level
                }
            
            # Overall flow volatility analysis
            flow_volatility = self._analyze_overall_volatility(ca_data, fa_data)
            
            # Compile crisis indicators
            crisis_indicators = self._compile_crisis_indicators(
                ca_analysis, fa_analysis, pressure_analysis
            )
            
            # Generate forecasts
            forecasts = self._generate_forecasts(ca_data, fa_data, gdp_data)
            
            # Generate policy implications
            policy_implications = self._generate_policy_implications(
                ca_analysis, fa_analysis, sustainability_metrics
            )
            
            return BOPAnalysisResult(
                current_account_analysis=ca_analysis,
                financial_account_analysis=fa_analysis,
                sustainability_metrics=sustainability_metrics,
                exchange_rate_pressure=pressure_analysis,
                flow_volatility=flow_volatility,
                crisis_indicators=crisis_indicators,
                forecasts=forecasts,
                policy_implications=policy_implications
            )
            
        except Exception as e:
            # Return empty result on error
            return BOPAnalysisResult(
                current_account_analysis={},
                financial_account_analysis={},
                sustainability_metrics={},
                exchange_rate_pressure={},
                flow_volatility={},
                crisis_indicators={},
                forecasts={},
                policy_implications={}
            )
            
    def _analyze_overall_volatility(self, ca_data: CurrentAccountData,
                                   fa_data: CapitalAccountData) -> Dict[str, Any]:
        """Analyze overall BOP volatility."""
        try:
            volatility_metrics = {}
            
            if ca_data:
                ca_volatility = ca_data.current_account_balance.pct_change().std()
                volatility_metrics['current_account_volatility'] = ca_volatility
                
            if fa_data:
                fa_volatility = fa_data.financial_account_balance.pct_change().std()
                volatility_metrics['financial_account_volatility'] = fa_volatility
                
            return volatility_metrics
        except:
            return {'volatility_analysis': 'failed'}
            
    def _compile_crisis_indicators(self, ca_analysis: Dict[str, Any],
                                  fa_analysis: Dict[str, Any],
                                  pressure_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Compile crisis indicators from all analyses."""
        try:
            crisis_signals = []
            
            # Current account crisis signals
            if 'sustainability' in ca_analysis:
                if ca_analysis['sustainability'].get('risk_level') == 'high':
                    crisis_signals.append('high_ca_deficit')
                    
            # Financial account crisis signals
            if 'crisis_indicators' in fa_analysis:
                fa_risk = fa_analysis['crisis_indicators'].get('risk_level', 'low')
                if fa_risk in ['high', 'moderate']:
                    crisis_signals.append('capital_flow_stress')
                    
            # Exchange rate pressure signals
            if 'early_warning_signals' in pressure_analysis:
                warning_level = pressure_analysis['early_warning_signals'].get('warning_level', 'low')
                if warning_level in ['high', 'moderate']:
                    crisis_signals.append('exchange_rate_pressure')
                    
            # Overall risk assessment
            if len(crisis_signals) >= 2:
                overall_risk = 'high'
            elif len(crisis_signals) == 1:
                overall_risk = 'moderate'
            else:
                overall_risk = 'low'
                
            return {
                'crisis_signals': crisis_signals,
                'overall_risk': overall_risk,
                'signal_count': len(crisis_signals)
            }
        except:
            return {'crisis_indicators': 'compilation_failed'}
            
    def _generate_forecasts(self, ca_data: CurrentAccountData,
                           fa_data: CapitalAccountData,
                           gdp_data: pd.Series) -> Dict[str, Any]:
        """Generate simple forecasts for key BOP components."""
        try:
            forecasts = {}
            
            if ca_data:
                ca_trend = self._calculate_trend(ca_data.current_account_balance)
                forecasts['current_account_forecast'] = {
                    'trend': ca_trend,
                    'direction': 'improving' if ca_trend > 0 else 'deteriorating'
                }
                
            if fa_data:
                fa_trend = self._calculate_trend(fa_data.financial_account_balance)
                forecasts['financial_account_forecast'] = {
                    'trend': fa_trend,
                    'direction': 'increasing' if fa_trend > 0 else 'decreasing'
                }
                
            return forecasts
        except:
            return {'forecasts': 'generation_failed'}
            
    def _generate_policy_implications(self, ca_analysis: Dict[str, Any],
                                     fa_analysis: Dict[str, Any],
                                     sustainability_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate policy recommendations based on analysis."""
        try:
            recommendations = {
                'fiscal_policy': [],
                'monetary_policy': [],
                'structural_reforms': [],
                'capital_controls': []
            }
            
            # Current account based recommendations
            if 'sustainability' in ca_analysis:
                ca_risk = ca_analysis['sustainability'].get('risk_level', 'low')
                if ca_risk == 'high':
                    recommendations['fiscal_policy'].append('reduce_fiscal_deficit')
                    recommendations['structural_reforms'].append('improve_export_competitiveness')
                    
            # Financial account based recommendations
            if 'crisis_indicators' in fa_analysis:
                fa_risk = fa_analysis['crisis_indicators'].get('risk_level', 'low')
                if fa_risk == 'high':
                    recommendations['monetary_policy'].append('maintain_adequate_reserves')
                    recommendations['capital_controls'].append('monitor_hot_money_flows')
                    
            # Sustainability based recommendations
            if sustainability_metrics:
                risk_level = sustainability_metrics.get('risk_level', 'low_risk')
                if risk_level in ['high_risk', 'very_high_risk']:
                    recommendations['fiscal_policy'].append('implement_adjustment_program')
                    recommendations['structural_reforms'].append('enhance_external_competitiveness')
                    
            # Determine priority level
            total_recommendations = sum(len(recs) for recs in recommendations.values())
            if total_recommendations >= 4:
                priority_level = 'high'
            elif total_recommendations >= 2:
                priority_level = 'moderate'
            else:
                priority_level = 'low'
                
            return {
                **recommendations,
                'priority_level': priority_level
            }
        except:
            return {'policy_implications': 'generation_failed'}
            
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
            
    def plot_analysis(self, results: BOPAnalysisResult, save_path: str = None) -> None:
        """Plot comprehensive BOP analysis results."""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Balance of Payments Analysis', fontsize=16, fontweight='bold')
            
            # Plot individual components
            self._plot_current_account(axes[0, 0], results.current_account_analysis)
            self._plot_financial_account(axes[0, 1], results.financial_account_analysis)
            self._plot_sustainability(axes[0, 2], results.sustainability_metrics)
            self._plot_pressure_index(axes[1, 0], results.exchange_rate_pressure)
            self._plot_crisis_indicators(axes[1, 1], results.crisis_indicators)
            self._plot_policy_implications(axes[1, 2], results.policy_implications)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()
                
        except Exception as e:
            print(f"Error creating plots: {e}")
            
    def _plot_current_account(self, ax, ca_analysis: Dict[str, Any]) -> None:
        """Plot current account analysis."""
        try:
            ax.set_title('Current Account Analysis', fontweight='bold')
            
            if 'basic_statistics' in ca_analysis:
                stats_data = ca_analysis['basic_statistics']
                components = ['trade_balance', 'services_balance', 'primary_income', 'secondary_income']
                means = [stats_data.get(comp, {}).get('mean', 0) for comp in components]
                
                ax.bar(components, means, alpha=0.7)
                ax.set_ylabel('Average Balance')
                ax.tick_params(axis='x', rotation=45)
            else:
                ax.text(0.5, 0.5, 'No current account data', ha='center', va='center', transform=ax.transAxes)
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Plot error: {str(e)[:50]}', ha='center', va='center', transform=ax.transAxes)
            
    def _plot_financial_account(self, ax, fa_analysis: Dict[str, Any]) -> None:
        """Plot financial account analysis."""
        try:
            ax.set_title('Financial Account Analysis', fontweight='bold')
            
            if 'capital_flows' in fa_analysis:
                flows_data = fa_analysis['capital_flows']
                flow_types = ['fdi_analysis', 'portfolio_analysis', 'other_investment_analysis']
                avg_flows = [flows_data.get(flow, {}).get('average_flow', 0) for flow in flow_types]
                labels = ['FDI', 'Portfolio', 'Other Investment']
                
                ax.bar(labels, avg_flows, alpha=0.7, color=['green', 'orange', 'blue'])
                ax.set_ylabel('Average Flow')
            else:
                ax.text(0.5, 0.5, 'No financial account data', ha='center', va='center', transform=ax.transAxes)
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Plot error: {str(e)[:50]}', ha='center', va='center', transform=ax.transAxes)
            
    def _plot_sustainability(self, ax, sustainability: Dict[str, Any]) -> None:
        """Plot sustainability metrics."""
        try:
            ax.set_title('Sustainability Metrics', fontweight='bold')
            
            if 'sustainability_score' in sustainability:
                score = sustainability['sustainability_score']
                risk_level = sustainability.get('risk_level', 'unknown')
                
                # Create a gauge-like plot
                colors = ['red', 'orange', 'yellow', 'green']
                thresholds = [25, 50, 75, 100]
                
                for i, (threshold, color) in enumerate(zip(thresholds, colors)):
                    if score <= threshold:
                        ax.bar(['Sustainability Score'], [score], color=color, alpha=0.7)
                        break
                        
                ax.set_ylim(0, 100)
                ax.set_ylabel('Score (0-100)')
                ax.text(0, score + 5, f'{score:.1f}\n({risk_level})', ha='center')
            else:
                ax.text(0.5, 0.5, 'No sustainability data', ha='center', va='center', transform=ax.transAxes)
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Plot error: {str(e)[:50]}', ha='center', va='center', transform=ax.transAxes)
            
    def _plot_pressure_index(self, ax, pressure: Dict[str, Any]) -> None:
        """Plot exchange rate pressure index."""
        try:
            ax.set_title('Exchange Rate Pressure', fontweight='bold')
            
            if 'pressure_index' in pressure and len(pressure['pressure_index']) > 0:
                pressure_index = pressure['pressure_index']
                threshold = pressure.get('pressure_threshold', 0)
                
                ax.plot(pressure_index.index, pressure_index.values, label='Pressure Index')
                ax.axhline(y=threshold, color='red', linestyle='--', label='Crisis Threshold')
                ax.set_ylabel('Pressure Index')
                ax.legend()
                ax.tick_params(axis='x', rotation=45)
            else:
                ax.text(0.5, 0.5, 'No pressure data', ha='center', va='center', transform=ax.transAxes)
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Plot error: {str(e)[:50]}', ha='center', va='center', transform=ax.transAxes)
            
    def _plot_crisis_indicators(self, ax, crisis: Dict[str, Any]) -> None:
        """Plot crisis indicators."""
        try:
            ax.set_title('Crisis Indicators', fontweight='bold')
            
            if 'crisis_signals' in crisis:
                signals = crisis['crisis_signals']
                overall_risk = crisis.get('overall_risk', 'unknown')
                
                # Create a simple bar chart of signal counts
                signal_counts = {signal: 1 for signal in signals}
                if signal_counts:
                    ax.bar(signal_counts.keys(), signal_counts.values(), alpha=0.7, color='red')
                    ax.set_ylabel('Signal Present')
                    ax.tick_params(axis='x', rotation=45)
                    ax.set_title(f'Crisis Indicators (Risk: {overall_risk})', fontweight='bold')
                else:
                    ax.text(0.5, 0.5, f'No crisis signals\n(Risk: {overall_risk})', ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, 'No crisis data', ha='center', va='center', transform=ax.transAxes)
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Plot error: {str(e)[:50]}', ha='center', va='center', transform=ax.transAxes)
            
    def _plot_policy_implications(self, ax, policy: Dict[str, Any]) -> None:
        """Plot policy implications."""
        try:
            ax.set_title('Policy Recommendations', fontweight='bold')
            
            if 'priority_level' in policy:
                categories = ['fiscal_policy', 'monetary_policy', 'structural_reforms', 'capital_controls']
                rec_counts = [len(policy.get(cat, [])) for cat in categories]
                labels = ['Fiscal', 'Monetary', 'Structural', 'Capital Controls']
                
                if sum(rec_counts) > 0:
                    ax.bar(labels, rec_counts, alpha=0.7)
                    ax.set_ylabel('Number of Recommendations')
                    ax.tick_params(axis='x', rotation=45)
                    priority = policy.get('priority_level', 'unknown')
                    ax.set_title(f'Policy Recommendations (Priority: {priority})', fontweight='bold')
                else:
                    priority = policy.get('priority_level', 'unknown')
                    ax.text(0.5, 0.5, f'No recommendations\n(Priority: {priority})', ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, 'No policy data', ha='center', va='center', transform=ax.transAxes)
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Plot error: {str(e)[:50]}', ha='center', va='center', transform=ax.transAxes)

if __name__ == "__main__":
    # Example usage
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='M')
    n_periods = len(dates)
    
    # Generate sample data
    np.random.seed(42)
    ca_data = CurrentAccountData(
        trade_balance=pd.Series(np.random.normal(-1000, 500, n_periods), index=dates),
        services_balance=pd.Series(np.random.normal(200, 100, n_periods), index=dates),
        primary_income=pd.Series(np.random.normal(-100, 50, n_periods), index=dates),
        secondary_income=pd.Series(np.random.normal(50, 25, n_periods), index=dates),
        current_account_balance=pd.Series(np.random.normal(-850, 400, n_periods), index=dates)
    )
    
    # Generate financial account data
    fa_data = CapitalAccountData(
        capital_account=pd.Series(np.random.normal(10, 5, n_periods), index=dates),
        direct_investment=pd.Series(np.random.normal(300, 100, n_periods), index=dates),
        portfolio_investment=pd.Series(np.random.normal(200, 200, n_periods), index=dates),
        other_investment=pd.Series(np.random.normal(100, 150, n_periods), index=dates),
        reserve_assets=pd.Series(np.random.normal(50, 100, n_periods), index=dates),
        financial_account_balance=pd.Series(np.random.normal(660, 300, n_periods), index=dates)
    )
    
    # Generate supporting data
    gdp_data = pd.Series(np.random.normal(50000, 2000, n_periods), index=dates)
    exchange_rates = pd.Series(np.random.normal(1.2, 0.1, n_periods), index=dates)
    interest_rates = pd.Series(np.random.normal(2.5, 0.5, n_periods), index=dates)
    external_debt = pd.Series(np.random.normal(25000, 1000, n_periods), index=dates)
    
    # Create BOP data dictionary
    bop_data = {
        'current_account_data': ca_data,
        'financial_account_data': fa_data,
        'gdp_data': gdp_data,
        'exchange_rates': exchange_rates,
        'interest_rates': interest_rates,
        'external_debt': external_debt
    }
    
    # Run analysis
    analyzer = BalanceOfPaymentsAnalyzer()
    
    # Perform analysis
    print("Running Balance of Payments Analysis...")
    results = analyzer.analyze(bop_data)
    
    # Print summary
    print("\n=== BALANCE OF PAYMENTS ANALYSIS SUMMARY ===")
    
    if results.current_account_analysis:
        print("\nCurrent Account Analysis:")
        if 'sustainability' in results.current_account_analysis:
            ca_risk = results.current_account_analysis['sustainability'].get('risk_level', 'unknown')
            print(f"  - Sustainability Risk: {ca_risk}")

    if results.sustainability_metrics:
        print(f"\nSustainability Metrics:")
        print(f"  - Overall Score: {results.sustainability_metrics['sustainability_score']:.1f}/100")
        print(f"  - Risk Level: {results.sustainability_metrics['risk_level']}")

    if results.crisis_indicators:
        print(f"\nCrisis Indicators:")
        print(f"  - Overall Risk: {results.crisis_indicators['overall_risk']}")
        print(f"  - Crisis Signals: {len(results.crisis_indicators['crisis_signals'])}")

    if results.exchange_rate_pressure:
        print(f"\nExchange Rate Pressure:")
        if 'early_warning_signals' in results.exchange_rate_pressure:
            warning_level = results.exchange_rate_pressure['early_warning_signals'].get('warning_level', 'unknown')
            print(f"  - Warning Level: {warning_level}")

    if results.policy_implications:
        print(f"\nPolicy Implications:")
        print(f"  - Priority Level: {results.policy_implications['priority_level']}")
        total_recommendations = sum(len(results.policy_implications.get(category, [])) 
                                  for category in ['fiscal_policy', 'monetary_policy', 
                                                 'structural_reforms', 'capital_controls'])
        print(f"  - Total Recommendations: {total_recommendations}")
    
    # Generate plots
    try:
        print("\nGenerating analysis plots...")
        analyzer.plot_analysis(results)
        print("Analysis complete!")
    except Exception as e:
        print(f"Error generating plots: {e}")
        print("Analysis complete (without plots)!")