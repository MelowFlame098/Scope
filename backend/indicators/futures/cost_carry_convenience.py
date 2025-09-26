from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, optimize
import warnings
warnings.filterwarnings('ignore')

@dataclass
class FuturesData:
    """Structure for futures market data"""
    spot_prices: List[float]
    futures_prices: List[float]
    timestamps: List[datetime]
    risk_free_rate: List[float]
    dividend_yield: List[float]
    storage_costs: List[float]
    time_to_maturity: List[float]  # in years
    contract_symbol: str
    underlying_asset: str
    
@dataclass
class CostOfCarryResult:
    """Results from cost-of-carry model"""
    theoretical_price: float
    actual_price: float
    mispricing: float
    mispricing_percentage: float
    cost_of_carry: float
    arbitrage_opportunity: bool
    arbitrage_profit: float
    
@dataclass
class ConvenienceYieldResult:
    """Results from convenience yield analysis"""
    convenience_yield: float
    implied_convenience_yield: float
    storage_premium: float
    backwardation_contango: str
    yield_volatility: float
    seasonal_pattern: Dict[str, float]
    
@dataclass
class FuturesAnalysisResult:
    """Comprehensive futures analysis results"""
    cost_carry_results: List[CostOfCarryResult]
    convenience_yield_results: List[ConvenienceYieldResult]
    arbitrage_opportunities: List[Dict[str, Any]]
    risk_metrics: Dict[str, float]
    trading_signals: List[str]
    insights: List[str]
    recommendations: List[str]
    model_performance: Dict[str, float]

class CostOfCarryModel:
    """Cost-of-carry model for futures pricing"""
    
    def __init__(self):
        self.model_name = "Cost-of-Carry Model"
        
    def calculate_theoretical_price(self, spot_price: float, risk_free_rate: float,
                                  dividend_yield: float, storage_cost: float,
                                  time_to_maturity: float) -> float:
        """Calculate theoretical futures price using cost-of-carry model"""
        # F = S * e^((r - q + u) * T)
        # Where: F = futures price, S = spot price, r = risk-free rate
        # q = dividend yield, u = storage cost, T = time to maturity
        
        cost_of_carry = risk_free_rate - dividend_yield + storage_cost
        theoretical_price = spot_price * np.exp(cost_of_carry * time_to_maturity)
        
        return theoretical_price
    
    def analyze_mispricing(self, futures_data: FuturesData) -> List[CostOfCarryResult]:
        """Analyze mispricing using cost-of-carry model"""
        results = []
        
        for i in range(len(futures_data.spot_prices)):
            theoretical_price = self.calculate_theoretical_price(
                futures_data.spot_prices[i],
                futures_data.risk_free_rate[i],
                futures_data.dividend_yield[i],
                futures_data.storage_costs[i],
                futures_data.time_to_maturity[i]
            )
            
            actual_price = futures_data.futures_prices[i]
            mispricing = actual_price - theoretical_price
            mispricing_percentage = (mispricing / theoretical_price) * 100
            
            cost_of_carry = (futures_data.risk_free_rate[i] - 
                           futures_data.dividend_yield[i] + 
                           futures_data.storage_costs[i])
            
            # Determine arbitrage opportunity (threshold: 1% mispricing)
            arbitrage_opportunity = abs(mispricing_percentage) > 1.0
            arbitrage_profit = abs(mispricing) if arbitrage_opportunity else 0.0
            
            results.append(CostOfCarryResult(
                theoretical_price=theoretical_price,
                actual_price=actual_price,
                mispricing=mispricing,
                mispricing_percentage=mispricing_percentage,
                cost_of_carry=cost_of_carry,
                arbitrage_opportunity=arbitrage_opportunity,
                arbitrage_profit=arbitrage_profit
            ))
        
        return results
    
    def identify_arbitrage_opportunities(self, results: List[CostOfCarryResult]) -> List[Dict[str, Any]]:
        """Identify and categorize arbitrage opportunities"""
        opportunities = []
        
        for i, result in enumerate(results):
            if result.arbitrage_opportunity:
                if result.mispricing > 0:
                    # Futures overpriced - sell futures, buy underlying
                    strategy = "Sell Futures, Buy Underlying"
                    action = "SHORT_FUTURES_LONG_SPOT"
                else:
                    # Futures underpriced - buy futures, sell underlying
                    strategy = "Buy Futures, Sell Underlying"
                    action = "LONG_FUTURES_SHORT_SPOT"
                
                opportunities.append({
                    'timestamp': i,
                    'strategy': strategy,
                    'action': action,
                    'mispricing': result.mispricing,
                    'mispricing_percentage': result.mispricing_percentage,
                    'expected_profit': result.arbitrage_profit,
                    'theoretical_price': result.theoretical_price,
                    'actual_price': result.actual_price
                })
        
        return opportunities

class ConvenienceYieldModel:
    """Convenience yield model for futures analysis"""
    
    def __init__(self):
        self.model_name = "Convenience Yield Model"
        
    def calculate_convenience_yield(self, spot_price: float, futures_price: float,
                                  risk_free_rate: float, storage_cost: float,
                                  time_to_maturity: float) -> float:
        """Calculate convenience yield from market prices"""
        # F = S * e^((r - q + u - c) * T)
        # Solving for convenience yield c:
        # c = r + u - q - ln(F/S) / T
        
        if time_to_maturity <= 0 or spot_price <= 0 or futures_price <= 0:
            return 0.0
        
        convenience_yield = (risk_free_rate + storage_cost - 
                           np.log(futures_price / spot_price) / time_to_maturity)
        
        return convenience_yield
    
    def analyze_convenience_yield(self, futures_data: FuturesData) -> List[ConvenienceYieldResult]:
        """Analyze convenience yield patterns"""
        results = []
        convenience_yields = []
        
        for i in range(len(futures_data.spot_prices)):
            convenience_yield = self.calculate_convenience_yield(
                futures_data.spot_prices[i],
                futures_data.futures_prices[i],
                futures_data.risk_free_rate[i],
                futures_data.storage_costs[i],
                futures_data.time_to_maturity[i]
            )
            
            convenience_yields.append(convenience_yield)
            
            # Calculate storage premium
            storage_premium = convenience_yield - futures_data.storage_costs[i]
            
            # Determine backwardation or contango
            if futures_data.futures_prices[i] < futures_data.spot_prices[i]:
                market_structure = "Backwardation"
            elif futures_data.futures_prices[i] > futures_data.spot_prices[i]:
                market_structure = "Contango"
            else:
                market_structure = "Neutral"
            
            results.append(ConvenienceYieldResult(
                convenience_yield=convenience_yield,
                implied_convenience_yield=convenience_yield,
                storage_premium=storage_premium,
                backwardation_contango=market_structure,
                yield_volatility=0.0,  # Will be calculated later
                seasonal_pattern={}
            ))
        
        # Calculate yield volatility
        if len(convenience_yields) > 1:
            yield_volatility = np.std(convenience_yields)
            for result in results:
                result.yield_volatility = yield_volatility
        
        # Analyze seasonal patterns
        seasonal_pattern = self._analyze_seasonal_patterns(futures_data.timestamps, convenience_yields)
        for result in results:
            result.seasonal_pattern = seasonal_pattern
        
        return results
    
    def _analyze_seasonal_patterns(self, timestamps: List[datetime], 
                                 convenience_yields: List[float]) -> Dict[str, float]:
        """Analyze seasonal patterns in convenience yield"""
        if len(timestamps) != len(convenience_yields):
            return {}
        
        # Group by month
        monthly_yields = {}
        for timestamp, yield_val in zip(timestamps, convenience_yields):
            month = timestamp.month
            if month not in monthly_yields:
                monthly_yields[month] = []
            monthly_yields[month].append(yield_val)
        
        # Calculate average yield by month
        seasonal_pattern = {}
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        for month, yields in monthly_yields.items():
            if yields:
                seasonal_pattern[month_names[month-1]] = np.mean(yields)
        
        return seasonal_pattern

class FuturesAnalyzer:
    """Comprehensive futures analyzer combining cost-of-carry and convenience yield models"""
    
    def __init__(self):
        self.cost_carry_model = CostOfCarryModel()
        self.convenience_yield_model = ConvenienceYieldModel()
        
    def analyze(self, futures_data: FuturesData) -> FuturesAnalysisResult:
        """Perform comprehensive futures analysis"""
        
        print(f"Analyzing futures data for {futures_data.contract_symbol}...")
        
        # Cost-of-carry analysis
        cost_carry_results = self.cost_carry_model.analyze_mispricing(futures_data)
        
        # Convenience yield analysis
        convenience_yield_results = self.convenience_yield_model.analyze_convenience_yield(futures_data)
        
        # Identify arbitrage opportunities
        arbitrage_opportunities = self.cost_carry_model.identify_arbitrage_opportunities(cost_carry_results)
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(cost_carry_results, convenience_yield_results)
        
        # Generate trading signals
        trading_signals = self._generate_trading_signals(cost_carry_results, convenience_yield_results)
        
        # Generate insights
        insights = self._generate_insights(cost_carry_results, convenience_yield_results, arbitrage_opportunities)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(risk_metrics, arbitrage_opportunities)
        
        # Calculate model performance
        model_performance = self._calculate_model_performance(cost_carry_results)
        
        return FuturesAnalysisResult(
            cost_carry_results=cost_carry_results,
            convenience_yield_results=convenience_yield_results,
            arbitrage_opportunities=arbitrage_opportunities,
            risk_metrics=risk_metrics,
            trading_signals=trading_signals,
            insights=insights,
            recommendations=recommendations,
            model_performance=model_performance
        )
    
    def _calculate_risk_metrics(self, cost_carry_results: List[CostOfCarryResult],
                              convenience_yield_results: List[ConvenienceYieldResult]) -> Dict[str, float]:
        """Calculate risk metrics"""
        if not cost_carry_results or not convenience_yield_results:
            return {}
        
        mispricings = [result.mispricing_percentage for result in cost_carry_results]
        convenience_yields = [result.convenience_yield for result in convenience_yield_results]
        
        return {
            'mispricing_volatility': np.std(mispricings),
            'average_mispricing': np.mean(np.abs(mispricings)),
            'max_mispricing': max(np.abs(mispricings)),
            'convenience_yield_volatility': np.std(convenience_yields),
            'average_convenience_yield': np.mean(convenience_yields),
            'arbitrage_frequency': sum(1 for result in cost_carry_results if result.arbitrage_opportunity) / len(cost_carry_results),
            'backwardation_frequency': sum(1 for result in convenience_yield_results if result.backwardation_contango == 'Backwardation') / len(convenience_yield_results)
        }
    
    def _generate_trading_signals(self, cost_carry_results: List[CostOfCarryResult],
                                convenience_yield_results: List[ConvenienceYieldResult]) -> List[str]:
        """Generate trading signals based on analysis"""
        signals = []
        
        for i, (cc_result, cy_result) in enumerate(zip(cost_carry_results, convenience_yield_results)):
            # Signal based on mispricing
            if cc_result.mispricing_percentage > 2.0:
                signals.append('SELL_FUTURES')  # Overpriced
            elif cc_result.mispricing_percentage < -2.0:
                signals.append('BUY_FUTURES')   # Underpriced
            # Signal based on convenience yield
            elif cy_result.convenience_yield > 0.05:  # High convenience yield
                signals.append('BUY_SPOT')      # Physical asset more valuable
            elif cy_result.convenience_yield < -0.02: # Negative convenience yield
                signals.append('BUY_FUTURES')   # Futures more attractive
            else:
                signals.append('HOLD')
        
        return signals
    
    def _generate_insights(self, cost_carry_results: List[CostOfCarryResult],
                         convenience_yield_results: List[ConvenienceYieldResult],
                         arbitrage_opportunities: List[Dict[str, Any]]) -> List[str]:
        """Generate analytical insights"""
        insights = []
        
        if not cost_carry_results or not convenience_yield_results:
            return insights
        
        # Mispricing analysis
        avg_mispricing = np.mean([abs(result.mispricing_percentage) for result in cost_carry_results])
        if avg_mispricing > 1.5:
            insights.append(f"High average mispricing detected: {avg_mispricing:.2f}%")
        
        # Arbitrage frequency
        arbitrage_freq = len(arbitrage_opportunities) / len(cost_carry_results)
        if arbitrage_freq > 0.2:
            insights.append(f"Frequent arbitrage opportunities: {arbitrage_freq:.1%} of observations")
        
        # Market structure analysis
        backwardation_count = sum(1 for result in convenience_yield_results 
                                if result.backwardation_contango == 'Backwardation')
        contango_count = sum(1 for result in convenience_yield_results 
                           if result.backwardation_contango == 'Contango')
        
        if backwardation_count > contango_count:
            insights.append("Market predominantly in backwardation - supply constraints likely")
        elif contango_count > backwardation_count:
            insights.append("Market predominantly in contango - ample supply conditions")
        
        # Convenience yield analysis
        avg_convenience_yield = np.mean([result.convenience_yield for result in convenience_yield_results])
        if avg_convenience_yield > 0.03:
            insights.append("High convenience yield suggests strong demand for physical asset")
        elif avg_convenience_yield < 0:
            insights.append("Negative convenience yield indicates storage burden exceeds benefits")
        
        # Volatility analysis
        convenience_yield_vol = np.std([result.convenience_yield for result in convenience_yield_results])
        if convenience_yield_vol > 0.05:
            insights.append("High convenience yield volatility indicates unstable supply-demand dynamics")
        
        return insights
    
    def _generate_recommendations(self, risk_metrics: Dict[str, float],
                                arbitrage_opportunities: List[Dict[str, Any]]) -> List[str]:
        """Generate trading recommendations"""
        recommendations = []
        
        # Risk-based recommendations
        mispricing_vol = risk_metrics.get('mispricing_volatility', 0)
        if mispricing_vol > 2.0:
            recommendations.append("High mispricing volatility - use smaller position sizes")
        
        arbitrage_freq = risk_metrics.get('arbitrage_frequency', 0)
        if arbitrage_freq > 0.3:
            recommendations.append("Frequent arbitrage opportunities - consider systematic arbitrage strategy")
        
        backwardation_freq = risk_metrics.get('backwardation_frequency', 0)
        if backwardation_freq > 0.7:
            recommendations.append("Persistent backwardation - consider long physical, short futures strategy")
        elif backwardation_freq < 0.3:
            recommendations.append("Persistent contango - consider storage arbitrage opportunities")
        
        # Opportunity-based recommendations
        if arbitrage_opportunities:
            avg_profit = np.mean([opp['expected_profit'] for opp in arbitrage_opportunities])
            recommendations.append(f"Average arbitrage profit potential: {avg_profit:.2f}")
        
        recommendations.append("Monitor storage costs and convenience yield for optimal entry/exit timing")
        recommendations.append("Consider seasonal patterns in convenience yield for strategic positioning")
        
        return recommendations
    
    def _calculate_model_performance(self, cost_carry_results: List[CostOfCarryResult]) -> Dict[str, float]:
        """Calculate model performance metrics"""
        if not cost_carry_results:
            return {}
        
        mispricings = [abs(result.mispricing_percentage) for result in cost_carry_results]
        
        return {
            'mean_absolute_error': np.mean(mispricings),
            'root_mean_square_error': np.sqrt(np.mean([mp**2 for mp in mispricings])),
            'prediction_accuracy': sum(1 for mp in mispricings if mp < 1.0) / len(mispricings),
            'arbitrage_detection_rate': sum(1 for result in cost_carry_results if result.arbitrage_opportunity) / len(cost_carry_results)
        }
    
    def plot_results(self, futures_data: FuturesData, results: FuturesAnalysisResult):
        """Plot comprehensive futures analysis results"""
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        
        timestamps = futures_data.timestamps
        
        # Plot 1: Spot vs Futures prices
        ax1 = axes[0, 0]
        ax1.plot(timestamps, futures_data.spot_prices, label='Spot Price', linewidth=2)
        ax1.plot(timestamps, futures_data.futures_prices, label='Futures Price', linewidth=2)
        
        # Add theoretical prices
        theoretical_prices = [result.theoretical_price for result in results.cost_carry_results]
        ax1.plot(timestamps, theoretical_prices, label='Theoretical Price', linestyle='--', alpha=0.7)
        
        ax1.set_title('Spot vs Futures vs Theoretical Prices', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Mispricing analysis
        ax2 = axes[0, 1]
        mispricings = [result.mispricing_percentage for result in results.cost_carry_results]
        ax2.plot(timestamps, mispricings, color='red', linewidth=2)
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax2.axhline(y=1, color='orange', linestyle='--', alpha=0.5, label='Arbitrage Threshold')
        ax2.axhline(y=-1, color='orange', linestyle='--', alpha=0.5)
        ax2.set_title('Mispricing Analysis (%)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Mispricing (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Convenience yield
        ax3 = axes[1, 0]
        convenience_yields = [result.convenience_yield for result in results.convenience_yield_results]
        ax3.plot(timestamps, convenience_yields, color='green', linewidth=2)
        ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax3.set_title('Convenience Yield Over Time', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Convenience Yield')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Market structure (Backwardation/Contango)
        ax4 = axes[1, 1]
        basis = [futures_data.futures_prices[i] - futures_data.spot_prices[i] 
                for i in range(len(futures_data.futures_prices))]
        colors = ['red' if b < 0 else 'blue' for b in basis]
        ax4.bar(range(len(basis)), basis, color=colors, alpha=0.7)
        ax4.axhline(y=0, color='black', linewidth=1)
        ax4.set_title('Market Structure (Basis = Futures - Spot)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Time Period')
        ax4.set_ylabel('Basis')
        ax4.grid(True, alpha=0.3)
        
        # Add legend for colors
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', alpha=0.7, label='Backwardation'),
                          Patch(facecolor='blue', alpha=0.7, label='Contango')]
        ax4.legend(handles=legend_elements)
        
        # Plot 5: Trading signals
        ax5 = axes[2, 0]
        signal_mapping = {'BUY_FUTURES': 1, 'SELL_FUTURES': -1, 'BUY_SPOT': 0.5, 'HOLD': 0}
        signal_values = [signal_mapping.get(signal, 0) for signal in results.trading_signals]
        
        colors = ['green' if s > 0 else 'red' if s < 0 else 'gray' for s in signal_values]
        ax5.bar(range(len(signal_values)), signal_values, color=colors, alpha=0.7)
        ax5.set_title('Trading Signals', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Time Period')
        ax5.set_ylabel('Signal Strength')
        ax5.set_ylim(-1.5, 1.5)
        
        # Add signal legend
        signal_legend = [Patch(facecolor='green', alpha=0.7, label='Buy'),
                        Patch(facecolor='red', alpha=0.7, label='Sell'),
                        Patch(facecolor='gray', alpha=0.7, label='Hold')]
        ax5.legend(handles=signal_legend)
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Risk metrics
        ax6 = axes[2, 1]
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
    
    def generate_report(self, futures_data: FuturesData, results: FuturesAnalysisResult) -> str:
        """Generate comprehensive futures analysis report"""
        report = []
        report.append("=== FUTURES COST-OF-CARRY & CONVENIENCE YIELD ANALYSIS REPORT ===")
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Contract: {futures_data.contract_symbol}")
        report.append(f"Underlying Asset: {futures_data.underlying_asset}")
        report.append(f"Analysis Period: {len(futures_data.timestamps)} observations")
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
        
        # Arbitrage Opportunities
        report.append(f"ARBITRAGE OPPORTUNITIES: {len(results.arbitrage_opportunities)}")
        if results.arbitrage_opportunities:
            total_profit = sum(opp['expected_profit'] for opp in results.arbitrage_opportunities)
            avg_profit = total_profit / len(results.arbitrage_opportunities)
            report.append(f"Total Expected Profit: {total_profit:.2f}")
            report.append(f"Average Profit per Opportunity: {avg_profit:.2f}")
            
            # Top opportunities
            sorted_opps = sorted(results.arbitrage_opportunities, 
                               key=lambda x: x['expected_profit'], reverse=True)
            report.append("Top 3 Opportunities:")
            for i, opp in enumerate(sorted_opps[:3]):
                report.append(f"{i+1}. {opp['strategy']} - Profit: {opp['expected_profit']:.2f} ({opp['mispricing_percentage']:.2f}%)")
        report.append("")
        
        # Market Structure Analysis
        backwardation_count = sum(1 for result in results.convenience_yield_results 
                                if result.backwardation_contango == 'Backwardation')
        contango_count = sum(1 for result in results.convenience_yield_results 
                           if result.backwardation_contango == 'Contango')
        
        report.append("MARKET STRUCTURE:")
        report.append(f"Backwardation Periods: {backwardation_count} ({backwardation_count/len(results.convenience_yield_results)*100:.1f}%)")
        report.append(f"Contango Periods: {contango_count} ({contango_count/len(results.convenience_yield_results)*100:.1f}%)")
        report.append("")
        
        # Convenience Yield Analysis
        convenience_yields = [result.convenience_yield for result in results.convenience_yield_results]
        report.append("CONVENIENCE YIELD ANALYSIS:")
        report.append(f"Average Convenience Yield: {np.mean(convenience_yields):.4f}")
        report.append(f"Convenience Yield Volatility: {np.std(convenience_yields):.4f}")
        report.append(f"Max Convenience Yield: {max(convenience_yields):.4f}")
        report.append(f"Min Convenience Yield: {min(convenience_yields):.4f}")
        report.append("")
        
        # Trading Signals Summary
        from collections import Counter
        signal_counts = Counter(results.trading_signals)
        report.append("TRADING SIGNALS SUMMARY:")
        for signal, count in signal_counts.items():
            percentage = (count / len(results.trading_signals)) * 100
            report.append(f"{signal}: {count} ({percentage:.1f}%)")
        report.append("")
        
        # Insights
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
    
    # Create sample data for crude oil futures
    n_periods = 100
    base_spot_price = 70.0
    base_futures_price = 72.0
    
    # Generate realistic time series
    timestamps = [datetime.now() - timedelta(days=i) for i in range(n_periods, 0, -1)]
    
    # Spot prices with some volatility
    spot_returns = np.random.normal(0, 0.02, n_periods)
    spot_prices = [base_spot_price]
    for ret in spot_returns[1:]:
        spot_prices.append(spot_prices[-1] * (1 + ret))
    
    # Futures prices with basis relationship
    basis_changes = np.random.normal(0, 0.01, n_periods)
    futures_prices = []
    for i, spot in enumerate(spot_prices):
        basis = 2.0 + basis_changes[i]  # Base contango of $2
        futures_prices.append(spot + basis)
    
    # Risk-free rates (varying around 3%)
    risk_free_rates = [0.03 + np.random.normal(0, 0.005) for _ in range(n_periods)]
    
    # Dividend yields (for equity index futures, 0 for commodities)
    dividend_yields = [0.0] * n_periods
    
    # Storage costs (varying around 1% annually)
    storage_costs = [0.01 + np.random.normal(0, 0.002) for _ in range(n_periods)]
    
    # Time to maturity (decreasing from 0.25 years to near 0)
    time_to_maturity = [0.25 - (i / n_periods) * 0.24 for i in range(n_periods)]
    
    # Create futures data
    futures_data = FuturesData(
        spot_prices=spot_prices,
        futures_prices=futures_prices,
        timestamps=timestamps,
        risk_free_rate=risk_free_rates,
        dividend_yield=dividend_yields,
        storage_costs=storage_costs,
        time_to_maturity=time_to_maturity,
        contract_symbol="CLZ23",
        underlying_asset="Crude Oil"
    )
    
    # Initialize analyzer
    analyzer = FuturesAnalyzer()
    
    try:
        # Perform analysis
        print("Starting Futures Cost-of-Carry and Convenience Yield Analysis...")
        results = analyzer.analyze(futures_data)
        
        # Print summary
        print("\n=== ANALYSIS SUMMARY ===")
        print(f"Contract: {futures_data.contract_symbol} ({futures_data.underlying_asset})")
        print(f"Analysis Period: {len(futures_data.timestamps)} observations")
        
        print("\nModel Performance:")
        for metric, value in results.model_performance.items():
            print(f"{metric}: {value:.4f}")
        
        print("\nRisk Metrics:")
        for metric, value in results.risk_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        print(f"\nArbitrage Opportunities: {len(results.arbitrage_opportunities)}")
        if results.arbitrage_opportunities:
            total_profit = sum(opp['expected_profit'] for opp in results.arbitrage_opportunities)
            print(f"Total Expected Profit: {total_profit:.2f}")
        
        print("\nKey Insights:")
        for insight in results.insights[:5]:
            print(f"• {insight}")
        
        print("\nRecommendations:")
        for rec in results.recommendations[:3]:
            print(f"• {rec}")
        
        # Generate report
        report = analyzer.generate_report(futures_data, results)
        
        # Plot results
        try:
            analyzer.plot_results(futures_data, results)
        except Exception as e:
            print(f"Plotting failed: {e}")
        
        print("\nFutures analysis completed successfully!")
        
    except Exception as e:
        print(f"Futures analysis failed: {e}")
        import traceback
        traceback.print_exc()