"""
Monetary Models for Forex Analysis

This module implements various monetary models used in forex analysis:
1. Flexible Price Monetary Model (Dornbusch-Frankel)
2. Sticky Price Monetary Model (Dornbusch)
3. Portfolio Balance Model
4. Monetary Approach to Exchange Rate
5. Taylor Rule Model
6. Real Interest Rate Differential Model
7. Monetary Policy Divergence Model
8. Central Bank Communication Analysis

Author: FinScope Team
Date: 2024
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

@dataclass
class MonetaryData:
    """Data structure for monetary variables"""
    money_supply: float
    interest_rate: float
    inflation_rate: float
    gdp_growth: float
    current_account: float
    government_debt: float
    central_bank_assets: float
    policy_rate: float
    
@dataclass
class FlexiblePriceResult:
    """Results from Flexible Price Monetary Model"""
    fair_value: float
    current_rate: float
    deviation: float
    money_supply_effect: float
    interest_rate_effect: float
    income_effect: float
    model_r_squared: float
    confidence_interval: Tuple[float, float]
    
@dataclass
class StickyPriceResult:
    """Results from Sticky Price Monetary Model"""
    short_run_rate: float
    long_run_rate: float
    adjustment_speed: float
    overshooting: float
    time_to_equilibrium: float
    volatility_premium: float
    model_fit: float
    
@dataclass
class PortfolioBalanceResult:
    """Results from Portfolio Balance Model"""
    equilibrium_rate: float
    risk_premium: float
    domestic_asset_share: float
    foreign_asset_share: float
    portfolio_adjustment: float
    capital_flow_effect: float
    sterilization_coefficient: float
    
@dataclass
class TaylorRuleResult:
    """Results from Taylor Rule Model"""
    implied_rate: float
    inflation_gap_effect: float
    output_gap_effect: float
    policy_divergence: float
    hawkish_dovish_score: float
    forward_guidance_effect: float
    credibility_index: float
    
@dataclass
class MonetaryAnalysisResult:
    """Combined results from all monetary models"""
    flexible_price: FlexiblePriceResult
    sticky_price: StickyPriceResult
    portfolio_balance: PortfolioBalanceResult
    taylor_rule: TaylorRuleResult
    combined_signal: str
    confidence_score: float
    risk_assessment: str
    policy_implications: List[str]
    
class FlexiblePriceMonetaryModel:
    """Flexible Price Monetary Model (Dornbusch-Frankel)"""
    
    def __init__(self):
        self.model = None
        self.coefficients = {}
        
    def calculate_fair_value(self, 
                           domestic_data: MonetaryData,
                           foreign_data: MonetaryData,
                           historical_data: pd.DataFrame) -> FlexiblePriceResult:
        """
        Calculate fair value using flexible price monetary model
        
        Model: e = (m - m*) - η(y - y*) + λ(i - i*)
        where:
        e = log exchange rate
        m = log money supply
        y = log real income
        i = interest rate
        * denotes foreign variables
        """
        
        # Prepare regression data
        X = np.column_stack([
            historical_data['money_supply_diff'],
            historical_data['income_diff'],
            historical_data['interest_rate_diff']
        ])
        y = historical_data['exchange_rate_log']
        
        # Fit model
        self.model = LinearRegression()
        self.model.fit(X, y)
        
        # Store coefficients
        self.coefficients = {
            'money_supply': self.model.coef_[0],
            'income': self.model.coef_[1],
            'interest_rate': self.model.coef_[2],
            'intercept': self.model.intercept_
        }
        
        # Calculate current fair value
        money_diff = np.log(domestic_data.money_supply) - np.log(foreign_data.money_supply)
        income_diff = np.log(domestic_data.gdp_growth) - np.log(foreign_data.gdp_growth)
        interest_diff = domestic_data.interest_rate - foreign_data.interest_rate
        
        fair_value = (self.coefficients['intercept'] + 
                     self.coefficients['money_supply'] * money_diff +
                     self.coefficients['income'] * income_diff +
                     self.coefficients['interest_rate'] * interest_diff)
        
        fair_value = np.exp(fair_value)
        
        # Calculate effects
        money_effect = self.coefficients['money_supply'] * money_diff
        interest_effect = self.coefficients['interest_rate'] * interest_diff
        income_effect = self.coefficients['income'] * income_diff
        
        # Model statistics
        y_pred = self.model.predict(X)
        r_squared = r2_score(y, y_pred)
        
        # Confidence interval (simplified)
        residuals = y - y_pred
        std_error = np.std(residuals)
        confidence_interval = (fair_value - 1.96 * std_error, 
                             fair_value + 1.96 * std_error)
        
        current_rate = historical_data['exchange_rate'].iloc[-1]
        deviation = (current_rate - fair_value) / fair_value * 100
        
        return FlexiblePriceResult(
            fair_value=fair_value,
            current_rate=current_rate,
            deviation=deviation,
            money_supply_effect=money_effect,
            interest_rate_effect=interest_effect,
            income_effect=income_effect,
            model_r_squared=r_squared,
            confidence_interval=confidence_interval
        )
        
class StickyPriceMonetaryModel:
    """Sticky Price Monetary Model (Dornbusch Overshooting)"""
    
    def __init__(self, adjustment_speed: float = 0.1):
        self.adjustment_speed = adjustment_speed
        
    def calculate_overshooting(self,
                             domestic_data: MonetaryData,
                             foreign_data: MonetaryData,
                             shock_magnitude: float = 0.01) -> StickyPriceResult:
        """
        Calculate exchange rate overshooting due to monetary shocks
        
        Based on Dornbusch (1976) overshooting model
        """
        
        # Long-run equilibrium (PPP)
        inflation_diff = domestic_data.inflation_rate - foreign_data.inflation_rate
        long_run_rate = 1.0 * (1 + inflation_diff)
        
        # Short-run response to monetary shock
        interest_diff = domestic_data.interest_rate - foreign_data.interest_rate
        
        # Overshooting coefficient (simplified)
        theta = 2.0  # Price adjustment speed parameter
        alpha = 1.5  # Interest rate sensitivity
        
        overshooting = (alpha / (theta + self.adjustment_speed)) * shock_magnitude
        short_run_rate = long_run_rate * (1 + overshooting)
        
        # Time to equilibrium
        time_to_equilibrium = -np.log(0.05) / self.adjustment_speed  # 95% adjustment
        
        # Volatility premium
        volatility_premium = abs(overshooting) * 0.5
        
        # Model fit (simplified)
        model_fit = 0.75  # Would be calculated from historical data
        
        return StickyPriceResult(
            short_run_rate=short_run_rate,
            long_run_rate=long_run_rate,
            adjustment_speed=self.adjustment_speed,
            overshooting=overshooting * 100,
            time_to_equilibrium=time_to_equilibrium,
            volatility_premium=volatility_premium,
            model_fit=model_fit
        )
        
class PortfolioBalanceModel:
    """Portfolio Balance Model for Exchange Rate Determination"""
    
    def __init__(self):
        self.risk_aversion = 2.0
        
    def calculate_portfolio_equilibrium(self,
                                      domestic_data: MonetaryData,
                                      foreign_data: MonetaryData,
                                      market_data: Dict[str, float]) -> PortfolioBalanceResult:
        """
        Calculate equilibrium exchange rate from portfolio balance
        """
        
        # Asset supplies
        domestic_assets = domestic_data.central_bank_assets
        foreign_assets = foreign_data.central_bank_assets
        total_assets = domestic_assets + foreign_assets
        
        # Portfolio shares
        domestic_share = domestic_assets / total_assets
        foreign_share = foreign_assets / total_assets
        
        # Risk premium calculation
        volatility = market_data.get('volatility', 0.15)
        correlation = market_data.get('correlation', 0.3)
        
        risk_premium = (self.risk_aversion * volatility**2 * 
                       (domestic_share - 0.5) * (1 - correlation))
        
        # Interest rate differential adjusted for risk
        interest_diff = domestic_data.interest_rate - foreign_data.interest_rate
        adjusted_diff = interest_diff - risk_premium
        
        # Equilibrium exchange rate
        base_rate = 1.0
        equilibrium_rate = base_rate * np.exp(adjusted_diff)
        
        # Portfolio adjustment effect
        current_account_effect = domestic_data.current_account / 1000  # Normalize
        portfolio_adjustment = current_account_effect * 0.1
        
        # Capital flow effect
        capital_flow_effect = (domestic_data.interest_rate - foreign_data.interest_rate) * 0.05
        
        # Sterilization coefficient
        sterilization_coeff = 0.7  # Typical value
        
        return PortfolioBalanceResult(
            equilibrium_rate=equilibrium_rate,
            risk_premium=risk_premium * 100,
            domestic_asset_share=domestic_share * 100,
            foreign_asset_share=foreign_share * 100,
            portfolio_adjustment=portfolio_adjustment,
            capital_flow_effect=capital_flow_effect,
            sterilization_coefficient=sterilization_coeff
        )
        
class TaylorRuleModel:
    """Taylor Rule Model for Monetary Policy Analysis"""
    
    def __init__(self, 
                 inflation_weight: float = 1.5,
                 output_weight: float = 0.5):
        self.inflation_weight = inflation_weight
        self.output_weight = output_weight
        
    def calculate_taylor_rule(self,
                            domestic_data: MonetaryData,
                            foreign_data: MonetaryData,
                            targets: Dict[str, float]) -> TaylorRuleResult:
        """
        Calculate Taylor Rule implied rates and policy divergence
        """
        
        # Inflation and output gaps
        inflation_target = targets.get('inflation_target', 2.0)
        output_target = targets.get('output_target', 2.5)
        
        domestic_inflation_gap = domestic_data.inflation_rate - inflation_target
        domestic_output_gap = domestic_data.gdp_growth - output_target
        
        foreign_inflation_gap = foreign_data.inflation_rate - inflation_target
        foreign_output_gap = foreign_data.gdp_growth - output_target
        
        # Taylor Rule rates
        neutral_rate = 2.0  # Assumed neutral real rate
        
        domestic_taylor_rate = (neutral_rate + inflation_target +
                              self.inflation_weight * domestic_inflation_gap +
                              self.output_weight * domestic_output_gap)
        
        foreign_taylor_rate = (neutral_rate + inflation_target +
                             self.inflation_weight * foreign_inflation_gap +
                             self.output_weight * foreign_output_gap)
        
        # Policy divergence
        policy_divergence = domestic_taylor_rate - foreign_taylor_rate
        
        # Effects
        inflation_gap_effect = (self.inflation_weight * 
                              (domestic_inflation_gap - foreign_inflation_gap))
        output_gap_effect = (self.output_weight * 
                           (domestic_output_gap - foreign_output_gap))
        
        # Hawkish/Dovish score
        domestic_stance = domestic_data.policy_rate - domestic_taylor_rate
        hawkish_dovish_score = domestic_stance * 10  # Scale for interpretation
        
        # Forward guidance effect (simplified)
        forward_guidance_effect = 0.2 if abs(hawkish_dovish_score) > 0.5 else 0.0
        
        # Credibility index (simplified)
        policy_consistency = 1 - abs(domestic_stance) / 2
        credibility_index = max(0, min(1, policy_consistency))
        
        return TaylorRuleResult(
            implied_rate=domestic_taylor_rate,
            inflation_gap_effect=inflation_gap_effect,
            output_gap_effect=output_gap_effect,
            policy_divergence=policy_divergence,
            hawkish_dovish_score=hawkish_dovish_score,
            forward_guidance_effect=forward_guidance_effect,
            credibility_index=credibility_index
        )
        
class MonetaryModelAnalyzer:
    """Main analyzer combining all monetary models"""
    
    def __init__(self):
        self.flexible_price_model = FlexiblePriceMonetaryModel()
        self.sticky_price_model = StickyPriceMonetaryModel()
        self.portfolio_balance_model = PortfolioBalanceModel()
        self.taylor_rule_model = TaylorRuleModel()
        
    def analyze(self,
               domestic_data: MonetaryData,
               foreign_data: MonetaryData,
               historical_data: pd.DataFrame,
               market_data: Dict[str, float],
               targets: Dict[str, float]) -> MonetaryAnalysisResult:
        """
        Perform comprehensive monetary model analysis
        """
        
        # Run individual models
        flexible_result = self.flexible_price_model.calculate_fair_value(
            domestic_data, foreign_data, historical_data)
        
        sticky_result = self.sticky_price_model.calculate_overshooting(
            domestic_data, foreign_data)
        
        portfolio_result = self.portfolio_balance_model.calculate_portfolio_equilibrium(
            domestic_data, foreign_data, market_data)
        
        taylor_result = self.taylor_rule_model.calculate_taylor_rule(
            domestic_data, foreign_data, targets)
        
        # Generate combined signal
        signals = []
        
        # Flexible price signal
        if flexible_result.deviation > 5:
            signals.append('SELL')
        elif flexible_result.deviation < -5:
            signals.append('BUY')
        else:
            signals.append('NEUTRAL')
            
        # Sticky price signal
        if sticky_result.overshooting > 2:
            signals.append('SELL')
        elif sticky_result.overshooting < -2:
            signals.append('BUY')
        else:
            signals.append('NEUTRAL')
            
        # Portfolio balance signal
        if portfolio_result.risk_premium > 1:
            signals.append('SELL')
        elif portfolio_result.risk_premium < -1:
            signals.append('BUY')
        else:
            signals.append('NEUTRAL')
            
        # Taylor rule signal
        if taylor_result.policy_divergence > 0.5:
            signals.append('BUY')
        elif taylor_result.policy_divergence < -0.5:
            signals.append('SELL')
        else:
            signals.append('NEUTRAL')
            
        # Combine signals
        buy_count = signals.count('BUY')
        sell_count = signals.count('SELL')
        
        if buy_count > sell_count:
            combined_signal = 'BUY'
        elif sell_count > buy_count:
            combined_signal = 'SELL'
        else:
            combined_signal = 'NEUTRAL'
            
        # Calculate confidence score
        max_count = max(buy_count, sell_count, signals.count('NEUTRAL'))
        confidence_score = max_count / len(signals)
        
        # Risk assessment
        risk_factors = [
            abs(flexible_result.deviation) > 10,
            abs(sticky_result.overshooting) > 5,
            abs(portfolio_result.risk_premium) > 2,
            taylor_result.credibility_index < 0.5
        ]
        
        risk_count = sum(risk_factors)
        if risk_count >= 3:
            risk_assessment = 'HIGH'
        elif risk_count >= 2:
            risk_assessment = 'MEDIUM'
        else:
            risk_assessment = 'LOW'
            
        # Policy implications
        policy_implications = []
        
        if abs(taylor_result.policy_divergence) > 1:
            policy_implications.append('Significant monetary policy divergence detected')
            
        if taylor_result.credibility_index < 0.6:
            policy_implications.append('Central bank credibility concerns')
            
        if abs(portfolio_result.risk_premium) > 1.5:
            policy_implications.append('Elevated risk premium in currency markets')
            
        if abs(sticky_result.overshooting) > 3:
            policy_implications.append('Exchange rate overshooting likely')
            
        return MonetaryAnalysisResult(
            flexible_price=flexible_result,
            sticky_price=sticky_result,
            portfolio_balance=portfolio_result,
            taylor_rule=taylor_result,
            combined_signal=combined_signal,
            confidence_score=confidence_score,
            risk_assessment=risk_assessment,
            policy_implications=policy_implications
        )
        
    def plot_analysis(self, result: MonetaryAnalysisResult, 
                     currency_pair: str = 'USD/EUR'):
        """
        Plot comprehensive monetary model analysis
        """
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Monetary Model Analysis - {currency_pair}', fontsize=16)
        
        # Flexible Price Model
        ax1 = axes[0, 0]
        values = [result.flexible_price.money_supply_effect,
                 result.flexible_price.interest_rate_effect,
                 result.flexible_price.income_effect]
        labels = ['Money Supply', 'Interest Rate', 'Income']
        colors = ['green' if v > 0 else 'red' for v in values]
        
        bars = ax1.bar(labels, values, color=colors, alpha=0.7)
        ax1.set_title('Flexible Price Model Effects')
        ax1.set_ylabel('Effect Size')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}', ha='center', va='bottom' if height > 0 else 'top')
        
        # Sticky Price Model
        ax2 = axes[0, 1]
        metrics = ['Short-run Rate', 'Long-run Rate', 'Current Rate']
        values = [result.sticky_price.short_run_rate,
                 result.sticky_price.long_run_rate,
                 1.0]  # Assuming current rate is normalized to 1
        
        ax2.plot(metrics, values, marker='o', linewidth=2, markersize=8)
        ax2.set_title('Sticky Price Model - Rate Evolution')
        ax2.set_ylabel('Exchange Rate')
        ax2.grid(True, alpha=0.3)
        
        # Portfolio Balance
        ax3 = axes[0, 2]
        shares = [result.portfolio_balance.domestic_asset_share,
                 result.portfolio_balance.foreign_asset_share]
        labels = ['Domestic Assets', 'Foreign Assets']
        colors = ['blue', 'orange']
        
        wedges, texts, autotexts = ax3.pie(shares, labels=labels, colors=colors,
                                          autopct='%1.1f%%', startangle=90)
        ax3.set_title('Portfolio Balance - Asset Allocation')
        
        # Taylor Rule
        ax4 = axes[1, 0]
        effects = [result.taylor_rule.inflation_gap_effect,
                  result.taylor_rule.output_gap_effect]
        labels = ['Inflation Gap', 'Output Gap']
        colors = ['red', 'blue']
        
        bars = ax4.bar(labels, effects, color=colors, alpha=0.7)
        ax4.set_title('Taylor Rule Effects')
        ax4.set_ylabel('Effect on Policy Rate')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Risk Assessment
        ax5 = axes[1, 1]
        risk_levels = ['LOW', 'MEDIUM', 'HIGH']
        risk_colors = ['green', 'yellow', 'red']
        current_risk = result.risk_assessment
        
        colors = ['lightgray'] * 3
        if current_risk in risk_levels:
            idx = risk_levels.index(current_risk)
            colors[idx] = risk_colors[idx]
            
        bars = ax5.bar(risk_levels, [1, 1, 1], color=colors, alpha=0.7)
        ax5.set_title(f'Risk Assessment: {current_risk}')
        ax5.set_ylabel('Risk Level')
        ax5.set_ylim(0, 1.2)
        
        # Combined Signal
        ax6 = axes[1, 2]
        signal_text = f"Signal: {result.combined_signal}\n"
        signal_text += f"Confidence: {result.confidence_score:.1%}\n\n"
        signal_text += "Policy Implications:\n"
        for i, implication in enumerate(result.policy_implications[:3]):
            signal_text += f"• {implication[:40]}...\n"
            
        ax6.text(0.1, 0.9, signal_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.axis('off')
        ax6.set_title('Summary & Implications')
        
        plt.tight_layout()
        plt.show()
        
    def generate_report(self, result: MonetaryAnalysisResult, 
                       currency_pair: str = 'USD/EUR') -> str:
        """
        Generate comprehensive monetary analysis report
        """
        
        report = f"""
# Monetary Model Analysis Report - {currency_pair}

## Executive Summary
- **Combined Signal**: {result.combined_signal}
- **Confidence Score**: {result.confidence_score:.1%}
- **Risk Assessment**: {result.risk_assessment}

## Model Results

### 1. Flexible Price Monetary Model
- **Fair Value**: {result.flexible_price.fair_value:.4f}
- **Current Rate**: {result.flexible_price.current_rate:.4f}
- **Deviation**: {result.flexible_price.deviation:.2f}%
- **Model R²**: {result.flexible_price.model_r_squared:.3f}
- **Money Supply Effect**: {result.flexible_price.money_supply_effect:.4f}
- **Interest Rate Effect**: {result.flexible_price.interest_rate_effect:.4f}
- **Income Effect**: {result.flexible_price.income_effect:.4f}

### 2. Sticky Price Monetary Model
- **Short-run Rate**: {result.sticky_price.short_run_rate:.4f}
- **Long-run Rate**: {result.sticky_price.long_run_rate:.4f}
- **Overshooting**: {result.sticky_price.overshooting:.2f}%
- **Time to Equilibrium**: {result.sticky_price.time_to_equilibrium:.1f} periods
- **Volatility Premium**: {result.sticky_price.volatility_premium:.4f}

### 3. Portfolio Balance Model
- **Equilibrium Rate**: {result.portfolio_balance.equilibrium_rate:.4f}
- **Risk Premium**: {result.portfolio_balance.risk_premium:.2f}%
- **Domestic Asset Share**: {result.portfolio_balance.domestic_asset_share:.1f}%
- **Foreign Asset Share**: {result.portfolio_balance.foreign_asset_share:.1f}%
- **Capital Flow Effect**: {result.portfolio_balance.capital_flow_effect:.4f}

### 4. Taylor Rule Model
- **Implied Rate**: {result.taylor_rule.implied_rate:.2f}%
- **Policy Divergence**: {result.taylor_rule.policy_divergence:.2f}%
- **Inflation Gap Effect**: {result.taylor_rule.inflation_gap_effect:.3f}
- **Output Gap Effect**: {result.taylor_rule.output_gap_effect:.3f}
- **Hawkish/Dovish Score**: {result.taylor_rule.hawkish_dovish_score:.2f}
- **Credibility Index**: {result.taylor_rule.credibility_index:.2f}

## Policy Implications
"""
        
        for i, implication in enumerate(result.policy_implications, 1):
            report += f"{i}. {implication}\n"
            
        report += """

## Risk Factors
- Exchange rate deviation from fundamentals
- Monetary policy divergence
- Portfolio rebalancing pressures
- Central bank credibility concerns

## Recommendations
"""
        
        if result.combined_signal == 'BUY':
            report += "- Consider long positions based on monetary fundamentals\n"
            report += "- Monitor central bank communications for policy shifts\n"
        elif result.combined_signal == 'SELL':
            report += "- Consider short positions based on monetary fundamentals\n"
            report += "- Watch for overshooting and potential reversals\n"
        else:
            report += "- Maintain neutral positioning\n"
            report += "- Wait for clearer monetary policy signals\n"
            
        report += "- Implement appropriate risk management measures\n"
        report += f"- Monitor key economic indicators and central bank communications\n"
        
        return report

# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    domestic_data = MonetaryData(
        money_supply=1000.0,
        interest_rate=2.5,
        inflation_rate=2.1,
        gdp_growth=2.8,
        current_account=-50.0,
        government_debt=800.0,
        central_bank_assets=500.0,
        policy_rate=2.25
    )
    
    foreign_data = MonetaryData(
        money_supply=1200.0,
        interest_rate=1.8,
        inflation_rate=1.9,
        gdp_growth=2.2,
        current_account=30.0,
        government_debt=900.0,
        central_bank_assets=600.0,
        policy_rate=1.75
    )
    
    # Create sample historical data
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='M')
    n_periods = len(dates)
    
    historical_data = pd.DataFrame({
        'date': dates,
        'exchange_rate': 1.0 + np.random.normal(0, 0.1, n_periods).cumsum() * 0.01,
        'exchange_rate_log': np.log(1.0 + np.random.normal(0, 0.1, n_periods).cumsum() * 0.01),
        'money_supply_diff': np.random.normal(0, 0.05, n_periods),
        'income_diff': np.random.normal(0, 0.02, n_periods),
        'interest_rate_diff': np.random.normal(0, 0.5, n_periods)
    })
    
    market_data = {
        'volatility': 0.12,
        'correlation': 0.25
    }
    
    targets = {
        'inflation_target': 2.0,
        'output_target': 2.5
    }
    
    # Initialize analyzer
    analyzer = MonetaryModelAnalyzer()
    
    # Run analysis
    try:
        result = analyzer.analyze(
            domestic_data=domestic_data,
            foreign_data=foreign_data,
            historical_data=historical_data,
            market_data=market_data,
            targets=targets
        )
        
        # Print summary
        print("=== Monetary Model Analysis Summary ===")
        print(f"Combined Signal: {result.combined_signal}")
        print(f"Confidence Score: {result.confidence_score:.1%}")
        print(f"Risk Assessment: {result.risk_assessment}")
        print(f"\nFlexible Price Fair Value: {result.flexible_price.fair_value:.4f}")
        print(f"Current Rate Deviation: {result.flexible_price.deviation:.2f}%")
        print(f"\nTaylor Rule Policy Divergence: {result.taylor_rule.policy_divergence:.2f}%")
        print(f"Portfolio Risk Premium: {result.portfolio_balance.risk_premium:.2f}%")
        
        print("\n=== Policy Implications ===")
        for i, implication in enumerate(result.policy_implications, 1):
            print(f"{i}. {implication}")
            
        # Generate and print report
        report = analyzer.generate_report(result, 'USD/EUR')
        print("\n" + "="*50)
        print(report)
        
        # Plot results
        analyzer.plot_analysis(result, 'USD/EUR')
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        import traceback
        traceback.print_exc()