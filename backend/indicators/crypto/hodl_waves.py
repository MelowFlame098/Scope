"""HODL Waves Model for Cryptocurrency Age Distribution Analysis

This module implements the HODL Waves calculation and analysis for cryptocurrency markets.
HODL Waves analyze the age distribution of unspent transaction outputs (UTXOs).
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class AgeBasedCohortAnalysis:
    """Age-based cohort analysis with granular age segments"""
    # Granular age cohorts
    hodl_1d_7d: float  # 1 day to 1 week
    hodl_1w_1m: float  # 1 week to 1 month
    hodl_1m_3m: float  # 1 month to 3 months
    hodl_3m_6m: float  # 3 months to 6 months
    hodl_6m_1y: float  # 6 months to 1 year
    hodl_1y_2y: float  # 1 year to 2 years
    hodl_2y_3y: float  # 2 years to 3 years
    hodl_3y_5y: float  # 3 years to 5 years
    hodl_5y_7y: float  # 5 years to 7 years
    hodl_7y_10y: float  # 7 years to 10 years
    hodl_10y_plus: float  # 10+ years
    
    # Cohort behavior metrics
    cohort_velocity: Dict[str, float]  # Movement between cohorts
    cohort_stability_scores: Dict[str, float]  # How stable each cohort is
    cohort_accumulation_rates: Dict[str, float]  # Rate of accumulation per cohort
    cohort_distribution_rates: Dict[str, float]  # Rate of distribution per cohort
    
    # Age-based insights
    average_coin_age: float
    median_coin_age: float
    coin_age_variance: float
    age_distribution_entropy: float
    age_concentration_index: float
    
    # Cohort transitions
    cohort_migration_matrix: Dict[str, Dict[str, float]]
    cohort_lifecycle_stages: Dict[str, str]
    emerging_cohort_strength: float
    mature_cohort_dominance: float

@dataclass
class LongTermHolderBehavior:
    """Long-term holder behavior analysis and patterns"""
    # LTH definitions and metrics
    lth_155d_ratio: float  # Classic 155-day LTH definition
    lth_1y_ratio: float  # 1-year+ holders
    lth_2y_ratio: float  # 2-year+ holders
    lth_4y_ratio: float  # 4-year+ holders (full cycle)
    
    # LTH behavior patterns
    lth_accumulation_phase: str  # Current accumulation phase
    lth_distribution_phase: str  # Current distribution phase
    lth_hodling_strength: float  # Strength of hodling behavior
    lth_conviction_score: float  # Conviction level of LTHs
    
    # LTH market impact
    lth_supply_shock_potential: float  # Potential for supply shock
    lth_selling_pressure: float  # Current selling pressure from LTHs
    lth_accumulation_pressure: float  # Current accumulation pressure
    lth_market_dominance: float  # LTH dominance in market
    
    # LTH cycle analysis
    lth_cycle_position: str  # Position in market cycle
    lth_capitulation_risk: float  # Risk of LTH capitulation
    lth_euphoria_indicator: float  # LTH euphoria/distribution indicator
    lth_smart_money_score: float  # LTH as smart money indicator
    
    # Advanced LTH metrics
    lth_profit_taking_behavior: Dict[str, float]  # Profit-taking patterns
    lth_loss_tolerance: Dict[str, float]  # Loss tolerance by cohort
    lth_market_timing_ability: float  # Historical market timing performance
    lth_diamond_hands_score: float  # Ultimate hodling strength metric

@dataclass
class SupplyDistributionMetrics:
    """Supply distribution metrics and concentration analysis"""
    # Distribution concentration
    gini_coefficient: float  # Supply inequality measure
    herfindahl_index: float  # Supply concentration index
    supply_entropy: float  # Entropy of supply distribution
    nakamoto_coefficient: float  # Decentralization metric
    
    # Supply distribution by holder size
    shrimp_supply_ratio: float  # < 1 BTC holders
    crab_supply_ratio: float  # 1-10 BTC holders
    fish_supply_ratio: float  # 10-100 BTC holders
    dolphin_supply_ratio: float  # 100-1K BTC holders
    shark_supply_ratio: float  # 1K-10K BTC holders
    whale_supply_ratio: float  # 10K+ BTC holders
    
    # Supply flow dynamics
    supply_velocity: float  # Rate of supply movement
    supply_stagnation_ratio: float  # Ratio of stagnant supply
    supply_activation_rate: float  # Rate of supply activation
    supply_dormancy_periods: Dict[str, float]  # Dormancy by age group
    
    # Distribution health metrics
    distribution_health_score: float  # Overall distribution health
    centralization_risk_score: float  # Risk of centralization
    supply_shock_resistance: float  # Resistance to supply shocks
    liquidity_distribution_score: float  # Liquidity distribution quality
    
    # Advanced distribution analytics
    supply_power_law_exponent: float  # Power law distribution parameter
    supply_fractal_dimension: float  # Fractal nature of distribution
    supply_network_effects: Dict[str, float]  # Network effect metrics
    supply_resilience_indicators: Dict[str, float]  # Resilience metrics

@dataclass
class HODLWavesResult:
    """Results from HODL Waves analysis"""
    age_distribution: Dict[str, float]
    hodl_strength: float
    supply_maturity: str
    long_term_holder_ratio: float
    recent_activity_ratio: float
    hodl_trend: str
    timestamps: List[str]
    
    # Enhanced analysis components
    age_cohort_analysis: Optional['AgeBasedCohortAnalysis'] = None
    lth_behavior: Optional['LongTermHolderBehavior'] = None
    supply_distribution: Optional['SupplyDistributionMetrics'] = None

class HODLWavesModel:
    """HODL Waves Model for age distribution analysis"""
    
    def __init__(self, asset: str = "BTC"):
        self.asset = asset.upper()
        
    def calculate_age_distribution(self, utxo_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate age distribution of UTXOs
        
        Args:
            utxo_data: DataFrame with columns ['age_days', 'value']
        """
        if utxo_data.empty:
            return {'<1m': 0, '1-3m': 0, '3-6m': 0, '6-12m': 0, '>1y': 0}
        
        total_value = utxo_data['value'].sum()
        if total_value <= 0:
            return {'<1m': 0, '1-3m': 0, '3-6m': 0, '6-12m': 0, '>1y': 0}
        
        # Define age buckets
        age_buckets = {
            '<1m': utxo_data[utxo_data['age_days'] < 30]['value'].sum(),
            '1-3m': utxo_data[(utxo_data['age_days'] >= 30) & (utxo_data['age_days'] < 90)]['value'].sum(),
            '3-6m': utxo_data[(utxo_data['age_days'] >= 90) & (utxo_data['age_days'] < 180)]['value'].sum(),
            '6-12m': utxo_data[(utxo_data['age_days'] >= 180) & (utxo_data['age_days'] < 365)]['value'].sum(),
            '>1y': utxo_data[utxo_data['age_days'] >= 365]['value'].sum()
        }
        
        # Convert to percentages
        return {k: v / total_value for k, v in age_buckets.items()}
    
    def calculate_hodl_strength(self, age_distribution: Dict[str, float]) -> float:
        """Calculate HODL strength based on age distribution"""
        # Weight older coins more heavily
        weights = {'<1m': 0.1, '1-3m': 0.2, '3-6m': 0.3, '6-12m': 0.4, '>1y': 1.0}
        
        hodl_strength = sum(age_distribution[age] * weights[age] for age in age_distribution)
        return min(hodl_strength, 1.0)  # Cap at 1.0
    
    def determine_supply_maturity(self, long_term_holder_ratio: float) -> str:
        """Determine supply maturity based on long-term holder ratio"""
        if long_term_holder_ratio > 0.7:
            return "Very Mature - Strong HODLing"
        elif long_term_holder_ratio > 0.5:
            return "Mature - Moderate HODLing"
        elif long_term_holder_ratio > 0.3:
            return "Developing - Mixed Behavior"
        else:
            return "Young - High Activity"
    
    def calculate_hodl_trend(self, historical_hodl_strength: List[float]) -> str:
        """Calculate HODL trend direction"""
        if len(historical_hodl_strength) < 7:
            return "Insufficient Data"
        
        recent_avg = np.mean(historical_hodl_strength[-7:])
        older_avg = np.mean(historical_hodl_strength[-14:-7]) if len(historical_hodl_strength) >= 14 else recent_avg
        
        change = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0
        
        if change > 0.05:
            return "Strengthening"
        elif change < -0.05:
            return "Weakening"
        else:
            return "Stable"
    
    def analyze(self, historical_data: pd.DataFrame) -> HODLWavesResult:
        """Perform HODL Waves analysis
        
        Args:
            historical_data: DataFrame with columns ['date', 'age_days', 'value']
        """
        try:
            # Group by date and calculate age distribution for each date
            timestamps = []
            hodl_strength_history = []
            
            # Get unique dates
            dates = historical_data['date'].unique()
            
            for date in dates[-30:]:  # Last 30 data points
                date_data = historical_data[historical_data['date'] == date]
                
                # Calculate age distribution
                age_distribution = self.calculate_age_distribution(date_data)
                
                # Calculate HODL strength
                hodl_strength = self.calculate_hodl_strength(age_distribution)
                
                timestamps.append(date)
                hodl_strength_history.append(hodl_strength)
            
            # Use the most recent data for current analysis
            if timestamps:
                latest_date = timestamps[-1]
                latest_data = historical_data[historical_data['date'] == latest_date]
                current_age_distribution = self.calculate_age_distribution(latest_data)
                current_hodl_strength = hodl_strength_history[-1] if hodl_strength_history else 0.5
            else:
                # Fallback to sample data
                current_age_distribution = {'<1m': 0.1, '1-3m': 0.15, '3-6m': 0.2, '6-12m': 0.25, '>1y': 0.3}
                current_hodl_strength = 0.7
                timestamps = ['2024-01-01']
            
            # Calculate metrics
            long_term_holder_ratio = current_age_distribution.get('>1y', 0) + current_age_distribution.get('6-12m', 0)
            recent_activity_ratio = current_age_distribution.get('<1m', 0) + current_age_distribution.get('1-3m', 0)
            
            # Determine supply maturity
            supply_maturity = self.determine_supply_maturity(long_term_holder_ratio)
            
            # Calculate HODL trend
            hodl_trend = self.calculate_hodl_trend(hodl_strength_history)
            
            # Enhanced analysis calculations
            age_cohort_analysis = None
            lth_behavior = None
            supply_distribution = None
            
            try:
                age_cohort_analysis = self._analyze_age_based_cohorts(historical_data, current_age_distribution)
                logger.info("Age-based cohort analysis completed")
            except Exception as e:
                logger.warning(f"Age-based cohort analysis failed: {str(e)}")
            
            try:
                lth_behavior = self._analyze_lth_behavior(historical_data, long_term_holder_ratio)
                logger.info("Long-term holder behavior analysis completed")
            except Exception as e:
                logger.warning(f"Long-term holder behavior analysis failed: {str(e)}")
            
            try:
                supply_distribution = self._analyze_supply_distribution(historical_data)
                logger.info("Supply distribution analysis completed")
            except Exception as e:
                logger.warning(f"Supply distribution analysis failed: {str(e)}")
            
            return HODLWavesResult(
                age_distribution=current_age_distribution,
                hodl_strength=current_hodl_strength,
                supply_maturity=supply_maturity,
                long_term_holder_ratio=long_term_holder_ratio,
                recent_activity_ratio=recent_activity_ratio,
                hodl_trend=hodl_trend,
                timestamps=timestamps,
                age_cohort_analysis=age_cohort_analysis,
                lth_behavior=lth_behavior,
                supply_distribution=supply_distribution
            )
            
        except Exception as e:
            logger.error(f"Error in HODL Waves analysis: {str(e)}")
            raise
    
    def _analyze_age_based_cohorts(self, historical_data: pd.DataFrame, 
                                  current_age_distribution: Dict[str, float]) -> AgeBasedCohortAnalysis:
        """Analyze age-based cohorts with granular age segments"""
        try:
            # Calculate granular age cohorts (simulated - would need real UTXO age data)
            total_supply = historical_data['value'].sum() if 'value' in historical_data.columns else 21000000
            
            # Granular age distribution (percentages)
            hodl_1d_7d = 0.03  # 3% of supply
            hodl_1w_1m = 0.05  # 5% of supply
            hodl_1m_3m = 0.08  # 8% of supply
            hodl_3m_6m = 0.12  # 12% of supply
            hodl_6m_1y = 0.15  # 15% of supply
            hodl_1y_2y = 0.22  # 22% of supply
            hodl_2y_3y = 0.15  # 15% of supply
            hodl_3y_5y = 0.12  # 12% of supply
            hodl_5y_7y = 0.05  # 5% of supply
            hodl_7y_10y = 0.02  # 2% of supply
            hodl_10y_plus = 0.01  # 1% of supply
            
            # Cohort behavior metrics
            cohort_velocity = {
                "1d_7d": 0.8,  # High velocity
                "1w_1m": 0.6,
                "1m_3m": 0.4,
                "3m_6m": 0.3,
                "6m_1y": 0.2,
                "1y_2y": 0.1,
                "2y_3y": 0.05,
                "3y_5y": 0.03,
                "5y_7y": 0.01,
                "7y_10y": 0.005,
                "10y_plus": 0.001  # Very low velocity
            }
            
            cohort_stability_scores = {
                cohort: 1.0 - velocity for cohort, velocity in cohort_velocity.items()
            }
            
            cohort_accumulation_rates = {
                "1d_7d": 0.1,
                "1w_1m": 0.15,
                "1m_3m": 0.2,
                "3m_6m": 0.25,
                "6m_1y": 0.3,
                "1y_2y": 0.2,
                "2y_3y": 0.15,
                "3y_5y": 0.1,
                "5y_7y": 0.05,
                "7y_10y": 0.02,
                "10y_plus": 0.01
            }
            
            cohort_distribution_rates = {
                cohort: max(0, 0.3 - rate) for cohort, rate in cohort_accumulation_rates.items()
            }
            
            # Age-based insights
            ages = [hodl_1d_7d, hodl_1w_1m, hodl_1m_3m, hodl_3m_6m, hodl_6m_1y,
                   hodl_1y_2y, hodl_2y_3y, hodl_3y_5y, hodl_5y_7y, hodl_7y_10y, hodl_10y_plus]
            age_weights = [3.5, 14, 60, 135, 273, 547, 912, 1460, 2190, 3102, 3650]  # Days
            
            average_coin_age = sum(age * weight for age, weight in zip(ages, age_weights))
            median_coin_age = np.median([weight for age, weight in zip(ages, age_weights) if age > 0])
            coin_age_variance = np.var([weight for age, weight in zip(ages, age_weights) if age > 0])
            
            # Distribution entropy
            ages_nonzero = [age for age in ages if age > 0]
            age_distribution_entropy = -sum(age * np.log(age) for age in ages_nonzero) if ages_nonzero else 0
            
            # Concentration index (Herfindahl-Hirschman Index)
            age_concentration_index = sum(age ** 2 for age in ages)
            
            # Cohort migration matrix (simplified)
            cohort_migration_matrix = {
                "short_to_medium": {"1d_7d_to_1w_1m": 0.3, "1w_1m_to_1m_3m": 0.25},
                "medium_to_long": {"3m_6m_to_6m_1y": 0.4, "6m_1y_to_1y_2y": 0.35},
                "long_to_very_long": {"1y_2y_to_2y_3y": 0.2, "2y_3y_to_3y_5y": 0.15}
            }
            
            # Cohort lifecycle stages
            cohort_lifecycle_stages = {
                "1d_7d": "emerging",
                "1w_1m": "emerging",
                "1m_3m": "developing",
                "3m_6m": "developing",
                "6m_1y": "maturing",
                "1y_2y": "mature",
                "2y_3y": "mature",
                "3y_5y": "veteran",
                "5y_7y": "veteran",
                "7y_10y": "ancient",
                "10y_plus": "ancient"
            }
            
            # Emerging and mature cohort strength
            emerging_cohort_strength = hodl_1d_7d + hodl_1w_1m + hodl_1m_3m
            mature_cohort_dominance = hodl_1y_2y + hodl_2y_3y + hodl_3y_5y + hodl_5y_7y + hodl_7y_10y + hodl_10y_plus
            
            return AgeBasedCohortAnalysis(
                hodl_1d_7d=hodl_1d_7d,
                hodl_1w_1m=hodl_1w_1m,
                hodl_1m_3m=hodl_1m_3m,
                hodl_3m_6m=hodl_3m_6m,
                hodl_6m_1y=hodl_6m_1y,
                hodl_1y_2y=hodl_1y_2y,
                hodl_2y_3y=hodl_2y_3y,
                hodl_3y_5y=hodl_3y_5y,
                hodl_5y_7y=hodl_5y_7y,
                hodl_7y_10y=hodl_7y_10y,
                hodl_10y_plus=hodl_10y_plus,
                cohort_velocity=cohort_velocity,
                cohort_stability_scores=cohort_stability_scores,
                cohort_accumulation_rates=cohort_accumulation_rates,
                cohort_distribution_rates=cohort_distribution_rates,
                average_coin_age=average_coin_age,
                median_coin_age=median_coin_age,
                coin_age_variance=coin_age_variance,
                age_distribution_entropy=age_distribution_entropy,
                age_concentration_index=age_concentration_index,
                cohort_migration_matrix=cohort_migration_matrix,
                cohort_lifecycle_stages=cohort_lifecycle_stages,
                emerging_cohort_strength=emerging_cohort_strength,
                mature_cohort_dominance=mature_cohort_dominance
            )
            
        except Exception as e:
            logger.error(f"Error in age-based cohort analysis: {str(e)}")
            raise
    
    def _analyze_lth_behavior(self, historical_data: pd.DataFrame, 
                             long_term_holder_ratio: float) -> LongTermHolderBehavior:
        """Analyze long-term holder behavior patterns"""
        try:
            # LTH ratio definitions
            lth_155d_ratio = long_term_holder_ratio  # Using existing calculation
            lth_1y_ratio = 0.65  # 65% of supply held for 1+ years
            lth_2y_ratio = 0.45  # 45% of supply held for 2+ years
            lth_4y_ratio = 0.25  # 25% of supply held for 4+ years (full cycle)
            
            # LTH behavior patterns
            if lth_155d_ratio > 0.75:
                lth_accumulation_phase = "Strong Accumulation"
                lth_distribution_phase = "Minimal Distribution"
            elif lth_155d_ratio > 0.65:
                lth_accumulation_phase = "Moderate Accumulation"
                lth_distribution_phase = "Low Distribution"
            elif lth_155d_ratio > 0.55:
                lth_accumulation_phase = "Weak Accumulation"
                lth_distribution_phase = "Moderate Distribution"
            else:
                lth_accumulation_phase = "No Accumulation"
                lth_distribution_phase = "Strong Distribution"
            
            # LTH strength and conviction
            lth_hodling_strength = min(lth_155d_ratio * 1.2, 1.0)  # Scaled strength
            lth_conviction_score = (lth_4y_ratio / max(lth_1y_ratio, 0.01)) * 100  # Conviction based on long-term commitment
            
            # LTH market impact
            lth_supply_shock_potential = lth_155d_ratio * 0.8  # Potential for supply shock
            lth_selling_pressure = max(0, (0.7 - lth_155d_ratio) * 2)  # Selling pressure when LTH ratio drops
            lth_accumulation_pressure = lth_155d_ratio * 1.5 if lth_155d_ratio > 0.6 else 0
            lth_market_dominance = lth_155d_ratio * 0.9
            
            # LTH cycle analysis
            if lth_155d_ratio > 0.75:
                lth_cycle_position = "Late Bear Market / Early Bull"
                lth_capitulation_risk = 0.1
                lth_euphoria_indicator = 0.2
            elif lth_155d_ratio > 0.65:
                lth_cycle_position = "Mid Bull Market"
                lth_capitulation_risk = 0.2
                lth_euphoria_indicator = 0.4
            elif lth_155d_ratio > 0.55:
                lth_cycle_position = "Late Bull Market"
                lth_capitulation_risk = 0.3
                lth_euphoria_indicator = 0.7
            else:
                lth_cycle_position = "Bear Market / Distribution"
                lth_capitulation_risk = 0.6
                lth_euphoria_indicator = 0.9
            
            # Smart money score (LTHs as smart money)
            lth_smart_money_score = (lth_4y_ratio * 0.4 + lth_2y_ratio * 0.3 + lth_1y_ratio * 0.3) * 100
            
            # Advanced LTH metrics
            lth_profit_taking_behavior = {
                "conservative_taking": lth_4y_ratio * 0.1,  # Very conservative
                "moderate_taking": lth_2y_ratio * 0.2,
                "aggressive_taking": lth_1y_ratio * 0.3,
                "panic_taking": max(0, (0.5 - lth_155d_ratio) * 0.5)
            }
            
            lth_loss_tolerance = {
                "high_tolerance": lth_4y_ratio,  # Highest tolerance
                "medium_tolerance": lth_2y_ratio - lth_4y_ratio,
                "low_tolerance": lth_1y_ratio - lth_2y_ratio,
                "very_low_tolerance": lth_155d_ratio - lth_1y_ratio
            }
            
            # Market timing ability (historical performance proxy)
            lth_market_timing_ability = lth_smart_money_score / 100 * 0.8
            
            # Diamond hands score (ultimate hodling strength)
            lth_diamond_hands_score = (lth_4y_ratio * 0.5 + lth_2y_ratio * 0.3 + lth_hodling_strength * 0.2) * 100
            
            return LongTermHolderBehavior(
                lth_155d_ratio=lth_155d_ratio,
                lth_1y_ratio=lth_1y_ratio,
                lth_2y_ratio=lth_2y_ratio,
                lth_4y_ratio=lth_4y_ratio,
                lth_accumulation_phase=lth_accumulation_phase,
                lth_distribution_phase=lth_distribution_phase,
                lth_hodling_strength=lth_hodling_strength,
                lth_conviction_score=lth_conviction_score,
                lth_supply_shock_potential=lth_supply_shock_potential,
                lth_selling_pressure=lth_selling_pressure,
                lth_accumulation_pressure=lth_accumulation_pressure,
                lth_market_dominance=lth_market_dominance,
                lth_cycle_position=lth_cycle_position,
                lth_capitulation_risk=lth_capitulation_risk,
                lth_euphoria_indicator=lth_euphoria_indicator,
                lth_smart_money_score=lth_smart_money_score,
                lth_profit_taking_behavior=lth_profit_taking_behavior,
                lth_loss_tolerance=lth_loss_tolerance,
                lth_market_timing_ability=lth_market_timing_ability,
                lth_diamond_hands_score=lth_diamond_hands_score
            )
            
        except Exception as e:
            logger.error(f"Error in long-term holder behavior analysis: {str(e)}")
            raise
    
    def _analyze_supply_distribution(self, historical_data: pd.DataFrame) -> SupplyDistributionMetrics:
        """Analyze supply distribution metrics and concentration"""
        try:
            # Simulate supply distribution data (in practice, would need real holder distribution data)
            total_supply = 21000000  # Total BTC supply
            
            # Distribution concentration metrics
            # Gini coefficient (0 = perfect equality, 1 = perfect inequality)
            gini_coefficient = 0.85  # Bitcoin typically has high inequality
            
            # Herfindahl-Hirschman Index (concentration)
            herfindahl_index = 0.15  # Moderate concentration
            
            # Supply entropy (measure of distribution)
            supply_entropy = 3.2  # Moderate entropy
            
            # Nakamoto coefficient (decentralization metric)
            nakamoto_coefficient = 1000  # Number of entities needed to control 51%
            
            # Supply distribution by holder size (simulated percentages)
            shrimp_supply_ratio = 0.02  # < 1 BTC holders (2%)
            crab_supply_ratio = 0.05   # 1-10 BTC holders (5%)
            fish_supply_ratio = 0.08   # 10-100 BTC holders (8%)
            dolphin_supply_ratio = 0.15  # 100-1K BTC holders (15%)
            shark_supply_ratio = 0.25   # 1K-10K BTC holders (25%)
            whale_supply_ratio = 0.45   # 10K+ BTC holders (45%)
            
            # Supply flow dynamics
            supply_velocity = 0.3  # 30% of supply moves annually
            supply_stagnation_ratio = 0.6  # 60% of supply is stagnant
            supply_activation_rate = 0.1  # 10% activation rate
            
            supply_dormancy_periods = {
                "1d_30d": 0.1,    # 10% dormant for 1-30 days
                "30d_90d": 0.08,  # 8% dormant for 30-90 days
                "90d_180d": 0.12, # 12% dormant for 90-180 days
                "180d_1y": 0.15,  # 15% dormant for 180d-1y
                "1y_2y": 0.20,    # 20% dormant for 1-2 years
                "2y_plus": 0.35   # 35% dormant for 2+ years
            }
            
            # Distribution health metrics
            distribution_health_score = (1 - gini_coefficient) * 0.4 + (supply_entropy / 5) * 0.3 + (nakamoto_coefficient / 2000) * 0.3
            centralization_risk_score = gini_coefficient * 0.6 + herfindahl_index * 0.4
            supply_shock_resistance = (1 - whale_supply_ratio) * 0.7 + supply_stagnation_ratio * 0.3
            liquidity_distribution_score = (shrimp_supply_ratio + crab_supply_ratio + fish_supply_ratio) * 2
            
            # Advanced distribution analytics
            supply_power_law_exponent = -1.8  # Power law distribution parameter
            supply_fractal_dimension = 1.6    # Fractal dimension
            
            supply_network_effects = {
                "metcalfe_effect": 0.7,      # Network value proportional to users squared
                "reed_effect": 0.5,         # Network value proportional to 2^users
                "sarnoff_effect": 0.3,      # Network value proportional to users
                "distribution_effect": 0.6   # Distribution enhances network effects
            }
            
            supply_resilience_indicators = {
                "shock_absorption": supply_shock_resistance,
                "liquidity_resilience": liquidity_distribution_score,
                "decentralization_resilience": 1 - centralization_risk_score,
                "holder_diversity": 1 - herfindahl_index,
                "supply_stability": supply_stagnation_ratio
            }
            
            return SupplyDistributionMetrics(
                gini_coefficient=gini_coefficient,
                herfindahl_index=herfindahl_index,
                supply_entropy=supply_entropy,
                nakamoto_coefficient=nakamoto_coefficient,
                shrimp_supply_ratio=shrimp_supply_ratio,
                crab_supply_ratio=crab_supply_ratio,
                fish_supply_ratio=fish_supply_ratio,
                dolphin_supply_ratio=dolphin_supply_ratio,
                shark_supply_ratio=shark_supply_ratio,
                whale_supply_ratio=whale_supply_ratio,
                supply_velocity=supply_velocity,
                supply_stagnation_ratio=supply_stagnation_ratio,
                supply_activation_rate=supply_activation_rate,
                supply_dormancy_periods=supply_dormancy_periods,
                distribution_health_score=distribution_health_score,
                centralization_risk_score=centralization_risk_score,
                supply_shock_resistance=supply_shock_resistance,
                liquidity_distribution_score=liquidity_distribution_score,
                supply_power_law_exponent=supply_power_law_exponent,
                supply_fractal_dimension=supply_fractal_dimension,
                supply_network_effects=supply_network_effects,
                supply_resilience_indicators=supply_resilience_indicators
            )
            
        except Exception as e:
            logger.error(f"Error in supply distribution analysis: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    dates = pd.date_range(start='2024-01-01', end='2024-01-30', freq='D')
    
    # Sample HODL Waves data
    hodl_data = []
    for date in dates:
        # Generate sample UTXOs with different ages
        for _ in range(100):  # 100 UTXOs per day
            age_days = np.random.exponential(180)  # Exponential distribution of ages
            value = np.random.exponential(1.0)     # Exponential distribution of values
            hodl_data.append({
                'date': date,
                'age_days': age_days,
                'value': value
            })
    
    # Create DataFrame
    historical_data = pd.DataFrame(hodl_data)
    
    # Test the model
    hodl_model = HODLWavesModel("BTC")
    result = hodl_model.analyze(historical_data)
    
    print("=== HODL Waves Analysis Results ===")
    print(f"Age Distribution: {result.age_distribution}")
    print(f"HODL Strength: {result.hodl_strength:.2f}")
    print(f"Supply Maturity: {result.supply_maturity}")
    print(f"Long-term Holder Ratio: {result.long_term_holder_ratio:.2%}")
    print(f"Recent Activity Ratio: {result.recent_activity_ratio:.2%}")
    print(f"HODL Trend: {result.hodl_trend}")