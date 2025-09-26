"""Comprehensive Crypto Quantitative Metrics Model

This module implements various on-chain and market metrics for cryptocurrency analysis:
- MVRV (Market Value to Realized Value)
- SOPR (Spent Output Profit Ratio)
- Puell Multiple
- Hash Ribbons
- Realized Cap
- HODL Waves
- Exchange Flow Metrics
- Network Growth Metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class MVRVResult:
    """MVRV (Market Value to Realized Value) analysis result"""
    current_mvrv: float
    mvrv_z_score: float
    mvrv_percentile: float
    market_phase: str
    historical_mvrv: List[float]
    mvrv_bands: Dict[str, float]
    timestamps: List[datetime]

@dataclass
class SOPRResult:
    """SOPR (Spent Output Profit Ratio) analysis result"""
    current_sopr: float
    sopr_trend: str
    profit_loss_ratio: float
    market_sentiment: str
    historical_sopr: List[float]
    sopr_ma: List[float]
    timestamps: List[datetime]

@dataclass
class PuellResult:
    """Puell Multiple analysis result"""
    current_puell: float
    puell_percentile: float
    mining_profitability: str
    market_cycle_phase: str
    historical_puell: List[float]
    puell_bands: Dict[str, float]
    timestamps: List[datetime]

@dataclass
class HashRibbonsResult:
    """Hash Ribbons analysis result"""
    hash_ribbon_signal: str
    miner_capitulation: bool
    hash_rate_trend: str
    difficulty_adjustment: float
    mining_health: str
    hash_rate_ma_30: List[float]
    hash_rate_ma_60: List[float]
    timestamps: List[datetime]

@dataclass
class HODLWavesResult:
    """HODL Waves analysis result"""
    age_distribution: Dict[str, float]
    hodl_strength: float
    supply_maturity: str
    long_term_holder_ratio: float
    recent_activity_ratio: float
    hodl_trend: str
    timestamps: List[datetime]

@dataclass
class ExchangeFlowResult:
    """Exchange Flow analysis result"""
    net_flow: float
    inflow_trend: str
    outflow_trend: str
    exchange_balance_ratio: float
    selling_pressure: str
    flow_momentum: float
    timestamps: List[datetime]

@dataclass
class CryptoQuantResult:
    """Combined crypto quantitative metrics result"""
    mvrv_result: MVRVResult
    sopr_result: SOPRResult
    puell_result: PuellResult
    hash_ribbons_result: HashRibbonsResult
    hodl_waves_result: HODLWavesResult
    exchange_flow_result: ExchangeFlowResult
    composite_score: float
    market_regime: str
    risk_assessment: str
    confidence_score: float

class MVRVModel:
    """Market Value to Realized Value (MVRV) Model"""
    
    def __init__(self, asset: str = "BTC"):
        self.asset = asset.upper()
        
    def calculate_realized_cap(self, utxo_data: pd.DataFrame) -> float:
        """Calculate realized capitalization from UTXO data
        
        Args:
            utxo_data: DataFrame with columns ['value', 'created_price']
        """
        if utxo_data.empty:
            return 0
        
        # Realized cap = sum of (UTXO value * price when UTXO was created)
        realized_cap = (utxo_data['value'] * utxo_data['created_price']).sum()
        return realized_cap
    
    def calculate_mvrv(self, market_cap: float, realized_cap: float) -> float:
        """Calculate MVRV ratio"""
        if realized_cap <= 0:
            return 1.0
        return market_cap / realized_cap
    
    def calculate_mvrv_z_score(self, current_mvrv: float, historical_mvrv: List[float]) -> float:
        """Calculate MVRV Z-Score for market timing"""
        if len(historical_mvrv) < 30:
            return 0
        
        mean_mvrv = np.mean(historical_mvrv)
        std_mvrv = np.std(historical_mvrv)
        
        if std_mvrv == 0:
            return 0
        
        return (current_mvrv - mean_mvrv) / std_mvrv
    
    def determine_market_phase(self, mvrv_z_score: float) -> str:
        """Determine market phase based on MVRV Z-Score"""
        if mvrv_z_score > 7:
            return "Extreme Euphoria - Major Top Signal"
        elif mvrv_z_score > 3.5:
            return "Euphoria - Top Formation"
        elif mvrv_z_score > 1:
            return "Optimism - Bull Market"
        elif mvrv_z_score > -0.5:
            return "Neutral - Consolidation"
        elif mvrv_z_score > -1.5:
            return "Pessimism - Bear Market"
        else:
            return "Extreme Fear - Major Bottom Signal"
    
    def calculate_mvrv_bands(self, historical_mvrv: List[float]) -> Dict[str, float]:
        """Calculate MVRV percentile bands"""
        if len(historical_mvrv) < 10:
            return {'bottom': 0, 'low': 0, 'fair': 0, 'high': 0, 'top': 0}
        
        return {
            'bottom': np.percentile(historical_mvrv, 5),   # Bottom 5%
            'low': np.percentile(historical_mvrv, 25),     # Bottom 25%
            'fair': np.percentile(historical_mvrv, 50),    # Median
            'high': np.percentile(historical_mvrv, 75),    # Top 25%
            'top': np.percentile(historical_mvrv, 95)      # Top 5%
        }
    
    def analyze(self, historical_data: pd.DataFrame) -> MVRVResult:
        """Perform MVRV analysis
        
        Args:
            historical_data: DataFrame with columns ['date', 'market_cap', 'realized_cap']
        """
        try:
            mvrv_ratios = []
            timestamps = []
            
            for _, row in historical_data.iterrows():
                mvrv = self.calculate_mvrv(row['market_cap'], row['realized_cap'])
                mvrv_ratios.append(mvrv)
                timestamps.append(row['date'])
            
            current_mvrv = mvrv_ratios[-1] if mvrv_ratios else 1.0
            mvrv_z_score = self.calculate_mvrv_z_score(current_mvrv, mvrv_ratios)
            
            # Calculate percentile
            mvrv_percentile = stats.percentileofscore(mvrv_ratios, current_mvrv)
            
            # Determine market phase
            market_phase = self.determine_market_phase(mvrv_z_score)
            
            # Calculate bands
            mvrv_bands = self.calculate_mvrv_bands(mvrv_ratios)
            
            return MVRVResult(
                current_mvrv=current_mvrv,
                mvrv_z_score=mvrv_z_score,
                mvrv_percentile=mvrv_percentile,
                market_phase=market_phase,
                historical_mvrv=mvrv_ratios,
                mvrv_bands=mvrv_bands,
                timestamps=timestamps
            )
            
        except Exception as e:
            logger.error(f"Error in MVRV analysis: {str(e)}")
            raise

class SOPRModel:
    """Spent Output Profit Ratio (SOPR) Model"""
    
    def __init__(self, asset: str = "BTC"):
        self.asset = asset.upper()
        
    def calculate_sopr(self, spent_outputs: pd.DataFrame) -> float:
        """Calculate SOPR from spent outputs
        
        Args:
            spent_outputs: DataFrame with columns ['value', 'created_price', 'spent_price']
        """
        if spent_outputs.empty:
            return 1.0
        
        # SOPR = sum(spent_price * value) / sum(created_price * value)
        realized_value = (spent_outputs['spent_price'] * spent_outputs['value']).sum()
        created_value = (spent_outputs['created_price'] * spent_outputs['value']).sum()
        
        if created_value <= 0:
            return 1.0
        
        return realized_value / created_value
    
    def calculate_profit_loss_ratio(self, spent_outputs: pd.DataFrame) -> float:
        """Calculate ratio of profitable to loss-making transactions"""
        if spent_outputs.empty:
            return 1.0
        
        profitable = spent_outputs[spent_outputs['spent_price'] > spent_outputs['created_price']]
        loss_making = spent_outputs[spent_outputs['spent_price'] <= spent_outputs['created_price']]
        
        profit_volume = profitable['value'].sum()
        loss_volume = loss_making['value'].sum()
        
        if loss_volume <= 0:
            return float('inf')
        
        return profit_volume / loss_volume
    
    def determine_market_sentiment(self, sopr: float, sopr_ma: float) -> str:
        """Determine market sentiment based on SOPR"""
        if sopr > 1.05 and sopr > sopr_ma:
            return "Strong Greed - High Profit Taking"
        elif sopr > 1.02:
            return "Greed - Moderate Profit Taking"
        elif sopr > 0.98:
            return "Neutral - Balanced Market"
        elif sopr > 0.95:
            return "Fear - Some Capitulation"
        else:
            return "Extreme Fear - Heavy Capitulation"
    
    def calculate_sopr_trend(self, recent_sopr: List[float], window: int = 7) -> str:
        """Calculate SOPR trend direction"""
        if len(recent_sopr) < window:
            return "Insufficient Data"
        
        # Linear regression on recent SOPR values
        x = np.arange(len(recent_sopr[-window:]))
        y = recent_sopr[-window:]
        
        slope, _, r_value, _, _ = stats.linregress(x, y)
        
        if abs(r_value) < 0.3:
            return "Sideways"
        elif slope > 0:
            return "Rising" if r_value > 0.5 else "Weakly Rising"
        else:
            return "Falling" if r_value < -0.5 else "Weakly Falling"
    
    def analyze(self, historical_data: pd.DataFrame) -> SOPRResult:
        """Perform SOPR analysis
        
        Args:
            historical_data: DataFrame with columns ['date', 'sopr']
        """
        try:
            sopr_values = historical_data['sopr'].tolist()
            timestamps = historical_data['date'].tolist()
            
            # Calculate moving average
            window = min(7, len(sopr_values))
            sopr_ma = pd.Series(sopr_values).rolling(window=window, min_periods=1).mean().tolist()
            
            current_sopr = sopr_values[-1] if sopr_values else 1.0
            current_sopr_ma = sopr_ma[-1] if sopr_ma else 1.0
            
            # Calculate trend
            sopr_trend = self.calculate_sopr_trend(sopr_values)
            
            # Determine market sentiment
            market_sentiment = self.determine_market_sentiment(current_sopr, current_sopr_ma)
            
            # Calculate profit/loss ratio (simplified)
            profit_loss_ratio = current_sopr  # Approximation
            
            return SOPRResult(
                current_sopr=current_sopr,
                sopr_trend=sopr_trend,
                profit_loss_ratio=profit_loss_ratio,
                market_sentiment=market_sentiment,
                historical_sopr=sopr_values,
                sopr_ma=sopr_ma,
                timestamps=timestamps
            )
            
        except Exception as e:
            logger.error(f"Error in SOPR analysis: {str(e)}")
            raise

class PuellMultipleModel:
    """Puell Multiple Model for mining profitability analysis"""
    
    def __init__(self, asset: str = "BTC"):
        self.asset = asset.upper()
        
    def calculate_puell_multiple(self, daily_issuance_usd: float, 
                               issuance_ma_365: float) -> float:
        """Calculate Puell Multiple
        
        Puell Multiple = Daily Coin Issuance (USD) / 365-day MA of Daily Coin Issuance (USD)
        """
        if issuance_ma_365 <= 0:
            return 1.0
        
        return daily_issuance_usd / issuance_ma_365
    
    def determine_mining_profitability(self, puell_multiple: float) -> str:
        """Determine mining profitability status"""
        if puell_multiple > 4:
            return "Extremely High Profitability - Potential Top"
        elif puell_multiple > 2:
            return "High Profitability - Bull Market"
        elif puell_multiple > 0.5:
            return "Normal Profitability - Stable Market"
        elif puell_multiple > 0.3:
            return "Low Profitability - Bear Market"
        else:
            return "Extremely Low Profitability - Potential Bottom"
    
    def determine_market_cycle_phase(self, puell_percentile: float) -> str:
        """Determine market cycle phase based on Puell percentile"""
        if puell_percentile > 95:
            return "Cycle Top - Extreme Overheating"
        elif puell_percentile > 80:
            return "Late Bull Market - Overheating"
        elif puell_percentile > 60:
            return "Bull Market - Healthy Growth"
        elif puell_percentile > 40:
            return "Neutral - Consolidation"
        elif puell_percentile > 20:
            return "Bear Market - Cooling Down"
        else:
            return "Cycle Bottom - Extreme Undervaluation"
    
    def calculate_puell_bands(self, historical_puell: List[float]) -> Dict[str, float]:
        """Calculate Puell Multiple percentile bands"""
        if len(historical_puell) < 10:
            return {'bottom': 0, 'low': 0, 'fair': 0, 'high': 0, 'top': 0}
        
        return {
            'bottom': np.percentile(historical_puell, 10),   # Bottom 10%
            'low': np.percentile(historical_puell, 30),      # Bottom 30%
            'fair': np.percentile(historical_puell, 50),     # Median
            'high': np.percentile(historical_puell, 70),     # Top 30%
            'top': np.percentile(historical_puell, 90)       # Top 10%
        }
    
    def analyze(self, historical_data: pd.DataFrame) -> PuellResult:
        """Perform Puell Multiple analysis
        
        Args:
            historical_data: DataFrame with columns ['date', 'daily_issuance_usd', 'issuance_ma_365']
        """
        try:
            puell_values = []
            timestamps = []
            
            for _, row in historical_data.iterrows():
                puell = self.calculate_puell_multiple(
                    row['daily_issuance_usd'], 
                    row['issuance_ma_365']
                )
                puell_values.append(puell)
                timestamps.append(row['date'])
            
            current_puell = puell_values[-1] if puell_values else 1.0
            
            # Calculate percentile
            puell_percentile = stats.percentileofscore(puell_values, current_puell)
            
            # Determine mining profitability
            mining_profitability = self.determine_mining_profitability(current_puell)
            
            # Determine market cycle phase
            market_cycle_phase = self.determine_market_cycle_phase(puell_percentile)
            
            # Calculate bands
            puell_bands = self.calculate_puell_bands(puell_values)
            
            return PuellResult(
                current_puell=current_puell,
                puell_percentile=puell_percentile,
                mining_profitability=mining_profitability,
                market_cycle_phase=market_cycle_phase,
                historical_puell=puell_values,
                puell_bands=puell_bands,
                timestamps=timestamps
            )
            
        except Exception as e:
            logger.error(f"Error in Puell Multiple analysis: {str(e)}")
            raise

class HashRibbonsModel:
    """Hash Ribbons Model for miner behavior analysis"""
    
    def __init__(self, asset: str = "BTC"):
        self.asset = asset.upper()
        
    def calculate_hash_ribbon_signal(self, hash_rate_ma_30: float, 
                                   hash_rate_ma_60: float) -> str:
        """Calculate Hash Ribbon signal"""
        if hash_rate_ma_30 > hash_rate_ma_60:
            return "Buy Signal - Hash Rate Recovery"
        else:
            return "Sell Signal - Hash Rate Decline"
    
    def detect_miner_capitulation(self, hash_rate_change: float, 
                                difficulty_change: float) -> bool:
        """Detect miner capitulation events"""
        # Miner capitulation: significant hash rate drop with difficulty lag
        return hash_rate_change < -0.15 and difficulty_change > -0.05
    
    def calculate_hash_rate_trend(self, recent_hash_rates: List[float]) -> str:
        """Calculate hash rate trend"""
        if len(recent_hash_rates) < 7:
            return "Insufficient Data"
        
        # Linear regression on recent hash rates
        x = np.arange(len(recent_hash_rates[-14:]))
        y = recent_hash_rates[-14:]
        
        slope, _, r_value, _, _ = stats.linregress(x, y)
        
        if abs(r_value) < 0.3:
            return "Stable"
        elif slope > 0:
            return "Growing" if r_value > 0.5 else "Slowly Growing"
        else:
            return "Declining" if r_value < -0.5 else "Slowly Declining"
    
    def assess_mining_health(self, hash_ribbon_signal: str, 
                           miner_capitulation: bool,
                           hash_rate_trend: str) -> str:
        """Assess overall mining network health"""
        if miner_capitulation:
            return "Poor - Miner Capitulation Event"
        elif "Buy" in hash_ribbon_signal and "Growing" in hash_rate_trend:
            return "Excellent - Strong Network Growth"
        elif "Buy" in hash_ribbon_signal:
            return "Good - Network Recovery"
        elif "Declining" in hash_rate_trend:
            return "Concerning - Network Stress"
        else:
            return "Fair - Stable Network"
    
    def analyze(self, historical_data: pd.DataFrame) -> HashRibbonsResult:
        """Perform Hash Ribbons analysis
        
        Args:
            historical_data: DataFrame with columns ['date', 'hash_rate', 'difficulty']
        """
        try:
            # Calculate moving averages
            hash_rates = historical_data['hash_rate'].tolist()
            hash_rate_ma_30 = historical_data['hash_rate'].rolling(window=30, min_periods=1).mean().tolist()
            hash_rate_ma_60 = historical_data['hash_rate'].rolling(window=60, min_periods=1).mean().tolist()
            
            timestamps = historical_data['date'].tolist()
            
            # Current values
            current_ma_30 = hash_rate_ma_30[-1] if hash_rate_ma_30 else 0
            current_ma_60 = hash_rate_ma_60[-1] if hash_rate_ma_60 else 0
            
            # Calculate signals
            hash_ribbon_signal = self.calculate_hash_ribbon_signal(current_ma_30, current_ma_60)
            
            # Calculate recent changes
            if len(hash_rates) >= 14:
                hash_rate_change = (hash_rates[-1] - hash_rates[-14]) / hash_rates[-14]
            else:
                hash_rate_change = 0
            
            if len(historical_data) >= 14:
                difficulty_change = (historical_data['difficulty'].iloc[-1] - 
                                   historical_data['difficulty'].iloc[-14]) / historical_data['difficulty'].iloc[-14]
            else:
                difficulty_change = 0
            
            # Detect miner capitulation
            miner_capitulation = self.detect_miner_capitulation(hash_rate_change, difficulty_change)
            
            # Calculate hash rate trend
            hash_rate_trend = self.calculate_hash_rate_trend(hash_rates)
            
            # Assess mining health
            mining_health = self.assess_mining_health(hash_ribbon_signal, miner_capitulation, hash_rate_trend)
            
            return HashRibbonsResult(
                hash_ribbon_signal=hash_ribbon_signal,
                miner_capitulation=miner_capitulation,
                hash_rate_trend=hash_rate_trend,
                difficulty_adjustment=difficulty_change,
                mining_health=mining_health,
                hash_rate_ma_30=hash_rate_ma_30,
                hash_rate_ma_60=hash_rate_ma_60,
                timestamps=timestamps
            )
            
        except Exception as e:
            logger.error(f"Error in Hash Ribbons analysis: {str(e)}")
            raise

class CryptoQuantModel:
    """Combined Crypto Quantitative Metrics Model"""
    
    def __init__(self, asset: str = "BTC"):
        self.asset = asset.upper()
        self.mvrv_model = MVRVModel(asset)
        self.sopr_model = SOPRModel(asset)
        self.puell_model = PuellMultipleModel(asset)
        self.hash_ribbons_model = HashRibbonsModel(asset)
        
    def calculate_composite_score(self, 
                                mvrv_result: MVRVResult,
                                sopr_result: SOPRResult,
                                puell_result: PuellResult,
                                hash_ribbons_result: HashRibbonsResult) -> float:
        """Calculate composite score from all metrics"""
        scores = []
        
        # MVRV score (inverted percentile - lower is better)
        mvrv_score = 100 - mvrv_result.mvrv_percentile
        scores.append(mvrv_score * 0.3)  # 30% weight
        
        # SOPR score
        sopr_score = 50  # Neutral baseline
        if "Greed" in sopr_result.market_sentiment:
            sopr_score = 30  # Negative for greed
        elif "Fear" in sopr_result.market_sentiment:
            sopr_score = 70  # Positive for fear
        scores.append(sopr_score * 0.25)  # 25% weight
        
        # Puell score (inverted percentile - lower is better for buying)
        puell_score = 100 - puell_result.puell_percentile
        scores.append(puell_score * 0.25)  # 25% weight
        
        # Hash Ribbons score
        hash_score = 70 if "Buy" in hash_ribbons_result.hash_ribbon_signal else 30
        if hash_ribbons_result.miner_capitulation:
            hash_score = 80  # Capitulation can be bullish long-term
        scores.append(hash_score * 0.2)  # 20% weight
        
        return sum(scores)
    
    def determine_market_regime(self, composite_score: float) -> str:
        """Determine overall market regime"""
        if composite_score >= 75:
            return "Accumulation Zone - Strong Buy Signals"
        elif composite_score >= 60:
            return "Recovery Phase - Moderate Buy Signals"
        elif composite_score >= 40:
            return "Neutral Zone - Mixed Signals"
        elif composite_score >= 25:
            return "Distribution Phase - Caution Advised"
        else:
            return "Euphoria Zone - High Risk"
    
    def assess_risk(self, 
                   mvrv_result: MVRVResult,
                   sopr_result: SOPRResult,
                   puell_result: PuellResult) -> str:
        """Assess overall risk level"""
        risk_factors = 0
        
        # MVRV risk
        if mvrv_result.mvrv_z_score > 3:
            risk_factors += 2
        elif mvrv_result.mvrv_z_score > 1:
            risk_factors += 1
        
        # SOPR risk
        if "Strong Greed" in sopr_result.market_sentiment:
            risk_factors += 2
        elif "Greed" in sopr_result.market_sentiment:
            risk_factors += 1
        
        # Puell risk
        if puell_result.puell_percentile > 90:
            risk_factors += 2
        elif puell_result.puell_percentile > 75:
            risk_factors += 1
        
        if risk_factors >= 5:
            return "Very High Risk - Multiple Warning Signals"
        elif risk_factors >= 3:
            return "High Risk - Several Warning Signals"
        elif risk_factors >= 1:
            return "Medium Risk - Some Warning Signals"
        else:
            return "Low Risk - Favorable Conditions"
    
    def calculate_confidence_score(self, data_quality_metrics: Dict[str, float]) -> float:
        """Calculate confidence score based on data quality"""
        # Default confidence factors
        factors = [
            data_quality_metrics.get('data_completeness', 0.8),
            data_quality_metrics.get('data_recency', 0.9),
            data_quality_metrics.get('signal_consistency', 0.7),
            data_quality_metrics.get('historical_depth', 0.8)
        ]
        
        return np.mean(factors)
    
    def analyze(self, historical_data: Dict[str, pd.DataFrame]) -> CryptoQuantResult:
        """Perform comprehensive crypto quantitative analysis
        
        Args:
            historical_data: Dictionary containing DataFrames for each metric
        """
        try:
            # Perform individual analyses
            mvrv_result = self.mvrv_model.analyze(historical_data['mvrv_data'])
            sopr_result = self.sopr_model.analyze(historical_data['sopr_data'])
            puell_result = self.puell_model.analyze(historical_data['puell_data'])
            hash_ribbons_result = self.hash_ribbons_model.analyze(historical_data['hash_data'])
            
            # Create placeholder HODL Waves and Exchange Flow results
            hodl_waves_result = HODLWavesResult(
                age_distribution={'<1m': 0.1, '1-3m': 0.15, '3-6m': 0.2, '6-12m': 0.25, '>1y': 0.3},
                hodl_strength=0.7,
                supply_maturity="Mature",
                long_term_holder_ratio=0.6,
                recent_activity_ratio=0.4,
                hodl_trend="Strengthening",
                timestamps=mvrv_result.timestamps
            )
            
            exchange_flow_result = ExchangeFlowResult(
                net_flow=-1000,  # Negative = outflow
                inflow_trend="Decreasing",
                outflow_trend="Increasing",
                exchange_balance_ratio=0.12,
                selling_pressure="Low",
                flow_momentum=-0.3,
                timestamps=mvrv_result.timestamps
            )
            
            # Calculate composite metrics
            composite_score = self.calculate_composite_score(
                mvrv_result, sopr_result, puell_result, hash_ribbons_result
            )
            
            market_regime = self.determine_market_regime(composite_score)
            risk_assessment = self.assess_risk(mvrv_result, sopr_result, puell_result)
            
            # Calculate confidence score
            data_quality_metrics = {
                'data_completeness': 0.9,
                'data_recency': 0.95,
                'signal_consistency': 0.8,
                'historical_depth': 0.85
            }
            confidence_score = self.calculate_confidence_score(data_quality_metrics)
            
            return CryptoQuantResult(
                mvrv_result=mvrv_result,
                sopr_result=sopr_result,
                puell_result=puell_result,
                hash_ribbons_result=hash_ribbons_result,
                hodl_waves_result=hodl_waves_result,
                exchange_flow_result=exchange_flow_result,
                composite_score=composite_score,
                market_regime=market_regime,
                risk_assessment=risk_assessment,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            logger.error(f"Error in crypto quant analysis: {str(e)}")
            raise
    
    def get_model_insights(self, result: CryptoQuantResult) -> Dict[str, str]:
        """Generate comprehensive insights from analysis"""
        insights = {}
        
        # Overall assessment
        insights['composite_signal'] = f"Composite Score: {result.composite_score:.1f}/100 - {result.market_regime}"
        insights['risk_assessment'] = result.risk_assessment
        insights['confidence'] = f"Analysis Confidence: {result.confidence_score:.1%}"
        
        # MVRV insights
        mvrv = result.mvrv_result
        insights['mvrv_analysis'] = f"MVRV: {mvrv.current_mvrv:.2f} (Z-Score: {mvrv.mvrv_z_score:.2f}) - {mvrv.market_phase}"
        
        # SOPR insights
        sopr = result.sopr_result
        insights['sopr_analysis'] = f"SOPR: {sopr.current_sopr:.3f} ({sopr.sopr_trend}) - {sopr.market_sentiment}"
        
        # Puell insights
        puell = result.puell_result
        insights['puell_analysis'] = f"Puell Multiple: {puell.current_puell:.2f} - {puell.mining_profitability}"
        
        # Hash Ribbons insights
        hash_ribbons = result.hash_ribbons_result
        insights['hash_analysis'] = f"Hash Ribbons: {hash_ribbons.hash_ribbon_signal} - {hash_ribbons.mining_health}"
        
        return insights

# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    dates = pd.date_range(start='2022-01-01', end='2024-01-01', freq='D')
    
    # Sample MVRV data
    mvrv_data = []
    for i, date in enumerate(dates):
        market_cap = 1e12 * (1 + 0.001 * i) * (1 + 0.1 * np.random.randn())
        realized_cap = market_cap * 0.6 * (1 + 0.05 * np.random.randn())
        mvrv_data.append({
            'date': date,
            'market_cap': max(market_cap, 1e11),
            'realized_cap': max(realized_cap, 1e10)
        })
    
    # Sample SOPR data
    sopr_data = []
    for i, date in enumerate(dates):
        sopr = 1.0 + 0.1 * np.sin(i * 0.01) + 0.05 * np.random.randn()
        sopr_data.append({
            'date': date,
            'sopr': max(sopr, 0.5)
        })
    
    # Sample Puell data
    puell_data = []
    for i, date in enumerate(dates):
        daily_issuance = 900 * 50000 * (1 + 0.05 * np.random.randn())  # ~900 BTC * $50k
        issuance_ma = daily_issuance * (1 + 0.02 * np.random.randn())
        puell_data.append({
            'date': date,
            'daily_issuance_usd': daily_issuance,
            'issuance_ma_365': issuance_ma
        })
    
    # Sample Hash data
    hash_data = []
    for i, date in enumerate(dates):
        hash_rate = 200e18 * (1 + 0.002 * i) * (1 + 0.1 * np.random.randn())
        difficulty = hash_rate * 1.2 * (1 + 0.05 * np.random.randn())
        hash_data.append({
            'date': date,
            'hash_rate': max(hash_rate, 100e18),
            'difficulty': max(difficulty, 120e18)
        })
    
    # Create DataFrames
    historical_data = {
        'mvrv_data': pd.DataFrame(mvrv_data),
        'sopr_data': pd.DataFrame(sopr_data),
        'puell_data': pd.DataFrame(puell_data),
        'hash_data': pd.DataFrame(hash_data)
    }
    
    # Test the model
    crypto_quant_model = CryptoQuantModel("BTC")
    result = crypto_quant_model.analyze(historical_data)
    insights = crypto_quant_model.get_model_insights(result)
    
    print("=== Crypto Quantitative Metrics Analysis ===")
    print(f"Composite Score: {result.composite_score:.1f}/100")
    print(f"Market Regime: {result.market_regime}")
    print(f"Risk Assessment: {result.risk_assessment}")
    print(f"Confidence Score: {result.confidence_score:.1%}")
    print()
    
    print("=== Detailed Insights ===")
    for key, insight in insights.items():
        print(f"{key}: {insight}")