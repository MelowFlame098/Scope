"""Exchange Flow Model for Cryptocurrency Exchange Activity Analysis

This module implements the Exchange Flow calculation and analysis for cryptocurrency markets.
Exchange Flow analyzes the movement of coins to and from exchanges.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from scipy import stats
import logging

logger = logging.getLogger(__name__)

@dataclass
class WhaleTrackingAnalysis:
    """Whale tracking and large holder flow analysis"""
    whale_inflow_volume: float
    whale_outflow_volume: float
    whale_net_flow: float
    whale_flow_percentage: float
    large_transaction_count: int
    whale_accumulation_score: float
    whale_distribution_score: float
    whale_flow_momentum: float
    whale_exchange_dominance: float
    whale_behavior_pattern: str
    whale_impact_on_price: float
    whale_capitulation_risk: float

@dataclass
class InstitutionalFlowDetection:
    """Institutional flow detection and analysis"""
    institutional_inflow_volume: float
    institutional_outflow_volume: float
    institutional_net_flow: float
    institutional_flow_percentage: float
    institutional_transaction_patterns: Dict[str, float]
    custody_flow_indicators: Dict[str, float]
    otc_flow_estimation: float
    institutional_accumulation_phase: str
    institutional_sentiment_score: float
    regulatory_flow_impact: float
    institutional_vs_retail_ratio: float
    institutional_flow_predictability: float

@dataclass
class ExchangeSpecificAnalytics:
    """Exchange-specific flow analytics and insights"""
    exchange_flow_breakdown: Dict[str, Dict[str, float]]
    exchange_market_share: Dict[str, float]
    exchange_flow_correlation: Dict[str, float]
    arbitrage_flow_indicators: Dict[str, float]
    exchange_liquidity_metrics: Dict[str, float]
    cross_exchange_flow_patterns: Dict[str, float]
    exchange_specific_whale_activity: Dict[str, Dict[str, float]]
    exchange_flow_anomalies: Dict[str, float]
    exchange_dominance_trends: Dict[str, str]
    exchange_flow_efficiency: Dict[str, float]
    regulatory_exchange_impact: Dict[str, float]
    exchange_flow_predictive_signals: Dict[str, float]

@dataclass
class ExchangeFlowResult:
    """Results from Exchange Flow analysis"""
    net_flow: float
    inflow_trend: str
    outflow_trend: str
    exchange_balance_ratio: float
    selling_pressure: str
    flow_momentum: float
    timestamps: List[str]
    
    # Enhanced analysis components
    whale_tracking: Optional['WhaleTrackingAnalysis'] = None
    institutional_flow: Optional['InstitutionalFlowDetection'] = None
    exchange_analytics: Optional['ExchangeSpecificAnalytics'] = None

class ExchangeFlowModel:
    """Exchange Flow Model for exchange activity analysis"""
    
    def __init__(self, asset: str = "BTC"):
        self.asset = asset.upper()
        
    def calculate_net_flow(self, inflow: float, outflow: float) -> float:
        """Calculate net flow (positive = net inflow, negative = net outflow)"""
        return inflow - outflow
    
    def calculate_flow_trend(self, recent_flows: List[float], window: int = 7) -> str:
        """Calculate flow trend direction"""
        if len(recent_flows) < window:
            return "Insufficient Data"
        
        # Linear regression on recent flows
        x = np.arange(len(recent_flows[-window:]))
        y = recent_flows[-window:]
        
        slope, _, r_value, _, _ = stats.linregress(x, y)
        
        if abs(r_value) < 0.3:
            return "Stable"
        elif slope > 0:
            return "Increasing" if r_value > 0.5 else "Slowly Increasing"
        else:
            return "Decreasing" if r_value < -0.5 else "Slowly Decreasing"
    
    def calculate_exchange_balance_ratio(self, exchange_balance: float, total_supply: float) -> float:
        """Calculate ratio of coins held on exchanges"""
        if total_supply <= 0:
            return 0.0
        
        return exchange_balance / total_supply
    
    def assess_selling_pressure(self, net_flow: float, inflow_trend: str, 
                              exchange_balance_ratio: float) -> str:
        """Assess selling pressure based on flow metrics"""
        if net_flow > 1000 and "Increasing" in inflow_trend:
            return "High - Strong Inflows to Exchanges"
        elif net_flow > 0 and exchange_balance_ratio > 0.15:
            return "Moderate - Some Selling Pressure"
        elif net_flow < -1000:
            return "Low - Strong Outflows from Exchanges"
        elif exchange_balance_ratio < 0.10:
            return "Very Low - Limited Exchange Supply"
        else:
            return "Neutral - Balanced Flow"
    
    def calculate_flow_momentum(self, recent_net_flows: List[float]) -> float:
        """Calculate flow momentum indicator"""
        if len(recent_net_flows) < 7:
            return 0.0
        
        # Calculate momentum as rate of change in net flows
        recent_avg = np.mean(recent_net_flows[-3:])
        older_avg = np.mean(recent_net_flows[-7:-3]) if len(recent_net_flows) >= 7 else recent_avg
        
        if older_avg == 0:
            return 0.0
        
        return (recent_avg - older_avg) / abs(older_avg)
    
    def analyze(self, historical_data: pd.DataFrame) -> ExchangeFlowResult:
        """Perform Exchange Flow analysis
        
        Args:
            historical_data: DataFrame with columns ['date', 'inflow', 'outflow', 'exchange_balance', 'total_supply']
        """
        try:
            # Calculate net flows
            net_flows = []
            inflows = []
            outflows = []
            timestamps = []
            
            for _, row in historical_data.iterrows():
                net_flow = self.calculate_net_flow(row['inflow'], row['outflow'])
                net_flows.append(net_flow)
                inflows.append(row['inflow'])
                outflows.append(row['outflow'])
                timestamps.append(row['date'])
            
            # Current values
            current_net_flow = net_flows[-1] if net_flows else 0
            
            # Calculate trends
            inflow_trend = self.calculate_flow_trend(inflows)
            outflow_trend = self.calculate_flow_trend(outflows)
            
            # Calculate exchange balance ratio
            if len(historical_data) > 0:
                latest_row = historical_data.iloc[-1]
                exchange_balance_ratio = self.calculate_exchange_balance_ratio(
                    latest_row['exchange_balance'], 
                    latest_row['total_supply']
                )
            else:
                exchange_balance_ratio = 0.12  # Default value
            
            # Assess selling pressure
            selling_pressure = self.assess_selling_pressure(
                current_net_flow, inflow_trend, exchange_balance_ratio
            )
            
            # Calculate flow momentum
            flow_momentum = self.calculate_flow_momentum(net_flows)
            
            # Enhanced analysis calculations
            whale_tracking = None
            institutional_flow = None
            exchange_analytics = None
            
            try:
                whale_tracking = self._analyze_whale_tracking(historical_data, net_flows, inflows, outflows)
                logger.info("Whale tracking analysis completed")
            except Exception as e:
                logger.warning(f"Whale tracking analysis failed: {str(e)}")
            
            try:
                institutional_flow = self._analyze_institutional_flow(historical_data, net_flows, inflows, outflows)
                logger.info("Institutional flow analysis completed")
            except Exception as e:
                logger.warning(f"Institutional flow analysis failed: {str(e)}")
            
            try:
                exchange_analytics = self._analyze_exchange_specific_analytics(historical_data, net_flows)
                logger.info("Exchange-specific analytics completed")
            except Exception as e:
                logger.warning(f"Exchange-specific analytics failed: {str(e)}")
            
            return ExchangeFlowResult(
                net_flow=current_net_flow,
                inflow_trend=inflow_trend,
                outflow_trend=outflow_trend,
                exchange_balance_ratio=exchange_balance_ratio,
                selling_pressure=selling_pressure,
                flow_momentum=flow_momentum,
                timestamps=timestamps,
                whale_tracking=whale_tracking,
                institutional_flow=institutional_flow,
                exchange_analytics=exchange_analytics
            )
            
        except Exception as e:
            logger.error(f"Error in Exchange Flow analysis: {str(e)}")
            raise
    
    def _analyze_whale_tracking(self, historical_data: pd.DataFrame, net_flows: List[float], 
                               inflows: List[float], outflows: List[float]) -> WhaleTrackingAnalysis:
        """Analyze whale tracking and large holder flow patterns"""
        try:
            # Calculate whale flow metrics (assuming whale transactions > 100 BTC)
            whale_threshold = 100.0
            total_inflow = sum(inflows)
            total_outflow = sum(outflows)
            
            # Estimate whale flows (simplified - in practice would need transaction size data)
            whale_inflow_volume = total_inflow * 0.3  # Assume 30% from whales
            whale_outflow_volume = total_outflow * 0.25  # Assume 25% from whales
            whale_net_flow = whale_inflow_volume - whale_outflow_volume
            
            # Calculate whale flow percentage
            total_flow = total_inflow + total_outflow
            whale_flow_percentage = ((whale_inflow_volume + whale_outflow_volume) / total_flow * 100) if total_flow > 0 else 0
            
            # Estimate large transaction count
            large_transaction_count = int(len(historical_data) * 0.15)  # Assume 15% are large transactions
            
            # Calculate accumulation/distribution scores
            whale_accumulation_score = max(0, whale_net_flow / max(abs(whale_net_flow), 1)) * 100
            whale_distribution_score = max(0, -whale_net_flow / max(abs(whale_net_flow), 1)) * 100
            
            # Calculate whale flow momentum
            recent_whale_flows = net_flows[-7:] if len(net_flows) >= 7 else net_flows
            whale_flow_momentum = np.mean(recent_whale_flows) if recent_whale_flows else 0
            
            # Calculate whale exchange dominance
            whale_exchange_dominance = whale_flow_percentage / 100 * 0.8  # Scaled metric
            
            # Determine whale behavior pattern
            if whale_accumulation_score > 60:
                whale_behavior_pattern = "Strong Accumulation"
            elif whale_distribution_score > 60:
                whale_behavior_pattern = "Strong Distribution"
            elif abs(whale_net_flow) < total_flow * 0.05:
                whale_behavior_pattern = "Neutral/Sideways"
            else:
                whale_behavior_pattern = "Mixed Signals"
            
            # Calculate whale impact on price (correlation estimate)
            whale_impact_on_price = abs(whale_net_flow) / max(total_flow, 1) * 0.7
            
            # Calculate whale capitulation risk
            whale_capitulation_risk = max(0, whale_distribution_score - 50) / 50 * 100
            
            return WhaleTrackingAnalysis(
                whale_inflow_volume=whale_inflow_volume,
                whale_outflow_volume=whale_outflow_volume,
                whale_net_flow=whale_net_flow,
                whale_flow_percentage=whale_flow_percentage,
                large_transaction_count=large_transaction_count,
                whale_accumulation_score=whale_accumulation_score,
                whale_distribution_score=whale_distribution_score,
                whale_flow_momentum=whale_flow_momentum,
                whale_exchange_dominance=whale_exchange_dominance,
                whale_behavior_pattern=whale_behavior_pattern,
                whale_impact_on_price=whale_impact_on_price,
                whale_capitulation_risk=whale_capitulation_risk
            )
            
        except Exception as e:
            logger.error(f"Error in whale tracking analysis: {str(e)}")
            raise
    
    def _analyze_institutional_flow(self, historical_data: pd.DataFrame, net_flows: List[float],
                                   inflows: List[float], outflows: List[float]) -> InstitutionalFlowDetection:
        """Analyze institutional flow detection and patterns"""
        try:
            total_inflow = sum(inflows)
            total_outflow = sum(outflows)
            
            # Estimate institutional flows (typically larger, more regular patterns)
            institutional_inflow_volume = total_inflow * 0.4  # Assume 40% institutional
            institutional_outflow_volume = total_outflow * 0.35  # Assume 35% institutional
            institutional_net_flow = institutional_inflow_volume - institutional_outflow_volume
            
            # Calculate institutional flow percentage
            total_flow = total_inflow + total_outflow
            institutional_flow_percentage = ((institutional_inflow_volume + institutional_outflow_volume) / total_flow * 100) if total_flow > 0 else 0
            
            # Analyze transaction patterns (simplified)
            institutional_transaction_patterns = {
                "regular_intervals": 0.7,  # High regularity
                "large_block_trades": 0.6,
                "off_hours_activity": 0.4,
                "cross_exchange_coordination": 0.5
            }
            
            # Custody flow indicators
            custody_flow_indicators = {
                "custody_inflows": institutional_inflow_volume * 0.6,
                "custody_outflows": institutional_outflow_volume * 0.4,
                "custody_net_change": institutional_net_flow * 0.5,
                "custody_dominance": 0.3
            }
            
            # OTC flow estimation
            otc_flow_estimation = abs(institutional_net_flow) * 0.3
            
            # Determine institutional accumulation phase
            if institutional_net_flow > total_flow * 0.1:
                institutional_accumulation_phase = "Strong Accumulation"
            elif institutional_net_flow < -total_flow * 0.1:
                institutional_accumulation_phase = "Distribution Phase"
            else:
                institutional_accumulation_phase = "Neutral Phase"
            
            # Calculate institutional sentiment score
            institutional_sentiment_score = (institutional_net_flow / max(abs(institutional_net_flow), 1)) * 50 + 50
            
            # Regulatory flow impact (simplified)
            regulatory_flow_impact = abs(institutional_net_flow) / max(total_flow, 1) * 0.2
            
            # Institutional vs retail ratio
            retail_flow = total_flow - (institutional_inflow_volume + institutional_outflow_volume)
            institutional_vs_retail_ratio = (institutional_inflow_volume + institutional_outflow_volume) / max(retail_flow, 1)
            
            # Flow predictability score
            flow_variance = np.var(net_flows) if len(net_flows) > 1 else 0
            institutional_flow_predictability = max(0, 100 - (flow_variance / max(np.mean(net_flows), 1)) * 10)
            
            return InstitutionalFlowDetection(
                institutional_inflow_volume=institutional_inflow_volume,
                institutional_outflow_volume=institutional_outflow_volume,
                institutional_net_flow=institutional_net_flow,
                institutional_flow_percentage=institutional_flow_percentage,
                institutional_transaction_patterns=institutional_transaction_patterns,
                custody_flow_indicators=custody_flow_indicators,
                otc_flow_estimation=otc_flow_estimation,
                institutional_accumulation_phase=institutional_accumulation_phase,
                institutional_sentiment_score=institutional_sentiment_score,
                regulatory_flow_impact=regulatory_flow_impact,
                institutional_vs_retail_ratio=institutional_vs_retail_ratio,
                institutional_flow_predictability=institutional_flow_predictability
            )
            
        except Exception as e:
            logger.error(f"Error in institutional flow analysis: {str(e)}")
            raise
    
    def _analyze_exchange_specific_analytics(self, historical_data: pd.DataFrame, 
                                           net_flows: List[float]) -> ExchangeSpecificAnalytics:
        """Analyze exchange-specific flow analytics and insights"""
        try:
            # Simulate exchange-specific data (in practice, would need real exchange data)
            exchanges = ["Binance", "Coinbase", "Kraken", "Bitfinex", "Huobi"]
            total_flow = sum(abs(flow) for flow in net_flows)
            
            # Exchange flow breakdown
            exchange_flow_breakdown = {}
            market_shares = [0.35, 0.25, 0.15, 0.15, 0.10]  # Simulated market shares
            
            for i, exchange in enumerate(exchanges):
                exchange_flow_breakdown[exchange] = {
                    "inflow": total_flow * market_shares[i] * 0.6,
                    "outflow": total_flow * market_shares[i] * 0.4,
                    "net_flow": total_flow * market_shares[i] * 0.2
                }
            
            # Exchange market share
            exchange_market_share = {exchange: share for exchange, share in zip(exchanges, market_shares)}
            
            # Exchange flow correlation
            exchange_flow_correlation = {
                exchange: np.random.uniform(0.3, 0.8) for exchange in exchanges
            }
            
            # Arbitrage flow indicators
            arbitrage_flow_indicators = {
                "cross_exchange_spread": 0.15,
                "arbitrage_volume": total_flow * 0.05,
                "arbitrage_frequency": 0.3,
                "arbitrage_efficiency": 0.7
            }
            
            # Exchange liquidity metrics
            exchange_liquidity_metrics = {
                exchange: {
                    "depth": np.random.uniform(1000, 5000),
                    "spread": np.random.uniform(0.01, 0.05),
                    "volume": total_flow * market_shares[i]
                } for i, exchange in enumerate(exchanges)
            }
            
            # Cross-exchange flow patterns
            cross_exchange_flow_patterns = {
                "flow_synchronization": 0.6,
                "lead_lag_relationships": 0.4,
                "flow_divergence": 0.3,
                "cross_exchange_momentum": 0.5
            }
            
            # Exchange-specific whale activity
            exchange_specific_whale_activity = {
                exchange: {
                    "whale_inflow": exchange_flow_breakdown[exchange]["inflow"] * 0.3,
                    "whale_outflow": exchange_flow_breakdown[exchange]["outflow"] * 0.25,
                    "whale_dominance": market_shares[i] * 0.4
                } for i, exchange in enumerate(exchanges)
            }
            
            # Exchange flow anomalies
            exchange_flow_anomalies = {
                exchange: abs(np.random.normal(0, 0.1)) for exchange in exchanges
            }
            
            # Exchange dominance trends
            exchange_dominance_trends = {
                exchange: "Increasing" if market_shares[i] > 0.2 else "Stable" 
                for i, exchange in enumerate(exchanges)
            }
            
            # Exchange flow efficiency
            exchange_flow_efficiency = {
                exchange: min(1.0, market_shares[i] * 2) for i, exchange in enumerate(exchanges)
            }
            
            # Regulatory exchange impact
            regulatory_exchange_impact = {
                exchange: np.random.uniform(0.1, 0.4) for exchange in exchanges
            }
            
            # Exchange flow predictive signals
            exchange_flow_predictive_signals = {
                exchange: np.random.uniform(0.3, 0.8) for exchange in exchanges
            }
            
            return ExchangeSpecificAnalytics(
                exchange_flow_breakdown=exchange_flow_breakdown,
                exchange_market_share=exchange_market_share,
                exchange_flow_correlation=exchange_flow_correlation,
                arbitrage_flow_indicators=arbitrage_flow_indicators,
                exchange_liquidity_metrics=exchange_liquidity_metrics,
                cross_exchange_flow_patterns=cross_exchange_flow_patterns,
                exchange_specific_whale_activity=exchange_specific_whale_activity,
                exchange_flow_anomalies=exchange_flow_anomalies,
                exchange_dominance_trends=exchange_dominance_trends,
                exchange_flow_efficiency=exchange_flow_efficiency,
                regulatory_exchange_impact=regulatory_exchange_impact,
                exchange_flow_predictive_signals=exchange_flow_predictive_signals
            )
            
        except Exception as e:
            logger.error(f"Error in exchange-specific analytics: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    dates = pd.date_range(start='2022-01-01', end='2024-01-01', freq='D')
    
    # Sample Exchange Flow data
    flow_data = []
    total_supply = 19_000_000  # Approximate BTC supply
    exchange_balance = total_supply * 0.12  # 12% on exchanges initially
    
    for i, date in enumerate(dates):
        # Generate realistic flow patterns
        base_inflow = 5000 + 1000 * np.sin(i * 0.01) + 500 * np.random.randn()
        base_outflow = 5200 + 800 * np.sin(i * 0.01 + np.pi/4) + 400 * np.random.randn()
        
        inflow = max(base_inflow, 0)
        outflow = max(base_outflow, 0)
        
        # Update exchange balance
        exchange_balance += inflow - outflow
        exchange_balance = max(exchange_balance, total_supply * 0.05)  # Minimum 5%
        exchange_balance = min(exchange_balance, total_supply * 0.20)  # Maximum 20%
        
        flow_data.append({
            'date': date,
            'inflow': inflow,
            'outflow': outflow,
            'exchange_balance': exchange_balance,
            'total_supply': total_supply
        })
    
    # Create DataFrame
    historical_data = pd.DataFrame(flow_data)
    
    # Test the model
    exchange_flow_model = ExchangeFlowModel("BTC")
    result = exchange_flow_model.analyze(historical_data)
    
    print("=== Exchange Flow Analysis Results ===")
    print(f"Net Flow: {result.net_flow:.0f} coins")
    print(f"Inflow Trend: {result.inflow_trend}")
    print(f"Outflow Trend: {result.outflow_trend}")
    print(f"Exchange Balance Ratio: {result.exchange_balance_ratio:.2%}")
    print(f"Selling Pressure: {result.selling_pressure}")
    print(f"Flow Momentum: {result.flow_momentum:.3f}")