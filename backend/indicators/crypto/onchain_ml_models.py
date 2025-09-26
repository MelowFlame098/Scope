"""On-chain ML Models for Cryptocurrency Analysis

This module implements advanced machine learning models for on-chain data analysis:
- Transaction Flow Analysis
- Address Clustering and Classification
- Whale Movement Detection
- Network Health Metrics
- Price Prediction from On-chain Data
- Anomaly Detection
- Market Cycle Identification
- Liquidity Analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
try:
    from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingRegressor
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, mean_squared_error, r2_score
    from sklearn.decomposition import PCA
    from sklearn.svm import OneClassSVM
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available. Using simplified implementations.")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logging.warning("NetworkX not available. Network analysis will be limited.")

logger = logging.getLogger(__name__)

@dataclass
class TransactionFlowResult:
    """Transaction flow analysis result"""
    total_volume: float
    unique_addresses: int
    transaction_count: int
    average_transaction_size: float
    large_transaction_ratio: float
    flow_concentration: float
    velocity: float
    circulation_ratio: float
    dormancy_flow: float
    exchange_flow_ratio: float
    timestamps: List[datetime]
    flow_metrics: List[float]

@dataclass
class AddressClusteringResult:
    """Address clustering and classification result"""
    exchange_addresses: List[str]
    whale_addresses: List[str]
    miner_addresses: List[str]
    defi_addresses: List[str]
    institutional_addresses: List[str]
    retail_addresses: List[str]
    cluster_confidence: Dict[str, float]
    address_risk_scores: Dict[str, float]
    clustering_quality: float
    total_clusters: int

@dataclass
class WhaleMovementResult:
    """Whale movement detection result"""
    whale_transactions: List[Dict]
    whale_accumulation_score: float
    whale_distribution_score: float
    whale_activity_trend: str
    large_holder_concentration: float
    whale_exchange_flows: Dict[str, float]
    dormant_whale_awakening: List[Dict]
    whale_sentiment_indicator: float
    market_impact_probability: float

@dataclass
class NetworkHealthResult:
    """Network health metrics result"""
    active_addresses: int
    new_addresses: int
    address_growth_rate: float
    transaction_throughput: float
    network_utilization: float
    fee_pressure: float
    congestion_score: float
    decentralization_index: float
    network_security_score: float
    adoption_momentum: float

@dataclass
class AnomalyDetectionResult:
    """Anomaly detection result"""
    anomalous_transactions: List[Dict]
    anomaly_score: float
    suspicious_patterns: List[str]
    risk_level: str
    confidence_score: float
    anomaly_types: Dict[str, int]
    temporal_anomalies: List[datetime]
    network_anomalies: List[str]

@dataclass
class MarketCycleResult:
    """Market cycle identification result"""
    current_cycle_phase: str
    cycle_completion: float
    cycle_duration_estimate: int
    next_phase_prediction: str
    cycle_strength: float
    historical_cycles: List[Dict]
    cycle_indicators: Dict[str, float]
    phase_transition_probability: float

@dataclass
class LiquidityAnalysisResult:
    """Liquidity analysis result"""
    liquid_supply: float
    illiquid_supply: float
    liquidity_ratio: float
    supply_shock_risk: float
    liquidity_trend: str
    exchange_reserves: float
    staking_ratio: float
    long_term_holder_ratio: float
    supply_distribution: Dict[str, float]

@dataclass
class OnChainMLResult:
    """Combined on-chain ML analysis result"""
    transaction_flow: TransactionFlowResult
    address_clustering: AddressClusteringResult
    whale_movement: WhaleMovementResult
    network_health: NetworkHealthResult
    anomaly_detection: AnomalyDetectionResult
    market_cycle: MarketCycleResult
    liquidity_analysis: LiquidityAnalysisResult
    price_prediction: float
    confidence_score: float
    overall_network_score: float

class TransactionFlowAnalyzer:
    """Analyzes transaction flows and patterns"""
    
    def __init__(self, asset: str = "BTC"):
        self.asset = asset.upper()
        self.large_transaction_threshold = 1000000  # $1M USD
        
    def analyze_transaction_flow(self, transaction_data: List[Dict]) -> TransactionFlowResult:
        """Analyze transaction flow patterns
        
        Args:
            transaction_data: List of transactions with 'amount', 'timestamp', 'from_address', 'to_address'
        """
        if not transaction_data:
            return self._create_empty_flow_result()
        
        # Basic metrics
        amounts = [tx.get('amount', 0) for tx in transaction_data]
        timestamps = [tx.get('timestamp', datetime.now()) for tx in transaction_data]
        from_addresses = [tx.get('from_address', '') for tx in transaction_data]
        to_addresses = [tx.get('to_address', '') for tx in transaction_data]
        
        total_volume = sum(amounts)
        transaction_count = len(transaction_data)
        unique_addresses = len(set(from_addresses + to_addresses))
        average_transaction_size = total_volume / transaction_count if transaction_count > 0 else 0
        
        # Large transaction analysis
        large_transactions = [amount for amount in amounts if amount >= self.large_transaction_threshold]
        large_transaction_ratio = len(large_transactions) / transaction_count if transaction_count > 0 else 0
        
        # Flow concentration (Gini coefficient approximation)
        flow_concentration = self._calculate_gini_coefficient(amounts)
        
        # Velocity calculation (simplified)
        velocity = self._calculate_velocity(transaction_data)
        
        # Circulation ratio
        circulation_ratio = self._calculate_circulation_ratio(transaction_data)
        
        # Dormancy flow
        dormancy_flow = self._calculate_dormancy_flow(transaction_data)
        
        # Exchange flow ratio
        exchange_flow_ratio = self._calculate_exchange_flow_ratio(transaction_data)
        
        # Flow metrics over time
        flow_metrics = self._calculate_flow_metrics_over_time(transaction_data)
        
        return TransactionFlowResult(
            total_volume=total_volume,
            unique_addresses=unique_addresses,
            transaction_count=transaction_count,
            average_transaction_size=average_transaction_size,
            large_transaction_ratio=large_transaction_ratio,
            flow_concentration=flow_concentration,
            velocity=velocity,
            circulation_ratio=circulation_ratio,
            dormancy_flow=dormancy_flow,
            exchange_flow_ratio=exchange_flow_ratio,
            timestamps=timestamps,
            flow_metrics=flow_metrics
        )
    
    def _calculate_gini_coefficient(self, amounts: List[float]) -> float:
        """Calculate Gini coefficient for transaction amount distribution"""
        if not amounts or len(amounts) < 2:
            return 0.0
        
        sorted_amounts = sorted(amounts)
        n = len(sorted_amounts)
        cumsum = np.cumsum(sorted_amounts)
        
        return (n + 1 - 2 * sum((n + 1 - i) * amount for i, amount in enumerate(sorted_amounts, 1))) / (n * sum(sorted_amounts))
    
    def _calculate_velocity(self, transaction_data: List[Dict]) -> float:
        """Calculate transaction velocity (simplified)"""
        if not transaction_data:
            return 0.0
        
        # Group transactions by day
        daily_volumes = defaultdict(float)
        for tx in transaction_data:
            date = tx.get('timestamp', datetime.now()).date()
            daily_volumes[date] += tx.get('amount', 0)
        
        if not daily_volumes:
            return 0.0
        
        # Average daily volume
        avg_daily_volume = sum(daily_volumes.values()) / len(daily_volumes)
        
        # Simplified velocity calculation
        return avg_daily_volume / 1000000  # Normalize
    
    def _calculate_circulation_ratio(self, transaction_data: List[Dict]) -> float:
        """Calculate circulation ratio"""
        if not transaction_data:
            return 0.0
        
        # Count unique addresses that have been active
        active_addresses = set()
        for tx in transaction_data:
            active_addresses.add(tx.get('from_address', ''))
            active_addresses.add(tx.get('to_address', ''))
        
        # Simplified calculation
        return min(len(active_addresses) / 1000000, 1.0)  # Normalize to total possible addresses
    
    def _calculate_dormancy_flow(self, transaction_data: List[Dict]) -> float:
        """Calculate dormancy flow (coins moving after being dormant)"""
        if not transaction_data:
            return 0.0
        
        # Simplified: assume some transactions involve dormant coins
        # In practice, this would require UTXO age analysis
        dormant_threshold_days = 365
        current_time = datetime.now()
        
        dormant_flow = 0
        for tx in transaction_data:
            # Simplified dormancy check
            if hasattr(tx, 'coin_age_days') and tx.coin_age_days > dormant_threshold_days:
                dormant_flow += tx.get('amount', 0)
        
        total_flow = sum(tx.get('amount', 0) for tx in transaction_data)
        return dormant_flow / total_flow if total_flow > 0 else 0.0
    
    def _calculate_exchange_flow_ratio(self, transaction_data: List[Dict]) -> float:
        """Calculate ratio of transactions involving exchanges"""
        if not transaction_data:
            return 0.0
        
        # Known exchange address patterns (simplified)
        exchange_patterns = ['exchange', 'binance', 'coinbase', 'kraken', 'bitfinex']
        
        exchange_transactions = 0
        for tx in transaction_data:
            from_addr = tx.get('from_address', '').lower()
            to_addr = tx.get('to_address', '').lower()
            
            if any(pattern in from_addr or pattern in to_addr for pattern in exchange_patterns):
                exchange_transactions += 1
        
        return exchange_transactions / len(transaction_data)
    
    def _calculate_flow_metrics_over_time(self, transaction_data: List[Dict]) -> List[float]:
        """Calculate flow metrics over time"""
        if not transaction_data:
            return []
        
        # Group by hour and calculate metrics
        hourly_metrics = []
        sorted_txs = sorted(transaction_data, key=lambda x: x.get('timestamp', datetime.now()))
        
        current_hour_txs = []
        current_hour = None
        
        for tx in sorted_txs:
            tx_hour = tx.get('timestamp', datetime.now()).replace(minute=0, second=0, microsecond=0)
            
            if current_hour is None:
                current_hour = tx_hour
            
            if tx_hour == current_hour:
                current_hour_txs.append(tx)
            else:
                # Calculate metric for current hour
                if current_hour_txs:
                    hour_volume = sum(tx.get('amount', 0) for tx in current_hour_txs)
                    hourly_metrics.append(hour_volume)
                
                current_hour = tx_hour
                current_hour_txs = [tx]
        
        # Add last hour
        if current_hour_txs:
            hour_volume = sum(tx.get('amount', 0) for tx in current_hour_txs)
            hourly_metrics.append(hour_volume)
        
        return hourly_metrics
    
    def _create_empty_flow_result(self) -> TransactionFlowResult:
        """Create empty flow result for edge cases"""
        return TransactionFlowResult(
            total_volume=0.0,
            unique_addresses=0,
            transaction_count=0,
            average_transaction_size=0.0,
            large_transaction_ratio=0.0,
            flow_concentration=0.0,
            velocity=0.0,
            circulation_ratio=0.0,
            dormancy_flow=0.0,
            exchange_flow_ratio=0.0,
            timestamps=[],
            flow_metrics=[]
        )

class AddressClusteringAnalyzer:
    """Analyzes and clusters blockchain addresses"""
    
    def __init__(self, asset: str = "BTC"):
        self.asset = asset.upper()
        self.whale_threshold = 1000  # 1000 BTC or equivalent
        
    def analyze_address_clustering(self, address_data: List[Dict]) -> AddressClusteringResult:
        """Analyze and cluster addresses
        
        Args:
            address_data: List of addresses with 'address', 'balance', 'transaction_count', 'labels'
        """
        if not address_data:
            return self._create_empty_clustering_result()
        
        # Extract features for clustering
        features = self._extract_address_features(address_data)
        
        # Perform clustering
        if SKLEARN_AVAILABLE and len(address_data) > 10:
            clusters = self._perform_ml_clustering(features)
        else:
            clusters = self._perform_rule_based_clustering(address_data)
        
        # Classify addresses
        classification = self._classify_addresses(address_data, clusters)
        
        # Calculate confidence scores
        confidence_scores = self._calculate_classification_confidence(address_data, classification)
        
        # Calculate risk scores
        risk_scores = self._calculate_address_risk_scores(address_data)
        
        # Calculate clustering quality
        clustering_quality = self._calculate_clustering_quality(features, clusters)
        
        return AddressClusteringResult(
            exchange_addresses=classification.get('exchange', []),
            whale_addresses=classification.get('whale', []),
            miner_addresses=classification.get('miner', []),
            defi_addresses=classification.get('defi', []),
            institutional_addresses=classification.get('institutional', []),
            retail_addresses=classification.get('retail', []),
            cluster_confidence=confidence_scores,
            address_risk_scores=risk_scores,
            clustering_quality=clustering_quality,
            total_clusters=len(set(clusters)) if clusters else 0
        )
    
    def _extract_address_features(self, address_data: List[Dict]) -> np.ndarray:
        """Extract features for address clustering"""
        features = []
        
        for addr_info in address_data:
            balance = addr_info.get('balance', 0)
            tx_count = addr_info.get('transaction_count', 0)
            
            # Feature vector: [log_balance, log_tx_count, balance_to_tx_ratio]
            log_balance = np.log1p(balance)
            log_tx_count = np.log1p(tx_count)
            balance_to_tx_ratio = balance / max(tx_count, 1)
            
            features.append([log_balance, log_tx_count, balance_to_tx_ratio])
        
        return np.array(features)
    
    def _perform_ml_clustering(self, features: np.ndarray) -> List[int]:
        """Perform ML-based clustering"""
        try:
            # Normalize features
            scaler = StandardScaler()
            normalized_features = scaler.fit_transform(features)
            
            # Use DBSCAN for clustering
            clustering = DBSCAN(eps=0.5, min_samples=5)
            clusters = clustering.fit_predict(normalized_features)
            
            return clusters.tolist()
        except Exception as e:
            logger.warning(f"ML clustering failed: {e}. Using rule-based clustering.")
            return list(range(len(features)))
    
    def _perform_rule_based_clustering(self, address_data: List[Dict]) -> List[int]:
        """Perform rule-based clustering"""
        clusters = []
        
        for addr_info in address_data:
            balance = addr_info.get('balance', 0)
            tx_count = addr_info.get('transaction_count', 0)
            labels = addr_info.get('labels', [])
            
            # Rule-based classification
            if balance >= self.whale_threshold:
                cluster = 0  # Whale cluster
            elif tx_count > 10000:
                cluster = 1  # High activity cluster (likely exchange)
            elif any('exchange' in label.lower() for label in labels):
                cluster = 1  # Exchange cluster
            elif any('miner' in label.lower() for label in labels):
                cluster = 2  # Miner cluster
            elif balance > 100:
                cluster = 3  # Institutional cluster
            else:
                cluster = 4  # Retail cluster
            
            clusters.append(cluster)
        
        return clusters
    
    def _classify_addresses(self, address_data: List[Dict], clusters: List[int]) -> Dict[str, List[str]]:
        """Classify addresses based on clusters"""
        classification = defaultdict(list)
        
        for i, addr_info in enumerate(address_data):
            address = addr_info.get('address', '')
            cluster = clusters[i] if i < len(clusters) else -1
            balance = addr_info.get('balance', 0)
            tx_count = addr_info.get('transaction_count', 0)
            labels = addr_info.get('labels', [])
            
            # Classify based on cluster and additional rules
            if balance >= self.whale_threshold:
                classification['whale'].append(address)
            elif any('exchange' in label.lower() for label in labels) or tx_count > 10000:
                classification['exchange'].append(address)
            elif any('miner' in label.lower() for label in labels):
                classification['miner'].append(address)
            elif any('defi' in label.lower() or 'uniswap' in label.lower() for label in labels):
                classification['defi'].append(address)
            elif balance > 100 or any('institutional' in label.lower() for label in labels):
                classification['institutional'].append(address)
            else:
                classification['retail'].append(address)
        
        return dict(classification)
    
    def _calculate_classification_confidence(self, address_data: List[Dict], 
                                           classification: Dict[str, List[str]]) -> Dict[str, float]:
        """Calculate confidence scores for classification"""
        confidence_scores = {}
        
        for category, addresses in classification.items():
            if not addresses:
                confidence_scores[category] = 0.0
                continue
            
            # Calculate confidence based on feature consistency
            category_confidence = 0.8  # Base confidence
            
            # Adjust based on category-specific factors
            if category == 'whale':
                # High confidence for clear whale addresses
                category_confidence = 0.9
            elif category == 'exchange':
                # Medium confidence for exchange detection
                category_confidence = 0.7
            elif category == 'retail':
                # Lower confidence for retail (catch-all category)
                category_confidence = 0.6
            
            confidence_scores[category] = category_confidence
        
        return confidence_scores
    
    def _calculate_address_risk_scores(self, address_data: List[Dict]) -> Dict[str, float]:
        """Calculate risk scores for addresses"""
        risk_scores = {}
        
        for addr_info in address_data:
            address = addr_info.get('address', '')
            balance = addr_info.get('balance', 0)
            tx_count = addr_info.get('transaction_count', 0)
            labels = addr_info.get('labels', [])
            
            risk_score = 0.0
            
            # High balance risk
            if balance >= self.whale_threshold:
                risk_score += 0.3
            
            # High activity risk
            if tx_count > 10000:
                risk_score += 0.2
            
            # Label-based risk
            risky_labels = ['mixer', 'tumbler', 'darknet', 'suspicious']
            if any(risky_label in ' '.join(labels).lower() for risky_label in risky_labels):
                risk_score += 0.5
            
            # Exchange addresses have medium risk
            if any('exchange' in label.lower() for label in labels):
                risk_score += 0.1
            
            risk_scores[address] = min(risk_score, 1.0)
        
        return risk_scores
    
    def _calculate_clustering_quality(self, features: np.ndarray, clusters: List[int]) -> float:
        """Calculate clustering quality score"""
        if len(set(clusters)) < 2:
            return 0.0
        
        try:
            if SKLEARN_AVAILABLE:
                from sklearn.metrics import silhouette_score
                return silhouette_score(features, clusters)
            else:
                # Simplified quality measure
                return 0.7  # Assume reasonable quality
        except:
            return 0.5
    
    def _create_empty_clustering_result(self) -> AddressClusteringResult:
        """Create empty clustering result for edge cases"""
        return AddressClusteringResult(
            exchange_addresses=[],
            whale_addresses=[],
            miner_addresses=[],
            defi_addresses=[],
            institutional_addresses=[],
            retail_addresses=[],
            cluster_confidence={},
            address_risk_scores={},
            clustering_quality=0.0,
            total_clusters=0
        )

class WhaleMovementDetector:
    """Detects and analyzes whale movements"""
    
    def __init__(self, asset: str = "BTC"):
        self.asset = asset.upper()
        self.whale_threshold = 1000  # 1000 BTC or equivalent
        self.large_tx_threshold = 100  # 100 BTC or equivalent
        
    def detect_whale_movements(self, whale_data: List[Dict]) -> WhaleMovementResult:
        """Detect and analyze whale movements
        
        Args:
            whale_data: List of whale transactions and holdings
        """
        if not whale_data:
            return self._create_empty_whale_result()
        
        # Filter whale transactions
        whale_transactions = self._filter_whale_transactions(whale_data)
        
        # Calculate accumulation/distribution scores
        accumulation_score = self._calculate_accumulation_score(whale_transactions)
        distribution_score = self._calculate_distribution_score(whale_transactions)
        
        # Determine activity trend
        activity_trend = self._determine_activity_trend(accumulation_score, distribution_score)
        
        # Calculate concentration
        concentration = self._calculate_whale_concentration(whale_data)
        
        # Analyze exchange flows
        exchange_flows = self._analyze_whale_exchange_flows(whale_transactions)
        
        # Detect dormant whale awakening
        dormant_awakenings = self._detect_dormant_whale_awakening(whale_transactions)
        
        # Calculate sentiment indicator
        sentiment_indicator = self._calculate_whale_sentiment(accumulation_score, distribution_score)
        
        # Calculate market impact probability
        impact_probability = self._calculate_market_impact_probability(whale_transactions)
        
        return WhaleMovementResult(
            whale_transactions=whale_transactions,
            whale_accumulation_score=accumulation_score,
            whale_distribution_score=distribution_score,
            whale_activity_trend=activity_trend,
            large_holder_concentration=concentration,
            whale_exchange_flows=exchange_flows,
            dormant_whale_awakening=dormant_awakenings,
            whale_sentiment_indicator=sentiment_indicator,
            market_impact_probability=impact_probability
        )
    
    def _filter_whale_transactions(self, whale_data: List[Dict]) -> List[Dict]:
        """Filter transactions involving whales"""
        whale_txs = []
        
        for data in whale_data:
            amount = data.get('amount', 0)
            if amount >= self.large_tx_threshold:
                whale_txs.append(data)
        
        return whale_txs
    
    def _calculate_accumulation_score(self, whale_transactions: List[Dict]) -> float:
        """Calculate whale accumulation score"""
        if not whale_transactions:
            return 0.0
        
        accumulation_volume = 0
        total_volume = 0
        
        for tx in whale_transactions:
            amount = tx.get('amount', 0)
            tx_type = tx.get('type', 'unknown')  # 'buy', 'sell', 'transfer'
            
            total_volume += amount
            if tx_type == 'buy' or (tx_type == 'transfer' and tx.get('to_exchange', False) == False):
                accumulation_volume += amount
        
        return accumulation_volume / total_volume if total_volume > 0 else 0.0
    
    def _calculate_distribution_score(self, whale_transactions: List[Dict]) -> float:
        """Calculate whale distribution score"""
        if not whale_transactions:
            return 0.0
        
        distribution_volume = 0
        total_volume = 0
        
        for tx in whale_transactions:
            amount = tx.get('amount', 0)
            tx_type = tx.get('type', 'unknown')
            
            total_volume += amount
            if tx_type == 'sell' or (tx_type == 'transfer' and tx.get('to_exchange', False) == True):
                distribution_volume += amount
        
        return distribution_volume / total_volume if total_volume > 0 else 0.0
    
    def _determine_activity_trend(self, accumulation_score: float, distribution_score: float) -> str:
        """Determine whale activity trend"""
        if accumulation_score > distribution_score + 0.2:
            return "Strong Accumulation"
        elif accumulation_score > distribution_score + 0.1:
            return "Moderate Accumulation"
        elif distribution_score > accumulation_score + 0.2:
            return "Strong Distribution"
        elif distribution_score > accumulation_score + 0.1:
            return "Moderate Distribution"
        else:
            return "Neutral Activity"
    
    def _calculate_whale_concentration(self, whale_data: List[Dict]) -> float:
        """Calculate whale concentration (Gini coefficient for large holders)"""
        balances = [data.get('balance', 0) for data in whale_data if data.get('balance', 0) >= self.whale_threshold]
        
        if len(balances) < 2:
            return 0.0
        
        sorted_balances = sorted(balances)
        n = len(sorted_balances)
        cumsum = np.cumsum(sorted_balances)
        
        return (n + 1 - 2 * sum((n + 1 - i) * balance for i, balance in enumerate(sorted_balances, 1))) / (n * sum(sorted_balances))
    
    def _analyze_whale_exchange_flows(self, whale_transactions: List[Dict]) -> Dict[str, float]:
        """Analyze whale flows to/from exchanges"""
        exchange_flows = {'inflow': 0.0, 'outflow': 0.0, 'net_flow': 0.0}
        
        for tx in whale_transactions:
            amount = tx.get('amount', 0)
            to_exchange = tx.get('to_exchange', False)
            from_exchange = tx.get('from_exchange', False)
            
            if to_exchange:
                exchange_flows['inflow'] += amount
            elif from_exchange:
                exchange_flows['outflow'] += amount
        
        exchange_flows['net_flow'] = exchange_flows['inflow'] - exchange_flows['outflow']
        
        return exchange_flows
    
    def _detect_dormant_whale_awakening(self, whale_transactions: List[Dict]) -> List[Dict]:
        """Detect dormant whale addresses becoming active"""
        awakenings = []
        
        for tx in whale_transactions:
            # Check if this is from a previously dormant address
            dormancy_days = tx.get('address_dormancy_days', 0)
            amount = tx.get('amount', 0)
            
            if dormancy_days > 365 and amount >= self.large_tx_threshold:  # Dormant for over a year
                awakenings.append({
                    'address': tx.get('from_address', ''),
                    'amount': amount,
                    'dormancy_days': dormancy_days,
                    'timestamp': tx.get('timestamp', datetime.now())
                })
        
        return awakenings
    
    def _calculate_whale_sentiment(self, accumulation_score: float, distribution_score: float) -> float:
        """Calculate whale sentiment indicator (-1 to 1)"""
        return accumulation_score - distribution_score
    
    def _calculate_market_impact_probability(self, whale_transactions: List[Dict]) -> float:
        """Calculate probability of market impact from whale activity"""
        if not whale_transactions:
            return 0.0
        
        # Calculate recent large transaction volume
        recent_threshold = datetime.now() - timedelta(hours=24)
        recent_volume = sum(
            tx.get('amount', 0) for tx in whale_transactions 
            if tx.get('timestamp', datetime.now()) > recent_threshold
        )
        
        # Normalize to probability (simplified)
        impact_probability = min(recent_volume / 10000, 1.0)  # 10k BTC = 100% probability
        
        return impact_probability
    
    def _create_empty_whale_result(self) -> WhaleMovementResult:
        """Create empty whale result for edge cases"""
        return WhaleMovementResult(
            whale_transactions=[],
            whale_accumulation_score=0.0,
            whale_distribution_score=0.0,
            whale_activity_trend="No Activity",
            large_holder_concentration=0.0,
            whale_exchange_flows={'inflow': 0.0, 'outflow': 0.0, 'net_flow': 0.0},
            dormant_whale_awakening=[],
            whale_sentiment_indicator=0.0,
            market_impact_probability=0.0
        )

class OnChainMLModel:
    """Combined On-chain ML Analysis Model"""
    
    def __init__(self, asset: str = "BTC"):
        self.asset = asset.upper()
        self.transaction_analyzer = TransactionFlowAnalyzer(asset)
        self.address_analyzer = AddressClusteringAnalyzer(asset)
        self.whale_detector = WhaleMovementDetector(asset)
        
    def analyze(self, 
               transaction_data: List[Dict] = None,
               address_data: List[Dict] = None,
               whale_data: List[Dict] = None,
               network_data: Dict = None) -> OnChainMLResult:
        """Perform comprehensive on-chain ML analysis
        
        Args:
            transaction_data: List of transaction records
            address_data: List of address information
            whale_data: List of whale-related data
            network_data: Network-level metrics
        """
        try:
            # Analyze transaction flows
            transaction_flow = self.transaction_analyzer.analyze_transaction_flow(transaction_data or [])
            
            # Analyze address clustering
            address_clustering = self.address_analyzer.analyze_address_clustering(address_data or [])
            
            # Detect whale movements
            whale_movement = self.whale_detector.detect_whale_movements(whale_data or [])
            
            # Analyze network health
            network_health = self._analyze_network_health(network_data or {})
            
            # Detect anomalies
            anomaly_detection = self._detect_anomalies(transaction_data or [], address_data or [])
            
            # Identify market cycles
            market_cycle = self._identify_market_cycle(transaction_flow, whale_movement)
            
            # Analyze liquidity
            liquidity_analysis = self._analyze_liquidity(whale_data or [], transaction_data or [])
            
            # Predict price movement
            price_prediction = self._predict_price_movement(
                transaction_flow, whale_movement, network_health
            )
            
            # Calculate overall scores
            confidence_score = self._calculate_confidence_score(
                transaction_flow, address_clustering, whale_movement
            )
            
            overall_network_score = self._calculate_overall_network_score(
                network_health, transaction_flow, whale_movement
            )
            
            return OnChainMLResult(
                transaction_flow=transaction_flow,
                address_clustering=address_clustering,
                whale_movement=whale_movement,
                network_health=network_health,
                anomaly_detection=anomaly_detection,
                market_cycle=market_cycle,
                liquidity_analysis=liquidity_analysis,
                price_prediction=price_prediction,
                confidence_score=confidence_score,
                overall_network_score=overall_network_score
            )
            
        except Exception as e:
            logger.error(f"Error in on-chain ML analysis: {str(e)}")
            raise
    
    def _analyze_network_health(self, network_data: Dict) -> NetworkHealthResult:
        """Analyze network health metrics"""
        active_addresses = network_data.get('active_addresses', 0)
        new_addresses = network_data.get('new_addresses', 0)
        transaction_count = network_data.get('transaction_count', 0)
        hash_rate = network_data.get('hash_rate', 0)
        
        # Calculate derived metrics
        address_growth_rate = new_addresses / max(active_addresses, 1) * 100
        transaction_throughput = transaction_count / 24  # Per hour
        network_utilization = min(transaction_count / 1000000, 1.0)  # Normalize
        fee_pressure = network_data.get('average_fee', 0) / 100  # Normalize
        congestion_score = min(fee_pressure * 2, 1.0)
        
        # Simplified calculations for other metrics
        decentralization_index = 0.8  # Assume good decentralization
        network_security_score = min(hash_rate / 100000000, 1.0)  # Normalize hash rate
        adoption_momentum = address_growth_rate / 10  # Normalize
        
        return NetworkHealthResult(
            active_addresses=active_addresses,
            new_addresses=new_addresses,
            address_growth_rate=address_growth_rate,
            transaction_throughput=transaction_throughput,
            network_utilization=network_utilization,
            fee_pressure=fee_pressure,
            congestion_score=congestion_score,
            decentralization_index=decentralization_index,
            network_security_score=network_security_score,
            adoption_momentum=adoption_momentum
        )
    
    def _detect_anomalies(self, transaction_data: List[Dict], address_data: List[Dict]) -> AnomalyDetectionResult:
        """Detect anomalies in on-chain data"""
        anomalous_transactions = []
        anomaly_score = 0.0
        suspicious_patterns = []
        
        # Detect unusually large transactions
        if transaction_data:
            amounts = [tx.get('amount', 0) for tx in transaction_data]
            if amounts:
                mean_amount = np.mean(amounts)
                std_amount = np.std(amounts)
                threshold = mean_amount + 3 * std_amount
                
                for tx in transaction_data:
                    if tx.get('amount', 0) > threshold:
                        anomalous_transactions.append(tx)
                        suspicious_patterns.append("Unusually large transaction")
        
        # Calculate anomaly score
        if transaction_data:
            anomaly_score = len(anomalous_transactions) / len(transaction_data)
        
        # Determine risk level
        if anomaly_score > 0.1:
            risk_level = "High"
        elif anomaly_score > 0.05:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        return AnomalyDetectionResult(
            anomalous_transactions=anomalous_transactions,
            anomaly_score=anomaly_score,
            suspicious_patterns=list(set(suspicious_patterns)),
            risk_level=risk_level,
            confidence_score=0.7,
            anomaly_types={'large_transactions': len(anomalous_transactions)},
            temporal_anomalies=[],
            network_anomalies=[]
        )
    
    def _identify_market_cycle(self, transaction_flow: TransactionFlowResult, 
                             whale_movement: WhaleMovementResult) -> MarketCycleResult:
        """Identify current market cycle phase"""
        # Simplified cycle identification based on whale activity and transaction flow
        whale_sentiment = whale_movement.whale_sentiment_indicator
        transaction_velocity = transaction_flow.velocity
        
        if whale_sentiment > 0.3 and transaction_velocity > 0.5:
            current_phase = "Accumulation"
            cycle_completion = 0.25
        elif whale_sentiment > 0 and transaction_velocity > 0.3:
            current_phase = "Mark-up"
            cycle_completion = 0.5
        elif whale_sentiment < -0.3 and transaction_velocity > 0.7:
            current_phase = "Distribution"
            cycle_completion = 0.75
        else:
            current_phase = "Mark-down"
            cycle_completion = 0.9
        
        return MarketCycleResult(
            current_cycle_phase=current_phase,
            cycle_completion=cycle_completion,
            cycle_duration_estimate=1460,  # ~4 years
            next_phase_prediction="TBD",
            cycle_strength=0.7,
            historical_cycles=[],
            cycle_indicators={'whale_sentiment': whale_sentiment, 'velocity': transaction_velocity},
            phase_transition_probability=0.3
        )
    
    def _analyze_liquidity(self, whale_data: List[Dict], transaction_data: List[Dict]) -> LiquidityAnalysisResult:
        """Analyze market liquidity"""
        # Simplified liquidity analysis
        total_supply = 21000000  # Bitcoin total supply
        
        # Calculate liquid vs illiquid supply
        whale_holdings = sum(data.get('balance', 0) for data in whale_data)
        liquid_supply = total_supply - whale_holdings
        illiquid_supply = whale_holdings
        
        liquidity_ratio = liquid_supply / total_supply if total_supply > 0 else 0
        
        # Supply shock risk
        supply_shock_risk = illiquid_supply / total_supply if total_supply > 0 else 0
        
        # Simplified metrics
        exchange_reserves = liquid_supply * 0.1  # Assume 10% on exchanges
        staking_ratio = 0.0  # Not applicable for Bitcoin
        long_term_holder_ratio = illiquid_supply / total_supply
        
        return LiquidityAnalysisResult(
            liquid_supply=liquid_supply,
            illiquid_supply=illiquid_supply,
            liquidity_ratio=liquidity_ratio,
            supply_shock_risk=supply_shock_risk,
            liquidity_trend="Stable",
            exchange_reserves=exchange_reserves,
            staking_ratio=staking_ratio,
            long_term_holder_ratio=long_term_holder_ratio,
            supply_distribution={'liquid': liquidity_ratio, 'illiquid': 1 - liquidity_ratio}
        )
    
    def _predict_price_movement(self, transaction_flow: TransactionFlowResult,
                              whale_movement: WhaleMovementResult,
                              network_health: NetworkHealthResult) -> float:
        """Predict price movement based on on-chain data"""
        # Simplified price prediction model
        factors = [
            transaction_flow.velocity * 0.3,
            whale_movement.whale_sentiment_indicator * 0.4,
            network_health.adoption_momentum * 0.3
        ]
        
        prediction_score = sum(factors)
        
        # Convert to percentage price change
        return prediction_score * 20  # Scale to reasonable percentage
    
    def _calculate_confidence_score(self, transaction_flow: TransactionFlowResult,
                                  address_clustering: AddressClusteringResult,
                                  whale_movement: WhaleMovementResult) -> float:
        """Calculate overall confidence score"""
        confidence_factors = [
            min(transaction_flow.transaction_count / 1000, 1.0),  # Data volume
            address_clustering.clustering_quality,
            min(len(whale_movement.whale_transactions) / 100, 1.0)  # Whale data quality
        ]
        
        return np.mean(confidence_factors)
    
    def _calculate_overall_network_score(self, network_health: NetworkHealthResult,
                                       transaction_flow: TransactionFlowResult,
                                       whale_movement: WhaleMovementResult) -> float:
        """Calculate overall network health score"""
        score_components = [
            network_health.network_security_score * 0.3,
            network_health.decentralization_index * 0.2,
            min(transaction_flow.velocity, 1.0) * 0.2,
            (1 - abs(whale_movement.whale_sentiment_indicator)) * 0.3  # Stability bonus
        ]
        
        return sum(score_components)
    
    def get_onchain_insights(self, result: OnChainMLResult) -> Dict[str, str]:
        """Generate comprehensive on-chain insights"""
        insights = {}
        
        # Transaction flow insights
        insights['transaction_flow'] = f"Volume: {result.transaction_flow.total_volume:,.0f}, Velocity: {result.transaction_flow.velocity:.2f}, Concentration: {result.transaction_flow.flow_concentration:.2f}"
        
        # Whale activity insights
        insights['whale_activity'] = f"Trend: {result.whale_movement.whale_activity_trend}, Sentiment: {result.whale_movement.whale_sentiment_indicator:+.2f}, Impact Risk: {result.whale_movement.market_impact_probability:.1%}"
        
        # Network health insights
        insights['network_health'] = f"Active Addresses: {result.network_health.active_addresses:,}, Growth: {result.network_health.address_growth_rate:+.1f}%, Security: {result.network_health.network_security_score:.1%}"
        
        # Market cycle insights
        insights['market_cycle'] = f"Phase: {result.market_cycle.current_cycle_phase}, Completion: {result.market_cycle.cycle_completion:.1%}, Strength: {result.market_cycle.cycle_strength:.1%}"
        
        # Liquidity insights
        insights['liquidity'] = f"Liquid Supply: {result.liquidity_analysis.liquidity_ratio:.1%}, LTH Ratio: {result.liquidity_analysis.long_term_holder_ratio:.1%}, Shock Risk: {result.liquidity_analysis.supply_shock_risk:.1%}"
        
        # Anomaly insights
        insights['anomalies'] = f"Risk Level: {result.anomaly_detection.risk_level}, Score: {result.anomaly_detection.anomaly_score:.1%}, Suspicious Transactions: {len(result.anomaly_detection.anomalous_transactions)}"
        
        # Overall insights
        insights['price_prediction'] = f"Predicted Price Movement: {result.price_prediction:+.1f}%"
        insights['confidence'] = f"Analysis Confidence: {result.confidence_score:.1%}"
        insights['network_score'] = f"Overall Network Score: {result.overall_network_score:.1%}"
        
        return insights

# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    sample_transactions = [
        {
            'amount': 1500,
            'timestamp': datetime.now() - timedelta(hours=1),
            'from_address': 'whale_address_1',
            'to_address': 'exchange_address_1',
            'type': 'transfer',
            'to_exchange': True
        },
        {
            'amount': 50,
            'timestamp': datetime.now() - timedelta(hours=2),
            'from_address': 'retail_address_1',
            'to_address': 'retail_address_2',
            'type': 'transfer',
            'to_exchange': False
        }
    ]
    
    sample_addresses = [
        {
            'address': 'whale_address_1',
            'balance': 5000,
            'transaction_count': 150,
            'labels': ['whale', 'early_adopter']
        },
        {
            'address': 'exchange_address_1',
            'balance': 50000,
            'transaction_count': 100000,
            'labels': ['exchange', 'binance']
        }
    ]
    
    sample_whales = [
        {
            'address': 'whale_address_1',
            'balance': 5000,
            'amount': 1500,
            'type': 'sell',
            'timestamp': datetime.now(),
            'to_exchange': True,
            'address_dormancy_days': 30
        }
    ]
    
    sample_network = {
        'active_addresses': 1000000,
        'new_addresses': 5000,
        'transaction_count': 300000,
        'hash_rate': 150000000,
        'average_fee': 25
    }
    
    # Test the model
    onchain_model = OnChainMLModel("BTC")
    result = onchain_model.analyze(
        transaction_data=sample_transactions,
        address_data=sample_addresses,
        whale_data=sample_whales,
        network_data=sample_network
    )
    
    insights = onchain_model.get_onchain_insights(result)
    
    print("=== On-chain ML Analysis ===")
    print(f"Price Prediction: {result.price_prediction:+.1f}%")
    print(f"Network Score: {result.overall_network_score:.1%}")
    print(f"Confidence: {result.confidence_score:.1%}")
    print()
    
    print("=== Detailed Insights ===")
    for key, insight in insights.items():
        print(f"{key}: {insight}")