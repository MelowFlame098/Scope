import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import aiohttp
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import networkx as nx
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Import modularized crypto models
from stock_to_flow_model import StockToFlowModel
from metcalfes_law_model import MetcalfesLawModel
from nvt_model import NVTModel
from crypto_quant_metrics import CryptoQuantMetrics
from logarithmic_regression_model import LogarithmicRegressionModel
from crypto_bert_sentiment import CryptoBERTSentiment
from onchain_ml_model import OnChainMLModel
from graph_neural_network import GraphNeuralNetwork
from reinforcement_learning_agent import ReinforcementLearningAgent

logger = logging.getLogger(__name__)

@dataclass
class CryptoIndicatorResult:
    """Result container for crypto indicator calculations"""
    indicator_name: str
    value: float
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any]
    signal: str  # 'buy', 'sell', 'hold'
    strength: float  # 0-1 signal strength

# StockToFlowModel has been moved to stock_to_flow_model.py

# MetcalfesLawModel has been moved to metcalfes_law_model.py

# NVTModel has been moved to nvt_model.py

# CryptoQuantMetrics has been moved to crypto_quant_metrics.py

# LogarithmicRegressionModel has been moved to logarithmic_regression_model.py

# CryptoBERTSentiment has been moved to crypto_bert_sentiment.py

# OnChainMLModel has been moved to onchain_ml_model.py

# GraphNeuralNetwork has been moved to graph_neural_network.py

# ReinforcementLearningAgent has been moved to reinforcement_learning_agent.py

class CryptoIndicatorEngine:
    """Main engine for crypto indicator calculations"""
    
    def __init__(self):
        self.s2f_model = StockToFlowModel()
        self.metcalfe_model = MetcalfesLawModel()
        self.nvt_model = NVTModel()
        self.quant_metrics = CryptoQuantMetrics()
        self.log_regression = LogarithmicRegressionModel()
        self.sentiment_analyzer = CryptoBERTSentiment()
        self.onchain_ml = OnChainMLModel()
        self.gnn = GraphNeuralNetwork()
        self.rl_agent = ReinforcementLearningAgent()
    
    async def calculate_all_indicators(self, 
                                     crypto_data: Dict[str, Any]) -> List[CryptoIndicatorResult]:
        """Calculate all crypto indicators"""
        results = []
        
        try:
            # Stock-to-Flow
            if 'current_supply' in crypto_data and 'annual_production' in crypto_data:
                s2f_result = self.s2f_model.calculate_stock_to_flow(
                    crypto_data['current_supply'],
                    crypto_data['annual_production'],
                    crypto_data.get('asset_type', 'bitcoin')
                )
                results.append(s2f_result)
            
            # Metcalfe's Law
            if 'active_addresses' in crypto_data and 'market_cap' in crypto_data:
                metcalfe_result = self.metcalfe_model.calculate_network_value(
                    crypto_data['active_addresses'],
                    crypto_data.get('transaction_count', 0),
                    crypto_data['market_cap']
                )
                results.append(metcalfe_result)
            
            # NVT Ratio
            if 'market_cap' in crypto_data and 'daily_transaction_volume' in crypto_data:
                nvt_result = self.nvt_model.calculate_nvt_ratio(
                    crypto_data['market_cap'],
                    crypto_data['daily_transaction_volume']
                )
                results.append(nvt_result)
            
            # NVM Ratio
            if 'market_cap' in crypto_data and 'active_addresses' in crypto_data:
                nvm_result = self.nvt_model.calculate_nvm_ratio(
                    crypto_data['market_cap'],
                    crypto_data['active_addresses']
                )
                results.append(nvm_result)
            
            # MVRV Ratio
            if 'market_cap' in crypto_data and 'realized_cap' in crypto_data:
                mvrv_result = self.quant_metrics.calculate_mvrv_ratio(
                    crypto_data['market_cap'],
                    crypto_data['realized_cap']
                )
                results.append(mvrv_result)
            
            # SOPR
            if 'spent_outputs_profit' in crypto_data and 'spent_outputs_loss' in crypto_data:
                sopr_result = self.quant_metrics.calculate_sopr(
                    crypto_data['spent_outputs_profit'],
                    crypto_data['spent_outputs_loss']
                )
                results.append(sopr_result)
            
            # Logarithmic Regression
            if 'price_history' in crypto_data and 'timestamp_history' in crypto_data:
                log_reg_result = self.log_regression.calculate_log_regression(
                    crypto_data['price_history'],
                    crypto_data['timestamp_history']
                )
                results.append(log_reg_result)
            
            # Sentiment Analysis
            if 'news_texts' in crypto_data:
                sentiment_result = self.sentiment_analyzer.analyze_sentiment(
                    crypto_data['news_texts'],
                    crypto_data.get('news_sources')
                )
                results.append(sentiment_result)
            
            # On-chain ML
            if 'onchain_features' in crypto_data:
                ml_result = self.onchain_ml.predict(
                    crypto_data['onchain_features']
                )
                results.append(ml_result)
            
            # Graph Neural Network
            if 'transactions' in crypto_data:
                gnn_result = self.gnn.build_transaction_graph(
                    crypto_data['transactions']
                )
                results.append(gnn_result)
            
            # Reinforcement Learning
            if 'state_features' in crypto_data:
                rl_result = self.rl_agent.get_action(
                    crypto_data['state_features']
                )
                results.append(rl_result)
            
        except Exception as e:
            logger.error(f"Error calculating crypto indicators: {e}")
        
        return results
    
    def get_available_indicators(self) -> List[str]:
        """Get list of available crypto indicators"""
        return [
            'Stock-to-Flow',
            'Metcalfes Law',
            'NVT Ratio',
            'NVM Ratio',
            'MVRV Ratio',
            'SOPR',
            'Logarithmic Regression',
            'CryptoBERT Sentiment',
            'On-chain ML',
            'Graph Neural Network',
            'RL Agent (PPO)'
        ]

# Example usage
if __name__ == "__main__":
    # Demo data
    demo_data = {
        'current_supply': 19500000,  # Bitcoin supply
        'annual_production': 328500,  # Bitcoin annual production
        'asset_type': 'bitcoin',
        'active_addresses': 1000000,
        'market_cap': 800000000000,  # $800B
        'transaction_count': 300000,
        'daily_transaction_volume': 15000000000,  # $15B
        'realized_cap': 600000000000,  # $600B
        'spent_outputs_profit': 8000000000,
        'spent_outputs_loss': 2000000000,
        'price_history': [30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000],
        'timestamp_history': [datetime.now() - timedelta(days=i*30) for i in range(8, 0, -1)],
        'news_texts': ['Bitcoin adoption increasing', 'Institutional investment growing'],
        'onchain_features': {
            'hash_rate': 200000000,
            'difficulty': 25000000000000,
            'mempool_size': 50000000,
            'fee_rate': 20
        },
        'transactions': [
            {'from_address': 'addr1', 'to_address': 'addr2', 'amount': 1.5, 'timestamp': datetime.now()},
            {'from_address': 'addr2', 'to_address': 'addr3', 'amount': 0.8, 'timestamp': datetime.now()}
        ],
        'state_features': {
            'price': 50000,
            'volume': 15000000000,
            'volatility': 0.04,
            'rsi': 65,
            'macd': 0.02
        }
    }
    
    # Initialize engine
    engine = CryptoIndicatorEngine()
    
    # Calculate indicators
    import asyncio
    results = asyncio.run(engine.calculate_all_indicators(demo_data))
    
    # Print results
    for result in results:
        print(f"{result.indicator_name}: {result.value:.4f} ({result.signal}) - Confidence: {result.confidence:.2f}")