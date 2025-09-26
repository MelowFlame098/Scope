import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from .base_model import BaseModel, ModelType, AssetCategory
import logging

class StockToFlowModel(BaseModel):
    """
    Stock-to-Flow (S2F) model for Bitcoin price prediction.
    Based on the scarcity principle - relates Bitcoin's stock (existing supply)
    to its flow (new production rate).
    """
    
    def __init__(self):
        super().__init__(
            model_id="crypto_s2f",
            name="Stock-to-Flow (S2F)",
            category=AssetCategory.CRYPTO,
            model_type=ModelType.FUNDAMENTAL,
            description="Bitcoin scarcity model based on stock-to-flow ratio"
        )
        self.halving_dates = [
            datetime(2012, 11, 28),
            datetime(2016, 7, 9),
            datetime(2020, 5, 11),
            datetime(2024, 4, 20)
        ]
    
    def prepare_data(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Prepare Bitcoin data with S2F calculations.
        """
        if not self.validate_data(data, ['timestamp', 'close', 'volume']):
            raise ValueError("Invalid data format for S2F model")
        
        df = data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate stock (total supply) - simplified approximation
        df['days_since_genesis'] = (df['timestamp'] - datetime(2009, 1, 3)).dt.days
        df['stock'] = self._calculate_bitcoin_supply(df['days_since_genesis'])
        
        # Calculate flow (annual production)
        df['flow'] = self._calculate_annual_flow(df['timestamp'])
        
        # Calculate S2F ratio
        df['s2f_ratio'] = df['stock'] / df['flow']
        df['log_s2f'] = np.log(df['s2f_ratio'])
        
        return df
    
    def _calculate_bitcoin_supply(self, days_since_genesis: pd.Series) -> pd.Series:
        """
        Calculate approximate Bitcoin supply based on halving schedule.
        """
        supply = pd.Series(index=days_since_genesis.index, dtype=float)
        
        for i, days in enumerate(days_since_genesis):
            if days <= 0:
                supply.iloc[i] = 0
            elif days <= 1458:  # First halving period (4 years)
                supply.iloc[i] = days * 50 * 144  # 50 BTC per block, ~144 blocks/day
            elif days <= 2916:  # Second halving period
                supply.iloc[i] = 1458 * 50 * 144 + (days - 1458) * 25 * 144
            elif days <= 4374:  # Third halving period
                supply.iloc[i] = 1458 * 50 * 144 + 1458 * 25 * 144 + (days - 2916) * 12.5 * 144
            else:  # Fourth halving period and beyond
                supply.iloc[i] = 1458 * 50 * 144 + 1458 * 25 * 144 + 1458 * 12.5 * 144 + (days - 4374) * 6.25 * 144
        
        return supply
    
    def _calculate_annual_flow(self, timestamps: pd.Series) -> pd.Series:
        """
        Calculate annual Bitcoin production flow.
        """
        flow = pd.Series(index=timestamps.index, dtype=float)
        
        for i, ts in enumerate(timestamps):
            if ts < self.halving_dates[0]:
                flow.iloc[i] = 50 * 144 * 365  # 50 BTC per block
            elif ts < self.halving_dates[1]:
                flow.iloc[i] = 25 * 144 * 365  # 25 BTC per block
            elif ts < self.halving_dates[2]:
                flow.iloc[i] = 12.5 * 144 * 365  # 12.5 BTC per block
            else:
                flow.iloc[i] = 6.25 * 144 * 365  # 6.25 BTC per block
        
        return flow
    
    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Train the S2F model using linear regression on log-log scale.
        """
        self.start_run()
        
        try:
            prepared_data = self.prepare_data(data)
            
            # Remove any infinite or NaN values
            prepared_data = prepared_data.replace([np.inf, -np.inf], np.nan).dropna()
            
            X = prepared_data['log_s2f'].values.reshape(-1, 1)
            y = np.log(prepared_data['close'].values)
            
            from sklearn.linear_model import LinearRegression
            from sklearn.model_selection import train_test_split
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            self.model_instance = LinearRegression()
            self.model_instance.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.model_instance.predict(X_test)
            metrics = self.calculate_common_metrics(y_test, y_pred)
            
            self.complete_run(metrics['r2'] * 100)
            
            return {
                'status': 'success',
                'metrics': metrics,
                'coefficients': {
                    'slope': self.model_instance.coef_[0],
                    'intercept': self.model_instance.intercept_
                }
            }
            
        except Exception as e:
            self.error_run(str(e))
            return {'status': 'error', 'message': str(e)}
    
    def predict(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Generate S2F price predictions.
        """
        if self.model_instance is None:
            return {'status': 'error', 'message': 'Model not trained'}
        
        try:
            prepared_data = self.prepare_data(data)
            X = prepared_data['log_s2f'].values.reshape(-1, 1)
            
            log_predictions = self.model_instance.predict(X)
            predictions = np.exp(log_predictions)
            
            # Calculate confidence intervals (simplified)
            residuals = np.std(log_predictions) * 1.96  # 95% CI
            upper_bound = np.exp(log_predictions + residuals)
            lower_bound = np.exp(log_predictions - residuals)
            
            return {
                'status': 'success',
                'predictions': predictions.tolist(),
                'upper_bound': upper_bound.tolist(),
                'lower_bound': lower_bound.tolist(),
                'timestamps': prepared_data['timestamp'].dt.isoformat().tolist()
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def evaluate(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """
        Evaluate S2F model performance.
        """
        return self.calculate_common_metrics(actual, predicted)

class MetcalfeModel(BaseModel):
    """
    Metcalfe's Law model for cryptocurrency valuation.
    Values network based on the square of active addresses.
    """
    
    def __init__(self):
        super().__init__(
            model_id="crypto_metcalfe",
            name="Metcalfe's Law",
            category=AssetCategory.CRYPTO,
            model_type=ModelType.FUNDAMENTAL,
            description="Network value analysis based on active addresses"
        )
    
    def prepare_data(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Prepare data with network metrics.
        """
        if not self.validate_data(data, ['timestamp', 'close', 'active_addresses']):
            raise ValueError("Invalid data format for Metcalfe model")
        
        df = data.copy()
        df['network_value'] = df['active_addresses'] ** 2
        df['log_network_value'] = np.log(df['network_value'])
        df['log_price'] = np.log(df['close'])
        
        return df
    
    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Train Metcalfe's Law model.
        """
        self.start_run()
        
        try:
            prepared_data = self.prepare_data(data)
            prepared_data = prepared_data.replace([np.inf, -np.inf], np.nan).dropna()
            
            X = prepared_data['log_network_value'].values.reshape(-1, 1)
            y = prepared_data['log_price'].values
            
            from sklearn.linear_model import LinearRegression
            from sklearn.model_selection import train_test_split
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            self.model_instance = LinearRegression()
            self.model_instance.fit(X_train, y_train)
            
            y_pred = self.model_instance.predict(X_test)
            metrics = self.calculate_common_metrics(y_test, y_pred)
            
            self.complete_run(metrics['r2'] * 100)
            
            return {
                'status': 'success',
                'metrics': metrics,
                'coefficients': {
                    'slope': self.model_instance.coef_[0],
                    'intercept': self.model_instance.intercept_
                }
            }
            
        except Exception as e:
            self.error_run(str(e))
            return {'status': 'error', 'message': str(e)}
    
    def predict(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Generate Metcalfe predictions.
        """
        if self.model_instance is None:
            return {'status': 'error', 'message': 'Model not trained'}
        
        try:
            prepared_data = self.prepare_data(data)
            X = prepared_data['log_network_value'].values.reshape(-1, 1)
            
            log_predictions = self.model_instance.predict(X)
            predictions = np.exp(log_predictions)
            
            return {
                'status': 'success',
                'predictions': predictions.tolist(),
                'timestamps': prepared_data['timestamp'].dt.isoformat().tolist()
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def evaluate(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        return self.calculate_common_metrics(actual, predicted)

class NVTModel(BaseModel):
    """
    Network Value to Transactions (NVT) model.
    Cryptocurrency valuation based on transaction volume.
    """
    
    def __init__(self):
        super().__init__(
            model_id="crypto_nvt",
            name="NVT / NVM",
            category=AssetCategory.CRYPTO,
            model_type=ModelType.FUNDAMENTAL,
            description="Network value to transactions ratio analysis"
        )
    
    def prepare_data(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Prepare data with NVT calculations.
        """
        if not self.validate_data(data, ['timestamp', 'close', 'transaction_volume', 'market_cap']):
            raise ValueError("Invalid data format for NVT model")
        
        df = data.copy()
        
        # Calculate NVT ratio
        df['nvt_ratio'] = df['market_cap'] / df['transaction_volume']
        
        # Calculate moving averages for smoothing
        df['nvt_ma_30'] = df['nvt_ratio'].rolling(window=30).mean()
        df['nvt_ma_90'] = df['nvt_ratio'].rolling(window=90).mean()
        
        # Calculate NVT signal (NVT with circulating supply adjustment)
        df['nvt_signal'] = df['market_cap'] / df['transaction_volume'].rolling(window=30).mean()
        
        return df
    
    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Train NVT model using statistical thresholds.
        """
        self.start_run()
        
        try:
            prepared_data = self.prepare_data(data)
            prepared_data = prepared_data.dropna()
            
            # Calculate NVT percentiles for signal generation
            nvt_values = prepared_data['nvt_ratio']
            
            self.parameters = {
                'nvt_low_threshold': np.percentile(nvt_values, 20),
                'nvt_high_threshold': np.percentile(nvt_values, 80),
                'nvt_mean': nvt_values.mean(),
                'nvt_std': nvt_values.std()
            }
            
            # Generate signals based on NVT levels
            prepared_data['signal'] = 0
            prepared_data.loc[prepared_data['nvt_ratio'] < self.parameters['nvt_low_threshold'], 'signal'] = 1  # Buy
            prepared_data.loc[prepared_data['nvt_ratio'] > self.parameters['nvt_high_threshold'], 'signal'] = -1  # Sell
            
            # Calculate accuracy based on future returns
            prepared_data['future_return'] = prepared_data['close'].pct_change(periods=30).shift(-30)
            
            buy_signals = prepared_data[prepared_data['signal'] == 1]
            sell_signals = prepared_data[prepared_data['signal'] == -1]
            
            buy_accuracy = (buy_signals['future_return'] > 0).mean() if len(buy_signals) > 0 else 0
            sell_accuracy = (sell_signals['future_return'] < 0).mean() if len(sell_signals) > 0 else 0
            
            overall_accuracy = (buy_accuracy + sell_accuracy) / 2 * 100
            
            self.complete_run(overall_accuracy)
            
            return {
                'status': 'success',
                'accuracy': overall_accuracy,
                'thresholds': self.parameters,
                'signal_distribution': prepared_data['signal'].value_counts().to_dict()
            }
            
        except Exception as e:
            self.error_run(str(e))
            return {'status': 'error', 'message': str(e)}
    
    def predict(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Generate NVT-based signals.
        """
        if not self.parameters:
            return {'status': 'error', 'message': 'Model not trained'}
        
        try:
            prepared_data = self.prepare_data(data)
            
            # Generate signals
            signals = []
            for nvt in prepared_data['nvt_ratio']:
                if pd.isna(nvt):
                    signals.append(0)
                elif nvt < self.parameters['nvt_low_threshold']:
                    signals.append(1)  # Buy signal
                elif nvt > self.parameters['nvt_high_threshold']:
                    signals.append(-1)  # Sell signal
                else:
                    signals.append(0)  # Hold
            
            return {
                'status': 'success',
                'signals': signals,
                'nvt_ratios': prepared_data['nvt_ratio'].tolist(),
                'timestamps': prepared_data['timestamp'].dt.isoformat().tolist()
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def evaluate(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        return self.calculate_common_metrics(actual, predicted)

class CryptoFinBERTModel(BaseModel):
    """
    FinBERT/CryptoBERT sentiment analysis model for crypto markets.
    """
    
    def __init__(self):
        super().__init__(
            model_id="crypto_finbert",
            name="FinBERT / CryptoBERT",
            category=AssetCategory.CRYPTO,
            model_type=ModelType.SENTIMENT,
            description="NLP sentiment analysis for crypto markets"
        )
    
    def prepare_data(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Prepare text data for sentiment analysis.
        """
        if not self.validate_data(data, ['timestamp', 'close', 'news_text']):
            raise ValueError("Invalid data format for FinBERT model")
        
        df = data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Train sentiment model (simplified implementation).
        """
        self.start_run()
        
        try:
            # Simplified sentiment analysis using TextBlob
            from textblob import TextBlob
            
            prepared_data = self.prepare_data(data)
            
            # Calculate sentiment scores
            sentiments = []
            for text in prepared_data['news_text']:
                if pd.isna(text):
                    sentiments.append(0)
                else:
                    blob = TextBlob(str(text))
                    sentiments.append(blob.sentiment.polarity)
            
            prepared_data['sentiment'] = sentiments
            
            # Calculate correlation with price movements
            prepared_data['price_change'] = prepared_data['close'].pct_change()
            correlation = prepared_data['sentiment'].corr(prepared_data['price_change'])
            
            self.parameters = {
                'sentiment_threshold_positive': 0.1,
                'sentiment_threshold_negative': -0.1,
                'correlation_with_price': correlation
            }
            
            accuracy = abs(correlation) * 100  # Simplified accuracy based on correlation
            self.complete_run(accuracy)
            
            return {
                'status': 'success',
                'correlation': correlation,
                'accuracy': accuracy,
                'parameters': self.parameters
            }
            
        except Exception as e:
            self.error_run(str(e))
            return {'status': 'error', 'message': str(e)}
    
    def predict(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Generate sentiment-based predictions.
        """
        try:
            from textblob import TextBlob
            
            prepared_data = self.prepare_data(data)
            
            sentiments = []
            signals = []
            
            for text in prepared_data['news_text']:
                if pd.isna(text):
                    sentiment = 0
                else:
                    blob = TextBlob(str(text))
                    sentiment = blob.sentiment.polarity
                
                sentiments.append(sentiment)
                
                # Generate trading signals based on sentiment
                if sentiment > self.parameters.get('sentiment_threshold_positive', 0.1):
                    signals.append(1)  # Buy
                elif sentiment < self.parameters.get('sentiment_threshold_negative', -0.1):
                    signals.append(-1)  # Sell
                else:
                    signals.append(0)  # Hold
            
            return {
                'status': 'success',
                'sentiments': sentiments,
                'signals': signals,
                'timestamps': prepared_data['timestamp'].dt.isoformat().tolist()
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def evaluate(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        return self.calculate_common_metrics(actual, predicted)

class CryptoRLModel(BaseModel):
    """
    Reinforcement Learning model for crypto trading.
    """
    
    def __init__(self):
        super().__init__(
            model_id="crypto_rl",
            name="Reinforcement Learning",
            category=AssetCategory.CRYPTO,
            model_type=ModelType.ML,
            description="Adaptive trading strategies using RL algorithms"
        )
    
    def prepare_data(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Prepare data for RL training.
        """
        if not self.validate_data(data, ['timestamp', 'open', 'high', 'low', 'close', 'volume']):
            raise ValueError("Invalid data format for RL model")
        
        df = data.copy()
        
        # Calculate technical indicators as features
        df['returns'] = df['close'].pct_change()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['rsi'] = self._calculate_rsi(df['close'])
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        return df.dropna()
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate RSI indicator.
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Train RL model (simplified implementation).
        """
        self.start_run()
        
        try:
            prepared_data = self.prepare_data(data)
            
            # Simplified Q-learning implementation
            states = ['oversold', 'neutral', 'overbought']
            actions = ['buy', 'hold', 'sell']
            
            # Initialize Q-table
            import random
            q_table = {}
            for state in states:
                q_table[state] = {action: random.uniform(-1, 1) for action in actions}
            
            # Training parameters
            learning_rate = 0.1
            discount_factor = 0.95
            epsilon = 0.1
            
            total_reward = 0
            position = 0
            
            for i in range(1, len(prepared_data)):
                # Determine state based on RSI
                rsi = prepared_data.iloc[i]['rsi']
                if rsi < 30:
                    state = 'oversold'
                elif rsi > 70:
                    state = 'overbought'
                else:
                    state = 'neutral'
                
                # Choose action (epsilon-greedy)
                if random.random() < epsilon:
                    action = random.choice(actions)
                else:
                    action = max(q_table[state], key=q_table[state].get)
                
                # Calculate reward
                price_change = prepared_data.iloc[i]['returns']
                
                if action == 'buy' and position <= 0:
                    reward = price_change
                    position = 1
                elif action == 'sell' and position >= 0:
                    reward = -price_change
                    position = -1
                else:
                    reward = 0
                
                total_reward += reward
                
                # Update Q-table
                if i < len(prepared_data) - 1:
                    next_rsi = prepared_data.iloc[i + 1]['rsi']
                    if next_rsi < 30:
                        next_state = 'oversold'
                    elif next_rsi > 70:
                        next_state = 'overbought'
                    else:
                        next_state = 'neutral'
                    
                    max_next_q = max(q_table[next_state].values())
                    q_table[state][action] += learning_rate * (reward + discount_factor * max_next_q - q_table[state][action])
            
            self.model_instance = q_table
            self.parameters = {
                'learning_rate': learning_rate,
                'discount_factor': discount_factor,
                'total_reward': total_reward
            }
            
            # Calculate accuracy based on positive rewards
            accuracy = max(0, min(100, (total_reward + 1) * 50))  # Normalize to 0-100
            self.complete_run(accuracy)
            
            return {
                'status': 'success',
                'total_reward': total_reward,
                'accuracy': accuracy,
                'q_table': q_table
            }
            
        except Exception as e:
            self.error_run(str(e))
            return {'status': 'error', 'message': str(e)}
    
    def predict(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Generate RL-based trading actions.
        """
        if self.model_instance is None:
            return {'status': 'error', 'message': 'Model not trained'}
        
        try:
            prepared_data = self.prepare_data(data)
            actions = []
            
            for _, row in prepared_data.iterrows():
                rsi = row['rsi']
                
                if rsi < 30:
                    state = 'oversold'
                elif rsi > 70:
                    state = 'overbought'
                else:
                    state = 'neutral'
                
                # Choose best action from Q-table
                best_action = max(self.model_instance[state], key=self.model_instance[state].get)
                
                if best_action == 'buy':
                    actions.append(1)
                elif best_action == 'sell':
                    actions.append(-1)
                else:
                    actions.append(0)
            
            return {
                'status': 'success',
                'actions': actions,
                'timestamps': prepared_data['timestamp'].dt.isoformat().tolist()
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def evaluate(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        return self.calculate_common_metrics(actual, predicted)