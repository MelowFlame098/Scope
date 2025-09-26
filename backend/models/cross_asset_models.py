import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from .base_model import BaseModel, ModelType, AssetCategory
import logging

class ARIMAModel(BaseModel):
    """
    ARIMA (AutoRegressive Integrated Moving Average) model for cross-asset prediction.
    """
    
    def __init__(self):
        super().__init__(
            model_id="cross_arima",
            name="ARIMA",
            category=AssetCategory.CROSS_ASSET,
            model_type=ModelType.STATISTICAL,
            description="ARIMA model for time series prediction across asset classes"
        )
    
    def prepare_data(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Prepare data for ARIMA analysis.
        """
        required_cols = ['timestamp', 'close']
        if not self.validate_data(data, required_cols):
            raise ValueError("Invalid data format for ARIMA model")
        
        df = data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Calculate returns for stationarity
        df['returns'] = df['close'].pct_change()
        
        # Calculate log prices
        df['log_price'] = np.log(df['close'])
        df['log_returns'] = df['log_price'].diff()
        
        # Add technical indicators
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        return df.dropna()
    
    def _check_stationarity(self, series: pd.Series) -> bool:
        """
        Simple stationarity check using rolling statistics.
        """
        # Calculate rolling statistics
        rolling_mean = series.rolling(window=12).mean()
        rolling_std = series.rolling(window=12).std()
        
        # Check if mean and std are relatively stable
        mean_stability = rolling_mean.std() / rolling_mean.mean() < 0.1
        std_stability = rolling_std.std() / rolling_std.mean() < 0.5
        
        return mean_stability and std_stability
    
    def _auto_arima(self, series: pd.Series, max_p: int = 3, max_d: int = 2, max_q: int = 3) -> Tuple[int, int, int]:
        """
        Simplified auto ARIMA order selection using AIC.
        """
        best_aic = float('inf')
        best_order = (1, 1, 1)
        
        # Test different combinations
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        # Simple ARIMA implementation using linear regression
                        if d > 0:
                            diff_series = series.diff(d).dropna()
                        else:
                            diff_series = series
                        
                        if len(diff_series) < max(p, q) + 10:
                            continue
                        
                        # Create lagged features
                        X = []
                        y = diff_series.values[max(p, q):]
                        
                        # AR terms
                        for lag in range(1, p + 1):
                            X.append(diff_series.shift(lag).values[max(p, q):])
                        
                        # MA terms (simplified as lagged residuals)
                        for lag in range(1, q + 1):
                            # Use lagged differences as proxy for MA terms
                            X.append(diff_series.shift(lag).values[max(p, q):])
                        
                        if X:
                            X = np.column_stack(X)
                            
                            # Fit linear regression
                            from sklearn.linear_model import LinearRegression
                            model = LinearRegression()
                            model.fit(X, y)
                            
                            # Calculate AIC (simplified)
                            y_pred = model.predict(X)
                            mse = np.mean((y - y_pred) ** 2)
                            n = len(y)
                            k = X.shape[1] + 1  # parameters + intercept
                            aic = n * np.log(mse) + 2 * k
                            
                            if aic < best_aic:
                                best_aic = aic
                                best_order = (p, d, q)
                    
                    except Exception:
                        continue
        
        return best_order
    
    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Train Ichimoku model.
        """
        self.start_run()
        
        try:
            prepared_data = self.prepare_data(data, **kwargs)
            
            # Analyze Ichimoku signal effectiveness
            signals = []
            returns = []
            
            for i in range(1, len(prepared_data)):
                row = prepared_data.iloc[i]
                prev_row = prepared_data.iloc[i-1]
                
                signal = 0
                
                # Ichimoku signals
                # TK Cross above cloud
                if (prev_row['tk_cross'] <= 0 and row['tk_cross'] > 0 and 
                    row['price_vs_cloud'] == 1):
                    signal = 1  # Bullish TK cross above cloud
                elif (prev_row['tk_cross'] >= 0 and row['tk_cross'] < 0 and 
                      row['price_vs_cloud'] == -1):
                    signal = -1  # Bearish TK cross below cloud
                
                # Price breakout signals
                elif (prev_row['price_vs_cloud'] <= 0 and row['price_vs_cloud'] == 1):
                    signal = 1  # Price breaks above cloud
                elif (prev_row['price_vs_cloud'] >= 0 and row['price_vs_cloud'] == -1):
                    signal = -1  # Price breaks below cloud
                
                signals.append(signal)
                
                # Calculate forward return
                if i < len(prepared_data) - 1:
                    forward_return = prepared_data.iloc[i+1]['returns']
                    returns.append(forward_return * signal)
                else:
                    returns.append(0)
            
            # Calculate performance metrics
            total_return = np.sum(returns)
            win_rate = len([r for r in returns if r > 0]) / len([r for r in returns if r != 0]) if len([r for r in returns if r != 0]) > 0 else 0
            
            # Sharpe ratio
            if np.std(returns) > 0:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
            else:
                sharpe_ratio = 0
            
            accuracy = win_rate * 100
            
            self.parameters = {
                'tenkan_period': kwargs.get('tenkan_period', 9),
                'kijun_period': kwargs.get('kijun_period', 26),
                'senkou_span_b_period': kwargs.get('senkou_span_b_period', 52),
                'displacement': kwargs.get('displacement', 26),
                'total_return': total_return,
                'win_rate': win_rate,
                'sharpe_ratio': sharpe_ratio
            }
            
            self.complete_run(accuracy)
            
            return {
                'status': 'success',
                'total_return': total_return,
                'win_rate': win_rate,
                'sharpe_ratio': sharpe_ratio,
                'accuracy': accuracy
            }
            
        except Exception as e:
            self.error_run(str(e))
            return {'status': 'error', 'message': str(e)}
    
    def predict(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Generate Ichimoku predictions.
        """
        if not self.parameters:
            return {'status': 'error', 'message': 'Model not trained'}
        
        try:
            prepared_data = self.prepare_data(data, **kwargs)
            
            signals = []
            ichimoku_data = {
                'tenkan_sen': [],
                'kijun_sen': [],
                'senkou_span_a': [],
                'senkou_span_b': [],
                'chikou_span': [],
                'cloud_top': [],
                'cloud_bottom': [],
                'price_vs_cloud': []
            }
            
            for i in range(len(prepared_data)):
                row = prepared_data.iloc[i]
                
                signal = 0
                
                if i > 0:
                    prev_row = prepared_data.iloc[i-1]
                    
                    # Ichimoku signals
                    if (prev_row['tk_cross'] <= 0 and row['tk_cross'] > 0 and 
                        row['price_vs_cloud'] == 1):
                        signal = 1  # Bullish TK cross above cloud
                    elif (prev_row['tk_cross'] >= 0 and row['tk_cross'] < 0 and 
                          row['price_vs_cloud'] == -1):
                        signal = -1  # Bearish TK cross below cloud
                    elif (prev_row['price_vs_cloud'] <= 0 and row['price_vs_cloud'] == 1):
                        signal = 1  # Price breaks above cloud
                    elif (prev_row['price_vs_cloud'] >= 0 and row['price_vs_cloud'] == -1):
                        signal = -1  # Price breaks below cloud
                
                signals.append(signal)
                
                # Store Ichimoku data
                for key in ichimoku_data.keys():
                    ichimoku_data[key].append(row[key] if pd.notna(row[key]) else None)
            
            return {
                'status': 'success',
                'signals': signals,
                'ichimoku_data': ichimoku_data,
                'timestamps': prepared_data['timestamp'].dt.isoformat().tolist()
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def evaluate(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        return self.calculate_common_metrics(actual, predicted)


class PPORLModel(BaseModel):
    """
    Proximal Policy Optimization Reinforcement Learning Model for trading.
    """
    
    def __init__(self):
        super().__init__(
            model_id="ppo_rl",
            name="PPO (RL)",
            category="Cross-Asset",
            model_type="Reinforcement Learning",
            description="Proximal Policy Optimization for trading decisions"
        )
    
    def prepare_data(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Prepare data for PPO RL model.
        """
        df = data.copy()
        
        # Ensure required columns
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
        
        # Calculate features for RL state
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std()
        df['rsi'] = self._calculate_rsi(df['close'])
        df['macd'], df['macd_signal'] = self._calculate_macd(df['close'])
        df['bb_upper'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'])
        
        # Normalize features
        feature_cols = ['returns', 'volatility', 'rsi', 'macd', 'macd_signal']
        for col in feature_cols:
            df[f'{col}_norm'] = (df[col] - df[col].mean()) / df[col].std()
        
        return df.dropna()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series) -> tuple:
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        return macd, signal
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20) -> tuple:
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        return upper, lower
    
    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Train PPO RL model.
        """
        self.start_run()
        
        try:
            prepared_data = self.prepare_data(data, **kwargs)
            
            # Simulate PPO training
            episodes = kwargs.get('episodes', 1000)
            learning_rate = kwargs.get('learning_rate', 0.0003)
            
            # Simple trading strategy simulation
            portfolio_value = 10000
            position = 0
            trades = []
            
            for i in range(1, len(prepared_data)):
                row = prepared_data.iloc[i]
                
                # Simple strategy based on multiple indicators
                rsi_signal = 1 if row['rsi'] < 30 else (-1 if row['rsi'] > 70 else 0)
                macd_signal = 1 if row['macd'] > row['macd_signal'] else -1
                
                # Combine signals
                action = 0
                if rsi_signal == 1 and macd_signal == 1:
                    action = 1  # Buy
                elif rsi_signal == -1 and macd_signal == -1:
                    action = -1  # Sell
                
                # Execute trade
                if action != 0 and position != action:
                    price = row['close']
                    if action == 1:  # Buy
                        shares = portfolio_value / price
                        portfolio_value = 0
                        position = 1
                    else:  # Sell
                        portfolio_value = shares * price
                        position = 0
                    
                    trades.append({
                        'timestamp': row['timestamp'],
                        'action': action,
                        'price': price,
                        'portfolio_value': portfolio_value if action == -1 else shares * price
                    })
            
            # Calculate performance
            if trades:
                final_value = trades[-1]['portfolio_value']
                total_return = (final_value - 10000) / 10000
                
                # Calculate Sharpe ratio
                returns = []
                for i in range(1, len(trades)):
                    ret = (trades[i]['portfolio_value'] - trades[i-1]['portfolio_value']) / trades[i-1]['portfolio_value']
                    returns.append(ret)
                
                if len(returns) > 0 and np.std(returns) > 0:
                    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
                else:
                    sharpe_ratio = 0
                
                accuracy = max(0, min(100, 50 + total_return * 100))
            else:
                total_return = 0
                sharpe_ratio = 0
                accuracy = 50
            
            self.parameters = {
                'episodes': episodes,
                'learning_rate': learning_rate,
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'num_trades': len(trades)
            }
            
            self.complete_run(accuracy)
            
            return {
                'status': 'success',
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'num_trades': len(trades),
                'accuracy': accuracy
            }
            
        except Exception as e:
            self.error_run(str(e))
            return {'status': 'error', 'message': str(e)}
    
    def predict(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Generate PPO RL predictions.
        """
        if not self.parameters:
            return {'status': 'error', 'message': 'Model not trained'}
        
        try:
            prepared_data = self.prepare_data(data, **kwargs)
            
            actions = []
            confidence_scores = []
            
            for i in range(len(prepared_data)):
                row = prepared_data.iloc[i]
                
                # Generate action based on trained strategy
                rsi_signal = 1 if row['rsi'] < 30 else (-1 if row['rsi'] > 70 else 0)
                macd_signal = 1 if row['macd'] > row['macd_signal'] else -1
                
                # Combine signals
                action = 0
                confidence = 0.5
                
                if rsi_signal == 1 and macd_signal == 1:
                    action = 1  # Buy
                    confidence = 0.8
                elif rsi_signal == -1 and macd_signal == -1:
                    action = -1  # Sell
                    confidence = 0.8
                elif rsi_signal != 0 or macd_signal != 0:
                    action = (rsi_signal + macd_signal) / 2
                    confidence = 0.6
                
                actions.append(action)
                confidence_scores.append(confidence)
            
            return {
                'status': 'success',
                'actions': actions,
                'confidence': confidence_scores,
                'timestamps': prepared_data['timestamp'].dt.isoformat().tolist()
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def evaluate(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        return self.calculate_common_metrics(actual, predicted)


class MarkowitzMPTModel(BaseModel):
    """
    Markowitz Modern Portfolio Theory Model for portfolio optimization.
    """
    
    def __init__(self):
        super().__init__(
            model_id="markowitz_mpt",
            name="Markowitz MPT",
            category="Cross-Asset",
            model_type="Portfolio Optimization",
            description="Modern Portfolio Theory for optimal asset allocation"
        )
    
    def prepare_data(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Prepare data for Markowitz MPT model.
        """
        df = data.copy()
        
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        
        # If multiple assets, calculate correlation matrix
        if 'asset' in df.columns:
            # Pivot to get returns for each asset
            returns_matrix = df.pivot(index='timestamp', columns='asset', values='returns')
            df['correlation_data'] = returns_matrix.to_json()
        
        return df.dropna()
    
    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Train Markowitz MPT model.
        """
        self.start_run()
        
        try:
            prepared_data = self.prepare_data(data, **kwargs)
            
            # Risk-free rate
            risk_free_rate = kwargs.get('risk_free_rate', 0.02)
            
            # If single asset, create simple portfolio metrics
            returns = prepared_data['returns'].dropna()
            
            if len(returns) == 0:
                raise ValueError("No valid returns data")
            
            # Calculate portfolio statistics
            expected_return = np.mean(returns) * 252  # Annualized
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            sharpe_ratio = (expected_return - risk_free_rate) / volatility if volatility > 0 else 0
            
            # Simulate efficient frontier points
            target_returns = np.linspace(expected_return * 0.5, expected_return * 1.5, 10)
            efficient_frontier = []
            
            for target_ret in target_returns:
                # Simple optimization: adjust volatility based on target return
                if target_ret <= expected_return:
                    portfolio_vol = volatility * (target_ret / expected_return)
                else:
                    portfolio_vol = volatility * (target_ret / expected_return) ** 1.5
                
                portfolio_sharpe = (target_ret - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
                
                efficient_frontier.append({
                    'return': target_ret,
                    'volatility': portfolio_vol,
                    'sharpe_ratio': portfolio_sharpe
                })
            
            # Find optimal portfolio (max Sharpe ratio)
            optimal_portfolio = max(efficient_frontier, key=lambda x: x['sharpe_ratio'])
            
            # Calculate accuracy based on Sharpe ratio
            accuracy = max(0, min(100, 50 + optimal_portfolio['sharpe_ratio'] * 20))
            
            self.parameters = {
                'risk_free_rate': risk_free_rate,
                'expected_return': expected_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'optimal_portfolio': optimal_portfolio,
                'efficient_frontier': efficient_frontier
            }
            
            self.complete_run(accuracy)
            
            return {
                'status': 'success',
                'expected_return': expected_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'optimal_portfolio': optimal_portfolio,
                'accuracy': accuracy
            }
            
        except Exception as e:
            self.error_run(str(e))
            return {'status': 'error', 'message': str(e)}
    
    def predict(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Generate MPT portfolio recommendations.
        """
        if not self.parameters:
            return {'status': 'error', 'message': 'Model not trained'}
        
        try:
            prepared_data = self.prepare_data(data, **kwargs)
            
            # Get recent returns for portfolio rebalancing
            recent_returns = prepared_data['returns'].tail(30).dropna()
            
            if len(recent_returns) == 0:
                return {'status': 'error', 'message': 'No recent returns data'}
            
            # Calculate current portfolio metrics
            current_return = np.mean(recent_returns) * 252
            current_volatility = np.std(recent_returns) * np.sqrt(252)
            
            optimal_portfolio = self.parameters['optimal_portfolio']
            
            # Portfolio recommendations
            recommendations = []
            
            # Compare current vs optimal
            if current_volatility > optimal_portfolio['volatility'] * 1.1:
                recommendations.append({
                    'action': 'reduce_risk',
                    'reason': 'Current volatility exceeds optimal level',
                    'target_volatility': optimal_portfolio['volatility']
                })
            elif current_volatility < optimal_portfolio['volatility'] * 0.9:
                recommendations.append({
                    'action': 'increase_risk',
                    'reason': 'Current volatility below optimal level',
                    'target_volatility': optimal_portfolio['volatility']
                })
            
            if current_return < optimal_portfolio['return'] * 0.9:
                recommendations.append({
                    'action': 'increase_return',
                    'reason': 'Current return below optimal level',
                    'target_return': optimal_portfolio['return']
                })
            
            return {
                'status': 'success',
                'current_metrics': {
                    'return': current_return,
                    'volatility': current_volatility,
                    'sharpe_ratio': (current_return - self.parameters['risk_free_rate']) / current_volatility if current_volatility > 0 else 0
                },
                'optimal_metrics': optimal_portfolio,
                'recommendations': recommendations,
                'efficient_frontier': self.parameters['efficient_frontier'],
                'timestamps': prepared_data['timestamp'].dt.isoformat().tolist()
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def evaluate(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        return self.calculate_common_metrics(actual, predicted)
    
    def predict(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Generate PPO RL predictions.
        """
        if not self.parameters:
            return {'status': 'error', 'message': 'Model not trained'}
        
        try:
            prepared_data = self.prepare_data(data, **kwargs)
            
            actions = []
            confidence = []
            
            for i in range(len(prepared_data)):
                row = prepared_data.iloc[i]
                
                # Generate action based on trained strategy
                rsi_signal = 1 if row['rsi'] < 30 else (-1 if row['rsi'] > 70 else 0)
                macd_signal = 1 if row['macd'] > row['macd_signal'] else -1
                
                action = 0
                conf = 0.5
                
                if rsi_signal == 1 and macd_signal == 1:
                    action = 1  # Buy
                    conf = 0.8
                elif rsi_signal == -1 and macd_signal == -1:
                    action = -1  # Sell
                    conf = 0.8
                elif rsi_signal != 0 or macd_signal != 0:
                    action = rsi_signal if abs(rsi_signal) > abs(macd_signal) else macd_signal
                    conf = 0.6
                
                actions.append(action)
                confidence.append(conf)
            
            return {
                'status': 'success',
                'actions': actions,
                'confidence': confidence,
                'timestamps': prepared_data['timestamp'].dt.isoformat().tolist()
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def evaluate(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        return self.calculate_common_metrics(actual, predicted)


class MarkowitzMPTModel(BaseModel):
    """
    Markowitz Modern Portfolio Theory Model.
    """
    
    def __init__(self):
        super().__init__(
            model_id="markowitz_mpt",
            name="Markowitz MPT",
            category="Cross-Asset",
            model_type="Portfolio Optimization",
            description="Modern Portfolio Theory for optimal asset allocation"
        )
    
    def prepare_data(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Prepare data for Markowitz MPT model.
        """
        df = data.copy()
        
        # Ensure required columns
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        
        return df.dropna()
    
    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Train Markowitz MPT model.
        """
        self.start_run()
        
        try:
            prepared_data = self.prepare_data(data, **kwargs)
            
            # Calculate expected returns and covariance
            returns = prepared_data['returns'].values
            expected_return = np.mean(returns)
            variance = np.var(returns)
            
            # Risk-free rate
            risk_free_rate = kwargs.get('risk_free_rate', 0.02) / 252  # Daily
            
            # Calculate Sharpe ratio
            if variance > 0:
                sharpe_ratio = (expected_return - risk_free_rate) / np.sqrt(variance)
            else:
                sharpe_ratio = 0
            
            # Calculate optimal portfolio weight (single asset case)
            if variance > 0:
                optimal_weight = (expected_return - risk_free_rate) / variance
                optimal_weight = max(0, min(1, optimal_weight))  # Constrain to [0,1]
            else:
                optimal_weight = 0.5
            
            accuracy = max(0, min(100, 50 + sharpe_ratio * 20))
            
            self.parameters = {
                'expected_return': expected_return,
                'variance': variance,
                'sharpe_ratio': sharpe_ratio,
                'optimal_weight': optimal_weight,
                'risk_free_rate': risk_free_rate
            }
            
            self.complete_run(accuracy)
            
            return {
                'status': 'success',
                'expected_return': expected_return,
                'variance': variance,
                'sharpe_ratio': sharpe_ratio,
                'optimal_weight': optimal_weight,
                'accuracy': accuracy
            }
            
        except Exception as e:
            self.error_run(str(e))
            return {'status': 'error', 'message': str(e)}
    
    def predict(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Generate MPT predictions.
        """
        if not self.parameters:
            return {'status': 'error', 'message': 'Model not trained'}
        
        try:
            prepared_data = self.prepare_data(data, **kwargs)
            
            # Generate portfolio allocation recommendations
            allocations = []
            expected_returns = []
            risks = []
            
            for i in range(len(prepared_data)):
                # Use trained optimal weight
                allocation = self.parameters['optimal_weight']
                expected_ret = self.parameters['expected_return']
                risk = np.sqrt(self.parameters['variance'])
                
                allocations.append(allocation)
                expected_returns.append(expected_ret)
                risks.append(risk)
            
            return {
                'status': 'success',
                'allocations': allocations,
                'expected_returns': expected_returns,
                'risks': risks,
                'sharpe_ratio': self.parameters['sharpe_ratio'],
                'timestamps': prepared_data['timestamp'].dt.isoformat().tolist()
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def evaluate(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        return self.calculate_common_metrics(actual, predicted)
        self.start_run()
        
        try:
            prepared_data = self.prepare_data(data, **kwargs)
            
            # Use log returns for modeling
            series = prepared_data['log_returns'].dropna()
            
            # Auto-select ARIMA order
            if 'order' in kwargs:
                p, d, q = kwargs['order']
            else:
                p, d, q = self._auto_arima(series)
            
            # Prepare data for ARIMA
            if d > 0:
                diff_series = series.diff(d).dropna()
            else:
                diff_series = series
            
            # Create features for ARIMA regression
            max_lag = max(p, q)
            if len(diff_series) < max_lag + 20:
                raise ValueError("Insufficient data for ARIMA model")
            
            X = []
            y = diff_series.values[max_lag:]
            
            # AR terms
            for lag in range(1, p + 1):
                X.append(diff_series.shift(lag).values[max_lag:])
            
            # MA terms (using lagged residuals approximation)
            residuals = diff_series - diff_series.shift(1)
            for lag in range(1, q + 1):
                X.append(residuals.shift(lag).values[max_lag:])
            
            if X:
                X = np.column_stack(X)
            else:
                X = np.ones((len(y), 1))  # Intercept only
            
            # Fit model
            from sklearn.linear_model import LinearRegression
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error, r2_score
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            self.model_instance = LinearRegression()
            self.model_instance.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model_instance.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Calculate accuracy
            accuracy = max(0, r2 * 100)
            
            self.parameters = {
                'order': (p, d, q),
                'mse': mse,
                'r2': r2,
                'coefficients': self.model_instance.coef_.tolist(),
                'intercept': self.model_instance.intercept_,
                'max_lag': max_lag
            }
            
            self.complete_run(accuracy)
            
            return {
                'status': 'success',
                'order': (p, d, q),
                'mse': mse,
                'r2': r2,
                'accuracy': accuracy
            }
            
        except Exception as e:
            self.error_run(str(e))
            return {'status': 'error', 'message': str(e)}
    
    def predict(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Generate ARIMA predictions.
        """
        if self.model_instance is None:
            return {'status': 'error', 'message': 'Model not trained'}
        
        try:
            prepared_data = self.prepare_data(data, **kwargs)
            series = prepared_data['log_returns'].dropna()
            
            p, d, q = self.parameters['order']
            max_lag = self.parameters['max_lag']
            
            predictions = []
            signals = []
            
            # Generate predictions for each point
            for i in range(max_lag, len(series)):
                # Prepare features
                if d > 0:
                    diff_series = series.diff(d)
                else:
                    diff_series = series
                
                X_pred = []
                
                # AR terms
                for lag in range(1, p + 1):
                    if i - lag >= 0:
                        X_pred.append(diff_series.iloc[i - lag])
                    else:
                        X_pred.append(0)
                
                # MA terms (simplified)
                residuals = diff_series - diff_series.shift(1)
                for lag in range(1, q + 1):
                    if i - lag >= 0:
                        X_pred.append(residuals.iloc[i - lag])
                    else:
                        X_pred.append(0)
                
                if X_pred:
                    X_pred = np.array(X_pred).reshape(1, -1)
                else:
                    X_pred = np.array([[0]])  # Fallback
                
                # Predict
                pred_return = self.model_instance.predict(X_pred)[0]
                predictions.append(pred_return)
                
                # Generate signal
                if pred_return > 0.005:  # 0.5% threshold
                    signals.append(1)  # Buy
                elif pred_return < -0.005:
                    signals.append(-1)  # Sell
                else:
                    signals.append(0)  # Hold
            
            # Pad with None for alignment
            predictions = [None] * max_lag + predictions
            signals = [0] * max_lag + signals
            
            return {
                'status': 'success',
                'predictions': predictions,
                'signals': signals,
                'order': self.parameters['order'],
                'timestamps': prepared_data['timestamp'].dt.isoformat().tolist()
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def evaluate(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        return self.calculate_common_metrics(actual, predicted)

class GARCHModel(BaseModel):
    """
    GARCH (Generalized Autoregressive Conditional Heteroskedasticity) model for volatility prediction.
    """
    
    def __init__(self):
        super().__init__(
            model_id="cross_garch",
            name="GARCH",
            category=AssetCategory.CROSS_ASSET,
            model_type=ModelType.STATISTICAL,
            description="GARCH model for volatility prediction across asset classes"
        )
    
    def prepare_data(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Prepare data for GARCH analysis.
        """
        required_cols = ['timestamp', 'close']
        if not self.validate_data(data, required_cols):
            raise ValueError("Invalid data format for GARCH model")
        
        df = data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        
        # Calculate squared returns (proxy for volatility)
        df['squared_returns'] = df['returns'] ** 2
        
        # Calculate rolling volatility
        df['rolling_vol'] = df['returns'].rolling(window=20).std()
        
        return df.dropna()
    
    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Train GARCH model (simplified implementation).
        """
        self.start_run()
        
        try:
            prepared_data = self.prepare_data(data, **kwargs)
            
            returns = prepared_data['returns'].values
            squared_returns = prepared_data['squared_returns'].values
            
            # GARCH(1,1) parameters
            p = kwargs.get('p', 1)  # ARCH order
            q = kwargs.get('q', 1)  # GARCH order
            
            # Simplified GARCH estimation using linear regression
            # σ²_t = ω + α * ε²_{t-1} + β * σ²_{t-1}
            
            max_lag = max(p, q)
            if len(squared_returns) < max_lag + 20:
                raise ValueError("Insufficient data for GARCH model")
            
            # Prepare features
            X = []
            y = squared_returns[max_lag:]
            
            # ARCH terms (lagged squared returns)
            for lag in range(1, p + 1):
                X.append(squared_returns[max_lag - lag:-lag])
            
            # GARCH terms (lagged conditional variances)
            # For simplicity, use lagged squared returns as proxy
            for lag in range(1, q + 1):
                X.append(squared_returns[max_lag - lag:-lag])
            
            X = np.column_stack(X)
            
            # Fit model
            from sklearn.linear_model import LinearRegression
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error, r2_score
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            self.model_instance = LinearRegression()
            self.model_instance.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model_instance.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Calculate volatility predictions
            vol_pred = np.sqrt(np.maximum(y_pred, 0))  # Ensure non-negative
            vol_actual = np.sqrt(y_test)
            
            vol_mse = mean_squared_error(vol_actual, vol_pred)
            
            accuracy = max(0, r2 * 100)
            
            self.parameters = {
                'p': p,
                'q': q,
                'omega': self.model_instance.intercept_,
                'coefficients': self.model_instance.coef_.tolist(),
                'mse': mse,
                'vol_mse': vol_mse,
                'r2': r2,
                'max_lag': max_lag
            }
            
            self.complete_run(accuracy)
            
            return {
                'status': 'success',
                'p': p,
                'q': q,
                'mse': mse,
                'vol_mse': vol_mse,
                'r2': r2,
                'accuracy': accuracy
            }
            
        except Exception as e:
            self.error_run(str(e))
            return {'status': 'error', 'message': str(e)}
    
    def predict(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Generate GARCH volatility predictions.
        """
        if self.model_instance is None:
            return {'status': 'error', 'message': 'Model not trained'}
        
        try:
            prepared_data = self.prepare_data(data, **kwargs)
            squared_returns = prepared_data['squared_returns'].values
            
            p = self.parameters['p']
            q = self.parameters['q']
            max_lag = self.parameters['max_lag']
            
            volatility_predictions = []
            variance_predictions = []
            signals = []
            
            for i in range(max_lag, len(squared_returns)):
                # Prepare features
                X_pred = []
                
                # ARCH terms
                for lag in range(1, p + 1):
                    if i - lag >= 0:
                        X_pred.append(squared_returns[i - lag])
                    else:
                        X_pred.append(0)
                
                # GARCH terms
                for lag in range(1, q + 1):
                    if i - lag >= 0:
                        X_pred.append(squared_returns[i - lag])
                    else:
                        X_pred.append(0)
                
                X_pred = np.array(X_pred).reshape(1, -1)
                
                # Predict variance
                pred_variance = self.model_instance.predict(X_pred)[0]
                pred_variance = max(pred_variance, 1e-6)  # Ensure positive
                
                # Convert to volatility
                pred_volatility = np.sqrt(pred_variance)
                
                variance_predictions.append(pred_variance)
                volatility_predictions.append(pred_volatility)
                
                # Generate signal based on volatility regime
                current_vol = prepared_data['rolling_vol'].iloc[i]
                
                if pred_volatility > current_vol * 1.2:  # High volatility expected
                    signals.append(-1)  # Sell - high risk
                elif pred_volatility < current_vol * 0.8:  # Low volatility expected
                    signals.append(1)  # Buy - low risk
                else:
                    signals.append(0)  # Hold
            
            # Pad with None for alignment
            volatility_predictions = [None] * max_lag + volatility_predictions
            variance_predictions = [None] * max_lag + variance_predictions
            signals = [0] * max_lag + signals
            
            return {
                'status': 'success',
                'volatility_predictions': volatility_predictions,
                'variance_predictions': variance_predictions,
                'signals': signals,
                'garch_order': (p, q),
                'timestamps': prepared_data['timestamp'].dt.isoformat().tolist()
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def evaluate(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        return self.calculate_common_metrics(actual, predicted)

class TransformerModel(BaseModel):
    """
    Transformer model for cross-asset prediction using attention mechanisms.
    """
    
    def __init__(self):
        super().__init__(
            model_id="cross_transformer",
            name="Transformer",
            category=AssetCategory.CROSS_ASSET,
            model_type=ModelType.ML,
            description="Transformer model with attention for cross-asset prediction"
        )
    
    def prepare_data(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Prepare data for Transformer model.
        """
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not self.validate_data(data, required_cols):
            raise ValueError("Invalid data format for Transformer model")
        
        df = data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Calculate technical features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Price features
        df['hl_ratio'] = df['high'] / df['low']
        df['oc_ratio'] = df['open'] / df['close']
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'price_to_sma_{window}'] = df['close'] / df[f'sma_{window}']
        
        # Volatility
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Volume features
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        return df.dropna()
    
    def _create_sequences(self, data: pd.DataFrame, sequence_length: int, target_col: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for transformer training.
        """
        feature_cols = ['returns', 'log_returns', 'hl_ratio', 'oc_ratio', 'volatility',
                       'volume_ratio', 'rsi', 'macd', 'macd_signal'] + \
                      [col for col in data.columns if 'price_to_sma' in col]
        
        # Select available features
        available_features = [col for col in feature_cols if col in data.columns]
        
        X = data[available_features].values
        y = data[target_col].values
        
        sequences_X = []
        sequences_y = []
        
        for i in range(sequence_length, len(X)):
            sequences_X.append(X[i-sequence_length:i])
            sequences_y.append(y[i])
        
        return np.array(sequences_X), np.array(sequences_y)
    
    def _simple_attention(self, X: np.ndarray) -> np.ndarray:
        """
        Simplified attention mechanism.
        """
        # X shape: (batch_size, sequence_length, features)
        batch_size, seq_len, features = X.shape
        
        # Simple self-attention
        # Q, K, V are the same (X)
        attention_scores = np.zeros((batch_size, seq_len, seq_len))
        
        for b in range(batch_size):
            for i in range(seq_len):
                for j in range(seq_len):
                    # Dot product attention
                    attention_scores[b, i, j] = np.dot(X[b, i], X[b, j])
        
        # Apply softmax
        attention_weights = np.exp(attention_scores) / np.sum(np.exp(attention_scores), axis=-1, keepdims=True)
        
        # Apply attention to values
        attended_output = np.zeros_like(X)
        for b in range(batch_size):
            for i in range(seq_len):
                attended_output[b, i] = np.sum(attention_weights[b, i, :, np.newaxis] * X[b], axis=0)
        
        return attended_output
    
    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Train Transformer model (simplified implementation).
        """
        self.start_run()
        
        try:
            prepared_data = self.prepare_data(data, **kwargs)
            
            sequence_length = kwargs.get('sequence_length', 20)
            target_col = kwargs.get('target_col', 'returns')
            
            # Create sequences
            X, y = self._create_sequences(prepared_data, sequence_length, target_col)
            
            if len(X) == 0:
                raise ValueError("Insufficient data for sequence creation")
            
            # Apply attention mechanism
            X_attended = self._simple_attention(X)
            
            # Flatten for regression
            X_flat = X_attended.reshape(X_attended.shape[0], -1)
            
            # Split data
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import mean_squared_error, r2_score
            
            X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, shuffle=False)
            
            # Use Random Forest as the final predictor
            self.model_instance = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.model_instance.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model_instance.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            accuracy = max(0, r2 * 100)
            
            self.parameters = {
                'sequence_length': sequence_length,
                'target_col': target_col,
                'n_features': X.shape[2],
                'mse': mse,
                'r2': r2
            }
            
            self.complete_run(accuracy)
            
            return {
                'status': 'success',
                'sequence_length': sequence_length,
                'n_features': X.shape[2],
                'mse': mse,
                'r2': r2,
                'accuracy': accuracy
            }
            
        except Exception as e:
            self.error_run(str(e))
            return {'status': 'error', 'message': str(e)}
    
    def predict(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Generate Transformer predictions.
        """
        if self.model_instance is None:
            return {'status': 'error', 'message': 'Model not trained'}
        
        try:
            prepared_data = self.prepare_data(data, **kwargs)
            
            sequence_length = self.parameters['sequence_length']
            target_col = self.parameters['target_col']
            
            # Create sequences
            X, _ = self._create_sequences(prepared_data, sequence_length, target_col)
            
            if len(X) == 0:
                return {
                    'status': 'success',
                    'predictions': [],
                    'signals': [],
                    'timestamps': []
                }
            
            # Apply attention
            X_attended = self._simple_attention(X)
            X_flat = X_attended.reshape(X_attended.shape[0], -1)
            
            # Generate predictions
            predictions = self.model_instance.predict(X_flat)
            
            # Generate signals
            signals = []
            for pred in predictions:
                if pred > 0.01:  # 1% threshold
                    signals.append(1)  # Buy
                elif pred < -0.01:
                    signals.append(-1)  # Sell
                else:
                    signals.append(0)  # Hold
            
            # Align timestamps
            timestamps = prepared_data['timestamp'].iloc[sequence_length:].dt.isoformat().tolist()
            
            return {
                'status': 'success',
                'predictions': predictions.tolist(),
                'signals': signals,
                'timestamps': timestamps
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def evaluate(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        return self.calculate_common_metrics(actual, predicted)

class LightGBMModel(BaseModel):
    """
    LightGBM model for cross-asset prediction.
    """
    
    def __init__(self):
        super().__init__(
            model_id="cross_lightgbm",
            name="LightGBM",
            category=AssetCategory.CROSS_ASSET,
            model_type=ModelType.ML,
            description="LightGBM gradient boosting for cross-asset prediction"
        )
    
    def prepare_data(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Prepare data for LightGBM model.
        """
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not self.validate_data(data, required_cols):
            raise ValueError("Invalid data format for LightGBM model")
        
        df = data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['hl_ratio'] = df['high'] / df['low']
        df['oc_ratio'] = df['open'] / df['close']
        
        # Lagged features
        for lag in [1, 2, 3, 5, 10]:
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'price_to_sma_{window}'] = df['close'] / df[f'sma_{window}']
            df[f'sma_slope_{window}'] = df[f'sma_{window}'].diff()
        
        # Volatility features
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = df['returns'].rolling(window=window).std()
        
        # Volume features
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['volume_change'] = df['volume'].pct_change()
        
        # Technical indicators
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        bb_window = 20
        bb_std = 2
        df['bb_middle'] = df['close'].rolling(window=bb_window).mean()
        bb_std_val = df['close'].rolling(window=bb_window).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std_val * bb_std)
        df['bb_lower'] = df['bb_middle'] - (bb_std_val * bb_std)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Time features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        
        return df.dropna()
    
    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Train LightGBM model.
        """
        self.start_run()
        
        try:
            prepared_data = self.prepare_data(data, **kwargs)
            
            # Define features and target
            feature_cols = [col for col in prepared_data.columns 
                          if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            X = prepared_data[feature_cols]
            
            # Target: next period return
            target_col = kwargs.get('target_col', 'returns')
            y = prepared_data[target_col].shift(-1).dropna()  # Next period target
            
            # Align X and y
            X = X.iloc[:-1]  # Remove last row to align with y
            
            # Remove any remaining NaN values
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[mask]
            y = y[mask]
            
            if len(X) == 0:
                raise ValueError("No valid data after preprocessing")
            
            # Split data
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import GradientBoostingRegressor  # LightGBM alternative
            from sklearn.metrics import mean_squared_error, r2_score
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            # Train model (using GradientBoostingRegressor as LightGBM alternative)
            self.model_instance = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            self.model_instance.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model_instance.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Feature importance
            feature_importance = dict(zip(feature_cols, self.model_instance.feature_importances_))
            
            accuracy = max(0, r2 * 100)
            
            self.parameters = {
                'feature_cols': feature_cols,
                'target_col': target_col,
                'mse': mse,
                'r2': r2,
                'feature_importance': feature_importance
            }
            
            self.complete_run(accuracy)
            
            return {
                'status': 'success',
                'n_features': len(feature_cols),
                'mse': mse,
                'r2': r2,
                'accuracy': accuracy,
                'top_features': sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            }
            
        except Exception as e:
            self.error_run(str(e))
            return {'status': 'error', 'message': str(e)}
    
    def predict(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Generate LightGBM predictions.
        """
        if self.model_instance is None:
            return {'status': 'error', 'message': 'Model not trained'}
        
        try:
            prepared_data = self.prepare_data(data, **kwargs)
            
            feature_cols = self.parameters['feature_cols']
            X = prepared_data[feature_cols]
            
            # Generate predictions
            predictions = self.model_instance.predict(X)
            
            # Generate signals
            signals = []
            for pred in predictions:
                if pred > 0.005:  # 0.5% threshold
                    signals.append(1)  # Buy
                elif pred < -0.005:
                    signals.append(-1)  # Sell
                else:
                    signals.append(0)  # Hold
            
            return {
                'status': 'success',
                'predictions': predictions.tolist(),
                'signals': signals,
                'timestamps': prepared_data['timestamp'].dt.isoformat().tolist()
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def evaluate(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        return self.calculate_common_metrics(actual, predicted)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from LightGBM model."""
        return self.parameters.get('feature_importance')

class RSIMomentumModel(BaseModel):
    """
    RSI Momentum model for cross-asset trading.
    """
    
    def __init__(self):
        super().__init__(
            model_id="cross_rsi_momentum",
            name="RSI Momentum",
            category=AssetCategory.CROSS_ASSET,
            model_type=ModelType.TECHNICAL,
            description="RSI-based momentum model for cross-asset trading"
        )
    
    def prepare_data(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Prepare data for RSI momentum analysis.
        """
        required_cols = ['timestamp', 'close']
        if not self.validate_data(data, required_cols):
            raise ValueError("Invalid data format for RSI Momentum model")
        
        df = data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Calculate RSI
        rsi_period = kwargs.get('rsi_period', 14)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # RSI momentum
        df['rsi_momentum'] = df['rsi'].diff()
        df['rsi_sma'] = df['rsi'].rolling(window=10).mean()
        
        # Price momentum
        df['returns'] = df['close'].pct_change()
        df['momentum_5'] = df['close'].pct_change(periods=5)
        df['momentum_10'] = df['close'].pct_change(periods=10)
        df['momentum_20'] = df['close'].pct_change(periods=20)
        
        # Moving averages for trend
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['trend'] = np.where(df['sma_20'] > df['sma_50'], 1, -1)
        
        return df.dropna()
    
    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Train RSI momentum model.
        """
        self.start_run()
        
        try:
            prepared_data = self.prepare_data(data, **kwargs)
            
            # Define RSI levels
            oversold_level = kwargs.get('oversold_level', 30)
            overbought_level = kwargs.get('overbought_level', 70)
            
            # Analyze RSI effectiveness
            signals = []
            returns = []
            
            for i in range(1, len(prepared_data)):
                row = prepared_data.iloc[i]
                prev_row = prepared_data.iloc[i-1]
                
                signal = 0
                
                # RSI-based signals
                if (prev_row['rsi'] < oversold_level and row['rsi'] > oversold_level and 
                    row['rsi_momentum'] > 0 and row['trend'] == 1):
                    signal = 1  # Buy signal
                elif (prev_row['rsi'] > overbought_level and row['rsi'] < overbought_level and 
                      row['rsi_momentum'] < 0 and row['trend'] == -1):
                    signal = -1  # Sell signal
                
                signals.append(signal)
                
                # Calculate forward return
                if i < len(prepared_data) - 1:
                    forward_return = prepared_data.iloc[i+1]['returns']
                    returns.append(forward_return * signal)  # Signal-weighted return
                else:
                    returns.append(0)
            
            # Calculate performance metrics
            total_return = np.sum(returns)
            win_rate = len([r for r in returns if r > 0]) / len([r for r in returns if r != 0]) if len([r for r in returns if r != 0]) > 0 else 0
            
            # Sharpe ratio (simplified)
            if np.std(returns) > 0:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
            else:
                sharpe_ratio = 0
            
            accuracy = win_rate * 100
            
            self.parameters = {
                'oversold_level': oversold_level,
                'overbought_level': overbought_level,
                'total_return': total_return,
                'win_rate': win_rate,
                'sharpe_ratio': sharpe_ratio,
                'rsi_period': kwargs.get('rsi_period', 14)
            }
            
            self.complete_run(accuracy)
            
            return {
                'status': 'success',
                'total_return': total_return,
                'win_rate': win_rate,
                'sharpe_ratio': sharpe_ratio,
                'accuracy': accuracy
            }
            
        except Exception as e:
            self.error_run(str(e))
            return {'status': 'error', 'message': str(e)}
    
    def predict(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Generate RSI momentum predictions.
        """
        if not self.parameters:
            return {'status': 'error', 'message': 'Model not trained'}
        
        try:
            prepared_data = self.prepare_data(data, **kwargs)
            
            oversold_level = self.parameters['oversold_level']
            overbought_level = self.parameters['overbought_level']
            
            signals = []
            rsi_values = []
            momentum_values = []
            
            for i in range(len(prepared_data)):
                row = prepared_data.iloc[i]
                
                signal = 0
                
                # Current RSI-based signals
                if (row['rsi'] < oversold_level and row['rsi_momentum'] > 0 and row['trend'] == 1):
                    signal = 1  # Buy signal
                elif (row['rsi'] > overbought_level and row['rsi_momentum'] < 0 and row['trend'] == -1):
                    signal = -1  # Sell signal
                elif oversold_level <= row['rsi'] <= overbought_level:
                    # Momentum-based signals in neutral zone
                    if row['momentum_10'] > 0.02 and row['rsi_momentum'] > 0:
                        signal = 1
                    elif row['momentum_10'] < -0.02 and row['rsi_momentum'] < 0:
                        signal = -1
                
                signals.append(signal)
                rsi_values.append(row['rsi'])
                momentum_values.append(row['rsi_momentum'])
            
            return {
                'status': 'success',
                'signals': signals,
                'rsi_values': rsi_values,
                'rsi_momentum': momentum_values,
                'oversold_level': oversold_level,
                'overbought_level': overbought_level,
                'timestamps': prepared_data['timestamp'].dt.isoformat().tolist()
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def evaluate(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        return self.calculate_common_metrics(actual, predicted)

class MACDModel(BaseModel):
    """
    MACD (Moving Average Convergence Divergence) model for cross-asset trading.
    """
    
    def __init__(self):
        super().__init__(
            model_id="cross_macd",
            name="MACD",
            category=AssetCategory.CROSS_ASSET,
            model_type=ModelType.TECHNICAL,
            description="MACD technical indicator for cross-asset trading"
        )
    
    def prepare_data(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Prepare data for MACD analysis.
        """
        required_cols = ['timestamp', 'close']
        if not self.validate_data(data, required_cols):
            raise ValueError("Invalid data format for MACD model")
        
        df = data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # MACD parameters
        fast_period = kwargs.get('fast_period', 12)
        slow_period = kwargs.get('slow_period', 26)
        signal_period = kwargs.get('signal_period', 9)
        
        # Calculate MACD
        exp_fast = df['close'].ewm(span=fast_period).mean()
        exp_slow = df['close'].ewm(span=slow_period).mean()
        df['macd'] = exp_fast - exp_slow
        df['macd_signal'] = df['macd'].ewm(span=signal_period).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # MACD derivatives
        df['macd_momentum'] = df['macd'].diff()
        df['signal_momentum'] = df['macd_signal'].diff()
        df['histogram_momentum'] = df['macd_histogram'].diff()
        
        # Price features
        df['returns'] = df['close'].pct_change()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['price_to_sma'] = df['close'] / df['sma_20']
        
        # Volatility
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        return df.dropna()
    
    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Train MACD model.
        """
        self.start_run()
        
        try:
            prepared_data = self.prepare_data(data, **kwargs)
            
            # Analyze MACD signal effectiveness
            signals = []
            returns = []
            
            for i in range(1, len(prepared_data)):
                row = prepared_data.iloc[i]
                prev_row = prepared_data.iloc[i-1]
                
                signal = 0
                
                # MACD crossover signals
                if (prev_row['macd'] <= prev_row['macd_signal'] and 
                    row['macd'] > row['macd_signal'] and 
                    row['macd_histogram'] > 0):
                    signal = 1  # Bullish crossover
                elif (prev_row['macd'] >= prev_row['macd_signal'] and 
                      row['macd'] < row['macd_signal'] and 
                      row['macd_histogram'] < 0):
                    signal = -1  # Bearish crossover
                
                # MACD histogram signals
                elif (row['macd_histogram'] > 0 and row['histogram_momentum'] > 0 and 
                      row['macd'] > 0):
                    signal = 1  # Strengthening bullish momentum
                elif (row['macd_histogram'] < 0 and row['histogram_momentum'] < 0 and 
                      row['macd'] < 0):
                    signal = -1  # Strengthening bearish momentum
                
                signals.append(signal)
                
                # Calculate forward return
                if i < len(prepared_data) - 1:
                    forward_return = prepared_data.iloc[i+1]['returns']
                    returns.append(forward_return * signal)
                else:
                    returns.append(0)
            
            # Calculate performance metrics
            total_return = np.sum(returns)
            win_rate = len([r for r in returns if r > 0]) / len([r for r in returns if r != 0]) if len([r for r in returns if r != 0]) > 0 else 0
            
            # Sharpe ratio
            if np.std(returns) > 0:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
            else:
                sharpe_ratio = 0
            
            # Count signal types
            bullish_signals = len([s for s in signals if s == 1])
            bearish_signals = len([s for s in signals if s == -1])
            
            accuracy = win_rate * 100
            
            self.parameters = {
                'fast_period': kwargs.get('fast_period', 12),
                'slow_period': kwargs.get('slow_period', 26),
                'signal_period': kwargs.get('signal_period', 9),
                'total_return': total_return,
                'win_rate': win_rate,
                'sharpe_ratio': sharpe_ratio,
                'bullish_signals': bullish_signals,
                'bearish_signals': bearish_signals
            }
            
            self.complete_run(accuracy)
            
            return {
                'status': 'success',
                'total_return': total_return,
                'win_rate': win_rate,
                'sharpe_ratio': sharpe_ratio,
                'bullish_signals': bullish_signals,
                'bearish_signals': bearish_signals,
                'accuracy': accuracy
            }
            
        except Exception as e:
            self.error_run(str(e))
            return {'status': 'error', 'message': str(e)}
    
    def predict(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Generate MACD predictions.
        """
        if not self.parameters:
            return {'status': 'error', 'message': 'Model not trained'}
        
        try:
            prepared_data = self.prepare_data(data, **kwargs)
            
            signals = []
            macd_values = []
            signal_values = []
            histogram_values = []
            
            for i in range(len(prepared_data)):
                row = prepared_data.iloc[i]
                
                signal = 0
                
                if i > 0:
                    prev_row = prepared_data.iloc[i-1]
                    
                    # MACD crossover signals
                    if (prev_row['macd'] <= prev_row['macd_signal'] and 
                        row['macd'] > row['macd_signal'] and 
                        row['macd_histogram'] > 0):
                        signal = 1  # Bullish crossover
                    elif (prev_row['macd'] >= prev_row['macd_signal'] and 
                          row['macd'] < row['macd_signal'] and 
                          row['macd_histogram'] < 0):
                        signal = -1  # Bearish crossover
                    
                    # Momentum signals
                    elif (row['macd_histogram'] > 0 and row['histogram_momentum'] > 0 and 
                          row['macd'] > 0 and row['macd_momentum'] > 0):
                        signal = 1  # Strong bullish momentum
                    elif (row['macd_histogram'] < 0 and row['histogram_momentum'] < 0 and 
                          row['macd'] < 0 and row['macd_momentum'] < 0):
                        signal = -1  # Strong bearish momentum
                
                signals.append(signal)
                macd_values.append(row['macd'])
                signal_values.append(row['macd_signal'])
                histogram_values.append(row['macd_histogram'])
            
            return {
                'status': 'success',
                'signals': signals,
                'macd': macd_values,
                'macd_signal': signal_values,
                'macd_histogram': histogram_values,
                'timestamps': prepared_data['timestamp'].dt.isoformat().tolist()
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def evaluate(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        return self.calculate_common_metrics(actual, predicted)

class IchimokuModel(BaseModel):
    """
    Ichimoku Cloud model for cross-asset trading.
    """
    
    def __init__(self):
        super().__init__(
            model_id="cross_ichimoku",
            name="Ichimoku",
            category=AssetCategory.CROSS_ASSET,
            model_type=ModelType.TECHNICAL,
            description="Ichimoku Cloud technical analysis for cross-asset trading"
        )
    
    def prepare_data(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Prepare data for Ichimoku analysis.
        """
        required_cols = ['timestamp', 'high', 'low', 'close']
        if not self.validate_data(data, required_cols):
            raise ValueError("Invalid data format for Ichimoku model")
        
        df = data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Ichimoku parameters
        tenkan_period = kwargs.get('tenkan_period', 9)
        kijun_period = kwargs.get('kijun_period', 26)
        senkou_span_b_period = kwargs.get('senkou_span_b_period', 52)
        displacement = kwargs.get('displacement', 26)
        
        # Calculate Ichimoku components
        # Tenkan-sen (Conversion Line)
        tenkan_high = df['high'].rolling(window=tenkan_period).max()
        tenkan_low = df['low'].rolling(window=tenkan_period).min()
        df['tenkan_sen'] = (tenkan_high + tenkan_low) / 2
        
        # Kijun-sen (Base Line)
        kijun_high = df['high'].rolling(window=kijun_period).max()
        kijun_low = df['low'].rolling(window=kijun_period).min()
        df['kijun_sen'] = (kijun_high + kijun_low) / 2
        
        # Senkou Span A (Leading Span A)
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(displacement)
        
        # Senkou Span B (Leading Span B)
        senkou_b_high = df['high'].rolling(window=senkou_span_b_period).max()
        senkou_b_low = df['low'].rolling(window=senkou_span_b_period).min()
        df['senkou_span_b'] = ((senkou_b_high + senkou_b_low) / 2).shift(displacement)
        
        # Chikou Span (Lagging Span)
        df['chikou_span'] = df['close'].shift(-displacement)
        
        # Cloud boundaries
        df['cloud_top'] = np.maximum(df['senkou_span_a'], df['senkou_span_b'])
        df['cloud_bottom'] = np.minimum(df['senkou_span_a'], df['senkou_span_b'])
        
        # Price position relative to cloud
        df['price_vs_cloud'] = np.where(
            df['close'] > df['cloud_top'], 1,  # Above cloud
            np.where(df['close'] < df['cloud_bottom'], -1, 0)  # Below cloud, In cloud
        )
        
        # TK cross
        df['tk_cross'] = np.where(df['tenkan_sen'] > df['kijun_sen'], 1, -1)
        
        # Additional features
        df['returns'] = df['close'].pct_change()
        
        return df.dropna()
    
    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Train Ichimoku model.
        """
        self.start_run()
        
        try:
            prepared_data = self.prepare_data(data, **kwargs)
            
            # Analyze Ichimoku signal effectiveness
            signals = []
            returns = []
            
            for i in range(1, len(prepared_data)):
                row = prepared_data.iloc[i]
                prev_row = prepared_data.iloc[i-1]
                
                signal = 0
                
                # Ichimoku signals
                # TK Cross above cloud
                if (prev_row['tk_cross'] <= 0 and row['tk_cross'] > 0 and 
                    row['price_vs_cloud'] == 1):
                    signal = 1  # Bullish TK cross above cloud
                elif (prev_row['tk_cross'] >= 0 and row['tk_cross'] < 0 and 
                      row['price_vs_cloud'] == -1):
                    signal = -1  # Bearish TK cross below cloud
                
                # Price breakout signals
                elif (prev_row['price_vs_cloud'] <= 0 and row['price_vs_cloud'] == 1):
                    signal = 1  # Price breaks above cloud
                elif (prev_row['price_vs_cloud'] >= 0 and row['price_vs_cloud'] == -1):
                    signal = -1  # Price breaks below cloud
                
                signals.append(signal)
                
                # Calculate forward return
                if i < len(prepared_data) - 1:
                    forward_return = prepared_data.iloc[i+1]['returns']
                    returns.append(forward_return * signal)
                else:
                    returns.append(0)
            
            # Calculate performance metrics
            total_return = np.sum(returns)
            win_rate = len([r for r in returns if r > 0]) / len([r for r in returns if r != 0]) if len([r for r in returns if r != 0]) > 0 else 0
            
            # Sharpe ratio
            if np.std(returns) > 0:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
            else:
                sharpe_ratio = 0
            
            accuracy = win_rate * 100
            
            self.parameters = {
                'tenkan_period': kwargs.get('tenkan_period', 9),
                'kijun_period': kwargs.get('kijun_period', 26),
                'senkou_span_b_period': kwargs.get('senkou_span_b_period', 52),
                'displacement': kwargs.get('displacement', 26),
                'total_return': total_return,
                'win_rate': win_rate,
                'sharpe_ratio': sharpe_ratio
            }
            
            self.complete_run(accuracy)
            
            return {
                'status': 'success',
                'total_return': total_return,
                'win_rate': win_rate,
                'sharpe_ratio': sharpe_ratio,
                'accuracy': accuracy
            }
            
        except Exception as e:
            self.error_run(str(e))
            return {'status': 'error', 'message': str(e)}
    
    def predict(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Generate Ichimoku predictions.
        """
        if not self.parameters:
            return {'status': 'error', 'message': 'Model not trained'}
        
        try:
            prepared_data = self.prepare_data(data, **kwargs)
            
            signals = []
            ichimoku_data = {
                'tenkan_sen': [],
                'kijun_sen': [],
                'senkou_span_a': [],
                'senkou_span_b': [],
                'chikou_span': [],
                'cloud_top': [],
                'cloud_bottom': [],
                'price_vs_cloud': []
            }
            
            for i in range(len(prepared_data)):
                row = prepared_data.iloc[i]
                
                signal = 0
                
                if i > 0:
                    prev_row = prepared_data.iloc[i-1]
                    
                    # Ichimoku signals
                    if (prev_row['tk_cross'] <= 0 and row['tk_cross'] > 0 and 
                        row['price_vs_cloud'] == 1):
                        signal = 1  # Bullish TK cross above cloud
                    elif (prev_row['tk_cross'] >= 0 and row['tk_cross'] < 0 and 
                          row['price_vs_cloud'] == -1):
                        signal = -1  # Bearish TK cross below cloud
                    elif (prev_row['price_vs_cloud'] <= 0 and row['price_vs_cloud'] == 1):
                        signal = 1  # Price breaks above cloud
                    elif (prev_row['price_vs_cloud'] >= 0 and row['price_vs_cloud'] == -1):
                        signal = -1  # Price breaks below cloud
                
                signals.append(signal)
                
                # Store Ichimoku data
                for key in ichimoku_data.keys():
                    ichimoku_data[key].append(row[key] if pd.notna(row[key]) else None)
            
            return {
                'status': 'success',
                'signals': signals,
                'ichimoku_data': ichimoku_data,
                'timestamps': prepared_data['timestamp'].dt.isoformat().tolist()
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def evaluate(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        return self.calculate_common_metrics(actual, predicted)