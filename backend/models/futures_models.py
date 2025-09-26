import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from .base_model import BaseModel, ModelType, AssetCategory
import logging

class CostOfCarryModel(BaseModel):
    """
    Cost-of-Carry model for futures pricing.
    F = S * e^((r - q) * T)
    Where F = futures price, S = spot price, r = risk-free rate, q = dividend yield, T = time to maturity
    """
    
    def __init__(self):
        super().__init__(
            model_id="futures_cost_carry",
            name="Cost-of-Carry Model",
            category=AssetCategory.FUTURES,
            model_type=ModelType.FUNDAMENTAL,
            description="Cost-of-carry model for futures pricing"
        )
    
    def prepare_data(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Prepare data for cost-of-carry analysis.
        """
        required_cols = ['timestamp', 'close', 'spot_price', 'time_to_maturity']
        if not self.validate_data(data, required_cols):
            raise ValueError("Invalid data format for Cost-of-Carry model")
        
        df = data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Get parameters
        risk_free_rate = kwargs.get('risk_free_rate', 0.03)  # 3% default
        dividend_yield = kwargs.get('dividend_yield', 0.02)  # 2% default
        storage_cost = kwargs.get('storage_cost', 0.01)  # 1% default
        
        # Calculate theoretical futures price
        carry_cost = risk_free_rate + storage_cost - dividend_yield
        df['theoretical_price'] = df['spot_price'] * np.exp(carry_cost * df['time_to_maturity'])
        
        # Calculate basis (futures - spot)
        df['basis'] = df['close'] - df['spot_price']
        df['theoretical_basis'] = df['theoretical_price'] - df['spot_price']
        
        # Calculate mispricing
        df['mispricing'] = df['close'] - df['theoretical_price']
        df['mispricing_pct'] = df['mispricing'] / df['theoretical_price']
        
        return df.dropna()
    
    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Train cost-of-carry model by analyzing historical mispricings.
        """
        self.start_run()
        
        try:
            prepared_data = self.prepare_data(data, **kwargs)
            
            # Analyze mispricing patterns
            mispricings = prepared_data['mispricing_pct']
            
            # Calculate statistics
            mean_mispricing = mispricings.mean()
            std_mispricing = mispricings.std()
            
            # Test for mean reversion in mispricing
            from sklearn.linear_model import LinearRegression
            
            # AR(1) model for mispricing
            y = mispricings[1:].values
            X = mispricings[:-1].values.reshape(-1, 1)
            
            reg = LinearRegression()
            reg.fit(X, y)
            
            phi = reg.coef_[0]  # Mean reversion coefficient
            
            # Calculate half-life of mean reversion
            half_life = -np.log(2) / np.log(abs(phi)) if abs(phi) < 1 else np.inf
            
            # Calculate accuracy based on theoretical vs actual correlation
            from sklearn.metrics import r2_score
            theoretical_prices = prepared_data['theoretical_price'].values
            actual_prices = prepared_data['close'].values
            
            correlation = np.corrcoef(theoretical_prices, actual_prices)[0, 1]
            accuracy = abs(correlation) * 100
            
            self.parameters = {
                'risk_free_rate': kwargs.get('risk_free_rate', 0.03),
                'dividend_yield': kwargs.get('dividend_yield', 0.02),
                'storage_cost': kwargs.get('storage_cost', 0.01),
                'mean_mispricing': mean_mispricing,
                'std_mispricing': std_mispricing,
                'phi': phi,
                'half_life': half_life,
                'correlation': correlation
            }
            
            self.complete_run(accuracy)
            
            return {
                'status': 'success',
                'mean_mispricing': mean_mispricing,
                'std_mispricing': std_mispricing,
                'half_life': half_life,
                'correlation': correlation,
                'accuracy': accuracy
            }
            
        except Exception as e:
            self.error_run(str(e))
            return {'status': 'error', 'message': str(e)}
    
    def predict(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Generate cost-of-carry based predictions.
        """
        if not self.parameters:
            return {'status': 'error', 'message': 'Model not trained'}
        
        try:
            prepared_data = self.prepare_data(data, **kwargs)
            
            predictions = []
            signals = []
            
            for _, row in prepared_data.iterrows():
                theoretical_price = row['theoretical_price']
                current_price = row['close']
                mispricing_pct = row['mispricing_pct']
                
                predictions.append(theoretical_price)
                
                # Generate trading signals based on mispricing
                threshold = self.parameters['std_mispricing']
                
                if mispricing_pct < -threshold:
                    signals.append(1)  # Buy futures - underpriced
                elif mispricing_pct > threshold:
                    signals.append(-1)  # Sell futures - overpriced
                else:
                    signals.append(0)  # Hold
            
            return {
                'status': 'success',
                'theoretical_prices': predictions,
                'signals': signals,
                'mispricings': prepared_data['mispricing'].tolist(),
                'basis': prepared_data['basis'].tolist(),
                'timestamps': prepared_data['timestamp'].dt.isoformat().tolist()
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def evaluate(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        return self.calculate_common_metrics(actual, predicted)

class ConvenienceYieldModel(BaseModel):
    """
    Convenience Yield model for commodity futures.
    Accounts for the benefit of holding the physical commodity.
    """
    
    def __init__(self):
        super().__init__(
            model_id="futures_convenience_yield",
            name="Convenience Yield",
            category=AssetCategory.FUTURES,
            model_type=ModelType.FUNDAMENTAL,
            description="Convenience yield model for commodity futures"
        )
    
    def prepare_data(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Prepare data for convenience yield analysis.
        """
        required_cols = ['timestamp', 'close', 'spot_price', 'time_to_maturity']
        if not self.validate_data(data, required_cols):
            raise ValueError("Invalid data format for Convenience Yield model")
        
        df = data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Get parameters
        risk_free_rate = kwargs.get('risk_free_rate', 0.03)
        storage_cost = kwargs.get('storage_cost', 0.02)
        
        # Calculate implied convenience yield
        # F = S * e^((r + s - c) * T)
        # Solving for c: c = r + s - ln(F/S) / T
        df['convenience_yield'] = (risk_free_rate + storage_cost - 
                                 np.log(df['close'] / df['spot_price']) / df['time_to_maturity'])
        
        # Calculate inventory levels proxy (using volume)
        if 'volume' in df.columns:
            df['inventory_proxy'] = 1 / (df['volume'].rolling(window=20).mean() + 1)
        else:
            df['inventory_proxy'] = 0.5  # Default value
        
        # Calculate volatility
        df['returns'] = df['spot_price'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        return df.dropna()
    
    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Train convenience yield model.
        """
        self.start_run()
        
        try:
            prepared_data = self.prepare_data(data, **kwargs)
            
            # Analyze relationship between convenience yield and market conditions
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score
            
            # Features: inventory proxy, volatility, time to maturity
            X = prepared_data[['inventory_proxy', 'volatility', 'time_to_maturity']].values
            y = prepared_data['convenience_yield'].values
            
            # Remove any NaN values
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[mask]
            y = y[mask]
            
            reg = LinearRegression()
            reg.fit(X, y)
            
            y_pred = reg.predict(X)
            r2 = r2_score(y, y_pred)
            
            # Calculate statistics
            mean_convenience_yield = prepared_data['convenience_yield'].mean()
            std_convenience_yield = prepared_data['convenience_yield'].std()
            
            accuracy = max(0, r2 * 100)
            
            self.parameters = {
                'coefficients': reg.coef_.tolist(),
                'intercept': reg.intercept_,
                'r2': r2,
                'mean_convenience_yield': mean_convenience_yield,
                'std_convenience_yield': std_convenience_yield,
                'risk_free_rate': kwargs.get('risk_free_rate', 0.03),
                'storage_cost': kwargs.get('storage_cost', 0.02)
            }
            
            self.complete_run(accuracy)
            
            return {
                'status': 'success',
                'r2': r2,
                'mean_convenience_yield': mean_convenience_yield,
                'accuracy': accuracy
            }
            
        except Exception as e:
            self.error_run(str(e))
            return {'status': 'error', 'message': str(e)}
    
    def predict(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Generate convenience yield predictions.
        """
        if not self.parameters:
            return {'status': 'error', 'message': 'Model not trained'}
        
        try:
            prepared_data = self.prepare_data(data, **kwargs)
            
            # Predict convenience yields
            X = prepared_data[['inventory_proxy', 'volatility', 'time_to_maturity']].values
            
            predicted_yields = []
            fair_values = []
            signals = []
            
            for i, row in prepared_data.iterrows():
                if not np.isnan(X[i]).any():
                    # Predict convenience yield
                    predicted_yield = (self.parameters['intercept'] + 
                                     np.dot(self.parameters['coefficients'], X[i]))
                    predicted_yields.append(predicted_yield)
                    
                    # Calculate fair value futures price
                    carry_cost = (self.parameters['risk_free_rate'] + 
                                self.parameters['storage_cost'] - predicted_yield)
                    fair_value = row['spot_price'] * np.exp(carry_cost * row['time_to_maturity'])
                    fair_values.append(fair_value)
                    
                    # Generate trading signal
                    current_price = row['close']
                    if current_price < fair_value * 0.98:  # 2% threshold
                        signals.append(1)  # Buy
                    elif current_price > fair_value * 1.02:
                        signals.append(-1)  # Sell
                    else:
                        signals.append(0)  # Hold
                else:
                    predicted_yields.append(None)
                    fair_values.append(None)
                    signals.append(0)
            
            return {
                'status': 'success',
                'predicted_yields': predicted_yields,
                'fair_values': fair_values,
                'signals': signals,
                'actual_yields': prepared_data['convenience_yield'].tolist(),
                'timestamps': prepared_data['timestamp'].dt.isoformat().tolist()
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def evaluate(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        return self.calculate_common_metrics(actual, predicted)

class SamuelsonEffectModel(BaseModel):
    """
    Samuelson Effect model - volatility increases as futures approach maturity.
    """
    
    def __init__(self):
        super().__init__(
            model_id="futures_samuelson",
            name="Samuelson Effect",
            category=AssetCategory.FUTURES,
            model_type=ModelType.STATISTICAL,
            description="Samuelson effect model for futures volatility"
        )
    
    def prepare_data(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Prepare data for Samuelson effect analysis.
        """
        required_cols = ['timestamp', 'close', 'time_to_maturity']
        if not self.validate_data(data, required_cols):
            raise ValueError("Invalid data format for Samuelson Effect model")
        
        df = data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate returns and volatility
        df['returns'] = df['close'].pct_change()
        
        # Calculate rolling volatility
        window = kwargs.get('volatility_window', 20)
        df['volatility'] = df['returns'].rolling(window=window).std() * np.sqrt(252)  # Annualized
        
        # Calculate log time to maturity
        df['log_time_to_maturity'] = np.log(df['time_to_maturity'] + 0.001)  # Add small value to avoid log(0)
        
        return df.dropna()
    
    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Train Samuelson effect model.
        """
        self.start_run()
        
        try:
            prepared_data = self.prepare_data(data, **kwargs)
            
            # Model: volatility = a + b * log(time_to_maturity)
            # Samuelson effect predicts b < 0 (volatility decreases with time to maturity)
            
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score
            
            X = prepared_data['log_time_to_maturity'].values.reshape(-1, 1)
            y = prepared_data['volatility'].values
            
            # Remove NaN values
            mask = ~(np.isnan(X.flatten()) | np.isnan(y))
            X = X[mask]
            y = y[mask]
            
            reg = LinearRegression()
            reg.fit(X, y)
            
            y_pred = reg.predict(X)
            r2 = r2_score(y, y_pred)
            
            slope = reg.coef_[0]
            intercept = reg.intercept_
            
            # Check if Samuelson effect is present (negative slope)
            samuelson_present = slope < 0
            
            # Calculate volatility statistics
            mean_volatility = prepared_data['volatility'].mean()
            std_volatility = prepared_data['volatility'].std()
            
            accuracy = max(0, r2 * 100)
            
            self.parameters = {
                'slope': slope,
                'intercept': intercept,
                'r2': r2,
                'samuelson_present': samuelson_present,
                'mean_volatility': mean_volatility,
                'std_volatility': std_volatility,
                'volatility_window': kwargs.get('volatility_window', 20)
            }
            
            self.complete_run(accuracy)
            
            return {
                'status': 'success',
                'slope': slope,
                'intercept': intercept,
                'r2': r2,
                'samuelson_present': samuelson_present,
                'accuracy': accuracy
            }
            
        except Exception as e:
            self.error_run(str(e))
            return {'status': 'error', 'message': str(e)}
    
    def predict(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Generate Samuelson effect predictions.
        """
        if not self.parameters:
            return {'status': 'error', 'message': 'Model not trained'}
        
        try:
            prepared_data = self.prepare_data(data, **kwargs)
            
            predicted_volatilities = []
            volatility_signals = []
            
            for _, row in prepared_data.iterrows():
                log_ttm = row['log_time_to_maturity']
                
                if not np.isnan(log_ttm):
                    # Predict volatility
                    predicted_vol = self.parameters['intercept'] + self.parameters['slope'] * log_ttm
                    predicted_volatilities.append(max(0, predicted_vol))
                    
                    # Generate volatility-based signals
                    current_vol = row['volatility']
                    if not np.isnan(current_vol):
                        if predicted_vol > current_vol * 1.2:  # Expecting higher volatility
                            volatility_signals.append(1)  # Increase position/hedge
                        elif predicted_vol < current_vol * 0.8:  # Expecting lower volatility
                            volatility_signals.append(-1)  # Reduce position
                        else:
                            volatility_signals.append(0)  # Hold
                    else:
                        volatility_signals.append(0)
                else:
                    predicted_volatilities.append(None)
                    volatility_signals.append(0)
            
            return {
                'status': 'success',
                'predicted_volatilities': predicted_volatilities,
                'volatility_signals': volatility_signals,
                'actual_volatilities': prepared_data['volatility'].tolist(),
                'time_to_maturity': prepared_data['time_to_maturity'].tolist(),
                'timestamps': prepared_data['timestamp'].dt.isoformat().tolist()
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def evaluate(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        return self.calculate_common_metrics(actual, predicted)

class FuturesRLModel(BaseModel):
    """
    Reinforcement Learning model for futures trading using SAC (Soft Actor-Critic).
    """
    
    def __init__(self):
        super().__init__(
            model_id="futures_rl_sac",
            name="RL (SAC)",
            category=AssetCategory.FUTURES,
            model_type=ModelType.ML,
            description="Soft Actor-Critic RL model for futures trading"
        )
    
    def prepare_data(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Prepare data for RL training.
        """
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not self.validate_data(data, required_cols):
            raise ValueError("Invalid data format for Futures RL model")
        
        df = data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate features for RL state space
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Technical indicators
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'])
        
        # MACD
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # Volatility
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price position indicators
        df['price_position'] = (df['close'] - df['low'].rolling(window=20).min()) / \
                              (df['high'].rolling(window=20).max() - df['low'].rolling(window=20).min())
        
        # Normalize features for RL
        feature_cols = ['returns', 'rsi', 'macd', 'volatility', 'volume_ratio', 'price_position']
        for col in feature_cols:
            if col in df.columns:
                df[f'{col}_norm'] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
        
        return df.dropna()
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _create_rl_environment(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Create RL environment data.
        """
        # State features
        state_cols = [col for col in data.columns if col.endswith('_norm')]
        states = data[state_cols].values
        
        # Actions: -1 (sell), 0 (hold), 1 (buy)
        # Rewards: based on returns and risk-adjusted performance
        returns = data['returns'].values
        
        return {
            'states': states,
            'returns': returns,
            'prices': data['close'].values,
            'state_dim': len(state_cols)
        }
    
    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Train RL model (simplified implementation).
        """
        self.start_run()
        
        try:
            prepared_data = self.prepare_data(data)
            env_data = self._create_rl_environment(prepared_data)
            
            # Simplified RL training using Q-learning approximation
            from sklearn.ensemble import RandomForestRegressor
            
            states = env_data['states']
            returns = env_data['returns']
            
            # Create training data for Q-function approximation
            X_train = []
            y_train = []
            
            lookback = kwargs.get('lookback', 10)
            
            for i in range(lookback, len(states) - 1):
                state = states[i].flatten()
                
                # Calculate rewards for different actions
                next_return = returns[i + 1]
                
                # Action rewards (simplified)
                if next_return > 0.001:  # Positive return
                    buy_reward = next_return
                    sell_reward = -next_return
                    hold_reward = 0
                elif next_return < -0.001:  # Negative return
                    buy_reward = next_return
                    sell_reward = -next_return
                    hold_reward = 0
                else:  # Small return
                    buy_reward = -0.0001  # Small penalty for transaction costs
                    sell_reward = -0.0001
                    hold_reward = 0
                
                # Add training samples for each action
                for action, reward in [(0, sell_reward), (1, hold_reward), (2, buy_reward)]:
                    state_action = np.append(state, action)
                    X_train.append(state_action)
                    y_train.append(reward)
            
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            # Train Q-function approximator
            self.model_instance = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            self.model_instance.fit(X_train, y_train)
            
            # Evaluate on training data
            y_pred = self.model_instance.predict(X_train)
            from sklearn.metrics import r2_score
            r2 = r2_score(y_train, y_pred)
            
            # Calculate trading performance
            positions = self._generate_positions(env_data)
            portfolio_returns = positions[:-1] * returns[1:len(positions)]
            
            # Performance metrics
            total_return = np.sum(portfolio_returns)
            sharpe_ratio = np.mean(portfolio_returns) / (np.std(portfolio_returns) + 1e-8) * np.sqrt(252)
            max_drawdown = self._calculate_max_drawdown(np.cumsum(portfolio_returns))
            
            accuracy = max(0, (sharpe_ratio + 1) * 50)  # Convert Sharpe to accuracy-like metric
            
            self.parameters = {
                'state_dim': env_data['state_dim'],
                'lookback': lookback,
                'r2': r2,
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            }
            
            self.complete_run(accuracy)
            
            return {
                'status': 'success',
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'accuracy': accuracy
            }
            
        except Exception as e:
            self.error_run(str(e))
            return {'status': 'error', 'message': str(e)}
    
    def _generate_positions(self, env_data: Dict[str, Any]) -> np.ndarray:
        """Generate trading positions using trained model."""
        states = env_data['states']
        positions = []
        
        for i, state in enumerate(states):
            if self.model_instance is not None:
                # Get Q-values for all actions
                q_values = []
                for action in [0, 1, 2]:  # sell, hold, buy
                    state_action = np.append(state.flatten(), action).reshape(1, -1)
                    q_value = self.model_instance.predict(state_action)[0]
                    q_values.append(q_value)
                
                # Choose action with highest Q-value
                best_action = np.argmax(q_values)
                position = best_action - 1  # Convert to -1, 0, 1
                positions.append(position)
            else:
                positions.append(0)
        
        return np.array(positions)
    
    def _calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / (peak + 1e-8)
        return np.min(drawdown)
    
    def predict(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Generate RL-based trading signals.
        """
        if self.model_instance is None:
            return {'status': 'error', 'message': 'Model not trained'}
        
        try:
            prepared_data = self.prepare_data(data)
            env_data = self._create_rl_environment(prepared_data)
            
            positions = self._generate_positions(env_data)
            
            return {
                'status': 'success',
                'positions': positions.tolist(),
                'signals': positions.tolist(),  # Same as positions for this model
                'timestamps': prepared_data['timestamp'].dt.isoformat().tolist()
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def evaluate(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        metrics = self.calculate_common_metrics(actual, predicted)
        
        # Add RL-specific metrics
        if hasattr(self, 'parameters') and self.parameters:
            metrics['sharpe_ratio'] = self.parameters.get('sharpe_ratio', 0)
            metrics['max_drawdown'] = self.parameters.get('max_drawdown', 0)
        
        return metrics