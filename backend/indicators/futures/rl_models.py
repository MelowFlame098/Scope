"""Reinforcement Learning Models for Futures Trading

This module provides reinforcement learning agents and environments for futures trading,
including PPO, SAC, and DDPG agents with a custom trading environment.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import warnings

# Conditional imports for RL
try:
    import gym
    from gym import spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    warnings.warn("Gym not available. RL functionality will be limited.")

try:
    from stable_baselines3 import PPO, SAC, DDPG
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    warnings.warn("Stable-baselines3 not available. Using simple rule-based agents.")

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    warnings.warn("TA-Lib not available. Using custom implementations.")

@dataclass
class FuturesData:
    """Data structure for futures market data"""
    prices: List[float]
    returns: List[float]
    volume: List[float]
    open_interest: List[float]
    timestamps: List[datetime]
    high: List[float]
    low: List[float]
    open: List[float]
    close: List[float]
    contract_symbol: str
    underlying_asset: str

@dataclass
class RLAgentResult:
    """Results from RL agent analysis"""
    agent_type: str
    actions: List[int]
    rewards: List[float]
    cumulative_returns: List[float]
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_trade_return: float

class SimpleFuturesTradingEnv:
    """Simplified trading environment for RL agents"""
    
    def __init__(self, futures_data: FuturesData, initial_balance: float = 10000.0):
        self.futures_data = futures_data
        self.initial_balance = initial_balance
        self.reset()
        
        # Action space: 0=Hold, 1=Buy, 2=Sell
        if GYM_AVAILABLE:
            self.action_space = spaces.Discrete(3)
            # Observation space: [price_change, volume_change, rsi]
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
            )
        
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0  # 0=no position, 1=long, -1=short
        self.entry_price = 0.0
        self.total_trades = 0
        self.trade_returns = []
        
        return self._get_observation()
    
    def step(self, action: int):
        """Execute one step in the environment"""
        if self.current_step >= len(self.futures_data.close) - 1:
            return self._get_observation(), 0, True, {}
        
        current_price = self.futures_data.close[self.current_step]
        next_price = self.futures_data.close[self.current_step + 1]
        
        reward = 0
        
        # Execute action
        if action == 1:  # Buy
            if self.position <= 0:  # Close short or open long
                if self.position == -1:  # Close short position
                    trade_return = (self.entry_price - current_price) / self.entry_price
                    self.balance *= (1 + trade_return)
                    self.trade_returns.append(trade_return)
                    self.total_trades += 1
                
                # Open long position
                self.position = 1
                self.entry_price = current_price
        
        elif action == 2:  # Sell
            if self.position >= 0:  # Close long or open short
                if self.position == 1:  # Close long position
                    trade_return = (current_price - self.entry_price) / self.entry_price
                    self.balance *= (1 + trade_return)
                    self.trade_returns.append(trade_return)
                    self.total_trades += 1
                
                # Open short position
                self.position = -1
                self.entry_price = current_price
        
        # Calculate reward based on position and price movement
        price_change = (next_price - current_price) / current_price
        
        if self.position == 1:  # Long position
            reward = price_change
        elif self.position == -1:  # Short position
            reward = -price_change
        else:  # No position
            reward = 0
        
        # Add small penalty for excessive trading
        if action != 0:  # Not holding
            reward -= 0.001  # Transaction cost
        
        self.current_step += 1
        done = self.current_step >= len(self.futures_data.close) - 1
        
        return self._get_observation(), reward, done, {}
    
    def _get_observation(self):
        """Get current observation"""
        if self.current_step == 0:
            return np.array([0.0, 0.0, 50.0], dtype=np.float32)
        
        # Price change
        price_change = (
            self.futures_data.close[self.current_step] - 
            self.futures_data.close[self.current_step - 1]
        ) / self.futures_data.close[self.current_step - 1]
        
        # Volume change
        if self.current_step > 0:
            volume_change = (
                self.futures_data.volume[self.current_step] - 
                self.futures_data.volume[self.current_step - 1]
            ) / self.futures_data.volume[self.current_step - 1]
        else:
            volume_change = 0.0
        
        # Simple RSI calculation
        rsi = self._calculate_simple_rsi()
        
        return np.array([price_change, volume_change, rsi], dtype=np.float32)
    
    def _calculate_simple_rsi(self, period: int = 14) -> float:
        """Calculate simple RSI for current observation"""
        if self.current_step < period:
            return 50.0
        
        prices = self.futures_data.close[self.current_step - period:self.current_step + 1]
        deltas = np.diff(prices)
        
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

class SimpleRLAgent:
    """Simple RL agent with multiple algorithm support"""
    
    def __init__(self, agent_type: str = "PPO"):
        self.agent_type = agent_type
        self.model = None
        self.is_trained = False
    
    def train(self, env: SimpleFuturesTradingEnv, total_timesteps: int = 10000):
        """Train the RL agent"""
        
        if not SB3_AVAILABLE:
            print(f"Stable-baselines3 not available. Using rule-based {self.agent_type} agent.")
            self.is_trained = True
            return
        
        try:
            # Create vectorized environment
            vec_env = DummyVecEnv([lambda: env])
            
            # Initialize model based on agent type
            if self.agent_type == "PPO":
                self.model = PPO("MlpPolicy", vec_env, verbose=0)
            elif self.agent_type == "SAC":
                self.model = SAC("MlpPolicy", vec_env, verbose=0)
            elif self.agent_type == "DDPG":
                self.model = DDPG("MlpPolicy", vec_env, verbose=0)
            else:
                print(f"Unknown agent type: {self.agent_type}. Using PPO.")
                self.model = PPO("MlpPolicy", vec_env, verbose=0)
            
            # Train the model
            self.model.learn(total_timesteps=total_timesteps)
            self.is_trained = True
            
        except Exception as e:
            print(f"Training failed for {self.agent_type}: {e}")
            print("Falling back to rule-based agent.")
            self.is_trained = True
    
    def predict(self, observation: np.ndarray) -> int:
        """Predict action given observation"""
        
        if self.model is not None and SB3_AVAILABLE:
            try:
                action, _ = self.model.predict(observation, deterministic=True)
                return int(action)
            except Exception:
                pass
        
        # Fallback rule-based strategy
        return self._rule_based_action(observation)
    
    def _rule_based_action(self, observation: np.ndarray) -> int:
        """Simple rule-based action selection"""
        
        price_change, volume_change, rsi = observation
        
        # Simple momentum + mean reversion strategy
        if self.agent_type == "PPO":
            # Momentum strategy
            if price_change > 0.01 and rsi < 70:
                return 1  # Buy
            elif price_change < -0.01 and rsi > 30:
                return 2  # Sell
            else:
                return 0  # Hold
        
        elif self.agent_type == "SAC":
            # Mean reversion strategy
            if rsi > 70:
                return 2  # Sell (overbought)
            elif rsi < 30:
                return 1  # Buy (oversold)
            else:
                return 0  # Hold
        
        elif self.agent_type == "DDPG":
            # Combined strategy
            momentum_signal = 1 if price_change > 0.005 else (-1 if price_change < -0.005 else 0)
            mean_reversion_signal = -1 if rsi > 70 else (1 if rsi < 30 else 0)
            
            combined_signal = momentum_signal + mean_reversion_signal
            
            if combined_signal > 0:
                return 1  # Buy
            elif combined_signal < 0:
                return 2  # Sell
            else:
                return 0  # Hold
        
        else:
            return 0  # Default hold

class RLAnalyzer:
    """Reinforcement Learning analyzer for futures trading"""
    
    def __init__(self):
        self.agents = {}
    
    def analyze(self, futures_data: FuturesData, 
               agent_types: List[str] = ["PPO", "SAC", "DDPG"],
               train_timesteps: int = 5000) -> Dict[str, RLAgentResult]:
        """Analyze using multiple RL agents"""
        
        results = {}
        
        for agent_type in agent_types:
            try:
                result = self._analyze_single_agent(
                    futures_data, agent_type, train_timesteps
                )
                results[agent_type] = result
            except Exception as e:
                print(f"Error analyzing {agent_type} agent: {e}")
                # Create default result
                results[agent_type] = self._create_default_result(
                    agent_type, len(futures_data.close)
                )
        
        return results
    
    def _analyze_single_agent(self, futures_data: FuturesData, 
                            agent_type: str, train_timesteps: int) -> RLAgentResult:
        """Analyze using a single RL agent"""
        
        # Create environment
        env = SimpleFuturesTradingEnv(futures_data)
        
        # Create and train agent
        agent = SimpleRLAgent(agent_type)
        agent.train(env, train_timesteps)
        
        # Evaluate agent
        env.reset()
        actions = []
        rewards = []
        observations = []
        
        obs = env._get_observation()
        done = False
        
        while not done:
            action = agent.predict(obs)
            obs, reward, done, _ = env.step(action)
            
            actions.append(action)
            rewards.append(reward)
            observations.append(obs)
        
        # Calculate performance metrics
        cumulative_returns = self._calculate_cumulative_returns(rewards)
        sharpe_ratio = self._calculate_sharpe_ratio(rewards)
        max_drawdown = self._calculate_max_drawdown(cumulative_returns)
        win_rate = self._calculate_win_rate(env.trade_returns)
        avg_trade_return = np.mean(env.trade_returns) if env.trade_returns else 0.0
        
        return RLAgentResult(
            agent_type=agent_type,
            actions=actions,
            rewards=rewards,
            cumulative_returns=cumulative_returns,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=env.total_trades,
            avg_trade_return=avg_trade_return
        )
    
    def _calculate_cumulative_returns(self, rewards: List[float]) -> List[float]:
        """Calculate cumulative returns from rewards"""
        if not rewards:
            return [0.0]
        
        cumulative = [rewards[0]]
        for reward in rewards[1:]:
            cumulative.append(cumulative[-1] + reward)
        
        return cumulative
    
    def _calculate_sharpe_ratio(self, rewards: List[float]) -> float:
        """Calculate Sharpe ratio from rewards"""
        if len(rewards) < 2:
            return 0.0
        
        mean_return = np.mean(rewards)
        std_return = np.std(rewards)
        
        if std_return == 0:
            return 0.0
        
        # Annualized Sharpe ratio (assuming daily data)
        return (mean_return / std_return) * np.sqrt(252)
    
    def _calculate_max_drawdown(self, cumulative_returns: List[float]) -> float:
        """Calculate maximum drawdown"""
        if len(cumulative_returns) < 2:
            return 0.0
        
        peak = cumulative_returns[0]
        max_dd = 0.0
        
        for value in cumulative_returns[1:]:
            if value > peak:
                peak = value
            else:
                drawdown = (peak - value) / peak if peak != 0 else 0
                max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _calculate_win_rate(self, trade_returns: List[float]) -> float:
        """Calculate win rate from trade returns"""
        if len(trade_returns) == 0:
            return 0.0
        
        winning_trades = sum(1 for ret in trade_returns if ret > 0)
        return winning_trades / len(trade_returns)
    
    def _create_default_result(self, agent_type: str, n_periods: int) -> RLAgentResult:
        """Create default result for failed analysis"""
        return RLAgentResult(
            agent_type=agent_type,
            actions=[0] * n_periods,
            rewards=[0.0] * n_periods,
            cumulative_returns=[0.0] * n_periods,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            total_trades=0,
            avg_trade_return=0.0
        )

# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    import random
    from datetime import timedelta
    
    n_periods = 100
    base_price = 100.0
    
    # Generate sample price data with some trend
    prices = [base_price]
    for i in range(n_periods - 1):
        # Add some trend and noise
        trend = 0.001 * np.sin(i / 10)  # Cyclical trend
        noise = random.uniform(-0.02, 0.02)
        change = trend + noise
        prices.append(prices[-1] * (1 + change))
    
    # Generate OHLC data
    high = [p * (1 + abs(random.uniform(0, 0.01))) for p in prices]
    low = [p * (1 - abs(random.uniform(0, 0.01))) for p in prices]
    open_prices = [prices[0]] + prices[:-1]
    
    # Generate other data
    volume = [random.uniform(1000, 10000) for _ in range(n_periods)]
    open_interest = [random.uniform(5000, 15000) for _ in range(n_periods)]
    returns = [(prices[i] - prices[i-1]) / prices[i-1] if i > 0 else 0 
              for i in range(n_periods)]
    
    timestamps = [datetime.now() + timedelta(days=i) for i in range(n_periods)]
    
    # Create FuturesData
    futures_data = FuturesData(
        prices=prices,
        returns=returns,
        volume=volume,
        open_interest=open_interest,
        timestamps=timestamps,
        high=high,
        low=low,
        open=open_prices,
        close=prices,
        contract_symbol="TEST_2024_03",
        underlying_asset="Test Asset"
    )
    
    # Test RL analysis
    analyzer = RLAnalyzer()
    results = analyzer.analyze(futures_data, train_timesteps=1000)
    
    print(f"RL Analysis Results:")
    for agent_type, result in results.items():
        print(f"\n{agent_type} Agent:")
        print(f"  Total Return: {result.cumulative_returns[-1]:.2%}")
        print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {result.max_drawdown:.2%}")
        print(f"  Win Rate: {result.win_rate:.1%}")
        print(f"  Total Trades: {result.total_trades}")
        print(f"  Avg Trade Return: {result.avg_trade_return:.2%}")