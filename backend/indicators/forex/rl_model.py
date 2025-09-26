"""Reinforcement Learning Model for Forex Trading

This module implements reinforcement learning agents for forex trading,
including custom trading environments and policy optimization.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

# RL imports with fallback
try:
    import gym
    from stable_baselines3 import PPO
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    print("Reinforcement Learning libraries not available. Using simplified RL implementation.")

@dataclass
class RLResults:
    """Reinforcement learning results"""
    actions: List[str]
    rewards: List[float]
    portfolio_value: List[float]
    model_performance: Dict[str, float]
    policy_parameters: Dict[str, any]
    training_metrics: Dict[str, List[float]]

class ForexTradingEnvironment:
    """Forex trading environment for reinforcement learning"""
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000):
        self.data = data
        self.initial_balance = initial_balance
        self.reset()
        
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0  # -1: short, 0: neutral, 1: long
        self.entry_price = 0
        self.total_reward = 0
        self.trade_history = []
        return self._get_observation()
    
    def _get_observation(self):
        """Get current market observation"""
        if self.current_step >= len(self.data):
            return np.zeros(10)  # Default observation
        
        # Get recent market data (last 10 features)
        obs = self.data.iloc[self.current_step].values[:10]
        
        # Add position and balance info
        position_info = np.array([self.position, self.balance / self.initial_balance])
        
        return np.concatenate([obs, position_info])
    
    def step(self, action):
        """Execute trading action"""
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0, True, {}
        
        current_price = self.data.iloc[self.current_step]['exchange_rate']
        next_price = self.data.iloc[self.current_step + 1]['exchange_rate']
        
        reward = 0
        
        # Action: 0=hold, 1=buy, 2=sell
        if action == 1 and self.position <= 0:  # Buy
            if self.position == -1:  # Close short position
                reward += (self.entry_price - current_price) * abs(self.position)
            self.position = 1
            self.entry_price = current_price
            
        elif action == 2 and self.position >= 0:  # Sell
            if self.position == 1:  # Close long position
                reward += (current_price - self.entry_price) * abs(self.position)
            self.position = -1
            self.entry_price = current_price
        
        # Calculate unrealized P&L
        if self.position != 0:
            if self.position == 1:  # Long position
                unrealized_pnl = (next_price - self.entry_price) * 0.1  # Small reward for unrealized gains
            else:  # Short position
                unrealized_pnl = (self.entry_price - next_price) * 0.1
            reward += unrealized_pnl
        
        # Penalty for excessive trading
        if action != 0:
            reward -= 0.001  # Small transaction cost
        
        self.current_step += 1
        self.total_reward += reward
        
        done = self.current_step >= len(self.data) - 1
        
        return self._get_observation(), reward, done, {'balance': self.balance}
    
    def get_portfolio_value(self) -> float:
        """Get current portfolio value"""
        if self.current_step >= len(self.data):
            return self.balance
        
        current_price = self.data.iloc[self.current_step]['exchange_rate']
        
        if self.position == 1:  # Long position
            unrealized_pnl = (current_price - self.entry_price) * abs(self.position)
        elif self.position == -1:  # Short position
            unrealized_pnl = (self.entry_price - current_price) * abs(self.position)
        else:
            unrealized_pnl = 0
        
        return self.balance + unrealized_pnl

class RLForexAgent:
    """Reinforcement Learning agent for forex trading"""
    
    def __init__(self, env_data: pd.DataFrame):
        self.env_data = env_data
        self.agent = None
        self.training_rewards = []
        
    def fit(self, total_timesteps: int = 10000) -> RLResults:
        """Train RL agent"""
        if not RL_AVAILABLE:
            return self._fit_simplified()
        
        # Create environment
        env = ForexTradingEnvironment(self.env_data)
        
        # Create agent
        self.agent = PPO('MlpPolicy', env, verbose=0)
        
        # Train agent
        self.agent.learn(total_timesteps=total_timesteps)
        
        # Test agent
        obs = env.reset()
        actions = []
        rewards = []
        portfolio_values = [env.initial_balance]
        
        for _ in range(len(self.env_data) - 1):
            action, _ = self.agent.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            actions.append(['HOLD', 'BUY', 'SELL'][action])
            rewards.append(reward)
            portfolio_values.append(info.get('balance', portfolio_values[-1]))
            
            if done:
                break
        
        # Calculate performance metrics
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        volatility = np.std(np.diff(portfolio_values) / portfolio_values[:-1])
        sharpe_ratio = total_return / (volatility + 1e-8)
        max_drawdown = self._calculate_max_drawdown(portfolio_values)
        
        performance = {
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': np.mean(np.array(rewards) > 0)
        }
        
        return RLResults(
            actions=actions,
            rewards=rewards,
            portfolio_value=portfolio_values,
            model_performance=performance,
            policy_parameters={'algorithm': 'PPO', 'total_timesteps': total_timesteps},
            training_metrics={'rewards': self.training_rewards}
        )
    
    def _fit_simplified(self) -> RLResults:
        """Simplified RL implementation when libraries are not available"""
        # Simple momentum-based strategy
        actions = []
        rewards = []
        portfolio_values = [10000]
        
        position = 0
        entry_price = 0
        
        for i in range(1, len(self.env_data)):
            current_price = self.env_data.iloc[i]['exchange_rate']
            prev_price = self.env_data.iloc[i-1]['exchange_rate']
            
            # Simple momentum strategy
            if i >= 5:
                recent_trend = np.mean(np.diff(self.env_data.iloc[i-5:i]['exchange_rate']))
                
                if recent_trend > 0.001 and position <= 0:
                    action = 'BUY'
                    if position == -1:
                        reward = (entry_price - current_price) * 100
                        portfolio_values.append(portfolio_values[-1] + reward)
                    position = 1
                    entry_price = current_price
                    
                elif recent_trend < -0.001 and position >= 0:
                    action = 'SELL'
                    if position == 1:
                        reward = (current_price - entry_price) * 100
                        portfolio_values.append(portfolio_values[-1] + reward)
                    position = -1
                    entry_price = current_price
                    
                else:
                    action = 'HOLD'
                    reward = 0
                    portfolio_values.append(portfolio_values[-1])
            else:
                action = 'HOLD'
                reward = 0
                portfolio_values.append(portfolio_values[-1])
            
            actions.append(action)
            rewards.append(reward)
        
        # Performance metrics
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        volatility = np.std(np.diff(portfolio_values) / portfolio_values[:-1])
        sharpe_ratio = total_return / (volatility + 1e-8)
        max_drawdown = self._calculate_max_drawdown(portfolio_values)
        
        performance = {
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': np.mean(np.array(rewards) > 0)
        }
        
        return RLResults(
            actions=actions,
            rewards=rewards,
            portfolio_value=portfolio_values,
            model_performance=performance,
            policy_parameters={'algorithm': 'simplified_momentum'},
            training_metrics={'rewards': rewards}
        )
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown"""
        peak = portfolio_values[0]
        max_dd = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def predict(self, observation: np.ndarray) -> Tuple[int, float]:
        """Make prediction using trained agent"""
        if self.agent is None:
            # Simple rule-based prediction
            if len(observation) > 2:
                recent_change = observation[0] - observation[1] if len(observation) > 1 else 0
                if recent_change > 0.001:
                    return 1, 0.7  # BUY with confidence
                elif recent_change < -0.001:
                    return 2, 0.7  # SELL with confidence
                else:
                    return 0, 0.5  # HOLD with neutral confidence
            return 0, 0.5
        
        action, _ = self.agent.predict(observation, deterministic=True)
        confidence = 0.8  # Default confidence for trained agent
        return action, confidence
    
    def evaluate_strategy(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate trading strategy on test data"""
        env = ForexTradingEnvironment(test_data)
        obs = env.reset()
        
        total_rewards = []
        portfolio_values = [env.initial_balance]
        
        for _ in range(len(test_data) - 1):
            action, confidence = self.predict(obs)
            obs, reward, done, info = env.step(action)
            
            total_rewards.append(reward)
            portfolio_values.append(env.get_portfolio_value())
            
            if done:
                break
        
        # Calculate evaluation metrics
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        volatility = np.std(np.diff(portfolio_values) / portfolio_values[:-1])
        sharpe_ratio = total_return / (volatility + 1e-8)
        max_drawdown = self._calculate_max_drawdown(portfolio_values)
        
        return {
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': np.mean(np.array(total_rewards) > 0),
            'avg_reward': np.mean(total_rewards),
            'total_trades': len([r for r in total_rewards if r != 0])
        }
    
    def plot_performance(self, results: RLResults):
        """Plot RL agent performance"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Portfolio value over time
            axes[0, 0].plot(results.portfolio_value, color='green', linewidth=2)
            axes[0, 0].axhline(y=results.portfolio_value[0], color='red', linestyle='--', alpha=0.5)
            axes[0, 0].set_title('Portfolio Value Over Time')
            axes[0, 0].set_xlabel('Time Steps')
            axes[0, 0].set_ylabel('Portfolio Value')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Cumulative rewards
            cumulative_rewards = np.cumsum(results.rewards)
            axes[0, 1].plot(cumulative_rewards, color='blue', linewidth=2)
            axes[0, 1].set_title('Cumulative Rewards')
            axes[0, 1].set_xlabel('Time Steps')
            axes[0, 1].set_ylabel('Cumulative Reward')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Action distribution
            action_counts = {action: results.actions.count(action) for action in set(results.actions)}
            colors = {'BUY': 'green', 'SELL': 'red', 'HOLD': 'gray'}
            action_colors = [colors.get(action, 'blue') for action in action_counts.keys()]
            
            axes[1, 0].pie(action_counts.values(), labels=action_counts.keys(), 
                          colors=action_colors, autopct='%1.1f%%', startangle=90)
            axes[1, 0].set_title('Action Distribution')
            
            # Reward distribution
            axes[1, 1].hist(results.rewards, bins=30, alpha=0.7, color='purple')
            axes[1, 1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
            axes[1, 1].set_title('Reward Distribution')
            axes[1, 1].set_xlabel('Reward')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for plotting")

# Example usage
if __name__ == "__main__":
    # Generate sample forex data
    np.random.seed(42)
    n_points = 200
    
    # Simulate exchange rate with trend and volatility
    base_rate = 1.2
    trend = np.linspace(0, 0.1, n_points)
    noise = np.random.normal(0, 0.02, n_points)
    exchange_rate = base_rate + trend + noise
    
    # Create sample data with required columns
    rl_data = pd.DataFrame({
        'exchange_rate': exchange_rate,
        'volatility': np.random.exponential(0.02, n_points),
        'volume': np.random.lognormal(10, 0.5, n_points),
        'rsi': np.random.uniform(20, 80, n_points),
        'macd': np.random.normal(0, 0.01, n_points),
        'interest_rate_diff': np.random.normal(0, 0.005, n_points),
        'inflation_diff': np.random.normal(0, 0.01, n_points)
    })
    
    # Initialize and train RL agent
    rl_agent = RLForexAgent(rl_data)
    
    try:
        print("Training RL agent...")
        results = rl_agent.fit(total_timesteps=5000)
        
        print("\n=== RL Agent Results ===")
        print(f"Total Return: {results.model_performance['total_return']:.3f}")
        print(f"Volatility: {results.model_performance['volatility']:.3f}")
        print(f"Sharpe Ratio: {results.model_performance['sharpe_ratio']:.3f}")
        print(f"Max Drawdown: {results.model_performance['max_drawdown']:.3f}")
        print(f"Win Rate: {results.model_performance['win_rate']:.3f}")
        
        # Action distribution
        action_counts = {action: results.actions.count(action) for action in set(results.actions)}
        print("\nAction Distribution:")
        for action, count in action_counts.items():
            percentage = (count / len(results.actions)) * 100
            print(f"  {action}: {count} ({percentage:.1f}%)")
        
        print(f"\nFinal Portfolio Value: ${results.portfolio_value[-1]:.2f}")
        print(f"Initial Portfolio Value: ${results.portfolio_value[0]:.2f}")
        
        # Test on new data (last 50 points)
        test_data = rl_data.tail(50)
        eval_results = rl_agent.evaluate_strategy(test_data)
        print("\n=== Evaluation on Test Data ===")
        for metric, value in eval_results.items():
            print(f"{metric}: {value:.4f}")
        
        print("\nRL agent training completed successfully!")
        
    except Exception as e:
        print(f"RL agent training failed: {e}")
        import traceback
        traceback.print_exc()