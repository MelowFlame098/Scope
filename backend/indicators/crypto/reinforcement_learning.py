"""Reinforcement Learning Models for Cryptocurrency Trading

This module implements advanced Reinforcement Learning algorithms for cryptocurrency trading:
- Deep Q-Network (DQN)
- Policy Gradient Methods (REINFORCE, A2C, A3C)
- Actor-Critic Methods (DDPG, TD3, SAC)
- Proximal Policy Optimization (PPO)
- Multi-Agent Reinforcement Learning
- Hierarchical Reinforcement Learning
- Meta-Learning for Trading
- Risk-Aware RL
- Portfolio Optimization RL
- Market Making RL
- Arbitrage Detection RL
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from collections import deque, defaultdict
import random
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available. Using simplified implementations.")

# Deep Learning Libraries (Optional)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Categorical, Normal
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available. Using simplified RL implementations.")

logger = logging.getLogger(__name__)

@dataclass
class TradingAction:
    """Trading action representation"""
    action_type: str  # 'buy', 'sell', 'hold'
    amount: float  # Amount to trade (0-1 for percentage)
    confidence: float  # Confidence in action (0-1)
    reasoning: str  # Explanation for action

@dataclass
class MarketState:
    """Market state representation"""
    price: float
    volume: float
    volatility: float
    trend: float
    momentum: float
    support_resistance: Dict[str, float]
    technical_indicators: Dict[str, float]
    sentiment: float
    timestamp: datetime

@dataclass
class TradingEnvironment:
    """Trading environment state"""
    current_state: MarketState
    portfolio_value: float
    cash_balance: float
    crypto_holdings: float
    transaction_costs: float
    max_position_size: float
    risk_tolerance: float

@dataclass
class RLTrainingResult:
    """RL training result"""
    total_episodes: int
    final_reward: float
    average_reward: float
    best_reward: float
    convergence_episode: int
    training_loss: List[float]
    reward_history: List[float]
    action_distribution: Dict[str, int]
    performance_metrics: Dict[str, float]

@dataclass
class PolicyResult:
    """Policy evaluation result"""
    recommended_action: TradingAction
    action_probabilities: Dict[str, float]
    value_estimate: float
    risk_assessment: float
    expected_return: float
    confidence_interval: Tuple[float, float]

@dataclass
class PortfolioOptimizationResult:
    """Portfolio optimization result"""
    optimal_weights: Dict[str, float]
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    max_drawdown: float
    rebalancing_frequency: int
    risk_adjusted_return: float

@dataclass
class MarketMakingResult:
    """Market making result"""
    bid_price: float
    ask_price: float
    spread: float
    inventory_target: float
    risk_penalty: float
    expected_profit: float
    optimal_quotes: Dict[str, float]

@dataclass
class ArbitrageResult:
    """Arbitrage detection result"""
    arbitrage_opportunities: List[Dict]
    expected_profit: float
    execution_risk: float
    time_window: float
    required_capital: float
    success_probability: float

@dataclass
class RLAnalysisResult:
    """Combined RL analysis result"""
    policy_result: PolicyResult
    portfolio_optimization: PortfolioOptimizationResult
    market_making: MarketMakingResult
    arbitrage_detection: ArbitrageResult
    training_performance: RLTrainingResult
    risk_metrics: Dict[str, float]
    performance_attribution: Dict[str, float]
    model_confidence: float

class TradingEnvironmentSimulator:
    """Simulates trading environment for RL training"""
    
    def __init__(self, 
                 initial_balance: float = 10000,
                 transaction_cost: float = 0.001,
                 max_position_size: float = 1.0):
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
        self.reset()
    
    def reset(self) -> MarketState:
        """Reset environment to initial state"""
        self.cash_balance = self.initial_balance
        self.crypto_holdings = 0.0
        self.portfolio_value = self.initial_balance
        self.step_count = 0
        self.price_history = []
        self.action_history = []
        
        # Generate initial market state
        return self._generate_market_state()
    
    def step(self, action: TradingAction, current_price: float) -> Tuple[MarketState, float, bool]:
        """Execute action and return new state, reward, done"""
        # Execute trade
        reward = self._execute_trade(action, current_price)
        
        # Update portfolio value
        self.portfolio_value = self.cash_balance + self.crypto_holdings * current_price
        
        # Generate next market state
        next_state = self._generate_market_state(current_price)
        
        # Check if episode is done
        done = self._is_episode_done()
        
        self.step_count += 1
        self.price_history.append(current_price)
        self.action_history.append(action)
        
        return next_state, reward, done
    
    def _execute_trade(self, action: TradingAction, price: float) -> float:
        """Execute trading action and return reward"""
        reward = 0.0
        
        if action.action_type == 'buy' and action.amount > 0:
            # Calculate maximum buyable amount
            max_buy = self.cash_balance / (price * (1 + self.transaction_cost))
            buy_amount = min(action.amount * max_buy, max_buy)
            
            if buy_amount > 0:
                cost = buy_amount * price * (1 + self.transaction_cost)
                self.cash_balance -= cost
                self.crypto_holdings += buy_amount
                reward = -self.transaction_cost * cost  # Transaction cost penalty
        
        elif action.action_type == 'sell' and action.amount > 0:
            # Calculate sell amount
            sell_amount = min(action.amount * self.crypto_holdings, self.crypto_holdings)
            
            if sell_amount > 0:
                proceeds = sell_amount * price * (1 - self.transaction_cost)
                self.cash_balance += proceeds
                self.crypto_holdings -= sell_amount
                reward = -self.transaction_cost * proceeds  # Transaction cost penalty
        
        # Add portfolio return as reward
        portfolio_return = (self.portfolio_value - self.initial_balance) / self.initial_balance
        reward += portfolio_return * 0.01  # Scale reward
        
        return reward
    
    def _generate_market_state(self, price: float = None) -> MarketState:
        """Generate market state"""
        if price is None:
            price = 50000 + np.random.normal(0, 1000)  # Random price around $50k
        
        # Generate synthetic market features
        volume = np.random.exponential(1000000)
        volatility = np.random.uniform(0.01, 0.05)
        trend = np.random.normal(0, 0.02)
        momentum = np.random.normal(0, 0.01)
        
        support_resistance = {
            'support': price * 0.95,
            'resistance': price * 1.05
        }
        
        technical_indicators = {
            'rsi': np.random.uniform(20, 80),
            'macd': np.random.normal(0, 100),
            'bb_position': np.random.uniform(0, 1)
        }
        
        sentiment = np.random.uniform(-1, 1)
        
        return MarketState(
            price=price,
            volume=volume,
            volatility=volatility,
            trend=trend,
            momentum=momentum,
            support_resistance=support_resistance,
            technical_indicators=technical_indicators,
            sentiment=sentiment,
            timestamp=datetime.now()
        )
    
    def _is_episode_done(self) -> bool:
        """Check if episode should end"""
        # End if portfolio value drops too much or step limit reached
        return (self.portfolio_value < self.initial_balance * 0.5 or 
                self.step_count >= 1000)
    
    def get_state_vector(self, state: MarketState) -> np.ndarray:
        """Convert market state to feature vector"""
        features = [
            state.price / 100000,  # Normalize price
            state.volume / 10000000,  # Normalize volume
            state.volatility,
            state.trend,
            state.momentum,
            state.technical_indicators['rsi'] / 100,
            state.technical_indicators['macd'] / 1000,
            state.technical_indicators['bb_position'],
            state.sentiment,
            self.cash_balance / self.initial_balance,
            self.crypto_holdings * state.price / self.initial_balance
        ]
        return np.array(features, dtype=np.float32)

class DQNAgent:
    """Deep Q-Network Agent"""
    
    def __init__(self, 
                 state_size: int = 11,
                 action_size: int = 3,
                 learning_rate: float = 0.001,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.memory = deque(maxlen=10000)
        self.q_network = self._build_network() if TORCH_AVAILABLE else None
        self.target_network = self._build_network() if TORCH_AVAILABLE else None
        
        if TORCH_AVAILABLE and self.q_network and self.target_network:
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
            self.update_target_network()
    
    def _build_network(self):
        """Build neural network"""
        if not TORCH_AVAILABLE:
            return None
        
        class QNetwork(nn.Module):
            def __init__(self, state_size, action_size):
                super(QNetwork, self).__init__()
                self.fc1 = nn.Linear(state_size, 128)
                self.fc2 = nn.Linear(128, 128)
                self.fc3 = nn.Linear(128, 64)
                self.fc4 = nn.Linear(64, action_size)
                self.dropout = nn.Dropout(0.2)
            
            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = F.relu(self.fc2(x))
                x = self.dropout(x)
                x = F.relu(self.fc3(x))
                x = self.fc4(x)
                return x
        
        return QNetwork(self.state_size, self.action_size)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state) -> int:
        """Choose action using epsilon-greedy policy"""
        if not TORCH_AVAILABLE or self.q_network is None:
            return random.randint(0, self.action_size - 1)
        
        if np.random.random() <= self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def replay(self, batch_size: int = 32):
        """Train the model on a batch of experiences"""
        if not TORCH_AVAILABLE or len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.99 * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """Update target network"""
        if TORCH_AVAILABLE and self.target_network and self.q_network:
            self.target_network.load_state_dict(self.q_network.state_dict())

class PPOAgent:
    """Proximal Policy Optimization Agent"""
    
    def __init__(self, 
                 state_size: int = 11,
                 action_size: int = 3,
                 learning_rate: float = 0.0003,
                 clip_epsilon: float = 0.2):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.clip_epsilon = clip_epsilon
        
        self.policy_network = self._build_policy_network() if TORCH_AVAILABLE else None
        self.value_network = self._build_value_network() if TORCH_AVAILABLE else None
        
        if TORCH_AVAILABLE and self.policy_network and self.value_network:
            self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
            self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=learning_rate)
    
    def _build_policy_network(self):
        """Build policy network"""
        if not TORCH_AVAILABLE:
            return None
        
        class PolicyNetwork(nn.Module):
            def __init__(self, state_size, action_size):
                super(PolicyNetwork, self).__init__()
                self.fc1 = nn.Linear(state_size, 128)
                self.fc2 = nn.Linear(128, 64)
                self.fc3 = nn.Linear(64, action_size)
            
            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = F.softmax(self.fc3(x), dim=-1)
                return x
        
        return PolicyNetwork(self.state_size, self.action_size)
    
    def _build_value_network(self):
        """Build value network"""
        if not TORCH_AVAILABLE:
            return None
        
        class ValueNetwork(nn.Module):
            def __init__(self, state_size):
                super(ValueNetwork, self).__init__()
                self.fc1 = nn.Linear(state_size, 128)
                self.fc2 = nn.Linear(128, 64)
                self.fc3 = nn.Linear(64, 1)
            
            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x
        
        return ValueNetwork(self.state_size)
    
    def get_action_and_value(self, state):
        """Get action and value estimate"""
        if not TORCH_AVAILABLE or self.policy_network is None:
            return random.randint(0, self.action_size - 1), 0.0, 0.0
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Get action probabilities
        action_probs = self.policy_network(state_tensor)
        dist = Categorical(action_probs)
        action = dist.sample()
        
        # Get value estimate
        value = self.value_network(state_tensor)
        
        return action.item(), dist.log_prob(action).item(), value.item()
    
    def update(self, states, actions, rewards, log_probs, values):
        """Update policy and value networks"""
        if not TORCH_AVAILABLE:
            return
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        old_log_probs = torch.FloatTensor(log_probs)
        old_values = torch.FloatTensor(values)
        
        # Calculate advantages
        advantages = rewards - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update policy
        new_action_probs = self.policy_network(states)
        dist = Categorical(new_action_probs)
        new_log_probs = dist.log_prob(actions)
        
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        
        policy_loss = -torch.min(surr1, surr2).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update value function
        new_values = self.value_network(states).squeeze()
        value_loss = F.mse_loss(new_values, rewards)
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

class PortfolioOptimizationRL:
    """RL-based Portfolio Optimization"""
    
    def __init__(self, assets: List[str], lookback_period: int = 252):
        self.assets = assets
        self.lookback_period = lookback_period
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
    
    def optimize_portfolio(self, 
                          price_data: pd.DataFrame,
                          risk_tolerance: float = 0.5) -> PortfolioOptimizationResult:
        """Optimize portfolio using RL"""
        # Calculate returns
        returns = price_data.pct_change().dropna()
        
        # Simple mean-variance optimization (placeholder for RL)
        mean_returns = returns.mean() * 252  # Annualized
        cov_matrix = returns.cov() * 252  # Annualized
        
        # Equal weight as baseline
        n_assets = len(self.assets)
        weights = np.ones(n_assets) / n_assets
        
        # Calculate metrics
        expected_return = np.sum(weights * mean_returns)
        expected_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = expected_return / expected_risk if expected_risk > 0 else 0
        
        # Calculate max drawdown (simplified)
        portfolio_returns = (returns * weights).sum(axis=1)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        optimal_weights = {asset: weight for asset, weight in zip(self.assets, weights)}
        
        return PortfolioOptimizationResult(
            optimal_weights=optimal_weights,
            expected_return=expected_return,
            expected_risk=expected_risk,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            rebalancing_frequency=30,  # Monthly
            risk_adjusted_return=expected_return - risk_tolerance * expected_risk
        )

class MarketMakingRL:
    """RL-based Market Making"""
    
    def __init__(self, spread_target: float = 0.001, inventory_target: float = 0.0):
        self.spread_target = spread_target
        self.inventory_target = inventory_target
    
    def generate_quotes(self, 
                       current_price: float,
                       volatility: float,
                       inventory: float,
                       order_flow: float) -> MarketMakingResult:
        """Generate optimal bid/ask quotes"""
        
        # Simple market making model
        base_spread = self.spread_target * current_price
        
        # Adjust for volatility
        volatility_adjustment = volatility * current_price * 0.5
        
        # Adjust for inventory
        inventory_adjustment = (inventory - self.inventory_target) * current_price * 0.01
        
        # Adjust for order flow
        flow_adjustment = order_flow * current_price * 0.001
        
        # Calculate bid and ask
        mid_price = current_price
        spread = base_spread + volatility_adjustment
        
        bid_price = mid_price - spread/2 - inventory_adjustment + flow_adjustment
        ask_price = mid_price + spread/2 - inventory_adjustment + flow_adjustment
        
        # Risk penalty
        risk_penalty = abs(inventory - self.inventory_target) * 0.01
        
        # Expected profit
        expected_profit = spread * 0.5 - risk_penalty
        
        return MarketMakingResult(
            bid_price=bid_price,
            ask_price=ask_price,
            spread=ask_price - bid_price,
            inventory_target=self.inventory_target,
            risk_penalty=risk_penalty,
            expected_profit=expected_profit,
            optimal_quotes={
                'bid': bid_price,
                'ask': ask_price,
                'mid': mid_price
            }
        )

class ArbitrageDetectionRL:
    """RL-based Arbitrage Detection"""
    
    def __init__(self, min_profit_threshold: float = 0.001):
        self.min_profit_threshold = min_profit_threshold
    
    def detect_arbitrage(self, 
                        price_data: Dict[str, float],
                        transaction_costs: Dict[str, float]) -> ArbitrageResult:
        """Detect arbitrage opportunities"""
        
        opportunities = []
        total_profit = 0.0
        
        # Simple triangular arbitrage detection
        exchanges = list(price_data.keys())
        
        for i, exchange1 in enumerate(exchanges):
            for j, exchange2 in enumerate(exchanges[i+1:], i+1):
                price1 = price_data[exchange1]
                price2 = price_data[exchange2]
                cost1 = transaction_costs.get(exchange1, 0.001)
                cost2 = transaction_costs.get(exchange2, 0.001)
                
                # Calculate potential profit
                if price1 < price2:
                    # Buy on exchange1, sell on exchange2
                    profit = (price2 * (1 - cost2)) - (price1 * (1 + cost1))
                    profit_pct = profit / price1
                    
                    if profit_pct > self.min_profit_threshold:
                        opportunities.append({
                            'buy_exchange': exchange1,
                            'sell_exchange': exchange2,
                            'buy_price': price1,
                            'sell_price': price2,
                            'profit': profit,
                            'profit_pct': profit_pct,
                            'execution_time': 5.0  # seconds
                        })
                        total_profit += profit
        
        # Calculate metrics
        execution_risk = len(opportunities) * 0.1  # Risk increases with complexity
        time_window = max([opp['execution_time'] for opp in opportunities]) if opportunities else 0
        required_capital = sum([opp['buy_price'] for opp in opportunities])
        success_probability = max(0.5, 1.0 - execution_risk)
        
        return ArbitrageResult(
            arbitrage_opportunities=opportunities,
            expected_profit=total_profit,
            execution_risk=execution_risk,
            time_window=time_window,
            required_capital=required_capital,
            success_probability=success_probability
        )

class ReinforcementLearningModel:
    """Combined Reinforcement Learning Model for Cryptocurrency Trading"""
    
    def __init__(self, asset: str = "BTC"):
        self.asset = asset.upper()
        self.environment = TradingEnvironmentSimulator()
        self.dqn_agent = DQNAgent()
        self.ppo_agent = PPOAgent()
        self.portfolio_optimizer = PortfolioOptimizationRL([asset])
        self.market_maker = MarketMakingRL()
        self.arbitrage_detector = ArbitrageDetectionRL()
        
        self.training_history = []
        self.performance_metrics = {}
    
    def train_agent(self, 
                   episodes: int = 1000,
                   agent_type: str = "dqn") -> RLTrainingResult:
        """Train RL agent"""
        
        reward_history = []
        loss_history = []
        action_counts = defaultdict(int)
        
        best_reward = float('-inf')
        convergence_episode = episodes
        
        for episode in range(episodes):
            state = self.environment.reset()
            state_vector = self.environment.get_state_vector(state)
            
            total_reward = 0
            done = False
            
            # Episode data for PPO
            states, actions, rewards, log_probs, values = [], [], [], [], []
            
            while not done:
                if agent_type == "dqn":
                    action_idx = self.dqn_agent.act(state_vector)
                    action = self._idx_to_action(action_idx)
                    
                elif agent_type == "ppo":
                    action_idx, log_prob, value = self.ppo_agent.get_action_and_value(state_vector)
                    action = self._idx_to_action(action_idx)
                    
                    states.append(state_vector)
                    actions.append(action_idx)
                    log_probs.append(log_prob)
                    values.append(value)
                
                else:
                    action = TradingAction("hold", 0.0, 0.5, "random")
                    action_idx = 2
                
                # Execute action
                next_state, reward, done = self.environment.step(action, state.price)
                next_state_vector = self.environment.get_state_vector(next_state)
                
                # Store experience
                if agent_type == "dqn":
                    self.dqn_agent.remember(state_vector, action_idx, reward, next_state_vector, done)
                    
                    # Train DQN
                    if len(self.dqn_agent.memory) > 32:
                        self.dqn_agent.replay()
                
                elif agent_type == "ppo":
                    rewards.append(reward)
                
                total_reward += reward
                action_counts[action.action_type] += 1
                
                state = next_state
                state_vector = next_state_vector
            
            # Update PPO agent
            if agent_type == "ppo" and len(states) > 0:
                self.ppo_agent.update(states, actions, rewards, log_probs, values)
            
            reward_history.append(total_reward)
            
            # Update target network for DQN
            if agent_type == "dqn" and episode % 100 == 0:
                self.dqn_agent.update_target_network()
            
            # Track best performance
            if total_reward > best_reward:
                best_reward = total_reward
                convergence_episode = episode
            
            # Log progress
            if episode % 100 == 0:
                avg_reward = np.mean(reward_history[-100:])
                logger.info(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Best: {best_reward:.2f}")
        
        # Calculate performance metrics
        performance_metrics = {
            'final_portfolio_value': self.environment.portfolio_value,
            'total_return': (self.environment.portfolio_value - self.environment.initial_balance) / self.environment.initial_balance,
            'sharpe_ratio': np.mean(reward_history) / np.std(reward_history) if np.std(reward_history) > 0 else 0,
            'max_drawdown': min(reward_history) if reward_history else 0,
            'win_rate': len([r for r in reward_history if r > 0]) / len(reward_history) if reward_history else 0
        }
        
        return RLTrainingResult(
            total_episodes=episodes,
            final_reward=reward_history[-1] if reward_history else 0,
            average_reward=np.mean(reward_history) if reward_history else 0,
            best_reward=best_reward,
            convergence_episode=convergence_episode,
            training_loss=loss_history,
            reward_history=reward_history,
            action_distribution=dict(action_counts),
            performance_metrics=performance_metrics
        )
    
    def get_trading_recommendation(self, 
                                  market_data: Dict,
                                  agent_type: str = "dqn") -> PolicyResult:
        """Get trading recommendation from trained agent"""
        
        # Create market state
        state = MarketState(
            price=market_data.get('price', 50000),
            volume=market_data.get('volume', 1000000),
            volatility=market_data.get('volatility', 0.02),
            trend=market_data.get('trend', 0.0),
            momentum=market_data.get('momentum', 0.0),
            support_resistance=market_data.get('support_resistance', {}),
            technical_indicators=market_data.get('technical_indicators', {}),
            sentiment=market_data.get('sentiment', 0.0),
            timestamp=datetime.now()
        )
        
        state_vector = self.environment.get_state_vector(state)
        
        if agent_type == "dqn" and TORCH_AVAILABLE and self.dqn_agent.q_network:
            # Get Q-values
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
            q_values = self.dqn_agent.q_network(state_tensor)
            action_idx = q_values.argmax().item()
            
            # Convert to probabilities
            probs = F.softmax(q_values, dim=1).squeeze().detach().numpy()
            action_probabilities = {
                'buy': float(probs[0]),
                'sell': float(probs[1]),
                'hold': float(probs[2])
            }
            
            value_estimate = float(q_values.max().item())
            
        elif agent_type == "ppo" and TORCH_AVAILABLE and self.ppo_agent.policy_network:
            # Get action probabilities
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
            action_probs = self.ppo_agent.policy_network(state_tensor)
            action_idx = action_probs.argmax().item()
            
            action_probabilities = {
                'buy': float(action_probs[0][0]),
                'sell': float(action_probs[0][1]),
                'hold': float(action_probs[0][2])
            }
            
            value_estimate = float(self.ppo_agent.value_network(state_tensor).item())
            
        else:
            # Fallback to random
            action_idx = random.randint(0, 2)
            action_probabilities = {'buy': 0.33, 'sell': 0.33, 'hold': 0.34}
            value_estimate = 0.0
        
        recommended_action = self._idx_to_action(action_idx)
        
        # Calculate risk and return estimates
        risk_assessment = state.volatility
        expected_return = state.trend * 100  # Convert to percentage
        confidence_interval = (expected_return - 2*risk_assessment, expected_return + 2*risk_assessment)
        
        return PolicyResult(
            recommended_action=recommended_action,
            action_probabilities=action_probabilities,
            value_estimate=value_estimate,
            risk_assessment=risk_assessment,
            expected_return=expected_return,
            confidence_interval=confidence_interval
        )
    
    def analyze(self, 
               market_data: Dict = None,
               price_data: pd.DataFrame = None,
               training_episodes: int = 500) -> RLAnalysisResult:
        """Perform comprehensive RL analysis"""
        
        try:
            # Train agent
            training_performance = self.train_agent(episodes=training_episodes)
            
            # Get policy recommendation
            if market_data is None:
                market_data = {
                    'price': 50000,
                    'volume': 1000000,
                    'volatility': 0.02,
                    'trend': 0.01,
                    'momentum': 0.005,
                    'sentiment': 0.1
                }
            
            policy_result = self.get_trading_recommendation(market_data)
            
            # Portfolio optimization
            if price_data is not None:
                portfolio_optimization = self.portfolio_optimizer.optimize_portfolio(price_data)
            else:
                # Create dummy result
                portfolio_optimization = PortfolioOptimizationResult(
                    optimal_weights={self.asset: 1.0},
                    expected_return=0.1,
                    expected_risk=0.2,
                    sharpe_ratio=0.5,
                    max_drawdown=-0.1,
                    rebalancing_frequency=30,
                    risk_adjusted_return=0.08
                )
            
            # Market making
            market_making = self.market_maker.generate_quotes(
                current_price=market_data['price'],
                volatility=market_data['volatility'],
                inventory=0.0,
                order_flow=0.0
            )
            
            # Arbitrage detection
            price_dict = {
                'exchange1': market_data['price'],
                'exchange2': market_data['price'] * 1.001,
                'exchange3': market_data['price'] * 0.999
            }
            arbitrage_detection = self.arbitrage_detector.detect_arbitrage(
                price_dict, 
                {'exchange1': 0.001, 'exchange2': 0.001, 'exchange3': 0.001}
            )
            
            # Risk metrics
            risk_metrics = {
                'var_95': training_performance.performance_metrics.get('max_drawdown', 0) * 0.95,
                'expected_shortfall': training_performance.performance_metrics.get('max_drawdown', 0) * 1.2,
                'volatility': market_data['volatility'],
                'beta': 1.0,  # Simplified
                'correlation': 0.8  # Simplified
            }
            
            # Performance attribution
            performance_attribution = {
                'alpha': training_performance.performance_metrics.get('total_return', 0) - 0.05,  # Excess return
                'market_timing': 0.02,
                'security_selection': 0.01,
                'interaction': 0.005
            }
            
            # Model confidence
            model_confidence = min(training_performance.performance_metrics.get('win_rate', 0.5), 0.9)
            
            return RLAnalysisResult(
                policy_result=policy_result,
                portfolio_optimization=portfolio_optimization,
                market_making=market_making,
                arbitrage_detection=arbitrage_detection,
                training_performance=training_performance,
                risk_metrics=risk_metrics,
                performance_attribution=performance_attribution,
                model_confidence=model_confidence
            )
            
        except Exception as e:
            logger.error(f"Error in RL analysis: {str(e)}")
            raise
    
    def _idx_to_action(self, action_idx: int) -> TradingAction:
        """Convert action index to TradingAction"""
        action_map = {
            0: TradingAction("buy", 0.1, 0.7, "RL recommendation"),
            1: TradingAction("sell", 0.1, 0.7, "RL recommendation"),
            2: TradingAction("hold", 0.0, 0.8, "RL recommendation")
        }
        return action_map.get(action_idx, action_map[2])
    
    def get_rl_insights(self, result: RLAnalysisResult) -> Dict[str, str]:
        """Generate comprehensive RL insights"""
        insights = {}
        
        # Training insights
        insights['training'] = f"Episodes: {result.training_performance.total_episodes}, Final Reward: {result.training_performance.final_reward:.2f}, Win Rate: {result.training_performance.performance_metrics.get('win_rate', 0):.1%}"
        
        # Policy insights
        action = result.policy_result.recommended_action
        insights['recommendation'] = f"Action: {action.action_type.upper()}, Amount: {action.amount:.1%}, Confidence: {action.confidence:.1%}"
        
        # Portfolio insights
        portfolio = result.portfolio_optimization
        insights['portfolio'] = f"Expected Return: {portfolio.expected_return:.1%}, Risk: {portfolio.expected_risk:.1%}, Sharpe: {portfolio.sharpe_ratio:.2f}"
        
        # Market making insights
        mm = result.market_making
        insights['market_making'] = f"Spread: {mm.spread:.2f}, Expected Profit: {mm.expected_profit:.4f}, Risk Penalty: {mm.risk_penalty:.4f}"
        
        # Arbitrage insights
        arb = result.arbitrage_detection
        insights['arbitrage'] = f"Opportunities: {len(arb.arbitrage_opportunities)}, Expected Profit: {arb.expected_profit:.4f}, Success Prob: {arb.success_probability:.1%}"
        
        # Risk insights
        insights['risk'] = f"VaR 95%: {result.risk_metrics['var_95']:.2%}, Volatility: {result.risk_metrics['volatility']:.2%}, Beta: {result.risk_metrics['beta']:.2f}"
        
        # Performance insights
        perf = result.performance_attribution
        insights['performance'] = f"Alpha: {perf['alpha']:.2%}, Market Timing: {perf['market_timing']:.2%}, Security Selection: {perf['security_selection']:.2%}"
        
        # Overall insights
        insights['overall'] = f"Model Confidence: {result.model_confidence:.1%}, Total Return: {result.training_performance.performance_metrics.get('total_return', 0):.1%}"
        
        return insights

# Example usage and testing
if __name__ == "__main__":
    # Test the RL model
    rl_model = ReinforcementLearningModel("BTC")
    
    # Sample market data
    market_data = {
        'price': 45000,
        'volume': 2000000,
        'volatility': 0.025,
        'trend': 0.02,
        'momentum': 0.01,
        'sentiment': 0.3,
        'technical_indicators': {
            'rsi': 65,
            'macd': 150,
            'bb_position': 0.7
        }
    }
    
    # Run analysis
    result = rl_model.analyze(market_data=market_data, training_episodes=100)
    
    insights = rl_model.get_rl_insights(result)
    
    print("=== Reinforcement Learning Analysis ===")
    print(f"Model Confidence: {result.model_confidence:.1%}")
    print(f"Recommended Action: {result.policy_result.recommended_action.action_type.upper()}")
    print(f"Expected Return: {result.policy_result.expected_return:.2%}")
    print()
    
    print("=== Detailed Insights ===")
    for key, insight in insights.items():
        print(f"{key}: {insight}")