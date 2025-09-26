"""Reinforcement Learning Service

Provides reinforcement learning capabilities for trading and portfolio optimization:
- Trading agents using Deep Q-Learning (DQN)
- Portfolio optimization agents
- Multi-agent trading systems
- Risk-aware RL agents
- Backtesting and simulation environments
- Agent training and evaluation
- Strategy optimization
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import pickle
from pathlib import Path
import random
from collections import deque
import threading

# Deep learning imports (would be installed via requirements)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Fallback implementations
    class nn:
        class Module: pass
        class Linear: pass
        class ReLU: pass
        class Dropout: pass

logger = logging.getLogger(__name__)

class ActionType(Enum):
    """Trading action types"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    BUY_STRONG = "buy_strong"
    SELL_STRONG = "sell_strong"

class AgentType(Enum):
    """RL agent types"""
    DQN = "dqn"
    DDPG = "ddpg"
    PPO = "ppo"
    A3C = "a3c"
    PORTFOLIO_OPTIMIZER = "portfolio_optimizer"
    RISK_AWARE = "risk_aware"

class RewardType(Enum):
    """Reward function types"""
    PROFIT_LOSS = "profit_loss"
    SHARPE_RATIO = "sharpe_ratio"
    RISK_ADJUSTED = "risk_adjusted"
    DRAWDOWN_PENALTY = "drawdown_penalty"
    TRANSACTION_COST = "transaction_cost"

@dataclass
class TradingState:
    """Trading environment state"""
    timestamp: datetime
    prices: Dict[str, float]
    technical_indicators: Dict[str, float]
    portfolio_value: float
    cash: float
    positions: Dict[str, float]
    market_features: Dict[str, float]
    risk_metrics: Dict[str, float]

@dataclass
class TradingAction:
    """Trading action"""
    action_type: ActionType
    symbol: str
    quantity: float
    price: Optional[float] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TrainingEpisode:
    """Training episode data"""
    episode_id: str
    agent_id: str
    start_time: datetime
    end_time: datetime
    total_reward: float
    final_portfolio_value: float
    num_trades: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    actions_taken: List[TradingAction]
    states_visited: List[TradingState]
    rewards_received: List[float]

@dataclass
class AgentPerformance:
    """Agent performance metrics"""
    agent_id: str
    evaluation_date: datetime
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    avg_trade_duration: float
    risk_adjusted_return: float

class TradingEnvironment:
    """Trading environment for RL agents"""
    
    def __init__(self, 
                 data: pd.DataFrame,
                 initial_cash: float = 100000,
                 transaction_cost: float = 0.001,
                 max_position_size: float = 0.1):
        self.data = data.copy()
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
        
        # Environment state
        self.current_step = 0
        self.cash = initial_cash
        self.positions = {}
        self.portfolio_history = []
        self.action_history = []
        
        # Features for state representation
        self.feature_columns = [col for col in data.columns if col not in ['timestamp', 'symbol']]
        self.lookback_window = 20
        
        # Reset environment
        self.reset()
    
    def reset(self) -> TradingState:
        """Reset environment to initial state"""
        self.current_step = self.lookback_window
        self.cash = self.initial_cash
        self.positions = {}
        self.portfolio_history = []
        self.action_history = []
        
        return self._get_current_state()
    
    def step(self, action: TradingAction) -> Tuple[TradingState, float, bool, Dict[str, Any]]:
        """Execute action and return next state, reward, done, info"""
        # Execute action
        reward = self._execute_action(action)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.data) - 1
        
        # Get next state
        next_state = self._get_current_state() if not done else None
        
        # Additional info
        info = {
            'portfolio_value': self._get_portfolio_value(),
            'cash': self.cash,
            'positions': self.positions.copy(),
            'step': self.current_step
        }
        
        return next_state, reward, done, info
    
    def _execute_action(self, action: TradingAction) -> float:
        """Execute trading action and return reward"""
        current_price = self.data.iloc[self.current_step]['close']
        symbol = action.symbol if action.symbol else 'default'
        
        # Calculate position change
        current_position = self.positions.get(symbol, 0)
        
        if action.action_type == ActionType.BUY:
            max_buy = min(self.cash / current_price, 
                         self.max_position_size * self.initial_cash / current_price)
            quantity = min(action.quantity, max_buy)
            
            if quantity > 0:
                cost = quantity * current_price * (1 + self.transaction_cost)
                if cost <= self.cash:
                    self.cash -= cost
                    self.positions[symbol] = current_position + quantity
        
        elif action.action_type == ActionType.SELL:
            quantity = min(action.quantity, current_position)
            
            if quantity > 0:
                proceeds = quantity * current_price * (1 - self.transaction_cost)
                self.cash += proceeds
                self.positions[symbol] = current_position - quantity
                if self.positions[symbol] <= 0:
                    del self.positions[symbol]
        
        elif action.action_type == ActionType.BUY_STRONG:
            # Buy with higher quantity
            max_buy = min(self.cash / current_price, 
                         self.max_position_size * 2 * self.initial_cash / current_price)
            quantity = min(action.quantity * 2, max_buy)
            
            if quantity > 0:
                cost = quantity * current_price * (1 + self.transaction_cost)
                if cost <= self.cash:
                    self.cash -= cost
                    self.positions[symbol] = current_position + quantity
        
        elif action.action_type == ActionType.SELL_STRONG:
            # Sell larger quantity
            quantity = min(action.quantity * 2, current_position)
            
            if quantity > 0:
                proceeds = quantity * current_price * (1 - self.transaction_cost)
                self.cash += proceeds
                self.positions[symbol] = current_position - quantity
                if self.positions[symbol] <= 0:
                    del self.positions[symbol]
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Store action
        self.action_history.append(action)
        
        return reward
    
    def _calculate_reward(self) -> float:
        """Calculate reward for current state"""
        current_portfolio_value = self._get_portfolio_value()
        
        if len(self.portfolio_history) == 0:
            reward = 0.0
        else:
            # Return-based reward
            previous_value = self.portfolio_history[-1]
            reward = (current_portfolio_value - previous_value) / previous_value
            
            # Add risk penalty for large drawdowns
            if len(self.portfolio_history) > 10:
                recent_values = self.portfolio_history[-10:] + [current_portfolio_value]
                peak = max(recent_values)
                drawdown = (peak - current_portfolio_value) / peak
                reward -= drawdown * 0.5  # Penalty for drawdown
        
        self.portfolio_history.append(current_portfolio_value)
        return reward
    
    def _get_portfolio_value(self) -> float:
        """Calculate current portfolio value"""
        current_price = self.data.iloc[self.current_step]['close']
        
        total_value = self.cash
        for symbol, quantity in self.positions.items():
            total_value += quantity * current_price
        
        return total_value
    
    def _get_current_state(self) -> TradingState:
        """Get current environment state"""
        current_row = self.data.iloc[self.current_step]
        
        # Get recent price data for technical indicators
        start_idx = max(0, self.current_step - self.lookback_window)
        recent_data = self.data.iloc[start_idx:self.current_step + 1]
        
        # Calculate technical indicators
        technical_indicators = self._calculate_technical_indicators(recent_data)
        
        # Market features
        market_features = {
            'price': current_row['close'],
            'volume': current_row.get('volume', 0),
            'volatility': recent_data['close'].std() if len(recent_data) > 1 else 0,
            'momentum': (current_row['close'] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0] if len(recent_data) > 1 else 0
        }
        
        # Risk metrics
        portfolio_value = self._get_portfolio_value()
        risk_metrics = {
            'portfolio_concentration': max(self.positions.values()) / portfolio_value if self.positions and portfolio_value > 0 else 0,
            'cash_ratio': self.cash / portfolio_value if portfolio_value > 0 else 1,
            'leverage': (portfolio_value - self.cash) / portfolio_value if portfolio_value > 0 else 0
        }
        
        return TradingState(
            timestamp=current_row.get('timestamp', datetime.now()),
            prices={'default': current_row['close']},
            technical_indicators=technical_indicators,
            portfolio_value=portfolio_value,
            cash=self.cash,
            positions=self.positions.copy(),
            market_features=market_features,
            risk_metrics=risk_metrics
        )
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate technical indicators from recent data"""
        if len(data) < 2:
            return {}
        
        indicators = {}
        
        # Simple moving averages
        if len(data) >= 5:
            indicators['sma_5'] = data['close'].tail(5).mean()
        if len(data) >= 10:
            indicators['sma_10'] = data['close'].tail(10).mean()
        if len(data) >= 20:
            indicators['sma_20'] = data['close'].tail(20).mean()
        
        # RSI
        if len(data) >= 14:
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs.iloc[-1]))
        
        # Price position relative to recent high/low
        recent_high = data['high'].max() if 'high' in data.columns else data['close'].max()
        recent_low = data['low'].min() if 'low' in data.columns else data['close'].min()
        current_price = data['close'].iloc[-1]
        
        if recent_high != recent_low:
            indicators['price_position'] = (current_price - recent_low) / (recent_high - recent_low)
        else:
            indicators['price_position'] = 0.5
        
        return indicators

class DQNNetwork(nn.Module if TORCH_AVAILABLE else object):
    """Deep Q-Network for trading decisions"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        if TORCH_AVAILABLE:
            super(DQNNetwork, self).__init__()
            self.fc1 = nn.Linear(state_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, hidden_size)
            self.fc4 = nn.Linear(hidden_size, action_size)
            self.dropout = nn.Dropout(0.2)
        else:
            # Fallback implementation
            self.state_size = state_size
            self.action_size = action_size
            self.weights = np.random.randn(state_size, action_size) * 0.1
    
    def forward(self, x):
        if TORCH_AVAILABLE:
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            x = F.relu(self.fc3(x))
            x = self.fc4(x)
            return x
        else:
            # Simple linear transformation as fallback
            return np.dot(x, self.weights)

class DQNAgent:
    """Deep Q-Learning agent for trading"""
    
    def __init__(self, 
                 state_size: int,
                 action_size: int,
                 learning_rate: float = 0.001,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 memory_size: int = 10000,
                 batch_size: int = 32):
        
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        # Experience replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Neural networks
        if TORCH_AVAILABLE:
            self.q_network = DQNNetwork(state_size, action_size)
            self.target_network = DQNNetwork(state_size, action_size)
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
            
            # Copy weights to target network
            self.update_target_network()
        else:
            # Fallback to simple Q-table
            self.q_table = np.random.randn(state_size, action_size) * 0.1
        
        # Training metrics
        self.training_history = []
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Choose action using epsilon-greedy policy"""
        if training and np.random.random() <= self.epsilon:
            return np.random.choice(self.action_size)
        
        if TORCH_AVAILABLE:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return np.argmax(q_values.cpu().data.numpy())
        else:
            # Fallback: simple linear model
            q_values = np.dot(state, self.q_table)
            return np.argmax(q_values)
    
    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self) -> float:
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        batch = random.sample(self.memory, self.batch_size)
        
        if TORCH_AVAILABLE:
            return self._replay_torch(batch)
        else:
            return self._replay_fallback(batch)
    
    def _replay_torch(self, batch) -> float:
        """PyTorch-based training"""
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
        
        return loss.item()
    
    def _replay_fallback(self, batch) -> float:
        """Fallback training without PyTorch"""
        total_loss = 0.0
        
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target += 0.99 * np.max(np.dot(next_state, self.q_table))
            
            current_q = np.dot(state, self.q_table)[action]
            loss = (target - current_q) ** 2
            total_loss += loss
            
            # Simple gradient update
            gradient = 2 * (current_q - target) * state
            self.q_table[:, action] -= self.learning_rate * gradient
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return total_loss / len(batch)
    
    def update_target_network(self):
        """Update target network weights"""
        if TORCH_AVAILABLE:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_model(self, filepath: str):
        """Save model to file"""
        if TORCH_AVAILABLE:
            torch.save({
                'q_network_state_dict': self.q_network.state_dict(),
                'target_network_state_dict': self.target_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon
            }, filepath)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'q_table': self.q_table,
                    'epsilon': self.epsilon
                }, f)
    
    def load_model(self, filepath: str):
        """Load model from file"""
        if TORCH_AVAILABLE:
            checkpoint = torch.load(filepath)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
        else:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.q_table = data['q_table']
                self.epsilon = data['epsilon']

class ReinforcementLearningService:
    """Main RL service for trading and portfolio optimization"""
    
    def __init__(self, models_path: str = "rl_models"):
        self.models_path = Path(models_path)
        self.models_path.mkdir(exist_ok=True)
        
        # Agents registry
        self.agents: Dict[str, DQNAgent] = {}
        self.agent_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Training environments
        self.environments: Dict[str, TradingEnvironment] = {}
        
        # Performance tracking
        self.performance_history: Dict[str, List[AgentPerformance]] = {}
        self.training_history: Dict[str, List[TrainingEpisode]] = {}
        
        # Action mapping
        self.action_mapping = {
            0: ActionType.HOLD,
            1: ActionType.BUY,
            2: ActionType.SELL,
            3: ActionType.BUY_STRONG,
            4: ActionType.SELL_STRONG
        }
    
    async def create_agent(self, 
                          agent_id: str,
                          agent_type: AgentType = AgentType.DQN,
                          state_size: int = 20,
                          **kwargs) -> str:
        """Create a new RL agent"""
        if agent_type == AgentType.DQN:
            agent = DQNAgent(
                state_size=state_size,
                action_size=len(self.action_mapping),
                **kwargs
            )
        else:
            raise ValueError(f"Agent type {agent_type} not implemented yet")
        
        self.agents[agent_id] = agent
        self.agent_metadata[agent_id] = {
            'agent_type': agent_type,
            'created_at': datetime.now(),
            'state_size': state_size,
            'parameters': kwargs
        }
        
        logger.info(f"Created {agent_type.value} agent: {agent_id}")
        return agent_id
    
    async def train_agent(self, 
                         agent_id: str,
                         training_data: pd.DataFrame,
                         episodes: int = 1000,
                         validation_split: float = 0.2) -> List[TrainingEpisode]:
        """Train an RL agent"""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        agent = self.agents[agent_id]
        
        # Split data
        split_idx = int(len(training_data) * (1 - validation_split))
        train_data = training_data.iloc[:split_idx]
        val_data = training_data.iloc[split_idx:]
        
        # Create training environment
        env = TradingEnvironment(train_data)
        
        training_episodes = []
        
        for episode in range(episodes):
            episode_start = datetime.now()
            state = env.reset()
            total_reward = 0
            actions_taken = []
            states_visited = []
            rewards_received = []
            
            done = False
            while not done:
                # Convert state to feature vector
                state_vector = self._state_to_vector(state)
                states_visited.append(state)
                
                # Choose action
                action_idx = agent.act(state_vector, training=True)
                action_type = self.action_mapping[action_idx]
                
                # Create trading action
                action = TradingAction(
                    action_type=action_type,
                    symbol='default',
                    quantity=1000,  # Fixed quantity for simplicity
                    confidence=1.0 - agent.epsilon
                )
                actions_taken.append(action)
                
                # Execute action
                next_state, reward, done, info = env.step(action)
                total_reward += reward
                rewards_received.append(reward)
                
                # Store experience
                if next_state is not None:
                    next_state_vector = self._state_to_vector(next_state)
                    agent.remember(state_vector, action_idx, reward, next_state_vector, done)
                
                # Train agent
                if len(agent.memory) > agent.batch_size:
                    loss = agent.replay()
                
                state = next_state
            
            # Update target network periodically
            if episode % 100 == 0:
                agent.update_target_network()
            
            # Create episode record
            episode_record = TrainingEpisode(
                episode_id=f"{agent_id}_episode_{episode}",
                agent_id=agent_id,
                start_time=episode_start,
                end_time=datetime.now(),
                total_reward=total_reward,
                final_portfolio_value=info['portfolio_value'],
                num_trades=len([a for a in actions_taken if a.action_type != ActionType.HOLD]),
                win_rate=self._calculate_win_rate(rewards_received),
                sharpe_ratio=self._calculate_sharpe_ratio(rewards_received),
                max_drawdown=self._calculate_max_drawdown(env.portfolio_history),
                actions_taken=actions_taken,
                states_visited=states_visited,
                rewards_received=rewards_received
            )
            
            training_episodes.append(episode_record)
            
            # Log progress
            if episode % 100 == 0:
                logger.info(f"Episode {episode}: Reward={total_reward:.4f}, Portfolio={info['portfolio_value']:.2f}, Epsilon={agent.epsilon:.3f}")
        
        # Store training history
        if agent_id not in self.training_history:
            self.training_history[agent_id] = []
        self.training_history[agent_id].extend(training_episodes)
        
        # Evaluate on validation data
        if len(val_data) > 0:
            await self.evaluate_agent(agent_id, val_data)
        
        # Save trained model
        model_path = self.models_path / f"{agent_id}_model.pth"
        agent.save_model(str(model_path))
        
        logger.info(f"Training completed for agent {agent_id}")
        return training_episodes
    
    async def evaluate_agent(self, 
                            agent_id: str, 
                            test_data: pd.DataFrame) -> AgentPerformance:
        """Evaluate agent performance on test data"""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        agent = self.agents[agent_id]
        env = TradingEnvironment(test_data)
        
        state = env.reset()
        total_return = 0
        trades = []
        portfolio_values = [env.initial_cash]
        
        done = False
        while not done:
            state_vector = self._state_to_vector(state)
            action_idx = agent.act(state_vector, training=False)  # No exploration
            action_type = self.action_mapping[action_idx]
            
            action = TradingAction(
                action_type=action_type,
                symbol='default',
                quantity=1000
            )
            
            next_state, reward, done, info = env.step(action)
            
            if action_type != ActionType.HOLD:
                trades.append({
                    'timestamp': state.timestamp,
                    'action': action_type,
                    'price': state.prices['default'],
                    'portfolio_value': info['portfolio_value']
                })
            
            portfolio_values.append(info['portfolio_value'])
            state = next_state
        
        # Calculate performance metrics
        final_value = portfolio_values[-1]
        total_return = (final_value - env.initial_cash) / env.initial_cash
        
        # Annualized return (assuming daily data)
        days = len(test_data)
        annualized_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
        
        # Volatility
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
        
        # Sharpe ratio
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Max drawdown
        max_drawdown = self._calculate_max_drawdown(portfolio_values)
        
        # Win rate
        winning_trades = len([t for t in trades if t['action'] in [ActionType.SELL, ActionType.SELL_STRONG]])
        win_rate = winning_trades / len(trades) if trades else 0
        
        # Profit factor
        profits = [r for r in returns if r > 0]
        losses = [abs(r) for r in returns if r < 0]
        profit_factor = sum(profits) / sum(losses) if losses else float('inf')
        
        performance = AgentPerformance(
            agent_id=agent_id,
            evaluation_date=datetime.now(),
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            num_trades=len(trades),
            avg_trade_duration=days / len(trades) if trades else 0,
            risk_adjusted_return=total_return / max_drawdown if max_drawdown > 0 else total_return
        )
        
        # Store performance
        if agent_id not in self.performance_history:
            self.performance_history[agent_id] = []
        self.performance_history[agent_id].append(performance)
        
        logger.info(f"Agent {agent_id} evaluation: Return={total_return:.2%}, Sharpe={sharpe_ratio:.2f}, Drawdown={max_drawdown:.2%}")
        return performance
    
    async def get_trading_signal(self, 
                                agent_id: str, 
                                current_data: pd.DataFrame) -> TradingAction:
        """Get trading signal from trained agent"""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        agent = self.agents[agent_id]
        env = TradingEnvironment(current_data)
        state = env._get_current_state()
        
        state_vector = self._state_to_vector(state)
        action_idx = agent.act(state_vector, training=False)
        action_type = self.action_mapping[action_idx]
        
        # Calculate confidence based on Q-values spread
        if TORCH_AVAILABLE:
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
            q_values = agent.q_network(state_tensor).cpu().data.numpy()[0]
            max_q = np.max(q_values)
            second_max_q = np.partition(q_values, -2)[-2]
            confidence = (max_q - second_max_q) / (max_q + 1e-8) if max_q != 0 else 0.5
        else:
            confidence = 0.7  # Default confidence for fallback
        
        return TradingAction(
            action_type=action_type,
            symbol='default',
            quantity=1000,
            price=state.prices['default'],
            confidence=min(max(confidence, 0.0), 1.0),
            metadata={
                'agent_id': agent_id,
                'timestamp': datetime.now(),
                'state_features': state_vector.tolist()
            }
        )
    
    def _state_to_vector(self, state: TradingState) -> np.ndarray:
        """Convert trading state to feature vector"""
        features = []
        
        # Price features
        features.extend(list(state.prices.values()))
        
        # Technical indicators
        features.extend(list(state.technical_indicators.values()))
        
        # Portfolio features
        features.append(state.portfolio_value)
        features.append(state.cash)
        features.extend(list(state.positions.values()))
        
        # Market features
        features.extend(list(state.market_features.values()))
        
        # Risk metrics
        features.extend(list(state.risk_metrics.values()))
        
        # Pad or truncate to fixed size
        target_size = 20  # Default state size
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        elif len(features) > target_size:
            features = features[:target_size]
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_win_rate(self, rewards: List[float]) -> float:
        """Calculate win rate from rewards"""
        if not rewards:
            return 0.0
        positive_rewards = [r for r in rewards if r > 0]
        return len(positive_rewards) / len(rewards)
    
    def _calculate_sharpe_ratio(self, rewards: List[float]) -> float:
        """Calculate Sharpe ratio from rewards"""
        if len(rewards) < 2:
            return 0.0
        mean_return = np.mean(rewards)
        std_return = np.std(rewards)
        return mean_return / std_return if std_return > 0 else 0.0
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown"""
        if len(portfolio_values) < 2:
            return 0.0
        
        peak = portfolio_values[0]
        max_drawdown = 0.0
        
        for value in portfolio_values[1:]:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def get_agent_info(self, agent_id: str) -> Dict[str, Any]:
        """Get agent information"""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        metadata = self.agent_metadata[agent_id].copy()
        metadata['performance_history'] = self.performance_history.get(agent_id, [])
        metadata['training_episodes'] = len(self.training_history.get(agent_id, []))
        
        return metadata
    
    def list_agents(self) -> List[str]:
        """List all available agents"""
        return list(self.agents.keys())

# Global instance
rl_service = ReinforcementLearningService()