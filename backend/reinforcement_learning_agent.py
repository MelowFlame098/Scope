import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

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

class ReinforcementLearningAgent:
    """RL agent for crypto trading decisions"""
    
    def __init__(self, state_size: int = 10, action_size: int = 3):
        self.state_size = state_size
        self.action_size = action_size  # 0: sell, 1: hold, 2: buy
        self.memory = []
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.q_table = np.random.uniform(-1, 1, (100, action_size))  # Simplified Q-table
        self.total_episodes = 0
        
    def _state_to_index(self, state: List[float]) -> int:
        """Convert continuous state to discrete index for Q-table"""
        try:
            # Simple hash-based discretization
            state_hash = hash(tuple(np.round(state, 2))) % len(self.q_table)
            return abs(state_hash)
        except Exception:
            return 0
    
    def get_action(self, state: List[float]) -> CryptoIndicatorResult:
        """Get trading action from RL agent"""
        try:
            if len(state) != self.state_size:
                # Pad or truncate state to match expected size
                if len(state) < self.state_size:
                    state = state + [0.0] * (self.state_size - len(state))
                else:
                    state = state[:self.state_size]
            
            state_index = self._state_to_index(state)
            
            # Epsilon-greedy action selection
            if np.random.random() <= self.epsilon:
                action = np.random.choice(self.action_size)
                confidence = 0.3  # Lower confidence for random actions
            else:
                action = np.argmax(self.q_table[state_index])
                confidence = 0.8  # Higher confidence for learned actions
            
            # Convert action to trading signal
            action_map = {0: 'sell', 1: 'hold', 2: 'buy'}
            signal = action_map[action]
            
            # Calculate action strength based on Q-values
            q_values = self.q_table[state_index]
            max_q = np.max(q_values)
            min_q = np.min(q_values)
            
            if max_q != min_q:
                strength = (q_values[action] - min_q) / (max_q - min_q)
            else:
                strength = 0.5
            
            # Update epsilon (decay exploration)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            self.total_episodes += 1
            
            return CryptoIndicatorResult(
                indicator_name='RL Trading Agent',
                value=float(action),
                confidence=confidence,
                timestamp=datetime.now(),
                metadata={
                    'action': action,
                    'q_values': q_values.tolist(),
                    'epsilon': self.epsilon,
                    'state_index': state_index,
                    'total_episodes': self.total_episodes,
                    'state_size': len(state)
                },
                signal=signal,
                strength=strength
            )
            
        except Exception as e:
            logger.error(f"Error getting RL action: {e}")
            return self._error_result('RL Trading Agent', str(e))
    
    def update_q_table(self, 
                      state: List[float], 
                      action: int, 
                      reward: float, 
                      next_state: List[float],
                      done: bool = False) -> bool:
        """Update Q-table based on experience"""
        try:
            # Ensure states are correct size
            if len(state) != self.state_size:
                if len(state) < self.state_size:
                    state = state + [0.0] * (self.state_size - len(state))
                else:
                    state = state[:self.state_size]
                    
            if len(next_state) != self.state_size:
                if len(next_state) < self.state_size:
                    next_state = next_state + [0.0] * (self.state_size - len(next_state))
                else:
                    next_state = next_state[:self.state_size]
            
            state_index = self._state_to_index(state)
            next_state_index = self._state_to_index(next_state)
            
            # Q-learning update
            current_q = self.q_table[state_index][action]
            
            if done:
                target_q = reward
            else:
                next_max_q = np.max(self.q_table[next_state_index])
                target_q = reward + 0.95 * next_max_q  # gamma = 0.95
            
            # Update Q-value
            self.q_table[state_index][action] += self.learning_rate * (target_q - current_q)
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating Q-table: {e}")
            return False
    
    def save_experience(self, 
                       state: List[float], 
                       action: int, 
                       reward: float, 
                       next_state: List[float], 
                       done: bool):
        """Save experience to memory for potential replay"""
        try:
            experience = {
                'state': state.copy(),
                'action': action,
                'reward': reward,
                'next_state': next_state.copy(),
                'done': done,
                'timestamp': datetime.now()
            }
            
            self.memory.append(experience)
            
            # Keep memory size manageable
            if len(self.memory) > 10000:
                self.memory.pop(0)
                
        except Exception as e:
            logger.error(f"Error saving experience: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            'total_episodes': self.total_episodes,
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'q_table_shape': self.q_table.shape,
            'learning_rate': self.learning_rate
        }
    
    def _error_result(self, indicator_name: str, error_msg: str) -> CryptoIndicatorResult:
        return CryptoIndicatorResult(
            indicator_name=indicator_name,
            value=0.0,
            confidence=0.0,
            timestamp=datetime.now(),
            metadata={'error': error_msg},
            signal='hold',
            strength=0.0
        )