"""Generalized Reward Policy Optimization for Trading"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class Experience:
    """Single experience tuple for RL"""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    info: Dict[str, Any]


class RewardFunction:
    """Customizable reward function for trading"""
    
    def __init__(self, 
                 profit_weight: float = 1.0,
                 risk_penalty: float = 0.1,
                 transaction_cost: float = 0.001,
                 max_drawdown_penalty: float = 0.5):
        """
        Initialize reward function
        
        Args:
            profit_weight: Weight for profit/loss component
            risk_penalty: Penalty for high volatility
            transaction_cost: Cost per transaction
            max_drawdown_penalty: Penalty for drawdown
        """
        self.profit_weight = profit_weight
        self.risk_penalty = risk_penalty
        self.transaction_cost = transaction_cost
        self.max_drawdown_penalty = max_drawdown_penalty
        
        # Track portfolio metrics
        self.portfolio_values = deque(maxlen=252)  # 1 year history
        self.returns = deque(maxlen=252)
        self.peak_value = 0.0
    
    def calculate_reward(self, 
                        action: int,
                        prev_action: int,
                        portfolio_value: float,
                        prev_portfolio_value: float,
                        market_return: float) -> float:
        """
        Calculate reward for given action
        
        Args:
            action: Current action (0: hold, 1: buy, 2: sell)
            prev_action: Previous action
            portfolio_value: Current portfolio value
            prev_portfolio_value: Previous portfolio value
            market_return: Market return for comparison
            
        Returns:
            Calculated reward
        """
        # Portfolio return
        if prev_portfolio_value > 0:
            portfolio_return = (portfolio_value - prev_portfolio_value) / prev_portfolio_value
        else:
            portfolio_return = 0.0
        
        # Update tracking
        self.portfolio_values.append(portfolio_value)
        self.returns.append(portfolio_return)
        self.peak_value = max(self.peak_value, portfolio_value)
        
        # Base reward: excess return over market
        excess_return = portfolio_return - market_return
        reward = self.profit_weight * excess_return
        
        # Risk penalty
        if len(self.returns) > 10:
            volatility = np.std(list(self.returns)[-10:])
            reward -= self.risk_penalty * volatility
        
        # Transaction cost
        if action != prev_action:
            reward -= self.transaction_cost
        
        # Drawdown penalty
        if self.peak_value > 0:
            current_drawdown = (self.peak_value - portfolio_value) / self.peak_value
            if current_drawdown > 0.05:  # 5% drawdown threshold
                reward -= self.max_drawdown_penalty * current_drawdown
        
        # Sharpe ratio bonus (if enough history)
        if len(self.returns) > 30:
            returns_array = np.array(list(self.returns))
            if np.std(returns_array) > 0:
                sharpe = np.mean(returns_array) / np.std(returns_array)
                reward += 0.1 * max(0, sharpe - 0.5)  # Bonus for Sharpe > 0.5
        
        return reward


class PolicyNetwork:
    """Simple policy network using linear approximation"""
    
    def __init__(self, state_dim: int, action_dim: int = 3, learning_rate: float = 0.001):
        """
        Initialize policy network
        
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension (hold, buy, sell)
            learning_rate: Learning rate
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        
        # Initialize weights (simple linear policy)
        self.W = np.random.normal(0, 0.1, (state_dim, action_dim))
        self.b = np.zeros(action_dim)
        
        # Adam optimizer parameters
        self.m_W = np.zeros_like(self.W)
        self.v_W = np.zeros_like(self.W)
        self.m_b = np.zeros_like(self.b)
        self.v_b = np.zeros_like(self.b)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0  # time step for Adam
    
    def forward(self, state: np.ndarray) -> np.ndarray:
        """Forward pass to get action probabilities"""
        logits = state @ self.W + self.b
        # Softmax for action probabilities
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)
    
    def sample_action(self, state: np.ndarray, exploration: float = 0.1) -> int:
        """Sample action from policy with optional exploration"""
        probs = self.forward(state)
        
        # Add exploration noise
        if exploration > 0:
            probs = (1 - exploration) * probs + exploration / self.action_dim
        
        return np.random.choice(self.action_dim, p=probs)
    
    def get_action_prob(self, state: np.ndarray, action: int) -> float:
        """Get probability of specific action"""
        probs = self.forward(state)
        return probs[action]
    
    def update(self, states: np.ndarray, actions: np.ndarray, advantages: np.ndarray):
        """Update policy using policy gradient"""
        self.t += 1
        batch_size = len(states)
        
        # Calculate gradients
        grad_W = np.zeros_like(self.W)
        grad_b = np.zeros_like(self.b)
        
        for i in range(batch_size):
            state = states[i]
            action = actions[i]
            advantage = advantages[i]
            
            # Get current probabilities
            probs = self.forward(state)
            
            # Policy gradient
            grad_logits = -probs.copy()
            grad_logits[action] += 1
            grad_logits *= advantage
            
            # Accumulate gradients
            grad_W += np.outer(state, grad_logits)
            grad_b += grad_logits
        
        # Average gradients
        grad_W /= batch_size
        grad_b /= batch_size
        
        # Adam update
        self.m_W = self.beta1 * self.m_W + (1 - self.beta1) * grad_W
        self.v_W = self.beta2 * self.v_W + (1 - self.beta2) * (grad_W ** 2)
        
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * grad_b
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (grad_b ** 2)
        
        # Bias correction
        m_W_hat = self.m_W / (1 - self.beta1 ** self.t)
        v_W_hat = self.v_W / (1 - self.beta2 ** self.t)
        m_b_hat = self.m_b / (1 - self.beta1 ** self.t)
        v_b_hat = self.v_b / (1 - self.beta2 ** self.t)
        
        # Update parameters
        self.W += self.learning_rate * m_W_hat / (np.sqrt(v_W_hat) + self.epsilon)
        self.b += self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)


class ValueNetwork:
    """Value function approximation"""
    
    def __init__(self, state_dim: int, learning_rate: float = 0.001):
        """Initialize value network"""
        self.state_dim = state_dim
        self.learning_rate = learning_rate
        
        # Linear value function
        self.w = np.random.normal(0, 0.1, state_dim)
        self.b = 0.0
        
        # Adam optimizer
        self.m_w = np.zeros_like(self.w)
        self.v_w = np.zeros_like(self.w)
        self.m_b = 0.0
        self.v_b = 0.0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0
    
    def forward(self, state: np.ndarray) -> float:
        """Estimate state value"""
        return np.dot(state, self.w) + self.b
    
    def update(self, states: np.ndarray, targets: np.ndarray):
        """Update value function using regression"""
        self.t += 1
        batch_size = len(states)
        
        # Calculate gradients
        predictions = np.array([self.forward(state) for state in states])
        errors = targets - predictions
        
        grad_w = -np.mean(errors[:, np.newaxis] * states, axis=0)
        grad_b = -np.mean(errors)
        
        # Adam update
        self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * grad_w
        self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (grad_w ** 2)
        
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * grad_b
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (grad_b ** 2)
        
        # Bias correction and update
        m_w_hat = self.m_w / (1 - self.beta1 ** self.t)
        v_w_hat = self.v_w / (1 - self.beta2 ** self.t)
        m_b_hat = self.m_b / (1 - self.beta1 ** self.t)
        v_b_hat = self.v_b / (1 - self.beta2 ** self.t)
        
        self.w += self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
        self.b += self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)


class GRPO:
    """Generalized Reward Policy Optimization Agent"""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int = 3,
                 policy_lr: float = 0.001,
                 value_lr: float = 0.001,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 1.0):
        """
        Initialize GRPO agent
        
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            policy_lr: Policy learning rate
            value_lr: Value learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Networks
        self.policy = PolicyNetwork(state_dim, action_dim, policy_lr)
        self.value_fn = ValueNetwork(state_dim, value_lr)
        
        # Reward function
        self.reward_fn = RewardFunction()
        
        # Experience buffer
        self.experiences = []
        
        # Training statistics
        self.episode_returns = []
        self.episode_lengths = []
        self.policy_losses = []
        self.value_losses = []
    
    def get_action(self, state: np.ndarray, exploration: float = 0.1) -> int:
        """Get action from current policy"""
        return self.policy.sample_action(state, exploration)
    
    def store_experience(self, experience: Experience):
        """Store experience in replay buffer"""
        self.experiences.append(experience)
    
    def calculate_gae(self, rewards: List[float], values: List[float], 
                     next_values: List[float], dones: List[bool]) -> Tuple[List[float], List[float]]:
        """Calculate Generalized Advantage Estimation"""
        advantages = []
        returns = []
        
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_values[i] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
        
        return advantages, returns
    
    def update(self, batch_size: int = 32, epochs: int = 4) -> Dict[str, float]:
        """Update policy using PPO-style optimization"""
        if len(self.experiences) < batch_size:
            return {}
        
        # Extract data from experiences
        states = np.array([exp.state for exp in self.experiences])
        actions = np.array([exp.action for exp in self.experiences])
        rewards = [exp.reward for exp in self.experiences]
        next_states = np.array([exp.next_state for exp in self.experiences])
        dones = [exp.done for exp in self.experiences]
        
        # Calculate values
        values = [self.value_fn.forward(state) for state in states]
        next_values = [self.value_fn.forward(state) for state in next_states]
        
        # Calculate advantages and returns using GAE
        advantages, returns = self.calculate_gae(rewards, values, next_values, dones)
        
        advantages = np.array(advantages)
        returns = np.array(returns)
        
        # Normalize advantages
        if np.std(advantages) > 0:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        # Store old policy probabilities
        old_probs = np.array([
            self.policy.get_action_prob(states[i], actions[i])
            for i in range(len(states))
        ])
        
        # Training statistics
        policy_losses = []
        value_losses = []
        
        # Multiple epochs of optimization
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(len(states))
            
            # Mini-batch updates
            for start in range(0, len(states), batch_size):
                end = min(start + batch_size, len(states))
                batch_indices = indices[start:end]
                
                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_probs = old_probs[batch_indices]
                
                # Calculate new probabilities
                new_probs = np.array([
                    self.policy.get_action_prob(batch_states[i], batch_actions[i])
                    for i in range(len(batch_states))
                ])
                
                # PPO ratio and clipping
                ratios = new_probs / (batch_old_probs + 1e-8)
                clipped_ratios = np.clip(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                
                # Policy loss (PPO objective)
                policy_loss = -np.mean(np.minimum(
                    ratios * batch_advantages,
                    clipped_ratios * batch_advantages
                ))
                
                # Add entropy bonus
                entropy = -np.mean(new_probs * np.log(new_probs + 1e-8))
                policy_loss -= self.entropy_coef * entropy
                
                # Value loss
                batch_values = np.array([self.value_fn.forward(state) for state in batch_states])
                value_loss = np.mean((batch_values - batch_returns) ** 2)
                
                # Update networks
                self.policy.update(batch_states, batch_actions, batch_advantages)
                self.value_fn.update(batch_states, batch_returns)
                
                policy_losses.append(policy_loss)
                value_losses.append(value_loss)
        
        # Clear experiences
        self.experiences.clear()
        
        # Update statistics
        avg_policy_loss = np.mean(policy_losses) if policy_losses else 0.0
        avg_value_loss = np.mean(value_losses) if value_losses else 0.0
        
        self.policy_losses.append(avg_policy_loss)
        self.value_losses.append(avg_value_loss)
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'avg_advantage': np.mean(advantages),
            'avg_return': np.mean(returns)
        }
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            'episode_returns': self.episode_returns.copy(),
            'episode_lengths': self.episode_lengths.copy(),
            'policy_losses': self.policy_losses.copy(),
            'value_losses': self.value_losses.copy(),
            'avg_return': np.mean(self.episode_returns[-10:]) if self.episode_returns else 0.0,
            'avg_policy_loss': np.mean(self.policy_losses[-10:]) if self.policy_losses else 0.0,
            'avg_value_loss': np.mean(self.value_losses[-10:]) if self.value_losses else 0.0
        }