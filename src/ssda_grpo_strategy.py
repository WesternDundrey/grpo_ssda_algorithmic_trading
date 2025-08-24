"""SSDA+GRPO Integrated Trading Strategy"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass

from ssda import SSDA
from grpo import GRPO, Experience
from backtester import BacktestEngine

logger = logging.getLogger(__name__)


@dataclass
class TradingState:
    """Comprehensive trading state representation"""
    # Market features
    prices: np.ndarray              # Recent price history
    volumes: np.ndarray             # Recent volume history
    technical_indicators: np.ndarray  # Technical analysis features
    denoised_features: np.ndarray   # SSDA denoised features
    state_representation: np.ndarray # SSDA state space representation
    
    # Portfolio state
    portfolio_value: float
    cash: float
    position: float                 # Current position size
    unrealized_pnl: float
    
    # Risk metrics
    volatility: float
    max_drawdown: float
    sharpe_ratio: float
    
    # Market context
    market_return: float
    relative_performance: float
    
    def to_vector(self) -> np.ndarray:
        """Convert state to feature vector for RL agent"""
        features = []
        
        # Price momentum (last 5 vs previous 5)
        if len(self.prices) >= 10:
            recent_momentum = np.mean(self.prices[-5:]) / np.mean(self.prices[-10:-5]) - 1
        else:
            recent_momentum = 0.0
        features.append(recent_momentum)
        
        # Volume trend
        if len(self.volumes) >= 10:
            volume_trend = np.mean(self.volumes[-5:]) / np.mean(self.volumes[-10:-5]) - 1
        else:
            volume_trend = 0.0
        features.append(volume_trend)
        
        # Technical indicators (first 5, with safe indexing)
        if len(self.technical_indicators) >= 5:
            tech_features = self.technical_indicators[:5]
        else:
            tech_features = np.zeros(5)
            if len(self.technical_indicators) > 0:
                tech_features[:len(self.technical_indicators)] = self.technical_indicators
        features.extend(tech_features)
        
        # Denoised signal strength
        if len(self.denoised_features) > 1:
            denoised_momentum = np.mean(self.denoised_features[-1]) - np.mean(self.denoised_features[-2])
        else:
            denoised_momentum = 0.0
        features.append(denoised_momentum)
        
        # State space representation (first 3 dimensions)
        if len(self.state_representation) > 0 and self.state_representation.shape[1] >= 3:
            state_features = self.state_representation[-1, :3]
        elif len(self.state_representation) > 0:
            # Pad with zeros if not enough features
            available_features = self.state_representation[-1]
            state_features = np.zeros(3)
            state_features[:len(available_features)] = available_features[:3]
        else:
            state_features = np.zeros(3)
        features.extend(state_features)
        
        # Portfolio features
        features.extend([
            self.position,                    # Current position
            self.unrealized_pnl / 10000,     # Normalized PnL
            self.volatility,                  # Portfolio volatility
            self.max_drawdown,               # Max drawdown
            self.relative_performance        # Performance vs market
        ])
        
        features_array = np.array(features, dtype=np.float32)
        
        # Ensure exactly 16 dimensions for GRPO compatibility
        if len(features_array) < 16:
            # Pad with zeros if too few features
            padded = np.zeros(16, dtype=np.float32)
            padded[:len(features_array)] = features_array
            return padded
        elif len(features_array) > 16:
            # Truncate if too many features  
            return features_array[:16]
        else:
            return features_array


class SSDAGRPOStrategy:
    """Integrated SSDA+GRPO Trading Strategy"""
    
    def __init__(self,
                 ticker: str,
                 ssda_params: Dict[str, Any] = None,
                 grpo_params: Dict[str, Any] = None,
                 lookback_window: int = 60,
                 training_episodes: int = 100,
                 position_size_limit: float = 0.3):
        """
        Initialize SSDA+GRPO strategy
        
        Args:
            ticker: Trading symbol
            ssda_params: SSDA configuration parameters
            grpo_params: GRPO configuration parameters
            lookback_window: Historical data window
            training_episodes: Number of training episodes
            position_size_limit: Maximum position size as fraction of portfolio
        """
        self.ticker = ticker
        self.lookback_window = lookback_window
        self.training_episodes = training_episodes
        self.position_size_limit = position_size_limit
        
        # Default parameters
        default_ssda_params = {
            'state_dim': 8,
            'hidden_dims': [32, 16, 8, 16, 32],
            'noise_factor': 0.05,
            'lookback_window': 20
        }
        default_grpo_params = {
            'state_dim': 16,  # Based on actual TradingState.to_vector() output
            'action_dim': 3,
            'policy_lr': 0.001,
            'value_lr': 0.003,
            'gamma': 0.99,
            'gae_lambda': 0.95
        }
        
        self.ssda_params = {**default_ssda_params, **(ssda_params or {})}
        self.grpo_params = {**default_grpo_params, **(grpo_params or {})}
        
        # Initialize models
        self.ssda = SSDA(**self.ssda_params)
        self.grpo = GRPO(**self.grpo_params)
        
        # Strategy state
        self.is_trained = False
        self.current_position = 0.0
        self.current_action = 0  # 0: hold, 1: buy, 2: sell
        self.previous_action = 0
        self.episode_count = 0
        
        # Historical data storage
        self.price_history = []
        self.volume_history = []
        self.portfolio_history = []
        self.action_history = []
        self.reward_history = []
        
        # Performance tracking
        self.training_stats = {
            'episodes': [],
            'returns': [],
            'sharpe_ratios': [],
            'max_drawdowns': [],
            'policy_losses': [],
            'value_losses': []
        }
    
    def _calculate_technical_indicators(self, prices: pd.Series, volumes: pd.Series) -> np.ndarray:
        """Calculate technical indicators"""
        indicators = []
        
        # RSI
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        indicators.append(rsi.iloc[-1] if not rsi.empty else 50)
        
        # MACD
        exp1 = prices.ewm(span=12).mean()
        exp2 = prices.ewm(span=26).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9).mean()
        indicators.append((macd.iloc[-1] - signal.iloc[-1]) if len(macd) > 0 else 0)
        
        # Bollinger Bands
        sma = prices.rolling(20).mean()
        std = prices.rolling(20).std()
        bb_upper = sma + (std * 2)
        bb_lower = sma - (std * 2)
        bb_position = (prices.iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
        indicators.append(bb_position if not np.isnan(bb_position) else 0.5)
        
        # Volume indicators
        volume_sma = volumes.rolling(20).mean()
        volume_ratio = volumes.iloc[-1] / volume_sma.iloc[-1] if volume_sma.iloc[-1] > 0 else 1.0
        indicators.append(min(volume_ratio, 5.0))  # Cap extreme values
        
        # Price momentum (5-day, 20-day)
        momentum_5 = (prices.iloc[-1] / prices.iloc[-6] - 1) if len(prices) > 5 else 0
        momentum_20 = (prices.iloc[-1] / prices.iloc[-21] - 1) if len(prices) > 20 else 0
        indicators.extend([momentum_5, momentum_20])
        
        return np.array(indicators, dtype=np.float32)
    
    def _create_trading_state(self, engine: BacktestEngine, data: pd.Series) -> TradingState:
        """Create comprehensive trading state"""
        # Get recent price and volume data
        recent_prices = engine.get_historical_prices(self.ticker, self.lookback_window)
        recent_volumes = engine.get_historical_prices(self.ticker, self.lookback_window, 'volume')
        
        if len(recent_prices) < 10:
            # Not enough data - return minimal state
            return TradingState(
                prices=np.array([engine.get_current_price(self.ticker)]),
                volumes=np.array([1000]),
                technical_indicators=np.zeros(6),
                denoised_features=np.zeros((1, 5)),
                state_representation=np.zeros((1, self.ssda_params['state_dim'])),
                portfolio_value=engine.portfolio.total_value,
                cash=engine.portfolio.cash,
                position=0.0,
                unrealized_pnl=0.0,
                volatility=0.01,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                market_return=0.0,
                relative_performance=0.0
            )
        
        # Calculate technical indicators
        technical_indicators = self._calculate_technical_indicators(recent_prices, recent_volumes)
        
        # Get SSDA denoised features and state representation
        price_df = pd.DataFrame({
            'close': recent_prices,
            'open': recent_prices,  # Simplified - using close as proxy
            'high': recent_prices * 1.01,
            'low': recent_prices * 0.99,
            'volume': recent_volumes
        })
        
        if self.ssda.is_fitted:
            try:
                denoised_features, state_representation = self.ssda.denoise_and_predict(price_df)
            except Exception as e:
                logger.warning(f"SSDA prediction failed: {e}")
                denoised_features = np.zeros((len(recent_prices), 5))
                state_representation = np.zeros((len(recent_prices), self.ssda_params['state_dim']))
        else:
            denoised_features = np.zeros((len(recent_prices), 5))
            state_representation = np.zeros((len(recent_prices), self.ssda_params['state_dim']))
        
        # Portfolio metrics
        current_position = engine.portfolio.get_position(self.ticker)
        position_size = current_position.shares if current_position else 0.0
        unrealized_pnl = current_position.unrealized_pnl if current_position else 0.0
        
        # Risk metrics
        returns = recent_prices.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0.01
        
        # Calculate drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = ((cumulative - running_max) / running_max).min() if len(cumulative) > 0 else 0.0
        
        # Sharpe ratio
        sharpe_ratio = (returns.mean() * 252) / (volatility + 1e-8) if len(returns) > 1 else 0.0
        
        # Market context
        market_return = returns.iloc[-1] if len(returns) > 0 else 0.0
        portfolio_return = (engine.portfolio.total_value - 100000) / 100000
        relative_performance = portfolio_return - market_return
        
        return TradingState(
            prices=recent_prices.values,
            volumes=recent_volumes.values,
            technical_indicators=technical_indicators,
            denoised_features=denoised_features,
            state_representation=state_representation,
            portfolio_value=engine.portfolio.total_value,
            cash=engine.portfolio.cash,
            position=position_size / 1000,  # Normalize position size
            unrealized_pnl=unrealized_pnl,
            volatility=volatility,
            max_drawdown=drawdown,
            sharpe_ratio=sharpe_ratio,
            market_return=market_return,
            relative_performance=relative_performance
        )
    
    def initialize(self, engine: BacktestEngine):
        """Initialize strategy (called by backtesting engine)"""
        logger.info(f"Initializing SSDA+GRPO strategy for {self.ticker}")
        
        # Pre-train SSDA if enough historical data is available
        if hasattr(engine, 'data') and len(engine.data) > 100:
            logger.info("Pre-training SSDA on historical data...")
            try:
                # Get historical data for SSDA training
                historical_data = engine.data.copy()
                
                if self.ticker in historical_data.columns.get_level_values('ticker'):
                    ticker_data = historical_data.loc[:, (slice(None), self.ticker)]
                    ticker_data.columns = [col[0] for col in ticker_data.columns]
                    
                    # Train SSDA
                    self.ssda.fit(ticker_data)
                    logger.info("SSDA pre-training completed")
                else:
                    logger.warning(f"Ticker {self.ticker} not found in historical data")
                    
            except Exception as e:
                logger.error(f"SSDA pre-training failed: {e}")
        
        self.episode_count = 0
        self.current_position = 0.0
        self.current_action = 0
        self.previous_action = 0
    
    def handle_data(self, engine: BacktestEngine, data):
        """Handle daily data and make trading decisions"""
        try:
            # Create current trading state
            current_state = self._create_trading_state(engine, data)
            state_vector = current_state.to_vector()
            
            # Debug logging for dimension mismatch
            if state_vector.shape[0] != 16:
                logger.error(f"State vector dimension mismatch: expected 16, got {state_vector.shape[0]}")
                logger.error(f"State vector: {state_vector}")
            
            # Get action from GRPO agent
            if self.is_trained:
                exploration = max(0.01, 0.3 - (self.episode_count / self.training_episodes) * 0.29)
            else:
                exploration = 0.3  # Higher exploration during training
            
            action = self.grpo.get_action(state_vector, exploration)
            
            # Calculate reward for previous action
            if hasattr(self, 'previous_state'):
                reward = self._calculate_reward(
                    action=self.current_action,
                    current_state=current_state,
                    previous_state=self.previous_state
                )
                
                # Store experience for training
                experience = Experience(
                    state=self.previous_state_vector,
                    action=self.current_action,
                    reward=reward,
                    next_state=state_vector,
                    done=False,
                    info={'portfolio_value': current_state.portfolio_value}
                )
                
                self.grpo.store_experience(experience)
                self.reward_history.append(reward)
            
            # Execute trading action
            self._execute_action(engine, action, current_state)
            
            # Store current state for next iteration
            self.previous_state = current_state
            self.previous_state_vector = state_vector
            self.previous_action = self.current_action
            self.current_action = action
            
            # Update GRPO periodically
            if len(self.grpo.experiences) >= 32 and not self.is_trained:
                training_stats = self.grpo.update()
                if training_stats:
                    self._update_training_stats(training_stats)
            
            # Track performance
            self.portfolio_history.append(current_state.portfolio_value)
            self.action_history.append(action)
            
            # Check if training is complete
            if not self.is_trained and self.episode_count >= self.training_episodes:
                self.is_trained = True
                logger.info(f"SSDA+GRPO training completed after {self.episode_count} episodes")
                
        except Exception as e:
            logger.error(f"Error in handle_data: {e}")
            # Safe fallback - hold position
            pass
    
    def _calculate_reward(self, action: int, current_state: TradingState, 
                         previous_state: TradingState) -> float:
        """Calculate reward for the given action"""
        # Portfolio return component
        portfolio_return = (current_state.portfolio_value - previous_state.portfolio_value) / previous_state.portfolio_value
        
        # Base reward from portfolio performance
        reward = portfolio_return * 100  # Scale up
        
        # Risk-adjusted reward
        if current_state.volatility > 0:
            risk_adjusted_return = portfolio_return / (current_state.volatility + 1e-8)
            reward += risk_adjusted_return * 10
        
        # Drawdown penalty
        if current_state.max_drawdown < -0.05:  # More than 5% drawdown
            reward -= abs(current_state.max_drawdown) * 50
        
        # Transaction cost penalty
        if action != self.previous_action:
            reward -= 0.01  # Small transaction cost
        
        # Sharpe ratio bonus
        if current_state.sharpe_ratio > 0.5:
            reward += (current_state.sharpe_ratio - 0.5) * 5
        
        # Market outperformance bonus
        if current_state.relative_performance > 0:
            reward += current_state.relative_performance * 20
        
        return reward
    
    def _execute_action(self, engine: BacktestEngine, action: int, state: TradingState):
        """Execute trading action"""
        current_price = engine.get_current_price(self.ticker)
        
        if np.isnan(current_price) or current_price <= 0:
            return
        
        target_position_pct = 0.0
        
        if action == 1:  # Buy
            target_position_pct = self.position_size_limit
        elif action == 2:  # Sell/Short
            target_position_pct = -self.position_size_limit
        # action == 0 is hold (target_position_pct = 0.0)
        
        # Execute order
        try:
            engine.order_target_percent(self.ticker, target_position_pct)
        except Exception as e:
            logger.warning(f"Failed to execute order for {self.ticker}: {e}")
    
    def _update_training_stats(self, training_stats: Dict[str, float]):
        """Update training statistics"""
        for key, value in training_stats.items():
            if key in self.training_stats:
                self.training_stats[key].append(value)
        
        self.episode_count += 1
    
    def get_strategy_stats(self) -> Dict[str, Any]:
        """Get comprehensive strategy statistics"""
        stats = {
            'is_trained': self.is_trained,
            'episode_count': self.episode_count,
            'total_rewards': sum(self.reward_history) if self.reward_history else 0.0,
            'avg_reward': np.mean(self.reward_history) if self.reward_history else 0.0,
            'portfolio_history': self.portfolio_history.copy(),
            'action_distribution': {
                'hold': self.action_history.count(0),
                'buy': self.action_history.count(1), 
                'sell': self.action_history.count(2)
            } if self.action_history else {'hold': 0, 'buy': 0, 'sell': 0},
            'ssda_fitted': self.ssda.is_fitted,
            'grpo_stats': self.grpo.get_training_stats()
        }
        
        # Add training statistics
        stats.update(self.training_stats)
        
        return stats