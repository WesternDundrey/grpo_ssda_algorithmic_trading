# SSDA+GRPO: Advanced AI Trading System

## Overview

This implementation combines **State-Space Denoising Autoencoders (SSDA)** with **Generalized Reward Policy Optimization (GRPO)** to create a sophisticated reinforcement learning-based trading system.

## 🧠 Architecture Components

### 1. SSDA (State-Space Denoising Autoencoder)

**File:** `src/ssda.py`

**Key Features:**
- **Kalman Filtering**: Uses state-space models to track latent market dynamics
- **Denoising Autoencoder**: Neural network removes noise from price signals
- **Technical Integration**: Combines price data with technical indicators
- **Signal Generation**: Produces clean trading signals from noisy market data

**Components:**
```python
class StateSpaceModel:
    # Kalman filter implementation
    # State transition: x(t+1) = A*x(t) + noise
    # Observation: y(t) = C*x(t) + noise
    
class DenoisingAutoencoder:
    # Neural denoising network
    # Input: noisy_features -> Output: clean_features
    
class SSDA:
    # Complete integration
    # Market data -> State representation + Denoised signals
```

**Technical Indicators Processed:**
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)  
- Bollinger Bands
- Volume indicators
- Price momentum (5-day, 20-day)

### 2. GRPO (Generalized Reward Policy Optimization)

**File:** `src/grpo.py`

**Key Features:**
- **Policy Gradient Learning**: Actor-critic reinforcement learning
- **PPO-Style Optimization**: Clipped policy updates for stability
- **GAE Advantages**: Generalized Advantage Estimation for variance reduction
- **Custom Reward Functions**: Trading-specific reward engineering

**Components:**
```python
class PolicyNetwork:
    # π(a|s) - action probabilities given state
    
class ValueNetwork:
    # V(s) - state value estimation
    
class RewardFunction:
    # Custom trading reward calculation
    # Factors: profit, risk, drawdown, Sharpe ratio
    
class GRPO:
    # Complete RL agent
    # Experience replay + policy updates
```

**Action Space:**
- 0: Hold position
- 1: Buy/Long position  
- 2: Sell/Short position

### 3. Integrated Strategy

**File:** `src/ssda_grpo_strategy.py`

**Integration Flow:**
1. **Market Data** → SSDA → **Clean Signals + State Representation**
2. **Trading State** (prices, indicators, portfolio) → **Feature Vector**
3. **Feature Vector** → GRPO → **Trading Action**
4. **Action Execution** → **Portfolio Update** → **Reward Calculation**
5. **Experience Storage** → **Policy Updates** (during training)

## 🔧 Implementation Details

### State Representation (15-dimensional vector):

1. **Price Momentum** (recent vs historical)
2. **Volume Trend** (volume change indicator)
3. **Technical Indicators** (RSI, MACD, Bollinger, momentum, volume ratio)
4. **Denoised Signal Strength** (SSDA output change)
5. **State Space Features** (Kalman filter state, 3D)
6. **Portfolio State** (position, PnL, volatility, drawdown, relative performance)

### Reward Function Components:

```python
reward = (
    profit_weight * portfolio_return +
    risk_bonus * sharpe_ratio_improvement +
    -risk_penalty * volatility +
    -transaction_cost * |action_change| +
    -drawdown_penalty * max_drawdown +
    outperformance_bonus * (portfolio_return - market_return)
)
```

### Training Process:

1. **SSDA Pre-training**: Historical data → Learn noise patterns
2. **Experience Collection**: Live trading simulation
3. **Policy Updates**: Every 32 experiences using PPO
4. **Exploration Decay**: High exploration → Low exploitation
5. **Performance Tracking**: Continuous monitoring

## 📊 Key Advantages

### vs Traditional Technical Analysis:
- **Noise Reduction**: SSDA removes market noise while preserving signals
- **Adaptive Learning**: RL adapts to changing market conditions
- **Multi-factor Integration**: Combines price, volume, and technical indicators

### vs Simple Machine Learning:
- **Sequential Decision Making**: Accounts for action consequences
- **Risk-Aware Optimization**: Reward function includes risk metrics
- **Real-time Adaptation**: Continuous learning from new experiences

### vs Standard RL:
- **Enhanced State Representation**: SSDA provides cleaner state features
- **Domain-Specific Rewards**: Trading-focused reward engineering
- **Stable Training**: PPO clipping prevents destructive policy updates

## 🚀 Usage Example

```python
from ssda_grpo_strategy import SSDAGRPOStrategy
from backtester import BacktestEngine

# Configure strategy
strategy = SSDAGRPOStrategy(
    ticker='SPY',
    ssda_params={
        'state_dim': 8,
        'hidden_dims': [32, 16, 8, 16, 32],
        'noise_factor': 0.05
    },
    grpo_params={
        'policy_lr': 0.001,
        'value_lr': 0.003,
        'gamma': 0.99
    },
    training_episodes=100,
    position_size_limit=0.3
)

# Run backtest
engine = BacktestEngine(market_data)
engine.register_initialize(strategy.initialize)
engine.register_handle_data(strategy.handle_data)

results = engine.run(start_date='2020-01-01', end_date='2023-01-01')

# Get performance metrics
stats = strategy.get_strategy_stats()
print(f"Final Return: {results['cumulative_returns'].iloc[-1]:.2%}")
print(f"Sharpe Ratio: {stats['sharpe_ratio']:.3f}")
```

## 📈 Expected Performance Characteristics

### Strengths:
- **Noise Robustness**: Better signal-to-noise ratio than raw TA
- **Adaptive Behavior**: Learns optimal entry/exit timing
- **Risk Management**: Built-in drawdown and volatility control
- **Market Regime Adaptation**: Adjusts to bull/bear markets

### Considerations:
- **Training Period**: Requires initial learning phase (50-100 episodes)
- **Computational Cost**: More intensive than simple strategies
- **Overfitting Risk**: May overfit to training period patterns
- **Market Impact**: Assumes trades don't affect prices

## 🔬 Validation & Testing

### Component Tests:
1. **SSDA Functionality**: Noise reduction and signal generation
2. **GRPO Learning**: Policy improvement over episodes
3. **Integration**: End-to-end strategy execution
4. **Backtesting**: Historical performance validation

### Performance Metrics:
- **Return Metrics**: Total return, annualized return, excess return
- **Risk Metrics**: Volatility, Sharpe ratio, maximum drawdown
- **Trading Metrics**: Win rate, average trade, transaction costs
- **RL Metrics**: Episode rewards, policy loss, value function accuracy

## 🛠 System Requirements

### Dependencies:
- **Core**: `numpy`, `pandas`, `scikit-learn`
- **Visualization**: `matplotlib`, `seaborn`, `plotly`
- **Optional**: `torch` (for advanced neural networks)

### Data Requirements:
- **OHLCV Data**: Open, High, Low, Close, Volume
- **Frequency**: Daily recommended (can adapt to intraday)
- **History**: Minimum 200+ data points for training
- **Quality**: Clean, adjusted for splits/dividends

## 🔮 Future Enhancements

### Potential Improvements:
1. **Multi-Asset Support**: Portfolio optimization across assets
2. **Deep Networks**: Replace MLPRegressor with proper deep learning
3. **Alternative Data**: News sentiment, economic indicators
4. **Risk Models**: Factor-based risk attribution
5. **Online Learning**: Real-time model updates
6. **Ensemble Methods**: Multiple SSDA+GRPO agents

### Advanced Features:
- **Hierarchical RL**: Multiple time horizons
- **Meta-Learning**: Fast adaptation to new markets
- **Adversarial Training**: Robustness to market shocks
- **Interpretability**: SHAP values for decision explanation

## 📚 Research Foundation

This implementation draws from:
- **Kalman Filters**: State-space models for time series
- **Denoising Autoencoders**: Noise reduction in neural networks  
- **PPO**: Proximal Policy Optimization for stable RL
- **GAE**: Generalized Advantage Estimation for variance reduction
- **Finance Literature**: Risk-adjusted performance metrics

The combination is novel and specifically designed for financial markets, addressing key challenges like noise, non-stationarity, and risk management.

---

**Note**: This is a research/educational implementation. For production trading, additional considerations include market microstructure, regime detection, risk management systems, and regulatory compliance.