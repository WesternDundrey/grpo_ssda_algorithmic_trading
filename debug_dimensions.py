#!/usr/bin/env python3

import numpy as np
import pandas as pd
import sys
sys.path.append('src')

from ssda_grpo_strategy import TradingState

# Create test data
test_data = {
    'open': [100, 101, 102, 103, 104],
    'high': [105, 106, 107, 108, 109],
    'low': [95, 96, 97, 98, 99],
    'close': [102, 103, 104, 105, 106],
    'volume': [1000, 1100, 1200, 1300, 1400]
}

df = pd.DataFrame(test_data)

# Test state vector dimensions
state = TradingState(
    prices=np.array([102.0, 103.0, 104.0, 105.0]),
    volumes=np.array([1000, 1100, 1200, 1400]),
    technical_indicators=np.array([0.5, 0.3, -0.1, 0.2, 0.1, 0.0]),  # 6 indicators
    denoised_features=np.random.random((5, 10)),  # Some denoised data
    state_representation=np.random.random((5, 8)),  # State space data
    portfolio_value=10000.0,
    cash=5000.0,
    position=0.0,
    unrealized_pnl=0.0,
    volatility=0.02,
    max_drawdown=0.0,
    sharpe_ratio=0.5,
    market_return=0.02,
    relative_performance=0.0
)

vector = state.to_vector()
print(f"State vector shape: {vector.shape}")
print(f"State vector length: {len(vector)}")
print("State vector contents:")
for i, val in enumerate(vector):
    print(f"  [{i:2d}]: {val:.6f}")

print("\nFeature breakdown:")
print("Price features (4): price_change, volatility, return_vs_market, volume_trend")
print("Technical indicators (5): from technical_indicators[:5]") 
print("Denoised momentum (1): momentum from denoised features")
print("State features (3): from state_representation[:3]")
print("Portfolio features (5): position, pnl, volatility, drawdown, performance")
print("Total expected: 4 + 5 + 1 + 3 + 5 = 18")