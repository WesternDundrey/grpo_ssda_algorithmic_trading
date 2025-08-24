#!/usr/bin/env python3
"""Comprehensive debugging of dimension mismatch"""

import sys
import numpy as np
import pandas as pd
import logging

sys.path.append('src')

from ssda_grpo_strategy import SSDAGRPOStrategy, TradingState
from grpo import GRPO

logging.basicConfig(level=logging.DEBUG)

def debug_grpo_initialization():
    """Debug GRPO initialization"""
    print("=== GRPO Initialization Debug ===")
    
    # Test different state dimensions
    for state_dim in [15, 16, 17]:
        try:
            grpo = GRPO(state_dim=state_dim)
            print(f"✅ GRPO initialized with state_dim={state_dim}")
            print(f"   Policy network W shape: {grpo.policy.W.shape}")
            print(f"   Value network w shape: {grpo.value_fn.w.shape}")
            
            # Test with actual vector
            test_vector = np.random.random(state_dim)
            action = grpo.get_action(test_vector)
            print(f"   Test action: {action}")
            
        except Exception as e:
            print(f"❌ GRPO failed with state_dim={state_dim}: {e}")
    
    return True

def debug_strategy_initialization():
    """Debug strategy initialization"""
    print("\n=== Strategy Initialization Debug ===")
    
    strategy = SSDAGRPOStrategy('TEST')
    print(f"Strategy GRPO params: {strategy.grpo_params}")
    print(f"GRPO policy W shape: {strategy.grpo.policy.W.shape}")
    print(f"GRPO value w shape: {strategy.grpo.value_fn.w.shape}")
    
    return strategy

def debug_state_creation():
    """Debug state vector creation during actual usage"""
    print("\n=== State Creation Debug ===")
    
    # Create minimal trading state like during early backtest
    minimal_state = TradingState(
        prices=np.array([100.0]),  # Only 1 price
        volumes=np.array([1000]),   # Only 1 volume
        technical_indicators=np.zeros(3),  # Only 3 indicators (less than 5)
        denoised_features=np.zeros((1, 2)),  # Minimal features
        state_representation=np.zeros((1, 2)),  # Minimal state
        portfolio_value=10000.0,
        cash=10000.0,
        position=0.0,
        unrealized_pnl=0.0,
        volatility=0.01,
        max_drawdown=0.0,
        sharpe_ratio=0.0,
        market_return=0.0,
        relative_performance=0.0
    )
    
    print("Minimal state (like early backtest):")
    vector = minimal_state.to_vector()
    print(f"  Vector shape: {vector.shape}")
    print(f"  Vector length: {len(vector)}")
    print(f"  Vector: {vector}")
    
    # Test what happens in PolicyNetwork.forward
    strategy = SSDAGRPOStrategy('TEST')
    try:
        # This is where the error occurs
        logits = vector @ strategy.grpo.policy.W + strategy.grpo.policy.b
        print(f"  Matrix multiplication successful: {logits.shape}")
    except Exception as e:
        print(f"  ❌ Matrix multiplication failed: {e}")
        print(f"  Vector shape: {vector.shape} vs W shape: {strategy.grpo.policy.W.shape}")
    
    return vector

def debug_matrix_shapes():
    """Debug all matrix shapes in detail"""
    print("\n=== Matrix Shapes Debug ===")
    
    strategy = SSDAGRPOStrategy('TEST')
    
    # Check GRPO network shapes
    print(f"GRPO Policy Network:")
    print(f"  W shape: {strategy.grpo.policy.W.shape}")
    print(f"  b shape: {strategy.grpo.policy.b.shape}")
    print(f"  Expected input: ({strategy.grpo.policy.W.shape[0]},)")
    
    print(f"GRPO Value Network:")
    print(f"  w shape: {strategy.grpo.value_fn.w.shape}")
    print(f"  Expected input: ({strategy.grpo.value_fn.w.shape[0]},)")
    
    # Test various state vector sizes
    for size in [15, 16, 17]:
        test_vector = np.random.random(size)
        print(f"\nTesting vector size {size}:")
        try:
            result = strategy.grpo.policy.forward(test_vector)
            print(f"  ✅ Forward pass successful: {result.shape}")
        except Exception as e:
            print(f"  ❌ Forward pass failed: {e}")

if __name__ == "__main__":
    debug_grpo_initialization()
    strategy = debug_strategy_initialization()
    vector = debug_state_creation()
    debug_matrix_shapes()
    
    print(f"\n=== Summary ===")
    print(f"Strategy expects: {strategy.grpo_params['state_dim']} dimensions")
    print(f"State vector provides: {len(vector)} dimensions")
    print(f"GRPO W matrix expects: {strategy.grpo.policy.W.shape[0]} dimensions")
    
    if len(vector) == strategy.grpo.policy.W.shape[0]:
        print("✅ Dimensions match!")
    else:
        print("❌ Dimension mismatch found!")