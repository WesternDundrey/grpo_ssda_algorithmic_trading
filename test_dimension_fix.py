#!/usr/bin/env python3
"""Test dimension fix for SSDA+GRPO strategy"""

import sys
import os
import numpy as np
import pandas as pd
import logging

sys.path.append('src')

from ssda_grpo_strategy import SSDAGRPOStrategy, TradingState
from backtester import BacktestEngine

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dimension_consistency():
    """Test that state vectors are always 16 dimensions"""
    print("=== Testing Dimension Consistency ===")
    
    strategy = SSDAGRPOStrategy('TEST')
    
    # Test 1: Create various TradingState objects with different data sizes
    test_cases = [
        # Minimal state (early in backtest)
        TradingState(
            prices=np.array([100.0]),
            volumes=np.array([1000]),
            technical_indicators=np.zeros(3),  # Less than 5
            denoised_features=np.zeros((1, 2)),
            state_representation=np.zeros((1, 2)),
            portfolio_value=10000.0,
            cash=10000.0,
            position=0.0,
            unrealized_pnl=0.0,
            volatility=0.01,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            market_return=0.0,
            relative_performance=0.0
        ),
        # Normal state
        TradingState(
            prices=np.array([100.0, 101.0, 102.0, 103.0]),
            volumes=np.array([1000, 1100, 1200, 1300]),
            technical_indicators=np.array([0.5, 0.3, -0.1, 0.2, 0.1, 0.0]),
            denoised_features=np.random.random((5, 10)),
            state_representation=np.random.random((5, 8)),
            portfolio_value=10500.0,
            cash=5000.0,
            position=0.5,
            unrealized_pnl=500.0,
            volatility=0.02,
            max_drawdown=-0.01,
            sharpe_ratio=0.8,
            market_return=0.01,
            relative_performance=0.005
        ),
        # Large technical indicators (more than 5)
        TradingState(
            prices=np.array([100.0, 101.0, 102.0, 103.0, 104.0]),
            volumes=np.array([1000, 1100, 1200, 1300, 1400]),
            technical_indicators=np.array([0.5, 0.3, -0.1, 0.2, 0.1, 0.0, 0.4, 0.6, -0.2]),  # 9 indicators
            denoised_features=np.random.random((10, 15)),
            state_representation=np.random.random((10, 12)),
            portfolio_value=11000.0,
            cash=3000.0,
            position=0.8,
            unrealized_pnl=1000.0,
            volatility=0.025,
            max_drawdown=-0.05,
            sharpe_ratio=1.2,
            market_return=0.015,
            relative_performance=0.01
        )
    ]
    
    for i, state in enumerate(test_cases):
        vector = state.to_vector()
        print(f"Test case {i+1}: vector shape = {vector.shape}, length = {len(vector)}")
        
        if len(vector) != 16:
            print(f"❌ FAILED: Expected 16 dimensions, got {len(vector)}")
            return False
        else:
            print(f"✅ PASSED: Exactly 16 dimensions")
    
    print("\n=== All dimension tests passed! ===")
    return True

def test_grpo_compatibility():
    """Test that GRPO can accept the state vectors"""
    print("\n=== Testing GRPO Compatibility ===")
    
    strategy = SSDAGRPOStrategy('TEST')
    
    # Create a test state
    state = TradingState(
        prices=np.array([100.0, 101.0, 102.0]),
        volumes=np.array([1000, 1100, 1200]),
        technical_indicators=np.array([0.5, 0.3, -0.1]),  # Only 3 indicators
        denoised_features=np.random.random((3, 5)),
        state_representation=np.random.random((3, 4)),
        portfolio_value=10000.0,
        cash=8000.0,
        position=0.2,
        unrealized_pnl=200.0,
        volatility=0.015,
        max_drawdown=-0.02,
        sharpe_ratio=0.6,
        market_return=0.008,
        relative_performance=0.002
    )
    
    try:
        vector = state.to_vector()
        print(f"State vector shape: {vector.shape}")
        
        # Test GRPO action selection
        action = strategy.grpo.get_action(vector, exploration=0.1)
        print(f"GRPO selected action: {action}")
        print("✅ GRPO compatibility test passed!")
        return True
        
    except Exception as e:
        print(f"❌ GRPO compatibility test failed: {e}")
        return False

if __name__ == "__main__":
    success1 = test_dimension_consistency()
    success2 = test_grpo_compatibility()
    
    if success1 and success2:
        print("\n🎉 All tests passed! Dimension fix is working correctly.")
        exit(0)
    else:
        print("\n❌ Some tests failed!")
        exit(1)