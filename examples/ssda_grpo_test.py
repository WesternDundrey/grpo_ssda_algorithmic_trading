"""Quick test of SSDA+GRPO system components"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
from ssda import SSDA
from grpo import GRPO, Experience
from ssda_grpo_strategy import SSDAGRPOStrategy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_ssda():
    """Test SSDA component"""
    logger.info("Testing SSDA component...")
    
    # Create synthetic price data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    
    # Generate realistic price movements
    returns = np.random.normal(0.001, 0.02, 100)
    prices = 100 * np.cumprod(1 + returns)
    
    # Create OHLCV data
    price_data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, 100)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, 100))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, 100))),
        'close': prices,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Initialize and train SSDA
    ssda = SSDA(state_dim=4, hidden_dims=[16, 8, 4, 8, 16], lookback_window=10)
    
    try:
        history = ssda.fit(price_data)
        logger.info(f"SSDA training completed. Loss: {history['autoencoder_history']['final_loss']:.6f}")
        
        # Test prediction
        recent_data = price_data.tail(20)
        denoised_features, states = ssda.denoise_and_predict(recent_data)
        
        logger.info(f"Denoised features shape: {denoised_features.shape}")
        logger.info(f"State representation shape: {states.shape}")
        
        # Test trading signal generation
        signal = ssda.get_trading_signal(recent_data)
        logger.info(f"Trading signal: {signal}")
        
        return True
        
    except Exception as e:
        logger.error(f"SSDA test failed: {e}")
        return False


def test_grpo():
    """Test GRPO component"""
    logger.info("Testing GRPO component...")
    
    try:
        # Initialize GRPO
        grpo = GRPO(state_dim=5, action_dim=3)
        
        # Create synthetic experiences
        for i in range(50):
            state = np.random.randn(5)
            action = np.random.randint(0, 3)
            reward = np.random.normal(0, 1)
            next_state = np.random.randn(5)
            
            experience = Experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=(i % 10 == 9),
                info={}
            )
            
            grpo.store_experience(experience)
        
        # Test action selection
        test_state = np.random.randn(5)
        action = grpo.get_action(test_state)
        logger.info(f"GRPO selected action: {action}")
        
        # Test training update
        update_stats = grpo.update(batch_size=16, epochs=2)
        logger.info(f"GRPO update stats: {update_stats}")
        
        # Test statistics
        stats = grpo.get_training_stats()
        logger.info(f"GRPO training stats keys: {list(stats.keys())}")
        
        return True
        
    except Exception as e:
        logger.error(f"GRPO test failed: {e}")
        return False


def test_integrated_strategy():
    """Test integrated SSDA+GRPO strategy"""
    logger.info("Testing integrated SSDA+GRPO strategy...")
    
    try:
        # Create strategy
        strategy = SSDAGRPOStrategy(
            ticker='TEST',
            ssda_params={'state_dim': 4, 'hidden_dims': [8, 4, 8], 'lookback_window': 5},
            grpo_params={'state_dim': 10, 'action_dim': 3},
            lookback_window=20,
            training_episodes=5,
            position_size_limit=0.2
        )
        
        logger.info("Strategy initialized successfully")
        
        # Test state creation with mock data
        mock_prices = pd.Series([100, 101, 99, 102, 98], name='close')
        mock_volumes = pd.Series([1000, 1100, 900, 1200, 800], name='volume')
        
        # Test technical indicators
        tech_indicators = strategy._calculate_technical_indicators(mock_prices, mock_volumes)
        logger.info(f"Technical indicators shape: {tech_indicators.shape}")
        logger.info(f"Technical indicators: {tech_indicators}")
        
        # Get strategy stats
        stats = strategy.get_strategy_stats()
        logger.info(f"Strategy stats keys: {list(stats.keys())}")
        
        return True
        
    except Exception as e:
        logger.error(f"Integrated strategy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    logger.info("=== SSDA+GRPO System Tests ===")
    
    tests = [
        ("SSDA Component", test_ssda),
        ("GRPO Component", test_grpo),
        ("Integrated Strategy", test_integrated_strategy)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Print summary
    logger.info("\n=== Test Results Summary ===")
    all_passed = True
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.info("\n🎉 All tests passed! SSDA+GRPO system is working correctly.")
        
        # Print system overview
        print("""
        ╔═══════════════════════════════════════════════════════════════════╗
        ║                    SSDA+GRPO SYSTEM OVERVIEW                      ║
        ╠═══════════════════════════════════════════════════════════════════╣
        ║                                                                   ║
        ║ 🧠 SSDA (State-Space Denoising Autoencoder):                     ║
        ║    • Kalman filtering for state representation                    ║
        ║    • Neural denoising of price signals                           ║
        ║    • Technical indicator integration                              ║
        ║                                                                   ║
        ║ 🤖 GRPO (Generalized Reward Policy Optimization):                ║
        ║    • Policy gradient reinforcement learning                       ║
        ║    • Value function approximation                                 ║
        ║    • PPO-style clipping and GAE advantages                       ║
        ║                                                                   ║
        ║ 🔗 Integration Features:                                          ║
        ║    • Denoised signals feed into RL state space                   ║
        ║    • Custom reward function for trading performance              ║
        ║    • Risk-adjusted position sizing                               ║
        ║    • Real-time adaptation to market conditions                   ║
        ║                                                                   ║
        ║ 📊 Key Capabilities:                                             ║
        ║    • Multi-timeframe signal processing                           ║
        ║    • Adaptive exploration vs exploitation                        ║
        ║    • Transaction cost and risk awareness                         ║
        ║    • Comprehensive performance analytics                         ║
        ║                                                                   ║
        ╚═══════════════════════════════════════════════════════════════════╝
        """)
        
        logger.info("System is ready for live trading backtests!")
        
    else:
        logger.error("\n❌ Some tests failed. Please check the implementation.")


if __name__ == "__main__":
    main()