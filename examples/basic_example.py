"""Basic example of using the backtesting system"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_loader import DataLoader
from backtester import BacktestEngine
from strategies import SimpleMovingAverageStrategy, MomentumStrategy
from analytics import PerformanceAnalytics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Run basic backtesting example"""
    
    # Initialize data loader - path to parent trading directory
    current_file = Path(__file__).resolve()
    data_root = current_file.parent.parent.parent  # Go up to trading directory
    
    logger.info(f"Current file: {current_file}")
    logger.info(f"Looking for data in: {data_root}")
    
    if not (data_root / 'us').exists():
        logger.error(f"US data directory not found at {data_root / 'us'}")
        logger.info("Available directories in data_root:")
        for item in data_root.iterdir():
            if item.is_dir():
                logger.info(f"  {item.name}")
        logger.info("Trying to find us directory...")
        
        # Look for us directory in current and parent directories
        search_paths = [
            Path.cwd(),
            Path.cwd().parent,
            Path.cwd() / 'trading',
            Path('/Users/salinecrop/Coding/trading')
        ]
        
        for search_path in search_paths:
            if (search_path / 'us').exists():
                logger.info(f"Found us directory at: {search_path}")
                data_root = search_path
                break
        else:
            logger.error("Could not find us data directory")
            return
    loader = DataLoader(data_root)
    
    logger.info("Loading sample data...")
    
    # Load some US stocks for example
    try:
        us_data = loader.load_us_data(['nasdaq etfs'])
        
        if not us_data:
            logger.error("No US data loaded. Please check data files.")
            return
        
        # Get some sample tickers
        sample_tickers = []
        for category_data in us_data.values():
            sample_tickers.extend(list(category_data.keys())[:10])  # First 10 tickers
            break
        
        if not sample_tickers:
            logger.error("No tickers found in data")
            return
        
        logger.info(f"Using tickers: {sample_tickers[:5]}...")  # Show first 5
        
        # Get combined data
        combined_data = loader.get_combined_data(sample_tickers[:5], market='us')
        
        if combined_data.empty:
            logger.error("No combined data available")
            return
        
        logger.info(f"Data shape: {combined_data.shape}")
        logger.info(f"Date range: {combined_data.index.min()} to {combined_data.index.max()}")
        
        # Example 1: Simple Moving Average Strategy
        logger.info("\n=== Running SMA Strategy ===")
        
        # Initialize backtest engine
        engine = BacktestEngine(combined_data, initial_capital=100000)
        
        # Create strategy
        sma_strategy = SimpleMovingAverageStrategy(
            tickers=sample_tickers[:3],  # Use first 3 tickers
            short_window=10,
            long_window=30,
            position_size=0.3  # 30% per position
        )
        
        # Register strategy functions
        engine.register_initialize(sma_strategy.initialize)
        engine.register_handle_data(sma_strategy.handle_data)
        
        # Run backtest
        sma_results = engine.run(
            start_date='2020-01-01',
            end_date='2023-01-01'
        )
        
        logger.info("SMA Strategy Results:")
        logger.info(f"Final Portfolio Value: ${sma_results['portfolio_value'].iloc[-1]:,.2f}")
        logger.info(f"Total Return: {sma_results['cumulative_returns'].iloc[-1]:.2%}")
        
        # Example 2: Momentum Strategy
        logger.info("\n=== Running Momentum Strategy ===")
        
        engine2 = BacktestEngine(combined_data, initial_capital=100000)
        
        momentum_strategy = MomentumStrategy(
            tickers=sample_tickers[:5],
            lookback=20,
            top_n=2,  # Hold top 2 momentum stocks
            rebalance_freq='M'  # Monthly rebalancing
        )
        
        engine2.register_initialize(momentum_strategy.initialize)
        engine2.register_handle_data(momentum_strategy.handle_data)
        
        momentum_results = engine2.run(
            start_date='2020-01-01',
            end_date='2023-01-01'
        )
        
        logger.info("Momentum Strategy Results:")
        logger.info(f"Final Portfolio Value: ${momentum_results['portfolio_value'].iloc[-1]:,.2f}")
        logger.info(f"Total Return: {momentum_results['cumulative_returns'].iloc[-1]:.2%}")
        
        # Performance Analytics
        logger.info("\n=== Performance Analytics ===")
        
        sma_analytics = PerformanceAnalytics(sma_results)
        momentum_analytics = PerformanceAnalytics(momentum_results)
        
        # Calculate metrics
        sma_metrics = sma_analytics.calculate_metrics()
        momentum_metrics = momentum_analytics.calculate_metrics()
        
        print("\nSMA Strategy Metrics:")
        for key, value in sma_metrics.items():
            if isinstance(value, float):
                if 'Return' in key or 'Drawdown' in key:
                    print(f"{key}: {value:.2%}")
                else:
                    print(f"{key}: {value:.3f}")
        
        print("\nMomentum Strategy Metrics:")
        for key, value in momentum_metrics.items():
            if isinstance(value, float):
                if 'Return' in key or 'Drawdown' in key:
                    print(f"{key}: {value:.2%}")
                else:
                    print(f"{key}: {value:.3f}")
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Portfolio values
        axes[0].plot(sma_results.index, sma_results['portfolio_value'], 
                    label='SMA Strategy', linewidth=2)
        axes[0].plot(momentum_results.index, momentum_results['portfolio_value'], 
                    label='Momentum Strategy', linewidth=2)
        axes[0].set_title('Portfolio Value Comparison')
        axes[0].set_ylabel('Portfolio Value ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Cumulative returns
        axes[1].plot(sma_results.index, sma_results['cumulative_returns'], 
                    label='SMA Strategy', linewidth=2)
        axes[1].plot(momentum_results.index, momentum_results['cumulative_returns'], 
                    label='Momentum Strategy', linewidth=2)
        axes[1].set_title('Cumulative Returns Comparison')
        axes[1].set_ylabel('Cumulative Returns')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('strategy_comparison.png', dpi=300, bbox_inches='tight')
        logger.info("Saved comparison plot as 'strategy_comparison.png'")
        
        # Show transactions
        logger.info("\n=== Sample Transactions ===")
        sma_transactions = engine.get_transactions()
        if not sma_transactions.empty:
            logger.info(f"SMA Strategy made {len(sma_transactions)} transactions")
            print(sma_transactions.head(10))
        
        momentum_transactions = engine2.get_transactions()
        if not momentum_transactions.empty:
            logger.info(f"Momentum Strategy made {len(momentum_transactions)} transactions")
            print(momentum_transactions.head(10))
        
        logger.info("\nBacktesting examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in backtesting: {e}")
        raise


if __name__ == "__main__":
    main()