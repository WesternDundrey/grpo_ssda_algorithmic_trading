"""SSDA+GRPO Advanced Trading Strategy Example"""

import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_loader import DataLoader
from backtester import BacktestEngine
from ssda_grpo_strategy import SSDAGRPOStrategy
from analytics import PerformanceAnalytics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def plot_training_progress(strategy_stats: dict):
    """Plot training progress metrics"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('SSDA+GRPO Training Progress', fontsize=16)
    
    # Portfolio value over time
    if strategy_stats['portfolio_history']:
        axes[0, 0].plot(strategy_stats['portfolio_history'], linewidth=2, color='blue')
        axes[0, 0].set_title('Portfolio Value During Training')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Policy and value losses
    grpo_stats = strategy_stats['grpo_stats']
    if grpo_stats['policy_losses']:
        axes[0, 1].plot(grpo_stats['policy_losses'], label='Policy Loss', color='red')
        axes[0, 1].plot(grpo_stats['value_losses'], label='Value Loss', color='orange')
        axes[0, 1].set_title('Training Losses')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Episode returns
    if grpo_stats['episode_returns']:
        axes[0, 2].plot(grpo_stats['episode_returns'], color='green', alpha=0.7)
        axes[0, 2].axhline(y=np.mean(grpo_stats['episode_returns']), 
                          color='darkgreen', linestyle='--', label='Mean Return')
        axes[0, 2].set_title('Episode Returns')
        axes[0, 2].set_ylabel('Return')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    
    # Action distribution
    action_dist = strategy_stats['action_distribution']
    actions = list(action_dist.keys())
    counts = list(action_dist.values())
    colors = ['gray', 'green', 'red']
    
    axes[1, 0].bar(actions, counts, color=colors)
    axes[1, 0].set_title('Action Distribution')
    axes[1, 0].set_ylabel('Count')
    
    # Reward history
    if hasattr(strategy_stats, 'reward_history') and strategy_stats.get('total_rewards', 0) != 0:
        # Calculate cumulative rewards
        rewards = [strategy_stats['total_rewards'] / max(1, len(strategy_stats['portfolio_history']))] * len(strategy_stats['portfolio_history'])
        cumulative_rewards = np.cumsum(rewards) if rewards else [0]
        
        axes[1, 1].plot(cumulative_rewards, color='purple')
        axes[1, 1].set_title('Cumulative Rewards')
        axes[1, 1].set_ylabel('Cumulative Reward')
        axes[1, 1].grid(True, alpha=0.3)
    
    # Training statistics summary
    stats_text = f"""
    Training Episodes: {strategy_stats['episode_count']}
    Final Portfolio: ${strategy_stats['portfolio_history'][-1]:,.2f}
    Total Rewards: {strategy_stats['total_rewards']:.4f}
    Avg Reward: {strategy_stats['avg_reward']:.6f}
    SSDA Fitted: {strategy_stats['ssda_fitted']}
    """
    
    axes[1, 2].text(0.1, 0.5, stats_text, transform=axes[1, 2].transAxes, 
                    fontsize=11, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    axes[1, 2].set_title('Training Summary')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    return fig


def compare_strategies(results_dict: dict):
    """Compare different strategy results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Strategy Comparison: SSDA+GRPO vs Traditional', fontsize=16)
    
    # Portfolio values
    for name, results in results_dict.items():
        axes[0, 0].plot(results.index, results['portfolio_value'], 
                       label=name, linewidth=2)
    axes[0, 0].set_title('Portfolio Value Comparison')
    axes[0, 0].set_ylabel('Portfolio Value ($)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Cumulative returns
    for name, results in results_dict.items():
        axes[0, 1].plot(results.index, results['cumulative_returns'], 
                       label=name, linewidth=2)
    axes[0, 1].set_title('Cumulative Returns Comparison')
    axes[0, 1].set_ylabel('Cumulative Returns')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Risk-return scatter
    metrics_data = []
    for name, results in results_dict.items():
        analytics = PerformanceAnalytics(results)
        metrics = analytics.calculate_metrics()
        
        axes[1, 0].scatter(metrics['Volatility'], metrics['Annualized Return'], 
                          s=100, label=name, alpha=0.7)
        
        metrics_data.append({
            'Strategy': name,
            'Return': metrics['Annualized Return'],
            'Volatility': metrics['Volatility'],
            'Sharpe': metrics['Sharpe Ratio'],
            'Max Drawdown': metrics['Max Drawdown']
        })
    
    axes[1, 0].set_xlabel('Volatility')
    axes[1, 0].set_ylabel('Annualized Return')
    axes[1, 0].set_title('Risk-Return Profile')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Performance metrics table
    metrics_df = pd.DataFrame(metrics_data)
    
    # Create table
    table_data = []
    for _, row in metrics_df.iterrows():
        table_data.append([
            row['Strategy'],
            f"{row['Return']:.2%}",
            f"{row['Volatility']:.3f}",
            f"{row['Sharpe']:.3f}",
            f"{row['Max Drawdown']:.2%}"
        ])
    
    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
    table = axes[1, 1].table(cellText=table_data,
                           colLabels=['Strategy', 'Return', 'Volatility', 'Sharpe', 'Max DD'],
                           cellLoc='center',
                           loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    axes[1, 1].set_title('Performance Metrics')
    
    plt.tight_layout()
    return fig


def main():
    """Run SSDA+GRPO strategy example"""
    
    # Initialize data loader
    current_file = Path(__file__).resolve()
    data_root = current_file.parent.parent.parent  # Go up to trading directory
    
    search_paths = [
        data_root,
        Path('/Users/salinecrop/Coding/trading')
    ]
    
    data_root = None
    for search_path in search_paths:
        if (search_path / 'us').exists():
            data_root = search_path
            break
    
    if data_root is None:
        logger.error("Could not find data directory")
        return
    
    loader = DataLoader(data_root)
    logger.info("Loading market data for SSDA+GRPO example...")
    
    try:
        # Load a subset of liquid US ETFs for testing
        us_data = loader.load_us_data(['nasdaq etfs'])
        
        if not us_data:
            logger.error("No US data loaded")
            return
        
        # Select a liquid ETF for testing
        test_ticker = 'SPY'  # Try SPY first
        available_tickers = []
        for category_data in us_data.values():
            available_tickers.extend(list(category_data.keys()))
        
        # If SPY not available, use first available ticker
        if test_ticker not in available_tickers:
            test_ticker = available_tickers[0]
        
        logger.info(f"Using ticker: {test_ticker}")
        
        # Get combined data for the selected ticker
        combined_data = loader.get_combined_data([test_ticker], market='us')
        
        if combined_data.empty:
            logger.error("No combined data available")
            return
        
        logger.info(f"Data shape: {combined_data.shape}")
        logger.info(f"Date range: {combined_data.index.min()} to {combined_data.index.max()}")
        
        # Define test period (use recent 2 years for faster training)
        start_date = '2022-01-01'
        end_date = '2024-01-01'
        
        # === SSDA+GRPO Strategy ===
        logger.info("\n=== Running SSDA+GRPO Strategy ===")
        
        # Configure SSDA+GRPO parameters
        ssda_params = {
            'state_dim': 6,
            'hidden_dims': [32, 16, 8, 16, 32],
            'noise_factor': 0.03,
            'lookback_window': 15
        }
        
        grpo_params = {
            'state_dim': 16,  # Fixed to match TradingState.to_vector() output
            'action_dim': 3,
            'policy_lr': 0.002,
            'value_lr': 0.005,
            'gamma': 0.95,
            'gae_lambda': 0.9
        }
        
        # Initialize strategy
        ssda_grpo_strategy = SSDAGRPOStrategy(
            ticker=test_ticker,
            ssda_params=ssda_params,
            grpo_params=grpo_params,
            lookback_window=40,
            training_episodes=50,  # Reduced for faster testing
            position_size_limit=0.3
        )
        
        # Initialize backtest engine
        engine1 = BacktestEngine(combined_data, initial_capital=100000)
        
        # Register strategy functions
        engine1.register_initialize(ssda_grpo_strategy.initialize)
        engine1.register_handle_data(ssda_grpo_strategy.handle_data)
        
        # Run backtest
        logger.info("Running SSDA+GRPO backtest...")
        ssda_grpo_results = engine1.run(start_date=start_date, end_date=end_date)
        
        # Get strategy statistics
        strategy_stats = ssda_grpo_strategy.get_strategy_stats()
        
        logger.info("SSDA+GRPO Strategy Results:")
        logger.info(f"Final Portfolio Value: ${ssda_grpo_results['portfolio_value'].iloc[-1]:,.2f}")
        logger.info(f"Total Return: {ssda_grpo_results['cumulative_returns'].iloc[-1]:.2%}")
        logger.info(f"Training Episodes: {strategy_stats['episode_count']}")
        logger.info(f"Average Reward: {strategy_stats['avg_reward']:.6f}")
        
        # === Compare with Simple Strategy ===
        logger.info("\n=== Running Simple Buy-and-Hold for Comparison ===")
        
        def simple_initialize(engine):
            engine.universe = [test_ticker]
        
        def simple_handle_data(engine, data):
            # Simple buy and hold
            current_pos = engine.portfolio.get_position(test_ticker)
            if not current_pos:
                engine.order_target_percent(test_ticker, 0.9)  # 90% position
        
        engine2 = BacktestEngine(combined_data, initial_capital=100000)
        engine2.register_initialize(simple_initialize)
        engine2.register_handle_data(simple_handle_data)
        
        simple_results = engine2.run(start_date=start_date, end_date=end_date)
        
        logger.info("Simple Buy-and-Hold Results:")
        logger.info(f"Final Portfolio Value: ${simple_results['portfolio_value'].iloc[-1]:,.2f}")
        logger.info(f"Total Return: {simple_results['cumulative_returns'].iloc[-1]:.2%}")
        
        # === Performance Analysis ===
        logger.info("\n=== Performance Analytics ===")
        
        ssda_grpo_analytics = PerformanceAnalytics(ssda_grpo_results)
        simple_analytics = PerformanceAnalytics(simple_results)
        
        ssda_grpo_metrics = ssda_grpo_analytics.calculate_metrics()
        simple_metrics = simple_analytics.calculate_metrics()
        
        print("\nSSDA+GRPO Strategy Metrics:")
        for key, value in ssda_grpo_metrics.items():
            if isinstance(value, float):
                if 'Return' in key or 'Drawdown' in key:
                    print(f"{key}: {value:.2%}")
                else:
                    print(f"{key}: {value:.3f}")
        
        print("\nSimple Buy-and-Hold Metrics:")
        for key, value in simple_metrics.items():
            if isinstance(value, float):
                if 'Return' in key or 'Drawdown' in key:
                    print(f"{key}: {value:.2%}")
                else:
                    print(f"{key}: {value:.3f}")
        
        # === Visualizations ===
        logger.info("\n=== Creating Visualizations ===")
        
        # Training progress plot
        training_fig = plot_training_progress(strategy_stats)
        training_fig.savefig('ssda_grpo_training.png', dpi=300, bbox_inches='tight')
        logger.info("Saved training progress plot as 'ssda_grpo_training.png'")
        
        # Strategy comparison
        results_dict = {
            'SSDA+GRPO': ssda_grpo_results,
            'Buy & Hold': simple_results
        }
        
        comparison_fig = compare_strategies(results_dict)
        comparison_fig.savefig('ssda_grpo_comparison.png', dpi=300, bbox_inches='tight')
        logger.info("Saved strategy comparison as 'ssda_grpo_comparison.png'")
        
        # Transaction analysis
        logger.info("\n=== Transaction Analysis ===")
        ssda_grpo_transactions = engine1.get_transactions()
        simple_transactions = engine2.get_transactions()
        
        logger.info(f"SSDA+GRPO made {len(ssda_grpo_transactions)} transactions")
        logger.info(f"Buy & Hold made {len(simple_transactions)} transactions")
        
        if not ssda_grpo_transactions.empty:
            print("\nSSDA+GRPO Recent Transactions:")
            print(ssda_grpo_transactions.tail(10))
            
            # Transaction analysis
            total_volume = ssda_grpo_transactions['value'].sum()
            avg_trade_size = ssda_grpo_transactions['value'].mean()
            
            logger.info(f"Total trading volume: ${total_volume:,.2f}")
            logger.info(f"Average trade size: ${avg_trade_size:,.2f}")
        
        # === Advanced Analysis ===
        logger.info("\n=== Advanced Analysis ===")
        
        # Calculate information coefficient (if signal data available)
        if len(strategy_stats['portfolio_history']) > 10:
            portfolio_returns = pd.Series(strategy_stats['portfolio_history']).pct_change().dropna()
            if len(portfolio_returns) > 0:
                logger.info(f"SSDA+GRPO Average Daily Return: {portfolio_returns.mean():.6f}")
                logger.info(f"SSDA+GRPO Daily Volatility: {portfolio_returns.std():.6f}")
                logger.info(f"SSDA+GRPO Daily Sharpe: {portfolio_returns.mean() / (portfolio_returns.std() + 1e-8):.3f}")
        
        # Risk attribution
        if 'max_drawdown' in strategy_stats:
            logger.info(f"Maximum Drawdown Period: Details in strategy statistics")
        
        logger.info("\n=== SSDA+GRPO Example Completed Successfully! ===")
        
        # Print summary
        print(f"""
        
        ╔═══════════════════════════════════════════════════════════╗
        ║                    SSDA+GRPO SUMMARY                      ║
        ╠═══════════════════════════════════════════════════════════╣
        ║ Strategy: State-Space Denoising + Reinforcement Learning ║
        ║ Ticker: {test_ticker:<15} Period: {start_date} to {end_date}        ║
        ║                                                           ║
        ║ SSDA+GRPO Performance:                                    ║
        ║   Final Value: ${ssda_grpo_results['portfolio_value'].iloc[-1]:>10,.2f}                             ║
        ║   Total Return: {ssda_grpo_results['cumulative_returns'].iloc[-1]:>9.2%}                              ║
        ║   Sharpe Ratio: {ssda_grpo_metrics['Sharpe Ratio']:>9.3f}                              ║
        ║                                                           ║
        ║ Buy & Hold Performance:                                   ║
        ║   Final Value: ${simple_results['portfolio_value'].iloc[-1]:>10,.2f}                             ║
        ║   Total Return: {simple_results['cumulative_returns'].iloc[-1]:>9.2%}                              ║
        ║   Sharpe Ratio: {simple_metrics['Sharpe Ratio']:>9.3f}                              ║
        ║                                                           ║
        ║ Training Statistics:                                      ║
        ║   Episodes: {strategy_stats['episode_count']:>6}                                    ║
        ║   Avg Reward: {strategy_stats['avg_reward']:>9.6f}                            ║
        ║   SSDA Fitted: {str(strategy_stats['ssda_fitted']):>7}                             ║
        ╚═══════════════════════════════════════════════════════════╝
        """)
        
    except Exception as e:
        logger.error(f"Error in SSDA+GRPO example: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()