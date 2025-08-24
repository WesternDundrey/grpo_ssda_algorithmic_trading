"""Pre-built trading strategies"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from backtester import BacktestEngine
import logging

logger = logging.getLogger(__name__)


class SimpleMovingAverageStrategy:
    """Simple Moving Average Crossover Strategy"""
    
    def __init__(self, 
                 tickers: List[str],
                 short_window: int = 10,
                 long_window: int = 30,
                 position_size: float = 0.1):
        """
        Initialize SMA strategy
        
        Args:
            tickers: List of tickers to trade
            short_window: Short MA period
            long_window: Long MA period  
            position_size: Position size as fraction of portfolio
        """
        self.tickers = tickers
        self.short_window = short_window
        self.long_window = long_window
        self.position_size = position_size
        self.positions = {}
    
    def initialize(self, engine: BacktestEngine):
        """Initialize strategy"""
        logger.info(f"Initializing SMA strategy with {len(self.tickers)} tickers")
        for ticker in self.tickers:
            self.positions[ticker] = 0
    
    def handle_data(self, engine: BacktestEngine, data):
        """Handle daily data"""
        for ticker in self.tickers:
            # Get historical prices
            prices = engine.get_historical_prices(ticker, self.long_window + 1)
            
            if len(prices) < self.long_window:
                continue
            
            # Calculate moving averages
            short_ma = prices.tail(self.short_window).mean()
            long_ma = prices.tail(self.long_window).mean()
            
            current_position = self.positions.get(ticker, 0)
            
            # Generate signals
            if short_ma > long_ma and current_position == 0:
                # Buy signal
                engine.order_target_percent(ticker, self.position_size)
                self.positions[ticker] = 1
                logger.debug(f"Buy signal for {ticker}")
                
            elif short_ma < long_ma and current_position == 1:
                # Sell signal
                engine.order_target_percent(ticker, 0)
                self.positions[ticker] = 0
                logger.debug(f"Sell signal for {ticker}")


class MeanReversionStrategy:
    """Mean Reversion Strategy using Bollinger Bands"""
    
    def __init__(self,
                 tickers: List[str],
                 window: int = 20,
                 num_std: float = 2.0,
                 position_size: float = 0.05):
        """
        Initialize Mean Reversion strategy
        
        Args:
            tickers: List of tickers to trade
            window: Bollinger Bands period
            num_std: Number of standard deviations
            position_size: Position size as fraction of portfolio
        """
        self.tickers = tickers
        self.window = window
        self.num_std = num_std
        self.position_size = position_size
        self.positions = {}
    
    def initialize(self, engine: BacktestEngine):
        """Initialize strategy"""
        logger.info(f"Initializing Mean Reversion strategy with {len(self.tickers)} tickers")
        for ticker in self.tickers:
            self.positions[ticker] = 0
    
    def handle_data(self, engine: BacktestEngine, data):
        """Handle daily data"""
        for ticker in self.tickers:
            # Get historical prices
            prices = engine.get_historical_prices(ticker, self.window + 1)
            
            if len(prices) < self.window:
                continue
            
            # Calculate Bollinger Bands
            sma = prices.tail(self.window).mean()
            std = prices.tail(self.window).std()
            upper_band = sma + (std * self.num_std)
            lower_band = sma - (std * self.num_std)
            
            current_price = prices.iloc[-1]
            current_position = self.positions.get(ticker, 0)
            
            # Generate signals
            if current_price < lower_band and current_position == 0:
                # Oversold - Buy signal
                engine.order_target_percent(ticker, self.position_size)
                self.positions[ticker] = 1
                logger.debug(f"Oversold buy signal for {ticker}")
                
            elif current_price > upper_band and current_position == 1:
                # Overbought - Sell signal
                engine.order_target_percent(ticker, 0)
                self.positions[ticker] = 0
                logger.debug(f"Overbought sell signal for {ticker}")
                
            elif abs(current_price - sma) < std * 0.5 and current_position == 1:
                # Close to mean - Close position
                engine.order_target_percent(ticker, 0)
                self.positions[ticker] = 0
                logger.debug(f"Mean reversion close for {ticker}")


class MomentumStrategy:
    """Momentum Strategy based on price momentum"""
    
    def __init__(self,
                 tickers: List[str],
                 lookback: int = 20,
                 top_n: int = 5,
                 rebalance_freq: str = 'W'):  # Weekly rebalancing
        """
        Initialize Momentum strategy
        
        Args:
            tickers: List of tickers to trade
            lookback: Lookback period for momentum calculation
            top_n: Number of top momentum stocks to hold
            rebalance_freq: Rebalancing frequency
        """
        self.tickers = tickers
        self.lookback = lookback
        self.top_n = top_n
        self.rebalance_freq = rebalance_freq
        self.last_rebalance = None
        self.current_holdings = set()
    
    def initialize(self, engine: BacktestEngine):
        """Initialize strategy"""
        logger.info(f"Initializing Momentum strategy with {len(self.tickers)} tickers")
    
    def should_rebalance(self, current_date):
        """Check if we should rebalance"""
        if self.last_rebalance is None:
            return True
        
        if self.rebalance_freq == 'D':  # Daily
            return True
        elif self.rebalance_freq == 'W':  # Weekly
            return (current_date - self.last_rebalance).days >= 7
        elif self.rebalance_freq == 'M':  # Monthly
            return (current_date - self.last_rebalance).days >= 30
        
        return False
    
    def handle_data(self, engine: BacktestEngine, data):
        """Handle daily data"""
        if not self.should_rebalance(engine.current_date):
            return
        
        momentum_scores = {}
        
        # Calculate momentum for each ticker
        for ticker in self.tickers:
            prices = engine.get_historical_prices(ticker, self.lookback + 1)
            
            if len(prices) < self.lookback:
                continue
            
            # Calculate momentum as return over lookback period
            momentum = (prices.iloc[-1] / prices.iloc[0]) - 1
            momentum_scores[ticker] = momentum
        
        # Select top N momentum stocks
        if momentum_scores:
            sorted_tickers = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
            selected_tickers = [ticker for ticker, _ in sorted_tickers[:self.top_n]]
            
            # Close positions not in selection
            for ticker in self.current_holdings:
                if ticker not in selected_tickers:
                    engine.order_target_percent(ticker, 0)
            
            # Open/adjust positions in selected tickers
            if selected_tickers:
                target_weight = 1.0 / len(selected_tickers)
                for ticker in selected_tickers:
                    engine.order_target_percent(ticker, target_weight)
            
            self.current_holdings = set(selected_tickers)
            self.last_rebalance = engine.current_date
            
            logger.info(f"Rebalanced to: {selected_tickers}")


class LongShortStrategy:
    """Long-Short Strategy similar to the zipline example"""
    
    def __init__(self,
                 tickers: List[str],
                 signal_data: Optional[pd.DataFrame] = None,
                 n_longs: int = 25,
                 n_shorts: int = 25,
                 min_positions: int = 10):
        """
        Initialize Long-Short strategy
        
        Args:
            tickers: List of tickers to trade
            signal_data: DataFrame with ML predictions/signals
            n_longs: Number of long positions
            n_shorts: Number of short positions
            min_positions: Minimum positions required to trade
        """
        self.tickers = tickers
        self.signal_data = signal_data
        self.n_longs = n_longs
        self.n_shorts = n_shorts
        self.min_positions = min_positions
        self.longs = set()
        self.shorts = set()
    
    def initialize(self, engine: BacktestEngine):
        """Initialize strategy"""
        logger.info(f"Initializing Long-Short strategy")
    
    def get_signals(self, current_date) -> Dict[str, float]:
        """Get signals for current date"""
        if self.signal_data is None:
            # Generate dummy signals based on momentum
            signals = {}
            for ticker in self.tickers:
                # This would be replaced with actual ML predictions
                signals[ticker] = np.random.randn()
            return signals
        else:
            # Use provided signal data
            if current_date in self.signal_data.index:
                return self.signal_data.loc[current_date].to_dict()
            return {}
    
    def handle_data(self, engine: BacktestEngine, data):
        """Handle daily data"""
        signals = self.get_signals(engine.current_date)
        
        if not signals:
            return
        
        # Filter positive and negative signals
        positive_signals = {k: v for k, v in signals.items() if v > 0}
        negative_signals = {k: v for k, v in signals.items() if v < 0}
        
        # Select top longs and shorts
        longs = sorted(positive_signals.items(), key=lambda x: x[1], reverse=True)[:self.n_longs]
        shorts = sorted(negative_signals.items(), key=lambda x: x[1])[:self.n_shorts]
        
        selected_longs = [ticker for ticker, _ in longs]
        selected_shorts = [ticker for ticker, _ in shorts]
        
        # Check minimum positions
        if len(selected_longs) < self.min_positions or len(selected_shorts) < self.min_positions:
            # Close all positions if not enough signals
            for ticker in self.longs | self.shorts:
                engine.order_target_percent(ticker, 0)
            self.longs = set()
            self.shorts = set()
            return
        
        # Close positions not in new selection
        current_holdings = self.longs | self.shorts
        new_holdings = set(selected_longs) | set(selected_shorts)
        
        for ticker in current_holdings:
            if ticker not in new_holdings:
                engine.order_target_percent(ticker, 0)
        
        # Set new positions
        if selected_longs:
            long_weight = 1.0 / len(selected_longs)
            for ticker in selected_longs:
                engine.order_target_percent(ticker, long_weight)
        
        if selected_shorts:
            short_weight = -1.0 / len(selected_shorts)
            for ticker in selected_shorts:
                engine.order_target_percent(ticker, short_weight)
        
        self.longs = set(selected_longs)
        self.shorts = set(selected_shorts)
        
        logger.info(f"Selected {len(selected_longs)} longs and {len(selected_shorts)} shorts")