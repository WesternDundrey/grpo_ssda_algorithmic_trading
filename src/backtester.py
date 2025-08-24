"""Modern backtesting engine inspired by zipline but simplified"""

import pandas as pd
import numpy as np
from typing import Dict, List, Callable, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents a position in a security"""
    ticker: str
    shares: float
    cost_basis: float
    current_price: float
    
    @property
    def market_value(self) -> float:
        return self.shares * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.cost_basis) * self.shares
    
    @property
    def weight(self) -> float:
        """Weight as percentage of total portfolio value"""
        return self.market_value


@dataclass
class Transaction:
    """Represents a trade transaction"""
    date: datetime
    ticker: str
    shares: float
    price: float
    commission: float = 0.0
    
    @property
    def value(self) -> float:
        return abs(self.shares * self.price)
    
    @property
    def side(self) -> str:
        return "BUY" if self.shares > 0 else "SELL"


class Portfolio:
    """Portfolio management and tracking"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.transactions: List[Transaction] = []
        
    def get_position(self, ticker: str) -> Optional[Position]:
        """Get position for a ticker"""
        return self.positions.get(ticker)
    
    def update_prices(self, prices: Dict[str, float]):
        """Update current prices for all positions"""
        for ticker, position in self.positions.items():
            if ticker in prices:
                position.current_price = prices[ticker]
    
    @property
    def total_value(self) -> float:
        """Total portfolio value (cash + positions)"""
        return self.cash + sum(pos.market_value for pos in self.positions.values())
    
    @property
    def positions_value(self) -> float:
        """Total value of all positions"""
        return sum(pos.market_value for pos in self.positions.values())
    
    @property
    def leverage(self) -> float:
        """Portfolio leverage"""
        total_abs_value = sum(abs(pos.market_value) for pos in self.positions.values())
        return total_abs_value / self.total_value if self.total_value > 0 else 0
    
    def execute_trade(self, date: datetime, ticker: str, shares: float, 
                     price: float, commission: float = 0.0):
        """Execute a trade and update portfolio"""
        if shares == 0:
            return
            
        transaction = Transaction(date, ticker, shares, price, commission)
        self.transactions.append(transaction)
        
        # Update cash
        self.cash -= (shares * price + commission)
        
        # Update position
        if ticker in self.positions:
            current_pos = self.positions[ticker]
            if current_pos.shares + shares == 0:
                # Closing position
                del self.positions[ticker]
            else:
                # Update position
                total_shares = current_pos.shares + shares
                if total_shares != 0:
                    # Update cost basis using weighted average
                    if shares > 0:  # Adding to position
                        new_cost = ((current_pos.shares * current_pos.cost_basis + 
                                   shares * price) / total_shares)
                    else:  # Reducing position
                        new_cost = current_pos.cost_basis
                    
                    current_pos.shares = total_shares
                    current_pos.cost_basis = new_cost
                    current_pos.current_price = price
                else:
                    del self.positions[ticker]
        else:
            # New position
            if shares != 0:
                self.positions[ticker] = Position(
                    ticker=ticker,
                    shares=shares,
                    cost_basis=price,
                    current_price=price
                )


class BacktestEngine:
    """Main backtesting engine"""
    
    def __init__(self, 
                 data: pd.DataFrame,
                 initial_capital: float = 100000,
                 commission_per_share: float = 0.001,
                 min_commission: float = 1.0):
        """
        Initialize backtest engine
        
        Args:
            data: Multi-level DataFrame with price data (OHLCV x tickers)
            initial_capital: Starting capital
            commission_per_share: Commission per share
            min_commission: Minimum commission per trade
        """
        self.data = data
        self.initial_capital = initial_capital
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission
        
        self.portfolio = Portfolio(initial_capital)
        self.results = []
        self.current_date = None
        self.current_prices = {}
        
        # Strategy functions
        self.initialize_func: Optional[Callable] = None
        self.before_trading_start_func: Optional[Callable] = None
        self.handle_data_func: Optional[Callable] = None
        self.rebalance_func: Optional[Callable] = None
        
    def register_initialize(self, func: Callable):
        """Register initialize function"""
        self.initialize_func = func
        
    def register_before_trading_start(self, func: Callable):
        """Register before trading start function"""
        self.before_trading_start_func = func
        
    def register_handle_data(self, func: Callable):
        """Register handle data function"""
        self.handle_data_func = func
        
    def register_rebalance(self, func: Callable):
        """Register rebalance function"""
        self.rebalance_func = func
    
    def get_current_price(self, ticker: str, price_type: str = 'close') -> float:
        """Get current price for a ticker"""
        try:
            return self.data.loc[self.current_date, (price_type, ticker)]
        except (KeyError, IndexError):
            return np.nan
    
    def get_historical_prices(self, ticker: str, lookback: int, 
                            price_type: str = 'close') -> pd.Series:
        """Get historical prices for a ticker"""
        try:
            end_idx = self.data.index.get_loc(self.current_date)
            start_idx = max(0, end_idx - lookback + 1)
            return self.data.iloc[start_idx:end_idx + 1][(price_type, ticker)]
        except (KeyError, IndexError):
            return pd.Series(dtype=float)
    
    def order(self, ticker: str, shares: float):
        """Place an order for a ticker"""
        if shares == 0:
            return
            
        price = self.get_current_price(ticker, 'open')  # Use next day's open
        if np.isnan(price):
            logger.warning(f"Could not get price for {ticker} on {self.current_date}")
            return
            
        # Calculate commission
        commission = max(abs(shares) * self.commission_per_share, self.min_commission)
        
        # Check if we have enough cash for buy orders
        if shares > 0:
            required_cash = shares * price + commission
            if required_cash > self.portfolio.cash:
                logger.warning(f"Insufficient cash for {ticker} order on {self.current_date}")
                return
        
        # Execute trade
        self.portfolio.execute_trade(self.current_date, ticker, shares, price, commission)
    
    def order_target_percent(self, ticker: str, target_percent: float):
        """Order to achieve target percentage of portfolio"""
        if target_percent == 0:
            # Close position
            current_pos = self.portfolio.get_position(ticker)
            if current_pos:
                self.order(ticker, -current_pos.shares)
            return
        
        target_value = self.portfolio.total_value * target_percent
        current_price = self.get_current_price(ticker, 'open')
        
        if np.isnan(current_price) or current_price <= 0:
            return
        
        target_shares = target_value / current_price
        current_shares = 0
        
        current_pos = self.portfolio.get_position(ticker)
        if current_pos:
            current_shares = current_pos.shares
        
        shares_to_trade = target_shares - current_shares
        
        if abs(shares_to_trade) > 0.1:  # Minimum trade threshold
            self.order(ticker, shares_to_trade)
    
    def order_target_value(self, ticker: str, target_value: float):
        """Order to achieve target dollar value"""
        current_price = self.get_current_price(ticker, 'open')
        
        if np.isnan(current_price) or current_price <= 0:
            return
        
        target_shares = target_value / current_price
        current_shares = 0
        
        current_pos = self.portfolio.get_position(ticker)
        if current_pos:
            current_shares = current_pos.shares
        
        shares_to_trade = target_shares - current_shares
        
        if abs(shares_to_trade) > 0.1:  # Minimum trade threshold
            self.order(ticker, shares_to_trade)
    
    def run(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """Run the backtest"""
        
        # Filter data by date range
        data_subset = self.data.copy()
        if start_date:
            data_subset = data_subset[data_subset.index >= start_date]
        if end_date:
            data_subset = data_subset[data_subset.index <= end_date]
        
        if data_subset.empty:
            raise ValueError("No data available for specified date range")
        
        # Initialize strategy
        if self.initialize_func:
            self.initialize_func(self)
        
        # Run backtest day by day
        for date in data_subset.index:
            self.current_date = date
            
            # Get current prices
            self.current_prices = {}
            for ticker in data_subset.columns.get_level_values('ticker').unique():
                price = self.get_current_price(ticker, 'close')
                if not np.isnan(price):
                    self.current_prices[ticker] = price
            
            # Update portfolio prices
            self.portfolio.update_prices(self.current_prices)
            
            # Before trading start
            if self.before_trading_start_func:
                self.before_trading_start_func(self, data_subset.loc[date])
            
            # Handle data
            if self.handle_data_func:
                self.handle_data_func(self, data_subset.loc[date])
            
            # Rebalance
            if self.rebalance_func:
                self.rebalance_func(self, data_subset.loc[date])
            
            # Record results
            self.results.append({
                'date': date,
                'portfolio_value': self.portfolio.total_value,
                'cash': self.portfolio.cash,
                'positions_value': self.portfolio.positions_value,
                'leverage': self.portfolio.leverage,
                'num_positions': len(self.portfolio.positions)
            })
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(self.results)
        results_df.set_index('date', inplace=True)
        
        # Calculate returns
        results_df['returns'] = results_df['portfolio_value'].pct_change()
        results_df['cumulative_returns'] = (1 + results_df['returns']).cumprod() - 1
        
        return results_df
    
    def get_transactions(self) -> pd.DataFrame:
        """Get all transactions as DataFrame"""
        if not self.portfolio.transactions:
            return pd.DataFrame()
        
        transactions_data = []
        for t in self.portfolio.transactions:
            transactions_data.append({
                'date': t.date,
                'ticker': t.ticker,
                'shares': t.shares,
                'price': t.price,
                'value': t.value,
                'side': t.side,
                'commission': t.commission
            })
        
        return pd.DataFrame(transactions_data)
    
    def get_positions(self) -> pd.DataFrame:
        """Get current positions as DataFrame"""
        if not self.portfolio.positions:
            return pd.DataFrame()
        
        positions_data = []
        for ticker, pos in self.portfolio.positions.items():
            positions_data.append({
                'ticker': ticker,
                'shares': pos.shares,
                'cost_basis': pos.cost_basis,
                'current_price': pos.current_price,
                'market_value': pos.market_value,
                'unrealized_pnl': pos.unrealized_pnl
            })
        
        return pd.DataFrame(positions_data)