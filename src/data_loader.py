"""Data loading utilities for JP and US market data"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Loads and processes market data from text files"""
    
    def __init__(self, data_root: Union[str, Path]):
        self.data_root = Path(data_root)
        self.jp_path = self.data_root / "jp"
        self.us_path = self.data_root / "us"
        
    def _parse_file(self, file_path: Path) -> pd.DataFrame:
        """Parse a single data file"""
        try:
            df = pd.read_csv(
                file_path,
                names=['ticker', 'period', 'date', 'time', 'open', 'high', 'low', 'close', 'volume', 'open_interest']
            )
            
            # Skip header row if present
            if df.iloc[0]['ticker'] == '<TICKER>':
                df = df.iloc[1:].copy()
            
            # Parse date
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
            
            # Convert price columns to numeric
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert volume to numeric
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            
            # Clean ticker name
            df['ticker'] = df['ticker'].str.replace('.JP', '').str.replace('.US', '')
            
            # Remove any invalid dates or prices
            df = df.dropna(subset=['date'] + price_cols)
            
            # Set date as index
            df = df.set_index('date').sort_index()
            
            return df
            
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return pd.DataFrame()
    
    def load_jp_data(self, categories: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """Load Japanese market data
        
        Args:
            categories: List of categories to load (e.g., ['tse etfs', 'tse indices'])
                       If None, loads all categories
        """
        if not self.jp_path.exists():
            raise FileNotFoundError(f"JP data path not found: {self.jp_path}")
        
        data = {}
        
        # Get all subdirectories if categories not specified
        if categories is None:
            categories = [d.name for d in self.jp_path.iterdir() if d.is_dir()]
        
        for category in categories:
            category_path = self.jp_path / category
            if not category_path.exists():
                logger.warning(f"Category not found: {category}")
                continue
                
            logger.info(f"Loading {category} data...")
            category_data = {}
            
            for file_path in category_path.glob("*.txt"):
                df = self._parse_file(file_path)
                if not df.empty:
                    ticker = file_path.stem.split('.')[0]
                    category_data[ticker] = df
            
            if category_data:
                data[category] = category_data
                logger.info(f"Loaded {len(category_data)} instruments from {category}")
        
        return data
    
    def load_us_data(self, categories: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """Load US market data
        
        Args:
            categories: List of categories to load (e.g., ['nasdaq stocks', 'nyse etfs'])
                       If None, loads all categories
        """
        if not self.us_path.exists():
            raise FileNotFoundError(f"US data path not found: {self.us_path}")
        
        data = {}
        
        # Get all subdirectories if categories not specified
        if categories is None:
            categories = [d.name for d in self.us_path.iterdir() if d.is_dir()]
        
        for category in categories:
            category_path = self.us_path / category
            if not category_path.exists():
                logger.warning(f"Category not found: {category}")
                continue
                
            logger.info(f"Loading {category} data...")
            category_data = {}
            
            # Handle nested directories (like nasdaq stocks/1/)
            if any(d.is_dir() for d in category_path.iterdir()):
                for subdir in category_path.iterdir():
                    if subdir.is_dir():
                        for file_path in subdir.glob("*.txt"):
                            df = self._parse_file(file_path)
                            if not df.empty:
                                ticker = file_path.stem.split('.')[0].upper()
                                category_data[ticker] = df
            else:
                for file_path in category_path.glob("*.txt"):
                    df = self._parse_file(file_path)
                    if not df.empty:
                        ticker = file_path.stem.split('.')[0].upper()
                        category_data[ticker] = df
            
            if category_data:
                data[category] = category_data
                logger.info(f"Loaded {len(category_data)} instruments from {category}")
        
        return data
    
    def get_combined_data(self, tickers: List[str], market: str = 'both') -> pd.DataFrame:
        """Get combined OHLCV data for specified tickers
        
        Args:
            tickers: List of ticker symbols
            market: 'jp', 'us', or 'both'
        
        Returns:
            Multi-level DataFrame with tickers as columns
        """
        all_data = []
        
        if market in ['jp', 'both']:
            jp_data = self.load_jp_data()
            for category_data in jp_data.values():
                for ticker, df in category_data.items():
                    if ticker.upper() in [t.upper() for t in tickers]:
                        df_copy = df.copy()
                        df_copy = df_copy.reset_index()  # Reset date index to column
                        df_copy['ticker'] = ticker.upper()
                        all_data.append(df_copy)
        
        if market in ['us', 'both']:
            us_data = self.load_us_data()
            for category_data in us_data.values():
                for ticker, df in category_data.items():
                    if ticker.upper() in [t.upper() for t in tickers]:
                        df_copy = df.copy()
                        df_copy = df_copy.reset_index()  # Reset date index to column
                        df_copy['ticker'] = ticker.upper()
                        all_data.append(df_copy)
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine all data
        combined = pd.concat(all_data, ignore_index=True)
        
        # Reset date from index if it's there
        if 'date' not in combined.columns and combined.index.name == 'date':
            combined = combined.reset_index()
        
        # Create multi-level DataFrame
        combined = combined.set_index(['date', 'ticker'])
        combined = combined[['open', 'high', 'low', 'close', 'volume']]
        
        # Pivot to get tickers as columns
        result = {}
        for col in ['open', 'high', 'low', 'close', 'volume']:
            result[col] = combined[col].unstack('ticker')
        
        # Create multi-level columns DataFrame
        multi_df = pd.concat(result, axis=1, keys=['open', 'high', 'low', 'close', 'volume'])
        multi_df.columns.names = ['price_type', 'ticker']
        
        return multi_df.sort_index()