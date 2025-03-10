"""
Data loaders for various financial data sources.
"""

import os
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple

class DataLoader(ABC):
    """
    Abstract base class for data loaders.
    """
    
    @abstractmethod
    def load(self, *args, **kwargs) -> pd.DataFrame:
        """
        Load data from a source.
        
        Returns
        -------
        pd.DataFrame
            The loaded data
        """
        pass

class FinancialDataLoader(DataLoader):
    """
    Unified loader for financial data with support for the specific folder structure.
    
    Parameters
    ----------
    base_dir : str
        Base directory containing financial data
    """
    
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
    
    def load(self, filepath: str, **kwargs) -> pd.DataFrame:
        """
        Load data from a CSV file.
        
        Parameters
        ----------
        filepath : str
            Path to the CSV file to load, relative to base_dir
        **kwargs :
            Additional parameters for pd.read_csv
            
        Returns
        -------
        pd.DataFrame
            The loaded data
        """
        full_path = os.path.join(self.base_dir, filepath)
        
        # Check if file is compressed
        compression = None
        if filepath.endswith('.zip'):
            compression = 'zip'
            
        df = pd.read_csv(full_path, compression=compression, **kwargs)
        
        # Process datetime columns if present
        for col in ['date', 'time', 'timestamp']:
            if col in df.columns:
                # Check if timestamps are in milliseconds
                if df[col].dtype == 'int64' or 'ms' in kwargs.get('unit', ''):
                    df[col] = pd.to_datetime(df[col], unit='ms')
                else:
                    df[col] = pd.to_datetime(df[col])
                df.set_index(col, inplace=True)
                break
                
        return df
    
    def load_crypto_ohlcv(self, symbol: str, timeframe: str = '1d', **kwargs) -> pd.DataFrame:
        """
        Load cryptocurrency OHLCV data by timeframe.
        
        Parameters
        ----------
        symbol : str
            Cryptocurrency symbol (e.g., 'BTCUSD')
        timeframe : str
            Timeframe ('1d', '1h', '5m', '1m')
        **kwargs :
            Additional parameters for pd.read_csv
            
        Returns
        -------
        pd.DataFrame
            OHLCV data for the cryptocurrency
        """
        filepath = os.path.join('crypto', timeframe, f"{symbol}.csv")
        return self.load(filepath, **kwargs)
    
    def load_stock_ohlcv(self, symbol: str, timeframe: str = '1d', **kwargs) -> pd.DataFrame:
        """
        Load stock OHLCV data by timeframe.
        
        Parameters
        ----------
        symbol : str
            Stock symbol
        timeframe : str
            Timeframe ('1d')
        **kwargs :
            Additional parameters for pd.read_csv
            
        Returns
        -------
        pd.DataFrame
            OHLCV data for the stock
        """
        filepath = os.path.join('stocks', timeframe, f"{symbol}.csv")
        return self.load(filepath, **kwargs)
    
    def load_crypto_tick_data(self, symbol: str, year: int, month: int, **kwargs) -> pd.DataFrame:
        """
        Load cryptocurrency tick data from the compressed monthly files.
        
        Parameters
        ----------
        symbol : str
            Cryptocurrency symbol (e.g., 'CFXUSDT')
        year : int
            Year of the data
        month : int
            Month of the data (1-12)
        **kwargs :
            Additional parameters for pd.read_csv
            
        Returns
        -------
        pd.DataFrame
            Trade-level tick data
        """
        filepath = os.path.join('crypto', 'tick', symbol, f"{symbol}-trades-{year}-{month:02d}.zip")
        return self.load(filepath, **kwargs)
    
    def load_available_symbols(self, asset_type: str = 'crypto', timeframe: str = '1d') -> List[str]:
        """
        Get list of available symbols for a specific asset type and timeframe.
        
        Parameters
        ----------
        asset_type : str
            Type of asset ('crypto', 'stocks', 'other')
        timeframe : str
            Timeframe to check
            
        Returns
        -------
        List[str]
            List of available symbols
        """
        directory = os.path.join(self.base_dir, asset_type, timeframe)
        
        if not os.path.exists(directory):
            return []
            
        # Get all CSV files and strip extension
        symbols = [file.split('.')[0] for file in os.listdir(directory) 
                  if file.endswith('.csv')]
        
        return symbols
    
    def load_available_tick_symbols(self) -> List[str]:
        """
        Get list of symbols with available tick data.
        
        Returns
        -------
        List[str]
            List of symbols with tick data
        """
        tick_dir = os.path.join(self.base_dir, 'crypto', 'tick')
        
        if not os.path.exists(tick_dir):
            return []
            
        symbols = [dir_name for dir_name in os.listdir(tick_dir) 
                  if os.path.isdir(os.path.join(tick_dir, dir_name))]
        
        return symbols 