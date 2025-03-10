"""
Data processors for cleaning and preprocessing financial data.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Union, Dict, Any

class DataProcessor:
    """
    Process and clean financial data.
    """
    
    @staticmethod
    def fill_missing_values(df: pd.DataFrame, method: str = 'ffill') -> pd.DataFrame:
        """
        Fill missing values in a DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the data
        method : str
            Method to use for filling missing values ('ffill', 'bfill', 'interpolate')
            
        Returns
        -------
        pd.DataFrame
            DataFrame with missing values filled
        """
        df = df.copy()
        
        if method == 'ffill':
            return df.ffill()
        elif method == 'bfill':
            return df.bfill()
        elif method == 'interpolate':
            return df.interpolate()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @staticmethod
    def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Resample OHLCV data to a different timeframe.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing OHLCV data
        timeframe : str
            Target timeframe (e.g., '1H', '4H', '1D')
            
        Returns
        -------
        pd.DataFrame
            Resampled OHLCV data
        """
        # Ensure the DataFrame has a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex")
        
        # Resample the data
        resampled = df.resample(timeframe).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
        
        return resampled.dropna() 