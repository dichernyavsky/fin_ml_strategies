"""
Data transformers for feature engineering and data transformation.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Union, Dict, Any

class DataTransformer:
    """
    Transform financial data for analysis and modeling.
    """
    
    @staticmethod
    def calculate_returns(df: pd.DataFrame, price_col: str = 'Close', periods: int = 1) -> pd.Series:
        """
        Calculate returns from price data.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing price data
        price_col : str
            Column name for price data
        periods : int
            Number of periods to use for returns calculation
            
        Returns
        -------
        pd.Series
            Series of returns
        """
        return df[price_col].pct_change(periods)
    
    @staticmethod
    def calculate_log_returns(df: pd.DataFrame, price_col: str = 'Close', periods: int = 1) -> pd.Series:
        """
        Calculate logarithmic returns from price data.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing price data
        price_col : str
            Column name for price data
        periods : int
            Number of periods to use for returns calculation
            
        Returns
        -------
        pd.Series
            Series of logarithmic returns
        """
        return np.log(df[price_col] / df[price_col].shift(periods))
    
    @staticmethod
    def calculate_volatility(df: pd.DataFrame, returns_col: str, window: int = 20) -> pd.Series:
        """
        Calculate rolling volatility from returns data.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing returns data
        returns_col : str
            Column name for returns data
        window : int
            Window size for volatility calculation
            
        Returns
        -------
        pd.Series
            Series of volatility values
        """
        return df[returns_col].rolling(window=window).std() * np.sqrt(window)
    
    @staticmethod
    def normalize(series: pd.Series, method: str = 'zscore') -> pd.Series:
        """
        Normalize a series of values.
        
        Parameters
        ----------
        series : pd.Series
            Series to normalize
        method : str
            Normalization method ('zscore', 'minmax')
            
        Returns
        -------
        pd.Series
            Normalized series
        """
        if method == 'zscore':
            return (series - series.mean()) / series.std()
        elif method == 'minmax':
            return (series - series.min()) / (series.max() - series.min())
        else:
            raise ValueError(f"Unknown normalization method: {method}") 