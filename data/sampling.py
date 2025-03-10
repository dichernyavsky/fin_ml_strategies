import pandas as pd
import numpy as np
from enum import Enum
from typing import Dict, Callable, Optional, Union, List

class SamplingMethod(Enum):
    """Enumeration of available sampling methods."""
    TIME_BASED = "time_based"
    VOLUME_BASED = "volume_based"
    TICK_IMBALANCE = "tick_imbalance"
    VOLUME_IMBALANCE = "volume_imbalance"
    DOLLAR_IMBALANCE = "dollar_imbalance"

class DataSampler:
    """
    A class for sampling raw trade data into various bar types.
    
    This class provides methods to transform raw trade data into different
    types of bars (time-based, volume-based, imbalance-based, etc.) that can
    be used by event generators.
    """
    
    def __init__(self):
        self._sampling_methods = {
            SamplingMethod.TIME_BASED: self._trades_to_time_ohlcv,
            SamplingMethod.VOLUME_BASED: self._trades_to_ohlc_volume,
            SamplingMethod.TICK_IMBALANCE: self._trades_to_tick_imbalance_bars,
            SamplingMethod.VOLUME_IMBALANCE: self._trades_to_volume_imbalance_bars,
            SamplingMethod.DOLLAR_IMBALANCE: self._trades_to_dollar_imbalance_bars
        }
    
    def sample(self, 
               df_trades: pd.DataFrame, 
               method: SamplingMethod, 
               **kwargs) -> pd.DataFrame:
        """
        Sample raw trade data using the specified method.
        
        Parameters
        ----------
        df_trades : pd.DataFrame
            DataFrame containing trade data with columns ['time', 'price', 'qty', 'is_buyer_maker']
        method : SamplingMethod
            The sampling method to use
        **kwargs : 
            Additional parameters specific to the sampling method
            
        Returns
        -------
        pd.DataFrame
            Sampled data in the form of bars
        """
        if method not in self._sampling_methods:
            raise ValueError(f"Unknown sampling method: {method}")
        
        return self._sampling_methods[method](df_trades, **kwargs)
    
    def _trades_to_time_ohlcv(self, df_trades: pd.DataFrame, time_frame: str = '1T') -> pd.DataFrame:
        """
        Aggregates raw trade data into standard time-based OHLCV bars.
        
        Parameters
        ----------
        df_trades : pd.DataFrame
            DataFrame containing trade data with columns ['time', 'price', 'qty', 'is_buyer_maker']
        time_frame : str
            String representing the resampling interval (e.g., '1T' for 1-minute, '5T' for 5-minute)
            
        Returns
        -------
        pd.DataFrame
            DataFrame with time-indexed OHLCV bars
        """
        df_trades = df_trades.copy()
        
        # Ensure 'time' is in datetime format
        df_trades['time'] = pd.to_datetime(df_trades['time'], unit='ms')

        # Set the timestamp as the index for resampling
        df_trades.set_index('time', inplace=True)

        # Create columns for volumes with is_buyer_maker == True (Sell) and False (Buy)
        df_trades['qty_buyer_maker'] = np.where(df_trades['is_buyer_maker'], df_trades['qty'], 0)
        df_trades['qty_non_buyer_maker'] = np.where(~df_trades['is_buyer_maker'], df_trades['qty'], 0)

        # Resample to time-based OHLCV bars
        ohlcv = df_trades.resample(time_frame).agg(
            Open=('price', 'first'),
            High=('price', 'max'),
            Low=('price', 'min'),
            Close=('price', 'last'),
            Volume=('qty', 'sum'),
            Volume_buy=('qty_non_buyer_maker', 'sum'),
            Volume_sell=('qty_buyer_maker', 'sum')
        )

        ohlcv.dropna(subset=['Open'], inplace=True)

        return ohlcv
    
    def _calculate_candle_indices(self, times, qty, threshold):
        indices = [times.iloc[0]]  # Initial timestamp for the first candle
        cumulative_sum = 0
        for time, volume in zip(times, qty):
            cumulative_sum += volume
            if cumulative_sum >= threshold:
                cumulative_sum = 0  # Reset cumulative sum after reaching the threshold
                indices.append(time)  # Add the timestamp of the next transaction as a new boundary
        return indices
    
    def _trades_to_ohlc_volume(self, df_trades, volume_threshold, dollar_vol=False):
        """
        Aggregates raw trade data into volume-based OHLCV bars.
        
        Parameters
        ----------
        df_trades : pd.DataFrame
            DataFrame containing trade data with columns ['time', 'price', 'qty', 'is_buyer_maker']
        volume_threshold : float
            Volume threshold to close a bar
        dollar_vol : bool, optional
            Whether to use dollar volume (price * qty) instead of quantity
            
        Returns
        -------
        pd.DataFrame
            DataFrame with volume-based OHLCV bars
        """
        df_trades = df_trades.copy()
        
        # Ensure 'time' is in datetime format
        df_trades['time'] = pd.to_datetime(df_trades['time'], unit='ms')
        
        # Whether base is volume or dollar volume
        base = 'quote_qty' if dollar_vol and 'quote_qty' in df_trades.columns else 'qty'
        if dollar_vol and base == 'qty':
            df_trades['quote_qty'] = df_trades['price'] * df_trades['qty']
            base = 'quote_qty'
        
        # Apply function to get timestamps for candle boundaries
        candle_boundaries = self._calculate_candle_indices(df_trades['time'], df_trades[base], volume_threshold)
        
        # Convert boundaries to numeric for np.digitize
        boundaries_numeric = np.array([t.value for t in candle_boundaries])
        time_numeric = df_trades['time'].astype(np.int64).values
        
        # Use boundaries to determine candle indices
        df_trades['candle_idx'] = np.digitize(time_numeric, bins=boundaries_numeric) - 1
        
        # Create columns for volumes with is_buyer_maker == True and False
        df_trades['qty_buyer_maker'] = np.where(df_trades['is_buyer_maker'], df_trades[base], 0)
        df_trades['qty_non_buyer_maker'] = np.where(~df_trades['is_buyer_maker'], df_trades[base], 0)

        # Group by candle index to calculate OHLCV and duration
        ohlcv = df_trades.groupby('candle_idx').agg(
            Open=('price', 'first'),
            High=('price', 'max'),
            Low=('price', 'min'),
            Close=('price', 'last'),
            Volume=('qty', 'sum'),
            Volume_buy=('qty_non_buyer_maker', 'sum'),
            Volume_sell=('qty_buyer_maker', 'sum'),
            OpenTime=('time', 'first'),
            CloseTime=('time', 'last')
        ).reset_index(drop=True)

        # Calculate the duration of each candle in seconds
        ohlcv['Duration'] = (ohlcv['CloseTime'] - ohlcv['OpenTime']).dt.total_seconds()
        
        # Set index to OpenTime
        ohlcv.index = ohlcv.OpenTime
        
        return ohlcv
    
    def _calculate_tick_imbalance_indices(self, times, tick_sign, threshold):
        """
        Calculate the timestamps at which the cumulative tick imbalance 
        (i.e. the cumulative sum of tick signs) exceeds the given threshold.
        """
        indices = [times.iloc[0]]  # start with the first timestamp
        imbalance_sum = 0
        for time, sign in zip(times, tick_sign):
            imbalance_sum += sign
            if abs(imbalance_sum) >= threshold:
                indices.append(time)
                imbalance_sum = 0  # reset after bar formation
        return indices
    
    def _trades_to_tick_imbalance_bars(self, df_trades, imbalance_threshold):
        """
        Constructs tick imbalance bars from raw trade data.
        
        Parameters
        ----------
        df_trades : pd.DataFrame
            DataFrame containing trade data with columns ['time', 'price', 'qty', 'is_buyer_maker']
        imbalance_threshold : float
            A numeric threshold for the cumulative tick imbalance
            
        Returns
        -------
        pd.DataFrame
            A DataFrame with tick imbalance bars
        """
        df = df_trades.copy()
        # Convert 'time' to datetime (assumes time is in milliseconds)
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        
        # Create a tick sign: if is_buyer_maker is True (seller-initiated), sign = -1, else +1.
        df['tick_sign'] = np.where(df['is_buyer_maker'], -1, 1)
        
        # Calculate the bar boundaries based on tick imbalance.
        boundaries = self._calculate_tick_imbalance_indices(df['time'], df['tick_sign'], imbalance_threshold)
        
        # Convert boundaries and df['time'] to their numeric (ns) representations.
        boundaries_numeric = np.array([t.value for t in boundaries])
        time_numeric = df['time'].astype(np.int64).values
        
        # Use np.digitize on the numeric values.
        df['bar_idx'] = np.digitize(time_numeric, bins=boundaries_numeric) - 1
        
        # Group by bar_idx and calculate OHLC and volume information.
        bars = df.groupby('bar_idx').agg(
            Open=('price', 'first'),
            High=('price', 'max'),
            Low=('price', 'min'),
            Close=('price', 'last'),
            Volume=('qty', 'sum'),
            StartTime=('time', 'first'),
            EndTime=('time', 'last'),
            TickImbalance=('tick_sign', 'sum')
        ).reset_index(drop=True)
        
        bars.index = bars.StartTime
        return bars
    
    def _calculate_volume_imbalance_indices(self, times, tick_sign, qty, threshold):
        """
        Returns a list of timestamps marking the boundaries of volume imbalance bars.
        
        A bar is closed (and a new one started) when:
            | sum_{i=1}^t (s_i * q_i) | >= threshold
        """
        indices = [times.iloc[0]]
        imbalance = 0.0
        for time, sign, volume in zip(times, tick_sign, qty):
            imbalance += sign * volume
            if abs(imbalance) >= threshold:
                indices.append(time)
                imbalance = 0.0  # reset the imbalance after bar formation
        return indices
    
    def _trades_to_volume_imbalance_bars(self, df_trades, volume_imbalance_threshold):
        """
        Constructs volume imbalance bars from trade data.
        
        Parameters
        ----------
        df_trades : pd.DataFrame
            DataFrame with at least the columns 'time', 'price', 'qty', and 'is_buyer_maker'
        volume_imbalance_threshold : float
            The threshold for the cumulative volume imbalance
            
        Returns
        -------
        pd.DataFrame
            A DataFrame of bars containing OHLC prices, total volume, start/end times, and the net volume imbalance
        """
        df = df_trades.copy()
        # Convert time to datetime
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        # Define trade sign: +1 if buyer-initiated, -1 if seller-initiated
        df['tick_sign'] = np.where(df['is_buyer_maker'], -1, 1)
        
        # Compute bar boundaries based on cumulative volume imbalance
        boundaries = self._calculate_volume_imbalance_indices(df['time'], df['tick_sign'], df['qty'], volume_imbalance_threshold)
        
        # Convert datetime boundaries to numeric (nanoseconds) for np.digitize
        boundaries_numeric = np.array([t.value for t in boundaries])
        time_numeric = df['time'].astype(np.int64).values
        df['bar_idx'] = np.digitize(time_numeric, bins=boundaries_numeric) - 1
        
        # Group by bar index and aggregate statistics
        bars = df.groupby('bar_idx').agg(
            Open=('price', 'first'),
            High=('price', 'max'),
            Low=('price', 'min'),
            Close=('price', 'last'),
            Volume=('qty', 'sum'),
            StartTime=('time', 'first'),
            EndTime=('time', 'last'),
            VolumeImbalance=('tick_sign', lambda x: (df.loc[x.index, 'tick_sign'] * df.loc[x.index, 'qty']).sum())
        ).reset_index(drop=True)
        
        bars.index = bars.StartTime
        return bars
    
    def _calculate_dollar_imbalance_indices(self, times, tick_sign, price, qty, threshold):
        """
        Returns a list of timestamps marking the boundaries of dollar imbalance bars.
        
        A bar is closed when:
            | sum_{i=1}^t (s_i * (p_i * q_i)) | >= threshold
        """
        indices = [times.iloc[0]]
        imbalance = 0.0
        for time, sign, p, volume in zip(times, tick_sign, price, qty):
            imbalance += sign * p * volume
            if abs(imbalance) >= threshold:
                indices.append(time)
                imbalance = 0.0
        return indices
    
    def _trades_to_dollar_imbalance_bars(self, df_trades, dollar_imbalance_threshold):
        """
        Constructs dollar imbalance bars from trade data.
        
        Parameters
        ----------
        df_trades : pd.DataFrame
            DataFrame with columns 'time', 'price', 'qty', and 'is_buyer_maker'
        dollar_imbalance_threshold : float
            The threshold for the cumulative dollar imbalance
            
        Returns
        -------
        pd.DataFrame
            A DataFrame of bars containing OHLC prices, total dollar volume, start/end times,
            and the net dollar imbalance
        """
        df = df_trades.copy()
        # Convert time to datetime
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        # Define trade sign: +1 if buyer-initiated, -1 if seller-initiated
        df['tick_sign'] = np.where(df['is_buyer_maker'], -1, 1)
        
        # Compute boundaries using dollar imbalance (price * qty)
        boundaries = self._calculate_dollar_imbalance_indices(df['time'], df['tick_sign'], df['price'], df['qty'], dollar_imbalance_threshold)
        
        # Convert boundaries to numeric for np.digitize
        boundaries_numeric = np.array([t.value for t in boundaries])
        time_numeric = df['time'].astype(np.int64).values
        df['bar_idx'] = np.digitize(time_numeric, bins=boundaries_numeric) - 1
        
        # Calculate dollar volume for each trade
        df['dollar_volume'] = df['price'] * df['qty']
        
        # Group by bar index and aggregate statistics
        bars = df.groupby('bar_idx').agg(
            Open=('price', 'first'),
            High=('price', 'max'),
            Low=('price', 'min'),
            Close=('price', 'last'),
            DollarVolume=('dollar_volume', 'sum'),
            StartTime=('time', 'first'),
            EndTime=('time', 'last'),
            DollarImbalance=('tick_sign', lambda x: (df.loc[x.index, 'tick_sign'] * df.loc[x.index, 'dollar_volume']).sum())
        ).reset_index(drop=True)
        
        bars.index = bars.StartTime
        return bars 