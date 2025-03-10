import numpy as np
import pandas as pd
from typing import Optional, List, Union
from .base import EventGenerator, EventType

class CUSUMGenerator(EventGenerator):
    """
    Event generator based on CUSUM filter for detecting persistent changes
    in the mean of a price series.
    """
    def __init__(
        self, 
        close: pd.Series,
        threshold: Optional[Union[float, pd.Series]] = None,
        vol_lookback: int = 100,
        vol_mult: float = 1.0
        ):
        """
        Parameters:
            close: Series of close prices
            threshold: Fixed threshold or None for dynamic threshold
            vol_lookback: Lookback period for volatility calculation
            vol_mult: Multiplier for volatility threshold
        """
        # CUSUM detects changes but doesn't specify direction
        super().__init__(event_type=EventType.DIRECTION_AGNOSTIC)
        self.close = close
        self.threshold = threshold
        self.vol_lookback = vol_lookback
        self.vol_mult = vol_mult
        self.events = None
        
    def _get_volatility(self) -> pd.Series:
        """Calculate daily volatility."""
        # Daily volatility calculation
        idx = self.close.index.searchsorted(
            self.close.index - pd.Timedelta(days=1))
        idx = idx[idx > 0]
        prev_idx = pd.Series(
            self.close.index[idx - 1], 
            index=self.close.index[self.close.shape[0] - idx.shape[0]:]
        )
        daily_ret = self.close.loc[prev_idx.index] / \
                   self.close.loc[prev_idx.values].values - 1
        return daily_ret.ewm(span=self.vol_lookback).std()
    
    def generate(self) -> pd.DatetimeIndex:
        """Generate CUSUM events."""
        if self.threshold is None:
            # Use dynamic threshold based on volatility
            threshold = self._get_volatility() * self.vol_mult
        else:
            threshold = self.threshold
            
        tEvents, sPos, sNeg = [], 0, 0
        diff = self.close.diff()
        
        for i in diff.index[1:]:
            h = threshold[i] if isinstance(threshold, pd.Series) else threshold
            
            sPos, sNeg = max(0, sPos + diff.loc[i]), min(0, sNeg + diff.loc[i])
            
            if sNeg < -h:
                sNeg = 0
                tEvents.append(i)
            elif sPos > h:
                sPos = 0
                tEvents.append(i)
        
        self.events = pd.DatetimeIndex(tEvents)
        return self.events
    
    def get_events(self) -> pd.DatetimeIndex:
        if self.events is None:
            raise ValueError("No events generated yet. Call generate() first.")
        return self.events


class VolatilityBreakoutGenerator(EventGenerator):
    """
    Detects events based on volatility breakouts using Bollinger Bands
    and volume confirmation.
    """
    def __init__(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        volume: pd.Series,
        window: int = 20,
        num_std: float = 2.5,
        volume_factor: float = 1.5
    ):
        """
        Parameters:
            close: Series of close prices
            high: Series of high prices
            low: Series of low prices
            volume: Series of volume
            window: Lookback window for calculations
            num_std: Number of standard deviations for Bollinger Bands
            volume_factor: Volume increase factor required for confirmation
        """
        self.close = close
        self.high = high
        self.low = low
        self.volume = volume
        self.window = window
        self.num_std = num_std
        self.volume_factor = volume_factor
        self.events = None
        
    def generate(self) -> pd.DatetimeIndex:
        """Generate volatility breakout events."""
        # Calculate Bollinger Bands
        ma = self.close.rolling(window=self.window).mean()
        std = self.close.rolling(window=self.window).std()
        upper_band = ma + (self.num_std * std)
        lower_band = ma - (self.num_std * std)
        
        # Volume threshold
        volume_ma = self.volume.rolling(window=self.window).mean()
        volume_threshold = volume_ma * self.volume_factor
        
        # Find breakouts with volume confirmation
        breaks_up = (self.high > upper_band) & (self.volume > volume_threshold)
        breaks_down = (self.low < lower_band) & (self.volume > volume_threshold)
        
        # Combine events
        self.events = self.close.index[breaks_up | breaks_down]
        return self.events
    
    def get_events(self) -> pd.DatetimeIndex:
        if self.events is None:
            raise ValueError("No events generated yet. Call generate() first.")
        return self.events
