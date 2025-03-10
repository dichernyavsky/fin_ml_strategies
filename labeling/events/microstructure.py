import pandas as pd
import numpy as np
from .base import EventGenerator, EventType

class OrderImbalanceEvents(EventGenerator):
    """
    Generate events based on order book imbalance.
    
    Parameters
    ----------
    threshold : float
        Threshold for order imbalance to generate an event
    window : int
        Window size for calculating order imbalance
    """
    
    def __init__(self, threshold=0.7, window=20):
        # Order imbalance events are directional (buy/sell pressure)
        super().__init__(event_type=EventType.DIRECTION_SPECIFIC)
        self.threshold = threshold
        self.window = window
        self.events = None
    
    def generate(self, order_data):
        """
        Generate events when order imbalance exceeds threshold.
        
        Parameters
        ----------
        order_data : pd.DataFrame
            DataFrame with columns 'bid_volume', 'ask_volume', and index as timestamps
            
        Returns
        -------
        pd.Series
            1 for buy imbalance, -1 for sell imbalance
        """
        # Calculate order imbalance
        imbalance = (order_data['bid_volume'] - order_data['ask_volume']) / (order_data['bid_volume'] + order_data['ask_volume'])
        
        # Smooth imbalance
        smooth_imbalance = imbalance.rolling(window=self.window).mean()
        
        # Generate events based on threshold
        events = pd.Series(0, index=order_data.index)
        events[smooth_imbalance > self.threshold] = 1
        events[smooth_imbalance < -self.threshold] = -1
        
        # Filter only the event points
        self.events = events[events != 0]
        return self.events
    
    def get_events(self):
        if self.events is None:
            raise ValueError("No events generated yet. Call generate() first.")
        return self.events

class VolumeSpikesEvents(EventGenerator):
    """
    Generate events based on abnormal trading volume.
    
    Parameters
    ----------
    threshold : float
        Number of standard deviations for volume to be considered abnormal
    window : int
        Window size for calculating volume statistics
    """
    
    def __init__(self, threshold=3.0, window=50):
        self.threshold = threshold
        self.window = window
    
    def get_events(self, volume_series, price_series=None, **kwargs):
        """
        Generate events when volume exceeds threshold.
        
        Parameters
        ----------
        volume_series : pd.Series
            Series with volume data and timestamps as index
        price_series : pd.Series, optional
            Series with price data to determine direction
            
        Returns
        -------
        pd.Series
            1 for volume spike with price increase, -1 for volume spike with price decrease
        """
        # Calculate rolling mean and std of volume
        vol_mean = volume_series.rolling(window=self.window).mean()
        vol_std = volume_series.rolling(window=self.window).std()
        
        # Identify volume spikes
        vol_spikes = (volume_series - vol_mean) / vol_std > self.threshold
        
        # Generate events
        events = pd.Series(0, index=volume_series.index)
        
        if price_series is not None:
            # If price data is available, determine direction
            price_change = price_series.pct_change()
            events[vol_spikes & (price_change > 0)] = 1
            events[vol_spikes & (price_change < 0)] = -1
        else:
            # Otherwise, just mark volume spikes
            events[vol_spikes] = 1
        
        # Filter only the event points
        events = events[events != 0]
        return events 