import pandas as pd
import numpy as np
from .base import EventGenerator, EventType

class EarningsEvents(EventGenerator):
    """
    Generate events based on earnings surprises.
    
    Parameters
    ----------
    surprise_threshold : float
        Threshold for earnings surprise to generate an event (percentage)
    """
    
    def __init__(self, surprise_threshold=5.0):
        # Earnings events are directional (positive/negative surprise)
        super().__init__(event_type=EventType.DIRECTION_SPECIFIC)
        self.surprise_threshold = surprise_threshold
        self.events = None
    
    def generate(self, earnings_data):
        """
        Generate events when earnings surprise exceeds threshold.
        
        Parameters
        ----------
        earnings_data : pd.DataFrame
            DataFrame with columns 'actual', 'estimate', and index as dates
            
        Returns
        -------
        pd.Series
            1 for positive surprise, -1 for negative surprise
        """
        # Calculate surprise percentage
        surprise_pct = (earnings_data['actual'] - earnings_data['estimate']) / abs(earnings_data['estimate']) * 100
        
        # Generate events based on threshold
        events = pd.Series(0, index=earnings_data.index)
        events[surprise_pct > self.surprise_threshold] = 1
        events[surprise_pct < -self.surprise_threshold] = -1
        
        # Filter only the event points
        self.events = events[events != 0]
        return self.events
    
    def get_events(self):
        if self.events is None:
            raise ValueError("No events generated yet. Call generate() first.")
        return self.events

class SentimentEvents(EventGenerator):
    """
    Generate events based on news sentiment analysis.
    
    Parameters
    ----------
    sentiment_threshold : float
        Threshold for sentiment score to generate an event
    """
    
    def __init__(self, sentiment_threshold=0.5):
        self.sentiment_threshold = sentiment_threshold
    
    def get_events(self, sentiment_series, **kwargs):
        """
        Generate events when sentiment exceeds threshold.
        
        Parameters
        ----------
        sentiment_series : pd.Series
            Series with sentiment scores (-1 to 1) and dates as index
            
        Returns
        -------
        pd.Series
            1 for positive sentiment, -1 for negative sentiment
        """
        # Generate events based on threshold
        events = pd.Series(0, index=sentiment_series.index)
        events[sentiment_series > self.sentiment_threshold] = 1
        events[sentiment_series < -self.sentiment_threshold] = -1
        
        # Filter only the event points
        events = events[events != 0]
        return events 