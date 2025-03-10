"""
Base classes for event generators used in labeling.
"""

import pandas as pd
from enum import Enum
from abc import ABC, abstractmethod
from typing import Optional, List, Union, Dict, Any

class EventType(Enum):
    """
    Enum for different types of events that can be generated.
    """
    DIRECTION_AGNOSTIC = "direction_agnostic"   # Example: volatility spikes
    DIRECTION_SPECIFIC = "direction_specific"   # Example: price breakouts
    REGIME_SHIFT = "regime_shift"               # Example: trend changes
    ABNORMAL_VOLUME = "abnormal_volume"         # Example: volume spikes
    PATTERN_BASED = "pattern_based"             # Example: chart patterns

class EventGenerator(ABC):
    """
    Abstract base class for all event generators.
    
    An event generator identifies points of interest in financial data,
    which can be used for labeling and strategy development.
    """
    
    def __init__(self, event_type: EventType = EventType.DIRECTION_AGNOSTIC):
        """
        Initialize the event generator.
        
        Parameters
        ----------
        event_type : EventType
            The type of events this generator produces
        """
        self.event_type = event_type
        self.events = None
    
    @abstractmethod
    def generate(self, **kwargs) -> pd.DatetimeIndex:
        """
        Generate events based on the implementation criteria.
        
        Returns
        -------
        pd.DatetimeIndex
            DatetimeIndex containing event timestamps
        """
        pass
    
    def get_events(self) -> pd.DatetimeIndex:
        """
        Return previously generated events.
        
        Returns
        -------
        pd.DatetimeIndex
            DatetimeIndex containing event timestamps
        
        Raises
        ------
        ValueError
            If no events have been generated yet
        """
        if self.events is None:
            raise ValueError("No events generated yet. Call generate() first.")
        return self.events
    
    def is_directional(self):
        """Return whether this generator provides directional information."""
        return self.event_type == EventType.DIRECTION_SPECIFIC
    
    def __str__(self):
        return self.__class__.__name__