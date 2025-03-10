"""
Labeling module for creating labels/targets for machine learning models.

This module provides tools for generating labels based on price movement,
events, and other criteria that are useful for training ML models.
"""

from .barriers import SimpleVerticalBarrier, ptsl_simple, get_bins
from .events.factory import EventFactory

# Convenience imports for common event generators
from .events.technical import CUSUMGenerator, VolatilityBreakoutGenerator
from .events.statistical import ZScoreGenerator, GARCHGenerator, StatisticalArbitrageGenerator

# Backward compatibility aliases
CUSUMEvents = CUSUMGenerator
ZScoreEvents = ZScoreGenerator
VolatilityEvents = VolatilityBreakoutGenerator

from .events import (
    EventGenerator, 
    EventType,
    ZScoreGenerator, 
    GARCHGenerator, 
    StatisticalArbitrageGenerator,
    KalmanFilterEventGenerator,
    HPFilterEventGenerator,
    EMAFilterEventGenerator,
    ButterworthFilterEventGenerator,
    SavitzkyGolayEventGenerator
)

__all__ = [
    'SimpleVerticalBarrier',
    'get_events',
    'get_bins',
    'ptsl_simple',
    'EventFactory',
    'CUSUMGenerator',
    'ZScoreGenerator',
    'GARCHGenerator',
    'VolatilityBreakoutGenerator',
    # Backward compatibility
    'CUSUMEvents',
    'ZScoreEvents',
    'VolatilityEvents',
    'EventGenerator', 'EventType',
    'StatisticalArbitrageGenerator',
    'KalmanFilterEventGenerator', 'HPFilterEventGenerator', 'EMAFilterEventGenerator',
    'ButterworthFilterEventGenerator', 'SavitzkyGolayEventGenerator'
]
