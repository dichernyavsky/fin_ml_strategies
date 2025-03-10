"""
Event-based labeling module for financial data.
"""

from .base import EventGenerator, EventType
from .technical import CUSUMGenerator, VolatilityBreakoutGenerator
from .statistical import (
    ZScoreGenerator, 
    GARCHGenerator, 
    StatisticalArbitrageGenerator,
    KalmanFilterEventGenerator,
    HPFilterEventGenerator,
    EMAFilterEventGenerator,
    ButterworthFilterEventGenerator,
    SavitzkyGolayEventGenerator
)
from .fundamental import EarningsEvents, SentimentEvents
from .microstructure import OrderImbalanceEvents, VolumeSpikesEvents
from .factory import EventFactory

# Backward compatibility aliases
CUSUMEvents = CUSUMGenerator
ZScoreEvents = ZScoreGenerator
VolatilityEvents = VolatilityBreakoutGenerator

__all__ = [
    'EventGenerator', 'EventType',
    'EventFactory',
    # Technical
    'CUSUMGenerator',
    'VolatilityBreakoutGenerator',
    # Statistical
    'ZScoreGenerator',
    'GARCHGenerator',
    'StatisticalArbitrageGenerator',
    'KalmanFilterEventGenerator',
    'HPFilterEventGenerator',
    'EMAFilterEventGenerator',
    'ButterworthFilterEventGenerator',
    'SavitzkyGolayEventGenerator',
    # Fundamental
    'EarningsEvents',
    'SentimentEvents',
    # Microstructure
    'OrderImbalanceEvents',
    'VolumeSpikesEvents',
    # Backward compatibility
    'CUSUMEvents',
    'ZScoreEvents',
    'VolatilityEvents'
]
