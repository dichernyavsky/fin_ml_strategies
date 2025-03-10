from typing import Dict, Type, Optional, Union
from .base import EventGenerator
from .technical import CUSUMGenerator, VolatilityBreakoutGenerator
from .statistical import ZScoreGenerator, GARCHGenerator, CointegrationGenerator, StatisticalArbitrageGenerator, KalmanFilterEvents, RegimeChangeEvents
from .fundamental import EarningsEvents, SentimentEvents
from .microstructure import OrderImbalanceEvents, VolumeSpikesEvents
from data.sampling import SamplingMethod
from ..sampled_events import SampledEventGenerator

class EventFactory:
    """Factory class for creating event generators."""
    
    _generators: Dict[str, Type[EventGenerator]] = {
        # Technical events
        'cusum': CUSUMGenerator,
        'volatility_breakout': VolatilityBreakoutGenerator,
        
        # Statistical events
        'zscore': ZScoreGenerator,
        'garch': GARCHGenerator,
        'cointegration': CointegrationGenerator,
        'stat_arb': StatisticalArbitrageGenerator,
        'kalman': KalmanFilterEvents,
        'regime_change': RegimeChangeEvents,
        
        # Fundamental events
        'earnings': EarningsEvents,
        'sentiment': SentimentEvents,
        
        # Microstructure events
        'order_imbalance': OrderImbalanceEvents,
        'volume_spikes': VolumeSpikesEvents
    }
    
    @classmethod
    def register(cls, name: str, generator: Type[EventGenerator]):
        """Register a new event generator type."""
        cls._generators[name] = generator
    
    @classmethod
    def create(cls, generator_type: str, **kwargs) -> EventGenerator:
        """Create an event generator instance."""
        if generator_type not in cls._generators:
            raise ValueError(f"Unknown generator type: {generator_type}. Available types: {list(cls._generators.keys())}")
        
        return cls._generators[generator_type](**kwargs)
    
    @classmethod
    def create_with_sampling(cls, 
                            generator_type: str, 
                            sampling_method: Union[str, SamplingMethod],
                            sampling_params: Optional[Dict[str, any]] = None,
                            **generator_kwargs) -> SampledEventGenerator:
        """
        Create an event generator with sampling.
        
        Parameters
        ----------
        generator_type : str
            Type of event generator to create
        sampling_method : Union[str, SamplingMethod]
            Sampling method to use
        sampling_params : dict, optional
            Parameters for the sampling method
        **generator_kwargs :
            Parameters for the event generator
            
        Returns
        -------
        SampledEventGenerator
            Event generator with sampling
        """
        # Create the event generator
        generator = cls.create(generator_type, **generator_kwargs)
        
        # Convert string to SamplingMethod enum if needed
        if isinstance(sampling_method, str):
            sampling_method = SamplingMethod(sampling_method)
        
        # Create and return the sampled event generator
        return SampledEventGenerator(
            event_generator=generator,
            sampling_method=sampling_method,
            sampling_params=sampling_params or {}
        )