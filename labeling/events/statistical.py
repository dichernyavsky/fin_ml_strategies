"""
Statistical event generators for financial data.
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, List, Tuple
from scipy import stats
from statsmodels.tsa.stattools import adfuller, coint
from arch import arch_model
from .base import EventGenerator, EventType
from labeling.filters import (
    KalmanFilter, 
    HodrickPrescottFilter, 
    ExponentialMovingAverage, 
    ButterworthFilter, 
    SavitzkyGolayFilter,
    WaveletFilter,
    EMDFilter
)


class ZScoreGenerator(EventGenerator):
    """
    Generates events when price z-scores exceed a threshold (it's a Bollinger Bands-like thing).    
    This generator identifies points where the price deviates significantly
    from its moving average, measured in standard deviations (z-score).
    """
    
    def __init__(self, close: pd.Series, window: int = 20, z_thresh: float = 2.0, 
                 min_periods: Optional[int] = None):
        """
        Parameters:
            close: Series of close prices
            window: Rolling window size for z-score calculation
            z_thresh: Z-score threshold to generate an event
            min_periods: Minimum periods for rolling calculations
        """
        # Z-score can be directional if we track the sign of the deviation
        super().__init__(event_type=EventType.DIRECTION_SPECIFIC)
        self.close = close
        self.window = window
        self.z_thresh = z_thresh
        self.min_periods = min_periods or window
        self.events = None
    
    def generate(self) -> pd.Series:
        """
        Generate events when z-score exceeds threshold.
        
        Returns:
            Series of event timestamps
        """
        # Calculate rolling mean and std
        rolling_mean = self.close.rolling(window=self.window, min_periods=self.min_periods).mean()
        rolling_std = self.close.rolling(window=self.window, min_periods=self.min_periods).std()
        
        # Calculate z-scores
        z_scores = (self.close - rolling_mean) / rolling_std
        
        # Find events where absolute z-score exceeds threshold
        events = pd.Series(0, index=self.close.index)
        
        events[z_scores > self.z_thresh] = -1  
        events[z_scores < -self.z_thresh] = 1 
        
        # Keep only non-zero events
        self.events = events[events != 0]
        return self.events
    
    def get_events(self) -> pd.Series:
        if self.events is None:
            raise ValueError("No events generated yet. Call generate() first.")
        return self.events


class GARCHGenerator(EventGenerator):
    """
    Generates events based on GARCH model volatility forecasts.
    
    This generator identifies points where the actual returns exceed
    the predicted volatility by a certain threshold, indicating potential
    market regime changes or abnormal price movements.
    """
    
    def __init__(self, close: pd.Series, window: int = 100, threshold: float = 2.0,
                 p: int = 1, q: int = 1, forecast_horizon: int = 1):
        """
        Parameters:
            close: Series of close prices
            window: Rolling window size for GARCH model
            threshold: Volatility threshold multiplier
            p: GARCH p parameter (lag order of ARCH term)
            q: GARCH q parameter (lag order of GARCH term)
            forecast_horizon: Forecast horizon for volatility
        """
        # GARCH events can be direction-specific based on return sign
        super().__init__(event_type=EventType.DIRECTION_SPECIFIC)
        self.close = close
        self.window = window
        self.threshold = threshold
        self.p = p
        self.q = q
        self.forecast_horizon = forecast_horizon
        self.events = None
    
    def generate(self) -> pd.Series:
        """
        Generate events when returns exceed GARCH volatility thresholds.
        
        Returns:
            Series with direction values (1 for positive excess returns, -1 for negative)
        """
        # Calculate returns
        returns = self.close.pct_change().dropna()
        
        # Initialize events series
        events = pd.Series(0, index=returns.index)
        
        # Rolling window approach for GARCH model
        for i in range(self.window, len(returns)):
            # Get window of returns
            window_returns = returns.iloc[i-self.window:i]
            
            try:
                # Fit GARCH model
                model = arch_model(window_returns, vol='GARCH', p=self.p, q=self.q)
                model_fit = model.fit(disp='off')
                
                # Get forecast
                forecast = model_fit.forecast(horizon=self.forecast_horizon)
                forecast_vol = np.sqrt(forecast.variance.iloc[-1, 0])
                
                # Check if actual return exceeds threshold * forecasted volatility
                # and assign direction based on return sign
                if returns.iloc[i] > self.threshold * forecast_vol:
                    events.iloc[i] = 1  # Positive excess return
                elif returns.iloc[i] < -self.threshold * forecast_vol:
                    events.iloc[i] = -1  # Negative excess return
            except:
                # Skip errors in GARCH fitting
                continue
        
        # Store and return only non-zero events
        self.events = events[events != 0]
        return self.events
    
    def get_events(self) -> pd.Series:
        if self.events is None:
            raise ValueError("No events generated yet. Call generate() first.")
        return self.events







# TODO: add dependent assets --------------------------------------------------
class CointegrationGenerator(EventGenerator):
    """
    Detects trading opportunities based on cointegration breaks
    between related assets.
    """
    def __init__(
        self,
        primary: pd.Series,
        secondary: pd.Series,
        window: int = 100,
        zscore_thresh: float = 2.0,
        coint_pvalue: float = 0.05
    ):
        """
        Parameters:
            primary: Primary asset price series
            secondary: Secondary asset price series
            window: Rolling window for cointegration testing
            zscore_thresh: Z-score threshold for spread divergence
            coint_pvalue: P-value threshold for cointegration test
        """
        self.primary = primary
        self.secondary = secondary
        self.window = window
        self.zscore_thresh = zscore_thresh
        self.coint_pvalue = coint_pvalue
        self.events = None
        
    def generate(self) -> pd.DatetimeIndex:
        """Generate events based on cointegration breaks."""
        events = []
        
        for i in range(self.window, len(self.primary)):
            # Test cointegration in rolling window
            _, pvalue, _ = coint(
                self.primary[i-self.window:i],
                self.secondary[i-self.window:i]
            )
            
            if pvalue < self.coint_pvalue:  # Series are cointegrated
                # Calculate spread
                spread = self.primary[i-self.window:i] - \
                        self.secondary[i-self.window:i]
                zscore = (spread - spread.mean()) / spread.std()
                
                # Check for deviations
                if abs(zscore.iloc[-1]) > self.zscore_thresh:
                    events.append(self.primary.index[i])
        
        self.events = pd.DatetimeIndex(events)
        return self.events
    
    def get_events(self) -> pd.DatetimeIndex:
        if self.events is None:
            raise ValueError("No events generated yet. Call generate() first.")
        return self.events


class StatisticalArbitrageGenerator(EventGenerator):
    """
    Generates events based on statistical arbitrage signals.
    
    This generator identifies mean-reversion opportunities by measuring
    how much a price series deviates from its statistical equilibrium.
    """
    
    def __init__(self, close: pd.Series, window: int = 60, entry_zscore: float = 2.0, 
                 exit_zscore: float = 0.0, lookback: int = 1, half_life: Optional[int] = None):
        """
        Parameters:
            close: Series of close prices
            window: Lookback window for z-score calculation
            entry_zscore: Z-score threshold for entry signals
            exit_zscore: Z-score threshold for exit (mean-reversion target)
            lookback: Minimum days between signals
            half_life: Half-life for exponential weighting (optional)
        """
        super().__init__(event_type=EventType.DIRECTION_SPECIFIC)
        self.close = close
        self.window = window
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.lookback = lookback
        self.half_life = half_life
        self.events = None
    
    def generate(self) -> pd.DatetimeIndex:
        """
        Generate events for statistical arbitrage entry points.
        
        Returns:
            DatetimeIndex of event timestamps
        """
        # Calculate rolling mean and std with optional exponential weighting
        if self.half_life:
            # Exponentially weighted moments
            span = self.half_life * 2
            ewm_mean = self.close.ewm(span=span).mean()
            ewm_std = self.close.ewm(span=span).std()
            z_scores = (self.close - ewm_mean) / ewm_std
        else:
            # Simple rolling moments
            rolling_mean = self.close.rolling(window=self.window).mean()
            rolling_std = self.close.rolling(window=self.window).std()
            z_scores = (self.close - rolling_mean) / rolling_std
        
        # Find potential entry signals
        entry_signals = (abs(z_scores) > self.entry_zscore)
        
        # Apply lookback constraint to avoid clustered signals
        entry_indices = []
        last_signal_idx = -self.lookback * 2  # Initialize to avoid starting with a wait
        
        for i, (idx, signal) in enumerate(entry_signals.items()):
            if signal and i - last_signal_idx >= self.lookback:
                entry_indices.append(idx)
                last_signal_idx = i
        
        # Store and return events
        self.events = pd.DatetimeIndex(entry_indices)
        return self.events
    
    def get_events(self) -> pd.DatetimeIndex:
        if self.events is None:
            raise ValueError("No events generated yet. Call generate() first.")
        return self.events


class KalmanFilterEvents(EventGenerator):
    """
    Generate events based on Kalman filter prediction errors.
    
    Parameters
    ----------
    transition_covariance : float
        Transition covariance for the Kalman filter
    observation_covariance : float
        Observation covariance for the Kalman filter
    threshold : float
        Threshold for prediction error to generate an event
    """
    
    def __init__(self, transition_covariance=0.01, observation_covariance=0.1, threshold=2.0):
        self.transition_covariance = transition_covariance
        self.observation_covariance = observation_covariance
        self.threshold = threshold
        
    def get_events(self, price_series, **kwargs):
        """
        Generate events when Kalman filter prediction error exceeds threshold.
        
        Returns
        -------
        pd.Series
            1 for positive error exceeding threshold, -1 for negative error
        """
        try:
            from pykalman import KalmanFilter
        except ImportError:
            raise ImportError("pykalman is required for KalmanFilterEvents")
        
        # Initialize Kalman filter
        kf = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            initial_state_mean=price_series.iloc[0],
            initial_state_covariance=1,
            transition_covariance=self.transition_covariance,
            observation_covariance=self.observation_covariance
        )
        
        # Run Kalman filter
        state_means, _ = kf.filter(price_series.values)
        
        # Calculate prediction errors
        pred_errors = price_series.values - state_means.flatten()
        
        # Normalize errors
        norm_errors = pred_errors / np.std(pred_errors)
        
        # Generate events based on threshold
        events = pd.Series(0, index=price_series.index)
        events[norm_errors > self.threshold] = 1
        events[norm_errors < -self.threshold] = -1
        
        # Filter only the event points
        events = events[events != 0]
        return events


class RegimeChangeEvents(EventGenerator):
    """
    Generate events based on Hidden Markov Model regime changes.
    
    Parameters
    ----------
    n_regimes : int
        Number of regimes to detect
    window : int
        Window size for regime detection
    """
    
    def __init__(self, n_regimes=2, window=50):
        self.n_regimes = n_regimes
        self.window = window
        
    def get_events(self, price_series, **kwargs):
        """
        Generate events when the market regime changes according to HMM.
        
        Returns
        -------
        pd.Series
            1 for transition to higher regime, -1 for transition to lower regime
        """
        try:
            from hmmlearn import hmm
        except ImportError:
            raise ImportError("hmmlearn is required for RegimeChangeEvents")
        
        # Calculate returns
        returns = price_series.pct_change().dropna()
        
        # Reshape for HMM
        X = returns.values.reshape(-1, 1)
        
        # Initialize and fit HMM
        model = hmm.GaussianHMM(n_components=self.n_regimes, covariance_type="full")
        model.fit(X)
        
        # Predict hidden states
        hidden_states = model.predict(X)
        
        # Create Series of states
        states = pd.Series(hidden_states, index=returns.index)
        
        # Detect regime changes
        regime_changes = states.diff().dropna()
        
        # Generate events
        events = pd.Series(0, index=regime_changes.index)
        events[regime_changes > 0] = 1  # Transition to higher regime
        events[regime_changes < 0] = -1  # Transition to lower regime
        
        # Filter only the event points
        events = events[events != 0]
        return events


# New filter-based event generators that use the filters.py classes

class KalmanFilterEventGenerator(EventGenerator):
    """
    Generates events using Kalman filter to detect regime changes or abnormal price movements.
    
    This generator uses a Kalman filter to smooth the price series and detect
    when the original price deviates significantly from the filtered state.
    """
    
    def __init__(self, close: pd.Series, 
                 transition_covariance: float = 0.01,
                 observation_covariance: float = 0.1,
                 initial_state_covariance: float = 1.0,
                 threshold: float = 2.0):
        """
        Parameters:
            close: Series of close prices
            transition_covariance: Process noise covariance
            observation_covariance: Measurement noise covariance
            initial_state_covariance: Initial state uncertainty
            threshold: Threshold for deviation between price and filtered state
        """
        super().__init__(event_type=EventType.DIRECTION_SPECIFIC)
        self.close = close
        self.threshold = threshold
        
        # Initialize the Kalman filter
        self.kf = KalmanFilter(
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            initial_state_covariance=initial_state_covariance
        )
    
    def generate(self) -> pd.DatetimeIndex:
        """
        Generate events when price deviates from Kalman filter state.
        
        Returns:
            DatetimeIndex of event timestamps
        """
        # Apply Kalman filter to the price series
        filtered_series = self.kf.filter_series(self.close)
        
        # Calculate deviation
        deviation = self.close - filtered_series
        
        # Scale deviation by rolling standard deviation to get z-score
        window = min(20, len(deviation) // 4)  # Adaptive window size
        deviation_std = deviation.rolling(window=window, min_periods=window//2).std()
        z_deviation = deviation / deviation_std
        
        # Find events where absolute z-deviation exceeds threshold
        events = self.close.index[abs(z_deviation) > self.threshold]
        
        # Store and return events
        self.events = pd.DatetimeIndex(events)
        return self.events


class HPFilterEventGenerator(EventGenerator):
    """
    Generates events using Hodrick-Prescott filter to identify cyclical components.
    
    This generator decomposes the price series into trend and cycle components using
    the HP filter, and generates events when the cyclical component exceeds a threshold.
    """
    
    def __init__(self, close: pd.Series, lambda_param: float = 1600, threshold: float = 2.0):
        """
        Parameters:
            close: Series of close prices
            lambda_param: Smoothing parameter for HP filter
            threshold: Standard deviation threshold for cyclical component
        """
        super().__init__(event_type=EventType.DIRECTION_SPECIFIC)
        self.close = close
        self.threshold = threshold
        
        # Initialize the HP filter
        self.hp_filter = HodrickPrescottFilter(lambda_param=lambda_param)
    
    def generate(self) -> pd.DatetimeIndex:
        """
        Generate events when cyclical component exceeds threshold.
        
        Returns:
            DatetimeIndex of event timestamps
        """
        # Apply HP filter to decompose price series
        trend, cycle = self.hp_filter.filter_series(self.close)
        
        # Calculate standardized cycle
        cycle_mean = cycle.mean()
        cycle_std = cycle.std()
        std_cycle = (cycle - cycle_mean) / cycle_std
        
        # Find events where absolute standardized cycle exceeds threshold
        events = self.close.index[abs(std_cycle) > self.threshold]
        
        # Store and return events
        self.events = pd.DatetimeIndex(events)
        return self.events


class EMAFilterEventGenerator(EventGenerator):
    """
    Generates events based on deviation from Exponential Moving Average.
    
    This generator identifies points where the price deviates significantly
    from its exponential moving average.
    """
    
    def __init__(self, close: pd.Series, span: int = 20, threshold: float = 2.0):
        """
        Parameters:
            close: Series of close prices
            span: Span parameter for EMA
            threshold: Threshold for deviation from EMA in standard deviations
        """
        super().__init__(event_type=EventType.DIRECTION_SPECIFIC)
        self.close = close
        self.threshold = threshold
        
        # Initialize the EMA filter
        self.ema_filter = ExponentialMovingAverage(span=span)
    
    def generate(self) -> pd.DatetimeIndex:
        """
        Generate events when price deviates from EMA.
        
        Returns:
            DatetimeIndex of event timestamps
        """
        # Apply EMA filter
        ema = self.ema_filter.filter_series(self.close)
        
        # Calculate deviation
        deviation = self.close - ema
        
        # Standardize deviation using rolling standard deviation
        window = min(20, len(deviation) // 4)  # Adaptive window
        std_dev = deviation.rolling(window=window, min_periods=window//2).std()
        z_deviation = deviation / std_dev
        
        # Find events where absolute standardized deviation exceeds threshold
        events = self.close.index[abs(z_deviation) > self.threshold]
        
        # Store and return events
        self.events = pd.DatetimeIndex(events)
        return self.events


class ButterworthFilterEventGenerator(EventGenerator):
    """
    Generates events using Butterworth filter to isolate frequency components.
    
    This generator applies a Butterworth filter to remove high-frequency noise
    and identifies significant deviations between the original and filtered series.
    """
    
    def __init__(self, close: pd.Series, cutoff: float = 0.1, 
                 order: int = 2, threshold: float = 2.0):
        """
        Parameters:
            close: Series of close prices
            cutoff: Cutoff frequency for the filter (0.0 to 1.0)
            order: Filter order
            threshold: Threshold for deviation between original and filtered series
        """
        super().__init__(event_type=EventType.DIRECTION_SPECIFIC)
        self.close = close
        self.threshold = threshold
        
        # Initialize the Butterworth filter
        self.butterworth_filter = ButterworthFilter(
            cutoff=cutoff,
            order=order,
            btype='low'  # Low-pass filter to remove high-frequency noise
        )
    
    def generate(self) -> pd.DatetimeIndex:
        """
        Generate events when original price deviates from filtered price.
        
        Returns:
            DatetimeIndex of event timestamps
        """
        # Apply Butterworth filter
        filtered_series = self.butterworth_filter.filter_series(self.close)
        
        # Calculate deviation
        deviation = self.close - filtered_series
        
        # Standardize deviation
        std_deviation = deviation.std()
        z_deviation = deviation / std_deviation
        
        # Find events where absolute standardized deviation exceeds threshold
        events = self.close.index[abs(z_deviation) > self.threshold]
        
        # Store and return events
        self.events = pd.DatetimeIndex(events)
        return self.events


class SavitzkyGolayEventGenerator(EventGenerator):
    """
    Generates events using Savitzky-Golay filter to smooth data while preserving features.
    
    This generator applies a Savitzky-Golay filter to smooth the price series and
    identifies significant deviations between the original and smoothed series.
    """
    
    def __init__(self, close: pd.Series, window_length: int = 11, 
                 polyorder: int = 2, threshold: float = 2.0):
        """
        Parameters:
            close: Series of close prices
            window_length: Length of the filter window (must be odd)
            polyorder: Order of the polynomial used for fitting
            threshold: Threshold for deviation between original and smoothed series
        """
        super().__init__(event_type=EventType.DIRECTION_SPECIFIC)
        self.close = close
        self.threshold = threshold
        
        # Initialize the Savitzky-Golay filter
        self.sg_filter = SavitzkyGolayFilter(
            window_length=window_length,
            polyorder=polyorder
        )
    
    def generate(self) -> pd.DatetimeIndex:
        """
        Generate events when original price deviates from smoothed price.
        
        Returns:
            DatetimeIndex of event timestamps
        """
        # Apply Savitzky-Golay filter
        smoothed_series = self.sg_filter.filter_series(self.close)
        
        # Calculate deviation
        deviation = self.close - smoothed_series
        
        # Standardize deviation
        std_deviation = deviation.std()
        z_deviation = deviation / std_deviation
        
        # Find events where absolute standardized deviation exceeds threshold
        events = self.close.index[abs(z_deviation) > self.threshold]
        
        # Store and return events
        self.events = pd.DatetimeIndex(events)
        return self.events


class WaveletEventGenerator(EventGenerator):
    """
    Generates events using wavelet decomposition to identify anomalies.
    
    This generator uses wavelet transforms to analyze time series at multiple
    resolutions and identify significant deviations in specific frequency bands.
    """
    
    def __init__(self, close: pd.Series, wavelet: str = 'db8', level: int = 3, 
                 threshold: float = 2.0, keep_levels: List[int] = None):
        """
        Parameters:
            close: Series of close prices
            wavelet: Wavelet type ('db8', 'sym4', 'haar', etc.)
            level: Decomposition level
            threshold: Threshold for deviation in standard deviations
            keep_levels: Decomposition levels to analyze (0=approx, 1...n=detail)
        """
        super().__init__(event_type=EventType.DIRECTION_SPECIFIC)
        self.close = close
        self.threshold = threshold
        
        # Initialize the wavelet filter
        self.wavelet_filter = WaveletFilter(wavelet=wavelet, level=level)
        self.keep_levels = keep_levels or [1, 2]  # Default: analyze detail levels 1 and 2
    
    def generate(self) -> pd.DatetimeIndex:
        """
        Generate events based on wavelet decomposition.
        
        Returns:
            DatetimeIndex of event timestamps
        """
        # Decompose using wavelet transform
        components = self.wavelet_filter.decompose(self.close)
        
        # Extract selected components
        selected_components = []
        for i in self.keep_levels:
            comp_key = 'approximation' if i == 0 else f'detail_{i}'
            if comp_key in components:
                selected_components.append(components[comp_key])
        
        # If no valid components, return empty index
        if not selected_components:
            self.events = pd.DatetimeIndex([])
            return self.events
        
        # Combine selected components
        combined = pd.concat(selected_components, axis=1)
        
        # Calculate sum of squares for combined components
        sum_squares = combined.pow(2).sum(axis=1).pow(0.5)
        
        # Calculate z-score
        zscore = (sum_squares - sum_squares.mean()) / sum_squares.std()
        
        # Find events where z-score exceeds threshold
        events = self.close.index[abs(zscore) > self.threshold]
        
        # Store and return events
        self.events = pd.DatetimeIndex(events)
        return self.events


class EMDEventGenerator(EventGenerator):
    """
    Generates events using Empirical Mode Decomposition to identify regime changes.
    
    This generator uses EMD to decompose price series into Intrinsic Mode Functions
    and identifies significant deviations in specific IMFs.
    """
    
    def __init__(self, close: pd.Series, max_imfs: int = 10, 
                 threshold: float = 2.0, imfs_to_analyze: List[int] = None):
        """
        Parameters:
            close: Series of close prices
            max_imfs: Maximum number of IMFs to extract
            threshold: Threshold for deviation in standard deviations
            imfs_to_analyze: IMF indices to analyze (1-based indexing)
        """
        super().__init__(event_type=EventType.DIRECTION_SPECIFIC)
        self.close = close
        self.threshold = threshold
        
        # Initialize the EMD filter
        self.emd_filter = EMDFilter(max_imfs=max_imfs)
        self.imfs_to_analyze = imfs_to_analyze or [2, 3]  # Default: analyze IMFs 2 and 3
    
    def generate(self) -> pd.DatetimeIndex:
        """
        Generate events based on EMD analysis.
        
        Returns:
            DatetimeIndex of event timestamps
        """
        # Decompose using EMD
        imfs = self.emd_filter.decompose(self.close)
        
        # Extract selected IMFs
        selected_imfs = []
        for i in self.imfs_to_analyze:
            imf_key = f'imf_{i}'
            if imf_key in imfs:
                selected_imfs.append(imfs[imf_key])
        
        # If no valid IMFs, return empty index
        if not selected_imfs:
            self.events = pd.DatetimeIndex([])
            return self.events
        
        # Combine selected IMFs
        combined = pd.concat(selected_imfs, axis=1)
        
        # Calculate envelope
        envelope = combined.pow(2).sum(axis=1).pow(0.5)
        
        # Calculate z-score
        zscore = (envelope - envelope.mean()) / envelope.std()
        
        # Find events where z-score exceeds threshold
        events = self.close.index[abs(zscore) > self.threshold]
        
        # Store and return events
        self.events = pd.DatetimeIndex(events)
        return self.events
