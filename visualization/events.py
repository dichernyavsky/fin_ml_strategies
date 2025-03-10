"""
Visualization tools for analyzing event generation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Union, Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_events(price_series: pd.Series, 
                events: Union[pd.DatetimeIndex, pd.Series],
                window: Optional[Union[int, timedelta, Tuple[datetime, datetime]]] = None,
                volume: Optional[pd.Series] = None,
                additional_indicators: Optional[Dict[str, pd.Series]] = None,
                title: str = "Price Chart with Detected Events",
                use_plotly: bool = False,
                figsize: tuple = (14, 8)) -> Any:
    """
    Plot price series with detected events.
    
    Parameters
    ----------
    price_series : pd.Series
        Series of price data with datetime index
    events : Union[pd.DatetimeIndex, pd.Series]
        Events to plot. If pd.Series, values indicate direction (1, -1)
    window : Optional[Union[int, timedelta, Tuple[datetime, datetime]]]
        Time window to display. Can be:
        - int: number of periods before and after each event
        - timedelta: time period before and after each event
        - tuple of (start_date, end_date): specific date range to display
        - None: shows all data
    volume : Optional[pd.Series]
        Volume data to plot in a subplot
    additional_indicators : Optional[Dict[str, pd.Series]]
        Additional indicators to plot (e.g., {'SMA': sma_series, 'RSI': rsi_series})
    title : str
        Plot title
    use_plotly : bool
        Whether to use Plotly (interactive) or Matplotlib (static)
    figsize : tuple
        Figure size for matplotlib
        
    Returns
    -------
    Any
        Matplotlib figure or Plotly figure
    """
    # Determine if events have direction
    has_direction = isinstance(events, pd.Series)
    
    # Filter data based on window parameter
    filtered_price = price_series.copy()
    filtered_volume = volume.copy() if volume is not None else None
    filtered_indicators = None
    
    if additional_indicators is not None:
        filtered_indicators = {k: v.copy() for k, v in additional_indicators.items()}
    
    if window is not None:
        if isinstance(window, tuple) and len(window) == 2:
            # Window is a date range (start_date, end_date)
            start_date, end_date = window
            mask = (price_series.index >= start_date) & (price_series.index <= end_date)
            filtered_price = price_series.loc[mask]
            
            if volume is not None:
                filtered_volume = volume.loc[mask] if all(mask) else volume.loc[volume.index.isin(filtered_price.index)]
                
            if filtered_indicators is not None:
                for k, v in filtered_indicators.items():
                    filtered_indicators[k] = v.loc[v.index.isin(filtered_price.index)]
                    
        elif isinstance(window, (int, timedelta)):
            # Window is a period around events
            event_times = events.index if has_direction else events
            
            # Create a mask for the time window around each event
            mask = pd.Series(False, index=price_series.index)
            
            for event_time in event_times:
                if isinstance(window, int):
                    # Window is number of periods
                    event_idx = price_series.index.get_loc(event_time)
                    start_idx = max(0, event_idx - window)
                    end_idx = min(len(price_series) - 1, event_idx + window)
                    window_indices = price_series.index[start_idx:end_idx+1]
                    mask.loc[window_indices] = True
                else:
                    # Window is timedelta
                    start_time = event_time - window
                    end_time = event_time + window
                    mask |= (price_series.index >= start_time) & (price_series.index <= end_time)
            
            filtered_price = price_series.loc[mask]
            
            if volume is not None:
                filtered_volume = volume.loc[volume.index.isin(filtered_price.index)]
                
            if filtered_indicators is not None:
                for k, v in filtered_indicators.items():
                    filtered_indicators[k] = v.loc[v.index.isin(filtered_price.index)]
    
    # Filter events to only include those in the filtered price range
    if has_direction:
        filtered_events = events.loc[events.index.isin(filtered_price.index)]
    else:
        filtered_events = events[events.isin(filtered_price.index)]
    
    if use_plotly:
        # Create subplots: 1 for price, 1 for volume if provided
        n_rows = 1 + (filtered_volume is not None) + (filtered_indicators is not None and len(filtered_indicators) > 0)
        fig = make_subplots(rows=n_rows, cols=1, 
                           shared_xaxes=True, 
                           vertical_spacing=0.05,
                           row_heights=[0.6] + [0.2] * (n_rows-1))
        
        # Add price data
        fig.add_trace(
            go.Scatter(x=filtered_price.index, y=filtered_price.values, 
                      name='Price', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Add events
        if has_direction:
            # Add bullish events (direction = 1)
            bullish_events = filtered_events[filtered_events == 1].index
            if len(bullish_events) > 0:
                fig.add_trace(
                    go.Scatter(x=bullish_events, 
                              y=filtered_price.loc[bullish_events],
                              mode='markers',
                              marker=dict(color='green', size=10, symbol='triangle-up'),
                              name='Bullish Events'),
                    row=1, col=1
                )
            
            # Add bearish events (direction = -1)
            bearish_events = filtered_events[filtered_events == -1].index
            if len(bearish_events) > 0:
                fig.add_trace(
                    go.Scatter(x=bearish_events, 
                              y=filtered_price.loc[bearish_events],
                              mode='markers',
                              marker=dict(color='red', size=10, symbol='triangle-down'),
                              name='Bearish Events'),
                    row=1, col=1
                )
        else:
            # Add direction-agnostic events
            fig.add_trace(
                go.Scatter(x=filtered_events, 
                          y=filtered_price.loc[filtered_events],
                          mode='markers',
                          marker=dict(color='purple', size=10),
                          name='Events'),
                row=1, col=1
            )
        
        # Add volume if provided
        if filtered_volume is not None:
            fig.add_trace(
                go.Bar(x=filtered_volume.index, y=filtered_volume.values, name='Volume', marker_color='lightblue'),
                row=2, col=1
            )
        
        # Add additional indicators
        if filtered_indicators is not None:
            row_idx = 2 if filtered_volume is None else 3
            for name, indicator in filtered_indicators.items():
                fig.add_trace(
                    go.Scatter(x=indicator.index, y=indicator.values, name=name),
                    row=row_idx, col=1
                )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_rangeslider_visible=False,
            height=600,
            width=1000,
            showlegend=True
        )
        
        return fig
    
    else:  # Use matplotlib
        # Create figure
        n_subplots = 1 + (filtered_volume is not None) + (filtered_indicators is not None and len(filtered_indicators) > 0)
        fig, axes = plt.subplots(n_subplots, 1, figsize=figsize, sharex=True, 
                                gridspec_kw={'height_ratios': [3] + [1] * (n_subplots-1)})
        
        if n_subplots == 1:
            axes = [axes]  # Make axes iterable if only one subplot
        
        # Plot price
        axes[0].plot(filtered_price.index, filtered_price.values, label='Price', color='blue')
        
        # Plot events
        if has_direction:
            # Plot bullish events
            bullish_events = filtered_events[filtered_events == 1].index
            if len(bullish_events) > 0:
                axes[0].scatter(bullish_events, filtered_price.loc[bullish_events], 
                               color='green', marker='^', s=100, label='Bullish Events')
            
            # Plot bearish events
            bearish_events = filtered_events[filtered_events == -1].index
            if len(bearish_events) > 0:
                axes[0].scatter(bearish_events, filtered_price.loc[bearish_events], 
                               color='red', marker='v', s=100, label='Bearish Events')
        else:
            # Plot direction-agnostic events
            axes[0].scatter(filtered_events, filtered_price.loc[filtered_events], 
                           color='purple', marker='o', s=80, label='Events')
        
        # Plot volume if provided
        if filtered_volume is not None:
            axes[1].bar(filtered_volume.index, filtered_volume.values, color='lightblue', alpha=0.7, label='Volume')
            axes[1].set_ylabel('Volume')
        
        # Plot additional indicators
        if filtered_indicators is not None:
            idx = 2 if filtered_volume is not None else 1
            for name, indicator in filtered_indicators.items():
                axes[idx].plot(indicator.index, indicator.values, label=name)
                axes[idx].set_ylabel(name)
                axes[idx].legend(loc='upper left')
        
        # Format x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        
        # Add legends and labels
        axes[0].set_title(title)
        axes[0].set_ylabel('Price')
        axes[0].legend(loc='upper left')
        
        # Rotate date labels
        fig.autofmt_xdate()
        
        plt.tight_layout()
        return fig

def plot_event_distribution(events: Union[pd.DatetimeIndex, pd.Series],
                           price_series: Optional[pd.Series] = None,
                           returns: Optional[pd.Series] = None,
                           bins: int = 50,
                           title: str = "Event Distribution Analysis",
                           use_plotly: bool = False,
                           figsize: tuple = (14, 10)) -> Any:
    """
    Plot distribution of events and related statistics.
    
    Parameters
    ----------
    events : Union[pd.DatetimeIndex, pd.Series]
        Events to analyze
    price_series : Optional[pd.Series]
        Price series for additional analysis
    returns : Optional[pd.Series]
        Returns series for distribution analysis
    bins : int
        Number of bins for histograms
    title : str
        Plot title
    use_plotly : bool
        Whether to use Plotly or Matplotlib
    figsize : tuple
        Figure size for matplotlib
        
    Returns
    -------
    Any
        Matplotlib figure or Plotly figure
    """
    has_direction = isinstance(events, pd.Series)
    event_times = events.index if has_direction else events
    
    if use_plotly:
        fig = make_subplots(rows=2, cols=2, 
                           subplot_titles=("Event Time Distribution", 
                                          "Events by Day of Week",
                                          "Events by Hour of Day",
                                          "Return Distribution After Events"))
        
        # Event time distribution
        fig.add_trace(
            go.Histogram(x=event_times, nbinsx=bins, name="Event Distribution"),
            row=1, col=1
        )
        
        # Events by day of week
        day_counts = pd.Series(event_times).dt.day_name().value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=day_counts.index, y=day_counts.values, name="Events by Day"),
            row=1, col=2
        )
        
        # Events by hour of day
        hour_counts = pd.Series(event_times).dt.hour.value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=hour_counts.index, y=hour_counts.values, name="Events by Hour"),
            row=2, col=1
        )
        
        # Return distribution after events (if returns provided)
        if returns is not None:
            # Get returns following events
            event_returns = []
            for event_time in event_times:
                if event_time in returns.index:
                    idx = returns.index.get_loc(event_time)
                    if idx + 1 < len(returns):
                        event_returns.append(returns.iloc[idx + 1])
            
            fig.add_trace(
                go.Histogram(x=event_returns, nbinsx=bins, name="Post-Event Returns"),
                row=2, col=2
            )
        
        fig.update_layout(
            title=title,
            height=800,
            width=1000,
            showlegend=True
        )
        
        return fig
    
    else:  # Use matplotlib
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Event time distribution
        axes[0, 0].hist(event_times, bins=bins, color='skyblue', alpha=0.7)
        axes[0, 0].set_title("Event Time Distribution")
        axes[0, 0].set_xlabel("Date")
        axes[0, 0].set_ylabel("Frequency")
        
        # Events by day of week
        day_counts = pd.Series(event_times).dt.day_name().value_counts().sort_index()
        day_counts.plot(kind='bar', ax=axes[0, 1], color='lightgreen')
        axes[0, 1].set_title("Events by Day of Week")
        axes[0, 1].set_xlabel("Day")
        axes[0, 1].set_ylabel("Count")
        
        # Events by hour of day
        hour_counts = pd.Series(event_times).dt.hour.value_counts().sort_index()
        hour_counts.plot(kind='bar', ax=axes[1, 0], color='salmon')
        axes[1, 0].set_title("Events by Hour of Day")
        axes[1, 0].set_xlabel("Hour")
        axes[1, 0].set_ylabel("Count")
        
        # Return distribution after events (if returns provided)
        if returns is not None:
            # Get returns following events
            event_returns = []
            for event_time in event_times:
                if event_time in returns.index:
                    idx = returns.index.get_loc(event_time)
                    if idx + 1 < len(returns):
                        event_returns.append(returns.iloc[idx + 1])
            
            axes[1, 1].hist(event_returns, bins=bins, color='lightblue', alpha=0.7)
            axes[1, 1].set_title("Return Distribution After Events")
            axes[1, 1].set_xlabel("Return")
            axes[1, 1].set_ylabel("Frequency")
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
        
        return fig 