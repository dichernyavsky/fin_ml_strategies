"""
Visualization tools for analyzing barrier placement and outcomes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Union, Optional, List, Dict, Any
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_barriers(price_series: pd.Series,
                 events_with_barriers: pd.DataFrame,
                 event_idx: Optional[Union[int, List[int]]] = None,
                 window: Optional[Union[int, timedelta]] = None,
                 title: str = "Triple Barrier Analysis",
                 use_plotly: bool = False,
                 figsize: tuple = (14, 8)) -> Any:
    """
    Plot price series with triple barriers for specific events.
    
    Parameters
    ----------
    price_series : pd.Series
        Series of price data with datetime index
    events_with_barriers : pd.DataFrame
        DataFrame with events and barriers (must have columns: 't1', 'pt', 'sl', 'bin')
    event_idx : Optional[Union[int, List[int]]]
        Index or list of indices of events to plot (if None, plots all events)
    window : Optional[Union[int, timedelta]]
        Time window to display around each event
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
    # If event_idx is None, plot all events
    if event_idx is None:
        event_idx = range(len(events_with_barriers))
    elif isinstance(event_idx, int):
        event_idx = [event_idx]
    
    if use_plotly:
        fig = go.Figure()
        
        for idx in event_idx:
            # Get event data
            event_time = events_with_barriers.index[idx]
            barrier_time = events_with_barriers.iloc[idx]['t1']
            pt_barrier = events_with_barriers.iloc[idx]['pt']
            sl_barrier = events_with_barriers.iloc[idx]['sl']
            outcome = events_with_barriers.iloc[idx]['bin']
            
            # Determine window to plot
            if window is None:
                # Default: show from event to barrier plus 20% margin
                time_diff = barrier_time - event_time
                start_time = event_time - 0.2 * time_diff
                end_time = barrier_time + 0.2 * time_diff
            elif isinstance(window, int):
                # Window specified as number of bars
                start_idx = max(0, price_series.index.get_loc(event_time) - window // 2)
                end_idx = min(len(price_series) - 1, 
                             price_series.index.get_loc(barrier_time) + window // 2)
                start_time = price_series.index[start_idx]
                end_time = price_series.index[end_idx]
            else:  # window is timedelta
                start_time = event_time - window / 2
                end_time = barrier_time + window / 2
            
            # Get price data for the window
            mask = (price_series.index >= start_time) & (price_series.index <= end_time)
            prices_window = price_series[mask]
            
            # Plot price
            fig.add_trace(
                go.Scatter(x=prices_window.index, y=prices_window.values,
                          mode='lines', name=f'Price (Event {idx})')
            )
            
            # Plot event point
            fig.add_trace(
                go.Scatter(x=[event_time], y=[price_series.loc[event_time]],
                          mode='markers', marker=dict(size=12, color='blue'),
                          name=f'Event {idx}')
            )
            
            # Plot vertical barrier
            fig.add_trace(
                go.Scatter(x=[barrier_time, barrier_time],
                          y=[prices_window.min() * 0.99, prices_window.max() * 1.01],
                          mode='lines', line=dict(color='black', dash='dash'),
                          name='Vertical Barrier')
            )
            
            # Plot profit-taking barrier
            fig.add_trace(
                go.Scatter(x=[event_time, barrier_time],
                          y=[pt_barrier, pt_barrier],
                          mode='lines', line=dict(color='green'),
                          name='Profit-Taking Barrier')
            )
            
            # Plot stop-loss barrier
            fig.add_trace(
                go.Scatter(x=[event_time, barrier_time],
                          y=[sl_barrier, sl_barrier],
                          mode='lines', line=dict(color='red'),
                          name='Stop-Loss Barrier')
            )
            
            # Highlight outcome
            outcome_color = 'green' if outcome == 1 else ('red' if outcome == -1 else 'gray')
            outcome_text = ('Profit-Taking' if outcome == 1 else 
                           ('Stop-Loss' if outcome == -1 else 'Time Exit'))
            
            # Add annotation for outcome
            fig.add_annotation(
                x=barrier_time, y=prices_window.loc[barrier_time],
                text=outcome_text,
                showarrow=True,
                arrowhead=1,
                arrowcolor=outcome_color,
                arrowsize=1.5,
                arrowwidth=2
            )
        
        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Price',
            legend_title='Legend',
            height=600,
            width=1000
        )
        
        return fig
    
    else:  # Use matplotlib
        fig, ax = plt.subplots(figsize=figsize)
        
        for idx in event_idx:
            # Get event data
            event_time = events_with_barriers.index[idx]
            barrier_time = events_with_barriers.iloc[idx]['t1']
            pt_barrier = events_with_barriers.iloc[idx]['pt']
            sl_barrier = events_with_barriers.iloc[idx]['sl']
            outcome = events_with_barriers.iloc[idx]['bin']
            
            # Determine window to plot
            if window is None:
                # Default: show from event to barrier plus 20% margin
                time_diff = barrier_time - event_time
                start_time = event_time - 0.2 * time_diff
                end_time = barrier_time + 0.2 * time_diff
            elif isinstance(window, int):
                # Window specified as number of bars
                start_idx = max(0, price_series.index.get_loc(event_time) - window // 2)
                end_idx = min(len(price_series) - 1, 
                             price_series.index.get_loc(barrier_time) + window // 2)
                start_time = price_series.index[start_idx]
                end_time = price_series.index[end_idx]
            else:  # window is timedelta
                start_time = event_time - window / 2
                end_time = barrier_time + window / 2
            
            # Get price data for the window
            mask = (price_series.index >= start_time) & (price_series.index <= end_time)
            prices_window = price_series[mask]
            
            # Plot price
            ax.plot(prices_window.index, prices_window.values, label=f'Price (Event {idx})')
            
            # Plot event point
            ax.scatter(event_time, price_series.loc[event_time], color='blue', s=100, 
                      label=f'Event {idx}')
            
            # Plot vertical barrier
            ax.axvline(x=barrier_time, color='black', linestyle='--', label='Vertical Barrier')
            
            # Plot profit-taking barrier
            ax.axhline(y=pt_barrier, color='green', xmin=0, 
                      xmax=(barrier_time - start_time) / (end_time - start_time),
                      label='Profit-Taking Barrier')
            
            # Plot stop-loss barrier
            ax.axhline(y=sl_barrier, color='red', xmin=0, 
                      xmax=(barrier_time - start_time) / (end_time - start_time),
                      label='Stop-Loss Barrier')
            
            # Highlight outcome
            outcome_color = 'green' if outcome == 1 else ('red' if outcome == -1 else 'gray')
            outcome_text = ('Profit-Taking' if outcome == 1 else 
                           ('Stop-Loss' if outcome == -1 else 'Time Exit'))
            
            # Add annotation for outcome
            ax.annotate(outcome_text, 
                       xy=(barrier_time, prices_window.loc[barrier_time]),
                       xytext=(10, 0), textcoords='offset points',
                       color=outcome_color, fontweight='bold')
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        # Add title and labels
        ax.set_title(title)
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        
        # Add legend (only show unique entries)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='best')
        
        # Rotate date labels
        fig.autofmt_xdate()
        
        plt.tight_layout()
        return fig

def plot_barrier_statistics(events_with_barriers: pd.DataFrame,
                           title: str = "Barrier Statistics Analysis",
                           use_plotly: bool = False,
                           figsize: tuple = (14, 10)) -> Any:
    """
    Plot statistics about barrier outcomes and characteristics.
    
    Parameters
    ----------
    events_with_barriers : pd.DataFrame
        DataFrame with events and barriers (must have columns: 't1', 'pt', 'sl', 'bin')
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
    # Calculate statistics
    outcome_counts = events_with_barriers['bin'].value_counts().sort_index()
    outcome_pct = outcome_counts / outcome_counts.sum() * 100
    
    # Calculate holding times
    events_with_barriers['holding_time'] = (events_with_barriers['t1'] - 
                                           events_with_barriers.index).dt.total_seconds() / 86400  # in days
    
    # Calculate barrier widths (as percentage of entry price)
    entry_prices = events_with_barriers.index.map(lambda x: events_with_barriers.loc[x, 'close'] 
                                                if 'close' in events_with_barriers.columns else None)
    if entry_prices.iloc[0] is not None:
        events_with_barriers['pt_width'] = (events_with_barriers['pt'] - entry_prices) / entry_prices * 100
        events_with_barriers['sl_width'] = (entry_prices - events_with_barriers['sl']) / entry_prices * 100
    
    if use_plotly:
        fig = make_subplots(rows=2, cols=2, 
                           subplot_titles=("Outcome Distribution", 
                                          "Holding Time Distribution",
                                          "Profit Target Width Distribution",
                                          "Stop Loss Width Distribution"))
        
        # Outcome distribution
        fig.add_trace(
            go.Bar(x=['Loss (-1)', 'Time Exit (0)', 'Profit (1)'], 
                  y=outcome_pct,
                  text=[f'{p:.1f}%' for p in outcome_pct],
                  textposition='auto',
                  marker_color=['red', 'gray', 'green']),
            row=1, col=1
        )
        
        # Holding time distribution
        fig.add_trace(
            go.Histogram(x=events_with_barriers['holding_time'], 
                        nbinsx=30,
                        marker_color='lightblue',
                        name='Holding Time (days)'),
            row=1, col=2
        )
        
        # Barrier width distributions
        if 'pt_width' in events_with_barriers.columns:
            fig.add_trace(
                go.Histogram(x=events_with_barriers['pt_width'], 
                            nbinsx=30,
                            marker_color='green',
                            name='Profit Target Width (%)'),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Histogram(x=events_with_barriers['sl_width'], 
                            nbinsx=30,
                            marker_color='red',
                            name='Stop Loss Width (%)'),
                row=2, col=2
            )
        
        fig.update_layout(
            title=title,
            height=800,
            width=1000,
            showlegend=False
        )
        
        return fig
    
    else:  # Use matplotlib
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Outcome distribution
        axes[0, 0].bar(['Loss (-1)', 'Time Exit (0)', 'Profit (1)'], 
                      outcome_pct,
                      color=['red', 'gray', 'green'])
        axes[0, 0].set_title("Outcome Distribution")
        axes[0, 0].set_ylabel("Percentage (%)")
        
        # Add percentage labels
        for i, p in enumerate(outcome_pct):
            axes[0, 0].annotate(f'{p:.1f}%', 
                               xy=(i, p), 
                               xytext=(0, 5),
                               textcoords='offset points',
                               ha='center')
        
        # Holding time distribution
        axes[0, 1].hist(events_with_barriers['holding_time'], bins=30, color='lightblue')
        axes[0, 1].set_title("Holding Time Distribution")
        axes[0, 1].set_xlabel("Holding Time (days)")
        axes[0, 1].set_ylabel("Frequency")
        
        # Barrier width distributions
        if 'pt_width' in events_with_barriers.columns:
            axes[1, 0].hist(events_with_barriers['pt_width'], bins=30, color='green', alpha=0.7)
            axes[1, 0].set_title("Profit Target Width Distribution")
            axes[1, 0].set_xlabel("Width (%)")
            axes[1, 0].set_ylabel("Frequency")
            
            axes[1, 1].hist(events_with_barriers['sl_width'], bins=30, color='red', alpha=0.7)
            axes[1, 1].set_title("Stop Loss Width Distribution")
            axes[1, 1].set_xlabel("Width (%)")
            axes[1, 1].set_ylabel("Frequency")
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
        
        return fig 