"""
Visualization tools for feature analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Union, Optional, Tuple, Any
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_feature_time_series(
    features_df: pd.DataFrame,
    price_series: Optional[pd.Series] = None,
    feature_names: Optional[List[str]] = None,
    n_features: int = 6,
    figsize: Tuple[int, int] = (15, 12),
    use_plotly: bool = False
) -> Any:
    """
    Plot time series of features with optional price overlay.
    
    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame containing generated features
    price_series : Optional[pd.Series]
        Optional price series to overlay
    feature_names : Optional[List[str]]
        List of feature names to plot. If None, plots top n_features
    n_features : int
        Number of features to plot if feature_names is None
    figsize : Tuple[int, int]
        Figure size for matplotlib plot
    use_plotly : bool
        Whether to use plotly for interactive plotting
        
    Returns
    -------
    Any
        Figure object
    """
    # Select features to plot
    if feature_names is None:
        # Just select the first n_features
        feature_names = features_df.columns[:n_features]
    
    if use_plotly:
        # Create subplot grid
        n_plots = len(feature_names)
        fig = make_subplots(
            rows=n_plots, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=feature_names
        )
        
        # Add price series if provided
        if price_series is not None:
            for i, feature_name in enumerate(feature_names):
                fig.add_trace(
                    go.Scatter(
                        x=price_series.index,
                        y=price_series.values,
                        mode='lines',
                        name='Price' if i == 0 else 'Price (hidden)',
                        line=dict(color='black', width=1),
                        opacity=0.5,
                        showlegend=(i == 0)
                    ),
                    row=i+1, col=1
                )
        
        # Add feature plots
        for i, feature_name in enumerate(feature_names):
            feature_data = features_df[feature_name]
            fig.add_trace(
                go.Scatter(
                    x=feature_data.index,
                    y=feature_data.values,
                    mode='lines',
                    name=feature_name,
                    line=dict(width=1.5)
                ),
                row=i+1, col=1
            )
        
        # Update layout
        fig.update_layout(
            height=300 * n_plots,
            width=1000,
            title_text="Feature Time Series",
            showlegend=True
        )
        
        return fig
    
    else:
        # Matplotlib implementation
        n_plots = len(feature_names)
        fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
        
        if n_plots == 1:
            axes = [axes]  # Make it iterable
        
        for i, feature_name in enumerate(feature_names):
            ax = axes[i]
            feature_data = features_df[feature_name]
            
            # Plot feature
            ax.plot(feature_data, label=feature_name)
            ax.set_title(feature_name)
            ax.legend(loc='upper left')
            
            # Add price overlay if provided
            if price_series is not None:
                ax2 = ax.twinx()
                ax2.plot(price_series, color='black', alpha=0.3, label='Price')
                if i == 0:  # Only add price legend on the first subplot
                    ax2.legend(loc='upper right')
            
            # Format dates on x-axis
            if i == n_plots - 1:  # Only on the last subplot
                plt.xticks(rotation=45)
        
        plt.tight_layout()
        return fig

def plot_feature_distributions(
    features_df: pd.DataFrame,
    target: Optional[pd.Series] = None,
    feature_names: Optional[List[str]] = None,
    n_features: int = 9,
    figsize: Tuple[int, int] = (15, 12),
    bins: int = 50
) -> plt.Figure:
    """
    Plot distributions of features, optionally grouped by target.
    
    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame containing generated features
    target : Optional[pd.Series]
        Optional target variable for grouped distributions
    feature_names : Optional[List[str]]
        List of feature names to plot. If None, plots top n_features
    n_features : int
        Number of features to plot if feature_names is None
    figsize : Tuple[int, int]
        Figure size for plot
    bins : int
        Number of bins for histograms
        
    Returns
    -------
    plt.Figure
        Figure object
    """
    # Select features to plot
    if feature_names is None:
        # Just select the first n_features
        feature_names = features_df.columns[:n_features]
    
    # Calculate grid dimensions
    n_cols = 3
    n_rows = (len(feature_names) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, feature_name in enumerate(feature_names):
        if i < len(axes):
            ax = axes[i]
            feature_data = features_df[feature_name]
            
            if target is not None:
                # Create a combined dataframe with feature and target
                plot_data = pd.DataFrame({
                    'feature': feature_data,
                    'target': target
                }).dropna()
                
                # Plot histograms grouped by target
                if plot_data['target'].nunique() <= 5:  # Categorical target
                    for target_value in sorted(plot_data['target'].unique()):
                        subset = plot_data[plot_data['target'] == target_value]['feature']
                        ax.hist(subset, bins=bins, alpha=0.5, label=f'Target={target_value}')
                    ax.legend()
                else:  # Continuous target (use hexbin)
                    hb = ax.hexbin(
                        plot_data['feature'], 
                        plot_data['target'],
                        gridsize=20, 
                        cmap='viridis'
                    )
                    plt.colorbar(hb, ax=ax)
            else:
                # Simple histogram
                ax.hist(feature_data.dropna(), bins=bins)
            
            ax.set_title(feature_name)
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
    
    # Hide unused subplots
    for i in range(len(feature_names), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig

def plot_feature_relationships(
    features_df: pd.DataFrame,
    target: Optional[pd.Series] = None,
    feature_names: Optional[List[str]] = None,
    n_features: int = 20
) -> plt.Figure:
    """
    Plot feature relationships using a correlation matrix.
    
    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame containing generated features
    target : Optional[pd.Series]
        Optional target variable for correlation analysis
    feature_names : Optional[List[str]]
        List of feature names to plot. If None, plots top n_features
    n_features : int
        Number of features to plot if feature_names is None
        
    Returns
    -------
    plt.Figure
        Figure object
    """
    # Select features to plot
    if feature_names is None:
        # Just select the first n_features
        feature_names = features_df.columns[:n_features]
    
    # Calculate correlation matrix
    corr_matrix = features_df[feature_names].corr()
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    ax.set_title('Feature Correlation Matrix')
    
    plt.tight_layout()
    plt.show()
    
    return fig 