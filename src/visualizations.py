"""
Visualization Functions for Ocean Wave Analysis

This module provides plotting functions for wave data visualization including:
- Time series plots
- Spectral analysis plots
- Heatmaps
- Statistical distributions
- Multi-station comparisons

Author: Josiah Panak
Date: August 2025
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy import signal
from typing import Optional, List, Tuple


class WaveVisualizer:
    """Create interactive visualizations for ocean wave data."""
    
    # Color schemes
    COLORS = {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e',
        'accent': '#2ca02c',
        'danger': '#d62728',
        'warning': '#ff9896',
        'swell': '#9467bd',
        'steep': '#e377c2',
    }
    
    def __init__(self, theme: str = 'plotly'):
        """
        Initialize visualizer.
        
        Args:
            theme: Plotly template to use
        """
        self.theme = theme
    
    def plot_time_series(
        self,
        df: pd.DataFrame,
        variables: List[str],
        title: str = "Wave Conditions Over Time",
        height: int = 600
    ) -> go.Figure:
        """
        Create interactive time series plot.
        
        Args:
            df: DataFrame with datetime index
            variables: List of column names to plot
            title: Plot title
            height: Figure height in pixels
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=len(variables), 
            cols=1,
            subplot_titles=variables,
            shared_xaxes=True,
            vertical_spacing=0.05
        )
        
        for i, var in enumerate(variables, 1):
            if var in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[var],
                        name=var,
                        mode='lines',
                        line=dict(width=1),
                        hovertemplate='%{x}<br>%{y:.2f}<extra></extra>'
                    ),
                    row=i, col=1
                )
        
        fig.update_layout(
            title=title,
            height=height,
            showlegend=False,
            hovermode='x unified',
            template=self.theme
        )
        
        fig.update_xaxes(title_text="Date", row=len(variables), col=1)
        
        return fig
    
    def plot_wave_conditions(
        self,
        df: pd.DataFrame,
        highlight_storms: bool = True
    ) -> go.Figure:
        """
        Create comprehensive wave conditions visualization.
        
        Args:
            df: Processed dataframe with wave parameters
            highlight_storms: Whether to highlight storm periods
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                'Significant Wave Height (Hs)',
                'Wave Period (Tp)',
                'Wind Speed'
            ),
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.4, 0.3, 0.3]
        )
        
        # Wave height
        if 'Hs' in df.columns:
            color = df['storm_flag'].map({0: self.COLORS['primary'], 1: self.COLORS['danger']}) \
                    if 'storm_flag' in df.columns else self.COLORS['primary']
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['Hs'],
                    name='Hs',
                    mode='lines',
                    line=dict(color=self.COLORS['primary'], width=1),
                    fill='tozeroy',
                    fillcolor='rgba(31, 119, 180, 0.3)'
                ),
                row=1, col=1
            )
            
            # Add storm threshold line
            if highlight_storms:
                fig.add_hline(
                    y=4.0, 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text="Storm threshold",
                    row=1, col=1
                )
        
        # Wave period
        if 'Tp' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['Tp'],
                    name='Tp',
                    mode='lines',
                    line=dict(color=self.COLORS['secondary'], width=1)
                ),
                row=2, col=1
            )
        
        # Wind speed
        if 'wind_speed' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['wind_speed'],
                    name='Wind Speed',
                    mode='lines',
                    line=dict(color=self.COLORS['accent'], width=1),
                    fill='tozeroy',
                    fillcolor='rgba(44, 160, 44, 0.2)'
                ),
                row=3, col=1
            )
        
        fig.update_yaxes(title_text="Hs (m)", row=1, col=1)
        fig.update_yaxes(title_text="Tp (s)", row=2, col=1)
        fig.update_yaxes(title_text="Speed (m/s)", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=1)
        
        fig.update_layout(
            height=800,
            showlegend=False,
            hovermode='x unified',
            template=self.theme,
            title_text="Wave Conditions Dashboard"
        )
        
        return fig
    
    def plot_heatmap(
        self,
        df: pd.DataFrame,
        variable: str = 'Hs',
        aggregation: str = 'mean',
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create heatmap showing temporal patterns.
        
        Args:
            df: DataFrame with datetime index
            variable: Variable to visualize
            aggregation: Aggregation method ('mean', 'max', 'median')
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        if variable not in df.columns:
            raise ValueError(f"Variable '{variable}' not found in dataframe")
        
        # Create pivot table: hour x day_of_year
        pivot_data = df.pivot_table(
            values=variable,
            index=df.index.hour,
            columns=df.index.dayofyear,
            aggfunc=aggregation
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='Viridis',
            hovertemplate='Day: %{x}<br>Hour: %{y}<br>Value: %{z:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title or f'{variable} - {aggregation.capitalize()} by Hour and Day of Year',
            xaxis_title='Day of Year',
            yaxis_title='Hour of Day',
            height=500,
            template=self.theme
        )
        
        return fig
    
    def plot_spectral_analysis(
        self,
        time_series: pd.Series,
        sampling_rate: float = 1/3600,  # Hourly data
        title: str = "Wave Spectrum (FFT)"
    ) -> go.Figure:
        """
        Perform FFT and plot power spectral density.
        
        Args:
            time_series: Time series data
            sampling_rate: Sampling rate in Hz
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        # Remove NaN values
        clean_data = time_series.dropna()
        
        if len(clean_data) < 10:
            raise ValueError("Insufficient data for spectral analysis")
        
        # Compute FFT
        frequencies, power = signal.periodogram(
            clean_data.values,
            fs=sampling_rate,
            scaling='density'
        )
        
        # Convert to periods (more intuitive for waves)
        periods = 1 / (frequencies + 1e-10)  # Avoid division by zero
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=periods[1:],  # Skip DC component
            y=power[1:],
            mode='lines',
            line=dict(color=self.COLORS['primary'], width=2),
            fill='tozeroy',
            fillcolor='rgba(31, 119, 180, 0.3)',
            name='Power Spectral Density'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Period (hours)',
            yaxis_title='Power Spectral Density',
            xaxis_type='log',
            yaxis_type='log',
            height=500,
            template=self.theme,
            showlegend=False
        )
        
        return fig
    
    def plot_wave_rose(
        self,
        df: pd.DataFrame,
        direction_col: str = 'Dp',
        magnitude_col: str = 'Hs',
        title: str = "Wave Rose"
    ) -> go.Figure:
        """
        Create a wave rose (polar histogram).
        
        Args:
            df: DataFrame with wave data
            direction_col: Direction column name
            magnitude_col: Magnitude column name
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        # Filter valid data
        valid_data = df[[direction_col, magnitude_col]].dropna()
        
        # Bin directions
        direction_bins = np.arange(0, 360, 22.5)
        valid_data['dir_bin'] = pd.cut(
            valid_data[direction_col],
            bins=direction_bins,
            labels=direction_bins[:-1],
            include_lowest=True
        )
        
        # Aggregate by direction
        rose_data = valid_data.groupby('dir_bin')[magnitude_col].mean().reset_index()
        rose_data['dir_bin'] = rose_data['dir_bin'].astype(float)
        
        fig = go.Figure()
        
        fig.add_trace(go.Barpolar(
            r=rose_data[magnitude_col],
            theta=rose_data['dir_bin'],
            width=22.5,
            marker_color=rose_data[magnitude_col],
            marker_colorscale='Viridis',
            marker_line_color='white',
            marker_line_width=1,
            hovertemplate='Direction: %{theta}°<br>Avg Hs: %{r:.2f}m<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            polar=dict(
                radialaxis=dict(showticklabels=True, ticks=''),
                angularaxis=dict(direction='clockwise')
            ),
            height=600,
            template=self.theme
        )
        
        return fig
    
    def plot_seasonal_comparison(
        self,
        df: pd.DataFrame,
        variable: str = 'Hs'
    ) -> go.Figure:
        """
        Create seasonal comparison boxplots.
        
        Args:
            df: DataFrame with seasonal data
            variable: Variable to compare
            
        Returns:
            Plotly figure object
        """
        if 'season_name' not in df.columns or variable not in df.columns:
            raise ValueError("DataFrame must have 'season_name' and specified variable")
        
        fig = go.Figure()
        
        seasons = ['Winter', 'Spring', 'Summer', 'Fall']
        colors = [self.COLORS['primary'], self.COLORS['accent'], 
                 self.COLORS['warning'], self.COLORS['secondary']]
        
        for season, color in zip(seasons, colors):
            season_data = df[df['season_name'] == season][variable].dropna()
            
            fig.add_trace(go.Box(
                y=season_data,
                name=season,
                marker_color=color,
                boxmean='sd'
            ))
        
        fig.update_layout(
            title=f'Seasonal Variation in {variable}',
            yaxis_title=variable,
            height=500,
            template=self.theme
        )
        
        return fig
    
    def plot_correlation_matrix(
        self,
        df: pd.DataFrame,
        variables: Optional[List[str]] = None
    ) -> go.Figure:
        """
        Create correlation matrix heatmap.
        
        Args:
            df: DataFrame with variables
            variables: List of variables to include (uses all numeric if None)
            
        Returns:
            Plotly figure object
        """
        if variables is None:
            # Use all numeric columns
            variables = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Compute correlation matrix
        corr_matrix = df[variables].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Variable Correlation Matrix',
            height=600,
            width=700,
            template=self.theme
        )
        
        return fig
    
    def plot_storm_events(
        self,
        df: pd.DataFrame,
        window_hours: int = 48
    ) -> go.Figure:
        """
        Visualize storm events with context window.
        
        Args:
            df: DataFrame with storm flags
            window_hours: Hours before/after storm to show
            
        Returns:
            Plotly figure object
        """
        if 'storm_flag' not in df.columns:
            raise ValueError("DataFrame must have 'storm_flag' column")
        
        # Find storm periods
        storm_indices = df[df['storm_flag'] == 1].index
        
        if len(storm_indices) == 0:
            raise ValueError("No storm events found in data")
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Wave Height During Storms', 'Wind Speed During Storms'),
            shared_xaxes=True
        )
        
        for storm_time in storm_indices[:5]:  # Show first 5 storms
            # Get window around storm
            start = storm_time - pd.Timedelta(hours=window_hours)
            end = storm_time + pd.Timedelta(hours=window_hours)
            window_data = df.loc[start:end]
            
            # Plot wave height
            if 'Hs' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=window_data.index,
                        y=window_data['Hs'],
                        mode='lines',
                        name=f'Storm {storm_time.strftime("%Y-%m-%d")}',
                        line=dict(width=2),
                        showlegend=True
                    ),
                    row=1, col=1
                )
            
            # Plot wind speed
            if 'wind_speed' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=window_data.index,
                        y=window_data['wind_speed'],
                        mode='lines',
                        name=f'Storm {storm_time.strftime("%Y-%m-%d")}',
                        line=dict(width=2),
                        showlegend=False
                    ),
                    row=2, col=1
                )
        
        fig.update_yaxes(title_text="Hs (m)", row=1, col=1)
        fig.update_yaxes(title_text="Wind (m/s)", row=2, col=1)
        fig.update_xaxes(title_text="Time", row=2, col=1)
        
        fig.update_layout(
            height=700,
            hovermode='x unified',
            template=self.theme,
            title_text="Storm Event Analysis"
        )
        
        return fig


def create_summary_figure(df: pd.DataFrame, station_id: str) -> go.Figure:
    """
    Create a comprehensive summary dashboard.
    
    Args:
        df: Processed wave data
        station_id: Station identifier
        
    Returns:
        Plotly figure object with multiple subplots
    """
    visualizer = WaveVisualizer()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Wave Height Time Series',
            'Seasonal Distribution',
            'Wave Period vs Height',
            'Wave Type Distribution'
        ),
        specs=[
            [{'type': 'scatter'}, {'type': 'box'}],
            [{'type': 'scatter'}, {'type': 'bar'}]
        ]
    )
    
    # Time series
    if 'Hs' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Hs'],
                mode='lines',
                name='Hs',
                line=dict(color=visualizer.COLORS['primary'], width=1)
            ),
            row=1, col=1
        )
    
    # Seasonal boxplot
    if 'season_name' in df.columns and 'Hs' in df.columns:
        for season in ['Winter', 'Spring', 'Summer', 'Fall']:
            season_data = df[df['season_name'] == season]['Hs'].dropna()
            fig.add_trace(
                go.Box(y=season_data, name=season, showlegend=False),
                row=1, col=2
            )
    
    # Scatter: Tp vs Hs
    if 'Tp' in df.columns and 'Hs' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['Tp'],
                y=df['Hs'],
                mode='markers',
                marker=dict(size=3, opacity=0.5, color=visualizer.COLORS['secondary']),
                name='Tp vs Hs',
                showlegend=False
            ),
            row=2, col=1
        )
    
    # Wave type distribution
    if 'wave_type' in df.columns:
        wave_counts = df['wave_type'].value_counts()
        fig.add_trace(
            go.Bar(
                x=wave_counts.index,
                y=wave_counts.values,
                marker_color=[visualizer.COLORS['swell'], 
                             visualizer.COLORS['primary'], 
                             visualizer.COLORS['steep']],
                showlegend=False
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        height=800,
        title_text=f"Wave Data Summary - Station {station_id}",
        showlegend=True,
        template='plotly'
    )
    
    return fig