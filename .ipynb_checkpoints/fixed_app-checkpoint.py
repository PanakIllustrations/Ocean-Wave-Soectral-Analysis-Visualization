"""
Ocean Wave Analysis Dashboard - FIXED VERSION

If main app.py is stuck loading, use this version.

Run with: streamlit run fixed_app.py

Author: Josiah Panak
Date: August 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / 'ocean-wave-analysis' / 'src'
if src_path.exists():
    sys.path.append(str(src_path))
else:
    # Try alternate path
    src_path = Path(__file__).parent / 'src'
    if src_path.exists():
        sys.path.append(str(src_path))

# Page configuration
st.set_page_config(
    page_title="Ocean Wave Analysis",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Try to import custom modules
try:
    from data_processor import NDBCDataProcessor
    from visualizations import WaveVisualizer, create_summary_figure
    MODULES_LOADED = True
except ImportError as e:
    MODULES_LOADED = False
    st.error(f"Error loading modules: {e}")
    st.info("Make sure you're running from the project root directory")
    st.stop()

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600, show_spinner=False)
def load_data(station_id: str, year: int):
    """Load and cache data from NDBC with timeout."""
    try:
        processor = NDBCDataProcessor()
        
        # Try to get historical data with shorter timeout
        raw_data = processor.get_historical_year(station_id, year, timeout=15)
        
        if raw_data is None:
            # Try multiple stations
            raw_data, station_id = processor.try_multiple_stations(year)
            
        if raw_data is not None and len(raw_data) > 0:
            processed_data = processor.clean_and_process(raw_data)
            stats = processor.get_summary_stats(processed_data)
            return processed_data, stats, None
        else:
            return None, None, "No data available for this station/year"
            
    except Exception as e:
        return None, None, f"Error: {str(e)}"


def main():
    """Main application."""
    
    # Header
    st.markdown('<h1 class="main-header">Ocean Wave Analysis Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar controls
    with st.sidebar:
        st.header("Configuration")
        
        # Station selection
        stations = {
            '44013': 'Boston, MA',
            '46042': 'Monterey Bay, CA',
            '41002': 'South Hatteras, NC',
            '46006': 'Southeast Papa',
            '51001': 'Northwest Hawaii',
        }
        
        selected_station = st.selectbox(
            "Select Buoy Station",
            options=list(stations.keys()),
            format_func=lambda x: stations[x],
            index=0  # Default to Boston
        )
        
        # Year selection
        current_year = datetime.now().year
        selected_year = st.selectbox(
            "Select Year",
            options=range(current_year, current_year - 3, -1),
            index=1
        )
        
        # Load button
        load_button = st.button("Load Data", type="primary", use_container_width=True)
        
        if load_button:
            with st.spinner(f"Fetching data from station {selected_station}..."):
                data, stats, error = load_data(selected_station, selected_year)
                
                if error:
                    st.error(error)
                    st.session_state.clear()
                elif data is not None:
                    st.session_state['data'] = data
                    st.session_state['stats'] = stats
                    st.session_state['station'] = selected_station
                    st.session_state['year'] = selected_year
                    st.success(f"Loaded {len(data)} records")
                    st.rerun()
                else:
                    st.error("Failed to load data")
        
        # Clear data button
        if 'data' in st.session_state:
            if st.button("Clear Data", use_container_width=True):
                st.session_state.clear()
                st.rerun()
        
        st.divider()
        
        # Visualization options (only if data loaded)
        if 'data' in st.session_state:
            st.subheader("Analysis Type")
            analysis_type = st.radio(
                "Select View",
                ["Overview", "Time Series", "Statistics"],
                label_visibility="collapsed"
            )
        else:
            analysis_type = "Overview"
    
    # Main content
    if 'data' not in st.session_state:
        # Welcome screen
        st.info("Select a station and year, then click **Load Data** to begin")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Available Stations", "100+")
        with col2:
            st.metric("Historical Data", "50+ years")
        with col3:
            st.metric("Hourly Measurements", "Millions")
        
        st.markdown("---")
        
        st.subheader("About")
        st.markdown("""
        This dashboard provides interactive analysis of ocean wave data from NOAA/NDBC buoys:
        
        - **Time Series Analysis**: Wave height, period, and wind patterns
        - **Statistical Summaries**: Data quality and distributions
        - **Storm Detection**: Automatic extreme event identification
        
        **Data Source**: NOAA National Data Buoy Center
        """)
        
        with st.expander("Troubleshooting"):
            st.markdown("""
            **If data loading fails:**
            1. Try a different year (2023 recommended)
            2. Try a different station
            3. Check your internet connection
            4. Wait a moment and try again (NOAA servers may be slow)
            
            **Still stuck?**
            - Make sure all dependencies are installed: `pip install -r requirements.txt`
            - Check the console for error messages
            """)
        
        return
    
    # Display loaded data
    data = st.session_state['data']
    stats = st.session_state['stats']
    station = st.session_state['station']
    year = st.session_state['year']
    
    # Metrics
    st.subheader(f"Station {station} - {stations.get(station, 'Unknown')} ({year})")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{stats['total_records']:,}")
    
    with col2:
        if 'Hs' in data.columns:
            avg_hs = data['Hs'].mean()
            st.metric("Avg Wave Height", f"{avg_hs:.2f} m")
        else:
            st.metric("Avg Wave Height", "N/A")
    
    with col3:
        storm_count = stats.get('storm_events', 0)
        st.metric("Storm Events", storm_count)
    
    with col4:
        total = stats['total_records']
        missing = stats.get('missing_Hs', 0)
        quality = 100 - (missing / total * 100) if total > 0 else 0
        st.metric("Data Quality", f"{quality:.1f}%")
    
    st.divider()
    
    # Visualizations
    try:
        visualizer = WaveVisualizer()
        
        if analysis_type == "Overview":
            st.subheader("Data Overview")
            
            # Basic time series
            if all(col in data.columns for col in ['Hs', 'Tp', 'wind_speed']):
                st.subheader("Wave Conditions")
                fig = visualizer.plot_time_series(
                    data, 
                    ['Hs', 'Tp', 'wind_speed'],
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Statistics table
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Summary Statistics")
                if 'Hs' in data.columns:
                    summary_cols = [c for c in ['Hs', 'Tp', 'wind_speed'] if c in data.columns]
                    st.dataframe(data[summary_cols].describe().round(2))
            
            with col2:
                st.subheader("Wave Type Distribution")
                if 'wave_type' in data.columns:
                    wave_counts = data['wave_type'].value_counts()
                    st.bar_chart(wave_counts)
        
        elif analysis_type == "Time Series":
            st.subheader("Time Series Analysis")
            
            if 'Hs' in data.columns:
                fig = visualizer.plot_wave_conditions(data, highlight_storms=True)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Wave height data not available")
        
        elif analysis_type == "Statistics":
            st.subheader("Statistical Analysis")
            
            # Detailed statistics
            numeric_cols = ['Hs', 'Tp', 'Tm', 'wind_speed', 'pressure']
            available_cols = [col for col in numeric_cols if col in data.columns]
            
            if available_cols:
                st.dataframe(data[available_cols].describe().round(3))
            
            # Correlation
            if len(available_cols) > 2:
                st.subheader("Variable Correlations")
                fig = visualizer.plot_correlation_matrix(data, available_cols)
                st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Visualization error: {e}")
        st.exception(e)
    
    # Export
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        csv = data.to_csv()
        st.download_button(
            label="Download Data (CSV)",
            data=csv,
            file_name=f"ndbc_{station}_{year}.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()