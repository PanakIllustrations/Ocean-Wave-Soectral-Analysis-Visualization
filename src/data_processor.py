"""
NOAA/NDBC Wave Data Processor

This module handles fetching, cleaning, and processing ocean wave data
from NOAA's National Data Buoy Center (NDBC).

Author: Josiah Panak
Date: August 2025
"""

import pandas as pd
import numpy as np
import requests
from io import StringIO
from datetime import datetime
from typing import Tuple, Optional, List
import warnings

warnings.filterwarnings('ignore')


class NDBCDataProcessor:
    """
    Process NOAA/NDBC historical wave and meteorological data.
    
    This class provides methods for:
    - Fetching data from NDBC stations
    - Cleaning and quality control
    - Deriving wave parameters
    - Creating analysis-ready datasets
    """
    
    # Station metadata
    POPULAR_STATIONS = {
        '46042': 'Monterey Bay, CA',
        '44013': 'Boston, MA',
        '41002': 'South Hatteras, NC',
        '46006': 'Southeast Papa (Pacific)',
        '51001': 'Northwest Hawaii',
        '42001': 'Gulf of Mexico',
        '44025': 'Long Island, NY',
    }
    
    def __init__(self, missing_values: Optional[List] = None):
        """
        Initialize the data processor.
        
        Args:
            missing_values: List of values to treat as missing/NaN
        """
        self.missing_values = missing_values or [99, 999, 9999, 99.0, 999.0, 9999.0]
        
    def get_recent_data(self, station_id: str = '46042', timeout: int = 10) -> Optional[pd.DataFrame]:
        """
        Get the most recent 45 days of data from any station.
        
        Args:
            station_id: NDBC station identifier
            timeout: Request timeout in seconds
            
        Returns:
            DataFrame with recent data, or None if fetch fails
        """
        url = f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.txt"
        
        try:
            print(f"Fetching recent data from station {station_id} ({self.POPULAR_STATIONS.get(station_id, 'Unknown')})...")
            response = requests.get(url, timeout=timeout)
            
            if response.status_code == 200:
                # Parse the realtime data format
                df = pd.read_csv(
                    StringIO(response.text), 
                    sep=r'\s+', 
                    comment='#', 
                    na_values=self.missing_values
                )
                
                # Skip the units row (first row after header)
                df = df.iloc[1:]
                
                print(f"Successfully fetched {len(df)} rows of recent data")
                return df
            else:
                print(f"Station {station_id} not available (HTTP {response.status_code})")
                return None
                
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    def get_historical_year(
        self, 
        station_id: str = '46042', 
        year: int = 2024,
        timeout: int = 30
    ) -> Optional[pd.DataFrame]:
        """
        Get a full year of historical data.
        
        Args:
            station_id: NDBC station identifier
            year: Year to fetch
            timeout: Request timeout in seconds
            
        Returns:
            DataFrame with historical data, or None if fetch fails
        """
        # Use the view_text_file endpoint
        url = (
            f"https://www.ndbc.noaa.gov/view_text_file.php?"
            f"filename={station_id}h{year}.txt.gz&dir=data/historical/stdmet/"
        )
        
        try:
            print(f"Fetching {year} data from station {station_id}...")
            response = requests.get(url, timeout=timeout)
            
            if response.status_code == 200 and len(response.text) > 100:
                # Parse the data
                df = pd.read_csv(
                    StringIO(response.text),
                    sep=r'\s+',
                    na_values=self.missing_values
                )
                
                # Skip the units row if present
                if df.iloc[0, 0] == '#yr' or str(df.iloc[0, 0]).startswith('#'):
                    df = df.iloc[1:]
                
                print(f"Successfully fetched {len(df)} rows for year {year}")
                return df
            else:
                print(f"No data available for {station_id} in {year}")
                return None
                
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def clean_and_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and process NDBC data into analysis-ready format.
        
        Processing steps:
        1. Create datetime index
        2. Standardize column names
        3. Convert units to canonical form
        4. Derive wave parameters
        5. Add temporal features
        6. Flag storm events
        7. Add quality control flags
        
        Args:
            df: Raw NDBC dataframe
            
        Returns:
            Cleaned and processed DataFrame
        """
        print("Cleaning and processing data...")
        
        # Create datetime index
        df = self._create_datetime_index(df)
        
        # Standardize column names (map NDBC codes to readable names)
        df = self._standardize_columns(df)
        
        # Derive wave parameters
        df = self._derive_wave_parameters(df)
        
        # Add temporal features
        df = self._add_temporal_features(df)
        
        # Detect storms
        df = self._detect_storms(df)
        
        # Add quality control flags
        df = self._add_qc_flags(df)
        
        print(f"Processed {len(df)} records")
        return df
    
    def _create_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create datetime index from NDBC timestamp columns."""
        if 'YY' in df.columns or '#YY' in df.columns:
            # Historical format
            if '#YY' in df.columns:
                df.rename(columns={'#YY': 'YY'}, inplace=True)
            
            # Handle 2-digit years
            if df['YY'].dtype in ['int64', 'float64']:
                if df['YY'].max() < 100:
                    df['YY'] = df['YY'].apply(
                        lambda x: int(x) + 1900 if x > 50 else int(x) + 2000
                    )
            
            # Add minutes if not present
            if 'mm' not in df.columns:
                df['mm'] = 0
            
            # Create datetime
            df['datetime'] = pd.to_datetime(
                df[['YY', 'MM', 'DD', 'hh', 'mm']].rename(columns={
                    'YY': 'year', 'MM': 'month', 'DD': 'day', 
                    'hh': 'hour', 'mm': 'minute'
                })
            )
        
        # Set datetime as index
        if 'datetime' in df.columns:
            df = df.set_index('datetime')
            df = df.sort_index()
        
        return df
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to readable format."""
        # Map NDBC abbreviations to readable names
        column_map = {
            'WVHT': 'Hs',  # Significant wave height
            'DPD': 'Tp',   # Dominant wave period
            'APD': 'Tm',   # Average wave period
            'MWD': 'Dp',   # Mean wave direction
            'WSPD': 'wind_speed',
            'WDIR': 'wind_dir',
            'GST': 'gust_speed',
            'PRES': 'pressure',
            'ATMP': 'air_temp',
            'WTMP': 'water_temp',
        }
        
        df.rename(columns=column_map, inplace=True)
        
        # Convert numeric columns to float
        numeric_cols = ['Hs', 'Tp', 'Tm', 'Dp', 'wind_speed', 'wind_dir', 
                       'gust_speed', 'pressure', 'air_temp', 'water_temp']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def _derive_wave_parameters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive additional wave parameters."""
        if 'Hs' in df.columns and 'Tp' in df.columns:
            # Wave steepness (deep water approximation)
            g = 9.81  # gravity (m/s²)
            df['wavelength'] = g * df['Tp']**2 / (2 * np.pi)
            df['steepness'] = df['Hs'] / df['wavelength']
            
            # Classify waves
            df['wave_type'] = 'NORMAL'
            df.loc[df['steepness'] < 0.025, 'wave_type'] = 'SWELL'
            df.loc[df['steepness'] > 0.05, 'wave_type'] = 'STEEP'
        
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features for analysis."""
        if isinstance(df.index, pd.DatetimeIndex):
            df['month'] = df.index.month
            df['day_of_year'] = df.index.dayofyear
            df['hour'] = df.index.hour
            
            # Season (meteorological)
            df['season'] = df.index.month % 12 // 3 + 1
            season_map = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
            df['season_name'] = df['season'].map(season_map)
        
        return df
    
    def _detect_storms(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect storm events based on wave height and wind speed."""
        if all(col in df.columns for col in ['Hs', 'wind_speed']):
            df['storm_flag'] = 0
            storm_conditions = (df['Hs'] > 4) & (df['wind_speed'] > 15)
            df.loc[storm_conditions, 'storm_flag'] = 1
        
        return df
    
    def _add_qc_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add quality control flags for suspicious values."""
        df['qc_flag'] = 0
        
        # Flag physically implausible values
        if 'Hs' in df.columns:
            df.loc[df['Hs'] > 20, 'qc_flag'] = 1  # Unrealistic wave height
            df.loc[df['Hs'] < 0, 'qc_flag'] = 1    # Negative wave height
        
        if 'wind_speed' in df.columns:
            df.loc[df['wind_speed'] < 0, 'qc_flag'] = 1
            df.loc[df['wind_speed'] > 50, 'qc_flag'] = 1
        
        return df
    
    def try_multiple_stations(
        self, 
        year: int = 2024,
        stations: Optional[List[str]] = None
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Try multiple stations until finding one with data.
        
        Args:
            year: Year to fetch
            stations: List of station IDs to try (uses defaults if None)
            
        Returns:
            Tuple of (DataFrame, station_id) or (None, None) if all fail
        """
        stations = stations or list(self.POPULAR_STATIONS.keys())
        
        # Try historical data first
        for station in stations:
            df = self.get_historical_year(station, year)
            if df is not None and len(df) > 100:
                print(f"Successfully got data from station {station}")
                return df, station
        
        # Fall back to recent data
        print("\nTrying recent data instead...")
        for station in stations:
            df = self.get_recent_data(station)
            if df is not None and len(df) > 10:
                print(f"Successfully got recent data from station {station}")
                return df, station
        
        print("✗ Could not fetch data from any station")
        return None, None
    
    def create_output_formats(
        self, 
        df: pd.DataFrame, 
        station_id: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create both wide and long format outputs.
        
        Args:
            df: Processed dataframe
            station_id: Station identifier
            
        Returns:
            Tuple of (wide_df, long_df)
        """
        # Wide format (original structure)
        wide_df = df.copy()
        wide_df['station_id'] = station_id
        
        # Long format (tidy data)
        value_cols = [
            col for col in df.columns 
            if col not in ['station_id', 'qc_flag', 'storm_flag']
        ]
        
        long_df = df[value_cols].reset_index().melt(
            id_vars=['datetime'] if 'datetime' in df.reset_index().columns else ['index'],
            var_name='variable',
            value_name='value'
        )
        long_df['station_id'] = station_id
        
        # Add units
        unit_map = {
            'Hs': 'm', 'Tp': 's', 'Tm': 's', 'Dp': 'degrees',
            'wind_speed': 'm/s', 'wind_dir': 'degrees',
            'pressure': 'hPa', 'air_temp': '°C', 'water_temp': '°C',
            'steepness': 'ratio', 'wavelength': 'm'
        }
        long_df['unit'] = long_df['variable'].map(unit_map).fillna('')
        
        return wide_df, long_df
    
    def get_summary_stats(self, df: pd.DataFrame) -> dict:
        """
        Generate summary statistics for the dataset.
        
        Args:
            df: Processed dataframe
            
        Returns:
            Dictionary with summary statistics
        """
        stats = {
            'total_records': len(df),
            'date_range': (df.index.min(), df.index.max()),
            'missing_Hs': df['Hs'].isna().sum() if 'Hs' in df.columns else 0,
            'storm_events': (df['storm_flag'] == 1).sum() if 'storm_flag' in df.columns else 0,
        }
        
        if 'wave_type' in df.columns:
            stats['wave_types'] = df['wave_type'].value_counts().to_dict()
        
        if 'Hs' in df.columns:
            stats['Hs_stats'] = df['Hs'].describe().to_dict()
        
        return stats


def main():
    """Main execution function for standalone use."""
    processor = NDBCDataProcessor()
    
    print("="*60)
    print("NOAA/NDBC Data Processing Pipeline")
    print("="*60)
    
    # Fetch data
    raw_data, station = processor.try_multiple_stations(year=2024)
    
    if raw_data is not None:
        # Process the data
        processed_data = processor.clean_and_process(raw_data)
        
        # Create output formats
        wide_df, long_df = processor.create_output_formats(processed_data, station)
        
        # Display results
        print("\n" + "="*60)
        print(f"Successfully processed data from Station {station}")
        print("="*60)
        print(f"Data shape: {processed_data.shape}")
        print(f"Date range: {processed_data.index.min()} to {processed_data.index.max()}")
        
        print("\nAvailable columns:")
        print(list(processed_data.columns))
        
        print("\nFirst 5 rows of key variables:")
        key_cols = [
            col for col in ['Hs', 'Tp', 'wind_speed', 'pressure', 'steepness'] 
            if col in processed_data.columns
        ]
        print(processed_data[key_cols].head())
        
        print("\nSummary statistics:")
        print(processed_data[key_cols].describe())
        
        # Save to files
        wide_df.to_csv(f'ndbc_{station}_wide.csv')
        long_df.to_csv(f'ndbc_{station}_long.csv')
        print(f"\nData saved to: ndbc_{station}_wide.csv and ndbc_{station}_long.csv")
        
        # Summary
        summary = processor.get_summary_stats(processed_data)
        print("\n" + "="*60)
        print("Data Quality Summary")
        print("="*60)
        for key, value in summary.items():
            print(f"{key}: {value}")
            
    else:
        print("\nCould not fetch data from any station.")
        print("Please check your internet connection or try again later.")


if __name__ == "__main__":
    main()