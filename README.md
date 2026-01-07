# Ocean Wave Spectral Analysis & Visualization

An interactive data visualization system for exploring ocean wave patterns using NOAA/NDBC buoy data. This project transforms complex wave spectra and temporal patterns into intuitive, actionable insights for marine engineers, coastal planners, and researchers.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Project Overview

Raw ocean wave data contains rich temporal and spectral information, but its complexity often limits accessibility. This project addresses that challenge through:

- **Interactive Visualizations**: Explore wave spectra, temporal patterns, and spatial differences
- **Real-time Data Processing**: Automated pipeline for NOAA/NDBC buoy data
- **Quality-Controlled Analysis**: Robust data cleaning and validation procedures
- **Multi-dimensional Insights**: Combine time-series, frequency-domain, and spatial views

## Key Features

### 1. **Interactive Spectrum Explorer**
- Dynamic FFT wave spectra visualization (frequency vs. energy density)
- Time slider for temporal animation
- Multi-buoy comparison panels
- Hover interactions revealing dominant frequency peaks

### 2. **Temporal Condition Analysis**
- Heatmap visualization of wave height, period, and direction
- Linked brushing between time and frequency views
- Storm event detection and highlighting
- Seasonal trend analysis

### 3. **Spatial Wave Field Mapping**
- Geographic visualization of buoy locations
- Directional rose plots showing wave/wind patterns
- Multi-site comparison for optimal location analysis

## Dataset

**Source**: NOAA National Data Buoy Center (NDBC)
- **Coverage**: 100+ buoy stations since 1970
- **Variables**: Wave height (Hs), periods (Tp, Tm), direction, wind, pressure, temperature
- **Resolution**: Hourly measurements
- **Volume**: Millions of historical records
