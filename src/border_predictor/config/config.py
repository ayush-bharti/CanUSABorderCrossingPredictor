"""Configuration settings for the border crossing prediction model."""
from typing import Dict, List, Optional, Union
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Data configuration
DATA_CONFIG = {
    "data_dir": DATA_DIR,
    "historical_data_path": DATA_DIR / "historical_wait_times.csv",
    "current_data_path": DATA_DIR / "current_wait_times.csv",
    "processed_data_path": DATA_DIR / "processed_data.csv",
    "monthly_data_path": DATA_DIR / "monthly_converted_data.csv",
    "live_data_path": DATA_DIR / "live_wait_times.csv",
}

LIVE_DATA_CONFIG = {
    "api_url": "https://bwt.cbp.gov/api/waittimes",
    "refresh_interval_minutes": 15,
    "max_retries": 3,
    "retry_delay_seconds": 5,
}

# Model configuration
MODEL_CONFIG = {
    "model_path": MODELS_DIR / "prophet_model.pkl",
    "forecast_horizon_days": 7,
    "forecast_frequency": "H",  # Hourly forecasts
    "changepoint_prior_scale": 0.05,
    "seasonality_prior_scale": 10.0,
    "holidays_prior_scale": 10.0,
    "daily_seasonality": True,
    "weekly_seasonality": True,
    "yearly_seasonality": False,  # Removed yearly seasonality as requested
}

# Border crossing specific configuration
BORDER_CONFIG = {
    "port_name": "Blaine",
    "crossing_name": "Peace Arch",
    "direction": "both",  # "northbound", "southbound", or "both"
    "countries": ["US", "Canada"],
    "lane_type": "standard",  # Default lane type for fetching live data
}

# Collector configuration: raw CBP readings, deduplicated by (port, lane, reported time)
COLLECTOR_CONFIG = {
    "api_url": "https://bwt.cbp.gov/api/waittimes",
    "output_dir": DATA_DIR / "collected",
    # Cascade Gateway ports: Peace Arch, Pacific Highway, Lynden, Sumas
    "port_numbers": ["300402", "300401", "302301", "300901"],
    "timeout_seconds": 20,
}

# Cascade Gateway Border Data API: historical 5-minute lane data
CASCADE_CONFIG = {
    "api_url": "https://cascadegatewaydata.com/CrossingLane/",
    "output_dir": DATA_DIR / "cascade",
    "lanes": {
        4: "Peace Arch South Cars",
        11: "Peace Arch South NEXUS",
        5: "Peace Arch North Cars",
        6: "Peace Arch North NEXUS",
        3: "Pacific Highway South Cars",
        9: "Pacific Highway South NEXUS",
        13: "Pacific Highway North Cars",
        14: "Pacific Highway North NEXUS",
    },
    "metrics": "avg-delay,avg-vol,avg-avail",
    "timeout_seconds": 120,
    "max_retries": 3,
    "retry_delay_seconds": 10,
    "request_delay_seconds": 0.5,
}

# Baseline predictor configuration
BASELINE_CONFIG = {
    "port_number": "300402",  # Peace Arch
    "lane": "passenger_standard",
    "timezone": "America/Vancouver",
    # Multipliers applied to the historical profile. Off (1.0) until fitted: guessed values raised
    # error in the May 2025 backtest.
    "holiday_multiplier": 1.0,
    "holiday_adjacent_multiplier": 1.0,
    # How fast the gap between the latest reading and the profile fades (minutes)
    "nowcast_decay_minutes": 180.0,
}

# Web application configuration
WEB_CONFIG = {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": True,
}
