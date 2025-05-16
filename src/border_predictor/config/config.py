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

# Web application configuration
WEB_CONFIG = {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": True,
}
