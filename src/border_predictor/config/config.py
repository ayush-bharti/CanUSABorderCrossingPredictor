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
    "yearly_seasonality": True,
}

# Border crossing specific configuration
BORDER_CONFIG = {
    "crossing_name": "Blaine Peace Arch",
    "direction": "both",  # "northbound", "southbound", or "both"
    "countries": ["US", "Canada"],
}

# Web application configuration
WEB_CONFIG = {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": True,
}
