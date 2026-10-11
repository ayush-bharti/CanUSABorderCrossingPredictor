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
    "timezone": "America/Vancouver",  # timestamps are the crossings' local wall-clock time
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

# Walk-forward evaluation on Cascade history. Training starts after the COVID border closure.
EVALUATION_CONFIG = {
    "train_start": "2023-01-01",
    "train_end": "2024-09-30",
    "validation_start": "2024-10-01",
    "validation_end": "2025-09-30",
    "test_start": "2025-10-01",
    "test_end": "2026-09-30",
    "horizons": [0, 1, 2, 3, 6, 12, 24, 72, 168],
    "decay_grid_minutes": [30, 60, 120, 180, 300, 480],
    "holiday_shrinkage_grid": [0.0, 0.25, 0.5, 1.0],
    "spike_minutes": 30,
}

# Baseline predictor configuration
# Profile, holiday multipliers and nowcast decay are fitted per lane by train_baseline.
BASELINE_CONFIG = {
    "default_lane_id": 4,  # Peace Arch South Cars
    "timezone": "America/Vancouver",
    "model_dir": MODELS_DIR,
}

# Web application configuration
WEB_CONFIG = {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": True,
}

# LightGBM predictor configuration (see gbm.py)
GBM_CONFIG = {
    # l1 targets the median (lowest MAE); l2 targets the mean, which reaches higher on spikes
    "objectives": ["l1", "l2"],
    # One model per band of horizons (hours ahead), each with its own early stopping: a single model
    # stopped on all horizons at once under-fits the short range. A band predicts every horizon up to
    # up_to (from the previous band's up_to + 1) and trains on train_on. The long-range band trains on
    # all horizons: pooling the short-range rows helps it more than its own rows alone.
    "bands": [
        {"up_to": 6, "train_on": list(range(0, 7))},
        {"up_to": 24, "train_on": list(range(7, 25))},
        {"up_to": 168, "train_on": list(range(0, 25)) + list(range(30, 169, 6))},
    ],
    # Objective used by train/predict
    "objective": "l1",
    "max_rounds": 3000,
    "early_stopping_rounds": 100,
    "params": {
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": 0,
    },
}
