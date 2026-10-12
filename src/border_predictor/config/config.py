"""Configuration settings for the border crossing predictor."""
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

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

# Open-Meteo hourly weather at the crossings (no API key). History comes from the reanalysis archive,
# which lags a few days; forecasts (with recent past days) come from the forecast API.
WEATHER_CONFIG = {
    "archive_url": "https://archive-api.open-meteo.com/v1/archive",
    "forecast_url": "https://api.open-meteo.com/v1/forecast",
    "latitude": 49.0,  # between Peace Arch and Pacific Highway
    "longitude": -122.75,
    "timezone": "America/Vancouver",
    "hourly": ["temperature_2m", "precipitation", "snowfall", "snow_depth"],
    "history_start": "2009-01-01",
    "output_path": DATA_DIR / "weather" / "hourly.csv",
    "past_days": 2,
    "forecast_days": 9,  # covers the longest horizon (168 h) from any hour today
    "timeout_seconds": 60,
}

# Pro sports games in Seattle and Vancouver from ESPN's public site API (unofficial, no key): Seahawks and
# World Cup games at Lumen Field, and the Mariners for Toronto's visits. season_types: 2 regular season,
# 3 playoffs. Other teams' games (Canucks, Kraken, Sounders, Whitecaps, other Mariners opponents) showed no
# effect on the test year.
EVENTS_CONFIG = {
    "api_url": "https://site.api.espn.com/apis/site/v2/sports",
    "teams": [
        {"league": "nfl", "sport": "football/nfl", "team": "sea", "season_types": [2, 3]},
        {"league": "mlb", "sport": "baseball/mlb", "team": "sea", "season_types": [2, 3]},
    ],
    # Tournaments read day by day from the scoreboard, keeping games in Seattle or Vancouver
    "tournaments": [{"league": "fifa", "sport": "soccer/fifa.world", "start": "2026-06-11", "end": "2026-07-19"}],
    "first_season": 2008,
    "timezone": "America/Vancouver",
    "window_hours": 12,  # a game affects travel from about half a day before its start to half a day after
    # Games at Lumen Field big enough to show in border traffic (the model's stadium-game feature)
    "stadium_leagues": ["nfl", "fifa"],
    "output_path": DATA_DIR / "events" / "games.csv",
    "timeout_seconds": 30,
    "request_delay_seconds": 0.2,
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
# Debug mode runs Werkzeug's interactive debugger, which executes code: keep it off and local by default
WEB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5000,
    "debug": False,
}

# LightGBM predictor configuration (see gbm.py)
GBM_CONFIG = {
    # Models trained per lane and band. median (l1) is the prediction: it had lower MAE than l2 (mean).
    # low/high bound an 80% interval. spike is the probability the hour's mean delay exceeds
    # EVALUATION_CONFIG["spike_minutes"]; it is trained only for lanes with spike_min_positives such
    # hours in the training period, and alerts above a threshold picked per band for F1 on validation.
    "models": {
        "median": {"objective": "l1"},
        "low": {"objective": "quantile", "alpha": 0.1},
        "high": {"objective": "quantile", "alpha": 0.9},
        "spike": {"objective": "binary"},
    },
    "spike_min_positives": 200,
    # Add every other lane's live reading and last-hour delay as features
    "cross_lane": True,
    # Pre-COVID history added to training rows (flagged pre_covid). A random origin_fraction of its
    # hours keeps memory in check; its rows' weight is picked per lane from weight_grid by the median
    # model's validation MAE (0 = no history).
    "history": {
        "start": "2009-01-01",
        "end": "2020-02-29",
        "origin_fraction": 0.33,
        "weight_grid": [0.0, 0.25, 1.0],
    },
    # One model per band of horizons (hours ahead), each with its own early stopping: a single model
    # stopped on all horizons at once under-fits the short range. A band predicts every horizon up to
    # up_to (from the previous band's up_to + 1) and trains on train_on. The long-range band trains on
    # all horizons: pooling the short-range rows helps it more than its own rows alone.
    "bands": [
        {"up_to": 6, "train_on": list(range(0, 7))},
        {"up_to": 24, "train_on": list(range(7, 25))},
        {"up_to": 168, "train_on": list(range(0, 25)) + list(range(30, 169, 6))},
    ],
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
