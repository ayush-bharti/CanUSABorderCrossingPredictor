"""Hourly weather at the crossings from Open-Meteo: history for training, forecasts for live predictions.

History is saved to data/weather/hourly.csv; run this module to download or
refresh it. Times are the crossings' local wall-clock time, like the Cascade data.
"""
from functools import lru_cache
from typing import Dict, Optional
import argparse
import io
from datetime import date, timedelta

import pandas as pd
import requests

from border_predictor.config.config import WEATHER_CONFIG

# Model feature name for each Open-Meteo variable
COLUMNS = {"temperature_2m": "wx_temp", "precipitation": "wx_precip", "snowfall": "wx_snowfall",
           "snow_depth": "wx_snow_depth"}


def parse(csv_text: str) -> pd.DataFrame:
    """
    Open-Meteo CSV output as a DataFrame indexed by local hour, columns named as in COLUMNS.

    The repeated hour when clocks fall back keeps its first reading.
    """
    lines = csv_text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("time,"))
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])))
    df.columns = ["time"] + [COLUMNS[c.split(" ")[0]] for c in df.columns[1:]]
    df["time"] = pd.to_datetime(df.time)
    return df.drop_duplicates("time").set_index("time").sort_index()


def fetch(url: str, params: Dict, config: Dict = WEATHER_CONFIG) -> pd.DataFrame:
    """One Open-Meteo request for the configured location and variables."""
    params = dict(params, latitude=config["latitude"], longitude=config["longitude"], timezone=config["timezone"],
                  hourly=",".join(config["hourly"]), format="csv")
    response = requests.get(url, params=params, timeout=config["timeout_seconds"])
    response.raise_for_status()
    return parse(response.text)


def download_history(config: Dict = WEATHER_CONFIG) -> pd.DataFrame:
    """Download the archive from history_start to the latest day it has and save it."""
    end = date.today() - timedelta(days=1)
    df = fetch(config["archive_url"], {"start_date": config["history_start"], "end_date": end.isoformat()}, config)
    # The archive lags a few days and leaves the latest hours empty
    df = df.dropna(how="all")
    path = config["output_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df


@lru_cache(maxsize=None)
def load_history(path: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Saved weather history, or None when it has not been downloaded."""
    path = path or WEATHER_CONFIG["output_path"]
    try:
        return pd.read_csv(path, index_col="time", parse_dates=True)
    except FileNotFoundError:
        return None


def fetch_forecast(config: Dict = WEATHER_CONFIG) -> pd.DataFrame:
    """Forecast for the coming days, plus the recent past days, by local hour."""
    return fetch(config["forecast_url"], {"past_days": config["past_days"],
                                          "forecast_days": config["forecast_days"]}, config).dropna(how="all")


def target_features(weather: Optional[pd.DataFrame], targets: pd.Series) -> pd.DataFrame:
    """
    Weather features for each target hour: the hour's own values and the target
    day's precipitation and snowfall totals. NaN where weather is unknown.
    """
    hours = pd.DatetimeIndex(targets)
    out = pd.DataFrame(index=range(len(hours)))
    if weather is None or weather.empty:
        for column in list(COLUMNS.values()) + ["wx_precip_day", "wx_snowfall_day"]:
            out[column] = float("nan")
        return out
    for column in COLUMNS.values():
        out[column] = weather[column].reindex(hours).values
    daily = weather[["wx_precip", "wx_snowfall"]].groupby(weather.index.normalize()).sum(min_count=1)
    days = hours.normalize()
    out["wx_precip_day"] = daily.wx_precip.reindex(days).values
    out["wx_snowfall_day"] = daily.wx_snowfall.reindex(days).values
    return out


def main() -> None:
    """Command line entry point: download or refresh the history."""
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    df = download_history()
    print(f"Saved {len(df)} hours, {df.index[0]} to {df.index[-1]}, to {WEATHER_CONFIG['output_path']}")


if __name__ == "__main__":
    main()
