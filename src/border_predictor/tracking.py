"""Live accuracy tracking: log forecasts as they are made, record what happened, and score them.

`snapshot` (run hourly, see deploy/) fetches each lane's recent readings, logs
its forecast for each tracked horizon to data/tracking/forecasts/YYYY-MM.csv,
and records the mean delay of each hour that is now complete in
data/tracking/actuals.csv. `report` scores the logged forecasts against those
actuals; the web app's /accuracy page shows the same scores.
"""
from typing import Callable, Dict, List, Optional
import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from border_predictor import gbm, weather
from border_predictor.baseline import BaselinePredictor
from border_predictor.cascade import fetch_recent, hourly_delay
from border_predictor.config.config import CASCADE_CONFIG, EVALUATION_CONFIG, TRACKING_CONFIG
from border_predictor.train_baseline import model_path

logger = logging.getLogger(__name__)

FORECAST_COLUMNS = ["made_at", "lane", "origin", "horizon", "target", "baseline", "gbm", "low", "high",
                    "spike_prob", "spike_alert", "model_version"]


def forecasts_path(month: pd.Timestamp, config: Dict = TRACKING_CONFIG) -> Path:
    return Path(config["output_dir"]) / "forecasts" / f"{month:%Y-%m}.csv"


def actuals_path(config: Dict = TRACKING_CONFIG) -> Path:
    return Path(config["output_dir"]) / "actuals.csv"


def lane_forecast(lane_id: int, recent: pd.DataFrame, others_recent: Dict[int, Optional[pd.DataFrame]],
                  wx: Optional[pd.DataFrame], made_at: datetime, config: Dict = TRACKING_CONFIG) -> pd.DataFrame:
    """One lane's forecast rows for the tracked horizons (FORECAST_COLUMNS); empty without models."""
    models = gbm.load_models(lane_id)
    if models is None:
        return pd.DataFrame(columns=FORECAST_COLUMNS)
    baseline = BaselinePredictor.load(model_path(lane_id))
    origin = gbm.forecast_origin(recent)
    targets = [origin + pd.Timedelta(hours=h) for h in config["horizons"]]
    predicted = gbm.forecast(recent, baseline, models, targets, others_recent, wx).reset_index()
    rows = predicted.rename(columns={"index": "target"})
    rows["horizon"] = ((rows.target - origin) / pd.Timedelta(hours=1)).astype(int)
    for column in ("spike_prob", "spike_alert"):
        if column not in rows:
            rows[column] = np.nan
    version = max(p.stat().st_mtime for p in (model_path(lane_id), gbm.meta_path(lane_id)) if p.exists())
    rows = rows.assign(made_at=made_at, lane=lane_id, origin=origin,
                       model_version=datetime.fromtimestamp(version).strftime("%Y-%m-%d %H:%M:%S"))
    return rows[FORECAST_COLUMNS]


def complete_hours(lane_id: int, recent: pd.DataFrame) -> pd.DataFrame:
    """Mean delay of each hour of recent readings that has its last poll (hh:55) in."""
    hourly = hourly_delay(recent).dropna()
    hourly = hourly[hourly.index + pd.Timedelta(minutes=55) <= recent.index[-1]]
    return pd.DataFrame({"lane": lane_id, "hour": hourly.index, "actual": hourly.values})


def append_forecasts(rows: pd.DataFrame, config: Dict = TRACKING_CONFIG) -> None:
    """Add forecast rows to the month file of when they were made."""
    for month, group in rows.groupby(pd.to_datetime(rows.made_at).dt.to_period("M")):
        path = forecasts_path(month.to_timestamp(), config)
        path.parent.mkdir(parents=True, exist_ok=True)
        group.to_csv(path, mode="a", header=not path.exists(), index=False)


def update_actuals(rows: pd.DataFrame, config: Dict = TRACKING_CONFIG) -> None:
    """Merge newly complete hours into the actuals file; a re-read hour keeps its latest value."""
    path = actuals_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        rows = pd.concat([pd.read_csv(path, parse_dates=["hour"]), rows])
    rows = rows.drop_duplicates(["lane", "hour"], keep="last").sort_values(["lane", "hour"])
    tmp = path.with_suffix(".tmp")
    rows.to_csv(tmp, index=False)
    tmp.replace(path)


def snapshot(now: Optional[datetime] = None, fetch: Callable = fetch_recent,
             fetch_weather: Callable = weather.fetch_forecast, config: Dict = TRACKING_CONFIG) -> int:
    """
    Log every lane's forecast now and record the hours completed since the last snapshot.

    A lane whose readings cannot be fetched, or whose latest reading is older
    than max_staleness_minutes, is skipped.

    Returns:
        The number of forecast rows logged.
    """
    now = now or pd.Timestamp.now(tz=CASCADE_CONFIG["timezone"]).tz_localize(None).to_pydatetime()
    recent = {}
    for lane_id in CASCADE_CONFIG["lanes"]:
        try:
            recent[lane_id] = fetch(lane_id)
        except Exception as e:  # one lane's failure should not stop the others being logged
            logger.warning("Cascade fetch failed for lane %s: %s", lane_id, e)
    try:
        wx = fetch_weather()
    except Exception as e:  # the models treat missing weather as missing values
        logger.warning("Weather fetch failed: %s", e)
        wx = None
    forecasts, actuals = [], []
    for lane_id, data in recent.items():
        if data is None or data.empty:
            continue
        actuals.append(complete_hours(lane_id, data))
        if now - data.index[-1] > pd.Timedelta(minutes=config["max_staleness_minutes"]):
            logger.warning("Lane %s: latest reading %s is stale, not logging a forecast", lane_id, data.index[-1])
            continue
        others = {other: recent.get(other) for other in CASCADE_CONFIG["lanes"] if other != lane_id}
        forecasts.append(lane_forecast(lane_id, data, others, wx, now, config))
    if actuals:
        update_actuals(pd.concat(actuals), config)
    rows = pd.concat(forecasts) if forecasts else pd.DataFrame(columns=FORECAST_COLUMNS)
    if len(rows):
        append_forecasts(rows, config)
    return len(rows)


def load_forecasts(since: pd.Timestamp, config: Dict = TRACKING_CONFIG) -> pd.DataFrame:
    """Logged forecasts for target hours from since on; a forecast logged twice keeps its first copy."""
    paths = sorted((Path(config["output_dir"]) / "forecasts").glob("*.csv"))
    # A month file holds forecasts made that month, whose targets run up to a week later
    paths = [p for p in paths if pd.Timestamp(p.stem) >= (since - pd.Timedelta(days=8)).to_period("M").to_timestamp()]
    if not paths:
        return pd.DataFrame(columns=FORECAST_COLUMNS)
    df = pd.concat(pd.read_csv(p, parse_dates=["made_at", "origin", "target"]) for p in paths)
    df = df.drop_duplicates(["lane", "origin", "horizon"])
    return df[df.target >= since]


def load_actuals(config: Dict = TRACKING_CONFIG) -> pd.DataFrame:
    path = actuals_path(config)
    if not path.exists():
        return pd.DataFrame(columns=["lane", "hour", "actual"])
    return pd.read_csv(path, parse_dates=["hour"])


def scored(forecasts: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    """Forecasts whose target hour has an actual, with an actual column."""
    joined = forecasts.merge(actuals.rename(columns={"hour": "target"}), on=["lane", "target"])
    return joined.dropna(subset=["actual", "gbm"])


def score(rows: pd.DataFrame, spike_minutes: float = EVALUATION_CONFIG["spike_minutes"]) -> pd.DataFrame:
    """
    Accuracy of scored forecasts by lane and horizon.

    Columns: n; mae (the model) and baseline_mae, in minutes; coverage, the
    share of actuals within the 80% range; spikes (actual over spike_minutes),
    alerts raised and hits (alerts on a spike), for lanes with a classifier.
    """
    if rows.empty:
        return pd.DataFrame(columns=["n", "mae", "baseline_mae", "coverage", "spikes", "alerts", "hits"])
    rows = rows.assign(
        error=(rows.gbm.clip(lower=0) - rows.actual).abs(),
        baseline_error=(rows.baseline - rows.actual).abs(),
        covered=(rows.actual >= rows.low) & (rows.actual <= rows.high),
        spike=rows.actual > spike_minutes,
        alert=rows.spike_alert.astype("boolean").fillna(False).astype(bool),
    )
    rows = rows.assign(hit=rows.alert & rows.spike)
    out = rows.groupby(["lane", "horizon"]).agg(
        n=("error", "size"), mae=("error", "mean"), baseline_mae=("baseline_error", "mean"),
        coverage=("covered", "mean"), spikes=("spike", "sum"), alerts=("alert", "sum"), hits=("hit", "sum"))
    has_classifier = rows.groupby(["lane", "horizon"]).spike_prob.apply(lambda s: s.notna().any())
    out.loc[~has_classifier, ["alerts", "hits"]] = np.nan
    return out


def daily_mae(rows: pd.DataFrame, max_horizon: int = 24) -> pd.DataFrame:
    """Daily model and baseline MAE per lane over horizons up to max_horizon, by target date."""
    rows = rows[rows.horizon <= max_horizon]
    if rows.empty:
        return pd.DataFrame(columns=["lane", "day", "mae", "baseline_mae", "n"])
    rows = rows.assign(day=rows.target.dt.normalize(), error=(rows.gbm.clip(lower=0) - rows.actual).abs(),
                       baseline_error=(rows.baseline - rows.actual).abs())
    return rows.groupby(["lane", "day"]).agg(mae=("error", "mean"), baseline_mae=("baseline_error", "mean"),
                                             n=("error", "size")).reset_index()


def recent_scores(days: int, now: Optional[pd.Timestamp] = None, config: Dict = TRACKING_CONFIG) -> pd.DataFrame:
    """Scored forecasts for target hours in the last days."""
    now = now or pd.Timestamp.now(tz=CASCADE_CONFIG["timezone"]).tz_localize(None)
    return scored(load_forecasts(now - pd.Timedelta(days=days), config), load_actuals(config))


def report(days: List[int], config: Dict = TRACKING_CONFIG) -> str:
    """MAE by lane and horizon, model vs baseline, for each window of days."""
    pd.set_option("display.width", 220)
    parts = []
    for window in days:
        scores = score(recent_scores(window, config=config))
        if scores.empty:
            parts.append(f"Last {window} days: no scored forecasts yet")
            continue
        table = scores[["mae", "baseline_mae"]].unstack("horizon").round(2)
        table.index = [f"{lane} {CASCADE_CONFIG['lanes'][lane]}" for lane in table.index]
        totals = scores.groupby("lane")[["n", "coverage", "spikes", "alerts", "hits"]].agg(
            {"n": "sum", "coverage": "mean", "spikes": "sum", "alerts": "sum", "hits": "sum"})
        parts.append(f"Last {window} days: MAE in minutes by horizon (hours ahead), model then baseline\n"
                     f"{table.to_string()}\n\nForecasts scored, 80% range coverage, spikes and alerts\n"
                     f"{totals.round(2).to_string()}")
    return "\n\n".join(parts)


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["snapshot", "report"])
    parser.add_argument("--days", type=int, nargs="*", default=TRACKING_CONFIG["report_days"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "snapshot":
        logger.info("Logged %d forecasts", snapshot())
    else:
        print(report(args.days))


if __name__ == "__main__":
    main()
