"""Gradient boosted (LightGBM) delay predictor on Cascade Gateway history.

One model per lane covers every horizon. Each row is an origin hour t and a
horizon h; the target is the mean delay of the hour starting at t + h. Features
use only what is known at t: the live reading, recent hourly delay and volume,
calendar and holiday distance for the target hour, and the baseline's own
prediction, so the model learns corrections to the baseline rather than the
whole pattern from scratch.
"""
from typing import Dict, Iterable, List, Optional, Tuple
import argparse
import math
from datetime import date, datetime, timedelta
from pathlib import Path

import holidays
import lightgbm as lgb
import numpy as np
import pandas as pd

from border_predictor.baseline import DAY_TYPES, BaselinePredictor, HolidayCalendar
from border_predictor.cascade import hourly_delay, load_lane
from border_predictor import train_baseline
from border_predictor.config.config import BASELINE_CONFIG, CASCADE_CONFIG, EVALUATION_CONFIG, GBM_CONFIG
from border_predictor.evaluate import live_readings, score, tune

HOLIDAY_DISTANCE_CAP = 7


def holiday_distances(days: Iterable[date]) -> pd.DataFrame:
    """
    Signed days to the nearest Canada/BC and US/WA holiday, capped at HOLIDAY_DISTANCE_CAP.

    Negative means the holiday is still ahead. Lets the model tell the Friday
    before a long weekend from the Monday holiday itself.
    """
    ca, us = holidays.CA(subdiv="BC"), holidays.US(subdiv="WA")
    out = {}
    for day in days:
        row = {}
        for name, cal in (("ca", ca), ("us", us)):
            nearest = HOLIDAY_DISTANCE_CAP
            for d in range(-HOLIDAY_DISTANCE_CAP, HOLIDAY_DISTANCE_CAP + 1):
                if day + timedelta(days=d) in cal and abs(d) < abs(nearest):
                    nearest = -d
            row[f"days_from_holiday_{name}"] = nearest
        out[day] = row
    return pd.DataFrame.from_dict(out, orient="index")


def build_features(
    lane: pd.DataFrame,
    hourly: pd.Series,
    live: pd.Series,
    baseline: BaselinePredictor,
    origins: pd.DatetimeIndex,
    horizons: Iterable[int],
    require_actual: bool = True,
) -> pd.DataFrame:
    """
    Feature rows for every origin hour and horizon.

    Rows without a live reading are dropped, and with require_actual rows
    without a target too, matching evaluate.backtest.

    Returns:
        DataFrame with columns origin, h, actual, last_reading, baseline and the features.
    """
    vol = lane["vol"].resample("H").sum(min_count=1)
    observed_at = origins - pd.Timedelta(minutes=5)
    expected_cache: Dict[pd.Timestamp, float] = {}

    def expected(ts: pd.Timestamp) -> float:
        if ts not in expected_cache:
            expected_cache[ts] = baseline.expected(ts.to_pydatetime())
        return expected_cache[ts]

    base = pd.DataFrame({
        "origin": origins,
        "live": live.reindex(origins).values,
        "lag_1h": hourly.reindex(origins - pd.Timedelta(hours=1)).values,
        "lag_2h": hourly.reindex(origins - pd.Timedelta(hours=2)).values,
        "lag_3h": hourly.reindex(origins - pd.Timedelta(hours=3)).values,
        "vol_1h": vol.reindex(origins - pd.Timedelta(hours=1)).values,
        "vol_2h": vol.reindex(origins - pd.Timedelta(hours=2)).values,
        "origin_hour": origins.hour,
        "expected_at_obs": [expected(t) for t in observed_at],
    })
    base["live_anomaly"] = base.live - base.expected_at_obs

    frames = []
    for h in horizons:
        f = base.copy()
        targets = origins + pd.Timedelta(hours=h)
        f["h"] = h
        f["actual"] = hourly.reindex(targets).values
        f["target"] = targets
        f["month"], f["weekday"], f["hour"] = targets.month, targets.weekday, targets.hour
        # Same hour one day / one week before the target, only when already known at the origin
        f["lag_target_24h"] = hourly.reindex(targets - pd.Timedelta(hours=24)).values if h < 24 else np.nan
        f["lag_target_168h"] = hourly.reindex(targets - pd.Timedelta(hours=168)).values if h < 168 else np.nan
        mid = targets + pd.Timedelta(minutes=30)
        f["expected"] = [expected(t) for t in mid]
        weight = math.exp(-(h * 60 + 35) / baseline.decay_minutes)
        f["baseline"] = np.maximum(0.0, f.expected + weight * f.live_anomaly)
        frames.append(f)

    df = pd.concat(frames, ignore_index=True)
    keep = df.live.notna() & (df.actual.notna() if require_actual else True)
    df = df[keep].reset_index(drop=True)
    calendar = HolidayCalendar()
    days = df.target.dt.date
    df["day_type"] = pd.Categorical([calendar.day_type(d) for d in days], categories=DAY_TYPES).codes
    dist = holiday_distances(sorted(set(days)))
    df = df.join(dist, on=days)
    df["last_reading"] = df.live
    return df


FEATURES = [
    "h", "live", "live_anomaly", "lag_1h", "lag_2h", "lag_3h", "vol_1h", "vol_2h", "origin_hour",
    "month", "weekday", "hour", "day_type", "days_from_holiday_ca", "days_from_holiday_us",
    "lag_target_24h", "lag_target_168h", "expected", "baseline",
]


def fit_gbm(train: pd.DataFrame, objective: str, num_rounds: int, valid: Optional[pd.DataFrame] = None):
    """Train a LightGBM regressor; with valid, stop early on validation MAE."""
    params = dict(GBM_CONFIG["params"], objective=objective, metric="l1", verbose=-1)
    dtrain = lgb.Dataset(train[FEATURES], train.actual, categorical_feature=["day_type"])
    if valid is None:
        return lgb.train(params, dtrain, num_boost_round=num_rounds)
    dvalid = lgb.Dataset(valid[FEATURES], valid.actual, reference=dtrain)
    return lgb.train(params, dtrain, num_boost_round=num_rounds, valid_sets=[dvalid],
                     callbacks=[lgb.early_stopping(GBM_CONFIG["early_stopping_rounds"], verbose=False)])


def band_of(h: int) -> Optional[int]:
    """Index of the horizon band whose model predicts h hours ahead, or None past the last band."""
    for i, band in enumerate(GBM_CONFIG["bands"]):
        if h <= band["up_to"]:
            return i
    return None


def training_horizons() -> List[int]:
    """Every horizon some band trains on."""
    return sorted(set().union(*(b["train_on"] for b in GBM_CONFIG["bands"])))


def fit_bands(
    train: pd.DataFrame, valid: pd.DataFrame, full: pd.DataFrame, objective: str
) -> Tuple[List[lgb.Booster], List[int]]:
    """
    One model per horizon band. Each band's round count is found by early
    stopping train against valid, then the model is refit on full.

    Returns:
        The refit boosters and their round counts, in band order.
    """
    boosters, rounds = [], []
    for band in GBM_CONFIG["bands"]:
        def pick(df):
            return df[df.h.isin(band["train_on"])]
        stopped = fit_gbm(pick(train), objective, GBM_CONFIG["max_rounds"], pick(valid))
        rounds.append(stopped.best_iteration)
        boosters.append(fit_gbm(pick(full), objective, stopped.best_iteration))
    return boosters, rounds


def predict_bands(boosters: List[lgb.Booster], features: pd.DataFrame) -> np.ndarray:
    """Predict each row with its horizon band's model; NaN for rows past the last band."""
    out = np.full(len(features), np.nan)
    bands = features.h.map(band_of)
    for i, booster in enumerate(boosters):
        rows = (bands == i).values
        if rows.any():
            out[rows] = np.maximum(0.0, booster.predict(features.loc[rows, FEATURES]))
    return out


def evaluate_lane(lane_id: int, config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Compare LightGBM with the baseline on the test period.

    Baseline settings are tuned as in evaluate. For each objective, each
    horizon band's model is trained on the training period with early stopping
    on validation, refit on train + validation with the stopped round count,
    and scored once on the test period.

    Returns:
        Output of evaluate.score, with chosen rounds and baseline settings in attrs.
    """
    config = config or EVALUATION_CONFIG
    lane = load_lane(lane_id)
    hourly, live = hourly_delay(lane), live_readings(lane)
    settings = tune(hourly, live, config)

    def span(start, end):
        return hourly[start:end].index

    train_h = training_horizons()
    base_train = BaselinePredictor.fit(hourly[config["train_start"]:config["train_end"]], **settings)
    train = build_features(lane, hourly, live, base_train, span(config["train_start"], config["train_end"]), train_h)
    valid = build_features(lane, hourly, live, base_train,
                           span(config["validation_start"], config["validation_end"]), train_h)

    base_full = BaselinePredictor.fit(hourly[config["train_start"]:config["validation_end"]], **settings)
    full = build_features(lane, hourly, live, base_full, span(config["train_start"], config["validation_end"]), train_h)
    test = build_features(lane, hourly, live, base_full,
                          span(config["test_start"], config["test_end"]), config["horizons"])

    rounds = {}
    for objective in GBM_CONFIG["objectives"]:
        boosters, rounds[objective] = fit_bands(train, valid, full, objective)
        test[f"gbm_{objective}"] = predict_bands(boosters, test)

    methods = ["baseline", "last_reading"] + [f"gbm_{o}" for o in GBM_CONFIG["objectives"]]
    scores = score(test, methods, config["spike_minutes"])
    scores.attrs.update(settings=settings, rounds=rounds, results=test)
    return scores


def model_paths(lane_id: int) -> List[Path]:
    """Where each horizon band's LightGBM model for a lane is stored."""
    return [Path(BASELINE_CONFIG["model_dir"]) / f"gbm_lane{lane_id}_upto{b['up_to']}h.txt"
            for b in GBM_CONFIG["bands"]]


def load_boosters(lane_id: int) -> Optional[List[lgb.Booster]]:
    """A lane's saved band models, or None if any is missing."""
    paths = model_paths(lane_id)
    if not all(p.exists() for p in paths):
        return None
    return [lgb.Booster(model_file=str(p)) for p in paths]


def train_lane(lane_id: int) -> List[lgb.Booster]:
    """
    Train and save one lane's LightGBM models and the baseline they build on.

    Round counts are found by early stopping on the validation period; the
    saved models are then refit with those counts on all data since training start.

    Args:
        lane_id: Cascade crossing lane ID.

    Returns:
        The trained boosters, one per horizon band.
    """
    config, horizons = EVALUATION_CONFIG, training_horizons()
    baseline, settings = train_baseline.train_lane(lane_id)
    lane = load_lane(lane_id)
    hourly, live = hourly_delay(lane), live_readings(lane)

    base_train = BaselinePredictor.fit(hourly[config["train_start"]:config["train_end"]], **settings)
    train = build_features(lane, hourly, live, base_train, hourly[config["train_start"]:config["train_end"]].index, horizons)
    valid = build_features(lane, hourly, live, base_train,
                           hourly[config["validation_start"]:config["validation_end"]].index, horizons)
    full = build_features(lane, hourly, live, baseline, hourly[config["train_start"]:].index, horizons)
    boosters, rounds = fit_bands(train, valid, full, GBM_CONFIG["objective"])

    for booster, path in zip(boosters, model_paths(lane_id)):
        path.parent.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(path))
    print(f"{CASCADE_CONFIG['lanes'][lane_id]} (lane {lane_id}): rounds per band {rounds}")
    return boosters


def forecast(
    recent: pd.DataFrame,
    baseline: BaselinePredictor,
    boosters: List[lgb.Booster],
    targets: Iterable[datetime],
) -> pd.DataFrame:
    """
    Predict the mean delay of each target hour from recent 5-minute readings.

    Models are trained on origins at hour boundaries, so the origin is the next
    boundary at or after the poll following the latest reading. The live reading
    is the mean of the last three polls, as in training, and the hour in
    progress counts as the last complete hour from what has been polled so far.

    Args:
        recent: Output of cascade.fetch_recent (at least 8 days for the week-ago lags).
        baseline: The lane's fitted baseline.
        boosters: The lane's trained LightGBM models, one per horizon band.
        targets: Local times; each is predicted for the hour it falls in.

    Returns:
        DataFrame indexed by target hour start with baseline and gbm predictions.
        Hours before the origin or past the last horizon band are left out.
    """
    origin = (recent.index[-1] + pd.Timedelta(minutes=5)).ceil("H")
    hours = pd.DatetimeIndex(sorted({pd.Timestamp(t).floor("H") for t in targets}))
    horizons = [int(h) for h in (hours - origin) / pd.Timedelta(hours=1) if h >= 0 and band_of(int(h)) is not None]
    out = pd.DataFrame(columns=["baseline", "gbm"])
    if horizons:
        hourly = hourly_delay(recent)
        live = pd.Series([recent["delay"].iloc[-3:].mean()], index=[origin])
        f = build_features(recent, hourly, live, baseline, pd.DatetimeIndex([origin]), horizons, require_actual=False)
        f["gbm"] = predict_bands(boosters, f)
        out = f.set_index("target")[["baseline", "gbm"]]
    out.attrs["origin"] = origin
    return out


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["evaluate", "train"])
    parser.add_argument("--lanes", type=int, nargs="*", default=list(CASCADE_CONFIG["lanes"]))
    args = parser.parse_args()

    pd.set_option("display.width", 220)
    for lane_id in args.lanes:
        if args.command == "train":
            train_lane(lane_id)
            continue
        scores = evaluate_lane(lane_id)
        print(f"\n=== {CASCADE_CONFIG['lanes'][lane_id]} (lane {lane_id}) ===", flush=True)
        print("baseline settings:", scores.attrs["settings"])
        print("rounds:", scores.attrs["rounds"])
        print(scores.drop(columns="n").unstack("method").round(2).to_string(), flush=True)


if __name__ == "__main__":
    main()
