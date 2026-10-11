"""Gradient boosted (LightGBM) delay predictor on Cascade Gateway history.

Each row is an origin hour t and a horizon h; the target is the mean delay of
the hour starting at t + h. Features use only what is known at t: the live
reading, recent hourly delay and volume, the other lanes' current delay,
calendar and holiday distance for the target hour, and the baseline's own
prediction, so the models learn corrections to the baseline rather than the
whole pattern from scratch.

Per lane and horizon band there is a median model (the prediction), 10% and
90% quantile models (an 80% interval), and for busy lanes a classifier for the
probability that the hour's delay exceeds the spike threshold.
"""
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple
import argparse
import json
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
# Hourly delay and live reading of another lane, used as cross-lane features
LaneInputs = Tuple[pd.Series, pd.Series]


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
    others: Optional[Dict[int, LaneInputs]] = None,
) -> pd.DataFrame:
    """
    Feature rows for every origin hour and horizon.

    Rows without a live reading are dropped, and with require_actual rows
    without a target too, matching evaluate.backtest.

    Args:
        others: Hourly delay and live readings of other lanes by lane ID; adds
            other<id>_live and other<id>_lag_1h columns.

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
    for lane_id, (other_hourly, other_live) in sorted((others or {}).items()):
        base[f"other{lane_id}_live"] = other_live.reindex(origins).values
        base[f"other{lane_id}_lag_1h"] = other_hourly.reindex(origins - pd.Timedelta(hours=1)).values

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


def feature_columns(df: pd.DataFrame, cross_lane: bool) -> List[str]:
    """FEATURES plus, with cross_lane, the other-lane columns present in df."""
    return FEATURES + ([c for c in df.columns if c.startswith("other")] if cross_lane else [])


def fit_gbm(
    train: pd.DataFrame,
    kind: str,
    num_rounds: int,
    valid: Optional[pd.DataFrame] = None,
    features: Optional[List[str]] = None,
) -> lgb.Booster:
    """
    Train one model of a kind in GBM_CONFIG["models"]; with valid, stop early on it.

    The spike kind is a classifier for actual > EVALUATION_CONFIG["spike_minutes"].
    """
    spec = GBM_CONFIG["models"][kind]
    features = features or FEATURES
    params = dict(GBM_CONFIG["params"], objective=spec["objective"], verbose=-1)
    if spec["objective"] == "quantile":
        params.update(alpha=spec["alpha"], metric="quantile")
    elif spec["objective"] == "binary":
        params["metric"] = "binary_logloss"
    else:
        params["metric"] = "l1"

    def label(df):
        if spec["objective"] == "binary":
            return (df.actual > EVALUATION_CONFIG["spike_minutes"]).astype(int)
        return df.actual

    dtrain = lgb.Dataset(train[features], label(train), categorical_feature=["day_type"])
    if valid is None:
        return lgb.train(params, dtrain, num_boost_round=num_rounds)
    dvalid = lgb.Dataset(valid[features], label(valid), reference=dtrain)
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


def predict_bands(boosters: List[lgb.Booster], features: pd.DataFrame) -> np.ndarray:
    """Predict each row with its horizon band's model; NaN for rows past the last band."""
    out = np.full(len(features), np.nan)
    bands = features.h.map(band_of)
    for i, booster in enumerate(boosters):
        rows = (bands == i).values
        if rows.any():
            out[rows] = np.maximum(0.0, booster.predict(features.loc[rows, booster.feature_name()]))
    return out


def best_f1_threshold(prob: np.ndarray, is_spike: np.ndarray) -> float:
    """Probability cut-off with the highest F1 for flagging spikes; 0.5 if there are none."""
    if not is_spike.any():
        return 0.5
    best, best_f1 = 0.5, -1.0
    for cut in np.arange(0.05, 0.96, 0.01):
        flagged = prob >= cut
        tp = (flagged & is_spike).sum()
        f1 = 2 * tp / max(flagged.sum() + is_spike.sum(), 1)
        if f1 > best_f1:
            best, best_f1 = float(cut), f1
    return round(best, 2)


def fit_bands(
    train: pd.DataFrame, valid: pd.DataFrame, full: pd.DataFrame, kind: str, features: List[str]
) -> Tuple[List[lgb.Booster], List[int], np.ndarray]:
    """
    One model per horizon band. Each band's round count is found by early
    stopping train against valid, then the model is refit on full.

    Returns:
        The refit boosters and their round counts, in band order, and the
        early-stopped models' predictions on valid (for picking thresholds).
    """
    boosters, rounds, stopped_models = [], [], []
    for band in GBM_CONFIG["bands"]:
        def pick(df):
            return df[df.h.isin(band["train_on"])]
        stopped = fit_gbm(pick(train), kind, GBM_CONFIG["max_rounds"], pick(valid), features)
        stopped_models.append(stopped)
        rounds.append(stopped.best_iteration)
        boosters.append(fit_gbm(pick(full), kind, stopped.best_iteration, features=features))
    return boosters, rounds, predict_bands(stopped_models, valid)


def band_thresholds(valid: pd.DataFrame, prob: np.ndarray) -> List[float]:
    """Spike alert threshold per band, for the best F1 on the validation rows the band predicts."""
    is_spike = (valid.actual > EVALUATION_CONFIG["spike_minutes"]).values
    bands = valid.h.map(band_of).values
    return [best_f1_threshold(prob[bands == i], is_spike[bands == i]) for i in range(len(GBM_CONFIG["bands"]))]


def has_enough_spikes(hourly: pd.Series, config: Dict) -> bool:
    """Whether a lane has enough spike hours in the training period for a classifier."""
    train = hourly[config["train_start"]:config["train_end"]]
    return int((train > config["spike_minutes"]).sum()) >= GBM_CONFIG["spike_min_positives"]


@lru_cache(maxsize=None)
def lane_history(lane_id: int) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """A lane's 5-minute readings, hourly delay and live readings, loaded once per process."""
    lane = load_lane(lane_id)
    return lane, hourly_delay(lane), live_readings(lane)


def other_lane_inputs(lane_id: int) -> Dict[int, LaneInputs]:
    """Hourly delay and live readings of every other configured lane, if cross-lane features are on."""
    if not GBM_CONFIG["cross_lane"]:
        return {}
    return {other: lane_history(other)[1:] for other in CASCADE_CONFIG["lanes"] if other != lane_id}


def alert_scores(results: pd.DataFrame, rules: Dict[str, pd.Series], spike: float) -> pd.DataFrame:
    """Recall and precision for flagging hours over the spike threshold, per horizon and rule."""
    out = []
    is_spike = results.actual > spike
    for h, rows in results.groupby("h").groups.items():
        for name, flagged in rules.items():
            f, s = flagged.loc[rows], is_spike.loc[rows]
            out.append({"h": h, "rule": name, "recall": (f & s).sum() / max(s.sum(), 1),
                        "precision": (f & s).sum() / max(f.sum(), 1), "flags": int(f.sum()), "spikes": int(s.sum())})
    return pd.DataFrame(out).set_index(["h", "rule"])


def interval_scores(results: pd.DataFrame) -> pd.DataFrame:
    """Share of hours inside [low, high] and the interval's mean width, per horizon."""
    inside = (results.actual >= results.low) & (results.actual <= results.high)
    return pd.DataFrame({
        "coverage": inside.groupby(results.h).mean(),
        "width": (results.high - results.low).groupby(results.h).mean(),
    })


def evaluate_lane(lane_id: int, config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Compare LightGBM with the baseline on the test period.

    Baseline settings are tuned as in evaluate. Each model's horizon bands are
    trained on the training period with early stopping on validation, refit on
    train + validation with the stopped round count, and scored once on the
    test period. The median model is trained with and without the other lanes'
    features (gbm and gbm_own) to measure what they add.

    Returns:
        Output of evaluate.score, with chosen rounds, baseline settings, spike
        thresholds, alert and interval scores, and per-row results in attrs.
    """
    config = config or EVALUATION_CONFIG
    lane, hourly, live = lane_history(lane_id)
    others = other_lane_inputs(lane_id)
    settings = tune(hourly, live, config)

    def span(start, end):
        return hourly[start:end].index

    train_h = training_horizons()
    base_train = BaselinePredictor.fit(hourly[config["train_start"]:config["train_end"]], **settings)
    base_full = BaselinePredictor.fit(hourly[config["train_start"]:config["validation_end"]], **settings)
    train = build_features(lane, hourly, live, base_train, span(config["train_start"], config["train_end"]),
                           train_h, others=others)
    valid = build_features(lane, hourly, live, base_train,
                           span(config["validation_start"], config["validation_end"]), train_h, others=others)
    full = build_features(lane, hourly, live, base_full, span(config["train_start"], config["validation_end"]),
                          train_h, others=others)
    test = build_features(lane, hourly, live, base_full,
                          span(config["test_start"], config["test_end"]), config["horizons"], others=others)

    own, cross = feature_columns(train, False), feature_columns(train, True)
    runs = {"gbm_own": ("median", own), "gbm": ("median", cross), "low": ("low", cross), "high": ("high", cross)}
    if has_enough_spikes(hourly, config):
        runs["spike_prob"] = ("spike", cross)
    rounds, thresholds = {}, None
    for column, (kind, features) in runs.items():
        boosters, rounds[column], valid_pred = fit_bands(train, valid, full, kind, features)
        test[column] = predict_bands(boosters, test)
        if kind == "spike":
            thresholds = band_thresholds(valid, valid_pred)
    test["low"] = np.minimum(test.low, test.gbm)
    test["high"] = np.maximum(test.high, test.gbm)

    spike = config["spike_minutes"]
    rules = {"baseline": test.baseline > spike, "gbm_own": test.gbm_own > spike, "gbm": test.gbm > spike,
             "high": test.high > spike}
    if thresholds is not None:
        test["spike_alert"] = test.spike_prob >= test.h.map(band_of).map(dict(enumerate(thresholds)))
        rules["classifier"] = test.spike_alert

    scores = score(test, ["baseline", "last_reading", "gbm_own", "gbm"], spike)
    scores.attrs.update(settings=settings, rounds=rounds, thresholds=thresholds, results=test,
                        alerts=alert_scores(test, rules, spike), intervals=interval_scores(test))
    return scores


def model_paths(lane_id: int, kind: str = "median") -> List[Path]:
    """Where each horizon band's LightGBM model of a kind for a lane is stored."""
    return [Path(BASELINE_CONFIG["model_dir"]) / f"gbm_lane{lane_id}_{kind}_upto{b['up_to']}h.txt"
            for b in GBM_CONFIG["bands"]]


def meta_path(lane_id: int) -> Path:
    """Where a lane's spike thresholds are stored."""
    return Path(BASELINE_CONFIG["model_dir"]) / f"gbm_lane{lane_id}_meta.json"


def load_models(lane_id: int) -> Optional[Dict]:
    """
    A lane's saved models: {kind: boosters} for every kind with all bands saved,
    plus "spike_thresholds" when the spike classifier is present. None without
    the median models.
    """
    models = {}
    for kind in GBM_CONFIG["models"]:
        paths = model_paths(lane_id, kind)
        if all(p.exists() for p in paths):
            models[kind] = [lgb.Booster(model_file=str(p)) for p in paths]
    if "median" not in models:
        return None
    if "spike" in models:
        models["spike_thresholds"] = json.loads(meta_path(lane_id).read_text())["spike_thresholds"]
    return models


def other_lanes_needed(models: Dict) -> List[int]:
    """Lane IDs whose readings a lane's saved models take as features."""
    names = set().union(*(b.feature_name() for kind in GBM_CONFIG["models"] for b in models.get(kind, [])))
    return sorted({int(n[len("other"):].split("_")[0]) for n in names if n.startswith("other")})


def train_lane(lane_id: int) -> Dict:
    """
    Train and save one lane's LightGBM models and the baseline they build on.

    Round counts are found by early stopping on the validation period; the
    saved models are then refit with those counts on all data since training start.

    Args:
        lane_id: Cascade crossing lane ID.

    Returns:
        The trained models, as load_models returns them.
    """
    config, horizons = EVALUATION_CONFIG, training_horizons()
    baseline, settings = train_baseline.train_lane(lane_id)
    lane, hourly, live = lane_history(lane_id)
    others = other_lane_inputs(lane_id)

    base_train = BaselinePredictor.fit(hourly[config["train_start"]:config["train_end"]], **settings)
    train = build_features(lane, hourly, live, base_train, hourly[config["train_start"]:config["train_end"]].index,
                           horizons, others=others)
    valid = build_features(lane, hourly, live, base_train,
                           hourly[config["validation_start"]:config["validation_end"]].index, horizons, others=others)
    full = build_features(lane, hourly, live, baseline, hourly[config["train_start"]:].index, horizons, others=others)
    features = feature_columns(train, cross_lane=True)

    kinds = [k for k in GBM_CONFIG["models"] if k != "spike" or has_enough_spikes(hourly, config)]
    models, rounds = {}, {}
    for kind in kinds:
        models[kind], rounds[kind], valid_pred = fit_bands(train, valid, full, kind, features)
        for booster, path in zip(models[kind], model_paths(lane_id, kind)):
            path.parent.mkdir(parents=True, exist_ok=True)
            booster.save_model(str(path))
        if kind == "spike":
            models["spike_thresholds"] = band_thresholds(valid, valid_pred)
            meta_path(lane_id).write_text(json.dumps({"spike_thresholds": models["spike_thresholds"]}))
    if "spike" not in kinds:
        # A classifier from an earlier training would no longer match: remove it
        for path in model_paths(lane_id, "spike") + [meta_path(lane_id)]:
            path.unlink(missing_ok=True)
    print(f"{CASCADE_CONFIG['lanes'][lane_id]} (lane {lane_id}): rounds per band {rounds}, "
          f"spike thresholds {models.get('spike_thresholds')}")
    return models


def recent_inputs(recent: pd.DataFrame, until: pd.Timestamp, origin: pd.Timestamp) -> LaneInputs:
    """Hourly delay and the live reading at origin from a lane's recent readings up to until."""
    recent = recent[recent.index <= until]
    live = recent["delay"].iloc[-3:].mean() if len(recent) else np.nan
    return hourly_delay(recent), pd.Series([live], index=[origin])


def forecast(
    recent: pd.DataFrame,
    baseline: BaselinePredictor,
    models: Dict,
    targets: Iterable[datetime],
    others_recent: Optional[Dict[int, Optional[pd.DataFrame]]] = None,
) -> pd.DataFrame:
    """
    Predict the mean delay of each target hour from recent 5-minute readings.

    Models are trained on origins at hour boundaries, so the origin is the next
    boundary at or after the poll following the latest reading. The live reading
    is the mean of the last three polls, as in training, and the hour in
    progress counts as the last complete hour from what has been polled so far.
    Other lanes are read up to the same poll; a lane missing from others_recent
    leaves its features empty, which the models treat as missing.

    Args:
        recent: Output of cascade.fetch_recent (at least 8 days for the week-ago lags).
        baseline: The lane's fitted baseline.
        models: The lane's trained models, as load_models returns them.
        targets: Local times; each is predicted for the hour it falls in.
        others_recent: Recent readings of the lanes in other_lanes_needed(models).

    Returns:
        DataFrame indexed by target hour start with baseline, gbm, low and high
        predictions, plus spike_prob and spike_alert when the lane has a
        classifier. Hours before the origin or past the last horizon band are left out.
    """
    last = recent.index[-1]
    origin = (last + pd.Timedelta(minutes=5)).ceil("H")
    hours = pd.DatetimeIndex(sorted({pd.Timestamp(t).floor("H") for t in targets}))
    horizons = [int(h) for h in (hours - origin) / pd.Timedelta(hours=1) if h >= 0 and band_of(int(h)) is not None]
    columns = ["baseline", "gbm", "low", "high"] + (["spike_prob", "spike_alert"] if "spike" in models else [])
    out = pd.DataFrame(columns=columns)
    if horizons:
        hourly, live = recent_inputs(recent, last, origin)
        empty = (pd.Series(dtype=float), pd.Series(dtype=float))
        others = {lane_id: recent_inputs(df, last, origin) if df is not None and len(df) else empty
                  for lane_id, df in (others_recent or {}).items()}
        for lane_id in other_lanes_needed(models):
            others.setdefault(lane_id, empty)
        f = build_features(recent, hourly, live, baseline, pd.DatetimeIndex([origin]), horizons,
                           require_actual=False, others=others)
        f["gbm"] = predict_bands(models["median"], f)
        f["low"] = np.minimum(predict_bands(models["low"], f), f.gbm) if "low" in models else np.nan
        f["high"] = np.maximum(predict_bands(models["high"], f), f.gbm) if "high" in models else np.nan
        if "spike" in models:
            f["spike_prob"] = predict_bands(models["spike"], f)
            f["spike_alert"] = f.spike_prob >= f.h.map(band_of).map(dict(enumerate(models["spike_thresholds"])))
        out = f.set_index("target")[columns]
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
        print("rounds:", scores.attrs["rounds"], "spike thresholds:", scores.attrs["thresholds"])
        print(scores.drop(columns="n").unstack("method").round(2).to_string())
        print(scores.attrs["alerts"].drop(columns="spikes").unstack("rule").round(2).to_string())
        print(scores.attrs["intervals"].round(2).to_string(), flush=True)


if __name__ == "__main__":
    main()
