"""Walk-forward evaluation of wait time predictors on Cascade Gateway history.

At every hour t in the evaluation period, a predictor sees the live reading
(mean of the three 5-minute readings before t) and predicts the mean delay of
the hour starting at t + h, for each horizon h. Settings are chosen on a
validation year, then the chosen model is refit on everything before the test
year and scored once on it.
"""
from typing import Callable, Dict, Iterable, List, Optional
import argparse
import itertools
from datetime import datetime

import numpy as np
import pandas as pd

from border_predictor.baseline import BaselinePredictor
from border_predictor.cascade import hourly_delay, load_lane
from border_predictor.config.config import CASCADE_CONFIG, EVALUATION_CONFIG


# (target_time, observed_at, observed_wait) -> prediction in minutes
Method = Callable[[datetime, Optional[datetime], Optional[float]], float]


def live_readings(lane: pd.DataFrame) -> pd.Series:
    """
    The live reading available at each hour: mean of the 5-minute readings at t-15, t-10 and t-5.

    Args:
        lane: Output of load_lane.

    Returns:
        Series indexed by hour start.
    """
    rolled = lane["delay"].rolling("15min").mean()
    at_hour = rolled.reindex(rolled.index[rolled.index.minute == 55])
    at_hour.index = at_hour.index + pd.Timedelta(minutes=5)
    return at_hour


def backtest(
    hourly: pd.Series,
    live: pd.Series,
    methods: Dict[str, Method],
    start: str,
    end: str,
    horizons: Iterable[int],
) -> pd.DataFrame:
    """
    Run every method over every origin hour and horizon in [start, end].

    Only origin/horizon pairs where both the live reading and the target are
    present are scored, so every method is judged on the same rows.

    Returns:
        One row per (origin, horizon) with the actual value and each method's prediction.
    """
    origins = hourly[start:end].index
    rows = []
    for h in horizons:
        targets = origins + pd.Timedelta(hours=h)
        actual = hourly.reindex(targets).values
        observed = live.reindex(origins).values
        keep = ~np.isnan(actual) & ~np.isnan(observed)
        for origin, target, y, obs in zip(origins[keep], targets[keep], actual[keep], observed[keep]):
            observed_at = (origin - pd.Timedelta(minutes=5)).to_pydatetime()
            when = (target + pd.Timedelta(minutes=30)).to_pydatetime()
            row = {"origin": origin, "h": h, "actual": y}
            for name, method in methods.items():
                row[name] = method(when, observed_at, obs)
            rows.append(row)
    return pd.DataFrame(rows)


def score(results: pd.DataFrame, methods: List[str], spike: float) -> pd.DataFrame:
    """
    Summarise backtest results per horizon and method.

    Returns:
        DataFrame indexed by (h, method) with MAE, MAE on spike hours, and
        recall/precision for flagging hours over the spike threshold.
    """
    out = []
    for h, group in results.groupby("h"):
        is_spike = group.actual > spike
        for name in methods:
            error = (group[name] - group.actual).abs()
            flagged = group[name] > spike
            out.append({
                "h": h, "method": name,
                "mae": error.mean(),
                "mae_spike": error[is_spike].mean(),
                "recall_spike": (flagged & is_spike).sum() / max(is_spike.sum(), 1),
                "precision_spike": (flagged & is_spike).sum() / max(flagged.sum(), 1),
                "n": len(group),
            })
    return pd.DataFrame(out).set_index(["h", "method"])


def standard_methods(model: BaselinePredictor) -> Dict[str, Method]:
    """The baseline with and without its nowcast, and plain persistence."""
    return {
        "baseline": model.predict,
        "profile_only": lambda when, at, obs: model.expected(when),
        "last_reading": lambda when, at, obs: obs,
    }


def tune(hourly: pd.Series, live: pd.Series, config: Dict) -> Dict:
    """
    Pick profile settings and nowcast decay on the validation period.

    Returns:
        The fit keyword arguments with the lowest mean MAE across horizons.
    """
    train = hourly[config["train_start"]:config["train_end"]]
    best, best_mae = None, float("inf")
    for statistic, profile_kind in itertools.product(["mean", "median"], ["month_weekday_hour", "weekday_hour_x_month"]):
        model = BaselinePredictor.fit(train, statistic=statistic, profile_kind=profile_kind)
        for shrinkage, decay in itertools.product(config["holiday_shrinkage_grid"], config["decay_grid_minutes"]):
            model.set_holiday_shrinkage(shrinkage)
            model.decay_minutes = decay
            res = backtest(hourly, live, {"baseline": model.predict},
                           config["validation_start"], config["validation_end"], config["horizons"])
            mae = (res.baseline - res.actual).abs().groupby(res.h).mean().mean()
            if mae < best_mae:
                best_mae = mae
                best = {"statistic": statistic, "profile_kind": profile_kind,
                        "holiday_shrinkage": shrinkage, "decay_minutes": decay}
    return best


def evaluate_lane(lane_id: int, config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Tune on validation, refit on train + validation, score on the test period.

    Returns:
        Output of score for the test period, with the chosen settings in attrs.
    """
    config = config or EVALUATION_CONFIG
    lane = load_lane(lane_id)
    hourly, live = hourly_delay(lane), live_readings(lane)
    settings = tune(hourly, live, config)
    model = BaselinePredictor.fit(hourly[config["train_start"]:config["validation_end"]], **settings)
    methods = standard_methods(model)
    res = backtest(hourly, live, methods, config["test_start"], config["test_end"], config["horizons"])
    scores = score(res, list(methods), config["spike_minutes"])
    scores.attrs["settings"] = settings
    scores.attrs["multipliers"] = model.multipliers
    return scores


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lanes", type=int, nargs="*", default=list(CASCADE_CONFIG["lanes"]))
    args = parser.parse_args()

    pd.set_option("display.width", 200)
    for lane_id in args.lanes:
        scores = evaluate_lane(lane_id)
        print(f"\n=== {CASCADE_CONFIG['lanes'][lane_id]} (lane {lane_id}) ===")
        print("settings:", scores.attrs["settings"])
        print("multipliers:", {k: round(v, 2) for k, v in scores.attrs["multipliers"].items()})
        print(scores.unstack("method").round(2).to_string())


if __name__ == "__main__":
    main()
