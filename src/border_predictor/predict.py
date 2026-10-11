"""Print delay predictions for a Cascade lane, anchored on its latest readings."""
from typing import List
import argparse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import lightgbm as lgb

from border_predictor import gbm
from border_predictor.baseline import BaselinePredictor
from border_predictor.cascade import fetch_current_delay, fetch_recent
from border_predictor.config.config import BASELINE_CONFIG, CASCADE_CONFIG
from border_predictor.train_baseline import model_path


def day_flag(predictor: BaselinePredictor, when: datetime) -> str:
    """Holiday label to print next to a prediction, if any."""
    day_type = predictor.calendar.day_type(when.date())
    return "" if day_type == "normal" else f"  ({day_type.replace('_', ' ')})"


def print_gbm(lane_id: int, predictor: BaselinePredictor, boosters: List[lgb.Booster], start: datetime, hours: int) -> None:
    """Hourly LightGBM predictions from the last days of readings."""
    recent = fetch_recent(lane_id)
    if recent.empty:
        raise SystemExit("No recent readings; try --model baseline --no-live")
    print(f"Latest reading: {recent['delay'].iloc[-1]:.0f} min at {recent.index[-1]:%Y-%m-%d %H:%M}")
    targets = [start + timedelta(hours=step) for step in range(hours + 1)]
    result = gbm.forecast(recent, predictor, boosters, targets)
    if result.empty:
        raise SystemExit("LightGBM predicts up to 168 hours past the latest reading; use --model baseline")
    print("Mean delay for each hour (LightGBM; baseline in brackets)")
    for hour, row in result.iterrows():
        print(f"{hour:%a %Y-%m-%d %H:%M}  {row.gbm:5.1f} min  [{row.baseline:5.1f}]{day_flag(predictor, hour)}")


def print_baseline(lane_id: int, predictor: BaselinePredictor, start: datetime, hours: int, live: bool) -> None:
    """Baseline predictions, optionally anchored on the current reading."""
    reading = fetch_current_delay(lane_id) if live else None
    if reading:
        print(f"Latest reading: {reading[1]:.0f} min at {reading[0]:%Y-%m-%d %H:%M}")
    elif live:
        print("No current reading available; using the historical profile only")
    observed_at, observed_wait = reading or (None, None)

    for step in range(hours + 1):
        when = start + timedelta(hours=step)
        prediction = predictor.predict(when, observed_at, observed_wait)
        print(f"{when:%a %Y-%m-%d %H:%M}  {prediction:5.1f} min{day_flag(predictor, when)}")


def main() -> None:
    """Command line entry point."""
    lanes = ", ".join(f"{i}={n}" for i, n in CASCADE_CONFIG["lanes"].items())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", type=int, default=BASELINE_CONFIG["default_lane_id"], help=lanes)
    parser.add_argument("--at", help='Local target time, "YYYY-MM-DD HH:MM". Defaults to now.')
    parser.add_argument("--hours", type=int, default=12, help="Hours to show from the target time")
    parser.add_argument("--model", choices=["gbm", "baseline"],
                        help="Defaults to gbm when a trained LightGBM model exists")
    parser.add_argument("--no-live", action="store_true", help="Ignore live readings (baseline only)")
    args = parser.parse_args()

    path = model_path(args.lane)
    if not path.exists():
        raise SystemExit(f"No model at {path}; run python -m border_predictor.gbm train --lanes {args.lane}")
    predictor = BaselinePredictor.load(path)
    boosters = gbm.load_boosters(args.lane)
    model = args.model or ("gbm" if boosters and not args.no_live else "baseline")
    if model == "gbm" and args.no_live:
        raise SystemExit("LightGBM needs live readings; use --model baseline with --no-live")
    if model == "gbm" and not boosters:
        raise SystemExit(f"No LightGBM model; run python -m border_predictor.gbm train --lanes {args.lane}")

    now = datetime.now(ZoneInfo(BASELINE_CONFIG["timezone"])).replace(tzinfo=None)
    start = datetime.strptime(args.at, "%Y-%m-%d %H:%M") if args.at else now

    print(CASCADE_CONFIG["lanes"][args.lane])
    if model == "gbm":
        print_gbm(args.lane, predictor, boosters, start, args.hours)
    else:
        print_baseline(args.lane, predictor, start, args.hours, live=not args.no_live)


if __name__ == "__main__":
    main()
