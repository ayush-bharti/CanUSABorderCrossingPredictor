"""Print baseline delay predictions for a Cascade lane, anchored on its current reading."""
import argparse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from border_predictor.baseline import BaselinePredictor
from border_predictor.cascade import fetch_current_delay
from border_predictor.config.config import BASELINE_CONFIG, CASCADE_CONFIG
from border_predictor.train_baseline import model_path


def main() -> None:
    """Command line entry point."""
    lanes = ", ".join(f"{i}={n}" for i, n in CASCADE_CONFIG["lanes"].items())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", type=int, default=BASELINE_CONFIG["default_lane_id"], help=lanes)
    parser.add_argument("--at", help='Local target time, "YYYY-MM-DD HH:MM". Defaults to now.')
    parser.add_argument("--hours", type=int, default=12, help="Hours to show from the target time")
    parser.add_argument("--no-live", action="store_true", help="Ignore the current reading")
    args = parser.parse_args()

    path = model_path(args.lane)
    if not path.exists():
        raise SystemExit(f"No model at {path}; run python -m border_predictor.train_baseline --lanes {args.lane}")
    predictor = BaselinePredictor.load(path)

    now = datetime.now(ZoneInfo(BASELINE_CONFIG["timezone"])).replace(tzinfo=None)
    start = datetime.strptime(args.at, "%Y-%m-%d %H:%M") if args.at else now

    reading = None if args.no_live else fetch_current_delay(args.lane)
    print(CASCADE_CONFIG["lanes"][args.lane])
    if reading:
        print(f"Latest reading: {reading[1]:.0f} min at {reading[0]:%Y-%m-%d %H:%M}")
    elif not args.no_live:
        print("No current reading available; using the historical profile only")
    observed_at, observed_wait = reading or (None, None)

    for step in range(args.hours + 1):
        when = start + timedelta(hours=step)
        prediction = predictor.predict(when, observed_at, observed_wait)
        day_type = predictor.calendar.day_type(when.date())
        flag = "" if day_type == "normal" else f"  ({day_type.replace('_', ' ')})"
        print(f"{when:%a %Y-%m-%d %H:%M}  {prediction:5.1f} min{flag}")


if __name__ == "__main__":
    main()
