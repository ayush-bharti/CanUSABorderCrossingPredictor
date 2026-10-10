"""Print baseline wait time predictions, anchored on the current CBP reading."""
from typing import Optional, Tuple
import argparse
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from border_predictor.baseline import BaselinePredictor
from border_predictor.collector import extract_readings, fetch_ports
from border_predictor.config.config import BASELINE_CONFIG, COLLECTOR_CONFIG


def latest_reading(port_number: str, lane: str, tz: ZoneInfo) -> Optional[Tuple[datetime, float]]:
    """
    Fetch the current CBP reading for one lane.

    Args:
        port_number: CBP port number.
        lane: Lane name as produced by the collector, e.g. "passenger_standard".
        tz: Local timezone of the crossing.

    Returns:
        (naive local reported time, wait in minutes), or None if unavailable.
    """
    ports = fetch_ports(COLLECTOR_CONFIG["api_url"], COLLECTOR_CONFIG["timeout_seconds"])
    readings = extract_readings(ports, [port_number], datetime.now(timezone.utc))
    for row in readings:
        if row["lane"] == lane and row["delay_minutes"] != "":
            reported = datetime.fromisoformat(row["reported_at_utc"]).astimezone(tz)
            return reported.replace(tzinfo=None), float(row["delay_minutes"])
    return None


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--at", help='Local target time, "YYYY-MM-DD HH:MM". Defaults to now.')
    parser.add_argument("--hours", type=int, default=12, help="Hours to show from the target time")
    parser.add_argument("--no-live", action="store_true", help="Ignore the current CBP reading")
    args = parser.parse_args()

    tz = ZoneInfo(BASELINE_CONFIG["timezone"])
    now = datetime.now(tz).replace(tzinfo=None)
    start = datetime.strptime(args.at, "%Y-%m-%d %H:%M") if args.at else now

    predictor = BaselinePredictor()
    reading = None if args.no_live else latest_reading(
        BASELINE_CONFIG["port_number"], BASELINE_CONFIG["lane"], tz
    )
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
