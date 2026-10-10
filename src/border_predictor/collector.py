"""Collect raw wait times from the CBP API into monthly CSV files.

Designed to run once per invocation from cron, launchd or a CI scheduler.
Each reading is keyed by (port_number, lane, reported_at_utc), where
reported_at_utc comes from CBP's own update time rather than the fetch time,
so polling more often than CBP updates never produces duplicate rows.
"""
from typing import Dict, Iterable, List, Optional, Set, Tuple
import argparse
import csv
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from border_predictor.config.config import COLLECTOR_CONFIG


FIELDS = [
    "reported_at_utc",
    "port_number",
    "port_name",
    "crossing_name",
    "lane",
    "delay_minutes",
    "lanes_open",
    "operational_status",
    "port_status",
    "fetched_at_utc",
]

LANE_GROUPS = {
    "passenger_vehicle_lanes": "passenger",
    "commercial_vehicle_lanes": "commercial",
}

TZ_OFFSETS_HOURS = {
    "EST": -5, "EDT": -4, "CST": -6, "CDT": -5, "MST": -7, "MDT": -6,
    "PST": -8, "PDT": -7, "AKST": -9, "AKDT": -8, "AST": -4, "ADT": -3,
}

UPDATE_TIME_RE = re.compile(
    r"(?:(\d{1,2}):(\d{2})\s*([ap]m)|(noon|midnight))\s+([A-Z]{3,4})", re.IGNORECASE
)


def fetch_ports(url: str, timeout: float) -> List[Dict]:
    """
    Fetch the full CBP wait time feed.

    Args:
        url: CBP API URL.
        timeout: Request timeout in seconds.

    Returns:
        List of port records.
    """
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def parse_reported_time(update_time: str, port_date: str, port_time: str) -> Optional[datetime]:
    """
    Turn CBP's "At 1:00 pm PDT" into a UTC timestamp.

    The update string has no date, so it is anchored to the port's local fetch
    date. If that puts the update after the fetch time (an 11 pm update fetched
    just past midnight), it belongs to the previous day.

    Args:
        update_time: Lane update string, e.g. "At 1:00 pm PDT".
        port_date: Port-local fetch date, "MM/DD/YYYY".
        port_time: Port-local fetch time, "HH:MM:SS".

    Returns:
        Timezone-aware UTC datetime, or None if the string can't be parsed.
    """
    match = UPDATE_TIME_RE.search(update_time or "")
    if not match or not port_date:
        return None

    hour_str, minute_str, am_pm, word, tz_abbr = match.groups()
    offset = TZ_OFFSETS_HOURS.get(tz_abbr.upper())
    if offset is None:
        return None

    if word:
        hour, minute = (12, 0) if word.lower() == "noon" else (0, 0)
    else:
        hour, minute = int(hour_str) % 12, int(minute_str)
        if am_pm.lower() == "pm":
            hour += 12

    local_date = datetime.strptime(port_date, "%m/%d/%Y")
    reported_local = local_date.replace(hour=hour, minute=minute)
    if port_time:
        fetched_local = datetime.combine(
            local_date.date(), datetime.strptime(port_time, "%H:%M:%S").time()
        )
        if reported_local > fetched_local + timedelta(minutes=5):
            reported_local -= timedelta(days=1)

    tz = timezone(timedelta(hours=offset))
    return reported_local.replace(tzinfo=tz).astimezone(timezone.utc)


def extract_readings(
    ports: Iterable[Dict], port_numbers: Iterable[str], fetched_at: datetime
) -> List[Dict]:
    """
    Flatten CBP port records into one row per lane reading.

    Lanes without an update time (closed or not applicable) are skipped,
    since they carry no timestamp to key on.

    Args:
        ports: Port records from the CBP API.
        port_numbers: Port numbers to keep.
        fetched_at: UTC time of the fetch.

    Returns:
        List of reading dicts with keys in FIELDS.
    """
    wanted = set(port_numbers)
    readings = []
    for port in ports:
        if port.get("port_number") not in wanted:
            continue
        for group_key, group_name in LANE_GROUPS.items():
            for lane_key, lane in (port.get(group_key) or {}).items():
                if not isinstance(lane, dict):
                    continue
                reported_at = parse_reported_time(
                    lane.get("update_time", ""), port.get("date", ""), port.get("time", "")
                )
                if reported_at is None:
                    continue
                readings.append({
                    "reported_at_utc": reported_at.isoformat(),
                    "port_number": port.get("port_number"),
                    "port_name": port.get("port_name"),
                    "crossing_name": port.get("crossing_name"),
                    "lane": f"{group_name}_{lane_key.replace('_lanes', '').lower()}",
                    "delay_minutes": lane.get("delay_minutes", ""),
                    "lanes_open": lane.get("lanes_open", ""),
                    "operational_status": lane.get("operational_status", ""),
                    "port_status": port.get("port_status", ""),
                    "fetched_at_utc": fetched_at.isoformat(),
                })
    return readings


def _row_key(row: Dict) -> Tuple[str, str, str]:
    return row["port_number"], row["lane"], row["reported_at_utc"]


def _existing_keys(path: Path) -> Set[Tuple[str, str, str]]:
    if not path.exists():
        return set()
    with path.open(newline="") as f:
        return {_row_key(row) for row in csv.DictReader(f)}


def save_readings(readings: List[Dict], output_dir: Path) -> int:
    """
    Append new readings to monthly CSV files, skipping ones already stored.

    Args:
        readings: Rows from extract_readings.
        output_dir: Directory holding YYYY-MM.csv files.

    Returns:
        Number of rows written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    by_month: Dict[str, List[Dict]] = {}
    for row in readings:
        by_month.setdefault(row["reported_at_utc"][:7], []).append(row)

    written = 0
    for month, rows in by_month.items():
        path = output_dir / f"{month}.csv"
        seen = _existing_keys(path)
        new_rows = []
        for row in rows:
            key = _row_key(row)
            if key not in seen:
                seen.add(key)
                new_rows.append(row)
        if not new_rows:
            continue
        is_new_file = not path.exists()
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if is_new_file:
                writer.writeheader()
            writer.writerows(new_rows)
        written += len(new_rows)
    return written


def collect_once(config: Optional[Dict] = None) -> int:
    """
    Fetch the CBP feed once and store any new readings.

    Args:
        config: Collector configuration. Defaults to COLLECTOR_CONFIG.

    Returns:
        Number of new rows written.
    """
    config = config or COLLECTOR_CONFIG
    fetched_at = datetime.now(timezone.utc).replace(microsecond=0)
    ports = fetch_ports(config["api_url"], config["timeout_seconds"])
    readings = extract_readings(ports, config["port_numbers"], fetched_at)
    return save_readings(readings, Path(config["output_dir"]))


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, help="Override the output directory")
    args = parser.parse_args()

    config = dict(COLLECTOR_CONFIG)
    if args.output_dir:
        config["output_dir"] = args.output_dir
    written = collect_once(config)
    print(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} wrote {written} new readings")


if __name__ == "__main__":
    main()
