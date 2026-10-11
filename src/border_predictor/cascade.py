"""Download historical 5-minute lane data from the Cascade Gateway Border Data API.

Data is saved as one CSV per lane per month under data/cascade/<lane_id>/YYYY-MM.csv.
Months already on disk are skipped, so the download can be stopped and resumed.
Requires CASCADE_API_KEY in the environment or in the project's .env file.
"""
from typing import Dict, Optional, Tuple, Union
import argparse
import io
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from border_predictor.config.config import CASCADE_CONFIG, PROJECT_ROOT


def load_api_key() -> str:
    """
    Read CASCADE_API_KEY from the environment, falling back to the .env file.

    Returns:
        The API key.

    Raises:
        RuntimeError: If no key is configured.
    """
    key = os.environ.get("CASCADE_API_KEY")
    env_path = PROJECT_ROOT / ".env"
    if not key and env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("CASCADE_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("'\"")
    if not key:
        raise RuntimeError("CASCADE_API_KEY is not set in the environment or .env")
    return key


def fetch_range(lane_id: int, start: date, end: date, key: str, config: Dict) -> str:
    """
    Fetch 5-minute data for a lane over a date range (inclusive) as CSV text.

    Args:
        lane_id: Cascade crossing lane ID.
        start: First local date.
        end: Last local date. Today returns readings up to the latest poll.
        key: API key.
        config: Cascade configuration.

    Returns:
        CSV text from the API.

    Raises:
        RuntimeError: On a non-200 response. The message never includes the key.
    """
    params = {
        "id": lane_id,
        "start": start.strftime("%m/%d/%Y"),
        "end": end.strftime("%m/%d/%Y"),
        "data": config["metrics"],
        "dg": "Day",
        "tg": "Poll",
        "format": "csv",
        "key": key,
    }
    response = requests.get(config["api_url"], params=params, timeout=config["timeout_seconds"])
    if response.status_code != 200:
        # requests' own errors embed the full URL, which contains the key
        raise RuntimeError(f"Lane {lane_id} {start}..{end}: HTTP {response.status_code}")
    return response.text


def fetch_month(lane_id: int, year: int, month: int, key: str, config: Dict) -> str:
    """Fetch one calendar month for a lane as CSV text (see fetch_range)."""
    last_day = date(year + month // 12, month % 12 + 1, 1) - timedelta(days=1)
    return fetch_range(lane_id, date(year, month, 1), last_day, key, config)


def parse_csv(text: Union[str, Path]) -> pd.DataFrame:
    """
    Parse API CSV (text or a saved file) into ts-indexed delay, vol and avail columns.
    """
    df = pd.read_csv(io.StringIO(text) if isinstance(text, str) else text)
    df.columns = ["ts", "delay", "vol", "avail"]
    df["ts"] = pd.to_datetime(df["ts"])
    return df.set_index("ts")


def fetch_recent(lane_id: int, days: int = 9, config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Fetch the last few days of 5-minute data for a lane, through the latest poll.

    Args:
        lane_id: Cascade crossing lane ID.
        days: Calendar days to fetch, including today.
        config: Cascade configuration. Defaults to CASCADE_CONFIG.

    Returns:
        DataFrame like load_lane, with polls not yet taken dropped.
    """
    config = config or CASCADE_CONFIG
    key = load_api_key()
    today = datetime.now(ZoneInfo(config["timezone"])).date()
    df = parse_csv(fetch_range(lane_id, today - timedelta(days=days - 1), today, key, config))
    last = df["avail"].gt(0) & df["delay"].notna()
    return df[: df.index[last][-1]] if last.any() else df.iloc[:0]


def load_lane(lane_id: int, config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Load every downloaded month for a lane.

    Args:
        lane_id: Cascade crossing lane ID.
        config: Cascade configuration. Defaults to CASCADE_CONFIG.

    Returns:
        DataFrame indexed by local timestamp with delay, vol and avail columns.

    Raises:
        FileNotFoundError: If nothing has been downloaded for the lane.
    """
    config = config or CASCADE_CONFIG
    paths = sorted((Path(config["output_dir"]) / str(lane_id)).glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No Cascade data for lane {lane_id}; run python -m border_predictor.cascade")
    return pd.concat(parse_csv(p) for p in paths).sort_index()


def hourly_delay(lane: pd.DataFrame) -> pd.Series:
    """
    Average delay per clock hour, left-labelled (10:00 covers 10:00-10:55).

    Args:
        lane: Output of load_lane.

    Returns:
        Series of mean delay in minutes; NaN where the lane reported nothing.
    """
    return lane["delay"].resample("H").mean()


def fetch_current_delay(lane_id: int, config: Optional[Dict] = None) -> Optional[Tuple[datetime, float]]:
    """
    Fetch the lane's current delay reading.

    Args:
        lane_id: Cascade crossing lane ID.
        config: Cascade configuration. Defaults to CASCADE_CONFIG.

    Returns:
        (naive local reading time, delay in minutes), or None if the lane has no current reading.

    Raises:
        RuntimeError: On a non-200 response. The message never includes the key.
    """
    config = config or CASCADE_CONFIG
    key = load_api_key()
    url = config["api_url"].rstrip("/") + f"/CurrentDelay/{lane_id}"
    response = requests.get(url, params={"format": "json", "key": key}, timeout=config["timeout_seconds"])
    if response.status_code != 200:
        raise RuntimeError(f"Current delay for lane {lane_id}: HTTP {response.status_code}")
    body = response.json()
    starts, values = body.get("GroupStarts") or [], body.get("Values") or []
    if not starts or not values or not values[0] or values[0][0] is None:
        return None
    # The epoch encodes the crossing's local wall-clock time, not UTC
    reading_time = datetime.fromtimestamp(starts[0], timezone.utc).replace(tzinfo=None)
    return reading_time, float(values[0][0])


def download(start: date, end: date, config: Optional[Dict] = None) -> None:
    """
    Download every configured lane for each month from start to end, inclusive.

    Args:
        start: First month to fetch (day is ignored).
        end: Last month to fetch (day is ignored).
        config: Cascade configuration. Defaults to CASCADE_CONFIG.
    """
    config = config or CASCADE_CONFIG
    key = load_api_key()
    months = [
        (y, m)
        for y in range(start.year, end.year + 1)
        for m in range(1, 13)
        if (start.year, start.month) <= (y, m) <= (end.year, end.month)
    ]
    for lane_id, lane_name in config["lanes"].items():
        lane_dir = Path(config["output_dir"]) / str(lane_id)
        lane_dir.mkdir(parents=True, exist_ok=True)
        fetched = 0
        for year, month in months:
            path = lane_dir / f"{year}-{month:02d}.csv"
            if path.exists():
                continue
            for attempt in range(config["max_retries"]):
                try:
                    text = fetch_month(lane_id, year, month, key, config)
                    break
                except (RuntimeError, requests.RequestException) as e:
                    message = str(e).replace(key, "***")
                    print(f"  retry {attempt + 1}: {message}")
                    time.sleep(config["retry_delay_seconds"] * (attempt + 1))
            else:
                print(f"  giving up on lane {lane_id} {year}-{month:02d}")
                continue
            # Write to a temp file first so an interrupted run never leaves a partial month
            tmp = path.with_suffix(".tmp")
            tmp.write_text(text)
            tmp.rename(path)
            fetched += 1
            time.sleep(config["request_delay_seconds"])
        print(f"{lane_name} (lane {lane_id}): fetched {fetched} new months")


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", default="2007-01", help="First month, YYYY-MM")
    parser.add_argument("--end", default=None, help="Last month, YYYY-MM (default: last full month)")
    args = parser.parse_args()

    start = date.fromisoformat(f"{args.start}-01")
    if args.end:
        end = date.fromisoformat(f"{args.end}-01")
    else:
        today = date.today()
        end = date(today.year - (today.month == 1), (today.month - 2) % 12 + 1, 1)
    download(start, end)


if __name__ == "__main__":
    main()
