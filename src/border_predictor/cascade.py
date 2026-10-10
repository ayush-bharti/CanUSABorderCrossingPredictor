"""Download historical 5-minute lane data from the Cascade Gateway Border Data API.

Data is saved as one CSV per lane per month under data/cascade/<lane_id>/YYYY-MM.csv.
Months already on disk are skipped, so the download can be stopped and resumed.
Requires CASCADE_API_KEY in the environment or in the project's .env file.
"""
from typing import Dict, Optional
import argparse
import os
import time
from datetime import date
from pathlib import Path

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


def fetch_month(lane_id: int, year: int, month: int, key: str, config: Dict) -> str:
    """
    Fetch one month of 5-minute data for a lane as CSV text.

    Args:
        lane_id: Cascade crossing lane ID.
        year: Year to fetch.
        month: Month to fetch.
        key: API key.
        config: Cascade configuration.

    Returns:
        CSV text from the API.

    Raises:
        RuntimeError: On a non-200 response. The message never includes the key.
    """
    last_day = (date(year + month // 12, month % 12 + 1, 1) - date(year, month, 1)).days
    params = {
        "id": lane_id,
        "start": f"{month:02d}/01/{year}",
        "end": f"{month:02d}/{last_day:02d}/{year}",
        "data": config["metrics"],
        "dg": "Day",
        "tg": "Poll",
        "format": "csv",
        "key": key,
    }
    response = requests.get(config["api_url"], params=params, timeout=config["timeout_seconds"])
    if response.status_code != 200:
        # requests' own errors embed the full URL, which contains the key
        raise RuntimeError(f"Lane {lane_id} {year}-{month:02d}: HTTP {response.status_code}")
    return response.text


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
