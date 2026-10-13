"""Pro sports games in Seattle and Vancouver that draw cross-border trips.

Schedules come from ESPN's public site API (unofficial, no key) and are saved
to data/events/games.csv; run this module to download or refresh them. Future
games are included, so a refresh covers the forecast horizon. Start times are
the crossings' local wall-clock time, like the Cascade data.
"""
from functools import lru_cache
from typing import Dict, Iterable, List, Optional
import argparse
import logging
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from border_predictor.config.config import EVENTS_CONFIG

logger = logging.getLogger(__name__)

CITIES = {"Seattle": "sea", "Vancouver": "van"}


def parse_events(data: Dict, league: str, timezone: str) -> List[Dict]:
    """Games in an ESPN schedule or scoreboard response played in Seattle or Vancouver."""
    out = []
    for event in data.get("events", []):
        competition = event["competitions"][0]
        # Cities come as "Seattle" or "Seattle, Washington"
        city = competition.get("venue", {}).get("address", {}).get("city", "").split(",")[0]
        if city not in CITIES:
            continue
        teams = {c["homeAway"]: c["team"].get("abbreviation", "") for c in competition["competitors"]}
        start = pd.Timestamp(event["date"]).tz_convert(timezone).tz_localize(None)
        out.append({"id": event["id"], "start": start, "city": CITIES[city], "league": league,
                    "home": teams.get("home", ""), "away": teams.get("away", "")})
    return out


def get_json(url: str, params: Dict, config: Dict) -> Dict:
    response = requests.get(url, params=params, timeout=config["timeout_seconds"])
    response.raise_for_status()
    time.sleep(config["request_delay_seconds"])
    return response.json()


def download_schedule(config: Dict = EVENTS_CONFIG) -> pd.DataFrame:
    """
    Download every configured team's games from first_season to next year and save them.

    Raises:
        RuntimeError: If any request failed; the saved file is left as it was.
    """
    rows, failures = [], 0
    last_season = date.today().year + 1
    for team in config["teams"]:
        url = f"{config['api_url']}/{team['sport']}/teams/{team['team']}/schedule"
        for season in range(config["first_season"], last_season + 1):
            for season_type in team["season_types"]:
                params = {"season": season, "seasontype": season_type}
                try:
                    rows += parse_events(get_json(url, params, config), team["league"], config["timezone"])
                except (requests.RequestException, ValueError) as e:
                    failures += 1
                    logger.warning("%s %s season %s type %s: %s", team["league"], team["team"], season, season_type, e)
    for tournament in config["tournaments"]:
        url = f"{config['api_url']}/{tournament['sport']}/scoreboard"
        day, end = date.fromisoformat(tournament["start"]), date.fromisoformat(tournament["end"])
        while day <= end:
            try:
                data = get_json(url, {"dates": day.strftime("%Y%m%d")}, config)
                rows += parse_events(data, tournament["league"], config["timezone"])
            except (requests.RequestException, ValueError) as e:
                failures += 1
                logger.warning("%s %s: %s", tournament["league"], day, e)
            day += timedelta(days=1)
    if failures:
        raise RuntimeError(f"{failures} schedule requests failed; kept {config['output_path']} unchanged")
    games = pd.DataFrame(rows).drop_duplicates("id").sort_values("start").reset_index(drop=True)
    path = config["output_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    games.to_csv(path, index=False)
    return games


def load_schedule(path: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Saved games, or None when they have not been downloaded. Re-read when the file changes."""
    path = Path(path or EVENTS_CONFIG["output_path"])
    try:
        return read_schedule(str(path), path.stat().st_mtime)
    except FileNotFoundError:
        return None


@lru_cache(maxsize=2)
def read_schedule(path: str, mtime: float) -> pd.DataFrame:
    """The games in a saved schedule; mtime only keys the cache."""
    return pd.read_csv(path, parse_dates=["start"])


def nearest_game(starts: np.ndarray, targets: np.ndarray, window_hours: float) -> np.ndarray:
    """
    Signed hours from the nearest game start to each target hour's middle
    (positive = after the start), NaN when none is within window_hours.
    Both arrays are datetime64 and starts is sorted.
    """
    out = np.full(len(targets), np.nan)
    if not len(starts):
        return out
    mid = targets + np.timedelta64(30, "m")
    i = np.searchsorted(starts, mid)
    for side in (np.clip(i - 1, 0, len(starts) - 1), np.clip(i, 0, len(starts) - 1)):
        hours = (mid - starts[side]) / np.timedelta64(1, "h")
        closer = (np.abs(hours) <= window_hours) & (np.isnan(out) | (np.abs(hours) < np.abs(out)))
        out[closer] = hours[closer]
    return out


def target_features(schedule: Optional[pd.DataFrame], targets: Iterable, config: Dict = EVENTS_CONFIG) -> pd.DataFrame:
    """
    Event features for each target hour.

    Only the games that showed an effect on the test year are used: the
    signed hours from the nearest stadium game in Seattle (Seahawks or World
    Cup, NaN when none is within window_hours), and whether Toronto plays the
    Mariners in Seattle that day. Other teams' games are too frequent to tell
    apart from an ordinary day.
    """
    hours = pd.DatetimeIndex(targets)
    out = pd.DataFrame(index=range(len(hours)))
    out["seattle_stadium_game_hours"] = np.nan
    out["toronto_at_seattle"] = 0
    if schedule is None or schedule.empty:
        return out
    stadium = schedule[(schedule.city == "sea") & schedule.league.isin(config["stadium_leagues"])].sort_values("start")
    out["seattle_stadium_game_hours"] = nearest_game(stadium.start.values.astype("datetime64[ns]"),
                                                        hours.values.astype("datetime64[ns]"), config["window_hours"])
    toronto = schedule[(schedule.city == "sea") & (schedule.league == "mlb") & (schedule.away == "TOR")]
    out["toronto_at_seattle"] = hours.normalize().isin(set(toronto.start.dt.normalize())).astype(int)
    return out


def main() -> None:
    """Command line entry point: download or refresh the schedules."""
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    games = download_schedule()
    print(f"Saved {len(games)} games, {games.start.min()} to {games.start.max()}, to {EVENTS_CONFIG['output_path']}")
    print(games.groupby(["city", "league"]).size().to_string())


if __name__ == "__main__":
    main()
