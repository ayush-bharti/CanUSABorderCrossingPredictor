"""Forecasts for the web app: loads each lane's models once and caches Cascade fetches."""
from typing import Dict, List, Optional
import logging
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from border_predictor import gbm
from border_predictor.baseline import BaselinePredictor
from border_predictor.cascade import fetch_recent
from border_predictor.config.config import BASELINE_CONFIG, CASCADE_CONFIG
from border_predictor.train_baseline import model_path

logger = logging.getLogger(__name__)

# Cascade polls every 5 minutes; refetching more often than that gains nothing
RECENT_TTL_SECONDS = 300


@dataclass(frozen=True)
class Lane:
    """One Cascade lane, split into the parts the UI groups by."""
    lane_id: int
    crossing: str   # "Peace Arch" or "Pacific Highway"
    direction: str  # "south" (into the US) or "north" (into Canada)
    kind: str       # "cars" or "nexus"


def lanes() -> List[Lane]:
    """Every configured lane, parsed from names like "Peace Arch South Cars"."""
    out = []
    for lane_id, name in CASCADE_CONFIG["lanes"].items():
        match = re.fullmatch(r"(.+) (South|North) (Cars|NEXUS)", name)
        if match:
            crossing, direction, kind = match.groups()
            out.append(Lane(lane_id, crossing, direction.lower(), kind.lower()))
    return out


def local_now() -> datetime:
    """Current wall-clock time at the crossings, naive like the Cascade data."""
    return datetime.now(ZoneInfo(BASELINE_CONFIG["timezone"])).replace(tzinfo=None)


class Forecaster:
    """Serves hourly forecasts per lane: LightGBM from recent readings, the baseline beyond its range."""

    def __init__(self, fetch=fetch_recent, ttl_seconds: float = RECENT_TTL_SECONDS):
        self._fetch = fetch
        self._ttl = ttl_seconds
        self._models: Dict[int, tuple] = {}
        self._recent: Dict[int, tuple] = {}
        self._lock = threading.Lock()

    def models(self, lane_id: int):
        """(baseline, boosters) for a lane; boosters is None if LightGBM is not trained."""
        if lane_id not in self._models:
            path = model_path(lane_id)
            if not path.exists():
                raise FileNotFoundError(f"No baseline for lane {lane_id}; run python -m border_predictor.gbm train")
            self._models[lane_id] = (BaselinePredictor.load(path), gbm.load_boosters(lane_id))
        return self._models[lane_id]

    def recent(self, lane_id: int) -> Optional[pd.DataFrame]:
        """Recent 5-minute readings, cached; None if Cascade cannot be reached."""
        with self._lock:
            cached = self._recent.get(lane_id)
            if cached and time.monotonic() - cached[0] < self._ttl:
                return cached[1]
        try:
            data = self._fetch(lane_id)
        except Exception as e:  # network or API errors: fall back to the profile, keep serving
            logger.warning("Cascade fetch failed for lane %s: %s", lane_id, e)
            data = None
        with self._lock:
            self._recent[lane_id] = (time.monotonic(), data)
        return data

    def forecast(self, lane_id: int, targets: List[datetime]) -> Dict:
        """
        Predicted mean delay for the hour each target falls in.

        Returns:
            {"latest": {"time", "delay"} or None, "hours": [{"time", "delay", "source", "day_type"}]}.
            source is "gbm" within LightGBM's range of the latest reading and
            "profile" otherwise (no live data, no trained model, or too far ahead).
        """
        baseline, boosters = self.models(lane_id)
        recent = self.recent(lane_id)
        hours = sorted({pd.Timestamp(t).floor("H") for t in targets})
        latest, predicted = None, {}
        if recent is not None and not recent.empty:
            latest = {"time": recent.index[-1].to_pydatetime(), "delay": float(recent["delay"].iloc[-1])}
            if boosters:
                result = gbm.forecast(recent, baseline, boosters, hours)
                predicted = {t: float(v) for t, v in result["gbm"].items()}

        out = []
        for hour in hours:
            if hour in predicted:
                delay, source = predicted[hour], "gbm"
            else:
                delay, source = baseline.expected((hour + timedelta(minutes=30)).to_pydatetime()), "profile"
            out.append({
                "time": hour.to_pydatetime(),
                "delay": max(0.0, delay),
                "source": source,
                "day_type": baseline.calendar.day_type(hour.date()),
            })
        return {"latest": latest, "hours": out}

    def typical_week(self, lane_id: int) -> pd.DataFrame:
        """Typical delay by weekday (rows, Monday first) and hour (columns), averaged over months."""
        baseline, _ = self.models(lane_id)
        profile = pd.Series(baseline.profile)
        profile.index.names = ["month", "weekday", "hour"]
        return profile.groupby(["weekday", "hour"]).mean().unstack("hour")
