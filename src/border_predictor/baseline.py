"""Baseline wait time predictor: historical profile plus a decaying nowcast.

Planning horizon: the CBP historical average for (month, weekday, hour),
scaled up on US/Washington and Canada/BC holidays and the days around them.

Short horizon: the latest reading's gap from the profile is carried forward
and fades exponentially, so a predicted wait starts at what was just observed
and drifts back to the typical value for that time.
"""
from typing import Dict, Optional
import math
import re
from datetime import date, datetime, timedelta
from pathlib import Path

import holidays
import pandas as pd

from border_predictor.config.config import BASELINE_CONFIG, DATA_DIR


MONTHS = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "June": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}
WEEKDAYS = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
    "Friday": 4, "Saturday": 5, "Sunday": 6,
}


def _parse_hour(label: str) -> int:
    label = label.strip()
    if label == "Midnight":
        return 0
    if label == "Noon":
        return 12
    match = re.fullmatch(r"(\d{1,2}):00\s*([AP]M)", label)
    if not match:
        raise ValueError(f"Unrecognised time label: {label!r}")
    hour = int(match.group(1)) % 12
    return hour + 12 if match.group(2) == "PM" else hour


def load_profile(data_dir: Path = DATA_DIR) -> Dict[tuple, float]:
    """
    Load the CBP monthly average tables into a lookup.

    Args:
        data_dir: Directory with the *_WaitTimes.csv files.

    Returns:
        Mapping of (month, weekday, hour) to average wait in minutes,
        with weekday 0 = Monday.
    """
    profile = {}
    for path in sorted(data_dir.glob("*_WaitTimes.csv")):
        month = MONTHS[path.name.split("_")[0]]
        df = pd.read_csv(path, encoding="utf-8-sig")
        for _, row in df.iterrows():
            hour = _parse_hour(row["Time"])
            for column in df.columns[1:]:
                weekday = WEEKDAYS[column.split(" ")[0]]
                profile[(month, weekday, hour)] = float(row[column])

    expected = 12 * 7 * 24
    if len(profile) != expected:
        raise ValueError(f"Profile has {len(profile)} cells, expected {expected}")
    return profile


class HolidayCalendar:
    """Classifies dates as holiday, holiday-adjacent or ordinary for both sides of the border."""

    def __init__(self):
        self._us = holidays.US(subdiv="WA")
        self._ca = holidays.CA(subdiv="BC")

    def is_holiday(self, day: date) -> bool:
        return day in self._us or day in self._ca

    def day_type(self, day: date) -> str:
        """
        Args:
            day: Local calendar date.

        Returns:
            "holiday", "holiday_adjacent" (day before or after a holiday) or "normal".
        """
        if self.is_holiday(day):
            return "holiday"
        if self.is_holiday(day - timedelta(days=1)) or self.is_holiday(day + timedelta(days=1)):
            return "holiday_adjacent"
        return "normal"


class BaselinePredictor:
    """Historical profile with holiday scaling and an exponentially decaying nowcast."""

    def __init__(self, config: Optional[Dict] = None, profile: Optional[Dict[tuple, float]] = None):
        """
        Args:
            config: Baseline configuration. Defaults to BASELINE_CONFIG.
            profile: Precomputed profile lookup. Loaded from data files if None.
        """
        self.config = {**BASELINE_CONFIG, **(config or {})}
        self.profile = profile if profile is not None else load_profile()
        self.calendar = HolidayCalendar()
        self._multipliers = {
            "holiday": self.config["holiday_multiplier"],
            "holiday_adjacent": self.config["holiday_adjacent_multiplier"],
            "normal": 1.0,
        }

    def expected(self, when: datetime) -> float:
        """
        Typical wait for a local time, ignoring any live reading.

        Args:
            when: Naive datetime in the crossing's local time.

        Returns:
            Expected wait in minutes.
        """
        base = self.profile[(when.month, when.weekday(), when.hour)]
        return base * self._multipliers[self.calendar.day_type(when.date())]

    def predict(
        self,
        when: datetime,
        observed_at: Optional[datetime] = None,
        observed_wait: Optional[float] = None,
    ) -> float:
        """
        Predict the wait at a local time, using the latest reading if given.

        Args:
            when: Naive target datetime in local time.
            observed_at: Naive local time of the latest reading.
            observed_wait: Latest observed wait in minutes.

        Returns:
            Predicted wait in minutes, never negative.
        """
        expected = self.expected(when)
        if observed_at is None or observed_wait is None:
            return expected

        minutes_ahead = max(0.0, (when - observed_at).total_seconds() / 60)
        weight = math.exp(-minutes_ahead / self.config["nowcast_decay_minutes"])
        anomaly = observed_wait - self.expected(observed_at)
        return max(0.0, expected + weight * anomaly)
