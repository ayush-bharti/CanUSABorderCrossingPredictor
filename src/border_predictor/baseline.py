"""Baseline wait time predictor: historical profile plus a decaying nowcast.

Planning horizon: a typical delay for (month, weekday, hour), fitted from
Cascade Gateway history on ordinary days, then scaled by multipliers fitted
for Canada/BC and US/WA holidays and the long-weekend days around them.

Short horizon: the latest reading's gap from the profile is carried forward
and fades exponentially, so a predicted wait starts at what was just observed
and drifts back to the typical value for that time.
"""
from typing import Dict, Optional
import json
import math
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path

import holidays
import pandas as pd


HOLIDAY_WINDOW_DAYS = 2
DAY_TYPES = ["normal", "holiday_ca", "holiday_us", "holiday_both", "window_ca", "window_us", "window_both"]


class HolidayCalendar:
    """Classifies dates by Canada/BC and US/WA holidays and the days around them."""

    def __init__(self):
        self._ca = holidays.CA(subdiv="BC")
        self._us = holidays.US(subdiv="WA")

    @staticmethod
    def _label(prefix: str, ca: bool, us: bool) -> Optional[str]:
        if ca and us:
            return f"{prefix}_both"
        if ca:
            return f"{prefix}_ca"
        if us:
            return f"{prefix}_us"
        return None

    @lru_cache(maxsize=None)
    def day_type(self, day: date) -> str:
        """
        Args:
            day: Local calendar date.

        Returns:
            One of DAY_TYPES. "holiday_*" is the holiday itself; "window_*" is
            within HOLIDAY_WINDOW_DAYS of one, which covers long-weekend travel days.
        """
        label = self._label("holiday", day in self._ca, day in self._us)
        if label:
            return label
        nearby = [day + timedelta(days=d) for d in range(-HOLIDAY_WINDOW_DAYS, HOLIDAY_WINDOW_DAYS + 1) if d]
        label = self._label("window", any(d in self._ca for d in nearby), any(d in self._us for d in nearby))
        return label or "normal"


class BaselinePredictor:
    """Historical profile with fitted holiday multipliers and an exponentially decaying nowcast."""

    def __init__(
        self,
        profile: Dict[tuple, float],
        multipliers: Optional[Dict[str, float]] = None,
        decay_minutes: float = 180.0,
    ):
        """
        Args:
            profile: Mapping of (month, weekday, hour) to typical delay in minutes, weekday 0 = Monday.
            multipliers: Day type to multiplier. Missing types use 1.0.
            decay_minutes: Time constant for the nowcast fading back to the profile.
        """
        self.profile = profile
        self.multipliers = {t: 1.0 for t in DAY_TYPES}
        self.multipliers.update(multipliers or {})
        self.raw_multipliers = dict(self.multipliers)
        self.decay_minutes = decay_minutes
        self.calendar = HolidayCalendar()

    @classmethod
    def fit(
        cls,
        hourly: pd.Series,
        statistic: str = "mean",
        profile_kind: str = "month_weekday_hour",
        decay_minutes: float = 180.0,
        holiday_shrinkage: float = 1.0,
        min_holiday_days: int = 4,
    ) -> "BaselinePredictor":
        """
        Fit the profile and holiday multipliers from an hourly delay series.

        Args:
            hourly: Hourly mean delay indexed by naive local time (NaN allowed).
            statistic: "mean" or "median" for each profile cell.
            profile_kind: "month_weekday_hour" uses each cell directly;
                "weekday_hour_x_month" multiplies a (weekday, hour) shape by a
                per-month level, which is less noisy with few years of data.
            decay_minutes: Nowcast time constant to store on the model.
            holiday_shrinkage: Fraction of each fitted holiday effect to keep (0 = ignore
                holidays, 1 = use as fitted). A few extreme days can inflate the raw ratios.
            min_holiday_days: Day types seen on fewer days than this keep a multiplier of 1.0.

        Returns:
            Fitted BaselinePredictor.
        """
        calendar = HolidayCalendar()
        df = hourly.dropna().to_frame("y")
        df["month"], df["weekday"], df["hour"] = df.index.month, df.index.weekday, df.index.hour
        df["day_type"] = [calendar.day_type(d) for d in df.index.date]
        normal = df[df.day_type == "normal"]

        if profile_kind == "month_weekday_hour":
            cells = normal.groupby(["month", "weekday", "hour"]).y.agg(statistic)
            profile = {k: float(v) for k, v in cells.items()}
        elif profile_kind == "weekday_hour_x_month":
            shape = normal.groupby(["weekday", "hour"]).y.agg(statistic)
            # Month levels use means: a median of 0 on quiet lanes would make every ratio undefined
            level = normal.groupby("month").y.mean() / normal.y.mean()
            profile = {
                (m, w, h): float(shape[(w, h)] * level.get(m, 1.0))
                for m in range(1, 13) for (w, h) in shape.index
            }
        else:
            raise ValueError(f"Unknown profile_kind: {profile_kind!r}")

        model = cls(profile, decay_minutes=decay_minutes)
        df["base"] = [model.profile.get(k, float("nan")) for k in zip(df.month, df.weekday, df.hour)]
        for day_type, group in df[df.day_type != "normal"].groupby("day_type"):
            group = group.dropna(subset=["base"])
            if group.index.normalize().nunique() >= min_holiday_days and group.base.sum() > 0:
                model.raw_multipliers[day_type] = float(group.y.sum() / group.base.sum())
        model.set_holiday_shrinkage(holiday_shrinkage)
        return model

    def set_holiday_shrinkage(self, shrinkage: float) -> None:
        """
        Pull the fitted holiday multipliers toward 1.0.

        Args:
            shrinkage: Fraction of each fitted effect to keep, from 0 to 1.
        """
        self.multipliers = {t: 1.0 + shrinkage * (m - 1.0) for t, m in self.raw_multipliers.items()}

    def expected(self, when: datetime) -> float:
        """
        Typical delay for a local time, ignoring any live reading.

        Args:
            when: Naive datetime in the crossing's local time.

        Returns:
            Expected delay in minutes.
        """
        base = self.profile.get((when.month, when.weekday(), when.hour), 0.0)
        return base * self.multipliers[self.calendar.day_type(when.date())]

    def predict(
        self,
        when: datetime,
        observed_at: Optional[datetime] = None,
        observed_wait: Optional[float] = None,
    ) -> float:
        """
        Predict the delay at a local time, using the latest reading if given.

        Args:
            when: Naive target datetime in local time.
            observed_at: Naive local time of the latest reading.
            observed_wait: Latest observed delay in minutes.

        Returns:
            Predicted delay in minutes, never negative.
        """
        expected = self.expected(when)
        if observed_at is None or observed_wait is None:
            return expected

        minutes_ahead = max(0.0, (when - observed_at).total_seconds() / 60)
        weight = math.exp(-minutes_ahead / self.decay_minutes)
        anomaly = observed_wait - self.expected(observed_at)
        return max(0.0, expected + weight * anomaly)

    def save(self, path: Path) -> None:
        """Write the fitted model as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "profile": {f"{m},{w},{h}": v for (m, w, h), v in self.profile.items()},
            "multipliers": self.multipliers,
            "decay_minutes": self.decay_minutes,
        }, indent=1))

    @classmethod
    def load(cls, path: Path) -> "BaselinePredictor":
        """Read a model written by save."""
        data = json.loads(path.read_text())
        profile = {tuple(int(x) for x in k.split(",")): v for k, v in data["profile"].items()}
        return cls(profile, data["multipliers"], data["decay_minutes"])
