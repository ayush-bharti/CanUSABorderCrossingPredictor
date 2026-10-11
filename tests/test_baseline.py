"""Tests for the baseline profile + nowcast predictor."""
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from border_predictor.baseline import BaselinePredictor, HolidayCalendar


@pytest.fixture
def flat_predictor():
    """Predictor with a constant 10 minute profile and simple multipliers."""
    profile = {(m, d, h): 10.0 for m in range(1, 13) for d in range(7) for h in range(24)}
    multipliers = {"holiday_ca": 2.0, "window_ca": 1.5}
    return BaselinePredictor(profile, multipliers, decay_minutes=60.0)


def test_holiday_calendar_labels_country_and_window():
    calendar = HolidayCalendar()
    assert calendar.day_type(date(2026, 7, 1)) == "holiday_ca"     # Canada Day
    assert calendar.day_type(date(2026, 7, 3)) == "holiday_us"     # observed Independence Day
    assert calendar.day_type(date(2026, 12, 25)) == "holiday_both"
    assert calendar.day_type(date(2026, 8, 1)) == "window_ca"      # Saturday before BC Day
    assert calendar.day_type(date(2026, 10, 21)) == "normal"


def test_expected_applies_multipliers(flat_predictor):
    assert flat_predictor.expected(datetime(2026, 10, 21, 12)) == 10.0
    assert flat_predictor.expected(datetime(2026, 7, 1, 12)) == 20.0
    assert flat_predictor.expected(datetime(2026, 8, 1, 12)) == 15.0
    assert flat_predictor.expected(datetime(2026, 7, 3, 12)) == 10.0   # unfitted type stays 1.0


def test_nowcast_starts_at_observation_and_decays_to_profile(flat_predictor):
    observed_at = datetime(2026, 10, 21, 12)

    at_observation = flat_predictor.predict(observed_at, observed_at, 40.0)
    one_hour = flat_predictor.predict(datetime(2026, 10, 21, 13), observed_at, 40.0)
    ten_hours = flat_predictor.predict(datetime(2026, 10, 21, 22), observed_at, 40.0)

    assert at_observation == pytest.approx(40.0)
    assert one_hour == pytest.approx(10.0 + 30.0 / np.e)
    assert ten_hours == pytest.approx(10.0, abs=0.01)


def test_nowcast_never_negative(flat_predictor):
    observed_at = datetime(2026, 10, 21, 12)
    assert flat_predictor.predict(datetime(2026, 10, 21, 12, 30), observed_at, 0.0) >= 0.0


def _synthetic_hourly():
    """Two years where weekday 5 pm is 30 min, everything else 5, and Canada-only holidays are doubled."""
    idx = pd.date_range("2023-01-01", "2024-12-31 23:00", freq="H")
    y = pd.Series(5.0, index=idx)
    y[(idx.hour == 17) & (idx.weekday < 5)] = 30.0
    calendar = HolidayCalendar()
    y[[calendar.day_type(d) == "holiday_ca" for d in idx.date]] *= 2
    return y


@pytest.mark.parametrize("profile_kind", ["month_weekday_hour", "weekday_hour_x_month"])
def test_fit_recovers_profile_and_holiday_effect(profile_kind):
    model = BaselinePredictor.fit(_synthetic_hourly(), profile_kind=profile_kind, min_holiday_days=2)

    # Month levels shift slightly with each month's mix of weekdays
    assert model.expected(datetime(2025, 3, 4, 17)) == pytest.approx(30.0, rel=0.01)
    assert model.expected(datetime(2025, 3, 4, 3)) == pytest.approx(5.0, rel=0.01)
    assert model.multipliers["holiday_ca"] == pytest.approx(2.0, rel=0.01)


def test_holiday_shrinkage_pulls_multipliers_toward_one():
    model = BaselinePredictor.fit(_synthetic_hourly(), min_holiday_days=2, holiday_shrinkage=0.5)
    assert model.multipliers["holiday_ca"] == pytest.approx(1.5)
    model.set_holiday_shrinkage(0.0)
    assert model.multipliers["holiday_ca"] == pytest.approx(1.0)


def test_fit_handles_lane_with_zero_median():
    quiet = pd.Series(0.0, index=pd.date_range("2023-01-01", "2023-12-31 23:00", freq="H"))
    quiet.iloc[::50] = 10.0
    model = BaselinePredictor.fit(quiet, statistic="median", profile_kind="weekday_hour_x_month")
    assert all(np.isfinite(v) for v in model.profile.values())


def test_save_and_load_round_trip(tmp_path, flat_predictor):
    path = tmp_path / "model.json"
    flat_predictor.save(path)
    loaded = BaselinePredictor.load(path)

    when, at = datetime(2026, 7, 1, 15), datetime(2026, 7, 1, 13)
    assert loaded.predict(when, at, 25.0) == pytest.approx(flat_predictor.predict(when, at, 25.0))
