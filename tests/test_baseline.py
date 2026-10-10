"""Tests for the baseline profile + nowcast predictor."""
from datetime import date, datetime

import pytest

from border_predictor.baseline import BaselinePredictor, HolidayCalendar, load_profile


@pytest.fixture
def flat_predictor():
    """Predictor with a constant 10 minute profile and simple multipliers."""
    profile = {(m, d, h): 10.0 for m in range(1, 13) for d in range(7) for h in range(24)}
    config = {"holiday_multiplier": 2.0, "holiday_adjacent_multiplier": 1.5,
              "nowcast_decay_minutes": 60.0}
    return BaselinePredictor(config=config, profile=profile)


def test_load_profile_covers_every_cell():
    profile = load_profile()
    assert len(profile) == 12 * 7 * 24
    assert all(v >= 0 for v in profile.values())


def test_holiday_calendar_covers_both_sides():
    calendar = HolidayCalendar()
    assert calendar.day_type(date(2026, 7, 1)) == "holiday"   # Canada Day
    assert calendar.day_type(date(2026, 7, 3)) == "holiday"   # observed Independence Day
    assert calendar.day_type(date(2026, 8, 2)) == "holiday_adjacent"  # day before BC Day
    assert calendar.day_type(date(2026, 10, 21)) == "normal"


def test_expected_applies_holiday_multipliers(flat_predictor):
    assert flat_predictor.expected(datetime(2026, 10, 21, 12)) == 10.0
    assert flat_predictor.expected(datetime(2026, 7, 1, 12)) == 20.0
    assert flat_predictor.expected(datetime(2026, 8, 2, 12)) == 15.0


def test_nowcast_starts_at_observation_and_decays_to_profile(flat_predictor):
    observed_at = datetime(2026, 10, 21, 12)

    at_observation = flat_predictor.predict(observed_at, observed_at, 40.0)
    one_hour = flat_predictor.predict(datetime(2026, 10, 21, 13), observed_at, 40.0)
    ten_hours = flat_predictor.predict(datetime(2026, 10, 21, 22), observed_at, 40.0)

    assert at_observation == pytest.approx(40.0)
    assert one_hour == pytest.approx(10.0 + 30.0 / 2.718281828, rel=1e-6)
    assert ten_hours == pytest.approx(10.0, abs=0.01)


def test_nowcast_never_negative(flat_predictor):
    observed_at = datetime(2026, 10, 21, 12)
    assert flat_predictor.predict(datetime(2026, 10, 21, 12, 30), observed_at, 0.0) >= 0.0
