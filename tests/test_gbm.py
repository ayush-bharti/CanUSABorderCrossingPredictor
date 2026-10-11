"""Tests for LightGBM feature building."""
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from border_predictor.baseline import BaselinePredictor
from border_predictor.evaluate import backtest, live_readings
from border_predictor.gbm import band_of, build_features, fit_gbm, forecast, holiday_distances


def test_holiday_distances_signed_and_capped():
    dist = holiday_distances([date(2026, 6, 29), date(2026, 7, 1), date(2026, 7, 2), date(2026, 3, 10)])
    assert dist.loc[date(2026, 6, 29), "days_from_holiday_ca"] == -2   # Canada Day ahead
    assert dist.loc[date(2026, 7, 1), "days_from_holiday_ca"] == 0
    assert dist.loc[date(2026, 7, 2), "days_from_holiday_ca"] == 1
    assert dist.loc[date(2026, 3, 10), "days_from_holiday_us"] == 7    # nothing within a week


@pytest.fixture
def synthetic_lane():
    """Two months of 5-minute readings where delay equals the hour of day."""
    idx = pd.date_range("2025-01-01", "2025-02-28 23:55", freq="5min")
    return pd.DataFrame({"delay": idx.hour.astype(float), "vol": 1.0, "avail": 1.0}, index=idx)


def test_features_match_backtest_and_use_only_past_data(synthetic_lane):
    hourly = synthetic_lane["delay"].resample("H").mean()
    live = live_readings(synthetic_lane)
    model = BaselinePredictor.fit(hourly["2025-01"], decay_minutes=60.0)
    horizons = [0, 1, 24, 168]

    f = build_features(synthetic_lane, hourly, live, model, hourly["2025-02-10":"2025-02-20"].index, horizons)
    b = backtest(hourly, live, {"baseline": model.predict}, "2025-02-10", "2025-02-20", horizons)

    assert len(f) == len(b)
    np.testing.assert_allclose(f.baseline.values, b.baseline.values)
    np.testing.assert_allclose(f.actual.values, b.actual.values)
    # Lags are hours that ended before the origin
    assert (f.lag_1h == (f.origin.dt.hour - 1) % 24).all()
    # Same-hour lags for the target exist only when that hour is already past at the origin
    assert f.loc[f.h >= 24, "lag_target_24h"].isna().all()
    assert f.loc[f.h >= 168, "lag_target_168h"].isna().all()
    assert f.loc[f.h < 24, "lag_target_24h"].notna().all()


def test_forecast_starts_after_latest_reading(synthetic_lane):
    hourly = synthetic_lane["delay"].resample("H").mean()
    model = BaselinePredictor.fit(hourly["2025-01"], decay_minutes=60.0)
    train = build_features(synthetic_lane, hourly, live_readings(synthetic_lane), model,
                           hourly["2025-01-10":"2025-01-31"].index, [0, 1, 2, 24])
    boosters = [fit_gbm(train, "l1", 20)] * 3

    recent = synthetic_lane["2025-02-18":"2025-02-26 20:55"]
    out = forecast(recent, model, boosters, [datetime(2025, 2, 26, 20, 30), datetime(2025, 2, 26, 21, 10),
                                            datetime(2025, 2, 27, 3), datetime(2025, 3, 10)])

    assert out.attrs["origin"] == pd.Timestamp("2025-02-26 21:00")
    # 20:30 is before the origin and 10 March is beyond a week, so both are dropped
    assert list(out.index) == [pd.Timestamp("2025-02-26 21:00"), pd.Timestamp("2025-02-27 03:00")]
    assert (out.gbm >= 0).all() and out.notna().all().all()


def test_forecast_mid_hour_uses_latest_polls(synthetic_lane):
    hourly = synthetic_lane["delay"].resample("H").mean()
    model = BaselinePredictor.fit(hourly["2025-01"], decay_minutes=60.0)
    boosters = [fit_gbm(build_features(synthetic_lane, hourly, live_readings(synthetic_lane), model,
                                       hourly["2025-01-10":"2025-01-31"].index, [0, 1]), "l1", 20)] * 3

    recent = synthetic_lane["2025-02-18":"2025-02-26 21:35"].copy()
    recent.loc["2025-02-26 21:25":, "delay"] = 50.0
    out = forecast(recent, model, boosters, [datetime(2025, 2, 26, 22)])

    # Origin moves to the next boundary instead of discarding the polls since 21:00
    assert out.attrs["origin"] == pd.Timestamp("2025-02-26 22:00")
    # The profile has no February cells, so the baseline is the decayed live reading: the
    # latest polls (50), not the 20:45-20:55 ones (20), 35 minutes before the target hour's midpoint
    assert out.baseline.iloc[0] == pytest.approx(50.0 * np.exp(-35 / 60))
