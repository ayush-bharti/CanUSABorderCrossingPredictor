"""Tests for LightGBM feature building, models and forecasts."""
import json
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from border_predictor.baseline import BaselinePredictor
from border_predictor.evaluate import backtest, live_readings
from border_predictor import gbm
from border_predictor.gbm import (band_of, best_f1_threshold, build_features, feature_columns, fit_gbm,
                                   forecast, holiday_distances)


def test_holiday_distances_signed_and_capped():
    dist = holiday_distances([date(2026, 6, 29), date(2026, 7, 1), date(2026, 7, 2), date(2026, 3, 10)])
    assert dist.loc[date(2026, 6, 29), "days_from_holiday_ca"] == -2   # Canada Day ahead
    assert dist.loc[date(2026, 7, 1), "days_from_holiday_ca"] == 0
    assert dist.loc[date(2026, 7, 2), "days_from_holiday_ca"] == 1
    assert dist.loc[date(2026, 3, 10), "days_from_holiday_us"] == 7    # nothing within a week


def test_holiday_names_are_stable_codes():
    dist = holiday_distances([date(2026, 6, 29), date(2026, 11, 27), date(2026, 3, 10), date(2027, 7, 5)])
    ca_names, us_names = gbm.holiday_calendars()["ca"][1], gbm.holiday_calendars()["us"][1]
    assert ca_names[int(dist.loc[date(2026, 6, 29), "holiday_ca"])] == "Canada Day"
    assert us_names[int(dist.loc[date(2026, 11, 27), "holiday_us"])] == "Thanksgiving"   # Black Friday
    assert np.isnan(dist.loc[date(2026, 3, 10), "holiday_us"])
    # Observed holidays map to the same code as the holiday itself
    assert us_names[int(dist.loc[date(2027, 7, 5), "holiday_us"])] == "Independence Day"


def test_history_rows_are_flagged_and_weighted(synthetic_lane):
    hourly = synthetic_lane["delay"].resample("H").mean()
    model = BaselinePredictor.fit(hourly["2025-01"], decay_minutes=60.0)
    f = build_features(synthetic_lane, hourly, live_readings(synthetic_lane), model,
                       hourly["2025-02-01":"2025-02-03"].index, [0])
    assert (f.pre_covid == 0).all()
    history = f.head(5).assign(pre_covid=1)
    combined = gbm.with_history(f, history, 0.25)
    assert len(combined) == len(f) + 5
    assert (combined.weight[:len(f)] == 1.0).all() and (combined.weight[len(f):] == 0.25).all()
    assert gbm.with_history(f, history, 0.0) is f
    # Weighted rows train without error
    fit_gbm(combined, "median", 5)


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
    models = {"median": [fit_gbm(train, "median", 20)] * 3}

    recent = synthetic_lane["2025-02-18":"2025-02-26 20:55"]
    out = forecast(recent, model, models, [datetime(2025, 2, 26, 20, 30), datetime(2025, 2, 26, 21, 10),
                                            datetime(2025, 2, 27, 3), datetime(2025, 3, 10)])

    assert out.attrs["origin"] == pd.Timestamp("2025-02-26 21:00")
    # 20:30 is before the origin and 10 March is beyond a week, so both are dropped
    assert list(out.index) == [pd.Timestamp("2025-02-26 21:00"), pd.Timestamp("2025-02-27 03:00")]
    assert (out.gbm >= 0).all() and out[["baseline", "gbm"]].notna().all().all()


def test_forecast_mid_hour_uses_latest_polls(synthetic_lane):
    hourly = synthetic_lane["delay"].resample("H").mean()
    model = BaselinePredictor.fit(hourly["2025-01"], decay_minutes=60.0)
    models = {"median": [fit_gbm(build_features(synthetic_lane, hourly, live_readings(synthetic_lane), model,
                                                hourly["2025-01-10":"2025-01-31"].index, [0, 1]), "median", 20)] * 3}

    recent = synthetic_lane["2025-02-18":"2025-02-26 21:35"].copy()
    recent.loc["2025-02-26 21:25":, "delay"] = 50.0
    out = forecast(recent, model, models, [datetime(2025, 2, 26, 22)])

    # Origin moves to the next boundary instead of discarding the polls since 21:00
    assert out.attrs["origin"] == pd.Timestamp("2025-02-26 22:00")
    # The profile has no February cells, so the baseline is the decayed live reading: the
    # latest polls (50), not the 20:45-20:55 ones (20), 35 minutes before the target hour's midpoint
    assert out.baseline.iloc[0] == pytest.approx(50.0 * np.exp(-35 / 60))


@pytest.fixture
def two_lanes(synthetic_lane):
    """On alternate days lane 2 queues 20 minutes at 16-18h and lane 1 then queues 40 minutes at 18h."""
    lane, other = synthetic_lane.copy(), synthetic_lane.copy()
    lane["delay"], other["delay"] = 2.0, 2.0
    busy_days = pd.date_range("2025-01-01", "2025-02-28", freq="2D")
    for day in busy_days:
        other.loc[day + pd.Timedelta(hours=16):day + pd.Timedelta(hours=17, minutes=55), "delay"] = 20.0
        lane.loc[day + pd.Timedelta(hours=18):day + pd.Timedelta(hours=18, minutes=55), "delay"] = 40.0
    return lane, other


def test_cross_lane_models_bounds_and_spike_alerts(two_lanes, monkeypatch):
    lane, other = two_lanes
    # Small synthetic set: allow small leaves
    monkeypatch.setitem(gbm.GBM_CONFIG, "params", dict(gbm.GBM_CONFIG["params"], min_data_in_leaf=5))
    hourly, other_hourly = lane["delay"].resample("H").mean(), other["delay"].resample("H").mean()
    model = BaselinePredictor.fit(hourly["2025-01"], decay_minutes=60.0)
    others = {2: (other_hourly, live_readings(other))}
    train = build_features(lane, hourly, live_readings(lane), model, hourly["2025-01-01":"2025-02-20"].index,
                           [0, 1, 2], others=others)
    assert {"other2_live", "other2_lag_1h"} <= set(train.columns)
    # Cross-lane features are what is known at the origin
    row = train[(train.origin == pd.Timestamp("2025-01-05 17:00"))].iloc[0]
    assert row.other2_lag_1h == other_hourly[pd.Timestamp("2025-01-05 16:00")]

    features = feature_columns(train, cross_lane=True)
    models = {kind: [fit_gbm(train, kind, 200, features=features)] * 3 for kind in ("median", "low", "high", "spike")}
    models["spike_thresholds"] = [0.5, 0.5, 0.5]
    assert gbm.other_lanes_needed(models) == [2]

    busy, quiet = pd.Timestamp("2025-02-24"), pd.Timestamp("2025-02-25")
    assert other_hourly[busy + pd.Timedelta(hours=16)] == 20 and other_hourly[quiet + pd.Timedelta(hours=16)] == 2
    for day, spike in ((busy, True), (quiet, False)):
        recent = lane[day - pd.Timedelta(days=8):day + pd.Timedelta(hours=16, minutes=55)]
        # The other lane has polled past this lane's latest reading; forecast must not use those polls
        other_recent = other[day - pd.Timedelta(days=8):day + pd.Timedelta(hours=17, minutes=30)]
        out = forecast(recent, model, models, [day + pd.Timedelta(hours=18)], {2: other_recent})
        assert out.attrs["origin"] == day + pd.Timedelta(hours=17)
        r = out.iloc[0]
        assert r.low <= r.gbm <= r.high
        assert bool(r.spike_alert) == spike, (day, r.spike_prob)

    # Without the other lane's readings its features are missing, not an error
    recent = lane[busy - pd.Timedelta(days=8):busy + pd.Timedelta(hours=16, minutes=55)]
    assert len(forecast(recent, model, models, [busy + pd.Timedelta(hours=18)])) == 1


def test_best_f1_threshold():
    prob = np.array([0.1, 0.2, 0.6, 0.7, 0.9])
    assert 0.2 < best_f1_threshold(prob, np.array([False, False, True, True, True])) <= 0.6
    assert best_f1_threshold(prob, np.zeros(5, dtype=bool)) == 0.5


def test_load_models_needs_median(tmp_path, monkeypatch, synthetic_lane):
    monkeypatch.setitem(gbm.BASELINE_CONFIG, "model_dir", tmp_path)
    assert gbm.load_models(4) is None
    hourly = synthetic_lane["delay"].resample("H").mean()
    model = BaselinePredictor.fit(hourly["2025-01"], decay_minutes=60.0)
    train = build_features(synthetic_lane, hourly, live_readings(synthetic_lane), model,
                           hourly["2025-01-10":"2025-01-31"].index, [0, 1])
    for kind in ("median", "spike"):
        for path in gbm.model_paths(4, kind):
            fit_gbm(train, kind, 5).save_model(str(path))
    gbm.meta_path(4).write_text(json.dumps({"spike_thresholds": [0.3, 0.4, 0.5]}))
    models = gbm.load_models(4)
    assert set(models) == {"median", "spike", "spike_thresholds"}
    assert models["spike_thresholds"] == [0.3, 0.4, 0.5]
