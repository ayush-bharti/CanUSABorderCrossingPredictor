"""Tests for the scheduled refresh's checks and swaps, and live accuracy tracking (no network)."""
import json
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from border_predictor import app as app_module
from border_predictor import refresh, tracking


@pytest.fixture
def dirs(tmp_path):
    config = dict(refresh.REFRESH_CONFIG, live_dir=tmp_path / "live", candidate_dir=tmp_path / "candidate",
                  previous_dir=tmp_path / "previous")
    for key in ("live_dir", "candidate_dir"):
        config[key].mkdir()
    return config


def save_lane(model_dir, lane_id, mae, tag, spike=False):
    (model_dir / f"baseline_lane{lane_id}.json").write_text(tag)
    kinds = ["median", "low", "high"] + (["spike"] if spike else [])
    for kind in kinds:
        (model_dir / f"gbm_lane{lane_id}_{kind}_upto6h.txt").write_text(tag)
    (model_dir / f"gbm_lane{lane_id}_meta.json").write_text(json.dumps({"validation_mae": mae}))


def test_check_lane_rejects_worse_validation_mae(dirs, monkeypatch):
    monkeypatch.setattr(refresh, "forecast_check", lambda lane_id, model_dir: None)
    save_lane(dirs["live_dir"], 4, 8.0, "old")
    save_lane(dirs["candidate_dir"], 4, 8.1, "new")  # within 2%
    assert refresh.check_lane(4, dirs)[0]
    save_lane(dirs["candidate_dir"], 4, 8.3, "new")
    ok, reason = refresh.check_lane(4, dirs)
    assert not ok and "tolerance" in reason
    monkeypatch.setattr(refresh, "forecast_check", lambda lane_id, model_dir: "non-finite forecast")
    save_lane(dirs["candidate_dir"], 4, 7.0, "new")
    assert refresh.check_lane(4, dirs) == (False, "non-finite forecast")


def test_swap_keeps_previous_and_rollback_restores(dirs):
    # The live lane has a spike classifier the candidate no longer trains; lane 11's files must not be touched
    save_lane(dirs["live_dir"], 1, 3.0, "old", spike=True)
    save_lane(dirs["live_dir"], 11, 2.0, "other")
    save_lane(dirs["candidate_dir"], 1, 2.9, "new")
    refresh.swap_lane(1, dirs["candidate_dir"], dirs)
    live, previous = dirs["live_dir"], dirs["previous_dir"]
    assert (live / "baseline_lane1.json").read_text() == "new"
    assert not (live / "gbm_lane1_spike_upto6h.txt").exists()
    assert (previous / "gbm_lane1_spike_upto6h.txt").read_text() == "old"
    assert (live / "baseline_lane11.json").read_text() == "other"
    assert refresh.lane_files(1, dirs["candidate_dir"]) == []

    refresh.swap_lane(1, previous, dirs)  # what rollback does for each lane
    assert (live / "gbm_lane1_spike_upto6h.txt").read_text() == "old"
    assert (previous / "baseline_lane1.json").read_text() == "new"


def readings(start, end, delay=5.0):
    idx = pd.date_range(start, end, freq="5min")
    return pd.DataFrame({"delay": delay, "vol": 1.0, "avail": 1}, index=idx)


def test_complete_hours_waits_for_the_last_poll():
    hours = tracking.complete_hours(4, readings("2026-10-13 08:00", "2026-10-13 10:50"))
    assert hours.hour.tolist() == [pd.Timestamp("2026-10-13 08:00"), pd.Timestamp("2026-10-13 09:00")]
    assert (hours.actual == 5.0).all() and (hours.lane == 4).all()


@pytest.fixture
def tracking_config(tmp_path):
    return dict(tracking.TRACKING_CONFIG, output_dir=tmp_path / "tracking")


def fake_forecast(lane_id, recent, others, wx, made_at, config):
    origin = (recent.index[-1] + pd.Timedelta(minutes=5)).ceil("H")
    targets = [origin + pd.Timedelta(hours=h) for h in (0, 1)]
    return pd.DataFrame({"made_at": made_at, "lane": lane_id, "origin": origin, "horizon": [0, 1],
                         "target": targets, "baseline": 6.0, "gbm": [4.0, 9.0], "low": 2.0, "high": 8.0,
                         "spike_prob": np.nan, "spike_alert": np.nan, "model_version": "v1"})


def test_snapshot_logs_fresh_lanes_and_records_actuals(tracking_config, monkeypatch):
    monkeypatch.setattr(tracking, "lane_forecast", fake_forecast)
    monkeypatch.setitem(tracking.CASCADE_CONFIG, "lanes", {4: "PA South Cars", 11: "PA South NEXUS"})
    data = {4: readings("2026-10-12 00:00", "2026-10-13 10:25"), 11: readings("2026-10-12 00:00", "2026-10-13 04:50")}
    logged = tracking.snapshot(now=datetime(2026, 10, 13, 10, 30), fetch=data.get, fetch_weather=lambda: None,
                               config=tracking_config)
    assert logged == 2  # lane 11's reading is stale: actuals recorded, no forecast
    actuals = tracking.load_actuals(tracking_config)
    assert set(actuals.lane) == {4, 11} and actuals.hour.max() == pd.Timestamp("2026-10-13 09:00")

    # The hour comes in: its forecasts are scored against it
    later = readings("2026-10-13 10:30", "2026-10-13 12:55", delay=[6.0] * 18 + [10.0] * 12)
    data[4] = pd.concat([data[4], later])
    tracking.snapshot(now=datetime(2026, 10, 13, 13, 0), fetch=data.get, fetch_weather=lambda: None,
                      config=tracking_config)
    rows = tracking.scored(tracking.load_forecasts(pd.Timestamp("2026-10-13"), tracking_config),
                           tracking.load_actuals(tracking_config))
    first = rows[rows.origin == pd.Timestamp("2026-10-13 11:00")].set_index("horizon")
    assert first.actual.tolist() == [6.0, 10.0]
    scores = tracking.score(first.reset_index()).loc[4]
    assert scores.mae.tolist() == [2.0, 1.0]
    assert scores.baseline_mae.tolist() == [0.0, 4.0]
    assert scores.coverage.tolist() == [1.0, 0.0]
    assert scores.alerts.isna().all()


def test_score_counts_spike_alerts():
    rows = pd.DataFrame({"lane": 4, "horizon": 0, "target": pd.date_range("2026-10-13", periods=4, freq="H"),
                         "baseline": 10.0, "gbm": 20.0, "low": 5.0, "high": 40.0, "actual": [35.0, 10.0, 45.0, 5.0],
                         "spike_prob": [0.6, 0.5, 0.1, 0.0], "spike_alert": [1.0, 1.0, 0.0, 0.0]})
    scores = tracking.score(rows).loc[(4, 0)]
    assert (scores.spikes, scores.alerts, scores.hits) == (2, 2, 1)
    assert scores.coverage == 0.75


def test_accuracy_page_renders_with_and_without_data(tracking_config, monkeypatch):
    client = app_module.app.test_client()
    monkeypatch.setattr(app_module.tracking, "recent_scores", lambda days: tracking.scored(
        tracking.load_forecasts(pd.Timestamp("2000-01-01"), tracking_config), tracking.load_actuals(tracking_config)))
    page = client.get("/accuracy").get_data(as_text=True)
    assert "No scored forecasts" in page

    forecasts = fake_forecast(4, readings("2026-10-13 08:00", "2026-10-13 09:55"), {}, None,
                              datetime(2026, 10, 13, 10), tracking_config)
    tracking.append_forecasts(forecasts, tracking_config)
    tracking.update_actuals(pd.DataFrame({"lane": 4, "hour": forecasts.target, "actual": 5.0}), tracking_config)
    page = client.get("/accuracy?days=7").get_data(as_text=True)
    assert "Daily error" in page and "1.0 min" in page


def test_recent_readings_stop_at_the_last_poll_with_a_delay(monkeypatch):
    # A NEXUS lane closed overnight: its downloaded month ends with polls that have no delay
    data = readings("2026-09-29 00:00", "2026-09-30 23:55")
    data.loc["2026-09-30 23:00":, "delay"] = np.nan
    monkeypatch.setattr(refresh.cascade, "load_lane", lambda lane_id, last_months=None: data)
    recent = refresh.recent_readings(6, days=1)
    assert recent.index[-1] == pd.Timestamp("2026-09-30 22:55")
    assert recent.index[0] == pd.Timestamp("2026-09-29 23:00")
