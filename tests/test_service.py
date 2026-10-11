"""Tests for the web forecast service and API (no network)."""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from border_predictor import app as app_module
from border_predictor import service
from border_predictor.baseline import BaselinePredictor


def test_lanes_parse_crossing_direction_and_kind():
    parsed = {lane.lane_id: (lane.crossing, lane.direction, lane.kind) for lane in service.lanes()}
    assert parsed[4] == ("Peace Arch", "south", "cars")
    assert parsed[14] == ("Pacific Highway", "north", "nexus")
    assert len(parsed) == 8


@pytest.fixture
def flat_baseline():
    profile = {(m, d, h): 10.0 for m in range(1, 13) for d in range(7) for h in range(24)}
    return BaselinePredictor(profile, decay_minutes=60.0)


def make_forecaster(baseline, fetch):
    forecaster = service.Forecaster(fetch=fetch)
    forecaster._models = {lane.lane_id: (baseline, None) for lane in service.lanes()}
    return forecaster


def test_forecast_falls_back_to_profile_when_cascade_fails(flat_baseline):
    def failing(lane_id):
        raise RuntimeError("Cascade down")

    forecaster = make_forecaster(flat_baseline, failing)
    result = forecaster.forecast(4, [datetime(2026, 10, 21, 12, 30)])
    assert result["latest"] is None
    assert result["hours"] == [{"time": datetime(2026, 10, 21, 12), "delay": 10.0,
                                "source": "profile", "day_type": "normal"}]


def test_recent_readings_are_cached(flat_baseline):
    calls = []
    idx = pd.date_range("2026-10-21 10:00", periods=6, freq="5min")
    data = pd.DataFrame({"delay": 5.0, "vol": 1.0, "avail": 1}, index=idx)

    def fetch(lane_id):
        calls.append(lane_id)
        return data

    forecaster = make_forecaster(flat_baseline, fetch)
    forecaster.forecast(4, [datetime(2026, 10, 21, 12)])
    result = forecaster.forecast(4, [datetime(2026, 10, 21, 13)])
    assert calls == [4]
    assert result["latest"] == {"time": datetime(2026, 10, 21, 10, 25), "delay": 5.0}


def test_typical_week_shape(flat_baseline):
    week = make_forecaster(flat_baseline, lambda lane_id: None).typical_week(4)
    assert week.shape == (7, 24)
    assert np.allclose(week.values, 10.0)


def test_api_returns_both_crossings(flat_baseline, monkeypatch):
    monkeypatch.setattr(app_module, "forecaster", make_forecaster(flat_baseline, lambda lane_id: None))
    client = app_module.app.test_client()

    forecast = client.get("/api/forecast?direction=north&kind=cars&hours=3").get_json()
    assert [c["crossing"] for c in forecast["crossings"]] == ["Peace Arch", "Pacific Highway"]
    assert len(forecast["crossings"][0]["hours"]) == 3

    plan = client.get("/api/plan?direction=south&kind=nexus&at=2026-12-24%2015:00").get_json()
    assert [h["time"] for h in plan["crossings"][0]["hours"]][2] == "2026-12-24 15:00"
    assert plan["crossings"][0]["hours"][2]["day_type"] == "Long weekend in both countries"

    assert client.get("/api/plan?direction=south&kind=cars&at=tomorrow").status_code == 400
    assert client.get("/api/forecast?direction=east&kind=cars").status_code == 404
    for page in ("/", "/visualize", "/about"):
        assert client.get(page).status_code == 200
