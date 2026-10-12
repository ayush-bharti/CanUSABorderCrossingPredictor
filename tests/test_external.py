"""Tests for the weather, school break and sports event features."""
from datetime import date

import numpy as np
import pandas as pd

from border_predictor import events, weather
from border_predictor.school_breaks import bc_breaks, school_breaks, wa_breaks


def test_school_breaks_match_published_calendars():
    # Surrey 2024-25: spring break Mar 17-28, back after Labour Day; Seattle 2025: spring break Apr 7-11
    assert bc_breaks(2025)[0] == (date(2025, 3, 15), date(2025, 3, 30))
    assert wa_breaks(2025)[1] == (date(2025, 4, 5), date(2025, 4, 13))
    assert school_breaks(date(2025, 3, 20)) == (1, 0)
    assert school_breaks(date(2025, 4, 9)) == (0, 1)
    assert school_breaks(date(2025, 2, 18)) == (0, 1)  # WA mid-winter break, Presidents' Day week
    assert school_breaks(date(2025, 9, 1)) == (1, 1)  # Labour Day
    assert school_breaks(date(2025, 9, 2)) == (0, 0)
    assert school_breaks(date(2026, 1, 2)) == (1, 1)  # previous year's winter break
    assert school_breaks(date(2026, 1, 5)) == (0, 0)


def test_weather_parse_and_target_features():
    csv = ("latitude,longitude\n49.0,-122.75\n\n"
           "time,temperature_2m (°C),precipitation (mm),snowfall (cm),snow_depth (m)\n"
           "2025-11-02T00:00,5.0,1.0,0.0,0.0\n"
           "2025-11-02T01:00,4.0,2.0,0.5,0.01\n"
           "2025-11-02T01:00,3.0,9.0,9.0,9.0\n"  # repeated hour when clocks fall back
           "2025-11-03T00:00,6.0,0.0,0.0,0.0\n")
    wx = weather.parse(csv)
    assert list(wx.columns) == list(weather.COLUMNS.values())
    assert wx.loc["2025-11-02 01:00", "wx_temp"] == 4.0
    f = weather.target_features(wx, pd.Series(pd.to_datetime(["2025-11-02 01:00", "2025-11-03 00:00",
                                                                "2025-12-01 00:00"])))
    assert f.wx_temp.tolist()[:2] == [4.0, 6.0]
    assert f.wx_precip_day.tolist()[:2] == [3.0, 0.0]
    assert f.wx_snowfall_day[0] == 0.5
    assert f.iloc[2].isna().all()
    assert weather.target_features(None, pd.Series(pd.to_datetime(["2025-11-02"]))).isna().all(axis=None)


def espn_event(event_id, when, city, home, away):
    return {"id": event_id, "date": when, "competitions": [{
        "venue": {"address": {"city": city}},
        "competitors": [{"homeAway": "home", "team": {"abbreviation": home}},
                        {"homeAway": "away", "team": {"abbreviation": away}}]}]}


def test_parse_events_keeps_local_games_in_local_time():
    data = {"events": [espn_event("1", "2025-07-04T20:10Z", "Seattle, Washington", "SEA", "TOR"),
                       espn_event("2", "2025-07-05T20:10Z", "Toronto", "TOR", "SEA")]}
    games = events.parse_events(data, "mlb", "America/Vancouver")
    assert len(games) == 1
    assert games[0]["start"] == pd.Timestamp("2025-07-04 13:10") and games[0]["city"] == "sea"


def test_event_features_stadium_games_and_toronto():
    schedule = pd.DataFrame({
        "id": ["1", "2", "3", "4"],
        "start": pd.to_datetime(["2025-07-04 13:00", "2025-07-04 19:00", "2025-09-14 13:00", "2025-09-21 13:00"]),
        "city": ["sea", "van", "sea", "sea"], "league": ["mlb", "fifa", "nfl", "mlb"],
        "home": ["SEA", "SUI", "SEA", "SEA"], "away": ["TOR", "CAN", "SF", "HOU"],
    })
    targets = pd.to_datetime(["2025-07-04 10:00", "2025-09-14 10:00", "2025-09-14 15:00",
                              "2025-09-14 00:00", "2025-09-21 13:00"])
    f = events.target_features(schedule, targets)
    # Only Seattle stadium games (NFL, World Cup) count; 00:00-01:00 is 12.5 h before kickoff, outside the window
    np.testing.assert_allclose(f.seattle_stadium_game_hours, [np.nan, -2.5, 2.5, np.nan, np.nan])
    assert f.toronto_at_seattle.tolist() == [1, 0, 0, 0, 0]
    empty = events.target_features(None, targets)
    assert empty.seattle_stadium_game_hours.isna().all() and (empty.toronto_at_seattle == 0).all()
