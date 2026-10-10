"""Tests for the CBP wait time collector."""
import csv
from datetime import datetime, timezone

from border_predictor.collector import extract_readings, parse_reported_time, save_readings


def _port(update_time="At 1:00 pm PDT", date="10/10/2026", time="13:35:36", delay="15"):
    lane = {"update_time": update_time, "operational_status": "delay",
            "delay_minutes": delay, "lanes_open": "4"}
    closed = {"update_time": "", "operational_status": "Lanes Closed",
              "delay_minutes": "", "lanes_open": ""}
    return {
        "port_number": "300402", "port_name": "Blaine", "crossing_name": "Peace Arch",
        "date": date, "time": time, "port_status": "Open",
        "passenger_vehicle_lanes": {"maximum_lanes": "7", "standard_lanes": lane,
                                    "ready_lanes": closed},
    }


FETCHED = datetime(2026, 10, 10, 20, 35, tzinfo=timezone.utc)


def test_parse_reported_time_converts_to_utc():
    assert parse_reported_time("At 1:00 pm PDT", "10/10/2026", "13:35:36") == \
        datetime(2026, 10, 10, 20, 0, tzinfo=timezone.utc)


def test_parse_reported_time_rolls_back_across_midnight():
    assert parse_reported_time("At 11:00 pm PST", "01/02/2026", "00:10:00") == \
        datetime(2026, 1, 2, 7, 0, tzinfo=timezone.utc)


def test_parse_reported_time_handles_noon_and_garbage():
    assert parse_reported_time("At Noon PDT", "10/10/2026", "13:00:00") == \
        datetime(2026, 10, 10, 19, 0, tzinfo=timezone.utc)
    assert parse_reported_time("", "10/10/2026", "13:00:00") is None


def test_extract_readings_skips_closed_lanes_and_other_ports():
    other = dict(_port(), port_number="999999")
    readings = extract_readings([_port(), other], ["300402"], FETCHED)

    assert len(readings) == 1
    assert readings[0]["lane"] == "passenger_standard"
    assert readings[0]["delay_minutes"] == "15"


def test_save_readings_deduplicates_on_reported_time(tmp_path):
    first = extract_readings([_port()], ["300402"], FETCHED)
    same_update_later_fetch = extract_readings(
        [_port(time="13:50:00")], ["300402"], datetime(2026, 10, 10, 20, 50, tzinfo=timezone.utc)
    )
    next_update = extract_readings([_port(update_time="At 2:00 pm PDT", time="14:05:00", delay="20")],
                                   ["300402"], FETCHED)

    assert save_readings(first, tmp_path) == 1
    assert save_readings(same_update_later_fetch, tmp_path) == 0
    assert save_readings(next_update, tmp_path) == 1

    with (tmp_path / "2026-10.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [r["delay_minutes"] for r in rows] == ["15", "20"]
