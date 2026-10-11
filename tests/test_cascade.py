"""Tests for Cascade Gateway parsing and live fetch handling (no network)."""
from datetime import date, datetime
from unittest import mock

import pandas as pd

from border_predictor import cascade

CSV = """Group Starts,Avg - Delay (Peace Arch South Cars),Avg - Volume (Peace Arch South Cars),Avg - Availability (Peace Arch South Cars)
2026-10-10 20:45:00,33,19,1
2026-10-10 20:50:00,31,17,1
2026-10-10 20:55:00,,,0
"""


def test_parse_csv():
    df = cascade.parse_csv(CSV)
    assert list(df.columns) == ["delay", "vol", "avail"]
    assert df.index[0] == pd.Timestamp("2026-10-10 20:45")
    assert df.loc["2026-10-10 20:50", "delay"] == 31


def test_fetch_recent_drops_polls_not_yet_taken():
    with mock.patch.object(cascade, "load_api_key", return_value="k"), \
         mock.patch.object(cascade, "fetch_range", return_value=CSV) as fetch:
        df = cascade.fetch_recent(4, days=9)
    assert df.index[-1] == pd.Timestamp("2026-10-10 20:50")
    start, end = fetch.call_args.args[1:3]
    assert (end - start).days == 8


def test_fetch_current_delay_reads_epoch_as_local_time():
    response = mock.Mock(status_code=200)
    # 2026-10-10 17:10 written as if it were UTC
    response.json.return_value = {"GroupStarts": [1791652200], "Values": [[2.0]]}
    with mock.patch.object(cascade, "load_api_key", return_value="k"), \
         mock.patch.object(cascade.requests, "get", return_value=response):
        assert cascade.fetch_current_delay(4) == (datetime(2026, 10, 10, 17, 10), 2.0)
