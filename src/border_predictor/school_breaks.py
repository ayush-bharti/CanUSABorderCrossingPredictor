"""Approximate school breaks in the Lower Mainland (BC) and western Washington.

Districts publish their calendars only a year or two ahead, so the breaks are
rules that match recent Surrey/Vancouver and Seattle/Bellingham calendars to
within a few days. Each break runs from the Saturday it starts to the Sunday
before school resumes, so the weekends around it count as part of it.
"""
from datetime import date, timedelta
from functools import lru_cache
from typing import List, Tuple

Break = Tuple[date, date]  # first and last day, inclusive


def nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The nth (1-based) given weekday (Monday = 0) of a month."""
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def on_or_after(day: date, weekday: int) -> date:
    """The first given weekday (Monday = 0) on or after day."""
    return day + timedelta(days=(weekday - day.weekday()) % 7)


def last_weekday(year: int, month: int, weekday: int) -> date:
    """The last given weekday (Monday = 0) of a month."""
    next_month = date(year + month // 12, month % 12 + 1, 1)
    return on_or_after(next_month - timedelta(days=7), weekday)


def winter_break(year: int) -> Break:
    """December holidays starting in year: Saturday on or after Dec 18 to the first Monday on or after Jan 2."""
    return on_or_after(date(year, 12, 18), 5), on_or_after(date(year + 1, 1, 2), 0) - timedelta(days=1)


def labour_day(year: int) -> date:
    """First Monday of September; the last day of summer break in both places."""
    return nth_weekday(year, 9, 0, 1)


@lru_cache(maxsize=None)
def bc_breaks(year: int) -> List[Break]:
    """BC: two-week spring break from the third Monday of March, summer from late June, winter break."""
    spring = nth_weekday(year, 3, 0, 3)
    return [
        (spring - timedelta(days=2), spring + timedelta(days=13)),
        (last_weekday(year, 6, 5), labour_day(year)),
        winter_break(year),
    ]


@lru_cache(maxsize=None)
def wa_breaks(year: int) -> List[Break]:
    """WA: mid-winter break in Presidents' Day week, spring break the week of the first Monday from April 3, summer from mid-June, winter break."""
    presidents = nth_weekday(year, 2, 0, 3)
    spring = on_or_after(date(year, 4, 3), 0)
    return [
        (presidents - timedelta(days=2), presidents + timedelta(days=6)),
        (spring - timedelta(days=2), spring + timedelta(days=6)),
        (on_or_after(date(year, 6, 15), 5), labour_day(year)),
        winter_break(year),
    ]


def in_break(day: date, breaks) -> bool:
    """Whether day falls in any break of its year or the previous year's winter break."""
    return any(start <= day <= end for year in (day.year - 1, day.year) for start, end in breaks(year))


def school_breaks(day: date) -> Tuple[int, int]:
    """(BC, WA) flags: 1 if schools there are on a break that day."""
    return int(in_break(day, bc_breaks)), int(in_break(day, wa_breaks))
