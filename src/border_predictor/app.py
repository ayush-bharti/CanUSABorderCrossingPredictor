"""Flask web application: compare Peace Arch and Pacific Highway delays now and ahead."""
from typing import Dict, List
from datetime import datetime, timedelta
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request

from border_predictor import tracking
from border_predictor.config.config import WEB_CONFIG
from border_predictor.service import Forecaster, Lane, lanes, local_now, model_version

app = Flask(__name__,
            template_folder=Path(__file__).parent.parent.parent / "templates",
            static_folder=Path(__file__).parent.parent.parent / "static")

forecaster = Forecaster()

DIRECTIONS = {"south": "Southbound into the US", "north": "Northbound into Canada"}
KINDS = {"cars": "Cars", "nexus": "NEXUS"}
DAY_TYPE_LABELS = {
    "holiday_ca": "Canadian holiday", "holiday_us": "US holiday", "holiday_both": "Holiday in both countries",
    "window_ca": "Canadian long weekend", "window_us": "US long weekend", "window_both": "Long weekend in both countries",
}
FORECAST_HOURS = 12
PLAN_DAYS = 14
ACCURACY_WINDOWS = [7, 30, 90]


def lanes_for(direction: str, kind: str) -> List[Lane]:
    """The two crossings' lanes for a direction and lane type, or 404."""
    found = [lane for lane in lanes() if lane.direction == direction and lane.kind == kind]
    if not found:
        abort(404)
    return found


def serialize(lane: Lane, result: Dict) -> Dict:
    """JSON-friendly forecast for one lane."""
    latest = result["latest"]
    return {
        "lane_id": lane.lane_id,
        "crossing": lane.crossing,
        "latest": latest and {"time": latest["time"].strftime("%Y-%m-%d %H:%M"), "delay": round(latest["delay"])},
        "hours": [{
            "time": h["time"].strftime("%Y-%m-%d %H:%M"),
            "label": h["time"].strftime("%a %H:%M"),
            "delay": round(h["delay"], 1),
            "low": None if h["low"] is None else round(h["low"], 1),
            "high": None if h["high"] is None else round(h["high"], 1),
            "spike_chance": None if h["spike_chance"] is None else round(h["spike_chance"], 2),
            "spike_alert": h["spike_alert"],
            "source": h["source"],
            "day_type": DAY_TYPE_LABELS.get(h["day_type"]),
        } for h in result["hours"]],
    }


@app.route("/")
def index():
    """Crossing chooser: live comparison plus trip planning."""
    now = local_now()
    return render_template(
        "index.html",
        directions=DIRECTIONS,
        kinds=KINDS,
        date_options=[(now + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(PLAN_DAYS)],
        hour_options=[f"{h:02d}:00" for h in range(24)],
        current_hour=f"{now.hour:02d}:00",
    )


@app.route("/api/forecast")
def api_forecast():
    """Next hours for both crossings, from the next hour boundary on."""
    direction, kind = request.args.get("direction", "south"), request.args.get("kind", "cars")
    hours = min(int(request.args.get("hours", FORECAST_HOURS)), 48)
    start = local_now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    targets = [start + timedelta(hours=i) for i in range(hours)]
    return jsonify({
        "direction": DIRECTIONS.get(direction), "kind": KINDS.get(kind),
        "crossings": [serialize(lane, forecaster.forecast(lane.lane_id, targets))
                      for lane in lanes_for(direction, kind)],
    })


@app.route("/api/plan")
def api_plan():
    """Both crossings for a chosen hour, with the hours either side to show when it eases."""
    direction, kind = request.args.get("direction", "south"), request.args.get("kind", "cars")
    try:
        at = datetime.strptime(request.args["at"], "%Y-%m-%d %H:%M")
    except (KeyError, ValueError):
        return jsonify({"error": 'Give "at" as "YYYY-MM-DD HH:MM"'}), 400
    targets = [at + timedelta(hours=i) for i in range(-2, 3)]
    return jsonify({
        "at": at.strftime("%Y-%m-%d %H:%M"),
        "crossings": [serialize(lane, forecaster.forecast(lane.lane_id, targets))
                      for lane in lanes_for(direction, kind)],
    })


@app.route("/visualize")
def visualize():
    """Typical delay by weekday and hour for each lane."""
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    charts, error = [], None
    try:
        for lane in lanes():
            week = forecaster.typical_week(lane.lane_id)
            charts.append({
                "id": f"lane-{lane.lane_id}",
                "title": f"{lane.crossing} — {DIRECTIONS[lane.direction]}, {KINDS[lane.kind]}",
                "z": week.round(1).values.tolist(),
                "x": [f"{h:02d}:00" for h in week.columns],
                "y": [days[d] for d in week.index],
            })
    except FileNotFoundError as e:
        error = str(e)
    return render_template("visualize.html", charts=charts, error=error)


def number(value, digits: int = 1):
    """A rounded float for JSON and templates, None for missing values."""
    return None if value is None or value != value else round(float(value), digits)


@app.route("/accuracy")
def accuracy():
    """How the logged live forecasts compare with what happened, by lane and horizon."""
    days = request.args.get("days", 30, type=int)
    days = days if days in ACCURACY_WINDOWS else 30
    rows = tracking.recent_scores(days)
    scores, daily = tracking.score(rows), tracking.daily_mae(rows)
    cards = []
    for lane in lanes():
        title = f"{lane.crossing} — {DIRECTIONS[lane.direction]}, {KINDS[lane.kind]}"
        lane_scores = scores.xs(lane.lane_id, level="lane") if lane.lane_id in scores.index.get_level_values(0) else None
        lane_daily = daily[daily.lane == lane.lane_id]
        cards.append({
            "id": f"lane-{lane.lane_id}",
            "title": title,
            "trained": model_version(lane.lane_id),
            "horizons": [] if lane_scores is None else [{
                "horizon": int(h), "n": int(r.n), "mae": number(r.mae), "baseline_mae": number(r.baseline_mae),
                "coverage": number(r.coverage * 100, 0), "spikes": int(r.spikes),
                "alerts": None if r.alerts != r.alerts else int(r.alerts),
                "hits": None if r.hits != r.hits else int(r.hits),
            } for h, r in lane_scores.iterrows()],
            "daily": {"x": [day.strftime("%Y-%m-%d") for day in lane_daily.day],
                      "mae": [number(v, 2) for v in lane_daily.mae],
                      "baseline": [number(v, 2) for v in lane_daily.baseline_mae]},
        })
    return render_template("accuracy.html", cards=cards, days=days, windows=ACCURACY_WINDOWS, scored=len(rows))


@app.route("/about")
def about():
    """Render the about page."""
    return render_template("about.html")


def main():
    """Run the Flask application."""
    app.run(host=WEB_CONFIG["host"], port=WEB_CONFIG["port"], debug=WEB_CONFIG["debug"])


if __name__ == "__main__":
    main()
