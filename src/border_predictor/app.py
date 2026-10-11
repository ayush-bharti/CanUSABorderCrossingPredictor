"""Flask web application: compare Peace Arch and Pacific Highway delays now and ahead."""
from typing import Dict, List
from datetime import datetime, timedelta
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request

from border_predictor.config.config import WEB_CONFIG
from border_predictor.service import Forecaster, Lane, lanes, local_now

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


@app.route("/about")
def about():
    """Render the about page."""
    return render_template("about.html")


def main():
    """Run the Flask application."""
    app.run(host=WEB_CONFIG["host"], port=WEB_CONFIG["port"], debug=WEB_CONFIG["debug"])


if __name__ == "__main__":
    main()
