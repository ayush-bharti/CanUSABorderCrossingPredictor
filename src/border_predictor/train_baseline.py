"""Fit and save the baseline predictor for each Cascade lane.

Settings are chosen on the validation period (see evaluate.tune), then the
model is refit on all data from the start of training to the latest download.
"""
from typing import Dict, Optional, Tuple
import argparse
from pathlib import Path

from border_predictor.baseline import BaselinePredictor
from border_predictor.cascade import hourly_delay, load_lane
from border_predictor.config.config import BASELINE_CONFIG, CASCADE_CONFIG, EVALUATION_CONFIG
from border_predictor.evaluate import live_readings, tune


def model_path(lane_id: int, model_dir: Optional[Path] = None) -> Path:
    """Where the fitted model for a lane is stored (default: the configured directory)."""
    return Path(model_dir or BASELINE_CONFIG["model_dir"]) / f"baseline_lane{lane_id}.json"


def train_lane(lane_id: int) -> Tuple[BaselinePredictor, Dict]:
    """
    Tune, refit on all data since training start, and save one lane's model.

    Args:
        lane_id: Cascade crossing lane ID.

    Returns:
        The fitted model and the chosen fit settings.
    """
    lane = load_lane(lane_id)
    hourly = hourly_delay(lane)
    settings = tune(hourly, live_readings(lane), EVALUATION_CONFIG)
    model = BaselinePredictor.fit(hourly[EVALUATION_CONFIG["train_start"]:], **settings)
    model.save(model_path(lane_id))
    print(f"{CASCADE_CONFIG['lanes'][lane_id]} (lane {lane_id}): {settings} -> {model_path(lane_id)}")
    return model, settings


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lanes", type=int, nargs="*", default=list(CASCADE_CONFIG["lanes"]))
    args = parser.parse_args()
    for lane_id in args.lanes:
        train_lane(lane_id)


if __name__ == "__main__":
    main()
