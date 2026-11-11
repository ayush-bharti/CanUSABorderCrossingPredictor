"""Refresh the data and retrain every lane, keeping a lane's new models only when they pass checks.

Meant to run monthly (see deploy/): downloads the latest full months of Cascade
data, the weather history and the game schedules, then trains every lane in
parallel processes into models/candidate/. A lane's candidate replaces its live
models when its validation MAE is no worse than theirs (within a tolerance) and
it forecasts finite delays from the latest readings. The replaced models move to
models/previous/, and `rollback` swaps them back. The web app picks up new
models without a restart.

The validation period is fixed (EVALUATION_CONFIG), so the MAE check catches a
broken retrain rather than measuring the new months; tracking.py measures how
the live models do on hours they have not seen.
"""
from typing import Dict, List, Optional, Tuple
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from border_predictor import cascade, events, gbm, weather
from border_predictor.baseline import BaselinePredictor
from border_predictor.config.config import CASCADE_CONFIG, REFRESH_CONFIG
from border_predictor.train_baseline import model_path


def download_data(config: Dict = REFRESH_CONFIG) -> None:
    """
    Fetch the Cascade months completed since the last run, the weather history and the game schedules.

    A failed weather or schedule download keeps the saved file and only warns:
    the retrain then uses the data already on disk.
    """
    end = cascade.last_full_month(date.today())
    start = pd.Timestamp(end) - pd.DateOffset(months=config["cascade_months_back"] - 1)
    cascade.download(start.date(), end)
    try:
        hours = weather.download_history()
        print(f"Weather: {len(hours)} hours to {hours.index[-1]}", flush=True)
    except (requests.RequestException, ValueError) as e:
        print(f"Weather download failed, keeping the saved history: {e}", flush=True)
    try:
        games = events.download_schedule()
        print(f"Games: {len(games)} to {games.start.max()}", flush=True)
    except RuntimeError as e:
        print(f"Game schedule download failed: {e}", flush=True)


def train_candidates(config: Dict = REFRESH_CONFIG) -> List[int]:
    """
    Train every lane into the candidate directory, one process per lane group, all at once.

    Returns:
        The lanes whose training process failed. Each group's output is in
        candidate_dir/train_<lanes>.log.
    """
    candidate = Path(config["candidate_dir"])
    shutil.rmtree(candidate, ignore_errors=True)
    candidate.mkdir(parents=True)
    env = dict(os.environ, BORDER_PREDICTOR_MODELS_DIR=str(candidate), OMP_NUM_THREADS=str(config["threads_per_job"]))
    jobs = []
    for group in config["lane_groups"]:
        log = open(candidate / f"train_{'_'.join(map(str, group))}.log", "w")
        command = [sys.executable, "-u", "-m", "border_predictor.gbm", "train", "--lanes", *map(str, group)]
        jobs.append((group, subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT), log))
    failed = []
    for group, process, log in jobs:
        if process.wait():
            failed += group
        log.close()
    return failed


def validation_mae(lane_id: int, model_dir: Path) -> Optional[float]:
    """The validation MAE saved with a lane's models, or None for models saved without it."""
    path = gbm.meta_path(lane_id, model_dir)
    return json.loads(path.read_text()).get("validation_mae") if path.exists() else None


def recent_readings(lane_id: int, days: int = 9) -> pd.DataFrame:
    """The last days of downloaded readings for a lane, as cascade.fetch_recent returns them."""
    lane = cascade.through_last_reading(cascade.load_lane(lane_id, last_months=2))
    return lane[lane.index > lane.index[-1] - pd.Timedelta(days=days)]


def forecast_check(lane_id: int, model_dir: Path) -> Optional[str]:
    """Forecast every horizon from the latest downloaded readings; a problem found, or None."""
    models = gbm.load_models(lane_id, model_dir)
    if models is None:
        return "no median models"
    baseline = BaselinePredictor.load(model_path(lane_id, model_dir))
    recent = recent_readings(lane_id)
    others = {other: recent_readings(other) for other in gbm.other_lanes_needed(models)}
    origin = gbm.forecast_origin(recent)
    targets = [origin + pd.Timedelta(hours=h) for h in range(gbm.GBM_CONFIG["bands"][-1]["up_to"] + 1)]
    predicted = gbm.forecast(recent, baseline, models, targets, others)
    if len(predicted) != len(targets):
        return f"forecast {len(predicted)} of {len(targets)} hours"
    values = predicted[["gbm", "low", "high"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        return "non-finite forecast"
    return None


def check_lane(lane_id: int, config: Dict = REFRESH_CONFIG) -> Tuple[bool, str]:
    """Whether a lane's candidate models should replace its live ones, and why."""
    candidate, live = Path(config["candidate_dir"]), Path(config["live_dir"])
    new_mae, old_mae = validation_mae(lane_id, candidate), validation_mae(lane_id, live)
    if new_mae is None:
        return False, "candidate has no validation MAE"
    problem = forecast_check(lane_id, candidate)
    if problem:
        return False, problem
    if old_mae is None:
        return True, f"validation MAE {new_mae:.3f} (live models have none saved)"
    change = f"validation MAE {old_mae:.3f} -> {new_mae:.3f}"
    if new_mae > old_mae * (1 + config["mae_tolerance"]):
        return False, f"{change}, worse than the {config['mae_tolerance']:.0%} tolerance"
    return True, change


def lane_files(lane_id: int, model_dir: Path) -> List[Path]:
    """A lane's saved baseline and LightGBM files in a directory, the meta file last."""
    files = [model_path(lane_id, model_dir)] + sorted(Path(model_dir).glob(f"gbm_lane{lane_id}_*.txt"))
    return [p for p in files + [gbm.meta_path(lane_id, model_dir)] if p.exists()]


def move_lane(lane_id: int, source: Path, target: Path) -> None:
    """Move a lane's model files from source to target, replacing any it has there."""
    target.mkdir(parents=True, exist_ok=True)
    for path in lane_files(lane_id, target):
        path.unlink()
    for path in lane_files(lane_id, source):
        os.replace(path, target / path.name)


def swap_lane(lane_id: int, incoming: Path, config: Dict = REFRESH_CONFIG) -> None:
    """
    Make the lane's models in incoming live, keeping the replaced ones in previous_dir.

    The meta file is written last, so the web app (which reloads on file times)
    sees a complete set.
    """
    live, previous = Path(config["live_dir"]), Path(config["previous_dir"])
    scratch = previous.parent / f".swap_lane{lane_id}"
    shutil.rmtree(scratch, ignore_errors=True)
    move_lane(lane_id, live, scratch)
    move_lane(lane_id, incoming, live)
    move_lane(lane_id, scratch, previous)
    scratch.rmdir()


def refresh(config: Dict = REFRESH_CONFIG, download: bool = True, force: bool = False, swap: bool = True) -> bool:
    """
    Download, retrain and swap in each lane that passes its checks (every trained lane with force).
    Without swap, the candidates are only checked and stay in candidate_dir.

    Returns:
        Whether every lane was trained and passed (and was replaced, with swap).
    """
    if download:
        download_data(config)
    failed = train_candidates(config)
    replaced = True
    for lane_id, name in CASCADE_CONFIG["lanes"].items():
        if lane_id in failed:
            ok, reason = False, f"training failed, see {config['candidate_dir']}"
        else:
            ok, reason = check_lane(lane_id, config)
            if force and not ok and gbm.load_models(lane_id, config["candidate_dir"]) is not None:
                ok, reason = True, f"{reason} (forced)"
        if ok and swap:
            swap_lane(lane_id, Path(config["candidate_dir"]), config)
        replaced &= ok
        outcome = ("replaced" if swap else "would replace") if ok else "kept live models"
        print(f"{name} (lane {lane_id}): {outcome}: {reason}", flush=True)
    return replaced


def rollback(config: Dict = REFRESH_CONFIG) -> None:
    """Swap each lane's live models with the ones its last swap replaced; running it again undoes it."""
    for lane_id, name in CASCADE_CONFIG["lanes"].items():
        if lane_files(lane_id, Path(config["previous_dir"])):
            swap_lane(lane_id, Path(config["previous_dir"]), config)
            print(f"{name} (lane {lane_id}): swapped with previous models")


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", nargs="?", choices=["run", "rollback"], default="run")
    parser.add_argument("--skip-download", action="store_true", help="Retrain on the data already on disk")
    parser.add_argument("--force", action="store_true", help="Replace every lane that trained, even if checks fail")
    parser.add_argument("--dry-run", action="store_true", help="Train and check, but leave the live models in place")
    args = parser.parse_args()
    if args.command == "rollback":
        rollback()
        return
    started = pd.Timestamp.now()
    ok = refresh(download=not args.skip_download, force=args.force, swap=not args.dry_run)
    minutes = math.ceil((pd.Timestamp.now() - started).total_seconds() / 60)
    passed = ("all lanes would be replaced" if args.dry_run else "all lanes replaced") if ok else \
        "some lanes kept their live models"
    print(f"Refresh finished in {minutes} min: {passed}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
