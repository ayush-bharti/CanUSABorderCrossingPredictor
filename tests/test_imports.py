"""Every module imports cleanly."""
import importlib

import pytest

MODULES = ["app", "baseline", "cascade", "collector", "evaluate", "gbm", "predict", "service", "train_baseline"]


@pytest.mark.parametrize("name", MODULES)
def test_import(name):
    importlib.import_module(f"border_predictor.{name}")
