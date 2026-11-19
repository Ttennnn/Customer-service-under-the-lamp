"""Thin ``scripts/`` wrappers must import without executing ``main``."""

import importlib

import pytest

WRAPPERS = (
    "scripts.calibrate_confidence",
    "scripts.cost_by_intent",
    "scripts.eval_flywheel",
    "scripts.flywheel_pipeline",
)


@pytest.mark.parametrize("module_name", WRAPPERS)
def test_importing_wrapper_does_not_run_main(module_name):
    # If the guard were missing, importing would call main() with pytest's
    # argv and argparse would raise SystemExit, failing this import.
    module = importlib.import_module(module_name)
    assert callable(module.main)
