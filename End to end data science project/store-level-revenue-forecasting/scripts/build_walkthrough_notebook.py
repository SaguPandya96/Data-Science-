from __future__ import annotations

import argparse
import json
from pathlib import Path


def _source(text: str) -> list[str]:
    return text.strip("\n").splitlines(keepends=True)


def _markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _source(text)}


def _code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source(text),
    }


def build_notebook() -> dict:
    """Return a deterministic, package-driven walkthrough notebook."""
    cells = [
        _markdown(
            """
# Store revenue forecasting: package walkthrough

This short notebook exercises the same package, configuration, artifact, and scoring contracts used by the command-line workflow. It uses deterministic synthetic data so it runs quickly and never presents demo metrics as business results.

The original portfolio notebook remains available as `Store_Level_Revenue_Forecasting_and_Scenario_Planning.ipynb`. This walkthrough is the clean operational companion.
"""
        ),
        _markdown(
            """
## The forecast contract

The model predicts one store for one future day. Before scoring day `t`, actual sales through day `t-1` must be available. Rolling demand features apply `shift(1)`, customer count is excluded, and multi-day forecasts require a separate recursive or direct-horizon design.
"""
        ),
        _code(
            """
from pathlib import Path
import tempfile

import pandas as pd
import yaml
from IPython.display import Image, display

from store_revenue_forecasting.pipeline import run_pipeline
from store_revenue_forecasting.scoring import score_future_plan

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

PROJECT_ROOT
"""
        ),
        _markdown(
            """
## Run the complete sample pipeline

Outputs go to a temporary directory, so executing this notebook does not overwrite the checked-in full-data figures or local model artifacts.
"""
        ),
        _code(
            """
temporary = tempfile.TemporaryDirectory()
run_root = Path(temporary.name)

config = yaml.safe_load((PROJECT_ROOT / "config.sample.yaml").read_text(encoding="utf-8"))
config["project_root"] = str(PROJECT_ROOT)
config["paths"] = {
    "processed": str(run_root / "processed"),
    "models": str(run_root / "models"),
    "metrics": str(run_root / "metrics"),
    "tables": str(run_root / "tables"),
    "figures": str(run_root / "figures"),
}
temporary_config = run_root / "config.yaml"
temporary_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

artifacts = run_pipeline(temporary_config)
sorted(artifacts)
"""
        ),
        _markdown(
            """
## Inspect measured outputs

The sample exists to prove that the pipeline works. The README reports the separate full Rossmann run used for portfolio conclusions.
"""
        ),
        _code(
            """
model_performance = pd.read_csv(artifacts["model_performance"])
scenario_summary = pd.read_csv(artifacts["scenario_summary"])

display(model_performance.round(4))
display(scenario_summary.round(4))
display(Image(filename=str(artifacts["model_comparison_figure"])))
display(Image(filename=str(artifacts["scenario_impact_figure"])))
"""
        ),
        _markdown(
            """
## Score the next operating day

The scoring interface combines the saved model, actual history, store metadata, and one future operating-plan CSV. It rejects multiple forecast dates because version 1 is intentionally one-day-ahead.
"""
        ),
        _code(
            """
scored = score_future_plan(
    model_path=artifacts["model"],
    history_path=PROJECT_ROOT / "data/sample/train.csv",
    future_path=PROJECT_ROOT / "data/sample/future.csv",
    store_path=PROJECT_ROOT / "data/sample/store.csv",
    lag_windows=config["features"]["lag_windows"],
    output_path=run_root / "processed/future_predictions.csv",
)

display(scored)
temporary.cleanup()
"""
        ),
        _markdown(
            """
## What this demonstration does not prove

- Synthetic data does not establish forecast quality.
- Promotion scenarios are model sensitivities, not causal ROI.
- A rolling one-day-ahead holdout is not a one-shot six-week forecast.
- Production use still needs an approved source, scheduled monitoring, prediction intervals, and a rollback owner.

Those boundaries are part of the project, not footnotes added after the model was built.
"""
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def render_notebook() -> str:
    return json.dumps(build_notebook(), ensure_ascii=False, indent=1) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the package walkthrough notebook.")
    parser.add_argument("--check", action="store_true", help="Fail when the notebook is stale.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    destination = project_root / "notebooks" / "01_package_walkthrough.ipynb"
    expected = render_notebook()
    if args.check:
        if not destination.exists() or destination.read_text(encoding="utf-8") != expected:
            raise SystemExit(
                "notebooks/01_package_walkthrough.ipynb is stale; "
                "run python scripts/build_walkthrough_notebook.py"
            )
        print("Walkthrough notebook is up to date.")
        return

    destination.write_text(expected, encoding="utf-8")
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
