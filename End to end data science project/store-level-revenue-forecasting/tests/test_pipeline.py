from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from store_revenue_forecasting.pipeline import run_pipeline


def test_end_to_end_pipeline_writes_expected_artifacts(tmp_path: Path, rossmann_frames) -> None:
    sales, stores = rossmann_frames
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    sales.to_csv(data_dir / "train.csv", index=False)
    stores.to_csv(data_dir / "store.csv", index=False)

    config = {
        "project_root": ".",
        "seed": 42,
        "data": {
            "train_path": "data/train.csv",
            "store_path": "data/store.csv",
            "download_if_missing": False,
        },
        "validation": {"holdout_weeks": 2},
        "features": {"lag_windows": [7, 30]},
        "model": {
            "kind": "random_forest",
            "name": "Random Forest",
            "params": {"n_estimators": 10, "max_depth": 5, "n_jobs": 1},
        },
        "scenarios": [
            {"name": "Baseline", "kind": "baseline"},
            {"name": "No Promotions", "kind": "set_promotions", "value": 0},
        ],
        "paths": {
            "processed": "outputs/processed",
            "models": "outputs/models",
            "metrics": "outputs/metrics",
            "tables": "outputs/tables",
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    artifacts = run_pipeline(config_path)

    assert all(path.exists() for path in artifacts.values())
    performance = pd.read_csv(artifacts["model_performance"])
    scenarios = pd.read_csv(artifacts["scenario_summary"])
    assert performance["Model"].tolist() == [
        "Naive recent-demand average",
        "Linear Regression",
        "Random Forest",
    ]
    assert scenarios["Scenario"].tolist() == ["Baseline", "No Promotions"]
