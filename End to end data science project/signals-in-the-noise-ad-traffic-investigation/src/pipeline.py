from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.database import build_database, build_feature_table, save_scores
from src.download_data import download_sample
from src.modeling import score_campaign_windows
from src.reporting import generate_reports
from src.simulate_abuse import add_experimental_scenarios, load_observed


def _resolve_data_dir(project_root: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else project_root / candidate


def run(rows: int, data_dir_value: str, force_download: bool = False) -> dict:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = _resolve_data_dir(project_root, data_dir_value)
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    reports_dir = project_root / "reports"
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads((project_root / "config" / "project.json").read_text(encoding="utf-8"))
    raw_path = raw_dir / "criteo_attribution_sample.tsv"
    metadata = download_sample(raw_path, requested_rows=rows, force=force_download)

    print("Reading the observed event sample ...")
    observed = load_observed(raw_path)
    observed_snapshot = observed.copy(deep=True)
    events, manifest = add_experimental_scenarios(observed, seed=int(config["random_seed"]))

    # A direct guard against accidentally changing the source portion during simulation.
    if not observed.equals(observed_snapshot):
        raise AssertionError("The simulation step mutated observed source rows.")
    if int(events["is_simulated_abuse"].sum()) == 0:
        raise AssertionError("No positive experimental rows were created.")

    manifest_path = processed_dir / "injection_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    data_quality = {
        "observed_rows": int(len(observed)),
        "combined_rows": int(len(events)),
        "simulated_rows": int((events["event_origin"] == "simulation").sum()),
        "simulated_abuse_rows": int(events["is_simulated_abuse"].sum()),
        "null_cells": int(events.isna().sum().sum()),
        "timestamp_min": int(events["timestamp"].min()),
        "timestamp_max": int(events["timestamp"].max()),
        "campaigns": int(events["campaign"].nunique()),
        "source_groups": int(events["source_id"].nunique()),
        "observed_rows_mutated": False,
    }
    (reports_dir / "data_quality.json").write_text(
        json.dumps(data_quality, indent=2), encoding="utf-8"
    )

    database_path = data_dir / "traffic.db"
    print(f"Loading {len(events):,} combined events into SQLite ...")
    build_database(events, database_path)

    print("Building campaign-window features with SQL ...")
    features = build_feature_table(
        database_path,
        project_root / "sql" / "01_campaign_window_features.sql",
        processed_dir / "campaign_window_features.csv",
    )

    print("Fitting the anomaly baseline and logistic model ...")
    result = score_campaign_windows(features, config, processed_dir)
    save_scores(database_path, result["scored"])

    print("Writing the investigation report, dashboard, and case notes ...")
    generate_reports(reports_dir, metadata, manifest, features, result)

    summary = {
        "source_rows": metadata["sample_rows"],
        "combined_events": len(events),
        "campaign_windows": len(features),
        "review_queue_windows": len(result["review_queue"]),
        "test_metrics": result["metrics"]["hybrid_test"],
        "reports_directory": "reports",
        "database": "<data-dir>/traffic.db",
    }
    (reports_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Signals in the Noise investigation.")
    parser.add_argument(
        "--rows",
        type=int,
        default=500_000,
        help="Number of leading timestamp-sorted source rows; use 0 for the complete dataset.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Runtime data directory. Relative paths are resolved from the project root.",
    )
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()
    if args.rows < 0:
        parser.error("--rows must be zero or a positive integer")
    run(args.rows, args.data_dir, force_download=args.force_download)


if __name__ == "__main__":
    main()
