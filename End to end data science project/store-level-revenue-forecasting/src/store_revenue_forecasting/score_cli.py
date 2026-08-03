from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config, resolve_path
from .scoring import score_future_plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score one future store date using actual history and an operating plan."
    )
    parser.add_argument("--config", default="config.yaml", help="Pipeline YAML configuration.")
    parser.add_argument("--future", required=True, help="Future operating-plan CSV.")
    parser.add_argument("--history", help="Historical sales CSV; defaults to data.train_path.")
    parser.add_argument("--stores", help="Store metadata CSV; defaults to data.store_path.")
    parser.add_argument("--model", help="Joblib model; defaults to paths.models.")
    parser.add_argument("--output", help="Prediction CSV; defaults to paths.processed.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config, project_root = load_config(Path(args.config))
    history_path = resolve_path(project_root, args.history or config["data"]["train_path"])
    store_path = resolve_path(project_root, args.stores or config["data"]["store_path"])
    model_path = resolve_path(
        project_root,
        args.model or Path(config["paths"]["models"]) / "forecasting_pipeline.joblib",
    )
    output_path = resolve_path(
        project_root,
        args.output or Path(config["paths"]["processed"]) / "future_predictions.csv",
    )
    future_path = resolve_path(project_root, args.future)
    lag_windows = [int(value) for value in config["features"]["lag_windows"]]

    scored = score_future_plan(
        model_path=model_path,
        history_path=history_path,
        future_path=future_path,
        store_path=store_path,
        lag_windows=lag_windows,
        output_path=output_path,
    )
    print(
        f"Scored {len(scored):,} stores for {scored['Date'].iloc[0].date()} "
        f"and wrote {output_path}"
    )


if __name__ == "__main__":
    main()
