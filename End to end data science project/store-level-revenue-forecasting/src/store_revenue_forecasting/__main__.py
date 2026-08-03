from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a rolling store-sales forecast and run planning scenarios."
    )
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML configuration.")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Refresh configured source files even when local copies exist.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifacts = run_pipeline(Path(args.config), force_download=args.force_download)
    print("Pipeline completed. Generated artifacts:")
    for name, path in artifacts.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
