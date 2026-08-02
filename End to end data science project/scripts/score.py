"""Validate and score a shipment-level CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from supplylens.config import load_config, resolve_path
from supplylens.modeling import load_bundle
from supplylens.scoring import score_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Shipment-level input CSV")
    parser.add_argument("--output", required=True, type=Path, help="Destination scored CSV")
    parser.add_argument("--model", type=Path, default=None, help="Optional model bundle path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Scoring input not found: {args.input}")
    config = load_config()
    model_path = args.model or resolve_path(config["outputs"]["model_path"])
    frame = pd.read_csv(args.input)
    print(f"Loaded {len(frame):,} rows from {args.input}")
    bundle = load_bundle(model_path)
    scored = score_frame(bundle, frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output, index=False)
    print(
        f"Validation passed; scored {len(scored):,} rows and flagged "
        f"{int(scored['review_flag'].sum()):,} for review."
    )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Scoring failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

