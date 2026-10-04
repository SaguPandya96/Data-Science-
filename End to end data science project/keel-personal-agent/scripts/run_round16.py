"""Round 16 of the memory benchmark: an int8 e5-large-v2.

Downloads the int8 copy of e5-large-v2 (337 MB, pinned by SHA-256) on first use, chooses
its weight on question sets already seen, then tests it once on the tenth held-out set if
it was within half a point of the full-precision model there. Writes
reports/metrics/round16.json and round16.md, and machine-dependent timings to
reports/metrics/round16_timing.json.

    python scripts/run_round16.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from keel.evaluation.round11 import run_round11, to_markdown

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "eval.toml"))
    parser.add_argument("--out", default=str(ROOT / "reports" / "metrics"))
    args = parser.parse_args()

    results, timings = run_round11(args.config, section="round16")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "round16.json").write_text(json.dumps(results, indent=2) + "\n")
    (out / "round16.md").write_text(to_markdown(results, section="round16"))
    (out / "round16_timing.json").write_text(json.dumps(timings, indent=2) + "\n")
    print(to_markdown(results, timings, section="round16"))


if __name__ == "__main__":
    main()
