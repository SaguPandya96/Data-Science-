"""Check that freshly regenerated metrics match the committed ones.

Every decision, label and count must match exactly. Rates and intervals may differ by at
most ``--tolerance`` (default 0.005, half a percentage point). Transformer encoders
run through ONNX Runtime, whose CPU kernels differ between processor types, so the last
bits of a vector can change from one CI runner to the next; very rarely that flips a single
near-tie, which moves a rate by about 0.0002.

    python scripts/check_metrics.py reports/metrics/round6.json ...
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


def differences(old: Any, new: Any, tolerance: float, path: str = "") -> list[str]:
    if isinstance(old, dict) and isinstance(new, dict):
        if old.keys() != new.keys():
            return [f"{path}: keys {sorted(old)} != {sorted(new)}"]
        return [
            d for key in old for d in differences(old[key], new[key], tolerance, f"{path}.{key}")
        ]
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            return [f"{path}: length {len(old)} != {len(new)}"]
        return [
            d
            for i, (a, b) in enumerate(zip(old, new, strict=True))
            for d in differences(a, b, tolerance, f"{path}[{i}]")
        ]
    is_rate = isinstance(old, float) and isinstance(new, float | int) and not isinstance(new, bool)
    if is_rate and math.isclose(old, new, rel_tol=tolerance, abs_tol=tolerance):
        return []
    if old == new and type(old) is type(new):
        return []
    return [f"{path}: {old!r} != {new!r}"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--tolerance", type=float, default=0.002)
    args = parser.parse_args()

    failed = False
    for file in args.files:
        committed = subprocess.run(
            ["git", "show", f"HEAD:./{file.as_posix()}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        found = differences(json.loads(committed), json.loads(file.read_text()), args.tolerance)
        for line in found:
            print(f"{file}{line}")
        failed = failed or bool(found)
        if not found:
            print(f"{file}: matches")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
