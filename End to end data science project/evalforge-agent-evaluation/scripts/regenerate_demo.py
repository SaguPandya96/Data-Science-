"""Regenerate every demonstration artifact from scratch.

Deletes the generated suite, stored runs and reports, then rebuilds them with fixed
seeds. Because every stochastic decision in EvalForge is seeded on its own coordinates,
the rebuilt artifacts are byte-identical to the committed ones — which is what makes
"regenerate and diff" a meaningful check that a change did what it claimed.

Usage::

    python scripts/regenerate_demo.py
    python scripts/regenerate_demo.py --count 40 --skip-annotations
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evalforge.config import load_config  # noqa: E402


def _run(command: list[str]) -> int:
    """Run a subcommand, echoing it first."""
    print(f"\n$ {' '.join(command)}")
    # Fixed argv derived from this script's own arguments; no shell involved.
    return subprocess.run(command, check=False).returncode


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--count", type=int, default=150, help="Scenarios to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Master seed.")
    parser.add_argument(
        "--skip-annotations",
        action="store_true",
        help="Do not generate synthetic annotations.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not delete existing artifacts first.",
    )
    args = parser.parse_args()

    config = load_config()

    if not args.keep_existing:
        print("Removing existing demonstration artifacts...")
        for path in (config.paths.runs, config.paths.generated_scenarios):
            if path.exists():
                shutil.rmtree(path)
                print(f"  removed {path}")
        for pattern in (
            "release_*.md",
            "release_*.json",
            "regression_*.md",
            "regression_*.json",
            "alignment_*.json",
        ):
            for path in config.paths.reports_dir.glob(pattern):
                path.unlink()
                print(f"  removed {path.name}")
    config.paths.ensure()

    python = sys.executable
    code = _run(
        [
            python,
            "-m",
            "evalforge.cli",
            "demo",
            "--count",
            str(args.count),
            "--seed",
            str(args.seed),
        ]
    )
    if code != 0:
        print(f"\nDemo exited with code {code}.")
        return code

    if not args.skip_annotations:
        code = _run(
            [
                python,
                str(PROJECT_ROOT / "scripts" / "simulate_annotations.py"),
                "--all-runs",
                "--sessions",
                "60",
                "--seed",
                str(args.seed),
            ]
        )
        if code != 0:
            return code

        # Re-render the reports so the alignment section is populated.
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from evalforge.storage.store import RunStore

        store = RunStore(config.paths.runs, config.storage.database_name)
        runs = {item.label: item.run_id for item in store.list_runs()}
        baseline = runs.get("baseline")
        for label, run_id in runs.items():
            command = [python, "-m", "evalforge.cli", "report", "--run-id", run_id]
            if label == "candidate" and baseline:
                command += ["--baseline", baseline]
            _run(command)

    print("\nDemonstration artifacts regenerated.")
    print(f"  reports: {config.paths.reports_dir}")
    print(f"  runs:    {config.paths.runs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
