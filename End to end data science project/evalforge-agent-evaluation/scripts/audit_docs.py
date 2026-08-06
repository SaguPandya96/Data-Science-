"""Verify that every factual claim in the documentation matches the code.

Documentation drifts silently. A README that says "21 deterministic evaluators" stays
convincing long after someone adds a twenty-second, and a reviewer who checks one number
and finds it wrong reasonably stops trusting the rest. This project makes claims about
its own accuracy, so stale ones cost more here than they would elsewhere.

The test-count badge drifted twice during development before this existed, both times
caught by luck rather than process. That is the specific failure this script prevents.

Exits non-zero when any claim fails, so CI can gate on it.

Usage:
    python scripts/audit_docs.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evalforge.evaluators.registry import DETERMINISTIC_EVALUATORS  # noqa: E402
from evalforge.schemas.common import (  # noqa: E402
    CRITICAL_FAILURE_CATEGORIES,
    FailureCategory,
    ScenarioCategory,
    ToolName,
)

#: Markers that should never survive into a published document.
LEFTOVER_MARKERS = ("TODO", "FIXME", "XXX", "HACK", "lorem ipsum", "placeholder text")

#: Directories whose contents are generated or vendored, so not worth auditing.
SKIP_PARTS = (".venv", "__pycache__", ".git", "node_modules")


class Audit:
    """Collects pass/fail results so the whole report can be printed at once."""

    def __init__(self) -> None:
        self.passed: list[str] = []
        self.failed: list[str] = []

    def check(self, ok: bool, description: str, detail: str = "") -> None:
        """Record one claim, with detail shown only when it fails."""
        if ok:
            self.passed.append(description)
        else:
            self.failed.append(f"{description}{f' — {detail}' if detail else ''}")

    def report(self) -> int:
        """Print the outcome and return a process exit code."""
        print(f"{len(self.passed)} checks passed")
        if not self.failed:
            print("Documentation matches the code.")
            return 0
        print(f"\n{len(self.failed)} FAILED:\n")
        for item in self.failed:
            print(f"  - {item}")
        return 1


def collected_test_count() -> int:
    """Total tests pytest collects.

    Summed from the per-file tallies rather than scraped from the summary line, which
    varies between pytest versions and plugin sets. An earlier version of this script
    grabbed the last line containing "test" and reported a filename's count as the
    total, which is exactly the kind of quiet wrongness being audited for.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return sum(
        int(match.group(1))
        for line in proc.stdout.splitlines()
        if (match := re.match(r"^tests[/\\].*:\s*(\d+)$", line.strip()))
    )


def check_links(audit: Audit) -> None:
    """Every relative markdown link must resolve to a file that exists."""
    documents = [PROJECT_ROOT / "README.md", *(PROJECT_ROOT / "docs").rglob("*.md")]
    for document in documents:
        base = document.parent
        for match in re.finditer(r"\]\((?!https?:)([^)#][^)]*?)\)", document.read_text("utf-8")):
            target = match.group(1).split("#")[0]
            if not target:
                continue
            resolved = (base / target).resolve()
            audit.check(
                resolved.exists(),
                f"link resolves: {document.name} -> {target}",
                "no such file",
            )


def check_counts(audit: Audit, readme: str) -> None:
    """Numeric claims in the README must match what the code actually defines."""
    # `none` is a sentinel meaning "no failure", not a failure category. Counting it
    # made an earlier version of this script report a phantom off-by-one.
    real_categories = [item for item in FailureCategory if item.value != "none"]

    claims: list[tuple[str, str, int]] = [
        (
            "deterministic evaluators",
            r"(\d+) deterministic evaluators",
            len(DETERMINISTIC_EVALUATORS),
        ),
        ("failure categories", r"(\d+) categories across seven families", len(real_categories)),
        (
            "scenario categories",
            r"(\d+) categories, fully reproducible",
            len(list(ScenarioCategory)),
        ),
        ("tests (badge)", r"tests-(\d+)%20passing", collected_test_count()),
        ("tests (body)", r"pytest -q\s+#\s+(\d+) tests", collected_test_count()),
    ]
    for name, pattern, actual in claims:
        match = re.search(pattern, readme)
        if match is None:
            audit.check(False, f"README states a {name} count", "claim not found")
            continue
        claimed = int(match.group(1))
        audit.check(claimed == actual, f"{name}: {actual}", f"README says {claimed}")

    audit.check(
        len(CRITICAL_FAILURE_CATEGORIES) == 6,
        f"critical failure categories: {len(CRITICAL_FAILURE_CATEGORIES)}",
        "README says six",
    )
    audit.check(len(list(ToolName)) == 8, f"tools: {len(list(ToolName))}", "README says 8")


def check_reported_metrics(audit: Audit, readme: str) -> None:
    """Figures quoted in the README must match the committed report artifacts."""
    for label in ("baseline", "candidate"):
        path = PROJECT_ROOT / "reports" / f"release_{label}.json"
        if not path.exists():
            continue
        scalars = json.loads(path.read_text("utf-8"))["metadata"]["metrics"]["scalars"]
        pass_rate = f"{scalars['pass_rate'] * 100:.1f}%"
        criticals = str(int(scalars["critical_failures"]))
        audit.check(pass_rate in readme, f"{label} pass rate {pass_rate} quoted correctly", "stale")
        audit.check(
            criticals in readme, f"{label} critical failures {criticals} quoted correctly", "stale"
        )


def check_structure(audit: Audit) -> None:
    """Directories the README documents must exist."""
    for name in (
        "configs",
        "dashboard",
        "data",
        "docs",
        "notebooks",
        "reports",
        "scripts",
        "src",
        "tests",
    ):
        audit.check((PROJECT_ROOT / name).is_dir(), f"{name}/ exists", "documented but missing")


def check_leftovers(audit: Audit) -> None:
    """No unfinished-work markers in shipped source or documentation."""
    for path in [*PROJECT_ROOT.rglob("*.py"), *PROJECT_ROOT.rglob("*.md")]:
        if any(part in SKIP_PARTS for part in path.parts) or path.name == Path(__file__).name:
            continue
        for number, line in enumerate(path.read_text("utf-8", errors="replace").splitlines(), 1):
            for marker in LEFTOVER_MARKERS:
                if marker.lower() in line.lower():
                    audit.check(
                        False,
                        f"no leftover markers in {path.relative_to(PROJECT_ROOT)}",
                        f"{marker} on line {number}",
                    )


def main() -> int:
    """Run every audit and report."""
    audit = Audit()
    readme = (PROJECT_ROOT / "README.md").read_text("utf-8")

    check_links(audit)
    check_counts(audit, readme)
    check_reported_metrics(audit, readme)
    check_structure(audit)
    check_leftovers(audit)

    return audit.report()


if __name__ == "__main__":
    raise SystemExit(main())
