"""Prepare or verify the pinned Ghostbuster main external corpus."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from authentitext.data.ghostbuster import (
    GhostbusterError,
    prepare_ghostbuster_main,
    render_report,
    verify_ghostbuster_main,
)

PINNED_REVISION = "86ebd72590556a81622986fab736ab9227a948af"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-dir",
        type=Path,
        default=repo_root / "data" / "raw" / "ghostbuster" / "repository",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "data" / "processed" / "ghostbuster" / "main.jsonl.gz",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=repo_root / "data" / "metadata" / "ghostbuster_main_manifest.json",
    )
    parser.add_argument("--expected-revision", default=PINNED_REVISION)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args(argv)


def _git(repository_dir: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository_dir), *arguments],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode:
        raise GhostbusterError(result.stderr.strip() or "Git repository verification failed")
    return result.stdout.strip()


def _verify_repository(repository_dir: Path, expected_revision: str) -> None:
    if not repository_dir.is_dir():
        raise GhostbusterError(f"Missing Ghostbuster repository: {repository_dir}")
    revision = _git(repository_dir, "rev-parse", "HEAD")
    if revision != expected_revision:
        raise GhostbusterError(f"Ghostbuster revision is {revision}; expected {expected_revision}")
    if _git(repository_dir, "status", "--porcelain", "--untracked-files=no"):
        raise GhostbusterError("Ghostbuster tracked working tree is not clean")


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        temporary.write_text(render_report(report), encoding="utf-8", newline="\n")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        _verify_repository(args.repository_dir, args.expected_revision)
        if args.verify_only:
            report = json.loads(args.report.read_text(encoding="utf-8"))
            if report.get("revision") != args.expected_revision:
                raise GhostbusterError("Preparation report has the wrong repository revision")
            verify_ghostbuster_main(report, args.repository_dir, args.output)
            print(f"verified {report['output']['rows']} prepared Ghostbuster documents")
            return 0

        report = prepare_ghostbuster_main(
            repository_root=args.repository_dir,
            output_path=args.output,
            revision=args.expected_revision,
        )
        verify_ghostbuster_main(report, args.repository_dir, args.output)
        _write_report(args.report, report)
        print(f"prepared and verified {report['output']['rows']} Ghostbuster documents")
    except (
        GhostbusterError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
