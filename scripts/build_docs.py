"""Assemble the MkDocs site from the READMEs already in the repository.

The site is generated rather than hand written so that the READMEs stay the
single source of truth. Relative links inside a project README keep pointing at
files that only exist on GitHub, so they are rewritten to absolute GitHub URLs
(blob for files, raw for images).
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from urllib.parse import quote

REPO = "SaguPandya96/Data-Science-"
BRANCH = "master"
BLOB = f"https://github.com/{REPO}/blob/{BRANCH}"
RAW = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}"

ROOT = Path(__file__).resolve().parents[1]
PROJECT_DIR = "End to end data science project"
DOCS = ROOT / "docs"

# Folder name -> page slug, in the order the front page presents them.
PROJECTS = [
    ("signals-in-the-noise-ad-traffic-investigation", "signals-in-the-noise"),
    ("SupplyLens", "supplylens"),
    ("store-level-revenue-forecasting", "store-revenue-forecasting"),
    ("Amazon Review Intelligence and Recommender System", "amazon-review-intelligence"),
    ("GitHub Open-Source Repository Recommendation System", "github-repo-recommender"),
    ("crypto-alternative-data-forecasting", "crypto-forecasting"),
    ("evasion-gap", "evasion-gap"),
    ("evalforge-agent-evaluation", "evalforge"),
    ("AuthentiText", "authentitext"),
]

STUDY_FOLDERS = [
    ("Python For Data Science", "Python, pandas, NumPy, Matplotlib and Seaborn notebooks."),
    ("Statistics for data science", "Hypothesis testing, t-tests, z-tests and outlier detection."),
    ("Feature Engineering For Data Science", "Missing values, encoding, scaling and imbalanced data."),
    ("Feature Selection For Data Science", "Variance threshold, correlation, chi-squared and information gain."),
    ("Machine Learning For Data Science", "Linear, multiple, ridge and lasso regression."),
]

LINK = re.compile(r"(!?)\[([^\]]*)\]\(([^)]+)\)")


def rewrite_project_links(text: str, folder: str) -> str:
    """Point relative links at GitHub, and the index link at the site home page."""
    base = quote(f"{PROJECT_DIR}/{folder}")

    def repl(match: re.Match[str]) -> str:
        bang, label, target = match.groups()
        if target.startswith(("http://", "https://", "mailto:", "#")):
            return match.group(0)
        if target in ("../../README.md", "../../README.MD"):
            return f"[{label}](../index.md)"
        host = RAW if bang else BLOB
        return f"{bang}[{label}]({host}/{base}/{quote(target)})"

    return LINK.sub(repl, text)


def build_index() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for folder, slug in PROJECTS:
        encoded = quote(f"{PROJECT_DIR}/{folder}/")
        text = text.replace(f"]({encoded})", f"](projects/{slug}.md)")
    text = text.replace(
        "Python · Statistics · Feature engineering · Feature selection · Machine learning",
        "Notebooks from the ground work behind the projects: "
        "[browse the study material](study.md).",
    )
    (DOCS / "index.md").write_text(text, encoding="utf-8")


def build_projects() -> None:
    out = DOCS / "projects"
    out.mkdir(parents=True, exist_ok=True)
    for folder, slug in PROJECTS:
        readme = ROOT / PROJECT_DIR / folder / "README.md"
        body = rewrite_project_links(readme.read_text(encoding="utf-8"), folder)
        (out / f"{slug}.md").write_text(body, encoding="utf-8")


def build_study() -> None:
    lines = [
        "# Study material",
        "",
        "The notebooks behind the projects. They are working notes rather than",
        "polished write-ups, and they live on GitHub so their outputs stay browsable.",
        "",
    ]
    for folder, blurb in STUDY_FOLDERS:
        count = len(list((ROOT / folder).glob("*.ipynb")))
        url = f"{BLOB}/{quote(folder)}"
        lines.append(f"### [{folder}]({url})")
        lines.append("")
        lines.append(f"{blurb} {count} notebooks.")
        lines.append("")
    (DOCS / "study.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if DOCS.exists():
        shutil.rmtree(DOCS)
    DOCS.mkdir()
    build_index()
    build_projects()
    build_study()
    print(f"built {len(PROJECTS)} project pages into {DOCS}")


if __name__ == "__main__":
    main()
