# Working in this repo

Notes for anyone changing this repository, including AI coding assistants. I build these
projects with AI help, and this file says how I want that help to work.

## Who decides what

I choose the questions, decide what gets tested and what counts as a result, and review
every change before it merges. An assistant can propose methods, write code and draft
docs, but it should flag decisions rather than make them quietly: a changed metric, a new
threshold, a result that looks too good, anything that would change a conclusion in a
README.

## Layout

Each project in `End to end data science project/` stands on its own: its own
dependencies, tests, README and CI workflow in `.github/workflows/`, filtered to that
project's folder. Change one project at a time. Don't share code between projects.

To work on a project, open its folder and follow its README. Most have a `Makefile` or a
`pyproject.toml`; the usual loop is:

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
python -m pytest
```

`render.yaml` at the root deploys AuthentiText and the OfferLift planner. A change to it
affects live services, so treat it with care.

## How the analysis should work

- **Compare against something simple.** Every model gets measured against a plain
  baseline: random targeting, a moving average, best-sellers, the supplier's own date. If
  the baseline wins, that's the result.
- **Validate the way the data arrives.** Time-ordered data gets time-ordered splits. No
  feature may use information that wouldn't exist at prediction time.
- **Fix the test before running it.** When a result needs confirming, write the design and
  the pass rule down first (OfferLift's `docs/ANALYSIS_PLAN.md` is the example). Anything
  decided after seeing results gets labeled that way.
- **Report what came out.** No tuning until something looks good, and no dropping
  inconvenient results. A method that didn't help is still worth writing up.
- **Every number traces back to code.** Figures in READMEs and docs come from files the
  pipeline writes (usually `reports/metrics/`). If you change the analysis, rerun it and
  update the text to match.

## Data

Raw data is downloaded by a script with a pinned checksum and is not committed, unless the
project's `data/README.md` says the license allows it. Never commit credentials or `.env`
files.

## Writing

READMEs are for people deciding whether to read further, so lead with the question and
the answer in plain language. Write in first person, keep sentences short, give numbers
with their intervals, and say what the result does not show. Skip hype and filler.

## Before you open a pull request

- Lint and tests pass for the project you changed.
- If results changed, the pipeline was rerun and the README matches the new numbers.
- The PR says what changed and why, and how it was checked.
