# Signals in the Noise

### A personal investigation of suspicious patterns in advertising event data

I started this project with a question that sounded simple: **if a campaign suddenly looks unusual, how much evidence would I need before treating it as abuse?**

The more I worked with the data, the less comfortable I became with shortcuts. A burst of clicks can be automation, but it can also be a real campaign taking off. Repeated activity can be suspicious, but shared devices and normal return visits exist. A weak conversion rate is useful context, not a verdict.

So this became less of a “fraud classifier” and more of an investigation workflow. It moves from raw events to SQL-based behavioral summaries, compares a transparent anomaly detector with a small supervised model, and ends with a review queue rather than an automatic blocklist.

## The question I explored

> Can I find deliberately planted abuse patterns in real advertising-event structure while keeping ordinary traffic out of the highest-risk queue?

I used anonymized impression-level data from the **Criteo Attribution Modeling for Bidding Dataset**. The source contains 30 days of real traffic, 16.5 million impressions, roughly 700 campaigns, clicks, conversions, and transformed cost fields. It does **not** contain fraud labels.

That last point matters. I preserve the observed data and add a separate simulation layer with a few documented behaviors: a concentrated click burst, a quieter low-and-slow pattern, and an impression flood. I also add a benign traffic spike as a negative control. The project only measures whether the workflow rediscovers those planted scenarios; it never claims that an original Criteo event is fraudulent.

Source and license:

- [Criteo dataset description](https://ailab.criteo.com/criteo-attribution-modeling-bidding-dataset/)
- [Hosted dataset file](https://huggingface.co/datasets/criteo/criteo-attribution-dataset)
- Data license: CC BY-NC-SA 4.0. The raw dataset is downloaded at runtime and is not committed here.

## What the pipeline does

```text
public .tsv.gz
      |
      v
validated local sample -----> untouched observed events
      |                                  |
      |                         documented simulation layer
      |                                  |
      +----------------+-----------------+
                       v
                 SQLite event store
                       |
                       v
             SQL campaign-window features
                       |
             +---------+----------+
             |                    |
     robust anomaly score   logistic model
             |                    |
             +---------+----------+
                       v
               cautious review queue
                       |
                       v
          report + case notes + dashboard
```

The model is intentionally implemented with NumPy instead of hiding the logic behind a large framework. This keeps the project easy to inspect: feature scaling, class weighting, gradient descent, threshold choice, and evaluation are all visible in `src/modeling.py`.

## Run it

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
.\run.ps1 -Rows 500000
```

The first run streams a chronological sample from the public compressed file. `500000` rows is enough for a local demonstration. Pass `-Rows 0` to process the complete source file, but expect substantially more time and disk usage.

Outputs are written to:

- `reports/investigation_report.md` — the main narrative
- `reports/dashboard.html` — a self-contained summary
- `reports/model_card.md` — model behavior and boundaries
- `reports/cases/` — evidence packets for the highest-risk windows
- `reports/review_queue.csv` — the included example run's blinded triage output
- `data/processed/review_queue.csv` — the current local run's ranked triage output
- `data/traffic.db` — SQLite database for follow-up analysis

The repository includes the compact results from a 500,000-row run. Raw events and the SQLite database stay untracked; rerunning the pipeline recreates them locally.

To run the tests:

```powershell
python -m unittest discover -s tests -v
```

## Repository map

```text
config/                 thresholds and reproducibility settings
data/                   ignored raw/processed runtime data
reports/                generated findings and case notes
docs/                   data dictionary and my decision log
sql/                    feature, trend, and triage queries
src/download_data.py    source acquisition and validation
src/simulate_abuse.py   explicit experimental scenarios
src/database.py         SQLite loading and SQL execution
src/modeling.py         anomaly and supervised models
src/reporting.py        report/dashboard generation
src/pipeline.py         end-to-end runner
tests/                  small data-contract and model tests
```

## What I would not conclude

- An anomaly is not automatically fraud.
- A non-converting click is not automatically invalid.
- The simulated labels do not validate performance against real attackers.
- The early chronological sample is useful for reproducibility, not population-level claims about all 30 days.
- Criteo's `cost` is transformed; I call it “cost units,” never dollars.

Those limits are part of the result. The useful output is a prioritized set of cases with enough context for a person to investigate—not a claim that a score has discovered the truth.
