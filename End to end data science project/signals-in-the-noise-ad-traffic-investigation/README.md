# Signals in the Noise

### A personal investigation of unusual patterns in advertising event data

I started this project with a question that sounded simple: **if a campaign suddenly looks unusual, how much evidence would I need before sending it for review?**

The more I worked with the data, the less comfortable I became with shortcuts. A burst of clicks can come from automation, but it can also come from a campaign taking off. Repeated activity can be suspicious, but normal return visits and shared devices exist. A weak conversion rate is useful context, not a verdict.

I ended up building two connected pieces of work:

1. An **observed-only quality investigation** that learns expected click and conversion behavior from earlier campaign history, measures residual surprises, and creates a neutral review queue.
2. A **controlled stress test** that adds documented experimental patterns to untouched source rows so I can measure what the workflow catches and misses.

The observed-only analysis is the main investigation. The simulation is an evaluation tool, not a source of claims about the original traffic.

## The data

I used the anonymized **Criteo Attribution Modeling for Bidding Dataset**. The source contains 30 days of advertising events, 16.5 million impressions, roughly 700 campaigns, clicks, conversion-linked impressions, and transformed cost fields. It does **not** contain invalid-traffic labels.

That missing label shaped the whole project. I do not call an original row fraudulent, train a real-fraud classifier, or report detection precision on observed traffic. I ask whether outcomes were expected, document what the score can see, and stop at a review recommendation.

Source and license:

- [Criteo dataset description](https://ailab.criteo.com/criteo-attribution-modeling-bidding-dataset/)
- [Hosted dataset file](https://huggingface.co/datasets/criteo/criteo-attribution-dataset)
- Data license: CC BY-NC-SA 4.0. Raw data is downloaded at runtime and is not committed here.

## How I worked through it

### 1. Get and validate the source

The downloader streams a chronological sample directly from the public compressed file. It checks the schema, row count, timestamp order, basic outcome counts, and file hash. The default local run uses 500,000 rows so the project is practical to reproduce.

### 2. Preserve observed rows

I load the source into a separate observed frame and keep a snapshot. The pipeline fails if the simulation step mutates that frame. Experimental rows carry their own origin, scenario, and episode fields.

### 3. Build the event store and SQL features

Events are loaded into SQLite. SQL creates 30-minute campaign windows with:

- impression, click, and conversion volume;
- click and conversion rates;
- returning-user and anonymized-source concentration;
- click timing and clicks per unique user;
- transformed cost exposure;
- rolling campaign baselines built only from earlier windows.

### 4. Model expected behavior on observed traffic

Every simulation-touched campaign window is removed from this track. I split the remaining windows chronologically, fit an expected click-rate model and an expected conversion-rate model on the earlier period, then replay them on the later period.

Both models are aggregated binomial logistic regressions implemented with NumPy. The click model uses information that would be available before observing clicks in the window. The conversion model can use observed click behavior because it answers a later-stage downstream-quality question.

### 5. Turn residuals into review leads

For each held-out window, I compare actual clicks and conversion-linked impressions with their expected values. The quality score combines:

- absolute click deviation;
- conversion shortfall;
- volume change against recent campaign history;
- returning-user and source concentration;
- transformed cost exposure.

Each component becomes a percentile against the training period. Only the highest held-out percentiles enter `observed_review_queue.csv`. The action is `review_traffic_quality`, never an automatic block.

### 6. Run a controlled stress test

The second track adds a concentrated click burst, a quieter low-and-slow pattern, and an impression flood. I also add a benign popularity spike as a negative control. A supervised baseline and transparent anomaly score try to recover those planted examples.

The held-out results are reported even when the more complicated hybrid score does not win. That is useful information: complexity has to earn its place.

### 7. Write cases a reviewer could actually use

The pipeline produces a narrative report, calibration tables, a dashboard, and individual case notes. Each case compares observed and expected outcomes, explains why the window surfaced, and lists the next evidence I would seek before making a decision.

## Pipeline map

```text
public Criteo file
        |
        v
validated chronological sample
        |
        +-------------------- untouched observed rows --------------------+
        |                                                                 |
        |                                                    documented simulation layer
        |                                                                 |
        +-----------------------------+-----------------------------------+
                                      v
                              SQLite event store
                                      |
                                      v
                         SQL campaign-window features
                                      |
                 +--------------------+---------------------+
                 |                                          |
        observed-only windows                       controlled stress test
                 |                                          |
       campaign rolling baselines                   anomaly + logistic model
                 |                                          |
     expected clicks and conversions                 held-out evaluation
                 |                                          |
        residual quality score                               |
                 +--------------------+---------------------+
                                      v
                        reports + dashboards + case notes
```

## Run it

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
.\run.ps1 -Rows 500000
```

The first run downloads the public source. Pass `-Rows 0` to process the complete file, but expect substantially more time, memory, and disk usage.

### Open the notebook

The executed notebook contains the same 500,000-row results committed with the project, so it can be read directly on GitHub. To rerun it locally:

```powershell
pip install -r requirements-notebook.txt
jupyter lab notebooks/01_observed_traffic_quality_walkthrough.ipynb
```

On a fresh clone, the notebook recreates missing pipeline artifacts automatically. Set `SIGNALS_DATA_DIR` before starting Jupyter if raw and processed data should live outside the repository.

To run the tests:

```powershell
python -m unittest discover -s tests -v
```

## Outputs worth opening first

- `notebooks/01_observed_traffic_quality_walkthrough.ipynb` - executed, step-by-step analysis with tables and charts
- `reports/observed_quality_report.md` - my observed-only investigation and conclusions
- `reports/observed_quality_dashboard.html` - calibration and the first review cases
- `reports/observed_review_queue.csv` - neutral analyst-facing queue from held-out observed windows
- `reports/observed_cases/` - expected-versus-observed case notes
- `reports/investigation_report.md` - the controlled stress-test narrative
- `reports/dashboard.html` - stress-test performance and experimental cases
- `reports/model_card.md` - intended use, limits, metrics, and safeguards
- `data/traffic.db` - local SQLite database for follow-up queries

The repository includes compact reports from a 500,000-row run. Raw events, the processed feature tables, and the SQLite database remain untracked and can be recreated.

## Repository map

```text
config/                       thresholds and reproducibility settings
data/                         ignored raw and processed runtime data
docs/                         data dictionary and my decision log
notebooks/                    executed investigation walkthrough
reports/                      generated findings, dashboards, and cases
sql/                          feature, history, trend, and triage queries
src/download_data.py          source acquisition and validation
src/simulate_abuse.py         explicit experimental scenarios
src/database.py               SQLite loading and SQL execution
src/quality_modeling.py       expected behavior and observed residual scoring
src/quality_reporting.py      observed report, dashboard, and case generation
src/modeling.py               controlled stress-test models
src/reporting.py              controlled stress-test reporting
src/pipeline.py               end-to-end runner
tests/                        data-contract, leakage, and model tests
```

## What I would not conclude

- An anomaly is not automatically fraud.
- A non-converting click is not automatically invalid.
- The expected-behavior models measure calibration, not abuse-detection accuracy.
- The simulated labels do not validate performance against real attackers.
- The early chronological sample is useful for reproducibility, not population-level claims about all 30 days.
- Criteo's `cost` is transformed; I call it cost units, never dollars.

The useful result is a reproducible way to turn unexplained behavior into a small, evidence-rich review queue while keeping the final judgment with a person.
