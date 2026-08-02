"""Build, execute, and export the canonical end-to-end notebook."""

from __future__ import annotations

import os
from pathlib import Path

# A restricted local Windows sandbox may explicitly opt into the Jupyter
# connection-file fallback. Normal project execution keeps secure writes on.
if os.getenv("SUPPLYLENS_RESTRICTED_WINDOWS_NOTEBOOK") == "1":
    os.environ.setdefault("JUPYTER_ALLOW_INSECURE_WRITES", "1")

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from nbconvert import HTMLExporter
from traitlets.config import Config

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "SupplyLens_End_to_End_Project.ipynb"
HTML_PATH = ROOT / "reports" / "html" / "SupplyLens_End_to_End_Project.html"


def markdown(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(text.strip())


def code(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(source.strip())


def build_notebook() -> nbformat.NotebookNode:
    cells = [
        markdown(
            """
# SupplyLens: Supplier Delivery Risk and Operational Decision Intelligence

## 1. Executive summary

SupplyLens uses real public-health commodity shipment history to rank ASN/DN shipments by
the probability of arriving more than seven days after schedule. The recommended system is
a calibrated logistic-regression pipeline operated as an exact top-20% review queue. All
values below are loaded from executed, machine-readable project artifacts.
"""
        ),
        markdown(
            """
## 2. Business problem

Operations teams cannot investigate every shipment. The decision is which scheduled
shipments should receive scarce review attention before the actual delivery outcome is known.

## 3. Stakeholders

Supply-chain operations, procurement, supplier-management, transportation, inventory
planning, and data science teams are potential consumers. No organizational adoption is
claimed.

## 4. Decisions supported

The system supports risk-ranked review, supplier investigation, capacity planning, and
historical lead-time analysis. It does not automate supplier penalties, purchasing, routing,
or inventory decisions.

## 5. Success criteria

The primary criteria are precision-recall performance, lift and recall at fixed review
capacity, calibrated probability quality, temporal generalization, and a reproducible scoring
contract. Accuracy is not the primary metric for this imbalanced outcome.
"""
        ),
        code(
            """
from pathlib import Path
import json
import os
import subprocess
import sys

import pandas as pd
from IPython.display import Image, Markdown, display

ROOT = Path.cwd()
assert (ROOT / "pyproject.toml").exists(), "Run the notebook from the repository root."
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda value: f"{value:,.4f}")
"""
        ),
        markdown(
            """
## 6. Data source

The selected source is the U.S. Agency for International Development Supply Chain Shipment
Pricing / SCMS Delivery History dataset. The retrieval script uses a commit-pinned public
mirror because the original catalog asset is no longer directly retrievable.

## 7. Data provenance

The raw file is checksum-verified and preserved unchanged. The accessible mirror does not
state an explicit redistribution license, so the raw CSV is downloaded locally and excluded
from version control. Full provenance and alternative-source review are in `docs/`.
"""
        ),
        code(
            """
# Execute the reproducible pipeline in dependency order. A pre-existing raw file is reused
# only after its pinned checksum is verified by the acquisition script.
commands = [
    [sys.executable, "scripts/download_data.py"],
    [sys.executable, "scripts/validate_data.py"],
    [sys.executable, "scripts/train.py"],
    [sys.executable, "scripts/build_reports.py"],
    [sys.executable, "scripts/build_readme.py"],
]
pipeline_log = []
for command in commands:
    completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    pipeline_log.append({"command": " ".join(command[1:]), "status": "completed"})
pd.DataFrame(pipeline_log)
"""
        ),
        markdown("## 8. Data dictionary\n\nThe field-level source-to-model mapping is maintained in `docs/DATA_DICTIONARY.md`."),
        code(
            """
quality = json.loads((ROOT / "reports/metrics/data_quality.json").read_text(encoding="utf-8"))
pd.DataFrame({
    "measure": ["source lines", "source columns", "modeled shipments", "severe delays", "severe-delay rate"],
    "value": [
        quality["raw"]["metrics"]["row_count"],
        quality["raw"]["metrics"]["column_count"],
        quality["processed"]["metrics"]["shipment_count"],
        quality["processed"]["metrics"]["severe_delay_count"],
        quality["processed"]["metrics"]["severe_delay_rate"],
    ],
})
"""
        ),
        markdown(
            """
## 9. Data quality

Automated checks cover file checksum, schema and shape, identifiers, date parsing, numeric
ranges, duplicates, sequence anomalies, and expected categories. Retained warnings are shown
below rather than silently removed.
"""
        ),
        code("pd.DataFrame({'retained_warning': quality['raw']['warnings']})"),
        markdown(
            """
## 10. Target definition

The prediction unit is one ASN/DN shipment. A positive outcome is actual delivery strictly
more than seven calendar days after the latest scheduled-delivery date across its source
lines. The choice preserves operational relevance and sufficient positive volume; it was not
selected by model performance.
"""
        ),
        code("pd.read_csv(ROOT / 'reports/tables/target_threshold_prevalence.csv')"),
        markdown(
            """
## 11. Leakage audit

Scoring occurs at scheduled-delivery commitment, used as the chronological proxy because the
source lacks the exact schedule-entry timestamp. Actual delivery, recorded date, computed
delay, outcomes, and target fields are explicitly blocklisted. Shifted historical target
features are implemented and tested but excluded from the production model because their
exact prediction-time availability cannot be guaranteed.
"""
        ),
        code(
            """
from supplylens.features import LEAKAGE_BLOCKLIST, MODEL_FEATURES, assert_no_leakage
assert_no_leakage(MODEL_FEATURES)
pd.DataFrame({"included_model_feature": MODEL_FEATURES})
"""
        ),
        markdown("## 12. Exploratory analysis\n\nThe following executed figures answer target prevalence, drift, supplier, mode, missingness, and delay-distribution questions."),
        code(
            """
for name in ["target_prevalence.png", "volume_and_target_drift.png", "supplier_performance.png", "shipment_mode_performance.png", "delivery_delay_distribution.png", "missingness.png"]:
    display(Image(filename=str(ROOT / "reports/figures" / name), width=820))
"""
        ),
        markdown(
            """
## 13. Feature engineering

Reusable transformations create calendar indicators, log-scaled continuous measures,
missingness flags, and categorical encodings. Unique identifiers and all post-outcome fields
are excluded from predictive features.

## 14. Temporal split

Model selection uses a chronological training period and 2013 validation period. The
2014-2015 test period remains untouched until the complete model and calibration choice is
fixed.
"""
        ),
        code(
            """
metrics = json.loads((ROOT / "reports/metrics/final_metrics.json").read_text(encoding="utf-8"))
pd.DataFrame(metrics["splits"]).T
"""
        ),
        code("display(Image(filename=str(ROOT / 'reports/figures/temporal_split_timeline.png'), width=900))"),
        markdown(
            """
## 15. Baselines

Validation compares the empirical-prevalence baseline, a smoothed prior-period supplier-rate
rule, and logistic regression. Continuous lead-time baselines include schedule, global,
supplier, and lane historical medians.

## 16. Advanced modeling

Histogram gradient boosting is the advanced classifier. Logistic regression remains the
selected model because gradient boosting did not clear the predeclared 0.01 validation
PR-AUC improvement margin.
"""
        ),
        code("pd.read_csv(ROOT / 'reports/tables/model_comparison.csv')"),
        markdown("## 17. Calibration\n\nNone, sigmoid, and isotonic calibration are compared only on validation labels. Isotonic calibration has the lowest validation Brier score and is retained."),
        code("pd.read_csv(ROOT / 'reports/tables/calibration_comparison.csv')"),
        code("display(Image(filename=str(ROOT / 'reports/figures/calibration_plot.png'), width=850))"),
        markdown(
            """
## 18. Threshold selection

The recommended policy ranks all shipments and reviews the exact highest-risk 20%. Capacity
policies of 5%, 10%, 20%, and 30% are evaluated. A separate cost-sensitive policy is labeled
as assumption-based because intervention economics are not observed in the source data.
"""
        ),
        code("pd.read_csv(ROOT / 'reports/tables/capacity_analysis_test.csv')"),
        markdown("## 19. Final evaluation\n\nThe following final-period metrics were generated once for the selected calibrated system."),
        code("pd.DataFrame([metrics['test_metrics']]).T.rename(columns={0: 'value'})"),
        code(
            """
for name in ["pr_and_roc_curves.png", "gains_and_lift.png", "confusion_matrix.png"]:
    display(Image(filename=str(ROOT / "reports/figures" / name), width=850))
"""
        ),
        markdown(
            """
## 20. Explainability

Permutation importance summarizes global predictive associations. Queue-level local
contributors are based on controlled feature perturbations and use non-causal language.
"""
        ),
        code("pd.read_csv(ROOT / 'reports/tables/permutation_importance.csv').head(15)"),
        code("display(Image(filename=str(ROOT / 'reports/figures/permutation_importance.png'), width=850))"),
        markdown("## 21. Segment analysis\n\nSegment metrics include sample size, prevalence, precision, recall, false-negative rate, Brier score, and a low-volume reliability flag."),
        code("pd.read_csv(ROOT / 'reports/tables/segment_performance.csv').head(20)"),
        markdown("## 22. Error analysis\n\nExamples below are drawn from the final temporal test period and labeled by error type."),
        code("pd.read_csv(ROOT / 'reports/tables/error_examples.csv').head(12)"),
        markdown(
            """
## 23. Lead-time experiment

Learned P50 and P90 gradient-boosting models are compared with scheduled and prior-period
median baselines. The learned P50 model is not recommended because it underperforms the
scheduled lead-time baseline; this negative result is preserved.
"""
        ),
        code("pd.read_csv(ROOT / 'reports/tables/lead_time_model_comparison.csv')"),
        code("display(Image(filename=str(ROOT / 'reports/figures/lead_time_model_comparison.png'), width=850))"),
        markdown("## 24. Shipment intervention queue\n\nThis final-period decision-support table is ranked by calibrated probability and carries the fixed capacity flag."),
        code("pd.read_csv(ROOT / 'reports/tables/shipment_intervention_queue.csv').head(10)"),
        markdown("## 25. Supplier scorecard\n\nSupplier summaries require a minimum volume before performance priority is considered eligible."),
        code("pd.read_csv(ROOT / 'reports/tables/supplier_scorecard.csv').head(10)"),
        markdown("## 26. Replenishment indicators\n\nThese are historical planning scenarios based only on observed shipment quantity and lead-time variability; they are not inventory recommendations."),
        code("pd.read_csv(ROOT / 'reports/tables/replenishment_risk_indicators.csv').head(10)"),
        markdown(
            """
## 27. Causal-readiness analysis

The observational fulfillment-path comparison has weak overlap and residual imbalance. The
project therefore reports diagnostics and associations, not a causal effect, and recommends
a randomized or credible quasi-experimental design with additional decision context.
"""
        ),
        code(
            """
causal = json.loads((ROOT / "reports/metrics/causal_readiness.json").read_text(encoding="utf-8"))
pd.json_normalize(causal).T
"""
        ),
        code("display(Image(filename=str(ROOT / 'reports/figures/causal_overlap.png'), width=850))"),
        markdown("## 28. Business-impact scenario\n\nEvery monetary input is a configurable, unobserved assumption. Results are expected-value scenarios, not realized savings."),
        code("pd.read_csv(ROOT / 'reports/tables/business_impact_sensitivity.csv')"),
        markdown(
            """
## 29. Deployment

The serialized bundle contains preprocessing, classifier, and calibrator. `scripts/score.py`
validates a shipment-level CSV and emits calibrated probability, rank, and review flag. The
dashboard reads only generated local artifacts and requires no external service.

## 30. Monitoring

The monitoring design covers schema, nulls, invalid dates, unseen categories, numeric and
category drift, score drift, queue size, and delayed-label performance. Thresholds,
ownership assumptions, rollback, and retraining criteria are documented in
`docs/DEPLOYMENT_AND_MONITORING.md`.
"""
        ),
        code(
            """
drift = json.loads((ROOT / "reports/metrics/monitoring_drift_example.json").read_text(encoding="utf-8"))
pd.json_normalize(drift).T
"""
        ),
        markdown(
            """
## 31. Limitations

The data is historical, ends in 2015, and describes one public-health supply-chain program.
Schedule-entry timestamps, on-hand inventory, intervention outcomes, urgency, contracts, and
many decision drivers are absent. Calibration has drifted in the final period. Segment
results are descriptive and may be unstable at low volume.

## 32. Final recommendation

Use the isotonic-calibrated logistic system as a transparent batch prioritization baseline at
the exact top-20% review capacity, with human review and drift monitoring. Do not deploy the
lead-time model or interpret observational fulfillment differences causally. Validate on
current local data before any real operational use.
"""
        ),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "version": "3.11"}
    return notebook


def execute_and_export() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    HTML_PATH.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    nbformat.write(notebook, NOTEBOOK_PATH)
    client = NotebookClient(
        notebook,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    try:
        client.execute()
    except CellExecutionError as exc:
        nbformat.write(notebook, NOTEBOOK_PATH)
        raise RuntimeError("Canonical notebook execution failed") from exc
    nbformat.write(notebook, NOTEBOOK_PATH)

    config = Config()
    config.HTMLExporter.embed_images = True
    exporter = HTMLExporter(config=config)
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    body, _ = exporter.from_notebook_node(notebook)
    HTML_PATH.write_text(body, encoding="utf-8")
    print(f"Executed {NOTEBOOK_PATH.relative_to(ROOT)} from a clean kernel")
    print(f"Exported {HTML_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    execute_and_export()
