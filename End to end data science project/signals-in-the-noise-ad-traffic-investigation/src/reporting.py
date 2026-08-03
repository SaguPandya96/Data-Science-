from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd


def _pct(value: float) -> str:
    return f"{100 * float(value):.1f}%"


def _number(value: float) -> str:
    return f"{int(value):,}"


def _metric_row(name: str, values: dict) -> str:
    return (
        f"| {name} | {_pct(values['precision'])} | {_pct(values['recall'])} | "
        f"{_pct(values['f1'])} | {values['average_precision']:.3f} | "
        f"{_pct(values['false_positive_rate'])} |\n"
    )


def write_investigation_report(
    reports_dir: Path,
    metadata: dict,
    manifest: pd.DataFrame,
    features: pd.DataFrame,
    result: dict,
) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics = result["metrics"]
    hybrid = metrics["hybrid_test"]
    scored = result["scored"]
    test = scored.loc[scored["data_period"] == "test"]
    review_threshold = metrics["thresholds"]["review"]
    simulation_rows = int(manifest["rows_added"].sum())
    sample_days = float(metadata["max_timestamp_seconds"]) / 86_400

    adaptive = test.loc[test["evaluation_scenario"] == "adaptive_low_and_slow"]
    adaptive_positives = int(adaptive["abuse_label"].sum())
    adaptive_caught = int((adaptive["risk_score"] >= review_threshold).sum())
    clean_spike = test.loc[test["evaluation_scenario"] == "clean_popularity_spike"]
    clean_spike_queued = int((clean_spike["risk_score"] >= review_threshold).sum())
    unplanted_alerts = test.loc[
        (test["abuse_label"] == 0) & (test["risk_score"] >= review_threshold)
    ].sort_values("risk_score", ascending=False)

    supervised = metrics["supervised_test"]
    if hybrid["f1"] + 1e-9 < supervised["f1"]:
        model_decision = (
            f"The hybrid did not beat the supervised model on the held-out period "
            f"(F1 {hybrid['f1']:.3f} versus {supervised['f1']:.3f}). I would not hide that result or "
            "promote the more complicated score by default. My next iteration would keep the supervised "
            "model as the working candidate and redesign the anomaly component specifically around unseen behavior."
        )
    else:
        model_decision = (
            "The hybrid improved held-out F1 over the supervised model, but I would still require a longer "
            "time-based replay before using the extra complexity operationally."
        )

    if not unplanted_alerts.empty:
        top_unplanted = unplanted_alerts.iloc[0]
        surprise = (
            f"The highest-scoring test window without a planted abuse label belonged to campaign "
            f"`{int(top_unplanted['campaign'])}` with a risk score of "
            f"{top_unplanted['risk_score']:.3f}. Its evidence was: "
            f"{top_unplanted['evidence_summary']}. I would treat this as a lead, not a false accusation: "
            "the original data has no ground-truth traffic-quality label."
        )
    else:
        surprise = (
            "No unlabeled test window crossed the review threshold. That sounds reassuring, but it may "
            "also mean the stress tests are easier than real abuse, so I would not read it as a production claim."
        )

    report = f"""# Investigation report

## What I wanted to understand

I wanted to see whether a small, inspectable workflow could separate suspicious behavioral changes from ordinary campaign volatility. I deliberately avoided asking the model to pronounce an event “fraud.” The useful decision here is whether a campaign window deserves another look.

## Data I worked with

I streamed **{_number(metadata['sample_rows'])}** timestamp-sorted rows from Criteo's public attribution dataset. This local sample covers the first **{sample_days:.2f} days** of the 30-day source and contains **{_number(metadata['clicks'])} clicks** and **{_number(metadata['conversions'])} conversion-linked impressions**.

The source has no invalid-traffic labels. I left all observed rows unchanged and added **{_number(simulation_rows)}** clearly marked experimental rows across **{len(manifest)} episodes**. Three behaviors were intended as abuse stress tests; a popularity spike was included as a negative control.

After aggregation, the unit of analysis was a campaign in a 30-minute window. There were **{_number(len(features))} windows** with at least five impressions.

## How I investigated it

1. I loaded the event stream into SQLite and built behavioral features in SQL.
2. I summarized volume, click rate, conversions per click, repeated-user activity, source concentration, click timing, and transformed cost.
3. I split the data chronologically at timestamp **{_number(result['cutoff'])}**. Nothing from the later period was used to fit feature scaling or model weights.
4. I compared a robust anomaly percentile with a class-weighted logistic model.
5. I combined them into a cautious hybrid score: 70% supervised probability and 30% anomaly percentile.
6. I converted the score into `monitor`, `review`, `temporary_throttle_candidate`, or `escalate_for_manual_decision`. These are offline recommendations, not actions taken against anyone.

## Results on the held-out time period

| Approach | Precision | Recall | F1 | Average precision | False-positive rate |
|---|---:|---:|---:|---:|---:|
{_metric_row('Robust anomaly baseline', metrics['anomaly_baseline_test'])}{_metric_row('Supervised logistic model', metrics['supervised_test'])}{_metric_row('Hybrid review score', hybrid)}
{model_decision}

The hybrid review threshold was **{review_threshold:.3f}**. It surfaced **{hybrid['true_positives']} of {hybrid['true_positives'] + hybrid['false_negatives']}** planted abuse windows and queued **{hybrid['false_positives']}** windows without a planted abuse label.

The adaptive low-and-slow scenario appeared only in the held-out period. **{adaptive_caught} of {adaptive_positives}** of its positive campaign-windows crossed the review threshold. The clean popularity-spike control produced **{clean_spike_queued}** queued windows.

That negative-control result is important: unusual volume can be legitimate, and the current model sometimes overreacts to it. I would add campaign-change context and a campaign-specific baseline before trusting a volume-driven alert.

## What caught my attention

{surprise}

The model coefficients are not causal explanations. They tell me which standardized features moved the fitted score, while the evidence summary gives a more concrete starting point for review. I would want corroborating information—campaign changes, network or user-agent patterns, historical source behavior, and downstream quality—before taking a restrictive action.

## The decision I would make

I would use this workflow to prioritize a finite review queue. I would not allow the current score to issue a permanent block because:

- the “positive” labels are simulated behaviors;
- unlabeled Criteo traffic is not verified-clean traffic;
- the local sample is an early chronological slice;
- several contextual variables are anonymized;
- the cost field is transformed and cannot be presented as dollars protected.

A reasonable next experiment would replay the scenarios over the full 30 days, hold out entire campaigns rather than only time, and have a second reviewer assess cases without seeing the simulation label. I would also monitor alert volume and feature drift before considering even temporary automated throttling.

## Reproducibility notes

- Random seed: `{json.dumps(42)}`
- Sample SHA-256: `{metadata['sha256']}`
- Source sampling: {metadata['sampling_method']}
- Source license: {metadata['license']}
- Evaluation labels remain in `evaluation_predictions.csv`; the analyst-facing `review_queue.csv` intentionally hides them.
"""
    (reports_dir / "investigation_report.md").write_text(report, encoding="utf-8")


def write_model_card(reports_dir: Path, result: dict) -> None:
    metrics = result["metrics"]
    artifact = result["model_artifact"]
    coefficients = result["coefficients"]
    coefficient_lines = "\n".join(
        f"| `{row.feature}` | {row.coefficient:.4f} |"
        for row in coefficients.itertuples(index=False)
    )
    card = f"""# Model card

## Intended use

Prioritize campaign-time windows for manual investigation in this offline experiment.

## Out-of-scope use

This model should not identify people, label original Criteo traffic as fraudulent, estimate real monetary loss, or take permanent enforcement actions.

## Model

- Supervised component: {artifact['model']}
- Anomaly component: robust feature distance converted to an empirical percentile
- Combination: `{artifact['hybrid_formula']}`
- Chronological test cutoff: `{result['cutoff']}` seconds
- Review threshold: `{metrics['thresholds']['review']:.4f}`

## Held-out experimental performance

- Precision: {_pct(metrics['hybrid_test']['precision'])}
- Recall: {_pct(metrics['hybrid_test']['recall'])}
- Average precision: `{metrics['hybrid_test']['average_precision']:.3f}`
- False-positive rate: {_pct(metrics['hybrid_test']['false_positive_rate'])}

These figures measure recovery of planted experimental scenarios. They are not estimates of real-world fraud-detection performance.

## Coefficients

Coefficients are based on standardized features. Magnitude indicates association with the fitted score, not causality.

| Feature | Coefficient |
|---|---:|
{coefficient_lines}

## Known risks

- Simulation-to-reality gap
- Unverified negative class
- Early-period sampling bias
- Possible campaign-specific memorization
- Anonymized contextual features
- Threshold instability under changing traffic

## Safeguards

- Human review is the default intervention.
- Simulation labels are hidden from the operational review queue.
- Every alert contains a plain-language evidence summary.
- Restrictive actions are described as candidates, not automatically executed.
- Permanent blocking is not supported.
"""
    (reports_dir / "model_card.md").write_text(card, encoding="utf-8")


def write_case_notes(reports_dir: Path, result: dict, limit: int = 5) -> None:
    cases_dir = reports_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    queue = result["review_queue"].head(limit)
    for position, row in enumerate(queue.itertuples(index=False), start=1):
        case = f"""# Case {position:02d}: campaign {int(row.campaign)}

## Why this reached the queue

Risk score: **{row.risk_score:.3f}**
Suggested next step: **{str(row.recommended_action).replace('_', ' ')}**

{row.evidence_summary.capitalize()}.

## Window summary

- Timestamp window start: `{int(row.window_start)}` seconds from the beginning of the source
- Impressions: `{int(row.impressions):,}`
- Clicks: `{int(row.clicks):,}`
- Conversions: `{int(row.conversions):,}`
- CTR: `{row.ctr:.3f}`
- Conversions per click: `{row.conversion_per_click:.3f}`
- Repeat-event share: `{row.repeat_event_share:.3f}`
- Largest anonymized-source share: `{row.top_source_share:.3f}`
- Transformed cost units: `{row.total_cost_units:.4f}`

## What I would check next

1. Compare the campaign with its own history rather than only a global baseline.
2. Check whether the volume change matches a launch, budget change, or seasonal event.
3. Look for corroborating network, user-agent, geography, and downstream-quality signals.
4. Review neighboring windows to see whether the pattern persists or disappears.
5. Document an innocent explanation before considering a restrictive response.

## Current judgment

This is a lead for review, not proof of abuse. The evidence is behavioral and the source identities are anonymized.
"""
        (cases_dir / f"case_{position:02d}_campaign_{int(row.campaign)}.md").write_text(
            case, encoding="utf-8"
        )


def _table_rows(frame: pd.DataFrame, columns: list[tuple[str, str]], limit: int = 20) -> str:
    rows: list[str] = []
    for _, item in frame.head(limit).iterrows():
        cells = "".join(
            f"<td>{html.escape(formatter.format(item[key]))}</td>" for key, formatter in columns
        )
        rows.append(f"<tr>{cells}</tr>")
    return "".join(rows)


def write_dashboard(reports_dir: Path, metadata: dict, result: dict) -> None:
    metrics = result["metrics"]
    hybrid = metrics["hybrid_test"]
    queue = result["review_queue"].copy()
    scenarios = result["scenario_summary"].copy()
    coefficients = result["coefficients"].copy()
    max_coefficient = max(float(coefficients["coefficient"].abs().max()), 1e-9)

    scenario_rows = _table_rows(
        scenarios,
        [
            ("evaluation_scenario", "{}"),
            ("windows", "{:,.0f}"),
            ("mean_risk", "{:.3f}"),
            ("max_risk", "{:.3f}"),
            ("queued_windows", "{:,.0f}"),
        ],
    )
    alert_rows = _table_rows(
        queue,
        [
            ("campaign", "{:.0f}"),
            ("risk_score", "{:.3f}"),
            ("recommended_action", "{}"),
            ("impressions", "{:,.0f}"),
            ("ctr", "{:.3f}"),
            ("evidence_summary", "{}"),
        ],
        limit=12,
    )
    coefficient_bars = "".join(
        f"<div class='bar-row'><span>{html.escape(str(row.feature))}</span>"
        f"<div class='track'><i style='width:{100 * abs(float(row.coefficient)) / max_coefficient:.1f}%'></i></div>"
        f"<strong>{float(row.coefficient):+.3f}</strong></div>"
        for row in coefficients.itertuples(index=False)
    )

    dashboard = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Signals in the Noise</title>
<style>
:root{{--ink:#17211c;--muted:#637069;--paper:#f7f4ed;--card:#fffdf8;--green:#225c46;--lime:#c8e36b;--line:#d9ddd5;--rust:#a94d32}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:1180px;margin:auto;padding:54px 28px 80px}} .eyebrow{{letter-spacing:.14em;text-transform:uppercase;color:var(--green);font-weight:750;font-size:12px}}
h1{{font-family:Georgia,serif;font-size:clamp(42px,7vw,76px);line-height:.96;margin:12px 0 18px;max-width:800px}} .lede{{max-width:760px;color:var(--muted);font-size:18px}}
.note{{border-left:4px solid var(--lime);padding:10px 16px;background:#eef2df;margin:28px 0 34px;max-width:900px}}
.cards{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:30px 0}} .card{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px}}
.card small{{display:block;color:var(--muted)}} .card b{{display:block;font:700 29px/1.2 Georgia,serif;margin-top:5px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:18px}} section{{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:24px;overflow:auto}}
section.wide{{grid-column:1/-1}} h2{{font-family:Georgia,serif;margin:0 0 8px;font-size:25px}} p{{color:var(--muted)}} table{{width:100%;border-collapse:collapse;min-width:620px}}
th,td{{text-align:left;padding:10px 9px;border-bottom:1px solid var(--line);vertical-align:top}} th{{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}}
.bar-row{{display:grid;grid-template-columns:190px 1fr 62px;gap:10px;align-items:center;margin:10px 0;font-size:13px}} .track{{height:9px;background:#e7e8e1;border-radius:8px;overflow:hidden}} .track i{{height:100%;display:block;background:var(--green)}}
footer{{margin-top:28px;color:var(--muted);font-size:13px}} @media(max-width:850px){{.cards{{grid-template-columns:1fr 1fr}}.grid{{grid-template-columns:1fr}}section.wide{{grid-column:auto}}}}
</style></head><body><main>
<div class="eyebrow">Personal data investigation</div><h1>Signals in the Noise</h1>
<div class="lede">I explored how far a transparent model can go when unusual ad traffic has several innocent explanations. The result is a review queue, not an automatic fraud verdict.</div>
<div class="note"><b>Important boundary:</b> the source has no fraud labels. Performance below measures recovery of documented scenarios I planted for an offline stress test.</div>
<div class="cards">
<div class="card"><small>Source rows</small><b>{_number(metadata['sample_rows'])}</b></div>
<div class="card"><small>Test average precision</small><b>{hybrid['average_precision']:.3f}</b></div>
<div class="card"><small>Review precision</small><b>{_pct(hybrid['precision'])}</b></div>
<div class="card"><small>Review recall</small><b>{_pct(hybrid['recall'])}</b></div>
<div class="card"><small>Queued windows</small><b>{_number(len(queue))}</b></div>
</div>
<div class="grid">
<section><h2>Experimental behavior by scenario</h2><p>The scenario field is used only for evaluation; it is hidden from the review queue.</p>
<table><thead><tr><th>Scenario</th><th>Windows</th><th>Mean risk</th><th>Max risk</th><th>Queued</th></tr></thead><tbody>{scenario_rows}</tbody></table></section>
<section><h2>What moved the fitted score</h2><p>Absolute bar length shows coefficient magnitude after standardization. This is association, not causality.</p>{coefficient_bars}</section>
<section class="wide"><h2>First cases I would inspect</h2><p>Evidence summaries explain why a window surfaced without revealing its experimental label.</p>
<table><thead><tr><th>Campaign</th><th>Risk</th><th>Action</th><th>Impressions</th><th>CTR</th><th>Evidence</th></tr></thead><tbody>{alert_rows}</tbody></table></section>
</div><footer>Generated by the reproducible pipeline. Criteo source data is CC BY-NC-SA 4.0; cost values are transformed units, not dollars.</footer>
</main></body></html>"""
    (reports_dir / "dashboard.html").write_text(dashboard, encoding="utf-8")


def generate_reports(
    reports_dir: Path,
    metadata: dict,
    manifest: pd.DataFrame,
    features: pd.DataFrame,
    result: dict,
) -> None:
    write_investigation_report(reports_dir, metadata, manifest, features, result)
    write_model_card(reports_dir, result)
    write_case_notes(reports_dir, result)
    write_dashboard(reports_dir, metadata, result)
    # These compact artifacts make the included example run inspectable without
    # redistributing the source event data or the local SQLite database.
    manifest.to_csv(reports_dir / "simulation_manifest.csv", index=False)
    result["scenario_summary"].to_csv(reports_dir / "scenario_summary.csv", index=False)
    result["coefficients"].to_csv(reports_dir / "model_coefficients.csv", index=False)
    (reports_dir / "metrics.json").write_text(
        json.dumps(result["metrics"], indent=2), encoding="utf-8"
    )
    (reports_dir / "model_artifact.json").write_text(
        json.dumps(result["model_artifact"], indent=2), encoding="utf-8"
    )
    publish_columns = [
        "window_start", "campaign", "risk_score", "recommended_action", "impressions",
        "clicks", "conversions", "ctr", "conversion_per_click", "repeat_event_share",
        "top_source_share", "active_click_minute_share", "total_cost_units",
        "evidence_summary", "data_period",
    ]
    result["review_queue"][publish_columns].to_csv(
        reports_dir / "review_queue.csv", index=False
    )
