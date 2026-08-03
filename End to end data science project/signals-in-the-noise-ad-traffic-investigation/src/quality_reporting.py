from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd


def _number(value: float) -> str:
    return f"{int(value):,}"


def _rate(value: float) -> str:
    return f"{100 * float(value):.2f}%"


def write_observed_quality_report(
    reports_dir: Path,
    metadata: dict,
    observed_features: pd.DataFrame,
    result: dict,
) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics = result["metrics"]
    click_metrics = metrics["expected_click_rate_test"]
    conversion_metrics = metrics["expected_conversion_rate_test"]
    split = metrics["data_split"]
    queue = result["review_queue"]
    sample_days = float(metadata["max_timestamp_seconds"]) / 86_400

    if queue.empty:
        queue_note = (
            "No held-out window crossed the review cutoff. I would check the cutoff against a longer "
            "replay before deciding that the traffic was quiet; absence of an alert is not proof of quality."
        )
    else:
        top = queue.iloc[0]
        queue_note = (
            f"The first case I would open is campaign `{int(top['campaign'])}` at timestamp "
            f"`{int(top['window_start'])}`. It reached a quality-risk percentile of "
            f"**{float(top['quality_risk_score']):.3f}** because {top['evidence_summary']}."
        )

    report = f"""# Observed traffic quality investigation

## The question I asked

After building the controlled stress test, I wanted a second analysis that did not depend on planted examples at all:

> Using earlier campaign history and the window's non-outcome context, were its clicks and conversion-linked impressions close to what I would have expected?

This is the part of the project I would use to discover leads in the public data. It does not assign a fraud label. A large residual tells me that the model was surprised; it does not tell me why.

## The data I kept

I started with **{_number(metadata['sample_rows'])}** source rows covering the first **{sample_days:.2f} days** of Criteo's timestamp-sorted file. I removed every campaign-window touched by my simulation layer before fitting this analysis. That left **{_number(len(observed_features))} observed-only campaign windows**.

The source has click and conversion outcomes, but it has no invalid-traffic truth, device evidence, network identifiers, or policy decisions. I treated those missing fields as a boundary, not something a model could fill in.

## How I worked through it

1. I built 30-minute campaign summaries in SQL.
2. For each campaign, I calculated rolling baselines from its previous 20 available windows. The current and future windows were excluded.
3. I split the sample chronologically at timestamp **{_number(result['cutoff'])}**. The first **{_number(split['train_windows'])}** windows were used to fit expected behavior; the remaining **{_number(split['test_windows'])}** windows were kept for replay.
4. I fitted one binomial model for click rate and another for conversion-linked impression rate. The second model can use the clicks already observed in that window because it is answering a later-stage quality question.
5. I converted click and conversion residuals into standardized surprises.
6. I combined those surprises with campaign-relative volume, user/source concentration, and transformed cost exposure. Each part was converted to a percentile against the earlier training period.
7. I queued only held-out windows with a quality-risk score at or above **{metrics['review_queue']['threshold']:.3f}**, a training-period percentile cutoff of {100 * metrics['review_queue']['threshold']:.1f}%. The queue is for investigation, not enforcement.

## Did the expected-behavior models travel forward in time?

| Held-out check | Click-rate model | Conversion-rate model |
|---|---:|---:|
| Observed rate | {_rate(click_metrics['observed_rate'])} | {_rate(conversion_metrics['observed_rate'])} |
| Mean expected rate | {_rate(click_metrics['mean_expected_rate'])} | {_rate(conversion_metrics['mean_expected_rate'])} |
| Weighted mean absolute error | {click_metrics['weighted_mean_absolute_error']:.5f} | {conversion_metrics['weighted_mean_absolute_error']:.5f} |
| Weighted Brier score | {click_metrics['weighted_brier_score']:.6f} | {conversion_metrics['weighted_brier_score']:.6f} |
| Weighted log loss | {click_metrics['weighted_log_loss']:.5f} | {conversion_metrics['weighted_log_loss']:.5f} |

These are calibration and forecasting checks, not detection accuracy. There are no real abuse labels with which to calculate precision or recall.

## What reached the queue

The held-out replay produced **{_number(len(queue))} review cases** after applying the percentile cutoff and a maximum queue size of **{metrics['review_queue']['maximum_published_windows']}**.

{queue_note}

I would read each evidence summary as a starting hypothesis. Before any restrictive decision, I would want campaign-change context, longer history, network and user-agent evidence, neighboring-window persistence, and an innocent explanation check.

## What I learned

Campaign history made the investigation more useful than a global outlier score. The same click rate can be routine for one campaign and surprising for another. Residuals also separated two questions that are easy to blur together: whether interaction was unusual, and whether downstream outcomes were weaker than expected after that interaction.

The remaining weakness is the short early-period sample. Some campaigns have little history, and the chronological replay does not cover the source's full 30 days. I would next run the entire dataset, add day-of-week effects, and measure whether the queue stays stable as traffic changes.

## Decision boundary

- An unexpected window is not automatically harmful.
- A weak conversion result is not proof that a click was invalid.
- The score prioritizes review; it does not block, throttle, or accuse anyone.
- Criteo's cost field is transformed, so the project reports cost units rather than dollars.
- Performance from the separate controlled stress test should not be presented as real-world fraud performance.
"""
    (reports_dir / "observed_quality_report.md").write_text(report, encoding="utf-8")


def write_observed_case_notes(reports_dir: Path, result: dict, limit: int = 5) -> None:
    cases_dir = reports_dir / "observed_cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    for position, row in enumerate(result["review_queue"].head(limit).itertuples(index=False), start=1):
        volume = "not available" if pd.isna(row.volume_change_ratio) else f"{row.volume_change_ratio:.2f}x"
        case = f"""# Observed case {position:02d}: campaign {int(row.campaign)}

## Why I opened this window

Quality-risk percentile: **{row.quality_risk_score:.3f}**

{row.evidence_summary.capitalize()}.

## Expected versus observed

| Measure | Observed | Expected |
|---|---:|---:|
| Clicks | {int(row.clicks):,} | {row.expected_clicks:.1f} |
| Click rate | {row.ctr:.3%} | {row.expected_ctr:.3%} |
| Conversion-linked impressions | {int(row.conversions):,} | {row.expected_conversions:.1f} |
| Conversion rate | {row.conversion_rate:.3%} | {row.expected_conversion_rate:.3%} |

- Click residual: `{row.click_deviation_z:.2f}` standard deviations
- Conversion residual: `{row.conversion_deviation_z:.2f}` standard deviations
- Volume versus recent campaign baseline: `{volume}`
- Earlier campaign windows available: `{int(row.history_windows)}`
- Returning-user event share: `{row.repeat_event_share:.3f}`
- Largest anonymized-source share: `{row.top_source_share:.3f}`
- Transformed cost units: `{row.total_cost_units:.4f}`

## What I would check next

1. Read the campaign's neighboring windows to see whether the change persisted.
2. Check for a launch, budget adjustment, creative change, or seasonal event.
3. Compare source, network, geography, device, and user-agent patterns if those fields are available internally.
4. Look at longer-horizon downstream outcomes before judging traffic quality.
5. Record an innocent explanation and the evidence that would disprove it.

## Current judgment

This window is unusual relative to the model and recent campaign history. The public data is not sufficient to determine whether the reason is harmful or legitimate.
"""
        (cases_dir / f"case_{position:02d}_campaign_{int(row.campaign)}.md").write_text(
            case, encoding="utf-8"
        )


def _queue_rows(queue: pd.DataFrame, limit: int = 15) -> str:
    rows: list[str] = []
    for row in queue.head(limit).itertuples(index=False):
        rows.append(
            "<tr>"
            f"<td>{int(row.campaign)}</td>"
            f"<td>{row.quality_risk_score:.3f}</td>"
            f"<td>{int(row.clicks):,} / {row.expected_clicks:.1f}</td>"
            f"<td>{int(row.conversions):,} / {row.expected_conversions:.1f}</td>"
            f"<td>{html.escape(str(row.evidence_summary))}</td>"
            "</tr>"
        )
    if not rows:
        return "<tr><td colspan='5'>No held-out windows crossed the review cutoff.</td></tr>"
    return "".join(rows)


def _calibration_rows(calibration: pd.DataFrame, model: str) -> str:
    rows: list[str] = []
    selected = calibration.loc[calibration["model"] == model]
    for row in selected.itertuples(index=False):
        rows.append(
            "<tr>"
            f"<td>{int(row.decile) + 1}</td>"
            f"<td>{int(row.windows):,}</td>"
            f"<td>{100 * row.mean_expected_rate:.3f}%</td>"
            f"<td>{100 * row.actual_rate:.3f}%</td>"
            "</tr>"
        )
    return "".join(rows)


def write_observed_dashboard(reports_dir: Path, metadata: dict, result: dict) -> None:
    metrics = result["metrics"]
    split = metrics["data_split"]
    click = metrics["expected_click_rate_test"]
    conversion = metrics["expected_conversion_rate_test"]
    queue = result["review_queue"]
    queue_rows = _queue_rows(queue)
    click_rows = _calibration_rows(result["calibration"], "click_rate")
    conversion_rows = _calibration_rows(result["calibration"], "conversion_rate")

    dashboard = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Observed traffic quality | Signals in the Noise</title>
<style>
:root{{--ink:#15241d;--muted:#68746e;--paper:#f4f1e9;--card:#fffdf8;--forest:#1f5b45;--mint:#b9dfc8;--amber:#e5aa55;--line:#d8ddd7}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:1200px;margin:auto;padding:52px 26px 76px}}.eyebrow{{color:var(--forest);font-size:12px;font-weight:800;letter-spacing:.15em;text-transform:uppercase}}
h1{{font:700 clamp(42px,7vw,76px)/.98 Georgia,serif;max-width:850px;margin:12px 0 18px}}.lede{{font-size:18px;color:var(--muted);max-width:820px}}
.boundary{{background:#edf3e8;border-left:5px solid var(--forest);padding:14px 18px;margin:28px 0;border-radius:0 10px 10px 0;max-width:940px}}
.cards{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:28px 0}}.card,section{{background:var(--card);border:1px solid var(--line);border-radius:16px}}
.card{{padding:18px}}.card small{{display:block;color:var(--muted)}}.card b{{display:block;font:700 28px/1.2 Georgia,serif;margin-top:5px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}section{{padding:24px;overflow:auto}}section.wide{{grid-column:1/-1}}h2{{font:700 25px/1.2 Georgia,serif;margin:0 0 8px}}p{{color:var(--muted)}}
table{{border-collapse:collapse;width:100%;min-width:520px}}th,td{{padding:10px 9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}th{{font-size:11px;color:var(--muted);letter-spacing:.08em;text-transform:uppercase}}
.method{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:16px 0}}.method div{{background:#f0f3ec;border-radius:10px;padding:12px;font-size:13px}}footer{{color:var(--muted);margin-top:28px;font-size:13px}}
@media(max-width:900px){{.cards{{grid-template-columns:1fr 1fr}}.grid{{grid-template-columns:1fr}}section.wide{{grid-column:auto}}.method{{grid-template-columns:1fr 1fr}}}}
</style></head><body><main>
<div class="eyebrow">Observed-only investigation</div><h1>What did the model not expect?</h1>
<div class="lede">I used earlier campaign behavior to estimate clicks and conversion-linked impressions, then ranked the held-out windows whose outcomes departed most from those expectations.</div>
<div class="boundary"><b>Decision boundary:</b> surprise is not a fraud label. Every item below is a lead for human review, and every simulation-touched window was excluded from this analysis.</div>
<div class="cards">
<div class="card"><small>Source rows</small><b>{_number(metadata['sample_rows'])}</b></div>
<div class="card"><small>Observed windows</small><b>{_number(split['train_windows'] + split['test_windows'])}</b></div>
<div class="card"><small>Held-out windows</small><b>{_number(split['test_windows'])}</b></div>
<div class="card"><small>Click MAE</small><b>{click['weighted_mean_absolute_error']:.4f}</b></div>
<div class="card"><small>Review cases</small><b>{_number(len(queue))}</b></div>
</div>
<section class="wide"><h2>From history to a reviewable lead</h2><div class="method"><div><b>1. Summarize</b><br>Build 30-minute campaign windows in SQL.</div><div><b>2. Expect</b><br>Fit click and conversion rates on the earlier period.</div><div><b>3. Compare</b><br>Measure residuals against observed outcomes.</div><div><b>4. Review</b><br>Rank only the most unusual held-out windows.</div></div></section>
<div class="grid">
<section><h2>Click-rate calibration</h2><p>Expected and observed rates by held-out prediction decile.</p><table><thead><tr><th>Decile</th><th>Windows</th><th>Expected</th><th>Observed</th></tr></thead><tbody>{click_rows}</tbody></table></section>
<section><h2>Conversion-rate calibration</h2><p>The outcome is conversion-linked impressions divided by impressions.</p><table><thead><tr><th>Decile</th><th>Windows</th><th>Expected</th><th>Observed</th></tr></thead><tbody>{conversion_rows}</tbody></table></section>
<section class="wide"><h2>First windows I would inspect</h2><p>Observed/expected pairs make the residuals concrete. The evidence column is a hypothesis, not a verdict.</p><table><thead><tr><th>Campaign</th><th>Risk percentile</th><th>Clicks O/E</th><th>Conversions O/E</th><th>Why it surfaced</th></tr></thead><tbody>{queue_rows}</tbody></table></section>
</div><footer>Generated from the reproducible observed-only pipeline. Criteo data is CC BY-NC-SA 4.0; cost values are transformed units, not dollars.</footer>
</main></body></html>"""
    (reports_dir / "observed_quality_dashboard.html").write_text(dashboard, encoding="utf-8")


def generate_observed_quality_reports(
    reports_dir: Path,
    metadata: dict,
    observed_features: pd.DataFrame,
    result: dict,
) -> None:
    write_observed_quality_report(reports_dir, metadata, observed_features, result)
    write_observed_case_notes(reports_dir, result)
    write_observed_dashboard(reports_dir, metadata, result)
    result["review_queue"][result["review_columns"]].to_csv(
        reports_dir / "observed_review_queue.csv", index=False
    )
    result["coefficients"].to_csv(
        reports_dir / "observed_model_coefficients.csv", index=False
    )
    result["calibration"].to_csv(reports_dir / "observed_calibration.csv", index=False)
    (reports_dir / "observed_model_metrics.json").write_text(
        json.dumps(result["metrics"], indent=2), encoding="utf-8"
    )
    (reports_dir / "observed_model_artifact.json").write_text(
        json.dumps(result["model_artifact"], indent=2), encoding="utf-8"
    )
