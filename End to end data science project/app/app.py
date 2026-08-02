"""Streamlit dashboard for verified SupplyLens artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "reports/metrics/final_metrics.json"
TABLES = ROOT / "reports/tables"
FIGURES = ROOT / "reports/figures"

st.set_page_config(
    page_title="SupplyLens",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --ink:#12263A; --blue:#2E86AB; --teal:#2A9D8F; --sand:#F7F3EA; }
    .stApp { background: linear-gradient(180deg, #F7FAFC 0%, #FFFFFF 36%); }
    .block-container { padding-top: 1.6rem; max-width: 1480px; }
    .hero {
        padding: 1.6rem 1.8rem; border-radius: 18px;
        background: linear-gradient(120deg, #12263A 0%, #1D526B 62%, #2A9D8F 100%);
        color: white; margin-bottom: 1.2rem; box-shadow: 0 12px 32px rgba(18,38,58,.16);
    }
    .hero h1 { margin:0; font-size:2.25rem; letter-spacing:-.03em; }
    .hero p { margin:.55rem 0 0; opacity:.88; max-width:850px; }
    [data-testid="stMetric"] {
        background:white; border:1px solid #E4EBF0; padding:1rem 1.05rem;
        border-radius:14px; box-shadow:0 4px 16px rgba(18,38,58,.05);
    }
    [data-testid="stMetricLabel"] { color:#52606D; }
    [data-testid="stMetricValue"] { color:#12263A; }
    .definition { color:#66788A; font-size:.9rem; }
    .notice { border-left:4px solid #E9C46A; background:#FFF9E8; padding:.75rem 1rem; border-radius:8px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_csv(path: Path, dates: tuple[str, ...] = ()) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in dates:
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def require_file(path: Path, command: str) -> bool:
    if path.exists():
        return True
    st.error(f"Required artifact is missing: `{path.relative_to(ROOT)}`")
    st.code(command)
    return False


st.markdown(
    """
    <div class="hero">
      <h1>SupplyLens</h1>
      <p>Supplier delivery risk and operational decision intelligence for public-health commodity shipments. Rankings use calibrated probabilities and a fixed review-capacity policy.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not require_file(METRICS_PATH, "python scripts/train.py && python scripts/build_reports.py"):
    st.stop()

metrics = load_json(METRICS_PATH)
test_metrics = metrics["test_metrics"]
policy = metrics["operating_policy"]["test_result"]

with st.sidebar:
    st.header("Artifact status")
    st.success("Verified project outputs loaded")
    st.caption("All displayed values are read from generated project artifacts.")
    st.divider()
    st.subheader("Definitions")
    st.markdown(
        "**Severe delay**  \nMore than 7 calendar days after the scheduled delivery date."
    )
    st.markdown(
        "**Review policy**  \nTop 20% of shipments by calibrated risk in each scoring batch."
    )
    st.markdown(
        "**Prediction point**  \nScheduled-delivery commitment, before actual delivery is known."
    )

overview, queue_tab, supplier_tab, model_tab, segment_tab, monitoring_tab, limits_tab = st.tabs(
    [
        "Executive overview",
        "Intervention queue",
        "Supplier scorecard",
        "Model performance",
        "Segment analysis",
        "Monitoring",
        "Limitations",
    ]
)

with overview:
    columns = st.columns(5)
    columns[0].metric("Test PR-AUC", f"{test_metrics['pr_auc']:.3f}")
    columns[1].metric("Test ROC-AUC", f"{test_metrics['roc_auc']:.3f}")
    columns[2].metric("Top-20% precision", f"{policy['precision']:.1%}")
    columns[3].metric("Top-20% recall", f"{policy['recall']:.1%}")
    columns[4].metric("Top-20% lift", f"{policy['lift']:.2f}×")
    st.caption(
        "Final test period: "
        f"{metrics['splits']['test']['start']} to {metrics['splits']['test']['end']} · "
        f"{metrics['splits']['test']['rows']:,} shipments · "
        f"{metrics['splits']['test']['prevalence']:.1%} severe-delay prevalence"
    )
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Risk-ranking performance")
        if (FIGURES / "gains_and_lift.png").exists():
            st.image(str(FIGURES / "gains_and_lift.png"), use_container_width=True)
    with right:
        st.subheader("Temporal design")
        if (FIGURES / "temporal_split_timeline.png").exists():
            st.image(str(FIGURES / "temporal_split_timeline.png"), use_container_width=True)
        st.markdown(
            '<div class="notice"><b>Decision use:</b> prioritize a scarce review queue. '
            "The score does not authorize automatic supplier penalties or operational action.</div>",
            unsafe_allow_html=True,
        )

with queue_tab:
    queue_path = TABLES / "shipment_intervention_queue.csv"
    if require_file(queue_path, "python scripts/train.py"):
        queue = load_csv(queue_path, dates=("scheduled_delivery_date",))
        st.subheader("Ranked shipment review queue")
        filter_a, filter_b, filter_c = st.columns(3)
        suppliers = filter_a.multiselect(
            "Supplier", sorted(queue["supplier"].dropna().unique()), placeholder="All suppliers"
        )
        countries = filter_b.multiselect(
            "Destination", sorted(queue["country"].dropna().unique()), placeholder="All destinations"
        )
        modes = filter_c.multiselect(
            "Shipment mode", sorted(queue["shipment_mode"].dropna().unique()), placeholder="All modes"
        )
        filtered = queue.copy()
        if suppliers:
            filtered = filtered[filtered["supplier"].isin(suppliers)]
        if countries:
            filtered = filtered[filtered["country"].isin(countries)]
        if modes:
            filtered = filtered[filtered["shipment_mode"].isin(modes)]
        review_only = st.toggle("Show review-flagged shipments only", value=True)
        if review_only:
            filtered = filtered[filtered["review_flag"] == 1]
        q1, q2, q3 = st.columns(3)
        q1.metric("Visible shipments", f"{len(filtered):,}")
        q2.metric(
            "Average predicted risk",
            f"{filtered['predicted_severe_delay_probability'].mean():.1%}" if len(filtered) else "—",
        )
        q3.metric(
            "Represented value",
            f"${filtered['total_value_usd'].sum()/1_000_000:.1f}M" if len(filtered) else "—",
        )
        display = filtered.rename(
            columns={
                "shipment_id": "Shipment",
                "supplier": "Supplier",
                "country": "Destination",
                "shipment_mode": "Mode",
                "scheduled_delivery_date": "Scheduled",
                "predicted_severe_delay_probability": "Risk",
                "risk_rank": "Rank",
                "total_value_usd": "Value (USD)",
                "top_prediction_contributors": "Top contributors",
            }
        )
        st.dataframe(
            display[
                [
                    "Rank",
                    "Shipment",
                    "Supplier",
                    "Destination",
                    "Mode",
                    "Scheduled",
                    "Risk",
                    "Value (USD)",
                    "Top contributors",
                ]
            ],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Risk": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=1),
                "Value (USD)": st.column_config.NumberColumn(format="$%,.0f"),
                "Scheduled": st.column_config.DateColumn(format="YYYY-MM-DD"),
            },
        )

with supplier_tab:
    scorecard_path = TABLES / "supplier_scorecard.csv"
    if require_file(scorecard_path, "python scripts/train.py"):
        scorecard = load_csv(scorecard_path)
        st.subheader("Test-period supplier decision support")
        st.caption(
            "Low-volume estimates are marked ineligible. The priority score supports review ordering; it is not a penalty score."
        )
        eligible_only = st.toggle("Minimum-volume eligible only", value=True, key="supplier_eligible")
        visible = scorecard[
            scorecard["minimum_volume_eligible"] if eligible_only else pd.Series(True, index=scorecard.index)
        ].sort_values("performance_priority_score", ascending=False)
        fig = px.scatter(
            visible,
            x="shipment_count",
            y="severe_delay_rate",
            size="total_represented_shipment_value_usd",
            color="average_predicted_risk",
            hover_name="supplier",
            color_continuous_scale=["#DCEAF0", "#2E86AB", "#E76F51"],
            labels={
                "shipment_count": "Test shipments",
                "severe_delay_rate": "Observed severe-delay rate",
                "average_predicted_risk": "Average risk",
            },
        )
        fig.update_layout(height=480, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(visible, use_container_width=True, hide_index=True)

with model_tab:
    st.subheader("Model comparison and calibration")
    model_comparison = load_csv(TABLES / "model_comparison.csv")
    calibration = load_csv(TABLES / "calibration_comparison.csv")
    capacity = load_csv(TABLES / "capacity_analysis_test.csv")
    a, b = st.columns(2)
    with a:
        st.markdown("**Validation model selection**")
        st.dataframe(model_comparison.round(4), use_container_width=True, hide_index=True)
        st.caption(metrics["model_selection"]["selection_reason"])
    with b:
        st.markdown("**Validation calibration selection**")
        st.dataframe(calibration.round(4), use_container_width=True, hide_index=True)
        st.caption(f"Selected method: {metrics['calibration']['selected_method']}")
    curve_left, curve_right = st.columns(2)
    curve_left.image(str(FIGURES / "pr_and_roc_curves.png"), use_container_width=True)
    curve_right.image(str(FIGURES / "calibration_plot.png"), use_container_width=True)
    st.subheader("Review-capacity tradeoff")
    capacity_display = capacity.copy()
    capacity_display["capacity_label"] = capacity_display["capacity"].map(lambda value: f"Top {value:.0%}")
    fig = px.line(
        capacity_display,
        x="capacity",
        y=["precision", "recall"],
        markers=True,
        labels={"capacity": "Review capacity", "value": "Metric", "variable": "Metric"},
        color_discrete_map={"precision": "#2E86AB", "recall": "#E76F51"},
    )
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(capacity.round(4), use_container_width=True, hide_index=True)

with segment_tab:
    segment_path = TABLES / "segment_performance.csv"
    if require_file(segment_path, "python scripts/train.py"):
        segments = load_csv(segment_path)
        st.subheader("Segment and error analysis")
        segment_type = st.selectbox(
            "Segment dimension", sorted(segments["segment_type"].unique())
        )
        reliable_only = st.toggle("Reliable-volume segments only", value=True)
        visible = segments[segments["segment_type"] == segment_type]
        if reliable_only:
            visible = visible[visible["reliable_volume"]]
        visible = visible.sort_values("sample_size", ascending=False)
        st.dataframe(visible, use_container_width=True, hide_index=True)
        errors = load_csv(TABLES / "error_examples.csv")
        st.markdown("**Illustrative final-test errors**")
        st.dataframe(errors, use_container_width=True, hide_index=True)

with monitoring_tab:
    monitoring_path = ROOT / "reports/metrics/monitoring_drift_example.json"
    if require_file(monitoring_path, "python scripts/build_reports.py"):
        monitoring = load_json(monitoring_path)
        st.subheader("Train-to-test drift example")
        m1, m2, m3 = st.columns(3)
        m1.metric("Prediction PSI", f"{monitoring['prediction_psi_train_to_test']:.3f}")
        m2.metric(
            "Supplier mix distance",
            f"{monitoring['category_frequency_distance_train_to_test']['supplier']:.3f}",
        )
        m3.metric(
            "Test queue rate", f"{monitoring['test_review_queue_rate']:.1%}"
        )
        st.markdown(
            '<div class="notice">These are retrospective examples from the temporal test period, not live production alerts.</div>',
            unsafe_allow_html=True,
        )
        numeric = pd.DataFrame.from_dict(
            monitoring["numeric_psi_train_to_test"], orient="index", columns=["PSI"]
        ).reset_index(names="feature")
        category = pd.DataFrame.from_dict(
            monitoring["category_frequency_distance_train_to_test"],
            orient="index",
            columns=["frequency distance"],
        ).reset_index(names="feature")
        left, right = st.columns(2)
        left.dataframe(numeric.round(3), use_container_width=True, hide_index=True)
        right.dataframe(category.round(3), use_container_width=True, hide_index=True)

with limits_tab:
    st.subheader("Responsible interpretation")
    st.markdown(
        """
        - The dataset covers health-commodity shipments scheduled from 2006–2015; relationships may not transfer to current operations.
        - The source does not record when a scheduled commitment was entered. Target-derived historical aggregates are therefore excluded from the production feature set.
        - Supplier and route identifiers may represent operational structures specific to the source program.
        - The test PR-AUC is modest, and review rankings should augment—not replace—operations judgment.
        - Supplier scorecards are decision support and must not be used as automatic penalty systems.
        - The impact calculator uses explicit cost and intervention-success assumptions; it does not estimate realized savings.
        - Fulfillment-method diagnostics are observational. Limited overlap and residual imbalance prevent a causal conclusion.
        """
    )
    lead = metrics["lead_time_experiment"]
    st.markdown(
        f"**Preserved negative finding:** the learned P50 lead-time model achieved "
        f"{next(item['mae_days'] for item in lead['test_metrics'] if item['model']=='learned_p50'):.2f} days MAE, "
        f"versus {next(item['mae_days'] for item in lead['test_metrics'] if item['model']=='scheduled_lead_time'):.2f} days for the scheduled baseline. "
        "The learned model is not recommended."
    )

st.caption("SupplyLens · Personal portfolio project by Sagar Pandya")
