"""OfferLift budget planner: who to email, and what it is worth compared with random."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from offerlift.planner import load_readout, plan, select_customers

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "reports/metrics"
FIGURES = ROOT / "reports/figures"

st.set_page_config(page_title="OfferLift budget planner", layout="centered")


@st.cache_data
def readout():
    return load_readout(METRICS)


@st.cache_data
def confirmation():
    return json.loads((METRICS / "confirmation.json").read_text())


effects = readout()
confirmed = confirmation()

st.title("OfferLift budget planner")
st.write(
    "A retailer's randomized email test showed that customers who bought **both men's and "
    "women's merchandise** last year respond about twice as strongly to the men's email. "
    "Set your list size and budget to see who to email and what it is worth."
)

planner_tab, results_tab, about_tab = st.tabs(["Plan a send", "Test results", "About"])

with planner_tab:
    list_size = st.number_input(
        "Customers on the list", min_value=1_000, max_value=10_000_000, value=100_000,
        step=1_000,
    )
    budget_pct = st.slider("Share of the list you can email (%)", 1, 100, 10)
    result = plan(effects, int(list_size), budget_pct / 100)

    st.subheader("Who gets the email")
    priority_share = effects.priority.share
    if result.other_emails:
        st.write(
            f"All **{result.priority_emails:,}** customers who bought both categories "
            f"(about {priority_share:.0%} of the list), then **{result.other_emails:,}** "
            "chosen at random from everyone else."
        )
    else:
        st.write(
            f"**{result.priority_emails:,}** customers who bought both categories last year, "
            "chosen at random from that group if it is larger than the budget."
        )
    if budget_pct / 100 > priority_share + 0.005:
        st.info(
            "Your budget is larger than the both-category group, so the remaining emails go "
            "to randomly chosen customers. No model beat random selection for that part of "
            "the list."
        )

    st.subheader("Expected extra site visits in two weeks")
    left, right = st.columns(2)
    left.metric(
        "Following the rule",
        f"{result.rule_visits:,.0f}",
        f"{result.extra_vs_random:+,.0f} vs random",
    )
    left.caption(f"95% range {result.rule_low:,.0f} to {result.rule_high:,.0f}")
    right.metric("Choosing at random", f"{result.random_visits:,.0f}")
    right.caption(f"95% range {result.random_low:,.0f} to {result.random_high:,.0f}")

    st.caption(
        "Estimates multiply the number of emails in each group by the email's effect on that "
        "group's visit rate in the randomized test, and the ranges carry that test's "
        "uncertainty. At a 10% budget the pre-committed confirmation test put the gain "
        f"over random at +{confirmed['gain_per_customers']:.1f} visits per 1,000 customers "
        f"(95% range {confirmed['ci_low']:.1f} to {confirmed['ci_high']:.1f})."
    )

    st.subheader("Build the email list")
    st.write(
        "Upload a CSV with one row per customer and 0/1 columns `mens` and `womens` "
        "(bought from that category in the last year). Other columns are kept as they are."
    )
    upload = st.file_uploader("Customer CSV", type="csv")
    if upload is not None:
        try:
            customers = pd.read_csv(upload)
            chosen = select_customers(customers, budget_pct / 100)
        except ValueError as error:
            st.error(str(error))
        else:
            n_email = int(chosen["send_email"].sum())
            n_both = int((chosen["send_email"] & chosen["bought_both"]).sum())
            st.success(
                f"{n_email:,} of {len(chosen):,} customers selected, "
                f"{n_both:,} of them both-category buyers."
            )
            st.dataframe(chosen.head(20), use_container_width=True)
            st.download_button(
                "Download the list",
                chosen.to_csv(index=False).encode(),
                file_name="offerlift_email_list.csv",
                mime="text/csv",
            )

with results_tab:
    st.image(str(FIGURES / "segment_lift_light.png"))
    st.image(str(FIGURES / "confirmation_light.png"))
    st.image(str(FIGURES / "budget_gain_light.png"))
    st.image(str(FIGURES / "gain_curves_light.png"))

with about_tab:
    st.markdown(
        """
**Data.** Hillstrom MineThatData email test (2008): 64,000 customers randomized into a
men's email, a women's email and no email, with visits, purchases and spend tracked for two
weeks.

**What was checked.** The groups were the planned size and looked alike before the email.
Three uplift models were compared against random targeting; the one result that beat it (a
10% budget) was confirmed with 20 repeated cross-fits, with the design fixed beforehand.
Explaining that model showed it picks almost exactly the both-category buyers.

**Limits.** One test, at one retailer, in 2008. The rule was re-checked on the same
customers, which rules out a lucky split but not something specific to this data. A small
confirmation send is the recommended next step.

Code, tests and the full readout:
[OfferLift on GitHub](https://github.com/SaguPandya96/Data-Science-/tree/master/End%20to%20end%20data%20science%20project/OfferLift)
"""
    )
