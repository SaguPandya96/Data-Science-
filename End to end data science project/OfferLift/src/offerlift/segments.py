"""Describe who a model targets, and check those groups against the randomized data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from offerlift.experiment import difference_in_means

PROFILE_NUMERIC = ["recency", "history", "mens", "womens", "newbie"]
PROFILE_CATEGORICAL = ["zip_code", "channel"]


def add_segments(frame: pd.DataFrame) -> pd.DataFrame:
    """Readable pre-treatment segments used to explain targeting."""
    out = frame.copy()
    out["purchased"] = np.select(
        [(out["mens"] == 1) & (out["womens"] == 1), out["mens"] == 1, out["womens"] == 1],
        ["both", "mens only", "womens only"],
        "neither",
    )
    out["history_band"] = np.where(out["history"] >= 200, "$200+", "under $200")
    out["recency_band"] = pd.cut(
        out["recency"], [0, 3, 6, 9, 12], labels=["1-3", "4-6", "7-9", "10-12"]
    ).astype(str)
    out["newbie_band"] = np.where(out["newbie"] == 1, "new", "returning")
    return out


SEGMENT_COLUMNS = ["purchased", "history_band", "recency_band", "newbie_band", "channel", "zip_code"]


def targeted_profile(frame: pd.DataFrame, targeted: np.ndarray) -> dict:
    """Mean of each numeric feature and share of each category, targeted vs the rest."""
    targeted = np.asarray(targeted).astype(bool)
    profile: dict = {"numeric": {}, "categorical": {}}
    for column in PROFILE_NUMERIC:
        profile["numeric"][column] = {
            "targeted": float(frame.loc[targeted, column].mean()),
            "rest": float(frame.loc[~targeted, column].mean()),
        }
    for column in PROFILE_CATEGORICAL:
        profile["categorical"][column] = {
            "targeted": frame.loc[targeted, column].value_counts(normalize=True).to_dict(),
            "rest": frame.loc[~targeted, column].value_counts(normalize=True).to_dict(),
        }
    return profile


def segment_lifts(frame: pd.DataFrame, column: str, outcome: str) -> list[dict]:
    """Treatment effect on ``outcome`` within each level of ``column``.

    ``frame`` must hold one treatment arm and control with a 0/1 ``treated`` column.
    """
    rows = []
    for level, group in frame.groupby(column, observed=True):
        estimate = difference_in_means(group[outcome], group["treated"])
        rows.append(
            {
                "segment": column,
                "level": str(level),
                "n": int(len(group)),
                "share": float(len(group) / len(frame)),
                "control_rate": estimate.control_mean,
                "lift": estimate.absolute_lift,
                "ci_low": estimate.ci_low,
                "ci_high": estimate.ci_high,
            }
        )
    return rows


def lift_difference(
    frame: pd.DataFrame, mask: np.ndarray, outcome: str, z: float = 1.959964
) -> dict:
    """Effect inside ``mask`` minus effect outside it, with a normal-approximation interval."""
    mask = np.asarray(mask).astype(bool)
    inside = difference_in_means(frame.loc[mask, outcome], frame.loc[mask, "treated"])
    outside = difference_in_means(frame.loc[~mask, outcome], frame.loc[~mask, "treated"])
    difference = inside.absolute_lift - outside.absolute_lift
    se = float(np.hypot(inside.standard_error, outside.standard_error))
    return {
        "lift_inside": inside.absolute_lift,
        "lift_outside": outside.absolute_lift,
        "difference": float(difference),
        "ci_low": float(difference - z * se),
        "ci_high": float(difference + z * se),
    }
