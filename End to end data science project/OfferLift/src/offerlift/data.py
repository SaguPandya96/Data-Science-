"""Load and validate the Hillstrom email experiment."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "recency": "int",
    "history_segment": "str",
    "history": "float",
    "mens": "int",
    "womens": "int",
    "zip_code": "str",
    "newbie": "int",
    "channel": "str",
    "segment": "str",
    "visit": "int",
    "conversion": "int",
    "spend": "float",
}
PRE_TREATMENT_FEATURES = ["recency", "history", "mens", "womens", "newbie", "zip_code", "channel"]
BINARY_COLUMNS = ["mens", "womens", "newbie", "visit", "conversion"]


def load(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    validate(frame)
    return frame


def validate(frame: pd.DataFrame) -> None:
    """Raise if the frame breaks the assumptions every later step relies on."""
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    nulls = frame[list(REQUIRED_COLUMNS)].isna().sum()
    if nulls.any():
        raise ValueError(f"Null values found: {nulls[nulls > 0].to_dict()}")
    for column in BINARY_COLUMNS:
        if not frame[column].isin([0, 1]).all():
            raise ValueError(f"Column {column} must be 0/1")
    if (frame["spend"] < 0).any() or (frame["history"] < 0).any():
        raise ValueError("spend and history must be non-negative")
    if (frame.loc[frame["conversion"] == 1, "visit"] == 0).any():
        raise ValueError("Every conversion must follow a visit")
    if (frame.loc[frame["conversion"] == 0, "spend"] > 0).any():
        raise ValueError("Spend without a conversion")


def contrast(frame: pd.DataFrame, arm_column: str, control: str, treatment: str) -> pd.DataFrame:
    """Rows from one treatment arm and control, with a 0/1 ``treated`` column."""
    subset = frame[frame[arm_column].isin([control, treatment])].copy()
    subset["treated"] = (subset[arm_column] == treatment).astype(int)
    return subset.reset_index(drop=True)


def feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode pre-treatment features only; outcomes never enter the model."""
    return pd.get_dummies(
        frame[PRE_TREATMENT_FEATURES], columns=["zip_code", "channel"], dtype=float
    )
