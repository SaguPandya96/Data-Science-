"""Turn uplift scores into a targeting decision under a mailing budget."""

from __future__ import annotations

import numpy as np
import pandas as pd

from offerlift.evaluation import uplift_at_fraction


def budget_table(
    scores: dict[str, np.ndarray],
    outcome: np.ndarray,
    treated: np.ndarray,
    fractions: list[float],
    per_customers: int = 1000,
) -> pd.DataFrame:
    """Expected incremental outcomes per ``per_customers`` customers for each policy.

    Random targeting earns the average effect on every mailed customer, so the
    model only adds value where its row beats the ``random`` row at the same budget.
    """
    outcome = np.asarray(outcome, dtype=float)
    treated = np.asarray(treated).astype(bool)
    average_effect = outcome[treated].mean() - outcome[~treated].mean()
    rows = []
    for fraction in fractions:
        mailed = fraction * per_customers
        rows.append(
            {
                "policy": "random",
                "budget_fraction": fraction,
                "uplift_in_targeted": average_effect,
                "incremental_per_customers": mailed * average_effect,
            }
        )
        for name, score in scores.items():
            lift = uplift_at_fraction(score, outcome, treated, fraction)
            rows.append(
                {
                    "policy": name,
                    "budget_fraction": fraction,
                    "uplift_in_targeted": lift,
                    "incremental_per_customers": mailed * lift,
                }
            )
    return pd.DataFrame(rows)
