from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_experiment() -> pd.DataFrame:
    """Hillstrom-shaped data where only recent customers respond to the email.

    The true uplift is known, so tests can check that methods recover it.
    """
    rng = np.random.default_rng(7)
    n = 12_000
    recency = rng.integers(1, 13, n)
    history = rng.gamma(2.0, 120.0, n).round(2)
    arms = rng.choice(["No E-Mail", "Mens E-Mail", "Womens E-Mail"], n)
    treated = arms != "No E-Mail"
    base = 0.08 + 0.0002 * history.clip(max=500)
    true_uplift = np.where(recency <= 4, 0.15, 0.0)
    visit = (rng.random(n) < base + treated * true_uplift).astype(int)
    conversion = visit * (rng.random(n) < 0.1).astype(int)
    spend = np.where(conversion == 1, (50 + 0.1 * history).round(2), 0.0)
    return pd.DataFrame(
        {
            "recency": recency,
            "history_segment": "1) $0 - $100",
            "history": history,
            "mens": rng.integers(0, 2, n),
            "womens": rng.integers(0, 2, n),
            "zip_code": rng.choice(["Urban", "Surburban", "Rural"], n),
            "newbie": rng.integers(0, 2, n),
            "channel": rng.choice(["Web", "Phone", "Multichannel"], n),
            "segment": arms,
            "visit": visit,
            "conversion": conversion,
            "spend": spend,
            "true_uplift": true_uplift,
        }
    )
