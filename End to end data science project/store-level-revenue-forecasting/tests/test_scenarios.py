from __future__ import annotations

import numpy as np
import pandas as pd

from store_revenue_forecasting.scenarios import run_scenarios


class ScenarioModel:
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return (
            features["Sales_Lag_7_RollingAvg"].to_numpy()
            + features["Promo"].to_numpy() * 100
        )


def test_scenarios_are_compared_with_baseline_without_mutation() -> None:
    features = pd.DataFrame(
        {
            "Promo": [1, 0],
            "Promo2Active": [0, 0],
            "PromoActive": [1, 0],
            "Sales_Lag_7_RollingAvg": [1000.0, 800.0],
            "Sales_Lag_30_RollingAvg": [900.0, 850.0],
        }
    )
    original = features.copy(deep=True)
    metadata = pd.DataFrame(
        {"Date": pd.to_datetime(["2024-01-01", "2024-01-02"]), "Store": ["1", "1"]}
    )
    scenarios = [
        {"name": "Baseline", "kind": "baseline"},
        {"name": "No Promotions", "kind": "set_promotions", "value": 0},
        {"name": "Demand Drop", "kind": "scale_recent_demand", "factor": 0.8},
    ]

    summary, daily = run_scenarios(ScenarioModel(), features, metadata, scenarios)

    assert summary.loc[summary["Scenario"].eq("Baseline"), "% Change"].iloc[0] == 0
    assert summary.loc[summary["Scenario"].eq("No Promotions"), "% Change"].iloc[0] < 0
    assert summary.loc[summary["Scenario"].eq("Demand Drop"), "% Change"].iloc[0] < 0
    assert len(daily) == len(features) * len(scenarios)
    pd.testing.assert_frame_equal(features, original)
