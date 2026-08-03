from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.quality_modeling import score_observed_quality


class ObservedQualityModelingTests(unittest.TestCase):
    def test_expected_behavior_scores_are_bounded_and_queue_is_neutral(self) -> None:
        rng = np.random.default_rng(31)
        records: list[dict] = []
        campaign_history: dict[int, list[tuple[int, float, float, float]]] = {
            campaign: [] for campaign in range(8)
        }
        for index in range(320):
            campaign = index % 8
            history = campaign_history[campaign][-20:]
            unusual = index in {300, 308, 316}
            impressions = int(rng.integers(45, 95))
            click_probability = 0.08 + 0.01 * (campaign % 3)
            clicks = int(rng.binomial(impressions, click_probability))
            conversions = int(rng.binomial(impressions, 0.012))
            repeat_share = float(rng.uniform(0.02, 0.14))
            top_source_share = float(rng.uniform(0.12, 0.30))
            if unusual:
                impressions = 520
                clicks = 310
                conversions = 0
                repeat_share = 0.91
                top_source_share = 0.95
            ctr = clicks / impressions
            conversion_rate = conversions / impressions
            avg_cost = 0.02 + 0.002 * campaign
            total_cost = impressions * avg_cost
            records.append(
                {
                    "window_start": index * 1800,
                    "campaign": campaign,
                    "impressions": impressions,
                    "clicks": clicks,
                    "conversions": conversions,
                    "total_cost_units": total_cost,
                    "avg_cost_units": avg_cost,
                    "active_click_minutes": min(30, clicks),
                    "unique_users": max(1, int(impressions * (0.91 if not unusual else 0.12))),
                    "unique_sources": 10 if not unusual else 1,
                    "repeat_event_share": repeat_share,
                    "top_source_share": top_source_share,
                    "ctr": ctr,
                    "conversion_rate": conversion_rate,
                    "active_click_minute_share": min(1.0, clicks / 30),
                    "clicks_per_unique_user": clicks / max(1, int(impressions * 0.91)),
                    "hour_of_day": int((index * 1800 / 3600) % 24),
                    "history_windows": len(history),
                    "baseline_impressions": np.mean([item[0] for item in history]) if history else np.nan,
                    "baseline_ctr": np.mean([item[1] for item in history]) if history else np.nan,
                    "baseline_conversion_rate": np.mean([item[2] for item in history]) if history else np.nan,
                    "baseline_cost_units": np.mean([item[3] for item in history]) if history else np.nan,
                    "volume_change_ratio": (
                        impressions / np.mean([item[0] for item in history]) if history else np.nan
                    ),
                    "ctr_change_from_baseline": (
                        ctr - np.mean([item[1] for item in history]) if history else np.nan
                    ),
                    "conversion_change_from_baseline": (
                        conversion_rate - np.mean([item[2] for item in history])
                        if history
                        else np.nan
                    ),
                }
            )
            campaign_history[campaign].append(
                (impressions, ctr, conversion_rate, total_cost)
            )

        project_root = Path(__file__).resolve().parents[1]
        output = project_root / "tests" / "runtime" / "observed_quality"
        config = {
            "train_fraction": 0.70,
            "observed_review_quantile": 0.98,
            "observed_review_limit": 10,
        }
        result = score_observed_quality(pd.DataFrame(records), config, output)

        scored = result["scored"]
        self.assertTrue(scored["expected_ctr"].between(0, 1).all())
        self.assertTrue(scored["expected_conversion_rate"].between(0, 1).all())
        self.assertTrue(scored["quality_risk_score"].between(0, 1).all())
        self.assertGreater(len(result["review_queue"]), 0)
        queue_columns = [column.lower() for column in pd.read_csv(
            output / "observed_review_queue.csv"
        ).columns]
        for prohibited in ("fraud", "abuse", "simulation", "label"):
            self.assertFalse(any(prohibited in column for column in queue_columns))


if __name__ == "__main__":
    unittest.main()
