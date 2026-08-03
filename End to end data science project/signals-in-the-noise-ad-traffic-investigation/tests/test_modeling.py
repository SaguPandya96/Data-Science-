from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.modeling import score_campaign_windows


class ModelingTests(unittest.TestCase):
    def test_time_split_scores_and_blind_queue(self) -> None:
        rng = np.random.default_rng(9)
        rows = 240
        positive_indexes = {20, 45, 75, 100, 130, 150, 175, 205, 225}
        records = []
        for index in range(rows):
            positive = int(index in positive_indexes)
            impressions = int(rng.integers(10, 70) + positive * 300)
            clicks = int(rng.integers(1, 12) + positive * 180)
            records.append(
                {
                    "window_start": index * 1800,
                    "campaign": index % 35,
                    "impressions": impressions,
                    "clicks": clicks,
                    "conversions": int(rng.integers(0, 3) if not positive else 0),
                    "total_cost_units": float(impressions * 0.02),
                    "avg_cost_units": 0.02,
                    "avg_seconds_since_click": 100.0,
                    "active_click_minutes": int(min(30, clicks)),
                    "unique_users": int(impressions * (0.9 if not positive else 0.1)),
                    "unique_sources": 12 if not positive else 1,
                    "repeat_event_share": 0.05 if not positive else 0.88,
                    "top_source_share": 0.15 if not positive else 0.96,
                    "ctr": clicks / impressions,
                    "conversion_per_click": 0.08 if not positive else 0.0,
                    "active_click_minute_share": min(1.0, clicks / 30),
                    "clicks_per_unique_user": clicks / max(1, int(impressions * (0.9 if not positive else 0.1))),
                    "observed_events": impressions,
                    "simulated_events": 0,
                    "abuse_label": positive,
                    "evaluation_scenario": "click_burst" if positive else "observed",
                }
            )
        frame = pd.DataFrame(records)
        config = {
            "train_fraction": 0.70,
            "throttle_threshold": 0.75,
            "escalate_threshold": 0.90,
        }
        project_root = Path(__file__).resolve().parents[1]
        folder = project_root / "tests" / "runtime"
        folder.mkdir(parents=True, exist_ok=True)
        result = score_campaign_windows(frame, config, folder)
        queue_columns = pd.read_csv(folder / "review_queue.csv").columns
        self.assertGreaterEqual(result["metrics"]["hybrid_test"]["average_precision"], 0.0)
        self.assertLessEqual(result["metrics"]["hybrid_test"]["average_precision"], 1.0)
        self.assertNotIn("abuse_label", queue_columns)
        self.assertNotIn("evaluation_scenario", queue_columns)


if __name__ == "__main__":
    unittest.main()
