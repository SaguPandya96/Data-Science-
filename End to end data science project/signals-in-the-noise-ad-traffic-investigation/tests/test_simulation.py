from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.simulate_abuse import EVENT_COLUMNS, add_experimental_scenarios


class SimulationTests(unittest.TestCase):
    def setUp(self) -> None:
        rows = 10_000
        rng = np.random.default_rng(7)
        self.observed = pd.DataFrame(
            {
                "timestamp": np.arange(rows, dtype="int64") * 2,
                "uid": np.arange(rows, dtype="int64"),
                "campaign": rng.integers(1, 80, rows),
                "conversion": rng.binomial(1, 0.03, rows),
                "click": rng.binomial(1, 0.25, rows),
                "cost": rng.uniform(0.001, 0.1, rows),
                "time_since_last_click": rng.integers(-1, 5_000, rows),
                "source_id": rng.integers(1, 200, rows),
                "event_origin": "observed",
                "is_simulated_abuse": 0,
                "scenario": "observed",
                "episode_id": "observed",
            }
        )[EVENT_COLUMNS]

    def test_scenarios_are_separate_and_reproducible(self) -> None:
        original = self.observed.copy(deep=True)
        combined_a, manifest_a = add_experimental_scenarios(self.observed, seed=42)
        combined_b, manifest_b = add_experimental_scenarios(self.observed, seed=42)

        pd.testing.assert_frame_equal(self.observed, original)
        pd.testing.assert_frame_equal(manifest_a, manifest_b)
        self.assertGreater(len(combined_a), len(self.observed))
        self.assertGreater(int(combined_a["is_simulated_abuse"].sum()), 0)
        self.assertIn("clean_popularity_spike", set(manifest_a["scenario"]))
        clean = manifest_a.loc[manifest_a["scenario"] == "clean_popularity_spike"]
        self.assertTrue((clean["evaluation_label"] == 0).all())
        self.assertEqual(len(combined_a), len(combined_b))


if __name__ == "__main__":
    unittest.main()
