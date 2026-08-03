from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from src.database import build_database, build_feature_table


class DatabaseTests(unittest.TestCase):
    def test_sql_feature_rollup(self) -> None:
        events = pd.DataFrame(
            [
                [10, 1, 11, 0, 1, 0.1, 20, 5, "observed", 0, "observed", "observed"],
                [20, 1, 11, 0, 1, 0.2, 10, 5, "observed", 0, "observed", "observed"],
                [30, 2, 11, 1, 0, 0.1, -1, 6, "simulation", 1, "click_burst", "E01"],
                [40, 3, 11, 0, 0, 0.1, -1, 6, "observed", 0, "observed", "observed"],
                [50, 4, 11, 0, 0, 0.1, -1, 6, "observed", 0, "observed", "observed"],
            ],
            columns=[
                "timestamp", "uid", "campaign", "conversion", "click", "cost",
                "time_since_last_click", "source_id", "event_origin",
                "is_simulated_abuse", "scenario", "episode_id",
            ],
        )
        project_root = Path(__file__).resolve().parents[1]
        folder = project_root / "tests" / "runtime"
        folder.mkdir(parents=True, exist_ok=True)
        database = folder / "test.db"
        output = folder / "features.csv"
        build_database(events, database)
        features = build_feature_table(
            database, project_root / "sql" / "01_campaign_window_features.sql", output
        )
        self.assertEqual(len(features), 1)
        row = features.iloc[0]
        self.assertEqual(int(row["impressions"]), 5)
        self.assertEqual(int(row["clicks"]), 2)
        self.assertEqual(int(row["unique_users"]), 4)
        self.assertEqual(int(row["abuse_label"]), 1)


if __name__ == "__main__":
    unittest.main()
