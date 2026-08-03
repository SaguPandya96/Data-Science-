from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


def build_database(events: pd.DataFrame, database_path: Path) -> None:
    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    if database_path.exists():
        database_path.unlink()

    with sqlite3.connect(database_path) as connection:
        events.to_sql("events", connection, index=False, if_exists="replace", chunksize=50_000)
        connection.executescript(
            """
            CREATE INDEX idx_events_time_campaign ON events(timestamp, campaign);
            CREATE INDEX idx_events_campaign_uid ON events(campaign, uid);
            CREATE INDEX idx_events_source ON events(source_id);
            """
        )


def build_feature_table(database_path: Path, sql_path: Path, output_csv: Path) -> pd.DataFrame:
    query = Path(sql_path).read_text(encoding="utf-8")
    with sqlite3.connect(database_path) as connection:
        features = pd.read_sql_query(query, connection)
        features.to_sql("campaign_window_features", connection, index=False, if_exists="replace")
        connection.execute(
            "CREATE INDEX idx_features_time ON campaign_window_features(window_start)"
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_csv, index=False)
    return features


def save_scores(database_path: Path, scored: pd.DataFrame) -> None:
    with sqlite3.connect(database_path) as connection:
        scored.to_sql("risk_scores", connection, index=False, if_exists="replace")
        connection.execute("CREATE INDEX idx_risk_score ON risk_scores(risk_score DESC)")
