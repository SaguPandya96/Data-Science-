from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


EVENT_COLUMNS = [
    "timestamp",
    "uid",
    "campaign",
    "conversion",
    "click",
    "cost",
    "time_since_last_click",
    "source_id",
    "event_origin",
    "is_simulated_abuse",
    "scenario",
    "episode_id",
]


@dataclass(frozen=True)
class Episode:
    fraction: float
    scenario: str
    abusive: bool
    sequence: int


EPISODES = [
    Episode(0.18, "click_burst", True, 1),
    Episode(0.27, "impression_flood", True, 2),
    Episode(0.36, "click_burst", True, 3),
    Episode(0.45, "clean_popularity_spike", False, 4),
    Episode(0.53, "impression_flood", True, 5),
    Episode(0.62, "click_burst", True, 6),
    Episode(0.74, "click_burst", True, 7),
    Episode(0.79, "adaptive_low_and_slow", True, 8),
    Episode(0.84, "clean_popularity_spike", False, 9),
    Episode(0.88, "impression_flood", True, 10),
    Episode(0.92, "adaptive_low_and_slow", True, 11),
    Episode(0.96, "clean_popularity_spike", False, 12),
]


def load_observed(path: Path) -> pd.DataFrame:
    usecols = [
        "timestamp",
        "uid",
        "campaign",
        "conversion",
        "click",
        "cost",
        "time_since_last_click",
        "cat1",
        "cat2",
    ]
    frame = pd.read_csv(path, sep="\t", usecols=usecols)
    if frame.empty:
        raise ValueError("Observed dataset is empty.")
    if not frame["timestamp"].is_monotonic_increasing:
        raise ValueError("Observed timestamps must be sorted.")

    # The dataset does not expose publishers or placements. I use a stable,
    # anonymized combination of two contextual fields and call it source_id.
    # It is a grouping device, not a recovered real-world identity.
    frame["source_id"] = (
        (frame["cat1"].astype("int64") * 31 + frame["cat2"].astype("int64")) % 10_007
    ).astype("int64")
    frame = frame.drop(columns=["cat1", "cat2"])
    frame["event_origin"] = "observed"
    frame["is_simulated_abuse"] = 0
    frame["scenario"] = "observed"
    frame["episode_id"] = "observed"
    return frame[EVENT_COLUMNS]


def _draw_ids(rng: np.random.Generator, prefix: int, unique_count: int, size: int) -> np.ndarray:
    values = np.arange(prefix, prefix + max(1, unique_count), dtype="int64")
    return rng.choice(values, size=size, replace=True)


def _make_episode(
    episode: Episode,
    start: int,
    duration: int,
    campaign: int,
    source_seed: int,
    cost_median: float,
    observed_ctr: float,
    observed_conversion_rate: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if episode.scenario == "click_burst":
        n, unique_users, sources = 900, 18, 1
        timestamps = start + np.sort(rng.integers(0, max(60, duration // 3), size=n))
        clicks = rng.binomial(1, 0.94, size=n)
        conversions = rng.binomial(1, 0.004, size=n) * clicks
        gaps = np.where(clicks == 1, rng.integers(1, 25, size=n), -1)
    elif episode.scenario == "impression_flood":
        n, unique_users, sources = 1_400, 820, 2
        timestamps = start + np.sort(rng.integers(0, duration, size=n))
        clicks = rng.binomial(1, 0.008, size=n)
        conversions = np.zeros(n, dtype="int64")
        gaps = np.full(n, -1, dtype="int64")
    elif episode.scenario == "adaptive_low_and_slow":
        n, unique_users, sources = 760, 310, 7
        timestamps = start + np.sort(rng.integers(0, duration * 2, size=n))
        clicks = rng.binomial(1, 0.52, size=n)
        conversions = rng.binomial(1, 0.006, size=n) * clicks
        gaps = np.where(clicks == 1, rng.integers(20, 900, size=n), -1)
    elif episode.scenario == "clean_popularity_spike":
        n, unique_users, sources = 1_050, 940, 35
        timestamps = start + np.sort(rng.integers(0, duration, size=n))
        clicks = rng.binomial(1, observed_ctr, size=n)
        conversions = rng.binomial(1, observed_conversion_rate, size=n)
        gaps = np.where(clicks == 1, rng.integers(30, 7_200, size=n), -1)
    else:
        raise ValueError(f"Unknown scenario: {episode.scenario}")

    episode_id = f"E{episode.sequence:02d}_{episode.scenario}"
    return pd.DataFrame(
        {
            "timestamp": timestamps.astype("int64"),
            "uid": _draw_ids(rng, 900_000_000 + episode.sequence * 100_000, unique_users, n),
            "campaign": np.full(n, campaign, dtype="int64"),
            "conversion": conversions.astype("int64"),
            "click": clicks.astype("int64"),
            "cost": np.maximum(1e-5, rng.lognormal(np.log(max(cost_median, 1e-5)), 0.35, size=n)),
            "time_since_last_click": gaps.astype("int64"),
            "source_id": _draw_ids(rng, 800_000 + source_seed * 100, sources, n),
            "event_origin": "simulation",
            "is_simulated_abuse": int(episode.abusive),
            "scenario": episode.scenario,
            "episode_id": episode_id,
        }
    )[EVENT_COLUMNS]


def add_experimental_scenarios(
    observed: pd.DataFrame, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add documented scenarios while leaving observed rows unchanged."""

    rng = np.random.default_rng(seed)
    minimum = int(observed["timestamp"].min())
    maximum = int(observed["timestamp"].max())
    span = maximum - minimum
    if span < 7_200:
        raise ValueError("The sample needs at least two hours of event history.")

    top_campaigns = observed["campaign"].value_counts().head(40).index.to_numpy(dtype="int64")
    if len(top_campaigns) < len(EPISODES):
        raise ValueError("Not enough campaigns to create independent episodes.")

    cost_median = float(max(observed["cost"].median(), 1e-5))
    observed_ctr = float(observed["click"].mean())
    observed_conversion_rate = float(observed["conversion"].mean())
    duration = int(max(900, min(3_600, span * 0.035)))

    synthetic_frames: list[pd.DataFrame] = []
    manifest_rows: list[dict] = []
    for index, episode in enumerate(EPISODES):
        start = int(minimum + episode.fraction * span)
        campaign = int(top_campaigns[(index * 3 + 1) % len(top_campaigns)])
        episode_frame = _make_episode(
            episode=episode,
            start=start,
            duration=duration,
            campaign=campaign,
            source_seed=episode.sequence,
            cost_median=cost_median,
            observed_ctr=observed_ctr,
            observed_conversion_rate=observed_conversion_rate,
            rng=rng,
        )
        synthetic_frames.append(episode_frame)
        manifest_rows.append(
            {
                "episode_id": episode_frame["episode_id"].iloc[0],
                "scenario": episode.scenario,
                "evaluation_label": int(episode.abusive),
                "start_timestamp": int(episode_frame["timestamp"].min()),
                "end_timestamp": int(episode_frame["timestamp"].max()),
                "campaign": campaign,
                "rows_added": len(episode_frame),
                "reason_for_inclusion": (
                    "negative control: unusual volume with ordinary behavior"
                    if not episode.abusive
                    else "documented abuse stress test"
                ),
            }
        )

    combined = pd.concat([observed, *synthetic_frames], ignore_index=True)
    combined = combined.sort_values(["timestamp", "campaign", "uid"], kind="stable").reset_index(drop=True)
    manifest = pd.DataFrame(manifest_rows)
    return combined[EVENT_COLUMNS], manifest
