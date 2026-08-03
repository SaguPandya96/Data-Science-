"""Experiment orchestration.

A run loads the corpus, scores the clean text to fix the thresholds, and then
sweeps every combination of split, attack and defense at each threshold.

Both the toxic and the benign split are swept. Content that evades detection and
content that is wrongly flagged are separate failures with different costs, and
the second one is only visible on the benign split.
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from .attacks import ATTACKS
from .defense import normalize_text
from .metrics import bootstrap_ci, rate_above, threshold_at_fpr, threshold_at_recall
from .model import Scorer

logger = logging.getLogger(__name__)

DEFENSES: Dict[str, Callable[[str], str]] = {
    "none": lambda text: text,
    "normalized": normalize_text,
}


@dataclass
class OperatingPoint:
    """A decision threshold and its measured behaviour on clean text."""

    name: str
    pin: str
    value: float
    threshold: float
    clean_recall: float
    benign_fpr: float

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "pinned_on": self.pin,
            "target": self.value,
            "threshold": self.threshold,
            "clean_recall": self.clean_recall,
            "benign_fpr": self.benign_fpr,
        }


@dataclass
class ExperimentResult:
    operating_points: List[OperatingPoint]
    sweep: pd.DataFrame
    clean_scores: Dict[str, np.ndarray] = field(default_factory=dict)
    config: dict = field(default_factory=dict)


def build_operating_point(
    spec: dict, toxic_scores: np.ndarray, benign_scores: np.ndarray
) -> OperatingPoint:
    """Turn a config entry into a threshold and measure it on clean text."""
    pin, value = spec["pin"], spec["value"]

    if pin == "recall":
        threshold = threshold_at_recall(toxic_scores, value)
    elif pin == "fpr":
        threshold = threshold_at_fpr(benign_scores, value)
    else:
        raise ValueError(f"unknown pin {pin!r}, expected 'recall' or 'fpr'")

    op = OperatingPoint(
        name=spec["name"],
        pin=pin,
        value=value,
        threshold=threshold,
        clean_recall=rate_above(toxic_scores, threshold),
        benign_fpr=rate_above(benign_scores, threshold),
    )
    logger.info(
        "%-12s threshold=%.4f clean_recall=%.3f fpr=%.3f",
        op.name, op.threshold, op.clean_recall, op.benign_fpr,
    )
    return op


def run_sweep(
    scorer: Scorer,
    splits: Dict[str, List[str]],
    operating_points: List[OperatingPoint],
    attacks: Dict[str, Callable[[str], str]] = None,
    defenses: Dict[str, Callable[[str], str]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Score every split, attack and defense combination at every threshold.

    Returns one row per combination. The `rate` column is recall on the toxic
    split and the false positive rate on the benign split. Thresholds are held
    fixed across all conditions, since a threshold chosen on clean data is what a
    deployed system would be using.
    """
    attacks = attacks or ATTACKS
    defenses = defenses or DEFENSES
    rows = []

    for split_name, texts in splits.items():
        for attack_name, attack in attacks.items():
            attacked = [attack(text) for text in texts]
            for defense_name, defense in defenses.items():
                scores = scorer([defense(text) for text in attacked])
                for op in operating_points:
                    lo, hi = bootstrap_ci(scores, op.threshold, seed=seed)
                    rows.append(
                        {
                            "split": split_name,
                            "attack": attack_name,
                            "defense": defense_name,
                            "operating_point": op.name,
                            "rate": rate_above(scores, op.threshold),
                            "ci_low": lo,
                            "ci_high": hi,
                            "mean_score": float(scores.mean()),
                        }
                    )
            logger.info("swept %s / %s", split_name, attack_name)

    df = pd.DataFrame(rows)

    keys = ["split", "defense", "operating_point"]
    baseline = df[df["attack"] == "clean"].set_index(keys)["rate"].rename("clean_rate")
    df = df.join(baseline, on=keys)
    df["delta"] = df["rate"] - df["clean_rate"]
    return df.sort_values(["split", "operating_point", "defense", "rate"]).reset_index(drop=True)


def run_experiment(config: dict) -> ExperimentResult:
    """Run the full experiment described by a config dict."""
    from .data import load_corpus

    eval_cfg = config["eval"]

    toxic, benign = load_corpus(**config["dataset"])
    scorer = Scorer(
        model_id=config["model_id"],
        batch_size=eval_cfg["batch_size"],
        max_length=eval_cfg["max_length"],
    )

    toxic_scores = scorer(toxic)
    benign_scores = scorer(benign)
    operating_points = [
        build_operating_point(spec, toxic_scores, benign_scores)
        for spec in eval_cfg["operating_points"]
    ]

    sweep = run_sweep(
        scorer,
        {"toxic": toxic, "benign": benign},
        operating_points,
        seed=config.get("seed", 42),
    )

    return ExperimentResult(
        operating_points=operating_points,
        sweep=sweep,
        clean_scores={"toxic": toxic_scores, "benign": benign_scores},
        config=config,
    )
