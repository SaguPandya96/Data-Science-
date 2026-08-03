"""Experiment orchestration: corpus -> operating points -> attack sweep."""

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from .attacks import ATTACKS
from .metrics import bootstrap_ci, rate_above, threshold_at_fpr, threshold_at_recall
from .model import Scorer

logger = logging.getLogger(__name__)


@dataclass
class OperatingPoint:
    """A threshold plus the clean-data behaviour that justifies it."""

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
    config: dict = field(default_factory=dict)


def build_operating_point(
    spec: dict,
    toxic_scores: np.ndarray,
    benign_scores: np.ndarray,
) -> OperatingPoint:
    """Resolve a config spec into a concrete threshold and measure it on clean data."""
    pin, value = spec["pin"], spec["value"]

    if pin == "recall":
        threshold = threshold_at_recall(toxic_scores, value)
    elif pin == "fpr":
        threshold = threshold_at_fpr(benign_scores, value)
    else:
        raise ValueError(f"unknown pin {pin!r}; expected 'recall' or 'fpr'")

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
        op.name,
        op.threshold,
        op.clean_recall,
        op.benign_fpr,
    )
    return op


def run_attack_sweep(
    scorer: Scorer,
    toxic: List[str],
    operating_points: List[OperatingPoint],
    attacks: Dict[str, Callable[[str], str]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Score every attack variant once, then evaluate it at every operating point.

    Thresholds never move between attacks -- an operating point chosen on clean data
    is what actually ships.
    """
    attacks = attacks or ATTACKS
    rows = []

    for name, transform in attacks.items():
        scores = scorer([transform(text) for text in toxic])
        for op in operating_points:
            recall = rate_above(scores, op.threshold)
            lo, hi = bootstrap_ci(scores, op.threshold, seed=seed)
            rows.append(
                {
                    "operating_point": op.name,
                    "attack": name,
                    "recall": recall,
                    "ci_low": lo,
                    "ci_high": hi,
                    "mean_score": float(scores.mean()),
                }
            )
        logger.info("%-12s mean_score=%.3f", name, float(scores.mean()))

    df = pd.DataFrame(rows)
    clean = df[df["attack"] == "clean"].set_index("operating_point")["recall"]
    df["recall_drop"] = df["operating_point"].map(clean) - df["recall"]
    return df.sort_values(["operating_point", "recall"]).reset_index(drop=True)


def run_experiment(config: dict) -> ExperimentResult:
    """End-to-end run driven by a config dict."""
    from .data import load_corpus

    ds_cfg = config["dataset"]
    eval_cfg = config["eval"]

    toxic, benign = load_corpus(
        name=ds_cfg["name"],
        n_toxic=ds_cfg["n_toxic"],
        n_benign=ds_cfg["n_benign"],
        toxic_min=ds_cfg["toxic_min"],
        benign_max=ds_cfg["benign_max"],
    )

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

    sweep = run_attack_sweep(scorer, toxic, operating_points, seed=config.get("seed", 42))
    return ExperimentResult(operating_points=operating_points, sweep=sweep, config=config)
