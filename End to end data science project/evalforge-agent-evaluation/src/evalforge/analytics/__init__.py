"""Metrics, statistics and evaluator-alignment analysis.

Reusable Python modules, deliberately. ``notebooks/evaluator_alignment_analysis.ipynb``
demonstrates this code; it does not contain a second copy of it.
"""

from __future__ import annotations

from evalforge.analytics.alignment import (
    build_alignment_report,
    cohens_kappa,
    confusion_matrix,
    disagreement_examples,
    krippendorff_alpha,
    weighted_kappa,
)
from evalforge.analytics.metrics import (
    LENGTH_BUCKETS,
    RunMetrics,
    bucket_length,
    compute_metrics,
    dimension_breakdown,
    failure_examples,
    judge_agreement_with_deterministic,
)
from evalforge.analytics.statistics import (
    Interval,
    bootstrap_mean,
    cliffs_delta,
    cohens_h,
    interpret_effect,
    interpret_kappa,
    spearman,
    wilson_interval,
)

__all__ = [
    "LENGTH_BUCKETS",
    "Interval",
    "RunMetrics",
    "bootstrap_mean",
    "bucket_length",
    "build_alignment_report",
    "cliffs_delta",
    "cohens_h",
    "cohens_kappa",
    "compute_metrics",
    "confusion_matrix",
    "dimension_breakdown",
    "disagreement_examples",
    "failure_examples",
    "interpret_effect",
    "interpret_kappa",
    "judge_agreement_with_deterministic",
    "krippendorff_alpha",
    "spearman",
    "weighted_kappa",
    "wilson_interval",
]
