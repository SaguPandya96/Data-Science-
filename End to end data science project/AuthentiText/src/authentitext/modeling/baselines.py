"""Train and verify deterministic CPU baselines."""

from __future__ import annotations

import gzip
import hashlib
import json
import platform
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASELINE_VERSION = 1
REPORT_SCHEMA_VERSION = 1
RANDOM_SEED = 1729
DEFAULT_TFIDF_CONFIG = {
    "analyzer": "word",
    "ngram_range": (1, 2),
    "lowercase": True,
    "min_df": 5,
    "max_df": 0.995,
    "max_features": 100_000,
    "sublinear_tf": True,
    "norm": "l2",
    "dtype": np.float32,
}
DEFAULT_LOGISTIC_CONFIG = {
    "C": 1.0,
    "class_weight": "balanced",
    "max_iter": 50,
    "random_state": RANDOM_SEED,
    "solver": "saga",
    "tol": 1e-3,
}


class BaselineError(RuntimeError):
    """Raised when baseline artifacts cannot be trained or verified."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_training_records(
    path: Path, expected_partition: str = "train"
) -> tuple[list[str], np.ndarray]:
    """Load only model inputs and targets from one sanitized partition."""
    texts: list[str] = []
    targets: list[int] = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise BaselineError(f"{path}:{line_number} is not valid JSON") from error
            if not isinstance(record, dict) or record.get("partition") != expected_partition:
                raise BaselineError(f"{path}:{line_number} has invalid partition metadata")
            text = record.get("text")
            target = record.get("target")
            if not isinstance(text, str) or target not in (0, 1):
                raise BaselineError(f"{path}:{line_number} has invalid model fields")
            texts.append(text)
            targets.append(target)
    if not texts or set(targets) != {0, 1}:
        raise BaselineError("Training data must be non-empty and contain both targets")
    return texts, np.asarray(targets, dtype=np.int8)


def length_features(texts: list[str]) -> np.ndarray:
    """Create the two diagnostic log-length features."""
    return np.asarray(
        [(np.log1p(len(text)), np.log1p(len(text.split()))) for text in texts],
        dtype=np.float64,
    )


def _artifact_entry(path: Path, model_type: str) -> dict[str, Any]:
    return {
        "model_type": model_type,
        "relative_path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _write_joblib(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        joblib.dump(payload, temporary, compress=3)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _serializable_config(config: dict[str, Any]) -> dict[str, Any]:
    rendered = {}
    for key, value in config.items():
        if value is np.float32:
            rendered[key] = "float32"
        elif isinstance(value, tuple):
            rendered[key] = list(value)
        else:
            rendered[key] = value
    return rendered


def train_baselines(
    *,
    train_path: Path,
    artifact_root: Path,
    dataset_id: str,
    revision: str,
    input_sha256: str,
    tfidf_config: dict[str, Any] | None = None,
    logistic_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train majority, length-only, and word TF-IDF logistic baselines."""
    tfidf_settings = {**DEFAULT_TFIDF_CONFIG, **(tfidf_config or {})}
    logistic_settings = {**DEFAULT_LOGISTIC_CONFIG, **(logistic_config or {})}
    load_started = time.perf_counter()
    texts, targets = load_training_records(train_path)
    load_seconds = time.perf_counter() - load_started
    target_counts = Counter(str(int(target)) for target in targets)
    majority_class = int(target_counts["1"] >= target_counts["0"])
    positive_prevalence = float(targets.mean())

    artifacts: list[dict[str, Any]] = []
    majority_path = artifact_root / "majority.joblib"
    _write_joblib(
        {
            "artifact_version": BASELINE_VERSION,
            "model_type": "majority",
            "majority_class": majority_class,
            "positive_probability": positive_prevalence,
        },
        majority_path,
    )
    majority_entry = _artifact_entry(majority_path, "majority")
    majority_entry["configuration"] = {
        "majority_class": majority_class,
        "positive_probability": round(positive_prevalence, 9),
    }
    artifacts.append(majority_entry)

    length_started = time.perf_counter()
    length_model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_SEED,
                    solver="lbfgs",
                    tol=1e-6,
                ),
            ),
        ]
    )
    length_model.fit(length_features(texts), targets)
    length_seconds = time.perf_counter() - length_started
    length_path = artifact_root / "length_logistic.joblib"
    _write_joblib(
        {
            "artifact_version": BASELINE_VERSION,
            "model_type": "length_logistic",
            "feature_order": ["log1p_characters", "log1p_whitespace_tokens"],
            "model": length_model,
        },
        length_path,
    )
    length_entry = _artifact_entry(length_path, "length_logistic")
    length_entry["configuration"] = {
        "features": ["log1p_characters", "log1p_whitespace_tokens"],
        "classifier": {
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 1000,
            "solver": "lbfgs",
            "tol": 1e-6,
        },
        "n_iter": int(length_model.named_steps["classifier"].n_iter_[0]),
    }
    length_entry["fit_seconds"] = round(length_seconds, 3)
    artifacts.append(length_entry)

    vectorizer_started = time.perf_counter()
    vectorizer = TfidfVectorizer(**tfidf_settings)
    matrix = vectorizer.fit_transform(texts)
    vectorizer_seconds = time.perf_counter() - vectorizer_started
    classifier = LogisticRegression(**logistic_settings)
    classifier_started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        classifier.fit(matrix, targets)
    classifier_seconds = time.perf_counter() - classifier_started
    convergence_warnings = [
        str(item.message) for item in caught if item.category is ConvergenceWarning
    ]
    tfidf_path = artifact_root / "word_tfidf_logistic.joblib"
    _write_joblib(
        {
            "artifact_version": BASELINE_VERSION,
            "model_type": "word_tfidf_logistic",
            "vectorizer": vectorizer,
            "classifier": classifier,
        },
        tfidf_path,
    )
    tfidf_entry = _artifact_entry(tfidf_path, "word_tfidf_logistic")
    tfidf_entry["configuration"] = {
        "vectorizer": _serializable_config(tfidf_settings),
        "classifier": _serializable_config(logistic_settings),
    }
    tfidf_entry["training_matrix"] = {
        "rows": matrix.shape[0],
        "features": matrix.shape[1],
        "nonzero": int(matrix.nnz),
        "density": round(matrix.nnz / (matrix.shape[0] * matrix.shape[1]), 9),
    }
    tfidf_entry["vocabulary_size"] = len(vectorizer.vocabulary_)
    tfidf_entry["n_iter"] = int(classifier.n_iter_[0])
    tfidf_entry["convergence_warnings"] = convergence_warnings
    tfidf_entry["vectorizer_fit_transform_seconds"] = round(vectorizer_seconds, 3)
    tfidf_entry["classifier_fit_seconds"] = round(classifier_seconds, 3)
    artifacts.append(tfidf_entry)

    del matrix
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "baseline_version": BASELINE_VERSION,
        "dataset_id": dataset_id,
        "revision": revision,
        "input": {
            "partition": "train",
            "relative_path": train_path.name,
            "bytes": train_path.stat().st_size,
            "sha256": input_sha256,
            "rows": len(texts),
            "target_counts": {target: target_counts[target] for target in ("0", "1")},
        },
        "configuration": {
            "random_seed": RANDOM_SEED,
            "metadata_features_used": [],
            "test_data_used": False,
            "validation_data_used": False,
            "machine_positive_target": 1,
        },
        "environment": {
            "python": platform.python_version(),
            "joblib": joblib.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "timing": {"training_data_load_seconds": round(load_seconds, 3)},
        "artifacts": artifacts,
        "validation": {
            "status": "pass" if not convergence_warnings else "warning",
            "artifacts_reload_and_score": False,
            "tfidf_converged_without_warning": not convergence_warnings,
        },
    }


def positive_scores(payload: dict[str, Any], texts: list[str]) -> np.ndarray:
    model_type = payload.get("model_type")
    if model_type == "majority":
        return np.full(len(texts), payload["positive_probability"], dtype=np.float64)
    if model_type == "length_logistic":
        return payload["model"].predict_proba(length_features(texts))[:, 1]
    if model_type == "word_tfidf_logistic":
        matrix = payload["vectorizer"].transform(texts)
        return payload["classifier"].predict_proba(matrix)[:, 1]
    raise BaselineError(f"Unsupported baseline model type: {model_type!r}")


def verify_baselines(report: dict[str, Any], artifact_root: Path) -> None:
    """Verify artifact identities, reloadability, types, and finite scores."""
    smoke_texts = ["A short verification sample.", "A second, longer verification sample."]
    observed_types = set()
    for artifact in report["artifacts"]:
        path = artifact_root / artifact["relative_path"]
        if not path.is_file():
            raise BaselineError(f"Missing baseline artifact: {path}")
        if path.stat().st_size != artifact["bytes"]:
            raise BaselineError(f"Baseline artifact size mismatch: {path}")
        if sha256_file(path) != artifact["sha256"]:
            raise BaselineError(f"Baseline artifact SHA-256 mismatch: {path}")
        payload = joblib.load(path)
        if payload.get("model_type") != artifact["model_type"]:
            raise BaselineError(f"Baseline artifact type mismatch: {path}")
        scores = positive_scores(payload, smoke_texts)
        if scores.shape != (2,) or not np.isfinite(scores).all():
            raise BaselineError(f"Baseline artifact returned invalid scores: {path}")
        if ((scores < 0) | (scores > 1)).any():
            raise BaselineError(f"Baseline artifact returned out-of-range scores: {path}")
        observed_types.add(payload["model_type"])
    expected_types = {"majority", "length_logistic", "word_tfidf_logistic"}
    if observed_types != expected_types:
        raise BaselineError(f"Baseline artifact types {observed_types!r} do not match expected")


def render_report(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
