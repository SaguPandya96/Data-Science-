"""Measure train-only CPU throughput for the pinned BERT-Tiny candidate."""

from __future__ import annotations

import gzip
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

from authentitext.data.cleaning import sha256_file

MODEL_ID = "google/bert_uncased_L-2_H-128_A-2"
MODEL_REVISION = "30b0a37ccaaa32f332884b96992754e246e48c5f"
SEED = 1729
MAX_LENGTH = 128
BATCH_SIZE = 32
PROBE_ROWS = 2048
WARMUP_STEPS = 4
MEASURED_STEPS = 60
FULL_TRAIN_ROWS = 287843
PLANNED_EPOCHS = 3
HOSTED_RUNNER_LIMIT_SECONDS = 6 * 60 * 60
RESERVED_OVERHEAD_SECONDS = 15 * 60


class TransformerProbeError(RuntimeError):
    """Raised when a throughput probe cannot satisfy its sealed protocol."""


def _checkpoint_files(checkpoint_dir: Path) -> list[dict[str, Any]]:
    files = []
    for path in sorted(checkpoint_dir.rglob("*")):
        if not path.is_file() or ".cache" in path.relative_to(checkpoint_dir).parts:
            continue
        files.append(
            {
                "relative_path": path.relative_to(checkpoint_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not files:
        raise TransformerProbeError("Pinned checkpoint download produced no files")
    return files


def _load_probe_records(train_path: Path, rows: int) -> tuple[list[str], list[int]]:
    texts: list[str] = []
    targets: list[int] = []
    with gzip.open(train_path, "rt", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise TransformerProbeError(
                    f"Training line {line_number} is invalid JSON"
                ) from error
            if not isinstance(record, dict) or record.get("partition") != "train":
                raise TransformerProbeError(f"Training line {line_number} has an invalid role")
            if record.get("target") not in (0, 1) or not isinstance(record.get("text"), str):
                raise TransformerProbeError(f"Training line {line_number} has invalid model fields")
            texts.append(record["text"])
            targets.append(record["target"])
            if len(texts) == rows:
                break
    if len(texts) != rows:
        raise TransformerProbeError(f"Training input contains fewer than {rows} probe rows")
    return texts, targets


def estimate_training_seconds(rows_per_second: float) -> dict[str, Any]:
    """Estimate the prespecified full run from measured optimizer throughput."""
    if rows_per_second <= 0:
        raise TransformerProbeError("Measured throughput must be positive")
    epoch_seconds = FULL_TRAIN_ROWS / rows_per_second
    training_seconds = epoch_seconds * PLANNED_EPOCHS
    total_seconds = training_seconds + RESERVED_OVERHEAD_SECONDS
    return {
        "full_train_rows": FULL_TRAIN_ROWS,
        "planned_epochs": PLANNED_EPOCHS,
        "estimated_epoch_seconds": epoch_seconds,
        "estimated_training_seconds": training_seconds,
        "reserved_overhead_seconds": RESERVED_OVERHEAD_SECONDS,
        "estimated_total_seconds": total_seconds,
        "hosted_runner_limit_seconds": HOSTED_RUNNER_LIMIT_SECONDS,
        "full_run_viable": total_seconds <= HOSTED_RUNNER_LIMIT_SECONDS,
    }


def run_probe(train_path: Path, checkpoint_dir: Path) -> dict[str, Any]:
    """Download the immutable checkpoint and measure CPU optimizer steps."""
    try:
        import resource

        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as error:
        raise TransformerProbeError(f"Transformer probe dependency is missing: {error}") from error

    if train_path.stat().st_size != 152640483 or sha256_file(train_path) != (
        "9bb04cac540ac2aad1249adbd7cf1023a6da538eff5519a7bb11024ffb4c6918"
    ):
        raise TransformerProbeError("Training input does not match the audited sanitized partition")

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=checkpoint_dir,
    )
    checkpoint_files = _checkpoint_files(checkpoint_dir)

    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    torch.set_num_interop_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_dir,
        local_files_only=True,
        num_labels=2,
    )
    model.train()

    texts, targets = _load_probe_records(train_path, PROBE_ROWS)
    tokenization_started = time.perf_counter()
    encoded = tokenizer(
        texts,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    tokenization_seconds = time.perf_counter() - tokenization_started
    labels = torch.tensor(targets, dtype=torch.long)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    total_steps = WARMUP_STEPS + MEASURED_STEPS
    if total_steps * BATCH_SIZE > PROBE_ROWS:
        raise TransformerProbeError("Probe configuration requests more rows than loaded")

    measured_seconds = 0.0
    for step in range(total_steps):
        start = step * BATCH_SIZE
        end = start + BATCH_SIZE
        batch = {name: values[start:end] for name, values in encoded.items()}
        batch_labels = labels[start:end]
        optimizer.zero_grad(set_to_none=True)
        step_started = time.perf_counter()
        logits = model(**batch).logits
        loss = torch.nn.functional.cross_entropy(logits, batch_labels)
        loss.backward()
        optimizer.step()
        if step >= WARMUP_STEPS:
            measured_seconds += time.perf_counter() - step_started

    measured_rows = MEASURED_STEPS * BATCH_SIZE
    rows_per_second = measured_rows / measured_seconds
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_bytes = int(peak_rss if platform.system() == "Darwin" else peak_rss * 1024)

    return {
        "schema_version": 1,
        "status": "pass",
        "candidate": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "checkpoint_files": checkpoint_files,
        },
        "protocol": {
            "seed": SEED,
            "max_length": MAX_LENGTH,
            "batch_size": BATCH_SIZE,
            "probe_rows": PROBE_ROWS,
            "warmup_steps": WARMUP_STEPS,
            "measured_steps": MEASURED_STEPS,
            "optimizer": "AdamW",
            "learning_rate": 2e-5,
            "device": "cpu",
            "test_data_read": False,
            "performance_metrics_reported": False,
        },
        "input": {
            "partition": "train",
            "rows": FULL_TRAIN_ROWS,
            "bytes": train_path.stat().st_size,
            "sha256": sha256_file(train_path),
        },
        "measurement": {
            "measured_rows": measured_rows,
            "measured_seconds": measured_seconds,
            "rows_per_second": rows_per_second,
            "tokenization_seconds": tokenization_seconds,
            "peak_rss_bytes": peak_rss_bytes,
            "logical_cpus": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
        },
        "estimate": estimate_training_seconds(rows_per_second),
        "validation": {
            "transformer_trained": False,
            "model_artifact_saved": False,
            "source_text_in_report": False,
        },
    }
