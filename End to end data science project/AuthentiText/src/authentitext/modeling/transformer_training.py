"""Train and evaluate the prespecified BERT-Tiny candidate on sealed roles."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from authentitext.data.cleaning import open_deterministic_gzip, sha256_file
from authentitext.data.transformer_train import sha256_gzip_content
from authentitext.modeling.transformer_probe import (
    BATCH_SIZE,
    FULL_TRAIN_ROWS,
    MAX_LENGTH,
    MODEL_ID,
    MODEL_REVISION,
    PLANNED_EPOCHS,
    SEED,
)

EVALUATION_BATCH_SIZE = 64
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
GRADIENT_CLIP_NORM = 1.0
EXPECTED_TRAIN_CONTENT_SHA256 = "40af4c0a731ea8e26649e41d610dc74cc53ed40fececc90fb3ff5ed666d1da17"
EXPECTED_VALIDATION_ROWS = 50509
EXPECTED_VALIDATION_CONTENT_SHA256 = (
    "53fefcd25bd61f7949cf02f543c68ad613e00dfb55eb10e1d5fd23d9fc33dd2c"
)


class TransformerTrainingError(RuntimeError):
    """Raised when full training or validation violates the frozen protocol."""


@dataclass(frozen=True)
class ValidationRecord:
    record_id: str
    source: str
    target: int
    text: str
    whitespace_tokens: int


def _load_partition(path: Path, partition: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise TransformerTrainingError(
                    f"{partition} line {line_number} is invalid JSON"
                ) from error
            if not isinstance(record, dict) or record.get("partition") != partition:
                raise TransformerTrainingError(
                    f"{partition} line {line_number} has invalid role metadata"
                )
            if (
                not isinstance(record.get("record_id"), str)
                or not isinstance(record.get("source"), str)
                or record.get("target") not in (0, 1)
                or not isinstance(record.get("text"), str)
            ):
                raise TransformerTrainingError(
                    f"{partition} line {line_number} has invalid model fields"
                )
            records.append(record)
    if not records or {record["target"] for record in records} != {0, 1}:
        raise TransformerTrainingError(f"{partition} must contain both targets")
    return records


def _tokenize_texts(tokenizer: Any, texts: list[str], torch: Any) -> dict[str, Any]:
    chunks: dict[str, list[Any]] = {}
    for start in range(0, len(texts), 1024):
        encoded = tokenizer(
            texts[start : start + 1024],
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        for name, values in encoded.items():
            chunks.setdefault(name, []).append(values)
    return {name: torch.cat(values, dim=0) for name, values in chunks.items()}


def _artifact_files(root: Path) -> tuple[list[dict[str, Any]], str, int]:
    files = []
    aggregate = hashlib.sha256()
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".cache" in path.relative_to(root).parts:
            continue
        relative_path = path.relative_to(root).as_posix()
        size = path.stat().st_size
        digest = sha256_file(path)
        files.append({"relative_path": relative_path, "bytes": size, "sha256": digest})
        aggregate.update(f"{relative_path}\0{size}\0{digest}\n".encode())
        total_bytes += size
    if not files:
        raise TransformerTrainingError("Trained model directory is empty")
    return files, aggregate.hexdigest(), total_bytes


def _write_predictions(
    path: Path,
    records: list[ValidationRecord],
    scores: Any,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        with open_deterministic_gzip(temporary) as handle:
            for record, score in zip(records, scores, strict=True):
                handle.write(
                    json.dumps(
                        {
                            "record_id": record.record_id,
                            "score": round(float(score), 12),
                            "source": record.source,
                            "target": record.target,
                            "whitespace_tokens": record.whitespace_tokens,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "relative_path": path.name,
        "rows": len(records),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def train_and_evaluate(
    *,
    train_path: Path,
    validation_path: Path,
    checkpoint_dir: Path,
    model_output_dir: Path,
    prediction_path: Path,
) -> dict[str, Any]:
    """Run the frozen three-epoch training protocol and validation evaluation."""
    try:
        import numpy as np
        import torch
        from huggingface_hub import snapshot_download
        from torch.utils.data import DataLoader, TensorDataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            get_linear_schedule_with_warmup,
        )

        from authentitext.modeling.evaluation import evaluate_scores
    except ImportError as error:
        raise TransformerTrainingError(f"Training dependency is missing: {error}") from error

    if sha256_gzip_content(train_path) != EXPECTED_TRAIN_CONTENT_SHA256:
        raise TransformerTrainingError("Training content identity does not match")
    if sha256_gzip_content(validation_path) != EXPECTED_VALIDATION_CONTENT_SHA256:
        raise TransformerTrainingError("Validation content identity does not match")

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    torch.set_num_interop_threads(1)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION, local_dir=checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_dir,
        local_files_only=True,
        num_labels=2,
    )

    train_records = _load_partition(train_path, "train")
    if len(train_records) != FULL_TRAIN_ROWS:
        raise TransformerTrainingError("Training row count does not match the frozen protocol")
    train_texts = [record["text"] for record in train_records]
    train_targets = torch.tensor([record["target"] for record in train_records], dtype=torch.long)
    preprocessing_started = time.perf_counter()
    train_encoded = _tokenize_texts(tokenizer, train_texts, torch)
    preprocessing_seconds = time.perf_counter() - preprocessing_started
    del train_texts, train_records

    input_names = sorted(train_encoded)
    train_dataset = TensorDataset(
        *(train_encoded[name] for name in input_names),
        train_targets,
    )
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    total_steps = len(train_loader) * PLANNED_EPOCHS
    warmup_steps = math.floor(total_steps * WARMUP_RATIO)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    model.train()
    epoch_reports = []
    training_started = time.perf_counter()
    for epoch in range(PLANNED_EPOCHS):
        epoch_started = time.perf_counter()
        loss_sum = 0.0
        rows_seen = 0
        for batch in train_loader:
            values = batch[:-1]
            labels = batch[-1]
            inputs = dict(zip(input_names, values, strict=True))
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**inputs, labels=labels)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            scheduler.step()
            loss_sum += float(outputs.loss.detach()) * len(labels)
            rows_seen += len(labels)
        epoch_seconds = time.perf_counter() - epoch_started
        epoch_reports.append(
            {
                "epoch": epoch + 1,
                "rows": rows_seen,
                "mean_training_loss": loss_sum / rows_seen,
                "seconds": epoch_seconds,
                "rows_per_second": rows_seen / epoch_seconds,
            }
        )
    training_seconds = time.perf_counter() - training_started

    model_output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_output_dir, safe_serialization=True)
    tokenizer.save_pretrained(model_output_dir)
    artifact_files, artifact_sha256, artifact_bytes = _artifact_files(model_output_dir)

    validation_data = _load_partition(validation_path, "validation")
    if len(validation_data) != EXPECTED_VALIDATION_ROWS:
        raise TransformerTrainingError("Validation row count does not match")
    validation_records = [
        ValidationRecord(
            record_id=record["record_id"],
            source=record["source"],
            target=record["target"],
            text=record["text"],
            whitespace_tokens=len(record["text"].split()),
        )
        for record in validation_data
    ]
    validation_texts = [record.text for record in validation_records]
    validation_targets = np.asarray([record.target for record in validation_records], dtype=np.int8)
    validation_sources = [record.source for record in validation_records]
    validation_lengths = np.asarray(
        [record.whitespace_tokens for record in validation_records], dtype=np.int32
    )
    validation_encoded = _tokenize_texts(tokenizer, validation_texts, torch)
    validation_input_names = sorted(validation_encoded)
    validation_dataset = TensorDataset(
        *(validation_encoded[name] for name in validation_input_names)
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=EVALUATION_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model.eval()
    score_chunks = []
    scoring_started = time.perf_counter()
    with torch.inference_mode():
        for batch in validation_loader:
            inputs = dict(zip(validation_input_names, batch, strict=True))
            probabilities = torch.softmax(model(**inputs).logits, dim=1)[:, 1]
            score_chunks.append(probabilities.cpu())
    scoring_seconds = time.perf_counter() - scoring_started
    scores = torch.cat(score_chunks).numpy().astype(np.float64)
    prediction_identity = _write_predictions(prediction_path, validation_records, scores)
    metrics = evaluate_scores(
        validation_targets,
        scores,
        validation_sources,
        validation_lengths,
    )

    reloaded = AutoModelForSequenceClassification.from_pretrained(
        model_output_dir,
        local_files_only=True,
    )
    reloaded.eval()
    first_inputs = {name: values[:8] for name, values in validation_encoded.items()}
    with torch.inference_mode():
        original_logits = model(**first_inputs).logits
        reloaded_logits = reloaded(**first_inputs).logits
    reload_matches = bool(torch.equal(original_logits, reloaded_logits))
    if not reload_matches:
        raise TransformerTrainingError("Reloaded model logits do not match")

    return {
        "schema_version": 1,
        "status": "pass",
        "dataset_id": "yaful/MAGE",
        "revision": "342663f0a2b775455c023f5d36a1341ff0ec5402",
        "candidate": {"model_id": MODEL_ID, "revision": MODEL_REVISION},
        "configuration": {
            "seed": SEED,
            "max_length": MAX_LENGTH,
            "epochs": PLANNED_EPOCHS,
            "batch_size": BATCH_SIZE,
            "evaluation_batch_size": EVALUATION_BATCH_SIZE,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "warmup_steps": warmup_steps,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "scheduler": "linear",
            "device": "cpu",
            "test_data_read": False,
        },
        "inputs": {
            "train": {
                "rows": FULL_TRAIN_ROWS,
                "content_sha256": EXPECTED_TRAIN_CONTENT_SHA256,
            },
            "validation": {
                "rows": EXPECTED_VALIDATION_ROWS,
                "content_sha256": EXPECTED_VALIDATION_CONTENT_SHA256,
            },
        },
        "training": {
            "preprocessing_seconds": preprocessing_seconds,
            "training_seconds": training_seconds,
            "total_optimizer_steps": total_steps,
            "epochs": epoch_reports,
        },
        "model_artifact": {
            "relative_path": model_output_dir.name,
            "bytes": artifact_bytes,
            "sha256": artifact_sha256,
            "files": artifact_files,
        },
        "predictions": prediction_identity,
        "scoring": {
            "seconds": scoring_seconds,
            "records_per_second": len(validation_records) / scoring_seconds,
            "batch_size": EVALUATION_BATCH_SIZE,
        },
        "metrics": metrics,
        "validation": {
            "status": "pass",
            "full_training_partition_used": True,
            "validation_partition_used_after_training": True,
            "test_data_read": False,
            "model_reload_logits_match": reload_matches,
            "source_text_in_report": False,
        },
    }
