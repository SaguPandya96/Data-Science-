from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
from pathlib import Path

import pytest

from keel.evaluation.benchmark import load_config
from keel.evaluation.round4 import run_round4, to_markdown
from keel.memory.embeddings import (
    DEFAULT_ENCODER,
    HashingEmbedder,
    default_transformer,
    download_model,
)
from keel.memory.retrieval import TRANSFORMER_WEIGHT, default_retriever

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "eval.toml"


def _archive(tmp_path: Path) -> tuple[str, str]:
    """A tiny model archive on disk, served through a file:// URL."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, data in (
            ("m/model.onnx", b"onnx"),
            ("m/tokenizer.json", b"{}"),
            ("m/._model.onnx", b"apple metadata"),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    path = tmp_path / "model.tar.gz"
    path.write_bytes(buffer.getvalue())
    return path.as_uri(), hashlib.sha256(buffer.getvalue()).hexdigest()


def test_download_verifies_unpacks_and_caches(tmp_path, monkeypatch):
    monkeypatch.delenv("KEEL_OFFLINE")
    url, digest = _archive(tmp_path)
    folder = download_model(url, digest, tmp_path / "cache")
    assert (folder / "model.onnx").read_bytes() == b"onnx"
    # Second call is served from disk even with downloads forbidden.
    monkeypatch.setenv("KEEL_OFFLINE", "1")
    assert download_model(url, digest, tmp_path / "cache") == folder


def test_checksum_mismatch_is_rejected_and_nothing_is_kept(tmp_path, monkeypatch):
    monkeypatch.delenv("KEEL_OFFLINE")
    url, _ = _archive(tmp_path)
    with pytest.raises(ValueError, match="checksum mismatch"):
        download_model(url, "0" * 64, tmp_path / "cache")
    leftovers = list((tmp_path / "cache").rglob("*"))
    assert all(p.is_dir() for p in leftovers)


def test_offline_refuses_to_download(tmp_path):
    url, digest = _archive(tmp_path)
    with pytest.raises(FileNotFoundError, match="KEEL_OFFLINE"):
        download_model(url, digest, tmp_path / "cache")


def test_offline_agent_falls_back_without_error():
    assert default_transformer() is None
    retriever = default_retriever()
    assert retriever.embedding_weight != TRANSFORMER_WEIGHT or retriever.embedder is None


def test_shipped_defaults_match_the_round4_result():
    results = json.loads((ROOT / "reports" / "metrics" / "round4.json").read_text())
    minilm = load_config(CONFIG)["round4"]["encoders"]["minilm"]
    assert results["pass_rule"]["passed"] is True
    assert results["chosen_encoder"] == DEFAULT_ENCODER["name"] == minilm["name"]
    assert results["chosen_weight"] == TRANSFORMER_WEIGHT
    assert DEFAULT_ENCODER["sha256"] == minilm["sha256"]
    assert DEFAULT_ENCODER["url"] == minilm["url"]


def test_round4_runner_with_stand_in_encoders(tmp_path):
    small = tmp_path / "eval.toml"
    small.write_text(CONFIG.read_text().replace("personas = 200", "personas = 3"))
    ticks = iter(range(100))
    results, timings = run_round4(
        small,
        encoders={"minilm": HashingEmbedder(64), "bge": HashingEmbedder(96)},
        wordllama=HashingEmbedder(),
        clock=lambda: float(next(ticks)),
    )
    assert set(results["splits"]) == {"holdout2", "holdout2_direct", "holdout2_indirect", "holdout"}
    assert "keel+transformer" in results["splits"]["holdout2"]
    assert set(timings) == {"minilm", "bge", "wordllama"}
    assert "Round 4: transformer sentence encoders" in to_markdown(results, timings)


@pytest.mark.skipif(not os.environ.get("KEEL_TEST_MODEL_DIR"), reason="needs a downloaded encoder")
def test_real_minilm_is_deterministic_and_semantic(monkeypatch):
    pytest.importorskip("onnxruntime")
    from keel.memory.embeddings import OnnxSentenceEmbedder

    monkeypatch.setenv("KEEL_MODEL_DIR", os.environ["KEEL_TEST_MODEL_DIR"])
    folder = download_model(DEFAULT_ENCODER["url"], DEFAULT_ENCODER["sha256"])
    texts = ["A man is playing a guitar.", "Someone strums a guitar.", "Stocks fell today."]
    first = OnnxSentenceEmbedder(folder, pooling="mean", name="minilm").embed(texts)
    second = OnnxSentenceEmbedder(folder, pooling="mean", name="minilm").embed(texts)
    assert first.shape == (3, 384)
    assert (first == second).all()
    assert first[0] @ first[1] > first[0] @ first[2] + 0.3
