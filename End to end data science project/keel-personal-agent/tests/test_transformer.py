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
from keel.evaluation.round5 import run_round5
from keel.evaluation.round5 import to_markdown as round5_markdown
from keel.evaluation.round6 import run_round6
from keel.evaluation.round6 import to_markdown as round6_markdown
from keel.memory.embeddings import (
    DEFAULT_ENCODER,
    HashingEmbedder,
    default_transformer,
    download_model,
)
from keel.memory.retrieval import TRANSFORMER_WEIGHT, default_retriever, embed_query_and_docs

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


def test_shipped_defaults_match_the_benchmark_results():
    """The agent ships whatever the latest adopted round chose, and nothing else."""
    metrics = ROOT / "reports" / "metrics"
    config = load_config(CONFIG)
    round12 = json.loads((metrics / "round12.json").read_text())
    plan12 = config["round12"]
    if round12["pass_rule"]["passed"]:
        key, weight = round12["chosen"]["encoder"], round12["chosen"]["weight"]
    else:
        key, weight = plan12["current"]["encoder"], plan12["current"]["weight"]
    spec = plan12["encoders"][key]
    fields = ("name", "pooling", "query_prefix", "doc_prefix", "files")
    assert {k: spec[k] for k in fields} == DEFAULT_ENCODER
    assert weight == TRANSFORMER_WEIGHT
    # Round 12's baseline must be what round 11 adopted.
    round11 = json.loads((metrics / "round11.json").read_text())
    plan11 = config["round11"]
    assert round11["pass_rule"]["passed"]
    assert plan12["current"]["weight"] == round11["chosen"]["weight"]
    current = plan12["encoders"][plan12["current"]["encoder"]]
    assert current == plan11["encoders"][round11["chosen"]["encoder"]]
    # Round 11's baseline must be what round 6 adopted and rounds 7 to 10 kept.
    round6 = json.loads((metrics / "round6.json").read_text())
    plan = config["round6"]
    assert plan11["current"]["encoder"] == round6["chosen"]["encoder"]
    assert plan11["current"]["weight"] == round6["chosen"]["weight"]
    assert plan11["encoders"]["mpnet"]["files"] == plan["encoders"]["mpnet"]["files"]
    # Round 6's baseline must be what rounds 4 and 5 had shipped.
    round5 = json.loads((metrics / "round5.json").read_text())
    assert plan["current"]["weight"] == round5["chosen_weight"]
    assert plan["encoders"]["minilm"]["name"] == round5["encoder"]


def test_round5_runner_with_stand_in_encoders(tmp_path):
    small = tmp_path / "eval.toml"
    small.write_text(CONFIG.read_text().replace("personas = 200", "personas = 3"))
    results = run_round5(small, encoder=HashingEmbedder(64), wordllama=HashingEmbedder())
    grid = load_config(CONFIG)["round5"]["weight_grid"]
    assert set(results["tuning"]) == {str(w) for w in grid}
    assert results["chosen_weight"] in grid
    assert set(results["splits"]) == {"holdout3", "holdout3_direct", "holdout3_indirect"}
    assert results["pass_rule"]["outcome"]
    if results["chosen_weight"] == results["current_weight"]:
        assert results["pass_rule"]["passed"] is None
    else:
        assert "keel+transformer(tuned)" in results["splits"]["holdout3"]
    assert "Round 5: a wider weight grid" in round5_markdown(results)


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
def test_real_default_encoder_is_deterministic_and_semantic(monkeypatch):
    pytest.importorskip("onnxruntime")

    monkeypatch.setenv("KEEL_MODEL_DIR", os.environ["KEEL_TEST_MODEL_DIR"])
    first, second = default_transformer(), default_transformer()
    assert first is not None and second is not None
    query = "How many people live in Berlin?"
    docs = ["Berlin has a population of 3.5 million people.", "The cat sat on the mat."]
    a = embed_query_and_docs(first, query, docs)
    b = embed_query_and_docs(second, query, docs)
    assert a.shape[0] == 3
    assert (a == b).all()
    assert a[0] @ a[1] > a[0] @ a[2] + 0.1


def test_round6_runner_with_stand_in_encoders(tmp_path):
    small = tmp_path / "eval.toml"
    small.write_text(CONFIG.read_text().replace("personas = 200", "personas = 3"))
    ticks = iter(range(100))
    encoders = {
        "minilm": HashingEmbedder(64),
        "bge_base": HashingEmbedder(96),
        "mpnet": HashingEmbedder(128),
    }
    results, timings = run_round6(small, encoders=encoders, clock=lambda: float(next(ticks)))
    plan = load_config(CONFIG)["round6"]
    assert set(results["tuning"]) == set(plan["encoders"])
    assert results["chosen"]["encoder"] in plan["encoders"]
    assert results["chosen"]["weight"] in plan["weight_grid"]
    assert set(results["splits"]) == {"holdout4", "holdout4_direct", "holdout4_indirect"}
    assert set(timings) == set(plan["encoders"])
    unchanged = results["chosen"] == {"encoder": "minilm", "weight": 6.0}
    assert (results["pass_rule"]["passed"] is None) == unchanged
    assert "Round 6: larger transformer encoders" in round6_markdown(results, timings)
