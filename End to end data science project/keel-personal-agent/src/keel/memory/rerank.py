"""Second-stage reranking of memory candidates with a cross-encoder.

Keel's hybrid score compares a question and a memory through separate vectors. A
cross-encoder reads the two together, which is slower but can catch an implied link
("Whose local sports team should I cheer for?" and "Moved to Lisbon"). Running it on every
memory would be too slow, so Keel's own score picks a short list first and the
cross-encoder only reorders that list.

Any object with ``score(query, docs) -> array of relevance scores`` works as a reranker.
Two ship:

- ``OnnxCrossEncoder``: a BERT-style cross-encoder (e.g. ms-marco MiniLM) exported to ONNX
  and run on CPU with ONNX Runtime. Install with ``pip install -e ".[transformer]"``.
- ``OverlapReranker``: shared character trigrams. No semantics; for tests only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

import numpy as np

from keel.memory.retrieval import KeelMemory, _top_k
from keel.memory.store import Memory


class Reranker(Protocol):
    name: str

    def score(self, query: str, docs: Sequence[str]) -> np.ndarray: ...


class _CachedReranker:
    """Scores each distinct (question, memory) pair once."""

    name = "cached"

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], float] = {}

    def _score_new(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        raise NotImplementedError

    def score(self, query: str, docs: Sequence[str]) -> np.ndarray:
        missing = [(query, d) for d in dict.fromkeys(docs) if (query, d) not in self._cache]
        if missing:
            values = np.asarray(self._score_new(missing), dtype=np.float32)
            self._cache.update(zip(missing, (float(v) for v in values), strict=True))
        return np.array([self._cache[(query, d)] for d in docs], dtype=np.float32)


class OnnxCrossEncoder(_CachedReranker):
    """A cross-encoder on CPU. Returns the model's relevance logit for each pair.

    Runs single-threaded so the same pair always gets the same score, bit for bit; the
    benchmark's committed numbers depend on that.
    """

    def __init__(self, model_dir: Path, *, name: str, max_length: int = 256) -> None:
        super().__init__()
        import onnxruntime
        from tokenizers import Tokenizer

        self.name = name
        self._tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length)
        self._tokenizer.enable_padding()
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self._session = onnxruntime.InferenceSession(
            str(model_dir / "model.onnx"), options, providers=["CPUExecutionProvider"]
        )
        self._inputs = {i.name for i in self._session.get_inputs()}

    def _score_new(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        out = []
        for start in range(0, len(pairs), 32):
            batch = self._tokenizer.encode_batch(pairs[start : start + 32])
            feed = {
                "input_ids": np.array([e.ids for e in batch], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in batch], dtype=np.int64),
            }
            if "token_type_ids" in self._inputs:
                feed["token_type_ids"] = np.array([e.type_ids for e in batch], dtype=np.int64)
            logits = self._session.run(None, feed)[0]
            out.append(logits.reshape(len(batch), -1)[:, 0])
        return np.concatenate(out)


class OverlapReranker(_CachedReranker):
    """Fraction of the question's character trigrams found in the memory. Tests only."""

    name = "overlap"

    @staticmethod
    def _grams(text: str) -> set[str]:
        padded = f"  {text.lower()}  "
        return {padded[i : i + 3] for i in range(len(padded) - 2)}

    def _score_new(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        scores = []
        for query, doc in pairs:
            q = self._grams(query)
            scores.append(len(q & self._grams(doc)) / len(q) if q else 0.0)
        return np.array(scores)


@dataclass
class RerankedMemory:
    """Keel's retriever with a cross-encoder second stage.

    The always-on constraints are chosen exactly as in ``KeelMemory``. The ``candidates``
    best other memories by Keel's hybrid score are then rescored:

    - with a ``rerank_weight``, the new score is the hybrid score plus that weight times
      the cross-encoder's logit;
    - with ``rerank_weight=None``, the cross-encoder's logit alone decides the order.

    Memories outside the short list are never shown.
    """

    base: KeelMemory
    reranker: Reranker
    candidates: int = 20
    rerank_weight: float | None = None
    name: str = "keel+rerank"

    def retrieve(
        self, memories: Sequence[Memory], query: str, now: datetime, k: int
    ) -> list[Memory]:
        chosen, rest, scores = self.base.score_pool(memories, query, now, k)
        if not rest:
            return chosen
        order = sorted(range(len(rest)), key=lambda i: (scores[i], rest[i].id), reverse=True)
        short = order[: self.candidates]
        pool = [rest[i] for i in short]
        logits = self.reranker.score(query, [self.base.document(m) for m in pool])
        if self.rerank_weight is None:
            final = [float(x) for x in logits]
        else:
            final = [
                scores[i] + self.rerank_weight * float(x)
                for i, x in zip(short, logits, strict=True)
            ]
        return chosen + _top_k(pool, final, k - len(chosen), dedupe=self.base.use_dedupe)
