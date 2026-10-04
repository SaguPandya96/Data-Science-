"""Choosing which memories go in the prompt.

Every retriever takes the full memory list and returns at most ``k`` memories. The
benchmark compares them on identical inputs, so each one must be a pure function of its
arguments.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import TYPE_CHECKING, Protocol

import numpy as np

from keel.memory.embeddings import Embedder, default_embedder, default_transformer
from keel.memory.store import Memory
from keel.memory.text import expand, tokens

if TYPE_CHECKING:
    from keel.memory.rerank import RerankedMemory


class Retriever(Protocol):
    name: str

    def retrieve(
        self, memories: Sequence[Memory], query: str, now: datetime, k: int
    ) -> list[Memory]: ...


def bm25_scores(
    docs: Sequence[str], query_weights: dict[str, float], k1: float = 1.2, b: float = 0.75
) -> list[float]:
    """Okapi BM25 with weighted query terms."""
    tokenized = [_cached_tokens(d) for d in docs]
    n = len(tokenized)
    if n == 0:
        return []
    avg_len = sum(len(t) for t in tokenized) / n or 1.0
    doc_freq: Counter[str] = Counter()
    for toks in tokenized:
        doc_freq.update(set(toks))
    scores = []
    for toks in tokenized:
        counts = Counter(toks)
        length_norm = k1 * (1 - b + b * len(toks) / avg_len)
        score = 0.0
        for term, weight in query_weights.items():
            tf = counts.get(term, 0)
            if tf == 0:
                continue
            idf = math.log(1 + (n - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            score += weight * idf * tf * (k1 + 1) / (tf + length_norm)
        scores.append(score)
    return scores


@lru_cache(maxsize=65536)
def _cached_tokens(text: str) -> tuple[str, ...]:
    return tuple(tokens(text))


def _top_k(
    memories: Sequence[Memory], scores: Sequence[float], k: int, *, dedupe: bool = False
) -> list[Memory]:
    # Ties go to the newer memory, which is the safer guess for personal facts.
    order = sorted(range(len(memories)), key=lambda i: (scores[i], memories[i].id), reverse=True)
    chosen: list[Memory] = []
    seen: set[str] = set()
    for i in order:
        if len(chosen) == k:
            break
        signature = " ".join(_cached_tokens(memories[i].text))
        if dedupe and signature in seen:
            continue
        seen.add(signature)
        chosen.append(memories[i])
    return chosen


def embed_query_and_docs(embedder: Embedder, query: str, docs: Sequence[str]) -> np.ndarray:
    """Embed a question and the texts it is compared with, adding the encoder's prefixes.

    With no prefixes (most encoders) the texts are embedded exactly as given.
    """
    query_prefix = getattr(embedder, "query_prefix", "")
    doc_prefix = getattr(embedder, "doc_prefix", "")
    return embedder.embed([query_prefix + query, *(doc_prefix + d for d in docs)])


def indexed_text(memory: Memory) -> str:
    """What Keel searches: the memory's key words, then its text.

    The key names the topic even when the text doesn't: "Adopted a parrot named Kiwi" is
    stored under "pet", so a question about pets can find it.
    """
    if memory.key is None:
        return memory.text
    return f"{memory.key.replace('_', ' ')}. {memory.text}"


class NoMemory:
    name = "none"

    def retrieve(
        self, memories: Sequence[Memory], query: str, now: datetime, k: int
    ) -> list[Memory]:
        return []


class RecentMemory:
    """The k newest memories, whatever the question."""

    name = "recent"

    def retrieve(
        self, memories: Sequence[Memory], query: str, now: datetime, k: int
    ) -> list[Memory]:
        visible = [m for m in memories if not m.deleted]
        return sorted(visible, key=lambda m: m.id, reverse=True)[:k]


class LexicalMemory:
    """Plain BM25 over every memory ever written: standard retrieval-augmented memory."""

    name = "lexical"

    def retrieve(
        self, memories: Sequence[Memory], query: str, now: datetime, k: int
    ) -> list[Memory]:
        visible = [m for m in memories if not m.deleted]
        weights = dict.fromkeys(tokens(query), 1.0)
        scores = bm25_scores([m.text for m in visible], weights)
        matched = [(m, s) for m, s in zip(visible, scores, strict=True) if s > 0]
        return _top_k([m for m, _ in matched], [s for _, s in matched], k)


class FullMemory:
    """Everything. Perfect recall at an unbounded prompt size."""

    name = "full"

    def retrieve(
        self, memories: Sequence[Memory], query: str, now: datetime, k: int
    ) -> list[Memory]:
        return [m for m in memories if not m.deleted]


class EmbeddingMemory:
    """Cosine similarity over every memory ever written: standard dense retrieval memory."""

    def __init__(self, embedder: Embedder, name: str = "embedding") -> None:
        self.embedder = embedder
        self.name = name

    def retrieve(
        self, memories: Sequence[Memory], query: str, now: datetime, k: int
    ) -> list[Memory]:
        visible = [m for m in memories if not m.deleted]
        if not visible or k <= 0:
            return []
        vectors = embed_query_and_docs(self.embedder, query, [m.text for m in visible])
        similarity = vectors[1:] @ vectors[0]
        return _top_k(visible, [float(x) for x in similarity], k)


@dataclass
class KeelMemory:
    """Keel's retriever.

    1. Only active memories: superseded and forgotten ones are never shown.
    2. Up to ``core_max`` constraints (allergies, hard limits) always go in, because
       forgetting one can hurt the user even when the question doesn't mention it. They
       never take more than half of ``k``.
    3. The remaining slots go to a hybrid score: BM25 over the key and text, with stemming
       and synonym expansion, plus smaller terms for recency and importance.
    4. Repeats of a memory already chosen are skipped, so five copies of the same small talk
       can't crowd out the answer.

    With an ``embedder``, the hybrid score also adds ``embedding_weight`` times the cosine
    similarity between the question and the memory's key and text, which lets a question
    find a memory it shares no words with.
    """

    core_max: int = 2
    lexical_weight: float = 1.0
    recency_weight: float = 0.25
    importance_weight: float = 0.15
    half_life_days: float = 30.0
    use_supersession: bool = True
    use_expansion: bool = True
    use_core: bool = True
    use_keys: bool = True
    use_dedupe: bool = True
    embedder: Embedder | None = None
    embedding_weight: float = 0.0
    name: str = "keel"

    def retrieve(
        self, memories: Sequence[Memory], query: str, now: datetime, k: int
    ) -> list[Memory]:
        chosen, rest, scores = self.score_pool(memories, query, now, k)
        return chosen + _top_k(rest, scores, k - len(chosen), dedupe=self.use_dedupe)

    def score_pool(
        self, memories: Sequence[Memory], query: str, now: datetime, k: int
    ) -> tuple[list[Memory], list[Memory], list[float]]:
        """Steps 1 to 3: the constraints that always go in, then every other candidate with
        its hybrid score."""
        if self.use_supersession:
            pool = [m for m in memories if m.active]
        else:
            pool = [m for m in memories if not m.deleted]
        if k <= 0 or not pool:
            return [], [], []

        chosen: list[Memory] = []
        if self.use_core and self.core_max > 0:
            core = [m for m in pool if m.kind == "constraint"]
            core.sort(key=lambda m: (m.importance, m.id), reverse=True)
            # Never more than half the budget, so a question can still be answered at small k.
            chosen = core[: min(self.core_max, k // 2)]

        rest = [m for m in pool if m not in chosen]
        if not rest:
            return chosen, [], []
        weights = expand(query) if self.use_expansion else dict.fromkeys(tokens(query), 1.0)
        docs = [self.document(m) for m in rest]
        lexical = bm25_scores(docs, weights)
        top = max(lexical) or 1.0
        semantic = [0.0] * len(rest)
        if self.embedder is not None and self.embedding_weight:
            vectors = embed_query_and_docs(self.embedder, query, docs)
            semantic = [float(x) for x in vectors[1:] @ vectors[0]]
        scores = []
        for memory, lex, sim in zip(rest, lexical, semantic, strict=True):
            age_days = max((now - memory.created_at).total_seconds() / 86400, 0.0)
            recency = 0.5 ** (age_days / self.half_life_days)
            importance = (memory.importance - 1) / 4
            scores.append(
                self.lexical_weight * lex / top
                + self.embedding_weight * sim
                + self.recency_weight * recency
                + self.importance_weight * importance
            )
        return chosen, rest, scores

    def document(self, memory: Memory) -> str:
        """The text Keel searches for this memory."""
        return indexed_text(memory) if self.use_keys else memory.text


# Chosen on the dev wordings in round 2 of the benchmark (reports/metrics/round2.json).
EMBEDDING_WEIGHT = 0.5
# Chosen with the encoder in round 12 on already-seen question sets, then confirmed on a
# fresh held-out set (reports/metrics/round12.json), and chosen again for its int8 copy in
# round 16 (reports/metrics/round16.json). MiniLM had shipped at 2.0, then 6.0, mpnet at
# 12.0 and e5-base at 48.0.
TRANSFORMER_WEIGHT = 64.0


def default_retriever(core_max: int = 2) -> KeelMemory | RerankedMemory:
    """The agent's retriever, best available first.

    1. Keel with the e5-large-v2 transformer encoder (round 12's winner), with its top
       candidates reranked by the ms-marco-MiniLM-L-6-v2 cross-encoder (round 7's winner, in
       the int8 version round 8 adopted), if ONNX Runtime is installed and both models are on
       disk or can be downloaded.
    2. The same without reranking, if only the encoder is available.
    3. Keel with WordLlama embeddings (round 2's winner), if installed.
    4. Keel on BM25 alone.
    """
    from keel.memory.rerank import (
        RERANK_CANDIDATES,
        RERANK_WEIGHT,
        RerankedMemory,
        default_reranker,
    )

    transformer = default_transformer()
    if transformer is not None:
        keel = KeelMemory(
            core_max=core_max, embedder=transformer, embedding_weight=TRANSFORMER_WEIGHT
        )
        reranker = default_reranker()
        if reranker is None:
            return keel
        return RerankedMemory(
            base=keel,
            reranker=reranker,
            candidates=RERANK_CANDIDATES,
            rerank_weight=RERANK_WEIGHT,
        )
    embedder = default_embedder()
    return KeelMemory(
        core_max=core_max,
        embedder=embedder,
        embedding_weight=EMBEDDING_WEIGHT if embedder is not None else 0.0,
    )


def render_context(memories: Sequence[Memory]) -> str:
    """The memory block placed in the prompt."""
    if not memories:
        return "(no relevant memories)"
    return "\n".join(f"- #{m.id} {m.render()}" for m in memories)
