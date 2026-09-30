"""Sentence embeddings for memory search.

Any object with ``embed(texts) -> array of unit vectors`` works as an embedder. Two ship:

- ``WordLlamaEmbedder``: WordLlama ``l2_supercat`` (256 dimensions, MIT licence). The
  weights and tokenizer are inside the ``wordllama`` package, so it loads offline in well
  under a second and needs no GPU. Install with ``pip install -e ".[embed]"``.
- ``HashingEmbedder``: character trigrams hashed into a fixed vector. No semantics at all;
  it exists so tests can exercise the embedding code path without the model.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class Embedder(Protocol):
    name: str

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


class _CachedEmbedder:
    """Embeds each distinct text once; memories and questions repeat a lot."""

    name = "cached"

    def __init__(self) -> None:
        self._cache: dict[str, np.ndarray] = {}

    def _embed_new(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            vectors = _normalize(np.asarray(self._embed_new(missing), dtype=np.float32))
            self._cache.update(zip(missing, vectors, strict=True))
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack([self._cache[t] for t in texts])


class WordLlamaEmbedder(_CachedEmbedder):
    name = "wordllama-l2_supercat-256"

    def __init__(self) -> None:
        super().__init__()
        self._model = _load_wordllama()

    def _embed_new(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._model.embed(texts))


@lru_cache(maxsize=1)
def _load_wordllama() -> Any:
    import wordllama
    from wordllama import WordLlama

    # The package keeps its bundled files in tokenizers/ and weights/ beside the code,
    # which is the layout the loader expects of a cache directory. Pointing it there
    # loads everything locally and never touches the network.
    package_dir = Path(wordllama.__file__).parent
    return WordLlama.load(
        config="l2_supercat", dim=256, cache_dir=package_dir, disable_download=True
    )


class HashingEmbedder(_CachedEmbedder):
    """Deterministic character-trigram vectors, for tests only."""

    name = "hashing"

    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        self.dim = dim

    def _embed_new(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            padded = f"  {text.lower()}  "
            for i in range(len(padded) - 2):
                digest = hashlib.blake2b(padded[i : i + 3].encode(), digest_size=4).digest()
                out[row, int.from_bytes(digest, "little") % self.dim] += 1.0
        return out


def default_embedder() -> Embedder | None:
    """WordLlama when it is installed, otherwise None (Keel then runs on BM25 alone)."""
    try:
        return WordLlamaEmbedder()
    except ImportError:
        return None
