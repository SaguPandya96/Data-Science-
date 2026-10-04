"""Sentence embeddings for memory search.

Any object with ``embed(texts) -> array of unit vectors`` works as an embedder. Three ship:

- ``WordLlamaEmbedder``: WordLlama ``l2_supercat`` (256 dimensions, MIT licence). The
  weights and tokenizer are inside the ``wordllama`` package, so it loads offline in well
  under a second and needs no GPU. Install with ``pip install -e ".[embed]"``.
- ``OnnxSentenceEmbedder``: a transformer sentence encoder (e.g. MiniLM, BGE) exported to
  ONNX and run on CPU with ONNX Runtime. ``download_model`` fetches and verifies the
  archive. Install with ``pip install -e ".[transformer]"``.
- ``HashingEmbedder``: character trigrams hashed into a fixed vector. No semantics at all;
  it exists so tests can exercise the embedding code path without a model.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import tempfile
import urllib.request
from collections.abc import Mapping, Sequence
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


def model_dir() -> Path:
    """Where downloaded encoders live: ``KEEL_MODEL_DIR``, else ~/.cache/keel/models."""
    return Path(os.environ.get("KEEL_MODEL_DIR", "~/.cache/keel/models")).expanduser()


def offline() -> bool:
    """``KEEL_OFFLINE=1`` forbids downloads; only models already on disk are used."""
    return os.environ.get("KEEL_OFFLINE", "").lower() in ("1", "true", "yes")


# The int8 copy of the encoder chosen in round 12, adopted in round 16 of the benchmark
# (reports/metrics/round16.json). Rounds 4 and 5 had shipped all-MiniLM-L6-v2, rounds 6 to
# 10 all-mpnet-base-v2, round 11 e5-base-v2 and rounds 12 to 15 full-precision e5-large-v2.
DEFAULT_ENCODER: dict[str, Any] = {
    "name": "e5-large-v2 (int8)",
    "pooling": "mean",
    "query_prefix": "query: ",
    "doc_prefix": "passage: ",
    "files": {
        "model.onnx": {
            "url": "https://huggingface.co/Xenova/e5-large-v2/resolve/"
            "840fd2207f68e253697ed85392a482ff7657ad11/onnx/model_quantized.onnx",
            "sha256": "80779f47fd485869cb5d12338e3956cab089075f136c39498f67d53bcc41b357",
        },
        "tokenizer.json": {
            "url": "https://huggingface.co/Xenova/e5-large-v2/resolve/"
            "840fd2207f68e253697ed85392a482ff7657ad11/tokenizer.json",
            "sha256": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
        },
    },
}


def download_model(url: str, sha256: str, cache_dir: Path | None = None) -> Path:
    """Fetch a model archive once, check its SHA-256, and unpack it.

    Returns the folder holding the ONNX file. A second call finds it on disk and makes no
    network request. A download whose hash doesn't match is deleted, not used.
    """
    root = (cache_dir or model_dir()) / sha256[:16]
    marker = root / ".verified"
    if marker.exists():
        return _model_folder(root)
    if offline():
        raise FileNotFoundError(f"{url} is not downloaded yet and KEEL_OFFLINE is set")
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=root, suffix=".tar.gz", delete=False) as tmp:
        archive = Path(tmp.name)
    try:
        with urllib.request.urlopen(url, timeout=120) as response, archive.open("wb") as out:
            shutil.copyfileobj(response, out)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != sha256:
            raise ValueError(f"checksum mismatch for {url}: got {digest}, expected {sha256}")
        with tarfile.open(archive) as tar:
            tar.extractall(root, filter="data")
    finally:
        archive.unlink(missing_ok=True)
    marker.write_text(url + "\n")
    return _model_folder(root)


def download_files(files: Mapping[str, Mapping[str, str]], cache_dir: Path | None = None) -> Path:
    """Fetch a model published as separate files rather than one archive.

    ``files`` maps each local file name to its ``url`` and ``sha256``. Every hash must be
    pinned; a file whose hash doesn't match is deleted, not used. Returns the folder.
    """
    if any(not spec.get("sha256") for spec in files.values()):
        raise ValueError("every file needs a pinned sha256")
    digest = hashlib.sha256("".join(sorted(s["sha256"] for s in files.values())).encode())
    root = (cache_dir or model_dir()) / digest.hexdigest()[:16]
    marker = root / ".verified"
    if marker.exists():
        return root
    if offline():
        raise FileNotFoundError(f"{root.name} is not downloaded yet and KEEL_OFFLINE is set")
    root.mkdir(parents=True, exist_ok=True)
    for name, spec in files.items():
        target = root / name
        if target.exists() and _sha256(target) == spec["sha256"]:
            continue
        with tempfile.NamedTemporaryFile(dir=root, delete=False) as tmp:
            partial = Path(tmp.name)
        try:
            with (
                urllib.request.urlopen(spec["url"], timeout=120) as response,
                partial.open("wb") as out,
            ):
                shutil.copyfileobj(response, out)
            got = _sha256(partial)
            if got != spec["sha256"]:
                raise ValueError(
                    f"checksum mismatch for {spec['url']}: got {got}, expected {spec['sha256']}"
                )
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)
    marker.write_text("".join(f"{name} {spec['url']}\n" for name, spec in files.items()))
    return root


def fetch_model(spec: Mapping[str, Any], cache_dir: Path | None = None) -> Path:
    """The folder of a pinned model: separate ``files``, or one archive at ``url``."""
    if "files" in spec:
        return download_files(spec["files"], cache_dir)
    return download_model(spec["url"], spec["sha256"], cache_dir)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_folder(root: Path) -> Path:
    models = [p for p in root.rglob("*.onnx") if not p.name.startswith("._")]
    if len(models) != 1:
        raise FileNotFoundError(f"expected one .onnx file under {root}, found {len(models)}")
    return models[0].parent


class OnnxSentenceEmbedder(_CachedEmbedder):
    """A transformer sentence encoder on CPU.

    Runs single-threaded so the same text always gives the same vector, bit for bit;
    the benchmark's committed numbers depend on that.
    """

    def __init__(
        self,
        model_dir: Path,
        *,
        pooling: str,
        name: str,
        max_length: int = 128,
        query_prefix: str = "",
        doc_prefix: str = "",
    ) -> None:
        super().__init__()
        import onnxruntime
        from tokenizers import Tokenizer

        # Some encoders were trained with a fixed prefix on questions and another on the
        # text being searched (e.g. "query: " and "passage: "). Retrievers add them.
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix

        if pooling not in ("mean", "cls"):
            raise ValueError("pooling must be 'mean' or 'cls'")
        self.name = name
        self.pooling = pooling
        self._tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length)
        self._tokenizer.enable_padding()
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        model = next(p for p in model_dir.glob("*.onnx") if not p.name.startswith("._"))
        self._session = onnxruntime.InferenceSession(
            str(model), options, providers=["CPUExecutionProvider"]
        )
        self._inputs = {i.name for i in self._session.get_inputs()}

    def _embed_new(self, texts: list[str]) -> np.ndarray:
        out = []
        for start in range(0, len(texts), 64):
            batch = self._tokenizer.encode_batch(texts[start : start + 64])
            ids = np.array([e.ids for e in batch], dtype=np.int64)
            mask = np.array([e.attention_mask for e in batch], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            hidden = self._session.run(None, feed)[0]
            if self.pooling == "cls":
                out.append(hidden[:, 0])
            else:
                weights = mask[..., None].astype(np.float32)
                out.append((hidden * weights).sum(axis=1) / weights.sum(axis=1))
        return np.concatenate(out)


def default_embedder() -> Embedder | None:
    """WordLlama when it is installed, otherwise None (Keel then runs on BM25 alone)."""
    try:
        return WordLlamaEmbedder()
    except ImportError:
        return None


def default_transformer() -> Embedder | None:
    """The default encoder when ONNX Runtime is installed and the model is on disk or
    downloadable.

    Returns None rather than raising, so the agent still starts offline or without the
    ``transformer`` extra; it then falls back to WordLlama.
    """
    try:
        import onnxruntime  # noqa: F401
        import tokenizers  # noqa: F401
    except ImportError:
        return None
    try:
        folder = download_files(DEFAULT_ENCODER["files"])
    except (OSError, ValueError):  # offline, blocked, or a corrupted download
        return None
    return OnnxSentenceEmbedder(
        folder,
        pooling=DEFAULT_ENCODER["pooling"],
        name=DEFAULT_ENCODER["name"],
        query_prefix=DEFAULT_ENCODER["query_prefix"],
        doc_prefix=DEFAULT_ENCODER["doc_prefix"],
    )
