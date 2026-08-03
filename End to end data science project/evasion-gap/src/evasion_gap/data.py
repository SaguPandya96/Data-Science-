"""Loading and caching of the evaluation corpus."""

import hashlib
import json
import logging
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[2] / "data"

# Used when the dataset cannot be reached, so that a run still completes.
# Results produced from this sample are only a check that the code runs.
FALLBACK_TOXIC = [
    "you are such an idiot and everyone knows it",
    "shut up you moron nobody asked",
    "what a pathetic loser, go away",
    "this is garbage and so are you",
    "stupid people like you ruin everything",
    "get lost you worthless fool",
    "i hate you and everything you stand for",
    "you disgust me, seriously",
]

FALLBACK_BENIGN = [
    "thanks for putting this together, really helpful",
    "i disagree but i see where you are coming from",
    "the weather has been great this week",
    "could you share the source for that claim?",
    "nice work on the redesign",
    "i think there is a typo in the second paragraph",
    "looking forward to the next update",
    "that restaurant was better than i expected",
]


def _cache_path(params: dict) -> Path:
    key = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]
    return CACHE_DIR / f"corpus_{key}.json"


def load_corpus(
    name: str = "google/civil_comments",
    n_toxic: int = 300,
    n_benign: int = 300,
    toxic_min: float = 0.8,
    benign_max: float = 0.1,
    max_chars: int = 1000,
    use_cache: bool = True,
) -> Tuple[List[str], List[str]]:
    """Return a toxic and a benign sample of comments.

    The dataset is streamed rather than downloaded in full, since only a few
    hundred rows are needed. Comments above the toxicity cutoff are rare, so the
    stream reads a large number of rows and the result is cached to data/.
    """
    params = {
        "name": name,
        "n_toxic": n_toxic,
        "n_benign": n_benign,
        "toxic_min": toxic_min,
        "benign_max": benign_max,
        "max_chars": max_chars,
    }
    cache = _cache_path(params)

    if use_cache and cache.exists():
        payload = json.loads(cache.read_text(encoding="utf-8"))
        logger.info("Loaded corpus from cache %s", cache.name)
        return payload["toxic"], payload["benign"]

    try:
        from datasets import load_dataset

        stream = load_dataset(name, split="train", streaming=True)
        toxic: List[str] = []
        benign: List[str] = []

        for row in stream:
            text = (row.get("text") or "").strip()
            if not text or len(text) > max_chars:
                continue
            score = row.get("toxicity", 0.0)
            if score >= toxic_min and len(toxic) < n_toxic:
                toxic.append(text)
            elif score <= benign_max and len(benign) < n_benign:
                benign.append(text)
            if len(toxic) >= n_toxic and len(benign) >= n_benign:
                break

        if toxic and benign:
            logger.info("Loaded %d toxic / %d benign from %s", len(toxic), len(benign), name)
            if use_cache:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                cache.write_text(
                    json.dumps({"params": params, "toxic": toxic, "benign": benign}),
                    encoding="utf-8",
                )
            return toxic, benign

        logger.warning("Stream returned too few rows, using the built-in sample.")
    except Exception as exc:  # noqa: BLE001 - a load failure should not stop the run
        logger.warning("Dataset load failed (%s), using the built-in sample.", exc)

    return list(FALLBACK_TOXIC), list(FALLBACK_BENIGN)
