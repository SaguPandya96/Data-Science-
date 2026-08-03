"""Corpus loading.

The toxic set drives recall; the benign set is what makes the threshold mean
something. Recall without a paired false-positive rate is not an operating point.
"""

import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Smoke-test fallback so a stalled download never blocks a run. Mild by design --
# numbers produced from this set are a pipeline check, not a result.
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


def load_corpus(
    name: str = "google/civil_comments",
    n_toxic: int = 300,
    n_benign: int = 300,
    toxic_min: float = 0.8,
    benign_max: float = 0.1,
    max_chars: int = 1000,
) -> Tuple[List[str], List[str]]:
    """Stream a labelled comment corpus and pull a balanced toxic/benign sample.

    Streamed rather than downloaded: the full corpus is several hundred MB and we
    need a few hundred rows. Falls back to a built-in sample on any load failure.
    """
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
            return toxic, benign
        logger.warning("Stream returned too few rows; falling back to built-in sample.")
    except Exception as exc:  # noqa: BLE001 - any load failure degrades to the fallback
        logger.warning("Dataset load failed (%s); falling back to built-in sample.", exc)

    return list(FALLBACK_TOXIC), list(FALLBACK_BENIGN)
