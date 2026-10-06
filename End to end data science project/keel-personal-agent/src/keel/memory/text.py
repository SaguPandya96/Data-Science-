"""Tokenizing, stemming and query expansion for memory search.

Deliberately small. The synonym groups cover everyday personal topics
in general terms; they were written before the benchmark's test templates were first run
(docs/ANALYSIS_PLAN.md) and must not be edited to chase a test-split result.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from difflib import SequenceMatcher
from functools import lru_cache

import snowballstemmer

_WORD = re.compile(r"[a-z0-9]+")
_STEMMER = snowballstemmer.stemmer("english")

STOPWORDS = frozenset(
    """
    a an the and or but if of to in on at by for with from as is are was were be been being
    am do does did have has had i me my mine we our you your he she it its they them their
    this that these those what which who whom whose when where why how there here so than
    too very can could should would will shall may might must just about into over also not
    no yes any some all each every more most other such only own same up down out off again
    then once ever still now days currently current user users s t
    """.split()  # noqa: SIM905
)

# Each group is a set of words people use interchangeably for one personal topic.
SYNONYM_GROUPS: tuple[frozenset[str], ...] = tuple(
    frozenset(group.split())
    for group in (
        "live lives living reside resides residence home house based city town move moved "
        "relocate relocated hometown",
        "work works working job employer company employed office career firm workplace",
        "role title position profession occupation",
        "partner spouse wife husband girlfriend boyfriend fiance fiancee married dating",
        "pet dog cat puppy kitten",
        "allergy allergic allergies intolerance intolerant react reaction",
        "diet eat eating vegetarian vegan pescatarian keto meal meals food",
        "cuisine restaurant dish dishes dinner",
        "gym workout exercise train training lift fitness",
        "wake waking morning alarm early",
        "budget spend spending limit afford money",
        "doctor physician gp clinic",
        "car drive driving vehicle",
        "sibling brother sister",
        "language learn learning study studying practice",
        "race marathon run running runner",
        "meeting meetings call calls schedule",
        "birthday born birth",
    )
)

_SYNONYMS: dict[str, frozenset[str]] = {}
for _group in SYNONYM_GROUPS:
    for _word in _group:
        _SYNONYMS[_word] = _SYNONYMS.get(_word, frozenset()) | _group


@lru_cache(maxsize=65536)
def stem(word: str) -> str:
    """Snowball (Porter2) English stem, so "sibling" and "siblings" both become "sibl"."""
    return _STEMMER.stemWord(word)


def words(text: str) -> list[str]:
    """Lowercased content words, stopwords removed, not stemmed."""
    return [w for w in _WORD.findall(text.lower()) if w not in STOPWORDS]


def tokens(text: str) -> list[str]:
    """Stemmed content words: the unit BM25 indexes."""
    return [stem(w) for w in words(text)]


def expand(text: str) -> dict[str, float]:
    """Query terms with weights: 1.0 for words typed, 0.6 for synonyms of them."""
    weights: dict[str, float] = {}
    for word in words(text):
        weights[stem(word)] = 1.0
    for word in words(text):
        for synonym in _SYNONYMS.get(word, ()):
            weights.setdefault(stem(synonym), 0.6)
    return weights


def canonical_key(key: str | None) -> str | None:
    """Normalize a memory key so trivial spelling differences still match.

    "Home City", "home-city" and "user_home_city" all become "home_city". Genuinely
    different names ("residence") stay different; the benchmark's key-noise arm measures
    what that costs.
    """
    if key is None:
        return None
    slug = "_".join(_WORD.findall(key.lower()))
    for prefix in ("user_", "my_", "current_"):
        if slug.startswith(prefix):
            slug = slug[len(prefix) :]
    return slug.removesuffix("_current") or None


def similar_keys(key: str | None, existing: Iterable[str]) -> list[str]:
    """Existing canonical keys that look like a variant of ``key``, closest first.

    Catches reordered words ("city_home" for "home_city"), an added or dropped word
    ("diet_type" for "diet") and misspellings ("employeer"). Synonyms with no word in
    common ("residence" for "home_city") are not caught; the prompt's list of keys in use
    is what guards against those.
    """
    canon = canonical_key(key)
    if canon is None:
        return []
    words = set(canon.split("_"))
    scored = []
    for other in set(existing) - {canon}:
        other_words = set(other.split("_"))
        overlap = len(words & other_words) / len(words | other_words)
        spelling = SequenceMatcher(None, canon, other).ratio()
        score = max(overlap, spelling)
        if overlap >= 0.5 or spelling >= 0.85:
            scored.append((-score, other))
    return [other for _, other in sorted(scored)]


def estimate_tokens(text: str) -> int:
    """Rough prompt size: about four characters per token for English."""
    return max(1, round(len(text) / 4))
