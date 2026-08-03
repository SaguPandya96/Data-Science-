"""Evasion transforms for probing text-moderation classifiers.

Each transform maps a string to an obfuscated variant that a human reader still
parses as the original. They are deliberately cheap: the point is that an
adversary needs no ML, no gradients, and no model access to apply them.
"""

from typing import Callable, Dict

ZWSP = "​"

# Latin -> Cyrillic codepoints that render near-identically in most fonts.
HOMOGLYPHS = {
    "a": "а",
    "c": "с",
    "e": "е",
    "i": "і",
    "o": "о",
    "p": "р",
    "x": "х",
    "y": "у",
}

LEET = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"}

VOWELS = "aeiou"


def identity(text: str) -> str:
    """No-op baseline, so 'clean' flows through the same code path as the attacks."""
    return text


def homoglyph(text: str) -> str:
    """Swap lowercase Latin letters for visually identical Cyrillic ones."""
    return "".join(HOMOGLYPHS.get(ch, ch) for ch in text)


def leetspeak(text: str) -> str:
    """Classic character-for-digit substitution."""
    return "".join(LEET.get(ch, ch) for ch in text)


def zero_width(text: str) -> str:
    """Insert a zero-width space between every character."""
    return ZWSP.join(text)


def spaced(text: str) -> str:
    """Space out characters within each word: 'hate' -> 'h a t e'."""
    return " ".join(" ".join(word) for word in text.split())


def repeated(text: str, n: int = 3) -> str:
    """Repeat every vowel: 'hate' -> 'haaate'."""
    return "".join(ch * n if ch.lower() in VOWELS else ch for ch in text)


def devowel(text: str) -> str:
    """Drop vowels after the first character of each word: 'hate' -> 'ht'."""
    return " ".join(
        word[0] + "".join(ch for ch in word[1:] if ch.lower() not in VOWELS)
        for word in text.split()
    )


ATTACKS: Dict[str, Callable[[str], str]] = {
    "clean": identity,
    "homoglyph": homoglyph,
    "leetspeak": leetspeak,
    "zero_width": zero_width,
    "spaced": spaced,
    "repeated": repeated,
    "devowel": devowel,
}
