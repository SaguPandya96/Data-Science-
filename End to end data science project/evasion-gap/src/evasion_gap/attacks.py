"""Text transformations used to test classifier robustness.

Each function obfuscates text while keeping it readable to a person. All of them
are simple string operations, which is the relevant point for threat modelling:
an attacker does not need model access or any ML to apply them.
"""

from typing import Callable, Dict

ZWSP = "​"

# Latin characters and Cyrillic characters that look the same in most fonts.
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
    """Return the text unchanged, used as the control condition."""
    return text


def homoglyph(text: str) -> str:
    """Replace lowercase Latin letters with Cyrillic look-alikes."""
    return "".join(HOMOGLYPHS.get(ch, ch) for ch in text)


def leetspeak(text: str) -> str:
    """Replace letters with visually similar digits."""
    return "".join(LEET.get(ch, ch) for ch in text)


def zero_width(text: str) -> str:
    """Insert a zero-width space between every character."""
    return ZWSP.join(text)


def spaced(text: str) -> str:
    """Separate the characters of each word with spaces."""
    return " ".join(" ".join(word) for word in text.split())


def repeated(text: str, n: int = 3) -> str:
    """Repeat each vowel n times."""
    return "".join(ch * n if ch.lower() in VOWELS else ch for ch in text)


def devowel(text: str) -> str:
    """Remove vowels from each word except the first character."""
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
