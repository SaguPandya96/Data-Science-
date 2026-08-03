"""Attack invariants.

An evasion is only interesting if a human still reads it as the original. These
tests pin that property down so a future transform cannot quietly degrade into noise.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evasion_gap.attacks import (  # noqa: E402
    ATTACKS,
    HOMOGLYPHS,
    ZWSP,
    devowel,
    homoglyph,
    leetspeak,
    repeated,
    spaced,
    zero_width,
)

SAMPLE = "you are such an idiot"


@pytest.mark.parametrize("name,transform", ATTACKS.items())
def test_returns_string(name, transform):
    assert isinstance(transform(SAMPLE), str)


@pytest.mark.parametrize("name,transform", [(n, t) for n, t in ATTACKS.items() if n != "clean"])
def test_actually_changes_text(name, transform):
    assert transform(SAMPLE) != SAMPLE, f"{name} left the input untouched"


@pytest.mark.parametrize("name,transform", ATTACKS.items())
def test_handles_empty_and_whitespace(name, transform):
    transform("")
    transform("   ")


def test_homoglyph_preserves_length():
    """One-for-one codepoint swap, so length is invariant."""
    assert len(homoglyph(SAMPLE)) == len(SAMPLE)


def test_homoglyph_leaves_no_swappable_latin():
    out = homoglyph(SAMPLE)
    assert not any(ch in HOMOGLYPHS for ch in out)


def test_zero_width_strips_back_to_original():
    """The defining property: invisible to a reader, disruptive to a tokenizer."""
    assert zero_width(SAMPLE).replace(ZWSP, "") == SAMPLE


def test_spaced_yields_single_characters():
    assert all(len(token) == 1 for token in spaced(SAMPLE).split())


def test_leetspeak_maps_known_characters():
    assert leetspeak("aeiost") == "431057"


def test_repeated_triples_vowels():
    assert repeated("hate") == "haaateee"


def test_devowel_keeps_word_initial_character():
    assert devowel("idiot apple") == "idt appl"


def test_devowel_preserves_word_count():
    assert len(devowel(SAMPLE).split()) == len(SAMPLE.split())
