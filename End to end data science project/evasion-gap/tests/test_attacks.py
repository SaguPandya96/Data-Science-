"""Tests for the text transformations.

The transformations are only meaningful if the output is still readable, so the
tests check that each one changes the text in the specific way it claims to and
does not simply destroy it.
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
def test_changes_text(name, transform):
    assert transform(SAMPLE) != SAMPLE


@pytest.mark.parametrize("name,transform", ATTACKS.items())
def test_handles_empty_and_whitespace(name, transform):
    transform("")
    transform("   ")


def test_homoglyph_preserves_length():
    assert len(homoglyph(SAMPLE)) == len(SAMPLE)


def test_homoglyph_replaces_all_mapped_characters():
    assert not any(ch in HOMOGLYPHS for ch in homoglyph(SAMPLE))


def test_zero_width_is_reversible():
    assert zero_width(SAMPLE).replace(ZWSP, "") == SAMPLE


def test_spaced_produces_single_characters():
    assert all(len(token) == 1 for token in spaced(SAMPLE).split())


def test_leetspeak_substitutions():
    assert leetspeak("aeiost") == "431057"


def test_repeated_triples_vowels():
    assert repeated("hate") == "haaateee"


def test_devowel_keeps_first_character():
    assert devowel("idiot apple") == "idt appl"


def test_devowel_preserves_word_count():
    assert len(devowel(SAMPLE).split()) == len(SAMPLE.split())
