"""Tests for the normalization pass."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evasion_gap.attacks import homoglyph, zero_width  # noqa: E402
from evasion_gap.defense import (  # noqa: E402
    fold_homoglyphs,
    normalize_text,
    strip_format_chars,
)

SAMPLE = "you are such an idiot"


def test_reverses_homoglyph_attack():
    assert normalize_text(homoglyph(SAMPLE)) == SAMPLE


def test_reverses_zero_width_attack():
    assert normalize_text(zero_width(SAMPLE)) == SAMPLE


def test_leaves_clean_text_unchanged():
    assert normalize_text(SAMPLE) == SAMPLE


def test_strip_format_chars_removes_zero_width():
    assert strip_format_chars("a​b‍b") == "abb"


def test_fold_homoglyphs_maps_cyrillic_to_latin():
    assert fold_homoglyphs("аео") == "aeo"


def test_nfkc_folds_fullwidth_characters():
    assert normalize_text("ｉｄｉｏｔ") == "idiot"


def test_handles_empty_string():
    assert normalize_text("") == ""
