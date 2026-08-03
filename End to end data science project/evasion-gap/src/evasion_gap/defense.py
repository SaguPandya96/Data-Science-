"""Unicode normalization applied to text before it reaches the classifier.

Implemented as preprocessing rather than a model change so it can be evaluated
without retraining and rolled back independently of the model.
"""

import unicodedata

from .attacks import HOMOGLYPHS

HOMOGLYPH_REVERSE = {v: k for k, v in HOMOGLYPHS.items()}


def strip_format_chars(text: str) -> str:
    """Remove Unicode category Cf characters such as zero-width spaces."""
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cf")


def fold_homoglyphs(text: str) -> str:
    """Map known Cyrillic look-alikes back to their Latin equivalents."""
    return "".join(HOMOGLYPH_REVERSE.get(ch, ch) for ch in text)


def normalize_text(text: str) -> str:
    """Apply the full normalization pass.

    NFKC runs last so that compatibility folding also handles fullwidth and
    styled character variants that the explicit homoglyph map does not list.
    """
    return unicodedata.normalize("NFKC", fold_homoglyphs(strip_format_chars(text)))
