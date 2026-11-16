"""Tests for the character error rate the take gate and ASR agreement use (T6)."""

from __future__ import annotations

import pytest

from scripts.voice_data.stages.cer import character_error_rate


def test_identical_text_scores_zero() -> None:
    assert character_error_rate("the cat sat", "the cat sat") == 0.0


@pytest.mark.parametrize(
    ("hypothesis", "expected"),
    [
        ("cut", 1 / 3),  # one substitution
        ("cats", 1 / 3),  # one insertion
        ("ct", 1 / 3),  # one deletion
        ("dog", 1.0),  # three substitutions
    ],
)
def test_each_edit_counts_one_character(hypothesis: str, expected: float) -> None:
    assert character_error_rate("cat", hypothesis) == pytest.approx(expected)


def test_the_space_between_words_is_a_character() -> None:
    # "a cat" (5 characters) against "acat": one deletion.
    assert character_error_rate("a cat", "acat") == pytest.approx(1 / 5)


def test_text_is_normalised_first() -> None:
    assert character_error_rate("Ten cats, at 6:30!", "10 cats at six thirty") == 0.0
    assert character_error_rate("  The   CAT.  ", "the cat") == 0.0


def test_an_empty_reference_scores_the_hypothesis_length() -> None:
    assert character_error_rate("", "") == 0.0
    assert character_error_rate("", "hi") == 2.0
    assert character_error_rate("...", "") == 0.0


def test_an_empty_hypothesis_scores_one() -> None:
    assert character_error_rate("hello there", "") == 1.0


def test_unicode_punctuation_and_letters_on_both_sides() -> None:
    assert character_error_rate("It’s the café’s “menu”", "its the café's menu") == 0.0
    assert character_error_rate("naïve résumé", "naïve résumé") == 0.0
    # Only non-ASCII text normalises to nothing, which scores as an empty reference.
    assert character_error_rate("日本語", "abc") == 3.0
