"""Tests for TTS text preprocessing.

Verifies that prosody-critical punctuation is preserved or substituted
correctly rather than stripped, so CSM-1B can produce natural pacing.
"""

import pytest

from src.multimodal.text_preprocessing import preprocess_text_for_tts


class TestPreprocessTextForTts:
    """Core preprocessing behavior."""

    def test_preserves_basic_punctuation(self):
        assert preprocess_text_for_tts("Hello, world!") == "hello, world!"

    def test_preserves_question_marks(self):
        assert preprocess_text_for_tts("How are you?") == "how are you?"

    def test_preserves_periods(self):
        assert preprocess_text_for_tts("Done.") == "done."

    def test_preserves_apostrophes(self):
        result = preprocess_text_for_tts("It's working")
        assert "it's" in result


class TestDashHandling:
    """Dashes and hyphens become word boundaries or pauses."""

    def test_hyphen_between_words_becomes_space(self):
        assert preprocess_text_for_tts("well-known") == "well known"

    def test_em_dash_becomes_comma(self):
        assert preprocess_text_for_tts("word\u2014another") == "word, another"

    def test_en_dash_becomes_comma(self):
        assert preprocess_text_for_tts("word\u2013another") == "word, another"


class TestPauseSubstitutions:
    """Colons and semicolons are replaced with equivalent pause markers."""

    def test_colons_become_periods(self):
        assert preprocess_text_for_tts("Example: this") == "example. this"

    def test_semicolons_become_commas(self):
        assert preprocess_text_for_tts("first; second") == "first, second"


class TestEllipses:
    """Ellipses preserved for trailing-off / hesitation."""

    def test_ellipses_preserved(self):
        assert preprocess_text_for_tts("Wait...") == "wait..."

    def test_two_dots_normalized_to_ellipsis(self):
        assert preprocess_text_for_tts("Wait..") == "wait..."

    def test_many_dots_normalized_to_ellipsis(self):
        assert preprocess_text_for_tts("Wait.....") == "wait..."


class TestParenthesesAndBrackets:
    """Grouping characters removed but content kept."""

    def test_parentheses_removed_content_kept(self):
        result = preprocess_text_for_tts("The algorithm (BERT) works")
        assert "bert" in result
        assert "(" not in result
        assert ")" not in result

    def test_brackets_removed(self):
        result = preprocess_text_for_tts("See [reference]")
        assert "[" not in result
        assert "]" not in result


class TestQuotationMarks:
    """Quotation marks stripped."""

    def test_double_quotes_removed(self):
        result = preprocess_text_for_tts('He said "hello"')
        assert '"' not in result
        assert "hello" in result

    def test_smart_double_quotes_removed(self):
        result = preprocess_text_for_tts("He said \u201chello\u201d")
        assert "\u201c" not in result
        assert "\u201d" not in result

    def test_curly_apostrophe_preserved_as_straight(self):
        """LLMs frequently emit curly apostrophes in contractions."""
        assert preprocess_text_for_tts("don\u2019t") == "don't"
        assert preprocess_text_for_tts("it\u2019s") == "it's"
        assert preprocess_text_for_tts("I\u2019m") == "i'm"

    def test_curly_single_quotes_become_apostrophe(self):
        result = preprocess_text_for_tts("\u2018hello\u2019")
        assert result == "'hello'"


class TestWhitespace:
    """Whitespace normalization."""

    def test_multiple_spaces_collapsed(self):
        assert preprocess_text_for_tts("hello   world") == "hello world"

    def test_leading_trailing_stripped(self):
        assert preprocess_text_for_tts("  hello  ") == "hello"

    def test_space_after_punctuation(self):
        # "hello,world" -> "hello, world"
        assert preprocess_text_for_tts("hello,world") == "hello, world"


class TestLowercasing:
    """CSM-1B expects lowercase input."""

    def test_lowercased(self):
        assert preprocess_text_for_tts("NASA launched") == "nasa launched"

    def test_mixed_case_lowercased(self):
        assert preprocess_text_for_tts("Hello World") == "hello world"


class TestCombined:
    """Integration-level tests with multiple transformations."""

    @pytest.mark.parametrize(
        "input_text, expected",
        [
            pytest.param(
                "Well -- it's complicated.",
                "well , it's complicated.",
                id="em-dash-in-sentence",
            ),
            pytest.param(
                "Step 1: Do this; then that.",
                "step 1. do this, then that.",
                id="colon-and-semicolon",
            ),
            pytest.param(
                "The model (CSM-1B) handles it...",
                "the model csm 1b handles it...",
                id="parens-hyphen-ellipsis",
            ),
        ],
    )
    def test_combined_transformations(self, input_text, expected):
        assert preprocess_text_for_tts(input_text) == expected

    def test_empty_string(self):
        assert preprocess_text_for_tts("") == ""

    def test_only_whitespace(self):
        assert preprocess_text_for_tts("   ") == ""
