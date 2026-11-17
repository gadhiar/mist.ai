"""Tests for SentenceBoundaryDetector and `find_clause_boundary`."""

import pytest

from backend.sentence_detector import SentenceBoundaryDetector, find_clause_boundary


class TestSentenceBoundaryDetector:
    """Tests for sentence boundary detection in token streams."""

    def setup_method(self):
        self.detector = SentenceBoundaryDetector()

    def test_simple_sentence(self):
        """Period followed by space triggers boundary."""
        result = self.detector.feed("Hello there. ")
        assert result == ["Hello there."]

    def test_multiple_sentences(self):
        """Multiple sentences in one feed."""
        result = self.detector.feed("First sentence. Second sentence. ")
        assert result == ["First sentence.", "Second sentence."]

    def test_exclamation_mark(self):
        """Exclamation triggers boundary."""
        result = self.detector.feed("Wow! ")
        # Too short (<8 chars), stays in buffer
        assert result == []
        result = self.detector.feed("That is really great! ")
        # Buffer had "Wow! ", now "Wow! That is really great!"
        assert len(result) == 1
        assert "Wow" in result[0]

    def test_question_mark(self):
        """Question mark triggers boundary."""
        result = self.detector.feed("How are you doing today? ")
        assert result == ["How are you doing today?"]

    def test_abbreviation_no_split(self):
        """Common abbreviations do not trigger boundary."""
        result = self.detector.feed("Dr. Smith went to the store. ")
        assert result == ["Dr. Smith went to the store."]

    def test_decimal_no_split(self):
        """Decimal numbers do not trigger boundary."""
        result = self.detector.feed("The value is 3.14 meters. ")
        assert result == ["The value is 3.14 meters."]

    def test_ellipsis(self):
        """Ellipsis treated as single boundary."""
        result = self.detector.feed("Well... I suppose that works. ")
        assert len(result) >= 1

    def test_accumulation_across_feeds(self):
        """Tokens accumulate across multiple feed calls."""
        assert self.detector.feed("Hello") == []
        assert self.detector.feed(" there") == []
        assert self.detector.feed(". ") == ["Hello there."]

    def test_flush_remaining(self):
        """Flush emits remaining buffer."""
        self.detector.feed("Partial sentence without")
        result = self.detector.flush()
        assert result == ["Partial sentence without"]

    def test_flush_empty(self):
        """Flush on empty buffer returns nothing."""
        assert self.detector.flush() == []

    def test_min_length_merging(self):
        """Short sentences merge with previous."""
        result = self.detector.feed("This is a full sentence. Yes. And another sentence here. ")
        # "Yes." is <20 chars, should merge
        assert len(result) == 2

    def test_closing_quote(self):
        """Closing quote after terminal punctuation."""
        tokens = 'She said "hello there." Then she left. '
        result = self.detector.feed(tokens)
        assert len(result) == 2

    def test_list_marker_no_split(self):
        """Period after single digit (list marker) does not split."""
        result = self.detector.feed("Here are the steps: 1. First do this thing. ")
        assert "1." in result[0]

    def test_end_of_stream(self):
        """Feed tokens then flush gets remaining buffer."""
        result1 = self.detector.feed("First sentence. ")
        assert result1 == ["First sentence."]
        self.detector.feed("Second without ending")
        sentences = self.detector.flush()
        assert sentences == ["Second without ending"]


class TestFindClauseBoundary:
    """`find_clause_boundary` decides where the first TTS segment, and the handler's first
    emission, may be cut. A cut inside a number or an abbreviation makes the voice pause
    mid-phrase, and one that waits for the end of the sentence delays the first audio.
    """

    def test_returns_the_index_just_past_a_comma_at_the_minimum(self):
        text = "Here is the first sentence of the reply, which runs long."

        cut = find_clause_boundary(text, 40)

        assert cut == 40
        assert text[:cut] == "Here is the first sentence of the reply,"

    def test_skips_a_boundary_before_the_minimum(self):
        text = "Here is the reply, which runs long enough to pass."

        assert find_clause_boundary(text, 40) is None
        assert find_clause_boundary(text, 0) == len("Here is the reply,")

    @pytest.mark.parametrize(
        "text",
        [
            pytest.param("Here is a clause,", id="comma-at-end"),
            pytest.param("It costs 1,000 dollars", id="thousands-separator"),
            pytest.param("Meet at 10:30 today", id="clock-time"),
            pytest.param("It is done.", id="period-at-end"),
        ],
    )
    def test_needs_whitespace_after_the_boundary(self, text):
        assert find_clause_boundary(text, 0) is None

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            pytest.param("First part; second part", "First part;", id="semicolon"),
            pytest.param("Note this: it matters", "Note this:", id="colon"),
            pytest.param("I thought so - but no", "I thought so -", id="hyphen-between-spaces"),
            pytest.param("I thought so — but no", "I thought so —", id="em-dash"),
            pytest.param("I thought so – but no", "I thought so –", id="en-dash"),
            pytest.param("I thought so -- but no", "I thought so --", id="double-hyphen"),
            pytest.param("It is done. Next one", "It is done.", id="sentence-period"),
            pytest.param("Is it done? Next one", "Is it done?", id="sentence-question"),
            pytest.param('He said "go." Then he left', 'He said "go."', id="closing-quote"),
            pytest.param('She said "yes," and left', 'She said "yes,"', id="comma-then-quote"),
        ],
    )
    def test_cuts_just_past_each_kind_of_boundary(self, text, expected):
        assert find_clause_boundary(text, 0) == len(expected)

    def test_a_dash_inside_a_word_is_not_a_boundary(self):
        assert find_clause_boundary("A well-known fact stays whole", 0) is None

    @pytest.mark.parametrize(
        "text",
        [
            pytest.param("Dr. Smith went home today", id="abbreviation"),
            pytest.param("It measured 3.5 metres today", id="decimal"),
            pytest.param("Steps are 1. mix and bake", id="list-marker"),
            pytest.param("Well... maybe later on", id="ellipsis"),
        ],
    )
    def test_periods_the_sentence_detector_skips_are_not_boundaries(self, text):
        assert find_clause_boundary(text, 0) is None

    def test_returns_the_first_boundary_at_or_after_the_minimum(self):
        text = "One, two, three, four"

        assert find_clause_boundary(text, 5) == len("One, two,")
        assert find_clause_boundary(text, len("One, two,")) == len("One, two,")
        assert find_clause_boundary(text, len("One, two,") + 1) == len("One, two, three,")

    def test_returns_none_for_text_without_a_boundary(self):
        assert find_clause_boundary("no boundary anywhere in here", 0) is None
        assert find_clause_boundary("", 0) is None
