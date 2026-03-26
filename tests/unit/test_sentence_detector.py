"""Tests for SentenceBoundaryDetector."""

from backend.sentence_detector import SentenceBoundaryDetector


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
