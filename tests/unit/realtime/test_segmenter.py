"""Tests for `SpeechSegmenter`, which cuts a reply's Token stream into TTS segments.

The first segment decides when MIST starts speaking, so it is cut early, at the first clause
boundary past `first_segment_min_chars`. The handler emits each gated sentence as one Token and
holds its trailing whitespace back for the next Token, so the segmenter must treat a Token's end
as a boundary: waiting for the whitespace would hold every segment back by a whole sentence. The
spans are what the playback ledger maps played audio back to, so each must locate its segment in
the text the client displays, the concatenation of every Token's text.
"""

import pytest

from backend.realtime.budget import SegmenterConfig
from backend.realtime.segmenter import SpeechSegment, SpeechSegmenter
from backend.realtime.speech import TextSegment


def _texts(segments: list[SpeechSegment]) -> list[str]:
    return [segment.segment.text for segment in segments]


def _feed_all(segmenter: SpeechSegmenter, tokens: list[str]) -> list[SpeechSegment]:
    segments: list[SpeechSegment] = []
    for token in tokens:
        segments += segmenter.feed(token, False)
    return segments + segmenter.flush()


class TestHandlerTokenShapes:
    def test_a_token_ending_a_long_first_sentence_is_cut_at_its_end(self):
        segmenter = SpeechSegmenter(SegmenterConfig())
        first = "Here is the reply, which runs long enough to pass."

        segments = segmenter.feed(first, False)

        assert len(first) == 50
        assert segments == [SpeechSegment(0, TextSegment(first, 0, 50))]

    def test_a_later_token_ending_a_sentence_is_cut_at_once_after_its_space(self):
        segmenter = SpeechSegmenter(SegmenterConfig())
        segmenter.feed("Here is the reply, which runs long enough to pass.", False)

        segments = segmenter.feed(" Second one.", False)

        assert segments == [SpeechSegment(1, TextSegment("Second one.", 51, 62))]

    def test_an_unfinished_token_waits_for_the_end_of_stream_flush(self):
        segmenter = SpeechSegmenter(SegmenterConfig())
        segmenter.feed("Here is the reply, which runs long enough to pass.", False)
        segmenter.feed(" Second one.", False)

        held = segmenter.feed(" And a third", False)
        flushed = segmenter.flush()

        assert held == []
        assert flushed == [SpeechSegment(2, TextSegment("And a third", 63, 74))]

    def test_a_comma_past_the_minimum_cuts_the_first_segment_then_the_token_end_cuts_the_rest(
        self,
    ):
        segmenter = SpeechSegmenter(SegmenterConfig())

        segments = segmenter.feed("Here is the first sentence of the reply, which runs long.", False)

        assert segments == [
            SpeechSegment(0, TextSegment("Here is the first sentence of the reply,", 0, 40)),
            SpeechSegment(1, TextSegment("which runs long.", 41, 57)),
        ]


class TestFirstSegment:
    def test_a_short_token_end_waits_for_the_minimum(self):
        segmenter = SpeechSegmenter(SegmenterConfig(first_segment_min_chars=40))

        held = segmenter.feed("Short one.", False)
        segments = segmenter.feed(" Then a longer continuation, past forty chars.", False)

        assert held == []
        assert _texts(segments) == ["Short one. Then a longer continuation, past forty chars."]

    @pytest.mark.parametrize(
        "token",
        [
            pytest.param("Here is a reply that runs on past the minimum,", id="comma"),
            pytest.param("Here is a reply that runs on past the minimum;", id="semicolon"),
            pytest.param("Here is a reply that runs on past the minimum:", id="colon"),
            pytest.param("Here is a reply that runs on past the minimum!", id="exclamation"),
            pytest.param('Here is a reply that runs on past "the minimum."', id="closing-quote"),
            pytest.param("Here is a reply that runs on past (the minimum.)", id="closing-bracket"),
        ],
    )
    def test_a_token_ending_in_clause_punctuation_cuts_the_first_segment(self, token):
        segmenter = SpeechSegmenter(SegmenterConfig(first_segment_min_chars=40))

        segments = segmenter.feed(token, False)

        assert _texts(segments) == [token]

    def test_a_token_ending_mid_word_does_not_cut(self):
        segmenter = SpeechSegmenter(SegmenterConfig(first_segment_min_chars=40))

        segments = segmenter.feed("Here is a reply that runs on past the minimum", False)

        assert segments == []


class TestLaterSegments:
    def test_later_segments_are_the_detectors_raw_segments(self):
        segmenter = SpeechSegmenter(SegmenterConfig(first_segment_min_chars=10))

        segments = segmenter.feed(
            "First clause here, and more. Second sentence is here. Third", False
        )

        assert _texts(segments) == [
            "First clause here,",
            "and more.",
            "Second sentence is here.",
        ]
        assert _texts(segmenter.flush()) == ["Third"]

    def test_a_later_token_ending_in_a_comma_is_not_cut(self):
        segmenter = SpeechSegmenter(SegmenterConfig(first_segment_min_chars=10))
        segmenter.feed("Okay then, here we go.", False)

        held = segmenter.feed(" And then,", False)
        segments = segmenter.feed(" more words.", False)

        assert held == []
        assert _texts(segments) == ["And then, more words."]

    @pytest.mark.parametrize("ending", ["?", "!", '."', ".)"])
    def test_a_later_token_ending_a_sentence_is_cut(self, ending):
        segmenter = SpeechSegmenter(SegmenterConfig(first_segment_min_chars=10))
        segmenter.feed("Okay then, here we go.", False)

        segments = segmenter.feed(f" Is that right{ending}", False)

        assert _texts(segments) == [f"Is that right{ending}"]


class TestMaxSegmentChars:
    def test_a_long_segment_is_split_at_the_last_clause_boundary_before_the_cap(self):
        segmenter = SpeechSegmenter(SegmenterConfig(first_segment_min_chars=10))
        segmenter.feed("Okay then, here we go.", False)
        sentence = (
            " This sentence runs on, and on, and keeps going well past the two hundred"
            " character cap that a single synthesis call is allowed to carry, so the"
            " segmenter has to split it somewhere sensible before it reaches the engine."
        )

        segments = segmenter.feed(sentence, False)

        assert all(len(text) <= 200 for text in _texts(segments))
        assert _texts(segments) == [
            "This sentence runs on, and on, and keeps going well past the two hundred"
            " character cap that a single synthesis call is allowed to carry,",
            "so the segmenter has to split it somewhere sensible before it reaches the engine.",
        ]

    def test_a_long_segment_without_a_clause_boundary_is_split_at_the_last_whitespace(self):
        segmenter = SpeechSegmenter(SegmenterConfig(first_segment_min_chars=10, max_segment_chars=30))

        segments = segmenter.feed("alpha bravo charlie delta echo foxtrot golf hotel", False)
        segments += segmenter.flush()

        assert _texts(segments) == ["alpha bravo charlie delta echo", "foxtrot golf hotel"]

    def test_text_past_the_cap_is_cut_before_its_sentence_ends(self):
        segmenter = SpeechSegmenter(SegmenterConfig(first_segment_min_chars=10, max_segment_chars=30))

        segments = segmenter.feed("alpha bravo charlie delta echo foxtrot", False)

        assert _texts(segments) == ["alpha bravo charlie delta echo"]

    def test_a_word_longer_than_the_cap_is_cut_at_the_cap(self):
        segmenter = SpeechSegmenter(SegmenterConfig(first_segment_min_chars=10, max_segment_chars=10))

        segments = segmenter.feed("abcdefghijklmnopqrstuvwxyz", False) + segmenter.flush()

        assert _texts(segments) == ["abcdefghij", "klmnopqrst", "uvwxyz"]


class TestErrorText:
    def test_error_text_is_never_segmented(self):
        segmenter = SpeechSegmenter(SegmenterConfig())

        segments = segmenter.feed("I encountered an error: the model is unreachable.", True)

        assert segments == []
        assert segmenter.flush() == []

    def test_an_error_token_cuts_the_text_before_it_and_spans_count_its_characters(self):
        segmenter = SpeechSegmenter(SegmenterConfig())
        before = "A reply that was cut off"
        error = "\n\nI encountered an error: boom"

        cut = segmenter.feed(before, False)
        at_error = segmenter.feed(error, True)
        after = segmenter.feed(" Later text.", False)

        assert cut == []
        assert at_error == [SpeechSegment(0, TextSegment(before, 0, len(before)))]
        start = len(before + error) + 1
        assert after == [SpeechSegment(1, TextSegment("Later text.", start, start + 11))]
        assert segmenter.text == before + error + " Later text."


class TestSpans:
    def test_every_span_locates_its_segment_in_the_displayed_text(self):
        segmenter = SpeechSegmenter(SegmenterConfig())
        tokens = [
            "Here is the plan for today, laid out step by step.",
            "\n\nFirst, we check the weather and pack accordingly.",
            "\n\n1. Bring a coat.",
            " Then we leave at nine, sharp, and walk to the station.",
            " Done",
        ]

        segments = _feed_all(segmenter, tokens)

        display = "".join(tokens)
        assert segmenter.text == display
        assert segments
        for segment in segments:
            text = segment.segment
            assert display[text.start_char : text.end_char] == text.text
            assert text.text == text.text.strip()
        assert [segment.index for segment in segments] == list(range(len(segments)))
        spoken = " ".join(display.split())
        assert " ".join(_texts(segments)) == spoken

    def test_next_index_names_the_index_of_the_next_segment(self):
        segmenter = SpeechSegmenter(SegmenterConfig())
        assert segmenter.next_index == 0

        segmenter.feed("Here is the reply, which runs long enough to pass.", False)
        segmenter.feed(" Second one.", False)

        assert segmenter.next_index == 2

    def test_whitespace_alone_makes_no_segment(self):
        segmenter = SpeechSegmenter(SegmenterConfig())

        segments = segmenter.feed("  \n ", False) + segmenter.flush()

        assert segments == []
