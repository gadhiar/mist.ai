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

        segments = segmenter.feed(
            "Here is the first sentence of the reply, which runs long.", False
        )

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
        segmenter = SpeechSegmenter(
            SegmenterConfig(first_segment_min_chars=10, max_segment_chars=30)
        )

        segments = segmenter.feed("alpha bravo charlie delta echo foxtrot golf hotel", False)
        segments += segmenter.flush()

        assert _texts(segments) == ["alpha bravo charlie delta echo", "foxtrot golf hotel"]

    def test_text_past_the_cap_is_cut_before_its_sentence_ends(self):
        segmenter = SpeechSegmenter(
            SegmenterConfig(first_segment_min_chars=10, max_segment_chars=30)
        )

        segments = segmenter.feed("alpha bravo charlie delta echo foxtrot", False)

        assert _texts(segments) == ["alpha bravo charlie delta echo"]

    def test_a_word_longer_than_the_cap_is_cut_at_the_cap(self):
        segmenter = SpeechSegmenter(
            SegmenterConfig(first_segment_min_chars=10, max_segment_chars=10)
        )

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


def _assert_spans_locate(segmenter: SpeechSegmenter, segments: list[SpeechSegment]) -> None:
    for segment in segments:
        text = segment.segment
        assert segmenter.text[text.start_char : text.end_char] == text.text
    assert [segment.index for segment in segments] == list(range(len(segments)))


class TestCodeBlocks:
    """Fenced code is displayed but never spoken: read aloud it is a string of symbols."""

    def test_a_fenced_block_is_never_segmented_and_the_prose_around_it_is(self):
        segmenter = SpeechSegmenter(SegmenterConfig(first_segment_min_chars=10))
        tokens = [
            "Here is the function.",
            "\n\n```python\ndef add(a, b):\n    return a + b.\n```",
            "\n\nCall it with two numbers.",
        ]

        segments = _feed_all(segmenter, tokens)

        assert _texts(segments) == ["Here is the function.", "Call it with two numbers."]
        assert segmenter.text == "".join(tokens)
        _assert_spans_locate(segmenter, segments)

    def test_prose_pending_when_a_fence_opens_is_cut_at_once(self):
        segmenter = SpeechSegmenter(SegmenterConfig(first_segment_min_chars=10))

        held = segmenter.feed("Here it is", False)
        at_fence = segmenter.feed("\n```\nprint(1)\n", False)

        assert held == []
        assert at_fence == [SpeechSegment(0, TextSegment("Here it is", 0, 10))]

    def test_a_fence_split_across_tokens_is_still_a_fence(self):
        segmenter = SpeechSegmenter(SegmenterConfig(first_segment_min_chars=10))
        tokens = ["Look at this.", "\n`", "``py\nx = 1.\n`", "``", "\nDone."]

        per_token = [segmenter.feed(token, False) for token in tokens]
        flushed = segmenter.flush()

        assert [_texts(segments) for segments in per_token] == [
            ["Look at this."],
            [],
            [],
            [],
            ["Done."],
        ]
        assert flushed == []
        _assert_spans_locate(segmenter, [s for segments in per_token for s in segments])

    def test_backticks_held_at_a_line_start_are_spoken_when_the_line_is_not_a_fence(self):
        segmenter = SpeechSegmenter(SegmenterConfig(first_segment_min_chars=10))

        first = segmenter.feed("Okay then.", False)
        held = segmenter.feed("\n`", False)
        inline = segmenter.feed("name` is the field.", False)

        assert _texts(first) == ["Okay then."]
        assert held == []
        assert inline == [SpeechSegment(1, TextSegment("`name` is the field.", 11, 31))]

    def test_a_tilde_fence_is_a_fence(self):
        segmenter = SpeechSegmenter(SegmenterConfig(first_segment_min_chars=10))
        tokens = ["Run this command now:", "\n~~~\nmake test.\n~~~\n", "That builds it."]

        segments = _feed_all(segmenter, tokens)

        assert _texts(segments) == ["Run this command now:", "That builds it."]

    @pytest.mark.parametrize(
        "tokens",
        [
            pytest.param(
                ["Okay then.", "\n```x``` is inline code here.", "\nAnd more prose."], id="whole"
            ),
            pytest.param(
                ["Okay then.", "\n``", "`x", "``` is inline", " code here.", "\nAnd more prose."],
                id="split",
            ),
        ],
    )
    def test_a_backtick_run_with_a_backtick_after_it_on_its_line_is_prose(self, tokens):
        segmenter = SpeechSegmenter(SegmenterConfig(first_segment_min_chars=10))

        segments = _feed_all(segmenter, tokens)

        assert _texts(segments) == [
            "Okay then.",
            "```x``` is inline code here.",
            "And more prose.",
        ]
        _assert_spans_locate(segmenter, segments)

    def test_a_backtick_after_the_run_shows_the_line_is_prose_before_the_line_ends(self):
        segmenter = SpeechSegmenter(SegmenterConfig(first_segment_min_chars=10))
        tokens = ["Okay then.", "\n```x", "``` is inline code here.", "\nMore."]

        per_token = [segmenter.feed(token, False) for token in tokens]

        assert [_texts(segments) for segments in per_token] == [
            ["Okay then."],
            [],
            ["```x``` is inline code here."],
            ["More."],
        ]
        assert segmenter.flush() == []

    def test_a_backtick_run_with_an_info_string_still_opens_a_fence(self):
        segmenter = SpeechSegmenter(SegmenterConfig(first_segment_min_chars=10))
        tokens = ["Here it is.", "\n```py", "thon", "\nprint(1).\n```", "\nDone."]

        per_token = [segmenter.feed(token, False) for token in tokens]

        assert [_texts(segments) for segments in per_token] == [
            ["Here it is."],
            [],
            [],
            [],
            ["Done."],
        ]
        assert segmenter.flush() == []

    def test_a_tilde_fence_with_a_backtick_in_its_info_string_still_opens_a_fence(self):
        segmenter = SpeechSegmenter(SegmenterConfig(first_segment_min_chars=10))
        tokens = ["Here it is.", "\n~~~ `odd` info\nprint(1).\n~~~", "\nDone."]

        segments = _feed_all(segmenter, tokens)

        assert _texts(segments) == ["Here it is.", "Done."]

    @pytest.mark.parametrize(
        "block",
        [
            pytest.param("````\n```\nstill code.\n````", id="shorter-run-does-not-close"),
            pytest.param("```\n~~~\nstill code.\n```", id="other-character-does-not-close"),
            pytest.param("```\n``` not a close\nstill code.\n```", id="text-after-does-not-close"),
            pytest.param("```\nstill code.\n   ```  ", id="indented-close-closes"),
        ],
    )
    def test_a_fence_closes_on_a_line_of_its_own_character_at_least_as_long(self, block):
        segmenter = SpeechSegmenter(SegmenterConfig(first_segment_min_chars=10))
        tokens = ["Before the block.", f"\n{block}", "\nAfter the block."]

        segments = _feed_all(segmenter, tokens)

        assert _texts(segments) == ["Before the block.", "After the block."]

    def test_an_indented_fence_in_a_list_item_is_a_fence(self):
        segmenter = SpeechSegmenter(SegmenterConfig(first_segment_min_chars=10))
        reply = (
            "Do these steps:\n\n1. Run the build:\n\n    ```bash\n    make all.\n    ```\n\n"
            "2. Check the output."
        )

        segments = _feed_all(segmenter, [reply])

        assert " ".join(_texts(segments)) == (
            "Do these steps: 1. Run the build: 2. Check the output."
        )
        _assert_spans_locate(segmenter, segments)

    def test_an_unclosed_fence_runs_to_the_end_of_the_stream(self):
        segmenter = SpeechSegmenter(SegmenterConfig(first_segment_min_chars=10))

        segments = _feed_all(segmenter, ["Here is a start.", "\n```\nunfinished code."])

        assert _texts(segments) == ["Here is a start."]

    def test_text_after_error_text_starts_a_line(self):
        segmenter = SpeechSegmenter(SegmenterConfig(first_segment_min_chars=10))
        segmenter.feed("I encountered an error: boom", True)

        segments = segmenter.feed("```\nprint(1).\n```", False) + segmenter.flush()

        assert segments == []


class TestTables:
    """Markdown table rows are displayed but never spoken: their cells make no sentence."""

    def test_table_rows_are_never_segmented_and_the_prose_around_them_is(self):
        segmenter = SpeechSegmenter(SegmenterConfig(first_segment_min_chars=10))
        tokens = [
            "Here are the totals.",
            "\n\n| Day | Total |\n|---|---|\n| Mon | 3. |\n",
            "\nThat is all.",
        ]

        segments = _feed_all(segmenter, tokens)

        assert _texts(segments) == ["Here are the totals.", "That is all."]
        _assert_spans_locate(segmenter, segments)

    def test_prose_pending_when_a_table_row_starts_is_cut_at_once(self):
        segmenter = SpeechSegmenter(SegmenterConfig(first_segment_min_chars=10))

        held = segmenter.feed("Totals by day", False)
        at_row = segmenter.feed("\n| Mon | 3 |", False)

        assert held == []
        assert at_row == [SpeechSegment(0, TextSegment("Totals by day", 0, 13))]

    def test_an_indented_row_is_a_table_row(self):
        segmenter = SpeechSegmenter(SegmenterConfig(first_segment_min_chars=10))

        segments = _feed_all(segmenter, ["Results follow.", "\n  | a | b. |\n", "Done here."])

        assert _texts(segments) == ["Results follow.", "Done here."]

    def test_a_row_split_across_tokens_is_still_a_row(self):
        segmenter = SpeechSegmenter(SegmenterConfig(first_segment_min_chars=10))
        tokens = ["Results follow.", "\n", "|", " a | b. |", "\n", "Done here."]

        per_token = [segmenter.feed(token, False) for token in tokens]

        assert [_texts(segments) for segments in per_token] == [
            ["Results follow."],
            [],
            [],
            [],
            [],
            ["Done here."],
        ]
        assert segmenter.flush() == []

    def test_a_pipe_inside_a_line_is_spoken(self):
        segmenter = SpeechSegmenter(SegmenterConfig(first_segment_min_chars=10))
        reply = "Pipe it with a | b, then read the output."

        segments = _feed_all(segmenter, [reply])

        assert " ".join(_texts(segments)) == reply


class TestSpansAroundUnspokenText:
    def test_spans_locate_every_segment_and_indexes_stay_contiguous(self):
        segmenter = SpeechSegmenter(SegmenterConfig())
        tokens = [
            "Here is the comparison you asked for, laid out as a table.",
            "\n\n| Plan | Cost |\n|---|---|\n| A | 3 |\n",
            "\nPlan A is cheaper. Here is how to install it:",
            "\n\n```sh\npip install a\n```",
            "\n\nThen run it once.",
            " Done",
        ]

        segments = _feed_all(segmenter, tokens)

        display = "".join(tokens)
        assert segmenter.text == display
        _assert_spans_locate(segmenter, segments)
        assert " ".join(_texts(segments)) == (
            "Here is the comparison you asked for, laid out as a table. Plan A is cheaper."
            " Here is how to install it: Then run it once. Done"
        )
