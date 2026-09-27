"""Tests for `SentenceBoundaryDetector.feed_segments` / `flush_segments`.

The raw-segment API is the formatting-preserving variant the chat streaming
pipeline (`ConversationHandler._stream_llm_pass`) uses. It lives beside
`feed` / `flush`, which the voice TTS path keeps using unchanged
(`tests/unit/test_sentence_detector.py` pins those). These tests live under
`tests/unit/chat/` because the chat pipeline is the segment API's consumer.
"""

import pytest

from backend.sentence_detector import SentenceBoundaryDetector

CORPUS = [
    "First paragraph here.\n\nSecond paragraph here.\n1. item one.\n2. item two.",
    "Hello world. How are you doing today? I am fine!",
    "Dr. Smith went to Washington. He arrived at 3.5 hours past noon.",
    "I see. OK. That makes sense to me now.",
    "Wait... what happened here? Nothing much.",
    "  Leading spaces here.   Trailing spaces too.   ",
    'She said "stop." Then she left the room.',
    "No terminal punctuation at all",
]


def _splits(text: str) -> list[list[str]]:
    return [
        [text],
        list(text),
        [text[i : i + 3] for i in range(0, len(text), 3)],
        [text[i : i + 7] for i in range(0, len(text), 7)],
    ]


def _run_segments(chunks: list[str]) -> list[str]:
    detector = SentenceBoundaryDetector()
    out: list[str] = []
    for chunk in chunks:
        out.extend(detector.feed_segments(chunk))
    out.extend(detector.flush_segments())
    return out


def _run_feed(chunks: list[str]) -> list[str]:
    detector = SentenceBoundaryDetector()
    out: list[str] = []
    for chunk in chunks:
        out.extend(detector.feed(chunk))
    out.extend(detector.flush())
    return out


@pytest.mark.parametrize("text", CORPUS)
def test_segments_concatenate_to_input_exactly(text):
    for chunks in _splits(text):
        assert "".join(_run_segments(chunks)) == text


@pytest.mark.parametrize("text", CORPUS)
def test_segment_boundaries_match_feed(text):
    """Same boundaries as `feed`: stripping each segment and collapsing its
    internal whitespace gives exactly what `feed` returns for the same
    chunking (`feed` joins merged short sentences with one space).
    """
    for chunks in _splits(text):
        segments = [" ".join(s.split()) for s in _run_segments(chunks) if s.strip()]
        sentences = [" ".join(s.split()) for s in _run_feed(chunks)]
        assert segments == sentences, chunks


def test_segment_keeps_newlines_that_feed_strips():
    detector = SentenceBoundaryDetector()
    assert detector.feed_segments("First paragraph here.\n\nSecond") == [
        "First paragraph here.\n\n"
    ]
    assert detector.flush_segments() == ["Second"]


def test_short_sentence_merges_with_original_whitespace():
    detector = SentenceBoundaryDetector()
    segments = detector.feed_segments("That is right.\nOK.\nNext sentence here. ")
    assert segments == ["That is right.\nOK.\n", "Next sentence here. "]


def test_flush_segments_empty_buffer():
    assert SentenceBoundaryDetector().flush_segments() == []
