"""Cut a reply's Token stream into the text segments TTS synthesizes one call at a time.

The first segment decides when MIST starts speaking, so it is cut at the first clause boundary
past `first_segment_min_chars` (`find_clause_boundary`, the same function that cuts the handler's
first emission). Later segments are whole sentences, the raw segments of
`SentenceBoundaryDetector.feed_segments`, so the voice keeps its sentence prosody.

The handler emits each gated sentence as one Token and holds its trailing whitespace back to
lead the next Token, and the next Token arrives only when its whole sentence is ready. A
boundary detector that waits for the whitespace after a period would therefore hold every
segment back by one sentence. So a Token's end is a boundary in its own right: a Token ending in
sentence punctuation (or, for the first segment, clause punctuation) is cut at its end at once.

Every segment's span locates it in the concatenation of all the turn's Token texts, error text
included, which is the text the client displays and the playback ledger maps played audio back
to.
"""

import re
from dataclasses import dataclass

from ..sentence_detector import SentenceBoundaryDetector, find_clause_boundary
from .budget import SegmenterConfig
from .speech import TextSegment

# A Token ending in one of these may be cut at its end: `.`, `!` or `?` for any segment, and `,`,
# `;` or `:` too for the first, each optionally followed by a closing quote or bracket.
_SENTENCE_END = re.compile(r"""[.!?]["')}\]]?$""")
_CLAUSE_END = re.compile(r"""[.!?,;:]["')}\]]?$""")


@dataclass(frozen=True, slots=True)
class SpeechSegment:
    """One TTS segment and its index in the turn.

    The index is the segment's position among the turn's segments, from 0. The turn controller
    anchors scene intents to it (`AtSegment`) and the pacer reports it when the segment's first
    frame goes out, so both must use the segmenter's count: a segment that synthesizes to no
    audio still takes its index.
    """

    index: int
    segment: TextSegment


class SpeechSegmenter:
    """Turns one reply's Tokens, fed in order, into TTS segments.

    One segmenter serves one turn. `feed` is called for each Token, with the Token's `is_error`;
    `flush` is called once at the end of the stream.
    """

    def __init__(self, config: SegmenterConfig) -> None:
        self._config = config
        self._text = ""
        # Start of the text not yet cut into a segment; everything before it is segmented,
        # error text, or whitespace between segments.
        self._pending_start = 0
        self._next_index = 0

    @property
    def text(self) -> str:
        """Every Token text fed so far, concatenated: the text the client displays."""
        return self._text

    @property
    def next_index(self) -> int:
        """The index the next segment will get."""
        return self._next_index

    def feed(self, text: str, is_error: bool) -> list[SpeechSegment]:
        """Take one Token's text and return the segments it completes.

        Error text is displayed but never spoken: it is counted in the spans and never
        segmented. Text pending when error text arrives is cut at once, so a segment never
        spans the error text.
        """
        if is_error:
            segments = self._cut_rest()
            self._text += text
            self._pending_start = len(self._text)
            return segments
        self._text += text
        return self._cut(token_end=text)

    def flush(self) -> list[SpeechSegment]:
        """Cut whatever text is still pending at the end of the stream."""
        return self._cut_rest()

    def _cut(self, token_end: str) -> list[SpeechSegment]:
        segments: list[SpeechSegment] = []
        while True:
            self._skip_whitespace()
            pending = self._text[self._pending_start :]
            if not pending:
                return segments
            cut = self._find_cut(pending, token_end)
            if cut is None:
                return segments
            segments += self._emit(cut)

    def _find_cut(self, pending: str, token_end: str) -> int | None:
        """Where the pending text is cut next, relative to its start, or None to wait."""
        if self._next_index == 0:
            cut = find_clause_boundary(pending, self._config.first_segment_min_chars)
            if cut is None and (
                _CLAUSE_END.search(token_end)
                and len(pending) >= self._config.first_segment_min_chars
            ):
                cut = len(pending)
        else:
            raw = SentenceBoundaryDetector().feed_segments(pending)
            cut = len(raw[0]) if raw else None
            if cut is None and _SENTENCE_END.search(token_end):
                cut = len(pending)
        if cut is None and len(pending) > self._config.max_segment_chars:
            # The pending text is already longer than a segment may be, so its first piece is
            # fixed whatever arrives next; `_emit` splits it at the cap.
            cut = len(pending)
        return cut

    def _cut_rest(self) -> list[SpeechSegment]:
        segments: list[SpeechSegment] = []
        while True:
            self._skip_whitespace()
            pending_chars = len(self._text) - self._pending_start
            if pending_chars == 0:
                return segments
            segments += self._emit(pending_chars)

    def _emit(self, cut: int) -> list[SpeechSegment]:
        """Cut the first `cut` characters of pending text as one segment, at most the cap long.

        The rest of a piece split at the cap stays pending and goes back through the cut rules:
        it may end at a boundary of its own, or wait for more text.
        """
        end = self._pending_start + cut
        self._skip_whitespace(limit=end)
        piece = self._text[self._pending_start : end]
        if not piece:
            return []
        length = len(piece)
        if length > self._config.max_segment_chars:
            length = self._split_point(piece)
        return [self._segment(length)]

    def _split_point(self, piece: str) -> int:
        """The cut for an over-long piece: its last clause boundary within the cap, else its
        last whitespace within the cap, else the cap itself.
        """
        cap = self._config.max_segment_chars
        window = piece[: cap + 1]
        best: int | None = None
        cut = find_clause_boundary(window, 1)
        while cut is not None and cut <= cap:
            best = cut
            cut = find_clause_boundary(window, cut + 1)
        if best is not None:
            return best
        space = next((i for i in range(len(window) - 1, 0, -1) if window[i].isspace()), None)
        return space if space is not None else cap

    def _segment(self, length: int) -> SpeechSegment:
        start = self._pending_start
        raw = self._text[start : start + length]
        text = raw.rstrip()
        segment = SpeechSegment(self._next_index, TextSegment(text, start, start + len(text)))
        self._next_index += 1
        self._pending_start = start + length
        return segment

    def _skip_whitespace(self, limit: int | None = None) -> None:
        end = len(self._text) if limit is None else limit
        while self._pending_start < end and self._text[self._pending_start].isspace():
            self._pending_start += 1
