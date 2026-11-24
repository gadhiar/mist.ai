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

Some of that text is displayed and never spoken: error text, fenced code blocks and Markdown
table rows. Code read aloud is a string of symbols, and table cells read in order make no
sentence, so a typed reply's code and tables stay on screen while the voice speaks the prose
around them. Unspoken text is counted in the spans and never segmented, and prose pending when
it starts is cut at once, so no segment spans it. The segmenter reads the text a line at a time:

- A fence opens on a line whose first character after any indentation (spaces or tabs) starts
  a run of three or more backticks or tildes. After a backtick run, the rest of the line (the
  info string) may not hold a backtick, or the line is prose: CommonMark's rule, so a line
  such as "```x``` is inline code" is spoken rather than silencing the rest of the reply. After
  a tilde run the rest of the line may hold anything.
- A fence closes on a later line holding, after any indentation, only a run of the same
  character at least as long as the opening run, then whitespace. An unclosed fence runs to
  the end of the stream. Every line from the opening to the closing fence is unspoken.
- A table row is a line, outside a fence, whose first character after any indentation is `|`.
  A pipe later in a line is prose, and so is a table written without leading pipes.

A line is classified from its first characters, which may arrive in different Tokens. Until
they show what the line is, its characters are held back, and prose before it is not cut:

- a line holding only indentation and one or two backticks or tildes waits for its next
  character, or for its end;
- a line starting with a run of three or more backticks waits for a backtick after the run,
  which makes it prose, or for its end, or for the end of the stream, because the rest of the
  line decides whether it opens a fence.

Error text ends a line, so the text after it is classified as a new line; a fence open before
it stays open.
"""

import re
from dataclasses import dataclass
from enum import Enum

from ..sentence_detector import SentenceBoundaryDetector, find_clause_boundary
from .budget import SegmenterConfig
from .speech import TextSegment

# A Token ending in one of these may be cut at its end: `.`, `!` or `?` for any segment, and `,`,
# `;` or `:` too for the first, each optionally followed by a closing quote or bracket.
_SENTENCE_END = re.compile(r"""[.!?]["')}\]]?$""")
_CLAUSE_END = re.compile(r"""[.!?,;:]["')}\]]?$""")

_FENCE_CHARS = "`~"
_INDENT = " \t"


class _Line(Enum):
    """What a line of reply text is, which decides whether it is spoken."""

    PROSE = "prose"
    TABLE_ROW = "table_row"
    FENCE_OPEN = "fence_open"
    CODE = "code"


@dataclass(frozen=True, slots=True)
class _Fence:
    """An open fence: its character and the length of its opening run."""

    char: str
    run: int


def _fence_of(line: str) -> _Fence:
    body = line.lstrip(_INDENT)
    char = body[0]
    return _Fence(char, len(body) - len(body.lstrip(char)))


def _closes(line: str, fence: _Fence) -> bool:
    body = line.strip()
    return len(body) >= fence.run and body == fence.char * len(body)


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
        # unspoken, or whitespace between segments.
        self._pending_start = 0
        # End of the prose known to be speakable. The text past it is unspoken, or a line whose
        # first characters do not yet show what it is.
        self._spoken_end = 0
        self._next_index = 0
        # The line being classified: where it starts, what it is once its first characters
        # show that, and the fence it lies in, if any.
        self._line_start = 0
        self._line: _Line | None = None
        self._fence: _Fence | None = None

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
        spans the error text. Code and table rows are skipped the same way, by
        `_classify_lines`.
        """
        if is_error:
            segments = self._classify_lines(final=True)
            segments += self._cut_rest()
            self._text += text
            self._pending_start = self._spoken_end = self._line_start = len(self._text)
            self._line = None
            return segments
        self._text += text
        segments = self._classify_lines(final=False)
        return segments + self._cut(token_end=text)

    def flush(self) -> list[SpeechSegment]:
        """Cut whatever text is still pending at the end of the stream."""
        segments = self._classify_lines(final=True)
        return segments + self._cut_rest()

    def _classify_lines(self, final: bool) -> list[SpeechSegment]:
        """Classify the lines fed so far, extend the speakable prose, and skip unspoken lines.

        Prose pending when an unspoken line starts goes through the usual cut rules up to the
        line's start, and what is left of it is cut there, so no segment spans the line.
        `final` means no more text follows the last line, so a line still waiting for its
        first characters is classified as it stands, and its kind is not kept.
        """
        segments: list[SpeechSegment] = []
        while True:
            newline = self._text.find("\n", self._line_start)
            complete = newline != -1
            end = newline + 1 if complete else len(self._text)
            line = self._text[self._line_start : end]
            kind = self._line if self._line is not None else self._kind_of(line, complete or final)
            if not final:
                self._line = kind
            if kind is None:
                # A line holding only indentation so far is whitespace, never spoken, so the
                # prose may run on over it. A line starting with backticks or tildes waits, with
                # the prose before it, until the line shows what it is.
                if line.strip():
                    self._spoken_end = max(self._spoken_end, self._line_start)
                else:
                    self._spoken_end = end
                return segments
            if kind is _Line.PROSE:
                self._spoken_end = end
            else:
                # No Token ends here, so the Token-end rule is off: an empty `token_end`.
                segments += self._cut("", limit=self._line_start)
                segments += self._cut_rest(limit=self._line_start)
                self._pending_start = self._spoken_end = max(self._pending_start, end)
                if kind is _Line.FENCE_OPEN:
                    # Recomputed while the line grows, so the run is the whole opening run.
                    self._fence = _fence_of(line)
                elif self._fence is not None and complete and _closes(line, self._fence):
                    self._fence = None
            if not complete:
                return segments
            self._line_start = end
            self._line = None

    def _kind_of(self, line: str, complete: bool) -> _Line | None:
        """What a line is, or None while the part of it fed so far cannot tell yet.

        Most lines are told by their first characters. A line starting with three backticks is
        prose as soon as a backtick follows the run, and a fence's opening line only once it is
        `complete` without one.
        """
        if self._fence is not None:
            return _Line.CODE
        body = line.lstrip(_INDENT)
        if body.startswith("|"):
            return _Line.TABLE_ROW
        if body.startswith("~~~"):
            return _Line.FENCE_OPEN
        if body.startswith("```"):
            if "`" in body.lstrip("`"):
                return _Line.PROSE
            return _Line.FENCE_OPEN if complete else None
        if not complete and (not body or any(body == char * len(body) for char in _FENCE_CHARS)):
            return None
        return _Line.PROSE

    def _cut(self, token_end: str, limit: int | None = None) -> list[SpeechSegment]:
        """Cut the pending prose, up to `limit` or the end of the speakable prose, by the rules.

        `token_end` is the Token just fed, for the Token-end rule; it is empty when the prose is
        cut short of the Token's end.
        """
        end = self._spoken_end if limit is None else limit
        segments: list[SpeechSegment] = []
        while True:
            self._skip_whitespace(limit=end)
            pending = self._text[self._pending_start : end]
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

    def _cut_rest(self, limit: int | None = None) -> list[SpeechSegment]:
        """Cut all the pending prose, up to `limit` or the end of the speakable prose."""
        end = self._spoken_end if limit is None else limit
        segments: list[SpeechSegment] = []
        while True:
            self._skip_whitespace(limit=end)
            pending_chars = end - self._pending_start
            if pending_chars <= 0:
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

    def _skip_whitespace(self, limit: int) -> None:
        while self._pending_start < limit and self._text[self._pending_start].isspace():
            self._pending_start += 1
