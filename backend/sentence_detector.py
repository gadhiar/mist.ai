"""Sentence boundary detection for streaming LLM token output.

Accumulates tokens and emits complete sentences as soon as a boundary is
detected.  Handles abbreviations, decimals, ellipses, and list markers.
"""

import re

# Common abbreviations that should NOT trigger sentence splits
_ABBREVIATIONS = frozenset(
    {
        "dr",
        "mr",
        "mrs",
        "ms",
        "prof",
        "sr",
        "jr",
        "vs",
        "etc",
        "inc",
        "ltd",
        "dept",
        "est",
        "approx",
        "vol",
        "no",
        "fig",
        "eq",
        "st",
        "ave",
        "blvd",
        "govt",
        "gen",
        "sgt",
        "cpl",
        "pvt",
        "rev",
        "hon",
        "corp",
    }
)

# Regex: terminal punctuation (.!?) optionally followed by closing quote/bracket,
# then whitespace or end-of-string
_BOUNDARY_RE = re.compile(
    r'([.!?]["\')}\]]?)'  # terminal punct + optional close
    r"(\s+)",  # followed by whitespace
)


class SentenceBoundaryDetector:
    """Detects sentence boundaries in a streaming token sequence.

    Feed tokens as they arrive from the LLM. The detector returns
    a list of complete sentences (0 or more) after each feed call.
    Call flush() at end-of-stream to emit any remaining text.
    """

    # Set to 8 instead of 20 to avoid over-merging short conversational
    # sentences like "I see." which should be their own TTS generation
    # for natural pacing.
    MIN_SENTENCE_LENGTH = 8

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, token: str) -> list[str]:
        """Feed a token, return list of complete sentences (0 or more).

        Args:
            token: Next token from the LLM stream.

        Returns:
            List of complete sentences detected. May be empty.
        """
        self._buffer += token
        sentences: list[str] = []
        search_start = 0

        while True:
            boundary = self._find_boundary(search_start)
            if boundary is None:
                break

            sentence = self._buffer[:boundary].strip()
            remaining = self._buffer[boundary:].lstrip()

            if len(sentence) < self.MIN_SENTENCE_LENGTH:
                if sentences:
                    # Merge short sentence with previous
                    sentences[-1] += " " + sentence
                    self._buffer = remaining
                    search_start = 0
                else:
                    # Too short and nothing to merge with -- skip this
                    # boundary and look for the next one so the short
                    # fragment merges forward into a larger sentence.
                    # If no further boundary exists, the fragment stays
                    # in the buffer until more tokens arrive or flush().
                    search_start = boundary
                    continue
            else:
                sentences.append(sentence)
                self._buffer = remaining
                search_start = 0

        return sentences

    def flush(self) -> list[str]:
        """Flush remaining buffer as final sentence.

        Returns:
            List containing the remaining text, or empty if buffer is empty.
        """
        remaining = self._buffer.strip()
        self._buffer = ""
        if remaining:
            return [remaining]
        return []

    def _find_boundary(self, search_start: int = 0) -> int | None:
        """Find first sentence boundary position in buffer.

        Args:
            search_start: Position to start searching from.

        Returns:
            Character index after the boundary (start of next sentence),
            or None if no boundary found.
        """
        for match in _BOUNDARY_RE.finditer(self._buffer, search_start):
            pos = match.start()
            end = match.end()

            # Check: is this an abbreviation?
            if self._buffer[match.start(1)] == ".":
                if self._is_abbreviation(pos):
                    continue
                if self._is_decimal(pos):
                    continue
                if self._is_list_marker(pos):
                    continue
                if self._is_ellipsis(pos):
                    continue

            # Valid boundary -- return position after the punctuation
            return end

        return None

    def _is_abbreviation(self, dot_pos: int) -> bool:
        """Check if period at dot_pos is part of an abbreviation."""
        # Find the word before the dot
        start = dot_pos
        while start > 0 and self._buffer[start - 1].isalpha():
            start -= 1
        word = self._buffer[start:dot_pos].lower()
        return word in _ABBREVIATIONS

    def _is_decimal(self, dot_pos: int) -> bool:
        """Check if period at dot_pos is a decimal point between digits."""
        if dot_pos == 0 or dot_pos >= len(self._buffer) - 1:
            return False
        return self._buffer[dot_pos - 1].isdigit() and self._buffer[dot_pos + 1].isdigit()

    def _is_list_marker(self, dot_pos: int) -> bool:
        """Check if period is after a single digit list marker (e.g., '1.')."""
        if dot_pos < 1:
            return False
        if self._buffer[dot_pos - 1].isdigit() and (
            dot_pos < 2 or self._buffer[dot_pos - 2].isspace()
        ):
            return True
        return False

    def _is_ellipsis(self, dot_pos: int) -> bool:
        """Check if period is part of an ellipsis (...)."""
        if dot_pos >= 2 and self._buffer[dot_pos - 2 : dot_pos + 1] == "...":
            return True
        if dot_pos < len(self._buffer) - 2 and self._buffer[dot_pos : dot_pos + 3] == "...":
            return True
        return False
