"""Segment cutting and the span arithmetic of the segment gate (Design 4, Interfaces: Gates).

A retrieved turn becomes one or more segments: trimmed to its VAD speech, kept whole when it is
3-12 s (the short set) or 12-30 s (the long-line set), and cut at VAD pauses into short segments
when it runs past 30 s. Everything here is arithmetic on spans and per-frame quality tracks, so
the extraction stage and its tests share one definition of each boundary.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..bindings.protocols import QualityTrack, Span

SHORT = "short"
LONG = "long"


@dataclass(frozen=True, slots=True)
class Cut:
    """A segment cut from a turn: its span, its set, or (`set` None) why it was dropped."""

    start_s: float
    end_s: float
    set: str | None
    reasons: tuple[str, ...] = ()

    @property
    def span(self) -> Span:
        """The cut as a span."""
        return Span(self.start_s, self.end_s)


def speech_within(turn: Span, speech: Sequence[Span]) -> list[Span]:
    """Return the parts of the VAD speech spans that fall inside `turn`, in time order."""
    inside = []
    for span in speech:
        start, end = max(turn.start_s, span.start_s), min(turn.end_s, span.end_s)
        if end > start:
            inside.append(Span(start, end))
    return inside


def _classify(start: float, end: float, limits: tuple[float, float, float, float]) -> Cut:
    min_s, max_s, long_min_s, long_max_s = limits
    length = end - start
    if min_s <= length <= max_s:
        return Cut(start, end, SHORT)
    if long_min_s <= length <= long_max_s:
        return Cut(start, end, LONG)
    return Cut(start, end, None, ("length",))


def cut_line(
    turn: Span,
    speech: Sequence[Span],
    *,
    min_s: float,
    max_s: float,
    long_min_s: float,
    long_max_s: float,
) -> list[Cut]:
    """Cut one turn into segments at VAD pauses.

    The line runs from the turn's first speech to its last. A line of `min_s`-`max_s` is one
    short segment and a line of `long_min_s`-`long_max_s` one long segment, kept whole so the
    long-line set holds real long lines. A longer line is cut at its pauses: speech pieces join
    a segment while it stays within `max_s`, and a piece that alone runs past `max_s` stands as
    its own segment (long when it is within `long_max_s`). A piece of the wrong length is
    returned with the reason `length`, so the drop is recorded.

    Args:
        turn: The turn's span.
        speech: The VAD speech spans of the channel the turn is cut from.
        min_s: Shortest short segment.
        max_s: Longest short segment.
        long_min_s: Shortest long segment; the gates set it equal to `max_s`.
        long_max_s: Longest long segment.

    Returns:
        The cuts in time order; none when the turn holds no speech.
    """
    pieces = speech_within(turn, speech)
    if not pieces:
        return []
    limits = (min_s, max_s, long_min_s, long_max_s)
    start, end = pieces[0].start_s, pieces[-1].end_s
    if end - start <= long_max_s:
        return [_classify(start, end, limits)]
    cuts: list[Cut] = []
    seg_start, seg_end = pieces[0].start_s, pieces[0].end_s
    for piece in pieces[1:]:
        if piece.end_s - seg_start <= max_s:
            seg_end = piece.end_s
            continue
        cuts.append(_classify(seg_start, seg_end, limits))
        seg_start, seg_end = piece.start_s, piece.end_s
    cuts.append(_classify(seg_start, seg_end, limits))
    return cuts


def intersects(span: Span, others: Sequence[Span]) -> bool:
    """Return whether `span` shares any time with one of `others`; touching ends do not count."""
    return any(max(span.start_s, o.start_s) < min(span.end_s, o.end_s) for o in others)


def overlap_ms(span: Span, overlaps: Sequence[Span]) -> int:
    """Return the milliseconds of `span` that intersect the overlapped-speech spans."""
    total = 0.0
    for other in overlaps:
        total += max(0.0, min(span.end_s, other.end_s) - max(span.start_s, other.start_s))
    return int(round(total * 1000.0))


def windows(span: Span, window_s: float, hop_s: float) -> list[Span]:
    """Return sliding windows of `window_s` over `span`, every `hop_s`, the last ending on it.

    With a hop of half a window, any stretch of a window and a hop holds a whole window, so a
    short other-speaker stretch inside a segment cannot fall between two windows. A span
    shorter than a window is one window.
    """
    length = span.end_s - span.start_s
    if length <= window_s:
        return [span]
    count = int(math.floor((length - window_s) / hop_s + 1e-9)) + 1
    found = [
        Span(span.start_s + i * hop_s, span.start_s + i * hop_s + window_s) for i in range(count)
    ]
    if found[-1].end_s < span.end_s - 1e-9:
        found.append(Span(span.end_s - window_s, span.end_s))
    return found


def tiles(span: Span, window_s: float) -> list[Span]:
    """Return non-overlapping windows of `window_s` from the start of `span`; a short tail is left.

    The window distribution is built from tiles, so no stretch of speech counts twice in it.
    """
    count = int(math.floor((span.end_s - span.start_s) / window_s + 1e-9))
    return [
        Span(span.start_s + i * window_s, span.start_s + (i + 1) * window_s) for i in range(count)
    ]


def subtract(spans: Sequence[Span], minus: Sequence[Span]) -> list[Span]:
    """Return the parts of `spans` that no span of `minus` covers, in time order."""
    result: list[Span] = []
    cuts = sorted(minus, key=lambda s: s.start_s)
    for span in spans:
        start = span.start_s
        for cut in cuts:
            if cut.end_s <= start or cut.start_s >= span.end_s:
                continue
            if cut.start_s > start:
                result.append(Span(start, cut.start_s))
            start = max(start, cut.end_s)
        if start < span.end_s:
            result.append(Span(start, span.end_s))
    return result


def span_quality(track: QualityTrack, span: Span) -> tuple[float, float]:
    """Return the median SNR and C50 in dB of the frames that start inside `span`.

    A span between two frame starts takes the frame its midpoint falls in.
    """
    count = len(track.snr_db)
    first = int(math.ceil(span.start_s / track.hop_s - 1e-9))
    last = int(math.ceil(span.end_s / track.hop_s - 1e-9))
    first, last = max(0, first), min(count, last)
    if last <= first:
        index = min(count - 1, max(0, int((span.start_s + span.end_s) / 2.0 / track.hop_s)))
        return float(track.snr_db[index]), float(track.c50_db[index])
    return (
        float(np.median(track.snr_db[first:last])),
        float(np.median(track.c50_db[first:last])),
    )


def quality_reasons(
    snr_db: float, c50_db: float, window_ok: bool, *, snr_min: float, c50_min: float
) -> list[str]:
    """Return the segment gates a cut fails, in the order `snr`, `c50`, `window`."""
    reasons = []
    if not snr_db >= snr_min:
        reasons.append("snr")
    if not c50_db >= c50_min:
        reasons.append("c50")
    if not window_ok:
        reasons.append("window")
    return reasons
