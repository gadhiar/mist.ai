"""Tests for `stages.segments`: cutting turns into segments at VAD pauses, and the span helpers.

Every function here is pure arithmetic on spans and quality tracks, so the tests build the spans
by hand and check the boundaries of the segment gate of Interfaces: 3-12 s, 12-30 s lines in the
long set, overlap 0 ms, SNR and C50 floors, and windows over a segment.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.voice_data.bindings.protocols import QualityTrack, Span
from scripts.voice_data.stages import segments

LIMITS = {"min_s": 3.0, "max_s": 12.0, "long_min_s": 12.0, "long_max_s": 30.0}


def _cut(turn: Span, speech: list[Span]) -> list[segments.Cut]:
    return segments.cut_line(turn, speech, **LIMITS)


def test_a_line_is_trimmed_to_its_speech() -> None:
    (cut,) = _cut(Span(10.0, 20.0), [Span(9.0, 11.5), Span(12.0, 17.0)])
    assert (cut.start_s, cut.end_s, cut.set, cut.reasons) == (10.0, 17.0, "short", ())


def test_a_short_line_is_dropped_for_length() -> None:
    (cut,) = _cut(Span(0.0, 2.5), [Span(0.0, 2.5)])
    assert cut.set is None
    assert cut.reasons == ("length",)


def test_a_line_with_no_speech_gives_no_cut() -> None:
    assert _cut(Span(0.0, 5.0), [Span(6.0, 9.0)]) == []


@pytest.mark.parametrize(("seconds", "kind"), [(3.0, "short"), (12.0, "short"), (12.5, "long")])
def test_set_boundaries(seconds: float, kind: str) -> None:
    (cut,) = _cut(Span(0.0, seconds), [Span(0.0, seconds)])
    assert cut.set == kind


def test_a_line_of_12_to_30_s_stays_whole_in_the_long_set() -> None:
    speech = [Span(0.0, 8.0), Span(8.5, 20.0)]
    (cut,) = _cut(Span(0.0, 20.0), speech)
    assert (cut.start_s, cut.end_s, cut.set) == (0.0, 20.0, "long")


def test_a_line_over_30_s_is_cut_at_pauses_into_short_segments() -> None:
    speech = [Span(0.0, 5.0), Span(5.5, 10.0), Span(10.5, 16.0), Span(16.4, 22.0), Span(22.5, 34.0)]
    cuts = _cut(Span(0.0, 34.0), speech)
    assert [(c.start_s, c.end_s, c.set) for c in cuts] == [
        (0.0, 10.0, "short"),
        (10.5, 22.0, "short"),
        (22.5, 34.0, "short"),
    ]
    for cut in cuts:
        assert any(cut.start_s == s.start_s for s in speech)
        assert any(cut.end_s == s.end_s for s in speech)


def test_a_piece_of_a_long_line_without_a_pause_over_12_s_goes_to_the_long_set() -> None:
    speech = [Span(0.0, 4.0), Span(4.5, 24.0), Span(24.5, 33.0)]
    cuts = _cut(Span(0.0, 33.0), speech)
    assert [(c.start_s, c.end_s, c.set) for c in cuts] == [
        (0.0, 4.0, "short"),
        (4.5, 24.0, "long"),
        (24.5, 33.0, "short"),
    ]


def test_an_unbroken_line_over_30_s_is_dropped_for_length() -> None:
    (cut,) = _cut(Span(0.0, 31.0), [Span(0.0, 31.0)])
    assert cut.reasons == ("length",)


def test_a_short_remainder_of_a_long_line_is_dropped_for_length() -> None:
    speech = [Span(0.0, 11.0), Span(11.5, 22.0), Span(22.5, 33.0), Span(33.5, 35.0)]
    cuts = _cut(Span(0.0, 35.0), speech)
    assert cuts[-1].reasons == ("length",)
    assert [c.set for c in cuts[:-1]] == ["short", "short", "short"]


def test_overlap_ms_counts_only_true_intersection() -> None:
    overlaps = [Span(5.0, 7.0), Span(20.0, 22.0)]
    assert segments.overlap_ms(Span(0.0, 5.0), overlaps) == 0
    assert segments.overlap_ms(Span(0.0, 5.5), overlaps) == 500
    assert segments.overlap_ms(Span(4.0, 23.0), overlaps) == 4000


def test_intersects() -> None:
    assert not segments.intersects(Span(0.0, 5.0), [Span(5.0, 6.0)])
    assert segments.intersects(Span(0.0, 5.01), [Span(5.0, 6.0)])


def test_sliding_windows_cover_the_span_and_end_on_it() -> None:
    wins = segments.windows(Span(0.0, 4.0), 1.5, 0.75)
    assert wins[0] == Span(0.0, 1.5)
    assert wins[-1] == Span(2.5, 4.0)
    starts = [w.start_s for w in wins]
    assert starts == sorted(starts)
    assert all(np.isclose(w.end_s - w.start_s, 1.5) for w in wins)


def test_any_gap_of_a_window_and_a_hop_holds_a_whole_window() -> None:
    wins = segments.windows(Span(0.0, 12.0), 1.5, 0.75)
    for start in np.arange(0.0, 9.75, 0.05):
        inside = [w for w in wins if w.start_s >= start - 1e-9 and w.end_s <= start + 2.25 + 1e-9]
        assert inside, start


def test_a_span_shorter_than_a_window_is_one_window() -> None:
    assert segments.windows(Span(1.0, 2.0), 1.5, 0.75) == [Span(1.0, 2.0)]


def test_tiles_do_not_overlap_and_drop_a_short_tail() -> None:
    assert segments.tiles(Span(0.0, 4.0), 1.5) == [Span(0.0, 1.5), Span(1.5, 3.0)]
    assert segments.tiles(Span(0.0, 1.0), 1.5) == []


def test_subtract_leaves_the_uncovered_parts() -> None:
    spans = [Span(0.0, 10.0), Span(20.0, 30.0)]
    minus = [Span(2.0, 4.0), Span(8.0, 22.0), Span(29.0, 40.0)]
    assert segments.subtract(spans, minus) == [Span(0.0, 2.0), Span(4.0, 8.0), Span(22.0, 29.0)]


def test_speech_within_clips_to_the_turn() -> None:
    speech = [Span(0.0, 3.0), Span(4.0, 6.0), Span(9.0, 12.0)]
    assert segments.speech_within(Span(2.0, 10.0), speech) == [
        Span(2.0, 3.0),
        Span(4.0, 6.0),
        Span(9.0, 10.0),
    ]


def test_span_quality_is_the_median_of_the_frames_inside() -> None:
    snr = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    track = QualityTrack(hop_s=0.5, snr_db=snr, c50_db=snr + 1.0)
    assert segments.span_quality(track, Span(0.5, 2.0)) == (30.0, 31.0)


def test_span_quality_of_a_span_between_frames_takes_the_nearest() -> None:
    track = QualityTrack(hop_s=1.0, snr_db=np.array([5.0, 7.0]), c50_db=np.array([1.0, 2.0]))
    assert segments.span_quality(track, Span(0.2, 0.4)) == (5.0, 1.0)


@pytest.mark.parametrize(
    ("snr", "c50", "window_ok", "reasons"),
    [
        (15.0, 20.0, True, []),
        (14.9, 20.0, True, ["snr"]),
        (15.0, 19.9, True, ["c50"]),
        (15.0, 20.0, False, ["window"]),
        (1.0, 1.0, False, ["snr", "c50", "window"]),
    ],
)
def test_quality_reasons(snr: float, c50: float, window_ok: bool, reasons: list[str]) -> None:
    assert segments.quality_reasons(snr, c50, window_ok, snr_min=15.0, c50_min=20.0) == reasons
