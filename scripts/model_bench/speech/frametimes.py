"""Frame-time figures for runs 8 and 9: average, 1% low and 99th-percentile frame time.

The source is a PresentMon CSV where PresentMon can capture the game; where it cannot, the game's
own benchmark figures or a frame rate Raj types stand in, and the row says which.

- Frame time is `MsBetweenPresents` (PresentMon 1.x and the 2.x `--v1_metrics` layout), else
  `FrameTime` (the 2.x default layout). `MsBetweenDisplayChange` (or 2.x `DisplayedTime`) is kept
  beside it: a frame with 0 there was never shown.
- Average frame rate is frames over total frame time, not the mean of per-frame rates, so a long
  frame counts for the time it took.
- The 1% low is 1000 over the mean of the slowest 1% of frame times (at least one frame, the
  count rounded up), the usual definition in game benchmarks.
- The 99th-percentile frame time is the nearest-rank percentile, as everywhere else in the bench.

stdlib only: the game session imports this on the Windows host.
"""

from __future__ import annotations

import csv
import io
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from . import core

FRAME_TIME_COLUMNS = ("MsBetweenPresents", "FrameTime")
DISPLAY_COLUMNS = ("MsBetweenDisplayChange", "DisplayedTime")
SOURCES = ("presentmon", "game_benchmark", "typed")


class FrameTimeError(core.SpeechBenchError):
    """A capture or a typed figure that cannot give a frame rate."""


@dataclass(frozen=True, slots=True)
class FrameSummary:
    """One pass's frame figures and where they came from."""

    source: str
    frames: int | None
    avg_fps: float | None
    low_1pct_fps: float | None
    p99_frame_ms: float | None
    duration_s: float | None = None
    displayed_frames: int | None = None

    def as_row(self) -> dict[str, object]:
        """The figures as a JSON-ready row."""
        return asdict(self)


def _number(value: str | None) -> float | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def frame_times(text: str) -> tuple[list[float], int | None]:
    """The positive frame times in a PresentMon CSV, and how many frames were displayed."""
    reader = csv.DictReader(io.StringIO(text))
    fields = reader.fieldnames or []
    time_col = next((c for c in FRAME_TIME_COLUMNS if c in fields), None)
    if time_col is None:
        raise FrameTimeError(
            f"no frame-time column ({' or '.join(FRAME_TIME_COLUMNS)}) in {fields}"
        )
    display_col = next((c for c in DISPLAY_COLUMNS if c in fields), None)
    times: list[float] = []
    displayed = 0
    for row in reader:
        value = _number(row.get(time_col))
        if value is None or value <= 0:
            continue
        times.append(value)
        shown = _number(row.get(display_col)) if display_col else None
        if shown is not None and shown > 0:
            displayed += 1
    return times, displayed if display_col else None


def summarize_frame_times(
    times: list[float], *, displayed: int | None = None, source: str = "presentmon"
) -> FrameSummary:
    """Average, 1% low and 99th-percentile frame time; None for each when there are no frames."""
    if not times:
        return FrameSummary(source, 0, None, None, None, 0.0, displayed)
    total_ms = sum(times)
    slowest = sorted(times, reverse=True)[: max(1, math.ceil(len(times) / 100))]
    return FrameSummary(
        source=source,
        frames=len(times),
        avg_fps=1000.0 * len(times) / total_ms,
        low_1pct_fps=1000.0 / (sum(slowest) / len(slowest)),
        p99_frame_ms=core.nearest_rank(times, 99),
        duration_s=total_ms / 1000.0,
        displayed_frames=displayed,
    )


def summarize_presentmon(path: Path) -> FrameSummary:
    """The figures of one PresentMon capture file."""
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as exc:
        raise FrameTimeError(f"cannot read {path}: {exc}") from exc
    times, displayed = frame_times(text)
    return summarize_frame_times(times, displayed=displayed)


def from_game_benchmark(
    avg_fps: float, low_1pct_fps: float | None = None, p99_frame_ms: float | None = None
) -> FrameSummary:
    """The game's own benchmark figures, where PresentMon cannot capture it."""
    return FrameSummary("game_benchmark", None, avg_fps, low_1pct_fps, p99_frame_ms)


def from_typed_fps(fps: float) -> FrameSummary:
    """A frame rate Raj reads off the game's counter: an average only."""
    return FrameSummary("typed", None, fps, None, None)


def parse_positive(text: str, what: str, *, optional: bool = False) -> float | None:
    """A typed figure: a positive number, or blank where `optional`."""
    if optional and not text.strip():
        return None
    value = _number(text)
    if value is None or value <= 0:
        raise FrameTimeError(f"{what}: {text!r} is not a positive number")
    return value
