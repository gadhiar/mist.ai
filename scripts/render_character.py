"""Render one WAV through her character stage, as the realtime downlink would play it.

    python -m scripts.render_character --in <wav> --out <wav> [--chunk-ms 80] [--double-gain G]
        [--no-limiter] [--compare-to <earlier wav>]

Raj's blind checks compare the streaming stage with the ear test's offline render, so this
streams a 24 kHz mono PCM16 WAV through `CharacterStage(CharacterConfig(double_gain=G))` in
fixed chunks, as one segment, and appends `end_segment()`'s tail, exactly as one reply's single
segment would reach the pacer. The stage ends with its peak limiter unless `--no-limiter` is
given, which renders with `limiter=None` and so reproduces the stage as it was before the
limiter, hard clipping included.

`--compare-to` compares the render, sample by sample, with an earlier render of the same input,
so a change to the stage can be checked on private clips without anyone listening. It counts the
earlier render's samples at or above the default limiter's knee (29204 steps: the knee, 29204.51,
rounded down, which covers every sample that passed it before rounding) and at full scale, the
samples that differ, and those that differ outside the limiter's reach of an earlier sample at or
above the knee (the reach is 0 with `--no-limiter`). A region is a maximal run of samples within
that reach; each region's change is `10 log10(sum((after - before)^2) / sum(before^2))`, and the
worst over the regions that differ is printed.

It runs with no GPU and no network, in a throwaway container of the backend image with the
checkout mounted read-only and the audio directory mounted writable. Clips are private and the
repository is public, so it refuses any path inside the checkout and never overwrites a render,
and it prints only the sample counts, the clip count, the limited count and, with
`--compare-to`, counts and dB, never a path or a file name. Every refusal exits 2, before
`--out` is written.
"""

import argparse
import math
import sys
import wave
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from backend.realtime.budget import CharacterConfig, LimiterConfig
from backend.realtime.character import CharacterStage, limiter_reach_samples

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 24000
# The default limiter's knee in PCM16 steps, rounded down: rounding moves a value by at most half
# a step, so every sample whose unrounded value passed the knee (29204.51) is at least this.
KNEE_STEPS = math.floor(32768.0 * 10.0 ** (LimiterConfig().knee_dbfs / 20.0))
FULL_SCALE_STEPS = 32767


class _RefusedError(Exception):
    """A refusal; its message names the argument, never the path."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in", dest="source", required=True, type=Path)
    parser.add_argument("--out", dest="target", required=True, type=Path)
    parser.add_argument("--chunk-ms", type=int, default=80)
    parser.add_argument("--double-gain", type=float, default=CharacterConfig().double_gain)
    parser.add_argument("--no-limiter", action="store_true")
    parser.add_argument("--compare-to", dest="earlier", type=Path)
    return parser


def _outside_checkout(path: Path, flag: str) -> Path:
    resolved = path.resolve()
    if resolved == REPO_ROOT or REPO_ROOT in resolved.parents:
        raise _RefusedError(f"{flag} lies inside the checkout; keep audio outside it")
    return resolved


def _read_pcm16(path: Path, flag: str) -> bytes:
    try:
        with wave.open(str(path), "rb") as handle:
            shape = (handle.getnchannels(), handle.getsampwidth(), handle.getframerate())
            if shape != (1, 2, SAMPLE_RATE) or handle.getcomptype() != "NONE":
                raise _RefusedError(f"{flag} is not {SAMPLE_RATE} Hz mono PCM16")
            return handle.readframes(handle.getnframes())
    except (wave.Error, EOFError) as exc:
        raise _RefusedError(f"{flag} is not a readable WAV file") from exc
    except OSError as exc:
        raise _RefusedError(f"{flag} cannot be read ({exc.strerror})") from exc


def _render(pcm: bytes, config: CharacterConfig, chunk_ms: int) -> tuple[bytes, CharacterStage]:
    stage = CharacterStage(config, sample_rate=SAMPLE_RATE)
    chunk = 2 * chunk_ms * SAMPLE_RATE // 1000
    out = [stage.process(pcm[start : start + chunk]) for start in range(0, len(pcm), chunk)]
    out.append(stage.end_segment())
    return b"".join(out), stage


def _write_pcm16(path: Path, pcm: bytes) -> None:
    try:
        with path.open("xb") as raw, wave.open(raw, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(SAMPLE_RATE)
            handle.writeframes(pcm)
    except FileExistsError as exc:
        raise _RefusedError("--out already exists; renders are never overwritten") from exc
    except OSError as exc:
        raise _RefusedError(f"--out cannot be written ({exc.strerror})") from exc


def _within_reach(loud: np.ndarray, reach: int) -> np.ndarray:
    """True at each sample that a `loud` sample at or before it, by fewer than `reach`, covers."""
    if reach == 0:
        return np.zeros(len(loud), dtype=bool)
    counts = np.concatenate(([0], np.cumsum(loud, dtype=np.int64)))
    ends = np.arange(1, len(loud) + 1)
    return counts[ends] - counts[np.maximum(ends - reach, 0)] > 0


def _regions(within: np.ndarray) -> list[tuple[int, int]]:
    """The maximal runs of `within`, as half-open (start, stop) pairs."""
    edges = np.diff(np.concatenate(([0], within.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))


def _comparison(rendered: bytes, earlier: bytes, reach: int) -> list[str]:
    """The comparison's printed lines: counts and dB only."""
    after = np.frombuffer(rendered, dtype="<i2").astype(np.int64)
    before = np.frombuffer(earlier, dtype="<i2").astype(np.int64)
    magnitude = np.abs(before)
    loud = magnitude >= KNEE_STEPS
    within = _within_reach(loud, reach)
    delta = after - before
    differs = delta != 0
    changes = []
    regions = _regions(within)
    for start, stop in regions:
        if np.any(differs[start:stop]):
            change = np.sum(np.square(delta[start:stop])) / np.sum(np.square(before[start:stop]))
            changes.append(10.0 * math.log10(change))
    worst = f"{max(changes):.1f} dB" if changes else "none"
    full_scale = int(np.count_nonzero(magnitude >= FULL_SCALE_STEPS))
    outside = int(np.count_nonzero(differs & ~within))
    return [
        f"compared samples: {len(before)}",
        f"earlier samples at or above the knee: {int(np.count_nonzero(loud))}",
        f"earlier samples at full scale: {full_scale}",
        f"differing samples: {int(np.count_nonzero(differs))}",
        f"differing samples outside the limiter's reach: {outside}",
        f"largest difference: {int(np.max(np.abs(delta), initial=0))} LSB",
        f"limited regions: {len(regions)}",
        f"worst region change: {worst}",
    ]


def main(argv: Sequence[str] | None = None) -> int:
    """Render `--in` to `--out`; 0 on success, 2 on any refusal."""
    args = _parser().parse_args(argv)
    try:
        source = _outside_checkout(args.source, "--in")
        target = _outside_checkout(args.target, "--out")
        earlier_path = (
            _outside_checkout(args.earlier, "--compare-to") if args.earlier is not None else None
        )
        if target.exists():
            raise _RefusedError("--out already exists; renders are never overwritten")
        if args.chunk_ms <= 0:
            raise _RefusedError("--chunk-ms must be positive")
        try:
            limiter = None if args.no_limiter else LimiterConfig()
            config = CharacterConfig(double_gain=args.double_gain, limiter=limiter)
        except ValueError as exc:
            raise _RefusedError(f"--double-gain: {exc}") from exc
        pcm = _read_pcm16(source, "--in")
        earlier = _read_pcm16(earlier_path, "--compare-to") if earlier_path is not None else None
        rendered, stage = _render(pcm, config, args.chunk_ms)
        if earlier is not None and len(earlier) != len(rendered):
            raise _RefusedError("--compare-to holds another number of samples than this render")
        _write_pcm16(target, rendered)
    except _RefusedError as exc:
        print(f"render_character: {exc}", file=sys.stderr)
        return 2
    print(f"input samples: {len(pcm) // 2}")
    print(f"output samples: {len(rendered) // 2}")
    print(f"clipped samples: {stage.clipped_samples}")
    print(f"limited samples: {stage.limited_samples}")
    if earlier is not None:
        reach = 0 if limiter is None else limiter_reach_samples(limiter, sample_rate=SAMPLE_RATE)
        for line in _comparison(rendered, earlier, reach):
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
