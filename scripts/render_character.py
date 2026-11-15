"""Render one WAV through her character stage, as the realtime downlink would play it.

    python -m scripts.render_character --in <wav> --out <wav> [--chunk-ms 80] [--double-gain G]

Raj's blind checks compare the streaming stage with the ear test's offline render, so this
streams a 24 kHz mono PCM16 WAV through `CharacterStage(CharacterConfig(double_gain=G))` in
fixed chunks, as one segment, and appends `end_segment()`'s tail, exactly as one reply's single
segment would reach the pacer.

It runs with no GPU and no network, in a throwaway container of the backend image with the
checkout mounted read-only and the audio directory mounted writable. Clips are private and the
repository is public, so it refuses either path inside the checkout and never overwrites a
render, and it prints only sample counts and the clip count, never a path or a file name. Every
refusal exits 2.
"""

import argparse
import sys
import wave
from collections.abc import Sequence
from pathlib import Path

from backend.realtime.budget import CharacterConfig
from backend.realtime.character import CharacterStage

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 24000


class _RefusedError(Exception):
    """A refusal; its message names the argument, never the path."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in", dest="source", required=True, type=Path)
    parser.add_argument("--out", dest="target", required=True, type=Path)
    parser.add_argument("--chunk-ms", type=int, default=80)
    parser.add_argument("--double-gain", type=float, default=CharacterConfig().double_gain)
    return parser


def _outside_checkout(path: Path, flag: str) -> Path:
    resolved = path.resolve()
    if resolved == REPO_ROOT or REPO_ROOT in resolved.parents:
        raise _RefusedError(f"{flag} lies inside the checkout; keep audio outside it")
    return resolved


def _read_pcm16(path: Path) -> bytes:
    try:
        with wave.open(str(path), "rb") as handle:
            shape = (handle.getnchannels(), handle.getsampwidth(), handle.getframerate())
            if shape != (1, 2, SAMPLE_RATE) or handle.getcomptype() != "NONE":
                raise _RefusedError(f"--in is not {SAMPLE_RATE} Hz mono PCM16")
            return handle.readframes(handle.getnframes())
    except (wave.Error, EOFError) as exc:
        raise _RefusedError("--in is not a readable WAV file") from exc
    except OSError as exc:
        raise _RefusedError(f"--in cannot be read ({exc.strerror})") from exc


def _render(pcm: bytes, config: CharacterConfig, chunk_ms: int) -> tuple[bytes, int]:
    stage = CharacterStage(config, sample_rate=SAMPLE_RATE)
    chunk = 2 * chunk_ms * SAMPLE_RATE // 1000
    out = [stage.process(pcm[start : start + chunk]) for start in range(0, len(pcm), chunk)]
    out.append(stage.end_segment())
    return b"".join(out), stage.clipped_samples


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


def main(argv: Sequence[str] | None = None) -> int:
    """Render `--in` to `--out`; 0 on success, 2 on any refusal."""
    args = _parser().parse_args(argv)
    try:
        source = _outside_checkout(args.source, "--in")
        target = _outside_checkout(args.target, "--out")
        if target.exists():
            raise _RefusedError("--out already exists; renders are never overwritten")
        if args.chunk_ms <= 0:
            raise _RefusedError("--chunk-ms must be positive")
        try:
            config = CharacterConfig(double_gain=args.double_gain)
        except ValueError as exc:
            raise _RefusedError(f"--double-gain: {exc}") from exc
        pcm = _read_pcm16(source)
        rendered, clipped = _render(pcm, config, args.chunk_ms)
        _write_pcm16(target, rendered)
    except _RefusedError as exc:
        print(f"render_character: {exc}", file=sys.stderr)
        return 2
    print(f"input samples: {len(pcm) // 2}")
    print(f"output samples: {len(rendered) // 2}")
    print(f"clipped samples: {clipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
