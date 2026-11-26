r"""Render the ear test's two voices, line by line, as she sounds live, for the blind A/B pack.

    python scripts/voice_bench/voice_ab.py --baseline <id> --trained <id> \
        --lines scripts/model_bench/speech/lines.json --out <private root>/bench/voice-ab/<run>

Each identity loads through the realtime adapter, `PocketCloneSynthesizer.load`, with the
placement the runtime would read from this environment (`MIST_VOICE_IDENTITY_ROOT`,
`MIST_VOICE_MODELS_DIR`, `MIST_VOICE_CACHE_DIR`, their defaults otherwise), so a trained
identity speaks its trained weights exactly as it would live. Every line is a neutral segment,
streamed chunk by chunk through her character stage (`CharacterStage`, on, double gain 0.35,
the limiter on) with the segment's tail, as the downlink plays one segment. One identity is
loaded at a time and released before the next. The lines are the file's `lines` then its
`acknowledgements`: the ear test's 20 lines and 6 acknowledgements.

`first_chunk_ms` is the time from the synthesis call to its first chunk, per line and arm: the
adapter's first chunk the bench gates judge. Each arm's integrated loudness is measured on the
stage output before any matching, which is what the loudness gate (-20 LUFS +/- 1.5 LU) reads;
then both arms of a line are brought to -20 LUFS with the backend's own `target_gain_db`, so
the ear test compares voices, not levels. The gain and the samples it pushed past full scale are
recorded beside it.

The clips are FLAC, PCM16, mono, 24 kHz and untagged, the format `pack voice-ab` copies byte
for byte, as the teacher takes are. `voice_ab.json` in the run folder lists, per line, its id,
its public text and per arm (`baseline`, `trained`) the root-relative clip, `first_chunk_ms`,
`loudness_lufs`, `gain_db`, `clipped_samples` and the adapter's `model_id`; the identity ids are
recorded once, at the top.

It runs on the CPU in the Pocket bench image, with the repository mounted and the private root
writable. The run folder must be `bench/voice-ab/<run>` of a marked private root, outside the
checkout, and must not exist: nothing is overwritten, and nothing is written until every line of
both arms has rendered. The summary on stdout is counts, milliseconds and a loudness range per
arm, then the run folder, and never a line id or a line's text. A refusal or a failure goes to
stderr as its error words it, and an engine error may name an identity id given on the command
line; only the pages `pack voice-ab` builds must name no identity. A refusal exits 2 before
anything loads; an engine failure exits 1.
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import math
import os
import re
import statistics
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    # Run as a file, the repository root is not on the path, and `backend` would not import.
    sys.path.insert(0, str(REPO_ROOT))

from backend.realtime.budget import CharacterConfig, PlacementConfig, load_voice_settings
from backend.realtime.character import CharacterStage
from backend.realtime.engines.audio import (
    OUTPUT_RATE,
    TARGET_LUFS,
    integrated_loudness,
    target_gain_db,
)
from backend.realtime.errors import SpeechEngineError
from backend.realtime.speech import CancelToken, SynthChunk, TextSegment
from scripts.voice_data.errors import RootError
from scripts.voice_data.root import Root, open_root

MANIFEST = "voice_ab.json"
CLIPS_DIR = "clips"
BENCH_DIR = ("bench", "voice-ab")
ARMS = ("baseline", "trained")
# Her character stage as she sounds live: on, each detuned copy at 0.35, the limiter on.
CHARACTER = CharacterConfig(double_gain=0.35)
_ID = re.compile(r"[a-z0-9][a-z0-9-]*")
_RUN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
PCM16_FULL_SCALE = 32768.0


class Synthesizer(Protocol):
    """What the renderer needs of a loaded adapter."""

    sample_rate: int

    @property
    def model_id(self) -> str:
        """The weights it speaks with, recorded per arm."""
        ...

    def synthesize(self, segment: TextSegment, token: CancelToken) -> Iterator[SynthChunk]:
        """The segment's audio, chunk by chunk."""
        ...

    def release(self) -> None:
        """Drop the model, before the next identity loads."""
        ...


Loader = Callable[[PlacementConfig, str], Synthesizer]
Clock = Callable[[], float]


class _RefusedError(Exception):
    """A refusal before anything loads; its message names the argument, never a path."""


@dataclass(frozen=True, slots=True)
class Line:
    """One public line of the ear test."""

    line_id: str
    text: str


@dataclass(frozen=True, slots=True)
class Take:
    """One arm's render of one line, after the stage and before any gain."""

    samples: np.ndarray
    first_chunk_ms: float


def load_pocket(placement: PlacementConfig, profile: str) -> Synthesizer:
    """The realtime adapter's own load, on the CPU, as the runtime's factory calls it."""
    from backend.realtime.engines.pocket_tts import PocketCloneSynthesizer, import_pocket

    return PocketCloneSynthesizer.load(placement, profile, library_loader=import_pocket)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", required=True, help="the baseline identity id (stock)")
    parser.add_argument("--trained", required=True, help="the trained identity id")
    parser.add_argument("--lines", required=True, type=Path, help="the public lines file")
    parser.add_argument(
        "--out", required=True, type=Path, help="<private root>/bench/voice-ab/<run>, new"
    )
    return parser


def _check_ids(baseline: str, trained: str) -> None:
    for flag, value in (("--baseline", baseline), ("--trained", trained)):
        if not _ID.fullmatch(value):
            raise _RefusedError(f"{flag} is not an identity id ([a-z0-9][a-z0-9-]*)")
    if baseline == trained:
        raise _RefusedError("--baseline and --trained must name two different identities")


def _open_run(out: Path) -> tuple[Root, str]:
    """The private root holding `out` and the run folder's root-relative path."""
    resolved = out.resolve()
    if resolved == REPO_ROOT or REPO_ROOT in resolved.parents:
        raise _RefusedError("--out lies inside the checkout; bench runs belong in the private root")
    if (
        len(resolved.parents) < 3
        or tuple(p.name for p in (resolved.parents[1], resolved.parent)) != BENCH_DIR
    ):
        raise _RefusedError("--out must be <private root>/bench/voice-ab/<run>")
    if not _RUN.fullmatch(resolved.name):
        raise _RefusedError("--out: a run name is letters, digits, '.', '_' and '-'")
    try:
        root = open_root(str(resolved.parents[2]))
    except RootError as exc:
        raise _RefusedError(f"--out: {exc}") from exc
    if resolved.exists():
        raise _RefusedError("--out already exists; a bench run is never overwritten")
    return root, root.rel(resolved.parent) + "/" + resolved.name


def _read_lines(path: Path) -> list[Line]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _RefusedError("--lines is missing or not JSON") from exc
    rows = doc.get("lines") if isinstance(doc, dict) else None
    acks = doc.get("acknowledgements", []) if isinstance(doc, dict) else None
    if not isinstance(rows, list) or not isinstance(acks, list) or not rows:
        raise _RefusedError("--lines holds no lines")
    rows = rows + acks
    lines = []
    for row in rows:
        if not (
            isinstance(row, dict)
            and isinstance(row.get("id"), str)
            and _RUN.fullmatch(row["id"])
            and isinstance(row.get("text"), str)
            and row["text"].strip()
        ):
            raise _RefusedError("--lines: a line lacks an id or a text")
        lines.append(Line(row["id"], row["text"]))
    if len({line.line_id for line in lines}) != len(lines):
        raise _RefusedError("--lines: two lines share an id")
    return lines


def _placement() -> tuple[PlacementConfig, str]:
    try:
        settings = load_voice_settings(os.environ)
    except ValueError as exc:
        raise _RefusedError(f"the voice settings are invalid: {exc}") from exc
    return settings.placement, settings.profile


def _render_line(synth: Synthesizer, line: Line, clock: Clock) -> Take:
    """One neutral segment through the adapter and her stage, chunk by chunk, with its tail."""
    stage = CharacterStage(CHARACTER, sample_rate=OUTPUT_RATE)
    segment = TextSegment(line.text, 0, len(line.text))
    started = clock()
    first: float | None = None
    staged = []
    for chunk in synth.synthesize(segment, CancelToken()):
        if first is None:
            first = clock()
        staged.append(stage.process(chunk.pcm16))
    if first is None:
        raise SpeechEngineError("the adapter returned no audio for a line")
    staged.append(stage.end_segment())
    pcm = np.frombuffer(b"".join(staged), dtype="<i2")
    return Take(pcm.astype(np.float64) / PCM16_FULL_SCALE, round((first - started) * 1000, 1))


def _render_arm(
    load: Loader, placement: PlacementConfig, profile: str, lines: Sequence[Line], clock: Clock
) -> tuple[str, list[Take]]:
    synth = load(placement, profile)
    try:
        if int(synth.sample_rate) != OUTPUT_RATE:
            raise SpeechEngineError(f"the adapter speaks at {synth.sample_rate} Hz, not 24000")
        return synth.model_id, [_render_line(synth, line, clock) for line in lines]
    finally:
        synth.release()


def _flac(samples: np.ndarray) -> bytes:
    import soundfile

    buffer = io.BytesIO()
    soundfile.write(
        buffer, samples.astype(np.float32), OUTPUT_RATE, format="FLAC", subtype="PCM_16"
    )
    return buffer.getvalue()


def _matched(take: Take) -> tuple[np.ndarray, dict[str, float | int]]:
    """The take at -20 LUFS, and what the manifest records of it."""
    loudness = integrated_loudness(take.samples, OUTPUT_RATE)
    gain_db = target_gain_db(loudness)
    gained = take.samples * 10.0 ** (gain_db / 20.0)
    clipped = int(np.count_nonzero(np.abs(gained) > 1.0))
    record = {
        "first_chunk_ms": take.first_chunk_ms,
        "loudness_lufs": loudness if math.isfinite(loudness) else None,
        "gain_db": gain_db,
        "clipped_samples": clipped,
    }
    return np.clip(gained, -1.0, 1.0), record


def _write_run(
    root: Root,
    run_rel: str,
    ids: Mapping[str, str],
    lines: Sequence[Line],
    renders: Mapping[str, tuple[str, list[Take]]],
) -> list[dict]:
    """Every clip, then the manifest last, so a run with a manifest is whole."""
    rows = []
    for index, line in enumerate(lines):
        arms = {}
        for arm in ARMS:
            model_id, takes = renders[arm]
            audio, record = _matched(takes[index])
            rel = f"{run_rel}/{CLIPS_DIR}/{line.line_id}-{arm}.flac"
            root.write_bytes(rel, _flac(audio))
            arms[arm] = {"clip": rel, **record, "model_id": model_id}
        rows.append({"line_id": line.line_id, "text": line.text, "arms": arms})
    manifest = {
        "schema": 1,
        "baseline": ids["baseline"],
        "trained": ids["trained"],
        "target_lufs": TARGET_LUFS,
        "sample_rate": OUTPUT_RATE,
        "character": {
            "double_gain": CHARACTER.double_gain,
            "limiter": CHARACTER.limiter is not None,
        },
        "lines": rows,
    }
    text = json.dumps(manifest, indent=2) + "\n"
    root.write_bytes(f"{run_rel}/{MANIFEST}", text.encode("utf-8"))
    return rows


def _nearest_rank(values: Sequence[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[max(1, math.ceil(p / 100 * len(ordered))) - 1]


def _summary(arm: str, rows: Sequence[dict]) -> str:
    firsts = [row["arms"][arm]["first_chunk_ms"] for row in rows]
    levels = [row["arms"][arm]["loudness_lufs"] for row in rows]
    finite = [level for level in levels if level is not None]
    span = f"{min(finite):.1f} to {max(finite):.1f} LUFS" if finite else "silent"
    clipped = sum(row["arms"][arm]["clipped_samples"] for row in rows)
    return (
        f"{arm}: {len(rows)} lines, first chunk median {statistics.median(firsts):.0f} ms, "
        f"p90 {_nearest_rank(firsts, 90):.0f} ms; loudness before matching {span}; "
        f"{clipped} samples clipped after matching"
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    load: Loader = load_pocket,
    clock: Clock = time.perf_counter,
) -> int:
    """Render both identities into a new bench run; 0 on success, 2 on a refusal, 1 on failure."""
    args = _parser().parse_args(argv)
    try:
        _check_ids(args.baseline, args.trained)
        root, run_rel = _open_run(args.out)
        lines = _read_lines(args.lines)
        placement, profile = _placement()
    except _RefusedError as exc:
        print(f"voice_ab: {exc}", file=sys.stderr)
        return 2
    ids = {"baseline": args.baseline, "trained": args.trained}
    try:
        renders = {
            arm: _render_arm(
                load, dataclasses.replace(placement, tts_voice=ids[arm]), profile, lines, clock
            )
            for arm in ARMS
        }
        rows = _write_run(root, run_rel, ids, lines, renders)
    except (SpeechEngineError, RootError) as exc:
        print(f"voice_ab: {exc}", file=sys.stderr)
        return 1
    for arm in ARMS:
        print(_summary(arm, rows))
    print(f"voice-ab: written to {run_rel}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
