"""`prepare`'s container side, and the material format runs 3 and 4 read.

Raj's recordings (WAV or M4A in `recordings/`, each named by its utterance id) are converted by
ffmpeg to 16 kHz mono 16-bit WAV, checked, and written beside `manifest.json` in the material
directory, in core's manifest format so `core.verify_material` reads it. The recordings are
inputs to speech recognition and end of turn only; nothing here gives audio to a speech engine,
and the only options are the recordings directory, the output directory and the scripts file.

The frame detector lives here because both sides of the material use it: `prepare` measures the
silence after the last word with it, and the turn probe finds its decision points with it, so the
two cannot disagree about where a silence is.

Runs in the base bench image (numpy and ffmpeg are there), with the repository at `/work:ro`, the
recordings mounted read-only and only the material directory writable. A refusal exits 2 before
anything is written.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import wave
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import core

SAMPLE_RATE = 16000
RECORDING_SUFFIXES = (".wav", ".m4a")
MIN_DURATION_S = 2.0
MAX_DURATION_S = 8.0
MIN_TRAILING_SILENCE_MS = 300

# The detector of plan v2's run 4: 20 ms frames, silent when under the utterance's own noise
# floor plus 10 dB, a silence counted from 200 ms. The floor is the level of the quietest 100 ms
# (five frames, power-averaged) anywhere in the utterance. A percentile of the frame levels would
# land inside the speech of a long, dense recording (an 8 s line with 300 ms of room is under 4%
# room); the quietest window needs only 100 ms of room, which the 300 ms after the last word
# provides, and averaging five frames keeps one quiet frame inside a word from setting the floor.
FRAME_MS = 20
MIN_SILENCE_MS = 200
FLOOR_MARGIN_DB = 10.0
FLOOR_WINDOW_FRAMES = 5
LEVEL_FLOOR_DB = -100.0

REFUSED_EXIT = 2


class PrepareError(core.SpeechBenchError):
    """One or more recordings were refused; `problems` names each."""

    def __init__(self, problems: Sequence[str]):
        super().__init__("; ".join(problems))
        self.problems = list(problems)


class MaterialFormatError(core.SpeechBenchError):
    """A WAV file is not 16-bit PCM mono."""


# ---------------------------------------------------------------------------
# The material format
# ---------------------------------------------------------------------------


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """A 16-bit mono PCM WAV as float32 samples in [-1, 1) and its sample rate."""
    with wave.open(str(path), "rb") as handle:
        channels, width, rate = handle.getnchannels(), handle.getsampwidth(), handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if channels != 1 or width != 2:
        raise MaterialFormatError(
            f"{path.name}: {channels} channels of {8 * width}-bit samples, not 16-bit mono"
        )
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0, rate


# ---------------------------------------------------------------------------
# The frame detector
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Silence:
    """A run of silent frames after the first word, in milliseconds from the start."""

    start_ms: int
    end_ms: int
    trailing: bool


def frame_levels_db(audio: np.ndarray, sample_rate: int, frame_ms: int = FRAME_MS) -> np.ndarray:
    """RMS level of each whole frame in dBFS, clamped at -100 dBFS so digital silence is finite."""
    size = sample_rate * frame_ms // 1000
    count = len(audio) // size
    frames = np.asarray(audio[: count * size], dtype=np.float64).reshape(count, size)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    return np.maximum(20.0 * np.log10(np.maximum(rms, 1e-12)), LEVEL_FLOOR_DB)


def noise_floor_db(levels: np.ndarray) -> float:
    """The power-averaged level of the quietest `FLOOR_WINDOW_FRAMES` consecutive frames."""
    power = 10.0 ** (np.asarray(levels, dtype=np.float64) / 10.0)
    width = min(FLOOR_WINDOW_FRAMES, len(power))
    windows = np.convolve(power, np.full(width, 1.0 / width), mode="valid")
    return float(10.0 * np.log10(windows.min()))


def speech_frames(levels: np.ndarray) -> np.ndarray:
    """True for each frame at or above the noise floor plus 10 dB."""
    if len(levels) == 0:
        return np.zeros(0, dtype=bool)
    return levels >= noise_floor_db(levels) + FLOOR_MARGIN_DB


def silences(audio: np.ndarray, sample_rate: int, *, min_ms: int = MIN_SILENCE_MS) -> list[Silence]:
    """Every silence of `min_ms` or more after the first word, the trailing one marked.

    Room tone before the first word is not a silence: nothing has been said yet, so no detector
    would offer a decision there.
    """
    speech = speech_frames(frame_levels_db(audio, sample_rate))
    spoken = np.flatnonzero(speech)
    if len(spoken) == 0:
        return []
    min_frames = -(-min_ms // FRAME_MS)
    found = []
    start = None
    for index in range(int(spoken[0]), len(speech) + 1):
        silent = index < len(speech) and not speech[index]
        if silent and start is None:
            start = index
        elif not silent and start is not None:
            if index - start >= min_frames:
                found.append(
                    Silence(start * FRAME_MS, index * FRAME_MS, trailing=index == len(speech))
                )
            start = None
    return found


def trailing_silence_ms(audio: np.ndarray, sample_rate: int) -> int | None:
    """Milliseconds from the end of the last speech frame to the end of the audio; None if mute."""
    speech = speech_frames(frame_levels_db(audio, sample_rate))
    spoken = np.flatnonzero(speech)
    if len(spoken) == 0:
        return None
    total_ms = len(audio) * 1000 // sample_rate
    return int(total_ms - (int(spoken[-1]) + 1) * FRAME_MS)


# ---------------------------------------------------------------------------
# Conversion and checks
# ---------------------------------------------------------------------------


def ffmpeg_argv(src: Path, dst: Path) -> list[str]:
    """Ffmpeg to 16 kHz mono 16-bit PCM, bit-exact so the same recording gives the same sha256."""
    return [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-map_metadata",
        "-1",
        "-fflags",
        "+bitexact",
        "-flags:a",
        "+bitexact",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        str(dst),
    ]


def ffmpeg_convert(src: Path, dst: Path) -> None:
    proc = subprocess.run(ffmpeg_argv(src, dst), capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise PrepareError([f"{src.name}: ffmpeg failed: {proc.stderr.strip()}"])


def check_recording(audio: np.ndarray, sample_rate: int) -> tuple[list[str], int | None]:
    """The reasons a converted recording is refused, and its trailing silence in ms."""
    problems = []
    if sample_rate != SAMPLE_RATE:
        return [f"sample rate {sample_rate}, not {SAMPLE_RATE} after conversion"], None
    duration = len(audio) / sample_rate
    if not MIN_DURATION_S <= duration <= MAX_DURATION_S:
        problems.append(
            f"{duration:.2f} s long, outside {MIN_DURATION_S:g} to {MAX_DURATION_S:g} s"
        )
    trailing = trailing_silence_ms(audio, sample_rate)
    if trailing is None:
        problems.append("no speech found above the noise floor")
    elif trailing < MIN_TRAILING_SILENCE_MS:
        problems.append(
            f"{trailing} ms after the last word, under {MIN_TRAILING_SILENCE_MS} ms of silence"
        )
    return problems, trailing


def find_recordings(recordings_dir: Path, ids: Sequence[str]) -> tuple[dict[str, Path], list[str]]:
    """Each utterance's recording by id, and the problems: missing, doubled or unnamed files."""
    known = set(ids)
    found: dict[str, list[Path]] = {}
    problems = []
    for path in sorted(recordings_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in RECORDING_SUFFIXES or path.stem not in known:
            problems.append(
                f"{path.name}: not named <utterance id>.wav or .m4a for a scripted utterance"
            )
            continue
        found.setdefault(path.stem, []).append(path)
    chosen = {}
    for uid in ids:
        paths = found.get(uid, [])
        if not paths:
            problems.append(f"{uid}: no recording")
        elif len(paths) > 1:
            problems.append(f"{uid}: more than one recording ({', '.join(p.name for p in paths)})")
        else:
            chosen[uid] = paths[0]
    return chosen, problems


def prepare(
    recordings_dir: Path,
    out_dir: Path,
    utterances: Sequence[Mapping[str, str]],
    *,
    convert: Callable[[Path, Path], None] = ffmpeg_convert,
) -> list[dict]:
    """Convert and check every scripted utterance's recording, then write WAVs and the manifest.

    Everything is converted into a scratch directory and checked first; only when no recording is
    refused are the WAVs and `manifest.json` written, each opened exclusively, so a refusal or a
    rerun never leaves or overwrites partial material.
    """
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        raise core.OutputExistsError(f"{manifest_path} already exists; refusing to overwrite it")
    ids = [str(u["id"]) for u in utterances]
    chosen, problems = find_recordings(recordings_dir, ids)
    for uid in chosen:
        if (out_dir / f"{uid}.wav").exists():
            problems.append(f"{uid}: {out_dir / f'{uid}.wav'} already exists")

    items = []
    with tempfile.TemporaryDirectory(prefix="mist-prepare-") as scratch:
        converted = {}
        for utterance in utterances:
            uid = str(utterance["id"])
            if uid not in chosen:
                continue
            target = Path(scratch) / f"{uid}.wav"
            try:
                convert(chosen[uid], target)
                audio, rate = read_wav(target)
            except (PrepareError, MaterialFormatError, OSError, wave.Error) as exc:
                problems.append(f"{uid}: cannot read the converted recording: {exc}")
                continue
            refused, trailing = check_recording(audio, rate)
            problems += [f"{uid}: {p}" for p in refused]
            converted[uid] = target
            items.append(
                {
                    "id": uid,
                    "file": f"{uid}.wav",
                    "kind": str(utterance["kind"]),
                    "text": str(utterance["text"]),
                    "duration_s": round(len(audio) / rate, 4),
                    "sha256": core.sha256_file(target),
                    "trailing_silence_ms": trailing,
                    "source": chosen[uid].name,
                    "source_sha256": core.sha256_file(chosen[uid]),
                }
            )
        if problems:
            raise PrepareError(problems)
        for uid, target in converted.items():
            with (out_dir / f"{uid}.wav").open("xb") as handle:
                handle.write(target.read_bytes())

    core.write_json_exclusive(
        manifest_path,
        {
            "schema": 1,
            "_comment": (
                "Written by speech_bench prepare (mist-model-host plan v2): Raj's recordings as "
                "16 kHz mono 16-bit WAV, read only by speech recognition and end of turn."
            ),
            "sample_rate": SAMPLE_RATE,
            "items": items,
        },
    )
    return items


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.model_bench.speech.probes.prepare_material",
        description="Convert and check the recordings, then write the material manifest.",
    )
    parser.add_argument("--recordings", type=Path, required=True, help="read-only recordings")
    parser.add_argument("--out", type=Path, required=True, help="the material directory")
    parser.add_argument(
        "--utterances", type=Path, default=core.UTTERANCES_PATH, help="the scripts file"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    utterances = core.load_json(args.utterances)["utterances"]
    try:
        items = prepare(args.recordings, args.out, utterances)
    except PrepareError as exc:
        for problem in exc.problems:
            print(f"[FAIL] {problem}")
        return REFUSED_EXIT
    except core.OutputExistsError as exc:
        print(f"[FAIL] {exc}")
        return REFUSED_EXIT
    print(f"[ok] {len(items)} recordings written with {args.out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
