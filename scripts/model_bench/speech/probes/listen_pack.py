"""Run 1's listening pack probe: loudness-match saved clips, run inside the base bench image.

Invocation, from `/work` (the repository, mounted read-only):

    python -m scripts.model_bench.speech.probes.listen_pack --jobs /jobs.json --speech /speech
        --out /out

`--jobs` lists `{"label", "src"}` pairs, `src` relative to the speech results root, which is
mounted with only its `tts/` directory present; `--out` is the page's `clips/` directory. For each
clip, ffmpeg's loudnorm filter runs twice: the first pass measures the clip, the second applies
those measurements in linear mode, so each clip gets one fixed gain rather than a level that
rides through the line. Every output is 48 kHz mono 16-bit with no metadata, so neither the
sample rate nor a header tells two engines' clips apart. A BS.1770 meter written here then
reads each output back, an independent check on what ffmpeg reports; one `LOUDNESS <json>` line
per clip goes to stdout for the host to keep with the key, since the measurements of the source
clips could tell engines apart and so must not sit in the page folder.

The probe synthesizes nothing and reads only the clips listed. Exit codes: 0 done, 2 refused
before any ffmpeg call (a malformed job, a source outside `tts/`, a missing source, an output that
already exists), 1 failed. Stdlib only.
"""

from __future__ import annotations

import argparse
import array
import itertools
import json
import math
import re
import subprocess
import sys
import wave
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from scripts.model_bench.speech import core
else:
    from .. import core

# -20 LUFS integrated is the realtime design's level (realtime-voice v1:663). The range target is
# set wide so that linear mode, which ffmpeg drops when a clip's measured range exceeds it, holds
# for speech; the true-peak ceiling leaves headroom for the 48 kHz resample.
TARGET_I = -20.0
TARGET_TP = -1.5
TARGET_LRA = 20.0
TOLERANCE_LU = 1.0
OUTPUT_RATE = 48000

REPORT_PREFIX = "LOUDNESS "
DONE_PREFIX = "LISTEN_PACK_DONE"
_LABEL_RE = re.compile(r"c[0-9]+")
_REPORT_FIELDS = (
    "input_i",
    "input_tp",
    "input_lra",
    "input_thresh",
    "output_i",
    "output_tp",
    "output_lra",
    "output_thresh",
    "target_offset",
)

FfmpegRun = Callable[[list[str]], tuple[int, str]]


class RefusedError(core.SpeechBenchError):
    """A job the probe refuses before calling ffmpeg."""


class FfmpegError(core.SpeechBenchError):
    """ffmpeg failed, or its loudnorm report is missing or unusable."""


# ---------------------------------------------------------------------------
# ffmpeg's two passes
# ---------------------------------------------------------------------------


def _target() -> str:
    return f"I={TARGET_I:g}:TP={TARGET_TP:g}:LRA={TARGET_LRA:g}"


def measure_argv(src: str) -> list[str]:
    """Pass 1: measure `src` and print loudnorm's report as JSON; write no audio."""
    loudnorm = f"loudnorm={_target()}:print_format=json"
    return [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-nostats",
        "-i",
        src,
        "-af",
        loudnorm,
        "-f",
        "null",
        "-",
    ]


def apply_argv(src: str, dst: str, measured: dict) -> list[str]:
    """Pass 2: one linear gain from pass 1's measurements, written as plain 48 kHz mono PCM."""
    loudnorm = (
        f"loudnorm={_target()}"
        f":measured_I={measured['input_i']:.2f}"
        f":measured_TP={measured['input_tp']:.2f}"
        f":measured_LRA={measured['input_lra']:.2f}"
        f":measured_thresh={measured['input_thresh']:.2f}"
        f":offset={measured['target_offset']:.2f}"
        ":linear=true:print_format=json"
    )
    return [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-nostats",
        "-n",
        "-i",
        src,
        "-af",
        loudnorm,
        "-ar",
        str(OUTPUT_RATE),
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        "-map_metadata",
        "-1",
        "-fflags",
        "+bitexact",
        "-flags:a",
        "+bitexact",
        dst,
    ]


def parse_loudnorm(stderr: str) -> dict:
    """The last loudnorm JSON report in ffmpeg's stderr, its numbers as floats (`-inf` kept)."""
    start = stderr.rfind("[Parsed_loudnorm")
    brace = stderr.find("{", start) if start >= 0 else -1
    end = stderr.find("}", brace) if brace >= 0 else -1
    if end < 0:
        raise FfmpegError("ffmpeg printed no loudnorm report")
    try:
        raw = json.loads(stderr[brace : end + 1])
        report = {key: float(raw[key]) for key in _REPORT_FIELDS}
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise FfmpegError(f"unreadable loudnorm report: {exc}") from exc
    report["normalization_type"] = str(raw.get("normalization_type", ""))
    return report


def run_ffmpeg(argv: list[str]) -> tuple[int, str]:
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, check=False, shell=False)
    except OSError as exc:
        raise FfmpegError(f"ffmpeg could not start: {exc}") from exc
    return proc.returncode, proc.stderr


# ---------------------------------------------------------------------------
# BS.1770 integrated loudness, for checking ffmpeg's output
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Biquad:
    """One second-order section: `b` the feed-forward and `a` the feedback coefficients (a0 = 1)."""

    b: tuple[float, float, float]
    a: tuple[float, float]


def k_weighting(rate: int) -> tuple[Biquad, Biquad]:
    """BS.1770's two K-weighting stages, the high shelf then the high pass, derived for `rate`.

    The standard tabulates the coefficients at 48 kHz only; the analogue prototypes below are the
    ones that reproduce that table, so clips at any rate are weighted the same way.
    """
    gain_db, q, fc = 3.999843853973347, 0.7071752369554196, 1681.974450955533
    k = math.tan(math.pi * fc / rate)
    vh = 10 ** (gain_db / 20)
    vb = vh**0.4996667741545416
    a0 = 1 + k / q + k * k
    shelf = Biquad(
        b=((vh + vb * k / q + k * k) / a0, 2 * (k * k - vh) / a0, (vh - vb * k / q + k * k) / a0),
        a=(2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0),
    )
    q, fc = 0.5003270373238773, 38.13547087602444
    k = math.tan(math.pi * fc / rate)
    a0 = 1 + k / q + k * k
    highpass = Biquad(b=(1.0, -2.0, 1.0), a=(2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0))
    return shelf, highpass


def _filter(samples: Sequence[float], section: Biquad) -> list[float]:
    b0, b1, b2 = section.b
    a1, a2 = section.a
    x1 = x2 = y1 = y2 = 0.0
    out = []
    append = out.append
    for x in samples:
        y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        x2, x1, y2, y1 = x1, x, y1, y
        append(y)
    return out


def _lufs(power: float) -> float:
    return -0.691 + 10 * math.log10(power) if power > 0 else float("-inf")


def integrated_loudness(samples: Sequence[float], rate: int) -> float:
    """Gated integrated loudness of a mono clip in LUFS, `-inf` for silence.

    400 ms blocks overlapping by 75%, the -70 LUFS absolute gate, then the gate 10 LU below the
    absolutely gated level. A clip shorter than one block is measured as a single block, which
    is what the acknowledgement clips need.
    """
    shelf, highpass = k_weighting(rate)
    weighted = _filter(_filter(samples, shelf), highpass)
    n = len(weighted)
    if n == 0:
        return float("-inf")
    cumulative = list(itertools.accumulate((v * v for v in weighted), initial=0.0))
    block = round(0.4 * rate)
    step = round(0.1 * rate)
    if n < block:
        powers = [cumulative[n] / n]
    else:
        powers = [
            (cumulative[s + block] - cumulative[s]) / block for s in range(0, n - block + 1, step)
        ]
    gated = [p for p in powers if _lufs(p) > -70.0]
    if not gated:
        return float("-inf")
    relative = _lufs(sum(gated) / len(gated)) - 10.0
    gated = [p for p in gated if _lufs(p) > relative]
    return _lufs(sum(gated) / len(gated))


def read_wav(path: Path) -> tuple[list[float], int]:
    """A 16-bit PCM WAV as floats in [-1, 1], channels averaged to mono, and its sample rate."""
    try:
        with wave.open(str(path), "rb") as clip:
            if clip.getsampwidth() != 2:
                raise FfmpegError(f"{path} is not 16-bit PCM")
            channels = clip.getnchannels()
            rate = clip.getframerate()
            frames = clip.readframes(clip.getnframes())
    except (OSError, wave.Error, EOFError) as exc:
        raise FfmpegError(f"cannot read {path}: {exc}") from exc
    pcm = array.array("h")
    pcm.frombytes(frames)
    if sys.byteorder == "big":
        pcm.byteswap()
    if channels == 1:
        return [v / 32768.0 for v in pcm], rate
    mono = [sum(pcm[i : i + channels]) / (channels * 32768.0) for i in range(0, len(pcm), channels)]
    return mono, rate


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Job:
    """One clip to loudness-match: its random label and its source under the speech root."""

    label: str
    src: Path
    dst: Path


def _source(speech: Path, raw: object) -> Path:
    rel = PurePosixPath(str(raw))
    if rel.is_absolute() or ".." in rel.parts or rel.parts[:1] != ("tts",) or len(rel.parts) < 3:
        raise RefusedError(f"source {raw!r} is outside the saved tts clips")
    path = speech.joinpath(*rel.parts)
    if not path.is_file():
        raise RefusedError(f"source {raw!r} does not exist")
    return path


def load_jobs(jobs_path: Path, speech: Path, out: Path) -> list[Job]:
    """Every job checked before ffmpeg runs, so a bad list writes no clip at all."""
    try:
        raw = json.loads(jobs_path.read_text(encoding="utf-8"))["clips"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RefusedError(f"cannot read the job list {jobs_path}: {exc}") from exc
    if not isinstance(raw, list) or not raw:
        raise RefusedError(f"{jobs_path} lists no clips")
    if not out.is_dir():
        raise RefusedError(f"--out {out} is not a directory")
    jobs, labels = [], set()
    for entry in raw:
        label = str(entry.get("label", "")) if isinstance(entry, dict) else ""
        if not _LABEL_RE.fullmatch(label) or label in labels:
            raise RefusedError(f"job {entry!r} has a missing, malformed or repeated label")
        labels.add(label)
        dst = out / f"{label}.wav"
        if dst.exists():
            raise RefusedError(f"{dst} already exists; refusing to overwrite it")
        jobs.append(Job(label, _source(speech, entry.get("src")), dst))
    return jobs


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def process(job: Job, run: FfmpegRun) -> dict:
    """Both loudnorm passes on one clip, then the meter on the output; the clip's report row."""
    code, stderr = run(measure_argv(str(job.src)))
    if code != 0:
        raise FfmpegError(f"{job.label}: the measuring pass exited {code}: {stderr[-400:]}")
    measured = parse_loudnorm(stderr)
    if not math.isfinite(measured["input_i"]):
        raise FfmpegError(f"{job.label}: the source clip is silent")
    code, stderr = run(apply_argv(str(job.src), str(job.dst), measured))
    if code != 0:
        raise FfmpegError(f"{job.label}: the normalising pass exited {code}: {stderr[-400:]}")
    applied = parse_loudnorm(stderr)
    samples, rate = read_wav(job.dst)
    lufs = integrated_loudness(samples, rate)
    return {
        "label": job.label,
        "input_i": _finite(measured["input_i"]),
        "input_tp": _finite(measured["input_tp"]),
        "input_lra": _finite(measured["input_lra"]),
        "input_thresh": _finite(measured["input_thresh"]),
        "target_offset": _finite(measured["target_offset"]),
        "output_i": _finite(applied["output_i"]),
        "normalization_type": applied["normalization_type"],
        "measured_lufs": _finite(lufs),
        "duration_s": len(samples) / rate if rate else 0.0,
        "on_target": math.isfinite(lufs) and abs(lufs - TARGET_I) <= TOLERANCE_LU,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m scripts.model_bench.speech.probes.listen_pack")
    parser.add_argument("--jobs", type=Path, required=True, help="the label and source list")
    parser.add_argument("--speech", type=Path, required=True, help="the speech results root")
    parser.add_argument("--out", type=Path, required=True, help="the page's clips directory")
    return parser


def _print_flush(line: str) -> None:
    print(line, flush=True)


def main(
    argv: Sequence[str] | None = None,
    *,
    run: FfmpegRun = run_ffmpeg,
    emit: Callable[[str], None] | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    say = emit or _print_flush
    try:
        jobs = load_jobs(args.jobs, args.speech, args.out)
    except RefusedError as exc:
        print(f"[REFUSED] {exc}", file=sys.stderr, flush=True)
        return 2
    try:
        for job in jobs:
            say(REPORT_PREFIX + json.dumps(process(job, run)))
    except core.SpeechBenchError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr, flush=True)
        return 1
    say(f"{DONE_PREFIX} {len(jobs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
