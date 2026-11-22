"""The verifier's training index: window specs over source files, never copies of audio.

The index (`index.jsonl`, one `Window` a line) names each window by its file's source, sha256
and path under the source's root, or for a recording its manifest entry index; its start and
end in samples; its label and kind; its cross-validation group; and the spotter cutoffs its
hit occurs at. `mine.py` writes the same records to `mined.jsonl`, and `build.py` checks every
one against the leakage guard before it writes the index and `provenance.json`. The trainer
reads a recording through the train-only manifest, so no private audio is copied.

A window is the verifier's (`backend/realtime/engines/verifier.py`): `BEFORE_MS` before the
hit frame to `AFTER_MS` after it, in whole 20 ms frames, clipped at the file's edges and never
padded, and kept only when it holds at least `MIN_WINDOW_MS`. A file mixed from others (a
synthetic clip on a background) is source `mixed`, and its window lists the files it was mixed
from in `inputs`, so the guard checks them too.
"""

import hashlib
import json
import math
import re
import wave
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np

from backend.realtime.engines.verifier import FRAME_MS, GRID

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
BEFORE_MS = 2000
AFTER_MS = 300
MIN_WINDOW_MS = 800
"""The shortest window kept: 40 frames, enough audio around the hit for the verifier to judge."""
BEFORE_FRAMES = BEFORE_MS // FRAME_MS
AFTER_FRAMES = AFTER_MS // FRAME_MS
MIN_WINDOW_FRAMES = MIN_WINDOW_MS // FRAME_MS

MINED_FILE = "mined.jsonl"
MINED_META = "meta.json"
INDEX_FILE = "index.jsonl"
INDEX_META = "meta.json"
PROVENANCE_FILE = "provenance.json"
CLIPS_FILE = "clips.jsonl"
SYNTH_META = "meta.json"

LABELS = ("positive", "negative")
KINDS = ("real_positive", "homophone", "media", "pre_name", "synthetic", "mined_hit", "random")


@dataclass(frozen=True, slots=True)
class Source:
    """A training source's licence and the version recorded in `provenance.json`."""

    licence: str
    version: str


SOURCES: Mapping[str, Source] = {
    "manifest": Source("private recordings, personal use only", "the manifest's train split"),
    "ami": Source(
        "CC BY 4.0",
        "AMI Meeting Corpus Mix-Headset, full-corpus ASR train meetings, annotations 1.6.2",
    ),
    "librispeech": Source("CC BY 4.0", "LibriSpeech train-clean-100 (OpenSLR 12)"),
    "musan": Source("CC BY 4.0", "MUSAN (OpenSLR 17)"),
    "rirs": Source("Apache-2.0", "RIRS_NOISES (OpenSLR 28)"),
    "piper": Source(
        "MIT (piper-sample-generator), CC BY 4.0 (its LibriTTS-R voice)",
        "piper-sample-generator",
    ),
    "chatterbox": Source("MIT (chatterbox-tts and the ResembleAI/chatterbox weights)", "0.1.7"),
    "mixed": Source("derived: each file carries its inputs' licences", "mine.py mixing"),
}
"""Every source a window or its inputs may come from."""

_SHA256 = re.compile(r"[0-9a-f]{64}")
_REF_KEYS = frozenset({"source", "sha256", "path", "entry", "samples"})
_WINDOW_KEYS = frozenset({"file", "start", "end", "label", "kind", "group", "cutoffs", "inputs"})


class DatasetError(ValueError):
    """A record, file or argument the data tools refuse; the CLIs exit 2 with it."""


@dataclass(frozen=True, slots=True)
class FileRef:
    """One source file: a manifest entry by index, or a path under the source's root.

    `samples` is its length at 16 kHz: a synthetic clip's after `synth.load_clip` resamples it,
    which is the audio a mix holds and the length `provenance.json`'s hours count.
    """

    source: str
    sha256: str
    path: str | None
    entry: int | None
    samples: int

    def to_json(self) -> dict[str, object]:
        """The record's JSON form."""
        return {
            "source": self.source,
            "sha256": self.sha256,
            "path": self.path,
            "entry": self.entry,
            "samples": self.samples,
        }


@dataclass(frozen=True, slots=True)
class Window:
    """One training window over `file`, `[start, end)` in samples."""

    file: FileRef
    start: int
    end: int
    label: str
    kind: str
    group: str
    cutoffs: tuple[float, ...]
    inputs: tuple[FileRef, ...] = ()

    def refs(self) -> tuple[FileRef, ...]:
        """Every file the window's audio was read or mixed from."""
        return (self.file, *self.inputs)

    def to_json(self) -> dict[str, object]:
        """The record's JSON form."""
        return {
            "file": self.file.to_json(),
            "start": self.start,
            "end": self.end,
            "label": self.label,
            "kind": self.kind,
            "group": self.group,
            "cutoffs": list(self.cutoffs),
            "inputs": [ref.to_json() for ref in self.inputs],
        }


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def ref_from_json(value: object, where: str) -> FileRef:
    """A checked `FileRef`: a known source, a sha256, and an entry index or a relative path.

    Raises:
        DatasetError: The record is malformed; the message starts with `where`.
    """
    if not isinstance(value, dict) or set(value) != _REF_KEYS:
        raise DatasetError(f"{where}: a file must be an object with keys {sorted(_REF_KEYS)}")
    source = value["source"]
    if source not in SOURCES:
        raise DatasetError(f"{where}: unknown source {source!r}")
    if not isinstance(value["sha256"], str) or not _SHA256.fullmatch(value["sha256"]):
        raise DatasetError(f"{where}: sha256 must be 64 lower-case hex digits")
    if not _is_int(value["samples"]) or value["samples"] < 1:
        raise DatasetError(f"{where}: samples must be a positive integer")
    path, entry = value["path"], value["entry"]
    if source == "manifest":
        if path is not None or not _is_int(entry) or entry < 0:
            raise DatasetError(f"{where}: a manifest file is named by its entry index alone")
    else:
        relative = PurePosixPath(path) if isinstance(path, str) else None
        if (
            entry is not None
            or relative is None
            or not path
            or "\\" in path
            or relative.is_absolute()
            or ".." in relative.parts
        ):
            raise DatasetError(f"{where}: a {source} file is named by a relative path alone")
    return FileRef(source, value["sha256"], path, entry, value["samples"])


def window_from_json(value: object, where: str) -> Window:
    """A checked `Window`.

    Raises:
        DatasetError: The record is malformed; the message starts with `where`.
    """
    if not isinstance(value, dict) or set(value) != _WINDOW_KEYS:
        raise DatasetError(f"{where}: a window must be an object with keys {sorted(_WINDOW_KEYS)}")
    file = ref_from_json(value["file"], f"{where} file")
    start, end = value["start"], value["end"]
    if not (_is_int(start) and _is_int(end) and 0 <= start < end <= file.samples):
        raise DatasetError(f"{where}: start and end must satisfy 0 <= start < end <= samples")
    if value["label"] not in LABELS:
        raise DatasetError(f"{where}: label must be one of {list(LABELS)}")
    if value["kind"] not in KINDS:
        raise DatasetError(f"{where}: kind must be one of {list(KINDS)}")
    if not isinstance(value["group"], str) or not value["group"]:
        raise DatasetError(f"{where}: group must be a non-empty string")
    cutoffs = value["cutoffs"]
    if not isinstance(cutoffs, list) or any(cutoff not in GRID for cutoff in cutoffs):
        raise DatasetError(f"{where}: cutoffs must be grid cutoffs, 0.01 to 0.99 by 0.01")
    inputs = value["inputs"]
    if not isinstance(inputs, list):
        raise DatasetError(f"{where}: inputs must be a list")
    return Window(
        file=file,
        start=start,
        end=end,
        label=value["label"],
        kind=value["kind"],
        group=value["group"],
        cutoffs=tuple(float(cutoff) for cutoff in cutoffs),
        inputs=tuple(
            ref_from_json(item, f"{where} input {number}") for number, item in enumerate(inputs)
        ),
    )


def write_windows(path: Path, windows: Iterable[Window]) -> None:
    """Write windows as JSON lines, keys sorted, so the same windows give the same bytes."""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for window in windows:
            handle.write(json.dumps(window.to_json(), sort_keys=True) + "\n")


def read_windows(path: Path) -> list[Window]:
    """Read and check a JSON-lines file of windows.

    Raises:
        DatasetError: The file is missing or a line is malformed; the message names the line.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DatasetError(f"{path.name} could not be read: {exc}") from exc
    windows = []
    for number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{path.name} line {number} is not JSON: {exc}") from exc
        windows.append(window_from_json(value, f"{path.name} line {number}"))
    return windows


def window_bounds(hit_pos: int, frames: int) -> tuple[int, int] | None:
    """The samples of the window around frame `hit_pos` in a file of `frames` whole frames.

    None when the hit is not in a whole frame or the window is under `MIN_WINDOW_FRAMES`.
    """
    if not 0 <= hit_pos < frames:
        return None
    first = max(0, hit_pos - BEFORE_FRAMES)
    last = min(frames - 1, hit_pos + AFTER_FRAMES)
    if last - first + 1 < MIN_WINDOW_FRAMES:
        return None
    return first * FRAME_SAMPLES, (last + 1) * FRAME_SAMPLES


def parse_root_map(values: Sequence[str]) -> dict[str, str]:
    """`--root-map <from>=<to>` arguments as `load_manifest`'s `root_map`.

    Raises:
        DatasetError: An argument has no `=` or an empty side.
    """
    mapping = {}
    for value in values:
        source, separator, target = value.partition("=")
        if not separator or not source or not target:
            raise DatasetError(f"--root-map must be <from>=<to>, got {value!r}")
        mapping[source] = target
    return mapping


def sha256_file(path: Path) -> str:
    """The file's sha256, read in blocks."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def sha256_bytes(data: bytes) -> str:
    """The sha256 of `data`."""
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True, slots=True)
class WavInfo:
    """A WAV header's format."""

    rate: int
    channels: int
    width: int
    frames: int


def wav_info(path: Path) -> WavInfo:
    """Read a WAV header.

    Raises:
        DatasetError: The file is not a PCM WAV file.
    """
    try:
        with wave.open(str(path), "rb") as handle:
            return WavInfo(
                handle.getframerate(),
                handle.getnchannels(),
                handle.getsampwidth(),
                handle.getnframes(),
            )
    except (wave.Error, EOFError, OSError) as exc:
        raise DatasetError(f"{path.name} is not a PCM WAV file: {exc}") from exc


def is_pcm16_16k_mono(info: WavInfo) -> bool:
    """True for the uplink's format, the one every training window is read in."""
    return (info.rate, info.channels, info.width) == (SAMPLE_RATE, 1, SAMPLE_WIDTH)


def read_wav_segment(path: Path, start: int, count: int) -> np.ndarray:
    """`count` int16 samples of a 16 kHz mono PCM16 WAV from sample `start`, fewer at its end.

    Raises:
        DatasetError: The file is not 16 kHz mono PCM16.
    """
    info = wav_info(path)
    if not is_pcm16_16k_mono(info):
        raise DatasetError(f"{path.name} is not 16 kHz mono PCM16")
    with wave.open(str(path), "rb") as handle:
        handle.setpos(min(max(start, 0), info.frames))
        data = handle.readframes(max(count, 0))
    return np.frombuffer(data, dtype="<i2").copy()


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    """A file's samples as float32 in [-1, 1], channels averaged, and its sample rate.

    WAV is read with the standard library (PCM16 or PCM32); FLAC, LibriSpeech's format, with
    soundfile, imported here only.

    Raises:
        DatasetError: The file cannot be read as audio.
    """
    if path.suffix.lower() == ".flac":
        import soundfile  # noqa: PLC0415 - only the FLAC sources need it

        try:
            data, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
        except (RuntimeError, OSError) as exc:
            raise DatasetError(f"{path.name} could not be read: {exc}") from exc
        return data.mean(axis=1).astype(np.float32), int(rate)
    info = wav_info(path)
    if info.width not in (2, 4):
        raise DatasetError(f"{path.name} has {8 * info.width}-bit samples; 16 or 32 are read")
    with wave.open(str(path), "rb") as handle:
        data = handle.readframes(info.frames)
    dtype, scale = ("<i2", 32768.0) if info.width == 2 else ("<i4", 2147483648.0)
    samples = np.frombuffer(data, dtype=dtype).astype(np.float32) / np.float32(scale)
    samples = samples.reshape(-1, info.channels).mean(axis=1)
    return samples.astype(np.float32), info.rate


def to_16k(samples: np.ndarray, rate: int) -> np.ndarray:
    """Float samples resampled to 16 kHz with a polyphase filter, unchanged when already so."""
    if rate == SAMPLE_RATE:
        return samples.astype(np.float32)
    from scipy.signal import resample_poly  # noqa: PLC0415 - only resampling needs scipy

    common = math.gcd(SAMPLE_RATE, rate)
    resampled = resample_poly(samples, SAMPLE_RATE // common, rate // common)
    return np.asarray(resampled, dtype=np.float32)


def to_pcm16(samples: np.ndarray) -> np.ndarray:
    """Float samples in [-1, 1] as int16, by 32768 as the verifier reads them back, clipped."""
    scaled = np.round(np.asarray(samples, dtype=np.float64) * 32768.0)
    return np.clip(scaled, -32768, 32767).astype("<i2")


def write_wav(path: Path, samples: np.ndarray, rate: int = SAMPLE_RATE) -> None:
    """Write int16 mono samples as a PCM16 WAV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(SAMPLE_WIDTH)
        handle.setframerate(rate)
        handle.writeframes(np.asarray(samples, dtype="<i2").tobytes())


def is_pcm16_16k(info: WavInfo) -> bool:
    """True for 16 kHz PCM16 in any channel count, the formats `downmix_pcm16` reads."""
    return (info.rate, info.width) == (SAMPLE_RATE, SAMPLE_WIDTH)


def downmix_pcm16(source: Path, dest: Path, block_frames: int = 1 << 18) -> int:
    """Write a 16 kHz PCM16 WAV as mono, each sample the mean of its channels, and count them.

    The mean is rounded to the nearest integer and clipped to int16: `read_audio` averages the
    channels and `to_pcm16` rounds and clips, so the mono file holds what this module reads from
    the multichannel one. The file is read in blocks, an AMI meeting at a time.

    Raises:
        DatasetError: `source` is not 16 kHz PCM16.
    """
    info = wav_info(source)
    if not is_pcm16_16k(info):
        raise DatasetError(f"{source.name} is not 16 kHz PCM16")
    frame_bytes = SAMPLE_WIDTH * info.channels
    written = 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(source), "rb") as reader, wave.open(str(dest), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(SAMPLE_WIDTH)
        writer.setframerate(SAMPLE_RATE)
        while data := reader.readframes(block_frames):
            whole = len(data) - len(data) % frame_bytes
            frames = np.frombuffer(data[:whole], dtype="<i2").reshape(-1, info.channels)
            mean = np.round(frames.astype(np.float64).mean(axis=1))
            writer.writeframes(np.clip(mean, -32768, 32767).astype("<i2").tobytes())
            written += len(frames)
    return written


def read_sums(path: Path) -> dict[str, str]:
    """A `sha256sum` listing as name to sha256: `<sha256>  <name>` or `<sha256> *<name>`.

    Raises:
        DatasetError: The file is missing or a line is malformed.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DatasetError(f"{path} could not be read: {exc}") from exc
    sums = {}
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        digest, _, name = line.partition(" ")
        name = name[1:] if name[:1] in (" ", "*") else name
        if not _SHA256.fullmatch(digest) or not name:
            raise DatasetError(f"{path} line {number} is not '<sha256>  <name>'")
        sums[name] = digest
    return sums


def sums_bytes(sums: Mapping[str, str]) -> bytes:
    """A `sha256sum` listing in name order."""
    return "".join(f"{sums[name]}  {name}\n" for name in sorted(sums)).encode("utf-8")
