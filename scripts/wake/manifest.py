"""The wake recording manifest: its check, and the training subset the model records.

A wake model is measured on recordings it never trained on, and a threshold is never chosen on
the recordings it is reported on. That only holds if the three splits share nothing a model can
learn from: no file, no copy of a file under another name, no recording session of any label
(the same room, microphone and voice on the same day), and no homophone sentence. This module
refuses a manifest that breaks any of those, naming both entries, so the check runs before
anything trains or evaluates.

The manifest is a JSON object:

    {"root": "<absolute directory>",
     "entries": [{"file", "label", "split", "phrase", "condition", "duration_ms", "session",
                  "reference_file"?}, ...]}

`file` and `reference_file` are paths relative to `root`. `label` is `positive`, `homophone` or
`media`; `split` is `train`, `tune` or `test`. `reference_file` is the PC's own output recorded
beside a microphone recording of media on speakers, which the harness scores for the gate's
self-media check. Every file is 16 kHz mono PCM16 WAV, the format the uplink carries.

`split` writes one split's entries as a manifest of their own and prints its sha256. The model's
`training` field records that hash, which is how the harness proves the model it measures was
trained from this manifest's `train` split and not from anything `tune` or `test` holds.

    python -m scripts.wake.manifest check --manifest <path>
    python -m scripts.wake.manifest split --manifest <path> --split train --out <path>
"""

import argparse
import hashlib
import json
import re
import sys
import wave
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

LABELS = ("positive", "homophone", "media")
SPLITS = ("train", "tune", "test")
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
DURATION_TOLERANCE_MS = 20
"""One uplink frame: a `duration_ms` within this of the audio's length is accepted."""

_MANIFEST_KEYS = frozenset({"root", "entries"})
_REQUIRED = frozenset({"file", "label", "split", "phrase", "condition", "duration_ms", "session"})
_OPTIONAL = frozenset({"reference_file"})
_WORDS = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")


class ManifestError(ValueError):
    """A manifest the evaluation cannot trust; the message names the entry at fault."""


@dataclass(frozen=True, slots=True)
class Entry:
    """One checked recording.

    `file` is the id the reports use: the path relative to the root, as the manifest gives it.
    `frames` is the WAV's sample count. `raw` is the entry as written, which the split subset
    reproduces byte for byte.
    """

    index: int
    file: str
    path: Path
    label: str
    split: str
    phrase: str
    condition: str
    duration_ms: int
    session: str
    reference_file: str | None
    reference_path: Path | None
    sha256: str
    frames: int
    raw: Mapping[str, object]

    @property
    def name(self) -> str:
        """How a refusal names the entry."""
        return f"entry {self.index} ({self.file})"


@dataclass(frozen=True, slots=True)
class RecordingManifest:
    """A manifest that passed every check."""

    path: Path
    root: str
    entries: tuple[Entry, ...]

    def split(self, name: str) -> tuple[Entry, ...]:
        """The entries of one split, in manifest order."""
        return tuple(item for item in self.entries if item.split == name)


def phrase_key(phrase: str) -> str:
    """A sentence compared without case, spacing or punctuation, as a speaker would say it."""
    return " ".join(_WORDS.findall(phrase.lower().replace("’", "'")))


def read_pcm16(path: Path) -> bytes:
    """The sample data of a WAV the manifest check accepted."""
    with wave.open(str(path), "rb") as handle:
        return handle.readframes(handle.getnframes())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _audio(path: Path, what: str, name: str) -> int:
    """Check one WAV's format and return its sample count."""
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            frames = handle.getnframes()
    except (wave.Error, EOFError, OSError) as exc:
        raise ManifestError(f"{name}: {what} is not a PCM WAV file: {exc}") from exc
    if rate != SAMPLE_RATE:
        raise ManifestError(f"{name}: {what} has sample rate {rate} Hz, expected {SAMPLE_RATE}")
    if channels != 1:
        raise ManifestError(f"{name}: {what} has {channels} channels, expected 1 channel (mono)")
    if width != SAMPLE_WIDTH:
        raise ManifestError(f"{name}: {what} has {8 * width}-bit samples, expected 16-bit PCM")
    return frames


def _locate(root: Path, value: object, key: str, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{name}: {key} must be a non-empty relative path")
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ManifestError(f"{name}: {key} {value} is outside the root {root}")
    if not path.is_file():
        raise ManifestError(f"{name}: {key} {value} not found under {root}")
    return path


def _text(value: object, key: str, name: str, *, empty: bool) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()):
        kind = "a string" if empty else "a non-empty string"
        raise ManifestError(f"{name}: {key} must be {kind}, got {value!r}")
    return value


def _entry(root: Path, index: int, value: object) -> Entry:
    file = value.get("file") if isinstance(value, dict) else None
    name = f"entry {index} ({file if isinstance(file, str) else '?'})"
    if not isinstance(value, dict):
        raise ManifestError(f"{name}: must be an object")
    unknown = sorted(set(value) - _REQUIRED - _OPTIONAL)
    missing = sorted(_REQUIRED - set(value))
    if unknown:
        raise ManifestError(f"{name}: unknown keys {unknown}")
    if missing:
        raise ManifestError(f"{name}: missing keys {missing}")
    if value["label"] not in LABELS:
        raise ManifestError(f"{name}: label must be one of {list(LABELS)}, got {value['label']!r}")
    if value["split"] not in SPLITS:
        raise ManifestError(f"{name}: split must be one of {list(SPLITS)}, got {value['split']!r}")
    session = _text(value["session"], "session", name, empty=False)
    phrase = _text(value["phrase"], "phrase", name, empty=True)
    condition = _text(value["condition"], "condition", name, empty=False)
    duration_ms = value["duration_ms"]
    if not isinstance(duration_ms, int) or isinstance(duration_ms, bool) or duration_ms < 1:
        raise ManifestError(f"{name}: duration_ms must be a positive integer, got {duration_ms!r}")

    path = _locate(root, file, "file", name)
    frames = _audio(path, f"file {file}", name)
    measured_ms = frames * 1000 / SAMPLE_RATE
    if abs(measured_ms - duration_ms) > DURATION_TOLERANCE_MS:
        raise ManifestError(
            f"{name}: duration_ms is {duration_ms}, but the audio lasts {measured_ms:.0f} ms"
        )
    reference = value.get("reference_file")
    reference_path = None
    if reference is not None:
        reference_path = _locate(root, reference, "reference_file", name)
        _audio(reference_path, f"reference_file {reference}", name)
    return Entry(
        index=index,
        file=file,
        path=path,
        label=value["label"],
        split=value["split"],
        phrase=phrase,
        condition=condition,
        duration_ms=duration_ms,
        session=session,
        reference_file=reference,
        reference_path=reference_path,
        sha256=_sha256(path),
        frames=frames,
        raw=value,
    )


def _both(first: Entry, second: Entry) -> str:
    return (
        f"entry {first.index} ({first.file}, {first.split}) and "
        f"entry {second.index} ({second.file}, {second.split})"
    )


def _check_splits(entries: Sequence[Entry]) -> None:
    """Refuse any file, copy, session or homophone sentence that two splits would share.

    A file or its contents appearing twice is refused even within one split, because it would
    count one recording twice. A session or a sentence is refused only across splits.
    """
    by_path: dict[Path, Entry] = {}
    by_sha: dict[str, Entry] = {}
    by_session: dict[str, Entry] = {}
    by_phrase: dict[str, Entry] = {}
    for item in entries:
        audio = [(item.path, item.sha256)]
        if item.reference_path is not None:
            audio.append((item.reference_path, _sha256(item.reference_path)))
        for path, digest in audio:
            if path in by_path:
                raise ManifestError(f"{_both(by_path[path], item)} list the same file {path}")
            if digest in by_sha:
                raise ManifestError(f"{_both(by_sha[digest], item)} hold audio with one sha256")
            by_path[path] = item
            by_sha[digest] = item
        earlier = by_session.setdefault(item.session, item)
        if earlier.split != item.split:
            raise ManifestError(
                f"{_both(earlier, item)} share session {item.session!r} across splits"
            )
        if item.label == "homophone":
            key = phrase_key(item.phrase)
            earlier = by_phrase.setdefault(key, item)
            if earlier.split != item.split:
                raise ManifestError(
                    f"{_both(earlier, item)} share a homophone sentence across splits"
                )


def load_manifest(path: Path) -> RecordingManifest:
    """Read and check a recording manifest.

    Raises:
        ManifestError: The manifest is malformed, an entry fails a check (named by its index
            and file), or two splits share a file, its contents, a session or a homophone
            sentence (both entries named).
    """
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ManifestError(f"{path}: manifest could not be read: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{path}: manifest is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict) or set(raw) != _MANIFEST_KEYS:
        raise ManifestError(f"{path}: manifest must be an object with exactly {{root, entries}}")
    root = raw["root"]
    if not isinstance(root, str) or not Path(root).is_absolute():
        raise ManifestError(f"{path}: root must be an absolute directory, got {root!r}")
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise ManifestError(f"{path}: root {root} is not a directory")
    if not isinstance(raw["entries"], list):
        raise ManifestError(f"{path}: entries must be a list")
    entries = tuple(_entry(root_path, index, item) for index, item in enumerate(raw["entries"]))
    _check_splits(entries)
    return RecordingManifest(path=Path(path), root=root, entries=entries)


def subset_bytes(manifest: RecordingManifest, split: str) -> bytes:
    """One split's entries as a manifest of their own, in a form whose hash is reproducible."""
    payload = {"root": manifest.root, "entries": [dict(item.raw) for item in manifest.split(split)]}
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
    return (text + "\n").encode("utf-8")


def subset_sha256(manifest: RecordingManifest, split: str) -> str:
    """The sha256 `split` prints for this split: the value a model's `training` field holds."""
    return hashlib.sha256(subset_bytes(manifest, split)).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m scripts.wake.manifest", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="check a manifest and print its counts")
    check.add_argument("--manifest", required=True, type=Path)
    split = commands.add_parser("split", help="write one split's entries and print its sha256")
    split.add_argument("--manifest", required=True, type=Path)
    split.add_argument("--split", required=True, choices=SPLITS)
    split.add_argument("--out", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run `check` or `split`; 0 on success, 2 when the manifest is refused."""
    args = _parser().parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
    except ManifestError as exc:
        print(f"manifest: {exc}", file=sys.stderr)
        return 2
    if args.command == "check":
        for split in SPLITS:
            entries = manifest.split(split)
            counts = ", ".join(
                f"{label} {sum(item.label == label for item in entries)}" for label in LABELS
            )
            print(f"{split}: {counts}")
        return 0
    data = subset_bytes(manifest, args.split)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(data)
    print(hashlib.sha256(data).hexdigest())
    return 0


if __name__ == "__main__":
    sys.exit(main())
