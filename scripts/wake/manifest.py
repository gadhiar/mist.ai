"""The wake recording manifest: its check, and the training subset the model records.

A wake model is measured on recordings it never trained on, and a threshold is never chosen on
the recordings it is reported on. That only holds if the three splits share no file, no copy of
a file under another name, no homophone sentence, and no session holding `media`, since a
programme heard in two splits would leak its content. Any other recording session (the same
room, microphone and voice on the same day) may span the splits: the model then trains on every
room, microphone and day that tune and test measure it on, as it will run in one user's own
rooms, so tune and test do not show how it does in a new room or on a new day. This module
refuses a manifest that breaks any of those rules, naming both entries, so the check runs
before anything trains or evaluates.

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

`assign` gives every entry a split, so a new set of recordings needs no hand-kept split list.
Its entries may leave `split` out, and any `split` present is replaced; every other field is
checked as `check` checks it. Recordings sharing a session or a homophone sentence are linked
transitively into one group, which goes to one split. `check` refuses a shared sentence, or a
shared session holding media, across splits; keeping every other session whole is `assign`'s
own rule, which `check` does not require. Each group belongs to the (label, condition) holding
most of its duration, so every condition of every label is split on its own. Within each,
`random.Random(seed)` shuffles the groups and each goes to the split furthest below its
60/20/20 share of the duration, ties to train, then tune, then test. That keeps each split
within one group's duration of its share, and the same seed always gives the same manifest. It
prints the seed and each split's count and share, never a file name, and refuses an `--out`
that exists.

    python -m scripts.wake.manifest check --manifest <path>
    python -m scripts.wake.manifest split --manifest <path> --split train --out <path>
    python -m scripts.wake.manifest assign --manifest <path> --seed N --out <path>
"""

import argparse
import dataclasses
import hashlib
import json
import random
import re
import sys
import wave
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

LABELS = ("positive", "homophone", "media")
SPLITS = ("train", "tune", "test")
SPLIT_FIFTHS = {"train": 3, "tune": 1, "test": 1}
"""Each split's target share of a (label, condition)'s duration, in fifths: 60/20/20."""
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
DURATION_TOLERANCE_MS = 20
"""One uplink frame: a `duration_ms` within this of the audio's length is accepted."""

UNASSIGNED = "unassigned"
"""The split of an entry `assign` has read and not yet placed; never written."""

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


def _entry(root: Path, index: int, value: object, *, assigning: bool = False) -> Entry:
    """Check one entry; `assigning` ignores its `split` and reads it as `UNASSIGNED`."""
    file = value.get("file") if isinstance(value, dict) else None
    name = f"entry {index} ({file if isinstance(file, str) else '?'})"
    if not isinstance(value, dict):
        raise ManifestError(f"{name}: must be an object")
    required = _REQUIRED - {"split"} if assigning else _REQUIRED
    unknown = sorted(set(value) - _REQUIRED - _OPTIONAL)
    missing = sorted(required - set(value))
    if unknown:
        raise ManifestError(f"{name}: unknown keys {unknown}")
    if missing:
        raise ManifestError(f"{name}: missing keys {missing}")
    if value["label"] not in LABELS:
        raise ManifestError(f"{name}: label must be one of {list(LABELS)}, got {value['label']!r}")
    split = UNASSIGNED if assigning else value["split"]
    if not assigning and split not in SPLITS:
        raise ManifestError(f"{name}: split must be one of {list(SPLITS)}, got {split!r}")
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
        split=split,
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
    """Refuse any file, copy, media session or homophone sentence that two splits would share.

    A file or its contents appearing twice is refused even within one split, because it would
    count one recording twice. A homophone sentence is refused only across splits. A session
    may span the splits, unless one of its entries is `media`: then the refusal names a media
    entry and an entry of the session in another split. Such a pair is found when its later
    entry is read, so every media session across splits is refused.
    """
    by_path: dict[Path, Entry] = {}
    by_sha: dict[str, Entry] = {}
    by_session: dict[str, dict[str, Entry]] = {}
    media_by_session: dict[str, dict[str, Entry]] = {}
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
        first = by_session.setdefault(item.session, {})
        media = media_by_session.setdefault(item.session, {})
        # A media entry clashes with any entry of its session in another split, and any other
        # entry with a media one; the first of each split stands for the rest.
        clashes = first if item.label == "media" else media
        earlier = next((found for split, found in clashes.items() if split != item.split), None)
        if earlier is not None:
            raise ManifestError(
                f"{_both(earlier, item)} share session {item.session!r} across splits, "
                "and a session holding media stays in one split"
            )
        first.setdefault(item.split, item)
        if item.label == "media":
            media.setdefault(item.split, item)
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
            and file), a file or its contents is listed twice, or two splits share a homophone
            sentence or a session holding media (both entries named). A session without media
            may span the splits.
    """
    root, entries = _read(path, assigning=False)
    _check_splits(entries)
    return RecordingManifest(path=Path(path), root=root, entries=entries)


def _read(path: Path, *, assigning: bool) -> tuple[str, tuple[Entry, ...]]:
    """The manifest's root as written and its entries, each checked on its own."""
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
    entries = tuple(
        _entry(root_path, index, item, assigning=assigning)
        for index, item in enumerate(raw["entries"])
    )
    return root, entries


# -- Assigning splits -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class KeyAssignment:
    """One (label, condition)'s groups as `assign` placed them: each split's entries."""

    label: str
    condition: str
    splits: Mapping[str, tuple[Entry, ...]]

    @property
    def duration_ms(self) -> int:
        """The key's whole duration, which each split's share is a share of."""
        return sum(item.duration_ms for entries in self.splits.values() for item in entries)


def _linked_groups(entries: Sequence[Entry]) -> list[list[Entry]]:
    """Entries sharing a session, or as homophones a sentence, transitively; in manifest order.

    Each group stays in one split. For a shared homophone sentence, and a shared session holding
    media, that is what `_check_splits` requires; for any other session it is `assign`'s own rule,
    since `_check_splits` lets such a session span the splits.
    """
    parent = list(range(len(entries)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        low, high = sorted((find(first), find(second)))
        parent[high] = low

    by_session: dict[str, int] = {}
    by_phrase: dict[str, int] = {}
    for index, item in enumerate(entries):
        union(index, by_session.setdefault(item.session, index))
        if item.label == "homophone":
            union(index, by_phrase.setdefault(phrase_key(item.phrase), index))
    groups: dict[int, list[Entry]] = {}
    for index, item in enumerate(entries):
        groups.setdefault(find(index), []).append(item)
    return list(groups.values())


def _group_key(group: Sequence[Entry]) -> tuple[str, str]:
    """The (label, condition) holding most of the group's duration; ties by label, condition."""
    totals: dict[tuple[str, str], int] = {}
    for item in group:
        key = (item.label, item.condition)
        totals[key] = totals.get(key, 0) + item.duration_ms
    return min(totals, key=lambda key: (-totals[key], LABELS.index(key[0]), key[1]))


def assign_splits(entries: Sequence[Entry], seed: int) -> tuple[KeyAssignment, ...]:
    """Place every linked group in a split, per (label, condition), reproducibly from `seed`.

    A group goes to the split furthest below its share of the key's duration, so no split ends
    further from its share than the largest group of the key: a split only receives a group
    while it is the furthest below, and the deficits sum to zero at the end.
    """
    by_key: dict[tuple[str, str], list[list[Entry]]] = {}
    for group in _linked_groups(entries):
        by_key.setdefault(_group_key(group), []).append(group)
    placed: list[KeyAssignment] = []
    for label, condition in sorted(by_key, key=lambda key: (LABELS.index(key[0]), key[1])):
        groups = by_key[(label, condition)]
        # Not a security use: the seed only makes the split assignment reproducible.
        random.Random(seed).shuffle(groups)  # nosec B311
        total = sum(item.duration_ms for group in groups for item in group)
        assigned = dict.fromkeys(SPLITS, 0)
        members: dict[str, list[Entry]] = {split: [] for split in SPLITS}
        for group in groups:
            # Below-share in fifths of a millisecond, so the comparison is exact; `max` keeps
            # the first of equals, which is train, then tune, then test.
            split = max(SPLITS, key=lambda name: SPLIT_FIFTHS[name] * total - 5 * assigned[name])
            assigned[split] += sum(item.duration_ms for item in group)
            members[split].extend(group)
        placed.append(
            KeyAssignment(
                label=label,
                condition=condition,
                splits={
                    split: tuple(sorted(found, key=lambda item: item.index))
                    for split, found in members.items()
                },
            )
        )
    return tuple(placed)


def assigned_bytes(root: str, entries: Sequence[Entry], splits: Mapping[int, str]) -> bytes:
    """The manifest with each entry's `split` set, in manifest order, written as `split` is."""
    payload = {
        "root": root,
        "entries": [{**item.raw, "split": splits[item.index]} for item in entries],
    }
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
    return (text + "\n").encode("utf-8")


_OUT_EXISTS = "--out already exists; assign writes a new manifest, never over one"


def _assign(manifest: Path, seed: int, out: Path) -> int:
    """Write `manifest` with splits assigned to `out`, and print the seed and shares.

    What it prints names no file, not even `out`, so a summary can be shared without exposing
    the recordings' names. A refusal names the entries at fault, as `check` does.
    """
    if out.exists():
        raise ManifestError(_OUT_EXISTS)
    root, entries = _read(manifest, assigning=True)
    # Every entry reads as one split here, so this refuses only a file or its audio listed twice.
    _check_splits(entries)
    placed = assign_splits(entries, seed)
    splits = {
        item.index: split
        for key in placed
        for split, members in key.splits.items()
        for item in members
    }
    _check_splits([dataclasses.replace(item, split=splits[item.index]) for item in entries])
    data = assigned_bytes(root, entries, splits)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with out.open("xb") as handle:
            handle.write(data)
    except FileExistsError as exc:
        raise ManifestError(_OUT_EXISTS) from exc
    print(f"seed {seed}")
    for key in placed:
        total = key.duration_ms
        shares = ", ".join(
            f"{split} {len(members)} ({sum(item.duration_ms for item in members) / total:.1%})"
            for split, members in key.splits.items()
        )
        print(f"{key.label} {key.condition}: {shares}")
    return 0


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
    assign = commands.add_parser("assign", help="write the manifest with every split assigned")
    assign.add_argument("--manifest", required=True, type=Path)
    assign.add_argument("--seed", required=True, type=int)
    assign.add_argument("--out", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run `check`, `split` or `assign`; 0 on success, 2 when the manifest is refused."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "assign":
            return _assign(args.manifest, args.seed, args.out)
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
