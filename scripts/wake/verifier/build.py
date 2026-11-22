"""Build the verifier's dataset index from the mined windows, behind the leakage guard.

    python -m scripts.wake.verifier.build --mined <dir> --manifest <full manifest> \
        --root-map <from>=<to> --ami-root <ami dir> --out <dir>

The verifier is judged on the manifest's tune and test recordings and on AMI dev, and its
threshold table on AMI calibration audio; a window from any of them in training would make
every one of those figures a measure of memory. So every file a window was read or mixed from
is checked here, and the build is refused (exit 2, nothing written) when any of them is:

- a manifest recording whose sha256 no entry of the full manifest has, or a tune or test entry
  has. A recording is resolved by sha256, because mine numbers it by its place in the
  train-only manifest (as train reads it), not by its place in the full one;
- an AMI dev or eval meeting (`stage.AMI_DEV`, `stage.AMI_EVAL`), a train meeting that failed
  the speaker filter, or a meeting outside the mining share of `ami/train/split.json`, or an
  AMI file whose sha256 is not the one `ami/train/SHA256SUMS` lists;
- any file whose sha256 is listed in `ami/dev/SHA256SUMS` or `ami/eval/SHA256SUMS`, or is a
  tune or test recording's (or its reference's);
- a mixed file whose bytes are not the ones mined.

It also refuses mined output whose recorded manifest sha256 is not the full manifest's train
`subset_sha256`: the miner must have read the train-only manifest of this very manifest. This
is the one process, with the host step before it, that loads the full manifest (which opens
and hashes every entry); the guard reads tune and test only for their sha256s.

It writes `index.jsonl` (the windows, as mined), `meta.json` (the source roots) and
`provenance.json`: `{"version": 1, "training": <train subset sha256>, "file_sha256": [every
file a window was read or mixed from, sorted], "sources": {<source>: {"licence", "version",
"files", "hours"}}, "excluded": {"ami_dev_sha256sums", "ami_eval_sha256sums"}}`, the last two
the held-out listings' own sha256s. It prints counts by source and label, never a file name.
Exit 0 built, 2 refused.
"""

import argparse
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from ..manifest import ManifestError, RecordingManifest, load_manifest, subset_sha256
from .dataset import (
    INDEX_FILE,
    INDEX_META,
    MINED_FILE,
    MINED_META,
    PROVENANCE_FILE,
    SAMPLE_RATE,
    SOURCES,
    DatasetError,
    FileRef,
    Window,
    parse_root_map,
    read_sums,
    read_windows,
    sha256_file,
    write_windows,
)
from .stage import AMI_DEV, AMI_EVAL, SPLIT, SUMS, WAV_SUFFIX


class GuardError(Exception):
    """A refusal by the leakage guard; the command exits 2 with the message."""


@dataclass(frozen=True, slots=True)
class Guard:
    """What the guard checks each file against."""

    recordings: Mapping[str, frozenset[str]]
    """Each full-manifest recording's sha256 and the splits of the entries holding it."""
    held_out: Mapping[str, str]
    train_sums: Mapping[str, str]
    mining: frozenset[str]
    excluded: frozenset[str]
    mined: Path

    def check(self, ref: FileRef, where: str) -> None:
        """Refuse `ref` if it is anything training must not read.

        Raises:
            GuardError: The file is outside what training may read; the message says why.
        """
        if ref.source == "manifest":
            self._recording(ref, where)
        elif ref.source == "ami":
            self._ami(ref, where)
        elif ref.source == "mixed" and sha256_file(self.mined / (ref.path or "")) != ref.sha256:
            raise GuardError(f"{where}: the mixed file's sha256 is not the one mined")
        if ref.sha256 in self.held_out:
            raise GuardError(f"{where}: its sha256 is {self.held_out[ref.sha256]}")

    def _recording(self, ref: FileRef, where: str) -> None:
        """Resolve a mined recording in the full manifest by its sha256, never by `ref.entry`.

        `ref.entry` is the recording's place in the train-only manifest mine and train read,
        which is not its place in the full manifest once tune or test entries sit between the
        train ones; the sha256 names the same recording in both.
        """
        splits = self.recordings.get(ref.sha256)
        if not splits:
            raise GuardError(
                f"{where}: the recording mined as train entry {ref.entry} has a sha256 no "
                "manifest entry has"
            )
        outside = sorted(splits - {"train"})
        if outside:
            raise GuardError(
                f"{where}: the recording mined as train entry {ref.entry} has the sha256 of a "
                f"{outside[0]} entry, not in the manifest's train split"
            )

    def _ami(self, ref: FileRef, where: str) -> None:
        name = PurePosixPath(ref.path or "").name
        if not name.endswith(WAV_SUFFIX) or name != ref.path:
            raise GuardError(f"{where}: an AMI file is <meeting>{WAV_SUFFIX} in ami/train")
        meeting = name[: -len(WAV_SUFFIX)]
        if meeting in AMI_DEV:
            raise GuardError(f"{where}: {meeting} is an AMI dev meeting")
        if meeting in AMI_EVAL:
            raise GuardError(f"{where}: {meeting} is an AMI eval meeting")
        if meeting in self.excluded:
            raise GuardError(f"{where}: {meeting} failed the speaker filter")
        if meeting not in self.mining:
            raise GuardError(f"{where}: {meeting} is not in the mining share")
        if self.train_sums.get(name) != ref.sha256:
            raise GuardError(f"{where}: {meeting}'s sha256 is not the one ami/train/{SUMS} lists")


def _json(path: Path, what: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GuardError(f"{what} could not be read: {exc}") from exc
    if not isinstance(value, dict):
        raise GuardError(f"{what} must be a JSON object")
    return value


def _guard(manifest: RecordingManifest, ami_root: Path, mined: Path) -> tuple[Guard, dict]:
    held_out: dict[str, str] = {}
    excluded_sums = {}
    for name in ("dev", "eval"):
        path = ami_root / name / SUMS
        if not path.is_file():
            raise GuardError(f"ami/{name}/{SUMS} is missing; the guard cannot check against it")
        for digest in read_sums(path).values():
            held_out.setdefault(digest, f"listed in ami/{name}/{SUMS}")
        excluded_sums[f"ami_{name}_sha256sums"] = sha256_file(path)
    for item in manifest.entries:
        if item.split != "train":
            held_out.setdefault(item.sha256, f"a {item.split} recording's")
            if item.reference_path is not None:
                held_out.setdefault(sha256_file(item.reference_path), f"a {item.split} reference's")
    split = _json(ami_root / "train" / SPLIT, f"ami/train/{SPLIT}")
    try:
        mining = frozenset(split["shares"]["mining"])
        excluded = frozenset(split["excluded"])
    except (KeyError, TypeError) as exc:
        raise GuardError(f"ami/train/{SPLIT} has no shares or excluded: {exc}") from exc
    recordings: dict[str, set[str]] = {}
    for item in manifest.entries:
        recordings.setdefault(item.sha256, set()).add(item.split)
    guard = Guard(
        recordings={digest: frozenset(splits) for digest, splits in recordings.items()},
        held_out=held_out,
        train_sums=read_sums(ami_root / "train" / SUMS),
        mining=mining,
        excluded=excluded,
        mined=mined,
    )
    return guard, excluded_sums


def _version(source: str, meta: Mapping[str, object]) -> str:
    synth = meta.get("synth")
    synth = synth if isinstance(synth, dict) else {}
    if source == "chatterbox" and "chatterbox" in synth:
        found = synth["chatterbox"]
        return (
            f"chatterbox-tts {found.get('package_version')}, ResembleAI/chatterbox at revision "
            f"{found.get('revision')}"
        )
    if source == "piper" and "piper" in synth:
        found = synth["piper"]
        return f"piper-sample-generator {found.get('generator')}, {found.get('per_text')} a text"
    if source == "mixed":
        return f"mine.py mixing, seed {meta.get('seed')}"
    return SOURCES[source].version


def provenance(
    windows: Sequence[Window], training: str, excluded: Mapping[str, str], meta: Mapping
) -> dict[str, object]:
    """`provenance.json`'s payload for the windows."""
    files: dict[tuple[str, str], FileRef] = {}
    for window in windows:
        for ref in window.refs():
            files.setdefault((ref.source, ref.sha256), ref)
    sources: dict[str, dict[str, object]] = {}
    for (source, _), ref in sorted(files.items()):
        entry = sources.setdefault(
            source,
            {
                "licence": SOURCES[source].licence,
                "version": _version(source, meta),
                "files": 0,
                "hours": 0.0,
            },
        )
        entry["files"] += 1
        entry["hours"] += ref.samples / SAMPLE_RATE / 3600
    return {
        "version": 1,
        "training": training,
        "file_sha256": sorted({digest for _, digest in files}),
        "sources": sources,
        "excluded": dict(excluded),
    }


def build(args: argparse.Namespace) -> int:
    """Check every mined window and write the index, its meta and the provenance."""
    if args.out.exists():
        raise GuardError("--out already exists; the dataset is written new")
    meta = _json(args.mined / MINED_META, f"mined {MINED_META}")
    windows = read_windows(args.mined / MINED_FILE)
    manifest = load_manifest(args.manifest, root_map=parse_root_map(args.root_map))
    training = subset_sha256(manifest, "train")
    if meta.get("manifest_sha256") != training:
        raise GuardError(
            f"the miner read a manifest with sha256 {meta.get('manifest_sha256')}, not this "
            f"manifest's train subset_sha256 {training}"
        )
    guard, excluded = _guard(manifest, args.ami_root, args.mined)
    for number, window in enumerate(windows, start=1):
        for ref in window.refs():
            guard.check(ref, f"{MINED_FILE} line {number}")
    payload = provenance(windows, training, excluded, meta)
    args.out.mkdir(parents=True)
    write_windows(args.out / INDEX_FILE, windows)
    roots = dict(meta.get("roots") or {})
    (args.out / INDEX_META).write_text(
        json.dumps(
            {"version": 1, "training": training, "roots": roots, "windows": len(windows)},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    (args.out / PROVENANCE_FILE).write_text(data, encoding="utf-8")
    counts = Counter((window.file.source, window.label) for window in windows)
    for (source, label), count in sorted(counts.items()):
        print(f"{source} {label}: {count}")
    print(f"provenance sha256 {sha256_file(args.out / PROVENANCE_FILE)}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.wake.verifier.build", description=__doc__
    )
    parser.add_argument("--mined", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--root-map", action="append", default=[])
    parser.add_argument("--ami-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the guard and the build; 0 built, 2 refused."""
    args = _parser().parse_args(argv)
    try:
        return build(args)
    except (GuardError, ManifestError, DatasetError) as exc:
        print(f"build: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
