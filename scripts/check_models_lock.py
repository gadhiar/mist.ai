#!/usr/bin/env python3
"""Check `docker/backend/models.lock` and, in the image build, fetch its pinned models.

The memory read path loads its reranker from a local path with `local_files_only=True`, and
the offline recall eval loads its embedder the same way, so the backend image bakes every
model the lock lists into `/opt/models/<name>`. A model is pinned by its Hugging Face commit
sha: a branch name such as `main` moves, and a rebuild would then bake different weights
under the same name.

Usage:
    python scripts/check_models_lock.py                 # structure and licenses
    python scripts/check_models_lock.py --strict        # also: every revision is a 40-hex sha
    python scripts/check_models_lock.py --download NAME LOCK   # image build only: fetch NAME

`--download` applies the strict rule to the entry it fetches, so it never downloads an
unpinned revision. Only the download path imports `huggingface_hub`; the checks use the
standard library alone.

Exit codes:
    0 - the lock passes (and the download, if asked for, succeeded)
    1 - the lock fails a check, or the download failed its file check
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

DEFAULT_LOCK = Path(__file__).resolve().parent.parent / "docker" / "backend" / "models.lock"
MODELS_ROOT = "/opt/models"

REVISION_RE = re.compile(r"[0-9a-f]{40}")
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
REPO_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
FIELDS = ("name", "repo", "revision", "license", "target")

# MIT-compatible licenses a baked model may carry (CLAUDE.md, External code), with the wording
# NOTICE uses for each. A model under any other license needs Raj's decision first.
LICENSE_NAMES = {
    "Apache-2.0": "Apache License 2.0",
    "MIT": "MIT License",
    "BSD-2-Clause": "BSD 2-Clause License",
    "BSD-3-Clause": "BSD 3-Clause License",
}

# What loading needs: the configs (including the sentence-transformers module files such as
# `modules.json` and `1_Pooling/config.json`), the tokenizer and the safetensors weights. The
# model card and any license file ride along for license compliance. A file no allow pattern
# matches is never fetched. huggingface_hub matches with `fnmatch`, where `*` also crosses
# `/`, so `*.json` alone would reach into the ONNX and OpenVINO trees; the ignore list removes
# those trees and the other weight formats, and ignore wins over allow.
ALLOW_PATTERNS = (
    "*.json",
    "vocab.txt",
    "*.safetensors",
    "README.md",
    "LICENSE*",
)
IGNORE_PATTERNS = (
    "onnx/*",
    "openvino/*",
    "*.onnx",
    "*.bin",
    "*.h5",
    "*.msgpack",
    "*.pt",
    "*.xml",
)
# Files every baked model must have after the fetch, or the build fails here instead of the
# read path failing at its first turn.
REQUIRED_FILES = ("config.json", "tokenizer_config.json", "model.safetensors")


class LockError(Exception):
    """The lock is malformed, or a fetched model is missing a file loading needs."""


@dataclass(frozen=True, slots=True)
class LockEntry:
    """One `[[model]]` table of the lock."""

    name: str
    repo: str
    revision: str
    license: str
    target: str


def parse_lock(path: Path) -> list[LockEntry]:
    """Read the lock's `[[model]]` tables; raise `LockError` when one lacks or adds a field."""
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise LockError(f"{path}: {exc}") from exc
    tables = data.get("model")
    if not isinstance(tables, list) or not tables:
        raise LockError(f"{path}: no [[model]] entries")
    entries = []
    for index, table in enumerate(tables):
        missing = [field for field in FIELDS if not isinstance(table.get(field), str)]
        extra = sorted(set(table) - set(FIELDS))
        if missing or extra:
            raise LockError(
                f"{path}: entry {index}: missing or non-string {missing}, unknown {extra}"
            )
        entries.append(LockEntry(**{field: table[field] for field in FIELDS}))
    return entries


def check_entry(entry: LockEntry, *, strict: bool) -> list[str]:
    """Problems with one entry; the revision rule applies only under `strict`."""
    problems = []
    label = entry.name or "<unnamed>"
    if not NAME_RE.fullmatch(entry.name):
        problems.append(f"{label}: name {entry.name!r} is not a plain directory name")
    if not REPO_RE.fullmatch(entry.repo):
        problems.append(f"{label}: repo {entry.repo!r} is not `owner/name`")
    if entry.target != f"{MODELS_ROOT}/{entry.name}":
        problems.append(f"{label}: target {entry.target!r} is not {MODELS_ROOT}/{entry.name}")
    if entry.license not in LICENSE_NAMES:
        problems.append(f"{label}: license {entry.license!r} is not one of {sorted(LICENSE_NAMES)}")
    if strict and not REVISION_RE.fullmatch(entry.revision):
        problems.append(
            f"{label}: revision {entry.revision!r} is not a 40-character lowercase hex sha"
        )
    return problems


def check_lock(entries: list[LockEntry], *, strict: bool) -> list[str]:
    """Problems with the whole lock: each entry's, plus duplicate names or targets."""
    problems = []
    for entry in entries:
        problems.extend(check_entry(entry, strict=strict))
    for field in ("name", "target"):
        values = [getattr(entry, field) for entry in entries]
        for value in sorted({v for v in values if values.count(v) > 1}):
            problems.append(f"duplicate {field} {value!r}")
    return problems


def download_entry(entry: LockEntry, *, fetch: Callable[..., str]) -> None:
    """Fetch one entry at its pinned revision into its target, then check the files.

    Args:
        entry: The lock entry to fetch.
        fetch: `huggingface_hub.snapshot_download` in the build; tests pass a fake.

    Raises:
        LockError: The revision is not a 40-hex sha, or a required file is missing after
            the fetch.
    """
    # `main` has already checked the whole lock; this guard keeps any other caller from
    # fetching a moving branch.
    if not REVISION_RE.fullmatch(entry.revision):
        raise LockError(
            f"{entry.name}: revision {entry.revision!r} is not a 40-character lowercase hex sha"
        )
    target = Path(entry.target)
    fetch(
        repo_id=entry.repo,
        revision=entry.revision,
        local_dir=str(target),
        allow_patterns=list(ALLOW_PATTERNS),
        ignore_patterns=list(IGNORE_PATTERNS),
    )
    # huggingface_hub keeps download metadata under `<local_dir>/.cache`; the image needs
    # only the model files.
    shutil.rmtree(target / ".cache", ignore_errors=True)
    missing = [name for name in REQUIRED_FILES if not (target / name).is_file()]
    if missing:
        raise LockError(f"{entry.name}: {entry.repo}@{entry.revision} lacks {missing}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("lock", nargs="?", type=Path, default=DEFAULT_LOCK)
    parser.add_argument(
        "--strict", action="store_true", help="require a 40-hex sha for every revision"
    )
    parser.add_argument("--download", metavar="NAME", help="fetch the named entry (image build)")
    args = parser.parse_args(argv)

    try:
        entries = parse_lock(args.lock)
    except LockError as exc:
        print(f"models.lock: {exc}", file=sys.stderr)
        return 1
    problems = check_lock(entries, strict=args.strict)
    if problems:
        for problem in problems:
            print(f"models.lock: {problem}", file=sys.stderr)
        return 1
    if args.download is None:
        print(f"models.lock: {len(entries)} entries pass ({'strict' if args.strict else 'basic'})")
        return 0

    by_name = {entry.name: entry for entry in entries}
    if args.download not in by_name:
        print(f"models.lock: no entry named {args.download!r}", file=sys.stderr)
        return 1
    from huggingface_hub import snapshot_download

    try:
        download_entry(by_name[args.download], fetch=snapshot_download)
    except LockError as exc:
        print(f"models.lock: {exc}", file=sys.stderr)
        return 1
    print(f"models.lock: fetched {args.download} into {by_name[args.download].target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
