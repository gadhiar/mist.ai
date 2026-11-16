"""`doctor`: check what the pipeline needs on this host, and print no private value.

Env vars are reported by name as `set` or `unset` only: their values are keys and private
endpoints, and a doctor run is exactly the output someone pastes into a chat or an issue.

Local stage images are compared with HEAD: each image in `families.json` is looked up by HEAD's
first 12 hex, and a present one's `/etc/mist-voice-image.json` must name HEAD and the sha256 of
each requirements file as the checkout has it. Images are built after merge, so a missing or
stale one is a warning and never changes the exit code.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Protocol

from .. import root as root_mod
from ..errors import InputError, RootError
from ..jobs import load_families
from ..records import sha256_file

# This module sits at `<checkout>/scripts/voice_data/commands/doctor.py`.
CHECKOUT = Path(__file__).resolve().parents[3]
IMAGE_RECORD = "/etc/mist-voice-image.json"
# A record's file names come from an image, so only plain requirements names are looked up.
REQUIREMENT_NAME = re.compile(r"requirements-[A-Za-z0-9_-]+\.txt")
PROBE_TIMEOUT_S = 120

ENV_NAMES = (
    root_mod.ENV_ROOT,
    "HF_TOKEN",
    "ELEVENLABS_API_KEY",
    "MIST_VOICE_TEXT_LLM_URL",
    "MIST_VOICE_TEXT_LLM_MODEL",
    "MIST_VOICE_TEXT_LLM_KEY",
    "MIST_VOICE_REGISTRY",
)
MIN_PYTHON = (3, 11, 13)


def python_warning(version: tuple[int, ...]) -> str | None:
    """Return a warning when the host Python is below 3.11.13, else `None`.

    The host runs only standard-library code, so an older Python still works for most
    commands; it is warned rather than refused.
    """
    if tuple(version[:3]) >= MIN_PYTHON:
        return None
    shown = ".".join(str(part) for part in version[:3])
    wanted = ".".join(str(part) for part in MIN_PYTHON)
    return f"warning: host Python {shown} is below {wanted}"


class ImageProbe(Protocol):
    """The git and docker calls of the image check; tests pass a fake."""

    def head(self) -> str | None:
        """Return HEAD's full sha in the checkout, or `None` when git cannot name it."""
        ...

    def record(self, tag: str) -> str | None:
        """Return the image's record text, `None` when no image has the tag, `""` if unreadable."""
        ...


class HostProbe:
    """git and the docker CLI on this host, called with argv lists and never a shell."""

    def __init__(self, checkout: Path) -> None:
        self._checkout = checkout

    def _run(self, argv: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            argv, cwd=cwd, capture_output=True, text=True, timeout=PROBE_TIMEOUT_S, check=False
        )

    def head(self) -> str | None:
        """Return HEAD's sha, or `None` when git is absent, times out or finds no commit."""
        try:
            proc = self._run(["git", "rev-parse", "--verify", "--quiet", "HEAD"], self._checkout)
        except (OSError, subprocess.SubprocessError):
            return None
        sha = proc.stdout.strip()
        return sha if proc.returncode == 0 and sha else None

    def record(self, tag: str) -> str | None:
        """Read the record with `cat` as the entrypoint, offline, so only the file reaches stdout.

        The image's own entrypoint is bypassed because a CUDA base image's entrypoint can print a
        banner before the command runs.
        """
        try:
            if self._run(["docker", "image", "inspect", "--format", "{{.Id}}", tag]).returncode:
                return None
            proc = self._run(
                ["docker", "run", "--rm", "--network", "none", "--entrypoint", "cat", tag]
                + [IMAGE_RECORD]
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        return proc.stdout if proc.returncode == 0 else ""


def make_probe(checkout: Path) -> ImageProbe:
    """Return the host probe; tests replace this function."""
    return HostProbe(checkout)


def record_differences(text: str, head: str, requirements_dir: Path) -> list[str]:
    """Return what differs between an image's record and HEAD's checkout, empty when nothing.

    Only names are returned, never a value: a requirements file whose sha256 differs or that the
    checkout lacks, `git sha`, or `requirements` for a name that is not a plain file name.
    """
    try:
        record = json.loads(text)
    except ValueError:
        return ["no readable record"]
    if not isinstance(record, dict):
        return ["no readable record"]
    differs: list[str] = []
    if record.get("git_sha") != head:
        differs.append("git sha")
    requirements = record.get("requirements")
    if not isinstance(requirements, dict):
        return [*differs, "requirements"]
    for name, digest in sorted(requirements.items()):
        if not REQUIREMENT_NAME.fullmatch(name):
            differs.append("requirements")
            continue
        path = requirements_dir / name
        if not path.is_file() or sha256_file(path) != digest:
            differs.append(name)
    return list(dict.fromkeys(differs))


def image_lines(probe: ImageProbe, checkout: Path) -> list[str]:
    """Return one line per distinct image in `families.json`: ok, missing or stale."""
    head = probe.head()
    if head is None:
        return ["images: not checked (no git HEAD)"]
    try:
        families = load_families(checkout / "scripts" / "voice_data" / "families.json")
    except InputError:
        return ["images: not checked (no families.json)"]
    names = dict.fromkeys(f.get("image") for f in families.values() if f.get("image"))
    lines: list[str] = []
    for name in names:
        text = probe.record(f"{name}:{head[:12]}")
        if text is None:
            status = "missing"
        else:
            differs = record_differences(text, head, checkout / "docker" / "voice-data")
            status = f"stale ({', '.join(differs)})" if differs else "ok"
        lines.append(f"image {name}: {status}")
    return lines


def register(subparsers: argparse._SubParsersAction) -> None:
    """Add the `doctor` command."""
    parser = subparsers.add_parser(
        "doctor",
        help="check docker, local images, the root, free disk, Python and env var names",
        description=(
            "Check this host; exit 2 when docker or the root fails. Images missing or stale "
            "against HEAD are warnings. Prints no env var value."
        ),
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    """Print one line per check and return 2 when docker or the root fails."""
    failed = False

    docker = shutil.which("docker")
    print(f"docker: {'found' if docker else 'missing'}")
    failed |= docker is None
    if docker is None:
        print("images: not checked (docker missing)")
    else:
        for line in image_lines(make_probe(CHECKOUT), CHECKOUT):
            print(line)

    disk_path = Path.cwd()
    if not (args.root or os.environ.get(root_mod.ENV_ROOT)):
        print("root: unset")
        failed = True
    else:
        try:
            root = root_mod.open_root(args.root)
        except RootError as exc:
            print(f"root: refused ({str(exc).removeprefix('refused: ')})")
            failed = True
        else:
            print(f"root: marker ok (kind {root.kind})")
            disk_path = root.path

    free_gib = shutil.disk_usage(disk_path).free / 2**30
    print(f"disk free: {free_gib:.1f} GiB")

    print(f"python: {'.'.join(str(part) for part in sys.version_info[:3])}")
    warning = python_warning(tuple(sys.version_info[:3]))
    if warning is not None:
        print(warning)

    for name in ENV_NAMES:
        print(f"{name}: {'set' if os.environ.get(name) else 'unset'}")
    return 2 if failed else 0
