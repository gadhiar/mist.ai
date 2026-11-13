"""`doctor`: check what the pipeline needs on this host, and print no private value.

Env vars are reported by name as `set` or `unset` only: their values are keys and private
endpoints, and a doctor run is exactly the output someone pastes into a chat or an issue.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from .. import root as root_mod
from ..errors import RootError

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


def register(subparsers: argparse._SubParsersAction) -> None:
    """Add the `doctor` command."""
    parser = subparsers.add_parser(
        "doctor",
        help="check docker, the root, free disk, Python and env var names",
        description="Check this host; exit 2 when a check fails. Prints no env var value.",
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    """Print one line per check and return 2 when docker or the root fails."""
    failed = False

    docker = shutil.which("docker")
    print(f"docker: {'found' if docker else 'missing'}")
    failed |= docker is None

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
