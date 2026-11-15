"""The `python -m scripts.voice_data` command line.

Commands are discovered from `scripts/voice_data/commands/`: each module defines
`register(subparsers)` and sets `func` on its parser, so a task adds a command by adding a module
and no two tasks edit this file. The package's refusals map to exit codes here, in one place.
"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
from collections.abc import Sequence

from . import commands
from .errors import BudgetError, GateError, InputError, JobError, RootError, VoiceDataError

EXIT_CODES: tuple[tuple[type[VoiceDataError], int], ...] = (
    (RootError, 2),
    (InputError, 2),
    (GateError, 3),
    (JobError, 4),
    (BudgetError, 5),
)


def command_names() -> list[str]:
    """Return the command module names, sorted, so `--help` lists them in a stable order."""
    return sorted(
        info.name
        for info in pkgutil.iter_modules(commands.__path__)
        if not info.name.startswith("_")
    )


def exit_code(exc: VoiceDataError) -> int:
    """Return the exit code for a refusal; an unmapped subclass counts as a refusal (2)."""
    for cls, code in EXIT_CODES:
        if isinstance(exc, cls):
            return code
    return 2


def build_parser() -> argparse.ArgumentParser:
    """Build the parser with every discovered command registered."""
    parser = argparse.ArgumentParser(
        prog="python -m scripts.voice_data",
        description="Voice-data pipeline over a private root outside every git work tree.",
    )
    parser.add_argument(
        "--root",
        help="the private root (default: the MIST_VOICE_PRIVATE_ROOT environment variable)",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    for name in command_names():
        module = importlib.import_module(f"{commands.__name__}.{name}")
        module.register(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command and return its exit code.

    A refusal prints `error: <reason>` on stderr; reasons carry no private values (output
    discipline), so they are safe to print as they are.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help(sys.stderr)
        return 2
    try:
        return int(func(args))
    except VoiceDataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exit_code(exc)
