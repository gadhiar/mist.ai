"""`init --kind private|rehearsal`: write the root's marker and layout."""

from __future__ import annotations

import argparse

from .. import root as root_mod


def register(subparsers: argparse._SubParsersAction) -> None:
    """Add the `init` command."""
    parser = subparsers.add_parser(
        "init",
        help="create the root's marker and layout",
        description=(
            "Create the marker and the missing layout. A non-empty unmarked directory is "
            "accepted only when it holds what may predate init; nothing found is read, moved "
            "or changed."
        ),
    )
    parser.add_argument("--kind", required=True, choices=root_mod.KINDS)
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    """Initialise the root named by `--root` or the environment."""
    path = root_mod.resolve_root_path(args.root)
    root = root_mod.init_root(path, args.kind)
    print(f"root: initialised (kind {root.kind}, {len(root_mod.LAYOUT)} layout directories)")
    return 0
