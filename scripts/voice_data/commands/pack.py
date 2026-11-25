"""`pack {labels,clean-ab,text,bakeoff,spotcheck,abx,voice-ab}` and `pack import`: blind pages.

Output names pack ids, counts and root-relative paths only: never a transcript, a script line, a
teacher's or student's label mapping, or a source file name.
"""

from __future__ import annotations

import argparse
import secrets
from pathlib import Path

from .. import packs
from ..root import open_root

# Each kind's own options: (flag, destination, help).
_OPTIONS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "labels": (("--run", "run", "the extraction run whose kept segments are labelled"),),
    "clean-ab": (
        ("--before", "before", "the run before cleaning"),
        ("--after", "after", "the run after cleaning"),
    ),
    "text": (("--version", "version", "the script version under text/"),),
    "bakeoff": (("--run", "run", "the bake-off run under teachers/bakeoff/"),),
    "spotcheck": (("--takes", "takes", "the batch's takes.jsonl, relative to the root"),),
    "abx": (("--manifest", "manifest", "the ABX manifest, relative to the root"),),
    "voice-ab": (("--run", "run", "the voice_ab.py bench run under bench/voice-ab/"),),
}

_HELP = {
    "labels": (
        "one clip per kept segment no earlier run labelled: who is audible, a keep range, "
        "energy, mood, tags and a note"
    ),
    "clean-ab": "before and after cleaning, in a seeded A/B order",
    "text": "script lines per state as text, each accepted or rejected",
    "bakeoff": "per state and line, every teacher's take, rated",
    "spotcheck": "random selected takes per state of one batch, each accepted or rejected",
    "abx": "seed, teacher and student ABX trials, plus ratings",
    "voice-ab": "the ear test: per line, two identities as A and B in a seeded order",
}


def register(subparsers: argparse._SubParsersAction) -> None:
    """Add the `pack` command and its kinds."""
    parser = subparsers.add_parser(
        "pack",
        help="blind listening pages for Raj's ratings, and their import",
        description=(
            "Build a blind page under packs/<pack_id>/ with its key sealed in "
            "packs/<pack_id>.key/, or import its ratings.csv, which unseals the key and writes "
            "the decisions."
        ),
    )
    kinds = parser.add_subparsers(dest="pack_kind", metavar="KIND", required=True)
    for kind in packs.KINDS:
        sub = kinds.add_parser(kind, help=_HELP[kind])
        for flag, dest, text in _OPTIONS[kind]:
            sub.add_argument(flag, dest=dest, required=True, help=text)
        sub.add_argument(
            "--seed",
            type=int,
            help="seed of the labels and order (default: fresh, recorded only in the key)",
        )
        if kind == "labels":
            sub.add_argument(
                "--replace",
                action="store_true",
                help="label a run again although seed/labels/<run>/ exists",
            )
        sub.set_defaults(func=run_build, replace=False)
    imp = kinds.add_parser("import", help="read a complete ratings.csv and write the decisions")
    imp.add_argument("--pack", required=True, help="the pack id")
    imp.add_argument("--ratings", required=True, type=Path, help="the exported ratings.csv")
    imp.add_argument(
        "--replace",
        action="store_true",
        help="labels packs only: replace the run's existing seed/labels/<run>/ files",
    )
    imp.set_defaults(func=run_import)


def _clips(n: int) -> str:
    return f"{n} clip" if n == 1 else f"{n} clips"


def run_build(args: argparse.Namespace) -> int:
    """Build one pack."""
    root = open_root(args.root)
    seed = args.seed if args.seed is not None else secrets.randbits(32)
    options = {dest: getattr(args, dest) for _, dest, _ in _OPTIONS[args.pack_kind]}
    built = packs.build_pack(root, args.pack_kind, options, seed, replace=args.replace)
    if built.pack_id is None:
        print(
            f"pack labels: every kept clip of run {options['run']} carries an earlier label "
            f"({_clips(built.carried)}), so there is no page to rate"
        )
        return 0
    carried = f", {_clips(built.carried)} carried from earlier runs" if built.carried else ""
    print(
        f"pack {built.kind}: {built.pack_id}, {built.items} items, {built.clips} clips"
        f"{carried}; open {packs.PACKS_DIR}/{built.pack_id}/{packs.PAGE_FILE}; the key stays "
        f"sealed in {packs.PACKS_DIR}/{built.pack_id}.key/ until pack import"
    )
    return 0


def run_import(args: argparse.Namespace) -> int:
    """Import one pack's ratings."""
    root = open_root(args.root)
    packs.import_pack(root, args.pack, args.ratings, replace=args.replace)
    return 0
