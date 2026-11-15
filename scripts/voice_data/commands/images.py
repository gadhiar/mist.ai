"""`images build`: build stage images from the code at a commit."""

from __future__ import annotations

import argparse
from pathlib import Path

from .. import images
from ..jobs import HostDocker

# This module sits at `<checkout>/scripts/voice_data/commands/images.py`.
DEFAULT_CHECKOUT = Path(__file__).resolve().parents[3]


def register(subparsers: argparse._SubParsersAction) -> None:
    """Add the `images` command and its `build` action."""
    parser = subparsers.add_parser("images", help="build stage images from a committed tree")
    actions = parser.add_subparsers(dest="images_action", metavar="ACTION", required=True)
    build = actions.add_parser(
        "build",
        help="build the base image and a family's image (or all) at a commit",
        description="Export the tree at --sha with git archive and build from it, base first.",
    )
    build.add_argument("--family", required=True, help="a family in families.json, or all")
    build.add_argument("--sha", required=True, help="the commit to build from")
    build.add_argument(
        "--checkout", type=Path, default=DEFAULT_CHECKOUT, help="the git checkout (default: this)"
    )
    build.set_defaults(func=run_build)


def run_build(args: argparse.Namespace) -> int:
    """Build the images and print each tag."""
    for tag in images.build(args.checkout, args.sha, args.family, HostDocker()):
        print(f"built {tag}")
    return 0
