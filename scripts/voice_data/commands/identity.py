"""`identity export --target T --voice-id V` and `identity check --voice-id V` (v6's format).

`export` writes a seed target's identity candidate, status `placeholder`, from each state's best
reference by leave-one-out SIM, so the `voice_id` every later package records names an identity
v6 can load; freezing it stays v6's step. `check` applies v6's rules to `voices/<V>/`. Output
names ids, states and the status; a transcript is never printed.
"""

from __future__ import annotations

import argparse

from .. import identity as identity_mod
from .. import target as target_mod
from ..errors import InputError
from ..root import open_root


def register(subparsers: argparse._SubParsersAction) -> None:
    """Add the `identity` command and its `export` and `check` actions."""
    parser = subparsers.add_parser(
        "identity",
        help="write a seed target's placeholder identity, or check one, in v6's format",
        description="Identities under voices/<voice_id>/ in goal mist-realtime-voice's format.",
    )
    actions = parser.add_subparsers(dest="identity_action", metavar="ACTION", required=True)
    export = actions.add_parser(
        "export",
        help="write voices/<V>/ (status placeholder) from a seed target's best references",
    )
    export.add_argument("--target", required=True, help="a target built in seed mode")
    export.add_argument("--voice-id", required=True, help="the target's voice id")
    export.set_defaults(func=run_export)
    check = actions.add_parser("check", help="apply v6's rules to voices/<V>/")
    check.add_argument("--voice-id", required=True, help="the identity to check")
    check.set_defaults(func=run_check)


def run_export(args: argparse.Namespace) -> int:
    """Export a seed target's references as its placeholder identity, then check it."""
    root = open_root(args.root)
    identity_mod.check_voice_id(args.voice_id)
    record = target_mod.read_target(root, args.target)
    if record.mode != target_mod.SEED:
        raise InputError(
            f"refused: target {record.target_id} is in mode {record.mode}, not seed; its voice "
            f"is the identity {record.voice_id}"
        )
    if args.voice_id != record.voice_id:
        raise InputError(
            f"refused: target {record.target_id}'s voice id is {record.voice_id}; export it "
            "under that voice id"
        )
    found = identity_mod.export(root, args.voice_id, target_mod.export_clips(root, record))
    print(
        f"identity {found.voice_id}: wrote {identity_mod.voice_rel(found.voice_id)}, status "
        f"{found.status}, states {', '.join(c.state for c in found.clips)}"
    )
    return 0


def run_check(args: argparse.Namespace) -> int:
    """Check `voices/<V>/` against v6's rules."""
    root = open_root(args.root)
    found = identity_mod.check_in_root(root, args.voice_id)
    print(
        f"identity {found.voice_id}: ok, status {found.status}, states "
        f"{', '.join(c.state for c in found.clips)}"
    )
    return 0
