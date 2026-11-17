"""`gate --run R | --check-record`: the seed gate on a run's `report.json`; exit 3 on a fail.

`--run R` evaluates the gate on `seed/runs/<R>/report.json`. Extract writes the report's
`rehearsal` as null, since the rehearsal runs in another root, so when it is null and the root
holds the rehearsal's precision record (`config/rehearsal.json`, A24) the gate evaluates with the
record's precision and thresholds instead. A report's own `rehearsal` object wins over the record.

`--check-record` compares the record's thresholds with the root's effective gate values, with no
run: the gate's own comparison happens only after H8's jobs, and a stale record should stop H8
before it spends GPU time. Exit 0 when they match, 3 naming the keys that differ, 2 when the
record is missing or malformed.

The gate writes nothing: the run's report is already written, and the gate prints criteria,
names and numbers only, with sources named by id.
"""

from __future__ import annotations

import argparse
import re

from .. import gates as gates_mod
from .. import root as root_mod
from .. import seed_gate
from ..errors import GateError, InputError
from ..records import read_json

# A run id names a folder under `seed/runs/`, so it is one plain path segment.
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def register(subparsers: argparse._SubParsersAction) -> None:
    """Add the `gate` command."""
    parser = subparsers.add_parser(
        "gate",
        help="the seed gate on a run's report; exit 3 on fail",
        description=(
            "Evaluate the seed gate on seed/runs/<run>/report.json, or check the rehearsal's "
            f"precision record ({seed_gate.REHEARSAL_RECORD_REL}) against the gates; exit 3 on "
            "a fail."
        ),
    )
    # argparse refuses a required argument inside a mutually exclusive group, so the group is the
    # required one: exactly one of the two flags.
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--run", help="the run id under seed/runs")
    which.add_argument(
        "--check-record",
        action="store_true",
        help=f"compare {seed_gate.REHEARSAL_RECORD_REL}'s thresholds with the gates; exit 3 on a "
        "difference",
    )
    parser.set_defaults(func=run)


def read_record(root: root_mod.Root) -> dict[str, object] | None:
    """Return the root's checked precision record, or `None` when the root has none.

    Raises:
        InputError: The file is not JSON or is malformed (`seed_gate.read_rehearsal_record`).
    """
    rel = seed_gate.REHEARSAL_RECORD_REL
    path = root.path / rel
    if not path.is_file():
        return None
    try:
        raw = read_json(path)
    except InputError as exc:
        raise InputError(f"refused: {rel} is not readable JSON") from exc
    return seed_gate.read_rehearsal_record(raw)


def check_record(root: root_mod.Root) -> int:
    """Compare the record's thresholds with the root's effective gate values.

    Raises:
        InputError: The record is missing or malformed.
        GateError: A key differs or is missing on either side.
    """
    record = read_record(root)
    if record is None:
        raise InputError(f"refused: {seed_gate.REHEARSAL_RECORD_REL} is missing")
    differ = seed_gate.record_differences(record, gates_mod.load(root.path).effective())
    if differ:
        print(f"rehearsal record differs: {', '.join(differ)}")
        raise GateError(f"{seed_gate.REHEARSAL_RECORD_REL} is from other gate values")
    print(f"rehearsal record matches the gates (rehearsal run {record['run']})")
    return 0


def run(args: argparse.Namespace) -> int:
    """Print the gate's criteria and verdict; raise `GateError` on a fail."""
    root = root_mod.open_root(args.root)
    if args.check_record:
        return check_record(root)
    if not RUN_ID.fullmatch(args.run) or ".." in args.run:
        raise InputError("refused: a run id is one folder name under seed/runs")
    rel = f"seed/runs/{args.run}/report.json"
    path = root.path / rel
    if not path.is_file():
        raise InputError(f"refused: {rel} is missing")
    report = read_json(path)
    if not isinstance(report, dict):
        raise InputError(f"refused: {rel} is not a JSON object")
    record = read_record(root) if report.get("rehearsal") is None else None
    if record is not None:
        rehearsal = {"precision": record["precision"], "thresholds": record["thresholds"]}
        report = {**report, "rehearsal": rehearsal}
    result = seed_gate.evaluate(report, gates_mod.load(root.path).effective())
    print(f"seed gate, run {args.run}:")
    if record is not None:
        print(
            f"  rehearsal record: {seed_gate.REHEARSAL_RECORD_REL} "
            f"(rehearsal run {record['run']})"
        )
    for line in seed_gate.source_lines(report):
        print(f"  {line}")
    for line in seed_gate.format_lines(result):
        print(line)
    if not result.passed:
        raise GateError(f"the seed gate failed for run {args.run}")
    return 0
