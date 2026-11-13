"""`gate --run R`: evaluate the seed gate on a run's `report.json`; exit 3 on a fail.

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
        description="Evaluate the seed gate on seed/runs/<run>/report.json; exit 3 on a fail.",
    )
    parser.add_argument("--run", required=True, help="the run id under seed/runs")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    """Print the gate's criteria and verdict; raise `GateError` on a fail."""
    root = root_mod.open_root(args.root)
    if not RUN_ID.fullmatch(args.run) or ".." in args.run:
        raise InputError("refused: a run id is one folder name under seed/runs")
    rel = f"seed/runs/{args.run}/report.json"
    path = root.path / rel
    if not path.is_file():
        raise InputError(f"refused: {rel} is missing")
    report = read_json(path)
    if not isinstance(report, dict):
        raise InputError(f"refused: {rel} is not a JSON object")
    result = seed_gate.evaluate(report, gates_mod.load(root.path).effective())
    print(f"seed gate, run {args.run}:")
    for line in seed_gate.source_lines(report):
        print(f"  {line}")
    for line in seed_gate.format_lines(result):
        print(line)
    if not result.passed:
        raise GateError(f"the seed gate failed for run {args.run}")
    return 0
