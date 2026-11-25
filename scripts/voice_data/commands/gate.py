"""`gate --run E [--clean C] | --check-record`: the seed gate on a run's report; exit 3 on a fail.

`--run E` evaluates the gate on extract run E's `seed/runs/<E>/report.json`. With `--clean C`,
a finished clean run whose plan names E as `from`, the gate reads the labelled, cleaned seed:
`clean_minutes`, `minutes_per_state` and `one_speaker.kept` come from C's report, `one_speaker`
gains the four labelled removal counts (reported, not gated), and `flagged_unconfirmed` becomes
the flagged-cluster segments kept in C that no label covers. The source facts, thresholds,
purity, minutes per channel and the retrieval and window counts stay E's. `merge_clean` makes
the merged report, so `seed_gate.evaluate` reads one report either way. Extract writes the report's
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
from collections.abc import Mapping

from .. import gates as gates_mod
from .. import labels as labels_mod
from .. import root as root_mod
from .. import seed_gate
from ..errors import GateError, InputError
from ..records import SegmentRecord, read_json, read_jsonl
from . import clean as clean_cmd

# A run id names a folder under `seed/runs/`, so it is one plain path segment.
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
FLAGGED = "flagged"
LABELLED_COUNTS = dict(zip(labels_mod.REMOVED, seed_gate.LABELLED_COUNTS, strict=True))
"""Clean's `removed_by_label` reason to the `one_speaker` key the gate reports it under."""


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
    parser.add_argument(
        "--clean",
        help="with --run: the clean run made from it, whose labelled minutes and counts the "
        "gate reads",
    )
    parser.set_defaults(func=run)


def _clean_doc(root: root_mod.Root, clean: str, name: str) -> dict[str, object]:
    rel = f"{clean_cmd.run_rel(clean)}/{name}"
    doc = read_json(root.path / rel)
    if not isinstance(doc, dict):
        raise InputError(f"refused: {rel} is not a JSON object")
    return doc


def merge_clean(
    root: root_mod.Root, run: str, clean: str, report: Mapping[str, object]
) -> dict[str, object]:
    """Return extract run `run`'s report with clean run `clean`'s labelled values in it.

    Raises:
        InputError: `clean` is not a finished clean run, its plan's `from` is not `run`, or
            its report lacks a value the gate reads.
    """
    if not RUN_ID.fullmatch(clean) or ".." in clean:
        raise InputError("refused: a clean run id is one folder name under seed/runs")
    names = (clean_cmd.PLAN_NAME, clean_cmd.REPORT_NAME, clean_cmd.SEGMENTS_NAME)
    if not all((root.path / clean_cmd.run_rel(clean) / name).is_file() for name in names):
        raise InputError(f"refused: run {clean} is not a finished clean run")
    plan = _clean_doc(root, clean, clean_cmd.PLAN_NAME)
    if plan.get("from") != run:
        raise InputError(f"refused: clean run {clean} was not made from run {run}")
    cleaned = _clean_doc(root, clean, clean_cmd.REPORT_NAME)
    counts = cleaned.get("segments")
    removed = counts.get("removed_by_label") if isinstance(counts, Mapping) else None
    if not isinstance(counts, Mapping) or not isinstance(removed, Mapping):
        raise InputError(f"refused: clean run {clean}'s report has no label counts")
    labels = plan.get("labels")
    labelled = set(labels) if isinstance(labels, Mapping) else set()
    rows = read_jsonl(root.path / clean_cmd.run_rel(clean) / clean_cmd.SEGMENTS_NAME, SegmentRecord)
    unconfirmed = sum(
        1 for r in rows if r.kept and r.cluster == FLAGGED and r.segment_id not in labelled
    )
    speaker = report.get("one_speaker")
    return {
        **report,
        "clean_minutes": cleaned.get("clean_minutes"),
        "minutes_per_state": cleaned.get("minutes_per_state"),
        "one_speaker": {
            **(speaker if isinstance(speaker, Mapping) else {}),
            "kept": counts.get("kept"),
            "flagged_unconfirmed": unconfirmed,
            **{key: removed.get(reason) for reason, key in LABELLED_COUNTS.items()},
        },
    }


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
        if args.clean is not None:
            raise InputError("refused: --clean goes with --run, not --check-record")
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
    if args.clean is not None:
        report = merge_clean(root, args.run, args.clean, report)
    record = read_record(root) if report.get("rehearsal") is None else None
    if record is not None:
        rehearsal = {"precision": record["precision"], "thresholds": record["thresholds"]}
        report = {**report, "rehearsal": rehearsal}
    result = seed_gate.evaluate(report, gates_mod.load(root.path).effective())
    print(f"seed gate, run {args.run}:")
    if args.clean is not None:
        print(f"  clean run {args.clean}: labelled minutes, states and kept count")
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
