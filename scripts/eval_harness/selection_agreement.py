"""Compute the selection set's label agreement, the figures `agreement.json` records.

ADR-054's v14 amendment (point 7) reports agreement between two label passes twice: over all
relations, and over judgements, where the edges one relation rail adds for one team count once.
This module reads the selection set's two blind passes and computes both with
`gold_agreement.set_agreement`, after `judgement_relations.judgement_filter` for the second. It
was the driver that wrote the committed `agreement.json` before any relation was adjudicated,
kept here so that the file can be recomputed and checked against its inputs.

No row records the scope a turn was spoken in, which the rails read, so a probe is read as
user-scope when either pass labels the user and as third-party otherwise. Ids are compared as
the rails store them (`judgement_relations.stored_row`).

The pack and `run_rails` are the caller's, as in `judgement_relations`, so importing this module
loads neither; only `main` resolves the pack, from the version the recorded file names.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .gold_agreement import LabelRow, set_agreement
from .judgement_relations import RunRails, judgement_filter, stored_row

USER_SCOPE = "user-scope"
THIRD_PARTY = "third-party"
LIST_GROUP = "list"
PROSE_GROUP = "prose"


def read_rows(path: Path) -> list[LabelRow]:
    """The rows of a JSON Lines file, blank lines skipped."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line]


def group_of(tag: str) -> str:
    """A list probe's tag holds `-list-`; every other probe is prose."""
    return LIST_GROUP if "-list-" in tag else PROSE_GROUP


def _holds_user(row: LabelRow) -> bool:
    ids = {entity["id"] for entity in row["expected_entities"]}
    for edge in row["expected_relationships"]:
        ids |= {edge["source"], edge["target"]}
    return "user" in ids


def probe_scope(row_a: LabelRow, row_b: LabelRow) -> str:
    """User-scope when either pass labels the user, otherwise third-party."""
    return USER_SCOPE if _holds_user(row_a) or _holds_user(row_b) else THIRD_PARTY


def _check_rows(
    rows_a: Sequence[LabelRow], rows_b: Sequence[LabelRow], selection: Sequence[LabelRow]
) -> list[str]:
    tags = [row["tag"] for row in selection]
    by_a = {row["tag"]: row for row in rows_a}
    by_b = {row["tag"]: row for row in rows_b}
    if len(set(tags)) != len(tags):
        raise ValueError("the selection set repeats a tag")
    if not set(tags) == set(by_a) == set(by_b) or len(rows_a) != len(by_a):
        raise ValueError("the two passes and the selection set do not hold the same tags")
    if len(rows_b) != len(by_b):
        raise ValueError("pass B repeats a tag")
    if any(by_a[tag]["utterance"] != by_b[tag]["utterance"] for tag in tags):
        raise ValueError("the two passes hold different utterances for one tag")
    return tags


def selection_agreement(
    rows_a: Sequence[LabelRow],
    rows_b: Sequence[LabelRow],
    selection: Sequence[LabelRow],
    *,
    pack: object,
    run_rails: RunRails,
) -> dict:
    """Both agreement figures for the two passes, in the layout `agreement.json` keeps.

    Args:
        rows_a: Pass A's rows.
        rows_b: Pass B's rows, one per tag of pass A.
        selection: The selection set's rows, which fix the probes measured.
        pack: The pack whose rails store the ids and derive the team relations.
        run_rails: The pack's rail runner, `run_rails(row, pack, *, relation_rails)`.

    Returns:
        The report: the pack's rails hash and version, the groups, the judgement filter's
        counts, and the `set_agreement` report over all relations and over judgements.

    Raises:
        ValueError: The passes and the selection set do not hold the same probes.
    """
    tags = _check_rows(rows_a, rows_b, selection)
    by_a = {row["tag"]: row for row in rows_a}
    by_b = {row["tag"]: row for row in rows_b}
    scope = {tag: probe_scope(by_a[tag], by_b[tag]) for tag in tags}

    def id_function(row: LabelRow) -> LabelRow:
        return stored_row(row, pack=pack, run_rails=run_rails, scope=scope[row["tag"]])

    filtered = [
        judgement_filter(by_a[tag], by_b[tag], pack=pack, run_rails=run_rails, scope=scope[tag])
        for tag in sorted(tags)
    ]
    pair_table = pack.pair_table
    all_relations = set_agreement(
        rows_a, rows_b, pair_table=pair_table, id_function=id_function, group_of=group_of
    )
    judgements = set_agreement(
        [probe.row_a for probe in filtered],
        [probe.row_b for probe in filtered],
        pair_table=pair_table,
        id_function=id_function,
        group_of=group_of,
    )
    return {
        "pack_rails_hash": pack.pack_rails_hash,
        "pack_version": pack.version,
        "groups": {LIST_GROUP: "tag holds -list-", PROSE_GROUP: "every other tag"},
        "filter": {
            "removed_a": sum(probe.removed_a for probe in filtered),
            "removed_b": sum(probe.removed_b for probe in filtered),
            "groups": sum(probe.groups for probe in filtered),
            "ungrouped_derived": sum(probe.ungrouped for probe in filtered),
            "probes_touched": sum(1 for probe in filtered if probe.removed_a or probe.removed_b),
        },
        "all_relations": all_relations.to_dict(),
        "judgement_relations": judgements.to_dict(),
    }


def render(report: dict) -> str:
    """The report as `agreement.json` holds it: two-space indent and a final newline."""
    return json.dumps(report, indent=2) + "\n"


def _pack_module(version: str) -> str:
    return "backend.knowledge.ontologies.packs.v" + version.replace(".", "_")


def main(argv: Sequence[str] | None = None) -> None:
    """Recompute the agreement under the pack the recorded file names, and write it out.

    Raises:
        SystemExit: The resolved pack's rails hash is not the one the recorded file holds.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pass-a", type=Path, required=True)
    parser.add_argument("--pass-b", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--recorded", type=Path, required=True, help="the agreement.json to match")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from backend.knowledge.extraction.rails import run_rails
    from backend.knowledge.ontologies.pack import load_pack

    recorded = json.loads(args.recorded.read_text(encoding="utf-8"))
    pack = load_pack(_pack_module(recorded["pack_version"]))
    if pack.pack_rails_hash != recorded["pack_rails_hash"]:
        raise SystemExit("the pack's rails hash differs from the one the recorded file holds")
    report = selection_agreement(
        read_rows(args.pass_a),
        read_rows(args.pass_b),
        read_rows(args.selection),
        pack=pack,
        run_rails=run_rails,
    )
    args.out.write_text(render(report), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
