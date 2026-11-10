"""The list miss classifier, on hand-built gold and raw files.

Every gold and raw file here is written under `tmp_path` with invented names, in the shapes
`score_extraction_run.iter_gold_probes` and the extraction-service gauntlet read and write. No
gold file in the repository and no raw run file is opened.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypedDict

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.eval_harness import diagnose_list_misses as dlm  # noqa: E402

PROSE = "I talk about tools and people."
LEAD_IN = "Tools at Acme Corp:\n- Python\n- Rust"

Rel = tuple[str, str, str, str, str]


class GoldRow(TypedDict):
    """One gold probe line, in the shape `iter_gold_probes` reads."""

    tag: str
    utterance: str
    expected_entities: list[dict[str, str]]
    expected_relationships: list[dict[str, str]]


def _probe(tag: str, entities: dict[str, str], rels: list[Rel], utterance: str = PROSE) -> GoldRow:
    return {
        "tag": tag,
        "utterance": utterance,
        "expected_entities": [{"id": i, "type": t} for i, t in entities.items()],
        "expected_relationships": [
            {"source": s, "source_type": st, "predicate": p, "target": t, "target_type": tt}
            for s, st, p, t, tt in rels
        ],
    }


def _record(tag: str, entities: dict[str, str], edges: list[tuple[str, str, str]]) -> dlm.RawRecord:
    return {
        "id": tag,
        "response": {
            "payload": {
                "entities": [{"id": i, "type": t} for i, t in entities.items()],
                "relationships": [{"source": s, "type": p, "target": t} for s, p, t in edges],
            }
        },
    }


def _write(path: Path, rows: Sequence[Mapping[str, object]]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _run(
    tmp_path: Path,
    probes: list[GoldRow],
    records: list[dlm.RawRecord],
    capsys: pytest.CaptureFixture[str],
) -> tuple[dict[tuple[str, str], tuple[str, str]], str]:
    """Run the CLI on one gold and raw pair; map (tag, gold key) to (class, detail and marks)."""
    gold = _write(tmp_path / "gold.jsonl", probes)
    raw = _write(tmp_path / "raw.jsonl", records)
    code = dlm.main(["--gold", str(gold), "--pre-h8-gold", str(gold), f"--run=r={raw}", "--detail"])
    assert code == 0
    out = capsys.readouterr().out
    misses = {}
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) == 4 and parts[2] in dlm.CLASSES:
            misses[(parts[0], parts[1])] = (parts[2], parts[3])
    return misses, out


RUST = {"user": "User", "rust": "Technology"}

# (class, gold probe, raw record or None, detail and marks the miss prints)
CLASS_CASES = [
    (
        "EMPTY",
        _probe("p", RUST, [("user", "User", "USES", "rust", "Technology")]),
        _record("p", {}, []),
        "",
    ),
    (
        "NOT_RUN",
        _probe("p", RUST, [("user", "User", "USES", "rust", "Technology")]),
        None,
        "",
    ),
    (
        "ID_VARIANT",
        _probe(
            "p",
            {"user": "User", "acme-corp": "Organization"},
            [("user", "User", "WORKS_AT", "acme-corp", "Organization")],
        ),
        _record(
            "p",
            {"user": "User", "The Acme Corp": "Organization"},
            [("user", "WORKS_AT", "the-acme-corp")],
        ),
        "n1",
    ),
    (
        "ID_LOOSE",
        _probe(
            "p",
            {"user": "User", "acme-corp": "Organization"},
            [("user", "User", "WORKS_AT", "acme-corp", "Organization")],
        ),
        _record(
            "p",
            {"user": "User", "acme-labs": "Organization"},
            [("user", "WORKS_AT", "acme-labs")],
        ),
        "acme-corp->acme-labs overlap",
    ),
    (
        "DIRECTION",
        _probe(
            "p",
            {"user": "User", "alice": "Person"},
            [("user", "User", "KNOWS", "alice", "Person")],
        ),
        _record("p", {"user": "User", "alice": "Person"}, [("alice", "KNOWS", "user")]),
        "",
    ),
    (
        "PREDICATE",
        _probe("p", RUST, [("user", "User", "USES", "rust", "Technology")]),
        _record("p", RUST, [("user", "PREFERS", "rust")]),
        "PREFERS",
    ),
    (
        "MERGED",
        _probe(
            "p",
            {"user": "User", "alice": "Person", "bob": "Person"},
            [("user", "User", "KNOWS", "alice", "Person")],
        ),
        _record("p", {"user": "User", "alice-and-bob": "Person"}, []),
        "alice-and-bob",
    ),
    (
        "PARENT_ABSENT",
        _probe(
            "p",
            {"acme-corp": "Organization", "python": "Technology"},
            [("acme-corp", "Organization", "USES", "python", "Technology")],
            LEAD_IN,
        ),
        _record("p", {"python": "Technology"}, []),
        " [frame]",
    ),
    (
        "USER_ABSENT",
        _probe(
            "p",
            {"user": "User", "alice": "Person"},
            [("user", "User", "KNOWS", "alice", "Person")],
        ),
        _record("p", {"alice": "Person"}, []),
        "",
    ),
    (
        "ENTITY_ABSENT",
        _probe("p", RUST, [("user", "User", "USES", "rust", "Technology")]),
        _record("p", {"user": "User"}, []),
        "rust",
    ),
    (
        "WRONG_ENDPOINT",
        _probe("p", RUST, [("user", "User", "USES", "rust", "Technology")]),
        _record("p", {**RUST, "golang": "Technology"}, [("user", "USES", "golang")]),
        "target golang",
    ),
    (
        "PARENT_LINK_OMITTED",
        _probe(
            "p",
            {"acme-corp": "Organization", "python": "Technology"},
            [("acme-corp", "Organization", "USES", "python", "Technology")],
            LEAD_IN,
        ),
        _record("p", {"acme-corp": "Organization", "python": "Technology"}, []),
        " [frame]",
    ),
    (
        "OMITTED",
        _probe("p", RUST, [("user", "User", "USES", "rust", "Technology")]),
        _record("p", RUST, []),
        "",
    ),
]


@pytest.mark.parametrize(("cls", "probe", "record", "detail"), CLASS_CASES)
def test_each_class_is_reached(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    cls: str,
    probe: GoldRow,
    record: dlm.RawRecord | None,
    detail: str,
) -> None:
    misses, out = _run(tmp_path, [probe], [record] if record else [], capsys)
    rel = probe["expected_relationships"][0]
    key = f"{rel['source']} {rel['predicate']} {rel['target']}"
    assert misses == {("p", key): (cls, detail)}
    assert f"| {cls} | 1 |" in out


def test_every_class_has_a_case() -> None:
    assert sorted(c[0] for c in CLASS_CASES) == sorted(dlm.CLASSES)


FAN_OUT_SOURCE = _probe(
    "fan",
    {"user": "User", "python": "Technology", "rust": "Technology", "javascript": "Technology"},
    [("user", "User", "USES", t, "Technology") for t in ("python", "rust", "javascript")],
)


def test_a_sibling_holding_the_shared_source_and_predicate_is_not_a_wrong_endpoint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # `js` is the model's id for the `javascript` item: it matches no gold entity, so the edge
    # `user USES js` is the sibling item's, not a wrong target for `python` or `rust`.
    record = _record(
        "fan",
        {"user": "User", "python": "Technology", "rust": "Technology", "js": "Technology"},
        [("user", "USES", "js")],
    )
    misses, _ = _run(tmp_path, [FAN_OUT_SOURCE], [record], capsys)
    assert misses == {
        ("fan", "user USES python"): ("OMITTED", " [fan-out]"),
        ("fan", "user USES rust"): ("OMITTED", " [fan-out]"),
        ("fan", "user USES javascript"): ("ENTITY_ABSENT", "javascript [fan-out]"),
    }


def test_a_sibling_holding_the_shared_predicate_and_target_is_not_a_wrong_endpoint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    people = ("alice", "bob", "carol")
    probe = _probe(
        "fan",
        {**{p: "Person" for p in people}, "acme": "Organization"},
        [(p, "Person", "WORKS_AT", "acme", "Organization") for p in people],
    )
    record = _record(
        "fan",
        {**{p: "Person" for p in people}, "dave": "Person", "acme": "Organization"},
        [("dave", "WORKS_AT", "acme")],
    )
    misses, _ = _run(tmp_path, [probe], [record], capsys)
    assert {m for m, _ in misses.values()} == {"OMITTED"}
    assert len(misses) == 3


def test_a_fan_out_item_linked_to_an_unknown_source_is_still_a_wrong_endpoint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The fix covers only the endpoint the fan-out shares; the item's own end still counts.
    record = _record(
        "fan",
        {
            **{e["id"]: e["type"] for e in FAN_OUT_SOURCE["expected_entities"]},
            "helper-bot": "Person",
        },
        [("helper-bot", "USES", "python")],
    )
    misses, _ = _run(tmp_path, [FAN_OUT_SOURCE], [record], capsys)
    assert misses[("fan", "user USES python")] == ("WRONG_ENDPOINT", "source helper-bot [fan-out]")
    assert misses[("fan", "user USES rust")] == ("OMITTED", " [fan-out]")


def test_relations_beyond_the_prompt_are_set_aside(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    full = _probe(
        "p",
        {"user": "User", "rust": "Technology", "alice": "Person"},
        [
            ("user", "User", "USES", "rust", "Technology"),
            ("user", "User", "KNOWS", "alice", "Person"),
        ],
    )
    pre = {**full, "expected_relationships": full["expected_relationships"][:1]}
    gold = _write(tmp_path / "gold.jsonl", [full])
    pre_gold = _write(tmp_path / "pre.jsonl", [pre])
    raw = _write(tmp_path / "raw.jsonl", [_record("p", {"user": "User"}, [])])
    assert dlm.main(["--gold", str(gold), "--pre-h8-gold", str(pre_gold), f"--run=r={raw}"]) == 0
    out = capsys.readouterr().out
    assert "Relations beyond the prompt set aside: 1\n- p: user KNOWS alice" in out
    assert "| gold relations | 1 |" in out


def test_the_form_kinds_are_imported_from_entity_alignment() -> None:
    # Moved, not copied: the classifier and the aligner read one definition of each form kind.
    from scripts.eval_harness import entity_alignment as ea
    from scripts.eval_harness import score_extraction_run as scorer

    assert dlm._variant_kind is ea.variant_kind
    assert dlm._type_compatible is ea.type_compatible
    assert scorer.types_match is ea.types_match


def test_entity_alignment_imports_neither_the_scorer_nor_the_classifier() -> None:
    import ast

    from scripts.eval_harness import entity_alignment as ea

    tree = ast.parse(Path(ea.__file__).read_text(encoding="utf-8"))
    imported = {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not any(
        name.endswith(("score_extraction_run", "diagnose_list_misses")) for name in imported
    )


@pytest.mark.parametrize(
    ("gold_id", "gold_type", "produced_id", "kind"),
    [
        # The classifier drops N1 words anywhere in the id, not only at the start.
        ("watering-plants", "Activity", "watering-the-plants", "n1"),
        ("acme-corp", "Organization", "the-acme-corp", "n1"),
        ("corp-acme", "Organization", "acme-corp", "reorder"),
        ("kestrel-board", "Project", "kestral-board", "typo"),
        # A Metric whose words are a subset of the other's reads as `metric` here.
        ("71-percent-cpu", "Metric", "71-percent-peak-cpu", "metric"),
        ("quarry-notes", "Document", "quarry-notes-draft", "paraphrase"),
        ("quarry-notes", "Document", "harbor-plan", None),
    ],
)
def test_variant_kind_keeps_its_own_readings(
    gold_id: str, gold_type: str, produced_id: str, kind: str | None
) -> None:
    assert dlm._variant_kind(gold_id, gold_type, produced_id) == kind


@pytest.mark.parametrize("missing", ["gold", "pre", "raw"])
def test_a_missing_file_exits_2(tmp_path: Path, missing: str) -> None:
    paths = {
        name: _write(tmp_path / f"{name}.jsonl", [_probe("p", RUST, [])])
        for name in ("gold", "pre", "raw")
    }
    paths[missing] = tmp_path / "absent.jsonl"
    with pytest.raises(SystemExit) as exc:
        dlm.main(
            [
                "--gold",
                str(paths["gold"]),
                "--pre-h8-gold",
                str(paths["pre"]),
                f"--run=r={paths['raw']}",
            ]
        )
    assert exc.value.code == 2
