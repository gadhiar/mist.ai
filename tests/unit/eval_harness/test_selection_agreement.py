"""Pin the selection set's committed agreement figures to the module that computes them.

`agreement.json` was read before any relation was adjudicated, so a reader can trust it only if
the same inputs give the same file again. These tests recompute it from the two blind passes
under the pack whose rails hash the file records, and compare. Each comparison is reduced to a
bool first, so a failure prints no row of the selection set.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.knowledge.extraction.rails import run_rails  # noqa: E402
from backend.knowledge.ontologies.pack import load_pack  # noqa: E402
from scripts.eval_harness import selection_agreement as selection_agreement_module  # noqa: E402
from scripts.eval_harness.selection_agreement import (  # noqa: E402
    main,
    read_rows,
    render,
    selection_agreement,
)

_LABELLING = _REPO_ROOT / "data" / "gold" / "extraction" / "labelling" / "selection-2026-11"
_PASS_A = _LABELLING / "pass-a.jsonl"
_PASS_B = _LABELLING / "pass-b.jsonl"
_COMMITTED = _LABELLING / "agreement.json"
_SELECTION = _REPO_ROOT / "data" / "gold" / "extraction" / "selection-2026-11.jsonl"

PACK = load_pack("backend.knowledge.ontologies.packs.v1_4_0")


def test_the_module_itself_imports_neither_the_pack_nor_the_rails():
    """Only `main` resolves the pack and the rails, so the ones measured are the caller's.

    Importing the module still loads the engine's v1.4.0 pack modules, through
    `judgement_relations`; this checks the module's own imports only.
    """
    tree = ast.parse(Path(selection_agreement_module.__file__).read_text(encoding="utf-8"))
    modules = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            modules |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            modules.add(node.module or "")

    fenced = {m for m in modules if m.startswith("backend.")}

    assert not fenced


def test_the_pack_is_the_one_the_committed_figures_record():
    recorded = json.loads(_COMMITTED.read_text(encoding="utf-8"))

    same_pack = (PACK.pack_rails_hash, PACK.version) == (
        recorded["pack_rails_hash"],
        recorded["pack_version"],
    )

    assert same_pack


def test_the_module_reproduces_the_committed_agreement():
    report = selection_agreement(
        read_rows(_PASS_A),
        read_rows(_PASS_B),
        read_rows(_SELECTION),
        pack=PACK,
        run_rails=run_rails,
    )

    same = render(report) == _COMMITTED.read_text(encoding="utf-8")

    assert same


def test_the_cli_writes_the_committed_file_byte_for_byte(tmp_path):
    out = tmp_path / "agreement.json"

    main(
        [
            "--pass-a",
            str(_PASS_A),
            "--pass-b",
            str(_PASS_B),
            "--selection",
            str(_SELECTION),
            "--recorded",
            str(_COMMITTED),
            "--out",
            str(out),
        ]
    )
    same = out.read_bytes() == _COMMITTED.read_bytes()

    assert same


# M63's scope rule on invented probes: the rails read a probe as user-scope when either pass
# labels the user, and as third-party when neither does. Each probe's marker is a person only
# it names, so a rail call is traced back to its probe.
_SCOPE_PROBES = {
    "pin-001-prose-pass-a-holds-the-user": (
        "ondrel-vask",
        "I know Ondrel Vask from the climbing wall; he works at Brask Mill.",
        [
            ("user", "User", "KNOWS_PERSON", "ondrel-vask", "Person"),
            ("ondrel-vask", "Person", "WORKS_AT", "brask-mill", "Organization"),
        ],
        [("ondrel-vask", "Person", "WORKS_AT", "brask-mill", "Organization")],
    ),
    "pin-002-prose-pass-b-holds-the-user": (
        "ysolde-marrick",
        "Ysolde Marrick, whom I met at the ferry, works at Corlew Yard.",
        [("ysolde-marrick", "Person", "WORKS_AT", "corlew-yard", "Organization")],
        [
            ("user", "User", "KNOWS_PERSON", "ysolde-marrick", "Person"),
            ("ysolde-marrick", "Person", "WORKS_AT", "corlew-yard", "Organization"),
        ],
    ),
    "pin-003-prose-neither-pass-holds-the-user": (
        "petrik-hallam",
        "Petrik Hallam works at Fenwhistle Dock.",
        [("petrik-hallam", "Person", "WORKS_AT", "fenwhistle-dock", "Organization")],
        [("petrik-hallam", "Person", "WORKS_AT", "fenwhistle-dock", "Organization")],
    ),
}


def _scope_row(tag: str, utterance: str, edges: list[tuple[str, str, str, str, str]]) -> dict:
    entities: dict[str, str] = {}
    relationships = []
    for source, source_type, predicate, target, target_type in edges:
        entities.setdefault(source, source_type)
        entities.setdefault(target, target_type)
        relationships.append(
            {
                "source": source,
                "source_type": source_type,
                "predicate": predicate,
                "target": target,
                "target_type": target_type,
            }
        )
    return {
        "tag": tag,
        "utterance": utterance,
        "expected_entities": [{"id": i, "type": t} for i, t in entities.items()],
        "expected_relationships": relationships,
    }


def _scopes_the_rails_read() -> dict[str, set]:
    rows_a = [_scope_row(tag, utt, a) for tag, (_, utt, a, _) in _SCOPE_PROBES.items()]
    rows_b = [_scope_row(tag, utt, b) for tag, (_, utt, _, b) in _SCOPE_PROBES.items()]
    seen: dict[str, set] = {tag: set() for tag in _SCOPE_PROBES}

    def recording_rails(row, pack, **kwargs):
        ids = {entity["id"] for entity in row.entities}
        for tag, (marker, _, _, _) in _SCOPE_PROBES.items():
            if marker in ids:
                seen[tag].add(row.scope)
        return run_rails(row, pack, **kwargs)

    selection_agreement(rows_a, rows_b, rows_a, pack=PACK, run_rails=recording_rails)
    return seen


def test_a_probe_is_user_scope_when_one_pass_labels_the_user():
    seen = _scopes_the_rails_read()

    assert seen["pin-001-prose-pass-a-holds-the-user"] == {"user-scope"}
    assert seen["pin-002-prose-pass-b-holds-the-user"] == {"user-scope"}


def test_a_probe_is_third_party_when_neither_pass_labels_the_user():
    seen = _scopes_the_rails_read()

    assert seen["pin-003-prose-neither-pass-holds-the-user"] == {"third-party"}
