"""Pin the judgement filter: the rails' derived relations of one team count once.

A roster of N people under a team heading that names the employer gives N `WORKS_AT` and N
`KNOWS_PERSON` edges, and every one of them the v1.4.0 rails add back from the team's
`MEMBER_OF` edges. Counted one by one, two passes that agree on the roster agree on 2N
relations they never had to judge, and two that disagree on one employer disagree N times.
The filter keeps one relation of each (rule, team, predicate) group, in both passes, so agreement
is measured on what a labeller decided. Every roster here is synthetic, with invented names and
headings that name the employer; the pack and `run_rails` are the caller's, as the module asks.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.knowledge.extraction.rails import run_rails  # noqa: E402
from backend.knowledge.ontologies.pack import load_pack  # noqa: E402
from scripts.eval_harness import judgement_relations  # noqa: E402
from scripts.eval_harness.judgement_relations import (  # noqa: E402
    batch_record,
    judgement_filter,
    stored_row,
)

PACK = load_pack("backend.knowledge.ontologies.packs.v1_4_0")
USER_SCOPE = "user-scope"


def _entity(entity_id: str, entity_type: str) -> dict:
    return {"id": entity_id, "type": entity_type}


def _edge(source: str, source_type: str, predicate: str, target: str, target_type: str) -> dict:
    return {
        "source": source,
        "source_type": source_type,
        "predicate": predicate,
        "target": target,
        "target_type": target_type,
    }


def _member(person: str, team: str) -> dict:
    return _edge(person, "Person", "MEMBER_OF", team, "Organization")


def _works_at(person: str, employer: str) -> dict:
    return _edge(person, "Person", "WORKS_AT", employer, "Organization")


def _knows(person: str) -> dict:
    return _edge("user", "User", "KNOWS_PERSON", person, "Person")


def _roster(team: str, employer: str, people: list[str]) -> tuple[list[dict], list[dict]]:
    """A team's entities and every edge a labeller writes for it: membership, job, knowing."""
    entities = [
        _entity(team, "Organization"),
        _entity(employer, "Organization"),
        *(_entity(person, "Person") for person in people),
    ]
    edges = [
        *(_member(person, team) for person in people),
        *(_works_at(person, employer) for person in people),
        *(_knows(person) for person in people),
    ]
    return entities, edges


def _row(utterance: str, entities: list[dict], edges: list[dict]) -> dict:
    return {
        "utterance": utterance,
        "tag": "synthetic-roster",
        "expected_entities": [_entity("user", "User"), *entities],
        "expected_relationships": edges,
    }


def _keys(row: dict) -> set[tuple[str, str, str]]:
    return {(e["source"], e["predicate"], e["target"]) for e in row["expected_relationships"]}


def _of(row: dict, predicate: str) -> set[tuple[str, str, str]]:
    return {key for key in _keys(row) if key[1] == predicate}


def _filter(row_a: dict, row_b: dict, scope: str | None = USER_SCOPE):
    return judgement_filter(row_a, row_b, pack=PACK, run_rails=run_rails, scope=scope)


HARBOUR = "## Harbour team at Tollvik\n- Brannoc Ashe\n- Celyn Dorr\n- Ewan Faull"
HARBOUR_PEOPLE = ["brannoc-ashe", "celyn-dorr", "ewan-faull"]


def test_one_roster_keeps_its_memberships_one_works_at_and_one_knows_person():
    entities, edges = _roster("harbour-team", "tollvik", HARBOUR_PEOPLE)
    row = _row(HARBOUR, entities, edges)

    result = _filter(row, row)

    for kept in (result.row_a, result.row_b):
        assert _of(kept, "MEMBER_OF") == {(p, "MEMBER_OF", "harbour-team") for p in HARBOUR_PEOPLE}
        assert _of(kept, "WORKS_AT") == {("brannoc-ashe", "WORKS_AT", "tollvik")}
        assert _of(kept, "KNOWS_PERSON") == {("user", "KNOWS_PERSON", "brannoc-ashe")}
    assert result.removed_a == 4
    assert result.removed_b == 4


def test_two_teams_keep_one_works_at_and_one_knows_person_each():
    utterance = (
        "## Harbour team at Tollvik\n- Brannoc Ashe\n- Celyn Dorr\n"
        "## Kiln crew at Morrow Glass\n- Ilse Jarn\n- Kesh Lowe"
    )
    harbour_entities, harbour_edges = _roster(
        "harbour-team", "tollvik", ["brannoc-ashe", "celyn-dorr"]
    )
    kiln_entities, kiln_edges = _roster("kiln-crew", "morrow-glass", ["ilse-jarn", "kesh-lowe"])
    row = _row(utterance, [*harbour_entities, *kiln_entities], [*harbour_edges, *kiln_edges])

    result = _filter(row, row)

    for kept in (result.row_a, result.row_b):
        assert len(_of(kept, "MEMBER_OF")) == 4
        assert _of(kept, "WORKS_AT") == {
            ("brannoc-ashe", "WORKS_AT", "tollvik"),
            ("ilse-jarn", "WORKS_AT", "morrow-glass"),
        }
        assert _of(kept, "KNOWS_PERSON") == {
            ("user", "KNOWS_PERSON", "brannoc-ashe"),
            ("user", "KNOWS_PERSON", "ilse-jarn"),
        }


def test_passes_that_differ_on_the_first_members_membership_differ_on_that_one_relation():
    entities, edges = _roster("harbour-team", "tollvik", HARBOUR_PEOPLE)
    first = _member("brannoc-ashe", "harbour-team")
    row_a = _row(HARBOUR, entities, edges)
    row_b = _row(HARBOUR, entities, [edge for edge in edges if edge != first])

    result = _filter(row_a, row_b)

    assert _keys(result.row_a) - _keys(result.row_b) == {
        ("brannoc-ashe", "MEMBER_OF", "harbour-team")
    }
    assert _keys(result.row_b) - _keys(result.row_a) == set()


def test_passes_that_name_different_employers_differ_on_one_works_at():
    utterance = "## Harbour team, Tollvik (formerly Quarrow)\n- Brannoc Ashe\n- Celyn Dorr"
    people = ["brannoc-ashe", "celyn-dorr"]
    row_a = _row(utterance, *_roster("harbour-team", "tollvik", people))
    row_b = _row(utterance, *_roster("harbour-team", "quarrow", people))

    result = _filter(row_a, row_b)

    differing = _keys(result.row_a) ^ _keys(result.row_b)
    assert len(differing) == 1
    assert {predicate for _, predicate, _ in differing} == {"WORKS_AT"}


def test_a_works_at_with_no_membership_beside_it_is_kept():
    utterance = (
        "Rafe Dunn also works at Tollvik.\n"
        "## Harbour team at Tollvik\n- Brannoc Ashe\n- Celyn Dorr"
    )
    entities, edges = _roster("harbour-team", "tollvik", ["brannoc-ashe", "celyn-dorr"])
    row = _row(
        utterance,
        [*entities, _entity("rafe-dunn", "Person")],
        [*edges, _works_at("rafe-dunn", "tollvik"), _knows("rafe-dunn")],
    )

    result = _filter(row, row)

    for kept in (result.row_a, result.row_b):
        assert ("rafe-dunn", "WORKS_AT", "tollvik") in _keys(kept)
        assert ("user", "KNOWS_PERSON", "rafe-dunn") in _keys(kept)
        assert len(_of(kept, "WORKS_AT")) == 2


def test_outside_a_user_scope_row_knows_person_is_the_labellers_own():
    entities, edges = _roster("harbour-team", "tollvik", HARBOUR_PEOPLE)
    row = _row(HARBOUR, entities, edges)

    result = _filter(row, row, scope="third-party")

    assert len(_of(result.row_a, "KNOWS_PERSON")) == 3
    assert len(_of(result.row_a, "WORKS_AT")) == 1


def test_the_two_passes_must_be_of_one_probe():
    row = _row(HARBOUR, *_roster("harbour-team", "tollvik", HARBOUR_PEOPLE))

    with pytest.raises(ValueError):
        _filter(row, {**row, "tag": "another-probe"})


def test_the_batch_record_is_the_services_one_entry_per_batch_under_a_heading():
    utterance = (
        "## Harbour team at Tollvik\n- Brannoc Ashe\n## Kiln crew at Morrow Glass\n- Ilse Jarn"
    )
    payload = {"entities": [{"id": "user", "type": "User"}], "relationships": []}

    record = batch_record(utterance, payload, pack=PACK)

    assert [entry["units"] for entry in record] == [
        [{"text": "Brannoc Ashe", "context": ["Harbour team at Tollvik"], "kind": "list_item"}],
        [{"text": "Ilse Jarn", "context": ["Kiln crew at Morrow Glass"], "kind": "list_item"}],
    ]
    assert all(entry["payload"] == payload for entry in record)


def test_stored_row_gives_the_stored_ids_and_adds_no_edge():
    entities, edges = _roster("harbour-team", "tollvik", HARBOUR_PEOPLE[:1])
    edges = [edge for edge in edges if edge["predicate"] == "MEMBER_OF"]
    row = _row(HARBOUR, entities, edges)

    stored = stored_row(row, pack=PACK, run_rails=run_rails, scope=USER_SCOPE)

    assert _keys(stored) == {("brannoc-ashe", "MEMBER_OF", "harbour-team")}
    assert stored["tag"] == row["tag"]
    assert row["expected_relationships"] == edges


def test_the_module_imports_neither_the_pack_nor_the_rails():
    tree = ast.parse(Path(judgement_relations.__file__).read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            modules.add(node.module or "")

    assert not {m for m in modules if m.startswith("backend.knowledge.ontologies")}
    assert "backend.knowledge.extraction.rails" not in modules


def test_importing_the_module_does_not_load_the_rails():
    """The engine's batcher loads the ontology package, so only the rails are fenced here."""
    probe = (
        "import sys; import scripts.eval_harness.judgement_relations; "
        "print(sorted(m for m in sys.modules if m.startswith('backend.knowledge.extraction.rails')))"
    )
    done = subprocess.run(
        [sys.executable, "-c", probe], cwd=_REPO_ROOT, capture_output=True, text=True, check=True
    )

    assert done.stdout.strip() == "[]"
