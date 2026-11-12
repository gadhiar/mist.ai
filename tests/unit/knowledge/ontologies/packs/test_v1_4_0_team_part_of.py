"""The v1.4.0 pack's team `PART_OF` rail: a named team is no `PART_OF` its employer.

ADR-054's v14 amendment (point 2, register row 3) and plan v15's T51 state the
`team-part-of` rule, an `edge_suppress` rule: a `PART_OF` from an `Organization` that a
`MEMBER_OF` targets, to another `Organization`, is not stored. Every other `PART_OF` is
kept.

Each case is a row run through the pack the registry serves, as apply, replay and the
scorer run it. The names are invented.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

from backend.knowledge.extraction import rails
from backend.knowledge.ontologies import registry
from backend.knowledge.ontologies.pack import RELATION, EdgeSuppress, Pack
from backend.knowledge.ontologies.packs.v1_4_0 import conventions as pack_conventions
from backend.knowledge.ontologies.packs.v1_4_0 import manifest as pack_manifest

_SCRIPTS = Path(__file__).resolve().parents[5] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from eval_harness.score_extraction_run import parse_produced  # noqa: E402

RULE_ID = "team-part-of"

LIVE = {
    "confidence": 0.9,
    "temporal_status": "current",
    "negated": False,
    "assertion_kind": "assert",
}


def _entity(entity_id: str, name: str, entity_type: str) -> dict:
    return {"id": entity_id, "name": name, "type": entity_type}


def _edge(source: str, target: str, edge_type: str) -> dict:
    return {"source": source, "target": target, "type": edge_type, "properties": dict(LIVE)}


BRIGHTWATER = _entity("brightwater", "Brightwater", "Organization")
GROUP = _entity("lindqvist-group", "Lindqvist Group", "Organization")
TEAM = _entity("harrow-data-team", "Harrow data team", "Organization")
AINO = _entity("aino-pelkonen", "Aino Pelkonen", "Person")
MIGRATION = _entity("harrow-migration", "Harrow migration", "Project")
HARBOUR = _entity("saltby-harbour", "Saltby Harbour", "Location")

ENTITIES = (BRIGHTWATER, GROUP, TEAM, AINO, MIGRATION, HARBOUR)
MEMBERSHIP = _edge("aino-pelkonen", "harrow-data-team", "MEMBER_OF")
TEAM_PART_OF = _edge("harrow-data-team", "brightwater", "PART_OF")


@pytest.fixture(scope="module")
def pack() -> Pack:
    """The v1.4.0 pack, as the registry serves it."""
    return registry.get_pack(pack_manifest.MANIFEST.version)


def _run(pack: Pack, relationships, *, scope: str | None = "third-party") -> list[dict]:
    row = rails.rail_row(
        copy.deepcopy(list(ENTITIES)), copy.deepcopy(list(relationships)), scope=scope
    )
    return rails.run_rails(row, pack)[1]


def _part_of(relationships) -> list[tuple[str, str]]:
    return [(rel["source"], rel["target"]) for rel in relationships if rel["type"] == "PART_OF"]


def test_the_rule_suppresses_a_teams_part_of_another_organization(pack: Pack):
    (rule,) = (rule for rule in pack.rules if rule.id == RULE_ID)

    assert (rule.kind, rule.phase, rule.prompt_rule) == ("edge_suppress", RELATION, "Rule 34")
    assert rule.params == EdgeSuppress(
        edge="PART_OF",
        source_types=("Organization",),
        target_types=("Organization",),
        source_is_target_of=("MEMBER_OF",),
    )


@pytest.mark.parametrize("scope", ["user-scope", "third-party", None])
def test_a_teams_part_of_its_employer_is_not_stored(pack: Pack, scope):
    # The shape of the tuning run's rows: no batch record, and the team's `PART_OF` its
    # employer, which the model wrote against Rule 34.
    relationships = _run(pack, (MEMBERSHIP, TEAM_PART_OF), scope=scope)

    assert _part_of(relationships) == []
    assert MEMBERSHIP in relationships


def test_a_projects_part_of_an_organization_is_kept(pack: Pack):
    project = _edge("harrow-migration", "brightwater", "PART_OF")
    # A `MEMBER_OF` targeting a Project is outside the pair table; the type alone keeps it.
    joined = _edge("aino-pelkonen", "harrow-migration", "MEMBER_OF")

    relationships = _run(pack, (MEMBERSHIP, joined, TEAM_PART_OF, project))

    assert _part_of(relationships) == [("harrow-migration", "brightwater")]


def test_an_organizations_part_of_another_that_no_member_of_targets_is_kept(pack: Pack):
    company = _edge("brightwater", "lindqvist-group", "PART_OF")
    works_at = _edge("aino-pelkonen", "brightwater", "WORKS_AT")

    relationships = _run(pack, (MEMBERSHIP, works_at, company, TEAM_PART_OF))

    assert _part_of(relationships) == [("brightwater", "lindqvist-group")]


def test_a_teams_part_of_something_other_than_an_organization_is_kept(pack: Pack):
    at_harbour = _edge("harrow-data-team", "saltby-harbour", "PART_OF")

    relationships = _run(pack, (MEMBERSHIP, at_harbour))

    assert _part_of(relationships) == [("harrow-data-team", "saltby-harbour")]


def test_the_scorer_drops_the_teams_part_of_by_importing_the_pack():
    content = json.dumps({"entities": list(ENTITIES), "relationships": [MEMBERSHIP, TEAM_PART_OF]})

    ok, _, _, relationships = parse_produced(content)

    assert ok
    assert [rel["predicate"] for rel in relationships] == ["MEMBER_OF"]


# --- the pinned case -----------------------------------------------------------------------


def _case():
    (rule,) = (rule for rule in pack_conventions.RULES if rule.id == RULE_ID)
    return rule.pinned_case


def test_the_pinned_case_is_a_third_party_row_with_a_batch_record():
    case = _case()

    assert case.scope == "third-party"
    assert case.batch_record


def test_the_pinned_case_drops_only_the_teams_part_of():
    case = _case()

    dropped = [rel for rel in case.relationships if rel not in case.expected_relationships]
    kept = _part_of(case.expected_relationships)

    assert [(rel["source"], rel["target"], rel["type"]) for rel in dropped] == [
        ("harrow-data-team", "brightwater", "PART_OF")
    ]
    assert kept == [("harrow-migration", "brightwater")]


def test_the_pinned_case_adds_the_employer_and_no_acquaintance():
    case = _case()

    added = [rel for rel in case.expected_relationships if rel not in case.relationships]

    assert [(rel["source"], rel["target"], rel["type"]) for rel in added] == [
        ("kofi-mensah", "brightwater", "WORKS_AT")
    ]
