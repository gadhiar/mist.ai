"""The v1.4.0 pack's team fan-out: a named team's members gain its employer and the user.

ADR-054's v14 amendment (point 2, register rows 1 and 2) and plan v15's T51 state the
`team-fan-out` rule, a `fan_out` rule on each team's `MEMBER_OF` edges:

- each member gains `WORKS_AT` the team's employer E when E resolves. It resolves only
  when the team's own heading or lead-in names an `Organization` E, read from the batch
  record or from the turn plan's template, and a member holds a current `WORKS_AT` E. A
  unit's own text, a member's item or a line of prose, is never read (lead ruling M61).
  A team's `PART_OF` its employer is no trigger: Rule 34 forbids that edge;
- in a user-scope row the user gains `KNOWS_PERSON` each member other than themself,
  whether or not an employer resolves;
- a member gains nothing unless every input edge the rule reads is a current assertion,
  and each added edge copies the member's own `MEMBER_OF` edge's `temporal_status` and
  confidence, and no date.

Each case is a row run through the pack the registry serves, as apply, replay and the
scorer run it. The names are invented.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

from backend.knowledge import version_stamps
from backend.knowledge.extraction import rails
from backend.knowledge.ontologies import registry
from backend.knowledge.ontologies.pack import RELATION, EdgeTrigger, FanOut, Pack, load_pack
from backend.knowledge.ontologies.packs.v1_4_0 import conventions as pack_conventions
from backend.knowledge.ontologies.packs.v1_4_0 import manifest as pack_manifest

_SCRIPTS = Path(__file__).resolve().parents[5] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from eval_harness.score_extraction_run import parse_produced  # noqa: E402

RULE_ID = "team-fan-out"
HEADING = "Kestrel platform team at Varnholt"

# The prompt hash and the extraction version before this rule: the rule runs after the
# model, so neither moves. The rails hash before it, which does.
PROMPT_HASH = "944c7fca0f474d70627a98836ba79d5fdf895dacda34550a3fcb732f0b1b078d"
EXTRACTION_VERSION = "2026-11-12-r9"
RAILS_HASH_BEFORE = "6705d826291cb01f7f9b9f3057fa72ad8ab6aaa1cc17c6bdf3568e7dc2703c05"

LIVE = {
    "confidence": 0.9,
    "temporal_status": "current",
    "negated": False,
    "assertion_kind": "assert",
}

# The four ways an input edge is not a current assertion.
NOT_LIVE = {
    "negated": {**LIVE, "negated": True},
    "past": {**LIVE, "temporal_status": "past"},
    "ceased": {**LIVE, "assertion_kind": "cease"},
    "retracted": {**LIVE, "assertion_kind": "retract"},
}


def _entity(entity_id: str, name: str, entity_type: str) -> dict:
    return {"id": entity_id, "name": name, "type": entity_type}


def _edge(source: str, target: str, edge_type: str, properties: dict | None = None) -> dict:
    edge = {"source": source, "target": target, "type": edge_type}
    if properties is not None:
        edge["properties"] = dict(properties)
    return edge


USER = _entity("user", "User", "User")
VARNHOLT = _entity("varnholt", "Varnholt", "Organization")
QUILLMERE = _entity("quillmere", "Quillmere", "Organization")
KESTREL = _entity("kestrel-platform-team", "Kestrel platform team", "Organization")
ORIANA = _entity("oriana-duarte", "Oriana Duarte", "Person")
TOMASZ = _entity("tomasz-wierzba", "Tomasz Wierzba", "Person")
ROWING = _entity("saltmarsh-rowing-club", "Saltmarsh rowing club", "Organization")
BOATHOUSE = _entity("copperfield-boathouse", "Copperfield Boathouse", "Organization")
ODILE = _entity("odile-brannagh", "Odile Brannagh", "Person")
CHESS = _entity("thursday-chess-circle", "Thursday chess circle", "Organization")
RUAIRI = _entity("ruairi-spelman", "Ruairi Spelman", "Person")

TEAM = (USER, VARNHOLT, KESTREL, ORIANA, TOMASZ)
TEAM_EDGES = (
    _edge("oriana-duarte", "kestrel-platform-team", "MEMBER_OF", LIVE),
    _edge("tomasz-wierzba", "kestrel-platform-team", "MEMBER_OF", LIVE),
    _edge("oriana-duarte", "varnholt", "WORKS_AT", LIVE),
)
COPIED = {"temporal_status": "current", "confidence": 0.9}


def _batch(context: tuple[str, ...], texts: tuple[str, ...], entities, kind="list_item") -> dict:
    """One Stage 2 call: its units, all read under `context`, and the entities it returned."""
    return {
        "units": [{"text": text, "context": list(context), "kind": kind} for text in texts],
        "context": [],
        "payload": {"entities": [dict(entity) for entity in entities], "relationships": []},
        "finish": "stop",
    }


TEAM_RECORD = [_batch((HEADING,), ("Oriana Duarte", "Tomasz Wierzba"), TEAM[1:])]


@pytest.fixture(scope="module")
def pack() -> Pack:
    """The v1.4.0 pack, as the registry serves it."""
    return registry.get_pack(pack_manifest.MANIFEST.version)


def _run(
    pack: Pack,
    entities,
    relationships,
    *,
    scope: str | None = "user-scope",
    batch_record=None,
    turn_plan=None,
) -> tuple[list[dict], list[dict]]:
    row = rails.rail_row(
        copy.deepcopy(list(entities)),
        copy.deepcopy(list(relationships)),
        scope=scope,
        batch_record=copy.deepcopy(batch_record),
        turn_plan=copy.deepcopy(turn_plan),
    )
    return rails.run_rails(row, pack)


def _pairs(relationships, edge_type: str) -> list[tuple[str, str]]:
    return [(rel["source"], rel["target"]) for rel in relationships if rel["type"] == edge_type]


def _rule(pack: Pack, rule_id: str = RULE_ID):
    (rule,) = (rule for rule in pack.rules if rule.id == rule_id)
    return rule


# --- the rule ------------------------------------------------------------------------------


def test_the_rule_fans_out_from_each_teams_member_of_edges(pack: Pack):
    rule = _rule(pack)

    assert (rule.kind, rule.phase) == ("fan_out", RELATION)
    assert isinstance(rule.params, FanOut)
    assert rule.params.trigger == EdgeTrigger(
        edge="MEMBER_OF",
        members="source",
        source_types=("Person", "User"),
        target_types=("Organization",),
    )
    assert "Rule 34" in rule.prompt_rule and "Rule 19" in rule.prompt_rule


def test_a_member_gains_the_employer_and_the_user_gains_the_member(pack: Pack):
    rule = _rule(pack)

    assert [(g.edge, g.source, g.target, g.scopes) for g in rule.params.gains] == [
        ("WORKS_AT", "member", "plugin:team-employer", ()),
        ("KNOWS_PERSON", "user", "member", ("user-scope",)),
    ]
    assert [g.unless_member_has for g in rule.params.gains] == [False, False]


# --- the employer --------------------------------------------------------------------------


def test_each_member_gains_the_employer_the_teams_heading_names(pack: Pack):
    _, relationships = _run(pack, TEAM, TEAM_EDGES, batch_record=TEAM_RECORD)

    assert _pairs(relationships, "WORKS_AT") == [
        ("oriana-duarte", "varnholt"),
        ("tomasz-wierzba", "varnholt"),
    ]
    assert relationships[-2] == _edge("tomasz-wierzba", "varnholt", "WORKS_AT", COPIED)


def test_a_unit_whose_own_text_names_the_team_and_its_employer_resolves_none(pack: Pack):
    # M61: the employer counts from the team's heading or lead-in only, never from a
    # unit's own text, even one that names the team.
    record = [
        _batch(
            ("Teams",),
            (f"{HEADING}: Oriana Duarte, Tomasz Wierzba", "Quillmere reading group: Oriana Duarte"),
            TEAM[1:],
        )
    ]

    _, relationships = _run(pack, TEAM, TEAM_EDGES, batch_record=record)

    assert _pairs(relationships, "WORKS_AT") == [("oriana-duarte", "varnholt")]


def test_a_line_of_prose_that_names_a_team_and_an_employer_resolves_none(pack: Pack):
    # Pass 1's probe P4: the user works at Varnholt and plays in a circle with Ruairi.
    # Ruairi gains no employer; the user's acquaintance needs none, so it still comes.
    entities = (USER, VARNHOLT, CHESS, RUAIRI)
    edges = (
        _edge("user", "varnholt", "WORKS_AT", LIVE),
        _edge("user", "thursday-chess-circle", "MEMBER_OF", LIVE),
        _edge("ruairi-spelman", "thursday-chess-circle", "MEMBER_OF", LIVE),
    )
    text = "I work at Varnholt and play in the Thursday chess circle with Ruairi Spelman."
    record = [_batch((), (text,), entities, kind="prose")]

    _, relationships = _run(pack, entities, edges, batch_record=record)

    assert _pairs(relationships, "WORKS_AT") == [("user", "varnholt")]
    assert _pairs(relationships, "KNOWS_PERSON") == [("user", "ruairi-spelman")]


def test_a_lead_in_under_the_teams_heading_resolves_it(pack: Pack):
    record = [
        _batch(
            ("Kestrel platform team", "All of them at Varnholt:"),
            ("Oriana Duarte", "Tomasz Wierzba"),
            TEAM[1:],
        )
    ]

    _, relationships = _run(pack, TEAM, TEAM_EDGES, batch_record=record)

    assert ("tomasz-wierzba", "varnholt") in _pairs(relationships, "WORKS_AT")


def test_a_lead_in_of_the_plans_template_resolves_it_with_no_batch_record(pack: Pack):
    plan = {
        "entities": [],
        "templates": [
            {
                "lead_in": f"{HEADING}:",
                "subject": "user",
                "predicate": "MEMBER_OF",
                "fixed_endpoint": "kestrel-platform-team",
                "fan_out": [],
            }
        ],
        "back_references": [],
    }

    _, relationships = _run(pack, TEAM, TEAM_EDGES, turn_plan=plan)

    assert ("tomasz-wierzba", "varnholt") in _pairs(relationships, "WORKS_AT")


def test_a_plans_template_for_another_team_names_no_employer_for_this_one(pack: Pack):
    plan = {
        "entities": [],
        "templates": [
            {
                "lead_in": "Varnholt reading group:",
                "subject": "user",
                "predicate": "MEMBER_OF",
                "fixed_endpoint": "varnholt-reading-group",
                "fan_out": [],
            }
        ],
        "back_references": [],
    }

    _, relationships = _run(pack, TEAM, TEAM_EDGES, turn_plan=plan)

    assert _pairs(relationships, "WORKS_AT") == [("oriana-duarte", "varnholt")]


def test_a_row_with_no_batch_record_and_no_plan_gains_no_employer(pack: Pack):
    # The shape of the tuning run's rows: no record, so no heading is read, and the
    # team's `PART_OF` its employer, which the model wrote against Rule 34, is no trigger.
    edges = (*TEAM_EDGES, _edge("kestrel-platform-team", "varnholt", "PART_OF", LIVE))

    _, relationships = _run(pack, TEAM, edges)

    assert _pairs(relationships, "WORKS_AT") == [("oriana-duarte", "varnholt")]
    assert _pairs(relationships, "KNOWS_PERSON") == [
        ("user", "oriana-duarte"),
        ("user", "tomasz-wierzba"),
    ]


def test_a_teams_part_of_its_employer_is_no_trigger(pack: Pack):
    edges = (*TEAM_EDGES, _edge("kestrel-platform-team", "varnholt", "PART_OF", LIVE))
    record = [_batch(("Kestrel platform team",), ("Oriana Duarte", "Tomasz Wierzba"), TEAM[1:])]

    _, relationships = _run(pack, TEAM, edges, batch_record=record)

    assert _pairs(relationships, "WORKS_AT") == [("oriana-duarte", "varnholt")]


def test_a_club_at_a_venue_where_no_member_works_gains_no_employer(pack: Pack):
    entities = (USER, ROWING, BOATHOUSE, ODILE)
    edges = (_edge("odile-brannagh", "saltmarsh-rowing-club", "MEMBER_OF", LIVE),)
    record = [
        _batch(("Saltmarsh rowing club at Copperfield Boathouse",), ("Odile Brannagh",), entities)
    ]

    _, relationships = _run(pack, entities, edges, batch_record=record)

    assert _pairs(relationships, "WORKS_AT") == []
    assert _pairs(relationships, "KNOWS_PERSON") == [("user", "odile-brannagh")]


def test_a_members_own_item_that_names_their_employer_names_none_for_the_team(pack: Pack):
    # ADR-054's v14 amendment rejects a club whose member works somewhere giving every
    # member that employer: a member's own item is not the team's heading, lead-in or unit.
    hospital = _entity("brigid-hospital", "Brigid Hospital", "Organization")
    femi = _entity("femi-achterberg", "Femi Achterberg", "Person")
    entities = (USER, ROWING, hospital, ODILE, femi)
    edges = (
        _edge("odile-brannagh", "saltmarsh-rowing-club", "MEMBER_OF", LIVE),
        _edge("odile-brannagh", "brigid-hospital", "WORKS_AT", LIVE),
        _edge("femi-achterberg", "saltmarsh-rowing-club", "MEMBER_OF", LIVE),
    )
    record = [
        _batch(
            ("Saltmarsh rowing club",),
            ("Odile Brannagh, a nurse at Brigid Hospital", "Femi Achterberg"),
            entities[1:],
        )
    ]

    _, relationships = _run(pack, entities, edges, batch_record=record)

    assert _pairs(relationships, "WORKS_AT") == [("odile-brannagh", "brigid-hospital")]


def test_a_club_beside_the_users_employer_gains_no_employer(pack: Pack):
    entities = (USER, VARNHOLT, CHESS, RUAIRI)
    edges = (
        _edge("user", "varnholt", "WORKS_AT", LIVE),
        _edge("user", "thursday-chess-circle", "MEMBER_OF", LIVE),
        _edge("ruairi-spelman", "thursday-chess-circle", "MEMBER_OF", LIVE),
    )
    record = [
        _batch((), ("I work at Varnholt.",), (USER, VARNHOLT), kind="prose"),
        _batch(("Thursday chess circle",), ("me", "Ruairi Spelman"), (USER, CHESS, RUAIRI)),
    ]

    _, relationships = _run(pack, entities, edges, batch_record=record)

    assert _pairs(relationships, "WORKS_AT") == [("user", "varnholt")]
    assert _pairs(relationships, "KNOWS_PERSON") == [("user", "ruairi-spelman")]


def test_the_employer_of_someone_outside_the_team_is_no_evidence(pack: Pack):
    entities = (USER, VARNHOLT, KESTREL, ORIANA, TOMASZ)
    edges = (*TEAM_EDGES[:2], _edge("user", "varnholt", "WORKS_AT", LIVE))

    _, relationships = _run(pack, entities, edges, batch_record=TEAM_RECORD)

    assert _pairs(relationships, "WORKS_AT") == [("user", "varnholt")]


@pytest.mark.parametrize("form", sorted(NOT_LIVE))
def test_a_members_employer_that_is_not_a_current_assertion_is_no_evidence(pack: Pack, form):
    edges = (*TEAM_EDGES[:2], _edge("oriana-duarte", "varnholt", "WORKS_AT", NOT_LIVE[form]))

    _, relationships = _run(pack, TEAM, edges, batch_record=TEAM_RECORD)

    assert _pairs(relationships, "WORKS_AT") == [("oriana-duarte", "varnholt")]


def test_the_employer_of_a_member_whose_membership_has_ended_is_no_evidence(pack: Pack):
    edges = (
        _edge("oriana-duarte", "kestrel-platform-team", "MEMBER_OF", NOT_LIVE["past"]),
        *TEAM_EDGES[1:],
    )

    _, relationships = _run(pack, TEAM, edges, batch_record=TEAM_RECORD)

    assert _pairs(relationships, "WORKS_AT") == [("oriana-duarte", "varnholt")]


def test_two_employers_the_heading_names_resolve_neither(pack: Pack):
    entities = (*TEAM, QUILLMERE)
    edges = (*TEAM_EDGES, _edge("tomasz-wierzba", "quillmere", "WORKS_AT", LIVE))
    record = [
        _batch((f"{HEADING} and Quillmere",), ("Oriana Duarte", "Tomasz Wierzba"), entities[1:])
    ]

    _, relationships = _run(pack, entities, edges, batch_record=record)

    assert _pairs(relationships, "WORKS_AT") == [
        ("oriana-duarte", "varnholt"),
        ("tomasz-wierzba", "quillmere"),
    ]


# --- the user's acquaintance ---------------------------------------------------------------


def test_the_user_knows_each_member_whether_or_not_an_employer_resolves(pack: Pack):
    _, resolved = _run(pack, TEAM, TEAM_EDGES, batch_record=TEAM_RECORD)
    _, unresolved = _run(pack, TEAM, TEAM_EDGES)

    expected = [("user", "oriana-duarte"), ("user", "tomasz-wierzba")]
    assert _pairs(resolved, "KNOWS_PERSON") == expected
    assert _pairs(unresolved, "KNOWS_PERSON") == expected
    assert _edge("user", "oriana-duarte", "KNOWS_PERSON", COPIED) in resolved


def test_the_user_on_the_team_gains_the_employer_and_no_edge_to_themself(pack: Pack):
    edges = (*TEAM_EDGES, _edge("user", "kestrel-platform-team", "MEMBER_OF", LIVE))

    _, relationships = _run(pack, TEAM, edges, batch_record=TEAM_RECORD)

    assert ("user", "varnholt") in _pairs(relationships, "WORKS_AT")
    assert [pair for pair in _pairs(relationships, "KNOWS_PERSON") if pair[1] == "user"] == []


@pytest.mark.parametrize("scope", ["third-party", "system-scope", "unknown", None])
def test_a_row_in_another_scope_gains_no_acquaintance(pack: Pack, scope):
    entities = TEAM[1:]

    stored, relationships = _run(pack, entities, TEAM_EDGES, scope=scope, batch_record=TEAM_RECORD)

    assert _pairs(relationships, "KNOWS_PERSON") == []
    assert ("tomasz-wierzba", "varnholt") in _pairs(relationships, "WORKS_AT")
    assert "user" not in [entity["id"] for entity in stored]


# --- the inputs ----------------------------------------------------------------------------


@pytest.mark.parametrize("form", sorted(NOT_LIVE))
def test_a_membership_that_is_not_a_current_assertion_adds_nothing(pack: Pack, form):
    edges = (
        TEAM_EDGES[0],
        _edge("tomasz-wierzba", "kestrel-platform-team", "MEMBER_OF", NOT_LIVE[form]),
        TEAM_EDGES[2],
    )

    _, relationships = _run(pack, TEAM, edges, batch_record=TEAM_RECORD)

    assert [rel for rel in relationships if "tomasz-wierzba" in (rel["source"], rel["target"])] == [
        edges[1]
    ]


def test_a_dated_membership_gives_its_confidence_and_status_and_no_date(pack: Pack):
    dated = {
        **LIVE,
        "confidence": 0.85,
        "start_date": "2026-03-01",
        "temporal_expression": "since March",
    }
    edges = (
        TEAM_EDGES[0],
        _edge("tomasz-wierzba", "kestrel-platform-team", "MEMBER_OF", dated),
        TEAM_EDGES[2],
    )

    _, relationships = _run(pack, TEAM, edges, batch_record=TEAM_RECORD)

    copied = {"temporal_status": "current", "confidence": 0.85}
    assert _edge("tomasz-wierzba", "varnholt", "WORKS_AT", copied) in relationships
    assert _edge("user", "tomasz-wierzba", "KNOWS_PERSON", copied) in relationships


# --- determinism ---------------------------------------------------------------------------


def test_the_rule_gives_the_same_bytes_for_the_same_row_and_pack(pack: Pack):
    first = _run(pack, TEAM, TEAM_EDGES, batch_record=TEAM_RECORD)
    again = _run(pack, TEAM, TEAM_EDGES, batch_record=TEAM_RECORD)
    fresh = _run(load_pack(pack_manifest.__package__), TEAM, TEAM_EDGES, batch_record=TEAM_RECORD)

    assert json.dumps(first) == json.dumps(again) == json.dumps(fresh)


# --- the scorer ----------------------------------------------------------------------------


def test_the_scorer_runs_the_rule_by_importing_the_pack():
    content = json.dumps({"entities": list(TEAM), "relationships": list(TEAM_EDGES)})

    ok, _, _, relationships = parse_produced(content, batch_record=copy.deepcopy(TEAM_RECORD))

    assert ok
    added = [(rel["source"], rel["target"], rel["predicate"]) for rel in relationships]
    assert ("tomasz-wierzba", "varnholt", "WORKS_AT") in added


# --- the hashes ----------------------------------------------------------------------------


def test_only_the_rails_hash_moves(pack: Pack):
    release = version_stamps.PACK_RELEASE

    assert version_stamps.EXTRACTION_VERSION == EXTRACTION_VERSION
    assert pack.pack_prompt_hash == release.pack_prompt_hash == PROMPT_HASH
    assert pack.pack_rails_hash == release.pack_rails_hash != RAILS_HASH_BEFORE


# --- the pinned case -----------------------------------------------------------------------


def _case_added(rule_id: str) -> list[dict]:
    """The edges a rule's pinned case expects that its row does not hold."""
    (rule,) = (rule for rule in pack_conventions.RULES if rule.id == rule_id)
    case = rule.pinned_case
    return [rel for rel in case.expected_relationships if rel not in case.relationships]


def test_the_pinned_case_is_a_user_scope_row_with_a_batch_record():
    case = next(rule.pinned_case for rule in pack_conventions.RULES if rule.id == RULE_ID)

    assert case.scope == "user-scope"
    assert case.batch_record
    assert [rel for rel in case.relationships if rel not in case.expected_relationships] == []


def test_the_pinned_case_adds_the_employer_once_and_the_acquaintances():
    added = _case_added(RULE_ID)
    copied = {"temporal_status": "current", "confidence": 0.85}

    # The club named in prose beside the user's employer gains no employer (M61); its
    # other member gains the acquaintance. Someone else's team at the user's employer:
    # its two current members gain the employer (one holds it already) and the user's
    # acquaintance. The negated, past, ceased and retracted members gain nothing. The
    # club at a venue and the club the user is in gain no employer; their other members
    # gain the acquaintance.
    assert [(rel["source"], rel["target"], rel["type"]) for rel in added] == [
        ("user", "signe-varga", "KNOWS_PERSON"),
        ("user", "oriana-duarte", "KNOWS_PERSON"),
        ("tomasz-wierzba", "varnholt", "WORKS_AT"),
        ("user", "tomasz-wierzba", "KNOWS_PERSON"),
        ("user", "odile-brannagh", "KNOWS_PERSON"),
        ("user", "femi-achterberg", "KNOWS_PERSON"),
        ("user", "ruairi-spelman", "KNOWS_PERSON"),
    ]
    # The dated membership's derived edges carry its confidence and status, no date.
    assert added[2]["properties"] == copied
    assert added[3]["properties"] == copied
