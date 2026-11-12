"""The rail runner's relation-phase switch, which the scorer's model-written column reads.

`run_rails(..., relation_rails=False)` must be the pack with its relation-phase rules
deleted, byte for byte, and the default call must be the call before the switch existed,
so the headline column scores what the graph stores and the model-written column scores
the same payload with only the relation rules left out (plan v15, T50 Acceptance 6).
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
import json

import pytest

from backend.knowledge.extraction.rails import _run_phases, rail_row, run_rails
from backend.knowledge.ontologies.pack import (
    RELATION,
    ConventionRule,
    EdgeSuppress,
    Pack,
    _rule_plugins,
    build_pack,
    load_pack,
)
from tests.unit.knowledge.ontologies.fixtures.pet_pack.conventions import PET_ROSTER_CASE

PET = load_pack("tests.unit.knowledge.ontologies.fixtures.pet_pack")

# `run_rails(row, PET)` on the pet roster case, written down before the switch existed.
_BEFORE_THE_SWITCH = (
    '[[{"id": "user", "name": "User", "type": "Deckhand"}, '
    '{"id": "biscuit", "name": "Biscuit", "type": "Pet"}, '
    '{"id": "mango", "name": "Mango", "type": "Pet"}, '
    '{"id": "corwen", "name": "Corwen", "type": "Deckhand"}, '
    '{"id": "sable", "name": "Sable", "type": "Pet"}, '
    '{"id": "pip", "name": "Pip", "type": "Pet"}], '
    '[{"source": "user", "target": "biscuit", "type": "HAS_PET"}, '
    '{"source": "corwen", "target": "sable", "type": "HAS_PET"}, '
    '{"source": "user", "target": "mango", "type": "HAS_PET", '
    '"properties": {"temporal_status": "current", "confidence": 0.9}}]]'
)


def _row():
    case = PET_ROSTER_CASE
    return rail_row(
        copy.deepcopy(list(case.entities)),
        copy.deepcopy(list(case.relationships)),
        scope=case.scope,
        batch_record=copy.deepcopy(case.batch_record),
        turn_plan=copy.deepcopy(case.turn_plan),
    )


def _without_relation_rules(pack: Pack) -> Pack:
    """`pack` again with every relation-phase rule, and the plug-ins only they ran, deleted."""
    return build_pack(
        manifest=pack.manifest,
        node_types=pack.node_types,
        edge_types=pack.edge_types,
        extractable_node_types=pack.extractable_node_types,
        extractable_edge_types=pack.extractable_edge_types,
        rules=tuple(rule for rule in pack.rules if rule.phase != RELATION),
        plugins=tuple(plugin for plugin in pack.plugins if plugin.phase != RELATION),
    )


def _bytes(result) -> str:
    return json.dumps(result)


def test_the_switch_is_keyword_only_and_on_by_default():
    parameter = inspect.signature(run_rails).parameters["relation_rails"]

    assert (parameter.kind, parameter.default) == (inspect.Parameter.KEYWORD_ONLY, True)


def test_the_fixture_pack_holds_a_fan_out_rule():
    assert "fan_out" in {rule.kind for rule in PET.rules_in(RELATION)}


def test_switched_off_is_the_pack_with_its_relation_rules_deleted_byte_for_byte():
    switched_off = run_rails(_row(), PET, relation_rails=False)
    deleted = run_rails(_row(), _without_relation_rules(PET))

    assert _bytes(switched_off) == _bytes(deleted)


def test_switched_off_leaves_out_the_edge_the_fan_out_adds():
    _, relationships = run_rails(_row(), PET, relation_rails=False)

    assert relationships == list(PET_ROSTER_CASE.relationships)


def test_the_default_call_is_byte_identical_to_the_call_before_the_switch():
    assert _bytes(run_rails(_row(), PET)) == _BEFORE_THE_SWITCH
    assert _bytes(run_rails(_row(), PET, relation_rails=True)) == _BEFORE_THE_SWITCH


# --- the v1.4.0 pack (plan v15, T50 Acceptance 7) ----------------------------------------------
#
# The id phase runs in both columns, so an Event dated by its own edge gets one id in each.
# Each relation-phase rule of the pack must change an edge in its own pinned case when it
# alone is deleted, or the two columns could not differ by it. The parameter set is the
# pack's relation-phase rules, each test named by the rule's id: a pack with none reports
# one skipped test whose reason names the pack, and nothing here asserts the set is
# non-empty.

V1_4_0 = load_pack("backend.knowledge.ontologies.packs.v1_4_0")


def _dated_event_response() -> dict:
    return {
        "payload": {
            "entities": [
                {"id": "user", "name": "User", "type": "User"},
                {"id": "harbour-gala", "name": "Harbour gala", "type": "Event"},
                {"id": "2026-11-18", "name": "2026-11-18", "type": "Date"},
            ],
            "relationships": [
                {"source": "user", "target": "harbour-gala", "type": "EXPERIENCED"},
                {"source": "harbour-gala", "target": "2026-11-18", "type": "OCCURRED_ON"},
            ],
        }
    }


def test_the_scorer_reads_the_v1_4_0_pack():
    from backend.knowledge.ontologies import registry

    assert registry.active_pack().version == V1_4_0.version == "1.4.0"


def test_a_dated_event_gets_the_same_id_in_both_columns():
    from scripts.eval_harness.score_extraction_run import produced_from_response

    columns = [
        produced_from_response("I went to the harbour gala", _dated_event_response(), **switch)
        for switch in ({}, {"relation_rails": False})
    ]
    events = [[e.id for e in produced.entities if e.type == "Event"] for produced in columns]

    assert events == [["harbour-gala-2026-11-18"], ["harbour-gala-2026-11-18"]]


def _edges(result) -> set[tuple[str, str, str]]:
    _, relationships = result
    return {(r["source"], r["type"], r["target"]) for r in relationships}


def _case_row(case):
    return rail_row(
        copy.deepcopy(list(case.entities)),
        copy.deepcopy(list(case.relationships)),
        scope=case.scope,
        batch_record=copy.deepcopy(case.batch_record),
        turn_plan=copy.deepcopy(case.turn_plan),
    )


def _without_rule(pack: Pack, rule) -> Pack:
    """`pack` again with `rule` deleted, and with it any plug-in no other rule names.

    `build_pack` refuses a pack carrying the release's version with hashes other than the
    release's, so the cut-down pack takes a version of its own.
    """
    rules = tuple(kept for kept in pack.rules if kept.id != rule.id)
    named = {name for kept in rules for name in _rule_plugins(kept)}
    return build_pack(
        manifest=dataclasses.replace(pack.manifest, version=f"{pack.version}-without-{rule.id}"),
        node_types=pack.node_types,
        edge_types=pack.edge_types,
        extractable_node_types=pack.extractable_node_types,
        extractable_edge_types=pack.extractable_edge_types,
        rules=rules,
        plugins=tuple(plugin for plugin in pack.plugins if plugin.name in named),
    )


def _changed_edges(pack: Pack, rule) -> set[tuple[str, str, str]]:
    """The edges `rule` alone adds or removes in its own pinned case.

    The full pack runs through `run_rails`, so its pinned cases are checked. The cut-down
    pack runs through `_run_phases`, which skips that check: with `rule` gone, another
    rule's case that leans on it no longer holds, and that is the difference measured here.
    """
    case = rule.pinned_case
    assert case is not None, f"{rule.id} has no pinned case"
    with_rule = _edges(run_rails(_case_row(case), pack))
    without_rule = _edges(_run_phases(_case_row(case), _without_rule(pack, rule)))
    return with_rule ^ without_rule


# The pet pack, plus an `edge_suppress` rule that matches nothing: no `Pet` is the source of
# a `HAS_PET`. Its pinned case is the roster case, where the working `pet-roster` fan-out
# adds an edge, so only a comparison that deletes the inert rule alone finds it inert.
_INERT = ConventionRule(
    id="inert-pet-suppress",
    kind="edge_suppress",
    phase=RELATION,
    params=EdgeSuppress(edge="HAS_PET", source_types=("Pet",)),
    pinned_case=PET_ROSTER_CASE,
    prompt_rule="N18",
)
_PET_AND_INERT = build_pack(
    manifest=PET.manifest,
    node_types=PET.node_types,
    edge_types=PET.edge_types,
    extractable_node_types=PET.extractable_node_types,
    extractable_edge_types=PET.extractable_edge_types,
    rules=(*PET.rules, _INERT),
    plugins=PET.plugins,
)


def test_a_rule_that_does_nothing_changes_no_edge_in_its_pinned_case():
    assert _changed_edges(_PET_AND_INERT, _INERT) == set()


def test_a_working_rule_changes_an_edge_in_its_pinned_case():
    (roster,) = (rule for rule in _PET_AND_INERT.rules if rule.id == "pet-roster")

    assert _changed_edges(_PET_AND_INERT, roster) == {("user", "HAS_PET", "mango")}


def _relation_rule_params(pack: Pack) -> list:
    rules = [pytest.param(rule, id=rule.id) for rule in pack.rules_in(RELATION)]
    if rules:
        return rules
    reason = f"pack {pack.version} has no relation-phase rule"
    return [pytest.param(None, id="no-relation-rule", marks=pytest.mark.skip(reason=reason))]


@pytest.mark.parametrize("rule", _relation_rule_params(V1_4_0))
def test_every_relation_rule_changes_an_edge_in_its_pinned_case(rule):
    assert _changed_edges(V1_4_0, rule)
