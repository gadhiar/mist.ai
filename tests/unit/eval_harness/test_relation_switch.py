"""The rail runner's relation-phase switch, which the scorer's model-written column reads.

`run_rails(..., relation_rails=False)` must be the pack with its relation-phase rules
deleted, byte for byte, and the default call must be the call before the switch existed,
so the headline column scores what the graph stores and the model-written column scores
the same payload with only the relation rules left out (plan v15, T50 Acceptance 6).
"""

from __future__ import annotations

import copy
import inspect
import json

from backend.knowledge.extraction.rails import rail_row, run_rails
from backend.knowledge.ontologies.pack import RELATION, Pack, build_pack, load_pack
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
