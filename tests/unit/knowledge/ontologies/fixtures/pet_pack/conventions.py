"""The pet fixture pack's rules: the base pack's, plus the `pet-roster` convention.

Under a roster lead-in whose plan template names `HAS_PET`, every `Pet` the batch record
places in that lead-in's batches gets `user HAS_PET`, unless the model already gave it a
`HAS_PET` from a named owner.
"""

from backend.knowledge.ontologies.pack import (
    PRE_LLM,
    RELATION,
    ConventionRule,
    FanOut,
    GainedEdge,
    PinnedCase,
    PlanTrigger,
    PlanVocabulary,
)
from tests.unit.knowledge.ontologies.fixtures.base_pack.conventions import RULES as BASE_RULES


def _unit(text: str, lead_in: str) -> dict:
    return {"text": text, "context": [lead_in], "kind": "list_item"}


def _batch(units: list[dict], entities: list[dict], relationships: list[dict]) -> dict:
    return {
        "units": units,
        "context": [],
        "payload": {"entities": entities, "relationships": relationships},
        "prompt_tokens": 120,
        "completion_tokens": 40,
        "finish": "stop",
    }


_USER = {"id": "user", "name": "User", "type": "Deckhand"}
_BISCUIT = {"id": "biscuit", "name": "Biscuit", "type": "Pet"}
_MANGO = {"id": "mango", "name": "Mango", "type": "Pet"}
_CORWEN = {"id": "corwen", "name": "Corwen", "type": "Deckhand"}
_SABLE = {"id": "sable", "name": "Sable", "type": "Pet"}
_PIP = {"id": "pip", "name": "Pip", "type": "Pet"}
_USER_HAS_BISCUIT = {"source": "user", "target": "biscuit", "type": "HAS_PET"}
_CORWEN_HAS_SABLE = {"source": "corwen", "target": "sable", "type": "HAS_PET"}

PET_ROSTER_CASE = PinnedCase(
    scope="user-scope",
    entities=(_USER, _BISCUIT, _MANGO, _CORWEN, _SABLE, _PIP),
    relationships=(_USER_HAS_BISCUIT, _CORWEN_HAS_SABLE),
    turn_plan={
        "entities": [_BISCUIT, _MANGO, _CORWEN, _SABLE, _PIP],
        "templates": [
            {
                "lead_in": "Our pets:",
                "subject": "user",
                "predicate": "HAS_PET",
                "fixed_endpoint": "",
                "fan_out": [],
            }
        ],
        "back_references": [],
    },
    batch_record=(
        _batch(
            [_unit("Biscuit, a terrier", "Our pets"), _unit("Mango, a parrot", "Our pets")],
            [_USER, _BISCUIT, _MANGO],
            [_USER_HAS_BISCUIT],
        ),
        _batch(
            [_unit("Corwen keeps a ferret called Sable", "Our pets")],
            [_CORWEN, _SABLE],
            [_CORWEN_HAS_SABLE],
        ),
        _batch([_unit("Pip", "The neighbours' animals")], [_PIP], []),
    ),
    expected_entities=(_USER, _BISCUIT, _MANGO, _CORWEN, _SABLE, _PIP),
    expected_relationships=(
        _USER_HAS_BISCUIT,
        _CORWEN_HAS_SABLE,
        {
            "source": "user",
            "target": "mango",
            "type": "HAS_PET",
            "properties": {"temporal_status": "current", "confidence": 0.9},
        },
    ),
)

RULES = (
    *BASE_RULES,
    ConventionRule(
        id="pet-roster",
        kind="fan_out",
        phase=RELATION,
        params=FanOut(
            trigger=PlanTrigger(predicate="HAS_PET", member_type="Pet"),
            gains=(
                GainedEdge(
                    edge="HAS_PET",
                    source="user",
                    target="member",
                    unless_member_has=True,
                    properties=(("temporal_status", "current"), ("confidence", 0.9)),
                ),
            ),
        ),
        prompt_rule="N18",
        pinned_case=PET_ROSTER_CASE,
    ),
    ConventionRule(
        id="pet-plan-words",
        kind="plan_vocabulary",
        phase=PRE_LLM,
        params=PlanVocabulary(
            entity_types=("Pet",), predicates=("HAS_PET",), fan_out_predicates=("HAS_PET",)
        ),
    ),
)
