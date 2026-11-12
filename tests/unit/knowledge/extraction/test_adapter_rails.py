"""Today's ontology through its pack: the rail runner gives the old id rails' output.

`every_rail_payload.py` holds one synthetic payload that fires every rail the id rails
ran before the pack, with the output they gave written out by hand.
"""

from __future__ import annotations

import copy

from backend.knowledge.extraction import id_rails
from backend.knowledge.extraction.rails import check_pinned_cases, rail_row, run_rails
from backend.knowledge.ontologies import registry
from tests.unit.knowledge.extraction import every_rail_payload


def _every_rail_payload() -> tuple[list[dict], list[dict]]:
    return (
        copy.deepcopy(every_rail_payload.ENTITIES),
        copy.deepcopy(every_rail_payload.RELATIONSHIPS),
    )


def test_run_rails_through_the_active_pack_gives_the_output_of_the_old_id_rails():
    entities, relationships = _every_rail_payload()

    out_entities, out_relationships = run_rails(
        rail_row(entities, relationships), registry.active_pack()
    )

    assert out_entities == every_rail_payload.EXPECTED_ENTITIES
    assert out_relationships == every_rail_payload.EXPECTED_RELATIONSHIPS
    assert [list(entity) for entity in out_entities] == [
        list(entity) for entity in every_rail_payload.EXPECTED_ENTITIES
    ]


def test_normalize_payload_is_run_rails_through_the_active_pack():
    entities, relationships = _every_rail_payload()

    assert id_rails.normalize_payload(entities, relationships) == run_rails(
        rail_row(*_every_rail_payload()), registry.active_pack()
    )


def _without_an_id_key() -> tuple[dict, dict]:
    """A reserved name with no type and a retired type with a default, neither with an id."""
    pack = registry.active_pack()
    reserved_name = pack.rules_of("reserved_identity")[0].params.names[0]
    retired = next(r.name for r in pack.manifest.retired_types if r.property_defaults)
    return {"name": reserved_name}, {"name": "Harbour Opening", "type": retired}


def test_an_entity_with_no_id_key_gets_its_keys_in_the_order_the_id_rails_gave():
    reserved, retired = _without_an_id_key()

    entities, _ = id_rails.normalize_payload([reserved, retired], [])

    assert [list(entity) for entity in entities] == [
        ["name", "id", "type"],
        ["name", "type", "id", "properties"],
    ]


def test_normalize_entity_gives_the_same_key_order_as_the_payload_path():
    reserved, retired = _without_an_id_key()

    id_rails.normalize_entity(reserved)
    id_rails.normalize_entity(retired)

    assert list(reserved) == ["name", "id", "type"]
    assert list(retired) == ["name", "type", "id", "properties"]


def test_the_active_packs_own_pinned_cases_hold():
    assert check_pinned_cases(registry.active_pack()) == []


def test_the_synthetic_payload_reproduces_the_old_id_rails_and_the_pack_kinds_are_pinned():
    """The payload stores what the old id rails stored, and the active pack's kinds are pinned.

    The payload was written for the old id rails, so it does not fire the kinds the team
    rules added; their own pinned cases cover those.
    """
    pack = registry.active_pack()
    post_llm = {rule.kind for rule in pack.rules if rule.phase != "pre_llm"}

    assert post_llm == {
        "reserved_identity",
        "id_exempt",
        "alias_vocabulary",
        "typed_id",
        "derived_edge",
        "fan_out",
    }
    stored = {entity["id"] for entity in every_rail_payload.EXPECTED_ENTITIES}
    assert {"mist-identity", "c-plus-plus", "quillscript", "nodejs", "kubernetes"} <= stored
    assert {"2-4-seconds-cold-start", "regatta-2026-05-30", "my-first-date", "odd-thing"} <= stored
