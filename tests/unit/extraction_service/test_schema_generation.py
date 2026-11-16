"""Stage 2's output schema is generated from a pack's extractable names.

Today's schema is permissive and names no type, so the generated schema equals
`EXTRACTION_OUTPUT_SCHEMA` for every pack that generates one. The generator still refuses
names it could not write as an enum, so a later, tighter schema refuses no pack that
loads today.
"""

from __future__ import annotations

import json

import pytest

from backend.errors import MistError
from backend.extraction_service.schemas import (
    EXTRACTION_OUTPUT_SCHEMA,
    SchemaGenerationError,
    generate_extraction_schema,
)
from backend.knowledge.ontologies import registry
from backend.knowledge.ontologies.pack import load_pack

BASE = load_pack("tests.unit.knowledge.ontologies.fixtures.base_pack")
PET = load_pack("tests.unit.knowledge.ontologies.fixtures.pet_pack")


def _generate(pack) -> dict:
    return generate_extraction_schema(
        entity_types=pack.extractable_node_types,
        relationship_types=pack.extractable_edge_types,
    )


def test_the_schema_generated_for_the_active_pack_is_the_schema_as_it_stood():
    generated = _generate(registry.active_pack())

    assert generated == EXTRACTION_OUTPUT_SCHEMA
    # Byte for byte: the same keys in the same order, not only an equal mapping.
    assert json.dumps(generated) == json.dumps(EXTRACTION_OUTPUT_SCHEMA)


def test_the_schema_as_it_stood_is_pinned():
    assert json.dumps(EXTRACTION_OUTPUT_SCHEMA) == (
        '{"type": "object", "properties": {"entities": {"type": "array", "items": '
        '{"type": "object", "additionalProperties": true}}, "relationships": {"type": '
        '"array", "items": {"type": "object", "additionalProperties": true}}}, '
        '"required": ["entities", "relationships"], "additionalProperties": true}'
    )


@pytest.mark.parametrize("pack", [BASE, PET], ids=["base", "pet"])
def test_a_fixture_pack_generates_its_schema(pack):
    assert _generate(pack) == EXTRACTION_OUTPUT_SCHEMA
    assert pack.extraction_schema == _generate(pack)


def test_each_call_returns_a_schema_of_its_own():
    first, second = _generate(BASE), _generate(BASE)
    first["properties"]["entities"]["items"]["additionalProperties"] = False

    assert second == EXTRACTION_OUTPUT_SCHEMA
    assert first != EXTRACTION_OUTPUT_SCHEMA


@pytest.mark.parametrize(
    ("entity_types", "relationship_types", "message"),
    [
        pytest.param((), ("CREWS_WITH",), "no extractable entity type", id="no-entity-type"),
        pytest.param(
            ("Deckhand",), (), "no extractable relationship type", id="no-relationship-type"
        ),
        pytest.param(
            ("Deckhand", "Mooring", "Deckhand"),
            ("CREWS_WITH",),
            "entity type Deckhand is listed twice",
            id="repeated-entity-type",
        ),
        pytest.param(
            ("Deckhand",),
            ("CREWS_WITH", "CREWS_WITH"),
            "relationship type CREWS_WITH is listed twice",
            id="repeated-relationship-type",
        ),
        pytest.param(("Deckhand", ""), ("CREWS_WITH",), "an empty entity type", id="empty-name"),
    ],
)
def test_names_an_enum_could_not_hold_do_not_generate(entity_types, relationship_types, message):
    with pytest.raises(SchemaGenerationError, match=message):
        generate_extraction_schema(entity_types=entity_types, relationship_types=relationship_types)


def test_the_generation_error_is_a_mist_error():
    assert issubclass(SchemaGenerationError, MistError)
