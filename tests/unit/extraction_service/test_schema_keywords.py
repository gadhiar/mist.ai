"""The Stage 2 schema uses only keywords llama.cpp's grammar converter reads.

The extraction host pins llama.cpp b11151 (`docker-compose.extraction.yml`). Its schema
parser, `common/json-schema.cpp`, reads the keywords below and drops any other without an
error: an `if`/`then` beside an object's keys would be ignored and the model left
unconstrained where the schema meant to constrain it. A llama.cpp build bump re-reads that
file and updates this set.
"""

from __future__ import annotations

import pytest

from backend.extraction_service.schemas import generate_extraction_schema
from backend.knowledge.ontologies import registry
from backend.knowledge.ontologies.pack import Pack, load_pack

# The keywords `common/json-schema.cpp` reads at b11151.
CONVERTER_KEYWORDS = frozenset(
    {
        "$ref",
        "oneOf",
        "anyOf",
        "allOf",
        "type",
        "const",
        "enum",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "prefixItems",
        "minItems",
        "maxItems",
        "pattern",
        "format",
        "minLength",
        "maxLength",
        "minimum",
        "exclusiveMinimum",
        "maximum",
        "exclusiveMaximum",
    }
)
# Keys whose values are data, not schemas: walked no further.
_DATA_KEYWORDS = frozenset({"enum", "const", "required"})


def _generate(pack: Pack) -> dict:
    by_name = {node.type_name: node for node in pack.node_types}
    return generate_extraction_schema(
        entity_types=tuple(by_name[name] for name in pack.extractable_node_types),
        relationship_types=pack.extractable_edge_types,
    )


def _keywords(schema: object) -> set[str]:
    """Every keyword a schema node uses, the children of `properties` read as names."""
    found: set[str] = set()
    if isinstance(schema, list):
        for item in schema:
            found |= _keywords(item)
    elif isinstance(schema, dict):
        for key, value in schema.items():
            found.add(key)
            if key in _DATA_KEYWORDS:
                continue
            if key == "properties" and isinstance(value, dict):
                for child in value.values():
                    found |= _keywords(child)
            else:
                found |= _keywords(value)
    return found


PACKS = [
    pytest.param(registry.active_pack(), id="active"),
    pytest.param(load_pack("tests.unit.knowledge.ontologies.fixtures.base_pack"), id="base"),
    pytest.param(load_pack("tests.unit.knowledge.ontologies.fixtures.pet_pack"), id="pet"),
]


@pytest.mark.parametrize("pack", PACKS)
def test_the_generated_schema_uses_only_keywords_the_converter_reads(pack):
    assert _keywords(_generate(pack)) <= CONVERTER_KEYWORDS


def test_a_conditional_is_caught():
    schema = _generate(registry.active_pack())
    schema["properties"]["entities"]["items"]["if"] = {"properties": {}}

    assert _keywords(schema) - CONVERTER_KEYWORDS == {"if"}


def test_property_names_and_enum_values_are_not_read_as_keywords():
    schema = {
        "type": "object",
        "properties": {"if": {"type": "string"}},
        "required": ["then"],
        "additionalProperties": True,
    }

    assert _keywords(schema) <= CONVERTER_KEYWORDS
