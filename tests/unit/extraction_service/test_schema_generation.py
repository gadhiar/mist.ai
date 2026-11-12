"""Stage 2's output schema is generated from a pack's extractable declarations.

An entity's `type` and a relationship's `type` are enums of the pack's extractable lists.
A type that declares a required property the model writes gets an `anyOf` branch of its
own, its `type` a `const`, stating its properties as the pack declares them; every other
type shares the rest branch. `EXTRACTION_OUTPUT_SCHEMA` stays as the schema for callers
that hold no pack, and is pinned here. No name of a pack is written into `schemas.py`.
"""

from __future__ import annotations

import ast
import dataclasses
import json
from pathlib import Path

import pytest

from backend.errors import MistError
from backend.extraction_service import schemas
from backend.extraction_service.schemas import (
    EXTRACTION_OUTPUT_SCHEMA,
    SchemaGenerationError,
    generate_extraction_schema,
)
from backend.knowledge.ontologies import registry
from backend.knowledge.ontologies.base import KnowledgeDomain, NodeTypeDefinition
from backend.knowledge.ontologies.base import PropertyDefinition as Prop
from backend.knowledge.ontologies.pack import Pack, load_pack

BASE = load_pack("tests.unit.knowledge.ontologies.fixtures.base_pack")
PET = load_pack("tests.unit.knowledge.ontologies.fixtures.pet_pack")
PACKS = [pytest.param(registry.active_pack(), id="active"), pytest.param(BASE, id="base")]
PACKS.append(pytest.param(PET, id="pet"))


def _definitions(pack: Pack) -> tuple[NodeTypeDefinition, ...]:
    by_name = {node.type_name: node for node in pack.node_types}
    return tuple(by_name[name] for name in pack.extractable_node_types)


def _generate(pack: Pack) -> dict:
    return generate_extraction_schema(
        entity_types=_definitions(pack), relationship_types=pack.extractable_edge_types
    )


def _branches(schema: dict) -> list[dict]:
    items = schema["properties"]["entities"]["items"]
    return items["anyOf"] if "anyOf" in items else [items]


def _node(name: str, *required: Prop, optional: tuple[Prop, ...] = ()) -> NodeTypeDefinition:
    return NodeTypeDefinition(
        type_name=name,
        description="",
        knowledge_domain=KnowledgeDomain.EXTERNAL,
        required_properties=required,
        optional_properties=optional,
    )


# --- the active pack's shape ------------------------------------------------------------


def test_the_active_pack_s_metric_branch_and_rest_branch():
    pack = registry.active_pack()
    schema = _generate(pack)
    items = schema["properties"]["entities"]["items"]
    metric, rest = items["anyOf"]

    assert metric["properties"]["type"] == {"const": "Metric"}
    assert list(metric["properties"]) == ["id", "name", "type", "properties"]
    assert metric["properties"]["id"] == metric["properties"]["name"] == {"type": "string"}
    assert metric["required"] == ["id", "name", "type"], "a Metric need not carry properties"
    keys = metric["properties"]["properties"]
    assert keys["properties"] == {
        "value": {"type": "number"},
        "unit": {"type": ["string", "null"]},
        "quantity": {"type": ["string", "null"]},
    }
    assert list(keys["properties"]) == ["value", "unit", "quantity"]
    assert keys["required"] == ["value", "unit", "quantity"]
    assert keys["additionalProperties"] is True
    others = [name for name in pack.extractable_node_types if name != "Metric"]
    assert rest["properties"]["type"] == {"enum": others}
    assert "Abstraction" in others, "the fallback type stays in the enum"
    assert list(rest["properties"]) == ["id", "name", "type"]
    assert rest["required"] == ["id", "name", "type"]


def test_the_active_pack_s_relationships_require_source_target_and_a_listed_type():
    pack = registry.active_pack()
    items = _generate(pack)["properties"]["relationships"]["items"]

    assert list(items["properties"]) == ["source", "target", "type"]
    assert items["properties"]["source"] == items["properties"]["target"] == {"type": "string"}
    assert items["properties"]["type"] == {"enum": list(pack.extractable_edge_types)}
    assert "RELATED_TO" in items["properties"]["type"]["enum"]
    assert items["required"] == ["source", "target", "type"]


@pytest.mark.parametrize("pack", PACKS)
def test_every_object_keeps_additional_properties_and_no_conditional(pack):
    def walk(node):
        if isinstance(node, dict):
            assert not {"if", "then", "else"} & set(node)
            if node.get("type") == "object":
                assert node["additionalProperties"] is True
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(_generate(pack))


@pytest.mark.parametrize("pack", [BASE, PET], ids=["base", "pet"])
def test_a_fixture_pack_has_one_branch_and_no_any_of(pack):
    schema = _generate(pack)
    items = schema["properties"]["entities"]["items"]

    assert "anyOf" not in items
    assert items["properties"]["type"] == {"enum": list(pack.extractable_node_types)}
    assert pack.extraction_schema == schema


# --- the names come from the pack (Acceptance 5) ----------------------------------------


@pytest.mark.parametrize("pack", PACKS)
def test_the_enums_and_the_branch_keys_read_back_from_the_pack(pack):
    schema = _generate(pack)
    branch_types: list[str] = []
    for branch in _branches(schema):
        stated = branch["properties"]["type"]
        names = [stated["const"]] if "const" in stated else stated["enum"]
        branch_types += names
        if "const" in stated:
            (node,) = (n for n in _definitions(pack) if n.type_name == stated["const"])
            declared = [p.name for p in (*node.required_properties, *node.optional_properties)]
            written = [p.name for p in node.required_properties if p.extracted]
            keys = branch["properties"]["properties"]
            assert keys["required"] == written
            assert set(keys["properties"]) <= set(declared)
    assert sorted(branch_types) == sorted(pack.extractable_node_types)
    assert len(branch_types) == len(set(branch_types)), "each type is in one branch"
    edges = schema["properties"]["relationships"]["items"]["properties"]["type"]["enum"]
    assert edges == list(pack.extractable_edge_types)
    assert pack.extraction_schema == schema


def test_schemas_py_names_no_type_edge_or_metric_property_of_the_pack():
    pack = registry.active_pack()
    tree = ast.parse(Path(schemas.__file__).read_text(encoding="utf-8"))
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Module | ast.FunctionDef | ast.ClassDef)
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    strings = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    }
    metric = next(node for node in pack.node_types if node.type_name == "Metric")
    names = {
        *(node.type_name for node in pack.node_types),
        *(edge.type_name for edge in pack.edge_types),
        *(p.name for p in (*metric.required_properties, *metric.optional_properties)),
    }

    assert not strings & names


# --- the generator's rules, on invented types ---------------------------------------------


def test_a_branch_states_its_written_keys_required_first_and_optional_after():
    tide = Prop(name="tide", type="int", required=True, extracted=True)
    note = Prop(name="note", type="string", nullable=True, extracted=True)
    berth = Prop(name="berth", type="string", required=True)
    gauge = _node("Gauge", tide, berth, optional=(note,))

    schema = generate_extraction_schema(
        entity_types=(_node("Deckhand"), gauge, _node("Mooring")), relationship_types=("MOORS",)
    )
    gauge_branch, rest = schema["properties"]["entities"]["items"]["anyOf"]

    assert gauge_branch["properties"]["type"] == {"const": "Gauge"}
    assert gauge_branch["properties"]["properties"]["properties"] == {
        "tide": {"type": "integer"},
        "note": {"type": ["string", "null"]},
    }
    assert gauge_branch["properties"]["properties"]["required"] == ["tide"]
    assert rest["properties"]["type"] == {"enum": ["Deckhand", "Mooring"]}


def test_with_every_type_a_branch_type_there_is_no_rest_branch():
    tide = Prop(name="tide", type="float", required=True, extracted=True)

    schema = generate_extraction_schema(
        entity_types=(_node("Gauge", tide),), relationship_types=("MOORS",)
    )
    items = schema["properties"]["entities"]["items"]

    assert "anyOf" not in items
    assert items["properties"]["type"] == {"const": "Gauge"}


def test_a_written_key_of_a_type_with_no_json_type_does_not_generate():
    tide = Prop(name="tide", type="datetime", required=True, extracted=True)

    with pytest.raises(SchemaGenerationError, match="Gauge.tide.*datetime"):
        generate_extraction_schema(entity_types=(_node("Gauge", tide),), relationship_types=("M",))


def test_each_call_returns_a_schema_of_its_own():
    first, second = _generate(BASE), _generate(BASE)
    first["properties"]["entities"]["items"]["additionalProperties"] = False

    assert second == _generate(BASE)
    assert first != second


def test_the_schema_as_it_stood_is_pinned():
    assert json.dumps(EXTRACTION_OUTPUT_SCHEMA) == (
        '{"type": "object", "properties": {"entities": {"type": "array", "items": '
        '{"type": "object", "additionalProperties": true}}, "relationships": {"type": '
        '"array", "items": {"type": "object", "additionalProperties": true}}}, '
        '"required": ["entities", "relationships"], "additionalProperties": true}'
    )


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
    nodes = tuple(_node(name) for name in entity_types)

    with pytest.raises(SchemaGenerationError, match=message):
        generate_extraction_schema(entity_types=nodes, relationship_types=relationship_types)


def test_the_generation_error_is_a_mist_error():
    assert issubclass(SchemaGenerationError, MistError)


def test_a_property_declaration_states_null_and_model_written_keys_only_when_set():
    unset = Prop(name="berth", type="string")

    assert (unset.nullable, unset.extracted) == (None, None)
    assert dataclasses.replace(unset, nullable=True, extracted=True).nullable is True
