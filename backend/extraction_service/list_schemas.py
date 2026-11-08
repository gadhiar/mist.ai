"""Strict JSON schemas for the list path's frame and item calls.

Unlike the three permissive stage schemas in `schemas.py`, these close every object
(`additionalProperties: False`) and carry `enum` on every type and predicate field, read
from the ontology, so constrained decoding can only emit an extractable type or
predicate. The prose schema stays open and enum-free, so the prose path and the r6
fallback send today's request.

Each object declares its keys in one fixed order, required keys first and the usually
null keys optional and last, which keeps a four-item output short. Key order is part of
the contract: the grammar emits declared keys in declared order, and the list prompt's
output section is generated from these schemas so it names the keys in that order.
"""

from __future__ import annotations

from backend.knowledge.ontologies import EXTRACTABLE_NODE_TYPES, EXTRACTABLE_RELATIONSHIP_TYPES

EXTRACTABLE_TYPE_ENUM: tuple[str, ...] = tuple(EXTRACTABLE_NODE_TYPES)
PREDICATE_ENUM: tuple[str, ...] = tuple(EXTRACTABLE_RELATIONSHIP_TYPES)

TEMPORAL_STATUS_ENUM: tuple[str, ...] = ("current", "past", "future")
ASSERTION_KIND_ENUM: tuple[str, ...] = ("assert", "cease", "retract")
UNCHECKED_ENUM: tuple[str, ...] = ("plan", "goal")

# The keys a list call may give an entity: `f` keys for the frame call's frame entities,
# `e` keys for the frame's facts and an item's own entities, so the two sets never
# overlap and an item's own key never shadows a frame key. No measure bounds how many a
# turn needs; twenty of each leaves room.
FRAME_KEY_ENUM: tuple[str, ...] = tuple(f"f{n}" for n in range(1, 21))
ENTITY_KEY_ENUM: tuple[str, ...] = tuple(f"e{n}" for n in range(1, 21))


def _object(required: dict[str, dict], optional: dict[str, dict] | None = None) -> dict:
    return {
        "type": "object",
        "properties": required | (optional or {}),
        "required": list(required),
        "additionalProperties": False,
    }


def _array(items: dict, max_items: int | None = None) -> dict:
    schema: dict = {"type": "array", "items": items}
    if max_items is not None:
        schema["maxItems"] = max_items
    return schema


def _enum(values: tuple[str, ...]) -> dict:
    return {"type": "string", "enum": list(values)}


_STRING = {"type": "string"}
_BOOLEAN = {"type": "boolean"}
_TYPE = _enum(EXTRACTABLE_TYPE_ENUM)
_PREDICATE = _enum(PREDICATE_ENUM)
_TEMPORAL_STATUS = _enum(TEMPORAL_STATUS_ENUM)
_ASSERTION_KIND = _enum(ASSERTION_KIND_ENUM)

# The main prompt's relationship property keys, in its order (`prompts.py`, OUTPUT
# SCHEMA). All optional: an item's template carries the frame's temporal fields, so most
# list relationships need none.
_PROPERTIES = _object(
    {},
    {
        "confidence": {"type": "number"},
        "temporal_status": _TEMPORAL_STATUS,
        "start_date": _STRING,
        "end_date": _STRING,
        "temporal_expression": _STRING,
        "context": _STRING,
        "negated": _BOOLEAN,
        "assertion_kind": _ASSERTION_KIND,
    },
)

_RELATIONSHIP = _object(
    {"source": _STRING, "target": _STRING, "type": _PREDICATE},
    {"properties": _PROPERTIES},
)

_ENTITY = _object(
    {"key": _STRING, "span": _STRING, "name": _STRING, "type": _TYPE},
    {"activity": _BOOLEAN, "value": _STRING, "unit": _STRING, "quantity": _STRING},
)

_FACTS = _object({"entities": _array(_ENTITY), "relationships": _array(_RELATIONSHIP)})

_FRAME_ENTITY = _object({"key": _STRING, "span": _STRING, "name": _STRING, "type": _TYPE})

_TEMPLATE = _object({"subject": _STRING, "predicate": _PREDICATE, "object": _STRING})

_GROUP = _object(
    {
        "group": _STRING,
        "durable": _BOOLEAN,
        "templates": _array(_TEMPLATE, max_items=3),
        "item_type": {"type": ["string", "null"], "enum": [*EXTRACTABLE_TYPE_ENUM, None]},
        "unchecked": _enum(UNCHECKED_ENUM),
    }
)

LIST_FRAME_OUTPUT_SCHEMA: dict = _object(
    {"frame_entities": _array(_FRAME_ENTITY), "groups": _array(_GROUP), "facts": _FACTS}
)

_ITEM_ENTITY = _object(
    {"span": _STRING, "name": _STRING, "type": _TYPE},
    {"activity": _BOOLEAN, "value": _STRING, "unit": _STRING, "quantity": _STRING},
)

_ITEM_TEMPLATE = _object(
    {"applies": _BOOLEAN, "assertion_kind": _ASSERTION_KIND, "temporal_status": _TEMPORAL_STATUS},
    {"start_date": _STRING, "end_date": _STRING},
)

# `template` comes before `entity`: the item says whether the group's templates apply
# before it names its entity, so a built schema can let `applies` pick the branch that
# then requires and narrows the entity, rather than the entity forcing `applies: false`.
_ITEM = _object(
    {
        "item": _STRING,
        "template": _ITEM_TEMPLATE,
        "entity": {"anyOf": [_ITEM_ENTITY, {"type": "null"}]},
        "entities": _array(_ENTITY),
        "relationships": _array(_RELATIONSHIP),
    }
)

LIST_ITEM_OUTPUT_SCHEMA: dict = _object({"items": _array(_ITEM)})
