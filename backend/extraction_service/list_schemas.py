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

import copy
from collections.abc import Collection, Sequence
from dataclasses import dataclass

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


@dataclass(frozen=True, slots=True)
class ItemSlot:
    """What one asked item's own entity must be, from the parsed frame.

    `list_frames.item_slots` computes it, and both the item call's built schema and the
    item parser read it, so the grammar and the parser agree. It lives here, as plain
    data, because `list_frames` imports this module.

    Attributes:
        item: The item's id, `G1.2`.
        nested: The item's group has a parent, so `parent` is an end of its edges.
        applying_types: The types an entity may take when `applies` is true; None
            when no template of the group names `item`, or the item is an unchecked
            box, which templates never apply to.
        required: An entity is required whatever `applies` says: the item is an
            unchecked goal box, the parent of a group whose template names `parent`,
            or an item a frame fact names.
        required_types: The types that requirement admits; None for any.
    """

    item: str
    nested: bool
    applying_types: tuple[str, ...] | None
    required: bool
    required_types: tuple[str, ...] | None


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


# -- per-call schemas -----------------------------------------------------------------------
#
# Each call's schema narrows its static superset from the turn's groups and items and,
# for an item call, from the item slots the parsed frame gives, so a shape the parsers
# would refuse cannot be decoded. A tuple is a `prefixItems` array bounded by
# `minItems` and `maxItems`, with no `items` beside it: llama.cpp's converter may read
# `items` first and ignore the tuple (the plan review's recollection, not checked
# against the host's build). The static supersets stay the prompt skeleton's source and
# the prompt hash's input.

_TEXT = {"type": "string", "minLength": 1}
_ROLES = ("user", "mist", "item")


def _const(value: str | bool) -> dict:
    return {"type": "boolean" if isinstance(value, bool) else "string", "const": value}


def _tuple(entries: list[dict]) -> dict:
    count = len(entries)
    return {"type": "array", "prefixItems": entries, "minItems": count, "maxItems": count}


def _types(types: Sequence[str] | None, item: str) -> dict:
    if types is not None and not types:
        raise ValueError(f"{item}: an entity of no type cannot be decoded")
    return _enum(EXTRACTABLE_TYPE_ENUM if types is None else tuple(types))


def _keyed(keys: tuple[str, ...]) -> dict:
    """`_ENTITY` or `_FRAME_ENTITY` with the key set narrowed and non-empty texts."""
    return _object(
        {"key": _enum(keys), "span": _TEXT, "name": _TEXT, "type": _TYPE},
        {"activity": _BOOLEAN, "value": _STRING, "unit": _STRING, "quantity": _STRING},
    )


def _relationship(ends: Sequence[str]) -> dict:
    end = _enum(tuple(ends))
    return _object({"source": end, "target": end, "type": _PREDICATE}, {"properties": _PROPERTIES})


def _item_refs(item_ids: Sequence[str]) -> tuple[str, ...]:
    return tuple(f"item:{item}" for item in item_ids)


def list_frame_output_schema(
    group_ids: Sequence[str], nested_ids: Collection[str], item_ids: Sequence[str]
) -> dict:
    """The frame call's schema for one turn.

    One `groups` entry per group, in Units order, each `group` a const; frame keys `f1`
    to `f20` and facts keys `e1` to `e20`; template ends `user`, `mist`, `item`, the
    frame keys and, in a nested group only, `parent`; fact ends `user`, `mist`, both key
    sets and `item:<id>` for each item of the turn; non-empty spans and names.

    Args:
        group_ids: The turn's group ids, in order.
        nested_ids: The ids of the groups that have a parent.
        item_ids: Every item id of the turn.
    """
    frame_entity = _object(
        {"key": _enum(FRAME_KEY_ENUM), "span": _TEXT, "name": _TEXT, "type": _TYPE}
    )

    def group(gid: str) -> dict:
        parent = ("parent",) if gid in nested_ids else ()
        role = _enum((*_ROLES, *parent, *FRAME_KEY_ENUM))
        template = _object({"subject": role, "predicate": _PREDICATE, "object": role})
        return _object(
            {
                "group": _const(gid),
                "durable": _BOOLEAN,
                "templates": _array(template, max_items=3),
                "item_type": _GROUP["properties"]["item_type"],
                "unchecked": _enum(UNCHECKED_ENUM),
            }
        )

    fact_ends = ("user", "mist", *FRAME_KEY_ENUM, *ENTITY_KEY_ENUM, *_item_refs(item_ids))
    facts = _object(
        {
            "entities": _array(_keyed(ENTITY_KEY_ENUM)),
            "relationships": _array(_relationship(fact_ends)),
        }
    )
    schema = _object(
        {
            "frame_entities": _array(frame_entity),
            "groups": _tuple([group(gid) for gid in group_ids]),
            "facts": facts,
        }
    )
    return copy.deepcopy(schema)


def list_item_output_schema(
    slots: Sequence[ItemSlot], frame_keys: Sequence[str], item_ids: Sequence[str]
) -> dict:
    """An item call's schema for one batch, its entries following each item's slot.

    One `items` entry per asked item, in order, each `item` a const; own keys `e1` to
    `e20`; edge ends `user`, `mist`, `item`, the declared frame keys, `e1` to `e20`,
    `item:<id>` for each item of the turn and, for an item of a nested group, `parent`.
    An item whose slot has `applying_types` gets two branches that `template.applies`
    picks, which is why `template` precedes `entity`: true requires an entity of an
    admitted type, false leaves it as the slot's requirement does. A required slot
    requires an entity in every branch, limited to `required_types`.

    Args:
        slots: The asked items' slots, from `list_frames.item_slots`, in order.
        frame_keys: The frame entities' keys.
        item_ids: Every item id of the turn.

    Raises:
        ValueError: A slot admits no type, so its entity could not be decoded.
    """
    refs = _item_refs(item_ids)
    entities = _array(_keyed(ENTITY_KEY_ENUM))

    def entity(types: Sequence[str] | None, item: str) -> dict:
        return _object(
            {"span": _TEXT, "name": _TEXT, "type": _types(types, item)},
            {"activity": _BOOLEAN, "value": _STRING, "unit": _STRING, "quantity": _STRING},
        )

    def template(applies: bool | None) -> dict:
        flag = _BOOLEAN if applies is None else _const(applies)
        return _object(
            {
                "applies": flag,
                "assertion_kind": _ASSERTION_KIND,
                "temporal_status": _TEMPORAL_STATUS,
            },
            {"start_date": _STRING, "end_date": _STRING},
        )

    def record(slot: ItemSlot, applies: bool | None, own: dict) -> dict:
        parent = ("parent",) if slot.nested else ()
        ends = (*_ROLES, *parent, *frame_keys, *ENTITY_KEY_ENUM, *refs)
        return _object(
            {
                "item": _const(slot.item),
                "template": template(applies),
                "entity": own,
                "entities": entities,
                "relationships": _array(_relationship(ends)),
            }
        )

    entries = []
    for slot in slots:
        required_types = slot.required_types if slot.required else None
        if slot.required:
            otherwise = entity(required_types, slot.item)
        else:
            otherwise = {"anyOf": [entity(None, slot.item), {"type": "null"}]}
        if slot.applying_types is None:
            entries.append(record(slot, None, otherwise))
            continue
        if not slot.applying_types:
            raise ValueError(f"{slot.item}: the templates admit no type")
        applying = [t for t in slot.applying_types if required_types is None or t in required_types]
        if not applying:
            raise ValueError(f"{slot.item}: the applying and required types do not meet")
        branches = [record(slot, True, entity(applying, slot.item))]
        branches.append(record(slot, False, otherwise))
        entries.append({"anyOf": branches})
    return copy.deepcopy(_object({"items": _tuple(entries)}))
