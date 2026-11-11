"""The turn plan: one call over a whole structured turn, made before its batches.

Stage 2 extracts a structured turn a few units at a time, so each batch
decides on its own what a lead-in means and which id an entity gets, and two
batches of one list can disagree. The turn plan makes those decisions once,
with the whole turn in view: the entities the turn names, one template per
heading or lead-in that states a relationship, and the lines that refer
back. Every batch of the turn then carries the plan in front of its own text.

This module holds the plan's text and shape and nothing that calls a model:
the request the plan call sends, the strict schema its reply is decoded
under, the strict parser, and the block a batch carries. The engine
(`backend.extraction_service.engine`) makes the call, behind
`ServiceSettings.turn_plan`, and hashes every string here and the schema
into `prompt_sha256` when the plan is on.

The entity types and relationship types the plan may name are read from the
ontology's derived extractable lists. No type or relationship name is
written in this module, so an ontology change reaches the plan without an
edit here.

The schema stays inside the subset of JSON Schema that llama.cpp's
schema-to-grammar converter handles without surprises: objects whose every
property is required and that allow no others, arrays, strings, and enums of
strings. A value the turn does not give is an empty string or an empty list,
never null or an absent key.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from backend.errors import ExtractionValidationError
from backend.knowledge.extraction.structure_splitter import InputUnit, render_batch
from backend.knowledge.ontologies import EXTRACTABLE_NODE_TYPES, EXTRACTABLE_RELATIONSHIP_TYPES

# The plan call's user message. It opens like a batch's user message (context,
# then subject scope) and then carries every unit of the turn, one per line,
# rendered as a batch renders them. It names no entity type or relationship
# type: it points at the lists the system prompt gives.
PLAN_REQUEST_TEMPLATE = """Context:
{context}
Subject scope: {subject_scope}
Message, one line per unit:
{units}

Do not extract this message yet. It is a structured turn (Rule 28) that is extracted a few lines at a time, and every line has to be read the same way. Plan that extraction from the whole message. Return ONLY a JSON object with these three lists, in place of the entities and relationships output:
- "entities": every entity the message names, each one once, and never the user. Each has the "id" Rules 2 and 28 give it, its "name" and its "type", one of the allowed entity types above.
- "templates": one for each heading or lead-in that gives the lines under it a relationship. A heading or lead-in that only names or groups its lines gets no template. "lead_in" is its text as written. "subject", "predicate" and "fixed_endpoint" state the relationship the lead-in itself gives: "subject" is "user" or an id from "entities", "predicate" is one of the allowed relationship types above, and "fixed_endpoint" is the id of the entity the lead-in names as the other end, or "" when the other end is the entity of each line under it. "fan_out" lists every further relationship each line under the lead-in gets from it, as "source", "predicate" and "target": write "each" for the line's own entity, and "user" or an id from "entities" for the other end.
- "back_references": one for each line that refers back to an earlier line. "item" is the line's text as written and "refers_to" is the id, from "entities", of what it refers to.
Leave a list empty when the message gives nothing for it.

Output:"""

# What stands for the entity of each line under a lead-in, in a template.
EACH_ITEM = "each"

# The plan block: the text every batch of a planned turn carries in front of
# its own user message. Each string is a constant so that the engine can hash
# all of them into `prompt_sha256`.
PLAN_BLOCK_HEADER = (
    "Turn plan, made from the whole message before its lines were batched. The lines of the "
    "Utterance below belong to that message. Use the plan's id and type for every entity it "
    "lists, and give each line the relationships its lead-in's template states."
)
PLAN_BLOCK_ENTITIES_HEADING = "Entities named in this message (id | name | type):"
PLAN_BLOCK_ENTITY_LINE = "- {id} | {name} | {type}"
PLAN_BLOCK_TEMPLATES_HEADING = (
    "Templates, one per lead-in or heading that states a relationship "
    '("each" is the entity of each line under it):'
)
PLAN_BLOCK_TEMPLATE_LINE = "- {lead_in}: {relationships}"
PLAN_BLOCK_RELATIONSHIP = "{source} {predicate} {target}"
PLAN_BLOCK_RELATIONSHIP_JOIN = "; "
PLAN_BLOCK_BACK_REFERENCES_HEADING = "Lines that refer back (line -> id):"
PLAN_BLOCK_BACK_REFERENCE_LINE = "- {item} -> {refers_to}"
PLAN_BLOCK_NONE = "(none)"
PLAN_BLOCK_FOOTER = "End of the turn plan."
PLAN_BLOCK_SEPARATOR = "\n\n"

# Every constant string of the plan block, in a fixed order, for `prompt_sha256`.
PLAN_BLOCK_CONSTANTS: tuple[str, ...] = (
    EACH_ITEM,
    PLAN_BLOCK_HEADER,
    PLAN_BLOCK_ENTITIES_HEADING,
    PLAN_BLOCK_ENTITY_LINE,
    PLAN_BLOCK_TEMPLATES_HEADING,
    PLAN_BLOCK_TEMPLATE_LINE,
    PLAN_BLOCK_RELATIONSHIP,
    PLAN_BLOCK_RELATIONSHIP_JOIN,
    PLAN_BLOCK_BACK_REFERENCES_HEADING,
    PLAN_BLOCK_BACK_REFERENCE_LINE,
    PLAN_BLOCK_NONE,
    PLAN_BLOCK_FOOTER,
    PLAN_BLOCK_SEPARATOR,
)


def _string() -> dict[str, Any]:
    return {"type": "string"}


def _one_of(values: Sequence[str]) -> dict[str, Any]:
    return {"type": "string", "enum": list(values)}


def _array_of(items: dict[str, Any]) -> dict[str, Any]:
    return {"type": "array", "items": items}


def _strict_object(properties: dict[str, Any]) -> dict[str, Any]:
    """An object with exactly `properties`: all required, no others allowed."""
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


# The plan call's output schema. The property order is the order the model
# writes them in under the grammar: the roster first, so the templates and the
# back references can name its ids.
PLAN_OUTPUT_SCHEMA: dict[str, Any] = _strict_object(
    {
        "entities": _array_of(
            _strict_object(
                {
                    "id": _string(),
                    "name": _string(),
                    "type": _one_of(EXTRACTABLE_NODE_TYPES),
                }
            )
        ),
        "templates": _array_of(
            _strict_object(
                {
                    "lead_in": _string(),
                    "subject": _string(),
                    "predicate": _one_of(EXTRACTABLE_RELATIONSHIP_TYPES),
                    "fixed_endpoint": _string(),
                    "fan_out": _array_of(
                        _strict_object(
                            {
                                "source": _string(),
                                "predicate": _one_of(EXTRACTABLE_RELATIONSHIP_TYPES),
                                "target": _string(),
                            }
                        )
                    ),
                }
            )
        ),
        "back_references": _array_of(
            _strict_object(
                {
                    "item": _string(),
                    "refers_to": _string(),
                }
            )
        ),
    }
)


def is_structured_turn(units: Sequence[InputUnit]) -> bool:
    """True unless `units` is the one context-free prose unit of a turn without structure.

    `split_structured` returns exactly that unit for text with no list,
    table, heading or lead-in, and such a turn has nothing to plan.
    """
    if len(units) != 1:
        return True
    (unit,) = units
    return unit.kind != "prose" or bool(unit.context)


def render_plan_request(units: Sequence[InputUnit], *, context: str, subject_scope: str) -> str:
    """The plan call's user message: the whole turn, then what to plan.

    Args:
        units: Every unit of the turn, unbatched.
        context: The conversation context, as a batch's user message shows it.
        subject_scope: The turn's Stage 1.5 scope label.
    """
    return PLAN_REQUEST_TEMPLATE.format(
        context=context,
        subject_scope=subject_scope,
        units=render_batch(units),
    )


def parse_turn_plan(raw: str) -> dict[str, Any]:
    """Parse the plan call's reply, accepting exactly what `PLAN_OUTPUT_SCHEMA` allows.

    The check walks the schema itself, so the parser and the grammar the
    reply was decoded under cannot drift apart, and a reply decoded without
    the grammar (constrained mode "json_object" or "none") is held to the
    same shape. An error names where the reply went wrong, as a path from
    `plan`, and never quotes the reply, because the turn is a real
    conversation.

    Raises:
        ExtractionValidationError: `raw` is not JSON, or the parsed value
            does not match the schema.
    """
    try:
        plan = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ExtractionValidationError(f"plan is not JSON: {exc}") from exc
    _check(plan, PLAN_OUTPUT_SCHEMA, "plan")
    return plan


def _check(value: Any, schema: Mapping[str, Any], path: str) -> None:
    """Raise unless `value` matches `schema`, one of the three kinds the plan schema uses."""
    kind = schema["type"]
    if kind == "object":
        if not isinstance(value, dict):
            raise ExtractionValidationError(f"{path} is not an object")
        properties: Mapping[str, Any] = schema["properties"]
        missing = sorted(set(properties) - set(value))
        if missing:
            raise ExtractionValidationError(f"{path} lacks {', '.join(missing)}")
        if len(value) != len(properties):
            raise ExtractionValidationError(f"{path} has a field the schema does not")
        for name, property_schema in properties.items():
            _check(value[name], property_schema, f"{path}.{name}")
    elif kind == "array":
        if not isinstance(value, list):
            raise ExtractionValidationError(f"{path} is not an array")
        for index, item in enumerate(value):
            _check(item, schema["items"], f"{path}[{index}]")
    else:
        if not isinstance(value, str):
            raise ExtractionValidationError(f"{path} is not a string")
        if "enum" in schema and value not in schema["enum"]:
            raise ExtractionValidationError(f"{path} is not one of its allowed values")


def _one_line(value: str) -> str:
    """`value` on one line, so a plan value cannot break the block's line structure."""
    return " ".join(value.split())


def _relationship(source: str, predicate: str, target: str) -> str:
    return PLAN_BLOCK_RELATIONSHIP.format(
        source=_one_line(source), predicate=predicate, target=_one_line(target)
    )


def _template_line(template: Mapping[str, Any]) -> str:
    """One template: the lead-in's own relationship, then each fan-out relationship."""
    relationships = [
        _relationship(
            template["subject"],
            template["predicate"],
            template["fixed_endpoint"] or EACH_ITEM,
        )
    ]
    relationships += [
        _relationship(edge["source"], edge["predicate"], edge["target"])
        for edge in template["fan_out"]
    ]
    return PLAN_BLOCK_TEMPLATE_LINE.format(
        lead_in=_one_line(template["lead_in"]),
        relationships=PLAN_BLOCK_RELATIONSHIP_JOIN.join(relationships),
    )


def render_plan_block(plan: Mapping[str, Any]) -> str:
    """The text every batch of the turn carries in front of its user message.

    It depends on the plan alone, so it is identical across the batches of
    one turn and llama-server's cached prefix runs through it. An empty list
    renders as `PLAN_BLOCK_NONE`, so the three headings are always there.

    Args:
        plan: A plan `parse_turn_plan` returned.
    """
    entities = [
        PLAN_BLOCK_ENTITY_LINE.format(
            id=_one_line(entity["id"]), name=_one_line(entity["name"]), type=entity["type"]
        )
        for entity in plan["entities"]
    ]
    templates = [_template_line(template) for template in plan["templates"]]
    back_references = [
        PLAN_BLOCK_BACK_REFERENCE_LINE.format(
            item=_one_line(reference["item"]), refers_to=_one_line(reference["refers_to"])
        )
        for reference in plan["back_references"]
    ]
    lines = [
        PLAN_BLOCK_HEADER,
        PLAN_BLOCK_ENTITIES_HEADING,
        *(entities or [PLAN_BLOCK_NONE]),
        PLAN_BLOCK_TEMPLATES_HEADING,
        *(templates or [PLAN_BLOCK_NONE]),
        PLAN_BLOCK_BACK_REFERENCES_HEADING,
        *(back_references or [PLAN_BLOCK_NONE]),
        PLAN_BLOCK_FOOTER,
    ]
    return "\n".join(lines) + PLAN_BLOCK_SEPARATOR
