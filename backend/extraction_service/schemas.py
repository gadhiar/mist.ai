"""Permissive JSON schemas for constrained decoding of the three stage outputs.

These are deliberately loose: each schema must accept every shape the
corresponding parser in `backend.knowledge.extraction` already accepts
(`parse_scope_output`, `parse_extraction_output`, `parse_derivation_output`).
`additionalProperties: True` throughout so a model that emits an extra
field (e.g. a "reasoning" aside, or a future ontology property) is not
rejected by the constrained-decoding grammar. Used when
`ServiceSettings.constrained_mode` (or the active adapter's
`default_constrained_mode`) resolves to `"schema"`.

Stage 2's schema is generated from a pack's extractable names
(`generate_extraction_schema`); the pack loader calls it and holds the result,
which the engine sends. The schema names no type yet, so every pack's is
`EXTRACTION_OUTPUT_SCHEMA`, the schema as it stood, which stays here as the
pin the generated one is tested against and for the callers that hold no pack.
"""

from collections.abc import Sequence

from backend.errors import MistError


class SchemaGenerationError(MistError):
    """A pack's names give no schema: the message says which name and why."""


SCOPE_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "scope": {"type": "string"},
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": ["scope"],
    "additionalProperties": True,
}

EXTRACTION_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {"type": "object", "additionalProperties": True},
        },
        "relationships": {
            "type": "array",
            "items": {"type": "object", "additionalProperties": True},
        },
    },
    "required": ["entities", "relationships"],
    "additionalProperties": True,
}


def _require_enum_members(what: str, names: Sequence[str]) -> None:
    """Refuse `names` unless they could stand as the members of a JSON Schema enum.

    Raises:
        SchemaGenerationError: There is no name, a name is empty or not a string, or
            a name is listed twice.
    """
    if not names:
        raise SchemaGenerationError(f"no extractable {what} type")
    seen: set[str] = set()
    for name in names:
        if not isinstance(name, str) or not name:
            raise SchemaGenerationError(f"an empty {what} type")
        if name in seen:
            raise SchemaGenerationError(f"{what} type {name} is listed twice")
        seen.add(name)


def generate_extraction_schema(
    *, entity_types: Sequence[str], relationship_types: Sequence[str]
) -> dict:
    """Stage 2's output schema for a pack whose model may produce these names.

    The schema is permissive and names no type, so the result does not yet depend
    on the names. They are checked all the same, as the members of the two enums a
    tighter schema states: a pack this refuses could not have that schema either, so
    tightening the schema later refuses no pack that loads now.

    Args:
        entity_types: The pack's extractable node types.
        relationship_types: The pack's extractable edge types.

    Returns:
        A schema of the caller's own: no two calls share a mapping.

    Raises:
        SchemaGenerationError: Either list is empty, or holds an empty or a repeated
            name.
    """
    _require_enum_members("entity", entity_types)
    _require_enum_members("relationship", relationship_types)
    return {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {"type": "object", "additionalProperties": True},
            },
            "relationships": {
                "type": "array",
                "items": {"type": "object", "additionalProperties": True},
            },
        },
        "required": ["entities", "relationships"],
        "additionalProperties": True,
    }


DERIVATION_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "operations": {
            "type": "array",
            "items": {"type": "object", "additionalProperties": True},
        },
    },
    "required": ["operations"],
    "additionalProperties": True,
}
