"""JSON schemas for constrained decoding of the three stage outputs.

The scope and derivation schemas are loose: each accepts every shape its parser in
`backend.knowledge.extraction` accepts (`parse_scope_output`, `parse_derivation_output`).
Every object keeps `additionalProperties: True`, so a model that emits an extra field (a
"reasoning" aside, a property no type declares) is not rejected by the
constrained-decoding grammar. Used when `ServiceSettings.constrained_mode` (or the active
adapter's `default_constrained_mode`) resolves to `"schema"`.

Stage 2's schema is generated from a pack's extractable declarations
(`generate_extraction_schema`); the pack loader calls it and holds the result, which the
engine sends. It limits each entity's and relationship's `type` to the pack's extractable
lists and states the keys the pack declares the model writes. `EXTRACTION_OUTPUT_SCHEMA`
is the permissive schema as it stood before generation, kept for the callers that hold
no pack.
"""

from collections.abc import Sequence

from backend.errors import MistError
from backend.knowledge.ontologies.base import NodeTypeDefinition, PropertyDefinition


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


# Declared property types the schema can state, and the JSON type each is written as.
_JSON_TYPES = {"float": "number", "int": "integer", "string": "string"}


def _written(node: NodeTypeDefinition) -> tuple[PropertyDefinition, ...]:
    """The properties of `node` the model writes: the required ones first, each in order."""
    return tuple(
        prop for prop in (*node.required_properties, *node.optional_properties) if prop.extracted
    )


def _property_schema(node: NodeTypeDefinition, prop: PropertyDefinition) -> dict:
    """The schema of one written property: its JSON type, or that type or null.

    Raises:
        SchemaGenerationError: The declared type has no JSON type the schema can state.
    """
    json_type = _JSON_TYPES.get(prop.type)
    if json_type is None:
        raise SchemaGenerationError(
            f"{node.type_name}.{prop.name} declares type {prop.type}, which has no JSON type"
        )
    return {"type": [json_type, "null"] if prop.nullable else json_type}


def _entity_branch(type_schema: dict, node: NodeTypeDefinition | None = None) -> dict:
    """An entity object whose `type` is `type_schema`, stating `node`'s written keys.

    The branch requires `id`, `name` and `type`, not `properties`: an entity may carry
    none. Where it does, the written keys the type requires are required. Keys are
    stated in the order the examples write them, which is the order the grammar emits
    required keys in.
    """
    properties: dict = {"id": {"type": "string"}, "name": {"type": "string"}, "type": type_schema}
    if node is not None:
        written = _written(node)
        properties["properties"] = {
            "type": "object",
            "properties": {prop.name: _property_schema(node, prop) for prop in written},
            "required": [prop.name for prop in node.required_properties if prop.extracted],
            "additionalProperties": True,
        }
    return {
        "type": "object",
        "properties": properties,
        "required": ["id", "name", "type"],
        "additionalProperties": True,
    }


def _entity_items(entity_types: Sequence[NodeTypeDefinition]) -> dict:
    """`entities.items`: a branch per type that requires a written key, and one for the rest.

    The branches are alternatives in an `anyOf`, the form the grammar converter reads;
    it drops a conditional without an error. The rest branch is left out when no type
    is left for it, since an empty `enum` does not parse.
    """
    branch_types = [
        node for node in entity_types if any(p.extracted for p in node.required_properties)
    ]
    branch_names = {node.type_name for node in branch_types}
    rest = [node.type_name for node in entity_types if node.type_name not in branch_names]
    branches = [_entity_branch({"const": node.type_name}, node) for node in branch_types]
    if rest:
        branches.append(_entity_branch({"enum": rest}))
    return branches[0] if len(branches) == 1 else {"anyOf": branches}


def generate_extraction_schema(
    *, entity_types: Sequence[NodeTypeDefinition], relationship_types: Sequence[str]
) -> dict:
    """Stage 2's output schema for a pack whose model may produce these types.

    Every name and key is read from the declarations passed in, so a pack states its
    own output keys in its own types and this module names none. A type that declares a
    required property the model writes gets a branch of its own, its `type` a `const`;
    every other type shares one branch whose `type` is an `enum`. Every object keeps
    `additionalProperties` true, so a key no type declares is still admitted.

    Args:
        entity_types: The pack's extractable node type definitions, in the order of its
            extractable list.
        relationship_types: The pack's extractable edge types, in list order.

    Returns:
        A schema of the caller's own: no two calls share a mapping.

    Raises:
        SchemaGenerationError: Either list is empty, or holds an empty or a repeated
            name, or a written property declares a type with no JSON type.
    """
    _require_enum_members("entity", [node.type_name for node in entity_types])
    _require_enum_members("relationship", relationship_types)
    return {
        "type": "object",
        "properties": {
            "entities": {"type": "array", "items": _entity_items(entity_types)},
            "relationships": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "target": {"type": "string"},
                        "type": {"enum": list(relationship_types)},
                    },
                    "required": ["source", "target", "type"],
                    "additionalProperties": True,
                },
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
