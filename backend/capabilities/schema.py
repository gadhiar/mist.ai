"""Deterministic grammar schemas and catalog lines for capability tools.

llama-server turns a JSON schema into a grammar, and the decode prompt and the tests both need
the same bytes for the same tool on every run. Pydantic's own output carries titles, `$defs`
references and `null` branches for optional fields, none of which the grammar needs, and its
enum order follows the declaration, so an enum built from a set would vary by process as the
chat tool definitions' enums do (`KNOWN_ISSUES.md`, tool-schema enum set-ordering). This module
reduces the schema to a closed keyword set in a fixed key order with sorted enums, and refuses
any constraint the grammar cannot express, so a tool author learns of it at registration rather
than through a validation failure at turn time.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping

from backend.errors import CapabilityRegistrationError

from .contracts import ToolSpec

# Output key order. Every schema object is rebuilt in this order, whatever order its input had.
_KEY_ORDER = (
    "type",
    "const",
    "enum",
    "minimum",
    "maximum",
    "items",
    "maxItems",
    "properties",
    "required",
    "additionalProperties",
    "anyOf",
)
_ALLOWED_KEYS = frozenset(_KEY_ORDER)
# Documentation-only keys: dropping them changes nothing a grammar would enforce.
_DROPPED_KEYS = frozenset({"title", "description", "default", "examples"})
_REF_PREFIX = "#/$defs/"


def to_args_schema(
    spec: ToolSpec,
    refine: Callable[[str, dict[str, object]], dict[str, object]] | None = None,
) -> dict[str, object]:
    """Build the arguments schema the decode grammar uses for one tool.

    An optional field becomes a non-required property of its base type with no `null` branch:
    the grammar then lets the model omit it, and Pydantic fills the default. Every object gets
    `additionalProperties: false`, every enum is sorted, and nested models are inlined.

    Args:
        spec: The tool whose `args_model` is converted.
        refine: Applied to the converted schema with the tool name, as
            `Capability.refine_schema` is; its output is normalised again, so an enum it adds is
            sorted and the result is byte-identical across calls.

    Returns:
        A new schema dict, safe for the caller to mutate.

    Raises:
        CapabilityRegistrationError: The model or the refinement uses a keyword outside the
            grammar's set (for example `minLength` or `pattern`), or the root is not an object.
    """
    raw = spec.args_model.model_json_schema()
    defs = raw.get("$defs", {})
    schema = _normalise_root(raw, defs, spec.name)
    if refine is not None:
        schema = _normalise_root(refine(spec.name, schema), {}, spec.name)
    return schema


def to_catalog_line(spec: ToolSpec, args_schema: Mapping[str, object]) -> str:
    """Render one tool as a single catalog line for the decode prompt.

    For example `lights.power(target?: bedroom|kitchen, on: true|false): turn lights on or off`.
    Enum members are joined by `|` and optional arguments are marked `?`, so the model sees the
    allowed values in the fewest tokens.

    Args:
        spec: The tool; its name and description are used.
        args_schema: The output of `to_args_schema`, refined or not.

    Returns:
        The line, with any whitespace run in the description collapsed to one space.
    """
    description = " ".join(spec.description.split())
    return f"{spec.name}({_render_params(args_schema)}): {description}"


# -- Normalisation --------------------------------------------------------------------------


def _normalise_root(node: object, defs: object, tool: str) -> dict[str, object]:
    schema = _normalise(node, defs, tool, "", ())
    if schema.get("type") != "object":
        raise CapabilityRegistrationError(f"{tool}: the arguments schema must be an object")
    return schema


def _normalise(
    node: object, defs: object, tool: str, path: str, ref_stack: tuple[str, ...]
) -> dict[str, object]:
    where = path or "/"
    if not isinstance(node, Mapping):
        raise CapabilityRegistrationError(f"{tool}: schema at '{where}' is not an object")

    if "$ref" in node:
        target, name = _resolve_ref(node, defs, tool, where, ref_stack)
        return _normalise(target, defs, tool, path, (*ref_stack, name))

    if "anyOf" in node:
        branches = node["anyOf"]
        if not isinstance(branches, list):
            raise CapabilityRegistrationError(f"{tool}: 'anyOf' at '{where}' is not a list")
        kept = [b for b in branches if not (isinstance(b, Mapping) and b.get("type") == "null")]
        if not kept:
            raise CapabilityRegistrationError(f"{tool}: schema at '{where}' allows only null")
        if len(kept) == 1:
            siblings = {k: v for k, v in node.items() if k != "anyOf"}
            return _normalise({**kept[0], **siblings}, defs, tool, path, ref_stack)

    for key in node:
        if key not in _ALLOWED_KEYS and key not in _DROPPED_KEYS and key != "$defs":
            raise CapabilityRegistrationError(
                f"{tool}: schema keyword '{key}' at '{where}' cannot be expressed in the grammar"
            )

    out: dict[str, object] = {}
    for key in _KEY_ORDER:
        if key not in node:
            continue
        value = node[key]
        if key == "type":
            out["type"] = _normalise_type(value, tool, where)
        elif key == "enum":
            out["enum"] = _sorted_enum(value, tool, where)
        elif key == "items":
            out["items"] = _normalise(value, defs, tool, f"{path}/items", ref_stack)
        elif key == "anyOf":
            out["anyOf"] = [
                _normalise(branch, defs, tool, f"{path}/anyOf", ref_stack)
                for branch in value
                if not (isinstance(branch, Mapping) and branch.get("type") == "null")
            ]
        elif key in ("properties", "required", "additionalProperties"):
            continue  # rebuilt together below
        else:
            out[key] = value

    if out.get("type") == "object" or "properties" in node:
        out.update(_normalise_object(node, defs, tool, path, ref_stack))
        out = {key: out[key] for key in _KEY_ORDER if key in out}

    if not any(key in out for key in ("type", "const", "enum", "anyOf")):
        raise CapabilityRegistrationError(f"{tool}: schema at '{where}' has no type")
    return out


def _resolve_ref(
    node: Mapping[str, object], defs: object, tool: str, where: str, ref_stack: tuple[str, ...]
) -> tuple[dict[str, object], str]:
    ref = node["$ref"]
    if not isinstance(ref, str) or not ref.startswith(_REF_PREFIX):
        raise CapabilityRegistrationError(f"{tool}: unsupported reference {ref!r} at '{where}'")
    name = ref.removeprefix(_REF_PREFIX)
    if name in ref_stack:
        raise CapabilityRegistrationError(f"{tool}: recursive model '{name}' at '{where}'")
    target = defs.get(name) if isinstance(defs, Mapping) else None
    if not isinstance(target, Mapping):
        raise CapabilityRegistrationError(f"{tool}: unresolved reference {ref!r} at '{where}'")
    siblings = {k: v for k, v in node.items() if k != "$ref"}
    return {**target, **siblings}, name


def _normalise_object(
    node: Mapping[str, object], defs: object, tool: str, path: str, ref_stack: tuple[str, ...]
) -> dict[str, object]:
    where = path or "/"
    properties = node.get("properties", {})
    if not isinstance(properties, Mapping):
        raise CapabilityRegistrationError(f"{tool}: 'properties' at '{where}' is not an object")
    required_raw = node.get("required", [])
    if not isinstance(required_raw, list) or not all(isinstance(r, str) for r in required_raw):
        raise CapabilityRegistrationError(f"{tool}: 'required' at '{where}' is not a name list")
    unknown = sorted(set(required_raw) - set(properties))
    if unknown:
        raise CapabilityRegistrationError(
            f"{tool}: required names {unknown} at '{where}' are not properties"
        )
    extra = node.get("additionalProperties", False)
    if not isinstance(extra, bool):
        raise CapabilityRegistrationError(
            f"{tool}: open-ended 'additionalProperties' at '{where}' cannot be expressed"
        )

    normalised = {
        name: _normalise(prop, defs, tool, f"{path}/{name}", ref_stack)
        for name, prop in properties.items()
    }
    required_set = set(required_raw)
    return {
        "type": "object",
        "properties": normalised,
        "required": [name for name in properties if name in required_set],
        "additionalProperties": False,
    }


def _normalise_type(value: object, tool: str, where: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        kept = [t for t in value if t != "null"]
        if len(kept) == 1 and isinstance(kept[0], str):
            return kept[0]
    raise CapabilityRegistrationError(f"{tool}: type {value!r} at '{where}' is not one type")


def _sorted_enum(value: object, tool: str, where: str) -> list[object]:
    if not isinstance(value, list):
        raise CapabilityRegistrationError(f"{tool}: 'enum' at '{where}' is not a list")
    members = [member for member in value if member is not None]
    for member in members:
        if not isinstance(member, str | int | float):
            raise CapabilityRegistrationError(f"{tool}: enum member {member!r} at '{where}'")
    # Sorting by type name first keeps mixed-type enums comparable and still deterministic.
    return sorted(set(members), key=lambda member: (type(member).__name__, member))


# -- Catalog rendering ----------------------------------------------------------------------


def _render_params(schema: Mapping[str, object]) -> str:
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    if not isinstance(properties, Mapping) or not isinstance(required, list):
        return ""
    parts = []
    for name, prop in properties.items():
        marker = "" if name in required else "?"
        rendered = _render_type(prop) if isinstance(prop, Mapping) else "any"
        parts.append(f"{name}{marker}: {rendered}")
    return ", ".join(parts)


def _render_type(node: Mapping[str, object]) -> str:
    if "const" in node:
        return _render_scalar(node["const"])
    enum = node.get("enum")
    if isinstance(enum, list):
        return "|".join(_render_scalar(member) for member in enum)
    branches = node.get("anyOf")
    if isinstance(branches, list):
        return "|".join(_render_type(b) for b in branches if isinstance(b, Mapping))
    kind = node.get("type")
    if kind == "boolean":
        return "true|false"
    if kind == "integer":
        return "int" + _render_range(node)
    if kind == "number":
        return "number" + _render_range(node)
    if kind == "string":
        return "str"
    if kind == "array":
        items = node.get("items")
        inner = _render_type(items) if isinstance(items, Mapping) else "any"
        limit = node.get("maxItems")
        return f"list[{inner}]" + (f" max {limit}" if limit is not None else "")
    if kind == "object":
        return "{" + _render_params(node) + "}"
    return str(kind)


def _render_range(node: Mapping[str, object]) -> str:
    low, high = node.get("minimum"), node.get("maximum")
    if low is not None and high is not None:
        return f" {low}-{high}"
    if low is not None:
        return f" >={low}"
    if high is not None:
        return f" <={high}"
    return ""


def _render_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    return json.dumps(value)
