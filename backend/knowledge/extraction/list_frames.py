"""Group a structured turn's units into list frames, and parse the list calls' outputs.

The list path sends a structured turn to one frame call, which decides what each group of
items states, and to item calls that read the items a few at a time. Both calls read the
same render of the turn: the prose units and the groups, each group's context written once
rather than fused into every item as `render_unit` does, so a call sees the list's shape
and can refer to any item by its id.

A group is a maximal run of list items or table rows that share one context tuple, as
`split_structured` writes it. A nested child's units carry a longer context, so they open a
group of their own without ending their parent's run, and a sibling after a nested child
stays in its group.

The parsers are strict: an output that misses a key, has a value of the wrong type or
outside its enum, holds a key its schema does not declare, gives a group more than 3
templates, or misses or invents a group or item id raises `ListOutputError`. They also
resolve every end at parse time, against the turn's groups, the parsed frame and the
item outputs already parsed, so an end that would resolve to no entity is refused in
the call that can fix it rather than dropped at assembly. `ListOutputError` subclasses
`ExtractionValidationError`, so the extraction service's attempt loop sends the repair
instruction and retries, as it does for a Stage 2 output.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from backend.errors import ExtractionValidationError
from backend.extraction_service.list_schemas import (
    ASSERTION_KIND_ENUM,
    ENTITY_KEY_ENUM,
    EXTRACTABLE_TYPE_ENUM,
    FRAME_KEY_ENUM,
    PREDICATE_ENUM,
    TEMPORAL_STATUS_ENUM,
    UNCHECKED_ENUM,
)

from .structure_splitter import InputUnit, _as_context

_LIST_KINDS = frozenset({"list_item", "table_row"})
_TEMPLATES_MAX = 3
_ITEM = "item"
_ITEM_REF = "item:"
# The ends every edge may name, and the roles a template adds; `parent` only in a
# nested group.
_FIXED_ENDS = frozenset({"user", "mist"})
_TEMPLATE_ENDS = _FIXED_ENDS | {_ITEM}


class ListOutputError(ExtractionValidationError):
    """A frame or item output that does not have the shape its stage asks for."""


@dataclass(frozen=True, slots=True)
class ListGroup:
    """One group of list items or table rows that share a context.

    Attributes:
        gid: `G1`, `G2` and so on, in the order the groups start.
        context: The headings, lead-ins and parent items above the items, as
            `split_structured` writes them.
        items: The group's units, in turn order.
        parent: `(group id, item id)` of the item this group is nested under,
            or None for a top-level group.
    """

    gid: str
    context: tuple[str, ...]
    items: tuple[InputUnit, ...]
    parent: tuple[str, str] | None

    @property
    def item_ids(self) -> tuple[str, ...]:
        """`G1.1`, `G1.2` and so on, one per item."""
        return tuple(f"{self.gid}.{n}" for n in range(1, len(self.items) + 1))


@dataclass(frozen=True, slots=True)
class Template:
    """One fact every item of a group states, its ends given as roles.

    A role is `user`, `mist`, `item` (each item's own entity), `parent` (the item a
    nested group sits under) or a frame entity's key.
    """

    subject: str
    predicate: str
    object: str


@dataclass(frozen=True, slots=True)
class GroupFrame:
    """What the frame call decided one group states.

    Attributes:
        group: The group's id, `G1`.
        templates: At most 3. None in a durable group means no fact is shared across
            the items, and each item states its own (a table whose fact is in each row).
        item_type: The type every item shares, or None.
        unchecked: `plan` or `goal`, what an unchecked box under this group is.
        durable: False when the group states nothing durable (an errand list), which
            then takes no templates. Last, with a default, so a frame built in code
            without it stays durable; the parser always sets it.
    """

    group: str
    templates: tuple[Template, ...]
    item_type: str | None
    unchecked: str
    durable: bool = True


@dataclass(frozen=True, slots=True)
class ListEntity:
    """An entity a list call names, keyed so the call's relationships can refer to it.

    `activity` marks a habit or recurring activity; `value`, `unit` and `quantity` are a
    Metric's, and a Date's ISO value is its `value`.
    """

    key: str
    span: str
    name: str
    type: str
    activity: bool = False
    value: str | None = None
    unit: str | None = None
    quantity: str | None = None


@dataclass(frozen=True, slots=True)
class ListRelationship:
    """A relationship whose ends are roles, keys or item references such as `item:G1.2`."""

    source: str
    target: str
    type: str
    properties: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ListFacts:
    """Entities and relationships a list call states beyond the group templates."""

    entities: tuple[ListEntity, ...]
    relationships: tuple[ListRelationship, ...]


@dataclass(frozen=True, slots=True)
class FrameOutput:
    """The frame call's output: frame entities, one frame per group, and turn facts."""

    frame_entities: tuple[ListEntity, ...]
    groups: tuple[GroupFrame, ...]
    facts: ListFacts


@dataclass(frozen=True, slots=True)
class ItemEntity:
    """An item's own entity; for a table row, its first column's.

    `value`, `unit` and `quantity` are a Metric item's, as a keyed Metric's are.
    """

    span: str
    name: str
    type: str
    activity: bool = False
    value: str | None = None
    unit: str | None = None
    quantity: str | None = None


@dataclass(frozen=True, slots=True)
class ItemTemplate:
    """Whether the group's templates hold for one item, and with what temporal fields."""

    applies: bool
    assertion_kind: str
    temporal_status: str
    start_date: str | None = None
    end_date: str | None = None


@dataclass(frozen=True, slots=True)
class ItemOutput:
    """One item's record from an item call."""

    item: str
    entity: ItemEntity | None
    template: ItemTemplate
    entities: tuple[ListEntity, ...]
    relationships: tuple[ListRelationship, ...]


@dataclass(slots=True)
class _Run:
    context: tuple[str, ...]
    start: int
    positions: list[int]
    items: list[InputUnit]


def _extends(context: tuple[str, ...], base: tuple[str, ...]) -> bool:
    return len(context) > len(base) and context[: len(base)] == base


def _parent(run: _Run, earlier: Sequence[tuple[str, _Run]]) -> tuple[str, str] | None:
    """The nearest item before `run` whose group context plus its text is `run`'s context.

    The item's text is compared as `split_structured` writes it into a child's context,
    trailing colon and whitespace stripped, so a parent item "Postgres:" is found.
    """
    best: tuple[int, str, str] | None = None
    for gid, other in earlier:
        if len(other.context) + 1 != len(run.context) or other.context != run.context[:-1]:
            continue
        for n, (position, unit) in enumerate(zip(other.positions, other.items), start=1):
            found = position < run.start and _as_context(unit.text) == run.context[-1]
            if found and (best is None or position > best[0]):
                best = (position, gid, f"{gid}.{n}")
    return None if best is None else (best[1], best[2])


def group_units(units: Sequence[InputUnit]) -> tuple[tuple[ListGroup, ...], tuple[InputUnit, ...]]:
    """Group the list items and table rows of a turn, and return its prose units apart.

    A unit closes every open group whose context it does not extend; a list item or table
    row then joins the open group with its own context, or starts a new one.

    Args:
        units: The turn's units, from `split_structured`.

    Returns:
        The groups, numbered in the order they start, and the prose units in turn order.
    """
    runs: list[_Run] = []
    open_runs: list[_Run] = []
    prose: list[InputUnit] = []
    for position, unit in enumerate(units):
        is_item = unit.kind in _LIST_KINDS
        joined: _Run | None = None
        still_open: list[_Run] = []
        for run in open_runs:
            if _extends(unit.context, run.context):
                still_open.append(run)
            elif is_item and run.context == unit.context:
                still_open.append(run)
                joined = run
        open_runs = still_open
        if not is_item:
            prose.append(unit)
        elif joined is not None:
            joined.positions.append(position)
            joined.items.append(unit)
        else:
            run = _Run(unit.context, position, [position], [unit])
            runs.append(run)
            open_runs.append(run)

    numbered = [(f"G{n}", run) for n, run in enumerate(runs, start=1)]
    groups = tuple(
        ListGroup(gid, run.context, tuple(run.items), _parent(run, numbered[:index]))
        for index, (gid, run) in enumerate(numbered)
    )
    return groups, tuple(prose)


def _one_line(text: str) -> str:
    return " ".join(text.split("\n"))


def _quoted(context: tuple[str, ...]) -> str:
    return " > ".join(f'"{part}"' for part in context)


def render_frame_input(groups: Sequence[ListGroup], prose: Sequence[InputUnit]) -> str:
    """The turn as the frame and item calls read it: prose units, then each group.

    Prose units are `P1`, `P2` in turn order. Each group's line names its context once
    and its parent item, and its items follow, indented, under their ids. The full
    utterance travels beside the render, so the prose-first order loses no text.
    """
    lines = []
    for n, unit in enumerate(prose, start=1):
        under = f"under {_quoted(unit.context)}: " if unit.context else ""
        lines.append(f"P{n} {under}{_one_line(unit.text)}")
    for group in groups:
        head = group.gid
        if group.context:
            head += f" under {_quoted(group.context)}"
        if group.parent is not None:
            head += f" (parent {group.parent[1]})"
        lines.append(head)
        lines.extend(
            f"  {item_id} {_one_line(unit.text)}"
            for item_id, unit in zip(group.item_ids, group.items)
        )
    return "\n".join(lines)


# -- strict parsers -------------------------------------------------------------------------

_Check = Callable[[object], bool]


def _text(value: object) -> bool:
    return isinstance(value, str) and value != ""


def _string(value: object) -> bool:
    return isinstance(value, str)


def _boolean(value: object) -> bool:
    return isinstance(value, bool)


def _number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _list(value: object) -> bool:
    return isinstance(value, list)


def _one_of(values: Sequence[str]) -> _Check:
    allowed = frozenset(values)
    return lambda value: isinstance(value, str) and value in allowed


def _object_or_null(value: object) -> bool:
    return value is None or isinstance(value, dict)


_TYPE = _one_of(EXTRACTABLE_TYPE_ENUM)
_PREDICATE = _one_of(PREDICATE_ENUM)
_TEMPORAL_STATUS = _one_of(TEMPORAL_STATUS_ENUM)
_ASSERTION_KIND = _one_of(ASSERTION_KIND_ENUM)

_PROPERTY_CHECKS: dict[str, _Check] = {
    "confidence": _number,
    "temporal_status": _TEMPORAL_STATUS,
    "start_date": _string,
    "end_date": _string,
    "temporal_expression": _string,
    "context": _string,
    "negated": _boolean,
    "assertion_kind": _ASSERTION_KIND,
}
_ENTITY_REQUIRED: dict[str, _Check] = {"key": _text, "span": _text, "name": _text, "type": _TYPE}
_ENTITY_OPTIONAL: dict[str, _Check] = {
    "activity": _boolean,
    "value": _string,
    "unit": _string,
    "quantity": _string,
}


def _record(
    value: object,
    where: str,
    required: Mapping[str, _Check],
    optional: Mapping[str, _Check] | None = None,
) -> dict:
    """The object's declared keys, each checked; an optional key given as null is left out.

    Raises:
        ListOutputError: `value` is not an object, misses a required key, holds an
            undeclared key, or has a value its check refuses.
    """
    optional = optional or {}
    if not isinstance(value, dict):
        raise ListOutputError(f"{where}: expected an object, got {type(value).__name__}")
    unknown = sorted(set(value) - set(required) - set(optional))
    if unknown:
        raise ListOutputError(f"{where}: undeclared keys {unknown}")
    checked = {}
    for key, check in required.items():
        if key not in value:
            raise ListOutputError(f"{where}: missing key {key!r}")
        if not check(value[key]):
            raise ListOutputError(f"{where}.{key}: wrong type or value {value[key]!r}")
        checked[key] = value[key]
    for key, check in optional.items():
        if value.get(key) is None:
            continue
        if not check(value[key]):
            raise ListOutputError(f"{where}.{key}: wrong type or value {value[key]!r}")
        checked[key] = value[key]
    return checked


def _load(text: str) -> dict:
    """The output's JSON object; a code fence or a line around it is tolerated, as Stage 2's is."""
    if not text or not text.strip():
        raise ListOutputError("empty list output")
    for candidate in (text, *re.findall(r"\{.*\}", text, re.DOTALL)):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ListOutputError(f"list output is not a JSON object: {text[:200]!r}")


def _key(key: str, where: str, keys: tuple[str, ...], what: str, taken: set[str]) -> None:
    """Refuse a key outside its set, or one already given in the same output."""
    if key not in keys:
        raise ListOutputError(f"{where}.key: {key!r} is not {what}, {keys[0]} to {keys[-1]}")
    if key in taken:
        raise ListOutputError(f"{where}.key: {key!r} is already taken")
    taken.add(key)


def _entities(values: list, where: str) -> tuple[ListEntity, ...]:
    """Entities keyed `e1` to `e20`, no key given twice."""
    entities = []
    taken: set[str] = set()
    for n, value in enumerate(values):
        record = _record(value, f"{where}[{n}]", _ENTITY_REQUIRED, _ENTITY_OPTIONAL)
        _key(record["key"], f"{where}[{n}]", ENTITY_KEY_ENUM, "an entity key", taken)
        entities.append(ListEntity(**record))
    return tuple(entities)


def _relationships(values: list, where: str) -> tuple[ListRelationship, ...]:
    relationships = []
    for n, value in enumerate(values):
        at = f"{where}[{n}]"
        record = _record(
            value,
            at,
            {"source": _text, "target": _text, "type": _PREDICATE},
            {"properties": lambda v: isinstance(v, dict)},
        )
        properties = _record(record.pop("properties", {}), f"{at}.properties", {}, _PROPERTY_CHECKS)
        relationships.append(ListRelationship(**record, properties=properties))
    return tuple(relationships)


def _facts(
    value: object, where: str, frame_keys: frozenset[str], items: frozenset[str]
) -> ListFacts:
    """The frame's facts; each end is `user`, `mist`, a declared key or an item of the turn.

    `item` and `parent` mean something only in a template or an item's own edge, so a
    fact that names one would resolve to nothing (U1).
    """
    record = _record(value, where, {"entities": _list, "relationships": _list})
    entities = _entities(record["entities"], f"{where}.entities")
    relationships = _relationships(record["relationships"], f"{where}.relationships")
    ends = _FIXED_ENDS | frame_keys | {e.key for e in entities} | items
    for n, relationship in enumerate(relationships):
        for side in ("source", "target"):
            end = getattr(relationship, side)
            if end not in ends:
                raise ListOutputError(
                    f"{where}.relationships[{n}].{side}: {end!r} is not user, mist, a "
                    "declared frame or facts key, or item:<id> for an item of the turn"
                )
    return ListFacts(entities, relationships)


def _ordered(found: Mapping[str, object], expected: Sequence[str], what: str) -> list:
    missing = [i for i in expected if i not in found]
    if missing:
        raise ListOutputError(f"missing {what} ids {missing}")
    return [found[i] for i in expected]


def _claim(found: dict, key: str, value: object, expected: set[str], what: str) -> None:
    if key not in expected:
        raise ListOutputError(f"unknown {what} id {key!r}")
    if key in found:
        raise ListOutputError(f"{what} id {key!r} given twice")
    found[key] = value


def _template_end(end: str, at: str, group: ListGroup, frame_keys: frozenset[str]) -> None:
    """Refuse a template end that names no role or frame key, or `parent` at the top level."""
    if end in _TEMPLATE_ENDS or end in frame_keys:
        return
    if end == "parent":
        if group.parent is not None:
            return
        raise ListOutputError(
            f"{at} ({group.gid}): {group.gid} is a top-level group, so no template can "
            "name parent"
        )
    parent = "parent, " if group.parent is not None else ""
    raise ListOutputError(
        f"{at} ({group.gid}): {end!r} is not user, mist, item, {parent}or a declared " "frame key"
    )


def _group_frame(
    value: object,
    where: str,
    frame_keys: frozenset[str],
    by_gid: Mapping[str, ListGroup],
) -> GroupFrame:
    record = _record(
        value,
        where,
        {
            "group": _text,
            "durable": _boolean,
            "templates": _list,
            "item_type": lambda v: v is None or _TYPE(v),
            "unchecked": _one_of(UNCHECKED_ENUM),
        },
    )
    if len(record["templates"]) > _TEMPLATES_MAX:
        raise ListOutputError(f"{where}.templates: more than {_TEMPLATES_MAX} templates")
    if not record["durable"] and record["templates"]:
        raise ListOutputError(
            f"{where} ({record['group']}): durable is false, so the group takes no "
            "templates; give durable true or no templates"
        )
    group = by_gid.get(record["group"])
    if group is None:
        raise ListOutputError(f"{where}: unknown group id {record['group']!r}")
    templates = []
    for n, value in enumerate(record["templates"]):
        at = f"{where}.templates[{n}]"
        template = Template(
            **_record(value, at, {"subject": _text, "predicate": _PREDICATE, "object": _text})
        )
        _template_end(template.subject, f"{at}.subject", group, frame_keys)
        _template_end(template.object, f"{at}.object", group, frame_keys)
        templates.append(template)
    if record["item_type"] is None and any(_ITEM in (t.subject, t.object) for t in templates):
        raise ListOutputError(
            f"{where} ({group.gid}): a template names item, so item_type must be set"
        )
    return GroupFrame(
        group.gid, tuple(templates), record["item_type"], record["unchecked"], record["durable"]
    )


def parse_frame_output(text: str, groups: Sequence[ListGroup]) -> FrameOutput:
    """Parse a frame call's output, its groups in the turn's order, every end resolved.

    Args:
        text: The model's output.
        groups: The turn's groups; the output must give each exactly once.

    Raises:
        ListOutputError: The output is not a JSON object of the FRAME shape; a key is
            outside its set (`f1` to `f20` for a frame entity, `e1` to `e20` for a
            facts entity); a template end is not `user`, `mist`, `item`, a declared
            frame key or, in a nested group, `parent`; a fact end is not `user`,
            `mist`, a declared key or an item of the turn; a group that names `item`
            has no `item_type`; or the group ids differ from the turn's.
    """
    record = _record(
        _load(text),
        "frame",
        {"frame_entities": _list, "groups": _list, "facts": lambda v: isinstance(v, dict)},
    )
    taken: set[str] = set()
    frame_entities = []
    for n, value in enumerate(record["frame_entities"]):
        at = f"frame_entities[{n}]"
        entity = ListEntity(**_record(value, at, _ENTITY_REQUIRED))
        _key(entity.key, at, FRAME_KEY_ENUM, "a frame key", taken)
        frame_entities.append(entity)
    frame_keys = frozenset(taken)
    by_gid = {group.gid: group for group in groups}
    expected = [group.gid for group in groups]
    found: dict[str, GroupFrame] = {}
    for n, value in enumerate(record["groups"]):
        group = _group_frame(value, f"groups[{n}]", frame_keys, by_gid)
        _claim(found, group.group, group, set(expected), "group")
    items = frozenset(_ITEM_REF + item for group in groups for item in group.item_ids)
    return FrameOutput(
        tuple(frame_entities),
        tuple(_ordered(found, expected, "group")),
        _facts(record["facts"], "facts", frame_keys, items),
    )


def _item_output(value: object, where: str) -> ItemOutput:
    record = _record(
        value,
        where,
        {
            "item": _text,
            "template": lambda v: isinstance(v, dict),
            "entity": _object_or_null,
            "entities": _list,
            "relationships": _list,
        },
    )
    where = f"{where} ({record['item']})"
    entity = None
    if record["entity"] is not None:
        entity = ItemEntity(
            **_record(
                record["entity"],
                f"{where}.entity",
                {"span": _text, "name": _text, "type": _TYPE},
                _ENTITY_OPTIONAL,
            )
        )
    template = ItemTemplate(
        **_record(
            record["template"],
            f"{where}.template",
            {
                "applies": _boolean,
                "assertion_kind": _ASSERTION_KIND,
                "temporal_status": _TEMPORAL_STATUS,
            },
            {"start_date": _string, "end_date": _string},
        )
    )
    return ItemOutput(
        item=record["item"],
        entity=entity,
        template=template,
        entities=_entities(record["entities"], f"{where}.entities"),
        relationships=_relationships(record["relationships"], f"{where}.relationships"),
    )


@dataclass(frozen=True, slots=True)
class _Turn:
    """What an item output's ends are checked against."""

    group_of: Mapping[str, ListGroup]
    frame_keys: frozenset[str]
    known: Mapping[str, ItemOutput]


def _no_entity(item_id: str, turn: _Turn) -> bool:
    """Whether the item's output is known, from an earlier batch or this one, with no entity.

    An item not yet parsed may still name its entity in a later call, so it is left to
    assembly's last-resort drop.
    """
    output = turn.known.get(item_id)
    return output is not None and output.entity is None


def _edge_end(end: str, at: str, output: ItemOutput, turn: _Turn) -> None:
    """Refuse an item edge's end that cannot resolve to an entity (U2 to U7)."""
    if end in _FIXED_ENDS or end in turn.frame_keys or end in {e.key for e in output.entities}:
        return
    if end == _ITEM:
        if output.entity is None:
            raise ListOutputError(f"{at}: item has no entity in this output; give one")
        return
    if end == "parent":
        group = turn.group_of.get(output.item)
        parent = None if group is None else group.parent
        if parent is None:
            raise ListOutputError(
                f"{at}: {output.item} is in a top-level group, so it has no parent"
            )
        if _no_entity(parent[1], turn):
            raise ListOutputError(f"{at}: parent {parent[1]} has no entity")
        return
    other = end[len(_ITEM_REF) :] if end.startswith(_ITEM_REF) else None
    if other is not None and other in turn.group_of:
        if _no_entity(other, turn):
            raise ListOutputError(f"{at}: {end} has no entity")
        return
    raise ListOutputError(
        f"{at}: {end!r} is not user, mist, item, parent, a declared frame key, one of "
        "the item's own keys, or item:<id> for an item of the turn"
    )


def parse_item_output(
    text: str,
    expected: Sequence[str],
    frame: FrameOutput,
    groups: Sequence[ListGroup],
    earlier: Mapping[str, ItemOutput],
) -> tuple[ItemOutput, ...]:
    """Parse an item call's output, one record per id in `expected`, every end resolved.

    An end that names an item, its parent or another item is checked against the
    outputs known so far: `earlier`, the outputs of the turn's earlier item calls, and
    the rest of this output. The call that wrote an edge can drop it, so the edge is
    refused here; an end whose item is not yet parsed is left to assembly.

    Args:
        text: The model's output.
        expected: The item ids the call asked for, in order.
        frame: The turn's parsed frame output.
        groups: The turn's groups.
        earlier: The item outputs already parsed in the turn, by item id.

    Raises:
        ListOutputError: The output is not a JSON object of the ITEMS shape; its item
            ids differ from `expected`; an own key is outside `e1` to `e20`; or an
            edge end does not resolve.
    """
    record = _record(_load(text), "items", {"items": _list})
    found: dict[str, ItemOutput] = {}
    at: dict[str, str] = {}
    for n, value in enumerate(record["items"]):
        item = _item_output(value, f"items[{n}]")
        _claim(found, item.item, item, set(expected), "item")
        at[item.item] = f"items[{n}] ({item.item})"
    outputs = tuple(_ordered(found, expected, "item"))
    turn = _Turn(
        group_of={item: group for group in groups for item in group.item_ids},
        frame_keys=frozenset(e.key for e in frame.frame_entities),
        known={**earlier, **found},
    )
    for output in outputs:
        for n, relationship in enumerate(output.relationships):
            for side in ("source", "target"):
                where = f"{at[output.item]}.relationships[{n}].{side}"
                _edge_end(getattr(relationship, side), where, output, turn)
    return outputs
