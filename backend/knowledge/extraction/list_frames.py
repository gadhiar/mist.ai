"""Group a structured turn's units into list frames for the list path.

The list path sends a structured turn to one frame call, which decides what each group of
items states, and to item calls that read the items a few at a time. Both calls read the
same render of the turn: the prose units and the groups, each group's context written once
rather than fused into every item as `render_unit` does, so a call sees the list's shape
and can refer to any item by its id.

A group is a maximal run of list items or table rows that share one context tuple, as
`split_structured` writes it. A nested child's units carry a longer context, so they open a
group of their own without ending their parent's run, and a sibling after a nested child
stays in its group.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .structure_splitter import InputUnit, _as_context

_LIST_KINDS = frozenset({"list_item", "table_row"})


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
        templates: At most 3; none means the group states nothing durable.
        item_type: The type every item shares, or None.
        unchecked: `plan` or `goal`, what an unchecked box under this group is.
    """

    group: str
    templates: tuple[Template, ...]
    item_type: str | None
    unchecked: str


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
    """An item's own entity; for a table row, its first column's."""

    span: str
    name: str
    type: str
    activity: bool = False


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
