"""Split a structured utterance into units that each keep their parent context.

Stage 2 of the extraction service extracts a structured utterance a few units
at a time (`backend.extraction_service.engine`), so one long list cannot run a
single call past its output limit. A list item read on its own loses what it
belongs to -- "eggs" under "Groceries:" -- so every unit carries the headings,
lead-in lines and parent items above it as `context`, and `render_unit` writes
that context in front of the text. Text with no list, table or heading comes
back as one prose unit holding the exact text, so a turn without structure
reaches the model unchanged.

The rules are deterministic and line based. They are written for conversation
turns, and so that the document path's `split_markdown` can later move onto
them.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from backend.extraction_contract.models import ExtractionPayload

UnitKind = Literal["prose", "list_item", "table_row"]

# The warning `merge_payloads` adds when two batches give one entity id two
# different types; the first batch's type is kept.
LIST_MERGE_TYPE_CONFLICT = "list_merge_type_conflict"

_CONTEXT_JOIN = " > "

# Bold-only lines sit below every Markdown heading level.
_BOLD_LEVEL = 7
_TAB_WIDTH = 4

_ITEM_RE = re.compile(r"^(?:[-*+•]|\d{1,3}[.)]|[a-z][.)])[ \t]+(?P<body>\S.*)$")
_CHECKBOX_RE = re.compile(r"^\[(?P<mark>[ xX])\][ \t]+(?P<body>\S.*)$")
_HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})[ \t]+(?P<text>.+?)(?:[ \t]+#+)?$")
_BOLD_RE = re.compile(r"^(?P<mark>\*\*|__)(?P<text>(?:(?!(?P=mark)).)+)(?P=mark)[ \t]*:?$")
_FENCE_RE = re.compile(r"^(?P<fence>`{3,}|~{3,})")
_THEMATIC_BREAK_RE = re.compile(r"^([-*_])(?:[ \t]*\1){2,}$")
_TABLE_SEPARATOR_RE = re.compile(r"^\|?[ \t]*:?-+:?[ \t]*(?:\|[ \t]*:?-+:?[ \t]*)*\|?$")
_UNESCAPED_PIPE_RE = re.compile(r"(?<!\\)\|")


@dataclass(frozen=True, slots=True)
class InputUnit:
    """One prose paragraph, list item or table row of an utterance.

    Attributes:
        text: The unit's own text, without its list marker.
        context: The headings, lead-in lines and parent items above it,
            outermost first.
        kind: What the unit was in the source text.
    """

    text: str
    context: tuple[str, ...]
    kind: UnitKind


@dataclass(slots=True)
class _Draft:
    """A unit while its text may still grow (an item's continuation lines)."""

    text: str
    context: tuple[str, ...]
    kind: UnitKind


@dataclass(slots=True)
class _Giver:
    """A heading, bold-only line or lead-in line: context for what follows.

    `source` is the line as written, kept as a prose unit if the giver is
    dropped before any unit used it, so no text of the utterance is lost.
    """

    text: str
    source: str
    level: int
    used: bool = False


def _indent(line: str) -> int:
    expanded = line.expandtabs(_TAB_WIDTH)
    return len(expanded) - len(expanded.lstrip())


def _as_context(text: str) -> str:
    return text.rstrip().rstrip(":").rstrip()


def _item_text(stripped: str) -> str | None:
    """The item's text without its marker, or None when the line is not an item."""
    match = _ITEM_RE.match(stripped)
    body = match.group("body") if match else stripped
    checkbox = _CHECKBOX_RE.match(body)
    if checkbox:
        mark = " " if checkbox.group("mark") == " " else "x"
        return f"[{mark}] {checkbox.group('body').rstrip()}"
    return body.rstrip() if match else None


def _table_cells(line: str) -> list[str]:
    row = line.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|") and not row.endswith("\\|"):
        row = row[:-1]
    return [cell.strip().replace("\\|", "|") for cell in _UNESCAPED_PIPE_RE.split(row)]


def _table_row_text(headers: list[str], cells: list[str]) -> str:
    parts = []
    for index, cell in enumerate(cells):
        if not cell:
            continue
        header = headers[index] if index < len(headers) else ""
        parts.append(f"{header}: {cell}" if header else cell)
    return "; ".join(parts)


def _is_table_start(lines: Sequence[str], index: int) -> bool:
    if index + 1 >= len(lines) or "|" not in lines[index]:
        return False
    separator = lines[index + 1].strip()
    return "|" in separator and bool(_TABLE_SEPARATOR_RE.match(separator))


class _Splitter:
    """One pass over the lines of an utterance, building the units in order."""

    def __init__(self, text: str) -> None:
        self._lines = text.splitlines()
        self._units: list[_Draft] = []
        self._headings: list[_Giver] = []
        self._lead_ins: list[_Giver] = []
        # True once a list item or table row has been emitted under the current
        # lead-ins: prose after that list ends their reach.
        self._lead_in_had_items = False
        self._items: list[tuple[int, _Draft]] = []
        self._paragraph: list[str] = []
        self.structured = False

    def split(self) -> tuple[InputUnit, ...]:
        index = 0
        while index < len(self._lines):
            index = self._step(index)
        self._flush_paragraph()
        self._retire_lead_ins()
        self._retire_headings(1)
        return tuple(InputUnit(d.text, d.context, d.kind) for d in self._units)

    # -- one line ---------------------------------------------------------

    def _step(self, index: int) -> int:
        line = self._lines[index]
        stripped = line.strip()

        if not stripped or _THEMATIC_BREAK_RE.match(stripped):
            self._flush_paragraph()
            return index + 1
        fence = _FENCE_RE.match(stripped)
        if fence:
            return self._fenced_block(index, fence.group("fence"))
        heading = _HEADING_RE.match(stripped)
        if heading:
            level = len(heading.group("hashes"))
            self._open_heading(_as_context(heading.group("text")), stripped, level)
            return index + 1
        bold = _BOLD_RE.match(stripped)
        if bold:
            self._open_heading(_as_context(bold.group("text")), stripped, _BOLD_LEVEL)
            return index + 1
        if _is_table_start(self._lines, index):
            return self._table(index)
        text = _item_text(stripped)
        if text is not None:
            self._list_item(_indent(line), text)
            return index + 1
        if self._items and _indent(line) > 0:
            draft = self._items[-1][1]
            draft.text = f"{draft.text} {stripped}"
            return index + 1
        if stripped.endswith(":") and self._has_more(index):
            self._open_lead_in(stripped)
            return index + 1
        self._prose_line(stripped)
        return index + 1

    def _has_more(self, index: int) -> bool:
        return any(line.strip() for line in self._lines[index + 1 :])

    # -- context givers ---------------------------------------------------

    def _context(self) -> tuple[str, ...]:
        return tuple(g.text for g in self._headings) + tuple(g.text for g in self._lead_ins)

    def _emit(self, text: str, kind: UnitKind, parents: tuple[str, ...] = ()) -> _Draft:
        draft = _Draft(text, self._context() + parents, kind)
        for giver in (*self._headings, *self._lead_ins):
            giver.used = True
        self._units.append(draft)
        return draft

    def _keep_unused(self, givers: list[_Giver], before: tuple[str, ...]) -> None:
        """Emit each unused giver's own line as prose, so its text is not lost."""
        outer = list(before)
        for giver in givers:
            if not giver.used:
                self._units.append(_Draft(giver.source, tuple(outer), "prose"))
            outer.append(giver.text)

    def _retire_lead_ins(self) -> None:
        headings = tuple(g.text for g in self._headings)
        self._keep_unused(self._lead_ins, headings)
        self._lead_ins = []
        self._lead_in_had_items = False

    def _retire_headings(self, level: int) -> None:
        keep = [g for g in self._headings if g.level < level]
        dropped = self._headings[len(keep) :]
        self._keep_unused(dropped, tuple(g.text for g in keep))
        self._headings = keep

    def _open_heading(self, text: str, source: str, level: int) -> None:
        self._flush_paragraph()
        self.structured = True
        self._items = []
        self._retire_lead_ins()
        self._retire_headings(level)
        self._headings.append(_Giver(text, source, level))

    def _open_lead_in(self, stripped: str) -> None:
        self._flush_paragraph()
        self.structured = True
        self._items = []
        if any(g.used for g in self._lead_ins):
            self._retire_lead_ins()
        self._lead_ins.append(_Giver(_as_context(stripped), stripped, 0))

    # -- blocks -----------------------------------------------------------

    def _flush_paragraph(self) -> None:
        if self._paragraph:
            self._emit("\n".join(self._paragraph), "prose")
            self._paragraph = []

    def _end_list(self) -> None:
        self._items = []
        if self._lead_in_had_items:
            self._retire_lead_ins()

    def _prose_line(self, stripped: str) -> None:
        if self._items or self._lead_in_had_items:
            self._end_list()
        self._paragraph.append(stripped)

    def _list_item(self, indent: int, text: str) -> None:
        self._flush_paragraph()
        self.structured = True
        while self._items and self._items[-1][0] >= indent:
            self._items.pop()
        parents = tuple(_as_context(d.text) for _, d in self._items)
        draft = self._emit(text, "list_item", parents)
        self._items.append((indent, draft))
        self._lead_in_had_items = True

    def _table(self, index: int) -> int:
        self._flush_paragraph()
        self.structured = True
        self._items = []
        headers = _table_cells(self._lines[index])
        index += 2
        rows = 0
        while index < len(self._lines) and "|" in self._lines[index]:
            text = _table_row_text(headers, _table_cells(self._lines[index]))
            if text:
                self._emit(text, "table_row")
                rows += 1
            index += 1
        if rows == 0:
            self._emit(self._lines[index - 2].strip(), "prose")
        self._lead_in_had_items = True
        return index

    def _fenced_block(self, index: int, fence: str) -> int:
        self._flush_paragraph()
        end = index + 1
        while end < len(self._lines):
            closing = self._lines[end].strip()
            if closing.startswith(fence) and not closing.lstrip(fence[0]):
                break
            end += 1
        block = self._lines[index : end + 1]
        self._emit("\n".join(block), "prose")
        return end + 1


def split_structured(text: str) -> tuple[InputUnit, ...]:
    """Split `text` into prose, list-item and table-row units, in order.

    Returns exactly `(InputUnit(text=text, context=(), kind="prose"),)` when
    the text holds no list, table, heading, bold-only line or lead-in line, so
    an utterance without structure is never rewritten. A fenced code block on
    its own is not structure; inside structured text it is one prose unit.

    Rules:
        - Items: `-`, `*`, `+` and `•` bullets and `1.`, `1)`, `a.`, `a)`
          numbering, each followed by whitespace. A checkbox keeps its state
          as `[ ]` or `[x]` at the start of the text. Nesting follows the
          indentation (a tab counts as four spaces), and a child carries its
          parent items as context. An indented line without a marker
          continues the last item.
        - Headings and bold-only lines give context to what follows until a
          heading of the same or a higher level; bold-only lines rank below
          every `#` level.
        - A line ending in a colon with more text after it is a lead-in: context
          for the paragraphs, items and rows that follow, until prose comes
          after its list. Two lead-ins in a row nest.
        - A pipe table's rows become `Header: value; Header: value`.
        - A heading or lead-in that nothing used is kept as a prose unit, so no
          text is dropped.

    Args:
        text: The utterance.

    Returns:
        The units, in the order they appear.
    """
    splitter = _Splitter(text)
    units = splitter.split()
    if not splitter.structured:
        return (InputUnit(text=text, context=(), kind="prose"),)
    return units


def render_unit(unit: InputUnit) -> str:
    """The unit's text, after its context joined with ` > ` and a `: `.

    For example `Groceries: eggs` or `Trip plan > Day 1: visit the museum`; a
    unit without context renders as its text unchanged.
    """
    if not unit.context:
        return unit.text
    return f"{_CONTEXT_JOIN.join(unit.context)}: {unit.text}"


def batch_units(units: Sequence[InputUnit], max_units: int) -> tuple[tuple[InputUnit, ...], ...]:
    """Consecutive runs of at most `max_units` units, in order.

    Raises:
        ValueError: `max_units` is below 1.
    """
    if max_units < 1:
        raise ValueError(f"max_units must be at least 1, got {max_units}")
    return tuple(tuple(units[i : i + max_units]) for i in range(0, len(units), max_units))


def render_batch(batch: Sequence[InputUnit]) -> str:
    """The batch's rendered units, one per line.

    A batch of one context-free unit renders as its text unchanged, so a turn
    without structure keeps its exact utterance.
    """
    return "\n".join(render_unit(unit) for unit in batch)


def merge_payloads(
    payloads: Sequence[ExtractionPayload],
) -> tuple[ExtractionPayload, list[str]]:
    """Merge the batches' Stage 2 payloads into the turn's one payload.

    Entities merge by `id` and relationships by `(source, type, target)`, the
    first occurrence winning, in batch order; an entity or relationship
    without a string key is kept as it is, unmerged. When two batches give one
    id different types, the warning `list_merge_type_conflict` is added once.
    A single payload is returned as it is, so a turn without structure keeps
    exactly the result one Stage 2 call gives.

    Returns:
        The merged payload and its warnings.
    """
    if len(payloads) == 1:
        return payloads[0], []

    entities: list[dict] = []
    entity_types: dict[str, object] = {}
    relationships: list[dict] = []
    relationship_keys: set[tuple[object, object, object]] = set()
    warnings: list[str] = []

    for payload in payloads:
        for entity in payload.entities:
            entity_id = entity.get("id")
            if not isinstance(entity_id, str):
                entities.append(entity)
                continue
            if entity_id in entity_types:
                conflict = entity.get("type") != entity_types[entity_id]
                if conflict and LIST_MERGE_TYPE_CONFLICT not in warnings:
                    warnings.append(LIST_MERGE_TYPE_CONFLICT)
                continue
            entity_types[entity_id] = entity.get("type")
            entities.append(entity)
        for relationship in payload.relationships:
            key = (
                relationship.get("source"),
                relationship.get("type"),
                relationship.get("target"),
            )
            if not all(isinstance(part, str) for part in key):
                relationships.append(relationship)
                continue
            if key in relationship_keys:
                continue
            relationship_keys.add(key)
            relationships.append(relationship)

    return ExtractionPayload(entities=entities, relationships=relationships), warnings
