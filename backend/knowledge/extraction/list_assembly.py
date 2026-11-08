"""Assemble a structured turn's list payload from its frame and item outputs.

The list path asks one frame call what each group of items states, as templates whose
ends are roles, and item calls for each item's own entity and facts. This module turns
those outputs into one payload in Stage 2's shape, so the rest of the pipeline reads a
list turn as it reads any other. The template edges are built here, by code, rather than
by the model once per item, which is how a lead-in such as "Tools I use at Thrennic:"
reaches every item below it.

`assemble` is pure: it makes no call and reads no file, and the same inputs give the
same payload and warnings.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from backend.extraction_contract.models import ExtractionPayload

from .list_frames import (
    FrameOutput,
    GroupFrame,
    ItemEntity,
    ItemOutput,
    ItemTemplate,
    ListEntity,
    ListGroup,
    ListRelationship,
)
from .structure_splitter import InputUnit, merge_payloads

LIST_ITEM_REF_UNRESOLVED = "list_item_ref_unresolved"
LIST_ITEM_EDGE_ON_TEMPLATE_PAIR = "list_item_edge_on_template_pair"

_USER: dict = {"id": "user", "name": "User", "type": "User"}
_MIST: dict = {"id": "mist-identity", "name": "MIST", "type": "MistIdentity"}
_FIXED_IDS = {"User": _USER, "MistIdentity": _MIST}
_ITEM_REF = "item:"


def _canonical(text: str) -> str:
    """The scorer's `canonical_id` steps, which are `_canonicalize`'s after its version pattern.

    A copy rather than an import, because `backend/` does not import `scripts/`; the tests
    pin it equal to both.
    """
    s = text.lower().strip()
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"[^a-z0-9\-]", "", s)
    s = re.sub(r"-{2,}", "-", s)
    s = s.strip("-")
    return s or text.lower().replace(" ", "-")


def _derive(
    span: str,
    name: str | None,
    etype: str,
    utterance: str,
    activity: bool,
    value: str | None,
    unit: str | None,
    quantity: str | None,
) -> tuple[str, str, list[str]]:
    """The entity's id, its name, and the warnings deriving them raised."""
    fixed = _FIXED_IDS.get(etype)
    if fixed is not None:
        return fixed["id"], fixed["name"], []
    return _canonical(span), span, []


def list_entity_id(
    span: str,
    name: str | None,
    etype: str,
    utterance: str,
    *,
    activity: bool = False,
    value: str | None = None,
    unit: str | None = None,
    quantity: str | None = None,
) -> tuple[str, list[str]]:
    """The id a list entity gets from its source span, and the warnings deriving it raised.

    Args:
        span: The span the model copied from the utterance.
        name: The model's name for the entity, or None.
        etype: The entity's type.
        utterance: The turn's full text, which the span is read against.
        activity: The entity is a habit or recurring activity.
        value: A Metric's value, or a Date's ISO value.
        unit: A Metric's unit.
        quantity: What a Metric measures.

    Returns:
        The id and the warnings.
    """
    entity_id, _name, warnings = _derive(
        span, name, etype, utterance, activity, value, unit, quantity
    )
    return entity_id, warnings


def _copy(record: dict) -> dict:
    copied = dict(record)
    if "properties" in copied:
        copied["properties"] = dict(copied["properties"])
    return copied


def _template_properties(template: ItemTemplate) -> dict:
    return {
        "temporal_status": template.temporal_status,
        "start_date": template.start_date,
        "end_date": template.end_date,
        "assertion_kind": template.assertion_kind,
    }


def _checkbox(unit: InputUnit) -> str | None:
    """`checked`, `unchecked`, or None, from the mark `split_structured` keeps at the start."""
    if unit.text.startswith("[ ] "):
        return "unchecked"
    if unit.text.startswith("[x] "):
        return "checked"
    return None


class _Payload:
    """One payload being built: its edges, and each entity an edge uses, in first use."""

    def __init__(self) -> None:
        self.entities: dict[str, dict] = {}
        self.relationships: list[dict] = []

    def add(self, source: dict, type_: str, target: dict, properties: Mapping) -> dict:
        for record in (source, target):
            self.entities.setdefault(record["id"], _copy(record))
        edge = {
            "source": source["id"],
            "target": target["id"],
            "type": type_,
            "properties": dict(properties),
        }
        self.relationships.append(edge)
        return edge

    def build(self) -> ExtractionPayload:
        return ExtractionPayload(
            entities=list(self.entities.values()), relationships=self.relationships
        )


class _Assembler:
    """The state of one `assemble` call: the turn's lookups, records and warnings."""

    def __init__(
        self,
        frame: FrameOutput,
        items: Sequence[ItemOutput],
        groups: Sequence[ListGroup],
        utterance: str,
        fallbacks: Mapping[str, ExtractionPayload],
    ) -> None:
        self.frame = frame
        self.utterance = utterance
        self.fallbacks = fallbacks
        self.warnings: list[str] = []
        self.frames = {g.group: g for g in frame.groups}
        self.outputs = {o.item: o for o in items if o.item not in fallbacks}
        self.places: dict[str, tuple[ListGroup, InputUnit]] = {
            item_id: (group, unit)
            for group in groups
            for item_id, unit in zip(group.item_ids, group.items)
        }
        self.order = [item_id for group in groups for item_id in group.item_ids]
        self.frame_keys = {e.key: e for e in frame.frame_entities}
        self.records: dict[object, dict] = {}
        self.template_pairs: dict[tuple[str, str], set[str]] = {}

    # -- entity records -------------------------------------------------------------------

    def _record(
        self,
        cache_key: object,
        span: str,
        name: str | None,
        etype: str,
        *,
        activity: bool = False,
        value: str | None = None,
        unit: str | None = None,
        quantity: str | None = None,
    ) -> dict:
        """The entity's record, derived once per turn so a warning is raised once."""
        if cache_key in self.records:
            return self.records[cache_key]
        entity_id, entity_name, warnings = _derive(
            span, name, etype, self.utterance, activity, value, unit, quantity
        )
        self.warnings.extend(warnings)
        record: dict = {"id": entity_id, "name": entity_name, "type": etype}
        if etype == "Metric":
            record["properties"] = {"value": value, "unit": unit, "quantity": quantity}
        self.records[cache_key] = record
        return record

    def _entity(self, entity: ListEntity) -> dict:
        return self._record(
            entity,
            entity.span,
            entity.name,
            entity.type,
            activity=entity.activity,
            value=entity.value,
            unit=entity.unit,
            quantity=entity.quantity,
        )

    def _item(self, item_id: str) -> dict | None:
        """The item's own entity, or None when it has none or went to the r6 fallback."""
        output = self.outputs.get(item_id)
        if output is None or output.entity is None:
            return None
        entity: ItemEntity = output.entity
        group_frame = self.frames.get(self.places[item_id][0].gid)
        etype = entity.type or (group_frame.item_type if group_frame else None) or ""
        return self._record(
            ("item", item_id), entity.span, entity.name, etype, activity=entity.activity
        )

    # -- endpoints ------------------------------------------------------------------------

    def _endpoint(
        self,
        ref: str,
        item_id: str | None,
        local: Mapping[str, ListEntity],
    ) -> dict | None:
        """A role, a key or an item reference as an entity record, or None if it resolves to none.

        `item` and `parent` mean something only inside an item's own edges, where
        `item_id` is set. A key is looked up among the call's own entities first, then
        among the frame entities.
        """
        if ref == "user":
            return _USER
        if ref == "mist":
            return _MIST
        if ref == "item":
            return None if item_id is None else self._item(item_id)
        if ref == "parent":
            if item_id is None:
                return None
            parent = self.places[item_id][0].parent
            return None if parent is None else self._item(parent[1])
        if ref.startswith(_ITEM_REF):
            other = ref[len(_ITEM_REF) :]
            return self._item(other) if other in self.places else None
        entity = local.get(ref) or self.frame_keys.get(ref)
        return None if entity is None else self._entity(entity)

    def _ends(
        self, source: str, target: str, item_id: str | None, local: Mapping[str, ListEntity]
    ) -> tuple[dict, dict] | None:
        ends = (self._endpoint(source, item_id, local), self._endpoint(target, item_id, local))
        if ends[0] is None or ends[1] is None:
            self.warnings.append(LIST_ITEM_REF_UNRESOLVED)
            return None
        return ends[0], ends[1]

    # -- edges ----------------------------------------------------------------------------

    def _template_edges(self, item_id: str, payload: _Payload) -> bool:
        """Add the item's template edges; False when an unchecked box keeps its own edges out."""
        output = self.outputs[item_id]
        group, unit = self.places[item_id]
        group_frame: GroupFrame | None = self.frames.get(group.gid)
        properties = _template_properties(output.template)
        if _checkbox(unit) == "unchecked":
            if group_frame is not None and group_frame.unchecked == "goal":
                ends = self._ends("user", "item", item_id, {})
                if ends is not None:
                    self._template_edge(payload, ends[0], "HAS_GOAL", ends[1], properties)
            return False
        if output.template.applies and group_frame is not None:
            for template in group_frame.templates:
                ends = self._ends(template.subject, template.object, item_id, {})
                if ends is not None:
                    self._template_edge(payload, ends[0], template.predicate, ends[1], properties)
        return True

    def _template_edge(
        self, payload: _Payload, source: dict, type_: str, target: dict, properties: Mapping
    ) -> None:
        edge = payload.add(source, type_, target, properties)
        self.template_pairs.setdefault((edge["source"], edge["target"]), set()).add(type_)

    def _fact_edges(
        self,
        relationships: Sequence[ListRelationship],
        entities: Sequence[ListEntity],
        item_id: str | None,
        payload: _Payload,
    ) -> None:
        """Add the frame's or an item's own edges, flagging one on a template pair."""
        local = {e.key: e for e in entities}
        for relationship in relationships:
            ends = self._ends(relationship.source, relationship.target, item_id, local)
            if ends is None:
                continue
            on_pair = self.template_pairs.get((ends[0]["id"], ends[1]["id"]))
            if on_pair and relationship.type not in on_pair:
                self.warnings.append(LIST_ITEM_EDGE_ON_TEMPLATE_PAIR)
            payload.add(ends[0], relationship.type, ends[1], relationship.properties)

    # -- the turn -------------------------------------------------------------------------

    def run(self) -> tuple[ExtractionPayload, list[str]]:
        # Every template edge first, so an edge of the frame or of any item can be checked
        # against every template pair of the turn.
        item_payloads: dict[str, _Payload] = {}
        keeps_own_edges: dict[str, bool] = {}
        for item_id in self.order:
            if item_id in self.outputs:
                item_payloads[item_id] = _Payload()
                keeps_own_edges[item_id] = self._template_edges(item_id, item_payloads[item_id])

        frame_payload = _Payload()
        facts = self.frame.facts
        self._fact_edges(facts.relationships, facts.entities, None, frame_payload)

        payloads = [frame_payload.build()]
        for item_id in self.order:
            if item_id in self.fallbacks:
                # A copy, because the normalizer downstream edits records in place.
                payloads.append(self.fallbacks[item_id].model_copy(deep=True))
                continue
            if item_id not in item_payloads:
                continue
            output = self.outputs[item_id]
            if keeps_own_edges[item_id]:
                self._fact_edges(
                    output.relationships, output.entities, item_id, item_payloads[item_id]
                )
            payloads.append(item_payloads[item_id].build())

        merged, merge_warnings = merge_payloads(payloads)
        return merged, self.warnings + merge_warnings


def assemble(
    frame: FrameOutput,
    items: Sequence[ItemOutput],
    groups: Sequence[ListGroup],
    prose: Sequence[InputUnit],
    utterance: str,
    fallbacks: Mapping[str, ExtractionPayload],
) -> tuple[ExtractionPayload, list[str]]:
    """The turn's one payload, from its frame output and its items' outputs.

    Rules:
        - An unchecked box is a plan: its item gets no edge of any kind, unless its
          group's `unchecked` is `goal`, which gives it exactly `user HAS_GOAL item`.
        - Every other item whose template applies gets each of its group's templates,
          roles substituted, with the item's assertion kind, temporal status and dates.
        - Roles and `item:G1.2` references in the frame's facts and the items' own
          edges are substituted once every item is known, so an item may refer to any
          other. Such an edge is kept beside the template edges; on a pair a template
          edge already joins, with another predicate, it adds
          `list_item_edge_on_template_pair`. An edge whose end does not resolve is
          dropped with `list_item_ref_unresolved`. Each warning is added once per edge,
          so a count of them is a count of edges.
        - A payload holds each entity a kept edge uses, so an unchecked plan item's
          entity is left out unless another edge uses it.
        - The frame's payload and one per item, in the groups' order, go through
          `merge_payloads`: the first record per id wins, relationships are
          deduplicated, and a type conflict adds `list_merge_type_conflict`. An item
          in `fallbacks` joins with its r6 payload in its place, and its item output,
          if any, is ignored.

    Args:
        frame: The frame call's output.
        items: The item calls' outputs.
        groups: The turn's groups, from `group_units`.
        prose: The turn's prose units, from `group_units`. Their facts reach the
            payload through the frame's facts, so assembly reads nothing from them.
        utterance: The turn's full text, which every span is read against.
        fallbacks: An item id to the payload its r6 fallback returned.

    Returns:
        The payload and its warnings: the assembly's, then the merge's.
    """
    del prose
    return _Assembler(frame, items, groups, utterance, fallbacks).run()
