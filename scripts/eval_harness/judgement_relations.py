"""The judgement filter: the relations a pack's rails derive count once per team.

A roster of N people under a team heading that names the employer is N `MEMBER_OF` edges,
and the rails then add N `WORKS_AT` and, on a user-scope row, N `KNOWS_PERSON`. A labeller
who writes those 2N edges decided two things, the employer and that the user knows the team,
not 2N. Counted edge by edge, two passes that agree on a roster agree 2N times on nothing they
judged, and two that name different employers disagree N times over one decision (ADR-054's
v14 amendment, point 7). This module removes the duplicates before agreement is measured.

- A derived relation: a relation of a pass is derived when `run_rails`, run over that pass
  with the relation removed, gives it back. Every copy the pass wrote of the relation is
  removed, so a duplicate does not give it back.
- The row the rails read holds the pass's entities and relations as its payload, the probe's
  scope, the batch record the extraction service writes for the probe's utterance
  (`batch_record`), and no turn plan.
- The group: a derived relation belongs to one `fan_out` rule of the pack, one team and its
  own predicate: the first rule, in the pack's order, and the first team, by stored id, whose
  trigger edges alone, the rule's other trigger edges removed, give it back. A derived
  relation no `fan_out` rule gives back is not grouped and is kept.
- The filter: the groups are the union of the two passes'. Of each group one relation is
  kept, the first by sorted key that both passes hold or, when they share none, the first in
  the union, and every other relation of the group is removed from both passes, whether the
  pass derived it or wrote it on its own.

Every relation is keyed by its triple after `run_rails`, so the two passes are compared under
the ids the graph stores. `stored_row` gives that row, and is the id function the agreement
module (`gold_agreement`) is handed.

The caller passes the pack and `run_rails`. This module imports neither a pack, the pack
registry, nor `backend.knowledge.extraction.rails`, so the pack a gold set is labelled under
does not define which of its relations count; it reads the pack's rules by attribute only.
The engine and the item filter it imports do load the ontology package, and with it the
v1.4.0 pack's modules, but nothing here reads them. The batch record is built with the
service's own splitter, item filter and batcher, and the engine's `_record_entry` writes each
entry, so no model is called.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from backend.extraction_contract.models import ExtractionPayload
from backend.extraction_service.engine import LIST_BATCH_MAX_UNITS, _batch_input, _record_entry
from backend.knowledge.extraction.pre_llm import filter_units
from backend.knowledge.extraction.preprocessor import PreProcessor
from backend.knowledge.extraction.structure_splitter import (
    batch_units,
    is_numbered_list,
    split_structured,
)
from backend.llm.models import LLMResponse

from .gold_agreement import LabelEdge, LabelRow

RunRails = Callable[..., tuple[list[dict], list[dict]]]
Triple = tuple[str, str, str]
GroupKey = tuple[str, str, str]

# A label edge's attributes, carried through the rails in the edge's `properties`. The
# rails read `assertion_kind` there to tell a live edge; they read no date.
_ATTRIBUTES: tuple[str, ...] = ("assertion_kind", "valid_from", "valid_to")
# The splitter and batcher read no date; the pre-processor asks for one.
_REFERENCE_DATE = datetime(2026, 11, 1, tzinfo=UTC)
# What a call that was never made stands in for: a call that stopped, with no usage.
_NO_CALL = LLMResponse(finish_reason="stop")
_FAN_OUT = "fan_out"


@dataclass(frozen=True, slots=True)
class RailInput:
    """The row `run_rails` reads, by the attributes `rails.RailRow` names."""

    entities: list[dict]
    relationships: list[dict]
    scope: str | None
    batch_record: tuple[Mapping, ...]
    turn_plan: Mapping | None = None


@dataclass(frozen=True, slots=True)
class FilteredProbe:
    """One probe's two passes after the filter, with counts beside them.

    Attributes:
        tag: The probe's tag.
        row_a: Pass A's row, its relations less those the filter removed.
        row_b: Pass B's row, likewise.
        removed_a: How many of pass A's relations were removed.
        removed_b: How many of pass B's relations were removed.
        groups: How many groups the two passes' derived relations formed.
        ungrouped: Derived relations no `fan_out` rule gives back, kept in both passes.
    """

    tag: str
    row_a: LabelRow
    row_b: LabelRow
    removed_a: int
    removed_b: int
    groups: int
    ungrouped: int


def _payload(row: LabelRow) -> tuple[list[dict], list[dict]]:
    """The row as a model payload: each id once, with its type, and each edge with its attributes.

    An edge's endpoint the entity list does not hold is added with the edge's type for it, as
    the agreement module reads it, so a rail that joins two payload entities finds both.
    """
    entities: list[dict] = []
    seen: set[str] = set()
    ends = [(end["id"], end["type"]) for end in row["expected_entities"]] + [
        (edge[end], edge[f"{end}_type"])  # type: ignore[literal-required]
        for edge in row["expected_relationships"]
        for end in ("source", "target")
    ]
    for entity_id, entity_type in ends:
        if entity_id not in seen:
            seen.add(entity_id)
            entities.append({"id": entity_id, "type": entity_type})
    relationships = [
        {
            "source": edge["source"],
            "target": edge["target"],
            "type": edge["predicate"],
            "properties": {name: edge[name] for name in _ATTRIBUTES if name in edge},
        }
        for edge in row["expected_relationships"]
    ]
    return entities, relationships


def batch_record(utterance: str, payload: Mapping, *, pack: object) -> tuple[dict, ...]:
    """The batch record the extraction service writes for `utterance`, with no model call.

    The utterance is pre-processed, split, item-filtered and batched as the engine's Stage 2
    does, and `_record_entry` writes one entry per batch. A gold row does not record which
    batch returned which entity, so every entry carries the whole of `payload`: a rail reads
    a batch's payload to find the batch that holds an entity, and every batch holds each one.

    Args:
        utterance: The probe's utterance.
        payload: The entities and relationships every entry records as returned.
        pack: The pack whose item filters run before batching.

    Returns:
        The entries, as the service dumps them.
    """
    pre_processed = PreProcessor().pre_process(
        utterance=utterance, conversation_history=[], reference_date=_REFERENCE_DATE
    )
    text = pre_processed.original_text
    units = filter_units(split_structured(text), pack)  # type: ignore[arg-type]
    numbered = is_numbered_list(text)
    returned = ExtractionPayload(
        entities=list(payload["entities"]), relationships=list(payload["relationships"])
    )
    entries: list[dict] = []
    start = 0
    for batch in batch_units(units, LIST_BATCH_MAX_UNITS):
        call = _batch_input(pre_processed, units, start, len(batch), numbered=numbered)
        entries.append(_record_entry(call, _NO_CALL, returned).model_dump(mode="json"))
        start += len(batch)
    return tuple(entries)


def _run(
    utterance: str,
    entities: Sequence[dict],
    relationships: Sequence[dict],
    *,
    pack: object,
    run_rails: RunRails,
    scope: str | None,
    relation_rails: bool = True,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Run the rails over copies; also return the copied edges, their endpoints now stored ids."""
    entities = copy.deepcopy(list(entities))
    relationships = copy.deepcopy(list(relationships))
    record = batch_record(
        utterance, {"entities": entities, "relationships": relationships}, pack=pack
    )
    row = RailInput(entities, relationships, scope, record)
    stored_entities, stored_relationships = run_rails(row, pack, relation_rails=relation_rails)
    return stored_entities, stored_relationships, relationships


def _key(relationship: Mapping) -> Triple:
    return (
        str(relationship.get("source", "")),
        str(relationship.get("type", "")),
        str(relationship.get("target", "")),
    )


def stored_row(row: LabelRow, *, pack: object, run_rails: RunRails, scope: str | None) -> LabelRow:
    """The row under the ids and types the graph stores, with no edge added or removed.

    The rails run with their relation phase skipped, so the row keeps the edges the labeller
    wrote: an edge a rule would add is the labeller's to write, and the judgement filter, not
    this function, decides how it counts. The row passed in is not changed.

    Args:
        row: One pass's row.
        pack: The pack whose rails run.
        run_rails: The pack's rail runner.
        scope: The probe's scope label.
    """
    entities, relationships = _payload(row)
    stored_entities, stored_relationships, _ = _run(
        row["utterance"],
        entities,
        relationships,
        pack=pack,
        run_rails=run_rails,
        scope=scope,
        relation_rails=False,
    )
    types: dict[str, str] = {}
    for entity in stored_entities:
        types.setdefault(str(entity["id"]), str(entity.get("type", "")))
    edges: list[LabelEdge] = []
    for relationship in stored_relationships:
        source, predicate, target = _key(relationship)
        edge: LabelEdge = {
            "source": source,
            "source_type": types.get(source, ""),
            "predicate": predicate,
            "target": target,
            "target_type": types.get(target, ""),
        }
        properties = relationship.get("properties") or {}
        for name in _ATTRIBUTES:
            if name in properties:
                edge[name] = properties[name]  # type: ignore[literal-required]
        edges.append(edge)
    return {
        "utterance": row["utterance"],
        "tag": row["tag"],
        "expected_entities": [{"id": i, "type": t} for i, t in types.items()],
        "expected_relationships": edges,
    }


@dataclass(frozen=True, slots=True)
class _Pass:
    """One pass's stored key for each of its relations, in order, and its derived groups."""

    keys: tuple[Triple, ...]
    groups: Mapping[Triple, GroupKey]
    ungrouped: int


class _Analysis:
    """The rails runs over one pass: which of its relations are derived, and in which group."""

    def __init__(
        self, row: LabelRow, *, pack: object, run_rails: RunRails, scope: str | None
    ) -> None:
        self._utterance = row["utterance"]
        self._pack = pack
        self._run_rails = run_rails
        self._scope = scope
        self._entities, self._relationships = _payload(row)
        _, _, remapped = self._run(self._relationships, relation_rails=False)
        self._stored = remapped
        self.keys = tuple(_key(relationship) for relationship in remapped)

    def _run(self, relationships: Sequence[dict], *, relation_rails: bool = True):
        return _run(
            self._utterance,
            self._entities,
            relationships,
            pack=self._pack,
            run_rails=self._run_rails,
            scope=self._scope,
            relation_rails=relation_rails,
        )

    def _gives_back(self, kept: Sequence[int], key: Triple) -> bool:
        """True when the rails over the relations at `kept` give back `key`."""
        _, stored, _ = self._run([self._relationships[i] for i in kept])
        return key in {_key(relationship) for relationship in stored}

    def _group(self, others: Sequence[int], key: Triple) -> GroupKey | None:
        """The first rule and team whose trigger edges alone give `key` back."""
        for rule in getattr(self._pack, "rules", ()):
            params = getattr(rule, "params", None)
            trigger = getattr(params, "trigger", None)
            if rule.kind != _FAN_OUT or not hasattr(trigger, "edge"):
                continue
            if key[1] not in {gain.edge for gain in params.gains}:
                continue
            end = "target" if trigger.members == "source" else "source"
            anchor_of = {
                i: self._stored[i][end]
                for i in others
                if self._relationships[i].get("type") == trigger.edge
            }
            for anchor in sorted(set(anchor_of.values())):
                alone = [i for i in others if anchor_of.get(i, anchor) == anchor]
                if self._gives_back(alone, key):
                    return (str(rule.id), str(anchor), key[1])
        return None

    def derived(self) -> _Pass:
        """Run the rails once per relation removed, and group each relation they give back."""
        groups: dict[Triple, GroupKey] = {}
        ungrouped = 0
        for key in dict.fromkeys(self.keys):
            others = [i for i, other in enumerate(self.keys) if other != key]
            if not self._gives_back(others, key):
                continue
            group = self._group(others, key)
            if group is None:
                ungrouped += 1
            else:
                groups[key] = group
        return _Pass(keys=self.keys, groups=groups, ungrouped=ungrouped)


def _without(row: LabelRow, keys: Sequence[Triple], removed: set[Triple]) -> LabelRow:
    edges = row["expected_relationships"]
    return {
        **row,
        "expected_relationships": [
            copy.deepcopy(edge) for edge, key in zip(edges, keys, strict=True) if key not in removed
        ],
    }


def judgement_filter(
    row_a: LabelRow,
    row_b: LabelRow,
    *,
    pack: object,
    run_rails: RunRails,
    scope: str | None,
) -> FilteredProbe:
    """Keep one relation of each (rule, team, predicate) group the two passes derive.

    Args:
        row_a: The probe's row in pass A.
        row_b: The same probe's row in pass B.
        pack: The pack whose rails derive the relations.
        run_rails: The pack's rail runner, `run_rails(row, pack, *, relation_rails)`.
        scope: The probe's scope label, which the rails read.

    Returns:
        The two rows with the removed relations left out, in the ids each pass wrote, and the
        counts.

    Raises:
        ValueError: The two rows carry different tags.
    """
    if row_a["tag"] != row_b["tag"]:
        raise ValueError("the two rows are of different probes")
    passes = [
        _Analysis(row, pack=pack, run_rails=run_rails, scope=scope).derived()
        for row in (row_a, row_b)
    ]
    held_a, held_b = set(passes[0].keys), set(passes[1].keys)
    members: dict[GroupKey, set[Triple]] = {}
    for analysed in passes:
        for key, group in analysed.groups.items():
            members.setdefault(group, set()).add(key)
    kept: set[Triple] = set()
    for keys in members.values():
        ordered = sorted(keys)
        shared = [key for key in ordered if key in held_a and key in held_b]
        kept.add(shared[0] if shared else ordered[0])
    removed = set().union(*members.values()) - kept
    filtered_a = _without(row_a, passes[0].keys, removed)
    filtered_b = _without(row_b, passes[1].keys, removed)
    return FilteredProbe(
        tag=row_a["tag"],
        row_a=filtered_a,
        row_b=filtered_b,
        removed_a=len(row_a["expected_relationships"]) - len(filtered_a["expected_relationships"]),
        removed_b=len(row_b["expected_relationships"]) - len(filtered_b["expected_relationships"]),
        groups=len(members),
        ungrouped=passes[0].ungrouped + passes[1].ungrouped,
    )
