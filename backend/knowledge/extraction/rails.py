"""The rail runner: what the engine does to an extraction payload after the model.

`run_rails(cache_row, pack)` is the one entry point. It interprets the post-LLM rules of
a pack (`backend.knowledge.ontologies.pack`) over one cache row and returns the payload
the graph stores. The runner knows the rule kinds and never a type or an edge: every
name it acts on is a parameter of a rule, so a new ontology version is a new pack.

The phases run in a fixed order, each a separate step:

1. payload: the user anchor and the reserved identities are settled, and a retired type
   is replaced. A settled entity takes no id rule and no type fallback.
2. relation: rules that add or remove an edge (`fan_out`, the relation form of
   `derived_edge`, `edge_suppress`). They read the entities under the ids the model
   wrote, because the endpoints still name those.
3. id: each entity's stored id is built, the endpoints are remapped to it, and the id
   form of `derived_edge` renames an entity from the node its edge points at. An id
   of a type no `id_exempt` rule names first loses the context its batch was read
   under, when the batch record names one (`_own_contexts`) and the context names no
   entity of the payload (`_strip_own_contexts`), so the id rules after it read that
   entity's own id. An exempt type's id keeps its context.
4. fallback: an entity whose type the pack does not define takes the manifest's
   fallback type.

Entities are then deduplicated first-wins by id and relationships by (source, target,
type). Within a phase the kinds the engine interprets run first, in the order above, and
the pack's `plugin` rules of that phase run after them, in the pack's order.

The row is the only input besides the pack: the payload, the scope, the batch record
and the turn plan. Nothing here reads a clock, a file or the graph, so the same row and
pack always give the same bytes.

The runner checks a pack itself. The first time it is handed a pack it runs the pack's
pinned cases (`require_pinned_cases`), and it refuses, with `PackLoadError`, a pack
whose case does not hold. The pack registry does not run the cases, so reading a pack
from the registry does not load this module: the extraction service must never import
it.
"""

from __future__ import annotations

import copy
import re
import weakref
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache

from backend.knowledge.extraction import id_rails
from backend.knowledge.extraction.canonical_id import canonical_id
from backend.knowledge.ontologies.pack import (
    ENDPOINT_ANCHOR,
    ENDPOINT_FIXED,
    ENDPOINT_MEMBER,
    ENDPOINT_PLUGIN_PREFIX,
    ENDPOINT_USER,
    ID,
    PAYLOAD,
    RELATION,
    ConventionRule,
    DerivedEdge,
    EdgeSuppress,
    EdgeTrigger,
    FanOut,
    GainedEdge,
    Pack,
    PackLoadError,
    Phase,
)

# The properties an edge added from a trigger edge copies from it. Dates are not
# copied: the member's own edge keeps them.
_COPIED_EDGE_PROPERTIES: tuple[str, ...] = ("temporal_status", "confidence")

_LEAD_IN_MARKS_RE = re.compile(r"^[#\s]+|[*_`]+|[\s:]+$")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class RailRow:
    """The cache row the rails read: nothing outside it reaches a rule.

    Attributes:
        entities: The payload's entity dicts, as the model wrote them.
        relationships: The payload's relationship dicts.
        scope: The row's scope label; None when the row was not classified.
        batch_record: One entry per Stage 2 call: its `units` (each a `text`, a
            `context` list and a `kind`), its `context`, its parsed `payload`, its token
            counts and its `finish`. Empty for a row extracted as one batch.
        turn_plan: The plan's payload (`entities`, `templates`, `back_references`);
            None for a row without a plan.
    """

    entities: list[dict]
    relationships: list[dict]
    scope: str | None = None
    batch_record: tuple[Mapping, ...] = ()
    turn_plan: Mapping | None = None


def rail_row(
    entities: list[dict],
    relationships: list[dict],
    *,
    scope: str | None = None,
    batch_record: Sequence[Mapping] | None = None,
    turn_plan: Mapping | None = None,
) -> RailRow:
    """Build the row from what a cache row holds.

    A row with no batch record or no plan is a row extracted as one batch with no plan:
    a rule that needs either adds nothing to it.

    Args:
        entities: The payload's entity dicts.
        relationships: The payload's relationship dicts.
        scope: The row's scope label.
        batch_record: The row's batch record, or None.
        turn_plan: The row's plan, as its payload or as the stored record that wraps the
            payload with its token counts; or None.
    """
    plan = turn_plan
    if isinstance(plan, Mapping) and "templates" not in plan and "payload" in plan:
        plan = plan["payload"]
    return RailRow(
        entities=entities,
        relationships=relationships,
        scope=scope,
        batch_record=tuple(batch_record) if batch_record else (),
        turn_plan=plan if isinstance(plan, Mapping) else None,
    )


@dataclass(frozen=True, slots=True)
class FanOutGroup:
    """What a `fan_out` endpoint plug-in is given: one member and the group it is in.

    Attributes:
        member: The member's id, as the model wrote it.
        members: The members of the group, in the order the row gives them. Under an
            edge trigger these are the members whose every trigger edge to the anchor
            is live; one with a trigger edge that is not live is left out.
        anchor: The endpoint the trigger edges share; None under a plan trigger.
        lead_in: The lead-in of the plan's template; None under an edge trigger.
        fixed_endpoint: The entity the template names as the other end, or None.
        entities: The payload's entities when the rule started.
        relationships: The payload's live relationships when the rule started: an edge
            that is not a current assertion is not shown, so a plug-in cannot build on
            one.
    """

    member: str
    members: tuple[str, ...]
    anchor: str | None
    lead_in: str | None
    fixed_endpoint: str | None
    entities: tuple[Mapping, ...]
    relationships: tuple[Mapping, ...]


def edge_is_live(relationship: Mapping) -> bool:
    """True when an edge is a current assertion: the only kind a rail may build on.

    An edge is live when its `assertion_kind` is absent or `assert`, it is not negated
    and its `temporal_status` is not `past`. Both the relationship and its `properties`
    are read, since the model writes the fields at either level.
    """
    properties = relationship.get("properties")
    properties = properties if isinstance(properties, Mapping) else {}
    for holder in (relationship, properties):
        kind = holder.get("assertion_kind")
        if kind is not None and str(kind).lower() != "assert":
            return False
        if holder.get("negated"):
            return False
        status = holder.get("temporal_status")
        if status is not None and str(status).lower() == "past":
            return False
    return True


# ---------------------------------------------------------------------------
# The runner's working state and a pack's rules, sorted once
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _State:
    """One run's working lists.

    `settled` maps an entity dict, by object identity, to its id, and `added_keys` maps
    one to the keys the payload phase added before it had an `id` key. Both follow the
    dict object, which is why a `plugin` rule must not return a copy of an entity.
    `named` holds the slug of the `id` field of every entity of the payload as the
    runner was handed it, read before any phase runs.
    """

    row: RailRow
    entities: list[dict]
    relationships: list[dict]
    settled: dict[int, str] = field(default_factory=dict)
    added_keys: dict[int, tuple[str, ...]] = field(default_factory=dict)
    named: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class _Rules:
    """A pack's post-LLM rules as the phases read them.

    `identities` holds what a settled entity goes by: the user anchor's id, each name a
    `reserved_identity` rule maps, and each id such a rule maps to.
    """

    reserved: Mapping[str, tuple[str, str]]
    identities: frozenset[str]
    exempt: frozenset[str]
    entity_id_rules: tuple[ConventionRule, ...]
    aliases: Mapping[str, Mapping[str, str]]
    registry_after: int
    derived_ids: tuple[ConventionRule, ...]
    relation_adds: tuple[ConventionRule, ...]
    relation_drops: tuple[ConventionRule, ...]
    plugins: Mapping[str, tuple[ConventionRule, ...]]


@lru_cache(maxsize=64)
def _rules(pack: Pack) -> _Rules:
    reserved: dict[str, tuple[str, str]] = {}
    for rule in pack.rules_of("reserved_identity"):
        for name in rule.params.names:
            reserved.setdefault(name, (rule.params.id, rule.params.type))
    exempt = frozenset(t for rule in pack.rules_of("id_exempt") for t in rule.params.types)
    entity_id_rules = tuple(
        rule for rule in pack.rules_in(ID) if rule.kind in ("alias_vocabulary", "typed_id")
    )
    # The engine's registry is keyed by the id the alias vocabularies give, so it is
    # read after the last of them, or straight after the slug in a pack with none.
    registry_after = max(
        (i + 1 for i, rule in enumerate(entity_id_rules) if rule.kind == "alias_vocabulary"),
        default=0,
    )
    return _Rules(
        reserved=reserved,
        identities=frozenset(
            {pack.manifest.user_id, *reserved, *(stored for stored, _ in reserved.values())}
        ),
        exempt=exempt,
        entity_id_rules=entity_id_rules,
        aliases={rule.id: dict(rule.params.aliases) for rule in pack.rules_of("alias_vocabulary")},
        registry_after=registry_after,
        derived_ids=pack.rules_of("derived_edge", ID),
        relation_adds=tuple(
            rule for rule in pack.rules_in(RELATION) if rule.kind in ("fan_out", "derived_edge")
        ),
        relation_drops=pack.rules_of("edge_suppress", RELATION),
        plugins={phase: pack.rules_of("plugin", phase) for phase in (PAYLOAD, RELATION, ID)},
    )


def _raw_id(entity: Mapping) -> str:
    """The entity's `id` field; the name stands in when it is empty or missing."""
    return str(entity.get("id") or entity.get("name") or "")


def _display_name(entity: Mapping) -> str:
    """The entity's display name; the `id` field stands in when it is missing."""
    return str(entity.get("name") or _raw_id(entity))


def _is_user(entity: Mapping, pack: Pack) -> bool:
    return _raw_id(entity).lower() == pack.manifest.user_id


def _run_plugin_rules(state: _State, pack: Pack, phase: Phase) -> None:
    """Run the phase's `plugin` rules: each is given the payload and returns it."""
    for rule in _rules(pack).plugins[phase]:
        fn = pack.plugins_by_name[rule.params.plugin].fn
        entities, relationships = fn(state.entities, state.relationships, state.row)
        state.entities, state.relationships = list(entities), list(relationships)


# ---------------------------------------------------------------------------
# Phase 1: payload
# ---------------------------------------------------------------------------


def reserved_identity(entity: Mapping, pack: Pack) -> tuple[str, str] | None:
    """The (id, type) a `reserved_identity` rule maps the entity to, or None.

    The lowered display name is looked up first, then the slug of the `id` field, since
    a reserved name may be written id-shaped. The user anchor is never reserved.
    """
    if _is_user(entity, pack):
        return None
    reserved = _rules(pack).reserved
    hit = reserved.get(_display_name(entity).lower().strip())
    if hit is None:
        hit = reserved.get(canonical_id(_raw_id(entity)))
    return hit


def _settle_entity(entity: dict, pack: Pack) -> tuple[str | None, tuple[str, ...]]:
    """Settle one entity, or replace its retired type.

    Returns:
        The id a settled entity is stored under (the user anchor's, or a reserved
        identity's, whose type is set here), or None for an entity the id phase builds;
        and the keys this step added to an entity that has no `id` key, which
        `_set_stored_id` moves behind the `id` key it adds.
    """
    before = None if "id" in entity else set(entity)
    stored = _settled_id(entity, pack)
    added = () if before is None else tuple(key for key in entity if key not in before)
    return stored, added


def _settled_id(entity: dict, pack: Pack) -> str | None:
    if _is_user(entity, pack):
        return pack.manifest.user_id
    reserved = reserved_identity(entity, pack)
    if reserved is not None:
        entity["type"] = reserved[1]
        return reserved[0]
    retired = pack.retired.get(entity.get("type", ""))
    if retired is not None:
        if retired.property_defaults:
            properties = entity.get("properties")
            if not isinstance(properties, dict):
                properties = {}
                entity["properties"] = properties
            for name, value in retired.property_defaults:
                properties.setdefault(name, value)
        entity["type"] = retired.replacement
    return None


def _set_stored_id(entity: dict, stored: str, added: Sequence[str]) -> None:
    """Write the stored id, keeping the key order the id rails always gave.

    The id rails wrote an entity's `id` before any other key they added. The payload
    phase runs before the id is built, so a key it added to an entity with no `id` key
    (`added`) is moved behind the `id` key written here.
    """
    entity["id"] = stored
    for key in added:
        entity[key] = entity.pop(key)


def _payload_phase(state: _State, pack: Pack) -> None:
    for entity in state.entities:
        final_id, added = _settle_entity(entity, pack)
        if final_id is not None:
            state.settled[id(entity)] = final_id
        if added:
            state.added_keys[id(entity)] = added
    _run_plugin_rules(state, pack, PAYLOAD)


# ---------------------------------------------------------------------------
# Phase 2: relation
# ---------------------------------------------------------------------------


def _edge_key(relationship: Mapping) -> tuple:
    return (
        relationship.get("source", ""),
        relationship.get("target", ""),
        relationship.get("type", ""),
    )


def _types_by_id(entities: Sequence[Mapping]) -> dict:
    """Each entity's type under the id the model wrote; the first of a repeated id wins."""
    types: dict = {}
    for entity in entities:
        types.setdefault(entity.get("id"), entity.get("type", ""))
    return types


def _add_edge(
    state: _State, source: object, target: object, edge: str, properties: Mapping
) -> None:
    """Add an edge between two payload entities, unless it is there or joins one to itself."""
    known = _types_by_id(state.entities)
    if source is None or target is None or source == target:
        return
    if source not in known or target not in known:
        return
    if (source, target, edge) in {_edge_key(rel) for rel in state.relationships}:
        return
    added: dict = {"source": source, "target": target, "type": edge}
    if properties:
        added["properties"] = dict(properties)
    state.relationships.append(added)


def _user_endpoint(state: _State, pack: Pack) -> object:
    """The user anchor's id as the payload writes it; the anchor is added when missing."""
    user_id = pack.manifest.user_id
    for entity in state.entities:
        if str(entity.get("id") or "").lower() == user_id:
            return entity["id"]
    state.entities.append({"id": user_id, "name": user_id, "type": pack.manifest.user_type})
    return user_id


def _lead_in_key(text: object) -> str:
    """A lead-in as the batch record's context writes it: no marks, no colon, one case."""
    bare = _LEAD_IN_MARKS_RE.sub("", str(text or ""))
    return _WHITESPACE_RE.sub(" ", bare).strip().casefold()


def _edge_groups(
    trigger: EdgeTrigger, relationships: Sequence[Mapping], types: Mapping
) -> list[tuple[object, list[tuple[object, list[Mapping]]]]]:
    """The trigger edges by the endpoint they share: (anchor, [(member, its edges)]).

    A member the payload joins to one anchor by several trigger edges is listed once,
    with every one of those edges in the payload's order.
    """
    groups: dict = {}
    for relationship in relationships:
        if relationship.get("type") != trigger.edge:
            continue
        source, target = relationship.get("source"), relationship.get("target")
        if trigger.source_types and types.get(source) not in trigger.source_types:
            continue
        if trigger.target_types and types.get(target) not in trigger.target_types:
            continue
        member, anchor = (source, target) if trigger.members == "source" else (target, source)
        members = groups.setdefault(anchor, [])
        for seen, edges in members:
            if seen == member:
                edges.append(relationship)
                break
        else:
            members.append((member, [relationship]))
    return list(groups.items())


def _plan_groups(
    predicate: str, member_type: str, state: _State
) -> list[tuple[Mapping, list[object]]]:
    """The plan's templates that name `predicate`, each with its lead-in's members.

    A batch belongs to a lead-in when one of its units carries the lead-in in its
    context. The members are the payload's entities of `member_type` that one of those
    batches' own payloads holds, in the payload's order.
    """
    plan = state.row.turn_plan or {}
    groups: list[tuple[Mapping, list[object]]] = []
    for template in plan.get("templates") or ():
        named = [template.get("predicate")]
        named += [gain.get("predicate") for gain in template.get("fan_out") or ()]
        lead_in = _lead_in_key(template.get("lead_in"))
        if predicate not in named or not lead_in:
            continue
        in_batches: set = set()
        for batch in state.row.batch_record:
            contexts = [c for unit in batch.get("units") or () for c in unit.get("context") or ()]
            if lead_in not in {_lead_in_key(context) for context in contexts}:
                continue
            payload = batch.get("payload") or {}
            in_batches.update(entity.get("id") for entity in payload.get("entities") or ())
        members: list[object] = []
        for entity in state.entities:
            member = entity.get("id")
            if entity.get("type") == member_type and member in in_batches and member not in members:
                members.append(member)
        groups.append((template, members))
    return groups


def _fixed_endpoint(template: Mapping, entities: Sequence[Mapping]) -> object:
    """The payload entity the template's `fixed_endpoint` names, or None."""
    named = template.get("fixed_endpoint")
    if not named:
        return None
    for entity in entities:
        if entity.get("id") == named:
            return named
    slug = canonical_id(str(named))
    for entity in entities:
        if canonical_id(str(entity.get("id") or "")) == slug:
            return entity.get("id")
    return None


def _gain(
    state: _State,
    pack: Pack,
    gain: GainedEdge,
    group: FanOutGroup,
    properties: Mapping,
    at_start: Sequence[Mapping],
) -> None:
    """Add one gained edge for one member, when its conditions hold."""
    if gain.scopes and state.row.scope not in gain.scopes:
        return
    if gain.unless_member_has:
        end = "source" if gain.source == ENDPOINT_MEMBER else "target"
        if any(rel.get("type") == gain.edge and rel.get(end) == group.member for rel in at_start):
            return
    ends: list[object] = []
    for reference in (gain.source, gain.target):
        if reference == ENDPOINT_MEMBER:
            ends.append(group.member)
        elif reference == ENDPOINT_ANCHOR:
            ends.append(group.anchor)
        elif reference == ENDPOINT_FIXED:
            ends.append(group.fixed_endpoint)
        elif reference == ENDPOINT_USER:
            ends.append(_user_endpoint(state, pack))
        else:
            name = reference[len(ENDPOINT_PLUGIN_PREFIX) :]
            ends.append(pack.plugins_by_name[name].fn(group, state.row))
    _add_edge(state, ends[0], ends[1], gain.edge, properties)


def _fan_out(state: _State, pack: Pack, params: FanOut) -> None:
    """Give each member of each group the rule's edges.

    Under an edge trigger a member gains nothing unless every trigger edge that joins
    it to the group's anchor is live, and each added edge copies the first such edge's
    `temporal_status` and confidence. Under a plan trigger the added edges carry the
    values the rule states.
    """
    at_start = tuple(state.relationships)
    live = tuple(rel for rel in at_start if edge_is_live(rel))
    entities = tuple(state.entities)
    if isinstance(params.trigger, EdgeTrigger):
        types = _types_by_id(entities)
        for anchor, members in _edge_groups(params.trigger, at_start, types):
            # Every trigger edge of a member is an input, and a rail adds an edge only when
            # every input it reads is live. The payload may state the edge twice, and the
            # dedupe step stores the first copy: built on a live second copy, the added
            # edges would stand on a stored edge that is not live. A member that fails is
            # not listed in its group either.
            passing = [
                (member, edges[0]) for member, edges in members if all(map(edge_is_live, edges))
            ]
            names = tuple(member for member, _ in passing)
            for member, edge in passing:
                own = edge.get("properties")
                own = own if isinstance(own, Mapping) else {}
                copied = {name: own[name] for name in _COPIED_EDGE_PROPERTIES if name in own}
                group = FanOutGroup(member, names, anchor, None, None, entities, live)
                for gain in params.gains:
                    _gain(state, pack, gain, group, copied, at_start)
        return
    trigger = params.trigger
    for template, members in _plan_groups(trigger.predicate, trigger.member_type, state):
        fixed = _fixed_endpoint(template, entities)
        lead_in = str(template.get("lead_in") or "")
        for member in members:
            group = FanOutGroup(member, tuple(members), None, lead_in, fixed, entities, live)
            for gain in params.gains:
                _gain(state, pack, gain, group, dict(gain.properties), at_start)


def _build_from_property(state: _State, params: DerivedEdge) -> None:
    """The relation form of `derived_edge`: a property becomes a node and an edge to it."""
    for entity in tuple(state.entities):
        properties = entity.get("properties")
        if entity.get("type") != params.source_type or not isinstance(properties, Mapping):
            continue
        value = properties.get(params.property)
        if not isinstance(value, str) or not value.strip():
            continue
        node_id = value.strip()
        if all(known.get("id") != node_id for known in state.entities):
            state.entities.append({"id": node_id, "name": node_id, "type": params.target_type})
        _add_edge(state, entity.get("id"), node_id, params.edge, dict(params.edge_properties))


def _is_end_of(
    node: object, end: str, edges: tuple[str, ...], relationships: Sequence[Mapping]
) -> bool:
    """True when no edge is asked for, or `node` is the `end` of one of `edges`."""
    return not edges or any(
        rel.get("type") in edges and rel.get(end) == node for rel in relationships
    )


def _suppress(state: _State, params: EdgeSuppress) -> None:
    """Drop the edges the pattern names; the conditions read the edges as the rule found them."""
    at_start = tuple(state.relationships)
    types = _types_by_id(state.entities)
    kept: list[dict] = []
    for relationship in state.relationships:
        source, target = relationship.get("source"), relationship.get("target")
        matches = (
            relationship.get("type") == params.edge
            and (not params.source_types or types.get(source) in params.source_types)
            and (not params.target_types or types.get(target) in params.target_types)
            and _is_end_of(source, "target", params.source_is_target_of, at_start)
            and _is_end_of(source, "source", params.source_is_source_of, at_start)
            and _is_end_of(target, "target", params.target_is_target_of, at_start)
            and _is_end_of(target, "source", params.target_is_source_of, at_start)
        )
        if not matches:
            kept.append(relationship)
    state.relationships = kept


def _relation_phase(state: _State, pack: Pack) -> None:
    rules = _rules(pack)
    for rule in rules.relation_adds:
        if isinstance(rule.params, FanOut):
            _fan_out(state, pack, rule.params)
        else:
            _build_from_property(state, rule.params)
    for rule in rules.relation_drops:
        _suppress(state, rule.params)
    _run_plugin_rules(state, pack, RELATION)


# ---------------------------------------------------------------------------
# Phase 3: id
# ---------------------------------------------------------------------------


def _own_contexts(batch_record: Sequence[Mapping]) -> dict[str, tuple[str, ...]]:
    """The context slugs each written id may lose from its start, longest first.

    A list item is extracted with the headings, the lead-in and the parent items above
    it as its context, and the model sometimes writes that context in front of the
    item's id. The batch record says which contexts a batch was read under and which
    entities it returned, per batch and not per entity. So:

    - an entity may lose a context line of any unit of a batch whose own payload holds
      the entity's `id` as the model wrote it. Which unit of the batch the entity came
      from is not recorded, so every unit's lines are offered, each line on its own;
    - an entity several batches hold is offered the lines of all of them;
    - a batch that does not hold the entity offers it nothing, and neither does an
      entry's own `context` list, the earlier units the call was shown for reference;
    - an entity no batch holds, and every entity of a row with no batch record, is
      offered nothing.

    What is offered is stripped only from an entity the id phase builds an id for, of a
    type no `id_exempt` rule names (`_build_entity_id`): a settled entity and an exempt
    type keep their ids whatever is offered. Which offered line is dropped, and when,
    is `_strip_own_contexts`.

    A line is slugged with `canonical_id`, the function an id is slugged with, so the
    two compare as written: a line that opens with a leading word keeps it.

    A second run of the runner over its own output, with the same record, gives the
    same ids for most rows and not for all. Two shapes differ:

    - the lookup is by the id the model wrote, so a stored id that equals an id another
      batch holds is offered that batch's lines on the second run;
    - a line is kept when it names an entity of the payload, by the id the model wrote.
      Where the pack stores that entity under another id (an alias), the second run's
      payload holds no entity the line names, and the strip goes through.

    A replay starts from the cache row, never from stored output.
    """
    offered: dict[str, set[str]] = {}
    for batch in batch_record:
        slugs = {
            canonical_id(str(line))
            for unit in batch.get("units") or ()
            for line in unit.get("context") or ()
        }
        payload = batch.get("payload") or {}
        for entity in payload.get("entities") or ():
            written = entity.get("id")
            if isinstance(written, str):
                offered.setdefault(written, set()).update(slugs)
    return {
        written: tuple(sorted(slugs - {""}, key=lambda slug: (-len(slug), slug)))
        for written, slugs in offered.items()
    }


def _strip_own_contexts(
    entity_id: str, contexts: Sequence[str], named: frozenset[str], pack: Pack
) -> str:
    """Drop the offered contexts from the start of the id, as whole words.

    The line tried is the longest of `contexts` that is the id or starts it with a
    hyphen after; `_own_contexts` gives them longest first. It is dropped, with its
    hyphen, only when all of these hold:

    1. something is left besides hyphens. An id that is the line and nothing more is
       the line's own entity. An id that is the line and hyphens is kept too:
       `canonical_id` gives text with no ASCII letter or digit back lowered, with a
       hyphen for each space, so such an id can end in one;
    2. the line is not in `named`. A heading the model also wrote as an entity is part
       of its items' own ids: with `redis` and `postgres` in the payload,
       `redis-cluster` and `postgres-cluster` are two things, not one `cluster`;
    3. what is left is not the user anchor's id, a name a `reserved_identity` rule
       maps, or the id such a rule maps to. The payload phase has already settled the
       entities that go by those, and a strip must not make another.

    After a drop the id is read again, so an id that carries two lines of its context
    loses both. A line that is refused ends the strip with the id as it then stands: no
    shorter line is tried, since a shorter line would take a part of a name the
    refusal kept whole.

    Args:
        entity_id: The slug of the entity's `id` field.
        contexts: The slugs the batch record offers the entity, longest first.
        named: The slug of the `id` field of every entity of the row's payload.
        pack: The pack whose reserved names and user anchor are read.
    """
    identities = _rules(pack).identities
    while True:
        line = next(
            (slug for slug in contexts if entity_id == slug or entity_id.startswith(f"{slug}-")),
            None,
        )
        if line is None:
            return entity_id
        rest = entity_id[len(line) + 1 :]
        if not rest.strip("-") or line in named or rest in identities:
            return entity_id
        entity_id = rest


def _build_entity_id(
    entity: dict,
    pack: Pack,
    own_contexts: Sequence[str] = (),
    named: frozenset[str] = frozenset(),
) -> tuple[str, bool]:
    """One entity's stored id, from its `id` field and the pack's id rules.

    The slug of the `id` field first loses the `own_contexts` it starts with
    (`_strip_own_contexts`), unless an `id_exempt` rule names the type. It then goes
    through the `alias_vocabulary` and `typed_id` rules in the pack's order; the
    engine's registry is read after the last alias vocabulary, and a hit sets the type
    and ends the steps; a type no `id_exempt` rule names then loses its leading words.

    Args:
        entity: The entity, its type already settled by the payload phase.
        pack: The pack whose id rules run.
        own_contexts: The context slugs the batch record offers this entity
            (`_own_contexts`), longest first; empty for an entity it offers none.
        named: The slug of the `id` field of every entity of the row's payload, which
            no context that equals one is stripped as.

    Returns:
        The id, and True when the registry settled the entity.
    """
    rules = _rules(pack)
    entity_type = str(entity.get("type", ""))
    entity_id = canonical_id(_raw_id(entity))
    if entity_type not in rules.exempt:
        entity_id = _strip_own_contexts(entity_id, own_contexts, named, pack)
    for position in range(len(rules.entity_id_rules) + 1):
        if position == rules.registry_after:
            registered = id_rails.CANONICAL_REGISTRY.get(entity_id)
            if registered is not None:
                # The id is written first, so an entity with neither key gets `id`
                # before `type`, as the id rails gave it.
                entity["id"], entity["type"] = registered
                return registered[0], True
        if position == len(rules.entity_id_rules):
            break
        rule = rules.entity_id_rules[position]
        params = rule.params
        if rule.kind == "typed_id":
            if entity_type == params.type:
                built = pack.plugins_by_name[params.plugin].fn(entity, entity_id, None)
                entity_id = entity_id if built is None else str(built)
            continue
        if params.types and entity_type not in params.types:
            continue
        aliases = rules.aliases[rule.id]
        if params.match == "name":
            entity_id = aliases.get(_display_name(entity).lower().strip(), entity_id)
        else:
            entity_id = aliases.get(entity_id, entity_id)
    if entity_type not in rules.exempt:
        entity_id = id_rails.n1_rail(entity_id)
    return entity_id, False


def _derive_ids(
    entities: list[dict], relationships: list[dict], pack: Pack, rule: ConventionRule
) -> dict[str, str]:
    """The id form of `derived_edge`: rename each entity its one qualifying target dates.

    For each entity of the rule's source type, the distinct targets of its `edge` edges
    that are entities of the target type are offered to the rule's plug-in; a target the
    plug-in answers None for does not qualify. With exactly one qualifying target the
    entity takes the id the plug-in built, and its display name gains the target's id
    unless it already ends with it. Entities are renamed in place; edges are not touched.

    Returns:
        Old id to new id, for every entity whose id changed.
    """
    params = rule.params
    fn = pack.plugins_by_name[params.plugin].fn
    targets: dict = {}
    for entity in entities:
        if entity.get("type") == params.target_type:
            targets.setdefault(entity.get("id"), entity)
    targets_by_source: dict = {}
    for relationship in relationships:
        target = relationship.get("target")
        if relationship.get("type") == params.edge and target in targets:
            seen = targets_by_source.setdefault(relationship.get("source"), [])
            if target not in seen:
                seen.append(target)

    renames: dict[str, str] = {}
    for entity in entities:
        if entity.get("type") != params.source_type:
            continue
        old_id = str(entity.get("id", ""))
        built = [
            (target, new_id)
            for target in targets_by_source.get(entity.get("id"), ())
            if (new_id := fn(entity, old_id, targets[target])) is not None
        ]
        if len(built) != 1:
            continue
        target_id, new_id = str(built[0][0]), str(built[0][1])
        name = entity.get("name")
        if (
            params.name_gains_target_id
            and isinstance(name, str)
            and name
            and not name.endswith(target_id)
        ):
            entity["name"] = f"{name} {target_id}"
        entity["id"] = new_id
        if new_id != old_id:
            renames[old_id] = new_id
    return renames


def derived_ids(entities: list[dict], relationships: list[dict], pack: Pack) -> dict[str, str]:
    """Run the pack's id-phase `derived_edge` rules over normalized entities.

    Args:
        entities: The payload's entity dicts, their stored ids already built.
        relationships: The payload's relationship dicts, endpoints already remapped.
        pack: The pack whose rules run.

    Returns:
        Old id to new id, for every entity a rule renamed. Edges are not touched.

    Raises:
        PackLoadError: One of the pack's pinned cases does not hold.
    """
    require_pinned_cases(pack)
    renames: dict[str, str] = {}
    for rule in _rules(pack).derived_ids:
        renames.update(_derive_ids(entities, relationships, pack, rule))
    return renames


def _remap_endpoints(relationships: list[dict], id_map: Mapping) -> None:
    for relationship in relationships:
        source = relationship.get("source", "")
        target = relationship.get("target", "")
        if source in id_map:
            relationship["source"] = id_map[source]
        if target in id_map:
            relationship["target"] = id_map[target]


def _id_phase(state: _State, pack: Pack) -> None:
    id_map: dict = {}
    own_contexts = _own_contexts(state.row.batch_record)
    for entity in state.entities:
        written = entity.get("id", "")
        stored = state.settled.get(id(entity))
        if stored is None:
            offered = own_contexts.get(written, ()) if isinstance(written, str) else ()
            stored, by_registry = _build_entity_id(entity, pack, offered, state.named)
            if by_registry:
                state.settled[id(entity)] = stored
        _set_stored_id(entity, stored, state.added_keys.get(id(entity), ()))
        id_map[written] = stored
    _remap_endpoints(state.relationships, id_map)
    for rule in _rules(pack).derived_ids:
        renames = _derive_ids(state.entities, state.relationships, pack, rule)
        _remap_endpoints(state.relationships, renames)
    _run_plugin_rules(state, pack, ID)


# ---------------------------------------------------------------------------
# Phase 4: the type fallback, then the runner
# ---------------------------------------------------------------------------


def _fallback_phase(state: _State, pack: Pack) -> None:
    for entity in state.entities:
        if id(entity) not in state.settled and entity.get("type", "") not in pack.type_names:
            entity["type"] = pack.manifest.fallback_type


def _dedupe(entities: list[dict], relationships: list[dict]) -> tuple[list[dict], list[dict]]:
    """Entities first-wins by id, relationships first-wins by (source, target, type)."""
    seen: set = set()
    kept_entities: list[dict] = []
    for entity in entities:
        if entity["id"] not in seen:
            seen.add(entity["id"])
            kept_entities.append(entity)
    seen_edges: set[tuple] = set()
    kept_relationships: list[dict] = []
    for relationship in relationships:
        key = _edge_key(relationship)
        if key not in seen_edges:
            seen_edges.add(key)
            kept_relationships.append(relationship)
    return kept_entities, kept_relationships


# The phases, in the runner's order. Each is a step of its own, so a caller that must
# leave one out (a replay that takes its edges as stored) drops that entry.
PHASES: tuple[tuple[str, Callable[[_State, Pack], None]], ...] = (
    ("payload", _payload_phase),
    ("relation", _relation_phase),
    ("id", _id_phase),
    ("fallback", _fallback_phase),
)


def _run_phases(cache_row: RailRow, pack: Pack) -> tuple[list[dict], list[dict]]:
    """The phases and the dedupe over one row, with no check of the pack."""
    state = _State(
        row=cache_row,
        entities=list(cache_row.entities),
        relationships=list(cache_row.relationships),
        named=frozenset(canonical_id(_raw_id(entity)) for entity in cache_row.entities),
    )
    for _, phase in PHASES:
        phase(state, pack)
    return _dedupe(state.entities, state.relationships)


# What `check_pinned_cases` gave for each pack the runner has checked: the ids of the
# rules whose case does not hold. A pack is keyed by the object and held weakly, so one
# that is dropped is forgotten. Besides the `_rules` cache this is the module's only
# state, and it never changes what a row gives: it decides only whether a pack runs.
_CASE_RESULTS: weakref.WeakKeyDictionary[Pack, tuple[str, ...]] = weakref.WeakKeyDictionary()


def require_pinned_cases(pack: Pack) -> None:
    """Refuse a pack whose pinned cases do not hold.

    The cases run the first time the runner is handed the pack. What they gave is kept,
    so each later call is a lookup, and a refused pack is refused again without running
    its cases again.

    Args:
        pack: The pack about to run.

    Raises:
        PackLoadError: Naming each rule whose pinned case does not hold.
    """
    failed = _CASE_RESULTS.get(pack)
    if failed is None:
        failed = tuple(check_pinned_cases(pack))
        _CASE_RESULTS[pack] = failed
    if failed:
        raise PackLoadError(f"pack {pack.version}: pinned case(s) do not hold: {', '.join(failed)}")


def run_rails(cache_row: RailRow, pack: Pack) -> tuple[list[dict], list[dict]]:
    """Run a pack's post-LLM rules over one cache row.

    The entity and relationship dicts of the row are changed in place; its lists are
    not. Deterministic: the same row and pack give the same output.

    Args:
        cache_row: The row, built with `rail_row`.
        pack: The pack whose rules run.

    Returns:
        The entities and relationships the graph stores.

    Raises:
        PackLoadError: One of the pack's pinned cases does not hold. The row is not
            touched.
    """
    require_pinned_cases(pack)
    return _run_phases(cache_row, pack)


def normalize_entity(entity: dict, pack: Pack) -> None:
    """Build one entity's stored id and type, in place, as `run_rails` would alone.

    The entity is settled or its retired type replaced, its id is built, and an unknown
    type takes the fallback. No relationship is read, so no `derived_edge` applies.

    Raises:
        PackLoadError: One of the pack's pinned cases does not hold.
    """
    require_pinned_cases(pack)
    stored, added = _settle_entity(entity, pack)
    settled = stored is not None
    if stored is None:
        stored, settled = _build_entity_id(entity, pack)
    _set_stored_id(entity, stored, added)
    if not settled and entity.get("type", "") not in pack.type_names:
        entity["type"] = pack.manifest.fallback_type


def check_pinned_cases(pack: Pack) -> list[str]:
    """Run every rule's pinned case through the pack.

    The cases go through the phases directly and not through `run_rails`, which calls
    this function for a pack it has not checked yet.

    Returns:
        The ids of the rules whose case does not give its expected payload; empty when
        every case holds.
    """
    failed: list[str] = []
    for rule in pack.rules:
        case = rule.pinned_case
        if case is None:
            continue
        row = rail_row(
            copy.deepcopy(list(case.entities)),
            copy.deepcopy(list(case.relationships)),
            scope=case.scope,
            batch_record=copy.deepcopy(case.batch_record),
            turn_plan=copy.deepcopy(case.turn_plan),
        )
        entities, relationships = _run_phases(row, pack)
        if entities != list(case.expected_entities) or relationships != list(
            case.expected_relationships
        ):
            failed.append(rule.id)
    return failed
