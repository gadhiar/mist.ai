"""The ontology pack: one ontology version's types and rules, as data the engine interprets.

A pack holds node and edge definitions, a manifest, convention rules and registered
plug-ins. The engine (`backend.knowledge.extraction.rails` after the model,
`backend.knowledge.extraction.pre_llm` before it) knows the rule kinds and never a type
or an edge: every name it acts on is a parameter of a rule in the pack, so a new ontology
version is a new pack and no engine edit.

`build_pack` is the loader. It refuses a pack whose rules name something the pack does
not define, whose plug-ins are not registered or not pinned, or whose pair table does not
build, and it computes the pack's two identities:

- `pack_prompt_hash` covers what feeds the model: the types, the edges and their pairs,
  the pre-LLM rules and the manifest's prompt-visible fields.
- `pack_rails_hash` covers what runs after the model: the post-LLM rules by phase, the
  plug-ins and the manifest's post-LLM data (the fallback type, the retired types, the
  user anchor).

The labelling text is held beside the rules and feeds neither hash, and so is a rule's
pinned case: a case is evidence about a rule, not part of what the rule does.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import inspect
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Literal

from backend.errors import MistError
from backend.knowledge.ontologies.base import EdgeTypeDefinition, NodeTypeDefinition

Phase = Literal["pre_llm", "payload", "relation", "id"]

PRE_LLM: Phase = "pre_llm"
PAYLOAD: Phase = "payload"
RELATION: Phase = "relation"
ID: Phase = "id"

# The runner's order. The type fallback follows the last of them and is manifest data.
POST_LLM_PHASES: tuple[Phase, ...] = (PAYLOAD, RELATION, ID)

# What a `fan_out` edge's endpoint may be: the member itself, the user, the endpoint the
# trigger edges share, the fixed endpoint the plan's template names, or a registered
# relation plug-in written `plugin:<name>`.
ENDPOINT_MEMBER = "member"
ENDPOINT_USER = "user"
ENDPOINT_ANCHOR = "anchor"
ENDPOINT_FIXED = "fixed_endpoint"
ENDPOINT_PLUGIN_PREFIX = "plugin:"

_TYPE_NAMES = {"names": "type"}
_EDGE_NAMES = {"names": "edge"}
_PLUGIN_NAME = {"names": "plugin"}
_ENDPOINT = {"names": "endpoint"}


class PackLoadError(MistError):
    """A pack was refused at load: the message names what is wrong with it."""


# ---------------------------------------------------------------------------
# Rule kinds: one frozen parameter class each
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IdExempt:
    """Types whose ids follow their own convention, so the leading-word rail skips them."""

    types: tuple[str, ...] = field(metadata=_TYPE_NAMES)


@dataclass(frozen=True)
class TypedId:
    """A type whose id an id plug-in builds; a None from the plug-in keeps the id."""

    type: str = field(metadata=_TYPE_NAMES)
    plugin: str = field(metadata=_PLUGIN_NAME)


@dataclass(frozen=True)
class ReservedIdentity:
    """Display names and id slugs that always resolve to one (id, type).

    Attributes:
        names: Lowered display names, or slugs of the `id` field.
        id: The id every one of them is stored under.
        type: The type the entity is given.
    """

    names: tuple[str, ...]
    id: str
    type: str = field(metadata=_TYPE_NAMES)


@dataclass(frozen=True)
class AliasVocabulary:
    """An alias map from a spelling to the id it is stored under.

    Attributes:
        aliases: (spelling, id) pairs.
        match: `name` looks up the lowered display name and replaces the slug, for a
            spelling a slug cannot hold; `id` looks up the id the earlier rules gave.
        types: The types the map applies to; empty means every type.
    """

    aliases: tuple[tuple[str, str], ...]
    match: Literal["name", "id"] = "id"
    types: tuple[str, ...] = field(default=(), metadata=_TYPE_NAMES)


@dataclass(frozen=True)
class DerivedEdge:
    """A link between an entity of one type and a node of another, read or built.

    Declared in the id phase, the rule reads the entity's `edge` edges: when exactly one
    distinct target of `target_type` qualifies, the id plug-in builds the entity's id
    from it. Declared in the relation phase, the rule builds: an entity whose properties
    hold `property` gains a node of `target_type` with that value as its id, and an
    `edge` edge to it carrying `edge_properties`.

    Attributes:
        source_type: The type of the entity the rule acts on.
        edge: The edge between the two.
        target_type: The type of the node at the other end.
        plugin: Id phase only: the id plug-in, called with the entity, its id and the
            target; None from it means the target does not qualify.
        name_gains_target_id: Id phase only: the display name gains the target's id.
        property: Relation phase only: the property the node is built from.
        edge_properties: Relation phase only: the properties the added edge carries.
    """

    source_type: str = field(metadata=_TYPE_NAMES)
    edge: str = field(metadata=_EDGE_NAMES)
    target_type: str = field(metadata=_TYPE_NAMES)
    plugin: str | None = field(default=None, metadata=_PLUGIN_NAME)
    name_gains_target_id: bool = True
    property: str | None = None
    edge_properties: tuple[tuple[str, str | float | bool], ...] = ()


@dataclass(frozen=True)
class EdgeTrigger:
    """A `fan_out` trigger: an edge pattern in the payload.

    The edges of type `edge` whose endpoint types match are grouped by the endpoint they
    share (the anchor); the other endpoint of each is a member.

    Attributes:
        edge: The trigger edge.
        members: Which endpoint of a trigger edge is the member.
        source_types: The types its source may have; empty means any.
        target_types: The types its target may have; empty means any.
    """

    edge: str = field(metadata=_EDGE_NAMES)
    members: Literal["source", "target"] = "source"
    source_types: tuple[str, ...] = field(default=(), metadata=_TYPE_NAMES)
    target_types: tuple[str, ...] = field(default=(), metadata=_TYPE_NAMES)


@dataclass(frozen=True)
class PlanTrigger:
    """A `fan_out` trigger: a lead-in whose template in the row's plan names `predicate`.

    The members are every entity of `member_type` that the batch record places in the
    lead-in's batches.
    """

    predicate: str = field(metadata=_EDGE_NAMES)
    member_type: str = field(metadata=_TYPE_NAMES)


@dataclass(frozen=True)
class GainedEdge:
    """One edge a `fan_out` member gains.

    Attributes:
        edge: The edge type added.
        source: Its source, as an endpoint reference (`ENDPOINT_*`).
        target: Its target, likewise.
        scopes: The row scopes under which it is added; empty means any scope.
        unless_member_has: Skip a member that already holds an edge of this type at its
            own end.
        properties: The values the edge carries under a plan trigger. Under an edge
            trigger the edge copies its member's input edge instead.
    """

    edge: str = field(metadata=_EDGE_NAMES)
    source: str = field(metadata=_ENDPOINT)
    target: str = field(metadata=_ENDPOINT)
    scopes: tuple[str, ...] = ()
    unless_member_has: bool = False
    properties: tuple[tuple[str, str | float | bool], ...] = ()


@dataclass(frozen=True)
class FanOut:
    """A trigger, and the edges each of its members gains."""

    trigger: EdgeTrigger | PlanTrigger
    gains: tuple[GainedEdge, ...]


@dataclass(frozen=True)
class EdgeSuppress:
    """An edge pattern dropped after the fan-out rules have run.

    An edge of type `edge` is dropped when its endpoint types match and every listed
    condition holds: `source_is_target_of` means some edge of one of those types ends at
    this edge's source, and the other three read the same way.
    """

    edge: str = field(metadata=_EDGE_NAMES)
    source_types: tuple[str, ...] = field(default=(), metadata=_TYPE_NAMES)
    target_types: tuple[str, ...] = field(default=(), metadata=_TYPE_NAMES)
    source_is_target_of: tuple[str, ...] = field(default=(), metadata=_EDGE_NAMES)
    source_is_source_of: tuple[str, ...] = field(default=(), metadata=_EDGE_NAMES)
    target_is_target_of: tuple[str, ...] = field(default=(), metadata=_EDGE_NAMES)
    target_is_source_of: tuple[str, ...] = field(default=(), metadata=_EDGE_NAMES)


@dataclass(frozen=True)
class PluginRule:
    """A convention the kinds cannot state: the named plug-in is the whole rule."""

    plugin: str = field(metadata=_PLUGIN_NAME)


@dataclass(frozen=True)
class ItemFilter:
    """Units in a structural state are dropped before the model sees them.

    Attributes:
        state: The structural state the engine recognises (`unchecked`, `checked`).
        keep_context_words: A unit is kept when its lead-in, or a heading or parent
            item above it, holds one of these as a whole word or phrase, in any case,
            singular or plural.
    """

    state: str
    keep_context_words: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlanVocabulary:
    """What the turn plan's template may name."""

    entity_types: tuple[str, ...] = field(default=(), metadata=_TYPE_NAMES)
    predicates: tuple[str, ...] = field(default=(), metadata=_EDGE_NAMES)
    fan_out_predicates: tuple[str, ...] = field(default=(), metadata=_EDGE_NAMES)


@dataclass(frozen=True)
class PromptFragment:
    """A rule's prompt text and examples, placed by the prompt assembler."""

    text: str
    position: int = 0
    examples: tuple[str, ...] = ()


RuleParams = (
    IdExempt
    | TypedId
    | ReservedIdentity
    | AliasVocabulary
    | DerivedEdge
    | FanOut
    | EdgeSuppress
    | PluginRule
    | ItemFilter
    | PlanVocabulary
    | PromptFragment
)

# Each kind: its parameter class and the phases its interpreter runs in. A rule declares
# its phase by what it does to the payload, and the loader refuses a phase the kind's
# interpreter does not have.
KINDS: Mapping[str, tuple[type, tuple[Phase, ...]]] = {
    "id_exempt": (IdExempt, (ID,)),
    "typed_id": (TypedId, (ID,)),
    "reserved_identity": (ReservedIdentity, (PAYLOAD,)),
    "alias_vocabulary": (AliasVocabulary, (ID,)),
    "derived_edge": (DerivedEdge, (ID, RELATION)),
    "fan_out": (FanOut, (RELATION,)),
    "edge_suppress": (EdgeSuppress, (RELATION,)),
    "plugin": (PluginRule, POST_LLM_PHASES),
    "item_filter": (ItemFilter, (PRE_LLM,)),
    "plan_vocabulary": (PlanVocabulary, (PRE_LLM,)),
    "prompt_fragment": (PromptFragment, (PRE_LLM,)),
}

ITEM_STATES: tuple[str, ...] = ("unchecked", "checked")


# ---------------------------------------------------------------------------
# Rules, plug-ins and the manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PinnedCase:
    """A cache row and the payload the pack's rails must turn it into.

    Attributes:
        entities: The row's entities, as the model wrote them.
        relationships: The row's relationships.
        expected_entities: The entities `run_rails` must return.
        expected_relationships: The relationships it must return.
        scope: The row's scope label.
        batch_record: The row's batch record; None for a row without one.
        turn_plan: The row's plan; None for a row without one.
    """

    entities: tuple[Mapping, ...]
    relationships: tuple[Mapping, ...]
    expected_entities: tuple[Mapping, ...]
    expected_relationships: tuple[Mapping, ...]
    scope: str | None = None
    batch_record: tuple[Mapping, ...] | None = None
    turn_plan: Mapping | None = None


@dataclass(frozen=True)
class ConventionRule:
    """One rule of a pack: a kind, its phase and its parameters.

    Attributes:
        id: The rule's name in the pack, which its register row cites.
        kind: A key of `KINDS`.
        phase: When the rule runs, declared by what it does to the payload: a payload
            rule merges or retypes entities, a relation rule adds or removes an edge,
            an id rule changes ids and remaps endpoints only.
        params: The kind's parameters.
        prompt_rule: The prompt rule the convention derives from. A relation rule must
            name one: a rail adds or removes an edge only for a convention the prompt
            states.
        pinned_case: A row and the payload the pack must produce from it. Required for
            a rule that names a plug-in.
    """

    id: str
    kind: str
    phase: Phase
    params: RuleParams
    prompt_rule: str | None = None
    pinned_case: PinnedCase | None = None


@dataclass(frozen=True)
class Plugin:
    """A function registered under the name a rule cites, for what data cannot state.

    One signature per use, all pure and deterministic given their arguments:

    - id phase, named by a `typed_id` or `derived_edge` rule:
      `(entity, entity_id, target) -> str | None`. `entity` is the entity as the payload
      holds it, `entity_id` the id built so far, `target` the edge's other entity or
      None. The result is the new id, or None for no change.
    - relation phase, named by a `fan_out` endpoint `plugin:<name>`:
      `(group, row) -> str | None`, where `group` is the runner's `FanOutGroup` and
      `row` its `RailRow`. The result is the id of a payload entity, or None for no
      edge.
    - any post-LLM phase, named by a `plugin` rule:
      `(entities, relationships, row) -> (entities, relationships)`.

    Attributes:
        name: The name rules cite.
        phase: The phase the plug-in runs in; it must equal the citing rule's.
        fn: The function.
    """

    name: str
    phase: Phase
    fn: Callable[..., object]


@dataclass(frozen=True)
class RetiredType:
    """A type an earlier version had, and the type that replaces it.

    Attributes:
        name: The retired type.
        replacement: The type an entity of the retired type is given.
        property_defaults: Properties such an entity gains unless it has them.
    """

    name: str
    replacement: str
    property_defaults: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Manifest:
    """A pack's version and the data the engine applies without a rule.

    Attributes:
        version: The ontology version, semver.
        fallback_type: The type an entity of an unknown type is given.
        user_id: The id of the user anchor. An entity whose id is this, in any case,
            is the user: it keeps its type and takes no other rule.
        user_type: The type a rail gives the user anchor when it has to add it.
        parent_version: The version this one follows.
        retired_types: Types of earlier versions and their replacements.
        supertypes: (supertype, anchor child) pairs: an edge that accepts the anchor
            child accepts the supertype.
        escape_edge: The edge the model uses when no typed edge fits.
        escape_rate_trigger: The share of escape edges that opens a growth review.
        prompt_token_budget: The most tokens the assembled prompt may take.
        released_against: The extraction version the pack was released against.
        migration_script_path: The migration that moves a graph to this version.
        deprecated: True when the registry must not pick the pack by default.
    """

    version: str
    fallback_type: str
    user_id: str
    user_type: str
    parent_version: str | None = None
    retired_types: tuple[RetiredType, ...] = ()
    supertypes: tuple[tuple[str, str], ...] = ()
    escape_edge: str | None = None
    escape_rate_trigger: float | None = None
    prompt_token_budget: int | None = None
    released_against: str | None = None
    migration_script_path: str | None = None
    deprecated: bool = False


PairTable = Mapping[str, tuple[frozenset[str] | None, frozenset[str] | None]]


@dataclass(frozen=True, eq=False)
class Pack:
    """A loaded pack. Build one with `build_pack` or `load_pack`, never directly.

    Attributes:
        manifest: The pack's manifest.
        node_types: Every node type.
        edge_types: Every edge type, after the manifest's retired types and supertypes
            were applied to its allowed sets.
        extractable_node_types: The types the model may produce.
        extractable_edge_types: The edges the model may produce.
        rules: The convention rules, in the pack's order.
        plugins: The registered plug-ins.
        labelling: The pack's labelling text. It feeds neither hash.
        pair_table: Each extractable edge's allowed source and target types; None
            stands for any type.
        pack_prompt_hash: sha256 over what feeds the model.
        pack_rails_hash: sha256 over what runs after it.
    """

    manifest: Manifest
    node_types: tuple[NodeTypeDefinition, ...]
    edge_types: tuple[EdgeTypeDefinition, ...]
    extractable_node_types: tuple[str, ...]
    extractable_edge_types: tuple[str, ...]
    rules: tuple[ConventionRule, ...]
    plugins: tuple[Plugin, ...]
    labelling: str
    pair_table: PairTable
    pack_prompt_hash: str
    pack_rails_hash: str

    @property
    def version(self) -> str:
        """The pack's ontology version."""
        return self.manifest.version

    @cached_property
    def type_names(self) -> frozenset[str]:
        """The name of every node type."""
        return frozenset(node.type_name for node in self.node_types)

    @cached_property
    def edges_by_name(self) -> Mapping[str, EdgeTypeDefinition]:
        """Each edge definition under its name."""
        return {edge.type_name: edge for edge in self.edge_types}

    @cached_property
    def plugins_by_name(self) -> Mapping[str, Plugin]:
        """Each plug-in under its name."""
        return {plugin.name: plugin for plugin in self.plugins}

    @cached_property
    def retired(self) -> Mapping[str, RetiredType]:
        """Each retired type under its name."""
        return {retired.name: retired for retired in self.manifest.retired_types}

    def rules_in(self, phase: Phase) -> tuple[ConventionRule, ...]:
        """The rules of one phase, in the pack's order."""
        return tuple(rule for rule in self.rules if rule.phase == phase)

    def rules_of(self, kind: str, phase: Phase | None = None) -> tuple[ConventionRule, ...]:
        """The rules of one kind, in the pack's order, optionally of one phase."""
        return tuple(
            rule
            for rule in self.rules
            if rule.kind == kind and (phase is None or rule.phase == phase)
        )


# ---------------------------------------------------------------------------
# The loader
# ---------------------------------------------------------------------------


def _apply_manifest(types: tuple[str, ...], manifest: Manifest) -> tuple[str, ...]:
    """One allowed-type tuple with retired types replaced and supertypes added.

    Order is kept and a repeated name dropped. A supertype joins wherever its anchor
    child is allowed. Applying it twice changes nothing.
    """
    replacements = {retired.name: retired.replacement for retired in manifest.retired_types}
    mapped: list[str] = []
    for name in types:
        current = replacements.get(name, name)
        if current not in mapped:
            mapped.append(current)
    for supertype, anchor in manifest.supertypes:
        if anchor in mapped and supertype not in mapped:
            mapped.append(supertype)
    return tuple(mapped)


def _duplicates(names: tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    repeated: list[str] = []
    for name in names:
        if name in seen and name not in repeated:
            repeated.append(name)
        seen.add(name)
    return repeated


def _build_pair_table(
    edge_types: tuple[EdgeTypeDefinition, ...],
    extractable_edge_types: tuple[str, ...],
    type_names: frozenset[str],
) -> PairTable:
    """Each extractable edge's allowed source and target sets, None for any type.

    Raises:
        PackLoadError: An extractable edge has no definition, or an edge allows a type
            the pack does not define, or allows none at one end.
    """
    by_name = {edge.type_name: edge for edge in edge_types}
    table: dict[str, tuple[frozenset[str] | None, frozenset[str] | None]] = {}
    for name in extractable_edge_types:
        edge = by_name.get(name)
        if edge is None:
            raise PackLoadError(f"pair table does not build: extractable edge {name} is undefined")
        ends: list[frozenset[str] | None] = []
        for end, allowed in (
            ("source", edge.allowed_source_types),
            ("target", edge.allowed_target_types),
        ):
            unknown = sorted(set(allowed) - type_names)
            if unknown:
                raise PackLoadError(
                    f"pair table does not build: edge {name} allows unknown {end} "
                    f"type(s) {', '.join(unknown)}"
                )
            if not allowed:
                raise PackLoadError(f"pair table does not build: edge {name} allows no {end} type")
            ends.append(None if set(allowed) >= type_names else frozenset(allowed))
        table[name] = (ends[0], ends[1])
    return table


def _named(params: object, what: str) -> list[str]:
    """Every value of the fields tagged `what`, through nested parameter classes."""
    found: list[str] = []
    for spec in dataclasses.fields(params):  # type: ignore[arg-type]
        value = getattr(params, spec.name)
        if spec.metadata.get("names") == what:
            if isinstance(value, str):
                found.append(value)
            elif value is not None:
                found.extend(value)
        elif dataclasses.is_dataclass(value):
            found.extend(_named(value, what))
        elif isinstance(value, tuple):
            for item in value:
                if dataclasses.is_dataclass(item):
                    found.extend(_named(item, what))
    return found


def _rule_plugins(rule: ConventionRule) -> list[str]:
    """The plug-ins a rule names, directly or as a `fan_out` endpoint."""
    names = _named(rule.params, "plugin")
    for endpoint in _named(rule.params, "endpoint"):
        if endpoint.startswith(ENDPOINT_PLUGIN_PREFIX):
            names.append(endpoint[len(ENDPOINT_PLUGIN_PREFIX) :])
    return names


def _check_shape(rule: ConventionRule) -> None:
    """Refuse parameters the rule's interpreter cannot run."""
    params = rule.params
    if isinstance(params, DerivedEdge):
        if rule.phase == ID and params.plugin is None:
            raise PackLoadError(f"rule {rule.id}: an id-phase derived_edge names an id plug-in")
        if rule.phase == RELATION and (params.property is None or params.plugin is not None):
            raise PackLoadError(
                f"rule {rule.id}: a relation-phase derived_edge names a property and no plug-in"
            )
    if isinstance(params, FanOut):
        fixed = {ENDPOINT_MEMBER, ENDPOINT_USER}
        fixed.add(ENDPOINT_ANCHOR if isinstance(params.trigger, EdgeTrigger) else ENDPOINT_FIXED)
        for gain in params.gains:
            ends = (gain.source, gain.target)
            for end in ends:
                if end not in fixed and not end.startswith(ENDPOINT_PLUGIN_PREFIX):
                    raise PackLoadError(f"rule {rule.id}: unknown endpoint {end}")
            if ENDPOINT_MEMBER not in ends:
                raise PackLoadError(f"rule {rule.id}: a gained edge has the member at one end")
    if isinstance(params, ItemFilter) and params.state not in ITEM_STATES:
        raise PackLoadError(f"rule {rule.id}: unknown item state {params.state}")
    if rule.phase == RELATION and not rule.prompt_rule:
        raise PackLoadError(
            f"rule {rule.id}: a relation rule names the prompt rule it derives from"
        )


def _check_rules(
    rules: tuple[ConventionRule, ...],
    plugins: tuple[Plugin, ...],
    type_names: frozenset[str],
    edge_names: frozenset[str],
) -> None:
    """Refuse an unknown kind, phase, type, edge or plug-in, and an unpinned plug-in.

    Raises:
        PackLoadError: Naming the rule and what it names that the pack lacks.
    """
    repeated = _duplicates(tuple(rule.id for rule in rules))
    if repeated:
        raise PackLoadError(f"rule id(s) used twice: {', '.join(repeated)}")
    repeated = _duplicates(tuple(plugin.name for plugin in plugins))
    if repeated:
        raise PackLoadError(f"plug-in(s) registered twice: {', '.join(repeated)}")
    registered = {plugin.name: plugin for plugin in plugins}
    pinned: set[str] = set()
    for rule in rules:
        if rule.kind not in KINDS:
            raise PackLoadError(f"rule {rule.id}: unknown kind {rule.kind}")
        params_class, phases = KINDS[rule.kind]
        if not isinstance(rule.params, params_class):
            raise PackLoadError(
                f"rule {rule.id}: kind {rule.kind} takes {params_class.__name__} parameters"
            )
        if rule.phase not in phases:
            raise PackLoadError(
                f"rule {rule.id}: kind {rule.kind} does not run in the {rule.phase} phase"
            )
        unknown = sorted(set(_named(rule.params, "type")) - type_names)
        if unknown:
            raise PackLoadError(f"rule {rule.id}: unknown type(s) {', '.join(unknown)}")
        unknown = sorted(set(_named(rule.params, "edge")) - edge_names)
        if unknown:
            raise PackLoadError(f"rule {rule.id}: unknown edge(s) {', '.join(unknown)}")
        _check_shape(rule)
        for name in _rule_plugins(rule):
            plugin = registered.get(name)
            if plugin is None:
                raise PackLoadError(f"rule {rule.id}: plug-in {name} is not registered")
            if plugin.phase != rule.phase:
                raise PackLoadError(
                    f"rule {rule.id}: plug-in {name} is registered for the {plugin.phase} "
                    f"phase, not {rule.phase}"
                )
            if rule.pinned_case is None:
                raise PackLoadError(f"rule {rule.id}: plug-in {name} has no pinned case")
            pinned.add(name)
    unpinned = sorted(set(registered) - pinned)
    if unpinned:
        raise PackLoadError(f"plug-in(s) with no pinned case: {', '.join(unpinned)}")


def _check_manifest(
    manifest: Manifest, type_names: frozenset[str], edge_names: frozenset[str]
) -> None:
    """Refuse a manifest that names a type or an edge the pack does not define."""
    named = [manifest.fallback_type, manifest.user_type]
    named += [retired.replacement for retired in manifest.retired_types]
    named += [name for pair in manifest.supertypes for name in pair]
    unknown = sorted(set(named) - type_names)
    if unknown:
        raise PackLoadError(f"manifest: unknown type(s) {', '.join(unknown)}")
    if manifest.escape_edge is not None and manifest.escape_edge not in edge_names:
        raise PackLoadError(f"manifest: unknown edge {manifest.escape_edge}")
    if not manifest.user_id:
        raise PackLoadError("manifest: the user anchor has no id")


def _canonical(value: object) -> object:
    """`value` as JSON-ready data, the one serialization both hashes are taken over."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        body = {
            spec.name: _canonical(getattr(value, spec.name)) for spec in dataclasses.fields(value)
        }
        return {"__class__": type(value).__name__, **body}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_canonical(item) for item in value]
    if isinstance(value, frozenset | set):
        return sorted(str(item) for item in value)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return repr(value)


def _sha256(value: object) -> str:
    text = json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _rule_identity(rule: ConventionRule) -> dict[str, object]:
    """What a rule does: everything but its pinned case."""
    return {
        "id": rule.id,
        "kind": rule.kind,
        "phase": rule.phase,
        "params": rule.params,
        "prompt_rule": rule.prompt_rule,
    }


def _plugin_identity(plugin: Plugin) -> dict[str, object]:
    """A plug-in's name, phase and the source of the module that defines it.

    The whole module is hashed, so a helper the plug-in calls is covered too.

    Raises:
        PackLoadError: The plug-in's source cannot be read.
    """
    module = inspect.getmodule(plugin.fn)
    try:
        source = inspect.getsource(module) if module is not None else None
    except (OSError, TypeError) as exc:
        raise PackLoadError(f"plug-in {plugin.name}: its source cannot be read") from exc
    if source is None:
        raise PackLoadError(f"plug-in {plugin.name}: its source cannot be read")
    return {
        "name": plugin.name,
        "phase": plugin.phase,
        "function": getattr(plugin.fn, "__qualname__", repr(plugin.fn)),
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
    }


def _by_phase(rules: tuple[ConventionRule, ...], phases: tuple[Phase, ...]) -> dict[str, object]:
    return {
        phase: [_rule_identity(rule) for rule in rules if rule.phase == phase] for phase in phases
    }


def build_pack(
    *,
    manifest: Manifest,
    node_types: tuple[NodeTypeDefinition, ...],
    edge_types: tuple[EdgeTypeDefinition, ...],
    extractable_node_types: tuple[str, ...],
    extractable_edge_types: tuple[str, ...],
    rules: tuple[ConventionRule, ...] = (),
    plugins: tuple[Plugin, ...] = (),
    labelling: str = "",
) -> Pack:
    """Validate a pack's parts and return the loaded pack with its two hashes.

    The manifest's retired types and supertypes are applied to every edge's allowed
    sets first, so the pair table and the prompt hash see the edges the engine uses.

    Args:
        manifest: The pack's manifest.
        node_types: Every node type.
        edge_types: Every edge type, as the pack writes them.
        extractable_node_types: The types the model may produce.
        extractable_edge_types: The edges the model may produce.
        rules: The convention rules, in the order their phases run them.
        plugins: The registered plug-ins.
        labelling: The pack's labelling text.

    Raises:
        PackLoadError: A name is unknown or repeated, a plug-in is not registered or
            has no pinned case, or the pair table does not build.
    """
    for what, names in (
        ("node type", tuple(node.type_name for node in node_types)),
        ("edge type", tuple(edge.type_name for edge in edge_types)),
    ):
        repeated = _duplicates(names)
        if repeated:
            raise PackLoadError(f"{what}(s) defined twice: {', '.join(repeated)}")
    type_names = frozenset(node.type_name for node in node_types)
    edge_names = frozenset(edge.type_name for edge in edge_types)
    unknown = sorted(set(extractable_node_types) - type_names)
    if unknown:
        raise PackLoadError(f"extractable type(s) undefined: {', '.join(unknown)}")
    _check_manifest(manifest, type_names, edge_names)

    built_edges = tuple(
        dataclasses.replace(
            edge,
            allowed_source_types=_apply_manifest(edge.allowed_source_types, manifest),
            allowed_target_types=_apply_manifest(edge.allowed_target_types, manifest),
        )
        for edge in edge_types
    )
    pair_table = _build_pair_table(built_edges, tuple(extractable_edge_types), type_names)
    _check_rules(tuple(rules), tuple(plugins), type_names, edge_names)

    prompt_hash = _sha256(
        {
            "node_types": node_types,
            "edge_types": built_edges,
            "extractable_node_types": extractable_node_types,
            "extractable_edge_types": extractable_edge_types,
            "pairs": {
                name: [None if end is None else sorted(end) for end in ends]
                for name, ends in pair_table.items()
            },
            "rules": _by_phase(tuple(rules), (PRE_LLM,)),
            "manifest": {
                "supertypes": manifest.supertypes,
                "escape_edge": manifest.escape_edge,
                "prompt_token_budget": manifest.prompt_token_budget,
            },
        }
    )
    rails_hash = _sha256(
        {
            "rules": _by_phase(tuple(rules), POST_LLM_PHASES),
            "plugins": [_plugin_identity(plugin) for plugin in plugins],
            "manifest": {
                "fallback_type": manifest.fallback_type,
                "retired_types": manifest.retired_types,
                "user_id": manifest.user_id,
                "user_type": manifest.user_type,
            },
        }
    )
    return Pack(
        manifest=manifest,
        node_types=tuple(node_types),
        edge_types=built_edges,
        extractable_node_types=tuple(extractable_node_types),
        extractable_edge_types=tuple(extractable_edge_types),
        rules=tuple(rules),
        plugins=tuple(plugins),
        labelling=labelling,
        pair_table=pair_table,
        pack_prompt_hash=prompt_hash,
        pack_rails_hash=rails_hash,
    )


def load_pack(package: str) -> Pack:
    """Load the pack package `package` and validate it.

    A pack package holds `types.py` (`NODE_TYPES`, `EDGE_TYPES`, `EXTRACTABLE_NODE_TYPES`,
    `EXTRACTABLE_EDGE_TYPES`), `manifest.py` (`MANIFEST`), `conventions.py` (`RULES`),
    `plugins.py` (`PLUGINS`) and `labelling.md`.

    Args:
        package: The package's dotted name.

    Raises:
        PackLoadError: A module or one of its names is missing, `labelling.md` cannot be
            read, or `build_pack` refuses the pack.
    """
    parts: dict[str, object] = {}
    directory: Path | None = None
    for module_name, names in (
        ("types", ("NODE_TYPES", "EDGE_TYPES", "EXTRACTABLE_NODE_TYPES", "EXTRACTABLE_EDGE_TYPES")),
        ("manifest", ("MANIFEST",)),
        ("conventions", ("RULES",)),
        ("plugins", ("PLUGINS",)),
    ):
        try:
            module = importlib.import_module(f"{package}.{module_name}")
        except ImportError as exc:
            raise PackLoadError(f"pack {package}: {module_name}.py does not import") from exc
        for name in names:
            if not hasattr(module, name):
                raise PackLoadError(f"pack {package}: {module_name}.py defines no {name}")
            parts[name] = getattr(module, name)
        directory = directory or Path(str(module.__file__)).parent
    try:
        labelling = ((directory or Path()) / "labelling.md").read_text(encoding="utf-8")
    except OSError as exc:
        raise PackLoadError(f"pack {package}: labelling.md cannot be read") from exc
    return build_pack(
        manifest=parts["MANIFEST"],  # type: ignore[arg-type]
        node_types=tuple(parts["NODE_TYPES"]),  # type: ignore[arg-type]
        edge_types=tuple(parts["EDGE_TYPES"]),  # type: ignore[arg-type]
        extractable_node_types=tuple(parts["EXTRACTABLE_NODE_TYPES"]),  # type: ignore[arg-type]
        extractable_edge_types=tuple(parts["EXTRACTABLE_EDGE_TYPES"]),  # type: ignore[arg-type]
        rules=tuple(parts["RULES"]),  # type: ignore[arg-type]
        plugins=tuple(parts["PLUGINS"]),  # type: ignore[arg-type]
        labelling=labelling,
    )
