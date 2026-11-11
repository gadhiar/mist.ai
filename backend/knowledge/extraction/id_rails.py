"""The id rails: how a stored entity id is built from the extraction payload's `id` field.

ADR-054 Decision 2: the graph stores the model's `id` field, not an id built from the
display name, and every convention the gold labels by id has a carrier, a prompt rule or a
deterministic rail. The rails are pure and synchronous, so that the normalizer
(`EntityNormalizer.normalize`) and the eval scorer (`scripts/eval_harness/
score_extraction_run.py`) build every id with the same functions, and a test can assert
that the id the graph stores is the id the scorer scores.

The module has two halves. The engine half is the functions those callers use: each runs
the rail runner (`backend.knowledge.extraction.rails`) with the registry's active pack
and names no type or edge. The ontology half, below its marker, holds what today's
ontology gives the adapter pack (`backend.knowledge.ontologies.registry`): the alias
maps, the reserved names, the exempt and retired types and the id plug-ins. A later
ontology version moves that half into a pack's own files and leaves this one.

This module imports no graph, embedding or I/O dependency, so the scorer can import it
without the normalizer's. The runner and the registry are imported inside the functions
that call them, because both import this module.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from backend.knowledge.extraction.canonical_id import (
    canonical_metric_id,
    canonical_metric_id_from_id,
)
from backend.knowledge.ontologies.hierarchy import RETIRED_TYPE_MAP
from backend.knowledge.ontologies.v1_0_0 import ALL_NODE_TYPE_NAMES

# Bounded curated (id, type) overrides for recurring high-value entities, keyed by the id
# the pack's alias vocabularies give. Seeded later from curated vocab (the versioned seed
# source), NEVER from F2 probes. Authoritative: a registry hit overrides both id and type
# and ends the steps. Empty for now. The runner reads this dict at each call, so an entry
# added in place takes effect at once.
CANONICAL_REGISTRY: dict[str, tuple[str, str]] = {}

# LABELLING.md N1: leading articles and possessive pronouns are dropped from ids.
N1_WORDS: frozenset[str] = frozenset(
    {"a", "an", "the", "my", "our", "your", "his", "her", "their", "its"}
)


def n1_rail(entity_id: str) -> str:
    """Drop the leading `N1_WORDS` tokens from an id; an id they would empty keeps its last."""
    tokens = entity_id.split("-")
    start = 0
    while start < len(tokens) - 1 and tokens[start] in N1_WORDS:
        start += 1
    return "-".join(tokens[start:])


def reserved_identity(entity: Mapping) -> tuple[str, str] | None:
    """The (id, type) a reserved name of the active pack maps the entity to, or None.

    The lowered display name is looked up first, then the slug of the `id` field, since
    a reserved name may be written id-shaped. The user entity is never reserved.
    """
    from backend.knowledge.extraction import rails
    from backend.knowledge.ontologies import registry

    return rails.reserved_identity(entity, registry.active_pack())


def normalize_entity(entity: dict) -> None:
    """Build one entity's stored id from its `id` field, in place (ADR-054 Decision 2).

    The steps are the active pack's, run by `rails.normalize_entity`: the user anchor
    keeps its id; a reserved name sets id and type and ends the steps; a retired type is
    replaced; the slug of the `id` field goes through the pack's alias vocabularies and
    typed-id rules; a hit in `CANONICAL_REGISTRY` sets id and type and ends the steps;
    the N1 rail drops the leading words of every type the pack does not exempt; an
    unknown type takes the pack's fallback type.

    Args:
        entity: A payload entity dict with `id`, `type`, and optionally `name` and
            `properties`.
    """
    from backend.knowledge.extraction import rails
    from backend.knowledge.ontologies import registry

    rails.normalize_entity(entity, registry.active_pack())


def stored_entity_id(
    raw_id: str,
    entity_type: str,
    properties: Mapping | None = None,
    name: str | None = None,
) -> str:
    """The id the graph stores for one entity, from `normalize_entity` on a copy.

    Idempotent on a stored id, so a caller holding a graph id may pass it through again.

    Args:
        raw_id: The entity's `id` field.
        entity_type: Its type.
        properties: Its properties, read by a typed-id rule; None for none.
        name: Its display name, read by the reserved-name, alias and typed-id steps; None
            lets the id stand in.

    Returns:
        The stored id.
    """
    entity: dict = {"id": raw_id, "type": entity_type, "properties": dict(properties or {})}
    if name is not None:
        entity["name"] = name
    normalize_entity(entity)
    return str(entity["id"])


def event_date_rail(entities: list[dict], relationships: list[dict]) -> dict[str, str]:
    """Rename each entity the active pack's id-phase `derived_edge` rules date.

    ADR-054 Decision 4, as today's pack states it: an event that the payload dates to
    exactly one ISO day loses a trailing `-YYYY-MM-DD` from its id and gains `-<that
    date>`, and its display name gains ` <that date>` unless it already ends with it;
    with no such day, or two or more, it is unchanged. The rail reads the date entity's
    id, never an edge's `start_date`, so it gives the same id whether or not the
    temporal resolver ran, and the scorer, which runs no resolver, mirrors it. Entities
    are renamed in place; edges are not touched.

    Args:
        entities: The payload's entity dicts, after `normalize_entity`.
        relationships: The payload's relationship dicts, endpoints already remapped.

    Returns:
        Old id to new id, for every entity whose id changed.
    """
    from backend.knowledge.extraction import rails
    from backend.knowledge.ontologies import registry

    return rails.derived_ids(entities, relationships, registry.active_pack())


def normalize_payload(
    entities: list[dict], relationships: list[dict]
) -> tuple[list[dict], list[dict]]:
    """Run the rails over one payload: `rails.run_rails` with the active pack.

    The payload stands for a row with no scope, batch record or plan. Each entity's
    stored id is built; relationship endpoints that name an entity's `id` field are
    remapped to its stored id; the id-phase `derived_edge` rules run, and the endpoints
    naming a renamed entity are remapped again; entities are deduplicated first-wins by
    id, and relationships by (source, target, type). The entity and relationship dicts
    are changed in place.

    Args:
        entities: The payload's entity dicts.
        relationships: The payload's relationship dicts.

    Returns:
        The deduplicated entities and relationships.
    """
    from backend.knowledge.extraction import rails
    from backend.knowledge.ontologies import registry

    return rails.run_rails(rails.rail_row(entities, relationships), registry.active_pack())


# Everything above this marker is engine: it names no type and no edge, and a test holds
# it to that. Everything below is today's ontology, read by the registry's adapter pack.
# === ontology half ===

# Static alias map: short name / alternate spelling -> canonical ID.
# IMPORTANT: Keys must be unambiguous. If a short form could refer to
# multiple technologies (e.g. "tf" -> terraform OR tensorflow), do NOT
# include it here. Let the LLM resolve ambiguity via the full name.
STATIC_ALIASES: dict[str, str] = {
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "rb": "ruby",
    "node.js": "nodejs",
    "node": "nodejs",
    "react.js": "react",
    "reactjs": "react",
    "react-js": "react",
    "vue.js": "vue",
    "vuejs": "vue",
    "vue-js": "vue",
    "angular.js": "angular",
    "angularjs": "angular",
    "next.js": "nextjs",
    "nuxt.js": "nuxtjs",
    "mongo": "mongodb",
    "postgres": "postgresql",
    "pg": "postgresql",
    "k8s": "kubernetes",
    "kube": "kubernetes",
    "vscode": "visual-studio-code",
    "vs-code": "visual-studio-code",
    "gcp": "google-cloud-platform",
    "aws": "amazon-web-services",
    "gh": "github",
    "cpp": "c-plus-plus",
    "csharp": "c-sharp",
    "golang": "go",
    "objc": "objective-c",
    "sklearn": "scikit-learn",
    "torch": "pytorch",
}

# Pre-canonicalization aliases: checked against the raw lowered display name, because
# the special characters of "c++" and "c#" cannot survive a slug.
PRE_CANON_ALIASES: dict[str, str] = {
    "c++": "c-plus-plus",
    "c#": "c-sharp",
}

# A version suffix on a display name: "Python 3.11", "Node 18.x", "React v18". Requires
# whitespace before the number or an explicit "v" prefix, so the trailing digits of
# "web3", "oauth2" or "base64" are not a version.
VERSION_PATTERN: re.Pattern = re.compile(
    r"(?:\s+v?|\bv)(\d+\.?\d*\.?\d*\.?\d*)([-.]?\w+)*$", re.IGNORECASE
)

# Bug G guard: reserved names for the MIST system itself always resolve to the canonical
# mist-identity node (seeded in mist-memory/seed/mist.md). Extraction that introduces new
# "mist" or "the-ai" entities pollutes the graph with duplicates of the system's own
# identity node.
#
# Cluster 1 extension: each entry maps to a (canonical_id, canonical_type) tuple. The type
# override is required because Cluster 1 validator constraints check MistIdentity as the
# source of IMPLEMENTED_WITH / MIST_HAS_CAPABILITY / MIST_HAS_TRAIT / MIST_HAS_PREFERENCE
# edges; an LLM-guessed type of "Organization" on the mist-identity node breaks those
# constraints. Forcing type=MistIdentity on reserved-name matches keeps node schema aligned
# with edge-contract validation.
RESERVED_NAMES: dict[str, tuple[str, str]] = {
    "mist": ("mist-identity", "MistIdentity"),
    "mist.ai": ("mist-identity", "MistIdentity"),
    "mist ai": ("mist-identity", "MistIdentity"),
    "mist-ai": ("mist-identity", "MistIdentity"),
    "the ai": ("mist-identity", "MistIdentity"),
    "the-ai": ("mist-identity", "MistIdentity"),
    "the assistant": ("mist-identity", "MistIdentity"),
    "the-assistant": ("mist-identity", "MistIdentity"),
}

ISO_DATE_RE: re.Pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")

USER_ID = "user"
FALLBACK_TYPE = "Abstraction"

# Types whose ids follow their own convention, so the N1 rail leaves them alone. Kept in
# the ontology's order, since the pack's hash reads them in order.
_N1_EXEMPT_TYPES = frozenset({"User", "MistIdentity", "Date", "Metric"})
ID_EXEMPT_TYPES: tuple[str, ...] = tuple(
    name for name in ALL_NODE_TYPE_NAMES if name in _N1_EXEMPT_TYPES
)

# What an entity of a retired type gains, unless it has it, when its type is replaced: a
# Milestone becomes an Event whose `event_type` says what it was.
_RETIRED_DEFAULTS: dict[str, tuple[tuple[str, str], ...]] = {
    "Milestone": (("event_type", "milestone"),),
}
RETIRED_PROPERTY_DEFAULTS: dict[str, tuple[tuple[str, str], ...]] = {
    name: _RETIRED_DEFAULTS.get(name, ()) for name in RETIRED_TYPE_MAP
}

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_TRAILING_ISO_DATE_RE = re.compile(r"-\d{4}-\d{2}-\d{2}$")


def version_rail(entity_id: str, name: str | None, entity_type: str) -> str:
    """Remove a Technology's version, read from its name, from the end of its id.

    Rule 3 collapses versions ("Python 3.11" is `python`). The rail fires only for a
    `Technology` whose name `VERSION_PATTERN` matches: the matched suffix is split on
    non-alphanumerics (` 3.11` gives `3`, `11`; ` v18` gives `v18`), and when the id ends
    with those tokens hyphen-joined (`-3-11`) or concatenated (`-311`), they are removed.
    Every other id keeps its digits, so `web3`, `covid-19` and an Event's date stay.
    """
    if entity_type != "Technology" or not name:
        return entity_id
    match = VERSION_PATTERN.search(name)
    if match is None:
        return entity_id
    tokens = [t for t in _NON_ALNUM_RE.split(match.group(0).lower()) if t]
    if not tokens:
        return entity_id
    for suffix in ("-".join(tokens), "".join(tokens)):
        tail = f"-{suffix}"
        if entity_id.endswith(tail) and len(entity_id) > len(tail):
            return entity_id[: -len(tail)]
    return entity_id


def technology_version_id(entity: Mapping, entity_id: str, target: Mapping | None) -> str:
    """The id plug-in of the Technology `typed_id` rule: `version_rail` on the entity."""
    name = str(entity.get("name") or entity.get("id") or "")
    return version_rail(entity_id, name, "Technology")


def metric_id(entity: Mapping, entity_id: str, target: Mapping | None) -> str:
    """The id plug-in of the Metric `typed_id` rule.

    The id is built with `canonical_metric_id` when the properties carry a value and a
    unit or a quantity, and otherwise from the raw `id` field with
    `canonical_metric_id_from_id`, so a dotted decimal in the id survives.
    """
    props = entity.get("properties")
    props = props if isinstance(props, Mapping) else {}
    value, unit, quantity = props.get("value"), props.get("unit"), props.get("quantity")
    if value is not None and (unit or quantity):
        return canonical_metric_id(
            value, str(unit) if unit else None, str(quantity) if quantity else None
        )
    return canonical_metric_id_from_id(str(entity.get("id") or entity.get("name") or ""))


def event_date_id(entity: Mapping, entity_id: str, target: Mapping | None) -> str | None:
    """The id plug-in of the Event-date `derived_edge` rule (ADR-054 Decision 4).

    A target whose id is not an ISO day does not date the Event. Otherwise the id loses
    a trailing `-YYYY-MM-DD` and gains `-<the target's id>`.
    """
    date = str((target or {}).get("id", ""))
    if not ISO_DATE_RE.match(date):
        return None
    return f"{_TRAILING_ISO_DATE_RE.sub('', entity_id)}-{date}"
