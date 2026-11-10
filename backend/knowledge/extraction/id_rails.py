"""The id rails: how a stored entity id is built from the extraction payload's `id` field.

ADR-054 Decision 2: the graph stores the model's `id` field, not an id built from the
display name, and every convention the gold labels by id has a carrier, a prompt rule or a
deterministic rail. The rails live here, pure and synchronous, so that the normalizer
(`EntityNormalizer.normalize`) and the eval scorer (`scripts/eval_harness/
score_extraction_run.py`) build every id with the same functions, and a test can assert
that the id the graph stores is the id the scorer scores.

This module imports no graph, embedding or I/O dependency, so the scorer can import it
without the normalizer's.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from backend.knowledge.extraction.canonical_id import (
    canonical_id,
    canonical_metric_id,
    canonical_metric_id_from_id,
)
from backend.knowledge.ontologies.hierarchy import RETIRED_TYPE_MAP
from backend.knowledge.ontologies.v1_0_0 import ALL_NODE_TYPE_NAMES

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

# Bounded curated (id, type) overrides for recurring high-value entities, keyed by the id
# after the slug, the version rail and the static aliases. Seeded later from curated vocab
# (the versioned seed source), NEVER from F2 probes. Authoritative: a registry hit
# overrides both id and type and ends the steps. Empty for now.
CANONICAL_REGISTRY: dict[str, tuple[str, str]] = {}

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

# LABELLING.md N1: leading articles and possessive pronouns are dropped from ids.
N1_WORDS: frozenset[str] = frozenset(
    {"a", "an", "the", "my", "our", "your", "his", "her", "their", "its"}
)

ISO_DATE_RE: re.Pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_USER_ID = "user"
# Types whose ids follow their own convention, so the N1 rail leaves them alone.
_N1_EXEMPT_TYPES = frozenset({"User", "MistIdentity", "Date", "Metric"})
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _raw_id(entity: Mapping) -> str:
    """The entity's `id` field; the name stands in when it is empty or missing."""
    return str(entity.get("id") or entity.get("name") or "")


def _display_name(entity: Mapping) -> str:
    """The entity's display name; the `id` field stands in when it is missing."""
    return str(entity.get("name") or _raw_id(entity))


def reserved_identity(entity: Mapping) -> tuple[str, str] | None:
    """The (id, type) a reserved name maps the entity to, or None.

    The lowered display name is looked up first, then the slug of the `id` field, since
    four of the keys are id-shaped. The user entity is never reserved.
    """
    raw_id = _raw_id(entity)
    if raw_id.lower() == _USER_ID:
        return None
    reserved = RESERVED_NAMES.get(_display_name(entity).lower().strip())
    if reserved is None:
        reserved = RESERVED_NAMES.get(canonical_id(raw_id))
    return reserved


def n1_rail(entity_id: str) -> str:
    """Drop the leading `N1_WORDS` tokens from an id; an id they would empty keeps its last."""
    tokens = entity_id.split("-")
    start = 0
    while start < len(tokens) - 1 and tokens[start] in N1_WORDS:
        start += 1
    return "-".join(tokens[start:])


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


def _resolver_passes(entity: dict, raw_id: str) -> None:
    """Passes 3 to 5: retired types, the Metric id, the Abstraction fallback.

    Pass 3 coerces a retired type (Topic to Concept, Milestone to Event); a Milestone also
    gets `properties.event_type` of `milestone` unless it has one. Pass 4 builds a
    Metric's id with `canonical_metric_id` when its properties carry a value and a unit or
    a quantity, and otherwise from the raw `id` field with `canonical_metric_id_from_id`,
    so a dotted decimal in the id survives. Pass 5 types an unknown type `Abstraction`.
    """
    etype = entity.get("type", "")

    if etype in RETIRED_TYPE_MAP:
        coerced = RETIRED_TYPE_MAP[etype]
        if etype == "Milestone":
            props = entity.get("properties")
            if not isinstance(props, dict):
                props = {}
                entity["properties"] = props
            props.setdefault("event_type", "milestone")
        entity["type"] = coerced
        etype = coerced

    if etype == "Metric":
        props = entity.get("properties")
        props = props if isinstance(props, dict) else {}
        value, unit, quantity = props.get("value"), props.get("unit"), props.get("quantity")
        if value is not None and (unit or quantity):
            entity["id"] = canonical_metric_id(
                value, str(unit) if unit else None, str(quantity) if quantity else None
            )
        else:
            entity["id"] = canonical_metric_id_from_id(raw_id)

    if entity.get("type", "") not in ALL_NODE_TYPE_NAMES:
        entity["type"] = "Abstraction"


def normalize_entity(entity: dict) -> None:
    """Build one entity's stored id from its `id` field, in place (ADR-054 Decision 2).

    The steps, in order: the user stays `user`; a reserved name sets id and type to
    `mist-identity` and `MistIdentity` and ends the steps; the pre-canonicalization
    aliases on the name, otherwise `canonical_id` of the `id` field; the version rail;
    the static aliases; the registry, whose hit sets id and type and ends the steps;
    resolver passes 3 to 5; the N1 rail, for every type but User, MistIdentity, Date and
    Metric. A Milestone also gets `properties.event_type`.

    Args:
        entity: A payload entity dict with `id`, `type`, and optionally `name` and
            `properties`.
    """
    raw_id = _raw_id(entity)
    if raw_id.lower() == _USER_ID:
        entity["id"] = _USER_ID
        return

    reserved = reserved_identity(entity)
    if reserved is not None:
        entity["id"], entity["type"] = reserved
        return

    name = _display_name(entity)
    pre_canon = PRE_CANON_ALIASES.get(name.lower().strip())
    entity_id = pre_canon if pre_canon is not None else canonical_id(raw_id)
    entity_id = version_rail(entity_id, name, str(entity.get("type", "")))
    entity_id = STATIC_ALIASES.get(entity_id, entity_id)

    registry = CANONICAL_REGISTRY.get(entity_id)
    if registry is not None:
        entity["id"], entity["type"] = registry
        return

    entity["id"] = entity_id
    _resolver_passes(entity, raw_id)
    if entity.get("type", "") not in _N1_EXEMPT_TYPES:
        entity["id"] = n1_rail(entity["id"])


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
        properties: Its properties, read by the Metric pass; None for none.
        name: Its display name, read by the reserved-name, alias and version steps; None
            lets the id stand in.

    Returns:
        The stored id.
    """
    entity: dict = {"id": raw_id, "type": entity_type, "properties": dict(properties or {})}
    if name is not None:
        entity["name"] = name
    normalize_entity(entity)
    return str(entity["id"])


def _remap_endpoints(relationships: list[dict], id_map: Mapping) -> None:
    for rel in relationships:
        source = rel.get("source", "")
        target = rel.get("target", "")
        if source in id_map:
            rel["source"] = id_map[source]
        if target in id_map:
            rel["target"] = id_map[target]


def normalize_payload(
    entities: list[dict], relationships: list[dict]
) -> tuple[list[dict], list[dict]]:
    """Run the id rails over one payload, in ADR-054's order.

    Each entity goes through `normalize_entity`; relationship endpoints that name an
    entity's `id` field are remapped to its stored id; entities are deduplicated
    first-wins by id, and relationships by (source, target, type). The entity and
    relationship dicts are changed in place.

    Args:
        entities: The payload's entity dicts.
        relationships: The payload's relationship dicts.

    Returns:
        The deduplicated entities and relationships.
    """
    id_map: dict = {}
    for entity in entities:
        old_id = entity.get("id", "")
        normalize_entity(entity)
        id_map[old_id] = entity["id"]
    _remap_endpoints(relationships, id_map)

    seen: set[str] = set()
    deduped_entities: list[dict] = []
    for entity in entities:
        if entity["id"] not in seen:
            seen.add(entity["id"])
            deduped_entities.append(entity)

    seen_rels: set[tuple] = set()
    deduped_rels: list[dict] = []
    for rel in relationships:
        key = (rel.get("source", ""), rel.get("target", ""), rel.get("type", ""))
        if key not in seen_rels:
            seen_rels.add(key)
            deduped_rels.append(rel)
    return deduped_entities, deduped_rels
