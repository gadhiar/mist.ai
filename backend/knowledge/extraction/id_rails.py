"""The id rails: how a stored entity id is built from the extraction payload's `id` field.

ADR-054 Decision 2: the graph stores the model's `id` field, not an id built from the
display name, and every convention the gold labels by id has a carrier, a prompt rule or a
deterministic rail. The rails are pure and synchronous, so that the normalizer
(`EntityNormalizer.normalize`) and the eval scorer (`scripts/eval_harness/
score_extraction_run.py`) build every id with the same functions, and a test can assert
that the id the graph stores is the id the scorer scores.

The module is engine: each function runs the rail runner
(`backend.knowledge.extraction.rails`) with the registry's active pack, and the module
names no type and no edge, which a test holds it to. What an ontology gives the rails,
the alias maps, the reserved names, the exempt and retired types and the id plug-ins,
is stated in its pack's own files.

This module imports no graph, embedding or I/O dependency, so the scorer can import it
without the normalizer's. The runner and the registry are imported inside the functions
that call them, because both import this module.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

# Bounded curated (id, type) overrides for recurring high-value entities, keyed by the id
# the pack's alias vocabularies give. Seeded later from curated vocab (the versioned seed
# source), NEVER from F2 probes. Authoritative: a registry hit overrides both id and type
# and ends the steps. Empty for now. The runner reads this dict at each call, so an entry
# added in place takes effect at once.
CANONICAL_REGISTRY: dict[str, tuple[str, str]] = {}

# An ISO day, the form a date's id takes. A date format, not ontology data: a pack's
# plug-in and the eval scorer both read it from here.
ISO_DATE_RE: re.Pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")

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
    entities: list[dict],
    relationships: list[dict],
    *,
    batch_record: Sequence[Mapping] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Run the rails over one payload: `rails.run_rails` with the active pack.

    The payload stands for a row with no scope and no plan, and with the batch record
    the caller hands over. Each entity's stored id is built: the id of an entity that
    is not settled, of a type the pack does not exempt, first loses the context its
    batch was read under when the record names one (`rails._own_contexts`) and that
    context may be dropped (`rails._strip_own_contexts`: it names no entity of the
    payload, and the id rules after the strip build from what is left neither the user
    anchor's id nor a reserved name or id). Relationship endpoints that name an
    entity's `id` field are remapped to its stored id; the id-phase `derived_edge`
    rules run, and the endpoints naming a renamed entity are remapped again; entities
    are deduplicated first-wins by id, and relationships by (source, target, type). The
    entity and relationship dicts are changed in place.

    Args:
        entities: The payload's entity dicts.
        relationships: The payload's relationship dicts.
        batch_record: The row's batch record, one entry per Stage 2 call, so that a
            rule that reads the record acts here as it does when the row is applied.
            None stands for a row extracted as one batch, which offers no context.

    Returns:
        The deduplicated entities and relationships.
    """
    from backend.knowledge.extraction import rails
    from backend.knowledge.ontologies import registry

    row = rails.rail_row(entities, relationships, batch_record=batch_record)
    return rails.run_rails(row, registry.active_pack())
