"""The pack registry: which ontology packs exist, and which one is active.

`active_pack()` is what the normalizer, the validator and the scorer's id rails run
with when they are given no pack. The active version is `MIST_ONTOLOGY_VERSION` when it
is set, and otherwise the newest pack that is not deprecated.

Until a pack ships as files of its own, the registry holds one pack, the adapter: today's
ontology read from where it lives (`v1_0_0.py`, `hierarchy.py` and the ontology half of
`backend/knowledge/extraction/id_rails.py`) and stated as pack rules, so the rail runner
gives the outputs the id rails gave before it. Its version is the ontology's own. A
later version adds a pack package and a line in `_BUILDERS`, and no engine edit.

The registry is a directory of packs and nothing more. A pack is built once, and what
the loader refuses (`PackLoadError`) is refused here. A pack's pinned cases are not run
here: the rail runner runs them the first time it is handed the pack, and the normalizer
when it is built. So this module imports nothing from the runner, and reading a pack
here does not load it: the extraction service must never import the runner.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from functools import cache

from backend.knowledge.extraction import id_rails
from backend.knowledge.ontologies.hierarchy import ANCHOR_CHILD, RETIRED_TYPE_MAP
from backend.knowledge.ontologies.pack import (
    ID,
    PAYLOAD,
    PRE_LLM,
    AliasVocabulary,
    ConventionRule,
    DerivedEdge,
    IdExempt,
    Manifest,
    Pack,
    PackLoadError,
    PinnedCase,
    PlanVocabulary,
    Plugin,
    ReservedIdentity,
    RetiredType,
    TypedId,
    build_pack,
)
from backend.knowledge.ontologies.v1_0_0 import (
    ALL_EDGE_TYPES,
    ALL_NODE_TYPES,
    DATE,
    EVENT,
    EXTRACTABLE_NODE_TYPES,
    EXTRACTABLE_RELATIONSHIP_TYPES,
    METRIC,
    OCCURRED_ON,
    ONTOLOGY_V1_0_0,
    RELATED_TO,
    TECHNOLOGY,
    USER,
)

ENV_VAR = "MIST_ONTOLOGY_VERSION"

_TECHNOLOGY_VERSION = "technology-version"
_METRIC_ID = "metric-id"
_EVENT_DATE = "event-date"


def _case(
    entities: tuple[Mapping, ...],
    expected: tuple[Mapping, ...],
    relationships: tuple[Mapping, ...] = (),
    expected_relationships: tuple[Mapping, ...] = (),
) -> PinnedCase:
    return PinnedCase(
        entities=entities,
        relationships=relationships,
        expected_entities=expected,
        expected_relationships=expected_relationships,
    )


def _adapter_rules() -> tuple[ConventionRule, ...]:
    """Today's id rails as pack rules, in the order `normalize_entity` ran them."""
    technology, metric = TECHNOLOGY.type_name, METRIC.type_name
    event, date, occurred_on = EVENT.type_name, DATE.type_name, OCCURRED_ON.type_name
    names_by_identity: dict[tuple[str, str], list[str]] = {}
    for name, identity in id_rails.RESERVED_NAMES.items():
        names_by_identity.setdefault(identity, []).append(name)
    reserved = tuple(
        ConventionRule(
            id=f"reserved-{stored_id}",
            kind="reserved_identity",
            phase=PAYLOAD,
            params=ReservedIdentity(names=tuple(names), id=stored_id, type=stored_type),
        )
        for (stored_id, stored_type), names in names_by_identity.items()
    )
    dated = {"source": "regatta", "target": "2026-05-30", "type": occurred_on}
    return (
        *reserved,
        ConventionRule(
            id="n1-exempt",
            kind="id_exempt",
            phase=ID,
            params=IdExempt(types=id_rails.ID_EXEMPT_TYPES),
        ),
        ConventionRule(
            id="pre-canon-aliases",
            kind="alias_vocabulary",
            phase=ID,
            params=AliasVocabulary(aliases=tuple(id_rails.PRE_CANON_ALIASES.items()), match="name"),
        ),
        ConventionRule(
            id=_TECHNOLOGY_VERSION,
            kind="typed_id",
            phase=ID,
            params=TypedId(type=technology, plugin=_TECHNOLOGY_VERSION),
            pinned_case=_case(
                ({"id": "quillscript-3-11", "name": "Quillscript 3.11", "type": technology},),
                ({"id": "quillscript", "name": "Quillscript 3.11", "type": technology},),
            ),
        ),
        ConventionRule(
            id="static-aliases",
            kind="alias_vocabulary",
            phase=ID,
            params=AliasVocabulary(aliases=tuple(id_rails.STATIC_ALIASES.items())),
        ),
        ConventionRule(
            id=_METRIC_ID,
            kind="typed_id",
            phase=ID,
            params=TypedId(type=metric, plugin=_METRIC_ID),
            pinned_case=_case(
                ({"id": "tool-selection-0.94", "name": "tool selection", "type": metric},),
                ({"id": "0-94-tool-selection", "name": "tool selection", "type": metric},),
            ),
        ),
        ConventionRule(
            id=_EVENT_DATE,
            kind="derived_edge",
            phase=ID,
            params=DerivedEdge(
                source_type=event, edge=occurred_on, target_type=date, plugin=_EVENT_DATE
            ),
            pinned_case=_case(
                (
                    {"id": "regatta", "name": "Regatta", "type": event},
                    {"id": "2026-05-30", "name": "2026-05-30", "type": date},
                ),
                (
                    {"id": "regatta-2026-05-30", "name": "Regatta 2026-05-30", "type": event},
                    {"id": "2026-05-30", "name": "2026-05-30", "type": date},
                ),
                relationships=(dated,),
                expected_relationships=({**dated, "source": "regatta-2026-05-30"},),
            ),
        ),
        ConventionRule(
            id="plan-vocabulary",
            kind="plan_vocabulary",
            phase=PRE_LLM,
            params=PlanVocabulary(
                entity_types=tuple(EXTRACTABLE_NODE_TYPES),
                predicates=tuple(EXTRACTABLE_RELATIONSHIP_TYPES),
                fan_out_predicates=tuple(EXTRACTABLE_RELATIONSHIP_TYPES),
            ),
        ),
    )


def _adapter_pack() -> Pack:
    """Today's ontology as a pack, read from the modules that still hold it."""
    manifest = Manifest(
        version=ONTOLOGY_V1_0_0.version,
        fallback_type=id_rails.FALLBACK_TYPE,
        user_id=id_rails.USER_ID,
        user_type=USER.type_name,
        parent_version=ONTOLOGY_V1_0_0.parent_version,
        retired_types=tuple(
            RetiredType(name, replacement, id_rails.RETIRED_PROPERTY_DEFAULTS.get(name, ()))
            for name, replacement in RETIRED_TYPE_MAP.items()
        ),
        supertypes=tuple(ANCHOR_CHILD.items()),
        escape_edge=RELATED_TO.type_name,
        migration_script_path=ONTOLOGY_V1_0_0.migration_script_path,
        deprecated=ONTOLOGY_V1_0_0.deprecated,
    )
    return build_pack(
        manifest=manifest,
        node_types=tuple(ALL_NODE_TYPES),
        edge_types=tuple(ALL_EDGE_TYPES),
        extractable_node_types=tuple(EXTRACTABLE_NODE_TYPES),
        extractable_edge_types=tuple(EXTRACTABLE_RELATIONSHIP_TYPES),
        rules=_adapter_rules(),
        plugins=(
            Plugin(_TECHNOLOGY_VERSION, ID, id_rails.technology_version_id),
            Plugin(_METRIC_ID, ID, id_rails.metric_id),
            Plugin(_EVENT_DATE, ID, id_rails.event_date_id),
        ),
    )


# Each version the registry knows, and what builds its pack.
_BUILDERS: dict[str, Callable[[], Pack]] = {
    ONTOLOGY_V1_0_0.version: _adapter_pack,
}


def _version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) if part.isdigit() else 0 for part in version.split("."))


def available_versions() -> tuple[str, ...]:
    """Every version the registry can build, oldest first."""
    return tuple(sorted(_BUILDERS, key=_version_key))


@cache
def get_pack(version: str) -> Pack:
    """The pack of one ontology version, built once.

    The pack's pinned cases are not run here; the rail runner and the normalizer run
    them.

    Args:
        version: A version `available_versions` lists.

    Raises:
        PackLoadError: The version is unknown, or the loader refuses the pack.
    """
    builder = _BUILDERS.get(version)
    if builder is None:
        known = ", ".join(available_versions())
        raise PackLoadError(f"no ontology pack for version {version} (known: {known})")
    return builder()


def active_pack() -> Pack:
    """The pack the engine runs with: `MIST_ONTOLOGY_VERSION`, or the newest in use.

    Raises:
        PackLoadError: The named version is unknown, every pack is deprecated, or the
            pack is refused at load.
    """
    named = os.environ.get(ENV_VAR, "").strip()
    if named:
        return get_pack(named)
    for version in reversed(available_versions()):
        pack = get_pack(version)
        if not pack.manifest.deprecated:
            return pack
    raise PackLoadError("no ontology pack is in use: every registered pack is deprecated")
