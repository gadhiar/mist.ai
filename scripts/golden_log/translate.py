"""Gold record -> extractor-native `(entities, relationships)`.

`LogRegenerator.rebuild` feeds a cached payload STRAIGHT into
`ValidationResult(valid=True, entities=..., relationships=...)` with no adaptation, so a
golden-log cache entry has to already be in the extractor's native shape. The gold corpus
is not in that shape. This module is the conversion, and the field mapping it uses lives in
`native_shape` alongside the reverse mapping the F2 scorer applies.

Both traps the shape difference sets are handled there rather than here: `predicate` becomes
`type` only via `build_native_relationship`, and gold's `source_type` / `target_type` cannot
reach the relationship because `build_native_relationship` has no parameter for them.

What gold's endpoint types ARE used for: cross-checking. Gold declares each endpoint's type
twice -- once in `expected_entities`, once on the relationship -- and the corpus is
internally consistent (verified across all 60 records). `translate_gold_record` enforces
that agreement, so the discarded field becomes a validation input rather than dropped data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .native_shape import (
    ASSERTION_KIND_KEY,
    GOLD_ENTITY_ID_FIELD,
    GOLD_ENTITY_TYPE_FIELD,
    GOLD_PREDICATE_FIELD,
    GOLD_SOURCE_FIELD,
    GOLD_TARGET_FIELD,
    GOLD_VALID_FROM_FIELD,
    GOLD_VALID_TO_FIELD,
    NativeShapeError,
    build_native_entity,
    build_native_relationship,
)

GOLD_ENTITIES_FIELD = "expected_entities"
GOLD_RELATIONSHIPS_FIELD = "expected_relationships"
GOLD_TAG_FIELD = "tag"
GOLD_UTTERANCE_FIELD = "utterance"


class GoldTranslationError(NativeShapeError):
    """Raised when a gold record cannot be translated into the native shape."""


def _endpoint_type_field(side: str) -> str:
    """Gold's field naming the ENTITY TYPE of `side` ("source" or "target")."""
    return f"{side}_type"


def _check_endpoint_types(tag: str, rel: dict[str, Any], entity_types: dict[str, str]) -> None:
    """Assert gold's endpoint types agree with the types on `expected_entities`.

    Raises:
        GoldTranslationError: On a missing endpoint entity or a type disagreement.
    """
    for side, field in ((GOLD_SOURCE_FIELD, "source"), (GOLD_TARGET_FIELD, "target")):
        endpoint_id = rel[side]
        declared = rel.get(_endpoint_type_field(field))
        if endpoint_id not in entity_types:
            raise GoldTranslationError(
                f"{tag}: relationship {field} {endpoint_id!r} has no matching entry in "
                f"{GOLD_ENTITIES_FIELD}, so its entity type cannot be carried onto an entity"
            )
        if declared is not None and entity_types[endpoint_id] != declared:
            raise GoldTranslationError(
                f"{tag}: relationship declares {field} type {declared!r} for "
                f"{endpoint_id!r} but {GOLD_ENTITIES_FIELD} types it "
                f"{entity_types[endpoint_id]!r}"
            )


def translate_gold_record(record: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    """Translate one gold record into native `(entities, relationships)`.

    Args:
        record: A gold corpus record -- `expected_entities` plus
            `expected_relationships`, each relationship carrying `predicate` and the
            endpoint ENTITY TYPES under `source_type` / `target_type`.

    Returns:
        `(entities, relationships)` in the extractor's native shape, ready to hand to
        `ExtractionCache.put`. Entity types survive onto the entities; relationships carry
        `type` and a provenance `source_type`.

    Raises:
        GoldTranslationError: On an endpoint with no entity, or an endpoint type that
            disagrees between the relationship and `expected_entities`.
    """
    tag = str(record.get(GOLD_TAG_FIELD, "<untagged>"))

    entity_types: dict[str, str] = {}
    entities: list[dict] = []
    for gold_entity in record.get(GOLD_ENTITIES_FIELD, []):
        entity_id = gold_entity[GOLD_ENTITY_ID_FIELD]
        entity_type = gold_entity[GOLD_ENTITY_TYPE_FIELD]
        entity_types[entity_id] = entity_type
        entities.append(build_native_entity(entity_id=entity_id, entity_type=entity_type))

    relationships: list[dict] = []
    for gold_rel in record.get(GOLD_RELATIONSHIPS_FIELD, []):
        _check_endpoint_types(tag, gold_rel, entity_types)
        relationships.append(
            build_native_relationship(
                source=gold_rel[GOLD_SOURCE_FIELD],
                target=gold_rel[GOLD_TARGET_FIELD],
                predicate=gold_rel[GOLD_PREDICATE_FIELD],
                assertion_kind=gold_rel.get(ASSERTION_KIND_KEY),
                valid_from=gold_rel.get(GOLD_VALID_FROM_FIELD),
                valid_to=gold_rel.get(GOLD_VALID_TO_FIELD),
            )
        )

    return entities, relationships


def load_gold_corpus(path: Path) -> dict[str, dict[str, Any]]:
    """Load a gold corpus JSONL into `{tag: record}`.

    Raises:
        GoldTranslationError: On malformed JSON, a missing tag, or a duplicate tag (which
            would make the derived event id ambiguous).
    """
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GoldTranslationError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            tag = record.get(GOLD_TAG_FIELD)
            if not tag:
                raise GoldTranslationError(
                    f"{path}:{line_no}: record has no {GOLD_TAG_FIELD}; event ids are "
                    "derived from it and must never be generated"
                )
            if tag in records:
                raise GoldTranslationError(f"{path}:{line_no}: duplicate tag {tag!r}")
            records[tag] = record
    return records
