"""One authority for the gold-corpus <-> extractor-native field mapping.

Two shapes describe the same facts and they are NOT the same shape:

- **Native** -- what the extractor emits, what `ExtractionCache` stores, and what
  `LogRegenerator.rebuild` feeds straight into `ValidationResult`. Relationships carry
  `type`, and `source_type` is PROVENANCE ("how this edge was acquired").
- **Gold** -- `data/ingest/extraction-gold-2026-06-14.jsonl`. Relationships carry
  `predicate`, and `source_type` / `target_type` are the ENTITY TYPES of the two endpoints.

`scripts/eval_harness/score_extraction_run.py` reads native and compares against gold;
`scripts/golden_log/translate.py` writes native from gold. Both directions of the same
mapping, so both go through this module. Answering the same question independently in two
places is how R1.4's C1 regression happened.

## The `source_type` collision

The name is shared and the meanings are not, so a field-copy translation writes
`r.source_type = "User"` onto every edge: a populated string of the right type in the right
field, passing every structural and schema check, silently corrupting provenance across the
whole corpus. `assert_valid_provenance` refuses that by checking the value against the
vocabulary the ONTOLOGY declares for the relationship property -- so the check cannot drift
from the ontology, and a node-type name fails it by construction.

Gold's `source_type` / `target_type` describe the endpoints and belong on the entities.
`ExtractionPipeline._apply_third_party_penalty` and the F2 scorer both read endpoint types
out of the entity list, never off the relationship, which is the same placement.
"""

from __future__ import annotations

from typing import Any

from backend.errors import MistError
from backend.knowledge.ontologies.v1_0_0 import (
    UNIVERSAL_ENTITY_PROPERTIES,
    UNIVERSAL_RELATIONSHIP_PROPERTIES,
)


class NativeShapeError(MistError):
    """Raised when a payload violates the extractor's native relationship shape."""


# --- native shape (extractor output / ExtractionCache payload) ---------------------------
#
# Read off `EdgeAssertion.from_rel_dict` (backend/knowledge/curation/reconciliation.py) and
# `GraphWriter._upsert_entity` (backend/knowledge/curation/graph_writer.py).

NATIVE_PREDICATE_FIELD = "type"
NATIVE_SOURCE_FIELD = "source"
NATIVE_TARGET_FIELD = "target"
NATIVE_PROPERTIES_FIELD = "properties"
NATIVE_PROVENANCE_FIELD = "source_type"
NATIVE_ENTITY_ID_FIELD = "id"
NATIVE_ENTITY_NAME_FIELD = "name"
NATIVE_ENTITY_TYPE_FIELD = "type"

# --- gold shape (data/ingest/extraction-gold-*.jsonl) ------------------------------------

GOLD_PREDICATE_FIELD = "predicate"
GOLD_SOURCE_FIELD = "source"
GOLD_TARGET_FIELD = "target"
GOLD_ENTITY_ID_FIELD = "id"
GOLD_ENTITY_TYPE_FIELD = "type"

# Gold fields naming the ENTITY TYPE of each endpoint. They describe the entities, so they
# belong on the entity dicts. Copying either onto a native relationship is trap 1.
GOLD_ENDPOINT_TYPE_FIELDS = ("source_type", "target_type")

# Gold valid-time bounds land in native `properties` under the keys the temporal resolver
# writes and `EdgeAssertion.from_rel_dict` reads back.
GOLD_VALID_FROM_FIELD = "valid_from"
GOLD_VALID_TO_FIELD = "valid_to"
NATIVE_START_DATE_KEY = "start_date"
NATIVE_END_DATE_KEY = "end_date"

# `derive_assertion_kind` reads the rel level first, then properties. The model only ever
# emits the field inside `properties`, so authored payloads put it there too.
ASSERTION_KIND_KEY = "assertion_kind"


def _allowed_provenance(properties: tuple[Any, ...]) -> frozenset[str]:
    """Pull the `source_type` vocabulary out of an ontology property tuple."""
    for prop in properties:
        if prop.name == NATIVE_PROVENANCE_FIELD:
            return frozenset(prop.allowed_values or ())
    raise NativeShapeError(
        f"ontology declares no {NATIVE_PROVENANCE_FIELD!r} property -- the provenance "
        "vocabulary has no source to derive from"
    )


# Derived from the ontology, never restated. `PropertyDefinition.allowed_values` for
# `source_type` is the vocabulary; a node-type name such as "User" is not in it.
RELATIONSHIP_PROVENANCE_VALUES: frozenset[str] = _allowed_provenance(
    UNIVERSAL_RELATIONSHIP_PROPERTIES
)
ENTITY_PROVENANCE_VALUES: frozenset[str] = _allowed_provenance(UNIVERSAL_ENTITY_PROPERTIES)

# The provenance of a fact recovered from a stored extraction. Pinned here rather than
# derived (the vocabulary is a set, not an ordered list, so "the first one" is not a
# meaning); `TestProvenanceVocabulary` pins it as a member of the ontology's vocabulary.
EXTRACTED_PROVENANCE = "extracted"


# --- native readers (used by the F2 scorer) ----------------------------------------------


def native_predicate(rel: dict[str, Any]) -> str:
    """Predicate of a native relationship dict.

    `EdgeAssertion.from_rel_dict` reads `rel["type"]`. A payload carrying gold's
    `predicate` yields an empty predicate here, which fails ontology lookup and drops the
    fact -- trap 2.
    """
    return str(rel.get(NATIVE_PREDICATE_FIELD, ""))


def native_endpoints(rel: dict[str, Any]) -> tuple[str, str]:
    """(source, target) ids of a native relationship dict."""
    return (
        str(rel.get(NATIVE_SOURCE_FIELD, "")),
        str(rel.get(NATIVE_TARGET_FIELD, "")),
    )


def native_properties(rel: dict[str, Any]) -> dict[str, Any]:
    """Properties of a native relationship dict; empty dict when absent or malformed."""
    props = rel.get(NATIVE_PROPERTIES_FIELD)
    return props if isinstance(props, dict) else {}


def native_entity_id(entity: dict[str, Any]) -> str:
    """Id of a native entity dict, falling back to `name` as the extractor's output does."""
    return str(entity.get(NATIVE_ENTITY_ID_FIELD) or entity.get(NATIVE_ENTITY_NAME_FIELD) or "")


def native_entity_type(entity: dict[str, Any]) -> str:
    """Ontology type of a native entity dict."""
    return str(entity.get(NATIVE_ENTITY_TYPE_FIELD, ""))


# --- native writers (used by the golden-log translator) ----------------------------------


def assert_valid_provenance(rel: dict[str, Any]) -> None:
    """Refuse a relationship whose `source_type` is not a provenance value.

    The failure this exists for: gold's `source_type` is the SUBJECT'S ENTITY TYPE, so a
    field-copy translation writes `source_type: "User"` onto the edge. That is structurally
    valid -- a non-empty string in a string field -- and every schema check passes, but it
    means "acquired via User", which is not a thing. The vocabulary is read from the
    ontology, so this cannot drift from the ontology's own declaration.

    Raises:
        NativeShapeError: When `source_type` is present and outside the ontology's
            `allowed_values` for the relationship property.
    """
    if NATIVE_PROVENANCE_FIELD not in rel and NATIVE_PROVENANCE_FIELD not in native_properties(rel):
        return
    value = native_properties(rel).get(NATIVE_PROVENANCE_FIELD) or rel.get(NATIVE_PROVENANCE_FIELD)
    if value not in RELATIONSHIP_PROVENANCE_VALUES:
        raise NativeShapeError(
            f"relationship {NATIVE_PROVENANCE_FIELD}={value!r} is not a provenance value. "
            f"The ontology allows {sorted(RELATIONSHIP_PROVENANCE_VALUES)}. Gold's "
            f"{GOLD_ENDPOINT_TYPE_FIELDS} name the ENDPOINT ENTITY TYPES and belong on the "
            "entity dicts, not on the relationship."
        )


def build_native_entity(
    *, entity_id: str, entity_type: str, provenance: str = EXTRACTED_PROVENANCE
) -> dict[str, Any]:
    """Build a native entity dict.

    `confidence` is deliberately omitted: `GraphWriter._upsert_entity` supplies its own
    default, and restating it here would be a second authority for the same number.
    """
    return {
        NATIVE_ENTITY_ID_FIELD: entity_id,
        NATIVE_ENTITY_TYPE_FIELD: entity_type,
        NATIVE_PROVENANCE_FIELD: provenance,
    }


def build_native_relationship(
    *,
    source: str,
    target: str,
    predicate: str,
    assertion_kind: str | None = None,
    valid_from: str | None = None,
    valid_to: str | None = None,
    provenance: str = EXTRACTED_PROVENANCE,
) -> dict[str, Any]:
    """Build a native relationship dict from endpoint ids and a predicate.

    Takes no endpoint TYPES. That is the point: gold carries them and they belong on the
    entities, so there is no parameter through which they could reach the edge.

    `confidence` is omitted for the same reason as in `build_native_entity` --
    `EdgeAssertion.from_rel_dict` supplies the default.

    Raises:
        NativeShapeError: If `provenance` is not in the ontology's vocabulary.
    """
    properties: dict[str, Any] = {NATIVE_PROVENANCE_FIELD: provenance}
    if assertion_kind is not None:
        properties[ASSERTION_KIND_KEY] = assertion_kind
    if valid_from is not None:
        properties[NATIVE_START_DATE_KEY] = valid_from
    if valid_to is not None:
        properties[NATIVE_END_DATE_KEY] = valid_to

    rel = {
        NATIVE_SOURCE_FIELD: source,
        NATIVE_TARGET_FIELD: target,
        NATIVE_PREDICATE_FIELD: predicate,
        NATIVE_PROPERTIES_FIELD: properties,
    }
    assert_valid_provenance(rel)
    return rel
