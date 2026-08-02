"""Gold -> extractor-native translation, and the two traps in the shape difference.

Trap 1 (`source_type`) is the dangerous one: gold uses the name for the SUBJECT'S ENTITY
TYPE, the pipeline uses it for PROVENANCE. A field-copy translation writes
`r.source_type = "User"` on every edge -- a populated string of the right type in the right
field, so it passes every structural and schema check while silently corrupting provenance
across the whole corpus. The tests here assert against the real consumers
(`EdgeAssertion.from_rel_dict`, the ontology's own `allowed_values`) rather than against a
restatement of the expected shape.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.knowledge.curation.reconciliation import EdgeAssertion, derive_assertion_kind
from backend.knowledge.ontologies import EDGE_TYPES_BY_NAME
from backend.knowledge.ontologies.v1_0_0 import ALL_NODE_TYPE_NAMES
from scripts.golden_log.native_shape import (
    EXTRACTED_PROVENANCE,
    GOLD_ENDPOINT_TYPE_FIELDS,
    RELATIONSHIP_PROVENANCE_VALUES,
    NativeShapeError,
    build_native_relationship,
    native_entity_id,
    native_entity_type,
    native_predicate,
    native_properties,
)
from scripts.golden_log.translate import (
    GoldTranslationError,
    load_gold_corpus,
    translate_gold_record,
)

GOLD_PATH = Path("data/ingest/extraction-gold-2026-06-14.jsonl")


@pytest.fixture(scope="module")
def gold_corpus() -> dict[str, dict]:
    return load_gold_corpus(GOLD_PATH)


def build_gold_record(
    *,
    tag: str = "test-record",
    entities: list[dict] | None = None,
    relationships: list[dict] | None = None,
) -> dict:
    """Build a valid gold-shaped record with overridable entities and relationships."""
    return {
        "tag": tag,
        "utterance": "I use Rust.",
        "expected_entities": (
            entities
            if entities is not None
            else [{"id": "user", "type": "User"}, {"id": "rust", "type": "Technology"}]
        ),
        "expected_relationships": (
            relationships
            if relationships is not None
            else [
                {
                    "source": "user",
                    "source_type": "User",
                    "predicate": "USES",
                    "target": "rust",
                    "target_type": "Technology",
                }
            ]
        ),
    }


class TestProvenanceIsNotAnEntityType:
    """Trap 1: gold's `source_type` is an entity type; the edge's is provenance."""

    def test_translated_relationship_carries_provenance_not_the_gold_entity_type(self):
        # Arrange: gold says source_type "User" -- the SUBJECT'S TYPE, not provenance.
        record = build_gold_record()

        # Act
        _entities, relationships = translate_gold_record(record)

        # Assert: the edge's source_type means "how this fact was acquired".
        assert native_properties(relationships[0])["source_type"] == EXTRACTED_PROVENANCE
        assert native_properties(relationships[0])["source_type"] != "User"

    def test_reconciliation_reads_a_provenance_value_off_the_translated_edge(self):
        # Arrange: the real consumer, not a restatement of the expected shape.
        record = build_gold_record()
        _entities, relationships = translate_gold_record(record)

        # Act
        assertion = EdgeAssertion.from_rel_dict(relationships[0], EDGE_TYPES_BY_NAME["USES"])

        # Assert: what lands on `r.source_type` in Neo4j is a provenance value.
        assert assertion.source_type in RELATIONSHIP_PROVENANCE_VALUES

    def test_no_gold_entity_type_reaches_any_relationship_in_the_corpus(self, gold_corpus):
        # Arrange: every ontology node-type name is a value trap 1 could smuggle through.
        node_type_names = set(ALL_NODE_TYPE_NAMES)

        # Act / Assert
        for tag, record in gold_corpus.items():
            _entities, relationships = translate_gold_record(record)
            for rel in relationships:
                provenance = native_properties(rel)["source_type"]
                assert provenance not in node_type_names, (
                    f"{tag}: relationship provenance {provenance!r} is an ontology node "
                    "type -- gold's endpoint entity type was copied onto the edge"
                )
                assert provenance in RELATIONSHIP_PROVENANCE_VALUES

    def test_no_relationship_carries_golds_endpoint_type_fields(self, gold_corpus):
        # Assert: `target_type` has no native meaning at all, so its presence is pure leak.
        for tag, record in gold_corpus.items():
            _entities, relationships = translate_gold_record(record)
            for rel in relationships:
                props = native_properties(rel)
                assert "target_type" not in rel, f"{tag}: gold target_type leaked onto edge"
                assert "target_type" not in props, f"{tag}: gold target_type leaked to props"

    def test_build_native_relationship_rejects_an_entity_type_as_provenance(self):
        # Act / Assert: the guard names the confusion rather than failing structurally.
        with pytest.raises(NativeShapeError, match="not a provenance value"):
            build_native_relationship(
                source="user", target="rust", predicate="USES", provenance="User"
            )

    def test_provenance_vocabulary_comes_from_the_ontology(self):
        # Assert: derived, so it cannot drift from the ontology's own declaration.
        assert EXTRACTED_PROVENANCE in RELATIONSHIP_PROVENANCE_VALUES
        assert not RELATIONSHIP_PROVENANCE_VALUES & set(ALL_NODE_TYPE_NAMES)

    def test_gold_endpoint_types_survive_onto_the_entities(self, gold_corpus):
        # Assert: the types gold declares are not discarded -- they land where they belong.
        for tag, record in gold_corpus.items():
            entities, _relationships = translate_gold_record(record)
            translated = {native_entity_id(e): native_entity_type(e) for e in entities}
            expected = {e["id"]: e["type"] for e in record["expected_entities"]}
            assert translated == expected, f"{tag}: entity types did not survive translation"

    def test_endpoint_type_fields_are_named_so_the_leak_is_greppable(self):
        # Assert: the collision is documented in one place, not rediscovered per call site.
        assert GOLD_ENDPOINT_TYPE_FIELDS == ("source_type", "target_type")


class TestPredicateBecomesType:
    """Trap 2: the pipeline reads `rel["type"]`; a payload carrying `predicate` drops."""

    def test_translated_relationship_carries_type_not_predicate(self):
        # Arrange
        record = build_gold_record()

        # Act
        _entities, relationships = translate_gold_record(record)

        # Assert
        assert relationships[0]["type"] == "USES"
        assert "predicate" not in relationships[0]

    def test_reconciliation_recovers_a_non_empty_predicate(self):
        # Arrange: the real consumer. A `predicate` payload yields "" here and drops.
        record = build_gold_record()
        _entities, relationships = translate_gold_record(record)

        # Act
        assertion = EdgeAssertion.from_rel_dict(relationships[0], EDGE_TYPES_BY_NAME["USES"])

        # Assert
        assert assertion.predicate == "USES"

    def test_every_corpus_predicate_survives_into_the_native_type_field(self, gold_corpus):
        # Act / Assert: an empty predicate would fail ontology lookup and drop the fact.
        for tag, record in gold_corpus.items():
            _entities, relationships = translate_gold_record(record)
            produced = [native_predicate(r) for r in relationships]
            expected = [r["predicate"] for r in record["expected_relationships"]]
            assert produced == expected, f"{tag}: predicate did not become type"
            assert all(produced), f"{tag}: empty predicate would drop the fact"


class TestAssertionKindAndValidTime:
    def test_cease_kind_reaches_the_engines_deriver(self, gold_corpus):
        # Arrange: ext-45 is gold's spoken-cease case.
        _entities, relationships = translate_gold_record(gold_corpus["ext-45-cease-learning"])

        # Act
        kind, past_mapped = derive_assertion_kind(relationships[0])

        # Assert
        assert kind.value == "cease"
        assert past_mapped is False

    def test_retract_kind_reaches_the_engines_deriver(self, gold_corpus):
        # Arrange
        _entities, relationships = translate_gold_record(
            gold_corpus["ext-48-retract-never-actually"]
        )

        # Act
        kind, _past_mapped = derive_assertion_kind(relationships[0])

        # Assert
        assert kind.value == "retract"

    def test_valid_to_becomes_the_end_date_the_engine_reads(self, gold_corpus):
        # Arrange: ext-35 bounds the LEARNING edge with valid_to 2025-12.
        _entities, relationships = translate_gold_record(gold_corpus["ext-35-validtime-until"])

        # Act
        assertion = EdgeAssertion.from_rel_dict(relationships[0], EDGE_TYPES_BY_NAME["LEARNING"])

        # Assert
        assert assertion.valid_to_stated == "2025-12"

    def test_absent_assertion_kind_is_not_authored_as_a_key(self):
        # Assert: an authored payload states only what gold states.
        _entities, relationships = translate_gold_record(build_gold_record())
        assert "assertion_kind" not in native_properties(relationships[0])


class TestEndpointTypeCrossCheck:
    """Gold declares each endpoint type twice; the translator makes them agree."""

    def test_raises_when_relationship_and_entity_list_disagree(self):
        # Arrange: entity list says Technology, relationship says Organization.
        record = build_gold_record(
            relationships=[
                {
                    "source": "user",
                    "source_type": "User",
                    "predicate": "USES",
                    "target": "rust",
                    "target_type": "Organization",
                }
            ]
        )

        # Act / Assert
        with pytest.raises(GoldTranslationError, match="but expected_entities types it"):
            translate_gold_record(record)

    def test_raises_when_an_endpoint_has_no_entity(self):
        # Arrange: the relationship references an entity gold never declared.
        record = build_gold_record(entities=[{"id": "user", "type": "User"}])

        # Act / Assert
        with pytest.raises(GoldTranslationError, match="no matching entry"):
            translate_gold_record(record)


class TestGoldCorpusLoad:
    def test_loads_all_sixty_records_keyed_by_tag(self, gold_corpus):
        assert len(gold_corpus) == 60

    def test_negative_controls_translate_to_nothing(self, gold_corpus):
        # Assert: a negative probe must not manufacture facts on the way through.
        entities, relationships = translate_gold_record(gold_corpus["ext-11-smalltalk-negative"])
        assert entities == []
        assert relationships == []

    def test_rejects_a_duplicate_tag(self, tmp_path):
        # Arrange: duplicate tags would make the derived event id ambiguous.
        path = tmp_path / "dupe.jsonl"
        path.write_text(
            '{"tag": "a", "utterance": "x"}\n{"tag": "a", "utterance": "y"}\n', encoding="utf-8"
        )

        # Act / Assert
        with pytest.raises(GoldTranslationError, match="duplicate tag"):
            load_gold_corpus(path)
