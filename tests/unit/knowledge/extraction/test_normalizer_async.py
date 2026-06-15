"""Tests for EntityNormalizer async migration."""

import pytest

from backend.knowledge.extraction.normalizer import EntityNormalizer
from backend.knowledge.extraction.ontology_extractor import ExtractionResult
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeGraphExecutor, FakeNeo4jConnection


def _result(entities, relationships=None):
    """Build an ExtractionResult with the real constructor signature."""
    return ExtractionResult(entities=entities, relationships=relationships or [])


@pytest.fixture
def normalizer_no_graph():
    """EntityNormalizer with no graph executor (local-only mode)."""
    return EntityNormalizer(
        embedding_generator=FakeEmbeddingGenerator(),
        executor=None,
    )


class TestNormalizerAsync:
    @pytest.mark.asyncio
    async def test_normalize_is_async(self):
        normalizer = EntityNormalizer(
            embedding_generator=FakeEmbeddingGenerator(),
            executor=None,
        )
        extraction = ExtractionResult(
            entities=[{"id": "Python", "name": "Python", "type": "Technology"}],
            relationships=[],
        )
        result = await normalizer.normalize(extraction)
        assert result.entities[0]["id"] == "python"

    @pytest.mark.asyncio
    async def test_graph_executor_present_does_not_change_string_canonicalization(self):
        # R1.1d: passing an executor no longer triggers graph-identity queries.
        # String canonicalization result must be identical with or without executor.
        conn = FakeNeo4jConnection()
        executor = FakeGraphExecutor(connection=conn)
        normalizer = EntityNormalizer(
            embedding_generator=FakeEmbeddingGenerator(),
            executor=executor,
        )
        extraction = ExtractionResult(
            entities=[{"id": "Python", "name": "Python", "type": "Technology"}],
            relationships=[],
        )
        result = await normalizer.normalize(extraction)
        assert result.entities[0]["id"] == "python"
        assert len(conn.queries) == 0

    @pytest.mark.asyncio
    async def test_normalize_without_executor_skips_graph(self):
        normalizer = EntityNormalizer(
            embedding_generator=FakeEmbeddingGenerator(),
            executor=None,
        )
        extraction = ExtractionResult(
            entities=[{"id": "JS", "name": "JS", "type": "Technology"}],
            relationships=[],
        )
        result = await normalizer.normalize(extraction)
        assert result.entities[0]["id"] == "javascript"


class TestReservedNamespaceGuard:
    """Bug G: reserved names for the MIST system resolve to mist-identity."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "input_name,expected_canonical",
        [
            ("MIST", "mist-identity"),
            ("MIST.AI", "mist-identity"),
            ("MIST AI", "mist-identity"),
            ("mist", "mist-identity"),
            ("mist.ai", "mist-identity"),
            ("mist-ai", "mist-identity"),
            ("MIST-AI", "mist-identity"),
            ("the AI", "mist-identity"),
            ("the-ai", "mist-identity"),
            ("the assistant", "mist-identity"),
            ("The Assistant", "mist-identity"),
            ("the-assistant", "mist-identity"),
        ],
    )
    async def test_reserved_name_maps_to_mist_identity(self, input_name, expected_canonical):
        normalizer = EntityNormalizer(
            embedding_generator=FakeEmbeddingGenerator(),
            executor=None,
        )
        extraction = ExtractionResult(
            entities=[{"id": input_name, "name": input_name, "type": "Organization"}],
            relationships=[],
        )
        result = await normalizer.normalize(extraction)
        assert result.entities[0]["id"] == expected_canonical, (
            f"Expected {input_name} to canonicalize to {expected_canonical}, "
            f"got {result.entities[0]['id']}"
        )

    @pytest.mark.asyncio
    async def test_non_reserved_name_is_unchanged(self):
        """Sanity check: ordinary names do NOT get remapped."""
        normalizer = EntityNormalizer(
            embedding_generator=FakeEmbeddingGenerator(),
            executor=None,
        )
        extraction = ExtractionResult(
            entities=[{"id": "mistletoe", "name": "Mistletoe", "type": "Concept"}],
            relationships=[],
        )
        result = await normalizer.normalize(extraction)
        assert result.entities[0]["id"] == "mistletoe"

    @pytest.mark.asyncio
    async def test_reserved_name_logs_warning(self, caplog):
        import logging

        normalizer = EntityNormalizer(
            embedding_generator=FakeEmbeddingGenerator(),
            executor=None,
        )
        extraction = ExtractionResult(
            entities=[{"id": "MIST", "name": "MIST", "type": "Organization"}],
            relationships=[],
        )
        with caplog.at_level(logging.WARNING, logger="backend.knowledge.extraction.normalizer"):
            await normalizer.normalize(extraction)
        assert any(
            "reserved name" in r.message.lower() for r in caplog.records
        ), f"Expected reserved-name warning, got logs: {[r.message for r in caplog.records]}"


class TestReservedNameTypeRemap:
    """Cluster 1: reserved-name matches must override entity_type to MistIdentity.

    An LLM frequently labels 'MIST' or 'MIST.AI' as Organization. Cluster 1
    validator constraints require the mist-identity node to carry the
    MistIdentity label for IMPLEMENTED_WITH / MIST_HAS_CAPABILITY /
    MIST_HAS_TRAIT / MIST_HAS_PREFERENCE edges. The normalizer must rewrite
    both id AND type so the graph-writer produces a validator-compliant node.
    """

    @pytest.mark.asyncio
    async def test_mist_name_remaps_id_and_type(self):
        # Arrange
        normalizer = EntityNormalizer(
            embedding_generator=FakeEmbeddingGenerator(),
            executor=None,
        )
        extraction = ExtractionResult(
            entities=[{"id": "mist", "name": "MIST", "type": "Organization"}],
            relationships=[],
        )

        # Act
        result = await normalizer.normalize(extraction)

        # Assert
        assert result.entities[0]["id"] == "mist-identity"
        assert result.entities[0]["type"] == "MistIdentity"

    @pytest.mark.asyncio
    async def test_mist_dot_ai_remaps(self):
        # Arrange
        normalizer = EntityNormalizer(
            embedding_generator=FakeEmbeddingGenerator(),
            executor=None,
        )
        extraction = ExtractionResult(
            entities=[{"id": "mist-ai-1", "name": "MIST.AI", "type": "Organization"}],
            relationships=[],
        )

        # Act
        result = await normalizer.normalize(extraction)

        # Assert
        assert result.entities[0]["id"] == "mist-identity"
        assert result.entities[0]["type"] == "MistIdentity"

    @pytest.mark.asyncio
    async def test_the_ai_remaps(self):
        # Arrange
        normalizer = EntityNormalizer(
            embedding_generator=FakeEmbeddingGenerator(),
            executor=None,
        )
        extraction = ExtractionResult(
            entities=[{"id": "ai-agent", "name": "the AI", "type": "Concept"}],
            relationships=[],
        )

        # Act
        result = await normalizer.normalize(extraction)

        # Assert
        assert result.entities[0]["id"] == "mist-identity"
        assert result.entities[0]["type"] == "MistIdentity"

    @pytest.mark.asyncio
    async def test_relationships_forwarded_after_remap(self):
        # Arrange
        normalizer = EntityNormalizer(
            embedding_generator=FakeEmbeddingGenerator(),
            executor=None,
        )
        extraction = ExtractionResult(
            entities=[
                {"id": "mist", "name": "MIST", "type": "Organization"},
                {"id": "lancedb", "name": "LanceDB", "type": "Technology"},
            ],
            relationships=[
                {"source": "mist", "target": "lancedb", "type": "USES"},
            ],
        )

        # Act
        result = await normalizer.normalize(extraction)

        # Assert
        mist_entity = next(e for e in result.entities if e["id"] == "mist-identity")
        assert mist_entity["type"] == "MistIdentity"
        assert len(result.relationships) == 1
        assert result.relationships[0]["source"] == "mist-identity"
        assert result.relationships[0]["target"] == "lancedb"
        assert result.relationships[0]["type"] == "USES"

    @pytest.mark.asyncio
    async def test_non_reserved_entities_unchanged(self):
        # Arrange
        normalizer = EntityNormalizer(
            embedding_generator=FakeEmbeddingGenerator(),
            executor=None,
        )
        extraction = ExtractionResult(
            entities=[{"id": "python", "name": "Python", "type": "Technology"}],
            relationships=[],
        )

        # Act
        result = await normalizer.normalize(extraction)

        # Assert
        assert result.entities[0]["id"] == "python"
        assert result.entities[0]["type"] == "Technology"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "input_name",
        [
            pytest.param("Mist", id="title-case"),
            pytest.param("MIST", id="upper-case"),
            pytest.param("mist", id="lower-case"),
        ],
    )
    async def test_reserved_name_case_insensitive(self, input_name):
        # Arrange
        normalizer = EntityNormalizer(
            embedding_generator=FakeEmbeddingGenerator(),
            executor=None,
        )
        extraction = ExtractionResult(
            entities=[{"id": input_name, "name": input_name, "type": "Organization"}],
            relationships=[],
        )

        # Act
        result = await normalizer.normalize(extraction)

        # Assert
        assert result.entities[0]["id"] == "mist-identity"
        assert result.entities[0]["type"] == "MistIdentity"

    @pytest.mark.asyncio
    async def test_reserved_name_with_trailing_whitespace_remaps(self):
        # Arrange
        normalizer = EntityNormalizer(
            embedding_generator=FakeEmbeddingGenerator(),
            executor=None,
        )
        extraction = ExtractionResult(
            entities=[{"id": "whitespace-mist", "name": "  mist  ", "type": "Organization"}],
            relationships=[],
        )

        # Act
        result = await normalizer.normalize(extraction)

        # Assert
        assert result.entities[0]["id"] == "mist-identity"
        assert result.entities[0]["type"] == "MistIdentity"


class TestResolverPasses:
    """Task 7: resolver passes 3-5 (retired-type coercion, Metric compound-id, parent
    fallback). These run AFTER _canonicalize in the normal loop path (non-reserved, non-user
    entities).
    """

    @pytest.fixture
    def normalizer(self):
        return EntityNormalizer(
            embedding_generator=FakeEmbeddingGenerator(),
            executor=None,
        )

    @pytest.mark.asyncio
    async def test_retired_type_topic_coerced_to_concept(self, normalizer):
        # Pass 3: Topic -> Concept coercion.
        out = await normalizer.normalize(
            _result([{"id": "backend-work", "name": "backend work", "type": "Topic"}])
        )
        assert out.entities[0]["type"] == "Concept"

    @pytest.mark.asyncio
    async def test_retired_type_milestone_coerced_to_event_with_event_type(self, normalizer):
        # Pass 3: Milestone -> Event coercion + event_type sentinel.
        out = await normalizer.normalize(
            _result(
                [
                    {
                        "id": "house-closing",
                        "name": "house closing",
                        "type": "Milestone",
                        "properties": {},
                    }
                ]
            )
        )
        assert out.entities[0]["type"] == "Event"
        assert out.entities[0]["properties"]["event_type"] == "milestone"

    @pytest.mark.asyncio
    async def test_metric_id_canonicalized_from_props(self, normalizer):
        # Pass 4: Metric compound-id from value + unit properties.
        out = await normalizer.normalize(
            _result(
                [
                    {
                        "id": "requests-per-second-12000",
                        "name": "12000 requests per second",
                        "type": "Metric",
                        "properties": {"value": "12000", "unit": "requests-per-second"},
                    }
                ]
            )
        )
        assert out.entities[0]["id"] == "12000-requests-per-second"

    @pytest.mark.asyncio
    async def test_unknown_abstract_type_falls_back_to_abstraction(self, normalizer):
        # Pass 5: unknown type -> Abstraction fallback.
        out = await normalizer.normalize(
            _result([{"id": "thing", "name": "thing", "type": "NotARealType"}])
        )
        assert out.entities[0]["type"] == "Abstraction"

    @pytest.mark.asyncio
    async def test_metric_relationship_target_remapped_to_post_resolver_id(self, normalizer):
        # Verify id_map captures the POST-resolver Metric id so relationship
        # source/target remapping uses the final compound id, not the pre-resolver
        # canonical_id.
        out = await normalizer.normalize(
            _result(
                entities=[
                    {
                        "id": "speed-metric",
                        "name": "12000 requests per second",
                        "type": "Metric",
                        "properties": {"value": "12000", "unit": "requests-per-second"},
                    },
                    {"id": "myapp", "name": "MyApp", "type": "Project"},
                ],
                relationships=[{"source": "myapp", "target": "speed-metric", "type": "HAS_METRIC"}],
            )
        )
        # The relationship target must point to the compound id, not the original.
        assert out.relationships[0]["target"] == "12000-requests-per-second"

    @pytest.mark.asyncio
    async def test_milestone_coercion_does_not_overwrite_existing_event_type(self, normalizer):
        # event_type already set -> setdefault must not overwrite it.
        out = await normalizer.normalize(
            _result(
                [
                    {
                        "id": "sprint-end",
                        "name": "sprint end",
                        "type": "Milestone",
                        "properties": {"event_type": "deadline"},
                    }
                ]
            )
        )
        assert out.entities[0]["type"] == "Event"
        assert out.entities[0]["properties"]["event_type"] == "deadline"

    @pytest.mark.asyncio
    async def test_metric_missing_unit_leaves_id_unchanged(self, normalizer):
        # Pass 4 guard: no unit -> string fallback, not compound-id rewrite.
        # "raw-metric" has no numeric token, so canonical_metric_id_from_id returns it unchanged.
        out = await normalizer.normalize(
            _result(
                [
                    {
                        "id": "raw-metric",
                        "name": "raw metric",
                        "type": "Metric",
                        "properties": {"value": "42"},
                    }
                ]
            )
        )
        # id should fall through to whatever _canonicalize produced, not compound id.
        assert out.entities[0]["type"] == "Metric"
        assert out.entities[0]["id"] != "42-"  # no blank-unit compound id

    @pytest.mark.asyncio
    async def test_metric_no_props_string_fallback_moves_number_to_front(self, normalizer):
        # Pass 4 string fallback: Metric with no value/unit props, bare id with numeric
        # token at the end is reordered to value-first by canonical_metric_id_from_id.
        # The entity carries NO name field so entity_name falls back to old_id, which
        # means _canonicalize receives "requests-per-second-12000" (already-hyphenated,
        # no version-strip match) and canonical_metric_id_from_id fires on the result.
        out = await normalizer.normalize(
            _result(
                [
                    {
                        "id": "requests-per-second-12000",
                        "type": "Metric",
                    }
                ]
            )
        )
        assert out.entities[0]["id"] == "12000-requests-per-second"

    @pytest.mark.asyncio
    async def test_valid_type_not_mutated_by_pass5(self, normalizer):
        # Pass 5 guard: known type must NOT be changed to Abstraction.
        out = await normalizer.normalize(
            _result([{"id": "python", "name": "Python", "type": "Technology"}])
        )
        assert out.entities[0]["type"] == "Technology"


class TestCanonicalRegistry:
    """Task 8: CANONICAL_REGISTRY pass overrides id and type, short-circuits graph dedup."""

    @pytest.mark.asyncio
    async def test_registry_overrides_id_and_type(self, normalizer_no_graph, monkeypatch):
        # Key is the CANONICAL id: "the graph" canonicalizes to "the-graph".
        monkeypatch.setitem(
            EntityNormalizer.CANONICAL_REGISTRY, "the-graph", ("neo4j", "Technology")
        )
        out = await normalizer_no_graph.normalize(
            _result([{"id": "x", "name": "the graph", "type": "Concept"}])
        )
        assert out.entities[0]["id"] == "neo4j"
        assert out.entities[0]["type"] == "Technology"


class TestNoGraphIdentityQuery:
    """R1.1d: After the strip, normalize() must not issue any graph identity queries."""

    @pytest.mark.asyncio
    async def test_normalize_issues_no_graph_identity_query(self):
        # After the strip, normalize() does pure string/registry/resolver canonicalization
        # -- it must NOT query the graph for identity (no ANN, no entity MATCH).
        conn = FakeNeo4jConnection()
        executor = FakeGraphExecutor(connection=conn)
        normalizer = EntityNormalizer(
            embedding_generator=FakeEmbeddingGenerator(),
            executor=executor,
        )
        extraction = ExtractionResult(
            entities=[{"id": "Python", "name": "Python", "type": "Technology"}],
            relationships=[],
        )

        await normalizer.normalize(extraction)

        assert not any("db.index.vector.queryNodes" in q for q, _ in conn.queries), conn.queries
        assert not any("MATCH (e:__Entity__)" in q for q, _ in conn.queries), conn.queries
