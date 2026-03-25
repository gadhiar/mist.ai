"""Unit tests for KnowledgeRouter.classify() -- routing destination logic.

Tests the four-step decision cascade:
  1. Discard empty/filler content
  2. Source-type fast path (MCP, document, system)
  3. Graph-worthy signal detection (conversational content)
  4. Fallback to vector-only

No I/O boundaries exist in this module -- KnowledgeRouter is pure regex
and heuristic logic. No fakes or mocks required.
"""

from __future__ import annotations

import pytest

from backend.knowledge.models import ContentSourceType, RoutingDestination
from backend.knowledge.routing import KnowledgeRouter

# ---------------------------------------------------------------------------
# Shared router instance -- KnowledgeRouter is stateless after init.
# ---------------------------------------------------------------------------

MODULE = "backend.knowledge.routing"


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestDiscardDetection:
    """Step 1: empty or filler content is discarded before any other check."""

    def test_empty_string_is_discarded(self):
        # Arrange
        router = KnowledgeRouter()

        # Act
        result = router.classify("", source_type=ContentSourceType.CONVERSATION.value)

        # Assert
        assert result.destination == RoutingDestination.DISCARD.value

    def test_whitespace_only_is_discarded(self):
        # Arrange
        router = KnowledgeRouter()

        # Act
        result = router.classify("   \t\n  ", source_type=ContentSourceType.CONVERSATION.value)

        # Assert
        assert result.destination == RoutingDestination.DISCARD.value

    def test_whitespace_only_has_full_confidence(self):
        # Arrange
        router = KnowledgeRouter()

        # Act
        result = router.classify("", source_type=ContentSourceType.CONVERSATION.value)

        # Assert
        assert result.confidence == 1.0

    @pytest.mark.parametrize(
        "acknowledgment",
        [
            pytest.param("ok", id="bare-ok"),
            pytest.param("thanks", id="bare-thanks"),
            pytest.param("yes", id="bare-yes"),
            pytest.param("got it", id="got-it"),
            pytest.param("yep", id="yep"),
            pytest.param("sure", id="sure"),
            pytest.param("noted", id="noted"),
            pytest.param("sounds good", id="sounds-good"),
            pytest.param("understood", id="understood"),
            pytest.param("will do", id="will-do"),
            pytest.param("mhm", id="mhm"),
        ],
    )
    def test_bare_acknowledgment_is_discarded(self, acknowledgment):
        # Arrange
        router = KnowledgeRouter()

        # Act
        result = router.classify(acknowledgment, source_type=ContentSourceType.CONVERSATION.value)

        # Assert
        assert result.destination == RoutingDestination.DISCARD.value

    @pytest.mark.parametrize(
        "text",
        [
            pytest.param("okay!", id="okay-exclamation"),
            pytest.param("thanks.", id="thanks-period"),
            pytest.param("yes?", id="yes-question"),
            pytest.param("OK", id="ok-uppercase"),
            pytest.param("Thanks", id="thanks-titlecase"),
        ],
    )
    def test_acknowledgment_with_punctuation_is_discarded(self, text):
        # Arrange
        router = KnowledgeRouter()

        # Act
        result = router.classify(text, source_type=ContentSourceType.CONVERSATION.value)

        # Assert
        assert result.destination == RoutingDestination.DISCARD.value

    def test_real_content_is_not_discarded(self):
        # Arrange
        router = KnowledgeRouter()
        content = "I prefer Python over JavaScript for backend services."

        # Act
        result = router.classify(content, source_type=ContentSourceType.CONVERSATION.value)

        # Assert
        assert result.destination != RoutingDestination.DISCARD.value

    def test_multi_word_real_content_is_not_discarded(self):
        # Arrange
        router = KnowledgeRouter()
        content = "I work at Anthropic on the alignment team."

        # Act
        result = router.classify(content, source_type=ContentSourceType.CONVERSATION.value)

        # Assert
        assert result.destination != RoutingDestination.DISCARD.value


class TestSourceTypeFastPath:
    """Step 2: certain source types bypass signal detection entirely."""

    def test_system_message_is_discarded(self):
        # Arrange
        router = KnowledgeRouter()
        content = "System initialized. Ready to process requests."

        # Act
        result = router.classify(content, source_type=ContentSourceType.SYSTEM.value)

        # Assert
        assert result.destination == RoutingDestination.DISCARD.value

    def test_system_message_discard_has_high_confidence(self):
        # Arrange
        router = KnowledgeRouter()

        # Act
        result = router.classify(
            "boot sequence complete", source_type=ContentSourceType.SYSTEM.value
        )

        # Assert
        assert result.confidence >= 0.90

    def test_transient_mcp_output_routes_to_mcp_only(self):
        # Arrange
        router = KnowledgeRouter()

        # Act
        result = router.classify(
            "tool result data",
            source_type=ContentSourceType.MCP_TOOL_OUTPUT.value,
            metadata={"transient": True},
        )

        # Assert
        assert result.destination == RoutingDestination.MCP_ONLY.value

    def test_re_invocable_mcp_output_routes_to_mcp_only(self):
        # Arrange
        router = KnowledgeRouter()

        # Act
        result = router.classify(
            "tool result data",
            source_type=ContentSourceType.MCP_TOOL_OUTPUT.value,
            metadata={"re_invocable": True},
        )

        # Assert
        assert result.destination == RoutingDestination.MCP_ONLY.value

    def test_persistent_mcp_output_routes_to_vector_only(self):
        # Arrange
        router = KnowledgeRouter()

        # Act
        result = router.classify(
            "tool result data",
            source_type=ContentSourceType.MCP_TOOL_OUTPUT.value,
            metadata={},
        )

        # Assert
        assert result.destination == RoutingDestination.VECTOR_ONLY.value

    def test_document_chunk_routes_to_vector_only(self):
        # Arrange
        router = KnowledgeRouter()
        content = "The mitochondria is the powerhouse of the cell."

        # Act
        result = router.classify(content, source_type=ContentSourceType.DOCUMENT_CHUNK.value)

        # Assert
        assert result.destination == RoutingDestination.VECTOR_ONLY.value

    def test_reference_lookup_routes_to_vector_only(self):
        # Arrange
        router = KnowledgeRouter()
        content = "FastAPI is a modern, fast web framework for Python."

        # Act
        result = router.classify(content, source_type=ContentSourceType.REFERENCE_LOOKUP.value)

        # Assert
        assert result.destination == RoutingDestination.VECTOR_ONLY.value

    def test_mcp_output_without_metadata_param_defaults_to_empty_dict(self):
        # Arrange
        router = KnowledgeRouter()

        # Act -- metadata omitted entirely
        result = router.classify(
            "tool result",
            source_type=ContentSourceType.MCP_TOOL_OUTPUT.value,
        )

        # Assert -- no transient/re_invocable flags, so persistent path applies
        assert result.destination == RoutingDestination.VECTOR_ONLY.value


class TestGraphSignalDetection:
    """Step 3: conversational content with graph-worthy signals routes to GRAPH_AND_VECTOR."""

    @pytest.mark.parametrize(
        "content",
        [
            pytest.param("I prefer Python over Java for this kind of work.", id="prefer"),
            pytest.param("I really like working with graph databases.", id="i-really-like"),
            pytest.param("I love how Neo4j handles traversals.", id="i-love"),
            pytest.param("I enjoy pair programming sessions.", id="i-enjoy"),
            pytest.param("I hate context switching mid-task.", id="i-hate"),
            pytest.param("My favorite editor is Neovim.", id="my-favorite"),
        ],
    )
    def test_preference_signal_routes_to_graph_and_vector(self, content):
        # Arrange
        router = KnowledgeRouter()

        # Act
        result = router.classify(content, source_type=ContentSourceType.CONVERSATION.value)

        # Assert
        assert result.destination == RoutingDestination.GRAPH_AND_VECTOR.value

    def test_preference_signal_reason_identifies_category(self):
        # Arrange
        router = KnowledgeRouter()
        content = "I prefer dark mode in all my editors."

        # Act
        result = router.classify(content, source_type=ContentSourceType.CONVERSATION.value)

        # Assert
        assert "preference" in result.reason

    @pytest.mark.parametrize(
        "content",
        [
            pytest.param("Actually, it is spelled differently.", id="actually-it"),
            pytest.param("Actually, that version was released in 2019.", id="actually-that"),
            pytest.param("No, it's the other way around.", id="no-its"),
            pytest.param("I meant the staging environment, not production.", id="i-meant"),
        ],
    )
    def test_correction_signal_routes_to_graph_and_vector(self, content):
        # Arrange
        router = KnowledgeRouter()

        # Act
        result = router.classify(content, source_type=ContentSourceType.CONVERSATION.value)

        # Assert
        assert result.destination == RoutingDestination.GRAPH_AND_VECTOR.value

    @pytest.mark.parametrize(
        "content",
        [
            pytest.param("My colleague Alice handles the deployment pipeline.", id="colleague"),
            pytest.param("My manager signed off on the architecture.", id="manager"),
            pytest.param("My wife suggested the project name.", id="wife"),
            pytest.param("My team is split across two time zones.", id="team"),
            pytest.param("My friend Bob introduced me to Rust.", id="friend"),
        ],
    )
    def test_relationship_signal_routes_to_graph_and_vector(self, content):
        # Arrange
        router = KnowledgeRouter()

        # Act
        result = router.classify(content, source_type=ContentSourceType.CONVERSATION.value)

        # Assert
        assert result.destination == RoutingDestination.GRAPH_AND_VECTOR.value

    @pytest.mark.parametrize(
        "content",
        [
            pytest.param("I'm good at system design and distributed systems.", id="good-at"),
            pytest.param("I am experienced at writing Cypher queries.", id="experienced-at"),
            pytest.param("I know how to configure Ollama for local inference.", id="i-know"),
            pytest.param("I use Docker for all service isolation.", id="i-use"),
            pytest.param("I work with Neo4j on a daily basis.", id="work-with"),
            pytest.param("I specialize in graph-based knowledge systems.", id="specialize-in"),
        ],
    )
    def test_skill_signal_routes_to_graph_and_vector(self, content):
        # Arrange
        router = KnowledgeRouter()

        # Act
        result = router.classify(content, source_type=ContentSourceType.CONVERSATION.value)

        # Assert
        assert result.destination == RoutingDestination.GRAPH_AND_VECTOR.value

    @pytest.mark.parametrize(
        "content",
        [
            pytest.param("The authentication service works at the edge layer.", id="works-at"),
            pytest.param("This module is part of the extraction pipeline.", id="part-of"),
            pytest.param("The routing decision is related to the graph store.", id="related-to"),
            pytest.param("This feature depends on the embedding service.", id="depends-on"),
            pytest.param("The config object belongs to the factory module.", id="belongs-to"),
            pytest.param("The CLI is built with Click and Typer.", id="built-with"),
        ],
    )
    def test_structural_signal_routes_to_graph_and_vector(self, content):
        # Arrange
        router = KnowledgeRouter()

        # Act
        result = router.classify(content, source_type=ContentSourceType.CONVERSATION.value)

        # Assert
        assert result.destination == RoutingDestination.GRAPH_AND_VECTOR.value

    def test_multiple_signal_categories_boost_confidence(self):
        # Arrange -- both preference and structural signals present
        router = KnowledgeRouter()
        content = "I prefer the extraction module because it is part of the core pipeline."

        # Act
        result = router.classify(content, source_type=ContentSourceType.CONVERSATION.value)

        # Assert
        assert result.destination == RoutingDestination.GRAPH_AND_VECTOR.value
        # Multi-category match adds 0.05 boost -- confidence must reflect it
        assert result.confidence > 0.85


class TestEntityDensity:
    """Entity density heuristic: high capitalization density boosts graph routing."""

    def test_high_capitalization_density_routes_to_graph_and_vector(self):
        # Arrange -- dense proper nouns, no explicit signal patterns
        router = KnowledgeRouter()
        # Words: "Alice", "Bob", "Carol", "Toronto", "Anthropic", "Mozilla" are all
        # capitalized non-common words at non-sentence-start positions.
        # This gives 5 entity hits out of 8 total words => density ~0.625, well above 0.15.
        content = "meeting between Alice Bob Carol from Anthropic Mozilla Toronto"

        # Act
        result = router.classify(content, source_type=ContentSourceType.CONVERSATION.value)

        # Assert
        assert result.destination == RoutingDestination.GRAPH_AND_VECTOR.value
        assert "entity_density" in result.reason

    def test_low_capitalization_density_does_not_force_graph_routing(self):
        # Arrange -- normal lowercase sentence, no signal words, no capitalized proper nouns
        router = KnowledgeRouter()
        content = "the quick brown fox jumped over the lazy dog near the river"

        # Act
        result = router.classify(content, source_type=ContentSourceType.CONVERSATION.value)

        # Assert -- entity_density heuristic should NOT fire
        assert "entity_density" not in (result.reason or "")

    def test_sentence_starter_capitals_do_not_count_as_entities(self):
        # Arrange -- first word capitalized as sentence start, rest lowercase
        router = KnowledgeRouter()
        # Only "The" is capitalized (word index 0, excluded), rest are lowercase common words.
        content = "The cat sat on the mat and the dog ran by the tree"

        # Act
        result = router.classify(content, source_type=ContentSourceType.CONVERSATION.value)

        # Assert -- no entity density boost
        assert "entity_density" not in (result.reason or "")


class TestFallbackRouting:
    """Step 4: content that passes all earlier gates falls back to VECTOR_ONLY."""

    def test_plain_factual_statement_falls_back_to_vector_only(self):
        # Arrange -- no signals, no source fast path, not empty
        router = KnowledgeRouter()
        content = "the python programming language was created in the late eighties"

        # Act
        result = router.classify(content, source_type=ContentSourceType.CONVERSATION.value)

        # Assert
        assert result.destination == RoutingDestination.VECTOR_ONLY.value

    def test_fallback_reason_indicates_no_signals(self):
        # Arrange
        router = KnowledgeRouter()
        content = "water freezes at zero degrees celsius"

        # Act
        result = router.classify(content, source_type=ContentSourceType.CONVERSATION.value)

        # Assert
        assert result.destination == RoutingDestination.VECTOR_ONLY.value
        assert "no graph-worthy signals" in result.reason

    def test_short_fallback_content_has_lower_confidence_than_long(self):
        # Arrange -- short content (<20 words) gets 0.5, long gets 0.6
        router = KnowledgeRouter()
        short_content = "water boils at one hundred degrees"
        long_content = (
            "water boils at one hundred degrees celsius at standard atmospheric pressure "
            "which is defined as one atmosphere or one hundred and one kilopascals"
        )

        # Act
        short_result = router.classify(
            short_content, source_type=ContentSourceType.CONVERSATION.value
        )
        long_result = router.classify(
            long_content, source_type=ContentSourceType.CONVERSATION.value
        )

        # Assert
        assert short_result.confidence < long_result.confidence
        assert short_result.confidence == 0.5
        assert long_result.confidence == 0.6

    def test_unknown_source_type_falls_through_to_signal_detection(self):
        # Arrange -- source type not in fast-path list, so signal detection runs
        router = KnowledgeRouter()
        # This content has no graph signal, so it should reach the fallback.
        content = "the log output was written to disk"

        # Act -- pass an unknown source type string
        result = router.classify(content, source_type="unknown_type")

        # Assert -- reached fallback, not a source-type fast-path destination
        assert result.destination == RoutingDestination.VECTOR_ONLY.value

    def test_routing_decision_is_a_dataclass_with_required_fields(self):
        # Arrange
        router = KnowledgeRouter()

        # Act
        result = router.classify(
            "some plain text here", source_type=ContentSourceType.CONVERSATION.value
        )

        # Assert -- all three required fields are present and non-empty
        assert result.destination is not None
        assert result.reason is not None
        assert len(result.reason) > 0
        assert 0.0 <= result.confidence <= 1.0
