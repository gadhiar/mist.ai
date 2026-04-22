"""Unit tests for QueryClassifier.

Verifies that the heuristic regex classifier routes the 31 labeled queries
from the intent_queries fixture to the expected intent type, and that the
classifier's output always satisfies basic structural invariants.
"""

import pytest

from backend.knowledge.config import QueryIntentConfig
from backend.knowledge.retrieval.query_classifier import QueryClassifier
from tests.unit.knowledge.fixtures.intent_queries import LABELED_QUERIES

MODULE = "backend.knowledge.retrieval.query_classifier"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Queries where the heuristic classifier diverges from the labeled intent.
# These are known accuracy gaps, not test bugs.
_KNOWN_MISCLASSIFIED = {
    "What did I say about Rust last week?",  # labeled factual, classified relational ("I" triggers relational)
    "What problems have I had with the tools on my current project?",  # labeled hybrid, "current" triggers live
    "Are there any open pull requests on the mist.ai repo?",  # labeled live, "open pull requests" triggers hybrid
}


def _mark_xfail(queries: list) -> list:
    """Wrap known-misclassified queries with pytest.param(..., marks=xfail)."""
    out = []
    for q in queries:
        if q.query in _KNOWN_MISCLASSIFIED:
            out.append(
                pytest.param(
                    q,
                    marks=pytest.mark.xfail(
                        reason="Heuristic classifier accuracy gap",
                        strict=False,
                    ),
                )
            )
        else:
            out.append(q)
    return out


FACTUAL_QUERIES = _mark_xfail([q for q in LABELED_QUERIES if q.intent == "factual"])
RELATIONAL_QUERIES = _mark_xfail([q for q in LABELED_QUERIES if q.intent == "relational"])
HYBRID_QUERIES = _mark_xfail([q for q in LABELED_QUERIES if q.intent == "hybrid"])
LIVE_QUERIES = _mark_xfail([q for q in LABELED_QUERIES if q.intent == "live"])

VALID_INTENTS = {"factual", "relational", "hybrid", "live", "historical", "identity"}
VALID_STORES = {
    "factual": ("vector",),
    "relational": ("graph",),
    # Phase 9 (ADR-010): hybrid now includes vault sidecar as a third leg.
    "hybrid": ("vector", "graph", "vault"),
    "live": ("mcp",),
    "identity": ("mist",),
    "historical": ("vault",),
}


# ---------------------------------------------------------------------------
# Intent classification -- parametrized over each labeled group
# ---------------------------------------------------------------------------


class TestIntentClassification:
    """Verifies that the classifier assigns the correct intent to each labeled query."""

    @pytest.fixture(autouse=True)
    def classifier(self):
        self._classifier = QueryClassifier(config=QueryIntentConfig())

    @pytest.mark.parametrize(
        "labeled",
        FACTUAL_QUERIES,
        ids=lambda q: q.query[:40],
    )
    def test_factual_queries(self, labeled):
        # Arrange
        query = labeled.query

        # Act
        result = self._classifier.classify(query)

        # Assert
        assert (
            result.intent == "factual"
        ), f"Query {query!r} expected 'factual', got {result.intent!r}"

    @pytest.mark.parametrize(
        "labeled",
        RELATIONAL_QUERIES,
        ids=lambda q: q.query[:40],
    )
    def test_relational_queries(self, labeled):
        # Arrange
        query = labeled.query

        # Act
        result = self._classifier.classify(query)

        # Assert
        assert (
            result.intent == "relational"
        ), f"Query {query!r} expected 'relational', got {result.intent!r}"

    @pytest.mark.parametrize(
        "labeled",
        HYBRID_QUERIES,
        ids=lambda q: q.query[:40],
    )
    def test_hybrid_queries(self, labeled):
        # Arrange
        query = labeled.query

        # Act
        result = self._classifier.classify(query)

        # Assert
        assert (
            result.intent == "hybrid"
        ), f"Query {query!r} expected 'hybrid', got {result.intent!r}"

    @pytest.mark.parametrize(
        "labeled",
        LIVE_QUERIES,
        ids=lambda q: q.query[:40],
    )
    def test_live_queries(self, labeled):
        # Arrange
        query = labeled.query

        # Act
        result = self._classifier.classify(query)

        # Assert
        assert result.intent == "live", f"Query {query!r} expected 'live', got {result.intent!r}"


# ---------------------------------------------------------------------------
# Structural invariants -- always hold regardless of query content
# ---------------------------------------------------------------------------


class TestClassifierEdgeCases:
    """Verifies structural invariants for edge-case inputs."""

    @pytest.fixture(autouse=True)
    def classifier(self):
        self._classifier = QueryClassifier(config=QueryIntentConfig())

    def test_empty_query_returns_valid_result(self):
        # Arrange
        query = ""

        # Act
        result = self._classifier.classify(query)

        # Assert
        assert result.intent in VALID_INTENTS
        assert result.suggested_stores == VALID_STORES[result.intent]
        assert 0.0 <= result.confidence <= 1.0

    def test_single_word_query_returns_valid_result(self):
        # Arrange
        query = "Python"

        # Act
        result = self._classifier.classify(query)

        # Assert
        assert result.intent in VALID_INTENTS
        assert result.suggested_stores == VALID_STORES[result.intent]
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.parametrize(
        "labeled",
        LABELED_QUERIES,
        ids=lambda q: q.query[:40],
    )
    def test_confidence_always_in_valid_range(self, labeled):
        # Arrange
        query = labeled.query

        # Act
        result = self._classifier.classify(query)

        # Assert
        assert (
            0.0 <= result.confidence <= 1.0
        ), f"Query {query!r} produced out-of-range confidence {result.confidence}"

    @pytest.mark.parametrize(
        "labeled",
        LABELED_QUERIES,
        ids=lambda q: q.query[:40],
    )
    def test_suggested_stores_match_intent(self, labeled):
        # Arrange
        query = labeled.query

        # Act
        result = self._classifier.classify(query)

        # Assert
        assert result.suggested_stores == VALID_STORES[result.intent], (
            f"Query {query!r}: intent={result.intent!r} but "
            f"suggested_stores={result.suggested_stores!r}"
        )


# ---------------------------------------------------------------------------
# Cluster 3: identity intent -- queries about MIST's own identity / traits
# ---------------------------------------------------------------------------


class TestIdentityIntent:
    """Cluster 3: queries about MIST's identity route to identity intent."""

    @pytest.fixture
    def classifier(self):
        from backend.knowledge.retrieval.query_classifier import QueryClassifier

        return QueryClassifier()

    @pytest.mark.parametrize(
        "query",
        [
            "what's your name?",
            "who are you?",
            "tell me about yourself",
            "what are your preferences?",
            "do you like emojis?",
            "are you MIST?",
            "what can you do?",
            "describe your personality",
            "what are your traits?",
            "what capabilities do you have?",
        ],
    )
    def test_identity_query_classifies_as_identity(self, classifier, query):
        result = classifier.classify(query)
        assert result.intent == "identity", f"Query {query!r} classified as {result.intent}"

    def test_user_fact_query_is_not_identity(self, classifier):
        result = classifier.classify("I use Python for data pipelines")
        assert result.intent != "identity"

    def test_document_query_is_not_identity(self, classifier):
        result = classifier.classify("what are the benefits of Neo4j?")
        assert result.intent != "identity"

    def test_live_query_is_not_identity(self, classifier):
        result = classifier.classify("what's the status of MIS-104 right now?")
        assert result.intent != "identity"

    def test_identity_intent_suggested_stores(self, classifier):
        """Identity intent should have a 'mist' suggested store."""
        result = classifier.classify("who are you?")
        assert result.intent == "identity"
        assert "mist" in result.suggested_stores

    @pytest.mark.parametrize(
        "query",
        [
            "do you have any idea",
            "do you have the latest data",
            "do you have access to my calendar",
        ],
    )
    def test_generic_have_query_not_identity(self, classifier, query):
        """Regression guard: 'do you have X' should not misroute to identity."""
        result = classifier.classify(query)
        assert (
            result.intent != "identity"
        ), f"Query {query!r} misclassified as identity (generic 'have' pattern)"

    @pytest.mark.parametrize(
        "query",
        [
            "do you have any preferences",
            "do you have any traits",
            "do you have any capabilities",
        ],
    )
    def test_have_with_identity_noun_still_identity(self, classifier, query):
        """Positive: 'do you have <identity-noun>' still classifies as identity."""
        result = classifier.classify(query)
        assert result.intent == "identity"


# ---------------------------------------------------------------------------
# Phase 9: historical intent -- routes prose recall queries to vault sidecar
# ---------------------------------------------------------------------------


class TestHistoricalIntent:
    """ADR-010 Phase 9: 'what did we discuss'-class queries route to the vault
    sidecar so prose conversation history surfaces alongside graph facts.
    """

    @pytest.fixture
    def classifier(self):
        return QueryClassifier()

    @pytest.mark.parametrize(
        "query",
        [
            "what did we discuss about persistent memory?",
            "what did we talk about yesterday?",
            "remember when we covered the vault architecture?",
            "last time we talked about Neo4j",
            "in our last session, what did you suggest?",
            "previously we agreed on the schema",
            "earlier today we set up the filewatcher",
            "we talked about RRF merge weights",
            "what was I saying about the LLM provider?",
            "what did we cover in the previous conversation?",
        ],
    )
    def test_historical_query_classifies_as_historical(self, classifier, query):
        result = classifier.classify(query)
        assert (
            result.intent == "historical"
        ), f"Query {query!r} classified as {result.intent} (expected historical)"

    def test_historical_intent_suggested_stores(self, classifier):
        result = classifier.classify("what did we discuss about Python?")
        assert result.intent == "historical"
        assert result.suggested_stores == ("vault",)

    @pytest.mark.parametrize(
        "query",
        [
            "what is Python?",  # factual
            "what do I know about Rust?",  # relational
            "what's your name?",  # identity
            "what's the status of MIS-104?",  # live
        ],
    )
    def test_non_historical_queries_not_historical(self, classifier, query):
        result = classifier.classify(query)
        assert (
            result.intent != "historical"
        ), f"Query {query!r} misrouted to historical (got {result.intent})"

    def test_historical_priority_below_identity(self, classifier):
        # "what did we discuss about your preferences" overlaps both intents;
        # identity wins because it has priority 0 (must never misroute).
        result = classifier.classify("what did we discuss about your preferences")
        assert result.intent == "identity"

    def test_hybrid_intent_includes_vault_store(self, classifier):
        # Phase 9: hybrid is now a three-way merge, so its suggested_stores
        # tuple must include "vault" alongside "vector" and "graph".
        # "describe my projects" matches both factual ("describe") and
        # relational ("my") so the classifier returns hybrid.
        result = classifier.classify("describe my projects in detail")
        assert result.intent == "hybrid"
        assert "vault" in result.suggested_stores
        assert "vector" in result.suggested_stores
        assert "graph" in result.suggested_stores
