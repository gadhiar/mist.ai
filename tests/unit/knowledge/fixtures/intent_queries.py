"""Labeled query fixtures for intent classification testing.

Each query is tagged with one of four intent types that determine
which retrieval backend should handle the query:

    factual (vector store)
        Questions about document contents, definitions, reference
        lookups, stored recall.  Indicators: "what does X say",
        "what did I say about", "describe", "summarize".

    relational (graph)
        Questions about entity relationships and graph traversal.
        Indicators: first-person + relationship verbs ("what do I
        use"), "how is X related to Y", "who", dependency queries.

    hybrid (both vector + graph)
        Mixed queries that need both factual recall from documents
        and relationship understanding from the graph.

    live (MCP tool)
        Current / real-time state queries.  Indicators: "status of
        ticket", "what PRs are open", "right now", "running",
        "current", references to external tool state.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LabeledQuery:
    """A single query labeled with its expected intent type."""

    query: str
    intent: str  # "factual" | "relational" | "hybrid" | "live"


VALID_INTENTS: frozenset[str] = frozenset({"factual", "relational", "hybrid", "live"})

LABELED_QUERIES: list[LabeledQuery] = [
    # ------------------------------------------------------------------
    # FACTUAL -- vector store retrieval
    # ------------------------------------------------------------------
    LabeledQuery(
        "What is the observer pattern?",
        "factual",
    ),
    LabeledQuery(
        "Describe how Neo4j indexes work",
        "factual",
    ),
    LabeledQuery(
        "What does the extraction pipeline do?",
        "factual",
    ),
    LabeledQuery(
        "Summarize what I said about microservices",
        "factual",
    ),
    LabeledQuery(
        "What are the benefits of event sourcing?",
        "factual",
    ),
    LabeledQuery(
        "When was the knowledge graph ontology created?",
        "factual",
    ),
    # Edge case: temporal marker ("last week") looks like a live query,
    # but the user is asking to recall stored content they previously
    # dictated -- this is document retrieval, not real-time state.
    LabeledQuery(
        "What did I say about Rust last week?",
        "factual",
    ),
    # ------------------------------------------------------------------
    # RELATIONAL -- graph traversal
    # ------------------------------------------------------------------
    LabeledQuery(
        "What technologies do I use?",
        "relational",
    ),
    LabeledQuery(
        "How is FastAPI related to Python?",
        "relational",
    ),
    LabeledQuery(
        "Who do I work with at my company?",
        "relational",
    ),
    LabeledQuery(
        "What projects am I working on?",
        "relational",
    ),
    LabeledQuery(
        "What am I an expert in?",
        "relational",
    ),
    LabeledQuery(
        "What skills does Sarah have?",
        "relational",
    ),
    # Edge case: no first-person pronoun, but the query is purely about
    # dependency edges between technology nodes -- pure graph traversal.
    LabeledQuery(
        "What technologies depend on PyTorch?",
        "relational",
    ),
    # Edge case: combines KNOWS_PERSON + USES relationship edges, but
    # the answer is entirely derivable from the graph without needing
    # any document content.
    LabeledQuery(
        "Do I know anyone who uses Kubernetes?",
        "relational",
    ),
    # ------------------------------------------------------------------
    # HYBRID -- vector store + graph
    # ------------------------------------------------------------------
    LabeledQuery(
        "What have I learned about the technologies I use?",
        "hybrid",
    ),
    LabeledQuery(
        "Compare the frameworks I know for building APIs",
        "hybrid",
    ),
    LabeledQuery(
        "What problems have I had with the tools on my " "current project?",
        "hybrid",
    ),
    LabeledQuery(
        "Summarize my goals and the skills they require",
        "hybrid",
    ),
    LabeledQuery(
        "What did my colleague say about the database we use?",
        "hybrid",
    ),
    LabeledQuery(
        "What concepts should I study based on my learning " "goals?",
        "hybrid",
    ),
    LabeledQuery(
        "How does my experience with Python relate to my " "current project?",
        "hybrid",
    ),
    LabeledQuery(
        "What do I know about the architecture of projects " "I contribute to?",
        "hybrid",
    ),
    # ------------------------------------------------------------------
    # LIVE -- MCP tool invocation
    # ------------------------------------------------------------------
    LabeledQuery(
        "What is the status of ticket MIS-58?",
        "live",
    ),
    LabeledQuery(
        "Are there any open pull requests on the mist.ai repo?",
        "live",
    ),
    LabeledQuery(
        "What issues were closed this sprint?",
        "live",
    ),
    LabeledQuery(
        "Is the backend server running right now?",
        "live",
    ),
    LabeledQuery(
        "What is the current Git branch?",
        "live",
    ),
    LabeledQuery(
        "How much GPU memory is available?",
        "live",
    ),
    LabeledQuery(
        "Show me the latest commit on feat/frontend",
        "live",
    ),
    LabeledQuery(
        "What tasks are assigned to me in Linear?",
        "live",
    ),
]


def get_queries_by_intent(intent: str) -> list[str]:
    """Return all query strings for a given intent type.

    Parameters
    ----------
    intent:
        One of ``VALID_INTENTS``.

    Raises:
    ------
    ValueError
        If *intent* is not a recognized intent type.
    """
    if intent not in VALID_INTENTS:
        raise ValueError(f"Unknown intent {intent!r}. " f"Valid: {sorted(VALID_INTENTS)}")
    return [q.query for q in LABELED_QUERIES if q.intent == intent]
