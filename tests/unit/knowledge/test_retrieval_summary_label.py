"""Tests for intent-aware labeling in RetrievalResult.summary().

Vault-only (intent == "historical") retrieval has entities_found == 0 by
design, so the generic "from 0 entities" phrasing reads like a failure. The
summary must report vault chunks instead for that path while preserving the
entity phrasing for graph-backed intents.
"""

from backend.knowledge.models import RetrievalResult


def _make_result(intent: str, total_facts: int, entities_found: int) -> RetrievalResult:
    """Build a minimal RetrievalResult with the fields summary() reads."""
    return RetrievalResult(
        query="test query",
        user_id="user-1",
        facts=[],
        entities_found=entities_found,
        total_facts=total_facts,
        formatted_context="",
        retrieval_time_ms=12.3,
        vector_search_time_ms=4.0,
        graph_traversal_time_ms=8.0,
        config_used={},
        intent=intent,
    )


def test_historical_summary_says_vault_chunks_not_entities():
    r = _make_result(intent="historical", total_facts=3, entities_found=0)
    s = r.summary()
    assert "vault chunk" in s.lower()
    assert "from 0 entities" not in s


def test_non_historical_summary_still_says_entities():
    r = _make_result(intent="factual", total_facts=5, entities_found=2)
    assert "entities" in r.summary()
