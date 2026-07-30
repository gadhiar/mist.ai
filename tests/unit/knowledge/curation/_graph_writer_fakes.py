"""Shared fakes for curation graph-writer unit tests.

Extracted so the utterance-provenance tests and the existing graph-writer
tests share one recording executor instead of two drifting copies.
"""

from typing import Any

from backend.knowledge.curation.graph_writer import CurationGraphWriter, RebuildStamps


class FakeExecutor:
    """Records every execute_write call as (query, params)."""

    def __init__(self) -> None:
        self.writes: list[tuple[str, dict[str, Any]]] = []

    async def execute_write(self, query: str, params: dict[str, Any] | None = None) -> list:
        self.writes.append((query, params or {}))
        return []

    async def execute_query(self, query: str, params: dict[str, Any] | None = None) -> list:
        return []


class FakeEmbeddingProvider:
    def generate_embedding(self, text: str) -> list[float]:
        return [0.0] * 384


class FakeConfidenceManager:
    def determine_domain(self, entity_type: str) -> Any:
        class _Domain:
            value = "technical"

        return _Domain()

    def reinforced_confidence(self, confidence: float, domain: Any) -> float:
        return confidence


def make_writer(
    rebuild_stamps: RebuildStamps | None = None,
) -> tuple[CurationGraphWriter, FakeExecutor]:
    """Build a CurationGraphWriter over a recording fake executor."""
    executor = FakeExecutor()
    writer = CurationGraphWriter(
        executor=executor,
        embedding_provider=FakeEmbeddingProvider(),
        confidence_manager=FakeConfidenceManager(),
        rebuild_stamps=rebuild_stamps,
    )
    return writer, executor


def writes_matching(executor: FakeExecutor, needle: str) -> list[tuple[str, dict[str, Any]]]:
    """Return recorded writes whose query contains `needle`."""
    return [(q, p) for q, p in executor.writes if needle in q]
