"""Knowledge domain test fixtures and factory functions."""

import pytest

from backend.knowledge.ontologies.v1_0_0 import ONTOLOGY_V1_0_0


@pytest.fixture
def sample_ontology():
    """The v1.0.0 ontology definition."""
    return ONTOLOGY_V1_0_0


def make_extraction_result(
    *,
    entities: list[dict] | None = None,
    relationships: list[dict] | None = None,
    raw_llm_output: str = '{"entities": [], "relationships": []}',
    extraction_time_ms: float = 10.0,
    source_utterance: str = "test utterance",
) -> dict:
    """Build an extraction result dict with sensible defaults.

    Returns a plain dict matching the ExtractionResult structure.
    Keyword-only args for type safety.
    """
    return {
        "entities": entities or [],
        "relationships": relationships or [],
        "raw_llm_output": raw_llm_output,
        "extraction_time_ms": extraction_time_ms,
        "source_utterance": source_utterance,
    }
