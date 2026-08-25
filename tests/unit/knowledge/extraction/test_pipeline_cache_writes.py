"""Tests for ExtractionPipeline writing decisions to the extraction cache.

Gate 0 (this file's first test): the too-short guard used to live in
conversation_handler.py, where it prevented extract_from_utterance from being
called at all -- so a gated turn produced no cache row. Moved into the
pipeline (Task 3) so the pipeline itself can record the skip.
"""

import pytest


@pytest.mark.asyncio
async def test_short_utterance_is_gated_inside_the_pipeline(pipeline_factory):
    """Gate 0: the pipeline itself decides, so the pipeline itself can record."""
    pipeline, spy_cache = pipeline_factory()
    result = await pipeline.extract_from_utterance(
        utterance="ok sure",  # two words
        conversation_history=[],
        event_id="evt-short",
        session_id="sess-1",
        recorded_at="2026-08-18T00:00:00+00:00",
    )
    assert result.entities == []
    assert spy_cache.calls == [
        ("evt-short", "skipped", "too_short"),
    ]
