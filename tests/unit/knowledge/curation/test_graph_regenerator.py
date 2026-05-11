"""Unit tests for GraphRegenerator.

Tests cover the ADR-010 invariant 5 closure: on user-edit detection,
DERIVED_FROM-scoped triples are orphan-marked and re-derived per bucket.

Bucket dispatch:
- Bucket 1 (identity/, users/): deterministic parse via bucket1_reader,
  no LLM call.
- Bucket 2/3 (sessions/, decisions/): async LLM re-extraction queued;
  result is deferred=True.

Fakes live in tests/fakes/ so they can be shared across test modules.
"""

import asyncio
from pathlib import Path

import pytest

from backend.knowledge.curation.graph_regenerator import (
    GraphRegenerator,
    RebuildResult,
)
from tests.fakes.extraction_pipeline import FakeExtractionPipeline
from tests.fakes.graph_store import FakeGraphStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_graph_store() -> FakeGraphStore:
    return FakeGraphStore()


@pytest.fixture()
def fake_extraction() -> FakeExtractionPipeline:
    return FakeExtractionPipeline()


@pytest.fixture()
def regenerator(
    fake_graph_store: FakeGraphStore,
    fake_extraction: FakeExtractionPipeline,
) -> GraphRegenerator:
    return GraphRegenerator(
        graph_store=fake_graph_store,
        extraction_pipeline=fake_extraction,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_IDENTITY_BODY = (
    "---\ntype: mist-identity\n---\n"
    "## Traits\n- **precise** [(sharp)] -- always accurate\n"
    "## Capabilities\n"
    "## Preferences\n"
)

_USER_BODY = (
    "---\ntype: mist-user\nuser_id: raj\n---\n"
    "## Tools and Technologies\n- **Python** (Technology)\n"
    "## Expertise\n"
    "## Currently Learning\n"
    "## Projects\n"
    "## Affiliations\n"
    "## Interests\n"
    "## Goals\n"
    "## Preferences\n"
    "## People\n"
)

_SESSION_BODY = "---\ntype: mist-session\n---\n" "## Turn 1\n**User:** hi\n**MIST:** hello\n"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_rebuild_marks_old_triples_orphaned(
    regenerator: GraphRegenerator,
    fake_graph_store: FakeGraphStore,
    tmp_path: Path,
) -> None:
    """Triples with DERIVED_FROM.path == edited_path are orphan-marked."""
    p = tmp_path / "sessions" / "2026-05-10-test.md"
    p.parent.mkdir()
    p.write_text(_SESSION_BODY, encoding="utf-8")

    # Pre-seed a triple linked to the path under edit
    fake_graph_store.add_triple(
        subject="user",
        predicate="USES",
        object="python",
        derived_from_path=str(p),
    )

    result: RebuildResult = asyncio.run(regenerator.rebuild_from_path(p))

    assert result.orphaned_triple_count == 1
    triple = fake_graph_store.get_triple("user", "USES", "python")
    assert triple is not None
    assert triple.status == "orphaned"


def test_rebuild_bucket1_identity_deterministic(
    regenerator: GraphRegenerator,
    fake_graph_store: FakeGraphStore,
    tmp_path: Path,
) -> None:
    """identity/mist.md triggers Bucket 1 deterministic re-derivation, no LLM."""
    p = tmp_path / "identity" / "mist.md"
    p.parent.mkdir()
    p.write_text(_IDENTITY_BODY, encoding="utf-8")

    result: RebuildResult = asyncio.run(regenerator.rebuild_from_path(p))

    assert result.bucket == "1"
    assert result.deferred is False
    # Trait was written to graph without an LLM call
    assert fake_graph_store.has_trait("precise")


def test_rebuild_bucket1_user_deterministic(
    regenerator: GraphRegenerator,
    fake_graph_store: FakeGraphStore,
    tmp_path: Path,
) -> None:
    """users/<user>.md triggers Bucket 1 deterministic re-derivation."""
    p = tmp_path / "users" / "raj.md"
    p.parent.mkdir()
    p.write_text(_USER_BODY, encoding="utf-8")

    result: RebuildResult = asyncio.run(regenerator.rebuild_from_path(p))

    assert result.bucket == "1"
    assert result.deferred is False
    # Verify a USES triple was written for the user
    triple = fake_graph_store.get_triple("raj", "USES", "Python")
    assert triple is not None


def test_rebuild_bucket2_session_defers_extraction(
    regenerator: GraphRegenerator,
    fake_extraction: FakeExtractionPipeline,
    tmp_path: Path,
) -> None:
    """sessions/* edit returns deferred=True and queues async extraction."""
    p = tmp_path / "sessions" / "2026-05-10-test.md"
    p.parent.mkdir()
    p.write_text(_SESSION_BODY, encoding="utf-8")

    result: RebuildResult = asyncio.run(regenerator.rebuild_from_path(p))

    assert result.bucket == "2"
    assert result.deferred is True
    # Extraction was scheduled (create_task fires within the same event loop
    # since asyncio.run runs the loop to completion)
    assert fake_extraction.scheduled_jobs >= 1


def test_rebuild_idempotent_no_proliferation(
    regenerator: GraphRegenerator,
    fake_graph_store: FakeGraphStore,
    tmp_path: Path,
) -> None:
    """Repeat rebuild from same content does not create duplicate triples."""
    p = tmp_path / "identity" / "mist.md"
    p.parent.mkdir()
    p.write_text(_IDENTITY_BODY, encoding="utf-8")

    asyncio.run(regenerator.rebuild_from_path(p))
    asyncio.run(regenerator.rebuild_from_path(p))

    # Exactly one HAS_TRAIT triple for "precise", regardless of rebuild count
    assert fake_graph_store.count_traits() == 1


# ---------------------------------------------------------------------------
# Tests: retry_orphaned (Task 22)
# ---------------------------------------------------------------------------


def test_retry_orphaned_reruns_extraction_for_orphaned_paths(
    regenerator: GraphRegenerator,
    fake_graph_store: FakeGraphStore,
    fake_extraction: FakeExtractionPipeline,
    tmp_path: Path,
) -> None:
    """retry_orphaned re-runs extraction for each orphaned provenance path."""
    p = tmp_path / "sessions" / "2026-05-10-test.md"
    p.parent.mkdir()
    p.write_text(_SESSION_BODY, encoding="utf-8")

    # Pre-seed a triple with 'orphaned' status for this path
    fake_graph_store.add_triple(
        subject="user",
        predicate="USES",
        object="python",
        derived_from_path=str(p),
        status="orphaned",
    )

    jobs_before = fake_extraction.scheduled_jobs
    asyncio.run(regenerator.retry_orphaned())

    # extraction should have been called once for the orphaned path
    assert fake_extraction.scheduled_jobs == jobs_before + 1
    assert fake_extraction.extract_from_file_calls[-1]["vault_note_path"] == str(p)


def test_retry_orphaned_skips_nonexistent_paths(
    regenerator: GraphRegenerator,
    fake_graph_store: FakeGraphStore,
    fake_extraction: FakeExtractionPipeline,
) -> None:
    """retry_orphaned skips paths that no longer exist on disk."""
    fake_graph_store.add_triple(
        subject="user",
        predicate="USES",
        object="python",
        derived_from_path="/nonexistent/path/file.md",
        status="orphaned",
    )

    asyncio.run(regenerator.retry_orphaned())

    # No extraction should have been attempted for the missing path
    assert fake_extraction.scheduled_jobs == 0


def test_retry_orphaned_noop_when_no_orphans(
    regenerator: GraphRegenerator,
    fake_graph_store: FakeGraphStore,
    fake_extraction: FakeExtractionPipeline,
) -> None:
    """retry_orphaned is a no-op when there are no orphaned triples."""
    # All triples active
    fake_graph_store.add_triple(
        subject="user",
        predicate="USES",
        object="python",
        derived_from_path="/some/path.md",
        status="active",
    )

    asyncio.run(regenerator.retry_orphaned())

    assert fake_extraction.scheduled_jobs == 0


def test_retry_orphaned_deduplicates_paths(
    regenerator: GraphRegenerator,
    fake_graph_store: FakeGraphStore,
    fake_extraction: FakeExtractionPipeline,
    tmp_path: Path,
) -> None:
    """retry_orphaned calls extraction once per unique path even with multiple orphaned triples."""
    p = tmp_path / "sessions" / "2026-05-10-test.md"
    p.parent.mkdir()
    p.write_text(_SESSION_BODY, encoding="utf-8")

    # Two orphaned triples for the same path
    fake_graph_store.add_triple(
        "user", "USES", "python", derived_from_path=str(p), status="orphaned"
    )
    fake_graph_store.add_triple(
        "user", "KNOWS", "django", derived_from_path=str(p), status="orphaned"
    )

    asyncio.run(regenerator.retry_orphaned())

    # FakeGraphStore.get_orphaned_provenance_paths deduplicates; extraction called once
    assert fake_extraction.scheduled_jobs == 1
