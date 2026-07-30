"""Unit tests for GraphRegenerator.

Tests cover the ADR-010 invariant 5 closure: on user-edit detection,
DERIVED_FROM-scoped triples are orphan-marked and re-derived per bucket.

Bucket dispatch:
- identity/mist.md, users/<id>.md: graph no-op (R1.3). The vault is not a
  fact source for these paths; an edit changes read-path prose only.
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


def test_rebuild_mist_md_is_graph_noop(
    regenerator: GraphRegenerator,
    fake_graph_store: FakeGraphStore,
    tmp_path: Path,
) -> None:
    """identity/mist.md is graph-canonical: an edit is a no-op.

    The self-model is not vault-derived (R1 truth model), so a mist.md edit
    must not orphan-mark or re-derive any graph state.
    """
    p = tmp_path / "identity" / "mist.md"
    p.parent.mkdir()
    p.write_text(_IDENTITY_BODY, encoding="utf-8")

    result: RebuildResult = asyncio.run(regenerator.rebuild_from_path(p))

    assert result.bucket == "ignored"
    assert result.deferred is False
    assert result.orphaned_triple_count == 0
    assert result.new_triple_count == 0
    # No graph write and no orphan-mark happened.
    assert not fake_graph_store.has_trait("precise")
    assert fake_graph_store.mark_orphaned_calls == []


def test_rebuild_user_file_is_graph_noop(
    regenerator: GraphRegenerator,
    fake_graph_store: FakeGraphStore,
    tmp_path: Path,
) -> None:
    """R1.3 spec 5.1: user-file edits write no graph facts.

    The user file is prose the read path injects, not a fact source. A rebuild
    must not orphan-mark, must not parse, and must not upsert.
    """
    users_dir = tmp_path / "users"
    users_dir.mkdir()
    user_file = users_dir / "raj.md"
    user_file.write_text(
        "---\nuser_id: raj\n---\n\n## Tools and Technologies\n- Python\n- Neo4j\n",
        encoding="utf-8",
    )

    result = asyncio.run(regenerator.rebuild_from_path(user_file))

    assert result.bucket == "ignored"
    assert result.new_triple_count == 0
    assert result.orphaned_triple_count == 0
    assert result.deferred is False
    assert fake_graph_store.upsert_user_calls == [], "no user facts may be written"
    assert fake_graph_store.mark_orphaned_calls == [], "no orphan-marking on a no-op"


def test_rebuild_identity_non_mist_file_is_graph_noop(
    regenerator: GraphRegenerator,
    fake_graph_store: FakeGraphStore,
    tmp_path: Path,
) -> None:
    """Any identity/ file is a graph no-op, not just identity/mist.md.

    The guard must match on the "identity" path segment, not only on the
    literal filename "mist.md" -- an identity/ file with a different name
    (e.g. a future identity/persona.md) is still read-path prose and must
    not fall through to Bucket 2/3 LLM re-extraction.
    """
    p = tmp_path / "identity" / "persona.md"
    p.parent.mkdir()
    p.write_text(_IDENTITY_BODY, encoding="utf-8")

    result: RebuildResult = asyncio.run(regenerator.rebuild_from_path(p))

    assert result.bucket == "ignored"
    assert result.deferred is False
    assert result.orphaned_triple_count == 0
    assert result.new_triple_count == 0
    assert fake_graph_store.mark_orphaned_calls == []


def test_rebuild_nested_user_file_is_graph_noop(
    regenerator: GraphRegenerator,
    fake_graph_store: FakeGraphStore,
    tmp_path: Path,
) -> None:
    """A users/ file nested below a subdirectory is still a graph no-op.

    The guard must match on the "users" path segment anywhere in the path,
    not only when "users" is the immediate parent directory -- a nested
    users/<subdir>/<file>.md edit must not fall through to Bucket 2/3 LLM
    re-extraction.
    """
    p = tmp_path / "users" / "archive" / "raj.md"
    p.parent.mkdir(parents=True)
    p.write_text(
        "---\nuser_id: raj\n---\n\n## Tools and Technologies\n- Python\n",
        encoding="utf-8",
    )

    result: RebuildResult = asyncio.run(regenerator.rebuild_from_path(p))

    assert result.bucket == "ignored"
    assert result.deferred is False
    assert result.orphaned_triple_count == 0
    assert result.new_triple_count == 0
    assert fake_graph_store.upsert_user_calls == []
    assert fake_graph_store.mark_orphaned_calls == []


def test_rebuild_bucket2_session_defers_extraction(
    regenerator: GraphRegenerator,
    fake_extraction: FakeExtractionPipeline,
    tmp_path: Path,
) -> None:
    """sessions/* edit returns deferred=True and queues async extraction."""
    p = tmp_path / "sessions" / "2026-05-10-test.md"
    p.parent.mkdir()
    p.write_text(_SESSION_BODY, encoding="utf-8")

    async def _run():
        result = await regenerator.rebuild_from_path(p)
        # Drain in-flight tasks so the inner wait_for pipeline call completes.
        await regenerator.aclose()
        return result

    result: RebuildResult = asyncio.run(_run())

    assert result.bucket == "2"
    assert result.deferred is True
    # Extraction completed (create_task + aclose drain ensures pipeline ran)
    assert fake_extraction.scheduled_jobs >= 1


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
