"""Unit tests for GraphRegenerator.

R1.3 retires both re-derivation buckets: Bucket 1 (deterministic user/
identity parse, Task 3) and Bucket 2/3 (LLM re-extraction of session and
decision notes, this task). `rebuild_from_path` is now an unconditional
graph no-op for every vault path -- Inv-A1: the vault is not a fact source;
facts enter the graph only through the utterance log.

Fakes live in tests/fakes/ so they can be shared across test modules.
"""

import asyncio
from pathlib import Path

import pytest

from backend.knowledge.curation.graph_regenerator import (
    GraphRegenerator,
    RebuildResult,
)
from tests.fakes.graph_store import FakeGraphStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_graph_store() -> FakeGraphStore:
    return FakeGraphStore()


class _NeverCalledExtractionPipeline:
    """Stub that fails loudly if the retired re-extraction path fires."""

    def __init__(self) -> None:
        self.extract_from_file_calls: list[dict] = []


@pytest.fixture()
def fake_extraction() -> _NeverCalledExtractionPipeline:
    return _NeverCalledExtractionPipeline()


@pytest.fixture()
def regenerator(fake_graph_store: FakeGraphStore) -> GraphRegenerator:
    return GraphRegenerator(graph_store=fake_graph_store)


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


def test_rebuild_does_not_orphan_existing_triples(
    regenerator: GraphRegenerator,
    fake_graph_store: FakeGraphStore,
    tmp_path: Path,
) -> None:
    """R1.3: a vault edit leaves pre-existing DERIVED_FROM-scoped triples untouched.

    Before R1.3 this same fixture (a triple scoped to the edited path) proved
    orphan-marking fired on rebuild. Bucket 2/3's retirement (this task)
    collapses rebuild_from_path to an unconditional no-op that never reaches
    the graph store, so the same fixture now proves the inverse: the triple
    survives the edit untouched.
    """
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

    assert result.orphaned_triple_count == 0
    assert fake_graph_store.mark_orphaned_calls == []
    triple = fake_graph_store.get_triple("user", "USES", "python")
    assert triple is not None
    assert triple.status == "active"


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
    """Any identity/ file remains a graph no-op under the unconditional no-op.

    Originally pinned a narrow-guard bug (Task 3 review: only literal
    "identity/mist.md" was inert; a differently-named identity/ file fell
    through to Bucket 2/3 LLM re-extraction). R1.3 retired the guard's
    path-segment matching entirely -- every path is now inert -- so this is
    retained as a regression check on that now-broader behavior.
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
    """A users/ file nested below a subdirectory remains a graph no-op.

    Originally pinned a narrow-guard bug (Task 3 review: "users" had to be
    the immediate parent directory). R1.3 retired the guard's path-segment
    matching entirely -- every path is now inert -- so this is retained as a
    regression check on that now-broader behavior.
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


def test_rebuild_session_file_is_graph_noop(
    regenerator: GraphRegenerator,
    fake_graph_store: FakeGraphStore,
    fake_extraction: "_NeverCalledExtractionPipeline",
    tmp_path: Path,
) -> None:
    """R1.3 spec 5.1: session-note edits trigger no LLM re-extraction."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    note = sessions_dir / "2026-07-30.md"
    note.write_text("# Turn 1\n\nI use Python at work.\n", encoding="utf-8")

    result = asyncio.run(regenerator.rebuild_from_path(note))

    assert result.bucket == "ignored"
    assert result.deferred is False
    assert result.new_triple_count == 0
    assert result.orphaned_triple_count == 0
    assert fake_extraction.extract_from_file_calls == [], "no LLM re-extraction may fire"


def test_rebuild_decision_file_is_graph_noop(
    regenerator: GraphRegenerator,
    fake_extraction: "_NeverCalledExtractionPipeline",
    tmp_path: Path,
) -> None:
    """decisions/ retires with sessions/ -- both were Bucket 2/3."""
    decisions_dir = tmp_path / "decisions"
    decisions_dir.mkdir()
    note = decisions_dir / "ADR-001.md"
    note.write_text("# ADR-001\n\nWe chose Neo4j.\n", encoding="utf-8")

    result = asyncio.run(regenerator.rebuild_from_path(note))

    assert result.bucket == "ignored"
    assert fake_extraction.extract_from_file_calls == []


def test_regenerator_has_no_async_lifecycle() -> None:
    """No in-flight tasks means no drain: aclose and retry_orphaned retire."""
    assert not hasattr(GraphRegenerator, "aclose")
    assert not hasattr(GraphRegenerator, "retry_orphaned")


def test_extraction_pipeline_has_no_extract_from_file() -> None:
    """The vault-file fact writer is deleted, not left dormant."""
    from backend.knowledge.extraction.pipeline import ExtractionPipeline

    assert not hasattr(ExtractionPipeline, "extract_from_file"), (
        "R1.3: extract_from_file is a vault-file->graph fact path and retires "
        "with its sole caller"
    )
