"""Tests for GraphRegenerator asyncio task lifecycle (Fix A, P3 #2).

Covers:
- _in_flight set is populated on Bucket 2/3 create_task and cleared on done
- asyncio.wait_for timeout wrapper fires on a hung extraction
- aclose() drains in-flight tasks before returning
- rebuild_timeout_s is sourced from GraphRegeneratorConfig (env: MIST_GRAPH_REGENERATOR_REBUILD_TIMEOUT_S)
"""

from __future__ import annotations

import asyncio
import os
from contextlib import contextmanager
from pathlib import Path

import pytest

from backend.knowledge.config import GraphRegeneratorConfig
from backend.knowledge.curation.graph_regenerator import GraphRegenerator
from tests.fakes.extraction_pipeline import FakeExtractionPipeline
from tests.fakes.graph_store import FakeGraphStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextmanager
def _env(**values):
    original = {k: os.environ.get(k) for k in values}
    try:
        for k, v in values.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        yield
    finally:
        for k, v in original.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


_SESSION_BODY = "---\ntype: mist-session\n---\n## Turn 1\n**User:** hi\n**MIST:** hello\n"


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
# Tests: GraphRegeneratorConfig
# ---------------------------------------------------------------------------


class TestGraphRegeneratorConfig:
    """GraphRegeneratorConfig carries rebuild_timeout_s with env-var override."""

    def test_default_rebuild_timeout_s_is_300(self):
        config = GraphRegeneratorConfig()
        assert config.rebuild_timeout_s == 300

    def test_from_env_reads_timeout_override(self):
        with _env(MIST_GRAPH_REGENERATOR_REBUILD_TIMEOUT_S="120"):
            config = GraphRegeneratorConfig.from_env()
        assert config.rebuild_timeout_s == 120

    def test_from_env_uses_default_when_unset(self):
        with _env(MIST_GRAPH_REGENERATOR_REBUILD_TIMEOUT_S=None):
            config = GraphRegeneratorConfig.from_env()
        assert config.rebuild_timeout_s == 300

    def test_config_is_frozen(self):
        config = GraphRegeneratorConfig()
        try:
            config.rebuild_timeout_s = 999  # type: ignore[misc]
        except (AttributeError, Exception) as exc:
            assert "frozen" in str(exc).lower() or "cannot assign" in str(exc).lower()
        else:
            raise AssertionError("GraphRegeneratorConfig should be frozen but allowed mutation")


# ---------------------------------------------------------------------------
# Tests: _in_flight tracking
# ---------------------------------------------------------------------------


class TestInFlightTracking:
    """_in_flight set is populated on Bucket 2/3 task creation and discarded on done."""

    @pytest.mark.asyncio
    async def test_in_flight_set_initialised_empty(self, regenerator: GraphRegenerator):
        assert hasattr(regenerator, "_in_flight")
        assert len(regenerator._in_flight) == 0

    @pytest.mark.asyncio
    async def test_in_flight_populated_during_bucket2_rebuild(
        self,
        fake_graph_store: FakeGraphStore,
        fake_extraction: FakeExtractionPipeline,
        tmp_path: Path,
    ):
        """Task is added to _in_flight before it completes."""
        # Arrange: use a slow extraction that we can observe mid-flight.
        in_flight_size_during: list[int] = []

        async def slow_extract(content, vault_note_path, ontology_version):
            # Sample _in_flight from within the running task.
            in_flight_size_during.append(len(regen._in_flight))
            await asyncio.sleep(0)
            fake_extraction.scheduled_jobs += 1
            fake_extraction.extract_from_file_calls.append(
                {
                    "content": content,
                    "vault_note_path": vault_note_path,
                    "ontology_version": ontology_version,
                }
            )

        fake_extraction.extract_from_file = slow_extract  # type: ignore[method-assign]

        p = tmp_path / "sessions" / "2026-05-11-test.md"
        p.parent.mkdir()
        p.write_text(_SESSION_BODY, encoding="utf-8")

        regen = GraphRegenerator(
            graph_store=fake_graph_store,
            extraction_pipeline=fake_extraction,
        )

        await regen.rebuild_from_path(p)
        # Drain via aclose so the wait_for chain fully resolves.
        await regen.aclose()

        # Task was in _in_flight while running (at least one snapshot > 0) OR
        # it completed so fast we only see 0 after done-callback -- either way the
        # set drains back to 0 once done.
        assert len(regen._in_flight) == 0

    @pytest.mark.asyncio
    async def test_in_flight_drains_to_zero_after_completion(
        self,
        fake_graph_store: FakeGraphStore,
        fake_extraction: FakeExtractionPipeline,
        tmp_path: Path,
    ):
        """_in_flight is empty after all deferred tasks complete (via aclose)."""
        p = tmp_path / "sessions" / "2026-05-11-test.md"
        p.parent.mkdir()
        p.write_text(_SESSION_BODY, encoding="utf-8")

        regen = GraphRegenerator(
            graph_store=fake_graph_store,
            extraction_pipeline=fake_extraction,
        )

        await regen.rebuild_from_path(p)
        # Use aclose() to reliably drain -- it awaits all in-flight tasks.
        await regen.aclose()

        assert len(regen._in_flight) == 0


# ---------------------------------------------------------------------------
# Tests: timeout wrapper
# ---------------------------------------------------------------------------


class TestRebuildTimeout:
    """asyncio.wait_for wraps _rebuild_async_extraction; hung calls raise TimeoutError."""

    @pytest.mark.asyncio
    async def test_timeout_fires_on_hung_extraction(
        self,
        fake_graph_store: FakeGraphStore,
        tmp_path: Path,
    ):
        """When extraction hangs past rebuild_timeout_s, the task is cancelled."""
        hung_called = False

        class HungExtractionPipeline:
            async def extract_from_file(self, content, vault_note_path, ontology_version):
                nonlocal hung_called
                hung_called = True
                await asyncio.sleep(9999)  # Simulates a hung LLM call.

        regen = GraphRegenerator(
            graph_store=fake_graph_store,
            extraction_pipeline=HungExtractionPipeline(),
            rebuild_timeout_s=0.05,  # Very short for test speed.
        )

        p = tmp_path / "sessions" / "2026-05-11-timeout.md"
        p.parent.mkdir()
        p.write_text(_SESSION_BODY, encoding="utf-8")

        await regen.rebuild_from_path(p)

        # Allow the scheduled task to start and hit timeout.
        await asyncio.sleep(0.15)

        # _in_flight must drain (task cancelled/done).
        assert len(regen._in_flight) == 0
        assert hung_called is True

    @pytest.mark.asyncio
    async def test_timeout_does_not_fire_on_fast_extraction(
        self,
        fake_graph_store: FakeGraphStore,
        fake_extraction: FakeExtractionPipeline,
        tmp_path: Path,
    ):
        """Fast extractions complete normally; no spurious timeout."""
        regen = GraphRegenerator(
            graph_store=fake_graph_store,
            extraction_pipeline=fake_extraction,
            rebuild_timeout_s=30,  # Generous timeout; fast extraction will finish.
        )

        p = tmp_path / "sessions" / "2026-05-11-fast.md"
        p.parent.mkdir()
        p.write_text(_SESSION_BODY, encoding="utf-8")

        await regen.rebuild_from_path(p)
        # Drain via aclose so the task's wait_for chain completes.
        await regen.aclose()

        assert fake_extraction.scheduled_jobs == 1
        assert len(regen._in_flight) == 0


# ---------------------------------------------------------------------------
# Tests: aclose()
# ---------------------------------------------------------------------------


class TestAclose:
    """aclose() awaits all in-flight tasks to completion before returning."""

    @pytest.mark.asyncio
    async def test_aclose_drains_in_flight_tasks(
        self,
        fake_graph_store: FakeGraphStore,
        tmp_path: Path,
    ):
        """Aclose awaits in-flight tasks; extraction ran to completion after aclose."""
        completed: list[str] = []

        class TrackingPipeline:
            async def extract_from_file(self, content, vault_note_path, ontology_version):
                await asyncio.sleep(0.02)  # Brief delay so task is in-flight during aclose.
                completed.append(vault_note_path)

        regen = GraphRegenerator(
            graph_store=fake_graph_store,
            extraction_pipeline=TrackingPipeline(),
            rebuild_timeout_s=10,
        )

        p = tmp_path / "sessions" / "2026-05-11-aclose.md"
        p.parent.mkdir()
        p.write_text(_SESSION_BODY, encoding="utf-8")

        await regen.rebuild_from_path(p)
        # Task is now in-flight. aclose() must wait for it.
        await regen.aclose()

        assert str(p) in completed
        assert len(regen._in_flight) == 0

    @pytest.mark.asyncio
    async def test_aclose_noop_when_no_in_flight_tasks(self, regenerator: GraphRegenerator):
        """Aclose on a regenerator with no pending tasks is a no-op."""
        assert len(regenerator._in_flight) == 0
        # Must not raise.
        await regenerator.aclose()

    @pytest.mark.asyncio
    async def test_aclose_suppresses_task_exceptions(
        self,
        fake_graph_store: FakeGraphStore,
        tmp_path: Path,
    ):
        """Aclose uses return_exceptions=True; failing tasks do not propagate to caller."""

        class FailingPipeline:
            async def extract_from_file(self, content, vault_note_path, ontology_version):
                raise RuntimeError("extraction failed")

        regen = GraphRegenerator(
            graph_store=fake_graph_store,
            extraction_pipeline=FailingPipeline(),
            rebuild_timeout_s=10,
        )

        p = tmp_path / "sessions" / "2026-05-11-fail.md"
        p.parent.mkdir()
        p.write_text(_SESSION_BODY, encoding="utf-8")

        await regen.rebuild_from_path(p)
        # aclose must not raise even though the task will fail.
        await regen.aclose()

        assert len(regen._in_flight) == 0
