"""Deep-review Batch E regressions: event-loop offloading and lifecycle.

Covers concurrency-async-2 (shared warmed embedder for curation),
concurrency-async-3 (sidecar work confined to a dedicated worker thread),
and concurrency-async-8 (synthesis idempotency on reconnect-driven
end_session).
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from backend.knowledge.config import SidecarIndexConfig, VaultConfig
from backend.vault.sidecar_index import VaultSidecarIndex
from backend.vault.writer import VaultWriter
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeGraphExecutor, FakeNeo4jConnection

# ---------------------------------------------------------------------------
# concurrency-async-3: sidecar worker confinement
# ---------------------------------------------------------------------------


def _sidecar(tmp_path: Path) -> VaultSidecarIndex:
    return VaultSidecarIndex(
        SidecarIndexConfig(
            enabled=True,
            db_path=str(tmp_path / "sidecar.db"),
            embedding_dimension=8,
        ),
        FakeEmbeddingGenerator(),
    )


class TestSidecarWorkerConfinement:
    @pytest.mark.asyncio
    async def test_run_async_confines_to_one_worker_thread(self, tmp_path: Path):
        idx = _sidecar(tmp_path)
        try:
            tids = [await idx.run_async(threading.get_ident) for _ in range(3)]
        finally:
            idx.close()

        assert len(set(tids)) == 1, "all sidecar work must run on ONE worker thread"
        assert tids[0] != threading.get_ident(), "work must leave the event-loop thread"

    @pytest.mark.asyncio
    async def test_close_is_idempotent_with_worker(self, tmp_path: Path):
        idx = _sidecar(tmp_path)
        await idx.run_async(lambda: None)
        idx.close()
        idx.close()  # second close must not raise


class _WorkerRoutingSidecar:
    """Sidecar double exposing run_async, to assert the filewatcher routes
    its blocking reindex/delete units through the worker API.
    """

    def __init__(self) -> None:
        self.run_async_calls = 0
        self.upsert_file_calls: list[str] = []
        self.delete_path_calls: list[str] = []

    async def run_async(self, fn, /, *args):
        self.run_async_calls += 1
        return fn(*args)

    def upsert_file(self, path, content, mtime, frontmatter=None) -> int:
        self.upsert_file_calls.append(path)
        return 1

    def delete_path(self, path) -> int:
        self.delete_path_calls.append(path)
        return 1

    def distinct_paths(self) -> list[str]:
        return []


class TestFilewatcherUsesWorker:
    @pytest.mark.asyncio
    async def test_reindex_routes_through_sidecar_worker(self, tmp_path: Path):
        from backend.knowledge.config import FilewatcherConfig
        from backend.vault.filewatcher import VaultFilewatcher

        sidecar = _WorkerRoutingSidecar()
        root = tmp_path / "vault"
        (root / "sessions").mkdir(parents=True)
        note = root / "sessions" / "n.md"
        note.write_text("---\ntype: mist-session\n---\n\nbody\n", encoding="utf-8")

        w = VaultFilewatcher(
            FilewatcherConfig(enabled=True, observer_type="polling", debounce_ms=100),
            root,
            sidecar,
            regenerator=None,
            invalidation_bus=None,
            writer=None,
        )

        await w._do_reindex(str(note), is_mist_write=True)

        assert sidecar.run_async_calls == 1
        assert sidecar.upsert_file_calls == [str(note)]

    @pytest.mark.asyncio
    async def test_delete_routes_through_sidecar_worker(self, tmp_path: Path):
        from backend.knowledge.config import FilewatcherConfig
        from backend.vault.filewatcher import VaultFilewatcher

        sidecar = _WorkerRoutingSidecar()
        w = VaultFilewatcher(
            FilewatcherConfig(enabled=True, observer_type="polling", debounce_ms=100),
            tmp_path / "vault",
            sidecar,
            regenerator=None,
            invalidation_bus=None,
            writer=None,
        )

        await w._do_delete(str(tmp_path / "vault" / "gone.md"))

        assert sidecar.run_async_calls == 1
        assert sidecar.delete_path_calls == [str(tmp_path / "vault" / "gone.md")]


# ---------------------------------------------------------------------------
# concurrency-async-2: curation shares the warmed embedding model
# ---------------------------------------------------------------------------


class TestCurationEmbeddingSharing:
    def test_build_curation_pipeline_wires_injected_provider_everywhere(self):
        from backend.factories import build_curation_pipeline
        from tests.mocks.config import build_test_config

        sentinel = FakeEmbeddingGenerator()
        executor = FakeGraphExecutor(connection=FakeNeo4jConnection())

        pipeline = build_curation_pipeline(
            build_test_config(), executor, embedding_provider=sentinel
        )

        # A private second EmbeddingGenerator is never warmed by ModelManager,
        # so the first tier-3 dedup or new-entity write cold-loads a
        # SentenceTransformer on the event loop UNDER the curation lock.
        assert pipeline._deduplicator._embedding_provider is sentinel
        assert pipeline._graph_writer._embedding_provider is sentinel


# ---------------------------------------------------------------------------
# concurrency-async-8: synthesis idempotency
# ---------------------------------------------------------------------------

_NOTE = (
    "---\n"
    "type: mist-session\n"
    "session_id: s1\n"
    "status: active\n"
    "authored_by: mist\n"
    "---\n\n"
    "## Turn 1\n\nhello\n\n"
    "<!-- MIST_APPEND_HERE -->\n"
)


class TestSynthesisIdempotency:
    @pytest.mark.asyncio
    async def test_second_synthesis_append_is_skipped(self, tmp_path: Path):
        writer = VaultWriter(
            VaultConfig(enabled=True, root=str(tmp_path / "vault"), git_auto_init=False)
        )
        await writer.start()
        try:
            note = Path(writer.config.root) / "sessions" / "s1.md"
            note.parent.mkdir(parents=True, exist_ok=True)
            note.write_text(_NOTE, encoding="utf-8")

            await writer.append_session_synthesis(str(note), "First synthesis.")
            await writer.append_session_synthesis(str(note), "Second synthesis.")
        finally:
            await writer.stop()

        content = note.read_text(encoding="utf-8")
        assert content.count("## Summary") == 1
        assert "Second synthesis." not in content

    @pytest.mark.asyncio
    async def test_completed_note_is_never_synthesized_again(self, tmp_path: Path):
        writer = VaultWriter(
            VaultConfig(enabled=True, root=str(tmp_path / "vault"), git_auto_init=False)
        )
        await writer.start()
        try:
            note = Path(writer.config.root) / "sessions" / "s2.md"
            note.parent.mkdir(parents=True, exist_ok=True)
            note.write_text(_NOTE.replace("status: active", "status: completed"), encoding="utf-8")

            await writer.append_session_synthesis(str(note), "Late synthesis.")
        finally:
            await writer.stop()

        content = note.read_text(encoding="utf-8")
        assert "## Summary" not in content
