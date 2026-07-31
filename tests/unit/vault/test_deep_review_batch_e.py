"""Deep-review Batch E regressions: event-loop offloading and lifecycle.

Covers concurrency-async-2 (shared warmed embedder for curation) and
concurrency-async-3 (sidecar work confined to a dedicated worker thread).

R1.3.1 removed concurrency-async-8's subject (the partial-append synthesis
idempotency guard in `append_session_synthesis`) along with the method
itself; `write_session_note`'s full-render idempotency is covered by
`tests/unit/vault/test_write_session_note.py::test_render_is_idempotent_byte_for_byte`.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from backend.knowledge.config import SidecarIndexConfig
from backend.vault.sidecar_index import VaultSidecarIndex
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
