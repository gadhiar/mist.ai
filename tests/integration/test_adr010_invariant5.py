"""ADR-010 invariant 5 contract test.

Extends the 2026-05-08 reviewer follow-up TestADR010Invariant4 (commit
0462802) to add invariant 5: "User can edit vault. User cannot edit
graph directly. Vault wins on conflict. On vault edit, affected graph
subgraph is rebuilt from the updated vault content."

The four-step coordination on vault user-edit:
1. authored_by transitions to user-edit
2. Old triples (DERIVED_FROM.path == edited_path) marked orphaned
3. New triples written from updated content (Bucket 1 deterministic)
4. mist_context cache invalidated for affected sessions

Approach: Fake-based (Option B). No real Neo4j required. FakeGraphStore
and FakeExtractionPipeline from tests/fakes/ satisfy the protocol
surfaces. VaultWriter.mark_authored_by_user_edit is called directly
(it is not queue-backed), so VaultWriter.start() is not needed for
this path.

The test calls VaultFilewatcher._do_reindex directly (bypassing the
watchdog observer thread) to exercise the in-process coordination
contract without thread / port dependencies.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.knowledge.config import FilewatcherConfig, VaultConfig
from backend.knowledge.curation.graph_regenerator import GraphRegenerator, RebuildResult
from backend.vault.filewatcher import VaultFilewatcher
from backend.vault.invalidation_bus import InvalidationBus
from backend.vault.writer import VaultWriter
from tests.fakes.extraction_pipeline import FakeExtractionPipeline
from tests.fakes.graph_store import FakeGraphStore

# ---------------------------------------------------------------------------
# Inline fakes / stubs
# ---------------------------------------------------------------------------


class _FakeSidecarIndex:
    """Minimal SidecarIndexProtocol double; records upsert calls."""

    def __init__(self) -> None:
        self.upsert_calls: list[str] = []

    def initialize(self) -> None:
        pass

    def close(self) -> None:
        pass

    def upsert_file(self, path: str, content: str, mtime: int, frontmatter=None) -> int:
        self.upsert_calls.append(path)
        return 1

    def delete_path(self, path: str) -> int:
        return 0

    def query_vector(self, embedding, k=10):
        return []

    def query_fts(self, text, k=10):
        return []

    def query_hybrid(self, embedding, text, k=10, rrf_k=60):
        return []

    def chunk_count(self) -> int:
        return 0

    def health_check(self) -> bool:
        return True


class _RecordingListener:
    """Subscribes to InvalidationBus and records published events."""

    def __init__(self) -> None:
        self.events: list[RebuildResult] = []

    async def __call__(self, event: RebuildResult) -> None:
        self.events.append(event)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_USER_FILE_INITIAL = (
    "---\n"
    "type: mist-user\n"
    "user_id: raj\n"
    "authored_by: mist\n"
    "last_updated: 2026-05-10\n"
    "---\n"
    "\n"
    "## Tools and Technologies\n"
    "- **python** (Technology)\n"
)

_USER_FILE_EDITED = (
    "---\n"
    "type: mist-user\n"
    "user_id: raj\n"
    "authored_by: mist\n"
    "last_updated: 2026-05-10\n"
    "---\n"
    "\n"
    "## Tools and Technologies\n"
    "- **rust** (Technology)\n"
)


def _build_filewatcher(
    vault_root: Path,
    sidecar: _FakeSidecarIndex,
    graph_store: FakeGraphStore,
    extraction_pipeline: FakeExtractionPipeline,
    bus: InvalidationBus,
    writer: VaultWriter,
) -> VaultFilewatcher:
    """Wire up a VaultFilewatcher with all Phase-3 dependencies injected."""
    regenerator = GraphRegenerator(
        graph_store=graph_store,
        extraction_pipeline=extraction_pipeline,
    )
    config = FilewatcherConfig(
        enabled=True,
        observer_type="polling",
        debounce_ms=500,
        staleness_slo_seconds=5,
        audit_interval_seconds=60,
    )
    return VaultFilewatcher(
        config=config,
        vault_root=vault_root,
        sidecar_index=sidecar,
        regenerator=regenerator,
        invalidation_bus=bus,
        writer=writer,
    )


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestADR010Invariant5:
    """Contract tests for ADR-010 invariant 5.

    Invariant 5: "User can edit vault. User cannot edit graph directly.
    Vault wins on conflict. On vault edit, affected graph subgraph is
    rebuilt from the updated vault content."

    Each test exercises VaultFilewatcher._do_reindex with is_mist_write=False
    (user-origin) and asserts the four-step post-sidecar coordination:

      Step 1 - authored_by -> user-edit   (VaultWriter.mark_authored_by_user_edit)
      Step 2 - old triples orphaned        (GraphRegenerator -> FakeGraphStore)
      Step 3 - new triples written         (Bucket 1 deterministic parse)
      Step 4 - bus publishes               (InvalidationBus -> listener)
    """

    @pytest.mark.asyncio
    async def test_user_edit_triggers_four_step_coordination(self, tmp_path: Path) -> None:
        """Core invariant-5 contract: all four steps fire on user-edit detection.

        A users/<user>.md file is pre-seeded in the graph with a USES edge
        derived from the file. The test simulates a user edit (python -> rust)
        and asserts each step of the coordination.
        """
        vault_root = tmp_path / "vault"
        vault_root.mkdir()
        (vault_root / "users").mkdir()

        user_file = vault_root / "users" / "raj.md"
        user_file.write_text(_USER_FILE_INITIAL, encoding="utf-8")

        # Pre-seed graph: USES/python triple derived from this file
        graph_store = FakeGraphStore()
        graph_store.add_triple(
            subject="raj",
            predicate="USES",
            object="python",
            derived_from_path=str(user_file),
            status="active",
        )
        old_triple = graph_store.get_triple("raj", "USES", "python")
        assert old_triple is not None
        assert old_triple.status == "active"

        extraction_pipeline = FakeExtractionPipeline()
        bus = InvalidationBus()
        listener = _RecordingListener()
        bus.subscribe(listener)

        vault_config = VaultConfig(
            enabled=True,
            root=str(vault_root),
            git_auto_init=False,
        )
        writer = VaultWriter(config=vault_config)
        # mark_authored_by_user_edit does NOT use the queue consumer;
        # no writer.start() required for this path.

        sidecar = _FakeSidecarIndex()
        fw = _build_filewatcher(vault_root, sidecar, graph_store, extraction_pipeline, bus, writer)

        # Simulate user edit: overwrite content (python -> rust)
        user_file.write_text(_USER_FILE_EDITED, encoding="utf-8")

        # Trigger the filewatcher coordination (bypass observer thread)
        await fw._do_reindex(str(user_file), is_mist_write=False)

        # ------------------------------------------------------------------
        # Step 1: authored_by transitioned to user-edit
        # ------------------------------------------------------------------
        file_text = user_file.read_text(encoding="utf-8")
        assert "authored_by: user-edit" in file_text, (
            "authored_by must be rewritten to 'user-edit' after user-edit detection; "
            f"got file content:\n{file_text}"
        )

        # ------------------------------------------------------------------
        # Step 2: old triple (USES/python) marked orphaned
        # ------------------------------------------------------------------
        assert old_triple.status == "orphaned", (
            "Old USES/python triple must be status='orphaned' after orphan-mark step; "
            f"got status={old_triple.status!r}"
        )
        assert str(user_file) in graph_store.mark_orphaned_calls, (
            "mark_orphaned_by_provenance_path must be called with the edited file path; "
            f"calls={graph_store.mark_orphaned_calls!r}"
        )

        # ------------------------------------------------------------------
        # Step 3: new triple written from updated content (Bucket 1, USES/rust)
        # ------------------------------------------------------------------
        new_triple = graph_store.get_triple("raj", "USES", "rust")
        assert new_triple is not None, (
            "New USES/rust triple must be written from the updated file content; "
            f"triples present: {[(t.subject, t.predicate, t.object) for t in graph_store._triples]}"
        )
        assert (
            new_triple.status != "orphaned"
        ), f"New USES/rust triple must not be orphaned; got status={new_triple.status!r}"

        # ------------------------------------------------------------------
        # Step 4: invalidation bus published the rebuild event
        # ------------------------------------------------------------------
        assert len(listener.events) == 1, (
            f"InvalidationBus must publish exactly one event after rebuild; "
            f"got {len(listener.events)} events"
        )
        event = listener.events[0]
        assert event.path == Path(user_file), (
            f"Published event.path must equal the edited file path; " f"got {event.path!r}"
        )
        assert event.bucket == "1", (
            f"users/ file must be classified as Bucket 1 (deterministic); "
            f"got bucket={event.bucket!r}"
        )
        assert not event.deferred, "Bucket 1 rebuild must not be deferred (no LLM queued)"

    @pytest.mark.asyncio
    async def test_mist_write_skips_invariant5_steps(self, tmp_path: Path) -> None:
        """Control: MIST-origin writes must NOT trigger the invariant-5 steps.

        When is_mist_write=True, authored_by must not be rewritten, graph
        must not be rebuilt, and the bus must not be published.
        """
        vault_root = tmp_path / "vault"
        vault_root.mkdir()
        (vault_root / "users").mkdir()

        user_file = vault_root / "users" / "raj.md"
        user_file.write_text(_USER_FILE_INITIAL, encoding="utf-8")

        graph_store = FakeGraphStore()
        graph_store.add_triple(
            subject="raj",
            predicate="USES",
            object="python",
            derived_from_path=str(user_file),
            status="active",
        )

        extraction_pipeline = FakeExtractionPipeline()
        bus = InvalidationBus()
        listener = _RecordingListener()
        bus.subscribe(listener)

        vault_config = VaultConfig(enabled=True, root=str(vault_root), git_auto_init=False)
        writer = VaultWriter(config=vault_config)
        sidecar = _FakeSidecarIndex()
        fw = _build_filewatcher(vault_root, sidecar, graph_store, extraction_pipeline, bus, writer)

        # MIST-origin path: authored_by writeback and graph rebuild must be skipped
        await fw._do_reindex(str(user_file), is_mist_write=True)

        # authored_by must remain unchanged
        file_text = user_file.read_text(encoding="utf-8")
        assert (
            "authored_by: mist" in file_text
        ), "authored_by must NOT be rewritten for MIST-origin writes"

        # Graph must not have been touched
        assert (
            graph_store.mark_orphaned_calls == []
        ), "mark_orphaned_by_provenance_path must not be called for MIST-origin writes"

        # Bus must not have published
        assert listener.events == [], "InvalidationBus must not publish for MIST-origin writes"

    @pytest.mark.asyncio
    async def test_bus_listener_receives_rebuild_result_for_cache_eviction(
        self, tmp_path: Path
    ) -> None:
        """End-to-end bus subscriber contract: listener receives RebuildResult.

        Verifies that a subscriber wired to evict a mist_context cache
        (mimicking ConversationHandler._on_vault_rebuild) receives the
        RebuildResult with the correct path, enabling targeted eviction.
        """
        vault_root = tmp_path / "vault"
        vault_root.mkdir()
        (vault_root / "users").mkdir()

        user_file = vault_root / "users" / "raj.md"
        user_file.write_text(_USER_FILE_INITIAL, encoding="utf-8")

        graph_store = FakeGraphStore()
        extraction_pipeline = FakeExtractionPipeline()
        bus = InvalidationBus()

        # Simulate ConversationHandler._on_vault_rebuild: evict sessions for raj
        evicted_sessions: list[str] = []

        async def _on_vault_rebuild(event: RebuildResult) -> None:
            """Evict mist_context cache for users/<user>.md rebuilds."""
            parts = event.path.parts
            if "users" in parts:
                user_id = event.path.stem
                # Evict sessions whose user_id matches the rebuilt file stem
                evicted_sessions.append(user_id)

        bus.subscribe(_on_vault_rebuild)

        vault_config = VaultConfig(enabled=True, root=str(vault_root), git_auto_init=False)
        writer = VaultWriter(config=vault_config)
        sidecar = _FakeSidecarIndex()
        fw = _build_filewatcher(vault_root, sidecar, graph_store, extraction_pipeline, bus, writer)

        # Simulate user edit
        user_file.write_text(_USER_FILE_EDITED, encoding="utf-8")
        await fw._do_reindex(str(user_file), is_mist_write=False)

        # Listener must have fired with the user_id "raj" (stem of users/raj.md)
        assert evicted_sessions == ["raj"], (
            f"Cache eviction listener must be called with user_id='raj'; "
            f"got evicted_sessions={evicted_sessions!r}"
        )
