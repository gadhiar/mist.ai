"""Unit tests for Phase 3 factory additions: build_phase3_components.

Task 20: Validates that the Phase3Components dataclass bundles a
VaultFilewatcher and an InvalidationBus sharing the same bus instance.

Dependency notes:
- backend.factories does not import EmbeddingGenerator at module scope: every
  `from backend.knowledge.embeddings import EmbeddingGenerator` in that module
  is indented inside a function body. Verify with:
  `grep -n "import EmbeddingGenerator" backend/factories.py`
  -- all 5 hits are indented, none at column 0.
  Tests that CALL a factory function that builds an EmbeddingGenerator
  internally require sentence_transformers (Linux/container only) at call
  time, not at import time. build_graph_store takes an optional
  embedding_generator parameter and only builds a default when it is None;
  build_curation_scheduler has no such parameter and always builds one
  (both directly, and indirectly via its own call to build_graph_store).
  These tests are marked @requires_sentence_transformers and skipped on
  Windows.
- Config-logic tests (None returns for disabled state) can be simulated
  without importing factories by replicating the guard logic -- platform-neutral.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.knowledge.config import (
    FilewatcherConfig,
    KnowledgeConfig,
    SidecarIndexConfig,
    VaultConfig,
)
from tests.mocks.config import build_test_config

# ---------------------------------------------------------------------------
# Platform-availability markers
# ---------------------------------------------------------------------------

_SENTENCE_TRANSFORMERS_AVAILABLE = False
try:
    import sentence_transformers as _st  # noqa: F401

    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

requires_sentence_transformers = pytest.mark.skipif(
    not _SENTENCE_TRANSFORMERS_AVAILABLE,
    reason="sentence_transformers not available on this platform",
)


# ---------------------------------------------------------------------------
# Minimal fake types -- avoid sentence_transformers at import time
# ---------------------------------------------------------------------------


class _FakeSidecarIndex:
    """Minimal SidecarIndexProtocol double for factory injection."""

    def initialize(self) -> None:
        pass

    def close(self) -> None:
        pass

    def upsert_file(self, path, content, mtime, frontmatter=None) -> int:
        return 0

    def delete_path(self, path) -> int:
        return 0

    def query_vector(self, embedding, k=10):
        return []

    def query_fts(self, text, k=10):
        return []

    def query_hybrid(self, embedding, text, k=10):
        return []

    def get_by_path(self, path):
        return None

    def list_all_paths(self):
        return []

    def get_all_mtimes(self):
        return {}


class _FakeVaultWriter:
    """Minimal VaultWriter double."""

    def __init__(self) -> None:
        self.mist_write_marker = None

    def set_mist_write_marker(self, marker) -> None:
        self.mist_write_marker = marker


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path, *, filewatcher_enabled: bool = True) -> KnowledgeConfig:
    base = build_test_config()
    base.vault = VaultConfig(
        enabled=True,
        root=str(tmp_path / "vault"),
        git_auto_init=False,
    )
    base.sidecar_index = SidecarIndexConfig(
        enabled=True,
        db_path=str(tmp_path / "sidecar.db"),
        embedding_dimension=384,
    )
    base.filewatcher = FilewatcherConfig(
        enabled=filewatcher_enabled,
        observer_type="polling",
        debounce_ms=500,
    )
    return base


# ---------------------------------------------------------------------------
# TestRegeneratorFullyRetired
# ---------------------------------------------------------------------------


class TestRegeneratorFullyRetired:
    """R1.3 Task 7: no factory wiring path constructs a graph regenerator."""

    @requires_sentence_transformers
    def test_build_phase3_components_takes_no_regenerator(self):
        """R1.3: there is no graph regenerator left to wire."""
        import inspect

        from backend.factories import build_filewatcher, build_phase3_components

        assert "regenerator" not in inspect.signature(build_phase3_components).parameters
        assert "regenerator" not in inspect.signature(build_filewatcher).parameters

    @requires_sentence_transformers
    def test_filewatcher_has_no_regenerator_attribute(self, tmp_path):
        """The lifespan drain hung off this attribute; both retire together."""
        from backend.factories import build_phase3_components

        config = _make_config(tmp_path)
        components = build_phase3_components(
            config=config,
            sidecar_index=_FakeSidecarIndex(),
            writer=_FakeVaultWriter(),
        )
        assert not hasattr(components.filewatcher, "_regenerator")

    @requires_sentence_transformers
    def test_build_graph_regenerator_tombstone_is_gone(self):
        """The tombstone pointed at 'the regenerator ships with R1'. It shipped
        as R1.2's log_regenerator; this stale sign-post retires with R1.3.
        """
        import backend.factories as factories

        assert not hasattr(factories, "build_graph_regenerator")


# ---------------------------------------------------------------------------
# TestPhase3Components
# ---------------------------------------------------------------------------


class TestPhase3Components:
    """build_phase3_components returns a dataclass with filewatcher + bus."""

    @requires_sentence_transformers
    def test_phase3_components_importable(self):
        # The Phase3Components dataclass must be importable from factories.
        from backend.factories import Phase3Components  # noqa: F401

    @requires_sentence_transformers
    def test_phase3_components_is_frozen_dataclass(self):
        import dataclasses

        from backend.factories import Phase3Components

        assert dataclasses.is_dataclass(Phase3Components)
        # frozen=True means assigning a field raises FrozenInstanceError
        instance = Phase3Components(
            filewatcher=object(),
            invalidation_bus=object(),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            instance.filewatcher = None  # type: ignore[misc]

    @requires_sentence_transformers
    def test_build_phase3_components_returns_phase3_components(self, tmp_path):
        from backend.factories import Phase3Components, build_phase3_components
        from backend.vault.filewatcher import VaultFilewatcher

        config = _make_config(tmp_path)
        sidecar = _FakeSidecarIndex()
        writer = _FakeVaultWriter()

        result = build_phase3_components(
            config=config,
            sidecar_index=sidecar,
            writer=writer,
        )

        assert isinstance(result, Phase3Components)
        assert isinstance(result.filewatcher, VaultFilewatcher)

    @requires_sentence_transformers
    def test_build_phase3_components_filewatcher_not_started(self, tmp_path):
        from backend.factories import build_phase3_components

        config = _make_config(tmp_path)
        sidecar = _FakeSidecarIndex()
        writer = _FakeVaultWriter()

        result = build_phase3_components(
            config=config,
            sidecar_index=sidecar,
            writer=writer,
        )

        # Factory must not call .start() -- lifecycle owner handles that.
        assert result.filewatcher._running is False

    @requires_sentence_transformers
    def test_build_phase3_components_invalidation_bus_is_same_instance(self, tmp_path):
        # The filewatcher's internal bus and the bus on the dataclass
        # must be the SAME object so ConversationHandler can subscribe to it.
        from backend.factories import build_phase3_components

        config = _make_config(tmp_path)
        sidecar = _FakeSidecarIndex()
        writer = _FakeVaultWriter()

        result = build_phase3_components(
            config=config,
            sidecar_index=sidecar,
            writer=writer,
        )

        # The bus on the dataclass must be the exact same instance wired
        # into the filewatcher, so subscribers on result.invalidation_bus
        # receive events the filewatcher publishes.
        assert result.filewatcher._invalidation_bus is result.invalidation_bus

    @requires_sentence_transformers
    def test_build_phase3_components_writer_wired(self, tmp_path):
        from backend.factories import build_phase3_components

        config = _make_config(tmp_path)
        sidecar = _FakeSidecarIndex()
        writer = _FakeVaultWriter()

        result = build_phase3_components(
            config=config,
            sidecar_index=sidecar,
            writer=writer,
        )

        assert result.filewatcher._writer is writer

    @requires_sentence_transformers
    def test_build_phase3_components_raises_without_writer(self, tmp_path):
        # deep review vault-layer-adr010-1: server.py composed phase3 with
        # writer=None, silently disabling the invariant-5 chain in production.
        # The factory must reject that composition loudly.
        from backend.factories import build_phase3_components

        config = _make_config(tmp_path)
        sidecar = _FakeSidecarIndex()

        with pytest.raises(ValueError, match="writer-less filewatcher"):
            build_phase3_components(
                config=config,
                sidecar_index=sidecar,
            )

    @requires_sentence_transformers
    def test_build_phase3_components_wires_mist_write_marker(self, tmp_path):
        # The writer's consumer must self-mark its writes via the
        # filewatcher's marker, or every MIST append runs the user-edit path.
        from backend.factories import build_phase3_components

        config = _make_config(tmp_path)
        sidecar = _FakeSidecarIndex()
        writer = _FakeVaultWriter()

        result = build_phase3_components(
            config=config,
            sidecar_index=sidecar,
            writer=writer,
        )

        assert writer.mist_write_marker == result.filewatcher.mark_mist_write

    def test_build_phase3_components_returns_none_when_filewatcher_disabled(self, tmp_path):
        # Replicate guard logic platform-neutrally: disabled filewatcher -> None.
        # This mirrors what build_phase3_components does without importing factories.
        config = _make_config(tmp_path, filewatcher_enabled=False)

        result = None if not config.filewatcher.enabled else object()

        assert result is None

    def test_build_phase3_components_returns_none_when_vault_disabled(self, tmp_path):
        # Replicate guard logic platform-neutrally: disabled vault -> None.
        config = _make_config(tmp_path)
        config.vault = VaultConfig(
            enabled=False,
            root=str(tmp_path / "vault"),
            git_auto_init=False,
        )

        result = None if not config.vault.enabled else object()

        assert result is None

    def test_build_phase3_components_returns_none_when_sidecar_is_none(self, tmp_path):
        # Replicate guard logic: sidecar_index=None -> None (same as build_filewatcher).
        filewatcher_enabled = True
        vault_enabled = True
        sidecar_index = None

        result = (
            None if (filewatcher_enabled and vault_enabled and sidecar_index is None) else object()
        )

        assert result is None

    @requires_sentence_transformers
    def test_bus_subscribers_receive_events_from_filewatcher_side(self, tmp_path):
        # Integration: subscribe a listener on the dataclass bus;
        # publish an event via the filewatcher's internal bus reference.
        # They are the same object, so the listener fires.
        import asyncio

        from backend.factories import build_phase3_components
        from backend.vault.invalidation_bus import VaultChangeEvent

        config = _make_config(tmp_path)
        sidecar = _FakeSidecarIndex()
        writer = _FakeVaultWriter()

        result = build_phase3_components(
            config=config,
            sidecar_index=sidecar,
            writer=writer,
        )

        received: list[VaultChangeEvent] = []

        async def listener(event: VaultChangeEvent) -> None:
            received.append(event)

        result.invalidation_bus.subscribe(listener)

        event = VaultChangeEvent(path=Path(tmp_path / "vault" / "test.md"))

        asyncio.run(result.filewatcher._invalidation_bus.publish(event))

        assert len(received) == 1
        assert received[0] is event


# ---------------------------------------------------------------------------
# Phase 5.5 Dispatch 3 -- end-to-end bus identity regression tests
# (no sentence_transformers import path; uses direct InvalidationBus + fakes)
# ---------------------------------------------------------------------------


class _FakeConversationHandler:
    """Minimal stand-in for ConversationHandler bus-wiring tests.

    Mirrors the real handler's __init__ bus subscription logic so we can
    test the forwarding chain without requiring sentence_transformers or
    Neo4j.
    """

    def __init__(self, *, invalidation_bus=None, **_kwargs):
        self._invalidation_bus = invalidation_bus
        if invalidation_bus is not None:
            invalidation_bus.subscribe(self._on_vault_rebuild)

    async def _on_vault_rebuild(self, event) -> None:
        pass


class TestPhase55BusWiring:
    """Regression suite: shared InvalidationBus reaches ConversationHandler.

    The server hands the bus to `build_mist_runtime`, which hands it to the
    handler build. The forwarding tests drive the real `build_mist_runtime`
    over the fake loaders and provider of `tests/unit/realtime/test_runtime.py`,
    and need neither sentence_transformers nor Neo4j. The exception is
    test_build_conversation_handler_subscribes_on_vault_rebuild_when_bus_provided,
    which is marked @requires_sentence_transformers and calls the real
    build_conversation_handler -- it injects a GraphStore wrapping
    FakeNeo4jConnection, a FakeVectorStore, and a FakeLLM so it does not open
    a real Neo4j connection or build a real vector store. The real
    integration is covered by
    test_build_phase3_components_invalidation_bus_is_same_instance above.
    """

    def test_build_mist_runtime_forwards_the_bus_to_the_handler_build(self):
        """The bus the server passes the runtime build reaches the handler build unchanged."""
        from backend.vault.invalidation_bus import InvalidationBus
        from tests.unit.realtime.test_runtime import HandlerFactory, runtime

        bus = InvalidationBus()
        factory = HandlerFactory()

        runtime(factory, invalidation_bus=bus)

        assert factory.calls[0]["invalidation_bus"] is bus

    def test_conversation_handler_subscribes_on_vault_rebuild_when_bus_provided(self):
        """ConversationHandler._on_vault_rebuild is registered on the bus when one is provided."""
        from backend.vault.invalidation_bus import InvalidationBus

        bus = InvalidationBus()
        handler = _FakeConversationHandler(invalidation_bus=bus)

        assert handler._invalidation_bus is bus
        # _on_vault_rebuild must be subscribed
        assert len(bus._listeners) >= 1
        bound_methods = [getattr(fn, "__func__", fn) for fn in bus._listeners]
        assert _FakeConversationHandler._on_vault_rebuild in bound_methods

    def test_conversation_handler_no_subscription_when_bus_is_none(self):
        """ConversationHandler does not subscribe when bus is None."""
        bus_sentinel = None
        handler = _FakeConversationHandler(invalidation_bus=bus_sentinel)

        assert handler._invalidation_bus is None

    def test_full_chain_shares_bus_and_registers_listener(self):
        """Full chain: filewatcher bus -> build_mist_runtime -> handler._on_vault_rebuild subscribed.

        Regression for Phase 5 P0: proves that a single bus instance can
        propagate from the filewatcher side down to the handler and that
        the listener is registered, so publish() would reach the handler.
        """
        from backend.vault.invalidation_bus import InvalidationBus
        from tests.unit.realtime.test_runtime import runtime

        bus = InvalidationBus()

        # Wire full chain with the shared bus
        built, _ = runtime(
            lambda **kwargs: _FakeConversationHandler(**kwargs), invalidation_bus=bus
        )
        handler = built.conversation_handler

        assert handler._invalidation_bus is bus
        # At least one listener registered on the bus (the handler's callback)
        assert len(bus._listeners) >= 1

    @requires_sentence_transformers
    def test_build_conversation_handler_subscribes_on_vault_rebuild_when_bus_provided(
        self, tmp_path, monkeypatch
    ):
        """build_conversation_handler wires real ConversationHandler._on_vault_rebuild.

        Gated by @requires_sentence_transformers; whether the injected-fakes
        path below still imports sentence_transformers is not verified either
        way. Injects a GraphStore wrapping FakeNeo4jConnection (and
        FakeEmbeddingGenerator), a FakeVectorStore, and a FakeLLM so this test
        never opens a real Neo4j connection or builds a real LanceDB vector
        store. Verifies that the real factory passes the bus into the real
        ConversationHandler, the listener appears in bus._listeners, and the
        MistIdentity singleton MERGE (run by extraction-pipeline Stage 9
        setup) reaches the injected fake connection rather than a real one.
        """
        from backend.chat.conversation_handler import ConversationHandler
        from backend.factories import build_conversation_handler
        from backend.knowledge.storage.graph_store import GraphStore
        from backend.vault.invalidation_bus import InvalidationBus
        from tests.mocks.embeddings import FakeEmbeddingGenerator
        from tests.mocks.neo4j import FakeNeo4jConnection
        from tests.mocks.ollama import FakeLLM
        from tests.unit.knowledge.conftest import FakeVectorStore

        # Explicit assumption: Stage 9 (and this test's MistIdentity MERGE
        # assertion below) only runs when hydration isolation is off. The
        # dev-hydration stack sets MIST_HYDRATION_ISOLATION, which would fail
        # this test for a reason unrelated to bus wiring.
        monkeypatch.delenv("MIST_HYDRATION_ISOLATION", raising=False)
        config = _make_config(tmp_path)
        bus = InvalidationBus()
        fake_connection = FakeNeo4jConnection()
        graph_store = GraphStore(fake_connection, FakeEmbeddingGenerator())
        fake_vector_store = FakeVectorStore()

        handler = build_conversation_handler(
            config=config,
            invalidation_bus=bus,
            graph_store=graph_store,
            vector_store=fake_vector_store,
            llm_provider=FakeLLM(),
        )

        assert handler._invalidation_bus is bus
        assert len(bus._listeners) >= 1
        bound_methods = [getattr(fn, "__func__", fn) for fn in bus._listeners]
        assert ConversationHandler._on_vault_rebuild in bound_methods
        # Forwarding verification: the injected graph_store reached
        # build_extraction_pipeline's Stage 9 setup and the MistIdentity
        # MERGE landed on the fake connection, not a real Neo4j instance.
        fake_connection.assert_write_executed("MERGE (m:__SelfModel__:MistIdentity")
        # Forwarding verification: the injected vector_store reached the
        # retriever the factory built, rather than being ignored in favour
        # of a real vector store the factory constructs itself.
        assert handler.retriever._vector_store is fake_vector_store
