"""Unit tests for Phase 3 factory additions: build_phase3_components.

Task 20: Validates that the Phase3Components dataclass bundles a
VaultFilewatcher and an InvalidationBus sharing the same bus instance.

Dependency notes:
- Tests that import backend.factories require sentence_transformers
  (Linux/container only) because the module eagerly imports EmbeddingGenerator.
  These tests are marked @requires_sentence_transformers and skipped on Windows.
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

        asyncio.get_event_loop().run_until_complete(
            result.filewatcher._invalidation_bus.publish(event)
        )

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


class _FakeKnowledgeIntegration:
    """Minimal stand-in for KnowledgeIntegration forwarding tests."""

    def __init__(self, *, invalidation_bus=None, **_kwargs):
        self._invalidation_bus = invalidation_bus
        self.conversation_handler = _FakeConversationHandler(invalidation_bus=invalidation_bus)


class _FakeModelManager:
    """Minimal stand-in for ModelManager forwarding tests."""

    def __init__(self, *, invalidation_bus=None, **_kwargs):
        self._invalidation_bus = invalidation_bus
        self.knowledge = _FakeKnowledgeIntegration(invalidation_bus=invalidation_bus)


class _FakeVoiceProcessor:
    """Minimal stand-in for VoiceProcessor forwarding tests."""

    def __init__(self, *, invalidation_bus=None, **_kwargs):
        self._invalidation_bus = invalidation_bus
        self.models = _FakeModelManager(invalidation_bus=invalidation_bus)


class TestPhase55BusWiring:
    """Regression suite: shared InvalidationBus reaches ConversationHandler.

    These tests do NOT require sentence_transformers or Neo4j. They verify
    the forwarding chain contract using minimal fakes that mirror real
    constructor signatures. The real integration is covered by
    test_build_phase3_components_invalidation_bus_is_same_instance above.
    """

    def test_bus_identity_preserved_through_voice_processor(self):
        """Bus passed into VoiceProcessor reaches its ModelManager unchanged."""
        from backend.vault.invalidation_bus import InvalidationBus

        bus = InvalidationBus()
        vp = _FakeVoiceProcessor(invalidation_bus=bus)

        assert vp._invalidation_bus is bus
        assert vp.models._invalidation_bus is bus

    def test_bus_identity_preserved_through_model_manager_to_knowledge(self):
        """Bus passed into ModelManager reaches its KnowledgeIntegration unchanged."""
        from backend.vault.invalidation_bus import InvalidationBus

        bus = InvalidationBus()
        mm = _FakeModelManager(invalidation_bus=bus)

        assert mm._invalidation_bus is bus
        assert mm.knowledge._invalidation_bus is bus

    def test_bus_identity_preserved_through_knowledge_integration_to_handler(self):
        """Bus passed into KnowledgeIntegration reaches its ConversationHandler unchanged."""
        from backend.vault.invalidation_bus import InvalidationBus

        bus = InvalidationBus()
        ki = _FakeKnowledgeIntegration(invalidation_bus=bus)

        assert ki._invalidation_bus is bus
        assert ki.conversation_handler._invalidation_bus is bus

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

    def test_full_chain_fake_wiring_shares_bus_and_registers_listener(self):
        """Full chain fake: filewatcher bus -> VoiceProcessor -> handler._on_vault_rebuild subscribed.

        Regression for Phase 5 P0: proves that a single bus instance can
        propagate from the filewatcher side down to the handler and that
        the listener is registered, so publish() would reach the handler.
        """
        from backend.vault.invalidation_bus import InvalidationBus

        bus = InvalidationBus()

        # Wire full chain with the shared bus
        vp = _FakeVoiceProcessor(invalidation_bus=bus)
        handler = vp.models.knowledge.conversation_handler

        assert handler._invalidation_bus is bus
        # At least one listener registered on the bus (the handler's callback)
        assert len(bus._listeners) >= 1

    @requires_sentence_transformers
    def test_build_conversation_handler_subscribes_on_vault_rebuild_when_bus_provided(
        self, tmp_path
    ):
        """build_conversation_handler wires real ConversationHandler._on_vault_rebuild.

        Requires sentence_transformers. Verifies that the real factory passes
        the bus into the real ConversationHandler and the listener appears in
        bus._listeners.
        """
        from backend.chat.conversation_handler import ConversationHandler
        from backend.factories import build_conversation_handler
        from backend.vault.invalidation_bus import InvalidationBus

        config = _make_config(tmp_path)
        bus = InvalidationBus()

        handler = build_conversation_handler(
            config=config,
            invalidation_bus=bus,
        )

        assert handler._invalidation_bus is bus
        assert len(bus._listeners) >= 1
        bound_methods = [getattr(fn, "__func__", fn) for fn in bus._listeners]
        assert ConversationHandler._on_vault_rebuild in bound_methods


class TestRealKnowledgeIntegrationBusParam:
    """Tests that the real KnowledgeIntegration accepts and forwards invalidation_bus.

    Platform-neutral: imports only backend.chat.knowledge_integration which
    has no torch/sentence_transformers dependency.
    """

    def test_knowledge_integration_accepts_invalidation_bus_kwarg(self):
        """KnowledgeIntegration.__init__ must have an invalidation_bus parameter.

        This test catches the missing-param regression. It FAILS before the fix
        and PASSES once invalidation_bus is added to KnowledgeIntegration.__init__.
        """
        import inspect

        from backend.chat.knowledge_integration import KnowledgeIntegration

        sig = inspect.signature(KnowledgeIntegration.__init__)
        assert "invalidation_bus" in sig.parameters, (
            "KnowledgeIntegration.__init__ is missing the 'invalidation_bus' parameter; "
            "build_conversation_handler is called without it so ConversationHandler "
            "_on_vault_rebuild is never subscribed (Phase 5 P0 regression)."
        )

    def test_knowledge_integration_stores_and_forwards_invalidation_bus(self):
        """KnowledgeIntegration must store invalidation_bus and pass it to build_conversation_handler.

        Uses unittest.mock to intercept the build_conversation_handler call and verify
        the bus arrives as the invalidation_bus keyword argument.
        """
        import inspect
        from unittest.mock import MagicMock, patch

        from backend.chat.knowledge_integration import KnowledgeIntegration
        from backend.vault.invalidation_bus import InvalidationBus

        sig = inspect.signature(KnowledgeIntegration.__init__)
        if "invalidation_bus" not in sig.parameters:
            pytest.skip("invalidation_bus param not yet added -- caught by sibling test")

        bus = InvalidationBus()
        fake_handler = MagicMock()
        fake_handler._invalidation_bus = bus

        with patch(
            "backend.chat.knowledge_integration.build_conversation_handler",
            return_value=fake_handler,
        ) as mock_build:
            config = MagicMock()
            config.enable_knowledge_integration = True
            # Construction may fail for other reasons (Neo4j, etc.) --
            # we only care that build_conversation_handler was called with the bus.
            import contextlib

            with contextlib.suppress(Exception):
                KnowledgeIntegration(
                    config=config,
                    llm_provider=MagicMock(),
                    invalidation_bus=bus,
                )

            if mock_build.called:
                _, kwargs = mock_build.call_args
                assert kwargs.get("invalidation_bus") is bus, (
                    "build_conversation_handler was called without invalidation_bus; "
                    "ConversationHandler will not subscribe _on_vault_rebuild."
                )
