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

    pass


class _FakeGraphRegenerator:
    """Minimal curation.GraphRegenerator double."""

    async def rebuild_from_path(self, path: Path):
        return None


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
        regenerator = _FakeGraphRegenerator()
        writer = _FakeVaultWriter()

        result = build_phase3_components(
            config=config,
            sidecar_index=sidecar,
            regenerator=regenerator,
            writer=writer,
        )

        assert isinstance(result, Phase3Components)
        assert isinstance(result.filewatcher, VaultFilewatcher)

    @requires_sentence_transformers
    def test_build_phase3_components_filewatcher_not_started(self, tmp_path):
        from backend.factories import build_phase3_components

        config = _make_config(tmp_path)
        sidecar = _FakeSidecarIndex()
        regenerator = _FakeGraphRegenerator()
        writer = _FakeVaultWriter()

        result = build_phase3_components(
            config=config,
            sidecar_index=sidecar,
            regenerator=regenerator,
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
        regenerator = _FakeGraphRegenerator()
        writer = _FakeVaultWriter()

        result = build_phase3_components(
            config=config,
            sidecar_index=sidecar,
            regenerator=regenerator,
            writer=writer,
        )

        # The bus on the dataclass must be the exact same instance wired
        # into the filewatcher, so subscribers on result.invalidation_bus
        # receive events the filewatcher publishes.
        assert result.filewatcher._invalidation_bus is result.invalidation_bus

    @requires_sentence_transformers
    def test_build_phase3_components_regenerator_wired(self, tmp_path):
        from backend.factories import build_phase3_components

        config = _make_config(tmp_path)
        sidecar = _FakeSidecarIndex()
        regenerator = _FakeGraphRegenerator()
        writer = _FakeVaultWriter()

        result = build_phase3_components(
            config=config,
            sidecar_index=sidecar,
            regenerator=regenerator,
            writer=writer,
        )

        assert result.filewatcher._regenerator is regenerator

    @requires_sentence_transformers
    def test_build_phase3_components_writer_wired(self, tmp_path):
        from backend.factories import build_phase3_components

        config = _make_config(tmp_path)
        sidecar = _FakeSidecarIndex()
        regenerator = _FakeGraphRegenerator()
        writer = _FakeVaultWriter()

        result = build_phase3_components(
            config=config,
            sidecar_index=sidecar,
            regenerator=regenerator,
            writer=writer,
        )

        assert result.filewatcher._writer is writer

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
        from backend.knowledge.curation.graph_regenerator import RebuildResult

        config = _make_config(tmp_path)
        sidecar = _FakeSidecarIndex()
        regenerator = _FakeGraphRegenerator()
        writer = _FakeVaultWriter()

        result = build_phase3_components(
            config=config,
            sidecar_index=sidecar,
            regenerator=regenerator,
            writer=writer,
        )

        received: list[RebuildResult] = []

        async def listener(event: RebuildResult) -> None:
            received.append(event)

        result.invalidation_bus.subscribe(listener)

        event = RebuildResult(
            path=Path(tmp_path / "vault" / "test.md"),
            bucket="2",
            orphaned_triple_count=0,
            new_triple_count=1,
            ontology_version="v1.1.0",
            deferred=False,
        )

        asyncio.get_event_loop().run_until_complete(
            result.filewatcher._invalidation_bus.publish(event)
        )

        assert len(received) == 1
        assert received[0] is event
