"""Phase 3 production wiring smoke test.

Root cause context (Phase 5 clean-room review): every Phase 3 P0 --
host_path typo, missing GraphStoreProtocol methods, missing extract_from_file,
orphan InvalidationBus, broken mist_admin argparse -- was caused by the absence
of any test exercising the REAL production wiring chain. factories.py was
un-importable on the Windows host (sentence_transformers unavailable), so every
factory-composition test on the host silently skipped.

This test adds the protective regression: build_phase3_components +
build_conversation_handler + the chain through a real GraphStore. R1.3
retired the graph-write half of the old invariant-5 chain (Inv-A1: the vault
is prose MIST reads, not a fact source) -- the headline test now asserts the
INVERSE of what it used to: a vault user-edit under users/<user>.md produces
zero graph edges and zero VaultNote provenance nodes, while the read-path
cache-invalidation signal still fires.

Skip gating: tests skip on any host where sentence_transformers is not
importable (Windows dev host) OR where Neo4j is not reachable at the URI
configured in NEO4J_URI (defaults to bolt://localhost:7687). Inside the
docker container both conditions are satisfied and the tests run to completion.

Run inside container:
    docker compose exec mist-backend python -m pytest
        tests/integration/test_phase3_production_wiring_smoke.py -v --tb=short
"""

from __future__ import annotations

import importlib.util
import os
import socket
from collections.abc import Callable
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Container-only skip gate
#
# Two conditions must both be true for the tests to run:
#   1. sentence_transformers is importable (rules out Windows host entirely)
#   2. Neo4j is reachable at the URI given by NEO4J_URI env var
#
# The existing conftest.skipif_no_services checks localhost:7687 which is
# correct for host-side developer runs but WRONG inside the Docker container
# (where Neo4j is mist-neo4j:7687). This gate parses NEO4J_URI directly so
# it works in both environments.
# ---------------------------------------------------------------------------


def _neo4j_reachable() -> bool:
    """Return True if Neo4j bolt port is reachable at NEO4J_URI."""
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    # Strip bolt:// or neo4j:// scheme
    host_port = uri.split("://", 1)[-1]
    host, _, port_str = host_port.partition(":")
    port = int(port_str) if port_str else 7687
    try:
        sock = socket.create_connection((host, port), timeout=3)
        sock.close()
        return True
    except (OSError, ConnectionRefusedError):
        return False


_SENTENCE_TRANSFORMERS_AVAILABLE = importlib.util.find_spec("sentence_transformers") is not None
_NEO4J_REACHABLE = _neo4j_reachable()

_SKIP_REASON_PARTS: list[str] = []
if not _SENTENCE_TRANSFORMERS_AVAILABLE:
    _SKIP_REASON_PARTS.append("sentence_transformers unavailable")
if not _NEO4J_REACHABLE:
    neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    _SKIP_REASON_PARTS.append(f"Neo4j not reachable at {neo4j_uri}")
_SKIP_REASON = "; ".join(_SKIP_REASON_PARTS) or "requirements met"

requires_container = pytest.mark.skipif(
    not (_SENTENCE_TRANSFORMERS_AVAILABLE and _NEO4J_REACHABLE),
    reason=f"requires container env: {_SKIP_REASON}",
)

# ---------------------------------------------------------------------------
# Minimal sidecar stub (no SQLite required; filewatcher just calls upsert_file)
# ---------------------------------------------------------------------------


class _NoopSidecarIndex:
    """Minimal SidecarIndexProtocol double that accepts all calls silently."""

    def initialize(self) -> None:
        pass

    def close(self) -> None:
        pass

    def upsert_file(self, path: str, content: str, mtime: int, frontmatter=None) -> int:
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


# ---------------------------------------------------------------------------
# Minimal writer stub -- build_phase3_components raises ValueError when the
# vault is enabled and writer= is None, so every call in this file needs one.
# ---------------------------------------------------------------------------


class _FakeVaultWriter:
    """Minimal VaultWriterProtocol double for the authored_by writeback surface.

    Records mark_authored_by_user_edit calls without driving VaultWriter's
    real queue-consumer lifecycle. set_mist_write_marker is a no-op: no test
    in this file drives the watchdog observer, so there is nothing for a
    self-marked write to suppress.
    """

    def __init__(self) -> None:
        self.authored_by_calls: list[Path] = []

    async def mark_authored_by_user_edit(self, path: Path) -> None:
        self.authored_by_calls.append(path)

    def set_mist_write_marker(self, marker: Callable[[str], None]) -> None:
        pass


# ---------------------------------------------------------------------------
# Helpers -- unique Neo4j node prefix so parallel test runs don't collide
# ---------------------------------------------------------------------------


def _cleanup_smoke_nodes(graph_store, user_id: str, path: str) -> None:
    """Remove this file's own synthetic test nodes from Neo4j.

    Matches only this test's own synthetic `user_id` (always a `smoke-r13-*`
    value, deliberately distinct from the real single-user graph's canonical
    `id: 'user'` node -- this query must never risk matching, let alone
    deleting, that node) and the `__Provenance__:VaultNote` keyed by this
    test's own path. This is best-effort hygiene for the shared dev graph,
    not the correctness guard: a regression that writes under a DIFFERENT id
    scheme (the actual failure mode a real reviewer caught -- see
    `_graph_counts` below) would leave residue this function cannot find by
    construction. The `entity-python`/`entity-rust` cleanup that used to live
    here matched the retired `upsert_user`'s naming scheme and is gone: no
    code path writes those ids anymore.
    """
    try:
        conn = graph_store.connection
        conn.execute_write(
            "MATCH (u:__Entity__:User {id: $uid}) DETACH DELETE u",
            {"uid": user_id},
        )
        conn.execute_write(
            "MATCH (vn:__Provenance__:VaultNote {path: $path}) DETACH DELETE vn",
            {"path": path},
        )
    except Exception:  # noqa: BLE001 -- cleanup; don't mask assertion failures
        pass


def _graph_counts(graph_store) -> tuple[int, int]:
    """Return (node_count, relationship_count) across the whole graph.

    The correctness guard for "a user-file edit writes nothing to the
    graph": a before/after delta around a single `_do_reindex` call is
    immune to the subject node's id scheme, edge direction, and exact
    provenance-path string -- a shape-anchored query like `MATCH (u
    {id: $user_id})-[r]->(t)` is not. A review mutation landed a real edge
    under the graph's actual canonical user id (`id: 'user'`, not this
    file's synthetic `smoke-r13-*` id) and a real inbound edge; both left
    the shape-anchored query empty while changing these counts.
    """
    conn = graph_store.connection
    nodes = conn.execute_query("MATCH (n) RETURN count(n) AS c", {})[0]["c"]
    rels = conn.execute_query("MATCH ()-[r]->() RETURN count(r) AS c", {})[0]["c"]
    return nodes, rels


# ---------------------------------------------------------------------------
# Test 1: Phase3Components composition
# ---------------------------------------------------------------------------


@requires_container
class TestPhase3ComponentsComposition:
    """Asserts build_phase3_components returns a correctly wired dataclass.

    No actual reindex is triggered. This test purely validates that the
    factory composes the filewatcher and bus with shared identity, which is
    the factory-level invariant that was untestable on the Windows host.
    """

    def test_returns_wired_filewatcher_and_bus_sharing_identity(self, tmp_path):
        """build_phase3_components returns Phase3Components with non-None bus
        and filewatcher._invalidation_bus is the SAME object as components.invalidation_bus.
        """
        from backend.factories import Phase3Components, build_phase3_components
        from backend.knowledge.config import FilewatcherConfig, KnowledgeConfig, VaultConfig

        vault_root = tmp_path / "vault"
        vault_root.mkdir()

        config = KnowledgeConfig.from_env()
        config.vault = VaultConfig(
            enabled=True,
            root=str(vault_root),
            git_auto_init=False,
        )
        config.filewatcher = FilewatcherConfig(enabled=True, observer_type="polling")

        sidecar_stub = _NoopSidecarIndex()

        components = build_phase3_components(
            config=config,
            sidecar_index=sidecar_stub,
            writer=_FakeVaultWriter(),
        )

        assert components is not None, (
            "build_phase3_components must return Phase3Components when filewatcher "
            "and vault are enabled and sidecar_index is not None"
        )
        assert isinstance(components, Phase3Components)

        assert (
            components.invalidation_bus is not None
        ), "Phase3Components.invalidation_bus must be non-None"
        assert components.filewatcher is not None, "Phase3Components.filewatcher must be non-None"

        # Identity check: filewatcher must hold the SAME bus instance
        assert components.filewatcher._invalidation_bus is components.invalidation_bus, (
            "filewatcher._invalidation_bus must be the exact same object as "
            "components.invalidation_bus; got different instances -- the bus "
            "shared-identity invariant is broken"
        )


# ---------------------------------------------------------------------------
# Test 2: ConversationHandler subscribes to the shared bus
# ---------------------------------------------------------------------------


@requires_container
class TestConversationHandlerBusSubscription:
    """Asserts that build_conversation_handler wires _on_vault_rebuild to the bus.

    The handler is built with the invalidation_bus from Phase3Components. After
    construction, the bus must have exactly one subscriber (the handler's
    _on_vault_rebuild coroutine).
    """

    def test_handler_subscribes_to_phase3_bus_when_passed_through_chain(self, tmp_path):
        """ConversationHandler registers _on_vault_rebuild on the shared bus.

        Exercises the full factory chain:
          build_phase3_components -> Phase3Components
          build_conversation_handler(invalidation_bus=components.invalidation_bus)
          -> handler._invalidation_bus is components.invalidation_bus
          -> bus._listeners contains handler._on_vault_rebuild
        """
        from backend.factories import build_conversation_handler, build_phase3_components
        from backend.knowledge.config import (
            EventStoreConfig,
            FilewatcherConfig,
            KnowledgeConfig,
            SidecarIndexConfig,
            VaultConfig,
        )

        vault_root = tmp_path / "vault"
        vault_root.mkdir()

        config = KnowledgeConfig.from_env()
        config.vault = VaultConfig(enabled=True, root=str(vault_root), git_auto_init=False)
        config.filewatcher = FilewatcherConfig(enabled=True, observer_type="polling")
        config.sidecar_index = SidecarIndexConfig(enabled=False)
        config.event_store = EventStoreConfig(enabled=False)

        sidecar_stub = _NoopSidecarIndex()

        components = build_phase3_components(
            config=config,
            sidecar_index=sidecar_stub,
            writer=_FakeVaultWriter(),
        )
        assert components is not None

        handler = build_conversation_handler(
            config=config,
            invalidation_bus=components.invalidation_bus,
            vault_writer=None,
            vault_sidecar=None,
        )

        # The handler must have stored the shared bus
        assert handler._invalidation_bus is components.invalidation_bus, (
            "handler._invalidation_bus must be the same instance passed to "
            "build_conversation_handler; got a different object"
        )

        # The bus must have at least one listener (handler._on_vault_rebuild)
        listeners = components.invalidation_bus._listeners
        assert len(listeners) >= 1, (
            f"InvalidationBus must have at least one listener after "
            f"build_conversation_handler; got {len(listeners)}"
        )

        # Verify the registered listener is the handler's method
        listener_names = [getattr(fn, "__name__", repr(fn)) for fn in listeners]
        assert "_on_vault_rebuild" in listener_names, (
            f"handler._on_vault_rebuild must be registered on the bus; "
            f"registered listeners: {listener_names!r}"
        )


# ---------------------------------------------------------------------------
# Shared config builder (Test 3 only -- Tests 1/2 keep their pre-existing
# inline config construction, which is deliberately left untouched)
# ---------------------------------------------------------------------------


def _smoke_config(tmp_path: Path):
    """Vault + filewatcher enabled over a temp vault root, real Neo4j from env."""
    from backend.knowledge.config import FilewatcherConfig, KnowledgeConfig, VaultConfig

    vault_root = tmp_path / "vault"
    vault_root.mkdir(exist_ok=True)
    config = KnowledgeConfig.from_env()
    config.vault = VaultConfig(enabled=True, root=str(vault_root), git_auto_init=False)
    config.filewatcher = FilewatcherConfig(enabled=True, observer_type="polling")
    return config


# ---------------------------------------------------------------------------
# Test 3: a real user-file edit produces no graph write, against real Neo4j
# ---------------------------------------------------------------------------


@requires_container
class TestVaultUserEditWritesNoGraphFactsProduction:
    """R1.3 against a real Neo4j: a user-file edit leaves the graph untouched.

    The predecessor asserted the opposite -- that the edit produced USES /
    EXPERT_IN edges via upsert_user. That was the vault->graph fact path
    Inv-A1 retires.
    """

    @pytest.mark.asyncio
    async def test_vault_user_edit_produces_no_graph_edges(self, tmp_path):
        from backend.factories import build_graph_store, build_phase3_components

        graph_store = build_graph_store(_smoke_config(tmp_path))
        user_id = "smoke-r13-user"
        target = tmp_path / "vault" / "users" / f"{user_id}.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "---\n"
            f"user_id: {user_id}\n"
            "---\n\n"
            "## Tools and Technologies\n"
            "- SmokeTestTechnology\n",
            encoding="utf-8",
        )
        _cleanup_smoke_nodes(graph_store, user_id, str(target))
        try:
            components = build_phase3_components(
                config=_smoke_config(tmp_path),
                sidecar_index=_NoopSidecarIndex(),
                writer=_FakeVaultWriter(),
            )
            before = _graph_counts(graph_store)

            await components.filewatcher._do_reindex(str(target), is_mist_write=False)

            # Diagnostics: narrower shape checks against this test's own
            # synthetic user_id and exact path. These give a specific,
            # readable message for the regression shape the retired
            # upsert_user actually wrote (outbound edge from the frontmatter
            # user_id, VaultNote at the exact edited path). They are not the
            # guard -- see the count-delta assert below.
            rows = graph_store.connection.execute_query(
                "MATCH (u:__Entity__ {id: $user_id})-[r]->(t) RETURN type(r) AS rel_type",
                {"user_id": user_id},
            )
            assert rows == [], (
                "R1.3 diagnostic: a user-file edit wrote an outbound edge from "
                f"id={user_id!r}; found {[r['rel_type'] for r in rows]}"
            )

            vault_notes = graph_store.connection.execute_query(
                "MATCH (vn:__Provenance__:VaultNote {path: $path}) RETURN vn.path AS path",
                {"path": str(target)},
            )
            assert (
                vault_notes == []
            ), "R1.3 diagnostic: a VaultNote provenance node exists for the edited file"

            # The actual guard: any change in total node/relationship counts
            # proves a graph write occurred, regardless of subject id scheme,
            # edge direction, or exact path string -- see _graph_counts.
            after = _graph_counts(graph_store)
            assert after == before, (
                "R1.3: a user-file edit must write nothing to the graph; "
                f"node/relationship counts changed from {before} to {after}"
            )
        finally:
            _cleanup_smoke_nodes(graph_store, user_id, str(target))
            graph_store.close()

    @pytest.mark.asyncio
    async def test_vault_user_edit_still_evicts_the_read_path_cache(self, tmp_path):
        """The kept half of the chain: a vault edit still evicts the cached persona.

        Walks the full production chain -- filewatcher publish -> the real
        ConversationHandler._on_vault_rebuild -> _mist_context_cache eviction
        -- not just the bus hop, so a regression that breaks the handler's
        subscription or its path-to-user_id matching still fails this test.
        """
        from backend.chat.mist_context import MistContext
        from backend.factories import (
            build_conversation_handler,
            build_graph_store,
            build_phase3_components,
        )

        config = _smoke_config(tmp_path)
        graph_store = build_graph_store(config)
        user_id = "smoke-r13-user"
        target = tmp_path / "vault" / "users" / f"{user_id}.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"---\nuser_id: {user_id}\n---\n\nEdited.\n", encoding="utf-8")

        _cleanup_smoke_nodes(graph_store, user_id, str(target))
        try:
            components = build_phase3_components(
                config=config,
                sidecar_index=_NoopSidecarIndex(),
                writer=_FakeVaultWriter(),
            )
            handler = build_conversation_handler(
                config=config,
                invalidation_bus=components.invalidation_bus,
                vault_writer=None,
                vault_sidecar=None,
            )

            session_id = "smoke-r13-session"
            handler.get_or_create_session(session_id, user_id=user_id)
            handler._mist_context_cache[session_id] = MistContext(
                display_name="MIST",
                pronouns="she/her",
                self_concept="test stub",
                traits=[],
                capabilities=[],
                preferences=[],
            )
            assert (
                session_id in handler._mist_context_cache
            ), "pre-condition: cache must be populated before the edit"

            await components.filewatcher._do_reindex(str(target), is_mist_write=False)

            assert session_id not in handler._mist_context_cache, (
                f"R1.3: _mist_context_cache must be evicted for session {session_id!r} "
                "after a vault user-edit -- the read-path cache-invalidation signal "
                "must still fire even though the graph-write half of the chain retired"
            )
        finally:
            _cleanup_smoke_nodes(graph_store, user_id, str(target))
            graph_store.close()
