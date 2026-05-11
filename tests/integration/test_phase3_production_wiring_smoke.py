"""Phase 3 production wiring smoke test.

Root cause context (Phase 5 clean-room review): every Phase 3 P0 --
host_path typo, missing GraphStoreProtocol methods, missing extract_from_file,
orphan InvalidationBus, broken mist_admin argparse -- was caused by the absence
of any test exercising the REAL production wiring chain. factories.py was
un-importable on the Windows host (sentence_transformers unavailable), so every
factory-composition test on the host silently skipped. The existing ADR-010
invariant-5 contract test was green only because it used FakeGraphStore +
FakeExtractionPipeline that DO implement the protocols.

This test adds the protective regression: build_phase3_components +
build_conversation_handler + the chain through real GraphStore + real
ExtractionPipeline. The headline test simulates one vault user-edit (Bucket 1,
users/<user>.md -- no LLM call) and asserts the four-step invariant-5
coordination fires correctly:

  Step 1 -- authored_by transitions to user-edit on the edited file
  Step 2 -- old triples (DERIVED_FROM.path == edited_path) marked orphaned
  Step 3 -- new triples written from the updated content
  Step 4 -- _mist_context_cache evicted for the affected session

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
# Vault file content fixtures (mirror test_adr010_invariant5.py)
# ---------------------------------------------------------------------------

_USER_FILE_INITIAL = (
    "---\n"
    "type: mist-user\n"
    "user_id: smoke-test-user\n"
    "authored_by: mist\n"
    "last_updated: 2026-05-11\n"
    "---\n"
    "\n"
    "## Tools and Technologies\n"
    "- **python** (Technology)\n"
)

_USER_FILE_EDITED = (
    "---\n"
    "type: mist-user\n"
    "user_id: smoke-test-user\n"
    "authored_by: mist\n"
    "last_updated: 2026-05-11\n"
    "---\n"
    "\n"
    "## Tools and Technologies\n"
    "- **rust** (Technology)\n"
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
# Helpers -- unique Neo4j node prefix so parallel test runs don't collide
# ---------------------------------------------------------------------------


def _cleanup_smoke_nodes(graph_store, user_id: str, path: str) -> None:
    """Remove smoke-test nodes from Neo4j to leave the graph clean.

    Matches the __Entity__:User node, the __Provenance__:VaultNote node (keyed
    by path), and the target entity nodes written by Bucket 1 upsert_user via
    the natural path (id = "entity-python", "entity-rust").
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
        # Clean up target entity nodes written by upsert_user natural path.
        # upsert_user generates id = "entity-{display_name.lower()...}".
        for eid in ("entity-python", "entity-rust"):
            conn.execute_write(
                "MATCH (e:__Entity__ {id: $eid}) DETACH DELETE e",
                {"eid": eid},
            )
    except Exception:  # noqa: BLE001 -- cleanup; don't mask assertion failures
        pass


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
# Test 3: Full invariant-5 cycle against real GraphStore + real ExtractionPipeline
# ---------------------------------------------------------------------------


@requires_container
class TestVaultUserEditInvariant5ProductionWiring:
    """Headline regression: the four-step invariant-5 coordination fires correctly
    against real GraphStore (Neo4j) + real ExtractionPipeline.

    Exercises Bucket 1 (users/<user>.md) -- deterministic, no LLM call.

    This is the test that would have caught every Phase 3 P0:
    - host_path typo -> build_phase3_components would have failed at import
    - missing GraphStoreProtocol methods -> mark_orphaned_by_provenance_path call
    - missing extract_from_file -> Bucket 1 rebuild dispatches to pipeline
    - orphan InvalidationBus -> bus.publish would have raised
    - broken mist_admin argparse -> unrelated, but factory import itself
    """

    @pytest.mark.asyncio
    async def test_vault_user_edit_drives_full_invariant5_cycle(self, tmp_path):
        """Four-step invariant-5 coordination against real Neo4j.

        Arrange:
          - Build real GraphStore + real ExtractionPipeline via factories
          - Pre-seed a USES/python triple derived from a tmp users/ file
          - Build Phase3Components (real filewatcher + real InvalidationBus)
          - Build ConversationHandler and subscribe to bus
          - Pre-populate _mist_context_cache for the session

        Act:
          - Write new content to the user file (python -> rust)
          - Call filewatcher._do_reindex(path, is_mist_write=False) directly

        Assert:
          - Step 1: authored_by == user-edit in the file
          - Step 2: get_orphaned_provenance_paths includes the edited path
          - Step 3: USES/rust triple written to Neo4j
          - Step 4: _mist_context_cache cleared for the affected session
        """
        from backend.chat.mist_context import MistContext
        from backend.factories import (
            build_conversation_handler,
            build_extraction_pipeline,
            build_graph_store,
            build_phase3_components,
        )
        from backend.knowledge.config import (
            EventStoreConfig,
            FilewatcherConfig,
            KnowledgeConfig,
            SidecarIndexConfig,
            VaultConfig,
        )
        from backend.knowledge.curation.graph_regenerator import GraphRegenerator
        from backend.vault.writer import VaultWriter

        # ------------------------------------------------------------------
        # Arrange: vault directory structure
        # ------------------------------------------------------------------
        vault_root = tmp_path / "vault"
        vault_root.mkdir()
        (vault_root / "users").mkdir()

        user_file = vault_root / "users" / "smoke-test-user.md"
        user_file.write_text(_USER_FILE_INITIAL, encoding="utf-8")

        # ------------------------------------------------------------------
        # Arrange: config (read from env so NEO4J_URI is honoured in container)
        # ------------------------------------------------------------------
        config = KnowledgeConfig.from_env()
        config.vault = VaultConfig(
            enabled=True,
            root=str(vault_root),
            default_user_id="smoke-test-user",
            git_auto_init=False,
        )
        config.filewatcher = FilewatcherConfig(enabled=True, observer_type="polling")
        config.sidecar_index = SidecarIndexConfig(enabled=False)
        config.event_store = EventStoreConfig(enabled=False)

        # ------------------------------------------------------------------
        # Arrange: real GraphStore + real ExtractionPipeline
        # ------------------------------------------------------------------
        graph_store = build_graph_store(config)
        extraction_pipeline = build_extraction_pipeline(
            config,
            graph_store=graph_store,
            include_curation=False,
            include_internal_derivation=False,
        )

        # ------------------------------------------------------------------
        # Arrange: pre-seed via natural upsert_user path.
        #
        # upsert_user now writes DERIVED_FROM edges from each typed entity to
        # the VaultNote (Phase 5.5 Bucket 1 fix). mark_orphaned_by_provenance_path
        # queries that relationship-type, so the natural upsert path is all that
        # is needed. No manual Cypher seeding required.
        # ------------------------------------------------------------------
        from backend.knowledge.curation.bucket1_reader import ParsedUser

        _path_str = str(user_file)
        initial_parsed = ParsedUser(
            user_id="smoke-test-user",
            tools_and_technologies=["python"],
            expertise=[],
            currently_learning=[],
            projects=[],
            affiliations=[],
            interests=[],
            goals=[],
            preferences=[],
            people=[],
        )
        await graph_store.upsert_user(initial_parsed, derived_from_path=_path_str)

        # Confirm the provenance edge landed and is not already orphaned
        orphaned_before = await graph_store.get_orphaned_provenance_paths()
        assert _path_str not in orphaned_before, (
            f"Pre-condition: {_path_str} must not be orphaned before the edit; "
            f"orphaned_before={orphaned_before!r}"
        )

        # ------------------------------------------------------------------
        # Arrange: Phase3Components (shared bus) + VaultWriter
        # ------------------------------------------------------------------
        vault_writer = VaultWriter(
            config.vault,
            debug_logger=None,
            model_hash=config.model_hash,
        )

        regenerator = GraphRegenerator(
            graph_store=graph_store,
            extraction_pipeline=extraction_pipeline,
        )

        sidecar_stub = _NoopSidecarIndex()

        components = build_phase3_components(
            config=config,
            sidecar_index=sidecar_stub,
            regenerator=regenerator,
            writer=vault_writer,
        )
        assert (
            components is not None
        ), "build_phase3_components must return non-None when filewatcher + vault are enabled"

        # ------------------------------------------------------------------
        # Arrange: ConversationHandler subscribes to bus
        # ------------------------------------------------------------------
        handler = build_conversation_handler(
            config=config,
            invalidation_bus=components.invalidation_bus,
            vault_writer=vault_writer,
            vault_sidecar=None,
        )

        # ------------------------------------------------------------------
        # Arrange: pre-populate _mist_context_cache for the session
        # ------------------------------------------------------------------
        session_id = "smoke-session-001"
        handler.get_or_create_session(session_id, user_id="smoke-test-user")
        stub_ctx = MistContext(
            display_name="MIST",
            pronouns="she/her",
            self_concept="test stub",
            traits=[],
            capabilities=[],
            preferences=[],
        )
        handler._mist_context_cache[session_id] = stub_ctx
        assert (
            session_id in handler._mist_context_cache
        ), "Pre-condition: cache must be populated before the edit"

        # ------------------------------------------------------------------
        # Act: simulate user edit (python -> rust) and drive filewatcher
        # ------------------------------------------------------------------
        user_file.write_text(_USER_FILE_EDITED, encoding="utf-8")

        await components.filewatcher._do_reindex(str(user_file), is_mist_write=False)

        # ------------------------------------------------------------------
        # Assert Step 1: authored_by rewritten to user-edit
        # ------------------------------------------------------------------
        file_text = user_file.read_text(encoding="utf-8")
        assert "authored_by: user-edit" in file_text, (
            "Step 1 FAILED: authored_by must be rewritten to 'user-edit' after "
            f"user-edit detection; got file content:\n{file_text}"
        )

        # ------------------------------------------------------------------
        # Assert Step 2: old provenance path marked orphaned in Neo4j
        # ------------------------------------------------------------------
        orphaned_paths = await graph_store.get_orphaned_provenance_paths()
        assert str(user_file) in orphaned_paths, (
            f"Step 2 FAILED: get_orphaned_provenance_paths must include the edited path; "
            f"got orphaned_paths={orphaned_paths!r}"
        )

        # ------------------------------------------------------------------
        # Assert Step 3: new USES/rust triple written to Neo4j.
        #
        # upsert_user (Bucket 1) writes:
        #   (:__Entity__:User {id: "smoke-test-user"}) -[:USES]->
        #   (:__Entity__ {id: "entity-rust", display_name: "rust"})
        # The User node key is `id` (not `user_id`). Match on that schema.
        # ------------------------------------------------------------------
        conn = graph_store.connection
        rows = conn.execute_query(
            "MATCH (u:__Entity__:User {id: $uid})-[r:USES]->(t) "
            "WHERE toLower(t.display_name) = 'rust' "
            "RETURN count(r) AS cnt",
            {"uid": "smoke-test-user"},
        )
        rust_count = rows[0].get("cnt", 0) if rows else 0
        assert rust_count >= 1, (
            f"Step 3 FAILED: a USES->rust edge must exist for smoke-test-user after rebuild; "
            f"got cnt={rust_count}. Bucket 1 re-derivation via upsert_user may not have run."
        )

        # ------------------------------------------------------------------
        # Assert Step 4: _mist_context_cache evicted for the affected session
        # ------------------------------------------------------------------
        assert session_id not in handler._mist_context_cache, (
            f"Step 4 FAILED: _mist_context_cache must be evicted for session "
            f"'{session_id}' after users/smoke-test-user.md rebuild; "
            f"cache keys still present: {list(handler._mist_context_cache.keys())!r}"
        )

        # ------------------------------------------------------------------
        # Cleanup: remove smoke-test nodes from Neo4j
        # ------------------------------------------------------------------
        _cleanup_smoke_nodes(graph_store, "smoke-test-user", str(user_file))
