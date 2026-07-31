"""Vault integration tests for ConversationHandler (Cluster 8 Phase 5).

R1.3.1: the per-turn vault append retired with the DERIVED_FROM->VaultNote
contract (ADR-011 amended). The vault is now written once, at session end,
via `end_session` -> `SessionSynthesizer.synthesize` -> `VaultWriter.write_session_note`.
These tests cover that path, failure isolation per ADR-010 Invariant 6, and
that `_derive_session_slug` / `_derive_session_slug_from_utterance` normalize
session identifiers correctly.

Uses FakeVaultWriter (defined inline) to record write calls without
touching the filesystem.
"""

import asyncio

import pytest

from backend.chat.conversation_handler import ConversationHandler
from backend.knowledge.retrieval.knowledge_retriever import KnowledgeRetriever
from backend.knowledge.storage.graph_store import GraphStore
from tests.mocks.config import build_test_config
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeNeo4jConnection
from tests.mocks.ollama import FakeLLM
from tests.unit.conftest import make_test_conventions_loader

# ---------------------------------------------------------------------------
# FakeVaultWriter test double
# ---------------------------------------------------------------------------


class FakeVaultWriter:
    """In-memory vault writer test double. Records all write calls.

    Shaped after `VaultWriterProtocol` post-R1.3.1: the only session-note
    write operation is `write_session_note`; the per-turn append surface
    (`append_turn_to_session`, `update_entities_extracted`,
    `append_session_synthesis`, `mark_session_completed`, `peek_turn_count`)
    is gone from the real writer and therefore gone from this double too --
    keeping it would let a retired call silently pass by matching stale
    method names (L20).
    """

    def __init__(self):
        self.start_calls: int = 0
        self.stop_calls: int = 0
        self.write_session_note_calls: list[dict] = []
        self.upsert_user_calls: list[dict] = []  # records (user_id, body_markdown)
        # C-pattern writeback now targets the derived snapshot, not user.md.
        self.upsert_user_snapshot_calls: list[dict] = []
        self.fail_on_write: bool = False  # Toggle per test to simulate failure

    async def start(self) -> None:
        self.start_calls += 1

    async def stop(self) -> None:
        self.stop_calls += 1

    def session_path(self, date: str, slug: str) -> str:
        return f"/tmp/vault/sessions/{date}-{slug}.md"

    async def write_session_note(
        self,
        vault_note_path: str,
        synthesis,
        related_entities: list[str] | None = None,
        status: str = "completed",
    ) -> str | None:
        if self.fail_on_write:
            raise RuntimeError("simulated vault write failure")
        self.write_session_note_calls.append(
            {
                "vault_note_path": vault_note_path,
                "synthesis": synthesis,
                "related_entities": related_entities or [],
                "status": status,
            }
        )
        return vault_note_path

    async def upsert_identity(self, *args, **kwargs) -> str:
        return ""

    async def upsert_user(self, *args, **kwargs) -> str:
        # Record both positional and keyword forms so tests can assert.
        record: dict = {}
        if args:
            if len(args) >= 1:
                record["user_id"] = args[0]
            if len(args) >= 2:
                record["body_markdown"] = args[1]
        if "user_id" in kwargs:
            record["user_id"] = kwargs["user_id"]
        if "body_markdown" in kwargs:
            record["body_markdown"] = kwargs["body_markdown"]
        self.upsert_user_calls.append(record)
        return f"/tmp/vault/users/{record.get('user_id', 'user')}.md"

    async def upsert_user_snapshot(self, *args, **kwargs) -> str:
        # C-pattern derived snapshot writeback. Records both positional and
        # keyword forms; targets the <user_id>-graph-snapshot.md stem.
        record: dict = {}
        if args:
            if len(args) >= 1:
                record["user_id"] = args[0]
            if len(args) >= 2:
                record["body_markdown"] = args[1]
        if "user_id" in kwargs:
            record["user_id"] = kwargs["user_id"]
        if "body_markdown" in kwargs:
            record["body_markdown"] = kwargs["body_markdown"]
        self.upsert_user_snapshot_calls.append(record)
        return f"/tmp/vault/users/{record.get('user_id', 'user')}-graph-snapshot.md"


# ---------------------------------------------------------------------------
# Test doubles for ExtractionPipeline
# ---------------------------------------------------------------------------


class FakeExtractionPipeline:
    """Minimal extraction pipeline. Returns one synthetic entity by default.

    Tests that need the zero-extraction path explicitly should pass
    `entities=[]` and `relationships=[]` to override.
    """

    def __init__(
        self,
        entities: list[dict] | None = None,
        relationships: list[dict] | None = None,
    ):
        self.calls: list[dict] = []
        self._entities = (
            entities
            if entities is not None
            else [{"entity_id": "synthetic", "entity_type": "Concept", "display_name": "Synthetic"}]
        )
        self._relationships = relationships if relationships is not None else []

    async def extract_from_utterance(self, **kwargs):
        self.calls.append(kwargs)
        from backend.knowledge.extraction.validator import ValidationResult

        return ValidationResult(
            valid=True, entities=self._entities, relationships=self._relationships
        )


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _make_retriever(config, gs):
    return KnowledgeRetriever(config=config, graph_store=gs)


def make_handler(vault_writer=None, event_store_enabled: bool = True):
    """Construct a ConversationHandler suitable for vault integration tests.

    Default `event_store_enabled=True`: `end_session` (R1.3.1) synthesizes
    the session note from event-store turns, so most vault-write tests need
    the event store enabled to exercise that path. Tests that explicitly
    want the no-event-store path can override.
    """
    conn = FakeNeo4jConnection()
    gs = GraphStore(conn, FakeEmbeddingGenerator())
    config = build_test_config(
        event_store_enabled=event_store_enabled,
        event_store_db_path=":memory:",
    )
    pipeline = FakeExtractionPipeline()
    return ConversationHandler(
        config=config,
        graph_store=gs,
        extraction_pipeline=pipeline,
        retriever=_make_retriever(config, gs),
        llm_provider=FakeLLM(),
        conventions_loader=make_test_conventions_loader(),
        vault_writer=vault_writer,
    )


# ---------------------------------------------------------------------------
# TestExtractionTaskLifecycle (deep review concurrency-async-4 / -8)
# ---------------------------------------------------------------------------


class TestExtractionTaskLifecycle:
    @pytest.mark.asyncio
    async def test_end_session_pops_vault_path_for_fresh_note(self):
        # A resumed conversation must allocate a fresh note; re-rendering
        # over an already-written note corrupts the record.
        fake_vault = FakeVaultWriter()
        handler = make_handler(vault_writer=fake_vault)
        await handler.handle_message(user_message="Working on MIST today", session_id="s1")
        await handler._drain_extraction_tasks()
        path = handler._vault_paths["s1"]

        await handler.end_session("s1")

        assert [c["vault_note_path"] for c in fake_vault.write_session_note_calls] == [path]
        assert "s1" not in handler._vault_paths

    @pytest.mark.asyncio
    async def test_end_session_drains_session_extraction_tasks_first(self):
        # Background extraction must not land AFTER the note write when a
        # reconnect fires end_session mid-turn.
        order: list[str] = []

        class _OrderedVault(FakeVaultWriter):
            async def write_session_note(self, **kwargs):
                order.append("written")
                return await super().write_session_note(**kwargs)

        fake_vault = _OrderedVault()
        handler = make_handler(vault_writer=fake_vault)
        await handler.handle_message(user_message="Working on MIST today", session_id="s1")
        await handler._drain_extraction_tasks()

        async def _slow_extraction():
            await asyncio.sleep(0.05)
            order.append("extraction")

        task = asyncio.create_task(_slow_extraction())
        handler._extraction_tasks[task] = "s1"

        await handler.end_session("s1")

        assert order == ["extraction", "written"]

    @pytest.mark.asyncio
    async def test_aclose_drains_all_inflight_extractions(self):
        # Shutdown must drain, not abandon: cancellation mid commit-protocol
        # retires a belief without writing its successor.
        handler = make_handler(vault_writer=FakeVaultWriter())
        flags: list[str] = []

        async def _bg(name: str):
            await asyncio.sleep(0.02)
            flags.append(name)

        for name, sid in (("a", "s1"), ("b", "s2")):
            t = asyncio.create_task(_bg(name))
            handler._extraction_tasks[t] = sid

        await handler.aclose()

        assert sorted(flags) == ["a", "b"]


class TestSessionEnd:
    """R1.3.1: session-end signal renders one session note.

    On WebSocket disconnect or other end-of-session trigger, the
    ConversationHandler synthesizes the session from its event-store turns
    and writes exactly one note via `VaultWriter.write_session_note`.
    """

    @pytest.mark.asyncio
    async def test_end_session_writes_note_for_specific_session(self):
        fake_vault = FakeVaultWriter()
        handler = make_handler(vault_writer=fake_vault)
        # Two turns so the session clears the synthesis threshold cleanly.
        await handler.handle_message(user_message="Working on MIST today", session_id="end-1")
        await handler.handle_message(user_message="Wired up the vault writer", session_id="end-1")
        await handler._drain_extraction_tasks()
        path = handler._vault_paths["end-1"]

        await handler.end_session("end-1")

        assert [c["vault_note_path"] for c in fake_vault.write_session_note_calls] == [
            path
        ], f"expected write_session_note once with {path}, got {fake_vault.write_session_note_calls}"

    @pytest.mark.asyncio
    async def test_end_session_with_no_arg_ends_all_active_sessions(self):
        fake_vault = FakeVaultWriter()
        handler = make_handler(vault_writer=fake_vault)
        await handler.handle_message(user_message="First session start", session_id="sess-a")
        await handler.handle_message(user_message="Second session start", session_id="sess-b")
        await handler._drain_extraction_tasks()
        path_a = handler._vault_paths["sess-a"]
        path_b = handler._vault_paths["sess-b"]

        await handler.end_session()  # no arg = all

        written_paths = [c["vault_note_path"] for c in fake_vault.write_session_note_calls]
        assert sorted(written_paths) == sorted([path_a, path_b])

    @pytest.mark.asyncio
    async def test_end_session_no_op_when_session_never_had_vault_path(self):
        fake_vault = FakeVaultWriter()
        handler = make_handler(vault_writer=fake_vault)
        # No turns sent for this session, so no path allocated
        await handler.end_session("never-touched")
        assert fake_vault.write_session_note_calls == []

    @pytest.mark.asyncio
    async def test_end_session_pops_path_and_warns_when_no_turns_available(self, caplog):
        """A session with an allocated vault path but zero event-store turns
        (no `handle_message` ever ran for it -- e.g. a stale or manually
        seeded path) still gets its path evicted. end_session means the
        session is over regardless of whether a note gets written; leaving
        the entry behind would both grow `_vault_paths` unboundedly and let
        a later reconnect under the same session_id silently reuse a path
        whose note was never written. The gap is logged at warning, not
        silently swallowed at the below-threshold debug level, because it
        signals a structural problem (no event-store turns reachable) rather
        than a quiet session.
        """
        import logging

        fake_vault = FakeVaultWriter()
        handler = make_handler(vault_writer=fake_vault)
        handler._vault_paths["ghost-session"] = "/tmp/vault/sessions/2026-07-31-ghost.md"

        with caplog.at_level(logging.WARNING, logger="backend.chat.conversation_handler"):
            await handler.end_session("ghost-session")

        assert "ghost-session" not in handler._vault_paths
        assert fake_vault.write_session_note_calls == []
        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            "no event-store turns available" in msg for msg in warning_messages
        ), f"Expected a no-turns-available warning. Got: {warning_messages}"

    @pytest.mark.asyncio
    async def test_end_session_no_op_when_vault_writer_none(self):
        handler = make_handler(vault_writer=None)
        # Should not raise
        await handler.end_session("any-session")


class TestUserVaultCPatternTrigger:
    """ADR-011 bucket 1 / C-pattern: the derived user snapshot
    (users/<user_id>-graph-snapshot.md) re-renders iff extraction touched
    user-scope (User entity OR User-source/target edge). The hand-curated
    users/<user_id>.md is user-authoritative (ADR-010 Invariant 5) and is
    NEVER written by this path.

    The C-pattern is the design call from 2026-05-06: don't re-render on
    every turn (noisy, churns filewatcher); don't make MIST decide via
    tool-call (mechanical, not cognitive); DO trigger on graph delta
    (deterministic, intentional). Unrelated to the R1.3.1 per-turn-append
    removal -- this trigger fires from `_maybe_refresh_user_vault`, not the
    retired session-note append path.
    """

    @pytest.mark.asyncio
    async def test_user_vault_re_renders_when_extraction_touches_user_source(self):
        # Arrange: extraction pipeline that returns a CurationResult-like
        # object with a User-source relationship in validated_relationships.
        from backend.knowledge.curation.deduplication import DeduplicationResult
        from backend.knowledge.curation.graph_writer import WriteResult
        from backend.knowledge.curation.pipeline import CurationResult
        from backend.knowledge.curation.reconciliation import ReconcileTurnResult

        class FakeUserScopePipeline:
            async def extract_from_utterance(self, **kwargs):
                return CurationResult(
                    write_result=WriteResult(),
                    dedup_result=DeduplicationResult(
                        entities=[], merge_actions=[], entities_merged=0
                    ),
                    reconcile_result=ReconcileTurnResult(),
                    curation_time_ms=1.0,
                    validated_entities=[],
                    validated_relationships=[
                        {"source": "user", "target": "python", "type": "USES"}
                    ],
                )

        fake_vault = FakeVaultWriter()
        handler = make_handler(vault_writer=fake_vault, event_store_enabled=True)
        handler._extraction_pipeline = FakeUserScopePipeline()

        # Act
        await handler.handle_message(
            user_message="I use Python every day", session_id="user-scope-sess"
        )
        await asyncio.sleep(0.1)  # let fire-and-forget extraction settle

        # Assert: the C-pattern wrote the DERIVED snapshot exactly once and
        # never touched the curated users/<uid>.md (upsert_user).
        assert len(fake_vault.upsert_user_snapshot_calls) == 1, (
            "expected 1 upsert_user_snapshot call, got "
            f"{len(fake_vault.upsert_user_snapshot_calls)}"
        )
        assert (
            len(fake_vault.upsert_user_calls) == 0
        ), "C-pattern must NOT write the curated user.md (upsert_user)"
        call = fake_vault.upsert_user_snapshot_calls[0]
        assert call["user_id"] == "user"
        assert "# " in call["body_markdown"], "body should be a rendered markdown body"

    @pytest.mark.asyncio
    async def test_user_vault_does_not_re_render_when_extraction_misses_user(self):
        from backend.knowledge.curation.deduplication import DeduplicationResult
        from backend.knowledge.curation.graph_writer import WriteResult
        from backend.knowledge.curation.pipeline import CurationResult
        from backend.knowledge.curation.reconciliation import ReconcileTurnResult

        class FakeNoUserPipeline:
            async def extract_from_utterance(self, **kwargs):
                return CurationResult(
                    write_result=WriteResult(),
                    dedup_result=DeduplicationResult(
                        entities=[], merge_actions=[], entities_merged=0
                    ),
                    reconcile_result=ReconcileTurnResult(),
                    curation_time_ms=1.0,
                    validated_entities=[{"entity_id": "neo4j", "entity_type": "Technology"}],
                    validated_relationships=[
                        {"source": "mist-identity", "target": "neo4j", "type": "USES"}
                    ],
                )

        fake_vault = FakeVaultWriter()
        handler = make_handler(vault_writer=fake_vault, event_store_enabled=True)
        handler._extraction_pipeline = FakeNoUserPipeline()

        await handler.handle_message(
            user_message="MIST uses Neo4j for storage", session_id="no-user-sess"
        )
        await asyncio.sleep(0.1)

        assert (
            len(fake_vault.upsert_user_snapshot_calls) == 0
        ), "C-pattern: must NOT write the user snapshot when extraction has no user-scope"
        assert (
            len(fake_vault.upsert_user_calls) == 0
        ), "C-pattern must never call upsert_user (curated user.md)"


class TestVaultWriteFailureIsolation:
    """ADR-010 Invariant 6: vault write failures are swallowed, not raised.

    R1.3.1 moved the only vault write from the per-turn path to
    `end_session`; failure isolation is exercised there now.
    """

    @pytest.mark.asyncio
    async def test_end_session_write_failure_does_not_raise(self):
        fake_vault = FakeVaultWriter()
        fake_vault.fail_on_write = True
        handler = make_handler(vault_writer=fake_vault)
        await handler.handle_message(
            user_message="This will trigger vault failure", session_id="fail-session"
        )
        await handler._drain_extraction_tasks()

        # Act -- must not raise even though the note write fails
        await handler.end_session("fail-session")

    @pytest.mark.asyncio
    async def test_end_session_write_failure_logs_invariant_6_warning(self, caplog):
        import logging

        fake_vault = FakeVaultWriter()
        fake_vault.fail_on_write = True
        handler = make_handler(vault_writer=fake_vault)
        await handler.handle_message(
            user_message="Trigger vault failure for logging test", session_id="fail-session-log"
        )
        await handler._drain_extraction_tasks()

        with caplog.at_level(logging.WARNING, logger="backend.chat.conversation_handler"):
            await handler.end_session("fail-session-log")

        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            "Invariant 6" in msg for msg in warning_messages
        ), f"Expected 'Invariant 6' in warning messages. Got: {warning_messages}"


# ---------------------------------------------------------------------------
# TestVaultDisabled
# ---------------------------------------------------------------------------


class TestVaultDisabled:
    @pytest.mark.asyncio
    async def test_handle_message_succeeds_when_vault_writer_is_none(self):
        # Arrange
        handler = make_handler(vault_writer=None)

        # Act
        response = await handler.handle_message(
            user_message="Hello without vault",
            session_id="no-vault-session",
        )
        await asyncio.sleep(0.05)

        # Assert
        assert response is not None

    def test_handler_does_not_crash_on_vault_attribute_access_when_none(self):
        # Arrange
        handler = make_handler(vault_writer=None)

        # Assert -- attributes exist and are properly initialized
        assert handler._vault_writer is None
        assert isinstance(handler._vault_paths, dict)


# ---------------------------------------------------------------------------
# TestSlugDerivation
# ---------------------------------------------------------------------------


class TestSlugDerivation:
    def test_standard_session_id_is_preserved(self):
        handler = make_handler()
        result = handler._derive_session_slug("test-session-1")
        assert result == "test-session-1"

    def test_special_chars_are_replaced_with_hyphens(self):
        handler = make_handler()
        result = handler._derive_session_slug("Test_Session 42!")
        # Uppercase -> lower; underscore -> hyphen; space -> hyphen; ! -> hyphen.
        # Trailing hyphens are stripped. Result must contain only safe chars.
        assert "-" in result
        assert result == result.lower()
        assert all(c in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in result)
        # Core words must be present after normalization
        assert "test" in result
        assert "session" in result
        assert "42" in result

    def test_empty_slug_falls_back_to_session(self):
        handler = make_handler()
        # All special chars produces empty slug after strip
        result = handler._derive_session_slug("!!!")
        assert result == "session"

    def test_long_session_id_is_truncated_to_50_chars(self):
        handler = make_handler()
        long_id = "a" * 100
        result = handler._derive_session_slug(long_id)
        assert len(result) <= 50

    def test_slug_contains_only_lowercase_alnum_and_hyphens(self):
        handler = make_handler()
        inputs = [
            "MySession.With.Dots",
            "session@domain.com",
            "UPPER_CASE_ID",
            "mixed-123-ABC",
        ]
        for sid in inputs:
            slug = handler._derive_session_slug(sid)
            assert all(
                c in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in slug
            ), f"Slug '{slug}' from '{sid}' contains invalid characters"


# ---------------------------------------------------------------------------
# TestPhase6PathPreAllocation -- ADR-010 Step 0
# ---------------------------------------------------------------------------


class TestPhase6PathPreAllocation:
    """ADR-010 Cluster 8 Phase 6: vault_note_path is allocated synchronously
    at Step 0 of handle_message. R1.3 retired the extraction-pipeline
    forwarding this class used to also cover (curation no longer anchors
    facts to a VaultNote); the path allocation itself -- caching, idempotency
    within a session, and distinctness across sessions -- is still real
    production behavior and is what remains under test here.
    """

    def test_get_or_allocate_returns_none_when_vault_disabled(self) -> None:
        # Arrange
        handler = make_handler(vault_writer=None)

        # Act
        path = handler._get_or_allocate_vault_path("any-session")

        # Assert
        assert path is None
        # Path map remains untouched
        assert handler._vault_paths == {}

    def test_get_or_allocate_returns_path_when_vault_enabled(self) -> None:
        # Arrange
        fake_vault = FakeVaultWriter()
        handler = make_handler(vault_writer=fake_vault)

        # Act
        path = handler._get_or_allocate_vault_path("session-x")

        # Assert
        assert path is not None
        assert path.endswith(".md")
        assert "session-x" in path
        # State recorded for reuse
        assert handler._vault_paths["session-x"] == path

    def test_get_or_allocate_is_idempotent_within_session(self) -> None:
        # Arrange
        fake_vault = FakeVaultWriter()
        handler = make_handler(vault_writer=fake_vault)

        # Act -- multiple calls for same session must return identical path
        path_1 = handler._get_or_allocate_vault_path("stable-session")
        path_2 = handler._get_or_allocate_vault_path("stable-session")
        path_3 = handler._get_or_allocate_vault_path("stable-session")

        # Assert
        assert path_1 == path_2 == path_3
        assert path_1 is not None

    def test_get_or_allocate_distinct_sessions_get_distinct_paths(self) -> None:
        # Arrange
        fake_vault = FakeVaultWriter()
        handler = make_handler(vault_writer=fake_vault)

        # Act
        path_a = handler._get_or_allocate_vault_path("session-a")
        path_b = handler._get_or_allocate_vault_path("session-b")

        # Assert
        assert path_a != path_b
        assert path_a is not None
        assert path_b is not None

    @pytest.mark.asyncio
    async def test_handle_message_does_not_pass_vault_note_path_to_extraction(self) -> None:
        # Arrange -- a handler with a real fake vault writer + a fake extraction
        # pipeline that records every kwargs dict. R1.3: handle_message still
        # primes the vault path cache at Step 0, but no longer dispatches the
        # allocated path to the extraction pipeline.
        fake_vault = FakeVaultWriter()
        handler = make_handler(
            vault_writer=fake_vault,
            event_store_enabled=True,  # event_id required to dispatch extraction
        )
        # Replace pipeline with a recorder so we can inspect kwargs
        recorder = FakeExtractionPipeline()
        handler._extraction_pipeline = recorder

        # Act
        await handler.handle_message(
            user_message="Talk about Python and Neo4j today.",
            session_id="phase6-session",
        )
        await handler._drain_extraction_tasks()

        # Assert -- the extraction pipeline never receives vault_note_path.
        assert len(recorder.calls) == 1
        assert "vault_note_path" not in recorder.calls[0]

    @pytest.mark.asyncio
    async def test_step_0_runs_even_when_extraction_skipped_for_short_message(self) -> None:
        # Arrange -- short messages skip extraction dispatch entirely. Path
        # pre-allocation still runs in handle_message (Phase 6 invariant --
        # R1.3.1: the path only feeds the session-end note, so priming it
        # here just keeps the slug stable for whenever end_session runs).
        fake_vault = FakeVaultWriter()
        handler = make_handler(vault_writer=fake_vault, event_store_enabled=True)
        recorder = FakeExtractionPipeline()
        handler._extraction_pipeline = recorder

        # Act -- a message under 3 words skips extraction dispatch.
        await handler.handle_message(user_message="Hi", session_id="short-phase6")
        await asyncio.sleep(0.05)

        # Assert -- path pre-allocated even though extraction was skipped.
        assert recorder.calls == []
        assert "short-phase6" in handler._vault_paths

    @pytest.mark.asyncio
    async def test_path_allocated_before_event_store_write(self) -> None:
        # Arrange + Assert -- structural check: _get_or_allocate_vault_path
        # must be reachable before _record_turn_event so the path is available
        # even when the event store is the source of the event_id. We verify
        # by toggling the vault writer and confirming the path lookup never
        # depends on event_id.
        fake_vault = FakeVaultWriter()
        handler = make_handler(vault_writer=fake_vault, event_store_enabled=False)

        # Act -- pre-allocate without any event_store interaction
        path = handler._get_or_allocate_vault_path("pre-allocation")

        # Assert
        assert path is not None
        assert "pre-allocation" in path


# ---------------------------------------------------------------------------
# TestDeriveSessionNotePath (R1.3.1 Task 6) -- catch-up's path derivation
# ---------------------------------------------------------------------------


class TestDeriveSessionNotePath:
    """`derive_session_note_path` gives startup catch-up the same path a
    live session would have allocated, without touching `_vault_paths`.
    """

    def test_returns_none_when_vault_disabled(self) -> None:
        # Arrange
        handler = make_handler(vault_writer=None)

        # Act
        path = handler.derive_session_note_path("any-session", "hello there", "2026-07-29")

        # Assert
        assert path is None

    def test_returns_a_path_derived_from_utterance_and_date(self) -> None:
        # Arrange
        fake_vault = FakeVaultWriter()
        handler = make_handler(vault_writer=fake_vault)

        # Act
        path = handler.derive_session_note_path(
            "s-crashed", "tell me about the vault architecture", "2026-07-29"
        )

        # Assert
        assert path is not None
        assert path.startswith("/tmp/vault/sessions/2026-07-29-")
        assert path.endswith(".md")
        assert "vault-architecture" in path

    def test_does_not_prime_the_live_vault_paths_cache(self) -> None:
        # Arrange -- catch-up runs for a session that is already over; it
        # must not make a later live lookup for the same session_id reuse a
        # catch-up-derived path.
        fake_vault = FakeVaultWriter()
        handler = make_handler(vault_writer=fake_vault)

        # Act
        handler.derive_session_note_path("s-crashed", "hello there", "2026-07-29")

        # Assert
        assert "s-crashed" not in handler._vault_paths

    def test_matches_the_live_allocation_path_for_the_same_inputs(self) -> None:
        # Arrange -- the load-bearing property: a catch-up note and a live
        # note for the same session_id + first utterance must derive the
        # same SLUG (dates naturally differ -- catch-up runs later than the
        # session did -- so the slug, not the full path, is what must
        # match, or the two note-writing paths silently diverge).
        fake_vault = FakeVaultWriter()
        handler = make_handler(vault_writer=fake_vault)
        utterance = "walk me through the extraction pipeline"

        # Act
        live_path = handler._get_or_allocate_vault_path("s-live", first_utterance=utterance)
        # A second handler models catch-up running in a fresh process, with
        # no `_vault_paths` cache primed for "s-live".
        catchup_handler = make_handler(vault_writer=FakeVaultWriter())
        catchup_path = catchup_handler.derive_session_note_path("s-live", utterance, "2026-07-29")
        expected_slug = handler._derive_session_slug_from_utterance(utterance, "s-live")

        # Assert
        assert live_path.endswith(f"-{expected_slug}.md")
        assert catchup_path.endswith(f"-{expected_slug}.md")


# ---------------------------------------------------------------------------
# TestPhase9SlugDerivation -- session slug from first utterance content
# ---------------------------------------------------------------------------


class TestPhase9SlugDerivation:
    """ADR-010 Phase 9: session-note slug derived from significant words in
    the first user utterance, with a 4-char session-id hash for uniqueness.
    """

    def test_significant_words_become_slug(self) -> None:
        handler = make_handler()
        slug = handler._derive_session_slug_from_utterance(
            "Tell me about the vault architecture for MIST",
            session_id="sess-001",
        )
        # Stopwords removed (tell/me/about/the/for), short tokens removed (none here)
        assert slug.startswith("vault-architecture-mist-")
        # Hash suffix appended for uniqueness (4 hex chars)
        assert len(slug.split("-")[-1]) == 4

    def test_falls_back_to_hash_when_no_significant_words(self) -> None:
        handler = make_handler()
        # Single short token + stopword
        slug = handler._derive_session_slug_from_utterance("Hi", session_id="abc-123")
        assert slug.startswith("session-")
        # 8-char hex digest
        assert len(slug.split("-", 1)[1]) == 8

    def test_short_tokens_filtered(self) -> None:
        handler = make_handler()
        slug = handler._derive_session_slug_from_utterance(
            "I am the so",
            session_id="x",
        )
        # All tokens are stopwords or <3 chars; falls back to hash.
        assert slug.startswith("session-")

    def test_punctuation_only_falls_back_to_hash(self) -> None:
        handler = make_handler()
        slug = handler._derive_session_slug_from_utterance("!!!???", session_id="x")
        # Empty token list -> hash fallback.
        assert slug.startswith("session-")

    def test_two_similar_utterances_distinct_session_ids_get_distinct_slugs(self) -> None:
        # Two sessions with the same opening content must NOT share a slug.
        # The 4-char session-id hash provides uniqueness.
        handler = make_handler()
        slug_a = handler._derive_session_slug_from_utterance(
            "Hello from session A",
            session_id="session-a",
        )
        slug_b = handler._derive_session_slug_from_utterance(
            "Hello from session B",
            session_id="session-b",
        )
        assert slug_a != slug_b

    def test_same_session_id_same_utterance_produces_same_slug(self) -> None:
        # Determinism: same inputs -> same slug across calls.
        handler = make_handler()
        slug_1 = handler._derive_session_slug_from_utterance(
            "Tell me about Python and async",
            session_id="sess-stable",
        )
        slug_2 = handler._derive_session_slug_from_utterance(
            "Tell me about Python and async",
            session_id="sess-stable",
        )
        assert slug_1 == slug_2

    def test_slug_caps_at_50_chars(self) -> None:
        handler = make_handler()
        long_utterance = (
            "discussing extensive comprehensive complete thorough exhaustive "
            "memory architecture rebuild determinism specification"
        )
        slug = handler._derive_session_slug_from_utterance(long_utterance, session_id="sess-long")
        assert len(slug) <= 50

    def test_slug_only_contains_lowercase_alnum_and_hyphens(self) -> None:
        handler = make_handler()
        slug = handler._derive_session_slug_from_utterance(
            "What's the BEST! way to handle Foo & Bar?",
            session_id="x",
        )
        assert all(c in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in slug)

    @pytest.mark.asyncio
    async def test_handle_message_uses_first_utterance_for_slug(self) -> None:
        # Arrange
        fake_vault = FakeVaultWriter()
        handler = make_handler(vault_writer=fake_vault)

        # Act
        await handler.handle_message(
            user_message="Discussing the vault architecture today",
            session_id="my-session-1",
        )

        # Assert -- the cached vault path's slug derives from utterance
        # content, not from "my-session-1" the session id.
        path = handler._vault_paths["my-session-1"]
        assert "discussing" in path or "vault" in path or "architecture" in path

    @pytest.mark.asyncio
    async def test_first_utterance_locks_slug_for_subsequent_turns(self) -> None:
        # Arrange
        fake_vault = FakeVaultWriter()
        handler = make_handler(vault_writer=fake_vault)
        session_id = "lock-test"

        # Act -- first utterance defines the slug; second turn must reuse it
        # rather than re-deriving from the new utterance.
        await handler.handle_message(
            user_message="The vault architecture for MIST", session_id=session_id
        )
        path_after_first = handler._vault_paths[session_id]

        await handler.handle_message(
            user_message="Now lets talk about something completely different",
            session_id=session_id,
        )
        path_after_second = handler._vault_paths[session_id]

        # Assert
        assert path_after_first == path_after_second
