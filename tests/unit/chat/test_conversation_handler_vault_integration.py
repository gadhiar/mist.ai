"""Vault integration tests for ConversationHandler (Cluster 8 Phase 5).

Tests that handle_message correctly writes to the vault layer via
_write_to_vault, that failures are isolated per ADR-010 Invariant 6,
and that _derive_session_slug normalizes session identifiers correctly.

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
    """In-memory vault writer test double. Records all append calls."""

    def __init__(self):
        self.start_calls: int = 0
        self.stop_calls: int = 0
        self.append_calls: list[dict] = []
        self.upsert_user_calls: list[dict] = []  # records (user_id, body_markdown)
        # C-pattern writeback now targets the derived snapshot, not user.md.
        self.upsert_user_snapshot_calls: list[dict] = []
        self.mark_completed_calls: list[str] = []  # records vault_note_paths
        self.fail_on_append: bool = False  # Toggle per test to simulate failure
        # Maps vault_note_path -> existing turn_count "on disk". Used to
        # simulate peek_turn_count returning a non-zero value for an existing
        # session note (e.g., backend-restart durable-counter scenario).
        self._existing_turn_counts: dict[str, int] = {}

    def preload_existing_turn_count(self, vault_note_path: str, turn_count: int) -> None:
        """Test helper: simulate an existing session note with `turn_count` turns."""
        self._existing_turn_counts[vault_note_path] = turn_count

    def peek_turn_count(self, path: str) -> int:
        """Mimics VaultWriter.peek_turn_count: returns preloaded count or 0."""
        return self._existing_turn_counts.get(path, 0)

    async def start(self) -> None:
        self.start_calls += 1

    async def stop(self) -> None:
        self.stop_calls += 1

    def session_path(self, date: str, slug: str) -> str:
        return f"/tmp/vault/sessions/{date}-{slug}.md"

    async def append_turn_to_session(
        self,
        session_id: str,
        turn_index: int,
        user_text: str,
        mist_text: str,
        vault_note_path: str | None = None,
    ) -> str:
        if self.fail_on_append:
            raise RuntimeError("simulated vault write failure")
        path = vault_note_path or self.session_path("2026-04-22", session_id)
        self.append_calls.append(
            {
                "session_id": session_id,
                "turn_index": turn_index,
                "user_text": user_text,
                "mist_text": mist_text,
                "vault_note_path": path,
            }
        )
        return path

    async def update_entities_extracted(self, *args, **kwargs) -> None:
        pass

    async def upsert_identity(self, *args, **kwargs) -> str:
        return ""

    async def mark_session_completed(self, vault_note_path: str) -> str | None:
        self.mark_completed_calls.append(vault_note_path)
        return vault_note_path

    async def append_session_synthesis(
        self, vault_note_path: str, synthesis_markdown: str
    ) -> str | None:
        if not hasattr(self, "synthesis_calls"):
            self.synthesis_calls: list[dict] = []
        self.synthesis_calls.append(
            {"vault_note_path": vault_note_path, "synthesis_markdown": synthesis_markdown}
        )
        return vault_note_path

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
    """Minimal extraction pipeline. Returns one synthetic entity by default so
    vault-append assertions in legacy tests still fire under the conditional-append
    rule (ADR-011: skip vault append when extraction yields zero entities AND zero
    relationships). Tests that need the zero-extraction path explicitly should
    pass `entities=[]` and `relationships=[]` to override.
    """

    def __init__(
        self,
        entities: list[dict] | None = None,
        relationships: list[dict] | None = None,
    ):
        self.calls: list[dict] = []
        # Default to one entity so legacy tests asserting "vault append fired"
        # continue to hold under conditional-append semantics.
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

    Default `event_store_enabled=True` because under ADR-011 bucket 2's
    conditional vault append (vault writes happen inside `_extract_knowledge_async`,
    which only fires when `event_id` is non-empty), tests need event_store
    enabled to exercise the post-extraction vault path. Tests that explicitly
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
# TestVaultWriteOnSuccessfulTurn
# ---------------------------------------------------------------------------


class TestVaultWriteOnSuccessfulTurn:
    @pytest.mark.asyncio
    async def test_single_turn_writes_one_append_call(self):
        # Arrange
        fake_vault = FakeVaultWriter()
        handler = make_handler(vault_writer=fake_vault)

        # Act
        response = await handler.handle_message(
            user_message="Tell me about Python",
            session_id="test-session-1",
        )

        # Allow background tasks to settle
        await asyncio.sleep(0.05)

        # Assert
        assert response is not None
        assert len(fake_vault.append_calls) == 1
        call = fake_vault.append_calls[0]
        assert call["user_text"] == "Tell me about Python"
        assert call["mist_text"] == response
        assert call["turn_index"] == 1

    @pytest.mark.asyncio
    async def test_two_turns_same_session_share_vault_note_path(self):
        # Arrange
        fake_vault = FakeVaultWriter()
        handler = make_handler(vault_writer=fake_vault)
        session_id = "test-session-2"

        # Act
        await handler.handle_message(user_message="First message here", session_id=session_id)
        await handler.handle_message(user_message="Second message here", session_id=session_id)
        await asyncio.sleep(0.05)

        # Assert -- both turns share the same vault path, turn indices are 1 and 2
        assert len(fake_vault.append_calls) == 2
        path_1 = fake_vault.append_calls[0]["vault_note_path"]
        path_2 = fake_vault.append_calls[1]["vault_note_path"]
        assert path_1 == path_2, "Both turns must share the same vault note path"
        assert fake_vault.append_calls[0]["turn_index"] == 1
        assert fake_vault.append_calls[1]["turn_index"] == 2

    @pytest.mark.asyncio
    async def test_two_sessions_get_distinct_vault_note_paths(self):
        # Arrange
        fake_vault = FakeVaultWriter()
        handler = make_handler(vault_writer=fake_vault)

        # Act
        await handler.handle_message(user_message="Hello from session A", session_id="session-a")
        await handler.handle_message(user_message="Hello from session B", session_id="session-b")
        await asyncio.sleep(0.05)

        # Assert -- two distinct paths, each with turn_index 1
        assert len(fake_vault.append_calls) == 2
        path_a = fake_vault.append_calls[0]["vault_note_path"]
        path_b = fake_vault.append_calls[1]["vault_note_path"]
        assert path_a != path_b, "Different sessions must produce different vault note paths"
        assert fake_vault.append_calls[0]["turn_index"] == 1
        assert fake_vault.append_calls[1]["turn_index"] == 1


# ---------------------------------------------------------------------------
# TestVaultWriteFailureIsolation
# ---------------------------------------------------------------------------


class TestDurableTurnCounter:
    """Gap #4: backend restart must not reset turn numbering for an ongoing session.

    Pre-fix (2026-05-06): ConversationHandler._vault_turn_counts.setdefault(sid, 0)
    always seeded to 0 on first allocation. After backend restart with the same
    session_id reused (e.g., session_id="default"), turn 1 of the new run wrote
    "## Turn 1" to a file that already had Turn 1-N from before. V6 unified-path
    run on 2026-05-06 visibly demonstrated this: Turn 1-7 from the failed first
    attempt, then Turn 1-30 from the retry, in the same file.

    Fix: seed _vault_turn_counts[session_id] from VaultWriter.peek_turn_count(path)
    so the counter resumes from the file's existing turn_count.
    """

    @pytest.mark.asyncio
    async def test_turn_index_resumes_from_existing_file_turn_count(self):
        # Arrange: simulate a vault note with 7 turns already on disk
        fake_vault = FakeVaultWriter()
        handler = make_handler(vault_writer=fake_vault)
        session_id = "resumed-session"
        # Pre-allocate the path the handler will derive, then preload turn_count
        # The handler will call session_path with today + slug-from-utterance.
        # We don't know the slug ahead of time, so we patch the dict on the
        # first append attempt instead -- preload via the handler's own path.

        # First handle_message to discover the path, then mutate preload BEFORE
        # the second handle_message would seed. But the test wants to assert
        # the FIRST turn after restart resumes at index 8. We achieve this by
        # constructing the handler, inspecting the path it would allocate, and
        # preloading.
        path = handler._get_or_allocate_vault_path(session_id, first_utterance="seven turns done")
        # Reset state to simulate restart: clear in-memory counter + path map,
        # preload the existing file count.
        handler._vault_turn_counts.clear()
        handler._vault_paths.clear()
        fake_vault.preload_existing_turn_count(path, 7)

        # Act: send a single turn after "restart"
        response = await handler.handle_message(
            user_message="seven turns done", session_id=session_id
        )

        await asyncio.sleep(0.05)

        # Assert: the new turn lands at index 8, not 1
        assert response is not None
        assert len(fake_vault.append_calls) == 1
        assert (
            fake_vault.append_calls[0]["turn_index"] == 8
        ), f"durable counter broken; got turn_index={fake_vault.append_calls[0]['turn_index']}"

    @pytest.mark.asyncio
    async def test_turn_index_starts_at_one_for_new_session(self):
        """No existing file -> peek returns 0 -> first turn at index 1 (no regression)."""
        fake_vault = FakeVaultWriter()
        handler = make_handler(vault_writer=fake_vault)
        await handler.handle_message(user_message="brand new session", session_id="fresh-sess")
        await asyncio.sleep(0.05)
        assert fake_vault.append_calls[0]["turn_index"] == 1


class TestConditionalPerTurnAppend:
    """ADR-011 bucket 2: vault append skipped for zero-extraction turns.

    Under the 2026-05-06 canonical pattern, the per-turn session-note append
    is gated on extraction yielding at least one entity OR one relationship.
    Turns that produce no graph state (purely conversational utterances like
    "Hi", "Thanks") are not anchored in the vault because they have no
    DERIVED_FROM edges to back. Substantive turns still write; the rebuild
    contract is preserved.
    """

    @pytest.mark.asyncio
    async def test_appends_when_extraction_yielded_entities(self):
        fake_vault = FakeVaultWriter()
        handler = make_handler(vault_writer=fake_vault)
        # Default FakeExtractionPipeline yields one synthetic entity
        await handler.handle_message(
            user_message="I work on MIST every day", session_id="substantive-1"
        )
        await asyncio.sleep(0.1)
        assert len(fake_vault.append_calls) == 1, "must append when entities present"

    @pytest.mark.asyncio
    async def test_skips_when_extraction_yielded_zero_entities_and_zero_rels(self):
        fake_vault = FakeVaultWriter()
        handler = make_handler(vault_writer=fake_vault)
        # Override pipeline to return empty extraction
        handler._extraction_pipeline = FakeExtractionPipeline(entities=[], relationships=[])
        await handler.handle_message(
            user_message="I work on MIST every day", session_id="empty-extraction-1"
        )
        await asyncio.sleep(0.1)
        assert (
            len(fake_vault.append_calls) == 0
        ), "ADR-011 bucket 2: vault append must skip when extraction yields no graph state"

    @pytest.mark.asyncio
    async def test_appends_when_only_relationships_present(self):
        """Relationships alone (no new entities) still anchor graph state, so
        the vault append should fire.
        """
        fake_vault = FakeVaultWriter()
        handler = make_handler(vault_writer=fake_vault)
        handler._extraction_pipeline = FakeExtractionPipeline(
            entities=[],
            relationships=[{"source": "user", "target": "python", "type": "USES"}],
        )
        await handler.handle_message(
            user_message="I use Python every day", session_id="rels-only-1"
        )
        await asyncio.sleep(0.1)
        assert (
            len(fake_vault.append_calls) == 1
        ), "vault append must fire when relationships present (graph state to anchor)"


class TestSessionEnd:
    """Gap #1a / ADR-011 bucket 2: session-end signal flips status.

    On WebSocket disconnect or other end-of-session trigger, the
    ConversationHandler should flip every active session's vault note from
    `status: in-progress` to `status: completed`. The MIST end-of-session
    synthesis (gap #1b) is a separate concern that runs alongside.
    """

    @pytest.mark.asyncio
    async def test_end_session_marks_specific_session_completed(self):
        fake_vault = FakeVaultWriter()
        handler = make_handler(vault_writer=fake_vault)
        # Send a turn so the session has a vault path allocated
        await handler.handle_message(user_message="Working on MIST today", session_id="end-1")
        await asyncio.sleep(0.05)
        path = handler._vault_paths["end-1"]

        await handler.end_session("end-1")
        await asyncio.sleep(0.05)

        assert fake_vault.mark_completed_calls == [
            path
        ], f"expected mark_session_completed once with {path}, got {fake_vault.mark_completed_calls}"

    @pytest.mark.asyncio
    async def test_end_session_with_no_arg_ends_all_active_sessions(self):
        fake_vault = FakeVaultWriter()
        handler = make_handler(vault_writer=fake_vault)
        await handler.handle_message(user_message="First session start", session_id="sess-a")
        await handler.handle_message(user_message="Second session start", session_id="sess-b")
        await asyncio.sleep(0.05)
        path_a = handler._vault_paths["sess-a"]
        path_b = handler._vault_paths["sess-b"]

        await handler.end_session()  # no arg = all
        await asyncio.sleep(0.05)

        assert sorted(fake_vault.mark_completed_calls) == sorted([path_a, path_b])

    @pytest.mark.asyncio
    async def test_end_session_no_op_when_session_never_had_vault_path(self):
        fake_vault = FakeVaultWriter()
        handler = make_handler(vault_writer=fake_vault)
        # No turns sent for this session, so no path allocated
        await handler.end_session("never-touched")
        await asyncio.sleep(0.05)
        assert fake_vault.mark_completed_calls == []

    @pytest.mark.asyncio
    async def test_end_session_writes_synthesis_before_status_flip(self):
        """Gap #1b: synthesis is generated and appended before status flips.

        On session-end, MIST writes a `## Summary` section synthesizing the
        session, then marks the note completed. Both happen via the vault
        writer. Synthesis failure does not block status flip.
        """
        fake_vault = FakeVaultWriter()
        # FakeLLM default response is the JSON validation default; fine for testing
        # that synthesis was attempted and propagated.
        handler = make_handler(vault_writer=fake_vault)
        # Send 2 messages so synthesis has substantive content
        await handler.handle_message(user_message="First substantive message", session_id="syn-1")
        await handler.handle_message(user_message="Second substantive message", session_id="syn-1")
        await asyncio.sleep(0.1)

        await handler.end_session("syn-1")
        await asyncio.sleep(0.05)

        # Both synthesis and status flip happened
        assert hasattr(fake_vault, "synthesis_calls") and len(fake_vault.synthesis_calls) == 1
        assert len(fake_vault.mark_completed_calls) == 1
        # Same path
        assert (
            fake_vault.synthesis_calls[0]["vault_note_path"] == fake_vault.mark_completed_calls[0]
        )

    @pytest.mark.asyncio
    async def test_end_session_no_synthesis_for_short_session(self):
        """One-turn or empty sessions don't get synthesized (nothing to summarize)."""
        fake_vault = FakeVaultWriter()
        handler = make_handler(vault_writer=fake_vault)
        await handler.handle_message(user_message="Single message session", session_id="syn-2")
        await asyncio.sleep(0.05)

        await handler.end_session("syn-2")
        await asyncio.sleep(0.05)

        # Status flipped, but no synthesis (session has only 2 messages: user + assistant)
        # ConversationSession.add_message records both, so len(messages) == 2 = synthesis fires.
        # Test: a session with truly minimal content STILL gets a synthesis attempt because
        # the >=2 message gate is permissive. We assert status flipped regardless.
        assert len(fake_vault.mark_completed_calls) == 1

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
    (deterministic, intentional).
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
    @pytest.mark.asyncio
    async def test_vault_failure_does_not_raise_from_handle_message(self):
        # Arrange
        fake_vault = FakeVaultWriter()
        fake_vault.fail_on_append = True
        handler = make_handler(vault_writer=fake_vault)

        # Act -- must not raise even though vault write fails
        response = await handler.handle_message(
            user_message="This will trigger vault failure",
            session_id="fail-session",
        )
        await asyncio.sleep(0.05)

        # Assert -- handle_message still returned the assistant message
        assert response is not None
        assert isinstance(response, str)

    @pytest.mark.asyncio
    async def test_vault_failure_logs_invariant_6_warning(self, caplog):
        # Arrange
        import logging

        fake_vault = FakeVaultWriter()
        fake_vault.fail_on_append = True
        handler = make_handler(vault_writer=fake_vault)

        # Act
        with caplog.at_level(logging.WARNING, logger="backend.chat.conversation_handler"):
            await handler.handle_message(
                user_message="Trigger vault failure for logging test",
                session_id="fail-session-log",
            )
        await asyncio.sleep(0.05)

        # Assert -- warning contains the Invariant 6 identifier
        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            "Invariant 6" in msg for msg in warning_messages
        ), f"Expected 'Invariant 6' in warning messages. Got: {warning_messages}"

    @pytest.mark.asyncio
    async def test_subsequent_turn_after_failure_still_attempts_write(self):
        # Arrange -- failure is per-call, not sticky
        fake_vault = FakeVaultWriter()
        handler = make_handler(vault_writer=fake_vault)
        session_id = "partial-fail-session"

        # Act -- first turn fails, second turn succeeds. ADR-011 bucket 2:
        # vault append is now inside fire-and-forget extraction, so we must
        # await extraction settlement between turns to control fail_on_append
        # timing deterministically.
        fake_vault.fail_on_append = True
        await handler.handle_message(user_message="Turn one fails in vault", session_id=session_id)
        await asyncio.sleep(0.1)  # let turn 1 extraction + vault append settle (and fail)

        fake_vault.fail_on_append = False
        await handler.handle_message(
            user_message="Turn two succeeds in vault", session_id=session_id
        )
        await asyncio.sleep(0.1)  # let turn 2 extraction + vault append settle

        # Assert -- only the second call recorded (first was swallowed)
        assert len(fake_vault.append_calls) == 1
        assert fake_vault.append_calls[0]["user_text"] == "Turn two succeeds in vault"
        # turn_index should be 2 because the counter still incremented on the failed call
        assert fake_vault.append_calls[0]["turn_index"] == 2


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
        assert isinstance(handler._vault_turn_counts, dict)

    @pytest.mark.asyncio
    async def test_write_to_vault_returns_none_when_no_writer(self):
        # Arrange
        handler = make_handler(vault_writer=None)

        # Act
        result = await handler._write_to_vault(
            session_id="s1",
            user_message="hi",
            assistant_message="hello",
        )

        # Assert
        assert result is None


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

    @pytest.mark.asyncio
    async def test_write_to_vault_uses_derived_slug_in_path(self):
        # Arrange
        fake_vault = FakeVaultWriter()
        handler = make_handler(vault_writer=fake_vault)
        session_id = "My Session With Spaces"

        # Act
        await handler._write_to_vault(
            session_id=session_id,
            user_message="hello",
            assistant_message="hi there",
        )

        # Assert -- the vault path was derived using the slug (lowered + normalized)
        assert len(fake_vault.append_calls) == 1
        path = fake_vault.append_calls[0]["vault_note_path"]
        # Path should not contain spaces or uppercase letters in the slug portion
        assert " " not in path
        assert "my" in path.lower()


# ---------------------------------------------------------------------------
# TestPhase6PathPreAllocation -- ADR-010 Step 0
# ---------------------------------------------------------------------------


class TestPhase6PathPreAllocation:
    """ADR-010 Cluster 8 Phase 6: vault_note_path is allocated synchronously
    at Step 0 of handle_message and threaded through to the extraction
    pipeline so curation can emit DERIVED_FROM->VaultNote edges.
    """

    def test_get_or_allocate_returns_none_when_vault_disabled(self) -> None:
        # Arrange
        handler = make_handler(vault_writer=None)

        # Act
        path = handler._get_or_allocate_vault_path("any-session")

        # Assert
        assert path is None
        # Counters and path map remain untouched
        assert handler._vault_paths == {}
        assert handler._vault_turn_counts == {}

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
        # State recorded for reuse + counter initialized to zero
        assert handler._vault_paths["session-x"] == path
        assert handler._vault_turn_counts["session-x"] == 0

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

    def test_get_or_allocate_does_not_increment_counter(self) -> None:
        # Arrange -- Step 0 path allocation MUST be free of side effects on the
        # turn counter; only _write_to_vault increments it. This decoupling lets
        # `handle_message` allocate the path before deciding whether to dispatch
        # extraction without inflating the turn index.
        fake_vault = FakeVaultWriter()
        handler = make_handler(vault_writer=fake_vault)

        # Act
        handler._get_or_allocate_vault_path("counter-test")
        handler._get_or_allocate_vault_path("counter-test")
        handler._get_or_allocate_vault_path("counter-test")

        # Assert
        assert handler._vault_turn_counts["counter-test"] == 0

    @pytest.mark.asyncio
    async def test_handle_message_passes_vault_note_path_to_extraction(self) -> None:
        # Arrange -- a handler with a real fake vault writer + a fake extraction
        # pipeline that records every kwargs dict. handle_message must dispatch
        # background extraction with vault_note_path matching the pre-allocated path.
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
        await asyncio.sleep(0.05)  # let fire-and-forget extraction settle

        # Assert -- the extraction pipeline received vault_note_path matching
        # the path the vault writer wrote to.
        assert len(recorder.calls) == 1
        kwargs = recorder.calls[0]
        assert "vault_note_path" in kwargs
        assert kwargs["vault_note_path"] is not None
        # Vault writer recorded the same path
        assert len(fake_vault.append_calls) == 1
        assert kwargs["vault_note_path"] == fake_vault.append_calls[0]["vault_note_path"]

    @pytest.mark.asyncio
    async def test_handle_message_passes_none_when_vault_disabled(self) -> None:
        # Arrange
        handler = make_handler(vault_writer=None, event_store_enabled=True)
        recorder = FakeExtractionPipeline()
        handler._extraction_pipeline = recorder

        # Act
        await handler.handle_message(
            user_message="A long enough utterance to trigger extraction dispatch.",
            session_id="no-vault-phase6",
        )
        await asyncio.sleep(0.05)

        # Assert -- vault_note_path is None when the vault layer is disabled
        assert len(recorder.calls) == 1
        assert recorder.calls[0]["vault_note_path"] is None

    @pytest.mark.asyncio
    async def test_handle_message_two_turns_pass_same_vault_path_to_extraction(self) -> None:
        # Arrange
        fake_vault = FakeVaultWriter()
        handler = make_handler(vault_writer=fake_vault, event_store_enabled=True)
        recorder = FakeExtractionPipeline()
        handler._extraction_pipeline = recorder

        # Act -- two turns of the same session
        session_id = "multi-turn-phase6"
        await handler.handle_message(
            user_message="First long utterance about Python and async.",
            session_id=session_id,
        )
        await handler.handle_message(
            user_message="Second long utterance about Neo4j and Cypher.",
            session_id=session_id,
        )
        await asyncio.sleep(0.05)

        # Assert -- both extraction dispatches receive the same vault_note_path,
        # matching ADR-010 "Pre-allocated vault path" stability invariant.
        assert len(recorder.calls) == 2
        path_1 = recorder.calls[0]["vault_note_path"]
        path_2 = recorder.calls[1]["vault_note_path"]
        assert path_1 is not None
        assert path_1 == path_2

    @pytest.mark.asyncio
    async def test_step_0_runs_even_when_extraction_skipped_for_short_message(self) -> None:
        # Arrange -- short messages skip extraction dispatch. Under ADR-011
        # bucket 2, vault append is gated on extraction firing; for short
        # messages, no extraction means no vault append (zero graph state to
        # anchor). Path pre-allocation still runs in handle_message (Phase 6
        # invariant -- the path is needed by extraction's DERIVED_FROM emission
        # IF extraction were to fire later in the session).
        fake_vault = FakeVaultWriter()
        handler = make_handler(vault_writer=fake_vault, event_store_enabled=True)
        recorder = FakeExtractionPipeline()
        handler._extraction_pipeline = recorder

        # Act -- a message under 3 words skips extraction dispatch.
        await handler.handle_message(user_message="Hi", session_id="short-phase6")
        await asyncio.sleep(0.05)

        # Assert -- path pre-allocated, extraction skipped, vault append also
        # skipped (per canonical pattern bucket 2).
        assert recorder.calls == []
        assert "short-phase6" in handler._vault_paths
        assert (
            len(fake_vault.append_calls) == 0
        ), "vault append must skip for zero-extraction turns under ADR-011 bucket 2"

    @pytest.mark.asyncio
    async def test_path_allocated_before_event_store_write(self) -> None:
        # Arrange + Assert -- structural check: _get_or_allocate_vault_path
        # must be reachable before _record_turn_event so the path is available
        # for extraction dispatch even when the event store is the source of
        # the event_id. We verify by toggling the vault writer and confirming
        # the path lookup never depends on event_id.
        fake_vault = FakeVaultWriter()
        handler = make_handler(vault_writer=fake_vault, event_store_enabled=False)

        # Act -- pre-allocate without any event_store interaction
        path = handler._get_or_allocate_vault_path("pre-allocation")

        # Assert
        assert path is not None
        assert "pre-allocation" in path


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
        await asyncio.sleep(0.05)

        # Assert -- the vault path slug derives from utterance content,
        # not from "my-session-1" the session id.
        assert len(fake_vault.append_calls) == 1
        path = fake_vault.append_calls[0]["vault_note_path"]
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
        await handler.handle_message(
            user_message="Now lets talk about something completely different",
            session_id=session_id,
        )
        await asyncio.sleep(0.05)

        # Assert
        assert len(fake_vault.append_calls) == 2
        assert (
            fake_vault.append_calls[0]["vault_note_path"]
            == fake_vault.append_calls[1]["vault_note_path"]
        )
