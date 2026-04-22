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

# ---------------------------------------------------------------------------
# FakeVaultWriter test double
# ---------------------------------------------------------------------------


class FakeVaultWriter:
    """In-memory vault writer test double. Records all append calls."""

    def __init__(self):
        self.start_calls: int = 0
        self.stop_calls: int = 0
        self.append_calls: list[dict] = []
        self.fail_on_append: bool = False  # Toggle per test to simulate failure

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

    async def upsert_user(self, *args, **kwargs) -> str:
        return ""


# ---------------------------------------------------------------------------
# Test doubles for ExtractionPipeline
# ---------------------------------------------------------------------------


class FakeExtractionPipeline:
    """Minimal extraction pipeline that never extracts."""

    def __init__(self):
        self.calls: list[dict] = []

    async def extract_from_utterance(self, **kwargs):
        self.calls.append(kwargs)
        from backend.knowledge.extraction.validator import ValidationResult

        return ValidationResult(valid=True, entities=[], relationships=[])


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _make_retriever(config, gs):
    return KnowledgeRetriever(config=config, graph_store=gs)


def make_handler(vault_writer=None, event_store_enabled: bool = False):
    """Construct a ConversationHandler suitable for vault integration tests."""
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

        # Act -- first turn fails, second turn succeeds
        fake_vault.fail_on_append = True
        await handler.handle_message(user_message="Turn one fails in vault", session_id=session_id)
        fake_vault.fail_on_append = False
        await handler.handle_message(
            user_message="Turn two succeeds in vault", session_id=session_id
        )
        await asyncio.sleep(0.05)

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
        # Arrange -- short messages skip extraction dispatch but still produce
        # a vault session note. The path must be allocated for both vault write
        # and (deferred) extraction even though no extraction fires for a short turn.
        fake_vault = FakeVaultWriter()
        handler = make_handler(vault_writer=fake_vault, event_store_enabled=True)
        recorder = FakeExtractionPipeline()
        handler._extraction_pipeline = recorder

        # Act -- a message under 3 words skips extraction dispatch.
        await handler.handle_message(user_message="Hi", session_id="short-phase6")
        await asyncio.sleep(0.05)

        # Assert -- vault write happened (path allocated), extraction skipped.
        assert len(fake_vault.append_calls) == 1
        assert recorder.calls == []
        assert "short-phase6" in handler._vault_paths

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
