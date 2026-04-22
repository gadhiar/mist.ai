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
