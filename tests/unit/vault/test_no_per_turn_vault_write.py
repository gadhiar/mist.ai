"""R1.3.1 guard: the per-turn path performs no SESSION-NOTE write.

Traps at the side-effect boundary (L19) rather than asserting a method does
not exist -- the latter passes if the behavior returns under a new name.

Final-review fix (I2): the invariant this file guards is "no per-turn
SESSION-NOTE write," not "no vault write at all." The pre-existing ADR-010
C-pattern user-snapshot writeback (`upsert_user_snapshot`, fired by
`_maybe_refresh_user_vault`) DOES write to the vault per-turn whenever
extraction touches user scope, and always has -- R1.3.1 never touched that
path. The prior version of this file asserted `writer.writes == []`
unconditionally and only passed because its fake extraction pipeline never
emitted a user-scope entity, so it never exercised that branch. That gap
made a per-turn SESSION-NOTE write GATED ON user-scope extraction invisible
to this guard: inserted into `_maybe_refresh_user_vault` (the existing
per-turn write site, and the natural place such a change would land), it
left the full suite green. The same write placed directly in
`handle_message` (unconditionally) is still caught -- the guard is live for
the shape it covers, it just did not cover enough shapes.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from backend.chat.conversation_handler import ConversationHandler
from backend.knowledge.extraction.validator import ValidationResult
from backend.knowledge.retrieval.knowledge_retriever import KnowledgeRetriever
from backend.knowledge.storage.graph_store import GraphStore
from tests.mocks.config import build_test_config
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeNeo4jConnection
from tests.mocks.ollama import FakeLLM
from tests.unit.conftest import make_test_conventions_loader

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _RecordingVaultWriter:
    """Records every write attempt. Never raises, so a swallowed failure
    cannot masquerade as 'no write occurred'.

    `session_path` is a genuinely synchronous, non-enqueued method on the
    real `VaultWriter` (pure path computation, no queue involved) -- built
    explicitly here rather than through `__getattr__` per L20, because the
    catch-all returns an async closure that `_get_or_allocate_vault_path`
    would call without awaiting, silently caching a coroutine object as the
    session's vault path instead of a string.
    """

    def __init__(self) -> None:
        self.writes: list[tuple[str, dict]] = []

    def session_path(self, session_date: str, session_slug: str) -> str:
        return f"/tmp/vault/sessions/{session_date}-{session_slug}.md"

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)

        async def _record(*args, **kwargs):
            self.writes.append((name, kwargs))
            return None

        return _record


class _FakeExtractionPipeline:
    """Minimal extraction pipeline double, shaped like the real
    `ValidationResult` (L20: built from the real collaborator's fields, not
    invented ones).

    `user_scope=False` (default) yields one non-User-scope entity
    (`Technology`) -- empty entities/relationships would make the
    zero-per-turn-write assertion vacuous, since the deleted
    `_maybe_append_session_turn` gate was itself a no-op on exactly that
    input. `user_scope=True` yields the User entity itself
    (`entity_id="user"`), which is `extraction_touched_user_scope`'s
    trigger condition for `_maybe_refresh_user_vault`'s C-pattern
    writeback (`backend/vault/user_snapshot.py::extraction_touched_user_scope`)
    -- the branch the prior version of this file never exercised.
    """

    def __init__(self, *, user_scope: bool = False) -> None:
        self.calls: list[dict] = []
        self._user_scope = user_scope

    async def extract_from_utterance(self, **kwargs):
        self.calls.append(kwargs)
        if self._user_scope:
            entities = [{"entity_id": "user", "entity_type": "Person", "display_name": "User"}]
        else:
            entities = [
                {"entity_id": "python", "entity_type": "Technology", "display_name": "Python"}
            ]
        return ValidationResult(valid=True, entities=entities, relationships=[])


# ---------------------------------------------------------------------------
# Fixture -- mirrors the make_handler() construction pattern from
# tests/unit/chat/test_conversation_handler_vault_integration.py
# ---------------------------------------------------------------------------


def _make_handler_with_recording_writer(
    *, user_scope: bool
) -> tuple[ConversationHandler, _RecordingVaultWriter]:
    conn = FakeNeo4jConnection()
    gs = GraphStore(conn, FakeEmbeddingGenerator())
    config = build_test_config(
        event_store_enabled=True,
        event_store_db_path=":memory:",
    )
    writer = _RecordingVaultWriter()
    handler = ConversationHandler(
        config=config,
        graph_store=gs,
        extraction_pipeline=_FakeExtractionPipeline(user_scope=user_scope),
        retriever=KnowledgeRetriever(config=config, graph_store=gs),
        llm_provider=FakeLLM(),
        conventions_loader=make_test_conventions_loader(),
        vault_writer=writer,
    )
    return handler, writer


@pytest_asyncio.fixture
async def conversation_handler_with_recording_writer():
    """Non-user-scope extraction (the common case): NO vault write of any
    kind should occur per-turn.
    """
    yield _make_handler_with_recording_writer(user_scope=False)


@pytest_asyncio.fixture
async def conversation_handler_with_recording_writer_user_scope():
    """User-scope extraction: the pre-existing C-pattern user-snapshot
    writeback IS expected to fire per-turn; only a SESSION-NOTE write is
    forbidden.
    """
    yield _make_handler_with_recording_writer(user_scope=True)


# ---------------------------------------------------------------------------
# Guard tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_message_performs_no_vault_write_for_non_user_scope(
    conversation_handler_with_recording_writer,
):
    """The narrower, TRUE zero-write case: a non-user-scope entity
    (Technology) triggers neither the retired per-turn session-note append
    nor the C-pattern user-snapshot writeback -- nothing in the vault
    should be touched at all here.
    """
    handler, writer = conversation_handler_with_recording_writer

    await handler.handle_message("I use Python for data-intensive systems", session_id="s-1")
    await handler._drain_extraction_tasks()

    assert (
        writer.writes == []
    ), f"R1.3.1: no vault write may occur on the per-turn path; got {writer.writes}"


@pytest.mark.asyncio
async def test_handle_message_performs_no_session_note_write_for_user_scope(
    conversation_handler_with_recording_writer_user_scope,
):
    """The general invariant, exercised on the branch the prior version of
    this file never reached: user-scope extraction is EXPECTED to fire the
    pre-existing C-pattern `upsert_user_snapshot` writeback, but must still
    never write a session note. Asserts on the KIND of write, not the
    count -- `writer.writes == []` would be false here by design.
    """
    handler, writer = conversation_handler_with_recording_writer_user_scope

    await handler.handle_message("My name is Alex", session_id="s-2")
    await handler._drain_extraction_tasks()

    write_names = [name for name, _ in writer.writes]
    assert (
        "write_session_note" not in write_names
    ), f"R1.3.1: no per-turn SESSION-NOTE write may occur; got {writer.writes}"


@pytest.mark.asyncio
async def test_end_session_writes_exactly_one_session_note(
    conversation_handler_with_recording_writer,
):
    handler, writer = conversation_handler_with_recording_writer

    await handler.handle_message("first turn about Python", session_id="s-2")
    await handler.handle_message("second turn about Neo4j", session_id="s-2")
    await handler._drain_extraction_tasks()
    await handler.end_session(session_id="s-2")

    names = [name for name, _ in writer.writes]
    assert names.count("write_session_note") == 1
    assert "append_turn_to_session" not in names
    assert "append_session_synthesis" not in names
    assert "mark_session_completed" not in names
