"""Tests for the C-pattern user-vault writeback TARGET.

`ConversationHandler._maybe_refresh_user_vault` renders a graph-derived
user snapshot after extraction and must persist it to a SEPARATE machine-
owned derived file (`users/<user_id>-graph-snapshot.md`) via
`VaultWriter.upsert_user_snapshot`. It must NOT call `upsert_user`, which
would clobber the hand-curated `users/<user_id>.md` (ADR-010: the curated
profile is user-authoritative).

The snapshot helpers (`extraction_touched_user_scope`, `query_user_snapshot`,
`render_user_snapshot_body`) are imported LAZILY inside the method body
(`from backend.vault.user_snapshot import ...` at call time), so they are not
bound to the `backend.chat.conversation_handler` module namespace. The correct
monkeypatch target is therefore `backend.vault.user_snapshot.<name>`.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from backend.chat.conversation_handler import ConversationHandler


def test_refresh_writes_snapshot_not_user_md(monkeypatch):
    # Arrange: construct a ConversationHandler without running __init__ and
    # wire only the attributes _maybe_refresh_user_vault reads.
    handler = ConversationHandler.__new__(ConversationHandler)
    handler._vault_writer = MagicMock()
    handler._vault_writer.upsert_user_snapshot = AsyncMock(
        return_value="users/user-graph-snapshot.md"
    )
    handler._vault_writer.upsert_user = AsyncMock(return_value="users/user.md")
    handler.graph_store = MagicMock()
    handler._user_id_for_vault = lambda: "user"

    extraction_result = MagicMock(
        validated_entities=[MagicMock()],
        validated_relationships=[],
    )

    # The helpers are imported lazily from backend.vault.user_snapshot inside
    # the method, so patch at that module (not on conversation_handler).
    monkeypatch.setattr(
        "backend.vault.user_snapshot.extraction_touched_user_scope",
        lambda *a, **k: True,
    )
    monkeypatch.setattr(
        "backend.vault.user_snapshot.query_user_snapshot",
        AsyncMock(return_value=MagicMock(edges_by_type={})),
    )
    monkeypatch.setattr(
        "backend.vault.user_snapshot.render_user_snapshot_body",
        lambda snapshot: "BODY",
    )

    # The method also builds a GraphExecutor(self.graph_store.connection);
    # stub it so no real Neo4j wiring is touched.
    monkeypatch.setattr(
        "backend.knowledge.storage.graph_executor.GraphExecutor",
        lambda *a, **k: MagicMock(),
    )

    # Act
    asyncio.run(handler._maybe_refresh_user_vault(extraction_result))

    # Assert: the snapshot writeback fired; the curated user.md writer did not.
    handler._vault_writer.upsert_user_snapshot.assert_awaited_once()
    handler._vault_writer.upsert_user.assert_not_awaited()

    # The snapshot write carries the real user_id and the rendered body.
    _, kwargs = handler._vault_writer.upsert_user_snapshot.call_args
    assert kwargs.get("user_id") == "user"
    assert kwargs.get("body_markdown") == "BODY"
