"""Startup catch-up for sessions interrupted before session-end synthesis.

Cost control is the whole design here. The expensive operation is an LLM
synthesis call, so every gate exists to avoid making one: sessions that
produced no graph state are skipped outright, sessions already recorded are
skipped, repeated failures are marked so they stop being retried, and the
whole pass yields while a conversation is active.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from backend.vault.models import parse_frontmatter

if TYPE_CHECKING:
    from backend.chat.session_synthesizer import SessionSynthesizer
    from backend.event_store.store import EventStore
    from backend.vault.writer import VaultWriter

logger = logging.getLogger(__name__)

_MAX_ATTEMPTS_PER_BOOT = 2


class SessionNoteCatchup:
    """Synthesizes vault notes for sessions that crashed before session end."""

    def __init__(
        self,
        event_store: EventStore,
        synthesizer: SessionSynthesizer,
        vault_writer: VaultWriter,
        vault_sessions_dir: Path,
        sessions_with_graph_state: Callable[[], set[str]],
        session_path_for: Callable[[str, str, str], str | None],
        is_conversation_active: Callable[[], bool],
    ) -> None:
        """Wire the collaborators. See module docstring for the cost model.

        Args:
            event_store: Source of truth for which sessions have turns and
                what those turns contain. Called synchronously -- SQLite is
                local and this mirrors the existing `ConversationHandler
                .end_session` convention, unlike the Neo4j call below.
            synthesizer: `SessionSynthesizer` (or a test double with the
                same `async synthesize(turns) -> SessionSynthesis | None`
                shape).
            vault_writer: `VaultWriter` (or a test double with the same
                `async write_session_note(...)` shape).
            vault_sessions_dir: `<vault_root>/sessions` -- scanned for
                existing note status.
            sessions_with_graph_state: Sync callable returning the set of
                session ids that produced at least one graph entity (the
                efficiency gate). Wired to a `GraphStore` method, which
                queries Neo4j synchronously -- `run()` calls it through
                `run_in_executor` rather than directly, per the codebase
                rule that sync Neo4j access must not happen on the event
                loop.
            session_path_for: Derives a session's note path from its first
                turn, reusing the live path's slug algorithm. Returns None
                when the vault layer is disabled.
            is_conversation_active: Cheap sync check (a live-session map
                being non-empty) -- read once per candidate so catch-up
                yields mid-pass, not just at the start.
        """
        self._event_store = event_store
        self._synthesizer = synthesizer
        self._writer = vault_writer
        self._sessions_dir = vault_sessions_dir
        self._sessions_with_graph_state = sessions_with_graph_state
        self._session_path_for = session_path_for
        self._is_conversation_active = is_conversation_active

    async def run(self) -> None:
        """One catch-up pass. Idempotent: a completed session is never redone."""
        if self._is_conversation_active():
            logger.debug("Catch-up deferred: a conversation is active")
            return

        try:
            candidates = self._event_store.list_sessions_with_turns()
        except Exception as exc:  # noqa: BLE001 -- catch-up is best-effort
            logger.warning("Catch-up could not list sessions (non-fatal): %s", exc)
            return

        if not candidates:
            return

        loop = asyncio.get_running_loop()

        # Both scans are potentially non-trivial I/O -- a vault directory
        # listing and frontmatter parse per note, a Neo4j query -- so both
        # run off the event loop. A large backlog must not stall the
        # WebSocket handler that boot is racing against (efficiency gate:
        # yield to live traffic).
        try:
            recorded = await loop.run_in_executor(None, self._recorded_statuses)
        except Exception as exc:  # noqa: BLE001 -- catch-up is best-effort
            logger.warning("Catch-up could not scan existing notes (non-fatal): %s", exc)
            return

        try:
            with_facts = await loop.run_in_executor(None, self._sessions_with_graph_state)
        except Exception as exc:  # noqa: BLE001 -- catch-up is best-effort
            logger.warning("Catch-up could not query graph state (non-fatal): %s", exc)
            return

        for session_id in candidates:
            if self._is_conversation_active():
                logger.debug("Catch-up yielding mid-pass: a conversation started")
                return

            status = recorded.get(session_id)
            if status in ("completed", "skipped"):
                continue
            if session_id not in with_facts:
                logger.debug("Catch-up skipping %s: produced no graph state", session_id)
                continue

            await self._synthesize_one(session_id)

    async def _synthesize_one(self, session_id: str) -> None:
        """Synthesize and write one session's note, bounded-retry on failure."""
        try:
            turns = self._event_store.get_turns(session_id)
        except Exception as exc:  # noqa: BLE001 -- best-effort
            logger.warning("Catch-up could not read turns for %s: %s", session_id, exc)
            return
        if not turns:
            return

        # The note path is derived from the FIRST turn, matching how a live
        # session allocates it -- same slug algorithm, and (for catch-up
        # specifically) the date the session actually happened on rather
        # than today's date, since this may run long after the session.
        first = turns[0]
        date = str(first.get("timestamp", ""))[:10]
        path = self._session_path_for(session_id, str(first.get("user_utterance", "")), date)
        if path is None:
            return

        for attempt in range(1, _MAX_ATTEMPTS_PER_BOOT + 1):
            try:
                synthesis = await self._synthesizer.synthesize(turns)
            except Exception as exc:  # noqa: BLE001 -- best-effort
                logger.warning(
                    "Catch-up synthesis raised for %s (attempt %d): %s",
                    session_id,
                    attempt,
                    exc,
                )
                synthesis = None

            if synthesis is not None:
                try:
                    await self._writer.write_session_note(
                        vault_note_path=path, synthesis=synthesis, related_entities=[]
                    )
                except Exception as exc:  # noqa: BLE001 -- Invariant 6
                    logger.warning(
                        "Catch-up note write failed for %s (non-fatal): %s", session_id, exc
                    )
                return

        # Exhausted attempts. Persist the skip in the vault so it survives a
        # restart -- an in-memory counter would make this a per-boot tax.
        try:
            await self._writer.write_session_note(
                vault_note_path=path, synthesis=None, status="skipped"
            )
        except Exception as exc:  # noqa: BLE001 -- Invariant 6
            logger.warning("Catch-up skip-marker write failed for %s: %s", session_id, exc)

    def _recorded_statuses(self) -> dict[str, str]:
        """Map session_id -> frontmatter status for existing notes.

        One directory listing plus a frontmatter read per note. A note that
        cannot be parsed is treated as absent and re-rendered, rather than
        parsed defensively.
        """
        statuses: dict[str, str] = {}
        if not self._sessions_dir.exists():
            return statuses

        for note in self._sessions_dir.glob("*.md"):
            try:
                fm, _ = parse_frontmatter(note.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 -- unreadable note = not recorded
                continue  # nosec B112 -- skipping one bad note is not a security decision
            sid = fm.get("session_id")
            status = fm.get("status")
            if sid and status:
                statuses[str(sid)] = str(status)
        return statuses
