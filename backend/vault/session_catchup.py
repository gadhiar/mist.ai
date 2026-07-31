"""Startup catch-up for sessions interrupted before session-end synthesis.

Cost control is the whole design here. The expensive operation is an LLM
synthesis call, so every gate exists to avoid making one: sessions that
produced no graph state are skipped outright, a session whose note at its
canonical path is already done (recorded, or user-authoritative and
therefore not ours to touch) is skipped, repeated synthesis failure is
marked so it stops being retried, the LLM backend must report itself ready
before a pass attempts any synthesis at all, and the whole pass yields while
a conversation is active. `run_forever` repeats the pass on an interval
rather than firing once at boot, so a cold LLM or a live conversation at
startup does not permanently strand the backlog for the life of the process.

R1.3.1 fix round 1: dedup is keyed on the note's own canonical path
(`session_path_for`), not on a frontmatter `session_id` field. The vault
writer's frontmatter `session_id` is always the path-derived slug
(`_session_id_from_path` in `backend/vault/writer.py`), never any external
or internal session identifier -- matching against it could never actually
correlate to a catch-up candidate. Checking "does my own canonical path
already have a note" needs no id at all: the live path and catch-up already
agree on where a session's note belongs via the shared slug algorithm, so
that agreement alone is the dedup key.

Final review, second round: "already done" is one decision, not two.
`_note_is_not_ours_to_write` treats a `completed`/`skipped` status and a
`user`/`user-edit` `authored_by` as the same underlying condition -- both
mean synthesizing again is pure waste. Splitting them into separate checks
was tried and rejected: `VaultWriter.write_session_note` refuses a
`user`/`user-edit` note silently (no exception, no status written), so a
status-only dedup check would never see that refusal reflected on disk and
would re-attempt synthesis -- a real LLM call -- on every `run_forever`
tick, forever.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING

from backend.vault.models import parse_frontmatter

if TYPE_CHECKING:
    from backend.chat.session_synthesizer import SessionSynthesizer
    from backend.event_store.store import EventStore
    from backend.vault.writer import VaultWriter

logger = logging.getLogger(__name__)

_MAX_ATTEMPTS_PER_PASS = 2
_DEFAULT_RETRY_INTERVAL_SECONDS = 300.0


class SessionNoteCatchup:
    """Synthesizes vault notes for sessions that crashed before session end."""

    def __init__(
        self,
        event_store: EventStore,
        synthesizer: SessionSynthesizer,
        vault_writer: VaultWriter,
        sessions_with_graph_state: Callable[[], set[str]],
        session_path_for: Callable[[str, str, str], str | None],
        is_conversation_active: Callable[[], bool],
        is_llm_ready: Callable[[], Awaitable[bool]],
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
            sessions_with_graph_state: Sync callable returning the set of
                session ids that produced at least one graph entity (the
                efficiency gate). Wired to a `GraphStore` method, which
                queries Neo4j synchronously -- `run()` calls it through
                `run_in_executor` rather than directly, per the codebase
                rule that sync Neo4j access must not happen on the event
                loop.
            session_path_for: Derives a session's note path from its first
                turn, reusing the live path's slug algorithm. Returns None
                when the vault layer is disabled. May raise (e.g. a
                pathological utterance producing an invalid slug); `run()`
                treats that as "skip this one session," never as a reason
                to abort the rest of the backlog.
            is_conversation_active: Cheap sync check (a live-session map
                being non-empty) -- read once per candidate so catch-up
                yields mid-pass, not just at the start.
            is_llm_ready: Async readiness probe (`SessionSynthesizer
                .is_ready`, which delegates to the LLM provider's
                `health_check`). A whole pass is deferred, not attempted,
                while this reports False -- a cold backend at boot must not
                be indistinguishable from a genuine synthesis failure, or
                every pending session gets permanently marked `skipped`
                before the model has even finished loading.
        """
        self._event_store = event_store
        self._synthesizer = synthesizer
        self._writer = vault_writer
        self._sessions_with_graph_state = sessions_with_graph_state
        self._session_path_for = session_path_for
        self._is_conversation_active = is_conversation_active
        self._is_llm_ready = is_llm_ready

    async def run(self) -> None:
        """One catch-up pass.

        Idempotent: a session whose canonical-path note is already done --
        recorded (`completed`/`skipped`) or user-authoritative
        (`authored_by` `user`/`user-edit`) -- is never redone. Safe to call
        repeatedly -- `run_forever` does exactly that.
        """
        if self._is_conversation_active():
            logger.debug("Catch-up deferred: a conversation is active")
            return

        try:
            candidates = self._event_store.list_sessions_with_turns()
        except Exception as exc:  # noqa: BLE001 -- catch-up is best-effort
            logger.warning("Catch-up could not list sessions (non-fatal): %s", exc)
            return

        if not candidates:
            # Nothing to do -- checked BEFORE the LLM readiness probe below
            # deliberately (R1.3.1 fix round 2, N2). The probe is an HTTP
            # call to llama-server; an idle backlog must not pay that cost
            # every tick forever just to decide it has nothing to attempt.
            return

        try:
            ready = await self._is_llm_ready()
        except Exception as exc:  # noqa: BLE001 -- readiness check is best-effort
            logger.warning("Catch-up could not check LLM readiness (non-fatal): %s", exc)
            return
        if not ready:
            logger.debug("Catch-up deferred: LLM backend not ready")
            return

        # Off the event loop: a Neo4j query. A large backlog must not stall
        # the WebSocket handler that boot is racing against (efficiency
        # gate: yield to live traffic).
        loop = asyncio.get_running_loop()
        try:
            with_facts = await loop.run_in_executor(None, self._sessions_with_graph_state)
        except Exception as exc:  # noqa: BLE001 -- catch-up is best-effort
            logger.warning("Catch-up could not query graph state (non-fatal): %s", exc)
            return

        for session_id in candidates:
            if self._is_conversation_active():
                logger.debug("Catch-up yielding mid-pass: a conversation started")
                return

            if session_id not in with_facts:
                logger.debug("Catch-up skipping %s: produced no graph state", session_id)
                continue

            await self._synthesize_one(session_id)

    async def run_forever(self, interval_seconds: float = _DEFAULT_RETRY_INTERVAL_SECONDS) -> None:
        """Run catch-up passes periodically for the life of the process.

        A single `run()` pass can come back empty-handed for reasons that
        are transient, not permanent: the LLM backend is still cold at
        boot, or a conversation was live for the whole pass. A one-shot
        catch-up would let either of those permanently strand the backlog
        until the next restart. Looping means the next tick picks up
        wherever this one left off -- every gate in `run()` is already
        idempotent, so repeated invocation is safe by construction.
        """
        while True:
            try:
                await self.run()
            except Exception as exc:  # noqa: BLE001 -- must survive to the next tick
                logger.warning("Session-note catch-up pass raised (non-fatal): %s", exc)
            await asyncio.sleep(interval_seconds)

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
        try:
            path = self._session_path_for(session_id, str(first.get("user_utterance", "")), date)
        except Exception as exc:  # noqa: BLE001 -- one bad session must not abort the backlog
            logger.warning(
                "Catch-up could not derive a note path for %s (non-fatal, skipping): %s",
                session_id,
                exc,
            )
            return
        if path is None:
            return

        loop = asyncio.get_running_loop()
        try:
            not_ours_to_write = await loop.run_in_executor(
                None, self._note_is_not_ours_to_write, path
            )
        except Exception as exc:  # noqa: BLE001 -- best-effort
            logger.warning(
                "Catch-up could not read existing note status for %s (non-fatal): %s",
                session_id,
                exc,
            )
            not_ours_to_write = False
        if not_ours_to_write:
            return

        for attempt in range(1, _MAX_ATTEMPTS_PER_PASS + 1):
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

        # Exhausted attempts with the LLM reporting itself ready throughout
        # this pass. Persist the skip in the vault so it survives a restart
        # -- an in-memory counter would make this a per-boot tax, and would
        # not survive `run_forever`'s later ticks either.
        try:
            await self._writer.write_session_note(
                vault_note_path=path, synthesis=None, status="skipped"
            )
        except Exception as exc:  # noqa: BLE001 -- Invariant 6
            logger.warning("Catch-up skip-marker write failed for %s: %s", session_id, exc)

    def _note_is_not_ours_to_write(self, path: str) -> bool:
        """Whether an existing note at `path` means this session is done.

        True when the note is already recorded (`status` is `completed` or
        `skipped`) OR is user-authoritative (`authored_by` is `user` or
        `user-edit`). Both are the same underlying decision -- "catch-up
        has nothing left to do here" -- expressed as one check rather than
        as two sibling special cases. That unification is load-bearing:
        `VaultWriter.write_session_note` refuses to touch a `user`/
        `user-edit` note (ADR-010 Invariant 5), and that refusal is silent
        to this caller (no exception, no status written). Treating
        `authored_by` as a SEPARATE condition from `status` would leave a
        user-protected note sitting at whatever non-terminal `status` it
        already has (typically `in-progress`, the legacy default for a
        session that crashed before completion) -- so this gate would
        never trip, and every `run_forever` tick would re-synthesize (a
        real LLM call) only to have the write refused, forever.

        A single targeted file read, not a directory scan -- the caller
        already knows exactly where this session's note would live.
        """
        note = Path(path)
        if not note.exists():
            return False
        try:
            fm, _ = parse_frontmatter(note.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 -- unreadable note is treated as absent
            return False
        return fm.get("status") in ("completed", "skipped") or fm.get("authored_by") in (
            "user",
            "user-edit",
        )
