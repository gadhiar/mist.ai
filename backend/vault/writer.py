"""Serialized vault writer for the MIST memory layer (ADR-010 Cluster 8).

All vault writes are serialized through a single asyncio.Queue consumer so
that concurrent callers cannot race on the same note file. Each caller
enqueues a `_WriteJob`, awaits its `result_future`, and receives either the
operation's return value or a `VaultWriteError` exception.

Per ADR-010 Invariant 6, vault write failures are swallowed at the consumer
boundary -- the graph is rebuildable from the event store alone if the vault
write fails. Errors are logged and set on the job's future; they never
propagate into the chat path.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from backend.errors import VaultWriteError
from backend.knowledge.config import VaultConfig
from backend.knowledge.version_stamps import EXTRACTION_VERSION, ONTOLOGY_VERSION
from backend.vault.models import (
    AuthoredBy,
    MistIdentityFrontmatter,
    MistSessionFrontmatter,
    MistUserFrontmatter,
    parse_frontmatter,
    render_frontmatter,
)

if TYPE_CHECKING:
    # Deferred: backend.chat imports backend.vault (SessionSynthesizer writes
    # through VaultWriter), so a runtime import here would cycle back.
    from backend.chat.session_synthesizer import SessionSynthesis

logger = logging.getLogger(__name__)

_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# Matches the leading YYYY-MM-DD- date prefix in a session filename stem.
_STEM_DATE_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-")
# Multiline-anchored, case-insensitive match for a markdown `## Provenance`
# heading at line start. Used by `_upsert_user_sync` and
# `_upsert_identity_body_sync` to decide whether to append a writer-supplied
# default Provenance section. The line anchor rejects quoted forms like
# `> ## Provenance`; the case-insensitive flag accepts `## provenance` as
# the same logical section.
_PROVENANCE_HEADING_RE = re.compile(r"(?im)^##\s+Provenance\s*$")


def _session_id_from_path(path: Path) -> str:
    """Derive a deterministic frontmatter session_id from the session note path.

    The filename stem has the form ``YYYY-MM-DD-<slug>``. Stripping the date
    prefix returns the pre-allocated human-readable slug (e.g.
    ``plan-new-feature-37a8``), which is used as the canonical session_id in
    frontmatter.

    This eliminates the legacy fallback that wrote the raw external session_id
    argument directly into frontmatter. Five of seven session notes in the
    2026-05-10 audit had ``session_id: default`` because
    KnowledgeIntegration.current_session_id is initialised to ``"default"``
    and that raw string propagated into the frontmatter without transformation.

    Args:
        path: Absolute path to the session note file. The filename stem must
            start with a ``YYYY-MM-DD-`` date prefix.

    Returns:
        The slug portion of the stem (everything after the date prefix). Falls
        back to the full stem if the prefix is not present (defensive -- the
        caller should always pass a well-formed path from `session_path`).
    """
    stem = path.stem
    m = _STEM_DATE_PREFIX_RE.match(stem)
    if m:
        return stem[m.end() :]
    return stem


def _date_from_path(path: Path) -> str:
    """Derive a deterministic frontmatter `date` from the session note path.

    Uses the same `YYYY-MM-DD-` stem prefix as `_session_id_from_path` --
    one rule rather than two -- so a full render of `write_session_note` is a
    pure function of its arguments, with no wall-clock read. This also makes
    the date *correct* rather than merely stable: a session that happened
    yesterday and is synthesized today (startup catch-up) gets the date it
    actually occurred on, not the render date.

    There is deliberately NO wall-clock fallback for a non-conforming stem.
    A fallback would make `write_session_note` idempotent for every path
    `session_path()` produces but silently non-deterministic for any other
    path -- a trap for a future caller (e.g. catch-up) that does not route
    through `session_path()`. Raising instead makes "this method requires a
    canonical path" an enforced precondition rather than a hoped-for one.

    Args:
        path: Absolute path to the session note file.

    Returns:
        The `YYYY-MM-DD` prefix from the filename stem.

    Raises:
        VaultWriteError: If the stem does not start with a `YYYY-MM-DD-`
            prefix, i.e. the path was not produced by `session_path()`.
    """
    stem = path.stem
    m = _STEM_DATE_PREFIX_RE.match(stem)
    if m:
        return m.group(0)[:-1]  # drop the trailing "-" the regex captures
    raise VaultWriteError(
        f"write_session_note requires a canonical session_path() output "
        f"(stem 'YYYY-MM-DD-<slug>'); got stem {stem!r} from {path}"
    )


# ---------------------------------------------------------------------------
# Internal job model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _WriteJob:
    """Unit of work enqueued by public write methods and consumed serially.

    `kind` dispatches to the correct handler inside the consumer coroutine.
    `args` carries all positional data needed by that handler.
    `result_future` is resolved (value or exception) by the consumer;
    the awaiting caller receives the result via `await job.result_future`.
    """

    kind: str
    args: dict[str, Any]
    result_future: asyncio.Future  # type: ignore[type-arg]


# Sentinel object placed in the queue to signal consumer shutdown
_STOP = object()

# ---------------------------------------------------------------------------
# VaultWriter
# ---------------------------------------------------------------------------


class VaultWriter:
    """Serialized writer for vault markdown notes (ADR-010 Layer 2).

    One asyncio consumer processes writes in enqueue order. Callers await
    completion and receive either a return value or a `VaultWriteError`.

    Usage::

        writer = VaultWriter(config)
        await writer.start()
        path = await writer.write_session_note(vault_note_path, synthesis)
        await writer.stop()
    """

    def __init__(
        self,
        config: VaultConfig,
        debug_logger: Any = None,
        model_hash: str | None = None,
    ) -> None:
        """Initialize the vault writer.

        Args:
            config: VaultConfig with root path + lifecycle flags.
            debug_logger: Optional DebugJSONLLogger (Cluster 8 Phase 12). When
                set + `MIST_DEBUG_VAULT_JSONL=1`, every consumer-side write
                op emits a `phase: "vault"` JSONL record with operation,
                path, duration_ms, ok, and any error_message. None preserves
                pre-Phase-12 silent operation.
            model_hash: Optional Phase 8 rebuild-determinism stamp. Mirrors
                the same value used for the EXTRACTED_FROM->ConversationContext
                and reconciled fact edges (via `RebuildStamps` on
                `CurationGraphWriter` -- R1.3 moved this anchor off
                DERIVED_FROM->VaultNote). When provided, populates the
                `model_hash` frontmatter field on every newly created session
                note. None preserves pre-fix behavior (frontmatter
                `model_hash: null`).
        """
        self.config = config
        self._root = Path(config.root)
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._consumer_task: asyncio.Task | None = None  # type: ignore[type-arg]
        self._stopped = False
        self._debug_logger = debug_logger
        self._model_hash = model_hash
        # Filewatcher MIST-write marker (VaultFilewatcher.mark_mist_write),
        # wired by build_phase3_components. Every consumer handler calls it
        # immediately before mutating a file so the filewatcher classifies
        # the resulting event as MIST-origin instead of a user edit. There
        # is deliberately NO clear-after-write: the watchdog event may
        # arrive after the write completes, so an eager clear would re-open
        # the user-edit misclassification; expired markers are reaped by
        # the filewatcher's TTL cleanup.
        self._mist_write_marker: Callable[[str], None] | None = None

    def set_mist_write_marker(self, marker: Callable[[str], None]) -> None:
        """Wire the filewatcher's mark_mist_write so consumer writes self-mark.

        Args:
            marker: Callable receiving the absolute path string about to be
                written (typically `VaultFilewatcher.mark_mist_write`).
        """
        self._mist_write_marker = marker

    def _mark_mist_write(self, path: Path | str) -> None:
        """Mark `path` as a MIST-origin write; never breaks the write path."""
        if self._mist_write_marker is None:
            return
        try:
            self._mist_write_marker(str(path))
        except Exception as exc:  # noqa: BLE001 -- marking must not block writes
            logger.debug("MIST-write marker failed for %s (non-fatal): %s", path, exc)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the consumer task and ensure vault directory structure exists.

        Creates required subdirectories under `config.root`. Optionally
        runs `git init` if `config.git_auto_init` is True and no `.git`
        directory is present. Idempotent -- safe to call multiple times.
        """
        if self._consumer_task is not None and not self._consumer_task.done():
            return

        self._stopped = False
        await self._ensure_directories()

        if self.config.git_auto_init:
            await self._maybe_git_init()

        self._consumer_task = asyncio.create_task(self._consume(), name="vault-writer-consumer")

    async def stop(self) -> None:
        """Drain the queue and shut down the consumer task.

        Safe to call multiple times. Waits for the consumer to process
        all enqueued jobs before returning.
        """
        if self._stopped:
            return
        self._stopped = True

        await self._queue.put(_STOP)
        if self._consumer_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._consumer_task
        self._consumer_task = None

    # ------------------------------------------------------------------
    # Public write methods (each enqueues a job and awaits its future)
    # ------------------------------------------------------------------

    async def upsert_identity(
        self,
        traits: list[dict],
        capabilities: list[dict],
        preferences: list[dict],
        rendered_at: str | None = None,
    ) -> str:
        """Write or overwrite the MIST identity note at `identity/mist.md`.

        Renders traits, capabilities, and preferences into structured
        markdown sections. Sorted alphabetically by `display_name` for
        determinism -- identical inputs always produce byte-identical output.

        Args:
            traits: Dicts with at least `display_name`, `description`, and
                optionally `axis`.
            capabilities: Dicts with at least `display_name` and
                `description`.
            preferences: Dicts with at least `display_name`, `description`,
                and optionally `enforcement` and `context`.
            rendered_at: Optional ISO 8601 timestamp string. When supplied,
                pins the frontmatter `last_updated` date and the Provenance
                `rendered_at` so the seeded identity note is byte-reproducible
                (seed bootstrap under replay). None means wall-clock -- the
                unchanged production default.

        Returns:
            Absolute path to the identity note.

        Raises:
            VaultWriteError: If the file write fails.
        """
        return await self._enqueue(
            "upsert_identity",
            {
                "traits": traits,
                "capabilities": capabilities,
                "preferences": preferences,
                "rendered_at": rendered_at,
            },
        )

    async def upsert_identity_body(
        self, body_markdown: str, source_path: str, rendered_at: str | None = None
    ) -> str:
        """Write or overwrite the MIST identity note at `identity/mist.md` from a prepared body.

        Sibling to `upsert_identity`, added for R1.4 Task 10: where that
        method renders structured trait/capability/preference dicts into
        markdown, this one writes a caller-provided body verbatim -- the
        shape the R1.4 seed source produces (`SeedDocument.body`, authored
        directly in `mist-memory/seed/mist.md` rather than assembled from
        per-item dicts `SeedFact` has no fields for). `upsert_identity` has
        no production caller left after this task but keeps its Protocol
        entry and existing tests untouched; this method is additive, not a
        replacement.

        Args:
            body_markdown: Caller-provided markdown body. A `## Provenance`
                section is appended automatically when the body does not
                already supply one (mirrors `upsert_user`'s dedup guard via
                `_PROVENANCE_HEADING_RE`).
            source_path: Origin of `body_markdown`, rendered into the
                appended Provenance section's `source` field (e.g. the
                seed document's `source_path`). Passed in rather than
                hardcoded so the footer never points at a deleted file.
            rendered_at: Optional ISO 8601 timestamp string. When supplied,
                pins the frontmatter `last_updated` date and the Provenance
                `rendered_at` so the seeded identity note is byte-reproducible
                (seed bootstrap under replay). None means wall-clock -- the
                unchanged production default.

        Returns:
            Absolute path to the identity note.

        Raises:
            VaultWriteError: If the file write fails.
        """
        return await self._enqueue(
            "upsert_identity_body",
            {
                "body_markdown": body_markdown,
                "source_path": source_path,
                "rendered_at": rendered_at,
            },
        )

    async def write_session_note(
        self,
        vault_note_path: str,
        synthesis: SessionSynthesis | None,
        related_entities: list[str] | None = None,
        status: str = "completed",
    ) -> str | None:
        """Render a session note in full and write it.

        R1.3.1: replaces `append_turn_to_session` + `append_session_synthesis`
        + `mark_session_completed`. Because the note is synthesis-only there is
        no accumulated content to preserve, so this renders the whole file --
        which makes it idempotent, and therefore safe to re-run after a partial
        failure. The rendered bytes are a pure function of `synthesis`,
        `related_entities`, `status`, `vault_note_path`, `self._model_hash`
        (fixed for the writer's lifetime), and the on-disk `authored_by` at
        the target path -- nothing else, including no wall-clock read. The
        on-disk `authored_by` is read back and carried forward (final-review
        fix, I1); when it is `user` or `user-edit` the write is refused
        entirely, per ADR-010 Invariant 5. The idempotency guarantee spans a
        backend restart because `model_hash` is stable across restarts in
        practice and a partial failure never changes `authored_by` on disk
        between retry attempts.

        Args:
            vault_note_path: Absolute path to the session note. Must be a
                `session_path()`-shaped path (stem `YYYY-MM-DD-<slug>`) --
                the date is derived from it, not read from the wall clock.
            synthesis: Prose from `SessionSynthesizer`. None renders a stub,
                which is how a `skipped` session is recorded.
            related_entities: Entity ids this session touched, for the
                prose-to-graph link. Deduped and sorted before rendering.
            status: `completed` normally; `skipped` when catch-up has given up
                on the session (bounded retry).

        Returns:
            The path written.

        Raises:
            VaultWriteError: If `vault_note_path`'s filename stem does not
                start with a `YYYY-MM-DD-` date prefix.
        """
        return await self._enqueue(
            "write_session_note",
            {
                "vault_note_path": vault_note_path,
                "synthesis": synthesis,
                "related_entities": related_entities or [],
                "status": status,
            },
        )

    async def mark_authored_by_user_edit(self, path: Path) -> None:
        """Set frontmatter authored_by=user-edit on a vault file.

        Implements ADR-010 Invariant 5 writeback: when the filewatcher detects
        that a human edited a vault file, the pipeline calls this method to
        flip `authored_by` so subsequent `upsert_user` calls respect the
        user-authoritative constraint and do not overwrite the body.

        Serialized through the consumer queue like every other write: the
        target may be the ACTIVE session note (user edits today's note in
        Obsidian while chatting), and an inline read-modify-write would race
        the consumer's append and silently drop a turn.

        Idempotent: a second call on a file already at `authored_by: user-edit`
        is a no-op (no rewrite, no temp file created). The handler self-marks
        the path as a MIST write so the writeback does not re-fire the
        filewatcher's user-edit sequence.

        Args:
            path: Absolute path to the vault markdown file to update.
        """
        await self._enqueue("mark_authored_by", {"path": str(path)})

    async def upsert_user(
        self, user_id: str, body_markdown: str, rendered_at: str | None = None
    ) -> str:
        """Write or update a user fact sheet at `users/<user_id>.md`.

        On existing files where `authored_by` is `user` or `user-edit`,
        the body is NOT overwritten (user's edits are authoritative per
        ADR-010 Invariant 5). Only `last_updated` and `related_sessions`
        are updated in that case.

        Args:
            user_id: User identifier (used as filename stem).
            body_markdown: Caller-provided markdown body. A `## Provenance`
                section is appended automatically.
            rendered_at: Optional ISO 8601 timestamp string. When supplied,
                it pins both the frontmatter `last_updated` date and the
                appended Provenance `rendered_at`, making the written file
                byte-reproducible (used by the seed bootstrap under replay).
                None means wall-clock -- the unchanged production default.

        Returns:
            Absolute path to the user note.

        Raises:
            VaultWriteError: If the file write fails.
        """
        return await self._enqueue(
            "upsert_user",
            {
                "user_id": user_id,
                "body_markdown": body_markdown,
                "rendered_at": rendered_at,
            },
        )

    async def upsert_user_snapshot(
        self, user_id: str, body_markdown: str, rendered_at: str | None = None
    ) -> str:
        """Write or update the graph-derived user snapshot.

        Persists the C-pattern machine writeback to a SEPARATE derived file
        `users/<user_id>-graph-snapshot.md`, decoupled from the hand-curated
        `users/<user_id>.md`. This snapshot is a machine-owned derived cache
        (graph-equivalent of the user's outbound edges), regenerated after
        extraction; it never competes with the curated profile in retrieval
        (excluded from sidecar indexing, see sidecar_index._is_excluded_from_indexing).

        Unlike `upsert_user`, the body is ALWAYS overwritten: the ADR-010
        Invariant-5 guard (which protects `authored_by: user`/`user-edit`
        bodies) does NOT apply here, because the snapshot is not user-editable.
        Frontmatter stays `mist-user` with `authored_by: mist`; the filename
        STEM is `<user_id>-graph-snapshot` while the `user_id` frontmatter
        field remains the real user_id.

        Args:
            user_id: User identifier. Used verbatim in the `user_id`
                frontmatter field and as the basis for the filename stem
                `<user_id>-graph-snapshot`.
            body_markdown: Caller-provided markdown body. A `## Provenance`
                section is appended automatically only when the body does not
                already supply one (same handling as `upsert_user`).
            rendered_at: Optional ISO 8601 timestamp string. When supplied, it
                pins the frontmatter `last_updated` date and any appended
                Provenance `rendered_at` for reproducible writes. None means
                wall-clock -- the unchanged production default. The C-pattern
                snapshot body already carries its own Provenance from
                `render_user_snapshot_body`, so this only affects the fallback
                writer section and the frontmatter date.

        Returns:
            Absolute path to the snapshot note.

        Raises:
            VaultWriteError: If the file write fails.
        """
        return await self._enqueue(
            "upsert_user_snapshot",
            {
                "user_id": user_id,
                "body_markdown": body_markdown,
                "rendered_at": rendered_at,
            },
        )

    def session_path(self, session_date: str, session_slug: str) -> str:
        """Return the absolute vault path for a session note.

        Pure function -- no I/O. Validates date format and slug format.

        Args:
            session_date: ISO date string `YYYY-MM-DD`.
            session_slug: Lowercase kebab-case session slug.

        Returns:
            Absolute path string: `<root>/sessions/<date>-<slug>.md`.

        Raises:
            ValueError: If `session_date` does not match `YYYY-MM-DD` or
                `session_slug` is not lowercase kebab-case.
        """
        if not _DATE_RE.fullmatch(session_date):
            raise ValueError(f"session_date must match YYYY-MM-DD, got: {session_date!r}")
        if not _SLUG_RE.fullmatch(session_slug):
            raise ValueError(f"session_slug must be lowercase kebab-case, got: {session_slug!r}")
        return str(self._root / "sessions" / f"{session_date}-{session_slug}.md")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _enqueue(self, kind: str, args: dict[str, Any]) -> Any:
        """Enqueue a write job and await its result.

        Logs a warning if the queue depth exceeds `writer_queue_max_depth`
        (backpressure signal per ADR-010 -- caller is not blocked per
        Invariant 6).

        Raises:
            VaultWriteError: When the consumer is not running (never started,
                already stopped, or crashed) -- enqueueing would await a
                future nothing will ever resolve.
        """
        if self._stopped or self._consumer_task is None or self._consumer_task.done():
            raise VaultWriteError(f"VaultWriter consumer is not running; cannot enqueue {kind!r}")
        if self._queue.qsize() > self.config.writer_queue_max_depth:
            logger.warning(
                "VaultWriter queue depth %d exceeds limit %d -- backpressure",
                self._queue.qsize(),
                self.config.writer_queue_max_depth,
            )

        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        job = _WriteJob(kind=kind, args=args, result_future=future)
        await self._queue.put(job)
        return await future

    async def _consume(self) -> None:
        """Consumer coroutine: processes jobs from the queue in order.

        Runs until the `_STOP` sentinel is dequeued. All handler exceptions
        are caught and set on the job's result_future; they never propagate
        into the caller's task.

        Phase 12: every dispatch is timed and a `phase: "vault"` debug record
        is emitted via the injected DebugJSONLLogger when the gate is on.
        Recording failures are swallowed -- observability never breaks the
        write path.
        """
        import time as _time

        while True:
            item = await self._queue.get()
            if item is _STOP:
                self._queue.task_done()
                break

            job: _WriteJob = item
            _start = _time.perf_counter()
            result_path: str | None = None
            ok = False
            error_message: str | None = None
            try:
                result = await self._dispatch(job)
                result_path = result if isinstance(result, str) else None
                ok = True
                if not job.result_future.done():
                    job.result_future.set_result(result)
            except VaultWriteError as exc:
                error_message = repr(exc)
                logger.error("VaultWriteError in consumer [%s]: %s", job.kind, exc)
                if not job.result_future.done():
                    job.result_future.set_exception(exc)
            except Exception as exc:  # noqa: BLE001
                wrapped = VaultWriteError(f"Unexpected error in vault consumer [{job.kind}]: {exc}")
                wrapped.__cause__ = exc
                error_message = repr(exc)
                logger.error(
                    "Unexpected error in vault consumer [%s]: %s",
                    job.kind,
                    exc,
                    exc_info=True,
                )
                if not job.result_future.done():
                    job.result_future.set_exception(wrapped)
            finally:
                duration_ms = (_time.perf_counter() - _start) * 1000.0
                self._maybe_record_vault_op(
                    operation=job.kind,
                    path=result_path,
                    duration_ms=duration_ms,
                    ok=ok,
                    error_message=error_message,
                    job_args=job.args,
                )
                self._queue.task_done()

    def _maybe_record_vault_op(
        self,
        *,
        operation: str,
        path: str | None,
        duration_ms: float,
        ok: bool,
        error_message: str | None,
        job_args: dict[str, Any],
    ) -> None:
        """Emit a `phase: "vault"` debug record. No-op when logger is None.

        Picks a small set of safe-to-serialize op-specific fields out of
        `job_args` so the record carries useful context (user_id for
        upsert_user) without leaking large payloads (full markdown).
        """
        if self._debug_logger is None:
            return
        try:
            extra: dict[str, Any] = {}
            if operation in ("upsert_user", "upsert_user_snapshot"):
                extra["user_id"] = job_args.get("user_id")

            self._debug_logger.record_vault_op(
                operation=operation,
                path=path,
                duration_ms=duration_ms,
                ok=ok,
                error_message=error_message,
                session_id=job_args.get("session_id"),
                extra=extra or None,
            )
        except Exception as exc:  # noqa: BLE001 -- never break the write path
            logger.debug("Vault debug record emission failed (non-fatal): %s", exc)

    async def _dispatch(self, job: _WriteJob) -> Any:
        """Route a job to its handler by `kind`."""
        handlers = {
            "upsert_identity": self._handle_upsert_identity,
            "upsert_identity_body": self._handle_upsert_identity_body,
            "upsert_user": self._handle_upsert_user,
            "upsert_user_snapshot": self._handle_upsert_user_snapshot,
            "mark_authored_by": self._handle_mark_authored_by,
            "write_session_note": self._handle_write_session_note,
        }
        handler = handlers.get(job.kind)
        if handler is None:
            raise VaultWriteError(f"Unknown job kind: {job.kind!r}")
        return await handler(job.args)

    # ------------------------------------------------------------------
    # Handlers (run inside consumer; may do blocking I/O via executor)
    # ------------------------------------------------------------------

    async def _handle_write_session_note(self, args: dict[str, Any]) -> str:
        path = Path(args["vault_note_path"])
        synthesis: SessionSynthesis | None = args["synthesis"]
        related_entities: list[str] = args["related_entities"]
        status: str = args["status"]

        self._mark_mist_write(path)
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                self._write_session_note_sync,
                path,
                synthesis,
                related_entities,
                status,
            )
        except OSError as exc:
            raise VaultWriteError(f"Failed to write session note {path}: {exc}") from exc

        return str(path)

    def _write_session_note_sync(
        self,
        path: Path,
        synthesis: SessionSynthesis | None,
        related_entities: list[str],
        status: str,
    ) -> None:
        """Synchronous full render, given the on-disk `authored_by`.

        Final-review fix (I1): every field EXCEPT `authored_by` is
        recomputed from the arguments, the path, and `self._model_hash` --
        `authored_by` is read back from the existing file (if any) and
        carried forward, mirroring `_upsert_user_sync` and
        `_upsert_identity_sync`. The prior version never read the existing
        file at all, so `authored_by` silently fell back to
        `MistSessionFrontmatter`'s default (`mist`) on every render --
        including over a note a user had hand-edited in Obsidian, which
        both flipped `authored_by` back to `mist` and discarded their prose
        (ADR-010 Invariant 5 violation). When the existing file's
        `authored_by` is `user` or `user-edit`, the write is refused
        entirely and logged -- there is no per-field update path for
        session notes the way `_upsert_user_sync` has (session notes carry
        no incrementally-touched field like `last_updated`), so "refuse"
        here means the file is not touched at all, closer to
        `_upsert_identity_sync`'s guard shape than `_upsert_user_sync`'s.

        This trades away being a pure function of only the passed-in
        arguments: calling this twice against the same path is idempotent
        given an unchanged on-disk `authored_by`, not unconditionally --
        the property Task 6's catch-up actually relies on (re-rendering
        after a partial failure) still holds, since a partial failure never
        changes `authored_by` on disk between attempts.

        Raises:
            VaultWriteError: Propagated from `_date_from_path` when `path`'s
                stem is not `session_path()`-shaped. Deliberately not
                caught here -- see that function's docstring.
        """
        canonical_session_id = _session_id_from_path(path)
        note_date = _date_from_path(path)

        authored_by = AuthoredBy.MIST
        if path.exists():
            existing_fm, _existing_body = parse_frontmatter(path.read_text(encoding="utf-8"))
            existing_authored_by = existing_fm.get("authored_by")
            if existing_authored_by in ("user", "user-edit"):
                logger.warning(
                    "Session note %s has authored_by=%r -- write refused to "
                    "preserve the user's edits (ADR-010 invariant 5)",
                    path,
                    existing_authored_by,
                )
                return
            if existing_authored_by is not None:
                try:
                    authored_by = AuthoredBy(existing_authored_by)
                except ValueError:
                    authored_by = AuthoredBy.MIST

        fm = MistSessionFrontmatter(
            session_id=canonical_session_id,
            title=synthesis.title if synthesis else canonical_session_id,
            date=note_date,
            status=status,
            authored_by=authored_by,
            related_entities=sorted(set(related_entities)),
            ontology_version=ONTOLOGY_VERSION,
            extraction_version=EXTRACTION_VERSION,
            model_hash=self._model_hash,
        )

        body_parts = [f"# Session: {fm.title}\n\n", f"**Date:** {fm.date}\n\n"]
        if synthesis is not None:
            body_parts.append(synthesis.body.rstrip() + "\n")

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_frontmatter(fm, "".join(body_parts)), encoding="utf-8")

    async def _handle_mark_authored_by(self, args: dict[str, Any]) -> None:
        path = Path(args["path"])
        # Self-mark inside the handler (not at enqueue time): the queue may
        # be deep enough that an enqueue-time marker expires before the
        # write executes, which would re-fire the user-edit sequence.
        self._mark_mist_write(path)
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, self._mark_authored_by_sync, path)
        except OSError as exc:
            raise VaultWriteError(f"Failed authored_by writeback on {path}: {exc}") from exc

    def _mark_authored_by_sync(self, path: Path) -> None:
        """Synchronous core: flip frontmatter authored_by to user-edit."""
        text = path.read_text(encoding="utf-8")
        new_text = re.sub(
            r"^authored_by:\s*\S+",
            "authored_by: user-edit",
            text,
            count=1,
            flags=re.MULTILINE,
        )
        if new_text == text:
            return
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(new_text, encoding="utf-8")
        tmp.replace(path)

    async def _handle_upsert_identity(self, args: dict[str, Any]) -> str:
        traits: list[dict] = args["traits"]
        capabilities: list[dict] = args["capabilities"]
        preferences: list[dict] = args["preferences"]
        rendered_at: str | None = args.get("rendered_at")

        path = self._root / "identity" / "mist.md"
        self._mark_mist_write(path)
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                self._upsert_identity_sync,
                path,
                traits,
                capabilities,
                preferences,
                rendered_at,
            )
        except OSError as exc:
            raise VaultWriteError(f"Failed to write identity note {path}: {exc}") from exc

        return str(path)

    def _upsert_identity_sync(
        self,
        path: Path,
        traits: list[dict],
        capabilities: list[dict],
        preferences: list[dict],
        rendered_at: str | None = None,
    ) -> None:
        """Synchronous core of `upsert_identity`.

        Preserves hand-edited files: when the existing file carries
        `authored_by: user-edit` (the value the invariant-5 writeback stamps
        on real human edits), the bootstrap does NOT overwrite it --
        identity/mist.md is the bucket-3 curated persona file whose user
        edits are the most authoritative content in the vault. Files still
        carrying the machine-stamped birth value are refreshed normally so
        the seed source's updates flow through. (No production caller
        remains as of R1.4 Task 10 -- see the sibling `upsert_identity_body`,
        which the seed bootstrap now uses -- but this method and its
        Protocol entry are retained rather than deleted; see that task's
        report for the reasoning.)

        `rendered_at`, when supplied, pins the frontmatter `last_updated` date
        and the Provenance timestamp for reproducible writes (None ->
        wall-clock).
        """
        now_iso = rendered_at if rendered_at is not None else datetime.now(UTC).isoformat()
        today = now_iso[:10]

        if path.exists():
            existing = path.read_text(encoding="utf-8")
            fm_dict, _body = parse_frontmatter(existing)
            if fm_dict.get("authored_by") == "user-edit":
                logger.warning(
                    "Identity note %s has authored_by=user-edit -- bootstrap "
                    "skipped to preserve hand edits (ADR-010 invariant 5)",
                    path,
                )
                return

        fm = MistIdentityFrontmatter(
            authored_by=AuthoredBy.USER,
            version="1.0",
            last_updated=today,
            tags=["identity", "traits", "preferences"],
        )

        # Sort for determinism
        sorted_traits = sorted(traits, key=lambda t: t.get("display_name", ""))
        sorted_caps = sorted(capabilities, key=lambda c: c.get("display_name", ""))
        sorted_prefs = sorted(preferences, key=lambda p: p.get("display_name", ""))

        lines = ["# MIST Identity\n", "\n## Traits\n"]
        for t in sorted_traits:
            display = t.get("display_name", "")
            axis = t.get("axis", "")
            desc = t.get("description", "")
            if axis:
                lines.append(f"- **{display}** ({axis}) -- {desc}\n")
            else:
                lines.append(f"- **{display}** -- {desc}\n")

        lines.append("\n## Capabilities\n")
        for c in sorted_caps:
            display = c.get("display_name", "")
            desc = c.get("description", "")
            lines.append(f"- **{display}** -- {desc}\n")

        lines.append("\n## Preferences\n")
        for p in sorted_prefs:
            display = p.get("display_name", "")
            enforcement = p.get("enforcement", "")
            context = p.get("context", p.get("description", ""))
            if enforcement:
                lines.append(f"- **{display}** ({enforcement}) -- {context}\n")
            else:
                lines.append(f"- **{display}** -- {context}\n")

        lines.append(
            f"\n## Provenance\n" f"- source: scripts/seed_data.yaml\n" f"- rendered_at: {now_iso}\n"
        )

        body = "".join(lines)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_frontmatter(fm, body), encoding="utf-8")

    async def _handle_upsert_identity_body(self, args: dict[str, Any]) -> str:
        body_markdown: str = args["body_markdown"]
        source_path: str = args["source_path"]
        rendered_at: str | None = args.get("rendered_at")

        path = self._root / "identity" / "mist.md"
        self._mark_mist_write(path)
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                self._upsert_identity_body_sync,
                path,
                body_markdown,
                source_path,
                rendered_at,
            )
        except OSError as exc:
            raise VaultWriteError(f"Failed to write identity note {path}: {exc}") from exc

        return str(path)

    def _upsert_identity_body_sync(
        self,
        path: Path,
        body_markdown: str,
        source_path: str,
        rendered_at: str | None = None,
    ) -> None:
        """Synchronous core of `upsert_identity_body`.

        Mirrors `_upsert_user_sync`'s `authored_by` guard and Provenance-dedup
        pattern -- the sibling `_upsert_identity_sync` renders from structured
        lists and has no caller-provided body to guard or dedupe against, so
        that logic is not shared code, only a shared pattern. `authored_by in
        {user, user-edit}` refuses the write per ADR-010 Invariant 5:
        identity/mist.md is bucket-3 curated persona content, and a hand-edit
        takes precedence over any re-seed.

        `rendered_at`, when supplied, pins the frontmatter `last_updated`
        date and the Provenance timestamp for reproducible writes (None ->
        wall-clock).
        """
        now_iso = rendered_at if rendered_at is not None else datetime.now(UTC).isoformat()
        today = now_iso[:10]

        if path.exists():
            existing = path.read_text(encoding="utf-8")
            fm_dict, existing_body = parse_frontmatter(existing)
            authored_by_val = fm_dict.get("authored_by")
            if authored_by_val in ("user", "user-edit"):
                logger.warning(
                    "Identity note %s has authored_by=%r -- body not overwritten "
                    "(ADR-010 invariant 5)",
                    path,
                    authored_by_val,
                )
                fm_dict["last_updated"] = today
                import yaml as _yaml

                new_yaml = _yaml.safe_dump(
                    fm_dict, sort_keys=False, default_flow_style=False, allow_unicode=True
                )
                new_content = f"---\n{new_yaml}---\n\n{existing_body}"
                path.write_text(new_content, encoding="utf-8")
                return

        fm = MistIdentityFrontmatter(
            authored_by=AuthoredBy.MIST,
            version="1.0",
            last_updated=today,
            tags=["identity", "traits", "preferences"],
        )

        if _PROVENANCE_HEADING_RE.search(body_markdown):
            full_body = body_markdown.rstrip("\n") + "\n"
        else:
            provenance_section = (
                f"\n## Provenance\n- source: {source_path}\n- rendered_at: {now_iso}\n"
            )
            full_body = body_markdown.rstrip("\n") + "\n" + provenance_section

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_frontmatter(fm, full_body), encoding="utf-8")

    async def _handle_upsert_user(self, args: dict[str, Any]) -> str:
        user_id: str = args["user_id"]
        body_markdown: str = args["body_markdown"]
        rendered_at: str | None = args.get("rendered_at")

        path = self._root / "users" / f"{user_id}.md"
        self._mark_mist_write(path)
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                self._upsert_user_sync,
                path,
                user_id,
                body_markdown,
                rendered_at,
            )
        except OSError as exc:
            raise VaultWriteError(f"Failed to write user note {path}: {exc}") from exc

        return str(path)

    def _upsert_user_sync(
        self, path: Path, user_id: str, body_markdown: str, rendered_at: str | None = None
    ) -> None:
        """Synchronous core of `upsert_user`.

        `rendered_at`, when supplied, pins the frontmatter `last_updated` date
        and the appended Provenance timestamp so the written file is
        reproducible (seed bootstrap under replay). None -> wall-clock.
        """
        now_iso = rendered_at if rendered_at is not None else datetime.now(UTC).isoformat()
        today = now_iso[:10]

        if path.exists():
            content = path.read_text(encoding="utf-8")
            fm_dict, existing_body = parse_frontmatter(content)
            authored_by_val = fm_dict.get("authored_by", "mist")

            if authored_by_val in ("user", "user-edit"):
                logger.warning(
                    "User note %s has authored_by=%r -- body not overwritten",
                    path,
                    authored_by_val,
                )
                # Update only last_updated in frontmatter; preserve body.
                # The Provenance-dedup logic at the bottom of this method
                # (`_PROVENANCE_HEADING_RE` check) is intentionally NOT
                # applied here: ADR-010 invariant 5 mandates that user
                # edits are authoritative on conflict. If the user has
                # taken ownership of the file, MIST does not rewrite or
                # de-duplicate the body's Provenance section even when it
                # was originally written by MIST. Side effect: a note
                # that accumulated duplicate Provenance sections BEFORE
                # the dedup fix landed will keep them after a user edit
                # locked the body. A future workstream addressing
                # invariant-5 vault re-derivation can revisit; for now
                # the asymmetry is documented and accepted.
                fm_dict["last_updated"] = today
                import yaml as _yaml

                new_yaml = _yaml.safe_dump(
                    fm_dict, sort_keys=False, default_flow_style=False, allow_unicode=True
                )
                new_content = f"---\n{new_yaml}---\n\n{existing_body}"
                path.write_text(new_content, encoding="utf-8")
                return

        fm = MistUserFrontmatter(
            user_id=user_id,
            authored_by=AuthoredBy.MIST,
            last_updated=today,
        )
        # Append a default `## Provenance` section ONLY when body_markdown
        # does not already include one. The C-pattern user-snapshot
        # renderer (backend/vault/user_snapshot.render_user_snapshot_body)
        # supplies a richer Provenance with source attribution; trust the
        # caller-provided section when present, fall back to a writer-
        # supplied minimal section otherwise. Without this guard, mid-V6
        # re-renders accumulated duplicate Provenance sections during
        # continuous use.
        #
        # Detection uses a multiline anchored regex (case-insensitive) so
        # quoted lines like `> ## Provenance` and lowercase variants like
        # `## provenance` are correctly handled. A raw substring check
        # would mismatch the quoted text and incorrectly suppress the
        # writer's section. Note: a `## Provenance` heading inside a
        # fenced code block would still match (regex is markdown-naive);
        # body_markdown produced by user_snapshot.render_user_snapshot_body
        # never contains code fences, so this trade-off is acceptable in
        # practice.
        if _PROVENANCE_HEADING_RE.search(body_markdown):
            full_body = body_markdown.rstrip("\n") + "\n"
        else:
            provenance_section = f"\n## Provenance\n- rendered_at: {now_iso}\n"
            full_body = body_markdown.rstrip("\n") + "\n" + provenance_section
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_frontmatter(fm, full_body), encoding="utf-8")

    async def _handle_upsert_user_snapshot(self, args: dict[str, Any]) -> str:
        user_id: str = args["user_id"]
        body_markdown: str = args["body_markdown"]
        rendered_at: str | None = args.get("rendered_at")

        # Filename STEM is decoupled from the user_id frontmatter field: the
        # derived snapshot lives at users/<user_id>-graph-snapshot.md so it
        # never collides with the curated users/<user_id>.md.
        path = self._root / "users" / f"{user_id}-graph-snapshot.md"
        self._mark_mist_write(path)
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                self._upsert_user_snapshot_sync,
                path,
                user_id,
                body_markdown,
                rendered_at,
            )
        except OSError as exc:
            raise VaultWriteError(f"Failed to write user snapshot {path}: {exc}") from exc

        return str(path)

    def _upsert_user_snapshot_sync(
        self, path: Path, user_id: str, body_markdown: str, rendered_at: str | None = None
    ) -> None:
        """Synchronous core of `upsert_user_snapshot`.

        Always overwrites the body. Unlike `_upsert_user_sync`, there is NO
        Invariant-5 `authored_by` user/user-edit guard: the snapshot is a
        machine-owned derived cache, regenerated on every user-scope extraction,
        and is never the user-authoritative profile. Authorship is always
        `mist`. Provenance dedup is preserved (a caller-supplied `## Provenance`
        section is trusted; otherwise a minimal writer section is appended) so
        repeated re-renders do not accumulate duplicate Provenance sections.

        `rendered_at`, when supplied, pins the frontmatter `last_updated` date
        and the fallback Provenance timestamp for reproducible writes (None ->
        wall-clock).
        """
        now_iso = rendered_at if rendered_at is not None else datetime.now(UTC).isoformat()
        today = now_iso[:10]

        fm = MistUserFrontmatter(
            user_id=user_id,
            authored_by=AuthoredBy.MIST,
            last_updated=today,
        )
        if _PROVENANCE_HEADING_RE.search(body_markdown):
            full_body = body_markdown.rstrip("\n") + "\n"
        else:
            provenance_section = f"\n## Provenance\n- rendered_at: {now_iso}\n"
            full_body = body_markdown.rstrip("\n") + "\n" + provenance_section
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_frontmatter(fm, full_body), encoding="utf-8")

    # ------------------------------------------------------------------
    # Directory and git setup
    # ------------------------------------------------------------------

    async def _ensure_directories(self) -> None:
        """Create vault subdirectory structure. Idempotent."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._ensure_directories_sync)

    def _ensure_directories_sync(self) -> None:
        for subdir in ("sessions", "identity", "users", "decisions", "meta"):
            (self._root / subdir).mkdir(parents=True, exist_ok=True)

    async def _maybe_git_init(self) -> None:
        """Run `git init` if no `.git` directory exists under `config.root`."""
        git_dir = self._root / ".git"
        if git_dir.exists():
            return

        self._root.mkdir(parents=True, exist_ok=True)
        proc = await asyncio.create_subprocess_exec(
            "git",
            "init",
            str(self._root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.warning("git init failed in %s: %s", self._root, stderr.decode(errors="replace"))
            return

        logger.info("git init completed for vault at %s", self._root)

        # Create initial commit so the repo has a valid HEAD
        proc = await asyncio.create_subprocess_exec(
            "git",
            "-C",
            str(self._root),
            "commit",
            "--allow-empty",
            "-m",
            "chore(vault): initialize MIST memory vault",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.warning(
                "git initial commit failed in %s: %s",
                self._root,
                stderr.decode(errors="replace"),
            )
