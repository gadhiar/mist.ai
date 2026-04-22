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
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.errors import VaultWriteError
from backend.knowledge.config import VaultConfig
from backend.vault.models import (
    AuthoredBy,
    MistIdentityFrontmatter,
    MistSessionFrontmatter,
    MistUserFrontmatter,
    parse_frontmatter,
    render_frontmatter,
)

logger = logging.getLogger(__name__)

_SENTINEL = "<!-- MIST_APPEND_HERE -->"
_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_ONTOLOGY_VERSION = "1.0.0"
_EXTRACTION_VERSION = "2026-04-17-r1"

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
        path = await writer.append_turn_to_session("sess-001", 1, "hi", "hello")
        await writer.stop()
    """

    def __init__(self, config: VaultConfig, debug_logger: Any = None) -> None:
        """Initialize the vault writer.

        Args:
            config: VaultConfig with root path + lifecycle flags.
            debug_logger: Optional DebugJSONLLogger (Cluster 8 Phase 12). When
                set + `MIST_DEBUG_VAULT_JSONL=1`, every consumer-side write
                op emits a `phase: "vault"` JSONL record with operation,
                path, duration_ms, ok, and any error_message. None preserves
                pre-Phase-12 silent operation.
        """
        self.config = config
        self._root = Path(config.root)
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._consumer_task: asyncio.Task | None = None  # type: ignore[type-arg]
        self._stopped = False
        self._debug_logger = debug_logger

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

    async def append_turn_to_session(
        self,
        session_id: str,
        turn_index: int,
        user_text: str,
        mist_text: str,
        vault_note_path: str | None = None,
    ) -> str:
        """Append a conversation turn to the active session note.

        If `vault_note_path` is None, derives the path via `session_path`
        using today's date and `session_id` as the slug.

        Creates the note file with frontmatter and the append sentinel on
        first call. Inserts the turn block above the sentinel on subsequent
        calls. Updates `turn_count` and `append_sentinel_offset` in
        frontmatter after each append.

        Args:
            session_id: Slug identifying the session.
            turn_index: 1-based turn number (used in the heading).
            user_text: Raw user utterance.
            mist_text: Raw MIST response.
            vault_note_path: Absolute path override. Derived when None.

        Returns:
            Absolute path to the session note file.

        Raises:
            VaultWriteError: If the file write fails irrecoverably.
        """
        return await self._enqueue(
            "append_turn",
            {
                "session_id": session_id,
                "turn_index": turn_index,
                "user_text": user_text,
                "mist_text": mist_text,
                "vault_note_path": vault_note_path,
            },
        )

    async def update_entities_extracted(
        self,
        vault_note_path: str,
        turn_index: int,
        entity_slugs: list[str],
    ) -> None:
        """Replace the `_[pending]_` placeholder with extracted entity wikilinks.

        Locates the `## Turn <N>` block in the file and updates the
        `**Entities extracted:**` line. Adds slugs to frontmatter
        `related_entities` (deduped). Idempotent.

        Args:
            vault_note_path: Absolute path to the session note.
            turn_index: 1-based turn number identifying the block.
            entity_slugs: Entity slugs to link (without `[[` brackets`]]).

        Raises:
            VaultWriteError: If the file cannot be read or written.
        """
        await self._enqueue(
            "update_entities",
            {
                "vault_note_path": vault_note_path,
                "turn_index": turn_index,
                "entity_slugs": entity_slugs,
            },
        )

    async def upsert_identity(
        self,
        traits: list[dict],
        capabilities: list[dict],
        preferences: list[dict],
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
            },
        )

    async def upsert_user(self, user_id: str, body_markdown: str) -> str:
        """Write or update a user fact sheet at `users/<user_id>.md`.

        On existing files where `authored_by` is `user` or `user-edit`,
        the body is NOT overwritten (user's edits are authoritative per
        ADR-010 Invariant 5). Only `last_updated` and `related_sessions`
        are updated in that case.

        Args:
            user_id: User identifier (used as filename stem).
            body_markdown: Caller-provided markdown body. A `## Provenance`
                section is appended automatically.

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
        """
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
        `job_args` so the record carries useful context (turn_index for
        appends, entity_count for backfills, user_id for upsert_user)
        without leaking large payloads (utterance bodies, full markdown).
        """
        if self._debug_logger is None:
            return
        try:
            extra: dict[str, Any] = {}
            if operation == "append_turn":
                extra["turn_index"] = job_args.get("turn_index")
                extra["session_id"] = job_args.get("session_id")
            elif operation == "update_entities":
                extra["turn_index"] = job_args.get("turn_index")
                slugs = job_args.get("entity_slugs") or []
                extra["entity_count"] = len(slugs)
            elif operation == "upsert_user":
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
            "append_turn": self._handle_append_turn,
            "update_entities": self._handle_update_entities,
            "upsert_identity": self._handle_upsert_identity,
            "upsert_user": self._handle_upsert_user,
        }
        handler = handlers.get(job.kind)
        if handler is None:
            raise VaultWriteError(f"Unknown job kind: {job.kind!r}")
        return await handler(job.args)

    # ------------------------------------------------------------------
    # Handlers (run inside consumer; may do blocking I/O via executor)
    # ------------------------------------------------------------------

    async def _handle_append_turn(self, args: dict[str, Any]) -> str:
        session_id: str = args["session_id"]
        turn_index: int = args["turn_index"]
        user_text: str = args["user_text"]
        mist_text: str = args["mist_text"]
        vault_note_path: str | None = args["vault_note_path"]

        if vault_note_path is None:
            today = datetime.now(UTC).date().isoformat()
            vault_note_path = self.session_path(today, session_id)

        path = Path(vault_note_path)

        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                self._append_turn_sync,
                path,
                session_id,
                turn_index,
                user_text,
                mist_text,
            )
        except OSError as exc:
            raise VaultWriteError(f"Failed to write session note {path}: {exc}") from exc

        return str(path)

    def _append_turn_sync(
        self,
        path: Path,
        session_id: str,
        turn_index: int,
        user_text: str,
        mist_text: str,
    ) -> None:
        """Synchronous core of `append_turn_to_session`.

        Creates the file if missing, then inserts the turn block above the
        sentinel and updates frontmatter counters.
        """
        today = datetime.now(UTC).date().isoformat()

        if not path.exists():
            fm = MistSessionFrontmatter(
                session_id=session_id,
                date=today,
                turn_count=0,
                ontology_version=_ONTOLOGY_VERSION,
                extraction_version=_EXTRACTION_VERSION,
            )
            initial_body = _SENTINEL + "\n"
            content = render_frontmatter(fm, initial_body)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

        content = path.read_text(encoding="utf-8")
        fm_dict, body = parse_frontmatter(content)

        # YAML may parse unquoted date values (e.g. 2026-04-21) as datetime.date
        # objects. Coerce back to ISO string before constructing the model.
        raw_date = fm_dict.get("date", today)
        if hasattr(raw_date, "isoformat"):
            raw_date = raw_date.isoformat()

        # Rebuild frontmatter model from parsed dict (tolerates missing optional fields)
        fm = MistSessionFrontmatter(
            session_id=fm_dict.get("session_id", session_id),
            date=raw_date,
            turn_count=fm_dict.get("turn_count", 0),
            participants=fm_dict.get("participants", ["user", "mist"]),
            authored_by=fm_dict.get("authored_by", AuthoredBy.MIST),
            status=fm_dict.get("status", "in-progress"),
            append_sentinel_offset=fm_dict.get("append_sentinel_offset"),
            related_entities=fm_dict.get("related_entities", []),
            ontology_version=fm_dict.get("ontology_version", _ONTOLOGY_VERSION),
            extraction_version=fm_dict.get("extraction_version", _EXTRACTION_VERSION),
            model_hash=fm_dict.get("model_hash"),
            tags=fm_dict.get("tags", []),
        )

        turn_block = (
            f"## Turn {turn_index}\n\n"
            f"**User:** {user_text}\n\n"
            f"**MIST:** {mist_text}\n\n"
            f"**Entities extracted:** _[pending]_\n\n"
        )

        sentinel_idx = body.find(_SENTINEL)
        if sentinel_idx == -1:
            logger.warning("Append sentinel missing in %s -- falling back to EOF append", path)
            body = body.rstrip("\n") + "\n\n" + turn_block + _SENTINEL + "\n"
        else:
            body = body[:sentinel_idx] + turn_block + _SENTINEL + "\n"

        fm = fm.model_copy(update={"turn_count": fm.turn_count + 1})

        # Compute byte offset of sentinel in final file for frontmatter field
        rendered_header = "---\n" + _yaml_dump_for_offset(fm) + "---\n\n"
        sentinel_byte_offset = len(rendered_header.encode("utf-8")) + len(
            body[: body.rfind(_SENTINEL)].encode("utf-8")
        )
        fm = fm.model_copy(update={"append_sentinel_offset": sentinel_byte_offset})

        new_content = render_frontmatter(fm, body)
        path.write_text(new_content, encoding="utf-8")

    async def _handle_update_entities(self, args: dict[str, Any]) -> None:
        vault_note_path: str = args["vault_note_path"]
        turn_index: int = args["turn_index"]
        entity_slugs: list[str] = args["entity_slugs"]

        path = Path(vault_note_path)
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                self._update_entities_sync,
                path,
                turn_index,
                entity_slugs,
            )
        except OSError as exc:
            raise VaultWriteError(f"Failed to update entities in {path}: {exc}") from exc

    def _update_entities_sync(
        self,
        path: Path,
        turn_index: int,
        entity_slugs: list[str],
    ) -> None:
        """Synchronous core of `update_entities_extracted`."""
        if not path.exists():
            raise VaultWriteError(f"Session note not found: {path}")

        content = path.read_text(encoding="utf-8")
        fm_dict, body = parse_frontmatter(content)

        turn_heading = f"## Turn {turn_index}"
        turn_start = body.find(turn_heading)
        if turn_start == -1:
            raise VaultWriteError(f"Turn {turn_index} block not found in {path}")

        # Find the next turn heading or the sentinel to bound the block
        next_turn_start = body.find("## Turn ", turn_start + len(turn_heading))
        sentinel_pos = body.find(_SENTINEL, turn_start)
        block_end = len(body)
        if next_turn_start != -1:
            block_end = next_turn_start
        if sentinel_pos != -1 and sentinel_pos < block_end:
            block_end = sentinel_pos

        block = body[turn_start:block_end]
        pending_marker = "**Entities extracted:** _[pending]_"

        if entity_slugs:
            wikilinks = ", ".join(f"[[{s}]]" for s in sorted(set(entity_slugs)))
            entity_line = f"**Entities extracted:** {wikilinks}"
        else:
            entity_line = "**Entities extracted:** _(none)_"

        # Idempotency: if already set to the same value, skip write
        if pending_marker not in block and entity_line in block:
            return

        updated_block = block.replace(pending_marker, entity_line, 1)
        # If pending marker was already replaced with a different value, replace that too
        if entity_line not in updated_block and pending_marker not in updated_block:
            # Replace existing entities line pattern
            updated_block = re.sub(
                r"\*\*Entities extracted:\*\* .*",
                entity_line,
                updated_block,
            )

        new_body = body[:turn_start] + updated_block + body[block_end:]

        # Update frontmatter related_entities (deduped, sorted)
        existing_entities: list[str] = fm_dict.get("related_entities", [])
        new_slugs_wikilinks = [f"[[{s}]]" for s in entity_slugs]
        merged = sorted(set(existing_entities) | set(new_slugs_wikilinks))
        fm_dict["related_entities"] = merged

        # Reconstruct frontmatter from dict (preserve all existing fields)
        import yaml as _yaml

        new_yaml = _yaml.safe_dump(
            fm_dict, sort_keys=False, default_flow_style=False, allow_unicode=True
        )
        new_content = f"---\n{new_yaml}---\n\n{new_body}"
        path.write_text(new_content, encoding="utf-8")

    async def _handle_upsert_identity(self, args: dict[str, Any]) -> str:
        traits: list[dict] = args["traits"]
        capabilities: list[dict] = args["capabilities"]
        preferences: list[dict] = args["preferences"]

        path = self._root / "identity" / "mist.md"
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                self._upsert_identity_sync,
                path,
                traits,
                capabilities,
                preferences,
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
    ) -> None:
        """Synchronous core of `upsert_identity`."""
        today = datetime.now(UTC).date().isoformat()
        now_iso = datetime.now(UTC).isoformat()

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

    async def _handle_upsert_user(self, args: dict[str, Any]) -> str:
        user_id: str = args["user_id"]
        body_markdown: str = args["body_markdown"]

        path = self._root / "users" / f"{user_id}.md"
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                self._upsert_user_sync,
                path,
                user_id,
                body_markdown,
            )
        except OSError as exc:
            raise VaultWriteError(f"Failed to write user note {path}: {exc}") from exc

        return str(path)

    def _upsert_user_sync(self, path: Path, user_id: str, body_markdown: str) -> None:
        """Synchronous core of `upsert_user`."""
        today = datetime.now(UTC).date().isoformat()
        now_iso = datetime.now(UTC).isoformat()
        provenance_section = f"\n## Provenance\n" f"- rendered_at: {now_iso}\n"

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
                # Update only last_updated in frontmatter; preserve body
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


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _yaml_dump_for_offset(model: MistSessionFrontmatter) -> str:
    """Render the YAML block used for byte-offset calculation.

    Matches the exact output of `render_frontmatter` so that the
    `append_sentinel_offset` byte position is correct.
    """
    import yaml

    data = model.model_dump(mode="json")
    return yaml.safe_dump(data, sort_keys=False, default_flow_style=False, allow_unicode=True)
