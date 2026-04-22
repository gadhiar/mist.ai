"""Vault filewatcher for live sidecar reindex on vault file changes (ADR-010 Cluster 8).

Runs a watchdog observer in a dedicated daemon thread inside the backend process.
Bridges filesystem events into the asyncio event loop via
`loop.call_soon_threadsafe`. Debounces event bursts so that Obsidian's
atomic-replace-on-save (delete + create within ~100ms) collapses to a single
reindex call. A periodic mtime audit job catches dropped events (Windows
ReadDirectoryChangesW overflow).

Lifecycle note: `atexit` registration is the caller's responsibility.
The factory (Phase 5) calls `atexit.register(filewatcher.stop)`. This class
only provides `start()` and `stop()`.

Under `uvicorn --reload`, the reload hook must call `stop()` before
re-instantiating. See `backend/factories.py` for wiring.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import platform
import time
from pathlib import Path

from watchdog.events import (
    FileMovedEvent,
    FileSystemEvent,
    FileSystemEventHandler,
)

from backend.errors import FilewatcherError, SidecarIndexError
from backend.interfaces import SidecarIndexProtocol
from backend.knowledge.config import FilewatcherConfig
from backend.vault.models import parse_frontmatter

logger = logging.getLogger(__name__)


def _is_tracked_path(path: str) -> bool:
    """Return True if the path should be tracked by the filewatcher.

    Filters to `.md` files only, excluding hidden files (basename starts with
    `.`) and anything inside a `.git` subdirectory.

    Args:
        path: Absolute or relative filesystem path string.

    Returns:
        True when the path targets a trackable vault markdown file.
    """
    p = Path(path)
    # Must be a .md file
    if p.suffix.lower() != ".md":
        return False
    # Exclude hidden files
    if p.name.startswith("."):
        return False
    # Exclude .git subtree (any part of the path is ".git")
    return ".git" not in p.parts


# ---------------------------------------------------------------------------
# Watchdog event handler
# ---------------------------------------------------------------------------


class _DebouncingEventHandler(FileSystemEventHandler):
    """Watchdog handler that marshals events into the asyncio loop.

    All public `on_*` methods are called from the watchdog observer thread.
    They immediately hand off to the loop via `loop.call_soon_threadsafe` so
    that no asyncio primitives are touched from the wrong thread.

    The actual debounce scheduling and sidecar calls happen on
    `_VaultFilewatcher._on_event_main_thread`, which runs in the event loop.
    """

    def __init__(self, watcher: VaultFilewatcher) -> None:
        super().__init__()
        self._watcher = watcher

    # ------------------------------------------------------------------
    # FileSystemEventHandler overrides
    # ------------------------------------------------------------------

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory or not _is_tracked_path(event.src_path):
            return
        self._watcher._marshal_event("modified", event.src_path, None)

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory or not _is_tracked_path(event.src_path):
            return
        self._watcher._marshal_event("created", event.src_path, None)

    def on_deleted(self, event: FileSystemEvent) -> None:
        if event.is_directory or not _is_tracked_path(event.src_path):
            return
        self._watcher._marshal_event("deleted", event.src_path, None)

    def on_moved(self, event: FileMovedEvent) -> None:
        if event.is_directory:
            return
        # Only handle when both paths are tracked (i.e. rename within vault)
        src_tracked = _is_tracked_path(event.src_path)
        dest_tracked = _is_tracked_path(event.dest_path)
        if src_tracked:
            self._watcher._marshal_event("deleted", event.src_path, None)
        if dest_tracked:
            self._watcher._marshal_event("created", event.dest_path, None)


# ---------------------------------------------------------------------------
# VaultFilewatcher
# ---------------------------------------------------------------------------


class VaultFilewatcher:
    """Watches a vault directory for markdown file changes and triggers sidecar reindex.

    Runs a watchdog observer in a dedicated daemon thread. Bridges events into
    the asyncio event loop via `loop.call_soon_threadsafe`. Debounces bursts
    per path. An optional mtime audit job catches dropped OS events.

    Usage::

        fw = VaultFilewatcher(config, vault_root, sidecar_index)
        fw.start()                  # Start observer + audit job
        fw.mark_mist_write(path)    # Called by VaultWriter before a write
        fw.clear_mist_write(path)   # Called by VaultWriter after a write
        fw.stop()                   # Stops observer + cancels timers

    The caller is responsible for registering `atexit.register(fw.stop)`.
    """

    def __init__(
        self,
        config: FilewatcherConfig,
        vault_root: str | Path,
        sidecar_index: SidecarIndexProtocol,
    ) -> None:
        self.config = config
        self.vault_root = Path(vault_root)
        self._sidecar = sidecar_index

        self._observer = None  # Set at start()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._handler: _DebouncingEventHandler | None = None

        # path -> (TimerHandle, action) where action is "reindex" or "delete".
        # Tracks pending debounced actions per path. Only one pending action
        # per path at a time; scheduling a new action cancels the old one.
        # The Obsidian atomic-save collapse (delete + create within debounce
        # window) works because the create's "reindex" replaces the delete's
        # pending "delete", so no delete_path call is issued to the sidecar.
        self._pending: dict[str, tuple[asyncio.TimerHandle, str]] = {}

        # path -> last-known mtime (int) for audit job
        self._known_mtimes: dict[str, int] = {}

        # path -> expiry_ts (float) for MIST-in-flight write tracking
        self._mist_writes_in_flight: dict[str, float] = {}
        self._mist_write_ttl_seconds: float = 2.0

        # asyncio.TimerHandle for the periodic audit job
        self._audit_handle: asyncio.TimerHandle | None = None

        self._running: bool = False

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    def start(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Start the filewatcher observer and audit job.

        Idempotent: if already running, this is a no-op.

        Args:
            loop: Asyncio event loop to use for timer scheduling and thread
                bridging. If None, `asyncio.get_running_loop()` is used,
                which requires this method to be called from a coroutine or
                a thread with a running loop.

        Raises:
            FilewatcherError: If the requested observer backend cannot be
                imported (e.g. `inotify` requested on macOS) or fails to
                start.
        """
        if self._running:
            return

        # Resolve the event loop
        if loop is not None:
            self._loop = loop
        else:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError as exc:
                raise FilewatcherError(
                    "No running asyncio loop found. Pass loop= explicitly or call start() "
                    "from within a coroutine."
                ) from exc

        # Resolve and instantiate the observer
        observer = self._create_observer()

        # Register the handler
        self._handler = _DebouncingEventHandler(self)
        observer.schedule(self._handler, str(self.vault_root), recursive=True)

        try:
            observer.start()
        except OSError as exc:
            raise FilewatcherError(
                f"Observer failed to start on '{self.vault_root}': {exc}"
            ) from exc

        self._observer = observer
        self._running = True

        # Populate initial mtime snapshot (no reindex on startup)
        self._scan_vault_mtimes()

        # Schedule the first audit job
        self._schedule_audit()

        logger.debug(
            "VaultFilewatcher started; observer=%s vault_root=%s debounce_ms=%d",
            type(observer).__name__,
            self.vault_root,
            self.config.debounce_ms,
        )

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the filewatcher observer and cancel all pending timers.

        Idempotent: safe to call multiple times or before `start()`.

        Args:
            timeout: Seconds to wait for the observer thread to join before
                logging a warning and continuing.
        """
        if not self._running:
            return

        self._running = False

        # Cancel audit timer
        if self._audit_handle is not None:
            self._audit_handle.cancel()
            self._audit_handle = None

        # Cancel all pending debounce timers
        for handle, _action in self._pending.values():
            handle.cancel()
        self._pending.clear()

        # Stop the observer thread
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=timeout)
            if self._observer.is_alive():
                logger.warning(
                    "VaultFilewatcher: observer thread did not stop within %.1fs timeout",
                    timeout,
                )
            self._observer = None

        logger.debug("VaultFilewatcher stopped")

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """True if the filewatcher observer is active."""
        return self._running

    @property
    def pending_paths(self) -> set[str]:
        """Set of paths that have a pending debounced action (reindex or delete) scheduled."""
        return set(self._pending.keys())

    # ------------------------------------------------------------------
    # MIST write coordination API
    # ------------------------------------------------------------------

    def mark_mist_write(self, path: str) -> None:
        """Record that MIST is about to write to `path`.

        Filesystem events arriving for this path within `_mist_write_ttl_seconds`
        of this call are attributed to MIST (not user-origin). Still triggers
        sidecar reindex; the distinction is only in log level and future
        `authored_by` attribution (Phase 6).

        Args:
            path: Absolute or relative vault file path.
        """
        expiry = time.monotonic() + self._mist_write_ttl_seconds
        self._mist_writes_in_flight[path] = expiry

    def clear_mist_write(self, path: str) -> None:
        """Remove a MIST-write marker early (VaultWriter calls this on completion).

        Args:
            path: Absolute or relative vault file path.
        """
        self._mist_writes_in_flight.pop(path, None)

    # ------------------------------------------------------------------
    # Thread -> loop bridge
    # ------------------------------------------------------------------

    def _marshal_event(self, kind: str, src_path: str, dest_path: str | None) -> None:
        """Called from the watchdog observer thread to marshal an event into the loop.

        Args:
            kind: One of "modified", "created", "deleted".
            src_path: Source path of the event.
            dest_path: Destination path (only for "moved" kind, unused here since
                moved events are decomposed into deleted + created before dispatch).
        """
        if self._loop is None:
            return
        self._loop.call_soon_threadsafe(self._on_event_main_thread, kind, src_path)

    def _on_event_main_thread(self, kind: str, path: str) -> None:
        """Process a filesystem event on the main event loop thread.

        Determines MIST vs user origin, updates log level accordingly, then
        schedules the appropriate action (debounced reindex or immediate delete).

        Args:
            kind: One of "modified", "created", "deleted".
            path: Affected vault file path.
        """
        if not self._running:
            return

        # Determine origin: MIST-write or user-edit
        now = time.monotonic()
        is_mist_write = (
            path in self._mist_writes_in_flight and self._mist_writes_in_flight[path] > now
        )

        if is_mist_write:
            logger.debug("VaultFilewatcher: MIST-origin %s event on %s", kind, path)
        else:
            logger.info("VaultFilewatcher: User edit detected at %s (event=%s)", path, kind)

        if kind == "deleted":
            # Deleted: schedule a debounced delete. If a create arrives within
            # the debounce window (Obsidian atomic-save), the create will
            # replace this pending delete with a reindex, collapsing the
            # delete+create burst into a single reindex (no delete_path call).
            self._schedule_action(path, "delete")
        else:
            # modified or created: debounced reindex. Cancels any pending
            # "delete" action for the same path (Obsidian atomic-save case).
            self._schedule_action(path, "reindex")

    # ------------------------------------------------------------------
    # Debounce scheduling
    # ------------------------------------------------------------------

    def _schedule_action(self, path: str, action: str) -> None:
        """Schedule (or reschedule) a debounced action for `path`.

        Cancels any existing pending action for this path (regardless of
        action type) and schedules a new timer. This is the core mechanism
        for the Obsidian atomic-save collapse:

        - `on_deleted` schedules action="delete".
        - `on_created` for the same path (within the debounce window)
          cancels the pending "delete" and schedules action="reindex".
        - Result: no delete_path call; only one upsert_file call.

        Args:
            path: Vault markdown file path.
            action: Either "reindex" or "delete".
        """
        self._cancel_pending(path)
        delay = self.config.debounce_ms / 1000.0
        if self._loop is not None:
            handle = self._loop.call_later(delay, self._fire_action, path, action)
            self._pending[path] = (handle, action)

    def _cancel_pending(self, path: str) -> None:
        """Cancel any pending debounce timer for `path` without firing it.

        Args:
            path: Vault markdown file path.
        """
        entry = self._pending.pop(path, None)
        if entry is not None:
            handle, _action = entry
            handle.cancel()

    def _fire_action(self, path: str, action: str) -> None:
        """Called when the debounce timer fires for `path`.

        Removes the path from the pending dict and dispatches the appropriate
        async coroutine (reindex or delete) to the event loop.

        Args:
            path: Vault markdown file path.
            action: Either "reindex" (call upsert_file) or "delete" (call delete_path).
        """
        self._pending.pop(path, None)
        if self._loop is None:
            return
        if action == "reindex":
            self._loop.create_task(self._do_reindex(path))
        elif action == "delete":
            self._loop.create_task(self._do_delete(path))

    # ------------------------------------------------------------------
    # Reindex and delete coroutines
    # ------------------------------------------------------------------

    async def _do_reindex(self, path: str) -> None:
        """Read and reindex a vault file in the sidecar index.

        Handles the race-condition case where the file is deleted between the
        event and the reindex read. Any SidecarIndexError is logged and
        swallowed so the watcher stays alive.

        Args:
            path: Absolute path to the vault markdown file.
        """
        try:
            content = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning(
                "VaultFilewatcher: file deleted between event and reindex, skipping %s (%s)",
                path,
                exc,
            )
            return
        except UnicodeDecodeError as exc:
            logger.warning("VaultFilewatcher: UTF-8 decode error on %s, skipping (%s)", path, exc)
            return

        try:
            frontmatter, _body = parse_frontmatter(content)
        except (
            Exception
        ) as exc:  # noqa: BLE001 -- parse_frontmatter should not raise but guard anyway
            logger.warning(
                "VaultFilewatcher: frontmatter parse failed on %s, using empty dict (%s)",
                path,
                exc,
            )
            frontmatter = {}

        try:
            mtime = int(Path(path).stat().st_mtime)
        except OSError as exc:
            logger.warning(
                "VaultFilewatcher: stat failed for %s after read, skipping (%s)", path, exc
            )
            return

        try:
            self._sidecar.upsert_file(path, content, mtime, frontmatter)
        except SidecarIndexError as exc:
            logger.error("VaultFilewatcher: sidecar upsert_file failed for %s: %s", path, exc)
            return
        except Exception as exc:  # noqa: BLE001 -- unexpected errors must not kill the watcher
            logger.exception(
                "VaultFilewatcher: unexpected error during upsert_file for %s: %s",
                path,
                exc,
            )
            return

        self._known_mtimes[path] = mtime

    async def _do_delete(self, path: str) -> None:
        """Remove a deleted vault file from the sidecar index.

        Args:
            path: Absolute path to the deleted vault markdown file.
        """
        try:
            self._sidecar.delete_path(path)
        except SidecarIndexError as exc:
            logger.error("VaultFilewatcher: sidecar delete_path failed for %s: %s", path, exc)
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "VaultFilewatcher: unexpected error during delete_path for %s: %s",
                path,
                exc,
            )
            return

        self._known_mtimes.pop(path, None)

    # ------------------------------------------------------------------
    # mtime audit job
    # ------------------------------------------------------------------

    def _schedule_audit(self) -> None:
        """Schedule the next mtime audit run.

        Uses `loop.call_later` so the audit fires as a sync callback that
        then creates the async coroutine task.
        """
        if not self._running or self._loop is None:
            return
        delay = float(self.config.audit_interval_seconds)
        self._audit_handle = self._loop.call_later(delay, self._fire_audit)

    def _fire_audit(self) -> None:
        """Sync callback invoked when the audit timer fires.

        Schedules the async audit coroutine and reschedules the next audit.
        """
        if not self._running or self._loop is None:
            return
        self._loop.create_task(self._run_audit())
        # Reschedule for the next interval
        self._schedule_audit()

    async def _run_audit(self) -> None:
        """Walk the vault and catch files with stale or missing sidecar entries.

        Compares disk mtime against `_known_mtimes`. Files that are newer
        trigger a debounced reindex. Files that have vanished from disk but
        are still in `_known_mtimes` trigger `delete_path`. Also cleans up
        stale MIST-write-in-flight markers.
        """
        self._cleanup_stale_mist_writes()

        try:
            disk_paths: set[str] = set()
            for dirpath, _dirnames, filenames in os.walk(self.vault_root):
                for fname in filenames:
                    full = str(Path(dirpath) / fname)
                    if not _is_tracked_path(full):
                        continue
                    disk_paths.add(full)
                    try:
                        stat_mtime = int(Path(full).stat().st_mtime)
                    except OSError:
                        # Broken symlink or race-deleted file -- skip
                        continue
                    known = self._known_mtimes.get(full)
                    if known is None or stat_mtime > known:
                        # New or updated file: trigger debounced reindex
                        self._schedule_action(full, "reindex")

            # Files in _known_mtimes that no longer exist on disk
            for tracked_path in list(self._known_mtimes.keys()):
                if tracked_path not in disk_paths:
                    logger.info(
                        "VaultFilewatcher audit: file gone from disk, deleting from sidecar: %s",
                        tracked_path,
                    )
                    self._known_mtimes.pop(tracked_path, None)
                    self._loop.create_task(self._do_delete(tracked_path))

        except Exception as exc:  # noqa: BLE001
            logger.exception("VaultFilewatcher audit job failed: %s", exc)

    def _cleanup_stale_mist_writes(self) -> None:
        """Remove expired MIST-write-in-flight markers.

        Called at the start of each audit run to prevent the dict from growing
        unboundedly if `clear_mist_write` is never called (e.g. VaultWriter
        crashed before the write completed).
        """
        now = time.monotonic()
        expired = [p for p, exp in self._mist_writes_in_flight.items() if exp <= now]
        for p in expired:
            self._mist_writes_in_flight.pop(p, None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scan_vault_mtimes(self) -> None:
        """Populate `_known_mtimes` with the current state of the vault.

        Called once at `start()` as a baseline snapshot. Does NOT trigger
        any reindex -- the sidecar is assumed to already reflect whatever
        state it was last written to.
        """
        try:
            for dirpath, _dirnames, filenames in os.walk(self.vault_root):
                for fname in filenames:
                    full = str(Path(dirpath) / fname)
                    if not _is_tracked_path(full):
                        continue
                    # Broken symlink or race -- skip silently
                    with contextlib.suppress(OSError):
                        self._known_mtimes[full] = int(Path(full).stat().st_mtime)
        except Exception as exc:  # noqa: BLE001
            logger.warning("VaultFilewatcher: initial mtime scan failed: %s", exc)

    def _create_observer(self):
        """Instantiate the watchdog observer for the configured observer_type.

        Returns:
            A watchdog BaseObserver subclass instance (not yet started).

        Raises:
            FilewatcherError: If the observer_type is unrecognised or cannot
                be imported on the current platform.
        """
        obs_type = self.config.observer_type.lower()

        if obs_type == "auto":
            try:
                from watchdog.observers import Observer

                return Observer()
            except ImportError as exc:
                raise FilewatcherError(
                    "Failed to import watchdog.observers.Observer: "
                    f"{exc}. Ensure watchdog>=3.0.0 is installed."
                ) from exc

        if obs_type == "polling":
            try:
                from watchdog.observers.polling import PollingObserver

                return PollingObserver()
            except ImportError as exc:
                raise FilewatcherError(f"Failed to import PollingObserver: {exc}") from exc

        if obs_type == "inotify":
            if platform.system() != "Linux":
                raise FilewatcherError(
                    f"observer_type='inotify' is only available on Linux "
                    f"(current platform: {platform.system()}). "
                    "Use observer_type='auto' or 'polling' instead."
                )
            try:
                from watchdog.observers.inotify import InotifyObserver

                return InotifyObserver()
            except ImportError as exc:
                raise FilewatcherError(
                    f"Failed to import InotifyObserver: {exc}. "
                    "Ensure watchdog[inotify] is installed and running on Linux."
                ) from exc

        if obs_type == "fsevents":
            if platform.system() != "Darwin":
                raise FilewatcherError(
                    f"observer_type='fsevents' is only available on macOS "
                    f"(current platform: {platform.system()}). "
                    "Use observer_type='auto' or 'polling' instead."
                )
            try:
                from watchdog.observers.fsevents import FSEventsObserver

                return FSEventsObserver()
            except ImportError as exc:
                raise FilewatcherError(
                    f"Failed to import FSEventsObserver: {exc}. "
                    "Ensure watchdog[fsevents] is installed and running on macOS."
                ) from exc

        raise FilewatcherError(
            f"Unknown observer_type '{self.config.observer_type}'. "
            "Valid values: auto, polling, inotify, fsevents."
        )
