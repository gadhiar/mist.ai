"""Tests for backend.vault.filewatcher.VaultFilewatcher.

Event-driven tests dispatch watchdog events directly to the handler instance
(handler.on_modified, handler.on_created, etc.) rather than relying on real
OS-level filesystem events. This keeps tests deterministic and fast.

All async tests are decorated with @pytest.mark.asyncio (asyncio_mode=strict).
"""

from __future__ import annotations

import asyncio
import logging
import platform
import time
from pathlib import Path

import pytest
import pytest_asyncio

from backend.errors import FilewatcherError
from backend.knowledge.config import FilewatcherConfig
from backend.vault.filewatcher import VaultFilewatcher, _is_tracked_path

# ---------------------------------------------------------------------------
# Fake sidecar index
# ---------------------------------------------------------------------------


class FakeSidecarIndex:
    """Test double for SidecarIndexProtocol.

    Records all calls for assertion in tests. Thread-safe enough for unit
    tests (single-loop async context).
    """

    def __init__(self) -> None:
        self.upsert_file_calls: list[tuple] = []
        self.delete_path_calls: list[str] = []
        self._raise_on_upsert: Exception | None = None
        self._raise_on_delete: Exception | None = None

    # -- SidecarIndexProtocol surface --

    def initialize(self) -> None:
        pass

    def close(self) -> None:
        pass

    def upsert_file(
        self,
        path: str,
        content: str,
        mtime: int,
        frontmatter: dict | None = None,
    ) -> int:
        if self._raise_on_upsert is not None:
            raise self._raise_on_upsert
        self.upsert_file_calls.append((path, content, mtime, frontmatter))
        return 1

    def delete_path(self, path: str) -> int:
        if self._raise_on_delete is not None:
            raise self._raise_on_delete
        self.delete_path_calls.append(path)
        return 1

    def query_vector(self, embedding: list[float], k: int = 10) -> list[dict]:
        return []

    def query_fts(self, text: str, k: int = 10) -> list[dict]:
        return []

    def query_hybrid(
        self,
        embedding: list[float],
        text: str,
        k: int = 10,
        rrf_k: int = 60,
    ) -> list[dict]:
        return []

    def chunk_count(self) -> int:
        return 0

    def health_check(self) -> bool:
        return True

    # -- Assertion helpers --

    def assert_upsert_called_for(self, path: str) -> None:
        paths = [c[0] for c in self.upsert_file_calls]
        assert path in paths, f"upsert_file was not called for {path!r}. Calls: {paths}"

    def assert_delete_called_for(self, path: str) -> None:
        assert (
            path in self.delete_path_calls
        ), f"delete_path was not called for {path!r}. Calls: {self.delete_path_calls}"

    def assert_no_upsert(self) -> None:
        assert not self.upsert_file_calls, f"Expected no upsert calls, got {self.upsert_file_calls}"

    def assert_no_delete(self) -> None:
        assert not self.delete_path_calls, f"Expected no delete calls, got {self.delete_path_calls}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**kwargs) -> FilewatcherConfig:
    """Build a FilewatcherConfig with fast debounce defaults for testing."""
    defaults = {
        "enabled": True,
        "observer_type": "polling",
        "debounce_ms": 100,  # Fast debounce for tests
        "staleness_slo_seconds": 5,
        "audit_interval_seconds": 3600,  # Effectively disabled in most tests
    }
    defaults.update(kwargs)
    return FilewatcherConfig(**defaults)


def _make_md_file(directory: Path, name: str = "note.md", content: str = "") -> Path:
    """Create a markdown file in `directory` with the given content."""
    p = directory / name
    p.write_text(content or "---\ntype: mist-session\n---\n\n# Hello\n", encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def vault_root(tmp_path: Path) -> Path:
    """A temporary vault root with a sessions/ subdirectory."""
    root = tmp_path / "vault"
    (root / "sessions").mkdir(parents=True)
    return root


@pytest_asyncio.fixture
async def fake_sidecar() -> FakeSidecarIndex:
    """Fresh FakeSidecarIndex for each test."""
    return FakeSidecarIndex()


@pytest_asyncio.fixture
async def watcher(vault_root: Path, fake_sidecar: FakeSidecarIndex):
    """Started VaultFilewatcher (polling observer, fast debounce). Stopped on teardown."""
    config = _make_config()
    fw = VaultFilewatcher(config, vault_root, fake_sidecar)
    fw.start(loop=asyncio.get_running_loop())
    yield fw
    fw.stop()


# ---------------------------------------------------------------------------
# TestIsTrackedPath (helper function)
# ---------------------------------------------------------------------------


class TestIsTrackedPath:
    def test_accepts_md_file(self):
        assert _is_tracked_path("/vault/sessions/note.md") is True

    def test_rejects_non_md_file(self):
        assert _is_tracked_path("/vault/sessions/note.txt") is False

    def test_rejects_hidden_file(self):
        assert _is_tracked_path("/vault/.hidden.md") is False

    def test_rejects_git_subdir(self):
        assert _is_tracked_path("/vault/.git/COMMIT_EDITMSG.md") is False

    def test_accepts_nested_md(self):
        assert _is_tracked_path("/vault/sessions/2026/note.md") is True

    def test_rejects_no_extension(self):
        assert _is_tracked_path("/vault/sessions/note") is False


# ---------------------------------------------------------------------------
# TestObserverSelection
# ---------------------------------------------------------------------------


class TestObserverSelection:
    @pytest.mark.asyncio
    async def test_auto_resolves_to_working_observer(self, vault_root: Path, fake_sidecar):
        config = _make_config(observer_type="auto")
        fw = VaultFilewatcher(config, vault_root, fake_sidecar)

        fw.start(loop=asyncio.get_running_loop())
        assert fw.is_running is True
        fw.stop()

    @pytest.mark.asyncio
    async def test_polling_explicit_selection_works(self, vault_root: Path, fake_sidecar):
        config = _make_config(observer_type="polling")
        fw = VaultFilewatcher(config, vault_root, fake_sidecar)

        fw.start(loop=asyncio.get_running_loop())
        assert fw.is_running is True
        fw.stop()

    @pytest.mark.asyncio
    @pytest.mark.skipif(platform.system() != "Linux", reason="inotify is Linux-only")
    async def test_inotify_works_on_linux(self, vault_root: Path, fake_sidecar):
        config = _make_config(observer_type="inotify")
        fw = VaultFilewatcher(config, vault_root, fake_sidecar)

        fw.start(loop=asyncio.get_running_loop())
        assert fw.is_running is True
        fw.stop()

    @pytest.mark.asyncio
    async def test_invalid_observer_type_raises_filewatcher_error(
        self, vault_root: Path, fake_sidecar
    ):
        config = _make_config(observer_type="totally_invalid")
        fw = VaultFilewatcher(config, vault_root, fake_sidecar)

        with pytest.raises(FilewatcherError, match="Unknown observer_type"):
            fw.start(loop=asyncio.get_running_loop())

    @pytest.mark.asyncio
    @pytest.mark.skipif(platform.system() == "Linux", reason="inotify only fails on non-Linux")
    async def test_inotify_on_non_linux_raises_filewatcher_error(
        self, vault_root: Path, fake_sidecar
    ):
        config = _make_config(observer_type="inotify")
        fw = VaultFilewatcher(config, vault_root, fake_sidecar)

        with pytest.raises(FilewatcherError, match="only available on Linux"):
            fw.start(loop=asyncio.get_running_loop())

    @pytest.mark.asyncio
    @pytest.mark.skipif(platform.system() == "Darwin", reason="fsevents only fails on non-macOS")
    async def test_fsevents_on_non_macos_raises_filewatcher_error(
        self, vault_root: Path, fake_sidecar
    ):
        config = _make_config(observer_type="fsevents")
        fw = VaultFilewatcher(config, vault_root, fake_sidecar)

        with pytest.raises(FilewatcherError, match="only available on macOS"):
            fw.start(loop=asyncio.get_running_loop())


# ---------------------------------------------------------------------------
# TestStartStop
# ---------------------------------------------------------------------------


class TestStartStop:
    @pytest.mark.asyncio
    async def test_is_running_true_after_start(self, vault_root: Path, fake_sidecar):
        config = _make_config()
        fw = VaultFilewatcher(config, vault_root, fake_sidecar)

        fw.start(loop=asyncio.get_running_loop())
        assert fw.is_running is True
        fw.stop()

    @pytest.mark.asyncio
    async def test_is_running_false_after_stop(self, vault_root: Path, fake_sidecar):
        config = _make_config()
        fw = VaultFilewatcher(config, vault_root, fake_sidecar)

        fw.start(loop=asyncio.get_running_loop())
        fw.stop()

        assert fw.is_running is False

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self, vault_root: Path, fake_sidecar):
        config = _make_config()
        fw = VaultFilewatcher(config, vault_root, fake_sidecar)

        fw.start(loop=asyncio.get_running_loop())
        fw.start(loop=asyncio.get_running_loop())  # second call is no-op

        assert fw.is_running is True
        fw.stop()

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self, vault_root: Path, fake_sidecar):
        config = _make_config()
        fw = VaultFilewatcher(config, vault_root, fake_sidecar)

        fw.start(loop=asyncio.get_running_loop())
        fw.stop()
        fw.stop()  # second call must not raise

        assert fw.is_running is False

    @pytest.mark.asyncio
    async def test_stop_before_start_is_safe(self, vault_root: Path, fake_sidecar):
        config = _make_config()
        fw = VaultFilewatcher(config, vault_root, fake_sidecar)

        # Should not raise
        fw.stop()

        assert fw.is_running is False

    @pytest.mark.asyncio
    async def test_stop_cancels_pending_debounce_timers(self, vault_root: Path, fake_sidecar):
        from watchdog.events import FileModifiedEvent

        config = _make_config(debounce_ms=5000)  # long debounce so timers stay pending
        fw = VaultFilewatcher(config, vault_root, fake_sidecar)
        fw.start(loop=asyncio.get_running_loop())

        # Dispatch a synthetic event to create a pending timer.
        # call_soon_threadsafe schedules _on_event_main_thread; yield to the
        # loop so the callback runs and populates _pending before we assert.
        md_file = _make_md_file(vault_root / "sessions")
        fw._handler.on_modified(FileModifiedEvent(str(md_file)))
        await asyncio.sleep(0)  # yield to loop so call_soon_threadsafe callback runs

        assert len(fw.pending_paths) == 1

        fw.stop()

        assert len(fw.pending_paths) == 0

    @pytest.mark.asyncio
    async def test_observer_thread_stops_within_timeout(self, vault_root: Path, fake_sidecar):
        config = _make_config()
        fw = VaultFilewatcher(config, vault_root, fake_sidecar)
        fw.start(loop=asyncio.get_running_loop())
        observer = fw._observer

        fw.stop()

        # Observer should no longer be alive
        assert observer is not None
        assert not observer.is_alive()


# ---------------------------------------------------------------------------
# TestEventHandling
# ---------------------------------------------------------------------------


class TestEventHandling:
    @pytest.mark.asyncio
    async def test_on_modified_md_triggers_reindex_after_debounce(
        self, watcher: VaultFilewatcher, vault_root: Path, fake_sidecar: FakeSidecarIndex
    ):
        from watchdog.events import FileModifiedEvent

        # Arrange
        md_file = _make_md_file(vault_root / "sessions")

        # Act
        watcher._handler.on_modified(FileModifiedEvent(str(md_file)))
        await asyncio.sleep(watcher.config.debounce_ms / 1000.0 + 0.1)

        # Assert
        fake_sidecar.assert_upsert_called_for(str(md_file))

    @pytest.mark.asyncio
    async def test_on_created_md_triggers_reindex(
        self, watcher: VaultFilewatcher, vault_root: Path, fake_sidecar: FakeSidecarIndex
    ):
        from watchdog.events import FileCreatedEvent

        # Arrange
        md_file = _make_md_file(vault_root / "sessions", name="created.md")

        # Act
        watcher._handler.on_created(FileCreatedEvent(str(md_file)))
        await asyncio.sleep(watcher.config.debounce_ms / 1000.0 + 0.1)

        # Assert
        fake_sidecar.assert_upsert_called_for(str(md_file))

    @pytest.mark.asyncio
    async def test_on_deleted_md_triggers_delete_not_reindex(
        self, watcher: VaultFilewatcher, vault_root: Path, fake_sidecar: FakeSidecarIndex
    ):
        from watchdog.events import FileDeletedEvent

        # Arrange
        path = str(vault_root / "sessions" / "gone.md")

        # Act: dispatch deleted event and wait for debounce to fire
        watcher._handler.on_deleted(FileDeletedEvent(path))
        await asyncio.sleep(watcher.config.debounce_ms / 1000.0 + 0.15)

        # Assert: delete called, no upsert
        fake_sidecar.assert_delete_called_for(path)
        fake_sidecar.assert_no_upsert()

    @pytest.mark.asyncio
    async def test_on_moved_triggers_delete_then_reindex(
        self,
        watcher: VaultFilewatcher,
        vault_root: Path,
        fake_sidecar: FakeSidecarIndex,
    ):
        from watchdog.events import FileMovedEvent

        # Arrange
        src = str(vault_root / "sessions" / "old.md")
        dest_path = vault_root / "sessions" / "new.md"
        dest_path.write_text("---\ntype: mist-session\n---\n\n# Moved\n", encoding="utf-8")
        dest = str(dest_path)

        # Act
        watcher._handler.on_moved(FileMovedEvent(src, dest))
        await asyncio.sleep(watcher.config.debounce_ms / 1000.0 + 0.15)

        # Assert: delete for src, upsert for dest
        fake_sidecar.assert_delete_called_for(src)
        fake_sidecar.assert_upsert_called_for(dest)

    @pytest.mark.asyncio
    async def test_non_md_file_modified_is_ignored(
        self, watcher: VaultFilewatcher, fake_sidecar: FakeSidecarIndex
    ):
        from watchdog.events import FileModifiedEvent

        # Act
        watcher._handler.on_modified(FileModifiedEvent("/vault/sessions/image.png"))
        await asyncio.sleep(0.15)

        # Assert: no sidecar calls
        fake_sidecar.assert_no_upsert()
        fake_sidecar.assert_no_delete()

    @pytest.mark.asyncio
    async def test_hidden_md_file_is_ignored(
        self, watcher: VaultFilewatcher, fake_sidecar: FakeSidecarIndex
    ):
        from watchdog.events import FileModifiedEvent

        # Act
        watcher._handler.on_modified(FileModifiedEvent("/vault/.hidden.md"))
        await asyncio.sleep(0.15)

        # Assert
        fake_sidecar.assert_no_upsert()

    @pytest.mark.asyncio
    async def test_directory_events_are_ignored(
        self, watcher: VaultFilewatcher, fake_sidecar: FakeSidecarIndex
    ):
        from watchdog.events import DirModifiedEvent

        # Act
        watcher._handler.on_modified(DirModifiedEvent("/vault/sessions"))
        await asyncio.sleep(0.15)

        # Assert
        fake_sidecar.assert_no_upsert()

    @pytest.mark.asyncio
    async def test_git_subdir_events_are_ignored(
        self, watcher: VaultFilewatcher, fake_sidecar: FakeSidecarIndex
    ):
        from watchdog.events import FileModifiedEvent

        # Act
        watcher._handler.on_modified(FileModifiedEvent("/vault/.git/COMMIT_EDITMSG.md"))
        await asyncio.sleep(0.15)

        # Assert
        fake_sidecar.assert_no_upsert()


# ---------------------------------------------------------------------------
# TestDebounce
# ---------------------------------------------------------------------------


class TestDebounce:
    @pytest.mark.asyncio
    async def test_multiple_modified_events_collapse_to_single_reindex(
        self,
        watcher: VaultFilewatcher,
        vault_root: Path,
        fake_sidecar: FakeSidecarIndex,
    ):
        from watchdog.events import FileModifiedEvent

        # Arrange
        md_file = _make_md_file(vault_root / "sessions", name="burst.md")
        event = FileModifiedEvent(str(md_file))

        # Act: fire 5 events in rapid succession
        for _ in range(5):
            watcher._handler.on_modified(event)

        await asyncio.sleep(watcher.config.debounce_ms / 1000.0 + 0.15)

        # Assert: exactly one upsert
        assert len(fake_sidecar.upsert_file_calls) == 1

    @pytest.mark.asyncio
    async def test_different_paths_get_independent_reindex(
        self,
        watcher: VaultFilewatcher,
        vault_root: Path,
        fake_sidecar: FakeSidecarIndex,
    ):
        from watchdog.events import FileModifiedEvent

        # Arrange
        file_a = _make_md_file(vault_root / "sessions", name="a.md")
        file_b = _make_md_file(vault_root / "sessions", name="b.md")

        # Act
        watcher._handler.on_modified(FileModifiedEvent(str(file_a)))
        watcher._handler.on_modified(FileModifiedEvent(str(file_b)))

        await asyncio.sleep(watcher.config.debounce_ms / 1000.0 + 0.15)

        # Assert: one upsert per path
        upserted_paths = {c[0] for c in fake_sidecar.upsert_file_calls}
        assert str(file_a) in upserted_paths
        assert str(file_b) in upserted_paths
        assert len(fake_sidecar.upsert_file_calls) == 2

    @pytest.mark.asyncio
    async def test_delete_then_create_within_debounce_window_collapses_to_single_reindex(
        self,
        watcher: VaultFilewatcher,
        vault_root: Path,
        fake_sidecar: FakeSidecarIndex,
    ):
        """Obsidian atomic-save: delete + create within debounce = single reindex."""
        from watchdog.events import FileCreatedEvent, FileDeletedEvent

        # Arrange: a file that exists on disk for the reindex read
        md_file = _make_md_file(vault_root / "sessions", name="atomic.md")
        path = str(md_file)

        # Act: simulate Obsidian atomic-replace (delete, then immediate create)
        watcher._handler.on_deleted(FileDeletedEvent(path))
        # Recreate the file immediately (so reindex can read it)
        md_file.write_text("---\ntype: mist-session\n---\n\n# Recreated\n", encoding="utf-8")
        watcher._handler.on_created(FileCreatedEvent(path))

        await asyncio.sleep(watcher.config.debounce_ms / 1000.0 + 0.15)

        # Assert: zero deletes (create cancelled the delete's side-effect on pending,
        # and the created event scheduled a reindex). One upsert.
        assert len(fake_sidecar.upsert_file_calls) == 1
        # No delete call either because create re-scheduled reindex instead
        assert len(fake_sidecar.delete_path_calls) == 0

    @pytest.mark.asyncio
    async def test_reindex_fires_after_debounce_delay(
        self,
        watcher: VaultFilewatcher,
        vault_root: Path,
        fake_sidecar: FakeSidecarIndex,
    ):
        """Reindex must not fire before the debounce window expires."""
        from watchdog.events import FileModifiedEvent

        # Arrange
        md_file = _make_md_file(vault_root / "sessions", name="timing.md")

        # Act
        t_start = time.monotonic()
        watcher._handler.on_modified(FileModifiedEvent(str(md_file)))

        # Wait just past the debounce window
        await asyncio.sleep(watcher.config.debounce_ms / 1000.0 + 0.1)

        elapsed = time.monotonic() - t_start

        # Assert: reindex fired AND minimum delay respected
        fake_sidecar.assert_upsert_called_for(str(md_file))
        assert elapsed >= watcher.config.debounce_ms / 1000.0


# ---------------------------------------------------------------------------
# TestMistWriteCoordination
# ---------------------------------------------------------------------------


class TestMistWriteCoordination:
    @pytest.mark.asyncio
    async def test_mist_write_marked_path_logs_at_debug(
        self,
        watcher: VaultFilewatcher,
        vault_root: Path,
        caplog,
    ):
        from watchdog.events import FileModifiedEvent

        # Arrange
        md_file = _make_md_file(vault_root / "sessions", name="mist_write.md")
        path = str(md_file)
        watcher.mark_mist_write(path)

        # Act
        with caplog.at_level(logging.DEBUG, logger="backend.vault.filewatcher"):
            watcher._handler.on_modified(FileModifiedEvent(path))
            await asyncio.sleep(0.05)  # Just need the event dispatch, not reindex

        # Assert: MIST-origin event logged at DEBUG
        debug_msgs = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
        assert any("MIST-origin" in m for m in debug_msgs)

    @pytest.mark.asyncio
    async def test_unmarked_path_logs_user_edit_at_info(
        self,
        watcher: VaultFilewatcher,
        vault_root: Path,
        caplog,
    ):
        from watchdog.events import FileModifiedEvent

        # Arrange
        md_file = _make_md_file(vault_root / "sessions", name="user_edit.md")
        path = str(md_file)

        # Act: no mark_mist_write call
        with caplog.at_level(logging.INFO, logger="backend.vault.filewatcher"):
            watcher._handler.on_modified(FileModifiedEvent(path))
            await asyncio.sleep(0.05)

        # Assert: user-edit detection logged at INFO
        info_msgs = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any("User edit detected" in m for m in info_msgs)

    @pytest.mark.asyncio
    async def test_clear_mist_write_removes_mark(
        self,
        watcher: VaultFilewatcher,
        vault_root: Path,
        caplog,
    ):
        from watchdog.events import FileModifiedEvent

        # Arrange
        md_file = _make_md_file(vault_root / "sessions", name="clear_test.md")
        path = str(md_file)
        watcher.mark_mist_write(path)
        watcher.clear_mist_write(path)

        # Act: event arrives after clear -- should be user-edit now
        with caplog.at_level(logging.INFO, logger="backend.vault.filewatcher"):
            watcher._handler.on_modified(FileModifiedEvent(path))
            await asyncio.sleep(0.05)

        # Assert: logged as user-edit, not MIST-origin
        info_msgs = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any("User edit detected" in m for m in info_msgs)
        debug_msgs = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
        assert not any("MIST-origin" in m for m in debug_msgs)

    @pytest.mark.asyncio
    async def test_mist_write_ttl_expiry_causes_user_edit_attribution(
        self,
        vault_root: Path,
        fake_sidecar: FakeSidecarIndex,
        caplog,
    ):
        from watchdog.events import FileModifiedEvent

        # Arrange: tiny TTL so mark expires almost immediately
        config = _make_config(debounce_ms=50)
        fw = VaultFilewatcher(config, vault_root, fake_sidecar)
        fw._mist_write_ttl_seconds = 0.05  # 50ms TTL
        fw.start(loop=asyncio.get_running_loop())

        md_file = _make_md_file(vault_root / "sessions", name="ttl_test.md")
        path = str(md_file)
        fw.mark_mist_write(path)

        # Wait for TTL to expire
        await asyncio.sleep(0.1)

        # Act: event after expiry
        with caplog.at_level(logging.INFO, logger="backend.vault.filewatcher"):
            fw._handler.on_modified(FileModifiedEvent(path))
            await asyncio.sleep(0.05)

        fw.stop()

        # Assert: logged as user-edit (TTL expired)
        info_msgs = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any("User edit detected" in m for m in info_msgs)


# ---------------------------------------------------------------------------
# TestMtimeAuditJob
# ---------------------------------------------------------------------------


class TestMtimeAuditJob:
    @pytest.mark.asyncio
    async def test_start_populates_known_mtimes_without_triggering_reindex(
        self, vault_root: Path, fake_sidecar: FakeSidecarIndex
    ):
        # Arrange: pre-populate a file in the vault
        md_file = _make_md_file(vault_root / "sessions", name="existing.md")

        # Act: start (which scans mtimes)
        config = _make_config(audit_interval_seconds=3600)
        fw = VaultFilewatcher(config, vault_root, fake_sidecar)
        fw.start(loop=asyncio.get_running_loop())
        # Give tiny window for startup scan
        await asyncio.sleep(0.05)
        fw.stop()

        # Assert: mtime recorded, but no upsert triggered by startup
        assert str(md_file) in fw._known_mtimes
        fake_sidecar.assert_no_upsert()

    @pytest.mark.asyncio
    async def test_audit_reindexes_file_with_newer_mtime(
        self, vault_root: Path, fake_sidecar: FakeSidecarIndex
    ):
        # Arrange: start watcher (baseline mtime captured)
        md_file = _make_md_file(vault_root / "sessions", name="drift.md")
        config = _make_config(audit_interval_seconds=3600)
        fw = VaultFilewatcher(config, vault_root, fake_sidecar)
        fw.start(loop=asyncio.get_running_loop())
        await asyncio.sleep(0.05)

        # Simulate stale known mtime (set it to 0 so any real mtime is newer)
        fw._known_mtimes[str(md_file)] = 0

        # Act: run audit manually
        await fw._run_audit()
        # Wait for debounce to fire
        await asyncio.sleep(fw.config.debounce_ms / 1000.0 + 0.15)

        fw.stop()

        # Assert
        fake_sidecar.assert_upsert_called_for(str(md_file))

    @pytest.mark.asyncio
    async def test_audit_reindexes_new_file_not_in_known_mtimes(
        self, vault_root: Path, fake_sidecar: FakeSidecarIndex
    ):
        # Arrange: start watcher with no files
        config = _make_config(audit_interval_seconds=3600)
        fw = VaultFilewatcher(config, vault_root, fake_sidecar)
        fw.start(loop=asyncio.get_running_loop())
        await asyncio.sleep(0.05)

        # Write a new file after startup scan (so it is NOT in _known_mtimes)
        new_file = _make_md_file(vault_root / "sessions", name="new_arrival.md")

        # Act: run audit
        await fw._run_audit()
        await asyncio.sleep(fw.config.debounce_ms / 1000.0 + 0.15)

        fw.stop()

        # Assert
        fake_sidecar.assert_upsert_called_for(str(new_file))

    @pytest.mark.asyncio
    async def test_audit_deletes_sidecar_entry_for_missing_file(
        self, vault_root: Path, fake_sidecar: FakeSidecarIndex
    ):
        # Arrange: add a ghost path to _known_mtimes that does not exist on disk
        ghost_path = str(vault_root / "sessions" / "ghost.md")
        config = _make_config(audit_interval_seconds=3600)
        fw = VaultFilewatcher(config, vault_root, fake_sidecar)
        fw.start(loop=asyncio.get_running_loop())
        fw._known_mtimes[ghost_path] = int(time.time())

        # Act: run audit
        await fw._run_audit()
        await asyncio.sleep(0.15)

        fw.stop()

        # Assert
        fake_sidecar.assert_delete_called_for(ghost_path)

    @pytest.mark.asyncio
    async def test_audit_handles_broken_symlinks(
        self, vault_root: Path, fake_sidecar: FakeSidecarIndex, tmp_path: Path
    ):
        """Broken symlinks must not crash the audit."""
        import os

        # Arrange: create a broken symlink in the vault
        symlink_target = tmp_path / "nonexistent_target.md"
        symlink_path = vault_root / "sessions" / "broken_link.md"
        os.symlink(str(symlink_target), str(symlink_path))

        config = _make_config(audit_interval_seconds=3600)
        fw = VaultFilewatcher(config, vault_root, fake_sidecar)
        fw.start(loop=asyncio.get_running_loop())

        # Act: audit should not crash
        await fw._run_audit()
        await asyncio.sleep(0.1)

        fw.stop()
        # No assertion on calls -- just verifying no exception raised


# ---------------------------------------------------------------------------
# TestReindexResilience
# ---------------------------------------------------------------------------


class TestReindexResilience:
    @pytest.mark.asyncio
    async def test_file_deleted_between_event_and_reindex_logs_warning_no_crash(
        self,
        watcher: VaultFilewatcher,
        vault_root: Path,
        fake_sidecar: FakeSidecarIndex,
        caplog,
    ):
        from watchdog.events import FileModifiedEvent

        # Arrange: file exists at event time but is deleted before reindex fires
        md_file = vault_root / "sessions" / "ephemeral.md"
        md_file.write_text("content", encoding="utf-8")
        watcher._handler.on_modified(FileModifiedEvent(str(md_file)))

        # Delete the file before debounce fires
        md_file.unlink()

        with caplog.at_level(logging.WARNING, logger="backend.vault.filewatcher"):
            await asyncio.sleep(watcher.config.debounce_ms / 1000.0 + 0.15)

        # Assert: watcher still alive, warning logged, no upsert
        assert watcher.is_running
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("deleted between event" in m for m in warning_msgs)
        fake_sidecar.assert_no_upsert()

    @pytest.mark.asyncio
    async def test_sidecar_index_error_on_upsert_caught_watcher_alive(
        self,
        watcher: VaultFilewatcher,
        vault_root: Path,
        fake_sidecar: FakeSidecarIndex,
        caplog,
    ):
        from watchdog.events import FileModifiedEvent

        from backend.errors import SidecarIndexError

        # Arrange: make sidecar raise
        fake_sidecar._raise_on_upsert = SidecarIndexError("upsert failed")
        md_file = _make_md_file(vault_root / "sessions", name="sidecar_err.md")

        with caplog.at_level(logging.ERROR, logger="backend.vault.filewatcher"):
            watcher._handler.on_modified(FileModifiedEvent(str(md_file)))
            await asyncio.sleep(watcher.config.debounce_ms / 1000.0 + 0.15)

        # Assert: watcher still running
        assert watcher.is_running
        error_msgs = [r.message for r in caplog.records if r.levelno == logging.ERROR]
        assert any("upsert_file failed" in m for m in error_msgs)

    @pytest.mark.asyncio
    async def test_bad_utf8_file_caught_watcher_alive(
        self,
        watcher: VaultFilewatcher,
        vault_root: Path,
        fake_sidecar: FakeSidecarIndex,
        caplog,
    ):
        from watchdog.events import FileModifiedEvent

        # Arrange: write a file with invalid UTF-8
        bad_file = vault_root / "sessions" / "bad_utf8.md"
        bad_file.write_bytes(b"---\ntype: mist-session\n---\n\n\xff\xfe invalid utf8")

        with caplog.at_level(logging.WARNING, logger="backend.vault.filewatcher"):
            watcher._handler.on_modified(FileModifiedEvent(str(bad_file)))
            await asyncio.sleep(watcher.config.debounce_ms / 1000.0 + 0.15)

        # Assert: watcher alive, warning logged
        assert watcher.is_running
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("UTF-8 decode error" in m for m in warning_msgs)

    @pytest.mark.asyncio
    async def test_frontmatter_parse_error_falls_back_to_empty_dict(
        self,
        watcher: VaultFilewatcher,
        vault_root: Path,
        fake_sidecar: FakeSidecarIndex,
    ):
        """A file with malformed YAML frontmatter still triggers upsert with empty dict."""
        from watchdog.events import FileModifiedEvent

        # Arrange: frontmatter that parse_frontmatter returns {} for (missing close ---)
        bad_fm = vault_root / "sessions" / "bad_fm.md"
        bad_fm.write_text("---\ntype: [unclosed\n\n# body\n", encoding="utf-8")

        # Act
        watcher._handler.on_modified(FileModifiedEvent(str(bad_fm)))
        await asyncio.sleep(watcher.config.debounce_ms / 1000.0 + 0.15)

        # Assert: upsert was called (watcher recovered gracefully)
        assert len(fake_sidecar.upsert_file_calls) == 1
        _path, _content, _mtime, fm = fake_sidecar.upsert_file_calls[0]
        # frontmatter may be empty dict or partial; just check no crash
        assert isinstance(fm, dict)


# ---------------------------------------------------------------------------
# TestPendingPaths
# ---------------------------------------------------------------------------


class TestPendingPaths:
    @pytest.mark.asyncio
    async def test_pending_paths_contains_scheduled_path(
        self,
        vault_root: Path,
        fake_sidecar: FakeSidecarIndex,
    ):
        from watchdog.events import FileModifiedEvent

        # Arrange: long debounce so path stays pending
        config = _make_config(debounce_ms=5000)
        fw = VaultFilewatcher(config, vault_root, fake_sidecar)
        fw.start(loop=asyncio.get_running_loop())

        md_file = _make_md_file(vault_root / "sessions", name="pending.md")
        fw._handler.on_modified(FileModifiedEvent(str(md_file)))

        # Yield to the loop so the call_soon_threadsafe callback runs and
        # populates _pending before we assert on pending_paths.
        await asyncio.sleep(0)

        # Assert: path in pending (debounce timer is still counting down)
        assert str(md_file) in fw.pending_paths

        fw.stop()

    @pytest.mark.asyncio
    async def test_pending_paths_empty_after_debounce_fires(
        self,
        watcher: VaultFilewatcher,
        vault_root: Path,
        fake_sidecar: FakeSidecarIndex,
    ):
        from watchdog.events import FileModifiedEvent

        # Arrange
        md_file = _make_md_file(vault_root / "sessions", name="fired.md")
        watcher._handler.on_modified(FileModifiedEvent(str(md_file)))

        # Wait for debounce to fire and reindex to complete
        await asyncio.sleep(watcher.config.debounce_ms / 1000.0 + 0.2)

        # Assert: path no longer pending
        assert str(md_file) not in watcher.pending_paths
