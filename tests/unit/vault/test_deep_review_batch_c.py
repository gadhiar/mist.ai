"""Deep-review Batch C regressions: filewatcher/writer wiring and contracts.

Covers vault-layer-adr010-1 (writer=None composition), concurrency-async-1
(MIST writes classified as user edits), vault-layer-adr010-3 (authored_by
writeback racing the consumer), concurrency-async-6 (unguarded invariant-5
tasks), vault-layer-adr010-4 (audit prune mass-deletion), vault-layer-adr010-5
(hidden-directory exclusion drift), and vault-layer-adr010-6 (identity
bootstrap clobbering hand edits).
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from backend.knowledge.config import FilewatcherConfig, VaultConfig
from backend.vault.filewatcher import VaultFilewatcher, _is_tracked_path
from backend.vault.writer import VaultWriter

# ---------------------------------------------------------------------------
# Stubs (mirroring tests/unit/vault/test_filewatcher.py)
# ---------------------------------------------------------------------------


class _RecordingSidecar:
    def __init__(self) -> None:
        self.upsert_file_calls: list[tuple] = []
        self.delete_path_calls: list[str] = []

    def upsert_file(self, path, content, mtime, frontmatter=None) -> int:
        self.upsert_file_calls.append((path, content, mtime, frontmatter))
        return 1

    def delete_path(self, path) -> int:
        self.delete_path_calls.append(path)
        return 1

    def distinct_paths(self) -> list[str]:
        return []


class _RecordingWriter:
    def __init__(self, raise_on_mark: Exception | None = None) -> None:
        self.mark_calls: list[Path] = []
        self._raise_on_mark = raise_on_mark

    async def mark_authored_by_user_edit(self, path: Path) -> None:
        if self._raise_on_mark is not None:
            raise self._raise_on_mark
        self.mark_calls.append(path)


class _RecordingRegenerator:
    def __init__(self) -> None:
        self.rebuild_calls: list[Path] = []

    async def rebuild_from_path(self, path: Path):
        self.rebuild_calls.append(path)
        return None


class _RecordingBus:
    def __init__(self) -> None:
        self.published: list[object] = []

    async def publish(self, event: object) -> None:
        self.published.append(event)


def _config(**kwargs) -> FilewatcherConfig:
    defaults = {
        "enabled": True,
        "observer_type": "polling",
        "debounce_ms": 100,
        "staleness_slo_seconds": 5,
        "audit_interval_seconds": 3600,
    }
    defaults.update(kwargs)
    return FilewatcherConfig(**defaults)


def _watcher(vault_root: Path, sidecar=None, writer=None, regenerator=None, bus=None):
    return VaultFilewatcher(
        _config(),
        vault_root,
        sidecar or _RecordingSidecar(),
        regenerator=regenerator or _RecordingRegenerator(),
        invalidation_bus=bus or _RecordingBus(),
        writer=writer,
    )


def _session_note(root: Path, name: str = "note.md") -> Path:
    sessions = root / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    p = sessions / name
    p.write_text(
        "---\ntype: mist-session\nauthored_by: mist\n---\n\n# Note\n",
        encoding="utf-8",
    )
    return p


# ---------------------------------------------------------------------------
# vault-layer-adr010-5: hidden-directory exclusion parity
# ---------------------------------------------------------------------------


class TestTrackedPathHiddenDirectories:
    @pytest.mark.parametrize(
        "rel, expected",
        [
            pytest.param("sessions/x.md", True, id="normal-note"),
            pytest.param(".trash/x.md", False, id="obsidian-trash"),
            pytest.param(".obsidian/templates/x.md", False, id="obsidian-config"),
            pytest.param(".git/x.md", False, id="git-subtree"),
            pytest.param("sub/.hidden.md", False, id="hidden-basename"),
            pytest.param("sub/.cache/x.md", False, id="nested-hidden-dir"),
        ],
    )
    def test_hidden_path_parts_are_rejected(self, tmp_path: Path, rel: str, expected: bool):
        full = tmp_path / "vault" / Path(rel)
        assert _is_tracked_path(str(full)) is expected

    def test_parity_with_cli_rebuild_walk(self, tmp_path: Path):
        # The CLI walk and the live watcher must agree on corpus membership.
        from scripts.mist_admin import _walk_vault_md_files

        root = tmp_path / "vault"
        keep = root / "sessions" / "a.md"
        trash = root / ".trash" / "b.md"
        hidden = root / "sub" / ".hidden.md"
        for p in (keep, trash, hidden):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("x", encoding="utf-8")

        cli_paths = {str(p) for p in _walk_vault_md_files(root)}
        watcher_paths = {str(p) for p in (keep, trash, hidden) if _is_tracked_path(str(p))}

        assert cli_paths == watcher_paths == {str(keep)}


# ---------------------------------------------------------------------------
# vault-layer-adr010-1 / concurrency-async-6: invariant-5 guard rails
# ---------------------------------------------------------------------------


class TestInvariant5GuardRails:
    @pytest.mark.asyncio
    async def test_none_writer_skips_invariant5_loudly(self, tmp_path: Path, caplog):
        sidecar = _RecordingSidecar()
        w = _watcher(tmp_path / "vault", sidecar=sidecar, writer=None)
        note = _session_note(tmp_path / "vault")

        with caplog.at_level("ERROR"):
            await w._do_reindex(str(note), is_mist_write=False)

        assert sidecar.upsert_file_calls, "sidecar reindex must still run"
        assert any("invariant-5" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_invariant5_failure_is_contained_and_retryable(self, tmp_path: Path, caplog):
        sidecar = _RecordingSidecar()
        writer = _RecordingWriter(raise_on_mark=RuntimeError("disk hiccup"))
        w = _watcher(tmp_path / "vault", sidecar=sidecar, writer=writer)
        note = _session_note(tmp_path / "vault")

        with caplog.at_level("ERROR"):
            await w._do_reindex(str(note), is_mist_write=False)  # must not raise

        assert any("invariant-5 sequence failed" in r.message for r in caplog.records)
        # mtime dropped so the audit job re-schedules and retries the path
        assert str(note) not in w._known_mtimes

    @pytest.mark.asyncio
    async def test_invariant5_success_runs_all_three_steps(self, tmp_path: Path):
        writer = _RecordingWriter()
        regen = _RecordingRegenerator()
        bus = _RecordingBus()
        w = _watcher(tmp_path / "vault", writer=writer, regenerator=regen, bus=bus)
        note = _session_note(tmp_path / "vault")

        await w._do_reindex(str(note), is_mist_write=False)

        assert writer.mark_calls == [Path(str(note))]
        assert regen.rebuild_calls == [Path(str(note))]
        assert len(bus.published) == 1


# ---------------------------------------------------------------------------
# concurrency-async-1: event-time classification carried through debounce
# ---------------------------------------------------------------------------


class TestMistWriteClassification:
    @pytest.mark.asyncio
    async def test_event_time_classification_survives_marker_expiry(self, tmp_path: Path):
        # A MIST write classified at event arrival must stay MIST-origin even
        # when the marker TTL lapses before the debounce timer fires.
        writer = _RecordingWriter()
        sidecar = _RecordingSidecar()
        root = tmp_path / "vault"
        w = _watcher(root, sidecar=sidecar, writer=writer)
        note = _session_note(root)
        w._loop = asyncio.get_running_loop()
        w._running = True

        w.mark_mist_write(str(note))
        w._on_event_main_thread("modified", str(note))
        # Force-expire the marker before the debounce fires
        w._mist_writes_in_flight[str(note)] = time.monotonic() - 1.0

        await asyncio.sleep(0.4)  # debounce_ms=100 + margin

        assert sidecar.upsert_file_calls, "reindex must run"
        assert writer.mark_calls == [], "MIST write must not trigger the user-edit sequence"

    @pytest.mark.asyncio
    async def test_user_edit_still_runs_invariant5(self, tmp_path: Path):
        writer = _RecordingWriter()
        root = tmp_path / "vault"
        w = _watcher(root, writer=writer)
        note = _session_note(root)
        w._loop = asyncio.get_running_loop()
        w._running = True

        w._on_event_main_thread("modified", str(note))  # no marker: user edit
        await asyncio.sleep(0.4)

        assert writer.mark_calls == [Path(str(note))]


# ---------------------------------------------------------------------------
# vault-layer-adr010-4: audit prune guard
# ---------------------------------------------------------------------------


class TestAuditPruneGuard:
    @pytest.mark.asyncio
    async def test_empty_walk_with_known_paths_skips_prune(self, tmp_path: Path, caplog):
        sidecar = _RecordingSidecar()
        # vault_root that does not exist: walk yields nothing + onerror fires
        w = _watcher(tmp_path / "missing-vault", sidecar=sidecar)
        w._loop = asyncio.get_running_loop()
        w._known_mtimes = {str(tmp_path / "missing-vault" / "a.md"): 5}

        with caplog.at_level("WARNING"):
            await w._run_audit()

        assert sidecar.delete_path_calls == []
        assert w._known_mtimes, "known paths must be preserved for the next clean audit"
        assert any("skipping" in r.message and "prune" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Writer-side contracts (consumer self-marking, queued writeback, identity)
# ---------------------------------------------------------------------------


def _vault_config(tmp_path: Path) -> VaultConfig:
    return VaultConfig(enabled=True, root=str(tmp_path / "vault"), git_auto_init=False)


class TestWriterSelfMarking:
    @pytest.mark.asyncio
    async def test_consumer_marks_path_before_each_write(self, tmp_path: Path):
        writer = VaultWriter(_vault_config(tmp_path))
        calls: list[tuple[str, bool]] = []
        writer.set_mist_write_marker(lambda p: calls.append((p, Path(p).exists())))
        await writer.start()
        try:
            path = await writer.append_turn_to_session("sess-mark", 1, "hi", "hello")
        finally:
            await writer.stop()

        assert [c[0] for c in calls] == [path]
        assert calls[0][1] is False, "marker must fire BEFORE the file is written"

    @pytest.mark.asyncio
    async def test_mark_authored_by_user_edit_is_queued_and_self_marked(self, tmp_path: Path):
        writer = VaultWriter(_vault_config(tmp_path))
        marked: list[str] = []
        writer.set_mist_write_marker(marked.append)
        await writer.start()
        try:
            note = Path(writer.config.root) / "sessions" / "n.md"
            note.parent.mkdir(parents=True, exist_ok=True)
            note.write_text(
                "---\ntype: mist-session\nauthored_by: mist\n---\n\nbody\n",
                encoding="utf-8",
            )
            await writer.mark_authored_by_user_edit(note)
        finally:
            await writer.stop()

        assert "authored_by: user-edit" in note.read_text(encoding="utf-8")
        assert str(note) in marked, "writeback must self-mark as a MIST write"


class TestIdentityBootstrapGuard:
    @pytest.mark.asyncio
    async def test_upsert_identity_preserves_user_edited_body(self, tmp_path: Path, caplog):
        writer = VaultWriter(_vault_config(tmp_path))
        await writer.start()
        try:
            identity = Path(writer.config.root) / "identity" / "mist.md"
            identity.parent.mkdir(parents=True, exist_ok=True)
            identity.write_text(
                "---\ntype: mist-identity\nauthored_by: user-edit\n---\n\nHAND EDITED PERSONA\n",
                encoding="utf-8",
            )
            with caplog.at_level("WARNING"):
                await writer.upsert_identity(
                    traits=[{"display_name": "Warm", "axis": "tone", "description": "d"}],
                    capabilities=[],
                    preferences=[],
                )
        finally:
            await writer.stop()

        assert "HAND EDITED PERSONA" in identity.read_text(encoding="utf-8")
        assert any("user-edit" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_upsert_identity_refreshes_machine_authored_file(self, tmp_path: Path):
        # Files still carrying the machine-stamped birth value must refresh
        # so seed_data.yaml updates flow through.
        writer = VaultWriter(_vault_config(tmp_path))
        await writer.start()
        try:
            await writer.upsert_identity(
                traits=[{"display_name": "Warm", "axis": "tone", "description": "first"}],
                capabilities=[],
                preferences=[],
            )
            await writer.upsert_identity(
                traits=[{"display_name": "Playful", "axis": "tone", "description": "second"}],
                capabilities=[],
                preferences=[],
            )
            identity = Path(writer.config.root) / "identity" / "mist.md"
            content = identity.read_text(encoding="utf-8")
        finally:
            await writer.stop()

        assert "Playful" in content
