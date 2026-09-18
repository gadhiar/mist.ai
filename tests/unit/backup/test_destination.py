"""MIS-140 T1: where a backup may be written, and every destination refused.

The line that decides WHERE is the most dangerous line in a backup tool, so the
negative cases carry the weight here. Four of them are mandated by the ticket:
an unset `MIST_BACKUP_ROOT`, a root under `./data`, a root under the repository,
and (in `test_stores.py`) the stale `*-backup-*.db` files.

The repository case is the arm this package ADDS. `assert_isolated_root` refuses
the repo root itself through its contains-arm, but `<repo>/backups` passes all
three of its arms -- a fact asserted directly below, because the new arm is only
justified while that stays true.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from backend.knowledge.eval_isolation import REPO_ROOT, assert_isolated_root
from scripts.backup.destination import (
    BACKUP_ROOT_ENV,
    assert_backup_destination,
    resolve_backup_file,
    resolve_backup_root,
)
from scripts.backup.errors import BackupDestinationError


class TestTheGuardItAddsTo:
    def test_shared_guard_accepts_a_backup_dir_inside_the_repo(self):
        # The premise of the new arm. `assert_isolated_root` is correct as
        # written and deliberately un-overridable, and it has no reason to know
        # that a path inside a working tree is a bad place to keep a backup.
        # If this ever starts raising, the arm below becomes redundant rather
        # than wrong -- but until then, removing it would reopen the hole.
        assert_isolated_root(REPO_ROOT / "backups", purpose="test")


class TestUnsetDestination:
    def test_unset_env_is_refused_with_no_fallback(self, monkeypatch):
        monkeypatch.delenv(BACKUP_ROOT_ENV, raising=False)
        with pytest.raises(BackupDestinationError) as excinfo:
            resolve_backup_root()
        message = str(excinfo.value)
        assert BACKUP_ROOT_ENV in message
        assert "NO default destination" in message

    @pytest.mark.parametrize("value", ["", "   "], ids=["empty", "whitespace"])
    def test_blank_env_counts_as_unset(self, monkeypatch, value):
        # `export MIST_BACKUP_ROOT=` yields "", and `Path("").resolve()` is the
        # current directory -- which for a run started at the repo root is the
        # repository itself.
        monkeypatch.setenv(BACKUP_ROOT_ENV, value)
        with pytest.raises(BackupDestinationError):
            resolve_backup_root()

    def test_no_directory_is_created_on_refusal(self, monkeypatch, tmp_path):
        monkeypatch.delenv(BACKUP_ROOT_ENV, raising=False)
        monkeypatch.chdir(tmp_path)
        with pytest.raises(BackupDestinationError):
            resolve_backup_root()
        assert list(tmp_path.iterdir()) == []


class TestRefusesLiveState:
    @pytest.mark.parametrize(
        "candidate",
        [
            pytest.param(REPO_ROOT / "data", id="is-live-data-root"),
            pytest.param(REPO_ROOT / "data" / "backups", id="under-live-data-root"),
            pytest.param(REPO_ROOT / "mist-memory", id="is-live-vault"),
            pytest.param(Path("/app/data") / "nightly", id="under-container-data-root"),
            pytest.param(Path.home() / ".mist", id="is-user-store"),
            pytest.param(Path.home(), id="is-home-directory"),
            pytest.param(Path(Path.cwd().anchor), id="is-filesystem-root"),
        ],
    )
    def test_live_destinations_are_refused(self, candidate):
        with pytest.raises(BackupDestinationError):
            assert_backup_destination(candidate)

    def test_refusal_names_the_path_the_reason_and_the_remedy(self):
        # The old `graph-backup` default. An operator reading this at 3am must
        # not need the source: the path, why it was refused, and what to set
        # instead are all in the one string.
        with pytest.raises(BackupDestinationError) as excinfo:
            assert_backup_destination(REPO_ROOT / "data" / "graph_snapshots")
        message = str(excinfo.value)
        assert str((REPO_ROOT / "data" / "graph_snapshots").resolve()) in message
        assert "live state" in message
        assert BACKUP_ROOT_ENV in message
        assert "Nothing was written" in message

    def test_env_supplied_live_root_is_refused_too(self, monkeypatch):
        # The env var is not a trusted channel; it goes through the same arms.
        monkeypatch.setenv(BACKUP_ROOT_ENV, str(REPO_ROOT / "data"))
        with pytest.raises(BackupDestinationError):
            resolve_backup_root()


class TestRefusesTheRepository:
    def test_repo_root_itself_is_refused(self):
        with pytest.raises(BackupDestinationError):
            assert_backup_destination(REPO_ROOT)

    @pytest.mark.parametrize(
        "relative",
        ["backups", "backups/nightly", "scripts", "tmp/artifacts"],
    )
    def test_any_path_inside_the_working_tree_is_refused(self, relative):
        with pytest.raises(BackupDestinationError) as excinfo:
            assert_backup_destination(REPO_ROOT / relative)
        message = str(excinfo.value)
        assert str(REPO_ROOT) in message
        assert "working tree" in message

    def test_dotdot_cannot_smuggle_a_path_back_into_the_repo(self, tmp_path):
        # Resolution happens before any check, so a respelling is not a bypass.
        sneaky = tmp_path / ".." / tmp_path.name
        assert_backup_destination(sneaky)
        with pytest.raises(BackupDestinationError):
            assert_backup_destination(REPO_ROOT / "scripts" / ".." / "backups")


class TestAcceptsAnOffsiteRoot:
    def test_a_root_outside_the_repo_and_live_state_is_accepted(self, backup_root):
        assert assert_backup_destination(backup_root) == backup_root.resolve()

    def test_env_supplied_offsite_root_resolves(self, monkeypatch, backup_root):
        monkeypatch.setenv(BACKUP_ROOT_ENV, str(backup_root))
        assert resolve_backup_root() == backup_root.resolve()

    def test_explicit_path_overrides_the_environment(self, monkeypatch, backup_root, tmp_path):
        monkeypatch.setenv(BACKUP_ROOT_ENV, str(REPO_ROOT / "data"))
        other = tmp_path / "elsewhere"
        other.mkdir()
        assert resolve_backup_root(other) == other.resolve()

    def test_resolution_creates_nothing(self, monkeypatch, tmp_path):
        target = tmp_path / "not-yet-there"
        monkeypatch.setenv(BACKUP_ROOT_ENV, str(target))
        assert resolve_backup_root() == target.resolve()
        assert not target.exists()


class TestResolveBackupFile:
    def test_default_lands_under_the_backup_root(self, monkeypatch, backup_root):
        monkeypatch.setenv(BACKUP_ROOT_ENV, str(backup_root))
        assert resolve_backup_file("full-backup.json") == backup_root.resolve() / "full-backup.json"

    def test_explicit_file_is_guarded_through_its_parent(self):
        with pytest.raises(BackupDestinationError):
            resolve_backup_file(
                "ignored.json",
                REPO_ROOT / "data" / "graph_snapshots" / "full-backup.json",
            )

    def test_explicit_file_inside_the_repo_is_refused(self):
        with pytest.raises(BackupDestinationError):
            resolve_backup_file("ignored.json", REPO_ROOT / "backups" / "full-backup.json")

    def test_bare_filename_resolves_against_cwd_and_is_refused_in_the_repo(self, monkeypatch):
        monkeypatch.chdir(REPO_ROOT)
        with pytest.raises(BackupDestinationError):
            resolve_backup_file("ignored.json", "full-backup.json")

    def test_explicit_offsite_file_is_accepted(self, backup_root):
        target = backup_root / "graphs" / "full-backup.json"
        assert resolve_backup_file("ignored.json", target) == target.resolve()

    def test_unset_env_refuses_before_touching_the_filesystem(self, monkeypatch):
        monkeypatch.delenv(BACKUP_ROOT_ENV, raising=False)
        assert os.environ.get(BACKUP_ROOT_ENV) is None
        with pytest.raises(BackupDestinationError):
            resolve_backup_file("full-backup.json")
