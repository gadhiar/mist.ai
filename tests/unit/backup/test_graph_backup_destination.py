"""MIS-140 T1: `mist_admin.py graph-backup` no longer writes inside live state.

Its default was `data/graph_snapshots/`
(`git show ebe1b0d:scripts/mist_admin.py | grep -n 'data/graph_snapshots'` ->
:427), which is under the state root it exists to protect: losing `./data` took
the event store, the extraction cache, the vault sidecar and every graph backup
in one stroke. The destination now comes from the same guard the dump leg uses,
and the resolution happens BEFORE the graph is read, so these tests never need a
connection.
"""

from __future__ import annotations

import argparse

import pytest

from backend.knowledge.eval_isolation import REPO_ROOT
from scripts.backup.destination import BACKUP_ROOT_ENV
from scripts.backup.errors import BackupDestinationError
from scripts.mist_admin import build_parser, cmd_graph_backup


def parse(argv: list[str]) -> argparse.Namespace:
    """Parse a `graph-backup` invocation through the real CLI parser."""
    return build_parser().parse_args(argv)


class TestNoLiveDefault:
    def test_unset_backup_root_is_refused_before_any_graph_read(self, monkeypatch):
        monkeypatch.delenv(BACKUP_ROOT_ENV, raising=False)
        with pytest.raises(BackupDestinationError) as excinfo:
            cmd_graph_backup(parse(["graph-backup"]))
        assert BACKUP_ROOT_ENV in str(excinfo.value)

    def test_the_old_default_directory_is_now_refused(self, monkeypatch):
        monkeypatch.delenv(BACKUP_ROOT_ENV, raising=False)
        with pytest.raises(BackupDestinationError):
            cmd_graph_backup(
                parse(["graph-backup", "--output", str(REPO_ROOT / "data" / "graph_snapshots")])
            )

    @pytest.mark.parametrize(
        "relative",
        ["data/graph_snapshots/full.json", "backups/full.json", "full.json"],
    )
    def test_no_output_inside_the_repo_is_accepted(self, monkeypatch, relative):
        monkeypatch.delenv(BACKUP_ROOT_ENV, raising=False)
        with pytest.raises(BackupDestinationError):
            cmd_graph_backup(parse(["graph-backup", "--output", str(REPO_ROOT / relative)]))

    def test_nothing_is_created_by_a_refusal(self, monkeypatch):
        monkeypatch.delenv(BACKUP_ROOT_ENV, raising=False)
        target = REPO_ROOT / "data" / "graph_snapshots"
        existed = target.exists()
        with pytest.raises(BackupDestinationError):
            cmd_graph_backup(parse(["graph-backup", "--output", str(target / "full.json")]))
        assert target.exists() is existed


class TestHelpTextNamesTheVariable:
    def test_the_output_flag_documents_the_required_variable(self, capsys):
        # The help text is where an operator learns there is no default, and it
        # is the only place they will look before the first run.
        with pytest.raises(SystemExit):
            build_parser().parse_args(["graph-backup", "--help"])
        printed = capsys.readouterr().out
        assert BACKUP_ROOT_ENV in printed
        assert "REQUIRED" in printed
