"""`mist_admin graph-reset` reports the reset guard's nodes and relationships apart.

MIS-177 i107: the guard (`admin.RESET_GUARD_CYPHER`) counts `:__Entity__` nodes
AND the relationships touching them, and the command used to print that sum as
"N non-seed entities". These tests pin the separate counts in every line that
shows them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

from backend.knowledge import admin

# scripts/ is not a package; insert repo root so mist_admin is importable.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import mist_admin  # noqa: E402  -- after sys.path insertion


class _GuardConnection:
    """Answers the reset guard and the two census counts; refuses every write."""

    def __init__(self, guard_nodes: int, guard_rels: int) -> None:
        self._guard = {"nodes": guard_nodes, "relationships": guard_rels}
        self.writes: list[str] = []

    def execute_query(self, query, params=None):
        if query == admin.RESET_GUARD_CYPHER:
            return [dict(self._guard)]
        return [{"count": 7}]

    def execute_write(self, query, params=None):
        self.writes.append(query)
        raise AssertionError(f"graph-reset wrote while refusing: {query}")

    def disconnect(self) -> None:
        pass


def _run(monkeypatch, capsys, connection, *flags: str) -> tuple[int, str]:
    class _Backend:
        pass

    be = _Backend()
    be.admin = admin
    monkeypatch.setattr(mist_admin, "_load_backend", lambda: be)
    monkeypatch.setattr(mist_admin, "_connect", lambda _be: connection)
    args = argparse.Namespace(
        dry_run="--dry-run" in flags,
        confirm="--confirm" in flags,
        include_derived=False,
        no_snapshot=True,
        snapshot_to=None,
    )
    code = mist_admin.cmd_graph_reset(args)
    return code, capsys.readouterr().out


@pytest.mark.parametrize("flags", [("--dry-run",), ("--confirm",)])
def test_refusal_names_nodes_and_relationships_separately(monkeypatch, capsys, flags):
    connection = _GuardConnection(guard_nodes=2, guard_rels=5)

    _code, out = _run(monkeypatch, capsys, connection, *flags)

    assert "2 non-seed node(s) and 5 non-seed relationship(s)" in out
    assert "entities" not in out, out
    assert "7 non-seed" not in out, "the two counts were summed under one label"
    assert connection.writes == []


def test_confirmed_refusal_exits_2(monkeypatch, capsys):
    code, out = _run(monkeypatch, capsys, _GuardConnection(0, 1), "--confirm")

    assert code == 2
    assert "REFUSING: 0 non-seed node(s) and 1 non-seed relationship(s) present" in out
