"""`graph-rebuild-from-log`'s read-only contract, enforced statically.

The command's entire advertised contract is "proof-first, dry-run only", and until
2026-08-05 it wrote to the LIVE SQLite event store on every run: `initialize()` on
both stores (`mkdir` + `executescript` + a conditional `ALTER TABLE`), then a
`rebuild-<epoch>-<uuid>` job row plus a checkpoint per turn, all doubled because
`_build_once` runs twice for the determinism gate.

Neither isolation guard could have caught it and neither ever will:
`assert_rebuild_target_not_live` and `assert_neo4j_isolated` both reason about bolt
URIs, and a SQLite path is invisible to them. The behavioural fix is the injected
journal (see `test_rebuild_journal_isolation.py`, which proves the regenerator never
writes to its replay source). What that proof CANNOT reach is the CLI's own wiring
-- `_build_log_regenerator` builds the stores, and reaching it in a test means a
backend load, a real embedding model, and a live Neo4j connection.

So this file checks the wiring the way the codebase already checks
`enforce_non_vacuity`'s dispatch sites: statically, so a regression fails the suite
rather than waiting for a review pass. Both rules are scoped to
`_build_log_regenerator` -- the one function that constructs the rebuild's stores.
Other `cmd_*` handlers in the same file legitimately initialize and write.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MIST_ADMIN = _REPO_ROOT / "scripts" / "mist_admin.py"
_BUILDER = "_build_log_regenerator"


def _builder_function() -> ast.FunctionDef:
    """Return the `_build_log_regenerator` AST node, failing loudly if it is renamed."""
    tree = ast.parse(_MIST_ADMIN.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == _BUILDER:
            return node
    pytest.fail(
        f"{_BUILDER} not found in {_MIST_ADMIN.name}. If it was renamed, update this "
        "file -- do not delete the check: it is the only thing standing between a "
        "dry-run command and the live event store."
    )


class TestTheReplaySourcesAreNeverInitialized:
    """`initialize()` is a WRITE: mkdir + executescript + a conditional ALTER TABLE."""

    def test_the_builder_initializes_neither_store(self):
        # Arrange
        builder = _builder_function()

        # Act
        initialized = [
            node.func.attr
            for node in ast.walk(builder)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "initialize"
        ]

        # Assert
        assert initialized == [], (
            f"{_BUILDER} calls .initialize() on a replay source. That is a write to "
            "live state from a dry-run command, and it also manufactures the absence "
            "it was meant to tolerate: an auto-created empty store is indistinguishable "
            "from a missing one. Use _assert_replay_source_exists instead."
        )

    def test_a_missing_replay_source_is_refused_rather_than_created(self):
        """The replacement for `initialize()` must actually be wired, not just present."""
        builder = _builder_function()

        guarded = [
            node.func.id
            for node in ast.walk(builder)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_assert_replay_source_exists"
        ]

        assert len(guarded) == 2, (
            "Expected _assert_replay_source_exists on BOTH replay sources (event store "
            f"and extraction cache); found {len(guarded)}. Dropping initialize() without "
            "this guard turns a missing store into a bare sqlite3.OperationalError."
        )


class TestTheJournalIsNonDurable:
    """A durable journal here sends the proof run's job rows to the live ledger."""

    def test_the_builder_wires_a_null_journal(self):
        # Arrange
        builder = _builder_function()

        # Act -- find the LogRegenerator(...) construction and read its journal= argument
        journals = [
            keyword.value
            for node in ast.walk(builder)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "LogRegenerator"
            for keyword in node.keywords
            if keyword.arg == "journal"
        ]

        # Assert
        assert len(journals) == 1, (
            f"Expected exactly one LogRegenerator(journal=...) in {_BUILDER}; found "
            f"{len(journals)}. `journal` is a required argument precisely so this "
            "decision is always visible at the call site."
        )
        journal = journals[0]
        assert (
            isinstance(journal, ast.Call)
            and isinstance(journal.func, ast.Name)
            and journal.func.id == "NullRebuildJournal"
        ), (
            "The rebuild CLI must wire NullRebuildJournal(). Anything durable writes "
            "job rows nothing reads (`get_reextraction_job` has no production caller) "
            "into the LIVE event store, twice per invocation."
        )
