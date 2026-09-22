"""`mist_admin.py uptime` -- the historical uptime report subcommand.

Exercises `cmd_uptime` directly with a parsed `argparse.Namespace`, following
the direct-call + capsys style of `tests/unit/test_admin_vault_cli.py` rather
than a subprocess. `_load_backend()` is patched to a minimal stub carrying
only `get_config()`, since `cmd_uptime` calls nothing else on it -- the same
pattern `tests/unit/scripts/test_rebuild_cli_is_read_only.py` uses for
`_build_log_regenerator`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import scripts.mist_admin as mist_admin
from tests.mocks.config import build_test_config

MODULE = "scripts.mist_admin"


def _make_args(*, job: str = "confidence_decay", db_path: str | None) -> argparse.Namespace:
    return argparse.Namespace(job=job, db_path=db_path)


def _stub_backend():
    config = build_test_config()
    return SimpleNamespace(get_config=lambda: config)


def _init_event_store(db_path: Path) -> None:
    """Create a real, schema-complete event store via EventStore.initialize().

    Uses the production schema rather than a hand-written CREATE TABLE, so
    these tests exercise the same table shape `_assert_replay_source_exists`
    and `read_curation_job_rows` see in production.
    """
    from backend.event_store.store import EventStore

    store = EventStore(str(db_path))
    store.initialize()
    store.close()


def _seed_two_scheduled_runs(db_path: Path) -> None:
    """Seed two scheduled `confidence_decay` rows, exactly one interval apart."""
    from backend.event_store.store import EventStore

    store = EventStore(str(db_path))
    store.initialize()
    store.append_curation_job_run(
        run_id="run-1",
        job_name="confidence_decay",
        trigger_source="scheduled",
        started_at="2026-01-01T00:00:00+00:00",
        duration_ms=1.0,
        outcome="completed",
        result_type="ConfidenceDecayResult",
        examined=0,
        produced=0,
        metrics="{}",
        error=None,
    )
    store.append_curation_job_run(
        run_id="run-2",
        job_name="confidence_decay",
        trigger_source="scheduled",
        started_at="2026-01-02T00:00:00+00:00",
        duration_ms=1.0,
        outcome="completed",
        result_type="ConfidenceDecayResult",
        examined=0,
        produced=0,
        metrics="{}",
        error=None,
    )
    store.close()


class TestEmptyVsUnopenableAreDistinct:
    """A readable-but-empty ledger and a missing store are different facts."""

    def test_empty_but_readable_ledger_exits_zero_with_nulls_and_a_reason(self, tmp_path, capsys):
        db_path = tmp_path / "event_store.db"
        _init_event_store(db_path)

        with patch(f"{MODULE}._load_backend", return_value=_stub_backend()):
            exit_code = mist_admin.cmd_uptime(_make_args(db_path=str(db_path)))

        out = capsys.readouterr().out
        assert exit_code == 0
        assert "REFUSED" not in out
        assert "nothing measured, not zero downtime" in out

    def test_missing_store_path_exits_two_refused_with_no_report_body(self, tmp_path, capsys):
        missing = tmp_path / "does-not-exist.db"

        with patch(f"{MODULE}._load_backend", return_value=_stub_backend()):
            exit_code = mist_admin.cmd_uptime(_make_args(db_path=str(missing)))

        out = capsys.readouterr().out
        assert exit_code == 2
        assert "[uptime] REFUSED:" in out
        # No report body: none of the report sections were reached.
        assert "WINDOW" not in out
        assert "LONGEST CONTINUOUS RUN" not in out


class TestReadOnly:
    def test_a_missing_path_creates_no_file(self, tmp_path):
        missing = tmp_path / "does-not-exist.db"

        with patch(f"{MODULE}._load_backend", return_value=_stub_backend()):
            mist_admin.cmd_uptime(_make_args(db_path=str(missing)))

        assert not missing.exists(), "a REFUSED run must not create the store it refused to find"

    def test_a_populated_store_is_unchanged_by_the_report(self, tmp_path):
        db_path = tmp_path / "event_store.db"
        _seed_two_scheduled_runs(db_path)

        before_bytes = db_path.read_bytes()
        before_stat = db_path.stat()

        with patch(f"{MODULE}._load_backend", return_value=_stub_backend()):
            exit_code = mist_admin.cmd_uptime(_make_args(db_path=str(db_path)))

        assert exit_code == 0
        assert db_path.read_bytes() == before_bytes, "the main database file changed"
        assert db_path.stat().st_size == before_stat.st_size, "the file size changed"
        assert db_path.stat().st_mtime == before_stat.st_mtime, "the file mtime changed"


class TestCaveatIsPresent:
    """The deliberate-shutdown caveat and the tolerance are product requirements."""

    def test_stdout_carries_the_deliberate_shutdown_caveat_and_the_tolerance(
        self, tmp_path, capsys
    ):
        db_path = tmp_path / "event_store.db"
        _seed_two_scheduled_runs(db_path)

        with patch(f"{MODULE}._load_backend", return_value=_stub_backend()):
            exit_code = mist_admin.cmd_uptime(_make_args(db_path=str(db_path)))

        out = capsys.readouterr().out
        assert exit_code == 0
        assert "cannot distinguish a deliberate shutdown" in out
        assert "docker compose" in out
        assert "300s" in out  # TOLERANCE_SECONDS, printed rather than inherited silently
