"""Guard: importing backend.server must have no filesystem side effects.

`backend/server.py` used to create `/app/logs` and open `mist-backend.log`
at import time (lines 74-80 before this fix). Five unit modules import
`backend.server`, so under CI's non-root user, collection failed outright
with `PermissionError: [Errno 13] Permission denied: '/app'`. Inside the
backend container the same code additionally appended every unit test run's
logging to the live, bind-mounted production log
(`docker-compose.yml` mounts `./logs:/app/logs`) -- exactly what
`tests/unit/test_module_identity.py:91-107` monkeypatches `FileHandler` to
dodge.

The fix moves the file handler behind `configure_file_logging()`, called
only from `lifespan()`, and resolves the log directory from `MIST_LOG_DIR`
at call time rather than baking `/app/logs` in at import time. These tests
assert on the root logger's handlers and on filesystem state -- not on
captured output -- because the guard this file exists to enforce is about
what happens on disk and in `sys.modules`, not about what gets printed.
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Probe run in a subprocess so the import's real, first-time effects on the
# root logger are observable, and so a write cannot leak into pytest's own
# process the way it could if some other already-collected test module had
# imported backend.server first (import is cached in sys.modules).
_IMPORT_PROBE = """
import json
import logging
import sys

sys.path.insert(0, {repo_root!r})

import backend.server  # noqa: F401

root = logging.getLogger()
file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
print(json.dumps({{"file_handler_count": len(file_handlers)}}))
""".format(repo_root=str(REPO_ROOT))


def test_import_backend_server_has_no_filesystem_side_effects(tmp_path):
    """Import alone must create no log file, attach no FileHandler, touch no /app.

    MIST_LOG_DIR points at a tmp dir, and HOME/cwd point at other tmp dirs so
    nothing incidental is writable either. /app's mtime (or absence) is
    recorded before and after: if /app exists (as it does in the runner
    container), the import must not have touched it; if it does not exist
    (as on a plain dev checkout), it must still not exist afterward.
    """
    log_dir = tmp_path / "mist-log-dir"
    home_dir = tmp_path / "home"
    cwd_dir = tmp_path / "cwd"
    home_dir.mkdir()
    cwd_dir.mkdir()

    app_dir = Path("/app")
    app_existed_before = app_dir.exists()
    app_mtime_before = app_dir.stat().st_mtime if app_existed_before else None

    env = dict(os.environ)
    env["MIST_LOG_DIR"] = str(log_dir)
    env["HOME"] = str(home_dir)
    env["USERPROFILE"] = str(home_dir)

    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE],
        cwd=str(cwd_dir),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, (
        f"import probe failed: returncode={result.returncode}\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["file_handler_count"] == 0

    assert not log_dir.exists()
    assert not (log_dir / "mist-backend.log").exists()

    if app_existed_before:
        assert app_dir.stat().st_mtime == app_mtime_before
    else:
        assert not app_dir.exists()


@pytest.fixture
def clean_root_logger():
    """Track FileHandlers added to the root logger and remove them after."""
    root = logging.getLogger()
    before = set(root.handlers)
    yield root
    for handler in list(root.handlers):
        if handler not in before:
            root.removeHandler(handler)
            handler.close()


def test_configure_file_logging_creates_file_logs_and_is_idempotent(tmp_path, clean_root_logger):
    """configure_file_logging creates the file, logs land in it, no duplicate handler."""
    from backend.server import configure_file_logging

    log_dir = tmp_path / "logs"
    log_file = log_dir / "mist-backend.log"

    first_handler = configure_file_logging(log_dir)
    assert log_file.exists()
    assert isinstance(first_handler, logging.FileHandler)
    assert first_handler in clean_root_logger.handlers

    probe_logger = logging.getLogger("test_server_file_logging.probe")
    probe_logger.setLevel(logging.DEBUG)
    probe_logger.debug("probe record for configure_file_logging")
    first_handler.flush()
    assert "probe record for configure_file_logging" in log_file.read_text(encoding="utf-8")

    second_handler = configure_file_logging(log_dir)
    assert second_handler is first_handler

    file_handlers = [h for h in clean_root_logger.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 1


def test_resolve_log_dir_prefers_env_var(monkeypatch, tmp_path):
    """The resolver returns MIST_LOG_DIR's value when the env var is set."""
    from backend.server import _resolve_log_dir

    target = tmp_path / "custom-log-dir"
    monkeypatch.setenv("MIST_LOG_DIR", str(target))
    assert _resolve_log_dir() == Path(str(target))


def test_resolve_log_dir_defaults_to_repo_logs(monkeypatch):
    """Without MIST_LOG_DIR, the resolver returns the repo-relative logs/ dir."""
    from backend.server import _resolve_log_dir

    monkeypatch.delenv("MIST_LOG_DIR", raising=False)
    assert _resolve_log_dir() == REPO_ROOT / "logs"
