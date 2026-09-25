"""T6 reviewer finding 4: the `extraction` suite's mist-backend image must be
resolved and validated in `cmd_run`'s PREFLIGHT -- before any suite runs and
before any file is written. A missing snapshot must print a clean [FAIL] and
write nothing (no meta.json, no vram.csv).

Reuses `test_host_run_cumulative.py`'s stub fixture (docker/props/probes
stubbed, no docker, no network, no GPU) rather than reimplementing it.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.model_bench import bench_host  # noqa: E402
from tests.unit.model_bench.test_host_run_cumulative import (  # noqa: E402
    _a_dir,
    _run,
    stubbed,
)

__all__ = ["stubbed"]  # re-exported fixture


def test_extraction_refused_with_missing_snapshot_writes_nothing(stubbed):
    a_dir = _a_dir(stubbed)

    rc = _run(stubbed, suites=["extraction"])

    assert rc == 1
    assert not (a_dir / "meta.json").exists()
    assert not (a_dir / "vram.csv").exists()
    assert not (a_dir / "extraction.jsonl").exists()
    assert not (a_dir / "extraction_summary.json").exists()


def test_extraction_refused_with_snapshot_missing_backend_key_writes_nothing(stubbed):
    a_dir = _a_dir(stubbed)
    session_dir = stubbed["results_root"] / "run1" / "session"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "snapshot.json").write_text('{"mist-llm": {"Image": "sha256:aa"}}', encoding="utf-8")

    rc = _run(stubbed, suites=["extraction"])

    assert rc == 1
    assert not (a_dir / "meta.json").exists()
    assert not (a_dir / "vram.csv").exists()


def test_extraction_with_valid_snapshot_passes_preflight_and_invokes_container(stubbed, monkeypatch):
    """Control case: a valid snapshot lets preflight pass, meta.json/vram.csv
    ARE written, and the container argv is built with the resolved image."""
    session_dir = stubbed["results_root"] / "run1" / "session"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "snapshot.json").write_text(
        '{"mist-backend": {"Image": "sha256:bb"}}', encoding="utf-8"
    )

    captured_argv = {}

    def fake_subprocess_run(argv, cwd=None, shell=False, **kwargs):
        captured_argv["argv"] = argv

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(bench_host.subprocess, "run", fake_subprocess_run)

    rc = _run(stubbed, suites=["extraction"])

    assert rc == 0
    a_dir = _a_dir(stubbed)
    assert (a_dir / "meta.json").exists()
    assert (a_dir / "vram.csv").exists()
    assert captured_argv["argv"][0:3] == ["docker", "run", "--rm"]
    assert "sha256:bb" in captured_argv["argv"]
