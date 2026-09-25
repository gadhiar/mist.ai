"""End-to-end `plan` and `selftest` invocation via main(), matching the
done-when acceptance command
(`python -m scripts.model_bench.bench_host plan` / `selftest`).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.model_bench.bench_host import main  # noqa: E402


def test_plan_ok_with_no_env_set(capsys, monkeypatch):
    for var in ("MODELS_DIR", "MODEL_BENCH_LAYOUT_DIR", "MODEL_BENCH_RESULTS_ROOT"):
        monkeypatch.delenv(var, raising=False)
    rc = main(["plan"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "plan ok" in out
    assert "[WARN]" in out


def test_plan_host_checks_exits_nonzero_when_a_host_check_fails(capsys, monkeypatch):
    # A scripted gate reads the exit code: a failed host check (e.g. an arm flag the pinned
    # build rejects, the defect that blocked S2) must not end in "plan ok" and exit 0.
    from scripts.model_bench import bench_host

    monkeypatch.setattr(bench_host, "_run_host_checks", lambda: False)
    rc = main(["plan", "--host-checks"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "plan ok" not in out


def test_plan_host_checks_exits_zero_when_all_host_checks_pass(capsys, monkeypatch):
    from scripts.model_bench import bench_host

    monkeypatch.setattr(bench_host, "_run_host_checks", lambda: True)
    rc = main(["plan", "--host-checks"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "plan ok" in out


def test_selftest_ok(capsys):
    rc = main(["selftest"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "selftest ok" in out
