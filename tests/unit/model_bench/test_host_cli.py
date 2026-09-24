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


def test_selftest_ok(capsys):
    rc = main(["selftest"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "selftest ok" in out
