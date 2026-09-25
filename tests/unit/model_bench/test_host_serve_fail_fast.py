"""`serve` must fail fast and keep the evidence when the container exits during
the /health wait -- the exact S2 shape: the pinned build rejected a flag and
exited in under a second, but `serve` kept polling /health for the full
900s timeout, then `--rm` deleted the container's stdout/stderr along with it.

Red before green: on the base commit (before this fix), `wait_for_llama_health`
has no way to notice the container already exited, so this test's stub server
(`/health` always unreachable, the container reporting State.Running=False on
every `docker inspect`) makes `serve` spin until its --timeout and then raise
an uncaught `TimeoutError` out of `main()` -- this test fails on that commit.
After the fix, `serve` notices the exited container on the first poll (well
under --timeout), returns non-zero, has written
`<results>/<run>/<arm>/serve_failed_*.log` with `docker logs`' stderr, and has
called `docker rm`.

Docker and /health are stubbed throughout; no docker, no network.
"""

from __future__ import annotations

import sys
import urllib.error
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.model_bench import bench_host  # noqa: E402

FAKE_ARMS_DOC = {
    "schema": 1,
    "common_args": ["--ctx-size", "32768"],
    "sampling": {"gemma": []},
    "thinking_args": {"off": [], "on": ["--reasoning-budget", "{budget}"]},
    "arms": {
        "test-arm": {
            "gguf": "fake/model.gguf",
            "image": "fake:image",
            "family": "gemma",
            "thinking": None,
            "suites": [],
            "harness": None,
        }
    },
}

FAKE_STDERR = "error: invalid argument: --no-mmap\n"


def _fake_docker_inspect(names):
    assert names == [bench_host.BENCH_LLM_CONTAINER]
    return {bench_host.BENCH_LLM_CONTAINER: {"State": {"Running": False, "ExitCode": 1}}}


def _always_unreachable_health(url, timeout=5.0):
    raise urllib.error.URLError("connection refused (stub: server never came up)")


def test_serve_fails_fast_and_saves_logs_when_container_exits_during_health_wait(
    monkeypatch, tmp_path, capsys
):
    rm_calls: list[list[str]] = []

    def _fake_docker_rm(names):
        rm_calls.append(list(names))

    monkeypatch.setattr(bench_host, "load_arms_doc", lambda path=bench_host.ARMS_JSON_PATH: FAKE_ARMS_DOC)
    monkeypatch.setattr(bench_host, "docker_is_running", lambda name: False)
    # Forward-compatible: does not exist on the base commit this test is red
    # against, only after the fix.
    monkeypatch.setattr(bench_host, "docker_container_exists", lambda name: False, raising=False)
    monkeypatch.setattr(bench_host, "docker_run_detached", lambda argv: "fake-container-id")
    monkeypatch.setattr(bench_host, "docker_inspect", _fake_docker_inspect)
    monkeypatch.setattr(bench_host, "http_get_json", _always_unreachable_health)
    monkeypatch.setattr(bench_host, "docker_logs", lambda name: ("", FAKE_STDERR))
    monkeypatch.setattr(bench_host, "docker_rm", _fake_docker_rm, raising=False)
    monkeypatch.setattr(bench_host.time, "sleep", lambda seconds: None)

    results_root = tmp_path / "results"
    models_dir = tmp_path / "models"

    rc = bench_host.main(
        [
            "serve",
            "test-arm",
            "--run",
            "run1",
            "--results-root",
            str(results_root),
            "--models-dir",
            str(models_dir),
            "--timeout",
            "0.05",
        ]
    )

    assert rc != 0, capsys.readouterr()

    arm_dir = results_root / "run1" / "test-arm"
    fail_logs = list(arm_dir.glob("serve_failed_*.log"))
    assert len(fail_logs) == 1, f"expected exactly one serve_failed_*.log, found {fail_logs}"
    log_text = fail_logs[0].read_text(encoding="utf-8")
    assert FAKE_STDERR.strip() in log_text

    assert rm_calls == [[bench_host.BENCH_LLM_CONTAINER]]
