"""`serve` must fail fast and keep the evidence when the container exits during
the /health wait -- the exact S2 shape: the pinned build rejected a flag and
exited in under a second, but `serve` kept polling /health for the full
900s timeout, then `--rm` deleted the container's stdout/stderr along with it.

This is scenario (b) from the fix-serve-inspect brief: a REAL exited state
(`State.Status == "exited"`, `ExitCode == 1`), which must still fail fast even
after the fix that stops the driver from treating an inspect FAILURE (empty
stderr, a timeout, ...) as a positive "exited" signal -- see
`test_host_serve_inspect_states.py` for that distinction (scenarios a/c).

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


def _fake_probe_exited(name, *, timeout_s=None):
    assert name == bench_host.BENCH_LLM_CONTAINER
    return bench_host.ContainerProbe("exited", "'mist-bench-llm' state is 'exited'", exit_code=1)


def _always_unreachable_health(url, timeout=5.0):
    raise urllib.error.URLError("connection refused (stub: server never came up)")


def test_serve_fails_fast_and_saves_logs_when_container_exits_during_health_wait(
    monkeypatch, tmp_path, capsys
):
    rm_calls: list[list[str]] = []

    def _fake_docker_rm(names):
        rm_calls.append(list(names))

    probe_calls = {"n": 0}

    def _preflight_then_exited(name, *, timeout_s=None):
        # First call is cmd_serve's own pre-flight existence check (must see
        # "absent" so serve proceeds); every call after that is from inside
        # wait_for_llama_health's poll loop, where the container has since
        # actually exited.
        probe_calls["n"] += 1
        if probe_calls["n"] == 1:
            return bench_host.ContainerProbe("absent", "no mist-bench-llm yet")
        return _fake_probe_exited(name, timeout_s=timeout_s)

    monkeypatch.setattr(bench_host, "load_arms_doc", lambda path=bench_host.ARMS_JSON_PATH: FAKE_ARMS_DOC)
    monkeypatch.setattr(bench_host, "docker_is_running", lambda name: False)
    monkeypatch.setattr(bench_host, "docker_run_detached", lambda argv: "fake-container-id")
    monkeypatch.setattr(bench_host, "probe_container_state", _preflight_then_exited)
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
