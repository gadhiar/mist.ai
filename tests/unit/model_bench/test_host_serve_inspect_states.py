"""`serve`'s handling of a docker inspect FAILURE during the /health wait, as
distinct from a confirmed exited/absent container (see `test_host_serve_fail_fast.py`
for that case). This is the exact defect host session S2 attempt 2 hit at 0d3aa54:

    `serve c3 --run mb1-fit30 --param ncmoe=30` printed
    `[FAIL] docker inspect ['mist-bench-llm'] failed: ` (empty stderr) 49s into a
    slow model load, while the container was actually `Up 55 seconds
    (health: starting)`. `docker_inspect` raised `DockerError` on ANY non-zero
    exit (no timeout, no way to distinguish "inspect itself failed" from "the
    container exited"), and `cmd_serve` only caught `ContainerExitedError` and
    `TimeoutError`, so the bare `DockerError` escaped to `main()` and was printed
    as a fail-fast verdict on a container that was actually healthy.

(a) A transient inspect failure (empty stderr, twice) must NOT be treated as a
    positive "exited" signal: once /health answers, `serve` returns 0 and never
    calls `docker rm` or `docker stop`.

(c) A SUSTAINED inspect failure (state stays "unknown" for longer than
    `unknown_limit_s`) must fail with a "could not be determined" message --
    never with a claim that the container "exited" -- and must NOT `docker rm`
    the container, since it may still be healthy.

(e) `serve`'s own pre-flight existence check must also refuse, rather than
    proceed, when it cannot determine whether `mist-bench-llm` already exists.

Both (a) and (c) reproduce on 0d3aa54 (before this fix): `wait_for_llama_health`
called `docker_inspect` directly with no timeout and no way to classify a bare
inspect failure as anything but escaping uncaught, so (a) sees `serve` return
non-zero instead of 0, and (c) sees a "docker inspect ... failed" message instead
of "could not be determined" (and often crashes before the sustained-unknown
logic, which does not exist on that commit, is ever reached).

Docker and /health are stubbed throughout; no docker, no network. Tests (a) and
(c) use a fake `time.monotonic`/`time.sleep` so real wall-clock time elapsed is
near zero regardless of the simulated duration.
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


class FakeClock:
    """A monotonic clock that advances by `step` seconds on every read.

    Self-consistent with `time.sleep` being a no-op: `wait_for_llama_health`'s
    deadline math and elapsed-time math both read this same clock, so a large
    `step` compresses a simulated multi-hundred-second wait into a handful of
    real Python calls without needing to count them precisely.
    """

    def __init__(self, step: float = 100.0):
        self._t = 0.0
        self._step = step

    def monotonic(self) -> float:
        self._t += self._step
        return self._t

    def sleep(self, seconds: float) -> None:
        return None


def _common_stubs(monkeypatch, *, docker_rm_calls, docker_stop_calls):
    monkeypatch.setattr(bench_host, "load_arms_doc", lambda path=bench_host.ARMS_JSON_PATH: FAKE_ARMS_DOC)
    monkeypatch.setattr(bench_host, "docker_is_running", lambda name: False)
    monkeypatch.setattr(bench_host, "docker_run_detached", lambda argv: "fake-container-id")
    monkeypatch.setattr(bench_host, "docker_logs", lambda name: ("stdout stub", "stderr stub"))
    monkeypatch.setattr(
        bench_host, "docker_rm", lambda names: docker_rm_calls.append(list(names)), raising=False
    )
    monkeypatch.setattr(
        bench_host, "docker_stop", lambda names: docker_stop_calls.append(list(names)), raising=False
    )


def _serve_argv(results_root, models_dir, *, timeout="5"):
    return [
        "serve",
        "test-arm",
        "--run",
        "run1",
        "--results-root",
        str(results_root),
        "--models-dir",
        str(models_dir),
        "--timeout",
        timeout,
    ]


def test_transient_inspect_failure_does_not_block_a_successful_serve(monkeypatch, tmp_path, capsys):
    """(a): inspect fails twice with empty stderr while /health is not up yet, then
    /health answers. serve must return 0 -- no rm, no stop."""
    rm_calls: list[list[str]] = []
    stop_calls: list[list[str]] = []
    _common_stubs(monkeypatch, docker_rm_calls=rm_calls, docker_stop_calls=stop_calls)

    preflight_probe = bench_host.ContainerProbe("absent", "no mist-bench-llm yet")
    unknown_probe = bench_host.ContainerProbe(
        "unknown", "docker inspect 'mist-bench-llm' failed (exit 1): "
    )
    probe_calls = {"n": 0}

    def _fake_probe(name, *, timeout_s=None):
        probe_calls["n"] += 1
        if probe_calls["n"] == 1:
            return preflight_probe  # cmd_serve's pre-flight existence check
        return unknown_probe  # every poll inside wait_for_llama_health

    health_calls = {"n": 0}

    def _fake_health(url, timeout=5.0):
        health_calls["n"] += 1
        if health_calls["n"] <= 2:
            raise urllib.error.URLError("connection refused (stub: still loading)")
        return {"status": "ok"}

    monkeypatch.setattr(bench_host, "probe_container_state", _fake_probe)
    monkeypatch.setattr(bench_host, "http_get_json", _fake_health)
    monkeypatch.setattr(bench_host.time, "sleep", lambda seconds: None)

    results_root = tmp_path / "results"
    models_dir = tmp_path / "models"
    rc = bench_host.main(_serve_argv(results_root, models_dir))

    out = capsys.readouterr().out
    assert rc == 0, out
    assert health_calls["n"] == 3
    # 2 unknown polls -> 2 [WARN] prints (well under the 30s rate limit, but
    # this is the FIRST warn for this wait, so it always prints).
    assert out.count("[WARN]") >= 1
    assert rm_calls == []
    assert stop_calls == []


def test_sustained_inspect_failure_raises_state_unknown_not_exited(monkeypatch, tmp_path, capsys):
    """(c): docker inspect stays unresolvable past unknown_limit_s (180s default).
    serve must fail with a "could not be determined" message, must NOT say
    "exited", and must NOT docker rm the container."""
    rm_calls: list[list[str]] = []
    stop_calls: list[list[str]] = []
    _common_stubs(monkeypatch, docker_rm_calls=rm_calls, docker_stop_calls=stop_calls)

    preflight_probe = bench_host.ContainerProbe("absent", "no mist-bench-llm yet")
    unknown_probe = bench_host.ContainerProbe("unknown", "docker inspect timed out after 15.0s")
    probe_calls = {"n": 0}

    def _fake_probe(name, *, timeout_s=None):
        probe_calls["n"] += 1
        if probe_calls["n"] == 1:
            return preflight_probe
        return unknown_probe

    def _always_unreachable_health(url, timeout=5.0):
        raise urllib.error.URLError("connection refused (stub: server never came up)")

    clock = FakeClock(step=100.0)
    monkeypatch.setattr(bench_host, "probe_container_state", _fake_probe)
    monkeypatch.setattr(bench_host, "http_get_json", _always_unreachable_health)
    monkeypatch.setattr(bench_host.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(bench_host.time, "sleep", clock.sleep)

    results_root = tmp_path / "results"
    models_dir = tmp_path / "models"
    # timeout is well above the 180s unknown_limit_s (in FakeClock's simulated
    # seconds) so ContainerStateUnknownError, not TimeoutError, fires first.
    rc = bench_host.main(_serve_argv(results_root, models_dir, timeout="100000"))

    out = capsys.readouterr().out
    assert rc == 1, out
    assert "could not be determined" in out
    assert "exited" not in out.lower()
    assert rm_calls == []
    assert stop_calls == []


def test_serve_preflight_refuses_when_existence_cannot_be_determined(monkeypatch, tmp_path, capsys):
    """(e): if the pre-flight probe of mist-bench-llm itself comes back unknown,
    serve must refuse rather than assume the name is free."""
    run_detached_calls: list[list[str]] = []

    monkeypatch.setattr(bench_host, "load_arms_doc", lambda path=bench_host.ARMS_JSON_PATH: FAKE_ARMS_DOC)
    monkeypatch.setattr(bench_host, "docker_is_running", lambda name: False)
    monkeypatch.setattr(
        bench_host,
        "docker_run_detached",
        lambda argv: run_detached_calls.append(argv) or "fake-container-id",
    )
    monkeypatch.setattr(
        bench_host,
        "probe_container_state",
        lambda name, timeout_s=None: bench_host.ContainerProbe(
            "unknown", "docker inspect 'mist-bench-llm' failed (exit 1): "
        ),
    )

    results_root = tmp_path / "results"
    models_dir = tmp_path / "models"
    rc = bench_host.main(_serve_argv(results_root, models_dir))

    out = capsys.readouterr().out
    assert rc == 1, out
    assert "could not determine whether" in out
    assert run_detached_calls == []
