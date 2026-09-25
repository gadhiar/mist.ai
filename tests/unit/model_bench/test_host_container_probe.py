"""probe_container_state()'s classification of docker inspect's raw output into
running/exited/absent/unknown -- the seam that replaces the S2 defect (`docker_inspect`
raising `DockerError` on ANY non-zero exit, with no timeout, which `wait_for_llama_health`
had no way to tell apart from "the container actually exited").

subprocess.run is stubbed throughout; no real docker.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.model_bench import bench_host  # noqa: E402

CONTAINER = "mist-bench-llm"


class FakeCompletedProcess:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _assert_docker_inspect_state_argv(argv):
    assert argv == ["docker", "inspect", "-f", "{{json .State}}", CONTAINER], argv


def test_probe_classifies_running(monkeypatch):
    def _fake_run(argv, **kwargs):
        _assert_docker_inspect_state_argv(argv)
        return FakeCompletedProcess(0, stdout='{"Status":"running","Running":true}\n')

    monkeypatch.setattr(bench_host.subprocess, "run", _fake_run)
    probe = bench_host.probe_container_state(CONTAINER)
    assert probe.state == "running"


def test_probe_classifies_exited_with_exit_code(monkeypatch):
    def _fake_run(argv, **kwargs):
        _assert_docker_inspect_state_argv(argv)
        return FakeCompletedProcess(0, stdout='{"Status":"exited","ExitCode":1}\n')

    monkeypatch.setattr(bench_host.subprocess, "run", _fake_run)
    probe = bench_host.probe_container_state(CONTAINER)
    assert probe.state == "exited"
    assert probe.exit_code == 1


def test_probe_classifies_dead_as_exited(monkeypatch):
    def _fake_run(argv, **kwargs):
        return FakeCompletedProcess(0, stdout='{"Status":"dead","ExitCode":137}\n')

    monkeypatch.setattr(bench_host.subprocess, "run", _fake_run)
    probe = bench_host.probe_container_state(CONTAINER)
    assert probe.state == "exited"
    assert probe.exit_code == 137


def test_probe_classifies_no_such_object_as_absent(monkeypatch):
    def _fake_run(argv, **kwargs):
        _assert_docker_inspect_state_argv(argv)
        return FakeCompletedProcess(1, stdout="", stderr=f"Error: No such object: {CONTAINER}\n")

    monkeypatch.setattr(bench_host.subprocess, "run", _fake_run)
    probe = bench_host.probe_container_state(CONTAINER)
    assert probe.state == "absent"


def test_probe_classifies_no_such_container_as_absent_case_insensitive(monkeypatch):
    def _fake_run(argv, **kwargs):
        return FakeCompletedProcess(1, stdout="", stderr=f"error: NO SUCH CONTAINER: {CONTAINER}\n")

    monkeypatch.setattr(bench_host.subprocess, "run", _fake_run)
    probe = bench_host.probe_container_state(CONTAINER)
    assert probe.state == "absent"


def test_probe_classifies_empty_stderr_failure_as_unknown(monkeypatch):
    """The exact S2 shape: docker inspect failed with no explanatory stderr at all."""

    def _fake_run(argv, **kwargs):
        _assert_docker_inspect_state_argv(argv)
        return FakeCompletedProcess(1, stdout="", stderr="")

    monkeypatch.setattr(bench_host.subprocess, "run", _fake_run)
    probe = bench_host.probe_container_state(CONTAINER)
    assert probe.state == "unknown"
    assert "exited" not in probe.detail.lower()


def test_probe_classifies_timeout_expired_as_unknown(monkeypatch):
    def _fake_run(argv, **kwargs):
        _assert_docker_inspect_state_argv(argv)
        raise subprocess.TimeoutExpired(cmd=argv, timeout=kwargs.get("timeout"))

    monkeypatch.setattr(bench_host.subprocess, "run", _fake_run)
    probe = bench_host.probe_container_state(CONTAINER, timeout_s=15.0)
    assert probe.state == "unknown"


def test_probe_classifies_missing_docker_binary_as_unknown(monkeypatch):
    def _fake_run(argv, **kwargs):
        raise FileNotFoundError("docker")

    monkeypatch.setattr(bench_host.subprocess, "run", _fake_run)
    probe = bench_host.probe_container_state(CONTAINER)
    assert probe.state == "unknown"


def test_probe_classifies_empty_stdout_as_unknown(monkeypatch):
    def _fake_run(argv, **kwargs):
        return FakeCompletedProcess(0, stdout="", stderr="")

    monkeypatch.setattr(bench_host.subprocess, "run", _fake_run)
    probe = bench_host.probe_container_state(CONTAINER)
    assert probe.state == "unknown"


def test_probe_classifies_garbled_json_as_unknown(monkeypatch):
    def _fake_run(argv, **kwargs):
        return FakeCompletedProcess(0, stdout="{not valid json", stderr="")

    monkeypatch.setattr(bench_host.subprocess, "run", _fake_run)
    probe = bench_host.probe_container_state(CONTAINER)
    assert probe.state == "unknown"


def test_probe_classifies_json_without_status_as_unknown(monkeypatch):
    def _fake_run(argv, **kwargs):
        return FakeCompletedProcess(0, stdout='{"Running":true}\n', stderr="")

    monkeypatch.setattr(bench_host.subprocess, "run", _fake_run)
    probe = bench_host.probe_container_state(CONTAINER)
    assert probe.state == "unknown"


def test_probe_passes_timeout_through_to_subprocess_run(monkeypatch):
    seen = {}

    def _fake_run(argv, **kwargs):
        seen["timeout"] = kwargs.get("timeout")
        return FakeCompletedProcess(0, stdout='{"Status":"running"}\n')

    monkeypatch.setattr(bench_host.subprocess, "run", _fake_run)
    bench_host.probe_container_state(CONTAINER, timeout_s=7.5)
    assert seen["timeout"] == 7.5
