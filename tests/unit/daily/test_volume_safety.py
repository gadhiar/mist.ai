"""Tests for `scripts/daily/volume_safety.py`, the guard every instance command passes first.

The module comes from `instance.volume_safety`, the copy `scripts/daily/instance.py` loaded by
file path, so the `DestructiveCommandError` these tests name is the class `instance.py` catches.
Nothing here runs Docker: `subprocess.run` is replaced, or the program run is Python itself.
"""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.daily import instance

volume_safety = instance.volume_safety

REPO_ROOT = Path(__file__).resolve().parents[3]
DAILY_SCRIPTS = REPO_ROOT / "scripts" / "daily"
GUARD = DAILY_SCRIPTS / "volume_safety.py"
NAME = "mist-daily-drill-20261023T093005Z"
COMPOSE = [
    "docker",
    "compose",
    "-p",
    "mist-daily",
    "-f",
    "/placeholder/root/release/daily-2026.10.23/deploy/daily/compose.yml",
    "--env-file",
    "/placeholder/root/daily.env",
]
DRILL_COMPOSE = [
    "docker",
    "compose",
    "-p",
    "mist-daily-drill",
    "-f",
    "/placeholder/root/release/daily-2026.10.23/deploy/daily/restore-drill.yml",
    "--env-file",
    "/placeholder/root/daily.env",
]

FORBIDDEN = {
    "short-v": [*COMPOSE, "rm", "-s", "-f", "-v"],
    "long-volumes": [*COMPOSE, "rm", "--volumes"],
    "long-volumes-value": [*COMPOSE, "rm", "--volumes=true"],
    "short-cluster": [*COMPOSE, "rm", "-fv"],
    "prune": ["docker", "system", "prune"],
    "down": [*COMPOSE, "down"],
    "volume-rm": ["volume", "rm", NAME],
    "docker-volume-rm": ["docker", "volume", "rm", NAME],
    "volume-last": ["docker", "volume"],
    "volume-prune": ["docker", "volume", "prune"],
}

ALLOWED = {
    "volume-inspect": ["docker", "volume", "inspect", "--format", "{{json .Labels}}", NAME],
    "volume-create": [
        "docker",
        "volume",
        "create",
        "--label",
        "org.mist.role=drill",
        "--label",
        "org.mist.instance=daily",
        NAME,
    ],
    "drill-rm": [*DRILL_COMPOSE, "rm", "-s", "-f"],
    "instance-rm": [*COMPOSE, "--profile", "backup", "rm", "-f"],
    "stop": [*COMPOSE, "--profile", "backup", "stop"],
    "up": [*COMPOSE, "up", "-d", "mist-neo4j", "mist-llm", "mist-backend"],
    "ps-volume": ["docker", "ps", "-a", "--filter", f"volume={NAME}", "-q"],
    "ps-label": [
        "docker",
        "ps",
        "-a",
        "--filter",
        "label=com.docker.compose.project=mist-daily-drill",
        "-q",
    ],
    "inspect": ["docker", "inspect", "--format", "{{json .Mounts}}", "c0ffee"],
}


@pytest.mark.parametrize("argv", list(FORBIDDEN.values()), ids=list(FORBIDDEN))
def test_the_guard_refuses_each_volume_deleting_form(argv):
    with pytest.raises(volume_safety.DestructiveCommandError) as caught:
        volume_safety.assert_volume_safe(argv)

    assert caught.value.argv == argv
    assert " ".join(argv) in str(caught.value)


@pytest.mark.parametrize("argv", list(ALLOWED.values()), ids=list(ALLOWED))
def test_the_guard_accepts_each_form_the_instance_runs(argv):
    volume_safety.assert_volume_safe(argv)


def test_subprocess_runner_refuses_before_running_anything(monkeypatch):
    calls = []

    def run(argv, **_kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(instance.subprocess, "run", run)

    with pytest.raises(instance.volume_safety.DestructiveCommandError):
        instance.subprocess_runner(["docker", "volume", "rm", NAME])
    with pytest.raises(instance.volume_safety.DestructiveCommandError):
        instance.subprocess_runner([*COMPOSE, "rm", "-f", "-v"], stdin=b"x")

    assert calls == []


def test_main_turns_a_refusal_into_exit_1_without_a_traceback(tmp_path, monkeypatch, capsys):
    root = tmp_path / "daily"
    root.mkdir()
    marker = {"v": 1, "instance": "daily", "created_at": "2026-10-01T00:00:00Z"}
    (root / ".mist-live-instance").write_text(json.dumps(marker), encoding="utf-8")
    (root / "daily.env").write_text("MIST_RELEASE_TAG=\n", encoding="utf-8")
    monkeypatch.setenv("MIST_LIVE_INSTANCES_FILE", str(tmp_path / "registry"))
    refused = ["docker", "volume", "rm", NAME]

    def runner(argv, *, stdin=None, env=None):
        raise instance.volume_safety.DestructiveCommandError(refused)

    ctx = instance.Context(runner=runner, environ={}, home=tmp_path)

    assert instance.main(["down", "--root", str(root)], ctx) == 1

    err = capsys.readouterr().err
    assert "instance.py down: refused a volume-deleting command:" in err
    assert " ".join(refused) in err
    assert "Traceback" not in err


def run_help(directory: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(directory / "instance.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(directory),
    )


def test_the_cli_runs_with_the_guard_beside_it_and_fails_without_it(tmp_path):
    shutil.copy2(DAILY_SCRIPTS / "instance.py", tmp_path / "instance.py")
    shutil.copy2(GUARD, tmp_path / "volume_safety.py")

    beside = run_help(tmp_path)
    assert beside.returncode == 0, beside.stderr

    (tmp_path / "volume_safety.py").unlink()
    without = run_help(tmp_path)
    assert without.returncode != 0
    assert "volume_safety.py" in without.stderr


def test_the_guard_imports_only_the_standard_library():
    imported = []
    for node in ast.walk(ast.parse(GUARD.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append("." * node.level + (node.module or ""))
    outside = [name for name in imported if name.split(".")[0] not in sys.stdlib_module_names]
    assert outside == []
