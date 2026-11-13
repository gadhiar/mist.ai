"""Tests for the CLI: command discovery, exit codes, `init` and `doctor` end to end."""

from __future__ import annotations

import argparse
import os
import stat
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from scripts.voice_data import cli, errors
from scripts.voice_data import root as root_mod
from scripts.voice_data.commands import doctor
from scripts.voice_data.commands import init as init_cmd

REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_NAMES = (
    "MIST_VOICE_PRIVATE_ROOT",
    "HF_TOKEN",
    "ELEVENLABS_API_KEY",
    "MIST_VOICE_TEXT_LLM_URL",
    "MIST_VOICE_TEXT_LLM_MODEL",
    "MIST_VOICE_TEXT_LLM_KEY",
    "MIST_VOICE_REGISTRY",
)


def _run(args: list[str], env_extra: dict[str, str] | None = None, path_prefix: str | None = None):
    env = {k: v for k, v in os.environ.items() if k not in ENV_NAMES}
    env.update(env_extra or {})
    if path_prefix is not None:
        env["PATH"] = path_prefix + os.pathsep + env.get("PATH", "")
    return subprocess.run(
        [sys.executable, "-m", "scripts.voice_data", *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


@pytest.fixture
def fake_docker(tmp_path: Path) -> str:
    """A directory holding an executable named `docker`, for PATH."""
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    docker.chmod(docker.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return str(bin_dir)


@pytest.fixture
def empty_path(tmp_path: Path) -> str:
    """A PATH entry with no docker, used as the whole PATH."""
    bin_dir = tmp_path / "empty-bin"
    bin_dir.mkdir()
    return str(bin_dir)


# --- discovery and exit codes ----------------------------------------------------------------


def test_commands_are_discovered_in_sorted_order() -> None:
    names = cli.command_names()
    assert names == sorted(names)
    assert {"init", "doctor"} <= set(names)


def test_help_lists_discovered_commands() -> None:
    result = _run(["--help"])
    assert result.returncode == 0
    assert "init" in result.stdout
    assert "doctor" in result.stdout


def test_no_command_is_a_usage_refusal() -> None:
    result = _run([])
    assert result.returncode == 2


def test_unknown_command_is_a_usage_refusal() -> None:
    result = _run(["no-such-command"])
    assert result.returncode == 2


@pytest.mark.parametrize(
    ("exc", "code"),
    [
        (errors.RootError, 2),
        (errors.InputError, 2),
        (errors.GateError, 3),
        (errors.JobError, 4),
        (errors.BudgetError, 5),
    ],
)
def test_errors_map_to_exit_codes(
    exc: type[errors.VoiceDataError],
    code: int,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    assert issubclass(exc, errors.VoiceDataError)
    assert issubclass(exc, RuntimeError)

    def raising(args: argparse.Namespace) -> int:
        raise exc("refused for a test reason")

    monkeypatch.setattr(init_cmd, "run", raising)
    assert cli.main(["--root", str(tmp_path / "r"), "init", "--kind", "private"]) == code
    assert "refused for a test reason" in capsys.readouterr().err


# --- init ------------------------------------------------------------------------------------


def test_init_end_to_end_with_root_option(tmp_path: Path) -> None:
    target = tmp_path / "root"
    result = _run(["--root", str(target), "init", "--kind", "private"])
    assert result.returncode == 0, result.stderr
    root = root_mod.open_root(str(target), environ={})
    assert root.kind == "private"
    assert str(tmp_path) not in result.stdout + result.stderr


def test_init_end_to_end_with_environment(tmp_path: Path) -> None:
    target = tmp_path / "root"
    result = _run(["init", "--kind", "rehearsal"], {"MIST_VOICE_PRIVATE_ROOT": str(target)})
    assert result.returncode == 0, result.stderr
    assert root_mod.open_root(str(target), environ={}).kind == "rehearsal"


def test_init_without_root_exits_2(tmp_path: Path) -> None:
    result = _run(["init", "--kind", "private"])
    assert result.returncode == 2
    assert "MIST_VOICE_PRIVATE_ROOT" in result.stderr


def test_init_refusal_names_no_entry(
    tmp_path: Path, canaries, assert_no_canary: Callable[[str], None]
) -> None:
    target = tmp_path / "busy"
    target.mkdir()
    (target / canaries.source_name).write_bytes(b"x")
    result = _run(["--root", str(target), "init", "--kind", "private"])
    assert result.returncode == 2
    assert_no_canary(result.stdout + result.stderr)
    assert not (target / root_mod.MARKER).exists()


def test_init_inside_git_exits_2(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    result = _run(["--root", str(tmp_path / "root"), "init", "--kind", "private"])
    assert result.returncode == 2
    assert ".git" in result.stderr


def test_init_requires_kind(tmp_path: Path) -> None:
    result = _run(["--root", str(tmp_path / "root"), "init"])
    assert result.returncode == 2


# --- doctor ----------------------------------------------------------------------------------


def test_doctor_reports_env_names_never_values(
    private_root: root_mod.Root,
    fake_docker: str,
    canaries,
    assert_no_canary: Callable[[str], None],
) -> None:
    env = {name: canaries.key for name in ("HF_TOKEN", "ELEVENLABS_API_KEY")}
    env["MIST_VOICE_TEXT_LLM_KEY"] = canaries.key
    env["MIST_VOICE_PRIVATE_ROOT"] = str(private_root.path)
    result = _run(["doctor"], env, path_prefix=fake_docker)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert_no_canary(output)
    assert str(private_root.path) not in output
    lines = result.stdout.splitlines()
    for name in ENV_NAMES:
        expected = "set" if name in env else "unset"
        assert f"{name}: {expected}" in lines, name
    assert "docker: found" in lines
    assert "root: marker ok (kind private)" in lines
    assert any(line.startswith("disk free: ") and line.endswith(" GiB") for line in lines)


def test_doctor_fails_without_docker(private_root: root_mod.Root, empty_path: str) -> None:
    env = {"MIST_VOICE_PRIVATE_ROOT": str(private_root.path), "PATH": empty_path}
    result = _run(["doctor"], env)
    assert result.returncode == 2
    assert "docker: missing" in result.stdout.splitlines()


def test_doctor_fails_without_marker(tmp_path: Path, fake_docker: str) -> None:
    bare = tmp_path / "bare"
    bare.mkdir()
    result = _run(["--root", str(bare), "doctor"], path_prefix=fake_docker)
    assert result.returncode == 2
    assert any(line.startswith("root: refused (no marker") for line in result.stdout.splitlines())


def test_doctor_reports_unset_root(fake_docker: str) -> None:
    result = _run(["doctor"], path_prefix=fake_docker)
    assert result.returncode == 2
    lines = result.stdout.splitlines()
    assert "MIST_VOICE_PRIVATE_ROOT: unset" in lines
    assert "root: unset" in lines


@pytest.mark.parametrize(
    ("version", "warned"),
    [((3, 11, 12), True), ((3, 11, 13), False), ((3, 12, 0), False), ((3, 10, 20), True)],
)
def test_doctor_warns_on_old_python(version: tuple[int, int, int], warned: bool) -> None:
    warning = doctor.python_warning(version)
    assert (warning is not None) is warned
    if warning is not None:
        assert "3.11.13" in warning
