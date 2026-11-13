"""Tests for the CLI: command discovery, exit codes, `init` and `doctor` end to end."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
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
    """A directory holding `docker`, a symlink to `true`, for PATH.

    Nothing written under tmp is executed: the link resolves to the system's `true`, so the
    fixture works where /tmp is mounted noexec.
    """
    true = shutil.which("true")
    if true is None:
        pytest.skip("no `true` on PATH")
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir()
    (bin_dir / "docker").symlink_to(true)
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
    assert "images: not checked (docker missing)" in result.stdout.splitlines()


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


# --- doctor: local images against HEAD ------------------------------------------------------

HEAD = "0123456789abcdef0123456789abcdef01234567"
IMAGE_FAMILIES = {
    "schema": 1,
    "families": {
        "score": {"image": "mist-voice-base", "venv": "/opt/venv-score"},
        "extract": {"image": "mist-voice-extract", "venv": "/opt/venv-extract"},
        "teacher-elevenlabs": {"image": None, "venv": None, "hosted": True},
    },
}
REQUIREMENTS = {
    "requirements-score.txt": "numpy==2.1.0\n",
    "requirements-extract.txt": "torch==2.6.0\n",
}


class FakeProbe:
    """The doctor's git and docker seam: HEAD's sha and each present image's record text."""

    def __init__(self, head: str | None, records: dict[str, str]) -> None:
        self._head = head
        self._records = records
        self.asked: list[str] = []

    def head(self) -> str | None:
        return self._head

    def record(self, tag: str) -> str | None:
        self.asked.append(tag)
        return self._records.get(tag)


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    """A checkout holding `families.json` and the requirements files the records name."""
    path = tmp_path / "checkout"
    (path / "scripts" / "voice_data").mkdir(parents=True)
    (path / "scripts" / "voice_data" / "families.json").write_text(
        json.dumps(IMAGE_FAMILIES), encoding="utf-8"
    )
    (path / "docker" / "voice-data").mkdir(parents=True)
    for name, text in REQUIREMENTS.items():
        (path / "docker" / "voice-data" / name).write_text(text, encoding="utf-8")
    return path


def _record(family: str, git_sha: str, names: tuple[str, ...], **override: str) -> str:
    digests = {
        name: hashlib.sha256(REQUIREMENTS[name].encode("utf-8")).hexdigest() for name in names
    }
    digests.update(override)
    return json.dumps({"family": family, "git_sha": git_sha, "requirements": digests})


def _current_records() -> dict[str, str]:
    return {
        f"mist-voice-base:{HEAD[:12]}": _record("score", HEAD, ("requirements-score.txt",)),
        f"mist-voice-extract:{HEAD[:12]}": _record(
            "extract", HEAD, ("requirements-score.txt", "requirements-extract.txt")
        ),
    }


def test_doctor_images_ok(checkout: Path) -> None:
    probe = FakeProbe(HEAD, _current_records())
    assert doctor.image_lines(probe, checkout) == [
        "image mist-voice-base: ok",
        "image mist-voice-extract: ok",
    ]
    assert probe.asked == [f"mist-voice-base:{HEAD[:12]}", f"mist-voice-extract:{HEAD[:12]}"]


def test_doctor_images_missing(checkout: Path) -> None:
    records = _current_records()
    del records[f"mist-voice-extract:{HEAD[:12]}"]
    assert doctor.image_lines(FakeProbe(HEAD, records), checkout) == [
        "image mist-voice-base: ok",
        "image mist-voice-extract: missing",
    ]


def test_doctor_images_stale(checkout: Path) -> None:
    other = "f" * 40
    records = {
        f"mist-voice-base:{HEAD[:12]}": _record("score", other, ("requirements-score.txt",)),
        f"mist-voice-extract:{HEAD[:12]}": _record(
            "extract",
            HEAD,
            ("requirements-score.txt", "requirements-extract.txt"),
            **{"requirements-extract.txt": "0" * 64, "requirements-gone.txt": "1" * 64},
        ),
    }
    assert doctor.image_lines(FakeProbe(HEAD, records), checkout) == [
        "image mist-voice-base: stale (git sha)",
        "image mist-voice-extract: stale (requirements-extract.txt, requirements-gone.txt)",
    ]


def test_doctor_images_stale_on_an_unreadable_record(checkout: Path) -> None:
    records = _current_records()
    records[f"mist-voice-base:{HEAD[:12]}"] = ""
    records[f"mist-voice-extract:{HEAD[:12]}"] = _record(
        "extract", HEAD, (), **{"../../outside.txt": "0" * 64}
    )
    assert doctor.image_lines(FakeProbe(HEAD, records), checkout) == [
        "image mist-voice-base: stale (no readable record)",
        "image mist-voice-extract: stale (requirements)",
    ]


def test_doctor_images_without_git(checkout: Path) -> None:
    probe = FakeProbe(None, _current_records())
    assert doctor.image_lines(probe, checkout) == ["images: not checked (no git HEAD)"]
    assert probe.asked == []


def test_doctor_image_warnings_keep_the_exit_code(
    private_root: root_mod.Root,
    checkout: Path,
    fake_docker: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    canaries,
    assert_no_canary: Callable[[str], None],
) -> None:
    # Images are built after merge, so a missing or stale one is a warning, not a failure.
    monkeypatch.setenv("PATH", fake_docker + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setattr(doctor, "CHECKOUT", checkout)
    monkeypatch.setattr(doctor, "make_probe", lambda path: FakeProbe(HEAD, {}))
    assert cli.main(["--root", str(private_root.path), "doctor"]) == 0
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert "image mist-voice-base: missing" in lines
    assert "image mist-voice-extract: missing" in lines
    assert_no_canary(captured.out + captured.err)


@pytest.mark.parametrize(
    ("version", "warned"),
    [((3, 11, 12), True), ((3, 11, 13), False), ((3, 12, 0), False), ((3, 10, 20), True)],
)
def test_doctor_warns_on_old_python(version: tuple[int, int, int], warned: bool) -> None:
    warning = doctor.python_warning(version)
    assert (warning is not None) is warned
    if warning is not None:
        assert "3.11.13" in warning


def test_importing_the_entry_module_parses_no_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    # The repository's import walks import every module under scripts/, `__main__` included,
    # with pytest's own argv in place; an unguarded entry call exits 2 there.
    monkeypatch.setattr(sys, "argv", ["walker", "--not-a-voice-data-option"])
    monkeypatch.delitem(sys.modules, "scripts.voice_data.__main__", raising=False)
    module = importlib.import_module("scripts.voice_data.__main__")
    assert callable(module.main)
