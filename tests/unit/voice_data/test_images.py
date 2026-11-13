"""Tests for `images build`: a committed tree exported with `git archive`, built base first.

Each test builds a small git repository in its tmp directory and passes a fake `DockerOps`, so no
image is built and the real checkout is never exported.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from scripts.voice_data import cli, images
from scripts.voice_data.errors import InputError, JobError

FAMILIES = {
    "schema": 1,
    "families": {
        "score": {"image": "mist-voice-base", "venv": "/opt/venv-score"},
        "extract": {"image": "mist-voice-extract", "venv": "/opt/venv-extract"},
        "teacher-indextts2": {
            "image": "mist-voice-teacher-indextts2",
            "venv": "/opt/venv-teacher-indextts2",
        },
        "teacher-elevenlabs": {"image": None, "venv": None},
    },
}
COMMITTED = {
    "scripts/voice_data/__init__.py": "",
    "scripts/voice_data/families.json": json.dumps(FAMILIES),
    "scripts/model_bench/wer.py": "WER = 1\n",
    "docker/voice-data/base.Dockerfile": "FROM scratch\n",
    "docker/voice-data/extract.Dockerfile": "FROM scratch\n",
    "docker/voice-data/teacher-indextts2.Dockerfile": "FROM scratch\n",
    "docker/backend/Dockerfile": "FROM scratch\n",
    "backend/server.py": "SERVER = 1\n",
    ".gitignore": "scripts/voice_data/ignored.txt\n",
}
EXPORTED = {
    path
    for path in COMMITTED
    if path.startswith(("scripts/voice_data/", "scripts/model_bench/", "docker/voice-data/"))
}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str]:
    """A git repository with one commit, an ignored file and an uncommitted edit on top.

    `GIT_*` variables are cleared first: a test runner may point `GIT_DIR` at the checkout under
    test, which would send every git call here to that repository instead.
    """
    for name in [name for name in os.environ if name.startswith("GIT_")]:
        monkeypatch.delenv(name)
    path = tmp_path / "repo"
    path.mkdir()
    _git(path, "init", "-q")
    for rel, text in COMMITTED.items():
        (path / rel).parent.mkdir(parents=True, exist_ok=True)
        (path / rel).write_text(text, encoding="utf-8")
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", "tree")
    (path / "scripts/voice_data/ignored.txt").write_text("ignored\n", encoding="utf-8")
    (path / "scripts/model_bench/wer.py").write_text("WER = 2\n", encoding="utf-8")
    return path, _git(path, "rev-parse", "HEAD")


class FakeDocker:
    """Records each build argv and the context's files at the time of the build."""

    def __init__(self, fail: bool = False) -> None:
        self.calls: list[list[str]] = []
        self.contexts: list[dict[str, str]] = []
        self.fail = fail

    def capture(self, argv: Sequence[str]) -> str:
        raise AssertionError("images build captures nothing")

    def check(self, argv: Sequence[str]) -> None:
        self.calls.append(list(argv))
        context = Path(argv[-1])
        self.contexts.append(
            {
                p.relative_to(context).as_posix(): p.read_text(encoding="utf-8")
                for p in context.rglob("*")
                if p.is_file()
            }
        )
        if self.fail:
            raise JobError("docker build exited 1")

    def run_logged(self, argv: Sequence[str], log: Path) -> int:
        raise AssertionError("images build runs no job")


def test_context_holds_only_the_committed_three_paths(repo) -> None:
    path, sha = repo
    docker = FakeDocker()
    images.build(path, sha, "extract", docker)
    context = docker.contexts[0]
    assert set(context) == EXPORTED
    assert "scripts/voice_data/ignored.txt" not in context
    assert context["scripts/model_bench/wer.py"] == "WER = 1\n"


def test_all_builds_runtime_then_base_then_each_image_once(repo) -> None:
    path, sha = repo
    docker = FakeDocker()
    tags = images.build(path, sha[:10], "all", docker)
    assert tags == [
        f"mist-voice-runtime:{sha[:12]}",
        f"mist-voice-base:{sha[:12]}",
        f"mist-voice-extract:{sha[:12]}",
        f"mist-voice-teacher-indextts2:{sha[:12]}",
    ]
    context = docker.calls[0][-1]
    assert docker.calls[0] == [
        "docker",
        "build",
        "-f",
        f"{context}/docker/voice-data/base.Dockerfile",
        "--target",
        "runtime",
        "-t",
        f"mist-voice-runtime:{sha[:12]}",
        context,
    ]
    assert docker.calls[1] == [
        "docker",
        "build",
        "-f",
        f"{context}/docker/voice-data/base.Dockerfile",
        "--build-arg",
        f"GIT_SHA={sha}",
        "-t",
        f"mist-voice-base:{sha[:12]}",
        context,
    ]
    assert docker.calls[2] == [
        "docker",
        "build",
        "-f",
        f"{context}/docker/voice-data/extract.Dockerfile",
        "--build-arg",
        f"GIT_SHA={sha}",
        "--build-arg",
        f"BASE_IMAGE=mist-voice-runtime:{sha[:12]}",
        "-t",
        f"mist-voice-extract:{sha[:12]}",
        context,
    ]
    assert [Path(call[3]).name for call in docker.calls] == [
        "base.Dockerfile",
        "base.Dockerfile",
        "extract.Dockerfile",
        "teacher-indextts2.Dockerfile",
    ]


def test_only_family_builds_get_the_base_image_arg(repo) -> None:
    # Family venvs build on the code-free runtime stage, so a new sha reuses their cached layers;
    # the runtime and base builds take no BASE_IMAGE.
    path, sha = repo
    docker = FakeDocker()
    images.build(path, sha, "all", docker)
    with_arg = [
        call[call.index("-t") + 1]
        for call in docker.calls
        if f"BASE_IMAGE=mist-voice-runtime:{sha[:12]}" in call
    ]
    assert with_arg == [
        f"mist-voice-extract:{sha[:12]}",
        f"mist-voice-teacher-indextts2:{sha[:12]}",
    ]
    assert [call for call in docker.calls if "--target" in call] == [docker.calls[0]]
    assert not any(arg.startswith("BASE_IMAGE=") for arg in docker.calls[0] + docker.calls[1])


def test_one_family_builds_runtime_and_base_then_its_image(repo) -> None:
    path, sha = repo
    assert images.build(path, sha, "teacher-indextts2", FakeDocker()) == [
        f"mist-voice-runtime:{sha[:12]}",
        f"mist-voice-base:{sha[:12]}",
        f"mist-voice-teacher-indextts2:{sha[:12]}",
    ]


def test_score_family_maps_to_base_and_builds_it_once(repo) -> None:
    path, sha = repo
    assert images.build(path, sha, "score", FakeDocker()) == [
        f"mist-voice-runtime:{sha[:12]}",
        f"mist-voice-base:{sha[:12]}",
    ]


def test_the_temporary_context_is_removed_after_a_failed_build(repo) -> None:
    path, sha = repo
    docker = FakeDocker(fail=True)
    with pytest.raises(JobError):
        images.build(path, sha, "extract", docker)
    assert not Path(docker.calls[0][-1]).exists()


def test_the_temporary_context_is_removed_after_a_build(repo) -> None:
    path, sha = repo
    docker = FakeDocker()
    images.build(path, sha, "extract", docker)
    assert not Path(docker.calls[0][-1]).exists()


def test_unknown_family_exits_2(repo) -> None:
    path, sha = repo
    docker = FakeDocker()
    with pytest.raises(InputError, match="nosuch") as caught:
        images.build(path, sha, "nosuch", docker)
    assert cli.exit_code(caught.value) == 2
    assert docker.calls == []


def test_unknown_sha_exits_2_through_the_cli(repo, capsys) -> None:
    path, _ = repo
    argv = ["images", "build", "--family", "all", "--sha", "0123456789ab", "--checkout", str(path)]
    assert cli.main(argv) == 2
    assert "not a commit" in capsys.readouterr().err
