"""Tests for `scripts/daily/instance.py`: init, retire and release-build.

Every Docker and git call goes through the injected runner, so no test here needs Docker, git
or the real home directory. The registry is pointed at `tmp_path` for every test.
"""

from __future__ import annotations

import ast
import io
import json
import os
import re
import subprocess
import sys
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from scripts.daily import instance

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "daily" / "instance.py"
ENV_EXAMPLE = REPO_ROOT / "deploy" / "daily" / "daily.env.example"
EVAL_ISOLATION = REPO_ROOT / "backend" / "knowledge" / "eval_isolation.py"

FIXED_NOW = datetime(2026, 10, 23, 9, 30, 5, tzinfo=UTC)
FIXED_STAMP = "2026-10-23T09:30:05Z"
SHA = "0123456789abcdef0123456789abcdef01234567"
IMAGE_ID = "sha256:" + "f" * 64
TAG = "daily-2026.10.23"
REPO_ARG = str(REPO_ROOT)
EIGHT_DIRS = ("state", "mist-memory", "logs", "ops", "models", "release", "retired", "bin")
MARKER = ".mist-live-instance"


class FakeRunner:
    """Answers each exact argv with a canned result and records every call.

    An argv with no canned answer fails the test, so every Docker and git call a command makes
    has to be one the test expected.
    """

    def __init__(self, responses: dict[tuple[str, ...], instance.CommandResult] | None = None):
        self.responses = dict(responses or {})
        self.calls: list[list[str]] = []
        self.stdin: list[bytes | None] = []

    def __call__(self, argv, *, stdin=None):
        self.calls.append(list(argv))
        self.stdin.append(stdin)
        key = tuple(argv)
        if key not in self.responses:
            raise AssertionError(f"unexpected command: {list(argv)}")
        return self.responses[key]


def ok(stdout: bytes = b"") -> instance.CommandResult:
    return instance.CommandResult(returncode=0, stdout=stdout, stderr=b"")


def fail(code: int = 1, stderr: bytes = b"boom") -> instance.CommandResult:
    return instance.CommandResult(returncode=code, stdout=b"", stderr=stderr)


@pytest.fixture(autouse=True)
def registry(tmp_path, monkeypatch) -> Path:
    """Point the registry at a scratch file so no test can write `~/.mist/live-instances`."""
    path = tmp_path / "registry" / "live-instances"
    monkeypatch.setenv("MIST_LIVE_INSTANCES_FILE", str(path))
    return path


@pytest.fixture
def home(tmp_path) -> Path:
    path = tmp_path / "home"
    path.mkdir()
    return path


@pytest.fixture
def root(tmp_path) -> Path:
    return (tmp_path / "instances" / "daily").resolve()


@pytest.fixture
def make_ctx(home):
    def _make(runner=None, script_path: Path | None = None) -> instance.Context:
        return instance.Context(
            runner=runner if runner is not None else FakeRunner(),
            clock=lambda: FIXED_NOW,
            home=home,
            script_path=script_path if script_path is not None else SCRIPT,
        )

    return _make


def run(ctx: instance.Context, *argv: str) -> int:
    return instance.main(list(argv), ctx)


def all_files(base: Path) -> dict[str, bytes]:
    return {
        p.relative_to(base).as_posix(): p.read_bytes()
        for p in sorted(base.rglob("*"))
        if p.is_file()
    }


def init_root(make_ctx, root: Path, *extra: str) -> None:
    assert run(make_ctx(), "init", "--root", str(root), *extra) == 0


# --- Import rule and runner --------------------------------------------------------------------


def disallowed_imports(source: str, directory: Path) -> list[str]:
    """Names of imports that are neither standard library nor a `.py` file in `directory`."""
    bad: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                bad.append("." * node.level + (node.module or ""))
                continue
            names = [node.module or ""]
        else:
            continue
        for name in names:
            top = name.split(".")[0]
            if top in sys.stdlib_module_names:
                continue
            if "." not in name and (directory / f"{top}.py").is_file():
                continue
            bad.append(name)
    return bad


def test_instance_imports_only_stdlib_or_sibling_modules():
    source = SCRIPT.read_text(encoding="utf-8")
    assert disallowed_imports(source, SCRIPT.parent) == []


def test_import_rule_admits_a_sibling_module_and_nothing_else(tmp_path):
    (tmp_path / "gamewatch.py").write_text("", encoding="utf-8")
    assert disallowed_imports("import gamewatch\nimport json\n", tmp_path) == []
    assert disallowed_imports("import backend.knowledge\n", tmp_path) == ["backend.knowledge"]
    assert disallowed_imports("from scripts.backup import dump\n", tmp_path) == ["scripts.backup"]
    assert disallowed_imports("import yaml\n", tmp_path) == ["yaml"]
    assert disallowed_imports("from . import gamewatch\n", tmp_path) == ["."]


def test_marker_and_registry_names_match_eval_isolation():
    """`instance.py` spells the names itself; this pins them to the guard's constants."""
    text = EVAL_ISOLATION.read_text(encoding="utf-8")
    marker = re.search(r'^LIVE_INSTANCE_MARKER\s*=\s*"([^"]+)"', text, re.M)
    env = re.search(r'^LIVE_INSTANCES_FILE_ENV\s*=\s*"([^"]+)"', text, re.M)
    assert marker is not None, "eval_isolation.py no longer defines LIVE_INSTANCE_MARKER"
    assert env is not None, "eval_isolation.py no longer defines LIVE_INSTANCES_FILE_ENV"
    assert instance.LIVE_INSTANCE_MARKER == marker.group(1) == MARKER
    assert instance.LIVE_INSTANCES_FILE_ENV == env.group(1) == "MIST_LIVE_INSTANCES_FILE"


def test_runs_as_a_script_by_path():
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(SCRIPT.parent.parent.parent.parent),
    )
    assert completed.returncode == 0, completed.stderr
    for command in ("init", "retire", "release-build"):
        assert command in completed.stdout


def test_script_by_path_initializes_a_root(tmp_path, registry):
    target = tmp_path / "by-path"
    env = dict(os.environ, MIST_LIVE_INSTANCES_FILE=str(registry))
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "init", "--root", str(target)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        cwd=str(tmp_path),
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads((target / MARKER).read_text(encoding="utf-8"))["instance"] == "daily"
    assert (target / "daily.env").is_file()


def test_subprocess_runner_returns_code_output_and_stdin():
    result = instance.subprocess_runner(
        [
            sys.executable,
            "-c",
            "import sys; d = sys.stdin.read(); sys.stdout.write(d.upper()); sys.exit(3)",
        ],
        stdin=b"abc",
    )
    assert result.returncode == 3
    assert result.stdout == b"ABC"


def test_subprocess_runner_reports_a_missing_program_as_127():
    result = instance.subprocess_runner(["mist-no-such-program-xyz"])
    assert result.returncode == 127
    assert b"mist-no-such-program-xyz" in result.stderr


def test_commands_call_no_subprocess_outside_the_runner(make_ctx, root, monkeypatch):
    """Every Docker and git call goes through `Context.runner`, never `subprocess` directly."""

    def forbidden(*args, **kwargs):
        raise AssertionError(f"direct subprocess call: {args}")

    monkeypatch.setattr(instance.subprocess, "run", forbidden)
    monkeypatch.setattr(instance.subprocess, "Popen", forbidden)
    init_root(make_ctx, root)
    runner = FakeRunner(release_responses())
    assert run(make_ctx(runner), "release-build", TAG, "--root", str(root), "--repo", REPO_ARG) == 0
    retire_runner = FakeRunner({tuple(compose_ps("daily")): ok()})
    assert run(make_ctx(retire_runner), "retire", "--root", str(root)) == 0
    assert retire_runner.calls == [compose_ps("daily")]


# --- init ------------------------------------------------------------------------------------


def read_marker(path: Path) -> dict:
    return json.loads((path / MARKER).read_text(encoding="utf-8"))


def test_init_defaults_instance_to_daily(make_ctx, root):
    assert run(make_ctx(), "init", "--root", str(root)) == 0
    assert read_marker(root)["instance"] == "daily"


def test_init_records_a_rehearsal_instance(make_ctx, root):
    assert run(make_ctx(), "init", "--root", str(root), "--instance", "rehearsal") == 0
    assert read_marker(root)["instance"] == "rehearsal"


def test_init_rejects_an_unknown_instance_name(make_ctx, root):
    with pytest.raises(SystemExit) as excinfo:
        run(make_ctx(), "init", "--root", str(root), "--instance", "prod")
    assert excinfo.value.code == 2
    assert not root.exists()


def test_init_creates_the_layout_markers_and_registry_line(make_ctx, root, registry):
    assert run(make_ctx(), "init", "--root", str(root)) == 0
    for name in EIGHT_DIRS:
        assert (root / name).is_dir(), name
    expected = {"v": 1, "instance": "daily", "created_at": FIXED_STAMP}
    for directory in (root, root / "state", root / "mist-memory"):
        assert read_marker(directory) == expected
    lines = registry.read_text(encoding="utf-8").splitlines()
    assert lines == [str(root)]
    assert Path(lines[0]).is_absolute()


@pytest.mark.parametrize("spelling", ["relative", "tilde"])
def test_registry_line_is_resolved_and_retire_removes_it(
    make_ctx, tmp_path, home, registry, monkeypatch, spelling
):
    """The guard fails closed on a relative or `~` line, so init writes the resolved path."""
    monkeypatch.chdir(tmp_path)
    if spelling == "relative":
        given, expected = "rel-root", (tmp_path / "rel-root").resolve()
    else:
        given, expected = "~/inst", (home / "inst").resolve()
    assert run(make_ctx(), "init", "--root", given) == 0
    lines = registry.read_text(encoding="utf-8").splitlines()
    assert lines == [str(expected)]
    assert Path(lines[0]).is_absolute()
    assert (expected / MARKER).is_file()

    runner = FakeRunner({tuple(compose_ps("daily")): ok()})
    assert run(make_ctx(runner), "retire", "--root", given) == 0
    assert registry.read_text(encoding="utf-8").splitlines() == []
    assert not (expected / MARKER).exists()


def test_retire_matches_a_registry_line_through_a_symlink(make_ctx, tmp_path, registry):
    real = (tmp_path / "real-root").resolve()
    link = tmp_path / "link-root"
    try:
        link.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are not available here")
    init_root(make_ctx, real)
    registry.write_text(str(link) + "\n", encoding="utf-8")
    runner = FakeRunner({tuple(compose_ps("daily")): ok()})
    assert run(make_ctx(runner), "retire", "--root", str(real)) == 0
    assert registry.read_text(encoding="utf-8").splitlines() == []


def test_retire_keeps_a_relative_registry_line_it_did_not_write(make_ctx, root, registry):
    init_root(make_ctx, root)
    with registry.open("a", encoding="utf-8") as handle:
        handle.write("relative/line\n")
    runner = FakeRunner({tuple(compose_ps("daily")): ok()})
    assert run(make_ctx(runner), "retire", "--root", str(root)) == 0
    assert registry.read_text(encoding="utf-8").splitlines() == ["relative/line"]


def test_init_appends_to_an_existing_registry(make_ctx, root, registry, tmp_path):
    other = str((tmp_path / "other-root").resolve())
    registry.parent.mkdir(parents=True)
    registry.write_text(other + "\n\n", encoding="utf-8")
    assert run(make_ctx(), "init", "--root", str(root)) == 0
    lines = [ln for ln in registry.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert lines == [other, str(root)]


def test_init_twice_keeps_one_registry_line_and_the_first_marker(make_ctx, root, registry, home):
    assert run(make_ctx(), "init", "--root", str(root)) == 0
    later = instance.Context(
        runner=FakeRunner(),
        clock=lambda: datetime(2027, 1, 1, tzinfo=UTC),
        home=home,
        script_path=SCRIPT,
    )
    assert instance.main(["init", "--root", str(root)], later) == 0
    assert registry.read_text(encoding="utf-8").splitlines() == [str(root)]
    assert read_marker(root)["created_at"] == FIXED_STAMP
    assert read_marker(root / "mist-memory")["created_at"] == FIXED_STAMP


def test_init_refuses_a_root_marked_for_another_instance(make_ctx, root, registry):
    init_root(make_ctx, root, "--instance", "rehearsal")
    before = all_files(root)
    assert run(make_ctx(), "init", "--root", str(root), "--instance", "daily") == 2
    assert all_files(root) == before


def test_init_registry_defaults_to_home_mist_live_instances(make_ctx, root, home, monkeypatch):
    monkeypatch.delenv("MIST_LIVE_INSTANCES_FILE")
    assert run(make_ctx(), "init", "--root", str(root)) == 0
    default = home / ".mist" / "live-instances"
    assert default.read_text(encoding="utf-8").splitlines() == [str(root)]


def test_init_treats_an_empty_registry_variable_as_unset(make_ctx, root, home, monkeypatch):
    monkeypatch.setenv("MIST_LIVE_INSTANCES_FILE", "  ")
    assert run(make_ctx(), "init", "--root", str(root)) == 0
    assert (home / ".mist" / "live-instances").is_file()


@pytest.mark.parametrize("git_kind", ["dir", "file"])
def test_init_refuses_a_root_inside_a_git_work_tree(make_ctx, tmp_path, registry, git_kind):
    repo = tmp_path / "repo"
    repo.mkdir()
    if git_kind == "dir":
        (repo / ".git").mkdir()
    else:
        (repo / ".git").write_text("gitdir: /elsewhere\n", encoding="utf-8")
    target = repo / "deploy" / "instance"
    assert run(make_ctx(), "init", "--root", str(target)) == 2
    assert not (repo / "deploy").exists()
    assert not registry.exists()


def test_init_refuses_a_drive_root(make_ctx, tmp_path, registry):
    drive_root = Path(tmp_path.anchor)
    assert run(make_ctx(), "init", "--root", str(drive_root)) == 2
    assert not (drive_root / MARKER).exists()
    assert not registry.exists()


def test_init_refuses_the_home_directory(make_ctx, home, registry):
    assert run(make_ctx(), "init", "--root", str(home)) == 2
    assert list(home.iterdir()) == []
    assert not registry.exists()


def test_init_writes_daily_env_from_the_example(make_ctx, root):
    assert run(make_ctx(), "init", "--root", str(root)) == 0
    written = (root / "daily.env").read_text(encoding="utf-8").splitlines()
    example = ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
    filled = {
        "MIST_DAILY_ROOT": root.as_posix(),
        "MIST_INSTANCE_NAME": "daily",
        "MODELS_DIR": (root / "models").as_posix(),
    }
    assert len(written) == len(example)
    for got, want in zip(written, example):
        key = want.split("=", 1)[0]
        if key in filled:
            assert got == f"{key}={filled[key]}"
        else:
            assert got == want


def test_init_keeps_an_existing_daily_env(make_ctx, root):
    root.mkdir(parents=True)
    (root / "daily.env").write_text("NEO4J_PASSWORD=kept\n", encoding="utf-8")
    assert run(make_ctx(), "init", "--root", str(root)) == 0
    assert (root / "daily.env").read_text(encoding="utf-8") == "NEO4J_PASSWORD=kept\n"


def test_init_refuses_when_the_env_example_is_missing(make_ctx, root, registry, tmp_path):
    lone = tmp_path / "lone" / "a" / "b" / "instance.py"
    assert run(make_ctx(script_path=lone), "init", "--root", str(root)) == 2
    assert not root.exists()
    assert not registry.exists()


def make_seed_fixture(base: Path) -> Path:
    fixture = base / "dev-instance"
    (fixture / "vault" / "users").mkdir(parents=True)
    (fixture / "vault" / "MIST.md").write_text("mist\n", encoding="utf-8")
    (fixture / "vault" / "users" / "dev-user.md").write_text("dev\n", encoding="utf-8")
    (fixture / "seed").mkdir()
    (fixture / "seed" / "mist.md").write_text("seed\n", encoding="utf-8")
    (fixture / "README.md").write_text("not copied\n", encoding="utf-8")
    return fixture


def test_init_seed_copies_vault_contents_and_seed_directory(make_ctx, root, tmp_path):
    fixture = make_seed_fixture(tmp_path)
    assert run(make_ctx(), "init", "--root", str(root), "--seed", str(fixture)) == 0
    memory = root / "mist-memory"
    assert all_files(memory) == {
        MARKER: (memory / MARKER).read_bytes(),
        "MIST.md": b"mist\n",
        "users/dev-user.md": b"dev\n",
        "seed/mist.md": b"seed\n",
    }


def test_init_seed_runs_on_a_root_whose_memory_holds_only_its_marker(make_ctx, root, tmp_path):
    init_root(make_ctx, root)
    fixture = make_seed_fixture(tmp_path)
    assert run(make_ctx(), "init", "--root", str(root), "--seed", str(fixture)) == 0
    assert (root / "mist-memory" / "seed" / "mist.md").is_file()


def test_init_seed_refuses_a_non_empty_mist_memory(make_ctx, root, tmp_path, registry):
    fixture = make_seed_fixture(tmp_path)
    (root / "mist-memory").mkdir(parents=True)
    (root / "mist-memory" / "note.md").write_text("mine\n", encoding="utf-8")
    assert run(make_ctx(), "init", "--root", str(root), "--seed", str(fixture)) == 2
    assert all_files(root) == {"mist-memory/note.md": b"mine\n"}
    assert not registry.exists()


def test_init_seed_refuses_a_directory_without_vault_and_seed(make_ctx, root, tmp_path, registry):
    fixture = tmp_path / "empty-fixture"
    fixture.mkdir()
    assert run(make_ctx(), "init", "--root", str(root), "--seed", str(fixture)) == 2
    assert not root.exists()
    assert not registry.exists()


# --- retire ----------------------------------------------------------------------------------


def compose_ps(name: str) -> list[str]:
    return ["docker", "compose", "-p", f"mist-{name}", "ps", "-q"]


def populate(root: Path) -> None:
    (root / "daily.env").write_text("NEO4J_PASSWORD=x\n", encoding="utf-8")
    (root / "state" / "event_store.db").write_bytes(b"db")
    (root / "mist-memory" / "notes.md").write_text("n\n", encoding="utf-8")
    (root / "ops" / "release-x.json").write_text("{}", encoding="utf-8")
    (root / "release" / TAG / "deploy").mkdir(parents=True)
    (root / "release" / TAG / "deploy" / "compose.yml").write_text("s\n", encoding="utf-8")


def test_retire_refuses_while_compose_lists_a_container(make_ctx, root, registry):
    init_root(make_ctx, root)
    before = all_files(root)
    runner = FakeRunner({tuple(compose_ps("daily")): ok(b"3f2a1b\n")})
    assert run(make_ctx(runner), "retire", "--root", str(root)) == 2
    assert runner.calls == [compose_ps("daily")]
    assert all_files(root) == before
    assert registry.read_text(encoding="utf-8").splitlines() == [str(root)]


def test_retire_uses_the_marker_instance_name(make_ctx, root):
    init_root(make_ctx, root, "--instance", "rehearsal")
    runner = FakeRunner({tuple(compose_ps("rehearsal")): ok()})
    assert run(make_ctx(runner), "retire", "--root", str(root)) == 0
    assert runner.calls == [compose_ps("rehearsal")]


def test_retire_refuses_when_compose_ps_fails(make_ctx, root, registry):
    init_root(make_ctx, root)
    runner = FakeRunner({tuple(compose_ps("daily")): fail(1, b"daemon not running")})
    assert run(make_ctx(runner), "retire", "--root", str(root)) == 2
    assert (root / MARKER).is_file()
    assert registry.read_text(encoding="utf-8").splitlines() == [str(root)]


def test_retire_refuses_an_unmarked_root(make_ctx, root):
    root.mkdir(parents=True)
    runner = FakeRunner()
    assert run(make_ctx(runner), "retire", "--root", str(root)) == 2
    assert runner.calls == []


def test_retire_removes_only_the_markers_and_registry_line(make_ctx, root, registry, tmp_path):
    init_root(make_ctx, root)
    other = str((tmp_path / "other-root").resolve())
    with registry.open("a", encoding="utf-8") as handle:
        handle.write(other + "\n")
    populate(root)
    marker_paths = {MARKER, f"state/{MARKER}", f"mist-memory/{MARKER}"}
    survivors = {k: v for k, v in all_files(root).items() if k not in marker_paths}
    dirs_before = sorted(p.relative_to(root) for p in root.rglob("*") if p.is_dir())

    runner = FakeRunner({tuple(compose_ps("daily")): ok()})
    assert run(make_ctx(runner), "retire", "--root", str(root)) == 0

    assert all_files(root) == survivors
    assert sorted(p.relative_to(root) for p in root.rglob("*") if p.is_dir()) == dirs_before
    assert registry.read_text(encoding="utf-8").splitlines() == [other]


# --- release-build ---------------------------------------------------------------------------


def make_archive(*, with_compose: bool = True, extra: tarfile.TarInfo | None = None) -> bytes:
    """A small release archive; `extra` is one more member, a file holding `EXTRA_DATA`."""
    files = {
        "deploy/daily/daily.env.example": b"MIST_RELEASE_TAG=<release-tag>\n",
        "deploy/README.md": b"deploy\n",
        "scripts/daily/__init__.py": b"",
        "scripts/daily/instance.py": b"print('instance')\n",
        "scripts/backup/dump.py": b"dump\n",
        "docker/backend/Dockerfile": b"FROM scratch\n",
        "backend/server.py": b"server\n",
    }
    if with_compose:
        files["deploy/daily/compose.yml"] = b"services: {}\n"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
        if extra is not None:
            tar.addfile(extra, io.BytesIO(EXTRA_DATA) if extra.isfile() else None)
    return buffer.getvalue()


EXTRA_DATA = b"escaped\n"


def member(name: str, kind: bytes = tarfile.REGTYPE, linkname: str = "") -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = kind
    info.linkname = linkname
    info.mode = 0o644
    info.size = len(EXTRA_DATA) if info.isfile() else 0
    return info


ARCHIVE = make_archive()


def git(repo: str, *args: str) -> list[str]:
    return ["git", "-C", repo, *args]


def docker_build(tag: str, commit: str = SHA) -> list[str]:
    return [
        "docker",
        "build",
        "-f",
        "docker/backend/Dockerfile",
        "-t",
        f"mist-backend:{tag}",
        "--build-arg",
        f"MIST_RELEASE_TAG={tag}",
        "--build-arg",
        f"MIST_RELEASE_COMMIT={commit}",
        "-",
    ]


def inspect(tag: str) -> list[str]:
    return ["docker", "image", "inspect", "--format", "{{.Id}}", f"mist-backend:{tag}"]


def release_calls(repo: str = REPO_ARG, tag: str = TAG) -> list[list[str]]:
    return [
        git(repo, "cat-file", "-t", f"refs/tags/{tag}"),
        git(repo, "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}"),
        git(repo, "merge-base", "--is-ancestor", SHA, "origin/main"),
        git(repo, "archive", "--format=tar", SHA),
        docker_build(tag),
        inspect(tag),
    ]


def release_responses(repo: str = REPO_ARG, tag: str = TAG, **overrides) -> dict:
    results = [
        ok(b"tag\n"),
        ok(SHA.encode() + b"\n"),
        ok(),
        ok(ARCHIVE),
        ok(b"built\n"),
        ok(IMAGE_ID.encode() + b"\n"),
    ]
    responses = {tuple(argv): result for argv, result in zip(release_calls(repo, tag), results)}
    for index, result in overrides.items():
        responses[tuple(release_calls(repo, tag)[int(index.lstrip("c"))])] = result
    return responses


@pytest.fixture
def built_root(make_ctx, root) -> Path:
    init_root(make_ctx, root)
    return root


def release_build(make_ctx, runner, root: Path, *args: str) -> int:
    return run(make_ctx(runner), "release-build", *args, "--root", str(root))


@pytest.mark.parametrize("tag", [TAG, "daily-2026.10.23.2"])
def test_release_build_pipes_the_archive_into_docker_build(make_ctx, built_root, tag):
    runner = FakeRunner(release_responses(tag=tag))
    assert release_build(make_ctx, runner, built_root, tag, "--repo", REPO_ARG) == 0
    assert runner.calls == release_calls(tag=tag)
    assert runner.stdin[runner.calls.index(docker_build(tag))] == ARCHIVE
    record = json.loads((built_root / "ops" / f"release-{tag}.json").read_text(encoding="utf-8"))
    assert record == {"tag": tag, "commit": SHA, "image_id": IMAGE_ID, "built_at": FIXED_STAMP}


def test_release_build_extracts_deploy_and_scripts_daily_only(make_ctx, built_root):
    runner = FakeRunner(release_responses())
    assert release_build(make_ctx, runner, built_root, TAG, "--repo", REPO_ARG) == 0
    assert all_files(built_root / "release" / TAG) == {
        "deploy/README.md": b"deploy\n",
        "deploy/daily/compose.yml": b"services: {}\n",
        "deploy/daily/daily.env.example": b"MIST_RELEASE_TAG=<release-tag>\n",
        "scripts/daily/__init__.py": b"",
        "scripts/daily/instance.py": b"print('instance')\n",
    }
    assert [p.name for p in (built_root / "release").iterdir()] == [TAG]


def test_release_build_replaces_an_earlier_extraction_of_the_tag(make_ctx, built_root):
    stale = built_root / "release" / TAG / "deploy" / "stale.yml"
    stale.parent.mkdir(parents=True)
    stale.write_text("old\n", encoding="utf-8")
    runner = FakeRunner(release_responses())
    assert release_build(make_ctx, runner, built_root, TAG, "--repo", REPO_ARG) == 0
    assert not stale.exists()
    assert (built_root / "release" / TAG / "deploy" / "daily" / "compose.yml").is_file()
    assert [p.name for p in (built_root / "release").iterdir()] == [TAG]


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(member("deploy/daily/link", tarfile.SYMTYPE, "../../../escape"), id="symlink"),
        pytest.param(member("deploy/hard", tarfile.LNKTYPE, "deploy/README.md"), id="hardlink"),
        pytest.param(member("deploy/fifo", tarfile.FIFOTYPE), id="fifo"),
        pytest.param(member("scripts/daily/tty", tarfile.CHRTYPE), id="char-device"),
        pytest.param(member("deploy/../../escape"), id="dotdot"),
        pytest.param(member("deploy/a\\..\\..\\..\\..\\escape"), id="backslash"),
        pytest.param(member("deploy/C:/Windows/escape"), id="drive"),
    ],
)
def test_release_build_refuses_an_unsafe_archive_member(make_ctx, built_root, tmp_path, bad):
    """`--rehearsal` builds any commit, so a hostile archive must be refused before the build."""
    runner = FakeRunner(release_responses(c3=ok(make_archive(extra=bad))))
    assert release_build(make_ctx, runner, built_root, TAG, "--repo", REPO_ARG) == 2
    assert runner.calls == release_calls()[:4]
    assert list((built_root / "release").iterdir()) == []
    assert list((built_root / "ops").iterdir()) == []
    assert not (built_root / "escape").exists()
    assert not (tmp_path / "escape").exists()


def test_release_build_skips_an_absolute_member_and_never_writes_it(make_ctx, built_root):
    absolute = "/deploy/daily/escape.yml"
    runner = FakeRunner(release_responses(c3=ok(make_archive(extra=member(absolute)))))
    assert release_build(make_ctx, runner, built_root, TAG, "--repo", REPO_ARG) == 0
    assert not Path(absolute).exists()
    assert "deploy/daily/escape.yml" not in all_files(built_root / "release" / TAG)


@pytest.mark.parametrize("name", ["deploy/../../escape", "deploy/../../../outside/escape"])
def test_extract_release_refuses_a_target_outside_the_staging_directory(tmp_path, name):
    """The resolved-target check stands behind the name checks, so it is driven directly."""
    release_dir = tmp_path / "root" / "release"
    with pytest.raises(OSError, match="outside"):
        instance._extract_release(release_dir, TAG, [("deploy/ok.yml", b"ok\n"), (name, b"x")])
    assert list(release_dir.iterdir()) == []
    assert not (tmp_path / "root" / "escape").exists()
    assert not (tmp_path / "root" / "outside").exists()


def fail_renames_onto(target: Path, monkeypatch, *, also_restore: bool) -> None:
    """Make renaming the new tree onto `target` fail, and optionally the move back too."""
    real_rename = Path.rename

    def rename(self, destination):
        moving_new_tree = ".old." not in str(self)
        if Path(destination) == target and (moving_new_tree or also_restore):
            raise OSError("rename refused")
        return real_rename(self, destination)

    monkeypatch.setattr(instance.Path, "rename", rename)


def test_release_build_restores_the_earlier_extraction_when_the_swap_fails(
    make_ctx, built_root, monkeypatch, capsys
):
    earlier = built_root / "release" / TAG / "deploy" / "earlier.yml"
    earlier.parent.mkdir(parents=True)
    earlier.write_text("earlier\n", encoding="utf-8")
    fail_renames_onto(built_root / "release" / TAG, monkeypatch, also_restore=False)
    runner = FakeRunner(release_responses())
    assert release_build(make_ctx, runner, built_root, TAG, "--repo", REPO_ARG) == 1
    assert earlier.read_text(encoding="utf-8") == "earlier\n"
    assert [p.name for p in (built_root / "release").iterdir()] == [TAG]
    assert "back in place" in capsys.readouterr().err


def test_release_build_names_where_the_earlier_extraction_is_kept(
    make_ctx, built_root, monkeypatch, capsys
):
    earlier = built_root / "release" / TAG / "deploy" / "earlier.yml"
    earlier.parent.mkdir(parents=True)
    earlier.write_text("earlier\n", encoding="utf-8")
    fail_renames_onto(built_root / "release" / TAG, monkeypatch, also_restore=True)
    runner = FakeRunner(release_responses())
    assert release_build(make_ctx, runner, built_root, TAG, "--repo", REPO_ARG) == 1
    kept = [p for p in (built_root / "release").iterdir() if ".old." in p.name]
    assert len(kept) == 1
    assert (kept[0] / TAG / "deploy" / "earlier.yml").read_text(encoding="utf-8") == "earlier\n"
    err = capsys.readouterr().err
    assert f"kept at {kept[0] / TAG}" in err
    assert "ops/release-" in err


@pytest.mark.parametrize(
    "tag",
    [
        "daily-2026.1.23",
        "daily-2026.10.23-rc1",
        "daily-2026.10.23.",
        "daily-2026.10.23.x",
        "v2026.10.23",
        "rehearsal-0123456",
        "daily-2026.10.23\n",
    ],
)
def test_release_build_refuses_a_tag_off_the_pattern(make_ctx, built_root, tag):
    runner = FakeRunner()
    assert release_build(make_ctx, runner, built_root, tag, "--repo", REPO_ARG) == 2
    assert runner.calls == []


def test_release_build_refuses_an_unannotated_tag(make_ctx, built_root):
    runner = FakeRunner(release_responses(c0=ok(b"commit\n")))
    assert release_build(make_ctx, runner, built_root, TAG, "--repo", REPO_ARG) == 2
    assert runner.calls == release_calls()[:1]
    assert list((built_root / "ops").iterdir()) == []


def test_release_build_refuses_a_missing_tag(make_ctx, built_root):
    runner = FakeRunner(release_responses(c0=fail(128, b"Not a valid object name")))
    assert release_build(make_ctx, runner, built_root, TAG, "--repo", REPO_ARG) == 2
    assert runner.calls == release_calls()[:1]


@pytest.mark.parametrize("code", [1, 128])
def test_release_build_refuses_a_tag_not_on_origin_main(make_ctx, built_root, code):
    runner = FakeRunner(release_responses(c2=fail(code, b"")))
    assert release_build(make_ctx, runner, built_root, TAG, "--repo", REPO_ARG) == 2
    assert runner.calls == release_calls()[:3]
    assert list((built_root / "ops").iterdir()) == []
    assert list((built_root / "release").iterdir()) == []


def test_release_build_refuses_an_archive_without_the_compose_file(make_ctx, built_root):
    runner = FakeRunner(release_responses(c3=ok(make_archive(with_compose=False))))
    assert release_build(make_ctx, runner, built_root, TAG, "--repo", REPO_ARG) == 2
    assert runner.calls == release_calls()[:4]


def test_release_build_fails_with_exit_1_when_git_archive_fails(make_ctx, built_root):
    runner = FakeRunner(release_responses(c3=fail(128)))
    assert release_build(make_ctx, runner, built_root, TAG, "--repo", REPO_ARG) == 1
    assert runner.calls == release_calls()[:4]


def test_release_build_fails_with_exit_1_when_docker_build_fails(make_ctx, built_root, capsys):
    runner = FakeRunner(release_responses(c4=fail(1, b"step 7 failed")))
    assert release_build(make_ctx, runner, built_root, TAG, "--repo", REPO_ARG) == 1
    assert runner.calls == release_calls()[:5]
    assert list((built_root / "release").iterdir()) == []
    assert list((built_root / "ops").iterdir()) == []
    assert "step 7 failed" in capsys.readouterr().err


def test_release_build_names_the_state_when_inspect_fails(make_ctx, built_root, capsys):
    runner = FakeRunner(release_responses(c5=fail(1)))
    assert release_build(make_ctx, runner, built_root, TAG, "--repo", REPO_ARG) == 1
    assert list((built_root / "ops").iterdir()) == []
    assert f"mist-backend:{TAG}" in capsys.readouterr().err


def test_release_build_rehearsal_skips_the_ancestry_check(make_ctx, built_root):
    tag = f"rehearsal-{SHA[:7]}"
    runner = FakeRunner(
        {
            tuple(git(REPO_ARG, "rev-parse", "--verify", "feature^{commit}")): ok(
                SHA.encode() + b"\n"
            ),
            tuple(git(REPO_ARG, "archive", "--format=tar", SHA)): ok(ARCHIVE),
            tuple(docker_build(tag)): ok(),
            tuple(inspect(tag)): ok(IMAGE_ID.encode()),
        }
    )
    assert (
        release_build(make_ctx, runner, built_root, "--rehearsal", "feature", "--repo", REPO_ARG)
        == 0
    )
    assert runner.calls == [
        git(REPO_ARG, "rev-parse", "--verify", "feature^{commit}"),
        git(REPO_ARG, "archive", "--format=tar", SHA),
        docker_build(tag),
        inspect(tag),
    ]
    record = json.loads((built_root / "ops" / f"release-{tag}.json").read_text(encoding="utf-8"))
    assert record["tag"] == tag
    assert (built_root / "release" / tag / "deploy" / "daily" / "compose.yml").is_file()


def test_release_build_rehearsal_refuses_an_unknown_ref(make_ctx, built_root):
    runner = FakeRunner({tuple(git(REPO_ARG, "rev-parse", "--verify", "nope^{commit}")): fail(128)})
    assert (
        release_build(make_ctx, runner, built_root, "--rehearsal", "nope", "--repo", REPO_ARG) == 2
    )


@pytest.mark.parametrize("args", [(), (TAG, "--rehearsal", "feature")])
def test_release_build_needs_exactly_one_of_tag_and_rehearsal(make_ctx, built_root, args):
    runner = FakeRunner()
    assert release_build(make_ctx, runner, built_root, *args, "--repo", REPO_ARG) == 2
    assert runner.calls == []


def test_release_build_refuses_an_unmarked_root(make_ctx, root):
    root.mkdir(parents=True)
    runner = FakeRunner()
    assert release_build(make_ctx, runner, root, TAG, "--repo", REPO_ARG) == 2
    assert runner.calls == []


def test_release_build_repo_defaults_to_the_scripts_work_tree(make_ctx, built_root, tmp_path):
    repo = tmp_path / "checkout"
    (repo / ".git").mkdir(parents=True)
    script = repo / "scripts" / "daily" / "instance.py"
    runner = FakeRunner(release_responses(repo=str(repo.resolve())))
    assert release_build(lambda r: make_ctx(r, script), runner, built_root, TAG) == 0
    assert runner.calls == release_calls(repo=str(repo.resolve()))


def test_release_build_requires_repo_when_run_from_bin(make_ctx, built_root):
    script = built_root / "bin" / "instance.py"
    runner = FakeRunner()
    assert release_build(lambda r: make_ctx(r, script), runner, built_root, TAG) == 2
    assert runner.calls == []


def test_release_build_from_bin_runs_with_an_explicit_repo(make_ctx, built_root, tmp_path):
    script = built_root / "bin" / "instance.py"
    repo = str((tmp_path / "checkout").resolve())
    runner = FakeRunner(release_responses(repo=repo))
    assert (
        release_build(lambda r: make_ctx(r, script), runner, built_root, TAG, "--repo", repo) == 0
    )
    assert runner.calls == release_calls(repo=repo)
