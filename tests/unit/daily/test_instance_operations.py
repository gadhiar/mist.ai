"""Tests for `scripts/daily/instance.py`: switch, up, down, status and backup arming.

Every Docker call goes through the injected runner, every sleep and poll through an injected
clock, and every HTTP read through an injected fetcher, so no test needs Docker, sleeps or
reaches a network. Instance roots and the registry live under `tmp_path`.
"""

from __future__ import annotations

import ast
import http.server
import json
import os
import re
import threading
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from scripts.daily import instance

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "daily" / "instance.py"
README = REPO_ROOT / "deploy" / "daily" / "README.md"
HEALTH = REPO_ROOT / "backend" / "health.py"

FIXED_NOW = datetime(2026, 10, 23, 9, 30, 5, tzinfo=UTC)
TAG = "daily-2026.10.23"
OLD_TAG = "daily-2026.10.01"
IMAGE_ID = "sha256:" + "a" * 64
OLD_IMAGE_ID = "sha256:" + "c" * 64
OTHER_ID = "sha256:" + "b" * 64
SHA = "0123456789abcdef0123456789abcdef01234567"
MARKER = ".mist-live-instance"
DOCKER_INFO = ("docker", "info", "--format", "{{.ServerVersion}}")
NETWORK_INSPECT = ("docker", "network", "inspect", "mist-llm-share")
NETWORK_CREATE = ("docker", "network", "create", "mist-llm-share")
INSPECT_FORMAT = "{{.Name}} {{.Image}} {{json .Config.Labels}}"


class FakeRunner:
    """Answers each exact argv with a canned result and records every call.

    A list answers successive calls in order and then repeats its last entry. An argv with no
    canned answer fails the test, so every Docker call a command makes is one the test expected.
    `events`, when shared with the clock, records calls and sleeps in one order.
    """

    def __init__(self, responses=None, events: list | None = None, on_call=None):
        self.responses = dict(responses or {})
        self.calls: list[list[str]] = []
        self.events = events if events is not None else []
        self.on_call = on_call

    def __call__(self, argv, *, stdin=None):
        self.calls.append(list(argv))
        self.events.append(("run", list(argv)))
        if self.on_call is not None:
            self.on_call(list(argv))
        key = tuple(argv)
        if key not in self.responses:
            raise AssertionError(f"unexpected command: {list(argv)}")
        answer = self.responses[key]
        if isinstance(answer, list):
            return answer.pop(0) if len(answer) > 1 else answer[0]
        return answer


class FakeClock:
    """Monotonic time that only moves when the code sleeps."""

    def __init__(self, events: list | None = None):
        self.now = 1000.0
        self.sleeps: list[float] = []
        self.events = events if events is not None else []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.events.append(("sleep", seconds))
        self.now += seconds


class FakeFetch:
    """Answers URLs with canned bytes, or raises `OSError` for an unknown one."""

    def __init__(self, pages: dict[str, bytes] | None = None):
        self.pages = dict(pages or {})
        self.urls: list[str] = []

    def __call__(self, url: str, timeout: float) -> bytes:
        self.urls.append(url)
        if url not in self.pages:
            raise OSError(f"connection refused: {url}")
        return self.pages[url]


def ok(stdout: bytes | str = b"") -> instance.CommandResult:
    if isinstance(stdout, str):
        stdout = stdout.encode()
    return instance.CommandResult(returncode=0, stdout=stdout, stderr=b"")


def fail(code: int = 1, stderr: bytes = b"boom") -> instance.CommandResult:
    return instance.CommandResult(returncode=code, stdout=b"", stderr=stderr)


@pytest.fixture(autouse=True)
def registry(tmp_path, monkeypatch) -> Path:
    """Point the registry at a scratch file so no test can read `~/.mist/live-instances`."""
    path = tmp_path / "registry" / "live-instances"
    monkeypatch.setenv("MIST_LIVE_INSTANCES_FILE", str(path))
    return path


@pytest.fixture
def home(tmp_path) -> Path:
    path = tmp_path / "home"
    path.mkdir()
    return path


@dataclass
class Root:
    path: Path
    instance: str
    project: str

    @property
    def env(self) -> Path:
        return self.path / "daily.env"

    @property
    def ops(self) -> Path:
        return self.path / "ops"

    def compose_file(self, tag: str = TAG) -> Path:
        return self.path / "release" / tag / "deploy" / "daily" / "compose.yml"

    def base(self, tag: str = TAG) -> list[str]:
        return [
            "docker",
            "compose",
            "-p",
            self.project,
            "-f",
            str(self.compose_file(tag)),
            "--env-file",
            str(self.env),
        ]


def add_release(root: Root, tag: str, image_id: str = IMAGE_ID, scripts: dict | None = None):
    compose = root.compose_file(tag)
    compose.parent.mkdir(parents=True, exist_ok=True)
    compose.write_text("name: mist-daily\n", encoding="utf-8")
    daily = root.path / "release" / tag / "scripts" / "daily"
    daily.mkdir(parents=True, exist_ok=True)
    for name, data in (scripts or {}).items():
        (daily / name).write_bytes(data)
    record = {"tag": tag, "commit": SHA, "image_id": image_id, "built_at": "2026-10-23T09:00:00Z"}
    (root.ops / f"release-{tag}.json").write_text(json.dumps(record), encoding="utf-8")


def make_root(
    tmp_path: Path,
    registry: Path,
    *,
    name: str = "daily",
    instance_name: str = "daily",
    tag: str = TAG,
    backup: str | None = None,
    extra: str = "",
) -> Root:
    path = (tmp_path / "instances" / name).resolve()
    for sub in ("state", "mist-memory", "logs", "ops", "models", "release", "retired", "bin"):
        (path / sub).mkdir(parents=True, exist_ok=True)
    marker = json.dumps({"v": 1, "instance": instance_name, "created_at": "2026-10-01T00:00:00Z"})
    for directory in (path, path / "state", path / "mist-memory"):
        (directory / MARKER).write_text(marker + "\n", encoding="utf-8")
    registry.parent.mkdir(parents=True, exist_ok=True)
    with registry.open("a", encoding="utf-8") as handle:
        handle.write(f"{path}\n")
    lines = [
        "# daily.env for a test instance",
        f"MIST_RELEASE_TAG={tag}",
        f"MIST_DAILY_ROOT={path.as_posix()}",
        f"MIST_INSTANCE_NAME={instance_name}",
        "NEO4J_PASSWORD=<neo4j-password>",
        "MIST_DAILY_NEO4J_VOLUME=placeholder-volume",
        f"MODELS_DIR={(path / 'models').as_posix()}",
    ]
    if backup is not None:
        lines.append(f"MIST_BACKUP_HOST_PATH={backup}")
    text = "\n".join(lines) + "\n" + extra
    (path / "daily.env").write_text(text, encoding="utf-8", newline="")
    root = Root(path=path, instance=instance_name, project=f"mist-{instance_name}")
    add_release(root, tag)
    return root


@pytest.fixture
def backup_dir(tmp_path) -> Path:
    path = (tmp_path / "backups").resolve()
    path.mkdir()
    return path


@pytest.fixture
def daily(tmp_path, registry, backup_dir) -> Root:
    return make_root(tmp_path, registry, backup=str(backup_dir))


@pytest.fixture
def make_ctx(home):
    def _make(runner=None, clock: FakeClock | None = None, fetch=None) -> instance.Context:
        clock = clock if clock is not None else FakeClock()
        return instance.Context(
            runner=runner if runner is not None else FakeRunner(),
            clock=lambda: FIXED_NOW,
            home=home,
            script_path=SCRIPT,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
            fetch=fetch if fetch is not None else FakeFetch(),
        )

    return _make


def run(ctx: instance.Context, *argv: str) -> int:
    return instance.main(list(argv), ctx)


def image_inspect(tag: str) -> tuple[str, ...]:
    return ("docker", "image", "inspect", "--format", "{{.Id}}", f"mist-backend:{tag}")


def project_ps(project: str) -> tuple[str, ...]:
    return (
        "docker",
        "ps",
        "--all",
        "--quiet",
        "--no-trunc",
        "--filter",
        f"label=com.docker.compose.project={project}",
    )


def container_inspect(cid: str) -> tuple[str, ...]:
    return ("docker", "inspect", "--format", INSPECT_FORMAT, cid)


def inspect_line(name: str, image_id: str, labels: dict) -> instance.CommandResult:
    return ok(f"/{name} {image_id} {json.dumps(labels)}\n")


def labels_for(root: Root, service: str, config_file: str | None = None) -> dict:
    return {
        "com.docker.compose.project": root.project,
        "com.docker.compose.service": service,
        "com.docker.compose.project.config_files": config_file or str(root.compose_file()),
        "org.opencontainers.image.revision": SHA,
    }


def containers_responses(root: Root, mismatched: dict[str, str] | None = None) -> dict:
    """The listing and inspection of the project's three containers."""
    mismatched = mismatched or {}
    responses: dict = {}
    ids = []
    for service in ("mist-neo4j", "mist-llm", "mist-backend"):
        cid = f"id-{service}"
        ids.append(cid)
        labels = labels_for(root, service, mismatched.get(service))
        responses[container_inspect(cid)] = inspect_line(
            f"{root.project}-{service}-1", IMAGE_ID, labels
        )
    responses[project_ps(root.project)] = ok("\n".join(ids) + "\n")
    return responses


def up_responses(root: Root, services=("mist-neo4j", "mist-llm", "mist-backend")) -> dict:
    responses = {
        DOCKER_INFO: ok("27.0.0\n"),
        image_inspect(TAG): ok(IMAGE_ID + "\n"),
        NETWORK_INSPECT: ok("[]"),
        (*root.base(), "up", "-d", *services): ok(),
        (*root.base(), "--profile", "backup", "up", "-d", "mist-backup"): ok(),
    }
    responses.update(containers_responses(root))
    return responses


def compose_calls(runner: FakeRunner) -> list[list[str]]:
    return [call for call in runner.calls if call[:2] == ["docker", "compose"]]


def set_env_line(root: Root, line: str) -> None:
    with root.env.open("a", encoding="utf-8", newline="") as handle:
        handle.write(line + "\n")


# --- Configuration refusals ------------------------------------------------------------------


def command_args(command: str, root: Root) -> list[str]:
    if command == "switch":
        return ["switch", TAG, "--root", str(root.path)]
    return [command, "--root", str(root.path)]


def switch_ready(root: Root) -> None:
    """Make `switch TAG` possible: daily.env names the old tag, both releases exist."""
    add_release(root, OLD_TAG, OLD_IMAGE_ID)
    text = root.env.read_text(encoding="utf-8")
    root.env.write_text(text.replace(f"MIST_RELEASE_TAG={TAG}", f"MIST_RELEASE_TAG={OLD_TAG}"))


CHECKED_COMMANDS = ("switch", "up", "arm-backups", "backup-now")


@pytest.mark.parametrize("command", CHECKED_COMMANDS)
@pytest.mark.parametrize(
    "line",
    [
        "MIST_DAILY_WS_PORT=8011",
        "MIST_DAILY_BOLT_PORT=7697",
        "MIST_DAILY_NEO4J_HTTP_PORT=7475",
    ],
)
def test_daily_refuses_a_non_canonical_port_before_any_docker_call(make_ctx, daily, command, line):
    switch_ready(daily)
    set_env_line(daily, line)
    before = daily.env.read_bytes()
    runner = FakeRunner()
    assert run(make_ctx(runner), *command_args(command, daily)) == 2
    assert runner.calls == []
    assert daily.env.read_bytes() == before
    assert not (daily.ops / "backups-armed").exists()


@pytest.mark.parametrize("command", CHECKED_COMMANDS)
@pytest.mark.parametrize("instance_name", ["daily", "rehearsal"])
@pytest.mark.parametrize(
    "line",
    [
        "MIST_DAILY_WS_PORT=8002",
        "MIST_DAILY_BOLT_PORT=7688",
        "MIST_DAILY_BOLT_PORT=7689",
        "MIST_DAILY_NEO4J_HTTP_PORT=7690",
    ],
)
def test_any_instance_refuses_an_allowlisted_port_before_any_docker_call(
    make_ctx, tmp_path, registry, backup_dir, command, instance_name, line
):
    root = make_root(
        tmp_path, registry, name=instance_name, instance_name=instance_name, backup=str(backup_dir)
    )
    switch_ready(root)
    set_env_line(root, line)
    runner = FakeRunner()
    assert run(make_ctx(runner), *command_args(command, root)) == 2
    assert runner.calls == []


def test_a_rehearsal_may_use_other_ports(make_ctx, tmp_path, registry, backup_dir):
    root = make_root(
        tmp_path, registry, name="rehearsal", instance_name="rehearsal", backup=str(backup_dir)
    )
    set_env_line(root, "MIST_DAILY_WS_PORT=8011")
    set_env_line(root, "MIST_DAILY_BOLT_PORT=7697")
    runner = FakeRunner(up_responses(root))
    assert run(make_ctx(runner), "up", "--root", str(root.path)) == 0


def test_daily_accepts_the_canonical_ports_spelled_out(make_ctx, daily):
    for line in ("MIST_DAILY_WS_PORT=8001", "MIST_DAILY_BOLT_PORT=7687"):
        set_env_line(daily, line)
    set_env_line(daily, "MIST_DAILY_NEO4J_HTTP_PORT=7474")
    runner = FakeRunner(up_responses(daily))
    assert run(make_ctx(runner), "up", "--root", str(daily.path)) == 0


def backup_path_cases(tmp_path: Path, root: Root) -> dict[str, Path]:
    other = (tmp_path / "other-root").resolve()
    other.mkdir()
    registry_file = Path(os.environ["MIST_LIVE_INSTANCES_FILE"])
    with registry_file.open("a", encoding="utf-8") as handle:
        handle.write(f"{other}\n")
    marked = (tmp_path / "marked").resolve()
    marked.mkdir()
    (marked / MARKER).write_text("{}", encoding="utf-8")
    repo = (tmp_path / "repo").resolve()
    (repo / ".git").mkdir(parents=True)
    return {
        "instance root": root.path / "backups",
        "the root itself": root.path,
        "registered root": other / "backups",
        "marked directory": marked / "nested" / "backups",
        "git work tree": repo / "backups",
    }


@pytest.mark.parametrize("command", CHECKED_COMMANDS)
@pytest.mark.parametrize(
    "case",
    ["instance root", "the root itself", "registered root", "marked directory", "git work tree"],
)
def test_refuses_a_backup_host_path_inside_a_protected_tree_before_any_docker_call(
    make_ctx, tmp_path, daily, command, case, capsys
):
    switch_ready(daily)
    target = backup_path_cases(tmp_path, daily)[case]
    text = daily.env.read_text(encoding="utf-8")
    text = re.sub(
        r"^MIST_BACKUP_HOST_PATH=.*$", f"MIST_BACKUP_HOST_PATH={target}", text, flags=re.M
    )
    daily.env.write_text(text, encoding="utf-8")
    runner = FakeRunner()
    assert run(make_ctx(runner), *command_args(command, daily)) == 2
    assert runner.calls == []
    assert "MIST_BACKUP_HOST_PATH" in capsys.readouterr().err


@pytest.mark.parametrize("command", CHECKED_COMMANDS)
def test_refuses_an_image_that_differs_from_the_release_record(make_ctx, daily, command, capsys):
    if command == "switch":
        switch_ready(daily)
    before = daily.env.read_bytes()
    runner = FakeRunner(
        {
            DOCKER_INFO: ok("27.0.0\n"),
            image_inspect(TAG): ok(OTHER_ID + "\n"),
        }
    )
    assert run(make_ctx(runner), *command_args(command, daily)) == 2
    # Only read-only probes ran: nothing was started, stopped, created or run.
    assert all(call[:2] != ["docker", "compose"] for call in runner.calls)
    assert all("network" not in call for call in runner.calls)
    assert daily.env.read_bytes() == before
    assert not (daily.ops / "backups-armed").exists()
    err = capsys.readouterr().err
    assert OTHER_ID in err and IMAGE_ID in err
    assert f"release-build {TAG}" in err


def test_refuses_a_release_with_no_record(make_ctx, daily):
    (daily.ops / f"release-{TAG}.json").unlink()
    runner = FakeRunner({DOCKER_INFO: ok("27.0.0\n"), image_inspect(TAG): ok(IMAGE_ID + "\n")})
    assert run(make_ctx(runner), "up", "--root", str(daily.path)) == 2
    assert compose_calls(runner) == []


@pytest.mark.parametrize("value", ["<release-tag>", "", "../escape"])
def test_up_refuses_a_daily_env_without_a_release_tag(make_ctx, daily, value):
    text = daily.env.read_text(encoding="utf-8")
    daily.env.write_text(text.replace(f"MIST_RELEASE_TAG={TAG}", f"MIST_RELEASE_TAG={value}"))
    runner = FakeRunner()
    assert run(make_ctx(runner), "up", "--root", str(daily.path)) == 2
    assert runner.calls == []


def test_up_refuses_a_root_without_its_marker(make_ctx, daily):
    (daily.path / MARKER).unlink()
    runner = FakeRunner()
    assert run(make_ctx(runner), "up", "--root", str(daily.path)) == 2
    assert runner.calls == []


def test_up_refuses_a_missing_release_directory(make_ctx, daily, capsys):
    text = daily.env.read_text(encoding="utf-8")
    daily.env.write_text(
        text.replace(f"MIST_RELEASE_TAG={TAG}", "MIST_RELEASE_TAG=daily-2026.09.01")
    )
    runner = FakeRunner()
    assert run(make_ctx(runner), "up", "--root", str(daily.path)) == 2
    assert runner.calls == []
    assert "release-build daily-2026.09.01" in capsys.readouterr().err


# --- switch ----------------------------------------------------------------------------------


def switch_responses(root: Root, *, backend_running: bool = True, backup: int = 0) -> dict:
    responses = up_responses(root)
    responses[image_inspect(OLD_TAG)] = ok(OLD_IMAGE_ID + "\n")
    running = [{"Service": "mist-backend", "State": "running", "Name": "mist-daily-mist-backend-1"}]
    responses[
        ("docker", "compose", "-p", root.project, "ps", "--status", "running", "--format", "json")
    ] = ok(json.dumps(running if backend_running else []))
    responses[
        (
            *root.base(OLD_TAG),
            "--profile",
            "backup",
            "run",
            "--rm",
            "mist-backup",
            "python",
            "-m",
            "scripts.backup.scheduler",
            "--once",
        )
    ] = (
        ok(b"backup ok\n") if backup == 0 else fail(backup, b"backup failed")
    )
    return responses


def backup_run_call(root: Root, tag: str) -> list[str]:
    return [
        *root.base(tag),
        "--profile",
        "backup",
        "run",
        "--rm",
        "mist-backup",
        "python",
        "-m",
        "scripts.backup.scheduler",
        "--once",
    ]


def test_switch_refuses_a_tag_with_no_release_directory(make_ctx, daily):
    switch_ready(daily)
    before = daily.env.read_bytes()
    runner = FakeRunner()
    assert run(make_ctx(runner), "switch", "daily-2026.11.01", "--root", str(daily.path)) == 2
    assert runner.calls == []
    assert daily.env.read_bytes() == before


def test_switch_refuses_a_tag_with_no_image(make_ctx, daily, capsys):
    switch_ready(daily)
    before = daily.env.read_bytes()
    runner = FakeRunner({image_inspect(TAG): fail(1, b"No such image")})
    assert run(make_ctx(runner), "switch", TAG, "--root", str(daily.path)) == 2
    assert runner.calls == [list(image_inspect(TAG))]
    assert daily.env.read_bytes() == before
    assert f"release-build {TAG}" in capsys.readouterr().err


def test_switch_refuses_a_rehearsal_tag_on_the_daily_instance(make_ctx, daily):
    add_release(daily, "rehearsal-0123456")
    before = daily.env.read_bytes()
    runner = FakeRunner()
    assert run(make_ctx(runner), "switch", "rehearsal-0123456", "--root", str(daily.path)) == 2
    assert runner.calls == []
    assert daily.env.read_bytes() == before


@pytest.mark.parametrize("tag", ["latest", "../daily-2026.10.23", "daily-2026.10.23/x"])
def test_switch_refuses_a_tag_off_both_patterns(make_ctx, daily, tag):
    runner = FakeRunner()
    assert run(make_ctx(runner), "switch", tag, "--root", str(daily.path)) == 2
    assert runner.calls == []


def test_switch_accepts_a_rehearsal_tag_on_a_rehearsal_instance(make_ctx, tmp_path, registry):
    root = make_root(
        tmp_path, registry, name="rehearsal", instance_name="rehearsal", tag="rehearsal-0123456"
    )
    add_release(root, TAG)
    runner = FakeRunner(up_responses(root))
    assert run(make_ctx(runner), "switch", TAG, "--root", str(root.path)) == 0
    assert f"MIST_RELEASE_TAG={TAG}\n" in root.env.read_text(encoding="utf-8")


def test_switch_rewrites_only_the_release_tag_line(make_ctx, daily):
    switch_ready(daily)
    # CRLF on one line, trailing spaces, a commented copy of the key and no final newline: every
    # byte but the tag value must survive.
    original = (
        daily.env.read_bytes()
        + b"# MIST_RELEASE_TAG=commented-out\r\n"
        + b"MIST_USAGE_TIMEZONE=Europe/London   \r\n"
        + b"MIST_BACKUP_RETAIN=7"
    )
    daily.env.write_bytes(original)
    runner = FakeRunner(switch_responses(daily))
    assert run(make_ctx(runner), "switch", TAG, "--root", str(daily.path)) == 0
    expected = original.replace(
        f"MIST_RELEASE_TAG={OLD_TAG}\n".encode(), f"MIST_RELEASE_TAG={TAG}\n".encode()
    )
    assert daily.env.read_bytes() == expected


def test_switch_copies_the_release_scripts_into_bin(make_ctx, daily):
    switch_ready(daily)
    scripts = {
        "instance.py": b"# instance from the release\nprint('x')\r\n",
        "gamewatch.py": b"# watcher from the release\n",
        "README.txt": b"not python",
    }
    add_release(daily, TAG, scripts=scripts)
    (daily.path / "bin" / "instance.py").write_bytes(b"# the old release's copy\n")
    runner = FakeRunner(switch_responses(daily))
    assert run(make_ctx(runner), "switch", TAG, "--root", str(daily.path)) == 0
    bin_dir = daily.path / "bin"
    assert (bin_dir / "instance.py").read_bytes() == scripts["instance.py"]
    assert (bin_dir / "gamewatch.py").read_bytes() == scripts["gamewatch.py"]
    assert not (bin_dir / "README.txt").exists()


def test_switch_backs_up_first_then_rewrites_then_runs_up(make_ctx, daily):
    switch_ready(daily)
    seen_tags: dict[str, str] = {}

    def on_call(argv):
        if "scheduler" in " ".join(argv):
            seen_tags["at backup"] = daily.env.read_text(encoding="utf-8")

    runner = FakeRunner(switch_responses(daily), on_call=on_call)
    assert run(make_ctx(runner), "switch", TAG, "--root", str(daily.path)) == 0
    # The backup runs on the release that wrote the data, before daily.env moves.
    assert f"MIST_RELEASE_TAG={OLD_TAG}" in seen_tags["at backup"]
    backup_index = runner.calls.index(backup_run_call(daily, OLD_TAG))
    up_index = runner.calls.index(
        [*daily.base(), "up", "-d", "mist-neo4j", "mist-llm", "mist-backend"]
    )
    assert backup_index < up_index


def test_switch_stops_with_the_backup_exit_code_and_changes_nothing(make_ctx, daily):
    switch_ready(daily)
    before = daily.env.read_bytes()
    add_release(daily, TAG, scripts={"instance.py": b"new\n"})
    runner = FakeRunner(switch_responses(daily, backup=3))
    assert run(make_ctx(runner), "switch", TAG, "--root", str(daily.path)) == 3
    assert daily.env.read_bytes() == before
    assert not (daily.path / "bin" / "instance.py").exists()
    assert [*daily.base(), "up", "-d", "mist-neo4j", "mist-llm", "mist-backend"] not in runner.calls


def test_switch_skip_backup_runs_no_backup(make_ctx, daily):
    switch_ready(daily)
    runner = FakeRunner(switch_responses(daily))
    assert run(make_ctx(runner), "switch", TAG, "--root", str(daily.path), "--skip-backup") == 0
    assert backup_run_call(daily, OLD_TAG) not in runner.calls
    assert not any("ps" in call and "--status" in call for call in runner.calls)


def test_switch_runs_no_backup_while_the_backend_is_stopped(make_ctx, daily):
    switch_ready(daily)
    runner = FakeRunner(switch_responses(daily, backend_running=False))
    assert run(make_ctx(runner), "switch", TAG, "--root", str(daily.path)) == 0
    assert backup_run_call(daily, OLD_TAG) not in runner.calls


def test_switch_runs_no_backup_without_a_backup_host_path(make_ctx, tmp_path, registry):
    root = make_root(tmp_path, registry, backup=None)
    switch_ready(root)
    runner = FakeRunner(switch_responses(root))
    assert run(make_ctx(runner), "switch", TAG, "--root", str(root.path)) == 0
    assert backup_run_call(root, OLD_TAG) not in runner.calls
    assert not any("--status" in call for call in runner.calls)


def test_switch_names_the_state_when_up_fails_afterwards(make_ctx, daily, capsys):
    switch_ready(daily)
    responses = switch_responses(daily)
    responses[(*daily.base(), "up", "-d", "mist-neo4j", "mist-llm", "mist-backend")] = fail()
    runner = FakeRunner(responses)
    assert run(make_ctx(runner), "switch", TAG, "--root", str(daily.path)) == 1
    assert f"MIST_RELEASE_TAG={TAG}" in daily.env.read_text(encoding="utf-8")
    err = capsys.readouterr().err
    assert "daily.env" in err and TAG in err


def test_switch_refuses_a_daily_env_without_one_release_tag_line(make_ctx, daily):
    switch_ready(daily)
    text = daily.env.read_text(encoding="utf-8")
    daily.env.write_text(text + f"MIST_RELEASE_TAG={OLD_TAG}\n", encoding="utf-8")
    before = daily.env.read_bytes()
    runner = FakeRunner()
    assert run(make_ctx(runner), "switch", TAG, "--root", str(daily.path)) == 2
    assert runner.calls == []
    assert daily.env.read_bytes() == before


# --- up --------------------------------------------------------------------------------------


def test_up_starts_the_three_services_in_one_call(make_ctx, daily):
    runner = FakeRunner(up_responses(daily))
    assert run(make_ctx(runner), "up", "--root", str(daily.path)) == 0
    assert compose_calls(runner) == [
        [*daily.base(), "up", "-d", "mist-neo4j", "mist-llm", "mist-backend"]
    ]


def test_up_leaves_mist_llm_out_while_the_gpu_is_yielded(make_ctx, daily):
    (daily.ops / "gpu-yielded").write_text('{"v": 1}', encoding="utf-8")
    runner = FakeRunner(up_responses(daily, services=("mist-neo4j", "mist-backend")))
    assert run(make_ctx(runner), "up", "--root", str(daily.path)) == 0
    assert compose_calls(runner) == [[*daily.base(), "up", "-d", "mist-neo4j", "mist-backend"]]


def test_up_starts_mist_backup_in_a_second_call_when_armed(make_ctx, daily):
    (daily.ops / "backups-armed").write_text("{}", encoding="utf-8")
    runner = FakeRunner(up_responses(daily))
    assert run(make_ctx(runner), "up", "--root", str(daily.path)) == 0
    assert compose_calls(runner) == [
        [*daily.base(), "up", "-d", "mist-neo4j", "mist-llm", "mist-backend"],
        [*daily.base(), "--profile", "backup", "up", "-d", "mist-backup"],
    ]


def test_up_reports_a_failed_backup_start_without_stopping_the_rest(make_ctx, daily, capsys):
    (daily.ops / "backups-armed").write_text("{}", encoding="utf-8")
    responses = up_responses(daily)
    responses[(*daily.base(), "--profile", "backup", "up", "-d", "mist-backup")] = fail()
    runner = FakeRunner(responses)
    assert run(make_ctx(runner), "up", "--root", str(daily.path)) == 1
    assert compose_calls(runner) == [
        [*daily.base(), "up", "-d", "mist-neo4j", "mist-llm", "mist-backend"],
        [*daily.base(), "--profile", "backup", "up", "-d", "mist-backup"],
    ]
    assert "mist-backup" in capsys.readouterr().err


def test_up_waits_for_docker_on_the_injected_clock(make_ctx, daily):
    responses = up_responses(daily)
    responses[DOCKER_INFO] = [fail(), fail(), ok("27.0.0\n")]
    runner = FakeRunner(responses)
    clock = FakeClock()
    assert run(make_ctx(runner, clock), "up", "--root", str(daily.path)) == 0
    assert runner.calls[:3] == [list(DOCKER_INFO)] * 3
    assert len(clock.sleeps) == 2


def test_up_gives_up_on_docker_after_wait_minutes(make_ctx, daily, capsys):
    runner = FakeRunner({DOCKER_INFO: fail()})
    clock = FakeClock()
    code = run(make_ctx(runner, clock), "up", "--root", str(daily.path), "--wait-minutes", "2")
    assert code == 1
    assert all(call == list(DOCKER_INFO) for call in runner.calls)
    assert 120 <= sum(clock.sleeps) < 130
    assert "docker" in capsys.readouterr().err.lower()


def test_up_waits_ten_minutes_by_default(make_ctx, daily):
    runner = FakeRunner({DOCKER_INFO: fail()})
    clock = FakeClock()
    assert run(make_ctx(runner, clock), "up", "--root", str(daily.path)) == 1
    assert 600 <= sum(clock.sleeps) < 610


def test_up_creates_the_share_network_when_missing(make_ctx, daily):
    responses = up_responses(daily)
    responses[NETWORK_INSPECT] = fail(1, b"network mist-llm-share not found")
    responses[NETWORK_CREATE] = ok("netid\n")
    runner = FakeRunner(responses)
    assert run(make_ctx(runner), "up", "--root", str(daily.path)) == 0
    assert runner.calls.index(list(NETWORK_CREATE)) < runner.calls.index(
        [*daily.base(), "up", "-d", "mist-neo4j", "mist-llm", "mist-backend"]
    )


def test_up_uses_the_configured_share_network_name(make_ctx, daily):
    set_env_line(daily, "MIST_LLM_SHARE_NETWORK=other-share")
    responses = up_responses(daily)
    del responses[NETWORK_INSPECT]
    responses[("docker", "network", "inspect", "other-share")] = ok("[]")
    runner = FakeRunner(responses)
    assert run(make_ctx(runner), "up", "--root", str(daily.path)) == 0


def test_up_fails_when_the_share_network_cannot_be_created(make_ctx, daily):
    responses = up_responses(daily)
    responses[NETWORK_INSPECT] = fail()
    responses[NETWORK_CREATE] = fail()
    runner = FakeRunner(responses)
    assert run(make_ctx(runner), "up", "--root", str(daily.path)) == 1
    assert compose_calls(runner) == []


def test_up_names_each_container_from_another_compose_file(make_ctx, daily, capsys):
    responses = up_responses(daily)
    responses.update(containers_responses(daily, {"mist-backend": "/elsewhere/docker-compose.yml"}))
    runner = FakeRunner(responses)
    assert run(make_ctx(runner), "up", "--root", str(daily.path)) == 1
    err = capsys.readouterr().err
    assert "mist-daily-mist-backend-1" in err
    assert "/elsewhere/docker-compose.yml" in err
    assert "mist-daily-mist-llm-1" not in err


def test_up_counts_a_label_naming_two_files_as_a_mismatch(make_ctx, daily):
    responses = up_responses(daily)
    both = f"{daily.compose_file()},/elsewhere/override.yml"
    responses.update(containers_responses(daily, {"mist-neo4j": both}))
    runner = FakeRunner(responses)
    assert run(make_ctx(runner), "up", "--root", str(daily.path)) == 1


def test_up_fails_when_compose_up_fails(make_ctx, daily):
    responses = up_responses(daily)
    responses[(*daily.base(), "up", "-d", "mist-neo4j", "mist-llm", "mist-backend")] = fail()
    runner = FakeRunner(responses)
    assert run(make_ctx(runner), "up", "--root", str(daily.path)) == 1


# --- down and the no-volume-deletion rule ----------------------------------------------------


def test_down_stops_the_project_including_mist_backup(make_ctx, daily):
    stop = (*daily.base(), "--profile", "backup", "stop")
    runner = FakeRunner({stop: ok()})
    assert run(make_ctx(runner), "down", "--root", str(daily.path)) == 0
    assert runner.calls == [list(stop)]


def test_down_stops_by_project_when_the_release_is_gone(make_ctx, daily):
    text = daily.env.read_text(encoding="utf-8")
    daily.env.write_text(
        text.replace(f"MIST_RELEASE_TAG={TAG}", "MIST_RELEASE_TAG=daily-2026.09.01")
    )
    stop = ("docker", "compose", "-p", "mist-daily", "stop")
    runner = FakeRunner({stop: ok()})
    assert run(make_ctx(runner), "down", "--root", str(daily.path)) == 0
    assert runner.calls == [list(stop)]


def string_lists(source: str) -> list[list[str]]:
    """Every list or tuple display in `source`, as its string constants."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.List | ast.Tuple):
            found.append(
                [
                    elt.value
                    for elt in node.elts
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                ]
            )
    return found


def test_no_argv_construction_can_delete_a_volume():
    source = SCRIPT.read_text(encoding="utf-8")
    constants = {
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    # No flag that removes volumes, no `rm` or `volume` subcommand, anywhere in the module.
    assert not constants & {"-v", "--volumes", "rm", "volume", "prune"}
    for items in string_lists(source):
        assert "down" not in items, items


def test_every_command_argv_recorded_here_is_volume_safe(make_ctx, daily):
    """The runtime argv of each operation, as a second check beside the source scan."""
    (daily.ops / "backups-armed").write_text("{}", encoding="utf-8")
    runner = FakeRunner(
        {
            **up_responses(daily),
            (*daily.base(), "--profile", "backup", "stop"): ok(),
        }
    )
    ctx = make_ctx(runner)
    assert run(ctx, "up", "--root", str(daily.path)) == 0
    assert run(ctx, "down", "--root", str(daily.path)) == 0
    for call in runner.calls:
        assert "-v" not in call and "--volumes" not in call
        assert "down" not in call and "rm" not in call
        assert call[:3] != ["docker", "volume", "rm"]


# --- arm-backups, disarm-backups, backup-now -------------------------------------------------


def test_arm_backups_writes_the_marker_and_starts_mist_backup(make_ctx, daily):
    start = (*daily.base(), "--profile", "backup", "up", "-d", "mist-backup")
    runner = FakeRunner({image_inspect(TAG): ok(IMAGE_ID + "\n"), start: ok()})
    assert run(make_ctx(runner), "arm-backups", "--root", str(daily.path)) == 0
    assert (daily.ops / "backups-armed").is_file()
    assert compose_calls(runner) == [list(start)]


def test_arm_backups_names_the_state_when_the_start_fails(make_ctx, daily, capsys):
    start = (*daily.base(), "--profile", "backup", "up", "-d", "mist-backup")
    runner = FakeRunner({image_inspect(TAG): ok(IMAGE_ID + "\n"), start: fail()})
    assert run(make_ctx(runner), "arm-backups", "--root", str(daily.path)) == 1
    assert (daily.ops / "backups-armed").is_file()
    assert "backups-armed" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["arm-backups", "backup-now"])
def test_backup_commands_refuse_without_a_backup_host_path(make_ctx, tmp_path, registry, command):
    root = make_root(tmp_path, registry, backup=None)
    runner = FakeRunner()
    assert run(make_ctx(runner), command, "--root", str(root.path)) == 2
    assert runner.calls == []
    assert not (root.ops / "backups-armed").exists()


def test_disarm_backups_removes_the_marker_and_stops_mist_backup(make_ctx, daily):
    (daily.ops / "backups-armed").write_text("{}", encoding="utf-8")
    stop = (*daily.base(), "--profile", "backup", "stop", "mist-backup")
    runner = FakeRunner({stop: ok()})
    assert run(make_ctx(runner), "disarm-backups", "--root", str(daily.path)) == 0
    assert not (daily.ops / "backups-armed").exists()
    assert runner.calls == [list(stop)]


def test_disarm_backups_names_the_state_when_the_stop_fails(make_ctx, daily, capsys):
    (daily.ops / "backups-armed").write_text("{}", encoding="utf-8")
    stop = (*daily.base(), "--profile", "backup", "stop", "mist-backup")
    runner = FakeRunner({stop: fail()})
    assert run(make_ctx(runner), "disarm-backups", "--root", str(daily.path)) == 1
    assert not (daily.ops / "backups-armed").exists()
    assert "mist-backup" in capsys.readouterr().err


@pytest.mark.parametrize("code", [0, 3])
def test_backup_now_runs_one_attempt_and_returns_its_exit_code(make_ctx, daily, code):
    once = tuple(backup_run_call(daily, TAG))
    runner = FakeRunner(
        {image_inspect(TAG): ok(IMAGE_ID + "\n"), once: ok() if code == 0 else fail(code)}
    )
    assert run(make_ctx(runner), "backup-now", "--root", str(daily.path)) == code
    assert compose_calls(runner) == [list(once)]


# --- status ----------------------------------------------------------------------------------


def health_payload(llm_status="up", llm_reason="ok") -> dict:
    return {
        "status": "degraded",
        "restart_recommended": False,
        "checks": {
            "neo4j": {"status": "up", "reason": "ok", "stale": False},
            "llm": {"status": llm_status, "reason": llm_reason, "stale": False},
            "backup": {"status": "down", "reason": "backup_never_run", "stale": False},
            "extraction_service": {"status": None, "reason": "disabled_by_config", "stale": True},
        },
        "extraction": {"pending": 4},
    }


def usage_payload() -> dict:
    return {
        "timezone": "UTC",
        "origin": "real",
        "days": [
            {"date": "2026-10-21", "real_turns": 0},
            {"date": "2026-10-22", "real_turns": 12},
            {"date": "2026-10-23", "real_turns": 3},
        ],
        "gate": {"window_days": 7, "required_days": 5, "days_used": 2, "met": False},
    }


HEALTH_URL = "http://127.0.0.1:8001/health"
USAGE_URL = "http://127.0.0.1:8001/usage/turns-per-day?days=7"


def status_responses(root: Root, *, backend_image: str = IMAGE_ID, mismatched=None) -> dict:
    responses = containers_responses(root, mismatched)
    ps = [
        {"ID": "id-mist-neo4j", "Service": "mist-neo4j", "State": "running", "Image": "neo4j:5"},
        {
            "ID": "id-mist-backend",
            "Service": "mist-backend",
            "State": "running",
            "Image": f"mist-backend:{TAG}",
        },
    ]
    # Newline-delimited objects, as current compose prints them.
    responses[(*root.base(), "ps", "--format", "json")] = ok("\n".join(json.dumps(p) for p in ps))
    responses[container_inspect("id-mist-backend")] = inspect_line(
        "mist-daily-mist-backend-1", backend_image, labels_for(root, "mist-backend")
    )
    responses[
        ("docker", "image", "inspect", "--format", "{{json .Config.Labels}}", backend_image)
    ] = ok(json.dumps({"org.opencontainers.image.revision": SHA}) + "\n")
    return responses


def status_fetch() -> FakeFetch:
    return FakeFetch(
        {
            HEALTH_URL: json.dumps(health_payload()).encode(),
            USAGE_URL: json.dumps(usage_payload()).encode(),
        }
    )


def test_status_prints_only_the_specified_lines(make_ctx, daily, capsys):
    (daily.ops / "gpu-yielded").write_text(
        json.dumps({"v": 1, "since": "2026-10-23T08:00:00Z", "by": "watcher"}), encoding="utf-8"
    )
    runner = FakeRunner(status_responses(daily))
    code = run(make_ctx(runner, fetch=status_fetch()), "status", "--root", str(daily.path))
    assert code == 0
    assert capsys.readouterr().out.splitlines() == [
        f"backend: tag {TAG}, image {IMAGE_ID} match, revision {SHA}",
        "gaming-mode: yielded since 2026-10-23T08:00:00Z by watcher",
        "health: degraded",
        "  neo4j: up ok",
        "  llm: up ok",
        "  backup: down backup_never_run",
        "  extraction_service: null disabled_by_config",
        "turns-per-day: 2026-10-21 0, 2026-10-22 12, 2026-10-23 3",
        "gate: not met, 2 of 7 days used, 5 required",
    ]


def test_status_reports_an_image_mismatch(make_ctx, daily, capsys):
    runner = FakeRunner(status_responses(daily, backend_image=OTHER_ID))
    run(make_ctx(runner, fetch=status_fetch()), "status", "--root", str(daily.path))
    first = capsys.readouterr().out.splitlines()[0]
    assert first == f"backend: tag {TAG}, image {OTHER_ID} MISMATCH, revision {SHA}"


def test_status_names_each_container_from_another_compose_file_and_exits_1(make_ctx, daily, capsys):
    runner = FakeRunner(
        status_responses(daily, mismatched={"mist-llm": "/elsewhere/docker-compose.yml"})
    )
    code = run(make_ctx(runner, fetch=status_fetch()), "status", "--root", str(daily.path))
    assert code == 1
    lines = capsys.readouterr().out.splitlines()
    assert lines[1] == (
        "config-file mismatch: mist-daily-mist-llm-1 uses /elsewhere/docker-compose.yml"
    )
    assert lines[2].startswith("gaming-mode: ")


def test_status_reads_the_configured_port(make_ctx, tmp_path, registry):
    root = make_root(tmp_path, registry, name="rehearsal", instance_name="rehearsal")
    set_env_line(root, "MIST_DAILY_WS_PORT=8011")
    fetch = FakeFetch()
    runner = FakeRunner(status_responses(root))
    run(make_ctx(runner, fetch=fetch), "status", "--root", str(root.path))
    assert fetch.urls == [
        "http://127.0.0.1:8011/health",
        "http://127.0.0.1:8011/usage/turns-per-day?days=7",
    ]


def test_status_says_so_when_the_backend_and_endpoints_are_down(make_ctx, daily, capsys):
    responses = status_responses(daily)
    responses[(*daily.base(), "ps", "--format", "json")] = ok("[]")
    runner = FakeRunner(responses)
    run(make_ctx(runner, fetch=FakeFetch()), "status", "--root", str(daily.path))
    assert capsys.readouterr().out.splitlines() == [
        "backend: not running",
        "gaming-mode: normal",
        "health: unreachable",
        "turns-per-day: unreachable",
    ]


def test_status_exits_1_when_the_container_listing_fails(make_ctx, daily, capsys):
    responses = status_responses(daily)
    responses[project_ps(daily.project)] = fail(1, b"daemon busy")
    runner = FakeRunner(responses)
    code = run(make_ctx(runner, fetch=status_fetch()), "status", "--root", str(daily.path))
    assert code == 1
    lines = capsys.readouterr().out.splitlines()
    assert lines[1].startswith("config-file check: failed (")
    assert "daemon busy" in lines[1]
    # The rest of the report still prints.
    assert lines[2].startswith("gaming-mode: ")
    assert lines[3] == "health: degraded"


# --- The loopback fetch ----------------------------------------------------------------------


class _Pages(http.server.BaseHTTPRequestHandler):
    seen: list[str] = []

    def do_GET(self):  # noqa: N802 - the name http.server dispatches to
        type(self).seen.append(self.path)
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/health")
            self.end_headers()
            return
        body = b'{"status": "healthy"}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def loopback_server():
    _Pages.seen = []
    server = http.server.HTTPServer(("127.0.0.1", 0), _Pages)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def test_fetch_opener_has_no_proxy_even_when_the_environment_sets_one(monkeypatch):
    for name in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        monkeypatch.setenv(name, "http://proxy.invalid:3128")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    # The default opener picks the proxy up from the environment...
    default = urllib.request.build_opener()
    assert any(isinstance(h, urllib.request.ProxyHandler) for h in default.handlers)
    # ...the loopback opener has no proxy handler at all: `ProxyHandler({})` registers no
    # protocol, so the opener drops it, and it displaces the default one.
    opener = instance.loopback_opener()
    assert not any(isinstance(h, urllib.request.ProxyHandler) for h in opener.handlers)
    assert not any(
        type(h) is urllib.request.HTTPRedirectHandler for h in opener.handlers
    ), "the default redirect handler is still installed"


def test_fetch_reads_loopback_directly_despite_a_proxy_setting(monkeypatch, loopback_server):
    # Port 9 on loopback refuses connections, so a fetch through this proxy would fail.
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    assert instance.urllib_fetch(f"{loopback_server}/health", 5.0) == b'{"status": "healthy"}'
    assert _Pages.seen == ["/health"]


def test_fetch_refuses_a_redirect_as_an_error(loopback_server):
    with pytest.raises(OSError):
        instance.urllib_fetch(f"{loopback_server}/redirect", 5.0)
    # The redirect target was never requested.
    assert _Pages.seen == ["/redirect"]


# --- README ----------------------------------------------------------------------------------


def readme_section(title: str) -> str:
    text = README.read_text(encoding="utf-8")
    match = re.search(rf"^## {re.escape(title)}\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    assert match is not None, f"README has no '## {title}' section"
    return match.group(1)


def test_readme_commands_section_covers_each_new_command():
    section = readme_section("Commands")
    for command in (
        "switch",
        "--skip-backup",
        "up",
        "--wait-minutes",
        "down",
        "status",
        "gaming-mode on",
        "gaming-mode off",
        "gaming-mode status",
        "arm-backups",
        "disarm-backups",
        "backup-now",
    ):
        assert command in section, command


def test_readme_says_switch_always_backs_up_unless_skip_backup():
    """Ruling (a): compose interpolation always requires the backup path, so `switch` always
    backs up a running instance unless `--skip-backup` is given.
    """
    section = " ".join(readme_section("Commands").split())
    assert "`compose.yml` declares `${MIST_BACKUP_HOST_PATH:?...}`" in section
    assert (
        "In practice `MIST_BACKUP_HOST_PATH` is therefore always set, and `switch` always backs "
        "up a running instance first. Pass `--skip-backup` to switch without that backup"
    ) in section


def test_readme_says_status_exits_1_when_the_containers_cannot_be_listed():
    section = " ".join(readme_section("Commands").split())
    status_row = next(line for line in section.split("| `") if line.startswith("status`"))
    assert "each container from another compose file (then exit 1)" in status_row
    assert "exits 1 too when the project's containers cannot be listed" in status_row


def test_readme_gaming_mode_section():
    section = readme_section("Gaming mode")
    for phrase in ("Whisper", "mist-llm", "keeps running", "yielded", "healthy", "CUDA context"):
        assert phrase in section, phrase


ITEM_3_REASONS = (
    "unavailable_at_boot",
    "disabled_by_config",
    "unreachable",
    "query_failed",
    "service_degraded",
    "epoch_mismatch",
    "stalled",
    "backlog_overdue",
    "dead_lettered",
    "backup_never_run",
    "backup_failed",
    "backup_overdue",
    "gpu_yielded",
    "resume_failed",
)


def test_readme_reading_health_gives_each_reason_and_remedy():
    section = readme_section("Reading /health")
    health_source = HEALTH.read_text(encoding="utf-8")
    for reason in ITEM_3_REASONS:
        assert f'"{reason}"' in health_source, f"{reason} is not a reason in backend/health.py"
        assert f"`{reason}`" in section, reason
    assert (
        "docker compose -p mist-daily exec -T mist-backend python -m "
        "backend.extraction_backlog.admin retry-dead-letters"
    ) in section
    assert "MSYS_NO_PATHCONV=1" in section
    assert "gaming-mode off" in section


def test_readme_claims_nothing_unmeasured_about_two_llama_servers():
    assert "one llama-server is what the 12 GB card holds" not in README.read_text(encoding="utf-8")
