"""Tests for `scripts/daily/instance.py`: the restore drill, promotion and backup baseline.

Every Docker call goes through the injected runner, which answers exact argument lists with
canned output and fails the test on any other, so no test needs Docker, touches a running stack
or reads live data. Instance roots, drill roots, backup roots and the registry live under
`tmp_path`; every value in a `daily.env` is a placeholder.
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from scripts.backup.target import RESTORE_MARKER_FILENAME
from scripts.daily import instance

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "daily" / "instance.py"
DAILY_DIR = REPO_ROOT / "deploy" / "daily"
README = DAILY_DIR / "README.md"
RESTORE_MD = DAILY_DIR / "RESTORE.md"
DRILL_COMPOSE = DAILY_DIR / "restore-drill.yml"
BACKUP_README = REPO_ROOT / "scripts" / "backup" / "README.md"

START = datetime(2026, 10, 23, 9, 30, 5, tzinfo=UTC)
LATER = START + timedelta(hours=1)
START_STAMP = "20261023T093005Z"
LATER_STAMP = "20261023T103005Z"
TAG = "daily-2026.10.23"
IMAGE_ID = "sha256:" + "a" * 64
OTHER_ID = "sha256:" + "b" * 64
SHA = "0123456789abcdef0123456789abcdef01234567"
MARKER = ".mist-live-instance"
ARTIFACT = "20261020T030000Z"
LIVE_VOLUME = "placeholder-live-volume"
VOLUME = f"mist-daily-drill-{START_STAMP}"
OTHER_DRILL_VOLUME = "mist-daily-drill-20261022T080000Z"
DRILL_PROJECT = "mist-daily-drill"
DRILL_URI = "bolt://mist-neo4j-dev:7687"
TOKEN = "/drill/standby/state"
DOCKER_INFO = ("docker", "info", "--format", "{{.ServerVersion}}")
INSPECT_FORMAT = "{{.Name}} {{.Image}} {{json .Config.Labels}}"
PROJECT_FORMAT = '{{.Label "com.docker.compose.project"}}'


SHORT_OPTION_CLUSTER = re.compile(r"-[A-Za-z]{2,}")


def refuse_volume_deleting_argv(argv: list[str]) -> None:
    """The drill tests' own check of v1's rule: no `down -v`, `rm -v` or `volume rm`.

    Written apart from `scripts/daily/volume_safety.py` on purpose, so a bug in the production
    guard cannot pass both.
    """
    for index, token in enumerate(argv):
        following = argv[index + 1] if index + 1 < len(argv) else None
        if (
            token == "-v"
            or token.startswith("--volumes")
            or token in ("down", "prune")
            or (SHORT_OPTION_CLUSTER.fullmatch(token) is not None and "v" in token)
            or (token == "volume" and following not in ("inspect", "create"))
        ):
            raise AssertionError(f"volume-deleting argv reached the runner: {argv}")


class FakeRunner:
    """Answers each exact argv with a canned result and records every call and its environment.

    Every argv first passes the test-side volume check. `answer`, when given, is asked first
    and may return a result for any argv, for a test that models Docker's state. Otherwise a
    list answers successive calls in order and then repeats its last entry, and an argv with no
    canned answer fails the test. `events`, when shared, records calls beside the filesystem
    steps a test observes, in one order.
    """

    def __init__(self, responses=None, events: list | None = None, on_call=None, answer=None):
        self.responses = dict(responses or {})
        self.calls: list[list[str]] = []
        self.envs: list[dict | None] = []
        self.events = events if events is not None else []
        self.on_call = on_call
        self.answer = answer

    def __call__(self, argv, *, stdin=None, env=None):
        refuse_volume_deleting_argv(list(argv))
        self.calls.append(list(argv))
        self.envs.append(None if env is None else dict(env))
        self.events.append(("run", list(argv)))
        if self.on_call is not None:
            self.on_call(list(argv))
        if self.answer is not None:
            result = self.answer(list(argv), env)
            if result is not None:
                return result
        key = tuple(argv)
        if key not in self.responses:
            raise AssertionError(f"unexpected command: {list(argv)}")
        answer = self.responses[key]
        if isinstance(answer, list):
            return answer.pop(0) if len(answer) > 1 else answer[0]
        return answer

    def env_for(self, argv) -> dict | None:
        return self.envs[self.calls.index(list(argv))]


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def ok(stdout: bytes | str = b"") -> instance.CommandResult:
    if isinstance(stdout, str):
        stdout = stdout.encode()
    return instance.CommandResult(returncode=0, stdout=stdout, stderr=b"")


def fail(
    code: int = 1, stderr: bytes | str = b"boom", stdout: bytes | str = b""
) -> instance.CommandResult:
    if isinstance(stderr, str):
        stderr = stderr.encode()
    if isinstance(stdout, str):
        stdout = stdout.encode()
    return instance.CommandResult(returncode=code, stdout=stdout, stderr=stderr)


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
    project: str = "mist-daily"

    @property
    def env(self) -> Path:
        return self.path / "daily.env"

    @property
    def ops(self) -> Path:
        return self.path / "ops"

    def release(self) -> Path:
        return self.path / "release" / TAG

    def compose_file(self) -> Path:
        return self.release() / "deploy" / "daily" / "compose.yml"

    def drill_file(self) -> Path:
        return self.release() / "deploy" / "daily" / "restore-drill.yml"

    def base(self) -> list[str]:
        return [
            "docker",
            "compose",
            "-p",
            self.project,
            "-f",
            str(self.compose_file()),
            "--env-file",
            str(self.env),
        ]

    def drill_base(self) -> list[str]:
        return [
            "docker",
            "compose",
            "-p",
            DRILL_PROJECT,
            "-f",
            str(self.drill_file()),
            "--env-file",
            str(self.env),
        ]


@pytest.fixture
def backup_dir(tmp_path) -> Path:
    path = (tmp_path / "backups").resolve()
    (path / ARTIFACT).mkdir(parents=True)
    (path / ARTIFACT / "manifest.json").write_text("{}", encoding="utf-8")
    return path


def write_marker(directory: Path) -> None:
    marker = {"v": 1, "instance": "daily", "created_at": "2026-10-01T00:00:00Z"}
    (directory / MARKER).write_text(json.dumps(marker) + "\n", encoding="utf-8")


@pytest.fixture
def daily(tmp_path, registry, backup_dir) -> Root:
    path = (tmp_path / "instances" / "daily").resolve()
    for sub in ("state", "mist-memory", "logs", "ops", "models", "release", "retired", "bin"):
        (path / sub).mkdir(parents=True, exist_ok=True)
    for directory in (path, path / "state", path / "mist-memory"):
        write_marker(directory)
    (path / "state" / "event_store.db").write_bytes(b"old-store")
    (path / "mist-memory" / "old.md").write_text("old\n", encoding="utf-8")
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(f"{path}\n", encoding="utf-8")
    lines = [
        "# daily.env for a test instance",
        f"MIST_RELEASE_TAG={TAG}",
        f"MIST_DAILY_ROOT={path.as_posix()}",
        "MIST_INSTANCE_NAME=daily",
        "NEO4J_PASSWORD=<neo4j-password>",
        f"MIST_DAILY_NEO4J_VOLUME={LIVE_VOLUME}",
        f"MODELS_DIR={(path / 'models').as_posix()}",
        f"MIST_BACKUP_HOST_PATH={backup_dir}",
    ]
    (path / "daily.env").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")
    root = Root(path)
    root.compose_file().parent.mkdir(parents=True, exist_ok=True)
    root.compose_file().write_text("name: mist-daily\n", encoding="utf-8")
    root.drill_file().write_text("services: {}\n", encoding="utf-8")
    record = {"tag": TAG, "commit": SHA, "image_id": IMAGE_ID, "built_at": "2026-10-23T09:00:00Z"}
    (root.ops / f"release-{TAG}.json").write_text(json.dumps(record), encoding="utf-8")
    return root


@pytest.fixture
def drill_root(tmp_path) -> Path:
    return (tmp_path / "drills" / "first").resolve()


@pytest.fixture
def make_ctx(home):
    def _make(runner=None, now: datetime = START) -> instance.Context:
        clock = FakeClock()
        return instance.Context(
            runner=runner if runner is not None else FakeRunner(),
            clock=lambda: now,
            environ={
                "PATH": "/usr/bin",
                "MIST_LIVE_INSTANCES_FILE": os.environ["MIST_LIVE_INSTANCES_FILE"],
            },
            home=home,
            script_path=SCRIPT,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
            fetch=lambda url, timeout: (_ for _ in ()).throw(OSError(url)),
        )

    return _make


def run(ctx: instance.Context, *argv: str) -> int:
    return instance.main(list(argv), ctx)


# --- argv builders ---------------------------------------------------------------------------


def image_inspect() -> tuple[str, ...]:
    return ("docker", "image", "inspect", "--format", "{{.Id}}", f"mist-backend:{TAG}")


def volume_inspect(name: str = VOLUME) -> tuple[str, ...]:
    return ("docker", "volume", "inspect", "--format", "{{json .Labels}}", name)


def volume_create(name: str = VOLUME) -> tuple[str, ...]:
    return (
        "docker",
        "volume",
        "create",
        "--label",
        "org.mist.role=drill",
        "--label",
        "org.mist.instance=daily",
        name,
    )


def volume_projects(name: str = VOLUME) -> tuple[str, ...]:
    return ("docker", "ps", "-a", "--filter", f"volume={name}", "--format", PROJECT_FORMAT)


def volume_ids(name: str = VOLUME) -> tuple[str, ...]:
    return ("docker", "ps", "-a", "--filter", f"volume={name}", "-q")


def drill_labels(role: str = "drill", inst: str = "daily") -> instance.CommandResult:
    return ok(json.dumps({"org.mist.role": role, "org.mist.instance": inst}) + "\n")


def restore_argv(root: Root, *extra: str) -> tuple[str, ...]:
    return (
        *root.drill_base(),
        "--profile",
        "run",
        "run",
        "--rm",
        "mist-restore",
        "-m",
        "scripts.backup.restore",
        "--artifact",
        f"/backup/{ARTIFACT}",
        "--target-root",
        "/drill/standby/state",
        "--target-vault-root",
        "/drill/standby/mist-memory",
        "--backup-root",
        "/drill/pre-restore",
        "--target-graph-uri",
        DRILL_URI,
        *extra,
    )


def verify_argv(root: Root) -> tuple[str, ...]:
    return (
        *root.drill_base(),
        "--profile",
        "run",
        "run",
        "--rm",
        "mist-restore",
        "-m",
        "scripts.backup.verify",
        "--artifact",
        f"/backup/{ARTIFACT}",
        "--graph-uri",
        DRILL_URI,
        "--state-root",
        "/drill/standby/state",
        "--vault-root",
        "/drill/standby/mist-memory",
    )


def drill_rm(root: Root) -> tuple[str, ...]:
    return (*root.drill_base(), "rm", "-s", "-f")


def instance_stop(root: Root) -> tuple[str, ...]:
    return (*root.base(), "--profile", "backup", "stop")


def instance_rm(root: Root) -> tuple[str, ...]:
    return (*root.base(), "--profile", "backup", "rm", "-f")


def drill_containers() -> tuple[str, ...]:
    return (
        "docker",
        "ps",
        "-a",
        "--filter",
        f"label=com.docker.compose.project={DRILL_PROJECT}",
        "-q",
    )


def service_ids(project: str, service: str) -> tuple[str, ...]:
    return (
        "docker",
        "ps",
        "-a",
        "--filter",
        f"label=com.docker.compose.project={project}",
        "--filter",
        f"label=com.docker.compose.service={service}",
        "-q",
    )


def drill_neo4j_ids() -> tuple[str, ...]:
    return service_ids(DRILL_PROJECT, "mist-neo4j-dev")


def instance_neo4j_ids() -> tuple[str, ...]:
    return service_ids("mist-daily", "mist-neo4j")


def mounts_inspect(container: str) -> tuple[str, ...]:
    return ("docker", "inspect", "--format", "{{json .Mounts}}", container)


def mounts(volume: str) -> instance.CommandResult:
    """`docker inspect --format '{{json .Mounts}}'` for a container with `volume` at /data."""
    entry = {
        "Type": "volume",
        "Name": volume,
        "Source": f"/var/lib/docker/volumes/{volume}/_data",
        "Destination": "/data",
        "Driver": "local",
        "Mode": "z",
        "RW": True,
        "Propagation": "",
    }
    return ok(json.dumps([entry]) + "\n")


def drill_neo4j_on(volume: str) -> dict:
    return {drill_neo4j_ids(): ok("c-drill\n"), mounts_inspect("c-drill"): mounts(volume)}


def volume_check_responses(name: str = VOLUME) -> dict:
    return {
        image_inspect(): ok(IMAGE_ID + "\n"),
        volume_inspect(name): drill_labels(),
        volume_projects(name): ok(f"{DRILL_PROJECT}\n"),
        **drill_neo4j_on(name),
    }


def up_responses(root: Root) -> dict:
    project_ps = (
        "docker",
        "ps",
        "--all",
        "--quiet",
        "--no-trunc",
        "--filter",
        "label=com.docker.compose.project=mist-daily",
    )
    responses: dict = {
        DOCKER_INFO: ok("27.0.0\n"),
        image_inspect(): ok(IMAGE_ID + "\n"),
        ("docker", "network", "inspect", "mist-llm-share"): ok("[]"),
        (*root.base(), "up", "-d", "mist-neo4j", "mist-llm", "mist-backend"): ok(),
        project_ps: ok("id-backend\n"),
    }
    labels = {
        "com.docker.compose.project": "mist-daily",
        "com.docker.compose.project.config_files": str(root.compose_file()),
    }
    responses[("docker", "inspect", "--format", INSPECT_FORMAT, "id-backend")] = ok(
        f"/mist-daily-mist-backend-1 {IMAGE_ID} {json.dumps(labels)}\n"
    )
    return responses


def compose_calls(runner: FakeRunner) -> list[list[str]]:
    return [call for call in runner.calls if call[:2] == ["docker", "compose"]]


def start_drill(make_ctx, root: Root, drill_root: Path) -> None:
    runner = FakeRunner(
        {
            image_inspect(): ok(IMAGE_ID + "\n"),
            drill_containers(): ok(""),
            volume_inspect(): fail(1, "Error: No such volume"),
            volume_create(): ok(VOLUME + "\n"),
        }
    )
    code = run(
        make_ctx(runner),
        "drill-start",
        "--root",
        str(root.path),
        "--drill-root",
        str(drill_root),
        "--artifact",
        ARTIFACT,
    )
    assert code == 0


def restore_into_standby(drill_root: Path) -> None:
    """What a successful drill-restore leaves in the standby directories."""
    state = drill_root / "standby" / "state"
    (state / RESTORE_MARKER_FILENAME).write_text("", encoding="utf-8")
    (state / "event_store.db").write_bytes(b"restored-store")
    memory = drill_root / "standby" / "mist-memory"
    memory.mkdir(parents=True, exist_ok=True)
    (memory / "restored.md").write_text("restored\n", encoding="utf-8")


VERIFY_MATCH = {
    "result": "match",
    "nodes": {"artifact": 3, "restored": 3},
    "relationships": {"artifact": 2, "restored": 2},
    "embeddings_equal": True,
    "stores": {"event_store.db": {"artifact": 3, "restored": 3}},
    "vault_files": {"artifact": 2, "restored": 2},
}


def write_drill_record(root: Root, drill: Path, *, stamp: str, verified_at: str, **changes):
    record = {
        "v": 1,
        "drill_root": str(drill),
        "volume": VOLUME,
        "artifact": ARTIFACT,
        "verified_at": verified_at,
        "exit_code": 0,
        **VERIFY_MATCH,
    }
    record.update(changes)
    (root.ops / f"drill-{stamp}.json").write_text(json.dumps(record), encoding="utf-8")


@pytest.fixture
def started(make_ctx, daily, drill_root) -> Path:
    start_drill(make_ctx, daily, drill_root)
    return drill_root


@pytest.fixture
def promote_ready(make_ctx, daily, started) -> Path:
    restore_into_standby(started)
    write_drill_record(daily, started, stamp=LATER_STAMP, verified_at="2026-10-23T10:30:05Z")
    return started


def tree(path: Path) -> dict[str, bytes]:
    return {
        p.relative_to(path).as_posix(): p.read_bytes()
        for p in sorted(path.rglob("*"))
        if p.is_file()
    }


# --- Pins ------------------------------------------------------------------------------------


def test_the_restore_marker_name_matches_the_restore_guard():
    assert instance.RESTORE_MARKER_FILENAME == RESTORE_MARKER_FILENAME


def test_the_drill_compose_file_named_here_exists_in_the_repository():
    assert (REPO_ROOT / instance.DRILL_COMPOSE_FILE) == DRILL_COMPOSE
    assert DRILL_COMPOSE.is_file()


def test_the_drill_graph_uri_is_the_drill_compose_files():
    services = yaml.safe_load(DRILL_COMPOSE.read_text(encoding="utf-8"))["services"]
    environment = dict(
        entry.partition("=")[::2] for entry in services["mist-restore"]["environment"]
    )
    assert instance.DRILL_GRAPH_URI == environment["NEO4J_URI"] == DRILL_URI


def test_the_release_extraction_carries_the_drill_compose_file():
    assert instance.DRILL_COMPOSE_FILE.startswith(instance.RELEASE_EXTRACT_PREFIXES)


def test_subprocess_runner_passes_the_given_environment():
    result = instance.subprocess_runner(
        [sys.executable, "-c", "import os, sys; sys.stdout.write(os.environ['MIST_DRILL_ROOT'])"],
        env={**os.environ, "MIST_DRILL_ROOT": "/placeholder/drill"},
    )
    assert result.returncode == 0
    assert result.stdout == b"/placeholder/drill"


# --- accept-backup-baseline ------------------------------------------------------------------


def accept_argv(root: Root, label: str) -> tuple[str, ...]:
    return (
        *root.base(),
        "--profile",
        "backup",
        "run",
        "--rm",
        "mist-backup",
        "python",
        "-m",
        "scripts.backup.scheduler",
        "--accept-suspect",
        label,
    )


@pytest.mark.parametrize("code", [0, 1, 2])
def test_accept_backup_baseline_runs_the_item_5_argv_and_returns_its_code(make_ctx, daily, code):
    label = "20261022T030000Z"
    runner = FakeRunner(
        {
            image_inspect(): ok(IMAGE_ID + "\n"),
            accept_argv(daily, label): fail(code) if code else ok(),
        }
    )

    result = run(
        make_ctx(runner), "accept-backup-baseline", "--root", str(daily.path), "--label", label
    )

    assert result == code
    assert compose_calls(runner) == [list(accept_argv(daily, label))]


@pytest.mark.parametrize("label", ["../escape", "a/b", "a\\b", ".hidden"])
def test_accept_backup_baseline_refuses_a_label_that_is_not_a_plain_name(make_ctx, daily, label):
    runner = FakeRunner()

    assert (
        run(make_ctx(runner), "accept-backup-baseline", "--root", str(daily.path), "--label", label)
        == 2
    )
    assert runner.calls == []


# --- drill-start -----------------------------------------------------------------------------


def drill_start_args(root: Root, drill_root: Path, artifact: str = ARTIFACT) -> list[str]:
    return [
        "drill-start",
        "--root",
        str(root.path),
        "--drill-root",
        str(drill_root),
        "--artifact",
        artifact,
    ]


def test_drill_start_creates_the_labelled_volume_only_after_inspect_fails(
    make_ctx, daily, drill_root
):
    runner = FakeRunner(
        {
            image_inspect(): ok(IMAGE_ID + "\n"),
            drill_containers(): ok(""),
            volume_inspect(): fail(1, "Error: No such volume"),
            volume_create(): ok(VOLUME + "\n"),
        }
    )

    assert run(make_ctx(runner), *drill_start_args(daily, drill_root)) == 0

    assert runner.calls == [
        list(image_inspect()),
        list(drill_containers()),
        list(volume_inspect()),
        list(volume_create()),
    ]


@pytest.mark.parametrize(
    "listing",
    [
        pytest.param(ok("c0ffee\n"), id="leftover-container"),
        pytest.param(fail(1, "Cannot connect to the Docker daemon"), id="listing-fails"),
    ],
)
def test_drill_start_refuses_while_a_drill_container_remains(
    make_ctx, daily, drill_root, listing, capsys
):
    runner = FakeRunner(
        {
            image_inspect(): ok(IMAGE_ID + "\n"),
            drill_containers(): listing,
            volume_inspect(): fail(1, "Error: No such volume"),
            volume_create(): ok(VOLUME + "\n"),
        }
    )

    assert run(make_ctx(runner), *drill_start_args(daily, drill_root)) == 2

    assert list(volume_create()) not in runner.calls
    assert not (drill_root / "drill.json").exists()
    assert "drill-end" in capsys.readouterr().err


def test_drill_start_writes_drill_json_and_the_standby_state_directory(make_ctx, daily, started):
    record = json.loads((started / "drill.json").read_text(encoding="utf-8"))

    assert record == {"volume": VOLUME, "artifact": ARTIFACT, "started_at": "2026-10-23T09:30:05Z"}
    assert (started / "standby" / "state").is_dir()
    # The restore handshake is the operator's to create, by hand.
    assert not (started / "standby" / "state" / RESTORE_MARKER_FILENAME).exists()


def test_drill_start_refuses_an_existing_volume_name(make_ctx, daily, drill_root):
    runner = FakeRunner(
        {
            image_inspect(): ok(IMAGE_ID + "\n"),
            drill_containers(): ok(""),
            volume_inspect(): drill_labels(),
        }
    )

    assert run(make_ctx(runner), *drill_start_args(daily, drill_root)) == 2

    assert list(volume_create()) not in runner.calls
    assert not (drill_root / "drill.json").exists()


def test_drill_start_refuses_a_drill_root_that_is_not_empty(make_ctx, daily, drill_root):
    drill_root.mkdir(parents=True)
    (drill_root / "leftover.txt").write_text("x", encoding="utf-8")
    runner = FakeRunner()

    assert run(make_ctx(runner), *drill_start_args(daily, drill_root)) == 2

    assert runner.calls == []
    assert sorted(p.name for p in drill_root.iterdir()) == ["leftover.txt"]


def test_drill_start_accepts_an_existing_empty_drill_root(make_ctx, daily, drill_root):
    drill_root.mkdir(parents=True)

    start_drill(make_ctx, daily, drill_root)

    assert (drill_root / "drill.json").is_file()


def protected_drill_roots(tmp_path: Path, root: Root) -> dict[str, Path]:
    other = (tmp_path / "other-root").resolve()
    other.mkdir()
    with Path(os.environ["MIST_LIVE_INSTANCES_FILE"]).open("a", encoding="utf-8") as handle:
        handle.write(f"{other}\n")
    marked = (tmp_path / "marked").resolve()
    marked.mkdir()
    (marked / MARKER).write_text("{}", encoding="utf-8")
    repo = (tmp_path / "repo").resolve()
    (repo / ".git").mkdir(parents=True)
    return {
        "instance root": root.path / "drill",
        "the root itself": root.path,
        "registered root": other / "drill",
        "marked directory": marked / "nested" / "drill",
        "git work tree": repo / "drill",
    }


@pytest.mark.parametrize(
    "case",
    ["instance root", "the root itself", "registered root", "marked directory", "git work tree"],
)
def test_drill_start_refuses_a_protected_drill_root_before_any_docker_call(
    make_ctx, tmp_path, daily, case, capsys
):
    target = protected_drill_roots(tmp_path, daily)[case]
    runner = FakeRunner()

    assert run(make_ctx(runner), *drill_start_args(daily, target)) == 2

    assert runner.calls == []
    assert "drill root" in capsys.readouterr().err


@pytest.mark.parametrize(
    "line",
    [
        "MIST_DAILY_WS_PORT=8011",
        "MIST_DAILY_BOLT_PORT=7688",
        "MIST_DAILY_NEO4J_HTTP_PORT=7690",
    ],
)
def test_drill_start_refuses_the_item_5_port_configurations_before_any_docker_call(
    make_ctx, daily, drill_root, line
):
    with daily.env.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    runner = FakeRunner()

    assert run(make_ctx(runner), *drill_start_args(daily, drill_root)) == 2
    assert runner.calls == []


def test_drill_start_refuses_a_backup_host_path_inside_the_instance_root(
    make_ctx, daily, drill_root
):
    inside = daily.path / "backups"
    (inside / ARTIFACT).mkdir(parents=True)
    (inside / ARTIFACT / "manifest.json").write_text("{}", encoding="utf-8")
    text = re.sub(
        r"^MIST_BACKUP_HOST_PATH=.*$",
        f"MIST_BACKUP_HOST_PATH={inside}",
        daily.env.read_text(encoding="utf-8"),
        flags=re.M,
    )
    daily.env.write_text(text, encoding="utf-8")
    runner = FakeRunner()

    assert run(make_ctx(runner), *drill_start_args(daily, drill_root)) == 2
    assert runner.calls == []


def test_drill_start_refuses_an_image_that_differs_from_the_release_record(
    make_ctx, daily, drill_root
):
    runner = FakeRunner({image_inspect(): ok(OTHER_ID + "\n")})

    assert run(make_ctx(runner), *drill_start_args(daily, drill_root)) == 2

    assert runner.calls == [list(image_inspect())]
    assert not drill_root.exists()


@pytest.mark.parametrize("artifact", ["../escape", "a/b", "-x", ".hidden"])
def test_drill_start_refuses_an_artifact_that_is_not_a_plain_name(
    make_ctx, daily, drill_root, artifact
):
    runner = FakeRunner()

    assert (
        run(
            make_ctx(runner),
            "drill-start",
            "--root",
            str(daily.path),
            "--drill-root",
            str(drill_root),
            f"--artifact={artifact}",
        )
        == 2
    )
    assert runner.calls == []


def test_drill_start_refuses_an_artifact_the_backup_root_does_not_hold(make_ctx, daily, drill_root):
    runner = FakeRunner()

    assert run(make_ctx(runner), *drill_start_args(daily, drill_root, "20990101T000000Z")) == 2
    assert runner.calls == []


def test_drill_start_refuses_a_release_without_the_drill_compose_file(make_ctx, daily, drill_root):
    daily.drill_file().unlink()
    runner = FakeRunner()

    assert run(make_ctx(runner), *drill_start_args(daily, drill_root)) == 2
    assert runner.calls == []


# --- The volume check: drill-restore, drill-verify, drill-end and promote --------------------


def command_args(command: str, root: Root, drill_root: Path) -> list[str]:
    return [command, "--root", str(root.path), "--drill-root", str(drill_root)]


VOLUME_CHECKED = ("drill-restore", "drill-verify", "drill-end", "promote")


@pytest.fixture
def checked(make_ctx, daily, promote_ready) -> Path:
    """A drill ready for every volume-checked command, promote included."""
    return promote_ready


@pytest.mark.parametrize("command", VOLUME_CHECKED)
@pytest.mark.parametrize(
    "labels",
    [
        pytest.param(ok("null\n"), id="no-labels"),
        pytest.param(drill_labels(role="data"), id="role-not-drill"),
        pytest.param(drill_labels(inst="rehearsal"), id="another-instance"),
        pytest.param(fail(1, "Error: No such volume"), id="volume-missing"),
    ],
)
def test_refuses_a_volume_without_the_drill_labels_before_any_compose_call(
    make_ctx, daily, checked, command, labels
):
    runner = FakeRunner({image_inspect(): ok(IMAGE_ID + "\n"), volume_inspect(): labels})
    before = tree(daily.path)

    assert run(make_ctx(runner, now=LATER), *command_args(command, daily, checked)) == 2

    assert compose_calls(runner) == []
    assert tree(daily.path) == before


@pytest.mark.parametrize("command", VOLUME_CHECKED)
def test_refuses_the_volume_daily_env_names_before_any_docker_call(
    make_ctx, daily, checked, command
):
    drill_json = checked / "drill.json"
    record = json.loads(drill_json.read_text(encoding="utf-8"))
    record["volume"] = LIVE_VOLUME
    drill_json.write_text(json.dumps(record), encoding="utf-8")
    write_drill_record(
        daily, checked, stamp=LATER_STAMP, verified_at="2026-10-23T10:30:05Z", volume=LIVE_VOLUME
    )
    runner = FakeRunner({image_inspect(): ok(IMAGE_ID + "\n")})

    assert run(make_ctx(runner, now=LATER), *command_args(command, daily, checked)) == 2

    assert runner.calls == []


@pytest.mark.parametrize("command", VOLUME_CHECKED)
@pytest.mark.parametrize(
    "projects",
    [
        pytest.param("mist-daily\n", id="instance-container"),
        pytest.param(f"{DRILL_PROJECT}\nmist-daily\n", id="drill-and-instance"),
        pytest.param("\n", id="container-outside-compose"),
    ],
)
def test_refuses_a_volume_another_container_mounts_before_any_compose_call(
    make_ctx, daily, checked, command, projects
):
    runner = FakeRunner(
        {
            image_inspect(): ok(IMAGE_ID + "\n"),
            volume_inspect(): drill_labels(),
            volume_projects(): ok(projects),
        }
    )

    assert run(make_ctx(runner, now=LATER), *command_args(command, daily, checked)) == 2

    assert compose_calls(runner) == []


@pytest.mark.parametrize("command", VOLUME_CHECKED)
def test_refuses_a_drill_root_drill_start_never_prepared(make_ctx, daily, tmp_path, command):
    runner = FakeRunner()

    assert run(make_ctx(runner), *command_args(command, daily, tmp_path / "never-started")) == 2
    assert runner.calls == []


# --- drill-restore ---------------------------------------------------------------------------


def test_drill_restore_runs_restore_without_a_token_so_restore_prints_it(
    make_ctx, daily, started, capsys
):
    refusal = "refusing to restore: no confirmation token was given. Re-run with exactly:\n    --confirm-target /drill/standby/state\n"
    runner = FakeRunner({**volume_check_responses(), restore_argv(daily): fail(2, refusal)})

    code = run(make_ctx(runner), *command_args("drill-restore", daily, started))

    assert code == 2
    assert compose_calls(runner) == [list(restore_argv(daily))]
    assert "--confirm-target /drill/standby/state" in capsys.readouterr().err


def test_drill_restore_passes_the_token_through_when_given(make_ctx, daily, started):
    argv = restore_argv(daily, "--confirm-target", TOKEN)
    runner = FakeRunner({**volume_check_responses(), argv: ok("[restore] Into ...\n")})

    code = run(
        make_ctx(runner), *command_args("drill-restore", daily, started), "--confirm-target", TOKEN
    )

    assert code == 0
    assert compose_calls(runner) == [list(argv)]


def test_drill_compose_calls_set_the_drill_variables_in_their_environment_only(
    make_ctx, daily, started
):
    runner = FakeRunner({**volume_check_responses(), restore_argv(daily): ok()})
    ctx = make_ctx(runner)

    run(ctx, *command_args("drill-restore", daily, started))

    env = runner.env_for(restore_argv(daily))
    assert env["MIST_DRILL_NEO4J_VOLUME"] == VOLUME
    assert env["MIST_DRILL_ROOT"] == started.as_posix()
    assert env["PATH"] == "/usr/bin"
    assert "MIST_DRILL_ROOT" not in ctx.environ
    assert "MIST_DRILL" not in daily.env.read_text(encoding="utf-8")


def test_nothing_creates_the_restore_marker_for_the_operator(make_ctx, daily, started):
    runner = FakeRunner({**volume_check_responses(), restore_argv(daily): fail(2)})

    run(make_ctx(runner), *command_args("drill-restore", daily, started))

    assert not (started / "standby" / "state" / RESTORE_MARKER_FILENAME).exists()


def test_drill_restore_refuses_a_drill_that_already_verified(make_ctx, daily, started, capsys):
    write_drill_record(daily, started, stamp=LATER_STAMP, verified_at="2026-10-23T10:30:05Z")
    runner = FakeRunner({**volume_check_responses(), restore_argv(daily): ok()})

    code = run(make_ctx(runner, now=LATER), *command_args("drill-restore", daily, started))

    assert code == 2
    assert compose_calls(runner) == []
    assert "drill-end" in capsys.readouterr().err


def test_drill_restore_runs_beside_a_mismatch_record(make_ctx, daily, started):
    write_drill_record(
        daily, started, stamp=LATER_STAMP, verified_at="2026-10-23T10:30:05Z", result="mismatch"
    )
    runner = FakeRunner({**volume_check_responses(), restore_argv(daily): ok()})

    assert run(make_ctx(runner, now=LATER), *command_args("drill-restore", daily, started)) == 0
    assert compose_calls(runner) == [list(restore_argv(daily))]


def test_drill_restore_runs_when_no_drill_neo4j_exists_yet(make_ctx, daily, started):
    responses = {**volume_check_responses(), restore_argv(daily): ok()}
    responses[drill_neo4j_ids()] = ok("")
    runner = FakeRunner(responses)

    assert run(make_ctx(runner), *command_args("drill-restore", daily, started)) == 0
    assert compose_calls(runner) == [list(restore_argv(daily))]


@pytest.mark.parametrize(
    "found",
    [
        pytest.param(mounts(OTHER_DRILL_VOLUME), id="another-drills-volume"),
        pytest.param(ok("[]\n"), id="no-data-mount"),
        pytest.param(fail(1, "Error: No such object"), id="inspect-fails"),
        pytest.param(ok("not json\n"), id="unreadable"),
    ],
)
def test_drill_restore_refuses_a_drill_neo4j_on_another_volume(
    make_ctx, daily, started, found, capsys
):
    responses = {**volume_check_responses(), restore_argv(daily): ok()}
    responses[mounts_inspect("c-drill")] = found
    runner = FakeRunner(responses)

    assert run(make_ctx(runner), *command_args("drill-restore", daily, started)) == 2

    assert compose_calls(runner) == []
    assert "drill-end" in capsys.readouterr().err


# --- drill-verify ----------------------------------------------------------------------------


def test_drill_verify_writes_the_record_with_the_result_and_counts(make_ctx, daily, started):
    runner = FakeRunner(
        {**volume_check_responses(), verify_argv(daily): ok(json.dumps(VERIFY_MATCH) + "\n")}
    )

    assert run(make_ctx(runner, now=LATER), *command_args("drill-verify", daily, started)) == 0

    record = json.loads((daily.ops / f"drill-{LATER_STAMP}.json").read_text(encoding="utf-8"))
    assert record["result"] == "match"
    assert record["nodes"] == VERIFY_MATCH["nodes"]
    assert record["relationships"] == VERIFY_MATCH["relationships"]
    assert record["embeddings_equal"] is True
    assert record["stores"] == VERIFY_MATCH["stores"]
    assert record["vault_files"] == VERIFY_MATCH["vault_files"]
    assert record["drill_root"] == str(started)
    assert record["volume"] == VOLUME
    assert record["artifact"] == ARTIFACT
    assert record["verified_at"] == "2026-10-23T10:30:05Z"
    assert record["exit_code"] == 0


def test_drill_verify_records_a_mismatch_and_exits_1(make_ctx, daily, started):
    mismatch = {**VERIFY_MATCH, "result": "mismatch", "embeddings_equal": False}
    runner = FakeRunner(
        {**volume_check_responses(), verify_argv(daily): fail(1, b"", json.dumps(mismatch))}
    )

    assert run(make_ctx(runner, now=LATER), *command_args("drill-verify", daily, started)) == 1

    record = json.loads((daily.ops / f"drill-{LATER_STAMP}.json").read_text(encoding="utf-8"))
    assert record["result"] == "mismatch"
    assert record["exit_code"] == 1


@pytest.mark.parametrize(
    ("found", "named"),
    [
        pytest.param(mounts(OTHER_DRILL_VOLUME), OTHER_DRILL_VOLUME, id="another-drills-volume"),
        pytest.param(fail(1, "Error: No such object"), "cannot", id="inspect-fails"),
    ],
)
def test_drill_verify_records_nothing_when_the_drill_neo4j_is_on_another_volume(
    make_ctx, daily, started, found, named, capsys
):
    responses = {**volume_check_responses(), verify_argv(daily): ok(json.dumps(VERIFY_MATCH))}
    responses[mounts_inspect("c-drill")] = found
    runner = FakeRunner(responses)

    assert run(make_ctx(runner, now=LATER), *command_args("drill-verify", daily, started)) == 1

    assert list(daily.ops.glob("drill-*.json")) == []
    assert compose_calls(runner) == [list(verify_argv(daily))]
    assert named in capsys.readouterr().err


def test_drill_verify_records_nothing_when_no_drill_neo4j_remains(make_ctx, daily, started):
    responses = {**volume_check_responses(), verify_argv(daily): ok(json.dumps(VERIFY_MATCH))}
    responses[drill_neo4j_ids()] = ok("")
    runner = FakeRunner(responses)

    assert run(make_ctx(runner, now=LATER), *command_args("drill-verify", daily, started)) == 1
    assert list(daily.ops.glob("drill-*.json")) == []


def test_drill_verify_writes_no_record_when_verify_prints_no_result(make_ctx, daily, started):
    runner = FakeRunner(
        {**volume_check_responses(), verify_argv(daily): fail(2, "[verify] REFUSED")}
    )

    assert run(make_ctx(runner, now=LATER), *command_args("drill-verify", daily, started)) == 2

    assert list(daily.ops.glob("drill-*.json")) == []


# --- drill-end -------------------------------------------------------------------------------


def test_drill_end_removes_the_drill_containers_and_keeps_the_volume(make_ctx, daily, started):
    runner = FakeRunner({**volume_check_responses(), drill_rm(daily): ok()})

    assert run(make_ctx(runner), *command_args("drill-end", daily, started)) == 0

    assert compose_calls(runner) == [list(drill_rm(daily))]
    assert runner.env_for(drill_rm(daily))["MIST_DRILL_NEO4J_VOLUME"] == VOLUME
    assert (started / "drill.json").is_file()


# --- Two drills of one instance --------------------------------------------------------------


class DrillHost:
    """The Docker state two drills of one instance share: their volumes and one container.

    Every drill runs in the one project `mist-daily-drill`, whose `mist-neo4j-dev` takes its
    volume when compose creates it, at the first `run`; later runs reuse it whatever volume
    they name.
    """

    CONTAINER = "c-drill"

    def __init__(self):
        self.volumes: set[str] = set()
        self.container_volume: str | None = None

    def __call__(self, argv: list[str], env) -> instance.CommandResult | None:
        key = tuple(argv)
        if key == image_inspect():
            return ok(IMAGE_ID + "\n")
        if key[:3] == ("docker", "volume", "inspect"):
            if key[-1] in self.volumes:
                return drill_labels()
            return fail(1, "Error: No such volume")
        if key[:3] == ("docker", "volume", "create"):
            self.volumes.add(key[-1])
            return ok(key[-1] + "\n")
        if key in (drill_containers(), drill_neo4j_ids()):
            return ok(f"{self.CONTAINER}\n" if self.container_volume else "")
        if key == mounts_inspect(self.CONTAINER) and self.container_volume:
            return mounts(self.container_volume)
        if key[:4] == ("docker", "ps", "-a", "--filter") and key[5:6] == ("--format",):
            mounted = key[4] == f"volume={self.container_volume}"
            return ok(f"{DRILL_PROJECT}\n" if mounted else "")
        if key[:2] == ("docker", "compose") and "run" in key:
            if self.container_volume is None:
                self.container_volume = env["MIST_DRILL_NEO4J_VOLUME"]
            if "scripts.backup.verify" in key:
                return ok(json.dumps(VERIFY_MATCH) + "\n")
            return ok("[restore] done\n")
        return None


@dataclass
class TwoDrills:
    root: Root
    a: Path
    b: Path
    volume_a: str
    volume_b: str
    host: DrillHost


@pytest.fixture
def two_drills(make_ctx, daily, tmp_path) -> TwoDrills:
    """drill-start A, then drill-start B a minute later, before either restores."""
    host = DrillHost()
    a = (tmp_path / "drills" / "a").resolve()
    b = (tmp_path / "drills" / "b").resolve()
    second = START + timedelta(minutes=1)
    for drill, now in ((a, START), (b, second)):
        code = run(make_ctx(FakeRunner(answer=host), now=now), *drill_start_args(daily, drill))
        assert code == 0
    return TwoDrills(
        daily, a, b, VOLUME, f"mist-daily-drill-{second.strftime('%Y%m%dT%H%M%SZ')}", host
    )


def drill_restore(make_ctx, drills: TwoDrills, drill: Path, now: datetime = LATER):
    runner = FakeRunner(answer=drills.host)
    args = [*command_args("drill-restore", drills.root, drill), "--confirm-target", TOKEN]
    return run(make_ctx(runner, now=now), *args), runner


def drill_verify(make_ctx, drills: TwoDrills, drill: Path, now: datetime = LATER):
    runner = FakeRunner(answer=drills.host)
    return run(make_ctx(runner, now=now), *command_args("drill-verify", drills.root, drill)), runner


def test_a_second_drill_never_restores_into_the_first_drills_neo4j(make_ctx, two_drills, capsys):
    code_a, _ = drill_restore(make_ctx, two_drills, two_drills.a)
    assert code_a == 0
    assert two_drills.host.container_volume == two_drills.volume_a
    capsys.readouterr()

    code_b, runner_b = drill_restore(make_ctx, two_drills, two_drills.b)

    assert code_b == 2
    assert compose_calls(runner_b) == []
    assert two_drills.volume_a in capsys.readouterr().err


def test_a_second_drill_verified_on_the_first_drills_neo4j_records_nothing(
    make_ctx, two_drills, capsys
):
    drill_restore(make_ctx, two_drills, two_drills.a)
    capsys.readouterr()

    code, runner = drill_verify(make_ctx, two_drills, two_drills.b)

    assert code == 1
    assert compose_calls(runner) != []
    assert list(two_drills.root.ops.glob("drill-*.json")) == []
    assert two_drills.volume_a in capsys.readouterr().err
    restore_into_standby(two_drills.b)
    promote_runner = FakeRunner(answer=two_drills.host)
    assert promote(make_ctx, two_drills.root, two_drills.b, promote_runner) == 2


def test_a_first_drill_never_restores_into_a_verified_second_drill(make_ctx, two_drills, capsys):
    assert drill_restore(make_ctx, two_drills, two_drills.b)[0] == 0
    assert drill_verify(make_ctx, two_drills, two_drills.b)[0] == 0
    assert len(list(two_drills.root.ops.glob("drill-*.json"))) == 1
    capsys.readouterr()

    later = LATER + timedelta(minutes=5)
    code, runner = drill_restore(make_ctx, two_drills, two_drills.a, now=later)

    assert code == 2
    assert compose_calls(runner) == []
    assert two_drills.volume_b in capsys.readouterr().err


# --- promote: refusals -----------------------------------------------------------------------


def promote(make_ctx, root: Root, drill_root: Path, runner: FakeRunner) -> int:
    return run(
        make_ctx(runner, now=LATER + timedelta(minutes=5)),
        *command_args("promote", root, drill_root),
    )


def test_promote_refuses_without_a_drill_record(make_ctx, daily, started):
    restore_into_standby(started)
    runner = FakeRunner()
    before = tree(daily.path)

    assert promote(make_ctx, daily, started, runner) == 2

    assert runner.calls == []
    assert tree(daily.path) == before


@pytest.mark.parametrize(
    "changes",
    [
        pytest.param({"result": "mismatch"}, id="mismatch"),
        pytest.param({"verified_at": "2026-10-23T09:00:00Z"}, id="older-than-drill-json"),
        pytest.param({"verified_at": "2026-10-23T09:30:05Z"}, id="same-second-as-drill-json"),
        pytest.param({"drill_root": "/placeholder/other-drill"}, id="another-drill-root"),
        pytest.param({"volume": "mist-daily-drill-other"}, id="another-volume"),
    ],
)
def test_promote_refuses_without_a_match_newer_than_drill_json(make_ctx, daily, started, changes):
    restore_into_standby(started)
    fields = {"verified_at": "2026-10-23T10:30:05Z", **changes}
    write_drill_record(daily, started, stamp=LATER_STAMP, **fields)
    runner = FakeRunner()
    before = tree(daily.path)

    assert promote(make_ctx, daily, started, runner) == 2

    assert runner.calls == []
    assert tree(daily.path) == before


def test_promote_takes_a_match_beside_an_older_mismatch(make_ctx, daily, promote_ready):
    write_drill_record(
        daily,
        promote_ready,
        stamp="20261023T100000Z",
        verified_at="2026-10-23T10:00:00Z",
        result="mismatch",
    )
    runner = FakeRunner({image_inspect(): ok(OTHER_ID + "\n")})

    # The match is found, so promote gets as far as its image check, which refuses.
    assert promote(make_ctx, daily, promote_ready, runner) == 2
    assert runner.calls == [list(image_inspect())]


def test_promote_refuses_a_drill_root_on_another_drive(make_ctx, daily, promote_ready, monkeypatch):
    monkeypatch.setattr(instance, "_same_drive", lambda _a, _b: False)
    runner = FakeRunner()
    before = tree(daily.path)

    assert promote(make_ctx, daily, promote_ready, runner) == 2

    assert runner.calls == []
    assert tree(daily.path) == before


def test_same_drive_holds_for_two_directories_on_one_filesystem(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    assert instance._same_drive(tmp_path / "a", tmp_path / "b")


def test_promote_refuses_a_standby_without_both_directories(make_ctx, daily, started):
    write_drill_record(daily, started, stamp=LATER_STAMP, verified_at="2026-10-23T10:30:05Z")
    runner = FakeRunner()

    assert promote(make_ctx, daily, started, runner) == 2
    assert runner.calls == []


# --- promote: the sequence -------------------------------------------------------------------


def promote_responses(root: Root, *, still_mounted: bool = False, data: str = VOLUME) -> dict:
    return {
        image_inspect(): ok(IMAGE_ID + "\n"),
        **volume_check_responses(),
        drill_rm(root): ok(),
        volume_ids(): ok("c0ffee\n" if still_mounted else ""),
        instance_stop(root): ok(),
        instance_rm(root): ok(),
        **up_responses(root),
        instance_neo4j_ids(): ok("id-neo4j\n"),
        mounts_inspect("id-neo4j"): mounts(data),
    }


def test_promote_exits_2_before_stop_when_the_drill_neo4j_still_mounts_the_volume(
    make_ctx, daily, promote_ready, capsys
):
    runner = FakeRunner(promote_responses(daily, still_mounted=True))
    before = tree(daily.path)
    env_before = daily.env.read_bytes()

    assert promote(make_ctx, daily, promote_ready, runner) == 2

    assert list(instance_stop(daily)) not in runner.calls
    assert runner.calls[-1] == list(volume_ids())
    assert tree(daily.path) == before
    assert daily.env.read_bytes() == env_before
    assert (promote_ready / "standby" / "state").is_dir()
    assert "nothing moved" in capsys.readouterr().err


def test_promote_runs_its_steps_in_the_specified_order(make_ctx, daily, promote_ready, monkeypatch):
    events: list = []
    real_rename = os.rename
    real_unlink = os.unlink
    real_write = instance._write_atomic

    def rename(source, destination):
        events.append(("rename", Path(source), Path(destination)))
        real_rename(source, destination)

    def unlink(path, *args, **kwargs):
        events.append(("unlink", Path(path)))
        real_unlink(path, *args, **kwargs)

    def write(path, data):
        events.append(("write", Path(path)))
        real_write(path, data)

    monkeypatch.setattr(os, "rename", rename)
    monkeypatch.setattr(os, "unlink", unlink)
    monkeypatch.setattr(instance, "_write_atomic", write)
    runner = FakeRunner(promote_responses(daily), events=events)

    assert promote(make_ctx, daily, promote_ready, runner) == 0

    root = daily.path
    retired = root / "retired" / "20261023T103505Z"
    standby = promote_ready / "standby"
    steps = [
        event
        for event in events
        if event[0] != "run"
        or event[1][:2] == ["docker", "compose"]
        or event[1][:3] == ["docker", "ps", "-a"]
    ]
    assert steps[:5] == [
        ("run", list(volume_projects())),
        ("run", list(drill_rm(daily))),
        ("run", list(volume_ids())),
        ("run", list(instance_stop(daily))),
        ("run", list(instance_rm(daily))),
    ]
    assert steps[5:13] == [
        ("rename", root / "state", retired / "state"),
        ("rename", root / "mist-memory", retired / "mist-memory"),
        ("rename", standby / "state", root / "state"),
        ("rename", standby / "mist-memory", root / "mist-memory"),
        ("unlink", root / "state" / RESTORE_MARKER_FILENAME),
        ("write", root / "state" / MARKER),
        ("write", root / "mist-memory" / MARKER),
        ("write", daily.env),
    ]
    assert steps[13:] == [
        ("run", [*daily.base(), "up", "-d", "mist-neo4j", "mist-llm", "mist-backend"]),
        ("run", list(instance_neo4j_ids())),
    ]
    assert runner.calls[-1] == list(mounts_inspect("id-neo4j"))


@pytest.mark.parametrize(
    ("found", "named"),
    [
        pytest.param(mounts(LIVE_VOLUME), LIVE_VOLUME, id="the-old-volume"),
        pytest.param(ok("[]\n"), "no /data", id="no-data-mount"),
        pytest.param(fail(1, "Error: No such object"), "cannot", id="inspect-fails"),
    ],
)
def test_promote_exits_1_when_mist_neo4j_comes_up_on_another_volume(
    make_ctx, daily, promote_ready, found, named, capsys
):
    responses = promote_responses(daily)
    responses[mounts_inspect("id-neo4j")] = found
    runner = FakeRunner(responses)

    assert promote(make_ctx, daily, promote_ready, runner) == 1

    err = capsys.readouterr().err
    assert named in err
    assert VOLUME in err


def test_promote_exits_1_when_no_mist_neo4j_container_is_found(
    make_ctx, daily, promote_ready, capsys
):
    responses = promote_responses(daily)
    responses[instance_neo4j_ids()] = ok("")
    runner = FakeRunner(responses)

    assert promote(make_ctx, daily, promote_ready, runner) == 1


def test_promote_exits_1_with_nothing_moved_when_rm_fails(make_ctx, daily, promote_ready, capsys):
    responses = promote_responses(daily)
    responses[instance_rm(daily)] = fail(1)
    runner = FakeRunner(responses)
    before = tree(daily.path)

    assert promote(make_ctx, daily, promote_ready, runner) == 1

    assert tree(daily.path) == before
    assert "nothing moved" in capsys.readouterr().err


def test_promote_moves_the_old_directories_aside_and_the_standby_in(make_ctx, daily, promote_ready):
    runner = FakeRunner(promote_responses(daily))

    assert promote(make_ctx, daily, promote_ready, runner) == 0

    retired = daily.path / "retired" / "20261023T103505Z"
    assert (retired / "state" / "event_store.db").read_bytes() == b"old-store"
    assert (retired / "mist-memory" / "old.md").is_file()
    assert (daily.path / "state" / "event_store.db").read_bytes() == b"restored-store"
    assert (daily.path / "mist-memory" / "restored.md").is_file()
    assert not (promote_ready / "standby" / "state").exists()
    assert not (promote_ready / "standby" / "mist-memory").exists()


def test_promote_deletes_the_promoted_restore_marker_and_marks_the_new_directories(
    make_ctx, daily, promote_ready
):
    runner = FakeRunner(promote_responses(daily))

    assert promote(make_ctx, daily, promote_ready, runner) == 0

    assert not (daily.path / "state" / RESTORE_MARKER_FILENAME).exists()
    root_marker = json.loads((daily.path / MARKER).read_text(encoding="utf-8"))
    for sub in ("state", "mist-memory"):
        assert json.loads((daily.path / sub / MARKER).read_text(encoding="utf-8")) == root_marker


def test_promote_rewrites_only_the_volume_line(make_ctx, daily, promote_ready):
    before = daily.env.read_text(encoding="utf-8").splitlines()
    runner = FakeRunner(promote_responses(daily))

    assert promote(make_ctx, daily, promote_ready, runner) == 0

    after = daily.env.read_text(encoding="utf-8").splitlines()
    assert len(after) == len(before)
    changed = [(old, new) for old, new in zip(before, after, strict=True) if old != new]
    assert changed == [
        (f"MIST_DAILY_NEO4J_VOLUME={LIVE_VOLUME}", f"MIST_DAILY_NEO4J_VOLUME={VOLUME}")
    ]


def test_promote_removes_drill_containers_without_removing_volumes(make_ctx, daily, promote_ready):
    runner = FakeRunner(promote_responses(daily))

    assert promote(make_ctx, daily, promote_ready, runner) == 0

    for call in runner.calls:
        assert "-v" not in call and "--volumes" not in call and "down" not in call
        assert call[:3] != ["docker", "volume", "rm"]


def is_string(node: ast.AST, *values: str) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in values


def test_no_argv_in_the_module_can_delete_a_volume():
    """The drill needs `volume inspect`, `volume create` and the drill project's `rm -s -f`.

    So in every list or tuple display and every call's positional arguments, a `"volume"` must
    be followed directly by `"inspect"` or `"create"`; the one exception is the sole argument
    of a `.get(...)`. No flag that removes volumes, no `prune` and no `down` in a list.
    """
    tree_ = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    constants = {
        node.value
        for node in ast.walk(tree_)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert not constants & {"-v", "--volumes", "prune"}
    offenders = []
    for node in ast.walk(tree_):
        if isinstance(node, ast.List | ast.Tuple):
            sequence = node.elts
            assert not any(is_string(elt, "down") for elt in sequence), node.lineno
        elif isinstance(node, ast.Call):
            sole_get = (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and len(node.args) == 1
                and not node.keywords
            )
            sequence = [] if sole_get else node.args
        else:
            continue
        for index, element in enumerate(sequence):
            following = sequence[index + 1] if index + 1 < len(sequence) else None
            if is_string(element, "volume") and not (
                following is not None and is_string(following, "inspect", "create")
            ):
                offenders.append(node.lineno)
    assert offenders == [], f"`volume` not followed by inspect or create at lines {offenders}"


def test_promote_refuses_a_daily_env_without_one_volume_line(make_ctx, daily, promote_ready):
    with daily.env.open("a", encoding="utf-8") as handle:
        handle.write("MIST_DAILY_NEO4J_VOLUME=second\n")
    runner = FakeRunner()

    assert promote(make_ctx, daily, promote_ready, runner) == 2
    assert runner.calls == []


def test_promote_names_the_state_when_up_fails_afterwards(make_ctx, daily, promote_ready, capsys):
    responses = promote_responses(daily)
    responses[(*daily.base(), "up", "-d", "mist-neo4j", "mist-llm", "mist-backend")] = fail(1)
    runner = FakeRunner(responses)

    assert promote(make_ctx, daily, promote_ready, runner) == 1

    err = capsys.readouterr().err
    assert "retired" in err and VOLUME in err


# --- Documentation ---------------------------------------------------------------------------


def section(path: Path, title: str) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(rf"^## {re.escape(title)}\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    assert match is not None, f"{path.name} has no '## {title}' section"
    return match.group(1)


def test_readme_commands_section_lists_the_drill_commands_and_the_baseline():
    commands = section(README, "Commands")
    for command in (
        "drill-start",
        "drill-restore",
        "drill-verify",
        "drill-end",
        "promote",
        "accept-backup-baseline",
        "--accept-suspect",
    ):
        assert f"`{command}" in commands or command in commands, command


def test_readme_reading_health_gives_backup_suspect_and_its_remedy():
    health = " ".join(section(README, "Reading /health").split())
    assert "`backup_suspect`" in health
    assert "<backup root>/suspect/<label>/manifest.json" in health
    assert "ops/backup-status.json" in health
    assert "accept-backup-baseline --label <label>" in health


def code_lines(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    blocks = re.findall(r"^```[a-z]*\n(.*?)^```", text, re.M | re.S)
    return [line.strip() for block in blocks for line in block.splitlines() if line.strip()]


def test_restore_runbook_gives_instance_py_commands_only():
    lines = code_lines(RESTORE_MD)
    assert lines
    assert all("instance.py " in line for line in lines), lines


def test_restore_runbook_runs_the_drill_steps_in_order():
    lines = code_lines(RESTORE_MD)
    order = ["drill-start", "drill-restore", "--confirm-target", "drill-verify", "drill-end"]
    positions = [next(i for i, line in enumerate(lines) if step in line) for step in order]
    assert positions == sorted(positions)
    promote_at = next(i for i, line in enumerate(lines) if " promote " in line)
    assert promote_at > positions[3]


def test_restore_runbook_keeps_every_gate_and_never_runs_the_drill_file_by_hand():
    text = " ".join(RESTORE_MD.read_text(encoding="utf-8").split())
    assert "Never run `restore-drill.yml` by hand" in text
    assert f"standby/state/{RESTORE_MARKER_FILENAME}" in text
    assert "by hand" in text
    assert "gaming-mode on" in text and "gaming-mode off" in text
    assert "/health" in text


def test_restore_runbook_says_one_drill_at_a_time_and_what_refuses_otherwise():
    text = " ".join(RESTORE_MD.read_text(encoding="utf-8").split())
    assert "one drill per instance at a time" in text.lower()
    for refusal in (
        "`drill-start` refuses while an earlier drill's containers remain",
        "`drill-restore` refuses while another drill's Neo4j container remains",
        "`drill-restore` refuses a drill that has already verified",
    ):
        assert refusal in text, refusal


def test_backup_readme_section_6_describes_the_scheduler():
    text = " ".join(section(BACKUP_README, "6. Scheduled backups").split())
    for phrase in (
        "python -m scripts.backup.scheduler",
        "backup-status.json",
        "MIST_BACKUP_MIN_NODE_RATIO",
        "suspect/",
        "--accept-suspect",
        "arm-backups",
        "rehearse first, arm second",
    ):
        assert phrase in text, phrase


def test_backup_readme_section_4_points_to_the_compose_drill():
    assert "deploy/daily/RESTORE.md" in section(BACKUP_README, "4. The restore rehearsal")
