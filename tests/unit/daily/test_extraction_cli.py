"""Tests for night extraction in `instance.py` (plan v4, T2).

`extraction on|off|auto|apply|status`, the window, the paused marker, and what `up`, `down`,
`switch`, `status`, `init` and `gaming-mode` do for the two extraction services. A stateful fake
Docker models the containers, so a test reads what an apply left behind as well as the argument
lists it ran; a fake clock drives every sleep and the UTC time, and `Context.local_zone` puts the
window in Europe/London, so midnight and both daylight-saving changes are driven exactly.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from scripts.daily import instance

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "daily" / "instance.py"
LONDON = ZoneInfo("Europe/London")
TAG = "daily-2026.11.15"
OLD_TAG = "daily-2026.11.01"
SHA = "0123456789abcdef0123456789abcdef01234567"
IMAGE_ID = "sha256:" + "a" * 64
EXTRACTION_IMAGE_ID = "sha256:" + "e" * 64
OLD_EXTRACTION_IMAGE_ID = "sha256:" + "d" * 64
OTHER_ID = "sha256:" + "b" * 64
LLM_IMAGE_ID = "sha256:" + "1" * 64
MARKER = ".mist-live-instance"
# Values a status line must never print.
MODEL_HASH_VALUE = "gpt-oss-20b-mxfp4-gptoss-v1-sentinel"
DEVICE_VALUE = "rtx-4070-super-sentinel"
# 01:00 GMT on a November night: inside the default window.
NIGHT = datetime(2026, 11, 16, 1, 0, 0, tzinfo=UTC)
# 12:00 GMT: outside it.
DAY = datetime(2026, 11, 16, 12, 0, 0, tzinfo=UTC)
LLM = "mist-extraction-llm"
SVC = "mist-extraction"
EXTRACTION_FORMAT = (
    "{{.Name}} {{.Image}} {{.State.Status}} "
    "{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}} {{json .Config.Labels}}"
)
CORE_FORMAT = "{{.Name}} {{.Image}} {{json .Config.Labels}}"
DOCKER_INFO = ("docker", "info", "--format", "{{.ServerVersion}}")
NETWORK_INSPECT = ("docker", "network", "inspect", "mist-llm-share")
# How the CLI asks whether the daily project's mist-backend runs: scoped by project, never by
# container name, so the development stack's mist-backend is never read.
PROJECT_RUNNING_PS = [
    "docker",
    "compose",
    "-p",
    "mist-daily",
    "ps",
    "--status",
    "running",
    "--format",
    "json",
]


def ok(stdout: bytes | str = b"") -> instance.CommandResult:
    if isinstance(stdout, str):
        stdout = stdout.encode()
    return instance.CommandResult(0, stdout, b"")


def fail(code: int = 1, stderr: bytes = b"boom") -> instance.CommandResult:
    return instance.CommandResult(code, b"", stderr)


# --- The instance root -----------------------------------------------------------------------


@dataclass
class Root:
    path: Path

    @property
    def env(self) -> Path:
        return self.path / "daily.env"

    @property
    def ops(self) -> Path:
        return self.path / "ops"

    @property
    def logs(self) -> Path:
        return self.path / "logs"

    def compose_file(self, tag: str = TAG) -> Path:
        return self.path / "release" / tag / "deploy" / "daily" / "compose.yml"

    def base(self, tag: str = TAG) -> list[str]:
        return [
            "docker",
            "compose",
            "-p",
            "mist-daily",
            "-f",
            str(self.compose_file(tag)),
            "--env-file",
            str(self.env),
        ]

    def dc(self, tag: str = TAG) -> list[str]:
        return [*self.base(tag), "--profile", "extraction"]

    @property
    def paused(self) -> Path:
        return self.ops / "extraction-paused"

    def paused_payload(self) -> dict | None:
        if not self.paused.exists():
            return None
        return json.loads(self.paused.read_text(encoding="utf-8"))

    def set_mode(self, mode: str) -> None:
        (self.ops / "extraction-mode").write_text(mode + "\n", encoding="utf-8")

    def mode(self) -> str | None:
        path = self.ops / "extraction-mode"
        return path.read_text(encoding="utf-8").strip() if path.exists() else None

    def set_env_line(self, line: str) -> None:
        with self.env.open("a", encoding="utf-8", newline="") as handle:
            handle.write(line + "\n")

    def yield_gpu(self) -> None:
        (self.ops / "gpu-yielded").write_text(
            json.dumps({"v": 1, "since": "2026-11-16T00:00:00Z", "by": "manual"}),
            encoding="utf-8",
        )

    def window_log(self) -> list[str]:
        path = self.logs / "extraction-window.log"
        return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


def add_release(root: Root, tag: str, extraction_image_id: str | None = EXTRACTION_IMAGE_ID):
    compose = root.compose_file(tag)
    compose.parent.mkdir(parents=True, exist_ok=True)
    compose.write_text("name: mist-daily\n", encoding="utf-8")
    record = {"tag": tag, "commit": SHA, "image_id": IMAGE_ID, "built_at": "2026-11-15T09:00:00Z"}
    if extraction_image_id is not None:
        record["extraction_image_id"] = extraction_image_id
    (root.ops / f"release-{tag}.json").write_text(json.dumps(record), encoding="utf-8")


@pytest.fixture(autouse=True)
def registry(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "registry" / "live-instances"
    monkeypatch.setenv("MIST_LIVE_INSTANCES_FILE", str(path))
    return path


@pytest.fixture
def home(tmp_path) -> Path:
    path = tmp_path / "home"
    path.mkdir()
    return path


def make_root(tmp_path: Path, *, configured: bool = True, tag: str = TAG) -> Root:
    path = (tmp_path / "instances" / "daily").resolve()
    for sub in ("state", "mist-memory", "logs", "ops", "models", "release", "retired", "bin"):
        (path / sub).mkdir(parents=True, exist_ok=True)
    marker = json.dumps({"v": 1, "instance": "daily", "created_at": "2026-11-01T00:00:00Z"})
    for directory in (path, path / "state", path / "mist-memory"):
        (directory / MARKER).write_text(marker + "\n", encoding="utf-8")
    lines = [
        f"MIST_RELEASE_TAG={tag}",
        f"MIST_DAILY_ROOT={path.as_posix()}",
        "MIST_INSTANCE_NAME=daily",
        "NEO4J_PASSWORD=<neo4j-password>",
    ]
    if configured:
        lines += [
            f"EXTRACTION_MODEL_HASH={MODEL_HASH_VALUE}",
            f"EXTRACTION_LLM_DEVICE={DEVICE_VALUE}",
        ]
    (path / "daily.env").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")
    root = Root(path)
    add_release(root, tag)
    return root


@pytest.fixture
def daily(tmp_path) -> Root:
    return make_root(tmp_path)


# --- The fake Docker -------------------------------------------------------------------------


@dataclass
class Box:
    """One container the fake knows: its state and the labels compose would give it."""

    cid: str
    service: str
    running: bool
    health: str
    image: str
    config_hash: str
    config_file: str
    polls_left: int = 0

    def labels(self, project: str) -> dict:
        return {
            "com.docker.compose.project": project,
            "com.docker.compose.service": self.service,
            "com.docker.compose.project.config_files": self.config_file,
            "com.docker.compose.config-hash": self.config_hash,
        }


@dataclass
class Snapshot:
    """What the root held when one command ran."""

    paused: dict | None
    yielded: bool
    env: str


@dataclass
class FakeDocker:
    """A Docker that models the project's containers for the commands extraction uses.

    Commands outside the model are answered from `canned`, by exact argument list; anything
    else fails the test. Each call records a snapshot of the root, so a test asserts what was on
    disk at the moment a container was stopped, removed or started.
    """

    root: Root
    events: list = field(default_factory=list)
    local_image: str | None = EXTRACTION_IMAGE_ID
    hashes: dict = field(default_factory=lambda: {LLM: "hash-llm-1", SVC: "hash-svc-1"})
    polls_to_healthy: int = 1
    never_healthy: bool = False
    fail_up: set = field(default_factory=set)
    # Whether the daily project's mist-backend runs, as `compose -p mist-daily ps` reports it.
    backend_running: bool = True
    canned: dict = field(default_factory=dict)
    on_call: object = None
    boxes: dict = field(default_factory=dict)
    calls: list = field(default_factory=list)
    snapshots: list = field(default_factory=list)
    serial: itertools.count = field(default_factory=lambda: itertools.count(1))

    # Setup.

    def add(self, service: str, *, running: bool = True, health: str = "healthy", **overrides):
        values = {
            "image": EXTRACTION_IMAGE_ID if service == SVC else LLM_IMAGE_ID,
            "config_hash": self.hashes.get(service, "hash-core"),
            "config_file": str(self.root.compose_file()),
        }
        values.update(overrides)
        cid = f"{service}-{next(self.serial)}"
        self.boxes[cid] = Box(cid, service, running, health if running else "", **values)
        return cid

    def add_both(self, **overrides) -> None:
        self.add(LLM, **overrides)
        self.add(SVC, **overrides)

    # Reading.

    def of(self, service: str) -> list[Box]:
        return [box for box in self.boxes.values() if box.service == service]

    def extraction_running(self) -> bool:
        both = [*self.of(LLM), *self.of(SVC)]
        return len(both) == 2 and all(box.running and box.health == "healthy" for box in both)

    def extraction_absent(self) -> bool:
        return not self.of(LLM) and not self.of(SVC)

    def ids(self) -> set[str]:
        return {box.cid for box in self.boxes.values() if box.service in (LLM, SVC)}

    def compose_tails(self) -> list[list[str]]:
        """The extraction compose calls, without the `docker compose ... --profile` prefix."""
        return [
            call[10:]
            for call in self.calls
            if call[:4] == ["docker", "compose", "-p", "mist-daily"]
            and call[8:10] == ["--profile", "extraction"]
        ]

    def acted(self) -> bool:
        """Whether any call started, stopped or removed a container."""
        return any(tail[:1] in (["up"], ["stop"], ["rm"]) for tail in self.compose_tails()) or any(
            call[:3] == ["docker", "rm", "-f"] or call[:2] == ["docker", "stop"]
            for call in self.calls
        )

    # The runner.

    def __call__(self, argv, *, stdin=None):
        argv = list(argv)
        self.calls.append(argv)
        self.events.append(("run", argv))
        self.snapshots.append(
            Snapshot(
                self.root.paused_payload(),
                (self.root.ops / "gpu-yielded").exists(),
                self.root.env.read_text(encoding="utf-8"),
            )
        )
        if self.on_call is not None:
            self.on_call(argv)
        key = tuple(argv)
        if key in self.canned:
            answer = self.canned[key]
            if isinstance(answer, list):
                return answer.pop(0) if len(answer) > 1 else answer[0]
            return answer
        if argv[:4] == ["docker", "compose", "-p", "mist-daily"] and argv[8:10] == [
            "--profile",
            "extraction",
        ]:
            return self._compose(argv[5], argv[10:])
        if argv[:5] == ["docker", "ps", "--all", "--quiet", "--no-trunc"]:
            return self._ps(argv[5:])
        if argv[:3] == ["docker", "inspect", "--format"] and len(argv) == 5:
            return self._inspect(argv[3], argv[4])
        if argv[:5] == ["docker", "image", "inspect", "--format", "{{.Id}}"] and argv[5].startswith(
            "mist-extraction:"
        ):
            if self.local_image is None:
                return fail(1, b"No such image")
            return ok(self.local_image + "\n")
        if argv[:3] == ["docker", "rm", "-f"] and len(argv) > 3:
            for cid in argv[3:]:
                if cid not in self.boxes:
                    return fail(1, f"No such container: {cid}".encode())
                del self.boxes[cid]
            return ok()
        if argv == PROJECT_RUNNING_PS:
            row = {"Service": "mist-backend", "State": "running"}
            return ok(json.dumps(row) + "\n" if self.backend_running else "")
        if argv[:2] == ["docker", "stop"] and argv[2:-1] in ([], ["-t", "10"]):
            box = self.boxes.get(argv[-1])
            if box is None:
                return fail(1, f"No such container: {argv[-1]}".encode())
            box.running, box.health = False, ""
            return ok()
        if argv[:4] == ["docker", "logs", "--tail", "200"] and len(argv) == 5:
            box = self.boxes.get(argv[4])
            if box is None:
                return fail(1, f"No such container: {argv[4]}".encode())
            return instance.CommandResult(0, f"out of {box.cid}\n".encode(), b"err\n")
        raise AssertionError(f"unexpected command: {argv}")

    def _ps(self, rest: list[str]) -> instance.CommandResult:
        wanted: dict[str, str] = {}
        for flag, value in zip(rest[::2], rest[1::2]):
            assert flag == "--filter" and value.startswith("label="), rest
            key, _, label_value = value[len("label=") :].partition("=")
            wanted[key] = label_value
        found = [
            box.cid
            for box in self.boxes.values()
            if all(box.labels("mist-daily").get(k) == v for k, v in wanted.items())
        ]
        return ok("".join(f"{cid}\n" for cid in found))

    def _inspect(self, fmt: str, cid: str) -> instance.CommandResult:
        box = self.boxes.get(cid)
        if box is None:
            return fail(1, b"No such object")
        labels = json.dumps(box.labels("mist-daily"))
        name = f"/mist-daily-{box.service}-1"
        if fmt == EXTRACTION_FORMAT:
            status = "running" if box.running else "exited"
            return ok(f"{name} {box.image} {status} {box.health or 'none'} {labels}\n")
        if fmt == CORE_FORMAT:
            return ok(f"{name} {box.image} {labels}\n")
        raise AssertionError(f"unexpected inspect format {fmt!r}")

    def _compose(self, compose_file: str, rest: list[str]) -> instance.CommandResult:
        services = (LLM, SVC)
        if rest[:2] == ["config", "--hash"] and len(rest) == 3 and rest[2] in services:
            return ok(f"{rest[2]} {self.hashes[rest[2]]}\n")
        if rest[:2] == ["up", "-d"] and len(rest) == 3 and rest[2] in services:
            service = rest[2]
            if service in self.fail_up:
                return fail(1, f"up {service} failed".encode())
            image = self.local_image if service == SVC else LLM_IMAGE_ID
            current = [
                box
                for box in self.of(service)
                if box.config_hash == self.hashes[service]
                and box.image == image
                and box.config_file == compose_file
            ]
            if current:
                box = current[0]
                if not box.running:
                    box.running, box.health = True, "starting"
                    box.polls_left = self.polls_to_healthy
                return ok()
            for box in self.of(service):
                del self.boxes[box.cid]
            cid = self.add(
                service,
                health="starting",
                image=image,
                config_hash=self.hashes[service],
                config_file=compose_file,
            )
            self.boxes[cid].polls_left = self.polls_to_healthy
            return ok()
        if rest[:4] == ["ps", "--all", "--format", "json"] and rest[4:] == [LLM, SVC]:
            rows = []
            for service in services:
                for box in self.of(service):
                    if box.running and box.health == "starting":
                        box.polls_left -= 1
                        if box.polls_left <= 0 and not self.never_healthy:
                            box.health = "healthy"
                    rows.append(
                        {
                            "Service": service,
                            "State": "running" if box.running else "exited",
                            "Health": box.health,
                        }
                    )
            return ok("\n".join(json.dumps(row) for row in rows))
        if rest in (["stop", "--timeout", "10", SVC], ["stop", LLM]):
            for box in self.of(rest[-1]):
                box.running, box.health = False, ""
            return ok()
        if rest[:4] == ["logs", "--no-color", "--tail", "200"] and rest[4:5] in ([LLM], [SVC]):
            return ok(f"last lines of {rest[4]}\n")
        if rest == ["rm", "-f", SVC, LLM]:
            for service in services:
                for box in self.of(service):
                    if box.running:
                        return fail(1, b"cannot remove a running container")
                    del self.boxes[box.cid]
            return ok()
        raise AssertionError(f"unexpected extraction compose command: {rest}")


class FakeClock:
    """UTC time and monotonic time that move only when the code sleeps."""

    def __init__(self, start: datetime, events: list):
        self.start = start
        self.now = 1000.0
        self.sleeps: list[float] = []
        self.events = events

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.events.append(("sleep", seconds))
        self.now += seconds

    def utc(self) -> datetime:
        return self.start + timedelta(seconds=self.now - 1000.0)


class FakeFetch:
    def __call__(self, url: str, timeout: float) -> bytes:
        raise OSError(f"connection refused: {url}")


class Harness:
    def __init__(self, root: Root, home: Path, *, at: datetime = NIGHT, **docker):
        self.root = root
        self.docker = FakeDocker(root, **docker)
        self.clock = FakeClock(at, self.docker.events)
        self.ctx = instance.Context(
            runner=self.docker,
            clock=self.clock.utc,
            local_zone=LONDON,
            home=home,
            script_path=SCRIPT,
            sleep=self.clock.sleep,
            monotonic=self.clock.monotonic,
            fetch=FakeFetch(),
        )

    def run(self, *argv: str) -> int:
        return instance.main([*argv, "--root", str(self.root.path)], self.ctx)

    def apply(self) -> int:
        return self.run("extraction", "apply")


def stamp(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- The decision table ----------------------------------------------------------------------

MODES = ("on", "off", "auto")
WINDOWS = ("inside", "outside")
RUNNING = ("none", "current", "stale")


def expected_outcome(mode, window, yielded, configured, image_ok, running) -> tuple[str, str]:
    """What apply must do, and the paused reason (or the start reason) it must record."""
    if yielded:
        reason = "gaming"
    elif mode == "off":
        reason = "mode"
    elif mode == "auto" and window == "outside":
        reason = "window"
    elif not configured:
        reason = "unconfigured"
    else:
        if not image_ok:
            return "refuse", ""
        if running == "current":
            return "nothing", ""
        return "start", "mode" if mode == "on" else "window"
    return ("stop" if running != "none" else "nothing"), reason


TABLE = list(
    itertools.product(MODES, WINDOWS, (False, True), (True, False), (True, False), RUNNING)
)


@pytest.mark.parametrize(("mode", "window", "yielded", "configured", "image_ok", "running"), TABLE)
def test_apply_decision_table(
    tmp_path, home, capsys, mode, window, yielded, configured, image_ok, running
):
    root = make_root(tmp_path, configured=configured)
    root.set_mode(mode)
    if yielded:
        root.yield_gpu()
    h = Harness(
        root,
        home,
        at=NIGHT if window == "inside" else DAY,
        local_image=EXTRACTION_IMAGE_ID if image_ok else OTHER_ID,
    )
    if running == "current":
        h.docker.add_both()
    elif running == "stale":
        h.docker.add_both(config_hash="hash-from-an-older-daily-env")
    before = h.docker.ids()

    code = h.apply()

    outcome, reason = expected_outcome(mode, window, yielded, configured, image_ok, running)
    # Missing identity while extraction is otherwise wanted: the stop path runs and the marker
    # reads unconfigured, but the apply is refused (exit 2), naming the variables.
    if reason == "unconfigured":
        assert code == 2
        err = capsys.readouterr().err
        assert "EXTRACTION_MODEL_HASH" in err and "EXTRACTION_LLM_DEVICE" in err
        code = 0
    if outcome == "refuse":
        assert code == 2
        assert h.docker.ids() == before
        assert not h.docker.acted()
        assert root.paused_payload() is None
    elif outcome == "start":
        assert code == 0
        assert h.docker.extraction_running()
        assert root.paused_payload() is None
        assert root.window_log() == [f"{stamp(h.clock.utc())} start {reason}"]
    elif outcome == "stop":
        assert code == 0
        assert h.docker.extraction_absent()
        assert root.paused_payload()["reason"] == reason
        assert root.window_log() == [f"{stamp(h.clock.utc())} stop {reason}"]
    else:
        assert code == 0
        assert h.docker.ids() == before
        assert not h.docker.acted()
        assert root.window_log() == []
        if reason:
            assert root.paused_payload()["reason"] == reason
        else:
            assert root.paused_payload() is None


def test_apply_refuses_a_release_without_an_extraction_image_and_names_it(daily, home, capsys):
    add_release(daily, TAG, extraction_image_id=None)
    daily.set_mode("on")
    h = Harness(daily, home)
    assert h.apply() == 2
    assert h.docker.calls == []
    err = capsys.readouterr().err
    assert TAG in err and "extraction_image_id" in err


@pytest.mark.parametrize("action", ["on", "off", "auto", "apply", "status"])
def test_every_extraction_command_refuses_a_pre_v4_release(daily, home, action):
    add_release(daily, TAG, extraction_image_id=None)
    h = Harness(daily, home)
    assert h.run("extraction", action) == 2
    assert h.docker.calls == []
    assert daily.mode() is None


def test_an_absent_mode_file_reads_auto(daily, home):
    h = Harness(daily, home, at=NIGHT)
    assert h.apply() == 0
    assert h.docker.extraction_running()
    h = Harness(daily, home, at=DAY)
    h.docker.add_both()
    assert h.apply() == 0
    assert h.docker.extraction_absent()
    assert daily.paused_payload()["reason"] == "window"


def test_an_unknown_mode_is_refused(daily, home, capsys):
    daily.set_mode("sometimes")
    h = Harness(daily, home)
    assert h.apply() == 2
    assert "extraction-mode" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["<placeholder>", ""])
def test_a_placeholder_or_empty_identity_is_refused_and_pauses_unconfigured(
    tmp_path, home, value, capsys
):
    root = make_root(tmp_path, configured=False)
    root.set_env_line(f"EXTRACTION_MODEL_HASH={value}")
    root.set_env_line(f"EXTRACTION_LLM_DEVICE={DEVICE_VALUE}")
    root.set_mode("on")
    h = Harness(root, home)
    h.docker.add_both()
    assert h.apply() == 2
    assert h.docker.extraction_absent()
    assert root.paused_payload()["reason"] == "unconfigured"
    captured = capsys.readouterr()
    assert "EXTRACTION_MODEL_HASH" in captured.err
    # Only the missing name: the configured one, and every value, stay out of the message.
    assert "EXTRACTION_LLM_DEVICE" not in captured.err
    assert DEVICE_VALUE not in captured.out + captured.err


def test_up_reports_a_refused_unconfigured_apply_as_a_problem_and_keeps_chat_up(
    tmp_path, home, capsys
):
    root = make_root(tmp_path, configured=False)
    root.set_mode("on")
    h = Harness(root, home)
    h.docker.canned.update(core_canned(root))
    assert h.run("up") == 1
    assert [*root.base(), "up", "-d", "mist-neo4j", "mist-llm", "mist-backend"] in h.docker.calls
    assert root.paused_payload()["reason"] == "unconfigured"
    err = capsys.readouterr().err
    assert f"release {TAG} is up, but" in err and "EXTRACTION_MODEL_HASH" in err


def test_apply_refuses_a_missing_local_image_while_extraction_is_wanted(daily, home, capsys):
    daily.set_mode("on")
    h = Harness(daily, home, local_image=None)
    assert h.apply() == 2
    assert h.docker.extraction_absent()
    assert not h.docker.acted()
    assert daily.paused_payload() is None
    assert daily.window_log() == []
    err = capsys.readouterr().err
    assert f"mist-extraction:{TAG}" in err and "release-build" in err


# --- The window ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("moment", "inside"),
    [
        pytest.param(datetime(2026, 11, 15, 23, 58, tzinfo=UTC), False, id="23:58 GMT"),
        pytest.param(datetime(2026, 11, 15, 23, 59, tzinfo=UTC), True, id="23:59 GMT"),
        pytest.param(datetime(2026, 11, 16, 0, 0, tzinfo=UTC), True, id="midnight GMT"),
        pytest.param(datetime(2026, 11, 16, 9, 59, tzinfo=UTC), True, id="09:59 GMT"),
        pytest.param(datetime(2026, 11, 16, 10, 0, tzinfo=UTC), False, id="10:00 GMT"),
        # Summer time: the window is London's wall clock, an hour ahead of UTC.
        pytest.param(datetime(2026, 7, 1, 22, 58, tzinfo=UTC), False, id="23:58 BST"),
        pytest.param(datetime(2026, 7, 1, 22, 59, tzinfo=UTC), True, id="23:59 BST"),
        pytest.param(datetime(2026, 7, 2, 8, 59, tzinfo=UTC), True, id="09:59 BST"),
        pytest.param(datetime(2026, 7, 2, 9, 0, tzinfo=UTC), False, id="10:00 BST"),
        # The spring change, 29 March 2026 at 01:00 UTC.
        pytest.param(datetime(2026, 3, 28, 9, 30, tzinfo=UTC), True, id="09:30 GMT before spring"),
        pytest.param(datetime(2026, 3, 29, 0, 30, tzinfo=UTC), True, id="00:30 GMT spring night"),
        pytest.param(datetime(2026, 3, 29, 1, 30, tzinfo=UTC), True, id="02:30 BST spring night"),
        pytest.param(datetime(2026, 3, 29, 8, 59, tzinfo=UTC), True, id="09:59 BST spring"),
        pytest.param(datetime(2026, 3, 29, 9, 0, tzinfo=UTC), False, id="10:00 BST spring"),
        # The autumn change, 25 October 2026 at 01:00 UTC; 01:30 local happens twice.
        pytest.param(datetime(2026, 10, 24, 22, 58, tzinfo=UTC), False, id="23:58 BST autumn"),
        pytest.param(datetime(2026, 10, 24, 22, 59, tzinfo=UTC), True, id="23:59 BST autumn"),
        pytest.param(datetime(2026, 10, 25, 0, 30, tzinfo=UTC), True, id="01:30 BST autumn"),
        pytest.param(datetime(2026, 10, 25, 1, 30, tzinfo=UTC), True, id="01:30 GMT autumn"),
        pytest.param(datetime(2026, 10, 25, 9, 59, tzinfo=UTC), True, id="09:59 GMT autumn"),
        pytest.param(datetime(2026, 10, 25, 10, 0, tzinfo=UTC), False, id="10:00 GMT autumn"),
        pytest.param(datetime(2026, 10, 24, 9, 30, tzinfo=UTC), False, id="10:30 BST day before"),
    ],
)
def test_the_default_window_is_23_59_to_10_00_london_wall_time(daily, home, moment, inside):
    daily.set_mode("auto")
    h = Harness(daily, home, at=moment)
    assert h.apply() == 0
    if inside:
        assert h.docker.extraction_running()
        assert daily.paused_payload() is None
    else:
        assert h.docker.extraction_absent()
        assert daily.paused_payload()["reason"] == "window"


@pytest.mark.parametrize(
    ("moment", "until"),
    [
        # Next start 23:59 GMT the same day, plus 15 minutes.
        (datetime(2026, 11, 16, 12, 0, tzinfo=UTC), "2026-11-17T00:14:00Z"),
        # Before the spring change the start is 23:59 GMT; on the day itself it is 23:59 BST.
        (datetime(2026, 3, 28, 12, 0, tzinfo=UTC), "2026-03-29T00:14:00Z"),
        (datetime(2026, 3, 29, 11, 0, tzinfo=UTC), "2026-03-29T23:14:00Z"),
        # The autumn change: 23:59 BST on the 24th, 23:59 GMT on the 25th.
        (datetime(2026, 10, 24, 11, 0, tzinfo=UTC), "2026-10-24T23:14:00Z"),
        (datetime(2026, 10, 25, 12, 0, tzinfo=UTC), "2026-10-26T00:14:00Z"),
        # 23:58 is a minute before today's start.
        (datetime(2026, 11, 16, 23, 58, tzinfo=UTC), "2026-11-17T00:14:00Z"),
    ],
)
def test_a_window_pause_lasts_until_the_next_start_plus_15_minutes(daily, home, moment, until):
    daily.set_mode("auto")
    h = Harness(daily, home, at=moment)
    assert h.apply() == 0
    assert daily.paused_payload() == {
        "v": 1,
        "since": stamp(moment),
        "reason": "window",
        "until": until,
    }


def test_a_window_that_does_not_cross_midnight(daily, home):
    daily.set_env_line("MIST_EXTRACTION_WINDOW=13:00-15:30")
    daily.set_mode("auto")
    h = Harness(daily, home, at=datetime(2026, 11, 16, 14, 0, tzinfo=UTC))
    assert h.apply() == 0
    assert h.docker.extraction_running()
    h = Harness(daily, home, at=datetime(2026, 11, 16, 15, 30, tzinfo=UTC))
    assert h.apply() == 0
    assert h.docker.extraction_absent()
    assert daily.paused_payload()["until"] == "2026-11-17T13:15:00Z"


@pytest.mark.parametrize(
    "value", ["10:00-10:00", "25:00-10:00", "23:59", "11pm-10am", "9:00-10:00"]
)
def test_a_malformed_window_is_refused(daily, home, value, capsys):
    daily.set_env_line(f"MIST_EXTRACTION_WINDOW={value}")
    h = Harness(daily, home)
    assert h.apply() == 2
    assert h.docker.calls == []
    assert "MIST_EXTRACTION_WINDOW" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["", "   "])
def test_an_empty_window_reads_as_the_default(daily, home, value, capsys):
    daily.set_env_line(f"MIST_EXTRACTION_WINDOW={value}")
    daily.set_mode("auto")
    assert Harness(daily, home, at=NIGHT).apply() == 0
    assert Harness(daily, home, at=DAY).apply() == 0
    assert daily.paused_payload()["reason"] == "window"
    capsys.readouterr()
    assert Harness(daily, home, at=DAY).run("extraction", "status") == 0
    assert "window 23:59-10:00 outside" in capsys.readouterr().out


def test_the_local_zone_defaults_to_the_hosts():
    local = instance._local_now(instance.Context(clock=lambda: NIGHT))
    assert local.tzinfo is not None
    assert local == NIGHT


# --- Stop ------------------------------------------------------------------------------------


def test_stop_writes_the_marker_waits_two_polls_stops_service_first_saves_logs_then_removes(
    daily, home
):
    daily.set_mode("off")
    (daily.logs / "extraction-last-mist-extraction.log").write_text("old\n", encoding="utf-8")
    h = Harness(daily, home)
    h.docker.add_both()
    assert h.apply() == 0

    acting = [
        (event[1][10:] if event[0] == "run" else event)
        for event in h.docker.events
        if event[0] == "sleep" or event[1][10:11] in (["stop"], ["logs"], ["rm"])
    ]
    assert acting == [
        ("sleep", 4.0),
        ["stop", "--timeout", "10", SVC],
        ["stop", LLM],
        ["logs", "--no-color", "--tail", "200", SVC],
        ["logs", "--no-color", "--tail", "200", LLM],
        ["rm", "-f", SVC, LLM],
    ]
    first_stop = next(i for i, c in enumerate(h.docker.calls) if c[10:11] == ["stop"])
    assert h.docker.snapshots[first_stop].paused["reason"] == "mode"
    for service in (SVC, LLM):
        saved = (daily.logs / f"extraction-last-{service}.log").read_text(encoding="utf-8")
        assert saved == f"last lines of {service}\n"
    assert h.docker.extraction_absent()
    assert daily.paused_payload() == {
        "v": 1,
        "since": stamp(NIGHT),
        "reason": "mode",
        "until": None,
    }


def test_stop_waits_twice_the_poll_interval_daily_env_sets(daily, home):
    daily.set_env_line("MIST_GPU_YIELD_POLL_S=3.5")
    daily.set_mode("off")
    h = Harness(daily, home)
    h.docker.add_both()
    assert h.apply() == 0
    assert h.clock.sleeps == [7.0]


def test_stop_skips_the_wait_when_the_backend_already_reads_paused(daily, home):
    daily.set_mode("off")
    daily.paused.write_text(
        json.dumps({"v": 1, "since": "2026-11-15T10:00:00Z", "reason": "mode", "until": None}),
        encoding="utf-8",
    )
    h = Harness(daily, home)
    h.docker.add_both()
    assert h.apply() == 0
    assert h.clock.sleeps == []
    assert daily.paused_payload()["since"] == "2026-11-15T10:00:00Z"


def test_stop_removes_a_stopped_leftover_without_logging_a_transition(daily, home):
    daily.set_mode("off")
    h = Harness(daily, home)
    h.docker.add(LLM, running=False)
    assert h.apply() == 0
    assert h.docker.extraction_absent()
    assert ["stop", "--timeout", "10", SVC] not in h.docker.compose_tails()
    assert ["logs", "--no-color", "--tail", "200", SVC] not in h.docker.compose_tails()
    assert h.clock.sleeps == []
    assert daily.window_log() == []


def test_a_failed_stop_exits_1_and_keeps_the_marker(daily, home, capsys):
    daily.set_mode("off")
    h = Harness(daily, home)
    h.docker.add_both()
    h.docker.canned[(*daily.dc(), "stop", "--timeout", "10", SVC)] = fail(1, b"daemon gone")
    assert h.apply() == 1
    assert daily.paused_payload()["reason"] == "mode"
    assert ["stop", LLM] not in h.docker.compose_tails()
    assert "daemon gone" in capsys.readouterr().err


# --- Start -----------------------------------------------------------------------------------


def test_start_marks_starting_then_starts_the_llm_then_the_service_and_waits_for_both(daily, home):
    daily.set_mode("on")
    h = Harness(daily, home, polls_to_healthy=3)
    assert h.apply() == 0
    tails = h.docker.compose_tails()
    assert tails == [
        ["up", "-d", LLM],
        ["up", "-d", SVC],
        ["ps", "--all", "--format", "json", LLM, SVC],
        ["ps", "--all", "--format", "json", LLM, SVC],
        ["ps", "--all", "--format", "json", LLM, SVC],
    ]
    first_up = next(i for i, c in enumerate(h.docker.calls) if c[10:12] == ["up", "-d"])
    assert h.docker.snapshots[first_up].paused == {
        "v": 1,
        "since": stamp(NIGHT),
        "reason": "starting",
        "until": stamp(NIGHT + timedelta(seconds=600)),
    }
    # The poll that saw both healthy ran with the marker still there; it goes only after.
    assert h.docker.snapshots[-1].paused["reason"] == "starting"
    assert daily.paused_payload() is None
    assert h.docker.extraction_running()
    assert h.clock.sleeps == [5.0, 5.0]
    assert daily.window_log() == [f"{stamp(NIGHT)} start mode"]


def test_a_second_apply_with_nothing_changed_recreates_nothing(daily, home):
    daily.set_mode("on")
    h = Harness(daily, home)
    assert h.apply() == 0
    ids = h.docker.ids()
    h.docker.calls.clear()
    assert h.apply() == 0
    assert h.docker.ids() == ids
    assert not h.docker.acted()
    assert [tail for tail in h.docker.compose_tails() if tail[:1] == ["config"]] == [
        ["config", "--hash", LLM],
        ["config", "--hash", SVC],
    ]
    assert len(daily.window_log()) == 1


def test_the_config_hash_is_read_with_the_extraction_profile(daily, home):
    daily.set_mode("on")
    h = Harness(daily, home)
    h.docker.add_both()
    assert h.apply() == 0
    hash_calls = [call for call in h.docker.calls if "--hash" in call]
    assert hash_calls == [
        [*daily.dc(), "config", "--hash", LLM],
        [*daily.dc(), "config", "--hash", SVC],
    ]


def test_running_and_healthy_removes_a_leftover_marker(daily, home):
    daily.set_mode("on")
    daily.paused.write_text(
        json.dumps({"v": 1, "since": stamp(NIGHT), "reason": "starting", "until": None}),
        encoding="utf-8",
    )
    h = Harness(daily, home)
    h.docker.add_both()
    assert h.apply() == 0
    assert daily.paused_payload() is None
    assert not h.docker.acted()


@pytest.mark.parametrize(
    "stale",
    [
        pytest.param({"service": SVC, "image": OLD_EXTRACTION_IMAGE_ID}, id="other image"),
        pytest.param({"service": LLM, "config_hash": "hash-ncmoe-21"}, id="other config hash"),
        pytest.param({"service": SVC, "config_file": "/old/release/compose.yml"}, id="other file"),
    ],
)
def test_a_container_that_differs_is_recreated_with_the_marker_written_first(daily, home, stale):
    daily.set_mode("on")
    h = Harness(daily, home)
    service = stale.pop("service")
    other = LLM if service == SVC else SVC
    h.docker.add(other)
    h.docker.add(service, **stale)
    old_ids = h.docker.ids()
    assert h.apply() == 0
    tails = h.docker.compose_tails()
    rm_index = tails.index(["rm", "-f", SVC, LLM])
    assert rm_index < tails.index(["up", "-d", LLM])
    first_stop = next(i for i, c in enumerate(h.docker.calls) if c[10:11] == ["stop"])
    assert h.docker.snapshots[first_stop].paused["reason"] == "starting"
    assert h.clock.sleeps[0] == 4.0
    assert h.docker.extraction_running()
    assert not (h.docker.ids() & old_ids)
    assert daily.paused_payload() is None


def test_a_stopped_current_pair_is_started_without_recreating(daily, home):
    daily.set_mode("on")
    h = Harness(daily, home)
    h.docker.add(LLM, running=False)
    h.docker.add(SVC, running=False)
    ids = h.docker.ids()
    assert h.apply() == 0
    assert h.docker.ids() == ids
    assert ["rm", "-f", SVC, LLM] not in h.docker.compose_tails()
    assert h.docker.extraction_running()


def test_a_start_that_never_turns_healthy_removes_both_and_the_marker(daily, home, capsys):
    daily.set_mode("on")
    h = Harness(daily, home, never_healthy=True)
    assert h.apply() == 1
    assert h.docker.extraction_absent()
    assert daily.paused_payload() is None
    assert 600 <= sum(h.clock.sleeps) < 610
    tails = h.docker.compose_tails()
    assert tails[-1] == ["rm", "-f", SVC, LLM]
    assert ["stop", "--timeout", "10", SVC] in tails
    for service in (SVC, LLM):
        assert (daily.logs / f"extraction-last-{service}.log").exists()
    assert daily.window_log()[-1].endswith(" stop failed")
    assert "healthy" in capsys.readouterr().err


def test_a_failed_up_removes_both_and_the_marker(daily, home):
    daily.set_mode("on")
    h = Harness(daily, home, fail_up={SVC})
    assert h.apply() == 1
    assert h.docker.extraction_absent()
    assert daily.paused_payload() is None
    tails = h.docker.compose_tails()
    assert tails.index(["up", "-d", LLM]) < tails.index(["rm", "-f", SVC, LLM])


@pytest.mark.parametrize("trigger", ["gpu-yielded", "mode off", "window end"])
def test_a_change_during_a_start_ends_with_both_stopped_and_removed(daily, home, trigger):
    daily.set_mode("on")
    at = NIGHT
    if trigger == "window end":
        daily.set_mode("auto")
        at = datetime(2026, 11, 16, 9, 59, 50, tzinfo=UTC)

    def on_call(argv):
        if argv[10:] == ["up", "-d", LLM] and trigger == "gpu-yielded":
            daily.yield_gpu()
        if argv[10:] == ["up", "-d", SVC] and trigger == "mode off":
            daily.set_mode("off")

    h = Harness(daily, home, at=at, polls_to_healthy=5, on_call=on_call)
    assert h.apply() == 0
    assert h.docker.extraction_absent()
    expected = {"gpu-yielded": "gaming", "mode off": "mode", "window end": "window"}[trigger]
    assert daily.paused_payload()["reason"] == expected
    assert daily.window_log()[-1].endswith(f" stop {expected}")
    if trigger == "window end":
        assert daily.paused_payload()["until"] == "2026-11-17T00:14:00Z"
    if trigger == "gpu-yielded":
        assert ["up", "-d", SVC] not in h.docker.compose_tails()


def test_identity_removed_from_daily_env_during_a_start_stops_it(daily, home):
    daily.set_mode("on")
    text = daily.env.read_text(encoding="utf-8")

    def on_call(argv):
        if argv[10:] == ["up", "-d", SVC]:
            daily.env.write_text(text.replace(f"EXTRACTION_LLM_DEVICE={DEVICE_VALUE}\n", ""))

    h = Harness(daily, home, on_call=on_call)
    assert h.apply() == 2
    assert h.docker.extraction_absent()
    assert daily.paused_payload()["reason"] == "unconfigured"


# --- on, off, auto, status -------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["on", "off", "auto"])
def test_setting_the_mode_writes_it_then_applies(daily, home, mode):
    daily.set_mode("off")
    seen: list[str | None] = []
    h = Harness(daily, home, at=DAY, on_call=lambda argv: seen.append(daily.mode()))
    assert h.run("extraction", mode) == 0
    assert daily.mode() == mode
    assert seen and set(seen) == {mode}
    assert h.docker.extraction_running() == (mode == "on")


def test_setting_the_mode_reports_a_refused_apply_as_not_converged(daily, home, capsys):
    h = Harness(daily, home, local_image=OTHER_ID)
    assert h.run("extraction", "on") == 1
    assert daily.mode() == "on"
    assert "release-build" in capsys.readouterr().err


def test_extraction_status_prints_one_line_and_no_daily_env_identity(daily, home, capsys):
    daily.set_mode("auto")
    h = Harness(daily, home)
    h.docker.add_both()
    assert h.run("extraction", "status") == 0
    out = capsys.readouterr().out
    assert out.splitlines() == [
        "extraction: mode auto, window 23:59-10:00 inside, running yes, paused no, "
        "image match, configured yes"
    ]
    assert MODEL_HASH_VALUE not in out and DEVICE_VALUE not in out
    assert not h.docker.acted()


def test_extraction_status_while_paused_and_unconfigured(tmp_path, home, capsys):
    root = make_root(tmp_path, configured=False)
    h = Harness(root, home, at=DAY, local_image=None)
    assert h.apply() == 0
    capsys.readouterr()
    assert h.run("extraction", "status") == 0
    assert capsys.readouterr().out.splitlines() == [
        "extraction: mode auto, window 23:59-10:00 outside, running no, paused window, "
        "image missing, configured no"
    ]


# --- gaming-mode -----------------------------------------------------------------------------


def stop_llm(root: Root) -> tuple[str, ...]:
    return (*root.base(), "stop", "mist-llm")


def test_gaming_mode_on_stops_all_three_gpu_services_service_before_its_llm(daily, home):
    daily.set_mode("on")
    h = Harness(daily, home)
    h.docker.add_both()
    h.docker.canned[stop_llm(daily)] = ok()
    assert h.run("gaming-mode", "on") == 0
    stops = [call for call in h.docker.calls if "stop" in call]
    assert stops == [
        list(stop_llm(daily)),
        [*daily.dc(), "stop", "--timeout", "10", SVC],
        [*daily.dc(), "stop", LLM],
    ]
    assert h.docker.extraction_absent()
    assert daily.paused_payload()["reason"] == "gaming"
    assert (daily.ops / "gpu-yielded").exists()
    # The two polls after the gpu-yielded write cover the extraction stop too.
    assert h.clock.sleeps == [4.0]
    first_extraction_stop = h.docker.calls.index([*daily.dc(), "stop", "--timeout", "10", SVC])
    assert h.docker.snapshots[first_extraction_stop].paused["reason"] == "gaming"


def test_gaming_mode_on_leaves_a_pre_v4_release_as_before(daily, home):
    add_release(daily, TAG, extraction_image_id=None)
    h = Harness(daily, home)
    h.docker.canned[stop_llm(daily)] = ok()
    assert h.run("gaming-mode", "on") == 0
    assert h.docker.calls == [list(stop_llm(daily))]
    assert daily.paused_payload() is None


def test_gaming_mode_off_applies_after_removing_gpu_yielded(daily, home):
    daily.set_mode("on")
    daily.yield_gpu()
    h = Harness(daily, home)
    h.docker.canned[(*daily.base(), "up", "-d", "mist-llm")] = ok()
    h.docker.canned[(*daily.base(), "ps", "--format", "json", "mist-llm")] = ok(
        json.dumps([{"Service": "mist-llm", "State": "running", "Health": "healthy"}])
    )
    assert h.run("gaming-mode", "off") == 0
    llm_up = h.docker.calls.index([*daily.base(), "up", "-d", "mist-llm"])
    extraction_up = h.docker.calls.index([*daily.dc(), "up", "-d", LLM])
    assert llm_up < extraction_up
    assert not h.docker.snapshots[extraction_up].yielded
    assert h.docker.extraction_running()


def test_gaming_mode_off_reports_an_apply_that_did_not_converge(daily, home, capsys):
    daily.set_mode("on")
    daily.yield_gpu()
    h = Harness(daily, home, never_healthy=True)
    h.docker.canned[(*daily.base(), "up", "-d", "mist-llm")] = ok()
    h.docker.canned[(*daily.base(), "ps", "--format", "json", "mist-llm")] = ok(
        json.dumps([{"Service": "mist-llm", "State": "running", "Health": "healthy"}])
    )
    assert h.run("gaming-mode", "off") == 1
    assert not (daily.ops / "gpu-yielded").exists()
    assert "extraction" in capsys.readouterr().err


def test_gaming_mode_off_in_process_abort_during_the_start_stops_extraction(daily, home):
    daily.set_mode("on")
    daily.yield_gpu()
    h = Harness(daily, home, polls_to_healthy=4)
    h.docker.canned[(*daily.base(), "up", "-d", "mist-llm")] = ok()
    h.docker.canned[(*daily.base(), "ps", "--format", "json", "mist-llm")] = ok(
        json.dumps([{"Service": "mist-llm", "State": "running", "Health": "healthy"}])
    )
    polls = iter([False, False, False, True])
    outcome = instance.gaming_mode_off(
        daily.path, h.ctx, wait_minutes=5, should_abort=lambda: next(polls, True)
    )
    assert outcome == "done"
    assert h.docker.extraction_absent()
    assert daily.paused_payload()["reason"] == "gaming"


# --- up, down, switch, status, init ----------------------------------------------------------


def core_canned(root: Root, tag: str = TAG) -> dict:
    return {
        DOCKER_INFO: ok("27.0.0\n"),
        ("docker", "image", "inspect", "--format", "{{.Id}}", f"mist-backend:{tag}"): ok(
            IMAGE_ID + "\n"
        ),
        NETWORK_INSPECT: ok("[]"),
        (*root.base(tag), "up", "-d", "mist-neo4j", "mist-llm", "mist-backend"): ok(),
    }


def test_up_applies_after_starting_the_services(daily, home):
    daily.set_mode("on")
    h = Harness(daily, home)
    h.docker.canned.update(core_canned(daily))
    assert h.run("up") == 0
    services_up = h.docker.calls.index(
        [*daily.base(), "up", "-d", "mist-neo4j", "mist-llm", "mist-backend"]
    )
    extraction_up = h.docker.calls.index([*daily.dc(), "up", "-d", LLM])
    assert services_up < extraction_up
    assert h.docker.extraction_running()


def test_up_reports_a_failed_apply_as_a_problem_and_keeps_chat_up(daily, home, capsys):
    daily.set_mode("on")
    h = Harness(daily, home, fail_up={LLM})
    h.docker.canned.update(core_canned(daily))
    assert h.run("up") == 1
    assert [*daily.base(), "up", "-d", "mist-neo4j", "mist-llm", "mist-backend"] in h.docker.calls
    err = capsys.readouterr().err
    assert "release daily-2026.11.15 is up, but" in err and "extraction" in err


def test_up_skips_apply_for_a_pre_v4_release_with_a_note(daily, home, capsys):
    add_release(daily, TAG, extraction_image_id=None)
    h = Harness(daily, home)
    h.docker.canned.update(core_canned(daily))
    assert h.run("up") == 0
    assert h.docker.compose_tails() == []
    out = capsys.readouterr()
    assert f"release {TAG} records no extraction image; extraction apply skipped" in out.out
    assert out.err == ""


def project_stop(root: Root) -> tuple[str, ...]:
    return (*root.base(), "--profile", "backup", "--profile", "extraction", "stop")


def test_apply_after_down_inside_the_window_starts_nothing_and_writes_no_marker(
    daily, home, capsys
):
    daily.set_mode("auto")

    def on_call(argv):
        if tuple(argv) == project_stop(daily):
            h.docker.backend_running = False
        if argv == [*daily.base(), "up", "-d", "mist-neo4j", "mist-llm", "mist-backend"]:
            h.docker.backend_running = True

    h = Harness(daily, home, at=NIGHT, on_call=on_call)
    h.docker.canned[project_stop(daily)] = ok()
    h.docker.canned.update(core_canned(daily))
    assert h.run("down") == 0
    capsys.readouterr()

    assert h.apply() == 0
    assert not h.docker.acted()
    assert daily.paused_payload() is None
    assert daily.window_log() == []
    assert PROJECT_RUNNING_PS in h.docker.calls
    out = capsys.readouterr().out.splitlines()
    assert out == ["extraction: mist-backend of project mist-daily is not running; started nothing"]

    # The next `up`, still inside the window, brings both back.
    assert h.run("up") == 0
    assert h.docker.extraction_running()
    assert daily.paused_payload() is None


@pytest.mark.parametrize(
    ("at", "mode", "reason"), [(DAY, "auto", "window"), (NIGHT, "off", "mode")]
)
def test_apply_with_the_backend_stopped_still_takes_the_ordered_stop_path(
    daily, home, at, mode, reason
):
    daily.set_mode(mode)
    h = Harness(daily, home, at=at, backend_running=False)
    h.docker.add_both()
    assert h.apply() == 0
    assert h.docker.extraction_absent()
    assert daily.paused_payload()["reason"] == reason
    tails = [t for t in h.docker.compose_tails() if t[:1] in (["stop"], ["rm"])]
    assert tails == [["stop", "--timeout", "10", SVC], ["stop", LLM], ["rm", "-f", SVC, LLM]]
    first_stop = next(i for i, c in enumerate(h.docker.calls) if c[10:11] == ["stop"])
    assert h.docker.snapshots[first_stop].paused["reason"] == reason


@pytest.mark.parametrize("answer", [fail(1, b"daemon busy"), ok("not json at all")])
def test_apply_exits_1_and_starts_nothing_when_the_backend_state_is_unreadable(
    daily, home, answer, capsys
):
    daily.set_mode("on")
    h = Harness(daily, home)
    h.docker.canned[tuple(PROJECT_RUNNING_PS)] = answer
    assert h.apply() == 1
    assert not h.docker.acted()
    assert daily.paused_payload() is None
    err = capsys.readouterr().err
    assert "mist-daily" in err and "started nothing" in err


def test_down_stops_the_extraction_profile_too(daily, home):
    h = Harness(daily, home)
    stop = (*daily.base(), "--profile", "backup", "--profile", "extraction", "stop")
    h.docker.canned[stop] = ok()
    assert h.run("down") == 0
    assert h.docker.calls == [list(stop)]


def point_at_old_tag(root: Root) -> None:
    text = root.env.read_text(encoding="utf-8")
    root.env.write_text(text.replace(f"MIST_RELEASE_TAG={TAG}", f"MIST_RELEASE_TAG={OLD_TAG}"))


def label_ps(service: str) -> list[str]:
    return [
        "docker",
        "ps",
        "--all",
        "--quiet",
        "--no-trunc",
        "--filter",
        "label=com.docker.compose.project=mist-daily",
        "--filter",
        f"label=com.docker.compose.service={service}",
    ]


def test_switch_removes_older_release_extraction_containers_by_label_then_status_exits_0(
    daily, home, capsys
):
    add_release(daily, OLD_TAG, extraction_image_id=OLD_EXTRACTION_IMAGE_ID)
    point_at_old_tag(daily)
    daily.set_mode("off")
    h = Harness(daily, home)
    old_file = str(daily.compose_file(OLD_TAG))
    h.docker.add(LLM, running=False, config_file=old_file)
    h.docker.add(SVC, running=False, config_file=old_file, image=OLD_EXTRACTION_IMAGE_ID)
    old_ids = sorted(h.docker.ids())
    h.docker.canned.update(core_canned(daily))
    assert h.run("switch", TAG) == 0

    removals = [call for call in h.docker.calls if call[:3] == ["docker", "rm", "-f"]]
    assert sorted(cid for call in removals for cid in call[3:]) == old_ids
    first_removal = h.docker.calls.index(removals[0])
    assert label_ps(LLM) in h.docker.calls[:first_removal]
    assert label_ps(SVC) in h.docker.calls[:first_removal]
    assert f"MIST_RELEASE_TAG={OLD_TAG}" in h.docker.snapshots[first_removal].env
    assert f"MIST_RELEASE_TAG={TAG}\n" in daily.env.read_text(encoding="utf-8")
    assert h.docker.extraction_absent()

    h.docker.canned[(*daily.base(), "ps", "--format", "json")] = ok("[]")
    capsys.readouterr()
    assert h.run("status") == 0
    out = capsys.readouterr().out.splitlines()
    assert not [line for line in out if line.startswith("config-file")]
    assert (
        "extraction: mode off, window 23:59-10:00 inside, running no, paused mode, "
        "image match, configured yes"
    ) in out


def test_switch_stops_running_extraction_service_before_its_llm_saves_logs_then_removes(
    daily, home
):
    add_release(daily, OLD_TAG, extraction_image_id=OLD_EXTRACTION_IMAGE_ID)
    point_at_old_tag(daily)
    daily.set_mode("off")
    h = Harness(daily, home)
    old_file = str(daily.compose_file(OLD_TAG))
    llm = h.docker.add(LLM, config_file=old_file)
    svc = h.docker.add(SVC, config_file=old_file, image=OLD_EXTRACTION_IMAGE_ID)
    h.docker.canned.update(core_canned(daily))
    assert h.run("switch", TAG) == 0

    acting = [
        call
        for call in h.docker.calls
        if call[:2] in (["docker", "stop"], ["docker", "logs"])
        or call[:3] == ["docker", "rm", "-f"]
    ]
    assert acting == [
        ["docker", "stop", "-t", "10", svc],
        ["docker", "stop", llm],
        ["docker", "logs", "--tail", "200", svc],
        ["docker", "logs", "--tail", "200", llm],
        ["docker", "rm", "-f", llm, svc],
    ]
    first_stop = h.docker.calls.index(acting[0])
    assert f"MIST_RELEASE_TAG={OLD_TAG}" in h.docker.snapshots[first_stop].env
    assert (
        daily.logs / f"extraction-last-{SVC}.log"
    ).read_bytes() == f"out of {svc}\nerr\n".encode()
    assert (
        daily.logs / f"extraction-last-{LLM}.log"
    ).read_bytes() == f"out of {llm}\nerr\n".encode()
    assert h.docker.extraction_absent()


def test_switch_keeps_daily_env_when_an_extraction_stop_fails(daily, home, capsys):
    add_release(daily, OLD_TAG, extraction_image_id=OLD_EXTRACTION_IMAGE_ID)
    point_at_old_tag(daily)
    h = Harness(daily, home)
    old_file = str(daily.compose_file(OLD_TAG))
    llm = h.docker.add(LLM, config_file=old_file)
    svc = h.docker.add(SVC, config_file=old_file, image=OLD_EXTRACTION_IMAGE_ID)
    h.docker.canned[("docker", "stop", "-t", "10", svc)] = fail(1, b"daemon gone")
    h.docker.canned.update(core_canned(daily))
    assert h.run("switch", TAG) == 1
    assert ["docker", "stop", llm] not in h.docker.calls
    assert not [call for call in h.docker.calls if call[:3] == ["docker", "rm", "-f"]]
    assert f"MIST_RELEASE_TAG={OLD_TAG}\n" in daily.env.read_text(encoding="utf-8")
    assert "daemon gone" in capsys.readouterr().err


def test_switch_to_a_pre_v4_release_exits_0_with_the_skip_note(daily, home, capsys):
    add_release(daily, OLD_TAG, extraction_image_id=None)
    daily.set_mode("on")
    h = Harness(daily, home)
    h.docker.add_both()
    h.docker.canned.update(core_canned(daily, OLD_TAG))
    assert h.run("switch", OLD_TAG) == 0
    assert h.docker.extraction_absent()
    assert not [tail for tail in h.docker.compose_tails() if tail[:1] == ["up"]]
    out = capsys.readouterr()
    assert f"release {OLD_TAG} records no extraction image; extraction apply skipped" in out.out
    assert out.err == ""


def test_status_prints_the_extraction_line_and_never_an_identity_value(daily, home, capsys):
    daily.set_mode("on")
    h = Harness(daily, home)
    h.docker.add_both()
    h.docker.canned[(*daily.base(), "ps", "--format", "json")] = ok("[]")
    assert h.run("status") == 0
    out = capsys.readouterr().out
    assert out.splitlines() == [
        "backend: not running",
        "gaming-mode: normal",
        "extraction: mode on, window 23:59-10:00 inside, running yes, paused no, "
        "image match, configured yes",
        "health: unreachable",
        "turns-per-day: unreachable",
    ]
    assert MODEL_HASH_VALUE not in out and DEVICE_VALUE not in out


def test_status_names_a_pre_v4_release_and_exits_0(daily, home, capsys):
    add_release(daily, TAG, extraction_image_id=None)
    h = Harness(daily, home)
    h.docker.canned[(*daily.base(), "ps", "--format", "json")] = ok("[]")
    assert h.run("status") == 0
    assert f"extraction: none in release {TAG}" in capsys.readouterr().out.splitlines()


def test_init_writes_mode_off_and_a_second_init_keeps_the_mode(tmp_path, home):
    root = (tmp_path / "fresh").resolve()
    ctx = instance.Context(runner=FakeDocker(Root(root)), home=home, script_path=SCRIPT)
    assert instance.main(["init", "--root", str(root)], ctx) == 0
    assert (root / "ops" / "extraction-mode").read_text(encoding="utf-8") == "off\n"
    (root / "ops" / "extraction-mode").write_text("auto\n", encoding="utf-8")
    assert instance.main(["init", "--root", str(root)], ctx) == 0
    assert (root / "ops" / "extraction-mode").read_text(encoding="utf-8") == "auto\n"


# --- Contracts -------------------------------------------------------------------------------


def test_the_paused_marker_name_is_the_contracts():
    """T3's `backend/extraction_hold.py` pins the same name on the backend side."""
    assert instance.EXTRACTION_PAUSED_MARKER == "extraction-paused"


def test_apply_calls_no_subprocess_outside_the_runner(daily, home, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError(f"direct subprocess call: {args}")

    monkeypatch.setattr(instance.subprocess, "run", forbidden)
    monkeypatch.setattr(instance.subprocess, "Popen", forbidden)
    daily.set_mode("on")
    h = Harness(daily, home)
    assert h.apply() == 0
    daily.set_mode("off")
    assert h.apply() == 0
    assert h.run("extraction", "status") == 0


def test_every_extraction_argv_is_volume_safe(daily, home):
    daily.set_mode("on")
    h = Harness(daily, home, never_healthy=True)
    h.apply()
    for call in h.docker.calls:
        instance.volume_safety.assert_volume_safe(call)
        assert "-v" not in call and "down" not in call
