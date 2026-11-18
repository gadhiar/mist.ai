"""Host CLI for a MIST daily instance: `python <path>/instance.py <command> --root <root>`.

An instance root is a host directory outside every git work tree that holds one instance's
state, vault, logs, operations records, model weights and extracted releases. This CLI is the
only tool that creates, retires, builds releases for, switches and operates one, so it runs on
host Python with the standard library alone: after `switch` it runs from `<root>/bin/`, where
neither the repository nor its dependencies exist. It imports nothing but the standard library
and modules beside it.

Every Docker and git call goes through `Context.runner`, one injected callable, every sleep and
poll through `Context.sleep` and `Context.monotonic`, and every HTTP read through
`Context.fetch`, so the tests assert exact argument lists with no Docker, git, network or
real waiting. Before any command runs, `subprocess_runner` passes it to `volume_safety.py`,
loaded from beside this file, which refuses every form that could delete a volume.

Exit codes: 0 done; 2 refused, nothing changed, except that `extraction apply` refuses a missing
identity variable after it has stopped extraction and written the `unconfigured` marker; 1
failed part-way, with the state named, or a command the volume guard refused.
`backup-now` returns the one-shot backup's own exit code, `accept-backup-baseline` the
scheduler's, and `drill-restore` and `drill-verify` those of restore and verify.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import importlib.util
import io
import ipaddress
import json
import logging
import logging.handlers
import math
import os
import re
import shutil
import subprocess  # nosec B404 - runs fixed docker and git argument lists, never a shell
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta, tzinfo
from datetime import time as wall_time
from pathlib import Path, PurePosixPath
from typing import Literal, Protocol


def _load_volume_safety():
    """Exec the guard module beside this file and return it.

    Loaded by path because this file runs two ways: as `scripts.daily.instance` under pytest
    and as `python <path>/instance.py`, `<root>/bin/` included. Neither puts `scripts/daily`
    on `sys.path`, so a bare import fails under pytest and a dotted one fails from `bin/`. A
    missing guard is an import error naming `volume_safety.py`: there is no fallback, so no
    command can run without it.
    """
    path = Path(__file__).with_name("volume_safety.py")
    spec = importlib.util.spec_from_file_location("volume_safety", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


volume_safety = _load_volume_safety()

# Both names are spelled here rather than imported, because this file runs where `backend` is
# not importable. They equal `LIVE_INSTANCE_MARKER` and `LIVE_INSTANCES_FILE_ENV` in
# `backend/knowledge/eval_isolation.py`, whose guard reads what `init` writes; a unit test pins
# the equality.
LIVE_INSTANCE_MARKER = ".mist-live-instance"
LIVE_INSTANCES_FILE_ENV = "MIST_LIVE_INSTANCES_FILE"
MARKER_VERSION = 1

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_REFUSED = 2

INSTANCE_NAMES = ("daily", "rehearsal")
DEFAULT_INSTANCE = "daily"
DAILY_INSTANCE = "daily"
INSTANCE_DIRS = (
    "state",
    "mist-memory",
    "logs",
    "ops",
    "models",
    "release",
    "retired",
    "bin",
    "secrets",
)
# A container that mounts only `state/` or `mist-memory/` sees neither the root nor the host
# registry, so each carries its own copy of the marker.
MARKED_SUBDIRS = ("state", "mist-memory")

ENV_FILE = "daily.env"
ENV_EXAMPLE = PurePosixPath("deploy/daily/daily.env.example")

RELEASE_TAG_PATTERN = re.compile(r"daily-\d{4}\.\d{2}\.\d{2}(\.\d+)?", re.ASCII)
REHEARSAL_TAG_PREFIX = "rehearsal-"
MAIN_REF = "origin/main"
IMAGE_REPOSITORY = "mist-backend"
DOCKERFILE = "docker/backend/Dockerfile"
COMPOSE_FILE = "deploy/daily/compose.yml"
EXTRACTION_IMAGE_REPOSITORY = "mist-extraction"
EXTRACTION_DOCKERFILE = "docker/extraction/Dockerfile"
RELEASE_EXTRACT_PREFIXES = ("deploy/", "scripts/daily/")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}", re.ASCII)
REHEARSAL_TAG_PATTERN = re.compile(r"rehearsal-[0-9a-f]{7}", re.ASCII)
STDERR_TAIL_LINES = 20

# The canonical host ports keep the daily instance inside the live-endpoint denylists of
# `backend/knowledge/eval_isolation.py`; the allowlisted ones are the eval, dev and staging
# endpoints (`eval_isolation.py:65,69,74,79`), which no instance may answer on.
CANONICAL_PORTS = {
    "MIST_DAILY_WS_PORT": "8001",
    "MIST_DAILY_BOLT_PORT": "7687",
    "MIST_DAILY_NEO4J_HTTP_PORT": "7474",
}
ALLOWLISTED_PORTS = frozenset({7688, 7689, 7690, 8002})

SERVICES = ("mist-neo4j", "mist-llm", "mist-backend")
LLM_SERVICE = "mist-llm"
BACKEND_SERVICE = "mist-backend"
BACKUP_SERVICE = "mist-backup"
BACKUP_PROFILE = "backup"
DEFAULT_SHARE_NETWORK = "mist-llm-share"
CONFIG_FILES_LABEL = "com.docker.compose.project.config_files"
PROJECT_LABEL = "com.docker.compose.project"
SERVICE_LABEL = "com.docker.compose.service"
CONFIG_HASH_LABEL = "com.docker.compose.config-hash"
# The night extraction services (plan v4). They sit under their own profile, so an `up` that
# names the three chat services never starts them; only `extraction apply` does.
EXTRACTION_PROFILE = "extraction"
EXTRACTION_SERVICE = "mist-extraction"
EXTRACTION_LLM_SERVICE = "mist-extraction-llm"
# Start order: the LLM first, since the service waits on its health.
EXTRACTION_SERVICES = (EXTRACTION_LLM_SERVICE, EXTRACTION_SERVICE)
NEO4J_SERVICE = "mist-neo4j"
# Where both compose files mount the graph volume inside the Neo4j container.
NEO4J_DATA_MOUNT = "/data"
REVISION_LABEL = "org.opencontainers.image.revision"
BACKUP_ONCE = ("python", "-m", "scripts.backup.scheduler", "--once")

ACCEPT_SUSPECT = ("python", "-m", "scripts.backup.scheduler", "--accept-suspect")

# Room devices (item 11): the opt-in overlay, used only while `daily.env` sets the bind. The
# forwarder publishes the one LAN port; nothing else of the instance leaves loopback.
DEVICES_COMPOSE_FILE = "deploy/daily/compose.devices.yml"
DEVICES_PROFILE = "devices"
DEVICE_FORWARD_SERVICE = "mist-device-forward"
DEVICE_BIND_KEY = "MIST_DAILY_DEVICE_BIND"
DEVICE_PORT_KEY = "MIST_DAILY_DEVICE_PORT"
DEVICE_DIR_KEY = "MIST_DEVICE_DIR"
# A LAN address can arrive after sign-in (DHCP, Wi-Fi), so `up` keeps retrying the forwarder on
# this period until `--wait-minutes` runs out.
DEVICE_FORWARD_RETRY_S = 30.0
# RFC 1918 only: the forwarder must face the home LAN, never every interface or the internet.
PRIVATE_IPV4_NETWORKS = tuple(
    ipaddress.IPv4Network(network) for network in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
)
CONTAINER_STATE_FORMAT = "{{.State.Status}}"

# The restore drill (item 8). The graph URI is the drill compose file's `NEO4J_URI`, which is in
# the dev allowlist of `backend/knowledge/eval_isolation.py`; a unit test pins the equality. The
# marker name is `RESTORE_MARKER_FILENAME` in `scripts/backup/target.py`, spelled here because
# this file runs where `scripts.backup` is not importable; a unit test pins that too.
DRILL_COMPOSE_FILE = "deploy/daily/restore-drill.yml"
DRILL_GRAPH_URI = "bolt://mist-neo4j-dev:7687"
RESTORE_MARKER_FILENAME = "MIST_RESTORE_TARGET"
DRILL_RECORD = "drill.json"
DRILL_ROLE_LABEL = "org.mist.role"
DRILL_ROLE = "drill"
DRILL_INSTANCE_LABEL = "org.mist.instance"
DRILL_RESTORE_SERVICE = "mist-restore"
DRILL_NEO4J_SERVICE = "mist-neo4j-dev"
DRILL_RUN_PROFILE = "run"
DRILL_STANDBY = "standby"
DRILL_PRE_RESTORE = "pre-restore"
STAMP_FORMAT = "%Y%m%dT%H%M%SZ"
# An artifact or backup label joined into a container path, and a volume name: a plain name.
PLAIN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*", re.ASCII)
VOLUME_LINE = re.compile(rb"^MIST_DAILY_NEO4J_VOLUME=[^\r\n]*", re.M)

GPU_YIELDED_MARKER = "gpu-yielded"
BACKUPS_ARMED_MARKER = "backups-armed"
GAMING_BY = ("manual", "watcher")
DEFAULT_GPU_YIELD_POLL_S = 2.0
DEFAULT_UP_WAIT_MINUTES = 10
DEFAULT_OFF_WAIT_MINUTES = 5
DOCKER_POLL_S = 5.0
LLM_POLL_S = 5.0
HTTP_TIMEOUT_S = 5.0
USAGE_DAYS = 7

# The gaming watcher (item 17, Watcher).
GAMEWATCH_FILE = "gamewatch.py"
DEFAULT_GAME_POLL_S = 10.0
DEFAULT_GAME_RESUME_DELAY_S = 120.0
GAMING_WATCH_LOG = "gaming-watch.log"
GAMING_WATCH_LOG_BYTES = 1024 * 1024
GAMING_WATCH_LOG_BACKUPS = 3

# Night extraction (plan v4, Interfaces and contracts). The paused marker's name is the
# backend's `EXTRACTION_PAUSED_MARKER` in `backend/extraction_hold.py`, spelled here because
# this file runs where `backend` is not importable; a test on each side pins it.
EXTRACTION_PAUSED_MARKER = "extraction-paused"
PAUSED_REASONS = ("window", "mode", "gaming", "starting", "unconfigured")
EXTRACTION_MODE_FILE = "extraction-mode"
EXTRACTION_MODES = ("on", "off", "auto")
DEFAULT_EXTRACTION_MODE = "auto"
INIT_EXTRACTION_MODE = "off"
EXTRACTION_WINDOW_VARIABLE = "MIST_EXTRACTION_WINDOW"
DEFAULT_EXTRACTION_WINDOW = "23:59-10:00"
EXTRACTION_WINDOW_PATTERN = re.compile(
    r"([01][0-9]|2[0-3]):([0-5][0-9])-([01][0-9]|2[0-3]):([0-5][0-9])", re.ASCII
)
# A window pause stays valid this long past the next start, so a task tick that runs late does
# not leave the backend reading a stopped service as down.
EXTRACTION_WINDOW_GRACE = timedelta(minutes=15)
# The only two names the extraction services read from `daily.env`; neither is required by
# compose, so a missing one never stops chat. Apply pauses extraction as unconfigured instead.
EXTRACTION_IDENTITY_VARIABLES = ("EXTRACTION_MODEL_HASH", "EXTRACTION_LLM_DEVICE")
# Inside the night task's 15-minute execution limit, with room for the two-poll waits.
EXTRACTION_START_DEADLINE_S = 600
EXTRACTION_POLL_S = 5.0
EXTRACTION_STOP_TIMEOUT_S = 10
EXTRACTION_LOG_LINES = 200
EXTRACTION_WINDOW_LOG = "extraction-window.log"
EXTRACTION_INSPECT_FORMAT = (
    "{{.Name}} {{.Image}} {{.State.Status}} "
    "{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}} {{json .Config.Labels}}"
)

# Windows' `CREATE_NO_WINDOW` process creation flag. `subprocess` defines it only on Windows,
# so the value is spelled here for the code path that reads it, and the tests that patch
# `os.name`.
CREATE_NO_WINDOW = 0x08000000


@dataclass(frozen=True, slots=True)
class CommandResult:
    """What one Docker or git command returned."""

    returncode: int
    stdout: bytes = b""
    stderr: bytes = b""


class Runner(Protocol):
    """Runs one command, with optional bytes on its standard input.

    `env`, when given, is the command's whole environment; only the restore drill passes one,
    to hand compose two variables that must never be written to `daily.env` (item 8).
    """

    def __call__(
        self,
        argv: Sequence[str],
        *,
        stdin: bytes | None = None,
        env: Mapping[str, str] | None = None,
    ) -> CommandResult:
        """Run `argv` and return its exit code and captured output."""
        ...


def subprocess_runner(
    argv: Sequence[str],
    *,
    stdin: bytes | None = None,
    env: Mapping[str, str] | None = None,
) -> CommandResult:
    """Run `argv` without a shell, capturing its output.

    A missing program reads as exit 127, the shell's convention, so a host without Docker on
    its PATH gets the same refusal or failure message as one whose daemon is down. On Windows
    every command starts with `CREATE_NO_WINDOW`, so `gaming-watch` under `pythonw` never
    flashes a console window for each Docker call.

    Raises:
        volume_safety.DestructiveCommandError: Before anything runs, for an argv that could
            delete a volume. Every git and Docker command passes here.
    """
    command = list(argv)
    volume_safety.assert_volume_safe(command)
    environment = None if env is None else dict(env)
    flags = CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        if stdin is None:
            completed = subprocess.run(  # nosec B603 - fixed argument list, no shell
                command,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                check=False,
                env=environment,
                creationflags=flags,
            )
        else:
            completed = subprocess.run(  # nosec B603 - fixed argument list, no shell
                command,
                input=stdin,
                capture_output=True,
                check=False,
                env=environment,
                creationflags=flags,
            )
    except FileNotFoundError as exc:
        return CommandResult(127, b"", f"{command[0]}: {exc}".encode())
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _default_script_path() -> Path:
    return Path(__file__).resolve()


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    """Turns any redirect into an `HTTPError`, so a loopback read never goes elsewhere."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        raise urllib.error.HTTPError(
            req.full_url, code, f"redirect to {newurl} refused", headers, fp
        )


def loopback_opener() -> urllib.request.OpenerDirector:
    """An opener with no proxy and no redirects, for the instance's own loopback endpoints.

    The default opener reads `HTTP_PROXY` from the environment, which does not exempt
    127.0.0.1 unless `NO_PROXY` says so, and follows redirects.
    """
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), _RefuseRedirects())


def urllib_fetch(url: str, timeout: float) -> bytes:
    """GET `url` directly, never through a proxy or a redirect, and return the body.

    Raises:
        OSError: When the connection fails, the reply is an HTTP error, or it redirects
            (`HTTPError` and `URLError` both subclass `OSError`).
    """
    with loopback_opener().open(url, timeout=timeout) as reply:  # nosec B310 - loopback http
        return reply.read()


@dataclass(frozen=True, slots=True)
class Context:
    """Everything a command reads from outside its arguments, injectable for tests.

    `script_path` is where this file runs from: `init` finds the env example relative to it,
    and `release-build` finds its default repository from it. `sleep` and `monotonic` drive
    every wait and poll, and `fetch` every HTTP read, so a test never waits or connects.
    `processes` lists the running processes for `gaming-watch` as `gamewatch.ProcessRecord`s;
    None means `gamewatch.list_processes`, which exists only once that command has loaded the
    file, so the annotation cannot name the record type. `local_zone` is the zone the extraction
    window is read in; None means the host's own, daylight saving included, and a test passes a
    `zoneinfo.ZoneInfo` to drive midnight and a clock change from `clock` alone.
    """

    runner: Runner = subprocess_runner
    clock: Callable[[], datetime] = _utc_now
    local_zone: tzinfo | None = None
    environ: Mapping[str, str] = field(default_factory=lambda: os.environ)
    home: Path = field(default_factory=Path.home)
    script_path: Path = field(default_factory=_default_script_path)
    sleep: Callable[[float], None] = time.sleep
    monotonic: Callable[[], float] = time.monotonic
    fetch: Callable[[str, float], bytes] = urllib_fetch
    processes: Callable[[], list] | None = None


class InstanceError(Exception):
    """A command stopped; `exit_code` says whether anything changed."""

    exit_code = EXIT_FAILED
    label = "failed"


class RefusedError(InstanceError):
    """A check failed before anything was written."""

    exit_code = EXIT_REFUSED
    label = "refused, nothing changed"


class FailedError(InstanceError):
    """A step failed after earlier steps took effect; the message names the state left."""

    exit_code = EXIT_FAILED
    label = "failed part-way"


# --- Paths, markers and the registry ---------------------------------------------------------


def _absolute_path(raw: str, home: Path) -> Path:
    """Resolve a path argument to the host-native absolute form the registry records.

    `~` expands against `Context.home`, not the process's home, so tests never touch the real
    one.
    """
    if raw == "~" or raw.startswith(("~/", "~\\")):
        path = Path(home, raw[2:])
    else:
        path = Path(raw).expanduser()
    return path.resolve()


def _format_utc(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_atomic(path: Path, data: bytes) -> None:
    """Write through a temporary file in the same directory, so readers never see half a file."""
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)
        raise


def _read_marker(directory: Path) -> dict | None:
    """The marker in `directory`, or None when there is none.

    Raises:
        RefusedError: When a marker exists but is not a version-1 marker naming an instance.
    """
    path = directory / LIVE_INSTANCE_MARKER
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as exc:
        raise RefusedError(f"cannot read {path}: {exc}") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RefusedError(f"{path} is not valid JSON: {exc}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("v") != MARKER_VERSION
        or not isinstance(payload.get("instance"), str)
    ):
        raise RefusedError(f"{path} is not a version-{MARKER_VERSION} instance marker")
    return payload


def _instance_name(root: Path) -> str:
    marker = _read_marker(root)
    if marker is None:
        raise RefusedError(f"{root} holds no {LIVE_INSTANCE_MARKER}; it is not an instance root")
    return marker["instance"]


def _registry_path(ctx: Context) -> Path:
    # An empty value counts as unset, as in the guard's reader, so it never means the current
    # directory.
    configured = ctx.environ.get(LIVE_INSTANCES_FILE_ENV, "").strip()
    if configured:
        return Path(configured)
    return ctx.home / ".mist" / "live-instances"


def _read_registry(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except (OSError, UnicodeDecodeError) as exc:
        raise RefusedError(f"cannot read the live-instance registry {path}: {exc}") from exc
    return text.splitlines()


def _names_root(line: str, root: Path) -> bool:
    """Whether a registry line names `root`, compared in resolved form.

    A relative line is never this CLI's (it writes absolute paths only), so it never matches
    and `retire` leaves it for a person to repair.
    """
    entry = line.strip()
    if not entry or not Path(entry).is_absolute():
        return False
    return os.path.normcase(str(Path(entry).resolve())) == os.path.normcase(str(root))


def _write_registry(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(f"{line}\n" for line in lines)
    _write_atomic(path, text.encode("utf-8"))


def _git_work_tree_holding(path: Path) -> Path | None:
    """The nearest directory at or above `path` holding `.git` (a directory or a worktree file)."""
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


# --- init ------------------------------------------------------------------------------------


def _check_init_root(root: Path, ctx: Context) -> None:
    if root.parent == root:
        raise RefusedError(f"{root} is a drive root; choose a directory for the instance")
    if root == ctx.home.resolve():
        raise RefusedError(f"{root} is the home directory; choose a directory for the instance")
    work_tree = _git_work_tree_holding(root)
    if work_tree is not None:
        raise RefusedError(
            f"{root} is inside the git work tree {work_tree}; an instance root must be outside "
            "every work tree, so no git or repository tool can touch it"
        )
    if root.exists() and not root.is_dir():
        raise RefusedError(f"{root} exists and is not a directory")


def _check_existing_markers(root: Path, instance: str) -> dict | None:
    """The root's marker when it already names `instance`; refuses any other instance's marker."""
    root_marker = _read_marker(root)
    for directory in (root, *(root / name for name in MARKED_SUBDIRS)):
        marker = root_marker if directory == root else _read_marker(directory)
        if marker is not None and marker["instance"] != instance:
            raise RefusedError(
                f"{directory / LIVE_INSTANCE_MARKER} names instance {marker['instance']!r}, "
                f"not {instance!r}"
            )
    return root_marker


def _check_seed(seed: Path, memory: Path) -> None:
    for name in ("vault", "seed"):
        if not (seed / name).is_dir():
            raise RefusedError(f"--seed {seed} has no {name}/ directory")
    clashes = sorted(
        entry.name
        for entry in (seed / "vault").iterdir()
        if entry.name in ("seed", LIVE_INSTANCE_MARKER)
    )
    if clashes:
        raise RefusedError(
            f"--seed {seed}: vault/ holds {', '.join(clashes)}, which init writes itself"
        )
    if memory.is_dir():
        others = sorted(e.name for e in memory.iterdir() if e.name != LIVE_INSTANCE_MARKER)
        if others:
            raise RefusedError(
                f"{memory} already holds {', '.join(others[:5])}; --seed fills an empty "
                "mist-memory/ only"
            )


def _render_env(example_text: str, values: Mapping[str, str], example: Path) -> str:
    rendered = example_text
    for key, value in values.items():
        pattern = re.compile(rf"^{re.escape(key)}=.*$", re.M)
        if pattern.search(rendered) is None:
            raise RefusedError(f"{example} has no {key}= line to fill")
        rendered = pattern.sub(lambda _match, k=key, v=value: f"{k}={v}", rendered, count=1)
    return rendered


def cmd_init(args: argparse.Namespace, ctx: Context) -> None:
    """Create an instance root, mark it in three places and register it.

    Every check runs before the first write, so a refusal leaves the disk as it was. A second
    `init` on the same root fills in whatever is missing and rewrites nothing.
    """
    root = _absolute_path(args.root, ctx.home)
    instance = args.instance
    _check_init_root(root, ctx)
    existing_marker = _check_existing_markers(root, instance)

    memory = root / "mist-memory"
    seed = _absolute_path(args.seed, ctx.home) if args.seed else None
    if seed is not None:
        _check_seed(seed, memory)

    env_path = root / ENV_FILE
    env_text: str | None = None
    if not env_path.exists():
        example = ctx.script_path.parent.parent.parent / ENV_EXAMPLE
        try:
            example_text = example.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise RefusedError(f"cannot read the env example {example}: {exc}") from exc
        env_text = _render_env(
            example_text,
            {
                "MIST_DAILY_ROOT": root.as_posix(),
                "MIST_INSTANCE_NAME": instance,
                "MODELS_DIR": (root / "models").as_posix(),
            },
            example,
        )

    registry = _registry_path(ctx)
    registry_lines = _read_registry(registry)
    registered = any(_names_root(line, root) for line in registry_lines)

    marker = existing_marker or {
        "v": MARKER_VERSION,
        "instance": instance,
        "created_at": _format_utc(ctx.clock()),
    }
    marker_bytes = (json.dumps(marker) + "\n").encode("utf-8")

    done: list[str] = []
    try:
        for name in INSTANCE_DIRS:
            (root / name).mkdir(parents=True, exist_ok=True)
        done.append("created the directories")
        for directory in (root, *(root / name for name in MARKED_SUBDIRS)):
            if not (directory / LIVE_INSTANCE_MARKER).exists():
                _write_atomic(directory / LIVE_INSTANCE_MARKER, marker_bytes)
        done.append("wrote the markers")
        # A new instance starts with extraction off until its epoch cutover; a second init
        # keeps whatever mode the operator set since.
        mode_path = root / "ops" / EXTRACTION_MODE_FILE
        if not mode_path.exists():
            _write_atomic(mode_path, f"{INIT_EXTRACTION_MODE}\n".encode())
            done.append(f"wrote {mode_path}")
        if seed is not None:
            for entry in sorted((seed / "vault").iterdir()):
                if entry.is_dir():
                    shutil.copytree(entry, memory / entry.name)
                else:
                    shutil.copy2(entry, memory / entry.name)
            shutil.copytree(seed / "seed", memory / "seed")
            done.append(f"copied the seed into {memory}")
        if env_text is not None:
            _write_atomic(env_path, env_text.encode("utf-8"))
            done.append(f"wrote {env_path}")
        if not registered:
            _write_registry(registry, [*registry_lines, str(root)])
    except OSError as exc:
        state = "; ".join(done) if done else "nothing written"
        raise FailedError(f"init of {root} stopped at {exc}; so far: {state}") from exc

    print(f"initialized {root} as instance {instance!r}")
    if env_text is not None:
        print(f"wrote {env_path} from the example; fill in its remaining placeholders")
    print(f"registry {registry} {'already listed' if registered else 'now lists'} {root}")


# --- retire ----------------------------------------------------------------------------------


def cmd_retire(args: argparse.Namespace, ctx: Context) -> None:
    """Unmark and unregister a stopped instance, deleting no other file.

    The instance's data stays where it is; only the guard's protection is lifted, which is why
    a listed container refuses: a running instance must stay protected.
    """
    root = _absolute_path(args.root, ctx.home)
    instance = _instance_name(root)
    registry = _registry_path(ctx)
    registry_lines = _read_registry(registry)

    project = f"mist-{instance}"
    result = ctx.runner(["docker", "compose", "-p", project, "ps", "-q"])
    if result.returncode != 0:
        raise RefusedError(
            f"cannot list project {project}'s containers, so cannot tell whether it runs: "
            f"{_stderr_tail(result)}"
        )
    running = result.stdout.decode("utf-8", "replace").split()
    if running:
        raise RefusedError(
            f"project {project} has {len(running)} running container(s) ({', '.join(running)}); "
            "stop the instance first"
        )

    removed: list[str] = []
    try:
        for directory in (root, *(root / name for name in MARKED_SUBDIRS)):
            path = directory / LIVE_INSTANCE_MARKER
            if path.exists():
                path.unlink()
                removed.append(str(path))
        kept = [line for line in registry_lines if not _names_root(line, root)]
        if len(kept) != len(registry_lines):
            _write_registry(registry, kept)
    except OSError as exc:
        state = f"removed {', '.join(removed)}" if removed else "no marker removed"
        raise FailedError(
            f"retire of {root} stopped at {exc}; {state}; registry {registry} unchanged"
        ) from exc

    print(f"retired {root}: removed {len(removed)} marker(s) and its line in {registry}")


# --- release-build ---------------------------------------------------------------------------


def _stderr_tail(result: CommandResult) -> str:
    lines = result.stderr.decode("utf-8", "replace").strip().splitlines()
    tail = "\n".join(lines[-STDERR_TAIL_LINES:])
    return f"exit {result.returncode}" + (f": {tail}" if tail else "")


def _default_repo(ctx: Context) -> Path:
    """The git work tree holding this script; there is none when it runs from `<root>/bin/`."""
    script_dir = ctx.script_path.parent
    if script_dir.name == "bin" and (script_dir.parent / LIVE_INSTANCE_MARKER).exists():
        raise RefusedError(f"{ctx.script_path} runs from an instance's bin/; pass --repo")
    work_tree = _git_work_tree_holding(script_dir)
    if work_tree is None:
        raise RefusedError(f"{ctx.script_path} is not inside a git work tree; pass --repo")
    return work_tree.resolve()


def _rev_parse_commit(ctx: Context, repo: Path, ref: str) -> str | None:
    result = ctx.runner(["git", "-C", str(repo), "rev-parse", "--verify", f"{ref}^{{commit}}"])
    commit = result.stdout.decode("ascii", "replace").strip()
    if result.returncode != 0 or COMMIT_PATTERN.fullmatch(commit) is None:
        return None
    return commit


def _resolve_release(ctx: Context, repo: Path, tag: str) -> str:
    """Check that `tag` is an annotated release tag on main and return its commit."""
    if RELEASE_TAG_PATTERN.fullmatch(tag) is None:
        raise RefusedError(f"{tag!r} does not match {RELEASE_TAG_PATTERN.pattern}")
    ref = f"refs/tags/{tag}"
    kind = ctx.runner(["git", "-C", str(repo), "cat-file", "-t", ref])
    if kind.returncode != 0:
        raise RefusedError(f"tag {tag} not found in {repo}: {_stderr_tail(kind)}")
    if kind.stdout.decode("ascii", "replace").strip() != "tag":
        raise RefusedError(f"tag {tag} is not annotated; a release tag is made with `git tag -a`")
    commit = _rev_parse_commit(ctx, repo, ref)
    if commit is None:
        raise RefusedError(f"tag {tag} does not resolve to a commit in {repo}")
    ancestry = ctx.runner(["git", "-C", str(repo), "merge-base", "--is-ancestor", commit, MAIN_REF])
    if ancestry.returncode == 1:
        raise RefusedError(f"tag {tag} ({commit}) is not an ancestor of {MAIN_REF}")
    if ancestry.returncode != 0:
        raise RefusedError(
            f"cannot check that tag {tag} is on {MAIN_REF}: {_stderr_tail(ancestry)}"
        )
    return commit


def _release_members(archive: bytes) -> tuple[set[str], list[tuple[str, bytes | None]]]:
    """Every path in the archive, and the entries under `RELEASE_EXTRACT_PREFIXES` to extract.

    Entries are read into memory and written by hand, because the container's Python predates
    `tarfile`'s extraction filters. Only plain files and directories are accepted, under names
    that stay inside `release/<tag>/` on Windows as well as POSIX: `--rehearsal` builds any
    commit, and on the Windows host a backslash or a drive colon in a name would join outside it.
    An absolute name never starts with an extracted prefix, so it is skipped, not written.
    """
    names: set[str] = set()
    selected: list[tuple[str, bytes | None]] = []
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
            for member in tar.getmembers():
                name = member.name.removeprefix("./").rstrip("/")
                names.add(name)
                if not f"{name}/".startswith(RELEASE_EXTRACT_PREFIXES):
                    continue
                if "\\" in name or ":" in name or ".." in name.split("/"):
                    raise RefusedError(f"the archive holds an unsafe path {member.name!r}")
                if member.isdir():
                    selected.append((name, None))
                elif member.isfile():
                    handle = tar.extractfile(member)
                    selected.append((name, handle.read() if handle is not None else b""))
                else:
                    raise RefusedError(f"the archive holds {member.name!r}, which is not a file")
    except tarfile.TarError as exc:
        raise RefusedError(f"git archive did not produce a readable tar: {exc}") from exc
    return names, selected


def _extract_release(release_dir: Path, tag: str, selected: list[tuple[str, bytes | None]]) -> None:
    """Write the entries into `release/<tag>/`, replacing an earlier extraction whole.

    Each target is resolved and checked to lie inside the staging directory before it is
    written, a second line behind the name checks in `_release_members`.

    Raises:
        OSError: When a write or rename fails, or a target would land outside the staging
            directory. When the new tree cannot take the earlier one's place and the earlier
            one cannot be moved back, the message names where the earlier one is kept.
    """
    release_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{tag}.", dir=release_dir))
    try:
        staging_resolved = staging.resolve()
        for name, data in selected:
            path = staging.joinpath(*name.split("/"))
            resolved = path.resolve()
            if resolved == staging_resolved or not resolved.is_relative_to(staging_resolved):
                raise OSError(f"archive entry {name!r} would be written outside {staging}")
            if data is None:
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
        target = release_dir / tag
        if target.exists():
            old = Path(tempfile.mkdtemp(prefix=f".{tag}.old.", dir=release_dir))
            kept = old / tag
            target.rename(kept)
            try:
                staging.rename(target)
            except OSError as exc:
                try:
                    kept.rename(target)
                except OSError as restore_exc:
                    raise OSError(
                        f"{exc}; the earlier release/{tag}/ could not be moved back "
                        f"({restore_exc}) and is kept at {kept}"
                    ) from exc
                old.rmdir()
                raise OSError(f"{exc}; the earlier release/{tag}/ is back in place") from exc
            shutil.rmtree(old)
        else:
            staging.rename(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _built_image_id(ctx: Context, image: str) -> str | None:
    inspected = ctx.runner(["docker", "image", "inspect", "--format", "{{.Id}}", image])
    image_id = inspected.stdout.decode("ascii", "replace").strip()
    return image_id if inspected.returncode == 0 and image_id else None


def cmd_release_build(args: argparse.Namespace, ctx: Context) -> None:
    """Build `mist-backend:<tag>` and `mist-extraction:<tag>` from a tag's archive, unpack it.

    Both images are built from the same `git archive` of the commit, never from a working tree,
    so what runs is exactly what the tag names. Both image IDs are recorded because a later
    `docker build -t <image>:<tag>` anywhere could re-point the tag; `up`, `switch` and
    `extraction apply` compare against the record. The extraction image carries the commit as
    its revision label, since its Dockerfile takes no build argument for it.
    """
    if (args.tag is None) == (args.rehearsal is None):
        raise RefusedError("give exactly one of TAG and --rehearsal REF")
    root = _absolute_path(args.root, ctx.home)
    _instance_name(root)
    repo = _absolute_path(args.repo, ctx.home) if args.repo else _default_repo(ctx)

    if args.rehearsal is not None:
        if args.rehearsal.startswith("-"):
            raise RefusedError(f"--rehearsal {args.rehearsal!r} is not a git ref")
        commit = _rev_parse_commit(ctx, repo, args.rehearsal)
        if commit is None:
            raise RefusedError(f"{args.rehearsal!r} does not resolve to a commit in {repo}")
        tag = f"{REHEARSAL_TAG_PREFIX}{commit[:7]}"
    else:
        tag = args.tag
        commit = _resolve_release(ctx, repo, tag)

    archived = ctx.runner(["git", "-C", str(repo), "archive", "--format=tar", commit])
    if archived.returncode != 0:
        raise FailedError(
            f"git archive of {commit} failed, nothing changed: {_stderr_tail(archived)}"
        )
    names, selected = _release_members(archived.stdout)
    missing = [
        path for path in (DOCKERFILE, EXTRACTION_DOCKERFILE, COMPOSE_FILE) if path not in names
    ]
    if missing:
        raise RefusedError(f"{commit} has no {', '.join(missing)}; it cannot be a daily release")

    image = f"{IMAGE_REPOSITORY}:{tag}"
    build = ctx.runner(
        [
            "docker",
            "build",
            "-f",
            DOCKERFILE,
            "-t",
            image,
            "--build-arg",
            f"MIST_RELEASE_TAG={tag}",
            "--build-arg",
            f"MIST_RELEASE_COMMIT={commit}",
            "-",
        ],
        stdin=archived.stdout,
    )
    if build.returncode != 0:
        raise FailedError(
            f"docker build of {image} failed; release/ and ops/ unchanged: {_stderr_tail(build)}"
        )

    image_id = _built_image_id(ctx, image)
    if image_id is None:
        raise FailedError(
            f"built {image} but could not read its image ID; "
            f"release/{tag}/ and ops/release-{tag}.json not written"
        )

    extraction_image = f"{EXTRACTION_IMAGE_REPOSITORY}:{tag}"
    extraction_build = ctx.runner(
        [
            "docker",
            "build",
            "-f",
            EXTRACTION_DOCKERFILE,
            "-t",
            extraction_image,
            "--label",
            f"{REVISION_LABEL}={commit}",
            "-",
        ],
        stdin=archived.stdout,
    )
    if extraction_build.returncode != 0:
        raise FailedError(
            f"built {image} ({image_id}), but docker build of {extraction_image} failed; "
            f"release/ and ops/ unchanged: {_stderr_tail(extraction_build)}"
        )
    extraction_image_id = _built_image_id(ctx, extraction_image)
    if extraction_image_id is None:
        raise FailedError(
            f"built {image} ({image_id}) and {extraction_image} but could not read the latter's "
            f"image ID; release/{tag}/ and ops/release-{tag}.json not written"
        )

    try:
        _extract_release(root / "release", tag, selected)
    except OSError as exc:
        raise FailedError(
            f"built {image} ({image_id}) and {extraction_image} ({extraction_image_id}) but could "
            f"not extract release/{tag}/: {exc}; ops/release-{tag}.json not written"
        ) from exc

    record = {
        "tag": tag,
        "commit": commit,
        "image_id": image_id,
        "extraction_image_id": extraction_image_id,
        "built_at": _format_utc(ctx.clock()),
    }
    record_path = root / "ops" / f"release-{tag}.json"
    try:
        record_path.parent.mkdir(parents=True, exist_ok=True)
        _write_atomic(record_path, (json.dumps(record, indent=2) + "\n").encode("utf-8"))
    except OSError as exc:
        raise FailedError(
            f"built {image} ({image_id}) and {extraction_image} ({extraction_image_id}) and "
            f"extracted release/{tag}/, but could not write {record_path}: {exc}"
        ) from exc

    print(f"built {image} ({image_id}) from {commit}")
    print(f"built {extraction_image} ({extraction_image_id}) from {commit}")
    print(f"extracted release/{tag}/ and recorded {record_path}")


# --- The instance's release and configuration ------------------------------------------------

TAG_LINE = re.compile(rb"^MIST_RELEASE_TAG=[^\r\n]*", re.M)
CONTAINER_INSPECT_FORMAT = "{{.Name}} {{.Image}} {{json .Config.Labels}}"


def _tag_is_valid(tag: str) -> bool:
    """Whether `tag` is a release or rehearsal tag; only those are ever joined into a path."""
    return bool(RELEASE_TAG_PATTERN.fullmatch(tag) or REHEARSAL_TAG_PATTERN.fullmatch(tag))


def _check_tag_name(tag: str, where: str) -> None:
    if not _tag_is_valid(tag):
        raise RefusedError(
            f"{where} {tag!r} is neither daily-YYYY.MM.DD[.N] nor {REHEARSAL_TAG_PREFIX}<sha7>"
        )


def _read_env(path: Path) -> dict[str, str]:
    """The `KEY=VALUE` lines of `daily.env`.

    Comment and blank lines are skipped, surrounding whitespace and one pair of matching outer
    quotes are stripped, and a repeated key keeps its last value.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RefusedError(f"{path} does not exist; run init and fill it in") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise RefusedError(f"cannot read {path}: {exc}") from exc
    values: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


@dataclass(frozen=True, slots=True)
class Instance:
    """One instance root, its `daily.env` and the release tag the commands act on.

    `tag` is `daily.env`'s `MIST_RELEASE_TAG`, except inside `switch`, which acts on the tag it
    switches to before it rewrites the file.
    """

    root: Path
    name: str
    env_path: Path
    env: Mapping[str, str]
    tag: str

    @property
    def project(self) -> str:
        """The compose project, named from the marker so two instances never share one."""
        return f"mist-{self.name}"

    @property
    def ops(self) -> Path:
        """The operations directory: release records, backup status and both markers."""
        return self.root / "ops"

    def release_dir(self) -> Path:
        """Where `release-build` extracted the tag's `deploy/` and `scripts/daily/`."""
        return self.root / "release" / self.tag

    def compose_file(self) -> Path:
        """The tag's own compose file, so a change on main reaches the instance only by tag."""
        return self.release_dir().joinpath(*COMPOSE_FILE.split("/"))

    def compose(self) -> list[str]:
        """The `docker compose` prefix every project command starts with (item 5).

        With the device bind set it adds the overlay and its profile, so `down` and `status` see
        the forwarder `up` started, and no call recreates mist-backend without its device mount.
        """
        prefix = [
            "docker",
            "compose",
            "-p",
            self.project,
            "-f",
            str(self.compose_file()),
            "--env-file",
            str(self.env_path),
        ]
        if _device_bind(self):
            prefix += ["-f", str(_devices_compose_file(self)), "--profile", DEVICES_PROFILE]
        return prefix


def _device_bind(inst: Instance) -> str:
    """`MIST_DAILY_DEVICE_BIND`, empty when unset; the one switch for every device change."""
    return inst.env.get(DEVICE_BIND_KEY, "").strip()


def _devices_compose_file(inst: Instance) -> Path:
    """The tag's own device overlay, beside its compose file."""
    return inst.release_dir().joinpath(*DEVICES_COMPOSE_FILE.split("/"))


def _load_instance(raw_root: str | Path, ctx: Context, *, require_release: bool = True) -> Instance:
    """Read the root's marker and `daily.env`; refuses when its release cannot be run.

    With `require_release` false the tag is not checked, for `switch` (which replaces it) and
    `down` (which must stop an instance whose release directory is gone).
    """
    root = _absolute_path(str(raw_root), ctx.home)
    name = _instance_name(root)
    env_path = root / ENV_FILE
    env = _read_env(env_path)
    inst = Instance(root, name, env_path, env, env.get("MIST_RELEASE_TAG", ""))
    if require_release:
        _check_release_present(inst, f"{env_path}'s MIST_RELEASE_TAG")
    return inst


def _check_release_present(inst: Instance, where: str) -> None:
    _check_tag_name(inst.tag, where)
    if not inst.compose_file().is_file():
        raise RefusedError(
            f"{inst.compose_file()} does not exist; run release-build {inst.tag} "
            f"--root {inst.root} first"
        )


def _check_ports(inst: Instance) -> None:
    for key, canonical in CANONICAL_PORTS.items():
        value = inst.env.get(key)
        if value is None:
            continue
        if inst.name == DAILY_INSTANCE and value != canonical:
            raise RefusedError(
                f"{inst.env_path} sets {key}={value}; the instance named daily keeps the "
                f"canonical port {canonical}, which the live-endpoint denylists protect"
            )
        if value.isdigit() and int(value) in ALLOWLISTED_PORTS:
            raise RefusedError(
                f"{inst.env_path} sets {key}={value}, an eval, dev or staging port "
                f"({', '.join(str(p) for p in sorted(ALLOWLISTED_PORTS))}); no instance may "
                "answer on one"
            )
    _check_device_settings(inst)


def _check_device_settings(inst: Instance) -> None:
    """Refuse a device port or bind that would publish the wrong thing, or the bind alone.

    The bind is the forwarder's only host address, so anything but one RFC 1918 IPv4 address
    (all interfaces, loopback, a public address) is refused rather than published. Compose would
    refuse a bind without `MIST_DEVICE_DIR` only once Docker runs; this refuses it first.
    """
    port = inst.env.get(DEVICE_PORT_KEY)
    if port is not None and port != "":
        websocket = CANONICAL_PORTS["MIST_DAILY_WS_PORT"]
        usable = port.isascii() and port.isdigit() and len(port) <= 5 and 1 <= int(port) <= 65535
        if not usable:
            raise RefusedError(f"{inst.env_path} sets {DEVICE_PORT_KEY}={port}, which is no port")
        if port == websocket or int(port) in ALLOWLISTED_PORTS:
            raise RefusedError(
                f"{inst.env_path} sets {DEVICE_PORT_KEY}={port}, the WebSocket port {websocket} "
                f"or an eval, dev or staging port; the forwarder needs a port of its own"
            )
    bind = _device_bind(inst)
    if not bind:
        return
    try:
        address = ipaddress.IPv4Address(bind)
    except ValueError:
        address = None
    if address is None or not any(address in net for net in PRIVATE_IPV4_NETWORKS):
        raise RefusedError(
            f"{inst.env_path} sets {DEVICE_BIND_KEY}={bind}; it must be the PC's LAN address, "
            "one private IPv4 address (10/8, 172.16/12 or 192.168/16)"
        )
    if not inst.env.get(DEVICE_DIR_KEY, "").strip():
        raise RefusedError(
            f"{inst.env_path} sets {DEVICE_BIND_KEY} but not {DEVICE_DIR_KEY}, the devices "
            "directory mist-backend mounts"
        )


def _backup_host_path(inst: Instance) -> str:
    return inst.env.get("MIST_BACKUP_HOST_PATH", "")


def _check_backup_path(inst: Instance, ctx: Context) -> None:
    """Refuse a backup destination a loss of the instance, or a repository tool, could reach.

    The destination guard inside mist-backup sees only `/backup`, so the host path is checked
    here, on the host.
    """
    raw = _backup_host_path(inst)
    if not raw:
        return
    label = f"MIST_BACKUP_HOST_PATH={raw}"
    if not (Path(raw).is_absolute() or raw.startswith("~")):
        raise RefusedError(f"{label} in {inst.env_path} is not an absolute path")
    _check_protected_path(_absolute_path(raw, ctx.home), label, inst, ctx)


def _check_protected_path(path: Path, label: str, inst: Instance, ctx: Context) -> None:
    """Refuse a resolved path inside the instance root, a live root, or a git work tree.

    The backup host path and a drill root share these refusals (item 5): a loss of the instance
    must not take either with it, and neither may sit where a repository tool can reach it.
    """
    if path.is_relative_to(inst.root):
        raise RefusedError(f"{label} is inside the instance root {inst.root}")
    for line in _read_registry(_registry_path(ctx)):
        entry = line.strip()
        if entry and Path(entry).is_absolute() and path.is_relative_to(Path(entry).resolve()):
            raise RefusedError(f"{label} is inside the registered live instance root {entry}")
    for candidate in (path, *path.parents):
        if (candidate / LIVE_INSTANCE_MARKER).exists():
            raise RefusedError(f"{label} is inside {candidate}, which holds {LIVE_INSTANCE_MARKER}")
    work_tree = _git_work_tree_holding(path)
    if work_tree is not None:
        raise RefusedError(f"{label} is inside the git work tree {work_tree}")


def _check_static_configuration(inst: Instance, ctx: Context) -> None:
    """Item 5's refusals that need no Docker: the ports and the backup host path."""
    _check_ports(inst)
    _check_backup_path(inst, ctx)


def _recorded_image_id(inst: Instance) -> str:
    """The image ID `release-build` recorded for the instance's tag."""
    record_path = inst.ops / f"release-{inst.tag}.json"
    fix = f"run release-build {inst.tag} --root {inst.root}"
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RefusedError(f"{record_path} does not exist; {fix}") from exc
    except (OSError, ValueError) as exc:
        raise RefusedError(f"cannot read {record_path}: {exc}; {fix}") from exc
    image_id = record.get("image_id") if isinstance(record, dict) else None
    if not isinstance(image_id, str) or not image_id:
        raise RefusedError(f"{record_path} records no image_id; {fix}")
    return image_id


def _check_release_image(inst: Instance, ctx: Context) -> None:
    """Refuse when `mist-backend:<tag>` is missing or is not the image the release recorded.

    A `docker build -t mist-backend:<tag>` run anywhere re-points the tag; the recorded ID is
    what `release-build` built from the tag's archive.
    """
    recorded = _recorded_image_id(inst)
    image = f"{IMAGE_REPOSITORY}:{inst.tag}"
    fix = f"run release-build {inst.tag} --root {inst.root}"
    inspected = ctx.runner(["docker", "image", "inspect", "--format", "{{.Id}}", image])
    actual = inspected.stdout.decode("ascii", "replace").strip()
    if inspected.returncode != 0 or not actual:
        raise RefusedError(f"there is no image {image} ({_stderr_tail(inspected)}); {fix}")
    if actual != recorded:
        raise RefusedError(
            f"{image} is {actual}, but ops/release-{inst.tag}.json records {recorded}: the tag "
            f"was re-pointed by another build; {fix}"
        )


def _require_backup_path(inst: Instance) -> None:
    if not _backup_host_path(inst):
        raise RefusedError(f"{inst.env_path} does not set MIST_BACKUP_HOST_PATH")


# --- Reading the project's containers --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ContainerInfo:
    """What `docker inspect` says about one container."""

    name: str
    image_id: str
    labels: Mapping[str, str]


def _parse_ps_json(stdout: bytes) -> list[dict]:
    """`docker compose ps --format json` rows, as one JSON array or one object per line.

    Raises:
        ValueError: When the output is neither.
    """
    text = stdout.decode("utf-8", "replace").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        raise ValueError("docker compose ps printed neither a JSON array nor JSON objects")
    return [row for row in parsed if isinstance(row, dict)]


def _inspect_container(ctx: Context, container: str) -> ContainerInfo:
    result = ctx.runner(["docker", "inspect", "--format", CONTAINER_INSPECT_FORMAT, container])
    if result.returncode != 0:
        raise FailedError(f"cannot inspect container {container}: {_stderr_tail(result)}")
    parts = result.stdout.decode("utf-8", "replace").strip().split(" ", 2)
    try:
        labels = json.loads(parts[2]) if len(parts) == 3 else None
    except json.JSONDecodeError:
        labels = None
    if len(parts) != 3 or not (labels is None or isinstance(labels, dict)):
        raise FailedError(f"cannot read docker inspect's output for container {container}")
    return ContainerInfo(parts[0].lstrip("/"), parts[1], labels or {})


def _project_containers(ctx: Context, project: str) -> list[ContainerInfo]:
    """Every container labelled with the project, running or not, whatever file made it."""
    listed = ctx.runner(
        [
            "docker",
            "ps",
            "--all",
            "--quiet",
            "--no-trunc",
            "--filter",
            f"label={PROJECT_LABEL}={project}",
        ]
    )
    if listed.returncode != 0:
        raise FailedError(f"cannot list project {project}'s containers: {_stderr_tail(listed)}")
    ids = listed.stdout.decode("utf-8", "replace").split()
    return [_inspect_container(ctx, container) for container in ids]


def _normalized(path: str) -> str:
    return os.path.normcase(os.path.normpath(path))


def _foreign_containers(inst: Instance, containers: Sequence[ContainerInfo]) -> list[str]:
    """`<name> uses <files>` for each container not made from the release's own compose files.

    Compose records the files a container was created from in a label, so a
    `docker compose -p mist-daily -f <another file> up` shows here. The release's files are its
    compose file alone, or that file then its device overlay. Compose recreates a container only
    when its configuration changes, so the services the overlay leaves alone may carry either
    label; mist-backend, which the overlay changes, must carry the overlay exactly when the bind
    is set, and the forwarder, which only the overlay defines, whenever the bind is set.
    """
    alone = [_normalized(str(inst.compose_file()))]
    with_overlay = [*alone, _normalized(str(_devices_compose_file(inst)))]
    bind = bool(_device_bind(inst))
    foreign = []
    for container in containers:
        label = str(container.labels.get(CONFIG_FILES_LABEL, ""))
        files = [_normalized(entry) for entry in label.split(",") if entry]
        service = container.labels.get(SERVICE_LABEL)
        if service == DEVICE_FORWARD_SERVICE and bind:
            accepted = [with_overlay]
        elif service == BACKEND_SERVICE:
            accepted = [with_overlay] if bind else [alone]
        else:
            accepted = [alone, with_overlay]
        if files not in accepted:
            foreign.append(f"{container.name} uses {label or '(no config-file label)'}")
    return foreign


# --- up and down -----------------------------------------------------------------------------


def _wait_for_docker(ctx: Context, wait_minutes: int) -> None:
    """Poll `docker info` until the daemon answers; Docker Desktop starts at sign-in too."""
    deadline = ctx.monotonic() + wait_minutes * 60
    while True:
        result = ctx.runner(["docker", "info", "--format", "{{.ServerVersion}}"])
        if result.returncode == 0:
            return
        if ctx.monotonic() >= deadline:
            raise FailedError(
                f"docker info did not succeed within {wait_minutes} minute(s) "
                f"({_stderr_tail(result)}); nothing was started"
            )
        ctx.sleep(DOCKER_POLL_S)


def _ensure_share_network(inst: Instance, ctx: Context) -> None:
    """Create the external network mist-llm joins; compose refuses to start without it."""
    name = inst.env.get("MIST_LLM_SHARE_NETWORK") or DEFAULT_SHARE_NETWORK
    if ctx.runner(["docker", "network", "inspect", name]).returncode == 0:
        return
    created = ctx.runner(["docker", "network", "create", name])
    if created.returncode != 0:
        raise FailedError(
            f"cannot create the share network {name}; nothing was started: "
            f"{_stderr_tail(created)}"
        )
    print(f"created the share network {name}")


def instance_up(inst: Instance, ctx: Context, wait_minutes: int) -> None:
    """Start the instance from its release, then check what runs is that release.

    mist-backup starts in a second call, so a broken backup destination never keeps the
    services that answer turns from starting. Extraction is applied after both, and a failed
    apply is reported, not fatal: turns queue while it is down. A release whose record has no
    extraction image predates night extraction, so it gets a note instead of an apply, and a
    rollback to it exits 0.
    """
    _check_static_configuration(inst, ctx)
    deadline = ctx.monotonic() + wait_minutes * 60
    _wait_for_docker(ctx, wait_minutes)
    _check_release_image(inst, ctx)
    _ensure_share_network(inst, ctx)

    yielded = (inst.ops / GPU_YIELDED_MARKER).exists()
    services = [s for s in SERVICES if not (yielded and s == LLM_SERVICE)]
    started = ctx.runner([*inst.compose(), "up", "-d", *services])
    if started.returncode != 0:
        raise FailedError(
            f"docker compose up -d {' '.join(services)} failed, so they may be partly started: "
            f"{_stderr_tail(started)}"
        )
    print(f"started {', '.join(services)} from release {inst.tag}")
    if yielded:
        print(f"gaming mode is on, so {LLM_SERVICE} stays stopped; `gaming-mode off` starts it")

    problems: list[str] = []
    if (inst.ops / BACKUPS_ARMED_MARKER).exists():
        backup = ctx.runner(
            [*inst.compose(), "--profile", BACKUP_PROFILE, "up", "-d", BACKUP_SERVICE]
        )
        if backup.returncode != 0:
            problems.append(
                f"{BACKUP_SERVICE} did not start ({_stderr_tail(backup)}); the other services "
                "keep running"
            )
        else:
            print(f"started {BACKUP_SERVICE}: backups are armed")
    if _extraction_image_id(inst) is None:
        print(f"release {inst.tag} records no extraction image; extraction apply skipped")
    else:
        try:
            _apply_extraction(inst, ctx)
        except InstanceError as exc:
            problems.append(
                f"extraction apply did not converge ({exc.label}: {exc}); turns queue until it does"
            )
    bind = _device_bind(inst)
    if bind:
        problem = _start_device_forward(inst, ctx, deadline)
        if problem is not None:
            problems.append(problem)
    try:
        containers = _project_containers(ctx, inst.project)
    except FailedError as exc:
        problems.append(f"the containers' compose files could not be checked: {exc}")
    else:
        if not bind:
            containers = _remove_leftover_forwarders(ctx, containers, problems)
        expected = str(inst.compose_file())
        if bind:
            expected = f"{expected} with {_devices_compose_file(inst)}"
        foreign = _foreign_containers(inst, containers)
        problems.extend(f"container {entry}, not {expected}" for entry in foreign)
    if problems:
        raise FailedError(f"release {inst.tag} is up, but: " + "; ".join(problems))


def _start_device_forward(inst: Instance, ctx: Context, deadline: float) -> str | None:
    """Start the forwarder, retrying until `deadline`; the problem to report, or None.

    It starts in its own call, after the services, because its port binds the LAN address, which
    can be missing at sign-in; a failure here never stops the services the desktop uses.
    """
    while True:
        started = ctx.runner([*inst.compose(), "up", "-d", DEVICE_FORWARD_SERVICE])
        if started.returncode == 0:
            print(f"started {DEVICE_FORWARD_SERVICE} on {_device_bind(inst)}")
            return None
        if ctx.monotonic() >= deadline:
            return (
                f"{DEVICE_FORWARD_SERVICE} did not start by the --wait-minutes deadline "
                f"({_stderr_tail(started)}); room devices cannot connect, and the other "
                "services keep running"
            )
        ctx.sleep(DEVICE_FORWARD_RETRY_S)


def _remove_leftover_forwarders(
    ctx: Context, containers: Sequence[ContainerInfo], problems: list[str]
) -> list[ContainerInfo]:
    """Remove each forwarder left from when the bind was set; the containers that remain.

    Without the bind no compose call names the overlay, so compose would never stop it, and it
    would go on publishing the LAN port. `rm -f` without `-v` removes the container only.
    """
    remaining = []
    for container in containers:
        if container.labels.get(SERVICE_LABEL) != DEVICE_FORWARD_SERVICE:
            remaining.append(container)
            continue
        removed = ctx.runner(["docker", "rm", "-f", container.name])
        if removed.returncode == 0:
            print(f"removed the leftover {DEVICE_FORWARD_SERVICE} {container.name}")
        else:
            problems.append(
                f"the leftover {DEVICE_FORWARD_SERVICE} {container.name} could not be removed "
                f"({_stderr_tail(removed)}); it may still publish the device port"
            )
    return remaining


def cmd_up(args: argparse.Namespace, ctx: Context) -> None:
    """Start the instance; the sign-in task runs this from `<root>/bin/`."""
    instance_up(_load_instance(args.root, ctx), ctx, args.wait_minutes)


def cmd_down(args: argparse.Namespace, ctx: Context) -> None:
    """Stop every container of the project and remove nothing.

    `stop`, never `down`: the containers, networks and volumes stay, so nothing a later
    `up` needs is deleted. Profiles `backup` and `extraction` are enabled so mist-backup and
    both extraction services stop too.
    """
    inst = _load_instance(args.root, ctx, require_release=False)
    if _tag_is_valid(inst.tag) and inst.compose_file().is_file():
        argv = [
            *inst.compose(),
            "--profile",
            BACKUP_PROFILE,
            "--profile",
            EXTRACTION_PROFILE,
            "stop",
        ]
    else:
        argv = ["docker", "compose", "-p", inst.project, "stop"]
    result = ctx.runner(argv)
    if result.returncode != 0:
        raise FailedError(
            f"stopping project {inst.project} failed, so it may be partly stopped: "
            f"{_stderr_tail(result)}"
        )
    print(f"stopped project {inst.project}")


# --- switch ----------------------------------------------------------------------------------


def _read_backend_running(inst: Instance, ctx: Context) -> bool:
    """Whether the instance's own mist-backend runs, asked of compose by project.

    Scoped by `-p <project>` and the service label compose reports, never by container name,
    so a development stack's container also called mist-backend is never read.

    Raises:
        ValueError: When the listing fails or cannot be parsed; the message says which.
    """
    result = ctx.runner(
        ["docker", "compose", "-p", inst.project, "ps", "--status", "running", "--format", "json"]
    )
    if result.returncode != 0:
        raise ValueError(f"cannot tell whether {inst.project} runs ({_stderr_tail(result)})")
    try:
        rows = _parse_ps_json(result.stdout)
    except ValueError as exc:
        raise ValueError(f"cannot read {inst.project}'s containers ({exc})") from exc
    return any(row.get("Service") == BACKEND_SERVICE for row in rows)


def _backend_running(inst: Instance, ctx: Context) -> bool:
    """`_read_backend_running` for switch, whose fix is its own `--skip-backup`."""
    try:
        return _read_backend_running(inst, ctx)
    except ValueError as exc:
        raise RefusedError(
            f"{exc}; nothing changed; pass --skip-backup to switch without the pre-switch backup"
        ) from exc


def cmd_switch(args: argparse.Namespace, ctx: Context) -> int | None:
    """Move the instance to another built release: back up, rewrite the tag, copy, `up`.

    `switch <previous tag>` is the rollback. It restores the code but not a data migration the
    newer release made, which is why a running instance is backed up first, on the release
    that wrote its data.

    Before `daily.env` moves, the extraction containers are removed, found by their project
    and service labels rather than through a compose file, so the target release need not
    define the services. One left stopped by an older release's compose file would otherwise
    read as foreign to `up` and `status`; the services hold no state.
    """
    _check_tag_name(args.tag, "TAG")
    current = _load_instance(args.root, ctx, require_release=False)
    if args.tag.startswith(REHEARSAL_TAG_PREFIX) and current.name == DAILY_INSTANCE:
        raise RefusedError(f"{args.tag} is a rehearsal build; the daily instance runs releases")
    target = replace(current, tag=args.tag)
    if not target.release_dir().is_dir():
        raise RefusedError(
            f"{target.release_dir()} does not exist; run release-build {args.tag} "
            f"--root {target.root} first"
        )
    _check_release_present(target, "TAG")
    try:
        env_bytes = target.env_path.read_bytes()
    except OSError as exc:
        raise RefusedError(f"cannot read {target.env_path}: {exc}") from exc
    tag_lines = len(TAG_LINE.findall(env_bytes))
    if tag_lines != 1:
        raise RefusedError(
            f"{target.env_path} has {tag_lines} MIST_RELEASE_TAG= lines; switch rewrites one"
        )
    _check_static_configuration(target, ctx)
    _check_release_image(target, ctx)

    if _backup_host_path(current) and not args.skip_backup and _backend_running(current, ctx):
        # The backup runs on the release that wrote the data, before daily.env moves.
        _check_release_present(current, f"the running release, {current.env_path}'s tag")
        code = _backup_once(current, ctx)
        if code != EXIT_OK:
            print(
                f"instance.py switch: the pre-switch backup exited {code}; nothing changed. "
                "Pass --skip-backup to switch without it.",
                file=sys.stderr,
            )
            return code

    try:
        _remove_extraction_by_label(current, ctx)
    except InstanceError as exc:
        raise FailedError(
            f"could not remove the extraction containers before the switch ({exc}); "
            f"{target.env_path} and bin/ unchanged"
        ) from exc

    new_bytes = TAG_LINE.sub(f"MIST_RELEASE_TAG={args.tag}".encode(), env_bytes, count=1)
    try:
        _write_atomic(target.env_path, new_bytes)
    except OSError as exc:
        raise FailedError(f"cannot rewrite {target.env_path}: {exc}; nothing changed") from exc

    # `bin/instance.py` and `bin/gamewatch.py` come from the same release (item 17).
    copied: list[str] = []
    try:
        for source in sorted(target.release_dir().joinpath("scripts", "daily").glob("*.py")):
            if source.is_file():
                _write_atomic(target.root / "bin" / source.name, source.read_bytes())
                copied.append(source.name)
    except OSError as exc:
        done = ", ".join(copied) if copied else "nothing"
        raise FailedError(
            f"{target.env_path} now names {args.tag}, but copying its scripts into bin/ stopped "
            f"at {exc} (copied: {done}); run switch {args.tag} again"
        ) from exc
    print(f"{target.env_path} now names {args.tag}; bin/ holds {', '.join(copied) or 'nothing'}")

    try:
        instance_up(target, ctx, DEFAULT_UP_WAIT_MINUTES)
    except InstanceError as exc:
        raise FailedError(
            f"{target.env_path} now names {args.tag} and bin/ holds its scripts, but up failed: "
            f"{exc}"
        ) from exc
    return None


# --- Backups ---------------------------------------------------------------------------------


def _echo(result: CommandResult) -> None:
    sys.stdout.write(result.stdout.decode("utf-8", "replace"))
    sys.stderr.write(result.stderr.decode("utf-8", "replace"))


def _backup_once(inst: Instance, ctx: Context) -> int:
    """Run one backup attempt in a one-shot container and return its exit code."""
    _require_backup_path(inst)
    _check_static_configuration(inst, ctx)
    _check_release_image(inst, ctx)
    result = ctx.runner(
        [*inst.compose(), "--profile", BACKUP_PROFILE, "run", "--rm", BACKUP_SERVICE, *BACKUP_ONCE]
    )
    _echo(result)
    return result.returncode


def cmd_backup_now(args: argparse.Namespace, ctx: Context) -> int:
    """One backup attempt now, whether or not backups are armed."""
    code = _backup_once(_load_instance(args.root, ctx), ctx)
    if code != EXIT_OK:
        print(f"instance.py backup-now: the backup attempt exited {code}", file=sys.stderr)
    return code


def cmd_arm_backups(args: argparse.Namespace, ctx: Context) -> None:
    """Mark backups armed, so every `up` starts mist-backup, and start it now.

    The deploy plan arms backups only after a restore drill passes: rehearse first, arm second.
    """
    inst = _load_instance(args.root, ctx)
    _require_backup_path(inst)
    _check_static_configuration(inst, ctx)
    _check_release_image(inst, ctx)
    marker = inst.ops / BACKUPS_ARMED_MARKER
    record = {"v": 1, "armed_at": _format_utc(ctx.clock())}
    try:
        _write_atomic(marker, (json.dumps(record) + "\n").encode("utf-8"))
    except OSError as exc:
        raise FailedError(f"cannot write {marker}: {exc}; nothing started") from exc
    result = ctx.runner([*inst.compose(), "--profile", BACKUP_PROFILE, "up", "-d", BACKUP_SERVICE])
    if result.returncode != 0:
        raise FailedError(
            f"wrote {marker}, but {BACKUP_SERVICE} did not start ({_stderr_tail(result)}); the "
            "next up starts it, or run arm-backups again"
        )
    print(f"backups armed: wrote {marker} and started {BACKUP_SERVICE}")


def cmd_disarm_backups(args: argparse.Namespace, ctx: Context) -> None:
    """Remove the armed marker first, so no later `up` restarts mist-backup, then stop it."""
    inst = _load_instance(args.root, ctx)
    marker = inst.ops / BACKUPS_ARMED_MARKER
    try:
        marker.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise FailedError(f"cannot remove {marker}: {exc}; nothing stopped") from exc
    result = ctx.runner([*inst.compose(), "--profile", BACKUP_PROFILE, "stop", BACKUP_SERVICE])
    if result.returncode != 0:
        raise FailedError(
            f"removed {marker}, but stopping {BACKUP_SERVICE} failed, so it may still be "
            f"running: {_stderr_tail(result)}"
        )
    print(f"backups disarmed: removed {marker} and stopped {BACKUP_SERVICE}")


def _check_plain_name(value: str, what: str) -> None:
    """Refuse a name that could leave the directory it is joined to, or read as a flag."""
    if not PLAIN_NAME.fullmatch(value):
        raise RefusedError(
            f"{what} {value!r} is not a plain name (letters, digits, '.', '_' and '-', "
            "starting with a letter or digit)"
        )


def cmd_accept_backup_baseline(args: argparse.Namespace, ctx: Context) -> int:
    """Make a reviewed suspect backup the baseline later attempts are judged against (item 6)."""
    _check_plain_name(args.label, "LABEL")
    inst = _load_instance(args.root, ctx)
    _require_backup_path(inst)
    _check_static_configuration(inst, ctx)
    _check_release_image(inst, ctx)
    result = ctx.runner(
        [
            *inst.compose(),
            "--profile",
            BACKUP_PROFILE,
            "run",
            "--rm",
            BACKUP_SERVICE,
            *ACCEPT_SUSPECT,
            args.label,
        ]
    )
    _echo(result)
    if result.returncode != EXIT_OK:
        print(
            f"instance.py accept-backup-baseline: the scheduler exited {result.returncode}",
            file=sys.stderr,
        )
    return result.returncode


# --- The restore drill and promotion ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Drill:
    """One drill root and what `drill-start` recorded in its `drill.json`."""

    root: Path
    volume: str
    artifact: str
    started_at: datetime

    def standby(self, name: str) -> Path:
        """The restore's target for `state` or `mist-memory`, and what `promote` moves in."""
        return self.root / DRILL_STANDBY / name


def _drill_project(inst: Instance) -> str:
    return f"{inst.project}-drill"


def _drill_compose(inst: Instance) -> list[str]:
    """The drill project's `docker compose` prefix: the release's drill file, the instance's env.

    `--env-file daily.env` gives the drill graph the instance's password, so a promoted volume
    keeps the authentication the instance's Neo4j expects (item 8).
    """
    return [
        "docker",
        "compose",
        "-p",
        _drill_project(inst),
        "-f",
        str(inst.release_dir().joinpath(*DRILL_COMPOSE_FILE.split("/"))),
        "--env-file",
        str(inst.env_path),
    ]


def _drill_env(ctx: Context, drill: Drill) -> dict[str, str]:
    """The process environment plus the two drill variables, which never reach `daily.env`."""
    return {
        **ctx.environ,
        "MIST_DRILL_NEO4J_VOLUME": drill.volume,
        "MIST_DRILL_ROOT": drill.root.as_posix(),
    }


def _run_drill(inst: Instance, ctx: Context, drill: Drill, *tail: str) -> CommandResult:
    return ctx.runner([*_drill_compose(inst), *tail], env=_drill_env(ctx, drill))


def _restore_in_drill(*tail: str) -> list[str]:
    """`run --rm mist-restore -m <module> ...`: the release image runs `python -m` there."""
    return [
        "--profile",
        DRILL_RUN_PROFILE,
        "run",
        "--rm",
        DRILL_RESTORE_SERVICE,
        "-m",
        *tail,
    ]


def _drill_root(raw: str, inst: Instance, ctx: Context) -> Path:
    path = _absolute_path(raw, ctx.home)
    _check_protected_path(path, f"drill root {path}", inst, ctx)
    return path


def _check_drill_file(inst: Instance) -> None:
    drill_file = inst.release_dir().joinpath(*DRILL_COMPOSE_FILE.split("/"))
    if not drill_file.is_file():
        raise RefusedError(
            f"{drill_file} does not exist; release {inst.tag} predates the restore drill, so "
            "switch to a release that carries it"
        )


def _parse_utc(raw: object) -> datetime | None:
    if not isinstance(raw, str):
        return None
    try:
        moment = datetime.strptime(raw, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return None
    return moment.replace(tzinfo=UTC)


def _read_drill(root: Path) -> Drill:
    """The drill `drill-start` prepared in `root`.

    Raises:
        RefusedError: When `root` holds no readable `drill.json` with a plain volume and
            artifact name and a start time.
    """
    path = root / DRILL_RECORD
    fix = "run drill-start for this drill root first"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RefusedError(f"{path} does not exist; {fix}") from exc
    except (OSError, ValueError) as exc:
        raise RefusedError(f"cannot read {path}: {exc}") from exc
    if not isinstance(record, dict):
        raise RefusedError(f"{path} is not a drill record")
    volume, artifact = record.get("volume"), record.get("artifact")
    started_at = _parse_utc(record.get("started_at"))
    if not isinstance(volume, str) or not isinstance(artifact, str) or started_at is None:
        raise RefusedError(f"{path} lacks volume, artifact or started_at")
    _check_plain_name(volume, f"{path}'s volume")
    _check_plain_name(artifact, f"{path}'s artifact")
    return Drill(root, volume, artifact, started_at)


def _check_not_instance_volume(inst: Instance, volume: str) -> None:
    if volume == inst.env.get("MIST_DAILY_NEO4J_VOLUME"):
        raise RefusedError(
            f"volume {volume} is the one {inst.env_path} names in MIST_DAILY_NEO4J_VOLUME; a "
            "drill never touches the instance's graph"
        )


def _check_drill_volume(inst: Instance, ctx: Context, volume: str) -> None:
    """Item 8's volume check, before any call that uses the volume.

    The volume must carry both labels `drill-start` gave it, and no container outside the drill
    project may mount it, so neither the instance's graph nor another instance's drill is ever
    restored into or removed.
    """
    _check_not_instance_volume(inst, volume)
    inspected = ctx.runner(["docker", "volume", "inspect", "--format", "{{json .Labels}}", volume])
    if inspected.returncode != 0:
        raise RefusedError(f"cannot inspect volume {volume}: {_stderr_tail(inspected)}")
    try:
        labels = json.loads(inspected.stdout.decode("utf-8", "replace") or "null")
    except json.JSONDecodeError:
        labels = None
    wanted = {DRILL_ROLE_LABEL: DRILL_ROLE, DRILL_INSTANCE_LABEL: inst.name}
    if not isinstance(labels, dict) or any(labels.get(k) != v for k, v in wanted.items()):
        raise RefusedError(
            f"volume {volume} is not labelled "
            + " and ".join(f"{k}={v}" for k, v in wanted.items())
            + f"; drill-start made no such volume (labels: {labels})"
        )
    listed = ctx.runner(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            f"volume={volume}",
            "--format",
            '{{.Label "' + PROJECT_LABEL + '"}}',
        ]
    )
    if listed.returncode != 0:
        raise RefusedError(
            f"cannot list the containers that mount {volume}: {_stderr_tail(listed)}"
        )
    project = _drill_project(inst)
    others = [
        line.strip() or "(no compose project)"
        for line in listed.stdout.decode("utf-8", "replace").splitlines()
        if line.strip() != project
    ]
    if others:
        raise RefusedError(
            f"volume {volume} is mounted by containers outside {project}: {', '.join(others)}"
        )


class MountReadError(Exception):
    """A service's containers, or one container's mounts, could not be read."""


def _data_volumes(ctx: Context, project: str, service: str) -> dict[str, str | None]:
    """The volume each container of `service` in `project` mounts at /data, by container id.

    The lookup is by compose label, so it finds a container whatever volume it took when compose
    created it; compose reuses that container on a later run even when the run names another
    volume. A container with no volume at /data maps to None.

    Raises:
        MountReadError: When the containers cannot be listed, or a container's mounts cannot be
            read or are not the list `docker inspect` prints.
    """
    listed = ctx.runner(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            f"label={PROJECT_LABEL}={project}",
            "--filter",
            f"label={SERVICE_LABEL}={service}",
            "-q",
        ]
    )
    if listed.returncode != 0:
        raise MountReadError(
            f"cannot list the {service} containers of {project}: {_stderr_tail(listed)}"
        )
    volumes: dict[str, str | None] = {}
    for container in listed.stdout.decode("utf-8", "replace").split():
        inspected = ctx.runner(["docker", "inspect", "--format", "{{json .Mounts}}", container])
        where = f"the mounts of {service} container {container}"
        if inspected.returncode != 0:
            raise MountReadError(f"cannot read {where}: {_stderr_tail(inspected)}")
        try:
            mounts = json.loads(inspected.stdout.decode("utf-8", "replace") or "null")
        except json.JSONDecodeError:
            mounts = None
        if not isinstance(mounts, list) or not all(isinstance(m, dict) for m in mounts):
            raise MountReadError(f"cannot read {where}: docker inspect printed no mount list")
        data = [m for m in mounts if m.get("Destination") == NEO4J_DATA_MOUNT]
        name = data[0].get("Name") if data and data[0].get("Type") == "volume" else None
        volumes[container] = name if isinstance(name, str) else None
    return volumes


def _mounts_other_than(volumes: dict[str, str | None], volume: str) -> list[str]:
    """One phrase per container whose /data is not `volume`."""
    return [
        (
            f"container {container} mounts {found} at {NEO4J_DATA_MOUNT}"
            if found
            else f"container {container} has no {NEO4J_DATA_MOUNT} volume"
        )
        for container, found in volumes.items()
        if found != volume
    ]


def _drill_neo4j_problem(
    inst: Instance, ctx: Context, drill: Drill, *, required: bool
) -> str | None:
    """Why the drill project's Neo4j is not on this drill's volume, or None when it is.

    Every drill of an instance runs in the one project `mist-<instance>-drill`, so a Neo4j
    container an earlier drill left behind would take this drill's restore, and its verify, on
    the earlier drill's volume. With no container, compose creates one on this drill's volume,
    which is no problem unless `required`.
    """
    project = _drill_project(inst)
    try:
        volumes = _data_volumes(ctx, project, DRILL_NEO4J_SERVICE)
    except MountReadError as exc:
        return str(exc)
    if not volumes:
        return f"no {DRILL_NEO4J_SERVICE} container of {project} remains" if required else None
    others = _mounts_other_than(volumes, drill.volume)
    if others:
        return (
            f"{project}'s {DRILL_NEO4J_SERVICE} is not on this drill's volume {drill.volume}: "
            + "; ".join(others)
        )
    return None


def _load_drill(args: argparse.Namespace, ctx: Context) -> tuple[Instance, Drill]:
    """The instance and its prepared drill, with every check that needs no Docker."""
    inst = _load_instance(args.root, ctx)
    drill = _read_drill(_drill_root(args.drill_root, inst, ctx))
    _check_not_instance_volume(inst, drill.volume)
    _check_drill_file(inst)
    _check_static_configuration(inst, ctx)
    _require_backup_path(inst)
    return inst, drill


def _check_no_drill_containers(inst: Instance, ctx: Context) -> None:
    """One drill per instance at a time: refuse while the drill project has any container.

    The drill project's Neo4j keeps the volume it was created on, so a new drill started beside
    an earlier drill's containers would restore into the earlier drill's graph.
    """
    project = _drill_project(inst)
    listed = ctx.runner(
        ["docker", "ps", "-a", "--filter", f"label={PROJECT_LABEL}={project}", "-q"]
    )
    if listed.returncode != 0:
        raise RefusedError(
            f"cannot list the containers of {project}, so an earlier drill's may remain: "
            f"{_stderr_tail(listed)}; once Docker answers, run drill-end for any earlier drill"
        )
    leftover = listed.stdout.decode("utf-8", "replace").split()
    if leftover:
        raise RefusedError(
            f"containers {', '.join(leftover)} of {project} remain from an earlier drill; one "
            "drill per instance runs at a time, so run drill-end for that drill first"
        )


def cmd_drill_start(args: argparse.Namespace, ctx: Context) -> None:
    """Create a fresh labelled volume and an empty standby for one restore drill.

    The restore handshake, `standby/state/MIST_RESTORE_TARGET`, is left for the operator to
    create by hand: nothing in MIST.AI creates it.
    """
    _check_plain_name(args.artifact, "--artifact")
    inst = _load_instance(args.root, ctx)
    root = _drill_root(args.drill_root, inst, ctx)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise RefusedError(f"drill root {root} is not an empty directory")
    _check_drill_file(inst)
    _check_static_configuration(inst, ctx)
    _require_backup_path(inst)
    manifest = _absolute_path(_backup_host_path(inst), ctx.home) / args.artifact / "manifest.json"
    if not manifest.is_file():
        raise RefusedError(f"{manifest} does not exist; name an artifact in the backup root")
    _check_release_image(inst, ctx)
    _check_no_drill_containers(inst, ctx)

    started = ctx.clock()
    volume = f"{_drill_project(inst)}-{started.astimezone(UTC).strftime(STAMP_FORMAT)}"
    _check_not_instance_volume(inst, volume)
    existing = ctx.runner(["docker", "volume", "inspect", "--format", "{{json .Labels}}", volume])
    if existing.returncode == 0:
        raise RefusedError(f"volume {volume} already exists; run drill-start again")
    created = ctx.runner(
        [
            "docker",
            "volume",
            "create",
            "--label",
            f"{DRILL_ROLE_LABEL}={DRILL_ROLE}",
            "--label",
            f"{DRILL_INSTANCE_LABEL}={inst.name}",
            volume,
        ]
    )
    if created.returncode != 0:
        raise RefusedError(f"cannot create volume {volume}: {_stderr_tail(created)}")
    record = {"volume": volume, "artifact": args.artifact, "started_at": _format_utc(started)}
    try:
        (root / DRILL_STANDBY / "state").mkdir(parents=True, exist_ok=True)
        (root / DRILL_PRE_RESTORE).mkdir(exist_ok=True)
        _write_atomic(root / DRILL_RECORD, (json.dumps(record) + "\n").encode("utf-8"))
    except OSError as exc:
        raise FailedError(
            f"created volume {volume}, but preparing {root} failed: {exc}; empty {root} and run "
            "drill-start again"
        ) from exc
    standby = root / DRILL_STANDBY / "state"
    print(f"drill ready: volume {volume}, artifact {args.artifact}, standby {standby}")
    print(f"next: create {standby / RESTORE_MARKER_FILENAME} by hand, then run drill-restore")


def cmd_drill_restore(args: argparse.Namespace, ctx: Context) -> int:
    """Restore the drill's artifact into the drill graph and the standby directories.

    Without `--confirm-target`, restore's own refusal prints the token to type; every gate of
    `scripts.backup.restore` applies unchanged.
    """
    inst, drill = _load_drill(args, ctx)
    matches = [entry for entry in _drill_records(inst, drill) if entry[1].get("result") == "match"]
    if matches:
        verified_at = _format_utc(max(matches, key=lambda entry: entry[0])[0])
        raise RefusedError(
            f"drill {drill.root} verified a match at {verified_at}; restoring again would change "
            "the graph that verified, so promote it, or run drill-end and start a new drill"
        )
    _check_release_image(inst, ctx)
    _check_drill_volume(inst, ctx, drill.volume)
    problem = _drill_neo4j_problem(inst, ctx, drill, required=False)
    if problem is not None:
        raise RefusedError(
            f"{problem}; restoring now would write another drill's graph. Run drill-end, which "
            f"removes every container of {_drill_project(inst)} and keeps every volume, then "
            "drill-restore again"
        )
    tail = [
        "scripts.backup.restore",
        "--artifact",
        f"/backup/{drill.artifact}",
        "--target-root",
        f"/drill/{DRILL_STANDBY}/state",
        "--target-vault-root",
        f"/drill/{DRILL_STANDBY}/mist-memory",
        "--backup-root",
        f"/drill/{DRILL_PRE_RESTORE}",
        "--target-graph-uri",
        DRILL_GRAPH_URI,
    ]
    if args.confirm_target is not None:
        tail += ["--confirm-target", args.confirm_target]
    result = _run_drill(inst, ctx, drill, *_restore_in_drill(*tail))
    _echo(result)
    return result.returncode


def _verify_result(stdout: bytes) -> dict | None:
    """The item 7 JSON object verify printed as its last line, or None."""
    lines = [line for line in stdout.decode("utf-8", "replace").splitlines() if line.strip()]
    if not lines:
        return None
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or payload.get("result") not in ("match", "mismatch"):
        return None
    return payload


def cmd_drill_verify(args: argparse.Namespace, ctx: Context) -> int:
    """Compare the restored drill with its artifact and record the counts in `ops/`."""
    inst, drill = _load_drill(args, ctx)
    _check_release_image(inst, ctx)
    _check_drill_volume(inst, ctx, drill.volume)
    result = _run_drill(
        inst,
        ctx,
        drill,
        *_restore_in_drill(
            "scripts.backup.verify",
            "--artifact",
            f"/backup/{drill.artifact}",
            "--graph-uri",
            DRILL_GRAPH_URI,
            "--state-root",
            f"/drill/{DRILL_STANDBY}/state",
            "--vault-root",
            f"/drill/{DRILL_STANDBY}/mist-memory",
        ),
    )
    _echo(result)
    problem = _drill_neo4j_problem(inst, ctx, drill, required=True)
    if problem is not None:
        raise FailedError(
            f"verify ran, but {problem}, so it did not compare this drill's graph; no drill "
            "record was written. Run drill-end, then drill-restore and drill-verify again"
        )
    verified = _verify_result(result.stdout)
    if verified is None:
        print(
            f"instance.py drill-verify: verify exited {result.returncode} without a result; "
            "no drill record was written",
            file=sys.stderr,
        )
        return result.returncode if result.returncode != EXIT_OK else EXIT_FAILED
    now = ctx.clock()
    record = {
        "v": 1,
        "drill_root": str(drill.root),
        "volume": drill.volume,
        "artifact": drill.artifact,
        "verified_at": _format_utc(now),
        "exit_code": result.returncode,
        **verified,
    }
    path = inst.ops / f"drill-{now.astimezone(UTC).strftime(STAMP_FORMAT)}.json"
    try:
        _write_atomic(path, (json.dumps(record, indent=2) + "\n").encode("utf-8"))
    except OSError as exc:
        raise FailedError(f"verify finished, but {path} could not be written: {exc}") from exc
    print(f"drill {verified['result']}: wrote {path}")
    return result.returncode


def cmd_drill_end(args: argparse.Namespace, ctx: Context) -> None:
    """Remove the drill's containers; the volume and the drill root stay for a person to delete."""
    inst, drill = _load_drill(args, ctx)
    _check_drill_volume(inst, ctx, drill.volume)
    result = _run_drill(inst, ctx, drill, "rm", "-s", "-f")
    if result.returncode != 0:
        raise FailedError(
            f"removing the drill containers failed, so some may remain: {_stderr_tail(result)}"
        )
    print(f"removed the drill containers of {_drill_project(inst)}; volume {drill.volume} stays")


def _same_drive(first: Path, second: Path) -> bool:
    """Whether a rename can move `first` beside `second`: one drive and one filesystem."""
    first_drive = os.path.normcase(os.path.splitdrive(str(first))[0])
    second_drive = os.path.normcase(os.path.splitdrive(str(second))[0])
    try:
        return first_drive == second_drive and first.stat().st_dev == second.stat().st_dev
    except OSError:
        return False


def _drill_records(inst: Instance, drill: Drill) -> list[tuple[datetime, dict]]:
    """This drill's `ops/` verify records newer than its `drill.json`, with their times.

    A record counts only when its drill root, volume and artifact are this drill's, so a record
    another drill wrote, or one from before `drill-start` reused the root, never counts.
    """
    records = []
    for path in sorted(inst.ops.glob("drill-*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(record, dict):
            continue
        verified_at = _parse_utc(record.get("verified_at"))
        same_drill = (
            isinstance(record.get("drill_root"), str)
            and _normalized(record["drill_root"]) == _normalized(str(drill.root))
            and record.get("volume") == drill.volume
            and record.get("artifact") == drill.artifact
        )
        if same_drill and verified_at is not None and verified_at > drill.started_at:
            records.append((verified_at, record))
    return records


def _check_verified_match(inst: Instance, drill: Drill) -> None:
    """Refuse unless this drill's newest verify record is a match newer than its `drill.json`."""
    records = _drill_records(inst, drill)
    if not records:
        raise RefusedError(
            f"ops/ holds no drill-verify record for {drill.root} newer than its drill.json; run "
            "drill-verify first"
        )
    _newest, record = max(records, key=lambda entry: entry[0])
    if record.get("result") != "match" or record.get("exit_code") != EXIT_OK:
        raise RefusedError(
            f"the newest drill-verify record for {drill.root} is not a match; promote only a "
            "drill that verified"
        )


def cmd_promote(args: argparse.Namespace, ctx: Context) -> None:
    """Make a verified drill the instance: its volume the graph, its standby the state and vault.

    The order is item 8's. The drill's containers are removed and the volume proven unmounted
    before the instance stops, so a refusal there leaves the instance as it was. The instance's
    containers are then stopped and removed, without their volumes, before anything moves, so
    `up` creates a Neo4j container on the promoted volume; after `up`, its /data mount is read
    back to prove it. The old directories are renamed into `retired/<stamp>/`, never deleted, so
    the move is reversible.
    """
    inst, drill = _load_drill(args, ctx)
    _check_verified_match(inst, drill)
    sources = [drill.standby(name) for name in MARKED_SUBDIRS]
    currents = [inst.root / name for name in MARKED_SUBDIRS]
    missing = [str(path) for path in (*sources, *currents) if not path.is_dir()]
    if missing:
        raise RefusedError(f"promote moves directories that do not exist: {', '.join(missing)}")
    if not _same_drive(drill.root, inst.root):
        raise RefusedError(
            f"drill root {drill.root} is on another drive or filesystem than {inst.root}; "
            "promote moves by rename, so start the drill on the instance's drive"
        )
    try:
        env_bytes = inst.env_path.read_bytes()
        marker_bytes = (inst.root / LIVE_INSTANCE_MARKER).read_bytes()
    except OSError as exc:
        raise RefusedError(f"cannot read the instance's daily.env or marker: {exc}") from exc
    volume_lines = len(VOLUME_LINE.findall(env_bytes))
    if volume_lines != 1:
        raise RefusedError(
            f"{inst.env_path} has {volume_lines} MIST_DAILY_NEO4J_VOLUME= lines; promote "
            "rewrites one"
        )
    stamp = ctx.clock().astimezone(UTC).strftime(STAMP_FORMAT)
    retired = inst.root / "retired" / stamp
    if retired.exists():
        raise RefusedError(f"{retired} already exists; run promote again")
    _check_release_image(inst, ctx)
    _check_drill_volume(inst, ctx, drill.volume)

    removed = _run_drill(inst, ctx, drill, "rm", "-s", "-f")
    if removed.returncode != 0:
        raise FailedError(
            "removing the drill containers failed, so some may remain; the instance was not "
            f"stopped and nothing moved: {_stderr_tail(removed)}"
        )
    mounted = ctx.runner(["docker", "ps", "-a", "--filter", f"volume={drill.volume}", "-q"])
    holders = mounted.stdout.decode("utf-8", "replace").split()
    if mounted.returncode != 0 or holders:
        reason = (
            f"containers {', '.join(holders)} still mount it"
            if holders
            else f"the containers that mount it cannot be listed ({_stderr_tail(mounted)})"
        )
        raise RefusedError(
            f"volume {drill.volume}: {reason}; the instance was not stopped and nothing moved"
        )
    stopped = ctx.runner([*inst.compose(), "--profile", BACKUP_PROFILE, "stop"])
    if stopped.returncode != 0:
        raise FailedError(
            f"stopping project {inst.project} failed, so it may be partly stopped; nothing "
            f"moved, and up --root {inst.root} restarts it: {_stderr_tail(stopped)}"
        )
    # A stopped container keeps its volume: `up` would restart the old graph unless it is gone.
    cleared = ctx.runner([*inst.compose(), "--profile", BACKUP_PROFILE, "rm", "-f"])
    if cleared.returncode != 0:
        raise FailedError(
            f"project {inst.project} is stopped, but removing its containers failed, so some "
            f"may remain; nothing moved, and up --root {inst.root} restarts it: "
            f"{_stderr_tail(cleared)}"
        )

    done: list[str] = []

    def failed(step: str, exc: OSError) -> FailedError:
        so_far = "; ".join(done) if done else "nothing moved"
        return FailedError(
            f"the instance is stopped and {step} failed ({exc}). Done so far: {so_far}. Put "
            f"back any directory {retired} holds before running up --root {inst.root}"
        )

    try:
        retired.mkdir(parents=True)
    except OSError as exc:
        raise failed(f"creating {retired}", exc) from exc
    for current in currents:
        try:
            os.rename(current, retired / current.name)
        except OSError as exc:
            raise failed(f"moving {current} into {retired}", exc) from exc
        done.append(f"{current.name} moved to {retired}")
    for source, current in zip(sources, currents, strict=True):
        try:
            os.rename(source, current)
        except OSError as exc:
            raise failed(f"moving {source} to {current}", exc) from exc
        done.append(f"the drill's {current.name} is now {current}")
    # A promoted handshake would make the live state a valid restore target.
    handshake = inst.root / "state" / RESTORE_MARKER_FILENAME
    try:
        os.unlink(handshake)
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise failed(f"deleting {handshake}", exc) from exc
    done.append(f"deleted {handshake}")
    for current in currents:
        try:
            _write_atomic(current / LIVE_INSTANCE_MARKER, marker_bytes)
        except OSError as exc:
            raise failed(f"marking {current}", exc) from exc
    done.append("state and mist-memory marked")
    new_env = VOLUME_LINE.sub(f"MIST_DAILY_NEO4J_VOLUME={drill.volume}".encode(), env_bytes)
    try:
        _write_atomic(inst.env_path, new_env)
    except OSError as exc:
        raise failed(f"rewriting {inst.env_path}", exc) from exc
    done.append(f"{inst.env_path} names volume {drill.volume}")
    print(f"promoted: {'; '.join(done)}")

    promoted = _load_instance(inst.root, ctx)
    try:
        instance_up(promoted, ctx, DEFAULT_UP_WAIT_MINUTES)
    except InstanceError as exc:
        raise FailedError(
            f"the drill is promoted ({'; '.join(done)}), but up failed: {exc}. The old state "
            f"and vault are in {retired}; daily.env names volume {drill.volume}"
        ) from exc
    try:
        volumes = _data_volumes(ctx, promoted.project, NEO4J_SERVICE)
    except MountReadError as exc:
        problem: str | None = str(exc)
    else:
        others = _mounts_other_than(volumes, drill.volume)
        if not volumes:
            problem = f"no {NEO4J_SERVICE} container of {promoted.project} was found"
        else:
            problem = "; ".join(others) if others else None
    if problem is not None:
        raise FailedError(
            f"the drill is promoted and up ran, but {problem}, where {NEO4J_SERVICE} must mount "
            f"the promoted volume {drill.volume} at {NEO4J_DATA_MOUNT}. daily.env names "
            f"{drill.volume}; the old state and vault are in {retired}"
        )
    print(f"{NEO4J_SERVICE} mounts {drill.volume} at {NEO4J_DATA_MOUNT}")


# --- Gaming mode -----------------------------------------------------------------------------


def _env_seconds(inst: Instance, name: str, default: float, *, allow_zero: bool = False) -> float:
    """`daily.env`'s value for `name` in seconds, or `default` when it is unset or empty.

    Raises:
        RefusedError: When the value is not a finite number above zero (or zero, with
            `allow_zero`).
    """
    raw = inst.env.get(name, "")
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise RefusedError(f"{inst.env_path} sets {name}={raw}, not seconds")
    return value


def _gpu_yield_poll_s(inst: Instance) -> float:
    return _env_seconds(inst, "MIST_GPU_YIELD_POLL_S", DEFAULT_GPU_YIELD_POLL_S)


def _read_gaming_marker(path: Path) -> tuple[bool, dict | None]:
    """Whether the marker exists, and its payload when it parses as item 17's marker.

    A marker that exists but does not parse still counts as present, as the backend reads it.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return False, None
    except (OSError, UnicodeDecodeError):
        return True, None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return True, None
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("since"), str)
        or payload.get("by") not in GAMING_BY
    ):
        return True, None
    return True, payload


def _write_gaming_marker(path: Path, since: str, by: str) -> None:
    payload = {"v": 1, "since": since, "by": by}
    try:
        _write_atomic(path, (json.dumps(payload) + "\n").encode("utf-8"))
    except OSError as exc:
        raise FailedError(f"cannot write {path}: {exc}; {LLM_SERVICE} not stopped") from exc


def gaming_mode_on(root: str | Path, ctx: Context, *, by: str = "manual") -> None:
    """Yield the GPU: mark the yield, let the backend see it, then stop mist-llm and extraction.

    The backend polls the marker every `MIST_GPU_YIELD_POLL_S`; waiting two polls before the
    stop means `/health` never sees a stopped LLM while the backend still reads normal. A
    manual `on` takes over a watcher's yield, so the watcher cannot end a hold Raj asked for;
    a watcher `on` never changes a manual one. Extraction stops on its own stop path, service
    first, for a release that has it.

    Raises:
        InstanceError: When the root cannot be read, or the stop fails twice; then the marker
            this call wrote is removed, so nothing reports yielded while the LLM holds the GPU.
    """
    inst = _load_instance(root, ctx)
    poll_s = _gpu_yield_poll_s(inst)
    marker = inst.ops / GPU_YIELDED_MARKER
    present, payload = _read_gaming_marker(marker)
    wrote = False
    if not present:
        _write_gaming_marker(marker, _format_utc(ctx.clock()), by)
        wrote = True
        ctx.sleep(2 * poll_s)
    elif payload is not None and by == "manual" and payload["by"] == "watcher":
        _write_gaming_marker(marker, payload["since"], "manual")

    stop = [*inst.compose(), "stop", LLM_SERVICE]
    result = ctx.runner(stop)
    if result.returncode != 0:
        result = ctx.runner(stop)
    if result.returncode != 0:
        if wrote:
            try:
                marker.unlink()
                state = f"removed the {marker} this call wrote"
            except OSError as exc:
                state = f"could not remove the {marker} this call wrote ({exc}); remove it by hand"
        else:
            state = f"left {marker}, which was there before"
        raise FailedError(f"stop {LLM_SERVICE} failed twice ({_stderr_tail(result)}); {state}")
    if _extraction_image_id(inst) is not None:
        # The backend has read `gpu-yielded` for two polls (or it was there before), and it
        # reports extraction as yielded ahead of paused, so this stop needs no wait of its own.
        try:
            _stop_extraction(inst, ctx, "gaming", None, backend_informed=True)
        except InstanceError as exc:
            raise FailedError(
                f"{LLM_SERVICE} is stopped and {marker} stays, but extraction did not stop: "
                f"{exc}; the next extraction apply stops it"
            ) from exc
    print(f"gaming mode on: {LLM_SERVICE} stopped and the backend has released the GPU")


def _llm_healthy(inst: Instance, ctx: Context) -> bool:
    result = ctx.runner([*inst.compose(), "ps", "--format", "json", LLM_SERVICE])
    if result.returncode != 0:
        return False
    try:
        rows = _parse_ps_json(result.stdout)
    except ValueError:
        return False
    return any(row.get("Service") == LLM_SERVICE and row.get("Health") == "healthy" for row in rows)


def gaming_mode_off(
    root: str | Path,
    ctx: Context,
    *,
    wait_minutes: int = DEFAULT_OFF_WAIT_MINUTES,
    should_abort: Callable[[], bool] | None = None,
) -> Literal["done", "aborted"]:
    """End the yield: start mist-llm, wait until compose calls it healthy, drop the marker, apply.

    While the LLM loads, turns still get the yielded reply rather than a failure, and a timeout
    leaves the marker, so `/health` keeps reading yielded, which is still true. `should_abort`,
    called on each poll, lets the watcher stop mist-llm again when a game starts meanwhile; the
    extraction apply that follows calls it too, and stops extraction for gaming when it fires.

    Raises:
        InstanceError: When mist-llm cannot be started, does not become healthy within
            `wait_minutes`, or an abort's stop fails; the marker stays in each case. Also when
            the extraction apply after the marker's removal does not converge.
    """
    inst = _load_instance(root, ctx)
    marker = inst.ops / GPU_YIELDED_MARKER
    started = ctx.runner([*inst.compose(), "up", "-d", LLM_SERVICE])
    if started.returncode != 0:
        raise FailedError(
            f"up -d {LLM_SERVICE} failed ({_stderr_tail(started)}); the marker stays, so "
            "/health keeps reading yielded"
        )
    deadline = ctx.monotonic() + wait_minutes * 60
    while True:
        if should_abort is not None and should_abort():
            stopped = ctx.runner([*inst.compose(), "stop", LLM_SERVICE])
            if stopped.returncode != 0:
                raise FailedError(
                    f"aborted, but stop {LLM_SERVICE} failed ({_stderr_tail(stopped)}); the "
                    "marker stays"
                )
            print(f"gaming mode stays on: aborted, and {LLM_SERVICE} is stopped again")
            return "aborted"
        if _llm_healthy(inst, ctx):
            break
        if ctx.monotonic() >= deadline:
            raise FailedError(
                f"{LLM_SERVICE} did not report healthy within {wait_minutes} minute(s); the "
                "marker stays, so /health keeps reading yielded"
            )
        ctx.sleep(LLM_POLL_S)
    try:
        marker.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise FailedError(f"{LLM_SERVICE} is healthy, but {marker} stays: {exc}") from exc
    print(f"gaming mode off: {LLM_SERVICE} is healthy and the marker is removed")
    if _extraction_image_id(inst) is not None:
        try:
            _apply_extraction(inst, ctx, should_abort=should_abort)
        except _ListingFailedError:
            raise
        except InstanceError as exc:
            raise FailedError(
                f"{LLM_SERVICE} is healthy and {marker} is removed, but extraction apply did not "
                f"converge: {exc}"
            ) from exc
    return "done"


def gaming_mode_line(root: Path) -> str:
    """The line `gaming-mode status` prints, read from the marker alone."""
    present, payload = _read_gaming_marker(root / "ops" / GPU_YIELDED_MARKER)
    if not present:
        return "gaming-mode: normal"
    if payload is None:
        return "gaming-mode: yielded since unknown by unknown"
    return f"gaming-mode: yielded since {payload['since']} by {payload['by']}"


def cmd_gaming_mode(args: argparse.Namespace, ctx: Context) -> None:
    """`gaming-mode on|off|status`: yield the GPU to a game, end the yield, or show it."""
    if args.action == "on":
        gaming_mode_on(args.root, ctx, by=args.by)
    elif args.action == "off":
        gaming_mode_off(args.root, ctx, wait_minutes=args.wait_minutes)
    else:
        root = _absolute_path(args.root, ctx.home)
        _instance_name(root)
        print(gaming_mode_line(root))


# --- Night extraction ------------------------------------------------------------------------
#
# `extraction apply` is a level-triggered supervisor: each call reads what the instance's files
# ask for now (the mode, the window, `gpu-yielded`, the two identity variables) and takes the
# two extraction services there, whatever state an earlier call or a crash left. The night
# task calls it every 5 minutes, so any order of calls converges without a lock. The paused
# marker that explains a stopped service is written before the stop and stays until the
# service is healthy again; the backend reads it to report `paused` rather than `down`.


def _local_now(ctx: Context) -> datetime:
    """Now on the local wall clock: `Context.local_zone`'s, or the host's when that is None."""
    return ctx.clock().astimezone(ctx.local_zone)


def _wall_to_utc(wall: datetime, ctx: Context) -> datetime:
    """A naive local wall-clock time as UTC, under the zone's own daylight-saving rules.

    With no zone injected, `astimezone` on a naive datetime reads it as the host's local time,
    so the conversion uses the host's rules for that date rather than today's offset.
    """
    if ctx.local_zone is None:
        return wall.astimezone(UTC)
    return wall.replace(tzinfo=ctx.local_zone).astimezone(UTC)


@dataclass(frozen=True, slots=True)
class ExtractionWindow:
    """`MIST_EXTRACTION_WINDOW`: the local wall-clock span, from `start` up to `end`."""

    start: wall_time
    end: wall_time

    def text(self) -> str:
        """The window as `daily.env` spells it."""
        return f"{self.start:%H:%M}-{self.end:%H:%M}"

    def contains(self, local: datetime) -> bool:
        """Whether the wall-clock time of `local` lies in the window; one that ends before it
        starts runs across midnight.
        """
        wall = local.time()
        if self.start < self.end:
            return self.start <= wall < self.end
        return wall >= self.start or wall < self.end

    def next_start(self, local: datetime, ctx: Context) -> datetime:
        """The window's next start after `local`, in UTC."""
        wall = local.replace(tzinfo=None)
        start = datetime.combine(wall.date(), self.start)
        if start <= wall:
            start += timedelta(days=1)
        return _wall_to_utc(start, ctx)


def _extraction_window(env: Mapping[str, str], env_path: Path) -> ExtractionWindow:
    """`daily.env`'s window, or the default when it is unset or empty.

    Raises:
        RefusedError: When it is not `HH:MM-HH:MM`, or both ends are equal, which could mean
            either always or never.
    """
    raw = env.get(EXTRACTION_WINDOW_VARIABLE, "") or DEFAULT_EXTRACTION_WINDOW
    match = EXTRACTION_WINDOW_PATTERN.fullmatch(raw)
    if match is None:
        raise RefusedError(
            f"{env_path} sets {EXTRACTION_WINDOW_VARIABLE}={raw}, not HH:MM-HH:MM in local time"
        )
    hours_a, minutes_a, hours_b, minutes_b = (int(part) for part in match.groups())
    window = ExtractionWindow(wall_time(hours_a, minutes_a), wall_time(hours_b, minutes_b))
    if window.start == window.end:
        raise RefusedError(
            f"{env_path} sets {EXTRACTION_WINDOW_VARIABLE}={raw}, whose ends are equal; use "
            "`extraction on` or `extraction off` for always or never"
        )
    return window


def _extraction_mode(ops: Path) -> str:
    """`ops/extraction-mode`, or `auto` when there is none (a root older than the file).

    Raises:
        RefusedError: When the file cannot be read or holds anything but on, off or auto.
    """
    path = ops / EXTRACTION_MODE_FILE
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return DEFAULT_EXTRACTION_MODE
    except (OSError, UnicodeDecodeError) as exc:
        raise RefusedError(f"cannot read {path}: {exc}") from exc
    mode = text.strip()
    if mode not in EXTRACTION_MODES:
        raise RefusedError(
            f"{path} holds {mode!r}, not on, off or auto; set it with extraction on|off|auto"
        )
    return mode


def _missing_extraction_identity(env: Mapping[str, str]) -> tuple[str, ...]:
    """The identity variables that hold no value; an example's `<placeholder>` is none."""
    missing = []
    for name in EXTRACTION_IDENTITY_VARIABLES:
        value = env.get(name, "").strip()
        if not value or (value.startswith("<") and value.endswith(">")):
            missing.append(name)
    return tuple(missing)


@dataclass(frozen=True, slots=True)
class ExtractionWant:
    """What the instance's files ask of extraction now.

    `reason` is why it runs (`mode` or `window`) when `wanted`, and otherwise the paused
    marker's reason; `until` is the marker's expiry for a window pause, else None. `missing`
    names the identity variables `daily.env` leaves unset, never their values.
    """

    wanted: bool
    reason: str
    mode: str
    window: ExtractionWindow
    inside: bool
    missing: tuple[str, ...]
    until: datetime | None = None

    @property
    def configured(self) -> bool:
        """Whether every identity variable holds a value."""
        return not self.missing


def _extraction_want(inst: Instance, ctx: Context) -> ExtractionWant:
    """Read the whole desired state from disk again: `daily.env`, the mode and the markers.

    A game outranks the mode, the mode outranks the window, and the window outranks missing
    identity, so the reason recorded is the one an operator would act on first.
    """
    env = _read_env(inst.env_path)
    window = _extraction_window(env, inst.env_path)
    mode = _extraction_mode(inst.ops)
    local = _local_now(ctx)
    inside = window.contains(local)
    missing = _missing_extraction_identity(env)
    until = None
    if (inst.ops / GPU_YIELDED_MARKER).exists():
        wanted, reason = False, "gaming"
    elif mode == "off":
        wanted, reason = False, "mode"
    elif mode == "auto" and not inside:
        wanted, reason = False, "window"
        until = window.next_start(local, ctx) + EXTRACTION_WINDOW_GRACE
    elif missing:
        wanted, reason = False, "unconfigured"
    else:
        wanted, reason = True, "mode" if mode == "on" else "window"
    return ExtractionWant(wanted, reason, mode, window, inside, missing, until)


def _extraction_image_id(inst: Instance) -> str | None:
    """The extraction image ID the release recorded, or None for a release that predates it."""
    record_path = inst.ops / f"release-{inst.tag}.json"
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = record.get("extraction_image_id") if isinstance(record, dict) else None
    return value if isinstance(value, str) and value else None


def _require_extraction_release(inst: Instance) -> str:
    image_id = _extraction_image_id(inst)
    if image_id is None:
        raise RefusedError(
            f"release {inst.tag} has no night extraction: ops/release-{inst.tag}.json records "
            "no extraction_image_id; switch to a release built with it"
        )
    return image_id


def _local_extraction_image(inst: Instance, ctx: Context) -> str | None:
    return _built_image_id(ctx, f"{EXTRACTION_IMAGE_REPOSITORY}:{inst.tag}")


@dataclass(frozen=True, slots=True)
class ExtractionContainer:
    """What `docker inspect` says about one extraction container."""

    cid: str
    service: str
    image_id: str
    status: str
    health: str
    labels: Mapping[str, str]

    @property
    def running(self) -> bool:
        """Whether Docker reports the container running."""
        return self.status == "running"

    @property
    def healthy(self) -> bool:
        """Whether it runs and its healthcheck passes."""
        return self.running and self.health == "healthy"


def _extraction_container_ids(inst: Instance, ctx: Context, service: str) -> list[str]:
    """Every container of `service` in the project, found by label, whatever file made it."""
    listed = ctx.runner(
        [
            "docker",
            "ps",
            "--all",
            "--quiet",
            "--no-trunc",
            "--filter",
            f"label={PROJECT_LABEL}={inst.project}",
            "--filter",
            f"label={SERVICE_LABEL}={service}",
        ]
    )
    if listed.returncode != 0:
        raise FailedError(
            f"cannot list project {inst.project}'s {service} containers: {_stderr_tail(listed)}"
        )
    return listed.stdout.decode("utf-8", "replace").split()


def _extraction_containers(inst: Instance, ctx: Context) -> list[ExtractionContainer]:
    found = []
    for service in EXTRACTION_SERVICES:
        for cid in _extraction_container_ids(inst, ctx, service):
            result = ctx.runner(["docker", "inspect", "--format", EXTRACTION_INSPECT_FORMAT, cid])
            if result.returncode != 0:
                raise FailedError(f"cannot inspect container {cid}: {_stderr_tail(result)}")
            parts = result.stdout.decode("utf-8", "replace").strip().split(" ", 4)
            try:
                labels = json.loads(parts[4]) if len(parts) == 5 else None
            except json.JSONDecodeError:
                labels = None
            if len(parts) != 5 or not (labels is None or isinstance(labels, dict)):
                raise FailedError(f"cannot read docker inspect's output for container {cid}")
            found.append(
                ExtractionContainer(cid, service, parts[1], parts[2], parts[3], labels or {})
            )
    return found


def _extraction_compose(inst: Instance) -> list[str]:
    """The compose prefix with the extraction profile, without which compose cannot resolve
    either service.
    """
    return [*inst.compose(), "--profile", EXTRACTION_PROFILE]


def _config_hash(inst: Instance, ctx: Context, service: str) -> str:
    """The config hash the release's compose file gives `service` with today's `daily.env`."""
    result = ctx.runner([*_extraction_compose(inst), "config", "--hash", service])
    if result.returncode == 0:
        for line in result.stdout.decode("utf-8", "replace").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] == service:
                return parts[1]
    raise FailedError(
        f"cannot read {service}'s config hash from {inst.compose_file()}: {_stderr_tail(result)}"
    )


def _is_current(
    inst: Instance, container: ExtractionContainer, config_hash: str, image_id: str
) -> bool:
    """Whether the container is what the release's compose file would make now.

    A `daily.env` change to a serving setting changes the config hash, and a container from
    another release's file or image is another release's service, so any of those is stale.
    """
    label = str(container.labels.get(CONFIG_FILES_LABEL, ""))
    files = [entry for entry in label.split(",") if entry]
    if len(files) != 1 or _normalized(files[0]) != _normalized(str(inst.compose_file())):
        return False
    if container.labels.get(CONFIG_HASH_LABEL) != config_hash:
        return False
    return container.service != EXTRACTION_SERVICE or container.image_id == image_id


def _read_paused_marker(path: Path) -> tuple[bool, dict | None]:
    """Whether the marker exists, and its payload when it parses as the contract's marker."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return False, None
    except (OSError, UnicodeDecodeError):
        return True, None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return True, None
    if (
        not isinstance(payload, dict)
        or payload.get("v") != 1
        or _parse_utc(payload.get("since")) is None
        or payload.get("reason") not in PAUSED_REASONS
        or not (payload.get("until") is None or _parse_utc(payload.get("until")) is not None)
    ):
        return True, None
    return True, payload


def _paused_marker_active(payload: dict | None, now: datetime) -> bool:
    """Whether the backend reads this marker as paused: it parses and has not expired."""
    if payload is None:
        return False
    until = _parse_utc(payload.get("until"))
    return until is None or until > now


def _set_paused(inst: Instance, ctx: Context, reason: str, until: datetime | None) -> bool:
    """Write the paused marker for `reason`; return whether the backend already read paused.

    `since` carries over from a marker that parses, since extraction has been stopped on
    purpose since then; an unchanged marker is not rewritten.
    """
    path = inst.ops / EXTRACTION_PAUSED_MARKER
    _present, payload = _read_paused_marker(path)
    now = ctx.clock()
    active = _paused_marker_active(payload, now)
    record = {
        "v": 1,
        "since": payload["since"] if payload is not None else _format_utc(now),
        "reason": reason,
        "until": None if until is None else _format_utc(until),
    }
    if record != payload:
        try:
            _write_atomic(path, (json.dumps(record) + "\n").encode("utf-8"))
        except OSError as exc:
            raise FailedError(f"cannot write {path}: {exc}; extraction left as it was") from exc
    return active


def _clear_paused(inst: Instance) -> None:
    path = inst.ops / EXTRACTION_PAUSED_MARKER
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise FailedError(f"extraction is healthy, but {path} stays: {exc}") from exc


def _log_transition(inst: Instance, ctx: Context, action: str, reason: str) -> None:
    """Append `<UTC> start|stop <reason>` to the window log; a failed write only warns."""
    path = inst.root / "logs" / EXTRACTION_WINDOW_LOG
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{_format_utc(ctx.clock())} {action} {reason}\n")
    except OSError as exc:
        print(f"instance.py: cannot append to {path}: {exc}", file=sys.stderr)


def _save_extraction_log(inst: Instance, ctx: Context, service: str) -> None:
    """Replace `logs/extraction-last-<service>.log` with the container's last lines.

    Removal deletes the container's own log, so this copy is what a failed start leaves for
    `/health`'s remedy to read.
    """
    result = ctx.runner(
        [
            *_extraction_compose(inst),
            "logs",
            "--no-color",
            "--tail",
            str(EXTRACTION_LOG_LINES),
            service,
        ]
    )
    if result.returncode == 0:
        data = result.stdout
    else:
        data = f"docker compose logs {service} failed: {_stderr_tail(result)}\n".encode()
    _write_extraction_log(inst, service, data)


def _write_extraction_log(inst: Instance, service: str, data: bytes) -> None:
    path = inst.root / "logs" / f"extraction-last-{service}.log"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_atomic(path, data)
    except OSError as exc:
        print(f"instance.py: cannot write {path}: {exc}", file=sys.stderr)


def _remove_extraction(inst: Instance, ctx: Context, present: set[str]) -> None:
    """Stop the service before its LLM, keep each one's last log lines, then remove both.

    The service stops first so the backend's in-flight call ends as a transport error, which
    costs no attempt; with the LLM gone first the service would answer an error envelope,
    which does. Removal names both services, since a bare `rm` would also remove a stopped
    mist-llm or mist-backend.

    Raises:
        FailedError: When a stop or the removal fails; the caller's marker stays.
    """
    dc = _extraction_compose(inst)
    stops = (
        (EXTRACTION_SERVICE, ["--timeout", str(EXTRACTION_STOP_TIMEOUT_S)]),
        (EXTRACTION_LLM_SERVICE, []),
    )
    for service, options in stops:
        if service not in present:
            continue
        stopped = ctx.runner([*dc, "stop", *options, service])
        if stopped.returncode != 0:
            raise FailedError(f"stop {service} failed ({_stderr_tail(stopped)})")
    for service, _options in stops:
        if service in present:
            _save_extraction_log(inst, ctx, service)
    removed = ctx.runner([*dc, "rm", "-f", EXTRACTION_SERVICE, EXTRACTION_LLM_SERVICE])
    if removed.returncode != 0:
        raise FailedError(
            f"removing {EXTRACTION_SERVICE} and {EXTRACTION_LLM_SERVICE} failed "
            f"({_stderr_tail(removed)})"
        )


def _stop_extraction(
    inst: Instance,
    ctx: Context,
    reason: str,
    until: datetime | None,
    containers: Sequence[ExtractionContainer] | None = None,
    *,
    backend_informed: bool = False,
) -> None:
    """The stop path: the marker first, two backend polls, then stop and remove both.

    The wait is skipped when nothing runs, when the backend already read paused, or when the
    caller's own wait covered it (`backend_informed`, for gaming mode's `gpu-yielded`).
    """
    if containers is None:
        containers = _extraction_containers(inst, ctx)
    active = _set_paused(inst, ctx, reason, until)
    if not containers:
        print(f"extraction: not running, paused for {reason}")
        return
    running = any(container.running for container in containers)
    if running and not active and not backend_informed:
        ctx.sleep(2 * _gpu_yield_poll_s(inst))
    try:
        _remove_extraction(inst, ctx, {container.service for container in containers})
    except FailedError as exc:
        raise FailedError(f"{exc}; {EXTRACTION_PAUSED_MARKER} stays, paused for {reason}") from exc
    if running:
        _log_transition(inst, ctx, "stop", reason)
    print(f"extraction: stopped and removed, paused for {reason}")


def _stop_unwanted(
    inst: Instance,
    ctx: Context,
    want: ExtractionWant,
    containers: Sequence[ExtractionContainer] | None = None,
) -> None:
    """The stop path for `want`; missing identity is then refused, naming only the variables.

    The marker reads `unconfigured` like any other pause; the refusal (plan v4: apply refuses
    on missing identity) is what tells the operator that this pause, unlike the window's,
    ends only when `daily.env` changes.

    Raises:
        RefusedError: When the pause is for missing identity.
        FailedError: From the stop path.
    """
    _stop_extraction(inst, ctx, want.reason, want.until, containers)
    if want.reason == "unconfigured":
        raise RefusedError(
            f"{inst.env_path} leaves {', '.join(want.missing)} unset, so extraction stays "
            "paused as unconfigured; set it, then run extraction apply"
        )


def _extraction_health(inst: Instance, ctx: Context) -> tuple[bool, str | None]:
    """Whether both services report healthy, and why the start failed if one has exited."""
    result = ctx.runner(
        [
            *_extraction_compose(inst),
            "ps",
            "--all",
            "--format",
            "json",
            EXTRACTION_LLM_SERVICE,
            EXTRACTION_SERVICE,
        ]
    )
    if result.returncode != 0:
        return False, None
    try:
        rows = _parse_ps_json(result.stdout)
    except ValueError:
        return False, None
    by_service = {row.get("Service"): row for row in rows}
    for service in EXTRACTION_SERVICES:
        state = by_service.get(service, {}).get("State")
        if state in ("exited", "dead"):
            return False, f"{service} is {state}"
    return (
        all(
            by_service.get(service, {}).get("Health") == "healthy"
            for service in EXTRACTION_SERVICES
        ),
        None,
    )


def _fail_start(inst: Instance, ctx: Context, why: str) -> None:
    """Remove both containers and the marker, so `/health` reads extraction as down.

    Raises:
        FailedError: Always, naming why the start failed and what was left.
    """
    try:
        present = {container.service for container in _extraction_containers(inst, ctx)}
        _remove_extraction(inst, ctx, present)
    except InstanceError as exc:
        raise FailedError(
            f"extraction did not start: {why}; removing its containers then failed ({exc}), so "
            f"{EXTRACTION_PAUSED_MARKER} stays until its start deadline"
        ) from exc
    _clear_paused(inst)
    _log_transition(inst, ctx, "stop", "failed")
    raise FailedError(
        f"extraction did not start: {why}; both containers and {EXTRACTION_PAUSED_MARKER} are "
        "removed, so /health reads it as down"
    )


def _start_extraction(
    inst: Instance,
    ctx: Context,
    want: ExtractionWant,
    containers: Sequence[ExtractionContainer],
    stale: bool,
    should_abort: Callable[[], bool] | None,
) -> None:
    """The start path: marker `starting`, the LLM, the service, then both healthy.

    After each step the whole desired state is read again; when extraction is no longer wanted
    the stop path runs instead. The marker goes only once both are healthy and still wanted.
    """
    deadline = ctx.monotonic() + EXTRACTION_START_DEADLINE_S
    until = ctx.clock() + timedelta(seconds=EXTRACTION_START_DEADLINE_S)
    active = _set_paused(inst, ctx, "starting", until)

    def no_longer_wanted() -> bool:
        now = _extraction_want(inst, ctx)
        if now.wanted and should_abort is not None and should_abort():
            now = replace(now, wanted=False, reason="gaming", until=None)
        if now.wanted:
            return False
        _stop_unwanted(inst, ctx, now)
        return True

    if stale:
        if any(container.running for container in containers) and not active:
            ctx.sleep(2 * _gpu_yield_poll_s(inst))
        try:
            _remove_extraction(inst, ctx, {container.service for container in containers})
        except FailedError as exc:
            raise FailedError(
                f"could not remove the stale extraction containers ({exc}); "
                f"{EXTRACTION_PAUSED_MARKER} stays until its start deadline"
            ) from exc
        if no_longer_wanted():
            return
    _log_transition(inst, ctx, "start", want.reason)
    for service in EXTRACTION_SERVICES:
        started = ctx.runner([*_extraction_compose(inst), "up", "-d", service])
        if started.returncode != 0:
            _fail_start(inst, ctx, f"up -d {service} failed ({_stderr_tail(started)})")
        if no_longer_wanted():
            return
    while True:
        healthy, problem = _extraction_health(inst, ctx)
        if healthy:
            break
        if problem is not None:
            _fail_start(inst, ctx, problem)
        if ctx.monotonic() >= deadline:
            _fail_start(
                inst,
                ctx,
                f"{EXTRACTION_LLM_SERVICE} and {EXTRACTION_SERVICE} did not both report healthy "
                f"within {EXTRACTION_START_DEADLINE_S} s",
            )
        ctx.sleep(EXTRACTION_POLL_S)
        if no_longer_wanted():
            return
    if no_longer_wanted():
        return
    _clear_paused(inst)
    print(f"extraction: running and healthy, started for {want.reason}")


def _apply_extraction(
    inst: Instance, ctx: Context, *, should_abort: Callable[[], bool] | None = None
) -> None:
    """Converge the two extraction services on what the instance's files ask for now.

    `should_abort`, from the gaming watcher, counts as `gpu-yielded` when it returns true.
    A start waits for the project's own mist-backend: while it is not running, apply starts
    nothing and writes no marker, but the stop path still runs whenever extraction is unwanted.

    Raises:
        RefusedError: Nothing changed: the release predates night extraction, a setting is
            malformed, or extraction is wanted but `mist-extraction:<tag>` is not the image
            the release recorded. The one exception is a missing identity variable, which the
            message names: that refusal comes after the stop path has run and written the
            `unconfigured` marker, so something did change.
        FailedError: A stop or start did not converge, or the backend's state could not be
            read; the message names what was left.
    """
    expected_image = _require_extraction_release(inst)
    want = _extraction_want(inst, ctx)
    if want.wanted and should_abort is not None and should_abort():
        want = replace(want, wanted=False, reason="gaming", until=None)
    containers = _extraction_containers(inst, ctx)
    if not want.wanted:
        _stop_unwanted(inst, ctx, want, containers)
        return
    image = f"{EXTRACTION_IMAGE_REPOSITORY}:{inst.tag}"
    local = _local_extraction_image(inst, ctx)
    fix = f"run release-build {inst.tag} --root {inst.root}"
    if local is None:
        raise RefusedError(f"there is no image {image}; {fix}")
    if local != expected_image:
        raise RefusedError(
            f"{image} is {local}, but ops/release-{inst.tag}.json records {expected_image}: the "
            f"tag was re-pointed by another build; {fix}"
        )
    services = [container.service for container in containers]
    hashes = {s: _config_hash(inst, ctx, s) for s in EXTRACTION_SERVICES if s in services}
    stale = len(set(services)) != len(services) or not all(
        _is_current(inst, container, hashes[container.service], expected_image)
        for container in containers
    )
    if not stale and len(containers) == 2 and all(c.healthy for c in containers):
        _clear_paused(inst)
        print("extraction: running and healthy from the current release; nothing to do")
        return
    try:
        backend = _read_backend_running(inst, ctx)
    except ValueError as exc:
        raise FailedError(f"{exc}; started nothing") from exc
    if not backend:
        # The backend sends extraction its work; started without it, the pair would hold the
        # GPU after `down` with nothing to do. `up` applies again once the backend runs.
        print(
            f"extraction: {BACKEND_SERVICE} of project {inst.project} is not running; "
            "started nothing"
        )
        return
    _start_extraction(inst, ctx, want, containers, stale, should_abort)


def extraction_apply(
    root: str | Path, ctx: Context, *, should_abort: Callable[[], bool] | None = None
) -> None:
    """`extraction apply` in process: load the root, then converge."""
    _apply_extraction(_load_instance(root, ctx), ctx, should_abort=should_abort)


def _remove_extraction_by_label(inst: Instance, ctx: Context) -> list[str]:
    """Stop, then remove, every extraction container of the project, found by label.

    Plain `docker` on container IDs, since the target release's compose file may not define
    the services. The order is the stop path's: the service before its LLM, so an in-flight
    call ends as a transport error rather than an error envelope that costs the turn an
    attempt; then each one's last log lines; then `docker rm -f`.

    Raises:
        FailedError: When listing, a stop or the removal fails; nothing is removed after a
            failed stop.
    """
    containers = _extraction_containers(inst, ctx)
    if not containers:
        return []
    ordered = sorted(containers, key=lambda c: c.service != EXTRACTION_SERVICE)
    for container in ordered:
        if not container.running:
            continue
        short = container.service == EXTRACTION_SERVICE
        timeout = ["-t", str(EXTRACTION_STOP_TIMEOUT_S)] if short else []
        stopped = ctx.runner(["docker", "stop", *timeout, container.cid])
        if stopped.returncode != 0:
            raise FailedError(
                f"docker stop of {container.service} ({container.cid}) failed: "
                f"{_stderr_tail(stopped)}"
            )
    for container in ordered:
        logs = ctx.runner(["docker", "logs", "--tail", str(EXTRACTION_LOG_LINES), container.cid])
        if logs.returncode == 0:
            # `docker logs` keeps the container's two streams apart; both are kept, in turn.
            data = logs.stdout + logs.stderr
        else:
            data = f"docker logs {container.cid} failed: {_stderr_tail(logs)}\n".encode()
        _write_extraction_log(inst, container.service, data)
    ids = [container.cid for container in containers]
    removed = ctx.runner(["docker", "rm", "-f", *ids])
    if removed.returncode != 0:
        raise FailedError(
            f"docker rm -f of the extraction containers failed: {_stderr_tail(removed)}"
        )
    print(f"removed {len(ids)} extraction container(s) before the switch")
    return ids


def extraction_line(inst: Instance, ctx: Context) -> str:
    """The one line `extraction status` and `status` print; it shows no `daily.env` identity."""
    expected = _extraction_image_id(inst)
    if expected is None:
        return f"extraction: none in release {inst.tag}"
    try:
        want = _extraction_want(inst, ctx)
    except RefusedError as exc:
        return f"extraction: settings refused ({exc})"
    try:
        containers = _extraction_containers(inst, ctx)
    except FailedError:
        running = "unknown"
    else:
        up = {container.service for container in containers if container.running}
        running = "yes" if len(up) == 2 else "no" if not up else "partial"
    present, payload = _read_paused_marker(inst.ops / EXTRACTION_PAUSED_MARKER)
    if payload is None:
        paused = "unreadable" if present else "no"
    elif _paused_marker_active(payload, ctx.clock()):
        paused = payload["reason"]
    else:
        paused = f"expired {payload['reason']}"
    local = _local_extraction_image(inst, ctx)
    image = "missing" if local is None else "match" if local == expected else "MISMATCH"
    return (
        f"extraction: mode {want.mode}, window {want.window.text()} "
        f"{'inside' if want.inside else 'outside'}, running {running}, paused {paused}, "
        f"image {image}, configured {'yes' if want.configured else 'no'}"
    )


def cmd_extraction(args: argparse.Namespace, ctx: Context) -> None:
    """`extraction on|off|auto|apply|status`: set the mode and converge, converge, or show it."""
    inst = _load_instance(args.root, ctx)
    _require_extraction_release(inst)
    if args.action == "status":
        print(extraction_line(inst, ctx))
        return
    if args.action in EXTRACTION_MODES:
        path = inst.ops / EXTRACTION_MODE_FILE
        try:
            _write_atomic(path, f"{args.action}\n".encode())
        except OSError as exc:
            raise FailedError(f"cannot write {path}: {exc}; nothing changed") from exc
        print(f"extraction mode: {args.action}")
        try:
            _apply_extraction(inst, ctx)
        except RefusedError as exc:
            raise FailedError(
                f"wrote mode {args.action} to {path}, but apply was refused: {exc}"
            ) from exc
        return
    _apply_extraction(inst, ctx)


# --- gaming-watch ----------------------------------------------------------------------------


@functools.cache
def _load_gamewatch():
    """Exec `gamewatch.py` from beside this file and return it.

    Loaded by path, as `volume_safety.py` is, and only by `gaming-watch`, so every other
    command runs from a `bin/` that lacks it. A successful load is kept, so the lister's
    `GameWatchError` is one class for the whole process.

    Raises:
        InstanceError: Exit 1, naming the file, when it is missing or cannot be imported.
    """
    path = Path(__file__).with_name(GAMEWATCH_FILE)
    spec = importlib.util.spec_from_file_location("gamewatch", path)
    if spec is None or spec.loader is None:
        raise InstanceError(f"cannot load {GAMEWATCH_FILE} from {path.parent}")
    module = importlib.util.module_from_spec(spec)
    # `dataclass` looks its class's module up in `sys.modules` while the module executes.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except (ImportError, OSError) as exc:
        del sys.modules[spec.name]
        raise InstanceError(
            f"cannot load {GAMEWATCH_FILE} from {path.parent} ({exc}); switch installs it in "
            "bin/ beside instance.py"
        ) from exc
    return module


class _ListingFailedError(InstanceError):
    """The watcher's process lister failed; the watch ends rather than acting blind."""


@contextlib.contextmanager
def _watch_log(path: Path):
    """A logger writing to `path`, rotated at 1 MiB with three old files kept.

    Lines carry the context clock's time, written by the caller, so the format is the message
    alone.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        path,
        maxBytes=GAMING_WATCH_LOG_BYTES,
        backupCount=GAMING_WATCH_LOG_BACKUPS,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger(f"mist.gaming-watch.{path}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(handler)
    try:
        yield logger
    finally:
        logger.removeHandler(handler)
        handler.close()


def _marker_by(path: Path) -> str | None:
    """The marker's `by`, None without a marker, and `unknown` for one that does not parse."""
    present, payload = _read_gaming_marker(path)
    if not present:
        return None
    return "unknown" if payload is None else payload["by"]


def _game_watch_setup(root: str | Path, ctx: Context, *, require_release: bool = True):
    """Load `gamewatch.py`, the root and its game rules, with the watcher's refusals.

    Returns the loaded module, the instance and its `GameRules`.

    Raises:
        InstanceError: Exit 1 when `gamewatch.py` cannot be loaded; `RefusedError` (exit 2) on
            a Python that is not 64-bit, or when `MIST_GAME_DIRS` and `MIST_GAME_RUNTIME_DIRS`
            are both unset or empty.
    """
    # First, before `daily.env` is read, so a `bin/` without the file says so whatever the
    # configuration.
    gamewatch = _load_gamewatch()
    if not gamewatch.is_64_bit():
        raise RefusedError(
            "the gaming watcher needs a 64-bit Python: it reads a Java process's working "
            "directory through the x64 process layout"
        )
    inst = _load_instance(root, ctx, require_release=require_release)
    try:
        env_text = inst.env_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise RefusedError(f"cannot read {inst.env_path}: {exc}") from exc
    rules = gamewatch.rules_from_env(env_text)
    if not rules.libraries and not rules.runtime_dirs:
        raise RefusedError(
            f"{inst.env_path} sets neither MIST_GAME_DIRS nor MIST_GAME_RUNTIME_DIRS; set "
            "MIST_GAME_DIRS to the game libraries or MIST_GAME_RUNTIME_DIRS to the game "
            "instance and server folders, each separated by ;"
        )
    return gamewatch, inst, rules


def _game_rule_entries(rules) -> list[tuple[str, str, bool]]:
    """Each configured entry as (`VARIABLE[k]`, entry, whether it is a directory)."""
    entries = [(f"MIST_GAME_DIRS[{k}]", entry, True) for k, entry in enumerate(rules.libraries)]
    entries += [
        (f"MIST_GAME_RUNTIME_DIRS[{k}]", entry, True) for k, entry in enumerate(rules.runtime_dirs)
    ]
    for k in range(max(len(rules.exclude_dirs), len(rules.exclude_names))):
        directory = rules.exclude_dirs[k] if k < len(rules.exclude_dirs) else ""
        name = rules.exclude_names[k] if k < len(rules.exclude_names) else ""
        entries.append((f"MIST_GAME_EXCLUDE[{k}]", directory or name, bool(directory)))
    return entries


def gaming_watch(root: str | Path, ctx: Context, *, max_ticks: int | None = None) -> None:
    """Yield the GPU while a game itself runs, and end the yield after it closes.

    Each tick lists the processes, asks `gamewatch.find_game` whether one is a game's own
    process, reads the marker, asks `gamewatch.decide` what to do, and runs the in-process
    `gaming-mode on --by watcher` or `gaming-mode off`. A failed command is logged and decided
    again on the next tick. The log names a game only by the variable and index of the entry it
    counted under, never by a process name, a path or an argument: a Java client's arguments
    carry its account's access token. `max_ticks` ends the loop for tests; the command itself
    runs until it is stopped.

    Raises:
        InstanceError: Exit 1 when `gamewatch.py` cannot be loaded or the processes cannot be
            listed; `RefusedError` (exit 2) on a Python that is not 64-bit, when neither
            `MIST_GAME_DIRS` nor `MIST_GAME_RUNTIME_DIRS` is set, or when an interval is not
            seconds.
    """
    gamewatch, inst, rules = _game_watch_setup(root, ctx)
    poll_s = _env_seconds(inst, "MIST_GAME_POLL_S", DEFAULT_GAME_POLL_S)
    resume_delay_s = _env_seconds(
        inst, "MIST_GAME_RESUME_DELAY_S", DEFAULT_GAME_RESUME_DELAY_S, allow_zero=True
    )
    lister = ctx.processes if ctx.processes is not None else gamewatch.list_processes
    marker = inst.ops / GPU_YIELDED_MARKER
    entries = _game_rule_entries(rules)
    exclude_count = sum(label.startswith("MIST_GAME_EXCLUDE[") for label, _e, _d in entries)
    missing = [label for label, entry, is_dir in entries if is_dir and not os.path.isdir(entry)]

    def find(log: logging.Logger):
        try:
            records = lister()
        except gamewatch.GameWatchError as exc:
            log.info(f"{_format_utc(ctx.clock())} listing processes failed: {exc}; stopping")
            raise _ListingFailedError(f"cannot list processes: {exc}") from exc
        return gamewatch.find_game(records, rules)

    with _watch_log(inst.root / "logs" / GAMING_WATCH_LOG) as log:

        def note(message: str) -> None:
            log.info(f"{_format_utc(ctx.clock())} {message}")

        note(
            f"watching {len(rules.libraries)} MIST_GAME_DIRS, {len(rules.runtime_dirs)} "
            f"MIST_GAME_RUNTIME_DIRS and {exclude_count} MIST_GAME_EXCLUDE entries every "
            f"{poll_s:g} s, resume delay {resume_delay_s:g} s; missing: "
            f"{', '.join(missing) or 'none'}"
        )
        state = gamewatch.INITIAL_STATE
        seen: str | None = None
        ticks = 0
        while True:
            verdict = find(log)
            label = None if verdict is None else f"{verdict.variable}[{verdict.index}]"
            if label != seen:
                note("no game running" if label is None else f"game seen under {label}")
                seen = label
            previous = state
            action, state = gamewatch.decide(
                previous, verdict is not None, _marker_by(marker), ctx.monotonic(), resume_delay_s
            )
            if state.suppressed and not previous.suppressed:
                note("gaming mode was ended by someone else; not yielding until no game runs")
            if action == "on":
                try:
                    gaming_mode_on(inst.root, ctx, by="watcher")
                    note("gaming-mode on --by watcher: done")
                except InstanceError as exc:
                    note(f"gaming-mode on --by watcher: {exc.label}: {exc}")
            elif action == "off":

                def game_running() -> bool:
                    return find(log) is not None

                try:
                    outcome = gaming_mode_off(inst.root, ctx, should_abort=game_running)
                    note(f"gaming-mode off: {outcome}")
                except _ListingFailedError:
                    raise
                except InstanceError as exc:
                    note(f"gaming-mode off: {exc.label}: {exc}")
            ticks += 1
            if max_ticks is not None and ticks >= max_ticks:
                return
            ctx.sleep(poll_s)


def cmd_gaming_watch(args: argparse.Namespace, ctx: Context) -> None:
    """`gaming-watch`: run the watcher until it is stopped."""
    gaming_watch(args.root, ctx)


# --- gaming-scan -----------------------------------------------------------------------------


def gaming_scan(root: str | Path, ctx: Context) -> list[str]:
    """What counts as a game among the running programs now, and why: the scan's lines.

    It loads `gamewatch.py` and the root as `gaming-watch` does, with its refusals, except
    that the root's release is not required: the scan starts and stops nothing, and the deploy
    runs it before `switch` writes the release tag. It lists once and writes no log.

    The lines are each configured entry, with ` (missing)` for a directory that does not
    exist; `game: yes` or `game: no`; each counting program; each program under a library that
    an exclusion removed; and each other shared runtime that does not count, with rule 1's
    reason. No argument is printed other than a folder taken from `--gameDir`, because a Java
    client's arguments carry its account's access token, and no other `daily.env` line.

    Raises:
        InstanceError: Exit 1 when `gamewatch.py` cannot be loaded or the processes cannot be
            listed; `RefusedError` (exit 2) on the refusals `gaming-watch` makes.
    """
    gamewatch, _inst, rules = _game_watch_setup(root, ctx, require_release=False)
    lister = ctx.processes if ctx.processes is not None else gamewatch.list_processes
    try:
        records = lister()
    except gamewatch.GameWatchError as exc:
        raise InstanceError(f"cannot list processes: {exc}") from exc

    lines = []
    for label, entry, is_dir in _game_rule_entries(rules):
        missing = " (missing)" if is_dir and not os.path.isdir(entry) else ""
        lines.append(f"{label} {entry}{missing}")
    lines.append("game: yes" if gamewatch.find_game(records, rules) is not None else "game: no")
    counted: list[str] = []
    excluded: list[str] = []
    not_counted: list[str] = []
    for record in records:
        verdict = gamewatch.classify(record, rules)
        if verdict.counts:
            folder = "" if verdict.folder is None else f" for {verdict.folder}"
            counted.append(f"counts {verdict.variable}[{verdict.index}]: {record.image}{folder}")
        elif gamewatch.lowest_index(record.image, rules.libraries) is not None:
            excluded.append(f"excluded {verdict.reason}: {record.image}")
        elif gamewatch.is_shared_runtime(record.image):
            not_counted.append(f"not counted {verdict.reason}: {record.image}")
    for group in (counted, excluded, not_counted):
        # Many processes of one program would repeat a line; one says the same.
        lines.extend(dict.fromkeys(group))
    return lines


def cmd_gaming_scan(args: argparse.Namespace, ctx: Context) -> None:
    """`gaming-scan`: print which running programs count as a game, and why."""
    for line in gaming_scan(args.root, ctx):
        print(line)


# --- status ----------------------------------------------------------------------------------


def _backend_line(inst: Instance, ctx: Context) -> str:
    """The running backend's tag, image ID against the release record, and revision label."""
    result = ctx.runner([*inst.compose(), "ps", "--format", "json"])
    if result.returncode != 0:
        return f"backend: unknown ({_stderr_tail(result)})"
    try:
        rows = _parse_ps_json(result.stdout)
    except ValueError:
        return "backend: unknown (unreadable docker compose ps output)"
    backend = next(
        (
            row
            for row in rows
            if row.get("Service") == BACKEND_SERVICE and row.get("State") == "running"
        ),
        None,
    )
    if backend is None:
        return "backend: not running"
    image_ref = str(backend.get("Image", ""))
    tag = image_ref.rpartition(":")[2] if ":" in image_ref else "unknown"
    try:
        info = _inspect_container(ctx, str(backend.get("ID") or backend.get("Name")))
    except FailedError as exc:
        return f"backend: tag {tag}, image unknown ({exc})"
    try:
        recorded = _recorded_image_id(replace(inst, tag=tag)) if _tag_is_valid(tag) else None
    except RefusedError:
        recorded = None
    verdict = "match" if recorded == info.image_id else "MISMATCH"
    revision = "unknown"
    labels_result = ctx.runner(
        ["docker", "image", "inspect", "--format", "{{json .Config.Labels}}", info.image_id]
    )
    if labels_result.returncode == 0:
        with contextlib.suppress(ValueError):
            labels = json.loads(labels_result.stdout)
            if isinstance(labels, dict) and isinstance(labels.get(REVISION_LABEL), str):
                revision = labels[REVISION_LABEL]
    return f"backend: tag {tag}, image {info.image_id} {verdict}, revision {revision}"


def _get_json(ctx: Context, url: str) -> object | None:
    try:
        return json.loads(ctx.fetch(url, HTTP_TIMEOUT_S))
    except (OSError, ValueError):
        return None


def _health_lines(inst: Instance, ctx: Context, port: str) -> list[str]:
    payload = _get_json(ctx, f"http://127.0.0.1:{port}/health")
    if not isinstance(payload, dict):
        return ["health: unreachable"]
    lines = [f"health: {payload.get('status')}"]
    checks = payload.get("checks")
    if isinstance(checks, dict):
        for name, check in checks.items():
            if isinstance(check, dict):
                status = check.get("status")
                lines.append(
                    f"  {name}: {'null' if status is None else status} {check.get('reason')}"
                )
    return lines


def _usage_lines(inst: Instance, ctx: Context, port: str) -> list[str]:
    payload = _get_json(ctx, f"http://127.0.0.1:{port}/usage/turns-per-day?days={USAGE_DAYS}")
    if not isinstance(payload, dict):
        return ["turns-per-day: unreachable"]
    days = payload.get("days")
    if not isinstance(days, list):
        return ["turns-per-day: unreadable"]
    entries = [day for day in days if isinstance(day, dict)]
    counts = ", ".join(f"{day.get('date')} {day.get('real_turns')}" for day in entries)
    # Derived here from the days returned, not read from the endpoint: the counts are reported,
    # and nothing waits on how many days had use (ADR-042).
    used = sum(
        1
        for day in entries
        if isinstance(day.get("real_turns"), int)
        and not isinstance(day.get("real_turns"), bool)
        and day["real_turns"] > 0
    )
    return [f"turns-per-day: {counts}", f"days used: {used} of {len(entries)}"]


def _device_forward_line(
    inst: Instance, ctx: Context, containers: Sequence[ContainerInfo] | None
) -> str | None:
    """The forwarder's state, or None when the bind is unset and no forwarder exists.

    `containers` is None when the listing failed. A forwarder left with the bind unset is named,
    since it may still publish the device port until the next `up` removes it.
    """
    bind = bool(_device_bind(inst))
    if containers is None:
        return "device-forward: unknown (the containers could not be listed)" if bind else None
    forwarders = [c for c in containers if c.labels.get(SERVICE_LABEL) == DEVICE_FORWARD_SERVICE]
    if not forwarders:
        return "device-forward: absent" if bind else None
    name = forwarders[0].name
    result = ctx.runner(["docker", "inspect", "--format", CONTAINER_STATE_FORMAT, name])
    state = result.stdout.decode("utf-8", "replace").strip()
    if result.returncode != 0 or not state:
        shown = f"unknown ({_stderr_tail(result)})"
    elif state in ("running", "exited"):
        shown = state
    else:
        shown = f"exited ({state})"
    suffix = "" if bind else f" ({DEVICE_BIND_KEY} is unset; up removes it)"
    return f"device-forward: {shown}{suffix}"


def cmd_status(args: argparse.Namespace, ctx: Context) -> int:
    """Print what runs, whether it is the release, the device forwarder, gaming mode, extraction,
    `/health` and use.

    Exits 1 when a container of the project was made from another compose file, or when the
    containers cannot be listed, since then that check did not run.
    """
    inst = _load_instance(args.root, ctx)
    print(_backend_line(inst, ctx))
    code = EXIT_OK
    containers: list[ContainerInfo] | None
    try:
        containers = _project_containers(ctx, inst.project)
    except FailedError as exc:
        containers = None
        print(f"config-file check: failed ({exc})")
        code = EXIT_FAILED
    else:
        for entry in _foreign_containers(inst, containers):
            print(f"config-file mismatch: {entry}")
            code = EXIT_FAILED
    device_line = _device_forward_line(inst, ctx, containers)
    if device_line is not None:
        print(device_line)
    print(gaming_mode_line(inst.root))
    print(extraction_line(inst, ctx))
    port = inst.env.get("MIST_DAILY_WS_PORT") or CANONICAL_PORTS["MIST_DAILY_WS_PORT"]
    if port.isdigit():
        lines = _health_lines(inst, ctx, port) + _usage_lines(inst, ctx, port)
    else:
        lines = ["health: unreachable", "turns-per-day: unreachable"]
    for line in lines:
        print(line)
    return code


# --- Entry point -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """The command-line parser; each subcommand names its handler in `handler`."""
    parser = argparse.ArgumentParser(
        prog="instance.py",
        description="Create, release, switch and operate a MIST daily instance root.",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    init = commands.add_parser("init", help="create, mark and register an instance root")
    init.add_argument("--root", required=True, help="instance root, outside every git work tree")
    init.add_argument("--instance", choices=INSTANCE_NAMES, default=DEFAULT_INSTANCE)
    init.add_argument(
        "--seed", metavar="DIR", help="fixture whose vault/ contents and seed/ fill mist-memory/"
    )
    init.set_defaults(handler=cmd_init)

    retire = commands.add_parser("retire", help="unmark and unregister a stopped instance root")
    retire.add_argument("--root", required=True)
    retire.set_defaults(handler=cmd_retire)

    release = commands.add_parser(
        "release-build",
        help="build mist-backend:<tag> and mist-extraction:<tag>, and extract the tag's deploy "
        "files",
    )
    release.add_argument("tag", nargs="?", help="annotated tag daily-YYYY.MM.DD[.N] on main")
    release.add_argument(
        "--rehearsal", metavar="REF", help="build rehearsal-<sha7> from any commit instead"
    )
    release.add_argument("--root", required=True)
    release.add_argument(
        "--repo", help="repository to archive; defaults to the work tree holding this script"
    )
    release.set_defaults(handler=cmd_release_build)

    switch = commands.add_parser(
        "switch", help="back up, move daily.env and bin/ to a built release, and run up"
    )
    switch.add_argument("tag", help="a tag release-build has built into release/<tag>/")
    switch.add_argument("--root", required=True)
    switch.add_argument(
        "--skip-backup", action="store_true", help="switch a running instance without a backup"
    )
    switch.set_defaults(handler=cmd_switch)

    up = commands.add_parser("up", help="start the instance from its release")
    up.add_argument("--root", required=True)
    up.add_argument(
        "--wait-minutes",
        type=_minutes,
        default=DEFAULT_UP_WAIT_MINUTES,
        help="how long to wait for Docker to answer",
    )
    up.set_defaults(handler=cmd_up)

    down = commands.add_parser("down", help="stop every container; deletes nothing")
    down.add_argument("--root", required=True)
    down.set_defaults(handler=cmd_down)

    status = commands.add_parser(
        "status", help="show the release, gaming mode, extraction, health and use"
    )
    status.add_argument("--root", required=True)
    status.set_defaults(handler=cmd_status)

    gaming = commands.add_parser("gaming-mode", help="yield the GPU to a game, or end the yield")
    gaming.add_argument("action", choices=("on", "off", "status"))
    gaming.add_argument("--root", required=True)
    gaming.add_argument("--by", choices=GAMING_BY, default="manual")
    gaming.add_argument(
        "--wait-minutes",
        type=_minutes,
        default=DEFAULT_OFF_WAIT_MINUTES,
        help="how long `off` waits for mist-llm to become healthy",
    )
    gaming.set_defaults(handler=cmd_gaming_mode)

    extraction = commands.add_parser(
        "extraction",
        help="set night extraction's mode and converge on it, converge, or show its state",
    )
    extraction.add_argument("action", choices=(*EXTRACTION_MODES, "apply", "status"))
    extraction.add_argument("--root", required=True)
    extraction.set_defaults(handler=cmd_extraction)

    watch = commands.add_parser(
        "gaming-watch",
        help="yield the GPU while a game itself runs; Windows, runs until stopped",
    )
    watch.add_argument("--root", required=True)
    watch.set_defaults(handler=cmd_gaming_watch)

    scan = commands.add_parser(
        "gaming-scan",
        help="print which running programs count as a game and why; Windows, writes no log",
    )
    scan.add_argument("--root", required=True)
    scan.set_defaults(handler=cmd_gaming_scan)

    arm = commands.add_parser("arm-backups", help="start scheduled backups, now and at every up")
    arm.add_argument("--root", required=True)
    arm.set_defaults(handler=cmd_arm_backups)

    disarm = commands.add_parser("disarm-backups", help="stop scheduled backups")
    disarm.add_argument("--root", required=True)
    disarm.set_defaults(handler=cmd_disarm_backups)

    backup_now = commands.add_parser("backup-now", help="run one backup attempt now")
    backup_now.add_argument("--root", required=True)
    backup_now.set_defaults(handler=cmd_backup_now)

    accept = commands.add_parser(
        "accept-backup-baseline", help="make a reviewed suspect backup the new baseline"
    )
    accept.add_argument("--root", required=True)
    accept.add_argument("--label", required=True, help="the directory name under suspect/")
    accept.set_defaults(handler=cmd_accept_backup_baseline)

    drill_start = commands.add_parser(
        "drill-start", help="create a fresh labelled volume and standby for a restore drill"
    )
    drill_start.add_argument("--root", required=True)
    drill_start.add_argument(
        "--drill-root", required=True, help="an empty directory outside the instance root"
    )
    drill_start.add_argument(
        "--artifact", required=True, help="a backup's directory name in MIST_BACKUP_HOST_PATH"
    )
    drill_start.set_defaults(handler=cmd_drill_start)

    drill_commands = (
        (
            "drill-restore",
            cmd_drill_restore,
            "restore the drill's artifact into the drill volume and standby",
        ),
        (
            "drill-verify",
            cmd_drill_verify,
            "compare the restored drill with its artifact and record the result",
        ),
        ("drill-end", cmd_drill_end, "remove the drill's containers; the volume stays"),
        ("promote", cmd_promote, "make a verified drill the instance's graph, state and vault"),
    )
    for name, handler, summary in drill_commands:
        sub = commands.add_parser(name, help=summary)
        sub.add_argument("--root", required=True)
        sub.add_argument("--drill-root", required=True)
        if name == "drill-restore":
            sub.add_argument(
                "--confirm-target",
                metavar="TOKEN",
                help="the token restore printed when run without it",
            )
        sub.set_defaults(handler=handler)
    return parser


def _minutes(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError:
        value = -1
    if value < 0:
        raise argparse.ArgumentTypeError(f"{raw!r} is not a whole number of minutes")
    return value


def main(argv: Sequence[str] | None = None, ctx: Context | None = None) -> int:
    """Run one command and return its exit code (argument errors exit 2 through argparse).

    A handler returns None for success, or an exit code of its own, which `backup-now` (and
    `switch`, when the pre-switch backup fails) passes through from the backup attempt.
    """
    args = build_parser().parse_args(argv)
    try:
        code = args.handler(args, ctx if ctx is not None else Context())
    except InstanceError as exc:
        print(f"instance.py {args.command}: {exc.label}: {exc}", file=sys.stderr)
        return exc.exit_code
    except volume_safety.DestructiveCommandError as exc:
        # Exit 1, not 2: a refusal mid-way, after `promote` has moved directories, has changed
        # something.
        print(
            f"instance.py {args.command}: refused a volume-deleting command: "
            f"{' '.join(exc.argv)}",
            file=sys.stderr,
        )
        return EXIT_FAILED
    return EXIT_OK if code is None else code


if __name__ == "__main__":
    sys.exit(main())
