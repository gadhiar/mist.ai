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
real waiting.

Exit codes: 0 done; 2 refused, nothing changed; 1 failed part-way, with the state named.
`backup-now` returns the one-shot backup's own exit code.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
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
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Literal, Protocol

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
INSTANCE_DIRS = ("state", "mist-memory", "logs", "ops", "models", "release", "retired", "bin")
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
REVISION_LABEL = "org.opencontainers.image.revision"
BACKUP_ONCE = ("python", "-m", "scripts.backup.scheduler", "--once")

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


@dataclass(frozen=True, slots=True)
class CommandResult:
    """What one Docker or git command returned."""

    returncode: int
    stdout: bytes = b""
    stderr: bytes = b""


class Runner(Protocol):
    """Runs one command, with optional bytes on its standard input."""

    def __call__(self, argv: Sequence[str], *, stdin: bytes | None = None) -> CommandResult:
        """Run `argv` and return its exit code and captured output."""
        ...


def subprocess_runner(argv: Sequence[str], *, stdin: bytes | None = None) -> CommandResult:
    """Run `argv` without a shell, capturing its output.

    A missing program reads as exit 127, the shell's convention, so a host without Docker on
    its PATH gets the same refusal or failure message as one whose daemon is down.
    """
    command = list(argv)
    try:
        if stdin is None:
            completed = subprocess.run(  # nosec B603 - fixed argument list, no shell
                command, stdin=subprocess.DEVNULL, capture_output=True, check=False
            )
        else:
            completed = subprocess.run(  # nosec B603 - fixed argument list, no shell
                command, input=stdin, capture_output=True, check=False
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
    """

    runner: Runner = subprocess_runner
    clock: Callable[[], datetime] = _utc_now
    environ: Mapping[str, str] = field(default_factory=lambda: os.environ)
    home: Path = field(default_factory=Path.home)
    script_path: Path = field(default_factory=_default_script_path)
    sleep: Callable[[float], None] = time.sleep
    monotonic: Callable[[], float] = time.monotonic
    fetch: Callable[[str, float], bytes] = urllib_fetch


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
    """Every path in the archive, and the `deploy/` and `scripts/daily/` entries to extract.

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


def cmd_release_build(args: argparse.Namespace, ctx: Context) -> None:
    """Build `mist-backend:<tag>` from a tag's archive and unpack its deploy files.

    The image is built from `git archive` of the commit, never from a working tree, so what
    runs is exactly what the tag names. The image ID is recorded because a later
    `docker build -t mist-backend:<tag>` anywhere could re-point the tag; `up` and `switch`
    compare against the record.
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
    missing = [path for path in (DOCKERFILE, COMPOSE_FILE) if path not in names]
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

    inspected = ctx.runner(["docker", "image", "inspect", "--format", "{{.Id}}", image])
    image_id = inspected.stdout.decode("ascii", "replace").strip()
    if inspected.returncode != 0 or not image_id:
        raise FailedError(
            f"built {image} but could not read its image ID ({_stderr_tail(inspected)}); "
            f"release/{tag}/ and ops/release-{tag}.json not written"
        )

    try:
        _extract_release(root / "release", tag, selected)
    except OSError as exc:
        raise FailedError(
            f"built {image} ({image_id}) but could not extract release/{tag}/: {exc}; "
            f"ops/release-{tag}.json not written"
        ) from exc

    record = {
        "tag": tag,
        "commit": commit,
        "image_id": image_id,
        "built_at": _format_utc(ctx.clock()),
    }
    record_path = root / "ops" / f"release-{tag}.json"
    try:
        record_path.parent.mkdir(parents=True, exist_ok=True)
        _write_atomic(record_path, (json.dumps(record, indent=2) + "\n").encode("utf-8"))
    except OSError as exc:
        raise FailedError(
            f"built {image} ({image_id}) and extracted release/{tag}/, but could not write "
            f"{record_path}: {exc}"
        ) from exc

    print(f"built {image} ({image_id}) from {commit}")
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
        """The `docker compose` prefix every project command starts with (item 5)."""
        return [
            "docker",
            "compose",
            "-p",
            self.project,
            "-f",
            str(self.compose_file()),
            "--env-file",
            str(self.env_path),
        ]


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
    path = _absolute_path(raw, ctx.home)
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
    """`<name> uses <files>` for each container not made from the release's compose file alone.

    Compose records the files a container was created from in a label, so a
    `docker compose -p mist-daily -f <another file> up` shows here.
    """
    expected = _normalized(str(inst.compose_file()))
    foreign = []
    for container in containers:
        label = str(container.labels.get(CONFIG_FILES_LABEL, ""))
        files = [entry for entry in label.split(",") if entry]
        if len(files) != 1 or _normalized(files[0]) != expected:
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
    services that answer turns from starting.
    """
    _check_static_configuration(inst, ctx)
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
    try:
        foreign = _foreign_containers(inst, _project_containers(ctx, inst.project))
    except FailedError as exc:
        problems.append(f"the containers' compose files could not be checked: {exc}")
    else:
        problems.extend(f"container {entry}, not {inst.compose_file()}" for entry in foreign)
    if problems:
        raise FailedError(f"release {inst.tag} is up, but: " + "; ".join(problems))


def cmd_up(args: argparse.Namespace, ctx: Context) -> None:
    """Start the instance; the sign-in task runs this from `<root>/bin/`."""
    instance_up(_load_instance(args.root, ctx), ctx, args.wait_minutes)


def cmd_down(args: argparse.Namespace, ctx: Context) -> None:
    """Stop every container of the project and remove nothing.

    `stop`, never `down`: the containers, networks and volumes stay, so nothing a later
    `up` needs is deleted. Profile `backup` is enabled so mist-backup stops too.
    """
    inst = _load_instance(args.root, ctx, require_release=False)
    if _tag_is_valid(inst.tag) and inst.compose_file().is_file():
        argv = [*inst.compose(), "--profile", BACKUP_PROFILE, "stop"]
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


def _backend_running(inst: Instance, ctx: Context) -> bool:
    result = ctx.runner(
        ["docker", "compose", "-p", inst.project, "ps", "--status", "running", "--format", "json"]
    )
    fix = "nothing changed; pass --skip-backup to switch without the pre-switch backup"
    if result.returncode != 0:
        raise RefusedError(
            f"cannot tell whether {inst.project} runs ({_stderr_tail(result)}); {fix}"
        )
    try:
        rows = _parse_ps_json(result.stdout)
    except ValueError as exc:
        raise RefusedError(f"cannot read {inst.project}'s containers ({exc}); {fix}") from exc
    return any(row.get("Service") == BACKEND_SERVICE for row in rows)


def cmd_switch(args: argparse.Namespace, ctx: Context) -> int | None:
    """Move the instance to another built release: back up, rewrite the tag, copy, `up`.

    `switch <previous tag>` is the rollback. It restores the code but not a data migration the
    newer release made, which is why a running instance is backed up first, on the release
    that wrote its data.
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


# --- Gaming mode -----------------------------------------------------------------------------


def _gpu_yield_poll_s(inst: Instance) -> float:
    raw = inst.env.get("MIST_GPU_YIELD_POLL_S", "")
    if not raw:
        return DEFAULT_GPU_YIELD_POLL_S
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not math.isfinite(value) or value <= 0:
        raise RefusedError(f"{inst.env_path} sets MIST_GPU_YIELD_POLL_S={raw}, not seconds")
    return value


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
    """Yield the GPU: mark the yield, let the backend see it, then stop mist-llm.

    The backend polls the marker every `MIST_GPU_YIELD_POLL_S`; waiting two polls before the
    stop means `/health` never sees a stopped LLM while the backend still reads normal. A
    manual `on` takes over a watcher's yield, so the watcher cannot end a hold Raj asked for;
    a watcher `on` never changes a manual one.

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
    """End the yield: start mist-llm, wait until compose calls it healthy, then drop the marker.

    While the LLM loads, turns still get the yielded reply rather than a failure, and a timeout
    leaves the marker, so `/health` keeps reading yielded, which is still true. `should_abort`,
    called on each poll, lets the watcher stop mist-llm again when a game starts meanwhile.

    Raises:
        InstanceError: When mist-llm cannot be started, does not become healthy within
            `wait_minutes`, or an abort's stop fails; the marker stays in each case.
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
    gate = payload.get("gate")
    if not isinstance(days, list) or not isinstance(gate, dict):
        return ["turns-per-day: unreadable"]
    counts = ", ".join(
        f"{day.get('date')} {day.get('real_turns')}" for day in days if isinstance(day, dict)
    )
    met = "met" if gate.get("met") is True else "not met"
    return [
        f"turns-per-day: {counts}",
        f"gate: {met}, {gate.get('days_used')} of {gate.get('window_days')} days used, "
        f"{gate.get('required_days')} required",
    ]


def cmd_status(args: argparse.Namespace, ctx: Context) -> int:
    """Print what runs, whether it is the release, gaming mode, `/health` and the use counts.

    Exits 1 when a container of the project was made from another compose file, or when the
    containers cannot be listed, since then that check did not run.
    """
    inst = _load_instance(args.root, ctx)
    print(_backend_line(inst, ctx))
    code = EXIT_OK
    try:
        foreign = _foreign_containers(inst, _project_containers(ctx, inst.project))
    except FailedError as exc:
        print(f"config-file check: failed ({exc})")
        code = EXIT_FAILED
    else:
        for entry in foreign:
            print(f"config-file mismatch: {entry}")
            code = EXIT_FAILED
    print(gaming_mode_line(inst.root))
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
        "release-build", help="build mist-backend:<tag> and extract the tag's deploy files"
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

    status = commands.add_parser("status", help="show the release, gaming mode, health and use")
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

    arm = commands.add_parser("arm-backups", help="start scheduled backups, now and at every up")
    arm.add_argument("--root", required=True)
    arm.set_defaults(handler=cmd_arm_backups)

    disarm = commands.add_parser("disarm-backups", help="stop scheduled backups")
    disarm.add_argument("--root", required=True)
    disarm.set_defaults(handler=cmd_disarm_backups)

    backup_now = commands.add_parser("backup-now", help="run one backup attempt now")
    backup_now.add_argument("--root", required=True)
    backup_now.set_defaults(handler=cmd_backup_now)
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
    return EXIT_OK if code is None else code


if __name__ == "__main__":
    sys.exit(main())
