"""Host CLI for a MIST daily instance: `python <path>/instance.py <command> --root <root>`.

An instance root is a host directory outside every git work tree that holds one instance's
state, vault, logs, operations records, model weights and extracted releases. This CLI is the
only tool that creates, retires or builds releases for one, so it runs on host Python with the
standard library alone: after `switch` it runs from `<root>/bin/`, where neither the repository
nor its dependencies exist. It imports nothing but the standard library and modules beside it.

Every Docker and git call goes through `Context.runner`, one injected callable, so the tests
assert exact argument lists with no Docker or git present.

Exit codes: 0 done; 2 refused, nothing changed; 1 failed part-way, with the state named.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import shutil
import subprocess  # nosec B404 - runs fixed docker and git argument lists, never a shell
import sys
import tarfile
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Protocol

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
STDERR_TAIL_LINES = 20


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


@dataclass(frozen=True, slots=True)
class Context:
    """Everything a command reads from outside its arguments, injectable for tests.

    `script_path` is where this file runs from: `init` finds the env example relative to it,
    and `release-build` finds its default repository from it.
    """

    runner: Runner = subprocess_runner
    clock: Callable[[], datetime] = _utc_now
    environ: Mapping[str, str] = field(default_factory=lambda: os.environ)
    home: Path = field(default_factory=Path.home)
    script_path: Path = field(default_factory=_default_script_path)


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


# --- Entry point -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """The command-line parser; each subcommand names its handler in `handler`."""
    parser = argparse.ArgumentParser(
        prog="instance.py",
        description="Create, retire and build releases for a MIST daily instance root.",
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
    return parser


def main(argv: Sequence[str] | None = None, ctx: Context | None = None) -> int:
    """Run one command and return its exit code (argument errors exit 2 through argparse)."""
    args = build_parser().parse_args(argv)
    try:
        args.handler(args, ctx if ctx is not None else Context())
    except InstanceError as exc:
        print(f"instance.py {args.command}: {exc.label}: {exc}", file=sys.stderr)
        return exc.exit_code
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
