"""Host driver for the mist-model-bench goal.

Runs on the Windows host in a no-gaming session (Python 3.11.9) and its
`voice` subcommand shells out to a copy of `probes/voice_vram.py` running
inside the mist-backend container (Python 3.11.0rc1). Stops MIST's
production containers, serves a pinned llama-server build under the name
`mist-bench-llm`, runs benchmark suites against it, and restores MIST
exactly. Stdlib only (see `probes/voice_vram.py` for the one exception).

Invocation: `python -m scripts.model_bench.bench_host <subcommand>` from the
repo root. See README.md for the full S1-S3 host procedure
(snapshot -> stop -> serve -> run -> unserve -> restore), the results
layout, and the environment variables below.

Configuration (flag, env fallback):
    --models-dir      MODELS_DIR                 mounted :ro at /models
    --layout-dir      MODEL_BENCH_LAYOUT_DIR      command-center layout spike dir
    --results-root    MODEL_BENCH_RESULTS_ROOT    MUST resolve outside this git work tree
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    # Allow `python scripts/model_bench/bench_host.py` in addition to the
    # documented `python -m scripts.model_bench.bench_host` invocation.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.model_bench.probes import help_flags as help_flags_probe
    from scripts.model_bench.probes import nvidia_smi as nvidia_smi_probe
else:
    from .probes import help_flags as help_flags_probe
    from .probes import nvidia_smi as nvidia_smi_probe

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
ARMS_JSON_PATH = PACKAGE_DIR / "arms.json"
DECISION_RULES_PATH = PACKAGE_DIR / "decision_rules.json"
FIXTURES_DIR = REPO_ROOT / "tests" / "unit" / "model_bench" / "fixtures" / "host"
DEFAULT_COMPOSE_PATH = REPO_ROOT / "docker-compose.yml"

BENCH_LLM_CONTAINER = "mist-bench-llm"
BENCH_LLM_PORT = 8080
BENCH_LLM_HOST_BIND = "127.0.0.1"
SNAPSHOT_CONTAINERS: tuple[str, ...] = ("mist-neo4j", "mist-llm", "mist-backend")

# Mirrors scripts/eval_harness/run.py:61-74 (DEFAULT_TEST_ORDER). Duplicated
# here, not imported, so this module stays stdlib-only -- the harness module
# imports PyYAML. The harness suite is invoked as a subprocess, never by
# direct import.
HARNESS_DEFAULT_TEST_ORDER: tuple[str, ...] = (
    "speed_minimal",
    "schema_conformance",
    "schema_conformance_lenient",
    "schema_conformance_fewshot",
    "tool_selection",
    "personality",
    "adversarial_persona",
    "rag_integration",
    "coherence",
    "cot_reasoning",
    "long_turn_coherence",
    "speed",
)
HARNESS_SCHEMA_JSON_OBJECT_TEST = "schema_conformance_json_object"
HARNESS_TUNING_TESTS: tuple[str, ...] = ("schema_conformance",)

LAYOUT_MAX_TOKENS_OFF = 256
LAYOUT_MAX_TOKENS_ON = 4096
LAYOUT_LAYOUTS_PER_SIZE = {"screen": 2, "finalist": 6}

RESTORE_DIFF_FIELDS: tuple[str, ...] = ("Id", "Image", "Config.Cmd", "Config.Env", "HostConfig")

KNOWN_ARM_KEYS = frozenset(
    {
        "base",
        "gguf",
        "image",
        "family",
        "thinking",
        "suites",
        "harness",
        "extra_args",
        "arg_overrides",
        "params_required",
        "param_arg_map",
        "stop_neo4j",
        "optional",
        "tuning",
        "tuning_note",
    }
)

ARM_DEFAULTS: dict[str, Any] = {
    "family": "gemma",
    "thinking": None,
    "suites": [],
    "harness": None,
    "extra_args": [],
    "arg_overrides": {},
    "params_required": [],
    "param_arg_map": {},
    "stop_neo4j": False,
    "optional": False,
    "tuning": False,
    "tuning_note": None,
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class BenchHostError(RuntimeError):
    """Base class for every refusal this driver raises."""


class ArmConfigError(BenchHostError):
    """arms.json failed to load or an arm failed to resolve."""


class ImageRefError(BenchHostError):
    """An image reference token failed to resolve to a pinned image."""


class ResultsRootError(BenchHostError):
    """--results-root is missing or resolves inside the git work tree."""


class DecisionRulesError(BenchHostError):
    """decision_rules.json is missing, untracked, or dirty."""


class ServedArmMismatchError(BenchHostError):
    """The running mist-bench-llm container does not match the requested arm."""


class DockerError(BenchHostError):
    """A docker subprocess call failed."""


class RestoreError(BenchHostError):
    """restore's post-start health/diff checks failed."""


class SuiteOutputExistsError(BenchHostError):
    """A suite's output path already exists; refusing to overwrite it."""


class RunMetaConfigMismatchError(BenchHostError):
    """This call's config differs from the arm dir's existing meta.json."""


class MissingParamError(ArmConfigError):
    """An arm's REQUIRED param was not supplied via --param."""


class ContainerExitedError(BenchHostError):
    """The container exited, or was removed, while `serve` was waiting for /health.

    Distinct from a plain health-check TimeoutError: the container is gone,
    not merely slow, so there is nothing further to wait for and the caller
    should stop immediately rather than let the full --timeout elapse.

    `absent=True` means `docker inspect` reported "no such object" (the
    container no longer exists at all, e.g. something else already removed
    it) rather than a parsed `State.Status` of `exited`/`dead`. The two are
    kept distinct in the message -- "exited (code=N)" vs "no longer
    exists" -- since callers (`cmd_serve`) also use `absent` to skip a
    `docker rm` that would otherwise fail on a container that is not there.
    """

    def __init__(self, name: str, exit_code: int | None, *, absent: bool = False):
        self.name = name
        self.exit_code = exit_code
        self.absent = absent
        detail = "no longer exists" if absent else f"exited (code={exit_code})"
        super().__init__(f"{name} {detail} while waiting for /health")


class ContainerStateUnknownError(BenchHostError):
    """The container's docker state could not be determined for too long.

    Raised by `wait_for_llama_health` when repeated `probe_container_state`
    calls return `unknown` (a slow or failing `docker inspect`, e.g. the S2
    incident's low-free-memory host) for longer than `unknown_limit_s`,
    with no known reading in between. This says nothing about whether the
    container exited -- it may well be healthy -- so the message never
    claims that, and callers must not stop or remove the container on the
    strength of this error alone.
    """

    def __init__(self, name: str, unknown_for_s: float):
        self.name = name
        self.unknown_for_s = unknown_for_s
        super().__init__(f"{name} state could not be determined for {unknown_for_s:.0f}s")


# ---------------------------------------------------------------------------
# arms.json loading and resolution
# ---------------------------------------------------------------------------


def load_arms_doc(path: Path = ARMS_JSON_PATH) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ArmConfigError(f"cannot read arms.json at {path}: {exc}") from exc
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ArmConfigError(f"arms.json at {path} is not valid JSON: {exc}") from exc
    if "arms" not in doc or not isinstance(doc["arms"], dict):
        raise ArmConfigError(f"arms.json at {path} has no top-level 'arms' object")
    return doc


def resolve_arm(arms_doc: dict[str, Any], arm_id: str, _stack: tuple[str, ...] = ()) -> dict[str, Any]:
    """Resolve one arm's full config, following `base` inheritance.

    Raises ArmConfigError on: unknown arm id, unknown keys on any arm in the
    chain, a base-inheritance cycle, a missing required key (gguf/image)
    after resolution, or a declared `params_required` entry with no
    matching `param_arg_map` template.
    """
    if arm_id in _stack:
        chain = " -> ".join((*_stack, arm_id))
        raise ArmConfigError(f"cycle in base inheritance: {chain}")
    arms = arms_doc["arms"]
    if arm_id not in arms:
        raise ArmConfigError(f"unknown arm {arm_id!r}")
    raw = arms[arm_id]
    unknown = set(raw) - KNOWN_ARM_KEYS
    if unknown:
        raise ArmConfigError(f"arm {arm_id!r} has unknown keys: {sorted(unknown)}")

    if "base" in raw:
        merged = dict(resolve_arm(arms_doc, raw["base"], (*_stack, arm_id)))
        merged.pop("id", None)
    else:
        merged = dict(ARM_DEFAULTS)

    for key, value in raw.items():
        if key == "base":
            continue
        merged[key] = value

    for required_key in ("gguf", "image"):
        if required_key not in merged:
            raise ArmConfigError(f"arm {arm_id!r} resolves without required key {required_key!r}")

    missing_templates = [p for p in merged["params_required"] if p not in merged["param_arg_map"]]
    if missing_templates:
        raise ArmConfigError(
            f"arm {arm_id!r} declares params_required without a param_arg_map "
            f"entry: {missing_templates}"
        )

    merged["id"] = arm_id
    return merged


def resolve_all_arms(arms_doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Resolve every arm in arms.json; raises on the first failure."""
    return {arm_id: resolve_arm(arms_doc, arm_id) for arm_id in arms_doc["arms"]}


def resolve_harness_tests(harness_cfg: dict[str, Any]) -> list[str]:
    tests = harness_cfg["tests"]
    if tests == "default":
        return [*HARNESS_DEFAULT_TEST_ORDER, HARNESS_SCHEMA_JSON_OBJECT_TEST]
    if tests == "tuning":
        return list(HARNESS_TUNING_TESTS)
    if isinstance(tests, list):
        return list(tests)
    raise ArmConfigError(f"unknown harness 'tests' sentinel: {tests!r}")


def layout_max_tokens(thinking: dict[str, Any] | None) -> int:
    mode = None if thinking is None else thinking.get("mode")
    return LAYOUT_MAX_TOKENS_ON if mode == "on" else LAYOUT_MAX_TOKENS_OFF


# ---------------------------------------------------------------------------
# Server argument building
# ---------------------------------------------------------------------------


def build_server_args(
    arms_doc: dict[str, Any], resolved_arm: dict[str, Any], params: dict[str, str]
) -> list[str]:
    """Build the full llama-server argv tail (everything after `-m <path>`).

    Raises MissingParamError if a REQUIRED param (arms.json's
    `params_required`) is absent from `params`.
    """
    missing = [p for p in resolved_arm["params_required"] if p not in params]
    if missing:
        raise MissingParamError(
            f"arm {resolved_arm['id']!r} requires --param for: {missing}"
        )

    args: list[str] = list(arms_doc["common_args"])

    # arg_overrides replaces a common_args flag's value IN PLACE rather than
    # appending a second occurrence of the flag (which would leave argv
    # carrying it twice; which occurrence llama.cpp applies is not verified
    # here, and this driver's own invariant is that no flag appears twice).
    # Only common_args flags may be overridden
    # -- an arg_overrides entry naming a flag common_args does not have is a
    # config error, not a silent no-op.
    overrides: dict[str, str] = resolved_arm["arg_overrides"]
    for flag, value in overrides.items():
        if flag not in args:
            raise ArmConfigError(
                f"arm {resolved_arm['id']!r} arg_overrides names {flag!r}, "
                f"which is not in common_args"
            )
        args[args.index(flag) + 1] = str(value)

    family = resolved_arm["family"]
    if family not in arms_doc["sampling"]:
        raise ArmConfigError(f"arm {resolved_arm['id']!r} has unknown family {family!r}")
    args.extend(arms_doc["sampling"][family])

    thinking = resolved_arm["thinking"]
    if thinking is not None:
        mode = thinking.get("mode")
        if mode not in arms_doc["thinking_args"]:
            raise ArmConfigError(f"arm {resolved_arm['id']!r} has unknown thinking mode {mode!r}")
        template = list(arms_doc["thinking_args"][mode])
        if mode == "on":
            budget = thinking.get("budget")
            if budget is None:
                raise ArmConfigError(f"arm {resolved_arm['id']!r} has thinking mode 'on' without a budget")
            template = [tok.replace("{budget}", str(budget)) for tok in template]
        args.extend(template)

    args.extend(resolved_arm["extra_args"])

    for param_name in resolved_arm["params_required"]:
        flag = resolved_arm["param_arg_map"][param_name]
        args.extend([flag, str(params[param_name])])

    return args


def build_model_args(resolved_arm: dict[str, Any], arms_doc: dict[str, Any], params: dict[str, str]) -> list[str]:
    """Full argv tail passed to the container's entrypoint (llama-server)."""
    gguf = resolved_arm["gguf"]
    return ["-m", f"/models/{gguf}", *build_server_args(arms_doc, resolved_arm, params)]


def build_docker_run_args(
    resolved_arm: dict[str, Any],
    arms_doc: dict[str, Any],
    *,
    image_ref: str,
    models_dir: str,
    params: dict[str, str] | None = None,
) -> list[str]:
    """Full `docker run` argv for serving `resolved_arm`, argv-list, shell=False.

    No `--rm`: `serve` must be able to `docker logs` the container after it
    exits (a failed start, e.g. a rejected flag) before removing it itself.
    An auto-removed container's logs are gone before anything could read them.
    """
    params = params or {}
    return [
        "docker",
        "run",
        "-d",
        "--name",
        BENCH_LLM_CONTAINER,
        "--gpus",
        "all",
        "-p",
        f"{BENCH_LLM_HOST_BIND}:{BENCH_LLM_PORT}:8080",
        "-v",
        f"{models_dir}:/models:ro",
        image_ref,
        *build_model_args(resolved_arm, arms_doc, params),
    ]


# ---------------------------------------------------------------------------
# Flag check against the pinned build's `llama-server --help`
# ---------------------------------------------------------------------------


def arm_targets_pinned_build(resolved_arm: dict[str, Any]) -> bool:
    """True if `resolved_arm`'s image token is the pinned compose build.

    c0-old targets `snapshot:mist-llm`, the current production b8808 snapshot
    -- a different llama.cpp build than the pinned b11151 `compose:mist-llm`
    image `plan --host-checks` fetches `--help` from. Checking c0-old's argv
    against b11151's flag list would check the wrong binary.
    """
    return resolved_arm["image"] == "compose:mist-llm"


def check_arm_flags(
    help_flags: set[str],
    arms_doc: dict[str, Any],
    resolved_arm: dict[str, Any],
    params: dict[str, str],
) -> list[str]:
    """Flags in `resolved_arm`'s built argv that `help_flags` does not list.

    Builds the arm's full argv (`-m <path>` plus every server flag) the same
    way `serve` would, with `params` filled in (the caller supplies a dummy
    value per REQUIRED param), then defers to
    `probes.help_flags.unknown_flags`.
    """
    argv = build_model_args(resolved_arm, arms_doc, params)
    return help_flags_probe.unknown_flags(argv, help_flags)


# ---------------------------------------------------------------------------
# Compose / snapshot image ref resolution
# ---------------------------------------------------------------------------

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def parse_compose_image(compose_path: Path, service: str = "mist-llm") -> str:
    """Parse the `image:` value for `service` out of a docker-compose YAML file.

    Stdlib line-based parsing (no PyYAML dependency): finds the `services:`
    block, then the `<service>:` entry inside it by indentation, then that
    entry's `image:` line. Refuses unless the image reference carries an
    `@sha256:<64 hex>` digest pin.
    """
    try:
        lines = compose_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ImageRefError(f"cannot read compose file at {compose_path}: {exc}") from exc

    services_indent: int | None = None
    current_service: str | None = None
    current_service_indent: int | None = None
    image_line: str | None = None

    for raw in lines:
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        stripped = raw.strip()

        if services_indent is None:
            if stripped == "services:":
                services_indent = indent
            continue

        if indent <= services_indent:
            # Left the services: block entirely.
            break

        is_service_header = (
            current_service_indent is None or indent <= current_service_indent
        ) and stripped.endswith(":") and not stripped.startswith("- ")
        if is_service_header:
            if current_service == service and image_line is not None:
                break
            current_service = stripped[:-1]
            current_service_indent = indent
            continue

        if current_service == service and stripped.startswith("image:"):
            image_line = stripped[len("image:") :].strip().strip('"').strip("'")
            break

    if current_service != service and image_line is None:
        raise ImageRefError(f"service {service!r} not found in {compose_path}")
    if image_line is None:
        raise ImageRefError(f"service {service!r} in {compose_path} has no 'image:' key")
    if "@sha256:" not in image_line:
        raise ImageRefError(
            f"image for service {service!r} in {compose_path} has no @sha256 digest pin: "
            f"{image_line!r}"
        )
    digest = image_line.split("@sha256:", 1)[1]
    if not _SHA256_RE.fullmatch(digest):
        raise ImageRefError(
            f"image digest for service {service!r} in {compose_path} is not 64 hex chars: "
            f"{digest!r}"
        )
    return image_line


def resolve_image_ref(
    token: str,
    *,
    compose_path: Path = DEFAULT_COMPOSE_PATH,
    snapshot_path: Path,
    service: str = "mist-llm",
) -> str:
    """Resolve an arms.json 'image' token to a concrete image reference."""
    if token == "compose:mist-llm":
        return parse_compose_image(compose_path, service=service)
    if token == "snapshot:mist-llm":
        if not snapshot_path.exists():
            raise ImageRefError(f"no snapshot at {snapshot_path}; run `snapshot` first")
        try:
            data = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ImageRefError(f"cannot read snapshot at {snapshot_path}: {exc}") from exc
        entry = data.get(service)
        if not entry or "Image" not in entry:
            raise ImageRefError(f"snapshot at {snapshot_path} has no {service!r}.Image entry")
        return entry["Image"]
    return token


# ---------------------------------------------------------------------------
# Results-root guard (decision 8: raw logs never land in the repo tree)
# ---------------------------------------------------------------------------


def check_results_root_outside_repo(results_root: Path, repo_root: Path = REPO_ROOT) -> None:
    """Refuse a --results-root that resolves inside the git work tree.

    Uses Path.resolve(strict=False), so this works before the directory
    exists. Comparison is by prefix on the resolved paths (Path.relative_to
    raising ValueError means "outside"), not by string containment.
    """
    rr = results_root.resolve()
    rp = repo_root.resolve()
    try:
        rr.relative_to(rp)
    except ValueError:
        return
    raise ResultsRootError(f"--results-root {rr} is inside the git work tree {rp}; refusing")


def discover_repo_root(start: Path) -> Path:
    """Walk upward from `start` looking for a `.git` entry (dir or file)."""
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return REPO_ROOT


# ---------------------------------------------------------------------------
# Host configuration
# ---------------------------------------------------------------------------


class HostConfig:
    """Resolved --models-dir / --layout-dir / --results-root, flag over env."""

    def __init__(self, models_dir: Path | None, layout_dir: Path | None, results_root: Path | None):
        self.models_dir = models_dir
        self.layout_dir = layout_dir
        self.results_root = results_root


def resolve_host_config(args: argparse.Namespace, *, enforce_results_root: bool = True) -> HostConfig:
    warnings: list[str] = []

    models_dir_raw = args.models_dir or os.environ.get("MODELS_DIR")
    models_dir = Path(models_dir_raw).resolve() if models_dir_raw else None
    if models_dir is None:
        warnings.append("[WARN] --models-dir / MODELS_DIR not set")

    layout_dir_raw = args.layout_dir or os.environ.get("MODEL_BENCH_LAYOUT_DIR")
    layout_dir = Path(layout_dir_raw).resolve() if layout_dir_raw else None
    if layout_dir is None:
        warnings.append("[WARN] --layout-dir / MODEL_BENCH_LAYOUT_DIR not set")

    results_root_raw = args.results_root or os.environ.get("MODEL_BENCH_RESULTS_ROOT")
    results_root = Path(results_root_raw).resolve() if results_root_raw else None
    if results_root is None:
        warnings.append("[WARN] --results-root / MODEL_BENCH_RESULTS_ROOT not set")
    elif enforce_results_root:
        check_results_root_outside_repo(results_root)

    for line in warnings:
        print(line)

    return HostConfig(models_dir, layout_dir, results_root)


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def _run_git(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    # GIT_DIR / GIT_WORK_TREE, if set in the environment, override `-C`
    # entirely (git checks them before deriving anything from cwd), which
    # would silently point every call here at some other repo/worktree
    # (worker containers export both, pinned to this worktree, for their own
    # sandboxing) instead of `cwd`. Scrubbed so `-C cwd` is the only thing
    # that decides which repo a call operates on.
    env = {k: v for k, v in os.environ.items() if k not in ("GIT_DIR", "GIT_WORK_TREE")}
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        shell=False,
        env=env,
    )


def git_head(repo_root: Path) -> str | None:
    proc = _run_git(["rev-parse", "HEAD"], cwd=repo_root)
    return proc.stdout.strip() if proc.returncode == 0 else None


def git_is_dirty(repo_root: Path) -> bool:
    proc = _run_git(["status", "--porcelain"], cwd=repo_root)
    return bool(proc.stdout.strip())


def collect_git_state(repo_root: Path, layout_dir: Path | None) -> dict[str, Any]:
    return {
        "mist_ai": git_head(repo_root),
        "mist_ai_dirty": git_is_dirty(repo_root),
        "command_center": git_head(layout_dir) if layout_dir is not None else None,
    }


def check_decision_rules_clean(path: Path, repo_root: Path) -> str:
    """Require `path` to exist, be tracked, and be clean; return its sha256 hex digest."""
    if not path.exists():
        raise DecisionRulesError(f"decision_rules.json missing at {path}")
    try:
        rel = path.resolve().relative_to(repo_root.resolve())
    except ValueError as exc:
        raise DecisionRulesError(f"{path} is not inside repo {repo_root}") from exc

    tracked = _run_git(["ls-files", "--error-unmatch", str(rel)], cwd=repo_root)
    if tracked.returncode != 0:
        raise DecisionRulesError(f"decision_rules.json at {path} is not tracked in git")

    status = _run_git(["status", "--porcelain", "--", str(rel)], cwd=repo_root)
    if status.stdout.strip():
        raise DecisionRulesError(f"decision_rules.json at {path} has uncommitted changes: {status.stdout.strip()}")

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


# ---------------------------------------------------------------------------
# Docker helpers (argv lists only, shell=False everywhere)
# ---------------------------------------------------------------------------


def docker_inspect(names: list[str]) -> dict[str, dict[str, Any]]:
    """docker inspect the given container names; returns {name: inspect_dict}."""
    proc = subprocess.run(
        ["docker", "inspect", *names], capture_output=True, text=True, shell=False
    )
    if proc.returncode != 0:
        raise DockerError(f"docker inspect {names} failed: {proc.stderr.strip()}")
    entries = json.loads(proc.stdout)
    by_name: dict[str, dict[str, Any]] = {}
    for entry in entries:
        name = entry.get("Name", "").lstrip("/")
        by_name[name] = entry
    return by_name


def docker_is_running(name: str) -> bool:
    proc = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", name],
        capture_output=True,
        text=True,
        shell=False,
    )
    return proc.returncode == 0 and proc.stdout.strip() == "true"


def docker_container_exists(name: str) -> bool:
    """True if `name` names any container, running or exited.

    Without `--rm` (see `build_docker_run_args`), a failed or stopped
    `mist-bench-llm` container lingers under its name until `unserve`
    removes it, and `docker run --name` refuses to reuse a taken name --
    `serve` must check this itself and refuse with a clear message rather
    than let that `docker run` failure surface as an opaque DockerError.

    NOT used by `cmd_serve`'s or `cmd_restore`'s own pre-flight checks --
    any `docker inspect` failure here, for any reason, reads as "no
    container", which fails OPEN on a slow or misbehaving docker CLI (the
    S2 host-checks review's "mirror flaw"). Those call sites use
    `probe_container_state` instead, which keeps a probe that could not be
    classified in its own `unknown` bucket rather than folding it into
    `False`. Retained here only in case some other caller wants the
    two-state simplification; do not add a new caller that needs the
    exists-vs-not distinction to be safe.
    """
    try:
        docker_inspect([name])
    except DockerError:
        return False
    return True


CONTAINER_PROBE_TIMEOUT_S = 15.0
"""Default per-call timeout for `probe_container_state`'s `docker inspect`.

15s: generous enough for a docker CLI/daemon under host memory pressure
(the S2 incident's 2.5 GB free RAM made ordinary `docker inspect` calls
slow) while still short enough that one hung call cannot, by itself,
consume a large share of `wait_for_llama_health`'s `unknown_limit_s`
budget (default 180s -- see `wait_for_llama_health`).
"""

_ABSENT_STDERR_MARKERS = ("no such object", "no such container")


class ContainerProbe:
    """One `probe_container_state` result.

    `state` is one of `"running"`, `"exited"`, `"absent"`, `"unknown"`; see
    `probe_container_state` for exactly what puts a probe in each bucket.
    `detail` is a short, always-printable human-readable reason. `exit_code`
    is only meaningful when `state == "exited"`.
    """

    def __init__(self, state: str, detail: str, *, exit_code: int | None = None):
        self.state = state
        self.detail = detail
        self.exit_code = exit_code


def probe_container_state(
    name: str, *, timeout_s: float = CONTAINER_PROBE_TIMEOUT_S
) -> ContainerProbe:
    """Classify `name`'s docker state without ever raising on a docker failure.

    Runs `docker inspect -f '{{json .State}}' <name>` through
    `subprocess.run(..., timeout=timeout_s)` and buckets the result into
    four states, in order of how positive the signal is:

    - `"exited"`: the call succeeded and parsed `State.Status` is `exited`
      or `dead`. `exit_code` carries `State.ExitCode`.
    - `"absent"`: the call failed (non-zero exit) AND its stderr contains
      "no such object" or "no such container" (case-insensitive) -- docker
      prints `Error: No such object: <name>` for `inspect`; both spellings
      are accepted since other docker subcommands use the other one.
    - `"running"`: the call succeeded and parsed `State.Status` is any
      other non-empty string (`running`, `created`, `restarting`, ...).
      Not exited, so treated as still alive.
    - `"unknown"`: everything else -- a non-zero exit without either
      "no such" phrase (including one with empty stderr, the exact S2
      failure shape), `subprocess.TimeoutExpired`, an `OSError` (e.g. the
      `docker` binary itself is missing), empty stdout, or stdout that does
      not parse as a JSON object with a usable `Status` field.

    Never raises `DockerError` or lets a docker-side exception escape --
    every failure mode above is folded into a `ContainerProbe`, which is
    what lets a caller (`wait_for_llama_health`) tell "docker inspect could
    not answer" apart from "docker inspect answered: the container is
    gone" instead of collapsing both into a single exception.
    """
    try:
        proc = subprocess.run(
            ["docker", "inspect", "-f", "{{json .State}}", name],
            capture_output=True,
            text=True,
            shell=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return ContainerProbe("unknown", f"docker inspect {name!r} timed out after {timeout_s}s")
    except OSError as exc:
        return ContainerProbe("unknown", f"docker inspect {name!r} raised {exc}")

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        if any(marker in stderr.lower() for marker in _ABSENT_STDERR_MARKERS):
            return ContainerProbe("absent", f"{name!r} does not exist: {stderr}")
        return ContainerProbe(
            "unknown", f"docker inspect {name!r} failed (exit {proc.returncode}): {stderr}"
        )

    stdout = (proc.stdout or "").strip()
    if not stdout:
        return ContainerProbe("unknown", f"docker inspect {name!r} returned empty output")
    try:
        state = json.loads(stdout)
    except json.JSONDecodeError as exc:
        return ContainerProbe("unknown", f"docker inspect {name!r} returned unparseable JSON: {exc}")
    if not isinstance(state, dict):
        return ContainerProbe("unknown", f"docker inspect {name!r} State was not a JSON object")

    status = state.get("Status")
    if status in ("exited", "dead"):
        return ContainerProbe("exited", f"{name!r} state is {status!r}", exit_code=state.get("ExitCode"))
    if isinstance(status, str) and status:
        return ContainerProbe("running", f"{name!r} state is {status!r}")
    return ContainerProbe("unknown", f"docker inspect {name!r} returned no usable Status field")


def docker_stop(names: list[str]) -> None:
    proc = subprocess.run(["docker", "stop", *names], capture_output=True, text=True, shell=False)
    if proc.returncode != 0:
        raise DockerError(f"docker stop {names} failed: {proc.stderr.strip()}")


def docker_rm(names: list[str]) -> None:
    proc = subprocess.run(["docker", "rm", *names], capture_output=True, text=True, shell=False)
    if proc.returncode != 0:
        raise DockerError(f"docker rm {names} failed: {proc.stderr.strip()}")


def docker_start(names: list[str]) -> None:
    proc = subprocess.run(["docker", "start", *names], capture_output=True, text=True, shell=False)
    if proc.returncode != 0:
        raise DockerError(f"docker start {names} failed: {proc.stderr.strip()}")


def docker_run_detached(argv: list[str]) -> str:
    proc = subprocess.run(argv, capture_output=True, text=True, shell=False)
    if proc.returncode != 0:
        raise DockerError(f"docker run failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


DOCKER_LOGS_TIMEOUT_S = 60.0


def docker_logs(name: str, *, timeout_s: float = DOCKER_LOGS_TIMEOUT_S) -> tuple[str, str]:
    """`docker logs <name>` as (stdout, stderr), bounded by `timeout_s`.

    The timeout matters on the unknown-state path of `serve`: that path exists because the
    docker CLI is slow or unresponsive, so an unbounded `docker logs` there could hang the
    driver. A timeout raises `DockerError`, which that path already reports as a `[WARN]`.
    """
    try:
        proc = subprocess.run(
            ["docker", "logs", name], capture_output=True, text=True, shell=False, timeout=timeout_s
        )
    except subprocess.TimeoutExpired as exc:
        raise DockerError(f"docker logs {name!r} timed out after {timeout_s}s") from exc
    return proc.stdout, proc.stderr


# ---------------------------------------------------------------------------
# Restore diff
# ---------------------------------------------------------------------------


def _get_dotted(d: dict[str, Any], dotted: str) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def compute_restore_diff(
    snapshot: dict[str, dict[str, Any]], current: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Diff `current` docker-inspect state against a prior `snapshot`.

    Compares RESTORE_DIFF_FIELDS per container. Returns
    `{"empty": bool, "id_changed": bool, "containers": {name: {...}}}`.
    `id_changed` (an Id difference, meaning a recreate) is surfaced
    separately from the rest so callers can treat it as an unconditional
    failure.
    """
    containers: dict[str, Any] = {}
    any_diff = False
    id_changed_any = False

    for name in sorted(snapshot):
        before = snapshot[name]
        after = current.get(name)
        if after is None:
            containers[name] = {
                "id_changed": True,
                "diffs": {"presence": {"before": "present", "after": "missing"}},
            }
            any_diff = True
            id_changed_any = True
            continue

        diffs: dict[str, Any] = {}
        id_changed = False
        for field in RESTORE_DIFF_FIELDS:
            b = _get_dotted(before, field)
            a = _get_dotted(after, field)
            if b != a:
                diffs[field] = {"before": b, "after": a}
                if field == "Id":
                    id_changed = True

        if diffs:
            any_diff = True
            containers[name] = {"id_changed": id_changed, "diffs": diffs}
            if id_changed:
                id_changed_any = True

    return {"empty": not any_diff, "id_changed": id_changed_any, "containers": containers}


# ---------------------------------------------------------------------------
# Served-arm mismatch guard
# ---------------------------------------------------------------------------


def extract_model_path_from_props(props: dict[str, Any]) -> str | None:
    """Best-effort extraction of the loaded model path from GET /props.

    UNVERIFIED: llama-server's /props response is assumed to carry a
    top-level `"model_path"` key (current llama.cpp server README). Falls
    back to `default_generation_settings.model` if present, since some
    server versions instead nest it there.
    """
    if isinstance(props.get("model_path"), str):
        return props["model_path"]
    dgs = props.get("default_generation_settings")
    if isinstance(dgs, dict) and isinstance(dgs.get("model"), str):
        return dgs["model"]
    return None


def check_served_arm(
    running_args: list[str],
    expected_args: list[str],
    props: dict[str, Any],
    resolved_arm: dict[str, Any],
    *,
    image_token: str,
    image_ref: str,
    running_config_image: str | None,
    running_image_id: str | None,
) -> None:
    """Refuse if the running mist-bench-llm container is not `resolved_arm`.

    Compares the container's actual `Args` (from `docker inspect`) against the argv this
    driver would have built for `resolved_arm`, confirms /props' model path ends with the
    arm's gguf filename, and confirms the container's image matches `image_ref`
    (`resolve_image_ref`'s result for `resolved_arm["image"]`): for the `compose:`/literal
    token case, against the container's `Config.Image` (the exact string `docker run` was
    given); for the `snapshot:mist-llm` case, against the container's `Image` id, since
    `image_ref` there already IS the snapshot's recorded Image id, not a repo:tag string.
    An arg/model-path match with the wrong image running (e.g. a stale `mist-bench-llm`
    left over from a prior arm after a failed `unserve`) would otherwise go undetected.
    """
    if running_args != expected_args:
        raise ServedArmMismatchError(
            f"mist-bench-llm Args do not match arm {resolved_arm['id']!r}: "
            f"running={running_args!r} expected={expected_args!r}"
        )
    model_path = extract_model_path_from_props(props)
    gguf = resolved_arm["gguf"]
    if model_path is None or not model_path.replace("\\", "/").endswith(gguf):
        raise ServedArmMismatchError(
            f"/props model path {model_path!r} does not end with arm "
            f"{resolved_arm['id']!r}'s gguf {gguf!r}"
        )
    if image_token == "snapshot:mist-llm":
        if running_image_id != image_ref:
            raise ServedArmMismatchError(
                f"mist-bench-llm Image id {running_image_id!r} does not match the "
                f"snapshot's Image id {image_ref!r} for arm {resolved_arm['id']!r}"
            )
    else:
        if running_config_image != image_ref:
            raise ServedArmMismatchError(
                f"mist-bench-llm Config.Image {running_config_image!r} does not match "
                f"resolved image_ref {image_ref!r} for arm {resolved_arm['id']!r}"
            )


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def http_get_json(url: str, *, timeout: float = 5.0) -> dict[str, Any]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


UNKNOWN_STATE_LIMIT_S = 180.0
"""Default `unknown_limit_s` for `wait_for_llama_health`.

How long a container's docker state may stay unresolved -- repeated
`probe_container_state` calls returning `"unknown"`, with no known reading
in between -- before this driver gives up trying to tell "healthy but
slow" apart from "something is wrong" and raises
`ContainerStateUnknownError` rather than silently waiting out the full
`--timeout` (900s by default). Must be >= 120s: the S2 incident's slow
`docker inspect` episode was resolved within well under a minute once
identified, so a shorter limit risks flagging an ordinary transient CLI/API
slowdown as unknown-for-too-long and printing a false failure exactly like
the one this fix responds to.
"""

_UNKNOWN_WARN_INTERVAL_S = 30.0
"""Minimum gap between `[WARN]` prints for a sustained-unknown probe state."""


def wait_for_llama_health(
    base_url: str,
    *,
    timeout: float = 900.0,
    poll_interval: float = 2.0,
    container_name: str | None = None,
    unknown_limit_s: float = UNKNOWN_STATE_LIMIT_S,
) -> None:
    """Poll GET /health until it responds 200, or raise TimeoutError.

    900s default because CPU-MoE loads (c3/c4) are slow.

    If `container_name` is given, `probe_container_state`s it on every poll
    that /health did not answer:

    - `"exited"` or `"absent"` raises `ContainerExitedError` immediately --
      a container that has exited, or been removed, will never answer
      /health, so waiting out the full `timeout` only delays discovering a
      startup failure (e.g. a flag the pinned build rejects) that is
      already final after well under a second.
    - `"running"` means keep waiting.
    - `"unknown"` (docker inspect itself failed, timed out, or returned
      something unparseable -- e.g. a slow docker CLI/daemon under host
      memory pressure, the exact S2 incident) is printed as a `[WARN]`, at
      most once per `_UNKNOWN_WARN_INTERVAL_S`, and the wait continues.
      Time spent unknown, with no known reading in between, is tracked; if
      it exceeds `unknown_limit_s` this raises `ContainerStateUnknownError`
      instead of either silently waiting out `timeout` or (the S2 bug)
      treating the first inspect failure as a positive "it exited" signal.
      That error says nothing about whether the container exited -- it may
      be healthy -- so the caller must not stop or remove it on the
      strength of this error alone.

    Each `probe_container_state` call's own timeout is capped at the time
    remaining before `timeout` elapses, so a single hung `docker inspect`
    cannot itself stall this loop past the overall deadline.
    """
    if unknown_limit_s < 120.0:
        raise ValueError(f"unknown_limit_s must be >= 120, got {unknown_limit_s}")
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    unknown_since: float | None = None
    last_unknown_warn_at: float | None = None
    while time.monotonic() < deadline:
        try:
            http_get_json(base_url.rstrip("/") + "/health", timeout=5.0)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
        if container_name is not None:
            remaining = deadline - time.monotonic()
            probe_timeout = max(0.1, min(CONTAINER_PROBE_TIMEOUT_S, remaining))
            probe = probe_container_state(container_name, timeout_s=probe_timeout)
            now = time.monotonic()
            if probe.state == "exited":
                raise ContainerExitedError(container_name, probe.exit_code)
            if probe.state == "absent":
                raise ContainerExitedError(container_name, None, absent=True)
            if probe.state == "unknown":
                if unknown_since is None:
                    unknown_since = now
                unknown_for = now - unknown_since
                if (
                    last_unknown_warn_at is None
                    or now - last_unknown_warn_at >= _UNKNOWN_WARN_INTERVAL_S
                ):
                    print(f"[WARN] {container_name} state unknown: {probe.detail}")
                    last_unknown_warn_at = now
                if unknown_for > unknown_limit_s:
                    raise ContainerStateUnknownError(container_name, unknown_for)
            else:
                unknown_since = None
        time.sleep(poll_interval)
    raise TimeoutError(f"{base_url}/health did not become healthy within {timeout}s: {last_error}")


def wait_for_llama_props(base_url: str, *, timeout: float = 30.0) -> dict[str, Any]:
    return http_get_json(base_url.rstrip("/") + "/props", timeout=timeout)


def wait_for_container_healthy(name: str, *, timeout: float = 180.0, poll_interval: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    last_status = None
    while time.monotonic() < deadline:
        inspected = docker_inspect([name])
        state = inspected.get(name, {}).get("State", {})
        last_status = state.get("Health", {}).get("Status")
        if last_status == "healthy":
            return
        time.sleep(poll_interval)
    raise TimeoutError(f"{name} did not report State.Health.Status=healthy within {timeout}s (last: {last_status})")


# ---------------------------------------------------------------------------
# GPU sampler (5 Hz nvidia-smi -> vram.csv)
# ---------------------------------------------------------------------------


class GpuSampler:
    """Runs a long-lived `nvidia-smi -lms 200` process, writing rows to a CSV."""

    def __init__(self, csv_path: Path, *, nvidia_smi_bin: str = "nvidia-smi"):
        self._csv_path = csv_path
        self._bin = nvidia_smi_bin
        self._proc: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._file = None
        self._writer = None

    def start(self) -> None:
        # Append, not truncate (finding 1c): a second `run` call on the same arm dir
        # (--rep 2, a later --layout-pass finalist, ...) must add rows to the existing
        # vram.csv, not discard the earlier call's samples. The header is written only
        # once, when the file is new or still empty.
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self._csv_path.exists() or self._csv_path.stat().st_size == 0
        self._file = open(self._csv_path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        if write_header:
            self._writer.writerow(nvidia_smi_probe.VRAM_CSV_HEADER)
            self._file.flush()
        argv = nvidia_smi_probe.build_query_args(nvidia_smi_bin=self._bin)
        self._proc = subprocess.Popen(argv, stdout=subprocess.PIPE, text=True, shell=False)
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            try:
                row = nvidia_smi_probe.parse_csv_line(line)
            except nvidia_smi_probe.NvidiaSmiParseError:
                continue
            self._writer.writerow(
                [time.time(), *[row[key] for key in nvidia_smi_probe.ROW_KEYS]]
            )
            self._file.flush()

    def stop(self) -> None:
        self._stop.set()
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._file is not None:
            self._file.close()


def sample_gpu_rows(seconds: float, *, nvidia_smi_bin: str = "nvidia-smi") -> list[dict[str, Any]]:
    """Bounded-duration sample, used by `vram-step` and `voice`."""
    argv = nvidia_smi_probe.build_query_args(nvidia_smi_bin=nvidia_smi_bin)
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, text=True, shell=False)
    rows: list[dict[str, Any]] = []
    deadline = time.monotonic() + seconds
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            try:
                row = nvidia_smi_probe.parse_csv_line(line)
            except nvidia_smi_probe.NvidiaSmiParseError:
                continue
            rows.append(row)
            if time.monotonic() >= deadline:
                break
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    return rows


# ---------------------------------------------------------------------------
# Results paths
# ---------------------------------------------------------------------------


def run_dir(results_root: Path, run: str) -> Path:
    return results_root / run


def arm_dir(results_root: Path, run: str, arm: str) -> Path:
    return run_dir(results_root, run) / arm


def save_serve_failed_log(
    results_root: Path, run: str, arm_id: str, *, container: str = BENCH_LLM_CONTAINER
) -> Path:
    """`docker logs` the failed `mist-bench-llm` container to a timestamped log.

    Writes `<results>/<run>/<arm>/serve_failed_<UTC>.log` (stdout then
    stderr, matching `unserve`'s server.log shape) and returns its path.
    Called before the container is removed, since without `--rm` its logs
    would otherwise survive on disk until the next `unserve` -- but a
    failed `serve` never reaches `unserve`, so this is the only chance to
    keep them.
    """
    stdout, stderr = docker_logs(container)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_path = arm_dir(results_root, run, arm_id) / f"serve_failed_{stamp}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(f"=== stdout ===\n{stdout}\n=== stderr ===\n{stderr}\n", encoding="utf-8")
    return log_path


RUN_SUITE_ORDER: tuple[str, ...] = ("ttft", "correctness", "harness", "layout")
LAYOUT_COPIED_FILES: tuple[str, ...] = ("calls.jsonl", "graded.jsonl", "manifest.json")


def validate_run_suites(
    arm: dict[str, Any], requested: list[str] | None, layout_dir: Path | None
) -> list[str]:
    """Resolve the suites `run` will execute, refusing any it could not actually run.

    A requested suite that is silently skipped would leave a result directory that looks complete
    but is missing a suite, so every suite named here either runs or the command refuses up front.

    Raises:
        ArmConfigError: A suite is unknown, not declared for this arm, has no harness candidate,
            or is `layout` with no `--layout-dir` configured.
    """
    suites = list(requested) if requested else list(arm["suites"])
    for suite in suites:
        if suite not in RUN_SUITE_ORDER:
            raise ArmConfigError(f"unknown suite {suite!r}; known: {list(RUN_SUITE_ORDER)}")
        if suite not in arm["suites"]:
            raise ArmConfigError(
                f"suite {suite!r} is not declared for arm {arm['id']!r} (declared: {arm['suites']})"
            )
    if "harness" in suites and arm.get("harness") is None:
        raise ArmConfigError(f"arm {arm['id']!r} requests harness but has no harness candidate")
    if "layout" in suites and layout_dir is None:
        raise ArmConfigError(
            "layout suite requested but --layout-dir / MODEL_BENCH_LAYOUT_DIR is not set"
        )
    return [s for s in RUN_SUITE_ORDER if s in suites]


def refuse_if_exists(path: Path, what: str) -> None:
    if path.exists():
        raise SuiteOutputExistsError(
            f"{what} already exists at {path}; pick a new run, layout pass, or rep instead of overwriting"
        )


def suite_output_paths(
    a_dir: Path, suites: list[str], *, rep: int, layout_pass: str
) -> dict[str, Path]:
    """The on-disk path each requested suite in `suites` would write to.

    Computed purely from `suites` (validate_run_suites' result), before any suite has
    run and before any file is opened, so `cmd_run` can existence-check every requested
    suite's path up front (finding 1a) instead of discovering a later suite's path is
    already taken only after an earlier suite in the same call already wrote its output.
    """
    paths: dict[str, Path] = {}
    if "ttft" in suites:
        paths["ttft"] = a_dir / "ttft.jsonl"
    if "correctness" in suites:
        paths["correctness"] = a_dir / f"correctness.r{rep}.jsonl"
    if "harness" in suites:
        paths["harness"] = a_dir / "harness"
    if "layout" in suites:
        paths["layout"] = a_dir / "layout" / layout_pass
    return paths


def suite_output_what(suite: str, path: Path, *, layout_pass: str) -> str:
    """The human-readable 'what' for refuse_if_exists, matching each suite's own message."""
    if suite == "harness":
        return "harness/"
    if suite == "layout":
        return f"layout/{layout_pass}/"
    return path.name


_META_EQUALITY_KEYS: tuple[str, ...] = (
    "arm_config",
    "server_args",
    "image_ref",
    "image_id",
    "params",
    "tuning_label",
    "decision_rules_sha256",
)


def merge_run_meta(existing: dict[str, Any] | None, call: dict[str, Any]) -> dict[str, Any]:
    """Merge one `run` call's results into an arm dir's cumulative meta.json.

    Pure function (no I/O), so it is unit tested directly. `existing` is the meta.json
    already on disk for this arm dir (or None for the first call), read once by the
    caller before this call's suites ran -- never re-read mid-call. `call` describes this
    call in progress: identity fields that must not change between calls on one arm dir
    (`arm_config`, `server_args`, `image_ref`, `image_id`, `params`, `tuning_label`,
    `decision_rules_sha256` -- see `_META_EQUALITY_KEYS`; one arm dir holds exactly one
    configuration, so a changed config is a new run id, not an amendment to this one),
    plus per-call fields that DO vary (`props`, `container_args`, `git`, `started_utc`,
    `finished_utc`) and the results this call has produced so far (`suites`,
    `suites_completed`, `rep`, `layout_pass`, `harness`, `layout_result`, `errors`).

    Raises RunMetaConfigMismatchError, naming the differing keys, if `existing` is not
    None and any equality-checked field differs from `call`'s. The caller must invoke
    this (and see the raise) BEFORE opening any file for writing, so a refused call
    leaves meta.json and vram.csv byte-identical to before the call.
    """
    if existing is not None:
        differing = [k for k in _META_EQUALITY_KEYS if existing.get(k) != call.get(k)]
        if differing:
            raise RunMetaConfigMismatchError(
                f"this call's config differs from the existing meta.json on: {differing}"
            )

    merged: dict[str, Any] = dict(existing) if existing is not None else {}
    for key in ("schema", "run", "arm", *_META_EQUALITY_KEYS):
        merged[key] = call[key]
    merged["git"] = call["git"]
    merged["props"] = call["props"]
    merged["container_args"] = call["container_args"]

    # layout: a dict keyed by pass, not a single last-pass dict -- screen and finalist
    # are separate calls and both must survive in meta.json.
    layout = dict(existing.get("layout") or {}) if existing is not None else {}
    if call.get("layout_pass") and call.get("layout_result") is not None:
        layout[call["layout_pass"]] = call["layout_result"]
    merged["layout"] = layout

    # harness: set by whichever call ran it; a call that did not run harness leaves the
    # existing value (if any) untouched.
    call_harness = call.get("harness")
    merged["harness"] = (
        call_harness
        if call_harness is not None
        else (existing.get("harness") if existing is not None else None)
    )

    existing_completed = set(existing.get("suites_completed", [])) if existing is not None else set()
    call_completed = set(call.get("suites_completed", []))
    merged["suites_completed"] = [s for s in RUN_SUITE_ORDER if s in (existing_completed | call_completed)]

    existing_errors = list(existing.get("errors", [])) if existing is not None else []
    merged["errors"] = existing_errors + list(call.get("errors", []))

    existing_calls = list(existing.get("calls", [])) if existing is not None else []
    call_entry = {
        "started_utc": call.get("started_utc"),
        "finished_utc": call.get("finished_utc"),
        "suites": list(call.get("suites", [])),
        "rep": call.get("rep"),
        "layout_pass": call.get("layout_pass"),
        "props": call["props"],
        "container_args": call["container_args"],
        "errors": list(call.get("errors", [])),
    }
    merged["calls"] = existing_calls + [call_entry]

    if existing is not None and existing.get("started_utc"):
        merged["started_utc"] = min(existing["started_utc"], call["started_utc"])
    else:
        merged["started_utc"] = call["started_utc"]
    existing_finished = existing.get("finished_utc") if existing is not None else None
    call_finished = call.get("finished_utc")
    if existing_finished and call_finished:
        merged["finished_utc"] = max(existing_finished, call_finished)
    else:
        merged["finished_utc"] = call_finished or existing_finished

    return merged


# ---------------------------------------------------------------------------
# Subcommand: plan
# ---------------------------------------------------------------------------


def cmd_plan(args: argparse.Namespace) -> int:
    resolve_host_config(args, enforce_results_root=False)

    try:
        arms_doc = load_arms_doc()
        resolved = resolve_all_arms(arms_doc)
    except ArmConfigError as exc:
        print(f"[FAIL] arms.json: {exc}")
        return 1

    results_root_raw = args.results_root or os.environ.get("MODEL_BENCH_RESULTS_ROOT")
    if results_root_raw:
        try:
            check_results_root_outside_repo(Path(results_root_raw).resolve())
        except ResultsRootError as exc:
            print(f"[FAIL] {exc}")
            return 1

    snapshot_path = REPO_ROOT / "session" / "snapshot.json"
    for arm_id, arm in resolved.items():
        print(f"arm {arm_id} (optional={arm['optional']}, tuning={arm['tuning']})")
        print(f"  suites: {arm['suites']}")
        try:
            image_ref = resolve_image_ref(arm["image"], snapshot_path=snapshot_path)
        except ImageRefError as exc:
            print(f"  [WARN] image ref {arm['image']!r} unresolved: {exc}")
            image_ref = f"<{arm['image']}>"
        params = {p: "<param>" for p in arm["params_required"]}
        try:
            argv = build_docker_run_args(
                arm, arms_doc, image_ref=image_ref, models_dir="<models_dir>", params=params
            )
        except ArmConfigError as exc:
            print(f"[FAIL] arm {arm_id}: {exc}")
            return 1
        print(f"  docker run: {argv}")
        if arm["harness"] is not None:
            tests = resolve_harness_tests(arm["harness"])
            print(f"  harness: candidate={arm['harness']['candidate']} tests={tests}")

    if args.host_checks and not _run_host_checks():
        # Exit non-zero on any [fail]: a scripted gate reads the exit code, and a failed flag
        # check (the class of defect that blocked S2) must not read as success.
        print("[FAIL] plan: one or more host checks failed")
        return 1

    print("plan ok")
    return 0


def _run_host_checks() -> bool:
    """Run the host-only checks, printing ok/fail per check. Returns True only if all pass."""
    all_ok = True
    checks: list[tuple[str, list[str]]] = [
        ("docker version", ["docker", "version"]),
        ("nvidia-smi query", nvidia_smi_probe.build_query_args(interval_ms=1)),
        ("python imports (yaml, openai, httpx)", [sys.executable, "-c", "import yaml, openai, httpx"]),
    ]
    for label, argv in checks:
        if label == "nvidia-smi query":
            # Single-shot: strip -lms continuous mode down to one row via a
            # short timeout instead of letting it stream forever.
            try:
                proc = subprocess.run(argv[:-2], capture_output=True, text=True, timeout=5, shell=False)
                ok = proc.returncode == 0
            except (subprocess.TimeoutExpired, OSError):
                ok = False
        else:
            try:
                proc = subprocess.run(argv, capture_output=True, text=True, timeout=30, shell=False)
                ok = proc.returncode == 0
            except OSError:
                ok = False
        print(f"[{'ok' if ok else 'fail'}] {label}")
        all_ok = all_ok and ok

    return _run_flag_checks() and all_ok


def _run_flag_checks() -> bool:
    """`docker run --rm <pinned image> --help`, then check every arm's argv against it.

    c0-old is skipped (see `arm_targets_pinned_build`): it targets the b8808
    snapshot build, not the pinned b11151 image this check fetches `--help`
    from. A missing snapshot for c0-old therefore never blocks this check.

    Returns True only if `--help` was fetched, parsed to a non-empty flag set,
    and every checked arm's flags are all known.
    """
    try:
        image_ref = parse_compose_image(DEFAULT_COMPOSE_PATH)
    except ImageRefError as exc:
        print(f"[fail] resolve pinned image from compose for --host-checks: {exc}")
        return False

    try:
        proc = subprocess.run(
            ["docker", "run", "--rm", image_ref, "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            shell=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        print(f"[fail] docker run {image_ref} --help: {exc}")
        return False
    if proc.returncode != 0:
        print(f"[fail] docker run {image_ref} --help exited {proc.returncode}: {proc.stderr.strip()}")
        return False

    return check_all_arm_flags(proc.stdout + "\n" + proc.stderr, image_ref)


def check_all_arm_flags(help_text: str, image_ref: str) -> bool:
    """Check every pinned-build arm's argv against parsed `--help` text, printing per arm.

    Split out of `_run_flag_checks` so its verdict is testable without docker. Returns
    False when the help text parses to no flags or any arm uses an unknown flag.
    """
    help_flags = help_flags_probe.parse_help_flags(help_text)
    if not help_flags:
        print(f"[fail] llama-server --help produced no parseable flags (image {image_ref})")
        return False
    print(f"[ok] llama-server --help parsed {len(help_flags)} flags (image {image_ref})")

    arms_doc = load_arms_doc()
    resolved = resolve_all_arms(arms_doc)
    all_known = True
    for arm_id, arm in resolved.items():
        if not arm_targets_pinned_build(arm):
            print(f"[skip] arm {arm_id}: image {arm['image']!r} is not the pinned build")
            continue
        params = {p: "1" for p in arm["params_required"]}
        unknown = check_arm_flags(help_flags, arms_doc, arm, params)
        if unknown:
            print(f"[fail] arm {arm_id}: unknown flags: {unknown}")
            all_known = False
        else:
            print(f"[ok] arm {arm_id}: flags known")
    return all_known


# ---------------------------------------------------------------------------
# Subcommand: selftest
# ---------------------------------------------------------------------------


def cmd_selftest(args: argparse.Namespace) -> int:
    arms_doc = load_arms_doc()
    resolved = resolve_all_arms(arms_doc)

    for arm_id, arm in resolved.items():
        params = {p: "12" for p in arm["params_required"]}
        argv = build_docker_run_args(
            arm, arms_doc, image_ref="dummy:image", models_dir="/tmp/models", params=params
        )
        assert all(isinstance(tok, str) for tok in argv), f"{arm_id}: non-str token in argv"
        if arm["params_required"]:
            try:
                build_docker_run_args(
                    arm, arms_doc, image_ref="dummy:image", models_dir="/tmp/models", params={}
                )
            except MissingParamError:
                pass
            else:
                raise AssertionError(f"{arm_id}: missing required param did not raise")

    # Restore diff: empty.
    same = {"Id": "sha256:a", "Image": "img", "Config": {"Cmd": [], "Env": []}, "HostConfig": {}}
    diff = compute_restore_diff({"mist-llm": same}, {"mist-llm": dict(same)})
    assert diff["empty"] is True

    # Restore diff: non-empty, Env changed.
    changed = {
        "Id": "sha256:a",
        "Image": "img",
        "Config": {"Cmd": [], "Env": ["A=1"]},
        "HostConfig": {},
    }
    diff = compute_restore_diff({"mist-llm": same}, {"mist-llm": changed})
    assert diff["empty"] is False
    assert diff["id_changed"] is False

    # Restore diff: Id changed (recreate).
    recreated = dict(same)
    recreated["Id"] = "sha256:b"
    diff = compute_restore_diff({"mist-llm": same}, {"mist-llm": recreated})
    assert diff["id_changed"] is True

    # Compose image parser + refusal.
    with_digest = FIXTURES_DIR / "compose_with_digest.yml"
    without_digest = FIXTURES_DIR / "compose_missing_digest.yml"
    ref = parse_compose_image(with_digest)
    assert "@sha256:" in ref
    try:
        parse_compose_image(without_digest)
    except ImageRefError:
        pass
    else:
        raise AssertionError("compose file missing a digest did not raise ImageRefError")

    # Results-root refusal.
    try:
        check_results_root_outside_repo(REPO_ROOT / "results")
    except ResultsRootError:
        pass
    else:
        raise AssertionError("results-root inside repo did not raise")
    check_results_root_outside_repo(REPO_ROOT.parent / "mist-model-bench-results")

    _selftest_probes()

    print("selftest ok")
    return 0


def _selftest_probes() -> None:
    from .probes import correctness as correctness_probe
    from .probes import ttft as ttft_probe

    sse_path = FIXTURES_DIR / "sse_stream.txt"
    lines = sse_path.read_text(encoding="utf-8").splitlines()
    events = list(ttft_probe.iter_sse_events(lines))
    row = ttft_probe.extract_ttft_row(events, first_content_t=1.5, start_t=1.0)
    assert row["ttft_ms"] == 500.0
    assert row["predicted_ms"] is not None

    csv_path = FIXTURES_DIR / "nvidia_smi_sample.csv"
    csv_lines = csv_path.read_text(encoding="utf-8").splitlines()
    parsed = [nvidia_smi_probe.parse_csv_line(line) for line in csv_lines]
    assert any(row["throttle_reasons"] is None for row in parsed), "no [N/A] row parsed to None"
    assert any(isinstance(row["memory_used_mib"], int) for row in parsed)

    voice_path = FIXTURES_DIR / "voice_probe_output.json"
    voice_result = json.loads(voice_path.read_text(encoding="utf-8"))
    assert set(voice_result) == {
        "max_memory_reserved_mib",
        "max_memory_allocated_mib",
        "stt_ok",
        "tts_ok",
        "error",
    }

    correctness_path = FIXTURES_DIR / "correctness_response.json"
    correctness_response = json.loads(correctness_path.read_text(encoding="utf-8"))
    correctness_row = correctness_probe.parse_correctness_response("p01", correctness_response)
    assert correctness_row["tokens_sha256"]
    assert correctness_row["error"] is None


# ---------------------------------------------------------------------------
# Subcommand: snapshot
# ---------------------------------------------------------------------------


def cmd_snapshot(args: argparse.Namespace) -> int:
    config = resolve_host_config(args)
    if config.results_root is None:
        print("[FAIL] --results-root / MODEL_BENCH_RESULTS_ROOT is required")
        return 1
    snap_path = run_dir(config.results_root, args.run) / "session" / "snapshot.json"
    refuse_if_exists(snap_path, "session/snapshot.json")

    inspected = docker_inspect(list(SNAPSHOT_CONTAINERS))
    doc: dict[str, Any] = {}
    for name, entry in inspected.items():
        doc[name] = {
            "Id": entry.get("Id"),
            "Image": entry.get("Image"),
            "Config": {
                "Cmd": entry.get("Config", {}).get("Cmd"),
                "Env": entry.get("Config", {}).get("Env"),
                "Entrypoint": entry.get("Config", {}).get("Entrypoint"),
            },
            "HostConfig": entry.get("HostConfig"),
            "State": {
                "Running": entry.get("State", {}).get("Running"),
                "Health": entry.get("State", {}).get("Health"),
            },
        }

    snap_path.parent.mkdir(parents=True, exist_ok=True)
    snap_path.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    print(f"snapshot written to {snap_path}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: stop
# ---------------------------------------------------------------------------


def cmd_stop(args: argparse.Namespace) -> int:
    config = resolve_host_config(args)
    if config.results_root is None:
        print("[FAIL] --results-root / MODEL_BENCH_RESULTS_ROOT is required")
        return 1
    snap_path = run_dir(config.results_root, args.run) / "session" / "snapshot.json"
    if not snap_path.exists():
        print(f"[FAIL] no snapshot at {snap_path}; run `snapshot` first")
        return 1

    names = ["mist-backend", "mist-llm"]
    if args.with_neo4j:
        names.append("mist-neo4j")
    docker_stop(names)
    print(f"stopped: {names}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: restore
# ---------------------------------------------------------------------------


def cmd_restore(args: argparse.Namespace) -> int:
    config = resolve_host_config(args)
    if config.results_root is None:
        print("[FAIL] --results-root / MODEL_BENCH_RESULTS_ROOT is required")
        return 1
    session_dir = run_dir(config.results_root, args.run) / "session"
    snap_path = session_dir / "snapshot.json"
    if not snap_path.exists():
        print(f"[FAIL] no snapshot at {snap_path}; run `snapshot` first")
        return 1

    probe = probe_container_state(BENCH_LLM_CONTAINER)
    if probe.state in ("running", "exited"):
        print(f"[FAIL] {BENCH_LLM_CONTAINER} exists (running or exited); run `unserve` first")
        return 1
    if probe.state == "unknown":
        print(
            f"[FAIL] could not determine whether {BENCH_LLM_CONTAINER} exists; retry "
            f"({probe.detail})"
        )
        return 1

    docker_start(list(SNAPSHOT_CONTAINERS))

    try:
        wait_for_llama_health(f"http://{BENCH_LLM_HOST_BIND}:{BENCH_LLM_PORT}", timeout=180.0)
        wait_for_llama_props(f"http://{BENCH_LLM_HOST_BIND}:{BENCH_LLM_PORT}")
        wait_for_container_healthy("mist-backend", timeout=180.0)
    except (TimeoutError, urllib.error.URLError) as exc:
        print(f"[FAIL] restore health checks: {exc}")
        return 1

    snapshot = json.loads(snap_path.read_text(encoding="utf-8"))
    current = docker_inspect(list(SNAPSHOT_CONTAINERS))
    diff = compute_restore_diff(snapshot, current)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    diff_path = session_dir / f"restore_diff_{stamp}.json"
    diff_path.write_text(json.dumps(diff, indent=2, sort_keys=True), encoding="utf-8")

    if diff["empty"]:
        print("restore diff: empty")
        return 0
    print(f"restore diff: {json.dumps(diff, indent=2)}")
    return 1


# ---------------------------------------------------------------------------
# Subcommand: serve
# ---------------------------------------------------------------------------


def _parse_param_args(pairs: list[str]) -> dict[str, str]:
    params: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ArmConfigError(f"--param must be key=value, got {pair!r}")
        key, value = pair.split("=", 1)
        params[key] = value
    return params


def cmd_serve(args: argparse.Namespace) -> int:
    config = resolve_host_config(args)
    if config.models_dir is None:
        print("[FAIL] --models-dir / MODELS_DIR is required")
        return 1
    if config.results_root is None:
        print("[FAIL] --results-root / MODEL_BENCH_RESULTS_ROOT is required")
        return 1

    arms_doc = load_arms_doc()
    try:
        arm = resolve_arm(arms_doc, args.arm)
    except ArmConfigError as exc:
        print(f"[FAIL] {exc}")
        return 1

    if docker_is_running("mist-llm"):
        print("[FAIL] mist-llm is running; run `stop` first")
        return 1
    if arm["stop_neo4j"] and docker_is_running("mist-neo4j"):
        print(f"[FAIL] arm {arm['id']} requires mist-neo4j stopped, but it is running")
        return 1
    # Without --rm (build_docker_run_args), a container left over from a
    # prior arm -- running or exited, e.g. after a failed serve this
    # function itself did not clean up -- keeps `docker run --name` from
    # succeeding. Refuse up front with a clear message rather than let that
    # surface as an opaque DockerError.
    probe = probe_container_state(BENCH_LLM_CONTAINER)
    if probe.state in ("running", "exited"):
        print(f"[FAIL] {BENCH_LLM_CONTAINER} already exists (running or exited); run `unserve` first")
        return 1
    if probe.state == "unknown":
        print(
            f"[FAIL] could not determine whether {BENCH_LLM_CONTAINER} exists; retry "
            f"({probe.detail})"
        )
        return 1

    params = _parse_param_args(args.param)
    snapshot_path = run_dir(config.results_root, args.run) / "session" / "snapshot.json"
    try:
        image_ref = resolve_image_ref(arm["image"], snapshot_path=snapshot_path)
        argv = build_docker_run_args(
            arm, arms_doc, image_ref=image_ref, models_dir=str(config.models_dir), params=params
        )
    except (MissingParamError, ImageRefError) as exc:
        print(f"[FAIL] {exc}")
        return 1

    docker_run_detached(argv)
    timeout = args.timeout if args.timeout else 900.0
    try:
        wait_for_llama_health(
            f"http://{BENCH_LLM_HOST_BIND}:{BENCH_LLM_PORT}",
            timeout=timeout,
            container_name=BENCH_LLM_CONTAINER,
        )
    except ContainerExitedError as exc:
        log_path = save_serve_failed_log(config.results_root, args.run, arm["id"])
        print(f"[FAIL] {exc}; log saved to {log_path}")
        # `absent` means docker inspect already reports no such container --
        # nothing left to remove, and a docker_rm here would itself fail.
        if not exc.absent:
            docker_rm([BENCH_LLM_CONTAINER])
        return 1
    except ContainerStateUnknownError as exc:
        # Unlike ContainerExitedError, this is NOT a positive signal that
        # anything is wrong -- mist-bench-llm may be loading normally (the
        # exact S2 false positive: a slow docker CLI under low host memory
        # was misread as the container having exited). Save what evidence
        # we can, but never stop or remove a container we could not
        # actually confirm is unhealthy.
        print(f"[FAIL] {exc}")
        try:
            log_path = save_serve_failed_log(config.results_root, args.run, arm["id"])
            print(f"log saved to {log_path}")
        except (DockerError, OSError) as log_exc:
            print(f"[WARN] could not save docker logs: {log_exc}")
        print(
            f"{BENCH_LLM_CONTAINER} was NOT stopped or removed -- its state could not be "
            f"confirmed and it may still be loading normally. Check it yourself with "
            f"`docker ps -a --filter name={BENCH_LLM_CONTAINER}`, then run `unserve` if "
            f"you want it gone."
        )
        return 1
    except TimeoutError:
        # The container is still running (ContainerExitedError above is
        # what fires when it is not) -- current behaviour is to propagate
        # this uncaught (main() only catches BenchHostError), but the logs
        # are worth keeping either way.
        save_serve_failed_log(config.results_root, args.run, arm["id"])
        raise

    print(f"serving arm {arm['id']} as {BENCH_LLM_CONTAINER}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: unserve
# ---------------------------------------------------------------------------


def cmd_unserve(args: argparse.Namespace) -> int:
    config = resolve_host_config(args)
    if config.results_root is None:
        print("[FAIL] --results-root / MODEL_BENCH_RESULTS_ROOT is required")
        return 1
    stdout, stderr = docker_logs(BENCH_LLM_CONTAINER)
    log_path = arm_dir(config.results_root, args.run, args.arm) / "server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(f"=== stdout ===\n{stdout}\n=== stderr ===\n{stderr}\n", encoding="utf-8")
    # Handles a running container, one that already exited on its own (e.g.
    # left behind by a `serve` failure this driver's caller did not clean
    # up), and one whose state could not be confirmed (a slow or failing
    # docker inspect) -- docker_stop is skipped only when the container is
    # confirmed already exited; an `unknown` probe still attempts
    # docker_stop rather than silently skipping it (that would risk
    # leaving an actually-running container behind), so any real failure
    # there surfaces as this call's own DockerError. docker_rm always runs:
    # without --rm (build_docker_run_args), nothing else removes it.
    probe = probe_container_state(BENCH_LLM_CONTAINER)
    if probe.state in ("running", "unknown"):
        docker_stop([BENCH_LLM_CONTAINER])
    docker_rm([BENCH_LLM_CONTAINER])
    print(f"unserved; log at {log_path}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: run
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    from .probes import correctness as correctness_probe
    from .probes import ttft as ttft_probe

    config = resolve_host_config(args)
    if config.results_root is None:
        print("[FAIL] --results-root / MODEL_BENCH_RESULTS_ROOT is required")
        return 1

    arms_doc = load_arms_doc()
    try:
        arm = resolve_arm(arms_doc, args.arm)
    except ArmConfigError as exc:
        print(f"[FAIL] {exc}")
        return 1

    params = _parse_param_args(args.param)
    snapshot_path = run_dir(config.results_root, args.run) / "session" / "snapshot.json"
    try:
        image_ref = resolve_image_ref(arm["image"], snapshot_path=snapshot_path)
        expected_args = build_model_args(arm, arms_doc, params)
    except (ImageRefError, ArmConfigError) as exc:
        print(f"[FAIL] {exc}")
        return 1

    # Every check below is preflight: nothing is opened for writing until all of them
    # pass (finding 1a). A refused call -- wrong served arm, dirty decision_rules.json, an
    # invalid/undeclared suite, an existing suite output, or a config mismatch against
    # this arm dir's existing meta.json -- must leave meta.json and vram.csv untouched.

    # (a) served-arm check, including the image (finding 6).
    inspected = docker_inspect([BENCH_LLM_CONTAINER])
    if BENCH_LLM_CONTAINER not in inspected:
        print(f"[FAIL] {BENCH_LLM_CONTAINER} is not running; run `serve {arm['id']}` first")
        return 1
    running_args = inspected[BENCH_LLM_CONTAINER].get("Args", [])
    running_image_id = inspected[BENCH_LLM_CONTAINER].get("Image")
    running_config_image = inspected[BENCH_LLM_CONTAINER].get("Config", {}).get("Image")
    base_url = f"http://{BENCH_LLM_HOST_BIND}:{BENCH_LLM_PORT}"
    try:
        props = wait_for_llama_props(base_url)
        check_served_arm(
            running_args,
            expected_args,
            props,
            arm,
            image_token=arm["image"],
            image_ref=image_ref,
            running_config_image=running_config_image,
            running_image_id=running_image_id,
        )
    except (ServedArmMismatchError, urllib.error.URLError) as exc:
        print(f"[FAIL] served-arm check: {exc}")
        return 1

    # (a) decision_rules.json.
    if not DECISION_RULES_PATH.exists():
        print(f"[FAIL] {DECISION_RULES_PATH} does not exist")
        return 1
    try:
        decision_rules_sha256 = check_decision_rules_clean(DECISION_RULES_PATH, REPO_ROOT)
    except DecisionRulesError as exc:
        print(f"[FAIL] {exc}")
        return 1

    # (a) suite validation.
    a_dir = arm_dir(config.results_root, args.run, arm["id"])
    try:
        suites = validate_run_suites(arm, args.suites, config.layout_dir)
    except ArmConfigError as exc:
        print(f"[FAIL] {exc}")
        return 1

    rep = args.rep if args.rep is not None else 1
    layout_pass = args.layout_pass or "screen"

    # (a) existence check for every requested suite's output path, computed up front so
    # a later suite in this same call cannot be found "taken" only after an earlier suite
    # in the call already wrote its output.
    try:
        for suite, path in suite_output_paths(a_dir, suites, rep=rep, layout_pass=layout_pass).items():
            refuse_if_exists(path, suite_output_what(suite, path, layout_pass=layout_pass))
    except SuiteOutputExistsError as exc:
        print(f"[FAIL] {exc}")
        return 1

    # (b) meta.json is cumulative: this call's identity config must match whatever is
    # already on disk for this arm dir. Read once, here; never re-read mid-call --
    # merge_run_meta is always called against this same snapshot plus the growing `call`
    # dict below, so every write_meta() recomputes the full cumulative document fresh.
    meta_path = a_dir / "meta.json"
    existing_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None

    started_utc = datetime.now(UTC).isoformat()
    call: dict[str, Any] = {
        "schema": 1,
        "run": args.run,
        "arm": arm["id"],
        "arm_config": arm,
        "image_ref": image_ref,
        "image_id": running_image_id,
        "server_args": expected_args,
        "params": params,
        "tuning_label": args.tuning_label,
        "decision_rules_sha256": decision_rules_sha256,
        "git": collect_git_state(REPO_ROOT, config.layout_dir),
        "container_args": running_args,
        "props": props,
        "started_utc": started_utc,
        "finished_utc": started_utc,
        "suites": suites,
        "suites_completed": [],
        "rep": rep if "correctness" in suites else None,
        "layout_pass": layout_pass if "layout" in suites else None,
        "harness": None,
        "layout_result": None,
        "errors": [],
    }
    try:
        merge_run_meta(existing_meta, call)  # validated here; the result is discarded
    except RunMetaConfigMismatchError as exc:
        print(f"[FAIL] {exc}")
        return 1

    # Every preflight check above has passed. Only now does anything get written.
    a_dir.mkdir(parents=True, exist_ok=True)

    def write_meta() -> None:
        call["finished_utc"] = datetime.now(UTC).isoformat()
        merged = merge_run_meta(existing_meta, call)
        meta_path.write_text(json.dumps(merged, indent=2, sort_keys=True, default=str), encoding="utf-8")

    sampler = GpuSampler(a_dir / "vram.csv")
    sampler.start()
    try:
        if "ttft" in suites:
            ttft_path = a_dir / "ttft.jsonl"
            props_now = wait_for_llama_props(base_url)
            n_ctx = props_now.get("default_generation_settings", {}).get("n_ctx", 32768)
            rows = ttft_probe.run_ttft_probe(base_url, n_ctx=n_ctx)
            with open(ttft_path, "w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row) + "\n")
            call["suites_completed"].append("ttft")
            write_meta()

        if "correctness" in suites:
            correctness_path = a_dir / f"correctness.r{rep}.jsonl"
            rows = correctness_probe.run_correctness_probe(base_url)
            with open(correctness_path, "w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row) + "\n")
            call["suites_completed"].append("correctness")
            write_meta()

        if "harness" in suites:
            harness_dir = a_dir / "harness"
            tests = resolve_harness_tests(arm["harness"])
            harness_argv = [
                sys.executable,
                "-m",
                "scripts.eval_harness.run",
                "--external",
                "--models",
                arm["harness"]["candidate"],
                "--tests",
                ",".join(tests),
                "--iterations",
                str(arm["harness"]["iterations"]),
                "--results-dir",
                str(harness_dir),
                "--run-name",
                "harness",
            ]
            proc = subprocess.run(harness_argv, cwd=REPO_ROOT, shell=False)
            if proc.returncode != 0:
                call["errors"].append(f"harness exited {proc.returncode}")
            call["harness"] = {
                "candidate": arm["harness"]["candidate"],
                "tests": tests,
                "iterations": arm["harness"]["iterations"],
            }
            # Exit 0 is necessary, not sufficient: run_candidate() logs server errors and
            # returns normally (scripts/eval_harness/run.py, the except blocks in run_candidate),
            # so analyse.py must still check the JSONL has every case x iteration.
            if proc.returncode == 0:
                call["suites_completed"].append("harness")
            write_meta()

        if "layout" in suites:
            assert config.layout_dir is not None  # validate_run_suites refused otherwise
            layout_out_dir = a_dir / "layout" / layout_pass
            thinking_mode = "off" if arm["thinking"] is None else arm["thinking"].get("mode", "off")
            max_tokens = layout_max_tokens(arm["thinking"])
            layouts_per_size = LAYOUT_LAYOUTS_PER_SIZE[layout_pass]
            result_id = f"mist-model-bench-{args.run}-{arm['id']}-{layout_pass}"
            run_host_argv = [
                sys.executable,
                "run_host.py",
                "all",
                "--out",
                f"results/{result_id}",
                "--phases",
                "accuracy",
                "--reps",
                "R1",
                "--layouts-per-size",
                str(layouts_per_size),
                "--thinking",
                thinking_mode,
                "--max-tokens",
                str(max_tokens),
            ]
            analyse_argv = [
                sys.executable,
                "analyse.py",
                "--results",
                f"results/{result_id}",
            ]
            call["layout_result"] = {
                "layouts_per_size": layouts_per_size,
                "thinking": thinking_mode,
                "max_tokens": max_tokens,
            }
            layout_ok = True
            for step_argv in (run_host_argv, analyse_argv):
                proc = subprocess.run(step_argv, cwd=config.layout_dir, shell=False)
                if proc.returncode != 0:
                    call["errors"].append(f"layout: {step_argv[1]} exited {proc.returncode}")
                    layout_ok = False
                    break
            if layout_ok:
                src_dir = config.layout_dir / "results" / result_id
                missing = [n for n in LAYOUT_COPIED_FILES if not (src_dir / n).is_file()]
                if missing:
                    # finding 5: a path relative to --layout-dir, never the host-absolute
                    # src_dir, so a public output that later copies meta.errors verbatim
                    # (which analyse.py must not do either -- see build_summary/
                    # compute_coverage) cannot leak this machine's directory layout.
                    rel_src = src_dir.relative_to(config.layout_dir)
                    call["errors"].append(f"layout: missing {missing} in {rel_src}")
                else:
                    layout_out_dir.mkdir(parents=True, exist_ok=True)
                    for name in LAYOUT_COPIED_FILES:
                        (layout_out_dir / name).write_bytes((src_dir / name).read_bytes())
                    call["suites_completed"].append("layout")
            write_meta()
    finally:
        sampler.stop()
        write_meta()

    print(f"run complete for arm {arm['id']}: suites_completed (this call)={call['suites_completed']}")
    if call["errors"]:
        print(f"[FAIL] run recorded errors (this call): {call['errors']}")
        return 1
    return 0


# ---------------------------------------------------------------------------
# Subcommand: vram-step
# ---------------------------------------------------------------------------


def cmd_vram_step(args: argparse.Namespace) -> int:
    config = resolve_host_config(args)
    if config.results_root is None:
        print("[FAIL] --results-root / MODEL_BENCH_RESULTS_ROOT is required")
        return 1
    session_dir = run_dir(config.results_root, args.run) / "session"
    session_dir.mkdir(parents=True, exist_ok=True)
    steps_path = session_dir / "vram_steps.json"

    rows = sample_gpu_rows(args.seconds)
    summary = nvidia_smi_probe.summarize_rows(rows)
    summary["label"] = args.label
    summary["t_utc"] = datetime.now(UTC).isoformat()

    doc: dict[str, Any] = {"steps": [], "voice_probe": None}
    if steps_path.exists():
        doc = json.loads(steps_path.read_text(encoding="utf-8"))
    doc["steps"] = [s for s in doc.get("steps", []) if s.get("label") != args.label]
    doc["steps"].append(summary)
    steps_path.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    print(f"vram-step {args.label}: {summary}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: voice
# ---------------------------------------------------------------------------


def cmd_voice(args: argparse.Namespace) -> int:
    config = resolve_host_config(args)
    if config.results_root is None:
        print("[FAIL] --results-root / MODEL_BENCH_RESULTS_ROOT is required")
        return 1
    session_dir = run_dir(config.results_root, args.run) / "session"
    session_dir.mkdir(parents=True, exist_ok=True)
    steps_path = session_dir / "vram_steps.json"

    probe_src = (PACKAGE_DIR / "probes" / "voice_vram.py").read_text(encoding="utf-8")

    rows: list[dict[str, Any]] = []
    stop_event = threading.Event()

    def sampler_thread() -> None:
        argv = nvidia_smi_probe.build_query_args()
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, text=True, shell=False)
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                if stop_event.is_set():
                    break
                try:
                    rows.append(nvidia_smi_probe.parse_csv_line(line))
                except nvidia_smi_probe.NvidiaSmiParseError:
                    continue
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    thread = threading.Thread(target=sampler_thread, daemon=True)
    thread.start()
    try:
        proc = subprocess.run(
            ["docker", "exec", "-i", "mist-backend", "python", "-"],
            input=probe_src,
            capture_output=True,
            text=True,
            shell=False,
        )
    finally:
        stop_event.set()
        thread.join(timeout=5)

    if proc.returncode != 0:
        print(f"[FAIL] voice probe exited {proc.returncode}: {proc.stderr}")
        return 1

    last_line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "{}"
    voice_result = json.loads(last_line)

    summary = nvidia_smi_probe.summarize_rows(rows)
    summary["label"] = "voice_peak"
    summary["t_utc"] = datetime.now(UTC).isoformat()

    doc: dict[str, Any] = {"steps": [], "voice_probe": None}
    if steps_path.exists():
        doc = json.loads(steps_path.read_text(encoding="utf-8"))
    doc["steps"] = [s for s in doc.get("steps", []) if s.get("label") != "voice_peak"]
    doc["steps"].append(summary)
    doc["voice_probe"] = voice_result
    steps_path.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")

    print(f"voice probe: {voice_result}")
    return 0


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


def _add_common_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--layout-dir", default=None)
    parser.add_argument("--results-root", default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m scripts.model_bench.bench_host")
    sub = parser.add_subparsers(dest="command", required=True)

    p_plan = sub.add_parser("plan", help="Dry-run validation of arms.json; no docker.")
    _add_common_config_args(p_plan)
    p_plan.add_argument("--host-checks", action="store_true")
    p_plan.set_defaults(func=cmd_plan)

    p_selftest = sub.add_parser("selftest", help="No docker, no network; exercises parsers and guards.")
    p_selftest.set_defaults(func=cmd_selftest)

    p_snapshot = sub.add_parser("snapshot", help="docker inspect the 3 MIST containers.")
    _add_common_config_args(p_snapshot)
    p_snapshot.add_argument("--run", required=True)
    p_snapshot.set_defaults(func=cmd_snapshot)

    p_stop = sub.add_parser("stop", help="Stop mist-backend/mist-llm (and optionally mist-neo4j).")
    _add_common_config_args(p_stop)
    p_stop.add_argument("--run", required=True)
    p_stop.add_argument("--with-neo4j", action="store_true")
    p_stop.set_defaults(func=cmd_stop)

    p_restore = sub.add_parser("restore", help="Start the 3 containers back up and diff against the snapshot.")
    _add_common_config_args(p_restore)
    p_restore.add_argument("--run", required=True)
    p_restore.set_defaults(func=cmd_restore)

    p_serve = sub.add_parser("serve", help="Serve one arm as mist-bench-llm.")
    _add_common_config_args(p_serve)
    p_serve.add_argument("arm")
    p_serve.add_argument("--run", required=True)
    p_serve.add_argument("--param", action="append", default=[])
    p_serve.add_argument("--timeout", type=float, default=None)
    p_serve.set_defaults(func=cmd_serve)

    p_unserve = sub.add_parser("unserve", help="Save server.log and stop mist-bench-llm.")
    _add_common_config_args(p_unserve)
    p_unserve.add_argument("--run", required=True)
    p_unserve.add_argument("--arm", required=True)
    p_unserve.set_defaults(func=cmd_unserve)

    p_run = sub.add_parser("run", help="Run benchmark suites against the currently served arm.")
    _add_common_config_args(p_run)
    p_run.add_argument("arm")
    p_run.add_argument("--run", required=True)
    p_run.add_argument("--suites", nargs="*", default=None)
    p_run.add_argument("--layout-pass", choices=("screen", "finalist"), default=None)
    p_run.add_argument("--tuning-label", default=None)
    p_run.add_argument("--rep", type=int, default=None)
    p_run.add_argument("--param", action="append", default=[])
    p_run.set_defaults(func=cmd_run)

    p_vram_step = sub.add_parser("vram-step", help="Sample GPU memory for a labeled step.")
    _add_common_config_args(p_vram_step)
    p_vram_step.add_argument("--run", required=True)
    p_vram_step.add_argument("--label", required=True)
    p_vram_step.add_argument("--seconds", type=float, default=10.0)
    p_vram_step.set_defaults(func=cmd_vram_step)

    p_voice = sub.add_parser("voice", help="Run the voice VRAM probe inside mist-backend.")
    _add_common_config_args(p_voice)
    p_voice.add_argument("--run", required=True)
    p_voice.set_defaults(func=cmd_voice)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except BenchHostError as exc:
        print(f"[FAIL] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
