"""Container jobs: the job model, the `Executor` protocol and the local Docker executor.

Every heavy stage runs as a job in its family's image, so the host needs only the standard
library. A job names its env vars, never their values: `docker run -e NAME` copies the value from
the host environment, so a key never appears in an argv, a log or a record. The container log goes
to `runs/<job_id>/log.txt` and is never printed, because a stage's log can hold transcripts.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Protocol

from .errors import InputError, JobError, RootError, VoiceDataError
from .records import JobInput, JobRecord, read_json, sha256_file, write_json
from .root import Root

JOB_SCHEMA = 1
FETCH_MODULE = "scripts.voice_data.stages.fetch"
FAMILIES_PATH = Path(__file__).with_name("families.json")
REMOTE_MODULE = "scripts.voice_data.remote"
# One `inspect` line per container: its name, then its device requests (`null` when it has none).
INSPECT_FORMAT = "{{.Name}} {{json .HostConfig.DeviceRequests}}"
# The preflight's nvidia-smi arguments; nvidia-smi itself is the entrypoint (see `preflight`).
NVIDIA_SMI_QUERY = ("--query-gpu=memory.free", "--format=csv,noheader,nounits")
# Comma-separated container names the preflight lets keep a GPU, such as the daily backend while
# a gaming-mode window has moved its speech engines to CPU but left its CUDA context open.
GPU_PEERS_ENV = "MIST_VOICE_GPU_PEERS"
# Optional per-family resource caps in `families.json`: key, `docker run` flag, value suffix.
# They keep a job inside the RAM and cores a shared host can spare beside the daily stack. A
# `cpus` budget also pins the job's CPU affinity (`--cpuset-cpus`, see `cpuset_for`).
FAMILY_CAPS = (
    ("shm_gb", "--shm-size", "g"),
    ("memory_gb", "--memory", "g"),
    ("cpus", "--cpus", ""),
)


class PreflightError(VoiceDataError):
    """The GPU preflight refused a job: a container holds a GPU, or VRAM is short (exit 2)."""


@dataclass(frozen=True, slots=True)
class JobSpec:
    """One container job, with its paths relative to the root (fields of `job.json`)."""

    job_id: str
    family: str
    image: str
    module: str
    args: tuple[str, ...]
    inputs: tuple[JobInput, ...]
    outputs: str
    gpu: bool
    network: bool
    env_names: tuple[str, ...] = ()
    optional_env_names: tuple[str, ...] = ()
    est_hours: float = 0.0
    # The VRAM this job needs, when it differs from the family's estimate: a probe's measured
    # batch, or a training run sized to fit beside a peer. `None` uses the family's `vram_gb`.
    vram_gb: float | None = None
    # Seconds Docker waits between SIGTERM and SIGKILL (`--stop-timeout`), for a job whose
    # SIGTERM save outlasts Docker's default 10 s. `None` keeps Docker's default.
    stop_timeout_s: int | None = None

    def record(self) -> JobRecord:
        """Return the job as its `runs/<job_id>/job.json` record."""
        values = {f.name: getattr(self, f.name) for f in fields(self)}
        return JobRecord(schema=JOB_SCHEMA, **values)


@dataclass(frozen=True, slots=True)
class JobResult:
    """A finished job: its exit code, its log's root-relative path and its verified outputs."""

    job_id: str
    exit_code: int
    log: str
    outputs: tuple[JobInput, ...]


class Executor(Protocol):
    """Runs a job somewhere: on this host's Docker, or on a rented pod (T13)."""

    def run(self, job: JobSpec) -> JobResult:
        """Run the job and verify its outputs; raise `JobError` (exit 4) on failure."""
        ...


class DockerOps(Protocol):
    """The docker CLI calls the executor and `images build` make; tests pass a fake."""

    def capture(self, argv: Sequence[str]) -> str:
        """Run `argv` and return its stdout; a non-zero exit raises `JobError`."""
        ...

    def check(self, argv: Sequence[str]) -> None:
        """Run `argv` with the console's output; a non-zero exit raises `JobError`."""
        ...

    def run_logged(self, argv: Sequence[str], log: Path) -> int:
        """Run `argv` with stdout and stderr written to `log`, and return its exit code."""
        ...


class HostDocker:
    """The docker CLI on this host, called with argv lists and never a shell."""

    def capture(self, argv: Sequence[str]) -> str:
        """Run and capture stdout; stderr is dropped from the error, as it can echo a path."""
        proc = subprocess.run(list(argv), capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise JobError(f"{' '.join(argv[:2])} exited {proc.returncode}")
        return proc.stdout

    def check(self, argv: Sequence[str]) -> None:
        """Run with the console's output, for image builds of public code."""
        code = subprocess.run(list(argv), check=False).returncode
        if code != 0:
            raise JobError(f"{' '.join(argv[:2])} exited {code}")

    def run_logged(self, argv: Sequence[str], log: Path) -> int:
        """Run with both streams in the log file, so nothing a stage prints reaches the console."""
        with log.open("wb") as handle:
            proc = subprocess.run(list(argv), stdout=handle, stderr=subprocess.STDOUT, check=False)
        return proc.returncode


def cpuset_for(cpus: float) -> str:
    """Return the `--cpuset-cpus` range that matches a family's `cpus` budget.

    `--cpus` is a CFS quota: it limits CPU time but leaves `os.sched_getaffinity(0)` at every
    core of the Docker VM, so a process pool or thread count sized from the affinity starts one
    worker per VM core. The Pocket TTS trainer's latent precompute does that, and 14 workers
    each importing torch ran past the job's memory cap. Pinning `ceil(cpus)` cores makes the
    affinity agree with the budget, while `--cpus` still caps a fractional one. The range starts
    at core 0 because `0-<k-1>` exists on any Docker VM that accepts `--cpus k`, while a range
    at the top end needs the VM's own core count, which this host cannot see. The count is
    clamped to the host's cores; when `os.cpu_count()` cannot tell, the budget is used as is.
    """
    cores = math.ceil(cpus)
    host = os.cpu_count()
    if host is not None:
        cores = min(cores, host)
    return "0" if cores == 1 else f"0-{cores - 1}"


def load_families(path: Path = FAMILIES_PATH) -> Mapping[str, Mapping[str, object]]:
    """Return the `families` map of `families.json`.

    Raises:
        InputError: The file is missing or not JSON.
    """
    data = read_json(path)
    if not isinstance(data, dict) or not isinstance(data.get("families"), dict):
        raise InputError(f"{path.name}: no families map")
    return data["families"]


class LocalDocker:
    """Runs jobs on this host's Docker, after the GPU-idle preflight for GPU jobs.

    `peers` names containers every GPU job may share the card with; `executor_for` reads them
    from `MIST_VOICE_GPU_PEERS`.
    """

    def __init__(
        self,
        root: Root,
        docker: DockerOps,
        families: Mapping[str, Mapping[str, object]],
        peers: frozenset[str] = frozenset(),
    ) -> None:
        self._root = root
        self._docker = docker
        self._families = families
        self._peers = peers

    def _family(self, name: str) -> Mapping[str, object]:
        family = self._families.get(name)
        if family is None:
            raise InputError(f"refused: unknown family {name}")
        return family

    def _inside(self, rel: str) -> Path:
        path = (self._root.path / rel).resolve()
        if path == self._root.path or not path.is_relative_to(self._root.path):
            raise RootError("refused: a job input resolves outside the root")
        return path

    def argv(self, job: JobSpec) -> list[str]:
        """Return the `docker run` argv for a job.

        A family's `cpus` budget adds `--cpuset-cpus` beside `--cpus`, because the quota alone
        leaves the container's affinity at every core of the Docker VM (see `cpuset_for`).

        Raises:
            RootError: An input or the outputs resolve outside the root.
            InputError: An `env_names` entry is unset on the host, the family is unknown, or a
                family cap is not a positive number.
        """
        missing = [name for name in job.env_names if not os.environ.get(name)]
        if missing:
            raise InputError(f"refused: env var {', '.join(missing)} is unset")
        family = self._family(job.family)
        venv = family["venv"]
        fetch = job.module == FETCH_MODULE
        argv = ["docker", "run", "--rm", "--name", f"mvd-{job.job_id}"]
        if not job.network:
            argv += ["--network", "none"]
        if job.gpu:
            argv += ["--gpus", "all"]
        for key, flag, suffix in FAMILY_CAPS:
            if key in family:
                value = family[key]
                if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
                    raise InputError(f"refused: {job.family} {key} is not a positive number")
                if key == "cpus":
                    argv += ["--cpuset-cpus", cpuset_for(value)]
                argv += [flag, f"{value:g}{suffix}"]
        if job.stop_timeout_s is not None:
            argv += ["--stop-timeout", str(job.stop_timeout_s)]
        for item in job.inputs:
            argv += ["-v", f"{self._inside(item.rel)}:/data/{item.rel}:ro"]
        argv += ["-v", f"{self._root.writable(job.outputs)}:/data/{job.outputs}"]
        argv += ["-v", f"{self._root.path / 'cache'}:/cache" + ("" if fetch else ":ro")]
        if not fetch:
            argv += ["-e", "HF_HUB_OFFLINE=1"]
        argv += ["-e", "HF_HOME=/cache/hf", "-e", "TORCH_HOME=/cache/torch"]
        for name in job.env_names:
            argv += ["-e", name]
        for name in job.optional_env_names:
            if os.environ.get(name):
                argv += ["-e", name]
        # `-u`: the log shows each line as it is printed; `-I` ignores `PYTHONUNBUFFERED`.
        argv += [job.image, f"{venv}/bin/python", "-I", "-u", "-m", job.module, "--data", "/data"]
        return argv + list(job.args)

    def preflight(self, job: JobSpec, allow: frozenset[str] = frozenset()) -> None:
        """Refuse a GPU job while another container holds a GPU or free VRAM is short.

        GPU users outside Docker are invisible here; the daily stack runs every GPU service in
        Docker. `allow` names containers that may keep their GPU for this call (the bench's E4B
        server), and the executor's peers may keep theirs for every call. A peer still uses
        VRAM, so the free-VRAM check runs even when only peers hold the card. nvidia-smi runs
        as the entrypoint because a CUDA base image's own entrypoint prints its banner to stdout
        first, and stdout must hold only one free-MiB line per GPU.

        Raises:
            PreflightError: A container outside `allow` and the peers holds a GPU, or free VRAM
                is below the job's `vram_gb`, or the family's when the job sets none.
        """
        allowed = allow | self._peers
        for cid in self._docker.capture(["docker", "ps", "-q"]).split():
            line = self._docker.capture(["docker", "inspect", "--format", INSPECT_FORMAT, cid])
            name, _, requests = line.strip().partition(" ")
            name = name.lstrip("/")
            if json.loads(requests or "null") and name not in allowed:
                raise PreflightError(f"refused: container {name} holds a GPU; stop it first")
        smi = ["docker", "run", "--rm", "--network", "none", "--gpus", "all"]
        smi += ["--entrypoint", "nvidia-smi", job.image, *NVIDIA_SMI_QUERY]
        free_mib = max(int(v) for v in self._docker.capture(smi).split())
        if job.vram_gb is not None:
            need_gb, who = float(job.vram_gb), f"job {job.job_id}"
        else:
            need_gb = float(self._family(job.family)["vram_gb"])  # type: ignore[arg-type]
            who = f"family {job.family}"
        if free_mib / 1024 < need_gb:
            raise PreflightError(
                f"refused: {free_mib / 1024:.1f} GB VRAM free, {who} needs {need_gb:g}"
            )

    def run(self, job: JobSpec, allow: frozenset[str] = frozenset()) -> JobResult:
        """Run a job, save its log, and verify its outputs against its `outputs.json`.

        Raises:
            JobError: The container exited non-zero, or an output does not match its sha256.
        """
        argv = self.argv(job)
        if job.gpu:
            self.preflight(job, allow)
        run_dir = f"runs/{job.job_id}"
        write_json(self._root, f"{run_dir}/job.json", job.record())
        self._root.writable(job.outputs).mkdir(parents=True, exist_ok=True)
        log_rel = f"{run_dir}/log.txt"
        code = self._docker.run_logged(argv, self._root.writable(log_rel))
        if code != 0:
            raise JobError(f"job {job.job_id} exited {code}; log {log_rel}")
        outputs = self._verify(job)
        print(f"job {job.job_id}: exit 0, log {log_rel}")
        return JobResult(job_id=job.job_id, exit_code=0, log=log_rel, outputs=outputs)

    def _verify(self, job: JobSpec) -> tuple[JobInput, ...]:
        manifest = self._root.path / job.outputs / "outputs.json"
        try:
            data = read_json(manifest)
            files = tuple(JobInput.from_json(item) for item in data["files"])  # type: ignore[index]
        except (InputError, KeyError, TypeError) as exc:
            raise JobError(f"job {job.job_id}: {job.outputs}/outputs.json is missing") from exc
        for item in files:
            path = self._root.path / item.rel
            if not path.is_file() or sha256_file(path) != item.sha256:
                raise JobError(f"job {job.job_id}: output {item.rel} does not match its sha256")
        self._root.write_bytes(f"runs/{job.job_id}/outputs.json", manifest.read_bytes())
        return files


def gpu_peers() -> frozenset[str]:
    """Return the container names `MIST_VOICE_GPU_PEERS` lists; blanks are dropped."""
    raw = os.environ.get(GPU_PEERS_ENV, "")
    return frozenset(name for name in (part.strip() for part in raw.split(",")) if name)


def executor_for(
    name: str,
    *,
    root: Root,
    docker: DockerOps | None = None,
    families: Mapping[str, Mapping[str, object]] | None = None,
) -> Executor:
    """Return the executor `local` or `remote:<alias>` names; every command gets one here.

    The remote executor is imported only when asked for, so local runs work before it exists.
    The local executor's GPU peers come from `MIST_VOICE_GPU_PEERS`, so no command needs a flag.

    Raises:
        InputError: An unknown name, or the remote executor is not installed.
    """
    if name == "local":
        return LocalDocker(
            root,
            docker or HostDocker(),
            load_families() if families is None else families,
            gpu_peers(),
        )
    kind, _, alias = name.partition(":")
    if kind != "remote" or not alias:
        raise InputError(f"refused: unknown executor {name!r}; use local or remote:<alias>")
    try:
        remote = importlib.import_module(REMOTE_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name != REMOTE_MODULE:
            raise
        raise InputError("refused: the remote executor is not available yet") from exc
    return remote.RemotePod(alias, root=root)
