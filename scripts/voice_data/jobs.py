"""Container jobs: the job model, the `Executor` protocol and the local Docker executor.

Every heavy stage runs as a job in its family's image, so the host needs only the standard
library. A job names its env vars, never their values: `docker run -e NAME` copies the value from
the host environment, so a key never appears in an argv, a log or a record. The container log goes
to `runs/<job_id>/log.txt` and is never printed, because a stage's log can hold transcripts.
"""

from __future__ import annotations

import importlib
import json
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
NVIDIA_SMI = ("nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits")


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
    """Runs jobs on this host's Docker, after the GPU-idle preflight for GPU jobs."""

    def __init__(
        self, root: Root, docker: DockerOps, families: Mapping[str, Mapping[str, object]]
    ) -> None:
        self._root = root
        self._docker = docker
        self._families = families

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

        Raises:
            RootError: An input or the outputs resolve outside the root.
            InputError: An `env_names` entry is unset on the host, or the family is unknown.
        """
        missing = [name for name in job.env_names if not os.environ.get(name)]
        if missing:
            raise InputError(f"refused: env var {', '.join(missing)} is unset")
        venv = self._family(job.family)["venv"]
        fetch = job.module == FETCH_MODULE
        argv = ["docker", "run", "--rm", "--name", f"mvd-{job.job_id}"]
        if not job.network:
            argv += ["--network", "none"]
        if job.gpu:
            argv += ["--gpus", "all"]
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
        argv += [job.image, f"{venv}/bin/python", "-I", "-m", job.module, "--data", "/data"]
        return argv + list(job.args)

    def preflight(self, job: JobSpec, allow: frozenset[str] = frozenset()) -> None:
        """Refuse a GPU job while another container holds a GPU or free VRAM is short.

        GPU users outside Docker are invisible here; the daily stack runs every GPU service in
        Docker. `allow` names containers that may keep their GPU (the bench's E4B server).

        Raises:
            PreflightError: A container outside `allow` holds a GPU, or free VRAM is below the
                family's `vram_gb`.
        """
        for cid in self._docker.capture(["docker", "ps", "-q"]).split():
            line = self._docker.capture(["docker", "inspect", "--format", INSPECT_FORMAT, cid])
            name, _, requests = line.strip().partition(" ")
            name = name.lstrip("/")
            if json.loads(requests or "null") and name not in allow:
                raise PreflightError(f"refused: container {name} holds a GPU; stop it first")
        smi = ["docker", "run", "--rm", "--network", "none", "--gpus", "all", job.image]
        free_mib = max(int(v) for v in self._docker.capture([*smi, *NVIDIA_SMI]).split())
        need_gb = float(self._family(job.family)["vram_gb"])  # type: ignore[arg-type]
        if free_mib / 1024 < need_gb:
            raise PreflightError(
                f"refused: {free_mib / 1024:.1f} GB VRAM free, family {job.family} needs {need_gb:g}"
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


def executor_for(
    name: str,
    *,
    root: Root,
    docker: DockerOps | None = None,
    families: Mapping[str, Mapping[str, object]] | None = None,
) -> Executor:
    """Return the executor `local` or `remote:<alias>` names; every command gets one here.

    The remote executor is imported only when asked for, so local runs work before it exists.

    Raises:
        InputError: An unknown name, or the remote executor is not installed.
    """
    if name == "local":
        return LocalDocker(
            root, docker or HostDocker(), load_families() if families is None else families
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
