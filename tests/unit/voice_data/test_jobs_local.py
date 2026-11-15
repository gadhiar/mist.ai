"""Tests for the local Docker executor: argv, refusals, GPU preflight, logs and output checks.

Every docker call goes through a fake `DockerOps`, so no test starts a container.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
import types
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from scripts.voice_data import cli, jobs
from scripts.voice_data.errors import InputError, JobError, RootError
from scripts.voice_data.records import JobInput
from scripts.voice_data.root import Root

FAMILY = "teacher-indextts2"
VENV = "/opt/venv-teacher-indextts2"
IMAGE = "mist-voice-teacher-indextts2:0123456789ab"
FAMILIES = {FAMILY: {"image": "mist-voice-teacher-indextts2", "venv": VENV, "vram_gb": 16}}
SHA = "0" * 64


class FakeDocker:
    """Records every argv; answers `ps`, `inspect` and the `nvidia-smi` run from fixed data."""

    def __init__(
        self,
        *,
        containers: dict[str, tuple[str, str]] | None = None,
        free_mib: str = "20000",
        exit_code: int = 0,
        log_text: str = "",
        on_run: Callable[[], None] | None = None,
    ) -> None:
        self.calls: list[list[str]] = []
        self.containers = containers or {}
        self.free_mib = free_mib
        self.exit_code = exit_code
        self.log_text = log_text
        self.on_run = on_run

    def capture(self, argv: Sequence[str]) -> str:
        self.calls.append(list(argv))
        if list(argv) == ["docker", "ps", "-q"]:
            return "".join(f"{cid}\n" for cid in self.containers)
        if argv[1] == "inspect":
            name, requests = self.containers[argv[-1]]
            return f"/{name} {requests}\n"
        return self.free_mib + "\n"

    def check(self, argv: Sequence[str]) -> None:
        self.calls.append(list(argv))

    def run_logged(self, argv: Sequence[str], log: Path) -> int:
        self.calls.append(list(argv))
        log.write_text(self.log_text, encoding="utf-8")
        if self.on_run is not None:
            self.on_run()
        return self.exit_code


def _job(**changes: object) -> jobs.JobSpec:
    base = jobs.JobSpec(
        job_id="job-20261112T1015Z-1",
        family=FAMILY,
        image=IMAGE,
        module="scripts.voice_data.stages.extract",
        args=("--run", "r1"),
        inputs=(JobInput(rel="seed/tracks/src-0001.mka", sha256=SHA),),
        outputs="seed/runs/r1",
        gpu=False,
        network=False,
    )
    return dataclasses.replace(base, **changes)


def _local(root: Root, docker: FakeDocker | None = None) -> jobs.LocalDocker:
    return jobs.LocalDocker(root, docker or FakeDocker(), FAMILIES)


def _pairs(argv: list[str], flag: str) -> list[str]:
    return [argv[i + 1] for i, value in enumerate(argv[:-1]) if value == flag]


def test_job_spec_and_result_are_frozen_slotted() -> None:
    job = _job()
    result = jobs.JobResult(job_id=job.job_id, exit_code=0, log="runs/x/log.txt", outputs=())
    for obj in (job, result):
        assert not hasattr(obj, "__dict__")
        with pytest.raises(dataclasses.FrozenInstanceError):
            obj.job_id = "other"  # type: ignore[misc]


def test_job_spec_record_matches_the_job_record() -> None:
    record = _job().record().to_json()
    assert record["schema"] == 1
    assert record["inputs"] == [{"rel": "seed/tracks/src-0001.mka", "sha256": SHA}]
    assert record["optional_env_names"] == []


def test_argv_is_exactly_the_interface(private_root: Root, monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    root = private_root.path
    assert _local(private_root).argv(_job()) == [
        "docker",
        "run",
        "--rm",
        "--name",
        "mvd-job-20261112T1015Z-1",
        "--network",
        "none",
        "-v",
        f"{root / 'seed/tracks/src-0001.mka'}:/data/seed/tracks/src-0001.mka:ro",
        "-v",
        f"{root / 'seed/runs/r1'}:/data/seed/runs/r1",
        "-v",
        f"{root / 'cache'}:/cache:ro",
        "-e",
        "HF_HUB_OFFLINE=1",
        "-e",
        "HF_HOME=/cache/hf",
        "-e",
        "TORCH_HOME=/cache/torch",
        IMAGE,
        f"{VENV}/bin/python",
        "-I",
        "-m",
        "scripts.voice_data.stages.extract",
        "--data",
        "/data",
        "--run",
        "r1",
    ]


def test_network_none_unless_the_job_has_network(private_root: Root) -> None:
    local = _local(private_root)
    assert "--network" in local.argv(_job())
    assert "--network" not in local.argv(_job(network=True))


def test_gpus_all_only_for_gpu_jobs(private_root: Root) -> None:
    local = _local(private_root)
    assert "--gpus" not in local.argv(_job())
    argv = local.argv(_job(gpu=True))
    assert argv[argv.index("--gpus") + 1] == "all"


def test_inputs_mount_read_only_and_cache_read_only(private_root: Root) -> None:
    job = _job(
        inputs=(
            JobInput(rel="seed/tracks/src-0001.mka", sha256=SHA),
            JobInput(rel="seed/raw/show/ep1.mkv", sha256=SHA),
        )
    )
    mounts = _pairs(_local(private_root).argv(job), "-v")
    assert mounts[0].endswith(":/data/seed/tracks/src-0001.mka:ro")
    assert mounts[1].endswith(":/data/seed/raw/show/ep1.mkv:ro")
    assert mounts[2].endswith(":/data/seed/runs/r1")
    assert mounts[3] == f"{private_root.path / 'cache'}:/cache:ro"


def test_fetch_job_has_a_writable_cache_and_no_offline_flag(private_root: Root) -> None:
    local = _local(private_root)
    argv = local.argv(_job(module=jobs.FETCH_MODULE, network=True))
    env = _pairs(argv, "-e")
    assert f"{private_root.path / 'cache'}:/cache" in _pairs(argv, "-v")
    assert "HF_HUB_OFFLINE=1" not in env
    assert {"HF_HOME=/cache/hf", "TORCH_HOME=/cache/torch"} <= set(env)
    other = _pairs(local.argv(_job()), "-e")
    assert {"HF_HUB_OFFLINE=1", "HF_HOME=/cache/hf", "TORCH_HOME=/cache/torch"} <= set(other)


def test_env_names_pass_by_name_only(private_root: Root, canaries, assert_no_canary) -> None:
    argv = _local(private_root).argv(_job(env_names=("ELEVENLABS_API_KEY",)))
    assert "ELEVENLABS_API_KEY" in _pairs(argv, "-e")
    assert_no_canary(" ".join(argv))


def test_optional_env_name_passed_when_set(private_root: Root, canaries, assert_no_canary) -> None:
    argv = _local(private_root).argv(_job(optional_env_names=("HF_TOKEN",)))
    assert "HF_TOKEN" in _pairs(argv, "-e")
    assert_no_canary(" ".join(argv))


def test_optional_env_name_omitted_when_unset(private_root: Root, monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    argv = _local(private_root).argv(_job(optional_env_names=("HF_TOKEN",)))
    assert "HF_TOKEN" not in argv


def test_interpreter_is_the_family_venv(private_root: Root) -> None:
    argv = _local(private_root).argv(_job())
    start = argv.index(IMAGE)
    assert argv[start + 1 : start + 7] == [
        f"{VENV}/bin/python",
        "-I",
        "-m",
        "scripts.voice_data.stages.extract",
        "--data",
        "/data",
    ]


@pytest.mark.parametrize("rel", ["../outside.mka", "/etc/passwd"])
def test_input_outside_the_root_is_refused(private_root: Root, rel: str) -> None:
    with pytest.raises(RootError):
        _local(private_root).argv(_job(inputs=(JobInput(rel=rel, sha256=SHA),)))


@pytest.mark.parametrize("rel", ["../outside", "/tmp/out"])
def test_output_outside_the_root_is_refused(private_root: Root, rel: str) -> None:
    with pytest.raises(RootError):
        _local(private_root).argv(_job(outputs=rel))


def test_unset_env_name_is_refused_naming_it(private_root: Root, monkeypatch) -> None:
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    with pytest.raises(InputError, match="ELEVENLABS_API_KEY") as caught:
        _local(private_root).argv(_job(env_names=("ELEVENLABS_API_KEY",)))
    assert cli.exit_code(caught.value) == 2


GPU_REQUEST = '[{"Driver":"","Count":-1,"DeviceIDs":null,"Capabilities":[["gpu"]],"Options":{}}]'


def test_preflight_refuses_a_gpu_holder_naming_it(private_root: Root) -> None:
    docker = FakeDocker(containers={"a1": ("mist-neo4j", "null"), "b2": ("mist-llm", GPU_REQUEST)})
    with pytest.raises(jobs.PreflightError, match="mist-llm") as caught:
        _local(private_root, docker).preflight(_job(gpu=True))
    assert cli.exit_code(caught.value) == 2
    assert docker.calls[0] == ["docker", "ps", "-q"]
    assert docker.calls[1] == ["docker", "inspect", "--format", jobs.INSPECT_FORMAT, "a1"]


def test_preflight_passes_an_allowed_gpu_holder(private_root: Root) -> None:
    docker = FakeDocker(containers={"b2": ("mist-llm", GPU_REQUEST)})
    _local(private_root, docker).preflight(_job(gpu=True), allow=frozenset({"mist-llm"}))
    nvidia = docker.calls[-1]
    assert nvidia[:2] == ["docker", "run"] and "--gpus" in nvidia and IMAGE in nvidia
    assert nvidia[nvidia.index(IMAGE) + 1] == "nvidia-smi"


def test_preflight_refuses_free_vram_below_the_family_need(private_root: Root) -> None:
    docker = FakeDocker(free_mib="8192")
    with pytest.raises(jobs.PreflightError, match="VRAM") as caught:
        _local(private_root, docker).preflight(_job(gpu=True))
    assert cli.exit_code(caught.value) == 2


def _writes_outputs(root: Root, files: dict[str, bytes], manifest: dict[str, str] | None = None):
    def write() -> None:
        listed = []
        for rel, data in files.items():
            path = root.path / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            digest = (manifest or {}).get(rel, hashlib.sha256(data).hexdigest())
            listed.append({"rel": rel, "sha256": digest})
        doc = {"schema": 1, "files": listed}
        (root.path / "seed/runs/r1/outputs.json").write_text(json.dumps(doc), encoding="utf-8")

    return write


def test_run_verifies_outputs_and_prints_only_status(
    private_root: Root, capsys, canaries, assert_no_canary
) -> None:
    files = {"seed/runs/r1/segments.jsonl": b"{}\n"}
    docker = FakeDocker(log_text=canaries.transcript, on_run=_writes_outputs(private_root, files))
    result = _local(private_root, docker).run(_job())
    assert result.exit_code == 0
    assert result.log == "runs/job-20261112T1015Z-1/log.txt"
    assert result.outputs == (
        JobInput(rel="seed/runs/r1/segments.jsonl", sha256=hashlib.sha256(b"{}\n").hexdigest()),
    )
    run_dir = private_root.path / "runs/job-20261112T1015Z-1"
    assert (run_dir / "outputs.json").is_file()
    assert json.loads((run_dir / "job.json").read_text(encoding="utf-8"))["family"] == FAMILY
    out = capsys.readouterr()
    assert "job-20261112T1015Z-1" in out.out and "runs/job-20261112T1015Z-1/log.txt" in out.out
    assert_no_canary(out.out + out.err)


def test_gpu_run_runs_the_preflight_before_the_job(private_root: Root) -> None:
    files = {"seed/runs/r1/a.bin": b"a"}
    docker = FakeDocker(on_run=_writes_outputs(private_root, files))
    _local(private_root, docker).run(_job(gpu=True))
    assert docker.calls[0] == ["docker", "ps", "-q"]
    assert docker.calls[-1][-3:] == ["/data", "--run", "r1"]


def test_non_zero_exit_is_job_error_with_the_log_saved_not_printed(
    private_root: Root, capsys, canaries, assert_no_canary
) -> None:
    docker = FakeDocker(exit_code=3, log_text=canaries.transcript)
    with pytest.raises(JobError) as caught:
        _local(private_root, docker).run(_job())
    assert cli.exit_code(caught.value) == 4
    assert "runs/job-20261112T1015Z-1/log.txt" in str(caught.value)
    log = private_root.path / "runs/job-20261112T1015Z-1/log.txt"
    assert log.read_text(encoding="utf-8") == canaries.transcript
    out = capsys.readouterr()
    assert_no_canary(out.out + out.err + str(caught.value))


def test_output_sha256_mismatch_is_job_error(private_root: Root) -> None:
    files = {"seed/runs/r1/a.bin": b"a"}
    write = _writes_outputs(private_root, files, manifest={"seed/runs/r1/a.bin": SHA})
    with pytest.raises(JobError, match="seed/runs/r1/a.bin") as caught:
        _local(private_root, FakeDocker(on_run=write)).run(_job())
    assert cli.exit_code(caught.value) == 4


def test_missing_outputs_manifest_is_job_error(private_root: Root) -> None:
    with pytest.raises(JobError, match="outputs.json"):
        _local(private_root).run(_job())


def test_executor_for_local(private_root: Root) -> None:
    executor = jobs.executor_for("local", root=private_root, docker=FakeDocker(), families={})
    assert isinstance(executor, jobs.LocalDocker)


def test_executor_for_remote_while_absent_exits_2(private_root: Root, monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "scripts.voice_data.remote", raising=False)
    with pytest.raises(InputError, match="remote") as caught:
        jobs.executor_for("remote:pod1", root=private_root)
    assert cli.exit_code(caught.value) == 2


def test_executor_for_remote_uses_the_remote_module(private_root: Root, monkeypatch) -> None:
    module = types.ModuleType("scripts.voice_data.remote")

    class RemotePod:
        def __init__(self, alias: str, *, root: Root) -> None:
            self.alias = alias
            self.root = root

    module.RemotePod = RemotePod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "scripts.voice_data.remote", module)
    executor = jobs.executor_for("remote:pod1", root=private_root)
    assert isinstance(executor, RemotePod) and executor.alias == "pod1"


@pytest.mark.parametrize("name", ["", "remote:", "cloud"])
def test_executor_for_refuses_an_unknown_name(private_root: Root, name: str) -> None:
    with pytest.raises(InputError):
        jobs.executor_for(name, root=private_root, families={})
