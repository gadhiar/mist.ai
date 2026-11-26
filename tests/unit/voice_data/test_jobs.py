"""Tests for the CPU affinity a job container gets from its family's `cpus` budget.

`--cpus` alone is a CFS quota: the container still sees every core of the Docker VM in
`os.sched_getaffinity(0)`, so a process pool sized from it (the Pocket TTS trainer's latent
precompute) starts one worker per VM core and runs out of memory. Each test pins the host's core
count through `jobs.os.cpu_count`, so no result depends on the machine running the suite.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from scripts.voice_data import jobs
from scripts.voice_data.records import JobInput
from scripts.voice_data.root import Root

FAMILY = "train-pocket"
VENV = "/opt/venv-train-pocket"
IMAGE = "mist-voice-train-pocket:0123456789ab"


class NoDocker:
    """A `DockerOps` for argv tests: building an argv never calls docker, so any call fails."""

    def capture(self, argv: Sequence[str]) -> str:
        raise AssertionError(f"unexpected docker call {list(argv)}")

    def check(self, argv: Sequence[str]) -> None:
        raise AssertionError(f"unexpected docker call {list(argv)}")

    def run_logged(self, argv: Sequence[str], log: Path) -> int:
        raise AssertionError(f"unexpected docker call {list(argv)}")


def _job() -> jobs.JobSpec:
    return jobs.JobSpec(
        job_id="job-20261126T0900Z-1",
        family=FAMILY,
        image=IMAGE,
        module="scripts.voice_data.stages.student_train",
        args=("--run", "r1"),
        inputs=(JobInput(rel="seed/tracks/src-0001.mka", sha256="0" * 64),),
        outputs="seed/runs/r1",
        gpu=True,
        network=False,
    )


def _argv(
    root: Root,
    monkeypatch: pytest.MonkeyPatch,
    family: Mapping[str, object],
    host_cores: int | None,
) -> list[str]:
    monkeypatch.setattr(jobs.os, "cpu_count", lambda: host_cores)
    return jobs.LocalDocker(root, NoDocker(), {FAMILY: family}).argv(_job())


def _pairs(argv: list[str], flag: str) -> list[str]:
    return [argv[i + 1] for i, value in enumerate(argv[:-1]) if value == flag]


def _family(**caps: object) -> dict[str, object]:
    return {"image": "mist-voice-train-pocket", "venv": VENV, "vram_gb": 16, **caps}


def test_whole_cpu_budget_pins_that_many_cores_from_core_zero(
    private_root: Root, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv = _argv(private_root, monkeypatch, _family(cpus=8), host_cores=16)

    assert _pairs(argv, "--cpuset-cpus") == ["0-7"]


def test_fractional_cpu_budget_rounds_the_core_count_up(
    private_root: Root, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv = _argv(private_root, monkeypatch, _family(cpus=2.5), host_cores=16)

    assert _pairs(argv, "--cpuset-cpus") == ["0-2"]


def test_fractional_cpu_budget_keeps_its_quota(
    private_root: Root, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv = _argv(private_root, monkeypatch, _family(cpus=2.5), host_cores=16)

    assert _pairs(argv, "--cpus") == ["2.5"]


def test_single_core_budget_pins_core_zero_alone(
    private_root: Root, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv = _argv(private_root, monkeypatch, _family(cpus=0.5), host_cores=16)

    assert _pairs(argv, "--cpuset-cpus") == ["0"]


def test_cpu_budget_above_the_host_cores_is_clamped_to_them(
    private_root: Root, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv = _argv(private_root, monkeypatch, _family(cpus=8), host_cores=4)

    assert _pairs(argv, "--cpuset-cpus") == ["0-3"]


def test_unknown_host_core_count_pins_the_whole_budget(
    private_root: Root, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv = _argv(private_root, monkeypatch, _family(cpus=8), host_cores=None)

    assert _pairs(argv, "--cpuset-cpus") == ["0-7"]


def test_affinity_flag_sits_immediately_before_the_cpu_quota(
    private_root: Root, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv = _argv(private_root, monkeypatch, _family(cpus=8), host_cores=16)

    assert argv.index("--cpuset-cpus") + 2 == argv.index("--cpus")


def test_family_without_a_cpu_budget_gets_no_affinity(
    private_root: Root, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv = _argv(private_root, monkeypatch, _family(memory_gb=9), host_cores=16)

    assert "--cpuset-cpus" not in argv


def test_train_pocket_family_pins_eight_cores_beside_its_quota(
    private_root: Root, monkeypatch: pytest.MonkeyPatch
) -> None:
    family = jobs.load_families()[FAMILY]

    argv = _argv(private_root, monkeypatch, family, host_cores=16)

    assert (_pairs(argv, "--cpus"), _pairs(argv, "--cpuset-cpus")) == (["8"], ["0-7"])
