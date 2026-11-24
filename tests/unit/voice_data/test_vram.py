"""Tests for `stages/vram.py`: the extract job's VRAM budget, allocator cap and out-of-memory test.

No GPU is used: every `torch.cuda` call the module makes is replaced with a recording fake, and
the out-of-memory errors are built on CPU torch.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch

from scripts.voice_data.commands import extract as extract_cmd
from scripts.voice_data.errors import JobError
from scripts.voice_data.stages import vram
from scripts.voice_data.stages.fetch import library_errors

GIB = 1 << 30
MIB = 1 << 20


@dataclass
class FakeCuda:
    """The `torch.cuda` calls `vram` makes, with the card's free and total bytes set."""

    free: int
    total: int = 12 * GIB
    fractions: list[float] = field(default_factory=list)
    resets: int = 0
    reserved: int = 0
    allocated: int = 0

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *args: (self.free, self.total))
        monkeypatch.setattr(
            torch.cuda, "set_per_process_memory_fraction", lambda f, *a: self.fractions.append(f)
        )
        monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", self._reset)
        monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *a: self.reserved)
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a: self.allocated)

    def _reset(self, *args: object) -> None:
        self.resets += 1


@pytest.fixture
def cuda(monkeypatch: pytest.MonkeyPatch) -> FakeCuda:
    fake = FakeCuda(free=int(10.4 * GIB))
    fake.install(monkeypatch)
    return fake


def test_start_caps_the_allocator_one_gib_under_the_budget(cuda: FakeCuda) -> None:
    started = vram.start("cuda", 8.0)
    assert started is not None and not started.short
    assert cuda.fractions == [pytest.approx(7.0 * GIB / (12 * GIB))]
    assert started.budget_gib == 8.0 and started.cap_gib == 7.0
    assert started.free_gib == pytest.approx(10.4) and started.total_gib == 12.0
    assert started.needed_gib == 7.5
    assert started.as_json() == {
        "budget_gib": 8.0,
        "cap_gib": 7.0,
        "free_gib": pytest.approx(10.4),
        "total_gib": 12.0,
    }


def test_start_is_short_below_the_budget_less_half_a_gib(cuda: FakeCuda) -> None:
    cuda.free = int(7.5 * GIB) - 1
    started = vram.start("cuda", 8.0)
    assert started is not None and started.short
    assert started.cap_gib is None and started.needed_gib == 7.5
    assert started.free_gib == pytest.approx(7.5, abs=0.01)
    assert cuda.fractions == []


def test_start_at_exactly_the_budget_less_half_a_gib_is_not_short(cuda: FakeCuda) -> None:
    cuda.free = int(7.5 * GIB)
    started = vram.start("cuda", 8.0)
    assert started is not None and not started.short
    assert len(cuda.fractions) == 1


@pytest.mark.parametrize("budget", [0.5, 1.0])
def test_start_refuses_a_budget_at_or_under_the_host_floor(cuda: FakeCuda, budget: float) -> None:
    assert vram.CAP_MARGIN_GIB == extract_cmd.VRAM_FLOOR_GIB
    with pytest.raises(ValueError, match="must be above 1 GiB"):
        vram.start("cuda", budget)
    assert cuda.fractions == []


def test_peaks_and_free_memory_in_mib(cuda: FakeCuda) -> None:
    cuda.reserved, cuda.allocated = 6000 * MIB + 5, 5000 * MIB
    vram.reset_peak("cuda")
    assert cuda.resets == 1
    assert vram.peak_mib("cuda") == {"reserved": 6000, "allocated": 5000}
    assert vram.free_mib("cuda") == int(10.4 * 1024)


def test_every_call_is_a_no_op_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object) -> None:
        raise AssertionError("a cpu run touches no CUDA call")

    for name in ("mem_get_info", "set_per_process_memory_fraction", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, refuse)
    assert vram.start("cpu", 8.0) is None
    assert vram.reset_peak("cpu") is None
    assert vram.peak_mib("cpu") is None
    assert vram.free_mib("cpu") is None


def _oom() -> torch.OutOfMemoryError:
    return torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")


def test_out_of_memory_bare() -> None:
    assert vram.out_of_memory(_oom())


def test_out_of_memory_wrapped_by_library_errors() -> None:
    with pytest.raises(JobError) as info, library_errors("separator-dme"):
        raise _oom()
    assert str(info.value) == "separator-dme: OutOfMemoryError"
    assert vram.out_of_memory(info.value)


def test_out_of_memory_raised_while_handling_another_error() -> None:
    with pytest.raises(ValueError) as info:
        try:
            raise _oom()
        except RuntimeError:
            raise ValueError("cleanup failed") from None
    assert info.value.__cause__ is None and info.value.__context__ is not None
    assert vram.out_of_memory(info.value)


@pytest.mark.parametrize(
    "message",
    [
        "CUDA error: out of memory\nCUDA kernel errors might be asynchronously reported",
        "CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`",
    ],
)
def test_out_of_memory_as_a_cuda_runtime_error(message: str) -> None:
    assert vram.out_of_memory(RuntimeError(message))


def test_other_errors_are_not_out_of_memory() -> None:
    assert not vram.out_of_memory(RuntimeError("CUDA error: device-side assert triggered"))
    assert not vram.out_of_memory(ValueError("out of memory"))
    assert not vram.out_of_memory(JobError("separator-dme: RuntimeError"))


def test_a_cycle_in_the_chain_ends() -> None:
    first, second = RuntimeError("a"), RuntimeError("b")
    first.__context__, second.__context__ = second, first
    assert not vram.out_of_memory(first)


def test_importing_vram_imports_no_model_package() -> None:
    tree = ast.parse(Path(vram.__file__).read_text(encoding="utf-8"))
    top: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            top.append(node.module)
    assert {name.split(".")[0] for name in top} <= {"__future__", "collections", "dataclasses"}
