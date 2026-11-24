"""The extract job's VRAM budget: the allocator cap, peak readings and the out-of-memory test.

A job that outgrows the card under WSL2 does not fail: the driver pages device memory to host
RAM and the job slows by orders of magnitude, with nothing in the log to say so. `start` caps
PyTorch's caching allocator 1 GiB under the job's budget, so an overrun raises
`torch.OutOfMemoryError` instead, and `out_of_memory` recognises that error however a binding
wrapped it. The 1 GiB between the cap and the budget is for memory the allocator does not count,
such as the CUDA context's growth and library workspaces.

torch is imported inside each function, so a module walk imports no model package, and every
VRAM function is a no-op that returns `None` on the CPU.
"""

from __future__ import annotations

from dataclasses import dataclass

CUDA = "cuda"
GIB = 1 << 30
MIB = 1 << 20
# A start with less free VRAM than the budget less this is refused before any model loads.
SHORT_MARGIN_GIB = 0.5
# The allocator's cap sits this far under the budget.
CAP_MARGIN_GIB = 1.0
# CUDA failures that are out-of-memory errors without being `torch.OutOfMemoryError`.
OOM_MESSAGES = ("CUDA error: out of memory", "CUBLAS_STATUS_ALLOC_FAILED")


@dataclass(frozen=True, slots=True)
class VramStart:
    """The card at the job's start, in GiB; `cap_gib` is `None` when free VRAM was short."""

    budget_gib: float
    free_gib: float
    total_gib: float
    needed_gib: float
    cap_gib: float | None

    @property
    def short(self) -> bool:
        """Whether free VRAM was below `needed_gib`, so no cap was set."""
        return self.cap_gib is None

    def as_json(self) -> dict[str, float | None]:
        """The `vram` object of an `ok` extract result."""
        return {
            "budget_gib": self.budget_gib,
            "cap_gib": self.cap_gib,
            "free_gib": self.free_gib,
            "total_gib": self.total_gib,
        }


def start(device: str, budget_gib: float) -> VramStart | None:
    """Read free VRAM and cap the allocator at the budget less 1 GiB.

    The `mem_get_info` call creates the job's CUDA context, so the free figure is net of it.
    Free VRAM below the budget less 0.5 GiB returns a short start and sets no cap.

    Raises:
        ValueError: The budget is 1 GiB or less, which leaves the allocator no room.
    """
    if device != CUDA:
        return None
    if budget_gib <= CAP_MARGIN_GIB:
        raise ValueError(f"a VRAM budget must be above {CAP_MARGIN_GIB:g} GiB")
    import torch

    free, total = torch.cuda.mem_get_info()
    needed = budget_gib - SHORT_MARGIN_GIB
    cap: float | None = None
    if free >= needed * GIB:
        cap = budget_gib - CAP_MARGIN_GIB
        torch.cuda.set_per_process_memory_fraction(cap * GIB / total)
    return VramStart(
        budget_gib=round(budget_gib, 2),
        free_gib=round(free / GIB, 2),
        total_gib=round(total / GIB, 2),
        needed_gib=round(needed, 2),
        cap_gib=None if cap is None else round(cap, 2),
    )


def reset_peak(device: str) -> None:
    """Start the allocator's peak readings again."""
    if device != CUDA:
        return None
    import torch

    torch.cuda.reset_peak_memory_stats()
    return None


def peak_mib(device: str) -> dict[str, int] | None:
    """Return the reserved and allocated peaks since the last reset, in MiB."""
    if device != CUDA:
        return None
    import torch

    return {
        "reserved": int(torch.cuda.max_memory_reserved()) // MIB,
        "allocated": int(torch.cuda.max_memory_allocated()) // MIB,
    }


def free_mib(device: str) -> int | None:
    """Return the card's free VRAM in MiB, as the driver reports it."""
    if device != CUDA:
        return None
    import torch

    free, _total = torch.cuda.mem_get_info()
    return int(free) // MIB


def out_of_memory(exc: BaseException) -> bool:
    """Whether `exc`, or an error it was raised from or while handling, is a CUDA OOM.

    A binding's `library_errors` raises `JobError` from the library's error, and an error in
    cleanup carries the original as its context, so both links are walked; a set of the errors
    already seen ends a cycle.
    """
    import torch

    seen: set[int] = set()
    pending: list[BaseException | None] = [exc]
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, torch.OutOfMemoryError):
            return True
        if isinstance(current, RuntimeError) and any(m in str(current) for m in OOM_MESSAGES):
            return True
        pending += [current.__cause__, current.__context__]
    return False
