"""System telemetry collector for the ADR-017 `system_status` WS emit.

Snapshots CPU + RAM via psutil and GPU (utilization / VRAM / temperature)
via NVML on a 5-second cadence (driven by the emit task in server.py).
Graceful degradation on systems without an NVIDIA GPU: init_gpu()
returns False, collect_metrics() emits a placeholder gpu block with
name='none'.

The collector is a free-function module rather than a class so the
emit task can call collect_metrics() with no state passed across the
async boundary. GPU handle lives in a module-level singleton because
NVML's nvmlInit() / nvmlShutdown() are process-scoped.

Also collects `uptime_seconds` (ADR-017 v1.2.0,
knowledge-vault/Decisions/ADR-017-frontend-websocket-message-contract.md:125).
That field measures the BACKEND PROCESS via `psutil.Process().create_time()`,
not the container or host machine -- `psutil.boot_time()` would measure
host uptime and must never be used here. `UPTIME_SOURCE` is the constant
emitted alongside it so the wire payload names what was measured.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import psutil

try:
    import pynvml

    _HAS_PYNVML = True
except ImportError:
    _HAS_PYNVML = False

logger = logging.getLogger(__name__)

_GIB: int = 1024**3

# NVML handle for device 0. None when (a) init_gpu() has not been called,
# (b) NVML is unavailable, (c) the system has no NVIDIA GPU, or
# (d) NVML initialization failed for any reason.
_GPU_HANDLE: object | None = None
_GPU_INIT_FAILED: bool = False

# Backend process start time (unix seconds), cached after the first
# successful read -- it cannot change for the life of the process. None
# when (a) it has not been read yet, or (b) the read failed and
# _PROCESS_START_TIME_READ_FAILED records that as a one-way door, mirroring
# _GPU_INIT_FAILED above, so a failing read is not retried (and re-logged)
# on every 5s tick.
_PROCESS_START_TIME: float | None = None
_PROCESS_START_TIME_READ_FAILED: bool = False

# Emitted verbatim on the wire beside uptime_seconds (ADR-017 v1.2.0) so the
# payload names what was measured. Not a SystemMetrics field: it never
# varies within a process, unlike the sampled values.
UPTIME_SOURCE: str = "process"


@dataclass(frozen=True)
class CPUMetrics:
    """CPU sample for a single tick."""

    percent: float
    cores: int


@dataclass(frozen=True)
class RAMMetrics:
    """RAM sample for a single tick. Values in GiB."""

    used_gb: float
    total_gb: float
    percent: float


@dataclass(frozen=True)
class GPUMetrics:
    """GPU sample for a single tick.

    Placeholder values (name='none', zeros, temperature=None) when no
    NVIDIA GPU is available or NVML failed to initialize. The FE
    renders 'none' gracefully (zero-filled bars, name label 'none').
    """

    name: str
    utilization_percent: float
    vram_used_gb: float
    vram_total_gb: float
    temperature_c: float | None


@dataclass(frozen=True)
class SystemMetrics:
    """Single-tick snapshot composed for the system_status WS payload.

    timestamp is unix milliseconds, captured at the moment of sampling.

    uptime_seconds has no default deliberately: every construction site
    (collect_metrics() below, and _fake_snapshot() in
    tests/unit/test_server_system_status.py) must supply it explicitly, so
    a future caller cannot silently emit null by omission.
    """

    timestamp: float
    cpu: CPUMetrics
    ram: RAMMetrics
    gpu: GPUMetrics
    uptime_seconds: float | None


def init_gpu() -> bool:
    """Initialize NVML and grab a handle for device 0.

    Idempotent: re-calling after success or recorded failure is a no-op.
    Safe to call from server startup unconditionally; failure paths set
    a module-level flag so subsequent collect_metrics() calls return
    the placeholder GPU block without re-trying NVML.

    Returns:
        True if NVML initialized and device 0 is accessible.
        False on any failure (NVML missing, no NVIDIA driver, no GPU,
        permission error, etc.).
    """
    global _GPU_HANDLE, _GPU_INIT_FAILED
    if _GPU_HANDLE is not None:
        return True
    if _GPU_INIT_FAILED:
        return False
    if not _HAS_PYNVML:
        _GPU_INIT_FAILED = True
        return False
    try:
        pynvml.nvmlInit()
        _GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
        return True
    except Exception as exc:
        logger.info("NVML init failed; GPU metrics will report placeholder values: %s", exc)
        _GPU_INIT_FAILED = True
        return False


def shutdown_gpu() -> None:
    """Release the NVML handle.

    Idempotent: safe to call regardless of whether init_gpu() succeeded
    or was ever called. Called from the server's shutdown path so NVML
    state is cleaned up before the process exits.
    """
    global _GPU_HANDLE
    if _HAS_PYNVML and _GPU_HANDLE is not None:
        try:
            pynvml.nvmlShutdown()
        except Exception as exc:
            logger.debug("NVML shutdown raised (non-fatal): %s", exc)
        _GPU_HANDLE = None


def reset_for_tests() -> None:
    """Reset the module-level GPU and process-uptime state for test isolation.

    Tests that exercise init_gpu() failure paths must reset the
    "already failed" flag between cases. Not exported for production
    use; the production path treats _GPU_INIT_FAILED (and, identically,
    _PROCESS_START_TIME_READ_FAILED) as a one-way door.
    """
    global _GPU_HANDLE, _GPU_INIT_FAILED, _PROCESS_START_TIME, _PROCESS_START_TIME_READ_FAILED
    _GPU_HANDLE = None
    _GPU_INIT_FAILED = False
    _PROCESS_START_TIME = None
    _PROCESS_START_TIME_READ_FAILED = False


def process_uptime_seconds(now: float | None = None) -> float | None:
    """Seconds since this backend process started.

    Sourced from `psutil.Process().create_time()` -- the BACKEND PROCESS's
    own start time, not the container's and not the host's. `psutil.boot_time()`
    would measure host uptime and is deliberately never called anywhere in
    this module (see the module docstring).

    The start time is read once and cached in the module-level
    `_PROCESS_START_TIME` global, mirroring the `_GPU_HANDLE` singleton
    pattern above -- it cannot change for the life of the process, so
    re-reading on every 5s tick would be wasted work. `reset_for_tests()`
    clears the cache for test isolation.

    Args:
        now: Injectable current time (unix seconds). Defaults to
            `time.time()`. This is the test seam: tests inject a fixed
            `(create_time, now)` pair rather than sleeping in real time
            (tests/CLAUDE.md forbids `time.sleep()` in tests as
            non-deterministic and slow).

    Returns:
        `now - create_time`, or `None` if the process start time could not
        be read (e.g. a permissions error, or the process object is gone).
        Never falls back to `0.0` -- `0.0` is a measurement and `None` is
        the absence of one (ADR-017 v1.2.0's null-never-zero rule). Logs
        one WARNING the first time the read fails; it is not retried on
        subsequent ticks (see `_PROCESS_START_TIME_READ_FAILED` above).
    """
    global _PROCESS_START_TIME, _PROCESS_START_TIME_READ_FAILED

    if _PROCESS_START_TIME is None and not _PROCESS_START_TIME_READ_FAILED:
        try:
            _PROCESS_START_TIME = psutil.Process().create_time()
        except Exception as exc:
            logger.warning(
                "Failed to read backend process start time; uptime_seconds will be null: %s",
                exc,
            )
            _PROCESS_START_TIME_READ_FAILED = True

    if _PROCESS_START_TIME is None:
        return None

    if now is None:
        now = time.time()

    return now - _PROCESS_START_TIME


def collect_metrics() -> SystemMetrics:
    """Snapshot CPU + RAM + GPU + process uptime into a SystemMetrics record.

    Non-blocking. psutil.cpu_percent(interval=None) returns the CPU
    utilization since the previous call; the first invocation after
    process start always returns 0.0 (no prior baseline). That is
    acceptable for a periodic 5s emit -- the second emit onward
    carries real values.
    """
    cpu_percent = psutil.cpu_percent(interval=None)
    cpu_cores = psutil.cpu_count(logical=True) or 1

    vm = psutil.virtual_memory()
    ram = RAMMetrics(
        used_gb=vm.used / _GIB,
        total_gb=vm.total / _GIB,
        percent=float(vm.percent),
    )

    gpu = _collect_gpu()

    return SystemMetrics(
        timestamp=time.time() * 1000,
        cpu=CPUMetrics(percent=float(cpu_percent), cores=int(cpu_cores)),
        ram=ram,
        gpu=gpu,
        uptime_seconds=process_uptime_seconds(),
    )


def _collect_gpu() -> GPUMetrics:
    """Sample GPU via NVML, or return placeholder when unavailable.

    Two failure modes return placeholders:
    1. NVML never initialized successfully (init_gpu() returned False).
    2. NVML initialized but a single sample raised (driver crash,
       container lost device access mid-run, etc.). The handle stays
       set -- a subsequent tick retries. We emit a marker name
       ("error") so the FE can distinguish this from "no GPU".
    """
    if _GPU_HANDLE is None:
        return GPUMetrics(
            name="none",
            utilization_percent=0.0,
            vram_used_gb=0.0,
            vram_total_gb=0.0,
            temperature_c=None,
        )
    try:
        name_raw = pynvml.nvmlDeviceGetName(_GPU_HANDLE)
        name = name_raw.decode("utf-8") if isinstance(name_raw, bytes) else str(name_raw)
        util = pynvml.nvmlDeviceGetUtilizationRates(_GPU_HANDLE)
        mem = pynvml.nvmlDeviceGetMemoryInfo(_GPU_HANDLE)
        try:
            temp_raw = pynvml.nvmlDeviceGetTemperature(_GPU_HANDLE, pynvml.NVML_TEMPERATURE_GPU)
            temp: float | None = float(temp_raw)
        except Exception:
            temp = None
        return GPUMetrics(
            name=name,
            utilization_percent=float(util.gpu),
            vram_used_gb=float(mem.used) / _GIB,
            vram_total_gb=float(mem.total) / _GIB,
            temperature_c=temp,
        )
    except Exception as exc:
        logger.debug("NVML sample failed; emitting error placeholder: %s", exc)
        return GPUMetrics(
            name="error",
            utilization_percent=0.0,
            vram_used_gb=0.0,
            vram_total_gb=0.0,
            temperature_c=None,
        )
