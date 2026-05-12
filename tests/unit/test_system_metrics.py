"""Unit tests for backend.system_metrics collector.

Validates the dataclass shape + range invariants + graceful
degradation paths. Real NVML calls are not exercised here (CI / dev
host may have no GPU); the GPU path tests monkeypatch the module-level
handle to None or to a fake to assert downstream behavior.
"""

from __future__ import annotations

import pytest

from backend import system_metrics
from backend.system_metrics import (
    CPUMetrics,
    GPUMetrics,
    RAMMetrics,
    SystemMetrics,
    collect_metrics,
    init_gpu,
    reset_for_tests,
    shutdown_gpu,
)


@pytest.fixture(autouse=True)
def _reset_module_state():
    """Reset NVML module state between cases so init failure flags don't leak."""
    reset_for_tests()
    yield
    reset_for_tests()


class TestCollectMetricsShape:
    """collect_metrics() returns a SystemMetrics with valid sub-dataclasses."""

    def test_returns_system_metrics_instance(self):
        result = collect_metrics()
        assert isinstance(result, SystemMetrics)
        assert isinstance(result.cpu, CPUMetrics)
        assert isinstance(result.ram, RAMMetrics)
        assert isinstance(result.gpu, GPUMetrics)

    def test_timestamp_is_unix_ms(self):
        import time as _t

        before = _t.time() * 1000
        result = collect_metrics()
        after = _t.time() * 1000
        # Timestamp is unix ms, between before and after.
        assert before <= result.timestamp <= after

    def test_cpu_percent_in_range(self):
        result = collect_metrics()
        assert 0.0 <= result.cpu.percent <= 100.0

    def test_cpu_cores_positive(self):
        result = collect_metrics()
        assert result.cpu.cores >= 1

    def test_ram_used_le_total(self):
        result = collect_metrics()
        assert result.ram.used_gb >= 0
        assert result.ram.total_gb > 0
        assert result.ram.used_gb <= result.ram.total_gb

    def test_ram_percent_in_range(self):
        result = collect_metrics()
        assert 0.0 <= result.ram.percent <= 100.0


class TestGPUPlaceholderWhenUnavailable:
    """When NVML is unavailable, GPU block emits a placeholder."""

    def test_gpu_name_none_when_handle_unset(self, monkeypatch):
        # Force the no-NVML path
        monkeypatch.setattr(system_metrics, "_GPU_HANDLE", None)
        result = collect_metrics()
        assert result.gpu.name == "none"
        assert result.gpu.utilization_percent == 0.0
        assert result.gpu.vram_used_gb == 0.0
        assert result.gpu.vram_total_gb == 0.0
        assert result.gpu.temperature_c is None


class TestInitGPU:
    """init_gpu() idempotency and graceful degradation."""

    def test_init_gpu_idempotent_when_no_gpu(self, monkeypatch):
        # Simulate "NVML not available" by forcing _HAS_PYNVML to False
        monkeypatch.setattr(system_metrics, "_HAS_PYNVML", False)
        reset_for_tests()
        first = init_gpu()
        second = init_gpu()
        assert first is False
        assert second is False

    def test_init_gpu_idempotent_on_repeated_failure(self, monkeypatch):
        """After a failed init, subsequent calls return False without retrying."""
        retry_count = {"n": 0}

        def fake_init():
            retry_count["n"] += 1
            raise RuntimeError("no NVIDIA driver loaded")

        if system_metrics._HAS_PYNVML:
            import pynvml as _pynvml

            monkeypatch.setattr(_pynvml, "nvmlInit", fake_init)

        reset_for_tests()
        first = init_gpu()
        second = init_gpu()
        third = init_gpu()
        assert first is False
        assert second is False
        assert third is False
        if system_metrics._HAS_PYNVML:
            # Only one real nvmlInit attempt; subsequent calls short-circuit
            # via the _GPU_INIT_FAILED flag.
            assert retry_count["n"] == 1

    def test_init_gpu_returns_true_on_success(self, monkeypatch):
        """When NVML succeeds, init_gpu returns True and the handle is set."""
        if not system_metrics._HAS_PYNVML:
            pytest.skip("pynvml not installed in this environment")

        import pynvml as _pynvml

        sentinel_handle = object()
        monkeypatch.setattr(_pynvml, "nvmlInit", lambda: None)
        monkeypatch.setattr(_pynvml, "nvmlDeviceGetHandleByIndex", lambda idx: sentinel_handle)

        reset_for_tests()
        assert init_gpu() is True
        assert system_metrics._GPU_HANDLE is sentinel_handle
        # Second call no-ops and still returns True
        assert init_gpu() is True


class TestShutdownGPU:
    """shutdown_gpu() idempotency."""

    def test_shutdown_gpu_idempotent_when_uninitialized(self):
        # Never called init_gpu(); shutdown should not raise.
        shutdown_gpu()
        shutdown_gpu()

    def test_shutdown_gpu_clears_handle(self, monkeypatch):
        if not system_metrics._HAS_PYNVML:
            pytest.skip("pynvml not installed in this environment")

        import pynvml as _pynvml

        sentinel_handle = object()
        monkeypatch.setattr(_pynvml, "nvmlInit", lambda: None)
        monkeypatch.setattr(_pynvml, "nvmlDeviceGetHandleByIndex", lambda idx: sentinel_handle)
        monkeypatch.setattr(_pynvml, "nvmlShutdown", lambda: None)

        reset_for_tests()
        init_gpu()
        assert system_metrics._GPU_HANDLE is sentinel_handle
        shutdown_gpu()
        assert system_metrics._GPU_HANDLE is None


class TestGPUSamplePath:
    """GPU sample with a fake NVML handle returns parsed values."""

    def test_collect_gpu_returns_real_payload_with_fake_handle(self, monkeypatch):
        if not system_metrics._HAS_PYNVML:
            pytest.skip("pynvml not installed in this environment")

        import pynvml as _pynvml

        class _FakeUtil:
            gpu = 42

        class _FakeMem:
            used = 4 * (1024**3)
            total = 12 * (1024**3)

        sentinel_handle = object()
        monkeypatch.setattr(system_metrics, "_GPU_HANDLE", sentinel_handle)
        monkeypatch.setattr(_pynvml, "nvmlDeviceGetName", lambda h: b"FakeGPU 4080")
        monkeypatch.setattr(_pynvml, "nvmlDeviceGetUtilizationRates", lambda h: _FakeUtil())
        monkeypatch.setattr(_pynvml, "nvmlDeviceGetMemoryInfo", lambda h: _FakeMem())
        monkeypatch.setattr(_pynvml, "nvmlDeviceGetTemperature", lambda h, kind: 65)

        result = collect_metrics()
        assert result.gpu.name == "FakeGPU 4080"
        assert result.gpu.utilization_percent == 42.0
        assert result.gpu.vram_used_gb == 4.0
        assert result.gpu.vram_total_gb == 12.0
        assert result.gpu.temperature_c == 65.0

    def test_collect_gpu_returns_error_placeholder_on_sample_failure(self, monkeypatch):
        """If NVML sampling raises (e.g., driver crash mid-run), the GPU
        block emits an 'error' marker placeholder rather than crashing the
        emit task.
        """
        if not system_metrics._HAS_PYNVML:
            pytest.skip("pynvml not installed in this environment")

        import pynvml as _pynvml

        sentinel_handle = object()
        monkeypatch.setattr(system_metrics, "_GPU_HANDLE", sentinel_handle)

        def _boom(*args, **kwargs):
            raise RuntimeError("NVML lost device handle")

        monkeypatch.setattr(_pynvml, "nvmlDeviceGetName", _boom)

        result = collect_metrics()
        assert result.gpu.name == "error"
        assert result.gpu.utilization_percent == 0.0
        assert result.gpu.temperature_c is None
