"""Tests for `GpuYield`, the gaming-mode marker watcher.

Every test uses a marker in `tmp_path` and recording callbacks. The backoff
test drives `run_forever` with an injected sleep that advances an injected
monotonic clock, so 240 simulated seconds take no real time.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path

import pytest

from backend.gpu_yield import DEFAULT_POLL_S, GpuYield, gpu_yield_poll_s

_SECRET = "CUDA out of memory: tried to allocate 1.2 GiB"


class Recorder:
    """Callbacks that record when, where and in which state they ran."""

    def __init__(self) -> None:
        self.gpu: GpuYield | None = None
        self.yields: list[str] = []
        self.resumes: list[str] = []
        self.threads: list[int] = []
        self.resume_failures_left = 0
        self.resume_error: type[BaseException] = RuntimeError

    def on_yield(self) -> None:
        self.threads.append(threading.get_ident())
        self.yields.append(self.gpu.state())

    def on_resume(self) -> None:
        self.threads.append(threading.get_ident())
        self.resumes.append(self.gpu.state())
        if self.resume_failures_left > 0:
            self.resume_failures_left -= 1
            raise self.resume_error(_SECRET)


def _build(marker: Path, recorder: Recorder, **kwargs) -> GpuYield:
    gpu = GpuYield(marker, recorder.on_yield, recorder.on_resume, poll_s=2.0, **kwargs)
    recorder.gpu = gpu
    return gpu


def _write_marker(marker: Path, text: str = '{"v": 1, "since": "2026-10-23T00:00:00Z"}') -> None:
    marker.write_text(text, encoding="utf-8")


@pytest.fixture
def marker(tmp_path) -> Path:
    return tmp_path / "gpu-yielded"


class TestConstruction:
    @pytest.mark.asyncio
    async def test_marker_present_at_construction_is_yielded_without_callbacks(self, marker):
        _write_marker(marker)
        recorder = Recorder()

        gpu = _build(marker, recorder)
        await gpu.poll_once()

        assert gpu.state() == "yielded"
        assert gpu.is_yielded() is True
        assert recorder.yields == []
        assert recorder.resumes == []

    def test_no_marker_at_construction_is_normal(self, marker):
        gpu = _build(marker, Recorder())

        assert gpu.state() == "normal"
        assert gpu.is_yielded() is False

    def test_unparsable_marker_counts_as_present(self, marker):
        _write_marker(marker, "{not json")

        gpu = _build(marker, Recorder())

        assert gpu.state() == "yielded"


class TestTransitions:
    @pytest.mark.asyncio
    async def test_marker_that_appears_calls_on_yield_once_in_yielded_state(self, marker):
        recorder = Recorder()
        gpu = _build(marker, recorder)

        _write_marker(marker)
        await gpu.poll_once()
        await gpu.poll_once()

        assert recorder.yields == ["yielded"]
        assert gpu.state() == "yielded"

    @pytest.mark.asyncio
    async def test_marker_that_goes_calls_on_resume_once_then_normal(self, marker):
        _write_marker(marker)
        recorder = Recorder()
        gpu = _build(marker, recorder)

        marker.unlink()
        await gpu.poll_once()
        await gpu.poll_once()

        assert recorder.resumes == ["yielded"]
        assert gpu.state() == "normal"

    @pytest.mark.asyncio
    async def test_failed_resume_sets_resume_failed(self, marker):
        _write_marker(marker)
        recorder = Recorder()
        recorder.resume_failures_left = 1
        gpu = _build(marker, recorder)

        marker.unlink()
        await gpu.poll_once()

        assert gpu.state() == "resume_failed"
        assert gpu.is_yielded() is True

    @pytest.mark.asyncio
    async def test_os_error_on_resume_also_sets_resume_failed(self, marker):
        _write_marker(marker)
        recorder = Recorder()
        recorder.resume_failures_left = 1
        recorder.resume_error = OSError
        gpu = _build(marker, recorder)

        marker.unlink()
        await gpu.poll_once()

        assert gpu.state() == "resume_failed"

    @pytest.mark.asyncio
    async def test_marker_reappearing_during_resume_failed_yields_again(self, marker):
        _write_marker(marker)
        recorder = Recorder()
        recorder.resume_failures_left = 1
        gpu = _build(marker, recorder)
        marker.unlink()
        await gpu.poll_once()
        assert gpu.state() == "resume_failed"

        _write_marker(marker)
        await gpu.poll_once()

        assert gpu.state() == "yielded"
        assert recorder.yields == ["yielded"]

    @pytest.mark.asyncio
    async def test_callbacks_never_run_on_the_event_loop_thread(self, marker):
        recorder = Recorder()
        gpu = _build(marker, recorder)
        loop_thread = threading.get_ident()

        _write_marker(marker)
        await gpu.poll_once()
        marker.unlink()
        await gpu.poll_once()

        assert len(recorder.threads) == 2
        assert loop_thread not in recorder.threads


class SimulationEndedError(Exception):
    """Ends `run_forever` from inside the injected sleep."""


class SimulatedTime:
    """An injected sleep that advances an injected monotonic clock."""

    def __init__(self, limit_s: float) -> None:
        self.now = 0.0
        self.limit_s = limit_s
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds
        if self.now > self.limit_s:
            raise SimulationEndedError


class TestResumeBackoff:
    @pytest.mark.asyncio
    async def test_retries_back_off_from_poll_s_to_60s_then_recover(self, marker, caplog):
        _write_marker(marker)
        time = SimulatedTime(limit_s=300.0)
        recorder = Recorder()
        recorder.resume_failures_left = 8
        resume_times: list[float] = []
        original_on_resume = recorder.on_resume

        def timed_on_resume() -> None:
            resume_times.append(time.now)
            original_on_resume()

        gpu = GpuYield(
            marker,
            recorder.on_yield,
            timed_on_resume,
            poll_s=2.0,
            sleep=time.sleep,
            clock=time.clock,
        )
        recorder.gpu = gpu
        marker.unlink()

        with (
            caplog.at_level(logging.INFO, logger="backend.gpu_yield"),
            pytest.raises(SimulationEndedError),
        ):
            await gpu.run_forever()

        gaps = [later - earlier for earlier, later in zip(resume_times, resume_times[1:])]
        assert gaps[:7] == [2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0]
        assert len(resume_times) == 9
        assert gpu.state() == "normal"
        assert set(time.sleeps) == {2.0}

        failure_lines = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(failure_lines) == 1
        assert "RuntimeError" in failure_lines[0].getMessage()
        assert failure_lines[0].exc_info is None
        recovery_lines = [r for r in caplog.records if "recovered" in r.getMessage()]
        assert len(recovery_lines) == 1
        assert all(_SECRET not in r.getMessage() for r in caplog.records)


class TestPollInterval:
    def test_default_is_two_seconds(self, monkeypatch):
        monkeypatch.delenv("MIST_GPU_YIELD_POLL_S", raising=False)

        assert gpu_yield_poll_s() == DEFAULT_POLL_S == 2.0

    def test_reads_a_positive_number(self, monkeypatch):
        monkeypatch.setenv("MIST_GPU_YIELD_POLL_S", "0.5")

        assert gpu_yield_poll_s() == 0.5

    @pytest.mark.parametrize(
        "raw",
        [
            pytest.param("zero", id="not-a-number"),
            pytest.param("0", id="zero"),
            pytest.param("-1", id="negative"),
        ],
    )
    def test_bad_value_is_logged_and_replaced_by_the_default(self, monkeypatch, caplog, raw):
        monkeypatch.setenv("MIST_GPU_YIELD_POLL_S", raw)

        with caplog.at_level(logging.ERROR, logger="backend.gpu_yield"):
            value = gpu_yield_poll_s()

        assert value == DEFAULT_POLL_S
        assert any("MIST_GPU_YIELD_POLL_S" in r.getMessage() for r in caplog.records)


def test_run_forever_polls_first_then_every_poll_s(marker):
    """A marker that came or went during the model load is acted on at once."""
    recorder = Recorder()
    time = SimulatedTime(limit_s=1.0)
    gpu = GpuYield(
        marker,
        recorder.on_yield,
        recorder.on_resume,
        poll_s=2.0,
        sleep=time.sleep,
        clock=time.clock,
    )
    recorder.gpu = gpu
    _write_marker(marker)

    with pytest.raises(SimulationEndedError):
        asyncio.run(gpu.run_forever())

    assert recorder.yields == ["yielded"]
    assert time.sleeps == [2.0]
