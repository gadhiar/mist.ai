"""Unit tests for the system_status periodic emit task in backend.server.

Exercises the task in isolation by patching backend.system_metrics
.collect_metrics() to return a controlled SystemMetrics. The real
broadcaster is replaced with a controlled message_queue that captures
each enqueued payload for assertion.
"""

from __future__ import annotations

import asyncio
import contextlib
import json

import pytest

from backend import system_metrics
from backend.system_metrics import (
    CPUMetrics,
    GPUMetrics,
    RAMMetrics,
    SystemMetrics,
)


def _fake_snapshot() -> SystemMetrics:
    return SystemMetrics(
        timestamp=1234567890.0,
        cpu=CPUMetrics(percent=12.5, cores=16),
        ram=RAMMetrics(used_gb=8.0, total_gb=32.0, percent=25.0),
        gpu=GPUMetrics(
            name="FakeGPU 4080",
            utilization_percent=42.0,
            vram_used_gb=4.0,
            vram_total_gb=12.0,
            temperature_c=65.0,
        ),
    )


class TestSystemStatusLoopEmits:
    """The loop pushes system_status payloads onto the message queue."""

    @pytest.mark.asyncio
    async def test_emits_at_interval(self, monkeypatch):
        """With interval=0.05s, ~3 ticks land in the queue within 0.25s."""
        from backend import server

        # Capture emits onto a private queue (the module-level message_queue
        # is the real broadcaster channel; we substitute it for isolation).
        captured: asyncio.Queue = asyncio.Queue()
        monkeypatch.setattr(server, "message_queue", captured)
        monkeypatch.setattr(system_metrics, "collect_metrics", _fake_snapshot)

        task = asyncio.create_task(server.system_status_loop(interval_seconds=0.05))
        await asyncio.sleep(0.18)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        # Drain captured queue
        emits: list[dict] = []
        while not captured.empty():
            emits.append(json.loads(await captured.get()))

        # 0.18s / 0.05s -> 3 sleeps, 3 emits (the loop sleeps FIRST then emits)
        assert 2 <= len(emits) <= 4, f"expected ~3 emits at 0.05s cadence, got {len(emits)}"
        for emit in emits:
            assert emit["type"] == "system_status"
            assert emit["cpu"] == {"percent": 12.5, "cores": 16}
            assert emit["ram"]["used_gb"] == 8.0
            assert emit["ram"]["total_gb"] == 32.0
            assert emit["gpu"]["name"] == "FakeGPU 4080"
            assert emit["gpu"]["temperature_c"] == 65.0

    @pytest.mark.asyncio
    async def test_handles_collector_failure(self, monkeypatch):
        """When collect_metrics raises, the loop logs + continues; subsequent
        ticks resume emitting once the collector recovers.
        """
        from backend import server

        captured: asyncio.Queue = asyncio.Queue()
        monkeypatch.setattr(server, "message_queue", captured)

        # First two ticks raise; subsequent ticks return a snapshot.
        state = {"calls": 0}

        def flaky_collect():
            state["calls"] += 1
            if state["calls"] <= 2:
                raise RuntimeError("collector hiccup")
            return _fake_snapshot()

        monkeypatch.setattr(system_metrics, "collect_metrics", flaky_collect)

        task = asyncio.create_task(server.system_status_loop(interval_seconds=0.05))
        await asyncio.sleep(0.25)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        # Collector was called multiple times despite raising on the first 2
        assert state["calls"] >= 3, f"loop must keep calling after exceptions; got {state['calls']}"

        # At least one emit landed after the collector recovered
        emits: list[dict] = []
        while not captured.empty():
            emits.append(json.loads(await captured.get()))
        assert len(emits) >= 1
        assert emits[0]["type"] == "system_status"

    @pytest.mark.asyncio
    async def test_payload_shape_matches_adr017(self, monkeypatch):
        """Single emit carries the exact ADR-017 system_status shape:
        type / timestamp / cpu / ram / gpu top-level keys; each block has
        the documented sub-keys.
        """
        from backend import server

        captured: asyncio.Queue = asyncio.Queue()
        monkeypatch.setattr(server, "message_queue", captured)
        monkeypatch.setattr(system_metrics, "collect_metrics", _fake_snapshot)

        task = asyncio.create_task(server.system_status_loop(interval_seconds=0.02))
        await asyncio.sleep(0.05)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        emit = json.loads(await captured.get())
        assert set(emit.keys()) == {"type", "timestamp", "cpu", "ram", "gpu"}
        assert set(emit["cpu"].keys()) == {"percent", "cores"}
        assert set(emit["ram"].keys()) == {"used_gb", "total_gb", "percent"}
        assert set(emit["gpu"].keys()) == {
            "name",
            "utilization_percent",
            "vram_used_gb",
            "vram_total_gb",
            "temperature_c",
        }


class TestSystemStatusConfig:
    """SystemStatusConfig defaults + env-var overrides via load_config()."""

    def test_defaults_match_adr017(self):
        from backend.config import SystemStatusConfig

        cfg = SystemStatusConfig()
        assert cfg.enabled is True
        assert cfg.interval_seconds == 5
        assert cfg.gpu_enabled is True

    def test_env_overrides_via_load_config(self, monkeypatch):
        from backend import config as config_module

        monkeypatch.setenv("MIST_SYSTEM_STATUS_ENABLED", "false")
        monkeypatch.setenv("MIST_SYSTEM_STATUS_INTERVAL_SECONDS", "10")
        monkeypatch.setenv("MIST_SYSTEM_STATUS_GPU_ENABLED", "false")

        cfg = config_module.load_config()
        assert cfg.system_status.enabled is False
        assert cfg.system_status.interval_seconds == 10
        assert cfg.system_status.gpu_enabled is False

    def test_invalid_bool_env_falls_back_to_default(self, monkeypatch):
        """A non-'true'/'false' env value defaults to False per the parser
        (strict-match by design; ambiguous strings should not flip the flag).
        """
        from backend import config as config_module

        monkeypatch.setenv("MIST_SYSTEM_STATUS_ENABLED", "yes")
        cfg = config_module.load_config()
        # "yes" is not "true"; the parser treats it as False
        assert cfg.system_status.enabled is False
