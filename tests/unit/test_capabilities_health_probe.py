"""Tests for `CapabilitiesProbe`, the `/health` check over the capability host's statuses.

The probe reads `statuses()` only, which each capability computes from memory, so a fake host
holding a status list is the whole dependency.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from backend.capabilities.contracts import CapabilityStatus
from backend.health import CapabilitiesProbe, ProbeFailed, ProbeUnavailable

AT = datetime(2026, 1, 5, 7, 0, tzinfo=UTC)


class FakeHost:
    """Answers `statuses()` from a list a test sets."""

    def __init__(self, *statuses: CapabilityStatus) -> None:
        self.current = list(statuses)
        self.reads = 0

    def statuses(self) -> list[CapabilityStatus]:
        self.reads += 1
        return list(self.current)


def _status(name: str, state: str, reason: str | None = None, *, dead: bool = False):
    return CapabilityStatus(name, state, reason, AT, task_dead=dead)  # type: ignore[arg-type]


async def _reason(probe: CapabilitiesProbe) -> str:
    with pytest.raises(ProbeFailed) as raised:
        await probe.check()
    return str(raised.value)


class TestShape:
    def test_name_severity_and_restart_repairs(self):
        probe = CapabilitiesProbe(lambda: FakeHost())

        assert probe.name == "capabilities"
        assert probe.severity == "degrades"
        assert probe.restart_repairs is False
        assert probe.timeout_seconds > 0


class TestUp:
    @pytest.mark.asyncio
    async def test_up_when_every_capability_is_available_or_degraded(self):
        host = FakeHost(
            _status("clock", "available"),
            _status("alarms", "degraded", "1 stored alarm quarantined; see the log"),
            _status("lights", "available"),
        )

        await CapabilitiesProbe(lambda: host).check()

        assert host.reads == 1

    @pytest.mark.asyncio
    async def test_a_disabled_capability_is_not_counted(self):
        host = FakeHost(_status("clock", "available"), _status("music", "disabled", "off"))

        await CapabilitiesProbe(lambda: host).check()


class TestDown:
    @pytest.mark.asyncio
    async def test_a_dead_task_behind_an_unavailable_capability_reads_task_dead(self):
        host = FakeHost(
            _status("clock", "available"),
            _status("lights", "unavailable", "Home Assistant unreachable"),
            _status("alarms", "unavailable", "alarm scheduler dead", dead=True),
        )

        assert await _reason(CapabilitiesProbe(lambda: host)) == "task_dead"

    @pytest.mark.asyncio
    async def test_any_other_unavailable_capability_reads_unreachable(self):
        host = FakeHost(
            _status("clock", "available"),
            _status("music", "unavailable", "not_authorized: run the authorize CLI"),
        )

        assert await _reason(CapabilitiesProbe(lambda: host)) == "unreachable"

    @pytest.mark.asyncio
    async def test_task_dead_on_an_available_capability_does_not_turn_the_probe_down(self):
        host = FakeHost(_status("clock", "available", dead=True))

        await CapabilitiesProbe(lambda: host).check()


class TestAbsent:
    @pytest.mark.asyncio
    async def test_no_host_is_not_measured(self):
        with pytest.raises(ProbeUnavailable) as raised:
            await CapabilitiesProbe(lambda: None).check()

        assert str(raised.value) == "disabled_by_config"

    @pytest.mark.asyncio
    async def test_a_host_with_only_disabled_capabilities_is_not_measured(self):
        host = FakeHost(_status("music", "disabled"))

        with pytest.raises(ProbeUnavailable) as raised:
            await CapabilitiesProbe(lambda: host).check()

        assert str(raised.value) == "disabled_by_config"

    @pytest.mark.asyncio
    async def test_the_host_is_read_at_each_check(self):
        hosts: list[FakeHost | None] = [None]
        probe = CapabilitiesProbe(lambda: hosts[0])

        with pytest.raises(ProbeUnavailable):
            await probe.check()
        hosts[0] = FakeHost(_status("clock", "unavailable", "MIST_TIMEZONE unset"))

        assert await _reason(probe) == "unreachable"
