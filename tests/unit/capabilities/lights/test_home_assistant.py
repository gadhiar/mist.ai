"""Tests for `HomeAssistantLightsProvider` over `McpConnection` and the fake Home Assistant.

The fake models the documented Home Assistant intents, listed bare (before 2026.11) or with the
source prefix Home Assistant 2026.11 listed on 26 Nov (`intent__HassTurnOn`), and the
`GetLiveContext` layout that server returned; the input schemas are still unverified (see
`fake_home_assistant.py`).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

import pytest

from backend.capabilities.lights.capability import LightAdjustment
from backend.capabilities.lights.catalog import LightTarget
from backend.capabilities.lights.home_assistant import (
    CATALOG_REFRESH_S,
    STATUS_TEXT_MAX_CHARS,
    HomeAssistantLightsProvider,
)
from backend.capabilities.mcp.connection import BackoffPolicy, McpConnection
from backend.capabilities.mcp.transport import McpToolResult
from backend.errors import (
    CapabilityInvocationError,
    CapabilityUnavailableError,
    McpTransportError,
)
from tests.unit.capabilities.fakes import FakeClock
from tests.unit.capabilities.mcp.fake_home_assistant import (
    GET_STATE,
    LIGHT_SET,
    LIVE_CONTEXT,
    TURN_OFF,
    TURN_ON,
    FakeHomeAssistant,
    prefixed_name,
    tool_error,
    unexpected_error,
)

LOGGER = "backend.capabilities.lights.home_assistant"
TIMEOUT_S = 4.5

LIVE_CONTEXT_TEXT = (
    "Live Context: An overview of the areas and the devices in this smart home:\n"
    "- names: Bedroom Lamp\n"
    "  domain: light\n"
    "  state: 'off'\n"
    "  areas: Bedroom\n"
    "- names: Ceiling Light\n"
    "  domain: light\n"
    "  state: 'on'\n"
    "  areas: Kitchen\n"
)


@dataclass
class Rig:
    server: FakeHomeAssistant
    clock: FakeClock
    connection: McpConnection
    provider: HomeAssistantLightsProvider

    def calls_to(self, name: str) -> list[dict[str, object]]:
        return [dict(call.arguments) for call in self.server.calls if call.name == name]


async def _until(predicate: Callable[[], bool], timeout_s: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    give_up_at = loop.time() + timeout_s
    while not predicate():
        if loop.time() > give_up_at:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.001)


def _build(server: FakeHomeAssistant, clock: FakeClock) -> Rig:
    connection = McpConnection(server.transport, clock, BackoffPolicy(jitter=lambda d: d))
    provider = HomeAssistantLightsProvider(connection, clock, call_timeout_s=TIMEOUT_S)
    return Rig(server, clock, connection, provider)


@contextlib.asynccontextmanager
async def _running(
    server: FakeHomeAssistant | None = None,
    *,
    connect: bool = True,
    live_context: str = LIVE_CONTEXT_TEXT,
) -> AsyncIterator[Rig]:
    server = server or FakeHomeAssistant(include_get_state=True)
    server.live_context = live_context
    rig = _build(server, FakeClock())
    await rig.provider.start()
    try:
        if connect:
            await _until(lambda: rig.connection.state().state == "available")
            # One poll of the refresh loop, so the on-connect refresh has run.
            await rig.clock.advance(1.0)
        yield rig
    finally:
        await rig.provider.stop()


class TestConstruction:
    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_call_timeout_must_be_finite_and_positive(self, bad):
        server = FakeHomeAssistant()
        clock = FakeClock()
        connection = McpConnection(server.transport, clock, BackoffPolicy(jitter=lambda d: d))
        with pytest.raises(ValueError, match="call_timeout_s"):
            HomeAssistantLightsProvider(connection, clock, call_timeout_s=bad)


class TestPower:
    @pytest.mark.asyncio
    async def test_power_on_a_known_light_uses_its_name_and_the_light_domain(self):
        async with _running() as rig:
            target = rig.provider.resolve("bedroom lamp")
            assert target == LightTarget(kind="name", value="Bedroom Lamp")
            await rig.provider.power(target, True)
            assert rig.calls_to(TURN_ON) == [{"name": "Bedroom Lamp", "domain": ["light"]}]

    @pytest.mark.asyncio
    async def test_power_off_a_known_area_uses_its_area(self):
        async with _running() as rig:
            target = rig.provider.resolve("KITCHEN")
            await rig.provider.power(target, False)
            assert rig.calls_to(TURN_OFF) == [{"area": "Kitchen", "domain": ["light"]}]

    @pytest.mark.asyncio
    async def test_an_unknown_target_does_not_resolve(self):
        async with _running() as rig:
            assert rig.provider.resolve("garage") is None

    @pytest.mark.asyncio
    async def test_with_no_catalog_a_target_passes_through_as_a_name(self):
        server = FakeHomeAssistant(omit=[LIVE_CONTEXT], include_get_state=True)
        async with _running(server) as rig:
            assert rig.provider.catalog().is_empty
            assert rig.provider.resolve("porch") == LightTarget(kind="name", value="porch")

    @pytest.mark.asyncio
    async def test_a_tool_error_raises_with_home_assistants_text(self):
        async with _running() as rig:
            rig.server.call_script.append(tool_error("Unable to find a device named Attic"))
            with pytest.raises(CapabilityInvocationError, match="Unable to find a device named"):
                await rig.provider.power(LightTarget(kind="name", value="Attic"), True)

    @pytest.mark.asyncio
    async def test_while_disconnected_a_call_is_unavailable(self):
        async with _running(connect=False) as rig:
            await rig.provider.stop()
            with pytest.raises(CapabilityUnavailableError) as raised:
                await rig.provider.power(LightTarget(kind="area", value="Kitchen"), True)

            # The connection's own reason, not "Home Assistant lacks HassTurnOn".
            assert str(raised.value) == "stopped"


class TestAdjust:
    @pytest.mark.asyncio
    async def test_adjust_maps_to_light_set_slots(self):
        async with _running() as rig:
            target = rig.provider.resolve("kitchen")
            await rig.provider.adjust(
                target, LightAdjustment(brightness_pct=30, color="red", kelvin=2700)
            )
            assert rig.calls_to(LIGHT_SET) == [
                {
                    "area": "Kitchen",
                    "domain": ["light"],
                    "brightness": 30,
                    "color": "red",
                    "temperature": 2700,
                }
            ]

    @pytest.mark.asyncio
    async def test_adjust_sends_only_the_settings_given(self):
        async with _running() as rig:
            await rig.provider.adjust(
                LightTarget(kind="name", value="Bedroom Lamp"), LightAdjustment(brightness_pct=0)
            )
            assert rig.calls_to(LIGHT_SET) == [
                {"name": "Bedroom Lamp", "domain": ["light"], "brightness": 0}
            ]

    @pytest.mark.asyncio
    async def test_without_light_set_adjust_is_unavailable_naming_the_tool(self):
        async with _running(FakeHomeAssistant(omit=[LIGHT_SET])) as rig:
            with pytest.raises(CapabilityUnavailableError, match=LIGHT_SET):
                await rig.provider.adjust(
                    LightTarget(kind="area", value="Kitchen"), LightAdjustment(brightness_pct=5)
                )
            assert rig.calls_to(LIGHT_SET) == []


class TestReadStatus:
    @pytest.mark.asyncio
    async def test_status_reads_live_context_and_lists_the_targets_lights(self):
        async with _running() as rig:
            reading = await rig.provider.read_status(LightTarget(kind="area", value="Kitchen"))
            assert reading.text == LIVE_CONTEXT_TEXT.strip()
            assert not reading.truncated
            assert reading.lights is not None
            assert [light.names for light in reading.lights] == [("Ceiling Light",)]
            everything = await rig.provider.read_status(None)
            assert everything.lights is not None
            assert len(everything.lights) == 2

    @pytest.mark.asyncio
    async def test_status_text_is_capped_for_the_model(self):
        async with _running() as rig:
            rig.server.live_context = LIVE_CONTEXT_TEXT + "#" * 5000
            reading = await rig.provider.read_status(None)
            assert len(reading.text) == STATUS_TEXT_MAX_CHARS == 1500
            assert reading.truncated

    @pytest.mark.asyncio
    async def test_status_refreshes_the_catalog(self):
        async with _running() as rig:
            rig.server.live_context = LIVE_CONTEXT_TEXT.replace("Kitchen", "Study")
            await rig.provider.read_status(None)
            assert rig.provider.catalog().areas == ("Bedroom", "Study")

    @pytest.mark.asyncio
    async def test_without_live_context_status_falls_back_to_get_state(self):
        server = FakeHomeAssistant(omit=[LIVE_CONTEXT], include_get_state=True)
        async with _running(server) as rig:
            rig.server.call_script.append(
                McpToolResult(is_error=False, text="Porch Light is on", structured=None)
            )
            reading = await rig.provider.read_status(LightTarget(kind="name", value="Porch"))
            assert reading.text == "Porch Light is on"
            assert reading.lights is None
            assert rig.calls_to(GET_STATE) == [{"name": "Porch", "domain": ["light"]}]

    @pytest.mark.asyncio
    async def test_get_state_with_no_target_names_only_the_domain(self):
        server = FakeHomeAssistant(omit=[LIVE_CONTEXT], include_get_state=True)
        async with _running(server) as rig:
            await rig.provider.read_status(None)
            assert rig.calls_to(GET_STATE) == [{"domain": ["light"]}]

    @pytest.mark.asyncio
    async def test_with_neither_status_tool_status_is_unavailable(self):
        async with _running(FakeHomeAssistant(omit=[LIVE_CONTEXT])) as rig:
            with pytest.raises(CapabilityUnavailableError, match="GetLiveContext"):
                await rig.provider.read_status(None)

    @pytest.mark.asyncio
    async def test_a_status_tool_error_raises_with_its_text(self):
        async with _running() as rig:
            rig.server.call_script.append(tool_error("context unavailable"))
            with pytest.raises(CapabilityInvocationError, match="context unavailable"):
                await rig.provider.read_status(None)


class TestStatus:
    @pytest.mark.asyncio
    async def test_before_start_lights_are_unavailable(self):
        rig = _build(FakeHomeAssistant(), FakeClock())
        status = rig.provider.status()
        assert (status.name, status.state, status.task_dead) == ("lights", "unavailable", False)
        assert status.reason == "not started"

    @pytest.mark.asyncio
    async def test_connected_with_every_tool_lights_are_available(self):
        async with _running() as rig:
            status = rig.provider.status()
            assert (status.name, status.state, status.reason) == ("lights", "available", None)

    @pytest.mark.asyncio
    async def test_a_dead_supervisor_gives_lights_unavailable_with_task_dead(self):
        server = FakeHomeAssistant()
        server.open_failures.append(unexpected_error())
        async with _running(server, connect=False) as rig:
            await _until(rig.connection.task_dead)
            status = rig.provider.status()
            assert (status.name, status.state, status.task_dead) == ("lights", "unavailable", True)
            assert status.reason == rig.connection.state().reason

    @pytest.mark.asyncio
    async def test_a_connect_failure_is_reported_under_the_name_lights(self):
        server = FakeHomeAssistant()
        server.open_gate = asyncio.Event()
        async with _running(server, connect=False) as rig:
            status = rig.provider.status()
            assert (status.name, status.state, status.reason) == (
                "lights",
                "unavailable",
                "connecting",
            )
            server.open_gate.set()

    @pytest.mark.parametrize("missing", [TURN_ON, TURN_OFF])
    @pytest.mark.asyncio
    async def test_a_missing_power_tool_makes_lights_unavailable(self, missing):
        async with _running(FakeHomeAssistant(omit=[missing])) as rig:
            status = rig.provider.status()
            assert status.state == "unavailable"
            assert missing in (status.reason or "")
            assert not status.task_dead

    @pytest.mark.asyncio
    async def test_a_missing_light_set_makes_lights_degraded(self):
        async with _running(FakeHomeAssistant(omit=[LIGHT_SET], include_get_state=True)) as rig:
            status = rig.provider.status()
            assert status.state == "degraded"
            assert LIGHT_SET in (status.reason or "")

    @pytest.mark.asyncio
    async def test_both_status_tools_missing_makes_lights_degraded_naming_both(self):
        async with _running(FakeHomeAssistant(omit=[LIVE_CONTEXT])) as rig:
            status = rig.provider.status()
            assert status.state == "degraded"
            assert LIVE_CONTEXT in (status.reason or "")
            assert GET_STATE in (status.reason or "")

    @pytest.mark.asyncio
    async def test_one_status_tool_is_enough(self):
        server = FakeHomeAssistant(omit=[LIVE_CONTEXT], include_get_state=True)
        async with _running(server) as rig:
            assert rig.provider.status().state == "available"
        async with _running(FakeHomeAssistant()) as rig:
            assert rig.provider.status().state == "available"


class TestProviderTools:
    @pytest.mark.asyncio
    async def test_sorted_tool_names_from_the_last_list(self):
        async with _running() as rig:
            assert rig.provider.provider_tools() == tuple(
                sorted([GET_STATE, LIGHT_SET, LIVE_CONTEXT, TURN_OFF, TURN_ON])
            )

    @pytest.mark.asyncio
    async def test_empty_while_disconnected(self):
        rig = _build(FakeHomeAssistant(), FakeClock())
        assert rig.provider.provider_tools() == ()
        async with _running() as rig:
            await rig.provider.stop()
            assert rig.provider.provider_tools() == ()


class TestCatalogRefresh:
    @pytest.mark.asyncio
    async def test_refreshes_on_connect(self):
        async with _running() as rig:
            assert rig.provider.catalog().names == ("Bedroom Lamp", "Ceiling Light")
            assert rig.provider.catalog().areas == ("Bedroom", "Kitchen")
            assert len(rig.calls_to(LIVE_CONTEXT)) == 1

    @pytest.mark.asyncio
    async def test_refreshes_every_ten_minutes(self):
        assert CATALOG_REFRESH_S == 600.0
        async with _running() as rig:
            rig.server.live_context = LIVE_CONTEXT_TEXT.replace("Kitchen", "Study")
            await rig.clock.advance(CATALOG_REFRESH_S - 2)
            assert len(rig.calls_to(LIVE_CONTEXT)) == 1
            assert "Kitchen" in rig.provider.catalog().areas
            await rig.clock.advance(2.0)
            assert len(rig.calls_to(LIVE_CONTEXT)) == 2
            assert rig.provider.catalog().areas == ("Bedroom", "Study")
            assert rig.clock.stalled == []

    @pytest.mark.asyncio
    async def test_refreshes_again_on_reconnect(self):
        async with _running() as rig:
            rig.server.live_context = LIVE_CONTEXT_TEXT.replace("Kitchen", "Study")
            # A transport error drops the session, and the supervisor reconnects at once,
            # possibly before the refresh loop's next poll sees it down.
            rig.server.call_script.append(McpTransportError("connection reset"))
            with pytest.raises(McpTransportError):
                await rig.provider.power(LightTarget(kind="area", value="Kitchen"), True)
            await _until(lambda: rig.connection.state().state == "available")
            await rig.clock.advance(1.0)
            assert len(rig.calls_to(LIVE_CONTEXT)) == 2
            assert rig.provider.catalog().areas == ("Bedroom", "Study")

    @pytest.mark.asyncio
    async def test_refreshes_after_a_reconnect_with_backoff(self):
        server = FakeHomeAssistant()
        async with _running(server) as rig:
            server.live_context = LIVE_CONTEXT_TEXT.replace("Kitchen", "Study")
            server.open_failures.append(McpTransportError("connection refused"))
            server.call_script.append(McpTransportError("connection reset"))
            with pytest.raises(McpTransportError):
                await rig.provider.power(LightTarget(kind="area", value="Kitchen"), True)
            await _until(lambda: rig.connection.state().reason == "connection refused")
            # The supervisor backs off 1 s before it reconnects.
            await rig.clock.advance(1.0)
            await _until(lambda: rig.connection.state().state == "available")
            await rig.clock.advance(1.0)
            assert rig.provider.catalog().areas == ("Bedroom", "Study")

    @pytest.mark.asyncio
    async def test_a_failed_refresh_keeps_the_catalog_and_the_loop(self, caplog):
        async with _running() as rig:
            rig.server.call_script.append(McpTransportError("connection reset"))
            with caplog.at_level(logging.WARNING, logger=LOGGER):
                await rig.clock.advance(CATALOG_REFRESH_S)
            assert rig.provider.catalog().areas == ("Bedroom", "Kitchen")
            assert any("refresh" in r.getMessage() for r in caplog.records if r.name == LOGGER)
            await _until(lambda: rig.provider.status().state == "available")

    @pytest.mark.asyncio
    async def test_a_refresh_bug_ends_the_loop_visibly(self, caplog):
        async with _running() as rig:
            rig.server.call_script.append(unexpected_error())
            with caplog.at_level(logging.ERROR, logger=LOGGER):
                await rig.clock.advance(CATALOG_REFRESH_S)
                await _until(lambda: rig.provider.status().state == "degraded")
            status = rig.provider.status()
            assert "catalog refresh" in (status.reason or "")
            errors = [r for r in caplog.records if r.name == LOGGER and r.levelno == logging.ERROR]
            assert len(errors) == 1

    @pytest.mark.asyncio
    async def test_every_call_passes_the_constructor_timeout(self):
        async with _running() as rig:
            target = rig.provider.resolve("kitchen")
            await rig.provider.power(target, True)
            await rig.provider.adjust(target, LightAdjustment(color="blue"))
            await rig.provider.read_status(None)
            names = [call.name for call in rig.server.calls]
            # The first GetLiveContext is the on-connect catalog refresh.
            assert names == [LIVE_CONTEXT, TURN_ON, LIGHT_SET, LIVE_CONTEXT]
            assert {call.timeout_s for call in rig.server.calls} == {TIMEOUT_S}

    @pytest.mark.asyncio
    async def test_stop_ends_the_refresh_loop(self):
        async with _running() as rig:
            pass
        assert rig.clock.pending == 0
        assert rig.connection.state().reason == "stopped"


def _prefixed(
    *, omit: list[str] | None = None, include_get_state: bool = False
) -> FakeHomeAssistant:
    return FakeHomeAssistant(prefixed=True, omit=omit or [], include_get_state=include_get_state)


AMBIGUOUS_TURN_ON = "Home Assistant lists HassTurnOn more than once (a__HassTurnOn, b__HassTurnOn)"


class TestPrefixedToolNames:
    """Home Assistant 2026.11 lists each tool as `<source>__<name>`."""

    @pytest.mark.asyncio
    async def test_a_prefixed_listing_gives_lights_available(self):
        async with _running(_prefixed()) as rig:
            status = rig.provider.status()

            assert (status.state, status.reason) == ("available", None)

    @pytest.mark.parametrize(("on", "tool"), [(True, TURN_ON), (False, TURN_OFF)])
    @pytest.mark.asyncio
    async def test_power_calls_the_listed_prefixed_name(self, on, tool):
        async with _running(_prefixed()) as rig:
            await rig.provider.power(LightTarget(kind="area", value="Kitchen"), on)

            assert rig.calls_to(prefixed_name(tool)) == [{"area": "Kitchen", "domain": ["light"]}]
            assert rig.calls_to(tool) == []

    @pytest.mark.asyncio
    async def test_adjust_calls_the_listed_prefixed_light_set(self):
        async with _running(_prefixed()) as rig:
            await rig.provider.adjust(
                LightTarget(kind="area", value="Kitchen"), LightAdjustment(brightness_pct=40)
            )

            assert rig.calls_to("light__HassLightSet") == [
                {"area": "Kitchen", "domain": ["light"], "brightness": 40}
            ]

    @pytest.mark.asyncio
    async def test_connect_fills_the_catalog_from_json_wrapped_prefixed_live_context(self):
        wrapped = json.dumps({"result": LIVE_CONTEXT_TEXT})

        async with _running(_prefixed(), live_context=wrapped) as rig:
            assert rig.provider.catalog().names == ("Bedroom Lamp", "Ceiling Light")
            assert len(rig.calls_to("homeassistant__GetLiveContext")) == 1

    @pytest.mark.asyncio
    async def test_read_status_calls_the_prefixed_live_context(self):
        async with _running(_prefixed()) as rig:
            reading = await rig.provider.read_status(None)

            assert reading.text == LIVE_CONTEXT_TEXT.strip()
            # The first call is the on-connect catalog refresh.
            assert len(rig.calls_to("homeassistant__GetLiveContext")) == 2

    @pytest.mark.asyncio
    async def test_read_status_falls_back_to_the_prefixed_get_state(self):
        server = _prefixed(omit=[LIVE_CONTEXT], include_get_state=True)
        async with _running(server) as rig:
            await rig.provider.read_status(None)

            assert rig.calls_to("intent__HassGetState") == [{"domain": ["light"]}]

    @pytest.mark.asyncio
    async def test_a_missing_tool_is_named_bare_in_the_reason(self):
        async with _running(_prefixed(omit=[TURN_OFF])) as rig:
            status = rig.provider.status()

            assert (status.state, status.reason) == (
                "unavailable",
                "Home Assistant lacks HassTurnOff",
            )

    @pytest.mark.asyncio
    async def test_a_tool_error_without_text_names_the_listed_tool(self):
        async with _running(_prefixed()) as rig:
            rig.server.call_script.append(tool_error(""))

            with pytest.raises(CapabilityInvocationError, match="^intent__HassTurnOn failed$"):
                await rig.provider.power(LightTarget(kind="area", value="Kitchen"), True)

    @pytest.mark.asyncio
    async def test_provider_tools_lists_the_names_as_the_server_lists_them(self):
        async with _running(_prefixed()) as rig:
            assert rig.provider.provider_tools() == (
                "homeassistant__GetLiveContext",
                "intent__HassTurnOff",
                "intent__HassTurnOn",
                "light__HassLightSet",
            )


class TestAmbiguousToolNames:
    """Two sources listing the same tool, with no bare name, count as missing."""

    @pytest.mark.asyncio
    async def test_two_prefixed_matches_give_unavailable_naming_both(self):
        server = FakeHomeAssistant(omit=[TURN_ON], extra_tools=["b__HassTurnOn", "a__HassTurnOn"])
        async with _running(server) as rig:
            status = rig.provider.status()

            assert (status.state, status.reason) == ("unavailable", AMBIGUOUS_TURN_ON)

    @pytest.mark.asyncio
    async def test_power_with_two_prefixed_matches_raises_and_calls_neither(self):
        server = FakeHomeAssistant(omit=[TURN_ON], extra_tools=["b__HassTurnOn", "a__HassTurnOn"])
        async with _running(server) as rig:
            with pytest.raises(CapabilityUnavailableError) as raised:
                await rig.provider.power(LightTarget(kind="area", value="Kitchen"), True)

            assert str(raised.value) == AMBIGUOUS_TURN_ON
            assert rig.calls_to("a__HassTurnOn") == rig.calls_to("b__HassTurnOn") == []

    @pytest.mark.asyncio
    async def test_a_bare_name_wins_over_a_prefixed_one(self):
        async with _running(FakeHomeAssistant(extra_tools=["x__HassTurnOn"])) as rig:
            await rig.provider.power(LightTarget(kind="area", value="Kitchen"), True)

            assert len(rig.calls_to(TURN_ON)) == 1
            assert rig.calls_to("x__HassTurnOn") == []


class TestBareToolNames:
    """A listing before Home Assistant 2026.11 keeps today's calls and reasons."""

    @pytest.mark.asyncio
    async def test_bare_listing_calls_the_bare_names(self):
        async with _running() as rig:
            target = LightTarget(kind="area", value="Kitchen")
            await rig.provider.power(target, True)
            await rig.provider.adjust(target, LightAdjustment(brightness_pct=10))
            await rig.provider.read_status(None)

            names = [call.name for call in rig.server.calls]
            assert names == [LIVE_CONTEXT, TURN_ON, LIGHT_SET, LIVE_CONTEXT]

    @pytest.mark.parametrize(
        ("omit", "state", "reason"),
        [
            pytest.param(
                [TURN_ON, TURN_OFF],
                "unavailable",
                "Home Assistant lacks HassTurnOn and HassTurnOff",
                id="both-power-tools",
            ),
            pytest.param(
                [LIGHT_SET, LIVE_CONTEXT],
                "degraded",
                "Home Assistant lacks HassLightSet and GetLiveContext and HassGetState",
                id="light-set-and-status-tools",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_reasons_read_as_today(self, omit, state, reason):
        async with _running(FakeHomeAssistant(omit=omit)) as rig:
            status = rig.provider.status()

            assert (status.state, status.reason) == (state, reason)

    @pytest.mark.asyncio
    async def test_power_without_its_tool_is_unavailable_and_calls_nothing(self):
        async with _running(FakeHomeAssistant(omit=[TURN_ON])) as rig:
            with pytest.raises(CapabilityUnavailableError) as raised:
                await rig.provider.power(LightTarget(kind="area", value="Kitchen"), True)

            assert str(raised.value) == "Home Assistant lacks HassTurnOn"
            assert rig.calls_to(TURN_ON) == []
