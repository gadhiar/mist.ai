"""Tests for `LightsCapability`: its tools, schema refinement, fast path and outcomes."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable

import pytest
from pydantic import ValidationError

from backend.capabilities.config import load_capabilities_config
from backend.capabilities.contracts import CapabilityStatus, InvocationContext, ToolOutcome
from backend.capabilities.decoder import DECODE_INSTRUCTIONS
from backend.capabilities.intents import normalize_utterance
from backend.capabilities.lights.capability import (
    KNOWN_TARGETS_NAMED,
    LightAdjustment,
    LightsCapability,
    StatusReading,
)
from backend.capabilities.lights.catalog import (
    EMPTY_CATALOG,
    LightCatalog,
    LightEntity,
    LightTarget,
)
from backend.capabilities.lights.home_assistant import HomeAssistantLightsProvider
from backend.capabilities.mcp.connection import BackoffPolicy, McpConnection
from backend.capabilities.replies import is_brief_sentence, ok_summary, render_read
from backend.capabilities.router import RouterConfig, ToolRouter, count_catalog_tokens
from backend.capabilities.schema import to_args_schema, to_catalog_line
from backend.chat.context_budget import ApproximateTokenCounter
from backend.errors import (
    CapabilityInvocationError,
    CapabilityUnavailableError,
    McpTransportError,
)
from tests.unit.capabilities.fakes import DEFAULT_START, FakeClock
from tests.unit.capabilities.mcp.fake_home_assistant import (
    FakeHomeAssistant,
    tool_error,
    unexpected_error,
)

COUNTER = ApproximateTokenCounter()
POWER, ADJUST, STATUS = "lights.power", "lights.adjust", "lights.status"
WAKE_WORDS = ("mist", "hey mist", "ok mist", "okay mist")


def _catalog(*areas: str, names: tuple[str, ...] = ()) -> LightCatalog:
    lights = [LightEntity(names=(f"{area} Light",), areas=(area,), state="off") for area in areas]
    lights.extend(LightEntity(names=(name,), areas=(), state="on") for name in names)
    return LightCatalog(lights=tuple(lights))


HOME = _catalog("Bedroom", "Kitchen", "Living Room")


class StubProvider:
    """A `LightsProvider` whose catalog, status and failures a test sets directly."""

    def __init__(self, catalog: LightCatalog = HOME) -> None:
        self.catalog_value = catalog
        self.status_value = CapabilityStatus(
            name="lights", state="available", reason=None, checked_at=DEFAULT_START
        )
        self.reading = StatusReading(text="lights text", lights=catalog.lights, truncated=False)
        self.error: BaseException | None = None
        self.calls: list[tuple[str, object, object]] = []
        self.tool_names: tuple[str, ...] = ("HassTurnOff", "HassTurnOn")
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    def status(self) -> CapabilityStatus:
        return self.status_value

    def catalog(self) -> LightCatalog:
        return self.catalog_value

    def resolve(self, target: str) -> LightTarget | None:
        if self.catalog_value.is_empty:
            return LightTarget(kind="name", value=target)
        return self.catalog_value.resolve(target)

    def provider_tools(self) -> tuple[str, ...]:
        return self.tool_names

    async def power(self, target: LightTarget, on: bool) -> None:
        self.calls.append(("power", target, on))
        self._raise()

    async def adjust(self, target: LightTarget, adjustment: LightAdjustment) -> None:
        self.calls.append(("adjust", target, adjustment))
        self._raise()

    async def read_status(self, target: LightTarget | None) -> StatusReading:
        self.calls.append(("status", target, None))
        self._raise()
        return self.reading

    def _raise(self) -> None:
        if self.error is not None:
            raise self.error


def _capability(
    provider: StubProvider | HomeAssistantLightsProvider | None = None,
    default_target: str | None = None,
) -> LightsCapability:
    return LightsCapability(
        provider or StubProvider(), default_target=default_target, counter=COUNTER
    )


def _ctx() -> InvocationContext:
    return InvocationContext(
        session_id="s1",
        turn_id="t1",
        utterance="",
        tool_call_id="c1",
        source="fastpath",
        now=DEFAULT_START,
    )


def _spec(capability: LightsCapability, name: str):
    return next(spec for spec in capability.tools() if spec.name == name)


async def _invoke(capability: LightsCapability, tool: str, **args: object) -> ToolOutcome:
    validated = _spec(capability, tool).args_model.model_validate(args)
    return await capability.invoke(tool, validated, _ctx())


def _lines(capability: LightsCapability, refine: bool) -> list[str]:
    hook: Callable[[str, dict[str, object]], dict[str, object]] | None = (
        capability.refine_schema if refine else None
    )
    return [to_catalog_line(spec, to_args_schema(spec, hook)) for spec in capability.tools()]


class TestTools:
    def test_three_tools_with_their_risk_and_policy(self):
        specs = {spec.name: spec for spec in _capability().tools()}
        assert list(specs) == [POWER, ADJUST, STATUS]
        assert (specs[POWER].risk, specs[POWER].default_policy) == ("physical", "confirm_inferred")
        assert (specs[ADJUST].risk, specs[ADJUST].default_policy) == (
            "physical",
            "confirm_inferred",
        )
        assert (specs[STATUS].risk, specs[STATUS].default_policy) == ("read", "allow")

    def test_every_tool_sets_a_short_ack_and_the_lights_view(self):
        for spec in _capability().tools():
            assert spec.ack
            assert len(spec.ack.split()) <= 5
            assert spec.view == "lights"

    def test_static_catalog_lines(self):
        assert _lines(_capability(), refine=False) == [
            "lights.power(target?: str, on: true|false): turn lights on or off",
            "lights.adjust(target?: str, brightness_pct?: int 0-100, color?: str, kelvin?: int"
            " 1000-10000): set brightness, color or color temperature",
            "lights.status(target?: str): report which lights are on",
        ]

    def test_name_and_allowance(self):
        capability = _capability()
        assert capability.name == "lights"
        assert capability.refine_allowance_tokens == 200

    def test_no_is_explicit_hook_and_no_escalation(self):
        capability = _capability()
        assert not hasattr(capability, "is_explicit")
        power = _spec(capability, POWER).args_model.model_validate({"on": True})
        assert capability.escalate(POWER, power) is False


class TestArguments:
    def test_adjust_needs_at_least_one_setting(self):
        model = _spec(_capability(), ADJUST).args_model
        with pytest.raises(ValidationError, match="at least one"):
            model.model_validate({"target": "kitchen"})
        with pytest.raises(ValidationError, match="at least one"):
            model.model_validate({"color": "  "})

    @pytest.mark.parametrize(
        "args",
        [
            {"brightness_pct": 101},
            {"brightness_pct": -1},
            {"kelvin": 999},
            {"kelvin": 10001},
            {"brightness_pct": 5, "hue": 3},
        ],
    )
    def test_adjust_ranges_and_extra_fields(self, args):
        with pytest.raises(ValidationError):
            _spec(_capability(), ADJUST).args_model.model_validate(args)

    def test_a_blank_target_reads_as_none(self):
        model = _spec(_capability(), POWER).args_model
        assert model.model_validate({"target": "   ", "on": True}).target is None
        assert model.model_validate({"target": " living   room ", "on": True}).target == (
            "living room"
        )

    def test_summaries_for_proposals(self):
        capability = _capability(default_target="Bedroom")
        power = _spec(capability, POWER)
        adjust = _spec(capability, ADJUST)
        status = _spec(capability, STATUS)
        assert power.summarize(power.args_model.model_validate({"on": False})) == (
            "turn off the Bedroom lights"
        )
        assert (
            power.summarize(power.args_model.model_validate({"target": "kitchen", "on": True}))
            == "turn on the kitchen lights"
        )
        assert (
            adjust.summarize(
                adjust.args_model.model_validate(
                    {"target": "kitchen", "brightness_pct": 30, "color": "red"}
                )
            )
            == "set the kitchen lights to 30 percent and red"
        )
        assert status.summarize(status.args_model.model_validate({})) == "check the lights"
        no_default = _capability()
        spec = _spec(no_default, POWER)
        assert spec.summarize(spec.args_model.model_validate({"on": True})) == (
            "turn on the lights"
        )


class TestRefinement:
    def test_ten_targets_give_a_sorted_enum_within_the_allowance(self):
        catalog = _catalog("Kitchen", "Attic", "Bedroom", "Hall", "Den")
        assert len(catalog.targets()) == 10
        capability = _capability(StubProvider(catalog))
        for tool in (POWER, ADJUST):
            schema = to_args_schema(_spec(capability, tool), capability.refine_schema)
            target = schema["properties"]["target"]
            assert target["enum"] == sorted(catalog.targets())
        status = to_args_schema(_spec(capability, STATUS), capability.refine_schema)
        assert status["properties"]["target"] == {"type": "string"}
        static = count_catalog_tokens(_lines(capability, refine=False), COUNTER)
        refined = count_catalog_tokens(_lines(capability, refine=True), COUNTER)
        assert static < refined <= static + capability.refine_allowance_tokens

    def test_fifty_targets_leave_target_a_free_string(self):
        catalog = _catalog(*(f"Guest Suite Number {n:02d}" for n in range(25)))
        assert len(catalog.targets()) == 50
        capability = _capability(StubProvider(catalog))
        for tool in (POWER, ADJUST, STATUS):
            schema = to_args_schema(_spec(capability, tool), capability.refine_schema)
            assert schema["properties"]["target"] == {"type": "string"}
        static = count_catalog_tokens(_lines(capability, refine=False), COUNTER)
        refined = count_catalog_tokens(_lines(capability, refine=True), COUNTER)
        assert refined == static <= static + capability.refine_allowance_tokens

    def test_an_empty_catalog_adds_no_enum(self):
        capability = _capability(StubProvider(EMPTY_CATALOG))
        assert _lines(capability, refine=True) == _lines(capability, refine=False)

    def test_refinement_follows_the_current_catalog(self):
        provider = StubProvider(_catalog("Kitchen"))
        capability = _capability(provider)
        first = to_args_schema(_spec(capability, POWER), capability.refine_schema)
        provider.catalog_value = _catalog("Study")
        second = to_args_schema(_spec(capability, POWER), capability.refine_schema)
        assert first["properties"]["target"]["enum"] == ["Kitchen", "Kitchen Light"]
        assert second["properties"]["target"]["enum"] == ["Study", "Study Light"]

    def test_dynamic_keywords_are_the_catalog_targets(self):
        capability = _capability()
        assert capability.dynamic_keywords() == HOME.targets()
        assert _capability(StubProvider(EMPTY_CATALOG)).dynamic_keywords() == ()

    def test_worst_case_fits_the_per_capability_budget_at_default_config(self):
        capability = _capability()
        config = load_capabilities_config({})
        router_config = RouterConfig(
            token_budget=config.tool_token_budget,
            reserved_tokens=COUNTER.count(DECODE_INSTRUCTIONS),
        )
        # Registration refuses a capability whose worst case exceeds the budget.
        router = ToolRouter([capability], router_config, COUNTER, None)
        static = count_catalog_tokens(_lines(capability, refine=False), COUNTER)
        worst = static + capability.refine_allowance_tokens
        assert worst <= router.per_capability_budget()

    def test_arguments_schemas_are_byte_stable(self):
        capability = _capability()
        first = [
            json.dumps(to_args_schema(spec, capability.refine_schema))
            for spec in capability.tools()
        ]
        second = [
            json.dumps(to_args_schema(spec, capability.refine_schema))
            for spec in capability.tools()
        ]
        assert first == second


class TestFastPath:
    def test_match_intent_uses_the_providers_catalog(self):
        capability = _capability()
        text = normalize_utterance("Turn off the bedroom", WAKE_WORDS)
        intent = capability.match_intent(text)
        assert intent is not None
        assert (intent.tool, dict(intent.args)) == (POWER, {"target": "bedroom", "on": False})
        assert _capability(StubProvider(EMPTY_CATALOG)).match_intent(text) is None

    def test_a_light_named_music_takes_turn_off_the_music(self):
        capability = _capability(StubProvider(_catalog("Bedroom", names=("Music",))))
        intent = capability.match_intent("turn off the music")
        assert intent is not None
        assert (intent.tool, dict(intent.args)) == (POWER, {"target": "music", "on": False})

    @pytest.mark.asyncio
    async def test_an_unknown_target_next_to_a_light_word_gives_invalid_args(self):
        provider = StubProvider()
        capability = _capability(provider)
        intent = capability.match_intent("turn off the garage lights")
        assert intent is not None
        outcome = await _invoke(capability, intent.tool, **intent.args)
        assert outcome.status == "invalid_args"
        for known in HOME.targets():
            assert known in outcome.summary
        assert "garage" in outcome.summary
        assert provider.calls == []


class TestPower:
    @pytest.mark.asyncio
    async def test_ok_summary_is_the_shared_template(self):
        provider = StubProvider()
        capability = _capability(provider)
        outcome = await _invoke(capability, POWER, target="bedroom", on=False)
        assert outcome.status == "ok"
        assert (
            outcome.summary
            == "Bedroom lights off."
            == ok_summary("lights.power", target="Bedroom", state="off")
        )
        assert is_brief_sentence(outcome.summary)
        assert provider.calls == [("power", LightTarget(kind="area", value="Bedroom"), False)]

    @pytest.mark.asyncio
    async def test_a_light_whose_name_ends_in_a_light_word_reads_naturally(self):
        capability = _capability(StubProvider(_catalog(names=("Bedroom Lamp",))))
        outcome = await _invoke(capability, POWER, target="bedroom lamp", on=True)
        assert outcome.summary == "Bedroom Lamp on."
        assert is_brief_sentence(outcome.summary)

    @pytest.mark.asyncio
    async def test_the_default_target_fills_a_missing_target(self):
        provider = StubProvider()
        capability = _capability(provider, default_target="Living Room")
        outcome = await _invoke(capability, POWER, on=True)
        assert outcome.summary == "Living Room lights on."
        assert provider.calls == [("power", LightTarget(kind="area", value="Living Room"), True)]

    @pytest.mark.parametrize("tool", [POWER, ADJUST])
    @pytest.mark.asyncio
    async def test_no_target_and_no_default_is_invalid_args(self, tool):
        provider = StubProvider()
        args = {"on": True} if tool == POWER else {"brightness_pct": 10}
        outcome = await _invoke(_capability(provider), tool, **args)
        assert outcome.status == "invalid_args"
        assert "say which light or room" in outcome.summary.casefold()
        assert provider.calls == []

    @pytest.mark.asyncio
    async def test_an_unknown_target_names_at_most_ten_known_targets(self):
        catalog = _catalog(*(f"Room {letter}" for letter in "ABCDEF"))
        assert len(catalog.targets()) == 12
        outcome = await _invoke(_capability(StubProvider(catalog)), POWER, target="Loft", on=True)
        assert outcome.status == "invalid_args"
        assert KNOWN_TARGETS_NAMED == 10
        assert outcome.summary == (
            "There is no light or room called Loft; known ones are Room A, Room A Light, Room B,"
            " Room B Light, Room C, Room C Light, Room D, Room D Light, Room E, Room E Light"
            " and 2 more."
        )

    @pytest.mark.asyncio
    async def test_a_free_string_target_matches_the_catalog_ignoring_case(self):
        provider = StubProvider()
        await _invoke(_capability(provider), POWER, target="LIVING ROOM", on=False)
        assert provider.calls == [("power", LightTarget(kind="area", value="Living Room"), False)]


class TestAdjust:
    @pytest.mark.asyncio
    async def test_brightness_uses_the_shared_template(self):
        provider = StubProvider()
        outcome = await _invoke(_capability(provider), ADJUST, target="kitchen", brightness_pct=30)
        assert (
            outcome.summary
            == "Kitchen lights set to 30 percent."
            == ok_summary("lights.adjust", target="Kitchen", percent=30)
        )
        assert provider.calls == [
            (
                "adjust",
                LightTarget(kind="area", value="Kitchen"),
                LightAdjustment(brightness_pct=30),
            )
        ]

    @pytest.mark.parametrize(
        ("args", "summary"),
        [
            ({"color": "red"}, "Kitchen lights set to red."),
            ({"kelvin": 2700}, "Kitchen lights set to 2700 kelvin."),
            (
                {"brightness_pct": 40, "color": "warm white", "kelvin": 3000},
                "Kitchen lights set to 40 percent, warm white and 3000 kelvin.",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_other_settings_give_a_brief_sentence(self, args, summary):
        outcome = await _invoke(_capability(), ADJUST, target="kitchen", **args)
        assert outcome.summary == summary
        assert is_brief_sentence(outcome.summary)


class TestStatusRead:
    @pytest.mark.asyncio
    async def test_status_renders_the_read_template_with_data(self):
        provider = StubProvider()
        provider.reading = StatusReading(
            text="context text",
            lights=(
                LightEntity(names=("Bedroom Lamp",), areas=("Bedroom",), state="off"),
                LightEntity(names=("Ceiling Light",), areas=("Kitchen",), state="on"),
            ),
            truncated=True,
        )
        outcome = await _invoke(_capability(provider), STATUS)
        assert outcome.status == "ok"
        assert outcome.summary == render_read(
            "lights.status", ["Bedroom Lamp is off", "Ceiling Light is on"]
        )
        assert outcome.summary.startswith("You have 2 lights.")
        assert outcome.data == {"text": "context text", "truncated": True}
        assert provider.calls == [("status", None, None)]

    @pytest.mark.asyncio
    async def test_status_resolves_its_target_and_ignores_the_default(self):
        provider = StubProvider()
        capability = _capability(provider, default_target="Bedroom")
        await _invoke(capability, STATUS, target="kitchen")
        await _invoke(capability, STATUS)
        assert provider.calls == [
            ("status", LightTarget(kind="area", value="Kitchen"), None),
            ("status", None, None),
        ]

    @pytest.mark.asyncio
    async def test_status_with_an_unknown_target_is_invalid_args(self):
        outcome = await _invoke(_capability(), STATUS, target="garage")
        assert outcome.status == "invalid_args"

    @pytest.mark.asyncio
    async def test_a_reading_without_a_light_list_gives_a_brief_sentence(self):
        provider = StubProvider()
        provider.reading = StatusReading(text="Porch Light is on", lights=None, truncated=False)
        outcome = await _invoke(_capability(provider), STATUS)
        assert outcome.status == "ok"
        assert is_brief_sentence(outcome.summary)
        assert outcome.data == {"text": "Porch Light is on", "truncated": False}


class TestFailures:
    @pytest.mark.parametrize(
        "error", [CapabilityUnavailableError("auth rejected"), McpTransportError("reset")]
    )
    @pytest.mark.asyncio
    async def test_an_unreachable_provider_is_unavailable(self, error):
        provider = StubProvider()
        provider.error = error
        outcome = await _invoke(_capability(provider), POWER, target="kitchen", on=True)
        assert outcome.status == "unavailable"
        assert outcome.reason == str(error)
        assert is_brief_sentence(outcome.summary)

    @pytest.mark.asyncio
    async def test_a_provider_error_is_error_with_its_text(self):
        provider = StubProvider()
        provider.error = CapabilityInvocationError("Unable to find a device named Kitchen")
        outcome = await _invoke(_capability(provider), ADJUST, target="kitchen", color="red")
        assert outcome.status == "error"
        assert outcome.reason == "Unable to find a device named Kitchen"
        assert is_brief_sentence(outcome.summary)


class TestOverHomeAssistant:
    """The capability over the real provider, connection and the fake Home Assistant."""

    @staticmethod
    async def _wait(predicate: Callable[[], bool]) -> None:
        loop = asyncio.get_running_loop()
        give_up_at = loop.time() + 2.0
        while not predicate():
            assert loop.time() < give_up_at, "condition not reached in time"
            await asyncio.sleep(0.001)

    @staticmethod
    def _build(server: FakeHomeAssistant) -> tuple[LightsCapability, McpConnection, FakeClock]:
        clock = FakeClock()
        connection = McpConnection(server.transport, clock, BackoffPolicy(jitter=lambda d: d))
        provider = HomeAssistantLightsProvider(connection, clock, call_timeout_s=8.0)
        return _capability(provider), connection, clock

    @pytest.mark.asyncio
    async def test_a_home_assistant_tool_error_becomes_error_with_its_text(self):
        server = FakeHomeAssistant()
        capability, connection, clock = self._build(server)
        await capability.start()
        try:
            await self._wait(lambda: capability.status().state == "available")
            await clock.advance(1.0)
            assert "Bedroom Lamp" in capability.dynamic_keywords()
            server.call_script.append(tool_error("Unable to find a device named Bedroom Lamp"))
            outcome = await _invoke(capability, POWER, target="bedroom lamp", on=True)
            assert outcome.status == "error"
            assert outcome.reason == "Unable to find a device named Bedroom Lamp"
            assert capability.provider_tools() == (
                "GetLiveContext",
                "HassLightSet",
                "HassTurnOff",
                "HassTurnOn",
            )
        finally:
            await capability.stop()

    @pytest.mark.asyncio
    async def test_a_dead_supervisor_gives_lights_unavailable_with_task_dead(self):
        server = FakeHomeAssistant()
        server.open_failures.append(unexpected_error())
        capability, connection, _ = self._build(server)
        await capability.start()
        try:
            await self._wait(connection.task_dead)
            status = capability.status()
            assert (status.name, status.state, status.task_dead) == ("lights", "unavailable", True)
            assert status.reason
            assert capability.provider_tools() == ()
        finally:
            await capability.stop()

    @pytest.mark.asyncio
    async def test_start_and_stop_drive_the_provider(self):
        provider = StubProvider()
        capability = _capability(provider)
        await capability.start()
        await capability.stop()
        assert provider.started and provider.stopped
        assert capability.status() is provider.status_value
