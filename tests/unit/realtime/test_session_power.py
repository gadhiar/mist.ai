"""T22: a session at the off level, gated on the model tier rather than on the level.

Turns gate on `PowerGate.model_available()`, the `interaction` tier's provider, and the level is
only shown. While the tier has no provider, a typed message goes to the capability-only path, an
unhandled one gets the refusal envelope, and voice takes spoken commands only when the setting
is on and STT and TTS are both loaded on the CPU. The real controller runs against fakes: the
responder that would reach the model fails if anything calls it.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from backend.capabilities.power import FixedPower, FixedTiers, PowerLevel, SingleModelTiers
from backend.chat.conversation_handler import CapabilityOnlyTurn
from backend.chat.stream_events import Complete, StreamEvent, Token, WSEvent
from backend.chat.turn_scope import CommitResult, TurnDelivery, TurnRequest, TurnScope
from backend.gpu_yield import YIELDED_MESSAGE
from backend.realtime.budget import BargeInConfig, VoiceSettings
from backend.realtime.power import NO_MODEL_MESSAGE, PowerGate, gpu_yield_refusal
from backend.realtime.protocol import EngineStatus, EngineStatuses
from backend.realtime.session import (
    ALL_OFF,
    OFF,
    InferenceLanes,
    RealtimeSession,
    SessionServices,
    SpeechEngines,
)
from backend.realtime.turn import VoiceStart, VoiceStop
from tests.fakes.speech import EnergyVad, FakeSynthesizer, ScriptedTranscriber

from .test_session import (
    FakeIngress,
    FakeSocket,
    RecordingController,
    RecordingHost,
    name_of,
)
from .test_session import voice_start as _voice_start
from .turn_fakes import (
    TEST_PLAYBACK,
    TEST_SEGMENTER,
    FakeClock,
    FakeResponder,
    IdCounter,
    InlineLane,
    Pause,
    floor_placement,
    settle,
)


def voice_start() -> dict[str, object]:
    """A `voice_start` the rig's engines serve: no server spotter is loaded."""
    return _voice_start(spotter="endpoint")


TURN_LOGGER = "backend.realtime.turn"

PROVIDER = object()
"""The interaction tier's provider; the gate only asks whether one exists."""


def cpu(model_id: str) -> EngineStatus:
    return EngineStatus(loaded=True, location="cpu", model_id=model_id, vram_mib=None)


def cuda(model_id: str) -> EngineStatus:
    return EngineStatus(loaded=True, location="cuda", model_id=model_id, vram_mib=900)


ON_CPU = EngineStatuses(vad=cpu("vad"), eot=OFF, stt=cpu("stt"), tts=cpu("tts"), kws=OFF)
STT_ON_CUDA = EngineStatuses(vad=cpu("vad"), eot=OFF, stt=cuda("stt"), tts=cpu("tts"), kws=OFF)


class Level:
    """A `PowerState` a test switches."""

    def __init__(self, level: PowerLevel) -> None:
        self.value: PowerLevel = level

    def level(self) -> PowerLevel:
        return self.value


class FailingResponder:
    """The model path fails if called; its capability-only port answers from `capability`.

    Any model call means a turn reached a provider that is not there. With no `capability`, the
    port answers None, as a responder with no capability host attached does.
    """

    def __init__(self, capability: FakeCapabilityOnly | None = None) -> None:
        self.capability = capability

    async def handle_capability_only(self, text: str, session_id: str) -> CapabilityOnlyTurn | None:
        if self.capability is None:
            return None
        return await self.capability(text, session_id)

    def open_turn(self, request: TurnRequest) -> TurnScope:
        raise AssertionError("open_turn reached the model path")

    def stream_turn(self, scope: TurnScope) -> AsyncIterator[StreamEvent]:
        raise AssertionError("stream_turn reached the model path")

    def commit_turn(self, scope: TurnScope, delivery: TurnDelivery) -> CommitResult:
        raise AssertionError("commit_turn reached the model path")

    def prefetch(self, session_id: str, partial_text: str | None) -> None:
        raise AssertionError("prefetch reached the model path")


class FakeCapabilityOnly:
    """`handle_capability_only`: records each call and returns the scripted result."""

    def __init__(self, result: CapabilityOnlyTurn | None) -> None:
        self.result = result
        self.calls: list[tuple[str, str]] = []

    async def __call__(self, text: str, session_id: str) -> CapabilityOnlyTurn | None:
        self.calls.append((text, session_id))
        return self.result


HANDLED = CapabilityOnlyTurn(
    handled=True,
    events=(
        WSEvent(payload={"type": "tool_call_started", "tool_call_id": "c1", "name": "lights"}),
        Token("Lights on."),
        Complete(final_response="Lights on."),
    ),
)
UNHANDLED = CapabilityOnlyTurn(
    handled=False, events=(WSEvent(payload={"type": "action_resolved", "action_id": "a1"}),)
)


class PowerRig:
    """One session over a power gate, with fake ingress and the real controller."""

    def __init__(
        self,
        *,
        power: object,
        tiers: object,
        refusal: object = lambda: NO_MODEL_MESSAGE,
        spoken_commands_at_off: bool = True,
        statuses: EngineStatuses = ON_CPU,
        responder: object | None = None,
        alarm_ringing: object | None = None,
        session_id: str = "s1",
    ) -> None:
        self.clock = FakeClock()
        self.ids = IdCounter()
        self.socket = FakeSocket()
        self.responder = responder if responder is not None else FakeResponder(clock=self.clock)
        self.host = RecordingHost()
        self.ingresses: list[FakeIngress] = []
        self.controllers: list[RecordingController] = []
        self.gate = PowerGate(
            power,  # type: ignore[arg-type]
            tiers,  # type: ignore[arg-type]
            refusal,  # type: ignore[arg-type]
            spoken_commands_at_off=spoken_commands_at_off,
        )
        self.engines = SpeechEngines(
            vad=EnergyVad(),
            transcriber=ScriptedTranscriber(["hello there"], ms_per_word=20),
            synthesizer=FakeSynthesizer(),
            statuses=statuses,
        )
        extra: dict[str, object] = {}
        if alarm_ringing is not None:
            extra["alarm_ringing"] = alarm_ringing
        self.services = SessionServices(
            responder=self.responder,  # type: ignore[arg-type]
            engines=self.engines,
            lanes=InferenceLanes(
                vad=InlineLane(),
                eot=InlineLane(),
                stt=InlineLane(),
                tts=InlineLane(),
                kws=InlineLane(),
            ),
            settings=VoiceSettings(
                playback=TEST_PLAYBACK,
                segmenter=TEST_SEGMENTER,
                barge_in=BargeInConfig(arm_delay_ms=0),
            ),
            placement=floor_placement,
            clock_factory=lambda: self.clock,
            utterance_ids=lambda: self.ids,
            ingress_factory=self._ingress,
            controller_factory=self._controller,
            power_gate=self.gate,
            **extra,  # type: ignore[arg-type]
        )
        self.session = RealtimeSession(
            session_id=session_id,
            websocket=self.socket,
            endpoint="desktop",
            services=self.services,
            host=self.host,  # type: ignore[arg-type]
        )
        self.task: asyncio.Task[None] | None = None
        RIGS.append(self)

    def _ingress(self, **kwargs: object) -> FakeIngress:
        ingress = FakeIngress(kwargs)
        self.ingresses.append(ingress)
        return ingress

    def _controller(self, **kwargs: object) -> RecordingController:
        controller = RecordingController(**kwargs)
        self.controllers.append(controller)
        return controller

    @property
    def controller(self) -> RecordingController:
        return self.controllers[0]

    @property
    def ingress(self) -> FakeIngress:
        return self.ingresses[-1]

    async def start(self) -> PowerRig:
        self.task = asyncio.get_running_loop().create_task(self.session.run())
        await settle()
        return self

    async def stop(self) -> None:
        self.socket.release()
        self.socket.disconnect()
        assert self.task is not None
        await asyncio.wait_for(self.task, timeout=5)

    async def send(self, *payloads: dict[str, object]) -> None:
        for payload in payloads:
            self.socket.text(payload)
        await settle()

    async def utterance(self, text: str = "turn the lights on") -> str:
        uid = self.ids()
        self.ingress.emit(self.ingress.started(uid))
        self.ingress.emit(self.ingress.ended(uid, text))
        await settle()
        return uid

    async def broadcast(self) -> None:
        """What the registry's `broadcast_voice_state` does for this session."""
        self.session.send_voice_state()
        await settle()

    def names(self) -> list[str]:
        return [name_of(item) for item in self.socket.sent]

    def of(self, name: str) -> list[dict[str, object]]:
        return [item for item in self.socket.sent if name_of(item) == name]  # type: ignore[misc]

    def voice_states(self, cause: str) -> list[dict[str, object]]:
        return [s for s in self.of("voice_state") if s["cause"] == cause]

    def errors(self) -> list[dict[str, object]]:
        return self.of("error")

    def tokens(self) -> list[str]:
        return [str(item["token"]) for item in self.of("stream_token")]


RIGS: list[PowerRig] = []


@pytest_asyncio.fixture(autouse=True)
async def _stop_rigs():
    yield
    while RIGS:
        rig = RIGS.pop()
        if rig.task is not None and not rig.task.done():
            await rig.stop()


def no_model(level: PowerLevel = "off") -> dict[str, object]:
    return {"power": FixedPower(level), "tiers": FixedTiers({})}


# -- The gate --------------------------------------------------------------------------------------


class TestPowerGate:
    def test_spoken_commands_at_off_default_to_true(self):
        gate = PowerGate(FixedPower("off"), FixedTiers({}), lambda: NO_MODEL_MESSAGE)

        assert gate.spoken_commands_at_off is True
        assert gate.spoken_commands_allowed(stt_on_cpu=True, tts_on_cpu=True) is True

    def test_spoken_commands_need_the_setting_and_both_engines_on_the_cpu(self):
        on = PowerGate(FixedPower("off"), FixedTiers({}), lambda: NO_MODEL_MESSAGE)
        off = PowerGate(
            FixedPower("off"),
            FixedTiers({}),
            lambda: NO_MODEL_MESSAGE,
            spoken_commands_at_off=False,
        )

        assert on.spoken_commands_allowed(stt_on_cpu=False, tts_on_cpu=True) is False
        assert on.spoken_commands_allowed(stt_on_cpu=True, tts_on_cpu=False) is False
        assert off.spoken_commands_allowed(stt_on_cpu=True, tts_on_cpu=True) is False

    def test_the_model_is_available_by_the_interaction_tier_never_by_the_level(self):
        at_off = PowerGate(
            FixedPower("off"), FixedTiers({"interaction": PROVIDER}), lambda: NO_MODEL_MESSAGE
        )
        at_full = PowerGate(FixedPower("full"), FixedTiers({}), lambda: NO_MODEL_MESSAGE)
        smart_only = PowerGate(
            FixedPower("full"), FixedTiers({"smart": PROVIDER}), lambda: NO_MODEL_MESSAGE
        )

        assert (at_off.level(), at_off.model_available()) == ("off", True)
        assert (at_full.level(), at_full.model_available()) == ("full", False)
        assert smart_only.model_available() is False

    def test_the_refusal_message_is_the_injected_refusals_text(self):
        gate = PowerGate(FixedPower("off"), FixedTiers({}), lambda: "a refusal")

        assert gate.refusal_message() == "a refusal"

    def test_the_yield_refusal_names_the_yield_while_gpu_yield_is_yielded(self):
        assert gpu_yield_refusal(lambda: "yielded")() == YIELDED_MESSAGE

    def test_the_yield_refusal_names_the_missing_model_otherwise(self):
        assert gpu_yield_refusal(lambda: "normal")() == NO_MODEL_MESSAGE


# -- Typed turns -----------------------------------------------------------------------------------


@pytest.mark.asyncio
class TestTypedTurns:
    async def test_at_off_with_a_provider_a_typed_question_still_reaches_it(self):
        rig = await PowerRig(
            power=FixedPower("off"), tiers=FixedTiers({"interaction": PROVIDER})
        ).start()

        await rig.send({"type": "text", "text": "what is a quasar"})

        assert [r.user_text for r in rig.responder.requests] == ["what is a quasar"]
        assert rig.responder.capability_calls == []
        assert "stream_complete" in rig.names()

    async def test_at_full_with_no_provider_a_typed_question_gets_the_refusal(self):
        rig = await PowerRig(
            power=FixedPower("full"), tiers=FixedTiers({}), responder=FailingResponder()
        ).start()

        await rig.send({"type": "text", "text": "what is a quasar"})

        errors = rig.errors()
        assert [(e["kind"], e["message"], e["retriable"]) for e in errors] == [
            ("server", NO_MODEL_MESSAGE, True)
        ]
        assert rig.task is not None and not rig.task.done()

    async def test_a_typed_message_goes_to_the_capability_only_path(self):
        capability = FakeCapabilityOnly(HANDLED)
        rig = await PowerRig(**no_model(), responder=FailingResponder(capability)).start()

        await rig.send({"type": "text", "text": "lights on"})

        assert capability.calls == [("lights on", "s1")]
        starts = rig.of("stream_start")
        assert [(s["channel"], s["origin"]) for s in starts] == [("text", "text")]
        names = rig.names()
        assert names.index("tool_call_started") < names.index("stream_complete")
        assert rig.tokens() == ["Lights on."]
        assert rig.errors() == []

    async def test_an_unhandled_message_gets_its_drained_events_then_the_refusal(self):
        capability = FakeCapabilityOnly(UNHANDLED)
        rig = await PowerRig(**no_model(), responder=FailingResponder(capability)).start()

        await rig.send({"type": "text", "text": "write me a poem"})

        names = rig.names()
        assert names.index("action_resolved") < names.index("error")
        assert [e["message"] for e in rig.errors()] == [NO_MODEL_MESSAGE]
        assert rig.tokens() == []

    async def test_the_refusal_names_the_yield_while_gpu_yield_is_yielded(self):
        rig = await PowerRig(
            **no_model(),
            refusal=gpu_yield_refusal(lambda: "yielded"),
            responder=FailingResponder(),
        ).start()

        await rig.send({"type": "text", "text": "hello"})

        assert [e["message"] for e in rig.errors()] == [YIELDED_MESSAGE]

    async def test_the_refusal_names_the_missing_model_when_the_gpu_is_not_yielded(self):
        rig = await PowerRig(
            **no_model(),
            refusal=gpu_yield_refusal(lambda: "normal"),
            responder=FailingResponder(),
        ).start()

        await rig.send({"type": "text", "text": "hello"})

        assert [e["message"] for e in rig.errors()] == [NO_MODEL_MESSAGE]


# -- Voice -----------------------------------------------------------------------------------------


@pytest.mark.asyncio
class TestVoiceAtOff:
    async def test_each_session_gets_its_own_voice_state_at_off(self):
        first = await PowerRig(**no_model(), session_id="s1").start()
        second = await PowerRig(**no_model(), session_id="s2").start()
        await first.send({"type": "mute", "active": True})

        await first.broadcast()
        await second.broadcast()

        a, b = first.voice_states("broadcast")[-1], second.voice_states("broadcast")[-1]
        assert (a["power_level"], a["muted"]) == ("off", True)
        assert (b["power_level"], b["muted"]) == ("off", False)

    async def test_spoken_commands_allowed_answer_voice_start_active_at_off(self):
        rig = await PowerRig(**no_model()).start()

        await rig.send(voice_start())

        answer = rig.voice_states("answer")[0]
        assert (answer["active"], answer["refused"], answer["power_level"]) == (True, None, "off")
        await rig.broadcast()
        state = rig.voice_states("broadcast")[-1]
        assert (state["active"], state["power_level"]) == (True, "off")

    async def test_with_the_setting_off_voice_start_is_refused_power_off(self):
        rig = await PowerRig(**no_model(), spoken_commands_at_off=False).start()

        await rig.send(voice_start())

        answer = rig.voice_states("answer")[0]
        assert (answer["active"], answer["refused"]) == (False, "power_off")
        assert rig.ingresses == []

    async def test_with_stt_on_cuda_voice_start_is_refused_power_off(self):
        rig = await PowerRig(**no_model(), statuses=STT_ON_CUDA).start()

        await rig.send(voice_start())

        answer = rig.voice_states("answer")[0]
        assert (answer["active"], answer["refused"]) == (False, "power_off")

    async def test_a_final_transcript_goes_to_capability_only_and_is_spoken(self):
        capability = FakeCapabilityOnly(HANDLED)
        rig = await PowerRig(**no_model(), responder=FailingResponder(capability)).start()
        await rig.send(voice_start())

        await rig.utterance("lights on")

        assert capability.calls == [("lights on", "s1")]
        starts = rig.of("stream_start")
        assert [(s["channel"], s["origin"]) for s in starts] == [("tts", "voice")]
        assert rig.tokens() == ["Lights on."]
        assert any(name == "0x01" for name in rig.names())

    async def test_an_unhandled_utterance_gets_the_envelope_and_its_text_is_spoken(self):
        capability = FakeCapabilityOnly(UNHANDLED)
        rig = await PowerRig(**no_model(), responder=FailingResponder(capability)).start()
        await rig.send(voice_start())

        await rig.utterance("write me a poem")

        assert [e["message"] for e in rig.errors()] == [NO_MODEL_MESSAGE]
        assert rig.tokens() == [NO_MODEL_MESSAGE]
        assert rig.of("stream_start")[0]["channel"] == "tts"
        assert any(name == "0x01" for name in rig.names())

    async def test_an_endpoint_candidate_opens_no_turn_while_the_tier_is_off(self):
        rig = await PowerRig(**no_model(), responder=FailingResponder()).start()
        await rig.send(voice_start())
        uid = rig.ids()
        rig.ingress.emit(rig.ingress.started(uid))

        rig.ingress.emit(rig.ingress.candidate(uid, "lights on"))
        await settle()

        assert rig.of("stream_start") == []


@pytest.mark.asyncio
class TestTheLevelDrops:
    def power(self, provider: bool = True) -> tuple[Level, SingleModelTiers]:
        level = Level("full")
        return level, SingleModelTiers(lambda: PROVIDER if provider else None, level)

    async def test_an_open_utterance_closes_with_dropped_power_off(self):
        level, tiers = self.power()
        rig = await PowerRig(power=level, tiers=tiers, spoken_commands_at_off=False).start()
        await rig.send(voice_start())
        uid = rig.ids()
        rig.ingress.emit(rig.ingress.started(uid))
        await settle()
        rig.ingress.close_result = rig.ingress.dropped(uid, "voice_stop", text="lights")

        level.value = "off"
        await rig.broadcast()

        dropped = rig.of("voice_decision:dropped")
        assert [(d["reason"], d["detail"]["utterance_id"]) for d in dropped] == [("power_off", uid)]
        assert ("close",) in rig.ingresses[0].calls
        assert rig.of("stream_start") == []

    async def test_an_utterance_ending_after_the_drop_closes_with_power_off_not_a_turn(self):
        level, tiers = self.power()
        rig = await PowerRig(power=level, tiers=tiers, spoken_commands_at_off=False).start()
        await rig.send(voice_start())
        uid = rig.ids()
        ingress = rig.ingress
        ingress.emit(ingress.started(uid))
        await settle()

        level.value = "off"
        ingress.emit(ingress.ended(uid, "lights on"))
        await settle()

        assert [d["reason"] for d in rig.of("voice_decision:dropped")] == ["power_off"]
        assert rig.of("stream_start") == []

    async def test_active_voice_reads_inactive_refused_power_off_when_not_allowed(self):
        level, tiers = self.power()
        rig = await PowerRig(power=level, tiers=tiers, spoken_commands_at_off=False).start()
        await rig.send(voice_start())

        level.value = "off"
        await rig.broadcast()

        state = rig.voice_states("broadcast")[-1]
        assert (state["active"], state["refused"], state["power_level"]) == (
            False,
            "power_off",
            "off",
        )
        assert any(isinstance(m, VoiceStop) for m in rig.controller.posted)

    async def test_active_voice_stays_active_at_off_when_spoken_commands_are_allowed(self):
        level, tiers = self.power()
        rig = await PowerRig(power=level, tiers=tiers).start()
        await rig.send(voice_start())

        level.value = "off"
        await rig.broadcast()

        state = rig.voice_states("broadcast")[-1]
        assert (state["active"], state["refused"], state["power_level"]) == (True, None, "off")
        assert not any(isinstance(m, VoiceStop) for m in rig.controller.posted)

    async def test_on_the_rise_voice_reads_active_again_with_no_new_voice_start(self):
        level, tiers = self.power()
        rig = await PowerRig(power=level, tiers=tiers, spoken_commands_at_off=False).start()
        await rig.send(voice_start())
        level.value = "off"
        await rig.broadcast()

        level.value = "full"
        await rig.broadcast()

        state = rig.voice_states("broadcast")[-1]
        assert (state["active"], state["refused"], state["power_level"]) == (True, None, "full")
        assert len(rig.voice_states("answer")) == 1
        starts = [m for m in rig.controller.posted if isinstance(m, VoiceStart)]
        assert len(starts) == 2
        assert len(rig.ingresses) == 2
        assert rig.session.voice_on

    async def test_a_voice_stop_while_held_off_forgets_the_kept_voice_start(self):
        level, tiers = self.power()
        rig = await PowerRig(power=level, tiers=tiers, spoken_commands_at_off=False).start()
        await rig.send(voice_start())
        level.value = "off"
        await rig.broadcast()

        await rig.send({"type": "voice_stop"})
        level.value = "full"
        await rig.broadcast()

        assert rig.voice_states("broadcast")[-1]["active"] is False
        assert len(rig.ingresses) == 1


@pytest.mark.asyncio
class TestServices:
    async def test_the_controller_receives_the_services_alarm_ringing(self):
        def ringing() -> bool:
            return True

        rig = await PowerRig(**no_model(), alarm_ringing=ringing).start()

        assert rig.controller.kwargs["alarm_ringing"] is ringing

    async def test_without_a_gate_the_level_comes_from_power_level(self):
        services = SessionServices(
            responder=FakeResponder(),  # type: ignore[arg-type]
            engines=SpeechEngines(statuses=ALL_OFF),
            lanes=InferenceLanes(
                vad=InlineLane(),
                eot=InlineLane(),
                stt=InlineLane(),
                tts=InlineLane(),
                kws=InlineLane(),
            ),
            settings=VoiceSettings(),
            placement=floor_placement,
        )

        assert services.power_gate is None
        assert services.alarm_ringing() is False
        assert not hasattr(services, "capability_only")  # the responder's port carries it


class HeldCapability(FakeCapabilityOnly):
    """The capability-only path whose first call waits, so its reply is still being formed."""

    def __init__(self) -> None:
        super().__init__(None)
        self.first = asyncio.Event()

    async def __call__(self, text: str, session_id: str) -> CapabilityOnlyTurn | None:
        self.calls.append((text, session_id))
        if len(self.calls) == 1:
            await self.first.wait()
        return None


async def say(rig: PowerRig, text: str) -> None:
    uid = rig.ids()
    rig.ingress.emit(rig.ingress.started(uid))
    rig.ingress.emit(rig.ingress.ended(uid, text))
    await settle()


@pytest.mark.asyncio
class TestAlarmStop:
    async def test_stop_while_ringing_reaches_the_inner_port_with_the_tier_available(self, caplog):
        responder = FakeResponder([Token("Hello there. "), Pause()])
        rig = await PowerRig(
            power=FixedPower("full"),
            tiers=FixedTiers({"interaction": PROVIDER}),
            responder=responder,
            alarm_ringing=lambda: True,
        ).start()
        await rig.send(voice_start())
        await say(rig, "tell me a story")
        assert rig.controller.phase in ("thinking", "speaking")

        with caplog.at_level(logging.ERROR, logger=TURN_LOGGER):
            await say(rig, "stop")

        assert responder.capability_calls == [("stop", "s1")]
        assert [b["reason"] for b in rig.of("voice_decision:barge_in")] == ["keyword"]
        assert "capability-only path failed" not in caplog.text
        assert rig.task is not None and not rig.task.done()

    async def test_stop_while_ringing_reaches_the_inner_port_with_the_tier_unavailable(
        self, caplog
    ):
        capability = HeldCapability()
        rig = await PowerRig(
            **no_model(), responder=FailingResponder(capability), alarm_ringing=lambda: True
        ).start()
        await rig.send(voice_start())
        await say(rig, "lights on")
        assert rig.controller.phase in ("thinking", "speaking")

        with caplog.at_level(logging.ERROR, logger=TURN_LOGGER):
            await say(rig, "stop")

        assert capability.calls == [("lights on", "s1"), ("stop", "s1")]
        assert [b["reason"] for b in rig.of("voice_decision:barge_in")] == ["keyword"]
        assert "capability-only path failed" not in caplog.text
        assert rig.task is not None and not rig.task.done()
