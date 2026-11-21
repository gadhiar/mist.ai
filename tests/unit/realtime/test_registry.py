"""T12: the session registry: routing, log streaming, voice activity and wake arbitration."""

from __future__ import annotations

import asyncio
import json
from typing import Literal

import pytest
import pytest_asyncio

from backend.chat.stream_events import Token
from backend.realtime.budget import BargeInConfig, VoiceSettings
from backend.realtime.capability_sink import CapabilitySink
from backend.realtime.registry import SessionRegistry
from backend.realtime.session import InferenceLanes, RealtimeSession, SessionServices, SpeechEngines
from backend.realtime.turn import WakeArbiter, WakeRank
from tests.fakes.speech import (
    EnergyVad,
    FakeSynthesizer,
    ScriptedNameSpotter,
    ScriptedTranscriber,
    ScriptedWakeVerifier,
)

from .test_session import FakeIngress, FakeSocket, RecordingController, name_of, voice_start
from .turn_fakes import (
    TEST_PLAYBACK,
    TEST_SEGMENTER,
    FakeCallLater,
    FakeClock,
    FakeResponder,
    IdCounter,
    InlineLane,
    floor_placement,
    settle,
)

pytestmark = pytest.mark.asyncio


class RecordingLogGate:
    def __init__(self) -> None:
        self.streaming: list[tuple[bool, list[str] | None]] = []
        self.levels: list[tuple[str, str]] = []

    def set_streaming(self, enabled: bool, levels: list[str] | None) -> None:
        self.streaming.append((enabled, levels))

    def set_logger_level(self, logger_name: str, level: str) -> None:
        self.levels.append((logger_name, level))


class Connected:
    def __init__(self, session: RealtimeSession, socket: FakeSocket, task: asyncio.Task[None]):
        self.session = session
        self.socket = socket
        self.task = task

    async def send(self, *payloads: dict[str, object]) -> None:
        for payload in payloads:
            self.socket.text(payload)
        await settle()

    def names(self) -> list[str]:
        return [name_of(item) for item in self.socket.sent]

    async def leave(self) -> None:
        self.socket.disconnect()
        await asyncio.wait_for(self.task, timeout=5)


class RegistryRig:
    def __init__(self, *scripts: object) -> None:
        self.clock = FakeClock()
        self.ids = IdCounter()
        self.later = FakeCallLater()
        self.gate = RecordingLogGate()
        self.responder = FakeResponder(*scripts, clock=self.clock)  # type: ignore[arg-type]
        self.ingresses: dict[int, FakeIngress] = {}
        self.controllers: list[RecordingController] = []
        self.settings = VoiceSettings(
            playback=TEST_PLAYBACK,
            segmenter=TEST_SEGMENTER,
            barge_in=BargeInConfig(arm_delay_ms=0, wake_arbitration_ms=300),
        )
        services = SessionServices(
            responder=self.responder,  # type: ignore[arg-type]
            engines=SpeechEngines(
                vad=EnergyVad(),
                transcriber=ScriptedTranscriber(["hello"]),
                synthesizer=FakeSynthesizer(),
                spotter=ScriptedNameSpotter(),
                verifier=ScriptedWakeVerifier(),
            ),
            lanes=InferenceLanes(
                vad=InlineLane(),
                eot=InlineLane(),
                stt=InlineLane(),
                tts=InlineLane(),
                kws=InlineLane(),
                verify=InlineLane(),
            ),
            settings=self.settings,
            placement=floor_placement,
            clock_factory=lambda: self.clock,
            utterance_ids=lambda: self.ids,
            ingress_factory=self._ingress,
            controller_factory=self._controller,
        )
        self.registry = SessionRegistry(services, log_gate=self.gate, call_later=self.later)
        self.connected: list[Connected] = []
        RIGS.append(self)

    def _ingress(self, **kwargs: object) -> FakeIngress:
        ingress = FakeIngress(kwargs)
        self.ingresses[id(kwargs["emit"].__self__)] = ingress  # type: ignore[attr-defined]
        return ingress

    def _controller(self, **kwargs: object) -> RecordingController:
        controller = RecordingController(**kwargs)
        self.controllers.append(controller)
        return controller

    def ingress_of(self, client: Connected) -> FakeIngress:
        return self.ingresses[id(client.session)]

    async def connect(
        self,
        session_id: str,
        endpoint: str = "desktop",
        endpoint_kind: Literal["desktop", "room"] | None = None,
    ) -> Connected:
        socket = FakeSocket()
        session = self.registry.create_session(
            session_id=session_id, websocket=socket, endpoint=endpoint, endpoint_kind=endpoint_kind
        )
        task = asyncio.get_running_loop().create_task(session.run())
        client = Connected(session, socket, task)
        self.connected.append(client)
        await settle()
        return client


RIGS: list[RegistryRig] = []


@pytest_asyncio.fixture(autouse=True)
async def _disconnect_all():
    yield
    while RIGS:
        rig = RIGS.pop()
        for client in rig.connected:
            if not client.task.done():
                await client.leave()


class TestRouting:
    async def test_count_follows_connects_and_disconnects(self):
        rig = RegistryRig()
        a = await rig.connect("a")
        await rig.connect("b")
        assert rig.registry.count() == 2

        await a.leave()

        assert rig.registry.count() == 1

    async def test_broadcast_reaches_every_bulk_lane_or_one_endpoints(self):
        rig = RegistryRig()
        desk = await rig.connect("a", "desktop")
        room = await rig.connect("b", "room")

        rig.registry.broadcast({"type": "alarm_state", "message": "all"})
        rig.registry.broadcast({"type": "alarm_state", "message": "rooms"}, endpoint="room")
        await settle()

        assert [m["message"] for m in desk.socket.sent] == ["all"]
        assert [m["message"] for m in room.socket.sent] == ["all", "rooms"]

    async def test_a_status_broadcast_reaches_the_desktop_and_not_a_room(self):
        rig = RegistryRig()
        desk = await rig.connect("a", "desktop")
        room = await rig.connect("b", "example-room", "room")

        rig.registry.broadcast({"type": "status", "message": "all"})
        await settle()

        assert desk.names() == ["status"]
        assert room.socket.sent == []
        assert (desk.session.withheld, room.session.withheld) == (0, 1)

    async def test_send_to_reaches_one_session(self):
        rig = RegistryRig()
        a = await rig.connect("a")
        b = await rig.connect("b")

        assert rig.registry.send_to("b", {"type": "status", "message": "hi"}) is True
        assert rig.registry.send_to("gone", {"type": "status", "message": "hi"}) is False
        await settle()

        assert a.socket.sent == []
        assert b.names() == ["status"]

    async def test_endpoint_present(self):
        rig = RegistryRig()
        room = await rig.connect("a", "room")

        assert rig.registry.endpoint_present("room") is True
        assert rig.registry.endpoint_present("desktop") is False
        await room.leave()
        assert rig.registry.endpoint_present("room") is False

    async def test_endpoint_of_names_each_sessions_endpoint_until_it_leaves(self):
        rig = RegistryRig()
        desk = await rig.connect("a", "desktop")
        device = await rig.connect("b", "example-room", "room")

        bound = (rig.registry.endpoint_of("a"), rig.registry.endpoint_of("b"))
        await device.leave()

        assert bound == ("desktop", "example-room")
        assert rig.registry.endpoint_of("b") is None
        assert rig.registry.endpoint_of("never") is None
        assert desk.session.endpoint_kind == "desktop"

    async def test_create_session_passes_the_kind_through(self):
        rig = RegistryRig()

        device = await rig.connect("a", "example-room", "room")

        assert (device.session.endpoint, device.session.endpoint_kind) == ("example-room", "room")
        assert rig.registry.endpoint_present("example-room") is True
        assert rig.registry.endpoint_present("room") is False

    async def test_a_broadcast_to_a_device_name_reaches_that_device_only(self):
        rig = RegistryRig()
        other = await rig.connect("a", "other-room", "room")
        device = await rig.connect("b", "example-room", "room")

        rig.registry.broadcast({"type": "alarm_fired", "message": "ring"}, endpoint="example-room")
        await settle()

        assert [m["message"] for m in device.socket.sent] == ["ring"]
        assert other.socket.sent == []

    async def test_a_room_gets_the_alarm_broadcasts_and_results_but_not_status_or_proposals(self):
        rig = RegistryRig()
        desk = await rig.connect("a", "desktop")
        room = await rig.connect("b", "example-room", "room")
        sink = CapabilitySink()
        sink.attach(rig.registry, asyncio.get_running_loop())

        # As the broadcaster routes `message_queue`: JSON text to every session.
        rig.registry.broadcast(json.dumps({"type": "heartbeat", "timestamp": 1}))
        rig.registry.broadcast(json.dumps({"type": "system_status", "timestamp": 1}))
        # As the turn controller sends a turn's proposal: on that session's own `turn` lane.
        for client in (desk, room):
            session_id = client.session.session_id
            proposal = {"type": "action_proposed", "proposal_id": "p1", "session_id": session_id}
            client.session.send_turn("turn-1", proposal)
        # As capabilities emit through their sink, which sends `alarm_fired` to the desktop.
        sink.emit({"type": "capabilities_status", "rev": 1})
        sink.emit({"type": "alarm_state", "rev": 1, "ringing": []})
        sink.emit({"type": "alarm_fired", "alarm_id": "a1"})
        # As a device's ring sink rings it, and as an alarm action is answered.
        rig.registry.broadcast(
            json.dumps({"type": "alarm_fired", "alarm_id": "a1"}), endpoint="example-room"
        )
        for session_id in ("a", "b"):
            rig.registry.send_to(session_id, {"type": "alarm_action_result", "alarm_id": "a1"})
        await settle()

        assert sorted(desk.names()) == sorted(
            [
                "alarm_action_result",
                "heartbeat",
                "system_status",
                "capabilities_status",
                "action_proposed",
                "alarm_state",
                "alarm_fired",
            ]
        )
        assert sorted(room.names()) == ["alarm_action_result", "alarm_fired", "alarm_state"]
        assert (desk.session.withheld, room.session.withheld) == (0, 4)

    async def test_each_session_gets_its_own_voice_state_on_a_change(self):
        rig = RegistryRig()
        a = await rig.connect("a")
        b = await rig.connect("b")
        await a.send({"type": "mute", "active": True})
        sent_before = len(a.socket.sent)

        rig.registry.broadcast_voice_state()
        await settle()

        latest_a = a.socket.sent[sent_before:]
        assert [(m["cause"], m["req"], m["muted"]) for m in latest_a] == [("broadcast", None, True)]
        assert [(m["cause"], m["muted"]) for m in b.socket.sent] == [("broadcast", False)]


class TestLogs:
    async def test_a_log_line_reaches_only_sessions_subscribed_at_its_level(self):
        rig = RegistryRig()
        errors = await rig.connect("a")
        everything = await rig.connect("b")
        quiet = await rig.connect("c")
        await errors.send({"type": "subscribe_logs", "enabled": True, "levels": ["error"]})
        await everything.send({"type": "subscribe_logs", "enabled": True, "levels": None})

        rig.registry.publish_log('{"type": "log", "level": "INFO"}', "INFO")
        rig.registry.publish_log('{"type": "log", "level": "ERROR"}', "ERROR")
        await settle()

        logs = lambda c: [m["level"] for m in c.socket.sent if m["type"] == "log"]  # noqa: E731
        assert logs(errors) == ["ERROR"]
        assert logs(everything) == ["INFO", "ERROR"]
        assert logs(quiet) == []

    async def test_the_log_gate_gets_the_union_of_levels_and_turns_off_with_the_last(self):
        rig = RegistryRig()
        a = await rig.connect("a")
        b = await rig.connect("b")

        await a.send({"type": "subscribe_logs", "enabled": True, "levels": ["ERROR"]})
        await b.send({"type": "subscribe_logs", "enabled": True, "levels": ["WARNING"]})
        await a.send({"type": "subscribe_logs", "enabled": False, "levels": None})
        await b.leave()

        assert rig.gate.streaming == [
            (True, ["ERROR"]),
            (True, ["ERROR", "WARNING"]),
            (True, ["WARNING"]),
            (False, None),
        ]

    async def test_any_subscriber_without_levels_opens_every_level(self):
        rig = RegistryRig()
        a = await rig.connect("a")
        b = await rig.connect("b")

        await a.send({"type": "subscribe_logs", "enabled": True, "levels": ["ERROR"]})
        await b.send({"type": "subscribe_logs", "enabled": True, "levels": None})

        assert rig.gate.streaming[-1] == (True, None)

    async def test_log_config_sets_the_loggers_level_on_the_gate(self):
        rig = RegistryRig()
        a = await rig.connect("a")

        await a.send({"type": "log_config", "action": "set_level", "level": "DEBUG", "logger": "x"})

        assert rig.gate.levels == [("x", "DEBUG")]


class TestVoiceActivity:
    async def test_activity_follows_any_sessions_open_utterance_and_fires_on_change(self):
        rig = RegistryRig()
        a = await rig.connect("a")
        await rig.connect("b")
        seen: list[bool] = []
        unsubscribe = rig.registry.subscribe_voice_activity(seen.append)
        await a.send(voice_start())
        ingress = rig.ingress_of(a)
        assert rig.registry.voice_activity() is False

        ingress.holding = True
        ingress.emit(ingress.started(rig.ids()))
        await settle()
        assert rig.registry.voice_activity() is True

        ingress.holding = False
        ingress.emit(ingress.dropped(rig.ids.issued[-1]))
        await settle()
        unsubscribe()
        ingress.holding = True
        ingress.emit(ingress.started(rig.ids()))
        await settle()

        assert seen == [True, False]

    async def test_a_disconnect_ends_its_activity(self):
        rig = RegistryRig()
        a = await rig.connect("a")
        seen: list[bool] = []
        rig.registry.subscribe_voice_activity(seen.append)
        await a.send(voice_start())
        ingress = rig.ingress_of(a)
        ingress.holding = True
        ingress.emit(ingress.started(rig.ids()))
        await settle()

        await a.leave()

        assert seen == [True, False]
        assert rig.registry.voice_activity() is False


class TestEchoText:
    async def test_text_spoken_in_one_session_reaches_another_sessions_controller(self):
        rig = RegistryRig([Token("I'm Mist.")])
        a = await rig.connect("a")
        await rig.connect("b")
        await a.send(voice_start())
        ingress = rig.ingress_of(a)
        uid = rig.ids()
        ingress.emit(ingress.started(uid))
        ingress.emit(ingress.ended(uid, "who are you"))
        await settle()
        rig.clock.now += 400

        b_controller = rig.controllers[1]
        assert "I'm Mist" in b_controller.kwargs["recent_tts_text"](5000)


class TestWakeArbitration:
    async def test_every_controller_holds_the_registrys_one_arbiter(self):
        rig = RegistryRig()
        await rig.connect("a")
        await rig.connect("b")

        first, second = rig.controllers
        assert isinstance(rig.registry.arbiter, WakeArbiter)
        assert first.kwargs["arbiter"] is second.kwargs["arbiter"] is rig.registry.arbiter

    async def test_with_one_voice_session_a_claim_proceeds_with_no_timer(self):
        rig = RegistryRig()
        a = await rig.connect("a")
        await rig.connect("b")  # connected, voice off: not a voice session
        await a.send(voice_start())
        verdicts: list[tuple[bool, float | None]] = []

        rig.registry.arbiter.claim(
            "a", WakeRank(0.0, None, None, 0.2, 0.7), lambda ok, w: verdicts.append((ok, w))
        )

        assert verdicts == [(True, None)]
        assert rig.later.timers == []

    @pytest.mark.parametrize(
        ("rank_a", "rank_b", "winner"),
        [
            (WakeRank(0.0, 0.5, 0.6, 0.1, 0.71), WakeRank(10.0, 2.0, 0.9, 0.3, 0.93), "a"),
            (WakeRank(0.0, 1.0, 0.4, 0.3, 0.71), WakeRank(10.0, 1.0, 0.8, 0.1, 0.93), "b"),
        ],
        ids=["fewer_false_accepts_wins", "tied_false_accepts_the_quantile_decides"],
    )
    async def test_two_voice_sessions_arbitrate_within_the_window(self, rank_a, rank_b, winner):
        rig = RegistryRig()
        a = await rig.connect("a")
        b = await rig.connect("b")
        await a.send(voice_start())
        await b.send(voice_start())
        verdicts: list[tuple[str, bool, float | None]] = []

        claim_a = rig.registry.arbiter.claim(
            "a", rank_a, lambda ok, w: verdicts.append(("a", ok, w))
        )
        claim_b = rig.registry.arbiter.claim(
            "b", rank_b, lambda ok, w: verdicts.append(("b", ok, w))
        )

        assert [delay for delay, _ in rig.later.timers] == [0.3]
        rig.later.fire()
        assert verdicts == [(winner, True, None)]

        # The winner's gate accepts it: the loser hears the winner's raw score.
        won, lost, won_rank = (claim_a, "b", rank_a) if winner == "a" else (claim_b, "a", rank_b)
        rig.registry.arbiter.settle(won, True)
        assert verdicts == [(winner, True, None), (lost, False, won_rank.score)]

    async def test_a_withdrawn_claim_gets_no_loss_when_the_winner_is_accepted(self):
        rig = RegistryRig()
        a = await rig.connect("a")
        b = await rig.connect("b")
        await a.send(voice_start())
        await b.send(voice_start())
        verdicts: list[tuple[str, bool, float | None]] = []
        arbiter = rig.registry.arbiter

        won = arbiter.claim(
            "a", WakeRank(0.0, 0.5, 0.6, 0.1, 0.9), lambda ok, w: verdicts.append(("a", ok, w))
        )
        gone = arbiter.claim(
            "b", WakeRank(10.0, 2.0, 0.9, 0.3, 0.8), lambda ok, w: verdicts.append(("b", ok, w))
        )
        rig.later.fire()
        arbiter.settle(gone, False)  # b's candidate was dropped while a's gate decoded
        arbiter.settle(won, True)

        assert verdicts == [("a", True, None)]
