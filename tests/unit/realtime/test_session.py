"""T12: one `RealtimeSession` per WebSocket, with its prioritized outbound lanes.

The session runs the real `TurnController` against a fake socket. Ingress is a fake built by the
session's ingress factory (a `RecordingIngress` that also takes frames), except in the tests that
name the real `AudioIngress`. The reconfigure tests drive the controller with ingress events
rather than synthetic audio: what they check is the session's routing and answers.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from typing import Literal

import pytest
import pytest_asyncio

from backend.chat.stream_events import Token
from backend.realtime.addressee import AddresseeGate
from backend.realtime.budget import BargeInConfig, VoiceSettings
from backend.realtime.ingress import AudioIngress, IngressSeed
from backend.realtime.playback import LanePurge
from backend.realtime.protocol import (
    Direction,
    Frame,
    MsgType,
    StreamStartMessage,
    decode_frame,
    encode_frame,
)
from backend.realtime.session import (
    InferenceLanes,
    RealtimeSession,
    SessionServices,
    SpeechEngines,
)
from backend.realtime.speech import SpotterCalibration
from backend.realtime.turn import (
    Interrupt,
    Mute,
    Reconfigure,
    Text,
    TurnController,
    VoiceStart,
    VoiceStop,
    Wake,
)
from tests.fakes.speech import (
    EnergyVad,
    FakeSynthesizer,
    ScriptedNameSpotter,
    ScriptedTranscriber,
)

from .turn_fakes import (
    TEST_PLAYBACK,
    TEST_SEGMENTER,
    FakeClock,
    FakeResponder,
    FakeWakeArbiter,
    IdCounter,
    InlineLane,
    RecordingIngress,
    floor_placement,
    settle,
)

pytestmark = pytest.mark.asyncio

CALIBRATION = {
    "server": SpotterCalibration(false_accepts=((0.5, 1.0),), positive_quantile=((0.5, 0.5),)),
    "endpoint": SpotterCalibration(false_accepts=((0.6, 2.0),), positive_quantile=((0.6, 0.4),)),
}


# -- Fakes ---------------------------------------------------------------------------------------


class FakeSocket:
    """A Starlette-shaped socket: ASGI receive dicts in, decoded text and frames out.

    `hold(when)` makes the next write the predicate matches wait until `release()`, as a slow
    client would; the item is recorded only once it is written.
    """

    def __init__(self) -> None:
        self.inbox: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        self.sent: list[object] = []
        self.closed_with: int | None = None
        self.fail_writes = False
        self._hold_when: Callable[[object], bool] | None = None
        self._gate: asyncio.Event | None = None
        self.holding: object | None = None

    async def receive(self) -> dict[str, object]:
        return await self.inbox.get()

    async def send_text(self, data: str) -> None:
        try:
            item: object = json.loads(data)
        except ValueError:
            item = data  # a raw bulk line
        await self._write(item)

    async def send_bytes(self, data: bytes) -> None:
        await self._write(decode_frame(data, direction=Direction.DOWN))

    async def close(self, code: int = 1000) -> None:
        self.closed_with = code

    async def _write(self, item: object) -> None:
        if self.fail_writes:
            raise RuntimeError("socket write failed")
        if self._hold_when is not None and self._hold_when(item):
            self._hold_when = None
            self._gate = asyncio.Event()
            self.holding = item
            await self._gate.wait()
            self.holding = None
        self.sent.append(item)

    def hold(self, when: Callable[[object], bool] = lambda _item: True) -> None:
        self._hold_when = when

    def release(self) -> None:
        if self._gate is not None:
            self._gate.set()
            self._gate = None

    def text(self, payload: dict[str, object] | str) -> None:
        raw = payload if isinstance(payload, str) else json.dumps(payload)
        self.inbox.put_nowait({"type": "websocket.receive", "text": raw})

    def binary(self, data: bytes) -> None:
        self.inbox.put_nowait({"type": "websocket.receive", "bytes": data})

    def disconnect(self) -> None:
        self.inbox.put_nowait({"type": "websocket.disconnect", "code": 1000})


def name_of(item: object) -> str:
    if isinstance(item, Frame):
        return f"0x{item.msg_type:02x}"
    assert isinstance(item, dict)
    if item["type"] == "voice_decision":
        return f"voice_decision:{item['kind']}"
    return str(item["type"])


class FakeIngress(RecordingIngress):
    """The ingress factory's product: records frames and references, and can drop on mute."""

    def __init__(self, kwargs: dict[str, object]) -> None:
        super().__init__(kwargs["new_utterance_id"], epoch=kwargs["epoch"])  # type: ignore[arg-type]
        self.kwargs = kwargs
        self.emit: Callable[[object], None] = kwargs["emit"]  # type: ignore[assignment]
        self.open_utterance_id: str | None = None

    def push_frame(self, frame: Frame, pos: int) -> None:
        self.calls.append(("frame", frame.seq, pos))

    def push_reference(self, seq: int, pcm: bytes) -> None:
        self.calls.append(("reference", seq))

    def set_muted(self, muted: bool) -> None:
        # The real ingress emits the open utterance's drop before `set_muted` returns.
        if muted and self.open_utterance_id is not None:
            self.emit(self.dropped(self.open_utterance_id, "muted"))
            self.open_utterance_id = None
        super().set_muted(muted)

    def handoff(self) -> IngressSeed:
        self.calls.append(("handoff",))
        return IngressSeed(frames=(), floor=7, vad_from=None)


class RecordingController(TurnController):
    """The real controller, with every message posted to its mailbox recorded."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.kwargs = kwargs
        self.posted: list[object] = []

    def post(self, message: object) -> None:
        self.posted.append(message)
        super().post(message)  # type: ignore[arg-type]


class RecordingHost:
    def __init__(self) -> None:
        self.arbiter = FakeWakeArbiter(solo=True)
        self.closed: list[str] = []
        self.activity: list[bool] = []
        self.subscriptions: list[tuple[bool, list[str] | None]] = []
        self.levels: list[tuple[str, str]] = []
        self.tts_text = ""

    def recent_tts_text(self, window_ms: int) -> str:
        return self.tts_text

    def session_closed(self, session: RealtimeSession) -> None:
        self.closed.append(session.session_id)

    def voice_activity_changed(self, session: RealtimeSession) -> None:
        self.activity.append(session.voice_activity)

    def set_log_subscription(
        self, session: RealtimeSession, levels: list[str] | None, enabled: bool
    ) -> None:
        self.subscriptions.append((enabled, levels))

    def set_logger_level(self, logger_name: str, level: str) -> None:
        self.levels.append((logger_name, level))


class Rig:
    def __init__(
        self,
        *scripts: object,
        spotter: ScriptedNameSpotter | None | bool = True,
        synthesizer: bool = True,
        vad: bool = True,
        endpoint: Literal["desktop", "room"] = "desktop",
        max_turn_items: int = 2048,
        real_ingress: bool = False,
        session_id: str = "s1",
        capability: Callable[[str, object], None] | None = None,
    ) -> None:
        self.clock = FakeClock()
        self.ids = IdCounter()
        self.socket = FakeSocket()
        self.responder = FakeResponder(*scripts, clock=self.clock)  # type: ignore[arg-type]
        self.host = RecordingHost()
        self.ingresses: list[FakeIngress] = []
        self.real_ingresses: list[AudioIngress] = []
        self.controllers: list[RecordingController] = []
        if spotter is True:
            spotter = ScriptedNameSpotter(hit_lag_ms=180, calibration=CALIBRATION)
        self.spotter = spotter or None
        self.engines = SpeechEngines(
            vad=EnergyVad() if vad else None,
            turn_end=None,
            transcriber=ScriptedTranscriber(["hello there"], ms_per_word=20),
            synthesizer=FakeSynthesizer() if synthesizer else None,
            spotter=self.spotter,
        )
        self.tts_lane = InlineLane()
        self.settings = VoiceSettings(
            playback=TEST_PLAYBACK,
            segmenter=TEST_SEGMENTER,
            barge_in=BargeInConfig(arm_delay_ms=0),
        )
        services = SessionServices(
            responder=self.responder,  # type: ignore[arg-type]
            engines=self.engines,
            lanes=InferenceLanes(
                vad=InlineLane(),
                eot=InlineLane(),
                stt=InlineLane(),
                tts=self.tts_lane,
                kws=InlineLane(),
            ),
            settings=self.settings,
            placement=floor_placement,
            clock_factory=lambda: self.clock,
            utterance_ids=lambda: self.ids,
            ingress_factory=self._real_ingress if real_ingress else self._ingress,
            controller_factory=self._controller,
            on_capability_message=capability or (lambda _sid, _msg: None),
        )
        self.session = RealtimeSession(
            session_id=session_id,
            websocket=self.socket,
            endpoint=endpoint,
            services=services,
            host=self.host,  # type: ignore[arg-type]
            max_turn_items=max_turn_items,
        )
        self.task: asyncio.Task[None] | None = None
        RIGS.append(self)

    def _ingress(self, **kwargs: object) -> FakeIngress:
        ingress = FakeIngress(kwargs)
        self.ingresses.append(ingress)
        return ingress

    def _real_ingress(self, **kwargs: object) -> AudioIngress:
        ingress = AudioIngress(**kwargs)  # type: ignore[arg-type]
        self.real_ingresses.append(ingress)
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

    async def start(self) -> Rig:
        self.task = asyncio.get_running_loop().create_task(self.session.run())
        await settle()
        return self

    async def stop(self) -> None:
        self.socket.release()
        self.socket.disconnect()
        assert self.task is not None
        await asyncio.wait_for(self.task, timeout=5)

    async def send(self, *payloads: dict[str, object] | str) -> None:
        for payload in payloads:
            self.socket.text(payload)
        await settle()

    async def mic(self, seq: int, payload: bytes = b"\x00\x00" * 320) -> None:
        frame = Frame(MsgType.MIC_AUDIO, 0, seq, 16000, payload)
        self.socket.binary(encode_frame(frame))
        await settle(5)

    async def reference(self, seq: int) -> None:
        frame = Frame(MsgType.REF_AUDIO, 0, seq, 16000, b"\x00\x00" * 320)
        self.socket.binary(encode_frame(frame))
        await settle(5)

    def names(self) -> list[str]:
        return [name_of(item) for item in self.socket.sent]

    def answers(self) -> list[dict[str, object]]:
        return [
            item
            for item in self.socket.sent
            if isinstance(item, dict)
            and item["type"] == "voice_state"
            and item["cause"] == "answer"
        ]

    def posted(self, kind: type) -> list[object]:
        return [m for m in self.controller.posted if isinstance(m, kind)]


RIGS: list[Rig] = []


@pytest_asyncio.fixture(autouse=True)
async def _stop_rigs():
    yield
    while RIGS:
        rig = RIGS.pop()
        if rig.task is not None and not rig.task.done():
            await rig.stop()


def voice_start(req: int = 1, **overrides: object) -> dict[str, object]:
    message: dict[str, object] = {
        "type": "voice_start",
        "req": req,
        "mode": "wake",
        "sample_rate": 16000,
        "frame_ms": 20,
        "aec": "headset",
        "playback": True,
        "endpoint": "desktop",
        "spotter": "server",
    }
    message.update(overrides)
    return message


def answer_of(answer: dict[str, object]) -> tuple[object, ...]:
    return (answer["req"], answer["active"], answer["refused"], answer["muted"])


# -- Dispatch and errors ---------------------------------------------------------------------------


class TestDispatch:
    async def test_a_text_message_opens_a_turn_whose_stream_reaches_the_socket(self):
        rig = await Rig().start()

        await rig.send({"type": "text", "text": "what time is it"})

        assert [m.text for m in rig.posted(Text)] == ["what time is it"]
        assert rig.responder.requests[0].user_text == "what time is it"
        assert "stream_start" in rig.names()
        assert "stream_complete" in rig.names()

    async def test_control_messages_reach_the_mailbox_as_their_controller_messages(self):
        rig = await Rig().start()
        rig.clock.now = 4321

        await rig.send(
            voice_start(),
            {"type": "wake", "keyword": "mist", "score": 0.7, "backlog_ms": 400},
            {"type": "interrupt"},
            {"type": "voice_stop"},
        )

        wake = rig.posted(Wake)[0]
        assert (wake.keyword, wake.score, wake.backlog_ms, wake.arrival) == (
            "mist",
            0.7,
            400,
            4321,
        )
        assert len(rig.posted(Interrupt)) == 1
        assert len(rig.posted(VoiceStop)) == 1

    async def test_protocol_errors_are_sent_at_most_once_a_second(self):
        rig = await Rig().start()

        await rig.send("not json", {"type": "no_such_type"}, {"type": "text"})
        rig.clock.now = 999
        await rig.send("still not json")
        rig.clock.now = 1000
        await rig.send({"type": "text", "text": 5})

        errors = [item for item in rig.socket.sent if name_of(item) == "error"]
        assert len(errors) == 2
        assert all(e["kind"] == "protocol" for e in errors)
        assert errors[1]["context"] == {"field": "text"}

    async def test_a_bad_binary_frame_is_a_protocol_error(self):
        rig = await Rig().start()

        rig.socket.binary(b"\x01\x02")
        await settle()

        assert rig.names() == ["error"]

    async def test_log_and_capability_messages_go_to_their_owners(self):
        seen: list[tuple[str, str]] = []
        rig = await Rig(capability=lambda sid, msg: seen.append((sid, msg.type))).start()

        await rig.send(
            {"type": "subscribe_logs", "enabled": True, "levels": ["ERROR"]},
            {"type": "log_config", "action": "set_level", "level": "DEBUG", "logger": "x.y"},
            {"type": "action_decision", "proposal_id": "p1", "decision": "approve"},
        )

        assert rig.host.subscriptions == [(True, ["ERROR"])]
        assert rig.host.levels == [("x.y", "DEBUG")]
        assert seen == [("s1", "action_decision")]
        assert "log_config_ack" in rig.names()

    async def test_gpu_yield_is_logged_once_per_session_at_info(self, caplog):
        rig = await Rig().start()

        with caplog.at_level(logging.INFO, logger="backend.realtime.session"):
            await rig.send(
                {"type": "gpu_yield", "active": True, "source": "manual"},
                {"type": "gpu_yield", "active": False, "source": "manual"},
            )

        yields = [r for r in caplog.records if "gpu_yield" in r.getMessage()]
        assert len(yields) == 1
        assert yields[0].levelno == logging.INFO


# -- Lanes ---------------------------------------------------------------------------------------


class TestLanes:
    async def test_the_sender_takes_control_before_turn_before_bulk(self):
        rig = await Rig().start()
        rig.socket.hold()
        rig.session.send_bulk("bulk-1")
        await settle()  # the sender holds `bulk-1`'s write

        rig.session.send_bulk({"type": "bulk", "n": 2})
        rig.session.send_turn("t1", {"type": "stream_token", "turn_id": "t1", "token": "a"})
        rig.session.send_control({"type": "status", "message": "c"})
        rig.socket.release()
        await settle()

        assert [name_of(i) if isinstance(i, dict) else i for i in rig.socket.sent] == [
            "bulk-1",
            "status",
            "stream_token",
            "bulk",
        ]

    async def test_a_full_bulk_lane_drops_its_oldest_and_counts(self):
        rig = await Rig().start()
        rig.socket.hold()
        rig.session.send_bulk("held")
        await settle()

        for n in range(515):
            rig.session.send_bulk(f"b{n}")
        rig.socket.release()
        await settle(200)

        assert rig.session.bulk_dropped == 3
        assert rig.socket.sent[1] == "b3"
        assert rig.socket.sent[-1] == "b514"

    async def test_purge_reports_a_queued_stream_start_and_the_first_token(self):
        rig = await Rig().start()
        rig.socket.hold()
        rig.session.send_control({"type": "status", "message": "held"})
        await settle()

        start = StreamStartMessage(turn_id="t1", channel="text", origin="text")
        rig.session.send_turn("t1", start, kind="stream_start")
        for index in range(2):
            rig.session.send_turn("t1", {"type": "stream_token"}, kind="token", token_index=index)
        purge = rig.session.purge_turn("t1")

        assert purge == LanePurge(terminal=False, stream_start=True, first_token=0)

    async def test_a_full_turn_lane_closes_with_1013_and_commits_the_turn(self):
        rig = await Rig(max_turn_items=2).start()
        rig.socket.hold()
        rig.session.send_control({"type": "status", "message": "held"})
        await settle()

        await rig.send({"type": "text", "text": "a long reply"})
        await asyncio.wait_for(rig.task, timeout=5)

        assert rig.socket.closed_with == 1013
        assert rig.host.closed == ["s1"]
        assert len(rig.responder.commits) == 1  # `controller.close()` committed the turn

    async def test_a_dead_sender_closes_with_1011(self):
        rig = await Rig().start()
        rig.socket.fail_writes = True

        rig.session.send_control({"type": "status", "message": "x"})
        await asyncio.wait_for(rig.task, timeout=5)

        assert rig.socket.closed_with == 1011
        assert rig.host.closed == ["s1"]

    async def test_a_disconnect_unregisters_and_closes_the_controller(self):
        rig = await Rig().start()

        await rig.stop()

        assert rig.host.closed == ["s1"]
        assert rig.controller.task is not None and rig.controller.task.done()
        assert rig.socket.closed_with is None

    async def test_two_sessions_are_isolated(self):
        a = await Rig(session_id="a").start()
        b = await Rig(session_id="b").start()

        await a.send({"type": "text", "text": "hello"})

        assert "stream_start" in a.names()
        assert b.socket.sent == []
        assert b.responder.requests == []


# -- Voice start and stop ------------------------------------------------------------------------


class TestVoiceStart:
    async def test_from_off_a_valid_voice_start_builds_ingress_answers_and_posts(self):
        rig = await Rig().start()

        await rig.send(voice_start(muted=True))

        ingress = rig.ingress
        assert ingress.calls[0] == ("set_muted", True)
        assert ingress.kwargs["epoch"] == 1
        assert ingress.kwargs["retention_ms"] == 7000
        assert ingress.kwargs["new_utterance_id"] is rig.controller.kwargs["new_utterance_id"]
        assert ingress.kwargs["seed"] is None
        assert [answer_of(a) for a in rig.answers()] == [(1, True, None, True)]
        assert rig.answers()[0]["mode"] == "wake"
        start = rig.posted(VoiceStart)[0]
        assert start.ingress is ingress and start.epoch == 1
        assert [m.active for m in rig.posted(Mute)] == [True]

    async def test_an_invalid_voice_start_is_answered_invalid_and_posts_nothing(self):
        rig = await Rig().start()

        await rig.send(voice_start(req=4, mode="ptt"), voice_start(req="x"))

        assert [answer_of(a) for a in rig.answers()] == [
            (4, False, "invalid", False),
            (None, False, "invalid", False),
        ]
        assert rig.ingresses == []
        assert rig.posted(VoiceStart) == []

    async def test_a_req_not_above_the_highest_is_invalid(self):
        rig = await Rig().start()

        await rig.send(voice_start(req=5), voice_start(req=5))

        assert [answer_of(a)[:3] for a in rig.answers()] == [
            (5, True, None),
            (5, True, "invalid"),
        ]

    async def test_reference_with_the_endpoint_spotter_is_invalid(self):
        rig = await Rig().start()

        await rig.send(voice_start(spotter="endpoint", reference=True))

        assert answer_of(rig.answers()[0]) == (1, False, "invalid", False)

    @pytest.mark.parametrize(
        ("rig_args", "reason"),
        [({"spotter": None}, "kws unavailable"), ({"vad": False}, "vad unavailable")],
    )
    async def test_an_unavailable_engine_refuses_the_mode(self, rig_args, reason):
        rig = await Rig(**rig_args).start()

        await rig.send(voice_start())

        assert answer_of(rig.answers()[0]) == (1, False, reason, False)
        assert rig.ingresses == []

    async def test_an_endpoint_spotter_needs_no_server_spotter(self):
        rig = await Rig(spotter=None).start()

        await rig.send(voice_start(spotter="endpoint"))

        assert answer_of(rig.answers()[0]) == (1, True, None, False)

    async def test_a_differing_endpoint_is_logged_and_ignored(self, caplog):
        rig = await Rig(endpoint="room").start()

        with caplog.at_level(logging.WARNING, logger="backend.realtime.session"):
            await rig.send(voice_start(endpoint="desktop"))

        assert answer_of(rig.answers()[0]) == (1, True, None, False)
        assert any("endpoint" in r.getMessage() for r in caplog.records)
        # A room session asks the spotter for its endpoint calibration (v5).
        assert rig.controller.kwargs["calibration"] is CALIBRATION["endpoint"]

    async def test_the_controller_is_wired_from_the_settings_and_the_spotter(self):
        rig = await Rig().start()

        kwargs = rig.controller.kwargs
        assert kwargs["calibration"] is CALIBRATION["server"]
        assert kwargs["hit_lag_ms"] == 180
        assert kwargs["arbiter"] is rig.host.arbiter
        assert isinstance(kwargs["gate"], AddresseeGate)
        assert kwargs["gate"].echo_guard_ms == rig.settings.barge_in.echo_guard_ms
        assert kwargs["playback_config"] is rig.settings.playback
        assert kwargs["segmenter_config"] is rig.settings.segmenter
        assert kwargs["turn_config"] is rig.settings.turn
        assert kwargs["wake"] is rig.settings.wake
        assert kwargs["tts_lane"] is rig.tts_lane
        assert kwargs["synthesizer"] is rig.engines.synthesizer

    async def test_with_no_spotter_or_synthesizer_the_controller_gets_none_and_zero(self):
        rig = await Rig(spotter=None, synthesizer=False).start()

        kwargs = rig.controller.kwargs
        assert kwargs["calibration"] is None
        assert kwargs["hit_lag_ms"] == 0
        assert kwargs["synthesizer"] is None
        assert kwargs["tts_lane"] is None

    async def test_voice_stop_closes_ingress_and_stops_routing(self):
        rig = await Rig().start()
        await rig.send(voice_start())
        ingress = rig.ingress

        await rig.send({"type": "voice_stop"})
        await rig.mic(seq=0)

        assert ("close",) in ingress.calls
        assert not any(call[0] == "frame" for call in ingress.calls)
        assert rig.session.ignored_frames == 1


# -- Uplink ----------------------------------------------------------------------------------------


class TestUplink:
    async def test_frames_before_the_first_voice_start_are_counted_and_discarded(self):
        rig = await Rig().start()

        await rig.mic(seq=0)
        await rig.mic(seq=1)

        assert rig.session.ignored_frames == 2

    async def test_frame_positions_keep_counting_across_stop_start_and_reconfigure(self):
        rig = await Rig().start()
        await rig.mic(seq=0)  # position 0, discarded
        await rig.send(voice_start(req=1))
        await rig.mic(seq=1)  # position 1
        await rig.send({"type": "voice_stop"})
        await rig.mic(seq=2)  # position 2, ignored
        await rig.send(voice_start(req=2))
        await rig.mic(seq=3)  # position 3
        await rig.send(voice_start(req=3, aec="native"))
        await rig.mic(seq=0)  # position 4: the client restarted seq after the answer

        frames = [
            (i.kwargs["epoch"], call[1], call[2])
            for i in rig.ingresses
            for call in i.calls
            if call[0] == "frame"
        ]
        assert frames == [(1, 1, 1), (2, 3, 3), (3, 0, 4)]

    async def test_reference_frames_reach_ingress_only_after_a_reference_voice_start(self):
        rig = await Rig().start()
        await rig.send(voice_start(req=1))
        await rig.reference(seq=1)
        await rig.send(voice_start(req=2, reference=True))
        await rig.reference(seq=2)

        assert ("reference", 1) not in rig.ingresses[0].calls
        assert ("reference", 2) in rig.ingresses[1].calls
        assert rig.session.dropped_references == 1


# -- Reconfigure and answers ---------------------------------------------------------------------


class TestReconfigure:
    async def test_a_quiescent_reconfigure_hands_off_to_a_seeded_ingress(self):
        rig = await Rig().start()
        await rig.send(voice_start(req=1, muted=True))
        first = rig.ingress

        await rig.send(voice_start(req=2, aec="native", muted=True))

        second = rig.ingress
        assert second is not first
        assert ("handoff",) in first.calls and ("close",) in first.calls
        assert second.kwargs["seed"] == IngressSeed(frames=(), floor=7, vad_from=None)
        assert second.kwargs["epoch"] == 2
        assert second.calls[0] == ("set_muted", True)
        assert [answer_of(a) for a in rig.answers()] == [
            (1, True, None, True),
            (2, True, None, True),
        ]
        assert isinstance(rig.posted(Reconfigure)[0], Reconfigure)

    async def test_an_early_answer_waits_for_a_pending_reconfigure(self):
        rig = await Rig().start()
        await rig.send(voice_start(req=1))
        rig.ingress.holding = True  # an utterance is open: no quiescent point

        await rig.send(voice_start(req=2, aec="native"), voice_start(req="bad"))
        assert [a["req"] for a in rig.answers()] == [1]

        rig.ingress.holding = False
        await rig.send({"type": "interrupt"})  # any mailbox step reaches the quiescent check

        assert [answer_of(a)[:3] for a in rig.answers()] == [
            (1, True, None),
            (2, True, None),
            (None, True, "invalid"),
        ]

    async def test_a_newer_request_supersedes_a_pending_one(self):
        rig = await Rig().start()
        await rig.send(voice_start(req=1))
        rig.ingress.holding = True

        await rig.send(voice_start(req=2, aec="native"), voice_start(req=3, aec="device"))
        rig.ingress.holding = False
        await rig.send({"type": "interrupt"})

        assert [(a["req"], a["refused"]) for a in rig.answers()] == [
            (1, None),
            (2, "superseded"),
            (3, None),
        ]
        assert len(rig.ingresses) == 2

    async def test_voice_stop_answers_a_pending_reconfigure_voice_stop(self):
        rig = await Rig().start()
        await rig.send(voice_start(req=1))
        rig.ingress.holding = True

        await rig.send(voice_start(req=2, aec="native"), {"type": "voice_stop"})

        assert [answer_of(a)[:3] for a in rig.answers()] == [
            (1, True, None),
            (2, False, "voice_stop"),
        ]

    async def test_a_refused_reconfigure_keeps_the_old_ingress(self):
        rig = await Rig(spotter=None).start()
        await rig.send(voice_start(req=1, spotter="endpoint"))

        await rig.send(voice_start(req=2, spotter="server"))

        assert [(a["req"], a["refused"]) for a in rig.answers()] == [
            (1, None),
            (2, "kws unavailable"),
        ]
        assert len(rig.ingresses) == 1

    async def test_a_disconnect_with_a_pending_reconfigure_sends_nothing(self):
        rig = await Rig().start()
        await rig.send(voice_start(req=1))
        rig.ingress.holding = True
        await rig.send(voice_start(req=2, aec="native"))

        await rig.stop()

        assert [a["req"] for a in rig.answers()] == [1]


# -- Mute ------------------------------------------------------------------------------------------


class TestMute:
    async def test_mute_tells_ingress_posts_mute_and_sends_this_session_a_broadcast(self):
        rig = await Rig().start()
        await rig.send(voice_start())

        rig.socket.text({"type": "mute", "active": True})
        rig.socket.binary(encode_frame(Frame(MsgType.MIC_AUDIO, 0, 9, 16000, b"\x00\x00" * 320)))
        await settle()

        calls = [c for c in rig.ingress.calls if c[0] in ("set_muted", "frame")]
        assert calls == [("set_muted", True), ("frame", 9, 0)]
        assert [m.active for m in rig.posted(Mute)] == [True]
        state = rig.socket.sent[-1]
        assert (state["type"], state["cause"], state["req"], state["muted"]) == (
            "voice_state",
            "broadcast",
            None,
            True,
        )

    async def test_an_open_utterance_drops_into_the_mailbox_before_mute(self):
        rig = await Rig().start()
        await rig.send(voice_start())
        rig.ingress.open_utterance_id = rig.ids()

        await rig.send({"type": "mute", "active": True})

        kinds = [type(m).__name__ for m in rig.controller.posted]
        assert kinds[-2:] == ["UtteranceDropped", "Mute"]

    async def test_a_deferred_unmute_reconfigure_then_mute_ends_muted(self):
        rig = await Rig().start()
        await rig.send(voice_start(req=1, muted=True))
        rig.ingress.holding = True

        await rig.send(voice_start(req=2, aec="native", muted=False))
        await rig.send({"type": "mute", "active": True})
        rig.ingress.holding = False
        await rig.send({"type": "interrupt"})

        assert rig.ingress.kwargs["epoch"] == 2
        assert rig.ingress.calls[0] == ("set_muted", True)
        assert rig.answers()[-1]["muted"] is True

    async def test_a_muted_wake_and_an_unmuted_utterance_get_different_ids(self):
        rig = await Rig().start()
        await rig.send(voice_start(muted=True))
        await rig.send({"type": "wake", "keyword": "mist", "score": 0.9, "backlog_ms": 0})
        await rig.send({"type": "mute", "active": False})
        uid = rig.ids()

        dropped = [i for i in rig.socket.sent if name_of(i) == "voice_decision:dropped"]
        assert dropped[0]["detail"]["utterance_id"] != uid
        assert dropped[0]["reason"] == "muted"


# -- Turns through the lanes ---------------------------------------------------------------------


async def spoken_turn(rig: Rig) -> None:
    """Start voice and publish one utterance; the turn it opens speaks its reply."""
    await rig.send(voice_start())
    uid = rig.ids()
    rig.ingress.emit(rig.ingress.started(uid))
    rig.ingress.emit(rig.ingress.ended(uid, "hello"))


class TestTurnLane:
    async def test_an_interrupt_while_a_write_is_held_purges_the_reply(self):
        rig = await Rig([Token("Hello there. How are you?")]).start()
        rig.socket.hold(lambda item: name_of(item) == "stream_start")
        await spoken_turn(rig)
        await settle()
        assert rig.socket.holding is not None

        await rig.send({"type": "interrupt"})
        rig.socket.release()
        await settle()

        names = rig.names()
        tail = names[names.index("stream_start") :]
        assert "0x02" not in tail
        assert tail.index("0x03") < tail.index("playback_stop") < tail.index("stream_cancelled")

    async def test_an_interrupt_after_the_sender_took_the_end_marker_sends_only_stop(self):
        rig = await Rig().start()
        rig.socket.hold(lambda item: name_of(item) == "0x02")
        await spoken_turn(rig)
        await settle()
        assert rig.socket.holding is not None

        await rig.send({"type": "interrupt"})
        rig.socket.release()
        await settle()

        names = rig.names()
        tail = names[names.index("0x02") :]
        assert "0x03" not in tail
        assert "playback_stop" in tail

    async def test_a_voice_turn_with_no_synthesizer_is_degraded_tts_unavailable(self, caplog):
        rig = await Rig(synthesizer=False).start()

        with caplog.at_level(logging.DEBUG):
            await spoken_turn(rig)
            await settle()

        names = rig.names()
        assert names.index("stream_start") + 1 == names.index("voice_decision:degraded")
        degraded = next(i for i in rig.socket.sent if name_of(i) == "voice_decision:degraded")
        assert degraded["reason"] == "tts_unavailable"
        assert "hello" not in caplog.text

    async def test_a_typed_turn_is_not_degraded(self):
        rig = await Rig(synthesizer=False).start()
        await rig.send(voice_start())

        await rig.send({"type": "text", "text": "hi"})

        assert "voice_decision:degraded" not in rig.names()

    async def test_the_tts_text_a_session_played_is_readable_for_the_echo_guard(self):
        rig = await Rig().start()
        await spoken_turn(rig)
        await settle()
        rig.clock.now += 400

        assert "Hello" in rig.session.recent_tts_text(5000)

    async def test_voice_activity_follows_utterances_and_voice_turns(self):
        pause_rig = await Rig().start()
        await pause_rig.send(voice_start())
        assert pause_rig.session.voice_activity is False

        pause_rig.ingress.holding = True
        pause_rig.ingress.emit(pause_rig.ingress.started(pause_rig.ids()))
        await settle()

        assert pause_rig.session.voice_activity is True
        assert pause_rig.host.activity[-1] is True


# -- The real ingress ------------------------------------------------------------------------------


class TestRealIngress:
    async def test_idle_uplink_for_600_seconds_produces_nothing(self):
        rig = await Rig(real_ingress=True).start()
        await rig.send(voice_start())
        before = list(rig.socket.sent)

        for seq in range(30_000):
            rig.socket.binary(
                encode_frame(Frame(MsgType.MIC_AUDIO, 0, seq, 16000, b"\x00\x00" * 320))
            )
            rig.clock.now += 20
            await settle(3)
        await settle()

        assert rig.socket.sent == before
        assert [m for m in rig.controller.posted if type(m).__module__.endswith("ingress")] == []
        assert rig.real_ingresses[0].stats().frames_received == 30_000

    async def test_a_seq_restart_after_an_apply_counts_no_gap(self):
        rig = await Rig(real_ingress=True).start()
        await rig.send(voice_start(req=1))
        for seq in range(5):
            await rig.mic(seq)

        await rig.send(voice_start(req=2, aec="native"))
        for seq in range(3):
            await rig.mic(seq)

        assert len(rig.real_ingresses) == 2
        assert rig.real_ingresses[1].stats().gap_frames == 0
        assert rig.real_ingresses[1].stats().frames_received == 3
