"""The fake device scenario: a room device through the real device app, end to end.

The device authenticates on the real `DeviceApp`, whose serve is the server's own
`_serve_session`, so its connection becomes a `RealtimeSession` bound to the device's name with
kind `room`. The runtime is `install_runtime`'s: the real session and controller over fake
engines on inline lanes. Its ingress is the recording fake, so the test emits the events the
real ingress emits for an endpoint wake once the backlog has arrived; that behaviour is pinned
by `tests/unit/realtime/test_ingress.py::TestEndpointWake`. The capability host is real, and
its alarm control writes a real `AlarmRecords` store, so `delivered_to` is what the store holds.

`alarm_fired` is queued with `SessionRegistry.broadcast(..., endpoint=<name>)`, which is how a
ring for one endpoint reaches it; the ring sinks that call it are not part of this task.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Literal

import pytest
import pytest_asyncio

from backend.capabilities.action_log import ActionLog
from backend.capabilities.alarms.notifier import alarm_fired_payload
from backend.capabilities.alarms.records import AlarmRecords, ArmRequest, ReportOutcome
from backend.capabilities.contracts import ToolOutcome
from backend.capabilities.db import CapabilitiesDb
from backend.capabilities.decoder import ActionDecoder
from backend.capabilities.host import CapabilityHost, HostSettings
from backend.capabilities.policy import PolicyTable
from backend.capabilities.router import RouterConfig, ToolRouter
from backend.capabilities.secrets import SecretRegistry
from backend.chat.context_budget import ApproximateTokenCounter
from backend.devices.limiter import AuthLimiter
from backend.devices.listener import build_device_app
from backend.devices.registry import DeviceRegistry
from backend.factories import CapabilityBuild
from backend.realtime.protocol import Direction, Frame, MsgType, decode_frame, encode_frame
from tests.unit.capabilities.fakes import FakeCapability, RecordingEventSink
from tests.unit.capabilities.fakes import FakeClock as HostClock
from tests.unit.devices.test_listener import (
    ROOM,
    AsgiDevice,
    FakeClock,
    entry,
    secret_for,
    write_registry,
)
from tests.unit.realtime.turn_fakes import settle
from tests.unit.test_session_id_propagation import (  # noqa: F401 -- autouse fixtures
    RecordingConversationHandler,
    install_runtime,
    no_runtime,
    reset_session_context,
)

pytestmark = pytest.mark.asyncio

DEVICE = "example-room"
BACKLOG_MS = 2000
FRAME_MS = 20
RING_TIMEOUT_S = 600.0


class DeviceClient(AsgiDevice):
    """The device's side: JSON and uplink audio in, the session's frames out."""

    def send(self, payload: dict[str, object]) -> None:
        self.text(json.dumps(payload))

    def mic(self, seq: int) -> None:
        frame = Frame(MsgType.MIC_AUDIO, 0, seq, 16000, b"\x00\x00" * 320)
        self.inbox.put_nowait({"type": "websocket.receive", "bytes": encode_frame(frame)})

    def messages(self) -> list[dict[str, object]]:
        """Every JSON message the session sent, in order."""
        return [json.loads(text) for text in self.texts()]

    def of_type(self, kind: str) -> list[dict[str, object]]:
        return [m for m in self.messages() if m.get("type") == kind]

    def audio(self) -> list[Frame]:
        """Every downlink binary frame, decoded."""
        return [
            decode_frame(bytes(m["bytes"]), direction=Direction.DOWN)  # type: ignore[arg-type]
            for m in self.sent
            if m["type"] == "websocket.send" and m.get("bytes") is not None
        ]


class RecordsAlarmControl:
    """The host's alarm control over a real store, without the scheduler that owns firing.

    `report` passes the report to `AlarmRecords.report` as the scheduler does, with the grace
    window at its 60 s floor; `act` is not reached by this scenario.
    """

    def __init__(self, records: AlarmRecords) -> None:
        self.records = records

    async def snapshot(self) -> dict[str, object]:
        raise AssertionError("the scenario sends no alarm_state request")

    async def act(
        self,
        alarm_id: str,
        action: Literal["stop", "snooze", "dismiss"],
        minutes: int | None,
        *,
        endpoint: str,
    ) -> ToolOutcome:
        raise AssertionError("the scenario sends no alarm_action")

    async def report(
        self,
        alarm_id: str,
        *,
        endpoint: str,
        outcome: ReportOutcome,
        at: datetime,
        rang_at: datetime,
        snoozed_until: datetime | None,
    ):
        return await self.records.report(
            alarm_id,
            endpoint=endpoint,
            outcome=outcome,
            at=at,
            rang_at=rang_at,
            snoozed_until=snoozed_until,
            now=datetime.now(UTC),
            next_fire_at=None,
            grace=timedelta(seconds=60),
        )


def build_host(records: AlarmRecords) -> CapabilityHost:
    alarms = FakeCapability("alarms")
    counter = ApproximateTokenCounter()
    return CapabilityHost(
        [alarms],
        router=ToolRouter(
            [alarms], RouterConfig(token_budget=1200, reserved_tokens=100), counter, None
        ),
        counter=counter,
        policy=PolicyTable.from_specs(list(alarms.tools()), None),
        decoder=ActionDecoder(),
        secrets=SecretRegistry(),
        action_log=ActionLog(CapabilitiesDb(":memory:")),
        event_sink=RecordingEventSink(),
        # The real clock's time: the report's window is checked against it.
        clock=HostClock(datetime.now(UTC), settle_timeout_s=0.3),
        settings=HostSettings(),
        alarm_control=RecordsAlarmControl(records),
    )


@pytest_asyncio.fixture
async def store(tmp_path):
    db = CapabilitiesDb(str(tmp_path / "capabilities.db"))
    records = AlarmRecords(db)
    records.migrate()
    yield records
    db.close()


@pytest_asyncio.fixture
async def host(monkeypatch, store):
    """A real host installed as `lifespan` leaves it, and a fresh broadcast queue."""
    from backend import server

    built = build_host(store)
    monkeypatch.setattr(server, "_capability_build", CapabilityBuild(built, None))
    monkeypatch.setattr(server, "message_queue", asyncio.Queue())
    yield built
    await built.stop()


@pytest.fixture
def device_app(tmp_path: Path):
    """The real device app over a registry naming the device, serving `_serve_session`."""
    from backend import server

    path = tmp_path / "registry.json"
    write_registry(path, [entry(DEVICE, secret_for(1))])
    clock = FakeClock()
    return build_device_app(
        DeviceRegistry.load(path), server._serve_session, AuthLimiter(clock), clock
    )


def calls(ingress: object, name: str) -> list[tuple[object, ...]]:
    return [call for call in ingress.calls if call[0] == name]  # type: ignore[attr-defined]


async def ringing_occurrence(records: AlarmRecords) -> str:
    """Arm an alarm due a moment ago and fire it, as the scheduler would: it is ringing."""
    now = datetime.now(UTC)
    due = now - timedelta(seconds=10)
    armed = await records.arm(
        ArmRequest(
            kind="alarm",
            fire_at=due,
            wall_time=time(due.hour, due.minute),
            duration_s=None,
            label=None,
            days=None,
            ring_on=frozenset({DEVICE}),
        ),
        at=now - timedelta(minutes=5),
        tz=UTC,
        source="model",
    )
    occurrence_id = armed.occurrence.occurrence_id
    assert await records.fire(
        occurrence_id, outcome="ringing", at=due, late_by_s=0.0, next_fire_at=None
    )
    return occurrence_id


@pytest.fixture
def rig(monkeypatch):
    return install_runtime(monkeypatch, RecordingConversationHandler())


@pytest_asyncio.fixture
async def device(rig, host, device_app):
    """The device, connected with its token; disconnected after the test if it is still on."""
    client = DeviceClient(device_app, f"Bearer {ROOM}")
    await settle()
    yield client
    if not client.task.done():
        client.leave()
        await client.ended()


async def test_a_device_wakes_with_a_backlog_hears_its_reply_and_reports_its_alarm(
    rig, store, device
):
    from backend import server

    assert device.accepted
    (started,) = device.of_type("session_started")
    session_id = str(started["session_id"])
    assert rig.registry.endpoint_of(session_id) == DEVICE

    # -- The wake: the device's spotter heard the name, and the backlog follows it.
    device.send(
        {
            "type": "voice_start",
            "req": 1,
            "mode": "wake",
            "sample_rate": 16000,
            "frame_ms": FRAME_MS,
            "aec": "native",
            "playback": True,
            "endpoint": "room",
            "spotter": "endpoint",
        }
    )
    await settle()
    (answer,) = (m for m in device.of_type("voice_state") if m["cause"] == "answer")
    assert (answer["req"], answer["active"], answer["refused"]) == (1, True, None)
    device.send({"type": "wake", "keyword": "mist", "score": 0.9, "backlog_ms": BACKLOG_MS})
    backlog = BACKLOG_MS // FRAME_MS
    for seq in range(backlog + 10):
        device.mic(seq)
    await settle(200)

    (ingress,) = rig.ingresses
    (expect,) = calls(ingress, "expect_endpoint_wake")
    assert expect[2]["backlog_ms"] == BACKLOG_MS
    assert len(calls(ingress, "frame")) == backlog + 10

    # What the ingress emits once the backlog's last frame is scored, then the utterance.
    ingress.emit(
        ingress.heard(None, from_pos=60, hit_pos=backlog - 1, spotter="endpoint", score=0.9)
    )
    await settle()
    ((_, mode, _from, uid),) = calls(ingress, "open_utterance")
    assert mode == "wake"
    ingress.emit(ingress.partial(uid, "mist what time is it", gate=True))
    ingress.emit(ingress.ended(uid, "Mist, what time is it?"))
    await settle(200)

    # -- The turn: its audio is tagged with the turn's tag and ends with the end marker.
    (stream_start,) = device.of_type("stream_start")
    assert stream_start["channel"] == "tts"
    tag = stream_start["audio_tag"]
    audio = device.audio()
    kinds = [frame.msg_type for frame in audio]
    assert MsgType.AUDIO_CHUNK in kinds and kinds[-1] == MsgType.AUDIO_COMPLETE
    assert {frame.turn_tag for frame in audio} == {tag}
    assert rig.handler.requests[0].session_id == session_id
    assert rig.handler.commits == []

    device.send({"type": "playback_done", "turn_id": stream_start["turn_id"], "played_ms": 900})
    await settle(200)
    assert [delivery.outcome for delivery in rig.handler.commits] == ["completed"]

    # -- The alarm: rung on this device, reported rang, delivered to this device.
    occurrence_id = await ringing_occurrence(store)
    (occurrence,) = (o for o in (await store.snapshot())[1] if o.occurrence_id == occurrence_id)
    rig.registry.broadcast(
        alarm_fired_payload(occurrence, RING_TIMEOUT_S, fallback=False), endpoint=DEVICE
    )
    await settle()
    (fired,) = device.of_type("alarm_fired")
    assert fired["alarm_id"] == occurrence_id

    rang = (datetime.now(UTC) - timedelta(seconds=2)).isoformat()
    device.send(
        {
            "type": "alarm_report",
            "alarm_id": occurrence_id,
            "outcome": "rang",
            "at": rang,
            "rang_at": rang,
            "request_id": "rep-1",
        }
    )
    for _ in range(50):
        await settle()
        if device.of_type("alarm_action_result"):
            break
        await asyncio.sleep(0.01)
    (result,) = device.of_type("alarm_action_result")
    assert (result["action"], result["outcome"], result["detail"]) == ("report", "ok", "applied")
    (after,) = (o for o in (await store.snapshot())[1] if o.occurrence_id == occurrence_id)
    assert after.delivered_to == frozenset({DEVICE})

    device.leave()
    await device.ended()
    assert server.session_registry.endpoint_of(session_id) is None
