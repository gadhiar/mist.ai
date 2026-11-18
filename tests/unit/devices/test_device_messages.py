"""A device token's session takes the device's message types only, end to end.

The device authenticates on the real `DeviceApp`, whose serve is the server's own
`_serve_session`, so its connection is a `RealtimeSession` bound to the device's name with kind
`room`, as `test_device_turn.py` builds it. The runtime is `install_runtime`'s, with a recording
log gate in place of the streaming log handler; the capability host is a real `CapabilityHost`
over a scripted alarm control, which records the endpoint each alarm action names.

What these pin: a type the device never sends is refused with a protocol error and changes
nothing, not the turns, the log stream's gate or a proposal, and the session goes on to take the
device's own types.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest
import pytest_asyncio

from backend.devices.limiter import AuthLimiter
from backend.devices.listener import build_device_app
from backend.devices.registry import DeviceRegistry
from backend.factories import CapabilityBuild
from tests.unit.devices.test_device_turn import DeviceClient
from tests.unit.devices.test_listener import ROOM, FakeClock, entry, secret_for, write_registry
from tests.unit.realtime.test_registry import RecordingLogGate
from tests.unit.realtime.turn_fakes import settle
from tests.unit.test_server_capability_messages import DecidingHandler
from tests.unit.test_server_capability_messages import Rig as CapabilityRig
from tests.unit.test_session_id_propagation import (  # noqa: F401 -- autouse fixtures
    drain_capability_calls,
    install_runtime,
    no_runtime,
    reset_session_context,
)

pytestmark = pytest.mark.asyncio

DEVICE = "example-room"
SESSION_LOGGER = "backend.realtime.session"

REFUSED = {
    "text": {"type": "text", "text": "read me the logs"},
    "subscribe_logs": {"type": "subscribe_logs", "enabled": True, "levels": None},
    "log_config": {"type": "log_config", "action": "set_level", "level": "DEBUG", "logger": "x"},
    "action_decision": {"type": "action_decision", "proposal_id": "p1", "decision": "approve"},
}


def refusal(message_type: str) -> dict[str, object]:
    return {
        "type": "error",
        "kind": "protocol",
        "message": f"{message_type} is not accepted from a room endpoint",
        "retriable": False,
        "context": {"field": "type"},
    }


@pytest.fixture
def gate() -> RecordingLogGate:
    return RecordingLogGate()


@pytest.fixture
def rig(monkeypatch, gate):
    return install_runtime(monkeypatch, DecidingHandler(), log_gate=gate)


@pytest_asyncio.fixture
async def host(monkeypatch):
    """A real host installed as `lifespan` leaves it, and a fresh broadcast queue."""
    from backend import server

    built = CapabilityRig()
    monkeypatch.setattr(server, "_capability_build", CapabilityBuild(built.host, None))
    monkeypatch.setattr(server, "message_queue", asyncio.Queue())
    yield built
    await built.host.stop()


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


@pytest_asyncio.fixture
async def device(rig, host, device_app):
    """The device, connected with its token; disconnected after the test if it is still on."""
    client = DeviceClient(device_app, f"Bearer {ROOM}")
    await settle()
    yield client
    if not client.task.done():
        client.leave()
        await client.ended()


async def deliver(device: DeviceClient, message: dict[str, object]) -> None:
    """Send one message, and let the session and any capability call it started finish."""
    device.send(message)
    await settle()
    await drain_capability_calls()


def stop_alarm(request_id: str) -> dict[str, object]:
    return {"type": "alarm_action", "alarm_id": "a-ringing", "action": "stop", "request_id": request_id}


@pytest.mark.parametrize("message_type", sorted(REFUSED))
async def test_a_device_token_cannot_send_a_type_the_device_does_not(
    rig, gate, host, device, caplog, message_type
):
    from backend import server

    assert device.accepted
    (started,) = device.of_type("session_started")
    session_id = str(started["session_id"])
    assert rig.registry.endpoint_of(session_id) == DEVICE
    sent = len(device.messages())

    with caplog.at_level(logging.WARNING, logger=SESSION_LOGGER):
        await deliver(device, REFUSED[message_type])

    assert device.messages()[sent:] == [refusal(message_type)]
    # Nothing changed: no turn, no log stream, no gate level, no proposal decided.
    assert rig.handler.requests == []
    assert gate.streaming == [] and gate.levels == []
    assert rig.handler.decisions == []
    assert server.message_queue.empty()
    (record,) = (r for r in caplog.records if r.getMessage().startswith("room_message_refused"))
    assert record.getMessage() == (
        f"room_message_refused session={session_id} endpoint={DEVICE} type={message_type}"
    )

    # The session is still open, and takes the device's own types.
    await deliver(device, stop_alarm("after-refusal"))
    (result,) = device.of_type("alarm_action_result")
    assert (result["request_id"], result["outcome"]) == ("after-refusal", "ok")


async def test_a_device_alarm_action_reaches_the_host_with_the_devices_name(host, device):
    await deliver(device, stop_alarm("req-1"))

    assert host.control.acts == [("a-ringing", "stop", None, DEVICE)]
    (result,) = device.of_type("alarm_action_result")
    assert result["outcome"] == "ok"
    assert device.of_type("error") == []
