"""Pin the WebSocket protocol version the backend announces, and `/ws` serving it (T13).

`backend/server.py` sends `PROTOCOL_VERSION` on `session_started`. Protocol 2.0.0 is the
realtime protocol (`backend/realtime/protocol.py`): main's 1.4.0 control messages plus the
voice path's own, with `audio` and `reset_vad` removed. A bump that is not also made in the
protocol module, or the reverse, should fail here rather than drift silently.

The endpoint tests drive the real `websocket_endpoint` with one `RealtimeSession` per
connection through a real `SessionRegistry`, over the fakes of
`tests/unit/test_session_id_propagation.py`.
"""

from __future__ import annotations

import re

import pytest

from backend import server
from backend.chat.stream_events import Token
from backend.realtime.runtime import voice_capabilities
from tests.unit.realtime.test_registry import RecordingLogGate
from tests.unit.realtime.turn_fakes import Pause, settle
from tests.unit.test_session_id_propagation import (  # noqa: F401 -- autouse fixtures
    FakeWebSocket,
    RecordingConversationHandler,
    connect,
    install_runtime,
    no_runtime,
    reset_session_context,
    run_endpoint,
)

EXPECTED_PROTOCOL_VERSION = "2.0.0"


def test_protocol_version_is_the_realtime_protocols() -> None:
    assert server.PROTOCOL_VERSION == EXPECTED_PROTOCOL_VERSION


def test_protocol_version_is_semver() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", server.PROTOCOL_VERSION)


def _names(ws: FakeWebSocket) -> list[str]:
    return [str(frame["type"]) for frame in ws.frames]


@pytest.mark.asyncio
async def test_session_started_carries_protocol_2_and_the_voice_capabilities(monkeypatch):
    rig = install_runtime(monkeypatch)

    ws = await run_endpoint(monkeypatch, None, [])

    started = ws.frames[0]
    assert started["type"] == "session_started"
    assert started["protocol_version"] == "2.0.0"
    assert started["mist_state"] == "idle"
    capabilities = started["capabilities"]
    assert isinstance(capabilities, dict)
    assert capabilities["voice"] == voice_capabilities(rig.engines).model_dump(mode="json")
    assert ws.accepted


@pytest.mark.asyncio
async def test_an_interrupt_mid_generation_is_handled_before_the_llm_finishes(monkeypatch):
    """The receive loop never awaits a turn, so `interrupt` cancels a stream still generating."""
    pause = Pause()
    rig = install_runtime(monkeypatch, RecordingConversationHandler([Token("One. "), pause]))
    connection = await connect()

    await connection.send({"type": "text", "text": "tell me a story"})
    assert pause.reached.is_set()
    await connection.send({"type": "interrupt"})

    assert "stream_cancelled" in _names(connection.ws)
    assert not pause.event.is_set(), "the fake LLM was still mid-generation"
    await connection.leave()
    assert [c.outcome for c in rig.handler.commits] == ["interrupted"]


@pytest.mark.asyncio
async def test_each_control_message_main_handled_reaches_its_handler(monkeypatch):
    """`text`, `interrupt`, `log_config`, `subscribe_logs` and the three capability messages."""
    gate = RecordingLogGate()
    rig = install_runtime(monkeypatch, log_gate=gate)
    capability_calls: list[tuple[object, str]] = []

    async def record(data, session_id, send):
        capability_calls.append((data["type"], session_id))

    monkeypatch.setattr(server, "_handle_capability_message", record)
    at = "2026-01-05T07:00:00Z"
    connection = await connect()

    await connection.send(
        {"type": "text", "text": "hi"},
        {"type": "interrupt"},
        {"type": "log_config", "action": "set_level", "level": "DEBUG", "logger": "x.y"},
        {"type": "subscribe_logs", "enabled": True, "levels": ["ERROR"]},
        {"type": "action_decision", "proposal_id": "p1", "decision": "approve"},
        {"type": "alarm_action", "alarm_id": "a1", "action": "stop"},
        {"type": "alarm_report", "alarm_id": "a1", "outcome": "rang", "at": at, "rang_at": at},
    )
    session_id = connection.ws.handshake_session_id
    await connection.leave()

    assert [r.user_text for r in rig.handler.requests] == ["hi"]
    assert "log_config_ack" in _names(connection.ws)
    assert gate.levels == [("x.y", "DEBUG")]
    assert gate.streaming[0] == (True, ["ERROR"])
    assert capability_calls == [
        ("action_decision", session_id),
        ("alarm_action", session_id),
        ("alarm_report", session_id),
    ]
    assert "error" not in _names(connection.ws)


@pytest.mark.asyncio
async def test_protocol_1_audio_and_reset_vad_are_unknown_types(monkeypatch):
    """Protocol 2 removed them (T1): each is a `protocol` error, and the session goes on."""
    install_runtime(monkeypatch)
    connection = await connect()

    await connection.send({"type": "reset_vad"}, {"type": "audio", "audio": [0.0]})
    await connection.send(
        {"type": "log_config", "action": "set_level", "level": "INFO", "logger": "x.y"}
    )
    await connection.leave()

    errors = [f for f in connection.ws.frames if f["type"] == "error"]
    # Protocol errors are sent at most once a second, and the fake clock stands still.
    assert [e["kind"] for e in errors] == ["protocol"]
    assert "reset_vad" in str(errors[0]["message"])
    assert "log_config_ack" in _names(connection.ws)


@pytest.mark.asyncio
async def test_disconnect_commits_the_turn_before_the_handler_ends_the_session(monkeypatch):
    rig = install_runtime(monkeypatch, RecordingConversationHandler([Token("One. "), Pause()]))
    connection = await connect()
    await connection.send({"type": "text", "text": "tell me a story"})

    await connection.leave()

    log = rig.handler.log
    commits = [i for i, entry in enumerate(log) if entry.startswith("commit:")]
    assert len(commits) == 1
    assert log[-1] == "end_session"
    assert commits[0] < log.index("end_session")
    assert rig.handler.ended_sessions == [connection.ws.handshake_session_id]
    assert rig.registry.count() == 0


@pytest.mark.asyncio
async def test_every_connection_is_its_own_session_in_the_registry(monkeypatch):
    rig = install_runtime(monkeypatch)
    first = await connect()
    second = await connect()

    assert rig.registry.count() == 2
    assert rig.registry.endpoint_present("desktop")
    assert first.ws.handshake_session_id != second.ws.handshake_session_id

    await first.leave()
    assert rig.registry.count() == 1
    await second.leave()
    assert rig.registry.count() == 0


@pytest.mark.asyncio
async def test_a_connection_before_the_runtime_is_built_is_told_not_ready(monkeypatch):
    ws = FakeWebSocket()

    await server.websocket_endpoint(ws)
    await settle()

    assert ws.frames == [
        {
            "type": "error",
            "kind": "server",
            "message": "Server not ready",
            "retriable": True,
            "context": None,
        }
    ]
    assert ws.closed_with == 1013
