"""T13: streamed log lines reach only the sessions subscribed to them.

The real chain: `WebSocketLogHandler` puts a `LogLine(text, level)` on `server.message_queue`,
`server.broadcast_messages` hands it to `SessionRegistry.publish_log`, and each subscribed
session queues it on its own `bulk` lane. `subscribe_logs` reaches the handler's
`set_streaming` through `SessionRegistry.set_log_subscription`, which gates the handler on
every live subscription.
"""

from __future__ import annotations

import asyncio
import json
import logging

import pytest

from backend import server
from backend.log_handler import LogLine, WebSocketLogHandler
from tests.unit.test_session_id_propagation import (  # noqa: F401 -- autouse fixtures
    connect,
    install_runtime,
    no_runtime,
    reset_session_context,
)

from .turn_fakes import settle

pytestmark = pytest.mark.asyncio


def _logs(frames: list[dict[str, object]]) -> list[dict[str, object]]:
    return [frame for frame in frames if frame.get("type") == "log"]


def _record(message: str, level: int = logging.ERROR) -> logging.LogRecord:
    return logging.LogRecord("backend.example", level, __file__, 1, message, None, None)


async def _flush(queue: asyncio.Queue) -> None:
    """Let the handler's thread-safe put land, and the broadcaster drain it."""
    for _ in range(5):
        await settle()
        await queue.join()
    await settle()


async def test_the_handler_puts_a_log_line_with_its_level_on_the_queue():
    queue: asyncio.Queue = asyncio.Queue()
    handler = WebSocketLogHandler(asyncio.get_running_loop(), queue)
    handler.set_streaming(True, None)

    handler.emit(_record("disk is full", logging.WARNING))
    await settle()

    line = queue.get_nowait()
    assert isinstance(line, LogLine)
    assert line.level == "WARNING"
    payload = json.loads(line.text)
    assert (payload["type"], payload["level"], payload["message"]) == (
        "log",
        "WARNING",
        "disk is full",
    )


async def test_only_the_subscribed_session_receives_the_line_and_streaming_stops_with_it(
    monkeypatch,
):
    queue: asyncio.Queue = asyncio.Queue()
    monkeypatch.setattr(server, "message_queue", queue)
    handler = WebSocketLogHandler(asyncio.get_running_loop(), queue)
    install_runtime(monkeypatch, log_gate=handler)
    broadcaster = asyncio.get_running_loop().create_task(server.broadcast_messages())
    try:
        subscribed = await connect()
        other = await connect()
        await subscribed.send({"type": "subscribe_logs", "enabled": True, "levels": ["ERROR"]})
        assert handler._streaming_enabled

        handler.emit(_record("engine failed"))
        handler.emit(_record("just chatter", logging.INFO))  # below the subscription
        await _flush(queue)

        (line,) = _logs(subscribed.ws.frames)
        assert line["message"] == "engine failed"
        assert _logs(other.ws.frames) == []

        await subscribed.leave()
        assert not handler._streaming_enabled, "streaming stays on with no subscriber left"
        handler.emit(_record("after the subscriber left"))
        await _flush(queue)
        assert _logs(other.ws.frames) == []
        await other.leave()
    finally:
        broadcaster.cancel()
        await asyncio.gather(broadcaster, return_exceptions=True)


async def test_any_other_queued_message_goes_to_every_session(monkeypatch):
    queue: asyncio.Queue = asyncio.Queue()
    monkeypatch.setattr(server, "message_queue", queue)
    install_runtime(monkeypatch)
    broadcaster = asyncio.get_running_loop().create_task(server.broadcast_messages())
    try:
        first = await connect()
        second = await connect()

        await queue.put(json.dumps({"type": "heartbeat", "timestamp": 1}))
        await _flush(queue)

        for connection in (first, second):
            assert {"type": "heartbeat", "timestamp": 1} in connection.ws.frames
            await connection.leave()
    finally:
        broadcaster.cancel()
        await asyncio.gather(broadcaster, return_exceptions=True)


async def test_with_no_registry_a_queued_message_is_dropped_and_the_broadcaster_goes_on(
    monkeypatch,
):
    queue: asyncio.Queue = asyncio.Queue()
    monkeypatch.setattr(server, "message_queue", queue)
    broadcaster = asyncio.get_running_loop().create_task(server.broadcast_messages())
    try:
        await queue.put(LogLine(text="{}", level="ERROR"))
        await queue.put("{}")
        await _flush(queue)

        assert queue.empty()
        assert not broadcaster.done()
    finally:
        broadcaster.cancel()
        await asyncio.gather(broadcaster, return_exceptions=True)
