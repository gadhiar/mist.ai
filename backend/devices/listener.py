"""The device route: `/v2/device`, authenticated before accept, then one realtime session.

A room device connects with `Authorization: Bearer <name>.<secret>`. The token is checked
before the WebSocket is accepted, so a refusal is a close 1008 (an HTTP 403) that reveals
nothing; `device_auth_failed` logs the claimed name and the peer, never the secret. The peer is
the forwarder's or Docker's proxy's address, so the limiter keys on the claimed name
(`limiter.py`).

An accepted device gets `serve(socket, <device name>, "room")`: the backend's per-connection
serve, which binds the session's endpoint to the device's name. A device holds one session: a
second connection replaces the first, which is closed 4001. The registry file is re-read when it
changes, at every attempt and every `REVOCATION_POLL_S` while a device is connected, and a
device revoked or removed meanwhile is closed 4003.

The listener closes a session by handing the session a disconnect through `DeviceSocket`, so the
session ends as it does when the device leaves (its turn committed), and only then sends the
close code.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Literal, Protocol

from starlette.applications import Starlette
from starlette.routing import WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect, WebSocketState

from .limiter import AuthLimiter, Clock
from .registry import DeviceRegistry, claimed_name, valid_name

logger = logging.getLogger(__name__)

DEVICE_PATH = "/v2/device"
CLOSE_POLICY = 1008
CLOSE_REPLACED = 4001
CLOSE_REVOKED = 4003
REVOCATION_POLL_S = 5.0
"""Half the 10 s a revoked device's session may outlive its revocation."""
REPLACE_WAIT_S = 5.0
"""How long a new connection waits for the session it replaces to end."""
BEARER_PREFIX = "bearer "

Serve = Callable[["DeviceSocket", str, Literal["room"]], Awaitable[None]]


class ListenerClock(Clock, Protocol):
    """The limiter's clock, plus the sleep the revocation check waits on."""

    async def sleep(self, seconds: float) -> None:
        """Wait `seconds`."""
        ...


class SystemClock:
    """The process's monotonic clock and the loop's sleep."""

    def monotonic(self) -> float:
        """The process's monotonic clock; injected so tests drive limiter and revocation time."""
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        """The loop's sleep, injected so tests can step the revocation poll without waiting."""
        await asyncio.sleep(seconds)


class DeviceSocket:
    """An accepted device socket as its session sees it, which the listener can end.

    A pump task reads the socket into an inbox, so `end` can put a disconnect in front of the
    session without cancelling a read. `close_code` is the code the connection ended with: the
    one this side closed with, else the device's own.
    """

    def __init__(self, websocket: WebSocket) -> None:
        self._websocket = websocket
        self._inbox: asyncio.Queue[Mapping[str, object]] = asyncio.Queue()
        self._pump: asyncio.Task[None] | None = None
        self._closed = False
        self.ending: int | None = None
        self.close_code: int | None = None
        self.done = asyncio.Event()

    def start(self) -> None:
        """Start reading the socket."""
        self._pump = asyncio.get_running_loop().create_task(self._read(), name="device-socket")

    async def _read(self) -> None:
        while True:
            message = await self._websocket.receive()
            self._inbox.put_nowait(message)
            if message.get("type") == "websocket.disconnect":
                if self.close_code is None:
                    code = message.get("code")
                    self.close_code = code if isinstance(code, int) else None
                return

    async def receive(self) -> Mapping[str, object]:
        """The next message from the device, or the disconnect `end` queued."""
        return await self._inbox.get()

    async def send_text(self, data: str) -> None:
        """Send one text frame."""
        await self._websocket.send_text(data)

    async def send_bytes(self, data: bytes) -> None:
        """Send one binary frame."""
        await self._websocket.send_bytes(data)

    async def send_json(self, payload: object) -> None:
        """Send one JSON text frame."""
        await self._websocket.send_json(payload)

    async def close(self, code: int = 1000) -> None:
        """Close the connection with `code`, once."""
        if self._closed:
            return
        self._closed = True
        if self.close_code is None:
            self.close_code = code
        if self._websocket.client_state is not WebSocketState.CONNECTED:
            return
        try:
            await self._websocket.close(code)
        except (OSError, WebSocketDisconnect) as exc:
            logger.debug("device socket already gone at close %d: %s", code, exc)

    def end(self, code: int) -> None:
        """End the session as a disconnect would; `finish` then closes with `code`."""
        if self.ending is not None:
            return
        self.ending = code
        self._inbox.put_nowait({"type": "websocket.disconnect", "code": code})

    async def finish(self) -> None:
        """Stop reading, and send the close code `end` asked for."""
        pump = self._pump
        if pump is not None:
            pump.cancel()
            with contextlib.suppress(asyncio.CancelledError, WebSocketDisconnect, RuntimeError):
                await pump
        if self.ending is not None:
            await self.close(self.ending)


def _bearer(header: str | None) -> str | None:
    if header is None or not header.lower().startswith(BEARER_PREFIX):
        return None
    token = header[len(BEARER_PREFIX) :].strip()
    return token or None


def _peer(websocket: WebSocket) -> str:
    client = websocket.client
    return "?" if client is None else f"{client.host}:{client.port}"


class DeviceApp(Starlette):
    """The device listener's ASGI app: one route and nothing else.

    Args:
        registry: The registered devices; re-read when the file changes.
        serve: The per-connection serve each accepted device gets.
        limiter: The per-id and global limits on failed attempts.
        clock: The revocation check's sleep.
    """

    def __init__(
        self,
        registry: DeviceRegistry,
        serve: Serve,
        limiter: AuthLimiter,
        clock: ListenerClock,
    ) -> None:
        super().__init__(routes=[WebSocketRoute(DEVICE_PATH, self._endpoint)])
        self._registry = registry
        self._serve = serve
        self._limiter = limiter
        self._clock = clock
        self._live: dict[str, DeviceSocket] = {}
        self._watch: asyncio.Task[None] | None = None

    def connected(self) -> list[str]:
        """The names of the devices with a live session, sorted."""
        return sorted(self._live)

    async def _endpoint(self, websocket: WebSocket) -> None:
        peer = _peer(websocket)
        bearer = _bearer(websocket.headers.get("authorization"))
        claimed = claimed_name(bearer) if bearer is not None else None
        device_id = claimed if claimed is not None and valid_name(claimed) else None
        limiter = self._limiter
        if limiter.paused():
            await self._refuse(websocket, device_id, peer, "paused")
            return
        if not limiter.allowed(device_id):
            await self._refuse(websocket, device_id, peer, "blocked")
            return
        self._reload()
        record = self._registry.verify(bearer) if bearer is not None else None
        if record is None:
            limiter.failed(device_id)
            await self._refuse(websocket, device_id, peer, self._reason(bearer, device_id))
            return
        await websocket.accept()
        await self._serve_device(websocket, record.name)

    def _reason(self, bearer: str | None, device_id: str | None) -> str:
        if bearer is None:
            return "no_token"
        if device_id is None:
            return "malformed"
        known = {record.name: record for record in self._registry.devices()}
        record = known.get(device_id)
        if record is None:
            return "unknown_device"
        return "revoked" if record.revoked else "bad_secret"

    async def _refuse(
        self, websocket: WebSocket, device_id: str | None, peer: str, reason: str
    ) -> None:
        logger.warning(
            "device_auth_failed device=%s peer=%s reason=%s", device_id or "?", peer, reason
        )
        await websocket.close(code=CLOSE_POLICY)

    async def _serve_device(self, websocket: WebSocket, name: str) -> None:
        connection = DeviceSocket(websocket)
        previous = self._live.get(name)
        self._live[name] = connection
        if previous is not None:
            logger.info("device_replaced device=%s", name)
            previous.end(CLOSE_REPLACED)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(previous.done.wait(), timeout=REPLACE_WAIT_S)
        logger.info("device_connected device=%s", name)
        self._ensure_watch()
        connection.start()
        try:
            await self._serve(connection, name, "room")
        finally:
            await connection.finish()
            if self._live.get(name) is connection:
                del self._live[name]
            connection.done.set()
            logger.info("device_disconnected device=%s code=%s", name, connection.close_code)
            if not self._live and self._watch is not None:
                self._watch.cancel()
                self._watch = None

    # -- Revocation ------------------------------------------------------------------------

    def _reload(self) -> None:
        if self._registry.reload_if_changed():
            self._close_revoked()

    def _close_revoked(self) -> None:
        admitted = {record.name for record in self._registry.devices() if not record.revoked}
        for name, connection in list(self._live.items()):
            if name not in admitted and connection.ending is None:
                logger.warning("device_revoked_closed device=%s", name)
                connection.end(CLOSE_REVOKED)

    def _ensure_watch(self) -> None:
        if self._watch is None or self._watch.done():
            self._watch = asyncio.get_running_loop().create_task(
                self._watch_registry(), name="device-registry-watch"
            )

    async def _watch_registry(self) -> None:
        while self._live:
            await self._clock.sleep(REVOCATION_POLL_S)
            self._reload()


def build_device_app(
    registry: DeviceRegistry,
    serve: Serve,
    limiter: AuthLimiter,
    clock: ListenerClock,
) -> DeviceApp:
    """The device listener's app over `registry`, serving each device with `serve`."""
    return DeviceApp(registry, serve, limiter, clock)
