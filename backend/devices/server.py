"""The device listener's server: a second uvicorn server inside the backend's process.

uvicorn's `Server.serve()` wraps itself in `capture_signals()`, which takes SIGINT and SIGTERM
from whoever held them and re-raises them when it exits, and its startup calls `sys.exit(1)` on
a bind error. A second server built that way would steal the main server's shutdown signal and
could end the process on a taken port. So:

- `DeviceServer` makes `capture_signals` a no-op: the main server keeps sole ownership of the
  signals.
- `DeviceListener` binds the socket itself and loads the TLS configuration before the serve task
  starts, so a taken port or a missing or bad certificate is an `OSError` caught at `start`,
  logged at ERROR, and the listener's state is `failed` while the desktop path runs on.
  `SystemExit` is caught too (`Config.load` raises it for an app it cannot import). Anything
  else that ends the serve task is logged by its done-callback, which also marks it `failed`.
- `log_config=None`, so the second server never reconfigures the process's logging, and
  `lifespan="off"`.
- `stop` asks the server to exit, waits at most `STOP_TIMEOUT_S`, then cancels the task, so a
  device connection that never closes cannot hold the backend's shutdown. The server's own
  graceful shutdown is capped at `GRACEFUL_SHUTDOWN_S`, inside that bound.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import socket
from collections.abc import Callable, Iterator
from typing import Literal

import uvicorn
from starlette.types import ASGIApp

from .config import DeviceListenerConfig

logger = logging.getLogger(__name__)

WS_MAX_SIZE = 65536
"""The largest WebSocket message a device may send: a 20 ms frame is 652 bytes."""
GRACEFUL_SHUTDOWN_S = 5
STOP_TIMEOUT_S = 6.0
CANCEL_WAIT_S = 1.0
"""How long `stop` waits for a cancelled serve task to finish unwinding."""
STARTUP_POLL_S = 0.01
STARTUP_POLLS = 500
"""`start` waits at most 5 s for the server to listen."""

ListenerState = Literal["off", "on", "failed"]


class DeviceServer(uvicorn.Server):
    """A uvicorn server that leaves the process's signal handlers alone."""

    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        """Install nothing: SIGINT and SIGTERM belong to the main server."""
        yield


def bind_socket(host: str, port: int) -> socket.socket:
    """A TCP socket bound to `host:port`, not yet listening.

    Raises:
        OSError: The address is taken or cannot be bound.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # A restarted backend can rebind while old connections sit in TIME_WAIT; a port another
        # process is listening on still refuses.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
    except OSError:
        sock.close()
        raise
    return sock


class DeviceListener:
    """Starts, stops and reports the device listener's server.

    Args:
        config: Where to listen and the TLS files.
        app: The ASGI app to serve (`build_device_app`).
        bind: Makes the bound socket; tests pass one that fails.
        server_class: The server type; tests pass one whose shutdown hangs.
    """

    def __init__(
        self,
        config: DeviceListenerConfig,
        app: ASGIApp,
        *,
        bind: Callable[[str, int], socket.socket] = bind_socket,
        server_class: type[DeviceServer] = DeviceServer,
    ) -> None:
        self._config = config
        self._app = app
        self._bind = bind
        self._server_class = server_class
        self._server: DeviceServer | None = None
        self._socket: socket.socket | None = None
        self._task: asyncio.Task[None] | None = None
        self._state: ListenerState = "off"
        self._stopping = False

    def state(self) -> ListenerState:
        """`off` (disabled or stopped), `on` (serving) or `failed` (could not serve)."""
        return self._state

    def port(self) -> int | None:
        """The port bound, or None when no socket is bound; useful when `config.port` is 0."""
        sock = self._socket
        if sock is None or sock.fileno() < 0:
            return None
        return int(sock.getsockname()[1])

    def server_config(self) -> uvicorn.Config | None:
        """The server's uvicorn configuration, once `start` has built it."""
        return None if self._server is None else self._server.config

    async def start(self) -> None:
        """Bind, load TLS and start serving as a task; never raises.

        With the listener disabled it does nothing. A failure leaves the state `failed`.
        """
        config = self._config
        if not config.enabled or self._task is not None:
            return
        try:
            sock = self._bind(config.host, config.port)
        except OSError as exc:
            self._fail(f"cannot bind {config.host}:{config.port}: {exc}")
            return
        server = self._server_class(
            uvicorn.Config(
                self._app,
                ssl_certfile=str(config.tls_cert),
                ssl_keyfile=str(config.tls_key),
                # The sans-I/O implementation: `auto` picks the `websockets` legacy one, which
                # `websockets` deprecates.
                ws="websockets-sansio",
                ws_max_size=WS_MAX_SIZE,
                lifespan="off",
                log_config=None,
                timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_S,
                proxy_headers=False,
                server_header=False,
            )
        )
        try:
            server.config.load()
        except (OSError, SystemExit) as exc:
            sock.close()
            self._fail(
                f"cannot load the certificate {config.tls_cert} and key {config.tls_key}, "
                f"or the app: {exc!r}"
            )
            return
        self._server = server
        self._socket = sock
        self._stopping = False
        self._state = "on"
        self._task = asyncio.get_running_loop().create_task(
            self._serve(server, sock), name="device-listener"
        )
        self._task.add_done_callback(self._serve_done)
        # uvicorn has no started event; wait for its flag, or for the task to end early.
        for _ in range(STARTUP_POLLS):
            if server.started or self._task.done():
                break
            await asyncio.sleep(STARTUP_POLL_S)
        if server.started:
            logger.info("device_listener_started port=%s", self.port())

    async def _serve(self, server: DeviceServer, sock: socket.socket) -> None:
        try:
            await server.serve(sockets=[sock])
        except (OSError, SystemExit) as exc:
            self._fail(f"the server stopped: {exc!r}")
        finally:
            sock.close()

    def _serve_done(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error("device_listener_crashed", exc_info=error)
            self._state = "failed"
        elif not self._stopping and self._state == "on":
            logger.error("device_listener_failed error=the server exited on its own")
            self._state = "failed"

    def _fail(self, reason: str) -> None:
        logger.error("device_listener_failed error=%s; the desktop path is unaffected", reason)
        self._state = "failed"

    async def stop(self, timeout_s: float = STOP_TIMEOUT_S) -> None:
        """Ask the server to exit, wait at most `timeout_s`, then cancel it; never raises."""
        task, server = self._task, self._server
        if task is None or server is None:
            return
        self._stopping = True
        server.should_exit = True
        done, _ = await asyncio.wait({task}, timeout=timeout_s)
        if not done:
            logger.warning(
                "device_listener_stop: still running after %.1f s; the task is cancelled",
                timeout_s,
            )
            task.cancel()
            await asyncio.wait({task}, timeout=CANCEL_WAIT_S)
        self._task = None
        if self._state == "on":
            self._state = "off"
        logger.info("device_listener_stopped")
