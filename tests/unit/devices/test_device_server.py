"""Tests for `backend/devices/server.py` and `config.py`: the second server in the process.

Every server test runs a real `DeviceServer` on `127.0.0.1:0` with TLS, inside the test's running
loop, so what it pins is what the backend's lifespan sees: the main server keeps SIGINT and
SIGTERM, a listener that cannot start leaves the process running, logging is left as it was,
and a stuck client cannot hold the backend's shutdown.

The certificate is made by `openssl` in a temporary directory, since `detect-private-key`
forbids committed keys. The apps served are small Starlette apps, since what is under test is
the server around them (`test_listener.py` covers the device app).
"""

from __future__ import annotations

import asyncio
import logging
import signal
import socket
import ssl
import subprocess  # nosec B404 - runs a fixed openssl argument list, never a shell
import time
from pathlib import Path

import pytest
import pytest_asyncio
import uvicorn
from starlette.applications import Starlette
from starlette.routing import WebSocketRoute
from starlette.websockets import WebSocket
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from backend.devices.config import DeviceListenerConfig, DeviceListenerConfigError
from backend.devices.server import (
    GRACEFUL_SHUTDOWN_S,
    STOP_TIMEOUT_S,
    WS_MAX_SIZE,
    DeviceListener,
    DeviceServer,
    bind_socket,
)

SERVER_LOGGER = "backend.devices.server"
LOOPBACK = "127.0.0.1"


# -- Fixtures ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tls_pair(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """A throwaway self-signed certificate for 127.0.0.1, and its key."""
    directory = tmp_path_factory.mktemp("tls")
    key, cert = directory / "server.key", directory / "server.pem"
    subprocess.run(  # nosec B603 B607 - fixed argument list, no shell
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "ec",
            "-pkeyopt",
            "ec_paramgen_curve:P-256",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-days",
            "1",
            "-subj",
            "/CN=device-listener-test",
            "-addext",
            "subjectAltName=IP:127.0.0.1",
        ],
        check=True,
        capture_output=True,
    )
    return cert, key


@pytest.fixture
def client_ssl(tls_pair) -> ssl.SSLContext:
    return ssl.create_default_context(cafile=str(tls_pair[0]))


def config_for(tls_pair, *, port: int = 0, cert: Path | None = None) -> DeviceListenerConfig:
    return DeviceListenerConfig(
        enabled=True,
        port=port,
        tls_cert=cert or tls_pair[0],
        tls_key=tls_pair[1],
        registry=Path("unused-registry.json"),
        host=LOOPBACK,
    )


class EchoApp:
    """A Starlette app with one route that echoes text and records what it saw."""

    def __init__(self) -> None:
        self.received: list[str] = []
        self.connected = asyncio.Event()
        self.app = Starlette(routes=[WebSocketRoute("/v2/device", self._endpoint)])

    async def _endpoint(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.connected.set()
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                return
            if "text" in message and message["text"] is not None:
                self.received.append(message["text"])
                await websocket.send_text(message["text"])


LISTENERS: list[DeviceListener] = []


@pytest_asyncio.fixture(autouse=True)
async def _stop_listeners():
    yield
    while LISTENERS:
        # Bounded, so a broken stop cap fails its test instead of hanging the suite.
        await asyncio.wait_for(LISTENERS.pop().stop(), timeout=10)


def listener_for(config: DeviceListenerConfig, app: object, **kwargs: object) -> DeviceListener:
    listener = DeviceListener(config, app, **kwargs)  # type: ignore[arg-type]
    LISTENERS.append(listener)
    return listener


def uri(listener: DeviceListener) -> str:
    return f"wss://{LOOPBACK}:{listener.port()}/v2/device"


def errors(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]


async def wait_for_state(listener: DeviceListener, state: str) -> None:
    for _ in range(200):
        if listener.state() == state:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"listener state stayed {listener.state()}, wanted {state}")


# -- Configuration -------------------------------------------------------------------------


class TestConfig:
    def test_with_nothing_set_the_listener_is_off_on_8443_with_the_mounted_paths(self):
        config = DeviceListenerConfig.from_env({})

        assert config.enabled is False
        assert config.port == 8443
        assert config.tls_cert == Path("/app/devices/tls/server.pem")
        assert config.tls_key == Path("/app/devices/tls/server.key")
        assert config.registry == Path("/app/devices/registry.json")
        assert config.host == "0.0.0.0"  # nosec B104 - asserting the container's bind

    def test_every_variable_is_read(self):
        config = DeviceListenerConfig.from_env(
            {
                "MIST_DEVICE_LISTENER": "on",
                "MIST_DEVICE_PORT": "9443",
                "MIST_DEVICE_TLS_CERT": "/x/cert.pem",
                "MIST_DEVICE_TLS_KEY": "/x/key.pem",
                "MIST_DEVICE_REGISTRY": "/x/registry.json",
            }
        )

        assert (config.enabled, config.port) == (True, 9443)
        assert (config.tls_cert, config.tls_key, config.registry) == (
            Path("/x/cert.pem"),
            Path("/x/key.pem"),
            Path("/x/registry.json"),
        )

    @pytest.mark.parametrize(
        "env",
        [
            pytest.param({"MIST_DEVICE_LISTENER": "yes"}, id="switch"),
            pytest.param({"MIST_DEVICE_PORT": "eighty"}, id="port-text"),
            pytest.param({"MIST_DEVICE_PORT": "70000"}, id="port-range"),
        ],
    )
    def test_a_bad_value_is_refused(self, env):
        with pytest.raises(DeviceListenerConfigError):
            DeviceListenerConfig.from_env(env)


# -- Off by default ------------------------------------------------------------------------


class RecordingBind:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def __call__(self, host: str, port: int) -> socket.socket:
        self.calls.append((host, port))
        return bind_socket(host, port)


@pytest.mark.asyncio
async def test_a_listener_that_is_off_opens_no_socket():
    bind = RecordingBind()
    listener = listener_for(DeviceListenerConfig(), EchoApp().app, bind=bind)

    await listener.start()

    assert bind.calls == []
    assert listener.state() == "off"


# -- Serving -------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_device_reaches_the_app_over_tls_and_stop_ends_it(tls_pair, client_ssl):
    echo = EchoApp()
    listener = listener_for(config_for(tls_pair), echo.app)

    await listener.start()
    async with connect(uri(listener), ssl=client_ssl) as client:
        await client.send("ping")
        answer = await client.recv()
    await listener.stop()

    assert answer == "ping"
    assert listener.state() == "off"


@pytest.mark.asyncio
async def test_the_server_is_configured_as_the_plan_says(tls_pair):
    listener = listener_for(config_for(tls_pair), EchoApp().app)

    await listener.start()
    config = listener.server_config()

    assert config is not None
    assert (config.ws_max_size, WS_MAX_SIZE) == (65536, 65536)
    assert config.lifespan == "off"
    assert config.log_config is None
    assert config.timeout_graceful_shutdown == GRACEFUL_SHUTDOWN_S == 5
    assert STOP_TIMEOUT_S == 6.0
    assert config.proxy_headers is False
    assert config.ssl is not None


@pytest.mark.asyncio
async def test_a_message_over_64_kib_closes_the_connection_1009(tls_pair, client_ssl):
    listener = listener_for(config_for(tls_pair), EchoApp().app)
    await listener.start()

    async with connect(uri(listener), ssl=client_ssl, max_size=None) as client:
        await client.send("x" * 1000)
        assert await client.recv() == "x" * 1000
        await client.send("x" * (WS_MAX_SIZE + 1))
        with pytest.raises(ConnectionClosed) as closed:
            await client.recv()

    assert closed.value.rcvd is not None
    assert closed.value.rcvd.code == 1009


# -- Signals -------------------------------------------------------------------------------


def handlers() -> tuple[object, object]:
    return signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)


@pytest.mark.asyncio
async def test_the_signal_handlers_are_the_same_before_while_serving_and_after(
    tls_pair, client_ssl
):
    echo = EchoApp()
    listener = listener_for(config_for(tls_pair), echo.app)
    before = handlers()

    await listener.start()
    async with connect(uri(listener), ssl=client_ssl) as client:
        await client.send("ping")
        await client.recv()
        serving = handlers()
    await listener.stop()
    after = handlers()

    assert serving == before
    assert after == before


async def hello(scope, receive, send) -> None:
    """A plain ASGI app for the main server."""
    if scope["type"] != "http":
        return
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


@pytest.mark.asyncio
async def test_a_sigterm_still_stops_the_main_server(tls_pair):
    received: list[int] = []
    original = signal.signal(signal.SIGTERM, lambda signum, frame: received.append(signum))
    try:
        main = uvicorn.Server(
            uvicorn.Config(
                hello,
                host=LOOPBACK,
                port=0,
                ws="websockets-sansio",
                lifespan="off",
                log_config=None,
            )
        )
        listener = listener_for(config_for(tls_pair), EchoApp().app)
        main_task = asyncio.get_running_loop().create_task(main.serve())
        for _ in range(200):
            if main.started:
                break
            await asyncio.sleep(0.01)
        await listener.start()

        signal.raise_signal(signal.SIGTERM)
        await asyncio.wait_for(main_task, timeout=5)

        assert main.should_exit
        # The main server re-raised the signal to the handler it found, after it stopped.
        assert received == [signal.SIGTERM]
        assert listener.state() == "on"
    finally:
        signal.signal(signal.SIGTERM, original)


# -- Failing to start leaves the process running -------------------------------------------


@pytest.mark.asyncio
async def test_a_port_already_taken_leaves_the_listener_failed(tls_pair, caplog):
    holder = bind_socket(LOOPBACK, 0)
    holder.listen()
    taken = holder.getsockname()[1]
    try:
        listener = listener_for(config_for(tls_pair, port=taken), EchoApp().app)
        with caplog.at_level(logging.ERROR, logger=SERVER_LOGGER):
            await listener.start()
    finally:
        holder.close()

    assert listener.state() == "failed"
    assert any("device_listener_failed" in line for line in errors(caplog))


@pytest.mark.asyncio
async def test_a_missing_certificate_leaves_the_listener_failed(tls_pair, tmp_path, caplog):
    listener = listener_for(config_for(tls_pair, cert=tmp_path / "missing.pem"), EchoApp().app)

    with caplog.at_level(logging.ERROR, logger=SERVER_LOGGER):
        await listener.start()

    assert listener.state() == "failed"
    assert any("missing.pem" in line for line in errors(caplog))


@pytest.mark.asyncio
async def test_a_system_exit_from_uvicorn_is_caught_and_leaves_the_listener_failed(
    tls_pair, caplog
):
    # uvicorn's `Config.load` calls `sys.exit(1)` for an app it cannot import.
    listener = listener_for(config_for(tls_pair), "no_such_module_for_device_tests:app")

    with caplog.at_level(logging.ERROR, logger=SERVER_LOGGER):
        await listener.start()

    assert listener.state() == "failed"
    assert any("device_listener_failed" in line for line in errors(caplog))


@pytest.mark.asyncio
async def test_an_os_error_inside_the_serve_task_leaves_the_listener_failed(tls_pair, caplog):
    def closed_socket(host: str, port: int) -> socket.socket:
        sock = bind_socket(host, port)
        sock.close()
        return sock

    listener = listener_for(config_for(tls_pair), EchoApp().app, bind=closed_socket)

    with caplog.at_level(logging.ERROR, logger=SERVER_LOGGER):
        await listener.start()
        await wait_for_state(listener, "failed")

    assert any("device_listener_failed" in line for line in errors(caplog))


@pytest.mark.asyncio
async def test_any_other_exception_in_the_serve_task_is_logged_and_failed(tls_pair, caplog):
    def datagram_socket(host: str, port: int) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((host, 0))
        return sock

    listener = listener_for(config_for(tls_pair), EchoApp().app, bind=datagram_socket)

    with caplog.at_level(logging.ERROR, logger=SERVER_LOGGER):
        await listener.start()
        await wait_for_state(listener, "failed")
        await asyncio.sleep(0)

    assert any("device_listener_crashed" in line for line in errors(caplog))


# -- Logging -------------------------------------------------------------------------------


LOGGERS = ("", "uvicorn", "uvicorn.error", "uvicorn.access", "uvicorn.asgi")


def logging_state() -> dict[str, tuple[list[logging.Handler], int, bool]]:
    state = {}
    for name in LOGGERS:
        log = logging.getLogger(name)
        state[name] = (list(log.handlers), log.level, log.propagate)
    return state


@pytest.mark.asyncio
async def test_the_root_and_uvicorn_loggers_are_unchanged(tls_pair, client_ssl):
    sentinel = logging.NullHandler()
    logging.getLogger().addHandler(sentinel)
    try:
        before = logging_state()
        listener = listener_for(config_for(tls_pair), EchoApp().app)

        await listener.start()
        async with connect(uri(listener), ssl=client_ssl) as client:
            await client.send("ping")
            await client.recv()
        serving = logging_state()
        await listener.stop()
        after = logging_state()
    finally:
        logging.getLogger().removeHandler(sentinel)

    assert serving == before
    assert after == before


# -- Stopping ------------------------------------------------------------------------------


async def stuck_client(port: int, context: ssl.SSLContext) -> asyncio.StreamWriter:
    """Complete the WebSocket handshake, then never read, answer or close."""
    reader, writer = await asyncio.open_connection(
        LOOPBACK, port, ssl=context, server_hostname=LOOPBACK
    )
    writer.write(
        (
            "GET /v2/device HTTP/1.1\r\n"
            f"Host: {LOOPBACK}:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        ).encode()
    )
    await writer.drain()
    await reader.readuntil(b"\r\n\r\n")
    return writer


class StubbornApp:
    """A session that ignores the disconnect and never returns, so the server cannot drain it.

    uvicorn drops an idle client by itself at shutdown; only a session that outlives its
    connection makes `stop` depend on the graceful bound and the stop cap.
    """

    def __init__(self) -> None:
        self.connected = asyncio.Event()
        self.app = Starlette(routes=[WebSocketRoute("/v2/device", self._endpoint)])

    async def _endpoint(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.connected.set()
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
        await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_a_session_that_never_returns_does_not_hold_stop_past_6_s(tls_pair, client_ssl):
    stubborn = StubbornApp()
    listener = listener_for(config_for(tls_pair), stubborn.app)
    await listener.start()
    writer = await stuck_client(listener.port(), client_ssl)
    await asyncio.wait_for(stubborn.connected.wait(), timeout=5)

    started = time.monotonic()
    # The outer bound only turns a hang into a failure; `stop`'s own cap is what is measured.
    await asyncio.wait_for(listener.stop(), timeout=STOP_TIMEOUT_S + 4)
    elapsed = time.monotonic() - started
    writer.transport.abort()

    # The graceful shutdown waits its 5 s for the session, then cancels it, inside the 6 s cap.
    assert GRACEFUL_SHUTDOWN_S <= elapsed <= STOP_TIMEOUT_S + 0.5, elapsed
    assert listener.state() == "off"


class HangingShutdown(DeviceServer):
    """A server whose shutdown never returns, so only `stop`'s own bound can end it."""

    async def shutdown(self, sockets: list[socket.socket] | None = None) -> None:
        await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_a_serve_task_that_outlives_its_bound_is_cancelled(tls_pair, caplog):
    listener = listener_for(config_for(tls_pair), EchoApp().app, server_class=HangingShutdown)
    await listener.start()

    started = time.monotonic()
    with caplog.at_level(logging.WARNING, logger=SERVER_LOGGER):
        # Bounded, so a broken cap fails here instead of hanging the suite.
        await asyncio.wait_for(listener.stop(timeout_s=0.3), timeout=5)
    elapsed = time.monotonic() - started

    assert elapsed < 2.0
    assert listener.state() == "off"
    assert any("cancelled" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_stop_before_start_and_twice_is_harmless(tls_pair):
    listener = listener_for(config_for(tls_pair), EchoApp().app)

    await listener.stop()
    await listener.start()
    await listener.stop()
    await listener.stop()

    assert listener.state() == "off"
