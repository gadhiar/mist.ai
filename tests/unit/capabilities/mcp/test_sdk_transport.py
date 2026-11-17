"""Tests for `McpSdkTransport`, the `McpTransport` over the `mcp` SDK.

Nothing here opens a socket. Tool behaviour runs against an `mcp.server.MCPServer` that
registers the tools `fake_home_assistant.py` models, reached in process through
`Client(server, raise_exceptions=True)`. The Streamable HTTP path (the bearer header, HTTP 401
and 403, connection failures) runs against an `httpx2.MockTransport` that answers MCP's JSON-RPC
messages itself.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Mapping
from pathlib import Path

import anyio
import httpx2
import pytest
from mcp import Client
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from backend import factories
from backend.capabilities.config import load_capabilities_config
from backend.capabilities.mcp.connection import BackoffPolicy, McpConnection
from backend.capabilities.mcp.sdk_transport import (
    MAX_CALL_TIMEOUT_S,
    McpSdkTransport,
    _sdk_timeout,
)
from backend.capabilities.replies import NoAddress
from backend.capabilities.secrets import EnvSecretSource, Secret, SecretRegistry
from backend.chat.context_budget import ApproximateTokenCounter
from backend.errors import CapabilityUnavailableError, McpTransportError
from tests.unit.capabilities.fakes import FakeClock, FakePower, FakeTiers, RecordingEventSink
from tests.unit.capabilities.mcp.fake_home_assistant import (
    DEFAULT_LIVE_CONTEXT,
    LIGHT_SET,
    LIVE_CONTEXT,
    TURN_OFF,
    TURN_ON,
    FakeHomeAssistant,
)

LOGGER = "backend.capabilities.mcp.sdk_transport"
HA_URL = "http://homeassistant.local:8123/api/mcp"
TOKEN = "ha-token-value-0123456789"  # noqa: S105 - a fake token for the redaction checks
MISSING_DEVICE = "Attic Lamp"
SLOW_TOOL = "SlowTool"

# -- The in-memory Home Assistant ----------------------------------------------------------------

_Text = str | None
_Names = list[str] | None


def _home_assistant(calls: list[tuple[str, dict[str, object]]] | None = None) -> MCPServer:
    """An `MCPServer` listing the tools `FakeHomeAssistant` lists, with its descriptions."""
    recorded = calls if calls is not None else []
    descriptions = {spec.name: spec.description for spec in FakeHomeAssistant().tool_specs()}
    server = MCPServer("fake-home-assistant")

    def answer(tool: str, **slots: object) -> dict[str, bool]:
        recorded.append((tool, {k: v for k, v in slots.items() if v is not None}))
        if slots.get("name") == MISSING_DEVICE:
            raise ToolError(f"Unable to find a device named {MISSING_DEVICE}")
        return {"success": True}

    def turn_on(
        name: _Text = None,
        area: _Text = None,
        floor: _Text = None,
        domain: _Names = None,
        device_class: _Names = None,
    ) -> dict[str, bool]:
        return answer(
            TURN_ON, name=name, area=area, floor=floor, domain=domain, device_class=device_class
        )

    def turn_off(
        name: _Text = None,
        area: _Text = None,
        floor: _Text = None,
        domain: _Names = None,
        device_class: _Names = None,
    ) -> dict[str, bool]:
        return answer(
            TURN_OFF, name=name, area=area, floor=floor, domain=domain, device_class=device_class
        )

    def light_set(
        name: _Text = None,
        area: _Text = None,
        floor: _Text = None,
        domain: _Names = None,
        brightness: int | None = None,
        color: _Text = None,
        temperature: int | None = None,
    ) -> dict[str, bool]:
        return answer(
            LIGHT_SET,
            name=name,
            area=area,
            floor=floor,
            domain=domain,
            brightness=brightness,
            color=color,
            temperature=temperature,
        )

    def live_context() -> str:
        recorded.append((LIVE_CONTEXT, {}))
        return DEFAULT_LIVE_CONTEXT

    async def slow() -> str:
        await anyio.sleep(30)
        return "late"

    server.add_tool(turn_on, name=TURN_ON, description=descriptions[TURN_ON])
    server.add_tool(turn_off, name=TURN_OFF, description=descriptions[TURN_OFF])
    server.add_tool(light_set, name=LIGHT_SET, description=descriptions[LIGHT_SET])
    server.add_tool(
        live_context,
        name=LIVE_CONTEXT,
        description=descriptions[LIVE_CONTEXT],
        structured_output=False,
    )
    server.add_tool(slow, name=SLOW_TOOL, description="Never answers in time")
    return server


class RecordingClient(Client):
    """A `Client` that records the task entering and the task exiting its context."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.entered_in: asyncio.Task[object] | None = None
        self.exited_in: asyncio.Task[object] | None = None

    async def __aenter__(self) -> RecordingClient:
        self.entered_in = asyncio.current_task()
        await super().__aenter__()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        self.exited_in = asyncio.current_task()
        await super().__aexit__(*exc_info)  # type: ignore[arg-type]


def _in_memory(server: MCPServer, clients: list[RecordingClient] | None = None) -> McpSdkTransport:
    def build() -> Client:
        client = RecordingClient(server, raise_exceptions=True)
        if clients is not None:
            clients.append(client)
        return client

    return McpSdkTransport(HA_URL, Secret(TOKEN), client_factory=build)


# -- A Streamable HTTP server answered by `httpx2.MockTransport` ---------------------------------


class ScriptedHttpServer:
    """Answers MCP's JSON-RPC over HTTP the way a handshake-era server does.

    `server/discover` is answered "method not found", so the client falls back to `initialize`.

    Attributes:
        authorizations: The Authorization header of every request, in order.
        methods: The JSON-RPC method of every request, in order.
        status: When set, every request is answered with this bare HTTP status.
        call_answer: Answers each `tools/call`: a JSON-RPC result or error object, or an
            exception the mock raises instead of answering.
        closed: The HTTP transport was closed.
    """

    def __init__(self) -> None:
        self.authorizations: list[str | None] = []
        self.methods: list[str] = []
        self.status: int | None = None
        self.call_answer: Callable[[], Mapping[str, object] | Exception] = lambda: {
            "result": {
                "content": [{"type": "text", "text": '{"success": true}'}],
                "structuredContent": {"success": True},
                "isError": False,
            }
        }
        self.closed = False

    def transport(self) -> httpx2.AsyncBaseTransport:
        server = self

        class _Mock(httpx2.MockTransport):
            async def aclose(self) -> None:
                server.closed = True
                await super().aclose()

        return _Mock(self._handle)

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        self.authorizations.append(request.headers.get("authorization"))
        if request.method != "POST":
            return httpx2.Response(405)
        message = json.loads(request.content)
        method = message.get("method", "")
        self.methods.append(method)
        if self.status is not None:
            return httpx2.Response(self.status, text="401: Unauthorized")
        if "id" not in message:
            return httpx2.Response(202)
        if method == "server/discover":
            answer: Mapping[str, object] = {"error": {"code": -32601, "message": "Not found"}}
        elif method == "initialize":
            answer = {
                "result": {
                    "protocolVersion": message["params"]["protocolVersion"],
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "fake-home-assistant", "version": "1"},
                }
            }
        elif method == "tools/list":
            answer = {
                "result": {
                    "tools": [
                        {
                            "name": spec.name,
                            "description": spec.description,
                            "inputSchema": dict(spec.input_schema),
                        }
                        for spec in FakeHomeAssistant().tool_specs()
                    ]
                }
            }
        else:
            scripted = self.call_answer()
            if isinstance(scripted, Exception):
                raise scripted
            answer = scripted
        return httpx2.Response(200, json={"jsonrpc": "2.0", "id": message["id"], **answer})


def _over_http(server: ScriptedHttpServer) -> McpSdkTransport:
    return McpSdkTransport(HA_URL, Secret(TOKEN), http_transport=server.transport())


def _assert_no_token(caplog: pytest.LogCaptureFixture, *errors: BaseException) -> None:
    for record in caplog.records:
        assert TOKEN not in record.getMessage(), record.getMessage()
        if record.exc_info:
            formatted = logging.Formatter().formatException(record.exc_info)
            assert TOKEN not in formatted
    for error in errors:
        assert TOKEN not in str(error)
        assert TOKEN not in repr(error)


# -- The timeout boundary ------------------------------------------------------------------------


class TestSdkTimeout:
    def test_the_bound_converts_to_the_sdk_form_of_3600_seconds(self, caplog):
        caplog.set_level(logging.WARNING, logger=LOGGER)

        converted = _sdk_timeout(MAX_CALL_TIMEOUT_S)

        assert MAX_CALL_TIMEOUT_S == 3600.0
        assert isinstance(converted, float)
        assert converted == 3600.0
        assert caplog.records == []

    def test_a_value_past_the_bound_is_clamped_with_one_warning(self, caplog):
        caplog.set_level(logging.WARNING, logger=LOGGER)

        converted = _sdk_timeout(1e300)

        assert converted == _sdk_timeout(MAX_CALL_TIMEOUT_S)
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "1e+300" in warnings[0].getMessage()
        assert "3600" in warnings[0].getMessage()

    @pytest.mark.parametrize("value", [8.0, 0.5, 3599.0])
    def test_a_value_within_the_bound_is_unchanged(self, value):
        assert _sdk_timeout(value) == value

    @pytest.mark.asyncio
    async def test_a_call_at_the_bound_returns_its_result(self):
        transport = _in_memory(_home_assistant())
        await transport.open()
        try:
            result = await transport.call_tool(
                TURN_ON, {"name": "Bedroom Lamp"}, MAX_CALL_TIMEOUT_S
            )
        finally:
            await transport.close()

        assert result.is_error is False
        assert result.structured == {"success": True}

    @pytest.mark.asyncio
    async def test_a_transport_warns_once_however_often_it_clamps(self, caplog):
        caplog.set_level(logging.WARNING, logger=LOGGER)
        transport = _in_memory(_home_assistant())
        await transport.open()
        try:
            for _ in range(3):
                result = await transport.call_tool(TURN_ON, {"name": "Bedroom Lamp"}, 1e300)
                assert result.is_error is False
        finally:
            await transport.close()

        clamps = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(clamps) == 1


# -- Tools in memory -----------------------------------------------------------------------------


class TestInMemory:
    @pytest.mark.asyncio
    async def test_list_tools_returns_the_servers_tools(self):
        transport = _in_memory(_home_assistant())
        await transport.open()
        try:
            tools = {spec.name: spec for spec in await transport.list_tools()}
        finally:
            await transport.close()

        expected = {spec.name: spec for spec in FakeHomeAssistant().tool_specs()}
        assert set(expected) <= set(tools)
        for name, spec in expected.items():
            assert tools[name].description == spec.description
            assert set(tools[name].input_schema["properties"]) == set(  # type: ignore[call-overload]
                spec.input_schema["properties"]  # type: ignore[call-overload]
            )

    @pytest.mark.asyncio
    async def test_a_call_reaches_the_tool_with_its_arguments(self):
        calls: list[tuple[str, dict[str, object]]] = []
        transport = _in_memory(_home_assistant(calls))
        await transport.open()
        try:
            result = await transport.call_tool(
                LIGHT_SET, {"name": "Bedroom Lamp", "brightness": 40}, 8.0
            )
        finally:
            await transport.close()

        assert calls == [(LIGHT_SET, {"name": "Bedroom Lamp", "brightness": 40})]
        assert result.is_error is False
        assert json.loads(result.text) == {"success": True}
        assert result.structured == {"success": True}

    @pytest.mark.asyncio
    async def test_a_text_result_has_no_structured_content(self):
        transport = _in_memory(_home_assistant())
        await transport.open()
        try:
            result = await transport.call_tool(LIVE_CONTEXT, {}, 8.0)
        finally:
            await transport.close()

        assert result.is_error is False
        assert result.text == DEFAULT_LIVE_CONTEXT
        assert result.structured is None

    @pytest.mark.asyncio
    async def test_a_tool_error_is_a_result_not_an_exception(self):
        transport = _in_memory(_home_assistant())
        await transport.open()
        try:
            result = await transport.call_tool(TURN_ON, {"name": MISSING_DEVICE}, 8.0)
        finally:
            await transport.close()

        assert result.is_error is True
        assert f"Unable to find a device named {MISSING_DEVICE}" in result.text

    @pytest.mark.asyncio
    async def test_a_call_that_outlives_its_timeout_raises_a_transport_error(self):
        transport = _in_memory(_home_assistant())
        await transport.open()
        try:
            with pytest.raises(McpTransportError, match="timed out"):
                await transport.call_tool(SLOW_TOOL, {}, 0.05)
        finally:
            await transport.close()

    @pytest.mark.asyncio
    async def test_a_call_with_no_open_session_raises_a_transport_error(self):
        transport = _in_memory(_home_assistant())

        with pytest.raises(McpTransportError):
            await transport.call_tool(TURN_ON, {}, 8.0)
        with pytest.raises(McpTransportError):
            await transport.list_tools()

        await transport.open()
        await transport.close()
        with pytest.raises(McpTransportError):
            await transport.call_tool(TURN_ON, {}, 8.0)

    @pytest.mark.asyncio
    async def test_close_without_open_does_nothing(self):
        await _in_memory(_home_assistant()).close()

    @pytest.mark.asyncio
    async def test_open_and_close_run_in_the_supervisor_task_and_calls_in_others(self):
        clients: list[RecordingClient] = []
        calls: list[tuple[str, dict[str, object]]] = []
        server = _home_assistant(calls)
        connection = McpConnection(
            lambda: _in_memory(server, clients), FakeClock(), BackoffPolicy(jitter=lambda d: d)
        )
        await connection.start()
        try:
            loop = asyncio.get_running_loop()
            give_up_at = loop.time() + 5.0
            while connection.state().state != "available":
                assert loop.time() < give_up_at, connection.state().reason
                await asyncio.sleep(0.005)
            names = {spec.name for spec in connection.tools()}
            other = asyncio.create_task(connection.call_tool(TURN_OFF, {"area": "Bedroom"}, 8.0))
            result = await other
        finally:
            await connection.stop()

        assert {TURN_ON, TURN_OFF, LIGHT_SET, LIVE_CONTEXT} <= names
        assert result.is_error is False
        assert calls == [(TURN_OFF, {"area": "Bedroom"})]
        (client,) = clients
        assert client.entered_in is not None
        assert client.entered_in is client.exited_in
        assert client.entered_in is not asyncio.current_task()
        assert client.entered_in is not other


# -- Streamable HTTP -----------------------------------------------------------------------------


class TestHttp:
    @pytest.mark.asyncio
    async def test_a_session_over_http_sends_the_bearer_token_on_every_request(self):
        server = ScriptedHttpServer()
        transport = _over_http(server)

        await transport.open()
        try:
            tools = await transport.list_tools()
            result = await transport.call_tool(TURN_ON, {"name": "Bedroom Lamp"}, 8.0)
        finally:
            await transport.close()

        assert [spec.name for spec in tools] == [
            spec.name for spec in FakeHomeAssistant().tool_specs()
        ]
        assert result.structured == {"success": True}
        assert "initialize" in server.methods
        assert "tools/call" in server.methods
        assert server.authorizations
        assert set(server.authorizations) == {f"Bearer {TOKEN}"}
        assert server.closed is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [401, 403])
    async def test_a_refused_credential_on_open_is_auth_rejected(self, status, caplog):
        caplog.set_level(logging.DEBUG)
        server = ScriptedHttpServer()
        server.status = status
        transport = _over_http(server)

        with pytest.raises(CapabilityUnavailableError) as raised:
            await transport.open()

        assert str(raised.value) == "auth rejected"
        assert not isinstance(raised.value, McpTransportError)
        assert set(server.authorizations) == {f"Bearer {TOKEN}"}
        assert server.closed is True
        _assert_no_token(caplog, raised.value)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [401, 403])
    async def test_a_refused_credential_on_a_call_is_auth_rejected(self, status):
        server = ScriptedHttpServer()
        transport = _over_http(server)
        await transport.open()
        try:
            server.status = status
            with pytest.raises(CapabilityUnavailableError, match="^auth rejected$"):
                await transport.call_tool(TURN_ON, {}, 8.0)
        finally:
            await transport.close()

    @pytest.mark.asyncio
    async def test_an_unreachable_server_on_open_is_a_transport_error(self, caplog):
        caplog.set_level(logging.DEBUG)
        server = ScriptedHttpServer()
        transport = McpSdkTransport(
            HA_URL,
            Secret(TOKEN),
            http_transport=httpx2.MockTransport(_refuse),
        )

        with pytest.raises(McpTransportError) as raised:
            await transport.open()

        assert "Connection refused" in str(raised.value)
        _assert_no_token(caplog, raised.value)
        assert server.methods == []

    @pytest.mark.asyncio
    async def test_a_connection_lost_mid_call_fails_the_call_and_spares_the_opener(self):
        server = ScriptedHttpServer()
        transport = _over_http(server)
        await transport.open()
        try:
            server.call_answer = lambda: httpx2.ConnectError("[Errno 111] Connection refused")
            with pytest.raises(McpTransportError, match="Connection refused"):
                await transport.call_tool(TURN_ON, {}, 8.0)
            # The SDK's task group was entered in this task; a failure escaping into it would
            # cancel this task at its next await.
            await asyncio.sleep(0.05)
        finally:
            await transport.close()
        assert asyncio.current_task().cancelling() == 0  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_a_body_lost_mid_read_fails_the_call_and_spares_the_opener(self):
        server = ScriptedHttpServer()
        transport = McpSdkTransport(
            HA_URL, Secret(TOKEN), http_transport=_BreaksOnCall(server.transport())
        )
        await transport.open()
        try:
            with pytest.raises(McpTransportError):
                await transport.call_tool(TURN_ON, {}, 8.0)
            await asyncio.sleep(0.05)
        finally:
            await transport.close()
        assert asyncio.current_task().cancelling() == 0  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_a_server_error_echoing_the_token_never_shows_it(self, caplog):
        caplog.set_level(logging.DEBUG)
        server = ScriptedHttpServer()
        server.call_answer = lambda: {
            "error": {"code": -32602, "message": f"bad request with token {TOKEN}"}
        }
        transport = _over_http(server)
        await transport.open()
        try:
            with pytest.raises(McpTransportError) as raised:
                await transport.call_tool(TURN_ON, {}, 8.0)
        finally:
            await transport.close()

        assert "bad request with token ***" in str(raised.value)
        assert raised.value.__cause__ is None or TOKEN not in str(raised.value.__cause__)
        records = [r for r in caplog.records if r.name == LOGGER]
        _assert_no_token(caplog, raised.value)
        assert all(TOKEN not in r.getMessage() for r in records)

    @pytest.mark.asyncio
    async def test_a_whole_session_logs_no_token(self, caplog):
        caplog.set_level(logging.DEBUG)
        server = ScriptedHttpServer()
        transport = _over_http(server)

        await transport.open()
        await transport.list_tools()
        await transport.call_tool(TURN_ON, {"name": "Bedroom Lamp"}, 1e300)
        await transport.close()

        assert caplog.records
        _assert_no_token(caplog)

    def test_the_token_is_not_in_the_transports_repr(self):
        transport = _over_http(ScriptedHttpServer())

        assert TOKEN not in repr(transport)
        assert TOKEN not in str(transport)


def _refuse(request: httpx2.Request) -> httpx2.Response:
    raise httpx2.ConnectError("[Errno 111] Connection refused", request=request)


class _BrokenBody(httpx2.AsyncByteStream):
    async def __aiter__(self):
        yield b'{"jsonrpc": "2.0", '
        raise httpx2.ReadError("connection reset mid-body")


class _BreaksOnCall(httpx2.AsyncBaseTransport):
    """Delegates to a scripted server, but a `tools/call` answer breaks off mid-body."""

    def __init__(self, inner: httpx2.AsyncBaseTransport) -> None:
        self._inner = inner

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        response = await self._inner.handle_async_request(request)
        if request.method == "POST" and json.loads(request.content).get("method") == "tools/call":
            return httpx2.Response(
                200, headers={"content-type": "application/json"}, stream=_BrokenBody()
            )
        return response

    async def aclose(self) -> None:
        await self._inner.aclose()


# -- The factory ---------------------------------------------------------------------------------


class TestFactory:
    @pytest.mark.asyncio
    async def test_lights_build_over_the_sdk_transport_when_none_is_injected(
        self, monkeypatch, tmp_path: Path
    ):
        monkeypatch.setenv("MIST_EVAL_ISOLATION", "0")
        monkeypatch.delenv("MIST_HYDRATION_ISOLATION", raising=False)
        monkeypatch.delenv("MIST_ALLOWED_ORIGINS", raising=False)
        factories_built: list[Callable[[], object]] = []
        real_connection = factories.McpConnection

        def recording_connection(transport_factory, *args, **kwargs):
            factories_built.append(transport_factory)
            return real_connection(transport_factory, *args, **kwargs)

        monkeypatch.setattr(factories, "McpConnection", recording_connection)
        config = load_capabilities_config(
            {
                "MIST_CAPABILITIES_ENABLED": "true",
                "MIST_CAPABILITIES_DB_PATH": str(tmp_path / "capabilities.db"),
                "MIST_CAPABILITIES": "lights",
                "MIST_HA_MCP_URL": HA_URL,
            }
        )

        result = factories.build_capability_host(
            config,
            event_sink=RecordingEventSink(),
            secret_source=EnvSecretSource({"MIST_HA_TOKEN": TOKEN}, lambda path: ""),
            secret_registry=SecretRegistry(),
            clock=FakeClock(),
            embedder=None,
            token_counter=ApproximateTokenCounter(),
            transport_factory=None,
            power=FakePower(),
            tiers=FakeTiers(),
            register=NoAddress(),
            desktop_present=lambda: False,
        )
        try:
            assert result.host is not None
            reasons = {status.name: status.reason for status in result.host.statuses()}
            assert reasons["lights"] != "mcp package not installed"
            (make_transport,) = factories_built
            assert isinstance(make_transport(), McpSdkTransport)
        finally:
            await result.close()
