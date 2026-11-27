"""A fake Home Assistant MCP server, reached through an in-memory `McpTransport`.

It models the documented Home Assistant built-in intents (contributors.home-assistant.io, Built-in
intents) as MCP tools: `HassTurnOn` and `HassTurnOff` with the slots `name`, `area`, `floor`,
`domain` and `device_class`; `HassLightSet` with `name`, `area`, `floor`, `domain`, `brightness`
(0-100), `color` (a name) and `temperature` (1000-10000 K); `GetLiveContext` with no arguments;
and, when asked for, `HassGetState`, given the same target slots as `HassTurnOn` because the
design's external facts quote none for it.

A tools/list read from Home Assistant 2026.11 on 26 Nov (inside the backend container, on
`/api/mcp` and `/api/mcp/assist` alike) listed these tools with a source prefix:
`intent__HassTurnOn`, `intent__HassTurnOff`, `light__HassLightSet` and
`homeassistant__GetLiveContext`, among others; `prefixed=True` lists them that way. Its
`GetLiveContext` text arrived JSON-wrapped, `{"result": "Live Context: ..."}`. The input schemas
here and the source of `HassGetState` (`intent` assumed) were not read, so they are still
UNVERIFIED; tests built on this fake prove MIST's behaviour against the documented intents and
the 26 Nov names, not against a live server.

One `FakeHomeAssistant` is the server and keeps state across connections; `transport()` is the
`transport_factory` an `McpConnection` takes, and builds a fresh transport per connect attempt.
Every scripting knob is a public attribute a test sets directly.
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from backend.capabilities.mcp.transport import McpToolResult, McpToolSpec
from backend.errors import CapabilityUnavailableError, McpTransportError

TURN_ON = "HassTurnOn"
TURN_OFF = "HassTurnOff"
LIGHT_SET = "HassLightSet"
LIVE_CONTEXT = "GetLiveContext"
GET_STATE = "HassGetState"

SOURCES: dict[str, str] = {
    TURN_ON: "intent",
    TURN_OFF: "intent",
    LIGHT_SET: "light",
    LIVE_CONTEXT: "homeassistant",
    GET_STATE: "intent",
}
"""The source each tool is listed under with `prefixed=True`, as `<source>__<name>`."""


def prefixed_name(name: str) -> str:
    """Return the name Home Assistant 2026.11 lists a tool under, such as `intent__HassTurnOn`."""
    return f"{SOURCES[name]}__{name}"


_TARGET_SLOTS: dict[str, object] = {
    "name": {"type": "string"},
    "area": {"type": "string"},
    "floor": {"type": "string"},
    "domain": {"type": "array", "items": {"type": "string"}},
    "device_class": {"type": "array", "items": {"type": "string"}},
}

_LIGHT_SET_SLOTS: dict[str, object] = {
    "name": {"type": "string"},
    "area": {"type": "string"},
    "floor": {"type": "string"},
    "domain": {"type": "array", "items": {"type": "string"}},
    "brightness": {"type": "integer", "minimum": 0, "maximum": 100},
    "color": {"type": "string"},
    "temperature": {"type": "integer", "minimum": 1000, "maximum": 10000},
}

DEFAULT_LIVE_CONTEXT = (
    "Live Context: An overview of the areas and the devices in this smart home:\n"
    "- names: Bedroom Lamp\n"
    "  domain: light\n"
    "  state: 'off'\n"
    "  areas: Bedroom\n"
)
"""A small stand-in for `GetLiveContext` text, in the layout Home Assistant 2026.11 returned on
26 Nov inside its JSON `result`; this default is left unwrapped."""

_OK_TEXT = json.dumps({"success": True})


def _object_schema(properties: Mapping[str, object]) -> dict[str, object]:
    return {"type": "object", "properties": dict(properties)}


def connect_refused() -> McpTransportError:
    """The error a transport raises when Home Assistant cannot be reached."""
    return McpTransportError("connection refused")


def auth_rejected() -> CapabilityUnavailableError:
    """The error a transport raises on HTTP 401 or 403 (T16 maps the SDK's error to it)."""
    return CapabilityUnavailableError("auth rejected")


def tool_error(text: str = "Unable to find a device named Attic Lamp") -> McpToolResult:
    """A tool result Home Assistant marks as an error, carrying its text."""
    return McpToolResult(is_error=True, text=text, structured=None)


def unexpected_error() -> RuntimeError:
    """An exception outside the supervisor's contained set: a bug, which ends the supervisor."""
    return RuntimeError("unexpected fault in the transport")


@dataclass(frozen=True, slots=True)
class RecordedCall:
    """One `call_tool` the fake received, with the task that made it."""

    name: str
    arguments: Mapping[str, object]
    timeout_s: float
    task: asyncio.Task[object] | None


class FakeHomeAssistant:
    """The fake server: its tool list, its scripted failures and what it received.

    Attributes:
        include_get_state: List `HassGetState` too.
        omit: Bare tool names left out of `list_tools`, prefixed or not; a call to one is
            answered as an unknown tool.
        prefixed: List each tool as `<source>__<name>` (`SOURCES`), as Home Assistant 2026.11
            does; off by default, so the bare names stay the default listing.
        extra_tools: Further names listed as given, with the target slots, answered with ok;
            for listings such as two sources offering the same tool.
        open_failures: Each `open` pops one and raises it while any remain.
        list_tools_failures: Each `list_tools` pops one and raises it while any remain.
        call_script: Each `call_tool` pops one while any remain: a result is returned, an
            exception raised. Once empty, calls get the default result.
        open_gate: When set, `open` waits on it first, so a test can hold a connect open.
        call_gate: When set, `call_tool` waits on it, bounded by the call's `timeout_s`; a call
            that outlives it raises `McpTransportError`, as a slow server would.
        close_error: When set, `close` raises it after recording the call.
        live_context: The text `GetLiveContext` returns by default.
        open_tasks, list_tools_tasks, close_tasks: The task each such call ran in.
        calls: Every `call_tool`, in order, including ones that failed.
        transports: Every transport `transport()` built.
    """

    def __init__(
        self,
        *,
        include_get_state: bool = False,
        omit: Sequence[str] = (),
        prefixed: bool = False,
        extra_tools: Sequence[str] = (),
    ) -> None:
        self.include_get_state = include_get_state
        self.omit: set[str] = set(omit)
        self.prefixed = prefixed
        self.extra_tools: list[str] = list(extra_tools)
        self.open_failures: deque[BaseException] = deque()
        self.list_tools_failures: deque[BaseException] = deque()
        self.call_script: deque[McpToolResult | BaseException] = deque()
        self.open_gate: asyncio.Event | None = None
        self.call_gate: asyncio.Event | None = None
        self.close_error: BaseException | None = None
        self.live_context = DEFAULT_LIVE_CONTEXT
        self.open_tasks: list[asyncio.Task[object] | None] = []
        self.list_tools_tasks: list[asyncio.Task[object] | None] = []
        self.close_tasks: list[asyncio.Task[object] | None] = []
        self.calls: list[RecordedCall] = []
        self.transports: list[FakeHomeAssistantTransport] = []

    def transport(self) -> FakeHomeAssistantTransport:
        """Build a new transport to this server; pass the bound method as a transport factory."""
        transport = FakeHomeAssistantTransport(self)
        self.transports.append(transport)
        return transport

    def tool_specs(self) -> list[McpToolSpec]:
        """Return the tools the server lists now, after every listing attribute."""
        listed = [
            replace(spec, name=prefixed_name(spec.name)) if self.prefixed else spec
            for spec in self._bare_specs()
        ]
        listed.extend(
            McpToolSpec(name, "An extra tool listed as given", _object_schema(_TARGET_SLOTS))
            for name in self.extra_tools
        )
        return listed

    def _bare_specs(self) -> list[McpToolSpec]:
        specs = [
            McpToolSpec(
                TURN_ON, "Turns on/opens a device or entity", _object_schema(_TARGET_SLOTS)
            ),
            McpToolSpec(
                TURN_OFF, "Turns off/closes a device or entity", _object_schema(_TARGET_SLOTS)
            ),
            McpToolSpec(
                LIGHT_SET,
                "Sets the brightness, color or temperature of a light",
                _object_schema(_LIGHT_SET_SLOTS),
            ),
            McpToolSpec(
                LIVE_CONTEXT,
                "Provides real-time information about the state of devices",
                _object_schema({}),
            ),
        ]
        if self.include_get_state:
            specs.append(
                McpToolSpec(
                    GET_STATE, "Gets the state of a device or entity", _object_schema(_TARGET_SLOTS)
                )
            )
        return [spec for spec in specs if spec.name not in self.omit]

    def default_result(self, name: str) -> McpToolResult:
        """Return the answer to a call nothing scripted: ok, an unknown-tool error, or context."""
        if name not in {spec.name for spec in self.tool_specs()}:
            return tool_error(f"Unknown tool: {name}")
        if name in (LIVE_CONTEXT, prefixed_name(LIVE_CONTEXT)):
            return McpToolResult(is_error=False, text=self.live_context, structured=None)
        return McpToolResult(is_error=False, text=_OK_TEXT, structured={"success": True})


class FakeHomeAssistantTransport:
    """One connection to a `FakeHomeAssistant`, implementing `McpTransport`."""

    def __init__(self, server: FakeHomeAssistant) -> None:
        self._server = server
        self.is_open = False

    async def open(self) -> None:
        self._server.open_tasks.append(asyncio.current_task())
        if self._server.open_gate is not None:
            await self._server.open_gate.wait()
        if self._server.open_failures:
            raise self._server.open_failures.popleft()
        self.is_open = True

    async def list_tools(self) -> list[McpToolSpec]:
        self._server.list_tools_tasks.append(asyncio.current_task())
        self._require_open()
        if self._server.list_tools_failures:
            raise self._server.list_tools_failures.popleft()
        return self._server.tool_specs()

    async def call_tool(
        self, name: str, arguments: Mapping[str, object], timeout_s: float
    ) -> McpToolResult:
        self._server.calls.append(
            RecordedCall(name, dict(arguments), timeout_s, asyncio.current_task())
        )
        self._require_open()
        if self._server.call_gate is not None:
            try:
                await asyncio.wait_for(self._server.call_gate.wait(), timeout_s)
            except TimeoutError as exc:
                raise McpTransportError(f"{name} timed out after {timeout_s} s") from exc
        if self._server.call_script:
            scripted = self._server.call_script.popleft()
            if isinstance(scripted, BaseException):
                raise scripted
            return scripted
        return self._server.default_result(name)

    async def close(self) -> None:
        self._server.close_tasks.append(asyncio.current_task())
        self.is_open = False
        if self._server.close_error is not None:
            raise self._server.close_error

    def _require_open(self) -> None:
        if not self.is_open:
            raise McpTransportError("the MCP session is not open")
