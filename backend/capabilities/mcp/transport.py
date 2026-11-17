"""The protocol an MCP transport implements, and the plain types it returns.

A transport is one session with one MCP server. Keeping it behind a protocol lets the connection
supervisor and the capabilities above it run against an in-memory fake, and keeps the `mcp` SDK
out of every module but the SDK transport itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class McpToolSpec:
    """One tool the server lists.

    Attributes:
        name: The server's tool name, for example `"HassTurnOn"`.
        description: The server's description of the tool.
        input_schema: The tool's JSON Schema for its arguments, as the server sent it.
    """

    name: str
    description: str
    input_schema: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class McpToolResult:
    """The server's answer to one tool call.

    Attributes:
        is_error: The server marked the result as a tool error; `text` carries its message.
        text: The result's text content, joined.
        structured: The result's structured content, when the server sent any.
    """

    is_error: bool
    text: str
    structured: Mapping[str, object] | None


class McpTransport(Protocol):
    """One session with an MCP server.

    `open` and `close` are called only from the one task that supervises the session, because
    an SDK-backed session holds anyio cancel scopes, which must be entered and exited in the
    same task. `call_tool` may be called from any task while the session is open.

    A transport raises `McpTransportError` for connect, session and call I/O failures, and
    `CapabilityUnavailableError` with the reason "auth rejected" when the server refuses its
    credentials, from `open`, `list_tools` or `call_tool`; the supervisor backs off from that
    for longer.

    An `open` that raises or is cancelled releases whatever it acquired, such as an SDK client
    context it entered, before the exception propagates. The supervisor calls `close` only on a
    transport whose `open` returned, so nothing else would release it.
    """

    async def open(self) -> None:
        """Connect and initialise the session."""
        ...

    async def list_tools(self) -> list[McpToolSpec]:
        """Return the tools the server lists now."""
        ...

    async def call_tool(
        self, name: str, arguments: Mapping[str, object], timeout_s: float
    ) -> McpToolResult:
        """Call one tool, failing with `McpTransportError` if it outlives `timeout_s`."""
        ...

    async def close(self) -> None:
        """End the session and release its resources."""
        ...
