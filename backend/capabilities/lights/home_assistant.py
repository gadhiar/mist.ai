"""`HomeAssistantLightsProvider`: MIST's lights calls mapped onto Home Assistant's MCP tools.

The mapping follows Home Assistant's documented built-in intents: power goes to `HassTurnOn` and
`HassTurnOff`, adjust to `HassLightSet`, both with a `name` or `area` slot and a `domain` slot
listing light, and status to `GetLiveContext`, or `HassGetState` when that is not listed.
Whether the live MCP server lists exactly these tools under the Assist API is unverified, so the
provider checks the listed tools and reports a missing one by name in its status.

The target catalog is refreshed on connect, every `CATALOG_REFRESH_S` and on every status read.
The refresh runs in a task of its own, outside any turn, so its call is bounded by the same
`call_timeout_s` as every other call, and the task is supervised the way the MCP connection's
is: an expected failure is logged and the loop carries on; anything else ends it, is logged at
ERROR, and leaves lights `degraded` with the reason.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from datetime import timedelta

from backend.errors import (
    CapabilityInvocationError,
    CapabilityUnavailableError,
    McpTransportError,
    MistError,
)

from ..clock import Clock
from ..contracts import CapabilityState, CapabilityStatus
from ..mcp.connection import McpConnection
from ..mcp.transport import McpToolResult
from .capability import LightAdjustment, StatusReading
from .catalog import (
    EMPTY_CATALOG,
    LightCatalog,
    LightEntity,
    LightTarget,
    live_context_body,
    parse_live_context,
)

logger = logging.getLogger(__name__)

TURN_ON = "HassTurnOn"
TURN_OFF = "HassTurnOff"
LIGHT_SET = "HassLightSet"
LIVE_CONTEXT = "GetLiveContext"
GET_STATE = "HassGetState"

CATALOG_REFRESH_S = 600.0
"""How often the catalog is read again while connected."""

STATUS_TEXT_MAX_CHARS = 1500
"""The most of a status tool's text that reaches the model."""

_REFRESH_INTERVAL = timedelta(seconds=CATALOG_REFRESH_S)
_POLL_S = 1.0
# The failures one refresh contains, as the connection's supervisor does; anything else is a
# bug and ends the refresh task (`_on_refresh_done`).
_CONTAINED_ERRORS = (MistError, OSError)
_NAME = "lights"
_DOMAIN = ["light"]


def _describe(exc: BaseException) -> str:
    return str(exc) or type(exc).__name__


class HomeAssistantLightsProvider:
    """A `LightsProvider` over one `McpConnection` to Home Assistant's MCP server.

    It owns the connection: `start` and `stop` start and stop it.
    """

    def __init__(
        self,
        connection: McpConnection,
        clock: Clock,
        *,
        call_timeout_s: float,
        stop_timeout_s: float = 5.0,
    ) -> None:
        """Build a stopped provider.

        Args:
            connection: The supervised connection to Home Assistant, not yet started.
            clock: Paces the catalog refresh and stamps statuses.
            call_timeout_s: Bounds every tool call, the catalog refresh included
                (`MIST_CAPABILITY_TOOL_TIMEOUT_S`, which also bounds a call in the host).
            stop_timeout_s: How long `stop` waits, in real loop time, for the refresh task.

        Raises:
            ValueError: A timeout is not finite or not above 0.
        """
        for label, value in (
            ("call_timeout_s", call_timeout_s),
            ("stop_timeout_s", stop_timeout_s),
        ):
            if not 0 < value < float("inf"):
                raise ValueError(f"{label} must be finite and above 0, not {value}")
        self._connection = connection
        self._clock = clock
        self._call_timeout_s = call_timeout_s
        self._stop_timeout_s = stop_timeout_s
        self._catalog = EMPTY_CATALOG
        self._task: asyncio.Task[None] | None = None
        self._stopping = False
        self._refresh_failure: str | None = None
        # Set when a call saw the session drop, so the next poll refreshes even when the
        # supervisor reconnected before the poll could see the connection down.
        self._stale = True
        self._next_refresh = clock.now()

    # -- Lifecycle -------------------------------------------------------------------------

    async def start(self) -> None:
        """Start the connection and the refresh task; returns without waiting to connect."""
        if self._task is not None:
            return
        await self._connection.start()
        self._task = asyncio.create_task(self._refresh_loop(), name="lights-catalog-refresh")
        self._task.add_done_callback(self._on_refresh_done)

    async def stop(self) -> None:
        """Stop the refresh task, waiting at most `stop_timeout_s`, then the connection."""
        self._stopping = True
        task = self._task
        if task is not None and not task.done():
            task.cancel()
            done, _ = await asyncio.wait({task}, timeout=self._stop_timeout_s)
            if not done:
                logger.warning(
                    "Lights catalog refresh still running %g s after stop; leaving it to end",
                    self._stop_timeout_s,
                )
        await self._connection.stop()

    # -- State -----------------------------------------------------------------------------

    def status(self) -> CapabilityStatus:
        """Return the lights status from the connection and the listed tools.

        While the connection is down this is its status under the name lights, `task_dead`
        included. Once it is up, a missing `HassTurnOn` or `HassTurnOff` makes lights
        `unavailable`; a missing `HassLightSet`, both status tools missing, or a dead refresh
        task makes them `degraded`. The reason names each missing tool.
        """
        connection = self._connection.state()
        if connection.state != "available":
            return CapabilityStatus(
                name=_NAME,
                state="unavailable",
                reason=connection.reason,
                checked_at=connection.checked_at,
                task_dead=connection.task_dead,
            )
        listed = self._listed()
        missing = [tool for tool in (TURN_ON, TURN_OFF) if tool not in listed]
        if missing:
            return self._status("unavailable", f"Home Assistant lacks {' and '.join(missing)}")
        problems = []
        lacking = [] if LIGHT_SET in listed else [LIGHT_SET]
        if LIVE_CONTEXT not in listed and GET_STATE not in listed:
            lacking.extend((LIVE_CONTEXT, GET_STATE))
        if lacking:
            problems.append(f"Home Assistant lacks {' and '.join(lacking)}")
        if self._refresh_failure is not None:
            problems.append(self._refresh_failure)
        if problems:
            return self._status("degraded", "; ".join(problems))
        return self._status("available", None)

    def catalog(self) -> LightCatalog:
        """Return the catalog from the last refresh, kept while disconnected."""
        return self._catalog

    def resolve(self, target: str) -> LightTarget | None:
        """Match `target` to the catalog, ignoring case.

        With no catalog, as when `GetLiveContext` is not listed, MIST cannot tell a light
        from an area or a known target from an unknown one, so the words pass through as a
        `name` and Home Assistant's own answer decides.
        """
        if self._catalog.is_empty:
            return LightTarget(kind="name", value=" ".join(target.split()))
        return self._catalog.resolve(target)

    def provider_tools(self) -> tuple[str, ...]:
        """Return the sorted tool names of the last `list_tools`, or `()` while disconnected."""
        if self._connection.state().state != "available":
            return ()
        return tuple(sorted(self._listed()))

    # -- Calls -----------------------------------------------------------------------------

    async def power(self, target: LightTarget, on: bool) -> None:
        """Call `HassTurnOn` or `HassTurnOff` for the target's lights."""
        await self._call(TURN_ON if on else TURN_OFF, _target_slots(target))

    async def adjust(self, target: LightTarget, adjustment: LightAdjustment) -> None:
        """Call `HassLightSet` with the settings given.

        Raises:
            CapabilityUnavailableError: Home Assistant does not list `HassLightSet`.
        """
        if LIGHT_SET not in self._listed():
            raise CapabilityUnavailableError(f"Home Assistant lacks {LIGHT_SET}")
        slots = _target_slots(target)
        for slot, value in (
            ("brightness", adjustment.brightness_pct),
            ("color", adjustment.color),
            ("temperature", adjustment.kelvin),
        ):
            if value is not None:
                slots[slot] = value
        await self._call(LIGHT_SET, slots)

    async def read_status(self, target: LightTarget | None) -> StatusReading:
        """Read `GetLiveContext`, refreshing the catalog from it, or else `HassGetState`.

        Raises:
            CapabilityUnavailableError: Home Assistant lists neither status tool.
        """
        listed = self._listed()
        if LIVE_CONTEXT in listed:
            result = await self._call(LIVE_CONTEXT, {})
            self._catalog = parse_live_context(result.text)
            self._next_refresh = self._clock.now() + _REFRESH_INTERVAL
            lights = self._catalog.lights_for(target) if not self._catalog.is_empty else None
            return _reading(live_context_body(result.text), lights)
        if GET_STATE in listed:
            slots = _target_slots(target) if target is not None else {"domain": list(_DOMAIN)}
            result = await self._call(GET_STATE, slots)
            return _reading(result.text, None)
        raise CapabilityUnavailableError(f"Home Assistant lacks {LIVE_CONTEXT} and {GET_STATE}")

    async def _call(self, tool: str, arguments: Mapping[str, object]) -> McpToolResult:
        """Call one tool, bounded by `call_timeout_s`; a tool error raises with its text.

        Raises:
            CapabilityInvocationError: Home Assistant marked the result as an error.
            CapabilityUnavailableError, McpTransportError: From the connection.
        """
        try:
            result = await self._connection.call_tool(
                tool, arguments, timeout_s=self._call_timeout_s
            )
        except (CapabilityUnavailableError, McpTransportError):
            self._stale = True
            raise
        if result.is_error:
            raise CapabilityInvocationError(result.text.strip() or f"{tool} failed")
        return result

    def _listed(self) -> frozenset[str]:
        return frozenset(tool.name for tool in self._connection.tools())

    def _status(self, state: CapabilityState, reason: str | None) -> CapabilityStatus:
        return CapabilityStatus(
            name=_NAME, state=state, reason=reason, checked_at=self._clock.now()
        )

    # -- Catalog refresh -------------------------------------------------------------------

    async def _refresh_loop(self) -> None:
        connected = False
        while not self._stopping:
            if self._connection.state().state == "available":
                due = not connected or self._stale or self._refresh_due()
                connected = True
                if due:
                    await self._refresh()
            else:
                connected = False
            if self._stopping:
                break
            # Outside the contained `try` in `_refresh`, as the connection's backoff sleep is.
            await self._clock.sleep(_POLL_S)

    def _refresh_due(self) -> bool:
        return self._clock.now() >= self._next_refresh

    async def _refresh(self) -> None:
        self._stale = False
        self._next_refresh = self._clock.now() + _REFRESH_INTERVAL
        if LIVE_CONTEXT not in self._listed():
            return
        try:
            result = await self._call(LIVE_CONTEXT, {})
        except _CONTAINED_ERRORS as exc:
            logger.warning("Lights catalog refresh failed: %s", _describe(exc))
            return
        self._catalog = parse_live_context(result.text)
        logger.info("Lights catalog refreshed with %d lights", len(self._catalog.lights))

    def _on_refresh_done(self, task: asyncio.Task[None]) -> None:
        """Record an end of the refresh task that `stop` did not ask for."""
        if self._stopping:
            return
        if task.cancelled():
            reason = "lights catalog refresh was cancelled without stop()"
            logger.error("%s", reason)
        else:
            exc = task.exception()
            if exc is None:
                reason = "lights catalog refresh returned, which it never does by design"
                logger.error("%s", reason)
            else:
                reason = f"lights catalog refresh died: {type(exc).__name__}: {exc}"
                logger.error("%s", reason, exc_info=exc)
        self._refresh_failure = reason


def _target_slots(target: LightTarget) -> dict[str, object]:
    return {target.kind: target.value, "domain": list(_DOMAIN)}


def _reading(text: str, lights: tuple[LightEntity, ...] | None) -> StatusReading:
    return StatusReading(
        text=text[:STATUS_TEXT_MAX_CHARS],
        lights=lights,
        truncated=len(text) > STATUS_TEXT_MAX_CHARS,
    )
