"""`LightsCapability`: MIST's lights contract, over any provider of the `LightsProvider` shape.

The tool contract (`lights.power`, `lights.adjust`, `lights.status`), its argument rules, the
fast path and the spoken summaries belong to MIST rather than to Home Assistant, so a provider
only maps resolved calls onto its own API. The provider owns the target catalog, because only it
can read one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.errors import (
    CapabilityInvocationError,
    CapabilityUnavailableError,
    McpTransportError,
)

from ..contracts import (
    Capability,
    CapabilityStatus,
    FastIntent,
    InvocationContext,
    RoutingHints,
    ToolOutcome,
    ToolSpec,
)
from ..replies import ok_summary, render_read
from ..router import count_catalog_tokens
from ..schema import to_args_schema, to_catalog_line
from .catalog import LightCatalog, LightEntity, LightTarget
from .intents import match_lights_intent

if TYPE_CHECKING:
    from backend.chat.context_budget import TokenCounter

POWER = "lights.power"
ADJUST = "lights.adjust"
STATUS = "lights.status"

REFINE_ALLOWANCE_TOKENS = 200
"""The most tokens the target enum may add to the lights catalog lines."""

KNOWN_TARGETS_NAMED = 10
"""How many known targets an unknown-target answer names; the rest are counted."""

_ENUM_TOOLS = (POWER, ADJUST)
# The user's words for these name a light already ("the bedroom lamp"), so a summary does not
# add "lights" after them.
_LIGHT_WORDS = frozenset({"light", "lights", "lamp", "lamps", "bulb", "bulbs", "strip", "strips"})
_NO_TARGET = "Say which light or room."

_Args = TypeVar("_Args", bound=BaseModel)


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    return " ".join(value.split()) or None


class PowerArgs(BaseModel):
    """Arguments of `lights.power`."""

    model_config = ConfigDict(extra="forbid")

    target: str | None = None
    on: bool

    @field_validator("target")
    @classmethod
    def clean_target(cls, value: str | None) -> str | None:
        """Collapse spacing; a blank target is no target."""
        return _clean(value)


class AdjustArgs(BaseModel):
    """Arguments of `lights.adjust`; at least one setting is required."""

    model_config = ConfigDict(extra="forbid")

    target: str | None = None
    brightness_pct: int | None = Field(default=None, ge=0, le=100)
    color: str | None = None
    kelvin: int | None = Field(default=None, ge=1000, le=10000)

    @field_validator("target", "color")
    @classmethod
    def clean_text(cls, value: str | None) -> str | None:
        """Collapse spacing; a blank value is no value."""
        return _clean(value)

    @model_validator(mode="after")
    def require_a_setting(self) -> AdjustArgs:
        """Refuse a call that would change nothing; the grammar cannot express this rule."""
        if self.brightness_pct is None and self.color is None and self.kelvin is None:
            raise ValueError("give at least one of brightness_pct, color and kelvin")
        return self


class StatusArgs(BaseModel):
    """Arguments of `lights.status`; no target means every light."""

    model_config = ConfigDict(extra="forbid")

    target: str | None = None

    @field_validator("target")
    @classmethod
    def clean_target(cls, value: str | None) -> str | None:
        """Collapse spacing; a blank target is no target."""
        return _clean(value)


@dataclass(frozen=True, slots=True)
class LightAdjustment:
    """The settings one `lights.adjust` call changes; None leaves a setting as it is."""

    brightness_pct: int | None = None
    color: str | None = None
    kelvin: int | None = None


@dataclass(frozen=True, slots=True)
class StatusReading:
    """What a provider read for `lights.status`.

    Attributes:
        text: The provider's own text, capped for the model.
        lights: The lights the target covers, or None when the provider's answer does not list
            lights individually.
        truncated: The provider's text was longer than the cap.
    """

    text: str
    lights: tuple[LightEntity, ...] | None
    truncated: bool


class LightsProvider(Protocol):
    """What `LightsCapability` needs from a lights backend.

    The calls raise `CapabilityUnavailableError` or `McpTransportError` when the backend cannot
    be reached, and `CapabilityInvocationError` carrying the backend's text when it refuses a
    call.
    """

    async def start(self) -> None:
        """Start connecting; returns without waiting for a connection."""
        ...

    async def stop(self) -> None:
        """Stop and release the connection."""
        ...

    def status(self) -> CapabilityStatus:
        """Return the lights status from in-memory state, named `lights`."""
        ...

    def catalog(self) -> LightCatalog:
        """Return the catalog from the last refresh; empty before the first."""
        ...

    def resolve(self, target: str) -> LightTarget | None:
        """Match a free-string target to the catalog, ignoring case; None when unknown."""
        ...

    def provider_tools(self) -> tuple[str, ...]:
        """Return the backend's own tool names, sorted; empty while disconnected."""
        ...

    async def power(self, target: LightTarget, on: bool) -> None:
        """Switch the target's lights on or off."""
        ...

    async def adjust(self, target: LightTarget, adjustment: LightAdjustment) -> None:
        """Change the target's brightness, color or color temperature."""
        ...

    async def read_status(self, target: LightTarget | None) -> StatusReading:
        """Read the state of the target's lights, or of every light for None."""
        ...


class LightsCapability(Capability):
    """Switches, dims and reports lights through a `LightsProvider`.

    No `is_explicit` hook exists: under the default policy every lights call the model chooses
    is confirmed, and only the anchored fast-path grammar runs one directly.
    """

    name = "lights"
    description = "Switch, dim and report the lights"
    refine_allowance_tokens = REFINE_ALLOWANCE_TOKENS

    def __init__(
        self, provider: LightsProvider, *, default_target: str | None, counter: TokenCounter
    ) -> None:
        """Build the capability; nothing connects until `start`.

        Args:
            provider: The lights backend, such as `HomeAssistantLightsProvider`.
            default_target: `MIST_LIGHTS_DEFAULT_TARGET`, the light or area a power or adjust
                call acts on when it names none; None makes such a call `invalid_args`.
            counter: The router's token counter, so the enum check here measures as the host's
                refinement check does.
        """
        self._provider = provider
        self._default_target = _clean(default_target)
        self._counter = counter
        # The catalog targets last measured and whether their enum fitted the allowance.
        self._enum_fit: tuple[tuple[str, ...], bool] = ((), False)
        self._tools = (
            ToolSpec(
                name=POWER,
                description="turn lights on or off",
                args_model=PowerArgs,
                risk="physical",
                default_policy="confirm_inferred",
                summarize=self._summarize_power,
                ack="Switching the lights.",
                view="lights",
            ),
            ToolSpec(
                name=ADJUST,
                description="set brightness, color or color temperature",
                args_model=AdjustArgs,
                risk="physical",
                default_policy="confirm_inferred",
                summarize=self._summarize_adjust,
                ack="Adjusting the lights.",
                view="lights",
            ),
            ToolSpec(
                name=STATUS,
                description="report which lights are on",
                args_model=StatusArgs,
                risk="read",
                default_policy="allow",
                summarize=self._summarize_status,
                ack="Checking the lights.",
                view="lights",
            ),
        )

    def tools(self) -> Sequence[ToolSpec]:
        """Return `lights.power`, `lights.adjust` and `lights.status`."""
        return self._tools

    def routing_hints(self) -> RoutingHints:
        """Return no keywords beyond the router's lights floor."""
        return RoutingHints(keywords=())

    def dynamic_keywords(self) -> tuple[str, ...]:
        """Return the catalog's light names and areas, so "the bedroom" routes to lights."""
        return self._provider.catalog().targets()

    async def start(self) -> None:
        """Start the provider, which connects in the background."""
        await self._provider.start()

    async def stop(self) -> None:
        """Stop the provider."""
        await self._provider.stop()

    def status(self) -> CapabilityStatus:
        """Return the provider's status."""
        return self._provider.status()

    def provider_tools(self) -> tuple[str, ...]:
        """Return the provider's own tool names."""
        return self._provider.provider_tools()

    def match_intent(self, text: str) -> FastIntent | None:
        """Match the lights fast-path rows against the current catalog's targets."""
        return match_lights_intent(text, self._provider.catalog().targets())

    def refine_schema(self, tool: str, schema: dict[str, object]) -> dict[str, object]:
        """Add the sorted target enum to `lights.power` and `lights.adjust` while it fits.

        The enum goes in only while both tools' lines with it stay within the static lines plus
        `refine_allowance_tokens`; past that `target` stays a free string, which the provider
        resolves. `lights.status` always keeps a free string.
        """
        if tool not in _ENUM_TOOLS:
            return schema
        targets = self._provider.catalog().targets()
        if not targets or not self._enum_fits(targets):
            return schema
        return _with_target_enum(schema, targets)

    async def invoke(self, tool: str, args: BaseModel, ctx: InvocationContext) -> ToolOutcome:
        """Resolve the target, call the provider and render the outcome.

        Raises:
            ValueError: `tool` is not one of this capability's tools.
        """
        try:
            if isinstance(args, PowerArgs) and tool == POWER:
                return await self._power(args)
            if isinstance(args, AdjustArgs) and tool == ADJUST:
                return await self._adjust(args)
            if isinstance(args, StatusArgs) and tool == STATUS:
                return await self._status(args)
        except (CapabilityUnavailableError, McpTransportError) as exc:
            return ToolOutcome(
                status="unavailable",
                summary="The lights can't be reached right now.",
                reason=str(exc) or type(exc).__name__,
            )
        except CapabilityInvocationError as exc:
            return ToolOutcome(
                status="error",
                summary="The lights did not respond to that.",
                reason=str(exc) or type(exc).__name__,
            )
        raise ValueError(f"lights has no tool {tool!r} taking {type(args).__name__}")

    # -- Tools -----------------------------------------------------------------------------

    async def _power(self, args: PowerArgs) -> ToolOutcome:
        target = self._resolve(args.target or self._default_target)
        if isinstance(target, ToolOutcome):
            return target
        await self._provider.power(target, args.on)
        state = "on" if args.on else "off"
        if _ends_in_light_word(target.value):
            summary = f"{_capitalised(target.value)} {state}."
        else:
            summary = ok_summary("lights.power", target=target.value, state=state)
        return ToolOutcome(
            status="ok", summary=summary, data={"target": target.value, "on": args.on}
        )

    async def _adjust(self, args: AdjustArgs) -> ToolOutcome:
        target = self._resolve(args.target or self._default_target)
        if isinstance(target, ToolOutcome):
            return target
        adjustment = LightAdjustment(
            brightness_pct=args.brightness_pct, color=args.color, kelvin=args.kelvin
        )
        await self._provider.adjust(target, adjustment)
        if args.color is None and args.kelvin is None:
            summary = ok_summary("lights.adjust", target=target.value, percent=args.brightness_pct)
        else:
            summary = f"{_capitalised(_lights_phrase(target.value))} set to {_settings(args)}."
        data: dict[str, object] = {"target": target.value}
        data.update(
            (key, value)
            for key, value in (
                ("brightness_pct", args.brightness_pct),
                ("color", args.color),
                ("kelvin", args.kelvin),
            )
            if value is not None
        )
        return ToolOutcome(status="ok", summary=summary, data=data)

    async def _status(self, args: StatusArgs) -> ToolOutcome:
        target: LightTarget | None = None
        if args.target is not None:
            resolved = self._resolve(args.target)
            if isinstance(resolved, ToolOutcome):
                return resolved
            target = resolved
        reading = await self._provider.read_status(target)
        data = {"text": reading.text, "truncated": reading.truncated}
        if reading.lights is None:
            return ToolOutcome(status="ok", summary="Here is the latest light status.", data=data)
        items = [f"{light.names[0]} is {light.state or 'unknown'}" for light in reading.lights]
        return ToolOutcome(status="ok", summary=render_read(STATUS, items), data=data)

    def _resolve(self, text: str | None) -> LightTarget | ToolOutcome:
        """Resolve a target, or return the `invalid_args` outcome that answers instead."""
        if text is None:
            return ToolOutcome(status="invalid_args", summary=_NO_TARGET, reason="no target")
        target = self._provider.resolve(text)
        if target is not None:
            return target
        known = self._provider.catalog().targets()
        named = ", ".join(known[:KNOWN_TARGETS_NAMED])
        if len(known) > KNOWN_TARGETS_NAMED:
            named += f" and {len(known) - KNOWN_TARGETS_NAMED} more"
        return ToolOutcome(
            status="invalid_args",
            summary=f"There is no light or room called {text}; known ones are {named}.",
            reason="unknown target",
            data={"known_targets": list(known[:KNOWN_TARGETS_NAMED])},
        )

    # -- Summaries for proposals -----------------------------------------------------------

    def _target_phrase(self, target: str | None) -> str:
        chosen = target or self._default_target
        return f"the {_lights_phrase(chosen)}" if chosen else "the lights"

    def _summarize_power(self, args: BaseModel) -> str:
        power = _expect(args, PowerArgs)
        return f"turn {'on' if power.on else 'off'} {self._target_phrase(power.target)}"

    def _summarize_adjust(self, args: BaseModel) -> str:
        adjust = _expect(args, AdjustArgs)
        return f"set {self._target_phrase(adjust.target)} to {_settings(adjust)}"

    def _summarize_status(self, args: BaseModel) -> str:
        target = _expect(args, StatusArgs).target
        return f"check the {_lights_phrase(target)}" if target else "check the lights"

    # -- Enum allowance --------------------------------------------------------------------

    def _enum_fits(self, targets: tuple[str, ...]) -> bool:
        cached_targets, fits = self._enum_fit
        if cached_targets == targets:
            return fits

        def with_enum(tool: str, schema: dict[str, object]) -> dict[str, object]:
            return _with_target_enum(schema, targets) if tool in _ENUM_TOOLS else schema

        static = [to_catalog_line(spec, to_args_schema(spec)) for spec in self._tools]
        refined = [to_catalog_line(spec, to_args_schema(spec, with_enum)) for spec in self._tools]
        added = count_catalog_tokens(refined, self._counter) - count_catalog_tokens(
            static, self._counter
        )
        fits = added <= self.refine_allowance_tokens
        self._enum_fit = (targets, fits)
        return fits


def _expect(args: BaseModel, model: type[_Args]) -> _Args:
    # Each spec pairs its summarize hook with its own args model, so a mismatch is a bug.
    if not isinstance(args, model):
        raise TypeError(f"expected {model.__name__}, got {type(args).__name__}")
    return args


def _with_target_enum(schema: dict[str, object], targets: tuple[str, ...]) -> dict[str, object]:
    properties = schema.get("properties")
    if not isinstance(properties, dict) or not isinstance(properties.get("target"), dict):
        return schema
    target = {**properties["target"], "enum": sorted(targets)}
    return {**schema, "properties": {**properties, "target": target}}


def _ends_in_light_word(target: str) -> bool:
    words = target.casefold().split()
    return bool(words) and words[-1] in _LIGHT_WORDS


def _lights_phrase(target: str) -> str:
    return target if _ends_in_light_word(target) else f"{target} lights"


def _settings(args: AdjustArgs) -> str:
    parts = []
    if args.brightness_pct is not None:
        parts.append(f"{args.brightness_pct} percent")
    if args.color is not None:
        parts.append(args.color)
    if args.kelvin is not None:
        parts.append(f"{args.kelvin} kelvin")
    if len(parts) == 1:
        return parts[0]
    return f"{', '.join(parts[:-1])} and {parts[-1]}"


def _capitalised(text: str) -> str:
    return text[:1].upper() + text[1:]
