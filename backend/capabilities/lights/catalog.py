"""The catalog of light targets (light names and areas), parsed from `GetLiveContext` text.

The catalog is what lets a short utterance name a light: the fast path matches "turn off the
bedroom" only when "bedroom" is a known target, the router routes on target names, and the
decode grammar can offer the targets as an enum.

The layout parsed here is the one the fake Home Assistant models: a header line, then a YAML
list with one mapping per exposed entity, carrying `names` (aliases joined by commas), `domain`,
`state` and `areas`; the whole text may also arrive inside a JSON object's `result`. Home
Assistant's real `GetLiveContext` output has not been read from a running server, so the parser
is defensive: any text it cannot read gives an empty catalog rather than an exception, and an
empty catalog only costs the enum, the dynamic keywords and the target-only fast path.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

import yaml

from ..intents import normalize_utterance

_LIGHT_DOMAIN = "light"
_LIST_ITEM = "- "


@dataclass(frozen=True, slots=True)
class LightTarget:
    """A target resolved against the catalog: a light by name, or every light in an area.

    Attributes:
        kind: Which Home Assistant slot carries `value`.
        value: The catalog's spelling, or the user's words when there is no catalog.
    """

    kind: Literal["name", "area"]
    value: str


@dataclass(frozen=True, slots=True)
class LightEntity:
    """One exposed light.

    Attributes:
        names: Its name first, then any aliases.
        areas: The areas it is in; empty when it has none.
        state: Its state as Home Assistant reported it ("on", "off", "unavailable"), or None.
    """

    names: tuple[str, ...]
    areas: tuple[str, ...]
    state: str | None


@dataclass(frozen=True, slots=True)
class LightCatalog:
    """The lights Home Assistant exposed at the last refresh."""

    lights: tuple[LightEntity, ...] = ()

    @property
    def is_empty(self) -> bool:
        """True when no light is known, as before the first refresh."""
        return not self.lights

    @property
    def names(self) -> tuple[str, ...]:
        """Every light name and alias, sorted ignoring case, without case duplicates."""
        return _unique(name for light in self.lights for name in light.names)

    @property
    def areas(self) -> tuple[str, ...]:
        """Every area holding a light, sorted ignoring case, without case duplicates."""
        return _unique(area for light in self.lights for area in light.areas)

    def targets(self) -> tuple[str, ...]:
        """Return the names and areas together, sorted ignoring case, without case duplicates."""
        return _unique((*self.names, *self.areas))

    def resolve(self, text: str) -> LightTarget | None:
        """Match `text` to a light name, else an area, ignoring case, spacing and punctuation.

        A name wins over an area spelled the same, because it is the narrower action.
        """
        key = _key(text)
        for name in self.names:
            if _key(name) == key:
                return LightTarget(kind="name", value=name)
        for area in self.areas:
            if _key(area) == key:
                return LightTarget(kind="area", value=area)
        return None

    def lights_for(self, target: LightTarget | None) -> tuple[LightEntity, ...]:
        """Return the lights a target covers; every light for None."""
        if target is None:
            return self.lights
        key = _key(target.value)
        field = "names" if target.kind == "name" else "areas"
        return tuple(
            light
            for light in self.lights
            if any(_key(value) == key for value in getattr(light, field))
        )


EMPTY_CATALOG = LightCatalog()


def live_context_body(text: str) -> str:
    """Return the context text itself, unwrapped from a JSON `result` when it has one."""
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            decoded = json.loads(stripped)
        except ValueError:
            return stripped
        if isinstance(decoded, Mapping) and isinstance(decoded.get("result"), str):
            return decoded["result"].strip()
    return stripped


def parse_live_context(text: str) -> LightCatalog:
    """Parse the lights out of `GetLiveContext` text (module docstring for the layout).

    Returns:
        The lights found, in the text's order; `EMPTY_CATALOG` when the text cannot be read.
        Entities of other domains are left out. Never raises on bad input.
    """
    body = live_context_body(text)
    lines = body.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith(_LIST_ITEM)), None)
    if start is None:
        return EMPTY_CATALOG
    try:
        entries = yaml.safe_load("\n".join(lines[start:]))
    except yaml.YAMLError:
        return EMPTY_CATALOG
    if not isinstance(entries, list):
        return EMPTY_CATALOG
    lights = tuple(light for light in (_entity(entry) for entry in entries) if light is not None)
    return LightCatalog(lights=lights)


def _entity(entry: object) -> LightEntity | None:
    if not isinstance(entry, Mapping) or entry.get("domain") != _LIGHT_DOMAIN:
        return None
    names = _phrases(entry.get("names"))
    if not names:
        return None
    return LightEntity(names=names, areas=_phrases(entry.get("areas")), state=_state(entry))


def _phrases(value: object) -> tuple[str, ...]:
    """Read a comma-joined string or a list of strings; anything else reads as none."""
    if isinstance(value, str):
        parts: Iterable[object] = value.split(",")
    elif isinstance(value, list):
        parts = value
    else:
        return ()
    return tuple(" ".join(part.split()) for part in parts if isinstance(part, str) and part.strip())


def _state(entry: Mapping[object, object]) -> str | None:
    state = entry.get("state")
    # YAML reads an unquoted `on` or `off` as a boolean.
    if isinstance(state, bool):
        return "on" if state else "off"
    if isinstance(state, str) and state.strip():
        return state.strip()
    return None


def _key(text: str) -> str:
    # The fast path hands over targets in `normalize_utterance` form ("kids lamp" for "Kid's
    # Lamp"), so the catalog compares in that form too.
    return normalize_utterance(text, ())


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    seen: dict[str, str] = {}
    for value in values:
        seen.setdefault(_key(value), value)
    return tuple(sorted(seen.values(), key=lambda value: (_key(value), value)))
