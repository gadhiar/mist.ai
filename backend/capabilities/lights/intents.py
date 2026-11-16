"""The lights rows of the fast-path grammar, over `normalize_utterance` output.

An anchored grammar is the one deterministic test of an explicit command, so a simple "lights
off" runs without the model. The rows (build plan, Fast-path intents; `[]` optional, `|`
alternatives, LIGHT is light, lights, lamp or lamps):

- (turn|switch) (on|off) [the] [<target>] LIGHT
- (turn|switch) (on|off) [the] LIGHT in [the] <target>
- [(turn|switch)] [the] [<target>] LIGHT (on|off)
- (turn|switch) (on|off) [the] <known target>
- (set|dim) [the] [<target>] LIGHT to <N> percent
- (set|dim) [the] <known target> to <N> percent

Slot rules:

- A `<target>` is 1 to 4 words and never holds a slot stop word (`SLOT_STOP_WORDS`), so "turn
  off the lights and play jazz" goes to the decode as a compound request.
- Next to a light word a `<target>` may be words the catalog lacks; the capability then answers
  `invalid_args` naming known targets. It may not hold one of the grammar's own words ("the",
  "turn", "on" and the like), which is what keeps questions and reports such as "should I turn
  the lights off" or "are the lights on" out.
- A `<known target>` stands without a light word, so it must name a catalog target, ignoring
  case: with no catalog it matches nothing, and "turn off the music" reaches lights only when a
  light or area is called "music".
- When the target and the light word together name a catalog light ("bedroom lamp"), the
  target keeps the light word, so the call acts on that light rather than its whole area.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from ..contracts import FastIntent
from ..intents import normalize_utterance

SLOT_STOP_WORDS: frozenset[str] = frozenset({"and", "then", "also", "with"})
"""Words a `<target>` never holds; a request containing one is a compound for the decode."""

_GRAMMAR_WORDS = frozenset({"turn", "switch", "set", "dim", "on", "off", "the", "to", "in"})

_WORD = r"[^\s]+"
_TARGET = rf"(?P<target>{_WORD}(?: {_WORD}){{0,3}})"
_LIGHT = r"(?P<light>lights?|lamps?)"
_VERB = r"(?:turn|switch)"
_STATE = r"(?P<state>on|off)"
_PERCENT = r"(?P<percent>\d{1,3}) percent"

_POWER_LIGHT = re.compile(rf"^{_VERB} {_STATE}(?: the)?(?: {_TARGET})? {_LIGHT}$")
_POWER_IN = re.compile(rf"^{_VERB} {_STATE}(?: the)? {_LIGHT} in(?: the)? {_TARGET}$")
_POWER_STATE_LAST = re.compile(rf"^(?:{_VERB} )?(?:the )?(?:{_TARGET} )?{_LIGHT} {_STATE}$")
_POWER_KNOWN = re.compile(rf"^{_VERB} {_STATE}(?: the)? {_TARGET}$")
_ADJUST_LIGHT = re.compile(rf"^(?:set|dim)(?: the)?(?: {_TARGET})? {_LIGHT} to {_PERCENT}$")
_ADJUST_KNOWN = re.compile(rf"^(?:set|dim)(?: the)? {_TARGET} to {_PERCENT}$")

_POWER = "lights.power"
_ADJUST = "lights.adjust"


def match_lights_intent(text: str, known_targets: Iterable[str]) -> FastIntent | None:
    """Match one normalised utterance against the lights rows.

    Args:
        text: `normalize_utterance` output.
        known_targets: The current catalog's light names and areas; empty before the first
            refresh.

    Returns:
        `lights.power` with `on` and, when one was said, `target`; or `lights.adjust` with
        `brightness_pct` and maybe `target`. Arguments are unvalidated, as `FastIntent`
        requires. None when no row matches.
    """
    # Catalog spellings go through the same normalisation as the utterance, so "Kid's Lamp"
    # or "Bedroom Two" compares equal to what the user's words became.
    known = frozenset(normalize_utterance(target, ()) for target in known_targets)

    for pattern in (_POWER_LIGHT, _POWER_IN, _POWER_STATE_LAST):
        match = pattern.match(text)
        if match is not None:
            allowed, target = _light_target(match, known, adjacent=pattern is not _POWER_IN)
            return _intent(_POWER, "on", match["state"] == "on", target) if allowed else None

    match = _POWER_KNOWN.match(text)
    if match is not None:
        target = _known_target(match["target"], known)
        return None if target is None else _intent(_POWER, "on", match["state"] == "on", target)

    match = _ADJUST_LIGHT.match(text)
    if match is not None:
        allowed, target = _light_target(match, known, adjacent=True)
        return (
            _intent(_ADJUST, "brightness_pct", int(match["percent"]), target) if allowed else None
        )

    match = _ADJUST_KNOWN.match(text)
    if match is not None:
        target = _known_target(match["target"], known)
        if target is not None:
            return _intent(_ADJUST, "brightness_pct", int(match["percent"]), target)
    return None


def _light_target(
    match: re.Match[str], known: frozenset[str], *, adjacent: bool
) -> tuple[bool, str | None]:
    """Return whether the match stands, and the target said beside the light word, if any.

    A match does not stand when its target holds a stop word or a grammar word.
    """
    target = match["target"]
    if target is not None and any(
        word in SLOT_STOP_WORDS or word in _GRAMMAR_WORDS for word in target.split()
    ):
        return False, None
    if adjacent:
        named = f"{target} {match['light']}" if target else match["light"]
        if named in known:
            return True, named
    return True, target


def _known_target(target: str, known: frozenset[str]) -> str | None:
    if any(word in SLOT_STOP_WORDS for word in target.split()):
        return None
    return target if target in known else None


def _intent(tool: str, slot: str, value: object, target: str | None) -> FastIntent:
    args: dict[str, object] = {slot: value}
    if target is not None:
        args["target"] = target
    return FastIntent(tool=tool, args=args)
