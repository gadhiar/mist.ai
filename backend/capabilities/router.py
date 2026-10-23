"""Per-turn capability routing under a token ceiling.

The action decode runs on a small model, which chooses better among fewer, shorter options, and
a short decode prompt prefills quickly. So each turn offers the tools of at most two
capabilities, chosen deterministically from the utterance and the session, never by the model.

The token ceiling holds by construction rather than by dropping capabilities at turn time:
registration refuses a capability whose worst-case catalog exceeds half of what the ceiling
leaves after the decode instructions, so any two admitted capabilities fit together.
"""

from __future__ import annotations

import functools
import math
import re
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol

from backend.errors import CapabilityConfigError, CapabilityRegistrationError

from .contracts import Capability
from .schema import to_args_schema, to_catalog_line

if TYPE_CHECKING:
    # Only the protocol is needed, for annotations: the router calls nothing but `count`, which
    # the planner's `ApproximateTokenCounter` provides.
    from backend.chat.context_budget import TokenCounter

_MAX_CAPABILITIES = 2
_MAX_TOOLS = 4

# Each capability's keyword floor, by capability name. A capability adds to these through
# `routing_hints()` and `dynamic_keywords()`; it cannot remove one.
_MINIMUM_KEYWORDS: Mapping[str, tuple[str, ...]] = {
    "lights": (
        "light",
        "lights",
        "lamp",
        "lamps",
        "bulb",
        "bulbs",
        "strip",
        "strips",
        "dim",
        "dimmer",
        "brighten",
        "brighter",
        "bright",
        "brightness",
        "dark",
        "darker",
        "warmer",
        "cooler",
        "color",
        "colour",
    ),
    "music": (
        "music",
        "play",
        "playing",
        "song",
        "songs",
        "track",
        "album",
        "artist",
        "playlist",
        "spotify",
        "pause",
        "resume",
        "skip",
        "next",
        "previous",
        "volume",
        "louder",
        "quieter",
    ),
    "alarms": ("alarm", "alarms", "wake", "snooze"),
    "timers": ("timer", "timers", "countdown"),
    "clock": ("time", "date", "day", "today"),
}


class Embedder(Protocol):
    """Turns texts into vectors for the embedding stage; one vector per text, in order."""

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return one embedding per text."""
        ...


@dataclass(frozen=True, slots=True)
class RouterConfig:
    """The router's settings.

    `CapabilitiesConfig` carries the environment's values for all but `reserved_tokens`, which
    depends on the decode instructions rather than on configuration.

    Attributes:
        token_budget: The ceiling on the decode prompt's capability catalog plus its
            instructions (`MIST_CAPABILITY_TOOL_TOKEN_BUDGET`).
        reserved_tokens: The decode instructions' token count, which the catalog cannot use.
        sticky_turns: How many of the session's last turns keep their capabilities routed.
        embedding_threshold: The cosine similarity an exemplar must reach to route; None keeps
            the embedding stage off.
    """

    token_budget: int
    reserved_tokens: int
    sticky_turns: int = 2
    embedding_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.reserved_tokens < 0:
            raise CapabilityConfigError(f"reserved_tokens={self.reserved_tokens} is negative")
        if (self.token_budget - self.reserved_tokens) // _MAX_CAPABILITIES < 1:
            raise CapabilityConfigError(
                f"token budget {self.token_budget} leaves no catalog room after "
                f"{self.reserved_tokens} reserved tokens"
            )
        if self.sticky_turns < 0:
            raise CapabilityConfigError(f"sticky_turns={self.sticky_turns} is negative")


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """The capabilities routed for one turn, and why.

    Attributes:
        capabilities: At most two names, in priority order.
        reasons: For each routed capability, `"active"`, `"keyword:<k>"`, `"sticky"` or
            `"embedding:<score>"`; the highest-priority reason when several apply.
        dropped: Each matched capability left out, mapped to `"cap"`. Size never drops one.
    """

    capabilities: tuple[str, ...]
    reasons: Mapping[str, str]
    dropped: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class _Registered:
    capability: Capability
    order: int
    static_keywords: tuple[str, ...]


class ToolRouter:
    """Selects the capabilities, not the tools, whose catalog a turn's decode offers.

    Candidates come from four stages, ranked in this order when more than two match: a ringing
    item forces its capability (`active()`); keyword hits, by the number of distinct keywords
    matched; capabilities the session used in its last `sticky_turns` turns, most recent
    first; and, when a threshold is configured, exemplar similarity. Ties keep registration
    order.
    """

    def __init__(
        self,
        capabilities: Sequence[Capability],
        config: RouterConfig,
        counter: TokenCounter,
        embedder: Embedder | None,
    ) -> None:
        """Register the capabilities, refusing any that could break the token ceiling.

        Args:
            capabilities: In registration order, which breaks ranking ties.
            config: Budget, stickiness and the embedding threshold.
            counter: Counts catalog tokens, through `count_catalog_tokens`.
            embedder: Required when `config.embedding_threshold` is set.

        Raises:
            CapabilityRegistrationError: Two capabilities share a name; one has more than 4
                tools, a blank keyword, a schema the grammar cannot express, or a worst-case
                catalog (static lines plus `refine_allowance_tokens`) over
                `per_capability_budget()`.
            CapabilityConfigError: A threshold is set without an embedder.
        """
        if config.embedding_threshold is not None and embedder is None:
            raise CapabilityConfigError("the embedding stage has a threshold but no embedder")
        self._config = config
        self._counter = counter
        self._embedder = embedder
        self._registered: list[_Registered] = []
        self._by_name: dict[str, _Registered] = {}
        self._history: dict[str, deque[tuple[str, ...]]] = {}
        self._exemplar_vectors: list[tuple[str, Sequence[float]]] | None = None
        for order, capability in enumerate(capabilities):
            self._register(capability, order)

    def per_capability_budget(self) -> int:
        """Return the most catalog tokens one capability may need in its worst case.

        The ceiling, less `config.reserved_tokens` (the host reserves the decode
        instructions), divided by 2, so two capabilities at the limit still fit.
        """
        return (self._config.token_budget - self._config.reserved_tokens) // _MAX_CAPABILITIES

    def select(self, utterance: str, session_id: str) -> RouteDecision:
        """Route one utterance.

        Args:
            utterance: The user's text; matching casefolds it and needs word boundaries, so
                "lamp" matches "the lamp" and not "lampshade".
            session_id: Whose stickiness applies.

        Returns:
            At most two capabilities with a reason each, plus every other matched capability
            in `dropped`.
        """
        text = utterance.casefold()
        reasons: dict[str, str] = {}
        ranked: list[str] = []

        def admit(name: str, reason: str) -> None:
            if name not in reasons:
                reasons[name] = reason
                ranked.append(name)

        for entry in self._registered:
            if entry.capability.active():
                admit(entry.capability.name, "active")

        keyword_hits = []
        for entry in self._registered:
            hits = _keyword_hits(text, (*entry.static_keywords, *_dynamic(entry.capability)))
            if hits:
                keyword_hits.append((-len(hits), entry.order, entry.capability.name, hits[0]))
        for _, _, name, first in sorted(keyword_hits):
            admit(name, f"keyword:{first}")

        for turn in reversed(self._history.get(session_id, ())):
            for name in sorted(turn, key=lambda used: self._by_name[used].order):
                admit(name, "sticky")

        if self._config.embedding_threshold is not None:
            for score, _, name in self._embedding_matches(utterance):
                admit(name, f"embedding:{score:.3f}")

        kept = tuple(ranked[:_MAX_CAPABILITIES])
        return RouteDecision(
            capabilities=kept,
            reasons=MappingProxyType({name: reasons[name] for name in kept}),
            dropped=MappingProxyType({name: "cap" for name in ranked[_MAX_CAPABILITIES:]}),
        )

    def record_turn(self, session_id: str, used: Iterable[str]) -> None:
        """Record the capabilities one turn used, for stickiness.

        Call it once per turn, with an empty `used` for a turn that used none, so stickiness
        ages by turns. Names that are not registered are ignored.
        """
        if self._config.sticky_turns == 0:
            return
        turns = self._history.setdefault(session_id, deque(maxlen=self._config.sticky_turns))
        turns.append(tuple(dict.fromkeys(name for name in used if name in self._by_name)))

    def _register(self, capability: Capability, order: int) -> None:
        name = capability.name
        if name in self._by_name:
            raise CapabilityRegistrationError(f"capability '{name}' is registered twice")
        tools = list(capability.tools())
        if len(tools) > _MAX_TOOLS:
            raise CapabilityRegistrationError(
                f"capability '{name}' has {len(tools)} tools; at most "
                f"{_MAX_TOOLS} tools are allowed"
            )
        static = count_catalog_tokens(
            [to_catalog_line(spec, to_args_schema(spec)) for spec in tools], self._counter
        )
        worst = static + capability.refine_allowance_tokens
        budget = self.per_capability_budget()
        if worst > budget:
            raise CapabilityRegistrationError(
                f"capability '{name}' needs up to {worst} catalog tokens ({static} static + "
                f"{capability.refine_allowance_tokens} refine allowance); the per-capability "
                f"budget is {budget}"
            )
        declared = capability.routing_hints().keywords
        if any(not keyword.strip() for keyword in declared):
            raise CapabilityRegistrationError(f"capability '{name}' declares a blank keyword")
        keywords = tuple(dict.fromkeys((*_MINIMUM_KEYWORDS.get(name, ()), *declared)))
        entry = _Registered(capability=capability, order=order, static_keywords=keywords)
        self._registered.append(entry)
        self._by_name[name] = entry

    def _embedding_matches(self, utterance: str) -> list[tuple[float, int, str]]:
        """Return `(score, order, name)` for each capability at or above the threshold."""
        threshold = self._config.embedding_threshold
        if threshold is None or self._embedder is None:
            return []
        exemplars = self._exemplars(self._embedder)
        if not exemplars:
            return []
        (query,) = self._embedder.embed([utterance])
        best: dict[str, float] = {}
        for name, vector in exemplars:
            score = _cosine(query, vector)
            best[name] = max(score, best.get(name, -1.0))
        matches = [
            (score, self._by_name[name].order, name)
            for name, score in best.items()
            if score >= threshold
        ]
        return sorted(matches, key=lambda match: (-match[0], match[1]))

    def _exemplars(self, embedder: Embedder) -> list[tuple[str, Sequence[float]]]:
        # Embedded on first use rather than at construction, so startup never waits on the
        # embedder, and only when the stage is on.
        if self._exemplar_vectors is None:
            pairs = [
                (entry.capability.name, exemplar)
                for entry in self._registered
                for exemplar in entry.capability.routing_hints().exemplars
            ]
            vectors = embedder.embed([text for _, text in pairs]) if pairs else []
            self._exemplar_vectors = [
                (name, vector) for (name, _), vector in zip(pairs, vectors, strict=True)
            ]
        return self._exemplar_vectors


def count_catalog_tokens(lines: Sequence[str], counter: TokenCounter) -> int:
    """Count catalog lines the way registration budgets them.

    Each line is counted with its trailing newline, as the decode prompt lays lines out. For a
    counter whose count of joined text never exceeds the sum of its parts' counts (true of
    `ApproximateTokenCounter`, a ceiling of characters over 3.6), the lines of two capabilities
    together then cost no more than the two counts added. A caller that measures refined lines
    against `per_capability_budget()` uses this function, so its measure and registration's
    agree.

    Args:
        lines: `to_catalog_line` outputs.
        counter: The token counter the router was built with.

    Returns:
        The summed count; 0 for no lines.
    """
    return sum(counter.count(f"{line}\n") for line in lines)


def _dynamic(capability: Capability) -> tuple[str, ...]:
    return tuple(keyword for keyword in capability.dynamic_keywords() if keyword.strip())


def _keyword_hits(text: str, keywords: Sequence[str]) -> list[str]:
    """Return the distinct keywords found in casefolded `text`, by first position."""
    positions: dict[str, int] = {}
    for keyword in keywords:
        match = _keyword_pattern(keyword).search(text)
        if match is not None and keyword not in positions:
            positions[keyword] = match.start()
    return sorted(positions, key=lambda keyword: (positions[keyword], -len(keyword)))


@functools.lru_cache(maxsize=4096)
def _keyword_pattern(keyword: str) -> re.Pattern[str]:
    # Cached because dynamic keywords are re-read every turn; bounded because a light catalog
    # can change while the process runs.
    body = r"\s+".join(re.escape(word) for word in keyword.casefold().split())
    return re.compile(rf"(?<!\w){body}(?!\w)")


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norm if norm else 0.0
