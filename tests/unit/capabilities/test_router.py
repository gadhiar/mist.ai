"""Tests for `ToolRouter`: keyword, sticky, active and embedding selection under a token ceiling.

Every utterance here is synthetic, written for these tests, apart from the ones copied from the
parent plan's live checklist (H8), which are themselves written examples.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import pytest
from pydantic import BaseModel

from backend.capabilities.contracts import RoutingHints, ToolSpec
from backend.capabilities.router import (
    RouteDecision,
    RouterConfig,
    ToolRouter,
    count_catalog_tokens,
)
from backend.capabilities.schema import to_args_schema, to_catalog_line
from backend.chat.context_budget import ApproximateTokenCounter
from backend.errors import CapabilityConfigError, CapabilityRegistrationError
from tests.unit.capabilities.fakes import FakeCapability, make_tool_spec

CAPABILITY_NAMES = ("lights", "music", "alarms", "timers", "clock")
CONFIG = RouterConfig(token_budget=1200, reserved_tokens=200)
COUNTER = ApproximateTokenCounter()
LIGHT_TARGETS = ("bedroom", "kitchen", "living room", "desk lamp")

MINIMUM_KEYWORDS = {
    "lights": (
        "light lights lamp lamps bulb bulbs strip strips dim dimmer brighten brighter bright "
        "brightness dark darker warmer cooler color colour"
    ).split(),
    "music": (
        "music play playing song songs track album artist playlist spotify pause resume skip "
        "next previous volume louder quieter"
    ).split(),
    "alarms": "alarm alarms wake snooze".split(),
    "timers": "timer timers countdown".split(),
    "clock": "time date day today".split(),
}


def _capabilities() -> dict[str, FakeCapability]:
    """One fake per capability, declaring no keywords of its own, so only the floor applies."""
    capabilities = {name: FakeCapability(name) for name in CAPABILITY_NAMES}
    capabilities["lights"].extra_keywords = LIGHT_TARGETS
    return capabilities


def _router(
    capabilities: dict[str, FakeCapability] | Sequence[FakeCapability] | None = None,
    *,
    config: RouterConfig = CONFIG,
    embedder: object | None = None,
) -> ToolRouter:
    if capabilities is None:
        capabilities = _capabilities()
    if isinstance(capabilities, dict):
        capabilities = list(capabilities.values())
    return ToolRouter(capabilities, config, COUNTER, embedder)  # type: ignore[arg-type]


class TestKeywords:
    def test_word_boundary_match(self):
        decision = _router().select("turn on the lamp", "s1")

        assert decision.capabilities == ("lights",)
        assert decision.reasons == {"lights": "keyword:lamp"}
        assert decision.dropped == {}

    def test_keyword_inside_a_longer_word_does_not_match(self):
        assert _router().select("how are lampshades made", "s1").capabilities == ()

    def test_a_listed_longer_word_matches(self):
        capabilities = _capabilities()
        capabilities["lights"].hints = RoutingHints(keywords=("lampshade",))

        decision = _router(capabilities).select("the lampshade is crooked", "s1")

        assert decision.reasons == {"lights": "keyword:lampshade"}

    def test_matching_is_casefolded(self):
        assert _router().select("Turn On The LAMP", "s1").capabilities == ("lights",)

    def test_keyword_next_to_punctuation_matches(self):
        assert _router().select("lamp, please!", "s1").capabilities == ("lights",)

    def test_reason_names_the_first_keyword_in_the_utterance(self):
        decision = _router().select("make the lamp brighter", "s1")

        assert decision.reasons == {"lights": "keyword:lamp"}

    def test_multi_word_dynamic_keyword_matches(self):
        decision = _router().select("it is gloomy in the living  room", "s1")

        assert decision.reasons == {"lights": "keyword:living room"}

    def test_dynamic_keywords_are_read_on_each_select(self):
        capabilities = _capabilities()
        router = _router(capabilities)
        assert router.select("the porch is gloomy", "s1").capabilities == ()

        capabilities["lights"].extra_keywords = ("porch",)

        assert router.select("the porch is gloomy", "s1").capabilities == ("lights",)

    def test_nothing_matched_routes_nothing(self):
        assert _router().select("tell me a joke", "s1") == RouteDecision(
            capabilities=(), reasons={}, dropped={}
        )

    @pytest.mark.parametrize(
        ("capability", "word"),
        [(name, word) for name, words in MINIMUM_KEYWORDS.items() for word in words],
    )
    def test_minimum_keyword_routes_to_its_capability(self, capability, word):
        decision = _router().select(f"um {word} ok", "s1")

        assert decision.capabilities == (capability,)
        assert decision.reasons == {capability: f"keyword:{word}"}

    def test_minimum_keywords_apply_only_to_their_capability_name(self):
        other = FakeCapability("weather")

        assert _router([other]).select("dim the lamp", "s1").capabilities == ()


class TestPriorityAndCap:
    def test_at_most_two_by_keyword_score_then_registration_order(self):
        decision = _router().select("play the next song and set a timer with the lamp on", "s1")

        assert decision.capabilities == ("music", "lights")
        assert decision.reasons == {"music": "keyword:play", "lights": "keyword:lamp"}
        assert decision.dropped == {"timers": "cap"}

    def test_active_capability_is_forced_first(self):
        capabilities = _capabilities()
        capabilities["alarms"].ringing = True

        decision = _router(capabilities).select("play the next song and dim the lamp", "s1")

        assert decision.capabilities == ("alarms", "music")
        assert decision.reasons == {"alarms": "active", "music": "keyword:play"}
        assert decision.dropped == {"lights": "cap"}

    def test_active_capability_routes_without_any_keyword(self):
        capabilities = _capabilities()
        capabilities["timers"].ringing = True

        decision = _router(capabilities).select("enough already", "s1")

        assert decision.reasons == {"timers": "active"}

    def test_keyword_outranks_sticky(self):
        router = _router()
        router.record_turn("s1", ["clock"])

        decision = router.select("dim the lamp and play a song", "s1")

        assert decision.capabilities == ("lights", "music")
        assert decision.dropped == {"clock": "cap"}

    def test_keyword_and_sticky_together(self):
        router = _router()
        router.record_turn("s1", ["clock"])

        decision = router.select("dim the lamp", "s1")

        assert decision.capabilities == ("lights", "clock")
        assert decision.reasons == {"lights": "keyword:dim", "clock": "sticky"}

    def test_a_capability_both_sticky_and_matched_gives_its_keyword_reason(self):
        router = _router()
        router.record_turn("s1", ["lights"])

        assert router.select("dim the lamp", "s1").reasons == {"lights": "keyword:dim"}


class TestStickiness:
    def test_used_capability_stays_routed_for_the_configured_turns(self):
        router = _router()
        router.record_turn("s1", ["music"])

        assert router.select("turn it up", "s1").reasons == {"music": "sticky"}
        router.record_turn("s1", [])
        assert router.select("turn it up", "s1").reasons == {"music": "sticky"}
        router.record_turn("s1", [])
        assert router.select("turn it up", "s1").capabilities == ()

    def test_sticky_turns_from_config(self):
        router = _router(
            config=RouterConfig(token_budget=1200, reserved_tokens=200, sticky_turns=1)
        )
        router.record_turn("s1", ["music"])
        assert router.select("turn it up", "s1").capabilities == ("music",)

        router.record_turn("s1", [])

        assert router.select("turn it up", "s1").capabilities == ()

    def test_zero_sticky_turns_disables_stickiness(self):
        router = _router(
            config=RouterConfig(token_budget=1200, reserved_tokens=200, sticky_turns=0)
        )
        router.record_turn("s1", ["music"])

        assert router.select("turn it up", "s1").capabilities == ()

    def test_stickiness_is_per_session(self):
        router = _router()
        router.record_turn("s1", ["music"])

        assert router.select("turn it up", "s2").capabilities == ()

    def test_most_recent_turn_ranks_first(self):
        router = _router()
        router.record_turn("s1", ["clock"])
        router.record_turn("s1", ["music"])

        assert router.select("and again", "s1").capabilities == ("music", "clock")

    def test_unknown_capability_names_are_ignored(self):
        router = _router()
        router.record_turn("s1", ["weather", "music"])

        assert router.select("turn it up", "s1").capabilities == ("music",)


class FakeEmbedder:
    """Maps known texts to fixed vectors, everything else to an unrelated one."""

    def __init__(self, vectors: dict[str, Sequence[float]]) -> None:
        self.vectors = vectors
        self.calls: list[tuple[str, ...]] = []

    def embed(self, texts: Sequence[str]) -> list[Sequence[float]]:
        self.calls.append(tuple(texts))
        return [self.vectors.get(text, (0.0, 0.0, 1.0)) for text in texts]


EXEMPLAR = "nothing is visible in this room"
MISS = "I can't see a thing in here"


def _embedding_setup(threshold: float | None, utterance_vector: Sequence[float]):
    capabilities = _capabilities()
    capabilities["lights"].hints = RoutingHints(keywords=(), exemplars=(EXEMPLAR,))
    embedder = FakeEmbedder({EXEMPLAR: (1.0, 0.0, 0.0), MISS: utterance_vector})
    config = RouterConfig(token_budget=1200, reserved_tokens=200, embedding_threshold=threshold)
    return capabilities, embedder, _router(capabilities, config=config, embedder=embedder)


class TestEmbeddingStage:
    def test_off_without_threshold_and_never_calls_the_embedder(self):
        _, embedder, router = _embedding_setup(None, (1.0, 0.0, 0.0))

        assert router.select(MISS, "s1").capabilities == ()
        assert embedder.calls == []

    def test_routes_at_or_above_threshold(self):
        _, _, router = _embedding_setup(0.8, (1.0, 0.0, 0.0))

        decision = router.select(MISS, "s1")

        assert decision.capabilities == ("lights",)
        assert decision.reasons == {"lights": "embedding:1.000"}

    def test_below_threshold_does_not_route(self):
        _, _, router = _embedding_setup(0.8, (0.6, 0.8, 0.0))

        assert router.select(MISS, "s1").capabilities == ()

    def test_exemplars_are_embedded_once(self):
        _, embedder, router = _embedding_setup(0.8, (1.0, 0.0, 0.0))

        router.select(MISS, "s1")
        router.select(MISS, "s1")

        assert embedder.calls == [(EXEMPLAR,), (MISS,), (MISS,)]

    def test_embedding_ranks_below_keywords(self):
        capabilities, _, router = _embedding_setup(0.5, (1.0, 0.0, 0.0))
        utterance = MISS
        capabilities["music"].extra_keywords = ("thing",)
        capabilities["clock"].extra_keywords = ("here",)

        decision = router.select(utterance, "s1")

        assert decision.capabilities == ("music", "clock")
        assert decision.dropped == {"lights": "cap"}

    def test_threshold_without_embedder_is_refused(self):
        config = RouterConfig(token_budget=1200, reserved_tokens=200, embedding_threshold=0.8)

        with pytest.raises(CapabilityConfigError, match="embedder"):
            _router(config=config, embedder=None)


class PowerArgs(BaseModel):
    target: str | None = None
    on: bool


def _tools(description: str) -> tuple[ToolSpec, ...]:
    return (
        make_tool_spec("edge.power", PowerArgs, description=description),
        make_tool_spec("edge.status", description="report state"),
    )


def _catalog_lines(capability: FakeCapability) -> list[str]:
    return [to_catalog_line(spec, to_args_schema(spec)) for spec in capability.tools()]


def _registers(capability: FakeCapability, config: RouterConfig) -> bool:
    try:
        _router([capability], config=config)
    except CapabilityRegistrationError:
        return False
    return True


def _largest_description_that_fits(name: str, config: RouterConfig) -> FakeCapability:
    """Binary-search the description length to put a capability exactly at the budget's edge."""
    low, high = 1, 4 * config.token_budget
    while low < high:
        middle = (low + high + 1) // 2
        if _registers(FakeCapability(name, tools=_tools("x" * middle)), config):
            low = middle
        else:
            high = middle - 1
    return FakeCapability(name, tools=_tools("x" * low))


class TestCountCatalogTokens:
    def test_counts_each_line_with_its_newline(self):
        lines = ["a" * 7, "b" * 11, "c"]

        expected = sum(COUNTER.count(line + "\n") for line in lines)

        assert count_catalog_tokens(lines, COUNTER) == expected

    def test_bounds_the_joined_block_of_any_split(self):
        lines = [f"cap.t{i}(x?: str): {'d' * (i * 13 % 29)}" for i in range(8)]

        total = count_catalog_tokens(lines, COUNTER)

        assert COUNTER.count("\n".join(lines)) <= total
        assert count_catalog_tokens(lines[:4], COUNTER) + count_catalog_tokens(
            lines[4:], COUNTER
        ) >= COUNTER.count("".join(f"{line}\n" for line in lines))

    def test_no_lines_cost_nothing(self):
        assert count_catalog_tokens([], COUNTER) == 0

    def test_registration_uses_it(self):
        capability = FakeCapability("edge", tools=_tools("y" * 40))
        static = count_catalog_tokens(_catalog_lines(capability), COUNTER)
        budget = (700 - 101) // 2
        capability.refine_allowance_tokens = budget - static
        config = RouterConfig(token_budget=700, reserved_tokens=101)
        assert _registers(capability, config)

        capability.refine_allowance_tokens += 1

        assert not _registers(capability, config)


class TestTokenCeiling:
    def test_per_capability_budget_is_half_the_ceiling_after_reserve(self):
        assert _router().per_capability_budget() == 500
        assert (
            _router(
                config=RouterConfig(token_budget=1001, reserved_tokens=0)
            ).per_capability_budget()
            == 500
        )

    def test_more_than_four_tools_is_refused(self):
        tools = [make_tool_spec(f"wide.t{i}") for i in range(5)]

        with pytest.raises(CapabilityRegistrationError, match="4 tools"):
            _router([FakeCapability("wide", tools=tools)])

    def test_four_tools_register(self):
        tools = [make_tool_spec(f"wide.t{i}") for i in range(4)]

        _router([FakeCapability("wide", tools=tools)])

    def test_duplicate_names_are_refused(self):
        with pytest.raises(CapabilityRegistrationError, match="twice"):
            _router([FakeCapability("lights"), FakeCapability("lights")])

    def test_budget_edge_one_more_character_is_refused(self):
        config = RouterConfig(token_budget=700, reserved_tokens=101)
        edge = _largest_description_that_fits("edge", config)
        description = edge.tools()[0].description

        over = FakeCapability("edge", tools=_tools(description + "x"))

        assert _registers(edge, config)
        assert not _registers(over, config)

    def test_refine_allowance_counts_against_the_budget(self):
        config = RouterConfig(token_budget=700, reserved_tokens=101)
        edge = _largest_description_that_fits("edge", config)
        assert _registers(edge, config)

        edge.refine_allowance_tokens = 1

        with pytest.raises(CapabilityRegistrationError, match="budget"):
            _router([edge], config=config)

    def test_any_two_admitted_at_the_edge_fit_the_ceiling(self):
        config = RouterConfig(token_budget=700, reserved_tokens=101)
        first = _largest_description_that_fits("first", config)
        second = _largest_description_that_fits("second", config)
        router = _router([first, second], config=config)
        lines = _catalog_lines(first) + _catalog_lines(second)

        joined = COUNTER.count("\n".join(lines))
        terminated = COUNTER.count("".join(f"{line}\n" for line in lines))

        assert joined + config.reserved_tokens <= config.token_budget
        assert terminated + config.reserved_tokens <= config.token_budget
        assert terminated > 2 * router.per_capability_budget() - 10  # the edge is really the edge

    def test_allowance_reserved_for_refinement_keeps_two_within_ceiling(self):
        config = RouterConfig(token_budget=700, reserved_tokens=101)
        allowance = 60
        capabilities = []
        for name in ("first", "second"):
            probe = _largest_description_that_fits(
                name, RouterConfig(token_budget=700 - 2 * allowance, reserved_tokens=101)
            )
            probe.refine_allowance_tokens = allowance
            capabilities.append(probe)
        _router(capabilities, config=config)
        lines = _catalog_lines(capabilities[0]) + _catalog_lines(capabilities[1])

        worst = COUNTER.count("".join(f"{line}\n" for line in lines)) + 2 * allowance

        assert worst + config.reserved_tokens <= config.token_budget

    @pytest.mark.parametrize(("budget", "reserved"), [(100, -1), (100, 100), (100, 99), (0, 0)])
    def test_config_leaving_no_room_is_refused(self, budget, reserved):
        with pytest.raises(CapabilityConfigError):
            RouterConfig(token_budget=budget, reserved_tokens=reserved)


# -- Labelled fixture ---------------------------------------------------------------------
#
# (utterance, expected capabilities, capabilities the session used on the previous turn).
# The previous-turn column models H8's step 7, where "turn it up" follows "skip".

FIXTURE: list[tuple[str, frozenset[str], tuple[str, ...]]] = [
    # lights
    ("It's dark in here", frozenset({"lights"}), ()),
    ("Turn off the living room lights", frozenset({"lights"}), ()),
    ("turn on the living room lights", frozenset({"lights"}), ()),
    ("Dim the bedroom to 30 percent", frozenset({"lights"}), ()),
    ("make it warmer", frozenset({"lights"}), ()),
    ("What lights are on?", frozenset({"lights"}), ()),
    ("why do my bedroom lights flicker?", frozenset({"lights"}), ()),
    ("turn off the lights", frozenset({"lights"}), ()),
    ("switch the kitchen lamp on", frozenset({"lights"}), ()),
    ("set the desk lamp to a cooler white", frozenset({"lights"}), ()),
    ("can you brighten things up a bit", frozenset({"lights"}), ()),
    ("change the colour of the strip to blue", frozenset({"lights"}), ()),
    ("turn off the bedroom", frozenset({"lights"}), ()),
    # music
    ("Play some jazz and tell me who Miles Davis was", frozenset({"music"}), ()),
    ("Pause", frozenset({"music"}), ()),
    ("skip", frozenset({"music"}), ()),
    ("turn it up", frozenset({"music"}), ("music",)),
    ("what's playing?", frozenset({"music"}), ()),
    ("play some jazz", frozenset({"music"}), ()),
    ("put on my workout playlist", frozenset({"music"}), ()),
    ("next track please", frozenset({"music"}), ()),
    ("who sings this song", frozenset({"music"}), ()),
    ("make the music quieter", frozenset({"music"}), ()),
    ("go back to the previous song", frozenset({"music"}), ()),
    ("resume spotify", frozenset({"music"}), ()),
    ("set the volume to 40 percent", frozenset({"music"}), ()),
    # alarms
    ("set an alarm for 7:30", frozenset({"alarms"}), ()),
    ("wake me up at six tomorrow", frozenset({"alarms"}), ()),
    ("cancel all my alarms", frozenset({"alarms"}), ()),
    ("list my alarms", frozenset({"alarms"}), ()),
    ("snooze", frozenset({"alarms"}), ()),
    ("snooze for five more minutes", frozenset({"alarms"}), ()),
    ("delete the alarm I set for Saturday", frozenset({"alarms"}), ()),
    ("I need an alarm at half past five", frozenset({"alarms"}), ()),
    ("change my alarm to 8", frozenset({"alarms"}), ()),
    ("turn off the alarm", frozenset({"alarms"}), ()),
    ("is my alarm still on for Monday", frozenset({"alarms"}), ()),
    # timers
    ("set a timer for ten minutes", frozenset({"timers"}), ()),
    ("start a 5 minute timer", frozenset({"timers"}), ()),
    ("how long is left on my timer", frozenset({"timers"}), ()),
    ("cancel the timer", frozenset({"timers"}), ()),
    ("stop the timer", frozenset({"timers"}), ()),
    ("list my timers", frozenset({"timers"}), ()),
    ("start a countdown for the eggs", frozenset({"timers"}), ()),
    ("add a timer called laundry for an hour", frozenset({"timers"}), ()),
    ("how many timers are running", frozenset({"timers"}), ()),
    ("turn off the timer", frozenset({"timers"}), ()),
    # clock
    ("What time is it?", frozenset({"clock"}), ()),
    ("what's the date", frozenset({"clock"}), ()),
    ("what day is it", frozenset({"clock"}), ()),
    ("what's today's date", frozenset({"clock"}), ()),
    ("do you know the time", frozenset({"clock"}), ()),
    ("what is the date today", frozenset({"clock"}), ()),
    ("which day of the week is it", frozenset({"clock"}), ()),
    ("what time is it right now", frozenset({"clock"}), ()),
    ("is today a weekday", frozenset({"clock"}), ()),
    ("what date is it", frozenset({"clock"}), ()),
    # two capabilities
    ("Turn off the bedroom lights and play some jazz", frozenset({"lights", "music"}), ()),
    ("what time is my alarm set for", frozenset({"alarms", "clock"}), ()),
    # none
    ("why does my laptop fan get loud?", frozenset(), ()),
    ("tell me a joke", frozenset(), ()),
    ("who was Miles Davis", frozenset(), ()),
    ("how do lampshades get made", frozenset(), ()),
    ("what's the capital of Portugal", frozenset(), ()),
    ("I had a long week", frozenset(), ()),
    ("remind me what we talked about yesterday", frozenset(), ()),
    ("explain how photosynthesis works", frozenset(), ()),
    ("the playground was busy", frozenset(), ()),
    ("how tall is Mount Everest", frozenset(), ()),
    ("what's the weather like", frozenset(), ()),
    ("I can't wait for the weekend", frozenset(), ()),
]


class TestLabelledFixture:
    @pytest.mark.parametrize(("utterance", "expected", "previous"), FIXTURE)
    def test_routes_as_labelled(self, utterance, expected, previous):
        router = _router()
        if previous:
            router.record_turn("s1", previous)

        assert frozenset(router.select(utterance, "s1").capabilities) == expected

    def test_fixture_covers_every_capability_and_none(self):
        single = Counter(next(iter(e)) for _, e, _ in FIXTURE if len(e) == 1)

        assert all(single[name] >= 10 for name in CAPABILITY_NAMES), single
        assert sum(1 for _, e, _ in FIXTURE if not e) >= 10
        assert ("It's dark in here", frozenset({"lights"}), ()) in FIXTURE
