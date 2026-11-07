"""Tests for `DecisionMatcher` and `ProposalStore`: cross-turn confirmation of tool calls."""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta

import pytest
import pytest_asyncio

from backend.capabilities.intents import normalize_utterance
from backend.capabilities.proposals import DecisionMatcher, Proposal, ProposalStore, ProposedCall
from backend.errors import CapabilityInvocationError
from tests.unit.capabilities.fakes import DEFAULT_START, EmptyArgs, FakeClock

WAKE_WORDS = ("mist", "hey mist", "ok mist", "okay mist")
TTL_S = 60.0


def _match(raw: str):
    return DecisionMatcher().match(normalize_utterance(raw, WAKE_WORDS))


def _call(summary: str, capability: str = "lights", tool: str = "lights.power") -> ProposedCall:
    return ProposedCall(
        capability=capability, tool=tool, args=EmptyArgs(), summary=summary, risk="physical"
    )


LIGHTS_OFF = _call("turn off the bedroom lights")
PLAY_JAZZ = _call("play jazz", capability="music", tool="music.play")


class TestDecisionMatcher:
    @pytest.mark.parametrize("raw", ["yes please", "Yes.", "OK, do it", "go ahead", "ok MIST, yes"])
    def test_approves(self, raw):
        assert _match(raw) == ("approve", None)

    @pytest.mark.parametrize("raw", ["no thanks", "never mind", "Don't."])
    def test_rejects(self, raw):
        assert _match(raw) == ("reject", None)

    @pytest.mark.parametrize(
        ("raw", "remainder"),
        [("yes, and dim them", "dim them"), ("yes then play jazz", "play jazz")],
    )
    def test_approves_with_remainder(self, raw, remainder):
        assert _match(raw) == ("approve", remainder)

    @pytest.mark.parametrize(
        "raw",
        [
            "okay, never mind",
            "yeah but only the lamp",
            "ok what's the weather",
            "yes and don't touch the lamp",
            "thanks",
            "no, play jazz instead",
            "what time is it",
        ],
    )
    def test_not_a_decision(self, raw):
        assert _match(raw) is None

    def test_please_do_is_one_approve_phrase(self):
        assert DecisionMatcher().match("please do it") == ("approve", None)

    def test_also_connective_approves_with_remainder(self):
        assert DecisionMatcher().match("sure also play jazz") == ("approve", "play jazz")

    def test_connective_with_nothing_after_it_approves_without_remainder(self):
        assert DecisionMatcher().match("yes and") == ("approve", None)

    def test_do_alone_approves_because_normalising_please_do_leaves_it(self):
        assert DecisionMatcher().match("do") == ("approve", None)

    def test_hold_on_in_the_remainder_is_a_hedge(self):
        assert DecisionMatcher().match("yes and hold on") is None

    def test_empty_text_is_not_a_decision(self):
        assert DecisionMatcher().match("") is None

    def test_fillers_alone_are_not_a_decision(self):
        assert DecisionMatcher().match("thank you") is None


class TestDesignPhrasesEndToEnd:
    """Every phrase the Design lists, run through `normalize_utterance` first, as the host will.

    Normalisation strips leading and trailing courtesies, so a listed phrase that starts or ends
    with one ("please do") reaches the matcher changed; these tests catch that class.
    """

    @pytest.mark.parametrize(
        "raw",
        [
            "yes",
            "yeah",
            "yep",
            "sure",
            "ok",
            "okay",
            "confirm",
            "do it",
            "go ahead",
            "please do",
            "Please do.",
            "yes please",
            "OK, do it",
        ],
    )
    def test_listed_approval(self, raw):
        assert _match(raw) == ("approve", None)

    @pytest.mark.parametrize(
        "raw", ["no", "nope", "cancel", "stop", "dont", "Don't", "never mind", "no thanks"]
    )
    def test_listed_rejection(self, raw):
        assert _match(raw) == ("reject", None)

    @pytest.mark.parametrize(
        ("raw", "remainder"),
        [
            ("yes, and dim them", "dim them"),
            ("yes then play jazz", "play jazz"),
            ("okay also play jazz", "play jazz"),
        ],
    )
    def test_listed_approval_with_remainder(self, raw, remainder):
        assert _match(raw) == ("approve", remainder)

    @pytest.mark.parametrize(
        "raw", ["okay, never mind", "yeah but only the lamp", "ok what's the weather", "thanks"]
    )
    def test_listed_non_decision(self, raw):
        assert _match(raw) is None


class _Expiries:
    def __init__(self) -> None:
        self.proposals: list[Proposal] = []

    def __call__(self, proposal: Proposal) -> None:
        self.proposals.append(proposal)


_STORES: list[ProposalStore] = []


@pytest_asyncio.fixture(autouse=True)
async def _close_stores():
    """Close every store a test built, so no expiry timer outlives the test's loop."""
    yield
    while _STORES:
        await _STORES.pop().close()


def _store(clock: FakeClock, expiries: _Expiries | None = None) -> ProposalStore:
    store = ProposalStore(clock, TTL_S, expiries if expiries is not None else _Expiries())
    _STORES.append(store)
    return store


def _timer_tasks() -> list[asyncio.Task[object]]:
    return [
        task
        for task in asyncio.all_tasks()
        if task.get_coro().__qualname__ == "ProposalStore._expire_after_ttl"
    ]


class TestPut:
    @pytest.mark.asyncio
    async def test_bundles_calls_in_order_with_joined_summary(self):
        store = _store(FakeClock())

        proposal = store.put("s1", [LIGHTS_OFF, PLAY_JAZZ])

        assert proposal.session_id == "s1"
        assert proposal.calls == (LIGHTS_OFF, PLAY_JAZZ)
        assert proposal.summary == "turn off the bedroom lights, then play jazz"
        assert proposal.expires_at == DEFAULT_START + timedelta(seconds=TTL_S)
        assert proposal.proposal_id

    @pytest.mark.asyncio
    async def test_single_call_summary_is_its_own(self):
        store = _store(FakeClock())

        assert store.put("s1", [PLAY_JAZZ]).summary == "play jazz"

    @pytest.mark.asyncio
    async def test_ids_differ_between_proposals(self):
        store = _store(FakeClock())

        first = store.put("s1", [LIGHTS_OFF])
        second = store.put("s2", [LIGHTS_OFF])

        assert first.proposal_id != second.proposal_id

    @pytest.mark.asyncio
    async def test_empty_call_list_is_refused(self):
        store = _store(FakeClock())

        with pytest.raises(ValueError, match="at least one call"):
            store.put("s1", [])

    @pytest.mark.asyncio
    async def test_new_proposal_supersedes_the_live_one(self):
        clock = FakeClock()
        expiries = _Expiries()
        store = _store(clock, expiries)
        old = store.put("s1", [LIGHTS_OFF])

        new = store.put("s1", [PLAY_JAZZ])

        assert store.pop("s1", None) == new
        assert store.pop("s1", old.proposal_id) is None
        await clock.advance(TTL_S * 2)
        assert expiries.proposals == []

    @pytest.mark.asyncio
    async def test_sessions_hold_one_live_proposal_each(self):
        store = _store(FakeClock())
        first = store.put("s1", [LIGHTS_OFF])
        second = store.put("s2", [PLAY_JAZZ])

        assert store.pop("s1", None) == first
        assert store.pop("s2", None) == second


class TestPop:
    @pytest.mark.asyncio
    async def test_pop_returns_the_proposal_once_then_none(self):
        store = _store(FakeClock())
        proposal = store.put("s1", [LIGHTS_OFF])

        assert store.pop("s1", proposal.proposal_id) == proposal
        assert store.pop("s1", proposal.proposal_id) is None
        assert store.pop("s1", None) is None

    @pytest.mark.asyncio
    async def test_pop_without_id_takes_the_live_proposal_once(self):
        store = _store(FakeClock())
        proposal = store.put("s1", [LIGHTS_OFF])

        assert store.pop("s1", None) == proposal
        assert store.pop("s1", None) is None
        assert store.pop("s1", proposal.proposal_id) is None

    @pytest.mark.asyncio
    async def test_pop_is_synchronous(self):
        store = _store(FakeClock())
        store.put("s1", [LIGHTS_OFF])

        result = store.pop("s1", None)

        assert not asyncio.iscoroutine(result)
        assert isinstance(result, Proposal)

    @pytest.mark.asyncio
    async def test_other_sessions_id_returns_none_and_leaves_it_live(self):
        store = _store(FakeClock())
        proposal = store.put("s1", [LIGHTS_OFF])

        assert store.pop("s2", proposal.proposal_id) is None
        assert store.pop("s1", proposal.proposal_id) == proposal

    @pytest.mark.asyncio
    async def test_other_sessions_resolved_id_returns_none(self):
        store = _store(FakeClock())
        proposal = store.put("s1", [LIGHTS_OFF])
        store.pop("s1", None)

        assert store.pop("s2", proposal.proposal_id) is None

    @pytest.mark.asyncio
    async def test_unknown_id_raises(self):
        store = _store(FakeClock())
        store.put("s1", [LIGHTS_OFF])

        with pytest.raises(CapabilityInvocationError, match="unknown"):
            store.pop("s1", "no-such-id")

    @pytest.mark.asyncio
    async def test_pop_without_id_and_nothing_live_returns_none(self):
        store = _store(FakeClock())

        assert store.pop("s1", None) is None

    @pytest.mark.asyncio
    async def test_popped_proposal_never_expires(self):
        clock = FakeClock()
        expiries = _Expiries()
        store = _store(clock, expiries)
        store.put("s1", [LIGHTS_OFF])
        store.pop("s1", None)

        await clock.advance(TTL_S * 2)

        assert expiries.proposals == []
        assert clock.pending == 0


class TestExpiry:
    @pytest.mark.asyncio
    async def test_expiry_at_ttl_calls_on_expire_once(self):
        clock = FakeClock()
        expiries = _Expiries()
        store = _store(clock, expiries)
        proposal = store.put("s1", [LIGHTS_OFF])
        await clock.wait_for_sleepers(1)

        await clock.advance(TTL_S - 1)
        assert expiries.proposals == []

        await clock.advance(1)
        assert expiries.proposals == [proposal]

        await clock.advance(TTL_S * 2)
        assert expiries.proposals == [proposal]
        assert clock.stalled == []

    @pytest.mark.asyncio
    async def test_expired_id_raises(self):
        clock = FakeClock()
        store = _store(clock)
        proposal = store.put("s1", [LIGHTS_OFF])
        await clock.wait_for_sleepers(1)
        await clock.advance(TTL_S)

        with pytest.raises(CapabilityInvocationError, match="expired"):
            store.pop("s1", proposal.proposal_id)
        assert store.pop("s1", None) is None

    @pytest.mark.asyncio
    async def test_pop_after_deadline_expires_even_before_the_timer_runs(self):
        clock = FakeClock()
        expiries = _Expiries()
        store = _store(clock, expiries)
        proposal = store.put("s1", [LIGHTS_OFF])
        # Move time without waking the timer task, as a busy loop might.
        clock._now = clock.now() + timedelta(seconds=TTL_S)

        with pytest.raises(CapabilityInvocationError, match="expired"):
            store.pop("s1", proposal.proposal_id)
        assert expiries.proposals == [proposal]

        await clock.advance(TTL_S)
        assert expiries.proposals == [proposal]

    @pytest.mark.asyncio
    async def test_pop_without_id_after_deadline_returns_none_and_expires(self):
        clock = FakeClock()
        expiries = _Expiries()
        store = _store(clock, expiries)
        proposal = store.put("s1", [LIGHTS_OFF])
        clock._now = clock.now() + timedelta(seconds=TTL_S)

        assert store.pop("s1", None) is None
        assert expiries.proposals == [proposal]

    @pytest.mark.asyncio
    async def test_failing_on_expire_is_logged_not_lost(self, caplog):
        clock = FakeClock()

        def explode(proposal: Proposal) -> None:
            raise RuntimeError("sink broke")

        store = ProposalStore(clock, TTL_S, explode)
        store.put("s1", [LIGHTS_OFF])
        await clock.wait_for_sleepers(1)

        with caplog.at_level(logging.ERROR, logger="backend.capabilities.proposals"):
            await clock.advance(TTL_S)

        assert any("sink broke" in (r.exc_text or str(r.exc_info)) for r in caplog.records)


_LOGGER = "backend.capabilities.proposals"


class TestClose:
    @pytest.mark.asyncio
    async def test_close_drops_live_proposals_and_ends_every_timer(self):
        clock = FakeClock()
        expiries = _Expiries()
        store = _store(clock, expiries)
        store.put("s1", [LIGHTS_OFF])
        store.put("s2", [PLAY_JAZZ])
        await clock.wait_for_sleepers(2)
        timers = _timer_tasks()
        assert len(timers) == 2

        await store.close()

        assert all(timer.done() for timer in timers)
        assert _timer_tasks() == []
        assert clock.pending == 0
        await clock.advance(TTL_S * 2)
        assert expiries.proposals == []

    @pytest.mark.asyncio
    async def test_close_ends_a_timer_that_has_not_started(self):
        clock = FakeClock()
        store = _store(clock)
        store.put("s1", [LIGHTS_OFF])
        (timer,) = _timer_tasks()

        await store.close()

        assert timer.done()
        assert clock.pending == 0

    @pytest.mark.asyncio
    async def test_close_logs_one_info_line_with_the_count(self, caplog):
        clock = FakeClock()
        store = _store(clock)
        store.put("s1", [LIGHTS_OFF])
        store.put("s2", [PLAY_JAZZ])

        with caplog.at_level(logging.INFO, logger=_LOGGER):
            await store.close()

        (record,) = (r for r in caplog.records if r.name == _LOGGER)
        assert record.levelno == logging.INFO
        assert "2" in record.getMessage()

    @pytest.mark.asyncio
    async def test_put_after_close_raises_and_creates_no_task(self):
        store = _store(FakeClock())
        await store.close()
        before = asyncio.all_tasks()

        with pytest.raises(CapabilityInvocationError, match="^proposal store is closed$"):
            store.put("s1", [LIGHTS_OFF])

        assert asyncio.all_tasks() == before
        assert store.pop("s1", None) is None

    @pytest.mark.asyncio
    async def test_pop_after_close_returns_none(self):
        store = _store(FakeClock())
        dropped = store.put("s1", [LIGHTS_OFF])
        resolved = store.put("s2", [PLAY_JAZZ])
        store.pop("s2", None)

        await store.close()

        assert store.pop("s1", None) is None
        assert store.pop("s1", dropped.proposal_id) is None
        assert store.pop("s2", resolved.proposal_id) is None
        assert store.pop("s1", "no-such-id") is None

    @pytest.mark.asyncio
    async def test_second_close_logs_nothing_more(self, caplog):
        store = _store(FakeClock())
        store.put("s1", [LIGHTS_OFF])

        def lines() -> int:
            return sum(1 for r in caplog.records if r.name == _LOGGER)

        with caplog.at_level(logging.DEBUG, logger=_LOGGER):
            await store.close()
            first = lines()
            await store.close()

        assert first == 1
        assert lines() == 1

    @pytest.mark.asyncio
    async def test_timer_cancelled_by_a_pop_just_before_close_has_ended(self):
        clock = FakeClock()
        store = _store(clock)
        store.put("s1", [LIGHTS_OFF])
        await clock.wait_for_sleepers(1)
        (timer,) = _timer_tasks()

        store.pop("s1", None)
        assert not timer.done()
        await store.close()

        assert timer.done()

    @pytest.mark.asyncio
    async def test_close_of_an_empty_store_returns(self, caplog):
        store = _store(FakeClock())

        with caplog.at_level(logging.INFO, logger=_LOGGER):
            await asyncio.wait_for(store.close(), 1.0)

        (record,) = (r for r in caplog.records if r.name == _LOGGER)
        assert "0" in record.getMessage()
