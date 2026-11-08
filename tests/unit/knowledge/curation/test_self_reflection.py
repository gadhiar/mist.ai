"""Tests for SelfReflectionJob batch internal derivation."""

import json
from datetime import UTC, datetime, timedelta

import pytest

from backend.knowledge.curation.self_reflection import ReflectionResult, SelfReflectionJob
from backend.knowledge.extraction.internal_derivation import InternalKnowledgeDeriver
from backend.knowledge.extraction.signal_detector import SignalDetector
from tests.mocks.neo4j import FakeGraphExecutor, FakeNeo4jConnection
from tests.mocks.ollama import FakeLLM


class FakeEventStore:
    """Minimal event store fake that returns canned turns from get_turns_since.

    Records each `since` it is asked for, so a test can check the window.
    """

    def __init__(self, turns: list[dict] | None = None):
        self._turns = turns or []
        self.since_calls: list[datetime] = []

    def get_turns_since(self, since: datetime) -> list[dict]:
        self.since_calls.append(since)
        return self._turns


class WindowedEventStore:
    """Applies `EventStore.get_turns_since`'s own predicate, `timestamp >= since`,
    compared as ISO-8601 text the way SQLite compares the column.
    """

    def __init__(self, turns: list[dict]) -> None:
        self._turns = turns

    def get_turns_since(self, since: datetime) -> list[dict]:
        bound = since.isoformat()
        return [turn for turn in self._turns if turn["timestamp"] >= bound]


class FakeClock:
    """A settable UTC wall clock for `SelfReflectionJob`."""

    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def _make_turn(
    *,
    utterance: str = "hello",
    response: str = "hi there",
    session_id: str = "sess-1",
    event_id: str = "evt-1",
) -> dict:
    """Build a conversation turn dict matching EventStore output format."""
    return {
        "event_id": event_id,
        "session_id": session_id,
        "user_utterance": utterance,
        "system_response": response,
        "timestamp": datetime.now(UTC).isoformat(),
    }


def _build_job(
    *,
    llm_response: str | None = None,
    event_store: FakeEventStore | None = None,
    connection: FakeNeo4jConnection | None = None,
    clock: FakeClock | None = None,
) -> tuple[SelfReflectionJob, FakeLLM, FakeGraphExecutor]:
    """Build a SelfReflectionJob with test doubles."""
    conn = connection or FakeNeo4jConnection()
    executor = FakeGraphExecutor(conn)
    llm = FakeLLM(default_response=llm_response or '{"operations": []}')
    deriver = InternalKnowledgeDeriver(llm=llm, executor=executor)
    extra = {"clock": clock} if clock is not None else {}
    job = SelfReflectionJob(
        executor=executor,
        internal_deriver=deriver,
        signal_detector=SignalDetector(),
        event_store=event_store,
        **extra,
    )
    return job, llm, executor


class TestProcessRecentEvents:
    @pytest.mark.asyncio
    async def test_calls_deriver_for_each_event(self):
        turns = [
            _make_turn(
                utterance="I prefer concise answers",
                response="Noted.",
                event_id="evt-1",
            ),
            _make_turn(
                utterance="I like when you explain things clearly",
                response="Thank you.",
                event_id="evt-2",
            ),
        ]
        store = FakeEventStore(turns=turns)

        llm_resp = json.dumps(
            {
                "operations": [
                    {
                        "op": "CREATE_PREFERENCE",
                        "id": "pref-concise",
                        "display_name": "Concise Answers",
                        "description": "User prefers concise answers",
                        "confidence": 0.8,
                    },
                ]
            }
        )

        job, llm, _ = _build_job(llm_response=llm_resp, event_store=store)

        result = await job.run(lookback_hours=24)

        assert isinstance(result, ReflectionResult)
        assert result.events_processed == 2
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_skips_turns_without_utterance(self):
        turns = [
            _make_turn(utterance="", response="system boot"),
            _make_turn(utterance="hello", response="hi"),
        ]
        store = FakeEventStore(turns=turns)
        job, llm, _ = _build_job(event_store=store)

        result = await job.run()

        assert result.events_processed == 1

    @pytest.mark.asyncio
    async def test_aggregates_operations_across_events(self):
        turns = [
            _make_turn(
                utterance="I prefer detailed explanations when debugging",
                response="Understood, I will be more detailed.",
                event_id="evt-1",
            ),
        ]
        store = FakeEventStore(turns=turns)

        llm_resp = json.dumps(
            {
                "operations": [
                    {
                        "op": "CREATE_CAPABILITY",
                        "id": "cap-debug",
                        "display_name": "Debugging",
                        "description": "Good at debugging",
                        "confidence": 0.8,
                    },
                    {
                        "op": "CREATE_TRAIT",
                        "id": "trait-analytical",
                        "display_name": "Analytical",
                        "description": "Analytical thinker",
                        "confidence": 0.7,
                    },
                ]
            }
        )

        job, _, _ = _build_job(llm_response=llm_resp, event_store=store)

        result = await job.run()

        assert result.operations_applied == 2


class TestNoEventStore:
    @pytest.mark.asyncio
    async def test_returns_zero_result_when_no_event_store(self):
        conn = FakeNeo4jConnection()
        executor = FakeGraphExecutor(conn)
        llm = FakeLLM()
        deriver = InternalKnowledgeDeriver(llm=llm, executor=executor)

        job = SelfReflectionJob(
            executor=executor,
            internal_deriver=deriver,
            signal_detector=SignalDetector(),
            event_store=None,
        )

        result = await job.run()

        assert result.events_processed == 0
        assert result.operations_applied == 0
        assert result.duration_ms == 0.0
        llm.assert_not_called()


class TestReflectionWindow:
    """The window starts where the previous run's ended, so a deferred run
    covers the hours it was deferred instead of a fixed 24 h.
    """

    _T0 = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)

    @pytest.mark.asyncio
    async def test_first_run_looks_back_24_hours(self):
        store = FakeEventStore()
        job, _, _ = _build_job(event_store=store, clock=FakeClock(self._T0))

        await job.run()

        assert store.since_calls == [self._T0 - timedelta(hours=24)]

    @pytest.mark.asyncio
    async def test_second_run_starts_at_the_first_runs_window_end(self):
        store = FakeEventStore()
        clock = FakeClock(self._T0)
        job, _, _ = _build_job(event_store=store, clock=clock)
        await job.run()

        clock.now = self._T0 + timedelta(hours=29)
        await job.run()

        assert store.since_calls[1] == self._T0

    @pytest.mark.asyncio
    async def test_explicit_lookback_hours_is_used_as_given(self):
        store = FakeEventStore()
        clock = FakeClock(self._T0)
        job, _, _ = _build_job(event_store=store, clock=clock)
        await job.run()

        clock.now = self._T0 + timedelta(hours=29)
        await job.run(lookback_hours=6)

        assert store.since_calls[1] == self._T0 + timedelta(hours=23)

    @pytest.mark.asyncio
    async def test_explicit_lookback_run_does_not_move_the_window(self):
        """A shorter manual window must not make the next scheduled run skip
        the hours between the last scheduled window and the manual one.
        """
        store = FakeEventStore()
        clock = FakeClock(self._T0)
        job, _, _ = _build_job(event_store=store, clock=clock)
        await job.run()
        clock.now = self._T0 + timedelta(hours=29)
        await job.run(lookback_hours=6)

        clock.now = self._T0 + timedelta(hours=30)
        await job.run()

        assert store.since_calls[2] == self._T0

    @pytest.mark.asyncio
    async def test_turn_stamped_at_the_window_end_is_derived_once_by_the_next_run(self):
        """The window is `[since, now)`: a turn committed at `now` before the
        query belongs to the next window, not to both.
        """
        earlier = _make_turn(utterance="earlier turn", event_id="evt-early")
        earlier["timestamp"] = (self._T0 - timedelta(hours=1)).isoformat()
        boundary = _make_turn(utterance="boundary turn", event_id="evt-boundary")
        boundary["timestamp"] = self._T0.isoformat()
        clock = FakeClock(self._T0)
        job, _, _ = _build_job(event_store=WindowedEventStore([earlier, boundary]), clock=clock)

        first = await job.run()
        clock.now = self._T0 + timedelta(hours=24)
        second = await job.run()

        assert (first.events_processed, second.events_processed) == (1, 1)


class TestEmptyEvents:
    @pytest.mark.asyncio
    async def test_returns_zero_when_no_recent_events(self):
        store = FakeEventStore(turns=[])
        job, llm, _ = _build_job(event_store=store)

        result = await job.run()

        assert result.events_processed == 0
        assert result.operations_applied == 0
        assert result.duration_ms > 0
        llm.assert_not_called()


class TestCapabilityTurns:
    """T31: a turn a capability answered alone is not reflected on."""

    @pytest.mark.asyncio
    async def test_a_capability_turn_is_not_derived_or_counted(self):
        capability = _make_turn(utterance="set a timer for ten minutes", event_id="evt-cap")
        capability["answered_by"] = "capability"
        model = _make_turn(utterance="I prefer concise answers", event_id="evt-model")
        model["answered_by"] = "model"
        old = _make_turn(utterance="I like clear explanations", event_id="evt-old")
        old["answered_by"] = None
        store = FakeEventStore(turns=[capability, model, old])
        job, _llm, _ = _build_job(event_store=store)
        detected: list[str] = []
        derived: list[str] = []
        real_detect = job._signal_detector.detect
        real_derive = job._deriver.derive

        def _detect(utterance, **kwargs):
            detected.append(utterance)
            return real_detect(utterance, **kwargs)

        async def _derive(**kwargs):
            derived.append(kwargs["event_id"])
            return await real_derive(**kwargs)

        job._signal_detector.detect = _detect
        job._deriver.derive = _derive

        result = await job.run(lookback_hours=24)

        assert result.events_processed == 2
        assert derived == ["evt-model", "evt-old"]
        assert detected == ["I prefer concise answers", "I like clear explanations"]

    @pytest.mark.asyncio
    async def test_a_turn_without_the_key_is_derived_as_today(self):
        store = FakeEventStore(turns=[_make_turn(utterance="hello", event_id="evt-1")])
        job, _llm, _ = _build_job(event_store=store)

        result = await job.run(lookback_hours=24)

        assert result.events_processed == 1
