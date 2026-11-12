"""The job_id idempotency cache: single-flight under cancellation and at completion.

`_IdempotencyCache` is exercised directly for the event-loop interleavings an
HTTP client cannot place deterministically (a waiter cancelled mid-run, a
duplicate arriving in the tick the run completes), and through the app for
the health-probe ordering.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from backend.extraction_contract.models import (
    CONTRACT_VERSION,
    ExtractionPayload,
    ExtractResponse,
    ResultStamps,
    ScopeOut,
    TimingsMs,
)
from backend.extraction_service.app import (
    ClientGoneError,
    JobCancelledError,
    _IdempotencyCache,
)
from backend.extraction_service.settings import ServiceSettings
from tests.unit.extraction_service.conftest import make_extract_request


def _response(job_id: str = "j") -> ExtractResponse:
    return ExtractResponse(
        contract_version=CONTRACT_VERSION,
        job_id=job_id,
        outcome="extracted",
        scope=ScopeOut(label="unknown", confidence=0.0),
        payload=ExtractionPayload(entities=[], relationships=[]),
        derivation=None,
        stamps=ResultStamps(
            extraction_version="ev",
            model_hash="m",
            prompt_sha256="p",
            llama_cpp_build="b",
            adapter="a",
        ),
        timings_ms=TimingsMs(scope=None, extract=1.0, derive=None, total=1.0),
        attempts=1,
        warnings=[],
    )


class _Run:
    """A factory that counts its runs and holds each one until released."""

    def __init__(self) -> None:
        self.runs = 0
        self.cancelled = 0
        self.release = asyncio.Event()

    async def __call__(self) -> ExtractResponse:
        self.runs += 1
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        return _response()


async def _ticks(n: int = 5) -> None:
    for _ in range(n):
        await asyncio.sleep(0)


@pytest.mark.asyncio
class TestCancellation:
    @pytest.mark.parametrize("cancelled", ["owner", "joiner"])
    async def test_one_cancelled_waiter_does_not_cancel_the_shared_run(self, cancelled):
        # Arrange: two callers share one in-flight run.
        cache = _IdempotencyCache(maxsize=4)
        run = _Run()
        owner = asyncio.create_task(cache.get_or_run("j", run))
        await _ticks()
        joiner = asyncio.create_task(cache.get_or_run("j", run))
        await _ticks()
        gone, stays = (owner, joiner) if cancelled == "owner" else (joiner, owner)

        # Act: one client disconnects, then the run finishes.
        gone.cancel()
        with pytest.raises(asyncio.CancelledError):
            await gone
        run.release.set()
        result = await asyncio.wait_for(stays, timeout=2.0)

        # Assert: the run was never cancelled, and it is stored for the job.
        assert result == _response()
        assert run.runs == 1
        assert run.cancelled == 0
        assert await cache.get_or_run("j", run) == _response()
        assert run.runs == 1

    async def test_a_run_whose_every_waiter_left_is_cancelled_and_nothing_is_kept(self):
        # Arrange
        cache = _IdempotencyCache(maxsize=4)
        run = _Run()
        only = asyncio.create_task(cache.get_or_run("j", run))
        await _ticks()

        # Act: the only client leaves; then the job is resubmitted.
        only.cancel()
        with pytest.raises(asyncio.CancelledError):
            await only
        known_after_leaving = cache.knows("j")
        await _ticks()
        run.release.set()
        again = await asyncio.wait_for(cache.get_or_run("j", run), timeout=2.0)

        # Assert: the first run was cancelled; the resubmission ran on its own.
        assert known_after_leaving is False
        assert run.cancelled == 1
        assert again == _response()
        assert run.runs == 2

    async def test_the_last_of_two_waiters_leaving_cancels_the_run(self):
        cache = _IdempotencyCache(maxsize=4)
        run = _Run()
        owner = asyncio.create_task(cache.get_or_run("j", run))
        await _ticks()
        joiner = asyncio.create_task(cache.get_or_run("j", run))
        await _ticks()

        owner.cancel()
        await _ticks()
        cancelled_with_one_left = run.cancelled
        joiner.cancel()
        await _ticks()

        assert cancelled_with_one_left == 0
        assert run.cancelled == 1
        assert not cache.knows("j")

    @pytest.mark.parametrize("gone", ["owner", "joiner"])
    async def test_a_waiter_whose_client_is_gone_leaves_the_run_to_the_other(self, gone):
        # Arrange: two callers share one run, each watching its own client.
        cache = _IdempotencyCache(maxsize=4)
        run = _Run()
        left = {"owner": asyncio.Event(), "joiner": asyncio.Event()}
        owner = asyncio.create_task(cache.get_or_run("j", run, gone=left["owner"].wait))
        await _ticks()
        joiner = asyncio.create_task(cache.get_or_run("j", run, gone=left["joiner"].wait))
        await _ticks()
        leaver, stays = (owner, joiner) if gone == "owner" else (joiner, owner)

        # Act
        left[gone].set()
        with pytest.raises(ClientGoneError):
            await asyncio.wait_for(leaver, timeout=2.0)
        run.release.set()
        result = await asyncio.wait_for(stays, timeout=2.0)

        # Assert
        assert result == _response()
        assert run.cancelled == 0
        assert run.runs == 1

    async def test_the_only_waiter_whose_client_is_gone_cancels_the_run(self):
        cache = _IdempotencyCache(maxsize=4)
        run = _Run()
        left = asyncio.Event()
        only = asyncio.create_task(cache.get_or_run("j", run, gone=left.wait))
        await _ticks()

        left.set()
        with pytest.raises(ClientGoneError):
            await asyncio.wait_for(only, timeout=2.0)
        await _ticks()

        assert run.cancelled == 1
        assert not cache.knows("j")

    async def test_a_run_that_finishes_answers_a_waiter_whose_client_stayed(self):
        cache = _IdempotencyCache(maxsize=4)
        run = _Run()
        run.release.set()

        result = await cache.get_or_run("j", run, gone=asyncio.Event().wait)

        assert result == _response()


@pytest.mark.asyncio
class TestCancel:
    async def test_cancelling_a_run_in_flight_raises_for_every_waiter(self):
        # Arrange
        cache = _IdempotencyCache(maxsize=4)
        run = _Run()
        owner = asyncio.create_task(cache.get_or_run("j", run))
        await _ticks()
        joiner = asyncio.create_task(cache.get_or_run("j", run))
        await _ticks()

        # Act
        outcome = cache.cancel("j")

        # Assert
        assert outcome == "cancelled"
        for waiter in (owner, joiner):
            with pytest.raises(JobCancelledError, match="'j'"):
                await asyncio.wait_for(waiter, timeout=2.0)
        assert run.cancelled == 1
        assert not cache.knows("j")

    async def test_a_cancelled_job_is_not_stored_and_runs_when_sent_again(self):
        cache = _IdempotencyCache(maxsize=4)
        run = _Run()
        owner = asyncio.create_task(cache.get_or_run("j", run))
        await _ticks()
        cache.cancel("j")
        with pytest.raises(JobCancelledError):
            await owner

        run.release.set()
        again = await cache.get_or_run("j", run)

        assert again == _response()
        assert run.runs == 2

    async def test_a_job_sent_again_before_the_cancelled_run_has_ended_gets_its_own_run(self):
        # Arrange
        cache = _IdempotencyCache(maxsize=4)
        run = _Run()
        owner = asyncio.create_task(cache.get_or_run("j", run))
        await _ticks()

        # Act: cancel and resubmit with no tick between them, so the first
        # run has not yet been told it is cancelled.
        cache.cancel("j")
        again = asyncio.create_task(cache.get_or_run("j", run))
        with pytest.raises(JobCancelledError):
            await owner
        await _ticks()
        still_in_flight = cache.knows("j")
        run.release.set()
        result = await asyncio.wait_for(again, timeout=2.0)

        # Assert: the first run's end neither un-registered nor answered the second.
        assert still_in_flight is True
        assert result == _response()
        assert run.runs == 2
        assert run.cancelled == 1

    async def test_a_cancelled_run_that_returns_anyway_is_not_stored(self):
        # Arrange: a run that swallows its cancellation and returns a result.
        cache = _IdempotencyCache(maxsize=4)
        release = asyncio.Event()
        runs = 0

        async def stubborn() -> ExtractResponse:
            nonlocal runs
            runs += 1
            with contextlib.suppress(asyncio.CancelledError):
                await release.wait()
            return _response()

        owner = asyncio.create_task(cache.get_or_run("j", stubborn))
        await _ticks()

        # Act
        cache.cancel("j")
        with pytest.raises(JobCancelledError):
            await asyncio.wait_for(owner, timeout=2.0)
        await _ticks()
        release.set()
        again = await cache.get_or_run("j", stubborn)

        # Assert
        assert again == _response()
        assert runs == 2

    async def test_a_finished_job_is_reported_finished_and_stays_stored(self):
        cache = _IdempotencyCache(maxsize=4)
        run = _Run()
        run.release.set()
        await cache.get_or_run("j", run)

        outcome = cache.cancel("j")

        assert outcome == "finished"
        assert await cache.get_or_run("j", run) == _response()
        assert run.runs == 1

    async def test_a_job_never_seen_is_reported_unknown(self):
        assert _IdempotencyCache(maxsize=4).cancel("j") == "unknown"

    async def test_a_finished_job_the_cache_did_not_retain_is_reported_unknown(self):
        cache = _IdempotencyCache(maxsize=0)
        run = _Run()
        run.release.set()
        await cache.get_or_run("j", run)

        assert cache.cancel("j") == "unknown"


class _YieldingLock(asyncio.Lock):
    """A lock that yields to the event loop for a few ticks on every acquire.

    The previous cache design un-registered the in-flight run and stored its
    result in two separately locked sections; an uncontended `asyncio.Lock`
    does not yield on acquire, which hid the gap between them. Several ticks
    per acquire let a duplicate's lookup land inside it. The current design
    stores the result and un-registers the run in one synchronous callback
    and takes no lock, so it never reads this attribute.
    """

    async def acquire(self) -> bool:
        for _ in range(3):
            await asyncio.sleep(0)
        return await super().acquire()


@pytest.mark.asyncio
class TestCompletionBoundary:
    async def test_a_duplicate_in_the_tick_the_run_completes_does_not_start_a_second_run(
        self,
    ):
        # Arrange
        cache = _IdempotencyCache(maxsize=4)
        cache._lock = _YieldingLock()
        run = _Run()
        first = asyncio.create_task(cache.get_or_run("j", run))
        await _ticks()

        # Act: release the run, and send a duplicate on every tick until the
        # first caller has its answer.
        run.release.set()
        duplicates = []
        while not first.done():
            duplicates.append(asyncio.create_task(cache.get_or_run("j", run)))
            await asyncio.sleep(0)
        results = await asyncio.gather(first, *duplicates)

        # Assert
        assert run.runs == 1
        assert all(r == _response() for r in results)
        assert len(duplicates) >= 2  # the loop really did span the completion


class TestCacheSize:
    def test_a_negative_cache_size_is_refused_by_the_settings(self):
        with pytest.raises(ValueError, match="idempotency_cache_size"):
            ServiceSettings(
                llm_base_url="http://x", model_hash="m", model_file="f", idempotency_cache_size=-1
            )

    def test_a_negative_cache_size_from_the_environment_is_refused(self, monkeypatch):
        monkeypatch.setenv("EXTRACTION_IDEMPOTENCY_CACHE_SIZE", "-3")

        with pytest.raises(ValueError, match="idempotency_cache_size"):
            ServiceSettings.from_env()

    def test_a_negative_size_is_refused_by_the_cache_itself(self):
        with pytest.raises(ValueError, match="maxsize"):
            _IdempotencyCache(maxsize=-1)

    @pytest.mark.asyncio
    async def test_size_zero_retains_nothing_but_still_single_flights(self):
        # Arrange
        cache = _IdempotencyCache(maxsize=0)
        run = _Run()

        # Act: two concurrent duplicates, then a later resubmission.
        a = asyncio.create_task(cache.get_or_run("j", run))
        b = asyncio.create_task(cache.get_or_run("j", run))
        await _ticks()
        run.release.set()
        concurrent = await asyncio.gather(a, b)
        runs_after_concurrent = run.runs
        later = await cache.get_or_run("j", run)

        # Assert
        assert concurrent == [_response(), _response()]
        assert runs_after_concurrent == 1
        assert later == _response()
        assert run.runs == 2  # nothing retained

    @pytest.mark.asyncio
    async def test_the_lru_keeps_the_newest_maxsize_jobs(self):
        cache = _IdempotencyCache(maxsize=2)
        run = _Run()
        run.release.set()

        for job in ("a", "b", "c"):
            await cache.get_or_run(job, run)
        await cache.get_or_run("c", run)
        await cache.get_or_run("b", run)
        await cache.get_or_run("a", run)

        # a, b, c ran; c and b were retained; a was evicted and ran again.
        assert run.runs == 4


@pytest.mark.asyncio
class TestHealthProbeOrdering:
    async def test_a_completed_job_is_served_from_the_cache_while_the_model_loads(
        self, client, fake_llama_state
    ):
        # Arrange: the job completes once.
        fake_llama_state.chat_responses = [
            '{"scope": "unknown", "confidence": 0.0}',
            '{"entities": [], "relationships": []}',
        ]
        body = make_extract_request(job_id="job-cached")
        first = await client.post("/v1/extract", json=body)
        assert first.status_code == 200

        # Act: llama-server starts reloading; the dispatcher resends the job.
        fake_llama_state.health_status = 503
        second = await client.post("/v1/extract", json=body)

        # Assert
        assert second.status_code == 200, second.text
        assert second.json() == first.json()
        assert len(fake_llama_state.chat_requests) == 2

    async def test_a_duplicate_of_an_in_flight_job_joins_it_while_the_model_loads(
        self, client, fake_llama_state, blocked_chat
    ):
        # Arrange: a job in flight, held inside its first LLM call.
        fake_llama_state.chat_responses = [
            '{"scope": "unknown", "confidence": 0.0}',
            '{"entities": [], "relationships": []}',
        ]
        body = make_extract_request(job_id="job-in-flight")
        first = asyncio.create_task(client.post("/v1/extract", json=body))
        while not fake_llama_state.chat_requests:
            await asyncio.sleep(0)

        # Act: the health turns to loading; a duplicate arrives while the
        # first run is still held. A duplicate that did not join would be
        # refused with a 503 instead of waiting on the held run.
        fake_llama_state.health_status = 503
        second_task = asyncio.create_task(client.post("/v1/extract", json=body))
        await _ticks(50)
        assert not second_task.done()
        blocked_chat.set()
        second = await second_task
        first_response = await first

        # Assert
        assert first_response.status_code == 200
        assert second.status_code == 200, second.text
        assert second.json() == first_response.json()
        assert len(fake_llama_state.chat_requests) == 2

    async def test_a_new_job_still_gets_model_loading(self, client, fake_llama_state):
        fake_llama_state.health_status = 503

        response = await client.post("/v1/extract", json=make_extract_request(job_id="new"))

        assert response.status_code == 503
        assert response.json()["error"]["code"] == "model_loading"
