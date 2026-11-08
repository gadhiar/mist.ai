"""Tests for `InferenceLane`, the single worker thread every model call runs on.

A lane is what keeps inference off the event loop and what lets a barge-in stop speech: a job
cancelled before it starts must never reach the GPU, and one cancelled mid-run must not deliver
audio for a turn that is gone. A lane whose worker dies must fail its callers instead of leaving
them awaiting forever. Jobs here block on `threading.Event`s the test controls, so ordering is
explicit and nothing depends on timing.
"""

import asyncio
import logging
import threading
from collections.abc import Iterator

import numpy as np
import pytest

from backend.realtime.errors import RealtimeError, SpeechEngineError
from backend.realtime.lanes import InferenceLane, LaneJobCancelled
from backend.realtime.speech import CancelToken
from tests.fakes.speech import ScriptedTranscriber

WAIT_S = 5.0


@pytest.fixture
def lane() -> Iterator[InferenceLane]:
    lane = InferenceLane("test")
    yield lane
    lane.shutdown(timeout_s=WAIT_S)


async def _wait_for(event: threading.Event) -> None:
    assert await asyncio.to_thread(event.wait, WAIT_S), "the job never reached this point"


class _BlockingJob:
    """A job that reports when it starts and then waits until the test releases it."""

    def __init__(self, result: object = "done") -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.result = result

    def __call__(self) -> object:
        self.started.set()
        self.release.wait(WAIT_S)
        self.finished.set()
        return self.result


def _lane_threads(name: str) -> list[threading.Thread]:
    return [thread for thread in threading.enumerate() if thread.name == f"inference-lane-{name}"]


class TestExecution:
    @pytest.mark.asyncio
    async def test_returns_the_result_of_the_job_with_its_arguments(self, lane):
        result = await lane.run(lambda a, b: a * b, 6, 7, token=CancelToken())

        assert result == 42

    @pytest.mark.asyncio
    async def test_jobs_never_run_on_the_event_loop_thread(self, lane):
        loop_thread = threading.get_ident()

        job_thread = await lane.run(threading.get_ident, token=CancelToken())

        assert job_thread != loop_thread

    @pytest.mark.asyncio
    async def test_jobs_run_in_submission_order_on_one_thread(self, lane):
        order: list[tuple[int, int]] = []
        gate = _BlockingJob()
        first = asyncio.ensure_future(lane.run(gate, token=CancelToken()))
        await _wait_for(gate.started)
        rest = [
            asyncio.ensure_future(
                lane.run(lambda i=i: order.append((i, threading.get_ident())), token=CancelToken())
            )
            for i in range(3)
        ]

        gate.release.set()
        await asyncio.gather(first, *rest)

        assert [i for i, _ in order] == [0, 1, 2]
        assert len({ident for _, ident in order}) == 1

    @pytest.mark.asyncio
    async def test_a_speech_engine_error_reaches_the_caller_unchanged(self, lane):
        error = SpeechEngineError("decode failed")

        def fail() -> None:
            raise error

        with pytest.raises(SpeechEngineError) as caught:
            await lane.run(fail, token=CancelToken())

        assert caught.value is error

    @pytest.mark.asyncio
    async def test_a_job_failing_with_speech_engine_error_leaves_the_same_worker_serving(
        self, lane
    ):
        def fail() -> None:
            raise SpeechEngineError("bad input")

        worker_before = await lane.run(threading.get_ident, token=CancelToken())
        with pytest.raises(SpeechEngineError):
            await lane.run(fail, token=CancelToken())

        assert await lane.run(threading.get_ident, token=CancelToken()) == worker_before
        assert lane.healthy()


class TestCancellation:
    """Who cancelled decides what the caller sees.

    `asyncio.CancelledError` means only that the awaiting task itself was cancelled. A job whose
    token another holder set (a barge-in, a GPU release), or a decode cut by its stream's
    `close()`, fails with `LaneJobCancelled`, so the caller's task is never told it was cancelled
    when it was not.
    """

    def test_lane_job_cancelled_is_a_realtime_error_apart_from_engine_failures(self):
        assert issubclass(LaneJobCancelled, RealtimeError)
        assert not issubclass(LaneJobCancelled, SpeechEngineError)
        assert not issubclass(LaneJobCancelled, asyncio.CancelledError)

    @pytest.mark.asyncio
    async def test_a_job_whose_token_another_holder_set_before_it_starts_never_runs(self, lane):
        gate = _BlockingJob()
        ran = threading.Event()
        token = CancelToken()
        first = asyncio.ensure_future(lane.run(gate, token=CancelToken()))
        await _wait_for(gate.started)
        queued = asyncio.ensure_future(lane.run(ran.set, token=token))
        await asyncio.sleep(0)

        token.set()
        gate.release.set()
        await first

        with pytest.raises(LaneJobCancelled):
            await queued
        assert await lane.run(lambda: "after", token=CancelToken()) == "after"
        assert not ran.is_set()

    @pytest.mark.asyncio
    async def test_a_job_whose_caller_is_cancelled_before_it_starts_never_runs(self, lane):
        gate = _BlockingJob()
        ran = threading.Event()
        token = CancelToken()
        first = asyncio.ensure_future(lane.run(gate, token=CancelToken()))
        await _wait_for(gate.started)
        queued = asyncio.ensure_future(lane.run(ran.set, token=token))
        await asyncio.sleep(0)

        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        gate.release.set()
        await first

        assert await lane.run(lambda: "after", token=CancelToken()) == "after"
        assert not ran.is_set()
        assert token.is_set()

    @pytest.mark.asyncio
    async def test_a_token_already_set_at_submission_fails_at_once_without_running(self, lane):
        ran = threading.Event()
        token = CancelToken()
        token.set()

        with pytest.raises(LaneJobCancelled):
            await lane.run(ran.set, token=token)

        assert await lane.run(lambda: "after", token=CancelToken()) == "after"
        assert not ran.is_set()

    @pytest.mark.asyncio
    async def test_a_job_cancelled_while_running_sets_its_token_and_discards_its_result(self, lane):
        job = _BlockingJob(result="stale audio")
        token = CancelToken()
        running = asyncio.ensure_future(lane.run(job, token=token))
        await _wait_for(job.started)

        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running

        assert token.is_set()
        job.release.set()
        assert await lane.run(lambda: "next", token=CancelToken()) == "next"
        assert job.finished.is_set()

    @pytest.mark.asyncio
    async def test_a_token_another_holder_set_while_running_discards_the_result(self, lane):
        job = _BlockingJob(result="stale audio")
        token = CancelToken()
        running = asyncio.ensure_future(lane.run(job, token=token))
        await _wait_for(job.started)

        token.set()
        job.release.set()

        with pytest.raises(LaneJobCancelled):
            await running

    @pytest.mark.asyncio
    async def test_a_decode_cut_by_close_fails_with_lane_job_cancelled_and_the_worker_serves_on(
        self, lane
    ):
        started, release = threading.Event(), threading.Event()

        def hold() -> None:
            started.set()
            release.wait(WAIT_S)

        stream = ScriptedTranscriber(["turn on the lights"], on_decode=hold).open_stream()
        stream.push(np.zeros(6400, dtype=np.float32))
        worker_before = await lane.run(threading.get_ident, token=CancelToken())
        decoding = asyncio.ensure_future(
            lane.run(stream.decode_partial, CancelToken(), token=CancelToken())
        )
        await _wait_for(started)

        stream.close()
        release.set()

        with pytest.raises(LaneJobCancelled):
            await decoding
        assert stream.partials == []
        assert await lane.run(threading.get_ident, token=CancelToken()) == worker_before
        assert lane.healthy()

    @pytest.mark.asyncio
    async def test_a_decode_queued_behind_a_close_fails_with_lane_job_cancelled(self, lane):
        stream = ScriptedTranscriber(["hi"]).open_stream()
        stream.close()

        with pytest.raises(LaneJobCancelled):
            await lane.run(stream.decode_final, CancelToken(), token=CancelToken())
        assert lane.healthy()


class TestDepth:
    @pytest.mark.asyncio
    async def test_depth_reports_the_jobs_waiting_behind_the_running_one(self, lane):
        gate = _BlockingJob()
        first = asyncio.ensure_future(lane.run(gate, token=CancelToken()))
        await _wait_for(gate.started)
        waiting = [asyncio.ensure_future(lane.run(int, token=CancelToken())) for _ in range(3)]
        await asyncio.sleep(0)

        depth_while_blocked = lane.depth()
        gate.release.set()
        await asyncio.gather(first, *waiting)

        assert depth_while_blocked == 3
        assert lane.depth() == 0

    @pytest.mark.asyncio
    async def test_a_cancelled_waiting_job_leaves_the_queue_at_once(self, lane):
        gate = _BlockingJob()
        first = asyncio.ensure_future(lane.run(gate, token=CancelToken()))
        await _wait_for(gate.started)
        waiting = asyncio.ensure_future(lane.run(int, token=CancelToken()))
        await asyncio.sleep(0)

        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting

        assert lane.depth() == 0
        gate.release.set()
        await first

    @pytest.mark.asyncio
    async def test_a_full_queue_refuses_the_job_with_speech_engine_error(self):
        lane = InferenceLane("full", max_queue=1)
        gate = _BlockingJob()
        try:
            first = asyncio.ensure_future(lane.run(gate, token=CancelToken()))
            await _wait_for(gate.started)
            second = asyncio.ensure_future(lane.run(int, token=CancelToken()))
            await asyncio.sleep(0)

            with pytest.raises(SpeechEngineError, match="full"):
                await lane.run(int, token=CancelToken())

            gate.release.set()
            await asyncio.gather(first, second)
        finally:
            lane.shutdown(timeout_s=WAIT_S)


class TestShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_joins_the_worker_thread(self):
        lane = InferenceLane("joined")
        await lane.run(int, token=CancelToken())
        assert len(_lane_threads("joined")) == 1

        lane.shutdown(timeout_s=WAIT_S)

        assert _lane_threads("joined") == []

    @pytest.mark.asyncio
    async def test_shutdown_fails_waiting_jobs_and_refuses_new_ones(self):
        lane = InferenceLane("closing")
        gate = _BlockingJob()
        first = asyncio.ensure_future(lane.run(gate, token=CancelToken()))
        await _wait_for(gate.started)
        waiting = asyncio.ensure_future(lane.run(int, token=CancelToken()))
        await asyncio.sleep(0)

        closer = asyncio.ensure_future(asyncio.to_thread(lane.shutdown, WAIT_S))
        with pytest.raises(SpeechEngineError, match="shut down"):
            await waiting
        gate.release.set()
        await closer
        await first

        with pytest.raises(SpeechEngineError, match="shut down"):
            await lane.run(int, token=CancelToken())


def _kill_worker() -> None:
    raise SystemExit("worker killed by its job")


# The worker's death is the behaviour under test. pytest reports the exception that ends the
# thread (`SystemExit`, or a job's non-engine exception) as an unhandled thread exception.
@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
class TestWorkerDeath:
    @pytest.mark.asyncio
    async def test_a_job_that_kills_the_worker_fails_it_and_every_pending_job(self, lane):
        gate = threading.Event()
        started = threading.Event()

        def die_when_released() -> None:
            started.set()
            gate.wait(WAIT_S)
            _kill_worker()

        dying = asyncio.ensure_future(lane.run(die_when_released, token=CancelToken()))
        await _wait_for(started)
        pending = [asyncio.ensure_future(lane.run(int, token=CancelToken())) for _ in range(2)]
        await asyncio.sleep(0)

        gate.set()
        results = await asyncio.gather(dying, *pending, return_exceptions=True)

        assert all(isinstance(result, SpeechEngineError) for result in results)
        assert lane.depth() == 0

    @pytest.mark.asyncio
    async def test_the_lane_restarts_its_worker_once_after_a_death(self, lane):
        before = await lane.run(threading.get_ident, token=CancelToken())
        with pytest.raises(SpeechEngineError):
            await lane.run(_kill_worker, token=CancelToken())

        after = await lane.run(threading.get_ident, token=CancelToken())

        assert lane.healthy()
        assert after != before

    @pytest.mark.asyncio
    async def test_a_second_death_makes_the_lane_unhealthy_and_new_jobs_fail_at_once(self, lane):
        with pytest.raises(SpeechEngineError):
            await lane.run(_kill_worker, token=CancelToken())
        with pytest.raises(SpeechEngineError):
            await lane.run(_kill_worker, token=CancelToken())
        ran = threading.Event()

        with pytest.raises(SpeechEngineError, match="unavailable"):
            await lane.run(ran.set, token=CancelToken())

        assert not lane.healthy()
        assert not ran.is_set()

    @pytest.mark.asyncio
    async def test_a_non_engine_exception_ends_the_worker_and_logs_its_traceback(
        self, lane, caplog
    ):
        # Adapters convert library errors to `SpeechEngineError`, so any other exception that
        # reaches the lane is a bug: it ends the worker instead of being wrapped and hidden.
        cause = RuntimeError("CUDA out of memory")

        def fail() -> None:
            raise cause

        worker_before = await lane.run(threading.get_ident, token=CancelToken())
        with (
            caplog.at_level(logging.ERROR, logger="backend.realtime.lanes"),
            pytest.raises(SpeechEngineError, match="worker thread died") as caught,
        ):
            await lane.run(fail, token=CancelToken())

        assert caught.value.__cause__ is None
        deaths = [record for record in caplog.records if "worker thread died" in record.message]
        assert len(deaths) == 1
        assert deaths[0].exc_info is not None
        assert deaths[0].exc_info[1] is cause
        assert lane.healthy()
        assert await lane.run(threading.get_ident, token=CancelToken()) != worker_before

    @pytest.mark.asyncio
    async def test_two_non_engine_exceptions_make_the_lane_unhealthy(self, lane):
        def fail() -> None:
            raise ValueError("bad input")

        with pytest.raises(SpeechEngineError, match="worker thread died"):
            await lane.run(fail, token=CancelToken())
        with pytest.raises(SpeechEngineError, match="worker thread died"):
            await lane.run(fail, token=CancelToken())

        with pytest.raises(SpeechEngineError, match="unavailable"):
            await lane.run(int, token=CancelToken())
        assert not lane.healthy()
