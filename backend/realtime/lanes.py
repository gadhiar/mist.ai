"""Inference lanes: one worker thread per engine, so model calls never block the event loop.

Every model call (VAD, end of turn, STT, TTS) is blocking and some hold the GPU for hundreds of
milliseconds. A lane runs them on its own thread, one at a time in arrival order, which also
serializes sessions that share an engine instead of letting them race on the GPU. Cancellation
is cooperative: a queued job whose `CancelToken` is set never starts, and a running job cannot
be interrupted from outside, so its token is set for the engine to check between chunks and its
result is discarded when it returns.

Who cancelled decides what the caller sees. `asyncio.CancelledError` means only that the awaiting
task itself was cancelled. A job whose token another holder set (a barge-in, a GPU release), and
a decode cut by its stream's `close()`, fail with `LaneJobCancelled`: raising `CancelledError`
into a task nobody cancelled would make it look cancelled to `asyncio.timeout`, `TaskGroup` and
every `except CancelledError` above it.
"""

import asyncio
import logging
import threading
from collections import deque
from collections.abc import Callable
from typing import TypeVar

from .errors import RealtimeError, SpeechEngineError
from .speech import CancelToken

logger = logging.getLogger(__name__)

T = TypeVar("T")

_MAX_WORKER_DEATHS = 2


# N818 asks for an `Error` suffix. This is a cancellation signal, not a failure, and it is named
# for what happened to the job, as `asyncio.CancelledError`'s counterpart.
class LaneJobCancelled(RealtimeError):  # noqa: N818
    """A lane job stopped because another holder cancelled it, not because its caller was.

    Raised by `InferenceLane.run` when the job's token was set by someone other than the
    awaiting task (queued, running, or already set at submission), and by a transcript stream's
    decode cut by `close()`. It is not a `SpeechEngineError`: the engine did not fail, the work
    was no longer wanted.
    """


class _Job:
    """A queued call, the loop it was submitted from and the future its caller awaits."""

    __slots__ = ("fn", "args", "token", "loop", "future")

    def __init__(
        self,
        fn: Callable[..., object],
        args: tuple[object, ...],
        token: CancelToken,
        loop: asyncio.AbstractEventLoop,
        future: "asyncio.Future[object]",
    ) -> None:
        self.fn = fn
        self.args = args
        self.token = token
        self.loop = loop
        self.future = future


class InferenceLane:
    """A FIFO queue of blocking engine calls served by one worker thread.

    A job that raises an exception fails only itself: the caller receives `SpeechEngineError`
    (the engine's own `SpeechEngineError` unchanged, any other exception wrapped with it as the
    cause), because the turn controller handles engine failures by that one type. A job that
    ends the worker thread itself (`SystemExit` or another exception outside `Exception`) fails
    with every job queued behind it; the lane starts a replacement worker once, and after a
    second death it is unhealthy and refuses new jobs at once, so its engine can be reported
    unavailable instead of hanging callers.
    """

    def __init__(self, name: str, *, max_queue: int = 64) -> None:
        if max_queue < 1:
            raise ValueError(f"max_queue must be at least 1, got {max_queue}")
        self._name = name
        self._max_queue = max_queue
        self._cond = threading.Condition()
        self._queue: deque[_Job] = deque()
        self._deaths = 0
        self._closed = False
        self._worker = self._start_worker()

    async def run(self, fn: Callable[..., T], *args: object, token: CancelToken) -> T:
        """Run `fn(*args)` on the lane's worker thread and return its result.

        Cancelling the awaiting task sets `token`, removes a job that has not started, and makes
        the worker discard the result of one that has.

        Raises:
            asyncio.CancelledError: The awaiting task itself was cancelled; the job's result, if
                any, is discarded.
            LaneJobCancelled: Another holder set `token` before submission, while the job was
                queued or while it ran, or the job was a decode cut by its stream's `close()`;
                the job's result, if any, is discarded.
            SpeechEngineError: The job failed, its worker thread died, the queue is full, or
                the lane is shut down or unhealthy.
        """
        if token.is_set():
            raise self._cancelled("before submission")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[object] = loop.create_future()
        job = _Job(fn, args, token, loop, future)
        with self._cond:
            if self._closed:
                raise SpeechEngineError(f"inference lane {self._name!r} is shut down")
            if self._deaths >= _MAX_WORKER_DEATHS:
                raise SpeechEngineError(
                    f"inference lane {self._name!r} is unavailable: its worker died twice"
                )
            if len(self._queue) >= self._max_queue:
                raise SpeechEngineError(
                    f"inference lane {self._name!r} queue is full ({self._max_queue} jobs)"
                )
            self._queue.append(job)
            self._cond.notify()
        try:
            return await future  # type: ignore[return-value]
        except asyncio.CancelledError:
            token.set()
            with self._cond:
                if job in self._queue:
                    self._queue.remove(job)
            raise

    def depth(self) -> int:
        """The number of jobs waiting behind the running one."""
        with self._cond:
            return len(self._queue)

    def healthy(self) -> bool:
        """False once the worker thread has died a second time."""
        with self._cond:
            return self._deaths < _MAX_WORKER_DEATHS

    def shutdown(self, timeout_s: float = 5.0) -> None:
        """Fail every waiting job, let the running one finish, and join the worker thread."""
        with self._cond:
            self._closed = True
            pending = list(self._queue)
            self._queue.clear()
            self._cond.notify_all()
            worker = self._worker
        error = SpeechEngineError(f"inference lane {self._name!r} is shut down")
        for job in pending:
            self._settle(job, error=error)
        if worker is threading.current_thread():
            return
        worker.join(timeout_s)
        if worker.is_alive():
            logger.warning(
                "Inference lane %r: worker still running a job %.1f s after shutdown",
                self._name,
                timeout_s,
            )

    def _start_worker(self) -> threading.Thread:
        worker = threading.Thread(
            target=self._work, name=f"inference-lane-{self._name}", daemon=True
        )
        worker.start()
        return worker

    def _work(self) -> None:
        current: _Job | None = None
        exited_cleanly = False
        try:
            while True:
                with self._cond:
                    while not self._queue and not self._closed:
                        self._cond.wait()
                    if not self._queue:
                        exited_cleanly = True
                        return
                    current = self._queue.popleft()
                self._execute(current)
                current = None
        finally:
            if not exited_cleanly:
                self._on_worker_death(current)

    def _execute(self, job: _Job) -> None:
        if job.token.is_set():
            self._settle(job, error=self._cancelled("before it started"))
            return
        try:
            result = job.fn(*job.args)
        except (SpeechEngineError, LaneJobCancelled) as exc:
            self._settle(job, error=exc)
            return
        # The job's exception is not handled here but forwarded: it fails the awaiting caller as
        # `SpeechEngineError`, the one type the controller handles engine failures by, with the
        # original as its cause. Letting it propagate instead would end the worker thread.
        except Exception as exc:  # noqa: BLE001 -- forwarded to the awaiting caller
            error = SpeechEngineError(f"inference lane {self._name!r}: job failed: {exc!r}")
            error.__cause__ = exc
            self._settle(job, error=error)
            return
        if job.token.is_set():
            self._settle(job, error=self._cancelled("while it ran"))
        else:
            self._settle(job, result=result)

    def _cancelled(self, when: str) -> LaneJobCancelled:
        return LaneJobCancelled(f"inference lane {self._name!r}: job cancelled {when}")

    def _on_worker_death(self, current: _Job | None) -> None:
        with self._cond:
            self._deaths += 1
            failed = ([current] if current is not None else []) + list(self._queue)
            self._queue.clear()
            restart = self._deaths < _MAX_WORKER_DEATHS and not self._closed
            if restart:
                self._worker = self._start_worker()
        logger.error(
            "Inference lane %r: worker thread died (death %d of %d); %s",
            self._name,
            self._deaths,
            _MAX_WORKER_DEATHS,
            "restarted" if restart else "lane unavailable",
        )
        error = SpeechEngineError(f"inference lane {self._name!r}: worker thread died")
        for job in failed:
            self._settle(job, error=error)

    @staticmethod
    def _settle(
        job: _Job,
        *,
        result: object = None,
        error: BaseException | None = None,
    ) -> None:
        def resolve() -> None:
            # A future already done is one whose awaiting task was cancelled.
            if job.future.done():
                return
            if error is not None:
                job.future.set_exception(error)
            else:
                job.future.set_result(result)

        try:
            job.loop.call_soon_threadsafe(resolve)
        except RuntimeError:
            # The caller's event loop is closed, so nobody is left awaiting this job.
            logger.debug("Inference lane job settled after its event loop closed")
