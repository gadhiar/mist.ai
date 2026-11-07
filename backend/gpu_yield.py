"""Gaming mode: yield the GPU to a game while a host-written marker exists.

MIST shares the one GPU of the host with games. `instance.py gaming-mode on`
writes `<MIST_OPS_DIR>/gpu-yielded` (JSON `{"v": 1, "since": ..., "by": ...}`,
written and removed only by the host CLI, on a mount the backend reads
read-only), waits two polls, then stops mist-llm. `GpuYield` watches that
marker. When it appears, the backend releases Whisper; when it goes, the
backend reloads Whisper and accepts turns again. Meanwhile the API, the event
log, the extraction backlog and backups keep running.

Only the marker's existence matters. Its content is for the CLI; an
unparsable marker counts as present, and so does one whose existence cannot be
checked, because yielding when unsure costs a refused turn, while not yielding
reports a stopped LLM as a crash.

States, read by `/health`'s llm check, the WebSocket and the curation
scheduler:
- `normal`: no marker, Whisper loaded.
- `yielded`: marker present. Set before `on_yield` runs, and kept until
  `on_resume` returns, so no turn is accepted while Whisper reloads.
- `resume_failed`: the marker went but `on_resume` raised. Retried after a
  delay that starts at `poll_s` and doubles up to 60 s; the marker poll keeps
  `poll_s`. A marker that reappears meanwhile sets `yielded` again.

Both callbacks run through `asyncio.to_thread`, never on the event loop's
thread, and one at a time, because `run_forever` awaits each before polling
again.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

YieldState = Literal["normal", "yielded", "resume_failed"]

MARKER_NAME = "gpu-yielded"
POLL_ENV = "MIST_GPU_YIELD_POLL_S"
DEFAULT_POLL_S = 2.0
MAX_RETRY_DELAY_S = 60.0

# The WebSocket's answer to an `audio` or `text` message while yielded, sent in
# the existing ADR-017 error envelope.
YIELDED_MESSAGE = "MIST has yielded the GPU to a game and will answer when gaming mode ends."
RESUME_FAILED_MESSAGE = "MIST could not reload her speech model after gaming mode and is retrying."


def turn_refusal(message: str) -> dict:
    """The ADR-017 error envelope for a turn refused because the GPU is yielded.

    Sent both by the WebSocket gate and by the voice processor, for a voice
    turn that passed the gate just before Whisper was released, so the client
    sees one shape whichever side refused the turn.
    """
    return {
        "type": "error",
        "kind": "server",
        "message": message,
        "retriable": True,
        "context": None,
    }


def gpu_yield_poll_s() -> float:
    """The marker poll interval in seconds, from `MIST_GPU_YIELD_POLL_S`.

    A value that is not a positive number is logged and replaced by the
    default rather than raised, because this is read while the backend starts
    and a typo here should not stop it.
    """
    raw = (os.environ.get(POLL_ENV) or "").strip()
    if not raw:
        return DEFAULT_POLL_S
    try:
        value = float(raw)
    except ValueError:
        value = 0.0
    if value > 0:
        return value
    logger.error("%s=%r is not a positive number; using %s", POLL_ENV, raw, DEFAULT_POLL_S)
    return DEFAULT_POLL_S


class GpuYield:
    """Watch the gaming-mode marker and release or reload the GPU models.

    A live state holder, built once by the server's lifespan; tests build
    their own over a temporary marker.
    """

    def __init__(
        self,
        marker_path: Path,
        on_yield: Callable[[], None],
        on_resume: Callable[[], None],
        *,
        poll_s: float,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Read the marker once and start `yielded` or `normal`, calling neither callback.

        A backend that boots while yielded skips the GPU models
        (`ModelManager.load_all_models(yielded=True)`), so there is nothing to
        release at construction.

        Args:
            marker_path: `<MIST_OPS_DIR>/gpu-yielded`.
            on_yield: Releases the GPU models; must be idempotent.
            on_resume: Reloads them; raising `RuntimeError` (which
                `torch.cuda.OutOfMemoryError` subclasses) or `OSError` sets
                `resume_failed`.
            poll_s: Seconds between marker polls, and the first retry delay.
            sleep: Awaitable sleep between polls.
            clock: Monotonic seconds, for the retry schedule.
        """
        self._marker_path = marker_path
        self._on_yield = on_yield
        self._on_resume = on_resume
        self._poll_s = poll_s
        self._sleep = sleep
        self._clock = clock
        self._stat_error_logged = False
        self._retry_delay = poll_s
        self._next_retry_at = 0.0
        self._failed_attempts = 0
        self._state: YieldState = "yielded" if self._marker_present() else "normal"

    def state(self) -> YieldState:
        """The current state; safe to call from any thread."""
        return self._state

    def is_yielded(self) -> bool:
        """True in `yielded` and `resume_failed`: MIST cannot take a turn."""
        return self._state != "normal"

    async def run_forever(self) -> None:
        """Poll the marker now, then every `poll_s`, until cancelled."""
        while True:
            await self.poll_once()
            await self._sleep(self._poll_s)

    async def poll_once(self) -> None:
        """Check the marker once and make the transition it calls for."""
        present = await asyncio.to_thread(self._marker_present)
        if present:
            if self._state != "yielded":
                await self._yield()
            return
        retry_due = self._state == "resume_failed" and self._clock() >= self._next_retry_at
        if self._state == "yielded" or retry_due:
            await self._resume()

    async def _yield(self) -> None:
        if self._state == "resume_failed":
            logger.info("Gaming-mode marker reappeared while the reload was failing; yielded again")
        else:
            logger.info("Gaming-mode marker present: releasing the GPU models")
        # Before the callback, so turns are refused from this point on.
        self._state = "yielded"
        self._failed_attempts = 0
        try:
            await asyncio.to_thread(self._on_yield)
        except (RuntimeError, OSError) as exc:
            # The class only: a CUDA error message can be long and is not the
            # operator's next step. The state stays yielded, which is true.
            logger.error("Releasing the GPU models failed (%s)", type(exc).__name__)

    async def _resume(self) -> None:
        try:
            await asyncio.to_thread(self._on_resume)
        except (RuntimeError, OSError) as exc:
            self._resume_failed(exc)
            return
        recovered = self._state == "resume_failed"
        self._state = "normal"
        if recovered:
            logger.info(
                "GPU models reloaded after gaming mode; recovered after %d failed attempts",
                self._failed_attempts,
            )
        else:
            logger.info("Gaming-mode marker gone: GPU models reloaded")
        self._failed_attempts = 0

    def _resume_failed(self, exc: BaseException) -> None:
        """Enter or stay in `resume_failed`, logging only the first failure."""
        if self._state == "resume_failed":
            self._retry_delay = min(self._retry_delay * 2, MAX_RETRY_DELAY_S)
        else:
            self._retry_delay = self._poll_s
            logger.error(
                "Reloading the GPU models after gaming mode failed (%s); retrying with a "
                "backoff up to %d s",
                type(exc).__name__,
                MAX_RETRY_DELAY_S,
            )
        self._failed_attempts += 1
        self._state = "resume_failed"
        self._next_retry_at = self._clock() + self._retry_delay

    def _marker_present(self) -> bool:
        try:
            self._marker_path.stat()
        except (FileNotFoundError, NotADirectoryError):
            self._stat_error_logged = False
            return False
        except OSError as exc:
            if not self._stat_error_logged:
                logger.warning(
                    "Cannot check the gaming-mode marker %s (%s); treating it as present",
                    self._marker_path,
                    type(exc).__name__,
                )
                self._stat_error_logged = True
            return True
        self._stat_error_logged = False
        return True
