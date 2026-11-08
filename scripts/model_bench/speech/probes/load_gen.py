"""Decode load for run 1's `decoding` condition and run 7's busy chained turns.

`LoadGenerator` keeps exactly one streamed 512-token completion in flight from the bench server,
starting the next the moment the last one ends, so the model is decoding through every synthesis
call it overlaps. Each completion's decode rate is read from the `timings` object llama-server
sends on the final chunk, not from client-side token counting, so it is the server's own figure.

Stdlib only: it runs on the host against `mist-bench-llm`'s loopback port.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from .. import core

MAX_TOKENS = 512

# `ignore_eos` (a llama-server sampling parameter it reads on the chat endpoint too) makes every
# completion run its full 512 tokens, so the load never pauses for a short reply. The prompt is
# invented and asks for long prose; its content is never read.
DEFAULT_BODY: dict = {
    "messages": [
        {
            "role": "user",
            "content": (
                "Write a long, slow-paced story about a lighthouse keeper on a quiet island, "
                "in plain prose with no headings or lists. Keep going until you are stopped."
            ),
        }
    ],
    "max_tokens": MAX_TOKENS,
    "temperature": 0.7,
    "seed": 7,
    "ignore_eos": True,
    "cache_prompt": True,
}

ChatFn = Callable[..., core.ChatStreamResult]


@dataclass(frozen=True, slots=True)
class Completion:
    """One finished completion: its times on the generator's clock and the server's timings."""

    seq: int
    t_request: float
    t_end: float
    t_start_unix: float
    t_end_unix: float
    predicted_n: int
    predicted_ms: float
    predicted_per_second: float
    prompt_n: int | None
    finish_reason: str | None

    def as_row(self) -> dict[str, object]:
        """The completion as a JSON-ready row."""
        return {
            "seq": self.seq,
            "t_request": self.t_request,
            "t_end": self.t_end,
            "t_start_unix": self.t_start_unix,
            "t_end_unix": self.t_end_unix,
            "predicted_n": self.predicted_n,
            "predicted_ms": self.predicted_ms,
            "predicted_per_second": self.predicted_per_second,
            "prompt_n": self.prompt_n,
            "finish_reason": self.finish_reason,
        }


def completion_from(
    result: core.ChatStreamResult, *, seq: int, t_start_unix: float, t_end_unix: float
) -> Completion:
    """Read a finished stream's decode figures from its final `timings`; refuse a stream without."""
    timings = result.timings or {}
    try:
        predicted_n = int(timings["predicted_n"])
        predicted_ms = float(timings["predicted_ms"])
        per_second = float(timings["predicted_per_second"])
    except (KeyError, TypeError, ValueError) as exc:
        raise core.StreamError(
            f"completion {seq} ended without usable timings ({timings!r}); "
            f"the decode rate cannot be read"
        ) from exc
    prompt_n = timings.get("prompt_n")
    return Completion(
        seq=seq,
        t_request=result.t_request,
        t_end=result.t_end,
        t_start_unix=t_start_unix,
        t_end_unix=t_end_unix,
        predicted_n=predicted_n,
        predicted_ms=predicted_ms,
        predicted_per_second=per_second,
        prompt_n=int(prompt_n) if prompt_n is not None else None,
        finish_reason=result.finish_reason,
    )


def decode_tps(completions: Iterable[Completion]) -> float | None:
    """Tokens a second over `completions`: total tokens decoded over total decode time.

    Weighting by decode time rather than averaging each completion's rate keeps a short, cut
    completion from counting as much as a full one.
    """
    done = list(completions)
    total_ms = sum(c.predicted_ms for c in done)
    if not done or total_ms <= 0:
        return None
    return sum(c.predicted_n for c in done) / (total_ms / 1000.0)


def within(completions: Iterable[Completion], start: float, end: float) -> list[Completion]:
    """The completions that ran wholly inside `[start, end]` on the generator's clock."""
    return [c for c in completions if c.t_request >= start and c.t_end <= end]


class LoadGenerator:
    """One streamed completion in flight at a time, back to back, until stopped.

    `chat` defaults to `core.stream_chat`; tests pass a fake. `clock` stamps the completions
    and must be the clock the caller marks its synthesis window with, so `within` compares like
    with like.
    """

    def __init__(
        self,
        base_url: str,
        *,
        body: Mapping[str, object] | None = None,
        chat: ChatFn = core.stream_chat,
        clock: Callable[[], float] = time.perf_counter,
        wall: Callable[[], float] = time.time,
    ):
        self.base_url = base_url
        self.body = dict(DEFAULT_BODY if body is None else body)
        self._chat = chat
        self._clock = clock
        self._wall = wall
        self._done: list[Completion] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: core.SpeechBenchError | None = None

    @property
    def completions(self) -> list[Completion]:
        """The completions finished so far."""
        return list(self._done)

    def complete_one(self) -> Completion:
        """Stream one completion to its end and record it."""
        t_start_unix = self._wall()
        result = self._chat(self.base_url, self.body, clock=self._clock)
        done = completion_from(
            result, seq=len(self._done), t_start_unix=t_start_unix, t_end_unix=self._wall()
        )
        self._done.append(done)
        return done

    def run_for(self, seconds: float) -> list[Completion]:
        """Run completions back to back in this thread until `seconds` have passed.

        The window is checked between completions, so the last one finishes past it rather than
        being cut short without its timings.
        """
        t0 = self._clock()
        first = len(self._done)
        while self._clock() - t0 < seconds:
            self.complete_one()
        return self._done[first:]

    def start(self) -> None:
        """Start the back-to-back loop in a background thread."""
        if self._thread is not None:
            raise core.SpeechBenchError("the load generator is already running")
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="load-gen", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.complete_one()
            except core.SpeechBenchError as exc:
                self.error = exc
                return

    def stop(self, timeout: float = 600.0) -> list[Completion]:
        """Stop after the completion in flight ends; return every completion of this run.

        Raises if the loop failed, since a `decoding` condition without decode load is not the
        condition it claims to be.
        """
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout)
            if thread.is_alive():
                raise core.SpeechBenchError(
                    f"the load generator's completion in flight did not end within {timeout} s"
                )
        if self.error is not None:
            raise core.SpeechBenchError(f"the load generator failed: {self.error}")
        return list(self._done)
