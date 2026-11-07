"""scripts/model_bench/speech/probes/load_gen.py: run 1's decode load generator.

The generator is driven through fake chat clients and scripted clocks: no network and no server.
The recorded stream is a hand-built OpenAI chat SSE ending with llama-server's `timings` object
and `data: [DONE]` (fixtures/speech/tts/sse_long_completion.txt).
"""

from __future__ import annotations

import io
import json
import sys
import threading
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402

from scripts.model_bench.speech import core  # noqa: E402
from scripts.model_bench.speech.probes import load_gen  # noqa: E402

FIXTURE = core.SPEECH_FIXTURES_DIR / "tts" / "sse_long_completion.txt"
FIXTURE_LINES = FIXTURE.read_text(encoding="utf-8").splitlines(keepends=True)


class StepClock:
    """Each read advances by `step` seconds, so every timestamp is known in advance."""

    def __init__(self, start: float = 0.0, step: float = 0.5):
        self.now = start
        self.step = step

    def __call__(self) -> float:
        self.now += self.step
        return self.now


def fixture_chat(url, body, *, clock):
    """A chat client that answers every request with the recorded stream."""
    return core.parse_chat_stream(FIXTURE_LINES, clock=clock, t_request=clock())


def make_result(t_request: float, t_end: float, *, predicted_n: int, predicted_ms: float):
    timings = {
        "predicted_n": predicted_n,
        "predicted_ms": predicted_ms,
        "predicted_per_second": predicted_n / predicted_ms * 1000.0,
        "prompt_n": 10,
    }
    return core.ChatStreamResult(
        content="x",
        reasoning_content="",
        tool_calls=(),
        t_request=t_request,
        t_first_content=t_request + 0.01,
        t_first_tool_call=None,
        t_end=t_end,
        finish_reason="length",
        timings=timings,
    )


def completion(t_request: float, t_end: float, n: int = 100, ms: float = 1000.0):
    return load_gen.completion_from(
        make_result(t_request, t_end, predicted_n=n, predicted_ms=ms),
        seq=0,
        t_start_unix=0.0,
        t_end_unix=0.0,
    )


# ---------------------------------------------------------------------------
# Tokens a second from the final timings
# ---------------------------------------------------------------------------


def test_a_completion_records_tokens_a_second_from_the_final_timings():
    generator = load_gen.LoadGenerator(
        "http://127.0.0.1:8080", chat=fixture_chat, clock=StepClock()
    )

    done = generator.complete_one()

    assert done.predicted_per_second == 75.0
    assert done.predicted_n == 512
    assert done.predicted_ms == 6826.7
    assert done.finish_reason == "length"


def test_a_completion_row_carries_its_request_and_end_times():
    clock = StepClock(start=10.0, step=0.5)
    generator = load_gen.LoadGenerator("http://127.0.0.1:8080", chat=fixture_chat, clock=clock)

    row = generator.complete_one().as_row()

    # The parser reads the clock once for the request, then once per `data:` event (13 events).
    assert row["t_request"] == 10.5
    assert row["t_end"] == 10.5 + 13 * 0.5
    assert row["predicted_per_second"] == 75.0


def test_a_stream_without_timings_is_refused():
    lines = [line for line in FIXTURE_LINES if '"timings"' not in line]

    def chat(url, body, *, clock):
        return core.parse_chat_stream(lines, clock=clock, t_request=clock())

    generator = load_gen.LoadGenerator("http://127.0.0.1:8080", chat=chat, clock=StepClock())

    with pytest.raises(core.StreamError, match="timings"):
        generator.complete_one()


def test_decode_tps_is_total_tokens_over_total_decode_time():
    done = [completion(0, 1, n=100, ms=1000.0), completion(1, 3, n=300, ms=2000.0)]

    assert load_gen.decode_tps(done) == pytest.approx(400 / 3.0)


def test_decode_tps_of_no_completion_is_none():
    assert load_gen.decode_tps([]) is None


def test_within_keeps_only_completions_inside_the_window():
    done = [completion(90, 110), completion(110, 150), completion(150, 190), completion(190, 210)]

    kept = load_gen.within(done, 100.0, 200.0)

    assert [(c.t_request, c.t_end) for c in kept] == [(110, 150), (150, 190)]


# ---------------------------------------------------------------------------
# The request body, through core.stream_chat
# ---------------------------------------------------------------------------


class FakeResponse(io.BytesIO):
    status = 200


def test_the_request_streams_one_512_token_completion_to_the_bench_server():
    seen = {}

    def opener(request, timeout):
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse("".join(FIXTURE_LINES).encode("utf-8"))

    def chat(url, body, *, clock):
        return core.stream_chat(url, body, clock=clock, opener=opener)

    generator = load_gen.LoadGenerator("http://127.0.0.1:8080", chat=chat, clock=StepClock())

    done = generator.complete_one()

    assert seen["url"] == "http://127.0.0.1:8080/v1/chat/completions"
    assert seen["body"]["stream"] is True
    assert seen["body"]["max_tokens"] == 512
    assert [m["role"] for m in seen["body"]["messages"]] == ["user"]
    assert done.predicted_n == 512


# ---------------------------------------------------------------------------
# Back to back, one in flight
# ---------------------------------------------------------------------------


def test_run_for_requests_back_to_back_until_the_window_has_passed():
    clock = StepClock(start=0.0, step=1.0)
    calls = []

    def chat(url, body, *, clock):
        t_request = clock()
        calls.append(t_request)
        return make_result(t_request, clock(), predicted_n=50, predicted_ms=500.0)

    generator = load_gen.LoadGenerator("http://127.0.0.1:8080", chat=chat, clock=clock)

    done = generator.run_for(10.0)

    # t0 = 1; each completion reads the clock twice and the loop once more before the next.
    assert calls == [3.0, 6.0, 9.0]
    assert len(done) == 3
    assert all(c.t_request < 1.0 + 10.0 for c in done)


def test_start_keeps_exactly_one_completion_in_flight_and_stop_waits_for_it():
    lock = threading.Lock()
    state = {"in_flight": 0, "max_in_flight": 0, "calls": 0}
    release = [threading.Event() for _ in range(3)]
    entered = [threading.Event() for _ in range(3)]

    def chat(url, body, *, clock):
        with lock:
            index = state["calls"]
            state["calls"] += 1
            state["in_flight"] += 1
            state["max_in_flight"] = max(state["max_in_flight"], state["in_flight"])
        entered[index].set()
        assert release[index].wait(timeout=10)
        with lock:
            state["in_flight"] -= 1
        return make_result(float(index), index + 0.5, predicted_n=10, predicted_ms=100.0)

    generator = load_gen.LoadGenerator("http://127.0.0.1:8080", chat=chat, clock=StepClock())

    generator.start()
    assert entered[0].wait(timeout=10)
    assert not entered[1].is_set()
    release[0].set()
    assert entered[1].wait(timeout=10)
    release[1].set()
    assert entered[2].wait(timeout=10)
    stopper = threading.Thread(target=lambda: state.setdefault("done", generator.stop()))
    stopper.start()
    release[2].set()
    stopper.join(timeout=10)

    assert state["max_in_flight"] == 1
    assert state["calls"] == 3
    assert [c.seq for c in state["done"]] == [0, 1, 2]


def test_a_stream_error_stops_the_loop_and_stop_raises():
    def chat(url, body, *, clock):
        raise core.StreamError("POST failed: connection refused")

    generator = load_gen.LoadGenerator("http://127.0.0.1:8080", chat=chat, clock=StepClock())

    generator.start()

    with pytest.raises(core.SpeechBenchError, match="connection refused"):
        generator.stop()
