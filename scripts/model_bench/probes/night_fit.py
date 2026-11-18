"""Night co-residence fit probe: can the extraction server share the GPU with chat?

The night extraction window runs the extraction llama-server (gpt-oss-20b) beside the
chat llama-server on one 12 GB card. This probe answers the two questions that decide
whether that pairing is usable, against two running servers:

- chat time-to-first-token while an extraction request is decoding (contended), with
  the idle TTFT beside it for comparison;
- the extraction decode rate with the chat model loaded and idle, which is the rate
  the night actually works at, with the contended rate beside it.

Phases, in order, each `--trials` times: chat TTFT with extraction idle; extraction
decode with chat idle; then contended trials, where one extraction stream is opened
and a chat request is sent only once that stream has produced its first decode token,
so the chat prefill meets a server that is decoding, not one that is still prefilling.
A contended trial whose extraction stream ended before its chat request returned is an
error (`--extraction-n-predict` too short), since part of that chat ran on an idle GPU.
Each contended trial records its timeline (first decode token, chat start and end,
stream end) and `extraction_decode_tps_overlap`: the decode rate counted from stream
events inside the chat window (a lower bound, since an event can carry more than one
token), because `timings` covers the whole decode, mostly with chat idle. Each phase is announced on stderr as it starts.

Bars: every contended chat TTFT must be at most `--ttft-bar` seconds, and the median
chat-idle extraction decode rate at least `--decode-bar` tokens per second. The idle
TTFT and both contended decode rates are recorded, not gated. One line per bar is
printed; exit 0 when every bar passes, 1 when one fails, 2 on error.

Prompts are `ttft.FILLER_TEXT`, tokenized once per server by that server's own
tokenizer and cycled to the requested length, so no real conversation text is sent.
Chat requests go through `ttft.run_one_request`. Extraction requests are streamed here
because the contended phase needs a hook at the first decode token, which
`run_one_request` does not offer; the stream is parsed with `ttft.iter_sse_events` and
reduced with `ttft.extract_ttft_row`, so decode rates are llama-server's own
`timings.predicted_per_second`.

Standard library only: it runs inside the mist-backend container, whose image copies
`scripts/`.

Usage:
    python -m scripts.model_bench.probes.night_fit --chat-url URL --extraction-url URL
        --out PATH [--trials 3] [--chat-prompt-tokens 2048]
        [--extraction-prompt-tokens 16384] [--extraction-n-predict 2048]
        [--ttft-bar 1.5] [--decode-bar 23]
"""

from __future__ import annotations

import argparse
import http.client
import json
import statistics
import sys
import threading
import time
import urllib.request
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import ttft

CHAT_N_PREDICT = ttft.N_PREDICT
"""Decode length of each chat request: the ttft probe's, so the chat side of the
contended phase is the same request shape the bench measures."""

FIRST_TOKEN_TIMEOUT_S = ttft.TTFT_HTTP_TIMEOUT_S
"""Longest wait for an extraction stream's first decode token, which covers its prefill."""


class NightFitError(RuntimeError):
    """The probe could not take a measurement it needs."""


PROBE_ERRORS: tuple[type[BaseException], ...] = (
    NightFitError,
    ttft.TtftProbeError,
    OSError,  # urllib.error.URLError, HTTPError, TimeoutError, connection resets
    http.client.HTTPException,
    ValueError,  # json.JSONDecodeError
)


@dataclass(frozen=True, slots=True)
class ServerIO:
    """The HTTP calls and clock the probe uses, injectable so tests need no server.

    `get_props(base_url)` returns `/props`; `tokenize(base_url, text)` returns token
    ids; `run_request(base_url, prompt_tokens, *, n_predict)` returns an
    `extract_ttft_row` dict; `open_stream(base_url, payload)` POSTs a `/completion`
    payload and yields the SSE response lines as text. `clock` stamps the contended
    timeline (stream events and the chat request's start and end) on one monotonic
    clock, so a test can drive the overlap arithmetic deterministically.
    """

    get_props: Callable[[str], dict[str, Any]]
    tokenize: Callable[[str, str], list[int]]
    run_request: Callable[..., dict[str, Any]]
    open_stream: Callable[[str, dict[str, Any]], Iterable[str]]
    clock: Callable[[], float] = time.monotonic


def _get_props(base_url: str, *, timeout: float = 30.0) -> dict[str, Any]:
    with urllib.request.urlopen(base_url.rstrip("/") + "/props", timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _open_stream(
    base_url: str, payload: dict[str, Any], *, timeout: float = ttft.TTFT_HTTP_TIMEOUT_S
) -> Iterator[str]:
    req = urllib.request.Request(
        base_url.rstrip("/") + "/completion",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_bytes in resp:
            yield raw_bytes.decode("utf-8", errors="replace")


def default_io() -> ServerIO:
    """The real HTTP layer: urllib against the two llama-servers."""
    return ServerIO(
        get_props=_get_props,
        tokenize=ttft.tokenize,
        run_request=ttft.run_one_request,
        open_stream=_open_stream,
        clock=time.monotonic,
    )


@dataclass(frozen=True, slots=True)
class FitParams:
    """The CLI's measurement settings and bars, read once by `main`."""

    trials: int
    chat_prompt_tokens: int
    extraction_prompt_tokens: int
    extraction_n_predict: int
    ttft_bar_s: float
    decode_bar_tps: float


@dataclass(frozen=True, slots=True)
class BarResult:
    """One gated bar: its printed line and whether it passed, for the exit code."""

    name: str
    passed: bool
    line: str


def read_n_ctx(props: dict[str, Any], label: str) -> int:
    """`n_ctx` from a llama-server `/props` body, where `bench_host.py` reads it too."""
    n_ctx = props.get("default_generation_settings", {}).get("n_ctx")
    if not isinstance(n_ctx, int) or n_ctx <= 0:
        raise NightFitError(f"{label} /props has no default_generation_settings.n_ctx: {props!r}")
    return n_ctx


def check_fits(label: str, n_ctx: int, prompt_tokens: int, n_predict: int) -> None:
    """Refuse a prompt that would not fit, rather than let the server truncate it."""
    needed = prompt_tokens + n_predict + ttft.CTX_SAFETY_MARGIN
    if needed > n_ctx:
        raise NightFitError(
            f"{label} prompt {prompt_tokens} + n_predict {n_predict} + margin "
            f"{ttft.CTX_SAFETY_MARGIN} = {needed} exceeds its n_ctx {n_ctx}"
        )


def completion_payload(prompt_tokens: list[int], n_predict: int) -> dict[str, Any]:
    """The request `ttft.run_one_request` sends, so both servers see the same shape."""
    return {
        "prompt": prompt_tokens,
        "stream": True,
        "cache_prompt": False,
        "n_predict": n_predict,
        "ignore_eos": True,
        "temperature": 0,
    }


def _is_decode_token(event: dict[str, Any]) -> bool:
    if event.get("stop"):
        return False
    return bool(event.get("content")) or bool(event.get("tokens"))


@dataclass(frozen=True, slots=True)
class ExtractionStream:
    """One streamed extraction request, with its events stamped on `ServerIO.clock`.

    `token_times` holds one stamp per decode event from the first decode token on.
    llama-server holds back a token whose text ends in an incomplete UTF-8 sequence and
    sends it merged into the next event, so the event count is a lower bound on tokens
    decoded; it equals the token count for the ASCII filler prompt with no stop strings.
    `end_t` is the arrival of the final `"stop": true` event.
    """

    row: dict[str, Any]
    start_t: float
    first_token_t: float | None
    end_t: float
    token_times: tuple[float, ...]


def stream_extraction(
    io: ServerIO,
    base_url: str,
    prompt_tokens: list[int],
    *,
    n_predict: int,
    on_first_token: Callable[[], None] = lambda: None,
) -> ExtractionStream:
    """Stream one extraction request; call `on_first_token` at its first decode token.

    The row is `extract_ttft_row`'s, so its decode rate is llama-server's `timings`.
    """
    lines = io.open_stream(base_url, completion_payload(prompt_tokens, n_predict))
    start_t = io.clock()
    first_t: float | None = None
    end_t: float | None = None
    token_times: list[float] = []
    events: list[dict[str, Any]] = []
    for event in ttft.iter_sse_events(lines):
        now = io.clock()
        events.append(event)
        if event.get("stop"):
            end_t = now
        elif first_t is not None or _is_decode_token(event):
            token_times.append(now)
            if first_t is None:
                first_t = now
                on_first_token()
    row = ttft.extract_ttft_row(events, first_content_t=first_t, start_t=start_t)
    return ExtractionStream(
        row=row,
        start_t=start_t,
        first_token_t=first_t,
        end_t=end_t if end_t is not None else io.clock(),
        token_times=tuple(token_times),
    )


def decode_tps(row: dict[str, Any], label: str) -> float:
    tps = row.get("predicted_per_second")
    if not isinstance(tps, int | float) or tps <= 0:
        raise NightFitError(f"{label}: no timings.predicted_per_second in the final event")
    return float(tps)


def chat_ttft_s(row: dict[str, Any], label: str) -> float:
    ttft_ms = row.get("ttft_ms")
    if ttft_ms is None:
        raise NightFitError(f"{label}: the chat stream produced no content")
    return ttft_ms / 1000.0


@dataclass(frozen=True, slots=True)
class ContendedTrial:
    """One contended trial's measurements; its timeline goes to the JSON separately."""

    chat_ttft_s: float
    decode_tps: float
    overlap_decode_tps: float


def overlap_timeline(
    stream: ExtractionStream, chat_start_t: float, chat_end_t: float
) -> dict[str, float | int]:
    """The trial's stamps, in seconds from the extraction request, and the decode inside
    the chat window: decode events arriving after the chat request started and by the
    time it returned, over that window's length. Events are a lower bound on tokens (see
    `ExtractionStream`), so this rate is too. `timings` covers the whole decode, most of
    it with chat idle, so it hides the slowdown this rate shows.
    """
    in_window = sum(1 for t in stream.token_times if chat_start_t < t <= chat_end_t)
    window = chat_end_t - chat_start_t
    first = stream.first_token_t
    return {
        "extraction_first_token_s": (first - stream.start_t) if first is not None else -1.0,
        "chat_start_s": chat_start_t - stream.start_t,
        "chat_end_s": chat_end_t - stream.start_t,
        "extraction_end_s": stream.end_t - stream.start_t,
        "overlap_tokens": in_window,
        "overlap_decode_tps": in_window / window if window > 0 else 0.0,
    }


def run_contended_trial(
    io: ServerIO,
    *,
    chat_url: str,
    chat_tokens: list[int],
    extraction_url: str,
    extraction_tokens: list[int],
    extraction_n_predict: int,
    trial: int,
    timelines: list[dict[str, float | int]],
) -> ContendedTrial:
    """One chat request sent into a decoding extraction stream.

    The extraction stream runs on a worker thread. The chat request is sent from this
    thread only after the worker has seen the stream's first decode token, so it meets
    the extraction server mid-decode rather than mid-prefill. The trial's timeline is
    appended to `timelines` before it is checked, so a refused trial is still recorded.
    A stream that ended before the chat request returned is refused: part of that chat
    request then ran on an idle GPU, so its TTFT is not a contended one.
    """
    label = f"contended trial {trial}"
    first_token_seen = threading.Event()
    worker_done = threading.Event()
    outcome: dict[str, Any] = {}

    def worker() -> None:
        try:
            outcome["stream"] = stream_extraction(
                io,
                extraction_url,
                extraction_tokens,
                n_predict=extraction_n_predict,
                on_first_token=first_token_seen.set,
            )
        except PROBE_ERRORS as exc:
            outcome["error"] = exc
        finally:
            worker_done.set()

    thread = threading.Thread(target=worker, name=f"night-fit-extraction-{trial}", daemon=True)
    thread.start()
    deadline = time.monotonic() + FIRST_TOKEN_TIMEOUT_S
    while not first_token_seen.is_set() and not worker_done.is_set():
        if time.monotonic() >= deadline:
            raise NightFitError(f"{label}: no decode token from the extraction stream in time")
        first_token_seen.wait(timeout=0.05)
    if not first_token_seen.is_set():
        if "error" in outcome:
            raise NightFitError(f"{label}: extraction stream failed: {outcome['error']}")
        raise NightFitError(f"{label}: the extraction stream ended with no decode token")

    chat_start_t = io.clock()
    chat_row = io.run_request(chat_url, chat_tokens, n_predict=CHAT_N_PREDICT)
    chat_end_t = io.clock()
    thread.join()
    if "error" in outcome:
        raise NightFitError(f"{label}: extraction stream failed: {outcome['error']}")
    stream: ExtractionStream = outcome["stream"]
    timeline = overlap_timeline(stream, chat_start_t, chat_end_t)
    timelines.append(timeline)
    if stream.end_t <= chat_end_t:
        raise NightFitError(
            f"{label}: the extraction stream ended at {timeline['extraction_end_s']:.3f} s, "
            f"before the chat request returned at {timeline['chat_end_s']:.3f} s, so the "
            f"chat was not measured against a decoding server; --extraction-n-predict "
            f"{extraction_n_predict} is too short for the overlap"
        )
    if chat_end_t <= chat_start_t:
        raise NightFitError(f"{label}: the chat request took no time on the probe's clock")
    return ContendedTrial(
        chat_ttft_s=chat_ttft_s(chat_row, label),
        decode_tps=decode_tps(stream.row, label),
        overlap_decode_tps=float(timeline["overlap_decode_tps"]),
    )


def run_night_fit(
    io: ServerIO,
    *,
    chat_url: str,
    extraction_url: str,
    params: FitParams,
    results: dict[str, Any],
) -> None:
    """Take every measurement, filling `results` as it goes so an error keeps what was read."""
    results["chat_n_ctx"] = read_n_ctx(io.get_props(chat_url), "chat")
    results["extraction_n_ctx"] = read_n_ctx(io.get_props(extraction_url), "extraction")
    check_fits("chat", results["chat_n_ctx"], params.chat_prompt_tokens, CHAT_N_PREDICT)
    check_fits(
        "extraction",
        results["extraction_n_ctx"],
        params.extraction_prompt_tokens,
        params.extraction_n_predict,
    )

    chat_tokens = ttft.build_prompt_tokens(
        io.tokenize(chat_url, ttft.FILLER_TEXT), params.chat_prompt_tokens
    )
    extraction_tokens = ttft.build_prompt_tokens(
        io.tokenize(extraction_url, ttft.FILLER_TEXT), params.extraction_prompt_tokens
    )

    idle_ttft = results["chat_ttft_idle_s"]
    _phase(f"idle chat TTFT, extraction idle, {params.trials} trials")
    for trial in range(params.trials):
        row = io.run_request(chat_url, chat_tokens, n_predict=CHAT_N_PREDICT)
        idle_ttft.append(chat_ttft_s(row, f"idle chat trial {trial}"))

    idle_tps = results["extraction_decode_tps_chat_idle_trials"]
    _phase(f"chat-idle extraction decode, {params.trials} trials")
    for trial in range(params.trials):
        stream = stream_extraction(
            io, extraction_url, extraction_tokens, n_predict=params.extraction_n_predict
        )
        idle_tps.append(decode_tps(stream.row, f"chat-idle extraction trial {trial}"))
    results["extraction_decode_tps_chat_idle"] = statistics.median(idle_tps)

    contended_ttft = results["chat_ttft_contended_s"]
    contended_tps = results["extraction_decode_tps_contended_trials"]
    overlap_tps = results["extraction_decode_tps_overlap_trials"]
    for trial in range(params.trials):
        _phase(f"contended trial {trial + 1}/{params.trials}")
        measured = run_contended_trial(
            io,
            chat_url=chat_url,
            chat_tokens=chat_tokens,
            extraction_url=extraction_url,
            extraction_tokens=extraction_tokens,
            extraction_n_predict=params.extraction_n_predict,
            trial=trial,
            timelines=results["contended_timeline"],
        )
        contended_ttft.append(measured.chat_ttft_s)
        contended_tps.append(measured.decode_tps)
        overlap_tps.append(measured.overlap_decode_tps)
    results["extraction_decode_tps_contended"] = statistics.median(contended_tps)
    results["extraction_decode_tps_overlap"] = statistics.median(overlap_tps)
    _phase("done")


def _phase(text: str) -> None:
    """Announce a phase on stderr as it starts, so whoever runs the probe can sample
    host memory during the contended trials.
    """
    print(f"night_fit: phase: {text}", file=sys.stderr, flush=True)


def evaluate_bars(results: dict[str, Any], params: FitParams) -> list[BarResult]:
    """Gate every contended chat TTFT and the chat-idle extraction decode rate."""
    ttfts: list[float] = results["chat_ttft_contended_s"]
    worst = max(ttfts)
    ttft_ok = worst <= params.ttft_bar_s
    listed = ", ".join(f"{v:.3f}" for v in ttfts)
    ttft_line = (
        f"bar chat_ttft_contended_s: {'PASS' if ttft_ok else 'FAIL'} max {worst:.3f} s "
        f"{'<=' if ttft_ok else '>'} {params.ttft_bar_s:g} s (trials: {listed})"
    )

    tps: float = results["extraction_decode_tps_chat_idle"]
    tps_ok = tps >= params.decode_bar_tps
    trials = ", ".join(f"{v:.2f}" for v in results["extraction_decode_tps_chat_idle_trials"])
    tps_line = (
        f"bar extraction_decode_tps_chat_idle: {'PASS' if tps_ok else 'FAIL'} median "
        f"{tps:.2f} tok/s {'>=' if tps_ok else '<'} {params.decode_bar_tps:g} tok/s "
        f"(trials: {trials})"
    )
    return [
        BarResult("chat_ttft_contended_s", ttft_ok, ttft_line),
        BarResult("extraction_decode_tps_chat_idle", tps_ok, tps_line),
    ]


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {text}")
    return value


def _positive_float(text: str) -> float:
    value = float(text)
    if not value > 0:
        raise argparse.ArgumentTypeError(f"must be a positive number, got {text}")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m scripts.model_bench.probes.night_fit",
        description="Night co-residence fit: chat TTFT and extraction decode rate on one GPU.",
    )
    p.add_argument("--chat-url", required=True, help="chat llama-server, e.g. http://mist-llm:8080")
    p.add_argument("--extraction-url", required=True, help="extraction llama-server base URL")
    p.add_argument("--out", required=True, help="path of the JSON results file")
    p.add_argument("--trials", type=_positive_int, default=3)
    p.add_argument("--chat-prompt-tokens", type=_positive_int, default=2048)
    p.add_argument("--extraction-prompt-tokens", type=_positive_int, default=16384)
    p.add_argument("--extraction-n-predict", type=_positive_int, default=2048)
    p.add_argument(
        "--ttft-bar", type=_positive_float, default=1.5, help="max contended chat TTFT, s"
    )
    p.add_argument(
        "--decode-bar",
        type=_positive_float,
        default=23.0,
        help="min chat-idle extraction decode rate, tokens/s",
    )
    return p.parse_args(argv)


def _write(out: Path, results: dict[str, Any]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None, *, io: ServerIO | None = None) -> int:
    args = parse_args(argv)
    params = FitParams(
        trials=args.trials,
        chat_prompt_tokens=args.chat_prompt_tokens,
        extraction_prompt_tokens=args.extraction_prompt_tokens,
        extraction_n_predict=args.extraction_n_predict,
        ttft_bar_s=args.ttft_bar,
        decode_bar_tps=args.decode_bar,
    )
    out = Path(args.out)
    results: dict[str, Any] = {
        "chat_url": args.chat_url,
        "extraction_url": args.extraction_url,
        "params": {
            "trials": params.trials,
            "chat_prompt_tokens": params.chat_prompt_tokens,
            "chat_n_predict": CHAT_N_PREDICT,
            "extraction_prompt_tokens": params.extraction_prompt_tokens,
            "extraction_n_predict": params.extraction_n_predict,
            "ttft_bar_s": params.ttft_bar_s,
            "decode_bar_tps": params.decode_bar_tps,
        },
        "chat_n_ctx": None,
        "extraction_n_ctx": None,
        "chat_ttft_idle_s": [],
        "chat_ttft_contended_s": [],
        "extraction_decode_tps_chat_idle": None,
        "extraction_decode_tps_chat_idle_trials": [],
        "extraction_decode_tps_contended": None,
        "extraction_decode_tps_contended_trials": [],
        "extraction_decode_tps_overlap": None,
        "extraction_decode_tps_overlap_trials": [],
        "contended_timeline": [],
        "bars": [],
        "passed": None,
        "error": None,
    }
    try:
        run_night_fit(
            io or default_io(),
            chat_url=args.chat_url,
            extraction_url=args.extraction_url,
            params=params,
            results=results,
        )
    except PROBE_ERRORS as exc:
        results["error"] = f"{type(exc).__name__}: {exc}"
        print(f"night_fit: error: {results['error']}", file=sys.stderr)
        try:
            _write(out, results)
        except OSError as write_exc:
            print(f"night_fit: error: could not write {out}: {write_exc}", file=sys.stderr)
        return 2

    bars = evaluate_bars(results, params)
    results["bars"] = [{"name": b.name, "passed": b.passed, "line": b.line} for b in bars]
    results["passed"] = all(b.passed for b in bars)
    try:
        _write(out, results)
    except OSError as exc:
        print(f"night_fit: error: could not write {out}: {exc}", file=sys.stderr)
        return 2
    for bar in bars:
        print(bar.line)
    return 0 if results["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
