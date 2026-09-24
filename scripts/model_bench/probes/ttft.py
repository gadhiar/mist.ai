"""Time-to-first-token probe against a running llama-server.

Deterministic filler text is tokenized via `/tokenize` (no special tokens) exactly once
per probe run. For each context target (2048, 8192, 32000, each capped at
`n_ctx - n_predict - 16` from /props), an exact-length prompt is built by cycling that
same id list end-to-end to the target length (`cycle_to_length`), not by tokenizing a
longer text and slicing -- slicing left the largest target (32000) unreachable whenever
FILLER_TEXT itself tokenized to fewer ids than that, which a fixed multiplier could not
guarantee. Cycling reaches every target exactly regardless of how many ids the filler
text tokenizes to; an empty tokenize result is refused (`TtftProbeError`) rather than
silently producing a shorter-than-requested prompt.

The HTTP/SSE plumbing (`run_ttft_probe`) needs a live server and is not unit
tested; the parsing it is built from (`iter_sse_events`,
`extract_ttft_row`) is pure and is tested against
`tests/unit/model_bench/fixtures/host/sse_stream.txt`. `cycle_to_length` is pure and
tested directly.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Iterator
from typing import Any

# Deterministic filler text, repeated and re-tokenized as needed to reach
# the longest target length. Content is arbitrary; determinism (same text
# every run, every arm) is what matters for comparing arms.
FILLER_TEXT = (
    "The quick brown fox jumps over the lazy dog near the riverbank at dawn, "
    "while the old lighthouse keeper records the tide tables in a worn leather "
    "notebook, unaware that the storm building over the western ridge will "
    "reach the coast well before nightfall. "
) * 400

TTFT_TARGETS: tuple[int, ...] = (2048, 8192, 32000)
N_PREDICT = 256
WARMUP_REPS = 1
MEASURED_REPS = 5
CTX_SAFETY_MARGIN = 16


class TtftProbeError(RuntimeError):
    """The server returned something the probe did not know how to parse."""


def cap_target(target: int, n_ctx: int, n_predict: int = N_PREDICT) -> int:
    """Cap a raw ctx target at `n_ctx - n_predict - CTX_SAFETY_MARGIN`."""
    ceiling = n_ctx - n_predict - CTX_SAFETY_MARGIN
    return min(target, max(ceiling, 0))


def tokenize(base_url: str, text: str, *, timeout: float = 30.0) -> list[int]:
    """POST /tokenize (no special tokens) and return the token id list."""
    payload = {"content": text, "add_special": False}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/tokenize",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    tokens = body.get("tokens")
    if not isinstance(tokens, list):
        raise TtftProbeError(f"/tokenize response has no 'tokens' list: {body!r}")
    return tokens


def cycle_to_length(base_ids: list[int], length: int) -> list[int]:
    """Repeat `base_ids` end-to-end until there are exactly `length` ids.

    Cycling (rather than a single tokenize-and-slice) is what lets every ctx target,
    including the largest (32000), be reached exactly regardless of how many ids
    FILLER_TEXT itself tokenizes to. Refuses -- rather than silently returning a
    shorter-than-requested prompt -- if `base_ids` is empty, since an empty list has
    nothing to cycle.
    """
    if not base_ids:
        raise TtftProbeError("filler token id list is empty; cannot build a prompt of any length")
    if length <= 0:
        return []
    reps = (length // len(base_ids)) + 1
    return (base_ids * reps)[:length]


def build_prompt_tokens(base_ids: list[int], target_len: int) -> list[int]:
    """Build one exact-length prompt by cycling the already-tokenized filler ids.

    `base_ids` is tokenized once per probe run (see `run_ttft_probe`), not once per
    ctx target, and is cycled here to reach `target_len` exactly.
    """
    return cycle_to_length(base_ids, target_len)


def iter_sse_events(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Parse `data: {...}` SSE lines into decoded JSON event dicts.

    Blank lines and lines not starting with `data:` (SSE comments, `event:`
    lines) are skipped. `data: [DONE]` (a sentinel some OpenAI-compatible
    servers emit) is skipped rather than JSON-decoded.
    """
    for raw_line in lines:
        line = raw_line.rstrip("\r\n")
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if not payload or payload == "[DONE]":
            continue
        yield json.loads(payload)


def extract_ttft_row(
    events: Iterable[dict[str, Any]],
    *,
    first_content_t: float | None,
    start_t: float,
) -> dict[str, Any]:
    """Reduce a decoded SSE event stream to the ttft.jsonl timing fields.

    `first_content_t` is the caller's wall-clock `time.monotonic()` reading
    at the first event with non-empty `content` (the caller must capture
    this live, during streaming, since replaying `events` after the fact
    cannot recover real arrival times). `start_t` is the request send time.

    Final timings come from the last event carrying `"stop": true` (per its
    `timings` object, per llama-server's completion stream contract).
    """
    final_timings: dict[str, Any] | None = None
    predicted_n: int | None = None
    for event in events:
        if event.get("stop"):
            final_timings = event.get("timings") or {}
            predicted_n = event.get("tokens_predicted", predicted_n)
    if final_timings is None:
        raise TtftProbeError("no event with 'stop': true carrying 'timings' seen")
    ttft_ms = None if first_content_t is None else (first_content_t - start_t) * 1000.0
    return {
        "ttft_ms": ttft_ms,
        "total_ms": final_timings.get("predicted_ms", 0) + final_timings.get("prompt_ms", 0)
        if "predicted_ms" in final_timings and "prompt_ms" in final_timings
        else None,
        "prompt_ms": final_timings.get("prompt_ms"),
        "prompt_per_second": final_timings.get("prompt_per_second"),
        "predicted_n": predicted_n,
        "predicted_ms": final_timings.get("predicted_ms"),
        "predicted_per_second": final_timings.get("predicted_per_second"),
    }


def run_one_request(
    base_url: str,
    prompt_tokens: list[int],
    *,
    n_predict: int = N_PREDICT,
    timeout: float = 300.0,
) -> dict[str, Any]:
    """Stream one /completion request and return an extract_ttft_row() dict.

    Not unit tested (needs a live server); kept small and delegates all
    parsing to `iter_sse_events` / `extract_ttft_row`, which are.
    """
    payload = {
        "prompt": prompt_tokens,
        "stream": True,
        "cache_prompt": False,
        "n_predict": n_predict,
        "ignore_eos": True,
        "temperature": 0,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/completion",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start_t = time.monotonic()
    first_content_t: float | None = None
    decoded_events: list[dict[str, Any]] = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_bytes in resp:
            line = raw_bytes.decode("utf-8", errors="replace")
            for event in iter_sse_events([line]):
                if first_content_t is None and event.get("content"):
                    first_content_t = time.monotonic()
                decoded_events.append(event)
    return extract_ttft_row(decoded_events, first_content_t=first_content_t, start_t=start_t)


def run_ttft_probe(
    base_url: str,
    *,
    n_ctx: int,
    n_predict: int = N_PREDICT,
    warmup_reps: int = WARMUP_REPS,
    measured_reps: int = MEASURED_REPS,
) -> list[dict[str, Any]]:
    """Run the full ttft probe (all targets, warmup + measured reps).

    Returns rows matching the ttft.jsonl schema, including warmup rows
    (`"warmup": true`) so the caller can choose to keep or drop them.
    """
    rows: list[dict[str, Any]] = []

    try:
        base_ids = tokenize(base_url, FILLER_TEXT)
    except (TtftProbeError, urllib.error.URLError) as exc:
        # Tokenizing the filler is a one-time, all-targets-shared step; if it fails
        # outright, every target is unreachable -- record one errored row per target
        # rather than raising and losing the other targets' rows.
        for target in TTFT_TARGETS:
            capped = cap_target(target, n_ctx, n_predict)
            rows.append(
                {
                    "ctx_target": target,
                    "prompt_tokens": capped,
                    "rep": 0,
                    "warmup": True,
                    "ttft_ms": None,
                    "total_ms": None,
                    "prompt_ms": None,
                    "prompt_per_second": None,
                    "predicted_n": None,
                    "predicted_ms": None,
                    "predicted_per_second": None,
                    "error": str(exc),
                }
            )
        return rows

    for target in TTFT_TARGETS:
        capped = cap_target(target, n_ctx, n_predict)
        try:
            prompt_tokens = build_prompt_tokens(base_ids, capped)
        except TtftProbeError as exc:
            rows.append(
                {
                    "ctx_target": target,
                    "prompt_tokens": capped,
                    "rep": 0,
                    "warmup": True,
                    "ttft_ms": None,
                    "total_ms": None,
                    "prompt_ms": None,
                    "prompt_per_second": None,
                    "predicted_n": None,
                    "predicted_ms": None,
                    "predicted_per_second": None,
                    "error": str(exc),
                }
            )
            continue
        for rep in range(warmup_reps + measured_reps):
            warmup = rep < warmup_reps
            row: dict[str, Any] = {
                "ctx_target": target,
                "prompt_tokens": len(prompt_tokens),
                "rep": rep if warmup else rep - warmup_reps,
                "warmup": warmup,
                "error": None,
            }
            try:
                row.update(run_one_request(base_url, prompt_tokens, n_predict=n_predict))
            except (TtftProbeError, urllib.error.URLError, TimeoutError) as exc:
                row.update(
                    {
                        "ttft_ms": None,
                        "total_ms": None,
                        "prompt_ms": None,
                        "prompt_per_second": None,
                        "predicted_n": None,
                        "predicted_ms": None,
                        "predicted_per_second": None,
                        "error": str(exc),
                    }
                )
            rows.append(row)
    return rows
