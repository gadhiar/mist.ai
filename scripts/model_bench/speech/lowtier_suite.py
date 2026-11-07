"""Run 7, the low-level models: the `lowtier` and `remote-memory` subcommands (plan v2, task T7).

`lowtier` measures one small model on one placement against a server that is already up, in one
of four modes:

- local (`--model`, `--placement`): against `mist-bench-small`, which `small-serve` started on
  127.0.0.1:8081. It refuses unless the served alias is the one asked for, then runs 2 warm-ups,
  3 cold requests on each prefix, 2 warm sessions of run 2's 10 turns and the 24 tool prompts at
  seeds 1, 2 and 3, and reads memory by the placement's method: GPU memory by difference from
  the bench run's `desktop` step for the 4070 placements, and the container's `memory.peak` for
  every placement, because llama.cpp keeps the Gemma 4 E models' per-layer embeddings in host
  memory, so a file size says little about the card's share.
- `--slice-search`: for `gpu-4g` and `gpu-2g`, bisects `-ngl` for the largest layer count whose
  footprint, by difference, is at or under the slice, serving each count in a fresh directory.
- `--e4b-reload`: times E4B's reload after `bench_host unserve`, three times.
- `--remote-url`: the host-1070 rows. It polls `/v1/models` on host-1070's port 8090, runs each
  expected alias's rows when that alias appears, records rows a window cut short as cut, and
  refuses an alias it was not given. Memory there comes from the H6 script's summary, which
  `remote-memory` reads.

Every request records the client's time to first token and the server's own `timings`, beside
each other, and flags a reply that still carries `reasoning_content` with thinking off.

Stdlib only: the host runs it under a plain Python.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol

from .. import bench_host
from . import core, host1070_script, toolcall

SUITE = "lowtier"
REMOTE_MEMORY_SUITE = "host1070"
LOWTIER_PREFIX_PATH = core.SPEECH_DIR / "lowtier_prefix.txt"
CACHE_TURNS_PATH = core.SPEECH_DIR / "cache_turns.json"
SMALL_MODELS_DIR_ENV = "MODEL_BENCH_SMALL_MODELS_DIR"

# Run 7's fixed date and time, which the low-level prompt states and C03 and C12 are scored from.
FIXED_NOW = datetime(2026, 11, 11, 21, 0)
FIXED_NOW_TEXT = "Wednesday 11 November 2026, 21:00"

LOWTIER_TOKENS = 1500
CACHE_PREFIX_TOKENS = 6100
PREFIX_LOWTIER = "lowtier-1500"
PREFIX_CACHE = "cache-6100"
WARMUPS = 2
COLD_REPS = 3
WARM_SESSIONS = 2
TOOL_SEEDS = (1, 2, 3)
BUSY_TOOL_SEEDS = (1,)
REPLY_TOKENS = 128
TOOL_REPLY_TOKENS = 256
OPENING_WORDS = 16
DEFAULT_SEED = 20261111
COLD_QUESTION = "In one short sentence, what can you do for me right now?"

FULL_NGL = 999
UNLOAD_SLACK_MIB = 128
UNLOAD_TIMEOUT_S = 60.0
UNLOAD_POLL_S = 0.5
DEFAULT_POLL_S = 10.0
DEFAULT_WAIT_S = 3600.0
E4B_ARM = "c0"
E4B_ROUNDS = 3
BENCH_LLM_URL = f"http://{bench_host.BENCH_LLM_HOST_BIND}:{bench_host.BENCH_LLM_PORT}"
REMOTE_MEMORY_METHOD = "host-1070 summary, read by remote-memory"
GPU_MEMORY_METHOD = "gpu by difference from the desktop step, and container memory.peak"
HOST_MEMORY_METHOD = "container memory.peak"
DESKTOP_STEP = "desktop"

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
_OFFLOADED_RE = re.compile(r"offloaded (\d+)/(\d+) layers")
_CPU_VARIANT_RE = re.compile(r"libggml-cpu-([A-Za-z0-9_]+)\.so")
_MEM_USAGE_RE = re.compile(r"([\d.]+)\s*([KMG]i?B)")
_PASS_SUFFIXES = ("idle", "busy", host1070_script.GAME_PASS)
# Invented words for the cold requests' opening lines; any 16 of them make a line no cache holds.
_OPENING_WORDS = (
    "amber", "basalt", "cedar", "drift", "ember", "fjord", "garnet", "harbor", "indigo", "juniper",
    "kelp", "lantern", "meadow", "nectar", "orchid", "pebble", "quartz", "ripple", "saffron",
    "thistle", "umber", "velvet", "willow", "yarrow", "zephyr", "acorn", "bramble", "copper",
    "dune", "estuary", "fennel", "glacier", "heather", "iris", "jasper", "kestrel", "lichen",
    "marble", "nettle", "oyster", "plover", "quill", "reed", "sorrel", "tundra", "upland", "vale",
    "wren", "yew", "zinnia", "anchor", "beacon", "cobalt", "delta", "ferry", "gravel", "hollow",
    "inlet", "jetty", "kiln", "ledge", "mill", "north", "orbit",
)  # fmt: skip


class AliasError(core.SpeechBenchError):
    """The server serves an alias this run was not given."""


# ---------------------------------------------------------------------------
# Inputs: the low-level prompt, run 2's turns, the tool set
# ---------------------------------------------------------------------------


def approx_tokenize(text: str) -> list[str]:
    """Words and punctuation marks: the fake `/tokenize` that holds the prompt to its target.

    A run reads the real count from the server's `/tokenize` and records it on every row.
    """
    return _TOKEN_RE.findall(text)


def load_lowtier_prompt(path: Path = LOWTIER_PREFIX_PATH) -> str:
    """The low-level prompt, refusing one that does not state run 7's fixed date and time."""
    try:
        text = path.read_text(encoding="utf-8").replace("\r\n", "\n").strip()
    except OSError as exc:
        raise core.InputError(f"cannot read {path}: {exc}") from exc
    if FIXED_NOW_TEXT not in text:
        raise core.InputError(
            f"{path} does not state the fixed date {FIXED_NOW_TEXT!r}, which C03 and C12 need"
        )
    return text


@dataclass(frozen=True, slots=True)
class CacheTurn:
    """One of run 2's ten turns."""

    id: str
    user: str


def load_cache_turns(path: Path = CACHE_TURNS_PATH) -> tuple[CacheTurn, ...]:
    """Run 2's ten turns, read when a run starts (run 2's task writes the file)."""
    if not path.is_file():
        raise core.InputError(f"{path} is missing; cache_turns.json holds run 2's ten turns")
    doc = core.load_json(path)
    turns = doc.get("turns")
    if doc.get("schema") != 1 or not isinstance(turns, list) or len(turns) != 10:
        raise core.InputError(f"{path}: cache_turns.json must be schema 1 with 10 turns")
    out = []
    for turn in turns:
        if not isinstance(turn, dict) or not all(
            isinstance(turn.get(k), str) and turn.get(k) for k in ("id", "user")
        ):
            raise core.InputError(f"{path}: cache_turns.json has a turn without id and user")
        out.append(CacheTurn(turn["id"], turn["user"]))
    return tuple(out)


@dataclass(frozen=True, slots=True)
class RunInputs:
    """What every row plan reads: the low-level prompt, the tool set and prompts, run 2's turns."""

    prompt: str
    tools: list
    prompts: dict
    turns: tuple[CacheTurn, ...]


def load_run_inputs(
    *,
    turns_path: Path = CACHE_TURNS_PATH,
    prompt_path: Path = LOWTIER_PREFIX_PATH,
    tools_path: Path = toolcall.TOOLS_SCHEMA_PATH,
    prompts_path: Path = toolcall.TOOL_PROMPTS_PATH,
) -> RunInputs:
    tools = toolcall.load_tools(tools_path)
    prompts = {p.id: p for p in toolcall.load_prompts(prompts_path, tools=tools)}
    return RunInputs(
        prompt=load_lowtier_prompt(prompt_path),
        tools=tools,
        prompts=prompts,
        turns=load_cache_turns(turns_path),
    )


def random_line(rng: random.Random, words: int = OPENING_WORDS) -> str:
    """A fresh opening line, so no cold request shares a prefix with anything cached."""
    picked = [rng.choice(_OPENING_WORDS) for _ in range(words)]
    return " ".join(picked).capitalize() + "."


# ---------------------------------------------------------------------------
# The server client
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChatTiming:
    """A streamed reply and the time its last tool-call fragment arrived."""

    result: core.ChatStreamResult
    t_last_tool_call: float | None


def _has_tool_delta(line: str) -> bool:
    payload = line[len("data:") :].strip()
    if not payload or payload == "[DONE]":
        return False
    try:
        event = json.loads(payload)
    except json.JSONDecodeError:
        return False
    if not isinstance(event, dict):
        return False
    return any((c.get("delta") or {}).get("tool_calls") for c in event.get("choices") or [])


class _RecordingResponse:
    """Passes the stream through, noting which `data:` lines carry a tool-call fragment."""

    def __init__(self, inner: object, tool_lines: list[int]):
        self._inner = inner
        self._entered: object = inner
        self._tool_lines = tool_lines

    def __enter__(self) -> _RecordingResponse:
        self._entered = self._inner.__enter__()
        return self

    def __exit__(self, *exc: object) -> object:
        return self._inner.__exit__(*exc)

    @property
    def status(self) -> object:
        return getattr(self._entered, "status", None)

    def __iter__(self) -> Iterator[bytes | str]:
        index = 0
        for raw in self._entered:
            line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            if line.startswith("data:"):
                if _has_tool_delta(line):
                    self._tool_lines.append(index)
                index += 1
            yield raw


class ServerClient:
    """llama-server's `/v1/models`, `/tokenize` and streamed `/v1/chat/completions`.

    `opener` defaults to `urllib.request.urlopen` and `clock` to `time.perf_counter`; tests pass
    fakes. Every call checks the status and raises `StreamError` on anything but 200.
    """

    def __init__(
        self,
        base_url: str,
        *,
        opener: Callable[..., object] | None = None,
        clock: core.Clock = time.perf_counter,
        timeout: float = 300.0,
    ):
        self.base_url = base_url.rstrip("/")
        self._opener = opener or urllib.request.urlopen
        self._clock = clock
        self._timeout = timeout

    def _json(self, path: str, body: Mapping[str, object] | None = None) -> dict:
        url = self.base_url + path
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if body is None else "POST",
        )
        try:
            with self._opener(request, timeout=self._timeout) as response:
                status = getattr(response, "status", None)
                if status != 200:
                    raise core.StreamError(f"{url} returned HTTP {status}")
                doc = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise core.StreamError(f"{url} returned HTTP {exc.code}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise core.StreamError(f"{url} failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise core.StreamError(f"{url} returned malformed JSON: {exc}") from exc
        if not isinstance(doc, dict):
            raise core.StreamError(f"{url} did not return a JSON object")
        return doc

    def models(self) -> list[str]:
        """The served model ids; llama-server lists its `--alias` here."""
        data = self._json("/v1/models").get("data")
        if not isinstance(data, list):
            raise core.StreamError(f"{self.base_url}/v1/models has no data list")
        return [str(entry.get("id")) for entry in data if isinstance(entry, dict)]

    def served_alias(self) -> str | None:
        """The one served alias, or None while nothing answers."""
        try:
            ids = self.models()
        except core.StreamError:
            return None
        return ids[0] if ids else None

    def tokenize(self, text: str) -> list[int]:
        """The server's own token ids for `text`, which the prefix builder counts."""
        tokens = self._json("/tokenize", {"content": text}).get("tokens")
        if not isinstance(tokens, list):
            raise core.StreamError(f"{self.base_url}/tokenize returned no token list")
        return tokens

    def chat(self, body: Mapping[str, object]) -> ChatTiming:
        """Stream one reply; the clock is read once before sending and once per `data:` line."""
        readings: list[float] = []
        tool_lines: list[int] = []

        def clock() -> float:
            value = self._clock()
            readings.append(value)
            return value

        def opener(request: urllib.request.Request, timeout: float) -> _RecordingResponse:
            return _RecordingResponse(self._opener(request, timeout=timeout), tool_lines)

        result = core.stream_chat(
            self.base_url, body, clock=clock, timeout=self._timeout, opener=opener
        )
        last = readings[1 + tool_lines[-1]] if tool_lines else None
        return ChatTiming(result, last)


# ---------------------------------------------------------------------------
# Row plans and rows
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RowSpec:
    """One request of a row plan."""

    kind: str
    prefix: str | None = None
    rep: int | None = None
    session: int | None = None
    turn: int | None = None
    prompt_id: str | None = None
    seed: int | None = None

    def as_row(self) -> dict[str, object]:
        """The spec's fields, which every row of `requests.jsonl` starts with."""
        return asdict(self)


def build_row_plan(
    *, busy: bool, prompts: Mapping[str, object] | Sequence[str], turns: Sequence[CacheTurn]
) -> list[RowSpec]:
    """Run 7's requests for one model and placement, in order.

    The busy pass on host-1070 runs the shorter set Run 7 gives it: the warm sessions and the tool
    prompts at seed 1, with no cold requests.
    """
    ids = list(prompts)
    plan = [RowSpec("warmup", rep=i) for i in range(1, WARMUPS + 1)]
    if not busy:
        plan += [
            RowSpec("cold", prefix=prefix, rep=rep)
            for prefix in (PREFIX_LOWTIER, PREFIX_CACHE)
            for rep in range(1, COLD_REPS + 1)
        ]
    plan += [
        RowSpec("warm", prefix=PREFIX_LOWTIER, session=session, turn=turn)
        for session in range(1, WARM_SESSIONS + 1)
        for turn in range(1, len(turns) + 1)
    ]
    seeds = BUSY_TOOL_SEEDS if busy else TOOL_SEEDS
    plan += [
        RowSpec("tool", prefix=PREFIX_LOWTIER, prompt_id=pid, seed=seed)
        for pid in ids
        for seed in seeds
    ]
    return plan


def _ms(later: float | None, earlier: float) -> float | None:
    return None if later is None else (later - earlier) * 1000.0


def request_row(
    spec: RowSpec,
    timing: ChatTiming,
    *,
    prefix_tokens: int | None,
    score: toolcall.PromptScore | None = None,
    opening_line: str | None = None,
) -> dict[str, object]:
    """One `requests.jsonl` row: client times beside the server's own `timings`."""
    result = timing.result
    timings = result.timings or {}
    row = spec.as_row()
    row.update(
        prefix_tokens=prefix_tokens,
        opening_line=opening_line,
        ttft_ms=_ms(result.t_first_delta, result.t_request),
        ttft_server_ms=timings.get("prompt_ms"),
        total_ms=_ms(result.t_end, result.t_request),
        tool_call_ms=_ms(timing.t_last_tool_call, result.t_request),
        cache_n=timings.get("cache_n"),
        prompt_n=timings.get("prompt_n"),
        prompt_ms=timings.get("prompt_ms"),
        predicted_n=timings.get("predicted_n"),
        predicted_ms=timings.get("predicted_ms"),
        predicted_per_second=timings.get("predicted_per_second"),
        finish_reason=result.finish_reason,
        served_model=result.model,
        reply=result.content,
        raw_tool_calls=[dict(c) for c in result.tool_calls],
        reasoning_flagged=bool(result.reasoning_content),
        reasoning_chars=len(result.reasoning_content),
        cut=False,
        error=None,
    )
    if score is not None:
        row.update(score.as_row())
    else:
        row.update(
            tool_valid_first=None, tool_valid_all=None, tool_n_calls=None, tool_problems=None
        )
    return row


def cut_row(spec: RowSpec, *, reason: str) -> dict[str, object]:
    """A row the window's end cut short: no request, or one that failed with the server gone."""
    row = spec.as_row()
    for key in (
        "prefix_tokens", "opening_line", "ttft_ms", "ttft_server_ms", "total_ms", "tool_call_ms",
        "cache_n", "prompt_n", "prompt_ms", "predicted_n", "predicted_ms", "predicted_per_second",
        "finish_reason", "served_model", "reply", "raw_tool_calls", "reasoning_flagged",
        "reasoning_chars", "tool_valid_first", "tool_valid_all", "tool_n_calls", "tool_problems",
    ):  # fmt: skip
        row[key] = None
    row.update(cut=True, error=reason)
    return row


@dataclass(frozen=True, slots=True)
class RowsOutcome:
    """How a row plan ended: `cut` when the server went away before the plan finished."""

    rows: int
    cut: bool


class _Prefixes:
    """The two prefixes and their token counts, built once per plan from the server's tokenizer."""

    def __init__(self, client: ServerClient, prompt: str):
        self._client = client
        self._text: dict[str, str] = {PREFIX_LOWTIER: prompt}
        self._tokens: dict[str, int] = {}

    def text(self, name: str) -> str:
        if name not in self._text:
            self._text[name] = core.build_prefix(CACHE_PREFIX_TOKENS, self._client.tokenize)
        return self._text[name]

    def tokens(self, name: str) -> int:
        if name not in self._tokens:
            self._tokens[name] = len(self._client.tokenize(self.text(name)))
        return self._tokens[name]


def run_rows(
    client: ServerClient,
    plan: Sequence[RowSpec],
    *,
    alias: str,
    inputs: RunInputs,
    writer: core.JsonlWriter,
    rng: random.Random,
    cut_on_error: bool,
) -> RowsOutcome:
    """Send each request of `plan` and write its row.

    With `cut_on_error`, a failed request is written as cut; if the server no longer serves
    `alias`, every remaining row is written as cut without a request, because the window ended.
    """
    prefixes = _Prefixes(client, inputs.prompt)
    sessions: dict[int, list[dict]] = {}
    cut = False
    any_cut = False
    for spec in plan:
        if cut:
            writer.write({**cut_row(spec, reason="window ended"), "alias": alias})
            continue
        session = None
        try:
            body, opening, prefix_tokens = _request_body(spec, alias, inputs, prefixes, rng)
            if spec.kind == "warm":
                session = sessions.setdefault(
                    spec.session, [{"role": "system", "content": inputs.prompt}]
                )
                session.append({"role": "user", "content": inputs.turns[spec.turn - 1].user})
                body["messages"] = [dict(m) for m in session]
            timing = client.chat(body)
        except core.StreamError as exc:
            if not cut_on_error:
                raise
            if session is not None:
                session.pop()
            any_cut = True
            writer.write({**cut_row(spec, reason=str(exc)), "alias": alias})
            cut = client.served_alias() != alias
            continue
        if session is not None:
            session.append({"role": "assistant", "content": timing.result.content})
        score = None
        if spec.kind == "tool":
            prompt = inputs.prompts[spec.prompt_id]
            score = toolcall.score_calls(prompt, timing.result.tool_calls, inputs.tools)
        row = request_row(
            spec, timing, prefix_tokens=prefix_tokens, score=score, opening_line=opening
        )
        writer.write({**row, "alias": alias})
    return RowsOutcome(rows=len(plan), cut=any_cut)


def _request_body(
    spec: RowSpec, alias: str, inputs: RunInputs, prefixes: _Prefixes, rng: random.Random
) -> tuple[dict, str | None, int | None]:
    if spec.kind == "warmup":
        return {**core.WARMUP_BODY, "model": alias}, None, None
    if spec.kind == "cold":
        line = random_line(rng)
        prefix = prefixes.text(spec.prefix)
        body = {
            "model": alias,
            "messages": [
                {"role": "system", "content": f"{line}\n\n{prefix}"},
                {"role": "user", "content": COLD_QUESTION},
            ],
            "max_tokens": REPLY_TOKENS,
            "cache_prompt": True,
        }
        return body, line, prefixes.tokens(spec.prefix)
    if spec.kind == "warm":
        body = {"model": alias, "messages": [], "max_tokens": REPLY_TOKENS, "cache_prompt": True}
        return body, None, prefixes.tokens(PREFIX_LOWTIER)
    if spec.kind == "tool":
        prompt = inputs.prompts[spec.prompt_id]
        body = {
            "model": alias,
            "messages": [
                {"role": "system", "content": inputs.prompt},
                {"role": "user", "content": prompt.text},
            ],
            "tools": inputs.tools,
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "max_tokens": TOOL_REPLY_TOKENS,
            "seed": spec.seed,
            "cache_prompt": True,
        }
        return body, None, prefixes.tokens(PREFIX_LOWTIER)
    raise core.InputError(f"unknown row kind {spec.kind!r}")


# ---------------------------------------------------------------------------
# The slice search (gpu-4g and gpu-2g)
# ---------------------------------------------------------------------------


def bisect_layers(fits: Callable[[int], bool], total: int) -> int | None:
    """The largest layer count in 0..`total` for which `fits` holds, on a monotone footprint.

    Full offload is tried first, so a model under the slice at full offload costs one probe; then
    `-ngl 0`, since buffers can stay on the card even there; None when not even 0 fits.
    """
    if fits(total):
        return total
    if not fits(0):
        return None
    lo, hi = 0, total
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid
    return lo


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """One served layer count: its footprint by difference, or the error that stopped it."""

    ngl: int
    footprint_mib: float | None
    total_layers: int | None
    error: str | None


@dataclass(frozen=True, slots=True)
class SliceResult:
    """The recorded slice search: the count `small-serve --ngl` takes, or None if none fits."""

    path: Path
    chosen_ngl: int | None
    total_layers: int | None
    under_at_full: bool


def _slice_mib(placement: str, arms: Mapping[str, object] | None = None) -> int:
    doc = arms if arms is not None else core.load_small_arms()
    cfg = doc["placements"].get(placement)
    if cfg is None or cfg.get("where") != "local" or "slice_mib" not in cfg:
        sliced = [p for p, c in doc["placements"].items() if "slice_mib" in c]
        raise core.PlacementError(
            f"placement {placement!r} has no slice; the search is for {sliced}"
        )
    return int(cfg["slice_mib"])


def run_slice_search(
    results_root: Path,
    run: str,
    model_id: str,
    placement: str,
    *,
    probe: Callable[[int], ProbeResult],
    git_sha: str | None = None,
    arms: Mapping[str, object] | None = None,
) -> SliceResult:
    """Find and record the layer count `small-serve --ngl` takes for a sliced placement."""
    slice_mib = _slice_mib(placement, arms)
    claim = core.claim_output(
        results_root,
        run,
        SUITE,
        f"slice/{model_id}/{placement}",
        git_sha=git_sha or core.current_git_sha(),
    )
    probes: list[ProbeResult] = []

    def measure(ngl: int) -> ProbeResult:
        result = probe(ngl)
        probes.append(result)
        return result

    full = measure(FULL_NGL)
    total = full.total_layers
    under = full.footprint_mib is not None and full.footprint_mib <= slice_mib
    chosen: int | None
    if under:
        chosen = total if total is not None else FULL_NGL
    elif total is None:
        chosen = None
    else:
        cache = {total: full}

        def fits(ngl: int) -> bool:
            if ngl not in cache:
                cache[ngl] = measure(ngl)
            found = cache[ngl].footprint_mib
            return found is not None and found <= slice_mib

        chosen = bisect_layers(fits, total)
    path = claim.path / "slice.json"
    core.write_json_exclusive(
        path,
        {
            **claim.stamp(),
            "model": model_id,
            "placement": placement,
            "slice_mib": slice_mib,
            "chosen_ngl": chosen,
            "total_layers": total,
            "under_at_full": under,
            "probes": [asdict(p) for p in probes],
        },
    )
    return SliceResult(path, chosen, total, under)


def make_serve_probe(
    model_id: str,
    placement: str,
    *,
    speech: Path,
    pins_path: Path,
    models_dirs: Mapping[str, Path | None],
    serve: Callable[..., object] = core.serve_small,
    unserve: Callable[..., object] = core.unserve_small,
    gpu_used_mib: Callable[[], float],
    logs: Callable[[str], str],
    container_state: Callable[[str], str] | None = None,
) -> Callable[[int], ProbeResult]:
    """A probe that serves `model_id` at a layer count and measures its footprint by difference.

    Each count is served into its own `small/<model>/<placement>-slice<n>/` directory, so no
    probe's record is ever reused, and is removed again before the next.
    """
    state = container_state or (lambda name: bench_host.probe_container_state(name).state)

    def probe(ngl: int) -> ProbeResult:
        out_dir = core.small_serve_dir(speech, model_id, placement, f"slice{ngl}")
        before = gpu_used_mib()
        try:
            serve(
                model_id,
                placement,
                out_dir=out_dir,
                pins_path=pins_path,
                models_dirs=models_dirs,
                ngl=ngl,
            )
        except core.ServeFailedError as exc:
            if state(core.SMALL_CONTAINER) != "absent" and out_dir.is_dir():
                unserve(out_dir)
            return ProbeResult(ngl, None, None, str(exc))
        after = gpu_used_mib()
        found = _OFFLOADED_RE.findall(logs(core.SMALL_CONTAINER))
        unserve(out_dir)
        total = int(found[-1][1]) if found else None
        return ProbeResult(ngl, after - before, total, None)

    return probe


# ---------------------------------------------------------------------------
# E4B's reload
# ---------------------------------------------------------------------------


def run_e4b_reload(
    results_root: Path,
    run: str,
    bench_run: str,
    *,
    models_dir: Path | None,
    bench_main: Callable[[list[str]], int] = bench_host.main,
    chat: Callable[..., core.ChatStreamResult] = core.stream_chat,
    clock: core.Clock = time.perf_counter,
    git_sha: str | None = None,
    rounds: int = E4B_ROUNDS,
) -> Path:
    """Unserve and serve `c0` `rounds` times, timing each reload to the first warm-up token.

    This is the cost of a level change back to E4B. `bench_host` does the serving, under the
    window's bench run id, so its snapshot and image checks apply unchanged.
    """
    core.validate_name(bench_run, "bench run")
    root = core.speech_root(results_root, run) / SUITE
    root.mkdir(parents=True, exist_ok=True)
    common = ["--run", bench_run, "--results-root", str(results_root)]
    serve_args = ["serve", E4B_ARM, *common]
    if models_dir is not None:
        serve_args += ["--models-dir", str(models_dir)]
    path = root / "e4b_reload.jsonl"
    stamp = {"run": run, "suite": SUITE, "git_sha": git_sha or core.current_git_sha()}
    with core.JsonlWriter(path, stamp=stamp) as writer:
        for rep in range(1, rounds + 1):
            t0 = clock()
            if bench_main(["unserve", *common, "--arm", E4B_ARM]) != 0:
                raise core.BenchRunError(f"bench_host unserve {E4B_ARM} failed in round {rep}")
            t1 = clock()
            if bench_main(serve_args) != 0:
                raise core.BenchRunError(f"bench_host serve {E4B_ARM} failed in round {rep}")
            warm = chat(BENCH_LLM_URL, dict(core.WARMUP_BODY), clock=clock)
            if warm.t_first_delta is None:
                raise core.BenchRunError(f"the warm-up after serve {E4B_ARM} returned no token")
            writer.write(
                {
                    "kind": "e4b_reload",
                    "arm": E4B_ARM,
                    "rep": rep,
                    "bench_run": bench_run,
                    "unserve_s": t1 - t0,
                    "reload_s": warm.t_first_delta - t1,
                    "warmup_ttft_s": warm.ttft_s,
                }
            )
    return path


# ---------------------------------------------------------------------------
# Local placements
# ---------------------------------------------------------------------------


class Sampler(Protocol):
    """bench_host's `GpuSampler`, or a fake."""

    def start(self) -> None:
        """Begin sampling GPU memory into the CSV."""
        ...

    def stop(self) -> None:
        """Stop sampling."""
        ...


class LocalProbes(Protocol):
    """The host reads a local placement needs; tests pass a fake."""

    def logs(self, name: str) -> str:
        """The container's log text."""
        ...

    def container_peak_mib(self, name: str) -> tuple[float | None, str]:
        """The container's peak host memory and the method that read it."""
        ...

    def gpu_used_mib(self) -> float:
        """GPU memory in use now."""
        ...

    def sampler(self, path: Path) -> Sampler:
        """A GPU sampler writing `vram.csv` at `path`."""
        ...


class HostProbes:
    """The real reads: docker logs, the cgroup's `memory.peak`, nvidia-smi."""

    def logs(self, name: str) -> str:
        """`docker logs`, both streams."""
        try:
            stdout, stderr = bench_host.docker_logs(name)
        except bench_host.DockerError:
            return ""
        return stdout + stderr

    def container_peak_mib(self, name: str) -> tuple[float | None, str]:
        """`memory.peak` through `docker exec`, else `docker stats`' current usage."""
        peak = _run(["docker", "exec", name, "cat", "/sys/fs/cgroup/memory.peak"])
        if peak is not None and peak.strip().isdigit():
            return int(peak.strip()) / (1024 * 1024), "cgroup memory.peak"
        stats = _run(["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", name])
        value = parse_mem_usage(stats or "")
        if value is not None:
            return value, "docker stats current usage (memory.peak unreadable)"
        return None, "unavailable"

    def gpu_used_mib(self) -> float:
        """One nvidia-smi reading."""
        out = _run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        if out is None or not out.strip():
            raise core.SpeechBenchError("nvidia-smi returned no memory reading")
        return float(out.strip().splitlines()[0])

    def sampler(self, path: Path) -> Sampler:
        """bench_host's sampler."""
        return bench_host.GpuSampler(path)


def _run(argv: list[str]) -> str | None:
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=30, shell=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout if proc.returncode == 0 else None


def parse_mem_usage(text: str) -> float | None:
    """MiB from `docker stats`' `MemUsage`, such as `1.5GiB / 31GiB`."""
    match = _MEM_USAGE_RE.search(text)
    if match is None:
        return None
    scale = {"K": 1 / 1024, "M": 1.0, "G": 1024.0}[match.group(2)[0]]
    return float(match.group(1)) * scale


@dataclass(frozen=True, slots=True)
class LocalRequest:
    """One local model and placement; `bench_run` names the window's `desktop` step."""

    results_root: Path
    run: str
    bench_run: str | None
    model: str
    placement: str
    unserve: bool = False


def desktop_baseline_mib(results_root: Path, bench_run: str) -> float:
    """The `desktop` step of the window's `vram_steps.json`: GPU memory with nothing loaded."""
    core.validate_name(bench_run, "bench run")
    path = results_root.resolve() / bench_run / "session" / "vram_steps.json"
    if not path.is_file():
        raise core.BenchRunError(f"no {path}; run `bench_host vram-step --label desktop` first")
    doc = core.load_json(path)
    for step in doc.get("steps", []):
        if step.get("label") == DESKTOP_STEP and step.get("median_mib") is not None:
            return step["median_mib"]
    raise core.BenchRunError(f"{path} has no {DESKTOP_STEP!r} step")


def peak_from_csv(path: Path) -> float | None:
    """The largest `memory_used_mib` in a sampler's CSV."""
    if not path.is_file():
        return None
    values = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                values.append(float(row["memory_used_mib"]))
            except (KeyError, TypeError, ValueError):
                continue
    return max(values) if values else None


def _number(value: float | None) -> float | int | None:
    if value is None:
        return None
    return int(value) if float(value).is_integer() else value


def run_local(
    request: LocalRequest,
    *,
    client: ServerClient,
    inputs: RunInputs,
    probes: LocalProbes,
    clock: core.Clock = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
    rows_for: Callable[[str], Sequence[RowSpec]] | None = None,
    unserve: Callable[..., float] = core.unserve_small,
    git_sha: str | None = None,
) -> Path:
    """Run 7's rows against `mist-bench-small` and the memory of the placement's method."""
    if request.placement not in core.LOCAL_PLACEMENTS:
        raise core.PlacementError(
            f"placement {request.placement!r} is not a local one; host-1070 rows take --remote-url"
        )
    arms = core.load_small_arms()
    placement_cfg = arms["placements"][request.placement]
    speech = core.speech_root(request.results_root, request.run)
    serve_dir = core.small_serve_dir(speech, request.model, request.placement)
    serve_path = serve_dir / "serve.json"
    if not serve_path.is_file():
        raise core.SpeechBenchError(f"no {serve_path}; run small-serve for this model first")
    serve_record = core.load_json(serve_path)
    alias = core.small_alias(request.model, request.placement)
    served = client.served_alias()
    if served != alias:
        raise AliasError(f"{client.base_url} serves {served!r}, not {alias}; refusing")
    gpu = bool(placement_cfg.get("gpus"))
    baseline = None
    if gpu:
        if request.bench_run is None:
            raise core.BenchRunError("--bench-run is required for a GPU placement")
        baseline = desktop_baseline_mib(request.results_root, request.bench_run)

    claim = core.claim_output(
        request.results_root,
        request.run,
        SUITE,
        f"{request.model}/{request.placement}",
        git_sha=git_sha or core.current_git_sha(),
    )
    plan = (
        rows_for(alias)
        if rows_for
        else build_row_plan(busy=False, prompts=inputs.prompts, turns=inputs.turns)
    )
    sampler = probes.sampler(claim.path / "vram.csv") if gpu else None
    if sampler is not None:
        sampler.start()
    try:
        with claim.jsonl("requests.jsonl") as writer:
            run_rows(
                client,
                plan,
                alias=alias,
                inputs=inputs,
                writer=writer,
                rng=rng or random.Random(DEFAULT_SEED),
                cut_on_error=False,
            )
    finally:
        if sampler is not None:
            sampler.stop()

    gpu_peak = peak_from_csv(claim.path / "vram.csv") if gpu else None
    host_peak, host_method = probes.container_peak_mib(core.SMALL_CONTAINER)
    log_text = probes.logs(core.SMALL_CONTAINER)
    core.write_text_exclusive(claim.path / "server.log", log_text)
    variant = _CPU_VARIANT_RE.findall(log_text)
    slice_record = speech / SUITE / "slice" / request.model / request.placement / "slice.json"
    memory: dict[str, object] = {
        **claim.stamp(),
        "alias": alias,
        "model": request.model,
        "placement": request.placement,
        "memory_method": GPU_MEMORY_METHOD if gpu else HOST_MEMORY_METHOD,
        "gpu_baseline_mib": _number(baseline),
        "gpu_peak_mib": _number(gpu_peak),
        "gpu_by_difference_mib": (
            _number(gpu_peak - baseline) if gpu_peak is not None and baseline is not None else None
        ),
        "host_peak_mib": host_peak,
        "host_peak_method": host_method,
        "cpu_variant": variant[-1] if variant else None,
        "load_s": serve_record.get("load_s"),
        "ngl": serve_record.get("ngl"),
        "image": serve_record.get("image"),
        "slice_chosen_ngl": (
            core.load_json(slice_record).get("chosen_ngl") if slice_record.is_file() else None
        ),
        "stop_s": None,
        "unload_s": None,
        "gpu_after_unload_mib": None,
        "memory_returned_mib": None,
    }
    if request.unserve:
        memory.update(_unserve_and_wait(serve_dir, gpu, baseline, probes, unserve, clock, sleep))
    core.write_json_exclusive(claim.path / "memory.json", memory)
    return claim.path


def _unserve_and_wait(
    serve_dir: Path,
    gpu: bool,
    baseline: float | None,
    probes: LocalProbes,
    unserve: Callable[..., float],
    clock: core.Clock,
    sleep: Callable[[float], None],
) -> dict[str, object]:
    """Unload time: from removing the container to the memory returned, read from the GPU."""
    before = probes.gpu_used_mib() if gpu else None
    t0 = clock()
    stop_s = unserve(serve_dir)
    if not gpu or baseline is None:
        return {"stop_s": stop_s, "unload_s": clock() - t0}
    used = probes.gpu_used_mib()
    while used > baseline + UNLOAD_SLACK_MIB and clock() - t0 < UNLOAD_TIMEOUT_S:
        sleep(UNLOAD_POLL_S)
        used = probes.gpu_used_mib()
    return {
        "stop_s": stop_s,
        "unload_s": clock() - t0,
        "gpu_after_unload_mib": _number(used),
        "memory_returned_mib": _number(before - used),
    }


# ---------------------------------------------------------------------------
# --remote-url: the host-1070 rows
# ---------------------------------------------------------------------------


def default_remote_aliases() -> list[str]:
    """Every alias the H6 script may serve, in its schedule's order."""
    return host1070_script.expected_aliases()


def validate_remote_aliases(aliases: Sequence[str]) -> tuple[str, ...]:
    """Refuse an alias that is not a small model on a host-1070 placement and pass."""
    arms = core.load_small_arms()
    for alias in aliases:
        model, _, placement = alias.partition("@")
        base = placement
        for suffix in _PASS_SUFFIXES:
            if placement == f"{host1070_script.SHARED}-{suffix}":
                base = host1070_script.SHARED
        valid_placement = base in core.HOST1070_PLACEMENTS and placement != host1070_script.SHARED
        if model not in arms["models"] or not valid_placement:
            raise core.PlacementError(
                f"{alias!r} is not a small model on a host-1070 placement and pass"
            )
    return tuple(aliases)


@dataclass(frozen=True, slots=True)
class RemoteRequest:
    """The host-1070 server and the aliases to wait for there."""

    results_root: Path
    run: str
    base_url: str
    expect: tuple[str, ...] = ()
    poll_interval_s: float = DEFAULT_POLL_S
    wait_s: float = DEFAULT_WAIT_S


@dataclass(frozen=True, slots=True)
class RemoteStatus:
    """Which expected aliases ran whole, which were cut and which never appeared."""

    done: tuple[str, ...]
    cut: tuple[str, ...]
    not_served: tuple[str, ...]


def run_remote(
    request: RemoteRequest,
    *,
    client: ServerClient,
    inputs: RunInputs,
    rows_for: Callable[[str], Sequence[RowSpec]] | None = None,
    clock: core.Clock = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
    git_sha: str | None = None,
    log: Callable[[str], None] = print,
) -> RemoteStatus:
    """Wait for each expected alias on host-1070 and run its rows while it is served.

    The script's schedule decides the order, so any expected alias may come next; an alias this
    run was not given stops the run before a request is sent. It gives up after `wait_s` with no
    new alias, and reports the rest as not served.
    """
    expect = validate_remote_aliases(request.expect or default_remote_aliases())
    core.speech_root(request.results_root, request.run)
    sha = git_sha or core.current_git_sha()
    generator = rng or random.Random(DEFAULT_SEED)
    plan_for = rows_for or (
        lambda alias: build_row_plan(
            busy=alias.endswith("-busy"), prompts=inputs.prompts, turns=inputs.turns
        )
    )
    pending = list(expect)
    done: list[str] = []
    cut: list[str] = []
    last_progress = clock()
    while pending:
        served = client.served_alias()
        if served is not None and served not in expect:
            raise AliasError(f"{client.base_url} serves {served!r}, which this run was not given")
        if served in pending:
            log(f"[lowtier] {served} is up; running its rows")
            outcome = _run_remote_alias(request, served, client, inputs, plan_for, generator, sha)
            pending.remove(served)
            (cut if outcome.cut else done).append(served)
            last_progress = clock()
            continue
        if clock() - last_progress > request.wait_s:
            log(f"[lowtier] no new alias in {request.wait_s:.0f} s; stopping")
            break
        sleep(request.poll_interval_s)
    return RemoteStatus(tuple(done), tuple(cut), tuple(pending))


def _run_remote_alias(
    request: RemoteRequest,
    alias: str,
    client: ServerClient,
    inputs: RunInputs,
    plan_for: Callable[[str], Sequence[RowSpec]],
    rng: random.Random,
    git_sha: str,
) -> RowsOutcome:
    model, _, placement = alias.partition("@")
    claim = core.claim_output(
        request.results_root, request.run, SUITE, f"{model}/{placement}", git_sha=git_sha
    )
    with claim.jsonl("requests.jsonl") as writer:
        outcome = run_rows(
            client,
            plan_for(alias),
            alias=alias,
            inputs=inputs,
            writer=writer,
            rng=rng,
            cut_on_error=True,
        )
    core.write_json_exclusive(
        claim.path / "memory.json",
        {
            **claim.stamp(),
            "alias": alias,
            "model": model,
            "placement": placement,
            "base_url": client.base_url,
            "memory_method": REMOTE_MEMORY_METHOD,
            "cut": outcome.cut,
        },
    )
    return outcome


# ---------------------------------------------------------------------------
# remote-memory
# ---------------------------------------------------------------------------


def _diff(a: object, b: object) -> float | int | None:
    if isinstance(a, int | float) and isinstance(b, int | float):
        return a - b
    return None


def memory_rows(summary: Mapping[str, object]) -> list[dict[str, object]]:
    """The H6 summary's figures per model and placement: windows, layer counts, skips, restore."""
    if summary.get("schema") != 1:
        raise core.InputError(f"the host-1070 summary has schema {summary.get('schema')!r}, not 1")
    for key in ("windows", "searches", "skips"):
        if not isinstance(summary.get(key), list):
            raise core.InputError(f"the host-1070 summary has no {key} list")
    card = summary.get("card_total_mib")
    rows: list[dict[str, object]] = []
    before = summary.get("before") if isinstance(summary.get("before"), dict) else {}
    rows.append(
        {
            "kind": "host",
            "variant": summary.get("variant"),
            "started_utc": summary.get("started_utc"),
            "finished_utc": summary.get("finished_utc"),
            "threads": summary.get("threads"),
            "card_total_mib": card,
            "gpu_used_before_mib": before.get("gpu_used_mib"),
            "free_available_before_mib": before.get("free_available_mib"),
        }
    )
    for window in summary["windows"]:
        model, _, placement = str(window.get("alias", "")).partition("@")
        row = {"kind": "window", "model": model or window.get("model"), "placement": placement}
        row.update({k: v for k, v in window.items() if k not in ("model", "placement")})
        row.update(
            base_placement=window.get("placement"),
            gpu_load_mib=_diff(window.get("gpu_after_load_mib"), window.get("gpu_before_mib")),
            gpu_free_at_peak_mib=_diff(card, window.get("gpu_peak_mib")),
            host_used_mib=_diff(
                window.get("free_available_before_mib"), window.get("free_available_peak_mib")
            ),
            memory_returned_mib=_diff(
                window.get("gpu_after_load_mib"), window.get("gpu_after_unload_mib")
            ),
        )
        rows.append(row)
    searched: dict[str, list[dict]] = {}
    for step in summary["searches"]:
        searched.setdefault(str(step.get("model")), []).append(step)
    for model, steps in searched.items():
        fitting = [s["ngl"] for s in steps if s.get("fits") and isinstance(s.get("ngl"), int)]
        rows.append(
            {
                "kind": "layers",
                "model": model,
                "placement": host1070_script.SHARED,
                "chosen_ngl": max(fitting) if fitting else None,
                "probes": [
                    {k: s.get(k) for k in ("ngl", "fits", "gpu_free_mib", "fa", "cache")}
                    for s in steps
                ],
            }
        )
    for skip in summary["skips"]:
        rows.append(
            {
                "kind": "skip",
                "model": skip.get("model"),
                "placement": skip.get("placement"),
                "reason": skip.get("reason"),
            }
        )
    restore = summary.get("restore") if isinstance(summary.get("restore"), dict) else {}
    diff = restore.get("inspect_diff") if isinstance(restore.get("inspect_diff"), dict) else {}
    rows.append(
        {
            "kind": "restore",
            "started": restore.get("started"),
            "health": restore.get("health"),
            "inspect_diff_empty": diff.get("empty"),
            "inspect_containers": diff.get("containers"),
            "exit_code": summary.get("exit_code"),
        }
    )
    return rows


def run_remote_memory(
    results_root: Path,
    run: str,
    *,
    summary_path: Path | None = None,
    git_sha: str | None = None,
) -> Path:
    """Write `<speech>/host1070/memory.jsonl` from the H6 summary the host-1070 session sent."""
    host_dir = core.speech_root(results_root, run) / "host1070"
    source = summary_path if summary_path is not None else host_dir / "summary.json"
    rows = memory_rows(core.load_json(source))
    host_dir.mkdir(parents=True, exist_ok=True)
    path = host_dir / "memory.jsonl"
    stamp = {"run": run, "suite": REMOTE_MEMORY_SUITE, "git_sha": git_sha or core.current_git_sha()}
    with core.JsonlWriter(path, stamp=stamp) as writer:
        for row in rows:
            writer.write({**row, "source": str(source)})
    return path


# ---------------------------------------------------------------------------
# CLI, plan hooks and selftest
# ---------------------------------------------------------------------------


def _env_path(name: str) -> Path | None:
    raw = os.environ.get(name)
    return Path(raw) if raw else None


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "lowtier", help="run 7: a small model on a placement, local or on host-1070"
    )
    core.add_results_args(parser)
    parser.add_argument("--bench-run", help="bench_host run id of this window, e.g. mh1-h5")
    parser.add_argument("--model", help="a model id from small_arms.json")
    parser.add_argument("--placement", help=f"one of {', '.join(core.LOCAL_PLACEMENTS)}")
    parser.add_argument(
        "--slice-search",
        action="store_true",
        help="bisect the layer count for gpu-4g or gpu-2g and record it",
    )
    parser.add_argument(
        "--unserve", action="store_true", help="after the rows, unserve and time the unload"
    )
    parser.add_argument(
        "--e4b-reload", action="store_true", help="time E4B's reload after unserve, three times"
    )
    parser.add_argument("--remote-url", help="host-1070's server, e.g. http://<host>:8090")
    parser.add_argument(
        "--expect",
        action="append",
        default=[],
        metavar="ALIAS",
        help="an alias to wait for on --remote-url (default: every alias of the H6 schedule)",
    )
    parser.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_S)
    parser.add_argument(
        "--wait", type=float, default=DEFAULT_WAIT_S, help="seconds to wait for a new alias"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="seed of the opening lines")
    parser.add_argument("--pins", type=Path, help="pin file (default <speech>/host/pins.json)")
    parser.add_argument(
        "--small-models-dir",
        type=Path,
        default=_env_path(SMALL_MODELS_DIR_ENV),
        help=f"small models directory (env {SMALL_MODELS_DIR_ENV})",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=_env_path("MODELS_DIR"),
        help="main models directory (env MODELS_DIR)",
    )
    parser.set_defaults(func=cmd_lowtier)

    memory = subparsers.add_parser(
        "remote-memory", help="read the H6 summary into memory figures per model and placement"
    )
    core.add_results_args(memory)
    memory.add_argument(
        "--summary", type=Path, help="the H6 summary (default <speech>/host1070/summary.json)"
    )
    memory.set_defaults(func=cmd_remote_memory)


def _mode(args: argparse.Namespace) -> str:
    target = args.model is not None or args.placement is not None
    if args.remote_url is not None:
        if target or args.slice_search or args.e4b_reload or args.unserve:
            raise core.SpeechBenchError(
                "--remote-url runs the host-1070 rows by alias; it takes no --model, --placement, "
                "--slice-search, --e4b-reload or --unserve"
            )
        return "remote"
    if args.expect:
        raise core.SpeechBenchError("--expect is for --remote-url")
    if args.e4b_reload:
        if target or args.slice_search or args.unserve:
            raise core.SpeechBenchError("--e4b-reload takes no --model, --placement or --unserve")
        if args.bench_run is None:
            raise core.SpeechBenchError("--e4b-reload needs --bench-run")
        return "e4b"
    if args.model is None or args.placement is None:
        raise core.SpeechBenchError("lowtier needs --model and --placement, or --remote-url")
    if args.slice_search:
        if args.unserve:
            raise core.SpeechBenchError("--slice-search unserves each count itself")
        return "slice"
    return "local"


def cmd_lowtier(args: argparse.Namespace) -> int:
    mode = _mode(args)
    core.speech_root(args.results_root, args.run)
    if mode == "e4b":
        path = run_e4b_reload(
            args.results_root, args.run, args.bench_run, models_dir=args.models_dir
        )
        print(f"E4B reload timed {E4B_ROUNDS} times; rows at {path}")
        return 0
    if mode == "slice":
        speech = core.speech_root(args.results_root, args.run)
        probes = HostProbes()
        probe = make_serve_probe(
            args.model,
            args.placement,
            speech=speech,
            pins_path=args.pins if args.pins is not None else speech / "host" / "pins.json",
            models_dirs={"main": args.models_dir, "small": args.small_models_dir},
            gpu_used_mib=probes.gpu_used_mib,
            logs=probes.logs,
        )
        result = run_slice_search(
            args.results_root, args.run, args.model, args.placement, probe=probe
        )
        print(f"{args.model}@{args.placement}: -ngl {result.chosen_ngl}; record at {result.path}")
        return 0 if result.chosen_ngl is not None else 1

    inputs = load_run_inputs()
    rng = random.Random(args.seed)
    if mode == "remote":
        request = RemoteRequest(
            results_root=args.results_root,
            run=args.run,
            base_url=args.remote_url,
            expect=tuple(args.expect),
            poll_interval_s=args.poll_interval,
            wait_s=args.wait,
        )
        status = run_remote(request, client=ServerClient(args.remote_url), inputs=inputs, rng=rng)
        print(
            f"host-1070 rows: done {list(status.done)}, cut {list(status.cut)}, "
            f"not served {list(status.not_served)}"
        )
        return 0
    request = LocalRequest(
        results_root=args.results_root,
        run=args.run,
        bench_run=args.bench_run,
        model=args.model,
        placement=args.placement,
        unserve=args.unserve,
    )
    out = run_local(
        request,
        client=ServerClient(core.SMALL_BASE_URL),
        inputs=inputs,
        probes=HostProbes(),
        rng=rng,
    )
    print(f"lowtier rows and memory at {out}")
    return 0


def cmd_remote_memory(args: argparse.Namespace) -> int:
    path = run_remote_memory(args.results_root, args.run, summary_path=args.summary)
    print(f"host-1070 memory figures at {path}")
    return 0


def container_argv(ctx: core.PlanContext) -> list[tuple[str, list[str]]]:
    """None: `lowtier` runs no container of its own.

    Local placements are served by `small-serve`, whose argv core prints, and the host-1070
    servers are started by the H6 script that `plan --host-1070-script` writes.
    """
    return []


def validate_inputs(repo_root: Path) -> list[str]:
    """Problems with run 7's committed inputs; run 2's turns are only noted while still unwritten."""
    speech = repo_root / core.SPEECH_DIR_REL
    problems = toolcall.validate_tool_files(
        speech / toolcall.TOOLS_SCHEMA_PATH.name, speech / toolcall.TOOL_PROMPTS_PATH.name
    )
    try:
        prompt = load_lowtier_prompt(speech / LOWTIER_PREFIX_PATH.name)
    except core.InputError as exc:
        problems.append(str(exc))
    else:
        n = len(approx_tokenize(prompt))
        if abs(n - LOWTIER_TOKENS) > core.PREFIX_TOLERANCE:
            problems.append(
                f"{LOWTIER_PREFIX_PATH.name} is {n} tokens by the approximate count, not "
                f"{LOWTIER_TOKENS} within {core.PREFIX_TOLERANCE}"
            )
    turns = speech / CACHE_TURNS_PATH.name
    if turns.exists():
        try:
            load_cache_turns(turns)
        except core.InputError as exc:
            problems.append(str(exc))
    else:
        print(
            f"[note] lowtier: {turns.name} is not written yet; `lowtier` refuses to run until it is"
        )
    try:
        arms = core.load_small_arms(speech / core.SMALL_ARMS_PATH.name)
        validate_remote_aliases(host1070_script.expected_aliases(arms))
    except (core.InputError, core.PlacementError, KeyError) as exc:
        problems.append(f"host-1070 aliases: {exc}")
    return problems


def _check(condition: bool, message: str) -> None:
    # Explicit raise rather than `assert`, which `python -O` strips.
    if not condition:
        raise AssertionError(message)


class _FixtureResponse:
    def __init__(self, text: str):
        self.status = 200
        self._lines = [line + "\n" for line in text.splitlines()]

    def __enter__(self) -> _FixtureResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def __iter__(self) -> Iterator[str]:
        return iter(self._lines)


@dataclass
class _Ticks:
    values: list[float] = field(default_factory=list)

    def __call__(self) -> float:
        return self.values.pop(0)


def selftest() -> None:
    fixtures = core.SPEECH_FIXTURES_DIR / "lowtier"
    _check(abs(len(approx_tokenize(load_lowtier_prompt())) - LOWTIER_TOKENS) <= 25, "prompt size")
    tools = toolcall.load_tools()
    prompts = {p.id: p for p in toolcall.load_prompts(tools=tools)}
    for line in (fixtures / "tool_calls.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        score = toolcall.score_calls(prompts[case["prompt_id"]], case["calls"], tools)
        _check(
            (score.valid_first, score.valid_all) == (case["valid_first"], case["valid_all"]),
            f"tool-call fixture case {case['case']!r} scored {score}",
        )
    text = (fixtures / "sse_tool_call.txt").read_text(encoding="utf-8")
    ticks = _Ticks([100.0, 100.05, 100.2, 100.25, 100.3, 100.42, 100.5, 100.51])
    client = ServerClient(
        "http://fixture", opener=lambda r, timeout: _FixtureResponse(text), clock=ticks
    )
    timing = client.chat({"messages": []})
    _check(len(timing.result.tool_calls) == 2, "the SSE fixture holds two tool calls")
    _check(timing.t_last_tool_call == 100.42, "the last tool-call fragment time")
    _check((timing.result.timings or {}).get("cache_n") == 1488, "timings read from the stream")
    _check(bisect_layers(lambda n: 500 + 100 * n <= 4096, 40) == 35, "slice bisection")
    _check(bisect_layers(lambda n: False, 40) is None, "slice bisection with nothing fitting")
    rows = memory_rows(core.load_json(fixtures / "host1070_summary.json"))
    idle = next(r for r in rows if r.get("alias") == "gemma4-e2b@gtx1070-shared-idle")
    _check(idle["gpu_load_mib"] == 352, "remote-memory's load figure")
    layers = {r["model"]: r["chosen_ngl"] for r in rows if r["kind"] == "layers"}
    _check(layers.get("gemma4-e2b") == 9, "remote-memory's layer count")
    aliases = host1070_script.expected_aliases()
    _check(validate_remote_aliases(aliases) == tuple(aliases), "the H6 aliases are host-1070 ones")
