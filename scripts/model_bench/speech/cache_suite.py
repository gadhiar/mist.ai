"""Run 2, the prompt cache: the `cache` subcommand (mist-model-host plan v2, task T4).

`cache` measures whether E4B reuses a cached prefix under one cache arm, against the arm
`bench_host serve` has already started as `mist-bench-llm`:

- `c0`, production as it runs today, which keeps b11151's default of 32 checkpoints per slot at
  least 8,192 tokens apart;
- `c0-ckpt`, production plus `--checkpoint-min-step 0`, so a checkpoint is saved before every
  last user message;
- `c0-swafull`, production plus `--swa-full`, which keeps the whole sliding-window cache.

Before anything is written it refuses unless the served container is the named arm, checked as
bench_host's `run` checks it: the container's `Args` against the argv bench_host builds for the
arm, `/props`' model path against the arm's file, and the container's image against the arm's
image reference, read under the `--bench-run` window whose snapshot must exist.

Per arm: 3 sessions of the 10 turns in `cache_turns.json`, each session a fresh conversation on
the same 6,100-token prefix (T1's builder, counted with the server's own `/tokenize`), each turn
appending the previous reply (at most 64 tokens) and the next user message. Every request is
streamed with `cache_prompt` true, temperature 0 and a fixed seed; `cache_n`, `prompt_n` and
`prompt_ms` come from the `timings` of the stream's final event. The server's log is kept beside
the rows and searched for "forcing full prompt re-processing" and for any mmproj load.

Stdlib only: the host runs it under a plain Python.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from .. import bench_host
from ..probes import help_flags
from . import core, lowtier_suite

SUITE = "cache"
CACHE_TURNS_PATH = lowtier_suite.CACHE_TURNS_PATH
CACHE_ARMS: tuple[str, ...] = ("c0", "c0-ckpt", "c0-swafull")
BASE_ARM = "c0"
# What each arm adds to production c0's argv; `validate_inputs` checks arms.json against it.
ARM_FLAGS: dict[str, list[str]] = {
    "c0": [],
    "c0-ckpt": ["--checkpoint-min-step", "0"],
    "c0-swafull": ["--swa-full"],
}
PREFIX_TOKENS = lowtier_suite.CACHE_PREFIX_TOKENS
SESSIONS = 3
TURNS = 10
REPLY_CAP_TOKENS = 64
SEED = 20261106
# About 90 tokens per user turn; the window is for the approximate count, which a run replaces
# with the server's own count in `prompt_n`.
TURN_TOKENS_MIN = 70
TURN_TOKENS_MAX = 115
BENCH_LLM_URL = f"http://{bench_host.BENCH_LLM_HOST_BIND}:{bench_host.BENCH_LLM_PORT}"
FULL_REPROCESS_TEXT = "forcing full prompt re-processing"
_MMPROJ_RE = re.compile(r"mmproj", re.IGNORECASE)


# ---------------------------------------------------------------------------
# The served-arm check
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ServedArm:
    """The arm `mist-bench-llm` was confirmed to serve, and what the check read."""

    arm: dict
    image_ref: str
    image_id: str | None
    expected_args: list[str]
    props: dict
    snapshot: Path


def _default_props(base_url: str) -> dict:
    return bench_host.wait_for_llama_props(base_url)


def check_served(
    arm_id: str,
    *,
    results_root: Path,
    bench_run: str,
    inspect: Callable[[list[str]], dict] = bench_host.docker_inspect,
    get_props: Callable[[str], dict] | None = None,
    arms_doc: Mapping[str, object] | None = None,
) -> ServedArm:
    """Refuse unless `mist-bench-llm` serves `arm_id`, checked as bench_host's `run` checks it.

    The window's snapshot must exist (`core.bench_run_snapshot`); bench_host resolves a
    `snapshot:` image token from it, and its absence means the window was never opened.
    """
    doc = dict(arms_doc) if arms_doc is not None else bench_host.load_arms_doc()
    arm = bench_host.resolve_arm(doc, arm_id)
    snapshot = core.bench_run_snapshot(results_root, bench_run)
    image_ref = bench_host.resolve_image_ref(arm["image"], snapshot_path=snapshot)
    expected = bench_host.build_model_args(arm, doc, {})
    inspected = inspect([bench_host.BENCH_LLM_CONTAINER])
    entry = inspected.get(bench_host.BENCH_LLM_CONTAINER)
    if entry is None:
        raise bench_host.ServedArmMismatchError(
            f"{bench_host.BENCH_LLM_CONTAINER} is not running; run `bench_host serve {arm_id}` first"
        )
    try:
        props = (get_props or _default_props)(BENCH_LLM_URL)
    except OSError as exc:
        raise bench_host.ServedArmMismatchError(f"GET /props failed: {exc}") from exc
    bench_host.check_served_arm(
        entry.get("Args", []),
        expected,
        props,
        arm,
        image_token=arm["image"],
        image_ref=image_ref,
        running_config_image=(entry.get("Config") or {}).get("Image"),
        running_image_id=entry.get("Image"),
    )
    return ServedArm(arm, image_ref, entry.get("Image"), expected, props, snapshot)


# ---------------------------------------------------------------------------
# The server log
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LogScan:
    """What the server's log says about cache reuse and the vision projector."""

    full_reprocess: int
    full_reprocess_lines: tuple[str, ...]
    mmproj_lines: tuple[str, ...]

    @property
    def mmproj_loaded(self) -> bool:
        """Any log line naming mmproj; production passes no mmproj flag, so any is a finding."""
        return bool(self.mmproj_lines)

    def as_dict(self) -> dict[str, object]:
        """The scan as `log_scan.json` holds it."""
        return {
            "full_reprocess_text": FULL_REPROCESS_TEXT,
            "full_reprocess": self.full_reprocess,
            "full_reprocess_lines": list(self.full_reprocess_lines),
            "mmproj_loaded": self.mmproj_loaded,
            "mmproj_lines": list(self.mmproj_lines),
        }


def count_full_reprocess(text: str) -> int:
    return sum(1 for line in text.splitlines() if FULL_REPROCESS_TEXT in line)


def scan_log(text: str) -> LogScan:
    lines = text.splitlines()
    reprocess = tuple(line for line in lines if FULL_REPROCESS_TEXT in line)
    mmproj = tuple(line for line in lines if _MMPROJ_RE.search(line))
    return LogScan(len(reprocess), reprocess, mmproj)


def docker_log_text(name: str = bench_host.BENCH_LLM_CONTAINER) -> str:
    """Both streams of `docker logs`; llama-server writes its log to stderr."""
    stdout, stderr = bench_host.docker_logs(name)
    return stdout + stderr


# ---------------------------------------------------------------------------
# Requests and rows
# ---------------------------------------------------------------------------


def turn_body(messages: Sequence[Mapping[str, str]]) -> dict:
    """One turn's request: the cache on, greedy decoding and a fixed seed, so replies repeat."""
    return {
        "messages": [dict(m) for m in messages],
        "max_tokens": REPLY_CAP_TOKENS,
        "temperature": 0,
        "seed": SEED,
        "cache_prompt": True,
    }


def _sum(a: object, b: object) -> int | None:
    if isinstance(a, int) and isinstance(b, int):
        return a + b
    return None


def cache_row(
    *,
    arm: str,
    session: int,
    turn: int,
    turn_id: str,
    result: core.ChatStreamResult,
    prefix_tokens: int,
    full_reprocess_logged: int,
) -> dict[str, object]:
    """One `cache/<arm>/requests.jsonl` row, its server figures from the final `timings`.

    `prompt_tokens` is the whole prompt: the tokens taken from the cache plus those processed.
    """
    timings = result.timings or {}
    first = result.t_first_delta
    return {
        "arm": arm,
        "session": session,
        "turn": turn,
        "turn_id": turn_id,
        "cold": session == 1 and turn == 1,
        "prefix_tokens": prefix_tokens,
        "prompt_tokens": _sum(timings.get("cache_n"), timings.get("prompt_n")),
        "cache_n": timings.get("cache_n"),
        "prompt_n": timings.get("prompt_n"),
        "prompt_ms": timings.get("prompt_ms"),
        "ttft_ms": None if first is None else (first - result.t_request) * 1000.0,
        "predicted_per_second": timings.get("predicted_per_second"),
        "reply_tokens": timings.get("predicted_n"),
        "finish_reason": result.finish_reason,
        "full_reprocess_logged": full_reprocess_logged,
        "reply": result.content,
    }


class ChatClient(Protocol):
    """The server's `/tokenize` and streamed chat; `lowtier_suite.ServerClient` or a fake."""

    def tokenize(self, text: str) -> list[int]:
        """The server's token ids for `text`."""
        ...

    def chat(self, body: Mapping[str, object]) -> lowtier_suite.ChatTiming:
        """Stream one reply."""
        ...


@dataclass(frozen=True, slots=True)
class CacheRequest:
    """One `cache` invocation: one arm, under one bench-run window."""

    results_root: Path
    run: str
    bench_run: str
    arm: str
    sessions: int = SESSIONS
    turns: int = TURNS


def _refuse_request(request: CacheRequest) -> None:
    if request.arm not in CACHE_ARMS:
        raise core.SpeechBenchError(
            f"{request.arm!r} is not a cache arm; run 2 measures {list(CACHE_ARMS)}"
        )
    if request.sessions < 1:
        raise core.SpeechBenchError(f"--sessions must be at least 1, got {request.sessions}")
    if not 1 <= request.turns <= TURNS:
        raise core.SpeechBenchError(f"--turns must be 1 to {TURNS}, got {request.turns}")


def run_cache(
    request: CacheRequest,
    *,
    client: ChatClient | None = None,
    logs: Callable[[], str] | None = None,
    inspect: Callable[[list[str]], dict] = bench_host.docker_inspect,
    get_props: Callable[[str], dict] | None = None,
    turns: Sequence[lowtier_suite.CacheTurn] | None = None,
    arms_doc: Mapping[str, object] | None = None,
    git_sha: str | None = None,
) -> Path:
    """Run the arm's sessions and write `cache/<arm>/`; returns that directory.

    Every refusal comes before the arm's directory is claimed. Once it is, the server's log is
    saved and scanned whatever happens, so a failed request still leaves the evidence behind.
    """
    _refuse_request(request)
    core.speech_root(request.results_root, request.run)
    plan_turns = tuple(turns if turns is not None else lowtier_suite.load_cache_turns())
    served = check_served(
        request.arm,
        results_root=request.results_root,
        bench_run=request.bench_run,
        inspect=inspect,
        get_props=get_props,
        arms_doc=arms_doc,
    )
    server = client or lowtier_suite.ServerClient(BENCH_LLM_URL)
    read_logs = logs or docker_log_text
    prefix = core.build_prefix(PREFIX_TOKENS, server.tokenize)
    prefix_tokens = len(server.tokenize(prefix))

    claim = core.claim_output(
        request.results_root,
        request.run,
        SUITE,
        request.arm,
        git_sha=git_sha or core.current_git_sha(),
    )
    core.write_json_exclusive(
        claim.path / "meta.json",
        {
            **claim.stamp(),
            "arm": request.arm,
            "bench_run": request.bench_run,
            "snapshot": str(served.snapshot),
            "image_ref": served.image_ref,
            "image_id": served.image_id,
            "server_args": served.expected_args,
            "arm_flags": ARM_FLAGS[request.arm],
            "prefix_tokens": prefix_tokens,
            "prefix_target": PREFIX_TOKENS,
            "prefix_sha256": hashlib.sha256(prefix.encode("utf-8")).hexdigest(),
            "sessions": request.sessions,
            "turns": [t.id for t in plan_turns[: request.turns]],
            "reply_cap_tokens": REPLY_CAP_TOKENS,
            "temperature": 0,
            "seed": SEED,
            "started_utc": datetime.now(UTC).isoformat(),
        },
    )
    try:
        with claim.jsonl("requests.jsonl") as writer:
            for session in range(1, request.sessions + 1):
                messages = [{"role": "system", "content": prefix}]
                for index, turn in enumerate(plan_turns[: request.turns], start=1):
                    messages.append({"role": "user", "content": turn.user})
                    before = count_full_reprocess(read_logs())
                    timing = server.chat(turn_body(messages))
                    after = count_full_reprocess(read_logs())
                    writer.write(
                        cache_row(
                            arm=request.arm,
                            session=session,
                            turn=index,
                            turn_id=turn.id,
                            result=timing.result,
                            prefix_tokens=prefix_tokens,
                            full_reprocess_logged=after - before,
                        )
                    )
                    messages.append({"role": "assistant", "content": timing.result.content})
    finally:
        text = read_logs()
        core.write_text_exclusive(claim.path / "server.log", text)
        core.write_json_exclusive(
            claim.path / "log_scan.json", {**claim.stamp(), **scan_log(text).as_dict()}
        )
    return claim.path


# ---------------------------------------------------------------------------
# CLI, plan hooks and selftest
# ---------------------------------------------------------------------------


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "cache", help="run 2: prefix reuse on the served cache arm (c0, c0-ckpt, c0-swafull)"
    )
    core.add_results_args(parser, bench_run=True)
    parser.add_argument(
        "--arm", required=True, choices=CACHE_ARMS, help="the arm mist-bench-llm serves"
    )
    parser.add_argument(
        "--sessions", type=int, default=SESSIONS, help="sessions of the turns (fewer for a smoke)"
    )
    parser.add_argument(
        "--turns", type=int, default=TURNS, help="turns per session (fewer for a smoke)"
    )
    parser.set_defaults(func=cmd_cache)


def cmd_cache(args: argparse.Namespace) -> int:
    request = CacheRequest(
        results_root=args.results_root,
        run=args.run,
        bench_run=args.bench_run,
        arm=args.arm,
        sessions=args.sessions,
        turns=args.turns,
    )
    out = run_cache(request)
    scan = core.load_json(out / "log_scan.json")
    print(
        f"cache {args.arm}: rows at {out / 'requests.jsonl'}; the server log forced full "
        f"re-processing {scan['full_reprocess']} times; mmproj loaded: {scan['mmproj_loaded']}"
    )
    return 0


def container_argv(ctx: core.PlanContext) -> list[tuple[str, list[str]]]:
    """bench_host's `serve` argv for each cache arm; `cache` itself runs no container."""
    doc = bench_host.load_arms_doc()
    blocks = []
    for arm_id in CACHE_ARMS:
        arm = bench_host.resolve_arm(doc, arm_id)
        argv = bench_host.build_docker_run_args(
            arm, doc, image_ref=ctx.cuda_llama_image, models_dir=ctx.models_dir
        )
        blocks.append((f"cache serve {arm_id}", argv))
    return blocks


def _validate_arms(doc: Mapping[str, object]) -> list[str]:
    problems = []
    known = help_flags.parse_help_flags(core.HELP_CAPTURE_PATH.read_text(encoding="utf-8"))
    try:
        base = bench_host.build_model_args(bench_host.resolve_arm(doc, BASE_ARM), doc, {})
    except bench_host.ArmConfigError as exc:
        return [f"arms.json: {BASE_ARM} does not resolve: {exc}"]
    for arm_id in CACHE_ARMS:
        try:
            arm = bench_host.resolve_arm(doc, arm_id)
            args = bench_host.build_model_args(arm, doc, {})
        except bench_host.ArmConfigError as exc:
            problems.append(f"arms.json: {arm_id} does not resolve: {exc}")
            continue
        if args != [*base, *ARM_FLAGS[arm_id]]:
            problems.append(f"arms.json: {arm_id} is not {BASE_ARM} plus {ARM_FLAGS[arm_id]}")
        unknown = help_flags.unknown_flags(args, known)
        if unknown:
            problems.append(f"arms.json: {arm_id} uses flags b11151 does not list: {unknown}")
        if any("mmproj" in a for a in args):
            problems.append(f"arms.json: {arm_id} passes an mmproj flag")
    return problems


def validate_inputs(repo_root: Path) -> list[str]:
    """Problems with run 2's committed inputs: the ten turns and the three arms."""
    problems = []
    try:
        turns = lowtier_suite.load_cache_turns(repo_root / core.SPEECH_DIR_REL / "cache_turns.json")
    except core.InputError as exc:
        problems.append(str(exc))
    else:
        ids = [t.id for t in turns]
        if ids != [f"T{i:02d}" for i in range(1, TURNS + 1)]:
            problems.append(f"cache_turns.json: ids are {ids}, expected T01 to T10 in order")
        for turn in turns:
            n = len(lowtier_suite.approx_tokenize(turn.user))
            if not TURN_TOKENS_MIN <= n <= TURN_TOKENS_MAX:
                problems.append(
                    f"cache_turns.json: {turn.id} is {n} tokens by the approximate count, "
                    f"outside {TURN_TOKENS_MIN} to {TURN_TOKENS_MAX}"
                )
    try:
        problems += _validate_arms(
            bench_host.load_arms_doc(repo_root / "scripts/model_bench/arms.json")
        )
    except bench_host.ArmConfigError as exc:
        problems.append(str(exc))
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

    def __iter__(self):
        return iter(self._lines)


def selftest() -> None:
    """The fixture's final `timings`, the request body, the log scan and the served check."""
    text = (core.SPEECH_FIXTURES_DIR / "cache" / "sse_with_timings.txt").read_text(encoding="utf-8")
    ticks = iter([10.0, 10.05, 10.12, 10.2, 10.3, 10.6, 10.61])
    client = lowtier_suite.ServerClient(
        "http://fixture",
        opener=lambda request, timeout: _FixtureResponse(text),
        clock=lambda: next(ticks),
    )
    body = turn_body([{"role": "user", "content": "x"}])
    _check(body["cache_prompt"] is True and body["temperature"] == 0, f"turn body {body}")
    row = cache_row(
        arm="c0",
        session=1,
        turn=1,
        turn_id="T01",
        result=client.chat(body).result,
        prefix_tokens=PREFIX_TOKENS,
        full_reprocess_logged=0,
    )
    _check((row["cache_n"], row["prompt_n"], row["prompt_ms"]) == (6188, 97, 61.3), f"{row}")
    _check(row["reply_tokens"] == 42, "the final event's timings win over an earlier one's")

    scan = scan_log(f"a\nslot 0 | {FULL_REPROCESS_TEXT}\nload mmproj file\n")
    _check(scan.full_reprocess == 1 and scan.mmproj_loaded, f"log scan {scan}")

    problems = _validate_arms(bench_host.load_arms_doc())
    _check(not problems, f"cache arms: {problems}")

    doc = bench_host.load_arms_doc()
    expected = bench_host.build_model_args(bench_host.resolve_arm(doc, BASE_ARM), doc, {})
    image = bench_host.parse_compose_image(bench_host.DEFAULT_COMPOSE_PATH)

    def inspect(names: list[str]) -> dict:
        entry = {"Args": expected, "Image": "sha256:0", "Config": {"Image": image}}
        return {bench_host.BENCH_LLM_CONTAINER: entry}

    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        (root / "w" / "session").mkdir(parents=True)
        (root / "w" / "session" / "snapshot.json").write_text("{}", encoding="utf-8")
        props = {"model_path": "/models/unsloth/gemma-4-E4B-it-Q5_K_M.gguf"}
        check_served(
            "c0", results_root=root, bench_run="w", inspect=inspect, get_props=lambda u: props
        )
        try:
            check_served(
                "c0-ckpt",
                results_root=root,
                bench_run="w",
                inspect=inspect,
                get_props=lambda u: props,
            )
        except bench_host.ServedArmMismatchError:
            pass
        else:
            raise AssertionError("a c0 server passed the served-arm check for c0-ckpt")
