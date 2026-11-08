"""scripts/model_bench/speech/lowtier_suite.py: run 7's `lowtier` and `remote-memory`.

Every server here is a fake opener answering `/v1/models`, `/tokenize` and streamed
`/v1/chat/completions` in llama-server's shapes, with a scripted clock; docker, nvidia-smi and the
cgroup reads are fakes too. No test touches a GPU, the network or an engine package.
"""

from __future__ import annotations

import io
import json
import random
import sys
import urllib.error
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402

from scripts.model_bench.speech import core, lowtier_suite, toolcall  # noqa: E402

FIXTURE = _REPO_ROOT / "tests" / "unit" / "model_bench" / "fixtures" / "speech" / "lowtier"
TURNS = [
    {"id": f"T{i:02d}", "user": f"Invented turn {i} about the tide museum."} for i in range(1, 11)
]
VALID_CALLS = {
    c["prompt_id"]: c["calls"]
    for c in (
        json.loads(line)
        for line in (FIXTURE / "tool_calls.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if c["valid_all"]
    and c["case"].endswith(("valid", "valid, both calls", "no call", "hand-off"))
    or c["case"].endswith("on the fixed date")
}


def words(text: str) -> list[int]:
    return list(range(len(text.split())))


class FakeTime:
    """A clock that moves 1 ms per read and by the full amount on `sleep`."""

    def __init__(self, start: float = 1000.0):
        self.t = start

    def now(self) -> float:
        self.t += 0.001
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


class FakeResponse:
    def __init__(self, text: str, status: int = 200):
        self.status = status
        self._data = text.encode("utf-8")

    def read(self) -> bytes:
        return self._data

    def __iter__(self):
        return iter(io.BytesIO(self._data).readlines())

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def sse(events: list[dict]) -> str:
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"


PROMPT_IDS = {p.id: p.text for p in toolcall.load_prompts()}
TEXT_TO_ID = {text: pid for pid, text in PROMPT_IDS.items()}


class FakeServer:
    """A llama-server stand-in whose served alias moves through `phases`.

    Each phase is `(alias or None, polls, chats)`: it ends after `polls` calls to `/v1/models` or
    `chats` chat requests, whichever comes first (None means never). With no alias, every request
    is refused as a connection error.
    """

    def __init__(self, phases, *, reasoning: bool = False):
        self.phases = list(phases)
        self.reasoning = reasoning
        self.chats: list[tuple[str, dict]] = []
        self._polls = 0
        self._chats = 0

    @property
    def alias(self):
        return self.phases[0][0] if self.phases else None

    def _advance(self):
        alias, polls, chats = self.phases[0]
        ended = (polls is not None and self._polls >= polls) or (
            chats is not None and self._chats >= chats
        )
        if ended and len(self.phases) > 1:
            self.phases.pop(0)
            self._polls = self._chats = 0

    def __call__(self, request, timeout):
        url = request.full_url
        alias = self.alias
        if url.endswith("/v1/models"):
            self._polls += 1
            self._advance()
            if alias is None:
                raise urllib.error.URLError("connection refused")
            return FakeResponse(json.dumps({"object": "list", "data": [{"id": alias}]}))
        if alias is None:
            raise urllib.error.URLError("connection refused")
        body = json.loads(request.data.decode("utf-8"))
        if url.endswith("/tokenize"):
            return FakeResponse(json.dumps({"tokens": words(body["content"])}))
        assert url.endswith("/v1/chat/completions"), url
        self.chats.append((alias, body))
        self._chats += 1
        self._advance()
        return FakeResponse(self.stream(alias, body))

    def stream(self, alias: str, body: dict) -> str:
        events = [{"choices": [{"index": 0, "delta": {"role": "assistant"}}], "model": alias}]
        if self.reasoning:
            events.append({"choices": [{"index": 0, "delta": {"reasoning_content": "Hm."}}]})
        user = body["messages"][-1]["content"]
        prompt_id = TEXT_TO_ID.get(user) if "tools" in body else None
        calls = VALID_CALLS.get(prompt_id, []) if prompt_id else []
        for index, c in enumerate(calls):
            arguments = c["function"]["arguments"]
            delta = {
                "index": index,
                "id": f"call_{index}",
                "type": "function",
                "function": {"name": c["function"]["name"], "arguments": arguments},
            }
            events.append({"choices": [{"index": 0, "delta": {"tool_calls": [delta]}}]})
        if not calls:
            events.append({"choices": [{"index": 0, "delta": {"content": "Fine."}}]})
        events.append(
            {
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "timings": {
                    "cache_n": 12,
                    "prompt_n": 30,
                    "prompt_ms": 25.0,
                    "predicted_n": 8,
                    "predicted_ms": 90.0,
                    "predicted_per_second": 85.0,
                },
            }
        )
        return sse(events)


def make_client(server: FakeServer, clock: FakeTime, base_url: str = "http://bench:8081"):
    return lowtier_suite.ServerClient(base_url, opener=server, clock=clock.now)


def write_turns(path: Path, turns=TURNS, schema=1) -> Path:
    path.write_text(json.dumps({"schema": schema, "turns": turns}), encoding="utf-8")
    return path


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@pytest.fixture()
def inputs(tmp_path):
    return lowtier_suite.load_run_inputs(turns_path=write_turns(tmp_path / "cache_turns.json"))


# ---------------------------------------------------------------------------
# The low-level prompt and the fixed date
# ---------------------------------------------------------------------------


def test_the_low_level_prompt_is_1500_tokens_within_25_by_a_fake_tokenize():
    prompt = lowtier_suite.load_lowtier_prompt()

    assert abs(len(lowtier_suite.approx_tokenize(prompt)) - 1500) <= 25


def test_the_low_level_prompt_gives_the_fixed_date_and_the_two_rules():
    prompt = lowtier_suite.load_lowtier_prompt()

    assert lowtier_suite.FIXED_NOW_TEXT == "Wednesday 11 November 2026, 21:00"
    assert lowtier_suite.FIXED_NOW_TEXT in prompt
    assert "one or two short sentences" in prompt
    assert "Call hand_off for anything needing memory, a long answer or careful thought" in prompt
    assert "M.I.S.T" in prompt and "the user" in prompt


def test_fixed_now_is_wednesday_11_november_2026_at_21_00():
    now = lowtier_suite.FIXED_NOW

    assert (now.year, now.month, now.day, now.hour, now.minute) == (2026, 11, 11, 21, 0)
    assert now.strftime("%A") == "Wednesday"


def test_load_lowtier_prompt_returns_the_same_text_every_call():
    assert lowtier_suite.load_lowtier_prompt() == lowtier_suite.load_lowtier_prompt()


def test_load_lowtier_prompt_refuses_a_prompt_without_the_fixed_date(tmp_path):
    path = tmp_path / "lowtier_prefix.txt"
    path.write_text("You are a helpful voice.\n", encoding="utf-8")

    with pytest.raises(core.InputError, match="fixed date"):
        lowtier_suite.load_lowtier_prompt(path)


# ---------------------------------------------------------------------------
# Run 2's turns, loaded at run time
# ---------------------------------------------------------------------------


def test_load_cache_turns_reads_the_ten_turns_in_order(tmp_path):
    turns = lowtier_suite.load_cache_turns(write_turns(tmp_path / "t.json"))

    assert [t.id for t in turns] == [f"T{i:02d}" for i in range(1, 11)]
    assert turns[0].user == TURNS[0]["user"]


@pytest.mark.parametrize(
    "doc",
    [
        {"schema": 2, "turns": TURNS},
        {"schema": 1, "turns": TURNS[:9]},
        {"schema": 1, "turns": [*TURNS[:9], {"id": "T10"}]},
    ],
)
def test_load_cache_turns_refuses_another_shape(tmp_path, doc):
    path = tmp_path / "t.json"
    path.write_text(json.dumps(doc), encoding="utf-8")

    with pytest.raises(core.InputError):
        lowtier_suite.load_cache_turns(path)


def test_load_cache_turns_refuses_a_missing_file(tmp_path):
    with pytest.raises(core.InputError, match="cache_turns.json"):
        lowtier_suite.load_cache_turns(tmp_path / "cache_turns.json")


def test_importing_the_suite_does_not_read_cache_turns(monkeypatch):
    import importlib

    seen = []
    real = Path.read_text

    def spy(self, *args, **kwargs):
        seen.append(self.name)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", spy)
    importlib.reload(lowtier_suite)

    assert "cache_turns.json" not in seen


def _copy_inputs(repo: Path) -> Path:
    speech = repo / core.SPEECH_DIR_REL
    speech.mkdir(parents=True)
    for name in ("tools_schema.json", "tool_prompts.json", "lowtier_prefix.txt", "small_arms.json"):
        (speech / name).write_text((core.SPEECH_DIR / name).read_text(encoding="utf-8"))
    return speech


def test_validate_inputs_reports_missing_turns_without_failing_plan(tmp_path, capsys):
    _copy_inputs(tmp_path)

    problems = lowtier_suite.validate_inputs(tmp_path)

    assert problems == []
    assert "cache_turns.json" in capsys.readouterr().out


def test_validate_inputs_reports_malformed_turns(tmp_path):
    speech = _copy_inputs(tmp_path)
    write_turns(speech / "cache_turns.json", turns=TURNS[:3])

    problems = lowtier_suite.validate_inputs(tmp_path)

    assert any("cache_turns.json" in p for p in problems)


def test_validate_inputs_reports_a_prompt_off_its_token_target(tmp_path):
    speech = _copy_inputs(tmp_path)
    (speech / "lowtier_prefix.txt").write_text(
        f"It is {lowtier_suite.FIXED_NOW_TEXT}. Short.\n", encoding="utf-8"
    )

    problems = lowtier_suite.validate_inputs(tmp_path)

    assert any("1500" in p for p in problems)


def test_validate_inputs_finds_no_problem_in_the_committed_inputs():
    assert lowtier_suite.validate_inputs(_REPO_ROOT) == []


# ---------------------------------------------------------------------------
# Requests: cold lines, the 6,100 prefix, warm sessions and tool prompts
# ---------------------------------------------------------------------------


def run_plan(inputs, *, busy=False, server=None, rng_seed=5, tmp_path):
    clock = FakeTime()
    alias = "gemma4-e2b@gpu-full"
    server = server or FakeServer([(alias, None, None)])
    client = make_client(server, clock)
    out = tmp_path / "requests.jsonl"
    plan = lowtier_suite.build_row_plan(busy=busy, prompts=inputs.prompts, turns=inputs.turns)
    with core.JsonlWriter(out, stamp={"run": "t", "suite": "lowtier", "git_sha": "x"}) as writer:
        lowtier_suite.run_rows(
            client,
            plan,
            alias=alias,
            inputs=inputs,
            writer=writer,
            rng=random.Random(rng_seed),
            cut_on_error=False,
        )
    return server, read_rows(out)


def test_the_full_plan_is_warmups_cold_warm_and_tool_rows(inputs):
    plan = lowtier_suite.build_row_plan(busy=False, prompts=inputs.prompts, turns=inputs.turns)
    kinds = [s.kind for s in plan]

    assert kinds.count("warmup") == 2
    assert [(s.prefix, s.rep) for s in plan if s.kind == "cold"] == [
        ("lowtier-1500", 1),
        ("lowtier-1500", 2),
        ("lowtier-1500", 3),
        ("cache-6100", 1),
        ("cache-6100", 2),
        ("cache-6100", 3),
    ]
    assert [(s.session, s.turn) for s in plan if s.kind == "warm"] == [
        (session, turn) for session in (1, 2) for turn in range(1, 11)
    ]
    assert [(s.prompt_id, s.seed) for s in plan if s.kind == "tool"] == [
        (p, seed) for p in PROMPT_IDS for seed in (1, 2, 3)
    ]


def test_the_busy_pass_runs_warm_sessions_and_the_tool_prompts_at_seed_1(inputs):
    plan = lowtier_suite.build_row_plan(busy=True, prompts=inputs.prompts, turns=inputs.turns)

    assert {s.kind for s in plan} == {"warmup", "warm", "tool"}
    assert sum(s.kind == "warm" for s in plan) == 20
    assert {s.seed for s in plan if s.kind == "tool"} == {1}
    assert sum(s.kind == "tool" for s in plan) == 24


def test_cold_requests_open_with_a_fresh_random_line(inputs, tmp_path):
    server, rows = run_plan(inputs, tmp_path=tmp_path)

    cold = [body for _, body in server.chats if body.get("_kind") == "cold"] or [
        body for (_, body), row in zip(server.chats, rows, strict=True) if row["kind"] == "cold"
    ]
    lines = [body["messages"][0]["content"].split("\n\n", 1)[0] for body in cold]
    assert len(cold) == 6
    assert len(set(lines)) == 6
    assert all(len(line.split()) == 16 for line in lines)
    assert [r["opening_line"] for r in rows if r["kind"] == "cold"] == lines


def test_cold_requests_carry_the_low_level_prompt_and_the_core_6100_prefix(inputs, tmp_path):
    server, rows = run_plan(inputs, tmp_path=tmp_path)

    cold = [
        body for (_, body), row in zip(server.chats, rows, strict=True) if row["kind"] == "cold"
    ]
    rests = [body["messages"][0]["content"].split("\n\n", 1)[1] for body in cold]
    assert rests[:3] == [inputs.prompt] * 3
    assert rests[3:] == [core.build_prefix(6100, words)] * 3
    tokens = [r["prefix_tokens"] for r in rows if r["kind"] == "cold"]
    assert tokens[3:] == [len(words(core.build_prefix(6100, words)))] * 3


def test_warm_sessions_send_run_2s_turns_on_the_1500_prefix(inputs, tmp_path):
    server, rows = run_plan(inputs, tmp_path=tmp_path)

    warm = [
        body for (_, body), row in zip(server.chats, rows, strict=True) if row["kind"] == "warm"
    ]
    assert len(warm) == 20
    for session in (0, 1):
        for turn in range(10):
            messages = warm[session * 10 + turn]["messages"]
            assert messages[0] == {"role": "system", "content": inputs.prompt}
            users = [m["content"] for m in messages if m["role"] == "user"]
            assert users == [t["user"] for t in TURNS[: turn + 1]]
            assert [m["role"] for m in messages[1:]] == ["user", "assistant"] * turn + ["user"]
    assert all(body["cache_prompt"] is True and body["max_tokens"] == 128 for body in warm)


def test_tool_rows_send_the_tool_set_and_score_the_calls(inputs, tmp_path):
    server, rows = run_plan(inputs, tmp_path=tmp_path)

    tool_bodies = [b for (_, b), r in zip(server.chats, rows, strict=True) if r["kind"] == "tool"]
    tool_rows = [r for r in rows if r["kind"] == "tool"]
    assert all(b["tools"] == inputs.tools for b in tool_bodies)
    assert all(b["messages"][0]["content"] == inputs.prompt for b in tool_bodies)
    assert [b["seed"] for b in tool_bodies[:3]] == [1, 2, 3]
    assert all(r["tool_valid_first"] and r["tool_valid_all"] for r in tool_rows)
    assert {r["prompt_id"] for r in tool_rows} == set(PROMPT_IDS)


def test_a_reply_row_records_client_and_server_first_token_times_and_tool_call_time(inputs):
    lines = (FIXTURE / "sse_tool_call.txt").read_text(encoding="utf-8")
    ticks = iter([100.0, 100.05, 100.20, 100.25, 100.30, 100.42, 100.50, 100.51])
    client = lowtier_suite.ServerClient(
        "http://bench:8081",
        opener=lambda req, timeout: FakeResponse(lines),
        clock=lambda: next(ticks),
    )
    spec = lowtier_suite.RowSpec(kind="tool", prompt_id="C06", seed=1)

    timing = client.chat({"messages": []})
    score = toolcall.score_calls(inputs.prompts["C06"], timing.result.tool_calls, inputs.tools)
    row = lowtier_suite.request_row(spec, timing, prefix_tokens=1500, score=score)

    assert row["ttft_ms"] == pytest.approx(200.0)
    assert row["ttft_server_ms"] == pytest.approx(42.5)
    assert row["tool_call_ms"] == pytest.approx(420.0)
    assert (row["cache_n"], row["prompt_n"], row["prompt_ms"]) == (1488, 31, 42.5)
    assert row["predicted_per_second"] == pytest.approx(89.53)
    assert (row["tool_valid_first"], row["tool_valid_all"]) == (True, True)
    assert [c["function"]["name"] for c in row["raw_tool_calls"]] == ["cancel_alarm", "set_alarm"]
    assert row["reasoning_flagged"] is False
    assert row["cut"] is False


def test_a_reply_carrying_reasoning_content_is_flagged(inputs, tmp_path):
    alias = "gemma4-e2b@gpu-full"
    server = FakeServer([(alias, None, None)], reasoning=True)

    _, rows = run_plan(inputs, server=server, tmp_path=tmp_path)

    assert all(r["reasoning_flagged"] for r in rows)
    assert all(r["reasoning_chars"] == 3 for r in rows)


# ---------------------------------------------------------------------------
# The slice search
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("total", "slice_mib", "expected"),
    [(40, 4096, 35), (40, 2048, 15), (30, 4096, 30), (40, 400, None), (40, 500, 0)],
)
def test_bisect_layers_finds_the_largest_count_at_or_under_the_slice(total, slice_mib, expected):
    calls = []

    def fits(n: int) -> bool:
        calls.append(n)
        return 500 + 100 * n <= slice_mib

    chosen = lowtier_suite.bisect_layers(fits, total)

    assert chosen == expected
    assert calls[0] == total
    assert len(calls) <= 8


def footprint_probe(per_layer=100.0, base=500.0, total=40):
    probes = []

    def probe(ngl: int) -> lowtier_suite.ProbeResult:
        probes.append(ngl)
        layers = min(ngl, total)
        return lowtier_suite.ProbeResult(
            ngl=ngl, footprint_mib=base + per_layer * layers, total_layers=total, error=None
        )

    return probe, probes


def test_slice_search_records_the_chosen_layer_count(tmp_path):
    probe, probes = footprint_probe()

    result = lowtier_suite.run_slice_search(
        tmp_path, "mh1", "gemma4-e2b", "gpu-4g", probe=probe, git_sha="abc"
    )

    record = json.loads(result.path.read_text(encoding="utf-8"))
    assert result.chosen_ngl == 35
    assert record["chosen_ngl"] == 35
    assert record["slice_mib"] == 4096
    assert record["under_at_full"] is False
    assert [p["ngl"] for p in record["probes"]] == probes
    assert probes[0] == lowtier_suite.FULL_NGL
    assert (
        result.path
        == tmp_path
        / "mh1"
        / "speech"
        / "lowtier"
        / "slice"
        / "gemma4-e2b"
        / "gpu-4g"
        / "slice.json"
    )


def test_slice_search_reports_a_model_under_the_slice_at_full_offload_once(tmp_path):
    probe, probes = footprint_probe(per_layer=10.0)

    result = lowtier_suite.run_slice_search(
        tmp_path, "mh1", "minicpm5-2b", "gpu-4g", probe=probe, git_sha="abc"
    )

    record = json.loads(result.path.read_text(encoding="utf-8"))
    assert probes == [lowtier_suite.FULL_NGL]
    assert record["under_at_full"] is True
    assert record["chosen_ngl"] == 40


def test_slice_search_refuses_a_placement_without_a_slice(tmp_path):
    probe, probes = footprint_probe()

    with pytest.raises(core.PlacementError):
        lowtier_suite.run_slice_search(tmp_path, "mh1", "gemma4-e2b", "gpu-full", probe=probe)
    assert probes == []


def test_the_serve_probe_measures_by_difference_in_a_fresh_directory_per_count(tmp_path):
    served = []
    gpu = iter([1800.0, 4300.0, 1800.0, 2900.0])

    def serve(model_id, placement, *, out_dir, ngl, **kwargs):
        served.append((out_dir, ngl))
        out_dir.mkdir(parents=True)

    def unserve(out_dir, **kwargs):
        return 0.5

    probe = lowtier_suite.make_serve_probe(
        "gemma4-e2b",
        "gpu-4g",
        speech=tmp_path / "mh1" / "speech",
        pins_path=tmp_path / "pins.json",
        models_dirs={"small": tmp_path},
        serve=serve,
        unserve=unserve,
        gpu_used_mib=lambda: next(gpu),
        logs=lambda name: "load_tensors: offloaded 36/36 layers to GPU\n",
    )

    full = probe(999)
    half = probe(18)

    assert (full.footprint_mib, full.total_layers) == (2500.0, 36)
    assert half.footprint_mib == 1100.0
    dirs = [d for d, _ in served]
    assert len(set(dirs)) == 2
    assert [d.name for d in dirs] == ["gpu-4g-slice999", "gpu-4g-slice18"]
    assert [n for _, n in served] == [999, 18]


def test_the_serve_probe_counts_a_load_failure_as_over_the_slice(tmp_path):
    def serve(model_id, placement, *, out_dir, ngl, **kwargs):
        raise core.ServeFailedError("out of memory")

    probe = lowtier_suite.make_serve_probe(
        "gemma4-e2b",
        "gpu-2g",
        speech=tmp_path,
        pins_path=tmp_path / "pins.json",
        models_dirs={},
        serve=serve,
        unserve=lambda out_dir, **kw: 0.0,
        gpu_used_mib=lambda: 1800.0,
        logs=lambda name: "",
        container_state=lambda name: "absent",
    )

    result = probe(20)

    assert result.footprint_mib is None
    assert "out of memory" in result.error


# ---------------------------------------------------------------------------
# E4B's reload
# ---------------------------------------------------------------------------


def test_e4b_reload_is_timed_three_times(tmp_path):
    clock = FakeTime()
    calls = []

    def bench_main(argv):
        calls.append(argv[0])
        clock.sleep(5.0 if argv[0] == "serve" else 1.0)
        return 0

    def chat(base_url, body, *, clock):
        t0 = clock()
        return core.ChatStreamResult(
            content="Hi",
            reasoning_content="",
            tool_calls=(),
            t_request=t0,
            t_first_content=t0 + 0.4,
            t_first_tool_call=None,
            t_end=t0 + 0.5,
            finish_reason="stop",
            timings=None,
        )

    path = lowtier_suite.run_e4b_reload(
        tmp_path,
        "mh1",
        "mh1-h5",
        models_dir=tmp_path / "models",
        bench_main=bench_main,
        chat=chat,
        clock=clock.now,
        git_sha="abc",
    )

    rows = read_rows(path)
    assert path == tmp_path / "mh1" / "speech" / "lowtier" / "e4b_reload.jsonl"
    assert calls == ["unserve", "serve"] * 3
    assert [r["rep"] for r in rows] == [1, 2, 3]
    assert all(r["reload_s"] == pytest.approx(5.4, abs=0.01) for r in rows)
    assert all(r["unserve_s"] == pytest.approx(1.0, abs=0.01) for r in rows)


def test_e4b_reload_stops_when_bench_host_fails(tmp_path):
    with pytest.raises(core.SpeechBenchError, match="serve"):
        lowtier_suite.run_e4b_reload(
            tmp_path,
            "mh1",
            "mh1-h5",
            models_dir=tmp_path,
            bench_main=lambda argv: 1 if argv[0] == "serve" else 0,
            chat=lambda *a, **k: None,
            clock=FakeTime().now,
            git_sha="abc",
        )


# ---------------------------------------------------------------------------
# --remote-url: polling, cut rows and unexpected aliases
# ---------------------------------------------------------------------------

A = "gemma4-e2b@gtx1070-shared-idle"
B = "gemma4-e2b@gtx1070-shared-busy"


def tiny_plan(alias):
    return [
        lowtier_suite.RowSpec(kind="warmup", rep=1),
        lowtier_suite.RowSpec(kind="warmup", rep=2),
        lowtier_suite.RowSpec(kind="tool", prompt_id="C01", seed=1),
    ]


def run_remote(tmp_path, server, expect, inputs, wait_s=600.0):
    clock = FakeTime()
    request = lowtier_suite.RemoteRequest(
        results_root=tmp_path,
        run="mh1",
        base_url="http://mist-extraction-gtx1070:8090",
        expect=tuple(expect),
        poll_interval_s=10.0,
        wait_s=wait_s,
    )
    return lowtier_suite.run_remote(
        request,
        client=make_client(server, clock, request.base_url),
        inputs=inputs,
        rows_for=tiny_plan,
        clock=clock.now,
        sleep=clock.sleep,
        rng=random.Random(1),
        git_sha="abc",
    )


def lowtier_dir(tmp_path, alias):
    model, placement = alias.split("@")
    return tmp_path / "mh1" / "speech" / "lowtier" / model / placement


def test_remote_polls_models_and_runs_each_expected_alias_when_it_appears(tmp_path, inputs):
    server = FakeServer([(None, 3, None), (A, 6, None), (None, 2, None), (B, None, None)])

    status = run_remote(tmp_path, server, [A, B], inputs)

    for alias in (A, B):
        rows = read_rows(lowtier_dir(tmp_path, alias) / "requests.jsonl")
        assert [r["kind"] for r in rows] == ["warmup", "warmup", "tool"]
        assert {r["alias"] for r in rows} == {alias}
        assert not any(r["cut"] for r in rows)
        memory = json.loads((lowtier_dir(tmp_path, alias) / "memory.json").read_text())
        assert memory["memory_method"] == "host-1070 summary, read by remote-memory"
    assert [a for a, _ in server.chats] == [A] * 3 + [B] * 3
    assert status.done == (A, B)
    assert status.not_served == ()


def test_remote_records_rows_a_window_cut_short_as_cut(tmp_path, inputs):
    server = FakeServer([(A, None, 1), (None, None, None)])

    status = run_remote(tmp_path, server, [A, B], inputs, wait_s=120.0)

    rows = read_rows(lowtier_dir(tmp_path, A) / "requests.jsonl")
    assert [r["cut"] for r in rows] == [False, True, True]
    assert rows[1]["ttft_ms"] is None
    assert [r["kind"] for r in rows] == ["warmup", "warmup", "tool"]
    assert status.cut == (A,)
    assert status.not_served == (B,)
    assert not lowtier_dir(tmp_path, B).exists()


def test_remote_refuses_an_alias_it_was_not_given(tmp_path, inputs):
    stranger = "qwen35-4b@gtx1070"
    server = FakeServer([(stranger, None, None)])

    with pytest.raises(lowtier_suite.AliasError, match=stranger):
        run_remote(tmp_path, server, [A], inputs)

    assert server.chats == []
    assert not lowtier_dir(tmp_path, stranger).exists()


def test_remote_refuses_an_expected_alias_outside_the_host_1070_placements(tmp_path, inputs):
    server = FakeServer([(A, None, None)])

    with pytest.raises(core.PlacementError):
        run_remote(tmp_path, server, ["gemma4-e2b@gpu-full"], inputs)


def test_the_default_expected_aliases_follow_the_h6_schedule():
    aliases = lowtier_suite.default_remote_aliases()

    assert aliases[:4] == [
        "gemma4-e2b@gtx1070-shared-idle",
        "gemma4-e2b@gtx1070-shared-busy",
        "gemma4-e2b@host1070-cpu",
        "minicpm5-2b@gtx1070-shared-idle",
    ]
    assert "e4b-ref@gtx1070-shared-busy" not in aliases
    assert aliases[-4:] == [
        f"{m}@gtx1070" for m in ("gemma4-e2b", "minicpm5-2b", "qwen35-4b", "e4b-ref")
    ]


# ---------------------------------------------------------------------------
# Local placements: alias check, memory by the placement's method, unserve
# ---------------------------------------------------------------------------


class FakeSampler:
    def __init__(self, path: Path, values):
        self.path = path
        self.values = values

    def start(self):
        lines = ["t_unix,memory_used_mib,memory_total_mib"]
        lines += [f"{i},{v},12282" for i, v in enumerate(self.values)]
        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def stop(self):
        pass


class FakeProbes:
    def __init__(
        self, *, gpu=(), vram=(1800, 4300, 4350), logs="", peak=(2100.0, "cgroup memory.peak")
    ):
        self._gpu = iter(gpu)
        self.vram = vram
        self._logs = logs
        self._peak = peak
        self.peak_reads = []

    def logs(self, name):
        return self._logs

    def container_peak_mib(self, name):
        self.peak_reads.append(name)
        return self._peak

    def gpu_used_mib(self):
        return next(self._gpu)

    def sampler(self, path):
        return FakeSampler(path, self.vram)


def prepare_local(tmp_path, model="gemma4-e2b", placement="gpu-full", desktop=1795):
    speech = tmp_path / "mh1" / "speech"
    serve_dir = core.small_serve_dir(speech, model, placement)
    serve_dir.mkdir(parents=True)
    (serve_dir / "serve.json").write_text(
        json.dumps(
            {"load_s": 12.5, "ngl": "999", "image": "cuda-image", "alias": f"{model}@{placement}"}
        ),
        encoding="utf-8",
    )
    session = tmp_path / "mh1-h5" / "session"
    session.mkdir(parents=True)
    (session / "vram_steps.json").write_text(
        json.dumps(
            {"steps": [{"label": "desktop", "median_mib": desktop, "max_mib": desktop + 5}]}
        ),
        encoding="utf-8",
    )
    return serve_dir


def run_local(
    tmp_path,
    inputs,
    probes,
    *,
    model="gemma4-e2b",
    placement="gpu-full",
    served=None,
    unserve=False,
    unserve_fn=None,
):
    clock = FakeTime()
    alias = served or f"{model}@{placement}"
    server = FakeServer([(alias, None, None)])
    request = lowtier_suite.LocalRequest(
        results_root=tmp_path,
        run="mh1",
        bench_run="mh1-h5",
        model=model,
        placement=placement,
        unserve=unserve,
    )
    return lowtier_suite.run_local(
        request,
        client=make_client(server, clock),
        inputs=inputs,
        probes=probes,
        clock=clock.now,
        sleep=clock.sleep,
        rng=random.Random(2),
        rows_for=tiny_plan,
        unserve=unserve_fn or (lambda out_dir, **kw: 1.25),
        git_sha="abc",
    )


def test_local_gpu_placement_reads_gpu_memory_by_difference_and_host_memory(tmp_path, inputs):
    prepare_local(tmp_path)
    probes = FakeProbes(logs="load_backend: loaded CPU backend from /app/libggml-cpu-zen4.so\n")

    out = run_local(tmp_path, inputs, probes)

    memory = json.loads((out / "memory.json").read_text(encoding="utf-8"))
    assert out == tmp_path / "mh1" / "speech" / "lowtier" / "gemma4-e2b" / "gpu-full"
    assert (
        memory["memory_method"]
        == "gpu by difference from the desktop step, and container memory.peak"
    )
    assert (memory["gpu_baseline_mib"], memory["gpu_peak_mib"]) == (1795, 4350)
    assert memory["gpu_by_difference_mib"] == 2555
    assert (memory["host_peak_mib"], memory["host_peak_method"]) == (2100.0, "cgroup memory.peak")
    assert memory["load_s"] == 12.5
    assert memory["ngl"] == "999"
    assert memory["cpu_variant"] == "zen4"
    assert (out / "vram.csv").is_file()
    assert (out / "server.log").is_file()
    assert len(read_rows(out / "requests.jsonl")) == 3


def test_local_cpu_placement_reads_host_memory_only(tmp_path, inputs):
    prepare_local(tmp_path, placement="cpu")
    probes = FakeProbes(logs="load_backend: loaded CPU backend from /app/libggml-cpu-zen4.so\n")

    out = run_local(tmp_path, inputs, probes, placement="cpu")

    memory = json.loads((out / "memory.json").read_text(encoding="utf-8"))
    assert memory["memory_method"] == "container memory.peak"
    assert memory["gpu_by_difference_mib"] is None
    assert memory["host_peak_mib"] == 2100.0
    assert memory["cpu_variant"] == "zen4"
    assert not (out / "vram.csv").exists()


def test_local_records_the_slice_search_layer_count(tmp_path, inputs):
    prepare_local(tmp_path, placement="gpu-4g")
    probe, _ = footprint_probe()
    lowtier_suite.run_slice_search(
        tmp_path, "mh1", "gemma4-e2b", "gpu-4g", probe=probe, git_sha="abc"
    )

    out = run_local(tmp_path, inputs, FakeProbes(), placement="gpu-4g")

    memory = json.loads((out / "memory.json").read_text(encoding="utf-8"))
    assert memory["slice_chosen_ngl"] == 35


def test_local_refuses_a_served_alias_that_differs_before_writing(tmp_path, inputs):
    prepare_local(tmp_path)

    with pytest.raises(lowtier_suite.AliasError, match="minicpm5-2b@gpu-full"):
        run_local(tmp_path, inputs, FakeProbes(), served="minicpm5-2b@gpu-full")

    assert not (tmp_path / "mh1" / "speech" / "lowtier").exists()


def test_local_refuses_without_a_serve_record(tmp_path, inputs):
    with pytest.raises(core.SpeechBenchError, match="small-serve"):
        run_local(tmp_path, inputs, FakeProbes())


def test_local_refuses_a_host_1070_placement(tmp_path, inputs):
    with pytest.raises(core.PlacementError):
        run_local(tmp_path, inputs, FakeProbes(), placement="gtx1070")


def test_local_unserve_times_the_unload_until_the_memory_returns(tmp_path, inputs):
    prepare_local(tmp_path)
    unserved = []
    probes = FakeProbes(gpu=[4350.0, 4300.0, 2600.0, 1900.0])

    def unserve(out_dir, **kwargs):
        unserved.append(out_dir)
        return 1.25

    out = run_local(tmp_path, inputs, probes, unserve=True, unserve_fn=unserve)

    memory = json.loads((out / "memory.json").read_text(encoding="utf-8"))
    assert unserved == [core.small_serve_dir(tmp_path / "mh1" / "speech", "gemma4-e2b", "gpu-full")]
    assert memory["gpu_after_unload_mib"] == 1900.0
    assert memory["memory_returned_mib"] == 2450.0
    assert memory["unload_s"] > 0
    assert memory["stop_s"] == 1.25


# ---------------------------------------------------------------------------
# remote-memory
# ---------------------------------------------------------------------------


def test_remote_memory_reads_the_summary_fixture_per_model_and_placement(tmp_path):
    host = tmp_path / "mh1" / "speech" / "host1070"
    host.mkdir(parents=True)
    (host / "summary.json").write_text(
        (FIXTURE / "host1070_summary.json").read_text(encoding="utf-8"), encoding="utf-8"
    )

    path = lowtier_suite.run_remote_memory(tmp_path, "mh1", git_sha="abc")

    rows = read_rows(path)
    windows = {(r["model"], r["placement"]): r for r in rows if r["kind"] == "window"}
    idle = windows[("gemma4-e2b", "gtx1070-shared-idle")]
    assert idle["ngl"] == 9
    assert idle["gpu_load_mib"] == 352
    assert idle["gpu_free_at_peak_mib"] == 537
    assert idle["host_used_mib"] == 2391
    assert idle["memory_returned_mib"] == 350
    assert idle["load_s"] == pytest.approx(14.212)
    assert windows[("gemma4-e2b", "gtx1070")]["fallback"] == "fa-off-f16"
    assert windows[("minicpm5-2b", "host1070-cpu")]["ngl"] is None
    assert windows[("minicpm5-2b", "gtx1070")]["status"] == "failed"
    skips = {(r["model"], r["placement"]): r["reason"] for r in rows if r["kind"] == "skip"}
    assert skips[("e4b-ref", "gtx1070-shared")] == "does not fit beside extraction in host memory"
    assert skips[("minicpm5-2b", "gtx1070-shared")] == "does not fit beside extraction"
    layers = {r["model"]: r["chosen_ngl"] for r in rows if r["kind"] == "layers"}
    assert layers == {"gemma4-e2b": 9, "minicpm5-2b": None}
    restore = [r for r in rows if r["kind"] == "restore"]
    assert restore[0]["inspect_diff_empty"] is True
    assert all(r["run"] == "mh1" and r["suite"] == "host1070" for r in rows)


def test_remote_memory_refuses_to_overwrite_its_output(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text((FIXTURE / "host1070_summary.json").read_text(encoding="utf-8"))
    lowtier_suite.run_remote_memory(tmp_path, "mh1", summary_path=summary, git_sha="abc")

    with pytest.raises(core.OutputExistsError):
        lowtier_suite.run_remote_memory(tmp_path, "mh1", summary_path=summary, git_sha="abc")


def test_remote_memory_refuses_a_summary_of_another_schema(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"schema": 9}), encoding="utf-8")

    with pytest.raises(core.InputError):
        lowtier_suite.run_remote_memory(tmp_path, "mh1", summary_path=summary, git_sha="abc")


# ---------------------------------------------------------------------------
# The registry contract
# ---------------------------------------------------------------------------


def test_the_suite_registers_lowtier_and_remote_memory():
    from scripts.model_bench import speech_bench

    parsers = speech_bench.subcommand_parsers(speech_bench.build_parser())

    lowtier = speech_bench.all_option_strings(parsers["lowtier"])
    assert {"--model", "--placement", "--remote-url", "--expect", "--slice-search"} <= lowtier
    assert {"--e4b-reload", "--unserve", "--bench-run"} <= lowtier
    assert "--summary" in speech_bench.all_option_strings(parsers["remote-memory"])


def test_lowtier_refuses_a_mode_mix_before_writing(tmp_path, capsys):
    from scripts.model_bench import speech_bench

    code = speech_bench.main(
        [
            "lowtier",
            "--results-root",
            str(tmp_path),
            "--run",
            "mh1",
            "--bench-run",
            "mh1-h5",
            "--remote-url",
            "http://mist-extraction-gtx1070:8090",
            "--model",
            "gemma4-e2b",
        ]
    )

    assert code != 0
    assert "[FAIL]" in capsys.readouterr().out
    assert not (tmp_path / "mh1").exists()


def test_selftest_passes():
    lowtier_suite.selftest()
