"""scripts/model_bench/speech/cache_suite.py: run 2, the prompt cache under each cache arm.

The server is a fake opener answering `/tokenize` and streamed `/v1/chat/completions` in
llama-server's shapes, with a scripted clock; `docker inspect`, `/props` and `docker logs` are
fakes too. No test touches a GPU, the network, Docker or an engine package.
"""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402

from scripts.model_bench import bench_host, speech_bench  # noqa: E402
from scripts.model_bench.probes import help_flags  # noqa: E402
from scripts.model_bench.speech import cache_suite, core, lowtier_suite  # noqa: E402

FIXTURE = _REPO_ROOT / "tests" / "unit" / "model_bench" / "fixtures" / "speech" / "cache"
HELP_TEXT = core.HELP_CAPTURE_PATH.read_text(encoding="utf-8")
BASE_COMMIT = "cae41f7"
ARMS_DOC = bench_host.load_arms_doc()
COMPOSE_IMAGE = bench_host.parse_compose_image(bench_host.DEFAULT_COMPOSE_PATH)
IMAGE_ID = "sha256:" + "ef" * 32
GGUF_PATH = "/models/unsloth/gemma-4-E4B-it-Q5_K_M.gguf"
TURNS = tuple(
    lowtier_suite.CacheTurn(f"T{i:02d}", f"Invented turn {i} on raised beds.") for i in range(1, 11)
)


def words(text: str) -> list[int]:
    return list(range(len(text.split())))


class FakeTime:
    """A clock that moves 1 ms per read."""

    def __init__(self, start: float = 1000.0):
        self.t = start

    def __call__(self) -> float:
        self.t += 0.001
        return self.t


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


def sse_reply(n: int) -> str:
    events = [
        {"choices": [{"index": 0, "delta": {"role": "assistant"}}]},
        {"choices": [{"index": 0, "delta": {"content": f"Reply {n} about the beds."}}]},
        {
            "choices": [{"index": 0, "finish_reason": "stop", "delta": {}}],
            "timings": {
                "cache_n": 6000 + n,
                "prompt_n": 90 + n,
                "prompt_ms": 50.0 + n,
                "predicted_n": 30,
                "predicted_ms": 375.0,
                "predicted_per_second": 80.0,
            },
        },
    ]
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"


class FakeServer:
    """`/tokenize` counts words; each chat request gets a numbered reply and is recorded."""

    def __init__(self, *, fail_at: int | None = None, on_chat=None):
        self.bodies: list[dict] = []
        self.fail_at = fail_at
        self.on_chat = on_chat

    def __call__(self, request, timeout):
        body = json.loads(request.data.decode("utf-8"))
        if request.full_url.endswith("/tokenize"):
            return FakeResponse(json.dumps({"tokens": words(body["content"])}))
        self.bodies.append(body)
        n = len(self.bodies)
        if self.on_chat is not None:
            self.on_chat(n)
        if self.fail_at == n:
            raise OSError("connection reset")
        return FakeResponse(sse_reply(n))


def client_for(server: FakeServer) -> lowtier_suite.ServerClient:
    return lowtier_suite.ServerClient("http://fake:8080", opener=server, clock=FakeTime())


def expected_args(arm_id: str) -> list[str]:
    return bench_host.build_model_args(bench_host.resolve_arm(ARMS_DOC, arm_id), ARMS_DOC, {})


def fake_inspect(arm_id: str, *, image: str = COMPOSE_IMAGE):
    def inspect(names):
        assert names == [bench_host.BENCH_LLM_CONTAINER]
        return {
            bench_host.BENCH_LLM_CONTAINER: {
                "Args": expected_args(arm_id),
                "Image": IMAGE_ID,
                "Config": {"Image": image},
            }
        }

    return inspect


def fake_props(base_url: str) -> dict:
    return {"model_path": GGUF_PATH}


def make_window(root: Path, bench_run: str = "mh1-h4") -> Path:
    session = root / bench_run / "session"
    session.mkdir(parents=True)
    (session / "snapshot.json").write_text("{}", encoding="utf-8")
    return session


class Logs:
    """`docker logs` of the bench server: a line per request, some forcing re-processing."""

    def __init__(self):
        self.lines = ["main: model loaded", "srv  update_slots: all slots are idle"]

    def add(self, n: int) -> None:
        self.lines.append(f"slot launch_slot_: id 0 | task {n} | processing task")
        if n in (1, 11):
            self.lines.append(
                "slot update_slots: id 0 | task 1 | forcing full prompt re-processing due to "
                "lack of cache data"
            )

    def __call__(self) -> str:
        return "\n".join(self.lines) + "\n"


def run(tmp_path: Path, arm: str = "c0-ckpt", **kwargs):
    root = tmp_path / "results"
    make_window(root)
    logs = kwargs.pop("logs", None) or Logs()
    server = kwargs.pop("server", None) or FakeServer(on_chat=logs.add)
    request = cache_suite.CacheRequest(
        results_root=root,
        run="mh1",
        bench_run="mh1-h4",
        arm=arm,
        sessions=kwargs.pop("sessions", cache_suite.SESSIONS),
        turns=kwargs.pop("turns", len(TURNS)),
    )
    out = cache_suite.run_cache(
        request,
        client=client_for(server),
        logs=logs,
        inspect=kwargs.pop("inspect", fake_inspect(arm)),
        get_props=kwargs.pop("get_props", fake_props),
        turns=kwargs.pop("turns_doc", TURNS),
        git_sha="abc123",
        **kwargs,
    )
    return out, server, logs


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# ---------------------------------------------------------------------------
# cache_turns.json
# ---------------------------------------------------------------------------


def test_cache_turns_json_loads_through_the_lowtier_loader_as_ten_turns():
    turns = lowtier_suite.load_cache_turns(cache_suite.CACHE_TURNS_PATH)

    assert [t.id for t in turns] == [f"T{i:02d}" for i in range(1, 11)]
    assert all(t.user.strip() for t in turns)


def test_each_cache_turn_is_about_ninety_tokens_by_the_approximate_count():
    for turn in lowtier_suite.load_cache_turns(cache_suite.CACHE_TURNS_PATH):
        n = len(lowtier_suite.approx_tokenize(turn.user))
        assert cache_suite.TURN_TOKENS_MIN <= n <= cache_suite.TURN_TOKENS_MAX, (turn.id, n)


def test_cache_turns_json_has_exactly_the_schema_the_loader_reads():
    doc = json.loads(cache_suite.CACHE_TURNS_PATH.read_text(encoding="utf-8"))

    assert set(doc) == {"schema", "turns"}
    assert doc["schema"] == 1
    assert all(set(turn) == {"id", "user"} for turn in doc["turns"])


# ---------------------------------------------------------------------------
# arms.json: the two cache arms, additive only
# ---------------------------------------------------------------------------


def _base_arms_doc() -> dict:
    proc = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "show", f"{BASE_COMMIT}:scripts/model_bench/arms.json"],
        capture_output=True,
        text=True,
        shell=False,
    )
    if proc.returncode != 0:
        pytest.skip(f"cannot read arms.json at {BASE_COMMIT}: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def test_arms_json_gains_only_c0_ckpt_and_c0_swafull():
    base = _base_arms_doc()

    assert set(ARMS_DOC["arms"]) - set(base["arms"]) == {"c0-ckpt", "c0-swafull"}
    for arm_id, raw in base["arms"].items():
        assert ARMS_DOC["arms"][arm_id] == raw, arm_id
    for key in ("common_args", "sampling", "thinking_args"):
        assert ARMS_DOC[key] == base[key]


@pytest.mark.parametrize(
    ("arm_id", "flags"),
    [("c0-ckpt", ["--checkpoint-min-step", "0"]), ("c0-swafull", ["--swa-full"])],
)
def test_each_cache_arm_is_production_c0_plus_its_one_flag(arm_id, flags):
    assert expected_args(arm_id) == [*expected_args("c0"), *flags]
    assert cache_suite.ARM_FLAGS[arm_id] == flags
    arm = bench_host.resolve_arm(ARMS_DOC, arm_id)
    assert arm["image"] == "compose:mist-llm"
    assert arm["gguf"] == "unsloth/gemma-4-E4B-it-Q5_K_M.gguf"


def test_production_c0_loads_no_mmproj():
    assert not any("mmproj" in arg for arg in expected_args("c0"))


@pytest.mark.parametrize("arm_id", cache_suite.CACHE_ARMS)
def test_cache_arm_flags_are_in_the_b11151_help_capture(arm_id):
    known = help_flags.parse_help_flags(HELP_TEXT)

    assert help_flags.unknown_flags(expected_args(arm_id), known) == []


# ---------------------------------------------------------------------------
# The served-arm check, as bench_host's `run` does it
# ---------------------------------------------------------------------------


def test_check_served_passes_for_the_named_arm(tmp_path):
    make_window(tmp_path)

    served = cache_suite.check_served(
        "c0-swafull",
        results_root=tmp_path,
        bench_run="mh1-h4",
        inspect=fake_inspect("c0-swafull"),
        get_props=fake_props,
    )

    assert served.expected_args == expected_args("c0-swafull")
    assert served.image_ref == COMPOSE_IMAGE
    assert served.snapshot == (tmp_path / "mh1-h4" / "session" / "snapshot.json").resolve()


def test_cache_refuses_when_a_different_arm_is_served(tmp_path):
    with pytest.raises(bench_host.ServedArmMismatchError, match="Args do not match"):
        run(tmp_path, arm="c0-ckpt", inspect=fake_inspect("c0"))

    assert not (tmp_path / "results" / "mh1").exists()


def test_cache_refuses_when_the_served_image_differs(tmp_path):
    with pytest.raises(bench_host.ServedArmMismatchError, match="Config.Image"):
        run(tmp_path, arm="c0", inspect=fake_inspect("c0", image="other:image"))

    assert not (tmp_path / "results" / "mh1").exists()


def test_cache_refuses_when_props_names_another_model(tmp_path):
    with pytest.raises(bench_host.ServedArmMismatchError, match="model path"):
        run(tmp_path, get_props=lambda url: {"model_path": "/models/other.gguf"})


def test_cache_refuses_when_mist_bench_llm_is_not_running(tmp_path):
    with pytest.raises(bench_host.ServedArmMismatchError, match="not running"):
        run(tmp_path, inspect=lambda names: {})

    assert not (tmp_path / "results" / "mh1").exists()


def test_cache_refuses_a_bench_run_without_a_snapshot(tmp_path):
    request = cache_suite.CacheRequest(
        results_root=tmp_path, run="mh1", bench_run="mh1-h9", arm="c0"
    )

    with pytest.raises(core.BenchRunError, match="snapshot"):
        cache_suite.run_cache(
            request,
            client=client_for(FakeServer()),
            logs=Logs(),
            inspect=fake_inspect("c0"),
            get_props=fake_props,
            turns=TURNS,
        )

    assert not (tmp_path / "mh1").exists()


def test_cache_refuses_an_arm_that_is_not_a_cache_arm(tmp_path):
    with pytest.raises(core.SpeechBenchError, match="cache arm"):
        run(tmp_path, arm="c0-ctx64k", inspect=fake_inspect("c0-ctx64k"))


def test_cache_refuses_to_overwrite_an_arm_it_already_ran(tmp_path):
    run(tmp_path, arm="c0", sessions=1, turns=1)

    with pytest.raises(core.OutputExistsError):
        cache_suite.run_cache(
            cache_suite.CacheRequest(
                results_root=tmp_path / "results", run="mh1", bench_run="mh1-h4", arm="c0"
            ),
            client=client_for(FakeServer()),
            logs=Logs(),
            inspect=fake_inspect("c0"),
            get_props=fake_props,
            turns=TURNS,
        )


# ---------------------------------------------------------------------------
# Requests and rows
# ---------------------------------------------------------------------------


def test_the_prefix_is_t1s_6100_token_build_and_every_session_starts_from_it(tmp_path):
    out, server, _ = run(tmp_path)

    prefix = core.build_prefix(cache_suite.PREFIX_TOKENS, words)
    assert cache_suite.PREFIX_TOKENS == 6100
    firsts = [b["messages"][0] for b in server.bodies]
    assert all(m == {"role": "system", "content": prefix} for m in firsts)
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert abs(meta["prefix_tokens"] - 6100) <= core.PREFIX_TOLERANCE
    assert meta["prefix_sha256"] == hashlib.sha256(prefix.encode("utf-8")).hexdigest()


def test_every_request_sends_cache_prompt_true_at_temperature_0_and_a_fixed_seed(tmp_path):
    _, server, _ = run(tmp_path)

    assert len(server.bodies) == cache_suite.SESSIONS * 10 == 30
    for body in server.bodies:
        assert body["cache_prompt"] is True
        assert body["stream"] is True
        assert body["temperature"] == 0
        assert body["seed"] == cache_suite.SEED
        assert body["max_tokens"] == cache_suite.REPLY_CAP_TOKENS == 64


def test_each_turn_appends_the_previous_reply_and_a_new_user_message(tmp_path):
    _, server, _ = run(tmp_path)

    third = server.bodies[2]["messages"]
    assert [m["role"] for m in third] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert [m["content"] for m in third[1::2]] == [TURNS[0].user, TURNS[1].user, TURNS[2].user]
    assert third[2]["content"] == "Reply 1 about the beds."
    second_session = server.bodies[10]["messages"]
    assert [m["role"] for m in second_session] == ["system", "user"]
    assert second_session[1]["content"] == TURNS[0].user


def test_rows_read_cache_n_prompt_n_and_prompt_ms_from_the_final_timings_fixture():
    text = (FIXTURE / "sse_with_timings.txt").read_text(encoding="utf-8")
    ticks = iter([10.0, 10.05, 10.12, 10.2, 10.3, 10.6, 10.61])
    client = lowtier_suite.ServerClient(
        "http://fake:8080",
        opener=lambda request, timeout: FakeResponse(text),
        clock=lambda: next(ticks),
    )
    timing = client.chat(cache_suite.turn_body([{"role": "user", "content": "x"}]))

    row = cache_suite.cache_row(
        arm="c0-ckpt",
        session=2,
        turn=3,
        turn_id="T03",
        result=timing.result,
        prefix_tokens=6093,
        full_reprocess_logged=0,
    )

    assert (row["cache_n"], row["prompt_n"], row["prompt_ms"]) == (6188, 97, 61.3)
    assert row["prompt_tokens"] == 6188 + 97
    assert row["reply_tokens"] == 42
    assert row["predicted_per_second"] == 80.71
    assert row["ttft_ms"] == pytest.approx(120.0)
    assert row["finish_reason"] == "stop"
    assert row["reply"].startswith("Put the tomatoes")
    assert (row["arm"], row["session"], row["turn"], row["turn_id"]) == ("c0-ckpt", 2, 3, "T03")


def test_requests_jsonl_holds_one_row_per_session_and_turn_with_the_results_fields(tmp_path):
    out, _, _ = run(tmp_path)

    written = rows(out / "requests.jsonl")
    assert [(r["session"], r["turn"]) for r in written] == [
        (s, t) for s in (1, 2, 3) for t in range(1, 11)
    ]
    for key in (
        "arm", "session", "turn", "prompt_tokens", "cache_n", "prompt_n", "prompt_ms",
        "ttft_ms", "predicted_per_second", "reply_tokens",
    ):  # fmt: skip
        assert all(key in r for r in written), key
    assert {r["run"] for r in written} == {"mh1"}
    assert {r["suite"] for r in written} == {"cache"}
    assert written[0]["cold"] is True
    assert not any(r["cold"] for r in written[1:])
    assert written[4]["cache_n"] == 6005


def test_sessions_and_turns_can_be_cut_for_a_smoke_pass(tmp_path):
    out, server, _ = run(tmp_path, sessions=2, turns=2)

    assert len(server.bodies) == 4
    assert len(rows(out / "requests.jsonl")) == 4


# ---------------------------------------------------------------------------
# The server log
# ---------------------------------------------------------------------------


def test_the_server_log_is_kept_and_searched_for_full_reprocessing_and_mmproj(tmp_path):
    out, _, logs = run(tmp_path)

    assert (out / "server.log").read_text(encoding="utf-8") == logs()
    scan = json.loads((out / "log_scan.json").read_text(encoding="utf-8"))
    assert scan["full_reprocess"] == 2
    assert scan["mmproj_loaded"] is False
    written = rows(out / "requests.jsonl")
    flagged = [(r["session"], r["turn"]) for r in written if r["full_reprocess_logged"]]
    assert flagged == [(1, 1), (2, 1)]


def test_scan_log_finds_an_mmproj_load():
    text = "clip_model_loader: loading model from /models/mmproj-F16.gguf\nother line\n"

    scan = cache_suite.scan_log(text)

    assert scan.mmproj_loaded is True
    assert scan.mmproj_lines == ("clip_model_loader: loading model from /models/mmproj-F16.gguf",)
    assert scan.full_reprocess == 0


def test_the_server_log_is_saved_even_when_a_request_fails(tmp_path):
    logs = Logs()
    server = FakeServer(fail_at=3, on_chat=logs.add)

    with pytest.raises(core.StreamError):
        run(tmp_path, logs=logs, server=server)

    arm_dir = tmp_path / "results" / "mh1" / "speech" / "cache" / "c0-ckpt"
    assert (arm_dir / "server.log").is_file()
    assert len(rows(arm_dir / "requests.jsonl")) == 2


# ---------------------------------------------------------------------------
# CLI, plan hooks and selftest
# ---------------------------------------------------------------------------


def test_the_cache_subcommand_requires_bench_run(tmp_path):
    with pytest.raises(SystemExit) as info:
        speech_bench.main(["cache", "--results-root", str(tmp_path), "--run", "mh1", "--arm", "c0"])

    assert info.value.code == 2


def test_the_cache_subcommand_refuses_an_unknown_arm(tmp_path):
    argv = ["cache", "--results-root", str(tmp_path), "--run", "mh1", "--bench-run", "w"]

    with pytest.raises(SystemExit):
        speech_bench.main([*argv, "--arm", "c2"])


def test_container_argv_prints_bench_hosts_serve_argv_for_each_cache_arm():
    ctx = core.build_plan_context()

    blocks = dict(cache_suite.container_argv(ctx))

    assert set(blocks) == {f"cache serve {arm}" for arm in cache_suite.CACHE_ARMS}
    assert blocks["cache serve c0-swafull"][-1] == "--swa-full"
    assert blocks["cache serve c0-ckpt"][-2:] == ["--checkpoint-min-step", "0"]


def test_validate_inputs_finds_no_problem_in_the_committed_inputs():
    assert cache_suite.validate_inputs(_REPO_ROOT) == []


def test_selftest_passes():
    cache_suite.selftest()
