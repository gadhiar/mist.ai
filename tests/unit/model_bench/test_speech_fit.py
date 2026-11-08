"""scripts/model_bench/speech/fit_suite.py: run 5, what fits beside E4B on the 12 GB card.

The loop containers, `bench_host vram-step`, the chat streams, `small-serve` and the docker state
reads are all fakes; `vram-step` is faked at `bench_host.main`'s argv, where the suite calls it.
No test touches a GPU, the network, Docker or an engine package.
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402

from scripts.model_bench import bench_host, speech_bench  # noqa: E402
from scripts.model_bench.speech import core, fit_suite, stt_suite, tts_suite  # noqa: E402

PINS = _REPO_ROOT / "tests" / "unit" / "model_bench" / "fixtures" / "speech" / "host" / "pins.json"
TOTAL_MIB = 12282
SMALL_MODELS = ("gemma4-e2b", "minicpm5-2b", "qwen35-4b")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeHandle:
    """A loop container: `ok` loads and runs, `fail` exits before READY, `crash` dies mid-run."""

    def __init__(self, argv: list[str], behaviour: str, tail: str):
        self.argv = argv
        self.behaviour = behaviour
        self.tail = tail
        self.stopped = False

    def wait_ready(self, timeout: float) -> bool:
        return self.behaviour != "fail"

    def running(self) -> bool:
        return self.behaviour == "ok" and not self.stopped

    def stop(self) -> None:
        self.stopped = True

    def outcome(self) -> fit_suite.LoopOutcome:
        if self.behaviour == "fail":
            return fit_suite.LoopOutcome(exit_code=1, ready=False, done=False, tail=self.tail)
        if self.behaviour == "crash":
            return fit_suite.LoopOutcome(exit_code=1, ready=True, done=False, tail=self.tail)
        return fit_suite.LoopOutcome(exit_code=143, ready=True, done=False, tail="READY")


class FakeLoops:
    """Starts a `FakeHandle` per argv, by the engine id the argv names."""

    def __init__(self, behaviours: dict[str, tuple[str, str]] | None = None):
        self.behaviours = behaviours or {}
        self.started: list[FakeHandle] = []
        self.logs: list[Path] = []

    def start(self, argv: list[str], *, log_path: Path) -> FakeHandle:
        engine = argv[argv.index("--engine") + 1]
        behaviour, tail = self.behaviours.get(engine, ("ok", ""))
        handle = FakeHandle(argv, behaviour, tail)
        self.started.append(handle)
        self.logs.append(log_path)
        return handle

    def engines(self) -> list[str]:
        return [h.argv[h.argv.index("--engine") + 1] for h in self.started]


class FakeBench:
    """`bench_host.main` for `vram-step`: writes the step into the window's vram_steps.json."""

    def __init__(self, root: Path, *, max_mib: dict[str, float] | None = None):
        self.root = root
        self.calls: list[list[str]] = []
        self.max_mib = max_mib or {}

    def __call__(self, argv: list[str]) -> int:
        self.calls.append(list(argv))
        assert argv[0] == "vram-step"
        run = argv[argv.index("--run") + 1]
        label = argv[argv.index("--label") + 1]
        path = Path(argv[argv.index("--results-root") + 1]) / run / "session" / "vram_steps.json"
        doc = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"steps": []}
        peak = self.max_mib.get(label, 10000.0)
        doc["steps"].append(
            {
                "label": label,
                "median_mib": peak - 300,
                "max_mib": peak,
                "total_mib": TOTAL_MIB,
                "samples": 50,
                "t_utc": "2026-11-07T01:00:00+00:00",
            }
        )
        path.write_text(json.dumps(doc), encoding="utf-8")
        return 0

    def labels(self) -> list[str]:
        return [c[c.index("--label") + 1] for c in self.calls]


class FakeChat:
    """Streams: E4B completions read their `timings`; the small model answers with set_timer."""

    def __init__(self, *, fail_e4b_at: int | None = None, fail_message: str = "boom"):
        self.lock = threading.Lock()
        self.calls: list[tuple[str, dict]] = []
        self.fail_e4b_at = fail_e4b_at
        self.fail_message = fail_message

    def __call__(self, base_url, body, *, clock=None, timeout=None) -> core.ChatStreamResult:
        with self.lock:
            self.calls.append((base_url, body))
            n_e4b = sum(1 for url, _ in self.calls if url == fit_suite.BENCH_LLM_URL)
        if base_url == fit_suite.BENCH_LLM_URL and n_e4b == self.fail_e4b_at:
            raise core.StreamError(self.fail_message)
        calls: tuple[dict, ...] = ()
        if base_url == core.SMALL_BASE_URL:
            calls = (
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "set_timer", "arguments": '{"duration_seconds": 600}'},
                },
            )
        return core.ChatStreamResult(
            content="" if calls else "story text",
            reasoning_content="",
            tool_calls=calls,
            t_request=1.0,
            t_first_content=1.1,
            t_first_tool_call=1.1 if calls else None,
            t_end=4.0,
            finish_reason="length",
            timings={"predicted_n": 256, "predicted_ms": 3200.0, "predicted_per_second": 80.0},
        )

    def e4b_bodies(self) -> list[dict]:
        return [b for url, b in self.calls if url == fit_suite.BENCH_LLM_URL]

    def small_bodies(self) -> list[dict]:
        return [b for url, b in self.calls if url == core.SMALL_BASE_URL]


class FakeSmall:
    """`core.serve_small` and `core.unserve_small`, recording each call."""

    def __init__(self, *, fail: set[str] | None = None):
        self.fail = fail or set()
        self.served: list[tuple[str, str, Path]] = []
        self.unserved: list[Path] = []
        self.up = False

    def serve(self, model_id, placement, *, out_dir, pins_path, models_dirs, **kwargs):
        self.served.append((model_id, placement, out_dir))
        out_dir.mkdir(parents=True)
        if model_id in self.fail:
            raise core.ServeFailedError(
                "mist-bench-small exited (code=1) while waiting for /health; "
                "cudaMalloc failed: out of memory"
            )
        self.up = True
        return core.ServeResult(
            model_id=model_id,
            placement=placement,
            alias=core.small_alias(model_id, placement),
            argv=(),
            image="cuda",
            ngl="999",
            load_s=4.0,
            warmup_ttft_s=0.1,
            base_url=core.SMALL_BASE_URL,
            out_dir=out_dir,
        )

    def unserve(self, out_dir, **kwargs) -> float:
        self.unserved.append(out_dir)
        self.up = False
        return 1.0

    def state(self, name: str) -> str:
        if name == core.SMALL_CONTAINER:
            return "running" if self.up else "absent"
        if name == bench_host.BENCH_LLM_CONTAINER:
            return "running"
        return "absent"


def make_material(tmp_path: Path) -> Path:
    material = tmp_path / "material"
    material.mkdir()
    data = b"RIFF fake wav"
    (material / "U01.wav").write_bytes(data)
    item = {
        "id": "U01",
        "file": "U01.wav",
        "kind": "complete",
        "text": "Set a timer for ten minutes.",
        "duration_s": 2.0,
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    (material / "manifest.json").write_text(json.dumps({"items": [item]}), encoding="utf-8")
    return material


def make_window(root: Path, bench_run: str, *, desktop: float | None = 1200.0) -> None:
    session = root / bench_run / "session"
    session.mkdir(parents=True)
    (session / "snapshot.json").write_text("{}", encoding="utf-8")
    if desktop is not None:
        step = {"label": "desktop", "median_mib": desktop, "max_mib": desktop + 50}
        (session / "vram_steps.json").write_text(json.dumps({"steps": [step]}), encoding="utf-8")


class Setup:
    """One fit invocation's request and fakes."""

    def __init__(self, tmp_path: Path, part: str, **kwargs):
        self.root = tmp_path / "results"
        self.bench_run = "mh1-h4" if part == "speech" else "mh1-h5"
        make_window(self.root, self.bench_run, desktop=kwargs.pop("desktop", 1200.0))
        self.speech_models = tmp_path / "speech-models"
        self.speech_models.mkdir()
        self.loops = kwargs.pop("loops", None) or FakeLoops()
        self.bench = kwargs.pop("bench", None) or FakeBench(self.root)
        self.chat = kwargs.pop("chat", None) or FakeChat()
        self.small = kwargs.pop("small", None) or FakeSmall()
        self.served_checks = 0
        self.state = kwargs.pop("state", None) or self.small.state
        self.request = fit_suite.FitRequest(
            results_root=self.root,
            run="mh1",
            bench_run=self.bench_run,
            part=part,
            material_dir=make_material(tmp_path),
            speech_models_dir=self.speech_models,
            only=tuple(kwargs.pop("only", ())),
            rounds=kwargs.pop("rounds", fit_suite.ROUNDS),
            pins_path=PINS,
            models_dirs={"small": tmp_path / "small", "main": tmp_path / "main"},
        )
        self.served_check = kwargs.pop("served_check", None) or self._served
        assert not kwargs, kwargs

    def _served(self):
        self.served_checks += 1

    def deps(self) -> fit_suite.FitDeps:
        return fit_suite.FitDeps(
            loops=self.loops,
            served_check=self.served_check,
            bench_main=self.bench,
            chat=self.chat,
            container_state=self.state,
            serve=self.small.serve,
            unserve=self.small.unserve,
        )

    def run(self) -> Path:
        return fit_suite.run_fit(self.request, self.deps(), git_sha="abc123")

    @property
    def steps_path(self) -> Path:
        return self.root / "mh1" / "speech" / "fit" / "steps.json"

    def steps(self) -> dict[str, dict]:
        doc = json.loads(self.steps_path.read_text(encoding="utf-8"))
        return {s["label"]: s for s in doc["steps"]}


COMBOS = [f"e4b+{t}+{s}" for t in fit_suite.FIT_TTS for s in fit_suite.FIT_STT]


# ---------------------------------------------------------------------------
# The steps
# ---------------------------------------------------------------------------


def test_the_speech_part_is_e4b_alone_then_the_12_combinations():
    steps = fit_suite.build_steps("speech")

    assert [s.label for s in steps] == ["e4b", *COMBOS]
    assert len(COMBOS) == 12
    assert fit_suite.FIT_TTS == ("qwen3", "turbo", "original")
    assert fit_suite.FIT_STT == ("whisper-base", "parakeet-gpu", "nemotron-160", "parakeet-cpu")


def test_the_small_part_is_9_steps_three_per_small_model_without_the_reference():
    steps = fit_suite.build_steps("small")

    assert fit_suite.fit_small_models() == SMALL_MODELS
    assert [s.label for s in steps] == [
        label
        for m in SMALL_MODELS
        for label in (f"e4b+{m}", f"e4b+{m}+qwen3+parakeet-gpu", f"e4b+{m}+turbo+parakeet-gpu")
    ]


def test_build_steps_refuses_an_unknown_part():
    with pytest.raises(core.SpeechBenchError):
        fit_suite.build_steps("game")


# ---------------------------------------------------------------------------
# vram-step under --bench-run
# ---------------------------------------------------------------------------


def test_every_step_is_sampled_by_vram_step_under_the_bench_run(tmp_path):
    setup = Setup(tmp_path, "speech")

    setup.run()

    assert setup.bench.labels() == [f"fit-{label}" for label in ["e4b", *COMBOS]]
    for call in setup.bench.calls:
        assert call[call.index("--run") + 1] == "mh1-h4"
        assert Path(call[call.index("--results-root") + 1]) == setup.root
        assert call[call.index("--seconds") + 1] == "10"


def test_steps_json_holds_one_step_per_combination_with_fits_and_headroom(tmp_path):
    setup = Setup(tmp_path, "speech", bench=None)
    setup.bench.max_mib = {"fit-e4b+turbo+parakeet-gpu": 11000.0}

    setup.run()

    steps = setup.steps()
    assert list(steps) == ["e4b", *COMBOS]
    step = steps["e4b+turbo+parakeet-gpu"]
    for key in ("median_mib", "max_mib", "total_mib", "samples", "t_utc"):
        assert key in step, key
    assert step["fits"] is True
    assert step["headroom_mib"] == TOTAL_MIB - 11000.0
    assert step["by_difference_mib"] == 11000.0 - 1200.0
    assert step["components"] == {
        "llm": "c0",
        "tts": "turbo",
        "stt": "parakeet-gpu",
        "small": None,
    }
    assert step["vram_step_label"] == "fit-e4b+turbo+parakeet-gpu"
    assert step["bench_run"] == "mh1-h4"
    assert (step["run"], step["suite"], step["git_sha"]) == ("mh1", "fit", "abc123")


def test_fit_entry_arithmetic_without_a_desktop_step():
    step = fit_suite.build_steps("speech")[0]
    summary = {"median_mib": 8000, "max_mib": 8400, "total_mib": 12282, "samples": 50}

    entry = fit_suite.fit_entry(step, summary, fits=True, message=None, oom=False, desktop_mib=None)

    assert entry["headroom_mib"] == 12282 - 8400
    assert entry["by_difference_mib"] is None


def test_fit_refuses_a_label_vram_steps_already_holds(tmp_path):
    setup = Setup(tmp_path, "speech")
    path = setup.root / "mh1-h4" / "session" / "vram_steps.json"
    path.write_text(json.dumps({"steps": [{"label": "fit-e4b"}]}), encoding="utf-8")

    with pytest.raises(core.OutputExistsError, match="fit-e4b"):
        setup.run()

    assert setup.bench.calls == []
    assert not (setup.root / "mh1").exists()


def test_fit_refuses_a_step_steps_json_already_holds(tmp_path):
    setup = Setup(tmp_path, "speech", only=["e4b"])
    setup.run()
    setup.bench = FakeBench(setup.root)

    with pytest.raises(core.OutputExistsError):
        setup.run()

    assert setup.bench.calls == []
    assert list(setup.steps()) == ["e4b"]


def test_fit_refuses_a_bench_run_without_a_snapshot(tmp_path):
    setup = Setup(tmp_path, "speech")
    (setup.root / "mh1-h4" / "session" / "snapshot.json").unlink()

    with pytest.raises(core.BenchRunError):
        setup.run()

    assert not (setup.root / "mh1").exists()


def test_fit_refuses_when_e4b_is_not_the_served_arm(tmp_path):
    def mismatch():
        raise bench_host.ServedArmMismatchError("mist-bench-llm Args do not match arm 'c0'")

    setup = Setup(tmp_path, "speech", served_check=mismatch)

    with pytest.raises(bench_host.ServedArmMismatchError):
        setup.run()

    assert setup.loops.started == []
    assert not (setup.root / "mh1").exists()


def test_fit_runs_only_the_named_steps(tmp_path):
    setup = Setup(tmp_path, "speech", only=["e4b+qwen3+parakeet-cpu"])

    setup.run()

    assert list(setup.steps()) == ["e4b+qwen3+parakeet-cpu"]


def test_fit_refuses_an_unknown_step_name(tmp_path):
    setup = Setup(tmp_path, "speech", only=["e4b+pocket+parakeet-cpu"])

    with pytest.raises(core.SpeechBenchError, match="pocket"):
        setup.run()


# ---------------------------------------------------------------------------
# The load: the T2 and T3 loops, the completions
# ---------------------------------------------------------------------------


def test_loops_are_started_from_t2s_and_t3s_loop_argv(tmp_path):
    setup = Setup(tmp_path, "speech", only=["e4b+qwen3+nemotron-160"])

    setup.run()

    tts_argv, stt_argv = (h.argv for h in setup.loops.started)
    assert tts_argv == tts_suite.loop_container_argv(
        "qwen3",
        line_id=fit_suite.DEFAULT_LINE,
        seconds=fit_suite.LOOP_SECONDS,
        repo_dir=core.REPO_ROOT.resolve().as_posix(),
        speech_models_dir=setup.speech_models.resolve().as_posix(),
        name=fit_suite.TTS_LOOP_NAME,
    )
    assert stt_argv == stt_suite.loop_container_argv(
        "nemotron-160",
        utt_id=fit_suite.DEFAULT_UTT,
        seconds=fit_suite.LOOP_SECONDS,
        repo_dir=core.REPO_ROOT.resolve().as_posix(),
        material_dir=setup.request.material_dir.resolve().as_posix(),
        weights_dir=setup.speech_models.resolve().as_posix(),
        name=fit_suite.STT_LOOP_NAME,
    )
    assert all(h.stopped for h in setup.loops.started)
    assert all(p.parent.is_dir() for p in setup.loops.logs)


def test_e4b_alone_starts_no_loop(tmp_path):
    setup = Setup(tmp_path, "speech", only=["e4b"])

    setup.run()

    assert setup.loops.started == []
    assert setup.steps()["e4b"]["fits"] is True


def test_each_step_runs_five_256_token_completions_on_e4b(tmp_path):
    setup = Setup(tmp_path, "speech", only=["e4b+turbo+whisper-base"])

    setup.run()

    bodies = setup.chat.e4b_bodies()
    assert len(bodies) == fit_suite.ROUNDS == 5
    assert all(b["max_tokens"] == fit_suite.COMPLETION_TOKENS == 256 for b in bodies)
    step = setup.steps()["e4b+turbo+whisper-base"]
    assert step["rounds_completed"] == 5
    assert [r["e4b_predicted_per_second"] for r in step["rounds"]] == [80.0] * 5


def test_a_speech_engine_that_fails_to_load_is_a_row_and_the_suite_goes_on(tmp_path):
    loops = FakeLoops({"qwen3": ("fail", "torch.OutOfMemoryError: CUDA out of memory.")})
    setup = Setup(tmp_path, "speech", loops=loops)

    setup.run()

    steps = setup.steps()
    assert list(steps) == ["e4b", *COMBOS]
    for stt in fit_suite.FIT_STT:
        failed = steps[f"e4b+qwen3+{stt}"]
        assert failed["fits"] is False
        assert failed["oom"] is True
        assert "qwen3" in failed["message"] and "out of memory" in failed["message"]
        assert failed["max_mib"] is None and failed["headroom_mib"] is None
    assert all(steps[f"e4b+turbo+{stt}"]["fits"] for stt in fit_suite.FIT_STT)
    assert not any("qwen3" in label for label in setup.bench.labels())
    assert all(h.stopped for h in loops.started)


def test_an_engine_that_runs_out_of_memory_during_the_sample_is_fits_false(tmp_path):
    loops = FakeLoops({"parakeet-gpu": ("crash", "RuntimeError: CUDA error: out of memory")})
    setup = Setup(tmp_path, "speech", loops=loops, only=["e4b+original+parakeet-gpu"])

    setup.run()

    step = setup.steps()["e4b+original+parakeet-gpu"]
    assert step["fits"] is False
    assert step["oom"] is True
    assert "parakeet-gpu" in step["message"]
    assert step["max_mib"] == 10000.0


def test_a_failed_e4b_completion_is_fits_false_with_its_message(tmp_path):
    chat = FakeChat(fail_e4b_at=2, fail_message="POST failed: out of memory")
    setup = Setup(tmp_path, "speech", chat=chat, only=["e4b+turbo+parakeet-cpu"])

    setup.run()

    step = setup.steps()["e4b+turbo+parakeet-cpu"]
    assert step["fits"] is False
    assert step["oom"] is True
    assert "round 2" in step["message"]
    assert step["rounds_completed"] == 4


def test_fit_stops_with_a_clear_error_when_mist_bench_llm_is_gone(tmp_path):
    calls = {"n": 0}

    def state(name):
        if name == bench_host.BENCH_LLM_CONTAINER:
            calls["n"] += 1
            return "running" if calls["n"] <= 1 else "exited"
        return "absent"

    setup = Setup(tmp_path, "speech", state=state)

    with pytest.raises(core.BenchRunError, match="mist-bench-llm"):
        setup.run()

    assert list(setup.steps()) == ["e4b"]


def test_fit_refuses_when_a_loop_container_name_is_taken(tmp_path):
    def state(name):
        taken = (fit_suite.TTS_LOOP_NAME, bench_host.BENCH_LLM_CONTAINER)
        return "running" if name in taken else "absent"

    setup = Setup(tmp_path, "speech", state=state)

    with pytest.raises(core.SpeechBenchError, match=fit_suite.TTS_LOOP_NAME):
        setup.run()

    assert setup.loops.started == []


def test_fit_refuses_material_that_lacks_the_loop_utterance(tmp_path):
    setup = Setup(tmp_path, "speech")
    request = fit_suite.FitRequest(
        **{
            **{f: getattr(setup.request, f) for f in setup.request.__dataclass_fields__},
            "utt_id": "U02",
        }
    )

    with pytest.raises(core.ManifestError):
        fit_suite.run_fit(request, setup.deps())


# ---------------------------------------------------------------------------
# The small-model steps
# ---------------------------------------------------------------------------


def test_each_small_model_is_served_through_serve_small_at_full_offload(tmp_path):
    setup = Setup(tmp_path, "small")

    setup.run()

    speech = setup.root.resolve() / "mh1" / "speech"
    assert setup.small.served == [
        (m, "gpu-full", core.small_serve_dir(speech, m, "gpu-full", "fit")) for m in SMALL_MODELS
    ]
    assert setup.small.unserved == [d for _, _, d in setup.small.served]
    assert len(setup.bench.calls) == 9
    assert all(c[c.index("--run") + 1] == "mh1-h5" for c in setup.bench.calls)
    steps = setup.steps()
    assert steps["e4b+qwen35-4b+turbo+parakeet-gpu"]["components"] == {
        "llm": "c0",
        "tts": "turbo",
        "stt": "parakeet-gpu",
        "small": "qwen35-4b",
    }
    assert all(s["fits"] for s in steps.values())


def test_each_small_step_sends_one_tool_prompt_per_round_and_scores_it(tmp_path):
    setup = Setup(tmp_path, "small", only=["e4b+gemma4-e2b"])

    setup.run()

    bodies = setup.chat.small_bodies()
    assert len(bodies) == 5
    assert {b["model"] for b in bodies} == {"gemma4-e2b@gpu-full"}
    assert all(b["tools"] and b["messages"][0]["role"] == "system" for b in bodies)
    assert [b["messages"][1]["content"] for b in bodies][0] == "Set a timer for ten minutes."
    rounds = setup.steps()["e4b+gemma4-e2b"]["rounds"]
    assert [r["prompt_id"] for r in rounds] == ["C01", "C02", "C03", "C04", "C05"]
    assert rounds[0]["tool_valid_first"] is True
    assert rounds[1]["tool_valid_first"] is False
    assert len(setup.chat.e4b_bodies()) == 5


def test_a_small_model_that_fails_to_load_records_its_three_steps_as_not_fitting(tmp_path):
    small = FakeSmall(fail={"minicpm5-2b"})
    setup = Setup(tmp_path, "small", small=small)

    setup.run()

    steps = setup.steps()
    assert len(steps) == 9
    failed = [label for label, s in steps.items() if not s["fits"]]
    assert failed == [label for label in steps if "minicpm5-2b" in label]
    assert all(steps[label]["oom"] for label in failed)
    assert all("minicpm5-2b" in steps[label]["message"] for label in failed)
    assert not any("minicpm5-2b" in label for label in setup.bench.labels())


def test_the_small_part_refuses_an_existing_fit_serve_directory(tmp_path):
    setup = Setup(tmp_path, "small")
    speech = setup.root.resolve() / "mh1" / "speech"
    core.small_serve_dir(speech, "qwen35-4b", "gpu-full", "fit").mkdir(parents=True)

    with pytest.raises(core.OutputExistsError):
        setup.run()

    assert setup.small.served == []


# ---------------------------------------------------------------------------
# CLI, plan hooks and selftest
# ---------------------------------------------------------------------------


def test_the_fit_subcommand_requires_bench_run(tmp_path):
    argv = ["fit", "--results-root", str(tmp_path), "--run", "mh1", "--part", "speech"]

    with pytest.raises(SystemExit) as info:
        speech_bench.main(argv)

    assert info.value.code == 2


def test_is_oom_recognises_the_engines_out_of_memory_messages():
    assert fit_suite.is_oom("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate")
    assert fit_suite.is_oom("ggml_backend_cuda_buffer_type_alloc_buffer: cudaMalloc failed")
    assert not fit_suite.is_oom("ModuleNotFoundError: No module named 'nemo'")


def test_container_argv_prints_each_loop_the_fit_suite_starts():
    blocks = dict(fit_suite.container_argv(core.build_plan_context()))

    assert set(blocks) == {f"fit loop {e}" for e in (*fit_suite.FIT_TTS, *fit_suite.FIT_STT)}
    assert fit_suite.TTS_LOOP_NAME in blocks["fit loop turbo"]
    assert fit_suite.STT_LOOP_NAME in blocks["fit loop parakeet-gpu"]


def test_validate_inputs_finds_no_problem_in_the_committed_inputs():
    assert fit_suite.validate_inputs(_REPO_ROOT) == []


def test_selftest_passes():
    fit_suite.selftest()
