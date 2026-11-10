"""Runs 8 and 9: `game_suite.py` (the guided `game-session`) and `frametimes.py`.

The session is driven with fakes for every outside thing: the prompts Raj answers, the clock,
the subprocess runner (PresentMon, typeperf and nvidia-smi write the committed fixtures instead),
bench_host's snapshot, stop and restore, small-serve, the chain services and the administrator
check. The fixtures hold invented data only. No test touches a GPU, the network, Docker,
PresentMon or an engine package.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import sys
import urllib.request
import wave
from datetime import UTC, datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402

from scripts.model_bench import speech_bench  # noqa: E402
from scripts.model_bench.probes import nvidia_smi  # noqa: E402
from scripts.model_bench.speech import chain, core, frametimes, game_suite  # noqa: E402

GAME_FIXTURES = _REPO_ROOT / "tests" / "unit" / "model_bench" / "fixtures" / "speech" / "game"
PRESENTMON_CSV = GAME_FIXTURES / "presentmon.csv"
TYPEPERF_CSV = GAME_FIXTURES / "typeperf.csv"
UTTERANCES = {
    u["id"]: u["text"]
    for u in json.loads(core.UTTERANCES_PATH.read_text(encoding="utf-8"))["utterances"]
}
NOW = datetime(2026, 11, 14, 19, 30, 5, tzinfo=UTC)
CARD_MIB = 12282


# ---------------------------------------------------------------------------
# frametimes.py
# ---------------------------------------------------------------------------


def test_presentmon_fixture_gives_average_one_percent_low_and_p99():
    summary = frametimes.summarize_presentmon(PRESENTMON_CSV)

    # 100 frames: 97 of 10 ms, 2 of 25 ms and 1 of 40 ms, 1,060 ms in all.
    assert summary.source == "presentmon"
    assert summary.frames == 100
    assert summary.avg_fps == pytest.approx(100_000 / 1060)
    # The slowest 1% is one frame of 40 ms: 1000 / 40.
    assert summary.low_1pct_fps == pytest.approx(25.0)
    assert summary.p99_frame_ms == pytest.approx(25.0)
    assert summary.duration_s == pytest.approx(1.06)
    assert summary.displayed_frames == 99


def test_one_percent_low_averages_the_slowest_one_percent_rounded_up():
    times = [10.0] * 147 + [30.0, 50.0, 70.0]

    summary = frametimes.summarize_frame_times(times)

    # 150 frames: the slowest 1% is 2 frames (1.5 rounded up), mean 60 ms.
    assert summary.low_1pct_fps == pytest.approx(1000 / 60)
    assert summary.p99_frame_ms == pytest.approx(50.0)


def test_frame_time_falls_back_to_presentmon_2_s_frame_time_column():
    text = "Application,ProcessID,FrameTime,DisplayedTime\nx.exe,1,20.0,20.0\nx.exe,1,30.0,0\n"

    times, displayed = frametimes.frame_times(text)

    assert times == [20.0, 30.0]
    assert displayed == 1


def test_a_capture_with_no_frames_reports_none_rather_than_dividing_by_zero():
    summary = frametimes.summarize_frame_times([])

    assert summary.frames == 0
    assert (summary.avg_fps, summary.low_1pct_fps, summary.p99_frame_ms) == (None, None, None)


def test_a_csv_without_a_frame_time_column_is_refused():
    with pytest.raises(frametimes.FrameTimeError):
        frametimes.frame_times("Application,ProcessID\nx.exe,1\n")


def test_the_game_s_own_benchmark_figures_stand_in_for_presentmon():
    summary = frametimes.from_game_benchmark(avg_fps=88.0, low_1pct_fps=61.5, p99_frame_ms=None)

    assert summary.source == "game_benchmark"
    assert (summary.avg_fps, summary.low_1pct_fps, summary.p99_frame_ms) == (88.0, 61.5, None)
    assert summary.frames is None


def test_a_typed_frame_rate_stands_in_with_no_one_percent_low():
    summary = frametimes.from_typed_fps(72.0)

    assert summary.source == "typed"
    assert (summary.avg_fps, summary.low_1pct_fps) == (72.0, None)


@pytest.mark.parametrize("text", ["", "abc", "-3", "0"])
def test_typed_figures_must_be_positive_numbers(text):
    with pytest.raises(frametimes.FrameTimeError):
        frametimes.parse_positive(text, "average frame rate")


def test_an_optional_typed_figure_may_be_left_blank():
    assert frametimes.parse_positive("  ", "1% low", optional=True) is None
    assert frametimes.parse_positive("61.5", "1% low", optional=True) == 61.5


# ---------------------------------------------------------------------------
# typeperf and nvidia-smi parsing
# ---------------------------------------------------------------------------


def test_typeperf_fixture_parses_the_total_cpu_and_a_process_s_counters():
    capture = game_suite.parse_typeperf_csv(TYPEPERF_CSV.read_text(encoding="utf-8"))

    assert len(capture.rows) == 5
    total = game_suite.counter_summary(capture, r"\Processor(_Total)\% Processor Time")
    assert (total["n"], total["mean"], total["max"]) == (4, 35.0, 50.0)
    cpu = game_suite.counter_summary(capture, r"\Process(java)\% Processor Time")
    assert (cpu["mean"], cpu["max"]) == (150.0, 210.0)
    memory = game_suite.counter_summary(capture, r"\Process(java)\Working Set - Private")
    assert memory["max"] / 2**20 == pytest.approx(2248.0)


def test_typeperf_argv_samples_the_total_cpu_at_1_hz_for_the_capture():
    argv = game_suite.typeperf_argv(Path("cpu.csv"), 120)

    assert argv[0] == "typeperf"
    assert r"\Processor(_Total)\% Processor Time" in argv
    assert argv[argv.index("-si") + 1] == "1" and argv[argv.index("-sc") + 1] == "120"
    assert argv[argv.index("-f") + 1] == "CSV" and argv[argv.index("-o") + 1] == "cpu.csv"


def test_typeperf_argv_adds_the_server_process_counters_for_run_9():
    argv = game_suite.typeperf_argv(Path("cpu.csv"), 120, server_process="java")

    assert r"\Process(java)\% Processor Time" in argv
    assert r"\Process(java)\Working Set - Private" in argv


def test_the_game_s_own_nvidia_smi_query_adds_utilization_without_changing_the_shared_one():
    argv = game_suite.nvidia_smi_argv(Path("gpu.csv"))
    query = argv[1].split("=", 1)[1].split(",")

    assert "utilization.gpu" in query
    assert {"memory.used", "memory.total", "clocks.sm", "clocks.mem", "power.draw"} <= set(query)
    assert argv[argv.index("-lms") + 1] == "500"
    assert argv[argv.index("-f") + 1] == "gpu.csv"
    assert "utilization.gpu" not in nvidia_smi.QUERY_FIELDS


def gpu_line(used: int, util: int = 80, total: int = CARD_MIB) -> str:
    return f"2026/11/14 19:31:00.000, {used}, {total}, {util}, 2505, 10501, 180.50\n"


def test_gpu_csv_parses_into_peak_memory_and_mean_utilisation():
    text = gpu_line(5000, 60) + gpu_line(6200, 90) + "garbage\n" + gpu_line(5800, 75)

    summary = game_suite.gpu_summary(game_suite.parse_gpu_csv(text))

    assert summary["samples"] == 3
    assert summary["peak_used_mib"] == 6200 and summary["total_mib"] == CARD_MIB
    assert summary["mean_util_pct"] == pytest.approx(75.0)
    assert summary["mean_power_w"] == pytest.approx(180.5)


def test_presentmon_argv_is_the_plan_s_timed_capture():
    argv = game_suite.presentmon_argv(
        Path("C:/pm/PresentMon.exe"), "examplegame.exe", Path("p.csv"), 120
    )

    assert argv[0] == "C:/pm/PresentMon.exe"
    assert argv[argv.index("--process_name") + 1] == "examplegame.exe"
    assert argv[argv.index("--output_file") + 1] == "p.csv"
    assert argv[argv.index("--timed") + 1] == "120"
    assert "--terminate_after_timed" in argv


# ---------------------------------------------------------------------------
# The 4070 slice
# ---------------------------------------------------------------------------

FOOTPRINTS = {
    "gpu-full": game_suite.SliceOption(3500.0, None),
    "gpu-4g": game_suite.SliceOption(3900.0, 30),
    "gpu-2g": game_suite.SliceOption(2000.0, 12),
}


def test_the_slice_is_the_largest_that_leaves_512_mib_beside_g0_s_peak():
    assert game_suite.choose_slice(CARD_MIB, 6000, FOOTPRINTS) == "gpu-full"
    # 12282 - 8000 = 4282 free: full needs 4012 with the margin, so it still fits.
    assert game_suite.choose_slice(CARD_MIB, 8000, FOOTPRINTS) == "gpu-full"
    # 3282 free: neither full (4012) nor 4g (4412) fits; 2g (2512) does.
    assert game_suite.choose_slice(CARD_MIB, 9000, FOOTPRINTS) == "gpu-2g"
    assert game_suite.choose_slice(CARD_MIB, 10000, FOOTPRINTS) is None


def test_the_slice_skips_a_placement_run_7_left_no_footprint_for():
    footprints = {"gpu-2g": FOOTPRINTS["gpu-2g"]}

    assert game_suite.choose_slice(CARD_MIB, 6000, footprints) == "gpu-2g"


def write_lowtier_results(speech: Path, model: str = "gemma4-e2b") -> None:
    full = speech / "lowtier" / model / "gpu-full"
    full.mkdir(parents=True)
    (full / "memory.json").write_text(json.dumps({"gpu_by_difference_mib": 3500}))
    for placement, ngl, footprint in (("gpu-4g", 30, 3900.0), ("gpu-2g", 12, 2000.0)):
        record = speech / "lowtier" / "slice" / model / placement
        record.mkdir(parents=True)
        probes = [{"ngl": ngl, "footprint_mib": footprint}, {"ngl": ngl + 1, "footprint_mib": 9e3}]
        doc = {"chosen_ngl": ngl, "probes": probes, "slice_mib": 4096}
        (record / "slice.json").write_text(json.dumps(doc))


def test_slice_footprints_are_read_from_run_7_s_results(tmp_path):
    write_lowtier_results(tmp_path)

    assert game_suite.slice_footprints(tmp_path, "gemma4-e2b") == FOOTPRINTS


# ---------------------------------------------------------------------------
# Fakes for the session
# ---------------------------------------------------------------------------


class Clock:
    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds

    def sleep(self, seconds: float) -> None:
        self.t += max(0.0, seconds)


class Asker:
    """Answers each prompt by the first key it contains; records every prompt."""

    def __init__(self, answers: dict[str, str]):
        self.answers = answers
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        for key, answer in self.answers.items():
            if key in prompt:
                return answer
        return ""


ANSWERS = {
    "process name": "examplegame",
    "game's name": "Example Game",
    "version": "1.4.2",
    "settings": "1440p, high preset, 120 fps cap, no frame generation",
}


class Handle:
    def __init__(self, log, argv):
        self.argv = argv
        self.terminated = False

    def wait(self, timeout):
        return 0

    def terminate(self):
        self.terminated = True


class FakeRunner:
    """Writes the fixtures where PresentMon, typeperf and nvidia-smi would write their files."""

    def __init__(self, gpu_used_for, *, presentmon_frames=True):
        self.gpu_used_for = gpu_used_for
        self.presentmon_frames = presentmon_frames
        self.started: list[list[str]] = []

    def start(self, argv, *, log_path):
        self.started.append(list(argv))
        if "--output_file" in argv:
            out = Path(argv[argv.index("--output_file") + 1])
            if self.presentmon_frames:
                shutil.copy(PRESENTMON_CSV, out)
            else:
                out.write_text(PRESENTMON_CSV.read_text().splitlines()[0] + "\n")
        elif argv[0] == "typeperf":
            shutil.copy(TYPEPERF_CSV, Path(argv[argv.index("-o") + 1]))
        elif argv[0] == "nvidia-smi":
            out = Path(argv[argv.index("-f") + 1])
            used = self.gpu_used_for(out)
            out.write_text(gpu_line(used - 300) + gpu_line(used) + gpu_line(used - 100))
        return Handle(log_path, argv)


class JsonResponse:
    def __init__(self, doc, clock, delay):
        self.status = 200
        self.headers = {}
        self._body = json.dumps(doc).encode()
        self._clock = clock
        self._delay = delay

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def read(self, *_):
        self._clock.advance(self._delay)
        return self._body


class SseResponse:
    def __init__(self, events, clock):
        self.status = 200
        self.headers = {}
        self._events = events
        self._clock = clock

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def __iter__(self):
        for delay, event in self._events:
            self._clock.advance(delay)
            yield (
                "data: " + (event if isinstance(event, str) else json.dumps(event)) + "\n"
            ).encode()


class PcmResponse:
    def __init__(self, clock):
        self.status = 200
        self.headers = {"X-Sample-Rate": "24000"}
        self._chunks = [(0.2, b"\x01\x00" * 100)]
        self._clock = clock

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def read1(self, *_):
        if not self._chunks:
            return b""
        delay, data = self._chunks.pop(0)
        self._clock.advance(delay)
        return data


def reply(text):
    return [
        (0.1, {"choices": [{"index": 0, "delta": {"content": text}}]}),
        (0.01, {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "timings": {}}),
        (0.0, "[DONE]"),
    ]


class FakeServers:
    """The low level's server, the chain services, host-1070's /v1/models and mist-llm."""

    def __init__(self, clock, *, remote_alias=None):
        self.clock = clock
        self.remote_alias = remote_alias
        self.chat_urls: list[str] = []
        self.e4b_requests = 0
        self.local_alias = None

    def __call__(self, request, timeout=None):
        url = request.full_url if isinstance(request, urllib.request.Request) else str(request)
        if url.endswith("/v1/models"):
            alias = self.remote_alias if "8090" in url else self.local_alias
            return JsonResponse({"data": [{"id": alias}] if alias else []}, self.clock, 0.0)
        if url.endswith("/transcribe"):
            return JsonResponse(
                {"text": "Set a timer for ten minutes.", "server_ms": 90.0}, self.clock, 0.15
            )
        if url.endswith("/speak"):
            return PcmResponse(self.clock)
        if url.endswith("/v1/chat/completions"):
            self.chat_urls.append(url)
            if url.startswith(game_suite.E4B_URL):
                self.e4b_requests += 1
                return SseResponse(
                    [
                        (2.5, {"choices": [{"index": 0, "delta": {"content": "Hi"}}]}),
                        (0.0, "[DONE]"),
                    ],
                    self.clock,
                )
            return SseResponse(
                reply("Your timer is set and running for ten minutes now, as asked."), self.clock
            )
        raise AssertionError(f"unexpected request to {url}")


class FakeBenchHost:
    def __init__(self, results_root, clock, *, restore_code=0):
        self.results_root = results_root
        self.clock = clock
        self.calls: list[list[str]] = []
        self.restore_code = restore_code

    def __call__(self, argv):
        self.calls.append(list(argv))
        if argv[0] == "snapshot":
            session = (
                Path(argv[argv.index("--results-root") + 1])
                / argv[argv.index("--run") + 1]
                / "session"
            )
            session.mkdir(parents=True)
            (session / "snapshot.json").write_text("{}")
        if argv[0] == "restore":
            self.clock.advance(30.0)
            return self.restore_code
        return 0


class FakeServe:
    def __init__(self, servers):
        self.servers = servers
        self.served: list[tuple[str, str, int | None, Path]] = []
        self.unserved: list[Path] = []

    def serve(self, model_id, placement, *, out_dir, pins_path, models_dirs, ngl=None):
        out_dir.mkdir(parents=True)
        self.served.append((model_id, placement, ngl, out_dir))
        self.servers.local_alias = core.small_alias(model_id, placement)

    def unserve(self, out_dir):
        self.unserved.append(out_dir)
        self.servers.local_alias = None
        return 1.0


class FakeServices:
    def __init__(self):
        self.events: list[str] = []

    def factory(self):
        services = self

        class _One:
            def start(self):
                services.events.append("start")

            def stop(self, log_dir):
                services.events.append(f"stop {Path(log_dir).name}")

        return _One()


def write_material(root: Path) -> Path:
    material = root / "material"
    material.mkdir()
    items = []
    for uid in chain.CHAIN_UTTERANCES:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as clip:
            clip.setnchannels(1)
            clip.setsampwidth(2)
            clip.setframerate(16000)
            clip.writeframes(b"\x10\x00" * 800)
        (material / f"{uid}.wav").write_bytes(buffer.getvalue())
        items.append(
            {"id": uid, "file": f"{uid}.wav", "kind": "complete", "text": UTTERANCES[uid],
             "duration_s": 0.05, "sha256": hashlib.sha256(buffer.getvalue()).hexdigest()}
        )  # fmt: skip
    (material / "manifest.json").write_text(json.dumps({"items": items}))
    return material


class Session:
    """One session's request and its fakes."""

    def __init__(self, tmp_path, *, admin=True, g0_used=6000, remote_alias=None,
                 presentmon_frames=True, answers=None, restore_code=0):  # fmt: skip
        self.root = tmp_path / "results"
        self.speech = self.root / "mh1" / "speech"
        write_lowtier_results(self.speech)
        self.material = write_material(tmp_path)
        self.clock = Clock()
        self.ask = Asker({**ANSWERS, **(answers or {})})
        self.said: list[str] = []
        self.servers = FakeServers(self.clock, remote_alias=remote_alias)
        self.runner = FakeRunner(
            lambda out: g0_used if "G0" in out.parts or "presentmon" in out.name else 9000,
            presentmon_frames=presentmon_frames,
        )
        self.bench = FakeBenchHost(self.root, self.clock, restore_code=restore_code)
        self.serve = FakeServe(self.servers)
        self.services = FakeServices()
        self.admin = admin
        self.env = game_suite.GameEnv(
            ask=self.ask, say=self.said.append, clock=self.clock, sleep=self.clock.sleep,
            now=lambda: NOW, runner=self.runner, bench_main=self.bench,
            is_admin=lambda: self.admin, opener=self.servers, serve=self.serve.serve,
            unserve=self.serve.unserve, services_factory=self.services.factory,
        )  # fmt: skip

    def request(self, **overrides):
        values = {
            "results_root": self.root,
            "run": "mh1",
            "model": "gemma4-e2b",
            "material_dir": self.material,
            "speech_models_dir": self.root,
            "presentmon": GAME_FIXTURES / "presentmon.csv",
            "manual_fps": False,
        }
        values.update(overrides)
        return game_suite.GameRequest(**values)

    def run(self, **overrides):
        return game_suite.run_game_session(self.request(**overrides), self.env)


def passes(session_dir: Path) -> list[dict]:
    text = (session_dir / "passes.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


BENCH_RUN = "mh1-game-20261114T193005Z"


# ---------------------------------------------------------------------------
# The bench run id and the refusals
# ---------------------------------------------------------------------------


def test_a_session_makes_its_own_bench_run_id_from_the_utc_time():
    assert game_suite.new_bench_run("mh1", NOW) == BENCH_RUN
    later = game_suite.new_bench_run("mh1", NOW.replace(second=6))
    assert later != BENCH_RUN


def test_a_session_refuses_a_bench_run_whose_session_exists(tmp_path):
    session = Session(tmp_path)
    (session.root / BENCH_RUN / "session").mkdir(parents=True)

    with pytest.raises(core.BenchRunError):
        session.run()

    assert session.bench.calls == []
    assert not (session.speech / "game").exists()


def test_two_sessions_never_share_a_bench_run(tmp_path):
    session = Session(tmp_path)
    session.run()

    with pytest.raises(core.BenchRunError):
        session.run()


def test_presentmon_without_administrator_rights_is_refused_before_anything(tmp_path):
    session = Session(tmp_path, admin=False)

    with pytest.raises(game_suite.GameSessionError, match="administrator"):
        session.run()

    assert session.bench.calls == [] and session.ask.prompts == []
    assert session.runner.started == []
    assert not (session.speech / "game").exists()


def test_manual_figures_need_no_administrator_rights(tmp_path):
    session = Session(tmp_path, admin=False, answers={"average frame rate": "88", "1% low": "61"})

    out = session.run(presentmon=None, manual_fps=True)

    assert {p["frames"]["source"] for p in passes(out)} == {"game_benchmark"}
    assert not any("--output_file" in argv for argv in session.runner.started)


def test_a_session_refuses_material_that_differs_from_its_manifest(tmp_path):
    session = Session(tmp_path)
    (session.material / "U01.wav").write_bytes(b"changed")

    with pytest.raises(core.ManifestError):
        session.run()
    assert session.bench.calls == []


# ---------------------------------------------------------------------------
# The walk: conditions, passes, prompts and captures
# ---------------------------------------------------------------------------


def test_the_session_asks_for_the_process_name_then_stops_mist_under_its_own_id(tmp_path):
    session = Session(tmp_path)

    out = session.run()

    assert "process name" in session.ask.prompts[0]
    commands = [c[0] for c in session.bench.calls]
    assert commands == ["snapshot", "stop", "restore"]
    for call in session.bench.calls:
        assert call[call.index("--run") + 1] == BENCH_RUN
    assert "--with-neo4j" in session.bench.calls[1]
    meta = json.loads((out / "meta.json").read_text())
    assert meta["process"] == "examplegame.exe" and meta["game"] == "Example Game"
    assert meta["version"] == "1.4.2" and meta["bench_run"] == BENCH_RUN
    assert out == session.speech / "game" / BENCH_RUN


def test_run_8_walks_five_conditions_of_three_passes(tmp_path):
    session = Session(tmp_path)

    out = session.run()

    rows = passes(out)
    conditions = [r["condition"] for r in rows]
    assert conditions == (
        ["G0"] * 3 + ["G1-gpu-full"] * 3 + ["G2-gpu-full"] * 3 + ["G1-cpu"] * 3 + ["G2-cpu"] * 3
    )
    assert [r["pass"] for r in rows[:3]] == [1, 2, 3]
    for row in rows:
        assert (out / row["condition"] / f"pass{row['pass']}" / "pass.json").is_file()


def test_each_pass_is_prompted_for_before_its_capture(tmp_path):
    session = Session(tmp_path)

    session.run()

    starts = [p for p in session.ask.prompts if "press Enter to capture" in p]
    assert len(starts) == 15 + 1  # every pass, and the PresentMon test capture first
    assert "capture 10 s" in starts[0]
    assert "G0, pass 1 of 3" in starts[1] and "capture 120 s" in starts[1]


def test_each_pass_captures_120_seconds(tmp_path):
    session = Session(tmp_path)

    out = session.run()

    durations = [r["capture_s"] for r in passes(out)]
    assert all(d >= 120.0 for d in durations)
    presentmon = [a for a in session.runner.started if "--output_file" in a]
    assert {a[a.index("--timed") + 1] for a in presentmon} == {"10", "120"}
    typeperf = [a for a in session.runner.started if a[0] == "typeperf"]
    assert {a[a.index("-sc") + 1] for a in typeperf} == {"120"}


def test_each_pass_records_frames_gpu_and_cpu_and_raj_s_note(tmp_path):
    session = Session(tmp_path, answers={"noticed": "a hitch when the map loaded"})

    out = session.run()

    first = passes(out)[0]
    assert first["frames"]["low_1pct_fps"] == pytest.approx(25.0)
    assert first["gpu"]["peak_used_mib"] == 6000
    assert first["cpu"]["mean"] == 35.0
    assert first["note"] == "a hitch when the map loaded"
    pass_dir = out / "G0" / "pass1"
    for name in ("presentmon.csv", "gpu.csv", "cpu.csv", "pass.json"):
        assert (pass_dir / name).is_file()


def test_g1_loads_the_low_level_idle_and_g2_adds_ten_turns_twelve_seconds_apart(tmp_path):
    session = Session(tmp_path)

    out = session.run()

    assert not (out / "G1-gpu-full" / "pass1" / "chain.jsonl").exists()
    turns = [
        json.loads(line)
        for line in (out / "G2-gpu-full" / "pass1" / "chain.jsonl").read_text().splitlines()
    ]
    assert len(turns) == 10
    assert [t["due_s"] for t in turns] == [12.0 * i for i in range(10)]
    assert all(t["late_s"] >= 0.0 for t in turns)
    assert {t["alias"] for t in turns} == {"gemma4-e2b@gpu-full"}
    assert all(t["turn_ms"] is not None for t in turns)


def test_the_low_level_is_served_and_its_services_started_per_placement(tmp_path):
    session = Session(tmp_path)

    session.run()

    assert [(m, p, n) for m, p, n, _ in session.serve.served] == [
        ("gemma4-e2b", "gpu-full", None),
        ("gemma4-e2b", "cpu", None),
    ]
    assert session.serve.served[0][3].name == f"gpu-full-{BENCH_RUN}"
    assert len(session.serve.unserved) == 2
    assert session.services.events == [
        "start", "stop services-gpu-full", "start", "stop services-cpu",
    ]  # fmt: skip


def test_the_4070_slice_follows_g0_s_peak_memory(tmp_path):
    session = Session(tmp_path, g0_used=9000)

    out = session.run()

    assert [p for _, p, _, _ in session.serve.served] == ["gpu-2g", "cpu"]
    assert session.serve.served[0][2] == 12
    chosen = json.loads((out / "slice.json").read_text())
    assert chosen["chosen"] == "gpu-2g" and chosen["g0_peak_mib"] == 9000
    assert chosen["margin_mib"] == 512


def test_no_slice_fits_and_the_4070_conditions_are_skipped(tmp_path):
    session = Session(tmp_path, g0_used=11000)

    out = session.run()

    conditions = {r["condition"] for r in passes(out)}
    assert conditions == {"G0", "G1-cpu", "G2-cpu"}
    status = [json.loads(line) for line in (out / "conditions.jsonl").read_text().splitlines()]
    assert any(s["status"] == "skipped" and s["placement"] == "gpu" for s in status)


def test_a_placement_raj_drops_is_not_run(tmp_path):
    session = Session(tmp_path)

    out = session.run(skip_placements=("cpu",))

    assert {r["condition"] for r in passes(out)} == {"G0", "G1-gpu-full", "G2-gpu-full"}


# ---------------------------------------------------------------------------
# --with-1070
# ---------------------------------------------------------------------------


def test_the_1070_conditions_run_first_only_with_the_flag(tmp_path):
    alias = "gemma4-e2b@gtx1070-shared-game"
    session = Session(tmp_path, remote_alias=alias)

    out = session.run(with_1070=True)

    rows = passes(out)
    first_six = ["G1-gtx1070-shared"] * 3 + ["G2-gtx1070-shared"] * 3
    assert [r["condition"] for r in rows[:6]] == first_six
    assert len(rows) == 21
    turns = (out / "G2-gtx1070-shared" / "pass1" / "chain.jsonl").read_text().splitlines()
    assert {json.loads(t)["alias"] for t in turns} == {alias}
    assert any(u.startswith(game_suite.DEFAULT_REMOTE_URL) for u in session.servers.chat_urls)


def test_the_1070_conditions_are_skipped_when_the_alias_is_not_served(tmp_path):
    session = Session(tmp_path, remote_alias=None)

    out = session.run(with_1070=True)

    assert not any(r["condition"].endswith("gtx1070-shared") for r in passes(out))
    status = [json.loads(line) for line in (out / "conditions.jsonl").read_text().splitlines()]
    skipped = [s for s in status if s["placement"] == "gtx1070-shared"]
    assert skipped and all(s["status"] == "skipped" for s in skipped)


def test_without_the_flag_host_1070_is_never_asked(tmp_path):
    session = Session(tmp_path, remote_alias="gemma4-e2b@gtx1070-shared-game")

    session.run()

    assert not any("8090" in u for u in session.servers.chat_urls)


# ---------------------------------------------------------------------------
# Restore and E4B's return
# ---------------------------------------------------------------------------


def test_mist_is_restored_and_e4b_s_return_is_timed(tmp_path):
    session = Session(tmp_path)

    out = session.run()

    back = json.loads((out / "e4b_return.json").read_text())
    # restore takes 30 s on the fake, then E4B's first warm-up token arrives 2.5 s later.
    assert back["e4b_return_s"] == pytest.approx(32.5)
    assert back["restore_exit"] == 0 and session.servers.e4b_requests == 1


def test_restore_runs_even_when_a_pass_fails(tmp_path):
    session = Session(tmp_path)

    def broken(*args, **kwargs):
        raise core.ServeFailedError("small-serve did not come up")

    session.env = game_suite.GameEnv(**{**session.env.__dict__, "serve": broken})

    with pytest.raises(core.ServeFailedError):
        session.run()

    assert [c[0] for c in session.bench.calls] == ["snapshot", "stop", "restore"]
    assert session.services.events[-1].startswith("stop")


def test_a_failed_restore_is_recorded_and_reported(tmp_path):
    session = Session(tmp_path, restore_code=1)

    with pytest.raises(game_suite.GameSessionError, match="restore"):
        session.run()

    back = json.loads((session.speech / "game" / BENCH_RUN / "e4b_return.json").read_text())
    assert back["restore_exit"] == 1


# ---------------------------------------------------------------------------
# Frames without PresentMon, and run 9
# ---------------------------------------------------------------------------


def test_a_test_capture_with_no_frames_falls_back_to_a_typed_frame_rate(tmp_path):
    session = Session(
        tmp_path, presentmon_frames=False, answers={"frame rate the game shows": "64"}
    )

    out = session.run()

    rows = passes(out)
    assert {r["frames"]["source"] for r in rows} == {"typed"}
    assert rows[0]["frames"]["avg_fps"] == 64.0
    meta = json.loads((out / "meta.json").read_text())
    assert meta["frames_mode"] == "typed"


def test_server_process_adds_its_counters_and_the_tick_reading(tmp_path):
    session = Session(tmp_path, answers={"tick": "mean tick 18.2 ms"})

    out = session.run(server_process="java")

    rows = passes(out)
    assert rows[0]["tick"] == "mean tick 18.2 ms"
    assert rows[0]["server"]["cpu"]["max"] == 210.0
    assert rows[0]["server"]["memory_max_mib"] == pytest.approx(2248.0)
    typeperf = [a for a in session.runner.started if a[0] == "typeperf"]
    assert all(r"\Process(java)\% Processor Time" in a for a in typeperf)


def test_without_server_process_no_tick_reading_is_asked(tmp_path):
    session = Session(tmp_path)

    out = session.run()

    assert not any("tick" in p for p in session.ask.prompts)
    assert passes(out)[0]["tick"] is None


# ---------------------------------------------------------------------------
# The CLI and the registry contract
# ---------------------------------------------------------------------------


def test_game_session_is_registered_with_its_options():
    parser = speech_bench.subcommand_parsers(speech_bench.build_parser())["game-session"]
    options = {o for a in parser._actions for o in a.option_strings}

    expected = {"--presentmon", "--manual-fps", "--with-1070", "--server-process", "--model"}
    assert expected <= options


def test_game_session_needs_presentmon_or_manual_figures(tmp_path):
    with pytest.raises(SystemExit):
        speech_bench.main(
            ["game-session", "--results-root", str(tmp_path), "--run", "mh1", "--model", "x"]
        )


def test_game_validate_inputs_finds_no_problem_in_the_committed_files():
    assert game_suite.validate_inputs(_REPO_ROOT) == []


def test_game_selftest_passes():
    game_suite.selftest()
