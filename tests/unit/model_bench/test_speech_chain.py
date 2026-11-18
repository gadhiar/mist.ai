"""Run 7's chained low-level turn: `chain.py` and the two loopback services it starts.

The Parakeet and Pocket TTS services are tested in-process on 127.0.0.1 with a fake transcriber
and a fake engine; the chain runner is tested against fake services (an opener that answers
`/transcribe`, `/speak`, `/v1/models` and streamed `/v1/chat/completions` in their shapes) with a
scripted clock that only the fakes move, so every stage time is exact. Docker is a fake too. No
test touches a GPU, the network or an engine package.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import sys
import threading
import urllib.error
import urllib.request
import wave
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402

from scripts.model_bench import speech_bench  # noqa: E402
from scripts.model_bench.speech import chain, core, lowtier_suite, toolcall  # noqa: E402
from scripts.model_bench.speech.probes import load_gen, stt_service, tts_service  # noqa: E402

FIXTURES = _REPO_ROOT / "tests" / "unit" / "model_bench" / "fixtures" / "speech"
UTTERANCES = {
    u["id"]: u["text"]
    for u in json.loads(core.UTTERANCES_PATH.read_text(encoding="utf-8"))["utterances"]
}
VALID_CALLS = {
    case["prompt_id"]: case["calls"]
    for case in (
        json.loads(line)
        for line in (FIXTURES / "lowtier" / "tool_calls.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    )
    if case["valid_all"] and case["calls"] and " valid" in case["case"]
}
DONE_REPLY = "All of that is taken care of and ready for you now, as asked. Anything else?"
SHORT_REPLY = "Octopuses have three hearts and blue blood, which carries copper. They also taste with their arms."


# ---------------------------------------------------------------------------
# Fakes: a clock only the services move, and the services themselves
# ---------------------------------------------------------------------------


class Clock:
    """A clock that stands still unless a fake service or `sleep` moves it."""

    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds

    def sleep(self, seconds: float) -> None:
        self.t += max(0.0, seconds)


class JsonResponse:
    def __init__(self, doc: dict, clock: Clock, delay: float, status: int = 200):
        self.status = status
        self.headers: dict[str, str] = {}
        self._body = json.dumps(doc).encode("utf-8")
        self._clock = clock
        self._delay = delay

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def read(self, *_):
        self._clock.advance(self._delay)
        body, self._body = self._body, b""
        return body


class SseResponse:
    """Yields each `data:` line after moving the clock by that line's delay."""

    def __init__(self, events: list[tuple[float, str]], clock: Clock):
        self.status = 200
        self.headers: dict[str, str] = {}
        self._events = events
        self._clock = clock

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def __iter__(self):
        for delay, line in self._events:
            self._clock.advance(delay)
            yield line.encode("utf-8")


class PcmResponse:
    """A chunked PCM reply: each `read1` moves the clock by that chunk's delay."""

    def __init__(self, chunks: list[tuple[float, bytes]], clock: Clock, sample_rate: int = 24000):
        self.status = 200
        self.headers = {"X-Sample-Rate": str(sample_rate)}
        self._chunks = list(chunks)
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


def sse(event: dict | str) -> str:
    return "data: " + (event if isinstance(event, str) else json.dumps(event)) + "\n"


def content(text: str) -> dict:
    return {"choices": [{"index": 0, "delta": {"content": text}}]}


def tool_delta(index: int, call_id: str, name: str, arguments: str) -> dict:
    call = {"index": index, "id": call_id, "type": "function"}
    call["function"] = {"name": name, "arguments": arguments}
    return {"choices": [{"index": 0, "delta": {"tool_calls": [call]}}]}


def finish(reason: str, prompt_ms: float = 40.0) -> dict:
    timings = {"prompt_n": 30, "prompt_ms": prompt_ms, "predicted_n": 20, "predicted_ms": 200.0}
    timings["predicted_per_second"] = 100.0
    return {"choices": [{"index": 0, "delta": {}, "finish_reason": reason}], "timings": timings}


ROLE = {"choices": [{"index": 0, "delta": {"role": "assistant", "content": None}}]}


class FakeServices:
    """Answers the chain's four routes; the LLM's replies are scripted per request."""

    def __init__(self, clock: Clock, *, alias: str = "gemma4-e2b@gpu-full"):
        self.clock = clock
        self.alias = alias
        self.requests: list[tuple[str, object]] = []
        self.chat_scripts: list[list[tuple[float, str]]] = []
        self.transcripts: list[str] = []
        self.stt_delay = 0.150
        self.tts_chunks: list[tuple[float, bytes]] = [
            (0.180, b"\x01\x00" * 480),
            (0.050, b"\x02\x00" * 480),
        ]
        self.spoken: list[str] = []
        self.chat_router = None

    def __call__(self, request, timeout=None):
        url = request.full_url if isinstance(request, urllib.request.Request) else str(request)
        data = request.data if isinstance(request, urllib.request.Request) else None
        if url.endswith("/v1/models"):
            return JsonResponse({"data": [{"id": self.alias}]}, self.clock, 0.0)
        if url.endswith("/transcribe"):
            self.requests.append(("transcribe", data))
            text = self.transcripts.pop(0)
            return JsonResponse({"text": text, "server_ms": 120.0}, self.clock, self.stt_delay)
        if url.endswith("/speak"):
            body = json.loads(data.decode("utf-8"))
            self.requests.append(("speak", body))
            self.spoken.append(body["text"])
            return PcmResponse(list(self.tts_chunks), self.clock)
        if url.endswith("/v1/chat/completions"):
            body = json.loads(data.decode("utf-8"))
            self.requests.append(("chat", body))
            script = self.chat_router(body) if self.chat_router else self.chat_scripts.pop(0)
            return SseResponse(script, self.clock)
        raise AssertionError(f"unexpected request to {url}")


def make_clients(fake: FakeServices, clock: Clock, base_url: str = core.SMALL_BASE_URL):
    llm = chain.LlmClient(base_url, opener=fake, clock=clock)
    stt = chain.SttClient(chain.STT_URL, opener=fake, clock=clock)
    tts = chain.TtsClient(chain.TTS_URL, opener=fake, clock=clock)
    return llm, stt, tts


# ---------------------------------------------------------------------------
# Material: synthetic WAVs written here, with a manifest whose text is the script
# ---------------------------------------------------------------------------


def write_wav(path: Path, n_samples: int = 1600, value: int = 1000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as clip:
        clip.setnchannels(1)
        clip.setsampwidth(2)
        clip.setframerate(16000)
        clip.writeframes(int(value).to_bytes(2, "little", signed=True) * n_samples)
    path.write_bytes(buffer.getvalue())
    return buffer.getvalue()


def make_material(root: Path, ids=chain.CHAIN_UTTERANCES) -> Path:
    material = root / "material"
    material.mkdir(parents=True)
    items = []
    for index, uid in enumerate(ids):
        data = write_wav(material / f"{uid}.wav", value=100 + index)
        items.append(
            {
                "id": uid,
                "file": f"{uid}.wav",
                "kind": "complete",
                "text": UTTERANCES[uid],
                "duration_s": 0.1,
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    (material / "manifest.json").write_text(json.dumps({"items": items}), encoding="utf-8")
    return material


@pytest.fixture
def material(tmp_path):
    return make_material(tmp_path)


@pytest.fixture
def inputs(material):
    return chain.load_chain_inputs(material)


# ---------------------------------------------------------------------------
# The utterances, their prompts and their labels
# ---------------------------------------------------------------------------


def test_the_chain_runs_raj_s_fourteen_recordings_as_run_7_lists_them():
    commands = ("U01", "U03", "U04", "U06", "U08", "U09", "U11", "U12", "U16", "U18")

    assert (*commands, "U19", "U20", "U07", "U15") == chain.CHAIN_UTTERANCES


def test_each_recording_is_scored_with_the_tool_prompt_of_the_same_words(inputs):
    by_utt = {uid: item.prompt.id for uid, item in inputs.items.items()}

    assert by_utt["U01"] == "C01"
    assert by_utt["U18"] == "C10"
    assert by_utt["U19"] == "N02" and by_utt["U20"] == "N01"
    assert by_utt["U07"] == "E02" and by_utt["U15"] == "E01"


def test_each_turn_is_labelled_command_short_or_hand_off(inputs):
    kinds = {uid: item.kind for uid, item in inputs.items.items()}

    assert {kinds[u] for u in chain.CHAIN_UTTERANCES[:10]} == {"command"}
    assert (kinds["U19"], kinds["U20"]) == ("short", "short")
    assert (kinds["U07"], kinds["U15"]) == ("hand_off", "hand_off")


def test_the_chain_sends_the_low_level_prompt_and_the_tool_set(inputs):
    assert inputs.system_prompt == lowtier_suite.load_lowtier_prompt()
    assert inputs.tools == toolcall.load_tools()


def test_chain_inputs_refuse_material_that_differs_from_its_manifest(material):
    (material / "U01.wav").write_bytes(b"changed")

    with pytest.raises(core.ManifestError):
        chain.load_chain_inputs(material)


def test_chain_inputs_refuse_a_manifest_without_a_chain_recording(tmp_path):
    material = make_material(tmp_path, ids=chain.CHAIN_UTTERANCES[:-1])

    with pytest.raises(core.ManifestError, match="U15"):
        chain.load_chain_inputs(material)


def test_chain_validate_inputs_finds_no_problem_in_the_committed_files():
    assert chain.validate_inputs(_REPO_ROOT) == []


# ---------------------------------------------------------------------------
# One turn against fake services and a scripted clock
# ---------------------------------------------------------------------------


def command_script() -> list[tuple[float, str]]:
    return [
        (0.080, sse(ROLE)),
        (0.020, sse(tool_delta(0, "call_1", "set_timer", '{"duration_seconds": '))),
        (0.010, sse(tool_delta(0, "", "", "600}"))),
        (0.010, sse(finish("tool_calls"))),
        (0.0, sse("[DONE]")),
    ]


def done_script() -> list[tuple[float, str]]:
    return [
        (0.090, sse(content("Your ten minute timer is set and running now, "))),
        (0.030, sse(content("and it rings at ten past nine."))),
        (0.010, sse(finish("stop"))),
        (0.0, sse("[DONE]")),
    ]


def test_a_command_turn_s_stage_times_add_up_to_the_turn(inputs):
    clock = Clock()
    fake = FakeServices(clock)
    fake.transcripts = ["Set a timer for ten minutes."]
    fake.chat_scripts = [command_script(), done_script()]
    llm, stt, tts = make_clients(fake, clock)

    row = chain.run_turn(
        inputs.items["U01"], rep=1, alias="gemma4-e2b@gpu-full", inputs=inputs,
        llm=llm, stt=stt, tts=tts, clock=clock,
    )  # fmt: skip

    assert row["kind"] == "command"
    assert row["stt_ms"] == pytest.approx(150.0)
    assert row["stt_server_ms"] == 120.0
    assert row["llm_first_token_ms"] == pytest.approx(100.0)
    assert row["tool_call_ms"] == pytest.approx(110.0)
    assert row["done_first_token_ms"] == pytest.approx(90.0)
    # The first request runs 120 ms to its end; the follow-up's first clause lands 90 ms later.
    assert row["first_clause_ms"] == pytest.approx(210.0)
    assert row["spoken_text"] == "Your ten minute timer is set and running now,"
    assert row["tts_first_audio_ms"] == pytest.approx(180.0)
    assert row["turn_ms"] == pytest.approx(row["stt_ms"] + row["first_clause_ms"] + 180.0)
    assert row["turn_ms"] == pytest.approx(540.0)
    # The rest of the follow-up reply (40 ms) is off the critical path but inside the wall time.
    assert row["turn_wall_ms"] == pytest.approx(580.0)
    assert row["tool_valid"] is True and row["tool_valid_first"] is True
    assert row["transcript_wer"] == 0.0
    assert row["error"] is None


def test_a_tool_call_gets_the_fixed_done_result_back(inputs):
    clock = Clock()
    fake = FakeServices(clock)
    fake.transcripts = ["Set a timer for ten minutes."]
    fake.chat_scripts = [command_script(), done_script()]
    llm, stt, tts = make_clients(fake, clock)

    chain.run_turn(
        inputs.items["U01"], rep=1, alias="a@b", inputs=inputs, llm=llm, stt=stt, tts=tts,
        clock=clock,
    )  # fmt: skip

    first, follow_up = (body for kind, body in fake.requests if kind == "chat")
    assert first["messages"][0] == {"role": "system", "content": inputs.system_prompt}
    assert first["messages"][1] == {"role": "user", "content": "Set a timer for ten minutes."}
    assert first["tools"] == inputs.tools and first["model"] == "a@b"
    assistant, tool_result = follow_up["messages"][2:]
    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"][0]["function"] == {
        "name": "set_timer",
        "arguments": '{"duration_seconds": 600}',
    }
    assert tool_result == {
        "role": "tool",
        "tool_call_id": "call_1",
        "name": "set_timer",
        "content": chain.DONE_RESULT,
    }
    assert chain.DONE_RESULT == "done"
    assert fake.spoken == ["Your ten minute timer is set and running now,"]


def test_a_hand_off_speaks_one_moment_without_a_second_request(inputs):
    clock = Clock()
    fake = FakeServices(clock)
    fake.transcripts = ["What did we decide about the database migration last week?"]
    arguments = json.dumps({"request": UTTERANCES["U07"], "reason": "memory"})
    fake.chat_scripts = [
        [
            (0.070, sse(tool_delta(0, "call_h", "hand_off", arguments[:20]))),
            (0.030, sse(tool_delta(0, "", "", arguments[20:]))),
            (0.010, sse(finish("tool_calls"))),
            (0.0, sse("[DONE]")),
        ]
    ]
    llm, stt, tts = make_clients(fake, clock)

    row = chain.run_turn(
        inputs.items["U07"], rep=2, alias="a@b", inputs=inputs, llm=llm, stt=stt, tts=tts,
        clock=clock,
    )  # fmt: skip

    assert fake.spoken == ["One moment."] and chain.HAND_OFF_ACK == "One moment."
    assert len([k for k, _ in fake.requests if k == "chat"]) == 1
    assert row["kind"] == "hand_off" and row["rep"] == 2
    assert row["first_clause_ms"] == pytest.approx(row["tool_call_ms"]) == pytest.approx(100.0)
    assert row["turn_ms"] == pytest.approx(150.0 + 100.0 + 180.0)
    assert row["tool_valid"] is True
    assert row["done_first_token_ms"] is None


def test_a_short_answer_speaks_the_reply_s_first_clause(inputs):
    clock = Clock()
    fake = FakeServices(clock)
    fake.transcripts = ["Tell me something interesting about octopuses."]
    fake.chat_scripts = [
        [
            (0.120, sse(content(SHORT_REPLY[:20]))),
            (0.015, sse(content(SHORT_REPLY[20:44]))),
            (0.015, sse(content(SHORT_REPLY[44:]))),
            (0.010, sse(finish("stop"))),
            (0.0, sse("[DONE]")),
        ]
    ]
    llm, stt, tts = make_clients(fake, clock)

    row = chain.run_turn(
        inputs.items["U20"], rep=1, alias="a@b", inputs=inputs, llm=llm, stt=stt, tts=tts,
        clock=clock,
    )  # fmt: skip

    assert row["kind"] == "short"
    assert fake.spoken == ["Octopuses have three hearts and blue blood,"]
    # The first clause (43 characters) is complete with the second piece, 135 ms in.
    assert row["first_clause_ms"] == pytest.approx(135.0)
    assert row["llm_first_token_ms"] == pytest.approx(120.0)
    assert row["tool_call_ms"] is None
    assert row["turn_ms"] == pytest.approx(150.0 + 135.0 + 180.0)
    assert row["tool_valid"] is True and row["tool_n_calls"] == 0


def test_a_wrong_tool_call_scores_invalid_and_the_turn_still_completes(inputs):
    clock = Clock()
    fake = FakeServices(clock)
    fake.transcripts = ["Set a timer for ten minutes."]
    fake.chat_scripts = [
        [
            (0.050, sse(tool_delta(0, "c", "set_alarm", '{"time": "21:10"}'))),
            (0.010, sse(finish("tool_calls"))),
            (0.0, sse("[DONE]")),
        ],
        done_script(),
    ]
    llm, stt, tts = make_clients(fake, clock)

    row = chain.run_turn(
        inputs.items["U01"], rep=1, alias="a@b", inputs=inputs, llm=llm, stt=stt, tts=tts,
        clock=clock,
    )  # fmt: skip

    assert row["tool_valid"] is False
    assert row["turn_ms"] is not None


def test_a_failed_stage_is_recorded_on_the_row_not_raised(inputs):
    clock = Clock()

    def broken(request, timeout=None):
        raise urllib.error.URLError("service down")

    llm, stt, tts = make_clients(FakeServices(clock), clock)
    stt = chain.SttClient(chain.STT_URL, opener=broken, clock=clock)

    row = chain.run_turn(
        inputs.items["U01"], rep=1, alias="a@b", inputs=inputs, llm=llm, stt=stt, tts=tts,
        clock=clock,
    )  # fmt: skip

    assert "service down" in row["error"]
    assert row["turn_ms"] is None


# ---------------------------------------------------------------------------
# The service argv
# ---------------------------------------------------------------------------


def _pairs(argv, flag):
    return [argv[i + 1] for i, a in enumerate(argv[:-1]) if a == flag]


@pytest.mark.parametrize("which", ["stt", "tts"])
def test_each_service_container_publishes_on_loopback_only_with_four_threads(which):
    build = chain.stt_service_argv if which == "stt" else chain.tts_service_argv
    argv = build(repo_dir="/repo", speech_models_dir="/models")
    port = chain.STT_PORT if which == "stt" else chain.TTS_PORT

    assert _pairs(argv, "-p") == [f"127.0.0.1:{port}:{port}"]
    assert "--gpus" not in argv
    assert _pairs(argv, "--cpus") == ["4"]
    assert "OMP_NUM_THREADS=4" in _pairs(argv, "-e")
    assert "HF_HUB_OFFLINE=1" in _pairs(argv, "-e")
    assert all(m.endswith(":ro") for m in _pairs(argv, "-v"))
    assert "/repo:/work:ro" in _pairs(argv, "-v")
    assert "--rm" not in argv


def test_the_stt_service_runs_parakeet_on_the_cpu_in_the_asr_image():
    argv = chain.stt_service_argv(repo_dir="/repo", speech_models_dir="/models")

    assert "mist-speech-bench-asr" in argv
    assert argv[argv.index("-m") + 1] == "scripts.model_bench.speech.probes.stt_service"
    assert _pairs(argv, "--engine") == ["parakeet-cpu"]
    assert _pairs(argv, "--threads") == ["4"]


def test_the_tts_service_runs_pocket_in_its_image_and_names_no_voice_file():
    argv = chain.tts_service_argv(repo_dir="/repo", speech_models_dir="/models")

    assert "mist-speech-bench-pocket" in argv
    assert argv[argv.index("-m") + 1] == "scripts.model_bench.speech.probes.tts_service"
    assert not any(a.endswith((".wav", ".mp3", ".flac")) for a in argv)


def test_chain_plan_argv_lists_both_services():
    ctx = core.PlanContext(cuda_llama_image="cuda")
    labels = [label for label, _ in chain.container_argv(ctx)]

    assert labels == ["chain stt-service", "chain tts-service"]


# ---------------------------------------------------------------------------
# Choosing the engines, and the cloned voice
# ---------------------------------------------------------------------------

# The argv the base recorded for both services; a run without `--stt-engine` or `--tts-engine`
# must start exactly these.
BASE_STT_ARGV = [
    "docker", "run", "-d", "--name", "mist-bench-chain-stt", "-p", "127.0.0.1:8091:8091",
    "--cpus", "4", "-e", "HF_HUB_OFFLINE=1", "-e", "OMP_NUM_THREADS=4", "-e", "MKL_NUM_THREADS=4",
    "-e", "OPENBLAS_NUM_THREADS=4", "-e", "PYTHONDONTWRITEBYTECODE=1", "-v", "/repo:/work:ro",
    "-v", "/models:/weights:ro", "-w", "/work", "mist-speech-bench-asr", "python", "-m",
    "scripts.model_bench.speech.probes.stt_service", "--engine", "parakeet-cpu", "--weights",
    "/weights", "--threads", "4", "--host", "0.0.0.0", "--port", "8091",
]  # fmt: skip
BASE_TTS_ARGV = [
    "docker", "run", "-d", "--name", "mist-bench-chain-tts", "-p", "127.0.0.1:8092:8092",
    "--cpus", "4", "-e", "HF_HUB_OFFLINE=1", "-e", "OMP_NUM_THREADS=4", "-e", "MKL_NUM_THREADS=4",
    "-e", "OPENBLAS_NUM_THREADS=4", "-e", "PYTHONDONTWRITEBYTECODE=1", "-v", "/repo:/work:ro",
    "-v", "/models:/models/speech:ro", "-w", "/work", "mist-speech-bench-pocket", "python", "-m",
    "scripts.model_bench.speech.probes.tts_service", "--models-dir", "/models/speech", "--host",
    "0.0.0.0", "--port", "8092",
]  # fmt: skip
CLIP_HOST = "/private/identity/clip.wav"
CLIP_MOUNT = f"{CLIP_HOST}:/voice/identity.wav:ro"
TRANSCRIPT = "the quartz heron counted nine amber lanterns"
ARMS = core.load_json(core.SPEECH_ARMS_PATH)


def test_the_chain_offers_two_stt_engines_and_four_tts_engines():
    assert chain.STT_ENGINES == ("parakeet-cpu", "nemotron-160")
    assert chain.TTS_ENGINES == ("pocket", "pocket-clone", "qwen3", "qwen3-design")
    assert (chain.STT_ENGINE, chain.TTS_ENGINE) == ("parakeet-cpu", "pocket")


def test_with_neither_engine_chosen_the_service_argv_are_the_base_s():
    paths = {"repo_dir": "/repo", "speech_models_dir": "/models"}

    assert chain.stt_service_argv(**paths) == BASE_STT_ARGV
    assert chain.tts_service_argv(**paths) == BASE_TTS_ARGV
    assert chain.stt_service_argv(**paths, engine="parakeet-cpu") == BASE_STT_ARGV
    assert chain.tts_service_argv(**paths, engine="pocket") == BASE_TTS_ARGV


def _check_device(argv, cfg):
    if cfg["device"] == "cuda":
        assert _pairs(argv, "--gpus") == ["all"]
        assert "--cpus" not in argv
    else:
        assert "--gpus" not in argv
        assert _pairs(argv, "--cpus") == ["4"]
        assert "OMP_NUM_THREADS=4" in _pairs(argv, "-e")


@pytest.mark.parametrize("engine", ["parakeet-cpu", "nemotron-160"])
def test_the_stt_service_runs_the_chosen_engine_in_its_image_on_its_device(engine):
    cfg = ARMS["stt"][engine]
    argv = chain.stt_service_argv(repo_dir="/repo", speech_models_dir="/models", engine=engine)

    assert _pairs(argv, "--engine") == [engine]
    assert cfg["image"] in argv
    assert _pairs(argv, "-p") == ["127.0.0.1:8091:8091"]
    assert all(m.endswith(":ro") for m in _pairs(argv, "-v"))
    _check_device(argv, cfg)


@pytest.mark.parametrize("engine", ["pocket", "pocket-clone", "qwen3", "qwen3-design"])
def test_the_tts_service_runs_the_chosen_engine_in_its_image_on_its_device(engine):
    cfg = ARMS["tts"][engine]
    clip = CLIP_HOST if engine == "pocket-clone" else None
    argv = chain.tts_service_argv(
        repo_dir="/repo", speech_models_dir="/models", engine=engine, voice_clip=clip
    )

    assert _pairs(argv, "--engine") == ([] if engine == "pocket" else [engine])
    assert cfg["image"] in argv
    assert _pairs(argv, "-p") == ["127.0.0.1:8092:8092"]
    assert all(m.endswith(":ro") for m in _pairs(argv, "-v"))
    _check_device(argv, cfg)
    clip_mounts = [m for m in _pairs(argv, "-v") if "/voice/" in m]
    assert clip_mounts == ([CLIP_MOUNT] if engine == "pocket-clone" else [])


@pytest.mark.parametrize(("engine", "clip"), [("pocket-clone", None), ("pocket", CLIP_HOST)])
def test_the_tts_service_refuses_a_clone_without_a_clip_and_a_clip_without_the_clone(engine, clip):
    with pytest.raises(core.VoiceClipError):
        chain.tts_service_argv(
            repo_dir="/repo", speech_models_dir="/models", engine=engine, voice_clip=clip
        )


def test_the_services_start_the_chosen_engines_with_the_clip_mounted():
    clock = Clock()
    services = chain.ChainServices.default(
        repo_dir="/repo", speech_models_dir="/models", docker=FakeDocker(),
        opener=HealthOpener(clock), clock=clock, sleep=clock.sleep,
        stt_engine="nemotron-160", tts_engine="pocket-clone", voice_clip=CLIP_HOST,
    )  # fmt: skip

    stt, tts = (list(spec.argv) for spec in services.specs)
    assert _pairs(stt, "--engine") == ["nemotron-160"]
    assert _pairs(tts, "--engine") == ["pocket-clone"]
    assert CLIP_MOUNT in _pairs(tts, "-v")


def write_identity_clip(root: Path) -> Path:
    """A sine tone standing in for the identity clip, with a transcript beside it."""
    folder = root / "private-identity"
    folder.mkdir(parents=True)
    clip = folder / "identity.wav"
    samples = (round(8000 * math.sin(2 * math.pi * 220 * i / 24000)) for i in range(2400))
    with wave.open(str(clip), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(24000)
        out.writeframes(b"".join(s.to_bytes(2, "little", signed=True) for s in samples))
    (folder / "clip.txt").write_text(TRANSCRIPT, encoding="utf-8")
    return clip


def one_rep_run(tmp_path, material, **overrides):
    clock = Clock()
    fake = fake_for_run(clock)
    fake.transcripts = [UTTERANCES[u] for u in chain.CHAIN_UTTERANCES]
    llm, stt, tts = make_clients(fake, clock)
    services = FakeServiceLifecycle()
    out = chain.run_chain(
        chain_request(tmp_path, material, reps=1, **overrides), llm=llm, stt=stt, tts=tts,
        services=services, container_state=lambda name: "running", clock=clock, git_sha="abc",
    )  # fmt: skip
    rows = [json.loads(line) for line in (out / "turns.jsonl").read_text().splitlines()]
    return out, rows, json.loads((out / "chain.json").read_text())


def test_without_a_choice_each_row_names_parakeet_and_pocket(tmp_path, material):
    _, rows, meta = one_rep_run(tmp_path, material)

    assert {(r["stt_engine"], r["tts_engine"]) for r in rows} == {("parakeet-cpu", "pocket")}
    assert (meta["stt_engine"], meta["tts_engine"], meta["voice"]) == (
        "parakeet-cpu",
        "pocket",
        "alba",
    )
    assert "voice_clip_sha256" not in meta


@pytest.mark.parametrize(
    ("stt_engine", "tts_engine"), [("nemotron-160", "qwen3"), ("parakeet-cpu", "qwen3-design")]
)
def test_each_row_and_the_meta_name_both_chosen_engines(tmp_path, material, stt_engine, tts_engine):
    _, rows, meta = one_rep_run(tmp_path, material, stt_engine=stt_engine, tts_engine=tts_engine)

    assert len(rows) == 14
    assert {(r["stt_engine"], r["tts_engine"]) for r in rows} == {(stt_engine, tts_engine)}
    assert (meta["stt_engine"], meta["tts_engine"]) == (stt_engine, tts_engine)
    assert meta["voice"] == ARMS["tts"][tts_engine]["timing_voice"]


def test_a_cloned_chain_records_identity_and_the_clip_s_sha_and_never_its_path_or_words(
    tmp_path, material, capsys, caplog
):
    clip = write_identity_clip(tmp_path)
    caplog.set_level("DEBUG")

    out, rows, meta = one_rep_run(tmp_path, material, tts_engine="pocket-clone", voice_clip=clip)

    assert {r["tts_engine"] for r in rows} == {"pocket-clone"}
    assert meta["voice"] == "identity"
    assert meta["voice_clip_sha256"] == hashlib.sha256(clip.read_bytes()).hexdigest()[:12]
    captured = capsys.readouterr()
    texts = [p.read_text(encoding="utf-8") for p in out.rglob("*") if p.is_file()]
    texts += [captured.out, captured.err, caplog.text]
    for text in texts:
        for secret in (str(clip), clip.as_posix(), clip.parent.name, TRANSCRIPT):
            assert secret not in text
    assert sorted(p.name for p in clip.parent.iterdir()) == ["clip.txt", "identity.wav"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"tts_engine": "pocket-clone"},
        {"tts_engine": "pocket", "voice_clip": "CLIP"},
        {"tts_engine": "pocket-clone", "voice_clip": "MISSING"},
    ],
    ids=["clone-without-clip", "clip-without-clone", "missing-clip"],
)
def test_run_chain_refuses_a_clip_misuse_before_anything(tmp_path, material, overrides):
    clip = write_identity_clip(tmp_path)
    values = dict(overrides)
    if values.get("voice_clip") == "CLIP":
        values["voice_clip"] = clip
    elif values.get("voice_clip") == "MISSING":
        values["voice_clip"] = clip.parent / "absent.wav"
    clock = Clock()
    llm, stt, tts = make_clients(fake_for_run(clock), clock)
    services = FakeServiceLifecycle()

    with pytest.raises(core.VoiceClipError) as caught:
        chain.run_chain(
            chain_request(tmp_path, material, **values), llm=llm, stt=stt, tts=tts,
            services=services, container_state=lambda name: "running", clock=clock,
        )  # fmt: skip

    assert str(clip.parent) not in str(caught.value)
    assert services.events == []
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize(
    "overrides", [{"stt_engine": "whisper-small"}, {"tts_engine": "chatterbox"}]
)
def test_run_chain_refuses_an_engine_the_chain_does_not_offer(tmp_path, material, overrides):
    clock = Clock()
    llm, stt, tts = make_clients(fake_for_run(clock), clock)
    services = FakeServiceLifecycle()

    with pytest.raises(core.SpeechBenchError, match="engine"):
        chain.run_chain(
            chain_request(tmp_path, material, **overrides), llm=llm, stt=stt, tts=tts,
            services=services, container_state=lambda name: "running", clock=clock,
        )  # fmt: skip

    assert services.events == []
    assert not (tmp_path / "results").exists()


def test_the_chain_cli_offers_the_engines_and_the_voice_clip():
    parser = speech_bench.subcommand_parsers(speech_bench.build_parser())["chain"]
    actions = {a.dest: a for a in parser._actions}

    assert tuple(actions["stt_engine"].choices) == chain.STT_ENGINES
    assert tuple(actions["tts_engine"].choices) == chain.TTS_ENGINES
    assert (actions["stt_engine"].default, actions["tts_engine"].default) == (
        "parakeet-cpu",
        "pocket",
    )
    assert actions["voice_clip"].default is None


def test_cmd_chain_hands_the_checked_clip_to_the_services_and_the_request(
    tmp_path, monkeypatch, capsys
):
    clip = write_identity_clip(tmp_path)
    seen = {}

    def services(**kwargs):
        seen["services"] = kwargs
        return FakeServiceLifecycle()

    def run_chain(request, **kwargs):
        seen["request"] = request
        out = tmp_path / "out"
        out.mkdir()
        (out / "turns.jsonl").write_text("", encoding="utf-8")
        return out

    monkeypatch.setattr(chain.ChainServices, "default", services)
    monkeypatch.setattr(chain, "run_chain", run_chain)
    argv = [
        "chain", "--results-root", str(tmp_path / "results"), "--run", "mh1",
        "--model", "gemma4-e2b", "--placement", "gpu-full", "--material-dir", str(tmp_path),
        "--speech-models-dir", str(tmp_path), "--stt-engine", "nemotron-160",
        "--tts-engine", "pocket-clone", "--voice-clip", str(clip),
    ]  # fmt: skip

    code = speech_bench.main(argv)

    assert code == 0
    request = seen["request"]
    assert (request.stt_engine, request.tts_engine) == ("nemotron-160", "pocket-clone")
    assert request.voice_clip == clip.resolve()
    assert seen["services"]["voice_clip"] == clip.resolve().as_posix()
    assert (seen["services"]["stt_engine"], seen["services"]["tts_engine"]) == (
        "nemotron-160",
        "pocket-clone",
    )
    assert str(clip.parent) not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The services' container lifecycle
# ---------------------------------------------------------------------------


class FakeDocker:
    def __init__(self, states=None):
        self.states = dict(states or {})
        self.calls: list[tuple[str, object]] = []

    def container_state(self, name):
        return self.states.get(name, "absent")

    def run_detached(self, argv):
        name = argv[argv.index("--name") + 1]
        self.states[name] = "running"
        self.calls.append(("run", name))
        return "id-" + name

    def logs(self, name):
        return (f"{name} out", "")

    def stop(self, name):
        self.calls.append(("stop", name))
        self.states[name] = "exited"

    def rm(self, name):
        self.calls.append(("rm", name))
        self.states.pop(name, None)


class HealthOpener:
    def __init__(self, clock, fail_first=0):
        self.clock = clock
        self.fail_first = fail_first
        self.urls = []

    def __call__(self, request, timeout=None):
        url = request.full_url if isinstance(request, urllib.request.Request) else request
        self.urls.append(url)
        if self.fail_first:
            self.fail_first -= 1
            raise urllib.error.URLError("not yet")
        return JsonResponse({"status": "ok"}, self.clock, 0.0)


def test_services_start_wait_for_health_and_save_logs_before_removal(tmp_path):
    clock = Clock()
    docker = FakeDocker()
    opener = HealthOpener(clock, fail_first=2)
    services = chain.ChainServices.default(
        repo_dir="/repo", speech_models_dir="/models", docker=docker, opener=opener,
        clock=clock, sleep=clock.sleep,
    )  # fmt: skip

    services.start()
    services.stop(tmp_path)

    assert [c for c in docker.calls if c[0] == "run"] == [
        ("run", chain.STT_CONTAINER),
        ("run", chain.TTS_CONTAINER),
    ]
    assert any(u.endswith(":8091/health") for u in opener.urls)
    assert any(u.endswith(":8092/health") for u in opener.urls)
    assert "mist-bench-chain-stt out" in (tmp_path / "stt_service.log").read_text(encoding="utf-8")
    assert (tmp_path / "tts_service.log").is_file()
    assert docker.calls[-1] == ("rm", chain.TTS_CONTAINER)


def test_services_refuse_to_start_over_an_existing_container():
    clock = Clock()
    docker = FakeDocker({chain.TTS_CONTAINER: "exited"})
    services = chain.ChainServices.default(
        repo_dir="/repo", speech_models_dir="/models", docker=docker,
        opener=HealthOpener(clock), clock=clock, sleep=clock.sleep,
    )  # fmt: skip

    with pytest.raises(core.SpeechBenchError, match=chain.TTS_CONTAINER):
        services.start()
    assert docker.calls == []


# ---------------------------------------------------------------------------
# The whole run
# ---------------------------------------------------------------------------


def scripted_llm(clock_delays=None):
    """An LLM that answers each chain prompt with its valid calls, then the done reply."""
    prompts = {p.text: p for p in toolcall.load_prompts()}

    def route(body):
        last = body["messages"][-1]
        if last["role"] == "tool":
            return done_script()
        prompt = prompts[last["content"]]
        if prompt.kind == "no_tool":
            return [
                (0.1, sse(content(SHORT_REPLY))),
                (0.01, sse(finish("stop"))),
                (0.0, sse("[DONE]")),
            ]
        events = []
        for index, call in enumerate(VALID_CALLS[prompt.id]):
            function = call["function"]
            events.append(
                (0.05, sse(tool_delta(index, f"c{index}", function["name"], function["arguments"])))
            )
        return [*events, (0.01, sse(finish("tool_calls"))), (0.0, sse("[DONE]"))]

    return route


def chain_request(tmp_path, material, **overrides):
    values = {
        "results_root": tmp_path / "results",
        "run": "mh1",
        "model": "gemma4-e2b",
        "placement": "gpu-full",
        "material_dir": material,
        "speech_models_dir": tmp_path,
    }
    values.update(overrides)
    return chain.ChainRequest(**values)


class FakeServiceLifecycle:
    def __init__(self):
        self.events = []

    def start(self):
        self.events.append("start")

    def stop(self, log_dir):
        self.events.append(("stop", Path(log_dir).name))


def fake_for_run(clock, alias="gemma4-e2b@gpu-full"):
    fake = FakeServices(clock, alias=alias)
    fake.chat_router = scripted_llm()
    fake.transcripts = [UTTERANCES[u] for u in chain.CHAIN_UTTERANCES for _ in range(2)]
    return fake


def test_run_chain_writes_two_rows_per_recording_with_every_stage(tmp_path, material):
    clock = Clock()
    fake = fake_for_run(clock)
    # Repetitions run recording by recording, so each recording's transcript is asked twice.
    fake.transcripts = [UTTERANCES[u] for u in chain.CHAIN_UTTERANCES for _ in range(2)]
    llm, stt, tts = make_clients(fake, clock)
    services = FakeServiceLifecycle()

    out = chain.run_chain(
        chain_request(tmp_path, material), llm=llm, stt=stt, tts=tts, services=services,
        container_state=lambda name: "running", clock=clock, git_sha="abc",
    )  # fmt: skip

    rows = [json.loads(line) for line in (out / "turns.jsonl").read_text().splitlines()]
    assert out == tmp_path / "results" / "mh1" / "speech" / "chain" / "gemma4-e2b" / "gpu-full"
    assert len(rows) == 28
    assert [(r["utt_id"], r["rep"]) for r in rows[:2]] == [("U01", 1), ("U01", 2)]
    assert {r["kind"] for r in rows} == {"command", "short", "hand_off"}
    for key in ("stt_ms", "llm_first_token_ms", "tool_call_ms", "first_clause_ms",
                "tts_first_audio_ms", "turn_ms", "transcript_wer", "tool_valid"):  # fmt: skip
        assert key in rows[0]
    assert all(r["tool_valid"] for r in rows)
    assert all(r["run"] == "mh1" and r["suite"] == "chain" and r["git_sha"] == "abc" for r in rows)
    assert "busy_decode_tps" not in rows[0]
    assert services.events == ["start", ("stop", "gpu-full")]
    meta = json.loads((out / "chain.json").read_text())
    assert meta["alias"] == "gemma4-e2b@gpu-full" and meta["voice"] == "alba"


def test_run_chain_refuses_a_server_serving_another_alias_before_writing(tmp_path, material):
    clock = Clock()
    fake = fake_for_run(clock, alias="minicpm5-2b@gpu-full")
    llm, stt, tts = make_clients(fake, clock)
    services = FakeServiceLifecycle()

    with pytest.raises(lowtier_suite.AliasError):
        chain.run_chain(
            chain_request(tmp_path, material), llm=llm, stt=stt, tts=tts, services=services,
            container_state=lambda name: "running", clock=clock,
        )  # fmt: skip

    assert not (tmp_path / "results" / "mh1").exists()
    assert services.events == []


def test_run_chain_refuses_an_existing_result_directory(tmp_path, material):
    clock = Clock()
    existing = tmp_path / "results" / "mh1" / "speech" / "chain" / "gemma4-e2b" / "gpu-full"
    existing.mkdir(parents=True)
    llm, stt, tts = make_clients(fake_for_run(clock), clock)

    with pytest.raises(core.OutputExistsError):
        chain.run_chain(
            chain_request(tmp_path, material), llm=llm, stt=stt, tts=tts,
            services=FakeServiceLifecycle(), container_state=lambda name: "running", clock=clock,
        )  # fmt: skip


def test_run_chain_refuses_a_results_root_inside_the_repository(material):
    clock = Clock()
    llm, stt, tts = make_clients(fake_for_run(clock), clock)
    request = chain_request(_REPO_ROOT, material, results_root=_REPO_ROOT / "results")

    with pytest.raises(core.ResultsRootError):
        chain.run_chain(
            request, llm=llm, stt=stt, tts=tts, services=FakeServiceLifecycle(),
            container_state=lambda name: "running", clock=clock,
        )  # fmt: skip


# ---------------------------------------------------------------------------
# --reply-pack: each turn's whole reply, saved for the blind reply A/B page
# ---------------------------------------------------------------------------

TIMING_FIELDS = (
    "stt_ms", "llm_first_token_ms", "tool_call_ms", "done_first_token_ms", "first_clause_ms",
    "tts_first_audio_ms", "tts_total_ms", "turn_ms", "turn_wall_ms", "t_turn_start",
    "t_turn_end",
)  # fmt: skip
WHOLE_DONE_REPLY = "Your ten minute timer is set and running now, and it rings at ten past nine."


def pack_run(tmp_path, material, name, **overrides):
    """A two-repetition run against the recorded fake stream, under its own results root."""
    clock = Clock()
    fake = fake_for_run(clock)
    llm, stt, tts = make_clients(fake, clock)
    services = FakeServiceLifecycle()
    request = chain_request(tmp_path, material, results_root=tmp_path / name, **overrides)
    out = chain.run_chain(
        request, llm=llm, stt=stt, tts=tts, services=services,
        container_state=lambda name: "running", clock=clock, git_sha="abc",
    )  # fmt: skip
    rows = [json.loads(line) for line in (out / "turns.jsonl").read_text().splitlines()]
    return rows, fake, services


def test_a_reply_pack_leaves_every_recorded_time_of_the_turn_unchanged(tmp_path, material):
    plain, plain_fake, _ = pack_run(tmp_path, material, "plain")
    packed, packed_fake, _ = pack_run(
        tmp_path, material, "packed", reply_pack=tmp_path / "replies-a"
    )

    assert len(packed) == len(plain) == 28
    for before, after in zip(plain, packed, strict=True):
        assert {f: after[f] for f in TIMING_FIELDS} == {f: before[f] for f in TIMING_FIELDS}
    assert [r["turn_ms"] for r in packed] == [r["turn_ms"] for r in plain]
    # The pack takes the reply the turn already streamed to its end: no request is added.
    assert [kind for kind, _ in packed_fake.requests] == [kind for kind, _ in plain_fake.requests]
    assert packed_fake.spoken == plain_fake.spoken


def test_a_reply_pack_saves_the_whole_reply_per_recording_and_repetition(tmp_path, material):
    pack = tmp_path / "replies-a"

    rows, fake, _ = pack_run(tmp_path, material, "results", reply_pack=pack)

    saved = [json.loads(line) for line in (pack / chain.REPLY_PACK_FILE).read_text().splitlines()]
    assert [(r["utt_id"], r["rep"]) for r in saved] == [(r["utt_id"], r["rep"]) for r in rows]
    by_turn = {(r["utt_id"], r["rep"]): r for r in saved}
    # The turn speaks only the first clause; the pack holds the whole reply.
    assert by_turn[("U01", 1)]["reply"] == WHOLE_DONE_REPLY
    assert "Your ten minute timer is set and running now," in fake.spoken
    assert by_turn[("U20", 2)]["reply"] == SHORT_REPLY
    assert by_turn[("U07", 1)]["reply"] == chain.HAND_OFF_ACK
    assert {r["kind"] for r in saved} == {"command", "short", "hand_off"}
    assert all(r["error"] is None for r in saved)
    meta = json.loads((pack / chain.REPLY_PACK_META).read_text())
    assert (meta["model"], meta["placement"], meta["alias"]) == (
        "gemma4-e2b",
        "gpu-full",
        "gemma4-e2b@gpu-full",
    )
    assert (meta["stt_engine"], meta["tts_engine"], meta["reps"]) == ("parakeet-cpu", "pocket", 2)
    assert meta["utterances"] == list(chain.CHAIN_UTTERANCES)


def test_a_reply_pack_keeps_a_failed_turn_with_its_error_and_no_reply(tmp_path, material):
    clock = Clock()
    fake = fake_for_run(clock)
    fake.transcripts = [UTTERANCES[u] for u in chain.CHAIN_UTTERANCES]
    route = fake.chat_router

    def failing(body):
        if body["messages"][-1]["content"] == UTTERANCES["U19"]:
            return [(0.05, sse({"error": {"message": "slot unavailable"}}))]
        return route(body)

    fake.chat_router = failing
    llm, stt, tts = make_clients(fake, clock)
    pack = tmp_path / "replies-a"

    chain.run_chain(
        chain_request(tmp_path, material, reps=1, reply_pack=pack), llm=llm, stt=stt, tts=tts,
        services=FakeServiceLifecycle(), container_state=lambda name: "running", clock=clock,
        git_sha="abc",
    )  # fmt: skip

    saved = {
        r["utt_id"]: r
        for r in (json.loads(x) for x in (pack / chain.REPLY_PACK_FILE).read_text().splitlines())
    }
    assert saved["U19"]["reply"] is None and "slot unavailable" in saved["U19"]["error"]
    assert saved["U20"]["reply"] == SHORT_REPLY


@pytest.mark.parametrize("where", ["inside-the-checkout", "existing"])
def test_a_reply_pack_is_refused_inside_the_checkout_or_over_an_earlier_one(
    tmp_path, material, where
):
    pack = _REPO_ROOT / "reply-pack-here" if where == "inside-the-checkout" else tmp_path / "old"
    if where == "existing":
        pack.mkdir()
    clock = Clock()
    llm, stt, tts = make_clients(fake_for_run(clock), clock)
    services = FakeServiceLifecycle()

    with pytest.raises(core.SpeechBenchError, match="--reply-pack"):
        chain.run_chain(
            chain_request(tmp_path, material, reply_pack=pack), llm=llm, stt=stt, tts=tts,
            services=services, container_state=lambda name: "running", clock=clock,
        )  # fmt: skip

    assert services.events == []
    assert not (tmp_path / "results").exists()
    assert not (_REPO_ROOT / "reply-pack-here").exists()


def test_cmd_chain_hands_the_reply_pack_to_the_request(tmp_path, monkeypatch):
    seen = {}

    def run_chain(request, **kwargs):
        seen["request"] = request
        out = tmp_path / "out"
        out.mkdir()
        (out / "turns.jsonl").write_text("", encoding="utf-8")
        return out

    monkeypatch.setattr(chain.ChainServices, "default", lambda **kwargs: FakeServiceLifecycle())
    monkeypatch.setattr(chain, "run_chain", run_chain)
    argv = [
        "chain", "--results-root", str(tmp_path / "results"), "--run", "mh1",
        "--model", "gemma4-e2b", "--placement", "gpu-full", "--material-dir", str(tmp_path),
        "--speech-models-dir", str(tmp_path), "--reply-pack", str(tmp_path / "replies-a"),
    ]  # fmt: skip

    assert speech_bench.main(argv) == 0
    assert seen["request"].reply_pack == tmp_path / "replies-a"
    parser = speech_bench.subcommand_parsers(speech_bench.build_parser())["chain"]
    assert {a.dest: a for a in parser._actions}["reply_pack"].default is None


def test_the_speech_client_renders_all_of_a_text_s_audio(tmp_path):
    clock = Clock()
    fake = FakeServices(clock)
    tts = chain.TtsClient(chain.TTS_URL, opener=fake, clock=clock)

    pcm, rate = tts.render(WHOLE_DONE_REPLY)

    assert fake.spoken == [WHOLE_DONE_REPLY]
    assert pcm == b"\x01\x00" * 480 + b"\x02\x00" * 480
    assert rate == 24000


def test_the_speech_only_service_runs_one_engine_with_its_clip():
    clock = Clock()
    docker = FakeDocker()
    services = chain.ChainServices.speech_only(
        repo_dir="/repo", speech_models_dir="/models", engine="pocket-clone",
        voice_clip=CLIP_HOST, docker=docker, opener=HealthOpener(clock), clock=clock,
        sleep=clock.sleep,
    )  # fmt: skip

    (spec,) = services.specs
    assert spec.container == chain.TTS_CONTAINER
    assert _pairs(list(spec.argv), "--engine") == ["pocket-clone"]
    assert CLIP_MOUNT in _pairs(list(spec.argv), "-v")
    services.start()
    assert [c for c in docker.calls if c[0] == "run"] == [("run", chain.TTS_CONTAINER)]


# ---------------------------------------------------------------------------
# --busy-arm c0: the chained turn while E4B writes
# ---------------------------------------------------------------------------


def completion(seq, start, end, tokens, ms):
    return load_gen.Completion(
        seq=seq, t_request=start, t_end=end, t_start_unix=0.0, t_end_unix=0.0,
        predicted_n=tokens, predicted_ms=ms, predicted_per_second=tokens / (ms / 1000),
        prompt_n=10, finish_reason="length",
    )  # fmt: skip


class FakeLoadGen:
    """Scripted completions: the baseline returns its own, the busy run ones spread over time."""

    instances: list[FakeLoadGen] = []

    def __init__(self, busy):
        self.busy = busy
        self.started = False
        self.stopped = False
        FakeLoadGen.instances.append(self)

    def run_for(self, seconds):
        return [completion(0, -70.0, -65.0, 400, 5000.0)]

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True
        return self.busy


def test_busy_arm_streams_from_mist_bench_llm_and_records_its_rate_per_turn(tmp_path, material):
    clock = Clock()
    fake = fake_for_run(clock)
    fake.transcripts = [UTTERANCES[u] for u in chain.CHAIN_UTTERANCES]
    llm, stt, tts = make_clients(fake, clock)
    # Two completions overlap the run: 300 tokens over 0 to 3 s, then 100 over 3 s to the end.
    busy = [completion(1, 0.0, 3.0, 300, 3000.0), completion(2, 3.0, 99.0, 1000, 20000.0)]
    FakeLoadGen.instances = []
    seen_states = []

    def state(name):
        seen_states.append(name)
        return "running"

    out = chain.run_chain(
        chain_request(tmp_path, material, busy_arm="c0", baseline_s=5.0),
        llm=llm, stt=stt, tts=tts, services=FakeServiceLifecycle(), container_state=state,
        load_gen_factory=lambda: FakeLoadGen(busy), clock=clock,
    )  # fmt: skip

    rows = [json.loads(line) for line in (out / "turns.jsonl").read_text().splitlines()]
    assert out.name == "gpu-full-e4b-writing"
    assert len(rows) == 14 and {r["rep"] for r in rows} == {1}
    assert "mist-bench-llm" in seen_states
    assert FakeLoadGen.instances[1].started and FakeLoadGen.instances[1].stopped
    first = rows[0]
    # The first turn ends before 3 s, so only the first completion overlaps it.
    assert first["busy_completions"] == 1
    assert first["busy_decode_tps"] == pytest.approx(100.0)
    last = rows[-1]
    assert last["busy_completions"] == 1
    assert last["busy_decode_tps"] == pytest.approx(50.0)
    decode = [json.loads(line) for line in (out / "decode.jsonl").read_text().splitlines()]
    assert [d["phase"] for d in decode] == ["alone", "with_chain", "with_chain"]


def test_busy_arm_refuses_any_placement_but_gpu_full(tmp_path, material):
    clock = Clock()
    llm, stt, tts = make_clients(fake_for_run(clock, alias="gemma4-e2b@cpu"), clock)

    with pytest.raises(core.SpeechBenchError, match="gpu-full"):
        chain.run_chain(
            chain_request(tmp_path, material, placement="cpu", busy_arm="c0"),
            llm=llm, stt=stt, tts=tts, services=FakeServiceLifecycle(),
            container_state=lambda name: "running", load_gen_factory=lambda: FakeLoadGen([]),
            clock=clock,
        )  # fmt: skip
    assert not (tmp_path / "results").exists()


def test_busy_arm_refuses_when_mist_bench_llm_is_not_running(tmp_path, material):
    clock = Clock()
    llm, stt, tts = make_clients(fake_for_run(clock), clock)

    with pytest.raises(core.SpeechBenchError, match="mist-bench-llm"):
        chain.run_chain(
            chain_request(tmp_path, material, busy_arm="c0"), llm=llm, stt=stt, tts=tts,
            services=FakeServiceLifecycle(), container_state=lambda name: "absent",
            load_gen_factory=lambda: FakeLoadGen([]), clock=clock,
        )  # fmt: skip
    assert not (tmp_path / "results").exists()


def test_the_chain_cli_names_the_load_generator_s_server():
    assert chain.LOAD_GEN_BASE_URL == "http://127.0.0.1:8080"
    parser = speech_bench.subcommand_parsers(speech_bench.build_parser())["chain"]
    options = {o for a in parser._actions for o in a.option_strings}

    assert {"--busy-arm", "--model", "--placement", "--material-dir", "--remote-url"} <= options


def test_a_host_1070_chain_takes_its_alias_from_the_placement(tmp_path, material):
    clock = Clock()
    alias = "gemma4-e2b@gtx1070-shared-busy"
    fake = fake_for_run(clock, alias=alias)
    fake.transcripts = [UTTERANCES[u] for u in chain.CHAIN_UTTERANCES]
    remote = "http://mist-extraction-gtx1070:8090"
    llm, stt, tts = make_clients(fake, clock, base_url=remote)

    out = chain.run_chain(
        chain_request(tmp_path, material, placement="gtx1070-shared-busy", remote_url=remote, reps=1),
        llm=llm, stt=stt, tts=tts, services=FakeServiceLifecycle(),
        container_state=lambda name: "running", clock=clock,
    )  # fmt: skip

    rows = [json.loads(line) for line in (out / "turns.jsonl").read_text().splitlines()]
    assert out.name == "gtx1070-shared-busy" and {r["alias"] for r in rows} == {alias}


def test_chain_selftest_passes():
    chain.selftest()


# ---------------------------------------------------------------------------
# The Parakeet service (probes/stt_service.py)
# ---------------------------------------------------------------------------


class FakeTranscriber:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, sample_rate):
        self.calls.append((len(audio), sample_rate))
        return "set a timer for ten minutes"


def serve_in_thread(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


def test_the_stt_service_binds_loopback_by_default():
    assert stt_service.DEFAULT_HOST == "127.0.0.1"
    args = stt_service.build_parser().parse_args([])
    assert (args.host, args.port, args.engine, args.threads) == (
        "127.0.0.1",
        8091,
        "parakeet-cpu",
        4,
    )


def test_the_stt_service_transcribes_a_posted_wav_with_the_adapter(tmp_path):
    transcriber = FakeTranscriber()
    ticks = iter([1.0, 1.25])
    server = stt_service.make_server(transcriber, port=0, clock=lambda: next(ticks))
    serve_in_thread(server)
    try:
        host, port = server.server_address[:2]
        assert host == "127.0.0.1"
        wav = write_wav(tmp_path / "u.wav", n_samples=3200)
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/transcribe", data=wav, method="POST",
            headers={"Content-Type": "audio/wav"},
        )  # fmt: skip
        with urllib.request.urlopen(request, timeout=10) as response:
            reply = json.loads(response.read())
    finally:
        server.shutdown()
        server.server_close()

    assert reply == {"text": "set a timer for ten minutes", "server_ms": 250.0, "audio_s": 0.2}
    assert transcriber.calls == [(3200, 16000)]


def test_the_stt_service_refuses_audio_that_is_not_16_khz_mono_pcm():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as clip:
        clip.setnchannels(2)
        clip.setsampwidth(2)
        clip.setframerate(16000)
        clip.writeframes(b"\x00\x00" * 200)

    with pytest.raises(stt_service.WavError):
        stt_service.decode_wav(buffer.getvalue())


def test_the_stt_service_loads_parakeet_through_t3_s_factory(monkeypatch):
    seen = {}

    def factory(engine_id, *, weights_dir, threads):
        seen.update(engine=engine_id, weights=weights_dir, threads=threads)
        return FakeTranscriber()

    monkeypatch.setattr(stt_service.stt_engines, "make_transcriber", factory)
    transcriber = stt_service.load_transcriber("parakeet-cpu", Path("/weights"), 4)

    assert isinstance(transcriber, FakeTranscriber)
    assert seen == {"engine": "parakeet-cpu", "weights": Path("/weights"), "threads": 4}


# ---------------------------------------------------------------------------
# The Pocket TTS service (probes/tts_service.py)
# ---------------------------------------------------------------------------


class FakeEngine:
    name = "pocket"
    device = "cpu"
    sample_rate = 24000

    def __init__(self):
        self.loaded = None
        self.texts = []

    def voices(self):
        return ["alba", "marius"]

    def load(self, voice):
        if voice not in self.voices():
            raise AssertionError(f"not a stock voice: {voice}")
        self.loaded = voice

    def stream(self, text):
        self.texts.append(text)
        yield [0.5] * 240
        yield [-0.5] * 240


def test_the_tts_service_binds_loopback_by_default():
    assert tts_service.DEFAULT_HOST == "127.0.0.1"
    args = tts_service.build_parser().parse_args([])
    assert (args.host, args.port) == ("127.0.0.1", 8092)


def test_the_tts_service_speaks_only_pocket_s_stock_timing_voice():
    engine = FakeEngine()
    seen = {}

    def make(engine_id, **kwargs):
        seen["engine"] = engine_id
        seen.update(kwargs)
        return engine

    loaded = tts_service.load_engine(Path("/models/speech"), make=make)

    assert loaded is engine and engine.loaded == "alba"
    assert seen["engine"] == "pocket" and seen["models_dir"] == Path("/models/speech")


def test_the_tts_service_streams_chunked_pcm_for_posted_text():
    engine = FakeEngine()
    engine.load("alba")
    server = tts_service.make_server(engine, port=0)
    serve_in_thread(server)
    try:
        host, port = server.server_address[:2]
        assert host == "127.0.0.1"
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/speak", method="POST",
            data=json.dumps({"text": "One moment."}).encode(),
            headers={"Content-Type": "application/json"},
        )  # fmt: skip
        with urllib.request.urlopen(request, timeout=10) as response:
            assert response.headers["Transfer-Encoding"] == "chunked"
            assert response.headers["X-Sample-Rate"] == "24000"
            body = response.read()
    finally:
        server.shutdown()
        server.server_close()

    assert engine.texts == ["One moment."]
    assert len(body) == 2 * 480
    assert body[:2] == (16383).to_bytes(2, "little", signed=True)


@pytest.mark.parametrize(
    "body",
    [{"text": "hi", "voice": "marius"}, {"text": ""}, {"words": "hi"}, ["hi"]],
)
def test_the_tts_service_takes_text_and_nothing_else(body):
    with pytest.raises(tts_service.SpeakError):
        tts_service.parse_speak_body(json.dumps(body).encode())


def test_the_tts_service_answers_a_refused_body_with_400():
    engine = FakeEngine()
    engine.load("alba")
    server = tts_service.make_server(engine, port=0)
    serve_in_thread(server)
    try:
        port = server.server_address[1]
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/speak", method="POST",
            data=json.dumps({"text": "hi", "voice": "marius"}).encode(),
        )  # fmt: skip
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=10)
    finally:
        server.shutdown()
        server.server_close()

    assert caught.value.code == 400
    assert engine.texts == []


def test_the_tts_service_takes_an_engine_and_defaults_to_pocket():
    assert tts_service.build_parser().parse_args([]).engine == "pocket"
    args = tts_service.build_parser().parse_args(["--engine", "qwen3-design"])
    assert args.engine == "qwen3-design"


class AnyVoiceEngine(FakeEngine):
    def voices(self):
        return [self.loaded_voice_hint]


@pytest.mark.parametrize("engine_id", ["pocket-clone", "qwen3", "qwen3-design"])
def test_the_tts_service_loads_the_chosen_engine_with_its_timing_voice(engine_id):
    voice = ARMS["tts"][engine_id]["timing_voice"]
    engine = AnyVoiceEngine()
    engine.loaded_voice_hint = voice
    seen = {}

    def make(name, **kwargs):
        seen["engine"] = name
        seen.update(kwargs)
        return engine

    loaded = tts_service.load_engine(Path("/models/speech"), engine_id=engine_id, make=make)

    assert loaded is engine and engine.loaded == voice
    assert seen["engine"] == engine_id and seen["models_dir"] == Path("/models/speech")
    assert voice == {"pocket-clone": "identity", "qwen3-design": "designed"}.get(engine_id, voice)
