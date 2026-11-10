"""scripts/model_bench/speech/core.py: the shared pieces every speech suite builds on.

Covers T1's refusals (results root inside the repository, an existing suite/arm directory, a
material file whose sha256 differs from the manifest, a model file whose sha256 or size differs
from the pin file, a bench run whose `session/` exists), the percentiles, the prefix builder
against a fake tokenizer, the streaming chat client on inline server-sent events with a scripted
clock, the committed material, the voice-sample search and the import-isolation check.

Fixtures: `fixtures/speech/material/manifest.json` pins the sha256 of two synthetic WAVs this file
writes with the stdlib `wave` module (no audio is committed: `*.wav` is gitignored), and
`fixtures/speech/host/pins.json` pins fake model files this file writes.
"""

from __future__ import annotations

import io
import json
import re
import struct
import subprocess
import sys
import wave
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402

from scripts.model_bench.speech import core  # noqa: E402

SPEECH_FIXTURES = _REPO_ROOT / "tests" / "unit" / "model_bench" / "fixtures" / "speech"
MANIFEST_PATH = SPEECH_FIXTURES / "material" / "manifest.json"
PINS_PATH = SPEECH_FIXTURES / "host" / "pins.json"
SPEECH_PKG_DIR = _REPO_ROOT / "scripts" / "model_bench" / "speech"

# (frames, step) per synthetic file: an integer-only sawtooth, so the bytes do not depend on a
# platform's floating-point sine.
SYNTHETIC_WAVS = {"S01.wav": (8000, 37), "S02.wav": (4000, 11)}

# Fake model file contents; their sha256 and sizes are pinned in fixtures/speech/host/pins.json.
FAKE_MODEL_BYTES = {
    "gemma4-e2b": b"GGUF fake gemma4-e2b weights\n" * 64,
    "minicpm5-2b": b"GGUF fake minicpm5-2b weights\n" * 48,
    "qwen35-4b": b"GGUF fake qwen35-4b weights\n" * 80,
    "e4b-ref": b"GGUF fake e4b-ref weights\n" * 96,
}


def write_synthetic_wav(path: Path, frames: int, step: int) -> None:
    samples = b"".join(struct.pack("<h", ((i * step) % 2000 - 1000) * 8) for i in range(frames))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(samples)


def write_material(directory: Path) -> None:
    for name, (frames, step) in SYNTHETIC_WAVS.items():
        write_synthetic_wav(directory / name, frames, step)


def write_fake_models(directory: Path) -> None:
    pins = json.loads(PINS_PATH.read_text(encoding="utf-8"))
    for model_id, data in FAKE_MODEL_BYTES.items():
        target = directory / pins["models"][model_id]["file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


# ---------------------------------------------------------------------------
# Results root and run ids
# ---------------------------------------------------------------------------


def test_speech_root_refuses_results_root_inside_repo():
    with pytest.raises(core.ResultsRootError):
        core.speech_root(_REPO_ROOT / "results", "mh1", repo_root=_REPO_ROOT)


def test_speech_root_is_run_speech_under_results_root(tmp_path):
    root = core.speech_root(tmp_path, "mh1", repo_root=_REPO_ROOT)

    assert root == tmp_path.resolve() / "mh1" / "speech"


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param("", id="empty"),
        pytest.param("..", id="parent"),
        pytest.param("a/b", id="slash"),
        pytest.param("a\\b", id="backslash"),
    ],
)
def test_speech_root_refuses_run_id_that_is_not_one_plain_name(tmp_path, bad):
    with pytest.raises(core.SpeechBenchError):
        core.speech_root(tmp_path, bad, repo_root=_REPO_ROOT)


# ---------------------------------------------------------------------------
# Output claims and the JSONL writer
# ---------------------------------------------------------------------------


def test_claim_output_refuses_an_existing_suite_arm_directory(tmp_path):
    core.claim_output(tmp_path, "mh1", "tts", "qwen3", git_sha="abc", repo_root=_REPO_ROOT)

    with pytest.raises(core.OutputExistsError):
        core.claim_output(tmp_path, "mh1", "tts", "qwen3", git_sha="abc", repo_root=_REPO_ROOT)


def test_claim_output_refuses_results_root_inside_repo():
    with pytest.raises(core.ResultsRootError):
        core.claim_output(_REPO_ROOT / "out", "mh1", "tts", "qwen3", git_sha="abc")


def test_jsonl_writer_stamps_run_suite_and_git_sha(tmp_path):
    claim = core.claim_output(tmp_path, "mh1", "cache", "c0", git_sha="deadbeef")

    with claim.jsonl("requests.jsonl") as writer:
        writer.write({"turn": 1, "cache_n": 30})

    rows = [json.loads(x) for x in (claim.path / "requests.jsonl").read_text().splitlines()]
    assert rows == [
        {"run": "mh1", "suite": "cache", "git_sha": "deadbeef", "turn": 1, "cache_n": 30}
    ]


def test_jsonl_writer_refuses_to_overwrite_an_existing_file(tmp_path):
    claim = core.claim_output(tmp_path, "mh1", "cache", "c0", git_sha="x")
    (claim.path / "requests.jsonl").write_text("{}\n")

    with pytest.raises(core.OutputExistsError):
        claim.jsonl("requests.jsonl")


def test_jsonl_writer_refuses_a_row_that_contradicts_its_stamp(tmp_path):
    claim = core.claim_output(tmp_path, "mh1", "cache", "c0", git_sha="x")

    with claim.jsonl("r.jsonl") as writer, pytest.raises(core.SpeechBenchError):
        writer.write({"run": "other"})


def test_shared_jsonl_appends_rows_from_two_claimed_arms(tmp_path):
    first = core.claim_output(tmp_path, "mh1", "tts", "qwen3", git_sha="x")
    second = core.claim_output(tmp_path, "mh1", "tts", "pocket", git_sha="x")

    with first.shared_jsonl("samples.jsonl") as writer:
        writer.write({"engine": "qwen3"})
    with second.shared_jsonl("samples.jsonl") as writer:
        writer.write({"engine": "pocket"})

    shared = first.suite_dir / "samples.jsonl"
    engines = [json.loads(x)["engine"] for x in shared.read_text().splitlines()]
    assert engines == ["qwen3", "pocket"]


# ---------------------------------------------------------------------------
# Percentiles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "n", "median", "p90"),
    [
        pytest.param(list(range(1, 11)), 10, 5.5, 9, id="one-to-ten"),
        pytest.param([5, 1, 3], 3, 3, 5, id="three-unsorted"),
        pytest.param([7], 1, 7, 7, id="single"),
        pytest.param(list(range(1, 21)), 20, 10.5, 18, id="one-to-twenty"),
        pytest.param([0.2, 0.4, 0.1, 0.3], 4, 0.25, 0.4, id="four-floats"),
    ],
)
def test_summarize_gives_count_median_and_nearest_rank_p90(values, n, median, p90):
    summary = core.summarize(values)

    assert (summary.n, summary.median, summary.p90) == (n, median, p90)


def test_summarize_of_no_values_has_count_zero_and_no_statistics():
    summary = core.summarize([])

    assert (summary.n, summary.median, summary.p90) == (0, None, None)


# ---------------------------------------------------------------------------
# Material manifest
# ---------------------------------------------------------------------------


def test_verify_material_accepts_files_matching_the_manifest(tmp_path):
    write_material(tmp_path)

    manifest = core.verify_material(MANIFEST_PATH, tmp_path)

    assert [item.id for item in manifest.items] == ["S01", "S02"]


def test_verify_material_refuses_a_file_whose_sha256_differs(tmp_path):
    write_material(tmp_path)
    data = bytearray((tmp_path / "S02.wav").read_bytes())
    data[-1] ^= 0xFF
    (tmp_path / "S02.wav").write_bytes(bytes(data))

    with pytest.raises(core.ManifestError, match="S02.wav"):
        core.verify_material(MANIFEST_PATH, tmp_path)


def test_verify_material_refuses_a_missing_file(tmp_path):
    write_material(tmp_path)
    (tmp_path / "S01.wav").unlink()

    with pytest.raises(core.ManifestError, match="S01.wav"):
        core.verify_material(MANIFEST_PATH, tmp_path)


def test_load_manifest_refuses_a_file_name_outside_the_material_directory(tmp_path):
    doc = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    doc["items"][0]["file"] = "../S01.wav"
    bad = tmp_path / "manifest.json"
    bad.write_text(json.dumps(doc))

    with pytest.raises(core.ManifestError):
        core.load_manifest(bad)


# ---------------------------------------------------------------------------
# Pin file
# ---------------------------------------------------------------------------


def test_verify_model_file_accepts_the_pinned_file(tmp_path):
    write_fake_models(tmp_path)
    pin = core.pinned_model(core.load_pins(PINS_PATH), "gemma4-e2b")

    core.verify_model_file(pin, tmp_path / pin.file)


def test_verify_model_file_refuses_a_file_whose_sha256_differs(tmp_path):
    write_fake_models(tmp_path)
    pin = core.pinned_model(core.load_pins(PINS_PATH), "qwen35-4b")
    path = tmp_path / pin.file
    data = bytearray(path.read_bytes())
    data[0] ^= 0xFF
    path.write_bytes(bytes(data))

    with pytest.raises(core.PinError, match="sha256"):
        core.verify_model_file(pin, path)


def test_verify_model_file_refuses_a_file_whose_size_differs(tmp_path):
    write_fake_models(tmp_path)
    pin = core.pinned_model(core.load_pins(PINS_PATH), "minicpm5-2b")
    path = tmp_path / pin.file
    path.write_bytes(path.read_bytes() + b"x")

    with pytest.raises(core.PinError, match="size"):
        core.verify_model_file(pin, path)


def test_pinned_model_refuses_a_model_the_pin_file_does_not_list():
    with pytest.raises(core.PinError):
        core.pinned_model(core.load_pins(PINS_PATH), "not-a-model")


# ---------------------------------------------------------------------------
# Bench-run windows
# ---------------------------------------------------------------------------


def test_begin_bench_window_refuses_a_bench_run_whose_session_exists(tmp_path):
    (tmp_path / "mh1-h4" / "session").mkdir(parents=True)

    with pytest.raises(core.BenchRunError):
        core.begin_bench_window(tmp_path, "mh1-h4", repo_root=_REPO_ROOT)


def test_begin_bench_window_returns_the_unused_session_path(tmp_path):
    session = core.begin_bench_window(tmp_path, "mh1-h5", repo_root=_REPO_ROOT)

    assert session == tmp_path.resolve() / "mh1-h5" / "session"
    assert not session.exists()


def test_bench_run_snapshot_requires_the_session_snapshot(tmp_path):
    with pytest.raises(core.BenchRunError):
        core.bench_run_snapshot(tmp_path, "mh1-h4", repo_root=_REPO_ROOT)


def test_bench_run_snapshot_returns_the_existing_snapshot(tmp_path):
    snap = tmp_path / "mh1-h4" / "session" / "snapshot.json"
    snap.parent.mkdir(parents=True)
    snap.write_text("{}")

    assert core.bench_run_snapshot(tmp_path, "mh1-h4", repo_root=_REPO_ROOT) == snap.resolve()


# ---------------------------------------------------------------------------
# Prefix builder
# ---------------------------------------------------------------------------


def fake_tokenize(text: str) -> list[int]:
    return list(range(len(text.split())))


@pytest.mark.parametrize("target", [1500, 6100])
def test_build_prefix_lands_within_25_tokens_of_the_target(target):
    prefix = core.build_prefix(target, fake_tokenize)

    assert abs(len(fake_tokenize(prefix)) - target) <= 25


def test_build_prefix_returns_the_same_bytes_on_every_call():
    first = core.build_prefix(6100, fake_tokenize).encode("utf-8")
    second = core.build_prefix(6100, fake_tokenize).encode("utf-8")

    assert first == second


def test_build_prefix_orders_system_prompt_then_sorted_vault_then_filler():
    prefix = core.build_prefix(6100, fake_tokenize)
    system = (_REPO_ROOT / "config" / "system_prompt.txt").read_text(encoding="utf-8")
    vault = sorted(
        (_REPO_ROOT / "tests" / "fixtures" / "test-vault").rglob("*.md"),
        key=lambda p: p.relative_to(_REPO_ROOT / "tests" / "fixtures" / "test-vault").as_posix(),
    )
    filler = core.CACHE_FILLER_PATH.read_text(encoding="utf-8")

    def first_line(text: str) -> str:
        return next(line for line in text.replace("\r\n", "\n").splitlines() if line.strip())

    positions = [prefix.index(first_line(system))]
    positions += [prefix.index(first_line(p.read_text(encoding="utf-8"))) for p in vault]
    positions.append(prefix.index(first_line(filler)))
    assert positions == sorted(positions)
    assert prefix.startswith(first_line(system))


def test_build_prefix_truncates_when_the_target_is_below_the_base_text():
    prefix = core.build_prefix(60, fake_tokenize)

    assert abs(len(fake_tokenize(prefix)) - 60) <= 25


def test_build_prefix_raises_when_no_cut_lands_within_tolerance():
    def coarse_tokenize(text: str) -> list[int]:
        return list(range(100 * len(text.split())))

    with pytest.raises(core.PrefixError):
        core.build_prefix(1550, coarse_tokenize)


# ---------------------------------------------------------------------------
# Streaming chat client
# ---------------------------------------------------------------------------

CONTENT_STREAM = """\
data: {"choices":[{"index":0,"delta":{"role":"assistant","content":null}}]}

data: {"choices":[{"index":0,"delta":{"reasoning_content":"hm"}}]}

data: {"choices":[{"index":0,"delta":{"content":"Hello"}}]}

data: {"choices":[{"index":0,"delta":{"content":" there"}}]}

data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"timings":{"cache_n":30,"prompt_n":12,"prompt_ms":8.5,"predicted_n":2,"predicted_per_second":90.0}}

data: [DONE]
"""

TOOL_STREAM = """\
: keep-alive comment
data: {"choices":[{"index":0,"delta":{"role":"assistant"}}]}

data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"set_timer","arguments":"{\\"min"}}]}}]}

data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"utes\\": 10}"}}]}}]}

data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"timings":{"prompt_n":40}}

data: [DONE]
"""


def scripted_clock(*values: float):
    it = iter(values)
    return lambda: next(it)


def test_parse_chat_stream_times_first_content_and_keeps_final_timings():
    clock = scripted_clock(10.1, 10.2, 10.35, 10.4, 10.5, 10.6)

    result = core.parse_chat_stream(CONTENT_STREAM.splitlines(), clock=clock, t_request=10.0)

    assert result.content == "Hello there"
    assert result.reasoning_content == "hm"
    assert result.ttft_s == pytest.approx(0.35)
    assert result.finish_reason == "stop"
    assert result.timings == {
        "cache_n": 30,
        "prompt_n": 12,
        "prompt_ms": 8.5,
        "predicted_n": 2,
        "predicted_per_second": 90.0,
    }
    assert result.t_end == pytest.approx(10.6)


def test_parse_chat_stream_assembles_tool_calls_and_times_the_first_tool_delta():
    clock = scripted_clock(5.1, 5.25, 5.3, 5.4, 5.5)

    result = core.parse_chat_stream(TOOL_STREAM.splitlines(), clock=clock, t_request=5.0)

    assert result.ttft_s == pytest.approx(0.25)
    assert result.t_first_content is None
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call["id"] == "call_1"
    assert call["function"]["name"] == "set_timer"
    assert json.loads(call["function"]["arguments"]) == {"minutes": 10}
    assert result.finish_reason == "tool_calls"


def test_parse_chat_stream_raises_on_an_error_event():
    lines = ['data: {"error": {"code": 500, "message": "boom"}}']

    with pytest.raises(core.StreamError, match="boom"):
        core.parse_chat_stream(lines, clock=scripted_clock(1.0), t_request=0.0)


class FakeResponse:
    def __init__(self, text: str, status: int = 200):
        self.status = status
        self._buf = io.BytesIO(text.encode("utf-8"))

    def __iter__(self):
        return iter(self._buf.readlines())

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_stream_chat_posts_a_streaming_request_and_parses_the_reply():
    sent = {}

    def opener(request, timeout):
        sent["url"] = request.full_url
        sent["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse(CONTENT_STREAM)

    clock = scripted_clock(1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6)
    body = {"messages": [{"role": "user", "content": "hi"}]}

    result = core.stream_chat("http://127.0.0.1:8081/", body, clock=clock, opener=opener)

    assert sent["url"] == "http://127.0.0.1:8081/v1/chat/completions"
    assert sent["body"]["stream"] is True
    assert result.content == "Hello there"
    assert result.ttft_s == pytest.approx(0.3)


def test_stream_chat_raises_on_a_non_200_status():
    def opener(request, timeout):
        return FakeResponse("", status=503)

    with pytest.raises(core.StreamError, match="503"):
        core.stream_chat("http://127.0.0.1:8081", {"messages": []}, opener=opener)


# ---------------------------------------------------------------------------
# Committed material
# ---------------------------------------------------------------------------

EXPECTED_LINES = {
    "L01": ("neutral", "Your timer is set for ten minutes, and I'll let you know when it's done."),
    "L02": (
        "neutral",
        "It's eleven degrees outside right now, with light rain expected after six.",
    ),
    "L03": (
        "tired, late at night",
        "It's nearly two in the morning, boss. Save it and finish this tomorrow.",
    ),
    "L04": (
        "tired, gentle",
        "Long day, wasn't it? I've set your alarm for eight. Get some sleep.",
    ),
    "L05": ("warm greeting", "Evening. Good to have you back. What are we working on tonight?"),
    "L06": (
        "dry humour",
        "Done. All four lights are off, which is a personal best for this room.",
    ),
    "L07": ("amused", "You asked me that yesterday too. Same answer: the blue one."),
    "L08": (
        "concerned, warning",
        "Careful there. Nobody actually has that number; what we have is an estimate.",
    ),
    "L09": ("urgent", "Your train leaves in twelve minutes. If you go now, you'll make it."),
    "L10": (
        "apologetic",
        "Sorry, I lost the connection to the lights. Give me a second to try again.",
    ),
    "L11": ("curious question", "That's the new speaker, is it? What do you want it doing?"),
    "L12": ("encouraging", "That's the third test passing in a row. Nice work, honestly."),
    "L13": (
        "calm reassurance",
        "Nothing's broken. The backup finished at four, and every file checked out.",
    ),
    "L14": (
        "list",
        "Headphones, charger, passport and the blue notebook. That's everything on your list.",
    ),
    "L15": (
        "figures",
        "The survey says sixty-three point eight percent, from about two thousand people in "
        "September.",
    ),
    "L16": (
        "long",
        "I've moved your dentist appointment to Thursday at half past three, and I'll remind you "
        "an hour before, so don't worry about it.",
    ),
    "L17": ("sympathetic", "That's a shame. I know you were looking forward to it."),
    "L18": (
        "playful refusal",
        "No, I'm not ordering pizza again. You told me to stop you, remember?",
    ),
    "L19": ("focused", "Give me a moment. I'm pulling the numbers from last week now."),
    "L20": ("signing off", "Alright, I'll go quiet. Shout if you need me."),
}

EXPECTED_ACKS = {
    "A01": "Checking.",
    "A02": "On it.",
    "A03": "Done.",
    "A04": "Let me find it.",
    "A05": "Got it, boss.",
    "A06": "One moment.",
}

EMOTIVE = {f"L{i:02d}" for i in [*range(3, 14), *range(17, 21)]}

EXPECTED_UTTERANCES = {
    "U01": "Set a timer for ten minutes.",
    "U02": "What's on my calendar tomorrow morning?",
    "U03": "Turn the living room lights down to thirty percent.",
    "U04": "Remind me to call the dentist on Monday at nine.",
    "U05": "How long would it take to drive to the airport right now?",
    "U06": "Play something calm, maybe some piano, not too loud.",
    "U07": "What did we decide about the database migration last week?",
    "U08": "Add oat milk, eggs and coffee filters to the shopping list.",
    "U09": "Cancel the alarm for six thirty and set one for seven fifteen instead.",
    "U10": "Is it going to rain this afternoon?",
    "U11": "Show me the report on the second monitor.",
    "U12": "Close that window and bring the chart back up.",
    "U13": "Read me the last message from the build server.",
    "U14": "What time is it in Tokyo right now?",
    "U15": "Sum up what I worked on today in two sentences.",
    "U16": "Pause the music and turn the volume down a bit.",
    "U17": "How many steps did I take yesterday compared to the day before?",
    "U18": "Okay, that's enough for tonight. Go to sleep.",
    "U19": "Spell the word necessary for me, slowly.",
    "U20": "Tell me something interesting about octopuses.",
    "P01": "Can you remind me to ... pick up the parcel tomorrow?",
    "P02": "Set an alarm for ... actually, make it seven thirty.",
    "P03": "What was the name of that ... the restaurant we tried last month?",
    "P04": "Turn on the ... kitchen lights, please.",
    "P05": "I want to ... no, wait, show me the calendar first.",
    "P06": "Add a note that says ... call the landlord about the heating.",
    "P07": "How far is it from here to ... the coast?",
    "P08": "So the thing I keep forgetting is ... where I put the spare keys.",
    "P09": "Play the playlist from ... last Sunday's run.",
    "P10": "Can you check whether ... the backup finished last night?",
}


def test_lines_json_holds_the_twenty_lines_exactly_as_the_plan_lists_them():
    doc = core.load_json(core.LINES_PATH)

    got = {line["id"]: (line["feeling"], line["text"]) for line in doc["lines"]}
    assert got == EXPECTED_LINES
    assert [line["id"] for line in doc["lines"]] == sorted(EXPECTED_LINES)


def test_lines_json_marks_l03_to_l13_and_l17_to_l20_emotive():
    doc = core.load_json(core.LINES_PATH)

    assert {line["id"] for line in doc["lines"] if line["emotive"]} == EMOTIVE


def test_lines_l01_to_l20_hold_5_to_25_words():
    doc = core.load_json(core.LINES_PATH)

    assert all(5 <= len(line["text"].split()) <= 25 for line in doc["lines"])


def test_lines_json_carries_turbo_tags_on_l03_l04_and_l07_only():
    doc = core.load_json(core.LINES_PATH)
    turbo = {line["id"]: line["turbo_text"] for line in doc["lines"] if "turbo_text" in line}

    assert turbo == {
        "L03": "[sigh] " + EXPECTED_LINES["L03"][1],
        "L04": "[sigh] " + EXPECTED_LINES["L04"][1],
        "L07": "You asked me that yesterday too. [chuckle] Same answer: the blue one.",
    }


def test_lines_json_holds_the_six_acknowledgements():
    doc = core.load_json(core.LINES_PATH)

    assert {ack["id"]: ack["text"] for ack in doc["acknowledgements"]} == EXPECTED_ACKS


def test_utterances_json_holds_the_twenty_complete_and_ten_pause_utterances():
    doc = core.load_json(core.UTTERANCES_PATH)

    got = {u["id"]: u["text"] for u in doc["utterances"]}
    kinds = {u["id"]: u["kind"] for u in doc["utterances"]}
    assert got == EXPECTED_UTTERANCES
    assert all(kinds[k] == ("complete" if k.startswith("U") else "pause") for k in kinds)
    assert all(("..." in text) == key.startswith("P") for key, text in got.items())


def test_speech_arms_lists_the_stock_voices_and_timing_voices():
    tts = core.load_json(core.SPEECH_ARMS_PATH)["tts"]

    assert tts["qwen3"]["voices"] == [
        "vivian",
        "serena",
        "uncle_fu",
        "dylan",
        "eric",
        "ryan",
        "aiden",
        "ono_anna",
        "sohee",
    ]
    assert tts["pocket"]["voices"] == [
        "alba",
        "marius",
        "javert",
        "jean",
        "fantine",
        "cosette",
        "eponine",
        "azelma",
    ]
    assert tts["turbo"]["voices"] == ["builtin"]
    assert tts["original"]["voices"] == ["builtin"]
    timing = {engine: cfg["timing_voice"] for engine, cfg in tts.items()}
    assert timing == {"qwen3": "aiden", "turbo": "builtin", "original": "builtin", "pocket": "alba"}
    assert set(tts["qwen3"]["listening_voices"]) == {
        "ryan",
        "vivian",
        "serena",
        "ono_anna",
        "sohee",
    }
    assert all(cfg["verified"] is False for cfg in tts.values())


def test_every_female_voice_but_the_timing_voice_is_a_listening_voice():
    tts = core.load_json(core.SPEECH_ARMS_PATH)["tts"]

    for engine, cfg in tts.items():
        expected = set(cfg["female_voices"]) - {cfg["timing_voice"]}
        assert expected <= set(cfg["listening_voices"]), engine
        assert set(cfg["female_voices"]) <= set(cfg["voices"]), engine


def test_speech_arms_holds_the_original_default_and_expressive_settings():
    original = core.load_json(core.SPEECH_ARMS_PATH)["tts"]["original"]["settings"]

    assert original["default"] == {"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8}
    assert original["expressive"]["exaggeration"] == 0.7
    assert original["expressive"]["cfg_weight"] == 0.3
    assert original["expressive"]["lines"] == "emotive"


def test_speech_arms_holds_the_stt_and_turn_engines():
    doc = core.load_json(core.SPEECH_ARMS_PATH)

    assert list(doc["stt"]) == ["whisper-base", "parakeet-gpu", "parakeet-cpu", "nemotron-160"]
    assert doc["stt"]["parakeet-cpu"]["threads"] == 4
    assert doc["stt"]["nemotron-160"]["chunk_ms"] == 160
    turn = doc["turn"]["smart-turn-v3.2"]
    assert (turn["model"]["file"], turn["model"]["bytes"]) == ("smart-turn-v3.2-cpu.onnx", 8679182)
    assert (turn["device"], turn["threads"]) == ("cpu", 4)


def test_small_arms_holds_four_models_and_no_file_pins():
    doc = core.load_json(core.SMALL_ARMS_PATH)

    assert list(doc["models"]) == ["gemma4-e2b", "minicpm5-2b", "qwen35-4b", "e4b-ref"]
    assert doc["models"]["e4b-ref"]["quant"] == "Q5_K_M"
    assert doc["sampling"] == {"temperature": 0.7, "top_p": 0.9, "top_k": 40, "min_p": 0}
    text = core.SMALL_ARMS_PATH.read_text(encoding="utf-8")
    assert not re.search(r'"(sha256|revision|size|file)"', text)


def test_core_validate_inputs_finds_no_problem_in_the_committed_material():
    assert core.validate_inputs(_REPO_ROOT) == []


# ---------------------------------------------------------------------------
# No voice sample anywhere (Raj, 2026-11-06)
# ---------------------------------------------------------------------------

VOICE_SAMPLE_STRINGS = ("voice_profiles", "--reference", "audio_prompt_path", "ref_audio")


def _searched_files() -> list[Path]:
    files = [p for p in SPEECH_PKG_DIR.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    files.append(_REPO_ROOT / "scripts" / "model_bench" / "speech_bench.py")
    files.append(_REPO_ROOT / "scripts" / "model_bench" / "README.md")
    return files


def test_no_speech_bench_file_names_a_voice_sample_or_profile_path():
    hits = []
    for path in _searched_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        hits += [f"{path.name}: {s}" for s in VOICE_SAMPLE_STRINGS if s in text]

    assert hits == []


# ---------------------------------------------------------------------------
# Import isolation: discovered, so later tasks' modules are covered too
# ---------------------------------------------------------------------------

ENGINE_PACKAGES = (
    "faster_qwen3_tts",
    "pocket_tts",
    "nemo",
    "onnx_asr",
    "chatterbox",
    "whisper",
    "onnxruntime",
)
HOST_FORBIDDEN = (*ENGINE_PACKAGES, "numpy", "torch", "soundfile")


def _speech_modules() -> list[str]:
    names = ["scripts.model_bench.speech_bench"]
    for path in sorted(SPEECH_PKG_DIR.rglob("*.py")):
        rel = path.relative_to(_REPO_ROOT).with_suffix("")
        parts = list(rel.parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        names.append(".".join(parts))
    return names


def _is_probe(name: str) -> bool:
    return ".speech.probes" in name


@pytest.mark.parametrize("module", _speech_modules())
def test_importing_a_speech_module_loads_no_forbidden_package(module):
    forbidden = ENGINE_PACKAGES if _is_probe(module) else HOST_FORBIDDEN
    code = (
        "import importlib, json, sys\n"
        f"importlib.import_module({module!r})\n"
        f"print(json.dumps([m for m in {list(forbidden)!r} if m in sys.modules]))\n"
    )

    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=_REPO_ROOT, capture_output=True, text=True, timeout=120
    )

    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout.strip().splitlines()[-1]) == []


def test_module_discovery_covers_core_and_the_engine_protocols():
    names = _speech_modules()

    assert "scripts.model_bench.speech.core" in names
    assert "scripts.model_bench.speech.probes.engines" in names
