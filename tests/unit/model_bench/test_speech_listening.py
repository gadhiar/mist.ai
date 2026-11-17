"""Run 1's blind listening pack: `listening.py`, `probes/listen_pack.py` and the page template.

The pack only reads clips `tts` already saved; these tests build that saved tree by hand, in the
layout the merged probe writes, so nothing is synthesized and no engine package is imported.
ffmpeg is a fake that answers the first pass with a loudnorm report and writes a synthetic tone on
the second, and the container runner is a fake that runs the real probe in-process, so the host's
argv, the probe and the host's checks are tested together without Docker.
"""

from __future__ import annotations

import array
import json
import math
import re
import sys
import wave
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402

from scripts.model_bench import speech_bench  # noqa: E402
from scripts.model_bench.speech import chain, core, listening  # noqa: E402
from scripts.model_bench.speech.probes import listen_pack, tts_probe  # noqa: E402

ARMS = core.load_json(core.SPEECH_ARMS_PATH)
TTS = ARMS["tts"]
LINES_DOC = core.load_json(core.LINES_PATH)
LINE_IDS = [line["id"] for line in LINES_DOC["lines"]]
ACK_IDS = [ack["id"] for ack in LINES_DOC["acknowledgements"]]
EMOTIVE = [line["id"] for line in LINES_DOC["lines"] if line["emotive"]]
TEXTS = {item["id"]: item["text"] for item in [*LINES_DOC["lines"], *LINES_DOC["acknowledgements"]]}
TIMING = {engine: cfg["timing_voice"] for engine, cfg in TTS.items()}
# The four arms every results root holds; the pack refuses a root missing any of them.
BASE_ARMS = ("qwen3", "turbo", "original", "pocket")


# ---------------------------------------------------------------------------
# A saved `tts` tree, as the merged probe writes it
# ---------------------------------------------------------------------------


def _write_wav(path: Path, samples, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = array.array("h", (max(-32768, min(32767, int(round(s * 32767)))) for s in samples))
    with wave.open(str(path), "wb") as clip:
        clip.setnchannels(1)
        clip.setsampwidth(2)
        clip.setframerate(rate)
        clip.writeframes(pcm.tobytes())


def tone(amplitude: float, seconds: float, rate: int, freq: float = 997.0) -> list[float]:
    n = int(seconds * rate)
    return [amplitude * math.sin(2 * math.pi * freq * i / rate) for i in range(n)]


TINY = [0.1, -0.1] * 5


def _jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def build_tts_tree(
    speech: Path,
    *,
    lines=None,
    acks=None,
    extra_voices: dict | None = None,
    engines=None,
) -> None:
    """Write `tts/<engine>/` as `tts` leaves it: timed rows and clips, then listening clips."""
    lines = LINE_IDS if lines is None else list(lines)
    acks = ACK_IDS if acks is None else list(acks)
    extra_voices = extra_voices or {}
    for engine in engines or TTS:
        cfg = TTS[engine]
        base = speech / "tts" / engine
        voice = cfg["timing_voice"]
        for condition in tts_probe.CONDITIONS:
            rows = [{"row_type": "load", "engine": engine, "voice": voice, "condition": condition}]
            for item in [*lines, *acks]:
                for rep in (1, 2, 3):
                    name = tts_probe.timed_wav_name(condition, item, rep)
                    rel = f"tts/{engine}/audio/{name}"
                    _write_wav(speech / rel, TINY, 24000)
                    rows.append(
                        {
                            "row_type": "sample",
                            "engine": engine,
                            "voice": voice,
                            "setting": "default",
                            "condition": condition,
                            "kind": "ack" if item in acks else "line",
                            "line_id": item,
                            "rep": rep,
                            "wav": rel,
                        }
                    )
            _jsonl(base / tts_probe.probe_file(condition), rows)
        rows = []
        voices = list(cfg["listening_voices"]) + list(extra_voices.get(engine, ()))
        renders = [(v, v, "default", lines) for v in voices]
        if "expressive" in cfg["settings"]:
            emotive = [i for i in lines if i in EMOTIVE]
            renders.append((voice, "expressive", "expressive", emotive))
        for render_voice, label, setting, ids in renders:
            for line_id in ids:
                rel = f"tts/{engine}/listening/{label}/{line_id}.wav"
                _write_wav(speech / rel, TINY, 24000)
                rows.append(
                    {
                        "engine": engine,
                        "voice": render_voice,
                        "setting": setting,
                        "line_id": line_id,
                        "rep": 1,
                        "audio_s": 0.1,
                        "wav": rel,
                    }
                )
        if rows or engine in ("qwen3", "pocket", "original"):
            _jsonl(base / tts_probe.LISTENING_FILE, rows)


def speech_dir(tmp_path: Path) -> Path:
    return tmp_path / "results" / "mh1" / "speech"


# ---------------------------------------------------------------------------
# Fake ffmpeg and a fake container runner running the real probe in-process
# ---------------------------------------------------------------------------

PASS1_REPORT = {
    "input_i": "-27.61",
    "input_tp": "-4.47",
    "input_lra": "6.06",
    "input_thresh": "-38.20",
    "output_i": "-20.02",
    "output_tp": "-1.50",
    "output_lra": "5.10",
    "output_thresh": "-30.60",
    "normalization_type": "dynamic",
    "target_offset": "0.02",
}

TARGET_AMPLITUDE = 10 ** ((-20.0 + 3.0103) / 20)


def _stderr(report: dict) -> str:
    body = ",\n".join(f'\t"{k}" : "{v}"' for k, v in report.items())
    return (
        "Input #0, wav, from 'x.wav':\n  Duration: 00:00:01.00\n"
        "[Parsed_loudnorm_0 @ 0x55d0c0a1b2c0] \n{\n" + body + "\n}\n"
    )


class FakeFfmpeg:
    """Answers pass 1 with a report and writes a tone at `amplitude` on pass 2."""

    def __init__(self, amplitude: float = TARGET_AMPLITUDE, seconds: float = 0.5):
        self.argvs: list[list[str]] = []
        self.samples = tone(amplitude, seconds, listen_pack.OUTPUT_RATE)

    def __call__(self, argv):
        self.argvs.append(list(argv))
        if argv[-3:] == ["-f", "null", "-"]:
            return 0, _stderr(PASS1_REPORT)
        out = Path(argv[-1])
        if out.exists():
            return 1, "File exists"
        _write_wav(out, self.samples, listen_pack.OUTPUT_RATE)
        report = dict(PASS1_REPORT, normalization_type="linear")
        return 0, _stderr(report)


def mounts(argv) -> dict[str, str]:
    """Container path to host path, from every `-v host:container[:ro]`."""
    table = {}
    for i, arg in enumerate(argv):
        if arg == "-v":
            spec = argv[i + 1]
            if spec.endswith(":ro"):
                spec = spec[: -len(":ro")]
            host, container = spec.rsplit(":", 1)
            table[container] = host
    return table


class FakeRunner:
    def __init__(self, ffmpeg: FakeFfmpeg | None = None, *, code: int | None = None):
        self.ffmpeg = ffmpeg or FakeFfmpeg()
        self.code = code
        self.argvs: list[list[str]] = []

    def run(self, argv, *, log_path, on_line):
        self.argvs.append(list(argv))
        Path(log_path).write_text("log\n", encoding="utf-8")
        if self.code is not None:
            return self.code
        table = mounts(argv)
        tail = argv[argv.index(listening.PROBE_MODULE) + 1 :]
        mapped = []
        for i, arg in enumerate(tail):
            if i > 0 and tail[i - 1] in ("--jobs", "--out"):
                mapped.append(table[arg])
            elif i > 0 and tail[i - 1] == "--speech":
                # Only `tts/` is mounted under the speech root, so the key is out of reach.
                mapped.append(str(Path(table[f"{arg}/tts"]).parent))
            else:
                mapped.append(arg)
        return listen_pack.main(mapped, run=self.ffmpeg, emit=on_line)


def build(tmp_path: Path, *, seed: int = 7, runner: FakeRunner | None = None, **kwargs) -> Path:
    request = listening.PackRequest(results_root=tmp_path / "results", run="mh1", seed=seed)
    return listening.build_pack(
        request, runner=runner or FakeRunner(), git_sha="abc123", emit=lambda line: None, **kwargs
    )


SMALL = {"lines": ["L01", "L03"], "acks": ["A01"]}


# ---------------------------------------------------------------------------
# Which clips: repeat 1 of five settings in each timing voice, plus every untimed clip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("held", [BASE_ARMS, tuple(TTS)], ids=["base-arms", "every-arm"])
def test_the_pack_takes_repeat_1_of_each_line_from_each_held_arm_in_its_timing_voice(
    tmp_path, held
):
    speech = speech_dir(tmp_path)
    build_tts_tree(speech, engines=held)

    clips = listening.collect_clips(speech, TTS, LINES_DOC)

    timed = [c for c in clips if c.condition is not None]
    assert {(c.engine, c.setting, c.voice) for c in timed} == {
        (engine, "default", TIMING[engine]) for engine in held
    }
    assert {c.rep for c in clips} == {1}
    assert {c.condition for c in timed} == {"idle"}
    for engine in held:
        got = sorted(c.line_id for c in timed if c.engine == engine)
        assert got == sorted(LINE_IDS + ACK_IDS)
    expressive = [c for c in clips if c.setting == "expressive"]
    assert sorted(c.line_id for c in expressive) == sorted(EMOTIVE)
    assert {(c.engine, c.voice) for c in expressive} == {("original", "builtin")}
    settings = {(c.engine, c.setting) for c in clips}
    assert settings == {(engine, "default") for engine in held} | {("original", "expressive")}


def test_a_pocket_clone_clip_passes_the_voice_check_in_the_identity_voice(tmp_path):
    speech = speech_dir(tmp_path)
    build_tts_tree(speech)

    clips = listening.collect_clips(speech, TTS, LINES_DOC)

    clone = {(c.voice, c.setting) for c in clips if c.engine == "pocket-clone"}
    assert clone == {("identity", "default")}


@pytest.mark.parametrize("held", [BASE_ARMS, tuple(TTS)], ids=["base-arms", "every-arm"])
def test_the_pack_includes_every_listening_only_and_extra_voice_clip(tmp_path, held):
    speech = speech_dir(tmp_path)
    build_tts_tree(speech, extra_voices={"qwen3": ["eric"], "pocket": ["marius"]}, engines=held)

    clips = listening.collect_clips(speech, TTS, LINES_DOC)

    untimed = [c for c in clips if c.condition is None and c.setting == "default"]
    by_voice = {}
    for clip in untimed:
        by_voice.setdefault((clip.engine, clip.voice), []).append(clip.line_id)
    expected = {(e, v) for e in held for v in TTS[e]["listening_voices"]}
    expected |= {("qwen3", "eric"), ("pocket", "marius")}
    assert set(by_voice) == expected
    assert all(sorted(ids) == sorted(LINE_IDS) for ids in by_voice.values())
    timed = len(held) * (len(LINE_IDS) + len(ACK_IDS))
    listening_only = sum(len(TTS[e]["listening_voices"]) for e in held) + 2
    assert len(clips) == timed + len(EMOTIVE) + listening_only * len(LINE_IDS)


def test_every_clip_the_pack_takes_is_a_file_tts_already_saved(tmp_path):
    speech = speech_dir(tmp_path)
    build_tts_tree(speech)
    saved = {p.relative_to(speech).as_posix() for p in (speech / "tts").rglob("*.wav")}

    clips = listening.collect_clips(speech, TTS, LINES_DOC)

    assert {c.src for c in clips} <= saved


@pytest.mark.parametrize("missing", BASE_ARMS)
def test_the_pack_refuses_a_base_arm_tts_has_not_saved(tmp_path, missing):
    speech = speech_dir(tmp_path)
    build_tts_tree(speech, engines=[e for e in TTS if e != missing])

    with pytest.raises(core.SpeechBenchError, match=f"for {missing}:"):
        listening.collect_clips(speech, TTS, LINES_DOC)


def test_the_pack_skips_each_new_arm_with_no_timed_clips_and_says_so_once(tmp_path):
    speech = speech_dir(tmp_path)
    build_tts_tree(speech, engines=BASE_ARMS)
    said: list[str] = []

    clips = listening.collect_clips(speech, TTS, LINES_DOC, emit=said.append)

    assert {c.engine for c in clips} == set(BASE_ARMS)
    assert said == [
        "[skip] pocket-clone: tts saved no timed clips for it; the pack leaves it out",
        "[skip] qwen3-design: tts saved no timed clips for it; the pack leaves it out",
    ]


def test_a_results_root_from_before_the_new_arms_packs_as_it_did(tmp_path):
    speech = speech_dir(tmp_path)
    build_tts_tree(speech, engines=BASE_ARMS, **SMALL)
    said: list[str] = []
    request = listening.PackRequest(results_root=tmp_path / "results", run="mh1", seed=7)

    page = listening.build_pack(request, runner=FakeRunner(), git_sha="abc123", emit=said.append)

    key = json.loads((speech / "listening-key" / "key.json").read_text(encoding="utf-8"))
    assert {c["engine"] for c in key["clips"]} == set(BASE_ARMS)
    timed = len(BASE_ARMS) * (len(SMALL["lines"]) + len(SMALL["acks"]))
    untimed = sum(len(TTS[e]["listening_voices"]) for e in BASE_ARMS) * len(SMALL["lines"])
    emotive = len([i for i in SMALL["lines"] if i in EMOTIVE])
    assert len(key["clips"]) == timed + untimed + emotive
    assert len(list((page.parent / "clips").glob("*.wav"))) == len(key["clips"])
    assert [line for line in said if line.startswith("[skip]")] == [
        "[skip] pocket-clone: tts saved no timed clips for it; the pack leaves it out",
        "[skip] qwen3-design: tts saved no timed clips for it; the pack leaves it out",
    ]


def test_the_blind_check_covers_the_new_arms_ids_voices_and_repositories():
    terms = listening.forbidden_terms(TTS)

    assert {"pocket-clone", "qwen3-design", "identity", "designed"} <= terms
    assert {"qwen3-tts-12hz-1.7b-voicedesign", "pocket-tts", "kyutai", "qwen"} <= terms
    assert listening.blind_hits("this one sounds like the identity clip", TTS) == ["identity"]
    assert listening.blind_hits("the designed one", TTS) == ["designed"]


def test_the_pack_refuses_a_saved_row_whose_clip_is_missing(tmp_path):
    speech = speech_dir(tmp_path)
    build_tts_tree(speech)
    (speech / "tts" / "qwen3" / "audio" / "idle_L01_r1.wav").unlink()

    with pytest.raises(core.SpeechBenchError, match="idle_L01_r1.wav"):
        listening.collect_clips(speech, TTS, LINES_DOC)


def test_the_pack_refuses_a_voice_the_engine_does_not_list(tmp_path):
    speech = speech_dir(tmp_path)
    build_tts_tree(speech, extra_voices={"pocket": ["nobody"]})

    with pytest.raises(core.SpeechBenchError, match="nobody"):
        listening.collect_clips(speech, TTS, LINES_DOC)


@pytest.mark.parametrize("wav", ["../outside.wav", "/abs/x.wav", "tts/turbo/audio/x.wav"])
def test_the_pack_refuses_a_clip_path_outside_the_engines_own_directory(tmp_path, wav):
    speech = speech_dir(tmp_path)
    build_tts_tree(speech)
    path = speech / "tts" / "qwen3" / tts_probe.LISTENING_FILE
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["wav"] = wav
    _jsonl(path, rows)

    with pytest.raises(core.SpeechBenchError, match="outside"):
        listening.collect_clips(speech, TTS, LINES_DOC)


def test_the_pack_refuses_a_clip_saved_twice(tmp_path):
    speech = speech_dir(tmp_path)
    build_tts_tree(speech)
    path = speech / "tts" / "pocket" / tts_probe.LISTENING_FILE
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    _jsonl(path, [*rows, rows[0]])

    with pytest.raises(core.SpeechBenchError, match="twice"):
        listening.collect_clips(speech, TTS, LINES_DOC)


# ---------------------------------------------------------------------------
# Seeded random labels and the page's grouping
# ---------------------------------------------------------------------------


def _clips(tmp_path: Path):
    speech = speech_dir(tmp_path)
    build_tts_tree(speech)
    return listening.collect_clips(speech, TTS, LINES_DOC)


def test_labels_are_a_permutation_that_changes_with_the_seed(tmp_path):
    clips = _clips(tmp_path)

    first = listening.assign_labels(clips, seed=1)
    again = listening.assign_labels(clips, seed=1)
    other = listening.assign_labels(clips, seed=2)

    width = len(str(len(clips)))
    expected = {f"c{i:0{width}d}" for i in range(1, len(clips) + 1)}
    assert set(first) == expected
    assert set(first.values()) == set(clips)
    assert first == again
    assert first != other
    in_source_order = [label for clip in clips for label, c in first.items() if c == clip]
    assert in_source_order != sorted(in_source_order)


def test_clips_are_grouped_by_line_in_shuffled_order_with_text_and_feeling(tmp_path):
    clips = _clips(tmp_path)
    labelled = listening.assign_labels(clips, seed=3)

    groups = listening.page_groups(labelled, LINES_DOC, seed=3)

    assert [g["line_id"] for g in groups] == LINE_IDS + ACK_IDS
    feelings = {line["id"]: line["feeling"] for line in LINES_DOC["lines"]}
    for group in groups:
        line_id = group["line_id"]
        assert group["text"] == TEXTS[line_id]
        assert group["feeling"] == feelings.get(line_id, listening.ACK_FEELING)
        assert sorted(group["clips"]) == sorted(
            label for label, clip in labelled.items() if clip.line_id == line_id
        )
    orders = [g["clips"] for g in groups]
    assert any(order != sorted(order) for order in orders)
    reshuffled = listening.page_groups(labelled, LINES_DOC, seed=4)
    assert [g["clips"] for g in reshuffled] != orders
    assert listening.page_groups(labelled, LINES_DOC, seed=3) == groups


def test_the_page_shows_the_plain_line_without_turbos_tags(tmp_path):
    clips = _clips(tmp_path)
    groups = listening.page_groups(listening.assign_labels(clips, seed=1), LINES_DOC, seed=1)

    shown = " ".join(g["text"] for g in groups)

    assert "[sigh]" not in shown
    assert "[chuckle]" not in shown


# ---------------------------------------------------------------------------
# Loudness: two-pass loudnorm in the container, checked by a BS.1770 meter
# ---------------------------------------------------------------------------


def test_the_probe_runs_ffmpegs_two_pass_loudnorm_with_the_first_pass_measurements(tmp_path):
    speech = tmp_path / "speech"
    _write_wav(speech / "tts" / "qwen3" / "audio" / "idle_L01_r1.wav", TINY, 24000)
    out = tmp_path / "clips"
    out.mkdir()
    jobs = tmp_path / "jobs.json"
    jobs.write_text(
        json.dumps({"clips": [{"label": "c1", "src": "tts/qwen3/audio/idle_L01_r1.wav"}]}),
        encoding="utf-8",
    )
    ffmpeg = FakeFfmpeg()
    lines: list[str] = []

    code = listen_pack.main(
        ["--jobs", str(jobs), "--speech", str(speech), "--out", str(out)],
        run=ffmpeg,
        emit=lines.append,
    )

    assert code == 0
    first, second = ffmpeg.argvs
    src = str(speech / "tts" / "qwen3" / "audio" / "idle_L01_r1.wav")
    assert first[0] == "ffmpeg"
    assert first[first.index("-i") + 1] == src
    assert first[first.index("-af") + 1] == "loudnorm=I=-20:TP=-1.5:LRA=20:print_format=json"
    assert first[-3:] == ["-f", "null", "-"]
    assert second[second.index("-i") + 1] == src
    assert second[second.index("-af") + 1] == (
        "loudnorm=I=-20:TP=-1.5:LRA=20:measured_I=-27.61:measured_TP=-4.47:"
        "measured_LRA=6.06:measured_thresh=-38.20:offset=0.02:linear=true:print_format=json"
    )
    for flag, val in (("-ar", "48000"), ("-ac", "1"), ("-c:a", "pcm_s16le")):
        assert second[second.index(flag) + 1] == val
    assert "-map_metadata" in second
    assert "-n" in second
    assert second[-1] == str(out / "c1.wav")
    row = json.loads(next(line for line in lines if line.startswith("LOUDNESS ")).split(" ", 1)[1])
    assert row["label"] == "c1"
    assert row["input_i"] == -27.61
    assert row["normalization_type"] == "linear"
    assert abs(row["measured_lufs"] - (-20.0)) < 0.1


def test_the_loudnorm_report_is_read_from_ffmpegs_stderr():
    report = listen_pack.parse_loudnorm(_stderr(dict(PASS1_REPORT, input_i="-inf")))

    assert report["input_tp"] == -4.47
    assert report["input_i"] == float("-inf")
    assert report["normalization_type"] == "dynamic"
    with pytest.raises(listen_pack.FfmpegError):
        listen_pack.parse_loudnorm("no report here")


@pytest.mark.parametrize("bad", ["../x.wav", "/etc/x.wav", "listening/c1.wav"])
def test_the_probe_refuses_a_source_outside_the_saved_tts_clips(tmp_path, bad):
    out = tmp_path / "clips"
    out.mkdir()
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps({"clips": [{"label": "c1", "src": bad}]}), encoding="utf-8")
    ffmpeg = FakeFfmpeg()

    code = listen_pack.main(
        ["--jobs", str(jobs), "--speech", str(tmp_path), "--out", str(out)],
        run=ffmpeg,
        emit=lambda line: None,
    )

    assert code == 2
    assert ffmpeg.argvs == []


def test_k_weighting_matches_the_bs1770_coefficients_at_48_khz():
    shelf, highpass = listen_pack.k_weighting(48000)

    expected_shelf = (1.53512485958697, -2.69169618940638, 1.19839281085285)
    expected_shelf_a = (-1.69065929318241, 0.73248077421585)
    assert shelf.b == pytest.approx(expected_shelf, abs=1e-4)
    assert shelf.a == pytest.approx(expected_shelf_a, abs=1e-4)
    assert highpass.b == pytest.approx((1.0, -2.0, 1.0), abs=1e-9)
    assert highpass.a == pytest.approx((-1.99004745483398, 0.99007225036621), abs=1e-4)


@pytest.mark.parametrize("rate", [48000, 24000])
@pytest.mark.parametrize("amplitude", [1.0, 0.1, 0.01])
def test_integrated_loudness_of_a_997_hz_tone(rate, amplitude):
    samples = tone(amplitude, 2.0, rate)

    lufs = listen_pack.integrated_loudness(samples, rate)

    assert lufs == pytest.approx(20 * math.log10(amplitude) - 3.01, abs=0.1)


def test_gating_leaves_silence_out_of_the_integrated_loudness():
    rate = 48000
    samples = tone(0.1, 2.0, rate) + [0.0] * (2 * rate)

    lufs = listen_pack.integrated_loudness(samples, rate)

    # The absolute gate drops the 17 silent blocks; the 17 blocks inside the tone and the three
    # straddling its end (75%, 50% and 25% tone, all above the relative gate) remain.
    kept = 10 * math.log10((17 + 0.75 + 0.5 + 0.25) / 20)
    assert lufs == pytest.approx(-23.01 + kept, abs=0.02)
    ungated = 10 * math.log10((17 + 0.75 + 0.5 + 0.25) / 37)
    assert lufs - (-23.01 + ungated) > 2.5


def test_silence_has_no_integrated_loudness():
    assert listen_pack.integrated_loudness([0.0] * 48000, 48000) == float("-inf")


def test_the_target_tone_reads_minus_20_lufs_through_the_wav_reader(tmp_path):
    path = tmp_path / "t.wav"
    _write_wav(path, tone(TARGET_AMPLITUDE, 1.0, 48000), 48000)

    samples, rate = listen_pack.read_wav(path)

    assert rate == 48000
    assert listen_pack.integrated_loudness(samples, rate) == pytest.approx(-20.0, abs=0.1)


# ---------------------------------------------------------------------------
# The built pack: page folder, key folder, container argv
# ---------------------------------------------------------------------------


def test_the_built_pack_holds_only_the_page_and_labelled_clips_and_the_key_sits_outside(
    tmp_path,
):
    speech = speech_dir(tmp_path)
    build_tts_tree(speech, **SMALL)

    page = build(tmp_path)

    assert page == speech / "listening" / "index.html"
    files = sorted(p.relative_to(speech / "listening").as_posix() for p in page.parent.rglob("*"))
    key = json.loads((speech / "listening-key" / "key.json").read_text(encoding="utf-8"))
    labels = sorted(c["label"] for c in key["clips"])
    assert files == ["clips", *[f"clips/{label}.wav" for label in labels], "index.html"]
    assert not list(page.parent.rglob("key*"))
    assert key["seed"] == 7
    first = key["clips"][0]
    assert set(first) >= {"label", "engine", "setting", "voice", "line_id", "rep", "src"}
    assert {(c["engine"], c["setting"]) for c in key["clips"]} >= {("original", "expressive")}
    loud = (speech / "listening-key" / "loudness.jsonl").read_text(encoding="utf-8").splitlines()
    assert sorted(json.loads(r)["label"] for r in loud) == labels


def _forbidden_terms() -> set[str]:
    terms = set()
    for engine, cfg in TTS.items():
        terms |= {engine, *cfg["voices"], *cfg["settings"], cfg["image"]}
        repo = str(cfg["model"].get("repository", ""))
        terms |= {part for part in repo.split("/") if part}
        terms |= set(cfg["packages"])
    return terms | {"qwen", "chatterbox", "kyutai", "resemble"}


def test_no_page_text_or_file_name_names_an_engine_setting_or_voice(tmp_path):
    speech = speech_dir(tmp_path)
    build_tts_tree(speech, **SMALL)
    page = build(tmp_path)

    texts = [p.read_text(encoding="utf-8") for p in page.parent.rglob("*.html")]
    names = " ".join(p.name for p in page.parent.rglob("*"))
    template = listening.TEMPLATE_PATH.read_text(encoding="utf-8")

    for text in [*texts, names, template]:
        hits = [
            t
            for t in _forbidden_terms()
            if re.search(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", text, re.I)
        ]
        assert hits == []
    assert listening.blind_hits(texts[0], TTS) == []
    assert "/tts/" not in texts[0]


def test_the_blind_check_finds_an_engine_name_and_the_pack_refuses_to_write_it(tmp_path):
    speech = speech_dir(tmp_path)
    build_tts_tree(speech, **SMALL)
    doc = json.loads(json.dumps(LINES_DOC))
    doc["lines"][0]["feeling"] = "neutral, like the pocket one"

    assert listening.blind_hits("this is Aiden speaking", TTS) == ["aiden"]
    with pytest.raises(core.SpeechBenchError, match="pocket"):
        build(tmp_path, lines_doc=doc)
    assert not (speech / "listening").exists()
    assert not (speech / "listening-key").exists()


def test_the_page_lists_each_line_with_its_text_feeling_and_shuffled_clips(tmp_path):
    speech = speech_dir(tmp_path)
    build_tts_tree(speech, **SMALL)
    page = build(tmp_path)

    data = listening.page_data(page.read_text(encoding="utf-8"))

    assert [g["line_id"] for g in data["groups"]] == ["L01", "L03", "A01"]
    assert data["groups"][1]["text"] == TEXTS["L03"]
    assert data["groups"][1]["feeling"] == "tired, late at night"
    assert data["groups"][2]["feeling"] == listening.ACK_FEELING
    clip_count = sum(len(g["clips"]) for g in data["groups"])
    assert clip_count == len(list((page.parent / "clips").glob("*.wav")))


def test_ratings_csv_has_exactly_the_results_columns(tmp_path):
    speech = speech_dir(tmp_path)
    build_tts_tree(speech, **SMALL)
    page = build(tmp_path)
    template = listening.TEMPLATE_PATH.read_text(encoding="utf-8")

    data = listening.page_data(page.read_text(encoding="utf-8"))

    assert listening.RATINGS_COLUMNS == ("label", "line_id", "natural", "feeling", "preferred")
    assert data["columns"] == list(listening.RATINGS_COLUMNS)
    assert data["ratings_file"] == "ratings.csv"
    assert "DATA.columns.join(" in template
    assert "DATA.columns.map(" in template


def test_the_container_argv_runs_the_probe_offline_in_the_base_image_with_no_gpu():
    argv = listening.pack_argv(repo_dir="/repo", speech_dir="/r/mh1/speech")

    assert argv[:3] == ["docker", "run", "--rm"]
    assert argv[argv.index("--network") + 1] == "none"
    assert "--gpus" not in argv
    assert "-p" not in argv
    assert listening.IMAGE == "mist-speech-bench-base"
    assert argv[argv.index(listening.PROBE_MODULE) - 3 :][:3] == [listening.IMAGE, "python", "-m"]
    table = mounts(argv)
    assert table == {
        "/work": "/repo",
        "/speech/tts": "/r/mh1/speech/tts",
        "/jobs.json": "/r/mh1/speech/listening-key/jobs.json",
        "/out": "/r/mh1/speech/listening/clips",
    }
    specs = [argv[i + 1] for i, a in enumerate(argv) if a == "-v"]
    assert [s for s in specs if not s.endswith(":ro")] == ["/r/mh1/speech/listening/clips:/out"]


def test_plan_prints_the_listen_pack_argv_and_its_inputs_validate(capsys):
    assert listening.validate_inputs(_REPO_ROOT) == []
    blocks = listening.container_argv(core.build_plan_context(_REPO_ROOT))
    assert [label for label, _ in blocks] == ["listen-pack", "reply-ab", "reply-ab speech-service"]

    code = speech_bench.main(["plan"])

    out = capsys.readouterr().out
    assert code == 0
    assert "[argv] listen-pack: docker run" in out
    assert "[argv] reply-ab: docker run" in out
    assert f"{core.VOICE_CLIP_PLACEHOLDER}:/voice/identity.wav:ro" in out
    assert "plan ok" in out


def test_selftest_passes():
    listening.selftest()


def test_nothing_is_synthesized_and_no_engine_adapter_is_imported():
    for module in (listening, listen_pack):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "tts_engines" not in source
        assert "make_engine" not in source
    argv = listening.pack_argv(repo_dir="/repo", speech_dir="/s")
    assert "HF_HUB_OFFLINE=1" not in argv
    assert not any("models" in a for a in argv)


def test_a_failed_container_leaves_no_page(tmp_path):
    speech = speech_dir(tmp_path)
    build_tts_tree(speech, **SMALL)

    with pytest.raises(core.SpeechBenchError, match="exited 1"):
        build(tmp_path, runner=FakeRunner(code=1))

    assert not (speech / "listening" / "index.html").exists()


def test_a_clip_off_the_loudness_target_is_flagged(tmp_path):
    speech = speech_dir(tmp_path)
    build_tts_tree(speech, **SMALL)
    said: list[str] = []
    request = listening.PackRequest(results_root=tmp_path / "results", run="mh1", seed=1)

    listening.build_pack(
        request, runner=FakeRunner(FakeFfmpeg(amplitude=0.01)), git_sha="x", emit=said.append
    )

    rows = (speech / "listening-key" / "loudness.jsonl").read_text(encoding="utf-8").splitlines()
    assert all(json.loads(r)["on_target"] is False for r in rows)
    assert any("off the -20 LUFS target" in line for line in said)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_reading_the_key_is_refused_until_ratings_csv_exists(tmp_path):
    speech = speech_dir(tmp_path)
    build_tts_tree(speech, **SMALL)
    build(tmp_path)

    with pytest.raises(listening.KeySealedError) as refused:
        listening.read_key(speech)
    assert issubclass(listening.KeySealedError, core.SpeechBenchError)
    assert "ratings.csv" in str(refused.value)

    (speech / "listening" / "ratings.csv").write_text(
        ",".join(listening.RATINGS_COLUMNS) + "\n", encoding="utf-8"
    )
    key = listening.read_key(speech)
    assert {c["engine"] for c in key["clips"]} == set(TTS)


def test_listen_pack_show_key_is_refused_before_ratings_csv_exists(tmp_path, capsys):
    speech = speech_dir(tmp_path)
    build_tts_tree(speech, **SMALL)
    build(tmp_path)
    argv = ["listen-pack", "--results-root", str(tmp_path / "results"), "--run", "mh1"]

    code = speech_bench.main([*argv, "--show-key"])

    out = capsys.readouterr().out
    assert code == 1
    assert "ratings.csv" in out
    assert "qwen3" not in out
    (speech / "listening" / "ratings.csv").write_text("label\n", encoding="utf-8")
    assert speech_bench.main([*argv, "--show-key"]) == 0
    assert "qwen3" in capsys.readouterr().out


def test_listen_pack_refuses_an_existing_pack_before_writing(tmp_path):
    speech = speech_dir(tmp_path)
    build_tts_tree(speech, **SMALL)
    (speech / "listening-key").mkdir()

    with pytest.raises(core.OutputExistsError):
        build(tmp_path)
    assert not (speech / "listening").exists()
    assert list((speech / "listening-key").iterdir()) == []


def test_listen_pack_refuses_a_results_root_inside_the_repository():
    request = listening.PackRequest(results_root=_REPO_ROOT / "results", run="mh1", seed=1)

    with pytest.raises(core.ResultsRootError):
        listening.build_pack(request, runner=FakeRunner(), git_sha="x", emit=lambda line: None)


def test_listen_pack_cli_builds_the_pack_through_the_runner(tmp_path, monkeypatch, capsys):
    speech = speech_dir(tmp_path)
    build_tts_tree(speech, **SMALL)
    runner = FakeRunner()
    monkeypatch.setattr(listening, "default_runner", lambda: runner)

    code = speech_bench.main(
        ["listen-pack", "--results-root", str(tmp_path / "results"), "--run", "mh1"]
    )

    out = capsys.readouterr().out
    assert code == 0
    assert (speech / "listening" / "index.html").is_file()
    assert len(runner.argvs) == 1
    assert "qwen3" not in out
    key = json.loads((speech / "listening-key" / "key.json").read_text(encoding="utf-8"))
    assert isinstance(key["seed"], int)


# ---------------------------------------------------------------------------
# `--from-dir`: a blind pack of a directory's WAVs
# ---------------------------------------------------------------------------

DIR_NAMES = [f"cand-warm-s{s}.wav" for s in (1, 2, 3)] + [
    "cand-bright-s1.wav",
    "cand-bright-s2.wav",
]


def make_wav_dir(root: Path, names=DIR_NAMES) -> Path:
    for name in names:
        _write_wav(root / name, TINY, 24000)
    return root


class DirFakeRunner(FakeRunner):
    """Runs the probe with the speech root it would see: only the directory under `tts/`."""

    def __init__(self, stage: Path, **kwargs):
        super().__init__(**kwargs)
        self.stage = stage

    def run(self, argv, *, log_path, on_line):
        self.argvs.append(list(argv))
        Path(log_path).write_text("log\n", encoding="utf-8")
        if self.code is not None:
            return self.code
        table = mounts(argv)
        assert f"{listening.CONTAINER_SPEECH}/tts" not in table
        mounted = Path(table[f"{listening.CONTAINER_SPEECH}/tts/{listening.FROM_DIR_MOUNT}"])
        staged = self.stage / "tts" / listening.FROM_DIR_MOUNT
        staged.mkdir(parents=True)
        for wav in mounted.iterdir():
            staged.joinpath(wav.name).write_bytes(wav.read_bytes())
        tail = argv[argv.index(listening.PROBE_MODULE) + 1 :]
        mapped = []
        for i, arg in enumerate(tail):
            if i > 0 and tail[i - 1] in ("--jobs", "--out"):
                mapped.append(table[arg])
            elif i > 0 and tail[i - 1] == "--speech":
                mapped.append(str(self.stage))
            else:
                mapped.append(arg)
        return listen_pack.main(mapped, run=self.ffmpeg, emit=on_line)


def build_from_dir(tmp_path: Path, wav_dir: Path, *, seed: int = 7, runner=None, **kwargs) -> Path:
    request = listening.PackRequest(
        results_root=tmp_path / "results", run="h15", seed=seed, from_dir=wav_dir
    )
    return listening.build_pack(
        request,
        runner=runner or DirFakeRunner(tmp_path / f"stage-{seed}"),
        git_sha="abc123",
        emit=lambda line: None,
        **kwargs,
    )


def test_from_dir_builds_a_blind_pack_of_every_wav_with_the_key_outside_the_page(tmp_path):
    wav_dir = make_wav_dir(tmp_path / "candidates")
    (wav_dir / "notes.txt").write_text("not a clip", encoding="utf-8")
    speech = tmp_path / "results" / "h15" / "speech"

    page = build_from_dir(tmp_path, wav_dir)

    assert page == speech / "listening" / "index.html"
    key = json.loads((speech / "listening-key" / "key.json").read_text(encoding="utf-8"))
    labels = sorted(c["label"] for c in key["clips"])
    assert sorted(c["name"] for c in key["clips"]) == sorted(DIR_NAMES)
    files = sorted(p.relative_to(page.parent).as_posix() for p in page.parent.rglob("*"))
    assert files == ["clips", *[f"clips/{label}.wav" for label in labels], "index.html"]
    assert key["seed"] == 7
    assert key["from_dir"] == str(wav_dir.resolve())
    loud = (speech / "listening-key" / "loudness.jsonl").read_text(encoding="utf-8").splitlines()
    assert sorted(json.loads(r)["label"] for r in loud) == labels
    data = listening.page_data(page.read_text(encoding="utf-8"))
    assert sorted(label for g in data["groups"] for label in g["clips"]) == labels


def test_from_dir_labels_are_seeded_and_the_page_names_no_source_file(tmp_path):
    wav_dir = make_wav_dir(tmp_path / "candidates")
    built: list[int] = []

    def mapping(seed):
        built.append(seed)
        request = listening.PackRequest(
            results_root=tmp_path / f"results-{len(built)}",
            run="h15",
            seed=seed,
            from_dir=wav_dir,
        )
        page = listening.build_pack(
            request,
            runner=DirFakeRunner(request.results_root / "stage"),
            git_sha="x",
            emit=lambda line: None,
        )
        key = json.loads((page.parents[1] / "listening-key" / "key.json").read_text("utf-8"))
        return page, {c["label"]: c["name"] for c in key["clips"]}

    page, first = mapping(1)
    _, again = mapping(1)
    _, other = mapping(2)

    assert first == again
    assert first != other
    html = page.read_text(encoding="utf-8")
    for name in DIR_NAMES:
        assert Path(name).stem not in html
    assert "cand-" not in html
    assert listening.blind_hits(html, TTS) == []


def test_from_dir_runs_the_loudness_probe_with_only_the_directory_mounted_read_only(tmp_path):
    wav_dir = make_wav_dir(tmp_path / "candidates")
    runner = DirFakeRunner(tmp_path / "stage")

    build_from_dir(tmp_path, wav_dir, runner=runner)

    (argv,) = runner.argvs
    speech = (tmp_path / "results" / "h15" / "speech").as_posix()
    assert argv[argv.index("--network") + 1] == "none"
    assert "--gpus" not in argv
    assert argv[argv.index(listening.PROBE_MODULE) - 3 :][:3] == [listening.IMAGE, "python", "-m"]
    assert mounts(argv) == {
        "/work": _REPO_ROOT.resolve().as_posix(),
        f"/speech/tts/{listening.FROM_DIR_MOUNT}": wav_dir.resolve().as_posix(),
        "/jobs.json": f"{speech}/listening-key/jobs.json",
        "/out": f"{speech}/listening/clips",
    }
    specs = [argv[i + 1] for i, a in enumerate(argv) if a == "-v"]
    assert [s for s in specs if not s.endswith(":ro")] == [f"{speech}/listening/clips:/out"]
    assert len(runner.ffmpeg.argvs) == 2 * len(DIR_NAMES)


def test_from_dir_refuses_a_page_that_would_name_an_arm(tmp_path):
    wav_dir = make_wav_dir(tmp_path / "candidates")
    word = listening.FROM_DIR_TEXT.split()[-1].strip(".,").lower()
    arms = {"tts": {word: {"voices": [], "settings": {}, "model": {}}}}
    speech = tmp_path / "results" / "h15" / "speech"

    with pytest.raises(core.SpeechBenchError, match=word):
        build_from_dir(tmp_path, wav_dir, arms=arms)

    assert not (speech / "listening").exists()
    assert not (speech / "listening-key").exists()


def test_from_dir_refuses_a_directory_inside_the_repository(tmp_path):
    request = listening.PackRequest(
        results_root=tmp_path / "results",
        run="h15",
        seed=1,
        from_dir=_REPO_ROOT / "tests",
    )

    with pytest.raises(core.RepoPathError):
        listening.build_pack(request, runner=FakeRunner(), git_sha="x", emit=lambda line: None)
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("kind", ["missing", "empty"])
def test_from_dir_refuses_a_directory_with_no_wav_before_writing(tmp_path, kind):
    wav_dir = tmp_path / "candidates"
    if kind == "empty":
        wav_dir.mkdir()
        (wav_dir / "notes.txt").write_text("x", encoding="utf-8")

    with pytest.raises(core.SpeechBenchError):
        build_from_dir(tmp_path, wav_dir)
    assert not (tmp_path / "results" / "h15" / "speech" / "listening-key").exists()


def test_listen_pack_from_dir_cli_prints_no_source_file_name(tmp_path, monkeypatch, capsys):
    wav_dir = make_wav_dir(tmp_path / "candidates")
    runner = DirFakeRunner(tmp_path / "stage")
    monkeypatch.setattr(listening, "default_runner", lambda: runner)
    argv = ["listen-pack", "--results-root", str(tmp_path / "results"), "--run", "h15"]

    code = speech_bench.main([*argv, "--from-dir", str(wav_dir)])

    captured = capsys.readouterr()
    assert code == 0
    assert (tmp_path / "results" / "h15" / "speech" / "listening" / "index.html").is_file()
    assert "cand-" not in captured.out + captured.err
    assert f"{len(DIR_NAMES)} clips" in captured.out


# ---------------------------------------------------------------------------
# `reply-ab`: a blind A/B page of two `chain --reply-pack` directories
# ---------------------------------------------------------------------------

UTTERANCE_TEXTS = {u["id"]: u["text"] for u in core.load_json(core.UTTERANCES_PATH)["utterances"]}
A_REPLIES = {
    "U19": "N, E, C, E, S, S, A, R, Y. Two s's in the middle.",
    "U01": "Your ten minute timer is running now.",
    "U07": "One moment.",
}
B_REPLIES = {
    "U01": "Done, ten minutes from now.",
    "U07": "One moment.",
    "U19": "Necessary: N, E, C, E, S, S, A, R, Y.",
}
SMALL_ARMS = core.load_small_arms()


def make_reply_pack(root: Path, name: str, model: str, replies: dict, **meta) -> Path:
    """A directory in the layout `chain --reply-pack` writes; a list gives one reply per rep."""
    pack = root / name
    pack.mkdir(parents=True)
    doc = {
        "schema": 1,
        "run": "h17",
        "git_sha": "abc",
        "alias": f"{model}@gpu-full",
        "model": model,
        "placement": "gpu-full",
        "stt_engine": "nemotron-160",
        "tts_engine": "pocket",
        "reps": 1,
        "utterances": list(replies),
    }
    doc.update(meta)
    (pack / chain.REPLY_PACK_META).write_text(json.dumps(doc), encoding="utf-8")
    rows = []
    for uid, texts in replies.items():
        for rep, text in enumerate(texts if isinstance(texts, list) else [texts], start=1):
            error = None if text else "the model returned no text to speak"
            row = {"run": "h17", "suite": "chain", "git_sha": "abc", "utt_id": uid, "rep": rep}
            rows.append({**row, "kind": "command", "reply": text or None, "error": error})
    _jsonl(pack / chain.REPLY_PACK_FILE, rows)
    return pack


def reply_packs(tmp_path: Path, a=None, b=None) -> tuple[Path, Path]:
    pack_a = make_reply_pack(
        tmp_path, "pack-a", "e4b-ref", A_REPLIES if a is None else a, tts_engine="pocket-clone"
    )
    pack_b = make_reply_pack(
        tmp_path, "pack-b", "gemma4-e2b", B_REPLIES if b is None else b, tts_engine="qwen3"
    )
    return pack_a, pack_b


def build_reply(tmp_path: Path, *, seed: int = 7, packs=None, runner=None, voice=None, **kwargs):
    pack_a, pack_b = packs or reply_packs(tmp_path)
    request = listening.ReplyRequest(
        results_root=tmp_path / "results", run="h17", seed=seed, a=pack_a, b=pack_b
    )
    return listening.build_reply_ab(
        request,
        runner=runner or FakeRunner(code=99),
        voice=voice,
        git_sha="abc123",
        emit=kwargs.pop("emit", lambda line: None),
        **kwargs,
    )


def reply_speech(tmp_path: Path) -> Path:
    return tmp_path / "results" / "h17" / "speech"


def reply_key(tmp_path: Path) -> dict:
    path = reply_speech(tmp_path) / listening.REPLY_KEY_SUITE / listening.KEY_FILE
    return json.loads(path.read_text(encoding="utf-8"))


def test_reply_ab_builds_one_item_per_utterance_with_its_line_and_both_replies(tmp_path):
    page = build_reply(tmp_path)

    assert page == reply_speech(tmp_path) / "reply-ab" / "index.html"
    data = listening.page_data(page.read_text(encoding="utf-8"))
    # In `utterances.json` order, whatever order the packs hold them in.
    assert [item["item_id"] for item in data["items"]] == ["U01", "U07", "U19"]
    for item in data["items"]:
        uid = item["item_id"]
        assert item["text"] == UTTERANCE_TEXTS[uid]
        assert sorted(r["text"] for r in item["replies"]) == sorted(
            [A_REPLIES[uid], B_REPLIES[uid]]
        )
    labels = [r["label"] for item in data["items"] for r in item["replies"]]
    assert len(labels) == len(set(labels)) == 6
    assert data["voiced"] is False
    assert data["columns"] == list(listening.REPLY_RATINGS_COLUMNS)
    assert listening.REPLY_RATINGS_COLUMNS == ("label", "item_id", "quality", "preferred")
    assert data["ratings_file"] == "ratings.csv"


def test_reply_ab_labels_are_seeded_and_the_key_ties_each_label_to_its_side(tmp_path):
    def mapping(seed, where):
        packs = reply_packs(tmp_path / where)
        build_reply(tmp_path / where, seed=seed, packs=packs)
        key = reply_key(tmp_path / where)
        return {r["label"]: (r["item_id"], r["side"]) for r in key["replies"]}

    first, again, other = mapping(1, "one"), mapping(1, "two"), mapping(2, "three")

    assert first == again
    assert first != other
    assert sorted(first.values()) == sorted((u, s) for u in A_REPLIES for s in ("a", "b"))


def test_reply_ab_keeps_the_key_outside_the_page_folder(tmp_path):
    pack_a, pack_b = reply_packs(tmp_path)

    page = build_reply(tmp_path, packs=(pack_a, pack_b))

    files = sorted(p.relative_to(page.parent).as_posix() for p in page.parent.rglob("*"))
    assert files == ["index.html"]
    key = reply_key(tmp_path)
    assert key["seed"] == 7
    assert (key["a"]["dir"], key["b"]["dir"]) == (str(pack_a.resolve()), str(pack_b.resolve()))
    assert (key["a"]["meta"]["model"], key["b"]["meta"]["model"]) == ("e4b-ref", "gemma4-e2b")
    assert key["voice"] is None
    by_label = {r["label"]: r for r in key["replies"]}
    data = listening.page_data(page.read_text(encoding="utf-8"))
    for item in data["items"]:
        for reply in item["replies"]:
            side = by_label[reply["label"]]["side"]
            expected = (A_REPLIES if side == "a" else B_REPLIES)[item["item_id"]]
            assert reply["text"] == expected
            assert by_label[reply["label"]]["rep"] == 1


def _arm_terms() -> set[str]:
    terms = set(SMALL_ARMS["models"]) | set(SMALL_ARMS["placements"])
    for model in SMALL_ARMS["models"]:
        terms |= {f"{model}@{placement}" for placement in SMALL_ARMS["placements"]}
    for model in SMALL_ARMS["models"].values():
        terms |= {part for part in model["repository"].split("/") if part}
    terms |= set(ARMS["stt"]) | set(ARMS["tts"])
    return terms | {"pack-a", "pack-b", "gemma", "minicpm", "qwen"}


def test_the_reply_page_names_no_model_arm_or_placement(tmp_path):
    page = build_reply(tmp_path)
    html = page.read_text(encoding="utf-8")

    hits = [
        t
        for t in _arm_terms()
        if re.search(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", html, re.I)
    ]
    assert hits == []
    assert re.search(r"\barm\b", html, re.I) is None
    assert listening.blind_hits(listening.REPLY_TEMPLATE, TTS) == []


@pytest.mark.parametrize(
    "leak",
    ["gemma4-e2b", "gpu-full", "e4b-ref@gpu-full", "Gemma", "nemotron-160", "pocket-clone"],
)
def test_reply_ab_refuses_a_page_that_would_name_a_model_arm_or_placement(tmp_path, leak):
    packs = reply_packs(tmp_path, b={**B_REPLIES, "U01": f"Done; this is {leak} talking."})

    with pytest.raises(core.SpeechBenchError, match=re.escape(leak.lower())):
        build_reply(tmp_path, packs=packs)

    assert not (reply_speech(tmp_path) / "reply-ab").exists()
    assert not (reply_speech(tmp_path) / "reply-ab-key").exists()


def test_reply_ab_refuses_a_page_that_names_a_pack_s_own_model(tmp_path):
    a = make_reply_pack(tmp_path, "pack-a", "custom-tune", {"U01": "Timer set."})
    b = make_reply_pack(tmp_path, "pack-b", "gemma4-e2b", {"U01": "Custom-tune here, timer set."})

    with pytest.raises(core.SpeechBenchError, match="custom-tune"):
        build_reply(tmp_path, packs=(a, b))


def test_reply_ab_takes_each_pack_s_first_reply_and_skips_a_line_one_pack_lacks(tmp_path):
    said: list[str] = []
    a = {**A_REPLIES, "U01": ["", "Ten minutes, starting now."]}
    b = {k: v for k, v in B_REPLIES.items() if k != "U19"}
    packs = reply_packs(tmp_path, a=a, b=b)

    page = build_reply(tmp_path, packs=packs, emit=said.append)

    data = listening.page_data(page.read_text(encoding="utf-8"))
    assert [item["item_id"] for item in data["items"]] == ["U01", "U07"]
    assert "Ten minutes, starting now." in [r["text"] for r in data["items"][0]["replies"]]
    key = reply_key(tmp_path)
    assert {(r["item_id"], r["side"], r["rep"]) for r in key["replies"]} >= {("U01", "a", 2)}
    assert key["skipped"] == [{"item_id": "U19", "missing": ["b"]}]
    assert any("U19" in line and "[skip]" in line for line in said)


def test_reply_ab_refuses_a_pack_inside_the_checkout(tmp_path):
    _, pack_b = reply_packs(tmp_path)

    with pytest.raises(core.RepoPathError):
        build_reply(tmp_path, packs=(_REPO_ROOT / "tests", pack_b))
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("kind", ["missing", "not-a-pack"])
def test_reply_ab_refuses_a_directory_that_is_not_a_reply_pack(tmp_path, kind):
    pack_a, _ = reply_packs(tmp_path)
    other = tmp_path / "other"
    if kind == "not-a-pack":
        other.mkdir()

    with pytest.raises(core.InputError):
        build_reply(tmp_path, packs=(pack_a, other))
    assert not (tmp_path / "results").exists()


def test_reply_ab_refuses_an_existing_pack_before_writing(tmp_path):
    (reply_speech(tmp_path) / "reply-ab-key").mkdir(parents=True)

    with pytest.raises(core.OutputExistsError):
        build_reply(tmp_path)
    assert not (reply_speech(tmp_path) / "reply-ab").exists()


def test_reading_the_reply_key_is_refused_until_its_ratings_csv_exists(tmp_path):
    build_reply(tmp_path)
    speech = reply_speech(tmp_path)
    where = {"suite": listening.REPLY_SUITE, "key_suite": listening.REPLY_KEY_SUITE}

    with pytest.raises(listening.KeySealedError, match="ratings.csv"):
        listening.read_key(speech, **where)
    # The listening pack's ratings do not open the reply key.
    (speech / "listening").mkdir()
    (speech / "listening" / "ratings.csv").write_text("label\n", encoding="utf-8")
    with pytest.raises(listening.KeySealedError):
        listening.read_key(speech, **where)

    (speech / "reply-ab" / "ratings.csv").write_text("label\n", encoding="utf-8")
    assert listening.read_key(speech, **where)["a"]["meta"]["model"] == "e4b-ref"


def test_reply_ab_show_key_is_refused_before_ratings_csv_exists(tmp_path, capsys):
    build_reply(tmp_path)
    argv = ["reply-ab", "--results-root", str(tmp_path / "results"), "--run", "h17"]

    code = speech_bench.main([*argv, "--show-key"])

    out = capsys.readouterr().out
    assert code == 1
    assert "ratings.csv" in out and "e4b-ref" not in out
    (reply_speech(tmp_path) / "reply-ab" / "ratings.csv").write_text("label\n", encoding="utf-8")
    assert speech_bench.main([*argv, "--show-key"]) == 0
    assert "e4b-ref" in capsys.readouterr().out


def test_reply_ab_cli_builds_a_text_page_and_prints_no_model(tmp_path, monkeypatch, capsys):
    pack_a, pack_b = reply_packs(tmp_path)
    runner = FakeRunner(code=99)
    monkeypatch.setattr(listening, "default_runner", lambda: runner)
    argv = ["reply-ab", "--results-root", str(tmp_path / "results"), "--run", "h17"]

    code = speech_bench.main([*argv, "--a", str(pack_a), "--b", str(pack_b)])

    out = capsys.readouterr().out
    assert code == 0
    assert (reply_speech(tmp_path) / "reply-ab" / "index.html").is_file()
    assert runner.argvs == []
    assert "reply-ab ok: 3 items" in out
    assert "e4b-ref" not in out and "gemma4-e2b" not in out
    assert isinstance(reply_key(tmp_path)["seed"], int)


# --voice-clip: both replies voiced by `pocket-clone` with the one clip, loudness matched

VOICE_TRANSCRIPT = "the quartz heron counted nine amber lanterns"
CLIP_SHA = "0123456789ab"


def write_identity_clip(root: Path) -> Path:
    """A tone standing in for the identity clip, with a transcript beside it."""
    folder = root / "private-identity"
    _write_wav(folder / "identity.wav", tone(0.2, 0.1, 24000, 220.0), 24000)
    (folder / "clip.txt").write_text(VOICE_TRANSCRIPT, encoding="utf-8")
    return folder / "identity.wav"


class PcmReply:
    status = 200

    def __init__(self, chunks):
        self.headers = {"X-Sample-Rate": "24000"}
        self._chunks = list(chunks)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def read1(self, *_):
        return self._chunks.pop(0) if self._chunks else b""


class FakeSpeechService:
    """The speech service and its container: a short tone per text, and a log kept on stop."""

    def __init__(self):
        self.events: list[str] = []
        self.spoken: list[str] = []

    def start(self):
        self.events.append("start")

    def stop(self, log_dir):
        self.events.append("stop")
        (Path(log_dir) / "tts_service.log").write_text("loaded identity\n", encoding="utf-8")

    def __call__(self, request, timeout=None):
        self.spoken.append(json.loads(request.data.decode("utf-8"))["text"])
        return PcmReply([b"\x10\x00" * 2400, b"\x20\x00" * 2400])


def fake_voice(service: FakeSpeechService) -> listening.ReplyVoice:
    client = chain.TtsClient(chain.TTS_URL, opener=service)
    return listening.ReplyVoice(services=service, client=client, sha256_prefix=CLIP_SHA)


def test_voiced_replies_are_spoken_by_one_service_and_loudness_matched(tmp_path):
    service = FakeSpeechService()
    runner = DirFakeRunner(tmp_path / "stage")

    page = build_reply(tmp_path, runner=runner, voice=fake_voice(service))

    assert service.events == ["start", "stop"]
    assert sorted(service.spoken) == sorted([*A_REPLIES.values(), *B_REPLIES.values()])
    speech = reply_speech(tmp_path).as_posix()
    (argv,) = runner.argvs
    assert argv[argv.index("--network") + 1] == "none" and "--gpus" not in argv
    assert mounts(argv) == {
        "/work": _REPO_ROOT.resolve().as_posix(),
        f"/speech/tts/{listening.FROM_DIR_MOUNT}": f"{speech}/reply-ab-key/voiced",
        "/jobs.json": f"{speech}/reply-ab-key/jobs.json",
        "/out": f"{speech}/reply-ab/clips",
    }
    key = reply_key(tmp_path)
    labels = sorted(r["label"] for r in key["replies"])
    files = sorted(p.relative_to(page.parent).as_posix() for p in page.parent.rglob("*"))
    assert files == ["clips", *[f"clips/{label}.wav" for label in labels], "index.html"]
    key_dir = reply_speech(tmp_path) / "reply-ab-key"
    loud = (key_dir / "loudness.jsonl").read_text(encoding="utf-8").splitlines()
    assert sorted(json.loads(r)["label"] for r in loud) == labels
    assert all(json.loads(r)["on_target"] for r in loud)
    assert key["voice"] == {"engine": "pocket-clone", "voice_clip_sha256": CLIP_SHA}
    assert (key_dir / "tts_service.log").is_file()
    data = listening.page_data(page.read_text(encoding="utf-8"))
    assert data["voiced"] is True


def test_a_failed_voicing_leaves_no_page_and_stops_the_service(tmp_path):
    service = FakeSpeechService()

    def broken(request, timeout=None):
        raise OSError("connection reset")

    voice = listening.ReplyVoice(
        services=service, client=chain.TtsClient(chain.TTS_URL, opener=broken), sha256_prefix="x"
    )

    with pytest.raises(core.SpeechBenchError, match="connection reset"):
        build_reply(tmp_path, runner=DirFakeRunner(tmp_path / "stage"), voice=voice)

    assert service.events == ["start", "stop"]
    assert not (reply_speech(tmp_path) / "reply-ab" / "index.html").exists()


def test_the_reply_voice_is_pocket_clone_with_the_clip_mounted_read_only(tmp_path):
    clip = core.check_voice_clip(write_identity_clip(tmp_path))

    voice = listening.reply_voice(clip, speech_models_dir=tmp_path / "models")

    (spec,) = voice.services.specs
    argv = list(spec.argv)
    assert argv[argv.index("--engine") + 1] == core.VOICE_CLIP_ENGINE
    mounted = [argv[i + 1] for i, a in enumerate(argv) if a == "-v"]
    assert f"{clip.path.as_posix()}:/voice/identity.wav:ro" in mounted
    assert voice.client.url == f"{chain.TTS_URL}/speak"
    assert voice.sha256_prefix == clip.sha256_prefix


def test_reply_ab_cli_voices_with_the_clip_and_never_records_its_path(
    tmp_path, monkeypatch, capsys, caplog
):
    pack_a, pack_b = reply_packs(tmp_path)
    clip = write_identity_clip(tmp_path)
    service = FakeSpeechService()
    seen = {}

    def voice(checked, *, speech_models_dir):
        seen.update(clip=checked, models=speech_models_dir)
        return fake_voice(service)

    monkeypatch.setattr(listening, "reply_voice", voice)
    monkeypatch.setattr(listening, "default_runner", lambda: DirFakeRunner(tmp_path / "stage"))
    caplog.set_level("DEBUG")
    argv = ["reply-ab", "--results-root", str(tmp_path / "results"), "--run", "h17"]
    argv += ["--a", str(pack_a), "--b", str(pack_b), "--voice-clip", str(clip)]

    code = speech_bench.main([*argv, "--speech-models-dir", str(tmp_path / "models")])

    captured = capsys.readouterr()
    assert code == 0
    assert seen["clip"].path == clip.resolve()
    assert seen["models"] == tmp_path / "models"
    texts = [p.read_text(encoding="utf-8") for p in (tmp_path / "results").rglob("*.json*")]
    texts += [p.read_text(encoding="utf-8") for p in (tmp_path / "results").rglob("*.log")]
    texts += [p.read_text(encoding="utf-8") for p in (tmp_path / "results").rglob("*.html")]
    texts += [captured.out, captured.err, caplog.text]
    for text in texts:
        for secret in (str(clip), clip.as_posix(), clip.parent.name, VOICE_TRANSCRIPT):
            assert secret not in text
    assert sorted(p.name for p in clip.parent.iterdir()) == ["clip.txt", "identity.wav"]


@pytest.mark.parametrize("case", ["missing-clip", "not-a-wav"])
def test_reply_ab_voice_clip_misuse_exits_2_before_any_container_or_write(
    tmp_path, monkeypatch, capsys, case
):
    pack_a, pack_b = reply_packs(tmp_path)
    clip = write_identity_clip(tmp_path)
    given = clip.parent / ("absent.wav" if case == "missing-clip" else "clip.txt")
    monkeypatch.setattr(listening, "default_runner", lambda: FakeRunner(code=99))
    monkeypatch.setattr(listening, "reply_voice", lambda *a, **k: pytest.fail("voice built"))
    argv = ["reply-ab", "--results-root", str(tmp_path / "results"), "--run", "h17"]
    argv += ["--a", str(pack_a), "--b", str(pack_b), "--voice-clip", str(given)]

    code = speech_bench.main([*argv, "--speech-models-dir", str(tmp_path)])

    captured = capsys.readouterr()
    assert code == 2
    assert "[REFUSED]" in captured.out
    for secret in (str(given), clip.parent.name, VOICE_TRANSCRIPT):
        assert secret not in captured.out + captured.err
    assert not (tmp_path / "results").exists()
