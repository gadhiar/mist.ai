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
from scripts.model_bench.speech import core, listening  # noqa: E402
from scripts.model_bench.speech.probes import listen_pack, tts_probe  # noqa: E402

ARMS = core.load_json(core.SPEECH_ARMS_PATH)
TTS = ARMS["tts"]
LINES_DOC = core.load_json(core.LINES_PATH)
LINE_IDS = [line["id"] for line in LINES_DOC["lines"]]
ACK_IDS = [ack["id"] for ack in LINES_DOC["acknowledgements"]]
EMOTIVE = [line["id"] for line in LINES_DOC["lines"] if line["emotive"]]
TEXTS = {item["id"]: item["text"] for item in [*LINES_DOC["lines"], *LINES_DOC["acknowledgements"]]}
TIMING = {engine: cfg["timing_voice"] for engine, cfg in TTS.items()}


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


def test_the_pack_takes_repeat_1_of_each_line_from_five_settings_in_each_timing_voice(tmp_path):
    speech = speech_dir(tmp_path)
    build_tts_tree(speech)

    clips = listening.collect_clips(speech, TTS, LINES_DOC)

    timed = [c for c in clips if c.condition is not None]
    assert {(c.engine, c.setting, c.voice) for c in timed} == {
        ("qwen3", "default", "aiden"),
        ("turbo", "default", "builtin"),
        ("original", "default", "builtin"),
        ("pocket", "default", "alba"),
    }
    assert {c.rep for c in clips} == {1}
    assert {c.condition for c in timed} == {"idle"}
    for engine in TTS:
        got = sorted(c.line_id for c in timed if c.engine == engine)
        assert got == sorted(LINE_IDS + ACK_IDS)
    expressive = [c for c in clips if c.setting == "expressive"]
    assert sorted(c.line_id for c in expressive) == sorted(EMOTIVE)
    assert {(c.engine, c.voice) for c in expressive} == {("original", "builtin")}
    settings = {(c.engine, c.setting) for c in clips}
    assert settings == {
        ("qwen3", "default"),
        ("turbo", "default"),
        ("original", "default"),
        ("original", "expressive"),
        ("pocket", "default"),
    }


def test_the_pack_includes_every_listening_only_and_extra_voice_clip(tmp_path):
    speech = speech_dir(tmp_path)
    build_tts_tree(speech, extra_voices={"qwen3": ["eric"], "pocket": ["marius"]})

    clips = listening.collect_clips(speech, TTS, LINES_DOC)

    untimed = [c for c in clips if c.condition is None and c.setting == "default"]
    by_voice = {}
    for clip in untimed:
        by_voice.setdefault((clip.engine, clip.voice), []).append(clip.line_id)
    expected = {(e, v) for e, cfg in TTS.items() for v in cfg["listening_voices"]}
    expected |= {("qwen3", "eric"), ("pocket", "marius")}
    assert set(by_voice) == expected
    assert all(sorted(ids) == sorted(LINE_IDS) for ids in by_voice.values())
    timed = 4 * (len(LINE_IDS) + len(ACK_IDS))
    listening_only = sum(len(cfg["listening_voices"]) for cfg in TTS.values()) + 2
    assert len(clips) == timed + len(EMOTIVE) + listening_only * len(LINE_IDS)


def test_every_clip_the_pack_takes_is_a_file_tts_already_saved(tmp_path):
    speech = speech_dir(tmp_path)
    build_tts_tree(speech)
    saved = {p.relative_to(speech).as_posix() for p in (speech / "tts").rglob("*.wav")}

    clips = listening.collect_clips(speech, TTS, LINES_DOC)

    assert {c.src for c in clips} <= saved


def test_the_pack_refuses_an_engine_tts_has_not_saved(tmp_path):
    speech = speech_dir(tmp_path)
    build_tts_tree(speech, engines=["qwen3", "turbo", "original"])

    with pytest.raises(core.SpeechBenchError, match="pocket"):
        listening.collect_clips(speech, TTS, LINES_DOC)


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
    assert [label for label, _ in blocks] == ["listen-pack"]

    code = speech_bench.main(["plan"])

    out = capsys.readouterr().out
    assert code == 0
    assert "[argv] listen-pack: docker run" in out
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
