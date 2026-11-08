"""`prepare`: converting the recordings, its refusals, the manifest and its container argv.

Recordings here are synthetic tones written in the test (`*.wav` is gitignored and no audio is
committed); the real material directory is never read. Most tests inject a converter that copies
the 16 kHz WAV through, so they need no ffmpeg; one runs the real ffmpeg when it is on the path.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import wave
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from scripts.model_bench import speech_bench  # noqa: E402
from scripts.model_bench.speech import core, stt_suite  # noqa: E402
from scripts.model_bench.speech.probes import prepare_material  # noqa: E402

SR = 16000
UTTERANCES = [
    {"id": "U01", "kind": "complete", "text": "Set a timer for ten minutes."},
    {"id": "P01", "kind": "pause", "text": "Can you remind me to ... pick up the parcel tomorrow?"},
]


def _tone(seconds: float, trailing: float, *, lead: float = 0.3, seed: int = 0) -> np.ndarray:
    """Room noise, a tone for the words, then `trailing` seconds of room after the last word."""
    rng = np.random.default_rng(seed)
    n_tone = int(round((seconds - lead - trailing) * SR))
    t = np.arange(n_tone) / SR
    return np.concatenate(
        [
            rng.normal(0, 0.001, int(round(lead * SR))),
            0.3 * np.sin(2 * np.pi * 180 * t),
            rng.normal(0, 0.001, int(round(trailing * SR))),
        ]
    )


def _write_wav(path: Path, samples: np.ndarray, *, rate: int = SR, channels: int = 1) -> None:
    data = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    if channels > 1:
        data = np.repeat(data[:, None], channels, axis=1).reshape(-1)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(data.tobytes())


def _copy(src: Path, dst: Path) -> None:
    shutil.copyfile(src, dst)


@pytest.fixture()
def dirs(tmp_path):
    recordings = tmp_path / "material" / "recordings"
    recordings.mkdir(parents=True)
    return recordings, tmp_path / "material"


def _record(recordings: Path, **durations: tuple[float, float]) -> None:
    for uid, (seconds, trailing) in durations.items():
        _write_wav(recordings / f"{uid}.wav", _tone(seconds, trailing))


# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------


def test_prepare_writes_16_khz_mono_wavs_and_a_manifest_verify_material_accepts(dirs):
    recordings, out = dirs
    _record(recordings, U01=(3.0, 0.5), P01=(5.0, 1.0))

    items = prepare_material.prepare(recordings, out, UTTERANCES, convert=_copy)

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert [i["id"] for i in manifest["items"]] == ["U01", "P01"]
    first = manifest["items"][0]
    assert first["file"] == "U01.wav"
    assert first["kind"] == "complete"
    assert first["text"] == "Set a timer for ten minutes."
    assert first["duration_s"] == pytest.approx(3.0)
    assert first["sha256"] == core.sha256_file(out / "U01.wav")
    assert first["trailing_silence_ms"] == 500
    assert first["source"] == "U01.wav"
    assert manifest["sample_rate"] == SR
    assert items == manifest["items"]
    verified = core.verify_material(out / "manifest.json")
    assert [i.id for i in verified.items] == ["U01", "P01"]
    with wave.open(str(out / "P01.wav"), "rb") as handle:
        assert (handle.getframerate(), handle.getnchannels(), handle.getsampwidth()) == (SR, 1, 2)


def test_the_manifest_lists_utterances_in_script_order_whatever_the_file_order(dirs):
    recordings, out = dirs
    _record(recordings, P01=(3.0, 0.5), U01=(3.0, 0.5))

    prepare_material.prepare(recordings, out, UTTERANCES, convert=_copy)

    ids = [i["id"] for i in json.loads((out / "manifest.json").read_text())["items"]]
    assert ids == ["U01", "P01"]


# ---------------------------------------------------------------------------
# Refusals: nothing is written when any recording is refused
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("seconds", "trailing", "message"),
    [
        pytest.param(1.5, 0.5, "outside 2 to 8 s", id="too-short"),
        pytest.param(8.5, 0.5, "outside 2 to 8 s", id="too-long"),
        pytest.param(3.0, 0.1, "under 300 ms of silence", id="trailing-silence"),
    ],
)
def test_prepare_refuses_a_bad_recording_and_writes_nothing(dirs, seconds, trailing, message):
    recordings, out = dirs
    _record(recordings, U01=(3.0, 0.5), P01=(seconds, trailing))

    with pytest.raises(prepare_material.PrepareError) as info:
        prepare_material.prepare(recordings, out, UTTERANCES, convert=_copy)

    assert message in str(info.value)
    assert "P01" in str(info.value)
    assert sorted(p.name for p in out.iterdir()) == ["recordings"]


def test_two_and_eight_seconds_are_both_accepted(dirs):
    recordings, out = dirs
    _record(recordings, U01=(2.0, 0.4), P01=(8.0, 0.4))

    items = prepare_material.prepare(recordings, out, UTTERANCES, convert=_copy)

    assert [i["duration_s"] for i in items] == [pytest.approx(2.0), pytest.approx(8.0)]


def test_prepare_refuses_a_recording_with_no_speech(dirs):
    recordings, out = dirs
    _record(recordings, U01=(3.0, 0.5))
    rng = np.random.default_rng(3)
    _write_wav(recordings / "P01.wav", rng.normal(0, 0.001, 3 * SR))

    with pytest.raises(prepare_material.PrepareError, match="no speech"):
        prepare_material.prepare(recordings, out, UTTERANCES, convert=_copy)


def test_prepare_refuses_a_missing_recording(dirs):
    recordings, out = dirs
    _record(recordings, U01=(3.0, 0.5))

    with pytest.raises(prepare_material.PrepareError, match="P01: no recording"):
        prepare_material.prepare(recordings, out, UTTERANCES, convert=_copy)


def test_prepare_refuses_a_file_not_named_by_an_utterance_id(dirs):
    recordings, out = dirs
    _record(recordings, U01=(3.0, 0.5), P01=(3.0, 0.5), X99=(3.0, 0.5))

    with pytest.raises(prepare_material.PrepareError, match="X99.wav"):
        prepare_material.prepare(recordings, out, UTTERANCES, convert=_copy)


def test_prepare_refuses_two_recordings_of_one_utterance(dirs):
    recordings, out = dirs
    _record(recordings, U01=(3.0, 0.5), P01=(3.0, 0.5))
    (recordings / "U01.m4a").write_bytes(b"m4a")

    with pytest.raises(prepare_material.PrepareError, match="U01: more than one recording"):
        prepare_material.prepare(recordings, out, UTTERANCES, convert=_copy)


def test_prepare_refuses_an_existing_manifest(dirs):
    recordings, out = dirs
    _record(recordings, U01=(3.0, 0.5), P01=(3.0, 0.5))
    (out / "manifest.json").write_text("{}")

    with pytest.raises(core.OutputExistsError):
        prepare_material.prepare(recordings, out, UTTERANCES, convert=_copy)

    assert not (out / "U01.wav").exists()


def test_prepare_refuses_a_converted_file_that_is_not_16_khz_mono(dirs):
    recordings, out = dirs
    _record(recordings, U01=(3.0, 0.5))
    _write_wav(recordings / "P01.wav", _tone(3.0, 0.5), rate=22050)

    with pytest.raises(prepare_material.PrepareError, match="P01.*16000"):
        prepare_material.prepare(recordings, out, UTTERANCES, convert=_copy)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not on the path")
def test_ffmpeg_converts_a_44_1_khz_stereo_recording_to_16_khz_mono_16_bit(dirs):
    recordings, out = dirs
    rate = 44100
    t = np.arange(int(rate * 2.4)) / rate
    tone = 0.3 * np.sin(2 * np.pi * 180 * t)
    tail = int(rate * 1.6)
    tone[tail:] = np.random.default_rng(5).normal(0, 0.001, len(tone) - tail)
    _write_wav(recordings / "U01.wav", tone, rate=rate, channels=2)

    items = prepare_material.prepare(recordings, out, UTTERANCES[:1])

    with wave.open(str(out / "U01.wav"), "rb") as handle:
        assert (handle.getframerate(), handle.getnchannels(), handle.getsampwidth()) == (SR, 1, 2)
    assert items[0]["duration_s"] == pytest.approx(2.4, abs=0.01)
    assert items[0]["trailing_silence_ms"] == pytest.approx(800, abs=20)


def test_the_ffmpeg_argv_writes_bit_exact_16_khz_mono_pcm():
    argv = prepare_material.ffmpeg_argv(Path("/in/U01.m4a"), Path("/tmp/U01.wav"))

    assert argv[0] == "ffmpeg"
    assert argv[argv.index("-i") + 1] == "/in/U01.m4a"
    assert argv[argv.index("-ac") + 1] == "1"
    assert argv[argv.index("-ar") + 1] == "16000"
    assert argv[argv.index("-c:a") + 1] == "pcm_s16le"
    assert "-nostdin" in argv
    assert argv[-1] == "/tmp/U01.wav"


def test_the_probe_parser_takes_no_voice_sample():
    options = {o for a in prepare_material.build_parser()._actions for o in a.option_strings}

    assert options == {"-h", "--help", "--recordings", "--out", "--utterances"}


# ---------------------------------------------------------------------------
# The host side: argv and refusals before docker
# ---------------------------------------------------------------------------


def _pairs(argv, flag):
    return [argv[i + 1] for i, a in enumerate(argv) if a == flag]


def test_prepare_container_mounts_the_recordings_read_only_and_only_the_output_writable():
    argv = stt_suite.prepare_container_argv(repo_dir="D:/repo", material_dir="D:/material")
    image = argv.index("mist-speech-bench-base")
    docker_part, tail = argv[:image], argv[image:]

    assert argv[:3] == ["docker", "run", "--rm"]
    assert "--gpus" not in argv
    assert _pairs(docker_part, "--network") == ["none"]
    assert _pairs(docker_part, "-v") == [
        "D:/repo:/work:ro",
        "D:/material:/material",
        "D:/material/recordings:/material/recordings:ro",
    ]
    assert "HF_HUB_OFFLINE=1" in _pairs(docker_part, "-e")
    assert tail[1:4] == ["python", "-m", "scripts.model_bench.speech.probes.prepare_material"]
    assert _pairs(tail, "--recordings") == ["/material/recordings"]
    assert _pairs(tail, "--out") == ["/material"]


def test_prepare_refuses_a_material_directory_inside_the_repository(monkeypatch, capsys):
    monkeypatch.setattr(stt_suite, "run_container", lambda argv: pytest.fail("ran docker"))

    code = speech_bench.main(["prepare", "--material-dir", str(_REPO_ROOT / "material")])

    assert code != 0
    assert "inside the repository" in capsys.readouterr().out


def test_prepare_refuses_a_material_directory_without_recordings(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(stt_suite, "run_container", lambda argv: pytest.fail("ran docker"))

    code = speech_bench.main(["prepare", "--material-dir", str(tmp_path)])

    assert code != 0
    assert "recordings" in capsys.readouterr().out


def test_prepare_refuses_an_existing_manifest_before_docker(dirs, monkeypatch, capsys):
    _, material = dirs
    (material / "manifest.json").write_text("{}")
    monkeypatch.setattr(stt_suite, "run_container", lambda argv: pytest.fail("ran docker"))

    code = speech_bench.main(["prepare", "--material-dir", str(material)])

    assert code != 0
    assert "already exists" in capsys.readouterr().out


def test_prepare_runs_one_container_and_checks_the_manifest_it_wrote(dirs, monkeypatch, capsys):
    recordings, material = dirs
    _record(recordings, U01=(3.0, 0.5), P01=(3.0, 0.5))
    argvs = []

    def runner(argv):
        argvs.append(argv)
        prepare_material.prepare(recordings, material, UTTERANCES, convert=_copy)
        return subprocess.CompletedProcess(argv, 0, "[ok] 2 recordings\n", "")

    monkeypatch.setattr(stt_suite, "run_container", runner)

    code = speech_bench.main(["prepare", "--material-dir", str(material)])

    out = capsys.readouterr().out
    assert code == 0, out
    assert len(argvs) == 1
    assert "2 recordings verified" in out


def test_prepare_passes_on_the_containers_refusal(dirs, monkeypatch, capsys):
    _, material = dirs

    def runner(argv):
        return subprocess.CompletedProcess(argv, 2, "[FAIL] P01: under 300 ms\n", "")

    monkeypatch.setattr(stt_suite, "run_container", runner)

    code = speech_bench.main(["prepare", "--material-dir", str(material)])

    out = capsys.readouterr().out
    assert code != 0
    assert "P01: under 300 ms" in out
