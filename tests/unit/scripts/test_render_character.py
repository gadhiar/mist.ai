"""Tests for `scripts.render_character`: one WAV through her character stage, offline.

Raj's blind check renders private clips with this script in a throwaway container, with the
checkout mounted read-only, so it must never read or write inside the checkout, never overwrite
a render, and refuse audio the stage was not built for. Every WAV here is a synthesized vowel
written by `wave` into pytest's temporary directory.
"""

import wave
from pathlib import Path

import numpy as np
import pytest

from backend.realtime.budget import CharacterConfig
from backend.realtime.character import CharacterStage
from scripts import render_character

FS = 24000
REPO_ROOT = Path(render_character.__file__).resolve().parents[1]


def _vowel_pcm(seconds: float) -> bytes:
    t = np.arange(round(seconds * FS)) / FS
    x = sum(np.sin(2 * np.pi * k * 200.0 * t) / k**2 for k in range(1, 56))
    return np.rint(0.2 * x * 32768).astype("<i2").tobytes()


def _write_wav(
    path: Path, pcm: bytes, *, rate: int = FS, channels: int = 1, width: int = 2
) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        handle.writeframes(pcm)
    return path


def _read_wav(path: Path) -> tuple[tuple[int, int, int], bytes]:
    """(channels, sample width, rate) and the frames."""
    with wave.open(str(path), "rb") as handle:
        shape = (handle.getnchannels(), handle.getsampwidth(), handle.getframerate())
        return shape, handle.readframes(handle.getnframes())


def test_renders_the_stage_in_the_given_chunks_then_the_tail(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pcm = _vowel_pcm(1.3)
    source = _write_wav(tmp_path / "in.wav", pcm)
    target = tmp_path / "out.wav"

    code = render_character.main(
        ["--in", str(source), "--out", str(target), "--chunk-ms", "80", "--double-gain", "0.5"]
    )

    stage = CharacterStage(CharacterConfig(double_gain=0.5), sample_rate=FS)
    chunk = 2 * 80 * FS // 1000
    expected = b"".join(stage.process(pcm[i : i + chunk]) for i in range(0, len(pcm), chunk))
    expected += stage.end_segment()
    shape, frames = _read_wav(target)
    assert code == 0
    assert shape == (1, 2, FS)
    assert frames == expected
    printed = capsys.readouterr().out
    assert str(tmp_path) not in printed
    assert "in.wav" not in printed and "out.wav" not in printed
    assert f"{len(pcm) // 2}" in printed
    assert f"{len(expected) // 2}" in printed


def test_default_chunking_is_80_ms_with_the_default_gain(tmp_path: Path) -> None:
    pcm = _vowel_pcm(0.5)
    source = _write_wav(tmp_path / "in.wav", pcm)
    target = tmp_path / "out.wav"

    code = render_character.main(["--in", str(source), "--out", str(target)])

    stage = CharacterStage(CharacterConfig(), sample_rate=FS)
    chunk = 2 * 1920
    expected = b"".join(stage.process(pcm[i : i + chunk]) for i in range(0, len(pcm), chunk))
    expected += stage.end_segment()
    assert code == 0
    assert _read_wav(target)[1] == expected


def test_refuses_an_input_inside_the_checkout(tmp_path: Path) -> None:
    target = tmp_path / "out.wav"

    code = render_character.main(
        ["--in", str(REPO_ROOT / "tests" / "absent.wav"), "--out", str(target)]
    )

    assert code == 2
    assert not target.exists()


def test_refuses_an_output_inside_the_checkout(tmp_path: Path) -> None:
    source = _write_wav(tmp_path / "in.wav", _vowel_pcm(0.2))
    target = REPO_ROOT / "render_character_refused.wav"

    code = render_character.main(["--in", str(source), "--out", str(target)])

    assert code == 2
    assert not target.exists()


def test_refuses_an_existing_output(tmp_path: Path) -> None:
    source = _write_wav(tmp_path / "in.wav", _vowel_pcm(0.2))
    target = tmp_path / "out.wav"
    target.write_bytes(b"kept")

    code = render_character.main(["--in", str(source), "--out", str(target)])

    assert code == 2
    assert target.read_bytes() == b"kept"


@pytest.mark.parametrize(
    "params",
    [
        pytest.param({"rate": 16000}, id="16kHz"),
        pytest.param({"channels": 2}, id="stereo"),
        pytest.param({"width": 1}, id="8-bit"),
    ],
)
def test_refuses_audio_that_is_not_24khz_mono_pcm16(tmp_path: Path, params: dict) -> None:
    source = _write_wav(tmp_path / "in.wav", _vowel_pcm(0.2), **params)
    target = tmp_path / "out.wav"

    code = render_character.main(["--in", str(source), "--out", str(target)])

    assert code == 2
    assert not target.exists()


def test_refuses_a_file_that_is_not_a_wav(tmp_path: Path) -> None:
    source = tmp_path / "in.wav"
    source.write_bytes(b"not a wav file")
    target = tmp_path / "out.wav"

    code = render_character.main(["--in", str(source), "--out", str(target)])

    assert code == 2
    assert not target.exists()


@pytest.mark.parametrize("gain", ["1.5", "-0.1"])
def test_refuses_a_double_gain_outside_0_to_1(tmp_path: Path, gain: str) -> None:
    source = _write_wav(tmp_path / "in.wav", _vowel_pcm(0.2))
    target = tmp_path / "out.wav"

    code = render_character.main(["--in", str(source), "--out", str(target), "--double-gain", gain])

    assert code == 2
    assert not target.exists()
