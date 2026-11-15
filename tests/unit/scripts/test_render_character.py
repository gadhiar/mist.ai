"""Tests for `scripts.render_character`: one WAV through her character stage, offline.

Raj's blind check renders private clips with this script in a throwaway container, with the
checkout mounted read-only, so it must never read or write inside the checkout, never overwrite
a render, and refuse audio the stage was not built for. Every WAV here is a synthesized vowel
written by `wave` into pytest's temporary directory.
"""

import math
import wave
from pathlib import Path

import numpy as np
import pytest

from backend.realtime.budget import CharacterConfig, LimiterConfig
from backend.realtime.character import CharacterStage, limiter_reach_samples
from scripts import render_character

FS = 24000
REPO_ROOT = Path(render_character.__file__).resolve().parents[1]
KNEE_STEPS = math.floor(32768 * 10 ** (LimiterConfig().knee_dbfs / 20))  # 29204
COMPARE_KEYS = [
    "compared samples",
    "earlier samples at or above the knee",
    "earlier samples at full scale",
    "differing samples",
    "differing samples outside the limiter's reach",
    "largest difference",
    "limited regions",
    "worst region change",
]


def _vowel_pcm(seconds: float) -> bytes:
    t = np.arange(round(seconds * FS)) / FS
    x = sum(np.sin(2 * np.pi * k * 200.0 * t) / k**2 for k in range(1, 56))
    return np.rint(0.2 * x * 32768).astype("<i2").tobytes()


def _overdriven_pcm() -> bytes:
    """Four abrupt 200 ms syllables of the vowel with 50 ms gaps: each onset passes full scale."""
    t = np.arange(round(1.0 * FS)) / FS
    x = sum(np.sin(2 * np.pi * k * 200.0 * t) / k**2 for k in range(1, 56))
    envelope = (t % 0.25) < 0.2
    return np.rint(0.3 * x * envelope * 32768).astype("<i2").tobytes()


def _lines(printed: str) -> dict[str, str]:
    return dict(line.split(": ", 1) for line in printed.splitlines())


def _samples(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.int64)


def _expected_comparison(after: np.ndarray, before: np.ndarray, reach: int) -> dict[str, str]:
    """The comparison's lines, computed here sample by sample from their definitions."""
    loud = [i for i in range(len(before)) if abs(before[i]) >= KNEE_STEPS]
    within = np.zeros(len(before), dtype=bool)
    for i in loud:
        within[i : i + reach] = True
    differs = after != before
    regions = []
    start = None
    for i, inside in enumerate([*within, False]):
        if inside and start is None:
            start = i
        elif not inside and start is not None:
            regions.append((start, i))
            start = None
    changes = []
    for lo, hi in regions:
        delta = after[lo:hi] - before[lo:hi]
        if np.any(delta):
            energy = np.sum(np.square(before[lo:hi].astype(float)))
            changes.append(10 * math.log10(np.sum(np.square(delta.astype(float))) / energy))
    return {
        "compared samples": str(len(before)),
        "earlier samples at or above the knee": str(len(loud)),
        "earlier samples at full scale": str(int(np.sum(np.abs(before) >= 32767))),
        "differing samples": str(int(np.sum(differs))),
        "differing samples outside the limiter's reach": str(int(np.sum(differs & ~within))),
        "largest difference": f"{int(np.max(np.abs(after - before)))} LSB",
        "limited regions": str(len(regions)),
        "worst region change": f"{max(changes):.1f} dB" if changes else "none",
    }


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


def _stage_render(pcm: bytes, config: CharacterConfig) -> tuple[bytes, CharacterStage]:
    stage = CharacterStage(config, sample_rate=FS)
    chunk = 2 * 1920
    out = b"".join(stage.process(pcm[i : i + chunk]) for i in range(0, len(pcm), chunk))
    return out + stage.end_segment(), stage


def test_prints_the_limited_count_after_the_clip_count(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pcm = _overdriven_pcm()
    source = _write_wav(tmp_path / "in.wav", pcm)

    code = render_character.main(["--in", str(source), "--out", str(tmp_path / "out.wav")])

    expected, stage = _stage_render(pcm, CharacterConfig())
    printed = capsys.readouterr().out.splitlines()
    assert code == 0
    assert _read_wav(tmp_path / "out.wav")[1] == expected
    assert stage.limited_samples > 0
    assert printed == [
        f"input samples: {len(pcm) // 2}",
        f"output samples: {len(expected) // 2}",
        "clipped samples: 0",
        f"limited samples: {stage.limited_samples}",
    ]


def test_no_limiter_renders_the_stage_without_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pcm = _overdriven_pcm()
    source = _write_wav(tmp_path / "in.wav", pcm)
    target = tmp_path / "out.wav"

    code = render_character.main(
        ["--in", str(source), "--out", str(target), "--double-gain", "0.5", "--no-limiter"]
    )

    expected, stage = _stage_render(pcm, CharacterConfig(double_gain=0.5, limiter=None))
    lines = _lines(capsys.readouterr().out)
    assert code == 0
    assert _read_wav(target)[1] == expected
    assert stage.clipped_samples > 0
    assert lines["clipped samples"] == str(stage.clipped_samples)
    assert lines["limited samples"] == "0"


def test_the_limiter_changes_an_earlier_render_only_within_its_reach(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _write_wav(tmp_path / "in.wav", _overdriven_pcm())
    earlier = tmp_path / "earlier.wav"
    assert render_character.main(["--in", str(source), "--out", str(earlier), "--no-limiter"]) == 0
    capsys.readouterr()
    target = tmp_path / "limited.wav"

    code = render_character.main(
        ["--in", str(source), "--out", str(target), "--compare-to", str(earlier)]
    )

    printed = capsys.readouterr().out
    lines = _lines(printed)
    after, before = _samples(_read_wav(target)[1]), _samples(_read_wav(earlier)[1])
    reach = limiter_reach_samples(LimiterConfig(), sample_rate=FS)
    assert code == 0
    assert lines["clipped samples"] == "0"
    assert int(lines["earlier samples at full scale"]) > 0
    assert int(lines["differing samples"]) > 0
    assert lines["differing samples outside the limiter's reach"] == "0"
    assert [line.split(": ")[0] for line in printed.splitlines()[4:]] == COMPARE_KEYS
    assert {key: lines[key] for key in COMPARE_KEYS} == _expected_comparison(after, before, reach)
    assert str(tmp_path) not in printed
    assert "wav" not in printed


def test_a_render_without_the_limiter_matches_an_earlier_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _write_wav(tmp_path / "in.wav", _overdriven_pcm())
    earlier = tmp_path / "earlier.wav"
    assert render_character.main(["--in", str(source), "--out", str(earlier), "--no-limiter"]) == 0
    capsys.readouterr()

    code = render_character.main(
        [
            "--in",
            str(source),
            "--out",
            str(tmp_path / "again.wav"),
            "--no-limiter",
            "--compare-to",
            str(earlier),
        ]
    )

    lines = _lines(capsys.readouterr().out)
    assert code == 0
    assert int(lines["earlier samples at or above the knee"]) > 0
    assert lines["differing samples"] == "0"
    assert lines["differing samples outside the limiter's reach"] == "0"
    assert lines["largest difference"] == "0 LSB"
    assert lines["limited regions"] == "0"
    assert lines["worst region change"] == "none"


def _refused_comparison(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], earlier: Path
) -> tuple[int, Path, str]:
    source = _write_wav(tmp_path / "in.wav", _vowel_pcm(0.2))
    target = tmp_path / "out.wav"
    code = render_character.main(
        ["--in", str(source), "--out", str(target), "--compare-to", str(earlier)]
    )
    captured = capsys.readouterr()
    return code, target, captured.out + captured.err


def _assert_refused(code: int, target: Path, printed: str, tmp_path: Path) -> None:
    assert code == 2
    assert not target.exists()
    assert str(tmp_path) not in printed
    assert ".wav" not in printed


def test_refuses_an_earlier_render_inside_the_checkout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    earlier = REPO_ROOT / "tests" / "absent.wav"

    code, target, printed = _refused_comparison(tmp_path, capsys, earlier)

    _assert_refused(code, target, printed, tmp_path)
    assert str(REPO_ROOT) not in printed
    # The path does not exist, so the missing-file refusal would also exit 2; only this reason
    # shows that the checkout check refused it first.
    assert "--compare-to lies inside the checkout" in printed


def test_refuses_a_missing_earlier_render(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, target, printed = _refused_comparison(tmp_path, capsys, tmp_path / "missing.wav")

    _assert_refused(code, target, printed, tmp_path)


def test_refuses_an_unreadable_earlier_render(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    earlier = tmp_path / "earlier.wav"
    earlier.write_bytes(b"not a wav file")

    code, target, printed = _refused_comparison(tmp_path, capsys, earlier)

    _assert_refused(code, target, printed, tmp_path)


@pytest.mark.parametrize(
    "params",
    [
        pytest.param({"rate": 16000}, id="16kHz"),
        pytest.param({"channels": 2}, id="stereo"),
        pytest.param({"width": 1}, id="8-bit"),
    ],
)
def test_refuses_an_earlier_render_that_is_not_24khz_mono_pcm16(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], params: dict
) -> None:
    earlier = _write_wav(tmp_path / "earlier.wav", _vowel_pcm(0.2), **params)

    code, target, printed = _refused_comparison(tmp_path, capsys, earlier)

    _assert_refused(code, target, printed, tmp_path)


def test_refuses_an_earlier_render_of_another_length(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    earlier = _write_wav(tmp_path / "earlier.wav", _vowel_pcm(0.3))

    code, target, printed = _refused_comparison(tmp_path, capsys, earlier)

    _assert_refused(code, target, printed, tmp_path)
