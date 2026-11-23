"""The server name spotter on the real microWakeWord model, against microWakeWord's own scores.

Two host tests, each skipping with the missing thing named: the runtime packages
(ai-edge-litert, pymicro-features), `MIST_VOICE_KWS_MODEL`, or that model's manifest or model
file under the voice settings' models directory (`MIST_VOICE_MODELS_DIR`, whose default
`/models` a host lacks). A model that is staged but fails to load fails the test: that is a
broken host, not a missing model.

- The load check: the spotter loads the staged model, and one second of digital silence scores
  below the `MIST_VOICE_KWS_THRESHOLD` in force.
- Parity, which also needs `MIST_VOICE_KWS_PARITY_DIR`: that directory holds 16 kHz mono PCM16
  WAVs and a `scores.json`, `{"<file>": [p, ...]}`, the per-step probabilities
  microWakeWord's `predict_clip` gave on each with the device's input formula. `probabilities`
  must equal every listed step within 1e-6. It may give one step more at the end, because
  microWakeWord's `generate_features_for_clip` stops before a chunk that ends exactly at the
  buffer's end.

The real model and the reference scores are never committed; the runner has neither, so both
tests skip there. The helper tests below them need neither and run everywhere.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import wave
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from backend.realtime.budget import PlacementConfig, VoiceSettings
from backend.realtime.engines.kws import SAMPLE_RATE, MicroNameSpotter, import_micro_runtime
from backend.realtime.errors import SpeechEngineError

TOLERANCE = 1e-6
PARITY_VARIABLE = "MIST_VOICE_KWS_PARITY_DIR"
SCORES_FILE = "scores.json"
SILENCE_SECONDS = 1

# Each module the runtime needs, with the package that provides it (kws.py's `_RUNTIME`).
RUNTIME = (("ai_edge_litert", "ai-edge-litert"), ("pymicro_features", "pymicro-features"))


def _missing_runtime() -> str | None:
    """The first runtime package not installed, found without importing it."""
    for module, package in RUNTIME:
        if importlib.util.find_spec(module) is None:
            return f"package {package} is not installed (no module {module!r})"
    return None


def _missing_model(placement: PlacementConfig) -> str | None:
    """What of the configured model is not staged: the variable, the manifest or the model file.

    Only presence is checked here. A manifest that is present but unreadable or refused is left
    to `MicroNameSpotter.load`, so it fails the test with the loader's own reason.
    """
    model_id = placement.kws_model
    if model_id is None:
        return "variable MIST_VOICE_KWS_MODEL is not set: it names the spotter's model id"
    directory = Path(placement.models_dir, "speech", "wake", model_id)
    manifest = directory / "manifest.json"
    if not manifest.is_file():
        return (
            f"manifest missing: {manifest} (models directory {placement.models_dir}, "
            "from MIST_VOICE_MODELS_DIR)"
        )
    try:
        raw = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    model = raw.get("model") if isinstance(raw, dict) else None
    file = model.get("file") if isinstance(model, dict) else None
    if isinstance(file, str) and file:
        path = directory.joinpath(*file.split("/"))
        if not path.is_file():
            return f"model missing: {path} (the manifest's model.file)"
    return None


def _load_spotter(settings: VoiceSettings) -> MicroNameSpotter:
    """Load the staged spotter the way the backend does, skipping naming what is missing."""
    for reason in (_missing_runtime(), _missing_model(settings.placement)):
        if reason is not None:
            pytest.skip(reason)
    try:
        return MicroNameSpotter.load(settings.placement, library_loader=import_micro_runtime)
    except SpeechEngineError as exc:
        pytest.fail(f"the staged spotter does not load: {exc}")


def _read_scores(directory: Path) -> dict[str, list[float]]:
    """The reference probabilities `scores.json` lists, by WAV file name.

    Raises:
        ValueError: The file is missing or malformed, or lists no WAV, so parity cannot pass by
            comparing nothing.
    """
    path = directory / SCORES_FILE
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"{path} could not be read: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f'{path} must be an object, {{"<file>": [p, ...]}}')
    if not raw:
        raise ValueError(f"{path} lists no WAV")
    scores = {}
    for name, values in raw.items():
        if not isinstance(values, list) or not all(
            isinstance(value, int | float) and not isinstance(value, bool) for value in values
        ):
            raise ValueError(f"{path}: {name!r} must map to a list of probabilities")
        scores[name] = [float(value) for value in values]
    return scores


def _read_pcm16(path: Path) -> bytes:
    """A WAV's samples as the spotter takes them: PCM16 mono 16 kHz, little-endian.

    Raises:
        ValueError: The file is missing, unreadable or in another format.
    """
    try:
        with wave.open(str(path), "rb") as wav:
            form = (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype())
            frames = wav.readframes(wav.getnframes())
    except (OSError, EOFError, wave.Error) as exc:
        raise ValueError(f"{path} could not be read as a WAV: {exc}") from exc
    if form != (1, 2, SAMPLE_RATE, "NONE"):
        channels, width, rate, compression = form
        raise ValueError(
            f"{path} is {channels} channel(s), {8 * width}-bit, {rate} Hz, {compression}; "
            f"parity takes mono 16-bit PCM at {SAMPLE_RATE} Hz"
        )
    return frames


def _parity_miss(file: str, expected: Sequence[float], got: Sequence[float]) -> str | None:
    """The first step where the spotter departs from microWakeWord, or None.

    Every listed step must match within `TOLERANCE`; the spotter may give one step more. A NaN
    on either side is a miss.
    """
    for step, (want, have) in enumerate(zip(expected, got, strict=False)):
        if not abs(have - want) <= TOLERANCE:
            return (
                f"{file}: step {step} differs: the spotter gives {have!r}, microWakeWord "
                f"{want!r} (tolerance {TOLERANCE})"
            )
    if len(got) < len(expected):
        step = len(got)
        return (
            f"{file}: step {step} differs: the spotter gives no step, microWakeWord "
            f"{expected[step]!r} ({len(got)} steps against {len(expected)})"
        )
    if len(got) > len(expected) + 1:
        step = len(expected) + 1
        return (
            f"{file}: step {step} differs: the spotter gives {got[step]!r}, microWakeWord no step "
            f"({len(got)} steps against {len(expected)}; at most one more is allowed)"
        )
    return None


def _largest_difference(expected: Sequence[float], got: Sequence[float]) -> float:
    return max((abs(h - w) for w, h in zip(expected, got, strict=False)), default=0.0)


# -- The real model ---------------------------------------------------------------------------


@pytest.mark.speech_models
def test_spotter_loads_and_scores_one_second_of_silence_below_the_threshold(
    voice_settings, speech_record
):
    spotter = _load_spotter(voice_settings)
    threshold = voice_settings.wake.kws_threshold

    series = spotter.score_series(bytes(2 * SAMPLE_RATE * SILENCE_SECONDS))

    assert series, f"{SILENCE_SECONDS} s of audio completed no step"
    peak = max(series)
    speech_record(
        "kws_silence", model=spotter.model_id, steps=len(series), peak=peak, threshold=threshold
    )
    assert peak < threshold, (
        f"{SILENCE_SECONDS} s of zeros scores {peak}, at or above the threshold {threshold} "
        "(MIST_VOICE_KWS_THRESHOLD)"
    )


@pytest.mark.speech_models
def test_probabilities_equal_microwakeword_on_the_parity_set(voice_settings, speech_record):
    directory = os.environ.get(PARITY_VARIABLE, "").strip()
    if not directory:
        pytest.skip(
            f"variable {PARITY_VARIABLE} is not set: it names the directory of WAVs and the "
            f"{SCORES_FILE} microWakeWord's inference wrote for them"
        )
    spotter = _load_spotter(voice_settings)
    try:
        scores = _read_scores(Path(directory))
    except ValueError as exc:
        pytest.fail(str(exc))

    misses = []
    for file, expected in scores.items():
        try:
            pcm = _read_pcm16(Path(directory, file))
        except ValueError as exc:
            pytest.fail(str(exc))
        got = spotter.probabilities(pcm)
        speech_record(
            "kws_parity",
            model=spotter.model_id,
            file=file,
            listed_steps=len(expected),
            steps=len(got),
            largest_difference=_largest_difference(expected, got),
        )
        miss = _parity_miss(file, expected, got)
        if miss is not None:
            misses.append(miss)

    assert not misses, "\n".join(misses)


# -- The helpers, without the runtime ---------------------------------------------------------


def _placement(models_dir: Path, kws_model: str | None) -> PlacementConfig:
    return PlacementConfig(models_dir=str(models_dir), kws_model=kws_model)


def _stage_manifest(models_dir: Path, manifest: Mapping[str, object]) -> Path:
    directory = Path(models_dir, "speech", "wake", "mist-mww-1")
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory


def test_parity_miss_names_the_file_the_first_differing_step_and_both_values():
    miss = _parity_miss("a.wav", [0.1, 0.2, 0.3], [0.1, 0.25, 0.9])

    assert miss == (
        "a.wav: step 1 differs: the spotter gives 0.25, microWakeWord 0.2 (tolerance 1e-06)"
    )


def test_parity_miss_accepts_values_within_the_tolerance_and_one_more_step():
    assert _parity_miss("a.wav", [0.1, 0.2], [0.1 + 5e-7, 0.2 - 5e-7, 0.7]) is None


def test_parity_miss_names_the_first_step_the_spotter_does_not_give():
    miss = _parity_miss("a.wav", [0.1, 0.2], [0.1])

    assert miss == (
        "a.wav: step 1 differs: the spotter gives no step, microWakeWord 0.2 (1 steps against 2)"
    )


def test_parity_miss_refuses_two_steps_more_than_listed():
    miss = _parity_miss("a.wav", [0.1], [0.1, 0.0, 0.5])

    assert miss is not None
    assert miss.startswith("a.wav: step 2 differs: the spotter gives 0.5, microWakeWord no step")


def test_parity_miss_counts_nan_as_a_miss():
    assert _parity_miss("a.wav", [0.1], [math.nan]) is not None


def test_missing_model_names_the_variable_when_no_model_is_set(tmp_path):
    reason = _missing_model(_placement(tmp_path, None))

    assert reason is not None and "MIST_VOICE_KWS_MODEL" in reason


def test_missing_model_names_the_manifest_path_and_the_models_variable(tmp_path):
    reason = _missing_model(_placement(tmp_path, "mist-mww-1"))

    manifest = Path(tmp_path, "speech", "wake", "mist-mww-1", "manifest.json")
    assert reason is not None
    assert str(manifest) in reason and "MIST_VOICE_MODELS_DIR" in reason


def test_missing_model_names_the_model_file_the_manifest_gives(tmp_path):
    directory = _stage_manifest(tmp_path, {"model": {"file": "device/mist.tflite"}})

    reason = _missing_model(_placement(tmp_path, "mist-mww-1"))

    assert reason is not None and str(directory / "device" / "mist.tflite") in reason


def test_missing_model_leaves_a_malformed_manifest_to_the_loader(tmp_path):
    _stage_manifest(tmp_path, {"model": "not an object"})

    assert _missing_model(_placement(tmp_path, "mist-mww-1")) is None


def test_read_scores_refuses_an_empty_listing(tmp_path):
    (tmp_path / SCORES_FILE).write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="lists no WAV"):
        _read_scores(tmp_path)


def test_read_scores_refuses_a_value_that_is_not_a_list_of_probabilities(tmp_path):
    (tmp_path / SCORES_FILE).write_text(json.dumps({"a.wav": [0.1, "0.2"]}), encoding="utf-8")

    with pytest.raises(ValueError, match="'a.wav' must map to a list of probabilities"):
        _read_scores(tmp_path)


def test_read_pcm16_refuses_a_wav_that_is_not_16_khz_mono_pcm16(tmp_path):
    path = tmp_path / "a.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(bytes(160))

    with pytest.raises(ValueError, match="parity takes mono 16-bit PCM at 16000 Hz"):
        _read_pcm16(path)
