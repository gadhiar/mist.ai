"""Tests for the gate defaults and the root's calibrated overlay (Interfaces, Gates)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.voice_data import gates
from scripts.voice_data.errors import InputError
from scripts.voice_data.root import Root
from scripts.voice_data.states import NEUTRAL_STATE, SPEAKING_STATES

# Every value the Interfaces table marks [E], plus the T5 bandwidth cutoff: H6 calibrates these.
CALIBRATED = {
    "anchors.sim_min": 0.6,
    "retrieval.anchor_floor": 0.6,
    "retrieval.margin": 0.1,
    "segment.snr_min_db": 15,
    "segment.c50_min_db": 20,
    "segment.window_percentile": 2,
    "cleaning.spectrum_max_db": 2.0,
    "state_floor.rehearsal_identity_window_floor": 0.6,
    "bakeoff.weak_rating_below": 3.5,
    "text.reject_max": 3,
    "spotcheck.reject_max": 5,
    "ingest.bandwidth_cutoff_db": 40,
    **{f"state_floor.offset.{state}": 0 for state in SPEAKING_STATES},
}

FIXED = {
    "anchors.min_s": 3,
    "anchors.max_s": 30,
    "segment.min_s": 3,
    "segment.max_s": 12,
    "segment.long_max_s": 30,
    "segment.overlap_max_ms": 0,
    "segment.window_s": 1.5,
    "seed.clean_minutes_min": 10,
    "seed.sample_rate_min_hz": 24000,
    "seed.rehearsal_precision_min": 0.95,
    "seed.state_seconds_min": 30,
    "state_floor.own_n_min": 20,
    "state_floor.thin_n_min": 5,
    "state_floor.identity_windows_min": 20,
    "take.cer_max": 0.03,
    "take.n": 3,
    "take.n.calm": 5,
    "take.n.sleepy": 5,
    "take.rounds_max": 3,
    "bench.first_audio_cpu_ms_max": 196,
    "bench.first_audio_gpu_stt_ms_max": 345,
    "bench.decode_loss_max": 0.05,
    "bench.sim_median_ratio_min": 0.9,
    "bench.cer_max": 0.03,
    "bench.f0_std_tolerance": 0.2,
}


def test_speaking_states_are_v6_five() -> None:
    assert SPEAKING_STATES == ("neutral", "calm", "awake", "sleepy", "excited")
    assert NEUTRAL_STATE == "neutral"


def _overlay(root: Root, data: object) -> None:
    (root.path / "config" / "gates.json").write_text(json.dumps(data), encoding="utf-8")


def test_defaults_carry_value_and_calibration_note() -> None:
    loaded = gates.load(None)
    assert loaded.defaults
    for key, default in loaded.defaults.items():
        assert isinstance(default.value, int | float) and not isinstance(default.value, bool), key
        assert isinstance(default.h6, bool), key
        assert default.note.strip(), key


@pytest.mark.parametrize(("key", "value"), sorted(CALIBRATED.items()))
def test_calibrated_gates_are_marked_for_h6(key: str, value: float) -> None:
    default = gates.load(None).defaults[key]
    assert default.value == value
    assert default.h6 is True


@pytest.mark.parametrize(("key", "value"), sorted(FIXED.items()))
def test_fixed_gates_are_not_calibrated(key: str, value: float) -> None:
    default = gates.load(None).defaults[key]
    assert default.value == value
    assert default.h6 is False


def test_no_overlay_gives_defaults(private_root: Root) -> None:
    loaded = gates.load(private_root.path)
    assert loaded.effective() == {k: d.value for k, d in loaded.defaults.items()}
    assert loaded.overlay == {}


def test_overlay_changes_only_its_keys(private_root: Root) -> None:
    _overlay(private_root, {"retrieval.margin": 0.15, "state_floor.offset.calm": 0.02})
    loaded = gates.load(private_root.path)
    effective = loaded.effective()
    defaults = {k: d.value for k, d in loaded.defaults.items()}
    changed = {k for k in effective if effective[k] != defaults[k]}
    assert changed == {"retrieval.margin", "state_floor.offset.calm"}
    assert effective["retrieval.margin"] == 0.15
    assert loaded.value("state_floor.offset.calm") == 0.02
    assert loaded.value("anchors.sim_min") == 0.6


def test_effective_is_a_copy(private_root: Root) -> None:
    loaded = gates.load(private_root.path)
    loaded.effective()["anchors.sim_min"] = 0.0
    assert loaded.value("anchors.sim_min") == 0.6


def test_overlay_unknown_key_is_input_error(private_root: Root) -> None:
    _overlay(private_root, {"anchors.sim_minimum": 0.5})
    with pytest.raises(InputError, match="unknown"):
        gates.load(private_root.path)


@pytest.mark.parametrize("bad", ["0.5", True, None, [0.5], {"value": 0.5}])
def test_overlay_non_number_is_input_error(private_root: Root, bad: object) -> None:
    _overlay(private_root, {"anchors.sim_min": bad})
    with pytest.raises(InputError, match="number"):
        gates.load(private_root.path)


def test_overlay_not_an_object_is_input_error(private_root: Root) -> None:
    _overlay(private_root, [1, 2])
    with pytest.raises(InputError):
        gates.load(private_root.path)


def test_overlay_bad_json_is_input_error(private_root: Root) -> None:
    (private_root.path / "config" / "gates.json").write_text("{", encoding="utf-8")
    with pytest.raises(InputError):
        gates.load(private_root.path)


def test_missing_default_file_is_input_error(tmp_path: Path) -> None:
    with pytest.raises(InputError, match="default"):
        gates.load(None, defaults_path=tmp_path / "absent.json")


def test_unknown_key_lookup_is_input_error() -> None:
    with pytest.raises(InputError, match="unknown"):
        gates.load(None).value("no.such.gate")
