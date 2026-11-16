"""Pins the shipped need model, `backend/memory/need_model.json` (plan mist-read-path v10, T75).

The file holds one record, fit by H23 on 2026-11-16 for shadow mode only: the `shared` gate
embedder (all-MiniLM-L6-v2, the fact embedder), the logistic form H18's cross-validation chose
for it (c=100), recall target 0.95 and band_k 1. No gate met the eligibility rule at H18, so no
rule chose this one, and its `source` says so; the turn PR's wiring refuses to act on it.

H23 printed, on 2026-11-16:

    h23 bank 4c02bfc07834d530571c8f74f3e015b74a5da8f7ce1c3b8f384fd29de92598f8 rows 3113
    h23 theta_low 0.008722613561843248 (H18 0.008722613561843248)
    h23 theta_high 0.16706492570300246 (H18 0.16706492570300246)
    h23 key c8b6ec7bf8d1939f91c7ca3d79a66a3b58955fec5b3a363ae9c837cbe70edcc2 w 384 b -2.5859461882963006

A change to the bank or to the embedder's lock entry changes the key, and this file's key test
fails before startup refuses to build the gate. A new bank needs a grid run on it and a refit
with `scripts/eval_harness/need_model_shadow.py` from that run's report.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from backend.memory.need_gate import (
    DEFAULT_MODEL_PATH,
    SHADOW_ONLY,
    NeedKey,
    NeedModel,
    load_bank,
    load_models,
    select_model,
)
from scripts.check_models_lock import parse_lock

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LOCK_PATH = _REPO_ROOT / "docker" / "backend" / "models.lock"
_FACT_EMBEDDER = "all-MiniLM-L6-v2"

# H23's printed values (2026-11-16).
H23_THETA_LOW = 0.008722613561843248
H23_THETA_HIGH = 0.16706492570300246
H23_KEY = "c8b6ec7bf8d1939f91c7ca3d79a66a3b58955fec5b3a363ae9c837cbe70edcc2"
H23_B = -2.5859461882963006
# H18's thresholds for the shared variant at recall 0.95, band_k 1 (h18-selection-grid.json).
H18_THETA_LOW = 0.008722613561843248
H18_THETA_HIGH = 0.16706492570300246
REL_TOL = 1e-6


@pytest.fixture(scope="module")
def shipped() -> NeedModel:
    """The one record, selected by the key the bank and the lock give today."""
    models = load_models()
    assert len(models) == 1
    entry = next(e for e in parse_lock(_LOCK_PATH) if e.name == _FACT_EMBEDDER)
    return select_model(models, NeedKey(load_bank().sha256, entry.name, entry.revision, ""))


def test_the_shipped_file_is_the_default_model_path() -> None:
    assert DEFAULT_MODEL_PATH == _REPO_ROOT / "backend" / "memory" / "need_model.json"
    assert DEFAULT_MODEL_PATH.is_file()


def test_the_file_holds_one_record() -> None:
    assert len(load_models()) == 1


def test_the_record_is_keyed_to_the_bank_and_the_locked_fact_embedder(shipped: NeedModel) -> None:
    assert shipped.key == H23_KEY
    assert shipped.embedder_name == _FACT_EMBEDDER
    assert shipped.query_prefix == ""
    assert shipped.normalisation == "l2"


def test_the_form_is_logistic_with_c_100_over_384_dimensions(shipped: NeedModel) -> None:
    assert shipped.form == "logistic"
    assert shipped.params["c"] == 100
    assert len(shipped.params["w"]) == 384
    assert shipped.params["b"] == H23_B


def test_band_k_and_recall_target(shipped: NeedModel) -> None:
    assert shipped.band_k == 1
    assert shipped.recall_target == 0.95


def test_the_thresholds_are_h23s(shipped: NeedModel) -> None:
    assert shipped.theta_low == H23_THETA_LOW
    assert shipped.theta_high == H23_THETA_HIGH


def test_the_thresholds_are_h18s_within_a_relative_1e_6(shipped: NeedModel) -> None:
    assert math.isclose(shipped.theta_low, H18_THETA_LOW, rel_tol=REL_TOL, abs_tol=0.0)
    assert math.isclose(shipped.theta_high, H18_THETA_HIGH, rel_tol=REL_TOL, abs_tol=0.0)


def test_the_source_marks_it_shadow_only(shipped: NeedModel) -> None:
    assert shipped.source.startswith(SHADOW_ONLY)
