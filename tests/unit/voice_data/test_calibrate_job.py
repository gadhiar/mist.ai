"""The calibration job: pools, impostor trials, the threshold rule and the bootstrap (T32).

Every test runs on synthetic embeddings or tiny synthetic audio; `load_embedders` is replaced by
fakes, so no model is ever loaded.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import re
import sys
import zipfile
import zlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
import soundfile

from scripts.voice_data.rehearsal import calibrate
from scripts.voice_data.rehearsal.mix import StageError

NAMES = ("wavlm-sv", "ecapa")
PLAN_REL = "rehearsal/calibration/c1/calibration-plan.json"
FOLDER_REL = "rehearsal/calibration/c1"

# Every path the process opens through Python while `_OPENED` is recording. An audit hook cannot
# be removed, so it is installed once and does nothing unless a test switches it on.
_OPENED: list[str] = []
_RECORDING = [False]


def _audit(event: str, args: tuple) -> None:
    if _RECORDING[0] and event == "open" and args and isinstance(args[0], str | bytes | Path):
        _OPENED.append(str(args[0]))


sys.addaudithook(_audit)


# ---------------------------------------------------------------------------
# Fakes and builders
# ---------------------------------------------------------------------------


class ToneEmbedder:
    """A cheap deterministic embedder: a fixed projection of the clip's log band spectrum."""

    def __init__(self, name: str) -> None:
        self.name = name
        rng = np.random.default_rng(zlib.crc32(name.encode("utf-8")))
        self._projection = rng.standard_normal((12, 32))
        self.calls = 0

    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        self.calls += 1
        spectrum = np.abs(np.fft.rfft(np.asarray(audio, dtype=np.float64)[:8192], n=8192))
        bands = np.log(spectrum[:4096].reshape(32, 128).sum(axis=1) + 1e-9)
        vector = self._projection @ (bands - bands.mean())
        return (vector / np.linalg.norm(vector)).astype(np.float32)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def make_speakers(
    sexes: dict[str, int],
    dataset: str = "librittsr",
    pieces: int = 50,
    noise: float = 0.35,
    seed: int = 0,
) -> list[calibrate.Speaker]:
    """Speakers whose embeddings scatter around one random direction each, per embedder."""
    rng = np.random.default_rng(seed)
    speakers = []
    number = 0
    for sex, count in sexes.items():
        for _ in range(count):
            number += 1
            means = {name: _unit(rng.standard_normal(16)) for name in NAMES}

            def emb(means: dict = means) -> dict[str, np.ndarray]:
                return {n: _unit(m + noise * rng.standard_normal(16)) for n, m in means.items()}

            speakers.append(
                calibrate.Speaker(
                    dataset=dataset,
                    speaker=f"{1000 + number}",
                    sex=sex,
                    anchors=tuple(emb() for _ in range(5)),
                    anchor_registers=("solo", "neutral", "neutral", "neutral", "neutral"),
                    pieces=tuple(emb() for _ in range(pieces)),
                    registers=tuple("neutral" for _ in range(pieces)),
                )
            )
    return speakers


SMALL = calibrate.TrialParams(bootstrap=30)


def _summary(datasets: Sequence[calibrate.DatasetTrials], seed: int = 7, alpha: float = 0.01):
    return calibrate.summarise(datasets, alpha=alpha, seed=seed, bootstrap=30, names=NAMES)


# ---------------------------------------------------------------------------
# The threshold rule
# ---------------------------------------------------------------------------

# Form f1 (hand-computed, alpha 0.1, 10 trials): wavlm ranks 0..9, ecapa ranks the trials
# 1, 2, 4, ... At m = 2 the top-2 sets share trial 1 (joint 1 of 10 = alpha); at m = 3 they share
# trials 1 and 2 (0.2). So m* = 2, beta* = 0.2999 (the largest grid beta with floor(10 beta) = 2),
# and each value is the grid value just above the third-highest score, 0.7001.
F1 = np.array(
    [
        [0.9, 0.0],
        [0.8, 0.9],
        [0.7, 0.8],
        [0.6, 0.1],
        [0.5, 0.7],
        [0.4, 0.6],
        [0.3, 0.5],
        [0.2, 0.4],
        [0.1, 0.3],
        [0.0, 0.2],
    ]
)
# Form f2: both embedders rank the trials alike, so the joint FAR is the marginal one; m* = 1,
# beta* = 0.1999, values just above the second-highest scores: 0.5501 and 0.8501.
F2 = np.array(
    [
        [0.6, 0.9],
        [0.55, 0.85],
        [0.5, 0.8],
        [0.45, 0.75],
        [0.4, 0.7],
        [0.35, 0.65],
        [0.3, 0.6],
        [0.25, 0.55],
        [0.2, 0.5],
        [0.15, 0.45],
    ]
)


def test_toy_form_matches_the_hand_computation() -> None:
    rule = calibrate.threshold(F1, alpha=0.1)
    assert rule.beta == 0.2999
    assert rule.values == (0.7001, 0.7001)
    assert rule.joint_far == 0.1
    rule2 = calibrate.threshold(F2, alpha=0.1)
    assert (rule2.beta, rule2.values, rule2.joint_far) == (0.1999, (0.5501, 0.8501), 0.1)


def test_toy_gate_takes_the_maximum_over_forms_and_holds_alpha_in_each() -> None:
    gate = calibrate.gate_values({("d", "f1"): F1, ("d", "f2"): F2}, alpha=0.1)
    assert gate.values == (0.7001, 0.8501)
    assert gate.set_by == (("d", "f1"), ("d", "f2"))
    assert calibrate.joint_far(F1, gate.values) == 0.1
    assert calibrate.joint_far(F2, gate.values) == 0.0
    for scores in (F1, F2):
        assert calibrate.joint_far(scores, gate.values) <= 0.1


def _t_alpha(scores: np.ndarray, allowed: int) -> float:
    """Brute force: the smallest 0.0001 grid value accepting at most `allowed` scores."""
    grid = np.arange(int(np.floor(scores.min() * 1e4)) - 2, int(np.ceil(scores.max() * 1e4)) + 3)
    for k in grid:
        if int(np.sum(scores >= k / 1e4 - 1e-9)) <= allowed:
            return round(k / 1e4, 4)
    raise AssertionError("no grid value")


def test_identical_embedders_give_each_value_its_own_t_alpha() -> None:
    column = np.random.default_rng(3).normal(0.3, 0.1, 100)
    rule = calibrate.threshold(np.column_stack([column, column]), alpha=0.01)
    assert rule.values == (_t_alpha(column, 1), _t_alpha(column, 1))
    assert rule.beta == 0.0199


def test_independent_embedders_spend_alpha_above_each_marginal() -> None:
    rng = np.random.default_rng(4)
    scores = rng.normal(0.3, 0.1, (2000, 2))
    rule = calibrate.threshold(scores, alpha=0.01)
    assert rule.beta > 0.01
    assert rule.joint_far <= 0.01


def test_datasets_are_never_pooled() -> None:
    rng = np.random.default_rng(5)
    low = rng.normal(0.2, 0.05, (200, 2))
    high = rng.normal(0.7, 0.05, (200, 2))
    gate = calibrate.gate_values({("libri", "f"): low, ("rav", "f"): high}, alpha=0.01)
    alone = calibrate.threshold(high, alpha=0.01)
    pooled = calibrate.threshold(np.vstack([low, high]), alpha=0.01)
    assert gate.values == alone.values
    assert gate.set_by == (("rav", "f"), ("rav", "f"))
    assert gate.values != pooled.values
    assert calibrate.joint_far(low, gate.values) <= 0.01
    assert calibrate.joint_far(high, gate.values) <= 0.01


def test_margin_values_go_negative_when_impostor_gaps_are() -> None:
    gaps = -np.abs(np.random.default_rng(6).normal(0.2, 0.05, (300, 2)))
    rule = calibrate.threshold(gaps, alpha=0.01)
    assert all(v < 0 for v in rule.values)
    # Tight speakers: each held-out piece sits far nearer its own rival centroid than A's model.
    speakers = make_speakers({"female": 3, "male": 3}, noise=0.05)
    trials = calibrate.dataset_trials("librittsr", speakers, SMALL, seed=1, names=NAMES)
    assert (trials.impostor[calibrate.MARGIN]["rival"].scores < 0).all()
    summary = _summary([trials])
    assert all(v < 0 for v in summary.values[calibrate.MARGIN])


def test_weights_count_a_trial_that_many_times() -> None:
    weights = np.array([2, 1, 1, 1, 1, 1, 1, 1, 1, 0])
    repeated = np.vstack([F1[0], F1[:9]])
    assert calibrate.threshold(F1, alpha=0.1, weights=weights) == calibrate.threshold(
        repeated, alpha=0.1
    )


# ---------------------------------------------------------------------------
# Trials and the summary
# ---------------------------------------------------------------------------


def test_trials_pair_same_sex_speakers_only_and_count_as_the_design() -> None:
    speakers = make_speakers({"female": 4, "male": 3})
    trials = calibrate.dataset_trials("librittsr", speakers, SMALL, seed=1, names=NAMES)
    pairs = 4 * 3 + 3 * 2
    sex = {i: s.sex for i, s in enumerate(speakers)}
    for gate in trials.impostor.values():
        for form in gate.values():
            assert form.pairs == pairs
            assert all(sex[a] == sex[b] and a != b for a, b in zip(form.a, form.b))
    floor = trials.impostor[calibrate.FLOOR]
    assert set(floor) == {f"{a}/n{n}" for a in ("solo", "centroid") for n in (10, 20, 30)}
    assert set(trials.reported[calibrate.FLOOR]) == {"solo/n5", "centroid/n5"}
    assert len(floor["solo/n10"].scores) == pairs * 10
    held = 50 - 30
    assert len(trials.impostor[calibrate.MARGIN]["rival"].scores) == pairs * held
    sim_min = trials.impostor[calibrate.SIM_MIN]
    assert {k: len(f.scores) for k, f in sim_min.items()} == {
        "k1": pairs * 5,
        "k2": pairs * 4 * 5,
        "k4": pairs * 4 * 5,
    }


def test_trial_scores_come_from_extract_functions() -> None:
    speakers = make_speakers({"female": 3, "male": 3})
    trials = calibrate.dataset_trials("librittsr", speakers, SMALL, seed=1, names=NAMES)
    a, b = speakers[0], speakers[1]
    first = trials.impostor[calibrate.SIM_MIN]["k1"]
    centre = calibrate.retrieve.centroids([a.anchors[0]])
    want = calibrate.retrieve.sims(b.anchors[0], centre)
    assert first.a[0] == 0 and first.b[0] == 1
    assert tuple(first.scores[0]) == tuple(want[n] for n in NAMES)
    margin = trials.impostor[calibrate.MARGIN]["rival"]
    model = calibrate.retrieve.centroids(list(a.pieces[:30]))
    rival = calibrate.retrieve.centroids(list(b.pieces[:20]))
    held = b.pieces[30]
    gap = [
        calibrate.retrieve.sims(held, model)[n] - calibrate.retrieve.sims(held, rival)[n]
        for n in NAMES
    ]
    assert np.allclose(margin.scores[0], gap, atol=0, rtol=0)


def test_summary_holds_alpha_in_every_form_and_dataset() -> None:
    libri = calibrate.dataset_trials(
        "librittsr", make_speakers({"female": 4, "male": 3}), SMALL, seed=1, names=NAMES
    )
    rav = calibrate.dataset_trials(
        "ravdess",
        make_speakers({"female": 3, "male": 3}, dataset="ravdess", seed=9, noise=0.6),
        SMALL,
        seed=1,
        names=NAMES,
    )
    summary = _summary([libri, rav])
    for gate, values in summary.values.items():
        for trials in (libri, rav):
            for form in trials.impostor[gate].values():
                assert calibrate.joint_far(form.scores, values) <= 0.01


def test_permuting_target_scores_changes_no_value() -> None:
    trials = calibrate.dataset_trials(
        "librittsr", make_speakers({"female": 3, "male": 3}), SMALL, seed=1, names=NAMES
    )
    rng = np.random.default_rng(11)
    shuffled = {
        gate: {
            form: dataclasses.replace(t, scores=rng.permutation(t.scores.ravel()).reshape(-1, 2))
            for form, t in forms.items()
        }
        for gate, forms in trials.target.items()
    }
    before = _summary([trials])
    after = _summary([dataclasses.replace(trials, target=shuffled)])
    assert before.values == after.values
    assert before.intervals == after.intervals


def test_bootstrap_repeats_under_one_seed() -> None:
    trials = calibrate.dataset_trials(
        "librittsr", make_speakers({"female": 3, "male": 3}), SMALL, seed=1, names=NAMES
    )
    first, second = _summary([trials], seed=21), _summary([trials], seed=21)
    assert first.intervals == second.intervals
    assert first.far_intervals == second.far_intervals
    assert first.intervals != _summary([trials], seed=22).intervals


# ---------------------------------------------------------------------------
# Pools
# ---------------------------------------------------------------------------

RATE = 1000


def test_a_piece_cut_drops_a_04_s_remainder_and_keeps_a_05_s_one() -> None:
    pieces, _ = calibrate.cut(np.ones(int(3.4 * RATE)), RATE, (3.0, 1.5, 6.0))
    assert [len(p) / RATE for p in pieces] == [3.0]
    pieces, after = calibrate.cut(np.ones(int(3.5 * RATE)), RATE, (3.0, 1.5, 6.0))
    assert [len(p) / RATE for p in pieces] == [3.0, 0.5]
    assert after == 2


def test_the_piece_cycle_runs_on_across_files() -> None:
    pieces, after = calibrate.cut(np.ones(int(8.0 * RATE)), RATE, (3.0, 1.5, 6.0), start=1)
    assert [len(p) / RATE for p in pieces] == [1.5, 6.0, 0.5]
    assert after == 4


def _libri_clips(count: int) -> list[np.ndarray]:
    return [np.ones(3 * RATE) for _ in range(count)]


def test_a_pool_under_40_pieces_makes_its_speaker_ineligible() -> None:
    params = dataclasses.replace(SMALL, piece_s=(3.0,))
    # Four 3 s files make the 13 s solo clip, four more the anchors, the rest one piece each.
    short = calibrate.librittsr_pool(_libri_clips(4 + 4 + 39), RATE, params)
    assert isinstance(short, str) and "39" in short
    pool = calibrate.librittsr_pool(_libri_clips(4 + 4 + 40), RATE, params)
    assert isinstance(pool, calibrate.Pool)
    assert len(pool.anchors) == 4 and len(pool.pieces) == 40
    assert len(pool.solo) == int((4 * 3 + 3 * 0.25) * RATE)
    capped = calibrate.librittsr_pool(_libri_clips(4 + 4 + 120), RATE, params)
    assert len(capped.pieces) == 100


def test_fewer_than_three_eligible_of_a_sex_refuses() -> None:
    calibrate.require_speakers("librittsr", {"female": 3, "male": 3})
    with pytest.raises(StageError, match="male.*2"):
        calibrate.require_speakers("librittsr", {"female": 11, "male": 2})


def test_ravdess_pool_takes_neutral_then_calm_for_the_solo_and_one_anchor_per_register() -> None:
    clips = []
    for register in ("neutral", "calm", "awake", "sleepy", "excited", "other"):
        count = 4 if register == "neutral" else 8 if register != "other" else 24
        clips += [calibrate.Clip(np.ones(int(1.5 * RATE)), register) for _ in range(count)]
    pool = calibrate.ravdess_pool(clips, RATE, SMALL)
    assert isinstance(pool, calibrate.Pool)
    assert pool.solo_registers == ("neutral",) * 4 + ("calm",) * 3
    assert [a.register for a in pool.anchors] == ["calm", "awake", "sleepy", "excited"]
    assert all(len(a.audio) == int((1.5 * 2 + 0.25) * RATE) for a in pool.anchors)
    assert len(pool.pieces) == 60 - 7 - 8


# ---------------------------------------------------------------------------
# RAVDESS names and the archive
# ---------------------------------------------------------------------------


def test_ravdess_names_parse_to_actor_sex_and_register() -> None:
    first = calibrate.parse_ravdess("Actor_01/03-01-03-02-01-01-01.wav")
    assert (first.actor, first.sex, first.register) == ("01", "male", "awake")
    second = calibrate.parse_ravdess("Actor_24/03-01-08-01-02-02-24.wav")
    assert (second.actor, second.sex, second.register) == ("24", "female", "excited")
    for code, register in (("01", "neutral"), ("02", "calm"), ("04", "sleepy")):
        assert calibrate.parse_ravdess(f"03-01-{code}-01-01-01-02.wav").register == register
    for code in ("05", "06", "07"):
        assert calibrate.parse_ravdess(f"03-01-{code}-01-01-01-02.wav").register == "other"


@pytest.mark.parametrize(
    "name",
    [
        "Actor_01/03-01-09-01-01-01-01.wav",
        "Actor_02/03-01-01-01-01-01-01.wav",
        "Actor_01/03-02-01-01-01-01-01.wav",
        "Actor_01/01-01-01-01-01-01-01.wav",
        "Actor_25/03-01-01-01-01-01-25.wav",
        "Actor_01/03-01-01-03-01-01-01.wav",
        "Actor_01/readme.txt",
        "other/03-01-01-01-01-01-01.wav",
    ],
)
def test_a_name_off_the_pattern_refuses(name: str) -> None:
    with pytest.raises(StageError, match="pattern"):
        calibrate.parse_ravdess(name)


def _wav_bytes(audio: np.ndarray, rate: int) -> bytes:
    buffer = io.BytesIO()
    soundfile.write(buffer, audio, rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _tone(freq: float, seconds: float, rate: int, seed: int) -> np.ndarray:
    t = np.arange(int(round(seconds * rate))) / rate
    noise = np.random.default_rng(seed).normal(0, 0.02, len(t))
    return (0.3 * np.sin(2 * np.pi * freq * t) + noise).astype(np.float32)


def ravdess_names(actor: int) -> list[str]:
    names = []
    for emotion in range(1, 9):
        for intensity in (1,) if emotion == 1 else (1, 2):
            for statement in (1, 2):
                for repetition in (1, 2):
                    names.append(
                        f"Actor_{actor:02d}/03-01-{emotion:02d}-{intensity:02d}-"
                        f"{statement:02d}-{repetition:02d}-{actor:02d}.wav"
                    )
    return names


def write_ravdess(path: Path, actors: Sequence[int], extra: Sequence[str] = ()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rate = 2000
    with zipfile.ZipFile(path, "w") as archive:
        for actor in actors:
            archive.writestr(f"Actor_{actor:02d}/", b"")
            for index, name in enumerate(ravdess_names(actor)):
                tone = _tone(150 + 60 * actor, 1.5, rate, seed=actor * 100 + index)
                archive.writestr(name, _wav_bytes(tone, rate))
        for name in extra:
            archive.writestr(name, b"x")


def test_the_archive_is_read_in_place(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = tmp_path / "public" / "ravdess" / "speech.zip"
    write_ravdess(archive, (1, 2))

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the archive was extracted")

    monkeypatch.setattr(zipfile.ZipFile, "extract", refuse)
    monkeypatch.setattr(zipfile.ZipFile, "extractall", refuse)
    before = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*"))
    clips = calibrate.read_ravdess(archive, 16000)
    after = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*"))
    assert before == after
    assert {a: len(c) for a, c in clips.items()} == {"01": 60, "02": 60}
    registers = [c.register for c in clips["01"]]
    assert registers.count("neutral") == 4 and registers.count("other") == 24
    assert len(clips["01"][0].audio) == int(1.5 * 16000)


def test_an_off_pattern_entry_refuses_before_any_audio_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "speech.zip"
    write_ravdess(archive, (1,), extra=("Actor_01/03-01-01-01-01-01-02.wav",))
    monkeypatch.setattr(calibrate, "read_audio_bytes", _no_read)
    with pytest.raises(StageError, match="pattern"):
        calibrate.read_ravdess(archive, 16000)


def _no_read(*args: object, **kwargs: object) -> None:
    raise AssertionError("audio was read")


# ---------------------------------------------------------------------------
# The job end to end
# ---------------------------------------------------------------------------

LIBRI = {"female": ("1111", "2222", "3333"), "male": ("4444", "5555", "6666")}
LIBRI_FILES = 36
LIBRI_RATE = 4000


def write_libri(data: Path) -> list[dict]:
    speakers = []
    for sex, ids in LIBRI.items():
        for number, speaker in enumerate(ids):
            files = []
            for index in range(LIBRI_FILES):
                rel = f"public/librittsr/test-clean/{speaker}/{speaker}_{index:03d}.wav"
                tone = _tone(
                    200 + 90 * number + (40 if sex == "male" else 0),
                    3.0,
                    LIBRI_RATE,
                    seed=int(speaker) + index,
                )
                path = data / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                soundfile.write(str(path), tone, LIBRI_RATE, subtype="PCM_16")
                files.append([rel, hashlib.sha256(path.read_bytes()).hexdigest()])
            speakers.append({"speaker": speaker, "sex": sex, "files": files})
    return speakers


def write_plan(data: Path, ravdess: bool = True, **overrides: object) -> dict:
    datasets: list[dict] = [
        {
            "name": "librittsr",
            "licence": "CC BY 4.0",
            "subset": "test-clean",
            "speakers": write_libri(data),
            "excluded": [{"speaker": "7777", "sex": "male", "reason": "build"}],
        }
    ]
    if ravdess:
        rel = "public/ravdess/Audio_Speech_Actors_01-24.zip"
        write_ravdess(data / rel, range(1, 7))
        sha = hashlib.sha256((data / rel).read_bytes()).hexdigest()
        datasets.append(
            {
                "name": "ravdess",
                "licence": "CC BY-NC-SA 4.0",
                "subset": "speech",
                "archive": [rel, sha],
            }
        )
    plan = {
        "schema": 1,
        "kind": "calibration-plan",
        "id": "c1",
        "alpha": 0.01,
        "seed": 20261116,
        "embedders": {"wavlm-sv": {"revision": "a" * 40}, "ecapa": {"revision": "b" * 40}},
        "trials": {
            "piece_s": [3.0, 1.5, 6.0],
            "max_pieces": 100,
            "model_pieces": 30,
            "rival_pieces": 20,
            "cluster_n": [10, 20, 30],
            "reported_n": [5],
            "draws": 10,
            "anchor_k": [1, 2, 4],
            "bootstrap": 5,
        },
        "datasets": datasets,
    }
    plan.update(overrides)
    (data / FOLDER_REL).mkdir(parents=True, exist_ok=True)
    (data / PLAN_REL).write_text(json.dumps(plan), encoding="utf-8")
    return plan


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> list[ToneEmbedder]:
    embedders = [ToneEmbedder("wavlm-sv"), ToneEmbedder("ecapa")]
    monkeypatch.setattr(calibrate, "load_embedders", lambda: embedders)
    return embedders


def _decoys(data: Path) -> None:
    for rel in ("rehearsal/build-plan.json", "rehearsal/build.json", "config/gates.json"):
        (data / rel).parent.mkdir(parents=True, exist_ok=True)
        (data / rel).write_text("{}", encoding="utf-8")


def test_main_writes_calibration_and_outputs_beside_the_plan(
    tmp_path: Path, fakes: list[ToneEmbedder], capsys: pytest.CaptureFixture[str]
) -> None:
    data = tmp_path / "data"
    write_plan(data)
    _decoys(data)
    _OPENED.clear()
    _RECORDING[0] = True
    try:
        code = calibrate.main(["--data", str(data), "--plan", PLAN_REL])
    finally:
        _RECORDING[0] = False
    out, err = capsys.readouterr()
    assert code == 0, err
    opened = [Path(p).name for p in _OPENED]
    assert not {"build-plan.json", "build.json", "gates.json"} & set(opened)
    result = json.loads((data / FOLDER_REL / "calibration.json").read_text(encoding="utf-8"))
    outputs = json.loads((data / FOLDER_REL / "outputs.json").read_text(encoding="utf-8"))
    rel = f"{FOLDER_REL}/calibration.json"
    digest = hashlib.sha256((data / rel).read_bytes()).hexdigest()
    assert outputs == {"schema": 1, "files": [{"rel": rel, "sha256": digest}]}
    keys = {
        f"{g}.{n}" for g in (calibrate.SIM_MIN, calibrate.FLOOR, calibrate.MARGIN) for n in NAMES
    }
    assert set(result["keys"]) == keys
    assert (result["id"], result["alpha"], result["seed"]) == ("c1", 0.01, 20261116)
    assert result["embedders"] == {
        "wavlm-sv": {"revision": "a" * 40},
        "ecapa": {"revision": "b" * 40},
    }
    assert [d["name"] for d in result["datasets"]] == ["librittsr", "ravdess"]
    libri = result["datasets"][0]
    assert libri["speakers"] == {
        "female": {"eligible": 3, "excluded": 0},
        "male": {"eligible": 3, "excluded": 1},
    }
    assert libri["excluded"] == [{"speaker": "7777", "sex": "male", "reason": "build"}]
    assert result["datasets"][1]["speakers"]["female"]["eligible"] == 3
    floor = result["gates"][calibrate.FLOOR]
    assert set(floor["forms"]) == {"librittsr", "ravdess"}
    for gate in result["gates"].values():
        for forms in gate["forms"].values():
            for form in forms.values():
                assert form["joint_far"] <= 0.01
                assert form["trials"] > 0 and form["pairs"] == 12
    assert set(floor["reported"]["librittsr"]) == {"solo/n5", "centroid/n5"}
    assert set(floor["intervals"]) == set(NAMES)
    for name in NAMES:
        assert result["keys"][f"{calibrate.FLOOR}.{name}"] == floor["values"][name]
    for key in keys:
        assert key in out
    assert "beta" in out and "FAR" in out
    for text in (out, err):
        assert re.search(r"\.(wav|zip)\b|public/", text) is None


def test_main_checks_every_sha256_before_reading_any_audio(
    tmp_path: Path,
    fakes: list[ToneEmbedder],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data = tmp_path / "data"
    plan = write_plan(data, ravdess=False)
    last = plan["datasets"][0]["speakers"][-1]["files"][-1][0]
    (data / last).write_bytes(b"changed")
    monkeypatch.setattr(calibrate, "read_audio", _no_read)
    assert calibrate.main(["--data", str(data), "--plan", PLAN_REL]) == 1
    err = capsys.readouterr().err
    assert "6666" in err and "sha256" in err
    assert ".wav" not in err
    assert fakes[0].calls == 0
    assert not (data / FOLDER_REL / "calibration.json").exists()


def test_main_checks_the_archive_sha256(
    tmp_path: Path, fakes: list[ToneEmbedder], capsys: pytest.CaptureFixture[str]
) -> None:
    data = tmp_path / "data"
    plan = write_plan(data)
    plan["datasets"][1]["archive"][1] = "0" * 64
    (data / PLAN_REL).write_text(json.dumps(plan), encoding="utf-8")
    assert calibrate.main(["--data", str(data), "--plan", PLAN_REL]) == 1
    err = capsys.readouterr().err
    assert "ravdess" in err and "sha256" in err and ".zip" not in err


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"alpha": 0.01234}, "alpha"),
        ({"seed": "x"}, "seed"),
        ({"schema": 2}, "schema"),
        ({"embedders": {}}, "embedder"),
        ({"trials": {"piece_s": [3.0]}}, "trials"),
    ],
)
def test_a_malformed_plan_refuses(
    tmp_path: Path,
    fakes: list[ToneEmbedder],
    capsys: pytest.CaptureFixture[str],
    change: dict,
    message: str,
) -> None:
    data = tmp_path / "data"
    (data / FOLDER_REL).mkdir(parents=True)
    plan = {
        "schema": 1,
        "kind": "calibration-plan",
        "id": "c1",
        "alpha": 0.01,
        "seed": 1,
        "embedders": {"wavlm-sv": {"revision": "a"}, "ecapa": {"revision": "b"}},
        "trials": dataclasses.asdict(calibrate.TrialParams()),
        "datasets": [
            {
                "name": "librittsr",
                "licence": "x",
                "subset": "y",
                "speakers": [
                    {"speaker": "1", "sex": "male", "files": [["public/a.wav", "0" * 64]]}
                ],
            }
        ],
    }
    plan.update(change)
    (data / PLAN_REL).write_text(json.dumps(plan), encoding="utf-8")
    assert calibrate.main(["--data", str(data), "--plan", PLAN_REL]) == 1
    assert message in capsys.readouterr().err


def test_a_speaker_with_no_files_or_a_bad_path_refuses(
    tmp_path: Path, fakes: list[ToneEmbedder], capsys: pytest.CaptureFixture[str]
) -> None:
    data = tmp_path / "data"
    (data / FOLDER_REL).mkdir(parents=True)
    plan = {
        "schema": 1,
        "kind": "calibration-plan",
        "id": "c1",
        "alpha": 0.01,
        "seed": 1,
        "embedders": {"wavlm-sv": {"revision": "a"}, "ecapa": {"revision": "b"}},
        "trials": dataclasses.asdict(calibrate.TrialParams()),
        "datasets": [
            {
                "name": "librittsr",
                "licence": "x",
                "subset": "y",
                "speakers": [
                    {"speaker": "12", "sex": "male", "files": []},
                    {"speaker": "34", "sex": "male", "files": [["../a.wav", "0" * 64]]},
                ],
            }
        ],
    }
    (data / PLAN_REL).write_text(json.dumps(plan), encoding="utf-8")
    assert calibrate.main(["--data", str(data), "--plan", PLAN_REL]) == 1
    err = capsys.readouterr().err
    assert "12" in err and "34" in err and "a.wav" not in err
