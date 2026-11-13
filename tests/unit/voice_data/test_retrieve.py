"""Tests for `stages.retrieve`: the anchor check, both retrieval passes and the separator score.

Retrieval works on embeddings only, so these tests build them by design: each voice is a unit
vector at a chosen cosine to the target's, and the second embedder sees the same voices through
a fixed rotation, so every cosine holds on both. Small deterministic jitter makes each turn a
little different, as real turns are. The end-to-end cases on synthetic audio are in
`test_extract.py`.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.voice_data.stages import retrieve

DIM = 8
EMBEDDERS = ("wavlm-sv", "ecapa")
_ROTATION = np.linalg.qr(np.random.default_rng(7).standard_normal((DIM, DIM)))[0]


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def voice(cos: float, axis: int) -> np.ndarray:
    """A voice at cosine `cos` to the target (axis 0), leaning towards `axis` otherwise."""
    v = np.zeros(DIM)
    v[0] = cos
    v[axis] = np.sqrt(1.0 - cos**2)
    return v


def emb(base: np.ndarray, seed: int, jitter: float = 0.01) -> retrieve.Embedding:
    """One clip of a voice on both embedders, with deterministic jitter."""
    noise = np.random.default_rng(seed).standard_normal(DIM) * jitter
    v = _unit(base + noise)
    return {"wavlm-sv": v.astype(np.float32), "ecapa": (_ROTATION @ v).astype(np.float32)}


TARGET = voice(1.0, 1)
IMPOSTOR = voice(0.8, 2)
OTHER_A = voice(0.1, 3)
OTHER_B = voice(0.05, 4)


def _turns(
    label: str, base: np.ndarray, n: int, seed: int, jitter: float = 0.01
) -> list[tuple[str, retrieve.Embedding]]:
    return [(label, emb(base, seed * 1000 + i, jitter)) for i in range(n)]


def _anchor_centroid() -> dict[str, np.ndarray]:
    return retrieve.centroids([emb(TARGET, 90_000 + i) for i in range(3)])


# The anchor check.


def test_agreeing_anchors_pass() -> None:
    ok, sims = retrieve.anchor_check([emb(TARGET, i) for i in range(3)], sim_min=0.6)
    assert ok
    assert len(sims) == 3
    assert all(s[name] > 0.99 for s in sims for name in EMBEDDERS)


def test_an_anchor_of_another_voice_disagrees() -> None:
    anchors = [emb(TARGET, 1), emb(TARGET, 2), emb(OTHER_A, 3)]
    ok, sims = retrieve.anchor_check(anchors, sim_min=0.6)
    assert not ok
    assert sims[2]["wavlm-sv"] < 0.6


def test_a_single_anchor_has_nothing_to_disagree_with() -> None:
    ok, sims = retrieve.anchor_check([emb(TARGET, 1)], sim_min=0.6)
    assert ok
    assert sims == [{}]


def test_anchors_must_agree_on_both_embedders() -> None:
    odd = {"wavlm-sv": emb(TARGET, 3)["wavlm-sv"], "ecapa": emb(OTHER_A, 3)["ecapa"]}
    ok, _ = retrieve.anchor_check([emb(TARGET, 1), emb(TARGET, 2), odd], sim_min=0.6)
    assert not ok


# The first pass.


def _roles(turns: list[tuple[str, retrieve.Embedding]]) -> dict[str, retrieve.ClusterRole]:
    return retrieve.cluster_roles(turns, _anchor_centroid(), floor=0.6)


def test_the_nearest_cluster_is_primary_and_far_ones_are_others() -> None:
    turns = _turns("spk-01", OTHER_A, 5, 1) + _turns("spk-02", TARGET, 8, 2)
    turns += _turns("spk-03", OTHER_B, 5, 3)
    roles = _roles(turns)
    assert {k: r.role for k, r in roles.items()} == {
        "spk-01": "other",
        "spk-02": "primary",
        "spk-03": "other",
    }
    assert roles["spk-02"].turns == 8
    assert roles["spk-02"].median_sim["wavlm-sv"] > 0.99


def test_an_impostor_at_cosine_0_8_is_flagged_not_admitted() -> None:
    turns = _turns("spk-01", TARGET, 8, 1) + _turns("spk-02", IMPOSTOR, 6, 2)
    roles = _roles(turns)
    assert roles["spk-01"].role == "primary"
    assert roles["spk-02"].role == "flagged"


def test_a_cluster_below_the_floor_on_one_embedder_is_not_flagged() -> None:
    half = [("spk-09", {"wavlm-sv": e["wavlm-sv"], "ecapa": o["ecapa"]}) for e, o in
            zip([emb(TARGET, 50 + i) for i in range(4)], [emb(OTHER_A, 60 + i) for i in range(4)],
                strict=True)]  # fmt: skip
    roles = _roles(_turns("spk-01", TARGET, 6, 1) + half)
    assert roles["spk-09"].role == "other"


def test_no_cluster_reaching_the_floor_means_no_primary() -> None:
    roles = _roles(_turns("spk-01", OTHER_A, 4, 1) + _turns("spk-02", OTHER_B, 4, 2))
    assert {r.role for r in roles.values()} == {"other"}


def test_equal_medians_go_to_the_cluster_with_more_turns() -> None:
    exact = {
        "wavlm-sv": TARGET.astype(np.float32),
        "ecapa": (_ROTATION @ TARGET).astype(np.float32),
    }
    turns = [("spk-01", exact)] * 3 + [("spk-02", exact)] * 5
    roles = _roles(turns)
    assert roles["spk-02"].role == "primary"
    assert roles["spk-01"].role == "flagged"


# The second pass.


def _episode(*, impostor: bool) -> list[tuple[str, retrieve.Embedding]]:
    turns = _turns("spk-01", TARGET, 12, 1) + _turns("spk-02", OTHER_A, 6, 2)
    turns += _turns("spk-03", OTHER_B, 6, 3)
    if impostor:
        turns += _turns("spk-04", IMPOSTOR, 6, 4)
    return turns


def _model(turns: list[tuple[str, retrieve.Embedding]]) -> retrieve.TargetModel:
    roles = _roles(turns)
    primary = [e for label, e in turns if roles[label].role == "primary"]
    model = retrieve.target_model(primary, ["src-0001"] * len(primary), percentile=2.0)
    assert model is not None
    return model


def test_the_impostor_does_not_move_the_centroid() -> None:
    with_impostor = _model(_episode(impostor=True))
    without = _model(_episode(impostor=False))
    for name in EMBEDDERS:
        np.testing.assert_array_equal(with_impostor.centroid[name], without.centroid[name])
        assert with_impostor.floor[name] == without.floor[name]


def test_the_impostors_turns_fail_the_floor() -> None:
    turns = _episode(impostor=True)
    model = _model(turns)
    roles = _roles(turns)
    rivals = retrieve.rival_centroids(turns, roles)
    for label, e in turns:
        if label != "spk-04":
            continue
        verdict = retrieve.judge(e, model, rivals, margin_min=0.1)
        assert not verdict.passed
        assert "floor:wavlm-sv" in verdict.reasons
        assert "floor:ecapa" in verdict.reasons


def test_rivals_are_the_non_admitted_clusters_only() -> None:
    turns = _episode(impostor=True)
    roles = _roles(turns)
    rivals = retrieve.rival_centroids(turns, roles)
    assert sorted(rivals) == ["spk-02", "spk-03"]


def test_a_split_register_is_kept_where_an_admitted_rival_would_drop_it() -> None:
    # One voice in two registers: the embedders place both at the target, as a split by
    # register leaves the voice unchanged; only the diarizer tells them apart.
    turns = _turns("spk-01", TARGET, 12, 1, jitter=0.0) + _turns("spk-05", TARGET, 6, 5, jitter=0.0)
    turns += _turns("spk-02", OTHER_A, 6, 2)
    roles = _roles(turns)
    assert roles["spk-01"].role == "primary"
    assert roles["spk-05"].role == "flagged"
    model = _model(turns)
    rivals = retrieve.rival_centroids(turns, roles)
    excited_turns = [e for label, e in turns if label == "spk-05"]
    kept = [retrieve.judge(e, model, rivals, margin_min=0.1) for e in excited_turns]
    floor_failures = [v for v in kept if any(r.startswith("floor") for r in v.reasons)]
    assert not [r for v in kept for r in v.reasons if r.startswith("margin")]
    assert len(floor_failures) == 0
    # Had the flagged cluster been a rival, the margin would have dropped every excited turn.
    with_admitted = dict(rivals, **retrieve.centroids_by_label(turns, {"spk-05"}))
    dropped = [retrieve.judge(e, model, with_admitted, margin_min=0.1) for e in excited_turns]
    assert all(any(r.startswith("margin") for r in v.reasons) for v in dropped)


def test_target_turns_pass_with_their_margin() -> None:
    turns = _episode(impostor=False)
    roles = _roles(turns)
    model = _model(turns)
    rivals = retrieve.rival_centroids(turns, roles)
    verdicts = [retrieve.judge(e, model, rivals, 0.1) for label, e in turns if label == "spk-01"]
    passed = [v for v in verdicts if v.passed]
    assert len(passed) >= 11
    assert all(v.margin is not None and v.margin > 0.5 for v in passed)


def test_a_turn_too_close_to_a_rival_fails_the_margin() -> None:
    model = _model(_episode(impostor=False))
    rival = emb(voice(0.97, 6), 77)
    turn = emb(voice(0.985, 6), 78)
    verdict = retrieve.judge(turn, model, {"spk-07": dict(rival)}, margin_min=0.1)
    assert "margin:wavlm-sv" in verdict.reasons


def test_no_rival_means_no_margin_and_no_margin_reason() -> None:
    model = _model(_episode(impostor=False))
    verdict = retrieve.judge(emb(TARGET, 5), model, {}, margin_min=0.1)
    assert verdict.margin is None
    assert verdict.passed


def test_the_floor_is_inclusive_up_to_rounding() -> None:
    exact = {
        "wavlm-sv": TARGET.astype(np.float32),
        "ecapa": (_ROTATION @ TARGET).astype(np.float32),
    }
    model = retrieve.target_model([exact] * 7, ["src-0001"] * 7, percentile=2.0)
    assert model is not None
    assert retrieve.judge(exact, model, {}, margin_min=0.1).passed


def test_the_target_model_needs_two_turns() -> None:
    assert retrieve.target_model([emb(TARGET, 1)], ["src-0001"], percentile=2.0) is None


def test_the_distribution_is_leave_one_out_across_sources() -> None:
    embs = [emb(TARGET, i) for i in range(6)]
    model = retrieve.target_model(embs, ["src-0001"] * 3 + ["src-0002"] * 3, percentile=2.0)
    assert model is not None
    dist = model.distribution["wavlm-sv"]
    assert dist.n == 6
    assert dist.p02 <= dist.p10 <= dist.median < 1.0
    assert model.floor["wavlm-sv"] == pytest.approx(dist.p02)


# The separator score.


def test_candidate_score_is_the_median_of_the_best_windows() -> None:
    reference = {"wavlm-sv": TARGET, "ecapa": _ROTATION @ TARGET}
    good = [emb(TARGET, i) for i in range(60)]
    bad = [emb(OTHER_A, 100 + i) for i in range(40)]
    score = retrieve.candidate_score(good + bad, reference, best_n=50)
    assert score is not None and score > 0.99
    assert retrieve.candidate_score(bad, reference, best_n=50) < 0.2  # type: ignore[operator]


def test_candidate_score_of_no_window_is_none() -> None:
    assert retrieve.candidate_score([], {"wavlm-sv": TARGET}, best_n=50) is None


def test_choose_separator_takes_the_best_and_ties_go_to_the_first() -> None:
    assert retrieve.choose_separator({"none": 0.9, "dme": 0.95, "roformer": 0.7}) == "dme"
    assert retrieve.choose_separator({"none": 0.9, "dme": 0.9}) == "none"
    assert retrieve.choose_separator({"none": None, "dme": 0.5}) == "dme"
    assert retrieve.choose_separator({"none": None, "dme": None}) == "none"
