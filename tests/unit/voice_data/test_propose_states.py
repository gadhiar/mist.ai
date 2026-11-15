"""Tests for `stages.propose_states` and `state_rules.json`.

Each state's rule is a centre in the feature space of arousal quantile and the z-scores of F0,
speech rate and energy; a segment is proposed the state of the nearest centre, with the margin to
the second nearest. One test per rule puts a segment at that rule's centre.
"""

from __future__ import annotations

import json

import pytest

from scripts.voice_data.records import Prosody
from scripts.voice_data.stages import propose_states
from scripts.voice_data.states import SPEAKING_STATES


@pytest.fixture(scope="module")
def rules() -> propose_states.Rules:
    return propose_states.load_rules()


def test_the_rules_cover_exactly_the_speaking_states(rules: propose_states.Rules) -> None:
    assert tuple(rules.centres) == SPEAKING_STATES


def test_the_rules_use_the_four_features(rules: propose_states.Rules) -> None:
    assert set(rules.weights) == set(propose_states.FEATURES)
    for centre in rules.centres.values():
        assert set(centre) == set(propose_states.FEATURES)


def _numbers_without_e(node: object, path: str = "") -> list[str]:
    """Paths of objects holding a calibratable number but no `[E]` comment."""
    found: list[str] = []
    if isinstance(node, dict):
        numbers = [k for k, v in node.items() if isinstance(v, int | float) and k != "schema"]
        nested_numbers = [k for k, v in node.items() if isinstance(v, dict) and k != "[E]"]
        if numbers and not (isinstance(node.get("[E]"), str) and node["[E]"].strip()):
            found.append(path or "/")
        for key in nested_numbers:
            found.extend(_numbers_without_e(node[key], f"{path}/{key}"))
    return found


def test_every_calibrated_number_carries_an_e_comment() -> None:
    data = json.loads(propose_states.RULES_PATH.read_text(encoding="utf-8"))
    assert _numbers_without_e(data) == []


def test_the_checker_flags_a_number_without_an_e_comment() -> None:
    assert _numbers_without_e({"schema": 1, "a": {"x": 1.0}}) == ["/a"]
    assert _numbers_without_e({"a": {"x": 1.0, "[E]": "H6"}}) == []


@pytest.mark.parametrize("state", SPEAKING_STATES)
def test_each_rule_proposes_its_state_at_its_centre(
    rules: propose_states.Rules, state: str
) -> None:
    proposal = propose_states.propose(rules.centres[state], rules)
    assert proposal.state == state
    assert proposal.margin > 0.0


def test_the_margin_is_the_gap_to_the_second_nearest(rules: propose_states.Rules) -> None:
    calm, neutral = rules.centres["calm"], rules.centres["neutral"]
    midway = {k: (calm[k] + neutral[k]) / 2.0 for k in propose_states.FEATURES}
    proposal = propose_states.propose(midway, rules)
    assert proposal.state in ("calm", "neutral")
    assert proposal.margin == pytest.approx(0.0, abs=1e-9)
    near = {k: 0.9 * calm[k] + 0.1 * neutral[k] for k in propose_states.FEATURES}
    assert propose_states.propose(near, rules).margin > 0.0


def test_features_are_arousal_quantiles_and_z_scores() -> None:
    prosodies = [Prosody(f0_mean_hz=f, f0_std_hz=10.0, rate_sps=r, energy_db=e)
                 for f, r, e in [(180.0, 3.0, -30.0), (200.0, 4.0, -25.0), (220.0, 5.0, -20.0)]]  # fmt: skip
    rows = propose_states.features(prosodies, [0.9, 0.1, 0.5])
    assert [row["arousal_q"] for row in rows] == [1.0, 0.0, 0.5]
    assert [row["f0_z"] for row in rows] == pytest.approx([-1.2247449, 0.0, 1.2247449])
    assert rows[1]["rate_z"] == pytest.approx(0.0)
    assert rows[2]["energy_z"] > 0.0


def test_ties_in_arousal_share_a_quantile() -> None:
    prosody = Prosody(f0_mean_hz=200.0, f0_std_hz=0.0, rate_sps=4.0, energy_db=-20.0)
    rows = propose_states.features([prosody] * 3, [0.4, 0.4, 0.8])
    assert rows[0]["arousal_q"] == rows[1]["arousal_q"] == 0.25
    assert rows[2]["arousal_q"] == 1.0
    assert all(row["f0_z"] == 0.0 for row in rows)


def test_a_single_segment_sits_at_the_middle() -> None:
    prosody = Prosody(f0_mean_hz=200.0, f0_std_hz=0.0, rate_sps=4.0, energy_db=-20.0)
    (row,) = propose_states.features([prosody], [0.7])
    assert row == {"arousal_q": 0.5, "f0_z": 0.0, "rate_z": 0.0, "energy_z": 0.0}


def test_a_run_from_low_to_high_energy_spans_sleepy_to_excited(
    rules: propose_states.Rules,
) -> None:
    steps = [(150.0, 2.0, -40.0, 0.05), (170.0, 3.0, -33.0, 0.25), (200.0, 4.0, -26.0, 0.5),
             (230.0, 5.0, -19.0, 0.75), (260.0, 6.0, -12.0, 0.95)]  # fmt: skip
    prosodies = [Prosody(f0_mean_hz=f, f0_std_hz=10.0, rate_sps=r, energy_db=e)
                 for f, r, e, _ in steps]  # fmt: skip
    proposals = propose_states.propose_all(prosodies, [a for *_, a in steps], rules)
    assert [p.state for p in proposals] == ["sleepy", "calm", "neutral", "awake", "excited"]
    assert all(p.margin > 0.0 for p in proposals)


def test_no_segment_no_proposal(rules: propose_states.Rules) -> None:
    assert propose_states.propose_all([], [], rules) == []
