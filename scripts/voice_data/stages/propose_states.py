"""State proposals for kept segments, from `state_rules.json` (Design 4, R2).

Each kept segment of a run gets four features: its arousal quantile within the run, and the
z-scores within the run of its F0 mean, speech rate and active energy. Relative features carry
the state contrast of one speaker in one show, whatever her absolute pitch and the mix level. A
segment is proposed the state whose centre in `state_rules.json` is nearest under the feature
weights, and its margin is how much nearer that centre is than the next. Raj confirms every
proposal in the labels pack; a proposal gates nothing.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..errors import InputError
from ..records import Prosody
from ..states import SPEAKING_STATES

RULES_PATH = Path(__file__).resolve().parents[1] / "state_rules.json"
FEATURES = ("arousal_q", "f0_z", "rate_z", "energy_z")
COMMENT = "[E]"


@dataclass(frozen=True, slots=True)
class Rules:
    """Feature weights and one centre per state, in the speaking states' order."""

    weights: dict[str, float]
    centres: dict[str, dict[str, float]]


@dataclass(frozen=True, slots=True)
class Proposal:
    """A proposed state, its margin over the next-nearest state, and the features behind it."""

    state: str
    margin: float
    features: dict[str, float]


def _centre(node: object, where: str) -> dict[str, float]:
    if not isinstance(node, Mapping):
        raise InputError(f"state rules: {where} is not an object")
    values = {k: v for k, v in node.items() if k != COMMENT}
    if set(values) != set(FEATURES):
        raise InputError(f"state rules: {where} must give exactly {', '.join(FEATURES)}")
    if not all(isinstance(v, int | float) and not isinstance(v, bool) for v in values.values()):
        raise InputError(f"state rules: {where} holds a value that is not a number")
    return {k: float(values[k]) for k in FEATURES}


def load_rules(path: Path = RULES_PATH) -> Rules:
    """Load and check `state_rules.json`.

    Raises:
        InputError: The file is unreadable, or its features or states are not the expected ones.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputError("state rules: missing or not JSON") from exc
    features = data.get("features") if isinstance(data, dict) else None
    states = data.get("states") if isinstance(data, dict) else None
    if not isinstance(features, dict) or set(features) != set(FEATURES):
        raise InputError(f"state rules: features must be exactly {', '.join(FEATURES)}")
    if not isinstance(states, dict) or set(states) != set(SPEAKING_STATES):
        raise InputError("state rules: states must be exactly the speaking states")
    weights = {}
    for name in FEATURES:
        weight = features[name].get("weight") if isinstance(features[name], dict) else None
        if not isinstance(weight, int | float) or isinstance(weight, bool) or weight < 0:
            raise InputError(f"state rules: feature {name} has no non-negative weight")
        weights[name] = float(weight)
    centres = {}
    for state in SPEAKING_STATES:
        entry = states[state]
        centre = entry.get("centre") if isinstance(entry, dict) else None
        centres[state] = _centre(centre, f"state {state} centre")
    return Rules(weights=weights, centres=centres)


def _quantiles(values: Sequence[float]) -> list[float]:
    # Mid-ranks over n - 1, so the lowest is 0, the highest 1, and ties share a quantile.
    n = len(values)
    if n == 1:
        return [0.5]
    order = np.argsort(values, kind="stable")
    ranks = np.empty(n)
    ranks[order] = np.arange(n, dtype=np.float64)
    array = np.asarray(values, dtype=np.float64)
    for value in np.unique(array):
        tied = array == value
        ranks[tied] = ranks[tied].mean()
    return (ranks / (n - 1)).tolist()


def _z(values: Sequence[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    spread = float(array.std())
    if spread == 0.0:
        return [0.0] * len(values)
    return ((array - array.mean()) / spread).tolist()


def features(prosodies: Sequence[Prosody], arousals: Sequence[float]) -> list[dict[str, float]]:
    """Return each segment's features within its run: arousal quantile and three z-scores."""
    if not prosodies:
        return []
    columns = {
        "arousal_q": _quantiles(list(arousals)),
        "f0_z": _z([p.f0_mean_hz for p in prosodies]),
        "rate_z": _z([p.rate_sps for p in prosodies]),
        "energy_z": _z([p.energy_db for p in prosodies]),
    }
    return [{name: columns[name][i] for name in FEATURES} for i in range(len(prosodies))]


def propose(row: Mapping[str, float], rules: Rules) -> Proposal:
    """Propose the state of the nearest centre; the margin is the next centre's extra distance."""
    distances = {
        state: float(np.sqrt(sum(rules.weights[k] * (row[k] - centre[k]) ** 2 for k in FEATURES)))
        for state, centre in rules.centres.items()
    }
    ranked = sorted(distances, key=lambda s: distances[s])
    margin = distances[ranked[1]] - distances[ranked[0]] if len(ranked) > 1 else 0.0
    return Proposal(state=ranked[0], margin=margin, features={k: float(row[k]) for k in FEATURES})


def propose_all(
    prosodies: Sequence[Prosody], arousals: Sequence[float], rules: Rules
) -> list[Proposal]:
    """Propose a state for every kept segment of a run, in the order given."""
    return [propose(row, rules) for row in features(prosodies, arousals)]
