"""Tests for backend.memory.need_gate: the calibrated probability that a query needs memory.

Every bank and model file here is built in a temporary directory; the shipped bank and model
files are other tasks' artifacts and are never read. Vectors come from a fake embedder that
returns fixed vectors, so each probability can be derived by hand.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from backend.errors import MemoryConfigError
from backend.memory import need_gate
from backend.memory.need_gate import (
    DEFAULT_BANK_PATHS,
    DEFAULT_MODEL_PATH,
    FORM_CANDIDATES,
    NEED_EMBEDDERS,
    NEWTON_TOLERANCE,
    CvResult,
    FormSpec,
    NeedBank,
    NeedEmbedderSpec,
    NeedExemplar,
    NeedFitError,
    NeedGate,
    NeedKey,
    NeedModel,
    auroc,
    bank_sha256,
    choose_form,
    context_text,
    cv_score,
    fit_form,
    grouped_cv,
    load_bank,
    load_models,
    need_embedder_kwargs,
    need_model_key,
    probabilities,
    select_model,
    theta,
)

LOGISTIC_1 = FormSpec("logistic", 1.0, None)


# --- helpers -----------------------------------------------------------------------------------


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def _unit(rows: Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(rows, dtype=np.float64)
    return array / np.linalg.norm(array, axis=1, keepdims=True)


def _balanced_weights(labels: Sequence[str]) -> np.ndarray:
    y = np.array([label == "need" for label in labels])
    n = len(labels)
    return np.where(y, n / (2 * y.sum()), n / (2 * (~y).sum()))


def _logistic_gradient(vectors, labels, c: float, w, b: float) -> np.ndarray:
    """The gradient of 0.5 * ||w||^2 + C * sum_i s_i * logloss_i, derived independently."""
    x = _unit(vectors)
    y = np.array([label == "need" for label in labels], dtype=np.float64)
    s = _balanced_weights(labels)
    z = x @ np.asarray(w) + b
    p = 1.0 / (1.0 + np.exp(-z))
    residual = c * s * (p - y)
    return np.concatenate([np.asarray(w) + x.T @ residual, [residual.sum()]])


def _hand_margins(vectors, labels, n: int, *, leave_one_out: bool) -> np.ndarray:
    """Mean of the n highest cosines to need rows minus the same to no-need rows (v7's margin)."""
    x = _unit(vectors)
    margins = []
    for i, row in enumerate(x):
        need_cos, no_need_cos = [], []
        for j, other in enumerate(x):
            if leave_one_out and i == j:
                continue
            (need_cos if labels[j] == "need" else no_need_cos).append(float(row @ other))
        top_need = sorted(need_cos, reverse=True)[:n]
        top_no = sorted(no_need_cos, reverse=True)[:n]
        margins.append(sum(top_need) / n - sum(top_no) / n)
    return np.array(margins)


def _platt_gradient(margins, labels, a: float, c: float) -> np.ndarray:
    """The gradient of sum_i s_i * logloss(t_i, sigmoid(a * m_i + c)) with Platt's targets."""
    y = np.array([label == "need" for label in labels])
    n_need, n_no = int(y.sum()), int((~y).sum())
    targets = np.where(y, (n_need + 1) / (n_need + 2), 1 / (n_no + 2))
    s = _balanced_weights(labels)
    p = 1.0 / (1.0 + np.exp(-(a * np.asarray(margins) + c)))
    residual = s * (p - targets)
    return np.array([residual @ np.asarray(margins), residual.sum()])


# Separable: need rows lean on the first axis, no-need rows on the second.
SEPARABLE_VECTORS = [
    [1.0, 0.1, 0.0],
    [0.9, 0.2, 0.1],
    [0.8, 0.0, 0.3],
    [0.1, 1.0, 0.0],
    [0.2, 0.9, 0.2],
    [0.0, 0.8, 0.4],
]
SEPARABLE_LABELS = ["need", "need", "need", "no_need", "no_need", "no_need"]

# Not separable: one row of each label shares a vector with a row of the other.
OVERLAP_VECTORS = [
    [1.0, 0.1, 0.0],
    [0.5, 0.5, 0.2],
    [0.2, 0.8, 0.1],
    [0.1, 1.0, 0.0],
    [0.5, 0.5, 0.2],
    [0.9, 0.2, 0.1],
]
OVERLAP_LABELS = ["need", "need", "need", "no_need", "no_need", "no_need"]


class FakeEmbedder:
    """Returns the fixed vector mapped to each text and records every call."""

    def __init__(self, vectors: dict[str, Sequence[float]]) -> None:
        self.vectors = vectors
        self.single_calls: list[str] = []
        self.batch_calls: list[list[str]] = []

    def generate_embedding(self, text: str) -> list[float]:
        self.single_calls.append(text)
        return list(self.vectors[text])

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        self.batch_calls.append(list(texts))
        return [list(self.vectors[text]) for text in texts]


def _row(
    id_: str, writer: str, label: str, text: str, *, hard: bool = False, category: str = "cat"
) -> dict:
    return {
        "id": id_,
        "writer": writer,
        "label": label,
        "category": category,
        "hard": hard,
        "text": text,
    }


def _write_jsonl(path: Path, rows: Sequence[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _two_writer_files(tmp_path: Path) -> tuple[Path, Path]:
    a = _write_jsonl(
        tmp_path / "writer_a.jsonl",
        [
            _row("a-need-001", "a", "need", "what did I say about my sister"),
            _row("a-no-need-001", "a", "no_need", "turn the lights off"),
        ],
    )
    b = _write_jsonl(
        tmp_path / "writer_b.jsonl",
        [
            _row("b-need-001", "b", "need", "play my usual"),
            _row("b-no-need-001", "b", "no_need", "what is the capital of peru", hard=True),
        ],
    )
    return a, b


# Gate bank: four need rows near the first axis, four no-need rows near the second; in bank
# order the cosines to QUERY_AXIS1 are not monotone, so the nearest-first order is tested.
GATE_BANK = [
    ("n1", "need", [1.0, 0.30, 0.0]),
    ("n2", "need", [1.0, 0.05, 0.0]),
    ("n3", "need", [1.0, 0.60, 0.0]),
    ("n4", "need", [1.0, 0.15, 0.0]),
    ("o1", "no_need", [0.20, 1.0, 0.0]),
    ("o2", "no_need", [0.60, 1.0, 0.0]),
    ("o3", "no_need", [0.05, 1.0, 0.0]),
    ("o4", "no_need", [0.40, 1.0, 0.0]),
]
PREFIX = "query: "


def _gate_bank(sha: str = "f" * 64) -> NeedBank:
    exemplars = tuple(
        NeedExemplar(id_, "a", label, "cat", False, f"text {id_}") for id_, label, _ in GATE_BANK
    )
    return NeedBank(exemplars, sha)


def _gate_embedder(prefix: str = PREFIX) -> FakeEmbedder:
    vectors = {f"{prefix}text {id_}": vec for id_, _, vec in GATE_BANK}
    return FakeEmbedder(vectors)


def _model(
    key: NeedKey,
    *,
    form: str = "logistic",
    params: dict | None = None,
    theta_low: float = 0.3,
    theta_high: float = 0.7,
    band_k: int = 2,
) -> NeedModel:
    if params is None:
        params = {"c": 1.0, "w": [4.0, -4.0, 0.0], "b": 0.0}
    return NeedModel(
        need_model_key(key),
        key.bank_sha256,
        key.embedder_name,
        key.embedder_revision,
        key.query_prefix,
        key.normalisation,
        form,
        params,
        theta_low,
        theta_high,
        band_k,
        0.95,
        "H18 test",
    )


GATE_KEY = NeedKey("f" * 64, "bge-small-en-v1.5", "rev1", PREFIX)


def _gate(**overrides) -> NeedGate:
    embedder = overrides.pop("embedder", _gate_embedder())
    model = overrides.pop("model", _model(GATE_KEY))
    kwargs = {
        "embedder_name": GATE_KEY.embedder_name,
        "embedder_revision": GATE_KEY.embedder_revision,
        "query_prefix": GATE_KEY.query_prefix,
    }
    kwargs.update(overrides)
    return NeedGate(embedder, _gate_bank(), model, **kwargs)


# --- module shape --------------------------------------------------------------------------------


class TestModuleImports:
    def test_uses_numpy_and_neither_scikit_learn_nor_scipy(self):
        tree = ast.parse(Path(need_gate.__file__).read_text(encoding="utf-8"))
        roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                roots.add(node.module.split(".")[0])
        assert "numpy" in roots
        assert not roots & {"sklearn", "scipy"}


class TestDefaults:
    def test_default_paths_name_files_beside_the_module(self):
        here = Path(need_gate.__file__).parent
        assert (
            here / "need_bank" / "writer_a.jsonl",
            here / "need_bank" / "writer_b.jsonl",
        ) == DEFAULT_BANK_PATHS
        assert here / "need_model.json" == DEFAULT_MODEL_PATH

    def test_form_candidates(self):
        assert (
            FormSpec("logistic", 0.1, None),
            FormSpec("logistic", 1.0, None),
            FormSpec("logistic", 10.0, None),
            FormSpec("logistic", 100.0, None),
            FormSpec("knn", None, 1),
            FormSpec("knn", None, 3),
            FormSpec("knn", None, 5),
        ) == FORM_CANDIDATES

    def test_dataclasses_are_frozen_and_slotted(self):
        for cls in (
            NeedExemplar,
            NeedBank,
            NeedKey,
            NeedModel,
            NeedEmbedderSpec,
            FormSpec,
            CvResult,
            need_gate.NeedDecision,
        ):
            assert cls.__dataclass_params__.frozen, cls
            assert hasattr(cls, "__slots__"), cls

    def test_form_spec_refuses_a_mismatched_parameter(self):
        with pytest.raises(ValueError):
            FormSpec("logistic", None, 1)
        with pytest.raises(ValueError):
            FormSpec("knn", 1.0, None)
        with pytest.raises(ValueError):
            FormSpec("knn", None, 0)


class TestEmbedders:
    def test_need_embedders_hold_the_prefixes_and_attention_settings(self):
        assert dict(NEED_EMBEDDERS) == {
            "shared": NeedEmbedderSpec("", None),
            "bge-small-en-v1.5": NeedEmbedderSpec(
                "Represent this sentence for searching relevant passages: ", None
            ),
            "granite-embedding-small-english-r2": NeedEmbedderSpec("", "sdpa"),
        }

    def test_kwargs_empty_without_an_attention_setting(self):
        assert need_embedder_kwargs(NEED_EMBEDDERS["shared"]) == {}
        assert need_embedder_kwargs(NEED_EMBEDDERS["bge-small-en-v1.5"]) == {}

    def test_kwargs_carry_the_one_keyword_with_an_attention_setting(self):
        spec = NEED_EMBEDDERS["granite-embedding-small-english-r2"]
        assert need_embedder_kwargs(spec) == {"attn_implementation": "sdpa"}


class TestContextText:
    def test_joins_with_one_newline(self):
        assert context_text("tell me about my sister", "and her job?") == (
            "tell me about my sister\nand her job?"
        )


# --- the logistic form ---------------------------------------------------------------------------


class TestLogisticFit:
    @pytest.mark.parametrize("c", [1.0, 100.0])
    @pytest.mark.parametrize(
        "vectors,labels",
        [(SEPARABLE_VECTORS, SEPARABLE_LABELS), (OVERLAP_VECTORS, OVERLAP_LABELS)],
        ids=["separable", "non-separable"],
    )
    def test_gradient_of_the_stated_objective_is_below_tolerance(self, vectors, labels, c):
        params = fit_form(vectors, labels, FormSpec("logistic", c, None))
        assert params["c"] == c
        gradient = _logistic_gradient(vectors, labels, c, params["w"], params["b"])
        assert np.max(np.abs(gradient)) < NEWTON_TOLERANCE
        assert NEWTON_TOLERANCE == 1e-8

    @pytest.mark.parametrize("c", [1.0, 10.0])
    def test_intercept_is_not_penalised(self, c):
        # Two need rows at e1, two no-need rows at +e2 and -e2. The no-need rows cancel in the
        # weight gradient, so w = (w1, 0) and the no-need logit is b. The unpenalised intercept
        # condition makes the two labels' logits opposite: b = -(w1 + b), so b = -w1 / 2; the
        # weight condition is w1 = K * sigmoid(-(w1 + b)) with K = C * N / 2 (the balanced
        # weights give each label N / 2). With t = w1 / 2: t = (K / 2) * sigmoid(-t), b = -t.
        vectors = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
        labels = ["need", "need", "no_need", "no_need"]
        k = c * len(labels) / 2
        lo, hi = 0.0, k
        for _ in range(200):
            mid = (lo + hi) / 2
            if mid - (k / 2) * _sigmoid(-mid) > 0:
                hi = mid
            else:
                lo = mid
        t = (lo + hi) / 2
        params = fit_form(vectors, labels, FormSpec("logistic", c, None))
        assert params["b"] == pytest.approx(-t, abs=1e-7)
        assert params["w"] == pytest.approx([2 * t, 0.0], abs=1e-7)

    def test_each_label_weighs_equally_in_total(self):
        # A 3-to-1 bank and the same bank with its minority row repeated three times put the
        # same ratio of weight on each label. The balanced weights keep the total weight at N,
        # so the repeated bank (N' rows) matches at C * N / N'.
        vectors = [[1.0, 0.2, 0.1], [0.3, 1.0, 0.0], [0.1, 0.9, 0.4], [0.5, 0.7, 0.2]]
        labels = ["need", "no_need", "no_need", "no_need"]
        repeated_vectors = vectors + [vectors[0], vectors[0]]
        repeated_labels = labels + ["need", "need"]
        c = 10.0
        plain = fit_form(vectors, labels, FormSpec("logistic", c, None))
        repeated = fit_form(
            repeated_vectors,
            repeated_labels,
            FormSpec("logistic", c * len(labels) / len(repeated_labels), None),
        )
        assert plain["w"] == pytest.approx(repeated["w"], abs=1e-7)
        assert plain["b"] == pytest.approx(repeated["b"], abs=1e-7)
        # Without the weights the 3-to-1 fit would differ: the stated objective's gradient is
        # zero only with them.
        gradient = _logistic_gradient(vectors, labels, c, plain["w"], plain["b"])
        assert np.max(np.abs(gradient)) < NEWTON_TOLERANCE

    def test_repeated_fits_are_bit_identical(self):
        first = fit_form(OVERLAP_VECTORS, OVERLAP_LABELS, FormSpec("logistic", 10.0, None))
        second = fit_form(OVERLAP_VECTORS, OVERLAP_LABELS, FormSpec("logistic", 10.0, None))
        assert first == second
        assert [float(v).hex() for v in first["w"]] == [float(v).hex() for v in second["w"]]

    def test_params_are_plain_json_values(self):
        params = fit_form(SEPARABLE_VECTORS, SEPARABLE_LABELS, LOGISTIC_1)
        assert set(params) == {"c", "w", "b"}
        assert json.loads(json.dumps(params)) == params

    def test_probabilities_are_the_sigmoid_of_the_normalised_logit(self):
        params = {"c": 1.0, "w": [2.0, -1.0], "b": 0.5}
        p = probabilities([[3.0, 4.0]], "logistic", params)
        assert p[0] == pytest.approx(_sigmoid(2.0 * 0.6 - 1.0 * 0.8 + 0.5), abs=1e-15)


# --- the knn form --------------------------------------------------------------------------------


class TestKnn:
    @pytest.mark.parametrize("n", [1, 3])
    def test_margin_is_the_hand_computed_mean(self, n):
        bank = [
            [1.0, 0.0, 0.0],
            [0.8, 0.6, 0.0],
            [0.6, 0.0, 0.8],
            [0.0, 1.0, 0.0],
            [0.0, 0.6, 0.8],
            [0.0, 0.0, 1.0],
        ]
        bank_labels = ["need", "need", "need", "no_need", "no_need", "no_need"]
        query = [0.6, 0.8, 0.0]
        # cosines to need rows: 0.6, 0.96, 0.36; to no-need rows: 0.8, 0.48, 0.0
        margins = {1: 0.96 - 0.8, 3: (0.96 + 0.6 + 0.36) / 3 - (0.8 + 0.48 + 0.0) / 3}
        margin = margins[n]
        p = probabilities(
            [query],
            "knn",
            {"n": n, "a": 1.0, "c": 0.0},
            bank_vectors=bank,
            bank_labels=bank_labels,
        )
        assert p[0] == pytest.approx(_sigmoid(margin), abs=1e-12)

    @pytest.mark.parametrize("n", [1, 2])
    def test_platt_fit_uses_leave_one_out_margins_and_smoothed_balanced_targets(self, n):
        vectors = OVERLAP_VECTORS
        labels = OVERLAP_LABELS
        params = fit_form(vectors, labels, FormSpec("knn", None, n))
        assert params["n"] == n
        loo = _hand_margins(vectors, labels, n, leave_one_out=True)
        assert np.max(np.abs(_platt_gradient(loo, labels, params["a"], params["c"]))) < 1e-8
        # With each row counted as its own neighbour the fit would be another one.
        self_included = _hand_margins(vectors, labels, n, leave_one_out=False)
        assert np.max(np.abs(_platt_gradient(self_included, labels, params["a"], params["c"]))) > (
            1e-6
        )

    def test_separable_bank_gives_finite_parameters(self):
        params = fit_form(SEPARABLE_VECTORS, SEPARABLE_LABELS, FormSpec("knn", None, 1))
        assert math.isfinite(params["a"]) and math.isfinite(params["c"])
        assert params["a"] > 0

    def test_platt_fit_weighs_each_label_equally_on_a_three_to_one_bank(self):
        vectors = [
            [1.0, 0.2, 0.1],
            [0.9, 0.1, 0.5],
            [0.3, 1.0, 0.0],
            [0.1, 0.9, 0.4],
            [0.5, 0.7, 0.2],
            [0.2, 0.3, 1.0],
            [0.6, 0.6, 0.6],
            [0.0, 1.0, 1.0],
        ]
        labels = ["need", "need"] + ["no_need"] * 6
        params = fit_form(vectors, labels, FormSpec("knn", None, 1))
        margins = _hand_margins(vectors, labels, 1, leave_one_out=True)
        balanced = _platt_gradient(margins, labels, params["a"], params["c"])
        assert np.max(np.abs(balanced)) < 1e-8
        # The unweighted objective's gradient is not zero there: the weights changed the fit.
        y = np.array([label == "need" for label in labels])
        targets = np.where(y, 3 / 4, 1 / 8)
        p = 1.0 / (1.0 + np.exp(-(params["a"] * margins + params["c"])))
        assert abs(float((p - targets).sum())) > 1e-3

    def test_repeated_fits_are_bit_identical(self):
        first = fit_form(OVERLAP_VECTORS, OVERLAP_LABELS, FormSpec("knn", None, 2))
        second = fit_form(OVERLAP_VECTORS, OVERLAP_LABELS, FormSpec("knn", None, 2))
        assert first == second

    def test_refuses_n_above_a_labels_other_rows(self):
        with pytest.raises(ValueError, match="n=3"):
            fit_form(OVERLAP_VECTORS, OVERLAP_LABELS, FormSpec("knn", None, 3))


class TestIterationCap:
    def test_capped_logistic_fit_names_form_parameter_and_gradient(self, monkeypatch):
        monkeypatch.setattr(need_gate, "NEWTON_MAX_ITERATIONS", 1)
        with pytest.raises(ValueError, match=r"logistic c=10 .*gradient") as caught:
            fit_form(OVERLAP_VECTORS, OVERLAP_LABELS, FormSpec("logistic", 10.0, None))
        assert isinstance(caught.value, NeedFitError)

    def test_capped_knn_fit_names_form_and_parameter(self, monkeypatch):
        monkeypatch.setattr(need_gate, "NEWTON_MAX_ITERATIONS", 1)
        with pytest.raises(NeedFitError, match=r"knn n=2 .*gradient"):
            fit_form(OVERLAP_VECTORS, OVERLAP_LABELS, FormSpec("knn", None, 2))

    def test_default_cap_is_one_hundred(self):
        assert need_gate.NEWTON_MAX_ITERATIONS == 100


class TestFitInputs:
    def test_refuses_a_zero_vector(self):
        with pytest.raises(ValueError, match="zero norm"):
            fit_form([[0.0, 0.0], [1.0, 0.0]], ["need", "no_need"], LOGISTIC_1)

    def test_refuses_one_label(self):
        with pytest.raises(ValueError, match="both labels"):
            fit_form([[1.0, 0.0], [0.0, 1.0]], ["need", "need"], LOGISTIC_1)

    def test_refuses_an_unknown_label(self):
        with pytest.raises(ValueError, match="label"):
            fit_form([[1.0, 0.0], [0.0, 1.0]], ["need", "maybe"], LOGISTIC_1)


# --- theta ---------------------------------------------------------------------------------------


class TestTheta:
    def test_midpoint_of_the_kth_and_the_next_lower_on_sorted_values(self):
        values = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
        # r = 0.8, N = 10: k = 8, the 8th largest is 0.2, the next lower is 0.1.
        assert theta(values, 0.8) == pytest.approx(0.15)
        # r = 0.95: k = ceil(9.5) = 10, the 10th is 0.05 and none is below: half of it.
        assert theta(values, 0.95) == pytest.approx(0.025)

    def test_order_of_the_input_does_not_matter(self):
        values = [0.1, 0.9, 0.3, 0.7, 0.5]
        # r = 0.6, N = 5: k = 3, sorted 0.9 0.7 0.5 0.3 0.1, the 3rd is 0.5, next lower 0.3.
        assert theta(values, 0.6) == pytest.approx(0.4)

    def test_ties_take_the_next_strictly_lower_value(self):
        values = [0.9, 0.6, 0.6, 0.6, 0.2]
        # r = 0.6: k = 3, the 3rd is 0.6; the largest value strictly below 0.6 is 0.2.
        assert theta(values, 0.6) == pytest.approx(0.4)
        assert sum(v >= theta(values, 0.6) for v in values) == 4

    def test_all_tied_takes_half(self):
        assert theta([0.5, 0.5, 0.5], 0.5) == pytest.approx(0.25)

    def test_decimal_share_is_read_as_written(self):
        # 0.88 * 25 is 22 as written; its binary float is a hair above 0.88, which must not
        # push k to 23.
        values = [1.0 - i / 100 for i in range(25)]
        assert theta(values, 0.88) == pytest.approx((values[21] + values[22]) / 2)

    @pytest.mark.parametrize("r", [0.5, 0.88, 0.95, 0.98, 1.0])
    def test_never_equal_to_an_input_and_keeps_at_least_the_share(self, r):
        values = [0.91, 0.42, 0.42, 0.77, 0.13, 0.99, 0.42, 0.65, 0.3, 0.3]
        result = theta(values, r)
        assert result not in values
        assert sum(v >= result for v in values) >= math.ceil(r * len(values) - 1e-9)

    def test_refuses_an_empty_set_or_share_out_of_range(self):
        with pytest.raises(ValueError):
            theta([], 0.9)
        with pytest.raises(ValueError):
            theta([0.5], 0.0)
        with pytest.raises(ValueError):
            theta([0.5], 1.5)


# --- cross-validation ----------------------------------------------------------------------------


def _cv_rows():
    """Writer a and b rows; the first coordinate is the probability the fake model gives."""
    rows = [
        # writer a
        ("a", "need", False, 0.90),
        ("a", "need", False, 0.80),
        ("a", "need", False, 0.40),
        ("a", "no_need", False, 0.10),
        ("a", "no_need", True, 0.50),
        ("a", "no_need", True, 0.30),
        # writer b
        ("b", "need", False, 0.95),
        ("b", "need", False, 0.70),
        ("b", "need", False, 0.60),
        ("b", "need", False, 0.20),
        ("b", "no_need", False, 0.05),
        ("b", "no_need", True, 0.65),
        ("b", "no_need", False, 0.15),
    ]
    vectors = [[p, 1.0, float(i)] for i, (_, _, _, p) in enumerate(rows)]
    labels = [label for _, label, _, _ in rows]
    writers = [writer for writer, _, _, _ in rows]
    hard = [h for _, _, h, _ in rows]
    return vectors, labels, writers, hard


class _FakeForms:
    """Stands in for fit_form and probabilities: p is each vector's first coordinate."""

    def __init__(self, failing: set[FormSpec] = frozenset()) -> None:
        self.failing = failing
        self.fits: list[tuple[FormSpec, tuple[float, ...]]] = []
        self.scored: list[tuple[tuple[float, ...], tuple[float, ...]]] = []

    def fit_form(self, vectors, labels, spec):
        rows = tuple(float(v[2]) for v in vectors)
        self.fits.append((spec, rows))
        if spec in self.failing:
            raise NeedFitError(f"{spec.form} reached the cap")
        return {"rows": rows}

    def probabilities(self, vectors, form, params, *, bank_vectors=None, bank_labels=None):
        scored = tuple(float(v[2]) for v in vectors)
        self.scored.append((params["rows"], scored))
        return np.array([float(v[0]) for v in vectors])


class TestGroupedCv:
    def _run(self, monkeypatch, specs, failing=frozenset()):
        fake = _FakeForms(set(failing))
        monkeypatch.setattr(need_gate, "fit_form", fake.fit_form)
        monkeypatch.setattr(need_gate, "probabilities", fake.probabilities)
        vectors, labels, writers, hard = _cv_rows()
        results = grouped_cv(vectors, labels, writers, hard, specs)
        return fake, results

    def test_each_fold_trains_on_one_writer_and_scores_the_other(self, monkeypatch):
        fake, _ = self._run(monkeypatch, (LOGISTIC_1,))
        a_rows = tuple(float(i) for i in range(6))
        b_rows = tuple(float(i) for i in range(6, 13))
        assert [rows for _, rows in fake.fits] == [a_rows, b_rows]
        assert fake.scored == [(a_rows, b_rows), (b_rows, a_rows)]

    def test_closure_and_hard_closure_by_hand(self, monkeypatch):
        _, (result,) = self._run(monkeypatch, (LOGISTIC_1,))
        # Fold 1 scores writer b. Need p: 0.95 0.7 0.6 0.2 (N = 4).
        #   R = 0.95: k = 4, the 4th is 0.2, none below: theta 0.1. No-need 0.05 0.65 0.15:
        #   closed 0.05 -> 1/3; hard 0.65 -> 0/1.
        #   R = 0.98: k = 4, the same: 1/3 and 0.
        # Fold 2 scores writer a. Need p: 0.9 0.8 0.4 (N = 3).
        #   R = 0.95 and 0.98: k = 3, the 3rd is 0.4, none below: theta 0.2. No-need 0.1 0.5
        #   0.3: closed 0.1 -> 1/3; hard 0.5 0.3 -> 0/2.
        assert result.closure == ((1 / 3, 1 / 3), (1 / 3, 1 / 3))
        assert result.hard_closure == ((0.0, 0.0), (0.0, 0.0))
        assert result.failed is False
        assert cv_score(result) == pytest.approx(1 / 3)

    def test_auroc_log_loss_and_brier_per_fold(self, monkeypatch):
        _, (result,) = self._run(monkeypatch, (LOGISTIC_1,))
        b_need, b_no = [0.95, 0.7, 0.6, 0.2], [0.05, 0.65, 0.15]
        a_need, a_no = [0.9, 0.8, 0.4], [0.1, 0.5, 0.3]

        def pairs(pos, neg):
            wins = sum(1.0 if p > q else 0.5 if p == q else 0.0 for p in pos for q in neg)
            return wins / (len(pos) * len(neg))

        def log_loss(pos, neg):
            terms = [-math.log(p) for p in pos] + [-math.log(1 - q) for q in neg]
            return sum(terms) / len(terms)

        def brier(pos, neg):
            terms = [(1 - p) ** 2 for p in pos] + [q**2 for q in neg]
            return sum(terms) / len(terms)

        assert result.auroc == pytest.approx((pairs(b_need, b_no), pairs(a_need, a_no)))
        assert result.log_loss == pytest.approx((log_loss(b_need, b_no), log_loss(a_need, a_no)))
        assert result.brier == pytest.approx((brier(b_need, b_no), brier(a_need, a_no)))

    def test_a_failing_candidate_is_marked_failed(self, monkeypatch):
        knn = FormSpec("knn", None, 1)
        _, results = self._run(monkeypatch, (LOGISTIC_1, knn), failing={knn})
        assert [r.spec for r in results] == [LOGISTIC_1, knn]
        assert results[0].failed is False
        assert results[1].failed is True
        with pytest.raises(ValueError):
            cv_score(results[1])

    def test_real_fit_matches_fitting_one_writer_and_scoring_the_other(self):
        vectors = SEPARABLE_VECTORS + OVERLAP_VECTORS
        labels = SEPARABLE_LABELS + OVERLAP_LABELS
        writers = ["a"] * 6 + ["b"] * 6
        hard = [False] * 4 + [True, True] + [False] * 4 + [True, True]
        spec = FormSpec("logistic", 1.0, None)
        (result,) = grouped_cv(vectors, labels, writers, hard, (spec,), targets=(0.95,))
        params = fit_form(vectors[:6], labels[:6], spec)
        p = probabilities(vectors[6:], "logistic", params)
        need_p = [float(v) for v, lab in zip(p, labels[6:]) if lab == "need"]
        no_p = [float(v) for v, lab in zip(p, labels[6:]) if lab == "no_need"]
        cut = theta(need_p, 0.95)
        assert result.closure[0] == (sum(v < cut for v in no_p) / len(no_p),)

    def test_capped_fits_mark_every_candidate_failed(self, monkeypatch):
        monkeypatch.setattr(need_gate, "NEWTON_MAX_ITERATIONS", 1)
        vectors = SEPARABLE_VECTORS + OVERLAP_VECTORS
        labels = SEPARABLE_LABELS + OVERLAP_LABELS
        writers = ["a"] * 6 + ["b"] * 6
        hard = [False] * 4 + [True, True] + [False] * 4 + [True, True]
        specs = (FormSpec("logistic", 100.0, None), FormSpec("knn", None, 1))
        results = grouped_cv(vectors, labels, writers, hard, specs)
        assert [r.failed for r in results] == [True, True]
        with pytest.raises(ValueError, match="every candidate"):
            choose_form(results, no_need_rows=6)

    def test_refuses_other_than_two_writers(self, monkeypatch):
        vectors, labels, _, hard = _cv_rows()
        with pytest.raises(ValueError, match="two writers"):
            grouped_cv(vectors, labels, ["a"] * len(labels), hard, (LOGISTIC_1,))


def _result(spec: FormSpec, score: float, *, failed: bool = False) -> CvResult:
    if failed:
        return CvResult(spec, (), (), (), (), (), failed=True)
    closure = ((score, score), (score, score))
    return CvResult(spec, closure, closure, (0.9, 0.9), (0.3, 0.3), (0.1, 0.1))


class TestChooseForm:
    # Best 0.9 over m = 100 no-need rows: one standard error is sqrt(0.9 * 0.1 / 100) = 0.03.

    def test_takes_the_best_when_no_other_is_within_one_standard_error(self):
        results = [
            _result(FormSpec("logistic", 0.1, None), 0.80),
            _result(FormSpec("knn", None, 3), 0.90),
        ]
        assert choose_form(results, no_need_rows=100) == FormSpec("knn", None, 3)

    def test_prefers_logistic_within_one_standard_error(self):
        results = [
            _result(FormSpec("knn", None, 1), 0.90),
            _result(FormSpec("logistic", 100.0, None), 0.88),
        ]
        assert choose_form(results, no_need_rows=100) == FormSpec("logistic", 100.0, None)

    def test_then_the_smaller_c(self):
        results = [
            _result(FormSpec("logistic", 10.0, None), 0.90),
            _result(FormSpec("logistic", 1.0, None), 0.875),
            _result(FormSpec("logistic", 0.1, None), 0.86),
        ]
        assert choose_form(results, no_need_rows=100) == FormSpec("logistic", 1.0, None)

    def test_then_the_larger_n(self):
        results = [
            _result(FormSpec("knn", None, 1), 0.90),
            _result(FormSpec("knn", None, 3), 0.88),
            _result(FormSpec("knn", None, 5), 0.875),
        ]
        assert choose_form(results, no_need_rows=100) == FormSpec("knn", None, 5)

    def test_ties_follow_the_preference(self):
        results = [
            _result(FormSpec("knn", None, 5), 1.0),
            _result(FormSpec("logistic", 10.0, None), 1.0),
            _result(FormSpec("logistic", 1.0, None), 1.0),
        ]
        # c = 1.0 makes the standard error zero: only the tied best are eligible.
        assert choose_form(results, no_need_rows=100) == FormSpec("logistic", 1.0, None)

    def test_skips_failed_candidates(self):
        results = [
            _result(FormSpec("logistic", 0.1, None), 0.0, failed=True),
            _result(FormSpec("knn", None, 1), 0.90),
        ]
        assert choose_form(results, no_need_rows=100) == FormSpec("knn", None, 1)

    def test_raises_when_every_candidate_failed(self):
        results = [_result(spec, 0.0, failed=True) for spec in FORM_CANDIDATES]
        with pytest.raises(ValueError, match="every candidate"):
            choose_form(results, no_need_rows=100)

    def test_refuses_a_non_positive_row_count(self):
        with pytest.raises(ValueError):
            choose_form([_result(LOGISTIC_1, 0.9)], no_need_rows=0)


class TestAuroc:
    def test_hand_computed(self):
        assert auroc([0.1, 0.4, 0.35, 0.8], ["no_need", "no_need", "need", "need"]) == 0.75

    def test_ties_count_half(self):
        assert auroc([0.5, 0.5], ["need", "no_need"]) == 0.5
        # pairs: 0.3>0.1 (1), 0.3<0.5 (0), 0.5=0.5 (0.5), 0.5>0.1 (1): 2.5 / 4
        assert auroc([0.3, 0.5, 0.5, 0.1], ["need", "need", "no_need", "no_need"]) == 0.625

    def test_accepts_booleans(self):
        assert auroc([0.9, 0.1], [True, False]) == 1.0

    def test_refuses_one_label(self):
        with pytest.raises(ValueError):
            auroc([0.1, 0.2], ["need", "need"])


# --- the bank ------------------------------------------------------------------------------------


class TestBankSha256:
    def test_is_the_sha256_of_each_files_digest_in_order(self):
        first, second = b"one\n", b"two\n"
        expected = hashlib.sha256(
            hashlib.sha256(first).digest() + hashlib.sha256(second).digest()
        ).hexdigest()
        assert bank_sha256([first, second]) == expected
        assert bank_sha256([second, first]) != expected


class TestLoadBank:
    def test_reads_both_files_and_hashes_the_bytes_it_parsed(self, tmp_path):
        a, b = _two_writer_files(tmp_path)
        bank = load_bank([a, b])
        assert [e.id for e in bank.exemplars] == [
            "a-need-001",
            "a-no-need-001",
            "b-need-001",
            "b-no-need-001",
        ]
        assert bank.exemplars[3] == NeedExemplar(
            "b-no-need-001", "b", "no_need", "cat", True, "what is the capital of peru"
        )
        assert bank.sha256 == bank_sha256([a.read_bytes(), b.read_bytes()])

    @pytest.mark.parametrize("which", [0, 1])
    def test_hash_changes_when_one_byte_of_either_file_does(self, tmp_path, which):
        paths = _two_writer_files(tmp_path)
        before = load_bank(paths).sha256
        data = paths[which].read_bytes()
        paths[which].write_bytes(data.replace(b"my", b"My", 1))
        assert load_bank(paths).sha256 != before

    def _refusal(self, tmp_path, b_rows, match, *, raw: str | None = None):
        a, b = _two_writer_files(tmp_path)
        if raw is not None:
            b.write_text(raw, encoding="utf-8")
        else:
            _write_jsonl(b, b_rows)
        with pytest.raises(MemoryConfigError, match=match):
            load_bank([a, b])

    def test_refuses_a_malformed_row_naming_file_and_line(self, tmp_path):
        good = json.dumps(_row("b-need-001", "b", "need", "play my usual"))
        self._refusal(tmp_path, None, r"writer_b\.jsonl:2", raw=good + "\n{not json\n")

    def test_refuses_a_row_missing_a_field(self, tmp_path):
        row = _row("b-need-001", "b", "need", "play my usual")
        del row["hard"]
        self._refusal(tmp_path, [row], r"writer_b\.jsonl:1")

    def test_refuses_an_unknown_label(self, tmp_path):
        rows = [_row("b-need-001", "b", "maybe", "play my usual")]
        self._refusal(tmp_path, rows, r"writer_b\.jsonl:1: .*label")

    def test_refuses_a_duplicate_id_across_files(self, tmp_path):
        rows = [_row("a-need-001", "b", "need", "play my usual")]
        self._refusal(tmp_path, rows, r"writer_b\.jsonl:1: .*duplicate id")

    def test_refuses_an_empty_text(self, tmp_path):
        rows = [_row("b-need-001", "b", "need", "   ")]
        self._refusal(tmp_path, rows, r"writer_b\.jsonl:1: .*empty text")

    def test_refuses_a_row_of_another_writer(self, tmp_path):
        rows = [_row("b-need-001", "a", "need", "play my usual")]
        self._refusal(tmp_path, rows, r"writer_b\.jsonl:1: .*writer")

    def test_refuses_a_bank_of_one_label(self, tmp_path):
        a = _write_jsonl(tmp_path / "writer_a.jsonl", [_row("a-need-001", "a", "need", "x")])
        b = _write_jsonl(tmp_path / "writer_b.jsonl", [_row("b-need-001", "b", "need", "y")])
        with pytest.raises(MemoryConfigError, match="one label"):
            load_bank([a, b])

    def test_refuses_a_missing_file(self, tmp_path):
        a, _ = _two_writer_files(tmp_path)
        with pytest.raises(MemoryConfigError, match="writer_c"):
            load_bank([a, tmp_path / "writer_c.jsonl"])

    def test_refuses_a_file_not_named_for_its_writer(self, tmp_path):
        a, b = _two_writer_files(tmp_path)
        other = tmp_path / "bank.jsonl"
        other.write_bytes(b.read_bytes())
        with pytest.raises(MemoryConfigError, match="bank.jsonl"):
            load_bank([a, other])


# --- model records -------------------------------------------------------------------------------


class TestNeedModelKey:
    def test_is_the_sha256_of_the_canonical_json(self):
        key = NeedKey("ab", "bge-small-en-v1.5", "rev", "q: ")
        canonical = (
            '{"bank_sha256":"ab","embedder_name":"bge-small-en-v1.5",'
            '"embedder_revision":"rev","normalisation":"l2","query_prefix":"q: "}'
        )
        assert need_model_key(key) == hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def test_changes_with_each_field(self):
        base = NeedKey("ab", "name", "rev", "q: ", "l2")
        variants = [
            NeedKey("cd", "name", "rev", "q: ", "l2"),
            NeedKey("ab", "other", "rev", "q: ", "l2"),
            NeedKey("ab", "name", "rev2", "q: ", "l2"),
            NeedKey("ab", "name", "rev", "", "l2"),
            NeedKey("ab", "name", "rev", "q: ", "none"),
        ]
        keys = {need_model_key(base)} | {need_model_key(v) for v in variants}
        assert len(keys) == 6

    def test_model_parts_are_its_five_key_fields(self):
        model = _model(GATE_KEY)
        assert model.parts() == GATE_KEY


def _record(need_key: NeedKey, **overrides) -> dict:
    record = {
        "key": need_model_key(need_key),
        "bank_sha256": need_key.bank_sha256,
        "embedder": {"name": need_key.embedder_name, "revision": need_key.embedder_revision},
        "query_prefix": need_key.query_prefix,
        "normalisation": need_key.normalisation,
        "form": "logistic",
        "params": {"c": 1.0, "w": [0.5, -0.5, 0.0], "b": 0.1},
        "theta_low": 0.3,
        "theta_high": 0.6,
        "band_k": 2,
        "recall_target": 0.95,
        "source": "H18 2026-11-15",
    }
    record.update(overrides)
    return record


def _write_models(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "need_model.json"
    path.write_text(json.dumps({"records": records}), encoding="utf-8")
    return path


class TestLoadModels:
    def test_reads_each_record(self, tmp_path):
        knn_key = NeedKey("aa", "shared", "rev", "")
        path = _write_models(
            tmp_path,
            [
                _record(GATE_KEY),
                _record(knn_key, form="knn", params={"n": 3, "a": 5.0, "c": -0.5}),
            ],
        )
        logistic, knn = load_models(path)
        assert logistic.parts() == GATE_KEY
        assert logistic.key == need_model_key(GATE_KEY)
        assert (logistic.form, logistic.theta_low, logistic.theta_high) == ("logistic", 0.3, 0.6)
        assert (logistic.band_k, logistic.recall_target) == (2, 0.95)
        assert list(logistic.params["w"]) == [0.5, -0.5, 0.0]
        assert logistic.source == "H18 2026-11-15"
        assert knn.form == "knn"
        assert dict(knn.params) == {"n": 3, "a": 5.0, "c": -0.5}

    def test_refuses_a_missing_file(self, tmp_path):
        with pytest.raises(MemoryConfigError, match="need_model.json"):
            load_models(tmp_path / "need_model.json")

    def test_refuses_a_malformed_file(self, tmp_path):
        path = tmp_path / "need_model.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(MemoryConfigError):
            load_models(path)

    @pytest.mark.parametrize(
        "overrides,match",
        [
            ({"theta_low": float("nan")}, "finite"),
            ({"params": {"c": 1.0, "w": [0.5, float("inf"), 0.0], "b": 0.1}}, "finite"),
            ({"theta_low": 0.7, "theta_high": 0.6}, "theta_low"),
            ({"band_k": 3}, "band_k"),
            ({"band_k": True}, "band_k"),
            ({"form": "centroid"}, "form"),
            ({"params": {"c": 1.0, "b": 0.1}}, "params"),
            ({"key": "0" * 64}, "key"),
        ],
    )
    def test_refuses_a_bad_record(self, tmp_path, overrides, match):
        path = _write_models(tmp_path, [_record(GATE_KEY, **overrides)])
        with pytest.raises(MemoryConfigError, match=match):
            load_models(path)

    def test_refuses_two_records_with_one_key(self, tmp_path):
        path = _write_models(tmp_path, [_record(GATE_KEY), _record(GATE_KEY)])
        with pytest.raises(MemoryConfigError, match="two records"):
            load_models(path)


class TestSelectModel:
    def test_returns_the_matching_record(self):
        other = _model(NeedKey("aa", "shared", "rev", ""))
        wanted = _model(GATE_KEY)
        assert select_model((other, wanted), GATE_KEY) is wanted

    def test_names_each_differing_field_of_the_closest_record(self):
        far = _model(NeedKey("00", "shared", "x", ""))
        near = _model(GATE_KEY)
        key = NeedKey(GATE_KEY.bank_sha256, GATE_KEY.embedder_name, "rev2", "other: ")
        with pytest.raises(MemoryConfigError) as caught:
            select_model((far, near), key)
        message = str(caught.value)
        assert "embedder_revision" in message and "query_prefix" in message
        assert "bank_sha256" not in message and "embedder_name" not in message

    def test_refuses_an_empty_file(self):
        with pytest.raises(MemoryConfigError, match="no record"):
            select_model((), GATE_KEY)


# --- the gate ------------------------------------------------------------------------------------


def _query(p: float) -> list[float]:
    """A unit vector whose logit under w = (4, -4, 0) is logit(p): x0 - x1 = logit(p) / 4."""
    d = math.log(p / (1 - p)) / 4
    # x0 - x1 = d with x0^2 + x1^2 = 1.
    x0 = (d + math.sqrt(2 - d * d)) / 2
    return [x0, x0 - d, 0.0]


class TestNeedGateConstruction:
    def test_embeds_the_bank_once_with_the_prefix(self):
        embedder = _gate_embedder()
        _gate(embedder=embedder)
        assert embedder.batch_calls == [[f"{PREFIX}text {id_}" for id_, _, _ in GATE_BANK]]
        assert embedder.single_calls == []

    def test_embed_query_makes_one_call_with_the_prefix(self):
        embedder = _gate_embedder()
        embedder.vectors[f"{PREFIX}hello"] = [1.0, 0.0, 0.0]
        gate = _gate(embedder=embedder)
        assert gate.embed_query("hello") == [1.0, 0.0, 0.0]
        assert embedder.single_calls == [f"{PREFIX}hello"]
        assert len(embedder.batch_calls) == 1

    def test_read_only_attributes(self):
        embedder = _gate_embedder()
        model = _model(GATE_KEY)
        gate = _gate(embedder=embedder, model=model)
        assert gate.embedder is embedder
        assert gate.model is model
        assert gate.query_prefix == PREFIX
        assert gate.shares_query_vector is False
        with pytest.raises(AttributeError):
            gate.query_prefix = ""

    @pytest.mark.parametrize(
        "field,overrides",
        [
            ("embedder_name", {"embedder_name": "shared"}),
            ("embedder_revision", {"embedder_revision": "rev2"}),
            ("query_prefix", {"query_prefix": "other: "}),
        ],
    )
    def test_refuses_a_model_keyed_to_another_embedder_or_prefix(self, field, overrides):
        embedder = FakeEmbedder({})
        with pytest.raises(MemoryConfigError, match=field):
            _gate(embedder=embedder, **overrides)
        assert embedder.batch_calls == []

    def test_refuses_a_model_keyed_to_another_bank_naming_each_field(self):
        model = _model(NeedKey("0" * 64, "bge-small-en-v1.5", "rev0", PREFIX))
        with pytest.raises(MemoryConfigError) as caught:
            _gate(model=model)
        message = str(caught.value)
        assert "bank_sha256" in message and "embedder_revision" in message
        assert "embedder_name" not in message and "query_prefix" not in message

    def test_refuses_a_shared_query_vector_with_a_prefix(self):
        with pytest.raises(MemoryConfigError, match="shares_query_vector"):
            _gate(shares_query_vector=True)

    def test_shares_the_query_vector_without_a_prefix(self):
        key = NeedKey("f" * 64, "shared", "rev1", "")
        gate = NeedGate(
            _gate_embedder(""),
            _gate_bank(),
            _model(key),
            embedder_name="shared",
            embedder_revision="rev1",
            query_prefix="",
            shares_query_vector=True,
        )
        assert gate.shares_query_vector is True

    def test_refuses_a_weight_vector_of_another_length(self):
        model = _model(GATE_KEY, params={"c": 1.0, "w": [1.0, 2.0], "b": 0.0})
        with pytest.raises(MemoryConfigError, match="length"):
            _gate(model=model)

    def test_refuses_n_above_a_labels_row_count(self):
        model = _model(GATE_KEY, form="knn", params={"n": 5, "a": 1.0, "c": 0.0})
        with pytest.raises(MemoryConfigError, match="n=5"):
            _gate(model=model)


class TestNeedGateDecide:
    def test_closed_below_theta_low(self):
        decision = _gate().decide(_query(0.2))
        assert decision.band == "closed"
        assert decision.p == pytest.approx(0.2, abs=1e-12)

    def test_band_from_theta_low_up_to_theta_high(self):
        assert _gate().decide(_query(0.5)).band == "band"

    def test_open_at_theta_high_and_above(self):
        assert _gate().decide(_query(0.9)).band == "open"

    def test_boundaries(self):
        vector = _query(0.45)
        p = _gate().decide(vector).p
        at_low = _gate(model=_model(GATE_KEY, theta_low=p, theta_high=0.9))
        assert at_low.decide(vector).band == "band"
        at_high = _gate(model=_model(GATE_KEY, theta_low=0.1, theta_high=p))
        assert at_high.decide(vector).band == "open"
        above_low = _gate(model=_model(GATE_KEY, theta_low=math.nextafter(p, 1.0), theta_high=0.9))
        assert above_low.decide(vector).band == "closed"

    def test_carries_the_models_thresholds_and_band_k(self):
        decision = _gate().decide(_query(0.5))
        assert (decision.theta_low, decision.theta_high, decision.band_k) == (0.3, 0.7, 2)

    def test_nearest_ids_three_of_each_nearest_first(self):
        decision = _gate().decide([1.0, 0.0, 0.0])
        assert decision.need_ids == ("n2", "n4", "n1")
        assert decision.no_need_ids == ("o2", "o4", "o1")

    def test_nearest_ids_under_knn(self):
        model = _model(GATE_KEY, form="knn", params={"n": 2, "a": 3.0, "c": 0.0})
        decision = _gate(model=model).decide([0.0, 1.0, 0.0])
        assert decision.need_ids == ("n3", "n1", "n4")
        assert decision.no_need_ids == ("o3", "o1", "o4")

    def test_knn_probability_is_the_sigmoid_of_the_margin(self):
        model = _model(GATE_KEY, form="knn", params={"n": 2, "a": 3.0, "c": -0.5})
        gate = _gate(model=model)
        bank = _unit([vec for _, _, vec in GATE_BANK])
        q = np.array([1.0, 0.0, 0.0])
        need_cos = sorted((bank[:4] @ q).tolist(), reverse=True)[:2]
        no_cos = sorted((bank[4:] @ q).tolist(), reverse=True)[:2]
        margin = sum(need_cos) / 2 - sum(no_cos) / 2
        assert gate.decide([2.0, 0.0, 0.0]).p == pytest.approx(_sigmoid(3.0 * margin - 0.5))

    def test_scales_the_vector_to_unit_length(self):
        gate = _gate()
        assert gate.decide([2.0, 0.0, 0.0]).p == gate.decide([1.0, 0.0, 0.0]).p

    def test_makes_no_model_call(self):
        embedder = _gate_embedder()
        gate = _gate(embedder=embedder)
        gate.decide(_query(0.5))
        assert len(embedder.batch_calls) == 1
        assert embedder.single_calls == []

    @pytest.mark.parametrize(
        "vector,match",
        [
            ([1.0, 0.0], "length"),
            ([1.0, float("nan"), 0.0], "finite"),
            ([0.0, 0.0, 0.0], "zero norm"),
        ],
    )
    def test_refuses_a_bad_vector(self, vector, match):
        with pytest.raises(ValueError, match=match):
            _gate().decide(vector)
