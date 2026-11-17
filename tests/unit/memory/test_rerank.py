"""Tests for backend.memory.rerank: the Reranker protocol, CrossEncoderReranker and NullReranker.

Every test injects FakeCrossEncoderClass in place of `sentence_transformers.CrossEncoder`, so no
weights are loaded and no network is touched. The cost calibration runs against FakeClock: the
fake model advances the clock by a scripted cost per call, so the measured costs are exact.

The release and reload tests hold a load or a forward pass open on a `threading.Event` and wait
for each step with a bounded `Event.wait` or `Thread.join`; none sleeps to let a thread run.
"""

from __future__ import annotations

import gc
import inspect
import math
import sys
import threading
import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType

import pytest

from backend.errors import MemoryRerankError
from backend.memory.config import MEMORY_ENV_VARS, MemoryConfig
from backend.memory.rerank import CrossEncoderReranker, NullReranker, Reranker
from tests.fakes.memory_fakes import FakeClock

# Upper bound on every wait for another thread. A healthy run never comes near it; it only turns
# a deadlock into a failure instead of a hung suite.
_WAIT_S = 5.0

# Predict calls one calibration makes: the untimed first call, then one pair and the whole batch
# in each of three rounds.
_WARM_UP_CALLS = 7


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class FakeCrossEncoderModel:
    """A loaded fake cross-encoder that records each `predict` call and charges the clock."""

    owner: FakeCrossEncoderClass
    model_name: str
    device: str
    local_files_only: bool
    predict_calls: list[dict[str, object]] = field(default_factory=list)

    def predict(
        self,
        sentences: list[tuple[str, str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
        activation_fct: Callable[[float], float] | None = None,
        convert_to_numpy: bool,
    ) -> list[float]:
        owner = self.owner
        call_index = len(self.predict_calls)
        if owner.on_predict is not None:
            owner.on_predict(call_index)
        self.predict_calls.append(
            {
                "sentences": list(sentences),
                "batch_size": batch_size,
                "show_progress_bar": show_progress_bar,
                "activation_fct": activation_fct,
                "convert_to_numpy": convert_to_numpy,
            }
        )
        if owner.clock is not None:
            extra = owner.extra_ms_by_call.get(call_index, 0.0)
            owner.clock.advance(ms=owner.overhead_ms + owner.ms_per_pair * len(sentences) + extra)
        if owner.predict_error is not None:
            raise owner.predict_error
        logits = [owner.logits.get(text, 0.0) for _, text in sentences]
        if owner.drop_one_score and logits:
            logits = logits[:-1]
        # The real class applies its checkpoint's default activation unless the caller passes
        # one, so the fake does too: a reranker that relied on the default would see it here.
        activation = activation_fct if activation_fct is not None else owner.default_activation
        scores = [float(activation(x)) for x in logits]
        if owner.labels > 1:
            # A multi-label head gives one row per pair, as the real class's 2-d array does.
            return [[score] * owner.labels for score in scores]  # type: ignore[misc]
        return scores


@dataclass
class FakeCrossEncoderClass:
    """Stands in for the `sentence_transformers.CrossEncoder` class: calling it loads a model.

    The constructor keywords the reranker passes are recorded on the model it returns, so a test
    can assert the device and `local_files_only` it received. `on_load` runs at the start of each
    load and `on_predict` at the start of each `predict` with that model's call index, so a thread
    test can hold either open on an event.
    """

    clock: FakeClock | None = None
    overhead_ms: float = 0.0
    ms_per_pair: float = 0.0
    extra_ms_by_call: dict[int, float] = field(default_factory=dict)
    logits: dict[str, float] = field(default_factory=dict)
    default_activation: Callable[[float], float] = lambda x: x  # noqa: E731
    load_error: Exception | None = None
    predict_error: Exception | None = None
    drop_one_score: bool = False
    labels: int = 1
    on_load: Callable[[], object] | None = None
    on_predict: Callable[[int], object] | None = None
    built: list[FakeCrossEncoderModel] = field(default_factory=list)

    def __call__(
        self, model_name: str, *, device: str, local_files_only: bool
    ) -> FakeCrossEncoderModel:
        if self.on_load is not None:
            self.on_load()
        if self.load_error is not None:
            raise self.load_error
        model = FakeCrossEncoderModel(self, model_name, device, local_files_only)
        self.built.append(model)
        return model

    @property
    def model(self) -> FakeCrossEncoderModel:
        assert len(self.built) == 1, f"expected one load, got {len(self.built)}"
        return self.built[0]


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    """An existing directory standing in for a baked cross-encoder."""
    path = tmp_path / "ms-marco-MiniLM-L-6-v2"
    path.mkdir()
    return path


def _build(
    model_dir: Path,
    fake_class: FakeCrossEncoderClass,
    *,
    device: str = "cpu",
    clock: FakeClock | None = None,
    batch_size: int = 32,
    load: bool = True,
) -> CrossEncoderReranker:
    return CrossEncoderReranker(
        str(model_dir),
        device=device,  # type: ignore[arg-type]
        cross_encoder_cls=fake_class,
        clock=(clock or FakeClock()).monotonic,
        batch_size=batch_size,
        load=load,
    )


def _wait(event: threading.Event) -> None:
    assert event.wait(_WAIT_S), "the other thread never reached the expected step"


class _Run(threading.Thread):
    """Runs `fn` on its own thread and keeps what it returned or the error it raised."""

    def __init__(self, fn: Callable[[], object]) -> None:
        super().__init__(daemon=True)
        self._fn = fn
        self.result: object = None
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            self.result = self._fn()
        except (MemoryRerankError, AssertionError, threading.BrokenBarrierError) as exc:
            self.error = exc

    def finish(self) -> _Run:
        """Join within `_WAIT_S` and fail if the thread is still running."""
        self.join(_WAIT_S)
        assert not self.is_alive(), "the thread did not finish: it is waiting on something"
        return self


class _WatchedLock:
    """A lock that sets `contended` when a thread finds it held and has to wait for it.

    Put in place of the reranker's lock, it lets a test know a second thread is blocked on the
    lock rather than not yet started, without sleeping.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.contended = threading.Event()

    def __enter__(self) -> _WatchedLock:
        if not self._lock.acquire(blocking=False):
            self.contended.set()
            self._lock.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._lock.release()


@pytest.fixture
def freed(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Records `gc.collect` and `torch.cuda.empty_cache` calls, in order, in place of both."""
    import torch

    calls: list[str] = []

    def collect(*args: object, **kwargs: object) -> int:
        calls.append("gc.collect")
        return 0

    monkeypatch.setattr(gc, "collect", collect)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty_cache"))
    return calls


# The real collector, kept before `freed` replaces `gc.collect`.
_REAL_COLLECT = gc.collect


def _assert_unreferenced(fake_class: FakeCrossEncoderClass, index: int) -> None:
    """The model the fake built at `index` has no reference left once the fake drops its own."""
    ref = weakref.ref(fake_class.built[index])
    fake_class.built.clear()
    _REAL_COLLECT()
    assert ref() is None, "the reranker still holds the model it was meant to drop"


# -- Protocol ---------------------------------------------------------------------------------


def _needs_reranker(reranker: Reranker) -> Reranker:
    return reranker


def test_score_signature_matches_the_protocol_including_optional_budget() -> None:
    params = inspect.signature(Reranker.score).parameters

    assert list(params) == ["self", "query", "texts", "budget_ms"]
    assert params["budget_ms"].default is None


@pytest.mark.parametrize("cls", [CrossEncoderReranker, NullReranker])
def test_implementations_share_the_protocol_signatures(cls: type) -> None:
    for name in ("score", "pairs_within", "release", "reload"):
        expected = inspect.signature(getattr(Reranker, name))
        actual = inspect.signature(getattr(cls, name))

        assert list(actual.parameters) == list(expected.parameters), (cls.__name__, name)
        assert [p.default for p in actual.parameters.values()] == [
            p.default for p in expected.parameters.values()
        ], (cls.__name__, name)


# -- Loading and device placement -------------------------------------------------------------


@pytest.mark.parametrize("device", [pytest.param("cpu", id="cpu"), pytest.param("cuda", id="cuda")])
def test_passes_the_device_explicitly_to_the_cross_encoder(model_dir: Path, device: str) -> None:
    fake_class = FakeCrossEncoderClass()

    _build(model_dir, fake_class, device=device)

    assert fake_class.model.device == device


def test_device_comes_from_mist_memory_rerank_device(
    model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in MEMORY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MIST_MEMORY_RERANK_DEVICE", "cuda")
    monkeypatch.setenv("MIST_MEMORY_RERANK_MODEL", str(model_dir))
    config = MemoryConfig.from_env(tz_resolver=lambda name: pytest.fail(f"resolved {name}"))
    fake_class = FakeCrossEncoderClass()

    CrossEncoderReranker(
        config.rerank_model,
        device=config.rerank_device,
        cross_encoder_cls=fake_class,
        clock=FakeClock().monotonic,
    )

    assert fake_class.model.device == "cuda"
    assert fake_class.model.model_name == str(model_dir)


def test_loads_from_local_files_only(model_dir: Path) -> None:
    fake_class = FakeCrossEncoderClass()

    _build(model_dir, fake_class)

    assert fake_class.model.local_files_only is True
    assert fake_class.model.model_name == str(model_dir)


def test_missing_model_path_raises_naming_the_path(tmp_path: Path) -> None:
    missing = tmp_path / "not-baked"
    fake_class = FakeCrossEncoderClass()

    with pytest.raises(MemoryRerankError, match="not-baked"):
        _build(missing, fake_class)

    assert fake_class.built == []


def test_model_path_that_is_a_file_raises(tmp_path: Path) -> None:
    weights_file = tmp_path / "model.safetensors"
    weights_file.write_bytes(b"")

    with pytest.raises(MemoryRerankError, match="model.safetensors"):
        _build(weights_file, FakeCrossEncoderClass())


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(b'{"version": "1.0", "truncation": nu', id="truncated"),
        pytest.param(b'{"normalizer": "\xff\xfe"}', id="not-utf8"),
    ],
)
def test_unreadable_tokenizer_json_raises_naming_path_and_device(
    model_dir: Path, content: bytes
) -> None:
    (model_dir / "tokenizer.json").write_bytes(content)
    fake_class = FakeCrossEncoderClass()

    with pytest.raises(MemoryRerankError, match="tokenizer.json") as excinfo:
        _build(model_dir, fake_class, device="cuda")

    assert str(model_dir) in str(excinfo.value)
    assert "cuda" in str(excinfo.value)
    assert fake_class.built == []


def test_tokenizer_json_that_parses_loads(model_dir: Path) -> None:
    (model_dir / "tokenizer.json").write_text('{"version": "1.0"}', encoding="utf-8")
    fake_class = FakeCrossEncoderClass()

    _build(model_dir, fake_class)

    assert fake_class.model.model_name == str(model_dir)


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(OSError("config.json not found"), id="os-error"),
        pytest.param(ValueError("unrecognised configuration"), id="value-error"),
        pytest.param(RuntimeError("CUDA out of memory"), id="runtime-error"),
    ],
)
def test_load_failure_raises_memory_rerank_error_naming_path_and_device(
    model_dir: Path, error: Exception
) -> None:
    fake_class = FakeCrossEncoderClass(load_error=error)

    with pytest.raises(MemoryRerankError) as excinfo:
        _build(model_dir, fake_class, device="cuda")

    assert str(model_dir) in str(excinfo.value)
    assert "cuda" in str(excinfo.value)
    assert excinfo.value.__cause__ is error


def test_unknown_device_is_refused_before_loading(model_dir: Path) -> None:
    fake_class = FakeCrossEncoderClass()

    with pytest.raises(MemoryRerankError, match="mps"):
        _build(model_dir, fake_class, device="mps")

    assert fake_class.built == []


def test_real_cross_encoder_accepts_the_constructor_and_predict_keywords(model_dir: Path) -> None:
    from sentence_transformers import CrossEncoder

    init = inspect.signature(CrossEncoder)
    predict = inspect.signature(CrossEncoder.predict)

    init.bind(str(model_dir), device="cpu", local_files_only=True)
    predict.bind(
        object(),
        [("q", "t")],
        batch_size=32,
        show_progress_bar=False,
        activation_fct=lambda x: x,
        convert_to_numpy=True,
    )


# -- Warm-up calibration and pairs_within -----------------------------------------------------


def test_warm_up_failure_raises_memory_rerank_error_naming_the_device(model_dir: Path) -> None:
    fake_class = FakeCrossEncoderClass(predict_error=RuntimeError("Found no NVIDIA driver"))

    with pytest.raises(MemoryRerankError, match="cuda"):
        _build(model_dir, fake_class, device="cuda")


def test_warm_up_records_cost_per_pair_and_call_overhead(model_dir: Path) -> None:
    clock = FakeClock()
    fake_class = FakeCrossEncoderClass(clock=clock, overhead_ms=4.0, ms_per_pair=2.0)

    reranker = _build(model_dir, fake_class, clock=clock)

    assert reranker.cost_per_pair_ms == pytest.approx(2.0)
    assert reranker.call_overhead_ms == pytest.approx(4.0)


def test_warm_up_excludes_the_first_call_from_the_measurement(model_dir: Path) -> None:
    clock = FakeClock()
    fake_class = FakeCrossEncoderClass(
        clock=clock, overhead_ms=4.0, ms_per_pair=2.0, extra_ms_by_call={0: 500.0}
    )

    reranker = _build(model_dir, fake_class, clock=clock)

    assert reranker.cost_per_pair_ms == pytest.approx(2.0)
    assert reranker.call_overhead_ms == pytest.approx(4.0)


@pytest.mark.parametrize("spike_call", [pytest.param(i, id=f"call-{i}") for i in range(1, 7)])
def test_warm_up_ignores_one_slow_timed_call(model_dir: Path, spike_call: int) -> None:
    clock = FakeClock()
    fake_class = FakeCrossEncoderClass(
        clock=clock, overhead_ms=4.0, ms_per_pair=2.0, extra_ms_by_call={spike_call: 80.0}
    )

    reranker = _build(model_dir, fake_class, clock=clock)

    assert reranker.cost_per_pair_ms == pytest.approx(2.0)
    assert reranker.call_overhead_ms == pytest.approx(4.0)


def test_warm_up_scores_a_fixed_batch(model_dir: Path) -> None:
    first, second = FakeCrossEncoderClass(), FakeCrossEncoderClass()

    _build(model_dir, first)
    _build(model_dir, second)

    first_batches = [call["sentences"] for call in first.model.predict_calls]
    second_batches = [call["sentences"] for call in second.model.predict_calls]
    assert first_batches == second_batches
    assert max(len(batch) for batch in first_batches) > 1  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("budget_ms", "expected"),
    [
        pytest.param(50.0, 23, id="fits-23"),
        pytest.param(6.0, 1, id="exactly-one"),
        pytest.param(5.9, 0, id="below-one-pair"),
        pytest.param(4.0, 0, id="overhead-only"),
        pytest.param(0.0, 0, id="zero"),
        pytest.param(-10.0, 0, id="negative"),
        pytest.param(math.nan, 0, id="nan"),
    ],
)
def test_pairs_within_counts_pairs_that_fit_the_budget(
    model_dir: Path, budget_ms: float, expected: int
) -> None:
    clock = FakeClock()
    fake_class = FakeCrossEncoderClass(clock=clock, overhead_ms=4.0, ms_per_pair=2.0)
    reranker = _build(model_dir, fake_class, clock=clock)

    assert reranker.pairs_within(budget_ms) == expected


def test_pairs_within_predicts_the_cost_of_the_cut_batch(model_dir: Path) -> None:
    clock = FakeClock()
    fake_class = FakeCrossEncoderClass(clock=clock, overhead_ms=4.0, ms_per_pair=2.0)
    reranker = _build(model_dir, fake_class, clock=clock)
    budget_ms = 31.0
    count = reranker.pairs_within(budget_ms)

    before = clock.monotonic()
    reranker.score("query", [f"text {i}" for i in range(count)])
    spent_ms = (clock.monotonic() - before) * 1000.0

    assert spent_ms <= budget_ms
    assert spent_ms + 2.0 > budget_ms  # one more pair would not have fit


def test_pairs_within_stays_finite_when_the_measured_cost_is_zero(model_dir: Path) -> None:
    clock = FakeClock()
    reranker = _build(model_dir, FakeCrossEncoderClass(clock=clock), clock=clock)

    assert reranker.pairs_within(50.0) > 0
    assert reranker.pairs_within(math.inf) > 0


# -- Scoring ----------------------------------------------------------------------------------


def test_scores_are_sigmoid_of_the_logits_in_input_order(model_dir: Path) -> None:
    fake_class = FakeCrossEncoderClass(logits={"a": -2.0, "b": 0.0, "c": 3.0})
    reranker = _build(model_dir, fake_class)

    scores = reranker.score("query", ["c", "a", "b"])

    assert scores == pytest.approx([_sigmoid(3.0), _sigmoid(-2.0), 0.5])


def test_sigmoid_is_applied_once_even_when_the_checkpoint_defaults_to_sigmoid(
    model_dir: Path,
) -> None:
    fake_class = FakeCrossEncoderClass(logits={"a": 2.0}, default_activation=_sigmoid)
    reranker = _build(model_dir, fake_class)

    assert reranker.score("query", ["a"]) == pytest.approx([_sigmoid(2.0)])


def test_extreme_logits_stay_within_unit_interval(model_dir: Path) -> None:
    fake_class = FakeCrossEncoderClass(logits={"low": -1000.0, "high": 1000.0})
    reranker = _build(model_dir, fake_class)

    scores = reranker.score("query", ["low", "high"])

    assert scores == [0.0, 1.0]


def test_scores_all_pairs_in_one_batched_call(model_dir: Path) -> None:
    fake_class = FakeCrossEncoderClass()
    reranker = _build(model_dir, fake_class, batch_size=8)
    warm_up_calls = len(fake_class.model.predict_calls)

    reranker.score("where did I park", ["level two", "by the lift", "row C"])

    calls = fake_class.model.predict_calls[warm_up_calls:]
    assert len(calls) == 1
    assert calls[0]["sentences"] == [
        ("where did I park", "level two"),
        ("where did I park", "by the lift"),
        ("where did I park", "row C"),
    ]
    assert calls[0]["batch_size"] == 8
    assert calls[0]["show_progress_bar"] is False


def test_empty_texts_score_nothing_without_calling_the_model(model_dir: Path) -> None:
    fake_class = FakeCrossEncoderClass()
    reranker = _build(model_dir, fake_class)
    warm_up_calls = len(fake_class.model.predict_calls)

    assert reranker.score("query", []) == []
    assert len(fake_class.model.predict_calls) == warm_up_calls


def test_local_model_ignores_budget_ms(model_dir: Path) -> None:
    fake_class = FakeCrossEncoderClass(logits={"a": 1.0, "b": -1.0})
    reranker = _build(model_dir, fake_class)

    scores = reranker.score("query", ["a", "b"], budget_ms=0.001)

    assert scores == pytest.approx([_sigmoid(1.0), _sigmoid(-1.0)])


def test_predict_failure_raises_memory_rerank_error_without_the_texts(model_dir: Path) -> None:
    fake_class = FakeCrossEncoderClass()
    reranker = _build(model_dir, fake_class)
    fake_class.predict_error = RuntimeError("CUDA error: device-side assert")

    with pytest.raises(MemoryRerankError) as excinfo:
        reranker.score("private query", ["private passage"])

    assert "private" not in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_score_count_mismatch_raises(model_dir: Path) -> None:
    fake_class = FakeCrossEncoderClass()
    reranker = _build(model_dir, fake_class)
    fake_class.drop_one_score = True

    with pytest.raises(MemoryRerankError, match="2 scores for 3"):
        reranker.score("query", ["a", "b", "c"])


@pytest.mark.parametrize(
    "logit",
    [
        pytest.param(math.nan, id="nan"),
        pytest.param(math.inf, id="inf"),
        pytest.param(-math.inf, id="minus-inf"),
    ],
)
def test_non_finite_logit_raises_naming_path_and_device(model_dir: Path, logit: float) -> None:
    fake_class = FakeCrossEncoderClass()
    reranker = _build(model_dir, fake_class, device="cuda")
    fake_class.logits["broken"] = logit

    with pytest.raises(MemoryRerankError, match="non-finite") as excinfo:
        reranker.score("query", ["fine", "broken"])

    assert str(model_dir) in str(excinfo.value)
    assert "cuda" in str(excinfo.value)


def test_score_count_mismatch_names_path_and_device(model_dir: Path) -> None:
    fake_class = FakeCrossEncoderClass()
    reranker = _build(model_dir, fake_class, device="cuda")
    fake_class.drop_one_score = True

    with pytest.raises(MemoryRerankError) as excinfo:
        reranker.score("query", ["a", "b"])

    assert str(model_dir) in str(excinfo.value)
    assert "cuda" in str(excinfo.value)


def test_multi_label_checkpoint_fails_the_warm_up_naming_path_and_device(
    model_dir: Path,
) -> None:
    fake_class = FakeCrossEncoderClass(labels=2)

    with pytest.raises(MemoryRerankError, match="one score per pair") as excinfo:
        _build(model_dir, fake_class, device="cuda")

    assert str(model_dir) in str(excinfo.value)
    assert "cuda" in str(excinfo.value)


def test_multi_label_output_in_score_raises(model_dir: Path) -> None:
    fake_class = FakeCrossEncoderClass()
    reranker = _build(model_dir, fake_class)
    fake_class.labels = 3

    with pytest.raises(MemoryRerankError, match="one score per pair"):
        reranker.score("query", ["a", "b"])


# -- Release and reload -----------------------------------------------------------------------


def test_score_after_release_raises_naming_path_and_device(model_dir: Path) -> None:
    reranker = _build(model_dir, FakeCrossEncoderClass(), device="cuda")

    reranker.release()

    with pytest.raises(MemoryRerankError, match="released") as excinfo:
        reranker.score("private query", ["private passage"])
    assert str(model_dir) in str(excinfo.value)
    assert "cuda" in str(excinfo.value)
    assert "private" not in str(excinfo.value)


def test_release_drops_the_model_and_its_calibration(model_dir: Path) -> None:
    clock = FakeClock()
    fake_class = FakeCrossEncoderClass(clock=clock, overhead_ms=4.0, ms_per_pair=2.0)
    reranker = _build(model_dir, fake_class, clock=clock)

    reranker.release()

    assert reranker.cost_per_pair_ms is None
    assert reranker.call_overhead_ms is None
    _assert_unreferenced(fake_class, 0)


@pytest.mark.parametrize(
    ("budget_ms", "expected"),
    [
        pytest.param(10.0, sys.maxsize, id="positive"),
        pytest.param(0.001, sys.maxsize, id="tiny"),
        pytest.param(math.inf, sys.maxsize, id="inf"),
        pytest.param(0.0, 0, id="zero"),
        pytest.param(-10.0, 0, id="negative"),
        pytest.param(math.nan, 0, id="nan"),
    ],
)
def test_pairs_within_while_released_lets_score_report_the_fault(
    model_dir: Path, budget_ms: float, expected: int
) -> None:
    reranker = _build(model_dir, FakeCrossEncoderClass())
    reranker.release()

    assert reranker.pairs_within(budget_ms) == expected


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        pytest.param("cuda", ["gc.collect", "empty_cache"], id="cuda"),
        pytest.param("cpu", ["gc.collect"], id="cpu"),
    ],
)
def test_release_collects_then_empties_the_cuda_cache_once(
    model_dir: Path, freed: list[str], device: str, expected: list[str]
) -> None:
    reranker = _build(model_dir, FakeCrossEncoderClass(), device=device)

    reranker.release()

    assert freed == expected


def test_a_second_release_is_a_no_op(model_dir: Path, freed: list[str]) -> None:
    fake_class = FakeCrossEncoderClass()
    reranker = _build(model_dir, fake_class, device="cuda")
    reranker.release()

    reranker.release()

    assert freed == ["gc.collect", "empty_cache"]
    assert len(fake_class.built) == 1
    with pytest.raises(MemoryRerankError, match="released"):
        reranker.score("query", ["a"])


def test_each_release_after_a_reload_frees_again(model_dir: Path, freed: list[str]) -> None:
    reranker = _build(model_dir, FakeCrossEncoderClass(), device="cuda")
    reranker.release()
    assert reranker.reload() is True

    reranker.release()

    assert freed == ["gc.collect", "empty_cache"] * 2


def test_reload_loads_through_the_factory_again_and_recalibrates(model_dir: Path) -> None:
    clock = FakeClock()
    fake_class = FakeCrossEncoderClass(
        clock=clock, overhead_ms=4.0, ms_per_pair=2.0, logits={"a": 1.0}
    )
    reranker = _build(model_dir, fake_class, device="cuda", clock=clock)
    reranker.release()
    fake_class.overhead_ms, fake_class.ms_per_pair = 6.0, 3.0

    assert reranker.reload() is True

    assert len(fake_class.built) == 2
    reloaded = fake_class.built[1]
    assert (reloaded.model_name, reloaded.device, reloaded.local_files_only) == (
        str(model_dir),
        "cuda",
        True,
    )
    assert len(reloaded.predict_calls) == _WARM_UP_CALLS
    assert reranker.cost_per_pair_ms == pytest.approx(3.0)
    assert reranker.call_overhead_ms == pytest.approx(6.0)
    assert reranker.score("query", ["a"]) == pytest.approx([_sigmoid(1.0)])
    assert len(reloaded.predict_calls) == _WARM_UP_CALLS + 1


def test_reload_while_loaded_returns_true_without_loading(model_dir: Path) -> None:
    fake_class = FakeCrossEncoderClass()
    reranker = _build(model_dir, fake_class)

    assert reranker.reload() is True

    assert len(fake_class.built) == 1


def test_load_false_checks_the_directory_and_loads_nothing(model_dir: Path) -> None:
    fake_class = FakeCrossEncoderClass()

    reranker = _build(model_dir, fake_class, device="cuda", load=False)

    assert fake_class.built == []
    assert reranker.pairs_within(10.0) == sys.maxsize
    assert reranker.cost_per_pair_ms is None
    with pytest.raises(MemoryRerankError, match="released") as excinfo:
        reranker.score("query", ["a"])
    assert str(model_dir) in str(excinfo.value)
    assert "cuda" in str(excinfo.value)


def test_load_false_then_reload_loads_and_calibrates(model_dir: Path) -> None:
    clock = FakeClock()
    fake_class = FakeCrossEncoderClass(clock=clock, overhead_ms=4.0, ms_per_pair=2.0)
    reranker = _build(model_dir, fake_class, clock=clock, load=False)

    assert reranker.reload() is True

    assert len(fake_class.model.predict_calls) == _WARM_UP_CALLS
    assert reranker.cost_per_pair_ms == pytest.approx(2.0)
    assert reranker.pairs_within(50.0) == 23


def test_load_false_still_refuses_a_missing_path(tmp_path: Path) -> None:
    fake_class = FakeCrossEncoderClass()

    with pytest.raises(MemoryRerankError, match="not-baked"):
        _build(tmp_path / "not-baked", fake_class, load=False)

    assert fake_class.built == []


def test_load_false_still_refuses_an_unknown_device(model_dir: Path) -> None:
    with pytest.raises(MemoryRerankError, match="mps"):
        _build(model_dir, FakeCrossEncoderClass(), device="mps", load=False)


def test_load_false_still_reads_tokenizer_json(model_dir: Path) -> None:
    (model_dir / "tokenizer.json").write_bytes(b'{"version": "1.0", "truncation": nu')

    with pytest.raises(MemoryRerankError, match="tokenizer.json"):
        _build(model_dir, FakeCrossEncoderClass(), load=False)


def test_reload_whose_load_fails_raises_and_stays_released(model_dir: Path) -> None:
    error = OSError("model.safetensors not found")
    fake_class = FakeCrossEncoderClass(load_error=error)
    reranker = _build(model_dir, fake_class, device="cuda", load=False)

    with pytest.raises(MemoryRerankError) as excinfo:
        reranker.reload()

    assert str(model_dir) in str(excinfo.value)
    assert "cuda" in str(excinfo.value)
    assert excinfo.value.__cause__ is error
    assert reranker.pairs_within(10.0) == sys.maxsize
    with pytest.raises(MemoryRerankError, match="released"):
        reranker.score("query", ["a"])
    fake_class.load_error = None
    assert reranker.reload() is True


def test_reload_whose_warm_up_fails_raises_stays_released_and_drops_the_model(
    model_dir: Path, freed: list[str]
) -> None:
    fake_class = FakeCrossEncoderClass(predict_error=RuntimeError("CUDA out of memory"))
    reranker = _build(model_dir, fake_class, device="cuda", load=False)

    with pytest.raises(MemoryRerankError, match="warm-up") as excinfo:
        reranker.reload()

    assert "cuda" in str(excinfo.value)
    assert reranker.cost_per_pair_ms is None
    with pytest.raises(MemoryRerankError, match="released"):
        reranker.score("query", ["a"])
    assert freed == ["gc.collect", "empty_cache"]
    # Checked while the error is still alive: its traceback must not keep the model either.
    _assert_unreferenced(fake_class, 0)


def test_release_during_a_forward_pass_completes_after_it(
    model_dir: Path, freed: list[str]
) -> None:
    entered, gate = threading.Event(), threading.Event()
    fake_class = FakeCrossEncoderClass(logits={"a": 1.0})
    reranker = _build(model_dir, fake_class, device="cuda")

    def hold_the_scoring_pass(call_index: int) -> None:
        if call_index == _WARM_UP_CALLS:
            entered.set()
            _wait(gate)
            freed.append("forward pass done")

    fake_class.on_predict = hold_the_scoring_pass
    lock = _WatchedLock()
    reranker._lock = lock  # type: ignore[assignment]
    scoring = _Run(lambda: reranker.score("query", ["a"]))
    scoring.start()
    _wait(entered)

    releasing = _Run(reranker.release)
    releasing.start()
    _wait(lock.contended)

    assert releasing.is_alive()
    assert freed == []
    gate.set()
    scoring.finish()
    releasing.finish()
    assert scoring.error is None
    assert scoring.result == pytest.approx([_sigmoid(1.0)])
    assert releasing.error is None
    assert freed == ["forward pass done", "gc.collect", "empty_cache"]
    with pytest.raises(MemoryRerankError, match="released"):
        reranker.score("query", ["a"])


def test_score_during_a_reload_raises_at_once_instead_of_waiting_for_the_load(
    model_dir: Path,
) -> None:
    loading, gate = threading.Event(), threading.Event()

    def hold_the_load() -> None:
        loading.set()
        _wait(gate)

    fake_class = FakeCrossEncoderClass(on_load=hold_the_load, logits={"a": 1.0})
    reranker = _build(model_dir, fake_class, load=False)
    reloading = _Run(reranker.reload)
    reloading.start()
    _wait(loading)

    scoring = _Run(lambda: reranker.score("query", ["a"]))
    scoring.start()
    scoring.finish()

    assert isinstance(scoring.error, MemoryRerankError)
    assert "released" in str(scoring.error)
    gate.set()
    reloading.finish()
    assert reloading.error is None
    assert reloading.result is True
    assert reranker.score("query", ["a"]) == pytest.approx([_sigmoid(1.0)])


def test_release_during_a_reload_load_returns_at_once_and_supersedes_it(
    model_dir: Path, freed: list[str]
) -> None:
    loading, gate = threading.Event(), threading.Event()

    def hold_the_load() -> None:
        loading.set()
        _wait(gate)

    fake_class = FakeCrossEncoderClass(on_load=hold_the_load)
    reranker = _build(model_dir, fake_class, device="cuda", load=False)
    reloading = _Run(reranker.reload)
    reloading.start()
    _wait(loading)

    releasing = _Run(reranker.release)
    releasing.start()
    releasing.finish()

    assert releasing.error is None
    assert freed == []  # nothing was loaded for the release to free
    gate.set()
    reloading.finish()
    assert reloading.error is None
    assert reloading.result is False
    assert fake_class.built[0].predict_calls == []  # superseded before calibrating
    assert freed == ["gc.collect", "empty_cache"]
    assert reranker.pairs_within(10.0) == sys.maxsize
    with pytest.raises(MemoryRerankError, match="released"):
        reranker.score("query", ["a"])
    _assert_unreferenced(fake_class, 0)


def test_release_during_a_reload_calibration_returns_at_once_and_supersedes_it(
    model_dir: Path, freed: list[str]
) -> None:
    calibrating, gate = threading.Event(), threading.Event()

    def hold_the_first_warm_up_call(call_index: int) -> None:
        if call_index == 0:
            calibrating.set()
            _wait(gate)

    fake_class = FakeCrossEncoderClass(on_predict=hold_the_first_warm_up_call)
    reranker = _build(model_dir, fake_class, device="cuda", load=False)
    reloading = _Run(reranker.reload)
    reloading.start()
    _wait(calibrating)

    releasing = _Run(reranker.release)
    releasing.start()
    releasing.finish()

    assert releasing.error is None
    assert freed == []
    gate.set()
    reloading.finish()
    assert reloading.error is None
    assert reloading.result is False
    assert freed == ["gc.collect", "empty_cache"]
    assert reranker.cost_per_pair_ms is None
    with pytest.raises(MemoryRerankError, match="released"):
        reranker.score("query", ["a"])
    _assert_unreferenced(fake_class, 0)


def test_two_racing_reloads_leave_one_model_and_both_return_true(model_dir: Path) -> None:
    # Neither load finishes until both have started, so the two reloads overlap.
    both_loading = threading.Barrier(2, timeout=_WAIT_S)
    fake_class = FakeCrossEncoderClass(on_load=both_loading.wait, logits={"a": 1.0})
    reranker = _build(model_dir, fake_class, load=False)
    first, second = _Run(reranker.reload), _Run(reranker.reload)

    first.start()
    second.start()
    first.finish()
    second.finish()

    assert (first.error, second.error) == (None, None)
    assert (first.result, second.result) == (True, True)
    assert len(fake_class.built) == 2
    before = [len(model.predict_calls) for model in fake_class.built]
    assert reranker.score("query", ["a"]) == pytest.approx([_sigmoid(1.0)])
    after = [len(model.predict_calls) for model in fake_class.built]
    assert sorted(a - b for a, b in zip(after, before)) == [0, 1]


# -- NullReranker -----------------------------------------------------------------------------


def test_null_reranker_satisfies_the_protocol() -> None:
    assert isinstance(_needs_reranker(NullReranker()), NullReranker)


@pytest.mark.parametrize("budget_ms", [0.0, 50.0, 1e9, math.inf])
def test_null_reranker_fits_no_pairs(budget_ms: float) -> None:
    assert NullReranker().pairs_within(budget_ms) == 0


def test_null_reranker_returns_no_scores_for_the_pairs_it_was_cut_to() -> None:
    assert NullReranker().score("query", [], budget_ms=50.0) == []


def test_null_reranker_release_and_reload_change_nothing() -> None:
    reranker = NullReranker()

    reranker.release()
    reranker.release()
    assert reranker.reload() is True
    assert reranker.reload() is True
    assert reranker.pairs_within(50.0) == 0
    assert reranker.score("query", []) == []


def test_null_reranker_refuses_to_invent_scores() -> None:
    texts: Sequence[str] = ["a", "b"]

    with pytest.raises(MemoryRerankError, match="pool order"):
        NullReranker().score("query", texts)
