"""Rerank step of the memory read path: the Reranker protocol and its local implementations.

The fetcher fuses the candidate pools with RRF, then asks the reranker how many of the best
pairs fit the deadline left (`pairs_within`) and scores only those (`score`); pairs left
unscored keep their RRF-normalised score (Design point 6). This module holds the local
placement, CrossEncoderReranker, and NullReranker, which stands in when the model is off or
failed to load. The remote placement over HTTP (plan task T31) is to implement the same
protocol.

The local model stays resident between turns and is released only on request (`release`), so
the GPU can be handed to another program, and loaded again with `reload`.
"""

from __future__ import annotations

import gc
import json
import logging
import math
import numbers
import statistics
import sys
import threading
import time
import traceback
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from safetensors import SafetensorError

from backend.errors import MemoryRerankError

from .config import RerankDevice

logger = logging.getLogger(__name__)

Clock = Callable[[], float]

_DEVICES: tuple[str, ...] = ("cpu", "cuda")

# Errors a checkpoint load or a forward pass raises for a missing or corrupt file, a config the
# library cannot read, an unavailable device or an out-of-memory condition. SafetensorError
# derives from Exception directly, so it has to be named.
_MODEL_ERRORS: tuple[type[Exception], ...] = (OSError, ValueError, RuntimeError, SafetensorError)

# The smallest cost per pair `pairs_within` divides by. A clock too coarse to see the
# calibration batch measures zero, and the count must stay finite.
_MIN_COST_PER_PAIR_MS = 1e-3

# Milliseconds are kept to six decimals (one nanosecond); see `_timed_ms`.
_TIMING_DECIMALS = 6

# Timed rounds per batch size. The median of three discards one call slowed by a GC pause or a
# competing thread without biasing the estimate the way a minimum would.
_CALIBRATION_ROUNDS = 3

# A fixed batch of generic pairs shaped like memory items (one sentence or two), so the
# calibration costs about what a real rerank costs per pair. Fixed so that two warm-ups of the
# same model on the same device measure the same work.
_CALIBRATION_QUERY = "what did we decide about the weekend trip and the train tickets"
_CALIBRATION_TEXTS: tuple[str, ...] = (
    "The user prefers window seats on long train journeys and books them a week ahead.",
    "We agreed to leave on Saturday morning and come back on Sunday evening.",
    "The user said the hotel near the station was too noisy last time.",
    "Train tickets are cheaper when bought as two singles rather than a return.",
    "The user works from home on Fridays and usually starts early.",
    "I suggested packing a light jacket because rain was forecast for the coast.",
    "The user's favourite cafe closes at four on weekends.",
    "We talked about visiting the museum if the weather turned bad.",
    "The user asked me to remind them to water the plants before leaving.",
    "Booking opened for the autumn timetable on the first of the month.",
    "The user mentioned the dog stays with a neighbour when they travel.",
    "I said the earlier train has a connection that is often missed.",
    "The user is reading a novel set in a coastal town in the north.",
    "We compared two routes, one with a change and one direct but slower.",
    "The user would rather not drive because parking near the old town is scarce.",
    "The user decided to buy the tickets on Thursday after payday.",
)


class Reranker(Protocol):
    """Scores query-text pairs in [0, 1] and says how many pairs fit a time budget.

    The fetcher calls `pairs_within` with the deadline left, cuts the best RRF candidates to
    that many, and passes them to `score`. A placement that cannot score at all reports zero
    pairs, so the cut empties the batch and every candidate keeps its RRF score; this is how
    NullReranker stands in without changing the return type.
    """

    def score(
        self, query: str, texts: Sequence[str], budget_ms: float | None = None
    ) -> list[float]:
        """Score each text against `query`.

        Args:
            query: The user's utterance.
            texts: Candidate texts, best RRF first.
            budget_ms: Time left before the fetch deadline. The local model ignores it, since a
                forward pass cannot be interrupted; the remote placement caps its timeout with
                it.

        Returns:
            One score in [0, 1] per text, in input order.

        Raises:
            MemoryRerankError: The pairs cannot be scored; the fetcher keeps the RRF scores.
        """
        ...

    def pairs_within(self, budget_ms: float) -> int:
        """How many pairs can be scored in `budget_ms`; zero when none fit."""
        ...

    def release(self) -> None:
        """Free the memory the placement holds for scoring, if any; idempotent.

        Safe to call from any thread, including while `score` runs on another. A placement that
        holds a model frees it after the forward pass in progress, and until `reload` its
        `score` raises MemoryRerankError for a non-empty batch. One that holds none does
        nothing.
        """
        ...

    def reload(self) -> bool:
        """Load the model again after `release`; idempotent.

        Returns:
            True when a model is loaded on return, or the placement holds none to load; False
            when a `release` that came in while it loaded superseded it, which leaves the
            placement released.

        Raises:
            MemoryRerankError: The load or its warm-up failed; the placement stays released.
        """
        ...


class CrossEncoderModel(Protocol):
    """The part of `sentence_transformers.CrossEncoder` the reranker calls."""

    def predict(
        self,
        sentences: list[tuple[str, str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
        activation_fct: Callable[[object], object],
        convert_to_numpy: bool,
    ) -> Iterable[float]:
        """Score each pair with the model's raw output passed through `activation_fct`."""
        ...


class CrossEncoderFactory(Protocol):
    """Loads a cross-encoder; `sentence_transformers.CrossEncoder` itself satisfies it."""

    def __call__(
        self, model_name: str, *, device: str, local_files_only: bool
    ) -> CrossEncoderModel:
        """Load the model at `model_name` onto `device`."""
        ...


@dataclass(frozen=True, slots=True)
class _Loaded:
    """A loaded model with the costs its warm-up measured, swapped in and out as one.

    One reference holds all three, so `pairs_within` reads the model's costs without the lock
    and never pairs one model's overhead with another's cost per pair.
    """

    model: CrossEncoderModel
    call_overhead_ms: float
    cost_per_pair_ms: float


def _identity(logits: object) -> object:
    """Pass the model's logits through unchanged.

    CrossEncoder applies the checkpoint's default activation when the caller passes none, and a
    checkpoint can name its own (`sbert_ce_default_activation_function` in its config) or fall
    back to a sigmoid for a one-label head. The output scale would then depend on the
    checkpoint, and a sigmoid applied here on top of one applied there would squeeze every score
    towards the middle. Asking for raw logits and applying the sigmoid once in `_sigmoid` keeps
    [0, 1] for any checkpoint `MIST_MEMORY_RERANK_MODEL` names.
    """
    return logits


def _sigmoid(x: float) -> float:
    """Logistic function written so that neither branch overflows for large |x|."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _check_tokenizer_json(model_dir: Path, device: str) -> None:
    """Parse the model directory's `tokenizer.json`, when it has one, before loading.

    tokenizers reports a truncated or undecodable `tokenizer.json` with a plain `Exception`,
    which the load cannot catch without a bare `except Exception`. Reading the file here first
    turns those faults into MemoryRerankError, so the factory can fall back to NullReranker.
    """
    path = model_dir / "tokenizer.json"
    if not path.is_file():
        return
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MemoryRerankError(
            f"reranker tokenizer.json in {model_dir} for {device} is unreadable: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _clear_frames(exc: BaseException) -> None:
    """Drop the locals of the finished frames in `exc`'s traceback and in its causes'.

    A failed load or warm-up leaves the new model in those frames (the factory's `self`, the
    `model` argument of `_calibrate` and `_predict`), so without this the caller's handler would
    keep the weights alive, on the GPU for `cuda`, until it let go of the error, and the cache
    emptied before then could not return their memory. Frames still running are left as they
    are.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        traceback.clear_frames(current.__traceback__)
        current = current.__cause__ or current.__context__


class CrossEncoderReranker:
    """A local cross-encoder on an explicit device, calibrated at each load for the deadline.

    By default the constructor loads the model and runs the warm-up, so the factory learns about
    a bad path or device at startup, when it can still fall back to NullReranker, rather than on
    the first turn. `release` drops the model and `reload` loads and calibrates a new one, so the
    GPU can be lent to another program and taken back while the service runs.

    One lock guards the loaded model. `score` holds it for the forward pass, and `release` and
    the swap at the end of `reload` hold it to change the model, so no model is dropped or
    replaced under a forward pass. A reload loads and calibrates its new model without the lock,
    so a `score` meanwhile raises at once (the instance is released) instead of waiting for the
    load. A generation counter, bumped by each `release`, tells a reload that a release came in
    while it loaded, so the later request wins.
    """

    def __init__(
        self,
        model_path: str,
        *,
        device: RerankDevice,
        cross_encoder_cls: CrossEncoderFactory,
        clock: Clock = time.perf_counter,
        batch_size: int = 32,
        load: bool = True,
    ) -> None:
        """Check the model directory and, unless `load` is False, load the model and calibrate.

        Args:
            model_path: Directory holding a baked cross-encoder (`MIST_MEMORY_RERANK_MODEL`).
                Loaded with `local_files_only`, so a missing file fails here instead of
                reaching for the network.
            device: `cpu` or `cuda` (`MIST_MEMORY_RERANK_DEVICE`). Passed to the model
                explicitly: left unset, sentence-transformers picks `cuda` whenever CUDA is
                visible, which would put the reranker on the GPU beside the main model.
            cross_encoder_cls: Loads the model; production passes
                `sentence_transformers.CrossEncoder`, tests a fake class.
            clock: Monotonic seconds, used to time the warm-up.
            batch_size: Pairs per forward pass.
            load: Load and calibrate now. False runs the device, path and `tokenizer.json`
                checks only and starts released, with no model and no calibration until
                `reload`.

        Raises:
            MemoryRerankError: The device is not `cpu` or `cuda`; the path is not a directory;
                its `tokenizer.json` cannot be read or is not valid JSON; or, when `load` is
                True, loading raises OSError, ValueError, RuntimeError or SafetensorError, or
                the warm-up batch fails with one of those or does not come back as one finite
                score per pair (a checkpoint with more than one label, for example). The message
                names the path and the device, for the startup log.

        Known limit: tokenizers reports some faults with a plain `Exception`, which this
        constructor does not catch, since a bare `except Exception` would also swallow
        programming errors. The read of `tokenizer.json` above turns a truncated or undecodable
        file into MemoryRerankError, but a file that is valid JSON and that tokenizers still
        rejects fails startup loudly instead of falling back to NullReranker. The baked model is
        pinned by revision (H3), so such a file means a broken bake rather than a runtime fault.
        """
        if device not in _DEVICES:
            raise MemoryRerankError(
                f"reranker device must be one of {_DEVICES}, got {device!r} for {model_path}"
            )
        if not Path(model_path).is_dir():
            raise MemoryRerankError(
                f"reranker model path {model_path} does not exist or is not a directory"
            )
        _check_tokenizer_json(Path(model_path), device)
        self._model_path = model_path
        self._device = device
        self._cross_encoder_cls = cross_encoder_cls
        self._clock = clock
        self._batch_size = batch_size
        self._lock = threading.Lock()
        # Bumped by every `release`, under the lock. A reload that finds it changed since it
        # started was superseded by a release and drops the model it loaded.
        self._generation = 0
        # Replaced as a whole, under the lock; None while released. `pairs_within` and the
        # cost properties read it without the lock, which is safe: one attribute read sees
        # either the old reference or the new one.
        self._loaded: _Loaded | None = None
        if load:
            self.reload()

    @property
    def cost_per_pair_ms(self) -> float | None:
        """Measured cost of each pair beyond the per-call overhead, in ms; None if released."""
        loaded = self._loaded
        return None if loaded is None else loaded.cost_per_pair_ms

    @property
    def call_overhead_ms(self) -> float | None:
        """Measured fixed cost of one `predict` call, in ms; None if released."""
        loaded = self._loaded
        return None if loaded is None else loaded.call_overhead_ms

    def score(
        self, query: str, texts: Sequence[str], budget_ms: float | None = None
    ) -> list[float]:
        """Score each text against `query` in one batched forward pass.

        `budget_ms` is ignored: a forward pass cannot stop part way, so the fetcher bounds
        the cost beforehand with `pairs_within`. The lock is held for the forward pass, so a
        `release` meanwhile waits for it to finish.

        Raises:
            MemoryRerankError: The reranker is released; or the forward pass failed or did not
                return one finite score per pair. The message names the path and the device,
                never the query or the texts.
        """
        if not texts:
            return []
        pairs = [(query, text) for text in texts]
        with self._lock:
            loaded = self._loaded
            if loaded is None:
                raise MemoryRerankError(
                    f"reranker at {self._model_path} on {self._device} is released; "
                    f"keep the RRF scores for the {len(pairs)} pairs"
                )
            logits = self._predict(loaded.model, pairs)
        return [_sigmoid(logit) for logit in logits]

    def pairs_within(self, budget_ms: float) -> int:
        """How many pairs one `score` call can finish in `budget_ms`.

        The cost model is the warm-up's: a fixed overhead per call plus a cost per pair. A
        straight cost per pair from one batch size would misjudge small batches, which is where
        a tight deadline puts the fetcher.

        While released there is no warm-up to go by, and any positive budget gets
        `sys.maxsize`: the fetcher then cuts the batch to its shortlist and calls `score`, which
        raises, so the fetch records the reranker as unavailable. A count of zero would make
        the fetcher skip `score` and record no reason at all.
        """
        if math.isnan(budget_ms):
            return 0
        loaded = self._loaded
        if loaded is None:
            return sys.maxsize if budget_ms > 0 else 0
        if budget_ms <= loaded.call_overhead_ms:
            return 0
        if math.isinf(budget_ms):
            return sys.maxsize
        return math.floor((budget_ms - loaded.call_overhead_ms) / loaded.cost_per_pair_ms)

    def release(self) -> None:
        """Drop the model and its calibration, then return their memory; idempotent.

        Waits for a forward pass in progress, never for a load: a reload running on another
        thread is superseded and drops the model it loaded. Outside the lock, once the instance
        holds no reference to the model, it collects garbage and, on `cuda`, empties PyTorch's
        cache. A release while released frees nothing, but still supersedes a reload in
        progress.
        """
        with self._lock:
            self._generation += 1
            loaded, self._loaded = self._loaded, None
        if loaded is None:
            return
        del loaded
        self._free_memory()
        logger.info("Reranker released from %s on %s", self._model_path, self._device)

    def reload(self) -> bool:
        """Load and calibrate a new model when none is loaded; idempotent.

        The load and the warm-up run on the calling thread without the lock, on the new model
        rather than the instance's, so a `score` meanwhile raises at once instead of waiting.
        The generation is checked before the warm-up and again under the lock at the swap: if
        a `release` came in since this call started, the new model is dropped and its memory
        freed. If another reload swapped a model in first, this one drops its own.

        Returns:
            True when a model is loaded at the swap, this call's or another reload's; False
            when a `release` superseded this call and no model is loaded, which leaves the
            instance released.

        Raises:
            MemoryRerankError: The load or the warm-up failed, for the causes the constructor
                lists. The instance stays released and the new model's memory is freed.
        """
        with self._lock:
            if self._loaded is not None:
                return True
            generation = self._generation
        try:
            model = self._load_model()
        except MemoryRerankError as exc:
            _clear_frames(exc)
            self._free_memory()
            raise
        with self._lock:
            superseded = self._generation != generation
            loaded_elsewhere = self._loaded is not None
        if superseded or loaded_elsewhere:
            del model
            self._free_memory()
            return loaded_elsewhere
        failure: MemoryRerankError | None = None
        try:
            candidate = _Loaded(model, *self._calibrate(model))
        except MemoryRerankError as exc:
            _clear_frames(exc)
            failure = exc
        # Dropped on both paths: on failure, before the memory is freed.
        del model
        if failure is not None:
            self._free_memory()
            raise failure
        with self._lock:
            superseded = self._generation != generation
            loaded_elsewhere = self._loaded is not None
            if not (superseded or loaded_elsewhere):
                self._loaded = candidate
        if superseded or loaded_elsewhere:
            del candidate
            self._free_memory()
            return loaded_elsewhere
        logger.info(
            "Reranker loaded from %s on %s: %.3f ms per pair, %.3f ms per call",
            self._model_path,
            self._device,
            candidate.cost_per_pair_ms,
            candidate.call_overhead_ms,
        )
        return True

    def _load_model(self) -> CrossEncoderModel:
        """Load a new model through the factory, turning its load errors into ours."""
        try:
            return self._cross_encoder_cls(
                self._model_path, device=self._device, local_files_only=True
            )
        except _MODEL_ERRORS as exc:
            raise MemoryRerankError(
                f"reranker failed to load from {self._model_path} on {self._device}: {exc}"
            ) from exc

    def _free_memory(self) -> None:
        """Collect garbage and, on `cuda`, return PyTorch's cached blocks to the driver.

        Called without the lock, once nothing here references the dropped model. `gc.collect`
        reclaims the model if a reference cycle still holds it, and `torch.cuda.empty_cache`
        then gives the blocks the caching allocator kept back to the driver, where another
        program on the GPU can use them.
        """
        gc.collect()
        if self._device == "cuda":
            # Imported here because only `cuda` needs it; a `cpu` reranker never loads torch
            # through this module.
            import torch

            torch.cuda.empty_cache()

    def _predict(self, model: CrossEncoderModel, pairs: list[tuple[str, str]]) -> list[float]:
        """Run one forward pass and check its output is one finite logit per pair.

        The checks run here, so the warm-up and `score` reject the same outputs. A checkpoint
        with more than one label returns a row per pair, which `float` cannot take, and a NaN
        or infinite logit means a broken forward pass (an fp16 overflow, for example), not a
        relevance judgement. The model is passed in, not read from the instance, so a reload
        can calibrate its new model before swapping it in.
        """
        where = f"reranker at {self._model_path} on {self._device}"
        try:
            raw = list(
                model.predict(
                    pairs,
                    batch_size=self._batch_size,
                    show_progress_bar=False,
                    activation_fct=_identity,
                    convert_to_numpy=True,
                )
            )
        except _MODEL_ERRORS as exc:
            # The pairs are user text, so only the exception type reaches the message.
            raise MemoryRerankError(
                f"{where} failed to score {len(pairs)} pairs: {type(exc).__name__}"
            ) from exc
        if len(raw) != len(pairs):
            raise MemoryRerankError(f"{where} returned {len(raw)} scores for {len(pairs)} pairs")
        # numpy registers its floating types as `numbers.Real`; a row of a 2-d array is not.
        if not all(isinstance(value, numbers.Real) for value in raw):
            raise MemoryRerankError(
                f"{where} did not return one score per pair; a checkpoint with more than one "
                "label cannot rerank"
            )
        logits = [float(value) for value in raw]
        if not all(math.isfinite(logit) for logit in logits):
            raise MemoryRerankError(f"{where} returned a non-finite logit")
        return logits

    def _calibrate(self, model: CrossEncoderModel) -> tuple[float, float]:
        """Measure `model`'s per-call overhead and per-pair cost on the fixed batch.

        The first call is not timed: it moves the weights to the device and pays the other
        one-off setup costs of a first forward pass, which no turn pays again. Each later round
        times one pair and the whole batch; the medians give a line through two points, whose
        slope is the cost per pair and whose remainder at one pair is the overhead.
        """
        batch = [(_CALIBRATION_QUERY, text) for text in _CALIBRATION_TEXTS]
        single = batch[:1]
        try:
            self._predict(model, batch)
            single_ms: list[float] = []
            batch_ms: list[float] = []
            for _ in range(_CALIBRATION_ROUNDS):
                single_ms.append(self._timed_ms(model, single))
                batch_ms.append(self._timed_ms(model, batch))
        except MemoryRerankError as exc:
            raise MemoryRerankError(
                f"reranker warm-up failed for {self._model_path} on {self._device}: {exc}"
            ) from exc
        one = statistics.median(single_ms)
        many = statistics.median(batch_ms)
        per_pair = max((many - one) / (len(batch) - 1), _MIN_COST_PER_PAIR_MS)
        overhead = max(one - per_pair, 0.0)
        return overhead, per_pair

    def _timed_ms(self, model: CrossEncoderModel, pairs: list[tuple[str, str]]) -> float:
        start = self._clock()
        self._predict(model, pairs)
        # A difference of two large clock readings carries float noise far below any timer's
        # resolution; rounding to a nanosecond keeps `pairs_within` from losing a pair to it
        # at an exact boundary.
        return round((self._clock() - start) * 1000.0, _TIMING_DECIMALS)


class NullReranker:
    """Stands in when reranking is off or the model failed to load.

    The plan has it return None so callers keep the RRF scores. The protocol's return type is
    `list[float]`, so the same outcome comes from `pairs_within`: it reports that no pair fits
    any budget, the fetcher's cut leaves nothing to score, and `score` on the empty batch
    returns an empty list. Every candidate keeps its RRF-normalised score.
    """

    def score(
        self, query: str, texts: Sequence[str], budget_ms: float | None = None
    ) -> list[float]:
        """Return no scores for the empty batch `pairs_within` cut the candidates to.

        Raises:
            MemoryRerankError: `texts` is not empty. A caller that skipped `pairs_within`
                gets the error every reranker fault raises, which keeps the RRF scores,
                rather than invented scores.
        """
        if texts:
            raise MemoryRerankError(
                f"NullReranker scores nothing; keep the RRF scores for the {len(texts)} pairs"
            )
        return []

    def pairs_within(self, budget_ms: float) -> int:
        """No pair fits: there is no model to score with."""
        return 0

    def release(self) -> None:
        """Nothing to free: no model is held."""

    def reload(self) -> bool:
        """Nothing to load; reports ready, since `pairs_within` already keeps the RRF scores."""
        return True
