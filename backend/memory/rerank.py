"""Rerank step of the memory read path: the Reranker protocol and its local implementations.

The fetcher fuses the candidate pools with RRF, then asks the reranker how many of the best
pairs fit the deadline left (`pairs_within`) and scores only those (`score`); pairs left
unscored keep their RRF-normalised score (Design point 6). This module holds the local
placement, CrossEncoderReranker, and NullReranker, which stands in when the model is off or
failed to load. The remote placement over HTTP (plan task T31) is to implement the same
protocol.
"""

from __future__ import annotations

import json
import logging
import math
import numbers
import statistics
import sys
import time
from collections.abc import Callable, Iterable, Sequence
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


class CrossEncoderReranker:
    """A local cross-encoder on an explicit device, calibrated at load for the deadline.

    Loading and the warm-up both run in the constructor, so an instance always has a measured
    cost and the factory learns about a bad path or device at startup, when it can still fall
    back to NullReranker, rather than on the first turn.
    """

    def __init__(
        self,
        model_path: str,
        *,
        device: RerankDevice,
        cross_encoder_cls: CrossEncoderFactory,
        clock: Clock = time.perf_counter,
        batch_size: int = 32,
    ) -> None:
        """Load the model from a local directory and measure its cost per pair.

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

        Raises:
            MemoryRerankError: The device is not `cpu` or `cuda`; the path is not a directory;
                its `tokenizer.json` cannot be read or is not valid JSON; loading raises
                OSError, ValueError, RuntimeError or SafetensorError; or the warm-up batch fails
                with one of those or does not come back as one finite score per pair (a
                checkpoint with more than one label, for example). The message names the path
                and the device, for the startup log.

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
        self._clock = clock
        self._batch_size = batch_size
        try:
            self._model = cross_encoder_cls(model_path, device=device, local_files_only=True)
        except _MODEL_ERRORS as exc:
            raise MemoryRerankError(
                f"reranker failed to load from {model_path} on {device}: {exc}"
            ) from exc
        self._call_overhead_ms, self._cost_per_pair_ms = self._calibrate()
        logger.info(
            "Reranker loaded from %s on %s: %.3f ms per pair, %.3f ms per call",
            model_path,
            device,
            self._cost_per_pair_ms,
            self._call_overhead_ms,
        )

    @property
    def cost_per_pair_ms(self) -> float:
        """Measured cost of each pair beyond the per-call overhead, in milliseconds."""
        return self._cost_per_pair_ms

    @property
    def call_overhead_ms(self) -> float:
        """Measured fixed cost of one `predict` call, in milliseconds."""
        return self._call_overhead_ms

    def score(
        self, query: str, texts: Sequence[str], budget_ms: float | None = None
    ) -> list[float]:
        """Score each text against `query` in one batched forward pass.

        `budget_ms` is ignored: a forward pass cannot stop part way, so the fetcher bounds
        the cost beforehand with `pairs_within`.

        Raises:
            MemoryRerankError: The forward pass failed or did not return one finite score per
                pair. The message names the path and the device, never the query or the texts.
        """
        if not texts:
            return []
        logits = self._predict([(query, text) for text in texts])
        return [_sigmoid(logit) for logit in logits]

    def pairs_within(self, budget_ms: float) -> int:
        """How many pairs one `score` call can finish in `budget_ms`.

        The cost model is the warm-up's: a fixed overhead per call plus a cost per pair. A
        straight cost per pair from one batch size would misjudge small batches, which is where
        a tight deadline puts the fetcher.
        """
        if math.isnan(budget_ms) or budget_ms <= self._call_overhead_ms:
            return 0
        if math.isinf(budget_ms):
            return sys.maxsize
        return math.floor((budget_ms - self._call_overhead_ms) / self._cost_per_pair_ms)

    def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Run one forward pass and check its output is one finite logit per pair.

        The checks run here, so the warm-up and `score` reject the same outputs. A checkpoint
        with more than one label returns a row per pair, which `float` cannot take, and a NaN
        or infinite logit means a broken forward pass (an fp16 overflow, for example), not a
        relevance judgement.
        """
        where = f"reranker at {self._model_path} on {self._device}"
        try:
            raw = list(
                self._model.predict(
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

    def _calibrate(self) -> tuple[float, float]:
        """Measure the per-call overhead and the per-pair cost on the fixed batch.

        The first call is not timed: it moves the weights to the device and pays the other
        one-off setup costs of a first forward pass, which no turn pays again. Each later round
        times one pair and the whole batch; the medians give a line through two points, whose
        slope is the cost per pair and whose remainder at one pair is the overhead.
        """
        batch = [(_CALIBRATION_QUERY, text) for text in _CALIBRATION_TEXTS]
        single = batch[:1]
        try:
            self._predict(batch)
            single_ms: list[float] = []
            batch_ms: list[float] = []
            for _ in range(_CALIBRATION_ROUNDS):
                single_ms.append(self._timed_ms(single))
                batch_ms.append(self._timed_ms(batch))
        except MemoryRerankError as exc:
            raise MemoryRerankError(
                f"reranker warm-up failed for {self._model_path} on {self._device}: {exc}"
            ) from exc
        one = statistics.median(single_ms)
        many = statistics.median(batch_ms)
        per_pair = max((many - one) / (len(batch) - 1), _MIN_COST_PER_PAIR_MS)
        overhead = max(one - per_pair, 0.0)
        return overhead, per_pair

    def _timed_ms(self, pairs: list[tuple[str, str]]) -> float:
        start = self._clock()
        self._predict(pairs)
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
