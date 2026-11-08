"""Run the memory fetcher over the recall fixture and write the run file the scorer reads.

The harness builds the read path's indexes from `tests/fixtures/memory_recall/`: the fact index
from `facts.jsonl` and a vault sidecar from `vault_notes.jsonl`. It then runs one
`MemoryFetcher.fetch` per probe, as a new session with no earlier turns, and writes one JSONL
record per probe in the shape `memory_recall.py --run` scores.

The indexes hold what MIST would have known at each probe's `now`: a fact once it is recorded
and until a correction removes it, a note once it is created. Probes run in `now` order and the
indexes are brought to each `now` before its probes run. Within one `now` every probe sees the
same indexes, and the fetcher filters each fact's versions against that `now`.

Each record carries, beside the scorer's keys (`probe_id`, `items`, `tokens`,
`budget_tokens`): the fetch's timings in milliseconds, with `total_ms` as `critical_path_ms`;
the pairs reranked; `shortlisted`, the size of the scored shortlist, so a short shortlist can
be told from a rerank cut; the degraded reason; and `delta_tokens_est`. With no conversation
history, the turn's uncached delta is the user's new message carrying the block, so the
estimate is the block's tokens plus the query's, both by the planner's counter. A fact list
item is written as one item per fact, all under the list's ref, so the scorer sees every
sibling the block named. Each item carries the fetcher's own rank for it, which the scorer's
MRR reads.

Models are either fakes, deterministic and weight-free (the floors test uses them), or the real
ones, loaded only from `/opt/models` (Design point 17): the embedder from
`/opt/models/all-MiniLM-L6-v2` and the cross-encoder from `MemoryConfig.rerank_model`, which
`--rerank-model` overrides. `--threads N` sets torch's intra-op thread count before any model
loads; without it torch keeps its default.

`--cpu-load-procs N` runs N busy-loop processes for the whole run, to measure the critical path
with the CPU contended the way a speaking turn contends it.

`--placement` builds the block for `user_prefix` (the default) or `system_tail`; it changes
only the wording of the user's past words inside the block (block.py, `BlockPlacement`).

`--rescan-facts N` runs forced full rescans of the fact index for the whole run, the way
`FactIndexRefresher` runs `diff_apply` on the `memory-index` executor, so the fetches meet the
contention of a refresh. Each rescan applies the latest snapshot plus N synthetic facts at a
new revision, so it re-embeds N facts with the run's embedder. `--rescan-log FILE` records
each cycle.

`--trace FILE` writes JSONL that attributes each probe's misses to a stage: a run header
(device, threads, model paths, the reranker's cost model, the effective MemoryConfig), then one
record per probe with every fused candidate's pool ranks, RRF and pair length, every scored
shortlist entry's terms, the rerank and embed calls, and for each expected and forbidden fact
or note its unbounded search ranks, its fused and shortlist positions, its disposition and an
oracle logit from the reranker with no deadline (`ProbeTracer`). The trace's own work runs
after each fetch returns, never inside the timed path.

Usage, from the repository root (`/app` in the image), so `backend` imports:
    python -m scripts.eval_harness.memory_recall_run --output run.jsonl
        [--embedder fake|real] [--reranker fake|real] [--device cpu|cuda]
        [--rerank-model PATH] [--threads N] [--placement user_prefix|system_tail]
        [--cpu-load-procs N] [--rescan-facts N --rescan-log FILE] [--trace FILE]
        [--block-tokens N] [--min-score X] [--rerank-top-n N]
        [--list-max-items N] [--deadline-ms N] [--fixture-dir DIR]
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import math
import multiprocessing
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import CancelledError, Executor
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol

import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from backend.chat.context_budget import ApproximateTokenCounter
from backend.errors import MistError
from backend.knowledge.config import SidecarIndexConfig
from backend.memory.block import BlockPlacement, MemoryBlockBuilder, PlacedList
from backend.memory.config import MemoryConfig
from backend.memory.fact_index import DiffReport, FactIndex, IndexedFact
from backend.memory.fetcher import (
    INDEX_EXECUTOR,
    MODEL_EXECUTOR,
    FetcherExecutors,
    FetchOutcome,
    MemoryFetcher,
    SystemClock,
)
from backend.memory.fts import query_tokens
from backend.memory.index_db import EMBEDDING_DIMENSION, open_index, open_reader
from backend.memory.models import FactRow, MemoryRequest, MemoryResult, NoteSource
from backend.memory.ranking import FusedCandidate
from backend.memory.rerank import CrossEncoderReranker, Reranker
from backend.memory.runtime import IndexStateProbe, build_memory_executors
from backend.memory.sources import (
    ENTITY_POOL,
    FACT_POOLS,
    FILE_HEADING,
    FTS_POOL,
    NOTE_FTS_POOL,
    NOTE_VECTOR_POOL,
    VECTOR_POOL,
    VaultCandidates,
    note_item_id,
)
from backend.vault.sidecar_index import VaultSidecarIndex
from scripts.eval_harness.memory_recall import (
    FIXTURE_DIR,
    Correction,
    Fact,
    Probe,
    VaultNote,
    load_corrections,
    load_facts,
    load_probes,
    load_vault_notes,
    parse_run_record,
    score_run,
)

MODELS_ROOT = Path("/opt/models")
EMBEDDER_PATH = MODELS_ROOT / "all-MiniLM-L6-v2"
EMBEDDING_MODEL_ID = "all-MiniLM-L6-v2"

ModelChoice = Literal["fake", "real"]
Device = Literal["cpu", "cuda"]

# How long stopping the load processes may take before the stragglers are terminated.
_LOAD_JOIN_TIMEOUT_S = 5.0
# Busy-loop iterations between checks of the stop event, so a check costs little of the load.
_SPIN_BATCH = 100_000


# -- the models ------------------------------------------------------------------------------


class SentenceTransformerEmbedder:
    """The production embedding behaviour on an explicit device, for the harness.

    EmbeddingGenerator chooses its device only through its caller, so this class builds the
    model itself with `--device` and repeats EmbeddingGenerator's encode calls exactly
    (embedding_generator.py, `generate_embedding` and `generate_embeddings_batch`), so the
    harness's vectors are the production embedder's for the same model. A test patches one
    fake `SentenceTransformer` into both modules and compares the calls.

    Args:
        model_path: A local model directory under `/opt/models`.
        device: The torch device to load onto.
    """

    def __init__(self, model_path: str, *, device: str) -> None:
        self.model_path = model_path
        self.device = device
        self._model = SentenceTransformer(model_path, device=device)

    def generate_embedding(self, text: str) -> list[float]:
        """The text's vector; 384 zeros, without encoding, for an empty or blank text."""
        if not text or not text.strip():
            return [0.0] * EMBEDDING_DIMENSION
        embedding = self._model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return embedding.tolist()

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """One vector per text; a blank entry is encoded as one space, as production does."""
        if not texts:
            return []
        valid_texts = [t if t and t.strip() else " " for t in texts]
        embeddings = self._model.encode(
            valid_texts,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100,
            batch_size=32,
        )
        return embeddings.tolist()


def _stem(token: str) -> str:
    """A light suffix strip, so "works", "working" and "worked" meet as "work".

    A doubled final consonant left by the strip is undoubled ("running" to "run").
    """
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            stem = token[: -len(suffix)]
            if len(stem) >= 3 and stem[-1] == stem[-2] and stem[-1] not in "aeiouls":
                stem = stem[:-1]
            return stem
    return token


def content_stems(text: str) -> set[str]:
    """The stemmed content words of `text`: the FTS query's tokens, suffixes stripped."""
    return {_stem(token) for token in query_tokens(text)}


class TokenEmbedder:
    """A weight-free embedder: each stemmed content word hashed to a signed dimension.

    Texts sharing content words have a positive cosine similarity and stopwords add nothing,
    so the vector pool behaves like a weak lexical pool. A text with no content word maps to
    the first basis vector, so every vector is a unit vector.
    """

    def generate_embedding(self, text: str) -> list[float]:
        """The unit vector of `text`'s stemmed content words."""
        vec = [0.0] * EMBEDDING_DIMENSION
        for stem in sorted(content_stems(text)):
            digest = hashlib.sha256(stem.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % EMBEDDING_DIMENSION
            vec[index] += 1.0 if digest[4] % 2 == 0 else -1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            vec[0] = 1.0
            return vec
        return [v / norm for v in vec]

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """`generate_embedding` of each text, in order."""
        return [self.generate_embedding(text) for text in texts]


class TokenReranker:
    """A weight-free reranker: the share of the query's stemmed content words in the text.

    Args:
        pairs_per_ms: How many pairs `pairs_within` says fit each millisecond.
    """

    def __init__(self, *, pairs_per_ms: float = 10.0) -> None:
        self.pairs_per_ms = pairs_per_ms

    def score(
        self, query: str, texts: Sequence[str], budget_ms: float | None = None
    ) -> list[float]:
        """Each text's share of the query's content stems, in [0, 1]."""
        wanted = content_stems(query)
        if not wanted:
            return [0.0 for _ in texts]
        return [len(wanted & content_stems(text)) / len(wanted) for text in texts]

    def pairs_within(self, budget_ms: float) -> int:
        """The pairs that fit `budget_ms`; 0 for no time."""
        if budget_ms <= 0:
            return 0
        return int(budget_ms * self.pairs_per_ms)


def _model_path(path: str | Path) -> str:
    """`path`, refused unless it is under `/opt/models`, so a run never downloads weights."""
    resolved = Path(path)
    if MODELS_ROOT not in (resolved, *resolved.parents):
        raise ValueError(f"real models load from {MODELS_ROOT} only, got {path}")
    return str(resolved)


def build_embedder(
    choice: ModelChoice, *, device: Device
) -> TokenEmbedder | SentenceTransformerEmbedder:
    """The fake or the real embedder."""
    if choice == "fake":
        return TokenEmbedder()
    return SentenceTransformerEmbedder(_model_path(EMBEDDER_PATH), device=device)


PairTokens = Callable[[str, str], int]


def word_pair_tokens(query: str, text: str) -> int:
    """The fake reranker's length of a pair: the whitespace words of the query and the text."""
    return len(query.split()) + len(text.split())


class CrossEncoderCapture:
    """Loads `CrossEncoder` for `CrossEncoderReranker` and keeps the model it loaded.

    Passed as the reranker's `cross_encoder_cls`, so the trace can count a pair's tokens with
    the model's own tokenizer, truncated as `CrossEncoder.predict` truncates the pairs it
    scores: the pair together, longest first, to the model's `max_length` (the tokenizer's
    `model_max_length` when that is None).
    """

    def __init__(self) -> None:
        self.model: CrossEncoder | None = None

    def __call__(self, model_name: str, *, device: str, local_files_only: bool) -> CrossEncoder:
        """Load the model, as `CrossEncoder` itself would be called, and keep it."""
        self.model = CrossEncoder(model_name, device=device, local_files_only=local_files_only)
        return self.model

    def pair_tokens(self, query: str, text: str) -> int:
        """The tokens the model reads for the pair, after truncation.

        Raises:
            RuntimeError: No model has been loaded through this capture.
        """
        if self.model is None:
            raise RuntimeError("no cross-encoder was loaded through this capture")
        encoded = self.model.tokenizer(
            query, text, truncation="longest_first", max_length=self.model.max_length
        )
        return len(encoded["input_ids"])


@dataclass(frozen=True, slots=True)
class BuiltReranker:
    """A reranker for the run, with what the trace reads about it.

    Attributes:
        reranker: The reranker the fetcher scores with.
        path: The model directory it loaded from; None for the fake.
        pair_tokens: Counts a pair's tokens as the reranker reads them.
    """

    reranker: Reranker
    path: str | None
    pair_tokens: PairTokens


def build_reranker(choice: ModelChoice, *, device: Device, config: MemoryConfig) -> BuiltReranker:
    """The fake or the real cross-encoder, the real one from `config.rerank_model`."""
    if choice == "fake":
        return BuiltReranker(reranker=TokenReranker(), path=None, pair_tokens=word_pair_tokens)
    path = _model_path(config.rerank_model)
    capture = CrossEncoderCapture()
    reranker = CrossEncoderReranker(path, device=device, cross_encoder_cls=capture)
    return BuiltReranker(reranker=reranker, path=path, pair_tokens=capture.pair_tokens)


# -- the fixture as the indexes see it ------------------------------------------------------


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def fact_row(fact: Fact) -> FactRow:
    """The index row of one fixture fact version."""
    return FactRow(
        row_key=fact.row_key,
        fact_key=fact.fact_key,
        subject_id=fact.subject_id,
        subject_name=fact.subject_name,
        subject_type=fact.subject_type,
        rel_type=fact.relation,
        object_id=fact.object_id,
        object_name=fact.object_name,
        object_type=fact.object_type,
        valid_from=_iso(fact.valid_from),
        valid_to=_iso(fact.valid_to),
        recorded_at=_iso(fact.recorded_at),
        confidence=fact.confidence,
        source_utterance_id=fact.source_utterance_id,
    )


def facts_known_at(
    facts: Sequence[Fact], corrections: Sequence[Correction], now: datetime
) -> list[FactRow]:
    """The fact versions the graph holds at `now`: recorded by then, not corrected away.

    A forget or correct removes the fact it names from the graph when it is recorded; a
    corrected fact's replacement is its own fixture row, recorded at the correction.
    """
    removed = {c.fact_key for c in corrections if c.recorded_at <= now}
    return [
        fact_row(fact) for fact in facts if fact.recorded_at <= now and fact.fact_key not in removed
    ]


@dataclass(frozen=True, slots=True)
class NoteFile:
    """One vault file as the sidecar indexes it."""

    content: str
    mtime: int
    frontmatter: dict[str, object]


def notes_known_at(notes: Sequence[VaultNote], now: datetime) -> dict[str, NoteFile]:
    """The vault files that exist at `now`, one per path, each fixture heading a `##` block."""
    by_path: dict[str, list[VaultNote]] = {}
    for note in notes:
        if note.created_at <= now:
            by_path.setdefault(note.path, []).append(note)
    files: dict[str, NoteFile] = {}
    for path, sections in by_path.items():
        first = sections[0]
        body = "\n\n".join(f"## {note.heading}\n\n{note.body}" for note in sections)
        updated = max(note.updated_at for note in sections)
        files[path] = NoteFile(
            content=f"# {first.title}\n\n{body}\n",
            mtime=int(updated.timestamp()),
            frontmatter={
                "authored_by": first.authored_by,
                "last_updated": updated.date().isoformat(),
                "tags": list(first.tags),
            },
        )
    return files


# -- forced rescans --------------------------------------------------------------------------


def synthetic_facts(template: FactRow, count: int, revision: int) -> list[FactRow]:
    """`count` facts cloned from `template`, each its own fact with its own subject.

    `revision` is part of each object name, so a snapshot at a new revision changes the text,
    and so the hash, of every one of them: `diff_apply` re-embeds all `count`. Distinct
    subjects and utterances keep them out of each other's sibling lists.
    """
    return [
        replace(
            template,
            row_key=f"rescan-{n:04d}",
            fact_key=f"rescan-{n:04d}",
            subject_id=f"rescan-subject-{n:04d}",
            subject_name=f"Rescan Subject {n:04d}",
            object_id=f"rescan-object-{n:04d}",
            object_name=f"rescan object {n:04d} revision {revision}",
            valid_to=None,
            source_utterance_id=f"rescan-utt-{n:04d}",
        )
        for n in range(count)
    ]


class ForcedRescan:
    """Forced full rescans of the fact index, back to back, for the whole run.

    `FactIndexRefresher` runs a full rescan as one `diff_apply` on the `memory-index`
    executor (fact_source.py, `FactIndexRefresher.refresh`); so does each cycle here, on the run's own executor,
    so the fetcher's index reads queue beside it as they would beside a refresh. A cycle
    applies the latest snapshot the run applied plus `count` synthetic facts at the next
    revision, which re-embeds `count` facts with the run's embedder. The run's own applies
    carry the synthetic facts at the revision last applied, so they never delete or re-embed
    them. One lock serializes the run's applies and the cycles, so a cycle never writes a
    snapshot older than the run's.

    The cycles start once the run has applied its first snapshot. The first cycle runs even
    when the run stops before it starts, so a log always shows the cost of one rescan; each
    cycle records when it began, relative to `start`, and how long it took.

    Args:
        count: Synthetic facts each cycle re-embeds.
        template: The fact row the synthetic facts are cloned from.
        clock: Monotonic seconds, for the cycle times.

    Raises:
        ValueError: `count` is not positive.
    """

    def __init__(
        self, count: int, template: FactRow, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        if count <= 0:
            raise ValueError(f"rescan facts must be positive, got {count}")
        self.count = count
        self._template = template
        self._clock = clock
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._index: FactIndex | None = None
        self._executor: Executor | None = None
        self._thread: threading.Thread | None = None
        self._latest: list[FactRow] | None = None
        self._revision = 0
        self._started = 0.0
        self._stopped: float | None = None
        self.cycles: list[dict[str, object]] = []

    def start(self, index: FactIndex, executor: Executor) -> None:
        """Start the rescan thread; its first cycle waits for the run's first `apply`."""
        self._index = index
        self._executor = executor
        self._started = self._clock()
        self._thread = threading.Thread(target=self._loop, name="memory-forced-rescan", daemon=True)
        self._thread.start()

    def apply(self, snapshot: Sequence[FactRow]) -> DiffReport:
        """The run's own apply: `snapshot` with the synthetic facts at the revision applied."""
        if self._index is None:
            raise RuntimeError("ForcedRescan.apply before start")
        rows = list(snapshot)
        with self._lock:
            self._latest = rows
            report = self._index.diff_apply(
                rows + synthetic_facts(self._template, self.count, self._revision)
            )
        self._ready.set()
        return report

    def stop(self) -> None:
        """Let the cycle in flight finish, then stop; the run calls it before its executors stop."""
        self._stop.set()
        self._ready.set()
        if self._thread is not None:
            self._thread.join()
        self._stopped = self._clock()

    def log(self) -> dict[str, object]:
        """The rescan log: every cycle and how many completed."""
        end = self._stopped if self._stopped is not None else self._clock()
        return {
            "rescan_facts": self.count,
            "run_seconds": round(end - self._started, 3),
            "cycles_completed": sum(1 for cycle in self.cycles if "error" not in cycle),
            "cycles": list(self.cycles),
        }

    def _cycle(self, revision: int) -> DiffReport:
        if self._index is None or self._latest is None:
            raise RuntimeError("ForcedRescan cycle before the run's first apply")
        with self._lock:
            report = self._index.diff_apply(
                self._latest + synthetic_facts(self._template, self.count, revision)
            )
            self._revision = revision
            return report

    def _loop(self) -> None:
        self._ready.wait()
        if self._latest is None or self._executor is None:
            return  # stopped before the run applied anything
        while True:
            revision = self._revision + 1
            began = self._clock()
            cycle: dict[str, object] = {
                "version": revision,
                "began_s": round(began - self._started, 3),
            }
            try:
                report = self._executor.submit(self._cycle, revision).result()
            except (MistError, ValueError, RuntimeError, OSError, CancelledError) as exc:
                # Recorded rather than raised, so the log shows why the rescans stopped.
                cycle["error"] = f"{type(exc).__name__}: {exc}"
                self.cycles.append(cycle)
                return
            cycle["seconds"] = round(self._clock() - began, 3)
            cycle["report"] = asdict(report)
            self.cycles.append(cycle)
            if self._stop.is_set():
                return


def rescan_template(fixture: Fixture) -> FactRow:
    """The earliest recorded fixture fact, so its clones are known at every probe's `now`."""
    return fact_row(min(fixture.facts, key=lambda fact: fact.recorded_at))


# -- the run ---------------------------------------------------------------------------------


def _item_ranks(result: MemoryResult) -> dict[int, int]:
    """Each block position's fetch rank: the lowest `MemoryCandidate.rank` injected there.

    A list's scored members are all injected at the list's position, so the list takes its
    best member's rank.
    """
    ranks: dict[int, int] = {}
    for candidate in result.stage.candidates:
        if candidate.disposition == "injected" and candidate.item is not None:
            ranks[candidate.item] = min(candidate.rank, ranks.get(candidate.item, candidate.rank))
    return ranks


def run_items(result: MemoryResult, lists: Sequence[PlacedList]) -> list[dict[str, object]]:
    """The block as the scorer reads it, each item with its fetch rank (plan v4, A26).

    A list's facts are each written under the list's ref, with the list's rank. A note is
    named by its path, the fixture's note id; a second chunk of a file already in the block is
    left out, since the scorer counts an item once, and the note keeps the best rank of its
    chunks. The rank is the fetcher's own, not the block position, because the block lays
    items out by section and the scorer's MRR reads the rank.

    Raises:
        ValueError: A block item has no injected candidate in the stage, so it has no rank
            and the scorer would refuse the record.
    """
    by_ref = {placed.ref: placed for placed in lists}
    ranks = _item_ranks(result)
    first: dict[tuple[str, str], tuple[str, float]] = {}
    best: dict[tuple[str, str], int] = {}

    def add(ref: str, kind: str, item_id: str, score: float, rank: int) -> None:
        key = (kind, item_id)
        first.setdefault(key, (ref, score))
        best[key] = min(rank, best.get(key, rank))

    for position, item in enumerate(result.items, start=1):
        rank = ranks.get(position)
        if rank is None:
            raise ValueError(
                f"block item {item.ref} has no injected candidate in the stage, so no rank"
            )
        placed = by_ref.get(item.ref)
        if placed is not None:
            for fact in placed.facts.facts:
                add(item.ref, "fact", fact.fact_key, item.score, rank)
        elif isinstance(item.source, NoteSource):
            add(item.ref, "note", item.source.path, item.score, rank)
        else:
            add(item.ref, item.kind, item.item_id, item.score, rank)
    return [
        {"ref": ref, "kind": key[0], "id": key[1], "score": round(score, 6), "rank": best[key]}
        for key, (ref, score) in first.items()
    ]


def run_record(
    probe: Probe,
    result: MemoryResult,
    lists: Sequence[PlacedList],
    *,
    counter: ApproximateTokenCounter,
    cpu_load_procs: int,
) -> dict[str, object]:
    """One run-file record: the scorer's keys and the per-probe measurements."""
    timings = result.timings
    return {
        "probe_id": probe.probe_id,
        "items": run_items(result, lists),
        "tokens": result.tokens,
        "budget_tokens": result.budget_tokens,
        "critical_path_ms": round(timings.total_ms, 3),
        "timings_ms": {
            "stage1": round(timings.stage1_ms, 3),
            "stage2": round(timings.stage2_ms, 3),
            "rerank": round(timings.rerank_ms, 3),
            "queue_wait": round(timings.queue_wait_ms, 3),
        },
        "reranked": result.reranked,
        # The stage records one candidate per scored shortlist entry, capped, and counts the rest.
        "shortlisted": len(result.stage.candidates) + result.stage.truncated,
        "degraded_reason": result.degraded_reason,
        "delta_tokens_est": counter.count(result.block_text or "") + counter.count(probe.query),
        "cpu_load_procs": cpu_load_procs,
    }


@dataclass(frozen=True, slots=True)
class Fixture:
    """The fixture rows the run reads."""

    probes: tuple[Probe, ...]
    facts: tuple[Fact, ...]
    notes: tuple[VaultNote, ...]
    corrections: tuple[Correction, ...]

    @classmethod
    def load(cls, directory: Path = FIXTURE_DIR) -> Fixture:
        """Every file the run needs from `directory`."""
        return cls(
            probes=load_probes(directory / "probes.jsonl"),
            facts=load_facts(directory / "facts.jsonl"),
            notes=load_vault_notes(directory / "vault_notes.jsonl"),
            corrections=load_corrections(directory / "corrections.jsonl"),
        )


# -- the trace -------------------------------------------------------------------------------

# A probability is clamped this far inside (0, 1) before its logit, so a saturated score (the
# fake reranker gives 0 and 1) still maps to a finite number.
_LOGIT_EPS = 1e-7


def logit(p: float) -> float:
    """`log(p / (1 - p))`, the inverse of the reranker's sigmoid, clamped to stay finite."""
    clamped = min(1.0 - _LOGIT_EPS, max(_LOGIT_EPS, p))
    return math.log(clamped / (1.0 - clamped))


def _ms(started: float | None, ended: float | None) -> float | None:
    if started is None or ended is None:
        return None
    return round((ended - started) * 1000.0, 3)


@dataclass(slots=True)
class _EmbedCall:
    started: float
    ended: float | None = None


class RecordingEmbedder:
    """The run's embedder with each query embedding stamped, for the fetcher alone.

    Inside the timed fetch it takes two `perf_counter` readings per call and nothing else;
    the fact index and the sidecar keep the raw embedder, so their writes are not recorded.

    Args:
        inner: The embedder the run built.
    """

    def __init__(self, inner: TokenEmbedder | SentenceTransformerEmbedder) -> None:
        self._inner = inner
        self._calls: list[_EmbedCall] = []

    def generate_embedding(self, text: str) -> list[float]:
        """The inner embedder's vector, with the call's start and end stamped."""
        call = _EmbedCall(started=time.perf_counter())
        self._calls.append(call)
        try:
            return self._inner.generate_embedding(text)
        finally:
            call.ended = time.perf_counter()

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """The inner embedder's vectors, unrecorded: the fetcher embeds one query at a time."""
        return self._inner.generate_embeddings(texts)

    def take(self) -> list[_EmbedCall]:
        """The calls since the last `take`."""
        calls, self._calls = self._calls, []
        return calls


@dataclass(slots=True)
class _RerankCall:
    budget_ms: float | None
    answer: int | None
    query: str | None = None
    texts: Sequence[str] | None = None
    scores: Sequence[float] | None = None
    started: float | None = None
    ended: float | None = None


class RecordingReranker:
    """The run's reranker with each budget question and each scored batch kept, for the fetcher.

    Inside the timed fetch it keeps references to the arguments and results and takes two
    `perf_counter` readings per `score`; the tokenizing happens in the trace, afterwards.

    Args:
        inner: The reranker the run built; the trace's oracle scores with it directly.
    """

    def __init__(self, inner: Reranker) -> None:
        self._inner = inner
        self._calls: list[_RerankCall] = []

    def pairs_within(self, budget_ms: float) -> int:
        """The inner reranker's answer, kept with the budget it was asked about."""
        answer = self._inner.pairs_within(budget_ms)
        self._calls.append(_RerankCall(budget_ms=budget_ms, answer=answer))
        return answer

    def score(
        self, query: str, texts: Sequence[str], budget_ms: float | None = None
    ) -> list[float]:
        """The inner reranker's scores, kept with the texts and the call's start and end.

        The fetcher asks `pairs_within` first, so the batch joins that question's record.
        """
        last = self._calls[-1] if self._calls else None
        if last is None or last.texts is not None:
            last = _RerankCall(budget_ms=budget_ms, answer=None)
            self._calls.append(last)
        last.query = query
        last.texts = texts
        last.started = time.perf_counter()
        try:
            scores = self._inner.score(query, texts, budget_ms)
            last.scores = scores
            return scores
        finally:
            last.ended = time.perf_counter()

    def take(self) -> list[_RerankCall]:
        """The calls since the last `take`."""
        calls, self._calls = self._calls, []
        return calls


@dataclass(frozen=True, slots=True)
class TraceSetup:
    """Where the trace goes, its run header, and how it counts a pair's tokens."""

    path: Path
    header: dict[str, object]
    pair_tokens: PairTokens


def _jsonable(value: object) -> object:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def trace_header(
    args: argparse.Namespace, config: MemoryConfig, built: BuiltReranker
) -> dict[str, object]:
    """The trace's first record: what the run measured with.

    `threads` is torch's thread count as the run left it; the reranker's cost model is the
    one its warm-up measured, None for the fake.
    """
    reranker = built.reranker
    return {
        "record": "run",
        "device": args.device,
        "threads": torch.get_num_threads(),
        "embedder": args.embedder,
        "embedder_path": str(EMBEDDER_PATH) if args.embedder == "real" else None,
        "reranker": args.reranker,
        "reranker_path": built.path,
        "cost_per_pair_ms": getattr(reranker, "cost_per_pair_ms", None),
        "call_overhead_ms": getattr(reranker, "call_overhead_ms", None),
        "config": {f.name: _jsonable(getattr(config, f.name)) for f in fields(config)},
        "placement": args.placement,
        "rescan_facts": args.rescan_facts or 0,
        "cpu_load_procs": args.cpu_load_procs,
    }


@dataclass(frozen=True, slots=True)
class _Searched:
    """The unbounded searches of one probe, run after its fetch."""

    fact_ranks: dict[str, dict[str, int]]
    versions: dict[str, tuple[IndexedFact, ...]]
    note_ranks: dict[str, dict[str, int]]
    chunks: dict[str, list[tuple[str, str]]]


def _idle() -> None:
    """Nothing; run on an executor, it returns once the work queued before it has finished."""


def _current(versions: Sequence[IndexedFact], now: datetime) -> IndexedFact | None:
    """The version current at `now`, chosen as `FactGroup.current_at` chooses."""
    current = [version for version in versions if version.is_current_at(now)]
    if not current:
        return None
    return max(current, key=lambda version: (version.recorded_at or "", version.row_key))


class ProbeTracer:
    """Writes the trace: the run header, then one record per probe, after its fetch returns.

    Nothing here runs inside a timed fetch. After a fetch the tracer first waits for the
    `memory-model` executor to drain, so a rerank the fetch stopped waiting for has finished
    and its record belongs to this probe; then it tokenizes, runs the unbounded searches on
    the `memory-index` executor, as the fetcher's searches run, and scores the targets with
    the raw reranker, with no deadline, on the `memory-model` executor.

    Candidate texts are not written, except the oracle texts of the probe's targets.

    Args:
        setup: The trace path, header and pair counter.
        index: The run's fact index.
        sidecar: The run's vault sidecar.
        embedder: The raw embedder, for the unbounded searches' query vector.
        reranker: The raw reranker, for the oracle.
        recorded_embedder: The fetcher's recording embedder.
        recorded_reranker: The fetcher's recording reranker.
        executors: The run's executors.
        config: The run's memory settings.
    """

    def __init__(
        self,
        setup: TraceSetup,
        *,
        index: FactIndex,
        sidecar: VaultSidecarIndex,
        embedder: TokenEmbedder | SentenceTransformerEmbedder,
        reranker: Reranker,
        recorded_embedder: RecordingEmbedder,
        recorded_reranker: RecordingReranker,
        executors: FetcherExecutors,
        config: MemoryConfig,
    ) -> None:
        self._setup = setup
        self._index = index
        self._sidecar = sidecar
        self._embedder = embedder
        self._reranker = reranker
        self._recorded_embedder = recorded_embedder
        self._recorded_reranker = recorded_reranker
        self._executors = executors
        self._top_n = config.rerank_top_n
        setup.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = setup.path.open("w", encoding="utf-8")
        self._write(setup.header)

    def close(self) -> None:
        """Close the trace file."""
        self._handle.close()

    async def trace(
        self,
        probe: Probe,
        outcome: FetchOutcome,
        *,
        indexed_notes: set[str],
    ) -> dict[str, object]:
        """Build and write the probe's trace record; returns it."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executors.model, _idle)
        embeds = self._recorded_embedder.take()
        reranks = self._recorded_reranker.take()
        query = probe.query
        token_cache: dict[str, int] = {}

        def tokens(text: str) -> int:
            if text not in token_cache:
                token_cache[text] = self._setup.pair_tokens(query, text)
            return token_cache[text]

        result = outcome.result
        fused = outcome.fused
        shortlist = fused[: self._top_n]
        by_rank = {candidate.rank: candidate for candidate in result.stage.candidates}
        fused_at = {(f.candidate.kind, f.candidate.item_id): n for n, f in enumerate(fused, 1)}
        ranked_at = {
            (s.candidate.kind, s.candidate.item_id): n for n, s in enumerate(outcome.ranked, 1)
        }

        def disposition(rank: int | None) -> str | None:
            stage = by_rank.get(rank) if rank is not None else None
            return stage.disposition if stage is not None else None

        fused_entries = [
            {
                "fused": n,
                "kind": f.candidate.kind,
                "item_id": f.candidate.item_id,
                "path": (
                    f.candidate.source.path if isinstance(f.candidate.source, NoteSource) else None
                ),
                "pools": dict(f.candidate.ranks),
                "rrf": f.rrf,
                "rrf_norm": f.rrf_norm,
                "text_chars": len(f.candidate.text),
                "pair_tokens": tokens(f.candidate.text),
                "shortlist": n if n <= self._top_n else None,
            }
            for n, f in enumerate(fused, 1)
        ]
        scored_entries = [
            {
                "rank": n,
                "kind": s.candidate.kind,
                "item_id": s.candidate.item_id,
                "reranked": s.reranked,
                "relevance": s.relevance,
                "logit": logit(s.relevance) if s.reranked else None,
                "time_term": s.time_term,
                "window_applied": s.window_applied,
                "confidence": s.confidence,
                "score": s.score,
                "disposition": disposition(n),
                "item": by_rank[n].item if n in by_rank else None,
            }
            for n, s in enumerate(outcome.ranked, 1)
        ]
        rerank_entries: list[dict[str, object]] = []
        matches = True
        for call in reranks:
            texts = list(call.texts) if call.texts is not None else []
            if call.texts is not None:
                head = [f.candidate.text for f in shortlist[: len(texts)]]
                matches = matches and texts == head
            scored = len(texts) if call.scores is not None else 0
            rerank_entries.append(
                {
                    "budget_ms": round(call.budget_ms, 3) if call.budget_ms is not None else None,
                    "pairs_within": call.answer,
                    "pairs_scored": scored,
                    "returned": call.scores is not None,
                    "wall_ms": _ms(call.started, call.ended),
                    "discarded": scored if result.reranked == 0 else 0,
                    "pair_tokens": [tokens(text) for text in texts],
                }
            )

        searched = await self._search(probe)
        block_ids: dict[str, set[str]] = {}
        for item in run_items(result, outcome.lists):
            block_ids.setdefault(str(item["kind"]), set()).add(str(item["id"]))
        targets = self._targets(
            probe,
            searched,
            fused,
            fused_at,
            ranked_at,
            disposition,
            block_ids,
            indexed_notes,
        )
        await self._oracle(query, targets)

        trace: dict[str, object] = {
            "record": "probe",
            "probe_id": probe.probe_id,
            "kind": probe.kind,
            "lexical": probe.lexical,
            "shape": probe.shape,
            "now": probe.now.isoformat(),
            "degraded_reason": result.degraded_reason,
            "reranked": result.reranked,
            "shortlisted": len(outcome.ranked),
            "fused": fused_entries,
            "scored": scored_entries,
            "rerank_calls": rerank_entries,
            "rerank_matches_shortlist": matches,
            "embed_calls": [{"wall_ms": _ms(call.started, call.ended)} for call in embeds],
            "targets": [target for target, _ in targets],
        }
        self._write(trace)
        return trace

    async def _search(self, probe: Probe) -> _Searched:
        """The probe's searches unbounded: every indexed fact, every note chunk."""
        fact_keys = [ref.id for ref in (*probe.expected, *probe.forbidden) if ref.kind == "fact"]
        want_notes = any(ref.kind == "note" for ref in (*probe.expected, *probe.forbidden))
        if not fact_keys and not want_notes:
            return _Searched(fact_ranks={}, versions={}, note_ranks={}, chunks={})
        loop = asyncio.get_running_loop()
        query = probe.query
        vector = await loop.run_in_executor(
            self._executors.model, self._embedder.generate_embedding, query
        )

        def work() -> _Searched:
            fact_ranks: dict[str, dict[str, int]] = {}
            versions: dict[str, tuple[IndexedFact, ...]] = {}
            if fact_keys:
                every = self._index.count()
                if every > 0:
                    pools = (
                        (FTS_POOL, self._index.search_fts(query, limit=every)),
                        (VECTOR_POOL, self._index.search_vector(vector, limit=every)),
                        (ENTITY_POOL, self._index.search_names([query], limit=every)),
                    )
                    for pool, groups in pools:
                        fact_ranks[pool] = {g.fact_key: n for n, g in enumerate(groups, 1)}
                versions = {key: self._index.versions(key) for key in fact_keys}
            note_ranks: dict[str, dict[str, int]] = {}
            chunks: dict[str, list[tuple[str, str]]] = {}
            if want_notes:
                every = self._sidecar.chunk_count()
                if every > 0:
                    lexical = " ".join(query_tokens(query))
                    rows = {
                        NOTE_VECTOR_POOL: self._sidecar.query_vector(vector, k=every),
                        NOTE_FTS_POOL: self._sidecar.query_fts(lexical, k=every, match="any"),
                    }
                    for pool, found in rows.items():
                        best: dict[str, int] = {}
                        for n, row in enumerate(found, 1):
                            best.setdefault(str(row["path"]), n)
                        note_ranks[pool] = best
                    # The vector search returns every chunk, so it lists each note's chunks.
                    for row in rows[NOTE_VECTOR_POOL]:
                        content = str(row.get("content") or "")
                        if not content.strip():
                            continue
                        heading = row.get("heading")
                        chunks.setdefault(str(row["path"]), []).append(
                            (str(heading) if heading is not None else FILE_HEADING, content)
                        )
            return _Searched(
                fact_ranks=fact_ranks, versions=versions, note_ranks=note_ranks, chunks=chunks
            )

        return await loop.run_in_executor(self._executors.index, work)

    def _targets(
        self,
        probe: Probe,
        searched: _Searched,
        fused: Sequence[FusedCandidate],
        fused_at: dict[tuple[str, str], int],
        ranked_at: dict[tuple[str, str], int],
        disposition: Callable[[int | None], str | None],
        block_ids: dict[str, set[str]],
        indexed_notes: set[str],
    ) -> list[tuple[dict[str, object], list[dict[str, object]]]]:
        """Each expected and forbidden item, with the oracle entries still to be scored."""
        note_keys: dict[str, list[tuple[str, str]]] = {}
        for f in fused:
            if isinstance(f.candidate.source, NoteSource):
                note_keys.setdefault(f.candidate.source.path, []).append(
                    (f.candidate.kind, f.candidate.item_id)
                )
        targets: list[tuple[dict[str, object], list[dict[str, object]]]] = []
        for role, refs in (("expected", probe.expected), ("forbidden", probe.forbidden)):
            for ref in refs:
                base: dict[str, object] = {"kind": ref.kind, "id": ref.id, "role": role}
                if ref.kind == "fact":
                    targets.append(
                        self._fact_target(
                            base, probe, searched, fused_at, ranked_at, disposition, block_ids
                        )
                    )
                elif ref.kind == "note":
                    targets.append(
                        self._note_target(
                            base,
                            searched,
                            note_keys.get(ref.id, []),
                            fused_at,
                            ranked_at,
                            disposition,
                            block_ids,
                            indexed_notes,
                        )
                    )
                else:
                    targets.append(({**base, "traced": False}, []))
        return targets

    def _fact_target(
        self,
        base: dict[str, object],
        probe: Probe,
        searched: _Searched,
        fused_at: dict[tuple[str, str], int],
        ranked_at: dict[tuple[str, str], int],
        disposition: Callable[[int | None], str | None],
        block_ids: dict[str, set[str]],
    ) -> tuple[dict[str, object], list[dict[str, object]]]:
        key = str(base["id"])
        versions = searched.versions.get(key, ())
        current = _current(versions, probe.now)
        fused = fused_at.get(("fact", key))
        rank = ranked_at.get(("fact", key))
        oracle: dict[str, object] | None = (
            {"text": current.text, "logit": None} if current is not None else None
        )
        target = {
            **base,
            "traced": True,
            "indexed": bool(versions),
            "indexed_at_now": current is not None,
            "ranks": {pool: searched.fact_ranks.get(pool, {}).get(key) for pool in FACT_POOLS},
            "fused": fused,
            "shortlist": fused if fused is not None and fused <= self._top_n else None,
            "rank": rank,
            "disposition": disposition(rank),
            "in_block": key in block_ids.get("fact", set()),
            "oracle": oracle,
        }
        return target, [oracle] if oracle is not None else []

    def _note_target(
        self,
        base: dict[str, object],
        searched: _Searched,
        keys: list[tuple[str, str]],
        fused_at: dict[tuple[str, str], int],
        ranked_at: dict[tuple[str, str], int],
        disposition: Callable[[int | None], str | None],
        block_ids: dict[str, set[str]],
        indexed_notes: set[str],
    ) -> tuple[dict[str, object], list[dict[str, object]]]:
        path = str(base["id"])
        fused_positions = [fused_at[key] for key in keys if key in fused_at]
        ranks = sorted(ranked_at[key] for key in keys if key in ranked_at)
        dispositions = [disposition(rank) for rank in ranks]
        best_fused = min(fused_positions) if fused_positions else None
        chunks: list[dict[str, object]] = []
        for heading, text in searched.chunks.get(path, []):
            item_id = note_item_id(path, None if heading == FILE_HEADING else heading)
            chunk_rank = ranked_at.get(("note", item_id))
            chunks.append(
                {
                    "heading": heading,
                    "item_id": item_id,
                    "fused": fused_at.get(("note", item_id)),
                    "rank": chunk_rank,
                    "disposition": disposition(chunk_rank),
                    "text": text,
                    "logit": None,
                }
            )
        if "injected" in dispositions:
            chosen: str | None = "injected"
        else:
            chosen = dispositions[0] if dispositions else None
        oracle: dict[str, object] | None = {"chunks": chunks, "max": None} if chunks else None
        target = {
            **base,
            "traced": True,
            "indexed": path in indexed_notes,
            "indexed_at_now": path in indexed_notes,
            "ranks": {
                pool: searched.note_ranks.get(pool, {}).get(path)
                for pool in (NOTE_VECTOR_POOL, NOTE_FTS_POOL)
            },
            "fused": best_fused,
            "shortlist": (
                best_fused if best_fused is not None and best_fused <= self._top_n else None
            ),
            "rank": ranks[0] if ranks else None,
            "disposition": chosen,
            "in_block": path in block_ids.get("note", set()),
            "oracle": oracle,
        }
        return target, chunks

    async def _oracle(
        self, query: str, targets: list[tuple[dict[str, object], list[dict[str, object]]]]
    ) -> None:
        """Score every target text with the raw reranker in one batch, with no deadline."""
        entries = [entry for _, pending in targets for entry in pending]
        if not entries:
            return
        texts = [str(entry["text"]) for entry in entries]
        loop = asyncio.get_running_loop()
        scores = await loop.run_in_executor(
            self._executors.model, self._reranker.score, query, texts
        )
        logits = [logit(float(score)) for score in scores]
        for entry, value in zip(entries, logits, strict=True):
            entry["logit"] = value
        offset = 0
        for target, pending in targets:
            values = logits[offset : offset + len(pending)]
            offset += len(pending)
            oracle = target.get("oracle")
            if target["kind"] == "note" and isinstance(oracle, dict) and values:
                oracle["max"] = max(values)

    def _write(self, record: dict[str, object]) -> None:
        self._handle.write(json.dumps(record) + "\n")
        self._handle.flush()


async def run_fixture(
    fixture: Fixture,
    *,
    config: MemoryConfig,
    embedder: TokenEmbedder | SentenceTransformerEmbedder,
    reranker: Reranker,
    workdir: Path,
    probes: Sequence[Probe] | None = None,
    cpu_load_procs: int = 0,
    placement: BlockPlacement = "user_prefix",
    rescan: ForcedRescan | None = None,
    trace: TraceSetup | None = None,
) -> list[dict[str, object]]:
    """Run each probe through a real fetcher over indexes built in `workdir`.

    Args:
        placement: The placement the block builder words the block for.
        rescan: Forced rescans to run on the run's `memory-index` executor for the whole
            run; started here and stopped before the executors shut down.
        trace: Write the per-probe candidate trace (`ProbeTracer`); the fetcher then gets
            recording wrappers of the embedder and the reranker.

    Returns:
        One record per probe, in the order of `probes` (all the fixture's by default).
    """
    chosen = tuple(probes if probes is not None else fixture.probes)
    counter = ApproximateTokenCounter()
    index_path = workdir / "memory_index.db"
    writer = open_index(index_path, embedding_model_id=EMBEDDING_MODEL_ID)
    reader = open_reader(index_path, embedding_model_id=EMBEDDING_MODEL_ID)
    probe_state = IndexStateProbe.open(str(index_path), embedding_model_id=EMBEDDING_MODEL_ID)
    sidecar = VaultSidecarIndex(
        SidecarIndexConfig(db_path=str(workdir / "vault_sidecar.db")), embedder
    )
    sidecar.initialize()
    executors = build_memory_executors()
    pools = FetcherExecutors(index=executors[INDEX_EXECUTOR], model=executors[MODEL_EXECUTOR])
    tracer: ProbeTracer | None = None
    try:
        index = FactIndex(writer, embedder, tz=config.tz, reader=reader)
        fetch_embedder: TokenEmbedder | SentenceTransformerEmbedder | RecordingEmbedder = embedder
        fetch_reranker: Reranker = reranker
        if trace is not None:
            # Only the fetcher records; the indexes embed with the raw embedder.
            recorded_embedder = RecordingEmbedder(embedder)
            recorded_reranker = RecordingReranker(reranker)
            fetch_embedder, fetch_reranker = recorded_embedder, recorded_reranker
            tracer = ProbeTracer(
                trace,
                index=index,
                sidecar=sidecar,
                embedder=embedder,
                reranker=reranker,
                recorded_embedder=recorded_embedder,
                recorded_reranker=recorded_reranker,
                executors=pools,
                config=config,
            )
        fetcher = MemoryFetcher(
            index=index,
            embedder=fetch_embedder,
            reranker=fetch_reranker,
            block_builder=MemoryBlockBuilder(counter=counter, config=config, placement=placement),
            config=config,
            executors=pools,
            clock=SystemClock(),
            index_state=probe_state,
            notes=VaultCandidates(sidecar, vault_root=str(workdir / "vault"), tz=config.tz),
        )
        apply = index.diff_apply
        if rescan is not None:
            rescan.start(index, executors[INDEX_EXECUTOR])
            apply = rescan.apply
        records: dict[str, dict[str, object]] = {}
        indexed_notes: set[str] = set()
        for now in sorted({probe.now for probe in chosen}):
            apply(facts_known_at(fixture.facts, fixture.corrections, now))
            for path, note in notes_known_at(fixture.notes, now).items():
                if path not in indexed_notes:
                    sidecar.upsert_file(path, note.content, note.mtime, note.frontmatter)
                    indexed_notes.add(path)
            for probe in (p for p in chosen if p.now == now):
                outcome = await fetcher.fetch_outcome(
                    MemoryRequest(
                        session_id=f"probe-{probe.probe_id}",
                        conversation_id=None,
                        query_text=probe.query,
                        now=probe.now,
                        recent_user_turns=(),
                        turn_index=0,
                        conversation_start=probe.conversation_start,
                    )
                )
                records[probe.probe_id] = run_record(
                    probe,
                    outcome.result,
                    outcome.lists,
                    counter=counter,
                    cpu_load_procs=cpu_load_procs,
                )
                if tracer is not None:
                    await tracer.trace(probe, outcome, indexed_notes=indexed_notes)
        return [records[probe.probe_id] for probe in chosen]
    finally:
        if rescan is not None:
            rescan.stop()
        if tracer is not None:
            tracer.close()
        for executor in executors.values():
            executor.shutdown(wait=True, cancel_futures=True)
        sidecar.close()
        probe_state.close()
        reader.close()
        writer.close()


# -- CPU load --------------------------------------------------------------------------------


class _Event(Protocol):
    def set(self) -> None: ...

    def is_set(self) -> bool: ...


class _Process(Protocol):
    def start(self) -> None: ...

    def join(self, timeout: float | None = None) -> None: ...

    def is_alive(self) -> bool: ...

    def terminate(self) -> None: ...


def spin(stop: _Event) -> None:
    """Keep one core busy until `stop` is set."""
    while not stop.is_set():
        for _ in range(_SPIN_BATCH):
            pass


def _default_context() -> multiprocessing.context.BaseContext:
    # fork where the platform has it: a spawned child would re-import this module and torch
    # before it could spin. The load starts before the run creates any thread.
    if "fork" in multiprocessing.get_all_start_methods():
        return multiprocessing.get_context("fork")
    return multiprocessing.get_context("spawn")


@contextlib.contextmanager
def cpu_load(
    procs: int,
    *,
    process_factory: Callable[..., _Process] | None = None,
    event_factory: Callable[[], _Event] | None = None,
    clock: Callable[[], float] = time.monotonic,
    join_timeout_s: float = _LOAD_JOIN_TIMEOUT_S,
) -> Iterator[list[_Process]]:
    """Run `procs` busy-loop processes for the body, then stop them.

    Stopping sets the shared event and joins each process within `join_timeout_s` in total,
    by `clock`; a process still alive then is terminated.

    Raises:
        ValueError: `procs` is negative.
    """
    if procs < 0:
        raise ValueError(f"cpu load processes must not be negative, got {procs}")
    if procs == 0:
        yield []
        return
    context = _default_context()
    make_process = process_factory or context.Process
    stop = (event_factory or context.Event)()
    started: list[_Process] = []
    try:
        for n in range(procs):
            process = make_process(target=spin, args=(stop,), name=f"memory-load-{n}", daemon=True)
            process.start()
            started.append(process)
        yield started
    finally:
        stop.set()
        deadline = clock() + join_timeout_s
        for process in started:
            process.join(timeout=max(0.0, deadline - clock()))
        for process in started:
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)


# -- the command -----------------------------------------------------------------------------


def config_from_args(args: argparse.Namespace) -> MemoryConfig:
    """MemoryConfig from the environment, with each value the command line sets.

    Raises:
        ValueError: `--rerank-model` is not under `/opt/models`; refused even with the fake
            reranker, so a run never records a model path it could not have loaded.
    """
    config = MemoryConfig.from_env()
    overrides: dict[str, object] = {}
    for name in ("block_tokens", "min_score", "rerank_top_n", "list_max_items", "deadline_ms"):
        value = getattr(args, name)
        if value is not None:
            overrides[name] = value
    if args.rerank_model is not None:
        overrides["rerank_model"] = _model_path(args.rerank_model)
    return replace(config, **overrides) if overrides else config


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {text}")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, required=True, help="run file to write (JSONL)")
    parser.add_argument("--embedder", choices=("fake", "real"), default="fake")
    parser.add_argument("--reranker", choices=("fake", "real"), default="fake")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--rerank-model", help="cross-encoder directory under /opt/models; overrides the config"
    )
    parser.add_argument(
        "--threads", type=_positive_int, help="torch intra-op threads, set before models load"
    )
    parser.add_argument(
        "--placement", choices=("user_prefix", "system_tail"), default="user_prefix"
    )
    parser.add_argument("--cpu-load-procs", type=int, default=0, help="busy-loop processes")
    parser.add_argument(
        "--rescan-facts",
        type=_positive_int,
        help="forced full rescans for the whole run, each re-embedding N synthetic facts",
    )
    parser.add_argument("--rescan-log", type=Path, help="rescan log to write (JSON)")
    parser.add_argument("--trace", type=Path, help="per-probe candidate trace to write (JSONL)")
    parser.add_argument("--block-tokens", type=int)
    parser.add_argument("--min-score", type=float)
    parser.add_argument("--rerank-top-n", type=int)
    parser.add_argument("--list-max-items", type=int)
    parser.add_argument("--deadline-ms", type=int)
    parser.add_argument("--fixture-dir", type=Path, default=FIXTURE_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fixture and write the run file; print the scorer's overall metrics."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.rescan_facts is not None and args.rescan_log is None:
        parser.error("--rescan-facts needs --rescan-log, the record that the rescans ran")
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    config = config_from_args(args)
    fixture = Fixture.load(args.fixture_dir)
    rescan = (
        ForcedRescan(args.rescan_facts, rescan_template(fixture))
        if args.rescan_facts is not None
        else None
    )
    with cpu_load(args.cpu_load_procs):
        embedder = build_embedder(args.embedder, device=args.device)
        built = build_reranker(args.reranker, device=args.device, config=config)
        trace = (
            TraceSetup(
                path=args.trace,
                header=trace_header(args, config, built),
                pair_tokens=built.pair_tokens,
            )
            if args.trace is not None
            else None
        )
        with tempfile.TemporaryDirectory(prefix="memory-recall-run-") as workdir:
            records = asyncio.run(
                run_fixture(
                    fixture,
                    config=config,
                    embedder=embedder,
                    reranker=built.reranker,
                    workdir=Path(workdir),
                    cpu_load_procs=args.cpu_load_procs,
                    placement=args.placement,
                    rescan=rescan,
                    trace=trace,
                )
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    if rescan is not None:
        args.rescan_log.parent.mkdir(parents=True, exist_ok=True)
        args.rescan_log.write_text(json.dumps(rescan.log(), indent=2) + "\n", encoding="utf-8")
    report = score_run(fixture.probes, [parse_run_record(record) for record in records])
    print(json.dumps(report.overall.to_json(), indent=2, sort_keys=True))
    print(f"wrote {len(records)} records to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "EMBEDDER_PATH",
    "BuiltReranker",
    "CrossEncoderCapture",
    "Fixture",
    "ForcedRescan",
    "NoteFile",
    "ProbeTracer",
    "RecordingEmbedder",
    "RecordingReranker",
    "SentenceTransformerEmbedder",
    "TokenEmbedder",
    "TokenReranker",
    "TraceSetup",
    "build_embedder",
    "build_reranker",
    "content_stems",
    "cpu_load",
    "fact_row",
    "facts_known_at",
    "logit",
    "main",
    "notes_known_at",
    "rescan_template",
    "run_fixture",
    "run_items",
    "run_record",
    "spin",
    "synthetic_facts",
    "trace_header",
    "word_pair_tokens",
]
