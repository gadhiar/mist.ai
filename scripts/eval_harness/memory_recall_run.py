"""Run the memory fetcher over the recall fixture and write the run file the scorer reads.

The harness builds the read path's indexes from `tests/fixtures/memory_recall/`: the fact index
from `facts.jsonl` and a vault sidecar from `vault_notes.jsonl`. It then runs one
`MemoryFetcher.fetch` per probe, as a new session with no earlier turns, and writes one JSONL
record per probe in the shape `memory_recall.py --run` scores.

The indexes hold what MIST would have known at each probe's `now`: a fact once it is recorded,
a note once it is created. A correction does not take its fact out of the index: the walk
records each fixture correction through `CorrectionService`, at its `recorded_at`, in an event
store of its own in the work directory, and each fetcher suppresses at read time what that log
names (`corrections.LoggedSuppression`), the read path T33 and T34 wire into the backend; a
recorded listener drops each fetcher's cached stage 1. Probes run in `now` order and the
indexes and the log are brought to each `now` before its probes run. Within one `now` every
probe sees the same indexes and log, and the fetcher filters each fact's versions against that
`now`.

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

`--probes PATH` picks the probe file, by default the fixture directory's `probes.jsonl`; the
held-out runs pass `heldout_probes.jsonl`. The facts, notes and corrections stay the fixture
directory's.

`--device` places the reranker and `--embedder-device` the embedder, by default on `--device`;
each record names the embedder's device as `embedder_device`.

`--cpu-load-procs N` runs N busy-loop processes for the whole run, to measure the critical path
with the CPU contended the way a speaking turn contends it. `--gpu-load-procs N` runs N
processes each looping a matmul on `cuda` for the whole run, to measure it with the GPU
contended; it is refused with exit 2 when CUDA is unavailable.

`--vram-log PATH` writes the process's GPU memory, in MiB, at named points (`VramRecorder`):
`before_cuda`, NVML's device-wide used memory before anything in the process touches CUDA;
`context`, once the process holds its CUDA context; `stt`, with `--vram-stt`; `loaded`, once
the reranker loads; `after_run`, after the last probe; and `released`, after the reranker's
`release()` and the Whisper model's release, as gaming mode releases both. From `context` on
each point also carries `mem_get_info`'s used memory and the caching allocator's four counts.
The derived figures are differences of NVML used memory: `process_mib` (after_run less
before_cuda, the gated figure), `context_mib`, `stt_mib`, `marginal_mib` (the reranker's own
share) and `residual_mib` (what a gaming release leaves). The log needs CUDA and NVML, and is
refused with exit 2 when the embedder would be on CUDA or with `--gpu-load-procs`, since either
would share the device-wide figure. `--vram-stt MODEL` loads that Whisper model on `cuda` before
the reranker and transcribes one fixed 30 s clip with `fp16=False`, as the speech path does, so
the process holds what the backend holds after a voice turn; it needs `--vram-log`. NVML,
torch's CUDA calls, the Whisper loader and the GPU load come through `GpuAccess`, which tests
replace.

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

`--fact-view` sets the fact view the index embeds, overriding the config's.

The need gate: `--need-gate on|shadow|off` overrides the config's mode, which without the flag
is `MIST_MEMORY_NEED_GATE`'s, `on` by default. `--need-embedder shared` (the default) builds the
gate on the fact embedder, sharing the fetch's query vector; its name and revision for the
gate's key are the fact embedder's entry in `--models-lock` with `--embedder real`, and
`FAKE_EMBEDDER_NAME` and `FAKE_EMBEDDER_REVISION` with `--embedder fake`. `--need-embedder PATH`
loads a gate embedder from a directory under `/opt/models` on the CPU, as
`EmbeddingGenerator(PATH, device="cpu", **need_embedder_kwargs(spec))`, and needs
`--need-embedder-name` (a `NEED_EMBEDDERS` name, whose spec gives the query prefix and the
attention setting) and `--need-embedder-revision`; both are refused with `shared`. The gate's
model is the record for its key in `--need-model`, by default the shipped need_model.json; with
the gate on or shadow and no such record the run exits 2 naming `--need-model`. With the gate
off neither the need bank nor the model file is read. `--need-context on|off` overrides the
config's `need_context`. Each record then carries `need`, the fetch's `NeedTrace` fields (None
with the gate off), and `stage_ids`; a shadow record whose fetch built a block also carries
`cuts`, the blocks a band turn would be served (`outcome_record`).

`run_settings` runs several settings, each a config and its gate, over one walk of the fixture,
so they share the index and its updates; the harness's command line runs one.

Usage, from the repository root (`/app` in the image), so `backend` imports:
    python -m scripts.eval_harness.memory_recall_run --output run.jsonl
        [--embedder fake|real] [--reranker fake|real] [--device cpu|cuda]
        [--embedder-device cpu|cuda] [--rerank-model PATH] [--threads N]
        [--placement user_prefix|system_tail] [--cpu-load-procs N] [--gpu-load-procs N]
        [--vram-log FILE [--vram-stt MODEL]] [--rescan-facts N --rescan-log FILE]
        [--trace FILE] [--block-tokens N] [--rerank-top-n N] [--select-k N]
        [--facets on|off] [--list-max-items N] [--deadline-ms N] [--fixture-dir DIR]
        [--probes FILE] [--fact-view F0|F1|F2|F3] [--need-gate on|shadow|off]
        [--need-embedder shared|PATH [--need-embedder-name NAME --need-embedder-revision SHA]]
        [--models-lock FILE] [--need-model FILE] [--need-context on|off]
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import gc
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

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from backend.chat.context_budget import ApproximateTokenCounter
from backend.errors import MemoryConfigError, MistError
from backend.event_store.store import EventStore
from backend.interfaces import EmbeddingProvider
from backend.knowledge.config import SidecarIndexConfig
from backend.knowledge.embeddings.embedding_generator import EmbeddingGenerator
from backend.memory.block import BlockPlacement, MemoryBlockBuilder, PlacedList
from backend.memory.config import MemoryConfig
from backend.memory.corrections import (
    CorrectedFact,
    CorrectionRecord,
    CorrectionService,
    CorrectionSurface,
    LoggedSuppression,
)
from backend.memory.fact_index import DiffReport, FactIndex, IndexedFact
from backend.memory.fetcher import (
    GATE_EXECUTOR,
    INDEX_EXECUTOR,
    MODEL_EXECUTOR,
    FetcherExecutors,
    FetchOutcome,
    MemoryFetcher,
    SystemClock,
)
from backend.memory.fts import query_tokens
from backend.memory.index_db import EMBEDDING_DIMENSION, open_index, open_reader
from backend.memory.models import (
    FactRow,
    FetchTimings,
    MemoryRequest,
    MemoryResult,
    NeedTrace,
    NoteSource,
)
from backend.memory.need_gate import (
    NEED_EMBEDDERS,
    NeedBank,
    NeedEmbedderSpec,
    NeedGate,
    NeedKey,
    NeedModel,
    load_bank,
    load_models,
    need_embedder_kwargs,
    select_model,
)
from backend.memory.ranking import FusedCandidate, ScoredCandidate
from backend.memory.rerank import CrossEncoderReranker, Reranker
from backend.memory.retrieval_view import FACT_VIEWS
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
from scripts.check_models_lock import LockError, parse_lock
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
# The lock the image copies beside the models (Dockerfile, the model layer's COPY).
DEFAULT_MODELS_LOCK = MODELS_ROOT / "models.lock"

# The fake embedder's name and revision in a shared need gate's key. `TokenEmbedder` has no
# weights, so its revision names the hashing it embeds by; change it with the vectors, so a gate
# model fit to the old vectors no longer matches the key.
FAKE_EMBEDDER_NAME = "token-embedder"
FAKE_EMBEDDER_REVISION = "content-stems-sha256-v1"

# What `run_settings`'s warm-up fetch asks.
WARM_UP_QUERY = "what do you remember about me"

ModelChoice = Literal["fake", "real"]
Device = Literal["cpu", "cuda"]

# How long stopping the load processes may take before the stragglers are terminated.
_LOAD_JOIN_TIMEOUT_S = 5.0
# Busy-loop iterations between checks of the stop event, so a check costs little of the load.
_SPIN_BATCH = 100_000
# The GPU load: square fp32 matrices of this side, and matmuls queued between synchronizes and
# checks of the stop event.
_MATMUL_SIZE = 2048
_MATMUL_BATCH = 16
# `--vram-stt`'s clip: 30 s at Whisper's 16 kHz, low-level noise from a fixed seed.
_STT_SAMPLE_RATE = 16_000
_STT_CLIP_S = 30
_STT_CLIP_SEED = 0
_STT_CLIP_LEVEL = 0.1
_MIB = 1024 * 1024


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

    def release(self) -> None:
        """Nothing: the fake holds no model memory, so `--vram-log` runs release it as a no-op."""

    def reload(self) -> bool:
        """True: there is no model to load again."""
        return True


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
    """The fact versions the graph holds at `now`: every one recorded by then.

    A corrected fact stays: the read path suppresses it through the correction log
    (`record_due_corrections`), as the backend does before T22 changes the graph. A corrected
    fact's replacement is its own fixture row, recorded at the correction. `corrections` is not
    read; it stays in the signature for the callers that pass the fixture's log, such as
    `memory_view_probe`.
    """
    del corrections  # suppressed at read time, not removed here
    return [fact_row(fact) for fact in facts if fact.recorded_at <= now]


def record_due_corrections(
    service: CorrectionService,
    corrections: Sequence[Correction],
    now: datetime,
    recorded: set[str],
) -> list[CorrectionRecord]:
    """Record each fixture correction due by `now` and not yet in `recorded`, in log order.

    Each is recorded at its own `recorded_at`, under its own id and origin, so the log holds
    what the backend's would. `recorded` gains each id recorded.

    Returns:
        The records made, in (`recorded_at`, `correction_id`) order.
    """
    due = sorted(
        (c for c in corrections if c.recorded_at <= now and c.correction_id not in recorded),
        key=lambda c: (c.recorded_at, c.correction_id),
    )
    records: list[CorrectionRecord] = []
    for correction in due:
        fact = CorrectedFact(
            subject_id=correction.subject_id,
            subject_name=correction.subject_name,
            relation=correction.relation,
            object_id=correction.object_id,
            object_name=correction.object_name,
        )
        surface = _SURFACES.get(correction.surface)
        if surface is None:
            raise ValueError(f"correction {correction.correction_id} has an unknown surface")
        if correction.action == "correct":
            if correction.replacement_object_name is None:
                raise ValueError(f"correction {correction.correction_id} has no replacement")
            record = service.correct(
                correction.fact_key,
                correction.replacement_object_name,
                surface,
                fact=fact,
                session_id=correction.session_id,
                conversation_id=correction.conversation_id,
                turn_index=correction.turn_index,
                source_event_id=correction.source_event_id,
                origin=correction.origin,
                recorded_at=correction.recorded_at,
                correction_id=correction.correction_id,
            )
        else:
            record = service.forget(
                correction.fact_key,
                surface,
                fact=fact,
                session_id=correction.session_id,
                conversation_id=correction.conversation_id,
                turn_index=correction.turn_index,
                source_event_id=correction.source_event_id,
                origin=correction.origin,
                recorded_at=correction.recorded_at,
                correction_id=correction.correction_id,
            )
        records.append(record)
        recorded.add(correction.correction_id)
    return records


# A fixture correction's surface as `CorrectionService` takes it.
_SURFACES: dict[str, CorrectionSurface] = {"ws": "ws", "tool": "tool", "cli": "cli"}


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
    earliest member's rank.
    """
    ranks: dict[int, int] = {}
    for candidate in result.stage.candidates:
        if candidate.disposition == "injected" and candidate.item is not None:
            ranks[candidate.item] = min(candidate.rank, ranks.get(candidate.item, candidate.rank))
    return ranks


def _pair_text(fused: FusedCandidate) -> str:
    """What the reranker reads of a candidate: its view, or its text when it has none."""
    return fused.candidate.view_text or fused.candidate.text


def run_items(
    result: MemoryResult,
    lists: Sequence[PlacedList],
    selected: Sequence[ScoredCandidate] = (),
) -> list[dict[str, object]]:
    """The block as the scorer reads it, each item with its fetch rank (plan v4, A26).

    A list's facts are each written under the list's ref, with the list's rank. A note is
    named by its path, the fixture's note id; a second chunk of a file already in the block is
    left out, since the scorer counts an item once, and the note keeps the best rank of its
    chunks. The rank is the fetcher's own, not the block position, because the block lays
    items out by section and the scorer's MRR reads the rank.

    An item's rank is the lowest rank among the injected candidates it stands for. A fact a
    selected facet placed also stands for the facet: `selected` (`FetchOutcome.selected`)
    carries it at the earlier of its own position and the facet's, and a member the pools
    never returned has no stage candidate of its own at all. A list's rank is the lowest over
    all its facts, since the block packs a list at its earliest member.

    Raises:
        ValueError: A block item has no injected candidate in the stage and no selected
            position, so it has no rank and the scorer would refuse the record.
    """
    by_ref = {placed.ref: placed for placed in lists}
    ranks = _item_ranks(result)
    placed_at = {(s.candidate.kind, s.candidate.item_id): s.position for s in selected}
    first: dict[tuple[str, str], tuple[str, float]] = {}
    best: dict[tuple[str, str], int] = {}

    def add(ref: str, kind: str, item_id: str, score: float, rank: int) -> None:
        key = (kind, item_id)
        first.setdefault(key, (ref, score))
        best[key] = min(rank, best.get(key, rank))

    for position, item in enumerate(result.items, start=1):
        placed = by_ref.get(item.ref)
        carried = (
            [("fact", fact.fact_key) for fact in placed.facts.facts]
            if placed is not None
            else [(item.kind, item.item_id)]
        )
        candidates = [ranks[position]] if position in ranks else []
        candidates += [placed_at[key] for key in carried if key in placed_at]
        if not candidates:
            raise ValueError(
                f"block item {item.ref} has no injected candidate in the stage, so no rank"
            )
        rank = min(candidates)
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


def dropped_dispositions(outcome: FetchOutcome) -> dict[tuple[str, str], str]:
    """The disposition of each candidate that left the shortlist before the order.

    A suppressed fact or note, a note whose whole-text read was late or failed, a facet
    without members and a facet whose member lookup was late are not in `outcome.ranked`, so
    the trace gives them no rank; the stage still records each, ranked after every ordered
    candidate and carrying its pool rank, which is its 1-based
    place in `outcome.fused`. Keyed by `(kind, item_id)`.
    """
    ordered = len(outcome.ranked)
    dropped: dict[tuple[str, str], str] = {}
    for stage in outcome.result.stage.candidates:
        if stage.rank <= ordered or stage.pool_rank is None:
            continue
        if not 1 <= stage.pool_rank <= len(outcome.fused):
            continue
        candidate = outcome.fused[stage.pool_rank - 1].candidate
        dropped[(candidate.kind, candidate.item_id)] = stage.disposition
    return dropped


def run_record(
    probe: Probe,
    result: MemoryResult,
    lists: Sequence[PlacedList],
    *,
    counter: ApproximateTokenCounter,
    cpu_load_procs: int,
    embedder_device: Device,
    selected: Sequence[ScoredCandidate] = (),
) -> dict[str, object]:
    """One run-file record: the scorer's keys and the per-probe measurements."""
    timings = result.timings
    return {
        "probe_id": probe.probe_id,
        "items": run_items(result, lists, selected),
        "tokens": result.tokens,
        "budget_tokens": result.budget_tokens,
        "critical_path_ms": round(timings.total_ms, 3),
        "timings_ms": _timings_ms(timings),
        "reranked": result.reranked,
        # The stage records one candidate per scored shortlist entry, capped, and counts the rest.
        "shortlisted": len(result.stage.candidates) + result.stage.truncated,
        "degraded_reason": result.degraded_reason,
        "delta_tokens_est": counter.count(result.block_text or "") + counter.count(probe.query),
        "cpu_load_procs": cpu_load_procs,
        "embedder_device": embedder_device,
    }


def _timings_ms(timings: FetchTimings) -> dict[str, float]:
    return {
        "stage1": round(timings.stage1_ms, 3),
        "stage2": round(timings.stage2_ms, 3),
        "rerank": round(timings.rerank_ms, 3),
        "queue_wait": round(timings.queue_wait_ms, 3),
    }


def need_json(need: NeedTrace | None) -> dict[str, object] | None:
    """The `NeedTrace`'s fields as the run file holds them, its tuples as lists; None for none."""
    if need is None:
        return None
    values: dict[str, object] = {}
    for f in fields(need):
        value = getattr(need, f.name)
        values[f.name] = list(value) if isinstance(value, tuple) else value
    return values


def stage_ids(outcome: FetchOutcome) -> dict[str, list[str]]:
    """Each stage's candidates as `kind:id`, in the stage's order.

    `fused` is `FetchOutcome.fused` (pool order, before the `rerank_top_n` cut), `shortlisted`
    is `FetchOutcome.ranked` (the shortlist after its drops, in position order) and `selected`
    is `FetchOutcome.selected` (selection and expansion, in packing order), so a reader can see
    at which stage an item left. All three are empty when stage 2 did not run.
    """

    def ids(entries: Sequence[FusedCandidate] | Sequence[ScoredCandidate]) -> list[str]:
        return [f"{entry.candidate.kind}:{entry.candidate.item_id}" for entry in entries]

    return {
        "fused": ids(outcome.fused),
        "shortlisted": ids(outcome.ranked),
        "selected": ids(outcome.selected),
    }


def outcome_record(
    probe: Probe,
    outcome: FetchOutcome,
    *,
    counter: ApproximateTokenCounter,
    cpu_load_procs: int,
    embedder_device: Device,
) -> dict[str, object]:
    """`run_record` of a fetch's block, with the need gate's decision, stage ids and cuts.

    `need` is `need_json` of the stage's `NeedTrace`, None when the fetch had no gate.
    `stage_ids` is `stage_ids(outcome)`. `cuts` is present only when the outcome carries cut
    blocks, which the fetcher builds in shadow mode for a fetch that built a block: for each
    `band_k`, the cut block's items as `run_items` writes them and its token count.
    """
    record = run_record(
        probe,
        outcome.result,
        outcome.lists,
        counter=counter,
        cpu_load_procs=cpu_load_procs,
        embedder_device=embedder_device,
        selected=outcome.selected,
    )
    record["need"] = need_json(outcome.result.stage.need)
    record["stage_ids"] = stage_ids(outcome)
    if outcome.cuts:
        record["cuts"] = {
            str(k): {
                "items": run_items(cut.result, cut.lists, outcome.selected),
                "tokens": cut.result.tokens,
            }
            for k, cut in sorted(outcome.cuts.items())
        }
    return record


def followup_record(followup_id: str, turn: str, outcome: FetchOutcome) -> dict[str, object]:
    """One fetch of a follow-up pair: which pair and turn, the gate's decision, its latency."""
    timings = outcome.result.timings
    return {
        "followup_id": followup_id,
        "turn": turn,
        "need": need_json(outcome.result.stage.need),
        "critical_path_ms": round(timings.total_ms, 3),
        "timings_ms": _timings_ms(timings),
    }


@dataclass(frozen=True, slots=True)
class Fixture:
    """The fixture rows the run reads."""

    probes: tuple[Probe, ...]
    facts: tuple[Fact, ...]
    notes: tuple[VaultNote, ...]
    corrections: tuple[Correction, ...]

    @classmethod
    def load(cls, directory: Path = FIXTURE_DIR, probes: Path | None = None) -> Fixture:
        """Every file the run needs from `directory`, the probes from `probes` when given.

        Args:
            directory: The fixture directory.
            probes: The probe file; None reads `directory`'s `probes.jsonl`.
        """
        return cls(
            probes=load_probes(probes if probes is not None else directory / "probes.jsonl"),
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
        "embedder_device": args.embedder_device,
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
        "gpu_load_procs": args.gpu_load_procs,
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
        by_rank = {candidate.rank: candidate for candidate in result.stage.candidates}
        fused_at = {(f.candidate.kind, f.candidate.item_id): n for n, f in enumerate(fused, 1)}
        ranked_at = {
            (s.candidate.kind, s.candidate.item_id): n for n, s in enumerate(outcome.ranked, 1)
        }

        def disposition(rank: int | None) -> str | None:
            stage = by_rank.get(rank) if rank is not None else None
            return stage.disposition if stage is not None else None

        dropped = dropped_dispositions(outcome)

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
                "pool_rank": f.pool_rank,
                "text_chars": len(f.candidate.text),
                "pair_tokens": tokens(_pair_text(f)),
                "shortlist": n if n <= self._top_n else None,
            }
            for n, f in enumerate(fused, 1)
        ]
        scored_entries = [
            {
                "rank": n,
                "kind": s.candidate.kind,
                "item_id": s.candidate.item_id,
                "position": s.position,
                "pool_rank": s.pool_rank,
                "ce_rank": s.ce_rank,
                "p_ce": s.p_ce,
                "logit": logit(s.p_ce) if s.p_ce is not None else None,
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
        # The reranker scores the head of the shortlist the fetch kept, in pool order.
        kept = sorted(outcome.ranked, key=lambda s: s.pool_rank)
        for call in reranks:
            texts = list(call.texts) if call.texts is not None else []
            if call.texts is not None:
                head = [s.candidate.view_text or s.candidate.text for s in kept[: len(texts)]]
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
        for item in run_items(result, outcome.lists, outcome.selected):
            block_ids.setdefault(str(item["kind"]), set()).add(str(item["id"]))
        targets = self._targets(
            probe,
            searched,
            fused,
            fused_at,
            ranked_at,
            disposition,
            dropped,
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
        dropped: dict[tuple[str, str], str],
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
                            base,
                            probe,
                            searched,
                            fused_at,
                            ranked_at,
                            disposition,
                            dropped,
                            block_ids,
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
                            dropped,
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
        dropped: dict[tuple[str, str], str],
        block_ids: dict[str, set[str]],
    ) -> tuple[dict[str, object], list[dict[str, object]]]:
        """A fact target; one dropped before the order has no rank but keeps its disposition."""
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
            "disposition": disposition(rank) if rank is not None else dropped.get(("fact", key)),
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
        dropped: dict[tuple[str, str], str],
        block_ids: dict[str, set[str]],
        indexed_notes: set[str],
    ) -> tuple[dict[str, object], list[dict[str, object]]]:
        """A note target; one dropped before the order, as when suppressed, keeps that reason."""
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
        elif dispositions:
            chosen = dispositions[0]
        else:
            chosen = next((dropped[key] for key in keys if key in dropped), None)
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


@dataclass(frozen=True, slots=True)
class RunSetting:
    """One fetcher's settings in a walk of the fixture.

    Attributes:
        config: The fetcher's and its block builder's memory settings. Its `need_gate` is the
            gate's mode, and its `need_context` is passed to the fetcher.
        need_gate: The gate, set exactly when `config.need_gate` is `on` or `shadow`.

    Raises:
        ValueError: A gate with `config.need_gate` off, or none with it on or shadow, so a
            setting never runs ungated while its config says it is gated.
    """

    config: MemoryConfig
    need_gate: NeedGate | None

    def __post_init__(self) -> None:
        gated = self.config.need_gate != "off"
        if gated != (self.need_gate is not None):
            given = "a gate" if self.need_gate is not None else "no gate"
            raise ValueError(
                f"a setting with need_gate {self.config.need_gate!r} was given {given}; it has "
                "a gate exactly when need_gate is on or shadow"
            )


def _need_mode(config: MemoryConfig) -> Literal["on", "shadow"]:
    # The fetcher reads its mode only when it has a gate; an `off` setting has none.
    return "shadow" if config.need_gate == "shadow" else "on"


def _check_settings(settings: Sequence[RunSetting]) -> None:
    """Refuse settings that cannot share one index: it embeds one fact view in one zone."""
    if not settings:
        raise ValueError("a walk of the fixture needs at least one setting")
    views = sorted({setting.config.fact_view for setting in settings})
    if len(views) > 1:
        raise ValueError(f"the settings of one walk share one fact view, got {views}")
    zones = {setting.config.tz for setting in settings}
    if len(zones) > 1:
        raise ValueError(f"the settings of one walk share one zone, got {sorted(map(str, zones))}")


async def run_settings(
    fixture: Fixture,
    *,
    settings: Sequence[RunSetting],
    embedder: TokenEmbedder | SentenceTransformerEmbedder,
    reranker: Reranker,
    workdir: Path,
    probes: Sequence[Probe] | None = None,
    cpu_load_procs: int = 0,
    embedder_device: Device = "cpu",
    placement: BlockPlacement = "user_prefix",
    rescan: ForcedRescan | None = None,
    warm_up: bool = False,
    followups: Sequence[tuple[str, str, str]] = (),
) -> list[list[dict[str, object]]]:
    """Run each probe through one fetcher per setting, over one walk of the indexes.

    The index and the sidecar are built once in `workdir` and brought to each `now` once; at
    each `now` every probe of that `now` runs through each setting's fetcher, in the settings'
    order. The fetchers share the models, the executors and the indexes, and each has its own
    sessions and block builder.

    Args:
        settings: The settings; they share one fact view and one zone, which the index is
            built with.
        embedder_device: The device the embedder is on, written in each probe record.
        placement: The placement every block builder words the block for.
        rescan: Forced rescans for the whole walk, as `run_fixture`'s.
        warm_up: Before each `now`'s probes, one fetch of `WARM_UP_QUERY` through the first
            setting's fetcher, in a session of its own and recorded nowhere, so no setting's
            latency carries the first read after an index update.
        followups: `(id, previous, query)` triples. At the fixture's last `now` (its latest
            probe's, walked even when `probes` leaves it out), each runs through each setting's
            fetcher as two fetches in session `followup-<id>`: `previous` with no recent turn,
            then `query` with `previous` as its one recent user turn.

    Returns:
        Per setting, in order: one record per probe (`outcome_record`), in the order of
        `probes` (all the fixture's by default), then two records per follow-up
        (`followup_record`), `previous` then `query`, in the order of `followups`.

    Raises:
        ValueError: No setting, or settings of two fact views or two zones.
    """
    return await _walk(
        fixture,
        settings=tuple(settings),
        embedder=embedder,
        reranker=reranker,
        workdir=workdir,
        probes=probes,
        cpu_load_procs=cpu_load_procs,
        embedder_device=embedder_device,
        placement=placement,
        rescan=rescan,
        trace=None,
        warm_up=warm_up,
        followups=tuple(followups),
    )


async def run_fixture(
    fixture: Fixture,
    *,
    config: MemoryConfig,
    embedder: TokenEmbedder | SentenceTransformerEmbedder,
    reranker: Reranker,
    workdir: Path,
    probes: Sequence[Probe] | None = None,
    cpu_load_procs: int = 0,
    embedder_device: Device = "cpu",
    placement: BlockPlacement = "user_prefix",
    rescan: ForcedRescan | None = None,
    trace: TraceSetup | None = None,
    need_gate: NeedGate | None = None,
) -> list[dict[str, object]]:
    """Run each probe through a real fetcher over indexes built in `workdir`.

    Args:
        embedder_device: The device the command line placed the embedder on, written in each
            record; the fakes run in Python, so the default suits them.
        placement: The placement the block builder words the block for.
        rescan: Forced rescans to run on the run's `memory-index` executor for the whole
            run; started here and stopped before the executors shut down.
        trace: Write the per-probe candidate trace (`ProbeTracer`); the fetcher then gets
            recording wrappers of the embedder and the reranker.
        need_gate: The need gate, run in `config.need_gate`'s mode; set exactly when that is
            `on` or `shadow` (`RunSetting`).

    Returns:
        One record per probe (`outcome_record`), in the order of `probes` (all the fixture's
        by default).

    Raises:
        ValueError: A gate with `config.need_gate` off, or none with it on or shadow.
    """
    runs = await _walk(
        fixture,
        settings=(RunSetting(config, need_gate),),
        embedder=embedder,
        reranker=reranker,
        workdir=workdir,
        probes=probes,
        cpu_load_procs=cpu_load_procs,
        embedder_device=embedder_device,
        placement=placement,
        rescan=rescan,
        trace=trace,
        warm_up=False,
        followups=(),
    )
    return runs[0]


async def _walk(
    fixture: Fixture,
    *,
    settings: tuple[RunSetting, ...],
    embedder: TokenEmbedder | SentenceTransformerEmbedder,
    reranker: Reranker,
    workdir: Path,
    probes: Sequence[Probe] | None,
    cpu_load_procs: int,
    embedder_device: Device,
    placement: BlockPlacement,
    rescan: ForcedRescan | None,
    trace: TraceSetup | None,
    warm_up: bool,
    followups: tuple[tuple[str, str, str], ...],
) -> list[list[dict[str, object]]]:
    """`run_settings`, and with one setting `run_fixture`, whose trace needs one fetcher."""
    _check_settings(settings)
    if trace is not None and len(settings) != 1:
        raise ValueError("the trace follows one fetcher, so it needs exactly one setting")
    first = settings[0].config
    chosen = tuple(probes if probes is not None else fixture.probes)
    last = max(probe.now for probe in fixture.probes) if followups else None
    nows = sorted({probe.now for probe in chosen} | ({last} if last is not None else set()))
    counter = ApproximateTokenCounter()
    index_path = workdir / "memory_index.db"
    writer = open_index(index_path, embedding_model_id=EMBEDDING_MODEL_ID)
    reader = open_reader(index_path, embedding_model_id=EMBEDDING_MODEL_ID)
    probe_state = IndexStateProbe.open(str(index_path), embedding_model_id=EMBEDDING_MODEL_ID)
    sidecar = VaultSidecarIndex(
        SidecarIndexConfig(db_path=str(workdir / "vault_sidecar.db")), embedder
    )
    sidecar.initialize()
    # The correction log, in a store of the run's own; each correction is stamped explicitly.
    event_store = EventStore(db_path=str(workdir / "event_store.db"))
    event_store.initialize()
    corrections = CorrectionService(event_store, now_fn=SystemClock().now)
    executors = build_memory_executors()
    pools = FetcherExecutors(
        index=executors[INDEX_EXECUTOR],
        model=executors[MODEL_EXECUTOR],
        gate=executors[GATE_EXECUTOR],
    )
    tracer: ProbeTracer | None = None
    try:
        index = FactIndex(
            writer, embedder, tz=first.tz, reader=reader, view=FACT_VIEWS[first.fact_view]
        )
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
                config=first,
            )
        notes = VaultCandidates(sidecar, vault_root=str(workdir / "vault"), tz=first.tz)
        fetchers = [
            MemoryFetcher(
                index=index,
                embedder=fetch_embedder,
                reranker=fetch_reranker,
                block_builder=MemoryBlockBuilder(
                    counter=counter, config=setting.config, placement=placement
                ),
                config=setting.config,
                executors=pools,
                need_gate=setting.need_gate,
                need_mode=_need_mode(setting.config),
                need_context=setting.config.need_context,
                clock=SystemClock(),
                index_state=probe_state,
                notes=notes,
                suppression=LoggedSuppression(event_store),
            )
            for setting in settings
        ]
        for fetcher in fetchers:
            corrections.add_recorded_listener(_invalidator(fetcher))
        recorded: set[str] = set()
        apply = index.diff_apply
        if rescan is not None:
            rescan.start(index, executors[INDEX_EXECUTOR])
            apply = rescan.apply
        records: list[dict[str, dict[str, object]]] = [{} for _ in settings]
        followed: list[list[dict[str, object]]] = [[] for _ in settings]
        indexed_notes: set[str] = set()
        for step, now in enumerate(nows):
            apply(facts_known_at(fixture.facts, fixture.corrections, now))
            record_due_corrections(corrections, fixture.corrections, now, recorded)
            for path, note in notes_known_at(fixture.notes, now).items():
                if path not in indexed_notes:
                    sidecar.upsert_file(path, note.content, note.mtime, note.frontmatter)
                    indexed_notes.add(path)
            if warm_up:
                # A session per `now`: one reused across them would serve its cached block.
                await fetchers[0].fetch_outcome(
                    MemoryRequest(
                        session_id=f"warm-up-{step}",
                        conversation_id=None,
                        query_text=WARM_UP_QUERY,
                        now=now,
                        recent_user_turns=(),
                        turn_index=0,
                    )
                )
            for probe in (p for p in chosen if p.now == now):
                request = MemoryRequest(
                    session_id=f"probe-{probe.probe_id}",
                    conversation_id=None,
                    query_text=probe.query,
                    now=probe.now,
                    recent_user_turns=(),
                    turn_index=0,
                    conversation_start=probe.conversation_start,
                )
                for n, fetcher in enumerate(fetchers):
                    outcome = await fetcher.fetch_outcome(request)
                    records[n][probe.probe_id] = outcome_record(
                        probe,
                        outcome,
                        counter=counter,
                        cpu_load_procs=cpu_load_procs,
                        embedder_device=embedder_device,
                    )
                    if tracer is not None:
                        await tracer.trace(probe, outcome, indexed_notes=indexed_notes)
            if now == last:
                for followup_id, previous, query in followups:
                    turns = (("previous", previous, ()), ("query", query, (previous,)))
                    for n, fetcher in enumerate(fetchers):
                        for turn_index, (turn, text, recent) in enumerate(turns):
                            outcome = await fetcher.fetch_outcome(
                                MemoryRequest(
                                    session_id=f"followup-{followup_id}",
                                    conversation_id=None,
                                    query_text=text,
                                    now=now,
                                    recent_user_turns=recent,
                                    turn_index=turn_index,
                                )
                            )
                            followed[n].append(followup_record(followup_id, turn, outcome))
        return [
            [records[n][probe.probe_id] for probe in chosen] + followed[n]
            for n in range(len(settings))
        ]
    finally:
        if rescan is not None:
            rescan.stop()
        if tracer is not None:
            tracer.close()
        for executor in executors.values():
            executor.shutdown(wait=True, cancel_futures=True)
        event_store.close()
        sidecar.close()
        probe_state.close()
        reader.close()
        writer.close()


def _invalidator(fetcher: MemoryFetcher) -> Callable[[CorrectionRecord], None]:
    """The recorded listener that drops `fetcher`'s cached stage 1 and blocks.

    An adapter, as `CorrectionService.add_recorded_listener` requires: the bare
    `fetcher.invalidate` would take the record as a session id and drop nothing.
    """

    def invalidate(record: CorrectionRecord) -> None:
        fetcher.invalidate()

    return invalidate


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


def matmul_loop(stop: _Event, device: str, size: int = _MATMUL_SIZE) -> None:
    """Keep `device` busy with square matmuls until `stop` is set.

    A batch of matmuls is queued, then waited for on CUDA, before each check of `stop`, so the
    queue stays short and the process stops within a batch of being told.
    """
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    on_cuda = torch.device(device).type == "cuda"
    while not stop.is_set():
        for _ in range(_MATMUL_BATCH):
            torch.matmul(a, b)
        if on_cuda:
            torch.cuda.synchronize(device)


@contextlib.contextmanager
def _load_processes(
    procs: int,
    *,
    what: str,
    name: str,
    target: Callable[..., None],
    args: tuple[object, ...],
    context: multiprocessing.context.BaseContext,
    process_factory: Callable[..., _Process] | None,
    event_factory: Callable[[], _Event] | None,
    clock: Callable[[], float],
    join_timeout_s: float,
) -> Iterator[list[_Process]]:
    """Run `procs` processes of `target(stop, *args)` for the body, then stop them.

    Stopping sets the shared event and joins each process within `join_timeout_s` in total,
    by `clock`; a process still alive then is terminated.

    Raises:
        ValueError: `procs` is negative.
    """
    if procs < 0:
        raise ValueError(f"{what} load processes must not be negative, got {procs}")
    if procs == 0:
        yield []
        return
    make_process = process_factory or context.Process
    stop = (event_factory or context.Event)()
    started: list[_Process] = []
    try:
        for n in range(procs):
            process = make_process(
                target=target, args=(stop, *args), name=f"{name}-{n}", daemon=True
            )
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


def cpu_load(
    procs: int,
    *,
    process_factory: Callable[..., _Process] | None = None,
    event_factory: Callable[[], _Event] | None = None,
    clock: Callable[[], float] = time.monotonic,
    join_timeout_s: float = _LOAD_JOIN_TIMEOUT_S,
) -> contextlib.AbstractContextManager[list[_Process]]:
    """Run `procs` busy-loop processes for the body, then stop them (`_load_processes`).

    Raises:
        ValueError: `procs` is negative.
    """
    return _load_processes(
        procs,
        what="cpu",
        name="memory-load",
        target=spin,
        args=(),
        context=_default_context(),
        process_factory=process_factory,
        event_factory=event_factory,
        clock=clock,
        join_timeout_s=join_timeout_s,
    )


def gpu_load(
    procs: int,
    *,
    process_factory: Callable[..., _Process] | None = None,
    event_factory: Callable[[], _Event] | None = None,
    clock: Callable[[], float] = time.monotonic,
    join_timeout_s: float = _LOAD_JOIN_TIMEOUT_S,
) -> contextlib.AbstractContextManager[list[_Process]]:
    """Run `procs` processes looping matmuls on `cuda` for the body, then stop them.

    Each process makes its own CUDA context, so the processes are spawned rather than forked: a
    forked child cannot use CUDA once its parent has initialized it. The caller checks that
    CUDA is available first. Stopping is `_load_processes`'s.

    Raises:
        ValueError: `procs` is negative.
    """
    return _load_processes(
        procs,
        what="gpu",
        name="memory-gpu-load",
        target=matmul_loop,
        args=("cuda",),
        context=multiprocessing.get_context("spawn"),
        process_factory=process_factory,
        event_factory=event_factory,
        clock=clock,
        join_timeout_s=join_timeout_s,
    )


# -- GPU memory ------------------------------------------------------------------------------


class GpuUnavailableError(RuntimeError):
    """NVML or CUDA, which a GPU measurement needs, is not available to the process."""


class NvmlReader(Protocol):
    """Device-wide used memory, read through NVML, which never touches CUDA."""

    def used_bytes(self) -> int:
        """The device's used memory in bytes, every process's and the driver's."""
        ...

    def close(self) -> None:
        """Release NVML."""
        ...


class PynvmlReader:
    """NVML through pynvml, on GPU index 0, the one device the run's `cuda` names.

    Args:
        nvml: The imported pynvml module.
        handle: The device's NVML handle.
    """

    def __init__(self, nvml: object, handle: object) -> None:
        self._nvml = nvml
        self._handle = handle

    @classmethod
    def open(cls, index: int = 0) -> PynvmlReader:
        """Initialize NVML and take device `index`'s handle.

        Raises:
            GpuUnavailableError: pynvml is not installed, or NVML cannot initialize or find the
                device.
        """
        try:
            import pynvml
        except ImportError as exc:
            raise GpuUnavailableError(f"NVML is not available: {exc}") from exc
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        except pynvml.NVMLError as exc:
            raise GpuUnavailableError(f"NVML is not available: {exc}") from exc
        return cls(pynvml, handle)

    def used_bytes(self) -> int:
        """The device's used memory in bytes, every process's and the driver's."""
        return int(self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used)  # type: ignore[attr-defined]

    def close(self) -> None:
        """Shut NVML down; an error doing so is ignored, since the readings are taken."""
        nvml_error = self._nvml.NVMLError  # type: ignore[attr-defined]
        with contextlib.suppress(nvml_error):
            self._nvml.nvmlShutdown()  # type: ignore[attr-defined]


class TorchCuda:
    """The CUDA calls the VRAM log makes, through one torch module, so a test can pass a fake.

    Args:
        module: The torch module; tests pass a recording stand-in.
    """

    def __init__(self, module: object = torch) -> None:
        self._torch = module

    def is_available(self) -> bool:
        """Whether torch can use CUDA."""
        return bool(self._torch.cuda.is_available())  # type: ignore[attr-defined]

    def create_context(self) -> None:
        """Create the process's CUDA context: a one-element tensor on `cuda`, then a synchronize.

        `torch.cuda.init()` alone initializes torch's CUDA state without necessarily creating
        the context, so the figure after it could miss the context's memory.
        """
        self._torch.ones(1, device="cuda")  # type: ignore[attr-defined]
        self._torch.cuda.synchronize()  # type: ignore[attr-defined]

    def reading(self) -> dict[str, int]:
        """The device's used memory by `mem_get_info` and the caching allocator's counts, in bytes.

        Synchronizes first, so work still queued has finished before the counts are read.
        """
        cuda = self._torch.cuda  # type: ignore[attr-defined]
        cuda.synchronize()
        free, total = cuda.mem_get_info()
        return {
            "mem_get_info_used": int(total - free),
            "memory_allocated": int(cuda.memory_allocated()),
            "max_memory_allocated": int(cuda.max_memory_allocated()),
            "memory_reserved": int(cuda.memory_reserved()),
            "max_memory_reserved": int(cuda.max_memory_reserved()),
        }

    def empty_cache(self) -> None:
        """Return the caching allocator's unused blocks to the driver."""
        self._torch.cuda.empty_cache()  # type: ignore[attr-defined]


class SttModel(Protocol):
    """The Whisper model's one call the VRAM log makes."""

    def transcribe(self, audio: np.ndarray, **kwargs: object) -> object:
        """Transcribe 16 kHz mono float audio."""
        ...


def load_whisper_model(name: str) -> SttModel:
    """Load Whisper model `name` on `cuda`; `whisper` is imported only here, for `--vram-stt`."""
    import whisper

    return whisper.load_model(name, device="cuda")


def stt_clip() -> np.ndarray:
    """The clip `--vram-stt` transcribes: 30 s at 16 kHz of low-level noise from a fixed seed."""
    rng = np.random.default_rng(_STT_CLIP_SEED)
    samples = rng.standard_normal(_STT_SAMPLE_RATE * _STT_CLIP_S) * _STT_CLIP_LEVEL
    return samples.astype(np.float32)


@dataclass(frozen=True, slots=True)
class GpuAccess:
    """What the GPU flags reach the GPU through; tests pass fakes for each.

    Attributes:
        cuda: Torch's CUDA calls.
        open_nvml: Opens the NVML reader; raises GpuUnavailableError without NVML.
        load_whisper: Loads a Whisper model on `cuda` by name.
        load_procs: Runs N GPU load processes for a `with` body.
    """

    cuda: TorchCuda
    open_nvml: Callable[[], NvmlReader]
    load_whisper: Callable[[str], SttModel]
    load_procs: Callable[[int], contextlib.AbstractContextManager[object]]


def default_gpu_access() -> GpuAccess:
    """The real GPU: torch, pynvml, openai-whisper and spawned matmul processes."""
    return GpuAccess(
        cuda=TorchCuda(torch),
        open_nvml=PynvmlReader.open,
        load_whisper=load_whisper_model,
        load_procs=gpu_load,
    )


def _mib(value: int) -> float:
    return round(value / _MIB, 3)


class VramRecorder:
    """Reads the process's GPU memory at the `--vram-log` points and derives the figures.

    `start` opens NVML and reads `before_cuda` before any CUDA call, then checks CUDA, creates
    the context and reads `context`; with an STT model name it loads that Whisper model,
    transcribes `stt_clip()` once with `fp16=False` and reads `stt`. The caller reads `loaded`
    and `after_run` with `read`, and `release` releases the reranker and the Whisper model
    before it reads `released`.

    Args:
        gpu: The GPU access.
        stt_model: The Whisper model to load, or None for no STT point.
    """

    def __init__(self, gpu: GpuAccess, *, stt_model: str | None) -> None:
        self._gpu = gpu
        self._stt_name = stt_model
        self._stt: SttModel | None = None
        self._nvml: NvmlReader | None = None
        self.points: dict[str, dict[str, float]] = {}

    def start(self) -> None:
        """Read `before_cuda`, make the context and read `context`, then the STT point.

        Raises:
            GpuUnavailableError: NVML or CUDA is not available.
        """
        self._nvml = self._gpu.open_nvml()
        self.points["before_cuda"] = {"nvml_used": _mib(self._nvml.used_bytes())}
        if not self._gpu.cuda.is_available():
            raise GpuUnavailableError("CUDA is not available")
        self._gpu.cuda.create_context()
        self.read("context")
        if self._stt_name is not None:
            model = self._gpu.load_whisper(self._stt_name)
            model.transcribe(stt_clip(), fp16=False)
            self._stt = model
            del model
            self.read("stt")

    def read(self, point: str) -> None:
        """Record `point`: NVML used memory, then torch's readings, all in MiB."""
        if self._nvml is None:
            raise RuntimeError("VramRecorder.read before start")
        cuda = self._gpu.cuda.reading()
        reading = {"nvml_used": _mib(self._nvml.used_bytes())}
        reading.update({key: _mib(value) for key, value in cuda.items()})
        self.points[point] = reading

    def release(self, reranker: Reranker) -> None:
        """Release the reranker, drop the Whisper model and empty the cache, then read `released`."""
        reranker.release()
        self._stt = None
        gc.collect()
        self._gpu.cuda.empty_cache()
        self.read("released")

    def close(self) -> None:
        """Shut the NVML reader down, if it was opened."""
        if self._nvml is not None:
            self._nvml.close()
            self._nvml = None

    def derived(self) -> dict[str, float | None]:
        """The figures over NVML used memory, each a difference of two points."""

        def used(point: str) -> float:
            return self.points[point]["nvml_used"]

        has_stt = "stt" in self.points
        return {
            "process_mib": round(used("after_run") - used("before_cuda"), 3),
            "context_mib": round(used("context") - used("before_cuda"), 3),
            "stt_mib": round(used("stt") - used("context"), 3) if has_stt else None,
            "marginal_mib": round(used("after_run") - used("stt" if has_stt else "context"), 3),
            "residual_mib": round(used("released") - used("before_cuda"), 3),
        }

    def log(self, *, device: Device, embedder_device: Device) -> dict[str, object]:
        """The `--vram-log` file's content."""
        return {
            "unit": "MiB",
            "device": device,
            "embedder_device": embedder_device,
            "vram_stt": self._stt_name,
            "points": dict(self.points),
            "derived": self.derived(),
        }


# -- the command -----------------------------------------------------------------------------


def config_from_args(args: argparse.Namespace) -> MemoryConfig:
    """MemoryConfig from the environment, with each value the command line sets.

    Raises:
        ValueError: `--rerank-model` is not under `/opt/models`; refused even with the fake
            reranker, so a run never records a model path it could not have loaded.
    """
    config = MemoryConfig.from_env()
    overrides: dict[str, object] = {}
    for name in ("block_tokens", "rerank_top_n", "select_k", "list_max_items", "deadline_ms"):
        value = getattr(args, name)
        if value is not None:
            overrides[name] = value
    if args.facets is not None:
        overrides["facets"] = args.facets == "on"
    if args.rerank_model is not None:
        overrides["rerank_model"] = _model_path(args.rerank_model)
    if args.fact_view is not None:
        overrides["fact_view"] = args.fact_view
    if args.need_gate is not None:
        overrides["need_gate"] = args.need_gate
    if args.need_context is not None:
        overrides["need_context"] = args.need_context == "on"
    need_embedder = "shared" if args.need_embedder == "shared" else args.need_embedder_name
    if need_embedder is not None:
        overrides["need_embedder"] = need_embedder
    return replace(config, **overrides) if overrides else config


@dataclass(frozen=True, slots=True)
class NeedGatePlan:
    """What the run's need gate is built from, read and checked before any model loads.

    Attributes:
        bank: The need bank, read once.
        model: The model file's record for the gate's key.
        spec: The gate embedder's `NEED_EMBEDDERS` spec: its query prefix and attention setting.
        embedder_path: The gate embedder's directory; None for `shared`, which is the fact
            embedder itself.
        embedder_name: The gate embedder's name in the key.
        embedder_revision: Its revision in the key.
    """

    bank: NeedBank
    model: NeedModel
    spec: NeedEmbedderSpec
    embedder_path: str | None
    embedder_name: str
    embedder_revision: str


def _locked_entry(parser: argparse.ArgumentParser, lock: Path, name: str) -> tuple[str, str]:
    """`name`'s name and revision in the models lock; exit 2 naming `--models-lock` without."""
    try:
        entries = parse_lock(lock)
    except LockError as exc:
        parser.error(f"--models-lock: {exc}")
    for entry in entries:
        if entry.name == name:
            return entry.name, entry.revision
    parser.error(f"--models-lock: {lock} has no entry named {name!r}, the fact embedder")


def plan_need_gate(
    args: argparse.Namespace, config: MemoryConfig, parser: argparse.ArgumentParser
) -> NeedGatePlan | None:
    """Read the bank and the gate model, and pick the record for the run's key.

    None with the gate off: then nothing is read, neither the bank, the model file nor the
    lock. A `shared` gate's name and revision are the fact embedder's: its entry in
    `--models-lock` with `--embedder real`, the `FAKE_EMBEDDER_*` constants with the fake. A
    gate embedder given by path has the name and revision its flags give. The query prefix is
    the spec's of `config.need_embedder`.

    Exits 2 (`parser.error`) for an unreadable lock or one without the fact embedder, naming
    `--models-lock`; for a bank `load_bank` refuses; and for a model file `load_models` refuses
    or one with no record for the key, naming `--need-model`.
    """
    if config.need_gate == "off":
        return None
    spec = NEED_EMBEDDERS[config.need_embedder]
    path: str | None = None
    if args.need_embedder != "shared":
        name, revision = args.need_embedder_name, args.need_embedder_revision
        path = _model_path(args.need_embedder)
    elif args.embedder == "fake":
        name, revision = FAKE_EMBEDDER_NAME, FAKE_EMBEDDER_REVISION
    else:
        name, revision = _locked_entry(parser, args.models_lock, EMBEDDER_PATH.name)
    try:
        bank = load_bank()
    except MemoryConfigError as exc:
        parser.error(f"the need bank: {exc}")
    key = NeedKey(bank.sha256, name, revision, spec.query_prefix)
    try:
        model = select_model(load_models(args.need_model), key)
    except MemoryConfigError as exc:
        parser.error(f"--need-model: {exc}")
    return NeedGatePlan(
        bank=bank,
        model=model,
        spec=spec,
        embedder_path=path,
        embedder_name=name,
        embedder_revision=revision,
    )


def build_need_gate(
    plan: NeedGatePlan, fact_embedder: TokenEmbedder | SentenceTransformerEmbedder
) -> NeedGate:
    """The gate the plan describes; it embeds the bank once, here.

    A `shared` gate is built on the fact embedder and shares the fetch's query vector. A gate
    embedder given by path loads on the CPU, as deployed, with exactly
    `need_embedder_kwargs(plan.spec)` beyond its path and device.

    Raises:
        MemoryConfigError: The gate refuses the model (`NeedGate`).
    """
    if plan.embedder_path is None:
        embedder: EmbeddingProvider = fact_embedder
        shares = True
    else:
        embedder = EmbeddingGenerator(
            plan.embedder_path, device="cpu", **need_embedder_kwargs(plan.spec)
        )
        shares = False
    return NeedGate(
        embedder,
        plan.bank,
        plan.model,
        embedder_name=plan.embedder_name,
        embedder_revision=plan.embedder_revision,
        query_prefix=plan.spec.query_prefix,
        shares_query_vector=shares,
    )


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
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), default="cpu", help="the reranker's device"
    )
    parser.add_argument(
        "--embedder-device",
        choices=("cpu", "cuda"),
        help="the embedder's device; defaults to --device",
    )
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
        "--gpu-load-procs", type=int, default=0, help="processes looping a matmul on cuda"
    )
    parser.add_argument(
        "--vram-log", type=Path, help="GPU memory at each point of the run to write (JSON, MiB)"
    )
    parser.add_argument(
        "--vram-stt",
        metavar="MODEL",
        help="Whisper model to load on cuda and run once before the reranker; needs --vram-log",
    )
    parser.add_argument(
        "--rescan-facts",
        type=_positive_int,
        help="forced full rescans for the whole run, each re-embedding N synthetic facts",
    )
    parser.add_argument("--rescan-log", type=Path, help="rescan log to write (JSON)")
    parser.add_argument("--trace", type=Path, help="per-probe candidate trace to write (JSONL)")
    parser.add_argument("--block-tokens", type=int)
    # No `--min-score`: selection is by rank, so argparse refuses the flag (exit 2).
    parser.add_argument("--rerank-top-n", type=int)
    parser.add_argument("--select-k", type=int, help="candidates selected by rank, 1 to 25")
    parser.add_argument("--facets", choices=("on", "off"), help="serve facets; overrides config")
    parser.add_argument("--list-max-items", type=int)
    parser.add_argument("--deadline-ms", type=int)
    parser.add_argument("--fixture-dir", type=Path, default=FIXTURE_DIR)
    parser.add_argument(
        "--probes", type=Path, help="probe file to run; defaults to the fixture's probes.jsonl"
    )
    parser.add_argument(
        "--fact-view", choices=tuple(FACT_VIEWS), help="the fact view the index embeds"
    )
    parser.add_argument(
        "--need-gate",
        choices=("on", "shadow", "off"),
        help="the need gate's mode; defaults to the config's (MIST_MEMORY_NEED_GATE)",
    )
    parser.add_argument(
        "--need-embedder",
        default="shared",
        metavar="shared|PATH",
        help="shared builds the gate on the fact embedder; a PATH under /opt/models loads one",
    )
    parser.add_argument(
        "--need-embedder-name",
        choices=tuple(name for name in NEED_EMBEDDERS if name != "shared"),
        help="the gate embedder's NEED_EMBEDDERS name; with a --need-embedder PATH only",
    )
    parser.add_argument(
        "--need-embedder-revision",
        metavar="SHA",
        help="the gate embedder's revision; with a --need-embedder PATH only",
    )
    parser.add_argument(
        "--models-lock",
        type=Path,
        default=DEFAULT_MODELS_LOCK,
        help="the lock giving a shared gate the fact embedder's revision with --embedder real",
    )
    parser.add_argument(
        "--need-model",
        type=Path,
        help="gate model file in need_model.json's format; defaults to the shipped file",
    )
    parser.add_argument(
        "--need-context", choices=("on", "off"), help="judge a follow-up with its previous turn"
    )
    return parser


def _check_need_flags(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Refuse, with exit 2, a gate embedder's name or revision that does not fit its source.

    `shared` takes the fact embedder's name and revision, so neither flag goes with it; a
    PATH has no other source of them, so it needs both.
    """
    given = [
        flag
        for flag, value in (
            ("--need-embedder-name", args.need_embedder_name),
            ("--need-embedder-revision", args.need_embedder_revision),
        )
        if value is not None
    ]
    if args.need_embedder == "shared":
        if given:
            parser.error(
                f"{' and '.join(given)} go with a --need-embedder PATH; shared takes the fact "
                "embedder's name and revision"
            )
        return
    if len(given) < 2:
        parser.error(
            "a --need-embedder PATH needs --need-embedder-name and --need-embedder-revision"
        )
    try:
        _model_path(args.need_embedder)
    except ValueError as exc:
        parser.error(f"--need-embedder: {exc}")


def _check_gpu_flags(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Refuse, with exit 2, the GPU flag combinations whose figures would mislead."""
    if args.vram_stt is not None and args.vram_log is None:
        parser.error("--vram-stt needs --vram-log, which records what the model holds")
    if args.vram_log is None:
        return
    if args.embedder_device == "cuda":
        parser.error(
            "--vram-log needs the embedder off the GPU (--embedder-device cpu): "
            "its memory would be counted in the reranker's figure"
        )
    if args.gpu_load_procs:
        parser.error(
            "--vram-log cannot run with --gpu-load-procs: the load processes' memory would "
            "be counted in the device-wide figure"
        )


def main(argv: Sequence[str] | None = None, *, gpu: GpuAccess | None = None) -> int:
    """Run the fixture and write the run file; print the scorer's overall metrics.

    Args:
        argv: The command line; None reads `sys.argv`.
        gpu: What the GPU flags reach the GPU through; None is the real GPU.
    """
    parser = _parser()
    args = parser.parse_args(argv)
    if args.rescan_facts is not None and args.rescan_log is None:
        parser.error("--rescan-facts needs --rescan-log, the record that the rescans ran")
    if args.embedder_device is None:
        args.embedder_device = args.device
    _check_gpu_flags(parser, args)
    _check_need_flags(parser, args)
    gpu = gpu if gpu is not None else default_gpu_access()
    if args.gpu_load_procs and not gpu.cuda.is_available():
        parser.error("--gpu-load-procs needs CUDA, which is not available")
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    config = config_from_args(args)
    fixture = Fixture.load(args.fixture_dir, probes=args.probes)
    need_plan = plan_need_gate(args, config, parser)
    rescan = (
        ForcedRescan(args.rescan_facts, rescan_template(fixture))
        if args.rescan_facts is not None
        else None
    )
    vram = VramRecorder(gpu, stt_model=args.vram_stt) if args.vram_log is not None else None
    try:
        records = _run(args, config, fixture, rescan, gpu, vram, parser, need_plan)
        vram_log = (
            vram.log(device=args.device, embedder_device=args.embedder_device)
            if vram is not None
            else None
        )
    finally:
        if vram is not None:
            vram.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    if rescan is not None:
        args.rescan_log.parent.mkdir(parents=True, exist_ok=True)
        args.rescan_log.write_text(json.dumps(rescan.log(), indent=2) + "\n", encoding="utf-8")
    if vram_log is not None:
        args.vram_log.parent.mkdir(parents=True, exist_ok=True)
        args.vram_log.write_text(json.dumps(vram_log, indent=2) + "\n", encoding="utf-8")
    report = score_run(fixture.probes, [parse_run_record(record) for record in records])
    print(json.dumps(report.overall.to_json(), indent=2, sort_keys=True))
    print(f"wrote {len(records)} records to {args.output}", file=sys.stderr)
    return 0


def _run(
    args: argparse.Namespace,
    config: MemoryConfig,
    fixture: Fixture,
    rescan: ForcedRescan | None,
    gpu: GpuAccess,
    vram: VramRecorder | None,
    parser: argparse.ArgumentParser,
    need_plan: NeedGatePlan | None,
) -> list[dict[str, object]]:
    """Load the models, build the need gate, and run the fixture under the asked-for load,
    reading the VRAM points.
    """
    with cpu_load(args.cpu_load_procs), gpu.load_procs(args.gpu_load_procs):
        if vram is not None:
            try:
                vram.start()
            except GpuUnavailableError as exc:
                parser.error(f"--vram-log: {exc}")
        embedder = build_embedder(args.embedder, device=args.embedder_device)
        built = build_reranker(args.reranker, device=args.device, config=config)
        need_gate: NeedGate | None = None
        if need_plan is not None:
            try:
                need_gate = build_need_gate(need_plan, embedder)
            except MemoryConfigError as exc:
                parser.error(f"the need gate: {exc}")
        if vram is not None:
            vram.read("loaded")
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
                    embedder_device=args.embedder_device,
                    placement=args.placement,
                    rescan=rescan,
                    trace=trace,
                    need_gate=need_gate,
                )
            )
        if vram is not None:
            vram.read("after_run")
            vram.release(built.reranker)
    return records


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "DEFAULT_MODELS_LOCK",
    "EMBEDDER_PATH",
    "FAKE_EMBEDDER_NAME",
    "FAKE_EMBEDDER_REVISION",
    "WARM_UP_QUERY",
    "BuiltReranker",
    "CrossEncoderCapture",
    "Fixture",
    "ForcedRescan",
    "GpuAccess",
    "GpuUnavailableError",
    "NeedGatePlan",
    "NoteFile",
    "NvmlReader",
    "ProbeTracer",
    "PynvmlReader",
    "RecordingEmbedder",
    "RecordingReranker",
    "RunSetting",
    "SentenceTransformerEmbedder",
    "SttModel",
    "TokenEmbedder",
    "TokenReranker",
    "TorchCuda",
    "TraceSetup",
    "VramRecorder",
    "build_embedder",
    "build_need_gate",
    "build_reranker",
    "content_stems",
    "cpu_load",
    "default_gpu_access",
    "dropped_dispositions",
    "fact_row",
    "facts_known_at",
    "followup_record",
    "gpu_load",
    "load_whisper_model",
    "logit",
    "main",
    "matmul_loop",
    "need_json",
    "notes_known_at",
    "outcome_record",
    "plan_need_gate",
    "rescan_template",
    "run_fixture",
    "run_items",
    "run_record",
    "run_settings",
    "spin",
    "stage_ids",
    "stt_clip",
    "synthetic_facts",
    "trace_header",
    "word_pair_tokens",
]
