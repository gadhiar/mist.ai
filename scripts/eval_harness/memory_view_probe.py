"""H13's view probe: pick the retrieval views, t*, s0, s1 and K on the dev probes, by a fixed rule.

Plan mist-read-path v5 fixes the rule before it runs (H13; T49 implements it here). It reads the
dev set, tests/fixtures/memory_recall/probes.jsonl, and nothing else: there is no probe-file
argument, so no choice made here can be made on the held-out set the PR 1 gates judge.

For each fact view and note view (backend/memory/retrieval_view.py), each dev probe of a gated
class, and each noise probe, has its query scored against every fact version current at its
`now` (the run harness's `facts_known_at` and its currency choice) and every note chunk the
vault sidecar makes of the notes that exist then:

- CE: the cross-encoder's sigmoid over (query, view text);
- BI: `b(sim) = clamp((sim - s0) / (s1 - s0), 0, 1)` over the bi-encoder cosine of the query
  and the embedded text, on the grid s0 in 0.30 to 0.70 and s1 in s0 + 0.05 to 0.90, both by
  0.05;
- MAX: `max(CE, BI)` on the same grid;
- BLEND: `w * CE + (1 - w) * BI`, reported and never chosen.

The fact embedded is its view text, which the index embeds after T50. A note's embedded text is
its chunk content under every note view: the sidecar embeds `chunk_content` only
(backend/vault/sidecar_index.py:554, `upsert_file`), and the note pool reads its similarity from
the sidecar (design point 2), so the note view changes only what the cross-encoder reads. A
note is one candidate (T50's collapse): its CE is the best over its chunks, and its similarity
too. For CE that is an upper estimate, because after T50 the reranker scores only the view of
the path's first chunk, not the best one.

Then, per combination:

- t* is the highest relevance any noise probe holds, so every dev noise probe stays empty;
- a target passes when its relevance is above t*; a class's figure is the mean over its probes
  of the share of expected targets that pass (no sibling expansion, every pool assumed to hold
  every target);
- the margin is the worst class figure minus that class's idle floor, over the recall gates of
  `memory_recall_gate.PR1_GATES` (`GATED_CLASSES`). The combination with the largest margin wins; on a tie the first in the order
  of `--fact-views`, `--note-views`, then CE, BI, MAX, then s0 and s1 ascending, and the output
  counts the tied settings.

The exit code is the winner's band: margin at least 0.05, proceed (0); from 0 to 0.05, proceed
and flag (3); below 0, stop (1). 2 is an input error. K is the smallest of 20, 30, 50, 75 and
100 at which 95% of the winner's fact targets that pass sit within that rank of its vector
order, or 100 with a flag when none does.

Reported beside the winner: the best blend and whether it beats the best MAX by more than 0.05
of margin (a flag for an amendment, never a change of winner); the share of cross-encoder-only
passes (CE above t*, b not) inside the first 20, 30 and 50 fused by RRF; the norms of the
embedder's vectors; and the sidecar similarity identity `cos = 1 - d^2 / 2`, with `d` read back
from `display_similarity = 1 / (1 + d)`, on the top hits of every probe.

The fused order for that check is rebuilt as the fetcher's after T50 and T53 would make it: the
fact FTS and entity pools from a fact index built at each `now` (searched by text only, so its
own vectors, from the weight-free `TokenEmbedder`, are never read), the fact vector pool as the
winner's vector order cut at K, and the note pools from the sidecar's hybrid query, one entry per
path with its best rank in each leg.

Usage, from the repository root (`/app` in the image), so `backend` imports:
    python -m scripts.eval_harness.memory_view_probe --embedder /opt/models/all-MiniLM-L6-v2
        --reranker /opt/models/ms-marco-MiniLM-L-6-v2 --output view_probe.json
        [--device cpu|cuda] [--fact-views F0,F1,F2,F3] [--note-views N0,N1]
        [--query-prefix TEXT] [--passage-prefix TEXT]

`--query-prefix` and `--passage-prefix` are an embedder's documented instruction prefixes
(bge-small-en-v1.5 asks for one on queries, for example). They are prepended only to the text the
bi-encoder embeds, the query and the passages; the cross-encoder always reads the bare query and
view text. Both are recorded in the output.
"""

from __future__ import annotations

import argparse
import json
import posixpath
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, tzinfo
from pathlib import Path, PurePosixPath
from typing import Literal

import numpy as np
from sentence_transformers import CrossEncoder

from backend.errors import MemoryConfigError, MemoryRerankError
from backend.interfaces import EmbeddingProvider
from backend.knowledge.config import SidecarIndexConfig
from backend.memory.config import MemoryConfig
from backend.memory.fact_index import FactIndex
from backend.memory.fts import query_tokens
from backend.memory.index_db import EMBEDDING_DIMENSION, open_index, open_reader
from backend.memory.models import FactRow
from backend.memory.ranking import RRF_K, rrf_score
from backend.memory.rerank import CrossEncoderReranker, Reranker
from backend.memory.retrieval_view import (
    FACT_VIEWS,
    NOTE_VIEWS,
    render_fact_view,
    render_note_view,
)
from backend.memory.sources import (
    DEFAULT_FACT_POOL_SIZE,
    DEFAULT_NOTE_K,
    ENTITY_POOL,
    FTS_POOL,
    NOTE_FTS_POOL,
    NOTE_VECTOR_POOL,
    VECTOR_POOL,
)
from backend.vault.sidecar_index import VaultSidecarIndex
from scripts.eval_harness.memory_recall import Probe
from scripts.eval_harness.memory_recall_gate import PR1_GATES, Gate

# `_current` is the harness's currency choice, the version `FactGroup.current_at` picks; reused
# so a probe here and a harness run offer the same version of each fact.
from scripts.eval_harness.memory_recall_run import (
    MODELS_ROOT,
    Fixture,
    SentenceTransformerEmbedder,
    TokenEmbedder,
    _current,
    fact_row,
    facts_known_at,
    notes_known_at,
)

Device = Literal["cpu", "cuda"]

EXIT_PROCEED = 0
EXIT_STOP = 1
EXIT_INPUT = 2
EXIT_FLAG = 3

CE = "CE"
BI = "BI"
MAX = "MAX"
BLEND = "BLEND"
CHOSEN_SCORERS: tuple[str, ...] = (CE, BI, MAX)

# The grid in hundredths, so 0.05 steps never drift: s0 0.30 to 0.70, s1 s0 + 0.05 to 0.90.
_S0_HUNDREDTHS = range(30, 71, 5)
_S1_TOP_HUNDREDTHS = 90
_STEP_HUNDREDTHS = 5
GRID: tuple[tuple[float, float], ...] = tuple(
    (s0 / 100, s1 / 100)
    for s0 in _S0_HUNDREDTHS
    for s1 in range(s0 + _STEP_HUNDREDTHS, _S1_TOP_HUNDREDTHS + 1, _STEP_HUNDREDTHS)
)
# The cross-encoder's weight in the reported blend.
BLEND_WEIGHTS: tuple[float, ...] = (0.25, 0.5, 0.75)

PROCEED_MARGIN = 0.05
# A blend this much better than the best MAX, in margin, is flagged for an amendment.
BLEND_FLAG_MARGIN = 0.05
K_STEPS: tuple[int, ...] = (20, 30, 50, 75, 100)
K_COVERAGE = 0.95
FUSED_DEPTHS: tuple[int, ...] = (20, 30, 50)

# Margins are rounded before they are compared, so 0.90 - 0.85 lands on the 0.05 band edge.
_MARGIN_DECIMALS = 9
# Top vector hits per probe on which the sidecar identity is checked.
_IDENTITY_SAMPLE = 3
# The fact index here is searched by text only; its vectors come from the weight-free fake.
_FTS_INDEX_MODEL_ID = "token-embedder"
_FACT_PREFIX = "fact:"
_NOTE_PREFIX = "note:"


# -- the gated classes -----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GateClass:
    """One recall gate of `PR1_GATES`: a class of dev probes and its idle recall floor.

    Every class counts only probes with at least one expected item, the probes recall is
    defined on; the gate's own predicates already require that, and `holds` keeps it for a
    class built by hand.
    """

    name: str
    floor: float
    member: Callable[[Probe], bool]

    def holds(self, probe: Probe) -> bool:
        """Whether `probe` is scored in this class."""
        return bool(probe.expected) and self.member(probe)


def _gated_classes(gates: Iterable[Gate]) -> tuple[GateClass, ...]:
    return tuple(
        GateClass(name=gate.name, floor=gate.floors["idle"], member=gate.predicate)
        for gate in gates
        if gate.measure == "recall" and gate.predicate is not None
    )


# Taken from the gate itself, so the probe tunes against the floors and classes it is judged by.
GATED_CLASSES: tuple[GateClass, ...] = _gated_classes(PR1_GATES)


def is_measured(probe: Probe, classes: Sequence[GateClass] = GATED_CLASSES) -> bool:
    """Whether the probe enters the selection: a noise probe, or one in a gated class."""
    return probe.kind == "noise" or any(gate.holds(probe) for gate in classes)


# -- the score matrices ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class ProbeScores:
    """One probe's candidates under one fact view and one note view.

    Attributes:
        probe: The dev probe.
        item_ids: `fact:<fact_key>` or `note:<path>`, one per candidate.
        ce: The cross-encoder's probability per candidate.
        cos: The bi-encoder cosine per candidate.
        vector_ranks: A fact's 1-based rank in the vector order of the facts known at `now`;
            None for a note.
        pool_ranks: Each candidate's ranks in the pools that do not depend on the view (fact
            FTS and entity, note vector and FTS), for the fused-order check.
        targets: Per expected item, the index of its candidate; None when it has none (a fact
            with no version current at `now`, a note not yet written, an episode).
    """

    probe: Probe
    item_ids: tuple[str, ...]
    ce: np.ndarray
    cos: np.ndarray
    vector_ranks: tuple[int | None, ...]
    pool_ranks: tuple[Mapping[str, int], ...]
    targets: tuple[int | None, ...]


@dataclass(frozen=True, slots=True)
class ViewPair:
    """Every measured probe under one fact view and one note view."""

    fact_view: str
    note_view: str
    probes: tuple[ProbeScores, ...]


@dataclass(frozen=True, slots=True)
class Setting:
    """One combination: the views, the scorer and, for BI, MAX and BLEND, s0 and s1."""

    fact_view: str
    note_view: str
    scorer: str
    s0: float | None = None
    s1: float | None = None
    weight: float | None = None

    def to_json(self) -> dict[str, object]:
        """The setting as the output records it."""
        return {
            "fact_view": self.fact_view,
            "note_view": self.note_view,
            "scorer": self.scorer,
            "s0": self.s0,
            "s1": self.s1,
            "weight": self.weight,
        }


def similarity_term(cos: np.ndarray, s0: float, s1: float) -> np.ndarray:
    """`b(sim) = clamp((sim - s0) / (s1 - s0), 0, 1)`, the design's bi-encoder relevance."""
    return np.clip((cos - s0) / (s1 - s0), 0.0, 1.0)


def relevance(scores: ProbeScores, setting: Setting) -> np.ndarray:
    """Each candidate's relevance under `setting`'s scorer.

    Raises:
        ValueError: The scorer is unknown, or a similarity scorer has no s0 and s1.
    """
    if setting.scorer == CE:
        return scores.ce
    if setting.s0 is None or setting.s1 is None:
        raise ValueError(f"scorer {setting.scorer} needs s0 and s1")
    b = similarity_term(scores.cos, setting.s0, setting.s1)
    if setting.scorer == BI:
        return b
    if setting.scorer == MAX:
        return np.maximum(scores.ce, b)
    if setting.scorer == BLEND and setting.weight is not None:
        return setting.weight * scores.ce + (1.0 - setting.weight) * b
    raise ValueError(f"unknown scorer {setting.scorer!r}")


@dataclass(frozen=True, slots=True)
class ClassResult:
    """One gated class under one setting."""

    name: str
    floor: float
    probes: int
    figure: float
    margin: float


@dataclass(frozen=True, slots=True)
class Outcome:
    """One setting's t*, class figures and margin."""

    setting: Setting
    t_star: float
    classes: tuple[ClassResult, ...]
    margin: float
    worst_class: str

    def to_json(self) -> dict[str, object]:
        """The outcome as the output records it."""
        return {
            **self.setting.to_json(),
            "t_star": self.t_star,
            "margin": self.margin,
            "worst_class": self.worst_class,
            "classes": {
                result.name: {
                    "figure": result.figure,
                    "floor": result.floor,
                    "margin": result.margin,
                    "probes": result.probes,
                }
                for result in self.classes
            },
        }


def _share_passing(scores: ProbeScores, values: np.ndarray, t_star: float) -> float:
    """The share of the probe's expected items whose candidate is above t*."""
    passed = sum(1 for t in scores.targets if t is not None and values[t] > t_star)
    return passed / len(scores.targets)


def evaluate(
    pair: ViewPair, setting: Setting, classes: Sequence[GateClass] = GATED_CLASSES
) -> Outcome:
    """t*, every class's figure and margin, and the worst margin of one setting.

    t* is the highest relevance of any candidate of any noise probe, 0 when no noise probe has
    a candidate, so no noise probe holds anything above it.

    Raises:
        ValueError: There is no noise probe, or a class has no probe, so the rule is undefined.
    """
    values = [relevance(scores, setting) for scores in pair.probes]
    noise = [v for scores, v in zip(pair.probes, values) if scores.probe.kind == "noise"]
    if not noise:
        raise ValueError("no noise probe: t* is undefined")
    t_star = max((float(v.max()) for v in noise if v.size), default=0.0)
    results: list[ClassResult] = []
    for gate in classes:
        shares = [
            _share_passing(scores, v, t_star)
            for scores, v in zip(pair.probes, values)
            if gate.holds(scores.probe)
        ]
        if not shares:
            raise ValueError(f"class {gate.name!r} has no probe")
        figure = sum(shares) / len(shares)
        results.append(
            ClassResult(
                name=gate.name,
                floor=gate.floor,
                probes=len(shares),
                figure=figure,
                margin=round(figure - gate.floor, _MARGIN_DECIMALS),
            )
        )
    worst = min(results, key=lambda result: result.margin)
    return Outcome(
        setting=setting,
        t_star=t_star,
        classes=tuple(results),
        margin=worst.margin,
        worst_class=worst.name,
    )


def settings_for(
    pair: ViewPair,
    grid: Sequence[tuple[float, float]] = GRID,
    blend_weights: Sequence[float] = BLEND_WEIGHTS,
) -> list[Setting]:
    """Every setting of one view pair, in the tie-break order: CE, BI, MAX, then the blends."""
    views = {"fact_view": pair.fact_view, "note_view": pair.note_view}
    settings = [Setting(scorer=CE, **views)]
    for scorer in (BI, MAX):
        settings.extend(Setting(scorer=scorer, s0=s0, s1=s1, **views) for s0, s1 in grid)
    settings.extend(
        Setting(scorer=BLEND, s0=s0, s1=s1, weight=weight, **views)
        for weight in blend_weights
        for s0, s1 in grid
    )
    return settings


def band(margin: float) -> tuple[str, int]:
    """The winner's band and exit code: proceed (0), flag (3) or stop (1)."""
    if margin >= PROCEED_MARGIN:
        return "proceed", EXIT_PROCEED
    if margin >= 0:
        return "flag", EXIT_FLAG
    return "stop", EXIT_STOP


@dataclass(frozen=True, slots=True)
class KChoice:
    """The fact vector pool depth K and the coverage it was read from.

    Attributes:
        value: The chosen K.
        flagged: No step reached the coverage (or no fact target passed), so K is the last
            step.
        targets: Fact targets above t* the coverage counts.
        coverage: Per step, the share of those targets within that rank; None with no target.
    """

    value: int
    flagged: bool
    targets: int
    coverage: Mapping[int, float | None]

    def to_json(self) -> dict[str, object]:
        """The choice as the output records it."""
        return {
            "value": self.value,
            "flag": self.flagged,
            "targets": self.targets,
            "coverage": {str(step): share for step, share in self.coverage.items()},
        }


def choose_k(ranks: Sequence[int]) -> KChoice:
    """The smallest step holding `K_COVERAGE` of `ranks`; the last step, flagged, otherwise."""
    count = len(ranks)
    coverage: dict[int, float | None] = {
        step: (sum(1 for rank in ranks if rank <= step) / count if count else None)
        for step in K_STEPS
    }
    for step in K_STEPS:
        share = coverage[step]
        if share is not None and share >= K_COVERAGE:
            return KChoice(value=step, flagged=False, targets=count, coverage=coverage)
    return KChoice(value=K_STEPS[-1], flagged=True, targets=count, coverage=coverage)


def fused_positions(scores: ProbeScores, k: int) -> list[int | None]:
    """Each candidate's 1-based position in the RRF order, None when no pool returned it.

    The vector pool is the view's vector order cut at `k`; the other pools are the recorded
    ones. Equal RRF scores are ordered by item id, as `ranking.fuse` orders them.
    """
    keyed: list[tuple[float, str, int]] = []
    for index, item_id in enumerate(scores.item_ids):
        ranks = dict(scores.pool_ranks[index])
        vector_rank = scores.vector_ranks[index]
        if vector_rank is not None and vector_rank <= k:
            ranks[VECTOR_POOL] = vector_rank
        if ranks:
            keyed.append((-rrf_score(ranks, k=RRF_K), item_id, index))
    positions: list[int | None] = [None] * len(scores.item_ids)
    for position, (_, _, index) in enumerate(sorted(keyed), start=1):
        positions[index] = position
    return positions


@dataclass(frozen=True, slots=True)
class Selection:
    """The rule's result over every view pair."""

    outcomes: tuple[Outcome, ...]
    winner: Outcome
    band: str
    exit_code: int
    tied: int
    best_by_scorer: Mapping[str, Outcome]
    blend_flag: bool
    k: KChoice
    ce_only: Mapping[str, object]


def _first_best(outcomes: Iterable[Outcome]) -> Outcome | None:
    best: Outcome | None = None
    for outcome in outcomes:
        if best is None or outcome.margin > best.margin:
            best = outcome
    return best


def _passing_targets(
    pair: ViewPair, outcome: Outcome, classes: Sequence[GateClass]
) -> Iterable[tuple[ProbeScores, int]]:
    """Each (probe, candidate) of a gated probe's expected items that passes under `outcome`."""
    for scores in pair.probes:
        if not any(gate.holds(scores.probe) for gate in classes):
            continue
        values = relevance(scores, outcome.setting)
        for target in scores.targets:
            if target is not None and values[target] > outcome.t_star:
                yield scores, target


def _ce_only(
    pair: ViewPair, outcome: Outcome, k: int, classes: Sequence[GateClass]
) -> dict[str, object]:
    """The share of cross-encoder-only passes inside the first 20, 30 and 50 fused."""
    setting = outcome.setting
    positions: list[int | None] = []
    for scores in pair.probes:
        if not any(gate.holds(scores.probe) for gate in classes):
            continue
        if setting.s0 is not None and setting.s1 is not None:
            b = similarity_term(scores.cos, setting.s0, setting.s1)
        else:
            b = np.zeros_like(scores.ce)
        fused: list[int | None] | None = None
        for target in scores.targets:
            if target is None:
                continue
            if scores.ce[target] > outcome.t_star and b[target] <= outcome.t_star:
                if fused is None:
                    fused = fused_positions(scores, k)
                positions.append(fused[target])
    count = len(positions)
    result: dict[str, object] = {
        "passes": count,
        "unfused": sum(1 for position in positions if position is None),
    }
    for depth in FUSED_DEPTHS:
        inside = sum(1 for position in positions if position is not None and position <= depth)
        result[str(depth)] = inside / count if count else None
    return result


def select(
    pairs: Sequence[ViewPair],
    classes: Sequence[GateClass] = GATED_CLASSES,
    grid: Sequence[tuple[float, float]] = GRID,
    blend_weights: Sequence[float] = BLEND_WEIGHTS,
) -> Selection:
    """Apply H13's rule to every setting of every view pair.

    Raises:
        ValueError: No view pair, or a pair the rule is undefined on (see `evaluate`).
    """
    if not pairs:
        raise ValueError("no view pair to select from")
    outcomes = [
        evaluate(pair, setting, classes)
        for pair in pairs
        for setting in settings_for(pair, grid, blend_weights)
    ]
    chosen = [o for o in outcomes if o.setting.scorer in CHOSEN_SCORERS]
    winner = _first_best(chosen)
    if winner is None:
        raise ValueError("no setting to choose from")
    best_by_scorer: dict[str, Outcome] = {}
    for scorer in (*CHOSEN_SCORERS, BLEND):
        best = _first_best(o for o in outcomes if o.setting.scorer == scorer)
        if best is not None:
            best_by_scorer[scorer] = best
    blend = best_by_scorer.get(BLEND)
    best_max = best_by_scorer.get(MAX)
    blend_flag = (
        blend is not None
        and best_max is not None
        and round(blend.margin - best_max.margin, _MARGIN_DECIMALS) > BLEND_FLAG_MARGIN
    )
    pair = next(
        p
        for p in pairs
        if (p.fact_view, p.note_view) == (winner.setting.fact_view, winner.setting.note_view)
    )
    fact_ranks: list[int] = []
    for scores, target in _passing_targets(pair, winner, classes):
        rank = scores.vector_ranks[target]
        if rank is not None:
            fact_ranks.append(rank)
    k = choose_k(fact_ranks)
    name, code = band(winner.margin)
    return Selection(
        outcomes=tuple(outcomes),
        winner=winner,
        band=name,
        exit_code=code,
        tied=sum(1 for o in chosen if o.margin == winner.margin),
        best_by_scorer=best_by_scorer,
        blend_flag=blend_flag,
        k=k,
        ce_only=_ce_only(pair, winner, k.value, classes),
    )


# -- measuring the fixture -------------------------------------------------------------------


class PassageEmbedder:
    """The bi-encoder on passages: the passage prefix prepended, each text embedded once.

    The sidecar embeds chunks through this object as well, so the vectors it stores are the
    ones the probe's own cosines use, and the identity check compares like with like.

    Args:
        inner: The embedder under test.
        prefix: The passage prefix; empty for an embedder that documents none.
    """

    def __init__(self, inner: EmbeddingProvider, prefix: str = "") -> None:
        self._inner = inner
        self._prefix = prefix
        self._vectors: dict[str, np.ndarray] = {}

    def generate_embedding(self, text: str) -> list[float]:
        """The passage vector of `text`, as the sidecar asks for it."""
        return self.vector(text).tolist()

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """The passage vectors of `texts`, the missing ones embedded in one batch."""
        self.embed_all(texts)
        return [self._vectors[text].tolist() for text in texts]

    def embed_all(self, texts: Iterable[str]) -> None:
        """Embed every text not yet embedded, in one batch."""
        missing = [text for text in dict.fromkeys(texts) if text not in self._vectors]
        if not missing:
            return
        vectors = self._inner.generate_embeddings([self._prefix + text for text in missing])
        for text, vector in zip(missing, vectors, strict=True):
            self._vectors[text] = np.asarray(vector, dtype=np.float64)

    def vector(self, text: str) -> np.ndarray:
        """The passage vector of `text`, embedded now when it is new."""
        if text not in self._vectors:
            self._vectors[text] = np.asarray(
                self._inner.generate_embedding(self._prefix + text), dtype=np.float64
            )
        return self._vectors[text]

    def norms(self) -> list[float]:
        """The norm of every vector of a non-blank text embedded so far."""
        return [float(np.linalg.norm(v)) for text, v in self._vectors.items() if text.strip()]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / norm) if norm > 0 else 0.0


@dataclass(frozen=True, slots=True)
class _Chunk:
    path: str
    heading: str | None
    content: str


@dataclass(slots=True)
class _ProbeParts:
    """One probe's measurements, per fact view and per note view, before they are paired."""

    probe: Probe
    fact_keys: list[str]
    fact_ce: dict[str, np.ndarray]
    fact_cos: dict[str, np.ndarray]
    fact_vector_ranks: dict[str, list[int]]
    fact_pools: list[dict[str, int]]
    paths: list[str]
    note_ce: dict[str, np.ndarray]
    note_cos: np.ndarray
    note_pools: list[dict[str, int]]

    def scores(self, fact_view: str, note_view: str) -> ProbeScores:
        """The probe's candidates under one view pair: its facts first, then its notes."""
        item_ids = tuple(
            [_FACT_PREFIX + key for key in self.fact_keys]
            + [_NOTE_PREFIX + path for path in self.paths]
        )
        where = {item_id: n for n, item_id in enumerate(item_ids)}
        targets: list[int | None] = []
        for ref in self.probe.expected:
            prefix = {"fact": _FACT_PREFIX, "note": _NOTE_PREFIX}.get(ref.kind)
            targets.append(where.get(prefix + ref.id) if prefix is not None else None)
        return ProbeScores(
            probe=self.probe,
            item_ids=item_ids,
            ce=np.concatenate([self.fact_ce[fact_view], self.note_ce[note_view]]),
            cos=np.concatenate([self.fact_cos[fact_view], self.note_cos]),
            vector_ranks=(*self.fact_vector_ranks[fact_view], *([None] * len(self.paths))),
            pool_ranks=(*self.fact_pools, *self.note_pools),
            targets=tuple(targets),
        )


@dataclass(frozen=True, slots=True)
class Measured:
    """The score matrices and the reported checks of one run."""

    pairs: tuple[ViewPair, ...]
    probes: int
    unit_norms: Mapping[str, object]
    sidecar_identity: Mapping[str, object]


def _score(reranker: Reranker, query: str, texts: Sequence[str]) -> np.ndarray:
    if not texts:
        return np.zeros(0)
    return np.asarray(reranker.score(query, list(texts)), dtype=np.float64)


class _Measurer:
    """Builds the indexes at each `now` and measures each probe there."""

    def __init__(
        self,
        fixture: Fixture,
        *,
        embedder: EmbeddingProvider,
        reranker: Reranker,
        fact_views: Sequence[str],
        note_views: Sequence[str],
        tz: tzinfo,
        query_prefix: str,
        passage_prefix: str,
    ) -> None:
        self._fixture = fixture
        self._embedder = embedder
        self._reranker = reranker
        self._fact_views = tuple(fact_views)
        self._note_views = tuple(note_views)
        self._query_prefix = query_prefix
        self.passages = PassageEmbedder(embedder, passage_prefix)
        self._rows: dict[str, FactRow] = {fact.row_key: fact_row(fact) for fact in fixture.facts}
        self._view_text = {
            (view, row_key): render_fact_view(row, FACT_VIEWS[view], tz)
            for view in self._fact_views
            for row_key, row in self._rows.items()
        }
        self.passages.embed_all(self._view_text.values())
        self._titles: dict[str, str] = {}
        for note in fixture.notes:
            self._titles.setdefault(note.path, note.title)
        self.query_norms: list[float] = []
        self.identity_errors: list[float] = []

    def run(self, probes: Sequence[Probe], workdir: Path, tz: tzinfo) -> dict[str, _ProbeParts]:
        """Bring the indexes to each `now` in turn, as the run harness does, and measure there.

        A note path is indexed once, when it first exists, as `run_fixture` indexes it.
        """
        index_path = workdir / "memory_index.db"
        writer = open_index(index_path, embedding_model_id=_FTS_INDEX_MODEL_ID)
        reader = open_reader(index_path, embedding_model_id=_FTS_INDEX_MODEL_ID)
        sidecar = VaultSidecarIndex(
            SidecarIndexConfig(db_path=str(workdir / "vault_sidecar.db")), self.passages
        )
        sidecar.initialize()
        parts: dict[str, _ProbeParts] = {}
        try:
            index = FactIndex(writer, TokenEmbedder(), tz=tz, reader=reader)
            indexed: set[str] = set()
            for now in sorted({probe.now for probe in probes}):
                known = facts_known_at(self._fixture.facts, self._fixture.corrections, now)
                index.diff_apply(known)
                for path, note in notes_known_at(self._fixture.notes, now).items():
                    if path not in indexed:
                        sidecar.upsert_file(path, note.content, note.mtime, note.frontmatter)
                        indexed.add(path)
                for probe in (p for p in probes if p.now == now):
                    parts[probe.probe_id] = self._probe(probe, now, known, index, sidecar)
        finally:
            sidecar.close()
            reader.close()
            writer.close()
        return parts

    def _probe(
        self,
        probe: Probe,
        now: datetime,
        known: Sequence[FactRow],
        index: FactIndex,
        sidecar: VaultSidecarIndex,
    ) -> _ProbeParts:
        query = np.asarray(
            self._embedder.generate_embedding(self._query_prefix + probe.query), dtype=np.float64
        )
        self.query_norms.append(float(np.linalg.norm(query)))

        by_key: dict[str, list[FactRow]] = {}
        for row in known:
            by_key.setdefault(row.fact_key, []).append(row)
        current: dict[str, str] = {}
        for key in sorted(by_key):
            version = _current(index.versions(key), now)
            if version is not None:
                current[key] = version.row_key
        keys = list(current)
        searched = (
            (FTS_POOL, index.search_fts(probe.query, limit=DEFAULT_FACT_POOL_SIZE)),
            (ENTITY_POOL, index.search_names([probe.query], limit=DEFAULT_FACT_POOL_SIZE)),
        )
        fact_pools: list[dict[str, int]] = [{} for _ in keys]
        slot = {key: n for n, key in enumerate(keys)}
        for pool, groups in searched:
            for rank, group in enumerate(groups, start=1):
                if group.fact_key in slot:
                    fact_pools[slot[group.fact_key]][pool] = rank
        fact_ce: dict[str, np.ndarray] = {}
        fact_cos: dict[str, np.ndarray] = {}
        fact_vector_ranks: dict[str, list[int]] = {}
        for view in self._fact_views:
            texts = [self._view_text[(view, current[key])] for key in keys]
            fact_ce[view] = _score(self._reranker, probe.query, texts)
            fact_cos[view] = np.asarray(
                [_cosine(query, self.passages.vector(text)) for text in texts], dtype=np.float64
            )
            best = {
                key: max(
                    _cosine(query, self.passages.vector(self._view_text[(view, row.row_key)]))
                    for row in rows
                )
                for key, rows in by_key.items()
            }
            order = sorted((-score, key) for key, score in best.items())
            rank_of = {key: n for n, (_, key) in enumerate(order, start=1)}
            fact_vector_ranks[view] = [rank_of[key] for key in keys]

        chunks = self._chunks(query, sidecar)
        paths = sorted({chunk.path for chunk in chunks})
        position = {path: n for n, path in enumerate(paths)}
        note_cos = np.full(len(paths), -1.0)
        for chunk in chunks:
            n = position[chunk.path]
            note_cos[n] = max(note_cos[n], _cosine(query, self.passages.vector(chunk.content)))
        note_ce: dict[str, np.ndarray] = {}
        for view in self._note_views:
            texts = [
                render_note_view(
                    self._titles.get(chunk.path), chunk.heading, chunk.content, NOTE_VIEWS[view]
                )
                for chunk in chunks
            ]
            per_chunk = _score(self._reranker, probe.query, texts)
            best_ce = np.zeros(len(paths))
            for chunk, value in zip(chunks, per_chunk, strict=True):
                n = position[chunk.path]
                best_ce[n] = max(best_ce[n], float(value))
            note_ce[view] = best_ce
        note_ranks = self._note_pools(query, probe.query, sidecar)
        return _ProbeParts(
            probe=probe,
            fact_keys=keys,
            fact_ce=fact_ce,
            fact_cos=fact_cos,
            fact_vector_ranks=fact_vector_ranks,
            fact_pools=fact_pools,
            paths=paths,
            note_ce=note_ce,
            note_cos=note_cos,
            note_pools=[note_ranks.get(path, {}) for path in paths],
        )

    def _chunks(self, query: np.ndarray, sidecar: VaultSidecarIndex) -> list[_Chunk]:
        """Every non-empty chunk, nearest first; the top ones check the sidecar identity."""
        count = sidecar.chunk_count()
        if count == 0:
            return []
        rows = sidecar.query_vector(query.tolist(), k=count)
        chunks: list[_Chunk] = []
        for row in rows:
            content = str(row.get("content") or "")
            if not content.strip():
                continue
            heading = row.get("heading")
            chunk = _Chunk(
                path=str(row["path"]),
                heading=str(heading) if heading is not None else None,
                content=content,
            )
            if len(chunks) < _IDENTITY_SAMPLE:
                display = float(row["score"])
                distance = 1.0 / display - 1.0
                identity = 1.0 - distance * distance / 2.0
                direct = _cosine(query, self.passages.vector(content))
                self.identity_errors.append(abs(identity - direct))
            chunks.append(chunk)
        return chunks

    def _note_pools(
        self, query: np.ndarray, text: str, sidecar: VaultSidecarIndex
    ) -> dict[str, dict[str, int]]:
        """The note pool after T50: one entry per path, its best rank in each leg, `k` paths."""
        lexical = " ".join(query_tokens(text))
        rows = sidecar.query_hybrid(query.tolist(), lexical, k=DEFAULT_NOTE_K * 2, match="any")
        ranks: dict[str, dict[str, int]] = {}
        for row in rows:
            if not str(row.get("content") or "").strip():
                continue
            path = str(row["path"])
            if path not in ranks:
                if len(ranks) >= DEFAULT_NOTE_K:
                    continue
                ranks[path] = {}
            for pool, key in ((NOTE_VECTOR_POOL, "vector_rank"), (NOTE_FTS_POOL, "fts_rank")):
                value = row.get(key)
                if isinstance(value, int):
                    ranks[path][pool] = min(value, ranks[path].get(pool, value))
        return ranks


def _norm_summary(norms: Sequence[float]) -> dict[str, object]:
    if not norms:
        return {"vectors": 0, "min": None, "max": None, "max_deviation": None}
    return {
        "vectors": len(norms),
        "min": min(norms),
        "max": max(norms),
        "max_deviation": max(abs(norm - 1.0) for norm in norms),
    }


def measure(
    fixture: Fixture,
    *,
    embedder: EmbeddingProvider,
    reranker: Reranker,
    fact_views: Sequence[str],
    note_views: Sequence[str],
    workdir: Path,
    tz: tzinfo,
    query_prefix: str = "",
    passage_prefix: str = "",
    classes: Sequence[GateClass] = GATED_CLASSES,
) -> Measured:
    """Score every measured dev probe under every fact view and note view.

    Args:
        fixture: The recall fixture; only its dev probes, facts, notes and corrections are read.
        embedder: The bi-encoder under test.
        reranker: The cross-encoder under test.
        fact_views: Fact view ids, in the tie-break order.
        note_views: Note view ids, in the tie-break order.
        workdir: An empty directory for the fact index and the sidecar.
        tz: Zone the fact views date a timestamp bound in, the index's.
        query_prefix: Prepended to each query before it is embedded.
        passage_prefix: Prepended to each fact view and chunk before it is embedded.
        classes: The gated classes; a probe in none of them, and not noise, is skipped.
    """
    probes = [probe for probe in fixture.probes if is_measured(probe, classes)]
    measurer = _Measurer(
        fixture,
        embedder=embedder,
        reranker=reranker,
        fact_views=fact_views,
        note_views=note_views,
        tz=tz,
        query_prefix=query_prefix,
        passage_prefix=passage_prefix,
    )
    parts = measurer.run(probes, workdir, tz)
    pairs = tuple(
        ViewPair(
            fact_view=fact_view,
            note_view=note_view,
            probes=tuple(parts[probe.probe_id].scores(fact_view, note_view) for probe in probes),
        )
        for fact_view in fact_views
        for note_view in note_views
    )
    errors = measurer.identity_errors
    return Measured(
        pairs=pairs,
        probes=len(probes),
        unit_norms={
            "passages": _norm_summary(measurer.passages.norms()),
            "queries": _norm_summary(measurer.query_norms),
        },
        sidecar_identity={
            "sampled": len(errors),
            "max_abs_error": max(errors) if errors else None,
            "mean_abs_error": (sum(errors) / len(errors)) if errors else None,
        },
    )


# -- the command line ------------------------------------------------------------------------


def model_dir(text: str) -> Path:
    """`text` as a model directory, refused unless it lies under `/opt/models`.

    The path is normalised first, so `..` cannot climb out of the root, and the root itself is
    not a model. As in the run harness, a model never loads from anywhere that could download.

    Raises:
        ValueError: The path is not under `/opt/models`.
    """
    path = PurePosixPath(posixpath.normpath(text))
    root = PurePosixPath(MODELS_ROOT.as_posix())
    if root not in path.parents:
        raise ValueError(f"models load from a directory under {root} only, got {text}")
    return Path(str(path))


def load_models(
    embedder_path: Path, reranker_path: Path, *, device: Device
) -> tuple[EmbeddingProvider, Reranker]:
    """The embedder and the cross-encoder, each from a local directory.

    The directory check comes first: `SentenceTransformer` reads a path that does not exist as
    a hub model name, and would try the network for it.

    Raises:
        ValueError: A path is not a directory.
        MemoryRerankError: The cross-encoder fails to load.
        OSError: The embedder fails to load.
    """
    for path in (embedder_path, reranker_path):
        if not path.is_dir():
            raise ValueError(f"model directory {path} does not exist")
    embedder = SentenceTransformerEmbedder(str(embedder_path), device=device)
    reranker = CrossEncoderReranker(
        str(reranker_path), device=device, cross_encoder_cls=CrossEncoder
    )
    return embedder, reranker


def _views(table: Mapping[str, object]) -> Callable[[str], tuple[str, ...]]:
    def parse(text: str) -> tuple[str, ...]:
        names = tuple(dict.fromkeys(part.strip() for part in text.split(",") if part.strip()))
        unknown = [name for name in names if name not in table]
        if not names or unknown:
            raise argparse.ArgumentTypeError(
                f"expected a comma-separated list of {', '.join(table)}, got {text!r}"
            )
        return names

    return parse


def build_parser() -> argparse.ArgumentParser:
    """The command line; it has no probe-file option, so only the dev set can be read."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--embedder", required=True, help="bi-encoder directory under /opt/models")
    parser.add_argument(
        "--reranker", required=True, help="cross-encoder directory under /opt/models"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--fact-views", type=_views(FACT_VIEWS), default=tuple(FACT_VIEWS), help="e.g. F0,F2"
    )
    parser.add_argument(
        "--note-views", type=_views(NOTE_VIEWS), default=tuple(NOTE_VIEWS), help="e.g. N0,N1"
    )
    parser.add_argument("--output", type=Path, required=True, help="result to write (JSON)")
    parser.add_argument(
        "--query-prefix", default="", help="prepended to each query before it is embedded"
    )
    parser.add_argument(
        "--passage-prefix",
        default="",
        help="prepended to each fact view and note chunk before it is embedded",
    )
    return parser


def report(selection: Selection, measured: Measured, args: argparse.Namespace) -> dict[str, object]:
    """The output JSON: inputs, every combination, the winner, its band, K and the checks."""
    flags: list[str] = []
    if selection.band == "flag":
        flags.append(f"margin {selection.winner.margin} is under {PROCEED_MARGIN}")
    if selection.band == "stop":
        flags.append(f"margin {selection.winner.margin} is below 0: stop")
    if selection.k.flagged:
        flags.append(f"no K step holds {K_COVERAGE} of the fact targets: K is {selection.k.value}")
    if selection.blend_flag:
        flags.append(f"the blend beats MAX by more than {BLEND_FLAG_MARGIN} of margin")
    return {
        "probe_set": "dev",
        "probes": measured.probes,
        "embedder": str(args.embedder),
        "reranker": str(args.reranker),
        "device": args.device,
        "query_prefix": args.query_prefix,
        "passage_prefix": args.passage_prefix,
        "fact_views": list(args.fact_views),
        "note_views": list(args.note_views),
        "grid": {"s0": [0.30, 0.70], "s1_max": 0.90, "step": 0.05, "settings": len(GRID)},
        "blend_weights": list(BLEND_WEIGHTS),
        "combinations": [outcome.to_json() for outcome in selection.outcomes],
        "winner": selection.winner.to_json(),
        "tied": selection.tied,
        "band": selection.band,
        "exit_code": selection.exit_code,
        "best_by_scorer": {
            scorer: outcome.to_json() for scorer, outcome in selection.best_by_scorer.items()
        },
        "blend_flag": selection.blend_flag,
        "k": selection.k.to_json(),
        "checks": {
            "ce_only_in_fused": selection.ce_only,
            "unit_norms": measured.unit_norms,
            "sidecar_identity": measured.sidecar_identity,
        },
        "flags": flags,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run H13's rule on the dev set; the exit code is the winner's band, or 2 on bad input."""
    args = build_parser().parse_args(argv)
    try:
        embedder_path = model_dir(args.embedder)
        reranker_path = model_dir(args.reranker)
        tz = MemoryConfig.from_env().tz
        embedder, reranker = load_models(embedder_path, reranker_path, device=args.device)
        dimension = len(embedder.generate_embedding(args.query_prefix + "dimension check"))
        if dimension != EMBEDDING_DIMENSION:
            raise ValueError(
                f"the embedder gives {dimension}-d vectors; the indexes hold "
                f"{EMBEDDING_DIMENSION}-d ones"
            )
    except (ValueError, OSError, MemoryRerankError, MemoryConfigError) as exc:
        print(f"memory_view_probe: {exc}", file=sys.stderr)
        return EXIT_INPUT
    fixture = Fixture.load()
    with tempfile.TemporaryDirectory(prefix="memory-view-probe-") as workdir:
        measured = measure(
            fixture,
            embedder=embedder,
            reranker=reranker,
            fact_views=args.fact_views,
            note_views=args.note_views,
            workdir=Path(workdir),
            tz=tz,
            query_prefix=args.query_prefix,
            passage_prefix=args.passage_prefix,
        )
    selection = select(measured.pairs)
    result = report(selection, measured, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    winner = selection.winner
    print(
        f"winner {winner.setting.fact_view}/{winner.setting.note_view} {winner.setting.scorer} "
        f"s0={winner.setting.s0} s1={winner.setting.s1} t*={winner.t_star:.4f} "
        f"margin={winner.margin} ({winner.worst_class}); band {selection.band}; "
        f"K={selection.k.value}{' (flag)' if selection.k.flagged else ''}"
    )
    print(f"wrote {args.output}", file=sys.stderr)
    return selection.exit_code


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "BI",
    "BLEND",
    "CE",
    "EXIT_FLAG",
    "EXIT_INPUT",
    "EXIT_PROCEED",
    "EXIT_STOP",
    "GATED_CLASSES",
    "GRID",
    "K_STEPS",
    "MAX",
    "ClassResult",
    "GateClass",
    "KChoice",
    "Measured",
    "Outcome",
    "PassageEmbedder",
    "ProbeScores",
    "Selection",
    "Setting",
    "ViewPair",
    "band",
    "build_parser",
    "choose_k",
    "evaluate",
    "fused_positions",
    "is_measured",
    "load_models",
    "main",
    "measure",
    "model_dir",
    "relevance",
    "report",
    "select",
    "settings_for",
    "similarity_term",
]
