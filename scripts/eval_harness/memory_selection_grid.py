"""Choose the need gate and the retrieval setting on the dev set by a rule fixed before the run.

H18 (plan mist-read-path v8, H18's rule) runs this once, with the real models, over the dev
probes (`probes.jsonl`, read through `Fixture.load`) and the dev follow-up probes
(`followup_probes.jsonl`, read by `load_followups`). It has no probe-file option, so it is never
pointed at the held-out set.

Step 1, the gate alone, per gate embedder (`shared`, which is the fact embedder, and each
`--need-embedder`): the bank is embedded with the embedder's query prefix; each form candidate is
cross-validated by writer (`grouped_cv`) and the form chosen by `choose_form`; a candidate whose
fit reaches the iteration cap is reported and skipped, in cross-validation and in the final fit
on both writers alike. The fitted gate gives `p` for every dev probe and follow-up text by the
call the fetcher makes. The variants are `R` in {0.95, 0.98} by `band_k` in {1, 2}, with
`theta_low = theta(R)` and `theta_high = theta(R - 0.10)` over the dev probes of the PR 1 recall
classes; a variant is eligible only when every dev noise probe has `p < theta_low`. The preferred
embedder is `shared` unless another's cross-validation score is at least 0.05 above `shared`'s
and above it in each fold at each target (the highest such).

Step 2: each retrieval setting (fact view by `select_k` by `rerank_top_n` by facets) runs once
per gate embedder with an eligible variant, with that embedder's gate in `shadow` mode: one
`run_settings` call per fact view and embedder, with `warm_up`. The shadow gate's thresholds are
placeholders; step 3 replaces them.

Step 3: each retrieval setting with each eligible variant of its embedder is a setting, its
records derived from the shadow run by `derive_record`, and scored with `PR1_GATES` through
`judge` (memory_recall_gate.py) plus empty_on_noise 1.0 and no `need_unknown`. The choice is
`choose`; the chosen setting runs again with its gate `on` under load (`load_check`), with the
`need_context` its gate's follow-up counts give, and a setting that misses a load gate is struck
and the rule chooses again, up to three strikes.

Exit 0: the choice has margin at least 0.05. Exit 3: the choice has margin under 0.05, to be
flagged to the lead. Exit 1: no setting passes (as when every candidate of every gate embedder
failed, which leaves no setting), or the load check struck three settings or every passing one.
Exit 2: an input error, found before any model loads (`main` lists them). Exit 4: a model
directory that exists fails to load; one stderr line names the directory and the error, and no
report is written. Any other error still ends in a traceback with Python's exit 1.

Usage, from the repository root (`/app` in the image):
    python scripts/eval_harness/memory_selection_grid.py --embedder PATH --reranker PATH
        [--need-embedder NAME=PATH@REVISION ...] --device cuda|cpu --embedder-device cpu|cuda
        --output PATH [--need-model-output PATH]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, tzinfo
from functools import partial
from pathlib import Path
from typing import Literal, TypeVar

# Run as a script path (H18's command), `sys.path` starts at this directory, not the repository
# root the absolute imports below need.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from safetensors import SafetensorError  # noqa: E402

from backend.errors import MemoryConfigError, MemoryRerankError, MistError  # noqa: E402
from backend.interfaces import EmbeddingProvider  # noqa: E402
from backend.knowledge.embeddings.embedding_generator import EmbeddingGenerator  # noqa: E402
from backend.memory.config import MemoryConfig  # noqa: E402
from backend.memory.need_gate import (  # noqa: E402
    FORM_CANDIDATES,
    NEED_EMBEDDERS,
    NORMALISATION,
    CvResult,
    FormSpec,
    NeedBank,
    NeedFitError,
    NeedGate,
    NeedKey,
    NeedModel,
    auroc,
    choose_form,
    context_text,
    cv_score,
    fit_form,
    grouped_cv,
    load_bank,
    need_embedder_kwargs,
    need_model_key,
    theta,
)
from backend.memory.rerank import Reranker  # noqa: E402
from backend.memory.temporal import parse_window  # noqa: E402
from scripts.check_models_lock import LockError, parse_lock  # noqa: E402
from scripts.eval_harness.memory_recall import (  # noqa: E402
    FIXTURE_DIR,
    ItemRef,
    Probe,
    parse_run_record,
    score_run,
)
from scripts.eval_harness.memory_recall_gate import (  # noqa: E402
    PR1_GATES,
    FetchLatency,
    judge,
    nearest_rank_p95,
)
from scripts.eval_harness.memory_recall_run import (  # noqa: E402
    DEFAULT_MODELS_LOCK,
    MODELS_ROOT,
    Fixture,
    ForcedRescan,
    RunSetting,
    SentenceTransformerEmbedder,
    TokenEmbedder,
    build_reranker,
    cpu_load,
    rescan_template,
    run_settings,
)

FOLLOWUP_FILE = "followup_probes.jsonl"
SHARED = "shared"

FACT_VIEWS_GRID = ("F2", "F3")
SELECT_KS = (5, 8, 10, 15)
RERANK_TOP_NS = (20, 30, 50)
FACETS = (True, False)
RECALL_TARGETS = (0.95, 0.98)
BAND_KS = (1, 2)
# `theta_high` keeps the top share `R - BAND_WIDTH` of the need probes open.
BAND_WIDTH = 0.10
MIN_MARGIN = 0.05
PREFERENCE_MARGIN = 0.05
MAX_STRIKES = 3
LOAD_CPU_PROCS = 8
LOAD_RESCAN_FACTS = 2000

EXIT_GO = 0
EXIT_STOP = 1
EXIT_INPUT = 2
EXIT_FLAG = 3
EXIT_LOAD = 4

# What a model directory that exists can raise while it loads: rerank.py's `_MODEL_ERRORS`, an
# ImportError from a missing optional package, and the MemoryRerankError `CrossEncoderReranker`
# turns those into. A TypeError or any other programming error is not caught.
_LOAD_ERRORS: tuple[type[Exception], ...] = (
    OSError,
    ValueError,
    RuntimeError,
    ImportError,
    SafetensorError,
    MemoryRerankError,
)

# A difference of two shares can land a rounding step under a bound it meets exactly.
_TOLERANCE = 1e-9
_EMPTY_STAGES = ("fused", "shortlisted", "selected")
_PLACEHOLDER_THETA = 0.5

Band = Literal["closed", "band", "open"]

# Everything not on the grid, for every setting: the plan's fixed values, and the gate fields a
# setting overrides. `MemoryConfig()` reads no environment, so a stray variable moves nothing.
BASE_CONFIG = MemoryConfig(
    block_tokens=500, list_max_items=25, deadline_ms=50, need_gate="off", need_context=False
)


# --- inputs --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Followup:
    """One follow-up probe: a short turn and the turn before it, labelled as the bank is."""

    followup_id: str
    label: Literal["need", "no_need"]
    previous: str
    query: str


def load_followups(path: Path = FIXTURE_DIR / FOLLOWUP_FILE) -> tuple[Followup, ...]:
    """Read the follow-up probes, one JSON object of `id`, `label`, `previous`, `query` a line.

    Raises:
        ValueError: A malformed row, naming its line.
        OSError: The file cannot be read.
    """
    rows: list[Followup] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{path.name}:{number}"
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{where}: not valid JSON ({exc.msg})") from exc
        if not isinstance(row, dict) or set(row) != {"id", "label", "previous", "query"}:
            raise ValueError(f"{where}: a row is an object of id, label, previous and query")
        if row["label"] not in ("need", "no_need"):
            raise ValueError(f"{where}: label {row['label']!r} is not need or no_need")
        if not all(isinstance(row[k], str) and row[k].strip() for k in ("id", "previous", "query")):
            raise ValueError(f"{where}: id, previous and query are non-empty strings")
        rows.append(Followup(row["id"], row["label"], row["previous"], row["query"]))
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class GridPoint:
    """One retrieval setting: the grid's four dimensions."""

    fact_view: str
    select_k: int
    rerank_top_n: int
    facets: bool


GRID: tuple[GridPoint, ...] = tuple(
    GridPoint(view, k, n, facets)
    for view in FACT_VIEWS_GRID
    for k in SELECT_KS
    for n in RERANK_TOP_NS
    for facets in FACETS
)


def grid_config(
    base: MemoryConfig,
    point: GridPoint,
    *,
    need_gate: Literal["on", "shadow"],
    need_embedder: str,
    need_context: bool,
) -> MemoryConfig:
    """`base` with the point's values and the gate's mode, embedder name and `need_context`."""
    return replace(
        base,
        fact_view=point.fact_view,
        select_k=point.select_k,
        rerank_top_n=point.rerank_top_n,
        facets=point.facets,
        need_gate=need_gate,
        need_embedder=need_embedder,
        need_context=need_context,
    )


@dataclass(frozen=True, slots=True)
class GateEmbedder:
    """A gate embedder and what its gate's key names it by.

    Attributes:
        name: Its `NEED_EMBEDDERS` name, which gives the query prefix; `shared` is the fact
            embedder, whose query vector the fetch shares with the gate.
        key_name: The embedder's name in the gate's key. `load_real_models` gives `shared` the
            fact embedder's models.lock name and every other embedder its `NEED_EMBEDDERS` name.
        revision: The embedder's revision in the key.
        embedder: The model.

    Raises:
        ValueError: A name outside `NEED_EMBEDDERS`.
    """

    name: str
    key_name: str
    revision: str
    embedder: EmbeddingProvider

    def __post_init__(self) -> None:
        if self.name not in NEED_EMBEDDERS:
            raise ValueError(f"{self.name!r} is not a NEED_EMBEDDERS name")

    @property
    def query_prefix(self) -> str:
        """The text put before every bank row and query, from the embedder's spec."""
        return NEED_EMBEDDERS[self.name].query_prefix

    @property
    def shares_query_vector(self) -> bool:
        """Whether the gate reads the fetch's own query vector: `shared` only."""
        return self.name == SHARED


@dataclass(frozen=True, slots=True)
class Models:
    """What the grid runs: the fact embedder, the reranker and the gate embedders, `shared` first."""

    fact_embedder: EmbeddingProvider
    reranker: Reranker
    gate_embedders: tuple[GateEmbedder, ...]


# --- step 1: the gate alone ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Variant:
    """One gate variant: a recall target, its two thresholds, the band's cap and eligibility."""

    recall_target: float
    band_k: int
    theta_low: float
    theta_high: float
    eligible: bool


@dataclass(frozen=True, slots=True)
class FollowupScore:
    """A follow-up's probabilities: its previous turn, its query and the two joined."""

    followup_id: str
    label: Literal["need", "no_need"]
    p_previous: float
    p_query: float
    p_context: float


@dataclass(frozen=True, slots=True)
class GateFit:
    """Step 1 for one gate embedder.

    Attributes:
        embedder: The gate embedder.
        bank: The bank it was fit on.
        cv: Every candidate's cross-validation, in `FORM_CANDIDATES` order.
        cv_failed: The candidates whose fit failed in cross-validation.
        fit_failed: The candidates cross-validation chose whose fit on both writers failed, in
            the order they were tried.
        form: The candidate fit on both writers; None when every candidate failed.
        params: Its fitted parameters; None with no form.
        gate: The shadow gate, with placeholder thresholds; None with no form.
        p: Each dev probe's probability, by probe id; empty with no form.
        variants: The four variants; empty with no form.
        followups: Each follow-up's probabilities; empty with no form.
    """

    embedder: GateEmbedder
    bank: NeedBank
    cv: tuple[CvResult, ...]
    cv_failed: tuple[FormSpec, ...]
    fit_failed: tuple[FormSpec, ...]
    form: FormSpec | None
    params: Mapping[str, object] | None
    gate: NeedGate | None
    p: Mapping[str, float]
    variants: tuple[Variant, ...]
    followups: tuple[FollowupScore, ...]

    @property
    def failed(self) -> tuple[FormSpec, ...]:
        """Every candidate skipped for a failed fit: cross-validation's, then the final fit's."""
        return self.cv_failed + self.fit_failed

    @property
    def chosen_cv(self) -> CvResult | None:
        """The chosen form's cross-validation; None with no form."""
        return next((r for r in self.cv if r.spec == self.form), None)


def is_pr1_recall(probe: Probe) -> bool:
    """Whether the probe is in a PR 1 recall class of `PR1_GATES`: the need probes of the rule."""
    return any(
        gate.predicate(probe)
        for gate in PR1_GATES
        if gate.measure == "recall" and gate.predicate is not None
    )


def band_of(p: float, theta_low: float, theta_high: float) -> Band:
    """`closed` below `theta_low`, `band` from it up to `theta_high`, `open` at and above that."""
    if p < theta_low:
        return "closed"
    if p < theta_high:
        return "band"
    return "open"


def gate_variants(p: Mapping[str, float], probes: Sequence[Probe]) -> tuple[Variant, ...]:
    """The four variants of one gate, from its dev probabilities.

    `theta_low = theta(R)` and `theta_high = theta(R - 0.10)`, both over the PR 1 recall
    probes; a variant is eligible only when every noise probe has `p < theta_low`. `theta`
    falls as its share rises, so `theta_high` is never below `theta_low`.

    Raises:
        ValueError: `theta` refuses the set (none, or a threshold that equals a value).
    """
    need = [p[probe.probe_id] for probe in probes if is_pr1_recall(probe)]
    noise = [p[probe.probe_id] for probe in probes if probe.kind == "noise"]
    variants: list[Variant] = []
    for target in RECALL_TARGETS:
        low = theta(need, target)
        high = theta(need, round(target - BAND_WIDTH, 2))
        eligible = all(value < low for value in noise)
        variants.extend(Variant(target, k, low, high, eligible) for k in BAND_KS)
    return tuple(variants)


def _model(
    embedder: GateEmbedder,
    bank: NeedBank,
    form: FormSpec,
    params: Mapping[str, object],
    *,
    theta_low: float,
    theta_high: float,
    band_k: int,
    recall_target: float,
    source: str,
) -> NeedModel:
    key = NeedKey(bank.sha256, embedder.key_name, embedder.revision, embedder.query_prefix)
    return NeedModel(
        key=need_model_key(key),
        bank_sha256=key.bank_sha256,
        embedder_name=key.embedder_name,
        embedder_revision=key.embedder_revision,
        query_prefix=key.query_prefix,
        normalisation=NORMALISATION,
        form=form.form,
        params=params,
        theta_low=theta_low,
        theta_high=theta_high,
        band_k=band_k,
        recall_target=recall_target,
        source=source,
    )


def _gate(embedder: GateEmbedder, bank: NeedBank, model: NeedModel) -> NeedGate:
    return NeedGate(
        embedder.embedder,
        bank,
        model,
        embedder_name=embedder.key_name,
        embedder_revision=embedder.revision,
        query_prefix=embedder.query_prefix,
        shares_query_vector=embedder.shares_query_vector,
    )


def _p(gate: NeedGate, text: str) -> float:
    """`p` by the fetcher's own call.

    A gate that shares the fetch's query vector reads the fact embedder's vector of the bare
    text; any other gate embeds the text with its prefix.
    """
    vector = (
        gate.embedder.generate_embedding(text)
        if gate.shares_query_vector
        else gate.embed_query(text)
    )
    return gate.decide(vector).p


def fit_gate(
    embedder: GateEmbedder,
    bank: NeedBank,
    probes: Sequence[Probe],
    followups: Sequence[Followup],
    *,
    fit: Callable[[object, Sequence[object], FormSpec], dict[str, object]] = fit_form,
) -> GateFit:
    """Step 1 for one gate embedder: choose the form, fit it, and score the dev texts.

    The form is `choose_form` over the candidates whose cross-validation did not fail. The
    chosen one is fit on both writers with `fit`; when that fit fails, the candidate is skipped
    and the form is chosen again from those left. Each dev probe's `p` and each follow-up's
    previous turn and query are scored as the fetcher scores a query (`_p`); the follow-up's
    joined text through the gate's own embedding, as the fetcher scores the context.

    Args:
        fit: The final fit; `fit_form`, which tests replace.
    """
    exemplars = bank.exemplars
    vectors = embedder.embedder.generate_embeddings(
        [embedder.query_prefix + exemplar.text for exemplar in exemplars]
    )
    labels = [exemplar.label for exemplar in exemplars]
    cv = grouped_cv(
        vectors,
        labels,
        [exemplar.writer for exemplar in exemplars],
        [exemplar.hard for exemplar in exemplars],
        FORM_CANDIDATES,
    )
    cv_failed = tuple(result.spec for result in cv if result.failed)
    remaining = [result for result in cv if not result.failed]
    no_need_rows = sum(1 for label in labels if label == "no_need")
    fit_failed: list[FormSpec] = []
    form: FormSpec | None = None
    params: dict[str, object] | None = None
    while remaining:
        spec = choose_form(remaining, no_need_rows=no_need_rows)
        try:
            params = fit(vectors, labels, spec)
        except NeedFitError:
            fit_failed.append(spec)
            remaining = [result for result in remaining if result.spec != spec]
            continue
        form = spec
        break
    if form is None or params is None:
        return GateFit(
            embedder, bank, cv, cv_failed, tuple(fit_failed), None, None, None, {}, (), ()
        )
    shadow = _model(
        embedder,
        bank,
        form,
        params,
        theta_low=_PLACEHOLDER_THETA,
        theta_high=_PLACEHOLDER_THETA,
        band_k=BAND_KS[0],
        recall_target=RECALL_TARGETS[0],
        source="memory_selection_grid shadow placeholder",
    )
    gate = _gate(embedder, bank, shadow)
    p = {probe.probe_id: _p(gate, probe.query) for probe in probes}
    scores = tuple(
        FollowupScore(
            followup_id=row.followup_id,
            label=row.label,
            p_previous=_p(gate, row.previous),
            p_query=_p(gate, row.query),
            p_context=gate.decide(gate.embed_query(context_text(row.previous, row.query))).p,
        )
        for row in followups
    )
    return GateFit(
        embedder=embedder,
        bank=bank,
        cv=cv,
        cv_failed=cv_failed,
        fit_failed=tuple(fit_failed),
        form=form,
        params=params,
        gate=gate,
        p=p,
        variants=gate_variants(p, probes),
        followups=scores,
    )


def variant_model(fit: GateFit, variant: Variant, *, source: str) -> NeedModel:
    """The fit's gate model with the variant's thresholds, band cap and recall target.

    Raises:
        ValueError: The fit has no form.
    """
    if fit.form is None or fit.params is None:
        raise ValueError(f"{fit.embedder.name} has no fitted form")
    return _model(
        fit.embedder,
        fit.bank,
        fit.form,
        fit.params,
        theta_low=variant.theta_low,
        theta_high=variant.theta_high,
        band_k=variant.band_k,
        recall_target=variant.recall_target,
        source=source,
    )


def build_gate(fit: GateFit, model: NeedModel) -> NeedGate:
    """A gate on the fit's embedder and bank with `model`; it embeds the bank once, here."""
    return _gate(fit.embedder, fit.bank, model)


def need_model_record(fit: GateFit, variant: Variant, *, source: str) -> dict[str, object]:
    """The variant's gate model as a record of need_model.json, which `load_models` reads."""
    model = variant_model(fit, variant, source=source)
    return {
        "key": model.key,
        "bank_sha256": model.bank_sha256,
        "embedder": {"name": model.embedder_name, "revision": model.embedder_revision},
        "query_prefix": model.query_prefix,
        "normalisation": model.normalisation,
        "form": model.form,
        "params": dict(model.params),
        "theta_low": model.theta_low,
        "theta_high": model.theta_high,
        "band_k": model.band_k,
        "recall_target": model.recall_target,
        "source": model.source,
    }


def preferred_embedder(cv: Mapping[str, CvResult]) -> str | None:
    """The preferred gate embedder from each embedder's chosen form's cross-validation.

    `shared`, unless another's score (`cv_score`) is at least 0.05 above `shared`'s and its
    closure is above `shared`'s in each fold at each target; among such, the highest score,
    the first in `cv`'s order on a tie. None when `cv` has no `shared` entry (its every
    candidate failed), so the choice's first rule yields nothing.
    """
    base = cv.get(SHARED)
    if base is None:
        return None
    base_score = cv_score(base)
    best: tuple[str, float] | None = None
    for name, result in cv.items():
        if name == SHARED:
            continue
        score = cv_score(result)
        if score - base_score < PREFERENCE_MARGIN - _TOLERANCE:
            continue
        above = all(
            mine > theirs
            for my_fold, their_fold in zip(result.closure, base.closure, strict=True)
            for mine, theirs in zip(my_fold, their_fold, strict=True)
        )
        if above and (best is None or score > best[1]):
            best = (name, score)
    return SHARED if best is None else best[0]


def common_set(
    p: Mapping[str, Mapping[str, float]],
    variants: Mapping[str, Sequence[Variant]],
    probes: Sequence[Probe],
) -> frozenset[str]:
    """The PR 1 recall probes every eligible variant of every embedder serves (band or open).

    Args:
        p: Per embedder, each probe's probability.
        variants: Per embedder, its variants; the ineligible ones are left out.
    """
    ids = [probe.probe_id for probe in probes if is_pr1_recall(probe)]
    kept = set(ids)
    for name, own in variants.items():
        for variant in own:
            if variant.eligible:
                kept &= {i for i in ids if p[name][i] >= variant.theta_low}
    return frozenset(kept)


def _variant_json(variant: Variant) -> dict[str, object]:
    return {
        "recall_target": variant.recall_target,
        "band_k": variant.band_k,
        "theta_low": variant.theta_low,
        "theta_high": variant.theta_high,
    }


def gate_report(
    p: Mapping[str, float], variants: Sequence[Variant], probes: Sequence[Probe]
) -> dict[str, object]:
    """What one gate's recorded probabilities show, choosing nothing.

    `auroc` is the PR 1 recall probes against the noise probes (None without both). Per
    variant: its thresholds, eligibility, the share of each probe kind and of the PR 1 recall
    probes (`pr1_recall`) it closes, bands and opens, and the share of PR 1 recall probes among
    the probes it bands (None when it bands none).
    """
    need = [p[probe.probe_id] for probe in probes if is_pr1_recall(probe)]
    noise = [p[probe.probe_id] for probe in probes if probe.kind == "noise"]
    value = (
        auroc(need + noise, [True] * len(need) + [False] * len(noise)) if need and noise else None
    )
    classes: dict[str, list[Probe]] = {}
    for probe in probes:
        classes.setdefault(probe.kind, []).append(probe)
    classes["pr1_recall"] = [probe for probe in probes if is_pr1_recall(probe)]
    entries: list[dict[str, object]] = []
    for variant in variants:
        bands = {
            probe.probe_id: band_of(p[probe.probe_id], variant.theta_low, variant.theta_high)
            for probe in probes
        }
        shares = {
            name: {
                band: sum(1 for probe in members if bands[probe.probe_id] == band) / len(members)
                for band in ("closed", "band", "open")
            }
            for name, members in classes.items()
            if members
        }
        banded = [probe for probe in probes if bands[probe.probe_id] == "band"]
        entries.append(
            {
                **_variant_json(variant),
                "eligible": variant.eligible,
                "shares": shares,
                "banded_need_share": (
                    sum(1 for probe in banded if is_pr1_recall(probe)) / len(banded)
                    if banded
                    else None
                ),
            }
        )
    return {"auroc": value, "variants": entries}


@dataclass(frozen=True, slots=True)
class FollowupCounts:
    """What the context path does for one gate variant, over the follow-up probes.

    Attributes:
        a: The need follow-ups the context moves out of closed.
        b: The no-need follow-ups it moves out of closed.
        closed_previous: The follow-ups whose previous turn the gate closes, so the previous
            fetch served no memory and the context path does not run.
    """

    a: int
    b: int
    closed_previous: tuple[str, ...]

    @property
    def need_context(self) -> bool:
        """On when the context opens more need follow-ups than no-need ones."""
        return self.a > self.b


def followup_counts(scores: Sequence[FollowupScore], variant: Variant) -> FollowupCounts:
    """A and B for one variant, from the gate alone.

    A follow-up counts when its previous turn is served (band or open), its query alone is
    closed, and the higher of the query's and the joined text's probabilities is not.
    """
    a = b = 0
    closed: list[str] = []
    for score in scores:
        if band_of(score.p_previous, variant.theta_low, variant.theta_high) == "closed":
            closed.append(score.followup_id)
            continue
        if band_of(score.p_query, variant.theta_low, variant.theta_high) != "closed":
            continue
        turn = max(score.p_query, score.p_context)
        if band_of(turn, variant.theta_low, variant.theta_high) == "closed":
            continue
        if score.label == "need":
            a += 1
        else:
            b += 1
    return FollowupCounts(a, b, tuple(closed))


# --- step 2: shadow runs -------------------------------------------------------------------------

ShadowRunner = Callable[[GateFit, str, Sequence[GridPoint]], list[list[dict[str, object]]]]
LoadRunner = Callable[[RunSetting], list[dict[str, object]]]


def fixture_shadow_runner(
    fixture: Fixture,
    *,
    fact_embedder: TokenEmbedder | SentenceTransformerEmbedder,
    reranker: Reranker,
    base: MemoryConfig,
    workdir: Path,
    embedder_device: Literal["cpu", "cuda"] = "cpu",
) -> ShadowRunner:
    """Step 2's runner: the given points of one fact view, with the fit's gate in shadow mode.

    Each call is one `run_settings` walk of every dev probe, with `warm_up`, in a fresh
    directory under `workdir`, one setting per point.
    """

    def run(fit: GateFit, view: str, points: Sequence[GridPoint]) -> list[list[dict[str, object]]]:
        if fit.gate is None:
            raise ValueError(f"{fit.embedder.name} has no gate to run in shadow mode")
        if any(point.fact_view != view for point in points):
            raise ValueError(f"every point of one shadow run has fact view {view}")
        settings = [
            RunSetting(
                grid_config(
                    base,
                    point,
                    need_gate="shadow",
                    need_embedder=fit.embedder.name,
                    need_context=False,
                ),
                fit.gate,
            )
            for point in points
        ]
        with tempfile.TemporaryDirectory(prefix="selection-grid-", dir=workdir) as directory:
            return asyncio.run(
                run_settings(
                    fixture,
                    settings=settings,
                    embedder=fact_embedder,
                    reranker=reranker,
                    workdir=Path(directory),
                    probes=None,
                    embedder_device=embedder_device,
                    warm_up=True,
                    followups=(),
                )
            )

    return run


def fixture_load_runner(
    fixture: Fixture,
    followups: Sequence[Followup],
    *,
    fact_embedder: TokenEmbedder | SentenceTransformerEmbedder,
    reranker: Reranker,
    workdir: Path,
    embedder_device: Literal["cpu", "cuda"] = "cpu",
) -> LoadRunner:
    """The load check's runner: one gated setting under H4's load, over the dev probes and the
    follow-ups.

    The load is `LOAD_CPU_PROCS` busy-loop processes and forced rescans of
    `LOAD_RESCAN_FACTS` facts for the whole walk; the walk has `warm_up`, and each follow-up
    runs as its two fetches in its own session.
    """

    def run(setting: RunSetting) -> list[dict[str, object]]:
        rescan = ForcedRescan(LOAD_RESCAN_FACTS, rescan_template(fixture))
        triples = [(row.followup_id, row.previous, row.query) for row in followups]
        with (
            cpu_load(LOAD_CPU_PROCS),
            tempfile.TemporaryDirectory(prefix="selection-load-", dir=workdir) as directory,
        ):
            [records] = asyncio.run(
                run_settings(
                    fixture,
                    settings=[setting],
                    embedder=fact_embedder,
                    reranker=reranker,
                    workdir=Path(directory),
                    cpu_load_procs=LOAD_CPU_PROCS,
                    embedder_device=embedder_device,
                    rescan=rescan,
                    warm_up=True,
                    followups=triples,
                )
            )
        return records

    return run


# --- step 3: derivation and scoring --------------------------------------------------------------


def derive_record(record: Mapping[str, object], variant: Variant) -> dict[str, object]:
    """The record a gated (`on`) fetch with the variant's gate would write, from a shadow one.

    - A shadow gate that was late or failed (`need.degraded`) closes the gated fetch with that
      reason: no block, nothing reranked or shortlisted, no stage ids, `degraded_reason` the
      gate's reason.
    - A variant that closes the probe serves no block and is not degraded.
    - A variant that bands it serves the shadow fetch's cut of `band_k` (items and tokens), or
      no block when the fetch built none.
    - An open probe keeps the shadow fetch's block.

    The need trace takes mode `on`, the band and the variant's thresholds and `band_k`; the
    cuts are dropped; `delta_tokens_est` moves with the block's tokens; the latency fields stay
    the shadow fetch's. The input is not changed.

    Raises:
        ValueError: The record has no shadow need trace.
    """
    need = record.get("need")
    if not isinstance(need, dict) or need.get("mode") != "shadow":
        raise ValueError(f"{record.get('probe_id')}: a derived record needs a shadow need trace")
    degraded = need["degraded"]
    p = need["p"]
    if degraded is None and not isinstance(p, float | int):
        raise ValueError(f"{record.get('probe_id')}: an undegraded shadow trace has no p")
    band: Band = (
        "closed"
        if degraded is not None
        else band_of(float(p), variant.theta_low, variant.theta_high)
    )
    derived = {key: value for key, value in record.items() if key != "cuts"}
    derived["need"] = {
        **need,
        "mode": "on",
        "band": band,
        "theta_low": variant.theta_low,
        "theta_high": variant.theta_high,
        "band_k": variant.band_k,
    }
    open_tokens = int(record["tokens"])
    if band == "closed":
        derived.update(
            items=[],
            tokens=0,
            reranked=0,
            shortlisted=0,
            degraded_reason=degraded,
            stage_ids={stage: [] for stage in _EMPTY_STAGES},
            delta_tokens_est=int(record["delta_tokens_est"]) - open_tokens,
        )
    elif band == "band":
        cuts = record.get("cuts")
        cut = cuts.get(str(variant.band_k)) if isinstance(cuts, dict) else None
        items, tokens = (list(cut["items"]), int(cut["tokens"])) if cut else ([], 0)
        derived.update(
            items=items,
            tokens=tokens,
            delta_tokens_est=int(record["delta_tokens_est"]) - open_tokens + tokens,
        )
    return derived


@dataclass(frozen=True, slots=True)
class Setting:
    """A retrieval setting with one gate variant of one embedder."""

    point: GridPoint
    embedder: str
    variant: Variant


def setting_json(setting: Setting) -> dict[str, object]:
    """The setting's values as the report holds them."""
    return {
        "fact_view": setting.point.fact_view,
        "select_k": setting.point.select_k,
        "rerank_top_n": setting.point.rerank_top_n,
        "facets": setting.point.facets,
        "embedder": setting.embedder,
        **_variant_json(setting.variant),
    }


@dataclass(frozen=True, slots=True)
class Scored:
    """One setting judged on the dev set.

    Attributes:
        passed: Every condition held (`failures` is empty).
        margin: The smallest recall class figure less its idle floor; None when a class has no
            figure.
        cost: The mean block tokens over the common set; None when it is empty.
        failures: The conditions it missed, by gate name, then `need_unknown`.
        report: Its entry in the report's `settings`.
    """

    setting: Setting
    passed: bool
    margin: float | None
    cost: float | None
    failures: tuple[str, ...]
    report: Mapping[str, object] = field(default_factory=dict)


def _latency(record: Mapping[str, object]) -> FetchLatency:
    return FetchLatency(
        critical_path_ms=float(record["critical_path_ms"]),
        degraded_reason=record["degraded_reason"],
        reranked=int(record["reranked"]),
        shortlisted=int(record["shortlisted"]),
    )


def _need_unknown(record: Mapping[str, object]) -> bool:
    need = record.get("need")
    traced = isinstance(need, dict) and need.get("degraded") == "need_unknown"
    return record.get("degraded_reason") == "need_unknown" or traced


def _staged(ref: ItemRef, ids: Sequence[str]) -> bool:
    # A note candidate is one chunk, `note:<path>#<heading>`; the probe names the path.
    if ref.kind == "note":
        return any(entry.startswith(f"note:{ref.id}#") for entry in ids)
    return f"{ref.kind}:{ref.id}" in ids


def _stage_shares(
    members: Sequence[Probe], by_id: Mapping[str, Mapping[str, object]]
) -> dict[str, object]:
    """The share of the class's expected items each stage reached, over all its probes."""
    counts = {"fused": 0, "shortlisted": 0, "selected": 0, "placed": 0}
    targets = 0
    for probe in members:
        record = by_id[probe.probe_id]
        stages = record.get("stage_ids")
        stages = stages if isinstance(stages, dict) else {}
        placed = {(item["kind"], item["id"]) for item in record["items"]}
        for ref in probe.expected:
            targets += 1
            for stage in _EMPTY_STAGES:
                counts[stage] += _staged(ref, stages.get(stage, []))
            counts["placed"] += (ref.kind, ref.id) in placed
    shares: dict[str, object] = {"targets": targets}
    shares.update({k: (v / targets if targets else None) for k, v in counts.items()})
    return shares


def score_setting(
    setting: Setting,
    records: Sequence[Mapping[str, object]],
    probes: Sequence[Probe],
    common: frozenset[str],
    dated: frozenset[str] = frozenset(),
) -> Scored:
    """Judge one setting's dev records with `PR1_GATES` and the rule's two extra conditions.

    Each gate of `PR1_GATES` is judged through `judge` on the dev set, idle: a recall class
    passes when it meets its idle floor (`judge` prints those as reported on dev), every other
    gate by its verdict, and a gate with no figure fails. On top, empty_on_noise must be 1.0
    and no record may be degraded `need_unknown`.

    Args:
        records: One derived record per dev probe.
        common: The common set's probe ids, which the cost is taken over.
        dated: The PR 1 recall probes that name a dated window, reported with their recall.
    """
    run = [parse_run_record(record) for record in records]
    lines = judge(probes, run, [_latency(r) for r in records], probe_set="dev", load="idle")
    figures = {
        line.name: {
            "value": line.figure.value,
            "n": line.figure.n,
            "floor": line.floor,
            "verdict": line.verdict,
        }
        for line in lines
    }
    failures: list[str] = []
    margins: list[float | None] = []
    for gate, line in zip(PR1_GATES, lines):
        value = line.figure.value
        if value is None or line.verdict in ("fail", "reported, below floor"):
            failures.append(gate.name)
        if gate.measure == "recall":
            margins.append(None if value is None else value - gate.floors["idle"])
    noise = figures["empty_on_noise"]["value"]
    if noise != 1.0 and "empty_on_noise" not in failures:
        failures.append("empty_on_noise")
    if any(_need_unknown(record) for record in records):
        failures.append("need_unknown")
    margin = None if any(m is None for m in margins) else min(margins)  # type: ignore[type-var]
    costed = [int(record["tokens"]) for record in records if record["probe_id"] in common]
    cost = sum(costed) / len(costed) if costed else None
    tokens = [int(record["tokens"]) for record in records]
    by_id = {str(record["probe_id"]): record for record in records}
    selected_facet = sum(
        1
        for record in records
        if any(
            str(entry).startswith("facet:")
            for entry in (record.get("stage_ids") or {}).get("selected", [])
        )
    )
    stage_shares = {
        gate.name: _stage_shares([p for p in probes if gate.predicate(p)], by_id)
        for gate in PR1_GATES
        if gate.measure == "recall" and gate.predicate is not None
    }
    dated_probes = [probe for probe in probes if probe.probe_id in dated]
    dated_recall = (
        score_run(dated_probes, [r for r in run if r.probe_id in dated]).overall.recall_at_block
        if dated_probes
        else None
    )
    report = {
        "setting": setting_json(setting),
        "figures": figures,
        "margin": margin,
        "cost": cost,
        "p95_tokens": nearest_rank_p95(tokens) if tokens else None,
        "mean_items": sum(len(r["items"]) for r in records) / len(records) if records else None,
        "facet_share": selected_facet / len(records) if records else None,
        "stage_shares": stage_shares,
        "dated_window_recall": dated_recall,
        "latency_p95_ms": figures["critical_path_p95_ms"]["value"],
        "deadline_share": figures["deadline_degraded_share"]["value"],
        "verdict": "fail" if failures else "pass",
        "failures": failures,
    }
    return Scored(setting, not failures, margin, cost, tuple(failures), report)


def dated_window_probes(probes: Sequence[Probe], tz: tzinfo) -> frozenset[str]:
    """The PR 1 recall probes whose query names a dated window at the probe's `now`."""
    return frozenset(
        probe.probe_id
        for probe in probes
        if is_pr1_recall(probe) and parse_window(probe.query, now=probe.now, tz=tz) is not None
    )


# --- the choice and the load check ---------------------------------------------------------------


def _tie_key(setting: Setting, order: Sequence[str]) -> tuple[object, ...]:
    """The tie-breaks, in order.

    The smaller `select_k`, the smaller `rerank_top_n`, F3, `shared` (then the other embedders
    in `order`), facets off, the larger `R`, the smaller `band_k`.
    """
    point, variant = setting.point, setting.variant
    if setting.embedder == SHARED:
        rank = 0
    else:
        # The plan ranks `shared` first and says nothing about two other embedders that tie;
        # this grid's own rule puts them in `--need-embedder` order, so the result is stable.
        rank = 1 + (order.index(setting.embedder) if setting.embedder in order else len(order))
    return (
        point.select_k,
        point.rerank_top_n,
        0 if point.fact_view == "F3" else 1,
        rank,
        1 if point.facets else 0,
        -variant.recall_target,
        variant.band_k,
    )


def _cost(scored: Scored) -> float:
    return math.inf if scored.cost is None else scored.cost


def choose(results: Sequence[Scored], preferred: str | None, order: Sequence[str]) -> Scored | None:
    """H18's choice: the first rule that yields a setting.

    1. Among passing settings of the preferred embedder with margin at least 0.05, the lowest
       cost.
    2. Among passing settings of any embedder with margin at least 0.05, the lowest cost.
    3. The passing setting with the largest margin.

    Ties go by `_tie_key`. None when no setting passes.

    Args:
        preferred: The preferred embedder; None makes rule 1 yield nothing.
        order: The gate embedders in the grid's order, for the embedder tie-break.
    """
    passing = [r for r in results if r.passed and r.margin is not None]
    wide = [r for r in passing if r.margin >= MIN_MARGIN - _TOLERANCE]
    first = [r for r in wide if r.setting.embedder == preferred]
    for pool in (first, wide):
        if pool:
            return min(pool, key=lambda r: (_cost(r), _tie_key(r.setting, order)))
    if passing:
        return min(passing, key=lambda r: (-r.margin, _tie_key(r.setting, order)))
    return None


@dataclass(frozen=True, slots=True)
class LoadJudgement:
    """The load gates one gated run missed, and its figures."""

    failures: tuple[str, ...]
    figures: Mapping[str, Mapping[str, object]]


def judge_load(records: Sequence[Mapping[str, object]], probes: Sequence[Probe]) -> LoadJudgement:
    """Judge a load run: every `PR1_GATES` gate with a load floor, and no `need_unknown`.

    Records with `probe_id` are the dev probes' fetches, those with `followup_id` the
    follow-ups'. The three latency gates (critical-path P95, deadline share, rerank depth) and
    `need_unknown` are taken over both, because a follow-up fetch is a fetch under the same
    load. The other gates read the dev probes. empty_on_noise must be 1.0, as idle.
    """
    recall = [r for r in records if "probe_id" in r]
    follow = [r for r in records if "followup_id" in r]
    run = [parse_run_record(record) for record in recall]
    latencies = [_latency(r) for r in recall + follow]
    lines = judge(probes, run, latencies, probe_set="dev", load="load")
    failures: list[str] = []
    figures: dict[str, Mapping[str, object]] = {}
    for gate, line in zip(PR1_GATES, lines):
        if "load" not in gate.floors:
            continue
        figures[gate.name] = {"value": line.figure.value, "n": line.figure.n}
        if line.verdict != "pass":
            failures.append(gate.name)
    if figures["empty_on_noise"]["value"] != 1.0 and "empty_on_noise" not in failures:
        failures.append("empty_on_noise")
    unknown = sum(1 for r in recall + follow if _need_unknown(r))
    figures["need_unknown"] = {"value": unknown, "n": len(recall) + len(follow)}
    if unknown:
        failures.append("need_unknown")
    return LoadJudgement(tuple(failures), figures)


@dataclass(frozen=True, slots=True)
class LoadOutcome:
    """The setting the load check kept, its `need_context`, and every check run."""

    choice: Scored | None
    need_context: bool | None
    checks: tuple[Mapping[str, object], ...]


def load_check(
    results: Sequence[Scored],
    preferred: str | None,
    order: Sequence[str],
    *,
    probes: Sequence[Probe],
    need_context: Callable[[Setting], bool],
    run: Callable[[Setting, bool], Sequence[Mapping[str, object]]],
) -> LoadOutcome:
    """Run the chosen setting under load; strike it on a missed gate and choose again.

    Before each check `need_context` gives the chosen setting's value, from its own gate, so a
    strike that moves the choice to another embedder or variant recomputes it. After
    `MAX_STRIKES` strikes, or when `choose` yields nothing, the outcome has no choice.

    Args:
        need_context: The setting's `need_context`.
        run: Runs the setting gated, with that `need_context`, and returns its records.
    """
    remaining = list(results)
    checks: list[Mapping[str, object]] = []
    while len(checks) < MAX_STRIKES:
        choice = choose(remaining, preferred, order)
        if choice is None:
            break
        context = need_context(choice.setting)
        judgement = judge_load(run(choice.setting, context), probes)
        struck = bool(judgement.failures)
        checks.append(
            {
                "setting": setting_json(choice.setting),
                "need_context": context,
                "struck": struck,
                "failures": list(judgement.failures),
                "figures": dict(judgement.figures),
            }
        )
        if not struck:
            return LoadOutcome(choice, context, tuple(checks))
        remaining = [r for r in remaining if r is not choice]
    return LoadOutcome(None, None, tuple(checks))


# --- the grid ------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GridOutcome:
    """The report, the exit code and the chosen gate's model record (None without a choice)."""

    report: dict[str, object]
    exit_code: int
    need_model: dict[str, object] | None


def cv_report(fit: GateFit) -> dict[str, object]:
    """One embedder's cross-validation table: each candidate per fold and target, and the form.

    A candidate's `failed` is `cross-validation`, `final fit` or None.
    """
    rows = []
    for result in fit.cv:
        if result.spec in fit.cv_failed:
            failed: str | None = "cross-validation"
        elif result.spec in fit.fit_failed:
            failed = "final fit"
        else:
            failed = None
        rows.append(
            {
                "form": result.spec.describe(),
                "failed": failed,
                "score": None if result.failed else cv_score(result),
                "closure": [list(fold) for fold in result.closure],
                "hard_closure": [list(fold) for fold in result.hard_closure],
                "auroc": list(result.auroc),
                "log_loss": list(result.log_loss),
                "brier": list(result.brier),
            }
        )
    return {
        "embedder": {"name": fit.embedder.key_name, "revision": fit.embedder.revision},
        "query_prefix": fit.embedder.query_prefix,
        "targets": list(RECALL_TARGETS),
        "candidates": rows,
        "form": None if fit.form is None else fit.form.describe(),
    }


def exit_code(choice: Scored | None) -> int:
    """0 for a choice with margin at least 0.05, 3 for one under it, 1 for no choice."""
    if choice is None or choice.margin is None:
        return EXIT_STOP
    return EXIT_GO if choice.margin >= MIN_MARGIN - _TOLERANCE else EXIT_FLAG


def run_grid(
    fixture: Fixture,
    followups: Sequence[Followup],
    bank: NeedBank,
    embedders: Sequence[GateEmbedder],
    *,
    points: Sequence[GridPoint] = GRID,
    base: MemoryConfig = BASE_CONFIG,
    shadow_runner: ShadowRunner,
    load_runner: LoadRunner,
    fit: Callable[[object, Sequence[object], FormSpec], dict[str, object]] = fit_form,
    source: str | None = None,
) -> GridOutcome:
    """Steps 1 to 3, the choice and the load check, over the dev probes.

    An embedder whose every candidate failed has no gate and no settings; one with no eligible
    variant has no shadow runs. Each other embedder's shadow runs are one `shadow_runner` call
    per fact view, in the order the views first appear in `points`.

    Args:
        points: The retrieval settings; `GRID` for H18.
        base: Everything not on the grid.
        shadow_runner: Step 2's runner.
        load_runner: Runs one gated `RunSetting` under load and returns its dev probe and
            follow-up records.
        fit: The final fit of each embedder's chosen form.
        source: The `need_model` record's `source`; `H18 <today>` when None.
    """
    probes = fixture.probes
    fits = {e.name: fit_gate(e, bank, probes, followups, fit=fit) for e in embedders}
    order = tuple(fits)
    gated = {name: f for name, f in fits.items() if f.gate is not None}
    preferred = preferred_embedder({name: f.chosen_cv for name, f in gated.items()})
    eligible = {name: tuple(v for v in f.variants if v.eligible) for name, f in gated.items()}
    common = common_set(
        {name: f.p for name, f in gated.items()},
        {name: f.variants for name, f in gated.items()},
        probes,
    )
    dated = dated_window_probes(probes, base.tz)
    views = list(dict.fromkeys(point.fact_view for point in points))
    results: list[Scored] = []
    for name, gate_fit in gated.items():
        if not eligible[name]:
            continue
        for view in views:
            own = [point for point in points if point.fact_view == view]
            runs = shadow_runner(gate_fit, view, own)
            for point, records in zip(own, runs, strict=True):
                for variant in eligible[name]:
                    derived = [derive_record(record, variant) for record in records]
                    results.append(
                        score_setting(Setting(point, name, variant), derived, probes, common, dated)
                    )
    counts = {
        (name, variant): followup_counts(gate_fit.followups, variant)
        for name, gate_fit in gated.items()
        for variant in gate_fit.variants
    }
    source = source if source is not None else f"H18 {date.today().isoformat()}"

    def gated_run(setting: Setting, context: bool) -> list[dict[str, object]]:
        gate_fit = gated[setting.embedder]
        config = grid_config(
            base,
            setting.point,
            need_gate="on",
            need_embedder=setting.embedder,
            need_context=context,
        )
        model = variant_model(gate_fit, setting.variant, source=source)
        return load_runner(RunSetting(config, build_gate(gate_fit, model)))

    outcome = load_check(
        results,
        preferred,
        order,
        probes=probes,
        need_context=lambda s: counts[(s.embedder, s.variant)].need_context,
        run=gated_run,
    )
    choice = outcome.choice
    need_model = (
        None
        if choice is None
        else need_model_record(
            gated[choice.setting.embedder], choice.setting.variant, source=source
        )
    )
    reports = {name: gate_report(f.p, f.variants, probes) for name, f in gated.items()}
    report: dict[str, object] = {
        "fixed": {
            "block_tokens": base.block_tokens,
            "list_max_items": base.list_max_items,
            "deadline_ms": base.deadline_ms,
            "shadow_need_context": False,
            "points": [
                {
                    "fact_view": p.fact_view,
                    "select_k": p.select_k,
                    "rerank_top_n": p.rerank_top_n,
                    "facets": p.facets,
                }
                for p in points
            ],
            "recall_targets": list(RECALL_TARGETS),
            "band_ks": list(BAND_KS),
        },
        "cv": {name: cv_report(f) for name, f in fits.items()},
        "preferred": preferred,
        "gates": {
            name: {
                "form": None if f.form is None else f.form.describe(),
                "variants": reports[name]["variants"] if name in reports else [],
            }
            for name, f in fits.items()
        },
        "settings": [scored.report for scored in results],
        "load_checks": list(outcome.checks),
        "followups": {
            name: [
                {
                    **_variant_json(variant),
                    "a": counts[(name, variant)].a,
                    "b": counts[(name, variant)].b,
                    "need_context": counts[(name, variant)].need_context,
                    "closed_previous": list(counts[(name, variant)].closed_previous),
                }
                for variant in f.variants
            ]
            for name, f in gated.items()
        },
        "reported": {
            "auroc": {name: r["auroc"] for name, r in reports.items()},
            "common_set": len(common),
            "dated_window_probes": sorted(dated),
            "skipped_embedders": [name for name in fits if not eligible.get(name)],
        },
        "need_context": outcome.need_context,
        "choice": (
            None
            if choice is None
            else {
                "setting": setting_json(choice.setting),
                "margin": choice.margin,
                "cost": choice.cost,
            }
        ),
        "need_model": need_model,
    }
    return GridOutcome(report, exit_code(choice), need_model)


# --- the command ---------------------------------------------------------------------------------


class GridInputError(ValueError):
    """An input the grid refuses before it runs: exit 2."""


class GridLoadError(MistError):
    """A model directory that exists failed to load: exit 4.

    The message is `<directory>: <type>: <message>`, naming the directory and the error the
    loader raised.
    """


_T = TypeVar("_T")


def _load(directory: str, loader: Callable[[], _T]) -> _T:
    """`loader()`, with any of `_LOAD_ERRORS` turned into GridLoadError naming `directory`."""
    try:
        return loader()
    except _LOAD_ERRORS as exc:
        raise GridLoadError(f"{directory}: {type(exc).__name__}: {exc}") from exc


def _gate_model(name: str, path: str) -> EmbeddingGenerator:
    """A gate embedder on the CPU, as deployed, with its spec's load keywords, loaded now.

    EmbeddingGenerator loads on its first use; `warmup` loads it here, so a directory that
    cannot load fails before the grid runs rather than inside the first fit.
    """
    model = EmbeddingGenerator(path, device="cpu", **need_embedder_kwargs(NEED_EMBEDDERS[name]))
    model.warmup()
    return model


def _under_models_root(path: str) -> bool:
    resolved = Path(path)
    return MODELS_ROOT in (resolved, *resolved.parents)


def _need_embedder_arg(text: str) -> tuple[str, str, str]:
    """`NAME=PATH@REVISION`, the revision after the last `@`."""
    name, sep, rest = text.partition("=")
    path, at, revision = rest.rpartition("@")
    if not sep or not at or not name or not path or not revision:
        raise argparse.ArgumentTypeError(f"{text!r} is not NAME=PATH@REVISION")
    if name == SHARED or name not in NEED_EMBEDDERS:
        names = sorted(n for n in NEED_EMBEDDERS if n != SHARED)
        raise argparse.ArgumentTypeError(f"{name!r} is not one of {names}")
    if not _under_models_root(path):
        raise argparse.ArgumentTypeError(f"{path} is not under {MODELS_ROOT}")
    return name, path, revision


def build_parser() -> argparse.ArgumentParser:
    """The command line; it has no probe-file option."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--embedder", required=True, help="fact embedder directory under /opt/models"
    )
    parser.add_argument(
        "--reranker", required=True, help="cross-encoder directory under /opt/models"
    )
    parser.add_argument(
        "--need-embedder",
        action="append",
        type=_need_embedder_arg,
        metavar="NAME=PATH@REVISION",
        help="a gate embedder beside shared, loaded on the CPU; repeat for each",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), required=True, help="the reranker's")
    parser.add_argument(
        "--embedder-device", choices=("cpu", "cuda"), required=True, help="the fact embedder's"
    )
    parser.add_argument("--output", type=Path, required=True, help="the report to write (JSON)")
    parser.add_argument(
        "--need-model-output",
        type=Path,
        help="the chosen gate's record to write in need_model.json's format",
    )
    return parser


def load_real_models(args: argparse.Namespace, *, models_lock: Path) -> Models:
    """The real models: the fact embedder, the cross-encoder and each gate embedder.

    `shared`'s key names the fact embedder by its models.lock entry, found by its directory's
    name. Each `--need-embedder` loads on the CPU, as deployed, with its spec's load keywords.
    Every model is loaded before this returns: the fact embedder and the cross-encoder load in
    their constructors, and each gate embedder, which would otherwise load on its first use,
    through `warmup`.

    Raises:
        GridInputError: A model directory does not exist, or the lock cannot be read or has no
            entry for the fact embedder.
        GridLoadError: A model directory exists but its model fails to load with one of
            `_LOAD_ERRORS`; the message names the directory and the error. Any other error, a
            TypeError say, is not caught.
    """
    paths = [args.embedder, args.reranker, *(path for _n, path, _r in args.need_embedder or ())]
    for path in paths:
        if not Path(path).is_dir():
            raise GridInputError(f"{path} is not a model directory")
    wanted = Path(args.embedder).name
    try:
        entries = parse_lock(models_lock)
    except LockError as exc:
        raise GridInputError(f"the models lock: {exc}") from exc
    entry = next((e for e in entries if e.name == wanted), None)
    if entry is None:
        raise GridInputError(f"{models_lock} has no entry named {wanted!r}, the fact embedder")
    fact = _load(
        args.embedder,
        lambda: SentenceTransformerEmbedder(args.embedder, device=args.embedder_device),
    )
    built = _load(
        args.reranker,
        lambda: build_reranker(
            "real", device=args.device, config=replace(BASE_CONFIG, rerank_model=args.reranker)
        ),
    )
    gates = [GateEmbedder(SHARED, entry.name, entry.revision, fact)]
    for name, path, revision in args.need_embedder or ():
        model = _load(path, partial(_gate_model, name, path))
        gates.append(GateEmbedder(name, name, revision, model))
    return Models(fact, built.reranker, tuple(gates))


def main(
    argv: Sequence[str] | None = None,
    *,
    models: Callable[[argparse.Namespace], Models] | None = None,
    points: Sequence[GridPoint] = GRID,
    base: MemoryConfig = BASE_CONFIG,
    shadow_runner: ShadowRunner | None = None,
    load_runner: LoadRunner | None = None,
    fixture_dir: Path = FIXTURE_DIR,
    models_lock: Path = DEFAULT_MODELS_LOCK,
) -> int:
    """Run the grid and write the report; return the exit code.

    These exit 2 through `parser.error`, before any model loads: a malformed argument, a model
    path outside /opt/models or not a directory, a gate embedder named twice, dev inputs that
    cannot be read, and a models lock that cannot be read or lacks the fact embedder. A model
    directory that exists but fails to load (`GridLoadError`) exits 4 with one line on stderr,
    `memory_selection_grid: exit 4, <directory>: <type>: <message>`, and writes no report. The
    keywords other than `argv` replace the real models, the grid, the fixed values and the
    runners, for tests.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    for flag, path in (("--embedder", args.embedder), ("--reranker", args.reranker)):
        if not _under_models_root(path):
            parser.error(f"{flag}: {path} is not under {MODELS_ROOT}")
    names = [name for name, _path, _revision in args.need_embedder or ()]
    if len(names) != len(set(names)):
        parser.error(f"--need-embedder names a gate embedder twice: {names}")
    try:
        fixture = Fixture.load(fixture_dir)
        followups = load_followups(fixture_dir / FOLLOWUP_FILE)
        bank = load_bank()
    except (OSError, ValueError, MemoryConfigError) as exc:
        parser.error(f"the dev inputs: {exc}")
    loader = models or (lambda a: load_real_models(a, models_lock=models_lock))
    try:
        loaded = loader(args)
    except GridInputError as exc:
        parser.error(str(exc))
    except GridLoadError as exc:
        print(f"memory_selection_grid: exit {EXIT_LOAD}, {exc}", file=sys.stderr)
        return EXIT_LOAD
    with tempfile.TemporaryDirectory(prefix="selection-grid-") as root:
        workdir = Path(root)
        outcome = run_grid(
            fixture,
            followups,
            bank,
            loaded.gate_embedders,
            points=points,
            base=base,
            shadow_runner=shadow_runner
            or fixture_shadow_runner(
                fixture,
                fact_embedder=loaded.fact_embedder,
                reranker=loaded.reranker,
                base=base,
                workdir=workdir,
                embedder_device=args.embedder_device,
            ),
            load_runner=load_runner
            or fixture_load_runner(
                fixture,
                followups,
                fact_embedder=loaded.fact_embedder,
                reranker=loaded.reranker,
                workdir=workdir,
                embedder_device=args.embedder_device,
            ),
        )
    report = outcome.report
    fixed = report["fixed"]
    assert isinstance(fixed, dict)
    fixed["models"] = {
        "embedder": args.embedder,
        "reranker": args.reranker,
        "device": args.device,
        "embedder_device": args.embedder_device,
        "need_embedders": [
            {"name": name, "path": path, "revision": revision}
            for name, path, revision in args.need_embedder or ()
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.need_model_output is not None and outcome.need_model is not None:
        args.need_model_output.parent.mkdir(parents=True, exist_ok=True)
        args.need_model_output.write_text(
            json.dumps({"records": [outcome.need_model]}, indent=2) + "\n", encoding="utf-8"
        )
    print(
        f"memory_selection_grid: exit {outcome.exit_code}, choice "
        f"{json.dumps(report['choice'], sort_keys=True)}",
        file=sys.stderr,
    )
    return outcome.exit_code


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "BASE_CONFIG",
    "GRID",
    "FollowupCounts",
    "Followup",
    "FollowupScore",
    "GateEmbedder",
    "GateFit",
    "GridInputError",
    "GridLoadError",
    "GridOutcome",
    "GridPoint",
    "LoadJudgement",
    "LoadOutcome",
    "Models",
    "Scored",
    "Setting",
    "Variant",
    "band_of",
    "build_gate",
    "build_parser",
    "choose",
    "common_set",
    "cv_report",
    "dated_window_probes",
    "derive_record",
    "exit_code",
    "fit_gate",
    "fixture_load_runner",
    "fixture_shadow_runner",
    "followup_counts",
    "gate_report",
    "gate_variants",
    "grid_config",
    "is_pr1_recall",
    "judge_load",
    "load_check",
    "load_followups",
    "load_real_models",
    "main",
    "need_model_record",
    "preferred_embedder",
    "run_grid",
    "score_setting",
    "setting_json",
    "variant_model",
]
