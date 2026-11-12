"""Score a MIST extraction run against a hand-labeled gold corpus (F2).

Runs the gold corpus through `mist_admin replay` (which emits a MIST_DEBUG_JSONL
with per-turn extraction `llm_call` records), joins each gold probe to its
produced extraction by utterance text, and computes entity/relationship P/R,
relationship-typing accuracy, RELATED_TO rate, and valid-time accuracy.

Reconciliation-action accuracy is NOT scored: the reconciliation engine (C2)
does not yet emit reconciliation telemetry. A hook reports SKIPPED until then.

Standalone, mirrors scripts/eval_harness/score_v8_probe_run.py.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Running this script by path puts scripts/eval_harness/ on sys.path[0], NOT the
# repo root -- so `import backend` would fail and typing accuracy would silently
# report 0.000. Add the repo root first (mirrors scripts/mist_admin.py:51-53).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Intentionally NOT guarded: a missing backend must fail loudly, never silently
# zero the typing metric.
from backend.knowledge.curation.reconciliation import (  # noqa: E402
    derive_assertion_kind,
)

# The rail runner and the id rails the normalizer runs, imported rather than copied, so a
# scored id is the id the graph stores.
from backend.knowledge.extraction import rails  # noqa: E402

# `canonical_id` lives in the backend so the graph and the scorer share it; it is
# re-exported here for the eval scripts and tests that import it from this module.
from backend.knowledge.extraction.canonical_id import (  # noqa: E402
    canonical_id,
)
from backend.knowledge.extraction.id_rails import (  # noqa: E402
    ISO_DATE_RE,
    stored_entity_id,
)
from backend.knowledge.extraction.validator import (  # noqa: E402
    RELATIONSHIP_CONSTRAINTS,
)

# The ontology names this scorer needs (the edge that dates an entity, the escape edge,
# the abstract cluster) are the active pack's, read when a run is scored.
from backend.knowledge.ontologies import registry  # noqa: E402
from backend.knowledge.ontologies.hierarchy import (  # noqa: E402
    canonical_type,
)
from backend.knowledge.ontologies.pack import ID as _ID_PHASE  # noqa: E402
from scripts.eval_harness import governance  # noqa: E402

# `types_match` lives with the other id form rules and is re-exported here for its callers.
from scripts.eval_harness.entity_alignment import (  # noqa: E402
    ALIGNMENT_KINDS,
    EXCLUSION_EXEMPT_KINDS,
    PENDING_KINDS,
    AlignedPair,
    EntityRef,
    align_entities,
    candidate_pairs,
    match_relations,
    types_match,
)
from scripts.golden_log.native_shape import (  # noqa: E402
    GOLD_PREDICATE_FIELD,
    GOLD_SOURCE_FIELD,
    GOLD_TARGET_FIELD,
    GOLD_VALID_FROM_FIELD,
    GOLD_VALID_TO_FIELD,
    native_endpoints,
    native_entity_type,
    native_predicate,
    native_properties,
)

_ISO_DATE_RE = ISO_DATE_RE

EXTRACTION_CALL_SITE = "extraction.ontology"
EXTRACTION_UTTERANCE_PATTERN = re.compile(r'Utterance:\s*"(.+?)"\s*\n\s*Output:', re.DOTALL)


def _produced_entity_match_id(raw_id: str, etype: str, properties: dict[str, Any]) -> str:
    """The id the graph stores for one entity: `stored_entity_id`, the normalizer's steps.

    The per-entity id rails only: the reserved names, the slug of the id, the version
    rail, the aliases, the registry, the Metric id (from `value` with `unit` or `quantity`,
    else from the raw id) and the N1 rail. The steps are idempotent on a stored id, so a
    caller holding a graph id (the retrieval scorer) gets it back unchanged. The
    Event-date rail needs the whole payload, so `parse_produced` runs it.
    """
    return stored_entity_id(raw_id, etype, properties)


def _norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True, slots=True)
class GoldEntity:
    """A gold-labeled entity: canonical id plus ontology type."""

    id: str
    type: str
    # Other canonical ids gold accepts for this entity, where the labelling rules admit two
    # forms; only the aligned scoring reads them.
    alternatives: tuple[str, ...] = ()


VALID_ASSERTION_KINDS = ("assert", "cease", "retract")


@dataclass(frozen=True, slots=True)
class GoldRel:
    """A gold-labeled relationship with optional valid-time bounds.

    `assertion_kind` records the temporal-assertion semantics of the gold edge:
    `assert` (default) states the fact holds, `cease` states it stopped holding
    (optionally bounded by `valid_to`), `retract` states it was never true.
    Bucket scoring on this label lands in a later C3 task; the loader validates
    it here so an unknown value fails loudly at corpus-author time.

    `predicate_alternatives` are other predicates gold accepts for the edge, read only by
    the aligned scoring. `beyond_prompt` marks an edge that a labelling rule beyond the
    extraction prompt added, so a report can score the asked-for relations apart.
    """

    source: str
    source_type: str
    predicate: str
    target: str
    target_type: str
    valid_from: str | None = None
    valid_to: str | None = None
    assertion_kind: str = "assert"
    predicate_alternatives: tuple[str, ...] = ()
    beyond_prompt: bool = False


@dataclass(frozen=True, slots=True)
class GoldProbe:
    """One gold corpus probe: utterance plus expected entities and relationships."""

    tag: str
    utterance: str
    entities: tuple[GoldEntity, ...]
    relationships: tuple[GoldRel, ...]


@dataclass(frozen=True, slots=True)
class Produced:
    """The extraction the pipeline produced for a single utterance."""

    utterance: str
    parse_ok: bool
    entities: tuple[GoldEntity, ...]
    entity_type_by_id: dict[str, str]
    relationships: tuple[dict[str, Any], ...]


def iter_gold_probes(path: Path) -> list[GoldProbe]:
    probes: list[GoldProbe] = []
    with path.open(encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            # Every id, alternative and endpoint takes the per-entity id rails with its type,
            # so gold holds the ids the graph would store (`vs-code` loads as
            # `visual-studio-code`). Gold Event ids keep their dates as labelled; the
            # event-date alignment kind bridges them.
            entities = tuple(
                GoldEntity(
                    id=stored_entity_id(e["id"], e["type"]),
                    type=e["type"],
                    alternatives=tuple(
                        stored_entity_id(a, e["type"]) for a in e.get("alternatives", ())
                    ),
                )
                for e in obj.get("expected_entities", [])
            )
            tag = obj.get("tag", f"probe-{line_no}")
            rels = tuple(
                GoldRel(
                    source=stored_entity_id(r[GOLD_SOURCE_FIELD], r["source_type"]),
                    source_type=r["source_type"],
                    predicate=r[GOLD_PREDICATE_FIELD],
                    target=stored_entity_id(r[GOLD_TARGET_FIELD], r["target_type"]),
                    target_type=r["target_type"],
                    valid_from=r.get(GOLD_VALID_FROM_FIELD),
                    valid_to=r.get(GOLD_VALID_TO_FIELD),
                    assertion_kind=str(r.get("assertion_kind", "assert")),
                    predicate_alternatives=tuple(r.get("predicate_alternatives", ())),
                    beyond_prompt=bool(r.get("beyond_prompt", False)),
                )
                for r in obj.get("expected_relationships", [])
            )
            for rel in rels:
                if rel.assertion_kind not in VALID_ASSERTION_KINDS:
                    raise ValueError(
                        f"{tag}: invalid assertion_kind {rel.assertion_kind!r} "
                        f"(expected one of {VALID_ASSERTION_KINDS})"
                    )
            probes.append(
                GoldProbe(
                    tag=tag,
                    utterance=obj["utterance"],
                    entities=entities,
                    relationships=rels,
                )
            )
    return probes


def iter_debug_records(path: Path, session_id: str | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"warn: skipping malformed debug line: {exc}", file=sys.stderr)
                continue
            if session_id and rec.get("session_id") != session_id:
                continue
            records.append(rec)
    return records


def _recover_utterance(request: dict[str, Any]) -> str | None:
    messages = request.get("messages") or []
    if not messages:
        return None
    content = messages[-1].get("content") or ""
    m = EXTRACTION_UTTERANCE_PATTERN.search(content)
    return _norm_ws(m.group(1)) if m else None


def parse_produced(
    content: str,
    *,
    batch_record: list[dict[str, Any]] | None = None,
    scope: str | None = None,
    turn_plan: Mapping[str, Any] | None = None,
) -> tuple[bool, tuple[GoldEntity, ...], dict[str, str], tuple[dict[str, Any], ...]]:
    """Parse a raw extraction-LLM JSON string into the entities and rels the graph stores.

    The payload goes through `rails.run_rails` with the registry's active pack, the runner
    the normalizer runs, so each id, type and edge is the stored one. The return shape is
    unchanged.

    Args:
        content: The payload as JSON text.
        batch_record: The batch record of the row the payload came from, handed to the
            rails as the apply hands it, so a rail that reads the record gives the
            scorer the id it gives the graph. None scores the payload as a row extracted
            as one batch.
        scope: The row's scope label; None for a row that was not classified.
        turn_plan: The row's plan, as its payload or as the stored record that wraps it;
            None for a row without a plan.
    """
    parsed: Any = None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
            except json.JSONDecodeError:
                parsed = None
    if not isinstance(parsed, dict):
        return False, (), {}, ()
    # The normalizer's own runner, on copies: the stored ids, types and edges, Event-date
    # rail and deduplication included, so the score is of what the graph stores.
    row = rails.rail_row(
        [copy.deepcopy(e) for e in parsed.get("entities") or [] if isinstance(e, dict)],
        [copy.deepcopy(r) for r in parsed.get("relationships") or [] if isinstance(r, dict)],
        scope=scope,
        batch_record=copy.deepcopy(batch_record),
        turn_plan=copy.deepcopy(turn_plan),
    )
    stored_entities, stored_rels = rails.run_rails(row, registry.active_pack())
    entities: list[GoldEntity] = []
    type_by_id: dict[str, str] = {}
    for e in stored_entities:
        cid = str(e.get("id") or "")
        if cid:
            etype = native_entity_type(e)
            entities.append(GoldEntity(id=cid, type=etype))
            type_by_id[cid] = etype
    rels: list[dict[str, Any]] = []
    for r in stored_rels:
        raw_source, raw_target = native_endpoints(r)
        # An endpoint that names no produced entity keeps the plain slug, as before the
        # rails, so a bare endpoint is credited as it was.
        rels.append(
            {
                "source": raw_source if raw_source in type_by_id else canonical_id(raw_source),
                "target": raw_target if raw_target in type_by_id else canonical_id(raw_target),
                "predicate": native_predicate(r),
                "properties": native_properties(r),
            }
        )
    return True, tuple(entities), type_by_id, tuple(rels)


def produced_from_response(utterance: str, response: Mapping[str, Any]) -> Produced:
    """The scorer's `Produced` for one extraction-service response, as a raw record holds it.

    The rails are handed the row the service recorded for the turn: the payload, the
    scope label, the batch record and the turn plan, each read from the response's dump
    (`ExtractResponse.model_dump(mode="json")`). A response with no `batch_record` or no
    `turn_plan` key stands for a row extracted as one batch, or with no plan.

    Args:
        utterance: The probe's utterance; the index key is its normalised form.
        response: The response dump, with a `payload` object.
    """
    scope = response.get("scope")
    label = scope.get("label") if isinstance(scope, Mapping) else None
    ok, entities, type_by_id, rels = parse_produced(
        json.dumps(response["payload"]),
        batch_record=response.get("batch_record"),
        scope=label if isinstance(label, str) else None,
        turn_plan=response.get("turn_plan"),
    )
    key = _norm_ws(utterance)
    return Produced(
        utterance=key,
        parse_ok=ok,
        entities=entities,
        entity_type_by_id=type_by_id,
        relationships=rels,
    )


def build_produced_index(records: list[dict[str, Any]]) -> dict[str, Produced]:
    """Index produced extractions by recovered (normalized) utterance."""
    index: dict[str, Produced] = {}
    for rec in records:
        if rec.get("phase") != "llm_call" or rec.get("call_site") != EXTRACTION_CALL_SITE:
            continue
        utterance = _recover_utterance(rec.get("request") or {})
        if not utterance:
            continue
        content = (rec.get("response") or {}).get("content") or ""
        ok, entities, type_by_id, rels = parse_produced(content)
        if utterance in index and not ok:
            continue
        index[utterance] = Produced(
            utterance=utterance,
            parse_ok=ok,
            entities=entities,
            entity_type_by_id=type_by_id,
            relationships=rels,
        )
    return index


@dataclass(slots=True)
class Report:
    """Accumulated extraction-accuracy tallies and the derived metrics."""

    total_probes: int = 0
    matched_probes: int = 0
    entity_tp: int = 0
    entity_fp: int = 0
    entity_fn: int = 0
    rel_tp: int = 0
    rel_fp: int = 0
    rel_fn: int = 0
    typing_total: int = 0
    typing_ok: int = 0
    related_to_count: int = 0
    produced_rel_total: int = 0
    valid_time_total: int = 0
    valid_time_ok: int = 0
    negative_probes: int = 0
    negative_violations: int = 0
    # Specificity: among TP-aligned abstract-cluster entity pairs, count those
    # where the model emitted the precise leaf rather than the Abstraction fallback.
    # specificity_denominator counts cluster gold-TP pairs; specificity_numerator
    # counts the precise-leaf subset. Derived from matched entity pairs only.
    specificity_numerator: int = 0
    specificity_denominator: int = 0
    # Per assertion-kind: {gold_total, found, correct}. found = gold edges of
    # that kind that were extracted at all (TP membership); correct = found
    # edges whose engine-derived kind matches gold. The explicit `found` count
    # catches a model that simply never emits cessation/retraction edges --
    # total omission would still clear the 0.80 overall-recall gate.
    assertion_kind_buckets: dict[str, dict[str, int]] = field(default_factory=dict)
    per_probe: list[dict[str, Any]] = field(default_factory=list)
    # Relations scored over aligned entity ids (plan v8, D8): each produced id stands for
    # at most one gold id, paired by `entity_alignment.align_entities`. The exact counts
    # above are unchanged by them.
    aligned_rel_tp: int = 0
    aligned_rel_fp: int = 0
    aligned_rel_fn: int = 0
    # Counted pairs of the kinds a rail or gold makes equal (exact, alternative, event-date,
    # metric), and all counted pairs; their ratio is `id_agreement`.
    id_rail_pairs: int = 0
    id_aligned_pairs: int = 0
    alignment_kinds: dict[str, int] = field(default_factory=dict)
    # Every typo or paraphrase candidate that could count if accepted: tag, utterance,
    # gold id and type, produced id and type, kind, accepted.
    pending_pairs: list[dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def _pr(tp: int, fp: int, fn: int) -> tuple[float, float]:
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / (tp + fn) if (tp + fn) else 1.0
        return precision, recall

    @property
    def entity_precision(self) -> float:
        """Entity precision across all probes."""
        return self._pr(self.entity_tp, self.entity_fp, self.entity_fn)[0]

    @property
    def entity_precision_denominator(self) -> int:
        """Comparisons behind `entity_precision`: TP + FP produced entities."""
        return self.entity_tp + self.entity_fp

    @property
    def entity_recall(self) -> float:
        """Entity recall across all probes."""
        return self._pr(self.entity_tp, self.entity_fp, self.entity_fn)[1]

    @property
    def entity_recall_denominator(self) -> int:
        """Comparisons behind `entity_recall`: TP + FN gold entities."""
        return self.entity_tp + self.entity_fn

    @property
    def rel_precision(self) -> float:
        """Relationship precision across all probes."""
        return self._pr(self.rel_tp, self.rel_fp, self.rel_fn)[0]

    @property
    def rel_precision_denominator(self) -> int:
        """Comparisons behind `rel_precision`: TP + FP produced relationships."""
        return self.rel_tp + self.rel_fp

    @property
    def rel_recall(self) -> float:
        """Relationship recall across all probes."""
        return self._pr(self.rel_tp, self.rel_fp, self.rel_fn)[1]

    @property
    def rel_recall_denominator(self) -> int:
        """Comparisons behind `rel_recall`: TP + FN gold relationships."""
        return self.rel_tp + self.rel_fn

    @property
    def aligned_rel_precision(self) -> float:
        """Relationship precision with endpoints compared through the entity alignment."""
        return self._pr(self.aligned_rel_tp, self.aligned_rel_fp, self.aligned_rel_fn)[0]

    @property
    def aligned_rel_recall(self) -> float:
        """Relationship recall with endpoints compared through the entity alignment."""
        return self._pr(self.aligned_rel_tp, self.aligned_rel_fp, self.aligned_rel_fn)[1]

    @property
    def aligned_rel_f1(self) -> float:
        """F1 of the aligned relationship counts, 1.0 when there is nothing to count."""
        denominator = 2 * self.aligned_rel_tp + self.aligned_rel_fp + self.aligned_rel_fn
        return 2 * self.aligned_rel_tp / denominator if denominator else 1.0

    @property
    def id_agreement(self) -> float:
        """Share of counted pairs whose ids the graph stores as one node, 1.0 with none.

        A pair of kinds 5 to 8 (n1, reorder, typo, paraphrase) is two nodes in the graph,
        so a fall in this share is recall bought by fragmenting the graph.
        """
        return self.id_rail_pairs / self.id_aligned_pairs if self.id_aligned_pairs else 1.0

    @property
    def typing_accuracy(self) -> float:
        """Fraction of produced relationships that are constraint-valid."""
        return self.typing_ok / self.typing_total if self.typing_total else 1.0

    @property
    def related_to_rate(self) -> float:
        """Fraction of produced relationships typed RELATED_TO."""
        return self.related_to_count / self.produced_rel_total if self.produced_rel_total else 0.0

    @property
    def valid_time_accuracy(self) -> float:
        """Fraction of valid-time gold relationships whose dates matched."""
        return self.valid_time_ok / self.valid_time_total if self.valid_time_total else 1.0

    @property
    def specificity(self) -> float:
        """Fraction of abstract-cluster TP entity pairs where the precise leaf was used.

        Vacuously 1.0 when no TP entity pair involves a cluster type -- guards against
        reporting 0.0 on corpora with no abstract entities.
        """
        if self.specificity_denominator == 0:
            return 1.0
        return self.specificity_numerator / self.specificity_denominator

    @property
    def vacuous(self) -> bool:
        """True when this run read zero probes from the gold corpus.

        Every per-metric property above (`entity_precision`, `rel_precision`,
        `typing_accuracy`, `specificity`, ...) takes its vacuous default of 1.0
        when its own denominator is 0, and `related_to_rate` takes 0.0 -- so a
        zero-probe run renders as a wall of passing rows, not a wall of zeros.
        This flag is the one field that is False by coincidence for a real run
        and True by construction for a run that examined nothing, so a renderer
        can say "no probes read" instead of printing those rows at all.
        """
        return self.total_probes == 0


def _date_matches(expected: str | None, produced: Any) -> bool:
    if expected is None:
        return True
    if not produced:
        return False
    p = str(produced)
    return p.startswith(expected) or expected.startswith(p)


def _typing_ok(predicate: str, s_type: str | None, t_type: str | None) -> bool:
    constraint = RELATIONSHIP_CONSTRAINTS.get(predicate)
    if constraint is None:
        return False  # unknown predicate -> typing fail
    allowed_src, allowed_tgt = constraint
    if allowed_src is not None and s_type not in allowed_src:
        return False
    if allowed_tgt is not None and t_type not in allowed_tgt:
        return False
    return True


def _dating_edges() -> frozenset[str]:
    """The edges that date an entity's id: the active pack's id-phase `derived_edge` rules'."""
    rules = registry.active_pack().rules_of("derived_edge", _ID_PHASE)
    return frozenset(rule.params.edge for rule in rules)


def _abstract_cluster() -> frozenset[str]:
    """The types under the active pack's fallback type, which the specificity ratio covers."""
    pack = registry.active_pack()
    fallback = pack.manifest.fallback_type
    return frozenset(
        node.type_name for node in pack.node_types if getattr(node, "parent_type", None) == fallback
    )


def _occurred_on(keys: Collection[tuple[str, str, str]]) -> dict[str, str]:
    """Each source's earliest date among the keys of a dating edge whose target is an ISO date.

    The dating edge is `OCCURRED_ON` in today's pack.
    """
    dating = _dating_edges()
    dates: dict[str, str] = {}
    for source, predicate, target in keys:
        if predicate in dating and _ISO_DATE_RE.match(target):
            dates[source] = min(dates.get(source, target), target)
    return dates


def gold_entity_refs(probe: GoldProbe) -> list[EntityRef]:
    """The probe's gold entities and relation endpoints, as the aligner reads them.

    An endpoint that is not a listed entity takes its type from the relation, and every
    entity takes the date of its own `OCCURRED_ON` edge in the probe.
    """
    dates = _occurred_on([(r.source, r.predicate, r.target) for r in probe.relationships])
    types: dict[str, str | None] = {}
    alternatives: dict[str, tuple[str, ...]] = {}
    for e in probe.entities:
        types.setdefault(e.id, e.type or None)
        alternatives.setdefault(e.id, e.alternatives)
    for r in probe.relationships:
        types.setdefault(r.source, r.source_type or None)
        types.setdefault(r.target, r.target_type or None)
    return [
        EntityRef(i, t, alternatives.get(i, ()), dates.get(i)) for i, t in sorted(types.items())
    ]


def produced_entity_refs(produced: Produced) -> list[EntityRef]:
    """The produced entities and relation endpoints, as the aligner reads them.

    An endpoint with no entity of its own is untyped, which the aligner treats as
    compatible with any gold type.
    """
    keys = [(r["source"], r["predicate"], r["target"]) for r in produced.relationships]
    dates = _occurred_on(keys)
    ids = {e.id for e in produced.entities}
    ids.update(i for s, _, t in keys for i in (s, t))
    return [
        EntityRef(i, produced.entity_type_by_id.get(i) or None, (), dates.get(i))
        for i in sorted(ids)
    ]


def score_run(
    probes: list[GoldProbe],
    produced_index: dict[str, Produced],
    *,
    accepted_pairs: Collection[tuple[str, str, str]] = (),
) -> Report:
    """Score every probe on exact ids, and its relations again over aligned entity ids.

    `accepted_pairs` holds (probe tag, gold id, produced id) rows an adjudicator accepted;
    a typo or paraphrase pair counts only when its own probe's row lists it, so with the
    default none counts and each such pair scores as a miss and a false positive.
    """
    report = Report(total_probes=len(probes))
    # Initialize all three kinds so an absent bucket reads zeros (not KeyError):
    # a corpus with no retract gold must still surface retract: 0/0 0/0.
    for kind in VALID_ASSERTION_KINDS:
        report.assertion_kind_buckets[kind] = {"gold_total": 0, "found": 0, "correct": 0}
    report.alignment_kinds = dict.fromkeys(ALIGNMENT_KINDS, 0)
    accepted_by_tag: dict[str, set[tuple[str, str]]] = {}
    for tag, gold_id, produced_id in accepted_pairs:
        accepted_by_tag.setdefault(tag, set()).add((gold_id, produced_id))
    _cluster = _abstract_cluster()
    escape_edge = registry.active_pack().manifest.escape_edge
    for probe in probes:
        produced = produced_index.get(_norm_ws(probe.utterance))
        if produced is not None:
            report.matched_probes += 1

        gold_entities = set(probe.entities)
        gold_list = list(probe.entities)
        prod_list = list(produced.entities) if produced else []

        # Hierarchy-aware entity matching: a produced entity with id X and type
        # Abstraction counts as TP for gold (X, Concept) because Abstraction is
        # Concept's parent. Build matched sets by walking gold and finding a
        # produced entity with the same id where types_match holds.
        prod_by_id: dict[str, GoldEntity] = {}
        for pe in prod_list:
            prod_by_id.setdefault(pe.id, pe)

        # hierarchy_tp: gold entity ids for which the produced output was TP.
        hierarchy_tp: set[str] = set()
        for ge in gold_list:
            pe = prod_by_id.get(ge.id)
            if pe is not None and types_match(pe.type, ge.type):
                hierarchy_tp.add(ge.id)
                # Accumulate specificity over TP abstract-cluster pairs.
                g_canon = canonical_type(ge.type)
                if g_canon in _cluster:
                    report.specificity_denominator += 1
                    if canonical_type(pe.type) == g_canon:
                        report.specificity_numerator += 1

        # Counts: TP = matched gold ids; FP = produced ids not in any gold match;
        # FN = unmatched gold ids.
        report.entity_tp += len(hierarchy_tp)
        report.entity_fp += sum(1 for pe in prod_list if pe.id not in hierarchy_tp)
        report.entity_fn += sum(1 for ge in gold_list if ge.id not in hierarchy_tp)

        gold_rel_keys = {(r.source, r.predicate, r.target) for r in probe.relationships}
        prod_rel_keys: set[tuple[str, str, str]] = set()
        # Map each produced key tuple to its rel dict so a matched gold edge can
        # be handed to the engine's derive_assertion_kind (anti-drift: the kind
        # is whatever the engine will derive, not a parallel reimplementation).
        prod_rel_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
        if produced:
            for r in produced.relationships:
                key = (r["source"], r["predicate"], r["target"])
                prod_rel_keys.add(key)
                prod_rel_by_key[key] = r
                report.produced_rel_total += 1
                if r["predicate"] == escape_edge:
                    report.related_to_count += 1
                report.typing_total += 1
                if _typing_ok(
                    r["predicate"],
                    produced.entity_type_by_id.get(r["source"]),
                    produced.entity_type_by_id.get(r["target"]),
                ):
                    report.typing_ok += 1
        report.rel_tp += len(gold_rel_keys & prod_rel_keys)
        report.rel_fp += len(prod_rel_keys - gold_rel_keys)
        report.rel_fn += len(gold_rel_keys - prod_rel_keys)

        aligned_pairs: tuple[AlignedPair, ...] = ()
        if produced is not None:
            gold_refs = gold_entity_refs(probe)
            produced_refs = produced_entity_refs(produced)
            accepted = accepted_by_tag.get(probe.tag, set())
            aligned_pairs = align_entities(gold_refs, produced_refs, accepted=accepted)
            # A pending candidate is listed only when neither id is held by a counted pair
            # of a rail or form kind, since only then could accepting it change the score.
            held_gold = {p.gold_id for p in aligned_pairs if p.kind not in PENDING_KINDS}
            held_produced = {p.produced_id for p in aligned_pairs if p.kind not in PENDING_KINDS}
            gold_types = {r.id: r.type for r in gold_refs}
            produced_types = {r.id: r.type for r in produced_refs}
            for pair in candidate_pairs(gold_refs, produced_refs):
                if pair.kind not in PENDING_KINDS:
                    continue
                if pair.gold_id in held_gold or pair.produced_id in held_produced:
                    continue
                report.pending_pairs.append(
                    {
                        "tag": probe.tag,
                        "utterance": probe.utterance,
                        "gold_id": pair.gold_id,
                        "gold_type": gold_types[pair.gold_id],
                        "produced_id": pair.produced_id,
                        "produced_type": produced_types[pair.produced_id],
                        "kind": pair.kind,
                        "accepted": (pair.gold_id, pair.produced_id) in accepted,
                    }
                )
        for pair in aligned_pairs:
            report.alignment_kinds[pair.kind] += 1
            report.id_aligned_pairs += 1
            report.id_rail_pairs += pair.kind in EXCLUSION_EXEMPT_KINDS
        predicate_alternatives: dict[tuple[str, str, str], tuple[str, ...]] = {}
        for r in probe.relationships:
            key = (r.source, r.predicate, r.target)
            known = predicate_alternatives.get(key, ())
            predicate_alternatives[key] = known + tuple(
                a for a in r.predicate_alternatives if a not in known
            )
        aligned = match_relations(
            sorted(gold_rel_keys),
            sorted(prod_rel_keys),
            aligned_pairs,
            predicate_alternatives=predicate_alternatives,
        )
        report.aligned_rel_tp += len(aligned.found)
        report.aligned_rel_fp += len(aligned.extra)
        report.aligned_rel_fn += len(aligned.missed)

        for gr in probe.relationships:
            bucket = report.assertion_kind_buckets[gr.assertion_kind]
            bucket["gold_total"] += 1
            matched = prod_rel_by_key.get((gr.source, gr.predicate, gr.target))
            if matched is None:
                continue  # gold edge not extracted at all -> not in found
            bucket["found"] += 1
            if derive_assertion_kind(matched)[0].value == gr.assertion_kind:
                bucket["correct"] += 1

        for gr in probe.relationships:
            if gr.valid_from is None and gr.valid_to is None:
                continue
            report.valid_time_total += 1
            if not produced:
                continue
            match = next(
                (
                    r
                    for r in produced.relationships
                    if (r["source"], r["predicate"], r["target"])
                    == (gr.source, gr.predicate, gr.target)
                ),
                None,
            )
            if (
                match
                and _date_matches(gr.valid_from, match["properties"].get("start_date"))
                and _date_matches(gr.valid_to, match["properties"].get("end_date"))
            ):
                report.valid_time_ok += 1

        if not probe.entities and not probe.relationships:
            report.negative_probes += 1
            if produced is not None and (produced.entities or produced.relationships):
                report.negative_violations += 1

        if produced is not None:
            # FP: produced entities whose id was not matched by any gold entity.
            entity_fps = sorted([pe.id, pe.type] for pe in prod_list if pe.id not in hierarchy_tp)
            # FN: gold entities whose id had no hierarchy-compatible produced match.
            entity_fns = sorted([ge.id, ge.type] for ge in gold_list if ge.id not in hierarchy_tp)
            rel_fps = sorted(list(t) for t in prod_rel_keys - gold_rel_keys)
            rel_fns = sorted(list(t) for t in gold_rel_keys - prod_rel_keys)
        else:
            entity_fps = []
            entity_fns = sorted([e.id, e.type] for e in gold_entities)
            rel_fps = []
            rel_fns = sorted(list(t) for t in gold_rel_keys)
        report.per_probe.append(
            {
                "tag": probe.tag,
                "matched": produced is not None,
                "gold_entities": len(gold_entities),
                "gold_relationships": len(gold_rel_keys),
                "entity_fps": entity_fps,
                "entity_fns": entity_fns,
                "rel_fps": rel_fps,
                "rel_fns": rel_fns,
                "aligned_rel_fps": sorted(list(t) for t in aligned.extra),
                "aligned_rel_fns": sorted(list(t) for t in aligned.missed),
                "aligned_pairs": [[p.gold_id, p.produced_id, p.kind] for p in aligned_pairs],
            }
        )
    return report


ENTITY_PRECISION_GATE = 0.90
ENTITY_RECALL_GATE = 0.80
REL_PRECISION_GATE = 0.90
REL_RECALL_GATE = 0.80
TYPING_ACCURACY_GATE = 0.90
RELATED_TO_RATE_LIMIT = 0.10
# Specificity floor: fraction of abstract-cluster TP entity pairs where the
# precise leaf (not the Abstraction parent fallback) was produced. Guards
# against inflating the typing gate by over-using the parent type.
# Vacuously 1.0 when no abstract-cluster gold entities are present (passes).
SPECIFICITY_FLOOR = 0.90


def reading_identity(gold: Path) -> dict[str, str | None]:
    """What a score of `gold` is taken under: this scorer, the gold and the active pack.

    Taken before anything is scored, so a reading names the code and pack that scored it
    (`governance` holds the rule for each field).
    """
    pack = registry.active_pack()
    return governance.reading_identity(
        root=_REPO_ROOT,
        gold=gold,
        pack_version=pack.version,
        pack_prompt_hash=pack.pack_prompt_hash,
        pack_rails_hash=pack.pack_rails_hash,
    )


def score_reconciliation() -> None:
    """Hook: reconciliation-action accuracy.

    The reconciliation engine (C2) does not yet emit reconciliation telemetry to
    the debug stream. Until C2 lands this reports SKIPPED.
    """
    print("reconciliation-action accuracy: SKIPPED (requires C2 telemetry)", file=sys.stderr)


def _render_count(numerator: int | None, denominator: int | None) -> str:
    """Render a gate's backing numerator/denominator, or `n/a` if absent.

    Mirrors the None-vs-0 discipline established in eval_harness/report.py's
    `_render_examined`: no count supplied (this gate has no denominator to
    show) renders as `n/a`, distinct from a real denominator of `0` (counted,
    found nothing), which renders as e.g. `0/0`.
    """
    if numerator is None or denominator is None:
        return "n/a"
    return f"{numerator}/{denominator}"


def _row(
    name: str,
    value: float,
    gate: float | None,
    op: str,
    *,
    numerator: int | None = None,
    denominator: int | None = None,
) -> str:
    count = _render_count(numerator, denominator)
    if gate is None:
        return f"| {name} | {value:.3f} | {count} | - | - |"
    ok = value >= gate if op == ">=" else value <= gate
    return f"| {name} | {value:.3f} | {count} | {op} {gate:.2f} | {'PASS' if ok else 'FAIL'} |"


def render_markdown(report: Report, identity: Mapping[str, object] | None = None) -> str:
    lines = [
        "# Extraction Accuracy Report (F2)",
        "",
    ]
    if identity is not None:
        lines += [f"- Reading: {governance.identity_text(identity)}", ""]
    if report.vacuous:
        lines += [
            "**VACUOUS RUN -- 0 probes were read from the gold corpus. No gate "
            "below was meaningfully evaluated and this is not a pass.** The "
            "metric rows still render because every precision/recall/typing "
            "ratio takes its no-denominator default (1.0, or 0.0 for RELATED_TO "
            "rate) -- they reflect an empty corpus, not a successful run.",
            "",
        ]
    lines += [
        f"- Probes: {report.total_probes} (matched in debug log: {report.matched_probes})",
        "",
        "| Metric | Value | Count | Gate | Pass |",
        "|---|---|---|---|---|",
        _row(
            "Entity precision",
            report.entity_precision,
            ENTITY_PRECISION_GATE,
            ">=",
            numerator=report.entity_tp,
            denominator=report.entity_precision_denominator,
        ),
        _row(
            "Entity recall",
            report.entity_recall,
            ENTITY_RECALL_GATE,
            ">=",
            numerator=report.entity_tp,
            denominator=report.entity_recall_denominator,
        ),
        _row(
            "Relationship precision",
            report.rel_precision,
            REL_PRECISION_GATE,
            ">=",
            numerator=report.rel_tp,
            denominator=report.rel_precision_denominator,
        ),
        _row(
            "Relationship recall",
            report.rel_recall,
            REL_RECALL_GATE,
            ">=",
            numerator=report.rel_tp,
            denominator=report.rel_recall_denominator,
        ),
        _row(
            "Typing accuracy",
            report.typing_accuracy,
            TYPING_ACCURACY_GATE,
            ">=",
            numerator=report.typing_ok,
            denominator=report.typing_total,
        ),
        _row(
            "RELATED_TO rate",
            report.related_to_rate,
            RELATED_TO_RATE_LIMIT,
            "<=",
            numerator=report.related_to_count,
            denominator=report.produced_rel_total,
        ),
        _row(
            "Valid-time accuracy",
            report.valid_time_accuracy,
            None,
            "",
            numerator=report.valid_time_ok,
            denominator=report.valid_time_total,
        ),
        _row(
            "Specificity (leaf vs parent)",
            report.specificity,
            SPECIFICITY_FLOOR,
            ">=",
            numerator=report.specificity_numerator,
            denominator=report.specificity_denominator,
        ),
        f"| Negative-control violations | {report.negative_violations} | "
        f"{_render_count(report.negative_violations, report.negative_probes)} | == 0 | "
        f"{'PASS' if report.negative_violations == 0 else 'FAIL'} |",
        "",
        "## Assertion-kind buckets",
        "",
        "Derivation is the engine's own `derive_assertion_kind` (shared, not "
        "reimplemented). `found` = gold edges of this kind extracted at all; "
        "`correct` = found edges whose derived kind matches gold.",
        "",
        "| Kind | correct/found | found/gold_total |",
        "|---|---|---|",
    ]
    for kind in VALID_ASSERTION_KINDS:
        b = report.assertion_kind_buckets.get(kind, {"gold_total": 0, "found": 0, "correct": 0})
        lines.append(f"| {kind} | {b['correct']}/{b['found']} | {b['found']}/{b['gold_total']} |")
    lines += [
        "",
        f"Matched probes: {report.matched_probes}/{report.total_probes}.",
        "Reconciliation-action accuracy: SKIPPED (requires C2 telemetry).",
    ]
    return "\n".join(lines) + "\n"


def render_json(report: Report, identity: Mapping[str, object] | None = None) -> str:
    """Render the machine-readable report, counts included.

    `render_markdown`'s Count column and these fields read the same Report
    attributes -- this is the durable artifact, so a ratio without its
    backing numerator/denominator here is exactly as unfalsifiable as it
    would be in the markdown table. `identity` (`reading_identity`) leads the
    report when given.
    """
    return json.dumps(
        {
            **(dict(identity) if identity is not None else {}),
            "total_probes": report.total_probes,
            "matched_probes": report.matched_probes,
            "vacuous": report.vacuous,
            "entity_precision": report.entity_precision,
            "entity_recall": report.entity_recall,
            "entity_tp": report.entity_tp,
            "entity_fp": report.entity_fp,
            "entity_fn": report.entity_fn,
            "rel_precision": report.rel_precision,
            "rel_recall": report.rel_recall,
            "rel_tp": report.rel_tp,
            "rel_fp": report.rel_fp,
            "rel_fn": report.rel_fn,
            "typing_accuracy": report.typing_accuracy,
            "typing_ok": report.typing_ok,
            "typing_total": report.typing_total,
            "related_to_rate": report.related_to_rate,
            "related_to_count": report.related_to_count,
            "produced_rel_total": report.produced_rel_total,
            "valid_time_accuracy": report.valid_time_accuracy,
            "valid_time_ok": report.valid_time_ok,
            "valid_time_total": report.valid_time_total,
            "specificity": report.specificity,
            "specificity_numerator": report.specificity_numerator,
            "specificity_denominator": report.specificity_denominator,
            "negative_probes": report.negative_probes,
            "negative_violations": report.negative_violations,
            "assertion_kind_buckets": report.assertion_kind_buckets,
            "per_probe": report.per_probe,
        },
        indent=2,
    )


def _gates_pass(report: Report) -> bool:
    if report.total_probes == 0:
        # The join check below cannot catch this: on an empty corpus
        # `matched_probes == total_probes` is `0 == 0`, and every precision
        # and typing metric takes its vacuous default of 1.0. A wrong --gold
        # path scored a full PASS until 2026-08-05.
        return False
    return (
        report.matched_probes == report.total_probes  # a broken join must not pass
        and report.negative_violations == 0  # no hallucinated facts on negative controls
        and report.entity_precision >= ENTITY_PRECISION_GATE
        and report.entity_recall >= ENTITY_RECALL_GATE
        and report.rel_precision >= REL_PRECISION_GATE
        and report.rel_recall >= REL_RECALL_GATE
        and report.typing_accuracy >= TYPING_ACCURACY_GATE
        and report.related_to_rate <= RELATED_TO_RATE_LIMIT
        and report.specificity >= SPECIFICITY_FLOOR
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score a MIST extraction run against a gold corpus.")
    p.add_argument("--gold", required=True, type=Path, help="Gold corpus JSONL.")
    p.add_argument(
        "--debug-jsonl", required=True, type=Path, help="MIST_DEBUG_JSONL from the replay run."
    )
    p.add_argument("--session-id", default=None, help="Filter debug records by session_id.")
    p.add_argument(
        "--output", default=None, type=Path, help="Write markdown report (default stdout)."
    )
    p.add_argument(
        "--json-output", default=None, type=Path, help="Write a machine-readable JSON report."
    )
    p.add_argument("--strict", action="store_true", help="Exit 1 if any core gate fails.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.gold.exists():
        print(f"error: gold corpus not found: {args.gold}", file=sys.stderr)
        return 2
    if not args.debug_jsonl.exists():
        print(f"error: debug jsonl not found: {args.debug_jsonl}", file=sys.stderr)
        return 2
    identity = reading_identity(args.gold)
    probes = iter_gold_probes(args.gold)
    records = iter_debug_records(args.debug_jsonl, session_id=args.session_id)
    report = score_run(probes, build_produced_index(records))
    score_reconciliation()
    md = render_markdown(report, identity)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(md, encoding="utf-8")
    else:
        print(md)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(render_json(report, identity), encoding="utf-8")
    if args.strict and not _gates_pass(report):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
