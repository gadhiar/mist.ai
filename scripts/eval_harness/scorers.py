"""Per-test scoring functions for the eval harness.

Each scorer takes one CaseResult dict (loaded from JSONL) plus the
corresponding expected payload and returns a CaseScore. The aggregator
combines scores per candidate and per test into a RunScores structure
that the report module consumes.

Scorer table:
    schema_conformance -> JSON validity + ontology type set + relationship
                          direction + confidence range + entity id resolution.
    tool_selection     -> tool name match + argument match (exact or keyword).
    personality        -> regex-based style markers + length envelope.
    rag_integration    -> keyword recall over gold facts in the context.
    coherence          -> heuristic checks (non-empty, no repetition loops).
    speed              -> metrics only; scored as normalized tokens/second.
"""

from __future__ import annotations

import json
import logging
import re
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ontology constants (mirror of backend/knowledge/ontologies/v1_0_0.py).
#
# RESYNCED 2026-06-14 (v1.4.0). These frozensets MUST stay aligned with
# EXTRACTABLE_NODE_TYPES and EXTRACTABLE_RELATIONSHIP_TYPES in
# backend/knowledge/ontologies/v1_0_0.py. The harness is designed to
# run standalone without a backend import at module load time, so the
# mirror is intentional -- but drift here silently mis-scores extraction
# quality (a type the model correctly produces gets counted as
# "bad_entity_types" if missing from this list).
#
# v1.4.0 changes: Topic and Milestone retired (Topic -> Concept,
# Milestone -> Event with event_type=milestone); Abstraction supertype added.
#
# Drift is caught by tests/unit/test_eval_harness_scorers.py which
# cross-checks these frozensets against the ontology source of truth
# every time the unit-test suite runs.
# ---------------------------------------------------------------------------

EXTRACTABLE_ENTITY_TYPES: frozenset[str] = frozenset(
    {
        # External domain -- original (10; Topic/Milestone retired in v1.4.0).
        "User",
        "Person",
        "Organization",
        "Technology",
        "Skill",
        "Project",
        "Concept",
        "Event",
        "Goal",
        "Preference",
        "Location",
        # External domain -- post-MVP additive 2026-04-22 (3; Milestone retired).
        "Date",
        "Metric",
        "Document",
        # External domain -- v1.1.0 additive 2026-05-06 (5).
        "Pattern",
        "Convention",
        "Mechanism",
        "Strategy",
        "DataStructure",
        # External domain -- v1.4.0 Abstraction supertype (1).
        "Abstraction",
        # MIST-scope identity -- Cluster 1 promoted from INTERNAL-only (1).
        "MistIdentity",
    }
)

EXTRACTABLE_RELATIONSHIP_TYPES: frozenset[str] = frozenset(
    {
        # External user-centric (13).
        "USES",
        "KNOWS",
        "WORKS_ON",
        "WORKS_AT",
        "INTERESTED_IN",
        "HAS_GOAL",
        "PREFERS",
        "DISLIKES",
        "EXPERT_IN",
        "LEARNING",
        "STRUGGLES_WITH",
        "DECIDED",
        "EXPERIENCED",
        # Structural (8 original).
        "IS_A",
        "PART_OF",
        "RELATED_TO",
        "DEPENDS_ON",
        "USED_FOR",
        "WORKS_WITH",
        "KNOWS_PERSON",
        "MEMBER_OF",
        # Cluster 1 MIST-scope (4).
        "IMPLEMENTED_WITH",
        "MIST_HAS_CAPABILITY",
        "MIST_HAS_TRAIT",
        "MIST_HAS_PREFERENCE",
        # Post-MVP additive temporal / quantified / document (4).
        "OCCURRED_ON",
        "HAS_METRIC",
        "REFERENCES_DOCUMENT",
        "PRECEDED_BY",
        # v1.1.0 additive mechanism / pattern / strategy (8).
        "MECHANISM_OF",
        "OPERATES_ON",
        "INPUT_TO",
        "IMPROVES",
        "COMPRISES",
        "APPLICABLE_TO",
        "STRATEGY_FOR",
        "NAMING_CONVENTION_OF",
        # v1.3.0 additive recommendation / habit (2).
        "RECOMMENDS",
        "HAS_HABIT",
    }
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CaseScore:
    """Score for a single case on a single candidate."""

    candidate_id: str
    test_name: str
    case_id: str
    iteration: int
    passed: bool
    score: float
    breakdown: dict[str, Any]
    examined: int | None
    error: str | None


@dataclass(frozen=True, slots=True)
class ScoreOutcome:
    """One scorer's verdict, carrying how much it looked at.

    `examined` is the count of items the scorer actually inspected -- entities
    and relationships for schema conformance, expected tool calls for tool
    selection, and so on. It exists because a scorer built from
    `len(bad_things) == 0` checks passes vacuously on an empty input: verified
    2026-08-04, an empty extraction scored (True, 1.0) on schema conformance.

    `None` and `0` are DIFFERENT facts and must not be collapsed. `0` means the
    scorer looked and found nothing to check, which is a hard failure. `None`
    means the scorer has no way to count its own input, which is a declared gap
    -- visible in the breakdown, but not a failure, because failing it would
    make such scorers permanently red and train readers to ignore the signal.
    """

    passed: bool
    score: float
    breakdown: dict[str, Any]
    examined: int | None


def enforce_non_vacuity(outcome: ScoreOutcome) -> ScoreOutcome:
    """Refuse a pass that examined nothing.

    Called explicitly at each of this project's scorer-dispatch sites --
    `_ingest_record` below, `exoneration_verdict.py`, and the three scoring
    blocks in `phase3_orchestrator.sh` -- rather than folded into each
    scorer, so a scorer added later cannot forget it at any one of them.

    That list is maintained by hand, not enforced: nothing currently fails a
    build if a new dispatch site reads `.score` off a `ScoreOutcome` without
    routing through this function first. Four of the five sites above were
    missed this way on 2026-08-04 and fixed after a whole-branch review, not
    before -- `SCORER_EXAMINES` and its guard test only check that a scorer
    name has a declared examined-count meaning, not that every caller
    applies this guard. Grep for raw `.score` reads on `ScoreOutcome`-typed
    values before trusting that a new dispatch site is covered.
    """
    if outcome.examined is None:
        return replace(outcome, breakdown={**outcome.breakdown, "examined_unknown": True})
    if outcome.examined == 0:
        return replace(
            outcome,
            passed=False,
            score=0.0,
            breakdown={**outcome.breakdown, "vacuous": True},
        )
    return outcome


@dataclass(slots=True)
class TestScores:
    """Aggregated scores for a test across all cases on one candidate."""

    test_name: str
    case_scores: list[CaseScore] = field(default_factory=list)
    pass_count: int = 0
    fail_count: int = 0
    error_count: int = 0

    @property
    def mean_score(self) -> float:
        """Return the mean of per-case scores, or 0.0 if no cases ran."""
        if not self.case_scores:
            return 0.0
        return statistics.mean(cs.score for cs in self.case_scores)

    @property
    def pass_rate(self) -> float:
        """Return pass / (pass + fail) across cases; 0.0 when nothing ran."""
        total = self.pass_count + self.fail_count
        if total == 0:
            return 0.0
        return self.pass_count / total

    @property
    def examined_total(self) -> int | None:
        """Sum of examined counts, or None if no case could report one.

        None propagates rather than reading as 0: a test whose scorers cannot
        count must not be reported as having examined nothing.
        """
        counts = [cs.examined for cs in self.case_scores if cs.examined is not None]
        if not counts:
            return None
        return sum(counts)


@dataclass(slots=True)
class CandidateScores:
    """All test scores for one candidate across the run."""

    candidate_id: str
    per_test: dict[str, TestScores] = field(default_factory=dict)
    tokens_per_second_samples: list[float] = field(default_factory=list)
    total_time_ms_samples: list[float] = field(default_factory=list)
    prompt_tokens_total: int = 0
    completion_tokens_total: int = 0
    error_count: int = 0

    @property
    def mean_tokens_per_second(self) -> float:
        """Return arithmetic mean of per-case tokens/sec samples."""
        if not self.tokens_per_second_samples:
            return 0.0
        return statistics.mean(self.tokens_per_second_samples)

    @property
    def p50_total_time_ms(self) -> float:
        """Return the median total wall time across cases in milliseconds."""
        if not self.total_time_ms_samples:
            return 0.0
        return statistics.median(self.total_time_ms_samples)

    @property
    def p95_total_time_ms(self) -> float:
        """Return the p95 total wall time across cases in milliseconds."""
        if not self.total_time_ms_samples:
            return 0.0
        n = len(self.total_time_ms_samples)
        sorted_vals = sorted(self.total_time_ms_samples)
        idx = min(int(round(0.95 * (n - 1))), n - 1)
        return sorted_vals[idx]

    def aggregate_quality_score(self, weights: dict[str, float] | None = None) -> float:
        """Weighted mean across test scores. Default: equal weights."""
        if not self.per_test:
            return 0.0
        if weights is None:
            return statistics.mean(ts.mean_score for ts in self.per_test.values())
        total_weight = 0.0
        weighted_sum = 0.0
        for name, ts in self.per_test.items():
            w = weights.get(name, 0.0)
            if w == 0:
                continue
            weighted_sum += w * ts.mean_score
            total_weight += w
        return weighted_sum / total_weight if total_weight > 0 else 0.0


@dataclass(slots=True)
class RunScores:
    """Full scoring output for the orchestrator + report."""

    per_candidate: dict[str, CandidateScores] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Scorer registry
# ---------------------------------------------------------------------------


Scorer = Callable[[dict[str, Any], dict[str, Any]], ScoreOutcome]


def score_schema_conformance(result: dict[str, Any], expected: dict[str, Any]) -> ScoreOutcome:
    """Score a JSON extraction response against MIST's ontology.

    Returns a ScoreOutcome where score in [0, 1] is the mean of five
    structural checks:
      1. JSON parses successfully.
      2. Top-level schema has `entities` (list) and `relationships` (list).
      3. Every entity type is in the extractable set.
      4. Every relationship type is in the extractable set.
      5. Every relationship's source/target resolves to an entity id.
    """
    raw = result.get("response_content", "")
    breakdown: dict[str, Any] = {
        "json_valid": False,
        "schema_shape_ok": False,
        "entity_types_ok": False,
        "relationship_types_ok": False,
        "relationship_refs_ok": False,
        "errors": [],
    }

    parsed = _parse_json_lenient(raw)
    if parsed is None:
        breakdown["errors"].append("JSON parse failed")
        # examined=0: nothing parsed, so there is no entities/relationships
        # list to count. Already a hard fail; the guard changes nothing here.
        return ScoreOutcome(passed=False, score=0.0, breakdown=breakdown, examined=0)
    breakdown["json_valid"] = True

    if not isinstance(parsed, dict) or "entities" not in parsed or "relationships" not in parsed:
        breakdown["errors"].append("missing entities/relationships keys")
        # examined=0: shape is wrong, so there is nothing to count items from
        # -- same reasoning as the JSON-parse-failure branch above.
        return ScoreOutcome(passed=False, score=0.2, breakdown=breakdown, examined=0)
    if not isinstance(parsed["entities"], list) or not isinstance(parsed["relationships"], list):
        breakdown["errors"].append("entities/relationships not lists")
        return ScoreOutcome(passed=False, score=0.2, breakdown=breakdown, examined=0)
    breakdown["schema_shape_ok"] = True

    bad_entity_types: list[str] = []
    entity_ids: set[str] = set()
    for ent in parsed["entities"]:
        if not isinstance(ent, dict):
            bad_entity_types.append("<non-dict entity>")
            continue
        et = ent.get("type")
        if et not in EXTRACTABLE_ENTITY_TYPES:
            bad_entity_types.append(str(et))
        if isinstance(ent.get("id"), str):
            entity_ids.add(ent["id"])
    breakdown["entity_types_ok"] = len(bad_entity_types) == 0
    breakdown["bad_entity_types"] = bad_entity_types

    bad_relationship_types: list[str] = []
    unresolved_refs: list[tuple[str, str]] = []
    for rel in parsed["relationships"]:
        if not isinstance(rel, dict):
            bad_relationship_types.append("<non-dict relationship>")
            continue
        rt = rel.get("type")
        if rt not in EXTRACTABLE_RELATIONSHIP_TYPES:
            bad_relationship_types.append(str(rt))
        src = rel.get("source")
        tgt = rel.get("target")
        if src not in entity_ids or tgt not in entity_ids:
            unresolved_refs.append((str(src), str(tgt)))
    breakdown["relationship_types_ok"] = len(bad_relationship_types) == 0
    breakdown["bad_relationship_types"] = bad_relationship_types
    breakdown["relationship_refs_ok"] = len(unresolved_refs) == 0
    breakdown["unresolved_refs"] = unresolved_refs

    # Gold-set delta check: compare to expected if provided.
    expected_entity_types = expected.get("entity_types")
    expected_relationship_types = expected.get("relationship_types")
    if expected_entity_types is not None:
        produced_entity_types = {
            ent.get("type") for ent in parsed["entities"] if isinstance(ent, dict)
        }
        breakdown["expected_entity_types"] = sorted(expected_entity_types)
        breakdown["produced_entity_types"] = sorted(
            t for t in produced_entity_types if t is not None
        )
        breakdown["entity_type_recall"] = _set_recall(expected_entity_types, produced_entity_types)
    if expected_relationship_types is not None:
        produced_relationship_types = {
            rel.get("type") for rel in parsed["relationships"] if isinstance(rel, dict)
        }
        breakdown["expected_relationship_types"] = sorted(expected_relationship_types)
        breakdown["produced_relationship_types"] = sorted(
            t for t in produced_relationship_types if t is not None
        )
        breakdown["relationship_type_recall"] = _set_recall(
            expected_relationship_types, produced_relationship_types
        )

    structural_checks = (
        breakdown["json_valid"],
        breakdown["schema_shape_ok"],
        breakdown["entity_types_ok"],
        breakdown["relationship_types_ok"],
        breakdown["relationship_refs_ok"],
    )
    score = sum(1 for c in structural_checks if c) / len(structural_checks)
    passed = all(structural_checks)

    # If the gold set was supplied, blend recall into the final score.
    if expected_entity_types is not None and expected_relationship_types is not None:
        recall_avg = (breakdown["entity_type_recall"] + breakdown["relationship_type_recall"]) / 2.0
        score = 0.5 * score + 0.5 * recall_avg
        passed = passed and recall_avg >= 0.8

    # examined = len(entities) + len(relationships): the graph elements this
    # scorer actually inspected while building bad_entity_types /
    # bad_relationship_types / unresolved_refs above. A structurally valid but
    # empty extraction (`{"entities": [], "relationships": []}`) examines 0 --
    # the exact input the audit found scoring a vacuous (True, 1.0).
    examined = len(parsed["entities"]) + len(parsed["relationships"])
    return ScoreOutcome(passed=passed, score=score, breakdown=breakdown, examined=examined)


def score_tool_selection(result: dict[str, Any], expected: dict[str, Any]) -> ScoreOutcome:
    """Score a tool-calling response against expected tool name + args."""
    tool_calls = result.get("response_tool_calls") or []
    expected_tool_name: str | None = expected.get("tool_name")
    expected_args: dict[str, Any] | None = expected.get("arguments")
    expected_no_call: bool = bool(expected.get("no_tool_call"))

    breakdown: dict[str, Any] = {
        "tool_calls_count": len(tool_calls),
        "expected_tool_name": expected_tool_name,
        "expected_no_call": expected_no_call,
    }

    if expected_no_call:
        # examined=1 either way: the one criterion checked is "did any tool
        # call occur." A correct no-call is a real, deliberate pass -- not
        # vacuous -- so it must not examine 0.
        if len(tool_calls) == 0:
            return ScoreOutcome(passed=True, score=1.0, breakdown=breakdown, examined=1)
        breakdown["errors"] = ["expected no tool call but model produced one"]
        return ScoreOutcome(passed=False, score=0.0, breakdown=breakdown, examined=1)

    if not tool_calls:
        # examined=0: no call was expected to be absent, none arrived, so
        # there is no name or argument to compare against. Already a hard
        # fail; the guard changes nothing here.
        breakdown["errors"] = ["no tool_calls in response"]
        return ScoreOutcome(passed=False, score=0.0, breakdown=breakdown, examined=0)

    first = tool_calls[0]
    actual_name = first.get("name", "")
    breakdown["actual_tool_name"] = actual_name
    name_match = expected_tool_name is None or actual_name == expected_tool_name
    # name_examined is 1 only when an expected name was actually supplied --
    # `expected_tool_name is None` short-circuits name_match to True without
    # comparing anything, so that case examines nothing on the name axis.
    name_examined = 1 if expected_tool_name is not None else 0
    if not name_match:
        breakdown["errors"] = [
            f"tool name mismatch: expected {expected_tool_name} got {actual_name}"
        ]
        return ScoreOutcome(passed=False, score=0.0, breakdown=breakdown, examined=name_examined)

    if expected_args is None:
        # examined=name_examined: if expected_tool_name was also None here,
        # `expected` supplied no criteria at all and this scorer validated
        # nothing -- examined=0 lets the non-vacuity guard catch that, closing
        # the same "expected={}" vacuous-pass hole the audit found in
        # schema_conformance (this branch used to return (True, 1.0)
        # unconditionally regardless of whether anything was checked).
        return ScoreOutcome(passed=True, score=1.0, breakdown=breakdown, examined=name_examined)

    try:
        actual_args = json.loads(first.get("arguments_json") or "{}")
    except json.JSONDecodeError:
        breakdown["errors"] = ["tool arguments_json is not valid JSON"]
        return ScoreOutcome(passed=False, score=0.25, breakdown=breakdown, examined=name_examined)

    breakdown["actual_args"] = actual_args
    arg_hits = 0
    arg_total = len(expected_args)
    for key, value in expected_args.items():
        if key in actual_args and _loose_arg_match(actual_args[key], value):
            arg_hits += 1
    arg_score = arg_hits / arg_total if arg_total else 1.0
    breakdown["arg_score"] = arg_score
    passed = name_match and arg_score >= 0.75
    # examined = name_examined + arg_total: every criterion this call actually
    # compared. If expected_tool_name was unset and expected_args == {}
    # (present but empty), examined is 0 -- nothing was really compared, same
    # vacuous shape as above.
    examined = name_examined + arg_total
    return ScoreOutcome(
        passed=passed, score=0.5 + 0.5 * arg_score, breakdown=breakdown, examined=examined
    )


def score_personality(result: dict[str, Any], expected: dict[str, Any]) -> ScoreOutcome:
    """Score a response against personality markers and length envelope."""
    content = result.get("response_content", "").strip()
    breakdown: dict[str, Any] = {
        "length_chars": len(content),
        "errors": [],
    }

    must_contain_patterns: list[str] = expected.get("must_contain_regex", [])
    must_not_contain_patterns: list[str] = expected.get("must_not_contain_regex", [])
    min_len: int | None = expected.get("min_chars")
    max_len: int | None = expected.get("max_chars")

    hit_must = 0
    for pattern in must_contain_patterns:
        if re.search(pattern, content, flags=re.IGNORECASE):
            hit_must += 1
        else:
            breakdown["errors"].append(f"missing required marker: {pattern}")
    must_score = hit_must / len(must_contain_patterns) if must_contain_patterns else 1.0

    hit_forbidden = 0
    for pattern in must_not_contain_patterns:
        if re.search(pattern, content, flags=re.IGNORECASE):
            hit_forbidden += 1
            breakdown["errors"].append(f"forbidden marker present: {pattern}")
    forbidden_score = (
        1.0 - (hit_forbidden / len(must_not_contain_patterns)) if must_not_contain_patterns else 1.0
    )

    length_ok = True
    if min_len is not None and len(content) < min_len:
        length_ok = False
        breakdown["errors"].append(f"response too short ({len(content)} < {min_len})")
    if max_len is not None and len(content) > max_len:
        length_ok = False
        breakdown["errors"].append(f"response too long ({len(content)} > {max_len})")
    length_score = 1.0 if length_ok else 0.5

    score = (must_score + forbidden_score + length_score) / 3.0
    passed = must_score >= 0.8 and hit_forbidden == 0 and length_ok
    breakdown["must_score"] = must_score
    breakdown["forbidden_score"] = forbidden_score
    breakdown["length_ok"] = length_ok
    # examined = length of the response content the regex/length checks above
    # actually ran over. An empty response examines 0 characters -- the exact
    # input the audit found scoring a vacuous (True, 1.0).
    return ScoreOutcome(passed=passed, score=score, breakdown=breakdown, examined=len(content))


def score_rag_integration(result: dict[str, Any], expected: dict[str, Any]) -> ScoreOutcome:
    """Score RAG recall: does the response surface the expected facts?"""
    content = result.get("response_content", "").lower()
    gold_facts: list[str] = expected.get("must_contain_facts", [])
    forbidden_facts: list[str] = expected.get("must_not_contain_facts", [])

    breakdown: dict[str, Any] = {
        "gold_count": len(gold_facts),
        "errors": [],
    }

    if not gold_facts:
        # examined=len(gold_facts)=0: forbidden_facts is never inspected on
        # this early-return path (pre-existing behavior, unchanged here), so
        # counting it would overstate what this call actually checked.
        return ScoreOutcome(passed=True, score=1.0, breakdown=breakdown, examined=len(gold_facts))

    hits = sum(1 for fact in gold_facts if fact.lower() in content)
    recall = hits / len(gold_facts)
    breakdown["recall"] = recall
    breakdown["missed"] = [f for f in gold_facts if f.lower() not in content]

    forbidden_hits = [f for f in forbidden_facts if f.lower() in content]
    breakdown["forbidden_hits"] = forbidden_hits
    if forbidden_hits:
        breakdown["errors"].append(f"forbidden facts present: {forbidden_hits}")

    passed = recall >= 0.7 and not forbidden_hits
    score = recall * (0.5 if forbidden_hits else 1.0)
    # examined = len(must_contain_facts) + len(must_not_contain_facts): every
    # fact this call actually inspected -- required facts for recall, plus
    # forbidden facts checked for false positives.
    examined = len(gold_facts) + len(forbidden_facts)
    return ScoreOutcome(passed=passed, score=score, breakdown=breakdown, examined=examined)


def score_coherence(result: dict[str, Any], expected: dict[str, Any]) -> ScoreOutcome:
    """Heuristic coherence checks: non-empty, no repetition loops, finish reason ok."""
    content = result.get("response_content", "").strip()
    finish_reason = result.get("finish_reason", "")
    breakdown: dict[str, Any] = {
        "length_chars": len(content),
        "finish_reason": finish_reason,
        "errors": [],
    }

    if not content:
        breakdown["errors"].append("empty response")
        # examined=0: matches breakdown["length_chars"] above. Already a hard
        # fail -- there is nothing here for the repetition/length/forbidden
        # checks below to run over.
        return ScoreOutcome(passed=False, score=0.0, breakdown=breakdown, examined=0)

    if finish_reason == "length":
        breakdown["errors"].append("finish_reason=length (likely mid-sentence cutoff)")

    # Repetition check: any trigram repeated 4+ times in a row is a loop.
    tokens = re.findall(r"\w+", content.lower())
    repetition_loop = False
    if len(tokens) >= 12:
        for i in range(len(tokens) - 11):
            trigram = tuple(tokens[i : i + 3])
            next_trigrams = [tuple(tokens[i + 3 * k : i + 3 * k + 3]) for k in range(1, 4)]
            if all(t == trigram for t in next_trigrams):
                repetition_loop = True
                break
    breakdown["repetition_loop"] = repetition_loop
    if repetition_loop:
        breakdown["errors"].append("trigram repetition loop detected")

    # Forbidden phrases from expected block (e.g., AI slop markers).
    forbidden_patterns: list[str] = expected.get("must_not_contain_regex", [])
    forbidden_hits: list[str] = []
    for pattern in forbidden_patterns:
        if re.search(pattern, content, flags=re.IGNORECASE):
            forbidden_hits.append(pattern)
    breakdown["forbidden_hits"] = forbidden_hits
    if forbidden_hits:
        breakdown["errors"].append(f"forbidden phrases: {forbidden_hits}")

    min_chars = expected.get("min_chars", 10)
    if len(content) < min_chars:
        breakdown["errors"].append(f"too short: {len(content)} < {min_chars}")

    checks_passed = [
        bool(content),
        finish_reason != "length",
        not repetition_loop,
        not forbidden_hits,
        len(content) >= min_chars,
    ]
    score = sum(1 for c in checks_passed if c) / len(checks_passed)
    passed = all(checks_passed)
    # examined = length of the content the heuristics above actually ran over,
    # mirroring breakdown["length_chars"] and score_personality's rule. The
    # fixed count of 5 heuristics is not used -- it never changes with input,
    # so it could not signal a vacuous case the way a real content-derived
    # count can.
    return ScoreOutcome(passed=passed, score=score, breakdown=breakdown, examined=len(content))


def score_speed(result: dict[str, Any], expected: dict[str, Any]) -> ScoreOutcome:
    """Score a speed sample: metrics only, no quality check.

    Score = tokens_per_second / target_tps, capped at 1.0.
    """
    metrics = result.get("metrics") or {}
    tps = float(metrics.get("tokens_per_second", 0.0))
    target_tps = float(expected.get("target_tokens_per_second", 30.0))
    score = min(tps / target_tps, 1.0) if target_tps > 0 else 0.0
    breakdown = {
        "tokens_per_second": tps,
        "target_tokens_per_second": target_tps,
        "total_time_ms": metrics.get("total_time_ms"),
        "completion_tokens": metrics.get("completion_tokens"),
    }
    passed = tps >= target_tps
    # This scorer is not counting items, it is answering "did I have a
    # metrics bundle to score" -- examined=1 when a real metrics record was
    # present (even a slow one), 0 when metrics is empty/missing entirely.
    # Zero-when-absent is correct, not None: this scorer CAN tell "no
    # metrics" apart from "a real sample," so None would wrongly claim it
    # cannot count when it can. It also matches this scorer's own existing
    # fail-closed behavior for the degenerate case (tps defaults to 0.0,
    # target_tps defaults to 30.0, so passed is already False here
    # regardless of the guard).
    examined = 1 if metrics else 0
    return ScoreOutcome(passed=passed, score=score, breakdown=breakdown, examined=examined)


def score_schema_conformance_lenient(
    result: dict[str, Any], expected: dict[str, Any]
) -> ScoreOutcome:
    """Score schema_conformance with minimum-scope output repair.

    Applies two repairs to raw response before scoring:
      (a) Strip trailing whitespace and control-character bursts.
      (b) Tolerate trailing commas before closing braces/brackets.
    Otherwise identical to strict score_schema_conformance: same five
    structural checks, same recall blending. Runs on all candidates for
    parity so delta vs strict quantifies each candidate's fragility under
    format-sensitive downstream consumers.
    """
    raw = result.get("response_content", "")
    cleaned = _apply_minimum_lenient_repair(raw)
    proxied = dict(result)
    proxied["response_content"] = cleaned
    outcome = score_schema_conformance(proxied, expected)
    breakdown = {**outcome.breakdown, "lenient_repairs_applied": cleaned != raw}
    # examined: this delegates to the strict scorer, but on the repaired
    # response, not the raw one -- so it is strict's count of the REPAIRED
    # extraction, which can exceed what strict would report on the raw
    # response. That gap is precisely the malformed-but-repairable case this
    # variant exists to handle, not an edge case to gloss over.
    return ScoreOutcome(
        passed=outcome.passed, score=outcome.score, breakdown=breakdown, examined=outcome.examined
    )


SCORER_REGISTRY: dict[str, Scorer] = {
    "schema_conformance": score_schema_conformance,
    "schema_conformance_lenient": score_schema_conformance_lenient,
    "tool_selection": score_tool_selection,
    "personality": score_personality,
    "rag_integration": score_rag_integration,
    "coherence": score_coherence,
    "speed": score_speed,
}


# scorer name -> what its `examined` count counts.
#
# An explicit declaration rather than a docstring convention or field-name
# sniffing, for the reason run_record.py:22-29 gives about its own registry: a
# heuristic silently mis-classifies the next entry someone adds, producing a
# number that reads as a diagnosis while being wrong. The guard test asserts
# this dict and SCORER_REGISTRY have identical keys, so an eighth scorer cannot
# ship an uninterpretable count.
SCORER_EXAMINES: dict[str, str] = {
    "schema_conformance": "entities plus relationships in the parsed extraction",
    "schema_conformance_lenient": (
        "entities plus relationships in the repaired extraction -- can exceed "
        "strict's count on the same input when repair recovers content strict "
        "could not parse at all"
    ),
    "tool_selection": "1 when no-call expected; else tool name (0/1) plus argument keys compared",
    "personality": "characters in the response content checked against style markers",
    "rag_integration": "gold facts checked for recall, forbidden facts for false positives",
    "coherence": "characters in the response content checked by the coherence heuristics",
    "speed": "whether a metrics bundle was present to score (1) or absent (0)",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_json_lenient(raw: str) -> dict[str, Any] | None:
    """Best-effort JSON parse: try direct, then regex-extract first object."""
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match is None:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _apply_minimum_lenient_repair(raw: str) -> str:
    """Minimum-scope repairs for schema_conformance_lenient.

    Repairs applied, in order:
      (a) Strip leading and trailing whitespace, collapse runs of 3+
          consecutive whitespace characters inside JSON to a single space.
      (b) Remove trailing commas before closing braces and brackets
          (`,}` -> `}`, `,]` -> `]`).

    No brace repair, no entity deduplication, no ID normalization. These
    two repairs mirror what any tolerant downstream JSON parser does.
    """
    if not raw:
        return raw
    cleaned = raw.strip()
    cleaned = re.sub(r"\s{3,}", " ", cleaned)
    cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)
    return cleaned


def _set_recall(expected: list[str], produced: set[Any]) -> float:
    expected_set = {e for e in expected if e}
    if not expected_set:
        return 1.0
    hits = sum(1 for e in expected_set if e in produced)
    return hits / len(expected_set)


def _loose_arg_match(actual: Any, expected: Any) -> bool:
    """Tolerant argument match: normalize strings and compare."""
    if isinstance(expected, str) and isinstance(actual, str):
        return expected.strip().lower() == actual.strip().lower()
    return actual == expected


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def score_run(jsonl_paths: list[Path], test_files: list[Any]) -> RunScores:
    """Read JSONL files from the orchestrator and produce RunScores.

    test_files parameter is kept as Any to avoid a circular import with
    run.py's TestFile dataclass. Only .test_type, .name, and .cases are
    accessed.
    """
    test_type_by_name = {tf.name: tf.test_type for tf in test_files}
    expected_by_test_case: dict[tuple[str, str], dict[str, Any]] = {}
    for tf in test_files:
        for case in tf.cases:
            expected_by_test_case[(tf.name, case.id)] = case.expected

    per_candidate: dict[str, CandidateScores] = {}

    for path in jsonl_paths:
        if not path.exists():
            logger.warning("results JSONL missing: %s", path)
            continue
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("bad JSONL line in %s: %s", path, exc)
                    continue
                _ingest_record(record, per_candidate, test_type_by_name, expected_by_test_case)

    return RunScores(per_candidate=per_candidate)


def _ingest_record(
    record: dict[str, Any],
    per_candidate: dict[str, CandidateScores],
    test_type_by_name: dict[str, str],
    expected_by_test_case: dict[tuple[str, str], dict[str, Any]],
) -> None:
    candidate_id = record.get("candidate_id", "")
    test_name = record.get("test_name", "")
    case_id = record.get("case_id", "")

    scores = per_candidate.setdefault(candidate_id, CandidateScores(candidate_id=candidate_id))
    test_scores = scores.per_test.setdefault(test_name, TestScores(test_name=test_name))

    metrics = record.get("metrics") or {}
    if metrics:
        tps = float(metrics.get("tokens_per_second", 0.0))
        total_ms = float(metrics.get("total_time_ms", 0.0))
        prompt_tok = int(metrics.get("prompt_tokens", 0))
        completion_tok = int(metrics.get("completion_tokens", 0))
        if tps > 0:
            scores.tokens_per_second_samples.append(tps)
        if total_ms > 0:
            scores.total_time_ms_samples.append(total_ms)
        scores.prompt_tokens_total += prompt_tok
        scores.completion_tokens_total += completion_tok

    if record.get("error"):
        scores.error_count += 1
        test_scores.error_count += 1
        # examined=None: no scorer ran here -- the candidate's run itself
        # errored before there was any response to score. That is a declared
        # gap, not a count of zero examined items.
        case_score = CaseScore(
            candidate_id=candidate_id,
            test_name=test_name,
            case_id=case_id,
            iteration=int(record.get("iteration", 1)),
            passed=False,
            score=0.0,
            breakdown={"error": record["error"]},
            examined=None,
            error=record["error"],
        )
        test_scores.case_scores.append(case_score)
        test_scores.fail_count += 1
        return

    test_type = test_type_by_name.get(test_name, test_name)
    scorer = SCORER_REGISTRY.get(test_type)
    if scorer is None:
        # Recording rather than returning: a bare `return` removed the case from
        # both numerator and denominator, so a renamed test type would silently
        # shrink the sample instead of failing. Found during the 2026-08-04
        # scorer audit grounding, not part of its filed findings.
        logger.warning("no scorer for test type %r", test_type)
        test_scores.case_scores.append(
            CaseScore(
                candidate_id=candidate_id,
                test_name=test_name,
                case_id=case_id,
                iteration=int(record.get("iteration", 1)),
                passed=False,
                score=0.0,
                breakdown={"no_scorer_for_test_type": test_type},
                examined=None,
                error=None,
            )
        )
        test_scores.fail_count += 1
        return

    expected = expected_by_test_case.get((test_name, case_id), {})
    try:
        outcome = enforce_non_vacuity(scorer(record, expected))
    except (KeyError, ValueError, TypeError) as exc:
        logger.exception("scorer crashed for %s/%s", candidate_id, case_id)
        outcome = ScoreOutcome(
            passed=False, score=0.0, breakdown={"scorer_error": str(exc)}, examined=None
        )

    case_score = CaseScore(
        candidate_id=candidate_id,
        test_name=test_name,
        case_id=case_id,
        iteration=int(record.get("iteration", 1)),
        passed=outcome.passed,
        score=outcome.score,
        breakdown=outcome.breakdown,
        examined=outcome.examined,
        error=None,
    )
    test_scores.case_scores.append(case_score)
    if outcome.passed:
        test_scores.pass_count += 1
    else:
        test_scores.fail_count += 1
