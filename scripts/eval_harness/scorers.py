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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ontology constants (mirror of backend/knowledge/ontologies/v1_0_0.py)
# ---------------------------------------------------------------------------

EXTRACTABLE_ENTITY_TYPES: frozenset[str] = frozenset(
    {
        "User",
        "Person",
        "Organization",
        "Technology",
        "Skill",
        "Project",
        "Concept",
        "Topic",
        "Event",
        "Goal",
        "Preference",
        "Location",
    }
)

EXTRACTABLE_RELATIONSHIP_TYPES: frozenset[str] = frozenset(
    {
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
        "IS_A",
        "PART_OF",
        "RELATED_TO",
        "DEPENDS_ON",
        "USED_FOR",
        "WORKS_WITH",
        "KNOWS_PERSON",
        "MEMBER_OF",
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
    error: str | None


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


Scorer = Callable[[dict[str, Any], dict[str, Any]], tuple[bool, float, dict[str, Any]]]


def score_schema_conformance(
    result: dict[str, Any], expected: dict[str, Any]
) -> tuple[bool, float, dict[str, Any]]:
    """Score a JSON extraction response against MIST's ontology.

    Returns a (passed, score, breakdown) triple where score in [0, 1] is
    the mean of five structural checks:
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
        return False, 0.0, breakdown
    breakdown["json_valid"] = True

    if not isinstance(parsed, dict) or "entities" not in parsed or "relationships" not in parsed:
        breakdown["errors"].append("missing entities/relationships keys")
        return False, 0.2, breakdown
    if not isinstance(parsed["entities"], list) or not isinstance(parsed["relationships"], list):
        breakdown["errors"].append("entities/relationships not lists")
        return False, 0.2, breakdown
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

    return passed, score, breakdown


def score_tool_selection(
    result: dict[str, Any], expected: dict[str, Any]
) -> tuple[bool, float, dict[str, Any]]:
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
        if len(tool_calls) == 0:
            return True, 1.0, breakdown
        breakdown["errors"] = ["expected no tool call but model produced one"]
        return False, 0.0, breakdown

    if not tool_calls:
        breakdown["errors"] = ["no tool_calls in response"]
        return False, 0.0, breakdown

    first = tool_calls[0]
    actual_name = first.get("name", "")
    breakdown["actual_tool_name"] = actual_name
    name_match = expected_tool_name is None or actual_name == expected_tool_name
    if not name_match:
        breakdown["errors"] = [
            f"tool name mismatch: expected {expected_tool_name} got {actual_name}"
        ]
        return False, 0.0, breakdown

    if expected_args is None:
        return True, 1.0, breakdown

    try:
        actual_args = json.loads(first.get("arguments_json") or "{}")
    except json.JSONDecodeError:
        breakdown["errors"] = ["tool arguments_json is not valid JSON"]
        return False, 0.25, breakdown

    breakdown["actual_args"] = actual_args
    arg_hits = 0
    arg_total = len(expected_args)
    for key, value in expected_args.items():
        if key in actual_args and _loose_arg_match(actual_args[key], value):
            arg_hits += 1
    arg_score = arg_hits / arg_total if arg_total else 1.0
    breakdown["arg_score"] = arg_score
    passed = name_match and arg_score >= 0.75
    return passed, 0.5 + 0.5 * arg_score, breakdown


def score_personality(
    result: dict[str, Any], expected: dict[str, Any]
) -> tuple[bool, float, dict[str, Any]]:
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
    return passed, score, breakdown


def score_rag_integration(
    result: dict[str, Any], expected: dict[str, Any]
) -> tuple[bool, float, dict[str, Any]]:
    """Score RAG recall: does the response surface the expected facts?"""
    content = result.get("response_content", "").lower()
    gold_facts: list[str] = expected.get("must_contain_facts", [])
    forbidden_facts: list[str] = expected.get("must_not_contain_facts", [])

    breakdown: dict[str, Any] = {
        "gold_count": len(gold_facts),
        "errors": [],
    }

    if not gold_facts:
        return True, 1.0, breakdown

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
    return passed, score, breakdown


def score_coherence(
    result: dict[str, Any], expected: dict[str, Any]
) -> tuple[bool, float, dict[str, Any]]:
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
        return False, 0.0, breakdown

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
    return passed, score, breakdown


def score_speed(
    result: dict[str, Any], expected: dict[str, Any]
) -> tuple[bool, float, dict[str, Any]]:
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
    return passed, score, breakdown


SCORER_REGISTRY: dict[str, Scorer] = {
    "schema_conformance": score_schema_conformance,
    "tool_selection": score_tool_selection,
    "personality": score_personality,
    "rag_integration": score_rag_integration,
    "coherence": score_coherence,
    "speed": score_speed,
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
        case_score = CaseScore(
            candidate_id=candidate_id,
            test_name=test_name,
            case_id=case_id,
            iteration=int(record.get("iteration", 1)),
            passed=False,
            score=0.0,
            breakdown={"error": record["error"]},
            error=record["error"],
        )
        test_scores.case_scores.append(case_score)
        test_scores.fail_count += 1
        return

    test_type = test_type_by_name.get(test_name, test_name)
    scorer = SCORER_REGISTRY.get(test_type)
    if scorer is None:
        logger.warning("no scorer for test type %r", test_type)
        return

    expected = expected_by_test_case.get((test_name, case_id), {})
    try:
        passed, score, breakdown = scorer(record, expected)
    except (KeyError, ValueError, TypeError) as exc:
        logger.exception("scorer crashed for %s/%s", candidate_id, case_id)
        passed, score, breakdown = False, 0.0, {"scorer_error": str(exc)}

    case_score = CaseScore(
        candidate_id=candidate_id,
        test_name=test_name,
        case_id=case_id,
        iteration=int(record.get("iteration", 1)),
        passed=passed,
        score=score,
        breakdown=breakdown,
        error=None,
    )
    test_scores.case_scores.append(case_score)
    if passed:
        test_scores.pass_count += 1
    else:
        test_scores.fail_count += 1
