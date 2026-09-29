"""Extraction gold gauntlet driven through the extraction service's HTTP endpoint.

`scripts.model_bench.probes.extraction` runs the 60-probe gold corpus through the
in-process production path against a local llama-server. This module is the
smallest adapter that points the SAME corpus and the SAME scorer at a remote
extraction service (`POST /v1/extract`, contract 1.x), so a cutover candidate
can be compared with the ADR-027 baseline. The preflight prints the serving config
`/v1/info` reports (constrained mode, reasoning effort, temperature, ctx size; contract
1.1.0) and the summary records it under `serving_config`; a field an older server does
not report gets a `[WARN]` line and is recorded as null.

Nothing is re-implemented:
- Corpus and scoring: `scripts.eval_harness.score_extraction_run` (`iter_gold_probes`,
  `parse_produced`, `score_run`, `_norm_ws`). The service's payload
  `{"entities": [...], "relationships": [...]}` has the shape of the raw extraction
  LLM JSON, so `parse_produced(json.dumps(payload))` yields the scorer's `Produced`,
  indexed by the normalised utterance. A failed case is absent from the index and
  scores as all false negatives.
- Metrics, CIs and per-item rows: `scripts.model_bench.probes.extraction`.
- Transport: `backend.extraction_backlog.inference.RemoteExtractionInference`.

Run discipline (shared single-GPU hardware): strictly sequential, a FRESH `job_id`
per call so the host idempotency cache cannot replay a result, and no retries -- a
failure is recorded, never retried.

Usage:
    python -m scripts.model_bench.probes.extraction_service
        --endpoint http://HOST:8090 --out DIR [--baseline DIR]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import time
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import httpx

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.extraction_backlog.errors import (  # noqa: E402
    ExtractionInferenceError,
    InferenceResponseInvalidError,
    InferenceServiceError,
    InferenceUnreachableError,
)
from backend.extraction_backlog.inference import RemoteExtractionInference  # noqa: E402
from backend.extraction_contract.models import (  # noqa: E402
    CONTRACT_VERSION,
    ErrorCode,
    ExpectStamps,
    ExtractRequest,
    ExtractResponse,
    InfoResponse,
    is_compatible,
)
from backend.knowledge.extraction.scope_classifier import parse_scope_output  # noqa: E402
from backend.knowledge.version_stamps import ONTOLOGY_VERSION  # noqa: E402
from scripts.eval_harness.score_extraction_run import (  # noqa: E402
    GoldProbe,
    Produced,
    _norm_ws,
    _recover_utterance,
    iter_gold_probes,
    parse_produced,
    score_run,
)
from scripts.model_bench.probes.extraction import (  # noqa: E402
    DEFAULT_GOLD_CORPUS,
    MIST_FIXED_CLOCK_PIN,
    _nearest_rank_percentile,
    build_per_item_rows,
    build_summary,
    load_bootstrap_params,
    refuse_if_exists,
)

SCOPE_CALL_SITE = "extraction.scope_classifier"

OUTCOME_OK = "ok"
OUTCOME_EMPTY = "empty"
OUTCOME_TIMEOUT = "timeout"
OUTCOME_CLIENT_TIMEOUT = "client_timeout"
OUTCOME_REPAIR_EXHAUSTED = "schema_repair_exhausted"
OUTCOME_UPSTREAM_LLM = "upstream_llm"
OUTCOME_UNREACHABLE = "unreachable"
OUTCOME_INVALID_RESPONSE = "invalid_response"
OUTCOME_OTHER_ERROR = "other_error"
OUTCOME_NOT_RUN = "not_run"

OUTCOME_CLASSES: tuple[str, ...] = (
    OUTCOME_OK,
    OUTCOME_EMPTY,
    OUTCOME_TIMEOUT,
    OUTCOME_CLIENT_TIMEOUT,
    OUTCOME_REPAIR_EXHAUSTED,
    OUTCOME_UPSTREAM_LLM,
    OUTCOME_UNREACHABLE,
    OUTCOME_INVALID_RESPONSE,
    OUTCOME_OTHER_ERROR,
)
_SUCCESS_CLASSES = frozenset({OUTCOME_OK, OUTCOME_EMPTY})

STOP_UNREACHABLE = "consecutive_unreachable"
STOP_CLIENT_TIMEOUT = "consecutive_client_timeout"
STOP_MAX_MINUTES = "max_minutes"
CONSECUTIVE_UNREACHABLE_LIMIT = 3
CONSECUTIVE_CLIENT_TIMEOUT_LIMIT = 2

# One job's worst case is a scope call plus two extraction attempts, each up to the
# service's LLM timeout (120-300s on the host), so the client waits well beyond a
# single call's budget. A shorter client timeout abandons a job the service keeps
# running (asyncio.shield in the service's idempotency cache) and the next request
# would then overlap it on the single GPU.
DEFAULT_CLIENT_TIMEOUT_S = 960.0
# After a client-side timeout, wait this long before the next request so the
# abandoned job can drain off the GPU.
DEFAULT_DRAIN_S = 120.0
DEFAULT_MAX_MINUTES = 50.0
CONNECT_TIMEOUT_S = 10.0

_UNPARSABLE_RE = re.compile(r"unparsable after \d+ attempts", re.IGNORECASE)

OUT_ROWS = "extraction_service.jsonl"
OUT_RAW = "extraction_service_raw.jsonl"
OUT_SUMMARY = "extraction_service_summary.json"
OUT_COMPARISON_JSON = "comparison.json"
OUT_COMPARISON_MD = "comparison.md"

# (summary key, display label) for the comparison metric table.
COMPARISON_METRICS: tuple[tuple[str, str], ...] = (
    ("entity_precision", "entity precision"),
    ("entity_recall", "entity recall"),
    ("rel_precision", "relationship precision"),
    ("rel_recall", "relationship recall"),
    ("rel_f1", "relationship F1"),
    ("typing_accuracy", "typing accuracy"),
    ("related_to_rate", "related_to rate"),
    ("valid_time_accuracy", "valid-time accuracy"),
)
_REGRESSION_COUNTERS = ("entity_fp", "entity_fn", "rel_fp", "rel_fn")

INTERPRETATION_WARNING = (
    "{failed} case(s) failed and {not_run} were not run. Precision and typing accuracy count "
    "only the output that was returned, so they EXCLUDE those cases and can look better than "
    "the system is; recall and F1 INCLUDE them as all false negatives. Read the metrics "
    "together with failure_classes, never precision alone."
)


# ---------------------------------------------------------------------------
# Outcome classification (pure)
# ---------------------------------------------------------------------------


def classify_outcome(
    *, response: ExtractResponse | None, error: BaseException | None = None
) -> tuple[str, str | None]:
    """Map one call's result to `(outcome_class, error_code)`.

    Exactly one of `response` / `error` is expected. A 200 is `empty` when it holds
    zero entities AND zero relationships, else `ok`. Errors map as: a contract
    `timeout` the service answered with (HTTP 504) -> `timeout`; a client-side read
    timeout, which the transport reports as `ErrorCode.TIMEOUT` with no HTTP status
    -> `client_timeout` (the service may still be running that job);
    `upstream_llm` whose message says the output
    was unparsable after N attempts -> `schema_repair_exhausted`; other
    `upstream_llm` -> `upstream_llm`; an unreachable service -> `unreachable`; an
    invalid body -> `invalid_response`; everything else -> `other_error`, keeping
    the code (contract code value, or the exception type name).

    Args:
        response: The validated 200 response, when the call succeeded.
        error: The exception the call raised, when it failed.

    Returns:
        The class and the error code (None for a 200 or when no code exists).
    """
    if response is not None:
        if not response.payload.entities and not response.payload.relationships:
            return OUTCOME_EMPTY, None
        return OUTCOME_OK, None
    if isinstance(error, InferenceUnreachableError):
        return OUTCOME_UNREACHABLE, None
    if isinstance(error, InferenceResponseInvalidError):
        return OUTCOME_INVALID_RESPONSE, None
    if isinstance(error, InferenceServiceError):
        code = error.code
        if code == ErrorCode.TIMEOUT:
            if error.http_status is None:
                return OUTCOME_CLIENT_TIMEOUT, code.value
            return OUTCOME_TIMEOUT, code.value
        if code == ErrorCode.UPSTREAM_LLM:
            if _UNPARSABLE_RE.search(str(error)):
                return OUTCOME_REPAIR_EXHAUSTED, code.value
            return OUTCOME_UPSTREAM_LLM, code.value
        return OUTCOME_OTHER_ERROR, code.value
    if isinstance(error, ExtractionInferenceError) and error.code is not None:
        return OUTCOME_OTHER_ERROR, error.code.value
    name = type(error).__name__ if error is not None else "unknown"
    return OUTCOME_OTHER_ERROR, name


def response_flags(response: ExtractResponse) -> tuple[bool, bool]:
    """Return `(scope_classification_failed, repaired)` for a successful response."""
    return "scope_classification_failed" in response.warnings, response.attempts > 1


# ---------------------------------------------------------------------------
# Running the corpus
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CaseResult:
    """One probe's call: identity, classification, latency and (when 200) the response."""

    probe_id: str
    utterance: str
    job_id: str
    outcome_class: str
    error_code: str | None
    error_message: str | None
    http_status: int | None
    latency_ms: float
    response: ExtractResponse | None


@dataclass(frozen=True, slots=True)
class RunResult:
    """All cases run, the ids never run, and why the run stopped early (if it did)."""

    cases: list[CaseResult]
    not_run: list[str]
    stopped_reason: str | None
    elapsed_s: float


def build_request(
    probe: GoldProbe, *, session_id: str, recorded_at: str, expect: ExpectStamps
) -> ExtractRequest:
    """Build one probe's request: fresh job/event/turn/request ids, no history, no derivation."""
    return ExtractRequest(
        contract_version=CONTRACT_VERSION,
        job_id=str(uuid.uuid4()),
        event_id=str(uuid.uuid4()),
        turn_id=str(uuid.uuid4()),
        request_id=str(uuid.uuid4()),
        session_id=session_id,
        recorded_at=recorded_at,
        turn_index=0,
        utterance=probe.utterance,
        conversation_history=[],
        expect=expect,
        derivation=None,
    )


async def run_cases(
    inference: RemoteExtractionInference,
    probes: list[GoldProbe],
    *,
    expect: ExpectStamps,
    session_id: str,
    recorded_at: str,
    max_seconds: float,
    drain_s: float = DEFAULT_DRAIN_S,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    on_case: Callable[[int, int, CaseResult], None] | None = None,
) -> RunResult:
    """Run every probe strictly one after another; never retry; never raise per case.

    Stops early, recording the remaining probe ids as `not_run`, after
    `CONSECUTIVE_UNREACHABLE_LIMIT` consecutive `unreachable` cases, after
    `CONSECUTIVE_CLIENT_TIMEOUT_LIMIT` consecutive `client_timeout` cases, or once
    `max_seconds` of `clock()` time have elapsed (checked before each case).

    After a `client_timeout` the service may still be running the abandoned job, so
    the run sleeps `drain_s` before the next request (never overlapping it on the GPU).

    Args:
        inference: The transport.
        probes: Gold probes, in run order.
        expect: Stamps sent as the request's `expect`.
        session_id: One session id for the whole run.
        recorded_at: ISO-8601 reference date sent with every request.
        max_seconds: Wall-clock budget.
        drain_s: Pause after a client-side timeout, before the next request.
        clock: Monotonic clock (injected by tests).
        sleep: Awaitable sleep (injected by tests).
        on_case: Optional progress callback `(index, total, case)`.
    """
    start = clock()
    cases: list[CaseResult] = []
    consecutive_unreachable = 0
    consecutive_client_timeout = 0
    stopped_reason: str | None = None
    total = len(probes)

    for index, probe in enumerate(probes):
        if clock() - start >= max_seconds:
            stopped_reason = STOP_MAX_MINUTES
            break
        request = build_request(
            probe, session_id=session_id, recorded_at=recorded_at, expect=expect
        )
        response: ExtractResponse | None = None
        error: BaseException | None = None
        t0 = time.perf_counter()
        try:
            response = await inference.extract(request)
        except (ExtractionInferenceError, httpx.HTTPError) as exc:
            error = exc
        latency_ms = (time.perf_counter() - t0) * 1000.0

        outcome_class, error_code = classify_outcome(response=response, error=error)
        case = CaseResult(
            probe_id=probe.tag,
            utterance=probe.utterance,
            job_id=request.job_id,
            outcome_class=outcome_class,
            error_code=error_code,
            error_message=None if error is None else str(error)[:500],
            http_status=getattr(error, "http_status", None),
            latency_ms=latency_ms,
            response=response,
        )
        cases.append(case)
        if on_case is not None:
            on_case(index + 1, total, case)

        consecutive_unreachable = (
            consecutive_unreachable + 1 if outcome_class == OUTCOME_UNREACHABLE else 0
        )
        if consecutive_unreachable >= CONSECUTIVE_UNREACHABLE_LIMIT:
            stopped_reason = STOP_UNREACHABLE
            break
        consecutive_client_timeout = (
            consecutive_client_timeout + 1 if outcome_class == OUTCOME_CLIENT_TIMEOUT else 0
        )
        if consecutive_client_timeout >= CONSECUTIVE_CLIENT_TIMEOUT_LIMIT:
            stopped_reason = STOP_CLIENT_TIMEOUT
            break
        if outcome_class == OUTCOME_CLIENT_TIMEOUT and index + 1 < total:
            await sleep(drain_s)

    not_run = [p.tag for p in probes[len(cases) :]]
    return RunResult(
        cases=cases, not_run=not_run, stopped_reason=stopped_reason, elapsed_s=clock() - start
    )


# ---------------------------------------------------------------------------
# Scoring and summary
# ---------------------------------------------------------------------------


def build_produced_index(cases: list[CaseResult]) -> dict[str, Produced]:
    """Index each successful case's payload as the scorer's `Produced`, by normalised utterance."""
    index: dict[str, Produced] = {}
    for case in cases:
        if case.response is None:
            continue
        ok, entities, type_by_id, rels = parse_produced(
            json.dumps(case.response.payload.model_dump(mode="json"))
        )
        index[_norm_ws(case.utterance)] = Produced(
            utterance=_norm_ws(case.utterance),
            parse_ok=ok,
            entities=entities,
            entity_type_by_id=type_by_id,
            relationships=rels,
        )
    return index


def latency_stats(values: list[float]) -> dict[str, float | int | None]:
    """p50 / p95 / max / mean of `values` (nearest-rank percentiles); None fields when empty."""
    if not values:
        return {"n": 0, "p50": None, "p95": None, "max": None, "mean": None}
    return {
        "n": len(values),
        "p50": _nearest_rank_percentile(values, 50.0),
        "p95": _nearest_rank_percentile(values, 95.0),
        "max": max(values),
        "mean": statistics.fmean(values),
    }


def scope_stats(decisions: dict[str, tuple[str, float]]) -> dict[str, Any]:
    """Label counts, user-scope rate and mean confidence over `id -> (label, confidence)`."""
    n = len(decisions)
    labels = Counter(label for label, _ in decisions.values())
    return {
        "n": n,
        "label_counts": dict(sorted(labels.items())),
        "user_scope_rate": (labels.get("user-scope", 0) / n) if n else None,
        "mean_confidence": (statistics.fmean(c for _, c in decisions.values()) if n else None),
    }


def host_scope_decisions(cases: list[CaseResult]) -> dict[str, tuple[str, float]]:
    """`probe id -> (scope label, confidence)` for every case that got a 200."""
    return {
        c.probe_id: (c.response.scope.label, c.response.scope.confidence)
        for c in cases
        if c.response is not None
    }


# The `/v1/info` fields (contract 1.1.0) that describe the serving config a run was
# measured under. A pre-1.1.0 server omits all four; a 1.1.0 server reports `ctx_size`
# as null when it could not read llama-server's `/props`.
SERVING_CONFIG_FIELDS = ("constrained_mode", "reasoning_effort", "temperature", "ctx_size")


def serving_config(info: InfoResponse) -> dict[str, Any]:
    """The serving-config fields of `info`, None where the service reported none."""
    return {name: getattr(info, name) for name in SERVING_CONFIG_FIELDS}


def log_serving_config(info: InfoResponse) -> list[str]:
    """Print the serving config, and one `[WARN]` per field the service did not report.

    A missing field does not fail the preflight: the run goes ahead and records the
    field as null, since an older server cannot report it.

    Returns:
        The names of the fields the service did not report.
    """
    config = serving_config(info)
    _log(
        "[INFO] serving config: "
        + " ".join(f"{name}={'null' if value is None else value}" for name, value in config.items())
    )
    missing = [name for name, value in config.items() if value is None]
    for name in missing:
        _log(
            f"[WARN] preflight: /v1/info did not report {name} (service contract "
            f"{info.contract_version}; a pre-1.1.0 server, or a value it could not read); "
            "continuing, recorded as null"
        )
    return missing


def build_run_summary(
    *,
    probes: list[GoldProbe],
    result: RunResult,
    gold_path: Path,
    info: InfoResponse,
    health: dict[str, Any],
    endpoint: str,
    session_id: str,
    recorded_at: str,
    client_timeout_s: float,
    max_minutes: float,
    limit: int | None,
    drain_s: float = DEFAULT_DRAIN_S,
) -> tuple[dict[str, Any], Any]:
    """Score the run and assemble `extraction_service_summary.json`.

    Returns:
        `(summary, report)` where `report` is the scorer's `Report`.
    """
    report = score_run(probes, build_produced_index(result.cases))
    seed, b, confidence, z = load_bootstrap_params()
    summary = build_summary(
        report,
        gold_path=gold_path,
        ontology_version=ONTOLOGY_VERSION,
        bootstrap_seed=seed,
        bootstrap_b=b,
        bootstrap_confidence=confidence,
        wilson_z=z,
        rate_limit_max_per_minute=None,
        pythonhashseed=None,
        mist_fixed_clock=recorded_at,
    )

    class_counts = {name: 0 for name in OUTCOME_CLASSES}
    for case in result.cases:
        class_counts[case.outcome_class] += 1
    other_codes = Counter(
        c.error_code or "none" for c in result.cases if c.outcome_class == OUTCOME_OTHER_ERROR
    )
    completed = [c for c in result.cases if c.outcome_class in _SUCCESS_CLASSES]
    scope_failed = sum(1 for c in completed if c.response and response_flags(c.response)[0])
    repaired = sum(1 for c in completed if c.response and response_flags(c.response)[1])

    summary["ontology_version_note"] = "scorer-side (local repo); the service reports none"
    summary["endpoint"] = {
        "url": endpoint,
        "info": info.model_dump(mode="json"),
        "health": health,
    }
    config = serving_config(info)
    summary["serving_config"] = config
    summary["serving_config_missing"] = [name for name, value in config.items() if value is None]
    summary["run"] = {
        "session_id": session_id,
        "recorded_at": recorded_at,
        "client_timeout_s": client_timeout_s,
        "drain_s": drain_s,
        "max_minutes": max_minutes,
        "limit": limit,
        "elapsed_s": result.elapsed_s,
        "stopped_reason": result.stopped_reason,
        "sequential": True,
        "retries": 0,
    }
    failed_ids = [c.probe_id for c in result.cases if c.outcome_class not in _SUCCESS_CLASSES]
    summary["cases_run"] = len(result.cases)
    summary["failed_cases"] = len(failed_ids)
    summary["failed_probe_ids"] = failed_ids
    summary["cases_not_run"] = len(result.not_run)
    summary["not_run_probe_ids"] = list(result.not_run)
    summary["interpretation_warning"] = (
        INTERPRETATION_WARNING.format(failed=len(failed_ids), not_run=len(result.not_run))
        if failed_ids or result.not_run
        else None
    )
    summary["failure_classes"] = class_counts
    summary["other_error_codes"] = dict(sorted(other_codes.items()))
    summary["flags"] = {"scope_classification_failed": scope_failed, "repaired": repaired}
    summary["latency_ms"] = latency_stats([c.latency_ms for c in completed])
    summary["latency_ms_all_cases"] = latency_stats([c.latency_ms for c in result.cases])
    summary["latency_notes"] = {
        "latency_ms": "client wall-clock over completed cases only (HTTP 200: ok and empty)",
        "latency_ms_all_cases": (
            "client wall-clock over every case that ran, failures included; a client_timeout "
            "contributes its full wait"
        ),
    }
    summary["service_timings_ms"] = {
        stage: latency_stats(
            [
                value
                for c in completed
                if c.response is not None
                and (value := getattr(c.response.timings_ms, stage)) is not None
            ]
        )
        for stage in ("scope", "extract", "derive", "total")
    }
    summary["scope"] = scope_stats(host_scope_decisions(result.cases))
    return summary, report


def case_row_fields(case: CaseResult) -> dict[str, Any]:
    """The per-case service fields of a row (everything except the scorer's counts)."""
    fields: dict[str, Any] = {
        "id": case.probe_id,
        "outcome_class": case.outcome_class,
        "error_code": case.error_code,
        "error_message": case.error_message,
        "http_status": case.http_status,
        "job_id": case.job_id,
        "latency_ms": case.latency_ms,
    }
    resp = case.response
    if resp is not None:
        scope_failed, repaired = response_flags(resp)
        fields.update(
            {
                "scope_classification_failed": scope_failed,
                "repaired": repaired,
                "scope_label": resp.scope.label,
                "scope_confidence": resp.scope.confidence,
                "attempts": resp.attempts,
                "warnings": list(resp.warnings),
                "timings_ms": resp.timings_ms.model_dump(mode="json"),
                "n_entities": len(resp.payload.entities),
                "n_relationships": len(resp.payload.relationships),
            }
        )
    return fields


def build_case_rows(report: Any, result: RunResult) -> list[dict[str, Any]]:
    """`extraction_service.jsonl` rows: the shared per-item fields plus per-case service data."""
    by_id = {c.probe_id: c for c in result.cases}
    rows: list[dict[str, Any]] = []
    for base in build_per_item_rows(report):
        row = dict(base)
        case = by_id.get(base["id"])
        if case is None:
            row.update({"outcome_class": OUTCOME_NOT_RUN, "job_id": None, "latency_ms": None})
        else:
            row.update(case_row_fields(case))
        rows.append(row)
    return rows


def raw_row(case: CaseResult) -> dict[str, Any] | None:
    """One `extraction_service_raw.jsonl` row: the full 200 response; None for a failed case."""
    if case.response is None:
        return None
    return {
        "id": case.probe_id,
        "job_id": case.job_id,
        "utterance": case.utterance,
        "response": case.response.model_dump(mode="json"),
    }


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Baseline:
    """What could be loaded from a baseline directory; absent parts are None and listed."""

    summary: dict[str, Any] | None
    rows: dict[str, dict[str, Any]] | None
    scope: dict[str, tuple[str, float]] | None
    omissions: list[str]


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def _read_rows(path: Path) -> dict[str, dict[str, Any]] | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    rows: dict[str, dict[str, Any]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and "id" in row:
            rows[str(row["id"])] = row
    return rows or None


def read_baseline_scope(path: Path, probes: list[GoldProbe]) -> dict[str, tuple[str, float]] | None:
    """Recover the baseline's scope decisions as `probe id -> (label, confidence)`.

    The decisions are the `extraction.scope_classifier` `llm_call` records in the
    debug JSONL. The utterance is recovered from the request the way the scorer does
    for the extraction call (`_recover_utterance`; the scope user template has the
    same `Utterance: "..."` / `Output:` shape), and the response is parsed with
    `parse_scope_output`. When an utterance appears more than once, the last
    parseable decision wins. Returns None when the file is missing or holds none.
    """
    id_by_utterance = {_norm_ws(p.utterance): p.tag for p in probes}
    decisions: dict[str, tuple[str, float]] = {}
    try:
        handle = path.open(encoding="utf-8")
    except OSError:
        return None
    with handle:
        for raw in handle:
            if SCOPE_CALL_SITE not in raw:
                continue
            try:
                rec = json.loads(raw)
            except ValueError:
                continue
            if rec.get("phase") != "llm_call" or rec.get("call_site") != SCOPE_CALL_SITE:
                continue
            utterance = _recover_utterance(rec.get("request") or {})
            probe_id = id_by_utterance.get(utterance or "")
            if probe_id is None:
                continue
            content = (rec.get("response") or {}).get("content") or ""
            label, confidence, _reasoning = parse_scope_output(content)
            decisions[probe_id] = (label, confidence)
    return decisions or None


def load_baseline(baseline_dir: Path, probes: list[GoldProbe]) -> Baseline:
    """Load `extraction_summary.json`, `extraction.jsonl` and `extraction_debug.jsonl`.

    A missing or unreadable file degrades to a stated omission; nothing raises.
    """
    omissions: list[str] = []
    summary = _read_json(baseline_dir / "extraction_summary.json")
    if summary is None:
        omissions.append("extraction_summary.json missing or unreadable: metric table omitted")
    rows = _read_rows(baseline_dir / "extraction.jsonl")
    if rows is None:
        omissions.append("extraction.jsonl missing or unreadable: regression list omitted")
    scope = read_baseline_scope(baseline_dir / "extraction_debug.jsonl", probes)
    if scope is None:
        omissions.append(
            "extraction_debug.jsonl missing or without scope records: "
            "baseline scope comparison omitted"
        )
    return Baseline(summary=summary, rows=rows, scope=scope, omissions=omissions)


def _baseline_ci(baseline: dict[str, Any], key: str) -> tuple[str, list[float]] | None:
    """The baseline's interval for `key`: bootstrap when present, else Wilson, else None."""
    for source in ("bootstrap", "wilson"):
        interval = (baseline.get(source) or {}).get(key)
        if isinstance(interval, list) and len(interval) == 2:
            return source, [float(interval[0]), float(interval[1])]
    return None


def compare_metrics(host: dict[str, Any], baseline: dict[str, Any]) -> list[dict[str, Any]]:
    """Metric table: host vs baseline, delta, and whether host lies inside baseline's CI."""
    table: list[dict[str, Any]] = []
    for key, label in COMPARISON_METRICS:
        host_value = host.get(key)
        base_value = baseline.get(key)
        ci = _baseline_ci(baseline, key)
        delta = (
            host_value - base_value
            if isinstance(host_value, int | float) and isinstance(base_value, int | float)
            else None
        )
        inside: bool | None = None
        if ci is not None and isinstance(host_value, int | float):
            inside = ci[1][0] <= host_value <= ci[1][1]
        table.append(
            {
                "metric": key,
                "label": label,
                "host": host_value,
                "baseline": base_value,
                "delta": delta,
                "baseline_ci": None if ci is None else ci[1],
                "baseline_ci_source": None if ci is None else ci[0],
                "host_inside_baseline_ci": inside,
            }
        )
    return table


def find_regressions(
    host_rows: list[dict[str, Any]], baseline_rows: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Probe ids that got worse than baseline, plus host ids the baseline has no row for.

    Regressed: any of `entity_fp`, `entity_fn`, `rel_fp`, `rel_fn` higher on the host
    than in the baseline for the same id, or matched in the baseline but failed
    (unmatched) on the host.
    """
    regressed: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in host_rows:
        base = baseline_rows.get(row["id"])
        if base is None:
            missing.append(row["id"])
            continue
        reasons = [
            f"{name} {base.get(name, 0)} -> {row.get(name, 0)}"
            for name in _REGRESSION_COUNTERS
            if row.get(name, 0) > base.get(name, 0)
        ]
        if base.get("matched") and not row.get("matched"):
            reasons.append("matched in baseline, failed on host")
        if reasons:
            regressed.append(
                {
                    "id": row["id"],
                    "reasons": reasons,
                    "outcome_class": row.get("outcome_class"),
                    "before": {k: base.get(k) for k in ("matched", *_REGRESSION_COUNTERS)},
                    "after": {k: row.get(k) for k in ("matched", *_REGRESSION_COUNTERS)},
                }
            )
    return regressed, missing


def compare_scope(
    host: dict[str, tuple[str, float]], baseline: dict[str, tuple[str, float]]
) -> dict[str, Any]:
    """Scope agreement with the baseline run (there is no gold scope label).

    Per-probe label agreement over ids both runs decided, both runs' user-scope
    rate, and both runs' mean confidence.
    """
    common = sorted(set(host) & set(baseline))
    per_probe = [
        {
            "id": pid,
            "host_label": host[pid][0],
            "baseline_label": baseline[pid][0],
            "agree": host[pid][0] == baseline[pid][0],
        }
        for pid in common
    ]
    agree = sum(1 for p in per_probe if p["agree"])
    return {
        "basis": "agreement with the baseline run's scope decisions (no gold scope label)",
        "n_compared": len(common),
        "n_agree": agree,
        "agreement_rate": (agree / len(common)) if common else None,
        "host": scope_stats(host),
        "baseline": scope_stats(baseline),
        "disagreements": [p for p in per_probe if not p["agree"]],
        "per_probe": per_probe,
    }


def build_comparison(
    host_summary: dict[str, Any],
    host_rows: list[dict[str, Any]],
    host_scope: dict[str, tuple[str, float]],
    baseline: Baseline,
) -> dict[str, Any]:
    """Assemble `comparison.json`; parts the baseline lacks are None and listed in `omissions`."""
    host_limit = (host_summary.get("run") or {}).get("limit")
    baseline_total = (baseline.summary or {}).get("total_probes")
    if baseline_total is None and baseline.rows is not None:
        baseline_total = len(baseline.rows)
    host_total = host_summary.get("total_probes")
    doc: dict[str, Any] = {
        "schema": 1,
        "host_run": {
            "complete": bool(host_summary.get("complete")),
            "total_probes": host_total,
            "matched_probes": host_summary.get("matched_probes"),
            "failed_cases": host_summary.get("failed_cases"),
            "failure_classes": host_summary.get("failure_classes"),
            "cases_not_run": host_summary.get("cases_not_run"),
            "not_run_probe_ids": host_summary.get("not_run_probe_ids"),
            "stopped_reason": (host_summary.get("run") or {}).get("stopped_reason"),
            "interpretation_warning": host_summary.get("interpretation_warning"),
        },
        "probe_sets": {
            "host_total": host_total,
            "baseline_total": baseline_total,
            "host_limit": host_limit,
            "differ": host_limit is not None
            or (baseline_total is not None and host_total != baseline_total),
        },
        "omissions": list(baseline.omissions),
        "metrics": None,
        "regressed": None,
        "host_ids_missing_from_baseline": None,
        "scope": {"host": scope_stats(host_scope), "comparison": None},
    }
    if baseline.summary is not None:
        doc["metrics"] = compare_metrics(host_summary, baseline.summary)
        doc["gold_corpus_sha256"] = {
            "host": host_summary.get("gold_corpus_sha256"),
            "baseline": baseline.summary.get("gold_corpus_sha256"),
            "match": host_summary.get("gold_corpus_sha256")
            == baseline.summary.get("gold_corpus_sha256"),
        }
    if baseline.rows is not None:
        regressed, missing = find_regressions(host_rows, baseline.rows)
        doc["regressed"] = regressed
        doc["host_ids_missing_from_baseline"] = missing
    if baseline.scope is not None:
        doc["scope"]["comparison"] = compare_scope(host_scope, baseline.scope)
    return doc


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_comparison_md(comparison: dict[str, Any]) -> str:
    """Render `comparison.json` as Markdown (plain ASCII)."""
    lines = ["# Extraction service vs baseline", ""]
    host_run = comparison["host_run"]
    incomplete = not host_run["complete"]
    if incomplete:
        lines += [
            "**[INCOMPLETE] THE HOST RUN IS INCOMPLETE. Do not read the metrics below as a "
            "verdict.**",
            "",
            f"- matched probes: {host_run['matched_probes']}/{host_run['total_probes']}",
            f"- failed cases: {host_run['failed_cases']}; not run: {host_run['cases_not_run']}"
            f" (stopped: {host_run['stopped_reason']})",
        ]
        if host_run["interpretation_warning"]:
            lines.append(f"- {host_run['interpretation_warning']}")
        lines.append("")
    probe_sets = comparison["probe_sets"]
    if probe_sets["differ"]:
        lines += [
            "**[WARNING] PROBE SETS DIFFER: host ran "
            f"{probe_sets['host_total']} probe(s) (limit {probe_sets['host_limit']}), baseline "
            f"has {probe_sets['baseline_total']}. Metrics are not directly comparable.**",
            "",
        ]
    lines += [
        "## Host run",
        "",
        f"- complete: {'yes' if host_run['complete'] else 'NO'}",
        f"- failure classes: {host_run['failure_classes']}",
        f"- not run: {host_run['cases_not_run']}",
        "",
    ]
    if comparison["omissions"]:
        lines.append("## Omissions")
        lines.append("")
        lines.extend(f"- {note}" for note in comparison["omissions"])
        lines.append("")

    sha = comparison.get("gold_corpus_sha256")
    if sha is not None and not sha["match"]:
        lines += [
            "[WARNING] gold corpus sha256 differs: "
            f"host {sha['host']} vs baseline {sha['baseline']}",
            "",
        ]

    lines += ["## Metrics", ""]
    if comparison["metrics"] is None:
        lines.append("Omitted (no baseline summary).")
    else:
        lines += [
            "| metric | host | baseline | delta | baseline 95% CI | host inside CI |",
            "|---|---|---|---|---|---|",
        ]
        for m in comparison["metrics"]:
            ci = m["baseline_ci"]
            ci_text = (
                "n/a" if ci is None else f"[{ci[0]:.4f}, {ci[1]:.4f}] ({m['baseline_ci_source']})"
            )
            inside = m["host_inside_baseline_ci"]
            inside_text = "n/a" if inside is None else ("yes" if inside else "NO")
            if inside is not None and incomplete:
                inside_text += " (INCOMPLETE run)"
            delta = m["delta"]
            delta_text = "n/a" if delta is None else f"{delta:+.4f}"
            lines.append(
                f"| {m['label']} | {_fmt(m['host'])} | {_fmt(m['baseline'])} | {delta_text} "
                f"| {ci_text} | {inside_text} |"
            )
    lines.append("")

    lines += ["## Regressed probes", ""]
    if comparison["regressed"] is None:
        lines.append("Omitted (no baseline rows).")
    elif not comparison["regressed"]:
        lines.append("None.")
    else:
        lines += ["| probe | class | reasons | before | after |", "|---|---|---|---|---|"]
        for r in comparison["regressed"]:
            before = ", ".join(f"{k}={v}" for k, v in r["before"].items())
            after = ", ".join(f"{k}={v}" for k, v in r["after"].items())
            lines.append(
                f"| {r['id']} | {r.get('outcome_class')} | {'; '.join(r['reasons'])} "
                f"| {before} | {after} |"
            )
    lines.append("")

    lines += ["## Scope", ""]
    scope = comparison["scope"]
    host_scope = scope["host"]
    lines.append(
        f"Host: {host_scope['n']} decisions, user-scope rate {_fmt(host_scope['user_scope_rate'])},"
        f" mean confidence {_fmt(host_scope['mean_confidence'])}."
    )
    cmp = scope["comparison"]
    if cmp is None:
        lines.append("Baseline scope comparison omitted (no baseline scope decisions).")
    else:
        base = cmp["baseline"]
        lines += [
            f"Baseline: {base['n']} decisions, user-scope rate {_fmt(base['user_scope_rate'])},"
            f" mean confidence {_fmt(base['mean_confidence'])}.",
            f"Label agreement with baseline (no gold scope label exists): {cmp['n_agree']}/"
            f"{cmp['n_compared']} ({_fmt(cmp['agreement_rate'])}).",
            "",
        ]
        if cmp["disagreements"]:
            lines += ["| probe | host | baseline |", "|---|---|---|"]
            lines.extend(
                f"| {d['id']} | {d['host_label']} | {d['baseline_label']} |"
                for d in cmp["disagreements"]
            )
        else:
            lines.append("No disagreements.")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration and CLI
# ---------------------------------------------------------------------------


def _append_jsonl(fh: TextIO, row: dict[str, Any]) -> None:
    """Write one row and flush, so the line survives a crash of the run."""
    fh.write(json.dumps(row, sort_keys=True) + "\n")
    fh.flush()


def _replace_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Replace `path` with `rows` via a temporary file, so a crash never leaves half a file."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _progress(index: int, total: int, case: CaseResult) -> None:
    _log(f"[{index}/{total}] {case.probe_id} {case.outcome_class} {case.latency_ms:.0f}ms")


async def execute(
    *,
    endpoint: str,
    out_dir: Path,
    gold_path: Path,
    client: httpx.AsyncClient,
    baseline_dir: Path | None = None,
    recorded_at: str = MIST_FIXED_CLOCK_PIN,
    client_timeout_s: float = DEFAULT_CLIENT_TIMEOUT_S,
    max_minutes: float = DEFAULT_MAX_MINUTES,
    limit: int | None = None,
    drain_s: float = DEFAULT_DRAIN_S,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> int:
    """Preflight, run the gauntlet, write every output, and return the process exit code.

    Returns 0 only when the run is complete (every probe matched); 1 on a refused
    overwrite, a failed preflight, or an incomplete run. Outputs are always written
    before an incomplete run returns.
    """
    out_paths = [out_dir / n for n in (OUT_ROWS, OUT_RAW, OUT_SUMMARY)]
    if baseline_dir is not None:
        out_paths += [out_dir / OUT_COMPARISON_JSON, out_dir / OUT_COMPARISON_MD]
    for path in out_paths:
        refuse_if_exists(path)

    probes = iter_gold_probes(gold_path)
    if limit is not None:
        probes = probes[: max(0, limit)]

    inference = RemoteExtractionInference(client, endpoint)
    try:
        info = await inference.info()
        health = await inference.health()
    except ExtractionInferenceError as exc:
        _log(f"[FAIL] preflight: {exc}")
        return 1
    if not is_compatible(info.contract_version):
        _log(f"[FAIL] preflight: incompatible contract_version {info.contract_version!r}")
        return 1
    log_serving_config(info)
    if health.status != "ok":
        _log(f"[FAIL] preflight: service health is {health.status!r}, not 'ok'")
        return 1

    session_id = f"gauntlet-service-{uuid.uuid4().hex[:12]}"
    expect = ExpectStamps(extraction_version=info.extraction_version, model_hash=info.model_hash)
    # Rows and raw responses are written one flushed line per completed case, so a crash
    # keeps partial results. Exclusive-create ("x") backs up the up-front overwrite check.
    # The rows file holds only the per-case fields until the run is scored; it is then
    # replaced with the full scored rows.
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / OUT_ROWS
    with (
        open(rows_path, "x", encoding="utf-8") as rows_fh,
        open(out_dir / OUT_RAW, "x", encoding="utf-8") as raw_fh,
    ):

        def on_case(index: int, total: int, case: CaseResult) -> None:
            _progress(index, total, case)
            _append_jsonl(rows_fh, {**case_row_fields(case), "scored": False})
            raw = raw_row(case)
            if raw is not None:
                _append_jsonl(raw_fh, raw)

        result = await run_cases(
            inference,
            probes,
            expect=expect,
            session_id=session_id,
            recorded_at=recorded_at,
            max_seconds=max_minutes * 60.0,
            drain_s=drain_s,
            clock=clock,
            sleep=sleep,
            on_case=on_case,
        )

    summary, report = build_run_summary(
        probes=probes,
        result=result,
        gold_path=gold_path,
        info=info,
        health=health.model_dump(mode="json"),
        endpoint=endpoint,
        session_id=session_id,
        recorded_at=recorded_at,
        client_timeout_s=client_timeout_s,
        max_minutes=max_minutes,
        limit=limit,
        drain_s=drain_s,
    )
    rows = build_case_rows(report, result)

    _replace_jsonl(rows_path, rows)
    (out_dir / OUT_SUMMARY).write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    if baseline_dir is not None:
        comparison = build_comparison(
            summary, rows, host_scope_decisions(result.cases), load_baseline(baseline_dir, probes)
        )
        (out_dir / OUT_COMPARISON_JSON).write_text(
            json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8"
        )
        (out_dir / OUT_COMPARISON_MD).write_text(render_comparison_md(comparison), encoding="utf-8")

    if not summary["complete"]:
        # `empty` is a 200 and may be correct (negative controls), so it is not a failure.
        failed = {
            k: v for k, v in summary["failure_classes"].items() if v and k not in _SUCCESS_CLASSES
        }
        _log(
            f"[FAIL] gauntlet incomplete: {summary['matched_probes']}/{summary['total_probes']} "
            f"probes matched; failed={summary['failed_cases']} {failed}; "
            f"not_run={summary['cases_not_run']}; stopped_reason={result.stopped_reason}; "
            f"empty_200s={summary['failure_classes'][OUTCOME_EMPTY]}"
        )
        return 1
    print(
        json.dumps(
            {
                "rel_precision": summary["rel_precision"],
                "typing_accuracy": summary["typing_accuracy"],
            }
        )
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the CLI arguments."""
    p = argparse.ArgumentParser(description="Extraction gold gauntlet against the service.")
    p.add_argument("--endpoint", required=True, help="Service root, e.g. http://host:8090")
    p.add_argument("--out", required=True, help="Output directory (created if absent).")
    p.add_argument("--gold", default=DEFAULT_GOLD_CORPUS, help="Gold corpus JSONL.")
    p.add_argument(
        "--baseline", default=None, help="Baseline run directory (extraction.jsonl ...)."
    )
    p.add_argument(
        "--client-timeout-s",
        type=float,
        default=DEFAULT_CLIENT_TIMEOUT_S,
        help="Client read timeout; must exceed one job's worst case (scope + 2 attempts).",
    )
    p.add_argument(
        "--drain-s",
        type=float,
        default=DEFAULT_DRAIN_S,
        help="Pause after a client-side timeout so the abandoned job drains off the GPU.",
    )
    p.add_argument("--max-minutes", type=float, default=DEFAULT_MAX_MINUTES)
    p.add_argument("--limit", type=int, default=None, help="Run only the first N probes.")
    p.add_argument("--recorded-at", default=MIST_FIXED_CLOCK_PIN, help="ISO-8601 reference date.")
    return p.parse_args(argv)


async def _amain(args: argparse.Namespace, client: httpx.AsyncClient | None) -> int:
    gold_path = Path(args.gold)
    if not gold_path.is_absolute():
        gold_path = _REPO_ROOT / gold_path
    kwargs: dict[str, Any] = {
        "endpoint": args.endpoint,
        "out_dir": Path(args.out),
        "gold_path": gold_path,
        "baseline_dir": Path(args.baseline) if args.baseline else None,
        "recorded_at": args.recorded_at,
        "client_timeout_s": args.client_timeout_s,
        "drain_s": args.drain_s,
        "max_minutes": args.max_minutes,
        "limit": args.limit,
    }
    if client is not None:
        return await execute(client=client, **kwargs)
    timeout = httpx.Timeout(args.client_timeout_s, connect=CONNECT_TIMEOUT_S)
    async with httpx.AsyncClient(timeout=timeout) as own_client:
        return await execute(client=own_client, **kwargs)


def main(argv: list[str] | None = None, *, client: httpx.AsyncClient | None = None) -> int:
    """CLI entry point. `client` is injected by tests; the CLI builds its own."""
    args = parse_args(argv)
    try:
        return asyncio.run(_amain(args, client))
    except (FileExistsError, OSError, ValueError) as exc:
        _log(f"[FAIL] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
