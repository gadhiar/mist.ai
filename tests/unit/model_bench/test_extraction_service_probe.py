"""Tests for `scripts.model_bench.probes.extraction_service`.

The probe is driven against a fake extraction service: a small FastAPI app built only
from `backend.extraction_contract`, reached through `httpx.ASGITransport` (no network).
`ScriptedTransport` adds the two failures an ASGI app cannot produce itself (a refused
connection and a client-side read timeout). All fixtures are small synthetic ones built
in `tmp_path`, except the echo test that reads the real gold corpus from the repo.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import httpx  # noqa: E402
import pytest  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.responses import JSONResponse, PlainTextResponse  # noqa: E402

from backend.extraction_backlog.errors import (  # noqa: E402
    InferenceResponseInvalidError,
    InferenceServiceError,
    InferenceUnreachableError,
)
from backend.extraction_contract.models import (  # noqa: E402
    CONTRACT_VERSION,
    ERROR_HTTP_STATUS,
    ErrorCode,
    ExtractionPayload,
    ExtractRequest,
    ExtractResponse,
    HealthResponse,
    InfoResponse,
    ResultStamps,
    ScopeOut,
    TimingsMs,
    error_envelope,
)
from backend.knowledge.extraction.scope_classifier import SCOPE_USER_TEMPLATE  # noqa: E402
from scripts.eval_harness.score_extraction_run import GoldProbe, iter_gold_probes  # noqa: E402
from scripts.golden_log.native_shape import (  # noqa: E402
    build_native_entity,
    build_native_relationship,
)
from scripts.model_bench.probes import extraction_service as probe  # noqa: E402
from scripts.model_bench.probes.extraction import MIST_FIXED_CLOCK_PIN  # noqa: E402

REAL_GOLD = _REPO_ROOT / "data" / "ingest" / "extraction-gold-2026-06-14.jsonl"
ENDPOINT = "http://fake-extraction"
EXTRACTION_VERSION = "ev-test"
MODEL_HASH = "hash-under-test"


@pytest.fixture(autouse=True)
def _fast_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use a small bootstrap B so a run is not dominated by resampling."""
    monkeypatch.setattr(probe, "load_bootstrap_params", lambda: (7, 200, 0.95, 1.96))


# ---------------------------------------------------------------------------
# Fake service
# ---------------------------------------------------------------------------


def echo_payload(gold: GoldProbe) -> dict[str, list[dict[str, Any]]]:
    """The raw-LLM-shaped payload that scores as exactly the gold probe."""
    return {
        "entities": [
            build_native_entity(entity_id=e.id, entity_type=e.type) for e in gold.entities
        ],
        "relationships": [
            build_native_relationship(
                source=r.source,
                target=r.target,
                predicate=r.predicate,
                assertion_kind=None if r.assertion_kind == "assert" else r.assertion_kind,
                valid_from=r.valid_from,
                valid_to=r.valid_to,
            )
            for r in gold.relationships
        ],
    }


@dataclass
class FakeState:
    """What the fake service reports and does; mutate it mid-test."""

    gold: dict[str, GoldProbe]
    contract_version: str = CONTRACT_VERSION
    health_status: str = "ok"
    # utterance -> behaviour name; anything absent echoes the gold probe.
    behaviour: dict[str, str] = field(default_factory=dict)
    scope_for: dict[str, tuple[str, float]] = field(default_factory=dict)
    connect_error_for: set[str] = field(default_factory=set)
    read_timeout_for: set[str] = field(default_factory=set)
    received: list[ExtractRequest] = field(default_factory=list)
    in_flight: int = 0
    max_in_flight: int = 0
    on_request: Callable[[int], None] | None = None


def _ok_response(
    req: ExtractRequest,
    state: FakeState,
    payload: dict[str, list[dict[str, Any]]],
    *,
    attempts: int = 1,
    warnings: tuple[str, ...] = (),
) -> ExtractResponse:
    label, confidence = state.scope_for.get(req.utterance, ("user-scope", 0.9))
    return ExtractResponse(
        contract_version=CONTRACT_VERSION,
        job_id=req.job_id,
        outcome="extracted",
        scope=ScopeOut(label=label, confidence=confidence),
        payload=ExtractionPayload(**payload),
        derivation=None,
        stamps=ResultStamps(
            extraction_version=EXTRACTION_VERSION,
            model_hash=MODEL_HASH,
            prompt_sha256="0" * 64,
            llama_cpp_build="b0",
            adapter="fake",
        ),
        timings_ms=TimingsMs(scope=1.0, extract=2.0, derive=None, total=3.0),
        attempts=attempts,
        warnings=list(warnings),
    )


def _error(code: ErrorCode, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=ERROR_HTTP_STATUS[code],
        content=error_envelope(code, message).model_dump(mode="json"),
    )


def build_fake_app(state: FakeState) -> FastAPI:
    """The service's three endpoints, answering from `state`."""
    app = FastAPI()

    @app.get("/v1/info")
    async def info() -> InfoResponse:
        return InfoResponse(
            contract_version=state.contract_version,
            extraction_version=EXTRACTION_VERSION,
            model_hash=MODEL_HASH,
            model_file="fake.gguf",
            llama_cpp_build="b0",
            adapter="gptoss",
            location_label="test-box",
        )

    @app.get("/v1/health")
    async def health() -> HealthResponse:
        return HealthResponse(status=state.health_status, llm_reachable=True, uptime_s=1.0)

    @app.post("/v1/extract")
    async def extract(req: ExtractRequest):
        state.received.append(req)
        state.in_flight += 1
        state.max_in_flight = max(state.max_in_flight, state.in_flight)
        try:
            # Yield a few times so an overlapping (concurrent) client would be observed.
            for _ in range(3):
                await asyncio.sleep(0)
            if state.on_request is not None:
                state.on_request(len(state.received))
            payload = echo_payload(state.gold[req.utterance])
            behaviour = state.behaviour.get(req.utterance, "echo")
            if behaviour == "timeout":
                return _error(ErrorCode.TIMEOUT, "Extraction LLM call timed out after 1s")
            if behaviour == "unparsable":
                return _error(
                    ErrorCode.UPSTREAM_LLM, "Extraction output unparsable after 2 attempts"
                )
            if behaviour == "upstream":
                return _error(ErrorCode.UPSTREAM_LLM, "Extraction LLM call failed: boom")
            if behaviour == "epoch":
                return _error(ErrorCode.EPOCH_MISMATCH, "expect stamps do not match")
            if behaviour == "gateway":
                return PlainTextResponse("bad gateway", status_code=502)
            if behaviour == "invalid":
                return JSONResponse({"unexpected": True})
            if behaviour == "empty":
                return _ok_response(req, state, {"entities": [], "relationships": []})
            if behaviour == "scope_warn":
                state.scope_for[req.utterance] = ("unknown", 0.0)
                return _ok_response(req, state, payload, warnings=("scope_classification_failed",))
            if behaviour == "attempts2":
                return _ok_response(req, state, payload, attempts=2)
            if behaviour == "extra_entity":
                payload["entities"].append(
                    build_native_entity(entity_id="bonus", entity_type="Technology")
                )
                return _ok_response(req, state, payload)
            return _ok_response(req, state, payload)
        finally:
            state.in_flight -= 1

    return app


class ScriptedTransport(httpx.AsyncBaseTransport):
    """ASGI transport that also raises a connect error / read timeout for scripted utterances."""

    def __init__(self, state: FakeState, app: FastAPI) -> None:
        self._state = state
        self._inner = httpx.ASGITransport(app=app)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/extract":
            utterance = json.loads(request.content)["utterance"]
            if utterance in self._state.connect_error_for:
                raise httpx.ConnectError("connection refused", request=request)
            if utterance in self._state.read_timeout_for:
                raise httpx.ReadTimeout("read timed out", request=request)
        return await self._inner.handle_async_request(request)


class FakeClock:
    """A settable monotonic clock."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class FakeSleep:
    """Records requested sleeps instead of sleeping."""

    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def write_gold(path: Path, n: int) -> Path:
    """A synthetic n-probe gold corpus: 2 entities and 1 relationship each."""
    lines = []
    for i in range(1, n + 1):
        lines.append(
            json.dumps(
                {
                    "utterance": f"I use tool{i} daily",
                    "tag": f"p-{i}",
                    "expected_entities": [
                        {"id": "user", "type": "User"},
                        {"id": f"tool{i}", "type": "Technology"},
                    ],
                    "expected_relationships": [
                        {
                            "source": "user",
                            "source_type": "User",
                            "predicate": "USES",
                            "target": f"tool{i}",
                            "target_type": "Technology",
                        }
                    ],
                }
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def make_world(gold_path: Path) -> tuple[FakeState, httpx.AsyncClient]:
    """A fake service over `gold_path`, and a client wired to it."""
    gold = {p.utterance: p for p in iter_gold_probes(gold_path)}
    state = FakeState(gold=gold)
    client = httpx.AsyncClient(transport=ScriptedTransport(state, build_fake_app(state)))
    return state, client


async def run_execute(
    *,
    state_and_client: tuple[FakeState, httpx.AsyncClient],
    gold_path: Path,
    out_dir: Path,
    **kwargs: Any,
) -> int:
    _, client = state_and_client
    kwargs.setdefault("sleep", FakeSleep())  # a test must never really sleep
    async with client:
        return await probe.execute(
            endpoint=ENDPOINT, out_dir=out_dir, gold_path=gold_path, client=client, **kwargs
        )


def read_rows(out_dir: Path) -> dict[str, dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in (out_dir / probe.OUT_ROWS).read_text(encoding="utf-8").splitlines()
    ]
    return {r["id"]: r for r in rows}


def read_summary(out_dir: Path) -> dict[str, Any]:
    return json.loads((out_dir / probe.OUT_SUMMARY).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The classifier (pure)
# ---------------------------------------------------------------------------


def _response(entities: int = 1, relationships: int = 0) -> ExtractResponse:
    return ExtractResponse(
        contract_version=CONTRACT_VERSION,
        job_id="j",
        outcome="extracted",
        scope=ScopeOut(label="user-scope", confidence=0.9),
        payload=ExtractionPayload(
            entities=[{"id": f"e{i}", "type": "Technology"} for i in range(entities)],
            relationships=[{"source": "a", "target": "b", "type": "USES"}] * relationships,
        ),
        derivation=None,
        stamps=ResultStamps(
            extraction_version="v",
            model_hash="h",
            prompt_sha256="0" * 64,
            llama_cpp_build="b",
            adapter="a",
        ),
        timings_ms=TimingsMs(scope=None, extract=1.0, derive=None, total=1.0),
        attempts=1,
    )


def _service_error(code: ErrorCode, message: str, status: int | None) -> InferenceServiceError:
    return InferenceServiceError(message, code=code, retryable=True, http_status=status)


@pytest.mark.parametrize(
    ("response", "error", "expected"),
    [
        (_response(1, 0), None, (probe.OUTCOME_OK, None)),
        (_response(0, 1), None, (probe.OUTCOME_OK, None)),
        (_response(0, 0), None, (probe.OUTCOME_EMPTY, None)),
        (None, _service_error(ErrorCode.TIMEOUT, "t", 504), (probe.OUTCOME_TIMEOUT, "timeout")),
        (
            None,
            _service_error(ErrorCode.TIMEOUT, "t", None),
            (probe.OUTCOME_CLIENT_TIMEOUT, "timeout"),
        ),
        (
            None,
            _service_error(
                ErrorCode.UPSTREAM_LLM,
                "upstream_llm (HTTP 502): Extraction output unparsable after 2 attempts",
                502,
            ),
            (probe.OUTCOME_REPAIR_EXHAUSTED, "upstream_llm"),
        ),
        (
            None,
            _service_error(ErrorCode.UPSTREAM_LLM, "Extraction LLM call failed: boom", 502),
            (probe.OUTCOME_UPSTREAM_LLM, "upstream_llm"),
        ),
        (
            None,
            _service_error(ErrorCode.EPOCH_MISMATCH, "epoch", 409),
            (probe.OUTCOME_OTHER_ERROR, "epoch_mismatch"),
        ),
        (
            None,
            _service_error(ErrorCode.MODEL_LOADING, "loading", 503),
            (probe.OUTCOME_OTHER_ERROR, "model_loading"),
        ),
        (
            None,
            _service_error(ErrorCode.CONTRACT_MISMATCH, "cm", 422),
            (probe.OUTCOME_OTHER_ERROR, "contract_mismatch"),
        ),
        (None, InferenceUnreachableError("down"), (probe.OUTCOME_UNREACHABLE, None)),
        (None, InferenceResponseInvalidError("bad"), (probe.OUTCOME_INVALID_RESPONSE, None)),
        (None, httpx.DecodingError("x"), (probe.OUTCOME_OTHER_ERROR, "DecodingError")),
    ],
)
def test_classify_outcome(
    response: ExtractResponse | None, error: BaseException | None, expected: tuple[str, str | None]
) -> None:
    assert probe.classify_outcome(response=response, error=error) == expected


def test_response_flags() -> None:
    plain = _response()
    assert probe.response_flags(plain) == (False, False)
    flagged = plain.model_copy(update={"warnings": ["scope_classification_failed"], "attempts": 2})
    assert probe.response_flags(flagged) == (True, True)


# ---------------------------------------------------------------------------
# Full runs against the fake service
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_echo_service_on_real_corpus_is_perfect_complete_and_sequential(
    tmp_path: Path,
) -> None:
    world = make_world(REAL_GOLD)
    state, _ = world
    out = tmp_path / "out"

    code = await run_execute(state_and_client=world, gold_path=REAL_GOLD, out_dir=out)

    assert code == 0
    summary = read_summary(out)
    assert summary["complete"] is True
    assert summary["total_probes"] == summary["matched_probes"] == len(state.received) == 60
    assert summary["entity_precision"] == 1.0
    assert summary["entity_recall"] == 1.0
    assert summary["rel_precision"] == 1.0
    assert summary["rel_recall"] == 1.0
    assert summary["cases_not_run"] == 0
    assert summary["run"]["stopped_reason"] is None

    # Preflight is recorded.
    info = summary["endpoint"]["info"]
    assert info["extraction_version"] == EXTRACTION_VERSION
    assert info["model_hash"] == MODEL_HASH
    assert info["adapter"] == "gptoss"
    assert info["location_label"] == "test-box"
    assert info["contract_version"] == CONTRACT_VERSION
    assert summary["endpoint"]["health"]["status"] == "ok"

    # Strictly sequential, fresh ids, fixed clock, one session, info's stamps as expect.
    assert state.max_in_flight == 1
    job_ids = [r.job_id for r in state.received]
    assert len(set(job_ids)) == 60
    assert len({r.event_id for r in state.received}) == 60
    assert len({r.request_id for r in state.received}) == 60
    assert len({r.session_id for r in state.received}) == 1
    assert all(r.recorded_at == MIST_FIXED_CLOCK_PIN for r in state.received)
    assert all(r.turn_index == 0 and r.conversation_history == [] for r in state.received)
    assert all(r.derivation is None for r in state.received)
    assert all(
        (r.expect.extraction_version, r.expect.model_hash) == (EXTRACTION_VERSION, MODEL_HASH)
        for r in state.received
    )

    rows = read_rows(out)
    assert {r["job_id"] for r in rows.values()} == set(job_ids)
    assert all(r["latency_ms"] >= 0 for r in rows.values())
    raw = [
        json.loads(line) for line in (out / probe.OUT_RAW).read_text(encoding="utf-8").splitlines()
    ]
    assert len(raw) == 60
    assert summary["latency_ms"]["n"] == 60
    assert set(summary["latency_ms"]) == {"n", "p50", "p95", "max", "mean"}
    # Bootstrap and Wilson CIs are present.
    assert summary["bootstrap"]["entity_recall"] is not None
    assert summary["wilson"]["entity_precision"] is not None


@pytest.mark.asyncio
async def test_each_failure_class_is_recorded_and_scored_as_false_negatives(
    tmp_path: Path,
) -> None:
    gold = write_gold(tmp_path / "gold.jsonl", 11)
    world = make_world(gold)
    state, _ = world
    behaviours = {
        2: "timeout",
        3: "unparsable",
        4: "upstream",
        5: "epoch",
        6: "empty",
        7: "scope_warn",
        8: "attempts2",
        9: "invalid",
        10: "gateway",
    }
    for i, name in behaviours.items():
        state.behaviour[f"I use tool{i} daily"] = name
    state.connect_error_for.add("I use tool11 daily")
    out = tmp_path / "out"

    code = await run_execute(state_and_client=world, gold_path=gold, out_dir=out)

    assert code == 1  # incomplete, but everything is written
    summary = read_summary(out)
    rows = read_rows(out)
    assert {pid: rows[pid]["outcome_class"] for pid in rows} == {
        "p-1": "ok",
        "p-2": "timeout",
        "p-3": "schema_repair_exhausted",
        "p-4": "upstream_llm",
        "p-5": "other_error",
        "p-6": "empty",
        "p-7": "ok",
        "p-8": "ok",
        "p-9": "invalid_response",
        "p-10": "unreachable",
        "p-11": "unreachable",
    }
    assert rows["p-5"]["error_code"] == "epoch_mismatch"
    assert summary["other_error_codes"] == {"epoch_mismatch": 1}
    assert summary["failure_classes"] == {
        "ok": 3,
        "empty": 1,
        "timeout": 1,
        "client_timeout": 0,
        "schema_repair_exhausted": 1,
        "upstream_llm": 1,
        "unreachable": 2,
        "invalid_response": 1,
        "other_error": 1,
    }
    # Independent flags on successful cases.
    assert rows["p-7"]["scope_classification_failed"] is True
    assert rows["p-7"]["repaired"] is False
    assert rows["p-8"]["repaired"] is True and rows["p-8"]["attempts"] == 2
    assert summary["flags"] == {"scope_classification_failed": 1, "repaired": 1}
    assert summary["scope"]["label_counts"]["unknown"] == 1

    # A failed case is unmatched and scores as all false negatives (2 entities, 1 rel).
    for pid in ("p-2", "p-3", "p-4", "p-5", "p-9", "p-10", "p-11"):
        assert rows[pid]["matched"] is False
        assert rows[pid]["errored"] is True
        assert (rows[pid]["entity_fn"], rows[pid]["rel_fn"]) == (2, 1)
        assert (rows[pid]["entity_fp"], rows[pid]["rel_fp"]) == (0, 0)
    # An empty 200 on a non-negative probe is matched but all-FN.
    assert rows["p-6"]["matched"] is True
    assert rows["p-6"]["outcome_class"] == "empty"
    assert rows["p-6"]["entity_fn"] == 2
    assert rows["p-1"]["matched"] is True and rows["p-1"]["entity_fn"] == 0
    assert summary["matched_probes"] == 4
    assert summary["complete"] is False
    assert sorted(summary["unmatched_probe_ids"]) == sorted(
        ["p-2", "p-3", "p-4", "p-5", "p-9", "p-10", "p-11"]
    )
    # Failed cases are not in the raw file; successful ones are.
    raw_ids = {
        json.loads(line)["id"]
        for line in (out / probe.OUT_RAW).read_text(encoding="utf-8").splitlines()
    }
    assert raw_ids == {"p-1", "p-6", "p-7", "p-8"}
    # Latency over completed (200) cases only.
    assert summary["latency_ms"]["n"] == 4
    assert summary["latency_ms_all_cases"]["n"] == 11
    assert set(summary["latency_notes"]) == {"latency_ms", "latency_ms_all_cases"}

    # Failed cases are counted and the summary warns that precision/typing exclude them.
    assert summary["failed_cases"] == 7
    assert sorted(summary["failed_probe_ids"]) == sorted(summary["unmatched_probe_ids"])
    assert summary["cases_not_run"] == 0
    warning = summary["interpretation_warning"]
    assert "7 case(s) failed" in warning
    assert "EXCLUDE" in warning and "INCLUDE" in warning
    assert "precision" in warning.lower() and "recall" in warning.lower()


@pytest.mark.asyncio
async def test_clean_run_has_no_interpretation_warning(tmp_path: Path) -> None:
    gold = write_gold(tmp_path / "gold.jsonl", 2)
    out = tmp_path / "out"

    code = await run_execute(state_and_client=make_world(gold), gold_path=gold, out_dir=out)

    assert code == 0
    summary = read_summary(out)
    assert summary["interpretation_warning"] is None
    assert summary["failed_cases"] == 0 and summary["failed_probe_ids"] == []
    assert all("scored" not in row for row in read_rows(out).values())


@pytest.mark.asyncio
async def test_client_read_timeout_is_its_own_class_and_drains_before_the_next_request(
    tmp_path: Path,
) -> None:
    gold = write_gold(tmp_path / "gold.jsonl", 3)
    world = make_world(gold)
    state, _ = world
    state.read_timeout_for.add("I use tool1 daily")
    sleep = FakeSleep()
    out = tmp_path / "out"

    await run_execute(
        state_and_client=world, gold_path=gold, out_dir=out, drain_s=42.0, sleep=sleep
    )

    rows = read_rows(out)
    assert rows["p-1"]["outcome_class"] == "client_timeout"
    assert rows["p-1"]["http_status"] is None
    assert rows["p-2"]["outcome_class"] == "ok"
    assert sleep.calls == [42.0]  # once, after the client timeout only
    summary = read_summary(out)
    assert summary["failure_classes"]["client_timeout"] == 1
    assert summary["failure_classes"]["timeout"] == 0
    assert summary["run"]["drain_s"] == 42.0


@pytest.mark.asyncio
async def test_service_504_is_a_timeout_and_does_not_drain(tmp_path: Path) -> None:
    gold = write_gold(tmp_path / "gold.jsonl", 2)
    world = make_world(gold)
    world[0].behaviour["I use tool1 daily"] = "timeout"
    sleep = FakeSleep()

    await run_execute(state_and_client=world, gold_path=gold, out_dir=tmp_path / "out", sleep=sleep)

    assert read_rows(tmp_path / "out")["p-1"]["outcome_class"] == "timeout"
    assert sleep.calls == []


@pytest.mark.asyncio
async def test_no_drain_after_a_client_timeout_on_the_last_probe(tmp_path: Path) -> None:
    gold = write_gold(tmp_path / "gold.jsonl", 2)
    world = make_world(gold)
    world[0].read_timeout_for.add("I use tool2 daily")
    sleep = FakeSleep()

    await run_execute(state_and_client=world, gold_path=gold, out_dir=tmp_path / "out", sleep=sleep)

    assert sleep.calls == []


@pytest.mark.asyncio
async def test_two_consecutive_client_timeouts_stop_the_run(tmp_path: Path) -> None:
    gold = write_gold(tmp_path / "gold.jsonl", 5)
    world = make_world(gold)
    state, _ = world
    state.read_timeout_for.update({"I use tool2 daily", "I use tool3 daily"})
    sleep = FakeSleep()
    out = tmp_path / "out"

    code = await run_execute(
        state_and_client=world, gold_path=gold, out_dir=out, drain_s=5.0, sleep=sleep
    )

    assert code == 1
    summary = read_summary(out)
    assert summary["cases_run"] == 3
    assert summary["not_run_probe_ids"] == ["p-4", "p-5"]
    assert summary["run"]["stopped_reason"] == probe.STOP_CLIENT_TIMEOUT
    # The scripted timeouts fail in the transport, so only p-1 reached the app; p-4 and p-5
    # were never sent (not_run above).
    assert len(state.received) == 1
    assert sleep.calls == [5.0]  # drained after the first only; the second stopped the run


@pytest.mark.asyncio
async def test_non_consecutive_client_timeouts_do_not_stop_the_run(tmp_path: Path) -> None:
    gold = write_gold(tmp_path / "gold.jsonl", 5)
    world = make_world(gold)
    world[0].read_timeout_for.update({"I use tool1 daily", "I use tool3 daily"})
    sleep = FakeSleep()
    out = tmp_path / "out"

    await run_execute(state_and_client=world, gold_path=gold, out_dir=out, sleep=sleep)

    summary = read_summary(out)
    assert summary["cases_run"] == 5
    assert summary["run"]["stopped_reason"] is None
    assert len(sleep.calls) == 2


def test_default_client_timeout_covers_one_jobs_worst_case() -> None:
    assert probe.DEFAULT_CLIENT_TIMEOUT_S == 960.0
    assert probe.DEFAULT_DRAIN_S == 120.0
    args = probe.parse_args(["--endpoint", ENDPOINT, "--out", "x"])
    assert args.client_timeout_s == 960.0 and args.drain_s == 120.0


def test_unparsable_marker_matches_the_message_the_engine_builds() -> None:
    """A rewording of the engine's message must fail this test, not silently misclassify."""
    source = (_REPO_ROOT / "backend" / "extraction_service" / "engine.py").read_text(
        encoding="utf-8"
    )
    assert "unparsable after" in source
    match = re.search(r'f"(Extraction output unparsable after \{[^}]+\} attempts)"', source)
    assert match is not None, "engine.py no longer builds the 'unparsable after N attempts' message"
    message = re.sub(r"\{[^}]+\}", "2", match.group(1))
    assert probe._UNPARSABLE_RE.search(message)
    error = InferenceServiceError(
        f"upstream_llm (HTTP 502): {message}",
        code=ErrorCode.UPSTREAM_LLM,
        retryable=True,
        http_status=502,
    )
    assert probe.classify_outcome(response=None, error=error)[0] == probe.OUTCOME_REPAIR_EXHAUSTED


@pytest.mark.asyncio
async def test_rows_and_raw_are_written_incrementally_and_survive_a_crash(
    tmp_path: Path,
) -> None:
    gold = write_gold(tmp_path / "gold.jsonl", 5)
    world = make_world(gold)
    state, _ = world

    def crash_on_third(n_received: int) -> None:
        if n_received == 3:
            raise RuntimeError("simulated crash of the service app")

    state.on_request = crash_on_third
    out = tmp_path / "out"

    with pytest.raises(RuntimeError):
        await run_execute(state_and_client=world, gold_path=gold, out_dir=out)

    row_lines = (out / probe.OUT_ROWS).read_text(encoding="utf-8").splitlines()
    raw_lines = (out / probe.OUT_RAW).read_text(encoding="utf-8").splitlines()
    assert [json.loads(x)["id"] for x in row_lines] == ["p-1", "p-2"]
    assert all(json.loads(x)["scored"] is False for x in row_lines)
    assert [json.loads(x)["id"] for x in raw_lines] == ["p-1", "p-2"]
    assert not (out / probe.OUT_SUMMARY).exists()  # the crash came before scoring


@pytest.mark.asyncio
async def test_three_consecutive_unreachable_stops_the_run(tmp_path: Path) -> None:
    gold = write_gold(tmp_path / "gold.jsonl", 6)
    world = make_world(gold)
    state, _ = world
    state.connect_error_for.update({"I use tool2 daily", "I use tool3 daily", "I use tool4 daily"})
    out = tmp_path / "out"

    code = await run_execute(state_and_client=world, gold_path=gold, out_dir=out)

    assert code == 1
    summary = read_summary(out)
    assert summary["cases_run"] == 4
    assert summary["not_run_probe_ids"] == ["p-5", "p-6"]
    assert summary["run"]["stopped_reason"] == probe.STOP_UNREACHABLE
    rows = read_rows(out)
    assert rows["p-5"]["outcome_class"] == "not_run" and rows["p-6"]["outcome_class"] == "not_run"
    assert rows["p-5"]["matched"] is False


@pytest.mark.asyncio
async def test_non_consecutive_unreachable_does_not_stop_the_run(tmp_path: Path) -> None:
    gold = write_gold(tmp_path / "gold.jsonl", 6)
    world = make_world(gold)
    state, _ = world
    state.connect_error_for.update({"I use tool1 daily", "I use tool2 daily", "I use tool4 daily"})
    out = tmp_path / "out"

    await run_execute(state_and_client=world, gold_path=gold, out_dir=out)

    summary = read_summary(out)
    assert summary["cases_run"] == 6
    assert summary["run"]["stopped_reason"] is None
    assert summary["failure_classes"]["unreachable"] == 3


@pytest.mark.asyncio
async def test_max_minutes_stops_cleanly_and_records_not_run(tmp_path: Path) -> None:
    gold = write_gold(tmp_path / "gold.jsonl", 5)
    world = make_world(gold)
    state, _ = world
    clock = FakeClock()

    def advance(n_received: int) -> None:
        if n_received == 2:
            clock.now += 120.0  # two minutes pass during the second call

    state.on_request = advance
    out = tmp_path / "out"

    code = await run_execute(
        state_and_client=world, gold_path=gold, out_dir=out, max_minutes=1.0, clock=clock
    )

    assert code == 1
    summary = read_summary(out)
    assert summary["cases_run"] == 2
    assert summary["not_run_probe_ids"] == ["p-3", "p-4", "p-5"]
    assert summary["run"]["stopped_reason"] == probe.STOP_MAX_MINUTES
    assert summary["complete"] is False
    assert len(state.received) == 2
    rows = read_rows(out)
    assert rows["p-3"]["outcome_class"] == "not_run"
    assert rows["p-1"]["outcome_class"] == "ok"


@pytest.mark.asyncio
async def test_limit_runs_only_the_first_n_probes(tmp_path: Path) -> None:
    gold = write_gold(tmp_path / "gold.jsonl", 5)
    world = make_world(gold)
    out = tmp_path / "out"

    code = await run_execute(state_and_client=world, gold_path=gold, out_dir=out, limit=2)

    assert code == 0
    summary = read_summary(out)
    assert summary["total_probes"] == 2 and summary["run"]["limit"] == 2
    assert len(world[0].received) == 2


@pytest.mark.asyncio
async def test_recorded_at_override_is_sent(tmp_path: Path) -> None:
    gold = write_gold(tmp_path / "gold.jsonl", 1)
    world = make_world(gold)

    await run_execute(
        state_and_client=world,
        gold_path=gold,
        out_dir=tmp_path / "out",
        recorded_at="2026-09-01T00:00:00+00:00",
    )

    assert world[0].received[0].recorded_at == "2026-09-01T00:00:00+00:00"


# ---------------------------------------------------------------------------
# Preflight and overwrite refusal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field_name", "value"),
    [("contract_version", "2.0.0"), ("health_status", "loading"), ("health_status", "degraded")],
)
async def test_preflight_refuses_incompatible_or_unhealthy_service(
    tmp_path: Path, field_name: str, value: str
) -> None:
    gold = write_gold(tmp_path / "gold.jsonl", 2)
    world = make_world(gold)
    setattr(world[0], field_name, value)
    out = tmp_path / "out"

    code = await run_execute(state_and_client=world, gold_path=gold, out_dir=out)

    assert code == 1
    assert world[0].received == []
    assert not (out / probe.OUT_SUMMARY).exists()


@pytest.mark.asyncio
async def test_preflight_refuses_an_unreachable_service(tmp_path: Path) -> None:
    gold = write_gold(tmp_path / "gold.jsonl", 1)

    class Down(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

    async with httpx.AsyncClient(transport=Down()) as client:
        code = await probe.execute(
            endpoint=ENDPOINT, out_dir=tmp_path / "out", gold_path=gold, client=client
        )

    assert code == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("existing", [probe.OUT_ROWS, probe.OUT_RAW, probe.OUT_SUMMARY])
async def test_refuses_to_overwrite_before_any_request(tmp_path: Path, existing: str) -> None:
    gold = write_gold(tmp_path / "gold.jsonl", 2)
    world = make_world(gold)
    out = tmp_path / "out"
    out.mkdir()
    (out / existing).write_text("keep me", encoding="utf-8")

    async with world[1] as client:
        with pytest.raises(FileExistsError):
            await probe.execute(endpoint=ENDPOINT, out_dir=out, gold_path=gold, client=client)

    assert (out / existing).read_text(encoding="utf-8") == "keep me"
    assert world[0].received == []


@pytest.mark.asyncio
async def test_refuses_to_overwrite_comparison_files(tmp_path: Path) -> None:
    gold = write_gold(tmp_path / "gold.jsonl", 1)
    world = make_world(gold)
    out = tmp_path / "out"
    out.mkdir()
    (out / probe.OUT_COMPARISON_MD).write_text("keep me", encoding="utf-8")

    async with world[1] as client:
        with pytest.raises(FileExistsError):
            await probe.execute(
                endpoint=ENDPOINT,
                out_dir=out,
                gold_path=gold,
                client=client,
                baseline_dir=tmp_path / "baseline",
            )


def test_main_returns_1_on_overwrite_and_0_on_a_clean_run(tmp_path: Path) -> None:
    gold = write_gold(tmp_path / "gold.jsonl", 2)
    out = tmp_path / "out"
    argv = ["--endpoint", ENDPOINT, "--out", str(out), "--gold", str(gold)]

    state, client = make_world(gold)
    assert probe.main(argv, client=client) == 0
    assert len(state.received) == 2

    state2, client2 = make_world(gold)
    assert probe.main(argv, client=client2) == 1  # outputs now exist
    assert state2.received == []


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------


def _scope_record(utterance: str, label: str, confidence: float) -> dict[str, Any]:
    return {
        "phase": "llm_call",
        "call_site": probe.SCOPE_CALL_SITE,
        "request": {
            "messages": [
                {"role": "system", "content": "scope prompt"},
                {"role": "user", "content": SCOPE_USER_TEMPLATE.format(utterance=utterance)},
            ]
        },
        "response": {
            "content": json.dumps({"scope": label, "confidence": confidence, "reasoning": "x"})
        },
    }


def write_baseline(directory: Path, n: int, *, with_debug: bool = True) -> Path:
    """A synthetic baseline in the existing probe's layout: everything matched, all user-scope."""
    directory.mkdir(parents=True)
    rows = [
        {
            "id": f"p-{i}",
            "matched": True,
            "errored": False,
            "gold_entities": 2,
            "gold_relationships": 1,
            "entity_fp": 0,
            "entity_fn": 0,
            "rel_fp": 0,
            "rel_fn": 0,
        }
        for i in range(1, n + 1)
    ]
    (directory / "extraction.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    summary = {
        "entity_precision": 1.0,
        "entity_recall": 1.0,
        "rel_precision": 1.0,
        "rel_recall": 1.0,
        "rel_f1": 1.0,
        "typing_accuracy": 1.0,
        "related_to_rate": 0.0,
        "valid_time_accuracy": 1.0,
        "gold_corpus_sha256": "abc",
        "bootstrap": {
            "entity_precision": [0.9, 1.0],
            "entity_recall": [0.95, 1.0],
            "rel_precision": [0.9, 1.0],
            "rel_recall": [0.95, 1.0],
        },
        "wilson": {"typing_accuracy": [0.9, 1.0]},
    }
    (directory / "extraction_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    if with_debug:
        records = [
            _scope_record(f"I use tool{i} daily", "user-scope", 0.95) for i in range(1, n + 1)
        ]
        # Noise the loader must skip: another call site and a malformed line.
        noise = {"phase": "llm_call", "call_site": "extraction.ontology", "request": {}}
        text = "\n".join(json.dumps(r) for r in [noise, *records]) + "\nnot json {{\n"
        (directory / "extraction_debug.jsonl").write_text(text, encoding="utf-8")
    return directory


@pytest.mark.asyncio
async def test_comparison_flags_regressed_probes_and_reports_scope_agreement(
    tmp_path: Path,
) -> None:
    gold = write_gold(tmp_path / "gold.jsonl", 4)
    baseline = write_baseline(tmp_path / "baseline", 4)
    world = make_world(gold)
    state, _ = world
    state.behaviour["I use tool2 daily"] = "timeout"  # matched in baseline, failed on host
    state.behaviour["I use tool3 daily"] = "extra_entity"  # entity_fp 0 -> 1
    state.scope_for["I use tool3 daily"] = ("third-party", 0.5)
    out = tmp_path / "out"

    await run_execute(state_and_client=world, gold_path=gold, out_dir=out, baseline_dir=baseline)

    comparison = json.loads((out / probe.OUT_COMPARISON_JSON).read_text(encoding="utf-8"))
    assert comparison["omissions"] == []
    regressed = {r["id"]: r for r in comparison["regressed"]}
    assert set(regressed) == {"p-2", "p-3"}
    assert "matched in baseline, failed on host" in regressed["p-2"]["reasons"]
    assert regressed["p-2"]["before"]["entity_fn"] == 0
    assert regressed["p-2"]["after"]["entity_fn"] == 2
    assert regressed["p-3"]["reasons"] == ["entity_fp 0 -> 1"]
    assert regressed["p-3"]["after"]["entity_fp"] == 1

    metrics = {m["metric"]: m for m in comparison["metrics"]}
    assert set(metrics) == {k for k, _ in probe.COMPARISON_METRICS}
    # Host entity recall: tp 5 (p-1, p-3, p-4 give 2 each = 6; p-2 all FN) -> 6/8.
    assert metrics["entity_recall"]["host"] == pytest.approx(0.75)
    assert metrics["entity_recall"]["baseline"] == 1.0
    assert metrics["entity_recall"]["delta"] == pytest.approx(-0.25)
    assert metrics["entity_recall"]["baseline_ci"] == [0.95, 1.0]
    assert metrics["entity_recall"]["host_inside_baseline_ci"] is False
    assert metrics["typing_accuracy"]["baseline_ci_source"] == "wilson"
    assert metrics["rel_f1"]["host_inside_baseline_ci"] is None  # baseline has no CI for it
    assert comparison["gold_corpus_sha256"]["match"] is False

    scope = comparison["scope"]["comparison"]
    assert scope["n_compared"] == 3  # p-2 failed on the host, so it has no host decision
    assert scope["n_agree"] == 2
    assert scope["agreement_rate"] == pytest.approx(2 / 3)
    assert [d["id"] for d in scope["disagreements"]] == ["p-3"]
    assert scope["baseline"]["user_scope_rate"] == 1.0
    assert scope["host"]["user_scope_rate"] == pytest.approx(2 / 3)
    assert scope["baseline"]["mean_confidence"] == pytest.approx(0.95)
    assert scope["host"]["mean_confidence"] == pytest.approx((0.9 + 0.5 + 0.9) / 3)

    # The host run failed p-2, so it is incomplete: the comparison says so up front.
    host_run = comparison["host_run"]
    assert host_run["complete"] is False
    assert host_run["failed_cases"] == 1
    assert host_run["failure_classes"]["timeout"] == 1
    assert host_run["cases_not_run"] == 0
    assert "EXCLUDE" in host_run["interpretation_warning"]
    assert comparison["probe_sets"]["differ"] is False

    text = (out / probe.OUT_COMPARISON_MD).read_text(encoding="utf-8")
    assert text.splitlines()[2].startswith("**[INCOMPLETE]")  # banner before any table
    assert text.index("[INCOMPLETE]") < text.index("## Metrics")
    assert "matched probes: 3/4" in text
    assert "failure classes:" in text
    assert "NO (INCOMPLETE run)" in text  # the plain inside-CI verdict is never bare
    assert "PROBE SETS DIFFER" not in text
    assert "p-2" in text and "p-3" in text
    assert "entity recall" in text
    assert "2/3" in text
    assert text.isascii()


@pytest.mark.asyncio
async def test_complete_run_comparison_has_no_incomplete_banner(tmp_path: Path) -> None:
    gold = write_gold(tmp_path / "gold.jsonl", 3)
    baseline = write_baseline(tmp_path / "baseline", 3)
    out = tmp_path / "out"

    code = await run_execute(
        state_and_client=make_world(gold), gold_path=gold, out_dir=out, baseline_dir=baseline
    )

    assert code == 0
    comparison = json.loads((out / probe.OUT_COMPARISON_JSON).read_text(encoding="utf-8"))
    assert comparison["host_run"]["complete"] is True
    assert comparison["host_run"]["interpretation_warning"] is None
    assert comparison["probe_sets"]["differ"] is False
    text = (out / probe.OUT_COMPARISON_MD).read_text(encoding="utf-8")
    assert "INCOMPLETE" not in text and "PROBE SETS DIFFER" not in text
    assert "complete: yes" in text


@pytest.mark.parametrize("via", ["limit", "baseline_larger"])
@pytest.mark.asyncio
async def test_comparison_warns_when_probe_sets_differ(tmp_path: Path, via: str) -> None:
    gold = write_gold(tmp_path / "gold.jsonl", 4)
    baseline = write_baseline(tmp_path / "baseline", 6 if via == "baseline_larger" else 4)
    out = tmp_path / "out"
    kwargs: dict[str, Any] = {"limit": 2} if via == "limit" else {}

    await run_execute(
        state_and_client=make_world(gold),
        gold_path=gold,
        out_dir=out,
        baseline_dir=baseline,
        **kwargs,
    )

    comparison = json.loads((out / probe.OUT_COMPARISON_JSON).read_text(encoding="utf-8"))
    assert comparison["probe_sets"]["differ"] is True
    text = (out / probe.OUT_COMPARISON_MD).read_text(encoding="utf-8")
    assert "PROBE SETS DIFFER" in text
    assert "not directly comparable" in text


@pytest.mark.parametrize("variant", ["empty_dir", "missing_dir", "no_debug"])
@pytest.mark.asyncio
async def test_missing_baseline_files_degrade_to_a_stated_omission(
    tmp_path: Path, variant: str
) -> None:
    gold = write_gold(tmp_path / "gold.jsonl", 2)
    if variant == "empty_dir":
        baseline = tmp_path / "baseline"
        baseline.mkdir()
    elif variant == "missing_dir":
        baseline = tmp_path / "does-not-exist"
    else:
        baseline = write_baseline(tmp_path / "baseline", 2, with_debug=False)
    world = make_world(gold)
    out = tmp_path / "out"

    code = await run_execute(
        state_and_client=world, gold_path=gold, out_dir=out, baseline_dir=baseline
    )

    assert code == 0  # the run itself is complete
    comparison = json.loads((out / probe.OUT_COMPARISON_JSON).read_text(encoding="utf-8"))
    assert comparison["omissions"]
    text = (out / probe.OUT_COMPARISON_MD).read_text(encoding="utf-8")
    assert "Omissions" in text
    if variant == "no_debug":
        assert comparison["metrics"] is not None and comparison["regressed"] == []
        assert comparison["scope"]["comparison"] is None
        assert comparison["scope"]["host"]["n"] == 2  # host scope stats still reported
    else:
        assert comparison["metrics"] is None and comparison["regressed"] is None


def test_read_baseline_scope_recovers_labels_and_skips_noise(tmp_path: Path) -> None:
    gold = write_gold(tmp_path / "gold.jsonl", 3)
    probes = iter_gold_probes(gold)
    lines = [
        json.dumps(_scope_record("I use tool1 daily", "system-scope", 0.7)),
        json.dumps(
            _scope_record("I use   tool2\ndaily", "third-party", 0.6)
        ),  # whitespace-normalised
        json.dumps(_scope_record("a stray utterance", "user-scope", 0.9)),  # not in the corpus
        json.dumps(
            {
                **_scope_record("I use tool3 daily", "user-scope", 0.9),
                "response": {"content": "garbage"},
            }
        ),
        "{ malformed",
    ]
    path = tmp_path / "debug.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    decisions = probe.read_baseline_scope(path, probes)

    assert decisions == {
        "p-1": ("system-scope", 0.7),
        "p-2": ("third-party", 0.6),
        "p-3": ("unknown", 0.0),  # parse_scope_output's failure result, kept as a decision
    }
    assert probe.read_baseline_scope(tmp_path / "absent.jsonl", probes) is None


def test_find_regressions_reports_only_worse_rows() -> None:
    base = {
        "a": {"id": "a", "matched": True, "entity_fp": 1, "entity_fn": 1, "rel_fp": 0, "rel_fn": 0},
        "b": {"id": "b", "matched": True, "entity_fp": 0, "entity_fn": 0, "rel_fp": 0, "rel_fn": 0},
    }
    host = [
        {"id": "a", "matched": True, "entity_fp": 0, "entity_fn": 1, "rel_fp": 0, "rel_fn": 0},
        {"id": "b", "matched": True, "entity_fp": 0, "entity_fn": 0, "rel_fp": 0, "rel_fn": 1},
        {"id": "c", "matched": True, "entity_fp": 9, "entity_fn": 9, "rel_fp": 9, "rel_fn": 9},
    ]
    regressed, missing = probe.find_regressions(host, base)
    assert [r["id"] for r in regressed] == ["b"]
    assert regressed[0]["reasons"] == ["rel_fn 0 -> 1"]
    assert missing == ["c"]
