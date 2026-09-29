"""Test doubles for the extraction-backlog suite.

`FakeExtractionService` is a small ASGI app built ONLY from
`backend.extraction_contract` (it does not import `backend.extraction_service`),
reached through `httpx.ASGITransport` -- no network. `SwitchableTransport`
wraps that transport so a test can take the service "down" (every request
raises `httpx.ConnectError`, which is what a refused connection looks like to
the client).

`FakeGraphCuration` stands in for `CurationPipeline` at the graph boundary: a
stateful in-memory graph that MERGEs nodes by id and edges by
(source, type, target), and records every `curate_and_store` call in order.
`FakeInternalDeriver` does the same for the two `InternalKnowledgeDeriver`
methods the backlog uses.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from backend.extraction_contract.models import (
    CONTRACT_VERSION,
    ERROR_HTTP_STATUS,
    DerivationOut,
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
from backend.knowledge.curation.deduplication import DeduplicationResult
from backend.knowledge.curation.graph_writer import WriteResult
from backend.knowledge.curation.pipeline import CurationResult
from backend.knowledge.curation.reconciliation import ReconcileTurnResult

SERVICE_URL = "http://fake-extraction"


class SimulatedCrashError(Exception):
    """Raised by a fake to model the process dying at a precise point.

    Deliberately NOT a `MistError`/`sqlite3.Error`/`OSError`, so the
    dispatcher's loop does not contain it: the loop task dies, exactly as the
    process would.
    """


def default_payload(req: ExtractRequest) -> tuple[list[dict], list[dict]]:
    """One Technology entity named after the utterance's last word."""
    word = req.utterance.rstrip(".!?").split()[-1].lower()
    return ([{"id": word, "type": "Technology", "name": word.capitalize()}], [])


@dataclass
class FakeServiceState:
    """What the fake service reports and does. Mutate it mid-test to change behaviour."""

    contract_version: str = CONTRACT_VERSION
    extraction_version: str = "ev-test"
    model_hash: str = "svc-model"
    location_label: str = "test-box"
    down: bool = False
    # utterance -> errors to answer with, consumed front to back before success.
    fail_script: dict[str, list[tuple[ErrorCode, bool]]] = field(default_factory=dict)
    # utterance -> derivation operations to return when derivation was requested.
    derivation_ops: dict[str, list[dict]] = field(default_factory=dict)
    payload_fn: Callable[[ExtractRequest], tuple[list[dict], list[dict]]] = default_payload
    # When set, /v1/extract awaits it before answering (to hold a job in flight).
    hold: asyncio.Event | None = None
    received: list[ExtractRequest] = field(default_factory=list)
    info_calls: int = 0

    @property
    def received_utterances(self) -> list[str]:
        return [r.utterance for r in self.received]


def build_fake_service_app(state: FakeServiceState) -> FastAPI:
    """The extraction service's three endpoints, answering from `state`."""
    app = FastAPI()

    @app.get("/v1/info")
    async def info() -> InfoResponse:
        state.info_calls += 1
        return InfoResponse(
            contract_version=state.contract_version,
            extraction_version=state.extraction_version,
            model_hash=state.model_hash,
            model_file="fake.gguf",
            llama_cpp_build="b0",
            adapter="fake",
            location_label=state.location_label,
            # Contract 1.1.0 serving config, as a current service reports it.
            constrained_mode="schema",
            reasoning_effort="low",
            temperature=0.0,
            ctx_size=8192,
        )

    @app.get("/v1/health")
    async def health() -> HealthResponse:
        return HealthResponse(status="ok", llm_reachable=True, uptime_s=1.0)

    @app.post("/v1/extract")
    async def extract(req: ExtractRequest):
        state.received.append(req)
        if state.hold is not None:
            await state.hold.wait()
        script = state.fail_script.get(req.utterance)
        if script:
            code, retryable = script.pop(0)
            envelope = error_envelope(code, f"scripted {code.value}", retryable)
            return JSONResponse(
                status_code=ERROR_HTTP_STATUS[code], content=envelope.model_dump(mode="json")
            )
        entities, relationships = state.payload_fn(req)
        derivation = None
        if req.derivation is not None:
            derivation = DerivationOut(operations=state.derivation_ops.get(req.utterance, []))
        return ExtractResponse(
            contract_version=CONTRACT_VERSION,
            job_id=req.job_id,
            outcome="extracted",
            scope=ScopeOut(label="user-scope", confidence=0.9),
            payload=ExtractionPayload(entities=entities, relationships=relationships),
            derivation=derivation,
            stamps=ResultStamps(
                extraction_version=state.extraction_version,
                model_hash=state.model_hash,
                prompt_sha256="0" * 64,
                llama_cpp_build="b0",
                adapter="fake",
            ),
            timings_ms=TimingsMs(scope=1.0, extract=2.0, derive=None, total=3.0),
            attempts=1,
        )

    return app


class SwitchableTransport(httpx.AsyncBaseTransport):
    """ASGI transport that raises `httpx.ConnectError` while `state.down` is set."""

    def __init__(self, state: FakeServiceState, app: FastAPI) -> None:
        self._state = state
        self._inner = httpx.ASGITransport(app=app)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self._state.down:
            raise httpx.ConnectError("connection refused (fake service down)", request=request)
        return await self._inner.handle_async_request(request)


def _empty_curation_result(validation_result: Any) -> CurationResult:
    return CurationResult(
        write_result=WriteResult(),
        dedup_result=DeduplicationResult(entities=[], merge_actions=[], entities_merged=0),
        reconcile_result=ReconcileTurnResult(),
        curation_time_ms=0.0,
        validated_entities=list(validation_result.entities),
        validated_relationships=list(validation_result.relationships),
    )


@dataclass
class CurationCall:
    """One `curate_and_store` call, as the fake saw it."""

    event_id: str
    session_id: str
    recorded_at: str | None
    source_metadata: Any
    entities: list[dict]
    relationships: list[dict]


@dataclass
class FakeGraphCuration:
    """Stateful in-memory stand-in for `CurationPipeline`.

    `crash_after_write_for` names an event_id whose NEXT call writes to the
    fake graph and then raises `SimulatedCrashError` -- a fault after the graph write
    and before any marker.
    """

    nodes: dict[str, dict] = field(default_factory=dict)
    edges: dict[tuple[str, str, str], dict] = field(default_factory=dict)
    calls: list[CurationCall] = field(default_factory=list)
    crash_after_write_for: str | None = None

    @property
    def event_ids(self) -> list[str]:
        return [c.event_id for c in self.calls]

    async def curate_and_store(
        self,
        validation_result: Any,
        event_id: str,
        session_id: str,
        source_metadata: Any = None,
        recorded_at: str | None = None,
    ) -> CurationResult:
        self.calls.append(
            CurationCall(
                event_id=event_id,
                session_id=session_id,
                recorded_at=recorded_at,
                source_metadata=source_metadata,
                entities=[dict(e) for e in validation_result.entities],
                relationships=[dict(r) for r in validation_result.relationships],
            )
        )
        for entity in validation_result.entities:
            node = self.nodes.setdefault(entity["id"], {})
            node.update({"type": entity.get("type"), "name": entity.get("name")})
        for rel in validation_result.relationships:
            key = (rel["source"], rel["type"], rel["target"])
            self.edges.setdefault(key, {}).update({"event_id": event_id})
        if self.crash_after_write_for == event_id:
            self.crash_after_write_for = None
            raise SimulatedCrashError(f"crash after graph write for {event_id}")
        return _empty_curation_result(validation_result)

    def graph_state(self) -> tuple[dict, dict]:
        return (dict(self.nodes), dict(self.edges))


@dataclass
class FakeInternalDeriver:
    """The two `InternalKnowledgeDeriver` methods the backlog calls.

    `self_model` MERGEs operations by id. `crash_after_apply_for` names an
    event_id whose NEXT apply writes and then raises `SimulatedCrashError`.
    """

    existing: str = "Existing internal entities:\n- [MistTrait] Concise (id: trait-concise)"
    self_model: dict[str, dict] = field(default_factory=dict)
    apply_calls: list[tuple[str, list[dict]]] = field(default_factory=list)
    crash_after_apply_for: str | None = None

    async def fetch_existing_internal_entities(self) -> str:
        return self.existing

    async def apply_operations(
        self, operations: list[dict], *, session_id: str, event_id: str
    ) -> tuple[dict, ...]:
        self.apply_calls.append((event_id, list(operations)))
        for op in operations:
            self.self_model[op["id"]] = dict(op)
        if self.crash_after_apply_for == event_id:
            self.crash_after_apply_for = None
            raise SimulatedCrashError(f"crash after derivation apply for {event_id}")
        return tuple(operations)
