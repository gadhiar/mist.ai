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
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from backend.extraction_backlog.store import PackHashes
from backend.extraction_contract.models import (
    CONTRACT_VERSION,
    ERROR_HTTP_STATUS,
    BatchRecordEntry,
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
    TurnPlanOut,
    error_envelope,
)
from backend.knowledge.curation.deduplication import DeduplicationResult
from backend.knowledge.curation.graph_writer import WriteResult
from backend.knowledge.curation.pipeline import CurationResult
from backend.knowledge.curation.reconciliation import ReconcileTurnResult
from backend.knowledge.ontologies.pack import PackLoadError

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


def default_serving_components() -> dict[str, Any]:
    """A contract 1.3.0 service's flat serving components, in the shape T1 reports them.

    A subset of the real names (`serving_identity.components_dict`): enough for
    a test to change one setting and see the identity move and the refusal
    name it.
    """
    return {
        "adapter": "fake",
        "model_file": "fake.gguf",
        "llama_cpp_build": "b0",
        "max_attempts": 2,
        "scope_enabled": True,
        "n_ctx": 8192,
        "request.extract.temperature": 0.0,
        "request.extract.top_p": 0.9,
        "request.extract.thinking_effort": "low",
        "server.ctx_size": "8192",
        "server.device": "test-gpu",
        "server.n_cpu_moe": "24",
    }


def serving_identity_of(components: dict[str, Any]) -> str:
    """The `si1-` identity of `components`, computed the way the service does."""
    canonical = json.dumps(components, sort_keys=True, separators=(",", ":"))
    return "si1-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class FakeServiceState:
    """What the fake service reports and does. Mutate it mid-test to change behaviour."""

    contract_version: str = CONTRACT_VERSION
    extraction_version: str = "ev-test"
    model_hash: str = "svc-model"
    location_label: str = "test-box"
    # Contract 1.2.0: reported on /v1/info and stamped on every result. None
    # answers the way a 1.1.0 service does.
    run_identity: str | None = None
    # Contract 1.3.0: the serving identity is computed from these on every
    # call, so mutating one models a restarted service with that setting
    # changed. `serving_identity_error` set (or `reports_serving_identity`
    # False, a pre-1.3.0 service) reports no identity.
    serving_components: dict[str, Any] = field(default_factory=default_serving_components)
    serving_identity_error: str | None = None
    reports_serving_identity: bool = True
    # When set, results are stamped with this serving identity instead of the
    # computed one: a service that changed between `/v1/info` and the reply.
    result_serving_identity: str | None = None
    # The ontology pack's two hashes, reported on /v1/info and stamped on every
    # result. None answers the way a service from before the hashes does.
    pack_prompt_hash: str | None = None
    pack_rails_hash: str | None = None
    # What every result carries as its Stage 2 batch record and its plan call.
    # None leaves the key out of the reply, as a service that records neither.
    batch_record: list[BatchRecordEntry] | None = None
    turn_plan: TurnPlanOut | None = None
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

    def serve_pack(self, carrier: Any) -> None:
        """Report `carrier`'s two pack hashes from now on.

        `carrier` is anything with `pack_prompt_hash` and `pack_rails_hash`: a
        cutover candidate, an epoch, a pack. It models the operator pointing
        the service at that pack.
        """
        self.pack_prompt_hash = carrier.pack_prompt_hash
        self.pack_rails_hash = carrier.pack_rails_hash

    @property
    def serving_identity(self) -> str | None:
        """The identity `/v1/info` reports now, or None (an error, or a pre-1.3.0 service)."""
        if not self.reports_serving_identity or self.serving_identity_error is not None:
            return None
        return serving_identity_of(self.serving_components)


def fake_pack_resolver(
    packs: dict[str, tuple[str | None, str | None]],
) -> Callable[[str], PackHashes]:
    """A `pack_for_version` for `admin.main`, over invented packs.

    `packs` maps an ontology version to that pack's `(pack_prompt_hash,
    pack_rails_hash)`. A version it does not hold raises `PackLoadError`, as
    `registry.get_pack` does.
    """

    def pack_for_version(version: str) -> PackHashes:
        if version not in packs:
            known = ", ".join(sorted(packs))
            raise PackLoadError(f"no ontology pack for version {version} (known: {known})")
        prompt, rails = packs[version]
        return PackHashes(pack_prompt_hash=prompt, pack_rails_hash=rails)

    return pack_for_version


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
            run_identity=state.run_identity,
            serving_identity=state.serving_identity,
            serving_identity_components=(
                dict(state.serving_components) if state.reports_serving_identity else None
            ),
            serving_identity_error=(
                state.serving_identity_error if state.reports_serving_identity else None
            ),
            pack_prompt_hash=state.pack_prompt_hash,
            pack_rails_hash=state.pack_rails_hash,
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
        expected = req.expect.serving_identity
        if expected is not None and expected != state.serving_identity:
            code = ErrorCode.EPOCH_MISMATCH
            envelope = error_envelope(
                code, f"serving identity {state.serving_identity} != expected {expected}", False
            )
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
                run_identity=state.run_identity,
                serving_identity=state.result_serving_identity or state.serving_identity,
                pack_prompt_hash=state.pack_prompt_hash,
                pack_rails_hash=state.pack_rails_hash,
            ),
            timings_ms=TimingsMs(scope=1.0, extract=2.0, derive=None, total=3.0),
            attempts=1,
            turn_plan=state.turn_plan,
            batch_record=state.batch_record,
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
