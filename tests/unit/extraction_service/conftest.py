"""Shared fixtures for the extraction service test suite.

Wires the real `LlamaServerProvider` (and a real `LlamaHealthProbe`) to a
fake llama-server that is itself a small ASGI app, per `tests/CLAUDE.md`'s
mock-only-at-I/O-boundaries rule -- no network, no GPU, and the service's
own HTTP layer (FastAPI/Starlette) runs unmodified end to end.
"""

from __future__ import annotations

import httpx
import pytest
import pytest_asyncio
from openai import AsyncOpenAI

from backend.extraction_service.adapters import get_adapter
from backend.extraction_service.app import LlamaHealthProbe, create_app
from backend.extraction_service.engine import ExtractionEngine
from backend.extraction_service.settings import ServiceSettings
from backend.knowledge.version_stamps import EXTRACTION_VERSION
from backend.llm.llama_server_provider import LlamaServerProvider
from tests.unit.extraction_service.fake_llama import FakeLlamaState, build_fake_llama_app

FAKE_BASE_URL = "http://fake-llama"


@pytest.fixture
def fake_llama_state() -> FakeLlamaState:
    return FakeLlamaState()


@pytest.fixture
def fake_llama_app(fake_llama_state: FakeLlamaState):
    return build_fake_llama_app(fake_llama_state)


@pytest.fixture
def wired_llm(fake_llama_app) -> LlamaServerProvider:
    """A real LlamaServerProvider whose OpenAI client talks to the fake app."""
    provider = LlamaServerProvider(base_url=FAKE_BASE_URL, model="fake-model")
    provider._async_client = AsyncOpenAI(
        base_url=f"{FAKE_BASE_URL}/v1",
        api_key="x",
        # max_retries=0: an error test that scripts a single 5xx response
        # must see exactly one call, not the openai client's own
        # retry-with-backoff burning through the engine's LLM timeout
        # before the real error ever surfaces.
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.ASGITransport(app=fake_llama_app)),
    )
    return provider


@pytest.fixture
def health_probe(fake_llama_app) -> LlamaHealthProbe:
    http_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=fake_llama_app))
    return LlamaHealthProbe(http_client=http_client, base_url=FAKE_BASE_URL)


@pytest.fixture
def service_settings() -> ServiceSettings:
    return ServiceSettings(
        llm_base_url=FAKE_BASE_URL,
        model_hash="test-model-hash",
        model_file="test-model.gguf",
        adapter_name="gptoss",
        reasoning_effort="low",
        max_attempts=2,
        llm_timeout_seconds=0.3,
        idempotency_cache_size=8,
    )


@pytest.fixture
def adapter(service_settings: ServiceSettings):
    return get_adapter(
        service_settings.adapter_name,
        reasoning_effort=service_settings.reasoning_effort,
        reasoning_budget_tokens=service_settings.reasoning_budget_tokens,
    )


@pytest.fixture
def engine(wired_llm, adapter, service_settings: ServiceSettings) -> ExtractionEngine:
    return ExtractionEngine(llm=wired_llm, adapter=adapter, settings=service_settings)


@pytest.fixture
def service_app(service_settings, engine, health_probe):
    return create_app(service_settings, engine, health_probe)


@pytest_asyncio.fixture
async def client(service_app):
    transport = httpx.ASGITransport(app=service_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://service") as c:
        yield c


def make_extract_request(
    *,
    job_id: str = "job-1",
    event_id: str = "evt-1",
    turn_id: str = "turn-1",
    request_id: str = "req-1",
    session_id: str = "sess-1",
    utterance: str = "I have been learning Rust for a few months",
    recorded_at: str = "2026-04-21T12:00:00+00:00",
    extraction_version: str = EXTRACTION_VERSION,
    model_hash: str = "test-model-hash",
    contract_version: str = "1.0.0",
    derivation: dict | None = None,
) -> dict:
    """Build a valid ExtractRequest body as a plain dict, for httpx POSTs."""
    return {
        "contract_version": contract_version,
        "job_id": job_id,
        "event_id": event_id,
        "turn_id": turn_id,
        "request_id": request_id,
        "session_id": session_id,
        "recorded_at": recorded_at,
        "turn_index": 0,
        "utterance": utterance,
        "conversation_history": [],
        "expect": {"extraction_version": extraction_version, "model_hash": model_hash},
        "derivation": derivation,
    }
