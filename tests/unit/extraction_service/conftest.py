"""Shared fixtures for the extraction service test suite.

Wires the real `LlamaServerProvider` (and a real `LlamaHealthProbe`) to a
fake llama-server that is itself a small ASGI app, per `TESTING.md`'s
mock-only-at-I/O-boundaries rule -- no network, no GPU, and the service's
own HTTP layer (FastAPI/Starlette) runs unmodified end to end.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
import pytest_asyncio
from openai import AsyncOpenAI

from backend.extraction_service.adapters import get_adapter
from backend.extraction_service.app import LlamaHealthProbe, LlamaPropsContextSize, create_app
from backend.extraction_service.engine import ExtractionEngine
from backend.extraction_service.settings import ServiceSettings
from backend.knowledge.version_stamps import EXTRACTION_VERSION
from backend.llm.llama_server_provider import LlamaServerProvider
from tests.unit.extraction_service.fake_llama import (
    FAKE_N_CTX,
    FakeLlamaState,
    StreamingASGITransport,
    build_fake_llama_app,
)

FAKE_BASE_URL = "http://fake-llama"

# The `EXTRACTION_SERVER_*` declarations a compose file gives the service, as
# `ServiceSettings.from_env` reads them: the required three, with the declared
# context size equal to the one the fake llama-server reports, so the service
# has a serving identity unless a test takes one away.
FAKE_SERVER_CONFIG: tuple[tuple[str, str], ...] = (
    ("ctx_size", str(FAKE_N_CTX)),
    ("device", "test-gpu"),
    ("image", "test-llama-server@sha256:" + "0" * 64),
)


@pytest.fixture
def fake_llama_state() -> FakeLlamaState:
    return FakeLlamaState()


@pytest.fixture
def fake_llama_app(fake_llama_state: FakeLlamaState):
    return build_fake_llama_app(fake_llama_state)


@pytest.fixture
def wired_llm(fake_llama_app) -> LlamaServerProvider:
    """A real LlamaServerProvider whose OpenAI client talks to the fake app.

    The client reaches the app through `StreamingASGITransport`, so a
    streamed reply arrives chunk by chunk and a call the service leaves is
    one the fake sees left (`FakeLlamaState.chat_aborted`).
    """
    provider = LlamaServerProvider(base_url=FAKE_BASE_URL, model="fake-model")
    provider._async_client = AsyncOpenAI(
        base_url=f"{FAKE_BASE_URL}/v1",
        api_key="x",
        # max_retries=0: an error test that scripts a single 5xx response
        # must see exactly one call, not the openai client's own
        # retry-with-backoff burning through the engine's LLM timeout
        # before the real error ever surfaces.
        max_retries=0,
        http_client=httpx.AsyncClient(transport=StreamingASGITransport(fake_llama_app)),
    )
    return provider


@pytest.fixture
def health_probe(fake_llama_app) -> LlamaHealthProbe:
    http_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=fake_llama_app))
    return LlamaHealthProbe(http_client=http_client, base_url=FAKE_BASE_URL)


@pytest.fixture
def ctx_size_source(fake_llama_app) -> LlamaPropsContextSize:
    http_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=fake_llama_app))
    return LlamaPropsContextSize(http_client=http_client, base_url=FAKE_BASE_URL)


@pytest_asyncio.fixture
async def blocked_chat(fake_llama_state: FakeLlamaState):
    """Hold every fake chat call on an event until the test (or teardown) sets it.

    A test that sets the event decides exactly when in-flight calls answer;
    a test that never sets it gets calls that never answer. Teardown sets it
    either way, inside the event loop, so no handler is left blocked.
    """
    gate = asyncio.Event()
    fake_llama_state.chat_gate = gate
    yield gate
    gate.set()
    for _ in range(5):
        await asyncio.sleep(0)


@pytest_asyncio.fixture
async def held_stream(fake_llama_state: FakeLlamaState):
    """Hold every streamed fake reply after its first content delta.

    The call is in flight with its stream open until the test, or teardown,
    sets the event. A call that is not streamed is not held.
    """
    gate = asyncio.Event()
    fake_llama_state.stream_gate = gate
    yield gate
    gate.set()
    for _ in range(5):
        await asyncio.sleep(0)


async def until(condition, *, ticks: int = 2000) -> None:
    """Yield to the event loop until `condition()` holds; fail when it never does.

    The fakes here run on the event loop with no socket and no thread, so a
    state another task is about to reach is reached in a bounded number of
    turns of the loop, and no wall clock is read.
    """
    for _ in range(ticks):
        if condition():
            return
        await asyncio.sleep(0)
    raise AssertionError(f"the condition did not hold within {ticks} turns of the event loop")


@pytest.fixture
def service_settings() -> ServiceSettings:
    # llm_timeout_seconds is a generous budget, not a test knob: every fake
    # call here answers in well under a millisecond, so no test may lean on
    # this margin. A test about the timeout itself builds its own settings
    # and a fake call that never answers (see `blocked_chat`).
    return ServiceSettings(
        llm_base_url=FAKE_BASE_URL,
        model_hash="test-model-hash",
        model_file="test-model.gguf",
        adapter_name="gptoss",
        reasoning_effort="low",
        max_attempts=2,
        llm_timeout_seconds=10.0,
        idempotency_cache_size=8,
        server_config=FAKE_SERVER_CONFIG,
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
def service_app(service_settings, engine, health_probe, ctx_size_source):
    return create_app(service_settings, engine, health_probe, ctx_size_source)


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
    serving_identity: str | None = None,
) -> dict:
    """Build a valid ExtractRequest body as a plain dict, for httpx POSTs.

    `expect.serving_identity` is sent only when `serving_identity` is given, so
    the default body is the one a pre-1.3.0 backend sends.
    """
    expect: dict[str, str] = {"extraction_version": extraction_version, "model_hash": model_hash}
    if serving_identity is not None:
        expect["serving_identity"] = serving_identity
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
        "expect": expect,
        "derivation": derivation,
    }
