"""The service extracts with an ontology pack.

The engine takes a pack, the registry's active one by default: the pack's item filters
choose the units the model is given, its assembled prompt and generated schema are what
Stage 2 sends, and its two hashes are stamped on every result and reported on `/v1/info`.
The active pack's fragments assemble to the prompt as it stood, and its one item filter,
`unchecked-goals`, drops an unchecked item under no goal word; `test_turn_plan_off_pins.py`
pins the bytes the service sends. The cases that need a fragment of their own or a
filter with other words run on the base fixture pack.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
from pathlib import Path

import httpx
import pytest
import pytest_asyncio

import backend.extraction_service as service_package
from backend.extraction_service import engine as engine_module
from backend.extraction_service.app import create_app, run_identity_components
from backend.extraction_service.engine import ExtractionEngine, _compute_prompt_sha256
from backend.extraction_service.schemas import EXTRACTION_OUTPUT_SCHEMA
from backend.knowledge.extraction.run_identity import RunIdentityComponents
from backend.knowledge.ontologies.pack import load_pack
from backend.knowledge.ontologies.registry import active_pack
from tests.unit.extraction_service.conftest import make_extract_request

BASE = load_pack("tests.unit.knowledge.ontologies.fixtures.base_pack")

# The heading over the numbered rules: text of the engine core, so of every pack's
# Stage 2 system prompt, and of no other stage's.
_STAGE2_MARK = "## EXTRACTION RULES"
# How the engine core's last rule ends: what the base pack's rules fragment follows.
_CORE_RULES_END = "even if first-person pronouns are present.\n"
_FRAGMENT = "F1. Each deckhand placed in a crew also SERVES_UNDER its skipper.\n"
_DECK_JOBS = "Deck jobs for Saturday:\n- [ ] coil the stern lines\n- [x] check the bilge pump\n"
_WATCH_LIST = "Watch list:\n- [ ] the frayed halyard\n- [x] the port winch\n"


def _text(request: dict) -> str:
    return "\n".join(str(message.get("content")) for message in request["messages"])


def _stage2_texts(state) -> list[str]:
    """The text of every request sent under the Stage 2 system prompt, in call order."""
    return [text for text in map(_text, state.chat_requests) if _STAGE2_MARK in text]


@pytest.fixture
def base_engine(wired_llm, adapter, service_settings) -> ExtractionEngine:
    return ExtractionEngine(llm=wired_llm, adapter=adapter, settings=service_settings, pack=BASE)


@pytest_asyncio.fixture
async def base_client(service_settings, base_engine, health_probe, ctx_size_source):
    app = create_app(service_settings, base_engine, health_probe, ctx_size_source)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://service") as c:
        yield c


# --- the pack in use by default ---------------------------------------------------------


def test_the_engine_takes_the_registrys_active_pack_by_default(engine):
    assert engine.pack == active_pack()


def test_the_active_packs_prompt_and_schema_are_the_ones_the_service_sent_before(adapter):
    pack = active_pack()

    # The Stage 2 system prompt as the service held it at 058e52c, and the prompt hash
    # `test_turn_plan_off_pins.py` pins, which moved when T48 added the earlier items'
    # heading to it.
    assert hashlib.sha256(pack.system_prompt.encode("utf-8")).hexdigest() == (
        "a486c5bfd3cd09175a5361d7de448a1254fc8f69475c3c967889849ea39584e4"
    )
    assert json.dumps(pack.extraction_schema) == json.dumps(EXTRACTION_OUTPUT_SCHEMA)
    assert _compute_prompt_sha256(adapter, pack=pack) == (
        "dfae2785c08ededffbbf4369ed468b9e803abde321566481d8f7bc4aae80298b"
    )


def _other_pack():
    """The active pack with a prompt and a schema of its own."""
    return dataclasses.replace(
        active_pack(),
        system_prompt="A harbour log, read on {reference_date}.",
        extraction_schema={"type": "object", "title": "harbour log"},
    )


def test_with_no_pack_the_prompt_hash_is_the_active_packs(adapter, monkeypatch):
    todays = _compute_prompt_sha256(adapter)
    other = _other_pack()
    monkeypatch.setattr(engine_module, "active_pack", lambda: other)

    assert _compute_prompt_sha256(adapter) == _compute_prompt_sha256(adapter, pack=other)
    assert _compute_prompt_sha256(adapter) != todays


def test_with_no_pack_the_run_identity_is_the_active_packs(service_settings, adapter, monkeypatch):
    pack = active_pack()
    todays = run_identity_components(service_settings, adapter)
    assert todays == run_identity_components(service_settings, adapter, pack=pack)

    other = _other_pack()
    monkeypatch.setattr(engine_module, "active_pack", lambda: other)

    moved = run_identity_components(service_settings, adapter)
    assert moved == run_identity_components(service_settings, adapter, pack=other)
    assert moved.prompt_sha256 != todays.prompt_sha256


@pytest.mark.asyncio
async def test_the_active_pack_drops_an_unchecked_item_under_no_goal_word(client, fake_llama_state):
    response = await client.post("/v1/extract", json=make_extract_request(utterance=_DECK_JOBS))

    assert response.status_code == 200
    (sent,) = _stage2_texts(fake_llama_state)
    assert "coil the stern lines" not in sent
    assert "check the bilge pump" in sent
    assert _FRAGMENT not in sent


# --- the two hashes ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_every_result_is_stamped_with_both_hashes_of_the_active_pack(client):
    response = await client.post("/v1/extract", json=make_extract_request())

    stamps = response.json()["stamps"]
    assert stamps["pack_prompt_hash"] == active_pack().pack_prompt_hash
    assert stamps["pack_rails_hash"] == active_pack().pack_rails_hash


@pytest.mark.asyncio
async def test_info_reports_both_hashes_of_the_active_pack(client):
    body = (await client.get("/v1/info")).json()

    assert body["pack_prompt_hash"] == active_pack().pack_prompt_hash
    assert body["pack_rails_hash"] == active_pack().pack_rails_hash


@pytest.mark.asyncio
async def test_an_engine_given_a_pack_stamps_and_reports_that_packs_hashes(base_client):
    stamps = (await base_client.post("/v1/extract", json=make_extract_request())).json()["stamps"]
    info = (await base_client.get("/v1/info")).json()

    assert BASE.pack_prompt_hash != active_pack().pack_prompt_hash
    for reported in (stamps, info):
        assert reported["pack_prompt_hash"] == BASE.pack_prompt_hash
        assert reported["pack_rails_hash"] == BASE.pack_rails_hash


# --- the hashes are stamps of their own -------------------------------------------------


def test_a_packs_hashes_are_no_component_of_the_run_identity(service_settings, adapter):
    names = [field.name for field in dataclasses.fields(RunIdentityComponents)]

    assert not [name for name in names if name.startswith("pack")]
    assert run_identity_components(
        service_settings, adapter, pack=active_pack()
    ) == run_identity_components(service_settings, adapter)


def test_a_packs_hashes_do_not_enter_the_prompt_hash(adapter):
    relabelled = dataclasses.replace(
        active_pack(), pack_prompt_hash="0" * 64, pack_rails_hash="1" * 64
    )

    assert _compute_prompt_sha256(adapter, pack=relabelled) == _compute_prompt_sha256(adapter)


@pytest.mark.asyncio
async def test_a_packs_hashes_are_no_component_of_the_serving_identity(client):
    components = (await client.get("/v1/info")).json()["serving_identity_components"]

    assert components
    assert not [name for name in components if "pack" in name]


def test_a_packs_prompt_and_its_schema_each_move_the_prompt_hash(adapter):
    tighter = dataclasses.replace(
        active_pack(), extraction_schema={**EXTRACTION_OUTPUT_SCHEMA, "additionalProperties": False}
    )

    assert _compute_prompt_sha256(adapter, pack=BASE) != _compute_prompt_sha256(adapter)
    assert _compute_prompt_sha256(adapter, pack=tighter) != _compute_prompt_sha256(adapter)


@pytest.mark.asyncio
async def test_the_prompt_hash_stamped_is_the_one_of_the_pack_in_use(base_client, adapter):
    stamps = (await base_client.post("/v1/extract", json=make_extract_request())).json()["stamps"]

    assert stamps["prompt_sha256"] == _compute_prompt_sha256(adapter, pack=BASE)


# --- a pack's fragment and filter reach Stage 2 -----------------------------------------


@pytest.mark.asyncio
async def test_stage_2_is_sent_under_the_packs_assembled_prompt(base_client, fake_llama_state):
    response = await base_client.post("/v1/extract", json=make_extract_request())

    assert response.status_code == 200
    (sent,) = _stage2_texts(fake_llama_state)
    assert f"{_CORE_RULES_END}{_FRAGMENT}\n## REFERENCE DATE\n" in sent


@pytest.mark.asyncio
async def test_an_item_the_packs_filter_names_is_not_sent(base_client, fake_llama_state):
    response = await base_client.post(
        "/v1/extract", json=make_extract_request(utterance=_DECK_JOBS)
    )

    assert response.status_code == 200
    (sent,) = _stage2_texts(fake_llama_state)
    assert "coil the stern lines" not in sent
    assert "check the bilge pump" in sent


@pytest.mark.asyncio
async def test_an_item_under_one_of_the_filters_keep_words_is_sent(base_client, fake_llama_state):
    response = await base_client.post(
        "/v1/extract", json=make_extract_request(utterance=_WATCH_LIST)
    )

    assert response.status_code == 200
    (sent,) = _stage2_texts(fake_llama_state)
    assert "the frayed halyard" in sent
    assert "the port winch" in sent


@pytest.mark.asyncio
async def test_the_plan_call_is_sent_under_the_packs_prompt_and_given_the_kept_units(
    wired_llm, adapter, service_settings, health_probe, ctx_size_source, fake_llama_state
):
    settings = dataclasses.replace(service_settings, turn_plan=True)
    engine = ExtractionEngine(llm=wired_llm, adapter=adapter, settings=settings, pack=BASE)
    app = create_app(settings, engine, health_probe, ctx_size_source)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://service") as c:
        response = await c.post("/v1/extract", json=make_extract_request(utterance=_DECK_JOBS))

    assert response.status_code == 200
    plan_call, batch_call = _stage2_texts(fake_llama_state)
    for sent in (plan_call, batch_call):
        assert f"{_CORE_RULES_END}{_FRAGMENT}\n## REFERENCE DATE\n" in sent
        assert "coil the stern lines" not in sent
        assert "check the bilge pump" in sent


# --- what the service may import --------------------------------------------------------


def _imported_modules(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def test_no_service_module_imports_the_post_llm_rails():
    rails = ("backend.knowledge.extraction.rails", "backend.knowledge.extraction.id_rails")
    service_files = sorted(Path(service_package.__file__).parent.rglob("*.py"))

    importers = {
        path.name: sorted(name for name in _imported_modules(path) if name.startswith(rails))
        for path in service_files
    }

    assert service_files
    assert {name: found for name, found in importers.items() if found} == {}
