"""Stage 2's list pass: one call per batch of units, each batch seeing the units before it.

The engine splits the utterance with `split_structured`, extracts at most
`LIST_BATCH_MAX_UNITS` units per Stage 2 call and gives each later batch the
nearest `LIST_CONTEXT_MAX_UNITS` earlier units as context. A turn without
structure must send exactly the one Stage 2 request it sent before the list
pass existed. The fake LLM here answers each Stage 2 call with one entity per
line of the batch it was sent, so a test can read off what each call saw.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

import pytest

from backend.extraction_contract.models import ExtractRequest
from backend.extraction_service.adapters import get_adapter
from backend.extraction_service.engine import (
    DERIVE_MAX_TOKENS,
    EXTRACT_MAX_TOKENS,
    LIST_BATCH_MAX_UNITS,
    LIST_CONTEXT_MAX_UNITS,
    SCOPE_MAX_TOKENS,
    ExtractionEngine,
    UpstreamLLMError,
    parse_recorded_at,
)
from backend.extraction_service.settings import ServiceSettings
from backend.knowledge.extraction.ontology_extractor import (
    NO_PRIOR_CONTEXT,
    render_extraction_messages,
)
from backend.knowledge.extraction.preprocessor import PreProcessor
from backend.knowledge.extraction.structure_splitter import (
    batch_units,
    render_batch,
    render_unit,
    split_structured,
)
from backend.llm.models import LLMRequest, LLMResponse
from backend.llm.provider import StreamingLLMProvider
from tests.unit.extraction_service.conftest import make_extract_request

_GOLD = Path(__file__).resolve().parents[3] / "data" / "ingest" / "extraction-gold-2026-06-14.jsonl"

_SCOPE_JSON = '{"scope": "user-scope", "confidence": 0.9, "reasoning": "first person"}'
_USER_MESSAGE_RE = re.compile(
    r'^Context:\n(?P<context>.*)\nSubject scope: [^\n]*\nUtterance: "(?P<utterance>.*)"\n\nOutput:$',
    re.DOTALL,
)


def _gold_utterances() -> list[tuple[str, str]]:
    rows = [json.loads(line) for line in _GOLD.read_text(encoding="utf-8").splitlines() if line]
    return [(row["tag"], row["utterance"]) for row in rows]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _user_message(request: LLMRequest) -> re.Match[str]:
    match = _USER_MESSAGE_RE.match(request.messages[1]["content"])
    assert match is not None, request.messages[1]["content"]
    return match


def batch_text(request: LLMRequest) -> str:
    """The utterance a Stage 2 request carries: the rendered batch."""
    return _user_message(request).group("utterance")


def context_lines(request: LLMRequest) -> list[str]:
    context = _user_message(request).group("context")
    return [] if context == NO_PRIOR_CONTEXT else context.split("\n")


def payload_for(text: str) -> dict:
    """One `Concept` per line of the batch, each linked to the user."""
    lines = text.split("\n")
    entities = [{"id": "user", "name": "User", "type": "User"}]
    entities += [{"id": _slug(line), "name": line, "type": "Concept"} for line in lines]
    relationships = [
        {"source": "user", "target": _slug(line), "type": "INTERESTED_IN"} for line in lines
    ]
    return {"entities": entities, "relationships": relationships}


StageAnswer = Callable[[LLMRequest, int], LLMResponse]


def echo(request: LLMRequest, call: int) -> LLMResponse:
    return LLMResponse(content=json.dumps(payload_for(batch_text(request))), finish_reason="stop")


class ScriptedLLM(StreamingLLMProvider):
    """Answers scope and derivation plainly and Stage 2 through `extract`.

    `extract` gets the request and the 1-based index of this Stage 2 call.
    """

    model = "scripted"

    def __init__(self, extract: StageAnswer = echo) -> None:
        self.requests: list[LLMRequest] = []
        self._extract = extract

    async def generate(self, request: LLMRequest, *, stream: bool = False):
        self.requests.append(request)
        yield self._answer(request)

    def generate_sync(self, request: LLMRequest, *, stream: bool = False):
        raise NotImplementedError

    def _answer(self, request: LLMRequest) -> LLMResponse:
        if request.max_tokens == SCOPE_MAX_TOKENS:
            return LLMResponse(content=_SCOPE_JSON, finish_reason="stop")
        if request.max_tokens == DERIVE_MAX_TOKENS:
            return LLMResponse(content='{"operations": []}', finish_reason="stop")
        assert request.max_tokens == EXTRACT_MAX_TOKENS
        return self._extract(request, len(self.stage2_requests))

    def requests_with(self, max_tokens: int) -> list[LLMRequest]:
        return [r for r in self.requests if r.max_tokens == max_tokens]

    @property
    def stage2_requests(self) -> list[LLMRequest]:
        return self.requests_with(EXTRACT_MAX_TOKENS)


def _settings() -> ServiceSettings:
    return ServiceSettings(
        llm_base_url="http://unused",
        model_hash="test-model-hash",
        model_file="test-model.gguf",
        max_attempts=2,
        llm_timeout_seconds=10.0,
    )


def make_engine(llm: StreamingLLMProvider) -> ExtractionEngine:
    return ExtractionEngine(llm=llm, adapter=get_adapter("gptoss"), settings=_settings())


def make_request(utterance: str, history: list[dict] | None = None, **kwargs) -> ExtractRequest:
    body = make_extract_request(utterance=utterance, **kwargs)
    body["conversation_history"] = history or []
    return ExtractRequest.model_validate(body)


def numbered_list(count: int, lead_in: str = "Things I want to try this year:") -> str:
    return lead_in + "\n" + "\n".join(f"{n}. thing {n}" for n in range(1, count + 1))


# ---------------------------------------------------------------------------
# The limits
# ---------------------------------------------------------------------------


def test_the_list_pass_limits():
    assert (LIST_BATCH_MAX_UNITS, LIST_CONTEXT_MAX_UNITS) == (4, 8)


# ---------------------------------------------------------------------------
# A turn without structure is sent exactly as before
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("tag,utterance", _gold_utterances())
async def test_a_gold_utterance_sends_one_unchanged_stage2_request(tag, utterance):
    llm = ScriptedLLM()
    req = make_request(utterance)

    response = await make_engine(llm).run(req)

    pre_processed = PreProcessor().pre_process(
        utterance=utterance,
        conversation_history=[],
        reference_date=parse_recorded_at(req.recorded_at),
        turn_index=req.turn_index,
    )
    pre_processed.metadata["subject_scope"] = "user-scope"
    (stage2,) = llm.stage2_requests
    assert stage2.messages == render_extraction_messages(pre_processed)
    assert response.stage2_calls == 1
    assert response.attempts == 1
    assert response.payload.model_dump() == payload_for(utterance)
    assert response.warnings == []


def test_the_60_gold_utterances_are_all_there():
    assert len(_gold_utterances()) == 60


@pytest.mark.asyncio
async def test_a_turn_without_structure_keeps_its_history_as_the_only_context():
    history = [
        {"role": "user", "content": "I moved teams"},
        {"role": "assistant", "content": "Which one?"},
    ]
    llm = ScriptedLLM()

    await make_engine(llm).run(make_request("The platform team", history))

    (stage2,) = llm.stage2_requests
    assert context_lines(stage2) == ["[user]: I moved teams", "[assistant]: Which one?"]


# ---------------------------------------------------------------------------
# Batches
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_thirty_item_list_is_extracted_four_units_a_call():
    text = numbered_list(30)
    batches = batch_units(split_structured(text), LIST_BATCH_MAX_UNITS)
    llm = ScriptedLLM()

    response = await make_engine(llm).run(make_request(text))

    assert len(batches) == 8
    assert [batch_text(r) for r in llm.stage2_requests] == [render_batch(b) for b in batches]
    assert response.stage2_calls == 8
    assert response.attempts == 1
    item_ids = [e["id"] for e in response.payload.entities if e["id"] != "user"]
    assert item_ids == [
        _slug(f"Things I want to try this year: thing {n}") for n in range(1, 31)
    ]
    assert [e["id"] for e in response.payload.entities].count("user") == 1
    assert len(response.payload.relationships) == 30


@pytest.mark.asyncio
async def test_each_item_reaches_the_model_with_its_parent_context():
    text = "Trip plan:\n- Day 1\n  - visit the museum\n- Day 2\n  - hike"
    llm = ScriptedLLM()

    await make_engine(llm).run(make_request(text))

    (stage2,) = llm.stage2_requests
    assert batch_text(stage2) == (
        "Trip plan: Day 1\n"
        "Trip plan > Day 1: visit the museum\n"
        "Trip plan: Day 2\n"
        "Trip plan > Day 2: hike"
    )


@pytest.mark.asyncio
async def test_scope_and_derivation_run_once_on_the_whole_utterance():
    text = numbered_list(9)
    derivation = {
        "signal_types": ["feedback"],
        "matched_patterns": ["feedback:stop"],
        "existing_internal_entities": "No existing internal entities.",
        "assistant_response": "Got it.",
    }
    llm = ScriptedLLM()

    response = await make_engine(llm).run(make_request(text, derivation=derivation))

    (scope,) = llm.requests_with(SCOPE_MAX_TOKENS)
    (derive,) = llm.requests_with(DERIVE_MAX_TOKENS)
    assert text in scope.messages[-1]["content"]
    assert text in derive.messages[-1]["content"]
    assert len(llm.stage2_requests) == 3
    assert response.scope.label == "user-scope"
    assert response.derivation is not None


@pytest.mark.asyncio
async def test_every_batch_carries_the_turn_scope():
    llm = ScriptedLLM()

    await make_engine(llm).run(make_request(numbered_list(9)))

    assert {
        r.messages[1]["content"].split("Subject scope: ")[1].split("\n")[0]
        for r in llm.stage2_requests
    } == {"user-scope"}


# ---------------------------------------------------------------------------
# Earlier units as context
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_later_batches_see_up_to_eight_earlier_units_after_the_history():
    text = numbered_list(30)
    units = split_structured(text)
    history = [{"role": "assistant", "content": "What are you planning?"}]
    llm = ScriptedLLM()

    await make_engine(llm).run(make_request(text, history))

    contexts = [context_lines(r) for r in llm.stage2_requests]
    assert contexts[0] == ["[assistant]: What are you planning?"]
    for index, lines in enumerate(contexts[1:], start=1):
        start = index * LIST_BATCH_MAX_UNITS
        earlier = units[max(0, start - LIST_CONTEXT_MAX_UNITS) : start]
        assert lines == ["[assistant]: What are you planning?"] + [
            f"[user]: {render_unit(u)}" for u in earlier
        ]
    assert len(contexts[1]) == 1 + 4
    assert len(contexts[2]) == 1 + 8
    assert len(contexts[7]) == 1 + 8


@pytest.mark.asyncio
async def test_an_item_can_refer_back_to_an_earlier_batch_item():
    text = (
        "Tools I picked up this year:\n"
        "- Rust, for the backend\n"
        "- Neovim, with the Lazy plugin setup\n"
        "- Terraform\n"
        "- Bun\n"
        "- the same plugin setup as the second one, on my laptop too"
    )
    llm = ScriptedLLM()

    await make_engine(llm).run(make_request(text))

    first, second = llm.stage2_requests
    assert batch_text(second) == (
        "Tools I picked up this year: the same plugin setup as the second one, on my laptop too"
    )
    assert "[user]: Tools I picked up this year: Neovim, with the Lazy plugin setup" in (
        context_lines(second)
    )
    assert context_lines(first) == []


# ---------------------------------------------------------------------------
# Merging, attempts, calls and failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_type_conflict_across_batches_warns():
    def answer(request: LLMRequest, call: int) -> LLMResponse:
        entity_type = "Technology" if call == 1 else "Concept"
        body = {"entities": [{"id": "rust", "name": "Rust", "type": entity_type}]}
        body["relationships"] = []
        return LLMResponse(content=json.dumps(body), finish_reason="stop")

    response = await make_engine(ScriptedLLM(answer)).run(make_request(numbered_list(5)))

    assert response.payload.entities == [{"id": "rust", "name": "Rust", "type": "Technology"}]
    assert "list_merge_type_conflict" in response.warnings


@pytest.mark.asyncio
async def test_a_repaired_batch_counts_its_attempts_and_calls():
    def answer(request: LLMRequest, call: int) -> LLMResponse:
        if call == 2:
            return LLMResponse(content="not json", finish_reason="stop")
        return echo(request, call)

    llm = ScriptedLLM(answer)

    response = await make_engine(llm).run(make_request(numbered_list(9)))

    assert response.attempts == 2
    assert response.stage2_calls == 4
    assert len(llm.stage2_requests) == 4
    assert batch_text(llm.stage2_requests[2]) == batch_text(llm.stage2_requests[1])
    item_ids = [e["id"] for e in response.payload.entities if e["id"] != "user"]
    assert len(item_ids) == 9


@pytest.mark.asyncio
async def test_a_failing_batch_fails_the_turn():
    def answer(request: LLMRequest, call: int) -> LLMResponse:
        if call >= 2:
            return LLMResponse(content="not json", finish_reason="stop")
        return echo(request, call)

    llm = ScriptedLLM(answer)

    with pytest.raises(UpstreamLLMError, match="unparsable after 2 attempts"):
        await make_engine(llm).run(make_request(numbered_list(9)))

    assert len(llm.stage2_requests) == 3


@pytest.mark.asyncio
async def test_the_service_reports_stage2_calls(client, fake_llama_state):
    fake_llama_state.chat_responses = [
        _SCOPE_JSON,
        '{"entities": [], "relationships": []}',
    ]

    response = await client.post(
        "/v1/extract", json=make_extract_request(utterance=numbered_list(5))
    )

    assert response.status_code == 200
    assert response.json()["stage2_calls"] == 2
    assert len(fake_llama_state.chat_requests) == 3
