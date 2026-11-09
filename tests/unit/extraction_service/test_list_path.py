"""Stage 2's list frame path: one frame call per structured turn, then item calls.

A structured turn (one with list items or table rows) is sent to one frame call,
which decides what each group of items states, and to item calls of at most
`LIST_ITEM_BATCH_MAX_UNITS` items each; code assembles the payload. A turn
without structure sends exactly the request it sent under r6. A frame that is cut
short or stays unparsable sends the turn through the r6 list pass; a single item
that does sends only that item through it. A timeout or a transport failure fails
the turn. Every list user message is fitted to `LIST_USER_MAX_CHARS`, and a list
repair runs only when `list_call_fits` holds for the whole message list.

The fake LLM tells the calls apart by their user message: `Stage: FRAME`,
`Stage: ITEMS`, or the r6 template. Every name in these turns is invented.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field

import pytest

from backend.errors import LLMConnectionError
from backend.extraction_service import engine
from backend.extraction_service.adapters import get_adapter
from backend.extraction_service.engine import (
    DERIVE_MAX_TOKENS,
    EXTRACT_MAX_TOKENS,
    LIST_BATCH_MAX_UNITS,
    LIST_FRAME_MAX_TOKENS,
    LIST_ITEM_BATCH_MAX_UNITS,
    LIST_ITEM_MAX_TOKENS,
    LIST_USER_MAX_CHARS,
    SCOPE_MAX_TOKENS,
    ExtractionEngine,
    ExtractionTimeoutError,
    UpstreamLLMError,
    _OutputUnparsableError,
    list_call_fits,
    parse_recorded_at,
)
from backend.extraction_service.list_schemas import (
    LIST_FRAME_OUTPUT_SCHEMA,
    LIST_ITEM_OUTPUT_SCHEMA,
    list_frame_output_schema,
    list_item_output_schema,
)
from backend.extraction_service.schemas import EXTRACTION_OUTPUT_SCHEMA
from backend.extraction_service.settings import ServiceSettings
from backend.knowledge.extraction.list_frames import (
    ListOutputError,
    group_units,
    item_slots,
    parse_frame_output,
    parse_item_output,
)
from backend.knowledge.extraction.list_prompts import (
    LIST_FRAME_REPAIR_INSTRUCTION,
    LIST_ITEM_REPAIR_INSTRUCTION,
    LIST_SYSTEM_PROMPT,
    list_repair_text,
)
from backend.knowledge.extraction.ontology_extractor import (
    ExtractionResult,
    render_extraction_messages,
)
from backend.knowledge.extraction.preprocessor import PreProcessor
from backend.knowledge.extraction.structure_splitter import (
    batch_units,
    render_batch,
    render_unit,
    split_structured,
)
from backend.knowledge.extraction.validator import ExtractionValidator
from backend.llm.models import LLMRequest, LLMResponse
from backend.llm.provider import StreamingLLMProvider
from tests.unit.extraction_service.test_list_pass import (
    batch_text,
    context_lines,
    make_request,
    payload_for,
)

_SCOPE_JSON = '{"scope": "user-scope", "confidence": 0.9, "reasoning": "first person"}'
_NAMES = (
    "Quillon",
    "Brask",
    "Tervane",
    "Ostrel",
    "Vimmet",
    "Calder",
    "Pronx",
    "Zelkor",
    "Harrowin",
    "Mispel",
)
_GROUP_LINE = re.compile(r"^(G\d+)\b", re.MULTILINE)
_ITEM_LINE = re.compile(r"^  (G\d+\.\d+) (.*)$", re.MULTILINE)
_ASKED = re.compile(r"^Extract these items: (.*)$", re.MULTILINE)
_CONTEXT = re.compile(r"^Stage: \w+\nContext:\n(?P<context>.*?)\nSubject scope: ", re.DOTALL)


def tool_list(count: int, lead_in: str = "Tools I use at work:") -> str:
    return lead_in + "\n" + "\n".join(f"- {_NAMES[n]}" for n in range(count))


# ---------------------------------------------------------------------------
# The fake LLM
# ---------------------------------------------------------------------------


def stage_of(request: LLMRequest) -> str:
    if request.max_tokens == SCOPE_MAX_TOKENS:
        return "scope"
    if request.max_tokens == DERIVE_MAX_TOKENS:
        return "derive"
    content = request.messages[1]["content"]
    if content.startswith("Stage: FRAME"):
        return "frame"
    if content.startswith("Stage: ITEMS"):
        return "items"
    return "r6"


def _units_section(request: LLMRequest) -> str:
    return request.messages[1]["content"].split("\nUnits:\n", 1)[1]


def group_ids(request: LLMRequest) -> list[str]:
    return _GROUP_LINE.findall(_units_section(request).split("\n\n", 1)[0])


def asked_items(request: LLMRequest) -> list[str]:
    match = _ASKED.search(request.messages[1]["content"])
    assert match is not None
    return match.group(1).split(", ")


def list_context(request: LLMRequest) -> str:
    match = _CONTEXT.match(request.messages[1]["content"])
    assert match is not None
    return match.group("context")


def frame_ok(request: LLMRequest, call: int) -> LLMResponse:
    groups = [
        {
            "group": gid,
            "durable": True,
            "templates": [{"subject": "user", "predicate": "USES", "object": "item"}],
            "item_type": "Technology",
            "unchecked": "plan",
        }
        for gid in group_ids(request)
    ]
    body = {"frame_entities": [], "groups": groups, "facts": {"entities": [], "relationships": []}}
    return LLMResponse(content=json.dumps(body), finish_reason="stop")


def items_ok(request: LLMRequest, call: int) -> LLMResponse:
    texts = dict(_ITEM_LINE.findall(_units_section(request)))
    records = [
        {
            "item": item_id,
            "entity": {"span": texts[item_id], "name": texts[item_id], "type": "Technology"},
            "template": {"applies": True, "assertion_kind": "assert", "temporal_status": "current"},
            "entities": [],
            "relationships": [],
        }
        for item_id in asked_items(request)
    ]
    return LLMResponse(content=json.dumps({"items": records}), finish_reason="stop")


def r6_echo(request: LLMRequest, call: int) -> LLMResponse:
    return LLMResponse(content=json.dumps(payload_for(batch_text(request))), finish_reason="stop")


Answer = Callable[[LLMRequest, int], LLMResponse]

_CUT_SHORT = LLMResponse(content='{"items": [{"item": "G1', finish_reason="length")
_NOT_JSON = LLMResponse(content="not json", finish_reason="stop")


class ListLLM(StreamingLLMProvider):
    """Answers each stage through its own function, given the 1-based call index."""

    model = "scripted-list"

    def __init__(self, *, frame: Answer = frame_ok, items: Answer = items_ok, r6: Answer = r6_echo):
        self.requests: list[LLMRequest] = []
        self._answers = {"frame": frame, "items": items, "r6": r6}

    async def generate(self, request: LLMRequest, *, stream: bool = False):
        self.requests.append(request)
        yield self._answer(request)

    def generate_sync(self, request: LLMRequest, *, stream: bool = False):
        raise NotImplementedError

    def _answer(self, request: LLMRequest) -> LLMResponse:
        stage = stage_of(request)
        if stage == "scope":
            return LLMResponse(content=_SCOPE_JSON, finish_reason="stop")
        if stage == "derive":
            return LLMResponse(content='{"operations": []}', finish_reason="stop")
        return self._answers[stage](request, len(self.of(stage)))

    def of(self, stage: str) -> list[LLMRequest]:
        return [r for r in self.requests if stage_of(r) == stage]

    @property
    def stage2(self) -> list[LLMRequest]:
        return [r for r in self.requests if stage_of(r) in ("frame", "items", "r6")]


@dataclass
class RecordingAdapter:
    """The gptoss adapter, recording the stage each request is prepared for."""

    inner: object = field(default_factory=lambda: get_adapter("gptoss"))
    stages: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def version(self) -> str:
        return self.inner.version

    @property
    def default_constrained_mode(self):
        return self.inner.default_constrained_mode

    def prepare(self, request: LLMRequest, stage: str) -> LLMRequest:
        self.stages.append(stage)
        return self.inner.prepare(request, stage)

    def final_text(self, response: LLMResponse) -> str:
        return self.inner.final_text(response)


def _settings(max_attempts: int = 2) -> ServiceSettings:
    return ServiceSettings(
        llm_base_url="http://unused",
        model_hash="test-model-hash",
        model_file="test-model.gguf",
        max_attempts=max_attempts,
        llm_timeout_seconds=10.0,
    )


def make_engine(llm: StreamingLLMProvider, *, max_attempts: int = 2, adapter=None):
    return ExtractionEngine(
        llm=llm, adapter=adapter or get_adapter("gptoss"), settings=_settings(max_attempts)
    )


def _is_repair(request: LLMRequest) -> bool:
    return len(request.messages) > 2


# ---------------------------------------------------------------------------
# The limits and the budget
# ---------------------------------------------------------------------------


def test_the_list_path_limits():
    assert (
        LIST_FRAME_MAX_TOKENS,
        LIST_ITEM_MAX_TOKENS,
        LIST_ITEM_BATCH_MAX_UNITS,
        LIST_USER_MAX_CHARS,
    ) == (2048, 2048, 4, 7200)


def test_the_budget_inequality_holds_on_the_list_system_prompt():
    assert (
        math.ceil(len(LIST_SYSTEM_PROMPT) / 3.6) + math.ceil(LIST_USER_MAX_CHARS / 3) + 256 <= 14336
    )


def test_list_call_fits_counts_the_system_prompt_and_every_other_message():
    system = {"role": "system", "content": "s" * 36}  # 10 tokens at 3.6 characters a token
    room = 14336 - 256 - 10
    fits = [system, {"role": "user", "content": "u" * (room * 3 - 3)}]
    fits.append({"role": "assistant", "content": "a" * 3})

    assert list_call_fits(fits)
    assert not list_call_fits([*fits, {"role": "user", "content": "r"}])


# ---------------------------------------------------------------------------
# A turn without structure sends r6's request
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "utterance,history",
    [
        ("I have been learning Kestrin for a few months", []),
        ("Marlow and I moved to Ostby last spring.", [{"role": "user", "content": "Hi"}]),
        ('Use {braces} and "quotes" here', [{"role": "assistant", "content": "Go on"}]),
    ],
)
async def test_a_prose_turn_sends_the_r6_request_byte_for_byte(utterance, history):
    llm = ListLLM()
    req = make_request(utterance, history)

    await make_engine(llm).run(req)

    pre_processed = PreProcessor().pre_process(
        utterance=utterance,
        conversation_history=history,
        reference_date=parse_recorded_at(req.recorded_at),
        turn_index=req.turn_index,
    )
    pre_processed.metadata["subject_scope"] = "user-scope"
    r6_request = get_adapter("gptoss").prepare(
        LLMRequest(
            messages=render_extraction_messages(pre_processed),
            temperature=_settings().temperature,
            max_tokens=EXTRACT_MAX_TOKENS,
            json_mode=False,
            response_schema=EXTRACTION_OUTPUT_SCHEMA,
        ),
        stage="extract",
    )
    (stage2,) = llm.stage2
    assert json.dumps(stage2.model_dump()) == json.dumps(r6_request.model_dump())


# ---------------------------------------------------------------------------
# A structured turn: one frame call, then item calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [1, 4, 5, 9])
async def test_a_structured_turn_makes_one_frame_call_and_an_item_call_per_four_items(count):
    llm = ListLLM()

    response = await make_engine(llm).run(make_request(tool_list(count)))

    assert len(llm.of("frame")) == 1
    assert len(llm.of("items")) == math.ceil(count / LIST_ITEM_BATCH_MAX_UNITS)
    assert llm.of("r6") == []
    assert response.stage2_calls == 1 + math.ceil(count / 4)
    assert response.attempts == 1
    assert response.warnings == []


@pytest.mark.asyncio
async def test_the_payload_is_assembled_from_the_frame_and_the_items():
    llm = ListLLM()

    response = await make_engine(llm).run(make_request(tool_list(5)))

    ids = [e["id"] for e in response.payload.entities]
    assert sorted(ids) == sorted(["user", *(name.lower() for name in _NAMES[:5])])
    triples = {(r["source"], r["type"], r["target"]) for r in response.payload.relationships}
    assert triples == {("user", "USES", name.lower()) for name in _NAMES[:5]}


@pytest.mark.asyncio
async def test_item_batches_follow_turn_order_and_may_span_groups():
    text = "Languages:\n- Quillon\n- Brask\n\nEditors:\n- Tervane\n- Ostrel\n- Vimmet"
    groups, _prose = group_units(split_structured(text))
    llm = ListLLM()

    await make_engine(llm).run(make_request(text))

    assert len(groups) == 2
    assert [asked_items(r) for r in llm.of("items")] == [
        ["G1.1", "G1.2", "G2.1", "G2.2"],
        ["G2.3"],
    ]


def built_schemas(text: str, frame_text: str, batches: list[list[str]]) -> tuple[dict, list[dict]]:
    """The frame schema and each batch's item schema the builders give for this turn."""
    groups, _prose = group_units(split_structured(text))
    item_ids = [item for group in groups for item in group.item_ids]
    nested = {group.gid for group in groups if group.parent is not None}
    frame = parse_frame_output(frame_text, groups)
    keys = [entity.key for entity in frame.frame_entities]
    frame_schema = list_frame_output_schema([g.gid for g in groups], nested, item_ids)
    item_schemas = [
        list_item_output_schema(item_slots(frame, groups, batch), keys, item_ids)
        for batch in batches
    ]
    return frame_schema, item_schemas


@pytest.mark.asyncio
async def test_the_list_calls_send_their_schemas_limits_and_one_system_prompt():
    llm = ListLLM()
    req = make_request(tool_list(5))

    await make_engine(llm).run(req)

    (frame,) = llm.of("frame")
    items = llm.of("items")
    frame_schema, item_schemas = built_schemas(
        tool_list(5), frame_ok(frame, 1).content, [asked_items(r) for r in items]
    )
    assert (frame.max_tokens, frame.response_schema) == (LIST_FRAME_MAX_TOKENS, frame_schema)
    assert {r.max_tokens for r in items} == {LIST_ITEM_MAX_TOKENS}
    assert [r.response_schema for r in items] == item_schemas
    assert frame.response_schema != LIST_FRAME_OUTPUT_SCHEMA
    assert all(r.response_schema != LIST_ITEM_OUTPUT_SCHEMA for r in items)
    reference_date = parse_recorded_at(req.recorded_at).strftime("%Y-%m-%d")
    system = LIST_SYSTEM_PROMPT.format(reference_date=reference_date)
    assert {r.messages[0]["content"] for r in [frame, *items]} == {system}
    assert "Subject scope: user-scope\n" in frame.messages[1]["content"]
    assert "Frames:\nG1: each item is the object of user USES" in items[0].messages[1]["content"]


_NESTED_TURN = "Projects at Zentrova:\n- Harbor\n  - Quillon\n  - Brask\n- Lumen\n  - Tervane"


def nested_frame(request: LLMRequest, call: int) -> LLMResponse:
    """A frame with a frame entity, and a parent template in each nested group."""
    groups = [
        {
            "group": gid,
            "durable": True,
            "templates": (
                []
                if gid == "G1"
                else [{"subject": "parent", "predicate": "DEPENDS_ON", "object": "item"}]
            ),
            "item_type": None if gid == "G1" else "Technology",
            "unchecked": "plan",
        }
        for gid in group_ids(request)
    ]
    frame_entities = [{"key": "f1", "span": "Zentrova", "name": "Zentrova", "type": "Organization"}]
    body = {
        "frame_entities": frame_entities,
        "groups": groups,
        "facts": {"entities": [], "relationships": []},
    }
    return LLMResponse(content=json.dumps(body), finish_reason="stop")


@pytest.mark.asyncio
async def test_a_nested_turn_sends_the_schemas_built_from_its_groups_and_frame():
    llm = ListLLM(frame=nested_frame)

    response = await make_engine(llm).run(make_request(_NESTED_TURN))

    (frame,) = llm.of("frame")
    items = llm.of("items")
    batches = [asked_items(r) for r in items]
    frame_schema, item_schemas = built_schemas(
        _NESTED_TURN, nested_frame(frame, 1).content, batches
    )
    assert batches == [["G1.1", "G2.1", "G2.2", "G1.2"], ["G3.1"]]
    assert frame.response_schema == frame_schema
    assert [r.response_schema for r in items] == item_schemas
    assert response.warnings == []


@pytest.mark.asyncio
async def test_frame_and_item_calls_are_prepared_as_the_extract_stage():
    adapter = RecordingAdapter()
    llm = ListLLM()

    await make_engine(llm, adapter=adapter).run(make_request(tool_list(5)))

    assert adapter.stages == ["scope", "extract", "extract", "extract"]


# ---------------------------------------------------------------------------
# Repair
# ---------------------------------------------------------------------------


def fail_first(answer: Answer, failure: LLMResponse = _NOT_JSON) -> Answer:
    def wrapped(request: LLMRequest, call: int) -> LLMResponse:
        return failure if call == 1 else answer(request, call)

    return wrapped


def refusal(parse: Callable[[], object]) -> ListOutputError:
    """The error the list parser refuses an output with."""
    with pytest.raises(ListOutputError) as caught:
        parse()
    return caught.value


def turn_groups(text: str):
    return group_units(split_structured(text))[0]


@pytest.mark.asyncio
async def test_a_malformed_frame_output_gets_the_frame_repair_then_succeeds():
    llm = ListLLM(frame=fail_first(frame_ok))

    response = await make_engine(llm).run(make_request(tool_list(3)))

    first, second = llm.of("frame")
    error = refusal(lambda: parse_frame_output("not json", turn_groups(tool_list(3))))
    assert second.messages[2:] == [
        {"role": "assistant", "content": "not json"},
        {"role": "user", "content": list_repair_text(LIST_FRAME_REPAIR_INSTRUCTION, error)},
    ]
    assert second.messages[3]["content"].startswith(LIST_FRAME_REPAIR_INSTRUCTION + "\n")
    assert second.messages[3]["content"].endswith(f"\nThe output was refused: {error}")
    assert second.messages[:2] == first.messages
    assert response.attempts == 2
    assert response.stage2_calls == 3
    assert response.warnings == []


@pytest.mark.asyncio
async def test_a_malformed_item_output_gets_the_item_repair_then_succeeds():
    llm = ListLLM(items=fail_first(items_ok))

    response = await make_engine(llm).run(make_request(tool_list(3)))

    first, second = llm.of("items")
    groups = turn_groups(tool_list(3))
    frame = parse_frame_output(frame_ok(llm.of("frame")[0], 1).content, groups)
    error = refusal(
        lambda: parse_item_output("not json", ["G1.1", "G1.2", "G1.3"], frame, groups, {})
    )
    assert second.messages[2:] == [
        {"role": "assistant", "content": "not json"},
        {"role": "user", "content": list_repair_text(LIST_ITEM_REPAIR_INSTRUCTION, error)},
    ]
    assert response.stage2_calls == 3
    assert response.warnings == []
    assert len(response.payload.relationships) == 3


# ---------------------------------------------------------------------------
# Fallbacks
# ---------------------------------------------------------------------------


def always(failure: LLMResponse) -> Answer:
    return lambda request, call: failure


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [_CUT_SHORT, _NOT_JSON], ids=["cut-short", "unparsable"])
async def test_a_frame_failure_sends_the_turn_through_the_r6_pass(failure):
    text = tool_list(6)
    llm = ListLLM(frame=always(failure))

    response = await make_engine(llm).run(make_request(text))

    batches = batch_units(split_structured(text), LIST_BATCH_MAX_UNITS)
    assert [batch_text(r) for r in llm.of("r6")] == [render_batch(b) for b in batches]
    assert llm.of("items") == []
    assert response.warnings == ["list_frame_fallback"]
    assert response.stage2_calls == len(llm.stage2)
    assert len(llm.of("frame")) == (1 if failure is _CUT_SHORT else 2)


def fail_when_asked(item_id: str, failure: LLMResponse) -> Answer:
    def answer(request: LLMRequest, call: int) -> LLMResponse:
        return failure if item_id in asked_items(request) else items_ok(request, call)

    return answer


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [_CUT_SHORT, _NOT_JSON], ids=["cut-short", "unparsable"])
async def test_one_item_failing_falls_back_for_that_item_only(failure):
    text = tool_list(5)
    units = split_structured(text)
    llm = ListLLM(items=fail_when_asked("G1.3", failure))

    response = await make_engine(llm).run(make_request(text))

    assert [asked_items(r) for r in llm.of("items") if not _is_repair(r)] == [
        ["G1.1", "G1.2", "G1.3", "G1.4"],
        ["G1.1", "G1.2"],
        ["G1.3", "G1.4"],
        ["G1.3"],
        ["G1.4"],
        ["G1.5"],
    ]
    (fallback,) = llm.of("r6")
    third = next(u for u in units if "Tervane" in u.text)
    assert batch_text(fallback) == render_unit(third)
    assert context_lines(fallback) == [
        f"[user]: {render_unit(u)}" for u in units[: units.index(third)]
    ]
    assert response.warnings == ["list_item_fallback"]
    assert response.stage2_calls == len(llm.stage2)
    ids = {e["id"] for e in response.payload.entities}
    assert {"quillon", "brask", "ostrel", "vimmet"} <= ids
    assert "tervane" not in ids
    assert payload_for(render_unit(third))["entities"][1]["id"] in ids


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["frame", "items"])
@pytest.mark.parametrize(
    "error,raised",
    [
        (TimeoutError(), ExtractionTimeoutError),
        (LLMConnectionError("connection refused"), UpstreamLLMError),
    ],
    ids=["timeout", "transport"],
)
async def test_a_timeout_or_transport_failure_fails_the_turn_with_no_fallback(stage, error, raised):
    def fail(request: LLMRequest, call: int) -> LLMResponse:
        raise error

    llm = ListLLM(**{stage: fail})

    with pytest.raises(raised) as caught:
        await make_engine(llm).run(make_request(tool_list(5)))

    assert not isinstance(caught.value, _OutputUnparsableError)
    assert llm.of("r6") == []
    assert len(llm.of(stage)) == 1


# ---------------------------------------------------------------------------
# The message budget
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_long_history_is_trimmed_oldest_first_to_fit_the_limit():
    history = [
        {"role": "user" if n % 2 == 0 else "assistant", "content": f"turn {n} " + "w" * 1500}
        for n in range(6)
    ]
    text = tool_list(5)
    llm = ListLLM()

    response = await make_engine(llm).run(make_request(text, history))

    full = [f"[{m['role']}]: {m['content']}" for m in history]
    (frame,) = llm.of("frame")
    kept = list_context(frame).split("\n")
    assert 0 < len(kept) < len(full)
    assert kept == full[len(full) - len(kept) :]
    assert len(frame.messages[1]["content"]) <= LIST_USER_MAX_CHARS
    longer = frame.messages[1]["content"].replace(
        list_context(frame), "\n".join(full[len(full) - len(kept) - 1 :])
    )
    assert len(longer) > LIST_USER_MAX_CHARS
    for request in llm.of("items"):
        assert len(request.messages[1]["content"]) <= LIST_USER_MAX_CHARS
        lines = list_context(request).split("\n")
        assert lines == kept[len(kept) - len(lines) :]
    assert response.warnings == ["list_history_trimmed"]


@pytest.mark.asyncio
async def test_a_frame_message_over_the_limit_without_history_takes_the_r6_pass():
    text = "Things to try:\n" + "\n".join(f"- {'Varnel ' * 6}number {n}" for n in range(120))
    llm = ListLLM()

    response = await make_engine(llm).run(make_request(text))

    batches = batch_units(split_structured(text), LIST_BATCH_MAX_UNITS)
    assert llm.of("frame") == [] and llm.of("items") == []
    assert len(llm.of("r6")) == len(batches)
    assert response.warnings == ["list_frame_over_budget"]


@pytest.mark.asyncio
async def test_an_item_message_over_the_limit_without_history_takes_the_item_fallback(
    monkeypatch,
):
    text = tool_list(5)
    probe = ListLLM()
    await make_engine(probe).run(make_request(text))
    (frame,) = probe.of("frame")
    monkeypatch.setattr(engine, "LIST_USER_MAX_CHARS", len(frame.messages[1]["content"]))
    llm = ListLLM()

    response = await make_engine(llm).run(make_request(text))

    assert len(llm.of("frame")) == 1
    assert llm.of("items") == []
    assert len(llm.of("r6")) == 5
    assert response.warnings == ["list_item_fallback"] * 5


@pytest.mark.asyncio
async def test_a_frame_repair_that_would_not_fit_is_skipped_and_the_turn_falls_back():
    long_output = LLMResponse(content="x" * 12000, finish_reason="stop")
    llm = ListLLM(frame=always(long_output))

    response = await make_engine(llm).run(make_request(tool_list(3)))

    assert len(llm.of("frame")) == 1
    assert len(llm.of("r6")) == 1
    assert response.warnings == ["list_repair_skipped", "list_frame_fallback"]
    assert response.stage2_calls == 2


@pytest.mark.asyncio
async def test_an_item_repair_that_would_not_fit_is_skipped():
    long_output = LLMResponse(content="x" * 12000, finish_reason="stop")
    llm = ListLLM(items=always(long_output))

    response = await make_engine(llm).run(make_request(tool_list(1)))

    assert len(llm.of("items")) == 1
    assert len(llm.of("r6")) == 1
    assert response.warnings == ["list_repair_skipped", "list_item_fallback"]


@pytest.mark.asyncio
async def test_with_three_attempts_the_second_repair_estimate_counts_both_failed_outputs(
    monkeypatch,
):
    seen: list[list[dict]] = []

    def recording_fits(messages):
        seen.append(list(messages))
        return list_call_fits(messages)

    monkeypatch.setattr(engine, "list_call_fits", recording_fits)
    outputs = iter(["bad one", "bad two", "bad three"])
    llm = ListLLM(
        frame=lambda request, call: LLMResponse(content=next(outputs), finish_reason="stop")
    )

    response = await make_engine(llm, max_attempts=3).run(make_request(tool_list(2)))

    assert len(seen) == 2
    assert [m["content"] for m in seen[1] if m["role"] == "assistant"] == ["bad one", "bad two"]
    groups = turn_groups(tool_list(2))
    assert [m["content"] for m in seen[1][2:] if m["role"] == "user"] == [
        list_repair_text(
            LIST_FRAME_REPAIR_INSTRUCTION, refusal(lambda t=text: parse_frame_output(t, groups))
        )
        for text in ("bad one", "bad two")
    ]
    assert len(llm.of("frame")) == 3
    assert response.warnings == ["list_frame_fallback"]


@pytest.mark.asyncio
async def test_the_r6_repair_is_not_budgeted():
    huge = LLMResponse(content="x" * 60000, finish_reason="stop")
    llm = ListLLM(r6=fail_first(r6_echo, huge))

    response = await make_engine(llm).run(make_request("I use Kestrin for my backend work"))

    assert [_is_repair(r) for r in llm.of("r6")] == [False, True]
    assert response.attempts == 2
    assert response.warnings == []


# ---------------------------------------------------------------------------
# The r6 path keeps its request and its fixed repair
# ---------------------------------------------------------------------------


def _dumped(request: LLMRequest) -> str:
    return json.dumps(request.model_dump())


def _r6_request(messages: list[dict]) -> LLMRequest:
    return get_adapter("gptoss").prepare(
        LLMRequest(
            messages=messages,
            temperature=_settings().temperature,
            max_tokens=EXTRACT_MAX_TOKENS,
            json_mode=False,
            response_schema=EXTRACTION_OUTPUT_SCHEMA,
        ),
        stage="extract",
    )


def _pre_processed(req):
    pre_processed = PreProcessor().pre_process(
        utterance=req.utterance,
        conversation_history=[],
        reference_date=parse_recorded_at(req.recorded_at),
        turn_index=req.turn_index,
    )
    pre_processed.metadata["subject_scope"] = "user-scope"
    return pre_processed


def _repaired(messages: list[dict], output: str) -> list[dict]:
    return [
        *messages,
        {"role": "assistant", "content": output},
        {"role": "user", "content": engine._REPAIR_INSTRUCTION},
    ]


@pytest.mark.asyncio
async def test_a_prose_turn_sends_the_fixed_r6_repair_byte_for_byte():
    req = make_request("I have been learning Kestrin for a few months")
    llm = ListLLM(r6=fail_first(r6_echo))

    response = await make_engine(llm).run(req)

    first = render_extraction_messages(_pre_processed(req))
    expected = [first, _repaired(first, "not json")]
    assert [_dumped(r) for r in llm.of("r6")] == [_dumped(_r6_request(m)) for m in expected]
    assert "The output was refused" not in engine._REPAIR_INSTRUCTION
    assert response.attempts == 2


@pytest.mark.asyncio
async def test_an_r6_list_pass_turn_sends_its_requests_and_fixed_repair_byte_for_byte(
    monkeypatch,
):
    monkeypatch.setattr(engine, "LIST_USER_MAX_CHARS", 0)
    req = make_request(tool_list(5))
    llm = ListLLM(r6=fail_first(r6_echo))

    response = await make_engine(llm).run(req)

    pre_processed = _pre_processed(req)
    units = split_structured(pre_processed.original_text)
    head = render_extraction_messages(engine._batch_input(pre_processed, units, 0, 4))
    tail = render_extraction_messages(engine._batch_input(pre_processed, units, 4, 1))
    expected = [head, _repaired(head, "not json"), tail]
    assert [_dumped(r) for r in llm.of("r6")] == [_dumped(_r6_request(m)) for m in expected]
    assert response.warnings == ["list_frame_over_budget"]


# ---------------------------------------------------------------------------
# Replays: each refused list output is repaired once, by the line naming the refusal
# ---------------------------------------------------------------------------


def scripted(*bodies: dict) -> Answer:
    """Answer the n-th call with the n-th body, and every later call with the last one."""

    def answer(request: LLMRequest, call: int) -> LLMResponse:
        body = bodies[min(call, len(bodies)) - 1]
        return LLMResponse(content=json.dumps(body), finish_reason="stop")

    return answer


def entity(span: str, etype: str, **extra: str) -> dict:
    return {"span": span, "name": span, "type": etype, **extra}


def keyed(key: str, span: str, etype: str, **extra: str) -> dict:
    return {"key": key, **entity(span, etype, **extra)}


def edge(source: str, type_: str, target: str) -> dict:
    return {"source": source, "target": target, "type": type_}


def group(gid: str, templates: list[tuple[str, str, str]], item_type: str | None) -> dict:
    return {
        "group": gid,
        "durable": True,
        "templates": [dict(zip(("subject", "predicate", "object"), t)) for t in templates],
        "item_type": item_type,
        "unchecked": "plan",
    }


def frame_body(groups: list[dict], frame_entities=(), facts_entities=()) -> dict:
    return {
        "frame_entities": list(frame_entities),
        "groups": groups,
        "facts": {"entities": list(facts_entities), "relationships": []},
    }


def item_record(item_id: str, own: dict | None, entities=(), relationships=()) -> dict:
    return {
        "item": item_id,
        "template": {"applies": True, "assertion_kind": "assert", "temporal_status": "current"},
        "entity": own,
        "entities": list(entities),
        "relationships": list(relationships),
    }


async def replay(text: str, *, frame: Answer, items: Answer = items_ok):
    """Run the turn with three attempts allowed, and check it ends clean after one repair."""
    llm = ListLLM(frame=frame, items=items)

    response = await make_engine(llm, max_attempts=3).run(make_request(text))

    assert "list_item_ref_unresolved" not in response.warnings
    assert "list_frame_fallback" not in response.warnings
    assert "list_item_fallback" not in response.warnings
    assert llm.of("r6") == []
    assert all(len(r.messages) <= 4 for r in llm.stage2), "a call took a second repair"
    return llm, response


def predicates(response) -> list[str]:
    return sorted(r["type"] for r in response.payload.relationships)


@pytest.mark.asyncio
async def test_a_people_list_whose_first_items_have_no_entity_is_repaired_u2():
    text = "People I work with at Zentrova:\n- Alba Venn\n- Corvel Osk"
    frame = frame_body(
        [group("G1", [("user", "KNOWS_PERSON", "item"), ("item", "WORKS_AT", "f1")], "Person")],
        frame_entities=[keyed("f1", "Zentrova", "Organization")],
    )
    empty = {"items": [item_record("G1.1", None), item_record("G1.2", None)]}
    named = {
        "items": [
            item_record("G1.1", entity("Alba Venn", "Person")),
            item_record("G1.2", entity("Corvel Osk", "Person")),
        ]
    }

    llm, response = await replay(text, frame=scripted(frame), items=scripted(empty, named))

    groups = turn_groups(text)
    parsed = parse_frame_output(json.dumps(frame), groups)
    error = refusal(
        lambda: parse_item_output(json.dumps(empty), ["G1.1", "G1.2"], parsed, groups, {})
    )
    assert "requires an entity" in str(error)
    _first, second = llm.of("items")
    assert second.messages[3]["content"] == list_repair_text(LIST_ITEM_REPAIR_INSTRUCTION, error)
    assert predicates(response) == ["KNOWS_PERSON"] * 2 + ["WORKS_AT"] * 2


@pytest.mark.asyncio
async def test_a_template_naming_parent_in_a_top_level_group_is_repaired_u4():
    text = tool_list(3)
    wrong = frame_body([group("G1", [("parent", "USES", "item")], "Technology")])
    right = frame_body([group("G1", [("user", "USES", "item")], "Technology")])

    llm, response = await replay(text, frame=scripted(wrong, right))

    error = refusal(lambda: parse_frame_output(json.dumps(wrong), turn_groups(text)))
    assert "top-level group" in str(error)
    _first, second = llm.of("frame")
    assert second.messages[3]["content"] == list_repair_text(LIST_FRAME_REPAIR_INSTRUCTION, error)
    assert predicates(response) == ["USES"] * 3


@pytest.mark.asyncio
async def test_an_item_edge_naming_an_entity_by_name_is_repaired_u7():
    text = "Tools I use at work:\n- Quillon with the Brask plugin\n- Tervane"
    frame = frame_body([group("G1", [("user", "USES", "item")], "Technology")])

    def items(target: str) -> dict:
        return {
            "items": [
                item_record(
                    "G1.1",
                    entity("Quillon", "Technology"),
                    entities=[keyed("e1", "Brask", "Technology")],
                    relationships=[edge("item", "DEPENDS_ON", target)],
                ),
                item_record("G1.2", entity("Tervane", "Technology")),
            ]
        }

    llm, response = await replay(
        text, frame=scripted(frame), items=scripted(items("Brask"), items("e1"))
    )

    groups = turn_groups(text)
    parsed = parse_frame_output(json.dumps(frame), groups)
    error = refusal(
        lambda: parse_item_output(json.dumps(items("Brask")), ["G1.1", "G1.2"], parsed, groups, {})
    )
    assert "'Brask' is not user" in str(error)
    _first, second = llm.of("items")
    assert second.messages[3]["content"] == list_repair_text(LIST_ITEM_REPAIR_INSTRUCTION, error)
    assert predicates(response) == ["DEPENDS_ON", "USES", "USES"]


@pytest.mark.asyncio
async def test_a_template_naming_a_facts_key_is_repaired_f1():
    text = "Tools we use at Zentrova:\n- Quillon\n- Brask"
    wrong = frame_body(
        [group("G1", [("user", "USES", "item"), ("e1", "USES", "item")], "Technology")],
        facts_entities=[keyed("e1", "Zentrova", "Organization")],
    )
    right = frame_body(
        [group("G1", [("user", "USES", "item"), ("f1", "USES", "item")], "Technology")],
        frame_entities=[keyed("f1", "Zentrova", "Organization")],
    )

    llm, response = await replay(text, frame=scripted(wrong, right))

    error = refusal(lambda: parse_frame_output(json.dumps(wrong), turn_groups(text)))
    assert "'e1' is not user, mist, item" in str(error)
    _first, second = llm.of("frame")
    assert second.messages[3]["content"] == list_repair_text(LIST_FRAME_REPAIR_INSTRUCTION, error)
    assert predicates(response) == ["USES"] * 4


@pytest.mark.asyncio
async def test_a_service_latency_table_keeps_its_per_row_metric_edges():
    text = (
        "Latency by service:\n\n"
        "| Service | Latency |\n|---|---|\n| Vornix | 40 ms |\n| Quellan | 75 ms |"
    )
    frame = frame_body([group("G1", [], "Technology")])

    def row(item_id: str, name: str, latency: str) -> dict:
        value, unit = latency.split()
        return item_record(
            item_id,
            entity(name, "Technology"),
            entities=[keyed("e1", latency, "Metric", value=value, unit=unit)],
            relationships=[edge("item", "HAS_METRIC", "e1")],
        )

    items = {"items": [row("G1.1", "Vornix", "40 ms"), row("G1.2", "Quellan", "75 ms")]}

    llm, response = await replay(text, frame=scripted(frame), items=scripted(items))

    by_id = {e["id"]: e for e in response.payload.entities}
    metric_edges = {
        (by_id[r["source"]]["name"], by_id[r["target"]]["type"])
        for r in response.payload.relationships
        if r["type"] == "HAS_METRIC"
    }
    assert metric_edges == {("Vornix", "Metric"), ("Quellan", "Metric")}
    assert len(llm.of("frame")) == len(llm.of("items")) == 1


@pytest.mark.asyncio
async def test_a_document_list_under_experienced_is_repaired_and_every_edge_validates():
    text = "Books I read this year:\n- The Glass Orchard\n- Northwind Ledger"
    wrong = frame_body([group("G1", [("user", "EXPERIENCED", "item")], "Document")])
    right = frame_body([group("G1", [("user", "REFERENCES_DOCUMENT", "item")], "Document")])
    items = {
        "items": [
            item_record("G1.1", entity("The Glass Orchard", "Document")),
            item_record("G1.2", entity("Northwind Ledger", "Document")),
        ]
    }

    llm, response = await replay(text, frame=scripted(wrong, right), items=scripted(items))

    error = refusal(lambda: parse_frame_output(json.dumps(wrong), turn_groups(text)))
    assert "EXPERIENCED does not admit Document" in str(error)
    _first, second = llm.of("frame")
    assert second.messages[3]["content"] == list_repair_text(LIST_FRAME_REPAIR_INSTRUCTION, error)
    payload = response.payload
    validated = ExtractionValidator(min_confidence=0.0).validate(
        ExtractionResult(entities=payload.entities, relationships=payload.relationships)
    )
    assert predicates(response) == ["REFERENCES_DOCUMENT"] * 2
    assert len(validated.relationships) == len(payload.relationships)


@pytest.mark.asyncio
async def test_a_batch_of_four_with_one_persistently_refused_item_halves_to_nine_calls():
    def answer(request: LLMRequest, call: int) -> LLMResponse:
        body = json.loads(items_ok(request, call).content)
        for record in body["items"]:
            if record["item"] == "G1.3":
                record["entity"] = None
        return LLMResponse(content=json.dumps(body), finish_reason="stop")

    llm = ListLLM(items=answer)

    response = await make_engine(llm).run(make_request(tool_list(4)))

    assert [asked_items(r) for r in llm.of("items")] == [
        ["G1.1", "G1.2", "G1.3", "G1.4"],
        ["G1.1", "G1.2", "G1.3", "G1.4"],
        ["G1.1", "G1.2"],
        ["G1.3", "G1.4"],
        ["G1.3", "G1.4"],
        ["G1.3"],
        ["G1.3"],
        ["G1.4"],
    ]
    assert len(llm.of("items")) + len(llm.of("r6")) == 9
    assert response.stage2_calls == 10
    assert response.warnings == ["list_item_fallback"]
