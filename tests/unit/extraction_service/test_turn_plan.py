"""The turn plan stage, behind `EXTRACTION_TURN_PLAN` (off by default).

With the flag on, a structured turn gets one plan call before its Stage 2
batches: the entities the turn names, one template per lead-in or heading,
and the lines that refer back. Every batch of the turn then carries the plan
in front of its own user message. These tests hold the stage to its edges:

- the plan's vocabulary is the ontology's, read from its derived lists;
- the schema stays inside what llama.cpp's grammar converter handles;
- a prose turn, and any turn with the flag off, is sent exactly as before
  (`test_turn_plan_off_pins.py` holds the flag-off literals);
- a plan call that fails costs the turn its plan and nothing else;
- the plan's text and schema move `prompt_sha256` only when the flag is on.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import re
from collections.abc import Awaitable, Callable
from types import SimpleNamespace

import httpx
import pytest
from openai import InternalServerError

from backend.errors import ExtractionValidationError, LLMConnectionError, LLMResponseError
from backend.extraction_contract.models import ExtractRequest, TurnPlanOut
from backend.extraction_service import engine as engine_module
from backend.extraction_service import turn_plan
from backend.extraction_service.adapters import get_adapter
from backend.extraction_service.app import (
    build_app,
    create_app,
    run_identity_components,
    serving_identity_components,
)
from backend.extraction_service.engine import (
    DERIVE_MAX_TOKENS,
    EXTRACT_MAX_TOKENS,
    LIST_BATCH_MAX_UNITS,
    SCOPE_MAX_TOKENS,
    ExtractionEngine,
)
from backend.extraction_service.settings import ServiceSettings
from backend.extraction_service.turn_plan import (
    EACH_ITEM,
    PLAN_BLOCK_CONSTANTS,
    PLAN_OUTPUT_SCHEMA,
    PLAN_REQUEST_TEMPLATE,
    is_structured_turn,
    parse_turn_plan,
    render_plan_block,
    render_plan_request,
)
from backend.knowledge.extraction.ontology_extractor import NO_PRIOR_CONTEXT
from backend.knowledge.extraction.serving_identity import (
    REQUEST_STAGES,
    components_dict,
    compute_serving_identity,
)
from backend.knowledge.extraction.structure_splitter import (
    InputUnit,
    batch_units,
    render_batch,
    split_structured,
)
from backend.knowledge.ontologies import EXTRACTABLE_NODE_TYPES, EXTRACTABLE_RELATIONSHIP_TYPES
from backend.knowledge.ontologies.pack import PlanVocabulary
from backend.knowledge.ontologies.registry import active_pack
from backend.llm.models import LLMRequest, LLMResponse, UsageMetadata
from tests.unit.extraction_service.conftest import make_extract_request
from tests.unit.extraction_service.fake_llama import FAKE_N_CTX
from tests.unit.extraction_service.scripted_llm import ScriptedProvider
from tests.unit.extraction_service.test_turn_plan_off_pins import (
    PROMPT_SHA256,
    PROSE_BODIES_SHA256,
    PROSE_UTTERANCE,
    SERVING_IDENTITY,
    STRUCTURED_BODIES_SHA256,
    STRUCTURED_UTTERANCE,
    T49_EXTRACT_FIELDS,
    T49_OTHER_FIELDS,
    canonical_sha256,
    split_t49,
)

_ENGINE_LOGGER = "backend.extraction_service.engine"
_FAILED_PREFIX = "turn plan failed: "

_SCOPE_REPLY = '{"scope": "user-scope", "confidence": 0.9, "reasoning": "first person"}'
_EXTRACT_REPLY = '{"entities": [], "relationships": []}'
_DERIVE_REPLY = '{"operations": []}'
_DERIVATION = {
    "signal_types": ["feedback"],
    "matched_patterns": ["feedback:stop"],
    "existing_internal_entities": "No existing internal entities.",
    "assistant_response": "Got it.",
}

# A plan with every part filled: a roster, one template with a fixed endpoint
# and a fan-out set, one template without either, and a back reference.
_PLAN: dict = {
    "entities": [
        {"id": "audio-engine-group", "name": "Audio engine group", "type": "Organization"},
        {"id": "glasswren-audio", "name": "Glasswren Audio", "type": "Organization"},
        {"id": "lisel", "name": "Lisel", "type": "Person"},
        {"id": "chip-carving", "name": "Chip carving", "type": "Skill"},
    ],
    "templates": [
        {
            "lead_in": "The audio engine group at Glasswren Audio",
            "subject": "user",
            "predicate": "MEMBER_OF",
            "fixed_endpoint": "audio-engine-group",
            "fan_out": [
                {"source": "each", "predicate": "MEMBER_OF", "target": "audio-engine-group"},
                {"source": "each", "predicate": "WORKS_AT", "target": "glasswren-audio"},
                {"source": "user", "predicate": "KNOWS_PERSON", "target": "each"},
            ],
        },
        {
            "lead_in": "Crafts I'm learning",
            "subject": "user",
            "predicate": "LEARNING",
            "fixed_endpoint": "",
            "fan_out": [],
        },
    ],
    "back_references": [{"item": "the same one as Lisel", "refers_to": "lisel"}],
}
_PLAN_JSON = json.dumps(_PLAN)
_EMPTY_PLAN: dict = {"entities": [], "templates": [], "back_references": []}

_PLAN_BLOCK = (
    "Turn plan, made from the whole message before its lines were batched. The lines of the "
    "Utterance below belong to that message. Use the plan's id and type for every entity it "
    "lists, and give each line the relationships its lead-in's template states.\n"
    "Entities named in this message (id | name | type):\n"
    "- audio-engine-group | Audio engine group | Organization\n"
    "- glasswren-audio | Glasswren Audio | Organization\n"
    "- lisel | Lisel | Person\n"
    "- chip-carving | Chip carving | Skill\n"
    "Templates, one per lead-in or heading that states a relationship "
    '("each" is the entity of each line under it):\n'
    "- The audio engine group at Glasswren Audio: user MEMBER_OF audio-engine-group; "
    "each MEMBER_OF audio-engine-group; each WORKS_AT glasswren-audio; user KNOWS_PERSON each\n"
    "- Crafts I'm learning: user LEARNING each\n"
    "Lines that refer back (line -> id):\n"
    "- the same one as Lisel -> lisel\n"
    "End of the turn plan.\n"
    "\n"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _numbered_list(count: int, lead_in: str = "Things I want to try this year:") -> str:
    return lead_in + "\n" + "\n".join(f"{n}. thing {n}" for n in range(1, count + 1))


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _is_plan_call(request: LLMRequest) -> bool:
    return request.response_schema == PLAN_OUTPUT_SCHEMA


def _is_plan_text(request: LLMRequest) -> bool:
    """The plan call told apart by its text, for a mode that sends no schema."""
    return "\nMessage, one line per unit:\n" in request.messages[1]["content"]


def _utterance(request: LLMRequest) -> str:
    """The rendered batch a Stage 2 request carries, from the first user message."""
    content = request.messages[1]["content"]
    return content.split('\nUtterance: "', 1)[1].rsplit('"\n\nOutput:', 1)[0]


def _echo(request: LLMRequest) -> LLMResponse:
    """One entity per line of the batch, so a test can read off what each call saw."""
    lines = _utterance(request).split("\n")
    body = {
        "entities": [{"id": _slug(line), "name": line, "type": "Concept"} for line in lines],
        "relationships": [],
    }
    return LLMResponse(content=json.dumps(body), finish_reason="stop")


def _plan_reply(request: LLMRequest) -> LLMResponse:
    return LLMResponse(
        content=_PLAN_JSON,
        finish_reason="stop",
        usage=UsageMetadata(prompt_tokens=13100, completion_tokens=842),
    )


Answer = Callable[[LLMRequest], LLMResponse | Awaitable[LLMResponse]]


class PlanLLM(ScriptedProvider):
    """Answers scope and derivation plainly, the plan call with `plan`, a batch with `extract`."""

    def __init__(
        self,
        plan: Answer = _plan_reply,
        extract: Answer = _echo,
        is_plan: Callable[[LLMRequest], bool] = _is_plan_call,
    ) -> None:
        self.requests: list[LLMRequest] = []
        self._plan = plan
        self._extract = extract
        self._is_plan = is_plan

    async def reply(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if request.max_tokens == SCOPE_MAX_TOKENS:
            return LLMResponse(content=_SCOPE_REPLY, finish_reason="stop")
        if request.max_tokens == DERIVE_MAX_TOKENS:
            return LLMResponse(content=_DERIVE_REPLY, finish_reason="stop")
        assert request.max_tokens == EXTRACT_MAX_TOKENS
        answer = (self._plan if self._is_plan(request) else self._extract)(request)
        if isinstance(answer, LLMResponse):
            return answer
        return await answer

    @property
    def plan_requests(self) -> list[LLMRequest]:
        return [r for r in self.requests if self._is_plan(r)]

    @property
    def batch_requests(self) -> list[LLMRequest]:
        return [
            r for r in self.requests if r.max_tokens == EXTRACT_MAX_TOKENS and not self._is_plan(r)
        ]


def _settings(**changes) -> ServiceSettings:
    values = {
        "llm_base_url": "http://unused",
        "model_hash": "test-model-hash",
        "model_file": "test-model.gguf",
        "max_attempts": 2,
        "llm_timeout_seconds": 10.0,
        "turn_plan": True,
    }
    values.update(changes)
    return ServiceSettings(**values)


def _engine(llm: PlanLLM, **changes) -> ExtractionEngine:
    return ExtractionEngine(llm=llm, adapter=get_adapter("gptoss"), settings=_settings(**changes))


def _request(utterance: str, history: list[dict] | None = None, **kwargs) -> ExtractRequest:
    body = make_extract_request(utterance=utterance, **kwargs)
    body["conversation_history"] = history or []
    return ExtractRequest.model_validate(body)


async def _flag_off_batches(utterance: str, extract: Answer = _echo) -> list[list[dict]]:
    """The messages of every Stage 2 request the same turn sends with the flag off."""
    llm = PlanLLM(extract=extract)
    await _engine(llm, turn_plan=False).run(_request(utterance))
    assert llm.plan_requests == []
    return [r.messages for r in llm.batch_requests]


def _schema_nodes(schema: dict, path: str = "plan"):
    """Every schema node of `schema`, with its path."""
    yield path, schema
    for name, child in schema.get("properties", {}).items():
        yield from _schema_nodes(child, f"{path}.{name}")
    if "items" in schema:
        yield from _schema_nodes(schema["items"], f"{path}[]")


def _plan_text_constants() -> dict[str, str]:
    """Every upper-case string constant `turn_plan` defines: the plan's whole text."""
    return {
        name: value
        for name, value in vars(turn_plan).items()
        if name.isupper() and isinstance(value, str)
    }


# ---------------------------------------------------------------------------
# The schema: the ontology's vocabulary, inside the converter's subset
# ---------------------------------------------------------------------------


class TestPlanSchema:
    def test_the_two_enums_are_the_ontologys_derived_lists(self):
        properties = PLAN_OUTPUT_SCHEMA["properties"]
        template = properties["templates"]["items"]["properties"]

        entity_types = properties["entities"]["items"]["properties"]["type"]["enum"]
        predicates = template["predicate"]["enum"]
        fan_out_predicates = template["fan_out"]["items"]["properties"]["predicate"]["enum"]

        assert entity_types == EXTRACTABLE_NODE_TYPES
        assert predicates == EXTRACTABLE_RELATIONSHIP_TYPES
        assert fan_out_predicates == EXTRACTABLE_RELATIONSHIP_TYPES

    def test_the_schema_is_built_from_the_active_packs_plan_vocabulary(self):
        (rule,) = active_pack().rules_of("plan_vocabulary")

        assert turn_plan.plan_output_schema(rule.params) == PLAN_OUTPUT_SCHEMA

    def test_a_vocabulary_gives_each_enum_its_own_list(self):
        vocabulary = PlanVocabulary(
            entity_types=("Boat", "Crew"),
            predicates=("CREWS_WITH",),
            fan_out_predicates=("SERVES_UNDER",),
        )

        properties = turn_plan.plan_output_schema(vocabulary)["properties"]
        template = properties["templates"]["items"]["properties"]

        assert properties["entities"]["items"]["properties"]["type"]["enum"] == [
            "Boat",
            "Crew",
        ]
        assert template["predicate"]["enum"] == ["CREWS_WITH"]
        assert template["fan_out"]["items"]["properties"]["predicate"]["enum"] == ["SERVES_UNDER"]

    def test_the_schema_holds_no_other_enum(self):
        enums = {path for path, node in _schema_nodes(PLAN_OUTPUT_SCHEMA) if "enum" in node}

        assert enums == {
            "plan.entities[].type",
            "plan.templates[].predicate",
            "plan.templates[].fan_out[].predicate",
        }

    def test_the_shape(self):
        properties = PLAN_OUTPUT_SCHEMA["properties"]

        assert list(properties) == ["entities", "templates", "back_references"]
        assert list(properties["entities"]["items"]["properties"]) == ["id", "name", "type"]
        assert list(properties["templates"]["items"]["properties"]) == [
            "lead_in",
            "subject",
            "predicate",
            "fixed_endpoint",
            "fan_out",
        ]
        fan_out = properties["templates"]["items"]["properties"]["fan_out"]
        assert list(fan_out["items"]["properties"]) == ["source", "predicate", "target"]
        assert list(properties["back_references"]["items"]["properties"]) == ["item", "refers_to"]

    def test_every_node_stays_inside_the_converters_subset(self):
        """Object, array, string and string enums only: no `$ref`, union, null or pattern."""
        for path, node in _schema_nodes(PLAN_OUTPUT_SCHEMA):
            kind = node["type"]
            if kind == "object":
                assert set(node) == {"type", "properties", "required", "additionalProperties"}, path
                assert node["required"] == list(node["properties"]), path
                assert node["additionalProperties"] is False, path
            elif kind == "array":
                assert set(node) == {"type", "items"}, path
            else:
                assert kind == "string", path
                assert set(node) <= {"type", "enum"}, path
                assert all(isinstance(value, str) for value in node.get("enum", [])), path

    def test_it_serialises_to_json(self):
        assert json.loads(json.dumps(PLAN_OUTPUT_SCHEMA, sort_keys=True)) == PLAN_OUTPUT_SCHEMA


class TestPlanTextNamesNoOntologyType:
    @pytest.mark.parametrize("name", sorted(_plan_text_constants()))
    def test_no_type_or_predicate_name_is_written_in_the_plan_text(self, name):
        text = _plan_text_constants()[name]

        written = [
            word
            for word in (*EXTRACTABLE_NODE_TYPES, *EXTRACTABLE_RELATIONSHIP_TYPES)
            if re.search(rf"\b{re.escape(word)}\b", text)
        ]

        assert written == [], f"{name} names {written}; read them from the ontology's lists"

    def test_the_scan_finds_a_name_that_is_written(self):
        """A rendered block does name them: the plan's own values, not the module's text."""
        assert "MEMBER_OF" in EXTRACTABLE_RELATIONSHIP_TYPES
        assert re.search(r"\bMEMBER_OF\b", _PLAN_BLOCK)


# ---------------------------------------------------------------------------
# The strict parser
# ---------------------------------------------------------------------------


def _edited(path: list, value: object) -> str:
    """`_PLAN` as JSON with the value at `path` replaced."""
    plan = json.loads(_PLAN_JSON)
    node = plan
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return json.dumps(plan)


def _without(path: list) -> str:
    plan = json.loads(_PLAN_JSON)
    node = plan
    for key in path[:-1]:
        node = node[key]
    del node[path[-1]]
    return json.dumps(plan)


class TestParseTurnPlan:
    def test_a_full_plan_parses_to_itself(self):
        assert parse_turn_plan(_PLAN_JSON) == _PLAN

    def test_an_empty_plan_parses(self):
        assert parse_turn_plan(json.dumps(_EMPTY_PLAN)) == _EMPTY_PLAN

    @pytest.mark.parametrize(
        ("raw", "where"),
        [
            pytest.param("", "not JSON", id="empty"),
            pytest.param("not json", "not JSON", id="prose"),
            pytest.param("```json\n" + _PLAN_JSON + "\n```", "not JSON", id="fenced"),
            pytest.param(_PLAN_JSON[:-20], "not JSON", id="cut-short"),
            pytest.param("[]", "plan is not an object", id="an-array"),
            pytest.param(_without(["templates"]), "plan lacks templates", id="a-missing-list"),
            pytest.param(
                _edited(["reasoning"], "because"), "plan has a field", id="an-extra-field"
            ),
            pytest.param(_edited(["entities"], {}), "plan.entities is not an array", id="a-map"),
            pytest.param(
                _edited(["entities", 0], "lisel"),
                "plan.entities[0] is not an object",
                id="an-item-that-is-not-an-object",
            ),
            pytest.param(
                _without(["entities", 2, "id"]), "plan.entities[2] lacks id", id="a-missing-id"
            ),
            pytest.param(
                _edited(["entities", 1, "name"], None),
                "plan.entities[1].name is not a string",
                id="a-null",
            ),
            pytest.param(
                _edited(["entities", 3, "type"], "Wizard"),
                "plan.entities[3].type is not one of its allowed values",
                id="a-type-outside-the-ontology",
            ),
            pytest.param(
                _edited(["templates", 0, "predicate"], "ADMIRES"),
                "plan.templates[0].predicate is not one of its allowed values",
                id="a-predicate-outside-the-ontology",
            ),
            pytest.param(
                _edited(["templates", 0, "fan_out", 1, "predicate"], ""),
                "plan.templates[0].fan_out[1].predicate is not one of its allowed values",
                id="an-empty-fan-out-predicate",
            ),
            pytest.param(
                _edited(["templates", 1, "fan_out"], ""),
                "plan.templates[1].fan_out is not an array",
                id="a-fan-out-that-is-not-a-list",
            ),
            pytest.param(
                _edited(["back_references", 0, "refers_to"], 7),
                "plan.back_references[0].refers_to is not a string",
                id="a-number",
            ),
        ],
    )
    def test_anything_else_is_rejected(self, raw, where):
        with pytest.raises(ExtractionValidationError, match=re.escape(where)):
            parse_turn_plan(raw)

    def test_a_rejection_never_quotes_the_reply(self):
        with pytest.raises(ExtractionValidationError) as caught:
            parse_turn_plan(_edited(["entities", 3, "type"], "Wizard"))

        assert "Wizard" not in str(caught.value)


# ---------------------------------------------------------------------------
# The plan request and the plan block
# ---------------------------------------------------------------------------


class TestRenderPlanRequest:
    def test_it_carries_the_context_the_scope_and_every_unit_on_its_own_line(self):
        units = split_structured(_numbered_list(9))

        text = render_plan_request(units, context="[user]: hello", subject_scope="user-scope")

        assert text == PLAN_REQUEST_TEMPLATE.format(
            context="[user]: hello", subject_scope="user-scope", units=render_batch(units)
        )
        assert text.startswith("Context:\n[user]: hello\nSubject scope: user-scope\n")
        assert render_batch(units) in text
        assert len(render_batch(units).split("\n")) == 9
        assert text.endswith("\n\nOutput:")

    def test_braces_and_quotes_in_the_turn_survive(self):
        units = (
            InputUnit(text='keep {this} and "that"', context=("Notes {a}",), kind="list_item"),
        )

        text = render_plan_request(units, context="{ctx}", subject_scope="unknown")

        assert 'Notes {a}: keep {this} and "that"' in text
        assert "Context:\n{ctx}\n" in text

    def test_the_request_names_the_user_anchor_and_the_each_token(self):
        assert '"user"' in PLAN_REQUEST_TEMPLATE
        assert f'"{EACH_ITEM}"' in PLAN_REQUEST_TEMPLATE
        for field in ("entities", "templates", "back_references", "fixed_endpoint", "fan_out"):
            assert f'"{field}"' in PLAN_REQUEST_TEMPLATE

    def test_a_heading_that_states_no_relationship_gets_no_template(self):
        """Every template carries a predicate, so a heading without one must be left out."""
        assert (
            "one for each heading or lead-in that gives the lines under it a relationship"
            in PLAN_REQUEST_TEMPLATE
        )
        assert "only names or groups its lines gets no template" in PLAN_REQUEST_TEMPLATE


class TestRenderPlanBlock:
    def test_a_full_plan_renders_as_the_pinned_block(self):
        assert render_plan_block(_PLAN) == _PLAN_BLOCK

    def test_an_empty_plan_keeps_its_three_headings(self):
        block = render_plan_block(_EMPTY_PLAN)

        assert block.count("\n(none)\n") == 3
        assert block.endswith("End of the turn plan.\n\n")

    def test_a_value_with_line_breaks_stays_on_one_line(self):
        plan = json.loads(_PLAN_JSON)
        plan["entities"][0]["name"] = "Audio\nengine\t group\n\nContext:"
        plan["back_references"][0]["item"] = "the same\none"

        block = render_plan_block(plan)

        assert "- audio-engine-group | Audio engine group Context: | Organization\n" in block
        assert "- the same one -> lisel\n" in block
        assert len(block.split("\n")) == len(_PLAN_BLOCK.split("\n"))

    def test_the_block_is_built_from_the_hashed_constants_alone(self):
        """Every character of an empty plan's block comes from `PLAN_BLOCK_CONSTANTS`."""
        rest = render_plan_block(_EMPTY_PLAN)
        for constant in sorted(PLAN_BLOCK_CONSTANTS, key=len, reverse=True):
            if "{" not in constant:
                rest = rest.replace(constant, "")

        assert rest.strip("\n") == ""

    def test_every_plan_string_is_hashed(self):
        """A string added to `turn_plan` and left out of the hash fails here."""
        assert set(_plan_text_constants().values()) == {
            PLAN_REQUEST_TEMPLATE,
            *PLAN_BLOCK_CONSTANTS,
        }
        assert len(set(PLAN_BLOCK_CONSTANTS)) == len(PLAN_BLOCK_CONSTANTS)


class TestIsStructuredTurn:
    @pytest.mark.parametrize(
        "text",
        [
            "I have been learning Rust for a few months",
            "",
            "Two paragraphs.\n\nBut no list, table or heading.",
            "```\n- inside a fence\n```",
        ],
    )
    def test_text_without_structure_is_not(self, text):
        assert not is_structured_turn(split_structured(text))

    @pytest.mark.parametrize(
        "text",
        [
            "Groceries:\n- eggs",
            "- one item",
            "# Heading\nA paragraph under it.",
            "| a | b |\n|---|---|\n| 1 | 2 |",
        ],
    )
    def test_a_list_a_heading_or_a_table_is(self, text):
        assert is_structured_turn(split_structured(text))


# ---------------------------------------------------------------------------
# Flag on, structured turn
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAStructuredTurnIsPlanned:
    async def test_exactly_one_plan_call_goes_out_before_the_first_batch(self):
        llm = PlanLLM()

        await _engine(llm).run(_request(_numbered_list(9)))

        extract_cap = [r for r in llm.requests if r.max_tokens == EXTRACT_MAX_TOKENS]
        assert len(llm.plan_requests) == 1
        assert _is_plan_call(extract_cap[0])
        assert not any(_is_plan_call(r) for r in extract_cap[1:])
        assert len(llm.batch_requests) == 3
        assert llm.requests[0].max_tokens == SCOPE_MAX_TOKENS

    async def test_the_plan_call_carries_the_whole_turn_under_the_batches_system_prompt(self):
        text = _numbered_list(9)
        units = split_structured(text)
        history = [{"role": "assistant", "content": "What are you planning?"}]
        llm = PlanLLM()

        await _engine(llm).run(_request(text, history))

        (plan,) = llm.plan_requests
        system, user = plan.messages
        assert system["role"] == "system"
        assert {r.messages[0]["content"] for r in llm.batch_requests} == {system["content"]}
        assert "2026-04-21" in system["content"]
        assert user == {
            "role": "user",
            "content": render_plan_request(
                units,
                context="[assistant]: What are you planning?",
                subject_scope="user-scope",
            ),
        }

    async def test_a_turn_without_history_shows_the_empty_context_placeholder(self):
        llm = PlanLLM()

        await _engine(llm).run(_request(_numbered_list(3)))

        (plan,) = llm.plan_requests
        assert plan.messages[1]["content"].startswith(f"Context:\n{NO_PRIOR_CONTEXT}\n")

    async def test_the_plan_call_is_an_extract_stage_request_with_the_plan_schema(self):
        llm = PlanLLM()

        await _engine(llm).run(_request(_numbered_list(3)))

        (plan,) = llm.plan_requests
        (batch,) = llm.batch_requests
        assert plan.response_schema == PLAN_OUTPUT_SCHEMA
        assert plan.max_tokens == batch.max_tokens == EXTRACT_MAX_TOKENS
        assert plan.thinking == batch.thinking
        assert (plan.temperature, plan.top_p, plan.json_mode) == (
            batch.temperature,
            batch.top_p,
            batch.json_mode,
        )

    async def test_every_batch_carries_the_plan_block_before_its_flag_off_message(self):
        text = _numbered_list(30)
        flag_off = await _flag_off_batches(text)
        llm = PlanLLM()

        await _engine(llm).run(_request(text))

        assert len(llm.batch_requests) == len(flag_off) == 8
        for request, plain in zip(llm.batch_requests, flag_off, strict=True):
            system, user = request.messages
            assert system == plain[0]
            assert user["role"] == "user"
            assert user["content"] == _PLAN_BLOCK + plain[1]["content"]

    async def test_the_batches_split_after_a_length_stop_carry_it_too(self):
        text = _numbered_list(5)

        def cut_short(request: LLMRequest) -> LLMResponse:
            if "\n" in _utterance(request):
                return LLMResponse(content='{"entities": [{"id": "u', finish_reason="length")
            return _echo(request)

        flag_off = await _flag_off_batches(text, extract=cut_short)
        llm = PlanLLM(extract=cut_short)

        response = await _engine(llm).run(_request(text))

        sizes = [_utterance(r).count("\n") + 1 for r in llm.batch_requests]
        assert sizes == [4, 2, 1, 1, 2, 1, 1, 1]
        assert len(llm.plan_requests) == 1
        assert [r.messages[1]["content"] for r in llm.batch_requests] == [
            _PLAN_BLOCK + plain[1]["content"] for plain in flag_off
        ]
        assert response.stage2_calls == 8

    async def test_a_repair_retry_carries_it_too(self):
        calls = 0

        def unparsable_once(request: LLMRequest) -> LLMResponse:
            nonlocal calls
            calls += 1
            if calls == 1:
                return LLMResponse(content="not json", finish_reason="stop")
            return _echo(request)

        llm = PlanLLM(extract=unparsable_once)

        response = await _engine(llm).run(_request(_numbered_list(3)))

        first, repair = llm.batch_requests
        assert len(repair.messages) == 4
        assert repair.messages[1] == first.messages[1]
        assert repair.messages[1]["content"].startswith(_PLAN_BLOCK)
        assert len(llm.plan_requests) == 1
        assert (response.attempts, response.stage2_calls) == (2, 2)

    async def test_the_response_keeps_the_plan_call(self):
        llm = PlanLLM()

        response = await _engine(llm).run(_request(_numbered_list(9)))

        assert response.turn_plan == TurnPlanOut(
            payload=_PLAN, prompt_tokens=13100, completion_tokens=842, finish_reason="stop"
        )
        assert response.warnings == []
        assert response.model_dump(mode="json")["turn_plan"] == {
            "payload": _PLAN,
            "prompt_tokens": 13100,
            "completion_tokens": 842,
            "finish_reason": "stop",
        }

    async def test_the_plan_call_is_not_a_stage2_call_or_an_attempt(self):
        text = _numbered_list(9)
        off = PlanLLM()
        expected = await _engine(off, turn_plan=False).run(_request(text))
        llm = PlanLLM()

        response = await _engine(llm).run(_request(text))

        assert (response.stage2_calls, response.attempts) == (3, 1)
        assert (expected.stage2_calls, expected.attempts) == (3, 1)
        assert response.payload == expected.payload

    async def test_the_plan_calls_time_is_in_the_total_and_not_in_extract(self, monkeypatch):
        """A fake clock: the plan call takes 5 s and each of the three batches 1 s."""
        clock = [100.0]
        monkeypatch.setattr(engine_module, "time", SimpleNamespace(perf_counter=lambda: clock[0]))

        def plan(request: LLMRequest) -> LLMResponse:
            clock[0] += 5.0
            return _plan_reply(request)

        def extract(request: LLMRequest) -> LLMResponse:
            clock[0] += 1.0
            return _echo(request)

        response = await _engine(PlanLLM(plan, extract)).run(_request(_numbered_list(9)))

        assert response.timings_ms.extract == pytest.approx(3000.0)
        assert response.timings_ms.total == pytest.approx(8000.0)

    async def test_the_plan_call_is_logged_like_every_other_call(self, caplog):
        caplog.set_level(logging.INFO, logger=_ENGINE_LOGGER)

        await _engine(PlanLLM()).run(_request(_numbered_list(9)))

        lines = [r.getMessage() for r in caplog.records if r.name == _ENGINE_LOGGER]
        calls = [line for line in lines if line.startswith("extraction call:")]
        # Scope, the plan call, three batches.
        assert len(calls) == 5
        assert calls[1] == (
            f"extraction call: cap {EXTRACT_MAX_TOKENS} prompt_tokens 13100 "
            "completion_tokens 842 finish stop"
        )
        assert not any(line.startswith(_FAILED_PREFIX) for line in lines)

    async def test_the_stages_and_the_request_profile_stay_three(self):
        on = _engine(PlanLLM())
        off = _engine(PlanLLM(), turn_plan=False)

        assert REQUEST_STAGES == ("scope", "extract", "derive")
        assert on.request_profile() == off.request_profile()
        assert tuple(p.stage for p in on.request_profile()) == REQUEST_STAGES

    @pytest.mark.parametrize("mode", ["json_object", "none"])
    async def test_a_mode_that_sends_no_schema_still_plans(self, mode):
        """Without the grammar, the strict parser alone holds the reply to the schema."""
        llm = PlanLLM(is_plan=_is_plan_text)

        response = await _engine(llm, constrained_mode=mode).run(_request(_numbered_list(3)))

        (plan,) = llm.plan_requests
        (batch,) = llm.batch_requests
        assert (plan.response_schema, batch.response_schema) == (None, None)
        assert plan.json_mode is batch.json_mode is (mode == "json_object")
        assert batch.messages[1]["content"].startswith(_PLAN_BLOCK)
        assert response.turn_plan is not None
        assert response.turn_plan.payload == _PLAN
        assert response.warnings == []

    async def test_the_schema_and_the_text_mark_the_same_call(self):
        llm = PlanLLM()

        await _engine(llm).run(_request(_numbered_list(9)))

        assert [_is_plan_text(r) for r in llm.requests] == [_is_plan_call(r) for r in llm.requests]


# ---------------------------------------------------------------------------
# Flag on, prose turn; flag off
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestATurnThatIsNotPlanned:
    @pytest.mark.parametrize(
        "utterance",
        [PROSE_UTTERANCE, "Two paragraphs.\n\nBut no list, table or heading."],
    )
    async def test_a_prose_turn_makes_no_plan_call_and_sends_the_flag_off_messages(self, utterance):
        flag_off = await _flag_off_batches(utterance)
        llm = PlanLLM()

        response = await _engine(llm).run(_request(utterance))

        assert llm.plan_requests == []
        assert [r.messages for r in llm.batch_requests] == flag_off
        assert len(flag_off) == 1
        assert response.turn_plan is None
        assert "turn_plan" not in response.model_dump(mode="json")
        assert '"turn_plan"' not in response.model_dump_json()

    async def test_the_flag_off_makes_no_plan_call_on_a_structured_turn(self):
        llm = PlanLLM()

        response = await _engine(llm, turn_plan=False).run(_request(_numbered_list(9)))

        assert llm.plan_requests == []
        assert len(llm.batch_requests) == 3
        assert not any(_PLAN_BLOCK[:40] in r.messages[1]["content"] for r in llm.batch_requests)
        assert response.turn_plan is None
        assert "turn_plan" not in response.model_dump(mode="json")

    async def test_the_flag_is_off_by_default(self):
        settings = ServiceSettings(llm_base_url="http://unused", model_hash="m", model_file="f")

        assert settings.turn_plan is False


# ---------------------------------------------------------------------------
# A plan call that fails
# ---------------------------------------------------------------------------


def _raises(error: Exception) -> Answer:
    def answer(request: LLMRequest) -> LLMResponse:
        raise error

    return answer


def _upstream_500(body_text: str) -> InternalServerError:
    """An openai status error as the client raises it, its message carrying the body."""
    response = httpx.Response(
        500, request=httpx.Request("POST", "http://llama/v1/chat/completions")
    )
    return InternalServerError(f"Error code: 500 - {body_text}", response=response, body=None)


async def _never_answers(request: LLMRequest) -> LLMResponse:
    await asyncio.Event().wait()
    raise AssertionError("unreachable")


def _cut_short_plan(request: LLMRequest) -> LLMResponse:
    return LLMResponse(
        content=_PLAN_JSON[:200],
        finish_reason="length",
        usage=UsageMetadata(prompt_tokens=13100, completion_tokens=EXTRACT_MAX_TOKENS),
    )


def _replies(content: str) -> Answer:
    def answer(request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            content=content,
            finish_reason="stop",
            usage=UsageMetadata(prompt_tokens=13100, completion_tokens=12),
        )

    return answer


_NOT_RETURNED = TurnPlanOut(payload=None)
_FAILURES = [
    pytest.param(
        _raises(LLMConnectionError("connection refused")),
        {},
        "LLMConnectionError",
        _NOT_RETURNED,
        id="a-transport-error",
    ),
    pytest.param(
        _raises(LLMResponseError("HTTP 500")),
        {},
        "LLMResponseError",
        _NOT_RETURNED,
        id="a-provider-error",
    ),
    pytest.param(
        _raises(_upstream_500("Failed to parse input: Orrin Vale leads the rota")),
        {},
        "InternalServerError (HTTP 500)",
        _NOT_RETURNED,
        id="an-upstream-error-whose-body-quotes-text",
    ),
    pytest.param(
        _never_answers,
        {"llm_timeout_seconds": 0.05},
        "timed out after 0.05s",
        _NOT_RETURNED,
        id="a-timeout",
    ),
    pytest.param(
        _cut_short_plan,
        {},
        f"output stopped at max_tokens ({EXTRACT_MAX_TOKENS})",
        TurnPlanOut(
            payload=None,
            prompt_tokens=13100,
            completion_tokens=EXTRACT_MAX_TOKENS,
            finish_reason="length",
        ),
        id="a-length-stop",
    ),
    pytest.param(
        _replies("I cannot plan this."),
        {},
        "plan is not JSON: Expecting value: line 1 column 1 (char 0)",
        TurnPlanOut(payload=None, prompt_tokens=13100, completion_tokens=12, finish_reason="stop"),
        id="a-reply-that-is-not-json",
    ),
    pytest.param(
        _replies(_edited(["entities", 3, "type"], "Wizard")),
        {},
        "plan.entities[3].type is not one of its allowed values",
        TurnPlanOut(payload=None, prompt_tokens=13100, completion_tokens=12, finish_reason="stop"),
        id="a-plan-the-parser-rejects",
    ),
    pytest.param(
        _replies(_EXTRACT_REPLY),
        {},
        "plan lacks back_references, templates",
        TurnPlanOut(payload=None, prompt_tokens=13100, completion_tokens=12, finish_reason="stop"),
        id="an-extraction-payload-in-place-of-a-plan",
    ),
]


@pytest.mark.asyncio
class TestAPlanCallThatFails:
    @pytest.mark.parametrize(("plan", "changes", "reason", "record"), _FAILURES)
    async def test_the_batches_run_without_a_plan_and_the_turn_succeeds(
        self, caplog, plan, changes, reason, record
    ):
        text = _numbered_list(9)
        flag_off = await _flag_off_batches(text)
        expected = await _engine(PlanLLM(), turn_plan=False).run(_request(text))
        caplog.clear()
        caplog.set_level(logging.INFO, logger=_ENGINE_LOGGER)
        llm = PlanLLM(plan=plan)

        response = await _engine(llm, **changes).run(_request(text))

        assert len(llm.plan_requests) == 1
        assert [r.messages for r in llm.batch_requests] == flag_off
        assert response.outcome == "extracted"
        assert response.payload == expected.payload
        assert (response.stage2_calls, response.attempts) == (3, 1)
        assert response.warnings == ["turn_plan_failed"]
        assert response.turn_plan == record
        failed = [
            r
            for r in caplog.records
            if r.name == _ENGINE_LOGGER and r.getMessage().startswith(_FAILED_PREFIX)
        ]
        assert [(r.levelno, r.getMessage()) for r in failed] == [
            (logging.WARNING, _FAILED_PREFIX + reason)
        ]

    async def test_a_failed_plan_is_serialized_with_a_null_payload(self):
        llm = PlanLLM(plan=_replies("not json"))

        response = await _engine(llm).run(_request(_numbered_list(3)))

        assert response.model_dump(mode="json")["turn_plan"] == {
            "payload": None,
            "prompt_tokens": 13100,
            "completion_tokens": 12,
            "finish_reason": "stop",
        }

    async def test_a_length_stop_is_not_retried_or_split(self):
        llm = PlanLLM(plan=_cut_short_plan)

        await _engine(llm).run(_request(_numbered_list(9)))

        assert len(llm.plan_requests) == 1
        assert all(len(r.messages) == 2 for r in llm.requests)

    async def test_an_unparsable_plan_gets_no_repair_retry(self):
        llm = PlanLLM(plan=_replies("not json"))

        await _engine(llm, max_attempts=3).run(_request(_numbered_list(9)))

        assert len(llm.plan_requests) == 1
        assert all(len(r.messages) == 2 for r in llm.requests)

    async def test_the_log_line_never_quotes_the_turn_or_the_reply(self, caplog):
        caplog.set_level(logging.INFO, logger=_ENGINE_LOGGER)
        llm = PlanLLM(plan=_replies(_edited(["entities", 2, "name"], 7)))

        await _engine(llm).run(_request("Marmot sanctuary rota:\n- Orrin Vale\n- Lisel"))

        logged = "\n".join(r.getMessage() for r in caplog.records if r.name == _ENGINE_LOGGER)
        assert _FAILED_PREFIX + "plan.entities[2].name is not a string" in logged
        for text in ("Orrin", "Lisel", "Marmot", "glasswren"):
            assert text not in logged


# ---------------------------------------------------------------------------
# prompt_sha256
# ---------------------------------------------------------------------------


def _hash(*, on: bool) -> str:
    return engine_module._compute_prompt_sha256(get_adapter("gptoss"), turn_plan=on)


class TestPromptSha256:
    def test_off_is_the_default_and_the_pinned_hash(self):
        assert _hash(on=False) == PROMPT_SHA256
        assert engine_module._compute_prompt_sha256(get_adapter("gptoss")) == PROMPT_SHA256

    def test_on_moves_the_hash(self):
        assert _hash(on=True) != _hash(on=False)
        assert re.fullmatch(r"[0-9a-f]{64}", _hash(on=True))

    def test_an_edit_to_the_plan_request_moves_the_on_hash_alone(self, monkeypatch):
        on, off = _hash(on=True), _hash(on=False)
        monkeypatch.setattr(
            engine_module, "PLAN_REQUEST_TEMPLATE", PLAN_REQUEST_TEMPLATE + " (edited)"
        )

        assert _hash(on=True) != on
        assert _hash(on=False) == off

    def test_an_edit_to_the_plan_schema_moves_the_on_hash_alone(self, monkeypatch):
        on, off = _hash(on=True), _hash(on=False)
        monkeypatch.setattr(
            engine_module, "PLAN_OUTPUT_SCHEMA", {**PLAN_OUTPUT_SCHEMA, "description": "edited"}
        )

        assert _hash(on=True) != on
        assert _hash(on=False) == off

    def test_a_change_to_the_ontologys_lists_would_move_the_on_hash(self, monkeypatch):
        """The lists reach the hash through the schema's enums."""
        on = _hash(on=True)
        edited = json.loads(json.dumps(PLAN_OUTPUT_SCHEMA))
        edited["properties"]["entities"]["items"]["properties"]["type"]["enum"].append("Wizard")
        monkeypatch.setattr(engine_module, "PLAN_OUTPUT_SCHEMA", edited)

        assert _hash(on=True) != on

    @pytest.mark.parametrize("index", range(len(PLAN_BLOCK_CONSTANTS)))
    def test_an_edit_to_each_plan_block_string_moves_the_on_hash_alone(self, monkeypatch, index):
        on, off = _hash(on=True), _hash(on=False)
        edited = tuple(
            value + " (edited)" if i == index else value
            for i, value in enumerate(PLAN_BLOCK_CONSTANTS)
        )
        monkeypatch.setattr(engine_module, "PLAN_BLOCK_CONSTANTS", edited)

        assert _hash(on=True) != on
        assert _hash(on=False) == off

    def test_text_moved_between_two_plan_strings_moves_the_on_hash(self, monkeypatch):
        on = _hash(on=True)
        first, second, *rest = PLAN_BLOCK_CONSTANTS
        monkeypatch.setattr(
            engine_module, "PLAN_BLOCK_CONSTANTS", (first + second[:1], second[1:], *rest)
        )

        assert _hash(on=True) != on

    @pytest.mark.asyncio
    @pytest.mark.parametrize("on", [True, False])
    async def test_every_result_is_stamped_with_the_hash_of_its_flag(self, on):
        response = await _engine(PlanLLM(), turn_plan=on).run(_request(_numbered_list(3)))

        assert response.stamps.prompt_sha256 == _hash(on=on)


# ---------------------------------------------------------------------------
# Through the service, against the fake llama-server
# ---------------------------------------------------------------------------


def _client_for(application) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://service"
    )


@pytest.fixture
def plan_settings(service_settings: ServiceSettings) -> ServiceSettings:
    return dataclasses.replace(service_settings, turn_plan=True)


@pytest.fixture
def plan_app(plan_settings, wired_llm, health_probe, ctx_size_source):
    """The service with the flag on, wired as the `service_app` fixture is."""
    adapter = get_adapter(
        plan_settings.adapter_name,
        reasoning_effort=plan_settings.reasoning_effort,
        reasoning_budget_tokens=plan_settings.reasoning_budget_tokens,
    )
    engine = ExtractionEngine(llm=wired_llm, adapter=adapter, settings=plan_settings)
    return create_app(plan_settings, engine, health_probe, ctx_size_source)


def _is_plan_body(body: dict) -> bool:
    schema = (body.get("response_format") or {}).get("json_schema", {}).get("schema")
    return schema == PLAN_OUTPUT_SCHEMA


@pytest.mark.asyncio
class TestTheServiceWithTheFlagOn:
    async def test_a_structured_turn_sends_the_plan_call_and_reports_it(
        self, plan_app, fake_llama_state
    ):
        fake_llama_state.chat_responses = [
            _SCOPE_REPLY,
            _PLAN_JSON,
            _EXTRACT_REPLY,
            _EXTRACT_REPLY,
            _DERIVE_REPLY,
        ]

        async with _client_for(plan_app) as client:
            response = await client.post(
                "/v1/extract",
                json=make_extract_request(utterance=STRUCTURED_UTTERANCE, derivation=_DERIVATION),
            )

        assert response.status_code == 200, response.text
        bodies = fake_llama_state.chat_requests
        assert [_is_plan_body(body) for body in bodies] == [False, True, False, False, False]
        assert [body["max_tokens"] for body in bodies] == [
            SCOPE_MAX_TOKENS,
            EXTRACT_MAX_TOKENS,
            EXTRACT_MAX_TOKENS,
            EXTRACT_MAX_TOKENS,
            DERIVE_MAX_TOKENS,
        ]
        # The plan call is sent as stage extract: budgeted and streamed, as a batch is.
        assert split_t49(bodies)[1] == [
            T49_OTHER_FIELDS,
            T49_EXTRACT_FIELDS,
            T49_EXTRACT_FIELDS,
            T49_EXTRACT_FIELDS,
            T49_OTHER_FIELDS,
        ]
        for batch in bodies[2:4]:
            assert batch["messages"][0] == bodies[1]["messages"][0]
            assert batch["messages"][1]["content"].startswith(_PLAN_BLOCK + "Context:\n")
        data = response.json()
        assert data["turn_plan"] == {
            "payload": _PLAN,
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "finish_reason": "stop",
        }
        assert list(data)[-2:] == ["turn_plan", "batch_record"]
        assert data["stage2_calls"] == 2
        assert data["warnings"] == []
        assert data["stamps"]["prompt_sha256"] == _hash(on=True)

    async def test_a_prose_turn_sends_the_flag_off_bodies_and_reports_no_plan(
        self, plan_app, fake_llama_state
    ):
        fake_llama_state.chat_responses = [_SCOPE_REPLY, _EXTRACT_REPLY, _DERIVE_REPLY]

        async with _client_for(plan_app) as client:
            response = await client.post(
                "/v1/extract",
                json=make_extract_request(utterance=PROSE_UTTERANCE, derivation=_DERIVATION),
            )

        assert response.status_code == 200, response.text
        before_t49, _set_by_t49 = split_t49(fake_llama_state.chat_requests)
        assert canonical_sha256(before_t49) == PROSE_BODIES_SHA256
        assert "turn_plan" not in response.json()

    async def test_a_failed_plan_leaves_the_flag_off_bodies_around_the_plan_call(
        self, plan_app, fake_llama_state
    ):
        fake_llama_state.chat_responses = [
            _SCOPE_REPLY,
            "not a plan",
            _EXTRACT_REPLY,
            _EXTRACT_REPLY,
            _DERIVE_REPLY,
        ]

        async with _client_for(plan_app) as client:
            response = await client.post(
                "/v1/extract",
                json=make_extract_request(utterance=STRUCTURED_UTTERANCE, derivation=_DERIVATION),
            )

        assert response.status_code == 200, response.text
        bodies = fake_llama_state.chat_requests
        assert [_is_plan_body(body) for body in bodies] == [False, True, False, False, False]
        before_t49, _set_by_t49 = split_t49([bodies[0], *bodies[2:]])
        assert canonical_sha256(before_t49) == STRUCTURED_BODIES_SHA256
        data = response.json()
        assert data["warnings"] == ["turn_plan_failed"]
        assert data["turn_plan"]["payload"] is None

    async def test_a_replayed_job_keeps_its_plan(self, plan_app, fake_llama_state):
        fake_llama_state.chat_responses = [_SCOPE_REPLY, _PLAN_JSON, _EXTRACT_REPLY]
        body = make_extract_request(utterance=STRUCTURED_UTTERANCE)

        async with _client_for(plan_app) as client:
            first = (await client.post("/v1/extract", json=body)).json()
            calls = len(fake_llama_state.chat_requests)
            second = (await client.post("/v1/extract", json=body)).json()

        assert second == first
        assert second["turn_plan"]["payload"] == _PLAN
        assert len(fake_llama_state.chat_requests) == calls

    async def test_the_flag_on_changes_no_batch_grouping(self, plan_app, fake_llama_state):
        fake_llama_state.chat_responses = [_SCOPE_REPLY, _PLAN_JSON, _EXTRACT_REPLY]
        text = _numbered_list(9)

        async with _client_for(plan_app) as client:
            response = await client.post("/v1/extract", json=make_extract_request(utterance=text))

        batches = batch_units(split_structured(text), LIST_BATCH_MAX_UNITS)
        sent = [
            body["messages"][1]["content"]
            for body in fake_llama_state.chat_requests
            if body["max_tokens"] == EXTRACT_MAX_TOKENS and not _is_plan_body(body)
        ]
        assert response.json()["stage2_calls"] == len(batches) == 3
        for content, batch in zip(sent, batches, strict=True):
            assert f'\nUtterance: "{render_batch(batch)}"\n\nOutput:' in content


# ---------------------------------------------------------------------------
# /v1/info: the service passes the flag to both identities
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestInfoReportsTheFlag:
    async def test_info_reports_the_flag_in_the_serving_identity_components(self, plan_app):
        async with _client_for(plan_app) as client:
            data = (await client.get("/v1/info")).json()

        assert data["serving_identity_components"]["turn_plan"] is True
        assert data["serving_identity_error"] is None

    async def test_info_reports_no_flag_when_it_is_off(self, client):
        data = (await client.get("/v1/info")).json()

        assert "turn_plan" not in data["serving_identity_components"]

    async def test_info_reports_the_flag_on_prompt_hash_in_the_run_identity(
        self, plan_settings, wired_llm, fake_llama_app
    ):
        http_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=fake_llama_app))
        application = build_app(plan_settings, wired_llm, http_client)

        async with _client_for(application) as client:
            data = (await client.get("/v1/info")).json()

        assert data["run_identity_components"]["prompt_sha256"] == _hash(on=True)

    async def test_info_reports_the_pinned_prompt_hash_when_the_flag_is_off(
        self, service_settings, wired_llm, fake_llama_app
    ):
        http_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=fake_llama_app))
        application = build_app(service_settings, wired_llm, http_client)

        async with _client_for(application) as client:
            data = (await client.get("/v1/info")).json()

        assert data["run_identity_components"]["prompt_sha256"] == PROMPT_SHA256


class TestTheFlagInBothIdentities:
    """The flag is one component of each identity, and with it off neither one moves."""

    def test_the_run_identity_differs_in_its_prompt_hash_alone(self, service_settings):
        adapter = get_adapter(service_settings.adapter_name)
        off = dataclasses.asdict(run_identity_components(service_settings, adapter))
        on = dataclasses.asdict(
            run_identity_components(dataclasses.replace(service_settings, turn_plan=True), adapter)
        )

        assert off["prompt_sha256"] == PROMPT_SHA256
        assert on["prompt_sha256"] == _hash(on=True)
        assert [name for name in off if off[name] != on[name]] == ["prompt_sha256"]

    def test_the_serving_identity_gains_the_flag_and_changes_no_other_component(
        self, service_settings, engine
    ):
        off = serving_identity_components(service_settings, engine, FAKE_N_CTX)
        on = serving_identity_components(
            dataclasses.replace(service_settings, turn_plan=True), engine, FAKE_N_CTX
        )

        assert compute_serving_identity(off) == SERVING_IDENTITY
        assert components_dict(on) == {**components_dict(off), "turn_plan": True}
        assert compute_serving_identity(on) != SERVING_IDENTITY
