"""Tests for the constrained action decoder: prompt, schema, parse and repair."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator, Mapping
from typing import Literal

import pytest
from pydantic import BaseModel, Field

from backend.capabilities import decoder as decoder_module
from backend.capabilities.contracts import ActionDecode, DecodeReply, ToolSpec
from backend.capabilities.decoder import DECODE_INSTRUCTIONS, ActionDecoder

# The decoder must stay inside the keyword set the grammar schema builder allows, so the test
# reads that set from its one definition: a keyword added to the decoder alone fails here.
from backend.capabilities.schema import _ALLOWED_KEYS, to_args_schema, to_catalog_line
from backend.errors import ActionDecodeError
from tests.unit.capabilities.fakes import EmptyArgs, make_tool_spec


class LightsPowerArgs(BaseModel):
    target: Literal["kitchen", "bedroom"] | None = None
    on: bool


class AlarmsSetArgs(BaseModel):
    time: str
    label: str | None = None


class LightsAdjustArgs(BaseModel):
    target: str | None = None
    brightness_pct: int | None = Field(default=None, ge=0, le=100)


LIGHTS_POWER = make_tool_spec(
    "lights.power", LightsPowerArgs, description="turn lights on or off", risk="physical"
)
ALARMS_SET = make_tool_spec("alarms.set", AlarmsSetArgs, description="set an alarm")
CLOCK_NOW = make_tool_spec("clock.now", EmptyArgs, description="tell the current time")
LIGHTS_ADJUST = make_tool_spec(
    "lights.adjust", LightsAdjustArgs, description="change brightness", risk="physical"
)


def offer(*specs: ToolSpec) -> list[tuple[ToolSpec, dict[str, object]]]:
    return [(spec, to_args_schema(spec)) for spec in specs]


def reply(payload: object, finish_reason: str | None = "stop") -> DecodeReply:
    return DecodeReply(content=json.dumps(payload), finish_reason=finish_reason)


def dump(decode: ActionDecode) -> str:
    """Serialise every field of a decode, so equal strings mean byte-identical requests."""
    return json.dumps(
        {
            "messages": list(decode.messages),
            "response_schema": decode.response_schema,
            "max_tokens": decode.max_tokens,
            "temperature": decode.temperature,
        }
    )


def schema_keywords(node: object) -> Iterator[str]:
    """Yield every keyword in a schema, skipping property names, which are not keywords."""
    if isinstance(node, Mapping):
        for key, value in node.items():
            yield key
            if key == "properties" and isinstance(value, Mapping):
                for prop in value.values():
                    yield from schema_keywords(prop)
            else:
                yield from schema_keywords(value)
    elif isinstance(node, list):
        for item in node:
            yield from schema_keywords(item)


@pytest.fixture
def decoder() -> ActionDecoder:
    return ActionDecoder()


# -- build: the prompt ---------------------------------------------------------------------


class TestBuildMessages:
    def test_one_system_message_with_instructions_then_catalog_in_name_order(self, decoder):
        decode = decoder.build(offer(LIGHTS_POWER, CLOCK_NOW, ALARMS_SET), [], "wake me at 7")

        system = [m for m in decode.messages if m["role"] == "system"]
        assert len(system) == 1
        assert decode.messages[0] is system[0]
        lines = [
            to_catalog_line(spec, to_args_schema(spec))
            for spec in (ALARMS_SET, CLOCK_NOW, LIGHTS_POWER)
        ]
        assert system[0]["content"] == DECODE_INSTRUCTIONS + "\n" + "\n".join(lines)

    @pytest.mark.parametrize(
        "clause",
        [
            "Choose only actions listed below.",
            "For a time or date, pass the user's own words.",
            "Output a call only for an action the user asked for.",
            "Output no call when the user is only talking about these things.",
            "Set also_reply to true when the user also asked or said something that needs an"
            " answer beyond the actions",
            "and set it to true when unsure.",
        ],
        ids=[
            "only-listed",
            "own-words-for-times",
            "call-only-when-asked",
            "no-call-when-talking",
            "also-reply-beyond-actions",
            "also-reply-when-unsure",
        ],
    )
    def test_instructions_carry_each_design_rule(self, clause):
        assert clause in DECODE_INSTRUCTIONS.replace("\n", " ")

    def test_keeps_the_last_four_user_and_assistant_messages(self, decoder):
        recent = [
            ("user", "first"),
            ("assistant", "second"),
            ("system", "never shown"),
            ("user", "third"),
            ("tool", "never shown either"),
            ("assistant", "fourth"),
            ("user", "fifth"),
            ("assistant", "sixth"),
        ]
        decode = decoder.build(offer(CLOCK_NOW), recent, "what time is it")

        assert list(decode.messages[1:]) == [
            {"role": "user", "content": "third"},
            {"role": "assistant", "content": "fourth"},
            {"role": "user", "content": "fifth"},
            {"role": "assistant", "content": "sixth"},
            {"role": "user", "content": "what time is it"},
        ]

    def test_caps_each_recent_message_at_300_characters(self, decoder):
        long_text = "x" * 299 + "yz" + "w" * 50
        decode = decoder.build(offer(CLOCK_NOW), [("assistant", long_text)], "and now")

        assert decode.messages[1] == {"role": "assistant", "content": "x" * 299 + "y"}

    def test_utterance_is_last_and_whole(self, decoder):
        utterance = "turn on the kitchen lights " * 20
        decode = decoder.build(offer(LIGHTS_POWER), [("user", "hi")], utterance)

        assert decode.messages[-1] == {"role": "user", "content": utterance}

    def test_no_recent_messages_gives_system_then_utterance(self, decoder):
        decode = decoder.build(offer(CLOCK_NOW), [], "time please")

        assert [m["role"] for m in decode.messages] == ["system", "user"]

    def test_nothing_offered_is_refused(self, decoder):
        with pytest.raises(ValueError, match="at least one"):
            decoder.build([], [], "hello")


# -- build: the schema ---------------------------------------------------------------------


class TestBuildSchema:
    def test_schema_matches_the_design(self, decoder):
        offered = offer(LIGHTS_POWER, CLOCK_NOW)
        decode = decoder.build(offered, [], "lights on")

        assert decode.response_schema == {
            "type": "object",
            "properties": {
                "calls": {
                    "type": "array",
                    "items": {
                        "anyOf": [
                            {
                                "type": "object",
                                "properties": {
                                    "tool": {"const": "clock.now"},
                                    "args": to_args_schema(CLOCK_NOW),
                                },
                                "required": ["tool", "args"],
                                "additionalProperties": False,
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "tool": {"const": "lights.power"},
                                    "args": to_args_schema(LIGHTS_POWER),
                                },
                                "required": ["tool", "args"],
                                "additionalProperties": False,
                            },
                        ]
                    },
                    "maxItems": 3,
                },
                "also_reply": {"type": "boolean"},
            },
            "required": ["calls", "also_reply"],
            "additionalProperties": False,
        }

    def test_enum_slots_reach_the_grammar(self, decoder):
        decode = decoder.build(offer(LIGHTS_POWER), [], "lights on")

        branch = decode.response_schema["properties"]["calls"]["items"]["anyOf"][0]
        target = branch["properties"]["args"]["properties"]["target"]
        assert target["enum"] == ["bedroom", "kitchen"]

    def test_uses_only_allowed_keywords(self, decoder):
        decode = decoder.build(offer(LIGHTS_POWER, ALARMS_SET, LIGHTS_ADJUST), [], "x")

        used = set(schema_keywords(decode.response_schema))
        assert used <= _ALLOWED_KEYS
        assert {"const", "anyOf"} <= used

    def test_schema_does_not_alias_the_caller_args_schema(self, decoder):
        args_schema = to_args_schema(LIGHTS_POWER)
        decode = decoder.build([(LIGHTS_POWER, args_schema)], [], "lights on")

        args_schema["properties"]["on"]["type"] = "string"
        branch = decode.response_schema["properties"]["calls"]["items"]["anyOf"][0]
        assert branch["properties"]["args"]["properties"]["on"] == {"type": "boolean"}


class TestBuildDecodeSettings:
    def test_max_tokens_and_temperature(self, decoder):
        decode = decoder.build(offer(CLOCK_NOW), [], "time")

        assert decode.max_tokens == 256
        assert decode.temperature == 0.0

    def test_byte_identical_for_the_same_inputs(self, decoder):
        recent = [("user", "hello"), ("assistant", "hi there")]
        first = decoder.build(offer(LIGHTS_POWER, ALARMS_SET), recent, "lights off")
        second = ActionDecoder().build(offer(ALARMS_SET, LIGHTS_POWER), recent, "lights off")

        assert dump(first) == dump(second)
        assert first == second


# -- parse -----------------------------------------------------------------------------------

OFFERED = frozenset({"lights.power", "clock.now"})


class TestParse:
    def test_returns_calls_with_minted_ids_and_also_reply(self, decoder, monkeypatch):
        ids = iter(
            [
                uuid.UUID("00000000-0000-4000-8000-000000000001"),
                uuid.UUID("00000000-0000-4000-8000-000000000002"),
            ]
        )
        monkeypatch.setattr(decoder_module.uuid, "uuid4", lambda: next(ids))
        payload = {
            "calls": [
                {"tool": "lights.power", "args": {"target": "kitchen", "on": False}},
                {"tool": "clock.now", "args": {}},
            ],
            "also_reply": True,
        }

        calls, also_reply = decoder.parse(reply(payload), OFFERED)

        assert also_reply is True
        assert [(c.id, c.name, c.arguments) for c in calls] == [
            (
                "00000000-0000-4000-8000-000000000001",
                "lights.power",
                {"target": "kitchen", "on": False},
            ),
            ("00000000-0000-4000-8000-000000000002", "clock.now", {}),
        ]

    def test_real_ids_are_distinct_uuid4(self, decoder):
        payload = {"calls": [{"tool": "clock.now", "args": {}}] * 2, "also_reply": False}

        calls, also_reply = decoder.parse(reply(payload), OFFERED)

        assert also_reply is False
        assert calls[0].id != calls[1].id
        assert all(uuid.UUID(call.id).version == 4 for call in calls)

    def test_no_calls(self, decoder):
        calls, also_reply = decoder.parse(reply({"calls": [], "also_reply": True}), OFFERED)

        assert calls == ()
        assert also_reply is True

    def test_finish_reason_none_is_accepted(self, decoder):
        calls, _ = decoder.parse(reply({"calls": [], "also_reply": True}, None), OFFERED)

        assert calls == ()

    def test_invalid_json_raises(self, decoder):
        with pytest.raises(ActionDecodeError, match="not JSON"):
            decoder.parse(DecodeReply(content='{"calls": [', finish_reason="stop"), OFFERED)

    def test_missing_field_raises(self, decoder):
        with pytest.raises(ActionDecodeError, match="also_reply"):
            decoder.parse(reply({"calls": []}), OFFERED)

    def test_tool_not_offered_raises(self, decoder):
        payload = {"calls": [{"tool": "alarms.set", "args": {"time": "7"}}], "also_reply": False}

        with pytest.raises(ActionDecodeError, match="alarms.set"):
            decoder.parse(reply(payload), OFFERED)

    def test_extra_keys_raise(self, decoder):
        payload = {"calls": [], "also_reply": False, "reason": "none needed"}

        with pytest.raises(ActionDecodeError, match="reason"):
            decoder.parse(reply(payload), OFFERED)

    def test_finish_reason_length_raises(self, decoder):
        payload = {"calls": [], "also_reply": False}

        with pytest.raises(ActionDecodeError, match="length"):
            decoder.parse(reply(payload, finish_reason="length"), OFFERED)

    @pytest.mark.parametrize(
        "payload",
        [
            [],
            {"calls": {}, "also_reply": True},
            {"calls": [], "also_reply": "yes"},
            {"calls": [{"tool": "clock.now", "args": {}}] * 4, "also_reply": True},
            {"calls": ["clock.now"], "also_reply": True},
            {"calls": [{"tool": "clock.now"}], "also_reply": True},
            {"calls": [{"tool": "clock.now", "args": {}, "why": "x"}], "also_reply": True},
            {"calls": [{"tool": 7, "args": {}}], "also_reply": True},
            {"calls": [{"tool": "clock.now", "args": []}], "also_reply": True},
        ],
        ids=[
            "not-an-object",
            "calls-not-a-list",
            "also-reply-not-bool",
            "too-many-calls",
            "call-not-an-object",
            "call-missing-args",
            "call-extra-key",
            "tool-not-a-string",
            "args-not-an-object",
        ],
    )
    def test_other_schema_violations_raise(self, decoder, payload):
        with pytest.raises(ActionDecodeError):
            decoder.parse(reply(payload), OFFERED)


# -- repair ----------------------------------------------------------------------------------


class TestBuildRepair:
    def test_schema_is_the_tool_args_schema_alone(self, decoder):
        args_schema = to_args_schema(LIGHTS_ADJUST)
        decode = decoder.build_repair(
            LIGHTS_ADJUST, args_schema, {"target": "desk"}, "set brightness_pct", "dim the desk"
        )

        assert decode.response_schema == args_schema
        assert decode.response_schema is not args_schema
        assert decode.max_tokens == 256
        assert decode.temperature == 0.0

    def test_prompt_carries_error_utterance_and_rejected_args_as_json(self, decoder):
        args_schema = to_args_schema(LIGHTS_ADJUST)
        decode = decoder.build_repair(
            LIGHTS_ADJUST,
            args_schema,
            {"target": "desk", "brightness_pct": None},
            "at least one of brightness_pct is required",
            "dim the desk a bit",
        )

        text = "\n".join(str(m["content"]) for m in decode.messages)
        assert "at least one of brightness_pct is required" in text
        assert "dim the desk a bit" in text
        assert '{"brightness_pct":null,"target":"desk"}' in text
        assert to_catalog_line(LIGHTS_ADJUST, args_schema) in text
        assert decode.messages[0]["role"] == "system"
        assert decode.messages[-1]["role"] == "user"

    def test_byte_identical_for_the_same_inputs(self, decoder):
        def build() -> ActionDecode:
            return ActionDecoder().build_repair(
                LIGHTS_ADJUST,
                to_args_schema(LIGHTS_ADJUST),
                {"target": "desk", "brightness_pct": 400},
                "brightness_pct must be at most 100",
                "make the desk brighter",
            )

        assert dump(build()) == dump(build())

    def test_uses_only_allowed_keywords(self, decoder):
        decode = decoder.build_repair(
            LIGHTS_ADJUST, to_args_schema(LIGHTS_ADJUST), {}, "error", "dim it"
        )

        assert set(schema_keywords(decode.response_schema)) <= _ALLOWED_KEYS


class TestParseRepair:
    def test_returns_the_arguments(self, decoder):
        args = decoder.parse_repair(reply({"target": "desk", "brightness_pct": 30}))

        assert args == {"target": "desk", "brightness_pct": 30}

    def test_invalid_json_raises(self, decoder):
        with pytest.raises(ActionDecodeError, match="not JSON"):
            decoder.parse_repair(DecodeReply(content="{target", finish_reason="stop"))

    def test_non_object_raises(self, decoder):
        with pytest.raises(ActionDecodeError, match="object"):
            decoder.parse_repair(reply(["desk"]))

    def test_finish_reason_length_raises(self, decoder):
        with pytest.raises(ActionDecodeError, match="length"):
            decoder.parse_repair(reply({"target": "desk"}, finish_reason="length"))
