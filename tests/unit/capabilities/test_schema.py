"""Tests for the deterministic arguments schema and catalog line builder."""

from __future__ import annotations

import enum
import json
from collections.abc import Iterator
from typing import Literal

import pytest
from pydantic import BaseModel, Field

from backend.capabilities.contracts import ToolSpec
from backend.capabilities.schema import to_args_schema, to_catalog_line
from backend.errors import CapabilityRegistrationError
from tests.unit.capabilities.fakes import EmptyArgs, make_tool_spec

ALLOWED_KEYWORDS = {
    "type",
    "properties",
    "required",
    "additionalProperties",
    "enum",
    "const",
    "items",
    "anyOf",
    "maxItems",
    "minimum",
    "maximum",
}


# -- Fixture argument models, one per tool in the plan's tool-contract table ------------


class AlarmsSetArgs(BaseModel):
    time: str
    day: str | None = None
    label: str | None = None
    repeat: Literal["none", "weekends", "daily", "weekdays"] = "none"


class AlarmsCancelArgs(BaseModel):
    alarm_id: str | None = None
    label: str | None = None
    time: str | None = None
    all: bool | None = None


class RingingAction(enum.Enum):
    SNOOZE = "snooze"
    STOP = "stop"


class AlarmsRingingArgs(BaseModel):
    action: RingingAction
    minutes: int | None = Field(default=None, ge=1, le=60)


class TimersStartArgs(BaseModel):
    hours: int | None = None
    minutes: int | None = None
    seconds: int | None = None
    label: str | None = None


class TimersCancelArgs(BaseModel):
    timer_id: str | None = None
    label: str | None = None
    all: bool | None = None


class LightsPowerArgs(BaseModel):
    target: str | None = None
    on: bool


class LightsAdjustArgs(BaseModel):
    target: str | None = None
    brightness_pct: int | None = Field(default=None, ge=0, le=100)
    color: str | None = None
    kelvin: int | None = Field(default=None, ge=1000, le=10000)


class LightsStatusArgs(BaseModel):
    target: str | None = None


class MusicPlayArgs(BaseModel):
    query: str | None = None
    kind: Literal["track", "album", "artist", "playlist"] | None = None
    device: str | None = None


class MusicControlArgs(BaseModel):
    action: Literal["pause", "resume", "next", "previous"]


class MusicVolumeArgs(BaseModel):
    level_pct: int | None = Field(default=None, ge=0, le=100)
    change: Literal["up", "down"] | None = None


class Point(BaseModel):
    x: int
    tag: Literal["b", "a"] | None = None


class NestedArgs(BaseModel):
    points: list[Point] = Field(default_factory=list, max_length=3)
    origin: Point | None = None
    mode: Literal["fixed"]


FIXTURE_SPECS: tuple[ToolSpec, ...] = (
    make_tool_spec("alarms.set", AlarmsSetArgs),
    make_tool_spec("alarms.list", EmptyArgs, risk="read"),
    make_tool_spec("alarms.cancel", AlarmsCancelArgs),
    make_tool_spec("alarms.ringing", AlarmsRingingArgs),
    make_tool_spec("timers.start", TimersStartArgs),
    make_tool_spec("timers.list", EmptyArgs, risk="read"),
    make_tool_spec("timers.cancel", TimersCancelArgs),
    make_tool_spec("timers.stop", EmptyArgs),
    make_tool_spec("clock.now", EmptyArgs, risk="read"),
    make_tool_spec(
        "lights.power", LightsPowerArgs, risk="physical", default_policy="confirm_inferred"
    ),
    make_tool_spec(
        "lights.adjust", LightsAdjustArgs, risk="physical", default_policy="confirm_inferred"
    ),
    make_tool_spec("lights.status", LightsStatusArgs, risk="read"),
    make_tool_spec("music.play", MusicPlayArgs, risk="physical", default_policy="confirm_inferred"),
    make_tool_spec("music.control", MusicControlArgs),
    make_tool_spec("music.volume", MusicVolumeArgs),
    make_tool_spec("music.now_playing", EmptyArgs, risk="read"),
    make_tool_spec("test.nested", NestedArgs),
)


def _schema_nodes(node: object) -> Iterator[dict[str, object]]:
    """Yield every schema object in `node`, skipping property-name maps and enum values."""
    if not isinstance(node, dict):
        return
    yield node
    for key, value in node.items():
        if key == "properties":
            assert isinstance(value, dict)
            for child in value.values():
                yield from _schema_nodes(child)
        elif key == "items":
            yield from _schema_nodes(value)
        elif key == "anyOf":
            assert isinstance(value, list)
            for child in value:
                yield from _schema_nodes(child)


def _add_target_enum(tool: str, schema: dict[str, object]) -> dict[str, object]:
    properties = schema["properties"]
    assert isinstance(properties, dict)
    properties["target"] = {"type": "string", "enum": ["kitchen", "bedroom", "hall"]}
    return schema


def _ids(specs: tuple[ToolSpec, ...]) -> list[str]:
    return [spec.name for spec in specs]


class TestArgsSchemaKeywords:
    @pytest.mark.parametrize("spec", FIXTURE_SPECS, ids=_ids(FIXTURE_SPECS))
    def test_uses_only_grammar_keywords(self, spec):
        schema = to_args_schema(spec)

        for node in _schema_nodes(schema):
            assert (
                set(node) <= ALLOWED_KEYWORDS
            ), f"{spec.name}: {sorted(set(node) - ALLOWED_KEYWORDS)}"

    @pytest.mark.parametrize("spec", FIXTURE_SPECS, ids=_ids(FIXTURE_SPECS))
    def test_emits_no_title_or_defs(self, spec):
        text = json.dumps(to_args_schema(spec))

        assert '"title"' not in text
        assert "$defs" not in text
        assert "$ref" not in text

    @pytest.mark.parametrize("spec", FIXTURE_SPECS, ids=_ids(FIXTURE_SPECS))
    def test_every_object_forbids_additional_properties(self, spec):
        objects = [n for n in _schema_nodes(to_args_schema(spec)) if n.get("type") == "object"]

        assert objects
        assert all(node["additionalProperties"] is False for node in objects)

    @pytest.mark.parametrize("spec", FIXTURE_SPECS, ids=_ids(FIXTURE_SPECS))
    def test_every_enum_is_sorted(self, spec):
        for node in _schema_nodes(to_args_schema(spec)):
            if "enum" in node:
                assert node["enum"] == sorted(node["enum"])  # type: ignore[type-var]


class TestArgsSchemaShape:
    def test_optional_field_is_a_non_required_property_of_its_base_type(self):
        schema = to_args_schema(make_tool_spec("lights.adjust", LightsAdjustArgs))

        assert schema["properties"]["target"] == {"type": "string"}  # type: ignore[index]
        assert schema["properties"]["brightness_pct"] == {  # type: ignore[index]
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
        }
        assert schema["required"] == []

    def test_optional_enum_field_has_no_null_branch(self):
        schema = to_args_schema(make_tool_spec("music.play", MusicPlayArgs))

        assert schema["properties"]["kind"] == {  # type: ignore[index]
            "type": "string",
            "enum": ["album", "artist", "playlist", "track"],
        }
        assert "null" not in json.dumps(schema)

    def test_required_fields_keep_declaration_order(self):
        schema = to_args_schema(make_tool_spec("lights.power", LightsPowerArgs))

        assert schema == {
            "type": "object",
            "properties": {"target": {"type": "string"}, "on": {"type": "boolean"}},
            "required": ["on"],
            "additionalProperties": False,
        }

    def test_enum_class_is_inlined_and_sorted(self):
        schema = to_args_schema(make_tool_spec("alarms.ringing", AlarmsRingingArgs))

        assert schema["properties"]["action"] == {  # type: ignore[index]
            "type": "string",
            "enum": ["snooze", "stop"],
        }
        assert schema["required"] == ["action"]

    def test_nested_models_and_arrays_are_inlined(self):
        schema = to_args_schema(make_tool_spec("test.nested", NestedArgs))

        point = {
            "type": "object",
            "properties": {"x": {"type": "integer"}, "tag": {"type": "string", "enum": ["a", "b"]}},
            "required": ["x"],
            "additionalProperties": False,
        }
        assert schema["properties"] == {
            "points": {"type": "array", "items": point, "maxItems": 3},
            "origin": point,
            "mode": {"type": "string", "const": "fixed"},
        }
        assert schema["required"] == ["mode"]

    def test_tool_with_no_arguments_is_an_empty_closed_object(self):
        schema = to_args_schema(make_tool_spec("clock.now", EmptyArgs))

        assert schema == {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }

    def test_constraint_the_grammar_cannot_express_is_refused(self):
        class Named(BaseModel):
            name: str = Field(min_length=1)

        with pytest.raises(CapabilityRegistrationError, match="minLength"):
            to_args_schema(make_tool_spec("x.named", Named))


class TestArgsSchemaDeterminism:
    @pytest.mark.parametrize("spec", FIXTURE_SPECS, ids=_ids(FIXTURE_SPECS))
    def test_two_calls_are_byte_identical(self, spec):
        assert json.dumps(to_args_schema(spec)) == json.dumps(to_args_schema(spec))

    def test_keys_come_in_a_fixed_order_whatever_the_input_order(self):
        def refine(tool: str, schema: dict[str, object]) -> dict[str, object]:
            schema["properties"]["target"] = {"enum": ["b", "a"], "type": "string"}  # type: ignore[index]
            return {"additionalProperties": False, **schema}

        spec = make_tool_spec("lights.adjust", LightsAdjustArgs)

        text = json.dumps(to_args_schema(spec, refine))

        assert text.startswith(
            '{"type": "object", "properties": {"target": {"type": "string", "enum"'
        )
        assert '"brightness_pct": {"type": "integer", "minimum": 0, "maximum": 100}' in text
        assert text.endswith('"required": [], "additionalProperties": false}')

    def test_refined_enum_is_sorted_and_byte_identical(self):
        spec = make_tool_spec("lights.power", LightsPowerArgs)

        first = to_args_schema(spec, _add_target_enum)
        second = to_args_schema(spec, _add_target_enum)

        assert first["properties"]["target"] == {  # type: ignore[index]
            "type": "string",
            "enum": ["bedroom", "hall", "kitchen"],
        }
        assert json.dumps(first) == json.dumps(second)

    def test_refine_receives_the_tool_name(self):
        seen: list[str] = []

        def refine(tool: str, schema: dict[str, object]) -> dict[str, object]:
            seen.append(tool)
            return schema

        to_args_schema(make_tool_spec("lights.power", LightsPowerArgs), refine)

        assert seen == ["lights.power"]

    def test_refine_output_is_held_to_the_keyword_set(self):
        def refine(tool: str, schema: dict[str, object]) -> dict[str, object]:
            schema["properties"]["target"] = {"type": "string", "pattern": ".*"}  # type: ignore[index]
            return schema

        with pytest.raises(CapabilityRegistrationError, match="pattern"):
            to_args_schema(make_tool_spec("lights.power", LightsPowerArgs), refine)


class TestCatalogLine:
    def test_renders_enums_optional_markers_and_description(self):
        spec = make_tool_spec("lights.power", LightsPowerArgs, description="turn lights on or off")
        schema = to_args_schema(spec, _add_target_enum)

        line = to_catalog_line(spec, schema)

        assert (
            line
            == "lights.power(target?: bedroom|hall|kitchen, on: true|false): turn lights on or off"
        )

    def test_renders_integer_ranges(self):
        spec = make_tool_spec("lights.adjust", LightsAdjustArgs, description="change brightness")

        line = to_catalog_line(spec, to_args_schema(spec))

        assert line == (
            "lights.adjust(target?: str, brightness_pct?: int 0-100, color?: str, "
            "kelvin?: int 1000-10000): change brightness"
        )

    def test_renders_a_tool_with_no_arguments(self):
        spec = make_tool_spec("clock.now", EmptyArgs, description="tell the current time")

        assert to_catalog_line(spec, to_args_schema(spec)) == "clock.now(): tell the current time"

    def test_collapses_a_multi_line_description_onto_one_line(self):
        spec = make_tool_spec(
            "music.control", MusicControlArgs, description="pause,\n  resume\tor skip"
        )

        line = to_catalog_line(spec, to_args_schema(spec))

        assert line == "music.control(action: next|pause|previous|resume): pause, resume or skip"
        assert "\n" not in line

    def test_renders_nested_objects_arrays_and_consts(self):
        spec = make_tool_spec("test.nested", NestedArgs, description="nested")

        line = to_catalog_line(spec, to_args_schema(spec))

        assert line == (
            "test.nested(points?: list[{x: int, tag?: a|b}] max 3, origin?: {x: int, tag?: a|b}, "
            "mode: fixed): nested"
        )

    @pytest.mark.parametrize("spec", FIXTURE_SPECS, ids=_ids(FIXTURE_SPECS))
    def test_two_calls_are_byte_identical(self, spec):
        schema = to_args_schema(spec)

        assert to_catalog_line(spec, schema) == to_catalog_line(spec, to_args_schema(spec))
