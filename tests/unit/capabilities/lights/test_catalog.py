"""Tests for the light target catalog parsed from Home Assistant's `GetLiveContext` text."""

from __future__ import annotations

import json

import pytest

from backend.capabilities.lights.catalog import (
    EMPTY_CATALOG,
    LightCatalog,
    LightEntity,
    LightTarget,
    live_context_body,
    parse_live_context,
)

# Written to the layout the fake Home Assistant models (`DEFAULT_LIVE_CONTEXT`): a header line,
# then a YAML list with one mapping per exposed entity, aliases joined by commas. Synthetic
# names only.
LIVE_CONTEXT = (
    "Live Context: An overview of the areas and the devices in this smart home:\n"
    "- names: Bedroom Lamp\n"
    "  domain: light\n"
    "  state: 'off'\n"
    "  areas: Bedroom\n"
    "- names: Ceiling Light, Big Light\n"
    "  domain: light\n"
    "  state: 'on'\n"
    "  areas: Kitchen\n"
    "  attributes:\n"
    "    brightness: 128\n"
    "- names: Desk Strip\n"
    "  domain: light\n"
    "  state: unavailable\n"
    "- names: Kettle\n"
    "  domain: switch\n"
    "  state: 'off'\n"
    "  areas: Kitchen\n"
    "- names: Sofa Lamp\n"
    "  domain: light\n"
    "  state: on\n"
    "  areas: Living Room\n"
)


class TestParse:
    def test_extracts_light_names_and_areas_from_the_documented_shape(self):
        catalog = parse_live_context(LIVE_CONTEXT)
        assert catalog.names == (
            "Bedroom Lamp",
            "Big Light",
            "Ceiling Light",
            "Desk Strip",
            "Sofa Lamp",
        )
        assert catalog.areas == ("Bedroom", "Kitchen", "Living Room")

    def test_keeps_lights_only_with_their_states(self):
        catalog = parse_live_context(LIVE_CONTEXT)
        assert catalog.lights == (
            LightEntity(names=("Bedroom Lamp",), areas=("Bedroom",), state="off"),
            LightEntity(names=("Ceiling Light", "Big Light"), areas=("Kitchen",), state="on"),
            LightEntity(names=("Desk Strip",), areas=(), state="unavailable"),
            # An unquoted `on` is a YAML boolean; it still reads as the state "on".
            LightEntity(names=("Sofa Lamp",), areas=("Living Room",), state="on"),
        )

    def test_reads_the_text_inside_a_json_result_wrapper(self):
        wrapped = json.dumps({"success": True, "result": LIVE_CONTEXT})
        assert parse_live_context(wrapped) == parse_live_context(LIVE_CONTEXT)
        assert live_context_body(wrapped) == LIVE_CONTEXT.strip()

    def test_accepts_list_valued_names_and_areas(self):
        text = "- names: [Hall Light]\n  domain: light\n  state: 'off'\n  areas: [Hall, Stairs]\n"
        catalog = parse_live_context(text)
        assert catalog.names == ("Hall Light",)
        assert catalog.areas == ("Hall", "Stairs")

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "Live Context: nothing exposed",
            "- names: [unclosed\n  domain: light",
            "{not json",
            json.dumps({"success": False}),
            json.dumps([1, 2]),
            "- just a string\n- 42\n",
            "- names: 7\n  domain: light\n",
            "key: value\n",
            "\x00\x01 binary",
        ],
    )
    def test_unparseable_text_gives_an_empty_catalog_without_raising(self, text):
        assert parse_live_context(text) == EMPTY_CATALOG
        assert parse_live_context(text).targets() == ()


class TestCatalog:
    def test_targets_are_names_and_areas_sorted_without_case_duplicates(self):
        catalog = LightCatalog(
            lights=(
                LightEntity(names=("kitchen",), areas=("Kitchen",), state="on"),
                LightEntity(names=("Bedroom Lamp",), areas=("Bedroom",), state="off"),
            )
        )
        assert catalog.targets() == ("Bedroom", "Bedroom Lamp", "kitchen")

    def test_resolve_ignores_case_and_prefers_a_light_name(self):
        catalog = parse_live_context(LIVE_CONTEXT)
        assert catalog.resolve("bedroom lamp") == LightTarget(kind="name", value="Bedroom Lamp")
        assert catalog.resolve("  LIVING   room ") == LightTarget(kind="area", value="Living Room")
        assert catalog.resolve("big light") == LightTarget(kind="name", value="Big Light")
        assert catalog.resolve("garage") is None

    def test_lights_for_a_target(self):
        catalog = parse_live_context(LIVE_CONTEXT)
        kitchen = catalog.lights_for(LightTarget(kind="area", value="Kitchen"))
        assert [light.names[0] for light in kitchen] == ["Ceiling Light"]
        lamp = catalog.lights_for(LightTarget(kind="name", value="Sofa Lamp"))
        assert [light.names[0] for light in lamp] == ["Sofa Lamp"]
        assert len(catalog.lights_for(None)) == 4

    def test_empty_catalog(self):
        assert EMPTY_CATALOG.is_empty
        assert not parse_live_context(LIVE_CONTEXT).is_empty
        assert EMPTY_CATALOG.resolve("bedroom") is None
