"""The serving identity: its components, its hash and the diff of two of them.

The serving identity keys an extraction epoch on every serving setting that
changes extraction output (MIS-196). Each component is shown to move the
hash, the flat form is pinned so a backend can store and diff it, and the
module is held to the standard library, because the extraction service image
imports it.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from backend.knowledge.extraction import serving_identity
from backend.knowledge.extraction.serving_identity import (
    REQUEST_STAGES,
    REQUIRED_SERVER_KEYS,
    SERVICE_SETTINGS_EXCLUDED,
    SERVING_IDENTITY_PREFIX,
    ServingIdentityComponents,
    StageRequestProfile,
    components_dict,
    compute_serving_identity,
    diff_components,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _profile(stage: str, **overrides) -> StageRequestProfile:
    values = {
        "stage": stage,
        "temperature": 0.0,
        "top_p": 0.9,
        "constrained": "schema",
        "thinking_effort": "low",
        "thinking_enabled": None,
        "thinking_budget_tokens": None,
    }
    values.update(overrides)
    return StageRequestProfile(**values)


def _components(**overrides) -> ServingIdentityComponents:
    values = {
        "adapter": "gptoss",
        "model_file": "gpt-oss-20b.gguf",
        "llama_cpp_build": "b11151",
        "max_attempts": 2,
        "scope_enabled": True,
        "requests": tuple(_profile(stage) for stage in REQUEST_STAGES),
        "server": (("ctx_size", "16384"), ("device", "rtx4070s"), ("image", "llama@sha256:a")),
        "n_ctx": 16384,
    }
    values.update(overrides)
    return ServingIdentityComponents(**values)


def _changed(value: object) -> object:
    """A different value of the same kind, for any flat component value."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 0.5
    if isinstance(value, str):
        return value + "-changed"
    if value is None:
        return 7
    raise AssertionError(f"no change rule for {value!r}")


_SCALAR_FIELDS = [
    f.name
    for f in dataclasses.fields(ServingIdentityComponents)
    if f.name not in ("requests", "server")
]
_PROFILE_FIELDS = [f.name for f in dataclasses.fields(StageRequestProfile) if f.name != "stage"]


# ---------------------------------------------------------------------------
# The constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_the_prefix_stages_and_required_keys(self):
        assert SERVING_IDENTITY_PREFIX == "si1-"
        assert REQUEST_STAGES == ("scope", "extract", "derive")
        assert REQUIRED_SERVER_KEYS == ("image", "device", "ctx_size")

    def test_every_excluded_setting_carries_a_reason(self):
        assert SERVICE_SETTINGS_EXCLUDED, "SERVICE_SETTINGS_EXCLUDED is empty"
        assert all(reason.strip() for reason in SERVICE_SETTINGS_EXCLUDED.values())

    def test_the_exclusions_cannot_be_edited_at_run_time(self):
        with pytest.raises(TypeError):
            SERVICE_SETTINGS_EXCLUDED["model_file"] = "edited"  # type: ignore[index]

    @pytest.mark.parametrize(
        ("instance", "field_name"),
        [
            pytest.param(_profile("scope"), "stage", id="stage-request-profile"),
            pytest.param(_components(), "adapter", id="serving-identity-components"),
        ],
    )
    def test_frozen_with_slots(self, instance, field_name):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(instance, field_name, "edited")
        assert not hasattr(instance, "__dict__")


# ---------------------------------------------------------------------------
# The flat form
# ---------------------------------------------------------------------------


class TestComponentsDict:
    def test_the_flat_keys_and_values(self):
        flat = components_dict(_components())

        assert flat == {
            "adapter": "gptoss",
            "model_file": "gpt-oss-20b.gguf",
            "llama_cpp_build": "b11151",
            "max_attempts": 2,
            "scope_enabled": True,
            "n_ctx": 16384,
            **{
                f"request.{stage}.{name}": getattr(_profile(stage), name)
                for stage in REQUEST_STAGES
                for name in _PROFILE_FIELDS
            },
            "server.ctx_size": "16384",
            "server.device": "rtx4070s",
            "server.image": "llama@sha256:a",
        }

    def test_every_profile_field_is_flattened(self):
        """Derived from the dataclass, so a field added later is flattened too."""
        flat = components_dict(_components())

        for stage in REQUEST_STAGES:
            for name in _PROFILE_FIELDS:
                assert f"request.{stage}.{name}" in flat

    def test_two_profiles_for_one_stage_are_refused(self):
        twice = (_profile("scope"), _profile("scope", temperature=1.0))

        with pytest.raises(ValueError, match="scope"):
            components_dict(_components(requests=twice))

    def test_a_server_key_declared_twice_is_refused(self):
        twice = (("device", "a"), ("device", "b"))

        with pytest.raises(ValueError, match="device"):
            components_dict(_components(server=twice))

    def test_the_values_are_json_scalars(self):
        flat = components_dict(_components())

        assert json.loads(json.dumps(flat)) == flat
        assert all(isinstance(v, str | int | float | bool | type(None)) for v in flat.values())


# ---------------------------------------------------------------------------
# The hash
# ---------------------------------------------------------------------------


class TestComputeServingIdentity:
    def test_format_is_si1_and_a_sha256(self):
        identity = compute_serving_identity(_components())

        assert identity.startswith("si1-")
        assert len(identity) == len("si1-") + 64
        int(identity[4:], 16)

    def test_hashes_the_canonical_json_of_the_flat_form(self):
        components = _components()
        canonical = json.dumps(components_dict(components), sort_keys=True, separators=(",", ":"))

        expected = "si1-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()

        assert compute_serving_identity(components) == expected

    def test_deterministic(self):
        assert compute_serving_identity(_components()) == compute_serving_identity(_components())

    @pytest.mark.parametrize("name", _SCALAR_FIELDS)
    def test_each_scalar_component_moves_the_identity(self, name):
        before = _components()
        moved = dataclasses.replace(before, **{name: _changed(getattr(before, name))})

        assert compute_serving_identity(moved) != compute_serving_identity(before)

    @pytest.mark.parametrize("stage", REQUEST_STAGES)
    @pytest.mark.parametrize("name", _PROFILE_FIELDS)
    def test_each_profile_field_of_each_stage_moves_the_identity(self, stage, name):
        before = _components()
        requests = tuple(
            (
                dataclasses.replace(p, **{name: _changed(getattr(p, name))})
                if p.stage == stage
                else p
            )
            for p in before.requests
        )

        moved = dataclasses.replace(before, requests=requests)

        assert compute_serving_identity(moved) != compute_serving_identity(before)

    @pytest.mark.parametrize(
        "server",
        [
            pytest.param(
                (("ctx_size", "16384"), ("device", "gtx1070"), ("image", "llama@sha256:a")),
                id="a-value-changes",
            ),
            pytest.param(
                (
                    ("ctx_size", "16384"),
                    ("device", "rtx4070s"),
                    ("image", "llama@sha256:a"),
                    ("n_cpu_moe", "13"),
                ),
                id="a-setting-is-added",
            ),
            pytest.param((("ctx_size", "16384"), ("device", "rtx4070s")), id="a-setting-is-dropped"),
        ],
    )
    def test_each_declared_server_setting_moves_the_identity(self, server):
        assert compute_serving_identity(_components(server=server)) != compute_serving_identity(
            _components()
        )

    def test_a_none_and_a_zero_budget_are_different_identities(self):
        none = _components(requests=(_profile("extract", thinking_budget_tokens=None),))
        zero = _components(requests=(_profile("extract", thinking_budget_tokens=0),))

        assert compute_serving_identity(none) != compute_serving_identity(zero)


# ---------------------------------------------------------------------------
# The diff
# ---------------------------------------------------------------------------


class TestDiffComponents:
    def test_equal_mappings_have_no_diff(self):
        flat = components_dict(_components())

        assert diff_components(flat, dict(flat)) == []

    def test_one_line_per_difference_sorted_by_name(self):
        expected = components_dict(_components())
        actual = {**expected, "n_ctx": 32768, "adapter": "qwen"}

        assert diff_components(expected, actual) == [
            'adapter: "gptoss" != "qwen"',
            "n_ctx: 16384 != 32768",
        ]

    def test_missing_keys_on_either_side_are_named(self):
        expected = {"server.device": "gtx1070", "n_ctx": 16384}
        actual = {"n_ctx": 16384, "server.n_cpu_moe": "13"}

        assert diff_components(expected, actual) == [
            'server.device: "gtx1070" != <missing>',
            'server.n_cpu_moe: <missing> != "13"',
        ]

    @pytest.mark.parametrize(
        ("expected", "actual"),
        [
            pytest.param(1, True, id="int-and-bool"),
            pytest.param(0, 0.0, id="int-and-float"),
            pytest.param(None, "null", id="none-and-a-string"),
        ],
    )
    def test_values_that_python_calls_equal_but_json_does_not_differ(self, expected, actual):
        assert diff_components({"x": expected}, {"x": actual}) != []

    def test_a_diff_against_nothing_lists_every_actual_component(self):
        actual = components_dict(_components())

        lines = diff_components({}, actual)

        assert len(lines) == len(actual)
        assert all(" <missing> != " in line for line in lines)


# ---------------------------------------------------------------------------
# The import closure
# ---------------------------------------------------------------------------


def test_the_module_imports_only_the_standard_library():
    tree = ast.parse(Path(serving_identity.__file__).read_text(encoding="utf-8"))
    imported = {
        (node.module if isinstance(node, ast.ImportFrom) else alias.name).split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in node.names
    }

    assert imported <= {
        "__future__",
        "collections",
        "dataclasses",
        "hashlib",
        "json",
        "types",
    }, imported
