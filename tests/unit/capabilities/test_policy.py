"""Tests for `PolicyTable`: file overrides and decisions by call source."""

from __future__ import annotations

import logging

import pytest
from pydantic import BaseModel

from backend.capabilities.contracts import ConfirmPolicy, InvocationContext, ToolSpec
from backend.capabilities.policy import PolicyTable
from backend.errors import CapabilityConfigError
from tests.unit.capabilities.fakes import DEFAULT_START, EmptyArgs, FakeCapability, make_tool_spec


class CancelArgs(BaseModel):
    alarm_id: str | None = None
    all: bool | None = None


LIGHTS_POWER = make_tool_spec("lights.power", risk="physical", default_policy="confirm_inferred")
ALARMS_CANCEL = make_tool_spec("alarms.cancel", CancelArgs)
CLOCK_NOW = make_tool_spec("clock.now", risk="read")
SPECS = (LIGHTS_POWER, ALARMS_CANCEL, CLOCK_NOW)


def _ctx(source: str) -> InvocationContext:
    return InvocationContext(
        session_id="s1",
        turn_id="t1",
        utterance="u",
        tool_call_id="c1",
        source=source,  # type: ignore[arg-type]
        now=DEFAULT_START,
    )


def _spec_with(policy: ConfirmPolicy) -> ToolSpec:
    return make_tool_spec("fake.do", default_policy=policy)


def _escalating_capability() -> FakeCapability:
    """A capability that escalates `alarms.cancel` with `all=true`, as the alarms one will."""
    capability = FakeCapability("alarms")
    capability.escalate_fn = lambda tool, args: tool == "alarms.cancel" and bool(
        getattr(args, "all", False)
    )
    return capability


class TestOverridesFile:
    def test_no_file_keeps_each_default(self):
        table = PolicyTable.from_specs(SPECS, None)

        assert table.decide(LIGHTS_POWER, EmptyArgs(), _ctx("model"), FakeCapability()) == "confirm"
        assert table.decide(CLOCK_NOW, EmptyArgs(), _ctx("model"), FakeCapability()) == "allow"

    def test_file_overrides_apply(self, tmp_path):
        path = tmp_path / "policy.yaml"
        path.write_text("lights.power: confirm\nclock.now: deny\n", encoding="utf-8")

        table = PolicyTable.from_specs(SPECS, str(path))

        assert (
            table.decide(LIGHTS_POWER, EmptyArgs(), _ctx("fastpath"), FakeCapability()) == "confirm"
        )
        assert table.decide(CLOCK_NOW, EmptyArgs(), _ctx("model"), FakeCapability()) == "deny"

    def test_unknown_tool_names_are_ignored_with_one_warning(self, tmp_path, caplog):
        path = tmp_path / "policy.yaml"
        path.write_text("weather.now: deny\nnews.read: deny\nclock.now: deny\n", encoding="utf-8")

        with caplog.at_level(logging.WARNING, logger="backend.capabilities.policy"):
            table = PolicyTable.from_specs(SPECS, str(path))

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "news.read" in warnings[0].getMessage()
        assert "weather.now" in warnings[0].getMessage()
        assert table.decide(CLOCK_NOW, EmptyArgs(), _ctx("model"), FakeCapability()) == "deny"

    @pytest.mark.parametrize(
        "mode",
        [
            pytest.param("confirm_unless_explicit", id="removed-mode"),
            pytest.param("sometimes", id="unknown-mode"),
            pytest.param("3", id="number"),
        ],
    )
    def test_invalid_mode_refuses_the_whole_file(self, tmp_path, mode):
        path = tmp_path / "policy.yaml"
        path.write_text(f"clock.now: deny\nlights.power: {mode}\n", encoding="utf-8")

        with pytest.raises(CapabilityConfigError, match="lights.power"):
            PolicyTable.from_specs(SPECS, str(path))

    @pytest.mark.parametrize(
        "content",
        [
            pytest.param("- clock.now\n", id="a-list"),
            pytest.param("clock.now: [deny\n", id="malformed-yaml"),
            pytest.param("1: deny\n", id="non-string-key"),
        ],
    )
    def test_malformed_file_is_refused(self, tmp_path, content):
        path = tmp_path / "policy.yaml"
        path.write_text(content, encoding="utf-8")

        with pytest.raises(CapabilityConfigError):
            PolicyTable.from_specs(SPECS, str(path))

    def test_missing_file_is_refused_naming_the_path(self, tmp_path):
        path = tmp_path / "absent.yaml"

        with pytest.raises(CapabilityConfigError, match="absent.yaml"):
            PolicyTable.from_specs(SPECS, str(path))

    def test_empty_file_keeps_the_defaults(self, tmp_path):
        path = tmp_path / "policy.yaml"
        path.write_text("", encoding="utf-8")

        table = PolicyTable.from_specs(SPECS, str(path))

        assert (
            table.decide(LIGHTS_POWER, EmptyArgs(), _ctx("fastpath"), FakeCapability()) == "allow"
        )


class TestDecisionBySource:
    @pytest.mark.parametrize(
        ("policy", "source", "expected"),
        [
            pytest.param("allow", "model", "allow", id="allow-model"),
            pytest.param("allow", "fastpath", "allow", id="allow-fastpath"),
            pytest.param("confirm_inferred", "fastpath", "allow", id="inferred-fastpath"),
            pytest.param("confirm_inferred", "model", "confirm", id="inferred-model"),
            pytest.param("confirm", "fastpath", "confirm", id="confirm-fastpath"),
            pytest.param("confirm", "model", "confirm", id="confirm-model"),
            pytest.param("deny", "model", "deny", id="deny-model"),
            pytest.param("deny", "fastpath", "deny", id="deny-fastpath"),
            pytest.param("deny", "proposal", "deny", id="deny-proposal"),
            pytest.param("allow", "proposal", "allow", id="allow-proposal"),
            pytest.param("confirm_inferred", "proposal", "allow", id="inferred-proposal"),
            pytest.param("confirm", "proposal", "allow", id="confirm-proposal"),
        ],
    )
    def test_mode_and_source_give_the_decision(self, policy, source, expected):
        spec = _spec_with(policy)
        table = PolicyTable.from_specs([spec], None)

        assert table.decide(spec, EmptyArgs(), _ctx(source), FakeCapability()) == expected


class TestEscalation:
    @pytest.mark.parametrize("source", ["model", "fastpath"])
    def test_escalate_raises_an_allowed_call_to_confirm(self, source):
        table = PolicyTable.from_specs(SPECS, None)

        decision = table.decide(
            ALARMS_CANCEL, CancelArgs(all=True), _ctx(source), _escalating_capability()
        )

        assert decision == "confirm"

    def test_call_that_does_not_escalate_stays_allowed(self):
        table = PolicyTable.from_specs(SPECS, None)

        decision = table.decide(
            ALARMS_CANCEL, CancelArgs(alarm_id="a1"), _ctx("model"), _escalating_capability()
        )

        assert decision == "allow"

    def test_approved_escalated_call_runs(self):
        table = PolicyTable.from_specs(SPECS, None)

        decision = table.decide(
            ALARMS_CANCEL, CancelArgs(all=True), _ctx("proposal"), _escalating_capability()
        )

        assert decision == "allow"

    def test_escalate_never_lowers_a_deny(self, tmp_path):
        path = tmp_path / "policy.yaml"
        path.write_text("alarms.cancel: deny\n", encoding="utf-8")
        table = PolicyTable.from_specs(SPECS, str(path))

        decision = table.decide(
            ALARMS_CANCEL, CancelArgs(all=True), _ctx("model"), _escalating_capability()
        )

        assert decision == "deny"

    def test_a_false_escalate_never_lowers_a_confirm(self):
        table = PolicyTable.from_specs(SPECS, None)
        capability = FakeCapability("lights")
        capability.escalate_fn = lambda tool, args: False

        assert table.decide(LIGHTS_POWER, EmptyArgs(), _ctx("model"), capability) == "confirm"

    def test_spec_missing_from_the_table_uses_its_own_default(self):
        table = PolicyTable.from_specs([], None)

        assert table.decide(LIGHTS_POWER, EmptyArgs(), _ctx("model"), FakeCapability()) == "confirm"
