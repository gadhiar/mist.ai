"""Tests for `tool_event_fields`: the capability fields a tool-call event adds (T25)."""

from __future__ import annotations

import json

import pytest

from backend.capabilities.contracts import ActionRecord, ToolOutcome
from backend.capabilities.secrets import Secret, SecretRegistry, redact
from backend.capabilities.tool_events import DATA_MAX_BYTES, tool_event_fields

TOKEN = "tok-5f1e2d9c"


def _registry() -> SecretRegistry:
    registry = SecretRegistry()
    registry.register(Secret(TOKEN))
    return registry


def _redactor(registry: SecretRegistry | None = None):
    secrets = registry or _registry()
    return lambda text: redact(text, secrets)


def _record(
    *,
    data: dict[str, object] | None = None,
    ack: str | None = "Starting it.",
    view: str | None = "timers",
    power_level: str | None = "low",
    source: str = "fastpath",
) -> ActionRecord:
    return ActionRecord(
        tool_call_id="call-1",
        tool="timers.start",
        capability="timers",
        risk="reversible",
        args_summary="start a 10 minute timer",
        outcome=ToolOutcome(status="ok", summary="Timer set for 10 minutes.", data=data),
        duration_ms=4,
        power_level=power_level,  # type: ignore[arg-type]
        ack=ack,
        view=view,  # type: ignore[arg-type]
        source=source,  # type: ignore[arg-type]
    )


class TestStarted:
    def test_started_gives_ack_and_view(self):
        fields = tool_event_fields(_record(data={"id": "t1"}), "started", redact=_redactor())

        assert fields == {"ack": "Starting it.", "view": "timers"}

    def test_started_leaves_absent_values_out(self):
        fields = tool_event_fields(_record(ack=None, view=None), "started", redact=_redactor())

        assert fields == {}


class TestCompleted:
    def test_completed_gives_source_power_level_view_and_data(self):
        data = {"timers": [{"id": "t1", "remaining_s": 600}]}

        fields = tool_event_fields(_record(data=data), "completed", redact=_redactor())

        assert fields == {
            "source": "fastpath",
            "power_level": "low",
            "view": "timers",
            "data": data,
        }
        assert "ack" not in fields

    def test_completed_leaves_absent_values_out(self):
        record = _record(data=None, view=None, power_level=None, source="model")

        assert tool_event_fields(record, "completed", redact=_redactor()) == {"source": "model"}

    def test_a_registered_secret_in_data_is_masked_in_keys_values_and_lists(self):
        data = {
            "url": f"https://ha.local/?token={TOKEN}",
            TOKEN: "as a key",
            "nested": {"items": [f"Bearer {TOKEN}", 3, True, None]},
        }

        fields = tool_event_fields(_record(data=data), "completed", redact=_redactor())

        assert TOKEN not in json.dumps(fields)
        assert fields["data"] == {
            "url": "https://ha.local/?token=***",
            "***": "as a key",
            "nested": {"items": ["Bearer ***", 3, True, None]},
        }

    def test_a_number_that_is_a_registered_secret_is_masked(self):
        registry = SecretRegistry()
        registry.register(Secret("482913"))

        fields = tool_event_fields(
            _record(data={"pin": 482913, "level": 40}), "completed", redact=_redactor(registry)
        )

        assert fields["data"] == {"pin": "***", "level": 40}

    def test_the_record_itself_is_not_changed(self):
        data = {"url": f"https://ha.local/?token={TOKEN}"}
        record = _record(data=data)

        tool_event_fields(record, "completed", redact=_redactor())

        assert record.outcome.data == {"url": f"https://ha.local/?token={TOKEN}"}

    def test_data_at_the_cap_is_kept_and_one_byte_over_is_left_out(self):
        overhead = len(json.dumps({"text": ""}, separators=(",", ":")).encode())
        at_cap = {"text": "a" * (DATA_MAX_BYTES - overhead)}
        over_cap = {"text": "a" * (DATA_MAX_BYTES - overhead + 1)}

        kept = tool_event_fields(_record(data=at_cap), "completed", redact=_redactor())
        dropped = tool_event_fields(_record(data=over_cap), "completed", redact=_redactor())

        assert DATA_MAX_BYTES == 2048
        assert kept["data"] == at_cap
        assert "data" not in dropped
        assert dropped == {"source": "fastpath", "power_level": "low", "view": "timers"}

    def test_the_cap_counts_utf8_bytes_not_characters(self):
        overhead = len(json.dumps({"text": ""}, separators=(",", ":")).encode())
        # Each "é" is two bytes, so half the cap's characters already fill it.
        over_cap = {"text": "é" * ((DATA_MAX_BYTES - overhead) // 2 + 1)}

        fields = tool_event_fields(_record(data=over_cap), "completed", redact=_redactor())

        assert "data" not in fields

    def test_the_cap_applies_after_redaction(self):
        overhead = len(json.dumps({"text": ""}, separators=(",", ":")).encode())
        # The secret is longer than its mask, so the redacted JSON fits where the raw would not.
        filler = "a" * (DATA_MAX_BYTES - overhead - len("***"))
        data = {"text": filler + TOKEN}

        fields = tool_event_fields(_record(data=data), "completed", redact=_redactor())

        assert fields["data"] == {"text": filler + "***"}


def test_an_unknown_phase_raises():
    with pytest.raises(ValueError, match="phase"):
        tool_event_fields(_record(), "finished", redact=_redactor())  # type: ignore[arg-type]
