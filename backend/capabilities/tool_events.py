"""The capability fields a turn handler adds to its `tool_call_started` and `tool_call_completed`.

A voice reply stays brief, so the depth of an outcome reaches the user on screen instead: the
completed event carries the outcome's `data` and the `view` it fills, and the started event the
`ack` a voice turn may speak while the call runs. The handler owns the events; this module only
says which fields an `ActionRecord` adds to each, so every path that emits them adds the same.

`data` leaves the host for every connected client, so it goes through `redact()` first, and a
payload whose JSON is over `DATA_MAX_BYTES` is left out rather than cut, because a cut payload
would not parse. The model still reads the whole outcome in the actions block.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Literal

from .contracts import ActionRecord

ToolEventPhase = Literal["started", "completed"]

DATA_MAX_BYTES = 2048
"""The most UTF-8 bytes of compact JSON a completed event's `data` may take."""


def tool_event_fields(
    record: ActionRecord, phase: ToolEventPhase, *, redact: Callable[[str], str]
) -> dict[str, object]:
    """Return the capability fields one tool-call event adds for `record`.

    Args:
        record: The call the event reports.
        phase: `started` gives `ack` and `view`; `completed` gives `source`, `power_level`,
            `view` and `data`.
        redact: Masks registered secrets in a string, such as `CapabilityTurnPort.redact`.
            Every string in `data`, keys included, passes through it.

    Returns:
        The fields whose values are set; an absent value is left out rather than sent as
        null. `data` is also left out when its redacted JSON is over `DATA_MAX_BYTES`.

    Raises:
        ValueError: `phase` is neither `started` nor `completed`.
    """
    if phase == "started":
        fields: dict[str, object | None] = {"ack": record.ack, "view": record.view}
    elif phase == "completed":
        fields = {
            "source": record.source,
            "power_level": record.power_level,
            "view": record.view,
            "data": _bounded_data(record.outcome.data, redact),
        }
    else:
        raise ValueError(f"unknown tool event phase {phase!r}")
    return {name: value for name, value in fields.items() if value is not None}


def _bounded_data(data: Mapping[str, object] | None, redact: Callable[[str], str]) -> object:
    if data is None:
        return None
    clean = _redacted(data, redact)
    encoded = json.dumps(clean, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return clean if len(encoded) <= DATA_MAX_BYTES else None


def _redacted(value: object, redact: Callable[[str], str]) -> object:
    """Return a JSON-ready copy of `value` with every string, and every number, redacted.

    A number is checked through its text, since a secret such as a PIN can arrive as an int;
    one that redaction changes becomes its redacted text. Anything that is not JSON is
    rendered with `str`, as `ToolOutcome.to_llm_json` renders it.
    """
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, int | float):
        text = str(value)
        masked = redact(text)
        return value if masked == text else masked
    if isinstance(value, Mapping):
        return {redact(str(key)): _redacted(item, redact) for key, item in value.items()}
    if isinstance(value, Sequence | set | frozenset) and not isinstance(value, bytes | bytearray):
        return [_redacted(item, redact) for item in value]
    return redact(str(value))
