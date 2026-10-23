"""The schema-constrained decode that lets a small model choose capability actions.

A small model chooses better among fewer, shorter options, and a short prompt prefills quickly.
So a routed turn gets a separate prompt, sharing nothing with the main chat prompt (no persona,
retrieval or profile text), whose only job is to choose among the offered catalog lines, and a
JSON schema that llama-server turns into a grammar, so tool names and enum slots are limited to
what the turn offers. The prompt and the schema are byte-identical for the same inputs, which
keeps the request reproducible in tests and logs.
"""

from __future__ import annotations

import copy
import json
import uuid
from collections.abc import Mapping, Sequence

from backend.errors import ActionDecodeError
from backend.llm.models import ToolCall

from .contracts import ActionDecode, DecodeReply, ToolSpec
from .schema import to_catalog_line

DECODE_INSTRUCTIONS = (
    "You choose actions for a voice assistant. Reply with JSON only.\n"
    "Choose only actions listed below.\n"
    "For a time or date, pass the user's own words.\n"
    "Output a call only for an action the user asked for. Output no call when the user is only"
    " talking about these things.\n"
    "Set also_reply to true when the user also asked or said something that needs an answer"
    " beyond the actions, and set it to true when unsure.\n"
    "Actions:"
)
"""The fixed system text of a turn decode, before the catalog lines.

A module constant so the factory can count its tokens once for the router's
`reserved_tokens`: the per-capability catalog budget is half of what the ceiling leaves after
it, `(ceiling - reserved_tokens) // 2`.
"""

REPAIR_INSTRUCTIONS = (
    "The arguments chosen for this action failed validation. Reply with corrected arguments"
    " as a JSON object only, following the user's request.\n"
    "Action:"
)
"""The fixed system text of a repair decode, before the tool's catalog line."""

MAX_CALLS = 3
"""The most calls one decode may choose; the grammar enforces it and `parse` checks it again."""

DECODE_MAX_TOKENS = 256
"""The token bound of every decode; a reply cut off at it is refused, never read in part."""

RECENT_MESSAGES = 4
"""How many recent user and assistant messages give the decode its conversational context."""

RECENT_MESSAGE_CHARS = 300
"""The cap on each recent message, so a long reply cannot crowd out the catalog."""

_RECENT_ROLES = frozenset({"user", "assistant"})
_DECODE_KEYS = frozenset({"calls", "also_reply"})
_CALL_KEYS = frozenset({"tool", "args"})


class ActionDecoder:
    """Builds the constrained decode requests and reads their replies.

    It holds no state and calls no model: it only knows the prompt and schema shape, so the
    code that routes a turn and the code that runs the request can each be tested without a
    model.
    """

    def build(
        self,
        offered: Sequence[tuple[ToolSpec, Mapping[str, object]]],
        recent: Sequence[tuple[str, str]],
        utterance: str,
    ) -> ActionDecode:
        """Build the turn decode over the tools the router offered.

        Tools are ordered by name, so the same set gives the same bytes whatever order the
        router produced. Only user and assistant messages are kept from `recent`, because the
        decode needs what was said, not tool or system traffic.

        Args:
            offered: Each offered tool with its arguments schema, refined or not.
            recent: The session's recent `(role, text)` messages, oldest first, without the
                current utterance.
            utterance: The text the decode chooses actions for; sent whole.

        Returns:
            The request, with `max_tokens` 256 and `temperature` 0.0.

        Raises:
            ValueError: Nothing is offered; an empty `anyOf` would be an invalid grammar.
        """
        if not offered:
            raise ValueError("build needs at least one offered tool")
        ordered = sorted(offered, key=lambda pair: pair[0].name)

        catalog = "\n".join(to_catalog_line(spec, schema) for spec, schema in ordered)
        messages: list[dict[str, object]] = [
            {"role": "system", "content": f"{DECODE_INSTRUCTIONS}\n{catalog}"}
        ]
        kept = [(role, text) for role, text in recent if role in _RECENT_ROLES]
        for role, text in kept[-RECENT_MESSAGES:]:
            messages.append({"role": role, "content": text[:RECENT_MESSAGE_CHARS]})
        messages.append({"role": "user", "content": utterance})

        branches = [_call_branch(spec.name, schema) for spec, schema in ordered]
        response_schema = {
            "type": "object",
            "properties": {
                "calls": {"type": "array", "items": {"anyOf": branches}, "maxItems": MAX_CALLS},
                "also_reply": {"type": "boolean"},
            },
            "required": ["calls", "also_reply"],
            "additionalProperties": False,
        }
        return ActionDecode(
            messages=tuple(messages),
            response_schema=response_schema,
            max_tokens=DECODE_MAX_TOKENS,
            temperature=0.0,
        )

    def build_repair(
        self,
        spec: ToolSpec,
        args_schema: Mapping[str, object],
        args: Mapping[str, object],
        error: str,
        utterance: str,
    ) -> ActionDecode:
        """Build the one repair decode for a call whose arguments failed validation.

        The grammar cannot express every rule (for example "at least one of brightness, color
        and kelvin"), so the model is shown the rule it broke and asked again under the tool's
        arguments schema alone. The rejected arguments are rendered with sorted keys so the
        prompt is byte-identical for the same inputs.

        Args:
            spec: The tool; its catalog line tells the model what the arguments are for.
            args_schema: The schema the turn decode used for this tool.
            args: The arguments that failed validation.
            error: The validation message.
            utterance: The user's text for the turn.

        Returns:
            The request, with `max_tokens` 256 and `temperature` 0.0.
        """
        rejected = json.dumps(
            dict(args), sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
        )
        system = f"{REPAIR_INSTRUCTIONS}\n{to_catalog_line(spec, args_schema)}"
        user = (
            f"User request: {utterance}\n"
            f"Rejected arguments: {rejected}\n"
            f"Validation error: {error}"
        )
        return ActionDecode(
            messages=(
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ),
            response_schema=copy.deepcopy(dict(args_schema)),
            max_tokens=DECODE_MAX_TOKENS,
            temperature=0.0,
        )

    def parse(
        self, reply: DecodeReply, offered: frozenset[str]
    ) -> tuple[tuple[ToolCall, ...], bool]:
        """Read a turn decode's reply into calls and `also_reply`.

        The grammar should already prevent every shape error checked here; the checks are a
        backstop for a provider that ignored the schema. Arguments are not validated: that is
        the invoke pipeline's step, which can repair them. Each call gets a fresh `uuid4` id, as
        `_dispatch_tool_with_observability` mints one when the provider sends none.

        Args:
            reply: The joined content and finish reason of the decode.
            offered: The tool names this turn's decode could choose.

        Returns:
            The calls in the model's order, and whether the main passes should still run.

        Raises:
            ActionDecodeError: The reply was cut off at the token bound, is not JSON, or does
                not match the decode schema, including a tool outside `offered`.
        """
        payload = _load_object(reply)
        _require_keys(payload, _DECODE_KEYS, "decode output")

        also_reply = payload["also_reply"]
        if not isinstance(also_reply, bool):
            raise ActionDecodeError("decode output 'also_reply' is not a boolean")
        raw_calls = payload["calls"]
        if not isinstance(raw_calls, list):
            raise ActionDecodeError("decode output 'calls' is not a list")
        if len(raw_calls) > MAX_CALLS:
            raise ActionDecodeError(
                f"decode output has {len(raw_calls)} calls; at most {MAX_CALLS} are allowed"
            )

        calls: list[ToolCall] = []
        for index, raw in enumerate(raw_calls):
            where = f"decode call {index}"
            if not isinstance(raw, dict):
                raise ActionDecodeError(f"{where} is not an object")
            _require_keys(raw, _CALL_KEYS, where)
            tool, args = raw["tool"], raw["args"]
            if not isinstance(tool, str):
                raise ActionDecodeError(f"{where} 'tool' is not a string")
            if tool not in offered:
                raise ActionDecodeError(
                    f"{where} names '{tool}', which this turn did not offer"
                    f" (offered: {', '.join(sorted(offered))})"
                )
            if not isinstance(args, dict):
                raise ActionDecodeError(f"{where} 'args' is not an object")
            calls.append(ToolCall(id=str(uuid.uuid4()), name=tool, arguments=args))
        return tuple(calls), also_reply

    def parse_repair(self, reply: DecodeReply) -> dict[str, object]:
        """Read a repair decode's reply into arguments, still unvalidated.

        Raises:
            ActionDecodeError: The reply was cut off at the token bound, is not JSON, or is not
                a JSON object.
        """
        return _load_object(reply)


def _call_branch(name: str, args_schema: Mapping[str, object]) -> dict[str, object]:
    # A deep copy, so a caller that later mutates its schema cannot change a built request.
    return {
        "type": "object",
        "properties": {"tool": {"const": name}, "args": copy.deepcopy(dict(args_schema))},
        "required": ["tool", "args"],
        "additionalProperties": False,
    }


def _load_object(reply: DecodeReply) -> dict[str, object]:
    if reply.finish_reason == "length":
        raise ActionDecodeError(
            "decode output was cut off at the token limit (finish_reason length)"
        )
    try:
        payload = json.loads(reply.content)
    except json.JSONDecodeError as exc:
        raise ActionDecodeError(f"decode output is not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ActionDecodeError("decode output is not a JSON object")
    return payload


def _require_keys(payload: Mapping[str, object], expected: frozenset[str], where: str) -> None:
    missing = sorted(expected - payload.keys())
    if missing:
        raise ActionDecodeError(f"{where} is missing {', '.join(missing)}")
    extra = sorted(payload.keys() - expected)
    if extra:
        raise ActionDecodeError(f"{where} has unexpected keys {', '.join(extra)}")
