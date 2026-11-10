"""Run 7's tool set, its 24 prompts and the tool-call scorer (mist-model-host plan v2, task T7).

`tools_schema.json` holds the 11 tools in OpenAI format, sent unchanged with every tool request;
`tool_prompts.json` holds the 24 prompts and what each must call. The scorer follows Run 7: a call
is valid when its arguments parse as JSON, it names the expected tool, it carries every required
argument with the type the schema gives (and inside any enum or range the schema sets), and it
matches every checked value. Two-action prompts are scored twice, on the first call alone and on
all calls in order, because a small model that does the first half reliably is a different finding
from one that does neither. A no-tool prompt is valid only with no call at all.

Stdlib only: the host runs it under a plain Python.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from . import core

TOOLS_SCHEMA_PATH = core.SPEECH_DIR / "tools_schema.json"
TOOL_PROMPTS_PATH = core.SPEECH_DIR / "tool_prompts.json"
EXPECTED_TOOL_COUNT = 11
EXPECTED_PROMPT_COUNT = 24
PROMPT_KINDS = ("command", "no_tool", "hand_off")
CHECK_KINDS = ("equals", "number", "time", "date", "contains", "items_contain")
HAND_OFF_TOOL = "hand_off"

_TIME_RE = re.compile(r"(\d{1,2}):(\d{2})(?::00)?")
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


class ToolSetError(core.InputError):
    """The tool set or the prompts do not hold together."""


@dataclass(frozen=True, slots=True)
class Check:
    """One checked argument value: `kind` says how `value` is compared."""

    arg: str
    kind: str
    value: object


@dataclass(frozen=True, slots=True)
class ExpectedCall:
    """A call a prompt expects, in order, with the values it checks."""

    tool: str
    checks: tuple[Check, ...]


@dataclass(frozen=True, slots=True)
class ToolPrompt:
    """One of the 24 prompts."""

    id: str
    kind: str
    text: str
    expect: tuple[ExpectedCall, ...]

    @property
    def two_action(self) -> bool:
        """A command that needs more than one call, scored on its first call and on all."""
        return self.kind == "command" and len(self.expect) > 1


@dataclass(frozen=True, slots=True)
class PromptScore:
    """The scored calls of one reply."""

    prompt_id: str
    valid_first: bool
    valid_all: bool
    n_calls: int
    problems: tuple[str, ...]

    def as_row(self) -> dict[str, object]:
        """The `requests.jsonl` fields of this score."""
        return {
            "tool_valid_first": self.valid_first,
            "tool_valid_all": self.valid_all,
            "tool_n_calls": self.n_calls,
            "tool_problems": list(self.problems),
        }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_tools(path: Path = TOOLS_SCHEMA_PATH) -> list[dict]:
    """The tool list, exactly as it is sent in each request's `tools`."""
    doc = core.load_json(path)
    tools = doc.get("tools")
    if not isinstance(tools, list) or not tools:
        raise ToolSetError(f"{path} has no tools list")
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if tool.get("type") != "function" or not isinstance(function, dict):
            raise ToolSetError(f"{path}: every tool must be an OpenAI function tool")
        params = function.get("parameters")
        if not isinstance(params, dict) or not isinstance(params.get("properties"), dict):
            raise ToolSetError(f"{path}: tool {function.get('name')!r} has no parameter object")
        missing = set(params.get("required", [])) - set(params["properties"])
        if missing:
            raise ToolSetError(
                f"{path}: tool {function.get('name')!r} requires undeclared {sorted(missing)}"
            )
    return tools


def tools_by_name(tools: Sequence[Mapping[str, object]]) -> dict[str, dict]:
    """Each tool's `function` object by its name."""
    return {tool["function"]["name"]: tool["function"] for tool in tools}


def load_prompts(
    path: Path = TOOL_PROMPTS_PATH, *, tools: Sequence[Mapping[str, object]] | None = None
) -> list[ToolPrompt]:
    """The prompts in file order; with `tools`, every expected tool and checked argument must exist.

    The cross-check runs at load so that a renamed argument cannot make every reply score invalid
    in a four-hour session before anyone notices.
    """
    doc = core.load_json(path)
    raw = doc.get("prompts")
    if not isinstance(raw, list):
        raise ToolSetError(f"{path} has no prompts list")
    by_name = tools_by_name(tools) if tools is not None else None
    prompts = []
    for entry in raw:
        try:
            expect = tuple(
                ExpectedCall(
                    tool=str(e["tool"]),
                    checks=tuple(
                        Check(arg=str(c["arg"]), kind=str(c["kind"]), value=c["value"])
                        for c in e.get("checks", [])
                    ),
                )
                for e in entry.get("expect", [])
            )
            prompt = ToolPrompt(
                id=str(entry["id"]), kind=str(entry["kind"]), text=str(entry["text"]), expect=expect
            )
        except (KeyError, TypeError, AttributeError) as exc:
            raise ToolSetError(f"{path}: malformed prompt {entry!r}: {exc}") from exc
        _check_prompt(prompt, by_name, path)
        prompts.append(prompt)
    ids = [p.id for p in prompts]
    if len(set(ids)) != len(ids):
        raise ToolSetError(f"{path}: prompt ids repeat")
    return prompts


def _check_prompt(prompt: ToolPrompt, by_name: Mapping[str, dict] | None, path: Path) -> None:
    if prompt.kind not in PROMPT_KINDS:
        raise ToolSetError(
            f"{path}: {prompt.id} has kind {prompt.kind!r}, not one of {PROMPT_KINDS}"
        )
    if prompt.kind == "no_tool" and prompt.expect:
        raise ToolSetError(f"{path}: no-tool prompt {prompt.id} expects a call")
    if prompt.kind == "hand_off" and [e.tool for e in prompt.expect] != [HAND_OFF_TOOL]:
        raise ToolSetError(f"{path}: hand-off prompt {prompt.id} must expect one {HAND_OFF_TOOL}")
    if prompt.kind == "command" and not prompt.expect:
        raise ToolSetError(f"{path}: command prompt {prompt.id} expects no call")
    for expected in prompt.expect:
        for check in expected.checks:
            if check.kind not in CHECK_KINDS:
                raise ToolSetError(f"{path}: {prompt.id} has check kind {check.kind!r}")
        if by_name is None:
            continue
        function = by_name.get(expected.tool)
        if function is None:
            raise ToolSetError(
                f"{path}: {prompt.id} expects {expected.tool!r}, not in the tool set"
            )
        properties = function["parameters"]["properties"]
        for check in expected.checks:
            if check.arg not in properties:
                raise ToolSetError(
                    f"{path}: {prompt.id} checks {check.arg!r}, which {expected.tool} lacks"
                )


def validate_tool_files(
    tools_path: Path = TOOLS_SCHEMA_PATH, prompts_path: Path = TOOL_PROMPTS_PATH
) -> list[str]:
    """Problems with the tool set and the prompts; an empty list when both hold together."""
    try:
        tools = load_tools(tools_path)
    except core.InputError as exc:
        return [str(exc)]
    problems = []
    if len(tools) != EXPECTED_TOOL_COUNT:
        problems.append(f"{tools_path} holds {len(tools)} tools, not {EXPECTED_TOOL_COUNT}")
    try:
        prompts = load_prompts(prompts_path, tools=tools)
    except core.InputError as exc:
        return [*problems, str(exc)]
    if len(prompts) != EXPECTED_PROMPT_COUNT:
        problems.append(f"{prompts_path} holds {len(prompts)} prompts, not {EXPECTED_PROMPT_COUNT}")
    return problems


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def parse_arguments(raw: object) -> dict | None:
    """A call's arguments as a dict: llama-server streams a JSON string, a fixture may hold a dict."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        value = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def normalise_time(value: object) -> str | None:
    """`HH:MM` from `H:MM`, `HH:MM` or `HH:MM:00`; None for anything that is not a clock reading."""
    if not isinstance(value, str):
        return None
    match = _TIME_RE.fullmatch(value.strip())
    if match is None:
        return None
    hour, minute = int(match.group(1)), int(match.group(2))
    if hour > 23 or minute > 59:
        return None
    return f"{hour:02d}:{minute:02d}"


def _type_ok(value: object, expected: str) -> bool:
    if expected == "integer":
        if isinstance(value, bool):
            return False
        return isinstance(value, int) or (isinstance(value, float) and value.is_integer())
    if expected == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return True


def _schema_problems(arguments: Mapping[str, object], function: Mapping[str, object]) -> list[str]:
    params = function["parameters"]
    properties = params["properties"]
    problems = []
    for name in params.get("required", []):
        if name not in arguments:
            problems.append(f"missing required argument {name}")
    for name, value in arguments.items():
        schema = properties.get(name)
        if schema is None:
            continue
        expected = schema.get("type")
        if expected and not _type_ok(value, expected):
            problems.append(f"{name} has type {type(value).__name__}, expected {expected}")
            continue
        if "enum" in schema and value not in schema["enum"]:
            problems.append(f"{name} is {value!r}, not one of {schema['enum']}")
        if expected in ("integer", "number"):
            if "minimum" in schema and value < schema["minimum"]:
                problems.append(f"{name} is {value}, below its minimum {schema['minimum']}")
            if "maximum" in schema and value > schema["maximum"]:
                problems.append(f"{name} is {value}, above its maximum {schema['maximum']}")
        string_items = expected == "array" and schema.get("items", {}).get("type") == "string"
        if string_items and not all(isinstance(item, str) for item in value):
            problems.append(f"{name} holds an item that is not a string")
    return problems


def _check_problem(check: Check, arguments: Mapping[str, object]) -> str | None:
    if check.arg not in arguments:
        return f"missing checked argument {check.arg}"
    actual = arguments[check.arg]
    if check.kind == "equals":
        ok = isinstance(actual, str) and actual.strip().lower() == str(check.value).lower()
    elif check.kind == "number":
        ok = (
            isinstance(actual, int | float)
            and not isinstance(actual, bool)
            and actual == check.value
        )
    elif check.kind == "time":
        ok = normalise_time(actual) == check.value
    elif check.kind == "date":
        ok = isinstance(actual, str) and bool(_DATE_RE.fullmatch(actual)) and actual == check.value
    elif check.kind == "contains":
        if isinstance(actual, str) and str(check.value).lower() in actual.lower():
            return None
        return f"{check.arg} does not contain {check.value!r}"
    elif check.kind == "items_contain":
        items = [str(i).lower() for i in actual] if isinstance(actual, list) else []
        for keyword in check.value:
            if not any(str(keyword).lower() in item for item in items):
                return f"{check.arg} has no item containing {keyword!r}"
        return None
    else:
        return f"unknown check kind {check.kind!r}"
    return None if ok else f"{check.arg} is {actual!r}, expected {check.value}"


def check_call(
    call: Mapping[str, object], expected: ExpectedCall, by_name: Mapping[str, dict]
) -> list[str]:
    """Problems with one call against the call expected in its place; empty when it is valid."""
    function = call.get("function") if isinstance(call, Mapping) else None
    name = function.get("name") if isinstance(function, Mapping) else None
    arguments = (
        parse_arguments(function.get("arguments")) if isinstance(function, Mapping) else None
    )
    if arguments is None:
        return ["arguments are not valid JSON"]
    if name != expected.tool:
        return [f"called {name!r}, expected {expected.tool}"]
    problems = _schema_problems(arguments, by_name[expected.tool])
    for check in expected.checks:
        problem = _check_problem(check, arguments)
        if problem is not None:
            problems.append(problem)
    return problems


def _calls_word(n: int) -> str:
    return "call" if n == 1 else "calls"


def score_calls(
    prompt: ToolPrompt,
    calls: Sequence[Mapping[str, object]],
    tools: Sequence[Mapping[str, object]],
) -> PromptScore:
    """Score a reply's calls against the prompt: the first call alone, and all calls in order."""
    by_name = tools_by_name(tools)
    n = len(calls)
    if not prompt.expect:
        problems = () if n == 0 else (f"{prompt.id} expects no call, got {n}",)
        return PromptScore(prompt.id, n == 0, n == 0, n, problems)

    problems: list[str] = []
    per_call: list[bool] = []
    for index, call in enumerate(calls[: len(prompt.expect)]):
        found = check_call(call, prompt.expect[index], by_name)
        per_call.append(not found)
        problems += [f"call {index + 1}: {p}" for p in found]
    if n != len(prompt.expect):
        want = len(prompt.expect)
        problems.append(f"expected {want} {_calls_word(want)}, got {n}")
    valid_first = bool(per_call) and per_call[0]
    valid_all = n == len(prompt.expect) and all(per_call)
    return PromptScore(prompt.id, valid_first, valid_all, n, tuple(problems))
