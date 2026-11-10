"""scripts/model_bench/speech/toolcall.py: run 7's tool set, its 24 prompts and the scorer.

The tool set and prompts are checked against Run 7's table in the plan, word for word. The scorer
runs on `fixtures/speech/lowtier/tool_calls.jsonl`: one valid case per prompt, and the failure
kinds Test strategy names (wrong tool, missing argument, wrong value, malformed JSON, a call on a
no-tool prompt, two-action prompts), plus C03 and C12 against the fixed date.
"""

from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402

from scripts.model_bench.speech import lowtier_suite, toolcall  # noqa: E402

FIXTURE = _REPO_ROOT / "tests" / "unit" / "model_bench" / "fixtures" / "speech" / "lowtier"
CASES = [
    json.loads(line)
    for line in (FIXTURE / "tool_calls.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]

EXPECTED_TOOLS = [
    "set_timer",
    "set_alarm",
    "cancel_alarm",
    "add_reminder",
    "add_to_list",
    "set_lights",
    "control_music",
    "show_panel",
    "close_panel",
    "sleep",
    "hand_off",
]

# Run 7's table: id, prompt text, and the expected calls with their checked values.
EXPECTED_PROMPTS = {
    "C01": (
        "Set a timer for ten minutes.",
        [("set_timer", {"duration_seconds": ("number", 600)})],
    ),
    "C02": (
        "Turn the living room lights down to thirty percent.",
        [("set_lights", {"room": ("equals", "living room"), "brightness": ("number", 30)})],
    ),
    "C03": (
        "Remind me to call the dentist on Monday at nine.",
        [
            (
                "add_reminder",
                {
                    "text": ("contains", "dentist"),
                    "date": ("date", "2026-11-16"),
                    "time": ("time", "09:00"),
                },
            )
        ],
    ),
    "C04": (
        "Play something calm, maybe some piano, not too loud.",
        [("control_music", {"action": ("equals", "play"), "query": ("contains", "piano")})],
    ),
    "C05": (
        "Add oat milk, eggs and coffee filters to the shopping list.",
        [
            (
                "add_to_list",
                {
                    "list_name": ("equals", "shopping"),
                    "items": ("items_contain", ["oat milk", "eggs", "coffee filters"]),
                },
            )
        ],
    ),
    "C06": (
        "Cancel the alarm for six thirty and set one for seven fifteen instead.",
        [
            ("cancel_alarm", {"time": ("time", "06:30")}),
            ("set_alarm", {"time": ("time", "07:15")}),
        ],
    ),
    "C07": (
        "Show me the report on the second monitor.",
        [("show_panel", {"panel": ("equals", "report"), "monitor": ("number", 2)})],
    ),
    "C08": (
        "Close that window and bring the chart back up.",
        [("close_panel", {}), ("show_panel", {"panel": ("equals", "chart")})],
    ),
    "C09": (
        "Pause the music and turn the volume down a bit.",
        [
            ("control_music", {"action": ("equals", "pause")}),
            ("control_music", {"action": ("equals", "volume_down")}),
        ],
    ),
    "C10": ("Okay, that's enough for tonight. Go to sleep.", [("sleep", {})]),
    "C11": (
        "Turn off all the lights.",
        [("set_lights", {"room": ("equals", "all"), "action": ("equals", "off")})],
    ),
    "C12": (
        "Set an alarm for seven thirty tomorrow.",
        [("set_alarm", {"date": ("date", "2026-11-12"), "time": ("time", "07:30")})],
    ),
    "C13": (
        "Put the calendar on the main screen.",
        [("show_panel", {"panel": ("equals", "calendar"), "monitor": ("number", 1)})],
    ),
    "C14": (
        "Start a timer for ninety seconds called tea.",
        [("set_timer", {"duration_seconds": ("number", 90), "label": ("contains", "tea")})],
    ),
    "C15": ("Skip this song.", [("control_music", {"action": ("equals", "next")})]),
    "C16": (
        "Dim the kitchen lights a little.",
        [("set_lights", {"room": ("equals", "kitchen"), "action": ("equals", "dim")})],
    ),
    "N01": ("Tell me something interesting about octopuses.", []),
    "N02": ("Spell the word necessary for me, slowly.", []),
    "N03": ("Thanks, that's all for now.", []),
    "N04": ("How are you doing today?", []),
    "E01": ("Sum up what I worked on today in two sentences.", [("hand_off", {})]),
    "E02": ("What did we decide about the database migration last week?", [("hand_off", {})]),
    "E03": ("Help me plan a three-day trip to Lisbon on a small budget.", [("hand_off", {})]),
    "E04": (
        "Why does my build keep failing on the second run? Think it through.",
        [("hand_off", {})],
    ),
}


@pytest.fixture(scope="module")
def tools():
    return toolcall.load_tools()


@pytest.fixture(scope="module")
def prompts(tools):
    return {p.id: p for p in toolcall.load_prompts(tools=tools)}


def call(name: str, **arguments) -> dict:
    return {"function": {"name": name, "arguments": json.dumps(arguments)}}


# ---------------------------------------------------------------------------
# The committed tool set and prompts
# ---------------------------------------------------------------------------


def test_tools_schema_holds_the_eleven_tools_in_openai_format(tools):
    assert [t["function"]["name"] for t in tools] == EXPECTED_TOOLS
    for tool in tools:
        assert tool["type"] == "function"
        params = tool["function"]["parameters"]
        assert params["type"] == "object"
        assert set(params["required"]) <= set(params["properties"])
        assert tool["function"]["description"]


def test_hand_off_is_the_tool_for_memory_long_answers_and_careful_thought(tools):
    hand_off = next(t for t in tools if t["function"]["name"] == "hand_off")["function"]

    assert hand_off["parameters"]["properties"]["reason"]["enum"] == [
        "memory",
        "long_answer",
        "careful_thought",
    ]
    for phrase in ("memory", "long answer", "careful thought"):
        assert phrase in hand_off["description"]


def test_tool_prompts_hold_the_24_prompts_exactly_as_run_7_lists_them(prompts):
    assert list(prompts) == list(EXPECTED_PROMPTS)
    for prompt_id, (text, expected) in EXPECTED_PROMPTS.items():
        prompt = prompts[prompt_id]
        got = [(e.tool, {c.arg: (c.kind, c.value) for c in e.checks}) for e in prompt.expect]
        assert prompt.text == text, prompt_id
        assert got == expected, prompt_id


def test_prompt_kinds_follow_their_id_letter(prompts):
    kinds = {p.id: p.kind for p in prompts.values()}

    assert {k for i, k in kinds.items() if i.startswith("C")} == {"command"}
    assert {k for i, k in kinds.items() if i.startswith("N")} == {"no_tool"}
    assert {k for i, k in kinds.items() if i.startswith("E")} == {"hand_off"}
    assert [p.id for p in prompts.values() if p.two_action] == ["C06", "C08", "C09"]


def test_c03_and_c12_expect_the_dates_that_follow_from_the_fixed_date(prompts):
    now = lowtier_suite.FIXED_NOW
    days_to_monday = (0 - now.weekday()) % 7 or 7
    monday = (now + timedelta(days=days_to_monday)).date().isoformat()
    tomorrow = (now + timedelta(days=1)).date().isoformat()

    c03 = {c.arg: c.value for c in prompts["C03"].expect[0].checks}
    c12 = {c.arg: c.value for c in prompts["C12"].expect[0].checks}
    assert (monday, tomorrow) == ("2026-11-16", "2026-11-12")
    assert (c03["date"], c03["time"]) == (monday, "09:00")
    assert (c12["date"], c12["time"]) == (tomorrow, "07:30")


def test_validate_tool_files_finds_no_problem_in_the_committed_files():
    assert toolcall.validate_tool_files() == []


def test_load_prompts_refuses_a_check_on_an_argument_the_tool_lacks(tmp_path, tools):
    doc = json.loads(toolcall.TOOL_PROMPTS_PATH.read_text(encoding="utf-8"))
    doc["prompts"][0]["expect"][0]["checks"][0]["arg"] = "minutes"
    path = tmp_path / "tool_prompts.json"
    path.write_text(json.dumps(doc), encoding="utf-8")

    with pytest.raises(toolcall.ToolSetError, match="minutes"):
        toolcall.load_prompts(path, tools=tools)


def test_load_prompts_refuses_a_tool_the_tool_set_lacks(tmp_path, tools):
    doc = json.loads(toolcall.TOOL_PROMPTS_PATH.read_text(encoding="utf-8"))
    doc["prompts"][0]["expect"][0]["tool"] = "order_pizza"
    path = tmp_path / "tool_prompts.json"
    path.write_text(json.dumps(doc), encoding="utf-8")

    assert any("order_pizza" in p for p in toolcall.validate_tool_files(prompts_path=path))


# ---------------------------------------------------------------------------
# The scorer on the recorded fixture
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=[c["case"] for c in CASES])
def test_fixture_case_scores_as_recorded(case, prompts, tools):
    score = toolcall.score_calls(prompts[case["prompt_id"]], case["calls"], tools)

    assert (score.valid_first, score.valid_all) == (case["valid_first"], case["valid_all"])
    if case["problem"] is None:
        assert score.problems == ()
    else:
        assert any(case["problem"] in p for p in score.problems), score.problems


def test_every_prompt_has_a_valid_fixture_case(prompts):
    valid = {c["prompt_id"] for c in CASES if c["valid_first"] and c["valid_all"]}

    assert set(prompts) <= valid


def test_the_fixture_covers_every_failure_kind_test_strategy_names():
    names = " | ".join(c["case"] for c in CASES if not c["valid_all"])

    for kind in (
        "wrong tool",
        "missing argument",
        "wrong value",
        "malformed JSON",
        "call on a no-tool prompt",
        "two-action",
    ):
        assert kind in names


@pytest.mark.parametrize("prompt_id", list(EXPECTED_PROMPTS))
def test_a_hand_off_call_is_valid_on_e01_to_e04_and_invalid_elsewhere(prompt_id, prompts, tools):
    calls = [call("hand_off", request="Look into this for me.", reason="memory")]

    score = toolcall.score_calls(prompts[prompt_id], calls, tools)

    expected = prompt_id.startswith("E")
    assert (score.valid_first, score.valid_all) == (expected, expected)


def test_a_hand_off_with_a_reason_outside_its_enum_is_invalid(prompts, tools):
    calls = [call("hand_off", request="Sum up my day.", reason="busy")]

    score = toolcall.score_calls(prompts["E01"], calls, tools)

    assert not score.valid_first
    assert any("not one of" in p for p in score.problems)


@pytest.mark.parametrize(
    ("prompt_id", "arguments", "valid"),
    [
        ("C03", {"text": "dentist", "date": "2026-11-16", "time": "09:00"}, True),
        ("C03", {"text": "dentist", "date": "2026-11-11", "time": "09:00"}, False),
        ("C03", {"text": "dentist", "date": "2026-11-23", "time": "09:00"}, False),
        ("C03", {"text": "dentist", "date": "2026-11-16", "time": "21:00"}, False),
        ("C12", {"date": "2026-11-12", "time": "07:30"}, True),
        ("C12", {"date": "2026-11-11", "time": "07:30"}, False),
        ("C12", {"date": "2026-11-13", "time": "07:30"}, False),
    ],
)
def test_c03_and_c12_score_against_the_fixed_date(prompt_id, arguments, valid, prompts, tools):
    tool = prompts[prompt_id].expect[0].tool

    score = toolcall.score_calls(prompts[prompt_id], [call(tool, **arguments)], tools)

    assert score.valid_first is valid


def test_two_action_prompts_score_the_first_call_and_all_calls_separately(prompts, tools):
    first_only = [call("close_panel")]
    both = [call("close_panel"), call("show_panel", panel="chart", monitor=1)]
    wrong_first = [call("show_panel", panel="chart", monitor=1), call("close_panel")]

    scores = [
        toolcall.score_calls(prompts["C08"], c, tools) for c in (first_only, both, wrong_first)
    ]

    assert [(s.valid_first, s.valid_all) for s in scores] == [
        (True, False),
        (True, True),
        (False, False),
    ]
    assert [s.n_calls for s in scores] == [1, 2, 2]


@pytest.mark.parametrize(
    ("value", "valid"),
    [("07:30", True), ("7:30", True), ("07:30:00", True), ("7.30", False), ("half seven", False)],
)
def test_time_values_are_read_as_24_hour_clock_readings(value, valid, prompts, tools):
    score = toolcall.score_calls(
        prompts["C12"], [call("set_alarm", date="2026-11-12", time=value)], tools
    )

    assert score.valid_first is valid


def test_an_integral_float_counts_as_an_integer_but_a_boolean_does_not(prompts, tools):
    as_float = toolcall.score_calls(
        prompts["C01"], [call("set_timer", duration_seconds=600.0)], tools
    )
    as_bool = toolcall.score_calls(
        prompts["C01"], [call("set_timer", duration_seconds=True)], tools
    )

    assert as_float.valid_first
    assert not as_bool.valid_first


def test_a_brightness_outside_its_range_is_invalid(prompts, tools):
    calls = [call("set_lights", room="living room", action="set", brightness=130)]

    score = toolcall.score_calls(prompts["C02"], calls, tools)

    assert not score.valid_first
    assert any("above its maximum" in p for p in score.problems)


def test_score_as_row_gives_the_requests_jsonl_fields(prompts, tools):
    score = toolcall.score_calls(prompts["C01"], [call("set_timer", duration_seconds=600)], tools)

    row = score.as_row()

    assert row["tool_valid_first"] is True
    assert row["tool_valid_all"] is True
    assert row["tool_n_calls"] == 1
    assert row["tool_problems"] == []
