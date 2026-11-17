"""Tests for the deterministic replies, notices and actions block over call outcomes."""

from __future__ import annotations

import dataclasses
import re

import pytest

from backend.capabilities.contracts import ActionRecord, OutcomeStatus, ToolOutcome
from backend.capabilities.proposals import ProposalStore, ProposedCall
from backend.capabilities.replies import (
    OK_TEMPLATES,
    READ_ITEMS_SPOKEN,
    READ_TEMPLATES,
    SPOKEN_WORDS_MAX,
    NoAddress,
    actions_block,
    is_brief_sentence,
    notices_for,
    ok_summary,
    render_read,
    reply_for,
    with_address,
)
from tests.unit.capabilities.fakes import EmptyArgs, FakeClock


def record(
    status: OutcomeStatus,
    summary: str = "",
    *,
    reason: str | None = None,
    tool: str = "lights.power",
    capability: str = "lights",
    args_summary: str = "turn off the bedroom lights",
    proposal_id: str | None = None,
    data: dict[str, object] | None = None,
    call_id: str = "call-1",
) -> ActionRecord:
    return ActionRecord(
        tool_call_id=call_id,
        tool=tool,
        capability=capability,
        risk="physical",
        args_summary=args_summary,
        outcome=ToolOutcome(
            status=status, summary=summary, data=data, proposal_id=proposal_id, reason=reason
        ),
        duration_ms=12,
    )


OK = record("ok", "Bedroom lights off.")
PENDING = record("pending_confirmation", "Waiting for confirmation.", proposal_id="p-1")
UNAVAILABLE = record("unavailable", "Lights are unavailable.", reason="unreachable")
ERROR = record("error", "The call failed.", reason="timed out after 8 s")
INVALID = record("invalid_args", "I don't know a light called garage. I know bedroom, kitchen.")
DENIED = record("denied", "Denied by policy.")
PLAY_JAZZ = record(
    "pending_confirmation",
    "Waiting for confirmation.",
    tool="music.play",
    capability="music",
    args_summary="play jazz",
    proposal_id="p-1",
    call_id="call-2",
)
CLOCK = record(
    "ok",
    "It's 7:42 PM, Thursday.",
    tool="clock.now",
    capability="clock",
    args_summary="tell the time",
    data={"iso": "2026-01-08T19:42:00"},
    call_id="call-3",
)

ALL = [OK, PENDING, UNAVAILABLE, ERROR, INVALID, DENIED]


def fresh_copies(records: list[ActionRecord]) -> list[ActionRecord]:
    """Equal records as new objects, so a byte-identical check cannot pass on identity alone."""
    return [dataclasses.replace(r, outcome=dataclasses.replace(r.outcome)) for r in records]


class TestReplyFor:
    def test_ok_gives_the_summary(self):
        assert reply_for([OK]) == "Bedroom lights off."

    def test_pending_asks_with_the_call_summary(self):
        assert reply_for([PENDING]) == "Should I turn off the bedroom lights? Say yes or no."

    def test_unavailable_names_the_capability_and_reason(self):
        assert reply_for([UNAVAILABLE]) == "Lights is unavailable: unreachable."

    def test_unavailable_without_reason(self):
        assert reply_for([record("unavailable", "x")]) == "Lights is unavailable."

    def test_error_gives_the_reason(self):
        assert reply_for([ERROR]) == "That didn't work: timed out after 8 s."

    def test_error_without_reason(self):
        assert reply_for([record("error", "x")]) == "That didn't work."

    def test_invalid_args_gives_the_summary(self):
        assert reply_for([INVALID]) == (
            "I don't know a light called garage. I know bedroom, kitchen."
        )

    def test_denied(self):
        assert reply_for([DENIED]) == "That action is turned off."

    def test_bundle_asks_once_with_the_joined_summary(self):
        text = reply_for([PENDING, PLAY_JAZZ])

        assert text == "Should I turn off the bedroom lights, then play jazz? Say yes or no."
        assert text.count("Should I") == 1

    def test_three_pending_calls_join_with_then(self):
        third = record("pending_confirmation", "x", args_summary="set an alarm for 7")

        assert reply_for([PENDING, PLAY_JAZZ, third]) == (
            "Should I turn off the bedroom lights, then play jazz, then set an alarm for 7?"
            " Say yes or no."
        )

    @pytest.mark.asyncio
    async def test_joined_summary_matches_the_proposal_summary(self):
        store = ProposalStore(FakeClock(), 60.0, lambda proposal: None)
        calls = [
            ProposedCall(r.capability, r.tool, EmptyArgs(), r.args_summary, r.risk)
            for r in (PENDING, PLAY_JAZZ)
        ]
        proposal = store.put("session-1", calls)
        store.pop("session-1", proposal.proposal_id)

        assert reply_for([PENDING, PLAY_JAZZ]) == f"Should I {proposal.summary}? Say yes or no."
        assert notices_for([PENDING, PLAY_JAZZ]) == (
            f"(Waiting for your yes or no: {proposal.summary}.)",
        )

    def test_mixed_outcomes_in_order_with_the_question_last(self):
        assert reply_for([PENDING, CLOCK, ERROR, PLAY_JAZZ]) == (
            "It's 7:42 PM, Thursday. That didn't work: timed out after 8 s."
            " Should I turn off the bedroom lights, then play jazz? Say yes or no."
        )

    def test_trailing_period_in_reason_is_not_doubled(self):
        assert reply_for([record("error", "x", reason="timed out.")]) == (
            "That didn't work: timed out."
        )

    def test_no_records(self):
        assert reply_for([]) == ""

    def test_byte_identical_for_the_same_records(self):
        assert reply_for(ALL) == reply_for(fresh_copies(ALL))


class TestNoticesFor:
    def test_ok_gives_no_notice(self):
        assert notices_for([OK]) == ()

    def test_pending(self):
        assert notices_for([PENDING]) == (
            "(Waiting for your yes or no: turn off the bedroom lights.)",
        )

    def test_unavailable(self):
        assert notices_for([UNAVAILABLE]) == ("(Lights is unavailable: unreachable.)",)

    def test_error(self):
        assert notices_for([ERROR]) == ("(That didn't work: timed out after 8 s.)",)

    def test_invalid_args(self):
        assert notices_for([INVALID]) == (
            "(I don't know a light called garage. I know bedroom, kitchen.)",
        )

    def test_denied_gives_no_notice(self):
        assert notices_for([DENIED]) == ()

    def test_bundle_waits_once_with_the_joined_summary(self):
        assert notices_for([PENDING, OK, PLAY_JAZZ]) == (
            "(Waiting for your yes or no: turn off the bedroom lights, then play jazz.)",
        )

    def test_order_follows_the_records_with_the_wait_last(self):
        assert notices_for([PENDING, ERROR, UNAVAILABLE]) == (
            "(That didn't work: timed out after 8 s.)",
            "(Lights is unavailable: unreachable.)",
            "(Waiting for your yes or no: turn off the bedroom lights.)",
        )

    def test_byte_identical_for_the_same_records(self):
        assert notices_for(ALL) == notices_for(fresh_copies(ALL))


class TestActionsBlock:
    @pytest.mark.parametrize(
        ("rec", "words"),
        [
            (OK, "done"),
            (PENDING, "waiting for the user's yes or no; not done"),
            (UNAVAILABLE, "failed: unreachable"),
            (ERROR, "failed: timed out after 8 s"),
            (INVALID, "failed: I don't know a light called garage. I know bedroom, kitchen"),
            (DENIED, "failed: turned off by policy"),
        ],
        ids=["ok", "pending", "unavailable", "error", "invalid_args", "denied"],
    )
    def test_one_line_per_status(self, rec, words):
        block = actions_block([rec])

        lines = block.splitlines()
        assert len(lines) == 2
        assert lines[1] == (
            f"- lights.power (turn off the bedroom lights): {words}. {rec.outcome.to_llm_json()}"
        )

    def test_header_tells_the_model_what_counts_as_done(self):
        header = actions_block([OK]).splitlines()[0]

        assert "done" in header
        assert header.endswith(":")

    def test_lists_every_call_in_order(self):
        block = actions_block([CLOCK, PENDING, PLAY_JAZZ])

        lines = block.splitlines()[1:]
        assert [line.split(" ")[1] for line in lines] == ["clock.now", "lights.power", "music.play"]
        assert '"data":{"iso":"2026-01-08T19:42:00"}' in lines[0]
        assert '"note":"not run' in lines[1]

    def test_no_records_gives_an_empty_block(self):
        assert actions_block([]) == ""

    def test_byte_identical_for_the_same_records(self):
        assert actions_block(ALL) == actions_block(fresh_copies(ALL))


# -- Brief by voice (T25) --------------------------------------------------------------------


def sentences(text: str) -> list[str]:
    return re.split(r"(?<=[.?!])\s+", text.strip())


class TestBriefSentence:
    @pytest.mark.parametrize(
        "text",
        [
            "Paused.",
            "Alarm set for 7:00 AM tomorrow.",
            "Is it on?",
            "One two three four five six seven eight nine ten eleven twelve.",
        ],
    )
    def test_one_sentence_of_at_most_twelve_words_is_brief(self, text):
        assert is_brief_sentence(text)

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "Alarm set",
            "Alarm set. Sleep well.",
            "One two three four five six seven eight nine ten eleven twelve thirteen.",
        ],
    )
    def test_no_end_two_sentences_or_thirteen_words_is_not_brief(self, text):
        assert not is_brief_sentence(text)

    def test_the_word_limit_is_twelve(self):
        assert SPOKEN_WORDS_MAX == 12


# Arguments for every ok template; a template added without a fixture fails the key check.
OK_FIXTURES: dict[str, dict[str, object]] = {
    "alarms.set": {"when": "7:00 AM tomorrow"},
    "alarms.set.already": {"when": "7:00 AM Monday"},
    "alarms.set.moved": {"when": "6:30 AM Thursday"},
    "alarms.cancel": {"when": "7:00 AM tomorrow"},
    "alarms.stop": {},
    "alarms.snooze": {"minutes": 9},
    "alarms.dismiss": {},
    "timers.start": {"duration": "10 minutes"},
    "timers.cancel": {},
    "timers.cancel.none": {},
    "timers.stop": {},
    "lights.power": {"target": "bedroom", "state": "off"},
    "lights.adjust": {"target": "kitchen", "percent": 30},
    "music.play": {"title": "Kind of Blue", "artist": "Miles Davis"},
    "music.pause": {},
    "music.resume": {},
    "music.volume": {"percent": 40},
    "music.idle": {},
}


class TestOkTemplates:
    def test_every_ok_template_has_a_fixture(self):
        assert set(OK_TEMPLATES) == set(OK_FIXTURES)

    @pytest.mark.parametrize("key", sorted(OK_FIXTURES))
    def test_every_ok_template_renders_one_sentence_of_at_most_twelve_words(self, key):
        summary = ok_summary(key, **OK_FIXTURES[key])

        assert is_brief_sentence(summary), summary
        assert reply_for([record("ok", summary)]) == summary

    def test_ok_summary_fills_the_template_and_starts_with_a_capital(self):
        assert ok_summary("lights.power", target="bedroom", state="off") == "Bedroom lights off."
        assert ok_summary("alarms.set", when="7:00 AM tomorrow") == (
            "Alarm set for 7:00 AM tomorrow."
        )

    def test_an_unknown_key_raises(self):
        with pytest.raises(KeyError):
            ok_summary("lights.explode")


# Fields and seven item phrases for every read template.
READ_FIXTURES: dict[str, tuple[dict[str, object], list[str]]] = {
    "alarms.list": (
        {},
        [
            "7:00 AM tomorrow, gym",
            "6:30 AM Monday",
            "8:00 AM Tuesday, school run",
            "9:15 AM Wednesday",
            "7:45 AM Thursday",
            "10:00 AM Saturday",
            "11:30 AM Sunday",
        ],
    ),
    "timers.list": (
        {},
        [
            "pasta, 4 minutes left",
            "tea, 2 minutes left",
            "laundry, 41 minutes left",
            "oven, 12 minutes left",
            "10 minute timer, 9 minutes left",
            "bread, 1 hour left",
            "plants, 3 hours left",
        ],
    ),
    "clock.now": (
        {"time": "7:42 PM", "date": "Thursday October 23"},
        [
            "London, 12:42 AM Friday",
            "New York, 7:42 PM",
            "Tokyo, 8:42 AM Friday",
            "Sydney, 10:42 AM Friday",
            "Delhi, 5:12 AM Friday",
            "Paris, 1:42 AM Friday",
            "Denver, 5:42 PM",
        ],
    ),
    "lights.status": (
        {},
        [
            "bedroom, on at 40 percent",
            "kitchen, off",
            "hall, on",
            "office, off",
            "porch, on at 10 percent",
            "garage, off",
            "lounge, on",
        ],
    ),
    "music.now_playing": (
        {"title": "So What", "artist": "Miles Davis"},
        [
            "Freddie Freeloader by Miles Davis",
            "Blue in Green by Miles Davis",
            "All Blues by Miles Davis",
            "Flamenco Sketches by Miles Davis",
            "Naima by John Coltrane",
            "Giant Steps by John Coltrane",
            "Mr. PC by John Coltrane",
        ],
    ),
}


class TestReadTemplates:
    def test_read_templates_name_the_five_reads(self):
        assert (
            set(READ_TEMPLATES)
            == set(READ_FIXTURES)
            == {
                "alarms.list",
                "timers.list",
                "clock.now",
                "lights.status",
                "music.now_playing",
            }
        )

    def test_a_read_speaks_at_most_five_items(self):
        assert READ_ITEMS_SPOKEN == 5

    @pytest.mark.parametrize("count", [0, 1, 5, 7])
    @pytest.mark.parametrize("tool", sorted(READ_FIXTURES))
    def test_a_read_leads_briefly_then_speaks_up_to_five_items(self, tool, count):
        fields, items = READ_FIXTURES[tool]

        text = render_read(tool, items[:count], **fields)

        lead, *spoken = sentences(text)
        assert is_brief_sentence(lead), lead
        assert len(spoken) == min(count, READ_ITEMS_SPOKEN)
        for sentence, item in zip(spoken, items, strict=False):
            assert sentence.lower().startswith(item.lower()), (sentence, item)
        if count > READ_ITEMS_SPOKEN:
            assert text.endswith(f"and {count - READ_ITEMS_SPOKEN} more on screen.")
        else:
            assert "more on screen" not in text

    def test_seven_alarms_speak_five_and_end_with_two_more_on_screen(self):
        _, items = READ_FIXTURES["alarms.list"]

        text = render_read("alarms.list", items)

        assert text == (
            "You have 7 alarms. 7:00 AM tomorrow, gym. 6:30 AM Monday. 8:00 AM Tuesday, school"
            " run. 9:15 AM Wednesday. 7:45 AM Thursday, and 2 more on screen."
        )
        assert text.endswith("and 2 more on screen.")
        assert "10:00 AM Saturday" not in text

    def test_no_alarms_is_the_lead_alone(self):
        assert render_read("alarms.list", []) == "You have no alarms set."

    def test_the_lead_is_filled_from_the_fields(self):
        assert render_read("clock.now", [], time="7:42 PM", date="Thursday October 23") == (
            "It's 7:42 PM, Thursday October 23."
        )

    def test_an_item_phrase_starts_its_sentence_with_a_capital_and_one_full_stop(self):
        assert render_read("timers.list", ["pasta, 4 minutes left."]) == (
            "1 timer is running. Pasta, 4 minutes left."
        )

    def test_an_unknown_read_raises(self):
        with pytest.raises(KeyError):
            render_read("weather.now", [])


class _Boss:
    def address_form(self) -> str | None:
        return "boss"


class TestAddress:
    def test_no_address_has_no_form(self):
        assert NoAddress().address_form() is None

    def test_the_first_reply_of_a_session_ends_with_the_address_form(self):
        assert with_address("Bedroom lights off.", _Boss(), first_in_session=True) == (
            "Bedroom lights off, boss."
        )

    def test_a_later_reply_has_no_address_form(self):
        assert with_address("Bedroom lights off.", _Boss(), first_in_session=False) == (
            "Bedroom lights off."
        )

    def test_the_form_goes_before_the_last_sentences_question_mark(self):
        reply = reply_for([PENDING])

        assert with_address(reply, _Boss(), first_in_session=True) == (
            "Should I turn off the bedroom lights? Say yes or no, boss."
        )

    def test_a_reply_without_a_full_stop_gains_the_form_at_its_end(self):
        assert with_address("Done", _Boss(), first_in_session=True) == "Done, boss"

    def test_an_empty_reply_stays_empty(self):
        assert with_address("", _Boss(), first_in_session=True) == ""

    @pytest.mark.parametrize("first", [True, False])
    def test_no_address_leaves_every_reply_byte_identical(self, first):
        replies = [reply_for([r]) for r in ALL] + [
            reply_for([OK, PENDING, PLAY_JAZZ]),
            render_read("alarms.list", READ_FIXTURES["alarms.list"][1]),
        ]

        for reply in replies:
            assert with_address(reply, NoAddress(), first_in_session=first) == reply
