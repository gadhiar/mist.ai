"""Tests for the deterministic replies, notices and actions block over call outcomes."""

from __future__ import annotations

import dataclasses

import pytest

from backend.capabilities.contracts import ActionRecord, OutcomeStatus, ToolOutcome
from backend.capabilities.proposals import ProposalStore, ProposedCall
from backend.capabilities.replies import actions_block, notices_for, reply_for
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
