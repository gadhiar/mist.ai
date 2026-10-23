"""Deterministic replies, notices and the actions block over capability call outcomes.

The reply, notices and block about an action are templates over its outcome rather than model
text, so they cannot claim an action ran when it did not. They take only `ActionRecord`s, so the
fast path, proposal decisions and the model path can all use them, and the same records always
give the same text.
"""

from __future__ import annotations

from collections.abc import Sequence

from .contracts import ActionRecord, ToolOutcome

_PENDING = "pending_confirmation"
# The design appends a notice after the main-pass text for each pending, unavailable, error or
# invalid outcome; ok and denied outcomes get none.
_NOTICE_STATUSES = frozenset({"pending_confirmation", "unavailable", "error", "invalid_args"})

_BLOCK_HEADER = (
    "Actions this turn, already decided. Describe an action as done only when its line says done:"
)
_BLOCK_WORDS = {
    "ok": "done",
    _PENDING: "waiting for the user's yes or no; not done",
}
_DENIED_REASON = "turned off by policy"


def reply_for(records: Sequence[ActionRecord]) -> str:
    """Render the spoken reply for a turn answered by its actions alone.

    Each non-pending outcome gives one sentence, in record order. A turn's pending calls form
    one bundled proposal, so they are asked about once, last, with their summaries joined: the
    user answers a single yes or no.

    Returns:
        The sentences joined by spaces; empty when there are no records.
    """
    sentences = [_sentence(r) for r in records if r.outcome.status != _PENDING]
    pending = _pending_summary(records)
    if pending:
        sentences.append(f"Should I {pending}? Say yes or no.")
    return " ".join(sentences)


def notices_for(records: Sequence[ActionRecord]) -> tuple[str, ...]:
    """Render the notices appended after main-pass text on a turn that also replied.

    A notice restates a pending, unavailable, error or invalid outcome in parentheses, so the
    user hears it whatever the model wrote. Pending calls share one notice, last, as they share
    one proposal.

    Returns:
        One notice per such outcome, in record order, then the pending notice.
    """
    notices = [
        f"({_sentence(r)})"
        for r in records
        if r.outcome.status in _NOTICE_STATUSES and r.outcome.status != _PENDING
    ]
    pending = _pending_summary(records)
    if pending:
        notices.append(f"(Waiting for your yes or no: {pending}.)")
    return tuple(notices)


def actions_block(records: Sequence[ActionRecord]) -> str:
    """Render the actions block the main pass reads on a turn that also replied.

    Each line names the call, says its outcome in words, and appends the outcome's compact JSON,
    which carries the data the model may need to answer (a clock reading, for example) and, for
    a pending call, the note that it has not run.

    Returns:
        A header and one line per call, in record order; empty when there are no records.
    """
    if not records:
        return ""
    lines = [_BLOCK_HEADER]
    for r in records:
        words = _BLOCK_WORDS.get(r.outcome.status) or f"failed: {_failure_reason(r.outcome)}"
        lines.append(f"- {r.tool} ({r.args_summary}): {words}. {r.outcome.to_llm_json()}")
    return "\n".join(lines)


def _sentence(record: ActionRecord) -> str:
    outcome = record.outcome
    if outcome.status in ("ok", "invalid_args"):
        return outcome.summary
    if outcome.status == "unavailable":
        name = record.capability[:1].upper() + record.capability[1:]
        return f"{name} is unavailable{_reason_suffix(outcome)}."
    if outcome.status == "error":
        return f"That didn't work{_reason_suffix(outcome)}."
    if outcome.status == "denied":
        return "That action is turned off."
    raise ValueError(f"no single sentence for status {outcome.status!r}")


def _reason_suffix(outcome: ToolOutcome) -> str:
    return f": {_trim(outcome.reason)}" if outcome.reason else ""


def _failure_reason(outcome: ToolOutcome) -> str:
    if outcome.status == "denied":
        return _DENIED_REASON
    return _trim(outcome.reason or outcome.summary or outcome.status)


def _pending_summary(records: Sequence[ActionRecord]) -> str:
    # The same join as `ProposalStore.put`, so the question names the calls exactly as the
    # proposal's summary does; only a final full stop is trimmed, as the templates add one.
    parts = [r.args_summary for r in records if r.outcome.status == _PENDING]
    return _trim(", then ".join(parts))


def _trim(text: str) -> str:
    # Templates add their own full stop, so one already present would be doubled.
    return text.strip().rstrip(".")
