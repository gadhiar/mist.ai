"""Deterministic replies, notices and the actions block over capability call outcomes.

The reply, notices and block about an action are templates over its outcome rather than model
text, so they cannot claim an action ran when it did not. They take only `ActionRecord`s, so the
fast path, proposal decisions and the model path can all use them, and the same records always
give the same text.

Brief by voice, depth on request and on screen:

- an ok summary of an action is one spoken sentence of at most `SPOKEN_WORDS_MAX` words, and
  `OK_TEMPLATES` holds the shared ones, which capabilities render through `ok_summary`;
- a read (`READ_TEMPLATES`) leads with one such sentence, speaks at most `READ_ITEMS_SPOKEN`
  items, and says how many more are on screen, where the outcome's `data` shows them all;
- the address form a `ReplyRegister` supplies ends a session's first templated reply only.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

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
    # proposal's summary does. `_trim` drops surrounding whitespace and trailing full stops,
    # because the templates add their own.
    parts = [r.args_summary for r in records if r.outcome.status == _PENDING]
    return _trim(", then ".join(parts))


def _trim(text: str) -> str:
    # Templates add their own full stop, so one already present would be doubled.
    return text.strip().rstrip(".")


# -- Brief by voice ----------------------------------------------------------------------------

SPOKEN_WORDS_MAX = 12
"""The most words in an ok summary of an action, and in a read's lead sentence."""

READ_ITEMS_SPOKEN = 5
"""The most items a read speaks; the rest are counted and left to the screen."""

_SENTENCE_ENDS = ".?!"
# A sentence end followed by more text starts a second sentence.
_INNER_SENTENCE_END = re.compile(r"[.?!]\s")


def is_brief_sentence(text: str, max_words: int = SPOKEN_WORDS_MAX) -> bool:
    """Return whether `text` is one sentence of at most `max_words` words.

    The check is the voice rule's, so capabilities' own tests can hold their summaries to it.
    A sentence ends with a full stop, question mark or exclamation mark; such a mark followed
    by more text starts a second one, so a summary avoids abbreviations with full stops.
    """
    stripped = text.strip()
    if not stripped or stripped[-1] not in _SENTENCE_ENDS:
        return False
    if _INNER_SENTENCE_END.search(stripped):
        return False
    return len(stripped.split()) <= max_words


OK_TEMPLATES: Mapping[str, str] = MappingProxyType(
    {
        "alarms.set": "Alarm set for {when}.",
        "alarms.set.already": "Your {when} alarm is already set.",
        "alarms.set.moved": "Alarm moved to {when}.",
        "alarms.cancel": "Alarm for {when} cancelled.",
        "alarms.stop": "Alarm stopped.",
        "alarms.snooze": "Snoozed for {minutes} minutes.",
        "alarms.dismiss": "Missed alarm dismissed.",
        "timers.start": "Timer set for {duration}.",
        "timers.cancel": "Timer cancelled.",
        "timers.cancel.none": "No timer is running.",
        "timers.stop": "Timer stopped.",
        "lights.power": "{target} lights {state}.",
        "lights.adjust": "{target} lights set to {percent} percent.",
        "music.play": "Playing {title} by {artist}.",
        "music.pause": "Paused.",
        "music.resume": "Resumed.",
        "music.volume": "Volume {percent} percent.",
        "music.idle": "Nothing is playing.",
    }
)
"""The ok summaries of actions, keyed by tool and, where a tool has several, by case.

Each renders as one sentence of at most `SPOKEN_WORDS_MAX` words for ordinary arguments. They
live here rather than in each capability so the voice rule is tested over all of them at once.
"""


def ok_summary(key: str, **fields: object) -> str:
    """Render the ok template `key` with `fields`, starting with a capital letter.

    Raises:
        KeyError: `key` names no template, or the template names a field not given.
    """
    return _capitalised(OK_TEMPLATES[key].format(**fields))


@dataclass(frozen=True, slots=True)
class ReadTemplate:
    """The lead sentence of one read, chosen by how many items it has.

    Each may name `{count}` and fields the capability passes to `render_read`.

    Attributes:
        none: The lead for no items.
        one: The lead for one item.
        many: The lead for two or more.
    """

    none: str
    one: str
    many: str


READ_TEMPLATES: Mapping[str, ReadTemplate] = MappingProxyType(
    {
        "alarms.list": ReadTemplate(
            none="You have no alarms set.",
            one="You have 1 alarm.",
            many="You have {count} alarms.",
        ),
        "timers.list": ReadTemplate(
            none="No timers are running.",
            one="1 timer is running.",
            many="{count} timers are running.",
        ),
        "clock.now": ReadTemplate(
            none="It's {time}, {date}.",
            one="It's {time}, {date}.",
            many="It's {time}, {date}.",
        ),
        "lights.status": ReadTemplate(
            none="I can't see any lights.",
            one="You have 1 light.",
            many="You have {count} lights.",
        ),
        "music.now_playing": ReadTemplate(
            none="Playing {title} by {artist}.",
            one="Playing {title} by {artist}, 1 track up next.",
            many="Playing {title} by {artist}, {count} tracks up next.",
        ),
    }
)
"""The reads that answer a request for depth, keyed by tool."""


def render_read(tool: str, items: Sequence[str], **fields: object) -> str:
    """Render a read: its lead, one sentence per item up to `READ_ITEMS_SPOKEN`, then the rest.

    Args:
        tool: A key of `READ_TEMPLATES`.
        items: One short phrase per item, in the order to speak them ("7:00 AM tomorrow,
            gym"). Each becomes a sentence of its own.
        fields: The lead's fields other than `count`, which is `len(items)`.

    Returns:
        The lead alone with no items. With more than `READ_ITEMS_SPOKEN`, the last spoken item
        ends ", and N more on screen.", N being the items not spoken.

    Raises:
        KeyError: `tool` names no read template, or the lead names a field not given.
    """
    template = READ_TEMPLATES[tool]
    count = len(items)
    lead = template.none if count == 0 else template.one if count == 1 else template.many
    spoken = [_capitalised(_trim(item)) for item in items[:READ_ITEMS_SPOKEN]]
    sentences = [_capitalised(lead.format(count=count, **fields))]
    sentences.extend(f"{phrase}." for phrase in spoken)
    unspoken = count - len(spoken)
    if unspoken:
        sentences[-1] = f"{spoken[-1]}, and {unspoken} more on screen."
    return " ".join(sentences)


def _capitalised(text: str) -> str:
    return text[:1].upper() + text[1:]


# -- The address form --------------------------------------------------------------------------


class ReplyRegister(Protocol):
    """Supplies how MIST addresses her owner ("boss" or similar), from the persona."""

    def address_form(self) -> str | None:
        """Return the address form, or None when she uses none."""
        ...


class NoAddress:
    """The register until the persona's loader exists: no address form, so replies are as is."""

    def address_form(self) -> str | None:
        """Return None."""
        return None


def with_address(reply: str, register: ReplyRegister, *, first_in_session: bool) -> str:
    """End a session's first templated reply with the register's address form.

    The form goes before the reply's final full stop, question mark or exclamation mark
    ("Bedroom lights off, boss."), or after the reply when it has none. Only the first reply
    of a session is addressed, so the form is heard once rather than on every action.

    Returns:
        `reply` unchanged when it is not the first, when it is blank, or when the register
        has no form; otherwise the addressed reply.
    """
    if not first_in_session or not reply.strip():
        return reply
    form = register.address_form()
    if not form:
        return reply
    body = reply.rstrip()
    if body[-1] in _SENTENCE_ENDS:
        return f"{body[:-1]}, {form}{body[-1]}"
    return f"{body}, {form}"
