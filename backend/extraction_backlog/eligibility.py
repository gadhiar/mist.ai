"""Which logged turns the memory writers leave out, and why (T31).

One predicate, read by every memory writer: the extraction backlog
(`store.BacklogStore.scan` and `fill_scan`), self-reflection
(`backend.knowledge.curation.self_reflection.SelfReflectionJob.run`), the live
session note (`ConversationHandler.end_session`) and session catch-up
(`backend.vault.session_catchup.SessionNoteCatchup._synthesize_one`). Keeping the
comparison here, and nowhere else in `backend/`, means a later policy changes in
one place.

A turn a capability answered alone (`answered_by == "capability"`) holds a
command and a template, not knowledge. A model turn, and a turn recorded before
`answered_by` existed (NULL), stay eligible.

This module imports only `backend.knowledge.extraction_cache`, which imports only
the standard library, and the package `__init__` resolves its names lazily, so
the chat, curation and vault modules import it without a cycle.
"""

from __future__ import annotations

from backend.knowledge.extraction_cache import SKIP_CAPABILITY_ONLY


def policy_skip_reason(answered_by: str | None) -> str | None:
    """The skip reason the policy gives a turn, or None when the turn is eligible.

    Args:
        answered_by: The turn's `answered_by` column: "model", "capability",
            or None for a turn recorded before the column existed.

    Returns:
        `SKIP_CAPABILITY_ONLY` for a capability-only turn, else None.
    """
    if answered_by == "capability":
        return SKIP_CAPABILITY_ONLY
    return None
