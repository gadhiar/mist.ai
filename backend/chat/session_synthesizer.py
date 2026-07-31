"""Session-end synthesis, sourced from the event store rather than live memory.

R1.3.1: synthesis must run both at session end and at startup catch-up. The
catch-up path has no in-memory session -- the process that held it is gone --
so synthesis takes turns as input and both callers feed it from the event
store. One path, no drift.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from backend.llm import LLMRequest, StreamingLLMProvider

logger = logging.getLogger(__name__)

_MIN_TURNS_FOR_SYNTHESIS = 1

# Tolerates the small model drifting from the exact `TITLE: ...` convention --
# markdown emphasis around the marker (`**TITLE:**`) and stray whitespace
# before the colon (`TITLE : ...`) are both still recognized as a title
# attempt, so the line is consumed and never leaks into the body.
_TITLE_LINE_RE = re.compile(r"^[*_]{0,2}\s*TITLE\s*:\s*[*_]{0,2}\s*(.*)$", re.IGNORECASE)

_PROMPT_TEMPLATE = (
    "You are MIST writing a session-end summary for the user's persistent "
    "memory vault. The conversation just ended. Read the transcript and "
    "produce a title followed by a concise markdown body with the four "
    "subsections below. If a subsection has no content, write `(none)`.\n\n"
    "Start with a single line of the exact form:\n"
    "TITLE: <a short human-readable session title, under 60 characters>\n\n"
    "Then a blank line, then the body. Do NOT include a leading `## Summary` "
    "header -- only the subsection content.\n\n"
    "### What Was Accomplished\n"
    "<bullet list of substantive accomplishments, each one line>\n\n"
    "### Decisions Made\n"
    "<bullet list of explicit decisions, each one line, or (none)>\n\n"
    "### Next Actions\n"
    "<bullet list of action items the user or MIST committed to, each one "
    "line, or (none)>\n\n"
    "### Context for Next Session\n"
    "<one to three sentences of prose summarizing what would be useful to "
    "remember when picking this conversation back up>\n\n"
    "---\n\n"
    "TRANSCRIPT:\n\n"
)


@dataclass(frozen=True, slots=True)
class SessionSynthesis:
    """A synthesized session note: the title and the body sections."""

    title: str
    body: str


class SessionSynthesizer:
    """Turns a session's transcript into vault-ready prose.

    Deliberately knows nothing about the vault, the event store, or live
    session state -- it is a pure transform so both the session-end path and
    the startup catch-up path can share it without either owning the other.
    """

    def __init__(
        self, llm_provider: StreamingLLMProvider, temperature: float, max_tokens: int
    ) -> None:
        self._llm = llm_provider
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def is_ready(self) -> bool:
        """Report whether the backing LLM is currently reachable.

        Delegates to `StreamingLLMProvider.health_check` (an existing
        interface method -- `LlamaServerProvider` genuinely probes the
        `/health` endpoint; the base class and `OllamaProvider` have their
        own implementations). Exists so a caller that must not commit to a
        synthesis failure while the backend is merely cold at boot (startup
        catch-up) can check readiness first, without depending on `_llm`
        directly.
        """
        try:
            return await self._llm.health_check()
        except Exception as exc:  # noqa: BLE001 -- readiness check is best-effort
            logger.warning("LLM readiness check failed (non-fatal): %s", exc)
            return False

    async def synthesize(self, turns: list[dict]) -> SessionSynthesis | None:
        """Synthesize a session note body from its turns.

        Returns None when there are no turns at all, or the model call fails.
        A None result means "write no note", which is the correct outcome for
        an empty session -- see the catch-up gating in
        `backend/vault/session_catchup.py` for the separate "was anything
        worth remembering" filter, which composes with this one rather than
        duplicating it.
        """
        if len(turns) < _MIN_TURNS_FOR_SYNTHESIS:
            return None

        transcript = "".join(
            f"**USER:** {t.get('user_utterance', '')}\n"
            f"**MIST:** {t.get('system_response', '')}\n\n"
            for t in turns
        )

        request = LLMRequest(
            messages=[{"role": "user", "content": _PROMPT_TEMPLATE + transcript}],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            top_p=0.9,
        )

        try:
            response = await self._llm.invoke(request)
        except Exception as exc:  # noqa: BLE001 -- synthesis is best-effort
            logger.warning("Session synthesis LLM call failed (non-fatal): %s", exc)
            return None

        return self._parse(getattr(response, "content", "") or "")

    @staticmethod
    def _parse(completion: str) -> SessionSynthesis | None:
        """Split `TITLE: ...` off the front of the completion.

        The model is small and will sometimes drift from the exact convention
        (markdown emphasis around the marker, stray whitespace before the
        colon, or omitting the line entirely) -- a near-miss still counts as
        a title attempt and must never leak into the body, and a missing
        title falls back rather than discarding a good body.
        """
        text = completion.strip()
        if not text:
            return None

        title = "Conversation"
        body = text

        first, _, rest = text.partition("\n")
        match = _TITLE_LINE_RE.match(first.strip())
        if match:
            candidate = match.group(1).strip()
            if candidate:
                title = candidate
            body = rest.strip()

        if not body:
            return None

        return SessionSynthesis(title=title, body=body)
