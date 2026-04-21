"""Canonical slop pattern catalogue + detection library.

Moved from scripts/check_ai_slop.py so backend code can import it
without reaching into scripts/. The CLI in scripts/check_ai_slop.py
now re-exports PATTERNS from here to keep a single source of truth.

Severity levels:
- critical: emoji, symbols, arrows. Must not appear in any output.
- warning: superlatives, hype words. Undesirable but not blocking.
- info: filler phrases, exclamation spam. Stylistic.

Severity floor semantics:
- severity_floor="critical" returns only critical findings
- severity_floor="warning" returns critical + warning
- severity_floor="info" returns all findings

Pattern overlap note:
High-code-point emoji characters such as the target, rocket, and party
glyphs are matched by both the `emoji` pattern (covering the unicode
ranges U+1F300-U+1F9FF, U+1FA70-U+1FAFF, and U+2600-U+27BF) and the
`emoji_symbols` pattern (a literal character set of commonly seen
decorative symbols). Therefore `detect()` may return two findings for
the same character position — one per pattern. This overlap is inherited
verbatim from the original script and is harmless for `strip_fixable()`
because both patterns use an empty-string replacement, but callers of
`detect()` should be aware that a raw count of findings is not the same
as a count of unique character offsets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

SeverityLevel = Literal["critical", "warning", "info"]

_SEVERITY_ORDER: dict[str, int] = {
    "critical": 0,
    "warning": 1,
    "info": 2,
}


@dataclass(frozen=True)
class SlopPattern:
    """Represents a detectable AI slop pattern."""

    name: str
    pattern: re.Pattern
    severity: str  # 'critical', 'warning', 'info'
    fixable: bool
    replacement: str = ""


@dataclass(frozen=True)
class SlopFinding:
    """A single match of a slop pattern in a text."""

    pattern_name: str
    severity: str
    matched_text: str
    replacement: str | None = None


# Canonical AI slop patterns — ported verbatim from scripts/check_ai_slop.py.
# Do not paraphrase or alter the regex bodies; they are battle-tested.
PATTERNS: list[SlopPattern] = [
    # CRITICAL: Emojis (absolute no-no)
    SlopPattern(
        name="emoji",
        pattern=re.compile(r"[\U0001F300-\U0001F9FF\U0001FA70-\U0001FAFF\U00002600-\U000027BF]"),
        severity="critical",
        fixable=True,
        replacement="",
    ),
    # CRITICAL: Common emoji-like unicode symbols
    SlopPattern(
        name="emoji_symbols",
        pattern=re.compile(r"[✓✗✅❌🎯🔧🚀💡⚠️📝📊🏗️🌟💪🤔👍👎🔥💯🎉🎊]"),
        severity="critical",
        fixable=True,
        replacement="",  # Just remove them
    ),
    # CRITICAL: Arrow symbols (use -> instead)
    SlopPattern(
        name="arrow_symbols",
        pattern=re.compile(r"[→←↔↑↓⇒⇐⇔⟹⟸⟺➜➝➞➟➠➡➢➣➤]"),
        severity="critical",
        fixable=True,
        replacement="->",
    ),
    # WARNING: Superlative adjectives (can be disabled with --critical-only)
    SlopPattern(
        name="superlatives",
        pattern=re.compile(
            r"\b(amazing|awesome|fantastic|incredible|wonderful|"
            r"outstanding|remarkable|extraordinary|exceptional|"
            r"phenomenal|spectacular|fabulous|magnificent|marvelous)\b",
            re.IGNORECASE,
        ),
        severity="warning",
        fixable=False,
    ),
    # WARNING: Over-hyped technical terms (can be disabled with --critical-only)
    # Note: "excellent", "revolutionary", "innovative", "groundbreaking" removed - sometimes valid
    SlopPattern(
        name="hype_words",
        pattern=re.compile(
            r"\b(seamless|cutting-edge|state-of-the-art|"
            r"world-class|enterprise-grade|battle-tested|"
            r"game-changing)\b",
            re.IGNORECASE,
        ),
        severity="warning",
        fixable=False,
    ),
    # INFO: Common AI filler phrases (can be disabled)
    SlopPattern(
        name="filler_phrases",
        pattern=re.compile(
            r"(let'?s dive (?:in|into)|first and foremost|"
            r"it'?s worth noting that|at the end of the day|moving forward)",
            re.IGNORECASE,
        ),
        severity="info",
        fixable=False,
    ),
    # INFO: Excessive exclamation marks
    SlopPattern(
        name="exclamation_spam",
        pattern=re.compile(r"!{3,}"),  # Changed from 2+ to 3+ (allow !!)
        severity="info",
        fixable=True,
        replacement="!",
    ),
]


def _severity_allowed(finding_severity: str, floor: str) -> bool:
    """Return True if finding_severity is at or above the floor threshold."""
    return _SEVERITY_ORDER[finding_severity] <= _SEVERITY_ORDER[floor]


class SlopDetector:
    """Detect and strip AI slop patterns from text.

    Uses the canonical PATTERNS catalogue by default. A custom pattern
    list can be injected for testing or specialised use cases.
    """

    def __init__(self, patterns: list[SlopPattern] | None = None) -> None:
        self._patterns = patterns if patterns is not None else PATTERNS

    def detect(
        self,
        text: str,
        severity_floor: SeverityLevel = "critical",
    ) -> list[SlopFinding]:
        """Return all findings at or above severity_floor.

        severity_floor="critical" -> critical only
        severity_floor="warning"  -> critical + warning
        severity_floor="info"     -> all findings
        """
        findings: list[SlopFinding] = []
        for pattern in self._patterns:
            if not _severity_allowed(pattern.severity, severity_floor):
                continue
            for match in pattern.pattern.finditer(text):
                findings.append(
                    SlopFinding(
                        pattern_name=pattern.name,
                        severity=pattern.severity,
                        matched_text=match.group(0),
                        replacement=pattern.replacement if pattern.fixable else None,
                    )
                )
        return findings

    def strip_fixable(self, text: str) -> str:
        """Mechanically remove/replace all fixable patterns.

        Non-fixable patterns (superlatives, hype_words, filler_phrases)
        are left untouched; they require human judgement.
        """
        result = text
        for pattern in self._patterns:
            if not pattern.fixable:
                continue
            result = pattern.pattern.sub(pattern.replacement, result)
        return result
