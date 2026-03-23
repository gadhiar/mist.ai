"""Internal knowledge signal detection (pre-check, no LLM).

Scans a conversation turn for signals that indicate MIST should update
its self-model. This is a lightweight keyword/pattern check that gates
the more expensive LLM-based internal derivation call.
"""

import re
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class SignalDetectionResult:
    """Result of signal detection on a conversation turn."""

    has_signals: bool
    signal_types: frozenset[str] = field(default_factory=frozenset)
    matched_patterns: tuple[str, ...] = ()


# Patterns that suggest user feedback about MIST's behavior
_FEEDBACK_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(i like when you|you('re| are) good at|great job|well done|good job)\b", re.I),
    re.compile(r"\b(stop|don'?t|quit|please don'?t)\b.*\b(\w+ing)\b", re.I),
    re.compile(r"\b(too (verbose|wordy|long|short|brief|vague|technical|simple))\b", re.I),
    re.compile(r"\b(i (like|love|hate|dislike|prefer) (when |how |that )?you)\b", re.I),
]

# Patterns that suggest the user is correcting MIST
_CORRECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(no[,.]?\s+(that'?s|it'?s|you'?re) (wrong|incorrect|not right))\b", re.I),
    re.compile(r"\b(actually[,.]?\s+(it'?s|that'?s|the answer is))\b", re.I),
    re.compile(r"\b(that'?s not what i (meant|asked|said))\b", re.I),
    re.compile(r"\b(you (misunderstood|got it wrong|missed))\b", re.I),
]

# Patterns that suggest the user is expressing a preference for MIST's behavior
_PREFERENCE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(i prefer|i'?d rather|i want you to|please (always|never))\b", re.I),
    re.compile(
        r"\b(from now on|going forward|in the future)[,.]?\s+(please|always|never|don'?t)\b", re.I
    ),
    re.compile(r"\b(can you (always|never|start|stop))\b", re.I),
]

# Keywords suggesting capability acknowledgment (combined with tool_calls)
_CAPABILITY_KEYWORDS: list[str] = [
    "good job",
    "well done",
    "great",
    "perfect",
    "exactly",
    "that helped",
    "useful",
    "thanks for finding",
]


class SignalDetector:
    """Detects internal knowledge signals in conversation turns.

    This is the gate for Stage 9 (internal derivation). If no signals
    are detected, the LLM call is skipped entirely.
    """

    def detect(
        self,
        utterance: str,
        tool_calls: list[dict] | None = None,
        assistant_response: str | None = None,
    ) -> SignalDetectionResult:
        """Check a conversation turn for internal knowledge signals.

        Args:
            utterance: The user's message text.
            tool_calls: Tool calls made during this turn (if any).
            assistant_response: MIST's response (for context).

        Returns:
            SignalDetectionResult with signal types found.
        """
        if len(utterance.split()) < 3:
            return SignalDetectionResult(has_signals=False)

        signal_types: set[str] = set()
        matched: list[str] = []

        # Check feedback patterns
        for pattern in _FEEDBACK_PATTERNS:
            match = pattern.search(utterance)
            if match:
                signal_types.add("feedback")
                matched.append(f"feedback:{match.group()}")
                break

        # Check correction patterns
        for pattern in _CORRECTION_PATTERNS:
            match = pattern.search(utterance)
            if match:
                signal_types.add("correction")
                matched.append(f"correction:{match.group()}")
                break

        # Check preference patterns
        for pattern in _PREFERENCE_PATTERNS:
            match = pattern.search(utterance)
            if match:
                signal_types.add("preference")
                matched.append(f"preference:{match.group()}")
                break

        # Check capability evidence (requires tool_calls context)
        if tool_calls:
            utterance_lower = utterance.lower()
            for keyword in _CAPABILITY_KEYWORDS:
                if keyword in utterance_lower:
                    signal_types.add("capability")
                    matched.append(f"capability:{keyword}")
                    break

        return SignalDetectionResult(
            has_signals=len(signal_types) > 0,
            signal_types=frozenset(signal_types),
            matched_patterns=tuple(matched),
        )
