"""Dispatcher configuration, read from `MIST_EXTRACTION_*` environment variables.

Lives in this package rather than on `KnowledgeConfig` because it configures
the backlog and the service seam, not the knowledge pipeline, and so that the
extraction-backlog code has one place to read it from.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

InferenceMode = Literal["service", "off"]

DEFAULT_SERVICE_URL = "http://mist-extraction:8090"


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


@dataclass(frozen=True, slots=True)
class DispatcherSettings:
    """How the extraction dispatcher runs.

    Attributes:
        mode: `service` dispatches to the extraction service; `off` logs turns
            and dispatches nothing (state `disabled`). There is no mode that
            extracts on the backend's own chat model.
        service_url: Root URL of the extraction service.
        max_attempts: Job-attributable failures after which a turn is
            dead-lettered as an `extraction_failed` skip.
        backoff_base_s: First retry delay after a job-attributable failure;
            doubles per failure.
        backoff_cap_s: Upper bound on any retry or reconnect delay.
        request_timeout_s: Client-side timeout for one `/v1/extract` call.
        idle_poll_s: How often an idle dispatcher rescans the log without a
            `wake()` (catches turns logged by another process).
        stall_recheck_s: How often a stalled dispatcher re-reads `/v1/info`.
        history_messages: Conversation-history messages sent with a job; the
            same count `ConversationHandler.handle_message` passes as
            `max_history` (default 10).
    """

    mode: InferenceMode = "service"
    service_url: str = DEFAULT_SERVICE_URL
    max_attempts: int = 5
    backoff_base_s: float = 2.0
    backoff_cap_s: float = 60.0
    request_timeout_s: float = 300.0
    idle_poll_s: float = 30.0
    stall_recheck_s: float = 30.0
    history_messages: int = 10

    def __post_init__(self) -> None:
        if self.mode not in ("service", "off"):
            raise ValueError(
                f"MIST_EXTRACTION_INFERENCE must be 'service' or 'off', got {self.mode!r}"
            )
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {self.max_attempts}")
        if self.backoff_base_s < 0 or self.backoff_cap_s < 0:
            raise ValueError("backoff delays must be non-negative")

    @classmethod
    def from_env(cls) -> DispatcherSettings:
        """Read settings from the environment; unset variables keep their defaults.

        Raises:
            ValueError: A variable is set to an unparsable or out-of-range value.
                Refused rather than defaulted: a mistyped mode that silently
                became `service` or `off` would either dispatch when told not to
                or stop extracting with no signal.
        """
        mode = (os.getenv("MIST_EXTRACTION_INFERENCE") or "service").strip().lower()
        return cls(
            mode=mode,  # type: ignore[arg-type]  -- validated in __post_init__
            service_url=(os.getenv("MIST_EXTRACTION_SERVICE_URL") or DEFAULT_SERVICE_URL).strip(),
            max_attempts=_int_env("MIST_EXTRACTION_MAX_ATTEMPTS", 5),
            backoff_base_s=_float_env("MIST_EXTRACTION_BACKOFF_BASE_S", 2.0),
            backoff_cap_s=_float_env("MIST_EXTRACTION_BACKOFF_CAP_S", 60.0),
            request_timeout_s=_float_env("MIST_EXTRACTION_REQUEST_TIMEOUT_S", 300.0),
            idle_poll_s=_float_env("MIST_EXTRACTION_IDLE_POLL_S", 30.0),
            stall_recheck_s=_float_env("MIST_EXTRACTION_STALL_RECHECK_S", 30.0),
            history_messages=_int_env("MIST_EXTRACTION_HISTORY_MESSAGES", 10),
        )

    def backoff_s(self, failures: int) -> float:
        """Delay before the next try after `failures` consecutive failures (>= 1)."""
        if failures < 1:
            return 0.0
        return min(self.backoff_cap_s, self.backoff_base_s * (2 ** (failures - 1)))
