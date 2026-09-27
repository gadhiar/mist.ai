"""Environment-driven configuration for the extraction service.

`ServiceSettings` is a frozen dataclass; `from_env()` is the ONLY function
in this module that reads `os.environ`. No other module in
`backend.extraction_service` performs I/O at import time -- every
collaborator (LLM provider, adapter, cache size, etc.) is threaded through
constructor parameters, per the project's no-hidden-construction rule.

Environment variable names are all prefixed `EXTRACTION_`, except
`LLAMA_CPP_BUILD` which is shared with the rest of the deployment (T1b's
image build records the llama.cpp build it ships).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

ConstrainedMode = Literal["schema", "json_object", "none"]
ReasoningEffort = Literal["low", "medium", "high"]

_TRUE_STRINGS = frozenset({"1", "true", "yes", "on"})


@dataclass(frozen=True, slots=True)
class ServiceSettings:
    """Extraction service configuration, loaded once at process start.

    Attributes:
        llm_base_url: Base URL of the service's own llama-server instance
            (never the backend's -- the service has no backend address).
        model_hash: Bare LLM model identity this service is configured
            against. Compared to `ExtractRequest.expect.model_hash` for the
            epoch-mismatch check, and echoed back in `ResultStamps.model_hash`
            and `InfoResponse.model_hash`.
        model_file: The deployed model file name/path, reported in
            `InfoResponse.model_file` for operator visibility.
        adapter_name: Model-family adapter to use (`"gptoss"`, `"qwen"`, or
            `"gemma"`). Switching models is a config change: a different
            `adapter_name` + `model_file` + `model_hash`, nothing else.
        reasoning_effort: gpt-oss/Qwen chat-template reasoning-effort hint.
            Default "low" per the T0 brief.
        reasoning_budget_tokens: Reasoning token budget. `None` means
            unbudgeted -- the field is omitted from the outgoing LLM
            request entirely (see `ThinkingConfig.budget_tokens`).
        constrained_mode: Overrides the adapter's `default_constrained_mode`
            when set. `None` means "use the adapter's default".
        location_label: Human-readable label for where this service is
            deployed (e.g. "gtx1070-host", "local"), reported in
            `InfoResponse.location_label`.
        llm_timeout_seconds: Per-LLM-call timeout. A call exceeding this
            maps to the service's 504 `timeout` error code.
        max_attempts: Maximum extraction attempts per job -- the first call
            plus repair retries on unparsable output. Default 2 (one
            repair retry).
        idempotency_cache_size: Max entries in the job_id -> ExtractResponse
            LRU cache.
        scope_enabled: Master switch for Stage 1.5. When False, scope is
            always reported as "unknown" with confidence 0.0 and no LLM
            call is made for it.
        temperature: Sampling temperature for all three LLM-calling stages.
        llama_cpp_build: The llama.cpp build this service's llama-server
            runs, reported in `ResultStamps.llama_cpp_build` and
            `InfoResponse.llama_cpp_build`.
        port: TCP port `python -m backend.extraction_service` binds to.
    """

    llm_base_url: str
    model_hash: str
    model_file: str
    adapter_name: str = "gptoss"
    reasoning_effort: ReasoningEffort = "low"
    reasoning_budget_tokens: int | None = None
    constrained_mode: ConstrainedMode | None = None
    location_label: str = "local"
    llm_timeout_seconds: float = 30.0
    max_attempts: int = 2
    idempotency_cache_size: int = 256
    scope_enabled: bool = True
    temperature: float = 0.0
    llama_cpp_build: str = "b11151"
    port: int = 8090

    @classmethod
    def from_env(cls) -> ServiceSettings:
        """Build settings from `EXTRACTION_*` (and `LLAMA_CPP_BUILD`) env vars.

        Every field has a default, so this never raises for a missing
        variable -- it raises only if a present variable fails to parse
        (e.g. `EXTRACTION_MAX_ATTEMPTS=notanumber`).
        """
        reasoning_effort_raw = os.environ.get("EXTRACTION_REASONING_EFFORT", "low")
        constrained_mode_raw = os.environ.get("EXTRACTION_CONSTRAINED_MODE")
        return cls(
            llm_base_url=os.environ.get("EXTRACTION_LLM_BASE_URL", "http://localhost:8080"),
            model_hash=os.environ.get("EXTRACTION_MODEL_HASH", "gpt-oss-20b"),
            model_file=os.environ.get("EXTRACTION_MODEL_FILE", "gpt-oss-20b.gguf"),
            adapter_name=os.environ.get("EXTRACTION_ADAPTER", "gptoss"),
            reasoning_effort=reasoning_effort_raw,  # type: ignore[arg-type]
            reasoning_budget_tokens=_int_or_none(
                os.environ.get("EXTRACTION_REASONING_BUDGET_TOKENS")
            ),
            constrained_mode=constrained_mode_raw,  # type: ignore[arg-type]
            location_label=os.environ.get("EXTRACTION_LOCATION_LABEL", "local"),
            llm_timeout_seconds=float(os.environ.get("EXTRACTION_LLM_TIMEOUT_SECONDS", "30.0")),
            max_attempts=int(os.environ.get("EXTRACTION_MAX_ATTEMPTS", "2")),
            idempotency_cache_size=int(os.environ.get("EXTRACTION_IDEMPOTENCY_CACHE_SIZE", "256")),
            scope_enabled=_as_bool(os.environ.get("EXTRACTION_SCOPE_ENABLED", "true")),
            temperature=float(os.environ.get("EXTRACTION_TEMPERATURE", "0.0")),
            llama_cpp_build=os.environ.get("LLAMA_CPP_BUILD", "b11151"),
            port=int(os.environ.get("EXTRACTION_PORT", "8090")),
        )


def _int_or_none(value: str | None) -> int | None:
    """Parse an optional integer env var. Empty/unset means None."""
    if value is None or value.strip() == "":
        return None
    return int(value)


def _as_bool(value: str) -> bool:
    """Parse a boolean env var. Case-insensitive; "1"/"true"/"yes"/"on" are True."""
    return value.strip().lower() in _TRUE_STRINGS
