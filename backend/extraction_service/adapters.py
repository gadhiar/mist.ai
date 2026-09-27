"""Model-family adapters for the extraction service.

A `ModelFamilyAdapter` translates the engine's model-agnostic `LLMRequest`
into whatever a specific model family needs (reasoning/thinking controls)
and extracts the model's final answer text back out of an `LLMResponse`,
stripping any family-specific markup (Harmony channel tags, a leading
`<think>` block). Swapping models is a configuration change: pick a
different `adapter_name` via `get_adapter()`, plus a different model file
and `model_hash` -- no engine code changes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from backend.extraction_service.settings import ConstrainedMode, ReasoningEffort
from backend.llm.models import LLMRequest, LLMResponse, ThinkingConfig


@runtime_checkable
class ModelFamilyAdapter(Protocol):
    """Contract every model-family adapter satisfies.

    Implemented by `GptOssAdapter`, `QwenAdapter`, `GemmaAdapter`.
    """

    name: str
    version: str
    default_constrained_mode: ConstrainedMode

    def prepare(self, request: LLMRequest, stage: str) -> LLMRequest:
        """Return a copy of `request` with model-family thinking controls set.

        Args:
            request: The model-agnostic request built by the engine.
            stage: Which pipeline stage this call is for -- one of
                "scope", "extract", "derive". Adapters that need
                stage-specific behavior (e.g. a different reasoning
                effort per stage) read it; the current adapters do not.
        """
        ...

    def final_text(self, response: LLMResponse) -> str:
        """Return the model's final answer text, with family markup stripped."""
        ...


# Harmony's final-channel message, e.g.:
#   <|channel|>analysis<|message|>...<|end|><|start|>assistant<|channel|>final<|message|>{...}
# llama-server is expected to return only the final-channel content in
# `message.content` -- this is a defensive fallback for when it instead
# returns the raw Harmony transcript (unverified on b11151 per the T0 brief).
_HARMONY_FINAL_RE = re.compile(
    r"<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|$)", re.DOTALL
)

# A leading <think>...</think> block some Qwen-family chat templates emit
# even when the request's thinking controls attempt to disable it.
_LEADING_THINK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


def _extract_harmony_final(content: str) -> str:
    """Pull the final-channel message out of raw Harmony markup.

    Returns `content` unchanged if no final-channel marker is present.
    """
    match = _HARMONY_FINAL_RE.search(content)
    if match:
        return match.group(1).strip()
    return content


@dataclass(frozen=True, slots=True)
class GptOssAdapter:
    """Adapter for gpt-oss models (OpenAI Harmony chat format).

    Reasoning is always on for gpt-oss (the T0 brief's extraction model
    choice); `reasoning_effort` controls how much. Constrained decoding
    defaults to "none": grammar-constrained decoding alongside Harmony
    reasoning is unverified on llama.cpp b11151, so the engine relies on
    parse-and-repair instead.
    """

    reasoning_effort: ReasoningEffort = "low"
    reasoning_budget_tokens: int | None = None
    name: str = "gptoss"
    version: str = "1"
    default_constrained_mode: ConstrainedMode = "none"

    def prepare(self, request: LLMRequest, stage: str) -> LLMRequest:
        """Set gpt-oss reasoning-effort/budget thinking controls on `request`."""
        thinking = ThinkingConfig(
            effort=self.reasoning_effort,
            budget_tokens=self.reasoning_budget_tokens,
        )
        return request.model_copy(update={"thinking": thinking})

    def final_text(self, response: LLMResponse) -> str:
        """Return `response.content`, recovering the final-channel text from raw Harmony markup."""
        content = response.content or ""
        if "<|channel|>" in content:
            return _extract_harmony_final(content)
        return content


@dataclass(frozen=True, slots=True)
class QwenAdapter:
    """Adapter for Qwen models. Thinking disabled; strips a leading <think> block."""

    name: str = "qwen"
    version: str = "1"
    default_constrained_mode: ConstrainedMode = "schema"

    def prepare(self, request: LLMRequest, stage: str) -> LLMRequest:
        """Disable Qwen chat-template thinking on `request`."""
        return request.model_copy(update={"thinking": ThinkingConfig(enabled=False)})

    def final_text(self, response: LLMResponse) -> str:
        """Return `response.content` with a leading <think>...</think> block stripped."""
        content = response.content or ""
        return _LEADING_THINK_RE.sub("", content)


@dataclass(frozen=True, slots=True)
class GemmaAdapter:
    """Adapter for Gemma models (the current in-process extraction model). No thinking."""

    name: str = "gemma"
    version: str = "1"
    default_constrained_mode: ConstrainedMode = "schema"

    def prepare(self, request: LLMRequest, stage: str) -> LLMRequest:
        """Disable thinking on `request` (Gemma has no reasoning-effort hint)."""
        return request.model_copy(update={"thinking": ThinkingConfig(enabled=False)})

    def final_text(self, response: LLMResponse) -> str:
        """Return `response.content` unchanged -- Gemma emits no family-specific markup."""
        return response.content or ""


_ADAPTERS: dict[str, type] = {
    "gptoss": GptOssAdapter,
    "qwen": QwenAdapter,
    "gemma": GemmaAdapter,
}


def get_adapter(
    name: str,
    *,
    reasoning_effort: ReasoningEffort = "low",
    reasoning_budget_tokens: int | None = None,
) -> ModelFamilyAdapter:
    """Look up a model-family adapter by name.

    Args:
        name: One of "gptoss", "qwen", "gemma".
        reasoning_effort: Forwarded to `GptOssAdapter`; ignored by the
            other adapters (Qwen/Gemma disable thinking outright).
        reasoning_budget_tokens: Forwarded to `GptOssAdapter`; ignored by
            the other adapters.

    Returns:
        A constructed adapter instance.

    Raises:
        ValueError: `name` is not a known adapter.
    """
    try:
        adapter_cls = _ADAPTERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown model-family adapter {name!r}; known adapters: {sorted(_ADAPTERS)}"
        ) from None

    if adapter_cls is GptOssAdapter:
        return GptOssAdapter(
            reasoning_effort=reasoning_effort,
            reasoning_budget_tokens=reasoning_budget_tokens,
        )
    return adapter_cls()
