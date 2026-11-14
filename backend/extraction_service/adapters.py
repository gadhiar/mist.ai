"""Model-family adapters for the extraction service.

A `ModelFamilyAdapter` translates the engine's model-agnostic `LLMRequest`
into whatever a specific model family needs (reasoning/thinking controls)
and extracts the model's final answer text back out of an `LLMResponse`,
stripping any family-specific markup (Harmony channel tags, a leading
`<think>` block). Swapping models is a configuration change: pick a
different `adapter_name` via `get_adapter()`, plus a different model file
and `model_hash` -- no engine code changes.

`CHAT_TEMPLATE_SHA256` pins the chat template an adapter's llama-server must
serve, and `chat_template_problem` says when `/props` serves another. The
gpt-oss template the model file embeds writes the server's clock date into
every prompt, so llama-server loads the repository's copy
(`docker/extraction/chat_templates/gpt-oss.jinja`), which writes the request's
`current_date` instead; the service gives no serving identity while another
template is served. The pin is a module-level mapping, not an adapter
attribute, because the adapters are slots dataclasses.
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
                "scope", "extract", "derive". `GptOssAdapter` reads it to
                send each stage at its `stage_effort`.
        """
        ...

    def stage_effort(self, stage: str) -> ReasoningEffort | None:
        """The reasoning effort `prepare` sends for `stage`, or None when it sends none.

        The engine sizes each stage's output token cap by this effort
        (`backend.extraction_service.engine.effective_cap`), so an adapter
        that sends no effort keeps low's caps whatever the configured effort.
        """
        ...

    def final_text(self, response: LLMResponse) -> str:
        """Return the model's final answer text, with family markup stripped."""
        ...


# Harmony's final-channel message, in either header form:
#   <|channel|>analysis<|message|>...<|end|><|start|>assistant<|channel|>final<|message|>{...}
#   <|channel|>final <|constrain|>JSON<|message|>{...}
# The second form carries a content-type constraint between the channel and
# the message. The uppercase `JSON` token is what gpt-oss-20b emitted on the
# extraction host (llama.cpp b11151, constrained mode "none"; see
# `GptOssAdapter`). OpenAI's Harmony format guide shows the same header with
# a lowercase `json` on tool calls (recalled, not re-checked against the guide
# when this was written), so the token is matched as any run of non-`<`
# characters rather than one spelling.
# llama-server is expected to return only the final-channel content in
# `message.content` -- this is a defensive fallback for when it instead
# returns the raw Harmony transcript (unverified on b11151 per the T0 brief).
_HARMONY_FINAL_RE = re.compile(
    r"<\|channel\|>final(?:\s*<\|constrain\|>[^<]*)?\s*<\|message\|>"
    r"(.*?)(?:<\|end\|>|<\|return\|>|$)",
    re.DOTALL,
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
    choice); `reasoning_effort` controls how much on the extract stage, and
    scope and derive are sent at low (`stage_effort`). Constrained decoding
    defaults to "schema". An A/B run on the extraction host (llama.cpp
    b11151, gpt-oss-20b) found that "none" breaks the scope call: the model
    wraps its answer as `<|channel|>final <|constrain|>JSON<|message|>{...}`,
    llama-server's peg parser rejects that wrapper, and the call returns
    HTTP 500 on every retry, so scope degrades to "unknown" with
    `scope_classification_failed`. "schema" and "json_object" both return
    HTTP 200 on the same call, and the extraction payload's entities and
    relationships were identical under all three modes. "schema" is the
    stricter of the two working modes and matches `QwenAdapter` and
    `GemmaAdapter`. Should raw Harmony markup reach `final_text` anyway,
    `_HARMONY_FINAL_RE` recovers the message from that
    `<|constrain|>`-wrapped header as well as from the plain one.
    """

    reasoning_effort: ReasoningEffort = "low"
    reasoning_budget_tokens: int | None = None
    name: str = "gptoss"
    version: str = "1"
    default_constrained_mode: ConstrainedMode = "schema"

    def prepare(self, request: LLMRequest, stage: str) -> LLMRequest:
        """Set the stage's reasoning effort on `request`, and the budget on extract.

        The budget is the extract stage's alone: it is there to bound the
        extract call's reasoning. Scope and derive are sent none; each runs
        at low effort under low's output cap (`stage_effort`). The turn plan
        call is sent as stage extract, so it carries the budget too.
        """
        thinking = ThinkingConfig(
            effort=self.stage_effort(stage),
            budget_tokens=self.reasoning_budget_tokens if stage == "extract" else None,
        )
        return request.model_copy(update={"thinking": thinking})

    def stage_effort(self, stage: str) -> ReasoningEffort:
        """`reasoning_effort` for extract; low for scope and derive.

        Scope is a classification and is sent at low under every setting. The
        engine sizes a stage's cap by this effort, so scope also keeps low's
        cap whatever the setting. Derive stays at low effort under every setting
        because no effort run calls derive (the probe client sends no
        derivation), so derive above low would ship unmeasured.
        """
        if stage == "extract":
            return self.reasoning_effort
        return "low"

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

    def stage_effort(self, stage: str) -> None:
        """None: Qwen thinking is disabled, so no stage is sent an effort."""
        return None

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

    def stage_effort(self, stage: str) -> None:
        """None: Gemma has no reasoning-effort hint, so no stage is sent an effort."""
        return None

    def final_text(self, response: LLMResponse) -> str:
        """Return `response.content` unchanged -- Gemma emits no family-specific markup."""
        return response.content or ""


_ADAPTERS: dict[str, type] = {
    "gptoss": GptOssAdapter,
    "qwen": QwenAdapter,
    "gemma": GemmaAdapter,
}

# The sha256 of the chat template each adapter's llama-server must serve on
# `/props`, as the UTF-8 bytes of its `chat_template` string. gptoss's is
# `docker/extraction/chat_templates/gpt-oss.jinja`; re-pin it with that file's
# provenance test (tests/unit/extraction_service/test_chat_template.py) in the
# commit that edits the file. Qwen and gemma pin none.
CHAT_TEMPLATE_SHA256: dict[str, str] = {
    "gptoss": "a7dd4be8cc1fcefeab5348d597d1064731af053a9a8ba3ab61fdc8bc8cff62f6",
}


def chat_template_problem(adapter_name: str, served_sha256: str | None) -> str | None:
    """Why the chat template llama-server serves is wrong for `adapter_name`, or None.

    An adapter `CHAT_TEMPLATE_SHA256` pins must be served exactly that
    template; any other, or none, is a problem. A registered adapter it does
    not pin must not be served another adapter's pinned template, so a switch
    to qwen that keeps gpt-oss's template is refused. A name `_ADAPTERS` does
    not register gets no check.

    Args:
        adapter_name: The engine's adapter name.
        served_sha256: The sha256 of the template `/props` serves, or None
            when it serves none.
    """
    if adapter_name not in _ADAPTERS:
        return None
    expected = CHAT_TEMPLATE_SHA256.get(adapter_name)
    if expected is not None:
        if served_sha256 == expected:
            return None
        observed = "none" if served_sha256 is None else served_sha256
        return f"served chat template sha256 {observed} != the {adapter_name} adapter's {expected}"
    for owner, pinned in CHAT_TEMPLATE_SHA256.items():
        if served_sha256 == pinned:
            return (
                f"served chat template sha256 {served_sha256} is the {owner} adapter's, "
                f"not the {adapter_name} adapter's"
            )
    return None


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
