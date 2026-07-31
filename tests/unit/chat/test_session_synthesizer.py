"""SessionSynthesizer: turns in, synthesis out. No vault, no event store."""

from __future__ import annotations

import pytest

from backend.chat.session_synthesizer import SessionSynthesis, SessionSynthesizer


class _FakeLLM:
    """Records the prompt it was given and returns a canned completion.

    `invoke` matches `StreamingLLMProvider.invoke` (backend/llm/provider.py)
    -- the real method every provider (llama-server, Ollama, instrumented
    wrapper) implements, and the one `conversation_handler.py` calls for this
    same synthesis use case. A fake built against any other method name would
    pass tests while hiding an `AttributeError` behind SessionSynthesizer's
    best-effort exception handling.
    """

    def __init__(self, completion: str) -> None:
        self.completion = completion
        self.prompts: list[str] = []

    async def invoke(self, request):
        self.prompts.append(request.messages[0]["content"])

        class _Resp:
            content = self.completion

        return _Resp()


def _turns(n: int) -> list[dict]:
    return [
        {"user_utterance": f"user says {i}", "system_response": f"mist says {i}"} for i in range(n)
    ]


@pytest.mark.asyncio
async def test_synthesizes_a_single_turn_session():
    """A single turn is still a real exchange worth remembering: the one
    session note MIST has ever produced in the vault came from a one-turn
    session (mist-memory/sessions/2026-06-09-know-give-concise-summary-37a8.md,
    turn_count: 1) and carries a genuinely useful summary. "Enough
    conversation to synthesize" and "was anything worth remembering" are
    separate gates -- the second lives in the catch-up graph-state filter,
    not here.
    """
    llm = _FakeLLM("TITLE: T\n\n### What Was Accomplished\n- x\n")
    synth = SessionSynthesizer(llm_provider=llm, temperature=0.3, max_tokens=512)

    result = await synth.synthesize(_turns(1))

    assert result is not None
    assert llm.prompts, "a single turn must still trigger the LLM call"


@pytest.mark.asyncio
async def test_returns_none_for_no_turns():
    llm = _FakeLLM("irrelevant")
    synth = SessionSynthesizer(llm_provider=llm, temperature=0.3, max_tokens=512)

    assert await synth.synthesize([]) is None
    assert llm.prompts == []


@pytest.mark.asyncio
async def test_transcript_includes_every_turn_both_sides():
    llm = _FakeLLM("TITLE: T\n\n### What Was Accomplished\n- x\n")
    synth = SessionSynthesizer(llm_provider=llm, temperature=0.3, max_tokens=512)

    await synth.synthesize(_turns(3))

    prompt = llm.prompts[0]
    for i in range(3):
        assert f"user says {i}" in prompt
        assert f"mist says {i}" in prompt


@pytest.mark.asyncio
async def test_parses_title_and_body_out_of_the_completion():
    llm = _FakeLLM(
        "TITLE: Vault write policy discussion\n\n"
        "### What Was Accomplished\n- Decided to drop per-turn appends\n"
    )
    synth = SessionSynthesizer(llm_provider=llm, temperature=0.3, max_tokens=512)

    result = await synth.synthesize(_turns(2))

    assert isinstance(result, SessionSynthesis)
    assert result.title == "Vault write policy discussion"
    assert result.body.startswith("### What Was Accomplished")
    assert "TITLE:" not in result.body, "the title line must not leak into the body"


@pytest.mark.asyncio
async def test_falls_back_to_a_derived_title_when_the_model_omits_one():
    """The model is small and will sometimes ignore the TITLE convention.
    A missing title must not lose the whole synthesis.
    """
    llm = _FakeLLM("### What Was Accomplished\n- Something happened\n")
    synth = SessionSynthesizer(llm_provider=llm, temperature=0.3, max_tokens=512)

    result = await synth.synthesize(_turns(2))

    assert result is not None
    assert result.title == "Conversation", "the documented fallback title, not just any string"
    assert result.body.startswith("### What Was Accomplished")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "completion, expected_title",
    [
        pytest.param(
            "TITLE : Near-miss with a space before the colon\n\n"
            "### What Was Accomplished\n- x\n",
            "Near-miss with a space before the colon",
            id="space-before-colon",
        ),
        pytest.param(
            "**TITLE:** Near-miss wrapped in markdown bold\n\n" "### What Was Accomplished\n- x\n",
            "Near-miss wrapped in markdown bold",
            id="markdown-bold-marker",
        ),
    ],
)
async def test_tolerates_near_miss_title_formats(completion, expected_title):
    """The model is small and drifts from the exact `TITLE:` convention --
    a near-miss must still parse into a title and must never leak the raw
    marker line into the body.
    """
    llm = _FakeLLM(completion)
    synth = SessionSynthesizer(llm_provider=llm, temperature=0.3, max_tokens=512)

    result = await synth.synthesize(_turns(2))

    assert result is not None
    assert result.title == expected_title
    assert result.body.startswith("### What Was Accomplished")
    assert "TITLE" not in result.body, "a recognized title attempt must not survive into the body"


@pytest.mark.asyncio
async def test_llm_failure_returns_none_rather_than_raising():
    class _Boom:
        async def invoke(self, request):
            raise RuntimeError("model down")

    synth = SessionSynthesizer(llm_provider=_Boom(), temperature=0.3, max_tokens=512)

    assert await synth.synthesize(_turns(2)) is None


class TestIsReady:
    """R1.3.1 fix round 1 (I4): is_ready must forward health_check faithfully.

    `SessionNoteCatchup`'s readiness gate depends on this: a cold LLM at
    boot must be indistinguishable from "not ready yet," never silently
    treated as "ready."
    """

    @pytest.mark.asyncio
    async def test_returns_true_when_the_provider_reports_healthy(self):
        class _Healthy:
            async def health_check(self) -> bool:
                return True

        synth = SessionSynthesizer(llm_provider=_Healthy(), temperature=0.3, max_tokens=512)

        assert await synth.is_ready() is True

    @pytest.mark.asyncio
    async def test_returns_false_when_the_provider_reports_unhealthy(self):
        class _Cold:
            async def health_check(self) -> bool:
                return False

        synth = SessionSynthesizer(llm_provider=_Cold(), temperature=0.3, max_tokens=512)

        assert await synth.is_ready() is False

    @pytest.mark.asyncio
    async def test_returns_false_rather_than_raising_when_the_check_itself_fails(self):
        class _Broken:
            async def health_check(self) -> bool:
                raise RuntimeError("connection refused")

        synth = SessionSynthesizer(llm_provider=_Broken(), temperature=0.3, max_tokens=512)

        assert await synth.is_ready() is False
