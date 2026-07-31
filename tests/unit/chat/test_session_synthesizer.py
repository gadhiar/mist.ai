"""SessionSynthesizer: turns in, synthesis out. No vault, no event store."""

from __future__ import annotations

import pytest

from backend.chat.session_synthesizer import SessionSynthesis, SessionSynthesizer


class _FakeLLM:
    """Records the prompt it was given and returns a canned completion."""

    def __init__(self, completion: str) -> None:
        self.completion = completion
        self.prompts: list[str] = []

    async def complete(self, request):  # matches the provider surface used below
        self.prompts.append(request.messages[0]["content"])

        class _Resp:
            content = self.completion

        return _Resp()


def _turns(n: int) -> list[dict]:
    return [
        {"user_utterance": f"user says {i}", "system_response": f"mist says {i}"} for i in range(n)
    ]


@pytest.mark.asyncio
async def test_returns_none_for_a_single_turn_session():
    """One turn is not a session worth remembering -- and synthesizing it
    would spend an LLM call to produce a note nobody wants.
    """
    llm = _FakeLLM("irrelevant")
    synth = SessionSynthesizer(llm_provider=llm, temperature=0.3, max_tokens=512)

    assert await synth.synthesize(_turns(1)) is None
    assert llm.prompts == [], "no LLM call may fire for a sub-threshold session"


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
    assert result.title, "a non-empty fallback title is required"
    assert result.body.startswith("### What Was Accomplished")


@pytest.mark.asyncio
async def test_llm_failure_returns_none_rather_than_raising():
    class _Boom:
        async def complete(self, request):
            raise RuntimeError("model down")

    synth = SessionSynthesizer(llm_provider=_Boom(), temperature=0.3, max_tokens=512)

    assert await synth.synthesize(_turns(2)) is None
