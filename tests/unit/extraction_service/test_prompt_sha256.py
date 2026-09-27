"""The `prompt_sha256` stamp covers every prompt text that shapes a stage's output.

`ExtractResponse.stamps.prompt_sha256` is the audit record of what the model was
asked. A prompt edit that leaves the stamp unchanged would let two different
extraction behaviours share one identity, so each prompt component must move
the hash on its own: the scope, extraction and derivation templates, the Stage 2
repair instruction sent on a retry, and the adapter identity.
"""

from __future__ import annotations

import pytest

from backend.extraction_service import engine
from backend.extraction_service.adapters import get_adapter
from backend.knowledge.extraction.preprocessor import PreProcessor
from backend.knowledge.extraction.scope_classifier import render_scope_messages

_COMPONENTS = (
    "SCOPE_CLASSIFIER_SYSTEM_PROMPT",
    "SCOPE_USER_TEMPLATE",
    "EXTRACTION_SYSTEM_PROMPT",
    "EXTRACTION_USER_TEMPLATE",
    "INTERNAL_DERIVATION_SYSTEM_PROMPT",
    "INTERNAL_DERIVATION_USER_TEMPLATE",
    "_REPAIR_INSTRUCTION",
)


def _hash() -> str:
    return engine._compute_prompt_sha256(get_adapter("gptoss"))


def test_hash_is_deterministic() -> None:
    assert _hash() == _hash()


@pytest.mark.parametrize("name", _COMPONENTS)
def test_each_prompt_component_moves_the_hash(monkeypatch, name) -> None:
    before = _hash()
    monkeypatch.setattr(engine, name, getattr(engine, name) + " (edited)")
    assert _hash() != before, f"editing {name} left prompt_sha256 unchanged"


def test_adapter_identity_moves_the_hash() -> None:
    assert _hash() != engine._compute_prompt_sha256(get_adapter("qwen"))


def test_text_moved_between_neighbouring_templates_changes_the_hash(monkeypatch) -> None:
    before = _hash()
    system = engine.EXTRACTION_SYSTEM_PROMPT
    user = engine.EXTRACTION_USER_TEMPLATE
    monkeypatch.setattr(engine, "EXTRACTION_SYSTEM_PROMPT", system + user[:1])
    monkeypatch.setattr(engine, "EXTRACTION_USER_TEMPLATE", user[1:])
    assert _hash() != before


def test_scope_user_template_renders_as_before() -> None:
    """Lifting the scope user message into a constant must not change the prompt."""
    from datetime import UTC, datetime

    pre = PreProcessor().pre_process(
        utterance='I moved to "Leeds" {last} year',
        conversation_history=[],
        reference_date=datetime(2026, 9, 26, tzinfo=UTC),
    )
    messages = render_scope_messages(pre)
    assert messages[1] == {
        "role": "user",
        "content": 'Utterance: "I moved to "Leeds" {last} year"\n\nOutput:',
    }
