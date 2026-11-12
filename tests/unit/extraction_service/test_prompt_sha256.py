"""The `prompt_sha256` stamp covers the prompt text and schemas sent to the model.

`ExtractResponse.stamps.prompt_sha256` is the per-result provenance record of
what the model was asked (not the epoch's identity, which is extraction_version
plus the composed model_hash). A prompt edit that left it unchanged would make
two different requests look identical in the audit trail, so each component
must move the hash on its own: the scope, extraction and derivation templates,
the Stage 2 empty-context placeholder and repair instruction, the three output
schemas, and the adapter identity.
"""

from __future__ import annotations

import dataclasses

import pytest

from backend.extraction_service import engine
from backend.extraction_service.adapters import get_adapter
from backend.knowledge.extraction.preprocessor import PreProcessor
from backend.knowledge.extraction.scope_classifier import render_scope_messages
from backend.knowledge.ontologies.registry import active_pack

_COMPONENTS = (
    "SCOPE_CLASSIFIER_SYSTEM_PROMPT",
    "SCOPE_USER_TEMPLATE",
    "EXTRACTION_USER_TEMPLATE",
    "INTERNAL_DERIVATION_SYSTEM_PROMPT",
    "INTERNAL_DERIVATION_USER_TEMPLATE",
    "_REPAIR_INSTRUCTION",
    "NO_PRIOR_CONTEXT",
)

# The Stage 2 system prompt and schema are the pack's, not a constant of the engine:
# `_serve` gives the engine an edited pack in their place.
_SCHEMAS = ("SCOPE_OUTPUT_SCHEMA", "DERIVATION_OUTPUT_SCHEMA")


def _hash() -> str:
    return engine._compute_prompt_sha256(get_adapter("gptoss"))


def _serve(monkeypatch, **changes) -> None:
    """Make the active pack, as the engine reads it, today's pack with `changes`."""
    edited = dataclasses.replace(active_pack(), **changes)
    monkeypatch.setattr(engine, "active_pack", lambda: edited)


def test_hash_is_deterministic() -> None:
    assert _hash() == _hash()


@pytest.mark.parametrize("name", _COMPONENTS)
def test_each_prompt_component_moves_the_hash(monkeypatch, name) -> None:
    before = _hash()
    monkeypatch.setattr(engine, name, getattr(engine, name) + " (edited)")
    assert _hash() != before, f"editing {name} left prompt_sha256 unchanged"


def test_the_packs_system_prompt_moves_the_hash(monkeypatch) -> None:
    before = _hash()
    _serve(monkeypatch, system_prompt=active_pack().system_prompt + " (edited)")
    assert _hash() != before, "editing the pack's prompt left prompt_sha256 unchanged"


def test_the_packs_extraction_schema_moves_the_hash(monkeypatch) -> None:
    before = _hash()
    edited = {**active_pack().extraction_schema, "description": "edited"}
    _serve(monkeypatch, extraction_schema=edited)
    assert _hash() != before, "editing the pack's schema left prompt_sha256 unchanged"


@pytest.mark.parametrize("name", _SCHEMAS)
def test_each_output_schema_moves_the_hash(monkeypatch, name) -> None:
    before = _hash()
    edited = {**getattr(engine, name), "description": "edited"}
    monkeypatch.setattr(engine, name, edited)
    assert _hash() != before, f"editing {name} left prompt_sha256 unchanged"


def test_adapter_identity_moves_the_hash() -> None:
    assert _hash() != engine._compute_prompt_sha256(get_adapter("qwen"))


def test_text_moved_between_neighbouring_templates_changes_the_hash(monkeypatch) -> None:
    before = _hash()
    system = active_pack().system_prompt
    user = engine.EXTRACTION_USER_TEMPLATE
    _serve(monkeypatch, system_prompt=system + user[:1])
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
