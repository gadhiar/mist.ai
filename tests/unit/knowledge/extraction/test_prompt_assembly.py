"""The system prompt is the engine core with a pack's fragments placed in its slots.

The core holds today's whole prompt, so with no fragment the assembled text is today's
byte for byte; `TestExtractionVersionDriftGuard` in `test_prompts.py` pins those bytes.
These tests fix where a fragment lands, that its text reaches the model as written, and
what the loader measures the assembled prompt by.
"""

from __future__ import annotations

import dataclasses
import math
from datetime import UTC, datetime

import pytest

from backend.errors import MistError
from backend.knowledge.extraction import ontology_extractor, prompts
from backend.knowledge.extraction.preprocessor import PreProcessor
from backend.knowledge.extraction.prompts import (
    CONSTRAINTS_SLOT,
    ENGINE_CORE,
    RULES_SLOT,
    SYSTEM_PROMPT_MAX_CHARACTERS,
    PromptAssemblyError,
    Slot,
    assemble_system_prompt,
    estimate_prompt_tokens,
)
from backend.knowledge.ontologies import registry
from backend.knowledge.ontologies.pack import PromptFragment, load_pack
from scripts.eval_harness import run as harness_run

BASE = load_pack("tests.unit.knowledge.ontologies.fixtures.base_pack")
PET = load_pack("tests.unit.knowledge.ontologies.fixtures.pet_pack")

_DATE = "2026-11-11"
# The prompt every reader sends: the active pack's, assembled by the loader.
_PROMPT = registry.active_pack().system_prompt
_UTTERANCE = "The harbour office opens at nine."
_RULE = PromptFragment(text="F1. A deckhand in a crew serves under its skipper.\n", position=1)
_BLOCK = PromptFragment(text="### Crew ranks\n- skipper, mate, hand\n\n", position=0)


def _fragments(pack) -> tuple[PromptFragment, ...]:
    return tuple(rule.params for rule in pack.rules_of("prompt_fragment"))


# --- the core ---------------------------------------------------------------------------


def test_the_core_with_no_fragment_is_the_prompt_as_it_stood():
    core_text = "".join(part for part in ENGINE_CORE if isinstance(part, str))

    assert assemble_system_prompt() == _PROMPT == core_text


def test_the_core_has_a_constraints_slot_a_rules_slot_and_an_examples_slot_in_that_order():
    slots = [part for part in ENGINE_CORE if isinstance(part, Slot)]

    assert [slot.position for slot in slots] == [CONSTRAINTS_SLOT, RULES_SLOT, None]
    assert (CONSTRAINTS_SLOT, RULES_SLOT) == (0, 1)


# --- where a fragment lands -------------------------------------------------------------


def test_a_rules_fragment_follows_the_last_rule_and_precedes_the_reference_date():
    assembled = assemble_system_prompt((_RULE,))

    last_rule_end = "a frame whose year is unclear gets no date.\n"
    assert f"{last_rule_end}{_RULE.text}\n## REFERENCE DATE\n" in assembled
    assert assembled.replace(_RULE.text, "") == _PROMPT


def test_a_constraints_fragment_closes_the_constraints_before_the_schema_note():
    assembled = assemble_system_prompt((_BLOCK,))

    assert f'(id="mist-identity").\n\n{_BLOCK.text}## OUTPUT SCHEMA\n' in assembled
    assert assembled.replace(_BLOCK.text, "") == _PROMPT


def test_examples_close_the_prompt_in_the_order_of_their_fragments():
    first = PromptFragment(text="", position=1, examples=("\n### Example A\n", "\n### Example B\n"))
    second = PromptFragment(text="", position=0, examples=("\n### Example C\n",))

    assembled = assemble_system_prompt((first, second))

    assert assembled == (_PROMPT + "\n### Example A\n\n### Example B\n\n### Example C\n")


def test_fragments_of_one_slot_keep_the_order_the_pack_gives_them():
    second = PromptFragment(text="F2. A mooring holds one boat.\n", position=1)

    assert f"{_RULE.text}{second.text}" in assemble_system_prompt((_RULE, second))
    assert f"{second.text}{_RULE.text}" in assemble_system_prompt((second, _RULE))


def test_the_assembler_adds_nothing_between_a_fragment_and_the_core():
    inline = PromptFragment(text="F1. No newline of its own.", position=1)

    assert "gets no date.\nF1. No newline of its own.\n## REFERENCE DATE" in (
        assemble_system_prompt((inline,))
    )


# --- a fragment reaches the model as written --------------------------------------------


def test_a_fragments_braces_reach_the_model_as_written():
    example = '\n### Example F1\nOutput:\n{"entities": [], "relationships": []}\n'
    fragment = PromptFragment(
        text='F1. Return {"entities": []} for a bare greeting.\n', position=1, examples=(example,)
    )

    rendered = assemble_system_prompt((fragment,)).format(reference_date=_DATE)

    assert fragment.text in rendered
    assert rendered.endswith(example)


def test_a_fragment_cannot_take_the_reference_date_field():
    fragment = PromptFragment(text="F1. The day is {reference_date}.\n", position=1)

    rendered = assemble_system_prompt((fragment,)).format(reference_date=_DATE)

    assert "F1. The day is {reference_date}.\n" in rendered
    assert f"Today's date: {_DATE}\n" in rendered


# --- refusals ---------------------------------------------------------------------------


def test_a_fragment_naming_a_position_the_core_has_no_slot_for_is_refused():
    with pytest.raises(PromptAssemblyError, match="position 7"):
        assemble_system_prompt((PromptFragment(text="F1.\n", position=7),))


def test_the_assembly_error_is_a_mist_error():
    assert issubclass(PromptAssemblyError, MistError)


# --- what the loader measures -----------------------------------------------------------


def test_the_character_guard_is_the_one_the_prompt_tests_hold_the_prompt_to():
    assert SYSTEM_PROMPT_MAX_CHARACTERS == 52000
    assert len(assemble_system_prompt()) <= SYSTEM_PROMPT_MAX_CHARACTERS


def test_tokens_are_estimated_from_the_prompt_as_the_model_reads_it():
    rendered = _PROMPT.format(reference_date=_DATE)

    assert estimate_prompt_tokens(_PROMPT) == math.ceil(
        len(rendered) / prompts.PROMPT_CHARACTERS_PER_TOKEN
    )
    assert prompts.PROMPT_CHARACTERS_PER_TOKEN == 3.6


def test_the_estimate_of_todays_prompt_is_near_the_count_the_runs_logged():
    # The paired runs logged about 13,300 prompt tokens for this prompt; the estimate is
    # a proxy for that count and must stay within a tenth of it.
    assert 12000 <= estimate_prompt_tokens(_PROMPT) <= 14600


# --- the fixture packs ------------------------------------------------------------------


def test_the_base_fixture_packs_fragment_is_placed_as_a_rule():
    (fragment,) = _fragments(BASE)

    assembled = assemble_system_prompt(_fragments(BASE))

    assert fragment.text.endswith("\n")
    assert f"gets no date.\n{fragment.text}\n## REFERENCE DATE\n" in assembled
    assert BASE.system_prompt == assembled


def test_the_worked_example_assembles_with_its_rule_text_and_its_two_examples():
    roster = next(rule.params for rule in PET.rules if rule.id == "pet-roster-text")

    assembled = assemble_system_prompt(_fragments(PET))

    assert len(roster.examples) == 2
    assert roster.text in assembled
    # The examples hold JSON, so they stand as written once the template is rendered.
    assert assembled.format(reference_date=_DATE).endswith("".join(roster.examples))
    assert assembled.index(_fragments(BASE)[0].text) < assembled.index(roster.text)
    assert PET.system_prompt == assembled
    assert "HAS_PET" in PET.system_prompt.format(reference_date=_DATE)


# --- who sends the prompt ---------------------------------------------------------------


def _other_pack():
    """The active pack with a prompt of its own, so the source a reader takes shows."""
    return dataclasses.replace(
        registry.active_pack(), system_prompt="A harbour log, read on {reference_date}."
    )


def _in_process_system_message() -> dict:
    pre_processed = PreProcessor().pre_process(
        utterance=_UTTERANCE,
        conversation_history=[],
        reference_date=datetime(2026, 11, 11, tzinfo=UTC),
    )
    return ontology_extractor.render_extraction_messages(pre_processed)[0]


def _harness_system_message() -> dict:
    return harness_run._render_production_extraction(_UTTERANCE, _DATE)[0]


@pytest.mark.parametrize(
    "system_message",
    [_in_process_system_message, _harness_system_message],
    ids=["in_process_extractor", "eval_harness"],
)
def test_a_reader_sends_the_active_packs_prompt_for_the_reference_date(monkeypatch, system_message):
    assert system_message() == {
        "role": "system",
        "content": _PROMPT.format(reference_date=_DATE),
    }

    other = _other_pack()
    monkeypatch.setattr(registry, "active_pack", lambda: other)

    assert system_message() == {
        "role": "system",
        "content": "A harbour log, read on 2026-11-11.",
    }
