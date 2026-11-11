"""What a pack feeds the model: its assembled prompt, its schema, and the hash over them.

The loader assembles the system prompt and generates the output schema, refuses a pack
it can do neither for, and keeps the two hashes apart: a change the model can see moves
`pack_prompt_hash` alone, and a change to what runs after the model moves
`pack_rails_hash` alone. Every case runs on the fixture packs.
"""

from __future__ import annotations

import dataclasses

import pytest

from backend.extraction_service import schemas
from backend.knowledge.extraction import prompts
from backend.knowledge.ontologies.pack import (
    PRE_LLM,
    ConventionRule,
    EdgeSuppress,
    ItemFilter,
    Pack,
    PackLoadError,
    PlanVocabulary,
    PromptFragment,
    ReservedIdentity,
    build_pack,
)
from tests.unit.knowledge.ontologies.fixtures.base_pack import conventions, manifest, plugins, types


def _build(**overrides) -> Pack:
    parts = {
        "manifest": manifest.MANIFEST,
        "node_types": types.NODE_TYPES,
        "edge_types": types.EDGE_TYPES,
        "extractable_node_types": types.EXTRACTABLE_NODE_TYPES,
        "extractable_edge_types": types.EXTRACTABLE_EDGE_TYPES,
        "rules": conventions.RULES,
        "plugins": plugins.PLUGINS,
        "labelling": "rows",
    }
    parts.update(overrides)
    return build_pack(**parts)


def _first(kind: str) -> ConventionRule:
    return next(rule for rule in conventions.RULES if rule.kind == kind)


def _with(rule: ConventionRule, **param_changes) -> tuple[ConventionRule, ...]:
    """The base rules, with `rule`'s parameters changed."""
    changed = dataclasses.replace(rule, params=dataclasses.replace(rule.params, **param_changes))
    return tuple(changed if each.id == rule.id else each for each in conventions.RULES)


def _moves_only_the_prompt_hash(changed: Pack) -> bool:
    base = _build()
    return (
        changed.pack_prompt_hash != base.pack_prompt_hash
        and changed.pack_rails_hash == base.pack_rails_hash
    )


def _moves_only_the_rails_hash(changed: Pack) -> bool:
    base = _build()
    return (
        changed.pack_rails_hash != base.pack_rails_hash
        and changed.pack_prompt_hash == base.pack_prompt_hash
    )


# --- what the loader builds -------------------------------------------------------------


def test_a_loaded_pack_holds_its_assembled_prompt_and_its_schema():
    pack = _build()
    fragments = tuple(rule.params for rule in pack.rules_of("prompt_fragment"))

    assert pack.system_prompt == prompts.assemble_system_prompt(fragments)
    assert pack.extraction_schema == schemas.generate_extraction_schema(
        entity_types=types.EXTRACTABLE_NODE_TYPES,
        relationship_types=types.EXTRACTABLE_EDGE_TYPES,
    )


def test_a_pack_with_no_fragment_holds_the_engine_core():
    without = tuple(rule for rule in conventions.RULES if rule.kind != "prompt_fragment")

    assert _build(rules=without).system_prompt == prompts.EXTRACTION_SYSTEM_PROMPT


# --- a prompt-visible change moves only the prompt hash ---------------------------------


@pytest.mark.parametrize(
    "changes",
    [
        pytest.param({"text": "F1. Reworded.\n"}, id="text"),
        pytest.param({"position": 0}, id="position"),
        pytest.param({"examples": ("\n### Example F1\n",)}, id="examples"),
    ],
)
def test_a_fragment_change_moves_only_the_prompt_hash(changes):
    assert _moves_only_the_prompt_hash(_build(rules=_with(_first("prompt_fragment"), **changes)))


def test_an_item_filter_change_moves_only_the_prompt_hash():
    rule = _first("item_filter")
    assert isinstance(rule.params, ItemFilter)

    changed = _build(rules=_with(rule, keep_context_words=("errand", "chore")))

    assert _moves_only_the_prompt_hash(changed)


def test_a_plan_vocabulary_change_moves_only_the_prompt_hash():
    rule = _first("plan_vocabulary")
    assert isinstance(rule.params, PlanVocabulary)

    changed = _build(rules=_with(rule, fan_out_predicates=("SERVES_UNDER", "CREWS_WITH")))

    assert _moves_only_the_prompt_hash(changed)


def test_the_generated_schema_moves_only_the_prompt_hash(monkeypatch):
    base = _build()
    tighter = {**base.extraction_schema, "additionalProperties": False}
    monkeypatch.setattr(schemas, "generate_extraction_schema", lambda **_names: tighter)

    changed = _build()

    assert changed.extraction_schema == tighter
    assert changed.pack_prompt_hash != base.pack_prompt_hash
    assert changed.pack_rails_hash == base.pack_rails_hash


def test_the_engine_core_is_not_the_packs_and_moves_neither_hash(monkeypatch):
    base = _build()
    monkeypatch.setattr(prompts, "ENGINE_CORE", ("A shorter core.\n", *prompts.ENGINE_CORE[1:]))

    changed = _build()

    assert changed.system_prompt != base.system_prompt
    assert (changed.pack_prompt_hash, changed.pack_rails_hash) == (
        base.pack_prompt_hash,
        base.pack_rails_hash,
    )


# --- a post-LLM rule change moves only the rails hash -----------------------------------


def test_a_payload_rule_change_moves_only_the_rails_hash():
    rule = _first("reserved_identity")
    assert isinstance(rule.params, ReservedIdentity)

    changed = _build(rules=_with(rule, names=(*rule.params.names, "the old man")))

    assert _moves_only_the_rails_hash(changed)


def test_a_relation_rule_change_moves_only_the_rails_hash():
    rule = _first("edge_suppress")
    assert isinstance(rule.params, EdgeSuppress)

    changed = _build(rules=_with(rule, source_is_target_of=("CREWS_WITH", "SERVES_UNDER")))

    assert _moves_only_the_rails_hash(changed)


def test_an_id_rule_change_moves_only_the_rails_hash():
    rule = _first("id_exempt")

    assert _moves_only_the_rails_hash(_build(rules=_with(rule, types=("Mooring", "Deckhand"))))


# --- what the loader refuses ------------------------------------------------------------


def test_a_schema_that_does_not_generate_is_refused():
    with pytest.raises(PackLoadError, match="schema does not generate: no extractable entity"):
        _build(extractable_node_types=())


def test_a_fragment_the_core_has_no_slot_for_is_refused():
    with pytest.raises(PackLoadError, match="prompt does not assemble: .*position 9"):
        _build(rules=_with(_first("prompt_fragment"), position=9))


def test_a_prompt_over_the_character_guard_is_refused():
    room = prompts.SYSTEM_PROMPT_MAX_CHARACTERS - len(_build().system_prompt)
    filler = ConventionRule(
        id="filler-text",
        kind="prompt_fragment",
        phase=PRE_LLM,
        params=PromptFragment(text="x" * (room + 1), position=1),
    )
    unbounded = dataclasses.replace(manifest.MANIFEST, prompt_token_budget=None)

    with pytest.raises(PackLoadError, match="52001 characters, over the 52000-character guard"):
        _build(manifest=unbounded, rules=(*conventions.RULES, filler))


def test_a_prompt_that_just_fits_the_character_guard_loads():
    room = prompts.SYSTEM_PROMPT_MAX_CHARACTERS - len(_build().system_prompt)
    filler = ConventionRule(
        id="filler-text",
        kind="prompt_fragment",
        phase=PRE_LLM,
        params=PromptFragment(text="x" * room, position=1),
    )
    unbounded = dataclasses.replace(manifest.MANIFEST, prompt_token_budget=None)

    pack = _build(manifest=unbounded, rules=(*conventions.RULES, filler))

    assert len(pack.system_prompt) == prompts.SYSTEM_PROMPT_MAX_CHARACTERS


def test_a_prompt_over_the_manifests_token_budget_is_refused():
    tokens = prompts.estimate_prompt_tokens(_build().system_prompt)
    tight = dataclasses.replace(manifest.MANIFEST, prompt_token_budget=tokens - 1)

    with pytest.raises(
        PackLoadError, match=f"about {tokens} tokens, over the manifest's budget of {tokens - 1}"
    ):
        _build(manifest=tight)


def test_a_prompt_at_the_manifests_token_budget_loads():
    tokens = prompts.estimate_prompt_tokens(_build().system_prompt)
    exact = dataclasses.replace(manifest.MANIFEST, prompt_token_budget=tokens)

    assert _build(manifest=exact).manifest.prompt_token_budget == tokens


def test_a_manifest_with_no_budget_is_held_to_the_character_guard_alone():
    unbounded = dataclasses.replace(manifest.MANIFEST, prompt_token_budget=None)

    assert _build(manifest=unbounded).system_prompt == _build().system_prompt
