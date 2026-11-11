"""The pre-LLM interpreters: the item filter and the plan vocabulary, on the fixture packs."""

from __future__ import annotations

from backend.knowledge.extraction.pre_llm import PlanWords, filter_units, plan_vocabulary
from backend.knowledge.extraction.structure_splitter import InputUnit, split_structured
from backend.knowledge.ontologies.pack import build_pack, load_pack

BASE = load_pack("tests.unit.knowledge.ontologies.fixtures.base_pack")
PET = load_pack("tests.unit.knowledge.ontologies.fixtures.pet_pack")


def _item(text: str, *context: str) -> InputUnit:
    return InputUnit(text=text, context=context, kind="list_item")


def test_an_unchecked_item_is_dropped():
    units = (_item("[ ] splice the stern line", "Jobs"), _item("[x] oil the winch", "Jobs"))

    assert filter_units(units, BASE) == (units[1],)


def test_an_unchecked_item_under_a_keep_word_is_kept():
    units = (_item("[ ] man overboard", "Saturday drill"),)

    assert filter_units(units, BASE) == units


def test_a_keep_word_matches_in_any_case_and_in_the_plural():
    units = (
        _item("[ ] fire", "DRILLS for the season"),
        _item("[ ] fog", "Autumn", "Night Drills"),
    )

    assert filter_units(units, BASE) == units


def test_a_keep_phrase_matches_as_a_phrase():
    units = (_item("[ ] the red ketch", "My watch list"), _item("[ ] the tide", "Watch the list"))

    assert filter_units(units, BASE) == (units[0],)


def test_a_keep_word_inside_another_word_does_not_keep():
    units = (_item("[ ] the bit", "Mandrill sizes"),)

    assert filter_units(units, BASE) == ()


def test_a_keep_word_in_the_items_own_text_does_not_keep():
    units = (_item("[ ] run the drill", "Jobs"),)

    assert filter_units(units, BASE) == ()


def test_only_the_named_state_is_filtered():
    units = (
        _item("splice the stern line", "Jobs"),
        _item("[x] oil the winch", "Jobs"),
        InputUnit(text="[ ] is how an open box is written", context=(), kind="prose"),
    )

    assert filter_units(units, BASE) == units


def test_the_filter_reads_the_units_the_splitter_gives():
    units = split_structured("Jobs:\n- [ ] splice the stern line\n- [x] oil the winch\n")

    assert [unit.text for unit in filter_units(units, BASE)] == ["[x] oil the winch"]


def test_a_pack_without_an_item_filter_keeps_every_unit():
    pack = build_pack(
        manifest=BASE.manifest,
        node_types=BASE.node_types,
        edge_types=BASE.edge_types,
        extractable_node_types=BASE.extractable_node_types,
        extractable_edge_types=BASE.extractable_edge_types,
    )
    units = (_item("[ ] splice the stern line", "Jobs"),)

    assert filter_units(units, pack) == units


def test_the_plan_vocabulary_is_the_packs_rule():
    assert plan_vocabulary(BASE) == PlanWords(
        entity_types=("Deckhand", "Mooring"),
        predicates=("CREWS_WITH", "MOORED_AT"),
        fan_out_predicates=("SERVES_UNDER",),
    )


def test_a_second_plan_vocabulary_rule_adds_its_words_in_order():
    assert plan_vocabulary(PET) == PlanWords(
        entity_types=("Deckhand", "Mooring", "Pet"),
        predicates=("CREWS_WITH", "MOORED_AT", "HAS_PET"),
        fan_out_predicates=("SERVES_UNDER", "HAS_PET"),
    )
