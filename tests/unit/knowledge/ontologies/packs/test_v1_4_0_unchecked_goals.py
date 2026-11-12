"""The v1.4.0 pack's unchecked-box filter: an unchecked item reaches the model only under goals.

Rule 29 gives an unchecked `[ ]` item no edge, except `HAS_GOAL` under a goals lead-in.
The pack's `unchecked-goals` rule applies the first half before the model is called: it
drops an unchecked item unless its lead-in, or a heading or parent bullet above it,
holds one of the goal words of ADR-054's v14 amendment (register row 4), as a whole
word in any case, singular or plural. An itinerary line with no box is no unchecked
item, so it is kept for Rule 29's Event handling.

The cases are written with invented names. The first is the shape of a long learning
backlog of dev-lists, where two boxes are checked and the rest are not.
"""

from __future__ import annotations

import pytest

from backend.knowledge.extraction.pre_llm import filter_units
from backend.knowledge.extraction.structure_splitter import InputUnit, split_structured
from backend.knowledge.ontologies import registry
from backend.knowledge.ontologies.pack import PRE_LLM, Pack
from backend.knowledge.ontologies.packs.v1_4_0 import manifest as pack_manifest
from tests.unit.knowledge.ontologies.packs.test_register import (
    carriers,
    quoted_words,
    read_register,
)

RULE_ID = "unchecked-goals"
GOAL_WORDS = ("goal", "objective", "resolution", "aim", "aspiration", "target", "bucket list")
_ROW_KEY = "List rules, unchecked box"

# Two checked boxes and ten unchecked ones under a lead-in that names no goal.
STUDY_PILE = (
    "Study pile:\n"
    "- [x] Quorvex query planning\n"
    "- [x] the basics of Brindle\n"
    "- [ ] Tessaly\n"
    "- [ ] Marlowe LiveGrid\n"
    "- [ ] a matrix calculus primer\n"
    "- [ ] sequential hypothesis testing\n"
    "- [ ] shorthand\n"
    "- [ ] Pellucid modules\n"
    "- [ ] kernel tracing with Fennick\n"
    "- [ ] contract testing\n"
    "- [ ] conversational Basque\n"
    "- [ ] bookbinding"
)


@pytest.fixture(scope="module")
def pack() -> Pack:
    return registry.get_pack(pack_manifest.MANIFEST.version)


def kept(text: str, pack: Pack) -> list[str]:
    """The text of each unit of `text` the pack gives the model."""
    return [unit.text for unit in filter_units(split_structured(text), pack)]


# --- the rule ---------------------------------------------------------------------------


def test_the_rule_drops_unchecked_items_and_keeps_those_under_the_goal_words(pack: Pack):
    (rule,) = (rule for rule in pack.rules if rule.id == RULE_ID)

    assert (rule.kind, rule.phase, rule.prompt_rule) == ("item_filter", PRE_LLM, "Rule 29")
    assert rule.params.state == "unchecked"
    assert rule.params.keep_context_words == GOAL_WORDS


def test_it_is_the_packs_one_item_filter(pack: Pack):
    assert [rule.id for rule in pack.rules_of("item_filter")] == [RULE_ID]


def test_the_unchecked_box_row_names_the_rule_and_quotes_its_words(pack: Pack):
    (row,) = (row for row in read_register(pack.labelling).rows if row.key == _ROW_KEY)
    (rule,) = (rule for rule in pack.rules if rule.id == RULE_ID)

    assert RULE_ID in carriers(row.carrier).ids
    assert sorted(quoted_words(row.convention)) == sorted(rule.params.keep_context_words)
    assert row.classes == "prompt rule and pre-LLM rule"


# --- the shape of the learning backlog ----------------------------------------------------


def test_a_backlog_keeps_its_checked_items_and_drops_its_unchecked_ones(pack: Pack):
    assert kept(STUDY_PILE, pack) == ["[x] Quorvex query planning", "[x] the basics of Brindle"]


def test_a_backlog_with_no_box_checked_leaves_no_unit(pack: Pack):
    text = "Study pile:\n- [ ] Tessaly\n- [ ] bookbinding"

    assert kept(text, pack) == []


# --- itinerary items ----------------------------------------------------------------------


def test_an_unchecked_itinerary_item_is_dropped_and_a_line_with_no_box_kept(pack: Pack):
    text = (
        "Saturday in Corbridge:\n"
        "- [ ] 10:00 the Harlow print museum\n"
        "- 13:00 lunch at the Weir House\n"
        "- [x] 15:00 the abbey tour"
    )

    assert kept(text, pack) == ["13:00 lunch at the Weir House", "[x] 15:00 the abbey tour"]


# --- the goal words -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "lead_in",
    [
        "Goals for 2027:",
        "My objective this quarter:",
        "New Year's resolutions:",
        "Aims for the allotment:",
        "Aspirations:",
        "Targets for March:",
        "Bucket list:",
    ],
    ids=list(GOAL_WORDS),
)
def test_an_unchecked_item_under_a_goal_word_is_kept(pack: Pack, lead_in: str):
    text = f"{lead_in}\n- [ ] sail to Haskeir\n- [x] the Aldmoor fell race"

    assert kept(text, pack) == ["[ ] sail to Haskeir", "[x] the Aldmoor fell race"]


@pytest.mark.parametrize("lead_in", ["GOALS:", "my bucket lists:", "Two targets:", "Aim:"])
def test_a_goal_word_is_matched_in_any_case_singular_or_plural(pack: Pack, lead_in: str):
    assert kept(f"{lead_in}\n- [ ] sail to Haskeir", pack) == ["[ ] sail to Haskeir"]


def test_a_goal_word_in_a_heading_two_levels_above_the_item_keeps_it(pack: Pack):
    text = "# 2027 goals\n## Health\n- [ ] run the Aldmoor half marathon"

    (unit,) = filter_units(split_structured(text), pack)
    assert unit == InputUnit(
        text="[ ] run the Aldmoor half marathon",
        context=("2027 goals", "Health"),
        kind="list_item",
    )


def test_a_goal_word_in_a_parent_bullet_keeps_the_items_under_it(pack: Pack):
    text = "Plans:\n- long-term aims\n  - [ ] learn to sail\n- [ ] repaint the shed"

    assert kept(text, pack) == ["long-term aims", "[ ] learn to sail"]


@pytest.mark.parametrize(
    "lead_in", ["Goalkeeping drills:", "Aiming high this week:", "Targeted ads to block:"]
)
def test_a_word_that_only_contains_a_goal_word_keeps_nothing(pack: Pack, lead_in: str):
    assert kept(f"{lead_in}\n- [ ] dive left", pack) == []


def test_a_goal_word_in_the_items_own_text_keeps_nothing(pack: Pack):
    assert kept("Errands:\n- [ ] set a savings target", pack) == []
