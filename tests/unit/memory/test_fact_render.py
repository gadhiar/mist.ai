"""Tests for backend.memory.fact_render: one sentence per fact, with its temporal qualifiers."""

from datetime import timedelta, timezone

import pytest

from backend.memory.fact_render import FACT_TEMPLATES, render_fact
from backend.memory.models import FactRow, fact_key

# One expected sentence per template, subject "Sam" and object "Kestrel". Adding a template
# without a row here fails `test_every_template_has_an_expected_sentence`.
_EXPECTED = {
    "USES": "Sam uses Kestrel.",
    "KNOWS": "Sam knows Kestrel.",
    "WORKS_ON": "Sam works on Kestrel.",
    "WORKS_AT": "Sam works at Kestrel.",
    "WORKS_WITH": "Sam works with Kestrel.",
    "INTERESTED_IN": "Sam is interested in Kestrel.",
    "HAS_GOAL": "Sam has the goal: Kestrel.",
    "PREFERS": "Sam prefers Kestrel.",
    "DISLIKES": "Sam dislikes Kestrel.",
    "EXPERT_IN": "Sam is an expert in Kestrel.",
    "LEARNING": "Sam is learning Kestrel.",
    "STRUGGLES_WITH": "Sam struggles with Kestrel.",
    "DECIDED": "Sam decided Kestrel.",
    "EXPERIENCED": "Sam experienced Kestrel.",
    "RECOMMENDS": "Sam recommends Kestrel.",
    "HAS_HABIT": "Sam has the habit: Kestrel.",
    "KNOWS_PERSON": "Sam knows Kestrel.",
    "MEMBER_OF": "Sam is a member of Kestrel.",
    "LEARNED_FROM": "Sam learned from Kestrel.",
    "IS_A": "Sam is a kind of Kestrel.",
    "PART_OF": "Sam is part of Kestrel.",
    "RELATED_TO": "Sam is related to Kestrel.",
    "DEPENDS_ON": "Sam depends on Kestrel.",
    "USED_FOR": "Sam is used for Kestrel.",
    "OCCURRED_ON": "Sam happened on Kestrel.",
    "HAS_METRIC": "Sam has the metric: Kestrel.",
    "REFERENCES_DOCUMENT": "Sam references the document Kestrel.",
    "PRECEDED_BY": "Sam came after Kestrel.",
    "MECHANISM_OF": "Sam is a mechanism of Kestrel.",
    "OPERATES_ON": "Sam operates on Kestrel.",
    "INPUT_TO": "Sam is an input to Kestrel.",
    "IMPROVES": "Sam improves Kestrel.",
    "COMPRISES": "Sam comprises Kestrel.",
    "APPLICABLE_TO": "Sam applies to Kestrel.",
    "STRATEGY_FOR": "Sam is a strategy for Kestrel.",
    "NAMING_CONVENTION_OF": "Sam is a naming convention for Kestrel.",
    "HAS_TRAIT": "Sam has the trait: Kestrel.",
    "HAS_CAPABILITY": "Sam has the capability: Kestrel.",
    "HAS_PREFERENCE": "Sam has the preference: Kestrel.",
    "IS_UNCERTAIN_ABOUT": "Sam is uncertain about Kestrel.",
    "ADAPTED_FOR": "Sam is adapted for Kestrel.",
    "LEARNED_SELF": "Sam learned about itself: Kestrel.",
    "IMPLEMENTED_WITH": "Sam is implemented with Kestrel.",
    "MIST_HAS_TRAIT": "Sam has the trait: Kestrel.",
    "MIST_HAS_CAPABILITY": "Sam has the capability: Kestrel.",
    "MIST_HAS_PREFERENCE": "Sam has the preference: Kestrel.",
}


def _row(
    rel_type: str = "USES",
    *,
    valid_from: str | None = None,
    valid_to: str | None = None,
    subject: str = "Sam",
    obj: str = "Kestrel",
) -> FactRow:
    return FactRow(
        row_key="v-1",
        fact_key=fact_key("sam", rel_type, "kestrel"),
        subject_id="sam",
        subject_name=subject,
        rel_type=rel_type,
        object_id="kestrel",
        object_name=obj,
        valid_from=valid_from,
        valid_to=valid_to,
        recorded_at="2026-10-19T18:00:00+00:00",
        confidence=0.9,
        source_utterance_id="utt-1",
    )


def test_every_template_has_an_expected_sentence():
    assert set(FACT_TEMPLATES) == set(_EXPECTED)


@pytest.mark.parametrize(("rel_type", "expected"), sorted(_EXPECTED.items()))
def test_each_template_renders_one_sentence(rel_type, expected):
    assert render_fact(_row(rel_type)) == expected


def test_unknown_relation_falls_back_to_the_generic_template():
    assert render_fact(_row("GREW_UP_IN")) == "Sam grew up in Kestrel."


def test_generic_fallback_keeps_qualifiers():
    assert (
        render_fact(_row("GREW_UP_IN", valid_from="1999")) == "Sam grew up in Kestrel since 1999."
    )


class TestTemporalQualifiers:
    def test_open_start_adds_since(self):
        assert render_fact(_row(valid_from="2024-03-05")) == "Sam uses Kestrel since 5 March 2024."

    def test_end_only_adds_until(self):
        assert render_fact(_row(valid_to="2025-06")) == "Sam uses Kestrel until June 2025."

    def test_both_bounds_add_from_until(self):
        sentence = render_fact(_row(valid_from="2024", valid_to="2025-06-01"))

        assert sentence == "Sam uses Kestrel from 2024 until 1 June 2025."

    def test_always_sentinel_adds_nothing(self):
        assert render_fact(_row(valid_from="-inf")) == "Sam uses Kestrel."

    def test_unparsable_bound_is_dropped(self):
        assert render_fact(_row(valid_from="sometime", valid_to="2025")) == (
            "Sam uses Kestrel until 2025."
        )

    def test_datetime_bound_renders_the_day_in_the_given_zone(self):
        """23:30 UTC on 4 March is already 5 March at UTC+2."""
        row = _row(valid_from="2024-03-04T23:30:00+00:00")
        plus_two = timezone(timedelta(hours=2))

        assert render_fact(row) == "Sam uses Kestrel since 4 March 2024."
        assert render_fact(row, tz=plus_two) == "Sam uses Kestrel since 5 March 2024."

    def test_naive_datetime_bound_is_read_as_utc(self):
        row = _row(valid_from="2024-03-04T23:30:00")

        assert render_fact(row, tz=timezone(timedelta(hours=2))) == (
            "Sam uses Kestrel since 5 March 2024."
        )


def test_always_sentinel_matches_the_interval_module():
    """fact_render copies the sentinel to avoid importing the curation package; pin the copy."""
    from backend.knowledge.curation.intervals import ALWAYS
    from backend.memory.fact_render import _ALWAYS

    assert _ALWAYS == ALWAYS


def test_names_are_rendered_verbatim():
    row = _row("WORKS_AT", subject="User", obj="Acme Labs")

    assert render_fact(row) == "User works at Acme Labs."
