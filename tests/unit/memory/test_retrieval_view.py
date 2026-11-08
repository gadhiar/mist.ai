"""Tests for backend.memory.retrieval_view: the fact and note texts the models score.

The fixture facts come from tests/fixtures/memory_recall/facts.jsonl through the run harness's
`fact_row`, the rows the harness's fact index renders, so F0 is checked against `render_fact`
on exactly the rows a probe run indexes.
"""

from __future__ import annotations

import dataclasses
from datetime import timedelta, timezone

import pytest

from backend.memory.fact_render import FACT_TEMPLATES, render_fact
from backend.memory.models import FactRow
from backend.memory.retrieval_view import (
    FACT_VIEWS,
    NOTE_VIEWS,
    SECOND_PERSON_TEMPLATES,
    NoteViewSpec,
    ViewSpec,
    render_fact_view,
    render_note_view,
)
from scripts.eval_harness.memory_recall import load_facts
from scripts.eval_harness.memory_recall_run import fact_row

# fact-01's expected fact: Wren (the user) prefers cortado, valid from 12 January 2026.
_CORTADO_KEY = "5cd20d1ba1287001b505ce56cba15b397c5fb5c6df138945e580ea25a3fbcee2"


def _row(
    *,
    subject_name: str = "Wren",
    subject_type: str | None = "User",
    rel_type: str = "WORKS_AT",
    object_name: str = "Brightgrid",
    object_type: str | None = "Organization",
    valid_from: str | None = "2024-03-05",
    valid_to: str | None = None,
) -> FactRow:
    return FactRow(
        row_key="r1",
        fact_key="k1",
        subject_id="s1",
        subject_name=subject_name,
        subject_type=subject_type,
        rel_type=rel_type,
        object_id="o1",
        object_name=object_name,
        object_type=object_type,
        valid_from=valid_from,
        valid_to=valid_to,
        recorded_at="2024-03-05T10:00:00+00:00",
        confidence=0.9,
        source_utterance_id=None,
    )


def _fixture_rows() -> list[FactRow]:
    return [fact_row(fact) for fact in load_facts()]


def _cortado() -> FactRow:
    [row] = [row for row in _fixture_rows() if row.fact_key == _CORTADO_KEY]
    return row


class TestSpecs:
    def test_fact_views_are_f0_to_f3(self):
        assert list(FACT_VIEWS) == ["F0", "F1", "F2", "F3"]
        assert FACT_VIEWS["F0"] == ViewSpec("F0", False, False, True)
        assert FACT_VIEWS["F1"] == ViewSpec("F1", True, False, True)
        assert FACT_VIEWS["F2"] == ViewSpec("F2", True, True, True)
        assert FACT_VIEWS["F3"] == ViewSpec("F3", True, True, False)
        for key, spec in FACT_VIEWS.items():
            assert spec.id == key

    def test_note_views_are_n0_and_n1(self):
        assert {"N0": NoteViewSpec("N0", False), "N1": NoteViewSpec("N1", True)} == NOTE_VIEWS

    def test_specs_are_frozen_with_slots(self):
        for spec in (*FACT_VIEWS.values(), *NOTE_VIEWS.values()):
            with pytest.raises(dataclasses.FrozenInstanceError):
                spec.id = "X"  # type: ignore[misc]
            assert not hasattr(spec, "__dict__")

    def test_view_tables_are_read_only(self):
        with pytest.raises(TypeError):
            FACT_VIEWS["F9"] = FACT_VIEWS["F0"]  # type: ignore[index]
        with pytest.raises(TypeError):
            SECOND_PERSON_TEMPLATES["USES"] = "x"  # type: ignore[index]


class TestSecondPersonTemplates:
    def test_keyed_exactly_by_fact_templates(self):
        assert set(SECOND_PERSON_TEMPLATES) == set(FACT_TEMPLATES)

    @pytest.mark.parametrize("rel_type", sorted(FACT_TEMPLATES))
    def test_each_template_starts_with_you_and_ends_with_the_object(self, rel_type):
        template = SECOND_PERSON_TEMPLATES[rel_type]
        assert template.startswith("You ")
        assert template.endswith("{o}")
        assert "{s}" not in template

    @pytest.mark.parametrize(
        ("rel_type", "expected"),
        [
            ("WORKS_AT", "You work at Brightgrid since 5 March 2024."),
            ("HAS_HABIT", "You have the habit: Brightgrid since 5 March 2024."),
            ("INTERESTED_IN", "You are interested in Brightgrid since 5 March 2024."),
            ("STRUGGLES_WITH", "You struggle with Brightgrid since 5 March 2024."),
        ],
    )
    def test_verbs_agree_with_you(self, rel_type, expected):
        assert render_fact_view(_row(rel_type=rel_type), FACT_VIEWS["F1"]) == expected

    def test_a_type_missing_from_both_tables_falls_back_as_render_fact_does(self):
        row = _row(rel_type="SAILED_WITH", subject_type="Person", subject_name="Ana")
        assert render_fact(row) == "Ana sailed with Brightgrid since 5 March 2024."
        assert render_fact_view(row, FACT_VIEWS["F1"]) == render_fact(row)

        user_row = _row(rel_type="SAILED_WITH")
        assert render_fact_view(user_row, FACT_VIEWS["F1"]) == (
            "You sailed with Brightgrid since 5 March 2024."
        )


class TestFactViews:
    def test_f0_equals_render_fact_for_every_fixture_fact_row(self):
        rows = _fixture_rows()
        assert rows
        for row in rows:
            assert render_fact_view(row, FACT_VIEWS["F0"]) == render_fact(row), row.row_key

    def test_f0_passes_the_zone_through(self):
        row = _row(valid_from="2024-03-05T23:30:00+00:00")
        zone = timezone(timedelta(hours=2))
        assert render_fact_view(row, FACT_VIEWS["F0"], zone) == render_fact(row, zone)
        assert render_fact_view(row, FACT_VIEWS["F1"], zone).endswith("since 6 March 2024.")

    def test_fact_01_under_f1_f2_f3(self):
        row = _cortado()
        assert render_fact(row) == "Wren prefers cortado since 12 January 2026."
        assert render_fact_view(row, FACT_VIEWS["F1"]) == (
            "You prefer cortado since 12 January 2026."
        )
        assert render_fact_view(row, FACT_VIEWS["F2"]) == (
            "You prefer cortado (preference) since 12 January 2026."
        )
        assert render_fact_view(row, FACT_VIEWS["F3"]) == "You prefer cortado (preference)."

    def test_f2_glosses_the_object_with_its_first_type_term(self):
        assert render_fact_view(_row(), FACT_VIEWS["F2"]) == (
            "You work at Brightgrid (organization) since 5 March 2024."
        )

    def test_a_person_subject_with_the_user_as_object_reads_knows_you(self):
        row = _row(
            subject_name="Ana",
            subject_type="Person",
            rel_type="KNOWS",
            object_name="Wren",
            object_type="User",
            valid_from=None,
        )
        assert render_fact(row) == "Ana knows Wren."
        assert render_fact_view(row, FACT_VIEWS["F1"]) == "Ana knows you."
        # The user as object is "you", never glossed "(user)".
        assert render_fact_view(row, FACT_VIEWS["F2"]) == "Ana knows you."

    def test_learned_self_whose_subject_is_mist_keeps_the_third_person(self):
        row = _row(
            subject_name="MIST.AI",
            subject_type="MistIdentity",
            rel_type="LEARNED_SELF",
            object_name="short answers suit mornings",
            object_type=None,
            valid_from=None,
        )
        expected = "MIST.AI learned about herself: short answers suit mornings."
        assert render_fact(row) == expected
        assert render_fact_view(row, FACT_VIEWS["F1"]) == expected
        assert render_fact_view(row, FACT_VIEWS["F2"]) == expected

    def test_a_fact_about_neither_the_user_nor_a_typed_object_reads_as_f0(self):
        row = _row(subject_name="Ana", subject_type="Person", object_type=None, valid_to="2025-01")
        for view in FACT_VIEWS.values():
            if view.qualifiers:
                assert render_fact_view(row, view) == render_fact(row), view.id

    def test_a_typed_object_of_a_third_person_fact_is_glossed(self):
        row = _row(subject_name="Ana", subject_type="Person")
        assert render_fact_view(row, FACT_VIEWS["F1"]) == render_fact(row)
        assert render_fact_view(row, FACT_VIEWS["F2"]) == (
            "Ana works at Brightgrid (organization) since 5 March 2024."
        )

    def test_an_object_type_outside_type_terms_adds_no_gloss(self):
        row = _row(object_type="MistTrait")
        assert render_fact_view(row, FACT_VIEWS["F2"]) == (
            "You work at Brightgrid since 5 March 2024."
        )

    def test_f3_drops_every_qualifier(self):
        row = _row(valid_from="2020", valid_to="2024-06")
        assert render_fact(row) == "Wren works at Brightgrid from 2020 until June 2024."
        assert render_fact_view(row, FACT_VIEWS["F3"]) == "You work at Brightgrid (organization)."


class TestNoteViews:
    def test_n0_is_the_content(self):
        assert render_note_view("Focaccia", "Method", "Cold ferment.", NOTE_VIEWS["N0"]) == (
            "Cold ferment."
        )

    def test_n1_is_title_heading_and_content_joined_by_full_stops(self):
        assert render_note_view("Focaccia", "Method", "Cold ferment.", NOTE_VIEWS["N1"]) == (
            "Focaccia. Method. Cold ferment."
        )

    def test_n1_leaves_out_the_heading_of_a_file_level_chunk(self):
        assert render_note_view("Focaccia", None, "Cold ferment.", NOTE_VIEWS["N1"]) == (
            "Focaccia. Cold ferment."
        )

    def test_n1_leaves_out_a_missing_title_and_adds_no_second_stop(self):
        n1 = NOTE_VIEWS["N1"]
        assert render_note_view(None, "Why?", "Cold ferment.", n1) == "Why? Cold ferment."
        assert render_note_view("Done.", "  ", "Cold ferment.", n1) == "Done. Cold ferment."
