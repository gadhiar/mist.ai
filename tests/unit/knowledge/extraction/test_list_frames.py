"""List frames: how a structured turn's units group, render and parse for the list path.

The list path (`backend.extraction_service.engine`, T19) sends a structured turn to one
frame call and a few item calls. These tests pin the grouping those calls see, the render
they read, and the strict parsers that turn their outputs into frames and item records.
Every input is synthetic, with invented names.
"""

from __future__ import annotations

from backend.knowledge.extraction.list_frames import (
    ListGroup,
    group_units,
    render_frame_input,
)
from backend.knowledge.extraction.structure_splitter import InputUnit, split_structured


def _item(text: str, *context: str) -> InputUnit:
    return InputUnit(text=text, context=tuple(context), kind="list_item")


def _texts(group: ListGroup) -> list[str]:
    return [u.text for u in group.items]


# --- grouping ---------------------------------------------------------------------------------


def test_a_flat_list_under_a_lead_in_is_one_group_with_numbered_items():
    units = split_structured("My kit:\n- Quorvane for notes\n- Brellix\n- Tamsic")

    groups, prose = group_units(units)

    assert prose == ()
    assert [g.gid for g in groups] == ["G1"]
    assert groups[0].context == ("My kit",)
    assert _texts(groups[0]) == ["Quorvane for notes", "Brellix", "Tamsic"]
    assert groups[0].item_ids == ("G1.1", "G1.2", "G1.3")
    assert groups[0].parent is None


def test_a_nested_list_makes_a_child_group_linked_to_its_parent_item():
    units = split_structured("Stack:\n- Quorvane\n  - Brellix plugin\n  - Tamsic plugin\n- Ostrel")

    groups, _ = group_units(units)

    assert [(g.gid, g.context, _texts(g)) for g in groups] == [
        ("G1", ("Stack",), ["Quorvane", "Ostrel"]),
        ("G2", ("Stack", "Quorvane"), ["Brellix plugin", "Tamsic plugin"]),
    ]
    assert groups[1].parent == ("G1", "G1.1")


def test_a_sibling_after_a_nested_child_stays_in_its_group():
    units = split_structured("- Quorvane\n  - Brellix\n- Ostrel\n  - Tamsic\n- Pindle")

    groups, _ = group_units(units)

    assert [(g.gid, _texts(g), g.parent) for g in groups] == [
        ("G1", ["Quorvane", "Ostrel", "Pindle"], None),
        ("G2", ["Brellix"], ("G1", "G1.1")),
        ("G3", ["Tamsic"], ("G1", "G1.2")),
    ]


def test_a_parent_item_ending_in_a_colon_is_found():
    units = split_structured("Servers:\n- Marrowgate:\n  - Velquist in front of it")

    groups, _ = group_units(units)

    assert groups[1].context == ("Servers", "Marrowgate")
    assert groups[1].parent == ("G1", "G1.1")


def test_a_checked_parent_item_is_found():
    units = split_structured("- [x] Quorvane:\n  - [ ] Brellix")

    groups, _ = group_units(units)

    assert groups[1].parent == ("G1", "G1.1")


def test_checkbox_items_keep_their_marks_in_one_group():
    units = split_structured("Goals this year:\n- [x] learn to weld\n- [ ] run a half marathon")

    groups, _ = group_units(units)

    assert _texts(groups[0]) == ["[x] learn to weld", "[ ] run a half marathon"]


def test_table_rows_under_a_lead_in_are_one_group():
    text = "Team:\n| Name | Role |\n|---|---|\n| Iselde | lead |\n| Tomasz | tester |"

    groups, prose = group_units(split_structured(text))

    assert prose == ()
    assert [(g.gid, g.context, _texts(g)) for g in groups] == [
        ("G1", ("Team",), ["Name: Iselde; Role: lead", "Name: Tomasz; Role: tester"]),
    ]


def test_prose_between_two_lists_ends_the_run_and_is_returned_in_order():
    text = (
        "We moved offices last week.\n\n"
        "- Quorvane\n- Brellix\n\n"
        "Then the second list.\n\n"
        "- Ostrel\n\n"
        "That was all."
    )

    groups, prose = group_units(split_structured(text))

    assert [(g.gid, _texts(g)) for g in groups] == [
        ("G1", ["Quorvane", "Brellix"]),
        ("G2", ["Ostrel"]),
    ]
    assert [p.text for p in prose] == [
        "We moved offices last week.",
        "Then the second list.",
        "That was all.",
    ]


def test_items_under_two_headings_are_two_groups():
    text = "# Home\n- Quorvane\n# Work\n- Brellix\n- Ostrel"

    groups, _ = group_units(split_structured(text))

    assert [(g.context, _texts(g)) for g in groups] == [
        (("Home",), ["Quorvane"]),
        (("Work",), ["Brellix", "Ostrel"]),
    ]


def test_hand_built_units_group_by_context_and_nesting():
    units = (
        _item("Quorvane", "Tools"),
        _item("Brellix", "Tools", "Quorvane"),
        _item("Ostrel", "Tools"),
        InputUnit(text="Nothing else.", context=(), kind="prose"),
        _item("Pindle", "Tools"),
    )

    groups, prose = group_units(units)

    assert [(g.gid, _texts(g)) for g in groups] == [
        ("G1", ["Quorvane", "Ostrel"]),
        ("G2", ["Brellix"]),
        ("G3", ["Pindle"]),
    ]
    assert [p.text for p in prose] == ["Nothing else."]


def test_an_all_prose_turn_has_no_groups():
    groups, prose = group_units(split_structured("I like plain sentences."))

    assert groups == ()
    assert [p.text for p in prose] == ["I like plain sentences."]


# --- render -----------------------------------------------------------------------------------


def test_the_render_lists_prose_then_each_group_with_its_context_once():
    text = (
        "We should finish the move this week.\n\n"
        "Things I run at work:\n"
        "- Quorvane for editing\n"
        "- Marrowgate:\n"
        "  - Velquist in front of it"
    )
    groups, prose = group_units(split_structured(text))

    assert render_frame_input(groups, prose) == (
        "P1 We should finish the move this week.\n"
        'G1 under "Things I run at work"\n'
        "  G1.1 Quorvane for editing\n"
        "  G1.2 Marrowgate:\n"
        'G2 under "Things I run at work" > "Marrowgate" (parent G1.2)\n'
        "  G2.1 Velquist in front of it"
    )


def test_a_group_without_context_and_prose_under_a_heading_render_plainly():
    units = (
        InputUnit(text="Quick note\non two lines", context=("Log",), kind="prose"),
        _item("Quorvane"),
    )
    groups, prose = group_units(units)

    assert render_frame_input(groups, prose) == (
        'P1 under "Log": Quick note on two lines\nG1\n  G1.1 Quorvane'
    )


def test_the_render_is_deterministic():
    text = "Stack:\n- Quorvane\n  - Brellix\n- Ostrel\n\nDone."
    first = render_frame_input(*group_units(split_structured(text)))

    assert all(render_frame_input(*group_units(split_structured(text))) == first for _ in range(5))
