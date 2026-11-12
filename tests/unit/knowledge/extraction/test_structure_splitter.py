"""The structure splitter: lists, tables and headings become units that keep their parents.

Stage 2 of the extraction service extracts a structured utterance in batches of
units (`backend.extraction_service.engine`); these tests pin how an utterance
is cut into units, how a unit renders with its context, how units batch, and
how the batches' payloads merge. Every one of the 60 dev gold utterances must
come back as one unchanged prose unit, so a turn without structure reaches the
model exactly as before.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from backend.extraction_contract.models import ExtractionPayload
from backend.knowledge.extraction.structure_splitter import (
    LIST_MERGE_TYPE_CONFLICT,
    InputUnit,
    batch_units,
    merge_payloads,
    render_batch,
    render_unit,
    split_structured,
)

_GOLD = Path(__file__).resolve().parents[4] / "data" / "ingest" / "extraction-gold-2026-06-14.jsonl"


def _gold_utterances() -> list[tuple[str, str]]:
    rows = [json.loads(line) for line in _GOLD.read_text(encoding="utf-8").splitlines() if line]
    return [(row["tag"], row["utterance"]) for row in rows]


def item(text: str, *context: str) -> InputUnit:
    return InputUnit(text=text, context=context, kind="list_item")


def prose(text: str, *context: str) -> InputUnit:
    return InputUnit(text=text, context=context, kind="prose")


def row(text: str, *context: str) -> InputUnit:
    return InputUnit(text=text, context=context, kind="table_row")


# ---------------------------------------------------------------------------
# The unit
# ---------------------------------------------------------------------------


class TestInputUnit:
    def test_frozen_with_slots(self):
        unit = prose("I use Rust")

        with pytest.raises(dataclasses.FrozenInstanceError):
            unit.text = "edited"  # type: ignore[misc]
        assert not hasattr(unit, "__dict__")


# ---------------------------------------------------------------------------
# Text with no structure comes back whole
# ---------------------------------------------------------------------------


class TestNoStructure:
    def test_the_60_gold_utterances_are_each_one_unchanged_prose_unit(self):
        utterances = _gold_utterances()
        assert len(utterances) == 60

        split = {tag: split_structured(text) for tag, text in utterances}

        assert split == {tag: (prose(text),) for tag, text in utterances}

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "I use Rust for my backend work",
            "Correction: I'm not actually learning Elixir",
            "  I moved to Berlin.  \n\nThen I picked up German.\n",
            "I ran 3.5 miles on 2026-05-20, then 10 more the week after",
            "Here is my config\n```\n- name: build\n# not a heading\n```\nIt works.",
        ],
        ids=["empty", "one-line", "inline-colon", "paragraphs", "numbers", "fence-only"],
    )
    def test_prose_without_a_list_table_or_heading_is_returned_exactly(self, text):
        assert split_structured(text) == (prose(text),)

    def test_a_line_ending_in_a_colon_with_nothing_after_it_is_prose(self):
        assert split_structured("Things I need to remember:") == (
            prose("Things I need to remember:"),
        )


# ---------------------------------------------------------------------------
# Lists
# ---------------------------------------------------------------------------


class TestLists:
    def test_a_single_bullet(self):
        assert split_structured("- I use Rust") == (item("I use Rust"),)

    @pytest.mark.parametrize("marker", ["-", "*", "+", "•"])
    def test_each_bullet_marker(self, marker):
        assert split_structured(f"{marker} eggs\n{marker} milk") == (item("eggs"), item("milk"))

    @pytest.mark.parametrize("marker", ["1.", "1)", "a.", "a)"])
    def test_each_numbering_style(self, marker):
        second = marker.replace("1", "2").replace("a", "b")

        assert split_structured(f"{marker} eggs\n{second} milk") == (item("eggs"), item("milk"))

    def test_a_lead_in_line_is_context_for_the_list_after_it(self):
        text = "Things I use every day:\n- Rust\n- Neovim\n\nI also like hiking."

        assert split_structured(text) == (
            item("Rust", "Things I use every day"),
            item("Neovim", "Things I use every day"),
            prose("I also like hiking."),
        )

    def test_a_blank_line_between_lead_in_and_list_keeps_the_lead_in(self):
        assert split_structured("Groceries:\n\n- eggs") == (item("eggs", "Groceries"),)

    def test_nested_items_carry_their_parent_items(self):
        text = (
            "Trip plan:\n"
            "- Day 1\n"
            "  - visit the museum\n"
            "  - dinner with Lena\n"
            "    - book a table\n"
            "- Day 2\n"
            "\t1. hike\n"
        )

        assert split_structured(text) == (
            item("Day 1", "Trip plan"),
            item("visit the museum", "Trip plan", "Day 1"),
            item("dinner with Lena", "Trip plan", "Day 1"),
            item("book a table", "Trip plan", "Day 1", "dinner with Lena"),
            item("Day 2", "Trip plan"),
            item("hike", "Trip plan", "Day 2"),
        )

    def test_a_parent_item_ending_in_a_colon_is_context_without_it(self):
        text = "- Languages:\n  - Rust\n  - Go"

        assert split_structured(text) == (
            item("Languages:"),
            item("Rust", "Languages"),
            item("Go", "Languages"),
        )

    def test_checkboxes_keep_their_checked_state(self):
        text = (
            "Before the trip:\n- [ ] renew passport\n- [x] book flights\n* [X] pack\n[ ] call mum"
        )

        assert split_structured(text) == (
            item("[ ] renew passport", "Before the trip"),
            item("[x] book flights", "Before the trip"),
            item("[x] pack", "Before the trip"),
            item("[ ] call mum", "Before the trip"),
        )

    def test_an_indented_line_without_a_marker_continues_the_item(self):
        text = "- I started using DuckDB\n  for the analytics work\n- I use Bun"

        assert split_structured(text) == (
            item("I started using DuckDB for the analytics work"),
            item("I use Bun"),
        )

    def test_mixed_prose_and_list(self):
        text = (
            "I picked up a few tools this year.\n"
            "- Bun\n"
            "- DuckDB\n"
            "Both have been great.\n"
            "My mentor Priya suggested them."
        )

        assert split_structured(text) == (
            prose("I picked up a few tools this year."),
            item("Bun"),
            item("DuckDB"),
            prose("Both have been great.\nMy mentor Priya suggested them."),
        )

    def test_prose_after_a_list_drops_the_lead_in(self):
        text = "My stack:\n- Rust\nAnd separately, I journal every night."

        assert split_structured(text) == (
            item("Rust", "My stack"),
            prose("And separately, I journal every night."),
        )

    def test_a_lead_in_reaches_prose_before_its_list(self):
        text = "Shopping for Saturday:\nI need these for the party.\n- eggs"

        assert split_structured(text) == (
            prose("I need these for the party.", "Shopping for Saturday"),
            item("eggs", "Shopping for Saturday"),
        )

    def test_two_lead_ins_in_a_row_nest(self):
        text = "Plans:\nWeekend:\n- hike"

        assert split_structured(text) == (item("hike", "Plans", "Weekend"),)

    def test_a_thirty_item_list(self):
        text = "Things I want to try this year:\n" + "\n".join(
            f"{n}. thing {n}" for n in range(1, 31)
        )

        units = split_structured(text)

        assert units == tuple(
            item(f"thing {n}", "Things I want to try this year") for n in range(1, 31)
        )

    def test_a_thematic_break_is_not_an_item(self):
        assert split_structured("- eggs\n---\n- milk") == (item("eggs"), item("milk"))


# ---------------------------------------------------------------------------
# Headings and bold lines
# ---------------------------------------------------------------------------


class TestHeadings:
    def test_headings_with_bullets(self):
        text = (
            "# Work\n"
            "## Tools\n"
            "- Neovim\n"
            "- Terraform\n"
            "## People\n"
            "- Priya is my mentor\n"
            "# Home\n"
            "- I journal every night\n"
        )

        assert split_structured(text) == (
            item("Neovim", "Work", "Tools"),
            item("Terraform", "Work", "Tools"),
            item("Priya is my mentor", "Work", "People"),
            item("I journal every night", "Home"),
        )

    def test_a_heading_is_context_for_prose(self):
        assert split_structured("## Health\nI do a long run every Sunday.") == (
            prose("I do a long run every Sunday.", "Health"),
        )

    @pytest.mark.parametrize("line", ["**Groceries**", "**Groceries:**", "__Groceries__:"])
    def test_a_bold_only_line_is_context(self, line):
        assert split_structured(f"{line}\n- eggs") == (item("eggs", "Groceries"),)

    def test_a_bold_line_sits_under_the_headings(self):
        text = "# Home\n**Groceries**\n- eggs\n**Chores**\n- laundry"

        assert split_structured(text) == (
            item("eggs", "Home", "Groceries"),
            item("laundry", "Home", "Chores"),
        )

    def test_a_heading_nothing_follows_is_kept_as_prose(self):
        text = "# Work\n- I use Rust\n# Home"

        assert split_structured(text) == (item("I use Rust", "Work"), prose("# Home"))

    def test_a_heading_text_drops_a_trailing_colon(self):
        assert split_structured("## Tools:\n- Neovim") == (item("Neovim", "Tools"),)

    def test_a_lead_in_a_heading_cuts_off_is_kept_as_prose(self):
        text = "Plans:\n# Work\n- I use Rust"

        assert split_structured(text) == (prose("Plans:"), item("I use Rust", "Work"))


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


class TestTables:
    def test_each_row_becomes_header_value_pairs(self):
        text = "| Tool | Use |\n|------|:---:|\n| Rust | backend |\n| Python | scripts |"

        assert split_structured(text) == (
            row("Tool: Rust; Use: backend"),
            row("Tool: Python; Use: scripts"),
        )

    def test_a_table_under_a_lead_in_and_a_heading(self):
        text = (
            "# Work\n"
            "My stack:\n"
            "Tool | Since\n"
            "--- | ---\n"
            "Rust | May 2026\n"
            "Go |\n"
            "\n"
            "That is all."
        )

        assert split_structured(text) == (
            row("Tool: Rust; Since: May 2026", "Work", "My stack"),
            row("Tool: Go", "Work", "My stack"),
            prose("That is all.", "Work"),
        )

    def test_an_escaped_pipe_stays_in_its_cell(self):
        text = "| Command | Meaning |\n|---|---|\n| a \\| b | pipe a into b |"

        assert split_structured(text) == (row("Command: a | b; Meaning: pipe a into b"),)


# ---------------------------------------------------------------------------
# Fenced code
# ---------------------------------------------------------------------------


class TestFencedCode:
    def test_a_fenced_block_is_one_prose_unit_and_never_split(self):
        fence = "```yaml\n- name: build\n# not a heading\n| a | b |\n|---|---|\n```"
        text = f"My config:\n- shell: zsh\n{fence}\n- editor: Neovim"

        assert split_structured(text) == (
            item("shell: zsh", "My config"),
            prose(fence, "My config"),
            item("editor: Neovim", "My config"),
        )

    def test_an_unclosed_fence_runs_to_the_end(self):
        text = "- Rust\n~~~\n- not an item"

        assert split_structured(text) == (item("Rust"), prose("~~~\n- not an item"))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


class TestRendering:
    def test_a_unit_without_context_renders_as_its_text(self):
        assert render_unit(prose("I use Rust")) == "I use Rust"

    def test_context_parts_join_with_an_arrow_then_a_colon(self):
        assert render_unit(item("eggs", "Groceries")) == "Groceries: eggs"
        assert render_unit(item("visit the museum", "Trip plan", "Day 1")) == (
            "Trip plan > Day 1: visit the museum"
        )

    def test_a_batch_renders_one_unit_per_line(self):
        batch = (item("eggs", "Groceries"), item("milk", "Groceries"))

        assert render_batch(batch) == "Groceries: eggs\nGroceries: milk"

    @pytest.mark.parametrize("tag,text", _gold_utterances())
    def test_a_turn_without_structure_renders_as_its_exact_utterance(self, tag, text):
        (batch,) = batch_units(split_structured(text), 4)

        assert render_batch(batch) == text


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


class TestBatching:
    def test_consecutive_units_at_most_max_per_batch(self):
        units = tuple(item(f"thing {n}") for n in range(30))

        batches = batch_units(units, 4)

        assert [len(b) for b in batches] == [4] * 7 + [2]
        assert tuple(u for b in batches for u in b) == units

    def test_no_units_no_batches(self):
        assert batch_units((), 4) == ()

    def test_max_units_must_be_positive(self):
        with pytest.raises(ValueError, match="max_units"):
            batch_units((prose("x"),), 0)


# ---------------------------------------------------------------------------
# Merging the batches' payloads
# ---------------------------------------------------------------------------


def _entity(entity_id: str, entity_type: str = "Technology", name: str | None = None) -> dict:
    return {"id": entity_id, "type": entity_type, "name": name or entity_id.title()}


def _rel(source: str, rel_type: str, target: str, confidence: float = 0.9) -> dict:
    return {
        "source": source,
        "type": rel_type,
        "target": target,
        "properties": {"confidence": confidence},
    }


class TestMergePayloads:
    def test_entities_merge_by_id_and_relationships_by_triple_first_wins(self):
        first = ExtractionPayload(
            entities=[_entity("user", "User"), _entity("rust", name="Rust")],
            relationships=[_rel("user", "USES", "rust", 0.9)],
        )
        second = ExtractionPayload(
            entities=[_entity("user", "User"), _entity("rust", name="rust-lang"), _entity("go")],
            relationships=[_rel("user", "USES", "rust", 0.5), _rel("user", "USES", "go")],
        )

        merged, warnings = merge_payloads([first, second])

        assert merged.entities == [
            _entity("user", "User"),
            _entity("rust", name="Rust"),
            _entity("go"),
        ]
        assert merged.relationships == [
            _rel("user", "USES", "rust", 0.9),
            _rel("user", "USES", "go"),
        ]
        assert warnings == []

    def test_a_type_conflict_on_one_id_warns_once(self):
        payloads = [
            ExtractionPayload(entities=[_entity("rust", "Technology")], relationships=[]),
            ExtractionPayload(entities=[_entity("rust", "Concept")], relationships=[]),
            ExtractionPayload(entities=[_entity("rust", "Skill")], relationships=[]),
        ]

        merged, warnings = merge_payloads(payloads)

        assert merged.entities == [_entity("rust", "Technology")]
        assert warnings == [LIST_MERGE_TYPE_CONFLICT]
        assert LIST_MERGE_TYPE_CONFLICT == "list_merge_type_conflict"

    def test_one_payload_comes_back_as_it_is(self):
        """A turn without structure keeps its result exactly, duplicates included."""
        payload = ExtractionPayload(
            entities=[_entity("rust"), _entity("rust")],
            relationships=[_rel("user", "USES", "rust"), _rel("user", "USES", "rust")],
        )

        merged, warnings = merge_payloads([payload])

        assert merged == payload
        assert warnings == []

    def test_an_entity_without_an_id_is_kept_not_merged(self):
        payloads = [
            ExtractionPayload(entities=[{"type": "Technology"}], relationships=[{"x": 1}]),
            ExtractionPayload(entities=[{"type": "Technology"}], relationships=[{"x": 1}]),
        ]

        merged, _ = merge_payloads(payloads)

        assert merged.entities == [{"type": "Technology"}, {"type": "Technology"}]
        assert merged.relationships == [{"x": 1}, {"x": 1}]

    def test_deterministic(self):
        payloads = [
            ExtractionPayload(entities=[_entity("a"), _entity("b")], relationships=[]),
            ExtractionPayload(
                entities=[_entity("b", "Concept"), _entity("c")],
                relationships=[_rel("a", "USES", "c")],
            ),
        ]

        assert merge_payloads(payloads) == merge_payloads(payloads)
