"""Pin how agreement between two label passes is counted.

The selection set becomes the verdict set only when the set kappa between its two blind
passes clears a bar, so the arithmetic behind that kappa is load-bearing: a candidate
universe that left out the both-absent cell would turn kappa into F1 under another name, a
degenerate table would divide by zero on the probes the passes agree on most, and an import
of the ontology pack would let the pack under test define its own agreement. Every row here
is synthetic, with a small pair table and id function of the test's own.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.eval_harness import gold_agreement  # noqa: E402
from scripts.eval_harness.gold_agreement import (  # noqa: E402
    Agreement,
    cohen_kappa,
    probe_agreement,
    set_agreement,
)

_PAIRS = {
    "USES": ({"User", "Person"}, {"Technology"}),
    "INTERESTED_IN": ({"User"}, {"Concept"}),
    "KNOWS_PERSON": ({"User"}, {"Person"}),
}
_WITH_PAIRS = {"WORKS_WITH": ({"Organization", "Technology"}, {"Technology"})}
_ANY_PAIRS = {"RELATED_TO": (None, None)}


def _same(row):
    return row


def _row(tag: str, entities: dict[str, str], edges: list[tuple[str, str, str]], **attrs) -> dict:
    """One label row in the gold schema; `attrs` are written on every edge."""
    return {
        "utterance": "synthetic",
        "tag": tag,
        "expected_entities": [{"id": i, "type": t} for i, t in entities.items()],
        "expected_relationships": [
            {
                "source": s,
                "source_type": entities[s],
                "predicate": p,
                "target": t,
                "target_type": entities[t],
                **attrs,
            }
            for s, p, t in edges
        ],
    }


def _tools(tag: str, used: str, known: str = "abc") -> dict:
    """A row whose entities are the user and one tool per letter of `known`."""
    entities = {"user": "User"} | {tool: "Technology" for tool in known}
    return _row(tag, entities, [("user", "USES", tool) for tool in used])


def _agree(row_a: dict, row_b: dict, pairs=_PAIRS, id_function=_same, undirected=()) -> Agreement:
    return probe_agreement(
        row_a, row_b, pair_table=pairs, id_function=id_function, undirected=undirected
    )


class TestCohenKappa:
    def test_a_mixed_table_follows_the_formula(self):
        kappa = cohen_kappa(both_present=1, only_a=1, only_b=0, both_absent=1)

        assert kappa == pytest.approx(0.4)

    def test_chance_level_agreement_is_zero(self):
        kappa = cohen_kappa(both_present=1, only_a=1, only_b=1, both_absent=1)

        assert kappa == pytest.approx(0.0)

    @pytest.mark.parametrize(
        ("both_present", "both_absent"),
        [
            pytest.param(0, 0, id="empty-universe"),
            pytest.param(0, 7, id="all-absent"),
            pytest.param(7, 0, id="all-present"),
            pytest.param(3, 4, id="mixed"),
        ],
    )
    def test_full_agreement_is_one_without_dividing_by_zero(self, both_present, both_absent):
        kappa = cohen_kappa(both_present=both_present, only_a=0, only_b=0, both_absent=both_absent)

        assert kappa == 1.0

    def test_one_pass_marking_nothing_is_zero_not_an_error(self):
        kappa = cohen_kappa(both_present=0, only_a=0, only_b=2, both_absent=5)

        assert kappa == pytest.approx(0.0)


class TestProbeAgreement:
    def test_candidates_neither_pass_wrote_fill_the_both_absent_cell(self):
        figures = _agree(_tools("p", used="ab"), _tools("p", used="a"))

        assert (figures.both_present, figures.only_a, figures.only_b) == (1, 1, 0)
        assert figures.both_absent == 1
        assert figures.universe == 3
        assert figures.f1 == pytest.approx(2 / 3)
        assert figures.kappa == pytest.approx(0.4)

    def test_the_universe_is_what_the_pair_table_admits_over_both_passes_entities(self):
        row_a = _row("p", {"user": "User", "zed": "Technology"}, [("user", "USES", "zed")])
        row_b = _row("p", {"user": "User", "olga": "Person"}, [("user", "KNOWS_PERSON", "olga")])

        figures = _agree(row_a, row_b)

        # user USES zed, olga USES zed and user KNOWS_PERSON olga; nothing else is admitted.
        assert figures.universe == 3
        assert (figures.both_present, figures.only_a, figures.only_b) == (0, 1, 1)
        assert figures.both_absent == 1

    def test_a_triple_with_one_endpoint_twice_is_no_candidate(self):
        row = _row("p", {"zed": "Technology"}, [])

        figures = _agree(row, row, pairs=_ANY_PAIRS)

        assert figures.universe == 0

    def test_identical_passes_agree_fully(self):
        row = _tools("p", used="ab")

        figures = _agree(row, row)

        assert (figures.only_a, figures.only_b) == (0, 0)
        assert figures.f1 == 1.0
        assert figures.kappa == 1.0

    @pytest.mark.parametrize(
        "row",
        [
            pytest.param(_row("p", {}, []), id="empty-universe"),
            pytest.param(_tools("p", used=""), id="all-absent"),
            pytest.param(_tools("p", used="abc"), id="all-present"),
        ],
    )
    def test_identical_passes_on_a_degenerate_table_give_one(self, row):
        figures = _agree(row, row)

        assert figures.f1 == 1.0
        assert figures.kappa == 1.0

    def test_ids_are_compared_after_the_id_function(self):
        def lowered(row: dict) -> dict:
            entities = [{**e, "id": e["id"].lower()} for e in row["expected_entities"]]
            edges = [
                {**r, "source": r["source"].lower(), "target": r["target"].lower()}
                for r in row["expected_relationships"]
            ]
            return {**row, "expected_entities": entities, "expected_relationships": edges}

        row_a = _row("p", {"user": "User", "Zed": "Technology"}, [("user", "USES", "Zed")])
        row_b = _row("p", {"user": "User", "zed": "Technology"}, [("user", "USES", "zed")])

        raw = _agree(row_a, row_b)
        railed = _agree(row_a, row_b, id_function=lowered)

        assert (raw.both_present, raw.only_a, raw.only_b) == (0, 1, 1)
        assert (railed.both_present, railed.only_a, railed.only_b) == (1, 0, 0)
        assert railed.universe == 1

    def test_an_undirected_predicate_is_one_candidate_in_either_direction(self):
        entities = {"hugo": "Technology", "zola": "Technology"}
        row_a = _row("p", entities, [("hugo", "WORKS_WITH", "zola")])
        row_b = _row("p", entities, [("zola", "WORKS_WITH", "hugo")])

        directed = _agree(row_a, row_b, pairs=_WITH_PAIRS)
        undirected = _agree(row_a, row_b, pairs=_WITH_PAIRS, undirected={"WORKS_WITH"})

        assert (directed.universe, directed.both_present) == (2, 0)
        assert (undirected.universe, undirected.both_present) == (1, 1)
        assert undirected.kappa == 1.0

    def test_an_undirected_predicate_is_admitted_when_either_direction_fits(self):
        entities = {"corp": "Organization", "zola": "Technology"}
        row = _row("p", entities, [("zola", "WORKS_WITH", "corp")])

        directed = _agree(row, row, pairs=_WITH_PAIRS)
        undirected = _agree(row, row, pairs=_WITH_PAIRS, undirected={"WORKS_WITH"})

        assert (directed.universe, directed.outside_a, directed.outside_b) == (1, 1, 1)
        assert (undirected.universe, undirected.both_present) == (1, 1)
        assert (undirected.outside_a, undirected.outside_b) == (0, 0)

    def test_an_id_listed_only_as_an_alternative_is_another_id(self):
        row_a = _row("p", {"user": "User", "zed": "Technology"}, [("user", "USES", "zed")])
        row_a["expected_entities"][1]["alternatives"] = ["zola"]
        row_b = _row("p", {"user": "User", "zola": "Technology"}, [("user", "USES", "zola")])

        figures = _agree(row_a, row_b)

        assert figures.universe == 2
        assert (figures.both_present, figures.only_a, figures.only_b) == (0, 1, 1)

    def test_an_entity_typed_differently_admits_candidates_under_either_type(self):
        row_a = _row("p", {"user": "User", "foo": "Technology"}, [("user", "USES", "foo")])
        row_b = _row("p", {"user": "User", "foo": "Concept"}, [("user", "INTERESTED_IN", "foo")])

        figures = _agree(row_a, row_b)

        assert figures.universe == 2
        assert (figures.both_present, figures.only_a, figures.only_b) == (0, 1, 1)
        assert figures.entities_typed_differently == 1

    def test_a_type_disagreement_alone_is_not_a_relation_disagreement(self):
        row_a = _row("p", {"user": "User", "foo": "Technology"}, [("user", "USES", "foo")])
        row_b = {
            **_row("p", {"user": "User", "foo": "Technology"}, [("user", "USES", "foo")]),
            "expected_entities": [{"id": "user", "type": "User"}, {"id": "foo", "type": "Concept"}],
        }
        row_b["expected_relationships"][0]["target_type"] = "Concept"

        figures = _agree(row_a, row_b)

        assert (figures.both_present, figures.only_a, figures.only_b) == (1, 0, 0)
        assert figures.entities_typed_differently == 1
        assert figures.kappa == 1.0

    def test_an_edge_the_pair_table_refuses_is_counted_outside_the_universe(self):
        entities = {"user": "User", "olga": "Person", "zed": "Technology"}
        row_a = _row("p", entities, [("user", "USES", "zed"), ("zed", "USES", "olga")])
        row_b = _row("p", entities, [("user", "USES", "zed"), ("user", "OWNS", "zed")])

        figures = _agree(row_a, row_b)

        assert figures.universe == 3
        assert (figures.outside_a, figures.outside_b) == (1, 1)
        assert (figures.both_present, figures.only_a, figures.only_b) == (1, 0, 0)

    def test_a_relation_both_wrote_with_other_attributes_is_counted_apart(self):
        entities = {"user": "User", "zed": "Technology"}
        row_a = _row("p", entities, [("user", "USES", "zed")], assertion_kind="cease")
        row_b = _row("p", entities, [("user", "USES", "zed")])
        row_c = _row("p", entities, [("user", "USES", "zed")], valid_from="2026-05")

        ceased = _agree(row_a, row_b)
        dated = _agree(row_b, row_c)
        same = _agree(
            row_b, _row("p", entities, [("user", "USES", "zed")], assertion_kind="assert")
        )

        assert (ceased.both_present, ceased.attribute_differences) == (1, 1)
        assert (dated.both_present, dated.attribute_differences) == (1, 1)
        assert (same.both_present, same.attribute_differences) == (1, 0)

    def test_rows_of_different_probes_are_refused(self):
        with pytest.raises(ValueError, match="different probes"):
            _agree(_tools("p", used="a"), _tools("q", used="a"))


def _report(rows_a: list[dict], rows_b: list[dict], group_of=lambda tag: tag.split("-")[0]):
    return set_agreement(rows_a, rows_b, pair_table=_PAIRS, id_function=_same, group_of=group_of)


class TestSetAgreement:
    def test_a_set_pools_its_probes_cells_before_kappa(self):
        rows_a = [_tools("list-1", used="ab"), _tools("list-2", used="ab", known="abcd")]
        rows_b = [_tools("list-1", used="a"), _tools("list-2", used="ac", known="abcd")]

        pooled = _report(rows_a, rows_b).sets["list"]

        assert pooled.probes == 2
        assert (pooled.figures.both_present, pooled.figures.only_a) == (2, 2)
        assert (pooled.figures.only_b, pooled.figures.both_absent) == (1, 2)
        assert pooled.figures.f1 == pytest.approx(4 / 7)
        assert pooled.figures.kappa == pytest.approx(0.16)

    def test_each_group_is_pooled_apart_and_probes_keep_their_own_figures(self):
        rows_a = [_tools("list-1", used="ab"), _tools("prose-1", used="a")]
        rows_b = [_tools("prose-1", used="a"), _tools("list-1", used="a")]

        report = _report(rows_a, rows_b)

        assert sorted(report.sets) == ["list", "prose"]
        assert report.sets["list"].figures.kappa == pytest.approx(0.4)
        assert report.sets["prose"].figures.kappa == 1.0
        assert [p.tag for p in report.probes] == ["list-1", "prose-1"]
        assert [p.figures.only_a for p in report.probes] == [1, 0]

    def test_identical_passes_give_a_set_kappa_of_one(self):
        rows = [_tools("list-1", used="ab"), _tools("list-2", used=""), _row("list-3", {}, [])]

        pooled = _report(rows, rows).sets["list"]

        assert pooled.figures.kappa == 1.0
        assert pooled.figures.f1 == 1.0

    def test_passes_over_different_tags_are_refused_with_counts_only(self):
        rows_a = [_tools("list-1", used="a"), _tools("list-2", used="a")]
        rows_b = [_tools("list-1", used="a"), _tools("list-3", used="a")]

        with pytest.raises(ValueError) as error:
            _report(rows_a, rows_b)

        assert str(error.value) == "the passes differ in tags: 1 only in pass A, 1 only in pass B"

    def test_a_repeated_tag_is_refused(self):
        rows = [_tools("list-1", used="a"), _tools("list-1", used="b")]

        with pytest.raises(ValueError, match="1 tags repeated in pass A"):
            _report(rows, [_tools("list-1", used="a")])

    def test_the_report_serializes_to_counts_and_figures_only(self):
        rows_a = [_tools("list-1", used="ab"), _tools("prose-1", used="a")]
        rows_b = [_tools("list-1", used="a"), _tools("prose-1", used="a")]

        data = json.loads(json.dumps(_report(rows_a, rows_b).to_dict()))

        assert sorted(data) == ["probes", "sets"]
        assert data["sets"]["list"]["probes"] == 1
        assert data["sets"]["list"]["kappa"] == pytest.approx(0.4)
        assert [p["tag"] for p in data["probes"]] == ["list-1", "prose-1"]
        texts = [
            value
            for entry in (*data["probes"], *data["sets"].values())
            for key, value in entry.items()
            if key != "tag" and not isinstance(value, int | float)
        ]
        assert texts == []


def test_the_module_imports_only_the_standard_library():
    """The caller passes the pair table and the id function, so no pack defines agreement."""
    tree = ast.parse(Path(gold_agreement.__file__).read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            modules.add("." * node.level + (node.module or "").split(".")[0])

    assert modules <= {"__future__", "collections", "dataclasses", "typing"}
