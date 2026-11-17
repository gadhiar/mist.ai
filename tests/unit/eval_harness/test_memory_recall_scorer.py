"""Hand-built runs pin each metric of the memory recall scorer.

The fetcher's floors (T9, T19, T21) and the real-model eval (H4) all read their numbers from
`scripts/eval_harness/memory_recall.py`, so a scorer that miscounts would move every floor at
once without any of those tests noticing. Each test here builds a small probe set and a run by
hand, works the expected value out in its docstring, and asserts the scorer reports exactly that.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.eval_harness.memory_recall import (  # noqa: E402
    BlockItem,
    ItemRef,
    Probe,
    RunRecord,
    load_run,
    main,
    parse_probe,
    parse_run_record,
    render_markdown,
    score_run,
)

NOW = datetime(2026, 3, 16, 9, 0, tzinfo=UTC)


def _ref(kind: str, item_id: str) -> ItemRef:
    return ItemRef(kind=kind, id=item_id)


def _probe(
    probe_id: str,
    *,
    kind: str = "fact",
    lexical: bool = True,
    expected: tuple[ItemRef, ...] = (),
    acceptable: tuple[ItemRef, ...] = (),
    forbidden: tuple[ItemRef, ...] = (),
    correction_id: str | None = None,
    shape: str = "plain",
) -> Probe:
    return Probe(
        probe_id=probe_id,
        kind=kind,
        lexical=lexical,
        query=f"query for {probe_id}",
        now=NOW,
        conversation_start=False,
        expected=expected,
        acceptable=acceptable,
        forbidden=forbidden,
        distractors=(),
        correction_id=correction_id,
        rationale="hand-built",
        shape=shape,
    )


def _record(
    probe_id: str,
    items: tuple[ItemRef, ...] = (),
    *,
    tokens: int = 100,
    budget_tokens: int = 500,
    ranks: tuple[int, ...] | None = None,
    refs: tuple[str, ...] | None = None,
) -> RunRecord:
    """A run record whose items are fetched in block order unless `ranks` says otherwise."""
    block = tuple(
        BlockItem(
            ref=refs[index - 1] if refs else f"m{index}",
            kind=item.kind,
            id=item.id,
            score=0.9 - index * 0.01,
            rank=ranks[index - 1] if ranks else index,
        )
        for index, item in enumerate(items, start=1)
    )
    return RunRecord(probe_id=probe_id, items=block, tokens=tokens, budget_tokens=budget_tokens)


FACT_A = _ref("fact", "fa")
FACT_B = _ref("fact", "fb")
FACT_X = _ref("fact", "fx")
EPISODE_A = _ref("episode", "e-a")
EXCHANGE_A = _ref("exchange", "evt-a")
NOTE_A = _ref("note", "notes/a.md")


class TestRecallAndMrr:
    def test_perfect_run_scores_one_on_every_ranking_metric(self) -> None:
        probes = [_probe("p1", expected=(FACT_A,)), _probe("p2", expected=(FACT_B, NOTE_A))]
        run = [_record("p1", (FACT_A,)), _record("p2", (NOTE_A, FACT_B))]

        overall = score_run(probes, run).overall

        assert overall.recall_at_block == 1.0
        assert overall.mrr == 1.0
        assert overall.precision_at_block == 1.0
        assert overall.budget_ok == 1.0

    def test_recall_is_the_share_of_expected_items_in_the_block(self) -> None:
        """Expected {fa, fb}, block [fx, fa]: one of two found, first hit at rank 2."""
        probes = [_probe("p1", expected=(FACT_A, FACT_B))]
        run = [_record("p1", (FACT_X, FACT_A))]

        overall = score_run(probes, run).overall

        assert overall.recall_at_block == 0.5
        assert overall.mrr == 0.5
        assert overall.precision_at_block == 0.5

    def test_recall_and_mrr_average_per_probe(self) -> None:
        """Ranks 1, 3 and none give MRR (1 + 1/3 + 0) / 3; recall (1 + 1 + 0) / 3."""
        probes = [
            _probe("p1", expected=(FACT_A,)),
            _probe("p2", expected=(FACT_A,)),
            _probe("p3", expected=(FACT_A,)),
        ]
        run = [
            _record("p1", (FACT_A,)),
            _record("p2", (FACT_X, FACT_B, FACT_A)),
            _record("p3", (FACT_X,)),
        ]

        overall = score_run(probes, run).overall

        assert overall.mrr == pytest.approx((1 + 1 / 3 + 0) / 3)
        assert overall.recall_at_block == pytest.approx(2 / 3)
        assert overall.recall_probes == 3

    def test_probes_with_no_expected_items_are_left_out_of_recall_and_mrr(self) -> None:
        """Recall is undefined without expected items, so p2 must not dilute p1's 1.0."""
        probes = [
            _probe("p1", expected=(FACT_A,)),
            _probe("p2", kind="correction", forbidden=(FACT_B,), correction_id="corr-1"),
        ]
        run = [_record("p1", (FACT_A,)), _record("p2", ())]

        overall = score_run(probes, run).overall

        assert overall.recall_at_block == 1.0
        assert overall.mrr == 1.0
        assert overall.recall_probes == 1
        assert overall.probes == 2

    def test_an_item_matches_only_with_the_same_kind(self) -> None:
        """An exchange whose id equals an expected fact's id is not that fact."""
        probes = [_probe("p1", expected=(FACT_A,))]
        run = [_record("p1", (_ref("exchange", "fa"),))]

        overall = score_run(probes, run).overall

        assert overall.recall_at_block == 0.0
        assert overall.mrr == 0.0
        assert overall.precision_at_block == 0.0


class TestMrrByFetchRank:
    """MRR reads each item's fetch rank, not its block position (ruling b, 7 Nov 2026).

    The block lays items out by section, facts before notes, so block position would charge a
    note the fetcher ranked first for every fact placed ahead of it.
    """

    def test_an_expected_note_ranked_first_behind_two_facts_scores_one(self) -> None:
        """Block [fa, fb, note] with fetch ranks [2, 3, 1]: the note was fetched first."""
        probes = [_probe("p1", kind="note", expected=(NOTE_A,))]
        run = [_record("p1", (FACT_A, FACT_B, NOTE_A), ranks=(2, 3, 1))]

        overall = score_run(probes, run).overall

        assert overall.mrr == 1.0

    def test_an_expected_item_ranked_third_in_first_place_scores_a_third(self) -> None:
        """Block [fa, fx] with ranks [3, 4]: the fetcher ranked two dropped candidates higher."""
        probes = [_probe("p1", expected=(FACT_A,))]
        run = [_record("p1", (FACT_A, FACT_X), ranks=(3, 4))]

        overall = score_run(probes, run).overall

        assert overall.mrr == pytest.approx(1 / 3)

    def test_the_lowest_rank_among_expected_items_is_scored(self) -> None:
        """Expected {fa, fb} at ranks 5 and 2: the reciprocal of 2, whatever the block order."""
        probes = [_probe("p1", expected=(FACT_A, FACT_B))]
        run = [_record("p1", (FACT_A, FACT_B), ranks=(5, 2))]

        overall = score_run(probes, run).overall

        assert overall.mrr == 0.5

    def test_items_sharing_a_ref_and_a_rank_score_that_rank(self) -> None:
        """A list's members are one block entry `m1` at rank 2; the expected member scores 1/2."""
        probes = [_probe("p1", expected=(FACT_B,))]
        run = [
            _record(
                "p1",
                (FACT_X, FACT_A, FACT_B),
                ranks=(1, 2, 2),
                refs=("m0", "m1", "m1"),
            )
        ]

        overall = score_run(probes, run).overall

        assert overall.mrr == 0.5


class TestPrecision:
    def test_acceptable_items_count_as_relevant(self) -> None:
        """Block [e-a, evt-a, fx]: the episode is expected and its exchange acceptable: 2/3."""
        probes = [_probe("p1", kind="episode", expected=(EPISODE_A,), acceptable=(EXCHANGE_A,))]
        run = [_record("p1", (EPISODE_A, EXCHANGE_A, FACT_X))]

        overall = score_run(probes, run).overall

        assert overall.precision_at_block == pytest.approx(2 / 3)
        assert overall.recall_at_block == 1.0

    def test_acceptable_items_do_not_count_toward_recall(self) -> None:
        probes = [_probe("p1", kind="episode", expected=(EPISODE_A,), acceptable=(EXCHANGE_A,))]
        run = [_record("p1", (EXCHANGE_A,))]

        overall = score_run(probes, run).overall

        assert overall.recall_at_block == 0.0
        assert overall.mrr == 0.0
        assert overall.precision_at_block == 1.0

    def test_empty_blocks_are_left_out_of_precision(self) -> None:
        """p2's empty block has no precision; p1's half-relevant block gives 0.5 alone."""
        probes = [_probe("p1", expected=(FACT_A,)), _probe("p2", expected=(FACT_B,))]
        run = [_record("p1", (FACT_A, FACT_X)), _record("p2", ())]

        overall = score_run(probes, run).overall

        assert overall.precision_at_block == 0.5
        assert overall.precision_probes == 1
        assert overall.recall_at_block == 0.5

    def test_items_served_to_a_probe_with_nothing_expected_are_all_irrelevant(self) -> None:
        """A noise probe given [fa, fb] scores precision 0 for that probe."""
        probes = [_probe("p1", expected=(FACT_A,)), _probe("p2", kind="noise")]
        run = [_record("p1", (FACT_A,)), _record("p2", (FACT_A, FACT_B))]

        overall = score_run(probes, run).overall

        assert overall.precision_at_block == 0.5
        assert overall.precision_probes == 2


class TestEmptyOnNoise:
    def test_all_noise_run_with_empty_blocks(self) -> None:
        """Every probe is noise and every block is empty: only noise and budget are defined."""
        probes = [_probe(f"p{n}", kind="noise", lexical=n % 2 == 0) for n in range(4)]
        run = [_record(f"p{n}", (), tokens=0) for n in range(4)]

        overall = score_run(probes, run).overall

        assert overall.empty_on_noise == 1.0
        assert overall.noise_probes == 4
        assert overall.recall_at_block is None
        assert overall.mrr is None
        assert overall.precision_at_block is None
        assert overall.forgotten_leak is None
        assert overall.budget_ok == 1.0

    def test_all_noise_run_counts_each_non_empty_block_as_a_miss(self) -> None:
        """One of four noise blocks holds an item: empty_on_noise 3/4, precision 0 on it."""
        probes = [_probe(f"p{n}", kind="noise") for n in range(4)]
        run = [_record("p0", (FACT_X,))] + [_record(f"p{n}", ()) for n in range(1, 4)]

        overall = score_run(probes, run).overall

        assert overall.empty_on_noise == 0.75
        assert overall.precision_at_block == 0.0
        assert overall.precision_probes == 1

    def test_empty_on_noise_ignores_other_kinds_with_empty_blocks(self) -> None:
        """An empty block on a fact probe is a recall miss, not an empty-on-noise success."""
        probes = [_probe("p1", expected=(FACT_A,)), _probe("p2", kind="noise")]
        run = [_record("p1", ()), _record("p2", (FACT_X,))]

        overall = score_run(probes, run).overall

        assert overall.empty_on_noise == 0.0
        assert overall.noise_probes == 1


class TestForgottenLeak:
    def test_a_block_holding_a_forbidden_item_leaks(self) -> None:
        """Two correction probes, one of which is served the forgotten fact: leak 1/2."""
        probes = [
            _probe("p1", kind="correction", forbidden=(FACT_A,), correction_id="corr-1"),
            _probe(
                "p2",
                kind="correction",
                expected=(FACT_B,),
                forbidden=(FACT_X, EPISODE_A),
                correction_id="corr-2",
            ),
        ]
        run = [_record("p1", ()), _record("p2", (FACT_B, EPISODE_A))]

        overall = score_run(probes, run).overall

        assert overall.forgotten_leak == 0.5
        assert overall.forgotten_probes == 2
        assert overall.recall_at_block == 1.0

    def test_leak_is_undefined_without_forbidden_items(self) -> None:
        probes = [_probe("p1", expected=(FACT_A,))]
        run = [_record("p1", (FACT_A,))]

        assert score_run(probes, run).overall.forgotten_leak is None


class TestBudgetOk:
    def test_budget_ok_is_the_share_of_blocks_within_their_budget(self) -> None:
        """Tokens 500 of 500 fits; 501 of 500 does not; 0 of 300 fits: 2/3."""
        probes = [_probe("p1"), _probe("p2"), _probe("p3", kind="noise")]
        run = [
            _record("p1", (), tokens=500, budget_tokens=500),
            _record("p2", (), tokens=501, budget_tokens=500),
            _record("p3", (), tokens=0, budget_tokens=300),
        ]

        assert score_run(probes, run).overall.budget_ok == pytest.approx(2 / 3)


class TestBreakdowns:
    def _report(self):
        probes = [
            _probe("f1", kind="fact", lexical=True, expected=(FACT_A,)),
            _probe("f2", kind="fact", lexical=False, expected=(FACT_B,)),
            _probe("n1", kind="noise", lexical=True),
            _probe("n2", kind="noise", lexical=False),
        ]
        run = [
            _record("f1", (FACT_A,)),
            _record("f2", (FACT_X,)),
            _record("n1", (FACT_X,), tokens=600),
            _record("n2", ()),
        ]
        return score_run(probes, run)

    def test_per_kind_metrics_are_computed_on_that_kind_alone(self) -> None:
        report = self._report()

        assert report.by_kind["fact"].recall_at_block == 0.5
        assert report.by_kind["fact"].empty_on_noise is None
        assert report.by_kind["noise"].empty_on_noise == 0.5
        assert report.by_kind["noise"].recall_at_block is None
        assert report.by_kind["noise"].budget_ok == 0.5

    def test_every_probe_kind_is_reported_even_without_probes(self) -> None:
        report = self._report()

        assert set(report.by_kind) == {
            "fact",
            "episode",
            "note",
            "continuity",
            "temporal",
            "correction",
            "noise",
        }
        assert report.by_kind["episode"].probes == 0
        assert report.by_kind["episode"].budget_ok is None

    def test_per_lexical_tag_metrics_split_lexical_from_paraphrase(self) -> None:
        report = self._report()

        assert set(report.by_lexical) == {"lexical", "paraphrase"}
        lexical = report.by_lexical["lexical"]
        paraphrase = report.by_lexical["paraphrase"]
        assert (lexical.recall_at_block, lexical.empty_on_noise) == (1.0, 0.0)
        assert (paraphrase.recall_at_block, paraphrase.empty_on_noise) == (0.0, 1.0)
        assert lexical.budget_ok == 0.5
        assert paraphrase.budget_ok == 1.0

    def test_overall_combines_every_probe(self) -> None:
        overall = self._report().overall

        assert overall.probes == 4
        assert overall.recall_at_block == 0.5
        assert overall.empty_on_noise == 0.5
        assert overall.precision_at_block == pytest.approx((1.0 + 0.0 + 0.0) / 3)


class TestShapeBreakdown:
    """The read path's list cap and type filter are judged on their own probes (plan T36)."""

    def _report(self):
        """Two list probes (one complete, one missing a sibling), a type probe and a plain one."""
        probes = [
            _probe("l1", shape="list_item", expected=(FACT_A, FACT_B)),
            _probe("l2", shape="list_item", expected=(FACT_A, FACT_B)),
            _probe("t1", shape="type_scoped", expected=(FACT_X,)),
            _probe("p1", kind="noise"),
        ]
        run = [
            _record("l1", (FACT_A, FACT_B)),
            _record("l2", (FACT_A,), tokens=301, budget_tokens=300),
            _record("t1", (FACT_B, FACT_X)),
            _record("p1", ()),
        ]
        return score_run(probes, run)

    def test_a_probe_built_without_a_shape_is_plain(self) -> None:
        probe = Probe(
            probe_id="p1",
            kind="fact",
            lexical=True,
            query="q",
            now=NOW,
            conversation_start=False,
            expected=(FACT_A,),
            acceptable=(),
            forbidden=(),
            distractors=(),
            correction_id=None,
            rationale="r",
        )

        assert probe.shape == "plain"

    def test_per_shape_metrics_are_computed_on_that_shape_alone(self) -> None:
        """list_item: recall (1 + 1/2) / 2 and budget 1/2; type_scoped: recall 1, MRR 1/2."""
        report = self._report()

        list_item = report.by_shape["list_item"]
        type_scoped = report.by_shape["type_scoped"]
        plain = report.by_shape["plain"]
        assert (list_item.probes, list_item.recall_at_block) == (2, 0.75)
        assert list_item.budget_ok == 0.5
        assert list_item.empty_on_noise is None
        assert (type_scoped.recall_at_block, type_scoped.mrr) == (1.0, 0.5)
        assert type_scoped.precision_at_block == 0.5
        assert (plain.probes, plain.empty_on_noise, plain.recall_at_block) == (1, 1.0, None)

    def test_every_shape_is_reported_even_without_probes(self) -> None:
        report = score_run([_probe("p1", expected=(FACT_A,))], [_record("p1", (FACT_A,))])

        assert set(report.by_shape) == {"plain", "list_item", "type_scoped"}
        assert report.by_shape["list_item"].probes == 0
        assert report.by_shape["list_item"].recall_at_block is None

    def test_shape_does_not_move_the_overall_kind_or_tag_metrics(self) -> None:
        """Relabelling every probe plain leaves every other breakdown as it was."""
        report = self._report()
        probes = [
            _probe("l1", expected=(FACT_A, FACT_B)),
            _probe("l2", expected=(FACT_A, FACT_B)),
            _probe("t1", expected=(FACT_X,)),
            _probe("p1", kind="noise"),
        ]
        run = [
            _record("l1", (FACT_A, FACT_B)),
            _record("l2", (FACT_A,), tokens=301, budget_tokens=300),
            _record("t1", (FACT_B, FACT_X)),
            _record("p1", ()),
        ]
        plain = score_run(probes, run)

        assert report.overall == plain.overall
        assert report.by_kind == plain.by_kind
        assert report.by_lexical == plain.by_lexical
        assert plain.by_shape["plain"] == plain.overall

    def test_markdown_and_json_carry_a_row_per_shape(self) -> None:
        report = self._report()

        text = render_markdown(report)
        data = report.to_json()

        for shape in ("plain", "list_item", "type_scoped"):
            assert f"| shape: {shape} |" in text
        assert data["by_shape"]["list_item"]["recall_at_block"] == 0.75


class TestRunValidation:
    def test_a_probe_without_a_run_record_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="p2"):
            score_run([_probe("p1"), _probe("p2")], [_record("p1")])

    def test_a_run_record_for_an_unknown_probe_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="p9"):
            score_run([_probe("p1")], [_record("p1"), _record("p9")])

    def test_two_records_for_one_probe_are_an_error(self) -> None:
        with pytest.raises(ValueError, match="p1"):
            score_run([_probe("p1")], [_record("p1"), _record("p1")])

    def test_the_same_item_twice_in_one_block_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="fa"):
            score_run([_probe("p1", expected=(FACT_A,))], [_record("p1", (FACT_A, FACT_A))])

    def test_a_record_with_an_unknown_item_kind_is_rejected(self) -> None:
        row = {
            "probe_id": "p1",
            "items": [{"ref": "m1", "kind": "memo", "id": "x", "score": 0.5, "rank": 1}],
            "tokens": 10,
            "budget_tokens": 500,
        }

        with pytest.raises(ValueError, match="memo"):
            parse_run_record(row)

    def test_a_record_may_carry_extra_keys_such_as_timings(self) -> None:
        row = {
            "probe_id": "p1",
            "items": [{"ref": "m1", "kind": "fact", "id": "fa", "score": 0.5, "rank": 1}],
            "tokens": 10,
            "budget_tokens": 500,
            "timings_ms": {"total": 12.5},
            "reranked": 4,
        }

        record = parse_run_record(row)

        assert record.items[0].item == FACT_A

    def test_an_item_carries_its_fetch_rank(self) -> None:
        row = {
            "probe_id": "p1",
            "items": [{"ref": "m1", "kind": "fact", "id": "fa", "score": 0.5, "rank": 7}],
            "tokens": 10,
            "budget_tokens": 500,
        }

        record = parse_run_record(row)

        assert record.items[0].rank == 7

    @pytest.mark.parametrize(
        "rank",
        [
            pytest.param(None, id="missing"),
            pytest.param(0, id="zero"),
            pytest.param(True, id="bool"),
        ],
    )
    def test_an_item_without_a_valid_rank_is_rejected(self, rank: object) -> None:
        """MRR reads the rank, so a run written without one must fail, not score position."""
        good = {"ref": "m1", "kind": "fact", "id": "fa", "score": 0.5, "rank": 1}
        bad = {"ref": "m2", "kind": "fact", "id": "fb", "score": 0.4}
        if rank is not None:
            bad["rank"] = rank
        row = {"probe_id": "p1", "items": [good, bad], "tokens": 10, "budget_tokens": 500}

        with pytest.raises(ValueError, match=r"run record p1 items\[1\].*rank"):
            parse_run_record(row)


class TestProbeParsing:
    def _row(self, **overrides: object) -> dict[str, object]:
        row: dict[str, object] = {
            "probe_id": "p-001",
            "kind": "fact",
            "lexical": True,
            "query": "where does Wren work",
            "now": "2026-03-16T09:00:00+00:00",
            "conversation_start": False,
            "expected": [{"kind": "fact", "id": "fa"}],
            "acceptable": [],
            "forbidden": [],
            "distractors": [{"kind": "fact", "id": "fb"}],
            "correction_id": None,
            "rationale": "why",
            "shape": "plain",
        }
        row.update(overrides)
        return row

    def test_a_valid_row_parses_into_a_probe(self) -> None:
        probe = parse_probe(self._row())

        assert probe.expected == (FACT_A,)
        assert probe.distractors == (FACT_B,)
        assert probe.now == NOW
        assert probe.shape == "plain"

    @pytest.mark.parametrize("shape", ["plain", "list_item", "type_scoped"])
    def test_each_shape_parses(self, shape: str) -> None:
        assert parse_probe(self._row(shape=shape)).shape == shape

    def test_a_row_without_a_shape_is_rejected(self) -> None:
        """The fixture is strict: a dropped `shape` is an error, never a silent plain."""
        row = self._row()
        del row["shape"]

        with pytest.raises(ValueError, match="shape"):
            parse_probe(row)

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            pytest.param({"kind": "trivia"}, "trivia", id="unknown-probe-kind"),
            pytest.param({"lexical": "yes"}, "lexical", id="lexical-not-bool"),
            pytest.param({"now": "2026-03-16T09:00:00"}, "now", id="naive-timestamp"),
            pytest.param({"expected": [{"kind": "memo", "id": "x"}]}, "memo", id="bad-ref-kind"),
            pytest.param({"surprise": 1}, "surprise", id="unknown-key"),
            pytest.param({"shape": "listy"}, "listy", id="unknown-shape"),
        ],
    )
    def test_an_invalid_row_is_rejected(self, overrides: dict[str, object], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            parse_probe(self._row(**overrides))

    def test_a_missing_key_is_rejected(self) -> None:
        row = self._row()
        del row["rationale"]

        with pytest.raises(ValueError, match="rationale"):
            parse_probe(row)


class TestCommandLine:
    def test_main_scores_a_run_file_and_writes_json(self, tmp_path: Path) -> None:
        probes_path = tmp_path / "probes.jsonl"
        run_path = tmp_path / "run.jsonl"
        json_path = tmp_path / "report.json"
        probe_rows = [
            TestProbeParsing()._row(probe_id="p1"),
            TestProbeParsing()._row(
                probe_id="p2", kind="noise", expected=[], distractors=[], lexical=False
            ),
        ]
        run_rows = [
            {
                "probe_id": "p1",
                "items": [{"ref": "m1", "kind": "fact", "id": "fa", "score": 0.8, "rank": 1}],
                "tokens": 40,
                "budget_tokens": 500,
            },
            {"probe_id": "p2", "items": [], "tokens": 0, "budget_tokens": 500},
        ]
        probes_path.write_text("".join(json.dumps(r) + "\n" for r in probe_rows), "utf-8")
        run_path.write_text("".join(json.dumps(r) + "\n" for r in run_rows), "utf-8")

        exit_code = main(
            [
                "--probes",
                str(probes_path),
                "--run",
                str(run_path),
                "--json-output",
                str(json_path),
            ]
        )

        report = json.loads(json_path.read_text("utf-8"))
        assert exit_code == 0
        assert report["overall"]["recall_at_block"] == 1.0
        assert report["overall"]["empty_on_noise"] == 1.0
        assert report["by_kind"]["noise"]["probes"] == 1
        assert report["by_lexical"]["paraphrase"]["empty_on_noise"] == 1.0
        assert report["by_shape"]["plain"]["probes"] == 2
        assert report["by_shape"]["list_item"]["probes"] == 0

    def test_load_run_reads_one_record_per_line(self, tmp_path: Path) -> None:
        run_path = tmp_path / "run.jsonl"
        run_path.write_text(
            json.dumps({"probe_id": "p1", "items": [], "tokens": 0, "budget_tokens": 500}) + "\n\n",
            "utf-8",
        )

        records = load_run(run_path)

        assert [r.probe_id for r in records] == ["p1"]

    def test_markdown_report_names_every_metric(self) -> None:
        report = score_run([_probe("p1", expected=(FACT_A,))], [_record("p1", (FACT_A,))])

        text = render_markdown(report)

        for metric in (
            "recall@block",
            "MRR",
            "precision@block",
            "empty_on_noise",
            "forgotten_leak",
            "budget_ok",
        ):
            assert metric in text
