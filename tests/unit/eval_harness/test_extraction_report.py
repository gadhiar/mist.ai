"""The dev and held-out extraction report, on hand-built gauntlet run directories.

Every run here is written under `tmp_path` in the shape
`scripts.model_bench.probes.extraction_service` writes: `extraction_service_summary.json`
and `extraction_service.jsonl`. No gold file is opened: each summary's `gold_corpus` names a
path that does not exist, so a report that tried to read it would fail these tests.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.eval_harness import extraction_report as er  # noqa: E402
from scripts.eval_harness.score_extraction_run import (  # noqa: E402
    GoldEntity,
    GoldProbe,
    GoldRel,
    Produced,
    score_run,
)
from scripts.model_bench.probes import extraction_service as gauntlet  # noqa: E402
from scripts.model_bench.probes.extraction import (  # noqa: E402
    build_per_item_rows,
    build_summary,
)

RID_A = "a" * 64
RID_B = "b" * 64


def _failure_classes(**counts: int) -> dict[str, int]:
    classes = {name: 0 for name in gauntlet.OUTCOME_CLASSES}
    classes.update(counts)
    return classes


def _summary(tmp_path: Path, corpus: str, **overrides: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "schema": 1,
        "gold_corpus": str(tmp_path / "never-created" / corpus),
        "total_probes": 3,
        "matched_probes": 3,
        "complete": True,
        "entity_precision": 0.9,
        "entity_recall": 0.6,
        "rel_precision": 0.75,
        "rel_recall": 0.5,
        "rel_f1": 0.6,
        "typing_accuracy": 0.95,
        "related_to_rate": 0.125,
        "wilson": {"typing_accuracy": [0.81, 0.99]},
        "bootstrap": {
            "method": "cluster-by-probe-id",
            "seed": 1234,
            "B": 2000,
            "confidence": 0.95,
            "entity_precision": [0.85, 0.95],
            "entity_recall": [0.5, 0.7],
            "rel_precision": [0.6, 0.875],
            "rel_recall": [0.4, 0.625],
            "rel_f1": [0.45, 0.72],
        },
        "run_identity": RID_A,
        "run_identity_components": {"prompt": "p"},
        "run_identity_mismatches": [],
        "cases_run": 3,
        "failed_cases": 0,
        "cases_not_run": 0,
        "not_run_probe_ids": [],
        "failure_classes": _failure_classes(ok=3),
    }
    summary.update(overrides)
    return summary


def _row(tag: str, *, gold_rel: int, rel_fp: int, rel_fn: int, **extra: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": tag,
        "matched": True,
        "errored": False,
        "gold_entities": 2,
        "gold_relationships": gold_rel,
        "entity_fp": 0,
        "entity_fn": 0,
        "rel_fp": rel_fp,
        "rel_fn": rel_fn,
        "outcome_class": "ok",
    }
    row.update(extra)
    return row


DEV_ROWS = [
    # list probes: tp 3+1=4, fp 1+1=2, fn 1+3=4 -> P 4/6, R 4/8
    _row("dvl-01-list-dash-current-tools", gold_rel=4, rel_fp=1, rel_fn=1, stage2_calls=1),
    _row("dvl-31-list-long-fresh-laptop-installs", gold_rel=4, rel_fp=1, rel_fn=3, stage2_calls=3),
    # not a list probe: its slug only contains "list"
    _row("ext-07-shopping-list", gold_rel=2, rel_fp=5, rel_fn=0, stage2_calls=1),
]


def _write_run(
    run_dir: Path, summary: dict[str, Any] | None, rows: list[dict[str, Any]] | None
) -> Path:
    run_dir.mkdir(parents=True)
    if summary is not None:
        (run_dir / "extraction_service_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
    if rows is not None:
        (run_dir / "extraction_service.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
        )
    return run_dir


def _run_report(tmp_path: Path, *runs: Path) -> tuple[int, str]:
    out = tmp_path / "report" / "extraction.md"
    argv: list[str] = []
    for run in runs:
        argv += ["--run", str(run)]
    code = er.main([*argv, "--out", str(out)])
    return code, out.read_text(encoding="utf-8") if out.exists() else ""


def _cells(md: str, label: str) -> list[str]:
    """The cells after the label of the one table row whose first cell is `label`."""
    hits = [
        [c.strip() for c in line.strip().strip("|").split("|")]
        for line in md.splitlines()
        if line.startswith("|") and line.strip("|").split("|")[0].strip() == label
    ]
    assert len(hits) == 1, f"expected one row {label!r}, got {hits}"
    return hits[0][1:]


@pytest.fixture
def dev_and_heldout(tmp_path: Path) -> tuple[Path, Path]:
    dev = _write_run(
        tmp_path / "runs" / "dev", _summary(tmp_path, "dev-lists-2026-11.jsonl"), DEV_ROWS
    )
    heldout = _write_run(
        tmp_path / "runs" / "heldout",
        _summary(
            tmp_path,
            "heldout-2026-11.jsonl",
            entity_precision=0.8,
            rel_precision=0.5,
            rel_recall=0.25,
            rel_f1=1 / 3,
            bootstrap={
                "method": "cluster-by-probe-id",
                "B": 2000,
                "confidence": 0.95,
                "rel_precision": None,
            },
            matched_probes=1,
            complete=False,
            cases_run=2,
            cases_not_run=1,
            failure_classes=_failure_classes(ok=1, timeout=1),
        ),
        [
            _row("ho-001-list-nested-tools", gold_rel=2, rel_fp=0, rel_fn=1),
            _row("ho-002-multi-fact-move", gold_rel=1, rel_fp=1, rel_fn=1),
        ],
    )
    return dev, heldout


def test_dev_and_heldout_side_by_side(
    tmp_path: Path, dev_and_heldout: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    dev, heldout = dev_and_heldout
    code, md = _run_report(tmp_path, dev, heldout)
    assert code == 0
    header = [c.strip() for c in md.splitlines()[_header_index(md)].strip("|").split("|")]
    assert header == ["metric", str(dev), str(heldout)]
    assert _cells(md, "gold corpus") == ["dev-lists-2026-11.jsonl", "heldout-2026-11.jsonl"]
    assert _cells(md, "run identity") == [f"`{RID_A}`", f"`{RID_A}`"]
    assert _cells(md, "probes matched") == ["3/3", "1/3"]
    assert _cells(md, "entity precision") == ["0.9000 [0.8500, 0.9500]", "0.8000 [no interval]"]
    assert _cells(md, "entity recall")[0] == "0.6000 [0.5000, 0.7000]"
    # Entity F1: 2PR/(P+R) of the written precision and recall, no interval.
    assert _cells(md, "entity F1")[0] == "0.7200"
    assert _cells(md, "relation precision") == ["0.7500 [0.6000, 0.8750]", "0.5000 [no interval]"]
    assert _cells(md, "relation recall")[0] == "0.5000 [0.4000, 0.6250]"
    # Relation F1 is the summary's own `rel_f1`, not recomputed, with the gauntlet's
    # interval when the run recorded one.
    assert _cells(md, "relation F1") == ["0.6000 [0.4500, 0.7200]", "0.3333 [no interval]"]
    assert _cells(md, "typing accuracy")[0] == "0.9500 [0.8100, 0.9900] Wilson"
    assert _cells(md, "RELATED_TO rate") == ["0.1250", "0.1250"]
    assert _cells(md, "outcomes") == ["ok 3", "ok 1, timeout 1"]
    assert _cells(md, "cases not run") == ["0", "1"]
    assert _cells(md, "bootstrap") == ["B=2000, confidence 0.95", "B=2000, confidence 0.95"]
    assert _cells(md, "Stage 2 calls") == ["5", "unrecorded"]
    assert _cells(md, "list-only relation P / R / F1") == [
        "0.6667 / 0.5000 / 0.5714 (2 probes)",
        "1.0000 / 0.5000 / 0.6667 (1 probe)",
    ]
    assert "[WARN]" not in md
    printed = capsys.readouterr()
    assert md.strip() in printed.out
    assert "[WARN]" not in printed.err


def _header_index(md: str) -> int:
    lines = md.splitlines()
    return next(i for i, line in enumerate(lines) if line.startswith("| metric"))


def test_one_table_only(tmp_path: Path, dev_and_heldout: tuple[Path, Path]) -> None:
    code, md = _run_report(tmp_path, *dev_and_heldout)
    assert code == 0
    assert sum(1 for line in md.splitlines() if line.startswith("|---")) == 1


def test_run_without_identity_is_unrecorded(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    old_summary = _summary(tmp_path, "dev-lists-2026-11.jsonl")
    del old_summary["run_identity"]
    del old_summary["run_identity_components"]
    del old_summary["run_identity_mismatches"]
    old = _write_run(tmp_path / "old", old_summary, DEV_ROWS)
    null = _write_run(
        tmp_path / "null", _summary(tmp_path, "heldout-2026-11.jsonl", run_identity=None), DEV_ROWS
    )
    code, md = _run_report(tmp_path, old, null)
    assert code == 0
    assert _cells(md, "run identity") == ["unrecorded", "unrecorded"]
    err = capsys.readouterr().err
    assert "differ" not in err
    assert f"[WARN] run identity unrecorded for {old}" in err


def test_single_unrecorded_run_has_no_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = _write_run(
        tmp_path / "only",
        _summary(tmp_path, "dev-lists-2026-11.jsonl", run_identity=None),
        DEV_ROWS,
    )
    code, md = _run_report(tmp_path, run)
    assert code == 0
    assert _cells(md, "run identity") == ["unrecorded"]
    assert "[WARN]" not in capsys.readouterr().err


def test_mismatched_identities_warn(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dev = _write_run(tmp_path / "dev", _summary(tmp_path, "dev-lists-2026-11.jsonl"), DEV_ROWS)
    heldout = _write_run(
        tmp_path / "heldout",
        _summary(tmp_path, "heldout-2026-11.jsonl", run_identity=RID_B),
        DEV_ROWS,
    )
    code, md = _run_report(tmp_path, dev, heldout)
    assert code == 0
    err = capsys.readouterr().err
    warning = f"[WARN] run identities differ: {dev}={RID_A}, {heldout}={RID_B}"
    assert warning in err
    assert warning in md


def test_unrecorded_beside_recorded_warns(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dev = _write_run(tmp_path / "dev", _summary(tmp_path, "dev-lists-2026-11.jsonl"), DEV_ROWS)
    old = _write_run(
        tmp_path / "old", _summary(tmp_path, "heldout-2026-11.jsonl", run_identity=None), DEV_ROWS
    )
    code, _ = _run_report(tmp_path, dev, old)
    assert code == 0
    err = capsys.readouterr().err
    assert f"[WARN] run identity unrecorded for {old}" in err
    assert "differ" not in err


def test_within_run_stamp_mismatch_warns(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mismatches = [{"id": "dvl-01-list-dash-current-tools", "run_identity": RID_B}]
    run = _write_run(
        tmp_path / "dev",
        _summary(tmp_path, "dev-lists-2026-11.jsonl", run_identity_mismatches=mismatches),
        DEV_ROWS,
    )
    code, md = _run_report(tmp_path, run)
    assert code == 0
    warning = f"[WARN] {run}: 1 result(s) stamped with a run identity other than the run's"
    assert warning in capsys.readouterr().err
    assert warning in md


@pytest.mark.parametrize(
    ("summary_text", "rows_text"),
    [
        (None, "{}\n"),
        ("{not json", "{}\n"),
        ("[1, 2]", "{}\n"),
        ("{}", None),
        ("{}", '{"id": "x"}\nnot json\n'),
        ("{}", "[1]\n"),
    ],
    ids=[
        "no-summary",
        "summary-not-json",
        "summary-not-object",
        "no-rows",
        "rows-bad-line",
        "rows-not-object",
    ],
)
def test_unreadable_run_exits_2(
    tmp_path: Path,
    dev_and_heldout: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
    summary_text: str | None,
    rows_text: str | None,
) -> None:
    bad = tmp_path / "bad"
    bad.mkdir()
    if summary_text is not None:
        (bad / "extraction_service_summary.json").write_text(summary_text, encoding="utf-8")
    if rows_text is not None:
        (bad / "extraction_service.jsonl").write_text(rows_text, encoding="utf-8")
    code, md = _run_report(tmp_path, dev_and_heldout[0], bad)
    assert code == 2
    assert md == ""
    assert f"[ERROR] run {bad}" in capsys.readouterr().err


def test_missing_run_directory_exits_2(tmp_path: Path) -> None:
    code, md = _run_report(tmp_path, tmp_path / "absent")
    assert code == 2
    assert md == ""


def test_file_names_are_the_gauntlets() -> None:
    assert er.SUMMARY_FILE == gauntlet.OUT_SUMMARY
    assert er.ROWS_FILE == gauntlet.OUT_ROWS


def test_list_only_row_agrees_with_the_scorer_on_an_all_list_run(tmp_path: Path) -> None:
    """On a run of list probes only, the list-only row equals the scorer's pooled values.

    The summary and rows come from the gauntlet's own builders (`build_summary`,
    `build_per_item_rows`) over `score_run`, so a renamed key or a different pooling
    breaks this test.
    """
    user = GoldEntity("user", "User")
    tools = ["git", "vim", "make"]
    probes = [
        GoldProbe(
            tag=f"dvl-0{i}-list-tools-{i}",
            utterance=f"utterance {i}",
            entities=(user, *(GoldEntity(t, "Tool") for t in tools)),
            relationships=tuple(GoldRel("user", "User", "USES", t, "Tool") for t in tools),
        )
        for i in range(1, 4)
    ]

    def produced(utterance: str, kept: list[str], extra: list[str]) -> Produced:
        ents = (user, *(GoldEntity(t, "Tool") for t in kept + extra))
        return Produced(
            utterance=utterance,
            parse_ok=True,
            entities=ents,
            entity_type_by_id={e.id: e.type for e in ents},
            relationships=tuple(
                {"source": "user", "predicate": "USES", "target": t, "properties": {}}
                for t in kept + extra
            ),
        )

    index = {
        "utterance 1": produced("utterance 1", ["git", "vim", "make"], []),
        "utterance 2": produced("utterance 2", ["git"], ["ed", "nano"]),
    }
    report = score_run(probes, index)
    gold = tmp_path / "gold.jsonl"
    gold.write_text("{}\n", encoding="utf-8")
    summary = build_summary(
        report,
        gold_path=gold,
        ontology_version="test",
        bootstrap_seed=1,
        bootstrap_b=50,
        bootstrap_confidence=0.95,
        wilson_z=1.96,
    )
    gold.unlink()
    summary.update(run_identity=RID_A, cases_not_run=1, failure_classes=_failure_classes(ok=2))
    run = _write_run(tmp_path / "run", summary, build_per_item_rows(report))
    code, md = _run_report(tmp_path, run)
    assert code == 0
    p, r, f1 = summary["rel_precision"], summary["rel_recall"], summary["rel_f1"]
    assert _cells(md, "list-only relation P / R / F1") == [
        f"{p:.4f} / {r:.4f} / {f1:.4f} (3 probes)"
    ]
    lo, hi = summary["bootstrap"]["rel_recall"]
    assert _cells(md, "relation recall") == [f"{r:.4f} [{lo:.4f}, {hi:.4f}]"]


def _relation_f1_cell(tmp_path: Path, bootstrap: Any) -> str:
    run = _write_run(
        tmp_path / "run",
        _summary(tmp_path, "dev-lists-2026-11.jsonl", bootstrap=bootstrap),
        DEV_ROWS,
    )
    code, md = _run_report(tmp_path, run)
    assert code == 0
    return _cells(md, "relation F1")[0]


def test_relation_f1_cell_prints_the_recorded_interval(tmp_path: Path) -> None:
    bootstrap = {"B": 2000, "confidence": 0.95, "rel_f1": [0.5, 0.6875]}
    assert _relation_f1_cell(tmp_path, bootstrap) == "0.6000 [0.5000, 0.6875]"


@pytest.mark.parametrize(
    "bootstrap",
    [
        {"B": 2000, "confidence": 0.95, "rel_precision": [0.6, 0.875]},
        {"B": 2000, "confidence": 0.95, "rel_f1": None},
        None,
    ],
    ids=["pre-t12-summary", "empty-corpus", "no-bootstrap-block"],
)
def test_relation_f1_cell_without_an_interval(tmp_path: Path, bootstrap: Any) -> None:
    assert _relation_f1_cell(tmp_path, bootstrap) == "0.6000 [no interval]"


@pytest.mark.parametrize(
    "rel_f1",
    [[0.5], [0.5, 0.6, 0.7], ["0.5", 0.7], [0.5, True], "0.5-0.7", {"lo": 0.5, "hi": 0.7}, 0.5],
    ids=["one-bound", "three-bounds", "string-bound", "bool-bound", "string", "object", "number"],
)
def test_relation_f1_cell_with_a_malformed_interval(tmp_path: Path, rel_f1: Any) -> None:
    bootstrap = {"B": 2000, "confidence": 0.95, "rel_f1": rel_f1}
    assert _relation_f1_cell(tmp_path, bootstrap) == "0.6000 [no interval]"


def test_relation_f1_cell_reads_the_gauntlets_own_interval(tmp_path: Path) -> None:
    """A summary from the gauntlet's own builder prints its `bootstrap.rel_f1` interval."""
    user = GoldEntity("user", "User")
    tools = ["git", "vim", "make"]
    probes = [
        GoldProbe(
            tag=f"ext-0{i}-tools-{i}",
            utterance=f"utterance {i}",
            entities=(user, *(GoldEntity(t, "Tool") for t in tools)),
            relationships=tuple(GoldRel("user", "User", "USES", t, "Tool") for t in tools),
        )
        for i in range(1, 4)
    ]
    ents = (user, GoldEntity("git", "Tool"), GoldEntity("ed", "Tool"))
    index = {
        "utterance 1": Produced(
            utterance="utterance 1",
            parse_ok=True,
            entities=ents,
            entity_type_by_id={e.id: e.type for e in ents},
            relationships=tuple(
                {"source": "user", "predicate": "USES", "target": t, "properties": {}}
                for t in ("git", "ed")
            ),
        )
    }
    report = score_run(probes, index)
    gold = tmp_path / "gold.jsonl"
    gold.write_text("{}\n", encoding="utf-8")
    summary = build_summary(
        report,
        gold_path=gold,
        ontology_version="test",
        bootstrap_seed=1,
        bootstrap_b=50,
        bootstrap_confidence=0.95,
        wilson_z=1.96,
    )
    gold.unlink()
    summary.update(run_identity=RID_A, failure_classes=_failure_classes(ok=1))
    run = _write_run(tmp_path / "run", summary, build_per_item_rows(report))
    code, md = _run_report(tmp_path, run)
    assert code == 0
    lo, hi = summary["bootstrap"]["rel_f1"]
    assert _cells(md, "relation F1") == [f"{summary['rel_f1']:.4f} [{lo:.4f}, {hi:.4f}]"]


def test_notes_say_which_f1_cells_carry_an_interval(
    tmp_path: Path, dev_and_heldout: tuple[Path, Path]
) -> None:
    code, md = _run_report(tmp_path, *dev_and_heldout)
    assert code == 0
    assert "computes no F1 interval" not in md
    assert "Relation F1 carries the gauntlet's cluster bootstrap when the run recorded one" in md
    assert "Entity F1 and the list-only row carry no interval" in md


_ALIGNED_ROWS = (
    "aligned relation precision",
    "aligned relation recall",
    "aligned relation F1",
    "id agreement",
)


def test_aligned_rows_print_the_summarys_values_and_intervals(tmp_path: Path) -> None:
    summary = _summary(
        tmp_path,
        "dev-lists-2026-11.jsonl",
        aligned_relationship_precision={"value": 0.8, "ci95": [0.7, 0.9]},
        aligned_relationship_recall={"value": 0.625, "ci95": [0.5, 0.75]},
        aligned_relationship_f1={"value": 0.7, "ci95": None},
        id_agreement=0.95,
    )
    run = _write_run(tmp_path / "run", summary, DEV_ROWS)
    code, md = _run_report(tmp_path, run)
    assert code == 0
    assert _cells(md, "aligned relation precision") == ["0.8000 [0.7000, 0.9000]"]
    assert _cells(md, "aligned relation recall") == ["0.6250 [0.5000, 0.7500]"]
    assert _cells(md, "aligned relation F1") == ["0.7000 [no interval]"]
    assert _cells(md, "id agreement") == ["0.9500"]
    # The exact rows keep their own values.
    assert _cells(md, "relation recall") == ["0.5000 [0.4000, 0.6250]"]


def test_aligned_rows_are_na_for_a_summary_from_before_aligned_scoring(
    tmp_path: Path, dev_and_heldout: tuple[Path, Path]
) -> None:
    code, md = _run_report(tmp_path, *dev_and_heldout)
    assert code == 0
    for label in _ALIGNED_ROWS:
        assert _cells(md, label) == ["n/a", "n/a"], label


def test_aligned_rows_read_the_gauntlets_own_summary(tmp_path: Path) -> None:
    """A summary from the gauntlet's builder over the real scorer prints its aligned values."""
    user = GoldEntity("user", "User")
    probes = [
        GoldProbe(
            tag="ext-01-tools",
            utterance="utterance 1",
            entities=(user, GoldEntity("quillpad", "Tool")),
            relationships=(GoldRel("user", "User", "USES", "quillpad", "Tool"),),
        )
    ]
    ents = (user, GoldEntity("the-quillpad", "Tool"))
    index = {
        "utterance 1": Produced(
            utterance="utterance 1",
            parse_ok=True,
            entities=ents,
            entity_type_by_id={e.id: e.type for e in ents},
            relationships=(
                {"source": "user", "predicate": "USES", "target": "the-quillpad", "properties": {}},
            ),
        )
    }
    report = score_run(probes, index)
    gold = tmp_path / "gold.jsonl"
    gold.write_text("{}\n", encoding="utf-8")
    summary = build_summary(
        report,
        gold_path=gold,
        ontology_version="test",
        bootstrap_seed=1,
        bootstrap_b=50,
        bootstrap_confidence=0.95,
        wilson_z=1.96,
    )
    gold.unlink()
    summary.update(run_identity=RID_A, failure_classes=_failure_classes(ok=1))
    run = _write_run(tmp_path / "run", summary, build_per_item_rows(report))
    code, md = _run_report(tmp_path, run)
    assert code == 0
    assert _cells(md, "relation recall") == ["0.0000 [0.0000, 0.0000]"]
    assert _cells(md, "aligned relation recall") == ["1.0000 [1.0000, 1.0000]"]
    assert _cells(md, "id agreement") == ["0.5000"]


def test_notes_say_what_the_aligned_rows_count(
    tmp_path: Path, dev_and_heldout: tuple[Path, Path]
) -> None:
    code, md = _run_report(tmp_path, *dev_and_heldout)
    assert code == 0
    assert "The aligned relation rows score relations over aligned entity ids" in md
    assert "Id agreement is the share of aligned entity pairs" in md
    # The existing notes stay.
    assert "Entity F1 and the list-only row carry no interval" in md


def test_module_docstring_says_which_f1_cells_carry_an_interval() -> None:
    doc = " ".join((er.__doc__ or "").split())
    assert "computes no interval for either F1" not in doc
    assert "relation F1 carries the gauntlet's cluster bootstrap when the run recorded one" in doc
    assert "entity F1 and the list-only row carry none" in doc
