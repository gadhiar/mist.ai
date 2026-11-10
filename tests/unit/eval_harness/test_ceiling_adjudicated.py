"""`adjudicated`, on a synthetic worksheet and adjudication.

The worksheet here has the shape `score_runs.py` writes: one probe per line with its counts and
items. Six of its gold relations lie beyond the prompt and one of the six is found, so the
265-basis recall must drop that found relation from F as well as all six from G.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.eval_harness.ceiling import adjudicated  # noqa: E402


def _miss(tag: str, n: int, beyond: bool) -> dict[str, object]:
    return {
        "id": f"{tag}:miss:{n}",
        "kind": "miss",
        "relation": ["user", "USES", f"{tag}-m{n}"],
        "class": "OMITTED",
        "beyond_prompt": beyond,
    }


def _fp(tag: str, n: int) -> dict[str, object]:
    return {
        "id": f"{tag}:fp:{n}",
        "kind": "fp",
        "relation": ["user", "USES", f"{tag}-f{n}"],
        "class": None,
        "beyond_prompt": False,
    }


def _probe(tag: str, counts: dict[str, int], items: list[dict[str, object]]) -> dict[str, object]:
    return {
        "tag": tag,
        "utterance": f"utterance of {tag}",
        "gold_entities": [],
        "gold_relations": [],
        "produced_entities": [],
        "produced_relations": [],
        "counts": counts,
        "items": items,
    }


# G 15, F 6, P 10, T 6; six relations beyond the prompt, one of them found.
_WORKSHEET = [
    _probe(
        "p1",
        {
            "gold": 10,
            "found": 4,
            "produced": 6,
            "matched": 4,
            "beyond_prompt_gold": 4,
            "beyond_prompt_found": 1,
        },
        [
            _miss("p1", 1, True),
            _miss("p1", 2, True),
            _miss("p1", 3, True),
            _miss("p1", 4, False),
            _miss("p1", 5, False),
            _miss("p1", 6, False),
            _fp("p1", 1),
            _fp("p1", 2),
        ],
    ),
    _probe(
        "p2",
        {
            "gold": 5,
            "found": 2,
            "produced": 4,
            "matched": 2,
            "beyond_prompt_gold": 2,
            "beyond_prompt_found": 0,
        },
        [
            _miss("p2", 1, True),
            _miss("p2", 2, True),
            _miss("p2", 3, False),
            _fp("p2", 1),
            _fp("p2", 2),
        ],
    ),
]

_CLASSES = {
    "p1:miss:1": "gold_wrong",
    "p1:miss:2": "scorer_artifact",
    "p1:miss:3": "answerer_error",
    "p1:miss:4": "gold_wrong",
    "p1:miss:5": "scorer_artifact",
    "p1:miss:6": "ontology_gap",
    "p1:fp:1": "gold_missing",
    "p1:fp:2": "scorer_artifact",
    "p2:miss:1": "scorer_artifact",
    "p2:miss:2": "ambiguous",
    "p2:miss:3": "answerer_error",
    "p2:fp:1": "gold_missing",
    "p2:fp:2": "answerer_error",
}


def _adjudication() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        {"id": item_id, "class": cls, "note": "One sentence.", "pair": None}
        for item_id, cls in _CLASSES.items()
    ]
    rows[4]["pair"] = "p1:fp:2"
    rows[7]["pair"] = "p1:miss:5"
    return rows


def _write(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _main(tmp_path: Path, rows: list[dict[str, object]], *extra: str) -> int:
    worksheet = _write(tmp_path / "worksheet.jsonl", _WORKSHEET)
    adjudication = _write(tmp_path / "adjudication.jsonl", rows)
    return adjudicated.main(
        ["--worksheet", str(worksheet), "--adjudication", str(adjudication), *extra]
    )


def test_raw_and_corrected_recall_on_all_gold(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _main(tmp_path, _adjudication()) == 0
    out = capsys.readouterr().out
    # (F + Ms + Fg) / (G - Mw + Fg) = (6 + 3 + 2) / (15 - 2 + 2).
    assert "relation recall, all gold: raw 0.400 (6/15), corrected 0.733 (11/15)" in out


def test_recall_without_the_relations_beyond_the_prompt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _main(tmp_path, _adjudication()) == 0
    out = capsys.readouterr().out
    # G 15 - 6, F 6 - 1 (the found one of the six), Mw 1 and Ms 1 off the six, Fg 2:
    # (5 + 1 + 2) / (9 - 1 + 2).
    assert (
        "relation recall, relations the prompt asks for: raw 0.556 (5/9), corrected 0.800 (8/10)"
        in out
    )


def test_raw_and_corrected_precision(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _main(tmp_path, _adjudication()) == 0
    out = capsys.readouterr().out
    # (T + Fs + Fg) / P = (6 + 1 + 2) / 10.
    assert "relation precision: raw 0.600 (6/10), corrected 0.900 (9/10)" in out


def test_class_counts_by_kind(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _main(tmp_path, _adjudication()) == 0
    out = capsys.readouterr().out
    assert (
        "miss classes: answerer_error 2, gold_wrong 2, scorer_artifact 3, ontology_gap 1, "
        "ambiguous 1" in out
    )
    assert (
        "fp classes: answerer_error 1, gold_missing 2, scorer_artifact 1, ontology_gap 0, "
        "ambiguous 0" in out
    )


def test_check_passes_on_a_complete_adjudication(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _main(tmp_path, _adjudication(), "--check") == 0
    assert "check: 13 items, one valid line each" in capsys.readouterr().out


def _drop(rows: list[dict[str, object]], item_id: str) -> list[dict[str, object]]:
    return [r for r in rows if r["id"] != item_id]


def _set(rows: list[dict[str, object]], item_id: str, **fields: object) -> list[dict[str, object]]:
    return [{**r, **fields} if r["id"] == item_id else r for r in rows]


_FAILURES = {
    "an item with no line": (_drop(_adjudication(), "p2:fp:2"), "p2:fp:2: no line"),
    "an item with two lines": (
        [*_adjudication(), _adjudication()[0]],
        "p1:miss:1: 2 lines",
    ),
    "a line for no item": (
        [*_adjudication(), {"id": "p9:miss:1", "class": "ambiguous", "note": "x.", "pair": None}],
        "p9:miss:1: not a worksheet item",
    ),
    "gold_wrong on an fp": (
        _set(_adjudication(), "p1:fp:1", **{"class": "gold_wrong"}),
        "p1:fp:1: class gold_wrong is not valid for an fp",
    ),
    "gold_missing on a miss": (
        _set(_adjudication(), "p1:miss:3", **{"class": "gold_missing"}),
        "p1:miss:3: class gold_missing is not valid for a miss",
    ),
    "an unknown class": (
        _set(_adjudication(), "p1:miss:3", **{"class": "model_error"}),
        "p1:miss:3: class model_error is not valid for a miss",
    ),
    "an empty note": (_set(_adjudication(), "p1:miss:3", note="  "), "p1:miss:3: no note"),
    "no note": (
        [{k: v for k, v in r.items() if k != "note"} for r in _adjudication()],
        "p1:miss:1: no note",
    ),
    "a pair naming no item": (
        _set(_adjudication(), "p1:miss:3", pair="p1:fp:9"),
        "p1:miss:3: pair p1:fp:9 is not another worksheet item",
    ),
    "a pair naming itself": (
        _set(_adjudication(), "p1:miss:3", pair="p1:miss:3"),
        "p1:miss:3: pair p1:miss:3 is not another worksheet item",
    ),
}


@pytest.mark.parametrize("case", sorted(_FAILURES))
def test_each_check_failure_exits_1(
    case: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows, problem = _FAILURES[case]
    assert _main(tmp_path, rows, "--check") == 1
    captured = capsys.readouterr()
    assert problem in captured.out
    assert "corrected" not in captured.out


def test_an_incomplete_adjudication_prints_no_figures_without_check(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _main(tmp_path, _drop(_adjudication(), "p2:fp:2")) == 1
    captured = capsys.readouterr()
    assert "p2:fp:2: no line" in captured.out
    assert "recall" not in captured.out


def test_an_adjudication_line_that_is_not_json_exits_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    worksheet = _write(tmp_path / "worksheet.jsonl", _WORKSHEET)
    adjudication = tmp_path / "adjudication.jsonl"
    adjudication.write_text("{not json\n", encoding="utf-8")
    code = adjudicated.main(
        ["--worksheet", str(worksheet), "--adjudication", str(adjudication), "--check"]
    )
    assert code == 1
    assert "adjudication.jsonl:1" in capsys.readouterr().err


def test_a_worksheet_whose_items_disagree_with_its_counts_exits_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    broken = json.loads(json.dumps(_WORKSHEET))
    broken[1]["counts"]["found"] = 3
    worksheet = _write(tmp_path / "worksheet.jsonl", broken)
    adjudication = _write(tmp_path / "adjudication.jsonl", _adjudication())
    code = adjudicated.main(["--worksheet", str(worksheet), "--adjudication", str(adjudication)])
    assert code == 1
    assert "p2" in capsys.readouterr().err


def test_runs_as_a_module(tmp_path: Path) -> None:
    worksheet = _write(tmp_path / "worksheet.jsonl", _WORKSHEET)
    adjudication = _write(tmp_path / "adjudication.jsonl", _adjudication())
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.eval_harness.ceiling.adjudicated",
            "--worksheet",
            str(worksheet),
            "--adjudication",
            str(adjudication),
            "--check",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "corrected 0.900 (9/10)" in result.stdout


# ---------------------------------------------------------------------------
# --aligned-check: the aligner must not credit a miss the adjudication blamed on the
# answerer or on gold
# ---------------------------------------------------------------------------
#
# One invented probe, scored by this checkout's scorer: gold `quillpad`, `brindle-app` and
# `kestrel`; the raw names them `the-quillpad` (n1), `brindel-app` (typo, pending) and
# `kestrel` (exact). The worksheet's misses are the two the exact scorer misses, numbered in
# its sorted order.

_TAG = "syn-01-list-tools"
_UTTERANCE = "Tools: the Quillpad, the Brindle app and Kestrel"


def _aligned_worksheet() -> list[dict[str, object]]:
    def item(kind: str, n: int, relation: list[str]) -> dict[str, object]:
        return {
            "id": f"{_TAG}:{kind}:{n}",
            "kind": kind,
            "relation": relation,
            "class": "ID_VARIANT" if kind == "miss" else None,
            "beyond_prompt": False,
        }

    return [
        {
            "tag": _TAG,
            "utterance": _UTTERANCE,
            "gold_entities": [
                ["user", "User"],
                ["quillpad", "Tool"],
                ["brindle-app", "Tool"],
                ["kestrel", "Tool"],
            ],
            "gold_relations": [
                ["user", "USES", "quillpad"],
                ["user", "USES", "brindle-app"],
                ["user", "USES", "kestrel"],
            ],
            "produced_entities": [],
            "produced_relations": [],
            "counts": {
                "gold": 3,
                "found": 1,
                "produced": 3,
                "matched": 1,
                "beyond_prompt_gold": 0,
                "beyond_prompt_found": 0,
            },
            "items": [
                item("miss", 1, ["user", "USES", "brindle-app"]),
                item("miss", 2, ["user", "USES", "quillpad"]),
                item("fp", 1, ["user", "USES", "brindel-app"]),
                item("fp", 2, ["user", "USES", "the-quillpad"]),
            ],
        }
    ]


def _aligned_raw(produced: list[str]) -> list[dict[str, object]]:
    return [
        {
            "id": _TAG,
            "job_id": "job-1",
            "utterance": _UTTERANCE,
            "response": {
                "payload": {
                    "entities": [
                        {"id": "user", "type": "User"},
                        *({"id": i, "type": "Tool"} for i in produced),
                    ],
                    "relationships": [
                        {"source": "user", "type": "USES", "target": i} for i in produced
                    ],
                }
            },
        }
    ]


def _aligned_main(
    tmp_path: Path, classes: dict[str, str], produced: list[str] | None = None
) -> int:
    worksheet = _write(tmp_path / "worksheet.jsonl", _aligned_worksheet())
    rows = [
        {"id": f"{_TAG}:{item_id}", "class": cls, "note": "One sentence.", "pair": None}
        for item_id, cls in classes.items()
    ]
    adjudication = _write(tmp_path / "adjudication.jsonl", rows)
    raw = _write(
        tmp_path / "raw.jsonl",
        _aligned_raw(produced or ["the-quillpad", "brindel-app", "kestrel"]),
    )
    return adjudicated.main(
        [
            "--worksheet",
            str(worksheet),
            "--adjudication",
            str(adjudication),
            "--aligned-check",
            str(raw),
        ]
    )


_ARTIFACTS = {
    "miss:1": "scorer_artifact",
    "miss:2": "scorer_artifact",
    "fp:1": "scorer_artifact",
    "fp:2": "scorer_artifact",
}


def test_the_aligned_check_passes_when_it_credits_only_scorer_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _aligned_main(tmp_path, _ARTIFACTS) == 0
    out = capsys.readouterr().out
    # Both misses are credited: the n1 pair at once, the typo pair as if accepted.
    assert (
        "aligned credits by class: answerer_error 0, gold_wrong 0, scorer_artifact 2, ambiguous 0"
        in out.splitlines()
    )
    # The figures still print, as without the check.
    assert "relation recall, all gold: raw 0.333 (1/3), corrected 1.000 (3/3)" in out


@pytest.mark.parametrize(
    ("item_id", "cls"),
    [("miss:2", "answerer_error"), ("miss:1", "gold_wrong"), ("miss:1", "answerer_error")],
    ids=["n1-answerer-error", "typo-gold-wrong", "pending-typo-answerer-error"],
)
def test_an_aligned_credit_on_an_answerer_error_or_gold_wrong_miss_exits_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], item_id: str, cls: str
) -> None:
    assert _aligned_main(tmp_path, {**_ARTIFACTS, item_id: cls}) == 1
    out = capsys.readouterr().out
    line = next(x for x in out.splitlines() if x.startswith("aligned credits by class: "))
    assert f"{cls} 1" in line


def test_a_miss_the_aligner_does_not_credit_is_not_counted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # `brandle-tool` shares no form with `brindle-app`, so its miss stays a miss.
    worksheet = _aligned_worksheet()
    probe = worksheet[0]
    probe["items"][2]["relation"] = ["user", "USES", "brandle-tool"]  # type: ignore[index]
    path = _write(tmp_path / "worksheet.jsonl", worksheet)
    rows = [
        {"id": f"{_TAG}:{item_id}", "class": cls, "note": "One sentence.", "pair": None}
        for item_id, cls in {**_ARTIFACTS, "miss:1": "answerer_error"}.items()
    ]
    adjudication = _write(tmp_path / "adjudication.jsonl", rows)
    raw = _write(tmp_path / "raw.jsonl", _aligned_raw(["the-quillpad", "brandle-tool", "kestrel"]))
    code = adjudicated.main(
        ["--worksheet", str(path), "--adjudication", str(adjudication), "--aligned-check", str(raw)]
    )
    assert code == 0, capsys.readouterr()
    out = capsys.readouterr().out
    assert (
        "aligned credits by class: answerer_error 0, gold_wrong 0, scorer_artifact 1, ambiguous 0"
        in out.splitlines()
    )


def test_a_raw_that_is_not_the_worksheets_run_exits_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # This raw finds `quillpad` exactly, so its exact misses are not the worksheet's.
    assert _aligned_main(tmp_path, _ARTIFACTS, ["quillpad", "brindel-app", "kestrel"]) == 1
    captured = capsys.readouterr()
    assert _TAG in captured.err
    assert "aligned credits" not in captured.out
