"""`score_runs.py`, run as a script against a stub code root.

The scorer and the classifier `score_runs.py` drives are imported from `--code-root`, so each
test writes a stub tree under `tmp_path` holding stub `score_extraction_run.py` and
`diagnose_list_misses.py` with the names `score_runs.py` calls, and runs the script in a
subprocess with that tree as cwd and as `--code-root`. An in-process import would bind this
worktree's own scorer, and the classifier does not exist at this worktree's base. The real
scorer is checked by the two `--expect` reproductions of the scoring run itself.

Gold, raw and pre-H8 gold files are synthetic, written by each test.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "eval_harness" / "ceiling" / "score_runs.py"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# The aligned tests' expected intervals come from the bootstrap the script calls.
from scripts.model_bench.probes import extraction as _bootstrap  # noqa: E402

_STUB_SCORER = '''
"""Stub scorer: the shapes of the real one, with plain set matching."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


def _norm_ws(text):
    return re.sub(r"\\s+", " ", text).strip()


@dataclass(frozen=True, slots=True)
class GoldEntity:
    id: str
    type: str


@dataclass(frozen=True, slots=True)
class GoldRel:
    source: str
    source_type: str
    predicate: str
    target: str
    target_type: str


@dataclass(frozen=True, slots=True)
class GoldProbe:
    tag: str
    utterance: str
    entities: tuple
    relationships: tuple


@dataclass(frozen=True, slots=True)
class Produced:
    utterance: str
    parse_ok: bool
    entities: tuple
    entity_type_by_id: dict
    relationships: tuple


def iter_gold_probes(path):
    probes = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        probes.append(
            GoldProbe(
                tag=obj["tag"],
                utterance=obj["utterance"],
                entities=tuple(GoldEntity(e["id"], e["type"]) for e in obj["expected_entities"]),
                relationships=tuple(
                    GoldRel(r["source"], "T", r["predicate"], r["target"], "T")
                    for r in obj["expected_relationships"]
                ),
            )
        )
    return probes


def parse_produced(content):
    parsed = json.loads(content)
    entities = tuple(GoldEntity(e["id"], e["type"]) for e in parsed.get("entities", []))
    rels = tuple(
        {"source": r["source"], "predicate": r["predicate"], "target": r["target"]}
        for r in parsed.get("relationships", [])
    )
    return True, entities, {e.id: e.type for e in entities}, rels


@dataclass(slots=True)
class Report:
    entity_tp: int = 0
    entity_fp: int = 0
    entity_fn: int = 0
    rel_tp: int = 0
    rel_fp: int = 0
    rel_fn: int = 0
    per_probe: list = field(default_factory=list)

    @staticmethod
    def _pr(tp, fp, fn):
        return (tp / (tp + fp) if tp + fp else 1.0, tp / (tp + fn) if tp + fn else 1.0)

    @property
    def entity_precision(self):
        return self._pr(self.entity_tp, self.entity_fp, self.entity_fn)[0]

    @property
    def entity_precision_denominator(self):
        return self.entity_tp + self.entity_fp

    @property
    def entity_recall(self):
        return self._pr(self.entity_tp, self.entity_fp, self.entity_fn)[1]

    @property
    def entity_recall_denominator(self):
        return self.entity_tp + self.entity_fn

    @property
    def rel_precision(self):
        return self._pr(self.rel_tp, self.rel_fp, self.rel_fn)[0]

    @property
    def rel_precision_denominator(self):
        return self.rel_tp + self.rel_fp

    @property
    def rel_recall(self):
        return self._pr(self.rel_tp, self.rel_fp, self.rel_fn)[1]

    @property
    def rel_recall_denominator(self):
        return self.rel_tp + self.rel_fn


def score_run(probes, produced_index):
    report = Report()
    for probe in probes:
        produced = produced_index.get(_norm_ws(probe.utterance))
        gold_ids = {e.id for e in probe.entities}
        prod_ids = [e.id for e in produced.entities] if produced else []
        report.entity_tp += len(gold_ids & set(prod_ids))
        report.entity_fp += sum(1 for i in prod_ids if i not in gold_ids)
        report.entity_fn += len(gold_ids - set(prod_ids))
        gold_keys = {(r.source, r.predicate, r.target) for r in probe.relationships}
        prod_keys = set()
        if produced:
            prod_keys = {(r["source"], r["predicate"], r["target"]) for r in produced.relationships}
        report.rel_tp += len(gold_keys & prod_keys)
        report.rel_fp += len(prod_keys - gold_keys)
        report.rel_fn += len(gold_keys - prod_keys)
        report.per_probe.append(
            {
                "tag": probe.tag,
                "matched": produced is not None,
                "gold_relationships": len(gold_keys),
                "rel_fps": sorted(list(t) for t in prod_keys - gold_keys),
                "rel_fns": sorted(list(t) for t in gold_keys - prod_keys),
            }
        )
    return report
'''

_STUB_CLASSIFIER = '''
"""Stub classifier: NOT_RUN without a record, EMPTY on an empty payload, else OMITTED."""

from __future__ import annotations

import json
from dataclasses import dataclass

from scripts.eval_harness.score_extraction_run import parse_produced

CLASSES = ("EMPTY", "NOT_RUN", "ID_VARIANT", "OMITTED")


@dataclass(frozen=True, slots=True)
class Miss:
    tag: str
    key: tuple
    cls: str
    detail: str
    frame_endpoint: bool
    fan_out: bool


@dataclass(frozen=True, slots=True)
class RunDiagnosis:
    total: int
    found: int
    misses: list
    found_keys: set


def _beyond_prompt(current, pre_h8):
    before = {p.tag: {(r.source, r.predicate, r.target) for r in p.relationships} for p in pre_h8}
    return {
        (p.tag, (r.source, r.predicate, r.target))
        for p in current
        for r in p.relationships
        if (r.source, r.predicate, r.target) not in before.get(p.tag, set())
    }


def diagnose(probes, records, excluded):
    total = 0
    found = set()
    misses = []
    for probe in probes:
        record = records.get(probe.tag)
        keys = set()
        payload = None
        if record is not None:
            payload = (record.get("response") or {}).get("payload")
            if payload:
                _, _, _, rels = parse_produced(json.dumps(payload))
                keys = {(r["source"], r["predicate"], r["target"]) for r in rels}
        for rel in probe.relationships:
            key = (rel.source, rel.predicate, rel.target)
            if (probe.tag, key) in excluded:
                continue
            total += 1
            if key in keys:
                found.add((probe.tag, key))
                continue
            cls = "NOT_RUN" if record is None else ("OMITTED" if payload else "EMPTY")
            misses.append(Miss(probe.tag, key, cls, "", False, False))
    return RunDiagnosis(total, len(found), misses, found)
'''


def _write_stub_root(root: Path, *, scorer_tail: str = "") -> Path:
    """Write a stub code root laid out as the pinned checkout is (no top-level markers)."""
    harness = root / "scripts" / "eval_harness"
    harness.mkdir(parents=True)
    (harness / "__init__.py").write_text("", encoding="utf-8")
    (harness / "score_extraction_run.py").write_text(
        textwrap.dedent(_STUB_SCORER) + textwrap.dedent(scorer_tail), encoding="utf-8"
    )
    (harness / "diagnose_list_misses.py").write_text(
        textwrap.dedent(_STUB_CLASSIFIER), encoding="utf-8"
    )
    return root


def _ent(eid: str, etype: str = "Concept") -> dict[str, str]:
    return {"id": eid, "type": etype}


def _rel(source: str, predicate: str, target: str) -> dict[str, str]:
    return {"source": source, "predicate": predicate, "target": target}


_GOLD = [
    {
        "tag": "t-01",
        "utterance": "first  probe",
        "expected_entities": [_ent("user", "Person"), _ent("a"), _ent("b"), _ent("c")],
        "expected_relationships": [
            _rel("user", "KNOWS", "a"),
            _rel("user", "KNOWS", "b"),
            _rel("user", "USES", "c"),
        ],
    },
    {
        "tag": "t-02",
        "utterance": "second probe",
        "expected_entities": [_ent("user", "Person"), _ent("d"), _ent("e")],
        "expected_relationships": [_rel("user", "USES", "d"), _rel("user", "USES", "e")],
    },
    {
        "tag": "t-03",
        "utterance": "third probe",
        "expected_entities": [_ent("user", "Person"), _ent("f")],
        "expected_relationships": [_rel("user", "LIKES", "f")],
    },
]

# Run a: t-01 finds two of three and adds one wrong edge; t-02 finds one of two; t-03 has no row.
_RAW_A = [
    {
        "id": "t-01",
        "job_id": "j1",
        "utterance": "first  probe",
        "response": {
            "payload": {
                "entities": [_ent("user", "Person"), _ent("a"), _ent("b"), _ent("x")],
                "relationships": [
                    _rel("user", "KNOWS", "a"),
                    _rel("user", "KNOWS", "b"),
                    _rel("user", "KNOWS", "x"),
                ],
            }
        },
    },
    {
        "id": "t-02",
        "job_id": "j2",
        "utterance": "second probe",
        "response": {
            "payload": {
                "entities": [_ent("user", "Person"), _ent("d")],
                "relationships": [_rel("user", "USES", "d")],
            }
        },
    },
]

# Run b: only t-02, with both of its relations.
_RAW_B = [
    {
        "id": "t-02",
        "job_id": "j3",
        "utterance": "second probe",
        "response": {
            "payload": {
                "entities": [_ent("user", "Person"), _ent("d"), _ent("e")],
                "relationships": [_rel("user", "USES", "d"), _rel("user", "USES", "e")],
            }
        },
    },
]


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _inputs(tmp_path: Path) -> dict[str, Path]:
    data = tmp_path / "data"
    data.mkdir()
    pre = [dict(row) for row in _GOLD]
    # The pre-H8 gold lacks t-01's USES edge, so that one relation lies beyond the prompt.
    pre[0] = {**pre[0], "expected_relationships": pre[0]["expected_relationships"][:2]}
    return {
        "gold": _jsonl(data / "gold.jsonl", _GOLD),
        "pre": _jsonl(data / "pre.jsonl", pre),
        "a": _jsonl(data / "a_raw.jsonl", _RAW_A),
        "b": _jsonl(data / "b_raw.jsonl", _RAW_B),
    }


def _run(root: Path, files: dict[str, Path], *extra: str) -> subprocess.CompletedProcess[str]:
    argv = [
        sys.executable,
        str(_SCRIPT),
        "--code-root",
        str(root),
        "--gold",
        str(files["gold"]),
        "--pre-h8-gold",
        str(files["pre"]),
        "--run",
        f"a={files['a']}",
        "--run",
        f"b={files['b']}",
        *extra,
    ]
    return subprocess.run(argv, cwd=root, capture_output=True, text=True, timeout=120, check=False)


# Every figure line ends with the reading's identity after this mark (plan v14 T50).
_IDENTITY_MARK = " | scorer_sha="


def _bare(result: subprocess.CompletedProcess[str]) -> list[str]:
    """The output's lines without the identity each figure line ends with."""
    return [line.split(_IDENTITY_MARK, 1)[0] for line in result.stdout.splitlines()]


def _lines(result: subprocess.CompletedProcess[str], run: str) -> list[str]:
    return [line for line in _bare(result) if line.startswith(f"[{run}] ")]


def _identity(root: Path, gold: Path, pack: object | None = None) -> str:
    from scripts.eval_harness import governance

    return governance.identity_text(
        {
            "scorer_sha": governance.scorer_sha(root),
            "gold_version": governance.gold_version(gold, root, getattr(pack, "version", None)),
            "pack_prompt_hash": getattr(pack, "pack_prompt_hash", None),
            "pack_rails_hash": getattr(pack, "pack_rails_hash", None),
        }
    )


def _figure_lines(result: subprocess.CompletedProcess[str]) -> list[str]:
    return [line for line in result.stdout.splitlines() if line.startswith(("[", "expect "))]


def test_every_figure_line_carries_the_scorer_the_gold_and_the_pack(tmp_path: Path) -> None:
    root = _write_stub_root(tmp_path / "code")
    files = _inputs(tmp_path)
    result = _run(root, files, "--expect", "a=0.500")
    assert result.returncode == 0, result.stderr
    # A code root without a pack names none; its harness files still give a scorer sha.
    identity = _identity(root, files["gold"])
    assert "pack_prompt_hash=none pack_rails_hash=none" in identity
    lines = _figure_lines(result)
    assert any(line.startswith("[paired ") for line in lines)
    assert any(line.startswith("expect a: ") for line in lines)
    assert all(line.endswith(f" | {identity}") for line in lines), lines
    assert f"identity: {identity}" in result.stdout.splitlines()


# A prior scorer that finds nothing: each run's found count moves under the current one.
_FINDS_NOTHING = """
_score_run = score_run


def score_run(probes, produced_index):
    report = _score_run(probes, produced_index)
    report.rel_fn += report.rel_tp
    report.rel_tp = 0
    return report
"""


def test_a_re_score_under_a_changed_scorer_prints_a_scorer_effect_line_per_run(
    tmp_path: Path,
) -> None:
    from scripts.eval_harness import governance

    prior = _write_stub_root(tmp_path / "prior", scorer_tail=_FINDS_NOTHING)
    root = _write_stub_root(tmp_path / "code")
    files = _inputs(tmp_path)
    result = _run(root, files, "--prior-code-root", str(prior))
    assert result.returncode == 0, result.stderr
    prior_sha = governance.scorer_sha(prior)
    assert prior_sha != governance.scorer_sha(root)
    effects = [line for line in _bare(result) if line.startswith("[scorer effect ")]
    assert effects == [
        "[scorer effect a] relations found under the prior scorer 0/6, under this scorer "
        f"3/6, diff +3; prior scorer_sha={prior_sha}",
        "[scorer effect b] relations found under the prior scorer 0/6, under this scorer "
        f"2/6, diff +2; prior scorer_sha={prior_sha}",
    ]
    identity = _identity(root, files["gold"])
    assert all(
        line.endswith(f" | {identity}")
        for line in result.stdout.splitlines()
        if line.startswith("[scorer effect ")
    )


def test_no_prior_code_root_prints_no_scorer_effect_line(tmp_path: Path) -> None:
    result = _run(_write_stub_root(tmp_path / "code"), _inputs(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "[scorer effect" not in result.stdout


def test_a_prior_code_root_that_is_not_a_directory_exits_1(tmp_path: Path) -> None:
    root = _write_stub_root(tmp_path / "code")
    result = _run(root, _inputs(tmp_path), "--prior-code-root", str(tmp_path / "nowhere"))
    assert result.returncode == 1
    assert "--prior-code-root" in result.stderr
    assert "[a]" not in result.stdout


def test_a_change_to_the_code_roots_harness_moves_the_scorer_sha(tmp_path: Path) -> None:
    files = _inputs(tmp_path)
    one = _run(_write_stub_root(tmp_path / "one"), files)
    two = _run(_write_stub_root(tmp_path / "two", scorer_tail="\n# changed\n"), files)
    first = [line for line in one.stdout.splitlines() if line.startswith("identity: ")]
    second = [line for line in two.stdout.splitlines() if line.startswith("identity: ")]
    assert first and second and first != second


def test_relation_recall_and_precision_are_the_scorer_sums(tmp_path: Path) -> None:
    root = _write_stub_root(tmp_path / "code")
    result = _run(root, _inputs(tmp_path))
    assert result.returncode == 0, result.stderr
    lines = _lines(result, "a")
    # Per probe: found 2 + 1 + 0 of gold 3 + 2 + 1; matched 2 + 1 of produced 3 + 1.
    assert "[a] relation recall: 0.500 (3/6)" in lines
    assert "[a] relation precision: 0.750 (3/4)" in lines
    # Entities: matched 4 + 2 + 0 of gold 4 + 3 + 2; produced 4 + 2.
    assert "[a] entity recall: 0.556 (5/9)" in lines
    assert "[a] entity precision: 0.833 (5/6)" in lines
    assert "[b] relation recall: 0.333 (2/6)" in _lines(result, "b")


def test_recall_over_probes_with_a_raw_row(tmp_path: Path) -> None:
    root = _write_stub_root(tmp_path / "code")
    result = _run(root, _inputs(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "[a] relation recall over probes with a raw row: 0.600 (3/5, probes 2 of 3)" in (
        _lines(result, "a")
    )
    assert "[b] relation recall over probes with a raw row: 1.000 (2/2, probes 1 of 3)" in (
        _lines(result, "b")
    )


def test_classifier_classes_and_recall_without_relations_beyond_the_prompt(
    tmp_path: Path,
) -> None:
    root = _write_stub_root(tmp_path / "code")
    result = _run(root, _inputs(tmp_path))
    assert result.returncode == 0, result.stderr
    lines = _lines(result, "a")
    # t-01's USES edge is set aside: found 2 + 1 + 0 of 2 + 2 + 1.
    assert "[a] classifier recall without the relations beyond the prompt: 0.600 (3/5)" in lines
    assert "[a] class EMPTY: 0" in lines
    assert "[a] class NOT_RUN: 1" in lines
    assert "[a] class ID_VARIANT: 0" in lines
    assert "[a] class OMITTED: 1" in lines
    # Run b has no row for t-01 (two counted relations) or t-03 (one).
    assert "[b] class NOT_RUN: 3" in _lines(result, "b")


def test_scorer_and_classifier_come_from_the_code_root(tmp_path: Path) -> None:
    root = _write_stub_root(tmp_path / "code")
    result = _run(root, _inputs(tmp_path))
    assert result.returncode == 0, result.stderr
    provenance = dict(
        line.split(": ", 1)
        for line in result.stdout.splitlines()
        if line.startswith(("scorer: ", "classifier: "))
    )
    assert (
        Path(provenance["scorer"])
        == (root / "scripts" / "eval_harness" / "score_extraction_run.py").resolve()
    )
    assert (
        Path(provenance["classifier"])
        == (root / "scripts" / "eval_harness" / "diagnose_list_misses.py").resolve()
    )


def test_a_module_outside_the_code_root_exits_2(tmp_path: Path) -> None:
    outside = tmp_path / "outside" / "backend"
    outside.mkdir(parents=True)
    (outside / "elsewhere.py").write_text("VALUE = 1\n", encoding="utf-8")
    tail = f"""
    import sys as _sys

    _sys.path.append({str(tmp_path / "outside")!r})
    import backend.elsewhere  # noqa: E402,F401
    """
    root = _write_stub_root(tmp_path / "code", scorer_tail=tail)
    result = _run(root, _inputs(tmp_path))
    assert result.returncode == 2
    assert "backend.elsewhere" in result.stderr
    assert "[a]" not in result.stdout


def test_every_expect_holding_prints_every_run(tmp_path: Path) -> None:
    root = _write_stub_root(tmp_path / "code")
    result = _run(root, _inputs(tmp_path), "--expect", "a=0.500")
    assert result.returncode == 0, result.stderr
    assert "expect a: 0.500 holds" in result.stdout
    assert _lines(result, "a")
    assert _lines(result, "b")


def test_an_expect_mismatch_prints_only_the_expected_runs_and_exits_1(tmp_path: Path) -> None:
    root = _write_stub_root(tmp_path / "code")
    result = _run(root, _inputs(tmp_path), "--expect", "a=0.428")
    assert result.returncode == 1
    assert "expect a: wanted 0.428, got 0.500" in result.stdout
    assert "[a] relation recall: 0.500 (3/6)" in result.stdout
    assert "[b]" not in result.stdout


def test_an_expect_for_an_unnamed_run_exits_1(tmp_path: Path) -> None:
    root = _write_stub_root(tmp_path / "code")
    result = _run(root, _inputs(tmp_path), "--expect", "c=0.500")
    assert result.returncode == 1
    assert "no --run named c" in result.stderr
    assert "[a]" not in result.stdout


def test_a_missing_raw_file_exits_1(tmp_path: Path) -> None:
    root = _write_stub_root(tmp_path / "code")
    files = _inputs(tmp_path)
    files["b"] = tmp_path / "data" / "absent.jsonl"
    result = _run(root, files)
    assert result.returncode == 1
    assert "absent.jsonl" in result.stderr


# ---------------------------------------------------------------------------
# The adjudication worksheet
# ---------------------------------------------------------------------------


def _worksheet(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _with_worksheet(
    tmp_path: Path, out: Path, *extra: str, files: dict[str, Path] | None = None
) -> subprocess.CompletedProcess[str]:
    root = tmp_path / "code"
    if not root.exists():
        _write_stub_root(root)
    files = files or _inputs(tmp_path)
    return _run(root, files, "--worksheet-run", "a", "--worksheet", str(out), *extra)


def test_worksheet_items_are_the_misses_and_the_false_positives(tmp_path: Path) -> None:
    out = tmp_path / "worksheet.jsonl"
    result = _with_worksheet(tmp_path, out)
    assert result.returncode == 0, result.stderr
    probes = _worksheet(out)
    assert [p["tag"] for p in probes] == ["t-01", "t-02", "t-03"]
    assert [p["counts"] for p in probes] == [
        {
            "gold": 3,
            "found": 2,
            "produced": 3,
            "matched": 2,
            "beyond_prompt_gold": 1,
            "beyond_prompt_found": 0,
        },
        {
            "gold": 2,
            "found": 1,
            "produced": 1,
            "matched": 1,
            "beyond_prompt_gold": 0,
            "beyond_prompt_found": 0,
        },
        {
            "gold": 1,
            "found": 0,
            "produced": 0,
            "matched": 0,
            "beyond_prompt_gold": 0,
            "beyond_prompt_found": 0,
        },
    ]
    for probe in probes:
        counts = probe["counts"]
        kinds = [item["kind"] for item in probe["items"]]
        assert kinds.count("miss") == counts["gold"] - counts["found"]
        assert kinds.count("fp") == counts["produced"] - counts["matched"]
    assert probes[0]["items"] == [
        {
            "id": "t-01:miss:1",
            "kind": "miss",
            "relation": ["user", "USES", "c"],
            "class": "OMITTED",
            "beyond_prompt": True,
        },
        {
            "id": "t-01:fp:1",
            "kind": "fp",
            "relation": ["user", "KNOWS", "x"],
            "class": None,
            "beyond_prompt": False,
        },
    ]
    assert probes[2]["items"][0]["class"] == "NOT_RUN"


def test_worksheet_holds_the_probe_and_both_sides(tmp_path: Path) -> None:
    out = tmp_path / "worksheet.jsonl"
    assert _with_worksheet(tmp_path, out).returncode == 0
    probes = _worksheet(out)
    first = probes[0]
    assert list(first) == [
        "tag",
        "utterance",
        "gold_entities",
        "gold_relations",
        "produced_entities",
        "produced_relations",
        "counts",
        "items",
    ]
    assert first["utterance"] == "first  probe"
    assert first["gold_entities"] == [
        ["user", "Person"],
        ["a", "Concept"],
        ["b", "Concept"],
        ["c", "Concept"],
    ]
    assert first["gold_relations"] == [
        ["user", "KNOWS", "a"],
        ["user", "KNOWS", "b"],
        ["user", "USES", "c"],
    ]
    assert first["produced_entities"] == [
        ["user", "Person"],
        ["a", "Concept"],
        ["b", "Concept"],
        ["x", "Concept"],
    ]
    assert first["produced_relations"] == [
        ["user", "KNOWS", "a"],
        ["user", "KNOWS", "b"],
        ["user", "KNOWS", "x"],
    ]
    assert probes[2]["produced_entities"] == []
    assert probes[2]["produced_relations"] == []


def test_worksheet_ids_are_stable(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    files = _inputs(tmp_path)
    assert _with_worksheet(tmp_path, first, files=files).returncode == 0
    assert _with_worksheet(tmp_path, second, files=files).returncode == 0
    assert first.read_bytes() == second.read_bytes()
    ids = [item["id"] for probe in _worksheet(first) for item in probe["items"]]
    assert ids == ["t-01:miss:1", "t-01:fp:1", "t-02:miss:1", "t-03:miss:1"]


def test_worksheet_counts_a_found_relation_beyond_the_prompt(tmp_path: Path) -> None:
    files = _inputs(tmp_path)
    pre = [dict(row) for row in _GOLD]
    # Now t-01's found KNOWS b edge lies beyond the prompt too, beside its missed USES c edge.
    pre[0] = {**pre[0], "expected_relationships": pre[0]["expected_relationships"][:1]}
    _jsonl(files["pre"], pre)
    out = tmp_path / "worksheet.jsonl"
    assert _with_worksheet(tmp_path, out, files=files).returncode == 0
    counts = _worksheet(out)[0]["counts"]
    assert (counts["beyond_prompt_gold"], counts["beyond_prompt_found"]) == (2, 1)


def test_worksheet_holds_no_recall_or_precision(tmp_path: Path) -> None:
    out = tmp_path / "worksheet.jsonl"
    assert _with_worksheet(tmp_path, out).returncode == 0
    text = out.read_text(encoding="utf-8")
    assert "recall" not in text
    assert "precision" not in text

    def floats(value: object) -> list[float]:
        if isinstance(value, float):
            return [value]
        if isinstance(value, dict):
            return [f for v in value.values() for f in floats(v)]
        if isinstance(value, list):
            return [f for v in value for f in floats(v)]
        return []

    assert floats(_worksheet(out)) == []


def test_an_existing_worksheet_exits_1_before_scoring(tmp_path: Path) -> None:
    outside = tmp_path / "outside" / "backend"
    outside.mkdir(parents=True)
    (outside / "elsewhere.py").write_text("VALUE = 1\n", encoding="utf-8")
    # Scoring against this root would exit 2, so exit 1 shows the refusal came first.
    tail = f"""
    import sys as _sys

    _sys.path.append({str(tmp_path / "outside")!r})
    import backend.elsewhere  # noqa: E402,F401
    """
    _write_stub_root(tmp_path / "code", scorer_tail=tail)
    out = tmp_path / "worksheet.jsonl"
    out.write_text("kept\n", encoding="utf-8")
    result = _with_worksheet(tmp_path, out)
    assert result.returncode == 1
    assert "already exists" in result.stderr
    assert result.stdout == ""
    assert out.read_text(encoding="utf-8") == "kept\n"


def test_an_expect_mismatch_writes_no_worksheet(tmp_path: Path) -> None:
    out = tmp_path / "worksheet.jsonl"
    result = _with_worksheet(tmp_path, out, "--expect", "b=0.900")
    assert result.returncode == 1
    assert not out.exists()
    assert "[a]" not in result.stdout


def test_the_worksheet_is_written_after_every_expect_holds(tmp_path: Path) -> None:
    out = tmp_path / "worksheet.jsonl"
    result = _with_worksheet(tmp_path, out, "--expect", "b=0.333")
    assert result.returncode == 0, result.stderr
    assert "worksheet a: probes 3, misses 3, fps 1" in result.stdout
    assert out.exists()


def test_a_worksheet_run_must_be_a_named_run(tmp_path: Path) -> None:
    root = _write_stub_root(tmp_path / "code")
    out = tmp_path / "worksheet.jsonl"
    result = _run(root, _inputs(tmp_path), "--worksheet-run", "c", "--worksheet", str(out))
    assert result.returncode == 1
    assert "no --run named c" in result.stderr
    assert not out.exists()


def test_worksheet_flags_go_together(tmp_path: Path) -> None:
    root = _write_stub_root(tmp_path / "code")
    out = tmp_path / "worksheet.jsonl"
    result = _run(root, _inputs(tmp_path), "--worksheet", str(out))
    assert result.returncode == 1
    assert "--worksheet-run" in result.stderr
    assert not out.exists()


# ---------------------------------------------------------------------------
# Recall paired on the probes two runs completed
# ---------------------------------------------------------------------------


def _run_named(
    root: Path, files: dict[str, Path], runs: list[tuple[str, Path]], *extra: str
) -> subprocess.CompletedProcess[str]:
    """Run the script with exactly the `--run` names given, in the order given."""
    argv = [
        sys.executable,
        str(_SCRIPT),
        "--code-root",
        str(root),
        "--gold",
        str(files["gold"]),
        "--pre-h8-gold",
        str(files["pre"]),
    ]
    for name, path in runs:
        argv += ["--run", f"{name}={path}"]
    argv += list(extra)
    return subprocess.run(argv, cwd=root, capture_output=True, text=True, timeout=120, check=False)


def _paired(result: subprocess.CompletedProcess[str]) -> list[str]:
    return [line for line in _bare(result) if line.startswith("[paired ")]


def test_paired_recall_is_the_scorer_sums_over_the_shared_probes(tmp_path: Path) -> None:
    root = _write_stub_root(tmp_path / "code")
    result = _run(root, _inputs(tmp_path))
    assert result.returncode == 0, result.stderr
    # Shared: t-02 only. Run a finds one of its two relations, run b both.
    assert _paired(result) == [
        "[paired a|b] relation recall over probes both runs completed: "
        "a 0.500 (1/2), b 1.000 (2/2), probes 1 of 3"
    ]


def test_a_row_with_a_null_response_is_outside_the_pair(tmp_path: Path) -> None:
    files = _inputs(tmp_path)
    failed = {"id": "t-01", "job_id": "j4", "utterance": "first  probe", "response": None}
    _jsonl(files["b"], [failed, *_RAW_B])
    root = _write_stub_root(tmp_path / "code")
    result = _run(root, files)
    assert result.returncode == 0, result.stderr
    # The per-run rule still counts the row: probes 2 of 3 for run b.
    assert "[b] relation recall over probes with a raw row: 0.400 (2/5, probes 2 of 3)" in (
        _lines(result, "b")
    )
    assert _paired(result) == [
        "[paired a|b] relation recall over probes both runs completed: "
        "a 0.500 (1/2), b 1.000 (2/2), probes 1 of 3"
    ]


def test_no_shared_probe_prints_no_ratio(tmp_path: Path) -> None:
    files = _inputs(tmp_path)
    only_third = {
        "id": "t-03",
        "job_id": "j5",
        "utterance": "third probe",
        "response": {"payload": {"entities": [], "relationships": []}},
    }
    _jsonl(files["b"], [only_third])
    root = _write_stub_root(tmp_path / "code")
    result = _run(root, files)
    assert result.returncode == 0, result.stderr
    assert _paired(result) == [
        "[paired a|b] relation recall over probes both runs completed: "
        "a n/a (0/0), b n/a (0/0), probes 0 of 3"
    ]


def test_three_runs_print_three_pairs_in_run_order(tmp_path: Path) -> None:
    files = _inputs(tmp_path)
    # Run c completes every probe and finds every relation.
    raw_c = [
        {
            "id": probe["tag"],
            "job_id": f"c{n}",
            "utterance": probe["utterance"],
            "response": {
                "payload": {
                    "entities": probe["expected_entities"],
                    "relationships": probe["expected_relationships"],
                }
            },
        }
        for n, probe in enumerate(_GOLD)
    ]
    c = _jsonl(tmp_path / "data" / "c_raw.jsonl", raw_c)
    root = _write_stub_root(tmp_path / "code")
    result = _run_named(root, files, [("b", files["b"]), ("c", c), ("a", files["a"])])
    assert result.returncode == 0, result.stderr
    assert _paired(result) == [
        "[paired b|c] relation recall over probes both runs completed: "
        "b 1.000 (2/2), c 1.000 (2/2), probes 1 of 3",
        "[paired b|a] relation recall over probes both runs completed: "
        "b 1.000 (2/2), a 0.500 (1/2), probes 1 of 3",
        "[paired c|a] relation recall over probes both runs completed: "
        "c 1.000 (5/5), a 0.600 (3/5), probes 2 of 3",
    ]


def test_paired_lines_follow_the_expectations_and_precede_the_worksheet(
    tmp_path: Path,
) -> None:
    out = tmp_path / "worksheet.jsonl"
    result = _with_worksheet(tmp_path, out, "--expect", "a=0.500")
    assert result.returncode == 0, result.stderr
    lines = _bare(result)
    holds = lines.index("expect a: 0.500 holds")
    paired = lines.index(_paired(result)[0])
    sheet = next(n for n, line in enumerate(lines) if line.startswith("worksheet a: "))
    assert holds < paired < sheet


def test_an_expect_mismatch_prints_no_paired_line(tmp_path: Path) -> None:
    root = _write_stub_root(tmp_path / "code")
    result = _run(root, _inputs(tmp_path), "--expect", "a=0.428")
    assert result.returncode == 1
    assert "[paired" not in result.stdout


def test_a_single_run_prints_no_paired_line(tmp_path: Path) -> None:
    files = _inputs(tmp_path)
    root = _write_stub_root(tmp_path / "code")
    result = _run_named(root, files, [("a", files["a"])])
    assert result.returncode == 0, result.stderr
    assert _lines(result, "a")
    assert "[paired" not in result.stdout


def test_unequal_gold_counts_over_a_pair_exit_1(tmp_path: Path) -> None:
    # A scorer whose recall denominator depends on the index breaks the shared-gold premise.
    tail = """
    _score_run = score_run


    def score_run(probes, produced_index):
        report = _score_run(probes, produced_index)
        report.rel_fn += len(produced_index)
        return report
    """
    root = _write_stub_root(tmp_path / "code", scorer_tail=tail)
    result = _run(root, _inputs(tmp_path))
    assert result.returncode == 1
    assert "paired a|b" in result.stderr
    assert "[paired" not in result.stdout


# ---------------------------------------------------------------------------
# Aligned scoring (plan v8, D8), against this checkout's own scorer
# ---------------------------------------------------------------------------
#
# The stub scorer has no aligned scoring, so these tests name this checkout as the code root.
# The gold and raw files are invented, written under `tmp_path`.

_KINDS_NONE = "alternative=0 event-date=0 metric=0"


def _typed_rel(
    source: str, source_type: str, predicate: str, target: str, target_type: str, **extra: object
) -> dict[str, object]:
    return {
        "source": source,
        "source_type": source_type,
        "predicate": predicate,
        "target": target,
        "target_type": target_type,
        **extra,
    }


def _aligned_gold(*, flag: bool = False) -> list[dict]:
    knows = {"beyond_prompt": True} if flag else {}
    return [
        {
            "tag": "syn-01-list-tools",
            "utterance": "Tools: Quill Pad and the Brindle app",
            "expected_entities": [
                _ent("user", "User"),
                _ent("quill-pad", "Technology"),
                _ent("brindle-app", "Technology"),
            ],
            "expected_relationships": [
                _typed_rel("user", "User", "USES", "quill-pad", "Technology"),
                _typed_rel("user", "User", "USES", "brindle-app", "Technology"),
            ],
        },
        {
            "tag": "syn-02-prose",
            "utterance": "I know Orla Venn",
            "expected_entities": [_ent("user", "User"), _ent("orla-venn", "Person")],
            "expected_relationships": [
                _typed_rel("user", "User", "KNOWS", "orla-venn", "Person", **knows)
            ],
        },
    ]


def _raw_row(probe: dict, entities: list[tuple[str, str]], edges: list[tuple[str, str, str]]):
    return {
        "id": probe["tag"],
        "job_id": f"job-{probe['tag']}",
        "utterance": probe["utterance"],
        "response": {
            "payload": {
                "entities": [{"id": i, "type": t} for i, t in entities],
                "relationships": [{"source": s, "type": p, "target": t} for s, p, t in edges],
            }
        },
    }


def _aligned_inputs(tmp_path: Path, *, gold_name: str = "gold.jsonl", flag: bool = False):
    """Gold, a candidate run (a reorder id, a typo id, one exact probe) and a run exact on both."""
    data = tmp_path / "aligned-data"
    data.mkdir()
    gold = _aligned_gold(flag=flag)
    first, second = gold
    cand = [
        _raw_row(
            first,
            [("user", "User"), ("pad-quill", "Technology"), ("brindel-app", "Technology")],
            [("user", "USES", "pad-quill"), ("user", "USES", "brindel-app")],
        ),
        _raw_row(
            second, [("user", "User"), ("orla-venn", "Person")], [("user", "KNOWS", "orla-venn")]
        ),
    ]
    base = [
        _raw_row(
            first,
            [("user", "User"), ("quill-pad", "Technology"), ("brindle-app", "Technology")],
            [("user", "USES", "quill-pad"), ("user", "USES", "brindle-app")],
        ),
        cand[1],
    ]
    return {
        "gold": _jsonl(data / gold_name, gold),
        "cand": _jsonl(data / "cand_raw.jsonl", cand),
        "base": _jsonl(data / "base_raw.jsonl", base),
    }


def _run_aligned(
    files: dict[str, Path], runs: list[str], *extra: str, root: Path = _REPO_ROOT
) -> subprocess.CompletedProcess[str]:
    argv = [sys.executable, str(_SCRIPT), "--code-root", str(root), "--gold", str(files["gold"])]
    for name in runs:
        argv += ["--run", f"{name}={files[name]}"]
    argv += list(extra)
    return subprocess.run(argv, cwd=root, capture_output=True, text=True, timeout=300, check=False)


def _aligned_line(result: subprocess.CompletedProcess[str], run: str) -> str:
    hits = [line for line in _bare(result) if line.startswith(f"[aligned {run}] ")]
    assert len(hits) == 1, result.stdout
    return hits[0]


def _ci(counts: dict[str, tuple[int, int, int]], kind: str) -> str:
    bounds = _bootstrap.cluster_bootstrap_ratio_ci(
        counts, n_replicates=10_000, seed=0, confidence=0.95, kind=kind
    )
    assert bounds is not None
    return f"ci95 [{bounds[0]:.3f}, {bounds[1]:.3f}]"


def _accepted(tmp_path: Path, rows: list[dict]) -> Path:
    return _jsonl(tmp_path / "accepted.jsonl", rows)


_TYPO_ROW = {"tag": "syn-01-list-tools", "gold_id": "brindle-app", "produced_id": "brindel-app"}


def test_the_aligned_line_carries_typing_and_list_only_recall(tmp_path: Path) -> None:
    result = _run_aligned(_aligned_inputs(tmp_path), ["cand"])
    assert result.returncode == 0, result.stderr
    line = _aligned_line(result, "cand")
    # syn-01: the reorder id finds one relation, the unaccepted typo id is a miss and an extra;
    # syn-02 is exact. Pairs: user twice and orla-venn exact, pad-quill reorder.
    counts = {"syn-01-list-tools": (1, 1, 1), "syn-02-prose": (1, 0, 0)}
    head, _, rest = line.partition("; typing ")
    assert head == (
        f"[aligned cand] recall 0.667 (2/3) {_ci(counts, 'recall')}; "
        f"precision 0.667 (2/3) {_ci(counts, 'precision')}; "
        f"f1 0.667 {_ci(counts, 'f1')}; id agreement 0.750 (3/4)"
    )
    # Typing is the scorer's relation-typing accuracy over the three produced relations.
    assert re.match(r"\d\.\d{3} \(\d/3\); list-only recall ", rest), rest
    assert line.endswith(
        "; list-only recall 0.500 (1/2); kinds exact=3 "
        f"{_KINDS_NONE} n1=0 reorder=1 typo=0 paraphrase=0; pending 1 accepted 0"
    )
    # The exact lines are still there, unchanged in form.
    assert "[cand] relation recall: 0.333 (1/3)" in _lines(result, "cand")


def test_a_code_root_with_a_pack_names_its_hashes_on_every_figure_line(tmp_path: Path) -> None:
    from backend.knowledge.ontologies import registry

    files = _aligned_inputs(tmp_path)
    result = _run_aligned(files, ["cand", "base"])
    assert result.returncode == 0, result.stderr
    identity = _identity(_REPO_ROOT, files["gold"], registry.active_pack())
    assert "pack_rails_hash=none" not in identity
    lines = _figure_lines(result)
    assert any(line.startswith("[paired-aligned ") for line in lines)
    assert all(line.endswith(f" | {identity}") for line in lines), lines


def test_a_gold_with_no_flags_and_no_pre_h8_gold_scores_every_relation(tmp_path: Path) -> None:
    result = _run_aligned(_aligned_inputs(tmp_path), ["cand"])
    assert result.returncode == 0, result.stderr
    assert "asked-for basis: n/a (no beyond-prompt flags)" in result.stdout.splitlines()
    lines = _lines(result, "cand")
    assert "[cand] classifier recall over every relation: 0.333 (1/3)" in lines
    assert not any("without the relations beyond the prompt" in line for line in lines)


def test_the_gold_flags_set_relations_aside_without_the_pre_h8_gold(tmp_path: Path) -> None:
    result = _run_aligned(_aligned_inputs(tmp_path, flag=True), ["cand"])
    assert result.returncode == 0, result.stderr
    assert "asked-for basis: the gold's beyond_prompt flags, 1 relation set aside" in (
        result.stdout.splitlines()
    )
    # syn-02's KNOWS edge is set aside; neither of syn-01's is found by exact ids.
    assert "[cand] classifier recall without the relations beyond the prompt: 0.000 (0/2)" in (
        _lines(result, "cand")
    )


def test_a_pairs_file_holds_no_run_name(tmp_path: Path) -> None:
    files = _aligned_inputs(tmp_path)
    out = tmp_path / "pairs"
    result = _run_aligned(files, ["cand"], "--pairs-out", str(out))
    assert result.returncode == 0, result.stderr
    assert sorted(p.name for p in out.iterdir()) == ["cand.jsonl"]
    text = (out / "cand.jsonl").read_text(encoding="utf-8")
    assert "cand" not in text
    assert [json.loads(line) for line in text.splitlines()] == [
        {
            "tag": "syn-01-list-tools",
            "utterance": "Tools: Quill Pad and the Brindle app",
            "gold_id": "brindle-app",
            "gold_type": "Technology",
            "produced_id": "brindel-app",
            "produced_type": "Technology",
            "kind": "typo",
        }
    ]


def test_an_accepted_row_moves_its_pair_into_the_aligned_counts(tmp_path: Path) -> None:
    files = _aligned_inputs(tmp_path)
    accepted = _accepted(tmp_path, [{**_TYPO_ROW, "accept": True}])
    result = _run_aligned(files, ["cand"], "--accepted-pairs", str(accepted))
    assert result.returncode == 0, result.stderr
    line = _aligned_line(result, "cand")
    assert line.startswith("[aligned cand] recall 1.000 (3/3) ci95 [1.000, 1.000]; ")
    assert "; precision 1.000 (3/3) " in line
    assert "typo=1" in line
    assert line.endswith("; pending 1 accepted 1")
    # The exact figure does not move.
    assert "[cand] relation recall: 0.333 (1/3)" in _lines(result, "cand")


@pytest.mark.parametrize(
    "rows",
    [
        [{**_TYPO_ROW, "accept": False}],
        [{**_TYPO_ROW, "tag": "syn-02-prose", "accept": True}],
        [],
    ],
    ids=["rejected", "another-probe", "absent"],
)
def test_a_rejected_or_absent_row_leaves_a_miss_and_a_false_positive(
    tmp_path: Path, rows: list[dict]
) -> None:
    files = _aligned_inputs(tmp_path)
    accepted = _accepted(tmp_path, rows)
    result = _run_aligned(files, ["cand"], "--accepted-pairs", str(accepted))
    assert result.returncode == 0, result.stderr
    line = _aligned_line(result, "cand")
    assert line.startswith("[aligned cand] recall 0.667 (2/3) ")
    assert "; precision 0.667 (2/3) " in line
    assert line.endswith("; pending 1 accepted 0")


def test_a_heldout_gold_with_pairs_out_exits_1_before_any_file_is_written(
    tmp_path: Path,
) -> None:
    files = _aligned_inputs(tmp_path, gold_name="heldout-synthetic.jsonl")
    out = tmp_path / "pairs"
    result = _run_aligned(files, ["cand"], "--pairs-out", str(out))
    assert result.returncode == 1
    assert "heldout-" in result.stderr
    assert not out.exists()
    assert "[cand]" not in result.stdout


def test_a_selection_gold_with_pairs_out_exits_1_before_any_file_is_written(
    tmp_path: Path,
) -> None:
    files = _aligned_inputs(tmp_path, gold_name="selection-synthetic.jsonl")
    out = tmp_path / "pairs"
    result = _run_aligned(files, ["cand"], "--pairs-out", str(out))
    assert result.returncode == 1
    assert "selection-" in result.stderr
    assert not out.exists()
    assert "[cand]" not in result.stdout


def test_an_existing_pairs_file_exits_1_before_scoring(tmp_path: Path) -> None:
    files = _aligned_inputs(tmp_path)
    out = tmp_path / "pairs"
    out.mkdir()
    (out / "cand.jsonl").write_text("kept\n", encoding="utf-8")
    result = _run_aligned(files, ["cand"], "--pairs-out", str(out))
    assert result.returncode == 1
    assert "cand.jsonl" in result.stderr
    assert (out / "cand.jsonl").read_text(encoding="utf-8") == "kept\n"
    assert "[cand]" not in result.stdout


@pytest.mark.parametrize(
    "row",
    [
        {"tag": "syn-01-list-tools", "gold_id": "brindle-app", "accept": True},
        {**_TYPO_ROW, "accept": "yes"},
        ["syn-01-list-tools", "brindle-app", "brindel-app"],
    ],
    ids=["no-produced-id", "accept-not-bool", "not-an-object"],
)
def test_a_malformed_accepted_row_exits_1(tmp_path: Path, row: object) -> None:
    files = _aligned_inputs(tmp_path)
    accepted = tmp_path / "accepted.jsonl"
    accepted.write_text(json.dumps(row) + "\n", encoding="utf-8")
    result = _run_aligned(files, ["cand"], "--accepted-pairs", str(accepted))
    assert result.returncode == 1
    assert "accepted.jsonl" in result.stderr
    assert "[cand]" not in result.stdout


def test_two_rows_disagreeing_on_one_pair_exit_1(tmp_path: Path) -> None:
    files = _aligned_inputs(tmp_path)
    one = _jsonl(tmp_path / "one.jsonl", [{**_TYPO_ROW, "accept": True}])
    two = _jsonl(tmp_path / "two.jsonl", [{**_TYPO_ROW, "accept": False}])
    result = _run_aligned(
        files, ["cand"], "--accepted-pairs", str(one), "--accepted-pairs", str(two)
    )
    assert result.returncode == 1
    assert "brindel-app" in result.stderr


def test_the_paired_aligned_line_is_second_minus_first(tmp_path: Path) -> None:
    result = _run_aligned(_aligned_inputs(tmp_path), ["cand", "base"])
    assert result.returncode == 0, result.stderr
    cand = {"syn-01-list-tools": (1, 1, 1), "syn-02-prose": (1, 0, 0)}
    base = {"syn-01-list-tools": (2, 0, 0), "syn-02-prose": (1, 0, 0)}

    def diff_ci(kind: str) -> str:
        bounds = _bootstrap.cluster_bootstrap_paired_difference_ci(
            cand, base, n_replicates=10_000, seed=0, confidence=0.95, kind=kind
        )
        assert bounds is not None
        return f"ci95 [{bounds[0]:+.3f}, {bounds[1]:+.3f}]"

    paired = [line for line in _bare(result) if line.startswith("[paired-aligned ")]
    assert paired == [
        f"[paired-aligned cand|base] recall diff +0.333 {diff_ci('recall')}; "
        f"precision diff +0.333 {diff_ci('precision')}; probes 2 of 2"
    ]
    # The exact paired line keeps its form, and no `[paired ` filter picks up the new one.
    assert _paired(result) == [
        "[paired cand|base] relation recall over probes both runs completed: "
        "cand 0.333 (1/3), base 1.000 (3/3), probes 2 of 2"
    ]


def test_a_scorer_without_aligned_scoring_refuses_pairs_out(tmp_path: Path) -> None:
    root = _write_stub_root(tmp_path / "code")
    files = _inputs(tmp_path)
    out = tmp_path / "pairs"
    result = _run(root, files, "--pairs-out", str(out))
    assert result.returncode == 1
    assert "aligned" in result.stderr
    assert not out.exists()


def test_a_scorer_without_aligned_scoring_prints_no_aligned_line(tmp_path: Path) -> None:
    root = _write_stub_root(tmp_path / "code")
    result = _run(root, _inputs(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "[aligned" not in result.stdout
    assert "[paired-aligned" not in result.stdout


# --- the headline and the model-written column (plan v15, T50 Acceptance 2) -------------------
#
# The candidate's reorder id and typo id are pairs of kinds 6 and 7, so the headline, which
# counts kinds 1 to 4 only, finds neither of its syn-01 relations. The pack at this checkout
# has no relation-phase rule, so the model-written column equals the headline here; the
# scorer's own tests pin a column that differs.

_CAND_HEADLINE = {"syn-01-list-tools": (0, 2, 2), "syn-02-prose": (1, 0, 0)}
_BASE_HEADLINE = {"syn-01-list-tools": (2, 0, 0), "syn-02-prose": (1, 0, 0)}


def _with_fields(path: Path, **fields: object) -> None:
    """Rewrite a raw file with `fields` in every row's response."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row["response"].update(fields)
    _jsonl(path, rows)


def _column_line(result: subprocess.CompletedProcess[str], prefix: str) -> str:
    hits = [line for line in _bare(result) if line.startswith(prefix)]
    assert len(hits) == 1, result.stdout
    return hits[0]


def _diff_ci(first: dict, second: dict, kind: str) -> str:
    bounds = _bootstrap.cluster_bootstrap_paired_difference_ci(
        first, second, n_replicates=10_000, seed=0, confidence=0.95, kind=kind
    )
    assert bounds is not None
    return f"ci95 [{bounds[0]:+.3f}, {bounds[1]:+.3f}]"


def _recorded_inputs(tmp_path: Path) -> dict[str, Path]:
    files = _aligned_inputs(tmp_path)
    for name in ("cand", "base"):
        _with_fields(files[name], batch_record=[])
    return files


def test_each_run_prints_the_headline_and_the_model_written_column(tmp_path: Path) -> None:
    result = _run_aligned(_recorded_inputs(tmp_path), ["cand"])
    assert result.returncode == 0, result.stderr
    figures = (
        f"recall 0.333 (1/3) {_ci(_CAND_HEADLINE, 'recall')}; "
        f"precision 0.333 (1/3) {_ci(_CAND_HEADLINE, 'precision')}; "
        f"f1 0.333 {_ci(_CAND_HEADLINE, 'f1')}"
    )
    assert _column_line(result, "[exact k1-4 cand] ") == f"[exact k1-4 cand] {figures}"
    assert _column_line(result, "[model-written cand] ") == f"[model-written cand] {figures}"
    # The aligned figures stay beside both columns.
    aligned = _aligned_line(result, "cand").removeprefix("[aligned cand] ")
    assert _column_line(result, "[aligned model-written cand] ") == (
        f"[aligned model-written cand] {aligned}"
    )
    assert _column_line(result, "[relation rails cand] ") == (
        "[relation rails cand] relation_rails_added=0 relation_rails_removed=0 "
        "found_only_by_relation_rails=0"
    )


def test_each_pair_prints_both_columns_with_paired_intervals(tmp_path: Path) -> None:
    result = _run_aligned(_recorded_inputs(tmp_path), ["cand", "base"])
    assert result.returncode == 0, result.stderr
    diffs = (
        f"recall diff +0.667 {_diff_ci(_CAND_HEADLINE, _BASE_HEADLINE, 'recall')}; "
        f"precision diff +0.667 {_diff_ci(_CAND_HEADLINE, _BASE_HEADLINE, 'precision')}; "
        f"f1 diff +0.667 {_diff_ci(_CAND_HEADLINE, _BASE_HEADLINE, 'f1')}; probes 2 of 2"
    )
    assert _column_line(result, "[paired-exact k1-4 ") == f"[paired-exact k1-4 cand|base] {diffs}"
    assert _column_line(result, "[paired-model-written ") == (
        f"[paired-model-written cand|base] {diffs}"
    )
    # No `[paired ` filter picks up either new line.
    assert len(_paired(result)) == 1


def test_the_new_lines_carry_the_identity(tmp_path: Path) -> None:
    result = _run_aligned(_recorded_inputs(tmp_path), ["cand", "base"])
    assert result.returncode == 0, result.stderr
    new = [
        line
        for line in result.stdout.splitlines()
        if line.startswith(("[exact k1-4 ", "[model-written ", "[relation rails ", "[paired-"))
    ]
    assert len(new) == 9, result.stdout
    assert all(_IDENTITY_MARK in line for line in new)


def test_a_scorer_without_aligned_scoring_prints_neither_column(tmp_path: Path) -> None:
    root = _write_stub_root(tmp_path / "code")
    result = _run(root, _inputs(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "[exact k1-4" not in result.stdout
    assert "model-written" not in result.stdout
