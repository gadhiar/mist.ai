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
import subprocess
import sys
import textwrap
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "eval_harness" / "ceiling" / "score_runs.py"

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


def _lines(result: subprocess.CompletedProcess[str], run: str) -> list[str]:
    return [line for line in result.stdout.splitlines() if line.startswith(f"[{run}] ")]


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
