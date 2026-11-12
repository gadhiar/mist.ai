"""Tests for cell P, the pack equivalence check, on synthetic payloads.

The base half always runs in a child interpreter. The equal case gives it this tree as
the base root, so both halves read the same code; the differing case gives it a stub
tree whose rails change nothing and whose prompt, pair table and schema are its own.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.knowledge.ontologies import registry  # noqa: E402
from backend.knowledge.ontologies.pack import Pack, load_pack  # noqa: E402
from scripts.eval_harness.ceiling import pack_equivalence  # noqa: E402

_HEADLINE_EQUAL = (
    "cell P: ids identical {n}/{n}; relations identical {n}/{n}; related_to_rate equal; "
    "prompt equal; pairs equal; schema equal"
)


def _entity(entity_id: str, name: str, entity_type: str, **properties) -> dict:
    entity: dict = {"id": entity_id, "name": name, "type": entity_type}
    if properties:
        entity["properties"] = properties
    return entity


def _edge(source: str, target: str, edge_type: str) -> dict:
    # A relationship's properties quote the utterance, as the service's do.
    return {
        "properties": {"context": f"WHISPERED about {target}"},
        "source": source,
        "target": target,
        "type": edge_type,
    }


_USER = _entity("user", "User", "User")

# One payload per thing the rails apply, then one they leave alone.
_PAYLOADS: list[tuple[list[dict], list[dict]]] = [
    # An alias of the static vocabulary.
    (
        [_USER, _entity("k8s", "Quaywater cluster", "Technology")],
        [_edge("user", "k8s", "USES")],
    ),
    # An event dated to one day, and an escape edge.
    (
        [
            _entity("regatta", "Harbourmaster regatta", "Event"),
            _entity("2026-05-30", "2026-05-30", "Date"),
            _entity("saltmarsh", "Saltmarsh", "Location"),
        ],
        [
            _edge("regatta", "2026-05-30", "OCCURRED_ON"),
            _edge("regatta", "saltmarsh", "RELATED_TO"),
        ],
    ),
    # A Metric that states its value.
    (
        [_USER, _entity("p50", "p50 latency", "Metric", quantity="latency", unit="ms", value=38)],
        [_edge("user", "p50", "HAS_METRIC")],
    ),
    # A retired type and a type the ontology does not have.
    (
        [
            _entity("first-release", "Brackenfold release", "Milestone"),
            _entity("hunch", "Lowtide hunch", "Feeling"),
        ],
        [],
    ),
    # Nothing a rule changes.
    (
        [_USER, _entity("zed", "Zed", "Technology")],
        [_edge("user", "zed", "USES")],
    ),
]

# Text of the rows that must never reach the report.
_ROW_TEXT = (
    "MURMURED",
    "WHISPERED",
    "Quaywater",
    "Harbourmaster",
    "Saltmarsh",
    "Brackenfold",
    "Lowtide",
    "k8s",
    "kubernetes",
    "regatta",
    "p50",
)


def _write_raw(path: Path, payloads=_PAYLOADS, *, with_empty_row: bool = True) -> Path:
    rows = [
        {
            "id": f"syn-{number:02d}",
            "utterance": f"MURMURED utterance {number}",
            "response": {
                "payload": {"entities": entities, "relationships": relationships},
                "scope": {"confidence": 0.9, "label": "user-scope"},
            },
        }
        for number, (entities, relationships) in enumerate(payloads, start=1)
    ]
    if with_empty_row:
        rows.append({"id": "syn-empty", "utterance": "MURMURED nothing", "response": None})
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


_STUB_FILES = {
    "backend/knowledge/extraction/id_rails.py": (
        "import sys\n"
        "\n"
        "if any(name.startswith('backend.knowledge.ontologies') for name in sys.modules):\n"
        "    raise ImportError('the other tree is loaded in the base half')\n"
        "\n"
        "\n"
        "def normalize_payload(entities, relationships):\n"
        "    return entities, relationships\n"
    ),
    "backend/knowledge/extraction/prompts.py": (
        f"{pack_equivalence.BASE_PROMPT_NAME} = 'stub system prompt'\n"
        "EXTRACTION_USER_TEMPLATE = 'stub user template'\n"
    ),
    "backend/knowledge/extraction/validator.py": (
        "RELATIONSHIP_CONSTRAINTS = {'USES': ({'User'}, None)}\n"
    ),
    "backend/extraction_service/schemas.py": "EXTRACTION_OUTPUT_SCHEMA = {'type': 'object'}\n",
}


def _stub_tree(root: Path) -> Path:
    """A base tree whose rails store a payload as written."""
    for name, source in _STUB_FILES.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    return root


def _files(root: Path) -> list[str]:
    return sorted(str(path.relative_to(root)) for path in root.rglob("*"))


def _variant(pack: Pack, **changes) -> Pack:
    rules = changes.pop("rules", pack.rules)
    return dataclasses.replace(
        pack,
        rules=tuple(dataclasses.replace(rule, pinned_case=None) for rule in rules),
        **changes,
    )


@pytest.fixture(autouse=True)
def _keep_the_bytecode_flag(monkeypatch):
    # The cell turns bytecode off for its own process; a test process gets it back.
    monkeypatch.setattr(sys, "dont_write_bytecode", sys.dont_write_bytecode)


@pytest.fixture
def raw(tmp_path: Path) -> Path:
    directory = tmp_path / "raw"
    directory.mkdir()
    return _write_raw(directory / "synthetic.jsonl")


def _run(capsys, *arguments: str) -> tuple[int, list[str], str]:
    status = pack_equivalence.main(list(arguments))
    captured = capsys.readouterr()
    return status, captured.out.splitlines(), captured.err


class TestEqualTrees:
    """With this tree as the base, every check holds."""

    def test_all_six_checks_hold_and_the_cell_exits_zero(self, raw: Path, capsys):
        status, lines, _ = _run(capsys, "--base-root", str(_REPO_ROOT), "--raw", f"synthetic={raw}")

        assert status == 0
        assert lines[0] == _HEADLINE_EQUAL.format(n=5)
        assert "set synthetic: 6 rows, 5 with a payload" in lines

    def test_each_thing_the_rails_apply_is_counted(self, raw: Path, capsys):
        _, lines, _ = _run(capsys, "--base-root", str(_REPO_ROOT), "--raw", f"synthetic={raw}")

        for fired in (
            "rule static-aliases: synthetic 1",
            "rule metric-id: synthetic 1",
            "rule event-date: synthetic 1",
            "manifest retired type Milestone: synthetic 1",
            "manifest fallback type Abstraction: synthetic 1",
        ):
            assert fired in lines
        assert "rule plan-vocabulary: runs before the model, so no payload fires it" in lines

    def test_a_rule_no_set_fires_rests_on_its_pinned_case(self, raw: Path, capsys, monkeypatch):
        pack = load_pack("backend.knowledge.ontologies.packs.v1_4_0")
        monkeypatch.setattr(registry, "active_pack", lambda: pack)

        status, lines, _ = _run(capsys, "--base-root", str(_REPO_ROOT), "--raw", f"synthetic={raw}")

        alone = "; no set fires it, so it rests on its pinned case alone"
        assert status == 0
        assert f"rule reserved-mist-identity: synthetic 0{alone}" in lines
        assert f"rule pre-canon-aliases: synthetic 0{alone}" in lines
        assert f"manifest retired type Topic: synthetic 0{alone}" in lines

    def test_a_rule_with_no_case_that_fires_it_is_named_so(self, raw: Path, capsys, monkeypatch):
        pack = _variant(registry.active_pack())
        monkeypatch.setattr(registry, "active_pack", lambda: pack)

        _, lines, _ = _run(capsys, "--base-root", str(_REPO_ROOT), "--raw", f"synthetic={raw}")

        unpinned = "; no set fires it, and no pinned case does either"
        assert f"rule reserved-mist-identity: synthetic 0{unpinned}" in lines

    def test_each_set_has_its_own_counts(self, raw: Path, tmp_path: Path, capsys):
        other = _write_raw(tmp_path / "other.jsonl", _PAYLOADS[-1:], with_empty_row=False)

        status, lines, _ = _run(
            capsys,
            "--base-root",
            str(_REPO_ROOT),
            "--raw",
            f"synthetic={raw}",
            "--raw",
            f"other={other}",
        )

        assert status == 0
        assert lines[0] == _HEADLINE_EQUAL.format(n=6)
        assert "set other: 1 rows, 1 with a payload" in lines
        assert "rule static-aliases: synthetic 1, other 0" in lines

    def test_the_report_holds_no_text_of_a_row(self, raw: Path, capsys):
        _, lines, err = _run(capsys, "--base-root", str(_REPO_ROOT), "--raw", f"synthetic={raw}")

        report = "\n".join(lines) + err
        assert [text for text in _ROW_TEXT if text in report] == []

    def test_the_cell_runs_as_a_module(self, raw: Path):
        done = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "scripts.eval_harness.ceiling.pack_equivalence",
                "--base-root",
                str(_REPO_ROOT),
                "--raw",
                f"synthetic={raw}",
            ],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )

        assert done.returncode == 0, done.stderr[-2000:]
        assert done.stdout.splitlines()[0] == _HEADLINE_EQUAL.format(n=5)


class TestADifferingBase:
    """A base whose rails, prompt, pair table and schema differ fails every check it should."""

    def test_each_difference_is_named_and_the_cell_exits_one(
        self, raw: Path, tmp_path: Path, capsys
    ):
        base = _stub_tree(tmp_path / "base")

        status, lines, _ = _run(capsys, "--base-root", str(base), "--raw", f"synthetic={raw}")

        assert status == 1
        assert lines[0] == (
            "cell P: ids identical 1/5; relations identical 2/5; related_to_rate equal; "
            "prompt differs; pairs differ; schema differs"
        )
        assert "set synthetic: rows whose ids differ: 1, 2, 3, 4" in lines
        assert "set synthetic: rows whose relations differ: 1, 2, 3" in lines

    def test_an_escape_edge_the_base_stores_and_this_tree_drops_moves_the_rate(
        self, tmp_path: Path, capsys
    ):
        # Written twice under two ids that the alias vocabulary joins: this tree keeps
        # one escape edge of the two, one edge in two, and the stub base keeps both,
        # two edges in three.
        entities = [
            _USER,
            _entity("k8s", "Quaywater cluster", "Technology"),
            _entity("kube", "Quaywater cluster", "Technology"),
            _entity("saltmarsh", "Saltmarsh", "Location"),
        ]
        edges = [
            _edge("k8s", "saltmarsh", "RELATED_TO"),
            _edge("kube", "saltmarsh", "RELATED_TO"),
            _edge("user", "k8s", "USES"),
        ]
        raw = _write_raw(tmp_path / "raw.jsonl", [(entities, edges)], with_empty_row=False)
        base = _stub_tree(tmp_path / "base")

        status, lines, _ = _run(capsys, "--base-root", str(base), "--raw", f"synthetic={raw}")

        assert status == 1
        assert "related_to_rate differs" in lines[0]

    def test_the_base_half_has_an_interpreter_of_its_own(self, raw: Path, tmp_path: Path, capsys):
        # The stub's rails refuse to import beside this tree's ontology modules, which
        # this process has loaded: a status of 1, not 2, says they ran in another one.
        assert "backend.knowledge.ontologies.registry" in sys.modules
        base = _stub_tree(tmp_path / "base")

        status, _, err = _run(capsys, "--base-root", str(base), "--raw", f"synthetic={raw}")

        assert (status, err) == (1, "")

    def test_a_relative_base_root_is_read_from_the_callers_directory(
        self, raw: Path, tmp_path: Path, capsys, monkeypatch
    ):
        # The base half runs with the base root as its working directory, so a relative
        # root it resolved there would name a directory inside the root.
        _stub_tree(tmp_path / "base")
        monkeypatch.chdir(tmp_path)

        status, lines, err = _run(capsys, "--base-root", "base", "--raw", f"synthetic={raw}")

        assert (status, err) == (1, "")
        assert lines[0] == (
            "cell P: ids identical 1/5; relations identical 2/5; related_to_rate equal; "
            "prompt differs; pairs differ; schema differs"
        )

    def test_nothing_is_written_into_the_base_root_or_beside_the_raw_file(
        self, raw: Path, tmp_path: Path, capsys
    ):
        base = _stub_tree(tmp_path / "base")
        before = (_files(base), _files(raw.parent), raw.read_bytes())

        _run(capsys, "--base-root", str(base), "--raw", f"synthetic={raw}")

        assert (_files(base), _files(raw.parent), raw.read_bytes()) == before

    def test_the_cell_reads_the_registrys_active_pack(self, raw: Path, capsys, monkeypatch):
        pack = registry.active_pack()
        kept = tuple(rule for rule in pack.rules if rule.id != "static-aliases")
        changed = _variant(pack, rules=kept)
        monkeypatch.setattr(registry, "active_pack", lambda: changed)

        status, lines, _ = _run(capsys, "--base-root", str(_REPO_ROOT), "--raw", f"synthetic={raw}")

        assert status == 1
        assert lines[0].startswith("cell P: ids identical 4/5; relations identical 4/5;")
        assert "set synthetic: rows whose ids differ: 1" in lines


class TestACellThatCannotRun:
    """A cell that cannot compare says why, shows no row, and exits 2."""

    def test_a_base_half_that_fails_is_reported_without_its_output(
        self, raw: Path, tmp_path: Path, capsys
    ):
        base = _stub_tree(tmp_path / "base")
        rails = base / "backend/knowledge/extraction/id_rails.py"
        rails.write_text(
            "def normalize_payload(entities, relationships):\n"
            "    raise ValueError(f'cannot store {entities}')\n",
            encoding="utf-8",
        )

        status, lines, err = _run(capsys, "--base-root", str(base), "--raw", f"synthetic={raw}")

        assert (status, lines) == (2, [])
        assert "the base half failed with status 1 (ValueError)" in err
        assert [text for text in _ROW_TEXT if text in err] == []

    def test_a_base_root_without_the_tree_is_refused(self, raw: Path, tmp_path: Path, capsys):
        status, lines, err = _run(capsys, "--base-root", str(tmp_path), "--raw", f"synthetic={raw}")

        assert (status, lines) == (2, [])
        assert "holds no backend/knowledge/extraction/id_rails.py" in err

    def test_a_raw_file_with_no_payload_is_refused(self, tmp_path: Path, capsys):
        raw = _write_raw(tmp_path / "empty.jsonl", [], with_empty_row=True)

        status, lines, err = _run(
            capsys, "--base-root", str(_REPO_ROOT), "--raw", f"synthetic={raw}"
        )

        assert (status, lines) == (2, [])
        assert "no row of" in err and "holds a payload" in err

    @pytest.mark.parametrize("value", ["synthetic", "=path", "synthetic="])
    def test_a_raw_argument_names_a_set_and_a_path(self, value: str, capsys):
        status, lines, err = _run(capsys, "--base-root", str(_REPO_ROOT), "--raw", value)

        assert (status, lines) == (2, [])
        assert "--raw takes NAME=PATH" in err

    def test_a_set_named_twice_is_refused(self, raw: Path, capsys):
        status, _, err = _run(
            capsys,
            "--base-root",
            str(_REPO_ROOT),
            "--raw",
            f"synthetic={raw}",
            "--raw",
            f"synthetic={raw}",
        )

        assert status == 2
        assert "names the set synthetic twice" in err
