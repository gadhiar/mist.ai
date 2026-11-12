"""Pin the selection set's committed agreement figures to the module that computes them.

`agreement.json` was read before any relation was adjudicated, so a reader can trust it only if
the same inputs give the same file again. These tests recompute it from the two blind passes
under the pack whose rails hash the file records, and compare. Each comparison is reduced to a
bool first, so a failure prints no row of the selection set.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.knowledge.extraction.rails import run_rails  # noqa: E402
from backend.knowledge.ontologies.pack import load_pack  # noqa: E402
from scripts.eval_harness import selection_agreement as selection_agreement_module  # noqa: E402
from scripts.eval_harness.selection_agreement import (  # noqa: E402
    main,
    read_rows,
    render,
    selection_agreement,
)

_LABELLING = _REPO_ROOT / "data" / "gold" / "extraction" / "labelling" / "selection-2026-11"
_PASS_A = _LABELLING / "pass-a.jsonl"
_PASS_B = _LABELLING / "pass-b.jsonl"
_COMMITTED = _LABELLING / "agreement.json"
_SELECTION = _REPO_ROOT / "data" / "gold" / "extraction" / "selection-2026-11.jsonl"

PACK = load_pack("backend.knowledge.ontologies.packs.v1_4_0")


def test_no_module_level_import_loads_the_pack_or_the_rails():
    """Only `main` resolves the pack, so a caller's pack and rails are the ones measured."""
    tree = ast.parse(Path(selection_agreement_module.__file__).read_text(encoding="utf-8"))
    modules = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            modules |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            modules.add(node.module or "")

    fenced = {m for m in modules if m.startswith("backend.")}

    assert not fenced


def test_the_pack_is_the_one_the_committed_figures_record():
    recorded = json.loads(_COMMITTED.read_text(encoding="utf-8"))

    same_pack = (PACK.pack_rails_hash, PACK.version) == (
        recorded["pack_rails_hash"],
        recorded["pack_version"],
    )

    assert same_pack


def test_the_module_reproduces_the_committed_agreement():
    report = selection_agreement(
        read_rows(_PASS_A),
        read_rows(_PASS_B),
        read_rows(_SELECTION),
        pack=PACK,
        run_rails=run_rails,
    )

    same = render(report) == _COMMITTED.read_text(encoding="utf-8")

    assert same


def test_the_cli_writes_the_committed_file_byte_for_byte(tmp_path):
    out = tmp_path / "agreement.json"

    main(
        [
            "--pass-a",
            str(_PASS_A),
            "--pass-b",
            str(_PASS_B),
            "--selection",
            str(_SELECTION),
            "--recorded",
            str(_COMMITTED),
            "--out",
            str(out),
        ]
    )
    same = out.read_bytes() == _COMMITTED.read_bytes()

    assert same
