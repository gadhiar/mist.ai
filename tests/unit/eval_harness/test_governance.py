"""The reading's identity: `scorer_sha`, `gold_version` and the pack hashes (plan v14 T50).

The sort and the bytes `scorer_sha` hashes are pinned on a laid-out tree, and the
repository's own sha is checked against the same rule computed here.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.eval_harness import governance

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _tree(root: Path, files: dict[str, bytes]) -> Path:
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return root


_FILES = {
    "scripts/eval_harness/z.py": b"z\n",
    "scripts/eval_harness/ceiling/relay.py": b"relay\n",
    "scripts/eval_harness/a.py": b"a\n",
    "scripts/eval_harness/README.md": b"not python\n",
    "scripts/eval_harness/tests/speed.yaml": b"not python\n",
    "backend/knowledge/extraction/rails.py": b"rails\n",
    "backend/knowledge/extraction/id_rails.py": b"id rails\n",
    "backend/knowledge/extraction/normalizer.py": b"not the scorer's\n",
}


def test_scorer_files_are_every_python_file_of_the_harness_and_the_two_rails_by_path(
    tmp_path: Path,
) -> None:
    root = _tree(tmp_path, _FILES)
    assert [p.relative_to(root).as_posix() for p in governance.scorer_files(root)] == [
        "backend/knowledge/extraction/id_rails.py",
        "backend/knowledge/extraction/rails.py",
        "scripts/eval_harness/a.py",
        "scripts/eval_harness/ceiling/relay.py",
        "scripts/eval_harness/z.py",
    ]


def test_scorer_sha_hashes_the_contents_in_path_order_with_nothing_between(
    tmp_path: Path,
) -> None:
    root = _tree(tmp_path, _FILES)
    expected = hashlib.sha256(b"id rails\nrails\na\nrelay\nz\n").hexdigest()
    assert governance.scorer_sha(root) == expected


def test_a_change_to_a_rail_or_a_harness_file_moves_the_sha(tmp_path: Path) -> None:
    root = _tree(tmp_path, _FILES)
    before = governance.scorer_sha(root)
    (root / "backend/knowledge/extraction/rails.py").write_bytes(b"rails changed\n")
    after_rails = governance.scorer_sha(root)
    (root / "scripts/eval_harness/ceiling/relay.py").write_bytes(b"relay changed\n")
    assert len({before, after_rails, governance.scorer_sha(root)}) == 3


def test_a_file_outside_the_scorer_does_not_move_the_sha(tmp_path: Path) -> None:
    root = _tree(tmp_path, _FILES)
    before = governance.scorer_sha(root)
    (root / "backend/knowledge/extraction/normalizer.py").write_bytes(b"changed\n")
    (root / "scripts/eval_harness/README.md").write_bytes(b"changed\n")
    assert governance.scorer_sha(root) == before


def test_the_repositorys_sha_is_the_rule_over_its_own_files() -> None:
    paths = sorted(
        list((_REPO_ROOT / "scripts" / "eval_harness").rglob("*.py"))
        + [
            _REPO_ROOT / "backend/knowledge/extraction/rails.py",
            _REPO_ROOT / "backend/knowledge/extraction/id_rails.py",
        ],
        key=lambda p: p.relative_to(_REPO_ROOT).as_posix(),
    )
    assert _REPO_ROOT / "scripts/eval_harness/ceiling/relay.py" in paths
    expected = hashlib.sha256(b"".join(p.read_bytes() for p in paths)).hexdigest()
    assert governance.scorer_sha(_REPO_ROOT) == expected


def test_gold_version_names_the_gold_the_guide_beside_it_and_the_pack(tmp_path: Path) -> None:
    root = _tree(tmp_path / "repo", {"data/gold/extraction/LABELLING.md": b"repo guide\n"})
    gold = _tree(tmp_path / "set", {"gold.jsonl": b"{}\n", "LABELLING.md": b"set guide\n"})
    gold_sha = hashlib.sha256(b"{}\n").hexdigest()[:12]
    guide_sha = hashlib.sha256(b"set guide\n").hexdigest()[:12]
    assert governance.gold_version(gold / "gold.jsonl", root, "1.4.0") == (
        f"gold-{gold_sha}.labelling-{guide_sha}.pack-1.4.0"
    )


def test_gold_version_falls_back_to_the_repositorys_guide(tmp_path: Path) -> None:
    root = _tree(tmp_path / "repo", {"data/gold/extraction/LABELLING.md": b"repo guide\n"})
    gold = _tree(tmp_path / "set", {"gold.jsonl": b"{}\n"}) / "gold.jsonl"
    guide_sha = hashlib.sha256(b"repo guide\n").hexdigest()[:12]
    assert f".labelling-{guide_sha}.pack-none" in governance.gold_version(gold, root, None)


def test_a_change_to_the_guide_moves_the_gold_version(tmp_path: Path) -> None:
    root = _tree(tmp_path, {"data/gold/extraction/LABELLING.md": b"one\n", "g.jsonl": b"{}\n"})
    before = governance.gold_version(root / "g.jsonl", root, "1.4.0")
    (root / "data/gold/extraction/LABELLING.md").write_bytes(b"two\n")
    assert governance.gold_version(root / "g.jsonl", root, "1.4.0") != before


_IDENTITY = {
    "scorer_sha": "a" * 64,
    "gold_version": "gold-x.labelling-y.pack-1.4.0",
    "pack_prompt_hash": "b" * 64,
    "pack_rails_hash": "c" * 64,
}


def test_a_summary_with_all_four_fields_is_a_reading() -> None:
    governance.require_reading({"rel_recall": 0.5, **_IDENTITY})


@pytest.mark.parametrize("field", governance.READING_FIELDS)
def test_a_summary_without_one_field_is_refused_and_the_field_named(field: str) -> None:
    summary = {**_IDENTITY, field: None}
    with pytest.raises(governance.NotAReadingError, match=field):
        governance.require_reading(summary)
    del summary[field]
    with pytest.raises(governance.NotAReadingError, match=field):
        governance.require_reading(summary)


def test_the_identity_text_names_every_field_and_none_for_an_absent_one() -> None:
    assert governance.identity_text({**_IDENTITY, "pack_rails_hash": None}) == (
        f"scorer_sha={'a' * 64} gold_version=gold-x.labelling-y.pack-1.4.0 "
        f"pack_prompt_hash={'b' * 64} pack_rails_hash=none"
    )
