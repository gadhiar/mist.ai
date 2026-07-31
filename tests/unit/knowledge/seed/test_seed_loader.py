from pathlib import Path

import pytest

from backend.errors import SeedSourceError
from backend.knowledge.seed.loader import load_seed_documents

_DOC = """---
type: mist-seed
seed_version: profile-v1
facts:
  - {subject: user, predicate: WORKS_AT, object: Slalom}
  - {subject: user, predicate: HAS_ROLE, object: "Consultant, Software Engineering"}
---

Raj works at Slalom as a Consultant, Software Engineering.
"""


def _write(tmp_path: Path, name: str, text: str) -> Path:
    d = tmp_path / "seed"
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(text, encoding="utf-8")
    return d


def test_loads_facts_and_body(tmp_path: Path):
    d = _write(tmp_path, "user.md", _DOC)

    docs = load_seed_documents(d)

    assert len(docs) == 1
    assert docs[0].seed_version == "profile-v1"
    assert len(docs[0].facts) == 2
    assert docs[0].facts[0].subject == "user"
    assert docs[0].facts[0].predicate == "WORKS_AT"
    assert docs[0].facts[0].object == "Slalom"
    assert "Raj works at Slalom" in docs[0].body


def test_rejects_mismatched_seed_versions(tmp_path: Path):
    """One global seed_version (O10) -- two files disagreeing is a bug, not a merge."""
    d = _write(tmp_path, "a.md", _DOC)
    _write(tmp_path, "b.md", _DOC.replace("profile-v1", "profile-v2"))

    with pytest.raises(SeedSourceError, match="seed_version"):
        load_seed_documents(d)


def test_rejects_missing_predicate(tmp_path: Path):
    bad = _DOC.replace("predicate: WORKS_AT, ", "")
    d = _write(tmp_path, "user.md", bad)

    with pytest.raises(SeedSourceError):
        load_seed_documents(d)


def test_ignores_non_markdown(tmp_path: Path):
    d = _write(tmp_path, "user.md", _DOC)
    (d / "notes.txt").write_text("not seed", encoding="utf-8")

    assert len(load_seed_documents(d)) == 1


def test_empty_dir_raises(tmp_path: Path):
    d = tmp_path / "seed"
    d.mkdir()

    with pytest.raises(SeedSourceError, match="no seed documents"):
        load_seed_documents(d)


def test_sorts_documents_by_filename(tmp_path: Path):
    """Application order must be deterministic -- sorted by filename, not directory order."""
    d = _write(tmp_path, "b-second.md", _DOC)
    _write(tmp_path, "a-first.md", _DOC)

    docs = load_seed_documents(d)

    assert [doc.source_path.name for doc in docs] == ["a-first.md", "b-second.md"]


def test_rejects_malformed_yaml_frontmatter(tmp_path: Path):
    """`parse_frontmatter` swallows yaml.YAMLError and returns `{}` -- indistinguishable
    from a file with no frontmatter at all. A `.md` file in a dedicated seed directory
    that opens with `---` but fails to parse must raise, not vanish as "not a seed doc".
    """
    malformed = "---\ntype: mist-seed\nfacts: [unterminated\n---\n\nbody text\n"
    d = _write(tmp_path, "broken.md", malformed)

    with pytest.raises(SeedSourceError, match="failed to parse"):
        load_seed_documents(d)


def test_skips_markdown_file_with_different_frontmatter_type(tmp_path: Path):
    """A `.md` file with well-formed frontmatter of a different `type` is a legitimate
    non-seed document (e.g. a stray note) and must be skipped, not raise.
    """
    d = _write(tmp_path, "user.md", _DOC)
    _write(tmp_path, "readme.md", "---\ntype: mist-session\ntitle: not a seed doc\n---\n\ntext\n")

    docs = load_seed_documents(d)

    assert len(docs) == 1
    assert docs[0].source_path.name == "user.md"
