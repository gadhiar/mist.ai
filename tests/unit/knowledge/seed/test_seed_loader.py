import logging
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

    with pytest.raises(SeedSourceError, match="resolved to no keys"):
        load_seed_documents(d)


def test_rejects_well_formed_yaml_that_resolves_to_no_keys(tmp_path: Path):
    """Frontmatter that parses cleanly but has no keys (e.g. comments only) is
    indistinguishable from broken YAML at the `parse_frontmatter` boundary, so it must
    also raise -- but the message must not claim a syntax error that did not happen.
    """
    comment_only = "---\n# just a comment, no keys\n---\n\nbody text\n"
    d = _write(tmp_path, "empty-frontmatter.md", comment_only)

    with pytest.raises(SeedSourceError, match="resolved to no keys"):
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


@pytest.mark.parametrize(
    "predicate_literal",
    [
        pytest.param('""', id="empty-string"),
        pytest.param('"   "', id="whitespace-only"),
    ],
)
def test_rejects_empty_predicate_through_full_load_path(tmp_path: Path, predicate_literal: str):
    """`SeedFact._non_empty` must actually fire on real load input.

    `test_rejects_missing_predicate` deletes the `predicate:` key entirely, which only
    exercises Pydantic's required-field check -- it never reaches the custom validator.
    This drives an empty/whitespace-only value through `load_seed_documents` so the
    validator itself is what raises.
    """
    bad = _DOC.replace("predicate: WORKS_AT", f"predicate: {predicate_literal}")
    d = _write(tmp_path, "user.md", bad)

    with pytest.raises(SeedSourceError, match="non-empty"):
        load_seed_documents(d)


def test_rejects_unknown_key_in_fact(tmp_path: Path):
    """A typo'd optional key (`valid_form` for `valid_from`) must not be silently
    dropped -- SeedFact sets `extra='forbid'` so it surfaces as a load-time error
    naming the offending key, rather than loading with `valid_from=None`.
    """
    bad = _DOC.replace(
        "{subject: user, predicate: WORKS_AT, object: Slalom}",
        '{subject: user, predicate: WORKS_AT, object: Slalom, valid_form: "2020-01-01"}',
    )
    d = _write(tmp_path, "user.md", bad)

    with pytest.raises(SeedSourceError, match="valid_form"):
        load_seed_documents(d)


def test_logs_info_for_fact_less_seed_document(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """Prose-only seed content (no `facts:` key) is legitimate and must still load, but
    is logged at INFO so a document that *should* have facts and lost them to a typo in
    the `facts:` key itself is still discoverable at load time.
    """
    prose_only = "---\ntype: mist-seed\nseed_version: profile-v1\n---\n\nJust prose, no facts.\n"
    d = _write(tmp_path, "identity.md", prose_only)

    with caplog.at_level(logging.INFO, logger="backend.knowledge.seed.loader"):
        docs = load_seed_documents(d)

    assert len(docs) == 1
    assert docs[0].facts == []
    assert any(
        "identity.md" in record.getMessage() and "no facts" in record.getMessage()
        for record in caplog.records
    )
