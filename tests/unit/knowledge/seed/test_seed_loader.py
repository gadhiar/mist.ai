import logging
from pathlib import Path

import pytest

from backend.errors import SeedSourceError
from backend.knowledge.seed.loader import load_seed_documents
from backend.knowledge.storage.partitions import ENTITY_LABEL, SELF_MODEL_LABEL

# R1.4 Task 11 (ADDENDUM): every fact's subject/object must have a matching
# `nodes:` entry (referential integrity), so this fixture defines nodes for
# both facts. `slalom`/`python` use real ontology node types; the old
# `HAS_ROLE` fact (object "Consultant, Software Engineering") is gone -- a
# free-text value was never node-reference-shaped, and referential integrity
# would have correctly rejected it as an undefined node id.
_DOC = """---
type: mist-seed
seed_version: profile-v1
nodes:
  - {id: user, type: User, display_name: "Raj Gadhia"}
  - {id: slalom, type: Organization, display_name: Slalom}
  - {id: python, type: Technology, display_name: Python}
facts:
  - {subject: user, predicate: WORKS_AT, object: slalom}
  - {subject: user, predicate: USES, object: python}
---

Raj works at Slalom as a Consultant, Software Engineering, and uses Python.
"""

# Prose-only variant (no nodes, no facts) for tests that only care about
# document-level mechanics (file ordering, seed_version agreement) and would
# otherwise collide with the new duplicate-node-id check if the same
# fact/node content were loaded from two files at once.
_PROSE_ONLY_DOC = """---
type: mist-seed
seed_version: profile-v1
---

Just prose, no facts, no nodes.
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
    assert docs[0].facts[0].object == "slalom"
    assert "Raj works at Slalom" in docs[0].body


def test_rejects_mismatched_seed_versions(tmp_path: Path):
    """One global seed_version (O10) -- two files disagreeing is a bug, not a merge.

    Uses the prose-only fixture: the version-mismatch check runs before the
    new node/referential-integrity checks (see loader.py's ordering), so
    this would pass either way, but prose-only keeps the test's intent
    (version disagreement, nothing else) unambiguous.
    """
    d = _write(tmp_path, "a.md", _PROSE_ONLY_DOC)
    _write(tmp_path, "b.md", _PROSE_ONLY_DOC.replace("profile-v1", "profile-v2"))

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
    """Application order must be deterministic -- sorted by filename, not directory order.

    Uses the prose-only fixture in both files: loading the same node/fact
    content from two files at once would (correctly) trip the new
    duplicate-node-id check, which is not what this test is about.
    """
    d = _write(tmp_path, "b-second.md", _PROSE_ONLY_DOC)
    _write(tmp_path, "a-first.md", _PROSE_ONLY_DOC)

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
        "{subject: user, predicate: WORKS_AT, object: slalom}",
        '{subject: user, predicate: WORKS_AT, object: slalom, valid_form: "2020-01-01"}',
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


def test_defaults_partition_to_the_entity_label_when_absent(tmp_path: Path):
    """`_DOC` carries no `partition:` key -- most seed documents are ordinary
    user/world facts, so the entity partition is the correct default rather than
    requiring every document to declare it.
    """
    d = _write(tmp_path, "user.md", _DOC)

    docs = load_seed_documents(d)

    assert docs[0].partition == ENTITY_LABEL


def test_loads_partition_field_from_frontmatter(tmp_path: Path):
    """A document whose content belongs in the self-model partition (e.g.
    seed/mist.md) must declare it, and the loader must read it through rather
    than silently defaulting -- this is R1.4 Task 4's rework: apply_seed_documents
    previously hardcoded every node to the entity partition regardless of what
    the document declared.
    """
    self_model_doc = _DOC.replace(
        "seed_version: profile-v1\n", f"seed_version: profile-v1\npartition: {SELF_MODEL_LABEL}\n"
    )
    d = _write(tmp_path, "mist.md", self_model_doc)

    docs = load_seed_documents(d)

    assert docs[0].partition == SELF_MODEL_LABEL


def test_rejects_unknown_partition_value(tmp_path: Path):
    """A typo'd partition (e.g. `__SelfModle__`) must fail loudly at load time as
    a `SeedSourceError`, not surface as a raw pydantic `ValidationError` -- and
    must never silently fall through to the entity-partition default, which
    would reproduce the exact bug this field exists to prevent.
    """
    bad = _DOC.replace("seed_version: profile-v1\n", "seed_version: profile-v1\npartition: bogus\n")
    d = _write(tmp_path, "user.md", bad)

    with pytest.raises(SeedSourceError, match="partition"):
        load_seed_documents(d)


# ---------------------------------------------------------------------------
# R1.4 Task 11 (ADDENDUM): node definitions
#
# `SeedNode` + three whole-corpus checks (_validate_node_types,
# _validate_unique_node_ids, _validate_referential_integrity). Referential
# integrity is the headline: it is the exact shape of Task 10's live defect
# (a fact referencing a node the seed source never defines), so it is tested
# on input where it actually fires, not just on the happy path.
# ---------------------------------------------------------------------------


def test_loads_node_definitions(tmp_path: Path):
    d = _write(tmp_path, "user.md", _DOC)

    docs = load_seed_documents(d)

    assert len(docs[0].nodes) == 3
    ids = {n.id for n in docs[0].nodes}
    assert ids == {"user", "slalom", "python"}
    user_node = next(n for n in docs[0].nodes if n.id == "user")
    assert user_node.type == "User"


def test_node_preserves_arbitrary_extra_properties(tmp_path: Path):
    """SeedNode's extra="allow" must actually flow properties through to the
    loaded model, not just accept them at the YAML level. Uses `display_name`
    (already in `_DOC`) plus a second, type-specific-shaped property
    (`title`) to prove this is not special-cased to one known field name.
    """
    with_extra = _DOC.replace(
        '{id: user, type: User, display_name: "Raj Gadhia"}',
        '{id: user, type: User, display_name: "Raj Gadhia", title: "Software Engineer"}',
    )
    d = _write(tmp_path, "user.md", with_extra)

    docs = load_seed_documents(d)

    user_node = next(n for n in docs[0].nodes if n.id == "user")
    assert user_node.display_name == "Raj Gadhia"
    assert user_node.title == "Software Engineer"


def test_defaults_nodes_to_empty_list_when_absent(tmp_path: Path):
    d = _write(tmp_path, "identity.md", _PROSE_ONLY_DOC)

    docs = load_seed_documents(d)

    assert docs[0].nodes == []


def test_rejects_unknown_node_type(tmp_path: Path):
    """A node `type` outside the ontology's node types must fail loudly at
    load time (Task 11's own validation pass, distinct from -- and in
    addition to -- Task 12's applier-side check at the Cypher boundary).
    """
    bad = _DOC.replace("type: Organization", "type: Compnay")
    d = _write(tmp_path, "user.md", bad)

    with pytest.raises(SeedSourceError, match="Compnay"):
        load_seed_documents(d)


def test_rejects_duplicate_node_id_within_one_document(tmp_path: Path):
    dupe = _DOC.replace(
        "  - {id: python, type: Technology, display_name: Python}",
        "  - {id: python, type: Technology, display_name: Python}\n"
        "  - {id: python, type: Technology, display_name: Python2}",
    )
    d = _write(tmp_path, "user.md", dupe)

    with pytest.raises(SeedSourceError, match="python"):
        load_seed_documents(d)


def test_rejects_duplicate_node_id_across_documents(tmp_path: Path):
    """The same node id defined in two different files is an authoring
    conflict -- the same id cannot mean two different node definitions.
    """
    doc_a = _DOC
    # doc_b references only "user" (already defined in doc_a) so it does not
    # ALSO trip referential integrity for an unrelated reason -- this test
    # isolates the duplicate-id check specifically.
    doc_b = """---
type: mist-seed
seed_version: profile-v1
nodes:
  - {id: user, type: User, display_name: "Raj Gadhia (duplicate)"}
facts: []
---

Duplicate user node, different file.
"""
    _write(tmp_path, "a.md", doc_a)
    d = _write(tmp_path, "b.md", doc_b)

    with pytest.raises(SeedSourceError, match="user"):
        load_seed_documents(d)


def test_rejects_fact_referencing_undefined_node(tmp_path: Path):
    """Referential integrity is the headline of this task: a fact whose
    object has no matching `nodes:` entry is the EXACT shape of the R1.4
    Task 10 live defect (a fact referencing an id the seed source never
    defines, which used to write to the graph silently with no type label
    and no descriptive properties). Must raise, naming the undefined id.
    """
    undefined_ref = _DOC.replace(
        "  - {subject: user, predicate: USES, object: python}",
        "  - {subject: user, predicate: USES, object: rust}",
    )
    d = _write(tmp_path, "user.md", undefined_ref)

    with pytest.raises(SeedSourceError, match="rust"):
        load_seed_documents(d)


def test_rejects_fact_whose_subject_is_undefined(tmp_path: Path):
    """Referential integrity applies to BOTH subject and object, not just
    object -- a fact's subject is exactly as writable-with-no-definition as
    its object under the old (defective) applier.
    """
    undefined_subject = _DOC.replace(
        "  - {subject: user, predicate: WORKS_AT, object: slalom}",
        "  - {subject: ghost, predicate: WORKS_AT, object: slalom}",
    )
    d = _write(tmp_path, "user.md", undefined_subject)

    with pytest.raises(SeedSourceError, match="ghost"):
        load_seed_documents(d)


def test_referential_integrity_resolves_across_documents(tmp_path: Path):
    """A fact in one document may reference a node defined in ANOTHER
    document -- referential integrity must check the whole corpus, not
    validate each document in isolation.
    """
    node_doc = """---
type: mist-seed
seed_version: profile-v1
nodes:
  - {id: neo4j, type: Technology, display_name: Neo4j}
facts: []
---

Neo4j is a graph database.
"""
    fact_doc = """---
type: mist-seed
seed_version: profile-v1
nodes:
  - {id: user, type: User, display_name: "Raj Gadhia"}
facts:
  - {subject: user, predicate: USES, object: neo4j}
---

Raj uses Neo4j.
"""
    _write(tmp_path, "a-technologies.md", node_doc)
    d = _write(tmp_path, "b-user.md", fact_doc)

    docs = load_seed_documents(d)

    assert len(docs) == 2
