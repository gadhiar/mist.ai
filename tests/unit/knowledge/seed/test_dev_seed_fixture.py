"""The synthetic dev instance fixture, `tests/fixtures/dev-instance/`.

Two consumers read it. `docker-compose.dev.yml`'s `mist-dev-seed` copies `vault/` into a dev
stack's state volume and seeds the graph from `seed/`; `scripts/daily/instance.py init --seed`
copies both into a rehearsal instance root. The layout test pins the names the second one
checks (`_check_seed`) without importing it, so the fixture cannot drift from either reader by
a rename. The rest pins that the seed loads and passes the seed gates that need no graph, and
that every vault note parses as the note type its folder implies.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from backend.knowledge.seed.applier import _assign_node_partitions, _validate_predicates
from backend.knowledge.seed.gates import check_containment, check_negation_proximity
from backend.knowledge.seed.loader import load_seed_documents
from backend.knowledge.storage.partitions import (
    ENTITY_LABEL,
    SELF_MODEL_LABEL,
    USER_ENTITY_ID,
)
from backend.vault.models import (
    MistIdentityFrontmatter,
    MistSessionFrontmatter,
    MistUserFrontmatter,
    parse_frontmatter,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "dev-instance"
SEED_DIR = FIXTURE_DIR / "seed"
VAULT_DIR = FIXTURE_DIR / "vault"

EXPECTED_FILES = {
    "README.md",
    "seed/mist.md",
    "seed/dev-user.md",
    "vault/MIST.md",
    "vault/identity/mist.md",
    "vault/users/dev-user.md",
    "vault/sessions/2026-01-01-dev-fixture-morning.md",
    "vault/sessions/2026-01-02-dev-fixture-evening.md",
}
# `instance.py init` writes these into the instance's mist-memory itself
# (`grep -n '"seed", LIVE_INSTANCE_MARKER' scripts/daily/instance.py`).
NAMES_INIT_WRITES = ("seed", ".mist-live-instance")
DEV_USER_ID = "dev-user"
DEV_USER_NAME = "Dev User"
MIST_IDENTITY_ID = "mist-identity"
# Node types and predicates that would put a real person, place, employer or date in the seed.
REAL_WORLD_NODE_TYPES = {"Person", "Organization", "Location", "Date"}
REAL_WORLD_PREDICATES = {"WORKS_AT", "MEMBER_OF", "KNOWS_PERSON", "OCCURRED_ON"}
SESSION_STEM = re.compile(r"^(\d{4}-\d{2}-\d{2})-([a-z0-9-]+)$")


@pytest.fixture(scope="module")
def documents():
    return load_seed_documents(SEED_DIR)


class TestLayout:
    def test_both_directories_the_rehearsal_init_needs(self):
        assert (FIXTURE_DIR / "vault").is_dir()
        assert (FIXTURE_DIR / "seed").is_dir()

    def test_vault_holds_nothing_init_writes_itself(self):
        names = {entry.name for entry in VAULT_DIR.iterdir()}
        assert not names & set(NAMES_INIT_WRITES), names

    def test_exactly_the_planned_files(self):
        found = {
            path.relative_to(FIXTURE_DIR).as_posix()
            for path in FIXTURE_DIR.rglob("*")
            if path.is_file()
        }
        assert found == EXPECTED_FILES


class TestSeed:
    def test_one_shared_seed_version(self, documents):
        assert len(documents) == 2
        assert len({doc.seed_version for doc in documents}) == 1

    def test_exactly_one_self_model_document(self, documents):
        self_model = [doc for doc in documents if doc.partition == SELF_MODEL_LABEL]
        assert len(self_model) == 1
        assert self_model[0].source_path.name == "mist.md"
        identities = [node for node in self_model[0].nodes if node.type == "MistIdentity"]
        assert [node.id for node in identities] == [MIST_IDENTITY_ID]

    def test_the_user_document_describes_the_dev_user(self, documents):
        (user_doc,) = (doc for doc in documents if doc.partition == ENTITY_LABEL)
        users = [node for node in user_doc.nodes if node.type == "User"]
        assert [node.id for node in users] == [USER_ENTITY_ID]
        assert getattr(users[0], "display_name", None) == DEV_USER_NAME

    def test_containment_gate_passes(self, documents):
        result = check_containment(documents)
        assert result.passed, result.failures

    def test_negation_proximity_gate_passes(self, documents):
        result = check_negation_proximity(documents)
        assert result.passed, result.failures

    def test_applier_accepts_the_predicates_and_partitions(self, documents):
        _validate_predicates(documents)
        partitions = _assign_node_partitions(documents)
        assert partitions[MIST_IDENTITY_ID] == SELF_MODEL_LABEL
        assert partitions[USER_ENTITY_ID] == ENTITY_LABEL

    def test_names_no_real_person_place_employer_or_date(self, documents):
        node_types = {node.type for doc in documents for node in doc.nodes}
        predicates = {fact.predicate for doc in documents for fact in doc.facts}
        assert not node_types & REAL_WORLD_NODE_TYPES, node_types
        assert not predicates & REAL_WORLD_PREDICATES, predicates
        for doc in documents:
            for fact in doc.facts:
                assert fact.valid_from is None and fact.valid_to is None, fact


def _frontmatter(path: Path) -> dict:
    frontmatter, _body = parse_frontmatter(path.read_text(encoding="utf-8"))
    assert frontmatter, f"{path} has no frontmatter"
    return frontmatter


class TestVault:
    def test_conventions_note_is_plain_markdown(self):
        text = (VAULT_DIR / "MIST.md").read_text(encoding="utf-8")
        assert not text.startswith("---")
        assert text.startswith("# ")

    def test_identity_note_parses(self):
        MistIdentityFrontmatter(**_frontmatter(VAULT_DIR / "identity" / "mist.md"))

    def test_user_note_parses_and_names_the_dev_user(self):
        path = VAULT_DIR / "users" / f"{DEV_USER_ID}.md"
        frontmatter = MistUserFrontmatter(**_frontmatter(path))
        assert frontmatter.user_id == DEV_USER_ID
        _fm, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        assert body.lstrip().startswith(f"# {DEV_USER_NAME}\n")

    @pytest.mark.parametrize(
        "name",
        sorted(p.name for p in (VAULT_DIR / "sessions").glob("*.md")),
    )
    def test_session_note_parses_with_the_writer_id_and_date(self, name):
        match = SESSION_STEM.fullmatch(Path(name).stem)
        assert match, name
        frontmatter = MistSessionFrontmatter(**_frontmatter(VAULT_DIR / "sessions" / name))
        assert frontmatter.date == match.group(1)
        assert frontmatter.session_id == match.group(2)
        assert frontmatter.status == "completed"
        assert DEV_USER_ID in frontmatter.related_entities
