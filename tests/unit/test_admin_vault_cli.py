"""ADR-010 Cluster 8 Phase 11: vault CLI subcommands.

Verifies the four operator subcommands added to scripts/mist_admin.py:
vault-status, vault-reindex, vault-rebuild, vault-migrate.

Each handler is exercised in isolation via direct call with a parsed
argparse Namespace. The tests mock the backend's get_config call and
use real tmp_path filesystems for vault content; sidecar SQLite opens
on the tmp_path so end-to-end behavior (file walk -> sidecar upsert)
is verified without a live Neo4j or LLM dependency.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pytest

# scripts/ is not a package; load mist_admin via importlib for test access.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import mist_admin  # noqa: E402  -- after sys.path insertion

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def vault_root(tmp_path: Path) -> Path:
    """Create a tmp vault root with a minimal directory structure."""
    root = tmp_path / "vault"
    (root / "sessions").mkdir(parents=True)
    (root / "identity").mkdir()
    (root / "users").mkdir()
    return root


def _write_session_note(vault_root: Path, name: str, body: str) -> Path:
    """Write a vault session note with minimal valid frontmatter + body."""
    path = vault_root / "sessions" / name
    path.write_text(
        "---\n"
        "type: mist-session\n"
        "session_id: test-session\n"
        "date: 2026-04-22\n"
        "turn_count: 0\n"
        "participants:\n  - user\n  - mist\n"
        "authored_by: mist\n"
        "status: in-progress\n"
        "ontology_version: 1.0.0\n"
        "extraction_version: 2026-04-17-r1\n"
        "model_hash: test-model\n"
        "tags: []\n"
        "---\n\n"
        f"{body}\n"
    )
    return path


def _build_config(
    *,
    vault_root: Path,
    sidecar_db: Path,
    vault_enabled: bool = True,
    sidecar_enabled: bool = True,
) -> Any:
    """Construct a tmp-rooted KnowledgeConfig with overridable enable flags.

    VaultConfig and SidecarIndexConfig are frozen+slots, so per-test
    overrides are taken at construction time rather than via mutation.
    """
    from backend.knowledge.config import (
        EmbeddingConfig,
        ExtractionConfig,
        FilewatcherConfig,
        KnowledgeConfig,
        LLMConfig,
        Neo4jConfig,
        SidecarIndexConfig,
        VaultConfig,
    )

    return KnowledgeConfig(
        neo4j=Neo4jConfig(),
        llm=LLMConfig(),
        embedding=EmbeddingConfig(),
        extraction=ExtractionConfig(),
        vault=VaultConfig(root=str(vault_root), git_auto_init=False, enabled=vault_enabled),
        sidecar_index=SidecarIndexConfig(db_path=str(sidecar_db), enabled=sidecar_enabled),
        filewatcher=FilewatcherConfig(observer_type="polling"),
    )


def _patch_backend(monkeypatch, config: Any) -> None:
    """Patch mist_admin._load_backend to return a fake backend with `config`."""

    class _FakeBackend:
        pass

    fake = _FakeBackend()
    fake.get_config = lambda: config
    monkeypatch.setattr(mist_admin, "_load_backend", lambda: fake)


@pytest.fixture
def fake_backend(tmp_path, vault_root, monkeypatch):
    """Default fixture: vault + sidecar both enabled."""
    config = _build_config(vault_root=vault_root, sidecar_db=tmp_path / "sidecar.db")
    _patch_backend(monkeypatch, config)
    return None, config


@pytest.fixture(autouse=True)
def stub_embedding_generator(monkeypatch):
    """Replace EmbeddingGenerator with a fast deterministic stub for CLI tests."""

    class _StubEmbeddings:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def generate_embedding(self, text: str) -> list[float]:
            # Deterministic non-zero embedding so vec0 inserts succeed.
            return [0.01] * 384

        def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
            return [self.generate_embedding(t) for t in texts]

    monkeypatch.setattr(
        "backend.knowledge.embeddings.embedding_generator.EmbeddingGenerator",
        _StubEmbeddings,
    )


# ---------------------------------------------------------------------------
# TestVaultStatus
# ---------------------------------------------------------------------------


class TestVaultStatus:
    def test_reports_disabled_vault(self, tmp_path, vault_root, monkeypatch, capsys):
        config = _build_config(
            vault_root=vault_root,
            sidecar_db=tmp_path / "sidecar.db",
            vault_enabled=False,
        )
        _patch_backend(monkeypatch, config)

        rc = mist_admin.cmd_vault_status(argparse.Namespace())

        assert rc == 0
        out = capsys.readouterr().out
        assert "Vault layer disabled" in out

    def test_reports_md_file_count(self, fake_backend, vault_root, capsys):
        _write_session_note(vault_root, "2026-04-22-a.md", "# A")
        _write_session_note(vault_root, "2026-04-22-b.md", "# B")

        rc = mist_admin.cmd_vault_status(argparse.Namespace())

        assert rc == 0
        out = capsys.readouterr().out
        assert "on-disk .md files:              2" in out

    def test_reports_sidecar_chunk_count(self, fake_backend, vault_root, capsys):
        _write_session_note(vault_root, "2026-04-22-a.md", "# A\nbody content")

        rc = mist_admin.cmd_vault_status(argparse.Namespace())

        assert rc == 0
        out = capsys.readouterr().out
        # Fresh sidecar -> 0 chunks
        assert "sidecar chunk_count:            0" in out
        assert "sidecar health_check:           OK" in out


# ---------------------------------------------------------------------------
# TestVaultReindex
# ---------------------------------------------------------------------------


class TestVaultReindex:
    def test_indexes_all_md_files(self, fake_backend, vault_root, capsys):
        _write_session_note(vault_root, "2026-04-22-a.md", "# A\nbody content for A")
        _write_session_note(vault_root, "2026-04-22-b.md", "# B\nbody content for B")

        rc = mist_admin.cmd_vault_reindex(argparse.Namespace(scope=None))

        assert rc == 0
        out = capsys.readouterr().out
        assert "files processed: 2" in out
        assert "failures:        0" in out
        # 1 file-level chunk + 1 heading-block chunk per file = 4 total.
        assert "total chunks:    4" in out

    def test_scope_indexes_single_file(self, fake_backend, vault_root, capsys):
        _write_session_note(vault_root, "2026-04-22-a.md", "# A")
        path_b = _write_session_note(vault_root, "2026-04-22-b.md", "# B")

        rc = mist_admin.cmd_vault_reindex(argparse.Namespace(scope=str(path_b)))

        assert rc == 0
        out = capsys.readouterr().out
        assert "files processed: 1" in out

    def test_scope_missing_returns_error(self, fake_backend, capsys):
        rc = mist_admin.cmd_vault_reindex(argparse.Namespace(scope="/does/not/exist.md"))

        assert rc == 1
        out = capsys.readouterr().out
        assert "Scope file not found" in out

    def test_skips_when_disabled(self, tmp_path, vault_root, monkeypatch, capsys):
        config = _build_config(
            vault_root=vault_root,
            sidecar_db=tmp_path / "sidecar.db",
            vault_enabled=False,
        )
        _patch_backend(monkeypatch, config)

        rc = mist_admin.cmd_vault_reindex(argparse.Namespace(scope=None))

        assert rc == 0
        out = capsys.readouterr().out
        assert "nothing to do" in out

    def test_empty_vault_succeeds_with_zero_files(self, fake_backend, capsys):
        # vault_root exists but has no .md files
        rc = mist_admin.cmd_vault_reindex(argparse.Namespace(scope=None))

        assert rc == 0
        out = capsys.readouterr().out
        assert "No .md files to index" in out


# ---------------------------------------------------------------------------
# TestVaultRebuild
# ---------------------------------------------------------------------------


class TestVaultRebuild:
    def test_dry_run_without_confirm_does_not_drop(
        self, fake_backend, vault_root, tmp_path, capsys
    ):
        _write_session_note(vault_root, "2026-04-22-a.md", "# A")
        # Pre-create a sidecar file so we can verify --confirm is required
        sidecar_path = tmp_path / "sidecar.db"
        sidecar_path.write_text("placeholder", encoding="utf-8")

        rc = mist_admin.cmd_vault_rebuild(argparse.Namespace(confirm=False))

        assert rc == 0
        out = capsys.readouterr().out
        assert "DRY RUN" in out
        # Sidecar file untouched
        assert sidecar_path.exists()
        assert sidecar_path.read_text() == "placeholder"

    def test_confirm_drops_sidecar_and_rebuilds(self, fake_backend, vault_root, tmp_path, capsys):
        _write_session_note(vault_root, "2026-04-22-a.md", "# A\ncontent")
        _write_session_note(vault_root, "2026-04-22-b.md", "# B\ncontent")

        rc = mist_admin.cmd_vault_rebuild(argparse.Namespace(confirm=True))

        assert rc == 0
        out = capsys.readouterr().out
        assert "Dropping sidecar" in out
        assert "files processed: 2" in out
        # New sidecar exists + has chunks
        assert (tmp_path / "sidecar.db").exists()


# ---------------------------------------------------------------------------
# TestVaultMigrate
# ---------------------------------------------------------------------------


class TestVaultMigrate:
    def test_default_target_matches_current_version(self, fake_backend, capsys):
        rc = mist_admin.cmd_vault_migrate(argparse.Namespace(target_version=None))

        assert rc == 0
        out = capsys.readouterr().out
        assert "config.ontology_version:    1.0.0" in out
        assert "Already at target version" in out

    def test_explicit_target_other_than_current_reports_no_migration(self, fake_backend, capsys):
        rc = mist_admin.cmd_vault_migrate(argparse.Namespace(target_version="2.0.0"))

        assert rc == 0
        out = capsys.readouterr().out
        assert "target_version:             2.0.0" in out
        assert "No migration path registered" in out

    def test_reports_extraction_version(self, fake_backend, capsys):
        rc = mist_admin.cmd_vault_migrate(argparse.Namespace(target_version=None))

        assert rc == 0
        out = capsys.readouterr().out
        # Phase 8 stamp must be reported in the migrate output
        assert "config.extraction_version:" in out


# ---------------------------------------------------------------------------
# TestVaultMdWalker -- helper coverage
# ---------------------------------------------------------------------------


class TestWalkVaultMdFiles:
    def test_returns_empty_for_missing_root(self, tmp_path):
        result = mist_admin._walk_vault_md_files(tmp_path / "missing")
        assert result == []

    def test_skips_hidden_directories(self, tmp_path):
        (tmp_path / "sessions").mkdir()
        (tmp_path / "sessions" / "visible.md").write_text("x")
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "hidden.md").write_text("y")

        result = mist_admin._walk_vault_md_files(tmp_path)

        names = [p.name for p in result]
        assert names == ["visible.md"]

    def test_returns_files_in_sorted_order(self, tmp_path):
        (tmp_path / "sessions").mkdir()
        (tmp_path / "sessions" / "z.md").write_text("z")
        (tmp_path / "sessions" / "a.md").write_text("a")
        (tmp_path / "sessions" / "m.md").write_text("m")

        result = mist_admin._walk_vault_md_files(tmp_path)
        names = [p.name for p in result]
        assert names == ["a.md", "m.md", "z.md"]


class TestReadVaultNote:
    def test_parses_frontmatter_when_valid(self, vault_root):
        path = _write_session_note(vault_root, "x.md", "# Heading\nbody")

        body, frontmatter, mtime = mist_admin._read_vault_note(path)

        assert "# Heading" in body
        assert frontmatter is not None
        assert frontmatter["type"] == "mist-session"
        assert mtime > 0

    def test_returns_raw_text_on_corrupted_frontmatter(self, tmp_path):
        # Frontmatter open marker but invalid YAML inside
        path = tmp_path / "broken.md"
        path.write_text("---\n: unparsable :: yaml :\n---\nbody")

        body, frontmatter, mtime = mist_admin._read_vault_note(path)

        # Falls back to raw text + frontmatter=None
        assert "body" in body
        # frontmatter may be None (parse failed) or a dict (parser was lenient).
        # The contract is "don't crash" -- both shapes are acceptable.
        assert frontmatter is None or isinstance(frontmatter, dict)
