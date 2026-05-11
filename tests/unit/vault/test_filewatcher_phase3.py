"""Phase 3 additions to VaultFilewatcher tests — Task 11.

Tests the exclusion of MIST.md, CLAUDE.md, and meta/ from the sidecar
reindex path. Uses the same FakeSidecarIndex from test_filewatcher.py
(duplicated here for self-containedness) and drives _do_reindex directly
so no debounce delay is needed.

All async tests are decorated with @pytest.mark.asyncio (asyncio_mode=strict).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import pytest_asyncio

from backend.knowledge.config import FilewatcherConfig
from backend.vault.filewatcher import VaultFilewatcher

# ---------------------------------------------------------------------------
# Fake sidecar index (minimal surface for Task 11)
# ---------------------------------------------------------------------------


class FakeSidecarIndex:
    """Minimal test double for SidecarIndexProtocol.

    Records upsert_file calls for assertion in exclusion tests.
    """

    def __init__(self) -> None:
        self.upsert_file_calls: list[tuple] = []
        self.delete_path_calls: list[str] = []

    def initialize(self) -> None:
        pass

    def close(self) -> None:
        pass

    def upsert_file(
        self,
        path: str,
        content: str,
        mtime: int,
        frontmatter: dict | None = None,
    ) -> int:
        self.upsert_file_calls.append((path, content, mtime, frontmatter))
        return 1

    def delete_path(self, path: str) -> int:
        self.delete_path_calls.append(path)
        return 1

    def query_vector(self, embedding: list[float], k: int = 10) -> list[dict]:
        return []

    def query_fts(self, text: str, k: int = 10) -> list[dict]:
        return []

    def query_hybrid(
        self,
        embedding: list[float],
        text: str,
        k: int = 10,
        rrf_k: int = 60,
    ) -> list[dict]:
        return []

    def chunk_count(self) -> int:
        return 0

    def health_check(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**kwargs) -> FilewatcherConfig:
    """Build a FilewatcherConfig with defaults for testing."""
    defaults = {
        "enabled": True,
        "observer_type": "polling",
        "debounce_ms": 100,
        "staleness_slo_seconds": 5,
        "audit_interval_seconds": 3600,
    }
    defaults.update(kwargs)
    return FilewatcherConfig(**defaults)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def vault_root(tmp_path: Path) -> Path:
    """A temporary vault root directory."""
    root = tmp_path / "vault"
    root.mkdir(parents=True)
    return root


@pytest_asyncio.fixture
async def fake_sidecar() -> FakeSidecarIndex:
    """Fresh FakeSidecarIndex for each test."""
    return FakeSidecarIndex()


@pytest_asyncio.fixture
async def watcher(vault_root: Path, fake_sidecar: FakeSidecarIndex):
    """Started VaultFilewatcher (polling, fast debounce). Stopped on teardown."""
    config = _make_config()
    fw = VaultFilewatcher(config, vault_root, fake_sidecar)
    fw.start(loop=asyncio.get_running_loop())
    yield fw
    fw.stop()


# ---------------------------------------------------------------------------
# TestFileWatcherExclusions
#
# Tests that _do_reindex is a no-op (does not call sidecar.upsert_file) for
# MIST.md, CLAUDE.md, and any path under meta/.
# ---------------------------------------------------------------------------


class TestFilewatcherExclusions:
    @pytest.mark.asyncio
    async def test_reindex_mist_md_does_not_call_upsert(
        self,
        watcher: VaultFilewatcher,
        vault_root: Path,
        fake_sidecar: FakeSidecarIndex,
    ):
        """_do_reindex on MIST.md must not call sidecar.upsert_file."""
        mist_md = vault_root / "MIST.md"
        mist_md.write_text("# Vault conventions\n", encoding="utf-8")

        await watcher._do_reindex(str(mist_md))

        assert (
            fake_sidecar.upsert_file_calls == []
        ), f"Expected no upsert calls for MIST.md, got {fake_sidecar.upsert_file_calls}"

    @pytest.mark.asyncio
    async def test_reindex_claude_md_does_not_call_upsert(
        self,
        watcher: VaultFilewatcher,
        vault_root: Path,
        fake_sidecar: FakeSidecarIndex,
    ):
        """_do_reindex on CLAUDE.md must not call sidecar.upsert_file."""
        claude_md = vault_root / "CLAUDE.md"
        claude_md.write_text("# Claude guide\n", encoding="utf-8")

        await watcher._do_reindex(str(claude_md))

        assert (
            fake_sidecar.upsert_file_calls == []
        ), f"Expected no upsert calls for CLAUDE.md, got {fake_sidecar.upsert_file_calls}"

    @pytest.mark.asyncio
    async def test_reindex_meta_file_does_not_call_upsert(
        self,
        watcher: VaultFilewatcher,
        vault_root: Path,
        fake_sidecar: FakeSidecarIndex,
    ):
        """_do_reindex on a meta/ file must not call sidecar.upsert_file."""
        meta_dir = vault_root / "meta"
        meta_dir.mkdir()
        schema_md = meta_dir / "schema.md"
        schema_md.write_text("# Schema\n", encoding="utf-8")

        await watcher._do_reindex(str(schema_md))

        assert (
            fake_sidecar.upsert_file_calls == []
        ), f"Expected no upsert calls for meta/schema.md, got {fake_sidecar.upsert_file_calls}"
