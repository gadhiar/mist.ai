"""Tests for `VaultWriter.upsert_user_snapshot`.

The C-pattern machine writeback persists a graph-derived user snapshot to a
SEPARATE derived file `users/<user_id>-graph-snapshot.md`, decoupled from the
hand-curated `users/<user_id>.md`. Unlike `upsert_user`, the snapshot writer:

- targets the `<user_id>-graph-snapshot` filename stem (the `user_id`
  frontmatter field stays the real user_id),
- ALWAYS overwrites the body (it is a machine-owned derived cache, NOT
  user-editable), so it does NOT apply the ADR-010 Invariant-5 guard that
  protects `authored_by: user` / `user-edit` bodies,
- keeps `mist-user` frontmatter with `authored_by: mist`.

Uses the real writer against a tmp root, mirroring TestUpsertUser in
test_writer.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio

from backend.knowledge.config import VaultConfig
from backend.vault.models import parse_frontmatter
from backend.vault.writer import VaultWriter


def _make_config(tmp_path: Path, **kwargs) -> VaultConfig:
    defaults = {
        "enabled": True,
        "root": str(tmp_path / "vault"),
        "default_user_id": "raj",
        "git_auto_init": False,
        "session_soft_cap_turns": 20,
        "session_soft_cap_tokens": 6000,
        "append_sentinel": "<!-- MIST_APPEND_HERE -->",
        "writer_queue_max_depth": 100,
    }
    defaults.update(kwargs)
    return VaultConfig(**defaults)


@pytest_asyncio.fixture
async def vault_writer(tmp_path: Path):
    """Yield a started VaultWriter; stop it on teardown."""
    config = _make_config(tmp_path)
    writer = VaultWriter(config)
    await writer.start()
    yield writer
    await writer.stop()


class TestUpsertUserSnapshot:
    @pytest.mark.asyncio
    async def test_writes_to_graph_snapshot_stem_not_user_md(
        self, vault_writer: VaultWriter, tmp_path: Path
    ):
        path_str = await vault_writer.upsert_user_snapshot(
            user_id="user", body_markdown="## Facts\n- Uses Python\n"
        )

        path = Path(path_str)
        users_dir = tmp_path / "vault" / "users"

        # Targets the derived snapshot stem, NOT the curated user.md.
        assert path == users_dir / "user-graph-snapshot.md"
        assert path.exists()
        assert not (users_dir / "user.md").exists()

    @pytest.mark.asyncio
    async def test_frontmatter_is_mist_user_with_real_user_id_and_mist_authorship(
        self, vault_writer: VaultWriter, tmp_path: Path
    ):
        path_str = await vault_writer.upsert_user_snapshot(
            user_id="user", body_markdown="## Facts\n- Uses Python\n"
        )

        fm_dict, body = parse_frontmatter(Path(path_str).read_text(encoding="utf-8"))

        assert fm_dict["type"] == "mist-user"
        # The frontmatter user_id is the REAL user_id, decoupled from the
        # filename stem.
        assert fm_dict["user_id"] == "user"
        assert fm_dict["authored_by"] == "mist"
        assert "Uses Python" in body

    @pytest.mark.asyncio
    async def test_overwrites_existing_snapshot_body(
        self, vault_writer: VaultWriter, tmp_path: Path
    ):
        await vault_writer.upsert_user_snapshot(
            user_id="user", body_markdown="Old snapshot body.\n"
        )
        await vault_writer.upsert_user_snapshot(
            user_id="user", body_markdown="New snapshot body.\n"
        )

        snapshot = tmp_path / "vault" / "users" / "user-graph-snapshot.md"
        content = snapshot.read_text(encoding="utf-8")
        assert "New snapshot body." in content
        assert "Old snapshot body." not in content

    @pytest.mark.asyncio
    async def test_overwrites_even_when_existing_snapshot_authored_by_user(
        self, vault_writer: VaultWriter, tmp_path: Path
    ):
        """The snapshot is machine-owned: the Invariant-5 guard does NOT apply.

        Even if a snapshot file somehow carries authored_by=user (e.g. a stray
        manual edit of the derived artifact), the writer must overwrite it --
        it is a derived cache, not the user-authoritative profile.
        """
        snapshot = tmp_path / "vault" / "users" / "user-graph-snapshot.md"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_text(
            "---\ntype: mist-user\nuser_id: user\nauthored_by: user\n"
            "last_updated: 2026-04-01\nrelated_sessions: []\ntags: []\n---\n\n"
            "Stale user-flagged snapshot body.\n",
            encoding="utf-8",
        )

        await vault_writer.upsert_user_snapshot(
            user_id="user", body_markdown="Fresh machine-rendered snapshot."
        )

        content = snapshot.read_text(encoding="utf-8")
        assert "Fresh machine-rendered snapshot." in content
        assert "Stale user-flagged snapshot body." not in content
        # Authorship is reset to MIST on the derived artifact.
        fm_dict, _ = parse_frontmatter(content)
        assert fm_dict["authored_by"] == "mist"

    @pytest.mark.asyncio
    async def test_does_not_touch_curated_user_md(self, vault_writer: VaultWriter, tmp_path: Path):
        """A sibling curated users/<uid>.md with authored_by:user is untouched."""
        curated = tmp_path / "vault" / "users" / "user.md"
        curated.parent.mkdir(parents=True, exist_ok=True)
        curated.write_text(
            "---\ntype: mist-user\nuser_id: user\nauthored_by: user\n"
            "last_updated: 2026-04-01\nrelated_sessions: []\ntags: []\n---\n\n"
            "Hand-curated authoritative profile.\n",
            encoding="utf-8",
        )

        await vault_writer.upsert_user_snapshot(
            user_id="user", body_markdown="Graph snapshot body."
        )

        # Curated file byte-for-byte unchanged.
        assert (
            curated.read_text(encoding="utf-8")
            == "---\ntype: mist-user\nuser_id: user\nauthored_by: user\n"
            "last_updated: 2026-04-01\nrelated_sessions: []\ntags: []\n---\n\n"
            "Hand-curated authoritative profile.\n"
        )
        # Snapshot landed in its own file.
        snapshot = tmp_path / "vault" / "users" / "user-graph-snapshot.md"
        assert "Graph snapshot body." in snapshot.read_text(encoding="utf-8")

    @pytest.mark.asyncio
    async def test_provenance_section_appended_when_absent(
        self, vault_writer: VaultWriter, tmp_path: Path
    ):
        """Provenance handling stays consistent with _upsert_user_sync."""
        path_str = await vault_writer.upsert_user_snapshot(
            user_id="user", body_markdown="# Snapshot\n"
        )
        _, body = parse_frontmatter(Path(path_str).read_text(encoding="utf-8"))
        assert "## Provenance" in body

    @pytest.mark.asyncio
    async def test_caller_provenance_not_duplicated(
        self, vault_writer: VaultWriter, tmp_path: Path
    ):
        """A caller-supplied Provenance section is trusted (no duplicate)."""
        body_md = "## Facts\n- x\n\n## Provenance\n- source: graph\n"
        path_str = await vault_writer.upsert_user_snapshot(user_id="user", body_markdown=body_md)
        _, body = parse_frontmatter(Path(path_str).read_text(encoding="utf-8"))
        assert body.count("## Provenance") == 1
