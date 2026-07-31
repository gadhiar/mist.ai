"""write_session_note renders the whole note, idempotently."""

from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio

from backend.chat.session_synthesizer import SessionSynthesis
from backend.errors import VaultWriteError
from backend.knowledge.config import VaultConfig
from backend.vault.models import parse_frontmatter
from backend.vault.writer import VaultWriter

# ---------------------------------------------------------------------------
# Fixtures
#
# No conftest.py under tests/unit/vault/ shares a `vault_writer` fixture --
# tests/unit/vault/test_writer.py defines its own, file-local. Mirrored here
# rather than promoted to a shared fixture, per the brief's instruction to
# reuse the existing construction shape without inventing a new one.
# ---------------------------------------------------------------------------


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


def _synthesis() -> SessionSynthesis:
    return SessionSynthesis(
        title="Vault write policy",
        body="### What Was Accomplished\n- Dropped per-turn appends\n",
    )


@pytest.mark.asyncio
async def test_renders_a_complete_note(vault_writer, tmp_path):
    path = tmp_path / "sessions" / "2026-07-30-vault-policy.md"

    written = await vault_writer.write_session_note(
        vault_note_path=str(path), synthesis=_synthesis()
    )

    assert written == str(path)
    text = path.read_text(encoding="utf-8")
    assert "title: Vault write policy" in text
    assert "status: completed" in text
    assert "### What Was Accomplished" in text
    assert "## Turn" not in text, "R1.3.1: notes carry no transcript"


@pytest.mark.asyncio
async def test_render_is_idempotent_byte_for_byte(vault_writer, tmp_path):
    """The load-bearing property: a note is a pure function of (turns, epoch).

    Re-rendering must be safe, because that is what makes partial-failure
    recovery work -- if synthesis succeeds but a later step fails, the next
    boot re-renders the whole file.
    """
    path = tmp_path / "sessions" / "2026-07-30-idem.md"

    await vault_writer.write_session_note(vault_note_path=str(path), synthesis=_synthesis())
    first = path.read_bytes()

    await vault_writer.write_session_note(vault_note_path=str(path), synthesis=_synthesis())
    second = path.read_bytes()

    assert first == second, "re-rendering the same synthesis must be byte-identical"


@pytest.mark.asyncio
async def test_renders_a_skipped_stub_without_synthesis(vault_writer, tmp_path):
    """Bounded retry persists in the vault, not in memory, so it survives a
    restart -- and is visible in Obsidian rather than buried in logs.
    """
    path = tmp_path / "sessions" / "2026-07-30-skipped.md"

    await vault_writer.write_session_note(
        vault_note_path=str(path), synthesis=None, status="skipped"
    )

    text = path.read_text(encoding="utf-8")
    assert "status: skipped" in text
    assert "### What Was Accomplished" not in text


@pytest.mark.asyncio
async def test_overwrites_a_pre_existing_note_rather_than_appending(vault_writer, tmp_path):
    path = tmp_path / "sessions" / "2026-07-30-overwrite.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("stale content that must not survive\n", encoding="utf-8")

    await vault_writer.write_session_note(vault_note_path=str(path), synthesis=_synthesis())

    text = path.read_text(encoding="utf-8")
    assert "stale content" not in text


@pytest.mark.asyncio
async def test_raises_on_a_path_without_a_canonical_date_prefix(vault_writer, tmp_path):
    """A non-`session_path()`-shaped path has no date to derive.

    Falling back to the wall clock here would make the render's date depend
    on when it happens to run -- silently reintroducing the exact
    non-determinism `write_session_note` exists to eliminate. Fail loudly
    instead of guessing.
    """
    path = tmp_path / "sessions" / "not-a-canonical-session-filename.md"

    with pytest.raises(VaultWriteError):
        await vault_writer.write_session_note(vault_note_path=str(path), synthesis=_synthesis())


@pytest.mark.asyncio
async def test_related_entities_are_deduped_and_sorted(vault_writer, tmp_path):
    path = tmp_path / "sessions" / "2026-07-30-entities.md"

    await vault_writer.write_session_note(
        vault_note_path=str(path),
        synthesis=_synthesis(),
        related_entities=["zeta", "alpha", "alpha"],
    )

    fm_dict, _body = parse_frontmatter(path.read_text(encoding="utf-8"))
    assert fm_dict["related_entities"] == ["alpha", "zeta"]


@pytest.mark.asyncio
async def test_frontmatter_carries_model_hash_when_provided(tmp_path):
    """Phase 8 stamps: when VaultWriter is constructed with a model_hash,
    new session-note frontmatter must carry it.

    Migrated from `test_writer.py::TestAppendTurnToSession` (R1.3.1): the
    property is unchanged, only the write path moved.
    """
    config = _make_config(tmp_path)
    writer = VaultWriter(config, model_hash="gemma-4-e4b-q5-k-m-test-v1")
    await writer.start()
    try:
        path = tmp_path / "sessions" / "2026-07-30-stamp-test.md"
        await writer.write_session_note(vault_note_path=str(path), synthesis=_synthesis())
        fm_dict, _body = parse_frontmatter(path.read_text(encoding="utf-8"))
        assert fm_dict["model_hash"] == "gemma-4-e4b-q5-k-m-test-v1"
    finally:
        await writer.stop()


@pytest.mark.asyncio
async def test_frontmatter_model_hash_null_when_unset(vault_writer, tmp_path):
    """Default fixture omits model_hash; frontmatter should serialize null."""
    path = tmp_path / "sessions" / "2026-07-30-no-stamp-test.md"

    await vault_writer.write_session_note(vault_note_path=str(path), synthesis=_synthesis())

    fm_dict, _body = parse_frontmatter(path.read_text(encoding="utf-8"))
    assert fm_dict["model_hash"] is None
