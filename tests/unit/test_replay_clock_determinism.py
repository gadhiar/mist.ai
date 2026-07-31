"""Determinism: replay-reproducible user-snapshot timestamps via clock injection.

The F2 extraction baseline replays a 60-probe corpus through the full
`handle_message` path and must be byte-for-byte reproducible at temperature 0.0.
A wall-clock `rendered_at` stamped into the seeded `users/<user_id>.md` (and the
C-pattern `-graph-snapshot.md`) reaches the chat system prompt and perturbs the
greedy reply, permanently diverging the conversation history from turn 2 on.

These tests pin the clock seam that makes the chain reproducible:

- `VaultWriter.upsert_user` / `upsert_user_snapshot` honor a caller-supplied
  `rendered_at`, producing a byte-identical file across two writes with the same
  fixed value (the seeded `users/user.md` is what turn-1's prompt reads).
- `bootstrap_vault_from_seed` threads an explicit `rendered_at` so the seeded
  user note is reproducible.
- `ConversationHandler` accepts an injected `now_fn`; the C-pattern writeback
  stamp uses it instead of inline `datetime.now(UTC)`.
- The factory env seam (`resolve_fixed_rendered_at`) returns None in production
  (wall-clock) and a fixed ISO string under the replay env var.

Production default is UNCHANGED: with no fixed value supplied, every site falls
back to wall-clock.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
import pytest_asyncio

from backend.knowledge import admin
from backend.knowledge.config import VaultConfig
from backend.vault.models import parse_frontmatter
from backend.vault.writer import VaultWriter

FIXED_RENDERED_AT = "2026-06-13T00:00:00+00:00"
FIXED_RENDERED_AT_2 = "2026-06-13T12:34:56+00:00"


def _make_config(tmp_path: Path, **kwargs) -> VaultConfig:
    defaults = {
        "enabled": True,
        "root": str(tmp_path / "vault"),
        "default_user_id": "user",
        "git_auto_init": False,
        "session_soft_cap_turns": 20,
        "session_soft_cap_tokens": 6000,
        "writer_queue_max_depth": 100,
    }
    defaults.update(kwargs)
    return VaultConfig(**defaults)


@pytest_asyncio.fixture
async def vault_writer(tmp_path: Path):
    config = _make_config(tmp_path)
    writer = VaultWriter(config)
    await writer.start()
    yield writer
    await writer.stop()


# ---------------------------------------------------------------------------
# Writer honors an injected rendered_at (the prompt-facing seed stamp)
# ---------------------------------------------------------------------------


class TestUpsertUserClockInjection:
    @pytest.mark.asyncio
    async def test_user_body_uses_injected_rendered_at(
        self, vault_writer: VaultWriter, tmp_path: Path
    ) -> None:
        # Body has no Provenance section, so the writer appends one with the
        # caller-supplied rendered_at instead of wall-clock.
        await vault_writer.upsert_user(
            user_id="user",
            body_markdown="# user\n\n## Profile\n- role: engineer\n",
            rendered_at=FIXED_RENDERED_AT,
        )

        path = tmp_path / "vault" / "users" / "user.md"
        _fm, body = parse_frontmatter(path.read_text(encoding="utf-8"))

        assert f"- rendered_at: {FIXED_RENDERED_AT}" in body
        assert "datetime" not in body  # sanity: no inline now() artifacts

    @pytest.mark.asyncio
    async def test_user_md_byte_identical_across_writes_with_fixed_clock(
        self, tmp_path: Path
    ) -> None:
        # Two fresh writers against two roots, same fixed clock + same body ->
        # byte-identical files. This is the seed-determinism contract: the
        # seeded users/user.md must not vary run-to-run.
        body = "# user\n\n## Profile\n- role: engineer\n"
        out: list[str] = []
        for sub in ("a", "b"):
            cfg = _make_config(tmp_path / sub)
            writer = VaultWriter(cfg)
            await writer.start()
            try:
                await writer.upsert_user(
                    user_id="user", body_markdown=body, rendered_at=FIXED_RENDERED_AT
                )
            finally:
                await writer.stop()
            out.append((tmp_path / sub / "vault" / "users" / "user.md").read_text("utf-8"))

        assert out[0] == out[1]

    @pytest.mark.asyncio
    async def test_user_md_default_uses_wall_clock(
        self, vault_writer: VaultWriter, tmp_path: Path
    ) -> None:
        # Production default (no rendered_at): a real wall-clock ISO timestamp is
        # stamped. Light assertion -- parses as a tz-aware datetime near now.
        before = datetime.now(UTC)
        await vault_writer.upsert_user(
            user_id="user", body_markdown="# user\n\n## Profile\n- role: engineer\n"
        )
        after = datetime.now(UTC)

        path = tmp_path / "vault" / "users" / "user.md"
        _fm, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        line = next(ln for ln in body.splitlines() if ln.strip().startswith("- rendered_at:"))
        stamped = datetime.fromisoformat(line.split("- rendered_at:", 1)[1].strip())

        assert stamped.tzinfo is not None
        assert before <= stamped <= after


class TestUpsertUserSnapshotClockInjection:
    @pytest.mark.asyncio
    async def test_snapshot_body_uses_injected_rendered_at(
        self, vault_writer: VaultWriter, tmp_path: Path
    ) -> None:
        await vault_writer.upsert_user_snapshot(
            user_id="user",
            body_markdown="# user\n\n## Tools and Technologies\n- **Python** (Technology)\n",
            rendered_at=FIXED_RENDERED_AT,
        )
        path = tmp_path / "vault" / "users" / "user-graph-snapshot.md"
        _fm, body = parse_frontmatter(path.read_text(encoding="utf-8"))

        assert f"- rendered_at: {FIXED_RENDERED_AT}" in body

    @pytest.mark.asyncio
    async def test_snapshot_default_uses_wall_clock(
        self, vault_writer: VaultWriter, tmp_path: Path
    ) -> None:
        before = datetime.now(UTC)
        await vault_writer.upsert_user_snapshot(
            user_id="user", body_markdown="# user\n\n## Tools\n- **X** (Technology)\n"
        )
        after = datetime.now(UTC)
        path = tmp_path / "vault" / "users" / "user-graph-snapshot.md"
        _fm, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        line = next(ln for ln in body.splitlines() if ln.strip().startswith("- rendered_at:"))
        stamped = datetime.fromisoformat(line.split("- rendered_at:", 1)[1].strip())

        assert before <= stamped <= after


# ---------------------------------------------------------------------------
# Seed bootstrap threads the fixed clock -> byte-identical seeded user note
# ---------------------------------------------------------------------------


class _RecordingWriter:
    """Records upsert_user/upsert_identity args to assert rendered_at threading."""

    def __init__(self) -> None:
        self.user_calls: list[dict] = []
        self.identity_calls: list[dict] = []

    async def upsert_identity(self, traits, capabilities, preferences, rendered_at=None) -> str:
        self.identity_calls.append({"rendered_at": rendered_at})
        return "/tmp/vault/identity/mist.md"

    async def upsert_user(self, user_id, body_markdown, rendered_at=None) -> str:
        self.user_calls.append({"user_id": user_id, "rendered_at": rendered_at})
        return f"/tmp/vault/users/{user_id}.md"


_SEED = {
    "mist_identity": {"id": "mist-identity", "entity_type": "MistIdentity", "display_name": "MIST"},
    "traits": [],
    "capabilities": [],
    "preferences": [],
    "user": {"id": "user", "entity_type": "User", "display_name": "Raj Gadhia"},
    "entities": [],
    "identity_relationships": [],
    "anchor_relationships": [],
}


class TestBootstrapSeedClockThreading:
    @pytest.mark.asyncio
    async def test_bootstrap_threads_rendered_at_to_upsert_user(self) -> None:
        writer = _RecordingWriter()
        await admin.bootstrap_vault_from_seed(writer, _SEED, rendered_at=FIXED_RENDERED_AT)

        assert writer.user_calls[0]["rendered_at"] == FIXED_RENDERED_AT

    @pytest.mark.asyncio
    async def test_seeded_user_md_byte_identical_across_seed_runs(self, tmp_path: Path) -> None:
        # End-to-end through the real VaultWriter: two seed runs against two
        # roots with the same fixed clock produce byte-identical users/user.md.
        out: list[str] = []
        for sub in ("a", "b"):
            cfg = _make_config(tmp_path / sub)
            writer = VaultWriter(cfg)
            await writer.start()
            try:
                await admin.bootstrap_vault_from_seed(writer, _SEED, rendered_at=FIXED_RENDERED_AT)
            finally:
                await writer.stop()
            out.append((tmp_path / sub / "vault" / "users" / "user.md").read_text("utf-8"))

        assert out[0] == out[1]


# ---------------------------------------------------------------------------
# Factory env seam: production unset -> None (wall-clock); replay -> fixed value
# ---------------------------------------------------------------------------


class TestResolveFixedRenderedAt:
    def test_unset_returns_none(self, monkeypatch) -> None:
        from backend import factories

        monkeypatch.delenv("MIST_FIXED_CLOCK", raising=False)
        assert factories.resolve_fixed_rendered_at() is None

    def test_set_returns_fixed_iso_string(self, monkeypatch) -> None:
        from backend import factories

        monkeypatch.setenv("MIST_FIXED_CLOCK", FIXED_RENDERED_AT)
        assert factories.resolve_fixed_rendered_at() == FIXED_RENDERED_AT

    def test_malformed_value_raises(self, monkeypatch) -> None:
        from backend import factories

        monkeypatch.setenv("MIST_FIXED_CLOCK", "not-a-timestamp")
        with pytest.raises(ValueError):
            factories.resolve_fixed_rendered_at()

    def test_clock_from_fixed_iso_is_constant(self, monkeypatch) -> None:
        # The handler-facing clock built from the env returns the SAME instant on
        # every call (det1 and det2 share one constant).
        from backend import factories

        monkeypatch.setenv("MIST_FIXED_CLOCK", FIXED_RENDERED_AT)
        now_fn = factories.build_now_fn()
        first = now_fn()
        second = now_fn()
        assert first == second
        assert first == datetime.fromisoformat(FIXED_RENDERED_AT)

    def test_clock_default_is_wall_clock(self, monkeypatch) -> None:
        from backend import factories

        monkeypatch.delenv("MIST_FIXED_CLOCK", raising=False)
        now_fn = factories.build_now_fn()
        value = now_fn()
        assert isinstance(value, datetime)
        assert value.tzinfo is not None


# ---------------------------------------------------------------------------
# ConversationHandler honors an injected now_fn at the C-pattern stamp site
# ---------------------------------------------------------------------------


class TestConversationHandlerClockInjection:
    @pytest.mark.asyncio
    async def test_c_pattern_writeback_uses_injected_now_fn(self, monkeypatch) -> None:
        # Construct a handler shell (bypass __init__) and wire only what
        # _maybe_refresh_user_vault reads, plus the injected now_fn.
        from unittest.mock import AsyncMock, MagicMock

        from backend.chat.conversation_handler import ConversationHandler

        handler = ConversationHandler.__new__(ConversationHandler)

        captured: dict[str, str] = {}

        class _Writer:
            async def upsert_user_snapshot(self, *, user_id, body_markdown, rendered_at=None):
                captured["rendered_at"] = rendered_at
                return f"users/{user_id}-graph-snapshot.md"

        handler._vault_writer = _Writer()
        handler.graph_store = MagicMock()
        handler._user_id_for_vault = lambda: "user"
        handler._now_fn = lambda: datetime.fromisoformat(FIXED_RENDERED_AT_2)

        monkeypatch.setattr(
            "backend.vault.user_snapshot.extraction_touched_user_scope",
            lambda *a, **k: True,
        )

        captured_snapshot_rendered_at: dict[str, str] = {}

        async def _fake_query(executor, user_id, rendered_at):
            captured_snapshot_rendered_at["value"] = rendered_at
            return MagicMock(edges_by_type={})

        monkeypatch.setattr(
            "backend.vault.user_snapshot.query_user_snapshot",
            AsyncMock(side_effect=_fake_query),
        )
        monkeypatch.setattr(
            "backend.vault.user_snapshot.render_user_snapshot_body",
            lambda snapshot: "BODY",
        )
        monkeypatch.setattr(
            "backend.knowledge.storage.graph_executor.GraphExecutor",
            lambda *a, **k: MagicMock(),
        )

        extraction_result = MagicMock(validated_entities=[MagicMock()], validated_relationships=[])
        await handler._maybe_refresh_user_vault(extraction_result)

        # The injected clock drives BOTH the snapshot query $now and the
        # writeback rendered_at -- no inline datetime.now().
        assert captured_snapshot_rendered_at["value"] == FIXED_RENDERED_AT_2
        assert captured["rendered_at"] == FIXED_RENDERED_AT_2
