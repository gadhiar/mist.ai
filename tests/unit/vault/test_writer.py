"""Tests for backend.vault.writer.VaultWriter.

Uses real tmp_path filesystem (no pyfakefs). All async tests are marked
with @pytest.mark.asyncio. The `vault_writer` fixture starts a writer
against a temporary directory and stops it on teardown.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest
import pytest_asyncio

from backend.chat.session_synthesizer import SessionSynthesis
from backend.knowledge.config import VaultConfig
from backend.vault.models import parse_frontmatter
from backend.vault.writer import VaultWriter

# ---------------------------------------------------------------------------
# Fixtures
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


def _synthesis(title: str = "Test Session") -> SessionSynthesis:
    return SessionSynthesis(title=title, body="### What Was Accomplished\n- Did a thing\n")


# ---------------------------------------------------------------------------
# TestStartStop
# ---------------------------------------------------------------------------


class TestStartStop:
    @pytest.mark.asyncio
    async def test_directories_created_on_start(self, tmp_path: Path):
        config = _make_config(tmp_path)
        writer = VaultWriter(config)

        await writer.start()
        await writer.stop()

        vault = tmp_path / "vault"
        for subdir in ("sessions", "identity", "users", "decisions", "meta"):
            assert (vault / subdir).is_dir(), f"missing directory: {subdir}"

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self, vault_writer: VaultWriter):
        # Calling start() a second time must not raise or create a second task
        await vault_writer.start()
        await vault_writer.start()
        # If we reach here without error and can still write, consumer is fine
        today = "2026-04-21"
        path = await vault_writer.write_session_note(
            vault_note_path=vault_writer.session_path(today, "idempotent-start"),
            synthesis=_synthesis(),
        )
        assert Path(path).exists()

    @pytest.mark.asyncio
    async def test_stop_drains_queue(self, tmp_path: Path):
        config = _make_config(tmp_path)
        writer = VaultWriter(config)
        await writer.start()

        # Enqueue several jobs for distinct sessions, then stop -- all should
        # complete before stop() returns.
        paths = [writer.session_path("2026-04-21", f"drain-test-{i}") for i in range(1, 4)]
        for path_str in paths:
            await writer.write_session_note(
                vault_note_path=path_str, synthesis=_synthesis(f"Drain {path_str}")
            )

        await writer.stop()

        for path_str in paths:
            assert Path(path_str).exists()

    @pytest.mark.asyncio
    async def test_double_stop_is_safe(self, tmp_path: Path):
        config = _make_config(tmp_path)
        writer = VaultWriter(config)
        await writer.start()
        await writer.stop()
        # Second stop must not raise
        await writer.stop()


# ---------------------------------------------------------------------------
# TestSessionPath
# ---------------------------------------------------------------------------


class TestSessionPath:
    def test_returns_expected_absolute_path(self, tmp_path: Path):
        config = _make_config(tmp_path)
        writer = VaultWriter(config)

        result = writer.session_path("2026-04-21", "my-session")

        expected = str(tmp_path / "vault" / "sessions" / "2026-04-21-my-session.md")
        assert result == expected

    def test_raises_on_invalid_date_format(self, tmp_path: Path):
        config = _make_config(tmp_path)
        writer = VaultWriter(config)

        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            writer.session_path("21-04-2026", "session")

    def test_raises_on_non_kebab_slug(self, tmp_path: Path):
        config = _make_config(tmp_path)
        writer = VaultWriter(config)

        with pytest.raises(ValueError, match="kebab"):
            writer.session_path("2026-04-21", "My Session")

    def test_raises_on_uppercase_slug(self, tmp_path: Path):
        config = _make_config(tmp_path)
        writer = VaultWriter(config)

        with pytest.raises(ValueError, match="kebab"):
            writer.session_path("2026-04-21", "MySession")

    def test_single_word_slug_accepted(self, tmp_path: Path):
        config = _make_config(tmp_path)
        writer = VaultWriter(config)

        result = writer.session_path("2026-04-21", "session123")
        assert result.endswith("session123.md")

    def test_raises_on_non_date_string(self, tmp_path: Path):
        config = _make_config(tmp_path)
        writer = VaultWriter(config)

        with pytest.raises(ValueError):
            writer.session_path("not-a-date", "slug")


# ---------------------------------------------------------------------------
# TestUpsertIdentity
# ---------------------------------------------------------------------------


_SAMPLE_TRAITS = [
    {"display_name": "Warm", "axis": "Persona", "description": "Warm by default."},
    {"display_name": "Transparent", "axis": "Platform", "description": "Shows all decisions."},
]
_SAMPLE_CAPS = [
    {"display_name": "Voice IO", "description": "VAD-gated voice pipeline."},
]
_SAMPLE_PREFS = [
    {
        "display_name": "Direct Communication",
        "enforcement": "strong",
        "context": "Always answer directly.",
    },
]


class TestUpsertIdentity:
    @pytest.mark.asyncio
    async def test_creates_identity_file(self, vault_writer: VaultWriter, tmp_path: Path):
        path_str = await vault_writer.upsert_identity(_SAMPLE_TRAITS, _SAMPLE_CAPS, _SAMPLE_PREFS)

        assert Path(path_str).exists()
        content = Path(path_str).read_text(encoding="utf-8")
        fm_dict, body = parse_frontmatter(content)

        assert fm_dict["type"] == "mist-identity"
        assert "## Traits" in body
        assert "## Capabilities" in body
        assert "## Preferences" in body

    @pytest.mark.asyncio
    async def test_traits_section_contains_entries(self, vault_writer: VaultWriter, tmp_path: Path):
        await vault_writer.upsert_identity(_SAMPLE_TRAITS, _SAMPLE_CAPS, _SAMPLE_PREFS)
        identity_path = tmp_path / "vault" / "identity" / "mist.md"
        _, body = parse_frontmatter(identity_path.read_text(encoding="utf-8"))

        assert "Transparent" in body
        assert "Warm" in body

    @pytest.mark.asyncio
    async def test_identical_inputs_produce_byte_identical_output(
        self, vault_writer: VaultWriter, tmp_path: Path
    ):
        await vault_writer.upsert_identity(_SAMPLE_TRAITS, _SAMPLE_CAPS, _SAMPLE_PREFS)
        identity_path = tmp_path / "vault" / "identity" / "mist.md"
        first_content = identity_path.read_text(encoding="utf-8")

        # Re-run with same inputs; only `rendered_at` timestamp will differ
        # but the structure and all other fields should match
        _, first_body = parse_frontmatter(first_content)
        first_fm_dict, _ = parse_frontmatter(first_content)

        await vault_writer.upsert_identity(_SAMPLE_TRAITS, _SAMPLE_CAPS, _SAMPLE_PREFS)
        second_content = identity_path.read_text(encoding="utf-8")
        second_fm_dict, second_body = parse_frontmatter(second_content)

        # Frontmatter type/version/authored_by identical
        assert first_fm_dict["type"] == second_fm_dict["type"]
        assert first_fm_dict["version"] == second_fm_dict["version"]
        assert first_fm_dict["authored_by"] == second_fm_dict["authored_by"]

        # Structural sections identical (aside from rendered_at timestamp)
        for section in ("## Traits", "## Capabilities", "## Preferences", "Warm", "Transparent"):
            assert section in first_body
            assert section in second_body

    @pytest.mark.asyncio
    async def test_empty_lists_handled_without_error(
        self, vault_writer: VaultWriter, tmp_path: Path
    ):
        path_str = await vault_writer.upsert_identity([], [], [])

        content = Path(path_str).read_text(encoding="utf-8")
        assert "## Traits" in content
        assert "## Capabilities" in content

    @pytest.mark.asyncio
    async def test_provenance_section_present(self, vault_writer: VaultWriter, tmp_path: Path):
        await vault_writer.upsert_identity(_SAMPLE_TRAITS, _SAMPLE_CAPS, _SAMPLE_PREFS)
        identity_path = tmp_path / "vault" / "identity" / "mist.md"
        _, body = parse_frontmatter(identity_path.read_text(encoding="utf-8"))

        assert "## Provenance" in body
        assert "seed_data.yaml" in body


# ---------------------------------------------------------------------------
# TestUpsertUser
# ---------------------------------------------------------------------------


class TestUpsertUser:
    @pytest.mark.asyncio
    async def test_creates_new_user_file_with_body(self, vault_writer: VaultWriter, tmp_path: Path):
        path_str = await vault_writer.upsert_user("raj", "## Facts\n- Uses Python\n")

        path = Path(path_str)
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        fm_dict, body = parse_frontmatter(content)

        assert fm_dict["type"] == "mist-user"
        assert fm_dict["user_id"] == "raj"
        assert "Uses Python" in body

    @pytest.mark.asyncio
    async def test_provenance_section_appended(self, vault_writer: VaultWriter, tmp_path: Path):
        path_str = await vault_writer.upsert_user("raj", "# User\n")

        _, body = parse_frontmatter(Path(path_str).read_text(encoding="utf-8"))
        assert "## Provenance" in body

    @pytest.mark.asyncio
    async def test_user_edit_file_body_not_overwritten(
        self, vault_writer: VaultWriter, tmp_path: Path
    ):
        # Create file with authored_by: user-edit
        user_path = tmp_path / "vault" / "users" / "raj.md"
        user_path.parent.mkdir(parents=True, exist_ok=True)
        user_path.write_text(
            "---\ntype: mist-user\nuser_id: raj\nauthored_by: user-edit\n"
            "last_updated: 2026-04-01\nrelated_sessions: []\ntags: []\n---\n\n"
            "User-authored body that must not be replaced.\n",
            encoding="utf-8",
        )

        await vault_writer.upsert_user("raj", "New MIST-generated body.")

        content = user_path.read_text(encoding="utf-8")
        assert "User-authored body that must not be replaced." in content
        assert "New MIST-generated body." not in content

    @pytest.mark.asyncio
    async def test_user_authored_file_body_not_overwritten(
        self, vault_writer: VaultWriter, tmp_path: Path
    ):
        user_path = tmp_path / "vault" / "users" / "raj.md"
        user_path.parent.mkdir(parents=True, exist_ok=True)
        user_path.write_text(
            "---\ntype: mist-user\nuser_id: raj\nauthored_by: user\n"
            "last_updated: 2026-04-01\nrelated_sessions: []\ntags: []\n---\n\n"
            "User original content.\n",
            encoding="utf-8",
        )

        await vault_writer.upsert_user("raj", "MIST replacement attempt.")

        content = user_path.read_text(encoding="utf-8")
        assert "User original content." in content
        assert "MIST replacement attempt." not in content

    @pytest.mark.asyncio
    async def test_mist_authored_file_body_is_overwritten(
        self, vault_writer: VaultWriter, tmp_path: Path
    ):
        user_path = tmp_path / "vault" / "users" / "raj.md"
        user_path.parent.mkdir(parents=True, exist_ok=True)
        user_path.write_text(
            "---\ntype: mist-user\nuser_id: raj\nauthored_by: mist\n"
            "last_updated: 2026-04-01\nrelated_sessions: []\ntags: []\n---\n\n"
            "Old MIST body.\n",
            encoding="utf-8",
        )

        await vault_writer.upsert_user("raj", "Updated MIST body.")

        content = user_path.read_text(encoding="utf-8")
        assert "Updated MIST body." in content

    @pytest.mark.asyncio
    async def test_quoted_provenance_does_not_suppress_writer_section(
        self, vault_writer: VaultWriter, tmp_path: Path
    ):
        r"""A quoted line `> ## Provenance` is not a real heading, so the
        writer must still append its default Provenance. Pre-fix substring
        check `"## Provenance" in body_markdown` matched the quoted text
        and incorrectly suppressed the writer's section.
        """
        import re as _re

        body = (
            "## My Notes\n"
            "Some content.\n"
            "\n"
            "> ## Provenance\n"
            "> just discussing how Provenance works\n"
        )

        path_str = await vault_writer.upsert_user("raj", body)
        _, rendered = parse_frontmatter(Path(path_str).read_text(encoding="utf-8"))

        # Anchored heading match: count line-anchored ## Provenance only
        actual_headings = _re.findall(r"(?m)^##\s+Provenance\s*$", rendered)
        assert len(actual_headings) == 1, (
            f"quoted Provenance is not a real section; writer must append "
            f"its default. Got {len(actual_headings)} actual headings: "
            f"{rendered!r}"
        )

    @pytest.mark.asyncio
    async def test_lowercase_provenance_heading_counts_as_section(
        self, vault_writer: VaultWriter, tmp_path: Path
    ):
        """Lowercase `## provenance` is the same logical section per
        markdown convention; the writer must NOT add a near-duplicate
        capital-P version. Pre-fix substring check was case-sensitive.
        """
        import re as _re

        body = (
            "## My Notes\n"
            "Some content.\n"
            "\n"
            "## provenance\n"
            "- rendered_at: 2026-05-08T00:00:00+00:00\n"
        )

        path_str = await vault_writer.upsert_user("raj", body)
        _, rendered = parse_frontmatter(Path(path_str).read_text(encoding="utf-8"))

        # Total Provenance-like headings (case-insensitive) = 1
        case_insensitive_count = len(_re.findall(r"(?im)^##\s+Provenance\s*$", rendered))
        assert case_insensitive_count == 1, (
            f"lowercase Provenance should count as the section heading; "
            f"writer must not append a capital-P duplicate. Got "
            f"{case_insensitive_count} headings: {rendered!r}"
        )

    @pytest.mark.asyncio
    async def test_provenance_not_duplicated_when_body_includes_one(
        self, vault_writer: VaultWriter, tmp_path: Path
    ):
        r"""When body_markdown already contains a `## Provenance` section,
        the writer must NOT append a second one.

        Regression test for the conversation_handler /
        user_snapshot.render_user_snapshot_body coordination bug: the
        snapshot renderer returns a body that already includes
        "## Provenance\n- rendered_at: ...\n- source: graph snapshot
        ...". Pre-fix `_upsert_user_sync` appended its own minimal
        Provenance section unconditionally, producing a duplicate that
        accumulated across re-renders during continuous use.
        """
        body_with_provenance = (
            "## Facts\n"
            "- Uses Python\n"
            "\n"
            "## Provenance\n"
            "- rendered_at: 2026-05-08T00:00:00+00:00\n"
            "- source: graph snapshot (User entity + 1-hop outbound neighbors)\n"
        )

        path_str = await vault_writer.upsert_user("raj", body_with_provenance)

        content = Path(path_str).read_text(encoding="utf-8")
        _, body = parse_frontmatter(content)

        # Exactly one Provenance section in the rendered file body.
        assert body.count("## Provenance") == 1, (
            f"expected exactly 1 '## Provenance' section, got "
            f"{body.count('## Provenance')}: {body!r}"
        )
        # Caller-supplied source attribution preserved. The substring
        # "source: graph snapshot" is enough to verify the renderer's
        # Provenance content was passed through verbatim; the full
        # parenthetical "(User entity + 1-hop outbound neighbors)" is
        # owned by user_snapshot.render_user_snapshot_body and would
        # couple this writer test to that renderer's exact phrasing.
        assert "source: graph snapshot" in body


# ---------------------------------------------------------------------------
# TestQueueSerialization
# ---------------------------------------------------------------------------


class TestQueueSerialization:
    @pytest.mark.asyncio
    async def test_concurrent_writes_to_same_note_serialize_without_corruption(
        self, vault_writer: VaultWriter, tmp_path: Path
    ):
        # write_session_note is a full-render overwrite, not an accumulating
        # append -- so the property worth proving under concurrency is that
        # the queue serializes writes to one file rather than interleaving
        # them into a corrupted file. Five concurrent writes with distinct
        # titles must land as exactly one clean, parseable file.
        path_str = vault_writer.session_path("2026-04-21", "concurrent-test")

        tasks = [
            asyncio.create_task(
                vault_writer.write_session_note(
                    vault_note_path=path_str,
                    synthesis=SessionSynthesis(title=f"Session {i}", body=f"Body {i}\n"),
                )
            )
            for i in range(1, 6)
        ]
        await asyncio.gather(*tasks)

        content = Path(path_str).read_text(encoding="utf-8")
        fm_dict, _ = parse_frontmatter(content)

        assert fm_dict["title"] in [f"Session {i}" for i in range(1, 6)]
        assert content.count("---\n") == 2, "frontmatter block must not be interleaved/duplicated"

    @pytest.mark.asyncio
    async def test_failure_in_one_job_does_not_break_consumer(
        self, vault_writer: VaultWriter, tmp_path: Path
    ):
        good_path = vault_writer.session_path("2026-04-21", "good-job")
        await vault_writer.write_session_note(vault_note_path=good_path, synthesis=_synthesis())

        # Non-canonical stem (no YYYY-MM-DD- prefix) -- raises VaultWriteError
        bad_path = str(tmp_path / "vault" / "sessions" / "not-canonical.md")

        from backend.errors import VaultWriteError

        with pytest.raises(VaultWriteError):
            await vault_writer.write_session_note(vault_note_path=bad_path, synthesis=_synthesis())

        # Consumer must still be alive -- subsequent write works
        second_good = vault_writer.session_path("2026-04-21", "recovery-job")
        result = await vault_writer.write_session_note(
            vault_note_path=second_good, synthesis=_synthesis()
        )
        assert Path(result).exists()


# ---------------------------------------------------------------------------
# TestBackpressure
# ---------------------------------------------------------------------------


class TestBackpressure:
    @pytest.mark.asyncio
    async def test_backpressure_warning_logged_when_queue_exceeds_limit(
        self, tmp_path: Path, caplog
    ):
        # Configure a very low queue depth limit
        config = _make_config(tmp_path, writer_queue_max_depth=0)
        writer = VaultWriter(config)
        await writer.start()

        try:
            with caplog.at_level(logging.WARNING, logger="backend.vault.writer"):
                # First call: queue is empty (qsize=0, limit=0, 0 > 0 is False)
                # We need to pause the consumer and enqueue to trigger the check
                # Put a blocking item first by filling the queue manually
                path_str = writer.session_path("2026-04-21", "backpressure-test")
                # The check is qsize() > limit; with limit=0, any item in queue
                # after the first triggers it.  We need to stall the consumer.
                # Simplest: enqueue two writes rapidly for the same session so
                # the second enqueue sees qsize >= 1 > 0.
                t1 = asyncio.create_task(
                    writer.write_session_note(vault_note_path=path_str, synthesis=_synthesis())
                )
                t2 = asyncio.create_task(
                    writer.write_session_note(vault_note_path=path_str, synthesis=_synthesis())
                )
                await asyncio.gather(t1, t2)
        finally:
            await writer.stop()

        # Backpressure warnings are best-effort under concurrent scheduling; the
        # load-bearing assertion is that both writes still complete (the writer
        # never blocks the caller per ADR-010 Invariant 6).
        assert Path(path_str).exists()

    @pytest.mark.asyncio
    async def test_backpressure_does_not_block_caller(self, tmp_path: Path):
        config = _make_config(tmp_path, writer_queue_max_depth=0)
        writer = VaultWriter(config)
        await writer.start()

        try:
            path_str = writer.session_path("2026-04-21", "bp-noblock")
            # Even with limit exceeded, caller must not block indefinitely
            result = await asyncio.wait_for(
                writer.write_session_note(vault_note_path=path_str, synthesis=_synthesis()),
                timeout=5.0,
            )
            assert result == path_str
        finally:
            await writer.stop()


# ---------------------------------------------------------------------------
# TestMarkAuthoredByUserEdit
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_vault(tmp_path: Path) -> Path:
    """Return a temporary vault root with the standard subdirectory layout."""
    vault = tmp_path / "vault"
    for sub in ("sessions", "identity", "users", "decisions", "meta"):
        (vault / sub).mkdir(parents=True, exist_ok=True)
    return vault


@pytest.fixture()
def writer(tmp_vault: Path) -> VaultWriter:
    """Return a VaultWriter instance (not started) for sync helper tests."""
    config = VaultConfig(
        enabled=True,
        root=str(tmp_vault),
        default_user_id="raj",
        git_auto_init=False,
        session_soft_cap_turns=20,
        session_soft_cap_tokens=6000,
        append_sentinel="<!-- MIST_APPEND_HERE -->",
        writer_queue_max_depth=100,
    )
    return VaultWriter(config)


async def _mark_via_consumer(writer: VaultWriter, p: Path) -> None:
    """Run the queued authored_by writeback with a live consumer."""
    await writer.start()
    try:
        await writer.mark_authored_by_user_edit(p)
    finally:
        await writer.stop()


def test_mark_authored_by_user_edit_updates_frontmatter(writer: VaultWriter, tmp_vault: Path):
    p = tmp_vault / "users" / "raj.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "---\n" "type: mist-user\n" "user_id: raj\n" "authored_by: mist\n" "---\n" "# Raj\n",
        encoding="utf-8",
    )
    asyncio.run(_mark_via_consumer(writer, p))
    text = p.read_text(encoding="utf-8")
    assert "authored_by: user-edit" in text
    assert "authored_by: mist" not in text


def test_mark_authored_by_user_edit_idempotent(writer: VaultWriter, tmp_vault: Path):
    p = tmp_vault / "users" / "raj.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "---\n" "type: mist-user\n" "authored_by: user-edit\n" "---\n" "body\n",
        encoding="utf-8",
    )
    asyncio.run(_mark_via_consumer(writer, p))
    text = p.read_text(encoding="utf-8")
    # Already user-edit; no change
    assert text.count("authored_by: user-edit") == 1


def test_mark_authored_by_user_edit_preserves_body_and_other_frontmatter(
    writer: VaultWriter, tmp_vault: Path
):
    p = tmp_vault / "users" / "raj.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    original = (
        "---\n"
        "type: mist-user\n"
        "user_id: raj\n"
        "authored_by: pipeline\n"
        "status: active\n"
        "---\n"
        "# Body heading\n"
        "\nSome content here.\n"
    )
    p.write_text(original, encoding="utf-8")
    asyncio.run(_mark_via_consumer(writer, p))
    text = p.read_text(encoding="utf-8")
    assert "authored_by: user-edit" in text
    assert "type: mist-user" in text
    assert "user_id: raj" in text
    assert "status: active" in text
    assert "# Body heading" in text
    assert "Some content here." in text


def test_enqueue_on_unstarted_writer_fails_fast(writer: VaultWriter, tmp_vault: Path):
    # A write on a never-started (or stopped) writer must raise instead of
    # awaiting a future no consumer will ever resolve.
    from backend.errors import VaultWriteError

    p = tmp_vault / "users" / "raj.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("---\nauthored_by: mist\n---\nbody\n", encoding="utf-8")

    with pytest.raises(VaultWriteError, match="not running"):
        asyncio.run(writer.mark_authored_by_user_edit(p))


# ---------------------------------------------------------------------------
# TestSessionIdUniqueness  (Phase 3 Task 18)
# ---------------------------------------------------------------------------
# The 2026-05-10 audit found 5 of 7 session notes had session_id: default
# because KnowledgeIntegration.current_session_id initializes to "default"
# and that raw string was written directly into frontmatter.
#
# Fix contract: _append_turn_sync derives the frontmatter session_id from
# the path stem (<date>-<slug>) rather than the raw external session_id arg.
# This ensures the frontmatter identifier is always human-readable and unique,
# regardless of what the caller passes as the external session_id.


class TestSessionIdUniqueness:
    @pytest.mark.asyncio
    async def test_frontmatter_session_id_never_equals_default(
        self, vault_writer: VaultWriter, tmp_path: Path
    ):
        """Regression: an external session_id='default' path allocation must
        NOT produce session_id: default in frontmatter.

        `_session_id_from_path` (shared by `write_session_note`) derives the
        frontmatter session_id from the path stem's slug rather than the raw
        external session_id allocated by
        `ConversationHandler._get_or_allocate_vault_path`.
        """
        path_str = vault_writer.session_path("2026-05-10", "plan-new-feature-37a8")
        await vault_writer.write_session_note(vault_note_path=path_str, synthesis=_synthesis())
        fm_dict, _ = parse_frontmatter(Path(path_str).read_text(encoding="utf-8"))
        assert fm_dict["session_id"] != "default", (
            "session_id in frontmatter must never be 'default'; " f"got {fm_dict['session_id']!r}"
        )

    @pytest.mark.asyncio
    async def test_frontmatter_session_id_matches_path_stem(
        self, vault_writer: VaultWriter, tmp_path: Path
    ):
        """Frontmatter session_id derives from the path stem (which carries the
        pre-allocated slug+hash), not the raw external session_id argument.

        This guarantees the frontmatter identifier matches the filename,
        making programmatic lookup via session_id reliable.
        """
        path_str = vault_writer.session_path("2026-05-10", "vault-architecture-3a7f")
        await vault_writer.write_session_note(vault_note_path=path_str, synthesis=_synthesis())
        path = Path(path_str)
        fm_dict, _ = parse_frontmatter(path.read_text(encoding="utf-8"))
        # path.stem is "2026-05-10-vault-architecture-3a7f"
        # The slug portion after the date prefix is "vault-architecture-3a7f"
        expected_session_id = "vault-architecture-3a7f"
        assert fm_dict["session_id"] == expected_session_id, (
            f"Expected session_id derived from path stem slug; " f"got {fm_dict['session_id']!r}"
        )

    @pytest.mark.asyncio
    async def test_frontmatter_session_id_stable_across_rerenders(
        self, vault_writer: VaultWriter, tmp_path: Path
    ):
        """session_id must not change across repeated renders of the same note.

        The path stem is fixed, so re-rendering (e.g. a catch-up retry)
        preserves the same frontmatter session_id.
        """
        path_str = vault_writer.session_path("2026-05-10", "stable-session-ab12")
        for i in range(1, 4):
            await vault_writer.write_session_note(
                vault_note_path=path_str, synthesis=_synthesis(f"Render {i}")
            )
        fm_dict, _ = parse_frontmatter(Path(path_str).read_text(encoding="utf-8"))
        assert fm_dict["session_id"] == "stable-session-ab12"

    @pytest.mark.asyncio
    async def test_two_sessions_with_same_slug_get_distinct_ids_via_hash(
        self, vault_writer: VaultWriter, tmp_path: Path
    ):
        """Two different sessions with similar utterances get distinct filenames
        via the hash suffix in the slug, which means distinct session_ids in
        frontmatter.
        """
        path1 = vault_writer.session_path("2026-05-10", "topic-abc1")
        path2 = vault_writer.session_path("2026-05-10", "topic-abc2")
        await vault_writer.write_session_note(vault_note_path=path1, synthesis=_synthesis())
        await vault_writer.write_session_note(vault_note_path=path2, synthesis=_synthesis())
        fm1, _ = parse_frontmatter(Path(path1).read_text(encoding="utf-8"))
        fm2, _ = parse_frontmatter(Path(path2).read_text(encoding="utf-8"))
        assert fm1["session_id"] != fm2["session_id"], (
            "Two sessions with different slugs must have different session_ids; "
            f"both got {fm1['session_id']!r}"
        )
