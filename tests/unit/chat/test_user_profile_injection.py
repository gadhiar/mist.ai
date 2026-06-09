"""Always-inject the known user's curated profile into the turn context.

When the user is known, MIST must ALWAYS include the body of
`users/<user_id>.md` in the assembled turn context -- independent of
retrieval similarity or intent -- mirroring how the MIST.md conventions are
always injected via ConventionsLoader. Root cause being fixed: the chat
path's vault auto-inject retrieves vault chunks by SEMANTIC SIMILARITY with a
small limit, and a meta-query like "what do you know about me?" embeds closer
to MIST's own first-person identity prose than to the user's third-person
profile, so the profile ranks below the cutoff and is never injected. The
user's own fact sheet must never be subject to a similarity gate.

These tests drive `_build_messages` directly (the single convergence point
for every conversation path: CLI chat, WebSocket streaming, voice), matching
the existing conventions-injection tests in test_conversation_handler_phase3.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.chat.conversation_handler import ConversationHandler
from backend.knowledge.models import (
    ConversationSession,
    RetrievalResult,
    RetrievedFact,
)
from backend.knowledge.retrieval.knowledge_retriever import KnowledgeRetriever
from backend.knowledge.storage.graph_store import GraphStore
from backend.vault.conventions import ConventionsLoader
from tests.mocks.config import build_test_config
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeNeo4jConnection
from tests.mocks.ollama import FakeLLM

# A distinctive sentence pulled from the curated profile body. If the
# always-inject works, this exact string appears in the assembled context.
_PROFILE_SENTINEL = "Raj Gadhia is a Slalom consultant building MIST.AI."
_PROFILE_BODY = (
    "# user\n\n"
    "## Professional\n\n"
    f"{_PROFILE_SENTINEL} He works directly and technically, no fluff.\n"
)
_PROFILE_FRONTMATTER = (
    "---\n"
    "type: mist-user\n"
    "user_id: user\n"
    "authored_by: user\n"
    "last_updated: '2026-06-09'\n"
    "related_sessions: []\n"
    "tags: []\n"
    "---\n\n"
)


class FakeExtractionPipelineUP:
    """Minimal extraction pipeline test double (matches phase3 pattern)."""

    async def extract_from_utterance(self, **kwargs):
        from backend.knowledge.extraction.validator import ValidationResult

        return ValidationResult(valid=True, entities=[], relationships=[])


def _seed_profile(vault_root: Path, *, filename: str = "user.md", body: str | None = None) -> None:
    """Write a curated `users/<filename>` profile into the temp vault root."""
    users_dir = vault_root / "users"
    users_dir.mkdir(parents=True, exist_ok=True)
    content = _PROFILE_FRONTMATTER + (body if body is not None else _PROFILE_BODY)
    (users_dir / filename).write_text(content, encoding="utf-8")


def _make_handler(vault_root: Path, *, user_id: str = "user") -> ConversationHandler:
    """Construct a ConversationHandler whose config.vault.root points at vault_root.

    `vault_user_id` is threaded into VaultConfig.default_user_id so
    `_user_id_for_vault()` resolves to it (anything other than the vestigial
    "raj" default is honored). The ConventionsLoader is pointed at the SAME
    vault root but the root carries no MIST.md, so the conventions block is
    absent and does not interfere with profile-block assertions.
    """
    conn = FakeNeo4jConnection()
    gs = GraphStore(conn, FakeEmbeddingGenerator())
    config = build_test_config(
        vault_root=str(vault_root),
        vault_enabled=True,
        vault_user_id=user_id,
    )
    retriever = KnowledgeRetriever(config=config, graph_store=gs)
    return ConversationHandler(
        config=config,
        graph_store=gs,
        extraction_pipeline=FakeExtractionPipelineUP(),
        retriever=retriever,
        llm_provider=FakeLLM(),
        conventions_loader=ConventionsLoader(vault_root),
    )


def _assembled_text(messages: list[dict[str, str]]) -> str:
    """Concatenate all message contents for substring/occurrence assertions."""
    return "\n".join(m.get("content", "") for m in messages)


class TestAlwaysInjectProfile:
    """The curated user profile is present regardless of retrieval similarity."""

    def test_profile_body_present_when_no_retrieval(self, tmp_path: Path) -> None:
        """Unrelated query, auto-inject returned nothing (retrieval_result=None):
        the profile body must STILL be present in the assembled context.
        """
        _seed_profile(tmp_path)
        handler = _make_handler(tmp_path)

        session = ConversationSession(session_id="up-s1", user_id="user")
        session.add_message("user", "what is the capital of France")

        messages = handler._build_messages(
            session=session,
            max_history=10,
            retrieval_result=None,
            mist_context=None,
        )

        assert _PROFILE_SENTINEL in _assembled_text(messages)

    def test_profile_block_is_a_distinct_labeled_message(self, tmp_path: Path) -> None:
        """The profile is injected as its own clearly-labeled context block,
        not folded into the static template or the conventions message.
        """
        _seed_profile(tmp_path)
        handler = _make_handler(tmp_path)

        session = ConversationSession(session_id="up-s2", user_id="user")
        session.add_message("user", "hello there friend")

        messages = handler._build_messages(
            session=session,
            max_history=10,
            retrieval_result=None,
            mist_context=None,
        )

        profile_msgs = [m for m in messages if _PROFILE_SENTINEL in m.get("content", "")]
        assert len(profile_msgs) == 1
        # Distinctive label so the model treats it as the user fact sheet.
        assert "user" in profile_msgs[0]["content"].lower()

    def test_profile_precedes_history(self, tmp_path: Path) -> None:
        """KV-cache discipline: the always-present profile block sits in the
        stable prefix, before the variable conversation-history tail.
        """
        _seed_profile(tmp_path)
        handler = _make_handler(tmp_path)

        session = ConversationSession(session_id="up-s3", user_id="user")
        session.add_message("user", "tell me a joke about cats")

        messages = handler._build_messages(
            session=session,
            max_history=10,
            retrieval_result=None,
            mist_context=None,
        )

        profile_index = next(
            i for i, m in enumerate(messages) if _PROFILE_SENTINEL in m.get("content", "")
        )
        history_index = next(
            i for i, m in enumerate(messages) if "joke about cats" in m.get("content", "")
        )
        assert profile_index < history_index


class TestUserIdCasingResolution:
    """The resolved user-id must locate the on-disk profile across casing."""

    def test_resolves_lowercase_user_md(self, tmp_path: Path) -> None:
        """Default known user resolves user_id 'user' -> users/user.md."""
        _seed_profile(tmp_path, filename="user.md")
        handler = _make_handler(tmp_path, user_id="user")

        session = ConversationSession(session_id="up-case1", user_id="User")
        session.add_message("user", "anything unrelated")

        messages = handler._build_messages(
            session=session,
            max_history=10,
            retrieval_result=None,
            mist_context=None,
        )

        assert _PROFILE_SENTINEL in _assembled_text(messages)

    def test_case_insensitive_fallback_when_exact_case_absent(self, tmp_path: Path) -> None:
        """When the exact-case file is absent, a case-insensitive match within
        users/ is used (request id 'User', on-disk file 'user.md').
        """
        # On-disk file is lowercase; resolved id is capitalized "User".
        _seed_profile(tmp_path, filename="user.md")
        handler = _make_handler(tmp_path, user_id="User")

        session = ConversationSession(session_id="up-case2", user_id="User")
        session.add_message("user", "something off topic")

        messages = handler._build_messages(
            session=session,
            max_history=10,
            retrieval_result=None,
            mist_context=None,
        )

        assert _PROFILE_SENTINEL in _assembled_text(messages)


class TestDedupAgainstAutoInject:
    """The profile must appear exactly once even if the auto-inject retrieves it."""

    def _retrieval_with_profile_chunk(self, profile_path: str) -> RetrievalResult:
        """Build a historical RetrievalResult whose facts include the profile's
        own chunk (subject='VaultNote', properties['path']=profile_path) plus
        an unrelated chunk, with formatted_context rendered the way the
        retriever renders the historical path (plain prose blocks).
        """
        profile_fact = RetrievedFact(
            subject="VaultNote",
            subject_type="VaultSession",
            predicate="MENTIONS",
            object="(file)",
            object_type="VaultChunk",
            properties={"path": profile_path, "text": _PROFILE_BODY, "content": _PROFILE_BODY},
            similarity_score=0.9,
            graph_distance=999,
        )
        other_fact = RetrievedFact(
            subject="VaultNote",
            subject_type="VaultSession",
            predicate="MENTIONS",
            object="Some Session",
            object_type="VaultChunk",
            properties={
                "path": "sessions/2026-06-01-other.md",
                "text": "Unrelated prose block about something else.",
                "content": "Unrelated prose block about something else.",
            },
            similarity_score=0.5,
            graph_distance=999,
        )
        formatted = (
            "Relevant prose from your vault (query: 'what do you know about me'):\n\n"
            f"{_PROFILE_BODY}\n\n"
            "Unrelated prose block about something else.\n\n"
        )
        return RetrievalResult(
            query="what do you know about me",
            user_id="user",
            facts=[profile_fact, other_fact],
            entities_found=0,
            total_facts=2,
            formatted_context=formatted,
            retrieval_time_ms=1.0,
            vector_search_time_ms=0.0,
            graph_traversal_time_ms=0.0,
            config_used={},
            intent="historical",
            document_chunks_used=2,
        )

    def test_profile_appears_exactly_once_when_autoinject_includes_it(self, tmp_path: Path) -> None:
        """Dedup: when the auto-inject retrieval includes the profile's own
        chunk, the final assembled context contains the profile body exactly
        once (the always-present block wins; the auto-inject copy is dropped).
        """
        _seed_profile(tmp_path)
        handler = _make_handler(tmp_path)

        session = ConversationSession(session_id="up-dedup1", user_id="user")
        session.add_message("user", "what do you know about me")

        retrieval = self._retrieval_with_profile_chunk("users/user.md")
        messages = handler._build_messages(
            session=session,
            max_history=10,
            retrieval_result=retrieval,
            mist_context=None,
        )

        text = _assembled_text(messages)
        assert text.count(_PROFILE_SENTINEL) == 1
        # The unrelated chunk must survive (dedup is surgical, profile-only).
        assert "Unrelated prose block about something else." in text

        # The single occurrence must be the always-present labeled profile
        # block (a user-role message), NOT the retrieval "Relevant prose"
        # system block -- proving dedup removed the auto-inject copy and the
        # always-block is the source of truth. Without dedup this would be 2.
        occurrence_msgs = [m for m in messages if _PROFILE_SENTINEL in m.get("content", "")]
        assert len(occurrence_msgs) == 1
        assert occurrence_msgs[0]["role"] == "user"
        assert "WHAT YOU KNOW ABOUT THE USER" in occurrence_msgs[0]["content"]

    def test_dedup_drops_profile_chunk_from_retrieval_facts(self, tmp_path: Path) -> None:
        """The profile chunk is removed from the retrieval facts so it is not
        re-rendered into the 'Relevant prose' block; the other chunk remains.
        """
        _seed_profile(tmp_path)
        handler = _make_handler(tmp_path)

        session = ConversationSession(session_id="up-dedup2", user_id="user")
        session.add_message("user", "what do you know about me")

        retrieval = self._retrieval_with_profile_chunk("users/user.md")
        handler._build_messages(
            session=session,
            max_history=10,
            retrieval_result=retrieval,
            mist_context=None,
        )

        remaining_paths = [
            (f.properties or {}).get("path") for f in retrieval.facts if f.subject == "VaultNote"
        ]
        assert "users/user.md" not in remaining_paths
        assert "sessions/2026-06-01-other.md" in remaining_paths


class TestGracefulAbsence:
    """When the profile file does not exist, assembly succeeds and omits it."""

    def test_no_block_and_no_error_when_profile_absent(self, tmp_path: Path) -> None:
        """Empty vault (no users/<id>.md): _build_messages must not error and
        must not inject any profile block.
        """
        # Note: tmp_path has NO users/ dir at all.
        handler = _make_handler(tmp_path)

        session = ConversationSession(session_id="up-absent", user_id="user")
        session.add_message("user", "what do you know about me")

        messages = handler._build_messages(
            session=session,
            max_history=10,
            retrieval_result=None,
            mist_context=None,
        )

        # Assembly succeeded (at least the static template + history are present)
        assert any(m["role"] == "system" for m in messages)
        # No profile sentinel anywhere.
        assert _PROFILE_SENTINEL not in _assembled_text(messages)

    def test_no_error_when_users_dir_exists_but_file_missing(self, tmp_path: Path) -> None:
        """users/ exists but the specific <user_id>.md is absent: still graceful."""
        (tmp_path / "users").mkdir(parents=True, exist_ok=True)
        # Seed an unrelated user file so the dir is non-empty but lacks user.md.
        _seed_profile(tmp_path, filename="someone-else.md")
        handler = _make_handler(tmp_path, user_id="user")

        session = ConversationSession(session_id="up-absent2", user_id="user")
        session.add_message("user", "hello")

        messages = handler._build_messages(
            session=session,
            max_history=10,
            retrieval_result=None,
            mist_context=None,
        )

        # someone-else.md must NOT be injected (only the resolved user's file).
        assert _PROFILE_SENTINEL not in _assembled_text(messages)


@pytest.mark.parametrize(
    "filename, configured_user_id, should_find",
    [
        pytest.param("user.md", "user", True, id="exact-lowercase"),
        pytest.param("user.md", "User", True, id="case-insensitive-fallback"),
        pytest.param("someone-else.md", "user", False, id="different-user-absent"),
    ],
)
def test_profile_resolution_matrix(
    tmp_path: Path, filename: str, configured_user_id: str, should_find: bool
) -> None:
    """Resolution matrix: the resolved user-id must locate its own profile and
    only its own profile.
    """
    _seed_profile(tmp_path, filename=filename)
    handler = _make_handler(tmp_path, user_id=configured_user_id)

    session = ConversationSession(session_id="up-matrix", user_id=configured_user_id)
    session.add_message("user", "unrelated query text here")

    messages = handler._build_messages(
        session=session,
        max_history=10,
        retrieval_result=None,
        mist_context=None,
    )

    present = _PROFILE_SENTINEL in _assembled_text(messages)
    assert present is should_find
