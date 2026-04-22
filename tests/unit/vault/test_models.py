"""Tests for backend.vault.models -- frontmatter schemas and utilities."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.vault.models import (
    AuthoredBy,
    MistDecisionFrontmatter,
    MistIdentityFrontmatter,
    MistSessionFrontmatter,
    MistUserFrontmatter,
    parse_frontmatter,
    render_frontmatter,
)

# ---------------------------------------------------------------------------
# TestAuthoredByEnum
# ---------------------------------------------------------------------------


class TestAuthoredByEnum:
    def test_five_values_exist(self):
        assert len(AuthoredBy) == 5

    def test_mist_value(self):
        assert AuthoredBy.MIST.value == "mist"

    def test_mist_pending_review_value(self):
        assert AuthoredBy.MIST_PENDING_REVIEW.value == "mist-pending-review"

    def test_user_value(self):
        assert AuthoredBy.USER.value == "user"

    def test_user_edit_value(self):
        assert AuthoredBy.USER_EDIT.value == "user-edit"

    def test_user_rejected_value(self):
        assert AuthoredBy.USER_REJECTED.value == "user-rejected"

    def test_value_is_usable_as_string(self):
        # AuthoredBy.value returns the string representation used in YAML
        assert AuthoredBy.MIST.value == "mist"
        # The enum members are also usable in string comparisons via .value
        assert AuthoredBy.MIST == "mist"

    def test_construct_from_string(self):
        assert AuthoredBy("user-edit") is AuthoredBy.USER_EDIT


# ---------------------------------------------------------------------------
# TestMistSessionFrontmatter
# ---------------------------------------------------------------------------


class TestMistSessionFrontmatter:
    def _make(self, **kwargs) -> MistSessionFrontmatter:
        defaults = {
            "session_id": "test-session",
            "date": "2026-04-21",
            "ontology_version": "1.0.0",
            "extraction_version": "2026-04-17-r1",
        }
        defaults.update(kwargs)
        return MistSessionFrontmatter(**defaults)

    def test_type_literal_is_mist_session(self):
        fm = self._make()
        assert fm.type == "mist-session"

    def test_defaults_applied(self):
        fm = self._make()
        assert fm.turn_count == 0
        assert fm.participants == ["user", "mist"]
        assert fm.authored_by == AuthoredBy.MIST
        assert fm.status == "in-progress"
        assert fm.append_sentinel_offset is None
        assert fm.related_entities == []
        assert fm.model_hash is None
        assert fm.tags == []

    def test_required_fields_raise_without_session_id(self):
        with pytest.raises(ValidationError):
            MistSessionFrontmatter(
                date="2026-04-21",
                ontology_version="1.0.0",
                extraction_version="2026-04-17-r1",
            )

    def test_required_fields_raise_without_ontology_version(self):
        with pytest.raises(ValidationError):
            MistSessionFrontmatter(
                session_id="test",
                date="2026-04-21",
                extraction_version="2026-04-17-r1",
            )

    def test_type_literal_rejects_wrong_value(self):
        with pytest.raises(ValidationError):
            MistSessionFrontmatter(
                type="mist-identity",  # type: ignore[arg-type]
                session_id="x",
                date="2026-04-21",
                ontology_version="1.0.0",
                extraction_version="2026-04-17-r1",
            )

    def test_model_dump_json_mode_is_yaml_friendly(self):
        fm = self._make(authored_by=AuthoredBy.MIST_PENDING_REVIEW)
        data = fm.model_dump(mode="json")

        # All values should be JSON primitives (no Enum instances)
        assert data["authored_by"] == "mist-pending-review"
        assert data["type"] == "mist-session"
        assert isinstance(data["turn_count"], int)
        assert isinstance(data["participants"], list)

    def test_status_enum_values(self):
        for status in ("in-progress", "completed", "archived"):
            fm = self._make(status=status)
            assert fm.status == status

    def test_model_copy_update(self):
        fm = self._make()
        updated = fm.model_copy(update={"turn_count": 5})
        assert updated.turn_count == 5
        assert fm.turn_count == 0  # original unchanged


# ---------------------------------------------------------------------------
# TestMistIdentityFrontmatter
# ---------------------------------------------------------------------------


class TestMistIdentityFrontmatter:
    def test_type_literal_is_mist_identity(self):
        fm = MistIdentityFrontmatter(version="1.0", last_updated="2026-04-21")
        assert fm.type == "mist-identity"

    def test_defaults(self):
        fm = MistIdentityFrontmatter(version="1.0", last_updated="2026-04-21")
        assert fm.authored_by == AuthoredBy.USER
        assert fm.tags == []

    def test_required_fields_raise_without_version(self):
        with pytest.raises(ValidationError):
            MistIdentityFrontmatter(last_updated="2026-04-21")

    def test_authored_by_accepts_enum(self):
        fm = MistIdentityFrontmatter(
            version="1.0",
            last_updated="2026-04-21",
            authored_by=AuthoredBy.USER_EDIT,
        )
        assert fm.authored_by == AuthoredBy.USER_EDIT


# ---------------------------------------------------------------------------
# TestMistUserFrontmatter
# ---------------------------------------------------------------------------


class TestMistUserFrontmatter:
    def test_type_literal_is_mist_user(self):
        fm = MistUserFrontmatter(user_id="raj", last_updated="2026-04-21")
        assert fm.type == "mist-user"

    def test_defaults(self):
        fm = MistUserFrontmatter(user_id="raj", last_updated="2026-04-21")
        assert fm.authored_by == AuthoredBy.MIST
        assert fm.related_sessions == []
        assert fm.tags == []

    def test_required_fields_raise_without_user_id(self):
        with pytest.raises(ValidationError):
            MistUserFrontmatter(last_updated="2026-04-21")

    def test_related_sessions_stored(self):
        fm = MistUserFrontmatter(
            user_id="raj",
            last_updated="2026-04-21",
            related_sessions=["[[2026-04-21-session]]"],
        )
        assert "[[2026-04-21-session]]" in fm.related_sessions


# ---------------------------------------------------------------------------
# TestMistDecisionFrontmatter
# ---------------------------------------------------------------------------


class TestMistDecisionFrontmatter:
    def _make(self, **kwargs) -> MistDecisionFrontmatter:
        defaults = {
            "id": "DEC-001",
            "title": "Use kebab-case slugs",
            "date": "2026-04-21",
        }
        defaults.update(kwargs)
        return MistDecisionFrontmatter(**defaults)

    def test_type_literal_is_mist_decision(self):
        fm = self._make()
        assert fm.type == "mist-decision"

    def test_defaults(self):
        fm = self._make()
        assert fm.status == "accepted"
        assert fm.authored_by == AuthoredBy.MIST
        assert fm.session is None
        assert fm.related_entities == []
        assert fm.tags == []

    def test_required_fields_raise_without_id(self):
        with pytest.raises(ValidationError):
            MistDecisionFrontmatter(title="x", date="2026-04-21")

    def test_required_fields_raise_without_title(self):
        with pytest.raises(ValidationError):
            MistDecisionFrontmatter(id="DEC-001", date="2026-04-21")

    def test_status_values(self):
        for status in ("proposed", "accepted", "superseded"):
            fm = self._make(status=status)
            assert fm.status == status

    def test_session_wikilink_stored(self):
        fm = self._make(session="[[2026-04-21-test]]")
        assert fm.session == "[[2026-04-21-test]]"


# ---------------------------------------------------------------------------
# TestParseFrontmatter
# ---------------------------------------------------------------------------


class TestParseFrontmatter:
    def test_parses_standard_frontmatter(self):
        text = "---\ntype: mist-session\nsession_id: abc\n---\n\nBody here.\n"
        fm_dict, body = parse_frontmatter(text)

        assert fm_dict["type"] == "mist-session"
        assert fm_dict["session_id"] == "abc"
        assert "Body here." in body

    def test_returns_empty_dict_when_no_frontmatter(self):
        text = "Just a body with no frontmatter."
        fm_dict, body = parse_frontmatter(text)

        assert fm_dict == {}
        assert body == text

    def test_handles_trailing_whitespace_in_body(self):
        text = "---\ntype: mist-user\nuser_id: raj\n---\n\nBody.   \n"
        fm_dict, body = parse_frontmatter(text)

        assert fm_dict["user_id"] == "raj"
        assert "Body." in body

    def test_handles_unicode_in_body(self):
        text = "---\ntype: mist-session\n---\n\nUnicode body: éàü\n"
        fm_dict, body = parse_frontmatter(text)

        assert "éàü" in body

    def test_handles_unicode_in_frontmatter_values(self):
        text = '---\ntitle: "Café notes"\n---\n\nbody\n'
        fm_dict, body = parse_frontmatter(text)

        assert "Café" in fm_dict["title"]

    def test_empty_frontmatter_block(self):
        text = "---\n---\n\nbody\n"
        fm_dict, body = parse_frontmatter(text)

        assert fm_dict == {}
        assert "body" in body

    def test_no_closing_delimiter_treated_as_no_frontmatter(self):
        text = "---\ntype: mist-session\nno closing delimiter"
        fm_dict, body = parse_frontmatter(text)

        assert fm_dict == {}
        assert body == text


# ---------------------------------------------------------------------------
# TestRenderFrontmatter
# ---------------------------------------------------------------------------


class TestRenderFrontmatter:
    def _make_session(self) -> MistSessionFrontmatter:
        return MistSessionFrontmatter(
            session_id="render-test",
            date="2026-04-21",
            ontology_version="1.0.0",
            extraction_version="2026-04-17-r1",
        )

    def test_output_starts_with_triple_dash(self):
        fm = self._make_session()
        result = render_frontmatter(fm, "body")

        assert result.startswith("---\n")

    def test_body_preserved_verbatim(self):
        fm = self._make_session()
        body = "## Turn 1\n\n**User:** hi\n\n"
        result = render_frontmatter(fm, body)

        assert body in result

    def test_round_trips_with_parse_frontmatter(self):
        fm = self._make_session()
        body = "Session body text.\n"
        rendered = render_frontmatter(fm, body)
        parsed_dict, parsed_body = parse_frontmatter(rendered)

        assert parsed_dict["session_id"] == "render-test"
        assert parsed_dict["type"] == "mist-session"
        assert "Session body text." in parsed_body

    def test_sort_keys_false_preserves_field_order(self):
        fm = self._make_session()
        result = render_frontmatter(fm, "body")

        # `type` should appear before `session_id` (field declaration order)
        type_idx = result.index("type:")
        session_idx = result.index("session_id:")
        assert type_idx < session_idx

    def test_authored_by_enum_serialized_as_string(self):
        fm = self._make_session()
        fm = fm.model_copy(update={"authored_by": AuthoredBy.MIST_PENDING_REVIEW})
        result = render_frontmatter(fm, "body")

        assert "mist-pending-review" in result
        # Should not contain the Python enum repr
        assert "AuthoredBy" not in result

    def test_none_values_preserved_as_null(self):
        fm = self._make_session()
        assert fm.model_hash is None
        result = render_frontmatter(fm, "body")

        assert "model_hash:" in result
