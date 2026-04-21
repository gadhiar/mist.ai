"""Unit tests for backend.chat.mist_context.MistContext."""

from backend.chat.mist_context import (
    MistCapability,
    MistContext,
    MistPreference,
    MistTrait,
)


class TestMistContextRender:
    """MistContext.as_system_prompt_block produces a deterministic persona block."""

    def test_renders_identity_header(self):
        ctx = MistContext(
            display_name="MIST",
            pronouns="she/her",
            self_concept="A cognitive architecture for personal knowledge.",
            traits=[],
            capabilities=[],
            preferences=[],
        )
        block = ctx.as_system_prompt_block()
        assert "MIST" in block
        assert "she/her" in block
        assert "cognitive architecture" in block

    def test_renders_absolute_preferences_as_hard_rules(self):
        ctx = MistContext(
            display_name="MIST",
            pronouns="she/her",
            self_concept="",
            traits=[],
            capabilities=[],
            preferences=[
                MistPreference(
                    id="pref-no-emoji",
                    display_name="No emoji or unicode decoration",
                    enforcement="absolute",
                    context="Hard rule across all output channels.",
                ),
                MistPreference(
                    id="pref-concise",
                    display_name="Prefer concise responses",
                    enforcement="preferred",
                    context="Unless depth is requested.",
                ),
            ],
        )
        block = ctx.as_system_prompt_block()
        assert "HARD RULE" in block or "MUST" in block or "non-negotiable" in block.lower()
        assert "No emoji or unicode decoration" in block
        assert "concise" in block.lower()

    def test_renders_traits(self):
        ctx = MistContext(
            display_name="MIST",
            pronouns="she/her",
            self_concept="",
            traits=[
                MistTrait(
                    id="trait-warm",
                    display_name="Warm",
                    axis="Persona",
                    description="Friendly default tone.",
                ),
                MistTrait(
                    id="trait-technical",
                    display_name="Technical",
                    axis="Persona",
                    description="Precise language.",
                ),
            ],
            capabilities=[],
            preferences=[],
        )
        block = ctx.as_system_prompt_block()
        assert "Warm" in block
        assert "Technical" in block

    def test_renders_capabilities(self):
        ctx = MistContext(
            display_name="MIST",
            pronouns="she/her",
            self_concept="",
            traits=[],
            capabilities=[
                MistCapability(
                    id="cap-tool-use", display_name="Tool use", description="Invokes MCP tools."
                ),
            ],
            preferences=[],
        )
        block = ctx.as_system_prompt_block()
        assert "Tool use" in block

    def test_empty_context_renders_minimal_block(self):
        ctx = MistContext(
            display_name="MIST",
            pronouns="",
            self_concept="",
            traits=[],
            capabilities=[],
            preferences=[],
        )
        block = ctx.as_system_prompt_block()
        assert "MIST" in block
        # Empty section headers should not appear as empty headers (no "Traits:\n\n").
        assert "Traits:\n\n" not in block
        assert "Capabilities:\n\n" not in block

    def test_absolute_and_preferred_rendered_separately(self):
        """Absolute prefs render above preferred prefs in distinct sections."""
        ctx = MistContext(
            display_name="MIST",
            pronouns="she/her",
            self_concept="",
            traits=[],
            capabilities=[],
            preferences=[
                MistPreference(
                    id="a", display_name="Absolute Rule", enforcement="absolute", context="Ctx A"
                ),
                MistPreference(
                    id="b", display_name="Preferred Guide", enforcement="preferred", context="Ctx B"
                ),
            ],
        )
        block = ctx.as_system_prompt_block()
        # Both rendered
        assert "Absolute Rule" in block
        assert "Preferred Guide" in block
        # Absolute MUST appear before preferred in the rendered order
        assert block.index("Absolute Rule") < block.index("Preferred Guide")


class TestMistContextForwardCompat:
    """Cluster 3 forward-compat: optional fields with safe defaults for Cluster 8 (ADR-010)."""

    def test_default_forward_compat_fields(self):
        ctx = MistContext(
            display_name="MIST",
            pronouns="she/her",
            self_concept="",
        )
        assert ctx.vault_note_path is None
        assert ctx.authored_by == "mist"

    def test_forward_compat_fields_are_overridable(self):
        ctx = MistContext(
            display_name="MIST",
            pronouns="she/her",
            self_concept="",
            vault_note_path="mist-memory/identity/mist.md",
            authored_by="user-edit",
        )
        assert ctx.vault_note_path == "mist-memory/identity/mist.md"
        assert ctx.authored_by == "user-edit"
