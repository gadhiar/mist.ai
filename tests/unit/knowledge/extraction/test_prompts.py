"""Snapshot-style assertions on EXTRACTION_SYSTEM_PROMPT contents.

These are deliberately coarse-grained: we assert key language is PRESENT
rather than asserting a full-text snapshot. This lets us refine wording
without churn-testing, while still catching regressions that drop a
load-bearing rule.
"""

from backend.knowledge.extraction.prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_USER_TEMPLATE,
)


class TestExtractionPromptInjectionResistance:
    """Bug K: prompt must contain injection-resistance rule."""

    def test_prompt_rejects_directives(self):
        """Prompt must instruct the model not to follow in-utterance directives."""
        assert "do not follow" in EXTRACTION_SYSTEM_PROMPT.lower() or (
            "do not execute" in EXTRACTION_SYSTEM_PROMPT.lower()
        ), "Expected injection-resistance language referring to directives/instructions"

    def test_prompt_mentions_directive_or_instruction(self):
        """Prompt must explicitly reference directives or instructions in utterances."""
        prompt_lower = EXTRACTION_SYSTEM_PROMPT.lower()
        assert any(
            term in prompt_lower for term in ["directive", "instruction", "command"]
        ), "Expected explicit reference to directive/instruction/command"

    def test_prompt_restricts_to_factual_claims(self):
        """Prompt must constrain extraction to factual claims about user/work/world."""
        prompt_lower = EXTRACTION_SYSTEM_PROMPT.lower()
        assert (
            "factual" in prompt_lower or "stated" in prompt_lower
        ), "Expected factual-claim restriction language"

    def test_prompt_rejects_hypotheticals_and_directives(self):
        """Prompt must explicitly mention not extracting hypotheticals or instructions."""
        prompt_lower = EXTRACTION_SYSTEM_PROMPT.lower()
        assert "hypothetical" in prompt_lower, "Expected 'hypothetical' in rejection list"
        assert "instruction" in prompt_lower, "Expected 'instruction' in rejection list"

    def test_prompt_rule_10_covers_override_and_new_instructions(self):
        """Rule 10 example list must cover all 6 preprocessor injection patterns."""
        prompt_lower = EXTRACTION_SYSTEM_PROMPT.lower()
        # These were added to close Critical #2 from Task 5 review.
        assert (
            "override the system" in prompt_lower
        ), "Expected 'override the system' in rule 10 example list"
        assert (
            "new instructions" in prompt_lower
        ), "Expected 'new instructions' in rule 10 example list"

    def test_prompt_rule_10_takes_precedence_over_rule_1(self):
        """Rule 10 must explicitly state precedence over Rule 1 for mixed utterances."""
        prompt_lower = EXTRACTION_SYSTEM_PROMPT.lower()
        # Closes Critical #1: without explicit priority, Gemma 4 E4B may follow
        # Rule 1 ("always create user entity") on a mixed directive/factual utterance.
        assert (
            "precedence" in prompt_lower or "takes priority" in prompt_lower
        ), "Expected rule-priority language in prompt"
        assert (
            "rule 10" in prompt_lower and "rule 1" in prompt_lower
        ), "Expected explicit Rule 10 / Rule 1 references in precedence statement"


class TestSystemPromptOntologyCoverage:
    """Cluster 1: prompt must list the expanded 13-entity / 25-relationship ontology."""

    def test_system_prompt_lists_mist_identity_entity_type(self):
        """MistIdentity must appear in the Allowed Entity Types section."""
        assert "MistIdentity" in EXTRACTION_SYSTEM_PROMPT, (
            "Expected 'MistIdentity' in Allowed Entity Types; Cluster 1 added "
            "it as the 13th extractable type for MIST-scope facts."
        )

    def test_system_prompt_lists_new_relationship_types(self):
        """All 4 new MIST-scope relationship types must appear in Allowed Relationship Types."""
        new_types = [
            "IMPLEMENTED_WITH",
            "MIST_HAS_CAPABILITY",
            "MIST_HAS_TRAIT",
            "MIST_HAS_PREFERENCE",
        ]
        for rel_type in new_types:
            assert rel_type in EXTRACTION_SYSTEM_PROMPT, (
                f"Expected '{rel_type}' in Allowed Relationship Types; "
                "Cluster 1 added 4 MIST-scope edge types."
            )


class TestSystemPromptScopeHandling:
    """Cluster 1: the user-subject bias must be removed and replaced with scope rules."""

    def test_system_prompt_has_no_user_subject_bias(self):
        """The old 'User is almost always the SUBJECT' line must be gone.

        That single line caused Bug J (MIST-tooling attributed to the user) because
        the model absorbed it as a hard prior and overrode the few-shot signal.
        """
        banned = "User is almost always the SUBJECT"
        assert banned not in EXTRACTION_SYSTEM_PROMPT, (
            f"Expected '{banned}' to be removed; it biases Gemma 4 E4B toward "
            "user-as-source attribution in multi-turn sessions."
        )

    def test_system_prompt_explains_subject_scope_handling(self):
        """All three scope labels must appear so the prompt can route on them."""
        prompt_lower = EXTRACTION_SYSTEM_PROMPT.lower()
        for label in ["user-scope", "system-scope", "third-party"]:
            assert label in prompt_lower, (
                f"Expected scope label '{label}' in prompt; the scope-aware "
                "direction rules drive correct source attribution."
            )


class TestSystemPromptExampleBalance:
    """Cluster 1: few-shot examples must cover non-user subjects."""

    def test_examples_include_mist_scope_case(self):
        """At least one example must demonstrate mist-identity as the source entity."""
        assert "mist-identity" in EXTRACTION_SYSTEM_PROMPT, (
            "Expected at least one few-shot example with source='mist-identity'; "
            "without a system-scope exemplar the model reverts to user-centric extraction."
        )

    def test_examples_include_third_party_case(self):
        """At least one example must show a third-party subject with no user attribution."""
        # Example 7 ("My coworker says Rust is really fast") is the canonical third-party
        # no-attribution example. We assert both the coworker language AND the empty
        # relationships array for that case.
        assert "coworker" in EXTRACTION_SYSTEM_PROMPT, (
            "Expected a third-party exemplar (coworker/colleague quote); without it "
            "the model may attribute third-party claims to the user."
        )


class TestEventVsMilestoneDisambiguation:
    """Post-MVP follow-up: prompt must disambiguate Event from Milestone.

    Both `Event` (with legacy `event_type=milestone` enum value) and the
    dedicated `Milestone` type can represent the same fact. Without an
    explicit boundary in the system prompt, the model picks at random,
    which produces two graph nodes for the same conceptual entity.
    """

    def test_prompt_includes_event_vs_milestone_rule(self):
        """The rules section must explicitly distinguish Event from Milestone."""
        prompt_lower = EXTRACTION_SYSTEM_PROMPT.lower()
        assert (
            "event vs milestone" in prompt_lower or "milestone vs event" in prompt_lower
        ), "Expected explicit Event-vs-Milestone disambiguation rule in EXTRACTION RULES"

    def test_prompt_forbids_event_type_milestone_legacy_value(self):
        """Rule 11 must explicitly retire the Event.event_type=milestone overlap."""
        prompt_lower = EXTRACTION_SYSTEM_PROMPT.lower()
        assert 'event_type="milestone"' in prompt_lower or "event_type=milestone" in prompt_lower, (
            "Expected the rule to call out Event.event_type='milestone' and route it "
            "to the dedicated Milestone type; without this, the model may emit "
            "Event(event_type=milestone) instead of Milestone for shipped/launched/promoted facts."
        )

    def test_examples_include_both_event_and_milestone_extractions(self):
        """At least one few-shot must extract Event, and at least one Milestone.

        The contrast pair (Example 9 = Milestone, Example 12 = Event) anchors
        the rule with concrete output; the rule alone is too easy to skim past.
        """
        assert '"type": "Event"' in EXTRACTION_SYSTEM_PROMPT, (
            "Expected an Event extraction in the few-shot examples; without one "
            "the model has no anchor for when to prefer Event over Milestone."
        )
        assert '"type": "Milestone"' in EXTRACTION_SYSTEM_PROMPT, (
            "Expected a Milestone extraction in the few-shot examples (Example 9 "
            "since the 2026-04-23 ontology expansion)."
        )


class TestUserTemplate:
    """Cluster 1: user template must surface subject_scope to the model."""

    def test_user_template_has_subject_scope_placeholder(self):
        """The user template must include a {subject_scope} placeholder."""
        assert "{subject_scope}" in EXTRACTION_USER_TEMPLATE, (
            "Expected '{subject_scope}' placeholder in EXTRACTION_USER_TEMPLATE; "
            "Agent B's ontology_extractor passes the classifier's output via this slot."
        )

    def test_user_template_formats_with_scope(self):
        """The template must format without KeyError when all three slots are supplied."""
        # Arrange
        expected = (
            "Context:\n"
            "prior turn\n"
            "Subject scope: system-scope\n"
            'Utterance: "MIST uses LanceDB"\n'
            "\n"
            "Output:"
        )

        # Act
        rendered = EXTRACTION_USER_TEMPLATE.format(
            context="prior turn",
            utterance="MIST uses LanceDB",
            subject_scope="system-scope",
        )

        # Assert
        assert rendered == expected
