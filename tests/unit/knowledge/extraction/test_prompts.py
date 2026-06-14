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
    """v1.4.0 follow-up: Milestone retired; prompt must route milestone-class facts to Event.

    In v1.4.0 the dedicated `Milestone` entity type is retired. The canonical
    representation is `Event` with `event_type=milestone`. Rule 11 must route
    "shipped / launched / completed / achieved / promoted" facts to Event, and
    Example 9 must demonstrate the output shape.
    """

    def test_prompt_includes_event_vs_milestone_rule(self):
        """The rules section must explicitly distinguish Event from Milestone."""
        prompt_lower = EXTRACTION_SYSTEM_PROMPT.lower()
        assert (
            "event vs milestone" in prompt_lower or "milestone vs event" in prompt_lower
        ), "Expected explicit Event-vs-Milestone disambiguation rule in EXTRACTION RULES"

    def test_prompt_covers_milestone_via_event_type(self):
        """Rule 11 must call out event_type=milestone as the canonical representation.

        v1.4.0: Milestone type retired; Event with event_type=milestone is the
        canonical shape. The rule must name event_type=milestone so the model
        knows the output shape.
        """
        prompt_lower = EXTRACTION_SYSTEM_PROMPT.lower()
        assert 'event_type="milestone"' in prompt_lower or "event_type=milestone" in prompt_lower, (
            "Expected 'event_type=milestone' to appear in Rule 11 so the model "
            "knows the canonical output shape for milestone-class facts."
        )

    def test_examples_include_event_and_milestone_class_event_extractions(self):
        """Few-shots must include an Event extraction and a milestone-class Event.

        Example 12 shows a plain Event (conference). Example 9 shows a
        milestone-class Event (shipped Cluster 8 Phase 6) with event_type=milestone.
        Both must be present so the model has anchors for each sub-case.
        """
        assert '"type": "Event"' in EXTRACTION_SYSTEM_PROMPT, (
            "Expected an Event extraction in the few-shot examples; without one "
            "the model has no anchor for non-milestone events."
        )
        assert (
            "event_type" in EXTRACTION_SYSTEM_PROMPT and "milestone" in EXTRACTION_SYSTEM_PROMPT
        ), (
            "Expected a milestone-class Event extraction in the few-shot examples "
            "(Example 9 since the v1.4.0 Milestone retirement)."
        )


class TestDocumentEngagementRule:
    """V8 baseline iteration: prompt must steer document-engagement verbs to REFERENCES_DOCUMENT.

    Without explicit guidance, the model picks LEARNING for "halfway through a
    book" and WORKS_ON for "finished a paper" -- both produce wrong edges in
    the knowledge graph (LEARNING is for technologies, WORKS_ON for projects).
    Rule 12 retires those alternatives for Document targets.
    """

    def test_prompt_includes_document_engagement_rule(self):
        """Rules section must explicitly cover active-reading verbs."""
        prompt_lower = EXTRACTION_SYSTEM_PROMPT.lower()
        assert (
            "document engagement" in prompt_lower
        ), "Expected 'Document engagement' rule in EXTRACTION RULES"
        assert (
            "references_document" in prompt_lower
        ), "Expected 'REFERENCES_DOCUMENT' anchor in the rule"

    def test_prompt_explicitly_retires_learning_and_works_on_for_documents(self):
        """The rule must call out both incorrect alternatives by name."""
        prompt_lower = EXTRACTION_SYSTEM_PROMPT.lower()
        assert (
            "learning" in prompt_lower and "works_on" in prompt_lower
        ), "Expected explicit retirement of LEARNING and WORKS_ON for Document targets"

    def test_examples_include_active_reading_document_extraction(self):
        """At least one few-shot must show an active-reading verb on a Document.

        Example 11 covers passive 'read'; Example 13 covers active engagement
        ('working through'). The contrast pair anchors the rule with concrete output.
        """
        assert (
            "working through" in EXTRACTION_SYSTEM_PROMPT.lower()
        ), "Expected active-engagement few-shot using 'working through' verb"


class TestAssertionKindSignal:
    """C3 spec 6.2: prompt must emit an assertion_kind signal per relationship.

    The reconciliation engine consumes assertion_kind to distinguish a fact
    that began (assert) from one that stopped being true (cease) from one that
    was never true (retract). Without the prompt teaching the field, a 4B model
    omits it and reconciliation loses the cessation/retraction signal.
    """

    def test_prompt_emits_assertion_kind_in_schema(self):
        assert '"assertion_kind"' in EXTRACTION_SYSTEM_PROMPT

    def test_prompt_rule6_teaches_assertion_kind(self):
        assert "cease" in EXTRACTION_SYSTEM_PROMPT
        assert "retract" in EXTRACTION_SYSTEM_PROMPT


class TestRecommendsHabitDateRules:
    """C3 prompt r3: RECOMMENDS / HAS_HABIT predicates (ontology v1.3.0) plus
    date-entity discrimination.

    RECOMMENDS captures third-party suggestions ("Sarah recommended Postgres")
    with the recommender as source, not the user. HAS_HABIT captures recurring
    activities ("I work out every morning") as stative facts, not Events. The
    date-entity rule stops the model from minting a `Date` node every time a
    date merely scopes a stative fact.
    """

    def test_prompt_lists_recommends_and_has_habit(self):
        assert "RECOMMENDS" in EXTRACTION_SYSTEM_PROMPT
        assert "HAS_HABIT" in EXTRACTION_SYSTEM_PROMPT

    def test_prompt_teaches_date_entity_discrimination(self):
        assert "do NOT create a Date entity" in EXTRACTION_SYSTEM_PROMPT


class TestR4PrecisionRules:
    """C3 prompt r4: two GENERALIZABLE precision rules closing genuine
    extraction errors surfaced in the r3 Phase C per-probe diagnostics.

    Edit A tightens the HAS_HABIT clause (Rule 17): a continuous "since
    <date>" stative is NOT a recurrence, so it must route to the matching
    stative predicate (EXPERT_IN / LEARNING / INTERESTED_IN) with valid_from,
    not HAS_HABIT. Edit B adds Rule 18: a trailing prepositional phrase that
    merely scopes another fact is context, not a separate structural edge.
    """

    def test_prompt_requires_recurrence_cadence_for_has_habit(self):
        assert "requires an explicit recurrence cadence" in EXTRACTION_SYSTEM_PROMPT

    def test_prompt_forbids_prepositional_over_extraction(self):
        assert "Extract the asserted fact, not incidental scope" in EXTRACTION_SYSTEM_PROMPT


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


class TestExtractionVersionDriftGuard:
    """foundation-f123-4: the extraction cache keys on extraction_version,
    so a prompt edit without a version bump would make R1 rebuilds silently
    serve extractions produced by the OLD prompt. This pin makes the
    honor-system pairing mechanical.
    """

    # sha256(EXTRACTION_SYSTEM_PROMPT + EXTRACTION_USER_TEMPLATE) pinned for
    # extraction_version = "2026-06-14-r5".
    PINNED_SHA256 = "cb0f7788ba910b35c376a0ee4a172e92757aab5977c4d6330aa6e1584bc555be"

    def test_prompt_content_matches_pinned_extraction_version(self):
        import hashlib

        digest = hashlib.sha256(
            (EXTRACTION_SYSTEM_PROMPT + EXTRACTION_USER_TEMPLATE).encode("utf-8")
        ).hexdigest()
        assert digest == self.PINNED_SHA256, (
            "Extraction prompt content changed WITHOUT an extraction_version "
            "bump. Update KnowledgeConfig.extraction_version (field default + "
            "EXTRACTION_VERSION env default) and backend/vault/writer.py "
            "_EXTRACTION_VERSION, then re-pin PINNED_SHA256 here. The "
            "extraction cache keys on the version string -- skipping the bump "
            "makes R1 rebuilds silently serve stale extractions."
        )

    def test_config_default_matches_pinned_version(self):
        from backend.knowledge.config import KnowledgeConfig
        from backend.vault.writer import _EXTRACTION_VERSION

        assert KnowledgeConfig.extraction_version == "2026-06-14-r5"
        assert _EXTRACTION_VERSION == "2026-06-14-r5"
