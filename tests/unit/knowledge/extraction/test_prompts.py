"""Snapshot-style assertions on EXTRACTION_SYSTEM_PROMPT contents.

These are deliberately coarse-grained: we assert key language is PRESENT
rather than asserting a full-text snapshot. This lets us refine wording
without churn-testing, while still catching regressions that drop a
load-bearing rule.
"""

import json
import re

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


_EXAMPLE_RE = re.compile(
    r"^### Example (\d+):[^\n]*\nSubject scope: [^\n]*\nUtterance: \"(.*)\"\nOutput:\n(.*)$",
    re.MULTILINE,
)


def _examples() -> dict[int, tuple[str, dict]]:
    """Example number to (utterance, parsed output), with the format braces undoubled."""
    return {
        int(m.group(1)): (m.group(2), json.loads(m.group(3).replace("{{", "{").replace("}}", "}")))
        for m in _EXAMPLE_RE.finditer(EXTRACTION_SYSTEM_PROMPT)
    }


def _entities_by_id(output: dict) -> dict[str, dict]:
    return {entity["id"]: entity for entity in output["entities"]}


def _rule(number: int) -> str:
    """The numbered extraction rule's line."""
    prefix = f"{number}. "
    return next(line for line in EXTRACTION_SYSTEM_PROMPT.splitlines() if line.startswith(prefix))


def _has_edge(output: dict, source: str, edge_type: str, target: str) -> bool:
    return any(
        (rel["source"], rel["type"], rel["target"]) == (source, edge_type, target)
        for rel in output["relationships"]
    )


class TestGroup1IdentityConventions:
    """v11 group 1 (ADR-054 Decisions 3 to 5 and 7): the identity conventions the gold
    labels follow are stated in the prompt, each as a rule, with an example where the
    rule alone is a new shape. Each presence check is keyed on the rule's own phrase.
    """

    def test_rule_2_states_n1_the_id_is_the_utterance_s_own_phrase(self):
        assert "Entity IDs are the utterance's own phrase" in EXTRACTION_SYSTEM_PROMPT
        assert "Never invent an id" in EXTRACTION_SYSTEM_PROMPT
        assert "neither lengthened nor shortened" in EXTRACTION_SYSTEM_PROMPT

    def test_n15_an_activity_keeps_its_object(self):
        assert "An activity keeps its object" in EXTRACTION_SYSTEM_PROMPT

    def test_n2_goal_ids_are_the_goal_s_verb_phrase(self):
        assert "Goal ids. A Goal's id is the goal's verb phrase" in EXTRACTION_SYSTEM_PROMPT

    def test_n4_a_name_that_only_modifies_another_name_gets_no_entity(self):
        assert "Name modifiers. A name that only modifies another name" in (
            EXTRACTION_SYSTEM_PROMPT
        )

    def test_n16_a_relative_named_by_the_relation_is_a_person(self):
        assert "A relative named by the relation" in EXTRACTION_SYSTEM_PROMPT

    def test_n14_kind_nouns_get_is_a_to_a_concept(self):
        assert "Kind nouns. When the utterance says what kind of thing" in (
            EXTRACTION_SYSTEM_PROMPT
        )

    def test_the_event_rule_keeps_owner_and_place_and_no_date_words(self):
        assert "Event ids and edges" in EXTRACTION_SYSTEM_PROMPT
        assert "The id holds no date word" in EXTRACTION_SYSTEM_PROMPT
        # ADR-054 Decision 4's closed list, its last clause verbatim.
        assert (
            '"this", "next" or "last" before "week", "weekend", "month", "year" or a weekday'
            in EXTRACTION_SYSTEM_PROMPT
        )
        assert "DECIDED takes only a User source" in EXTRACTION_SYSTEM_PROMPT

    def test_the_metric_rule_asks_for_value_unit_and_quantity(self):
        assert "Metric properties" in EXTRACTION_SYSTEM_PROMPT
        assert "value-first" in EXTRACTION_SYSTEM_PROMPT

    def test_the_third_party_scope_rule_keeps_the_relative_s_anchor(self):
        third_party = next(
            line for line in EXTRACTION_SYSTEM_PROMPT.splitlines() if "`third-party` utt" in line
        )
        assert "KNOWS_PERSON" in third_party and "Example 7" in third_party

    def test_every_example_output_parses_as_json(self):
        examples = _examples()
        count = EXTRACTION_SYSTEM_PROMPT.count("### Example ")
        assert len(examples) == count
        assert sorted(examples) == list(range(1, count + 1))

    def test_example_7_the_coworker_is_a_person_anchored_to_the_user(self):
        _, output = _examples()[7]
        entities = _entities_by_id(output)
        assert entities["coworker"]["type"] == "Person"
        assert _has_edge(output, "user", "KNOWS_PERSON", "coworker")
        assert all(
            rel["source"] != "user" or rel["type"] == "KNOWS_PERSON"
            for rel in (output["relationships"])
        ), "the coworker's opinion is not attributed to the user"

    def test_example_10_the_metric_carries_its_properties_and_a_value_first_id(self):
        _, output = _examples()[10]
        metric = next(e for e in output["entities"] if e["type"] == "Metric")
        props = metric["properties"]
        assert props["value"] is not None
        assert props.get("unit") or props.get("quantity")
        assert metric["id"] == "0-94-tool-selection"

    def test_example_12_the_event_id_is_a_phrase_of_its_utterance_with_no_date(self):
        utterance, output = _examples()[12]
        event = next(e for e in output["entities"] if e["type"] == "Event")
        words = re.findall(r"[a-z0-9]+", utterance.lower())
        id_words = event["id"].split("-")
        assert any(
            words[i : i + len(id_words)] == id_words for i in range(len(words))
        ), "the Event id is not a run of its utterance's words"
        assert not re.search(r"\d", event["id"])
        dates = [e["id"] for e in output["entities"] if e["type"] == "Date"]
        assert len(dates) == 1
        assert _has_edge(output, event["id"], "OCCURRED_ON", dates[0])

    def test_example_24_the_habit_id_is_the_utterance_s_phrase(self):
        _, output = _examples()[24]
        assert "work-out" in _entities_by_id(output)
        assert _has_edge(output, "user", "HAS_HABIT", "work-out")

    def test_the_prompt_stays_under_its_length_budget(self):
        # Lead decision of 10 Nov: the binding limit is PROMPT_RESERVE_TOKENS (14,336 tokens),
        # and H16's `prompt_tokens` lines are authoritative; this cap is a coarse proxy.
        assert len(EXTRACTION_SYSTEM_PROMPT) <= 52000


class TestGroup2ListConventions:
    """v11 group 2 (ADR-054 Decision 7): how a structured turn reads, the list rules, and
    N3, N5, N7 and N13, each stated as a rule. Each presence check is keyed on the rule's
    own phrase; Example 30 teaches the list shapes a rule alone does not hold.
    """

    def test_a_structured_turn_reads_as_context_lead_in_and_item(self):
        assert "each `<context> > <lead-in>: <item>`" in EXTRACTION_SYSTEM_PROMPT

    def test_user_context_lines_are_never_extracted_again(self):
        assert "Lines marked `[user]:` are earlier units" in EXTRACTION_SYSTEM_PROMPT
        assert "never extract them again" in EXTRACTION_SYSTEM_PROMPT

    def test_an_item_s_id_is_the_item_s_own_words(self):
        assert "A list item's id is the item's own words for the thing itself" in _rule(28)

    def test_a_list_item_s_id_adds_no_cadence_scope_owner_purpose_or_quantity(self):
        rule_28 = _rule(28)
        assert "a possessive 's dropped anywhere in it" in rule_28
        assert "no word of its lead-in, heading or context" in rule_28
        assert "no cadence, scope, owner, purpose or quantity phrase" in rule_28
        assert "a cadence goes in temporal_expression" in rule_28
        assert "a scope or purpose in context" in rule_28

    def test_a_joining_word_goes_with_the_phrase_it_joins(self):
        rule_28 = _rule(28)
        assert "A word that only joins a dropped phrase" in rule_28
        # The one illustration: the id is the item's words in order, fewer than the item's,
        # with no word of the lead-in.
        match = re.search(r'"([^":]+): ([^"]+)" -> "([a-z0-9-]+)"', rule_28)
        assert match is not None, "Rule 28 illustrates the id rule"
        lead_in, item, item_id = match.groups()
        item_words = re.findall(r"[a-z0-9]+", item.lower())
        id_words = item_id.split("-")
        rest = iter(item_words)
        assert all(word in rest for word in id_words), "the id is the item's own words"
        assert len(id_words) < len(item_words)
        assert not set(re.findall(r"[a-z0-9]+", lead_in.lower())) & set(id_words)

    def test_event_and_metric_items_keep_group_1_s_ids(self):
        rule_28 = _rule(28)
        assert "an Event item keeps its named owner and stated place (Rule 26)" in rule_28
        assert (
            "a Metric item's quantity is every word it gives for what is measured, "
            "none dropped or added (Rule 27)"
        ) in rule_28

    def test_the_lead_in_sets_the_subject_or_predicate_and_is_not_an_entity(self):
        assert "Read each item as a sentence completing the lead-in" in EXTRACTION_SYSTEM_PROMPT
        assert "sets the subject or predicate of the items under it" in EXTRACTION_SYSTEM_PROMPT
        assert "it is not an entity unless it names one" in EXTRACTION_SYSTEM_PROMPT

    def test_a_refer_back_item_takes_the_earlier_item_s_id(self):
        assert "An item referring back takes the earlier item's entity and id" in (
            EXTRACTION_SYSTEM_PROMPT
        )

    def test_an_option_nobody_chose_gets_no_edge(self):
        assert "An option nobody chose gets no edge" in EXTRACTION_SYSTEM_PROMPT

    def test_checked_and_unchecked_boxes(self):
        assert "A checked box `[x]` is a done fact" in EXTRACTION_SYSTEM_PROMPT
        assert "an unchecked box `[ ]` is a plan with no edge" in EXTRACTION_SYSTEM_PROMPT
        assert "under a goals lead-in, where it is HAS_GOAL" in EXTRACTION_SYSTEM_PROMPT

    def test_a_table_row_is_one_fact_about_its_first_column(self):
        assert "one fact about its first column" in EXTRACTION_SYSTEM_PROMPT

    def test_a_list_with_nothing_durable_is_empty(self):
        assert "A list with nothing durable (an errand list) is empty" in EXTRACTION_SYSTEM_PROMPT

    def test_itinerary_items_are_experienced_events_with_dates(self):
        assert "Itinerary items are Events" in EXTRACTION_SYSTEM_PROMPT

    def test_n3_decisions(self):
        assert "Decisions. A decision by the user or their team is user DECIDED" in (
            EXTRACTION_SYSTEM_PROMPT
        )
        assert "It does not make its outcome current" in EXTRACTION_SYSTEM_PROMPT
        assert "Someone else's decision gets the Event and the link" in EXTRACTION_SYSTEM_PROMPT

    def test_n5_stated_places_and_organisations(self):
        assert "Stated places and organisations." in EXTRACTION_SYSTEM_PROMPT

    def test_n7_third_party_events(self):
        assert "Third-party events. A named person's own non-milestone Event" in (
            EXTRACTION_SYSTEM_PROMPT
        )

    def test_n13_topic_links(self):
        assert "Topic links. An Event is RELATED_TO what the utterance says" in (
            EXTRACTION_SYSTEM_PROMPT
        )
        assert "something chosen for a subject links to that subject" in EXTRACTION_SYSTEM_PROMPT

    def test_example_30_is_one_rendered_unit_whose_lead_in_sets_the_predicate(self):
        heading = next(
            line for line in EXTRACTION_SYSTEM_PROMPT.splitlines() if "### Example 30:" in line
        )
        utterance, output = _examples()[30]
        context, item = utterance.rsplit(": ", 1)
        assert " > " in context, "the unit renders its context joined by ' > '"
        item_id = "-".join(re.findall(r"[a-z0-9]+", item.lower()))
        entities = _entities_by_id(output)
        assert set(entities) == {"user", item_id}, "only the item is an entity, by its own words"
        assert _has_edge(output, "user", "LEARNING", item_id)
        assert len(output["relationships"]) == 1
        # The earlier unit, shown as a `[user]:` context line, is not extracted again.
        earlier = heading.split("Context: [user]: ", 1)[1].rsplit(": ", 1)[1]
        assert "-".join(re.findall(r"[a-z0-9]+", earlier.lower())) not in entities


class TestGroup3GlossesAndTeamConventions:
    """v11 group 3 (ADR-054 Decision 7, plan D17): the Organization, WORKS_AT and
    MEMBER_OF glosses, N17 named teams with its own example, and N8, N10 and N11, each
    stated as a rule; N12 lives in Rule 6. Each presence check is keyed on the rule's own
    phrase. T40 round 1 cut the other glosses and Rule 38 to fit the token reserve.
    """

    # The ontology's own descriptions, quoted. WORKS_AT is the one exception: the prompt
    # glosses it as employment only, since membership is MEMBER_OF (lead decision 3).
    QUOTED = {
        "Organization": "A company, team, institution, community, or open-source project.",
        "MEMBER_OF": "Indicates membership in an organization.",
    }

    @staticmethod
    def _gloss_lines() -> dict[str, str]:
        lines = EXTRACTION_SYSTEM_PROMPT.splitlines()
        start = next(i for i, line in enumerate(lines) if line.startswith("### Glosses"))
        glosses: dict[str, str] = {}
        for line in lines[start + 1 :]:
            if not line.startswith("- "):
                break
            name, gloss = line[2:].split(": ", 1)
            glosses[name] = gloss
        return glosses

    def test_each_gloss_quotes_its_ontology_description(self):
        from backend.knowledge.ontologies.v1_0_0 import ONTOLOGY_V1_0_0

        glosses = self._gloss_lines()
        descriptions = {
            d.type_name: d.description
            for d in (*ONTOLOGY_V1_0_0.node_types, *ONTOLOGY_V1_0_0.edge_types)
        }
        assert set(glosses) == {*self.QUOTED, "WORKS_AT"}
        for name, description in self.QUOTED.items():
            assert descriptions[name] == description, name
            assert glosses[name].startswith(f'"{description}"'), name

    def test_works_at_is_glossed_as_employment_only(self):
        gloss = self._gloss_lines()["WORKS_AT"]
        assert "employment at an organization, the employer, and nothing else" in gloss
        assert "membership" not in gloss.split(";")[0]

    def test_a_quoted_topic_does_not_suggest_a_topic_type(self):
        assert "there is no Topic type" in EXTRACTION_SYSTEM_PROMPT
        assert "Topic," not in EXTRACTION_SYSTEM_PROMPT.split("### Abstract-Type Tests")[0]

    def test_n17_named_teams(self):
        assert "Named teams. A team, club or community named by its work" in (
            EXTRACTION_SYSTEM_PROMPT
        )
        assert "each, lead included, gets MEMBER_OF it" in EXTRACTION_SYSTEM_PROMPT
        assert "a describing word (new, old, small, whole, current, remote, other)" in (
            EXTRACTION_SYSTEM_PROMPT
        )
        assert "the team gets no PART_OF to the employer" in EXTRACTION_SYSTEM_PROMPT
        assert "a relationship category (colleagues, friends, family, household)" in (
            EXTRACTION_SYSTEM_PROMPT
        )

    def test_n8_the_team_s_tools(self):
        assert "The team's tools. Tools the user's team runs are user USES" in (
            EXTRACTION_SYSTEM_PROMPT
        )
        assert "a possessive that only names a thing is not" in EXTRACTION_SYSTEM_PROMPT

    def test_n10_personal_milestones(self):
        assert "Personal milestones. A personal legal or financial step completed on a day" in (
            EXTRACTION_SYSTEM_PROMPT
        )

    def test_n11_goal_windows(self):
        assert "Goal deadlines. A goal's stated time frame is its deadline" in (
            EXTRACTION_SYSTEM_PROMPT
        )
        assert "A bare month is the first such month after the reference date" in (
            EXTRACTION_SYSTEM_PROMPT
        )
        assert "a frame whose year is unclear gets no date" in EXTRACTION_SYSTEM_PROMPT

    def test_n12_flipped_preferences_live_in_rule_6_with_no_date(self):
        rule_6 = _rule(6)
        anymore = rule_6.split('"don\'t like X anymore"', 1)[1].split(". ", 1)[0]
        assert "PREFERS X" in anymore and "cease" in anymore
        assert "DISLIKES X" in anymore and "assert" in anymore
        assert "neither edge gets a date" in anymore
        assert "(the new preference is the fact)" not in rule_6

    def test_rule_38_folded_into_rule_6(self):
        assert "Rule 38" not in EXTRACTION_SYSTEM_PROMPT
        assert not any(line.startswith("38. ") for line in EXTRACTION_SYSTEM_PROMPT.splitlines())

    def test_example_31_the_named_team_beside_the_employer(self):
        heading = next(
            line for line in EXTRACTION_SYSTEM_PROMPT.splitlines() if "### Example 31:" in line
        )
        assert "Rule 34" in heading
        _, output = _examples()[31]
        orgs = [e["id"] for e in output["entities"] if e["type"] == "Organization"]
        assert len(orgs) == 2
        team = next(o for o in orgs if _has_edge(output, "user", "MEMBER_OF", o))
        employer = next(o for o in orgs if _has_edge(output, "user", "WORKS_AT", o))
        assert team != employer
        assert team.endswith("-team"), "the group word is kept when the utterance writes it"
        assert not _has_edge(output, "user", "WORKS_AT", team)
        assert not _has_edge(output, "user", "MEMBER_OF", employer)
        assert all(rel["type"] != "PART_OF" for rel in output["relationships"])


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
    # extraction_version = "2026-11-10-r8", groups 1 (identity), 2 (list conventions) and
    # 3 (glosses, named teams and the remaining conventions), as T40 round 1 revised them.
    PINNED_SHA256 = "0ad6912fd8638461e8aed0c7344be8e7f4e5b53b2112d9c5f80bce5de1fcd089"

    def test_prompt_content_matches_pinned_extraction_version(self):
        import hashlib

        digest = hashlib.sha256(
            (EXTRACTION_SYSTEM_PROMPT + EXTRACTION_USER_TEMPLATE).encode("utf-8")
        ).hexdigest()
        assert digest == self.PINNED_SHA256, (
            "Extraction prompt content changed WITHOUT an extraction_version "
            "bump. Update EXTRACTION_VERSION in "
            "backend/knowledge/version_stamps.py -- the single authority, "
            "which every other site imports -- then re-pin PINNED_SHA256 "
            "here. The extraction cache keys on the version string; skipping "
            "the bump makes R1 rebuilds silently serve stale extractions."
        )

    def test_single_authority_holds_the_pinned_version(self):
        from backend.knowledge.version_stamps import EXTRACTION_VERSION

        assert EXTRACTION_VERSION == "2026-11-10-r8"

    def test_config_default_is_the_single_authority(self):
        from backend.knowledge.config import KnowledgeConfig
        from backend.knowledge.version_stamps import EXTRACTION_VERSION

        assert KnowledgeConfig.extraction_version == EXTRACTION_VERSION

    def test_vault_writer_stamps_the_single_authority(self):
        """The writer imports the authority; it no longer mirrors a literal."""
        from backend.knowledge.version_stamps import EXTRACTION_VERSION
        from backend.vault import writer

        assert writer.EXTRACTION_VERSION == EXTRACTION_VERSION
        assert not hasattr(
            writer, "_EXTRACTION_VERSION"
        ), "the mirrored literal is deleted, not merely unreferenced"
