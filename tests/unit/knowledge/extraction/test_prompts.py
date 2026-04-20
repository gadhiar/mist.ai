"""Snapshot-style assertions on EXTRACTION_SYSTEM_PROMPT contents.

These are deliberately coarse-grained: we assert key language is PRESENT
rather than asserting a full-text snapshot. This lets us refine wording
without churn-testing, while still catching regressions that drop a
load-bearing rule.
"""

from backend.knowledge.extraction.prompts import EXTRACTION_SYSTEM_PROMPT


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
