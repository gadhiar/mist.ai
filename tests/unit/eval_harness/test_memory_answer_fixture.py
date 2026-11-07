"""The answer probes and noise facts are well formed, tied to the recall fixture and big enough.

H5 seeds `noise_facts.jsonl` beside the recall fixture and scores `answer_probes.jsonl` with
`scripts/eval_harness/memory_answers.py`. A probe naming a fact that does not exist, a gold answer
its own scorer marks wrong, or a canary that also sits in a clean note would each move H5's gate
without the model changing. This file checks, against `ANSWER_PROBES.md`:

- every row of both files against its schema, through the loaders, whose strict parsing and kind
  rules are themselves tested here;
- the minimum counts of plan task T27: 150 noise facts, 60 probes, 8 of each kind;
- the noise facts: hashed keys, extractable ontology types consistent with the clean facts, low
  confidence above the extraction threshold, a real source turn, no clash with a clean fact and
  no corrected object;
- referential integrity: every item a probe names exists in the recall fixture, the noise file
  or a seed note, and is available and not suppressed at the probe's `now`;
- the authoring rules of each kind, and that a run made of the gold answers scores perfectly.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.knowledge.ontologies.v1_0_0 import (  # noqa: E402
    EDGE_TYPES_BY_NAME,
    EXTRACTABLE_NODE_TYPES,
    EXTRACTABLE_RELATIONSHIP_TYPES,
    EXTRACTION_RULES,
)
from scripts.eval_harness.memory_answers import (  # noqa: E402
    ANSWER_KINDS,
    MEMORY_TOOLS,
    NOISE_KINDS,
    AnswerProbe,
    AnswerRecord,
    NoiseFact,
    ToolCall,
    contains_term,
    is_correct,
    load_answer_probes,
    load_noise_facts,
    misattributes,
    parse_answer_probe,
    parse_noise_fact,
    score_answers,
)
from scripts.eval_harness.memory_recall import (  # noqa: E402
    FIXTURE_DIR,
    Correction,
    Episode,
    Fact,
    ItemRef,
    Turn,
    VaultNote,
    load_corrections,
    load_episodes,
    load_facts,
    load_turns,
    load_vault_notes,
)

# Plan task T27, Acceptance.
NOISE_MINIMUM = 150
NOISE_KIND_MINIMUM = 40
PROBE_MINIMUM = 60
PROBE_KIND_MINIMUM = 8
# A noise fact sits at or above the extractor's threshold, so it would have been stored, and at
# or below this ceiling, below every clean current fact (0.9 or 1.0).
NOISE_CONFIDENCE_CEILING = 0.6
CONVERSATION_GAP = timedelta(minutes=30)
PROBE_ID = re.compile(r"ans-(?P<kind>[a-z-]+)-\d{2}")


@dataclass(frozen=True, slots=True)
class Fixture:
    facts: tuple[Fact, ...]
    noise: tuple[NoiseFact, ...]
    turns: tuple[Turn, ...]
    episodes: tuple[Episode, ...]
    notes: tuple[VaultNote, ...]
    corrections: tuple[Correction, ...]
    probes: tuple[AnswerProbe, ...]

    @property
    def history_end(self) -> datetime:
        return max(turn.timestamp for turn in self.turns)

    @property
    def seed_notes(self) -> tuple[VaultNote, ...]:
        return tuple(p.seed_note for p in self.probes if p.seed_note is not None)


@pytest.fixture(scope="module")
def fixture() -> Fixture:
    return Fixture(
        facts=load_facts(),
        noise=load_noise_facts(),
        turns=load_turns(),
        episodes=load_episodes(),
        notes=load_vault_notes(),
        corrections=load_corrections(),
        probes=load_answer_probes(),
    )


def _rows(name: str) -> list[dict[str, object]]:
    lines = (FIXTURE_DIR / name).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _row_of_kind(kind: str) -> dict[str, object]:
    return next(row for row in _rows("answer_probes.jsonl") if row["kind"] == kind)


def _item_texts(fixture: Fixture) -> dict[ItemRef, str]:
    """The searchable text of every item a probe can name, keyed by its reference."""
    texts: dict[ItemRef, str] = {}
    for fact in (*fixture.facts, *(n.fact for n in fixture.noise)):
        ref = ItemRef(kind="fact", id=fact.fact_key)
        texts[ref] = texts.get(ref, "") + f" {fact.subject_name} {fact.object_name} {fact.text}"
    for turn in fixture.turns:
        texts[ItemRef(kind="exchange", id=turn.event_id)] = (
            f"{turn.user_utterance} {turn.system_response}"
        )
    for episode in fixture.episodes:
        texts[ItemRef(kind="episode", id=episode.episode_id)] = f"{episode.title} {episode.summary}"
    for note in (*fixture.notes, *fixture.seed_notes):
        texts[ItemRef(kind="note", id=note.path)] = f"{note.title} {note.heading} {note.body}"
    return texts


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

PARSERS: list[tuple[str, Callable[[Mapping[str, object]], object]]] = [
    ("noise_facts.jsonl", parse_noise_fact),
    ("answer_probes.jsonl", parse_answer_probe),
]


class TestSchemas:
    def test_both_files_load_through_their_schemas(self, fixture: Fixture) -> None:
        assert fixture.noise and fixture.probes

    @pytest.mark.parametrize(("name", "parse"), PARSERS, ids=[name for name, _ in PARSERS])
    def test_a_row_with_an_unknown_key_is_refused(
        self, name: str, parse: Callable[[Mapping[str, object]], object]
    ) -> None:
        row = _rows(name)[0]
        row["unexpected_key"] = "x"

        with pytest.raises(ValueError, match="unexpected_key"):
            parse(row)

    @pytest.mark.parametrize(("name", "parse"), PARSERS, ids=[name for name, _ in PARSERS])
    def test_a_row_missing_any_key_is_refused(
        self, name: str, parse: Callable[[Mapping[str, object]], object]
    ) -> None:
        """Every key is required, nullable ones included, so a dropped key is never a default."""
        row = _rows(name)[0]
        for key in list(row):
            partial = {k: v for k, v in row.items() if k != key}

            with pytest.raises(ValueError, match=key):
                parse(partial)

    @pytest.mark.parametrize(
        ("kind", "key", "value"),
        [
            ("negative_control", "expect_tool", None),
            ("negative_control", "expected_tools", ["query_vault"]),
            ("negative_control", "distractor_terms", []),
            ("block_insufficient", "expect_tool", None),
            ("block_insufficient", "expected_tools", []),
            ("block_insufficient", "expected_tools", ["frontend.switch_form"]),
            ("recall", "expect_tool", True),
            ("recall", "answer_refs", []),
            ("injection", "canary", None),
            ("injection", "canary", "canary-lowercase"),
            ("injection", "seed_note", None),
            ("misattribution", "misattributed_terms", []),
            ("distractor", "distractor_refs", []),
            ("recall", "required_terms", []),
            ("recall", "required_terms", [[]]),
        ],
    )
    def test_a_probe_that_contradicts_its_kind_is_refused(
        self, kind: str, key: str, value: object
    ) -> None:
        row = _row_of_kind(kind)
        row[key] = value

        with pytest.raises(ValueError):
            parse_answer_probe(row)

    def test_a_noise_fact_with_an_unknown_noise_kind_is_refused(self) -> None:
        row = _rows("noise_facts.jsonl")[0]
        row["noise_kind"] = "typo"

        with pytest.raises(ValueError, match="noise_kind"):
            parse_noise_fact(row)

    def test_ids_canaries_and_seed_paths_are_unique(self, fixture: Fixture) -> None:
        for values in (
            [n.fact.row_key for n in fixture.noise],
            [p.probe_id for p in fixture.probes],
            [p.canary for p in fixture.probes if p.canary is not None],
            [note.path for note in fixture.seed_notes],
        ):
            assert [v for v, count in Counter(values).items() if count > 1] == []

    def test_probe_ids_name_their_kind(self, fixture: Fixture) -> None:
        for probe in fixture.probes:
            match = PROBE_ID.fullmatch(probe.probe_id)
            assert match is not None, probe.probe_id
            assert match["kind"] == probe.kind.replace("_", "-"), probe.probe_id


# ---------------------------------------------------------------------------
# Counts (plan task T27, Acceptance)
# ---------------------------------------------------------------------------


class TestMinimumCounts:
    def test_noise_facts(self, fixture: Fixture) -> None:
        kinds = Counter(noise.noise_kind for noise in fixture.noise)

        assert len(fixture.noise) >= NOISE_MINIMUM
        assert set(kinds) == set(NOISE_KINDS)
        assert min(kinds.values()) >= NOISE_KIND_MINIMUM

    def test_probes_cover_every_kind(self, fixture: Fixture) -> None:
        kinds = Counter(probe.kind for probe in fixture.probes)

        assert len(fixture.probes) >= PROBE_MINIMUM
        assert set(kinds) == set(ANSWER_KINDS)
        assert min(kinds.values()) >= PROBE_KIND_MINIMUM


# ---------------------------------------------------------------------------
# Noise facts
# ---------------------------------------------------------------------------


class TestNoiseFacts:
    def test_row_keys_follow_the_noise_pattern(self, fixture: Fixture) -> None:
        """`nf-` keeps a noise row apart from a clean `fv-` row in any trace."""
        for noise in fixture.noise:
            assert re.fullmatch(r"nf-\d{3}", noise.fact.row_key), noise.fact.row_key

    def test_fact_key_is_the_hash_of_the_triple(self, fixture: Fixture) -> None:
        for noise in fixture.noise:
            fact = noise.fact
            triple = f"{fact.subject_id}|{fact.relation}|{fact.object_id}"
            assert fact.fact_key == hashlib.sha256(triple.encode()).hexdigest(), fact.row_key

    def test_no_noise_fact_shares_a_key_with_a_clean_fact(self, fixture: Fixture) -> None:
        """A shared key would make a noise row a version of a clean fact, not a separate item."""
        clean = {fact.fact_key for fact in fixture.facts}
        noise = [n.fact.fact_key for n in fixture.noise]

        assert clean.isdisjoint(noise)
        assert len(set(noise)) == len(noise)

    def test_relations_and_types_are_extractable_ontology_types(self, fixture: Fixture) -> None:
        for noise in fixture.noise:
            fact = noise.fact
            assert fact.relation in EXTRACTABLE_RELATIONSHIP_TYPES, fact.row_key
            assert EDGE_TYPES_BY_NAME[fact.relation].directional, fact.row_key
            assert fact.subject_type in EXTRACTABLE_NODE_TYPES, fact.row_key
            assert fact.object_type in EXTRACTABLE_NODE_TYPES, fact.row_key

    def test_an_entity_has_one_type_across_clean_and_noise_facts(self, fixture: Fixture) -> None:
        """A noise row that retyped Wren or Biscuit would split one node into two in the graph."""
        seen: dict[str, str] = {}
        for fact in (*fixture.facts, *(n.fact for n in fixture.noise)):
            for entity_id, node_type in (
                (fact.subject_id, fact.subject_type),
                (fact.object_id, fact.object_type),
            ):
                assert seen.setdefault(entity_id, node_type) == node_type, (
                    fact.row_key,
                    entity_id,
                )

    def test_confidence_is_low_but_above_the_extraction_threshold(self, fixture: Fixture) -> None:
        """Below the threshold the extractor drops a fact, so it could never pollute the graph."""
        floor = EXTRACTION_RULES.min_confidence_threshold
        for noise in fixture.noise:
            assert floor <= noise.fact.confidence <= NOISE_CONFIDENCE_CEILING, noise.fact.row_key

    def test_every_noise_fact_comes_from_a_real_earlier_turn(self, fixture: Fixture) -> None:
        turns = {turn.event_id: turn for turn in fixture.turns}
        for noise in fixture.noise:
            fact = noise.fact
            assert fact.source_utterance_id is not None, fact.row_key
            turn = turns[fact.source_utterance_id]
            assert turn.timestamp <= fact.recorded_at <= fixture.history_end, fact.row_key
            assert fact.valid_from is not None and fact.valid_from <= fact.recorded_at

    def test_noise_facts_are_current_open_rows(self, fixture: Fixture) -> None:
        """Extraction stored them as fact; nothing superseded, retracted or corrected them."""
        for noise in fixture.noise:
            fact = noise.fact
            assert fact.state == "current", fact.row_key
            assert fact.valid_to is None and fact.superseded_by is None, fact.row_key
            assert fact.source_correction_id is None, fact.row_key

    def test_no_noise_fact_names_a_corrected_object(self, fixture: Fixture) -> None:
        """A noise row naming a forgotten object would change the recall fixture's suppression."""
        corrected = [correction.object_name for correction in fixture.corrections]
        for noise in fixture.noise:
            text = f"{noise.fact.object_name} {noise.fact.text}"
            assert not [name for name in corrected if contains_term(text, name)], noise.fact.row_key

    def test_a_misattributed_relation_moves_a_clean_object_or_states_the_assistant(
        self, fixture: Fixture
    ) -> None:
        """Each one either gives a clean fact's object to another subject, or comes from a turn
        whose assistant half, not its user half, says it.
        """
        clean = {(f.subject_id, f.relation, f.object_id) for f in fixture.facts}
        clean_objects = {f.object_id for f in fixture.facts}
        turns = {turn.event_id: turn for turn in fixture.turns}
        for noise in fixture.noise:
            if noise.noise_kind != "misattributed_relation":
                continue
            fact = noise.fact
            assert (fact.subject_id, fact.relation, fact.object_id) not in clean, fact.row_key
            if fact.object_id in clean_objects:
                continue
            turn = turns[fact.source_utterance_id or ""]
            words = [w for w in re.findall(r"[a-z0-9']+", fact.object_name.lower()) if len(w) > 3]
            assert any(w in turn.system_response.lower() for w in words), fact.row_key


# ---------------------------------------------------------------------------
# Probe references
# ---------------------------------------------------------------------------


def _valid_at(fact: Fact, now: datetime) -> bool:
    return (
        fact.recorded_at <= now
        and (fact.valid_from is None or fact.valid_from <= now)
        and (fact.valid_to is None or fact.valid_to > now)
    )


class TestProbeReferences:
    def test_every_referenced_item_exists(self, fixture: Fixture) -> None:
        """Facts in facts.jsonl or the noise file; turns, episodes and notes in the recall
        fixture; a note may also be one of the probes' seed notes.
        """
        known = set(_item_texts(fixture))
        for probe in fixture.probes:
            for ref in (*probe.answer_refs, *probe.distractor_refs):
                assert ref in known, (probe.probe_id, ref)

    def test_every_probe_is_asked_after_the_history(self, fixture: Fixture) -> None:
        for probe in fixture.probes:
            assert probe.now > fixture.history_end + CONVERSATION_GAP, probe.probe_id

    def test_answer_facts_are_clean_and_available(self, fixture: Fixture) -> None:
        """A gold answer rests on clean facts only. Outside block-insufficient probes, which ask
        for superseded or future versions on purpose, a version must be valid at `now`.
        """
        noise_keys = {n.fact.fact_key for n in fixture.noise}
        versions: dict[str, list[Fact]] = {}
        for fact in fixture.facts:
            versions.setdefault(fact.fact_key, []).append(fact)
        for probe in fixture.probes:
            for ref in probe.answer_refs:
                if ref.kind != "fact":
                    continue
                assert ref.id not in noise_keys, probe.probe_id
                rows = versions[ref.id]
                assert any(row.recorded_at <= probe.now for row in rows), probe.probe_id
                if probe.kind != "block_insufficient":
                    assert any(_valid_at(row, probe.now) for row in rows), probe.probe_id

    def test_no_answer_rests_on_a_corrected_fact(self, fixture: Fixture) -> None:
        for probe in fixture.probes:
            corrected = {c.fact_key for c in fixture.corrections if c.recorded_at <= probe.now}
            for ref in probe.answer_refs:
                assert not (ref.kind == "fact" and ref.id in corrected), probe.probe_id

    def test_answer_and_distractor_refs_are_disjoint(self, fixture: Fixture) -> None:
        for probe in fixture.probes:
            assert set(probe.answer_refs).isdisjoint(probe.distractor_refs), probe.probe_id


# ---------------------------------------------------------------------------
# Authoring rules per kind
# ---------------------------------------------------------------------------


class TestGoldAnswers:
    def test_every_gold_answer_is_correct_by_its_own_terms(self, fixture: Fixture) -> None:
        for probe in fixture.probes:
            assert is_correct(probe, probe.gold_answer), probe.probe_id

    def test_no_gold_answer_leaks(self, fixture: Fixture) -> None:
        canaries = [p.canary for p in fixture.probes if p.canary is not None]
        for probe in fixture.probes:
            gold = probe.gold_answer
            assert not [t for t in probe.distractor_terms if contains_term(gold, t)], probe.probe_id
            assert not [c for c in canaries if contains_term(gold, c)], probe.probe_id
            if probe.misattributed_terms:
                assert not misattributes(gold, probe.misattributed_terms), probe.probe_id

    def test_a_run_of_the_gold_answers_scores_perfectly(self, fixture: Fixture) -> None:
        """The probes and the scorer agree: gold answers, plus a memory tool where one must fire."""
        run = [
            AnswerRecord(
                probe_id=probe.probe_id,
                answer=probe.gold_answer,
                tool_calls=(
                    (ToolCall(name=probe.expected_tools[0], arguments="{}"),)
                    if probe.expect_tool
                    else ()
                ),
            )
            for probe in fixture.probes
        ]

        overall = score_answers(fixture.probes, run).overall

        assert overall.correctness == 1.0
        assert overall.tool_firing == 1.0
        assert overall.negative_control_fp == 0.0
        assert overall.distractor_leak == 0.0
        assert overall.canary_leak == 0.0
        assert overall.misattribution == 0.0


class TestKindRules:
    def test_block_insufficient_probes_accept_only_memory_tools(self, fixture: Fixture) -> None:
        for probe in fixture.probes:
            if probe.kind == "block_insufficient":
                assert set(probe.expected_tools) <= set(MEMORY_TOOLS), probe.probe_id

    def test_distractor_terms_come_from_the_distractor_items(self, fixture: Fixture) -> None:
        """Each term names something a distractor item says, so a leak traces to an item."""
        texts = _item_texts(fixture)
        for probe in fixture.probes:
            if probe.kind != "distractor":
                continue
            carried = " ".join(texts[ref] for ref in probe.distractor_refs).lower()
            for term in probe.distractor_terms:
                assert term.lower() in carried, (probe.probe_id, term)

    def test_negative_control_terms_name_something_memory_holds(self, fixture: Fixture) -> None:
        """A negative control is tempting only if its memory terms could really be retrieved."""
        memory = " ".join(_item_texts(fixture).values()).lower()
        for probe in fixture.probes:
            if probe.kind != "negative_control":
                continue
            for term in probe.distractor_terms:
                assert term.lower() in memory, (probe.probe_id, term)

    def test_injection_seed_notes_carry_their_canary_and_nothing_else_does(
        self, fixture: Fixture
    ) -> None:
        clean_paths = {note.path for note in fixture.notes}
        clean_text = " ".join(
            [f"{n.title} {n.body}" for n in fixture.notes]
            + [f.text for f in fixture.facts]
            + [n.fact.text for n in fixture.noise]
            + [f"{t.user_utterance} {t.system_response}" for t in fixture.turns]
        )
        start = min(turn.timestamp for turn in fixture.turns)
        for probe in fixture.probes:
            if probe.kind != "injection":
                continue
            note = probe.seed_note
            assert note is not None and probe.canary is not None
            assert probe.canary in note.body, probe.probe_id
            assert probe.canary not in clean_text, probe.probe_id
            assert note.path not in clean_paths and not note.path.startswith("users/")
            assert start <= note.created_at <= note.updated_at <= fixture.history_end
            assert ItemRef(kind="note", id=note.path) in probe.answer_refs, probe.probe_id
            others = [p.canary for p in fixture.probes if p.canary and p.canary != probe.canary]
            assert not [c for c in others if c in note.body], probe.probe_id

    def test_seed_notes_name_no_corrected_object(self, fixture: Fixture) -> None:
        corrected = [correction.object_name for correction in fixture.corrections]
        for note in fixture.seed_notes:
            text = f"{note.title} {note.heading} {note.body}"
            assert not [name for name in corrected if contains_term(text, name)], note.path

    def test_misattributed_claims_are_the_assistants_words(self, fixture: Fixture) -> None:
        """Each claim term is in a referenced turn's response and in none of its user halves."""
        turns = {turn.event_id: turn for turn in fixture.turns}
        for probe in fixture.probes:
            if probe.kind != "misattribution":
                continue
            exchanges = [turns[r.id] for r in probe.answer_refs if r.kind == "exchange"]
            assert exchanges, probe.probe_id
            for term in probe.misattributed_terms:
                assert any(contains_term(t.system_response, term) for t in exchanges), term
                assert not any(contains_term(t.user_utterance, term) for t in exchanges), term

    def test_misattribution_probes_seed_the_misattributed_noise(self, fixture: Fixture) -> None:
        """The noisy graph holds the wrong attribution, extracted from the probe's own turn."""
        noise = {n.fact.fact_key: n for n in fixture.noise}
        for probe in fixture.probes:
            if probe.kind != "misattribution":
                continue
            turns = {r.id for r in probe.answer_refs if r.kind == "exchange"}
            seeded = [
                noise[r.id] for r in probe.distractor_refs if r.kind == "fact" and r.id in noise
            ]
            assert any(
                n.noise_kind == "misattributed_relation" and n.fact.source_utterance_id in turns
                for n in seeded
            ), probe.probe_id
