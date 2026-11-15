"""The synthetic recall fixture is well formed, consistent and big enough to measure.

The fetcher floors (T9, T19, T21), the segmenter (T17), the answer probes (T27) and the
real-model eval (H4) all trust `tests/fixtures/memory_recall/`. A probe naming an id that does
not exist, an episode boundary that does not follow Design point 11's rule, or a lexical tag that
does not describe its query would move each of those numbers silently. This file checks:

- every row of every file against its schema (through the loaders in
  `scripts/eval_harness/memory_recall.py`, whose strict parsing is itself tested here);
- referential integrity: every id a probe, correction, fact or episode names exists, and episode
  boundaries refer to real turns of their conversation;
- every conversation boundary and every episode boundary, recomputed from conversations.jsonl;
- the minimum counts of plan task T1, and the shape minimums of plan task T36;
- the fact types: every subject and object typed with an extractable ontology node type, one type
  per entity;
- the list utterances: at least three turns whose sibling facts overflow the reader's 300-token
  block, each turn stating its whole list;
- the probe authoring rules: availability at the probe's `now`, correction suppression, the
  lexical tag (judged against the text the index serves, a fact's `render_fact` sentence), and
  the `list_item` and `type_scoped` shapes;
- the held-out probe set: every per-probe rule above, its own minimums, ids unique across both
  probe files, and a cap on how close a held-out query may come to a dev query with the same
  reference set.

The segmentation here is written independently of any backend segmenter on purpose: it is the
reference T17's segmenter must reproduce, so it must not share that segmenter's code. It depends
on nothing in backend/chat/context_budget.py but `ApproximateTokenCounter().count`.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.chat.context_budget import ApproximateTokenCounter  # noqa: E402
from backend.knowledge.ontologies.v1_0_0 import (  # noqa: E402
    EDGE_TYPES_BY_NAME,
    EXTRACTABLE_NODE_TYPES,
    EXTRACTABLE_RELATIONSHIP_TYPES,
)
from backend.memory.fact_render import render_fact  # noqa: E402
from scripts.eval_harness.memory_recall import (  # noqa: E402
    FIXTURE_DIR,
    PROBE_KINDS,
    PROBE_SHAPES,
    Correction,
    Episode,
    Fact,
    ItemRef,
    Probe,
    Turn,
    VaultNote,
    load_corrections,
    load_episodes,
    load_facts,
    load_probes,
    load_turns,
    load_vault_notes,
    parse_correction,
    parse_episode,
    parse_fact,
    parse_probe,
    parse_turn,
    parse_vault_note,
)
from scripts.eval_harness.memory_recall_gate import PR1_GATES  # noqa: E402

# Design point 11 and the Phase B defaults (MIST_CONVERSATION_IDLE_GAP_MIN,
# MIST_EPISODE_MAX_TURNS, MIST_EPISODE_MAX_TOKENS).
CONVERSATION_GAP = timedelta(minutes=30)
EPISODE_MAX_TURNS = 20
EPISODE_MAX_TOKENS = 6000
SUMMARY_MAX_WORDS = 80
# The reader's memory block cap (plan T36): a list utterance must not fit in it.
BLOCK_CAP_TOKENS = 300
NEW_SHAPE_MINIMUM = 12

# The dev set every setting is chosen on, and the held-out set the read path's first PR is
# judged on. The per-probe rules hold for both; the file-level counts are each file's own.
DEV_PROBES = "probes.jsonl"
HELDOUT_PROBES = "heldout_probes.jsonl"
PROBE_FILES = (DEV_PROBES, HELDOUT_PROBES)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Fixture:
    facts: tuple[Fact, ...]
    turns: tuple[Turn, ...]
    episodes: tuple[Episode, ...]
    notes: tuple[VaultNote, ...]
    corrections: tuple[Correction, ...]
    probes: tuple[Probe, ...]

    @property
    def history_end(self) -> datetime:
        return max(turn.timestamp for turn in self.turns)


@pytest.fixture(scope="module")
def fixture() -> Fixture:
    return Fixture(
        facts=load_facts(),
        turns=load_turns(),
        episodes=load_episodes(),
        notes=load_vault_notes(),
        corrections=load_corrections(),
        probes=load_probes(),
    )


@pytest.fixture(scope="module", params=PROBE_FILES)
def probed(request: pytest.FixtureRequest, fixture: Fixture) -> Fixture:
    """The corpus with one probe file's probes, so every per-probe rule runs on both files."""
    return replace(fixture, probes=load_probes(FIXTURE_DIR / request.param))


@pytest.fixture(scope="module")
def heldout(fixture: Fixture) -> Fixture:
    return replace(fixture, probes=load_probes(FIXTURE_DIR / HELDOUT_PROBES))


def _first_row(name: str) -> dict[str, object]:
    line = (FIXTURE_DIR / name).read_text(encoding="utf-8").splitlines()[0]
    return json.loads(line)


PARSERS: list[tuple[str, Callable[[Mapping[str, object]], object]]] = [
    ("facts.jsonl", parse_fact),
    ("conversations.jsonl", parse_turn),
    ("episodes.jsonl", parse_episode),
    ("vault_notes.jsonl", parse_vault_note),
    ("corrections.jsonl", parse_correction),
    (DEV_PROBES, parse_probe),
    (HELDOUT_PROBES, parse_probe),
]


class TestSchemas:
    def test_every_file_loads_through_its_schema(self, fixture: Fixture) -> None:
        """Loading the module fixture parses every row; a bad row raises and names its line."""
        assert fixture.facts and fixture.turns and fixture.episodes
        assert fixture.notes and fixture.corrections and fixture.probes

    @pytest.mark.parametrize(("name", "parse"), PARSERS, ids=[name for name, _ in PARSERS])
    def test_a_row_with_an_unknown_key_is_refused(
        self, name: str, parse: Callable[[Mapping[str, object]], object]
    ) -> None:
        row = _first_row(name)
        row["unexpected_key"] = "x"

        with pytest.raises(ValueError, match="unexpected_key"):
            parse(row)

    @pytest.mark.parametrize(("name", "parse"), PARSERS, ids=[name for name, _ in PARSERS])
    def test_a_row_missing_any_key_is_refused(
        self, name: str, parse: Callable[[Mapping[str, object]], object]
    ) -> None:
        """Every key is required, nullable ones included, so a dropped key is never a default."""
        row = _first_row(name)
        for key in list(row):
            partial = {k: v for k, v in row.items() if k != key}

            with pytest.raises(ValueError, match=key):
                parse(partial)

    def test_every_id_is_unique_within_its_file(self, fixture: Fixture, heldout: Fixture) -> None:
        for ids in (
            [f.row_key for f in fixture.facts],
            [t.event_id for t in fixture.turns],
            [e.episode_id for e in fixture.episodes],
            [n.path for n in fixture.notes],
            [c.correction_id for c in fixture.corrections],
            [p.probe_id for p in fixture.probes],
            [p.probe_id for p in heldout.probes],
        ):
            duplicates = [item for item, count in Counter(ids).items() if count > 1]
            assert duplicates == []


# ---------------------------------------------------------------------------
# Counts (plan task T1, Acceptance)
# ---------------------------------------------------------------------------


class TestMinimumCounts:
    def test_facts(self, fixture: Fixture) -> None:
        states = Counter(fact.state for fact in fixture.facts)

        assert len(fixture.facts) >= 150
        assert states["superseded"] >= 15
        assert states["future"] >= 5
        assert states["retracted"] >= 5

    def test_conversations_span_fourteen_days_and_three_hundred_turns(
        self, fixture: Fixture
    ) -> None:
        days = {turn.timestamp.date() for turn in fixture.turns}

        assert len({turn.conversation_id for turn in fixture.turns}) >= 30
        assert len(days) == 14
        assert (max(days) - min(days)).days == 13
        assert len(fixture.turns) >= 300

    def test_episodes_cut_by_each_limit(self, fixture: Fixture) -> None:
        closed_by = Counter(episode.closed_by for episode in fixture.episodes)

        assert closed_by["turn_limit"] >= 2
        assert closed_by["token_limit"] >= 2

    def test_notes_corrections_and_probes(self, fixture: Fixture) -> None:
        assert len(fixture.notes) >= 40
        assert len(fixture.corrections) >= 10
        assert len(fixture.probes) >= 120

    def test_every_probe_kind_is_present(self, fixture: Fixture) -> None:
        assert {probe.kind for probe in fixture.probes} == set(PROBE_KINDS)

    def test_at_least_forty_percent_of_probes_are_lexical_false(self, fixture: Fixture) -> None:
        paraphrase = sum(1 for probe in fixture.probes if not probe.lexical)

        assert paraphrase / len(fixture.probes) >= 0.40

    def test_at_least_ten_noise_probes_are_lexical(self, fixture: Fixture) -> None:
        """T9 floors empty_on_noise at 0.90 on lexical probes; ten leave room for one miss."""
        lexical_noise = [p for p in fixture.probes if p.kind == "noise" and p.lexical]

        assert len(lexical_noise) >= 10

    def test_each_new_probe_shape_has_at_least_twelve_probes(self, fixture: Fixture) -> None:
        shapes = Counter(probe.shape for probe in fixture.probes)

        assert set(shapes) == set(PROBE_SHAPES)
        assert shapes["list_item"] >= NEW_SHAPE_MINIMUM
        assert shapes["type_scoped"] >= NEW_SHAPE_MINIMUM


# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------


class TestFacts:
    def test_fact_key_is_the_hash_of_the_triple(self, fixture: Fixture) -> None:
        """Design point 2: `fact_key = sha256(subject_id | rel_type | object_id)`."""
        for fact in fixture.facts:
            triple = f"{fact.subject_id}|{fact.relation}|{fact.object_id}"
            assert fact.fact_key == hashlib.sha256(triple.encode()).hexdigest(), fact.row_key

    def test_relations_are_directional_extractable_ontology_types(self, fixture: Fixture) -> None:
        """An undirected type would need the canonical id order in its key; none is used."""
        for fact in fixture.facts:
            assert fact.relation in EXTRACTABLE_RELATIONSHIP_TYPES, fact.row_key
            assert EDGE_TYPES_BY_NAME[fact.relation].directional, fact.row_key

    def test_subject_and_object_types_are_extractable_node_types(self, fixture: Fixture) -> None:
        """A type-scoped probe filters on these types, so each must be one the extractor emits."""
        for fact in fixture.facts:
            assert fact.subject_type in EXTRACTABLE_NODE_TYPES, fact.row_key
            assert fact.object_type in EXTRACTABLE_NODE_TYPES, fact.row_key

    def test_an_entity_has_one_type_in_every_fact(self, fixture: Fixture) -> None:
        """An entity typed two ways would fall inside and outside the same type scope."""
        types: dict[str, str] = {}
        for fact in fixture.facts:
            for entity_id, node_type in (
                (fact.subject_id, fact.subject_type),
                (fact.object_id, fact.object_type),
            ):
                assert types.setdefault(entity_id, node_type) == node_type, (
                    fact.row_key,
                    entity_id,
                )

    def test_every_row_of_a_fact_key_names_the_same_triple(self, fixture: Fixture) -> None:
        names: dict[str, tuple[str, str, str]] = {}
        for fact in fixture.facts:
            triple = (fact.subject_name, fact.relation, fact.object_name)
            assert names.setdefault(fact.fact_key, triple) == triple, fact.row_key

    def test_source_utterance_is_a_real_earlier_turn(self, fixture: Fixture) -> None:
        turns = {turn.event_id: turn for turn in fixture.turns}
        for fact in fixture.facts:
            if fact.source_utterance_id is None:
                continue
            turn = turns[fact.source_utterance_id]
            assert turn.timestamp <= fact.recorded_at, fact.row_key

    def test_confidence_is_a_probability(self, fixture: Fixture) -> None:
        assert all(0.0 < fact.confidence <= 1.0 for fact in fixture.facts)

    def test_state_matches_the_validity_interval(self, fixture: Fixture) -> None:
        """Each state label is checked against the interval at the end of the history."""
        rows = {fact.row_key: fact for fact in fixture.facts}
        end = fixture.history_end
        for fact in fixture.facts:
            assert fact.recorded_at <= end, fact.row_key
            if fact.valid_from is not None and fact.valid_to is not None:
                assert fact.valid_from < fact.valid_to, fact.row_key
            if fact.state == "current":
                assert fact.valid_to is None and fact.superseded_by is None, fact.row_key
                assert fact.valid_from is None or fact.valid_from <= end, fact.row_key
            elif fact.state == "future":
                assert fact.valid_from is not None and fact.valid_from > end, fact.row_key
                assert fact.valid_to is None and fact.superseded_by is None, fact.row_key
            elif fact.state == "retracted":
                assert fact.valid_to is not None and fact.valid_to <= end, fact.row_key
                assert fact.superseded_by is None, fact.row_key
            else:
                assert fact.valid_to is not None and fact.superseded_by is not None, fact.row_key
                newer = rows[fact.superseded_by]
                assert (newer.subject_id, newer.relation) == (fact.subject_id, fact.relation)
                # A gap is allowed: Portuguese lapsed in 2023 and resumed in 2026.
                assert newer.valid_from is not None, fact.row_key
                assert newer.valid_from >= fact.valid_to, fact.row_key

    def test_a_correction_replacement_matches_its_correction(self, fixture: Fixture) -> None:
        corrections = {c.correction_id: c for c in fixture.corrections}
        for fact in fixture.facts:
            if fact.source_correction_id is None:
                continue
            correction = corrections[fact.source_correction_id]
            assert correction.action == "correct", fact.row_key
            assert (fact.subject_id, fact.relation) == (
                correction.subject_id,
                correction.relation,
            )
            assert fact.object_name == correction.replacement_object_name, fact.row_key
            assert fact.recorded_at == correction.recorded_at, fact.row_key
            assert fact.valid_from == correction.recorded_at, fact.row_key


# ---------------------------------------------------------------------------
# Conversations and episodes
# ---------------------------------------------------------------------------


def _turn_tokens(turn: Turn, counter: ApproximateTokenCounter) -> int:
    return counter.count(turn.user_utterance + "\n" + turn.system_response)


def _recompute_conversations(turns: tuple[Turn, ...]) -> list[list[str]]:
    """Cut the time-ordered turns wherever the gap between two turns exceeds 30 minutes."""
    ordered = sorted(turns, key=lambda turn: turn.timestamp)
    groups: list[list[str]] = []
    previous: datetime | None = None
    for turn in ordered:
        if previous is None or turn.timestamp - previous > CONVERSATION_GAP:
            groups.append([])
        groups[-1].append(turn.event_id)
        previous = turn.timestamp
    return groups


@dataclass(frozen=True, slots=True)
class _Cut:
    event_ids: tuple[str, ...]
    token_count: int
    closed_by: str


def _recompute_episodes(conversation: list[Turn]) -> list[_Cut]:
    """Design point 11: a new episode starts once the open one holds 20 turns, or once adding
    the next turn would take its summed tokens past 6000; the last closes at conversation end.
    """
    counter = ApproximateTokenCounter()
    cuts: list[_Cut] = []
    open_ids: list[str] = []
    open_tokens = 0
    for turn in conversation:
        tokens = _turn_tokens(turn, counter)
        if open_ids and len(open_ids) == EPISODE_MAX_TURNS:
            cuts.append(_Cut(tuple(open_ids), open_tokens, "turn_limit"))
            open_ids, open_tokens = [], 0
        elif open_ids and open_tokens + tokens > EPISODE_MAX_TOKENS:
            cuts.append(_Cut(tuple(open_ids), open_tokens, "token_limit"))
            open_ids, open_tokens = [], 0
        open_ids.append(turn.event_id)
        open_tokens += tokens
    if open_ids:
        cuts.append(_Cut(tuple(open_ids), open_tokens, "conversation_end"))
    return cuts


def _conversations(turns: tuple[Turn, ...]) -> dict[str, list[Turn]]:
    grouped: dict[str, list[Turn]] = defaultdict(list)
    for turn in sorted(turns, key=lambda turn: turn.timestamp):
        grouped[turn.conversation_id].append(turn)
    return dict(grouped)


class TestConversations:
    def test_turn_timestamps_are_strictly_increasing_in_file_order(self, fixture: Fixture) -> None:
        stamps = [turn.timestamp for turn in fixture.turns]

        assert all(a < b for a, b in zip(stamps, stamps[1:]))

    def test_conversation_ids_follow_the_thirty_minute_gap_rule(self, fixture: Fixture) -> None:
        recomputed = _recompute_conversations(fixture.turns)
        recorded = [[t.event_id for t in turns] for turns in _conversations(fixture.turns).values()]

        assert recorded == recomputed

    def test_gap_boundaries_are_exercised_on_both_sides(self, fixture: Fixture) -> None:
        """A gap of exactly 30:00 stays inside a conversation and one of 30:01 starts a new one,
        so a segmenter using `>=` or a minute-rounded gap is caught.
        """
        ordered = sorted(fixture.turns, key=lambda turn: turn.timestamp)
        gaps = {
            (b.timestamp - a.timestamp, a.conversation_id == b.conversation_id)
            for a, b in zip(ordered, ordered[1:])
        }

        assert (CONVERSATION_GAP, True) in gaps
        assert (CONVERSATION_GAP + timedelta(seconds=1), False) in gaps

    def test_sessions_and_conversations_cross_both_ways(self, fixture: Fixture) -> None:
        """One session holds two conversations and one conversation spans two sessions (a
        reconnect inside the gap), so a segmenter that cuts on session id is caught.
        """
        sessions_of: dict[str, set[str]] = defaultdict(set)
        conversations_of: dict[str, set[str]] = defaultdict(set)
        for turn in fixture.turns:
            sessions_of[turn.conversation_id].add(turn.session_id)
            conversations_of[turn.session_id].add(turn.conversation_id)

        assert any(len(sessions) > 1 for sessions in sessions_of.values())
        assert any(len(conversations) > 1 for conversations in conversations_of.values())

    def test_turn_index_counts_each_session_from_zero(self, fixture: Fixture) -> None:
        by_session: dict[str, list[int]] = defaultdict(list)
        for turn in fixture.turns:
            by_session[turn.session_id].append(turn.turn_index)

        for indexes in by_session.values():
            assert indexes == list(range(len(indexes)))


class TestEpisodes:
    def test_every_boundary_matches_the_segmentation_rule(self, fixture: Fixture) -> None:
        """Recompute every episode from conversations.jsonl and compare boundaries, token sums
        and close reasons with episodes.jsonl. On a mismatch after extending the fixture, the
        assertion shows the boundaries the rule gives.
        """
        recomputed = [
            (conversation_id, cut.event_ids, cut.token_count, cut.closed_by)
            for conversation_id, turns in _conversations(fixture.turns).items()
            for cut in _recompute_episodes(turns)
        ]
        recorded = [
            (e.conversation_id, e.event_ids, e.token_count, e.closed_by) for e in fixture.episodes
        ]

        assert recorded == recomputed

    def test_each_limit_is_hit_exactly_at_its_edge(self, fixture: Fixture) -> None:
        """One episode closes holding exactly 20 turns and one holds exactly 6000 tokens, so an
        off-by-one in either comparison changes a boundary.
        """
        assert any(
            e.closed_by == "turn_limit" and e.turn_count == EPISODE_MAX_TURNS
            for e in fixture.episodes
        )
        assert any(e.token_count == EPISODE_MAX_TOKENS for e in fixture.episodes)

    def test_episode_id_is_the_hash_of_its_joined_event_ids(self, fixture: Fixture) -> None:
        """Design point 11: `e-<sha256(joined event ids)[:16]>`, joined with `|`."""
        for episode in fixture.episodes:
            digest = hashlib.sha256("|".join(episode.event_ids).encode()).hexdigest()
            assert episode.episode_id == f"e-{digest[:16]}"

    def test_boundaries_refer_to_real_turns_of_the_conversation(self, fixture: Fixture) -> None:
        turns = {turn.event_id: turn for turn in fixture.turns}
        for episode in fixture.episodes:
            members = [turns[event_id] for event_id in episode.event_ids]
            assert {t.conversation_id for t in members} == {episode.conversation_id}
            assert episode.first_event_id == episode.event_ids[0]
            assert episode.last_event_id == episode.event_ids[-1]
            assert episode.start == members[0].timestamp
            assert episode.end == members[-1].timestamp
            assert episode.turn_count == len(members)

    def test_summaries_stay_within_eighty_words(self, fixture: Fixture) -> None:
        for episode in fixture.episodes:
            assert len(episode.summary.split()) <= SUMMARY_MAX_WORDS, episode.episode_id


# ---------------------------------------------------------------------------
# List utterances
# ---------------------------------------------------------------------------


def _sibling_groups(fixture: Fixture) -> dict[tuple[str, str, str], tuple[Fact, ...]]:
    """Facts extracted together from one list: same subject, relation and source turn.

    A group needs two distinct fact keys; one fact with several versions is not a list.
    """
    grouped: dict[tuple[str, str, str], list[Fact]] = defaultdict(list)
    for fact in fixture.facts:
        if fact.source_utterance_id is not None:
            key = (fact.subject_id, fact.relation, fact.source_utterance_id)
            grouped[key].append(fact)
    return {
        key: tuple(facts)
        for key, facts in grouped.items()
        if len({fact.fact_key for fact in facts}) > 1
    }


def _group_tokens(facts: tuple[Fact, ...]) -> int:
    counter = ApproximateTokenCounter()
    return sum(counter.count(fact.text) for fact in facts)


def _list_utterances(fixture: Fixture) -> dict[tuple[str, str, str], tuple[Fact, ...]]:
    """The sibling groups whose texts alone overflow the reader's 300-token block."""
    return {
        key: facts
        for key, facts in _sibling_groups(fixture).items()
        if _group_tokens(facts) > BLOCK_CAP_TOKENS
    }


class TestListUtterances:
    def test_at_least_three_list_utterances_overflow_the_block(self, fixture: Fixture) -> None:
        """The reader's cap is only exercised when a list's siblings cannot all be served."""
        sums = {key: _group_tokens(facts) for key, facts in _list_utterances(fixture).items()}

        assert len(sums) >= 3, sums
        assert all(total > BLOCK_CAP_TOKENS for total in sums.values())

    def test_list_utterances_include_recommends_and_has_habit(self, fixture: Fixture) -> None:
        relations = {relation for _, relation, _ in _list_utterances(fixture)}

        assert {"RECOMMENDS", "HAS_HABIT"} <= relations

    def test_each_list_turn_states_every_sibling(self, fixture: Fixture) -> None:
        """A list fact must be read off its turn, not attached to a turn that never says it."""
        turns = {turn.event_id: turn for turn in fixture.turns}
        for (_, _, event_id), facts in _list_utterances(fixture).items():
            utterance = turns[event_id].user_utterance
            for fact in facts:
                assert _whole_word(fact.object_name, utterance), (event_id, fact.object_name)

    def test_each_list_turn_has_a_conversation_and_session_of_its_own(
        self, fixture: Fixture
    ) -> None:
        """The list turns were added as new conversations, so no existing boundary moved."""
        turns = {turn.event_id: turn for turn in fixture.turns}
        for _, _, event_id in _list_utterances(fixture):
            turn = turns[event_id]
            conversation = {
                t.event_id for t in fixture.turns if t.conversation_id == turn.conversation_id
            }
            session = {t.event_id for t in fixture.turns if t.session_id == turn.session_id}
            assert conversation == session, event_id


# ---------------------------------------------------------------------------
# Notes and corrections
# ---------------------------------------------------------------------------


class TestNotes:
    def test_notes_are_markdown_paths_dated_within_the_history(self, fixture: Fixture) -> None:
        for note in fixture.notes:
            assert note.path.endswith(".md"), note.path
            assert not note.path.startswith("users/"), "the profile note is never a candidate"
            assert note.created_at <= note.updated_at <= fixture.history_end, note.path


class TestCorrections:
    def test_correction_ids_have_the_logged_shape(self, fixture: Fixture) -> None:
        for correction in fixture.corrections:
            assert re.fullmatch(r"corr-[0-9a-f]{32}", correction.correction_id)

    def test_each_correction_names_an_existing_earlier_fact(self, fixture: Fixture) -> None:
        for correction in fixture.corrections:
            rows = [f for f in fixture.facts if f.fact_key == correction.fact_key]
            assert rows, correction.correction_id
            fact = rows[0]
            assert (fact.subject_id, fact.relation, fact.object_id) == (
                correction.subject_id,
                correction.relation,
                correction.object_id,
            )
            assert (fact.subject_name, fact.object_name) == (
                correction.subject_name,
                correction.object_name,
            )
            assert min(f.recorded_at for f in rows) < correction.recorded_at

    def test_correct_has_a_replacement_fact_and_forget_has_none(self, fixture: Fixture) -> None:
        replaced = {f.source_correction_id for f in fixture.facts if f.source_correction_id}
        for correction in fixture.corrections:
            if correction.action == "correct":
                assert correction.replacement_object_name is not None
                assert correction.correction_id in replaced
            else:
                assert correction.replacement_object_name is None
                assert correction.correction_id not in replaced

    def test_surface_decides_the_issuing_context(self, fixture: Fixture) -> None:
        """A tool correction is recorded with its turn's event id and time (Design point 12);
        a WS correction belongs to a session with no turn; a CLI correction to neither.
        """
        turns = {turn.event_id: turn for turn in fixture.turns}
        conversations = {turn.conversation_id for turn in fixture.turns}
        sessions = {turn.session_id for turn in fixture.turns}
        surfaces = Counter(c.surface for c in fixture.corrections)
        for correction in fixture.corrections:
            if correction.surface == "tool":
                turn = turns[correction.source_event_id or ""]
                assert correction.recorded_at == turn.timestamp
                assert correction.session_id == turn.session_id
                assert correction.conversation_id == turn.conversation_id
                assert correction.turn_index == turn.turn_index
            elif correction.surface == "ws":
                assert correction.source_event_id is None and correction.turn_index is None
                assert correction.session_id in sessions
                assert correction.conversation_id in conversations
            else:
                assert correction.source_event_id is None and correction.turn_index is None
                assert correction.session_id is None and correction.conversation_id is None

        assert set(surfaces) == {"tool", "ws", "cli"}


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _RenderedVersion:
    """One fact version in the shape `render_fact` reads (`RenderableFact`).

    The bounds are `isoformat()` strings, as the index's rows carry them, so a timestamp bound
    renders as a date in the zone passed to `render_fact`.
    """

    subject_name: str
    rel_type: str
    object_name: str
    valid_from: str | None
    valid_to: str | None

    @classmethod
    def of(cls, fact: Fact) -> _RenderedVersion:
        return cls(
            subject_name=fact.subject_name,
            rel_type=fact.relation,
            object_name=fact.object_name,
            valid_from=None if fact.valid_from is None else fact.valid_from.isoformat(),
            valid_to=None if fact.valid_to is None else fact.valid_to.isoformat(),
        )


def _item_texts(fixture: Fixture) -> dict[ItemRef, str]:
    """The text each item is served under, which is what the lexical tag is judged against.

    A fact's text is the `render_fact` sentence of every version of its key, rendered at UTC,
    plus its subject and object names: the index stores and matches that sentence, not the
    authored `text`, so a tag judged on the authored wording would describe a query the fetcher
    never sees. Every version is included, so the text does not depend on a probe's `now`: a
    word any version's sentence carries counts toward a word match and against a paraphrase.
    UTC is the zone every fixture timestamp is written in; a far-off zone can move a timestamp
    bound to another day, and with it a date word.
    """
    texts: dict[ItemRef, str] = {}
    for fact in fixture.facts:
        ref = ItemRef("fact", fact.fact_key)
        sentence = render_fact(_RenderedVersion.of(fact), UTC)
        texts[ref] = f"{texts.get(ref, '')} {sentence} {fact.subject_name} {fact.object_name}"
    for episode in fixture.episodes:
        texts[ItemRef("episode", episode.episode_id)] = " ".join(
            (episode.title, episode.summary, *episode.open_threads)
        )
    for note in fixture.notes:
        texts[ItemRef("note", note.path)] = f"{note.title} {note.heading} {note.body}"
    for turn in fixture.turns:
        texts[ItemRef("exchange", turn.event_id)] = f"{turn.user_utterance} {turn.system_response}"
    return texts


def _available_at(fixture: Fixture, now: datetime) -> set[ItemRef]:
    """Items the read path can serve at `now`: a fact only by a version valid at `now`."""
    available: set[ItemRef] = set()
    for fact in fixture.facts:
        starts = fact.valid_from is None or fact.valid_from <= now
        ends = fact.valid_to is not None and fact.valid_to <= now
        if fact.recorded_at <= now and starts and not ends:
            available.add(ItemRef("fact", fact.fact_key))
    available |= {ItemRef("episode", e.episode_id) for e in fixture.episodes if e.end <= now}
    available |= {ItemRef("note", n.path) for n in fixture.notes if n.created_at <= now}
    available |= {ItemRef("exchange", t.event_id) for t in fixture.turns if t.timestamp <= now}
    return available


def _whole_word(name: str, text: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text, flags=re.IGNORECASE) is not None


def _suppressed_by(fixture: Fixture, correction: Correction) -> set[ItemRef]:
    """Design point 12's read-time suppression: the fact key, plus every episode, exchange and
    note recorded at or before the correction whose text holds the object name as a whole word.
    """
    name = correction.object_name
    at = correction.recorded_at
    suppressed = {ItemRef("fact", correction.fact_key)}
    for episode in fixture.episodes:
        text = " ".join((episode.title, episode.summary, *episode.open_threads))
        if episode.end <= at and _whole_word(name, text):
            suppressed.add(ItemRef("episode", episode.episode_id))
    for turn in fixture.turns:
        text = f"{turn.user_utterance} {turn.system_response}"
        if turn.timestamp <= at and _whole_word(name, text):
            suppressed.add(ItemRef("exchange", turn.event_id))
    for note in fixture.notes:
        text = f"{note.title} {note.heading} {note.body}"
        if note.updated_at <= at and _whole_word(name, text):
            suppressed.add(ItemRef("note", note.path))
    return suppressed


STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are as at be because been before
    being below between both but by can could did do does doing down during each few for from
    further had has have having he her here hers herself him himself his how i if in into is it
    its itself just me more most my myself no nor not now of off on once only or other our ours
    ourselves out over own same she should so some such than that the their theirs them
    themselves then there these they this those through to too under until up very was we were
    what when where which while who whom why will with would you your yours yourself yourselves
    s t d ll m re ve
    hey hi ok okay please remind tell know again any anything something thing things get got
    let us mist
    """.split()
)


def _content_tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in STOPWORDS}


def _stem(token: str) -> str:
    """A deliberately crude stem, used only to make the lexical-false rule stricter.

    It over-conflates on purpose ("dislike", "dislikes" and "disliked" all become "dislik";
    "running" and "run" both become "run"): a false match only makes a paraphrase probe harder
    to author, while a missed match would let a word-matching fetcher pass a paraphrase probe.
    """
    for suffix, replacement in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            token = token[: -len(suffix)] + replacement
            break
    if token.endswith("e") and len(token) >= 4:
        token = token[:-1]
    if len(token) >= 4 and token[-1] == token[-2] and token[-1] not in "aeiou":
        token = token[:-1]
    return token


def _lexical_violation(probe: Probe, texts: dict[ItemRef, str]) -> str | None:
    """Check the lexical tag: true when a word matcher can get the probe right.

    `texts` is `_item_texts`, the text the index serves for each item, so a fact is judged by its
    rendered sentence and not by its authored `text`.

    A noise probe is right when its block is empty, so its reference is every retrievable item,
    conversation turns included. Lexical true means the query shares no content token, not even
    a crude stem, with any item, so a word matcher returns nothing; lexical false means it shares
    at least one content token with some item, which tempts a word matcher into serving it.

    Any other probe's reference items are its expected ones, failing those its forbidden ones (a
    correction probe after a forget). Lexical true needs an exact shared content token with every
    reference item; lexical false needs no shared stem with any of them, so only a semantic match
    finds them.
    """
    query = _content_tokens(probe.query)
    query_stems = {_stem(token) for token in query}
    if probe.kind == "noise":
        for ref, text in texts.items():
            tokens = _content_tokens(text)
            if probe.lexical and query_stems & {_stem(token) for token in tokens}:
                overlap = sorted(query_stems & {_stem(token) for token in tokens})
                return f"noise tagged lexical but shares {overlap} with {ref.kind} {ref.id}"
            if not probe.lexical and query & tokens:
                return None
        return None if probe.lexical else "noise tagged paraphrase but shares no word with any item"
    reference = list(probe.expected) if probe.expected else list(probe.forbidden)
    shared = [query & _content_tokens(texts[ref]) for ref in reference]
    if probe.lexical:
        ok = bool(reference) and all(shared)
        return None if ok else "tagged lexical but shares no content word with its items"
    for ref in reference:
        overlap = query_stems & {_stem(token) for token in _content_tokens(texts[ref])}
        if overlap:
            return f"tagged paraphrase but shares {sorted(overlap)} with {ref.kind} {ref.id}"
    return None


class TestProbes:
    def test_every_referenced_item_exists(self, probed: Fixture) -> None:
        texts = _item_texts(probed)
        for probe in probed.probes:
            for ref in (*probe.expected, *probe.acceptable, *probe.forbidden, *probe.distractors):
                assert ref in texts, (probe.probe_id, ref)

    def test_reference_lists_do_not_overlap(self, probed: Fixture) -> None:
        for probe in probed.probes:
            lists = [probe.expected, probe.acceptable, probe.forbidden, probe.distractors]
            flat = [ref for refs in lists for ref in refs]
            assert len(flat) == len(set(flat)), probe.probe_id

    def test_probes_are_asked_after_the_history_closes(self, probed: Fixture) -> None:
        """Every conversation is idle past the gap, so every episode is closed and indexed."""
        for probe in probed.probes:
            assert probe.now > probed.history_end + CONVERSATION_GAP, probe.probe_id

    def test_kind_rules(self, probed: Fixture) -> None:
        corrections = {c.correction_id for c in probed.corrections}
        expected_kinds = {
            "fact": {"fact"},
            "note": {"note"},
            "episode": {"episode"},
            "continuity": {"episode"},
            "temporal": {"episode"},
        }
        for probe in probed.probes:
            if probe.kind == "noise":
                assert not (probe.expected or probe.acceptable or probe.forbidden)
            if probe.kind == "correction":
                assert probe.correction_id in corrections, probe.probe_id
                assert probe.forbidden, probe.probe_id
            else:
                assert probe.correction_id is None and not probe.forbidden, probe.probe_id
            if probe.kind in expected_kinds:
                assert probe.expected, probe.probe_id
                assert {ref.kind for ref in probe.expected} <= expected_kinds[probe.kind]

    def test_continuity_probes_open_a_conversation(self, probed: Fixture) -> None:
        for probe in probed.probes:
            if probe.kind == "continuity":
                assert probe.conversation_start, probe.probe_id

    def test_each_correction_has_one_probe(self, fixture: Fixture) -> None:
        """A file-level rule, here on the dev set; TestHeldOutProbes checks the held-out set."""
        asked = Counter(p.correction_id for p in fixture.probes if p.kind == "correction")

        assert asked == Counter(c.correction_id for c in fixture.corrections)

    def test_expected_and_acceptable_items_are_available_at_now(self, probed: Fixture) -> None:
        for probe in probed.probes:
            available = _available_at(probed, probe.now)
            for ref in (*probe.expected, *probe.acceptable):
                assert ref in available, (probe.probe_id, ref)

    def test_correction_probes_forbid_everything_the_correction_suppresses(
        self, probed: Fixture
    ) -> None:
        corrections = {c.correction_id: c for c in probed.corrections}
        replacements = {
            f.source_correction_id: ItemRef("fact", f.fact_key)
            for f in probed.facts
            if f.source_correction_id
        }
        for probe in probed.probes:
            if probe.kind != "correction":
                continue
            correction = corrections[probe.correction_id or ""]
            assert probe.now > correction.recorded_at, probe.probe_id
            assert set(probe.forbidden) == _suppressed_by(probed, correction), probe.probe_id
            if correction.action == "correct":
                assert replacements[correction.correction_id] in probe.expected

    def test_no_probe_expects_an_item_a_prior_correction_suppresses(self, probed: Fixture) -> None:
        suppressions = [(c.recorded_at, _suppressed_by(probed, c)) for c in probed.corrections]
        for probe in probed.probes:
            for recorded_at, suppressed in suppressions:
                if recorded_at < probe.now:
                    assert not suppressed & {*probe.expected, *probe.acceptable}, probe.probe_id

    def test_list_item_probes_expect_every_sibling_of_one_utterance(self, probed: Fixture) -> None:
        """Recall on a list probe measures how much of one list survives the block cap."""
        lists = {
            frozenset(ItemRef("fact", fact.fact_key) for fact in facts)
            for facts in _sibling_groups(probed).values()
        }
        for probe in probed.probes:
            if probe.shape == "list_item":
                assert probe.kind == "fact", probe.probe_id
                assert frozenset(probe.expected) in lists, probe.probe_id

    def test_type_scoped_probes_name_a_type_not_an_entity(self, probed: Fixture) -> None:
        """A type-scoped probe expects every served fact of one object type for its subject and
        relation, and its query names none of those objects, so only the type can find them.
        """
        rows = {fact.fact_key: fact for fact in probed.facts}
        for probe in probed.probes:
            if probe.shape != "type_scoped":
                continue
            assert probe.kind == "fact" and probe.expected, probe.probe_id
            assert {ref.kind for ref in probe.expected} == {"fact"}, probe.probe_id
            facts = [rows[ref.id] for ref in probe.expected]
            types = {fact.object_type for fact in facts}
            assert len(types) == 1, (probe.probe_id, types)
            for fact in facts:
                assert not _whole_word(fact.object_name, probe.query), (probe.probe_id, fact)
            suppressed: set[ItemRef] = set()
            for correction in probed.corrections:
                if correction.recorded_at < probe.now:
                    suppressed |= _suppressed_by(probed, correction)
            scopes = {(fact.subject_id, fact.relation) for fact in facts}
            in_scope = {
                ItemRef("fact", fact.fact_key)
                for fact in probed.facts
                if (fact.subject_id, fact.relation) in scopes and fact.object_type in types
            }
            served = (in_scope & _available_at(probed, probe.now)) - suppressed
            assert set(probe.expected) == served, probe.probe_id

    def test_a_fact_is_judged_by_its_rendered_sentence(self, fixture: Fixture) -> None:
        """fact-01's tag changed when the judged text moved from the authored sentence to the
        one the index stores, so its key must read as `render_fact` renders it.
        """
        probe = next(p for p in fixture.probes if p.probe_id == "fact-01")

        text = _item_texts(fixture)[probe.expected[0]]

        assert "Wren prefers cortado since 12 January 2026." in text
        assert "usual coffee order" not in text

    def test_lexical_tags_describe_the_query(self, probed: Fixture) -> None:
        texts = _item_texts(probed)
        violations = {
            probe.probe_id: problem
            for probe in probed.probes
            if (problem := _lexical_violation(probe, texts)) is not None
        }

        assert violations == {}


# ---------------------------------------------------------------------------
# Held-out probes
# ---------------------------------------------------------------------------

# The held-out fact minimums (plan task T54). The corpus fixes the targets of a list or a
# type-scoped probe, so only the wording separates such a probe from its dev twin.
HELDOUT_FACT_MINIMUM = 69
HELDOUT_FACT_PARAPHRASE_MINIMUM = 49
HELDOUT_PLAIN_MINIMUM = 43
HELDOUT_PLAIN_PARAPHRASE_MINIMUM = 27
HELDOUT_LIST_PROBES_PER_LIST = 3
HELDOUT_LIST_PARAPHRASE_MINIMUM = 8
HELDOUT_TYPE_SCOPED_MINIMUM = 14
# Shared content stems over their union, between a held-out query and a dev query that share a
# reference set. Above this, a held-out probe re-asks a dev probe and measures nothing new.
MAX_QUERY_OVERLAP = 0.5
# The held-out set's own file-level rules (plan task T55). Its kinds are the ones PR 1 gates on:
# the episode, continuity and temporal kinds wait for phase 2.
HELDOUT_KINDS = ("fact", "note", "correction", "noise")
HELDOUT_PARAPHRASE_SHARE = 0.40
HELDOUT_NOTE_MINIMUM = 29
HELDOUT_NOTE_PARAPHRASE_MINIMUM = 12
HELDOUT_NOISE_MINIMUM = 20
HELDOUT_LEXICAL_NOISE_MINIMUM = 10
# The pooled recall classes of `PR1_GATES`, by name, and the probes each must average over.
HELDOUT_POOLED_MINIMUMS = {
    "recall phase 1 pooled": 103,
    "recall phase 1 pooled, paraphrase": 63,
}


@dataclass(frozen=True, slots=True)
class _HeldOutProfile:
    """The held-out file's counts, so a file-level test holds no probe row as a local.

    pytest runs with `--showlocals`, and a failing test would otherwise print every held-out
    query and rationale, which spends the set.
    """

    total: int
    kinds: Counter[str]
    paraphrase_by_kind: Counter[str]
    lexical_by_kind: Counter[str]
    shapes: Counter[str]
    corrections_asked: Counter[str | None]
    forget_probes_expecting: tuple[str, ...]
    gate_counts: dict[str, int]


def _heldout_profile() -> _HeldOutProfile:
    probes = load_probes(FIXTURE_DIR / HELDOUT_PROBES)
    actions = {c.correction_id: c.action for c in load_corrections()}
    gates = {gate.name: gate.predicate for gate in PR1_GATES if gate.predicate is not None}
    return _HeldOutProfile(
        total=len(probes),
        kinds=Counter(p.kind for p in probes),
        paraphrase_by_kind=Counter(p.kind for p in probes if not p.lexical),
        lexical_by_kind=Counter(p.kind for p in probes if p.lexical),
        shapes=Counter(p.shape for p in probes),
        corrections_asked=Counter(p.correction_id for p in probes if p.kind == "correction"),
        forget_probes_expecting=tuple(
            p.probe_id
            for p in probes
            if p.expected and actions.get(p.correction_id or "") == "forget"
        ),
        gate_counts={
            name: sum(1 for p in probes if gates[name](p)) for name in HELDOUT_POOLED_MINIMUMS
        },
    )


def _reference_set(probe: Probe) -> frozenset[ItemRef]:
    """The items the lexical tag is judged against: the expected ones, else the forbidden."""
    return frozenset(probe.expected or probe.forbidden)


def _query_stems(probe: Probe) -> set[str]:
    return {_stem(token) for token in _content_tokens(probe.query)}


def _jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _overlap_failures(
    heldout_probes: tuple[Probe, ...], dev_probes: tuple[Probe, ...]
) -> list[str]:
    """One message per held-out and dev pair whose queries overlap past the cap.

    A pair is compared when both probes have the same reference set, or both are noise. A
    message names the two ids and the overlap and never a query or a stem, so fixing a failure
    does not mean reading the dev query.
    """
    failures = []
    for held in heldout_probes:
        reference = _reference_set(held)
        for dev in dev_probes:
            both_noise = held.kind == dev.kind == "noise"
            if not (both_noise or (reference and reference == _reference_set(dev))):
                continue
            overlap = _jaccard(_query_stems(held), _query_stems(dev))
            if overlap > MAX_QUERY_OVERLAP:
                failures.append(
                    f"{held.probe_id} and {dev.probe_id} overlap {overlap:.2f} "
                    f"(cap {MAX_QUERY_OVERLAP})"
                )
    return failures


def _made_up_probe(probe_id: str, query: str, expected: tuple[ItemRef, ...]) -> Probe:
    return Probe(
        probe_id=probe_id,
        kind="fact",
        lexical=False,
        query=query,
        now=datetime(2026, 3, 16, tzinfo=UTC),
        conversation_start=True,
        expected=expected,
        acceptable=(),
        forbidden=(),
        distractors=(),
        correction_id=None,
        rationale="",
    )


class TestHeldOutProbes:
    def test_probe_ids_are_unique_across_both_files(self, heldout: Fixture) -> None:
        """A run file keys its records by probe id, so one id in both sets would merge them."""
        dev_ids = {probe.probe_id for probe in load_probes(FIXTURE_DIR / DEV_PROBES)}
        heldout_ids = {probe.probe_id for probe in heldout.probes}

        assert dev_ids & heldout_ids == set()

    def test_probe_ids_name_the_heldout_set_and_their_kind(self, heldout: Fixture) -> None:
        for probe in heldout.probes:
            assert re.fullmatch(rf"ho-{probe.kind}-\d{{2,}}", probe.probe_id), probe.probe_id

    def test_fact_probe_minimums(self, heldout: Fixture) -> None:
        facts = [probe for probe in heldout.probes if probe.kind == "fact"]
        plain = [probe for probe in facts if probe.shape == "plain"]
        list_items = [probe for probe in facts if probe.shape == "list_item"]
        type_scoped = [probe for probe in facts if probe.shape == "type_scoped"]

        assert len(facts) >= HELDOUT_FACT_MINIMUM
        assert sum(not p.lexical for p in facts) >= HELDOUT_FACT_PARAPHRASE_MINIMUM
        assert len(plain) >= HELDOUT_PLAIN_MINIMUM
        assert sum(not p.lexical for p in plain) >= HELDOUT_PLAIN_PARAPHRASE_MINIMUM
        assert sum(not p.lexical for p in list_items) >= HELDOUT_LIST_PARAPHRASE_MINIMUM
        assert len(type_scoped) >= HELDOUT_TYPE_SCOPED_MINIMUM
        assert all(not p.lexical for p in type_scoped), "every type-scoped probe is a paraphrase"

    def test_each_list_utterance_has_three_list_probes(self, heldout: Fixture) -> None:
        lists = {
            frozenset(ItemRef("fact", fact.fact_key) for fact in facts)
            for facts in _list_utterances(heldout).values()
        }
        asked = Counter(
            frozenset(probe.expected) for probe in heldout.probes if probe.shape == "list_item"
        )

        assert set(asked) == lists
        assert set(asked.values()) == {HELDOUT_LIST_PROBES_PER_LIST}

    def test_every_heldout_kind_is_present_and_no_other(self) -> None:
        """The held-out set carries the kinds PR 1 gates on, not the dev set's seven."""
        profile = _heldout_profile()

        assert set(profile.kinds) == set(HELDOUT_KINDS)

    def test_at_least_forty_percent_of_heldout_probes_are_paraphrase(self) -> None:
        profile = _heldout_profile()
        paraphrase = sum(profile.paraphrase_by_kind.values())

        assert paraphrase / profile.total >= HELDOUT_PARAPHRASE_SHARE

    def test_each_new_shape_meets_its_minimum_on_the_heldout_set(self) -> None:
        profile = _heldout_profile()

        assert set(profile.shapes) == set(PROBE_SHAPES)
        assert profile.shapes["list_item"] >= NEW_SHAPE_MINIMUM
        assert profile.shapes["type_scoped"] >= NEW_SHAPE_MINIMUM

    def test_note_probe_minimums(self) -> None:
        profile = _heldout_profile()

        assert profile.kinds["note"] >= HELDOUT_NOTE_MINIMUM
        assert profile.paraphrase_by_kind["note"] >= HELDOUT_NOTE_PARAPHRASE_MINIMUM

    def test_noise_probe_minimums(self) -> None:
        """T9 floors empty_on_noise at 0.90 on lexical probes; ten leave room for one miss."""
        profile = _heldout_profile()

        assert profile.kinds["noise"] >= HELDOUT_NOISE_MINIMUM
        assert profile.lexical_by_kind["noise"] >= HELDOUT_LEXICAL_NOISE_MINIMUM

    def test_each_correction_has_one_heldout_probe(self) -> None:
        profile = _heldout_profile()

        assert profile.corrections_asked == Counter(c.correction_id for c in load_corrections())

    def test_a_heldout_forget_probe_expects_nothing(self) -> None:
        """After a forget nothing is right to serve, so recall is undefined and only the leak
        is measured; a `correct` probe's expected replacement is checked per probe above.
        """
        profile = _heldout_profile()

        assert profile.forget_probes_expecting == ()

    def test_pooled_recall_classes_meet_their_minimums(self) -> None:
        """PR 1's pooled floors average over the fact, note and correction probes that expect
        something, picked by the gates' own predicates, so the counts are the ones the gate
        measures.
        """
        profile = _heldout_profile()

        for name, minimum in HELDOUT_POOLED_MINIMUMS.items():
            assert profile.gate_counts[name] >= minimum, name

    def test_no_query_re_asks_a_dev_probe(self) -> None:
        """A held-out probe with a dev probe's reference set, or any two noise probes, must
        differ in wording: their content stems overlap by at most half their union.

        The failure names the pair and the overlap, not the dev query, so fixing it means
        rewording the held-out probe without reading the dev set. Both files are loaded inside
        the call, so no probe row is a local of this frame (pytest runs with `--showlocals`).
        """
        failures = _overlap_failures(
            load_probes(FIXTURE_DIR / HELDOUT_PROBES), load_probes(FIXTURE_DIR / DEV_PROBES)
        )

        assert not failures, "; ".join(failures)

    def test_an_overlap_failure_names_the_ids_and_the_figure_not_the_words(self) -> None:
        """Fixing a failure must not mean reading the dev query, so the message carries neither
        query nor any of their stems.
        """
        fact = ItemRef("fact", "k1")
        held = _made_up_probe("ho-fact-98", "quartz lantern meadow", (fact,))
        dev = _made_up_probe("fact-98", "quartz lantern meadow river", (fact,))

        message = "; ".join(_overlap_failures((held,), (dev,)))

        assert "ho-fact-98" in message
        assert "fact-98" in message.replace("ho-fact-98", "")
        assert "0.75" in message
        for word in ("quartz", "lantern", "meadow", "river"):
            assert word not in message

    def test_the_overlap_cap_reads_reference_sets_and_crude_stems(self) -> None:
        """A forget probe's reference set is its forbidden items, and an inflection is no
        rewording: "bicycles" and "bicycle" share a stem.
        """
        fact = ItemRef("fact", "k1")
        other = ItemRef("fact", "k2")
        base = _made_up_probe("x", "which red bicycle", (fact,))
        forget = replace(base, kind="correction", expected=(), forbidden=(fact,))

        assert _reference_set(base) == _reference_set(forget) == frozenset({fact})
        assert _reference_set(replace(base, expected=(other,))) != _reference_set(base)
        reworded = replace(base, query="red bicycles bell")
        assert _jaccard(_query_stems(base), _query_stems(reworded)) == 2 / 3
