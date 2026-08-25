"""The golden-log generator: determinism, isolation, stamp authority, gap-schedule shape.

The generator is scripted rather than hand-transcribed (R1.4 T13's rule) and its output is
a checked-in artifact rather than a SQLite file, because "re-running produces byte-identical
output" is a claim only a text artifact can support.
"""

from __future__ import annotations

import inspect
from datetime import datetime
from pathlib import Path

import pytest

from backend.knowledge.config import KnowledgeConfig
from backend.knowledge.extraction_cache import cache_key
from backend.knowledge.version_stamps import EXTRACTION_VERSION, ONTOLOGY_VERSION
from scripts.golden_log.generate import (
    ARTIFACT_PATH,
    EVENT_ID_PREFIX,
    GoldenLogError,
    assert_isolated_root,
    build_golden_turns,
    event_id_for,
    load_artifact,
    load_schedule,
    materialize_isolated,
    render_artifact,
    session_id_for,
)
from scripts.golden_log.native_shape import native_predicate
from tests.unit.knowledge.test_version_stamps import _literal_stamps

EXPECTED_TURN_COUNT = 87
GOLD_RECORD_COUNT = 60


@pytest.fixture(scope="module")
def turns():
    return build_golden_turns()


@pytest.fixture(scope="module")
def schedule():
    return load_schedule()


class TestSchedule:
    def test_every_gold_record_is_placed(self, schedule):
        # Assert: the decision was "use all 60", so a dropped tag is a regression.
        referenced = {entry["id"] for entry in schedule["turns"] if "record" not in entry}
        assert len(referenced) == GOLD_RECORD_COUNT

    def test_rejects_turns_out_of_timestamp_order(self, tmp_path):
        # Arrange: rowid order must match timestamp order -- the replay reads rowid order.
        path = tmp_path / "schedule.yaml"
        path.write_text(
            "gold_corpus: data/ingest/extraction-gold-2026-06-14.jsonl\n"
            'log_end: "2026-07-15T12:00:00+00:00"\n'
            "turns:\n"
            '  - {id: b, at: "2026-01-02T00:00:00+00:00", record: {}}\n'
            '  - {id: a, at: "2026-01-01T00:00:00+00:00", record: {}}\n',
            encoding="utf-8",
        )

        # Act / Assert
        with pytest.raises(GoldenLogError, match="strictly increasing timestamp order"):
            load_schedule(path)

    def test_rejects_a_duplicate_turn_id(self, tmp_path):
        # Arrange: event ids are derived from the id, so a duplicate collides on cache key.
        path = tmp_path / "schedule.yaml"
        path.write_text(
            "gold_corpus: data/ingest/extraction-gold-2026-06-14.jsonl\n"
            'log_end: "2026-07-15T12:00:00+00:00"\n'
            "turns:\n"
            '  - {id: a, at: "2026-01-01T00:00:00+00:00", record: {}}\n'
            '  - {id: a, at: "2026-01-02T00:00:00+00:00", record: {}}\n',
            encoding="utf-8",
        )

        # Act / Assert
        with pytest.raises(GoldenLogError, match="duplicate turn id"):
            load_schedule(path)

    def test_rejects_an_empty_schedule(self, tmp_path):
        # Arrange: a log that replays nothing satisfies every downstream assertion.
        path = tmp_path / "schedule.yaml"
        path.write_text(
            "gold_corpus: data/ingest/extraction-gold-2026-06-14.jsonl\n"
            'log_end: "2026-07-15T12:00:00+00:00"\n'
            "turns: []\n",
            encoding="utf-8",
        )

        # Act / Assert
        with pytest.raises(GoldenLogError, match="replays nothing"):
            load_schedule(path)

    def test_references_a_tag_the_gold_corpus_does_not_have(self, tmp_path):
        # Arrange
        path = tmp_path / "schedule.yaml"
        path.write_text(
            "gold_corpus: data/ingest/extraction-gold-2026-06-14.jsonl\n"
            'log_end: "2026-07-15T12:00:00+00:00"\n'
            "turns:\n"
            '  - {id: ext-99-does-not-exist, at: "2026-01-01T00:00:00+00:00"}\n',
            encoding="utf-8",
        )

        # Act / Assert
        with pytest.raises(GoldenLogError, match="no matching tag"):
            build_golden_turns(schedule_path=path)


class TestDerivedIdentifiers:
    def test_event_ids_are_derived_from_the_schedule_id(self, turns, schedule):
        # Assert: derived, never generated -- the cache key must survive regeneration.
        expected = [event_id_for(entry["id"]) for entry in schedule["turns"]]
        assert [turn.event_id for turn in turns] == expected
        assert all(turn.event_id.startswith(EVENT_ID_PREFIX) for turn in turns)

    def test_gold_turn_event_ids_carry_the_gold_tag(self, turns):
        # Assert: `golden-ext-01-uses` traces straight back to the gold record.
        assert any(turn.event_id == "golden-ext-01-uses" for turn in turns)

    def test_session_id_is_the_calendar_month_of_the_timestamp(self, turns):
        # Assert: the derivation rule, pinned rather than left implicit.
        assert session_id_for("2025-09-02T08:00:00+00:00") == "golden-2025-09"
        assert all(turn.session_id == session_id_for(turn.timestamp) for turn in turns)

    def test_turn_index_is_contiguous_from_zero_within_each_session(self, turns):
        # Assert: `get_turns` orders by turn_index, so gaps would reorder a session.
        by_session: dict[str, list[int]] = {}
        for turn in turns:
            by_session.setdefault(turn.session_id, []).append(turn.turn_index)
        for session_id, indices in by_session.items():
            assert indices == list(range(len(indices))), f"{session_id}: non-contiguous"


class TestArtifactIsDeterministic:
    def test_two_renders_are_byte_identical(self):
        # Act: build twice from scratch, not render the same object twice.
        first = render_artifact(build_golden_turns())
        second = render_artifact(build_golden_turns())

        # Assert
        assert first == second

    def test_checked_in_artifact_matches_a_fresh_render(self, turns):
        # Assert: the committed fixture is regenerable, so it can be reviewed as a diff.
        assert ARTIFACT_PATH.read_bytes() == render_artifact(turns).encode("utf-8")

    def test_artifact_holds_every_turn(self, turns):
        assert len(load_artifact()) == EXPECTED_TURN_COUNT
        assert len(turns) == EXPECTED_TURN_COUNT

    def test_artifact_round_trips(self, turns):
        # Assert: loading the artifact yields the same turns the schedule builds.
        assert [t.to_artifact_row() for t in load_artifact()] == [
            t.to_artifact_row() for t in turns
        ]

    def test_artifact_has_no_carriage_returns(self):
        # Assert: LF only, so the byte-identity claim holds on Windows checkouts too.
        assert b"\r" not in ARTIFACT_PATH.read_bytes()


class TestStampsHaveOneAuthority:
    """The generator must not be able to state its own stamp triple."""

    def test_materialize_takes_no_stamp_parameter(self):
        # Assert: no parameter means no call site can inject a drifted stamp.
        params = set(inspect.signature(materialize_isolated).parameters)
        assert not params & {"ontology_version", "extraction_version", "model_hash", "stamps"}

    def test_no_golden_log_module_restates_a_stamp(self):
        # Arrange: `backend/` has an AST guard; `scripts/golden_log/` needs the same one.
        # Reuses that guard's helper rather than re-deriving the literal shapes.
        sources = sorted(Path("scripts/golden_log").rglob("*.py"))
        assert sources, "no golden-log sources found to scan"

        # Act / Assert
        for path in sources:
            for fragment in ("ontology_version", "extraction_version"):
                offenders = _literal_stamps(path, fragment)
                assert not offenders, f"{path}: restates {fragment} as a literal: {offenders}"

    def test_epoch_carries_the_single_authority_stamps(self, turns, tmp_path):
        # Act
        materialized = materialize_isolated(turns, root=tmp_path / "isolated")

        # Assert: identical to what LogRegenerator will hash into the cache key.
        assert materialized.epoch["ontology_version"] == ONTOLOGY_VERSION
        assert materialized.epoch["extraction_version"] == EXTRACTION_VERSION
        assert materialized.epoch["model_hash"] == KnowledgeConfig.from_env().model_hash

    def test_epoch_model_hash_is_the_bare_config_value_not_the_composed_writer_stamp(
        self, turns, tmp_path
    ):
        # Arrange: two triples exist and they differ in model_hash. `factories.py` composes
        # f"{config.model_hash}|emb:{config.embedding.model_name}" into RebuildStamps, which
        # is what lands on every edge. `EventStore` epoch rows carry the BARE value, and the
        # cache key hashes the EPOCH's. Harmonizing the epoch onto the composed form would
        # take this corpus permanently cold, deterministically, on every turn.
        materialized = materialize_isolated(turns, root=tmp_path / "isolated")

        # Assert
        assert materialized.epoch["model_hash"] == KnowledgeConfig.from_env().model_hash
        assert "|emb:" not in materialized.epoch["model_hash"]

    def test_no_golden_log_module_reaches_for_the_writer_stamps(self):
        # Assert: RebuildStamps is the OTHER triple. Reading it here is the whole bug.
        for path in sorted(Path("scripts/golden_log").rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            assert "RebuildStamps" not in source, f"{path}: reads the writer stamp triple"

    def test_epoch_is_written_to_the_isolated_store_not_read_from_live(self, turns, tmp_path):
        # Assert: the fixture's epoch is epoch 1 OF ITS OWN LEDGER. Live epoch_id 1 carries
        # a pre-collapse extraction_version and must never be depended on.
        materialized = materialize_isolated(turns, root=tmp_path / "isolated")
        assert materialized.event_store.list_epochs() == [materialized.epoch]


class TestIsolation:
    """The golden log is a fixture and is never written to live state."""

    @pytest.mark.parametrize(
        "live_root",
        [
            pytest.param(Path("data"), id="repo-data-dir"),
            pytest.param(Path("data") / "golden-log", id="under-repo-data-dir"),
            pytest.param(Path.home() / ".mist", id="default-event-store-dir"),
        ],
    )
    def test_refuses_a_live_data_root(self, live_root):
        # Act / Assert: mechanical, not documentary.
        if not live_root.exists():
            pytest.skip(f"{live_root} not present in this environment")
        with pytest.raises(GoldenLogError, match="under the live data directory"):
            assert_isolated_root(live_root)

    def test_accepts_a_tmp_root(self, tmp_path):
        assert_isolated_root(tmp_path)  # no raise

    def test_refuses_to_materialize_zero_turns(self, tmp_path):
        # Assert: fail closed. A log that replays nothing passes every other assertion.
        with pytest.raises(GoldenLogError, match="replays nothing"):
            materialize_isolated([], root=tmp_path / "isolated")


class TestMaterialize:
    def test_writes_every_turn_to_the_isolated_event_store(self, turns, tmp_path):
        # Act
        materialized = materialize_isolated(turns, root=tmp_path / "isolated")

        # Assert
        assert materialized.turn_count == EXPECTED_TURN_COUNT
        assert materialized.event_store.get_turn_count() == EXPECTED_TURN_COUNT

    def test_rowid_order_matches_timestamp_order(self, turns, tmp_path):
        # Assert: the replay reads rowid order and stamps recorded_at from the timestamp,
        # so a disagreement would reconcile facts out of chronological order.
        materialized = materialize_isolated(turns, root=tmp_path / "isolated")

        replay_order = materialized.event_store.get_all_turns_for_reextraction()
        timestamps = [row["timestamp"] for row in replay_order]

        assert timestamps == sorted(timestamps)
        assert [row["event_id"] for row in replay_order] == [t.event_id for t in turns]

    def test_cache_covers_every_turn_under_the_epoch_stamps(self, turns, tmp_path):
        # Assert: 100% coverage, which is what keeps rebuild off ColdCacheError.
        materialized = materialize_isolated(turns, root=tmp_path / "isolated")
        epoch = materialized.epoch

        hits = [
            materialized.extraction_cache.get(
                turn.event_id,
                epoch["extraction_version"],
                epoch["model_hash"],
            )
            for turn in turns
        ]

        assert all(hit is not None for hit in hits)
        assert len(hits) == EXPECTED_TURN_COUNT

    def test_the_write_stamp_pair_is_the_read_stamp_pair(self, turns, tmp_path):
        # Assert: the key the cache was WRITTEN under is the key the rebuild COMPUTES.
        # `LogRegenerator` derives its lookup from epoch[extraction|model_hash] (D3 dropped
        # ontology_version from the key), so this recomputes that key independently and
        # requires a row under it.
        materialized = materialize_isolated(turns, root=tmp_path / "isolated")
        epoch = materialized.epoch

        rebuild_key = cache_key(
            turns[0].event_id,
            epoch["extraction_version"],
            epoch["model_hash"],
        )
        stored = (
            materialized.extraction_cache._get_connection()
            .execute(
                "SELECT cache_key FROM extraction_cache WHERE event_id = ?", (turns[0].event_id,)
            )
            .fetchone()
        )

        assert stored is not None
        assert stored["cache_key"] == rebuild_key

    def test_the_composed_writer_model_hash_would_miss(self, turns, tmp_path):
        # Assert: names the concrete failure mode rather than trusting it cannot happen.
        # Keying on `factories`' composed model_hash produces a different key -- a total,
        # permanent cold cache. This is why the epoch triple is the one that matters.
        materialized = materialize_isolated(turns, root=tmp_path / "isolated")
        epoch = materialized.epoch
        composed = f"{epoch['model_hash']}|emb:all-MiniLM-L6-v2"

        assert (
            materialized.extraction_cache.get(
                turns[0].event_id,
                epoch["extraction_version"],
                composed,
            )
            is None
        )

    def test_cached_payload_is_the_native_shape_the_replay_feeds_forward(self, turns, tmp_path):
        # Assert: rebuild hands `cached["relationships"]` straight to ValidationResult, so
        # what comes back out of the cache must already read as native.
        materialized = materialize_isolated(turns, root=tmp_path / "isolated")
        epoch = materialized.epoch
        cached = materialized.extraction_cache.get(
            "golden-ext-01-uses",
            epoch["extraction_version"],
            epoch["model_hash"],
        )

        assert native_predicate(cached["relationships"][0]) == "USES"
        assert cached["entities"][0]["type"] == "User"

    def test_sessions_are_marked_as_fixture_traffic(self, turns, tmp_path):
        # Assert: `origin` exists so genuine usage can be told from probe traffic.
        materialized = materialize_isolated(turns, root=tmp_path / "isolated")
        session = materialized.event_store.get_session(turns[0].session_id)
        assert session is not None
        assert session.origin == "test"


class TestGapScheduleShape:
    """The anti-calibration property, machine-checked so a later edit cannot erode it.

    GAP-SCHEDULE.md states it in prose: no single staleness window may classify every
    never-restated LEARNING assertion the same way. That holds only while the elapsed times
    stay spread, so the spread is asserted here rather than trusted.
    """

    @staticmethod
    def _learning_facts(turns) -> dict[tuple[str, str], list]:
        facts: dict[tuple[str, str], list] = {}
        for turn in turns:
            for rel in turn.relationships:
                if native_predicate(rel) != "LEARNING":
                    continue
                key = (rel["source"], rel["target"])
                kind = rel.get("properties", {}).get("assertion_kind", "assert")
                facts.setdefault(key, []).append((turn.timestamp, kind))
        return facts

    def test_never_restated_learning_elapsed_times_span_an_order_of_magnitude(
        self, turns, schedule
    ):
        # Arrange: facts asserted exactly once and never ceased or retracted.
        log_end = datetime.fromisoformat(schedule["log_end"])
        facts = self._learning_facts(turns)
        elapsed = [
            (log_end - datetime.fromisoformat(events[0][0])).days
            for events in facts.values()
            if len(events) == 1 and events[0][1] == "assert"
        ]

        # Assert: window-independent. Any constant splits this set.
        assert len(elapsed) >= 5, f"too few never-restated LEARNING facts: {elapsed}"
        assert max(elapsed) / max(min(elapsed), 1) >= 10, f"gaps are too clustered: {elapsed}"

    def test_a_restated_learning_control_exists(self, turns):
        # Assert: negative control A -- a window that ages out everything must fail.
        facts = self._learning_facts(turns)
        restated = [key for key, events in facts.items() if len(events) >= 3]
        assert restated, "no LEARNING fact is asserted three or more times"

    def test_the_spoken_cease_control_exists(self, turns):
        # Assert: negative control B -- C3 already closes this; R1.5 must not double-handle.
        facts = self._learning_facts(turns)
        kinds = facts[("user", "clojure")]
        assert [kind for _ts, kind in kinds] == ["assert", "cease"]

    def test_a_multi_assertion_sequence_exists_for_a_non_learning_predicate(self, turns):
        # Assert: spec 4b -- a re-assertion takes the REINFORCE path, which does not
        # advance recorded_at, so only a sequence like this can prove last_asserted_at moves.
        timestamps = [
            turn.timestamp
            for turn in turns
            for rel in turn.relationships
            if native_predicate(rel) == "USES" and rel["target"] == "postgresql"
        ]
        assert len(timestamps) == 4
        assert timestamps == sorted(timestamps)

    def test_every_cease_and_retract_has_an_earlier_assertion_of_the_same_fact(self, turns):
        # Assert: without a prior, a cease plans FLAG_AMBIGUOUS instead of closing, and the
        # close-reason coverage the schedule claims would not actually fire.
        asserted: set[tuple[str, str, str]] = set()
        for turn in turns:
            for rel in turn.relationships:
                key = (rel["source"], native_predicate(rel), rel["target"])
                kind = rel.get("properties", {}).get("assertion_kind", "assert")
                if kind == "assert":
                    asserted.add(key)
                    continue
                assert key in asserted, (
                    f"{turn.event_id}: {kind} of {key} has no earlier assertion, so it "
                    "would flag cease_without_prior instead of closing"
                )
