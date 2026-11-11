"""The two pack hashes, stored: on the candidate row, and read back by the guards.

`test_pack_hash_identity.py` pins the comparisons over stored sides it builds by
hand. This file pins the stored side itself: `BacklogStore.begin_cutover` and
`cutover.begin_cutover` write both hashes to the `epoch_cutover` row, `admin
cutover begin` takes them from the pack its resolver holds for the candidate's
ontology version, and the three guards refuse from what a real `EventStore`
holds. Every store is a file under `tmp_path`, or in memory.
"""

from __future__ import annotations

import asyncio
import dataclasses
import io
import sqlite3

import pytest

from backend.event_store.store import EventStore
from backend.extraction_backlog import admin
from backend.extraction_backlog.cutover import (
    CutoverRefusedError,
    begin_cutover,
    candidate_pack_refusal,
)
from backend.extraction_backlog.store import BacklogStore, Cutover, Epoch, PackHashes
from backend.knowledge.extraction_cache import ExtractionCache
from backend.knowledge.ontologies import registry
from backend.knowledge.version_stamps import ONTOLOGY_VERSION as CODE_ONTOLOGY_VERSION
from tests.unit.event_store.test_epoch_ledger import make_pre_pack_hash
from tests.unit.extraction_backlog.conftest import (
    EMBEDDING_MODEL,
    ONTOLOGY_VERSION,
    _build_world,
    composed,
    wait_until,
)
from tests.unit.extraction_backlog.fakes import fake_pack_resolver

PROMPT = "a" * 64
RAILS = "b" * 64
OTHER = "f" * 64
NEXT_PROMPT = "c" * 64
NEXT_RAILS = "d" * 64
CANDIDATE = "svc-model-2"
NOW = "2026-11-11T00:00:00+00:00"


def _hashes(carrier) -> tuple[str | None, str | None]:
    return (carrier.pack_prompt_hash, carrier.pack_rails_hash)


def _cutover_rows(world) -> list[dict]:
    return world.event_store.list_epoch_cutovers()


@pytest.fixture
def stored_world(tmp_path):
    """A world whose event store is a file and whose active epoch stores (PROMPT, RAILS).

    The fake service reports the same two hashes, and a dispatcher built with
    no `writer_stamps` writes under them: every side agrees until a test moves
    one.
    """
    return _build_world(
        with_deriver=False,
        db_path=str(tmp_path / "events.db"),
        pack_prompt_hash=PROMPT,
        pack_rails_hash=RAILS,
    )


async def _stop(world) -> None:
    for dispatcher in world.dispatchers:
        await dispatcher.stop(timeout=2.0)
    for client in world.clients:
        await client.aclose()


class TestRowsFromBeforeTheColumns:
    """Decision 3: a row with no hash column is a legacy row, not an error."""

    def test_rows_of_a_database_this_code_never_initialized_read_with_no_hash(self, tmp_path):
        # Arrange: both epoch tables from the earlier DDL, one row each, in a
        # file `EventStore.initialize()` never ran on.
        path = str(tmp_path / "old.db")
        make_pre_pack_hash(path)
        conn = sqlite3.connect(path)
        ledger_columns = [row[1] for row in conn.execute("PRAGMA table_info(epoch_ledger)")]
        conn.close()
        events = EventStore(db_path=path)

        # Act
        epoch = Epoch.from_row(events.get_current_epoch())
        cutover = BacklogStore(events, ExtractionCache(":memory:")).open_cutover()
        events.close()

        # Assert
        assert "pack_prompt_hash" not in ledger_columns
        assert _hashes(epoch) == (None, None)
        assert cutover is not None
        assert _hashes(cutover) == (None, None)
        assert _hashes(cutover.target) == (None, None)


class TestTheCandidateRowStoresItsHashes:
    """`BacklogStore.begin_cutover` and `cutover.begin_cutover` write what they are given."""

    def _store_begin(self, world, **hashes) -> Cutover:
        cutover = world.store.begin_cutover(
            ontology_version=ONTOLOGY_VERSION,
            extraction_version="candidate-ev",
            model_hash=composed(CANDIDATE),
            bare_model_hash=CANDIDATE,
            source_epoch_id=world.epoch_id,
            requested_at=NOW,
            **hashes,
        )
        assert cutover is not None
        return cutover

    def _begin(self, world, **overrides) -> Cutover:
        arguments = {
            "bare_model_hash": CANDIDATE,
            "extraction_version": world.service.extraction_version,
            "ontology_version": ONTOLOGY_VERSION,
            "embedding_model_name": world.embedding_model,
            "now_iso": NOW,
        }
        arguments.update(overrides)
        return begin_cutover(world.store, **arguments)

    def test_the_backlog_store_hands_both_hashes_to_the_row(self, stored_world):
        cutover = self._store_begin(
            stored_world, pack_prompt_hash=NEXT_PROMPT, pack_rails_hash=NEXT_RAILS
        )

        (row,) = _cutover_rows(stored_world)
        assert (row["pack_prompt_hash"], row["pack_rails_hash"]) == (NEXT_PROMPT, NEXT_RAILS)
        assert _hashes(cutover) == (NEXT_PROMPT, NEXT_RAILS)
        assert _hashes(stored_world.store.open_cutover()) == (NEXT_PROMPT, NEXT_RAILS)

    def test_the_backlog_store_given_no_hash_stores_null(self, stored_world):
        cutover = self._store_begin(stored_world)

        (row,) = _cutover_rows(stored_world)
        assert (row["pack_prompt_hash"], row["pack_rails_hash"]) == (None, None)
        assert _hashes(cutover) == (None, None)

    def test_the_backlog_stores_pack_hashes_are_keyword_only_and_default_to_none(self):
        import inspect

        parameters = inspect.signature(BacklogStore.begin_cutover).parameters

        for name in ("pack_prompt_hash", "pack_rails_hash"):
            assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
            assert parameters[name].default is None

    def test_begin_cutover_stores_the_two_hashes_it_compared(self, stored_world):
        cutover = self._begin(stored_world, pack_prompt_hash=NEXT_PROMPT, pack_rails_hash=RAILS)

        (row,) = _cutover_rows(stored_world)
        assert (row["pack_prompt_hash"], row["pack_rails_hash"]) == (NEXT_PROMPT, RAILS)
        assert _hashes(cutover) == (NEXT_PROMPT, RAILS)

    def test_begin_cutover_given_no_hash_stores_null(self, stored_world):
        cutover = self._begin(stored_world)

        (row,) = _cutover_rows(stored_world)
        assert (row["pack_prompt_hash"], row["pack_rails_hash"]) == (None, None)
        assert _hashes(cutover) == (None, None)

    def test_a_refused_begin_writes_no_row(self, stored_world):
        """The same five stamps as the active epoch: nothing to cut over, nothing stored."""
        with pytest.raises(CutoverRefusedError, match="nothing to cut over"):
            self._begin(
                stored_world,
                bare_model_hash=stored_world.service.model_hash,
                pack_prompt_hash=PROMPT,
                pack_rails_hash=RAILS,
            )

        assert _cutover_rows(stored_world) == []


class TestTheGuardsRefuseFromStoredValues:
    """Acceptance 4: the active epoch stores (PROMPT, RAILS) in a real `EventStore` file."""

    def test_the_stored_epoch_reads_back_with_its_hashes(self, stored_world):
        assert _hashes(stored_world.store.active_epoch()) == (PROMPT, RAILS)

    @pytest.mark.asyncio
    async def test_every_side_on_the_stored_pack_dispatches_and_applies(self, stored_world, ts):
        world = stored_world
        event_id = world.log_turn(
            session_id="s1", turn_index=0, timestamp=ts(0), utterance="I really use rust"
        )
        dispatcher = world.build_dispatcher()

        try:
            await dispatcher.start()
            assert await dispatcher.drain(timeout=5.0)
        finally:
            await _stop(world)

        assert world.curation.event_ids == [event_id]

    @pytest.mark.asyncio
    async def test_the_writer_guard_refuses_a_backend_on_another_rails_hash(self, stored_world, ts):
        # Arrange
        world = stored_world
        world.log_turn(
            session_id="s1", turn_index=0, timestamp=ts(0), utterance="I really use rust"
        )
        writer = dataclasses.replace(world.active_epoch_stamps(), pack_rails_hash=OTHER)
        dispatcher = world.build_dispatcher(writer_stamps=writer)

        # Act
        try:
            await dispatcher.start()
            await wait_until(lambda: dispatcher.state == "stalled")
            await asyncio.sleep(0.1)  # several recheck periods: the guard holds
        finally:
            await _stop(world)

        # Assert
        assert f"pack_rails_hash={OTHER!r}" in dispatcher.reason
        assert f"pack_rails_hash={RAILS!r}" in dispatcher.reason
        assert "pack_prompt_hash" not in dispatcher.reason
        assert world.service.received == []
        assert world.service.info_calls == 0
        assert world.curation.calls == []

    @pytest.mark.asyncio
    async def test_the_service_check_refuses_a_service_on_another_prompt_hash(
        self, stored_world, ts
    ):
        # Arrange
        world = stored_world
        world.log_turn(
            session_id="s1", turn_index=0, timestamp=ts(0), utterance="I really use rust"
        )
        world.service.pack_prompt_hash = OTHER
        dispatcher = world.build_dispatcher()

        # Act
        try:
            await dispatcher.start()
            await wait_until(lambda: dispatcher.state == "epoch_mismatch")
            await asyncio.sleep(0.1)
        finally:
            await _stop(world)

        # Assert
        assert f"pack_prompt_hash={OTHER!r}" in dispatcher.reason
        assert f"pack_prompt_hash={PROMPT!r}" in dispatcher.reason
        assert "pack_rails_hash" not in dispatcher.reason
        assert world.service.received == []
        assert world.curation.calls == []

    def test_the_rebuild_refuses_a_pack_that_differs_from_the_open_candidates(self, stored_world):
        # Arrange: a candidate begun under another pack than the active epoch's.
        world = stored_world
        begin_cutover(
            world.store,
            bare_model_hash=CANDIDATE,
            extraction_version=world.service.extraction_version,
            ontology_version=ONTOLOGY_VERSION,
            embedding_model_name=world.embedding_model,
            now_iso=NOW,
            pack_prompt_hash=NEXT_PROMPT,
            pack_rails_hash=NEXT_RAILS,
        )
        candidate = world.store.open_cutover()

        # Act
        refusal = candidate_pack_refusal(candidate, PackHashes(PROMPT, RAILS))

        # Assert
        assert refusal is not None
        assert f"pack_prompt_hash={NEXT_PROMPT!r}" in refusal
        assert f"pack_rails_hash={RAILS!r}" in refusal
        assert f"cutover {candidate.cutover_id}" in refusal
        assert candidate_pack_refusal(candidate, PackHashes(NEXT_PROMPT, NEXT_RAILS)) is None

    def test_the_rebuild_wiring_refuses_from_the_stored_candidate_before_it_connects(
        self, stored_world, monkeypatch
    ):
        """`build_rebuild_deps_from_env` over the row the store returns, not a built one."""
        from backend.extraction_backlog.cutover import build_rebuild_deps_from_env
        from tests.mocks.config import build_test_config

        world = stored_world
        begin_cutover(
            world.store,
            bare_model_hash=CANDIDATE,
            extraction_version=world.service.extraction_version,
            ontology_version=ONTOLOGY_VERSION,
            embedding_model_name=world.embedding_model,
            now_iso=NOW,
            pack_prompt_hash=NEXT_PROMPT,
            pack_rails_hash=NEXT_RAILS,
        )
        candidate = world.store.open_cutover()
        config = dataclasses.replace(
            build_test_config(embedding_model=EMBEDDING_MODEL),
            ontology_version=candidate.ontology_version,
            extraction_version=candidate.extraction_version,
            model_hash=candidate.bare_model_hash,
        )
        active = dataclasses.replace(
            registry.active_pack(), pack_prompt_hash=NEXT_PROMPT, pack_rails_hash=OTHER
        )
        reached: list[str] = []
        monkeypatch.setattr("backend.knowledge.config.get_config", lambda: config)
        monkeypatch.setattr(registry, "active_pack", lambda: active)
        monkeypatch.setattr(
            "backend.knowledge.eval_isolation.assert_rebuild_target_not_live",
            lambda *args, **kwargs: reached.append("connections"),
        )

        with pytest.raises(CutoverRefusedError) as raised:
            build_rebuild_deps_from_env(world.store, "bolt://staging.invalid:7687", candidate)

        assert f"pack_rails_hash={OTHER!r}" in str(raised.value)
        assert f"pack_rails_hash={NEXT_RAILS!r}" in str(raised.value)
        assert "pack_prompt_hash" not in str(raised.value)
        assert reached == []


def _cli_begin(world, *flags: str, ontology_version: str | None = ONTOLOGY_VERSION, **main_kwargs):
    """`admin cutover begin` for a candidate on another model, with `main_kwargs` for `main`."""
    argv = [
        "cutover",
        "begin",
        "--model-hash",
        CANDIDATE,
        "--extraction-version",
        world.service.extraction_version,
        *flags,
    ]
    if ontology_version is not None:
        argv += ["--ontology-version", ontology_version]
    out = io.StringIO()
    code = admin.main(
        argv, store=world.store, out=out, embedding_model_name=EMBEDDING_MODEL, **main_kwargs
    )
    return code, out.getvalue()


class TestCutoverBeginCli:
    """Acceptance 5: `cutover begin` resolves the candidate's pack and stores its hashes."""

    def test_the_candidate_row_holds_the_hashes_of_the_pack_the_resolver_returns(
        self, stored_world
    ):
        resolver = fake_pack_resolver({ONTOLOGY_VERSION: (NEXT_PROMPT, NEXT_RAILS)})

        code, text = _cli_begin(stored_world, pack_for_version=resolver)

        assert code == 0, text
        (row,) = _cutover_rows(stored_world)
        assert (row["pack_prompt_hash"], row["pack_rails_hash"]) == (NEXT_PROMPT, NEXT_RAILS)

    def test_the_resolver_is_asked_for_the_candidates_ontology_version(self, stored_world):
        """Two packs held: the candidate takes the one of the version it names."""
        stored_world.bind_active_epoch()
        resolver = fake_pack_resolver(
            {ONTOLOGY_VERSION: (PROMPT, RAILS), "1.5.0": (NEXT_PROMPT, NEXT_RAILS)}
        )

        code, text = _cli_begin(stored_world, ontology_version="1.5.0", pack_for_version=resolver)

        assert code == 0, text
        candidate = stored_world.store.open_cutover()
        assert candidate.ontology_version == "1.5.0"
        assert _hashes(candidate) == (NEXT_PROMPT, NEXT_RAILS)

    def test_no_ontology_version_flag_resolves_the_codes_version(self, stored_world):
        asked: list[str] = []

        def resolver(version: str) -> PackHashes:
            asked.append(version)
            return PackHashes(NEXT_PROMPT, NEXT_RAILS)

        code, text = _cli_begin(stored_world, ontology_version=None, pack_for_version=resolver)

        assert code == 0, text
        assert asked == [CODE_ONTOLOGY_VERSION]
        assert _hashes(stored_world.store.open_cutover()) == (NEXT_PROMPT, NEXT_RAILS)

    def test_an_ontology_version_the_resolver_holds_no_pack_for_is_refused(self, stored_world):
        resolver = fake_pack_resolver({ONTOLOGY_VERSION: (PROMPT, RAILS)})

        code, text = _cli_begin(stored_world, ontology_version="9.9.9", pack_for_version=resolver)

        assert code == 2
        assert "[cutover] REFUSED:" in text
        assert "no ontology pack for version 9.9.9" in text
        assert _cutover_rows(stored_world) == []
        assert stored_world.store.open_cutover() is None

    def test_the_default_resolver_is_the_registrys(self, stored_world):
        """No `pack_for_version`: the hashes are `registry.get_pack`'s for the version."""
        pack = registry.get_pack(ONTOLOGY_VERSION)

        code, text = _cli_begin(stored_world)

        assert code == 0, text
        assert _hashes(stored_world.store.open_cutover()) == (
            pack.pack_prompt_hash,
            pack.pack_rails_hash,
        )

    def test_the_default_resolver_refuses_a_version_the_registry_does_not_hold(self, stored_world):
        assert "9.9.9" not in registry.available_versions()

        code, text = _cli_begin(stored_world, ontology_version="9.9.9")

        assert code == 2
        assert "no ontology pack for version 9.9.9" in text
        assert _cutover_rows(stored_world) == []

    def test_the_registry_holds_a_pack_for_the_codes_ontology_version(self):
        """Otherwise `cutover begin` with no `--ontology-version` could never open a candidate."""
        pack = registry.get_pack(CODE_ONTOLOGY_VERSION)

        assert pack.pack_prompt_hash is not None
        assert pack.pack_rails_hash is not None

    def test_a_settings_change_under_the_stored_pack_stores_the_same_hashes(self, stored_world):
        """The triple and both hashes equal the active epoch's: a settings change, and stored."""
        stored_world.bind_active_epoch()
        resolver = fake_pack_resolver({ONTOLOGY_VERSION: (PROMPT, RAILS)})
        out = io.StringIO()

        code = admin.main(
            [
                "cutover",
                "begin",
                "--model-hash",
                stored_world.service.model_hash,
                "--extraction-version",
                stored_world.service.extraction_version,
                "--ontology-version",
                ONTOLOGY_VERSION,
                "--settings-change",
            ],
            store=stored_world.store,
            out=out,
            embedding_model_name=EMBEDDING_MODEL,
            pack_for_version=resolver,
        )

        assert code == 0, out.getvalue()
        assert _hashes(stored_world.store.open_cutover()) == (PROMPT, RAILS)
