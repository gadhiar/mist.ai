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


UNBOUND_SHARES_ROWS = (
    "epoch {n} has no identity yet, so this candidate cannot be shown to share its rows; "
    "wait until admin status shows it bound, or run admin epoch accept"
)

# The three ways a candidate keeps the active pair and the active prompt hash
# and still differs from the active epoch: class 3.
SHARES_THE_PAIRS_ROWS = pytest.mark.parametrize(
    ("ontology_version", "rails"),
    [
        pytest.param(ONTOLOGY_VERSION, NEXT_RAILS, id="another-rails-hash"),
        pytest.param("1.5.0", RAILS, id="another-ontology-version"),
        pytest.param("1.5.0", NEXT_RAILS, id="both"),
    ],
)


def _cli_begin_on_the_active_pair(
    world,
    *flags: str,
    pack: tuple[str | None, str | None],
    ontology_version: str = ONTOLOGY_VERSION,
):
    """`admin cutover begin` with the active epoch's extraction version and model hash.

    `pack` is the `(pack_prompt_hash, pack_rails_hash)` the resolver holds for
    the candidate's ontology version, so the candidate differs from the active
    epoch in its ontology version, in a pack hash, or in nothing.
    """
    out = io.StringIO()
    code = admin.main(
        [
            "cutover",
            "begin",
            "--model-hash",
            world.service.model_hash,
            "--extraction-version",
            world.service.extraction_version,
            "--ontology-version",
            ontology_version,
            *flags,
        ],
        store=world.store,
        out=out,
        embedding_model_name=EMBEDDING_MODEL,
        pack_for_version=fake_pack_resolver({ontology_version: pack}),
    )
    return code, out.getvalue()


class TestCandidateClassesAtBegin:
    """A candidate's class at `cutover begin`, against an epoch that stores (PROMPT, RAILS).

    Class 1 has the same stamps, the triple and both hashes: a settings change.
    Class 2 keeps the extraction version and the model hash and changes the
    prompt hash: refused. Class 3 keeps the pair otherwise, with another
    ontology version or another rails hash: it shares the pair's cached rows and
    needs the active epoch bound. Class 4 is any other candidate.
    """

    # -- class 1: the same stamps ---------------------------------------------

    def test_the_same_stamps_without_the_flag_are_refused(self, stored_world):
        stored_world.bind_active_epoch()

        code, text = _cli_begin_on_the_active_pair(stored_world, pack=(PROMPT, RAILS))

        assert code == 2
        assert "nothing to cut over" in text
        assert "--settings-change" in text
        assert _cutover_rows(stored_world) == []

    def test_the_same_stamps_with_the_flag_open_a_candidate(self, stored_world):
        stored_world.bind_active_epoch()

        code, text = _cli_begin_on_the_active_pair(
            stored_world, "--settings-change", pack=(PROMPT, RAILS)
        )

        assert code == 0, text
        candidate = stored_world.store.open_cutover()
        assert candidate.state == "filling"
        assert _hashes(candidate) == (PROMPT, RAILS)

    # -- class 2: the pair, under another prompt hash ---------------------------

    @pytest.mark.parametrize(
        ("ontology_version", "rails"),
        [
            pytest.param(ONTOLOGY_VERSION, RAILS, id="the-prompt-hash-alone"),
            pytest.param(ONTOLOGY_VERSION, NEXT_RAILS, id="both-hashes"),
            pytest.param("1.5.0", NEXT_RAILS, id="another-ontology-version-too"),
        ],
    )
    def test_the_pair_under_another_prompt_hash_is_refused(
        self, stored_world, ontology_version, rails
    ):
        stored_world.bind_active_epoch()

        code, text = _cli_begin_on_the_active_pair(
            stored_world, pack=(NEXT_PROMPT, rails), ontology_version=ontology_version
        )

        assert code == 2
        assert "[cutover] REFUSED:" in text
        assert "EXTRACTION_VERSION" in text
        assert f"pack_prompt_hash={NEXT_PROMPT!r}" in text
        assert f"pack_prompt_hash={PROMPT!r}" in text
        assert _cutover_rows(stored_world) == []
        assert stored_world.store.open_cutover() is None

    def test_the_refusal_does_not_wait_for_the_active_epoch_to_be_bound(self, stored_world):
        """Class 2 is refused bound or not: binding the epoch would not make it valid."""
        code, text = _cli_begin_on_the_active_pair(stored_world, pack=(NEXT_PROMPT, RAILS))

        assert code == 2
        assert "EXTRACTION_VERSION" in text
        assert "has no identity yet" not in text

    def test_another_prompt_hash_under_another_model_is_a_cutover_of_its_own(self, stored_world):
        """Class 4: the pair differs, so the candidate's rows are its own and it opens unbound."""
        resolver = fake_pack_resolver({ONTOLOGY_VERSION: (NEXT_PROMPT, NEXT_RAILS)})

        code, text = _cli_begin(stored_world, pack_for_version=resolver)

        assert code == 0, text
        assert _hashes(stored_world.store.open_cutover()) == (NEXT_PROMPT, NEXT_RAILS)

    # -- class 3: the pair otherwise -------------------------------------------

    @SHARES_THE_PAIRS_ROWS
    def test_the_pair_otherwise_needs_the_active_epoch_bound(
        self, stored_world, ontology_version, rails
    ):
        code, text = _cli_begin_on_the_active_pair(
            stored_world, pack=(PROMPT, rails), ontology_version=ontology_version
        )

        assert code == 2
        assert UNBOUND_SHARES_ROWS.format(n=stored_world.epoch_id) in text
        assert _cutover_rows(stored_world) == []

    @SHARES_THE_PAIRS_ROWS
    def test_the_pair_otherwise_opens_once_the_active_epoch_is_bound(
        self, stored_world, ontology_version, rails
    ):
        stored_world.bind_active_epoch()

        code, text = _cli_begin_on_the_active_pair(
            stored_world, pack=(PROMPT, rails), ontology_version=ontology_version
        )

        assert code == 0, text
        candidate = stored_world.store.open_cutover()
        assert candidate.state == "filling"
        assert candidate.ontology_version == ontology_version
        assert _hashes(candidate) == (PROMPT, rails)

    # -- --settings-change on a candidate that differs in a pack hash ----------

    def test_settings_change_under_another_rails_hash_is_refused(self, stored_world):
        stored_world.bind_active_epoch()

        code, text = _cli_begin_on_the_active_pair(
            stored_world, "--settings-change", pack=(PROMPT, NEXT_RAILS)
        )

        assert code == 2
        assert "--settings-change keeps the active epoch" in text
        assert f"pack_rails_hash={NEXT_RAILS!r}" in text
        assert f"pack_rails_hash={RAILS!r}" in text
        assert "omit the flag" in text
        assert _cutover_rows(stored_world) == []

    def test_settings_change_under_another_prompt_hash_is_refused_and_not_told_to_omit_the_flag(
        self, stored_world
    ):
        """Without the flag the candidate is class 2, so the refusal does not send it there."""
        stored_world.bind_active_epoch()

        code, text = _cli_begin_on_the_active_pair(
            stored_world, "--settings-change", pack=(NEXT_PROMPT, RAILS)
        )

        assert code == 2
        assert "--settings-change keeps the active epoch" in text
        assert f"pack_prompt_hash={NEXT_PROMPT!r}" in text
        assert f"pack_prompt_hash={PROMPT!r}" in text
        assert "EXTRACTION_VERSION" in text
        assert "omit the flag" not in text
        assert _cutover_rows(stored_world) == []

    # -- an epoch from before the hashes ---------------------------------------

    def test_an_epoch_that_stores_no_hash_is_compared_on_the_triple_alone(self, tmp_path):
        """No stored prompt hash to differ from: the triple decides, as before the hashes."""
        world = _build_world(with_deriver=False, db_path=str(tmp_path / "legacy.db"))
        world.bind_active_epoch()

        same_triple = _cli_begin_on_the_active_pair(world, pack=(NEXT_PROMPT, NEXT_RAILS))
        ontology_only = _cli_begin_on_the_active_pair(
            world, pack=(NEXT_PROMPT, NEXT_RAILS), ontology_version="1.5.0"
        )

        assert same_triple[0] == 2
        assert "nothing to cut over" in same_triple[1]
        assert ontology_only[0] == 0, ontology_only[1]
        assert _hashes(world.store.open_cutover()) == (NEXT_PROMPT, NEXT_RAILS)
