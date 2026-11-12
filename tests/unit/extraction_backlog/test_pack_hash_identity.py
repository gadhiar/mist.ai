"""The two pack hashes as part of an epoch's identity.

`pack_prompt_hash` covers what the ontology pack feeds the model and
`pack_rails_hash` what runs on a payload after it. An epoch or a cutover
candidate that holds one refuses a service, a reply or a backend writer whose
hash differs or is missing. A stored null is a target from before the hashes
existed: that hash is not compared.

The stored side here is built directly: a row dict that carries the keys, or
`dataclasses.replace` on what the store returned. The suite's world seeds its
epoch with no hash, so what its store returns reads as legacy. The same checks
over hashes a real `EventStore` holds are in `test_pack_hash_persistence.py`.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from backend.extraction_backlog.cutover import (
    CutoverRefusedError,
    begin_cutover,
    candidate_pack_refusal,
)
from backend.extraction_backlog.errors import InferenceServiceError
from backend.extraction_backlog.store import (
    PACK_HASH_FIELDS,
    BacklogStore,
    Cutover,
    Epoch,
    PackHashes,
    Stamps,
    has_same_stamps,
    pack_hash_mismatches,
)
from backend.extraction_contract.models import (
    CONTRACT_VERSION,
    ErrorCode,
    ExtractionPayload,
    ExtractResponse,
    InfoResponse,
    ResultStamps,
    ScopeOut,
    TimingsMs,
)
from tests.unit.extraction_backlog.conftest import ONTOLOGY_VERSION, composed

PROMPT = "a" * 64
RAILS = "b" * 64
OTHER = "f" * 64
STORED = {"pack_prompt_hash": PROMPT, "pack_rails_hash": RAILS}

BOTH_FIELDS = pytest.mark.parametrize("field", ["pack_prompt_hash", "pack_rails_hash"])

_LEDGER_ROW = {
    "epoch_id": 3,
    "ontology_version": ONTOLOGY_VERSION,
    "extraction_version": "ev-test",
    "model_hash": "composed-model",
}
_CUTOVER_ROW = {
    "cutover_id": 7,
    "ontology_version": ONTOLOGY_VERSION,
    "extraction_version": "ev-test",
    "model_hash": "composed-model",
    "bare_model_hash": "bare-model",
    "requested_at": "2026-11-11T00:00:00+00:00",
    "state": "filling",
    "source_epoch_id": 3,
    "rebuild_job_id": None,
    "rebuilt_through_event_id": None,
    "check_report": None,
    "promoted_epoch_id": None,
    "updated_at": "2026-11-11T00:00:00+00:00",
}


def _stamps(world, **hashes) -> ResultStamps:
    """Result stamps that match the world's epoch on everything but the pack hashes."""
    return ResultStamps(
        extraction_version=world.service.extraction_version,
        model_hash=world.service.model_hash,
        prompt_sha256="0" * 64,
        llama_cpp_build="b0",
        adapter="fake",
        serving_identity=world.service.serving_identity,
        **hashes,
    )


def _response(world, **hashes) -> ExtractResponse:
    return ExtractResponse(
        contract_version=CONTRACT_VERSION,
        job_id="job-1",
        outcome="extracted",
        scope=ScopeOut(label="user-scope", confidence=0.9),
        payload=ExtractionPayload(entities=[], relationships=[]),
        derivation=None,
        stamps=_stamps(world, **hashes),
        timings_ms=TimingsMs(scope=1.0, extract=2.0, derive=None, total=3.0),
        attempts=1,
    )


def _holding(epoch: Epoch, **hashes) -> Epoch:
    """`epoch` as it reads once its row holds the given pack hashes."""
    return dataclasses.replace(epoch, **hashes)


class TestTheNullRule:
    """`pack_hash_mismatches(stored, offered)`: the one comparison every site uses."""

    def test_the_two_hashes_are_the_fields_compared(self):
        assert PACK_HASH_FIELDS == ("pack_prompt_hash", "pack_rails_hash")

    def test_equal_hashes_match(self):
        assert pack_hash_mismatches(PackHashes(PROMPT, RAILS), PackHashes(PROMPT, RAILS)) == ()

    @BOTH_FIELDS
    def test_a_different_hash_is_named(self, field):
        offered = PackHashes(**{**STORED, field: OTHER})

        assert pack_hash_mismatches(PackHashes(**STORED), offered) == (field,)

    def test_a_stored_null_is_legacy_and_not_compared(self):
        assert pack_hash_mismatches(PackHashes(None, None), PackHashes(OTHER, OTHER)) == ()

    @BOTH_FIELDS
    def test_one_stored_null_leaves_the_other_hash_compared(self, field):
        other_field = next(name for name in PACK_HASH_FIELDS if name != field)
        stored = PackHashes(**{**STORED, field: None})

        assert pack_hash_mismatches(stored, PackHashes(OTHER, OTHER)) == (other_field,)

    @BOTH_FIELDS
    def test_an_offered_null_against_a_stored_hash_is_named(self, field):
        offered = PackHashes(**{**STORED, field: None})

        assert pack_hash_mismatches(PackHashes(**STORED), offered) == (field,)


_TRIPLE = {"ontology_version": "1.4.0", "extraction_version": "ev", "model_hash": "m|emb:e"}


class TestTheSameStamps:
    """`has_same_stamps(source, candidate)`: the one meaning of "the same stamps"."""

    def test_an_equal_triple_and_equal_hashes_are_the_same_stamps(self):
        assert has_same_stamps(Stamps(**_TRIPLE, **STORED), Stamps(**_TRIPLE, **STORED)) is True

    @pytest.mark.parametrize("field", sorted(_TRIPLE))
    def test_a_different_triple_field_is_not_the_same_stamps(self, field):
        candidate = Stamps(**{**_TRIPLE, field: "another"}, **STORED)

        assert has_same_stamps(Stamps(**_TRIPLE, **STORED), candidate) is False

    @BOTH_FIELDS
    def test_a_different_pack_hash_is_not_the_same_stamps(self, field):
        candidate = Stamps(**_TRIPLE, **{**STORED, field: OTHER})

        assert has_same_stamps(Stamps(**_TRIPLE, **STORED), candidate) is False

    @BOTH_FIELDS
    def test_a_candidate_without_a_hash_the_source_holds_is_not_the_same_stamps(self, field):
        candidate = Stamps(**_TRIPLE, **{**STORED, field: None})

        assert has_same_stamps(Stamps(**_TRIPLE, **STORED), candidate) is False

    def test_a_source_that_holds_no_hash_compares_the_triple_alone(self):
        """A legacy source: nothing says which pack it ran under (`pack_hash_mismatches`)."""
        candidate = Stamps(**_TRIPLE, pack_prompt_hash=OTHER, pack_rails_hash=OTHER)

        assert has_same_stamps(Stamps(**_TRIPLE), candidate) is True

    @BOTH_FIELDS
    def test_a_source_that_holds_one_hash_compares_that_one(self, field):
        other_field = next(name for name in PACK_HASH_FIELDS if name != field)
        source = Stamps(**_TRIPLE, **{field: STORED[field]})

        assert has_same_stamps(source, Stamps(**_TRIPLE, **{**STORED, other_field: OTHER}))
        assert not has_same_stamps(source, Stamps(**_TRIPLE, **{**STORED, field: OTHER}))

    def test_an_epoch_and_a_candidate_are_compared_as_the_store_returns_them(self):
        """The sites hand over what they read: an `Epoch`, and a `Cutover` or its target."""
        source = Epoch.from_row({**_LEDGER_ROW, **STORED})
        settings_change = Cutover.from_row({**_CUTOVER_ROW, **STORED})
        replay = Cutover.from_row({**_CUTOVER_ROW, **STORED, "pack_rails_hash": OTHER})

        assert has_same_stamps(source, settings_change) is True
        assert has_same_stamps(source, settings_change.target) is True
        assert has_same_stamps(source, replay) is False
        assert has_same_stamps(source, replay.target) is False


class TestRowReaders:
    """`Epoch.from_row` and `Cutover.from_row` take the hashes from the row when it has them."""

    def test_an_epoch_row_with_the_keys_gives_both_hashes(self):
        epoch = Epoch.from_row({**_LEDGER_ROW, **STORED})

        assert (epoch.pack_prompt_hash, epoch.pack_rails_hash) == (PROMPT, RAILS)

    def test_an_epoch_row_without_the_keys_reads_as_legacy(self):
        epoch = Epoch.from_row(dict(_LEDGER_ROW))

        assert (epoch.pack_prompt_hash, epoch.pack_rails_hash) == (None, None)

    def test_an_epoch_row_with_null_values_reads_as_legacy(self):
        epoch = Epoch.from_row({**_LEDGER_ROW, "pack_prompt_hash": None, "pack_rails_hash": None})

        assert (epoch.pack_prompt_hash, epoch.pack_rails_hash) == (None, None)

    def test_a_cutover_row_with_the_keys_gives_both_hashes(self):
        cutover = Cutover.from_row({**_CUTOVER_ROW, **STORED})

        assert (cutover.pack_prompt_hash, cutover.pack_rails_hash) == (PROMPT, RAILS)

    def test_a_cutover_row_without_the_keys_reads_as_legacy(self):
        cutover = Cutover.from_row(dict(_CUTOVER_ROW))

        assert (cutover.pack_prompt_hash, cutover.pack_rails_hash) == (None, None)

    def test_a_candidates_target_carries_its_hashes(self):
        """The fill step compares replies with `Cutover.target`, so the hashes must reach it."""
        target = Cutover.from_row({**_CUTOVER_ROW, **STORED}).target

        assert (target.pack_prompt_hash, target.pack_rails_hash) == (PROMPT, RAILS)
        assert target.cutover_id == 7

    def test_what_the_event_store_returns_today_reads_as_legacy(self, backlog_world):
        """An epoch seeded with no hash stores NULL in both columns, so it compares neither."""
        epoch = backlog_world.store.active_epoch()

        assert (epoch.pack_prompt_hash, epoch.pack_rails_hash) == (None, None)


class TestEpochMatches:
    """`ExtractionDispatcher._epoch_matches`: a service's stamps against a target."""

    def test_equal_hashes_match(self, backlog_world):
        dispatcher = backlog_world.build_dispatcher()
        epoch = _holding(backlog_world.store.active_epoch(), **STORED)

        assert dispatcher._epoch_matches(_stamps(backlog_world, **STORED), epoch) is True

    @BOTH_FIELDS
    def test_a_different_hash_is_refused(self, backlog_world, field):
        dispatcher = backlog_world.build_dispatcher()
        epoch = _holding(backlog_world.store.active_epoch(), **STORED)
        stamps = _stamps(backlog_world, **{**STORED, field: OTHER})

        assert dispatcher._epoch_matches(stamps, epoch) is False

    def test_a_legacy_epoch_compares_neither_hash(self, backlog_world):
        dispatcher = backlog_world.build_dispatcher()
        epoch = backlog_world.store.active_epoch()

        reported = _stamps(backlog_world, pack_prompt_hash=OTHER, pack_rails_hash=OTHER)

        assert dispatcher._epoch_matches(reported, epoch) is True
        assert dispatcher._epoch_matches(_stamps(backlog_world), epoch) is True

    @BOTH_FIELDS
    def test_a_reply_without_a_hash_the_epoch_holds_is_refused(self, backlog_world, field):
        dispatcher = backlog_world.build_dispatcher()
        epoch = _holding(backlog_world.store.active_epoch(), **STORED)
        stamps = _stamps(backlog_world, **{**STORED, field: None})

        assert dispatcher._epoch_matches(stamps, epoch) is False

    def test_the_version_and_the_model_are_still_compared(self, backlog_world):
        dispatcher = backlog_world.build_dispatcher()
        epoch = _holding(backlog_world.store.active_epoch(), **STORED)
        stamps = _stamps(backlog_world, **STORED)

        assert not dispatcher._epoch_matches(
            stamps.model_copy(update={"extraction_version": "ev-other"}), epoch
        )
        assert not dispatcher._epoch_matches(
            stamps.model_copy(update={"model_hash": "other-model"}), epoch
        )

    @BOTH_FIELDS
    def test_a_candidate_refuses_a_reply_under_another_pack(self, backlog_world, field):
        """T45 Acceptance 1, the cutover side: the fill compares with the candidate's hashes."""
        dispatcher = backlog_world.build_dispatcher()
        candidate = Cutover.from_row(
            {
                **_CUTOVER_ROW,
                "extraction_version": backlog_world.service.extraction_version,
                "model_hash": composed(backlog_world.service.model_hash),
                **STORED,
            }
        ).target

        assert dispatcher._epoch_matches(_stamps(backlog_world, **STORED), candidate) is True
        assert (
            dispatcher._epoch_matches(_stamps(backlog_world, **{**STORED, field: OTHER}), candidate)
            is False
        )


class TestReplyValidation:
    """`_validate_response` refuses a reply extracted under another pack as `epoch_mismatch`."""

    def test_a_reply_with_the_epochs_hashes_passes(self, backlog_world):
        dispatcher = backlog_world.build_dispatcher()
        epoch = _holding(backlog_world.store.active_epoch(), **STORED)

        dispatcher._validate_response(
            SimpleNamespace(job_id="job-1"), _response(backlog_world, **STORED), epoch
        )

    @BOTH_FIELDS
    def test_a_reply_with_another_hash_is_an_epoch_mismatch_that_names_it(
        self, backlog_world, field
    ):
        dispatcher = backlog_world.build_dispatcher()
        epoch = _holding(backlog_world.store.active_epoch(), **STORED)
        response = _response(backlog_world, **{**STORED, field: OTHER})

        with pytest.raises(InferenceServiceError) as raised:
            dispatcher._validate_response(SimpleNamespace(job_id="job-1"), response, epoch)

        assert raised.value.code == ErrorCode.EPOCH_MISMATCH
        assert raised.value.retryable is False
        assert f"{field}={OTHER!r}" in str(raised.value)
        assert f"{field}={STORED[field]!r}" in str(raised.value)

    @BOTH_FIELDS
    def test_a_reply_without_the_hash_is_an_epoch_mismatch(self, backlog_world, field):
        dispatcher = backlog_world.build_dispatcher()
        epoch = _holding(backlog_world.store.active_epoch(), **STORED)
        response = _response(backlog_world, **{**STORED, field: None})

        with pytest.raises(InferenceServiceError) as raised:
            dispatcher._validate_response(SimpleNamespace(job_id="job-1"), response, epoch)

        assert raised.value.code == ErrorCode.EPOCH_MISMATCH
        assert f"{field}=None" in str(raised.value)

    def test_a_legacy_epoch_takes_a_reply_with_any_hashes(self, backlog_world):
        dispatcher = backlog_world.build_dispatcher()
        epoch = backlog_world.store.active_epoch()

        dispatcher._validate_response(
            SimpleNamespace(job_id="job-1"),
            _response(backlog_world, pack_prompt_hash=OTHER, pack_rails_hash=OTHER),
            epoch,
        )


class TestServiceCheck:
    """`_check_service` reads `/v1/info`, which reports the service's pack hashes."""

    @pytest.mark.asyncio
    async def test_a_service_on_the_epochs_pack_is_accepted(self, backlog_world):
        backlog_world.service.pack_prompt_hash = PROMPT
        backlog_world.service.pack_rails_hash = RAILS
        dispatcher = backlog_world.build_dispatcher()
        epoch = _holding(backlog_world.store.active_epoch(), **STORED)

        info = await dispatcher._check_service(epoch)

        assert isinstance(info, InfoResponse)
        assert (info.pack_prompt_hash, info.pack_rails_hash) == (PROMPT, RAILS)

    @pytest.mark.asyncio
    @BOTH_FIELDS
    async def test_a_service_on_another_pack_stalls_in_epoch_mismatch(self, backlog_world, field):
        backlog_world.service.pack_prompt_hash = PROMPT
        backlog_world.service.pack_rails_hash = RAILS
        setattr(backlog_world.service, field, OTHER)
        dispatcher = backlog_world.build_dispatcher()
        epoch = _holding(backlog_world.store.active_epoch(), **STORED)

        result = await dispatcher._check_service(epoch)

        assert not isinstance(result, InfoResponse)
        assert dispatcher.state == "epoch_mismatch"
        assert f"{field}={OTHER!r}" in dispatcher.reason
        assert f"{field}={STORED[field]!r}" in dispatcher.reason

    @pytest.mark.asyncio
    async def test_a_service_that_reports_no_hashes_stalls_against_an_epoch_that_holds_them(
        self, backlog_world
    ):
        dispatcher = backlog_world.build_dispatcher()
        epoch = _holding(backlog_world.store.active_epoch(), **STORED)

        result = await dispatcher._check_service(epoch)

        assert not isinstance(result, InfoResponse)
        assert dispatcher.state == "epoch_mismatch"
        assert "pack_prompt_hash=None" in dispatcher.reason

    @pytest.mark.asyncio
    async def test_a_legacy_epoch_accepts_a_service_on_any_pack(self, backlog_world):
        backlog_world.service.pack_prompt_hash = OTHER
        backlog_world.service.pack_rails_hash = OTHER
        dispatcher = backlog_world.build_dispatcher()

        info = await dispatcher._check_service(backlog_world.store.active_epoch())

        assert isinstance(info, InfoResponse)


class TestWriterStampGuard:
    """The backend's writer stamps carry its own pack's hashes; the guard compares them too."""

    def _dispatcher(self, world, **hashes):
        writer = dataclasses.replace(world.active_epoch_stamps(), **hashes)
        return world.build_dispatcher(writer_stamps=writer)

    def test_a_writer_on_the_epochs_pack_passes(self, backlog_world):
        dispatcher = self._dispatcher(backlog_world, **STORED)
        epoch = _holding(backlog_world.store.active_epoch(), **STORED)

        assert dispatcher._writer_stamp_mismatch(epoch) is None

    @BOTH_FIELDS
    def test_a_writer_on_another_pack_is_refused_with_the_hash_named(self, backlog_world, field):
        dispatcher = self._dispatcher(backlog_world, **{**STORED, field: OTHER})
        epoch = _holding(backlog_world.store.active_epoch(), **STORED)

        reason = dispatcher._writer_stamp_mismatch(epoch)

        assert reason is not None
        assert f"{field}={OTHER!r}" in reason
        assert f"{field}={STORED[field]!r}" in reason

    def test_a_legacy_epoch_compares_neither_hash_with_the_writer(self, backlog_world):
        dispatcher = self._dispatcher(backlog_world, pack_prompt_hash=OTHER, pack_rails_hash=OTHER)

        assert dispatcher._writer_stamp_mismatch(backlog_world.store.active_epoch()) is None

    @BOTH_FIELDS
    def test_a_writer_without_a_hash_the_epoch_holds_is_refused(self, backlog_world, field):
        dispatcher = self._dispatcher(backlog_world, **{**STORED, field: None})
        epoch = _holding(backlog_world.store.active_epoch(), **STORED)

        reason = dispatcher._writer_stamp_mismatch(epoch)

        assert reason is not None
        assert f"{field}=None" in reason


class _StoreWhoseEpochHolds(BacklogStore):
    """A backlog store whose active epoch reads as holding the given pack hashes."""

    def __init__(self, world, **hashes) -> None:
        super().__init__(world.event_store, world.cache)
        self._hashes = hashes

    def active_epoch(self) -> Epoch | None:
        epoch = super().active_epoch()
        return None if epoch is None else dataclasses.replace(epoch, **self._hashes)


def _begin(store: BacklogStore, world, **overrides) -> Cutover:
    arguments = {
        "bare_model_hash": world.service.model_hash,
        "extraction_version": world.service.extraction_version,
        "ontology_version": ONTOLOGY_VERSION,
        "embedding_model_name": world.embedding_model,
        "now_iso": "2026-11-11T00:00:00+00:00",
    }
    arguments.update(overrides)
    return begin_cutover(store, **arguments)


class TestBeginCutover:
    """The candidate's stamps are the triple and both hashes; the active epoch's likewise."""

    def test_the_same_triple_and_the_same_hashes_is_nothing_to_cut_over(self, backlog_world):
        store = _StoreWhoseEpochHolds(backlog_world, **STORED)

        with pytest.raises(CutoverRefusedError, match="nothing to cut over"):
            _begin(store, backlog_world, **STORED)

    def test_the_same_triple_under_another_rails_hash_is_a_cutover_of_its_own(self, backlog_world):
        """A rails-only change: the candidate shares the pair's rows, so it opens."""
        backlog_world.bind_active_epoch()
        store = _StoreWhoseEpochHolds(backlog_world, **STORED)

        cutover = _begin(store, backlog_world, **{**STORED, "pack_rails_hash": OTHER})

        assert cutover.state == "filling"

    def test_the_same_pair_under_another_prompt_hash_needs_a_new_extraction_version(
        self, backlog_world
    ):
        """What feeds the model changed, and the pair's cache key would not say so."""
        backlog_world.bind_active_epoch()
        store = _StoreWhoseEpochHolds(backlog_world, **STORED)

        with pytest.raises(CutoverRefusedError) as raised:
            _begin(store, backlog_world, **{**STORED, "pack_prompt_hash": OTHER})

        assert "EXTRACTION_VERSION" in str(raised.value)
        assert f"pack_prompt_hash={OTHER!r}" in str(raised.value)
        assert f"pack_prompt_hash={PROMPT!r}" in str(raised.value)
        assert backlog_world.store.open_cutover() is None

    @BOTH_FIELDS
    def test_a_settings_change_may_not_change_a_hash(self, backlog_world, field):
        backlog_world.bind_active_epoch()
        store = _StoreWhoseEpochHolds(backlog_world, **STORED)

        with pytest.raises(CutoverRefusedError) as raised:
            _begin(store, backlog_world, settings_change=True, **{**STORED, field: OTHER})

        assert "--settings-change keeps the active epoch" in str(raised.value)
        assert f"{field}={OTHER!r}" in str(raised.value)
        assert backlog_world.store.open_cutover() is None

    def test_a_legacy_epoch_compares_only_the_triple(self, backlog_world):
        """An epoch with null hashes: a same-triple candidate is still nothing to cut over."""
        with pytest.raises(CutoverRefusedError, match="nothing to cut over"):
            _begin(backlog_world.store, backlog_world, **STORED)

    @BOTH_FIELDS
    def test_a_candidate_without_a_hash_the_epoch_holds_does_not_equal_it(
        self, backlog_world, field
    ):
        backlog_world.bind_active_epoch()
        store = _StoreWhoseEpochHolds(backlog_world, **STORED)

        with pytest.raises(CutoverRefusedError) as raised:
            _begin(store, backlog_world, settings_change=True, **{**STORED, field: None})

        assert f"{field}=None" in str(raised.value)


class TestRebuildUnderTheCandidatesPack:
    """`candidate_pack_refusal`: the staging rebuild runs the backend's pack over the rows."""

    def _candidate(self, **hashes) -> Cutover:
        return Cutover.from_row({**_CUTOVER_ROW, **hashes})

    def test_the_candidates_pack_may_rebuild_it(self):
        assert candidate_pack_refusal(self._candidate(**STORED), PackHashes(**STORED)) is None

    @BOTH_FIELDS
    def test_another_pack_is_refused_with_the_hash_named(self, field):
        pack = PackHashes(**{**STORED, field: OTHER})

        refusal = candidate_pack_refusal(self._candidate(**STORED), pack)

        assert refusal is not None
        assert f"{field}={OTHER!r}" in refusal
        assert f"{field}={STORED[field]!r}" in refusal
        assert "cutover 7" in refusal

    def test_a_legacy_candidate_is_rebuilt_under_any_pack(self):
        assert candidate_pack_refusal(self._candidate(), PackHashes(OTHER, OTHER)) is None

    @BOTH_FIELDS
    def test_a_pack_without_a_hash_the_candidate_holds_is_refused(self, field):
        pack = PackHashes(**{**STORED, field: None})

        refusal = candidate_pack_refusal(self._candidate(**STORED), pack)

        assert refusal is not None
        assert f"{field}=None" in refusal

    def test_the_replay_stages_are_built_over_the_pack_that_was_checked(self):
        """The rebuild's normalizer and validator run the pack the refusal compared."""
        from backend.extraction_backlog.cutover import replay_pack_consumers
        from backend.knowledge.ontologies import registry

        pack = registry.active_pack()

        normalizer, validator = replay_pack_consumers(pack, min_confidence=0.25)

        assert normalizer._pack is pack
        assert validator._pack is pack
        assert validator.min_confidence == 0.25

    def test_the_replay_stages_hold_a_pack_that_is_not_the_registrys(self):
        """Mutant this kills: dropping `pack=pack` from either constructor call.

        A consumer built without a pack resolves the registry's active one, so
        the pack handed in here is one the registry does not return.
        """
        from backend.extraction_backlog.cutover import replay_pack_consumers
        from backend.knowledge.ontologies import registry

        marked = _marked_pack()

        normalizer, validator = replay_pack_consumers(marked, min_confidence=0.25)

        assert marked is not registry.active_pack()
        assert normalizer._pack is marked
        assert validator._pack is marked
        assert normalizer._pack is not registry.active_pack()
        assert validator._pack is not registry.active_pack()


def _marked_pack(**hashes):
    """A real `Pack` the registry does not return: its active one under other hashes."""
    from backend.knowledge.ontologies import registry

    return dataclasses.replace(registry.active_pack(), **{**STORED, **hashes})


class _ReachedTheConnectionsError(Exception):
    """Raised in place of the first step the rebuild wiring takes after the pack check."""


class TestTheRebuildWiringChecksThePackFirst:
    """`build_rebuild_deps_from_env` reads the active pack and refuses before it connects."""

    def _wire(self, monkeypatch, *, active, **stored):
        from backend.extraction_backlog.cutover import build_rebuild_deps_from_env
        from backend.knowledge.ontologies import registry
        from backend.knowledge.version_stamps import compose_model_hash
        from tests.mocks.config import build_test_config

        config = build_test_config()
        candidate_config = dataclasses.replace(
            config,
            ontology_version=_CUTOVER_ROW["ontology_version"],
            extraction_version=_CUTOVER_ROW["extraction_version"],
            model_hash=_CUTOVER_ROW["bare_model_hash"],
        )
        cutover = Cutover.from_row(
            {**_CUTOVER_ROW, "model_hash": compose_model_hash(candidate_config), **stored}
        )

        def reached(*_args, **_kwargs):
            raise _ReachedTheConnectionsError

        monkeypatch.setattr("backend.knowledge.config.get_config", lambda: config)
        monkeypatch.setattr(registry, "active_pack", lambda: active)
        monkeypatch.setattr(
            "backend.knowledge.eval_isolation.assert_rebuild_target_not_live", reached
        )
        return build_rebuild_deps_from_env(None, "bolt://staging.invalid:7687", cutover)

    def test_the_candidates_pack_goes_on_to_the_connections(self, monkeypatch):
        with pytest.raises(_ReachedTheConnectionsError):
            self._wire(monkeypatch, active=_marked_pack(), **STORED)

    def test_a_legacy_candidate_goes_on_under_any_pack(self, monkeypatch):
        with pytest.raises(_ReachedTheConnectionsError):
            self._wire(monkeypatch, active=_marked_pack())

    @BOTH_FIELDS
    def test_another_active_pack_is_refused_before_any_connection(self, monkeypatch, field):
        active = _marked_pack(**{field: OTHER})

        with pytest.raises(CutoverRefusedError) as raised:
            self._wire(monkeypatch, active=active, **STORED)

        assert f"{field}={OTHER!r}" in str(raised.value)
        assert f"{field}={STORED[field]!r}" in str(raised.value)
        assert "cutover 7" in str(raised.value)
