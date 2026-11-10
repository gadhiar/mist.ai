"""The ceiling answer store: append-only lines, a hash chain, and the state-file check.

Every file here is synthetic and lives under `tmp_path`. The store is the only record of
what the answerer said, so each refusal rule in the plan's Interfaces section has a test,
and so does each way of tampering with a committed store: an edited, deleted, reordered or
inserted line, a truncated store, and a rewritten last line.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402

from backend.errors import MistError  # noqa: E402
from backend.extraction_service.schemas import (  # noqa: E402
    EXTRACTION_OUTPUT_SCHEMA,
    SCOPE_OUTPUT_SCHEMA,
)
from scripts.eval_harness.ceiling import store as st  # noqa: E402

ZERO = "0" * 64


def _key(n: int) -> str:
    return hashlib.sha256(f"request-{n}".encode()).hexdigest()


def _extract_answer(n: int) -> dict:
    return {"entities": [{"id": f"e{n}", "type": "Person"}], "relationships": []}


def _offer(*keys: str, kind: str = "extract") -> tuple[st.OfferedRequest, ...]:
    schema = EXTRACTION_OUTPUT_SCHEMA if kind == "extract" else SCOPE_OUTPUT_SCHEMA
    return tuple(st.OfferedRequest(key=k, kind=kind, schema=schema) for k in keys)


def _store_with(path: Path, count: int) -> st.AnswerStore:
    store = st.AnswerStore.load(path)
    store.offer(1, _offer(*(_key(n) for n in range(1, count + 1))))
    for n in range(1, count + 1):
        store.add(_key(n), _extract_answer(n))
    return store


def _lines(path: Path) -> list[bytes]:
    data = path.read_bytes()
    assert data.endswith(b"\n")
    return data[:-1].split(b"\n")


def _write_lines(path: Path, lines: list[bytes]) -> None:
    path.write_bytes(b"".join(line + b"\n" for line in lines))


def _encode(record: dict) -> bytes:
    return json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _rechain(records: list[dict]) -> list[bytes]:
    """Re-link `records` into a valid chain, as a careful forger would."""
    out: list[bytes] = []
    prev = ZERO
    for record in records:
        line = _encode({**record, "prev": prev})
        out.append(line)
        prev = hashlib.sha256(line).hexdigest()
    return out


# ---------------------------------------------------------------------------
# Add, then reload
# ---------------------------------------------------------------------------


def test_add_then_reload_round_trips_the_lines(tmp_path: Path) -> None:
    path = tmp_path / "answers-1.jsonl"
    store = _store_with(path, 3)

    reloaded = st.AnswerStore.load(path)

    assert len(reloaded) == 3
    assert reloaded.answers() == {_key(n): _extract_answer(n) for n in range(1, 4)}
    assert [e.seq for e in reloaded.entries] == [1, 2, 3]
    assert reloaded.entries == store.entries

    lines = _lines(path)
    first = json.loads(lines[0])
    assert list(first) == ["seq", "key", "round", "kind", "answer", "chars", "prev"]
    assert first["seq"] == 1
    assert first["key"] == _key(1)
    assert first["round"] == 1
    assert first["kind"] == "extract"
    assert first["answer"] == _extract_answer(1)
    assert first["chars"] == len(json.dumps(_extract_answer(1), ensure_ascii=False))
    assert first["prev"] == ZERO
    for earlier, later in zip(lines, lines[1:], strict=False):
        assert json.loads(later)["prev"] == hashlib.sha256(earlier).hexdigest()


def test_served_text_is_the_compact_dump_the_chars_field_counts() -> None:
    answer = {"scope": "personal", "reasoning": "café"}

    assert st.served_text(answer) == json.dumps(answer, ensure_ascii=False)


def test_a_missing_store_loads_empty(tmp_path: Path) -> None:
    store = st.AnswerStore.load(tmp_path / "absent.jsonl")

    assert len(store) == 0
    assert store.answers() == {}
    assert store.state().lines == 0
    assert store.state().last_line_sha256 == ZERO


def test_add_appends_and_never_rewrites_earlier_bytes(tmp_path: Path) -> None:
    path = tmp_path / "answers-1.jsonl"
    store = _store_with(path, 2)
    before = path.read_bytes()

    store.offer(2, _offer(_key(3)))
    store.add(_key(3), _extract_answer(3))

    after = path.read_bytes()
    assert after.startswith(before)
    assert json.loads(_lines(path)[2])["round"] == 2


def test_errors_belong_to_the_mist_error_hierarchy() -> None:
    assert issubclass(st.AnswerStoreError, MistError)
    for cls in (
        st.StoreIntegrityError,
        st.AnswerRefusedError,
        st.OfferError,
        st.SharedKeyError,
        st.StateFileError,
        st.StoreChangedError,
    ):
        assert issubclass(cls, st.AnswerStoreError)


# ---------------------------------------------------------------------------
# add's refusals
# ---------------------------------------------------------------------------


def test_add_refuses_a_key_already_stored(tmp_path: Path) -> None:
    path = tmp_path / "answers-1.jsonl"
    store = _store_with(path, 1)
    size = path.stat().st_size

    with pytest.raises(st.KeyAlreadyStoredError):
        store.add(_key(1), _extract_answer(9))
    assert path.stat().st_size == size


def test_add_refuses_a_key_not_offered_in_the_latest_round(tmp_path: Path) -> None:
    path = tmp_path / "answers-1.jsonl"
    store = st.AnswerStore.load(path)
    store.offer(1, _offer(_key(1)))
    store.offer(2, _offer(_key(2)))

    with pytest.raises(st.KeyNotOfferedError):
        store.add(_key(1), _extract_answer(1))
    with pytest.raises(st.KeyNotOfferedError):
        store.add(_key(3), _extract_answer(3))
    assert not path.exists()


def test_add_refuses_everything_before_any_round_is_offered(tmp_path: Path) -> None:
    store = st.AnswerStore.load(tmp_path / "answers-1.jsonl")

    with pytest.raises(st.KeyNotOfferedError):
        store.add(_key(1), _extract_answer(1))


def test_add_refuses_a_schema_invalid_answer_with_its_path_and_message(tmp_path: Path) -> None:
    path = tmp_path / "answers-1.jsonl"
    store = st.AnswerStore.load(path)
    store.offer(1, _offer(_key(1)))

    with pytest.raises(st.AnswerSchemaError) as caught:
        store.add(_key(1), {"entities": [{"id": "e1"}], "relationships": ["not an object"]})

    assert caught.value.path == "$.relationships[0]"
    assert "object" in caught.value.message
    assert not path.exists()


def test_add_refuses_a_scope_answer_missing_its_required_field(tmp_path: Path) -> None:
    store = st.AnswerStore.load(tmp_path / "answers-1.jsonl")
    store.offer(1, _offer(_key(1), kind="scope"))

    with pytest.raises(st.AnswerSchemaError) as caught:
        store.add(_key(1), {"confidence": 0.9})

    assert caught.value.path == "$"
    assert "scope" in caught.value.message


def test_add_refuses_an_answer_whose_served_text_holds_harmony_markup(tmp_path: Path) -> None:
    store = st.AnswerStore.load(tmp_path / "answers-1.jsonl")
    store.offer(1, _offer(_key(1)))

    with pytest.raises(st.HarmonyMarkupError):
        store.add(
            _key(1),
            {"entities": [{"id": "e1", "note": "<|channel|>final"}], "relationships": []},
        )


def test_add_refuses_an_answer_that_is_not_strict_json(tmp_path: Path) -> None:
    store = st.AnswerStore.load(tmp_path / "answers-1.jsonl")
    store.offer(1, _offer(_key(1)))

    with pytest.raises(st.AnswerNotJsonError):
        store.add(_key(1), {"entities": [{"score": float("nan")}], "relationships": []})


# ---------------------------------------------------------------------------
# offer's refusals
# ---------------------------------------------------------------------------


def test_offer_refuses_a_round_that_does_not_advance(tmp_path: Path) -> None:
    store = st.AnswerStore.load(tmp_path / "answers-1.jsonl")
    store.offer(2, _offer(_key(1)))

    with pytest.raises(st.OfferError):
        store.offer(2, _offer(_key(2)))
    with pytest.raises(st.OfferError):
        store.offer(1, _offer(_key(2)))


def test_offer_refuses_a_stored_key_a_repeated_key_and_a_bad_record(tmp_path: Path) -> None:
    store = _store_with(tmp_path / "answers-1.jsonl", 1)

    with pytest.raises(st.OfferError):
        store.offer(2, _offer(_key(1)))
    with pytest.raises(st.OfferError):
        store.offer(2, _offer(_key(2), _key(2)))
    with pytest.raises(st.OfferError):
        store.offer(2, (st.OfferedRequest(key="abc", kind="extract", schema={}),))
    with pytest.raises(st.OfferError):
        store.offer(2, (st.OfferedRequest(key=_key(2), kind="chat", schema={}),))
    with pytest.raises(st.OfferError):
        store.offer(2, (st.OfferedRequest(key=_key(2), kind="extract", schema={"type": 3}),))


# ---------------------------------------------------------------------------
# load's refusals
# ---------------------------------------------------------------------------


def test_load_refuses_a_gap_in_seq(tmp_path: Path) -> None:
    path = tmp_path / "answers-1.jsonl"
    _store_with(path, 3)
    records = [json.loads(line) for line in _lines(path)]
    records[2]["seq"] = 4
    _write_lines(path, _rechain(records))

    with pytest.raises(st.SeqGapError):
        st.AnswerStore.load(path)


def test_load_refuses_a_repeated_key(tmp_path: Path) -> None:
    path = tmp_path / "answers-1.jsonl"
    _store_with(path, 3)
    records = [json.loads(line) for line in _lines(path)]
    records[2]["key"] = records[0]["key"]
    _write_lines(path, _rechain(records))

    with pytest.raises(st.RepeatedKeyError):
        st.AnswerStore.load(path)


def test_load_refuses_a_broken_prev_chain(tmp_path: Path) -> None:
    path = tmp_path / "answers-1.jsonl"
    _store_with(path, 2)
    lines = _lines(path)
    second = json.loads(lines[1])
    second["prev"] = "f" * 64
    _write_lines(path, [lines[0], _encode(second)])

    with pytest.raises(st.ChainBrokenError):
        st.AnswerStore.load(path)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda r: r.update(chars=r["chars"] + 1), id="chars"),
        pytest.param(lambda r: r.update(kind="chat"), id="kind"),
        pytest.param(lambda r: r.update(key="XYZ"), id="key"),
        pytest.param(lambda r: r.pop("round"), id="missing-field"),
        pytest.param(lambda r: r.update(extra=1), id="extra-field"),
    ],
)
def test_load_refuses_a_malformed_line(tmp_path: Path, mutate) -> None:
    path = tmp_path / "answers-1.jsonl"
    _store_with(path, 2)
    records = [json.loads(line) for line in _lines(path)]
    mutate(records[1])
    _write_lines(path, _rechain(records))

    with pytest.raises(st.MalformedLineError):
        st.AnswerStore.load(path)


def test_load_refuses_a_last_line_without_its_newline(tmp_path: Path) -> None:
    path = tmp_path / "answers-1.jsonl"
    _store_with(path, 2)
    path.write_bytes(path.read_bytes()[:-1])

    with pytest.raises(st.MalformedLineError):
        st.AnswerStore.load(path)


# ---------------------------------------------------------------------------
# The chain catches an edited, deleted, reordered or inserted line
# ---------------------------------------------------------------------------


def test_chain_catches_an_edited_line(tmp_path: Path) -> None:
    path = tmp_path / "answers-1.jsonl"
    _store_with(path, 3)
    lines = _lines(path)
    edited = json.loads(lines[1])
    edited["answer"] = _extract_answer(99)
    edited["chars"] = len(st.served_text(edited["answer"]))
    _write_lines(path, [lines[0], _encode(edited), lines[2]])

    with pytest.raises(st.ChainBrokenError):
        st.AnswerStore.load(path)


def test_chain_catches_a_deleted_line(tmp_path: Path) -> None:
    path = tmp_path / "answers-1.jsonl"
    _store_with(path, 3)
    lines = _lines(path)
    _write_lines(path, [lines[0], lines[2]])

    with pytest.raises(st.StoreIntegrityError):
        st.AnswerStore.load(path)


def test_chain_catches_a_deleted_line_even_when_seq_is_renumbered(tmp_path: Path) -> None:
    path = tmp_path / "answers-1.jsonl"
    _store_with(path, 3)
    lines = _lines(path)
    third = json.loads(lines[2])
    third["seq"] = 2
    _write_lines(path, [lines[0], _encode(third)])

    with pytest.raises(st.ChainBrokenError):
        st.AnswerStore.load(path)


def test_chain_catches_reordered_lines(tmp_path: Path) -> None:
    path = tmp_path / "answers-1.jsonl"
    _store_with(path, 3)
    lines = _lines(path)
    _write_lines(path, [lines[0], lines[2], lines[1]])

    with pytest.raises(st.StoreIntegrityError):
        st.AnswerStore.load(path)


def test_chain_catches_reordered_lines_even_when_seq_is_renumbered(tmp_path: Path) -> None:
    path = tmp_path / "answers-1.jsonl"
    _store_with(path, 3)
    lines = _lines(path)
    second, third = json.loads(lines[1]), json.loads(lines[2])
    second["seq"], third["seq"] = 3, 2
    _write_lines(path, [lines[0], _encode(third), _encode(second)])

    with pytest.raises(st.ChainBrokenError):
        st.AnswerStore.load(path)


def test_chain_catches_an_inserted_line(tmp_path: Path) -> None:
    path = tmp_path / "answers-1.jsonl"
    _store_with(path, 2)
    lines = _lines(path)
    forged = json.loads(lines[1])
    forged.update(key=_key(50), seq=2)
    later = json.loads(lines[1])
    later["seq"] = 3
    # The forged line links to line 1, but the old line 2 still names line 1 as prev.
    forged_line = _encode({**forged, "prev": hashlib.sha256(lines[0]).hexdigest()})
    _write_lines(path, [lines[0], forged_line, _encode(later)])

    with pytest.raises(st.ChainBrokenError):
        st.AnswerStore.load(path)


# ---------------------------------------------------------------------------
# The state file
# ---------------------------------------------------------------------------


def test_state_round_trips_with_the_offered_requests(tmp_path: Path) -> None:
    path = tmp_path / "answers-1.jsonl"
    state_path = tmp_path / "state" / "state-1.json"
    store = _store_with(path, 2)
    offered = _offer(_key(3), kind="scope")
    written = store.offer(2, offered)

    st.write_state(state_path, written)
    read = st.read_state(state_path)

    assert read == written
    assert read.lines == 2
    assert read.last_line_sha256 == hashlib.sha256(_lines(path)[-1]).hexdigest()
    assert read.round == 2
    assert read.offered == offered


def test_read_state_returns_none_for_a_missing_file(tmp_path: Path) -> None:
    assert st.read_state(tmp_path / "state-1.json") is None


def test_read_state_refuses_a_malformed_file(tmp_path: Path) -> None:
    state_path = tmp_path / "state-1.json"
    state_path.write_text('{"lines": "two"}', encoding="utf-8")

    with pytest.raises(st.StateFileError):
        st.read_state(state_path)


def test_load_with_state_restores_the_round_and_lets_add_continue(tmp_path: Path) -> None:
    path = tmp_path / "answers-1.jsonl"
    state_path = tmp_path / "state-1.json"
    store = _store_with(path, 1)
    st.write_state(state_path, store.offer(2, _offer(_key(2), _key(3))))

    resumed = st.AnswerStore.load(path, state_path=state_path)
    resumed.add(_key(2), _extract_answer(2))
    st.write_state(state_path, resumed.state())

    again = st.AnswerStore.load(path, state_path=state_path)
    assert again.round == 2
    assert len(again) == 2
    assert again.state().lines == 2
    again.add(_key(3), _extract_answer(3))
    assert json.loads(_lines(path)[2])["round"] == 2


def test_load_refuses_a_store_shorter_than_the_state_records(tmp_path: Path) -> None:
    path = tmp_path / "answers-1.jsonl"
    state_path = tmp_path / "state-1.json"
    store = _store_with(path, 3)
    st.write_state(state_path, store.state())
    _write_lines(path, _lines(path)[:2])

    # The chain alone cannot see a dropped last line.
    assert len(st.AnswerStore.load(path)) == 2
    with pytest.raises(st.StateMismatchError):
        st.AnswerStore.load(path, state_path=state_path)


def test_load_refuses_a_rewritten_line_at_the_recorded_count(tmp_path: Path) -> None:
    path = tmp_path / "answers-1.jsonl"
    state_path = tmp_path / "state-1.json"
    store = _store_with(path, 3)
    st.write_state(state_path, store.state())
    lines = _lines(path)
    last = json.loads(lines[2])
    last["answer"] = _extract_answer(77)
    last["chars"] = len(st.served_text(last["answer"]))
    _write_lines(path, [*lines[:2], _encode(last)])

    # The chain alone cannot see a rewritten last line.
    assert len(st.AnswerStore.load(path)) == 3
    with pytest.raises(st.StateMismatchError):
        st.AnswerStore.load(path, state_path=state_path)


def test_load_accepts_lines_appended_after_the_recorded_count(tmp_path: Path) -> None:
    path = tmp_path / "answers-1.jsonl"
    state_path = tmp_path / "state-1.json"
    store = _store_with(path, 1)
    st.write_state(state_path, store.offer(2, _offer(_key(2))))
    store.add(_key(2), _extract_answer(2))

    assert len(st.AnswerStore.load(path, state_path=state_path)) == 2


def test_load_without_a_state_file_checks_the_chain_only(tmp_path: Path) -> None:
    path = tmp_path / "answers-1.jsonl"
    _store_with(path, 2)

    store = st.AnswerStore.load(path, state_path=tmp_path / "absent.json")

    assert len(store) == 2
    assert store.round == 1


def test_add_refuses_when_the_file_changed_after_load(tmp_path: Path) -> None:
    path = tmp_path / "answers-1.jsonl"
    store = _store_with(path, 1)
    store.offer(2, _offer(_key(2)))
    _write_lines(path, [])

    with pytest.raises(st.StoreChangedError):
        store.add(_key(2), _extract_answer(2))


# ---------------------------------------------------------------------------
# merge_stores
# ---------------------------------------------------------------------------


def test_merge_stores_joins_disjoint_stores(tmp_path: Path) -> None:
    first = tmp_path / "answers-1.jsonl"
    second = tmp_path / "answers-2.jsonl"
    _store_with(first, 2)
    other = st.AnswerStore.load(second)
    other.offer(1, _offer(_key(10)))
    other.add(_key(10), _extract_answer(10))

    merged = st.merge_stores([first, second])

    assert merged == {
        _key(1): _extract_answer(1),
        _key(2): _extract_answer(2),
        _key(10): _extract_answer(10),
    }


def test_merge_stores_refuses_a_key_held_by_two_stores(tmp_path: Path) -> None:
    first = tmp_path / "answers-1.jsonl"
    second = tmp_path / "answers-2.jsonl"
    _store_with(first, 2)
    other = st.AnswerStore.load(second)
    other.offer(1, _offer(_key(2)))
    other.add(_key(2), _extract_answer(2))

    with pytest.raises(st.SharedKeyError):
        st.merge_stores([first, second])


def test_merge_stores_refuses_a_broken_store(tmp_path: Path) -> None:
    first = tmp_path / "answers-1.jsonl"
    _store_with(first, 3)
    lines = _lines(first)
    _write_lines(first, [lines[0], lines[2]])

    with pytest.raises(st.StoreIntegrityError):
        st.merge_stores([first])
