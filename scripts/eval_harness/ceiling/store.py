"""Append-only, hash-chained store of the answerer's replies, and its state-file check.

Run A replays from committed files, so the store is the whole record of what the answerer
said. Each line names the SHA-256 of the line before it, which makes an edited, deleted,
reordered or inserted line visible on load. The chain cannot see a dropped or rewritten
last line, so `.local/ceiling/state-K.json` records the line count and the last line's
hash after every round and every `--add`, and `AnswerStore.load` checks the store
against it.

The line format, in key order:
`{"seq", "key", "round", "kind", "answer", "chars", "prev"}`, where `seq` counts from 1,
`chars` is the length of the text the relay serves, and `prev` is the SHA-256 hex of the
previous line's bytes (without its newline), or 64 zeros on the first line.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema.exceptions import SchemaError, best_match
from jsonschema.validators import validator_for

from backend.errors import MistError

ZERO_HASH = "0" * 64
ANSWER_KINDS = frozenset({"scope", "extract", "repair", "derive"})
HARMONY_MARKER = "<|channel|>"

_LINE_KEYS = ("seq", "key", "round", "kind", "answer", "chars", "prev")
_HEX64 = re.compile(r"[0-9a-f]{64}")


class AnswerStoreError(MistError):
    """Base of every refusal the store makes, so a caller can stop on any of them."""


class StoreIntegrityError(AnswerStoreError):
    """The store on disk is not a store this module wrote and nothing then changed."""


class MalformedLineError(StoreIntegrityError):
    """A line is not JSON, lacks or adds a field, or holds a value of the wrong shape."""


class SeqGapError(StoreIntegrityError):
    """A line's `seq` is not its 1-based position in the file."""


class RepeatedKeyError(StoreIntegrityError):
    """Two lines answer the same request key."""


class ChainBrokenError(StoreIntegrityError):
    """A line's `prev` is not the SHA-256 of the line before it."""


class StateMismatchError(StoreIntegrityError):
    """The store is shorter than the state file records, or its recorded line changed."""


class StateFileError(AnswerStoreError):
    """The state file exists but does not hold a valid state."""


class StoreChangedError(AnswerStoreError):
    """The file changed on disk between `load` and an append."""


class OfferError(AnswerStoreError):
    """A round's offer is not one the store can accept."""


class SharedKeyError(AnswerStoreError):
    """`merge_stores` found one key in two stores."""


class AnswerRefusedError(AnswerStoreError):
    """`add` refused an answer; nothing was written."""


class KeyAlreadyStoredError(AnswerRefusedError):
    """The key already has an answer, and nothing rewrites a line."""


class KeyNotOfferedError(AnswerRefusedError):
    """The key was not offered in the store's latest round."""


class AnswerNotJsonError(AnswerRefusedError):
    """The answer has no strict JSON text, for example because it holds NaN."""


class HarmonyMarkupError(AnswerRefusedError):
    """The served text holds `<|channel|>`, which `GptOssAdapter.final_text` would rewrite."""


class AnswerSchemaError(AnswerRefusedError):
    """The answer fails the request's response schema.

    Attributes:
        path: The failing location as `$`, `$.field` or `$.field[0]`.
        message: jsonschema's message for that location.
    """

    def __init__(self, key: str, path: str, message: str) -> None:
        super().__init__(f"{key[:12]} {path}: {message}")
        self.path = path
        self.message = message


@dataclass(frozen=True, slots=True)
class StoredAnswer:
    """One line of the store, as written and as reloaded."""

    seq: int
    key: str
    round: int
    kind: str
    answer: Any
    chars: int
    prev: str


@dataclass(frozen=True, slots=True)
class OfferedRequest:
    """A pending request put to the answerer, with the schema its answer must meet.

    The schema is the request's own `response_format.json_schema.schema`, so the store
    applies the same constraint the grammar applies to gpt-oss.
    """

    key: str
    kind: str
    schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class StoreState:
    """What `.local/ceiling/state-K.json` holds between round-tool runs."""

    lines: int
    last_line_sha256: str
    round: int
    offered: tuple[OfferedRequest, ...]


def served_text(answer: Any) -> str:
    """Return the text the relay serves for `answer`.

    The store counts `chars` and checks for Harmony markup on this exact text, so the
    relay must serve it and nothing else.

    Raises:
        ValueError: `answer` holds NaN or an infinity, which strict JSON cannot carry.
        TypeError: `answer` holds a value JSON cannot encode.
    """
    return json.dumps(answer, ensure_ascii=False, allow_nan=False)


def line_sha256(line: bytes) -> str:
    """Return the hash the next line's `prev` must carry, so callers chain the same bytes."""
    return hashlib.sha256(line).hexdigest()


def _json_path(parts: Iterable[Any]) -> str:
    out = "$"
    for part in parts:
        out += f"[{part}]" if isinstance(part, int) else f".{part}"
    return out


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_hex64(value: Any) -> bool:
    return isinstance(value, str) and _HEX64.fullmatch(value) is not None


def _check_offered(request: OfferedRequest) -> None:
    if not _is_hex64(request.key):
        raise OfferError(f"offered key is not 64 lowercase hex: {request.key!r}")
    if request.kind not in ANSWER_KINDS:
        raise OfferError(f"offered kind {request.kind!r} is not one of {sorted(ANSWER_KINDS)}")
    if not isinstance(request.schema, dict):
        raise OfferError(f"offered schema for {request.key[:12]} is not an object")
    try:
        validator_for(request.schema).check_schema(request.schema)
    except SchemaError as exc:
        raise OfferError(f"offered schema for {request.key[:12]} is invalid: {exc}") from exc


def _parse_line(index: int, raw: bytes) -> StoredAnswer:
    where = f"line {index + 1}"
    try:
        record = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MalformedLineError(f"{where} is not JSON: {exc}") from exc
    if not isinstance(record, dict) or tuple(record) != _LINE_KEYS:
        found = list(record) if isinstance(record, dict) else type(record).__name__
        raise MalformedLineError(f"{where} has fields {found}, expected {list(_LINE_KEYS)}")
    seq, key, round_, kind, answer, chars, prev = (record[k] for k in _LINE_KEYS)
    if not (_is_int(seq) and _is_int(round_) and round_ >= 1 and _is_int(chars)):
        raise MalformedLineError(f"{where}: seq, round and chars must be integers, round >= 1")
    if not (_is_hex64(key) and _is_hex64(prev)):
        raise MalformedLineError(f"{where}: key and prev must be 64 lowercase hex")
    if kind not in ANSWER_KINDS:
        raise MalformedLineError(f"{where}: kind {kind!r} is not one of {sorted(ANSWER_KINDS)}")
    try:
        served = served_text(answer)
    except ValueError as exc:
        raise MalformedLineError(f"{where}: answer is not strict JSON: {exc}") from exc
    if chars != len(served):
        raise MalformedLineError(f"{where}: chars {chars} != served length {len(served)}")
    return StoredAnswer(
        seq=seq, key=key, round=round_, kind=kind, answer=answer, chars=chars, prev=prev
    )


def _encode_line(entry: StoredAnswer) -> bytes:
    record = {k: getattr(entry, k) for k in _LINE_KEYS}
    return json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _split_lines(path: Path, data: bytes) -> list[bytes]:
    if not data:
        return []
    if not data.endswith(b"\n"):
        raise MalformedLineError(f"{path}: the last line has no newline (a partial write?)")
    return data[:-1].split(b"\n")


def write_state(path: Path, state: StoreState) -> None:
    """Record `state` so the next `load` can see a truncated or rewritten store.

    The round tool calls this after each round and each `--add`. The file is replaced
    whole through a sibling temporary file, so a crash leaves the old state or the new
    one, never half of either.
    """
    payload = {
        "lines": state.lines,
        "last_line_sha256": state.last_line_sha256,
        "round": state.round,
        "offered": [{"key": o.key, "kind": o.kind, "schema": o.schema} for o in state.offered],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_state(path: Path) -> StoreState | None:
    """Read the recorded state, or None when there is none to hold the store to.

    A missing file is not an error: the integration worktree has no `.local/` state after
    the answerer branches merge, and the chain check alone applies there.

    Raises:
        StateFileError: The file exists but is not a valid state.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    try:
        raw = json.loads(text)
        lines, last, round_, offered = (
            raw["lines"],
            raw["last_line_sha256"],
            raw["round"],
            raw["offered"],
        )
        if not (_is_int(lines) and lines >= 0 and _is_int(round_) and round_ >= 0):
            raise StateFileError(f"{path}: lines and round must be non-negative integers")
        if not _is_hex64(last) or not isinstance(offered, list):
            raise StateFileError(f"{path}: last_line_sha256 or offered has the wrong shape")
        requests = tuple(
            OfferedRequest(key=o["key"], kind=o["kind"], schema=o["schema"]) for o in offered
        )
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise StateFileError(f"{path}: not a valid state file: {exc}") from exc
    for request in requests:
        try:
            _check_offered(request)
        except OfferError as exc:
            raise StateFileError(f"{path}: {exc}") from exc
    return StoreState(lines=lines, last_line_sha256=last, round=round_, offered=requests)


class AnswerStore:
    """The answerer's replies for one store file, loaded with every integrity check run.

    Construct it through `load`, never directly, so no instance exists for a store that
    failed a check.
    """

    def __init__(
        self,
        path: Path,
        entries: tuple[StoredAnswer, ...],
        size: int,
        last_hash: str,
        state: StoreState | None,
    ) -> None:
        self._path = path
        self._entries = list(entries)
        self._keys = {e.key for e in entries}
        self._size = size
        self._last_hash = last_hash
        self._round = state.round if state else max((e.round for e in entries), default=0)
        self._offered = {o.key: o for o in state.offered} if state else {}

    @classmethod
    def load(cls, path: Path, *, state_path: Path | None = None) -> AnswerStore:
        """Load a store, refusing one that is not exactly what appends produced.

        A missing store file loads empty, because round 1 starts before any answer exists.
        With `state_path` naming an existing file, the store must hold at least the recorded
        line count, and its line at that count must hash as recorded; the round and offered
        requests then come from that state, so `add` can continue the round. Without one,
        only the chain is checked and nothing is offered.

        Raises:
            StoreIntegrityError: A malformed line, a `seq` gap, a repeated key, a broken
                chain, or a mismatch with the state file.
            StateFileError: The state file exists but is not valid.
        """
        path = Path(path)
        state = read_state(Path(state_path)) if state_path is not None else None
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            data = b""
        raw_lines = _split_lines(path, data)
        entries: list[StoredAnswer] = []
        seen: set[str] = set()
        expected_prev = ZERO_HASH
        for index, raw in enumerate(raw_lines):
            entry = _parse_line(index, raw)
            if entry.seq != index + 1:
                raise SeqGapError(f"line {index + 1} has seq {entry.seq}")
            if entry.key in seen:
                raise RepeatedKeyError(f"line {index + 1} repeats key {entry.key[:12]}")
            if entry.prev != expected_prev:
                raise ChainBrokenError(f"line {index + 1}'s prev does not hash line {index}")
            seen.add(entry.key)
            entries.append(entry)
            expected_prev = line_sha256(raw)
        if state is not None:
            if len(raw_lines) < state.lines:
                raise StateMismatchError(
                    f"{path} has {len(raw_lines)} lines, the state records {state.lines}"
                )
            if state.lines and line_sha256(raw_lines[state.lines - 1]) != state.last_line_sha256:
                raise StateMismatchError(f"{path} line {state.lines} changed since the state")
        return cls(path, tuple(entries), len(data), expected_prev, state)

    @property
    def path(self) -> Path:
        """The store file appends go to."""
        return self._path

    @property
    def round(self) -> int:
        """The latest round offered, which `add` stamps on each line."""
        return self._round

    @property
    def entries(self) -> tuple[StoredAnswer, ...]:
        """Every line in file order, for callers that need seq, round or kind."""
        return tuple(self._entries)

    @property
    def offered(self) -> tuple[OfferedRequest, ...]:
        """The latest round's offered requests, in offer order."""
        return tuple(self._offered.values())

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: object) -> bool:
        return key in self._keys

    def answers(self) -> dict[str, Any]:
        """Return key to answer, the mapping `build_relay` serves from."""
        return {e.key: e.answer for e in self._entries}

    def state(self) -> StoreState:
        """Snapshot the store for `write_state`, after an `--add` or a round."""
        return StoreState(
            lines=len(self._entries),
            last_line_sha256=self._last_hash,
            round=self._round,
            offered=self.offered,
        )

    def offer(self, round_: int, offered: Iterable[OfferedRequest]) -> StoreState:
        """Open round `round_` with the requests put to the answerer, replacing the last offer.

        Only the latest round's requests may be answered, so an answer cannot land on a
        request the current service run no longer makes. Rounds only advance, which keeps
        "latest" unambiguous. The caller writes the returned state with `write_state`.

        Raises:
            OfferError: The round does not advance, a key is stored or repeated, or a
                request has a bad key, kind or schema.
        """
        if not _is_int(round_) or round_ <= self._round:
            raise OfferError(f"round {round_} does not advance past round {self._round}")
        requests = tuple(offered)
        by_key: dict[str, OfferedRequest] = {}
        for request in requests:
            _check_offered(request)
            if request.key in self._keys:
                raise OfferError(f"offered key {request.key[:12]} already has an answer")
            if request.key in by_key:
                raise OfferError(f"offered key {request.key[:12]} appears twice")
            by_key[request.key] = request
        self._round = round_
        self._offered = by_key
        return self.state()

    def add(self, key: str, answer: Any) -> StoredAnswer:
        """Append the answerer's reply to an offered request, after every check passes.

        The answer must meet the request's schema, which is what the grammar guarantees
        gpt-oss, and its served text must not hold `<|channel|>`, which `final_text` would
        rewrite. Either failure would put an answer in the model's seat that gpt-oss could
        not have given. A refused answer writes nothing.

        Raises:
            KeyAlreadyStoredError: The key already has an answer.
            KeyNotOfferedError: The key is not in the latest round's offer.
            AnswerNotJsonError: The answer has no strict JSON text.
            AnswerSchemaError: The answer fails the request's schema.
            HarmonyMarkupError: The served text holds `<|channel|>`.
            StoreChangedError: The file changed on disk since `load`.
        """
        if key in self._keys:
            raise KeyAlreadyStoredError(f"{key[:12]} already has an answer")
        request = self._offered.get(key)
        if request is None:
            raise KeyNotOfferedError(f"{key[:12]} was not offered in round {self._round}")
        try:
            text = served_text(answer)
        except (TypeError, ValueError) as exc:
            raise AnswerNotJsonError(f"{key[:12]}: {exc}") from exc
        error = best_match(validator_for(request.schema)(request.schema).iter_errors(answer))
        if error is not None:
            raise AnswerSchemaError(key, _json_path(error.absolute_path), error.message)
        if HARMONY_MARKER in text:
            raise HarmonyMarkupError(f"{key[:12]}: served text holds {HARMONY_MARKER}")
        entry = StoredAnswer(
            seq=len(self._entries) + 1,
            key=key,
            round=self._round,
            kind=request.kind,
            answer=answer,
            chars=len(text),
            prev=self._last_hash,
        )
        line = _encode_line(entry)
        self._append(line + b"\n")
        self._entries.append(entry)
        self._keys.add(key)
        self._last_hash = line_sha256(line)
        return entry

    def _append(self, data: bytes) -> None:
        try:
            size = self._path.stat().st_size
        except FileNotFoundError:
            size = 0
        if size != self._size:
            raise StoreChangedError(f"{self._path} is {size} bytes, {self._size} at load")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("ab") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        self._size += len(data)


def merge_stores(paths: Iterable[Path]) -> dict[str, Any]:
    """Join the answerers' stores into the one mapping the replay serves.

    Each store loads with the chain check. A key in two stores would make the replay's
    answer depend on merge order, so it is refused rather than resolved.

    Raises:
        StoreIntegrityError: A store fails its own load.
        SharedKeyError: A key appears in two stores.
    """
    merged: dict[str, Any] = {}
    owner: dict[str, Path] = {}
    for path in paths:
        for key, answer in AnswerStore.load(Path(path)).answers().items():
            if key in merged:
                raise SharedKeyError(f"key {key[:12]} is in {owner[key]} and {path}")
            merged[key] = answer
            owner[key] = Path(path)
    return merged
