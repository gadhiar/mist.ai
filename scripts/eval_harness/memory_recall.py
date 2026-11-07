"""Synthetic memory recall fixture: typed loaders and the recall scorer.

The read path (plan mist-read-path, Goal G8) is measured on a synthetic fixture under
`tests/fixtures/memory_recall/`. This module is that fixture's schema home: it parses each of the
six JSONL files into frozen dataclasses, refusing rows that do not match, and it scores a run file
against the probes.

A run file is JSONL with one record per probe, written by the run harness (T9's
`memory_recall_run.py`) or built by hand in tests:

    {"probe_id": "fact-01", "items": [{"ref": "m1", "kind": "fact", "id": "...", "score": 0.82}],
     "tokens": 212, "budget_tokens": 500}

`items` is the block in block order, with the same `{ref, kind, id, score}` shape as the turn
log's `retrieval_context.memory.items`. Extra keys (timings, `reranked`, `delta_tokens_est`) are
allowed and ignored, so the harness can record more than the scorer reads.

Metrics, each a mean over the probes it is defined on:

- recall@block: the share of a probe's expected items present in its block. Defined on probes
  with at least one expected item.
- MRR: the reciprocal of the 1-based block position of the first expected item, 0 when none is
  present. Defined on the same probes as recall.
- precision@block: the share of a block's items that are expected or acceptable. Defined on
  probes whose block is not empty, so a noise probe served anything scores 0 here as well.
- empty_on_noise: 1 when a noise probe's block is empty. Defined on noise probes.
- forgotten_leak: 1 when a block holds any of the probe's forbidden items (a forgotten fact or the
  narrative that names it). Defined on probes with forbidden items. Its floor is 0.
- budget_ok: 1 when the block's tokens are within its budget. Defined on every probe.

Each is reported overall, per probe kind, per lexical tag and per probe shape, with `None` where
no probe defines it, so a missing number is never confused with a zero. The shape separates the
probes that exercise the reader's list cap (`list_item`) and its type filter (`type_scoped`) from
the rest (`plain`); a probe built in code without a shape is `plain`, so the overall, kind and tag
numbers do not depend on it.

Usage:
    python scripts/eval_harness/memory_recall.py --run run.jsonl [--probes probes.jsonl]
        [--json-output report.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TypeVar

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "memory_recall"

PROBE_KINDS = ("fact", "episode", "note", "continuity", "temporal", "correction", "noise")
PROBE_SHAPES = ("plain", "list_item", "type_scoped")
ITEM_KINDS = ("fact", "episode", "exchange", "note")
LEXICAL_TAGS = {True: "lexical", False: "paraphrase"}
FACT_STATES = ("current", "superseded", "future", "retracted")
EPISODE_CLOSED_BY = ("turn_limit", "token_limit", "conversation_end")
NOTE_AUTHORS = ("user", "user-edit", "mist")
CORRECTION_ACTIONS = ("forget", "correct")
CORRECTION_SURFACES = ("ws", "tool", "cli")

_T = TypeVar("_T")


# ---------------------------------------------------------------------------
# Row validation helpers
# ---------------------------------------------------------------------------


def _check_keys(
    row: Mapping[str, object], keys: Iterable[str], where: str, *, strict: bool
) -> None:
    """Refuse a row with a missing key, or with an unknown key when `strict`.

    Fixture rows are strict so that a typo in a key name fails the fixture test instead of being
    read as an absent optional field. Run records are lenient because the harness adds fields
    the scorer does not read.
    """
    wanted = set(keys)
    missing = sorted(wanted - set(row))
    if missing:
        raise ValueError(f"{where}: missing key(s) {missing}")
    if strict:
        unknown = sorted(set(row) - wanted)
        if unknown:
            raise ValueError(f"{where}: unknown key(s) {unknown}")


def _str(row: Mapping[str, object], key: str, where: str) -> str:
    value = row[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where}: {key} must be a non-empty string, got {value!r}")
    return value


def _opt_str(row: Mapping[str, object], key: str, where: str) -> str | None:
    return None if row[key] is None else _str(row, key, where)


def _bool(row: Mapping[str, object], key: str, where: str) -> bool:
    value = row[key]
    if not isinstance(value, bool):
        raise ValueError(f"{where}: {key} must be a boolean, got {value!r}")
    return value


def _int(row: Mapping[str, object], key: str, where: str) -> int:
    value = row[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{where}: {key} must be a non-negative integer, got {value!r}")
    return value


def _float(row: Mapping[str, object], key: str, where: str) -> float:
    value = row[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{where}: {key} must be a number, got {value!r}")
    return float(value)


def _choice(row: Mapping[str, object], key: str, choices: Sequence[str], where: str) -> str:
    value = _str(row, key, where)
    if value not in choices:
        raise ValueError(f"{where}: {key} {value!r} is not one of {list(choices)}")
    return value


def _datetime(row: Mapping[str, object], key: str, where: str) -> datetime:
    """Parse an ISO 8601 timestamp that carries its offset.

    A naive timestamp is refused: the temporal parser resolves "yesterday" in a configured zone,
    and a timestamp without an offset would be read in whatever zone the reader assumed.
    """
    text = _str(row, key, where)
    try:
        value = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{where}: {key} is not an ISO 8601 timestamp: {text!r}") from exc
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{where}: {key} must carry a UTC offset: {text!r}")
    return value


def _opt_datetime(row: Mapping[str, object], key: str, where: str) -> datetime | None:
    return None if row[key] is None else _datetime(row, key, where)


def _str_list(row: Mapping[str, object], key: str, where: str) -> tuple[str, ...]:
    value = row[key]
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        raise ValueError(f"{where}: {key} must be a list of non-empty strings, got {value!r}")
    return tuple(value)


# ---------------------------------------------------------------------------
# Item references, probes and run records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ItemRef:
    """One memory item named by kind and id, the identity the block reports.

    The kind is part of the identity: an exchange's event id and a fact's key live in different
    namespaces, and matching on the id alone could credit the wrong item.
    """

    kind: str
    id: str


def _refs(row: Mapping[str, object], key: str, where: str) -> tuple[ItemRef, ...]:
    value = row[key]
    if not isinstance(value, list):
        raise ValueError(f"{where}: {key} must be a list of item references")
    refs: list[ItemRef] = []
    for index, entry in enumerate(value):
        entry_where = f"{where} {key}[{index}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{entry_where}: must be an object with kind and id")
        _check_keys(entry, ("kind", "id"), entry_where, strict=True)
        refs.append(
            ItemRef(
                kind=_choice(entry, "kind", ITEM_KINDS, entry_where),
                id=_str(entry, "id", entry_where),
            )
        )
    return tuple(refs)


@dataclass(frozen=True, slots=True)
class Probe:
    """One recall probe: a user turn and the block it should and should not produce.

    `expected` items must appear for recall; `acceptable` items are relevant for precision but
    not required (an exchange inside an expected episode); `forbidden` items must never appear
    (a forgotten fact and the narrative naming it); `distractors` share words or entities with
    the expected items but answer a different question, and are recorded so a reader can see
    what the probe is designed to separate.

    `lexical` is true when a word matcher can get the probe right. For a noise probe, whose
    right answer is an empty block, that means the query shares no content word with any stored
    item; false marks a tempting noise probe that does. For any other probe, true means the query
    shares a content word with every expected item, or with every forbidden item when nothing is
    expected (a correction probe after a forget); false marks a paraphrase that shares no word
    stem with them, which only a semantic match finds.

    `shape` says what the probe asks for: `list_item` expects every sibling fact of one list
    utterance, more than a 300-token block holds, so it measures the reader's cap; `type_scoped`
    names an entity type rather than an entity ("which apps do I use"), so it measures the type
    filter; `plain` is everything else. It defaults to `plain` for probes built in code, while
    a fixture row must carry it.
    """

    probe_id: str
    kind: str
    lexical: bool
    query: str
    now: datetime
    conversation_start: bool
    expected: tuple[ItemRef, ...]
    acceptable: tuple[ItemRef, ...]
    forbidden: tuple[ItemRef, ...]
    distractors: tuple[ItemRef, ...]
    correction_id: str | None
    rationale: str
    shape: str = "plain"


_PROBE_KEYS = (
    "probe_id",
    "kind",
    "lexical",
    "query",
    "now",
    "conversation_start",
    "expected",
    "acceptable",
    "forbidden",
    "distractors",
    "correction_id",
    "rationale",
    "shape",
)


def parse_probe(row: Mapping[str, object]) -> Probe:
    """Parse one probes.jsonl row, refusing missing, unknown or mistyped keys."""
    where = f"probe {row.get('probe_id', '?')}"
    _check_keys(row, _PROBE_KEYS, where, strict=True)
    return Probe(
        probe_id=_str(row, "probe_id", where),
        kind=_choice(row, "kind", PROBE_KINDS, where),
        lexical=_bool(row, "lexical", where),
        query=_str(row, "query", where),
        now=_datetime(row, "now", where),
        conversation_start=_bool(row, "conversation_start", where),
        expected=_refs(row, "expected", where),
        acceptable=_refs(row, "acceptable", where),
        forbidden=_refs(row, "forbidden", where),
        distractors=_refs(row, "distractors", where),
        correction_id=_opt_str(row, "correction_id", where),
        rationale=_str(row, "rationale", where),
        shape=_choice(row, "shape", PROBE_SHAPES, where),
    )


@dataclass(frozen=True, slots=True)
class BlockItem:
    """One item of a run's block, in the turn log's `{ref, kind, id, score}` shape."""

    ref: str
    kind: str
    id: str
    score: float

    @property
    def item(self) -> ItemRef:
        """The item's identity, so a block item compares equal to a probe's reference."""
        return ItemRef(kind=self.kind, id=self.id)


@dataclass(frozen=True, slots=True)
class RunRecord:
    """What the read path produced for one probe: its block and the block's token cost."""

    probe_id: str
    items: tuple[BlockItem, ...]
    tokens: int
    budget_tokens: int


def parse_run_record(row: Mapping[str, object]) -> RunRecord:
    """Parse one run-file row; extra keys are ignored, missing or mistyped ones refused."""
    where = f"run record {row.get('probe_id', '?')}"
    _check_keys(row, ("probe_id", "items", "tokens", "budget_tokens"), where, strict=False)
    raw_items = row["items"]
    if not isinstance(raw_items, list):
        raise ValueError(f"{where}: items must be a list")
    items: list[BlockItem] = []
    for index, entry in enumerate(raw_items):
        entry_where = f"{where} items[{index}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{entry_where}: must be an object")
        _check_keys(entry, ("ref", "kind", "id", "score"), entry_where, strict=False)
        items.append(
            BlockItem(
                ref=_str(entry, "ref", entry_where),
                kind=_choice(entry, "kind", ITEM_KINDS, entry_where),
                id=_str(entry, "id", entry_where),
                score=_float(entry, "score", entry_where),
            )
        )
    return RunRecord(
        probe_id=_str(row, "probe_id", where),
        items=tuple(items),
        tokens=_int(row, "tokens", where),
        budget_tokens=_int(row, "budget_tokens", where),
    )


# ---------------------------------------------------------------------------
# Fixture rows
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Fact:
    """One fact version, a row of the memory index's `facts` table.

    Several rows can share a `fact_key`: a fact that held in two separate periods keeps one row
    per interval, which is how read-time currency picks the version that holds at `now`.

    `subject_type` and `object_type` are ontology node types (`EXTRACTABLE_NODE_TYPES`), the
    labels the graph gives each end, so a type-scoped probe can be answered by type alone.
    """

    row_key: str
    fact_key: str
    subject_id: str
    subject_name: str
    subject_type: str
    relation: str
    object_id: str
    object_name: str
    object_type: str
    text: str
    valid_from: datetime | None
    valid_to: datetime | None
    recorded_at: datetime
    confidence: float
    source_utterance_id: str | None
    state: str
    superseded_by: str | None
    source_correction_id: str | None


def parse_fact(row: Mapping[str, object]) -> Fact:
    where = f"fact {row.get('row_key', '?')}"
    _check_keys(row, Fact.__slots__, where, strict=True)
    return Fact(
        row_key=_str(row, "row_key", where),
        fact_key=_str(row, "fact_key", where),
        subject_id=_str(row, "subject_id", where),
        subject_name=_str(row, "subject_name", where),
        subject_type=_str(row, "subject_type", where),
        relation=_str(row, "relation", where),
        object_id=_str(row, "object_id", where),
        object_name=_str(row, "object_name", where),
        object_type=_str(row, "object_type", where),
        text=_str(row, "text", where),
        valid_from=_opt_datetime(row, "valid_from", where),
        valid_to=_opt_datetime(row, "valid_to", where),
        recorded_at=_datetime(row, "recorded_at", where),
        confidence=_float(row, "confidence", where),
        source_utterance_id=_opt_str(row, "source_utterance_id", where),
        state=_choice(row, "state", FACT_STATES, where),
        superseded_by=_opt_str(row, "superseded_by", where),
        source_correction_id=_opt_str(row, "source_correction_id", where),
    )


@dataclass(frozen=True, slots=True)
class Turn:
    """One recorded turn, as `conversation_turn_events` holds it plus its conversation id."""

    event_id: str
    session_id: str
    conversation_id: str
    turn_index: int
    timestamp: datetime
    user_utterance: str
    system_response: str


def parse_turn(row: Mapping[str, object]) -> Turn:
    where = f"turn {row.get('event_id', '?')}"
    _check_keys(row, Turn.__slots__, where, strict=True)
    return Turn(
        event_id=_str(row, "event_id", where),
        session_id=_str(row, "session_id", where),
        conversation_id=_str(row, "conversation_id", where),
        turn_index=_int(row, "turn_index", where),
        timestamp=_datetime(row, "timestamp", where),
        user_utterance=_str(row, "user_utterance", where),
        system_response=_str(row, "system_response", where),
    )


@dataclass(frozen=True, slots=True)
class Episode:
    """One episode: its boundaries as event ids, why it closed, and its authored summary."""

    episode_id: str
    conversation_id: str
    event_ids: tuple[str, ...]
    first_event_id: str
    last_event_id: str
    start: datetime
    end: datetime
    turn_count: int
    token_count: int
    closed_by: str
    title: str
    summary: str
    open_threads: tuple[str, ...]
    entities: tuple[str, ...]


def parse_episode(row: Mapping[str, object]) -> Episode:
    where = f"episode {row.get('episode_id', '?')}"
    _check_keys(row, Episode.__slots__, where, strict=True)
    open_threads = row["open_threads"]
    if not isinstance(open_threads, list) or not all(isinstance(t, str) for t in open_threads):
        raise ValueError(f"{where}: open_threads must be a list of strings")
    return Episode(
        episode_id=_str(row, "episode_id", where),
        conversation_id=_str(row, "conversation_id", where),
        event_ids=_str_list(row, "event_ids", where),
        first_event_id=_str(row, "first_event_id", where),
        last_event_id=_str(row, "last_event_id", where),
        start=_datetime(row, "start", where),
        end=_datetime(row, "end", where),
        turn_count=_int(row, "turn_count", where),
        token_count=_int(row, "token_count", where),
        closed_by=_choice(row, "closed_by", EPISODE_CLOSED_BY, where),
        title=_str(row, "title", where),
        summary=_str(row, "summary", where),
        open_threads=tuple(open_threads),
        entities=_str_list(row, "entities", where),
    )


@dataclass(frozen=True, slots=True)
class VaultNote:
    """One vault note section; its item id is `path`, one section per note in this fixture."""

    path: str
    title: str
    heading: str
    authored_by: str
    created_at: datetime
    updated_at: datetime
    tags: tuple[str, ...]
    body: str


def parse_vault_note(row: Mapping[str, object]) -> VaultNote:
    where = f"note {row.get('path', '?')}"
    _check_keys(row, VaultNote.__slots__, where, strict=True)
    return VaultNote(
        path=_str(row, "path", where),
        title=_str(row, "title", where),
        heading=_str(row, "heading", where),
        authored_by=_choice(row, "authored_by", NOTE_AUTHORS, where),
        created_at=_datetime(row, "created_at", where),
        updated_at=_datetime(row, "updated_at", where),
        tags=_str_list(row, "tags", where),
        body=_str(row, "body", where),
    )


@dataclass(frozen=True, slots=True)
class Correction:
    """One correction log entry with the `memory_corrections` columns of Design point 12."""

    correction_id: str
    recorded_at: datetime
    action: str
    fact_key: str
    subject_id: str
    subject_name: str
    relation: str
    object_id: str
    object_name: str
    replacement_object_name: str | None
    surface: str
    session_id: str | None
    conversation_id: str | None
    turn_index: int | None
    source_event_id: str | None
    origin: str


def parse_correction(row: Mapping[str, object]) -> Correction:
    where = f"correction {row.get('correction_id', '?')}"
    _check_keys(row, Correction.__slots__, where, strict=True)
    return Correction(
        correction_id=_str(row, "correction_id", where),
        recorded_at=_datetime(row, "recorded_at", where),
        action=_choice(row, "action", CORRECTION_ACTIONS, where),
        fact_key=_str(row, "fact_key", where),
        subject_id=_str(row, "subject_id", where),
        subject_name=_str(row, "subject_name", where),
        relation=_str(row, "relation", where),
        object_id=_str(row, "object_id", where),
        object_name=_str(row, "object_name", where),
        replacement_object_name=_opt_str(row, "replacement_object_name", where),
        surface=_choice(row, "surface", CORRECTION_SURFACES, where),
        session_id=_opt_str(row, "session_id", where),
        conversation_id=_opt_str(row, "conversation_id", where),
        turn_index=None if row["turn_index"] is None else _int(row, "turn_index", where),
        source_event_id=_opt_str(row, "source_event_id", where),
        origin=_choice(row, "origin", ("user",), where),
    )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path, parse: Callable[[Mapping[str, object]], _T]) -> tuple[_T, ...]:
    """Parse every non-blank line of a JSONL file, naming the line of any bad row."""
    rows: list[_T] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{number}: not valid JSON ({exc.msg})") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path.name}:{number}: a row must be a JSON object")
        try:
            rows.append(parse(row))
        except ValueError as exc:
            raise ValueError(f"{path.name}:{number}: {exc}") from exc
    return tuple(rows)


def load_probes(path: Path = FIXTURE_DIR / "probes.jsonl") -> tuple[Probe, ...]:
    return _load_jsonl(path, parse_probe)


def load_run(path: Path) -> tuple[RunRecord, ...]:
    return _load_jsonl(path, parse_run_record)


def load_facts(path: Path = FIXTURE_DIR / "facts.jsonl") -> tuple[Fact, ...]:
    return _load_jsonl(path, parse_fact)


def load_turns(path: Path = FIXTURE_DIR / "conversations.jsonl") -> tuple[Turn, ...]:
    return _load_jsonl(path, parse_turn)


def load_episodes(path: Path = FIXTURE_DIR / "episodes.jsonl") -> tuple[Episode, ...]:
    return _load_jsonl(path, parse_episode)


def load_vault_notes(path: Path = FIXTURE_DIR / "vault_notes.jsonl") -> tuple[VaultNote, ...]:
    return _load_jsonl(path, parse_vault_note)


def load_corrections(path: Path = FIXTURE_DIR / "corrections.jsonl") -> tuple[Correction, ...]:
    return _load_jsonl(path, parse_correction)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MetricSet:
    """The six metrics over one group of probes, each with the count it averages over.

    The counts travel with the means so a reader can tell a 1.0 over forty probes from a 1.0
    over one, and a `None` (no probe defines the metric) from a measured zero.
    """

    probes: int
    recall_at_block: float | None
    mrr: float | None
    recall_probes: int
    precision_at_block: float | None
    precision_probes: int
    empty_on_noise: float | None
    noise_probes: int
    forgotten_leak: float | None
    forgotten_probes: int
    budget_ok: float | None

    def to_json(self) -> dict[str, float | int | None]:
        """Every field by name, with `None` kept so JSON readers see undefined metrics."""
        return {name: getattr(self, name) for name in self.__slots__}


@dataclass(frozen=True, slots=True)
class RecallReport:
    """The metrics overall, per probe kind, per lexical tag and per probe shape."""

    overall: MetricSet
    by_kind: Mapping[str, MetricSet]
    by_lexical: Mapping[str, MetricSet]
    by_shape: Mapping[str, MetricSet]

    def to_json(self) -> dict[str, object]:
        """The report as plain JSON, the shape `--json-output` writes."""
        return {
            "overall": self.overall.to_json(),
            "by_kind": {kind: metrics.to_json() for kind, metrics in self.by_kind.items()},
            "by_lexical": {tag: metrics.to_json() for tag, metrics in self.by_lexical.items()},
            "by_shape": {shape: metrics.to_json() for shape, metrics in self.by_shape.items()},
        }


@dataclass(frozen=True, slots=True)
class _ProbeScore:
    """One probe's contribution to each metric, `None` where the metric is undefined for it."""

    recall: float | None
    reciprocal_rank: float | None
    precision: float | None
    empty_on_noise: float | None
    leaked: float | None
    budget_ok: float


def _score_probe(probe: Probe, record: RunRecord) -> _ProbeScore:
    block = [item.item for item in record.items]
    served = set(block)
    expected = set(probe.expected)
    relevant = expected | set(probe.acceptable)

    recall: float | None = None
    reciprocal_rank: float | None = None
    if expected:
        recall = len(expected & served) / len(expected)
        reciprocal_rank = 0.0
        for position, item in enumerate(block, start=1):
            if item in expected:
                reciprocal_rank = 1.0 / position
                break

    precision = None
    if block:
        precision = sum(1 for item in block if item in relevant) / len(block)

    empty_on_noise = None
    if probe.kind == "noise":
        empty_on_noise = 1.0 if not block else 0.0

    leaked = None
    if probe.forbidden:
        leaked = 1.0 if served & set(probe.forbidden) else 0.0

    return _ProbeScore(
        recall=recall,
        reciprocal_rank=reciprocal_rank,
        precision=precision,
        empty_on_noise=empty_on_noise,
        leaked=leaked,
        budget_ok=1.0 if record.tokens <= record.budget_tokens else 0.0,
    )


def _mean(values: Sequence[float | None]) -> tuple[float | None, int]:
    defined = [value for value in values if value is not None]
    if not defined:
        return None, 0
    return sum(defined) / len(defined), len(defined)


def _aggregate(scores: Sequence[_ProbeScore]) -> MetricSet:
    recall, recall_probes = _mean([s.recall for s in scores])
    mrr, _ = _mean([s.reciprocal_rank for s in scores])
    precision, precision_probes = _mean([s.precision for s in scores])
    empty_on_noise, noise_probes = _mean([s.empty_on_noise for s in scores])
    forgotten_leak, forgotten_probes = _mean([s.leaked for s in scores])
    budget_ok, _ = _mean([s.budget_ok for s in scores])
    return MetricSet(
        probes=len(scores),
        recall_at_block=recall,
        mrr=mrr,
        recall_probes=recall_probes,
        precision_at_block=precision,
        precision_probes=precision_probes,
        empty_on_noise=empty_on_noise,
        noise_probes=noise_probes,
        forgotten_leak=forgotten_leak,
        forgotten_probes=forgotten_probes,
        budget_ok=budget_ok,
    )


def _index_run(probes: Sequence[Probe], run: Sequence[RunRecord]) -> dict[str, RunRecord]:
    """Pair each probe with exactly one run record, refusing any mismatch.

    A probe without a record is refused rather than scored as an empty block: a harness that
    skipped a probe would otherwise lift empty_on_noise and hide the skip.
    """
    records: dict[str, RunRecord] = {}
    for record in run:
        if record.probe_id in records:
            raise ValueError(f"run file has two records for probe {record.probe_id}")
        seen: set[ItemRef] = set()
        for item in record.items:
            if item.item in seen:
                raise ValueError(
                    f"run record {record.probe_id} serves {item.kind} {item.id} twice in one block"
                )
            seen.add(item.item)
        records[record.probe_id] = record
    probe_ids = {probe.probe_id for probe in probes}
    missing = sorted(probe_ids - set(records))
    if missing:
        raise ValueError(f"run file has no record for probe(s) {missing}")
    unknown = sorted(set(records) - probe_ids)
    if unknown:
        raise ValueError(f"run file has records for unknown probe(s) {unknown}")
    return records


def score_run(probes: Sequence[Probe], run: Sequence[RunRecord]) -> RecallReport:
    """Score a run against the probes it answers, overall, per kind, per tag and per shape."""
    records = _index_run(probes, run)
    scored = [(probe, _score_probe(probe, records[probe.probe_id])) for probe in probes]
    return RecallReport(
        overall=_aggregate([score for _, score in scored]),
        by_kind={
            kind: _aggregate([score for probe, score in scored if probe.kind == kind])
            for kind in PROBE_KINDS
        },
        by_lexical={
            tag: _aggregate([score for probe, score in scored if probe.lexical is flag])
            for flag, tag in LEXICAL_TAGS.items()
        },
        by_shape={
            shape: _aggregate([score for probe, score in scored if probe.shape == shape])
            for shape in PROBE_SHAPES
        },
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_COLUMNS = (
    ("probes", "probes"),
    ("recall@block", "recall_at_block"),
    ("MRR", "mrr"),
    ("precision@block", "precision_at_block"),
    ("empty_on_noise", "empty_on_noise"),
    ("forgotten_leak", "forgotten_leak"),
    ("budget_ok", "budget_ok"),
)


def _cell(value: float | int | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.3f}"


def render_markdown(report: RecallReport) -> str:
    """Render the report as one markdown table, a row per group, `-` where undefined."""
    header = "| group | " + " | ".join(label for label, _ in _COLUMNS) + " |"
    rule = "|" + "---|" * (len(_COLUMNS) + 1)
    rows = [("overall", report.overall)]
    rows += [(f"kind: {kind}", metrics) for kind, metrics in report.by_kind.items()]
    rows += [(f"tag: {tag}", metrics) for tag, metrics in report.by_lexical.items()]
    rows += [(f"shape: {shape}", metrics) for shape, metrics in report.by_shape.items()]
    lines = ["# Memory recall", "", header, rule]
    for name, metrics in rows:
        cells = " | ".join(_cell(getattr(metrics, field)) for _, field in _COLUMNS)
        lines.append(f"| {name} | {cells} |")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score a memory recall run file.")
    parser.add_argument("--run", type=Path, required=True, help="run file (JSONL)")
    parser.add_argument(
        "--probes", type=Path, default=FIXTURE_DIR / "probes.jsonl", help="probes file (JSONL)"
    )
    parser.add_argument("--json-output", type=Path, help="also write the report as JSON here")
    args = parser.parse_args(argv)

    report = score_run(load_probes(args.probes), load_run(args.run))
    sys.stdout.write(render_markdown(report))
    if args.json_output is not None:
        args.json_output.write_text(json.dumps(report.to_json(), indent=2) + "\n", "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
