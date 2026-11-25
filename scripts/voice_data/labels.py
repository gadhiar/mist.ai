"""Clip labels: who is audible and how she sounds, and their projection onto the speaking states.

A label records, per kept clip of one extract run, who is audible (`voices`), her energy and mood
on 1-5 scales, up to `MAX_TAGS` tags from a fixed vocabulary, an optional keep range for a clip
with other voices nearby, and whether Raj wrote a note. The note itself is private text: it lives
in the run's `notes.private.jsonl`, which no stage reads, and never in a label, plan or report.

Labels live per extract run, under `seed/labels/<run>/`, because segment ids restart at 1 in
every run. A later run's clip that Raj already heard under an earlier label carries that label
forward (`carry`), so he rates only new clips.

The runtime keeps its five speaking states (`states.py`). `delivery.json` gives each state a
region over energy, mood and tags, and `project` maps a label onto the first state whose region
holds it, or onto None. The regions are data, so moving clips between states needs no relabel.

This is a host module: it imports only the stdlib and this package.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from . import gates as gates_mod
from .errors import InputError
from .records import _Record, read_jsonl
from .root import Root
from .states import SPEAKING_STATES

VOICES = ("her", "her_overlap", "her_nearby", "not_her")
MODES = ("whispered", "laughing", "sighing", "shouted")
ATTITUDES = (
    "warm",
    "playful",
    "teasing",
    "sarcastic",
    "tender",
    "confident",
    "curious",
    "surprised",
    "worried",
    "sad",
    "annoyed",
    "tired",
)
TAGS = MODES + ATTITUDES
LOW_TAGS = ("tired", "sad", "annoyed", "worried")
MAX_TAGS = 3
NOTE_MAX = 200
REMOVED = ("not-her", "flag-group", "overlap", "nearby")

LABELS_NAME = "labels.jsonl"
NOTES_NAME = "notes.private.jsonl"
REPORT_NAME = "report.json"
LEGACY_LABELS_REL = "seed/labels.csv"
PREFILL_NAME = "prefill.jsonl"
PREFILL_PLAN_NAME = "prefill-plan.json"
CARRY_SHARE = 0.9
CARRY_TOL_S = 0.01
OTHER_MIN_S = 0.1

DELIVERY_PATH = Path(__file__).with_name("delivery.json")
DELIVERY_SCHEMA = 1
REPORT_SCHEMA = 1
SCALE = (1, 5)
STATE_SECONDS_MIN_GATE = "seed.state_seconds_min"

_LABELS_DIR = "seed/labels"
_PREFILL_DIR = "seed/prefill"
# A run id names a folder, so it is one plain path segment (as `commands/extract.py` checks).
_RUN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
# Float slack for comparisons on spans the extract stage wrote as floats.
_EPS = 1e-9
# A carried keep range is re-expressed rounded to 0.01 s, so its source-time ends can differ from
# the earlier label's by up to half that step and still be the same range.
_RANGE_TOL_S = CARRY_TOL_S / 2 + _EPS
_TAG_INDEX = {tag: index for index, tag in enumerate(TAGS)}
_PREFILL_VOICES = VOICES[:3]
_CLAUSE_KEYS = frozenset({"energy", "mood", "tags_any", "tags_none"})


def _check_run(run: object) -> str:
    if not isinstance(run, str) or not _RUN.fullmatch(run) or ".." in run:
        raise InputError("refused: a run id is one folder name")
    return run


def labels_rel(run: str) -> str:
    """Return a run's label folder, relative to the root.

    Raises:
        InputError: `run` is not one plain folder name.
    """
    return f"{_LABELS_DIR}/{_check_run(run)}"


def prefill_rel(run: str) -> str:
    """Return a run's pre-fill folder, relative to the root.

    Raises:
        InputError: `run` is not one plain folder name.
    """
    return f"{_PREFILL_DIR}/{_check_run(run)}"


# ---------------------------------------------------------------------------------------------
# Value checks shared by the records
# ---------------------------------------------------------------------------------------------


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_scale(value: object) -> bool:
    return _is_int(value) and SCALE[0] <= value <= SCALE[1]  # type: ignore[operator]


def _tags_problem(tags: object) -> str | None:
    """Return why `tags` is not a valid tag tuple in vocabulary order, or None."""
    if not isinstance(tags, tuple) or not all(isinstance(t, str) for t in tags):
        return "tags must be a list of tag names"
    unknown = [t for t in tags if t not in _TAG_INDEX]
    if unknown:
        return f"tags outside the vocabulary {sorted(unknown)}"
    if len(set(tags)) != len(tags):
        return "a tag is repeated"
    if len(tags) > MAX_TAGS:
        return f"more than {MAX_TAGS} tags"
    if list(tags) != sorted(tags, key=_TAG_INDEX.__getitem__):
        return "tags are not in vocabulary order"
    return None


def _range_problem(voices: str, keep_from: object, keep_to: object, duration: float) -> str | None:
    """Return why a keep range is not valid for `voices` on a clip of `duration`, or None."""
    if keep_from is None and keep_to is None:
        return None
    if voices != "her_nearby":
        return "a keep range is only for her_nearby"
    if keep_from is None or keep_to is None:
        return "a keep range needs both ends"
    if not (_is_number(keep_from) and _is_number(keep_to)):
        return "a keep range is two numbers of seconds"
    if not 0 <= keep_from < keep_to <= duration + _EPS:  # type: ignore[operator]
        return "the keep range does not lie inside the clip"
    return None


def _span_problem(start: object, end: object, duration: object) -> str | None:
    if not (_is_number(start) and _is_number(end) and _is_number(duration)):
        return "start_s, end_s and duration_s must be numbers"
    if not 0 <= start < end or duration <= 0:  # type: ignore[operator]
        return "the span is empty or reversed"
    return None


# ---------------------------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PrefillRecord(_Record):
    """One line of `seed/prefill/<run>/prefill.jsonl`: the model's values and the pre-fill.

    `arousal`, `dominance` and `valence` are the emotion model's outputs clipped to 0..1;
    `energy` and `mood` are `scale_from` of arousal and valence. `voices` and the keep range are
    `prefill_voices`, never `not_her`: no evidence on a kept clip says she is absent. `other_s`
    and `overlap_ms` are that evidence.
    """

    segment_id: str
    source_id: str
    start_s: float
    end_s: float
    duration_s: float
    arousal: float
    dominance: float
    valence: float
    energy: int
    mood: int
    voices: str
    keep_from_s: float | None
    keep_to_s: float | None
    other_s: float
    overlap_ms: int

    def __post_init__(self) -> None:
        problem = _span_problem(self.start_s, self.end_s, self.duration_s)
        if problem is None:
            problem = self._problem()
        if problem is not None:
            raise InputError(f"PrefillRecord {self.segment_id}: {problem}")

    def _problem(self) -> str | None:
        if not (isinstance(self.segment_id, str) and isinstance(self.source_id, str)):
            return "segment_id and source_id must be text"
        for name in ("arousal", "dominance", "valence"):
            value = getattr(self, name)
            if not _is_number(value) or not 0 <= value <= 1:
                return f"{name} must lie in 0..1"
        if not (_is_scale(self.energy) and _is_scale(self.mood)):
            return "energy and mood must be integers 1-5"
        if self.voices not in _PREFILL_VOICES:
            return f"voices must be one of {list(_PREFILL_VOICES)}"
        problem = _range_problem(self.voices, self.keep_from_s, self.keep_to_s, self.duration_s)
        if problem is not None:
            return problem
        if not _is_number(self.other_s) or self.other_s < 0:
            return "other_s must be seconds, 0 or more"
        if not _is_int(self.overlap_ms) or self.overlap_ms < 0:
            return "overlap_ms must be an integer, 0 or more"
        return None


@dataclass(frozen=True, slots=True)
class LabelRecord(_Record):
    """One line of `seed/labels/<run>/labels.jsonl`: one kept clip's label.

    `source_id`, the span, `channel` and `separator` are the segment's, so a later run can tell
    whether it holds the same audio. `state` is the projection (null when removed or outside
    every region) and `removed` the reason the clip leaves the seed, if it does. `carried_from`
    is empty for a clip Raj rated; for a carried clip it is every `<run>/<segment_id>` the label
    came from, sorted. `has_note` says a note exists; the note itself is never here.
    """

    run: str
    segment_id: str
    source_id: str
    start_s: float
    end_s: float
    channel: str
    separator: str
    duration_s: float
    voices: str
    energy: int
    mood: int
    tags: tuple[str, ...]
    keep_from_s: float | None
    keep_to_s: float | None
    has_note: bool
    state: str | None
    removed: str | None
    carried_from: tuple[str, ...]

    def __post_init__(self) -> None:
        problem = _span_problem(self.start_s, self.end_s, self.duration_s)
        if problem is None:
            problem = self._problem()
        if problem is not None:
            raise InputError(f"LabelRecord {self.segment_id}: {problem}")

    def _problem(self) -> str | None:
        if not (isinstance(self.run, str) and _RUN.fullmatch(self.run)):
            return "run must be a run id"
        texts = (self.segment_id, self.source_id, self.channel, self.separator)
        if not all(isinstance(value, str) and value for value in texts):
            return "segment_id, source_id, channel and separator must be text"
        if self.voices not in VOICES:
            return f"voices must be one of {list(VOICES)}"
        if not (_is_scale(self.energy) and _is_scale(self.mood)):
            return "energy and mood must be integers 1-5"
        problem = _tags_problem(self.tags) or _range_problem(
            self.voices, self.keep_from_s, self.keep_to_s, self.duration_s
        )
        if problem is not None:
            return problem
        if not isinstance(self.has_note, bool):
            return "has_note must be true or false"
        if self.state is not None and self.state not in SPEAKING_STATES:
            return "state must be a speaking state or null"
        if self.removed is not None and self.removed not in REMOVED:
            return f"removed must be null or one of {list(REMOVED)}"
        if self.removed is not None and self.state is not None:
            return "a removed clip has no state"
        return _carried_problem(self.carried_from)


def _carried_problem(carried_from: object) -> str | None:
    if not isinstance(carried_from, tuple):
        return "carried_from must be a list"
    for item in carried_from:
        parts = item.split("/") if isinstance(item, str) else []
        if len(parts) != 2 or not _RUN.fullmatch(parts[0]) or not parts[1]:
            return "carried_from holds <run>/<segment_id> entries only"
    if list(carried_from) != sorted(set(carried_from)):
        return "carried_from must be sorted, without repeats"
    return None


# ---------------------------------------------------------------------------------------------
# delivery.json and the projection
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Clause:
    """One clause of a region: inclusive energy and mood bounds and tag conditions.

    A bound of None means any value; `tags_any` None means no tag is required.
    """

    energy: tuple[int, int] | None
    mood: tuple[int, int] | None
    tags_any: tuple[str, ...] | None
    tags_none: tuple[str, ...]

    def holds(self, energy: int, mood: int, tags: Iterable[str]) -> bool:
        """Return whether a clip with these values lies in this clause."""
        present = set(tags)
        if self.energy is not None and not self.energy[0] <= energy <= self.energy[1]:
            return False
        if self.mood is not None and not self.mood[0] <= mood <= self.mood[1]:
            return False
        if self.tags_any is not None and not present.intersection(self.tags_any):
            return False
        return not present.intersection(self.tags_none)


@dataclass(frozen=True, slots=True)
class Region:
    """A speaking state's region: its tagless example and the clauses any of which admits a clip."""

    state: str
    example: tuple[int, int]
    clauses: tuple[Clause, ...]

    def holds(self, energy: int, mood: int, tags: Iterable[str]) -> bool:
        """Return whether a clip with these values lies in this region."""
        tags = tuple(tags)
        return any(clause.holds(energy, mood, tags) for clause in self.clauses)


@dataclass(frozen=True, slots=True)
class Delivery:
    """`delivery.json`: the pre-fill formulas (as text) and the regions, in file order."""

    schema: int
    prefill: dict[str, str]
    regions: tuple[Region, ...]


def _object(
    value: object, keys: set[str], what: str, *, optional: frozenset[str] = frozenset()
) -> dict:
    if not isinstance(value, dict):
        raise InputError(f"delivery.json: {what} must be an object")
    unknown = sorted(set(value) - keys - optional)
    missing = sorted(keys - set(value))
    if unknown or missing:
        raise InputError(f"delivery.json: {what} has unknown keys {unknown} or lacks {missing}")
    return value


def _bound(value: object, what: str) -> tuple[int, int]:
    if not (
        isinstance(value, list)
        and len(value) == 2
        and all(_is_scale(v) for v in value)
        and value[0] <= value[1]
    ):
        raise InputError(f"delivery.json: {what} must be [low, high] with 1 <= low <= high <= 5")
    return (value[0], value[1])


def _tag_list(value: object, what: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise InputError(f"delivery.json: {what} must be a non-empty list of tags")
    if not all(isinstance(t, str) and t in _TAG_INDEX for t in value):
        raise InputError(f"delivery.json: {what} names a tag outside the vocabulary")
    if len(set(value)) != len(value):
        raise InputError(f"delivery.json: {what} repeats a tag")
    return tuple(value)


def _clause(value: object, what: str) -> Clause:
    data = _object(value, set(), what, optional=_CLAUSE_KEYS)
    return Clause(
        energy=_bound(data["energy"], f"{what} energy") if "energy" in data else None,
        mood=_bound(data["mood"], f"{what} mood") if "mood" in data else None,
        tags_any=_tag_list(data["tags_any"], f"{what} tags_any") if "tags_any" in data else None,
        tags_none=_tag_list(data["tags_none"], f"{what} tags_none") if "tags_none" in data else (),
    )


def _region(value: object, index: int) -> Region:
    data = _object(value, {"state", "example", "any"}, f"states[{index}]")
    state = data["state"]
    if state not in SPEAKING_STATES:
        raise InputError(f"delivery.json: states[{index}] is not a speaking state")
    example = _object(data["example"], {"energy", "mood"}, f"{state}'s example")
    if not (_is_scale(example["energy"]) and _is_scale(example["mood"])):
        raise InputError(f"delivery.json: {state}'s example needs energy and mood 1-5")
    clauses = data["any"]
    if not isinstance(clauses, list) or not clauses:
        raise InputError(f"delivery.json: {state}'s any must be a non-empty list of clauses")
    return Region(
        state=state,
        example=(example["energy"], example["mood"]),
        clauses=tuple(_clause(c, f"{state}'s clause {k}") for k, c in enumerate(clauses)),
    )


def load_delivery(path: Path = DELIVERY_PATH) -> Delivery:
    """Load and check `delivery.json`.

    Raises:
        InputError: The file is missing or not JSON; a key is unknown or missing; the schema is
            not 1; its states are not exactly the speaking states; a clause names a tag outside
            `TAGS` or a bound outside 1-5; or an example does not project to its own state.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputError("delivery.json: missing or not a JSON file") from exc
    data = _object(data, {"schema", "prefill", "states"}, "the file")
    if data["schema"] != DELIVERY_SCHEMA or isinstance(data["schema"], bool):
        raise InputError(f"delivery.json: schema must be {DELIVERY_SCHEMA}")
    prefill = _object(data["prefill"], {"energy", "mood", "note"}, "prefill")
    if not all(isinstance(text, str) and text for text in prefill.values()):
        raise InputError("delivery.json: prefill's energy, mood and note must be text")
    if not isinstance(data["states"], list):
        raise InputError("delivery.json: states must be a list")
    regions = tuple(_region(entry, k) for k, entry in enumerate(data["states"]))
    names = [region.state for region in regions]
    if sorted(names) != sorted(SPEAKING_STATES):
        raise InputError("delivery.json: states must name each speaking state exactly once")
    delivery = Delivery(schema=DELIVERY_SCHEMA, prefill=dict(prefill), regions=regions)
    for region in regions:
        if project(delivery, *region.example, ()) != region.state:
            raise InputError(f"delivery.json: {region.state}'s example does not project to it")
    return delivery


def project(delivery: Delivery, energy: int, mood: int, tags: Iterable[str]) -> str | None:
    """Return the first state, in file order, whose region holds the clip, or None.

    Raises:
        InputError: Energy or mood is not an integer 1-5, or the tags are not distinct tags of
            the vocabulary, at most `MAX_TAGS`.
    """
    if not (_is_scale(energy) and _is_scale(mood)):
        raise InputError("project: energy and mood must be integers 1-5")
    tags = tuple(tags)
    problem = _tags_problem(tuple(sorted(tags, key=lambda t: _TAG_INDEX.get(t, -1))))
    if problem is not None:
        raise InputError(f"project: {problem}")
    for region in delivery.regions:
        if region.holds(energy, mood, tags):
            return region.state
    return None


def scale_from(value: float) -> int:
    """Map a model output on 0..1 to the 1-5 scale: `1 + floor(4 x v + 0.5)`, v clipped to 0..1.

    Halves round up. Python's `round` rounds halves to even, which would put 0.125 on 1 and
    0.625 on 3.

    Raises:
        InputError: `value` is NaN.
    """
    if math.isnan(value):
        raise InputError("scale_from: the value is not a number")
    clipped = min(1.0, max(0.0, float(value)))
    return 1 + math.floor(4 * clipped + 0.5)


# ---------------------------------------------------------------------------------------------
# The voices pre-fill
# ---------------------------------------------------------------------------------------------


class _Span(Protocol):
    source_id: str
    start_s: float
    end_s: float


class _Segment(_Span, Protocol):
    """The `SegmentRecord` fields the pre-fill and the carry rule read."""

    segment_id: str
    channel: str
    separator: str
    diar_cluster: str | None
    overlap_ms: int


class _Turn(_Span, Protocol):
    role: str


def _union(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def prefill_voices(
    segment: _Segment, turns: Iterable[_Turn], min_s: float
) -> tuple[str, float | None, float | None, float]:
    """Pre-fill a kept segment's `voices` and keep range from the run's diarizer turns.

    Only an off-centre segment (no `diar_cluster`) can carry evidence of another voice: a
    turn-cut segment lies inside one turn of an exclusive diarization, and the overlap gate
    dropped every turn-cut segment with marked overlap, so it is always `her`.

    Args:
        segment: The kept segment (`SegmentRecord`).
        turns: The run's turns (`TurnRecord`s); those of other sources and roles are ignored.
        min_s: `segment.min_s`: a keep range shorter than this is left empty.

    Returns:
        `(voices, keep_from_s, keep_to_s, other_s)`. `other_s` is the seconds of the segment's
        span that turns of role `other` cover, to 0.001 s. The keep range is set only for
        `her_nearby`: the longest stretch of the span no such turn covers, clip-relative and
        rounded to 0.01 s, or both None when that stretch is under `min_s`.
    """
    start, end = segment.start_s, segment.end_s
    covered = _union(
        (max(start, t.start_s), min(end, t.end_s))
        for t in turns
        if t.source_id == segment.source_id
        and t.role == "other"
        and t.start_s < end
        and t.end_s > start
    )
    other_s = round(sum(b - a for a, b in covered), 3)
    if segment.diar_cluster is not None:
        return "her", None, None, other_s
    if segment.overlap_ms > 0:
        return "her_overlap", None, None, other_s
    if other_s <= OTHER_MIN_S:
        return "her", None, None, other_s
    edges = [start, *(edge for interval in covered for edge in interval), end]
    gaps = [(edges[k], edges[k + 1]) for k in range(0, len(edges), 2)]
    best = max(gaps, key=lambda gap: gap[1] - gap[0])
    keep_from, keep_to = round(best[0] - start, 2), round(best[1] - start, 2)
    if keep_to - keep_from < min_s - _EPS:
        return "her_nearby", None, None, other_s
    return "her_nearby", keep_from, keep_to, other_s


# ---------------------------------------------------------------------------------------------
# Reading and writing a run's labels
# ---------------------------------------------------------------------------------------------


def _refuse_legacy(root: Root) -> None:
    if (root.path / LEGACY_LABELS_REL).exists():
        raise InputError(
            f"refused: {LEGACY_LABELS_REL} is the old labels pack's file, which nothing reads "
            "now; move it out of the root and make a new pack with pack labels --run <run>"
        )


def read_labels(root: Root, run: str) -> tuple[LabelRecord, ...] | None:
    """Return run's labels, in file order, or None when the run has no `labels.jsonl`.

    Raises:
        InputError: `seed/labels.csv` exists; a record is malformed, names another run, or
            repeats a segment.
    """
    _refuse_legacy(root)
    path = root.path / labels_rel(run) / LABELS_NAME
    if not path.is_file():
        return None
    records = tuple(read_jsonl(path, LabelRecord))
    for record in records:
        if record.run != run:
            raise InputError(f"{labels_rel(run)}/{LABELS_NAME}: a record names run {record.run}")
    ids = [record.segment_id for record in records]
    if len(set(ids)) != len(ids):
        raise InputError(f"{labels_rel(run)}/{LABELS_NAME}: a segment is labelled twice")
    return records


def _notes_text(records: Sequence[LabelRecord], notes: Mapping[str, str]) -> str:
    """Check the notes against the records' `has_note` and return the notes file's text."""
    noted = {record.segment_id for record in records if record.has_note}
    if set(notes) != noted:
        raise InputError("refused: the notes do not match the labels' has_note")
    lines = []
    for record in records:
        note = notes.get(record.segment_id)
        if note is None:
            continue
        if not isinstance(note, str) or not note.strip() or len(note) > NOTE_MAX:
            raise InputError(f"refused: {record.segment_id}'s note is empty or too long")
        if "\n" in note or "\r" in note:
            raise InputError(f"refused: {record.segment_id}'s note is more than one line")
        lines.append(json.dumps({"segment_id": record.segment_id, "note": note}) + "\n")
    return "".join(lines)


def write_labels(
    root: Root,
    run: str,
    records: Sequence[LabelRecord],
    notes: Mapping[str, str],
    tables: Mapping[str, object],
    replace: bool = False,
    *,
    pack: str | None = None,
) -> None:
    """Write run's `labels.jsonl`, `notes.private.jsonl` and `report.json`, each atomically.

    Each file goes through `Root.write_bytes`. `labels.jsonl` is written last, so a reader that
    finds it finds the other two as well.

    Args:
        root: The opened root.
        run: The extract run the labels belong to.
        records: One label per kept clip of the run, each naming `run`.
        notes: Segment id to note, for exactly the records with `has_note`.
        tables: `tables(...)` of the records, stored in `report.json`.
        replace: Overwrite an existing label folder.
        pack: The pack id the labels were imported from, or None (all carried).

    Raises:
        InputError: The folder exists and `replace` is false; a record names another run or
            repeats a segment; the notes do not match `has_note`, or a note is empty, longer
            than `NOTE_MAX` or more than one line.
    """
    rel = labels_rel(run)
    if (root.path / rel).exists() and not replace:
        raise InputError(f"refused: {rel} exists; pass --replace to overwrite its labels")
    if any(record.run != run for record in records):
        raise InputError(f"refused: a label names another run than {run}")
    ids = [record.segment_id for record in records]
    if len(set(ids)) != len(ids):
        raise InputError("refused: a segment is labelled twice")
    notes_text = _notes_text(records, notes)
    report = {"schema": REPORT_SCHEMA, "run": run, "pack": pack, "tables": dict(tables)}
    labels_text = "".join(json.dumps(record.to_json()) + "\n" for record in records)
    root.write_bytes(f"{rel}/{NOTES_NAME}", notes_text.encode("utf-8"))
    root.write_bytes(f"{rel}/{REPORT_NAME}", (json.dumps(report, indent=2) + "\n").encode("utf-8"))
    root.write_bytes(f"{rel}/{LABELS_NAME}", labels_text.encode("utf-8"))


# ---------------------------------------------------------------------------------------------
# Carrying labels forward
# ---------------------------------------------------------------------------------------------


def earlier_labels(root: Root, run: str) -> tuple[LabelRecord, ...]:
    """Return the rated labels (`carried_from` empty) of every run other than `run`.

    Runs are read in sorted folder order. A carried label is skipped, so a label carried from
    `run` into another run never comes back to `run`, and provenance cannot loop.

    Raises:
        InputError: `seed/labels.csv` exists, or another run's labels are malformed.
    """
    _check_run(run)
    _refuse_legacy(root)
    folder = root.path / _LABELS_DIR
    if not folder.is_dir():
        return ()
    found: list[LabelRecord] = []
    for entry in sorted(folder.iterdir(), key=lambda p: p.name):
        if entry.name == run or not entry.is_dir() or not _RUN.fullmatch(entry.name):
            continue
        records = read_labels(root, entry.name)
        if records is not None:
            found.extend(record for record in records if not record.carried_from)
    return tuple(found)


def _source_range(record: LabelRecord) -> tuple[float, float] | None:
    if record.keep_from_s is None or record.keep_to_s is None:
        return None
    return (record.start_s + record.keep_from_s, record.start_s + record.keep_to_s)


def _same_range(a: tuple[float, float] | None, b: tuple[float, float] | None) -> bool:
    if a is None or b is None:
        return a is b
    return abs(a[0] - b[0]) <= _RANGE_TOL_S and abs(a[1] - b[1]) <= _RANGE_TOL_S


def _agree(a: LabelRecord, b: LabelRecord) -> bool:
    """Return whether two labels agree on voices, energy, mood, tags and the source-time range."""
    return (
        a.voices == b.voices
        and a.energy == b.energy
        and a.mood == b.mood
        and a.tags == b.tags
        and _same_range(_source_range(a), _source_range(b))
    )


def _matches(segment: _Segment, earlier: LabelRecord) -> bool:
    """Return whether Raj heard all of `segment`'s audio under `earlier` and decided it there."""
    if (
        earlier.removed == "flag-group"
        or earlier.source_id != segment.source_id
        or earlier.channel != segment.channel
        or earlier.separator != segment.separator
    ):
        return False
    inside = (
        segment.start_s >= earlier.start_s - CARRY_TOL_S - _EPS
        and segment.end_s <= earlier.end_s + CARRY_TOL_S + _EPS
    )
    long_enough = (segment.end_s - segment.start_s) >= CARRY_SHARE * (
        earlier.end_s - earlier.start_s
    ) - _EPS
    return inside and long_enough


def carry(
    segments: Iterable[_Segment], earlier: Iterable[LabelRecord], *, run: str
) -> dict[str, LabelRecord]:
    """Carry earlier runs' labels onto the segments of `run` that Raj already heard.

    A label matches a segment when it is from the same source, channel and separator, the
    segment's span lies inside the label's to `CARRY_TOL_S`, the segment is at least
    `CARRY_SHARE` of the label's length, and the label was not removed as `flag-group` (such a
    clip was usually never judged on its own). A segment carries only when every match agrees on
    `voices`, `energy`, `mood`, `tags` and the keep range in source time, and, for `her_nearby`,
    that range lies inside the segment; a `her_nearby` match without a range, or with one
    outside, leaves the segment to be asked again. Carried labels and labels of `run` itself
    are never sources.

    Args:
        segments: The kept segments of `run` (`SegmentRecord`s).
        earlier: The candidate labels, as `earlier_labels` returns them.
        run: The run the segments belong to, which the carried labels name.

    Returns:
        Segment id to its carried label: the segment's own span, `channel`, `separator` and
        duration, the agreed values with the range re-expressed from the segment's start,
        `has_note` false, `carried_from` every match sorted, and `state` and `removed` null,
        for the caller to recompute in `run`.
    """
    _check_run(run)
    sources = [e for e in earlier if not e.carried_from and e.run != run]
    carried: dict[str, LabelRecord] = {}
    for segment in segments:
        matches = [e for e in sources if _matches(segment, e)]
        if not matches or not all(_agree(matches[0], other) for other in matches[1:]):
            continue
        first = matches[0]
        start, end = segment.start_s, segment.end_s
        duration = round(end - start, 2)
        keep_from = keep_to = None
        if first.voices == "her_nearby":
            source_range = _source_range(first)
            if source_range is None or not (
                source_range[0] >= start - CARRY_TOL_S - _EPS
                and source_range[1] <= end + CARRY_TOL_S + _EPS
            ):
                continue
            keep_from = max(0.0, round(source_range[0] - start, 2))
            keep_to = min(duration, round(source_range[1] - start, 2))
        carried[segment.segment_id] = LabelRecord(
            run=run,
            segment_id=segment.segment_id,
            source_id=segment.source_id,
            start_s=start,
            end_s=end,
            channel=segment.channel,
            separator=segment.separator,
            duration_s=duration,
            voices=first.voices,
            energy=first.energy,
            mood=first.mood,
            tags=first.tags,
            keep_from_s=keep_from,
            keep_to_s=keep_to,
            has_note=False,
            state=None,
            removed=None,
            carried_from=tuple(sorted(f"{m.run}/{m.segment_id}" for m in matches)),
        )
    return carried


def carried_stale(root: Root, labels: Iterable[LabelRecord]) -> tuple[str, ...]:
    """Return the segment ids of carried labels that no longer hold, in input order.

    A carried label is stale when any source it lists is missing from its run's current
    `labels.jsonl`, no longer agrees with it on `voices`, `energy`, `mood`, `tags` and the range
    in source time, or would no longer be carried: its `removed` is now `flag-group`, or its
    `carried_from` is not empty.

    Raises:
        InputError: `seed/labels.csv` exists, or a source run's labels are malformed.
    """
    _refuse_legacy(root)
    current: dict[str, dict[str, LabelRecord]] = {}

    def source(ref: str) -> LabelRecord | None:
        run, segment_id = ref.split("/", 1)
        if run not in current:
            records = read_labels(root, run) or ()
            current[run] = {record.segment_id: record for record in records}
        return current[run].get(segment_id)

    stale: list[str] = []
    for record in labels:
        for ref in record.carried_from:
            earlier = source(ref)
            if (
                earlier is None
                or earlier.removed == "flag-group"
                or earlier.carried_from
                or not _agree(earlier, record)
            ):
                stale.append(record.segment_id)
                break
    return tuple(stale)


# ---------------------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------------------


def _kept_seconds(record: LabelRecord) -> float:
    if record.keep_from_s is not None and record.keep_to_s is not None:
        return record.keep_to_s - record.keep_from_s
    return record.duration_s


def _cell(count: int, seconds: float) -> dict[str, object]:
    return {"count": count, "seconds": round(seconds, 2)}


def _default_state_seconds_min() -> float:
    return gates_mod.load_defaults()[STATE_SECONDS_MIN_GATE].value


def tables(
    labels: Iterable[LabelRecord],
    prefill: Iterable[PrefillRecord] | None,
    *,
    state_seconds_min: float | None = None,
) -> dict[str, object]:
    """Return the label tables Raj reads after `pack import` and `clean`.

    Who is audible and what it removed count every label. The energy by mood grid, the tags and
    the states count the clips that stay in the seed (`removed` null), at their kept seconds
    (a `her_nearby` clip's keep range). The agreement compares the rated clips (`carried_from`
    empty) that have a pre-fill record with it, and is absent when `prefill` is None.

    Args:
        labels: The labels.
        prefill: The run's pre-fill records, or None.
        state_seconds_min: A state with fewer seconds is flagged thin; None reads
            `seed.state_seconds_min` from the gate defaults.

    Returns:
        A JSON-ready dict of counts, seconds and ids only.
    """
    records = list(labels)
    threshold = _default_state_seconds_min() if state_seconds_min is None else state_seconds_min
    kept = [r for r in records if r.removed is None]
    out: dict[str, object] = {
        "clips": _cell(len(records), sum(r.duration_s for r in records)),
        "voices": {
            v: _cell(
                sum(1 for r in records if r.voices == v),
                sum(r.duration_s for r in records if r.voices == v),
            )
            for v in VOICES
        },
        "removed": {
            reason: _cell(
                sum(1 for r in records if r.removed == reason),
                sum(r.duration_s for r in records if r.removed == reason),
            )
            for reason in REMOVED
        },
        "trimmed_s": round(sum(r.duration_s - _kept_seconds(r) for r in kept), 2),
        "energy_mood": _grid(kept),
        "tags": {
            tag: _cell(
                sum(1 for r in kept if tag in r.tags),
                sum(_kept_seconds(r) for r in kept if tag in r.tags),
            )
            for tag in TAGS
        },
        "untagged": sum(1 for r in kept if not r.tags),
        "state_seconds_min": threshold,
        "states": {},
        "state_none": _cell(
            sum(1 for r in kept if r.state is None),
            sum(_kept_seconds(r) for r in kept if r.state is None),
        ),
    }
    states: dict[str, object] = {}
    for state in SPEAKING_STATES:
        seconds = sum(_kept_seconds(r) for r in kept if r.state == state)
        cell = _cell(sum(1 for r in kept if r.state == state), seconds)
        states[state] = {**cell, "thin": seconds < threshold}
    out["states"] = states
    if prefill is not None:
        out["agreement"] = _agreement(records, prefill)
    runs: dict[str, int] = {}
    for record in records:
        for run in sorted({ref.split("/", 1)[0] for ref in record.carried_from}):
            runs[run] = runs.get(run, 0) + 1
    out["carried"] = {
        "clips": sum(1 for r in records if r.carried_from),
        "runs": dict(sorted(runs.items())),
    }
    out["notes"] = sum(1 for r in records if r.has_note)
    return out


def _grid(kept: Sequence[LabelRecord]) -> dict[str, object]:
    size = SCALE[1]
    count = [[0] * size for _ in range(size)]
    seconds = [[0.0] * size for _ in range(size)]
    for record in kept:
        count[record.energy - 1][record.mood - 1] += 1
        seconds[record.energy - 1][record.mood - 1] += _kept_seconds(record)
    return {
        "count": count,
        "seconds": [[round(s, 2) for s in row] for row in seconds],
        "energy": {
            "count": [sum(row) for row in count],
            "seconds": [round(sum(row), 2) for row in seconds],
        },
        "mood": {
            "count": [sum(row[k] for row in count) for k in range(size)],
            "seconds": [round(sum(row[k] for row in seconds), 2) for k in range(size)],
        },
    }


def _agreement(
    records: Sequence[LabelRecord], prefill: Iterable[PrefillRecord]
) -> dict[str, object]:
    by_id = {p.segment_id: p for p in prefill}
    pairs = [
        (by_id[r.segment_id], r) for r in records if not r.carried_from and r.segment_id in by_id
    ]
    n = len(pairs)

    def change(field: str) -> dict[str, float]:
        diffs = [abs(getattr(p, field) - getattr(r, field)) for p, r in pairs]
        if not n:
            return {"mean_abs_change": 0.0, "share_changed": 0.0}
        return {
            "mean_abs_change": round(sum(diffs) / n, 3),
            "share_changed": round(sum(1 for d in diffs if d) / n, 3),
        }

    voices = {pre: {final: 0 for final in VOICES} for pre in VOICES}
    for p, r in pairs:
        voices[p.voices][r.voices] += 1
    return {"clips": n, "energy": change("energy"), "mood": change("mood"), "voices": voices}


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _cells(entries: Mapping[str, Mapping[str, object]]) -> str:
    return "; ".join(f"{k} {v['count']} ({v['seconds']:.2f} s)" for k, v in entries.items())


def table_lines(tables: Mapping[str, object]) -> list[str]:
    """Return the tables as printable lines of counts, seconds and ids only."""
    t: dict = dict(tables)
    clips = t["clips"]
    lines = [f"labels: {_plural(clips['count'], 'clip')}, {clips['seconds']:.2f} s"]
    lines.append(f"voices: {_cells(t['voices'])}")
    lines.append(f"removed: {_cells(t['removed'])}; trimmed away {t['trimmed_s']:.2f} s")
    grid = t["energy_mood"]
    lines.append("energy by mood (count/seconds), rows energy 1-5, columns mood 1-5 | total:")
    for k in range(SCALE[1]):
        cells = " ".join(
            f"{c}/{s:.2f}" for c, s in zip(grid["count"][k], grid["seconds"][k], strict=True)
        )
        total = f"{grid['energy']['count'][k]}/{grid['energy']['seconds'][k]:.2f}"
        lines.append(f"  energy {k + 1}: {cells} | {total}")
    moods = " ".join(
        f"{c}/{s:.2f}" for c, s in zip(grid["mood"]["count"], grid["mood"]["seconds"], strict=True)
    )
    lines.append(f"  mood total: {moods}")
    used = {tag: v for tag, v in t["tags"].items() if v["count"]}
    tag_text = _cells(used) + "; " if used else ""
    lines.append(f"tags: {tag_text}none {t['untagged']}")
    none = t["state_none"]
    lines.append(f"states: {_cells(t['states'])}; none {none['count']} ({none['seconds']:.2f} s)")
    thin = [state for state, v in t["states"].items() if v["thin"]]
    if thin:
        lines.append(f"thin states (under {t['state_seconds_min']:g} s): {', '.join(thin)}")
    agreement = t.get("agreement")
    if agreement is not None:
        lines.append(f"pre-fill agreement over {agreement['clips']} rated clips:")
        for field in ("energy", "mood"):
            values = agreement[field]
            lines.append(
                f"  {field}: mean change {values['mean_abs_change']:.3f}, "
                f"changed in {values['share_changed']:.1%}"
            )
        lines.append("  voices pre-filled -> final:")
        for pre, row in agreement["voices"].items():
            lines.append(f"    {pre}: " + ", ".join(f"{k} {v}" for k, v in row.items()))
    carried = t["carried"]
    runs = ", ".join(f"{run} {n}" for run, n in carried["runs"].items())
    lines.append(
        f"carried: {_plural(carried['clips'], 'clip')}" + (f"; from {runs}" if runs else "")
    )
    lines.append(f"notes: {_plural(t['notes'], 'clip')}")
    return lines
