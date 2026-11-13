"""Blind listening packs for Raj's ratings, their sealed keys, and `pack import`.

A pack is a static page, `packs/<pack_id>/index.html`, with its clips in `clips/` beside it and
the key that ties each seeded label to what made it in `packs/<pack_id>.key/`. Six kinds share one
page template, driven by the page data: `labels`, `clean-ab`, `text`, `bakeoff`, `spotcheck` and
`abx`. Raj rates the page in a browser and exports `ratings.csv`; `pack import` checks that every
item is rated before it opens the key, then writes the records later stages read.

Clips are copied byte for byte. Every clip a pack takes was already written at its stored gain by
the stage that made it, so the energy differences between states reach Raj's ear unchanged, and
packing needs no audio processing and no job. Instead, each clip's FLAC header is read, and a pack
whose clips differ in sample rate, channels or bit depth, or carry tags, is refused, so no header
gives a teacher or student away. The page is refused too if it names a teacher, student, voice
family, source file or private term.

Host-side and standard library only. `pack_id`, `render_page` and `page_data` are copies of the
speech bench's (`scripts/model_bench/speech/listening.py:248-263`), because importing that module
reaches PyYAML through `tts_suite` and `probes/tts_engines.py`; a test checks the copies against
the originals in the container.
"""

from __future__ import annotations

import csv
import io
import json
import os
import random
import re
import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path, PurePosixPath

from . import gates as gates_mod
from .errors import InputError
from .jobs import load_families
from .records import SegmentRecord, TakeRecord, read_json, read_jsonl, sha256_file, write_json
from .root import Root
from .states import SPEAKING_STATES

KINDS = ("labels", "clean-ab", "text", "bakeoff", "spotcheck", "abx")
PLACEHOLDER = "__PACK_DATA__"
RATINGS_COLUMNS = ("label", "line_id", "natural", "feeling", "preferred")
TEMPLATE_PATH = Path(__file__).with_name("pack_template.html")
PACKS_DIR = "packs"
PAGE_FILE = "index.html"
CLIPS_DIR = "clips"
KEY_FILE = "key.json"
RATINGS_FILE = "ratings.csv"
KEY_SCHEMA = 1
LABELS_REL = "seed/labels.csv"
LEADS_PER_GROUP = 3
# Families that make or learn a voice; their names, and the name after the prefix, would tell a
# listener which teacher or student made a clip. Job families (`score`, `extract`, `convert`)
# name no voice.
VOICE_FAMILY_PREFIXES = ("teacher-", "train-", "ref-")

_DATA_RE = re.compile(
    r'<script id="pack-data" type="application/json">(?P<data>.*?)</script>', re.DOTALL
)
_PACK_ID_RE = re.compile(r"[0-9a-f]{16}")

# FLAC metadata block types (the FLAC format specification, METADATA_BLOCK_HEADER).
_STREAMINFO = 0
_VORBIS_COMMENT = 4
_TAG_BLOCKS = frozenset({2, 5, 6})  # APPLICATION, CUESHEET, PICTURE


# ---------------------------------------------------------------------------
# Copies of the speech bench's page helpers
# ---------------------------------------------------------------------------


def pack_id(seed: int) -> str:
    """Return the pack's id, derived from the seed, so a rebuild with one seed finds one page.

    A copy of `listening.pack_id` (`listening.py:248-250`): the host runs the standard library
    only, and `listening.py` reaches PyYAML through `tts_engines`.
    """
    return f"{random.Random(f'{seed}:pack').getrandbits(64):016x}"


def render_page(template: str, data: Mapping[str, object]) -> str:
    """Return the template with the page data in its JSON block.

    `</` is escaped, so a value cannot close the block. A copy of `listening.render_page` (`listening.py:253-256`): the host runs the standard
    library only, and `listening.py` reaches PyYAML through `tts_engines`.
    """
    payload = json.dumps(data, ensure_ascii=True, indent=1).replace("</", "<\\/")
    return template.replace(PLACEHOLDER, payload)


def page_data(html: str) -> dict:
    """Return the page data a rendered page holds.

    A copy of `listening.page_data` (`listening.py:259-263`), raising this package's
    `InputError` (exit 2): the host runs the standard library only, and `listening.py` reaches
    PyYAML through `tts_engines`.

    Raises:
        InputError: The page holds no pack-data block, or the block is not JSON.
    """
    match = _DATA_RE.search(html)
    if match is None:
        raise InputError("the page holds no pack-data block")
    try:
        return json.loads(match.group("data"))
    except json.JSONDecodeError as exc:
        raise InputError("the page's pack-data block is not JSON") from exc


def load_template(path: Path = TEMPLATE_PATH) -> str:
    """Read the page template.

    Raises:
        InputError: It is unreadable or does not hold the placeholder exactly once.
    """
    try:
        template = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InputError("the pack page template is unreadable") from exc
    if template.count(PLACEHOLDER) != 1:
        raise InputError(f"the pack page template must hold {PLACEHOLDER} exactly once")
    return template


# ---------------------------------------------------------------------------
# FLAC headers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FlacInfo:
    """What a FLAC file's metadata says about it: its format and whether it carries tags."""

    sample_rate: int
    channels: int
    bits_per_sample: int
    tagged: bool


def flac_info(path: Path) -> FlacInfo:
    """Read a FLAC file's STREAMINFO and note any tag block, without decoding audio.

    A Vorbis comment block counts as tags only when it holds a comment: its vendor string alone
    is written by every encoder. APPLICATION, CUESHEET and PICTURE blocks always count.

    Raises:
        InputError: The file is unreadable, not FLAC, or its metadata is truncated.
    """
    info: tuple[int, int, int] | None = None
    tagged = False
    try:
        with path.open("rb") as handle:
            if handle.read(4) != b"fLaC":
                raise InputError("refused: a clip is not a FLAC file")
            last = False
            while not last:
                header = handle.read(4)
                if len(header) != 4:
                    raise InputError("refused: a clip's FLAC metadata is truncated")
                last = bool(header[0] & 0x80)
                kind = header[0] & 0x7F
                length = int.from_bytes(header[1:4], "big")
                if kind not in (_STREAMINFO, _VORBIS_COMMENT):
                    tagged = tagged or kind in _TAG_BLOCKS
                    handle.seek(length, os.SEEK_CUR)
                    continue
                body = handle.read(length)
                if len(body) != length:
                    raise InputError("refused: a clip's FLAC metadata is truncated")
                if kind == _STREAMINFO:
                    info = _streaminfo(body)
                else:
                    tagged = tagged or _comment_count(body) > 0
    except OSError as exc:
        raise InputError("refused: a clip is unreadable") from exc
    if info is None:
        raise InputError("refused: a clip's FLAC header has no STREAMINFO")
    return FlacInfo(*info, tagged=tagged)


def _streaminfo(body: bytes) -> tuple[int, int, int]:
    if len(body) < 18:
        raise InputError("refused: a clip's FLAC STREAMINFO is truncated")
    # Bytes 10-17: sample rate (20 bits), channels - 1 (3), bits per sample - 1 (5), samples (36).
    packed = int.from_bytes(body[10:18], "big")
    return packed >> 44, ((packed >> 41) & 0x7) + 1, ((packed >> 36) & 0x1F) + 1


def _comment_count(body: bytes) -> int:
    vendor = int.from_bytes(body[0:4], "little")
    start = 4 + vendor
    if len(body) < start + 4:
        raise InputError("refused: a clip's FLAC comment block is truncated")
    return int.from_bytes(body[start : start + 4], "little")


def check_formats(paths: Iterable[Path]) -> None:
    """Refuse clips that differ in sample rate, channels or bit depth, or that carry tags.

    Raises:
        InputError: The clips' formats differ, a clip carries tags, or a clip is not FLAC.
    """
    infos = [flac_info(path) for path in paths]
    formats = {(i.sample_rate, i.channels, i.bits_per_sample) for i in infos}
    if len(formats) > 1:
        raise InputError(
            f"refused: the pack's clips come in {len(formats)} formats (sample rate, channels "
            "or bit depth), which would give their makers away"
        )
    tagged = sum(i.tagged for i in infos)
    if tagged:
        raise InputError(f"refused: {tagged} of the pack's clips carry tags")


# ---------------------------------------------------------------------------
# The blind check
# ---------------------------------------------------------------------------


def forbidden_names(root: Root, extra: Iterable[str] = ()) -> frozenset[str]:
    """Return every name the page must not show, lower-cased.

    These are the voice families of `families.json` (with and without their prefix), the
    students under `students/`, each source file's name and stem from `seed/inventory.jsonl`,
    Raj's private terms from `guard/terms.txt`, and `extra`: the teacher and student names of
    the pack's own inputs.

    Raises:
        InputError: The inventory or terms file is unreadable.
    """
    names: set[str] = set(extra)
    for family in load_families():
        for prefix in VOICE_FAMILY_PREFIXES:
            if family.startswith(prefix):
                names.update((family, family[len(prefix) :]))
    students = root.path / "students"
    if students.is_dir():
        names.update(entry.name for entry in students.iterdir())
    inventory = root.path / "seed" / "inventory.jsonl"
    if inventory.is_file():
        for row in _read_objects(inventory):
            source = PurePosixPath(str(row.get("path", "")))
            names.update((source.name, source.stem))
    terms = root.path / "guard" / "terms.txt"
    if terms.is_file():
        try:
            lines = terms.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            raise InputError("guard/terms.txt is unreadable") from exc
        names.update(line.strip() for line in lines if not line.strip().startswith("#"))
    return frozenset(name.strip().lower() for name in names if name.strip())


def _hits(text: str, forbidden: Iterable[str]) -> int:
    return sum(
        1
        for term in forbidden
        if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text, re.IGNORECASE)
    )


def render(template: str, data: Mapping[str, object], forbidden: Iterable[str]) -> str:
    """Render the page, refusing one that names a forbidden name.

    The page is searched as written and with its data unescaped, so a name with non-ASCII
    characters is found too. The refusal gives a count only: the names are private.

    Raises:
        InputError: The page would name a teacher, student, family, source file or term.
    """
    page = render_page(template, data)
    plain = json.dumps(data, ensure_ascii=False)
    hits = _hits(page + "\n" + plain, forbidden)
    if hits:
        raise InputError(
            f"refused: the page would name {hits} teacher, student, family, source file or "
            "private term(s)"
        )
    return page


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _read_path(root: Root, rel: str) -> Path:
    """Resolve a root-relative input path, refusing one outside the root or missing."""
    path = (root.path / rel).resolve()
    if path == root.path or not path.is_relative_to(root.path):
        raise InputError("refused: an input path resolves outside the root")
    if not path.is_file():
        raise InputError(f"refused: {root.rel(path)} is missing")
    return path


def _read_objects(path: Path) -> list[dict]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise InputError(f"{path.name} is missing or unreadable") from exc
    rows = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InputError(f"{path.name} line {number}: not JSON") from exc
        if not isinstance(row, dict):
            raise InputError(f"{path.name} line {number}: not a JSON object")
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Drafts: what a pack shows and what its key records
# ---------------------------------------------------------------------------

_SCALE = {"type": "scale", "min": 1, "max": 5}
_VERDICT_FIELDS = {
    "verdict": {"type": "choice", "title": "Verdict", "options": ["accept", "reject"]},
    "reason": {"type": "text", "title": "Reason", "required_if": ["verdict", "reject"]},
}
_RATING_FIELDS = {
    "natural": {**_SCALE, "title": "Natural (1-5)"},
    "feeling": {**_SCALE, "title": "Feeling (1-5)"},
    "preferred": {"type": "pick", "title": "Favourite"},
}


@dataclass(frozen=True, slots=True)
class DraftItem:
    """One rated row: its clips (source path and shown name), text, fields and key entry."""

    clips: tuple[tuple[str, str], ...]
    fields: tuple[str, ...]
    key: Mapping[str, object]
    defaults: Mapping[str, object] = field(default_factory=dict)
    text: str | None = None


@dataclass(frozen=True, slots=True)
class DraftGroup:
    """A block of items on the page; `id` is the `line_id` column where a kind has one."""

    id: str
    heading: str
    items: tuple[DraftItem, ...]
    note: str = ""


@dataclass(frozen=True, slots=True)
class Draft:
    """A pack before labels are assigned: groups, fields, columns and the key's inputs."""

    kind: str
    title: str
    intro: tuple[str, ...]
    fields: Mapping[str, Mapping[str, object]]
    columns: tuple[str, ...]
    groups: tuple[DraftGroup, ...]
    inputs: Mapping[str, str]
    names: frozenset[str] = frozenset()


def _labels(prefix: str, count: int, rng: random.Random) -> list[str]:
    """Labels `<prefix>01` to `<prefix>N` in a seeded permutation, so a number says nothing."""
    width = max(2, len(str(count)))
    numbers = list(range(1, count + 1))
    rng.shuffle(numbers)
    return [f"{prefix}{n:0{width}d}" for n in numbers]


def _finalise(
    draft: Draft, seed: int, pid: str
) -> tuple[dict, dict[str, Mapping[str, object]], dict[str, str]]:
    """Assign seeded labels; return the page data, the key's items and the clip sources.

    An item of one clip is labelled by its clip (`c01`); an item of several clips or of text
    gets an item label (`i01`).
    """
    rng = random.Random(f"{seed}:labels")
    all_items = [item for group in draft.groups for item in group.items]
    clip_labels = iter(_labels("c", sum(len(item.clips) for item in all_items), rng))
    multi = sum(1 for item in all_items if len(item.clips) != 1)
    item_labels = iter(_labels("i", multi, rng))
    clips: dict[str, str] = {}
    key_items: dict[str, Mapping[str, object]] = {}
    groups = []
    for group in draft.groups:
        page_items = []
        for item in group.items:
            shown = []
            for src, name in item.clips:
                label = next(clip_labels)
                clips[label] = src
                shown.append({"label": label, "name": name})
            label = shown[0]["label"] if len(shown) == 1 else next(item_labels)
            key_items[label] = item.key
            page_items.append(
                {
                    "label": label,
                    "clips": shown,
                    "text": item.text,
                    "fields": list(item.fields),
                    "defaults": dict(item.defaults),
                }
            )
        groups.append(
            {"id": group.id, "heading": group.heading, "note": group.note, "items": page_items}
        )
    data = {
        "pack": pid,
        "kind": draft.kind,
        "title": draft.title,
        "intro": list(draft.intro),
        "columns": list(draft.columns),
        "fields": {name: dict(spec) for name, spec in draft.fields.items()},
        "ratings_file": RATINGS_FILE,
        "groups": groups,
    }
    return data, key_items, clips


def draft_labels(root: Root, run: str, seed: int) -> Draft:
    """One clip per kept segment, its proposed state preselected, with a "not her" choice.

    Three segments of each flag group with more than one segment lead the page, one block per
    group, so a cluster that is not her is caught early; every other segment follows.
    """
    path = _read_path(root, f"seed/runs/{run}/segments.jsonl")
    kept = sorted(
        (s for s in read_jsonl(path, SegmentRecord) if s.kept), key=lambda s: s.segment_id
    )
    if not kept:
        raise InputError(f"refused: run {run} has no kept segments")
    if any(s.state_proposed not in SPEAKING_STATES for s in kept):
        raise InputError("refused: a segment proposes a state outside the speaking states")
    rng = random.Random(f"{seed}:order")
    by_group: dict[str, list[SegmentRecord]] = {}
    for seg in kept:
        if seg.flag_group:
            by_group.setdefault(seg.flag_group, []).append(seg)
    multi = sorted(name for name, members in by_group.items() if len(members) > 1)
    rng.shuffle(multi)

    def item(seg: SegmentRecord) -> DraftItem:
        return DraftItem(
            clips=((seg.audio, ""),),
            fields=("state", "not_her"),
            key={"segment_id": seg.segment_id, "flag_group": seg.flag_group},
            defaults={"state": seg.state_proposed, "not_her": 0},
        )

    groups = []
    leads: set[str] = set()
    for number, name in enumerate(multi, start=1):
        chosen = rng.sample(by_group[name], min(LEADS_PER_GROUP, len(by_group[name])))
        leads.update(seg.segment_id for seg in chosen)
        groups.append(
            DraftGroup(
                id=f"g{number:02d}",
                heading=f"Set {number} of {len(multi)}",
                note="These clips may be one speaker. Mark any that is not her.",
                items=tuple(item(seg) for seg in chosen),
            )
        )
    rest = [seg for seg in kept if seg.segment_id not in leads]
    rng.shuffle(rest)
    if rest:
        groups.append(
            DraftGroup(id="rest", heading="Segments", items=tuple(item(seg) for seg in rest))
        )
    return Draft(
        kind="labels",
        title="Labels",
        intro=(
            "For each clip, confirm or change its state, and mark it not her if it is someone "
            "else.",
        ),
        fields={
            "state": {"type": "choice", "title": "State", "options": list(SPEAKING_STATES)},
            "not_her": {"type": "flag", "title": "Not her"},
        },
        columns=("label", "state", "not_her"),
        groups=tuple(groups),
        inputs={"run": run},
    )


def draft_clean_ab(root: Root, before: str, after: str, seed: int) -> Draft:
    """Each segment kept after cleaning, before and after, in a seeded A/B order."""
    old = {
        s.segment_id: s
        for s in read_jsonl(_read_path(root, f"seed/runs/{before}/segments.jsonl"), SegmentRecord)
    }
    new = read_jsonl(_read_path(root, f"seed/runs/{after}/segments.jsonl"), SegmentRecord)
    pairs = sorted((s for s in new if s.kept and s.segment_id in old), key=lambda s: s.segment_id)
    if not pairs:
        raise InputError("refused: the two runs share no kept segment")
    rng = random.Random(f"{seed}:order")
    rng.shuffle(pairs)
    items = []
    for seg in pairs:
        sources = {"before": old[seg.segment_id].audio, "after": seg.audio}
        sides = ("after", "before") if rng.random() < 0.5 else ("before", "after")
        items.append(
            DraftItem(
                clips=((sources[sides[0]], "A"), (sources[sides[1]], "B")),
                fields=("preferred", "voice_changed"),
                key={"segment_id": seg.segment_id, "A": sides[0], "B": sides[1]},
            )
        )
    return Draft(
        kind="clean-ab",
        title="Before and after",
        intro=(
            "Each row is one line twice, A and B in a random order. Pick the one you prefer, "
            "and mark voice changed if the two do not sound like the same voice.",
        ),
        fields={
            "preferred": {"type": "choice", "title": "Preferred", "options": ["A", "B"]},
            "voice_changed": {"type": "flag", "title": "Voice changed"},
        },
        columns=("label", "preferred", "voice_changed"),
        groups=(DraftGroup(id="pairs", heading="Pairs", items=tuple(items)),),
        inputs={"before": before, "after": after},
    )


def draft_text(root: Root, version: str, seed: int, per_state: int) -> Draft:
    """`per_state` script lines per state, shown as text, each accepted or rejected."""
    rows = _read_objects(_read_path(root, f"text/{version}/lines.jsonl"))
    for row in rows:
        if not all(isinstance(row.get(k), str) for k in ("line_id", "state", "text")):
            raise InputError("refused: a line lacks a line_id, state or text")
        if row["state"] not in SPEAKING_STATES:
            raise InputError("refused: a line names a state outside the speaking states")
    rng = random.Random(f"{seed}:order")
    groups = []
    for state in SPEAKING_STATES:
        lines = sorted((r for r in rows if r["state"] == state), key=lambda r: r["line_id"])
        if not lines:
            continue
        chosen = rng.sample(lines, min(per_state, len(lines)))
        items = tuple(
            DraftItem(
                clips=(),
                fields=("verdict", "reason"),
                key={"line_id": row["line_id"], "state": state},
                text=row["text"],
            )
            for row in chosen
        )
        groups.append(DraftGroup(id=state, heading=f"State: {state}", items=items))
    if not groups:
        raise InputError(f"refused: text version {version} has no lines")
    return Draft(
        kind="text",
        title="Script lines",
        intro=(
            "Read each line for its state. Accept it, or reject it and say why in a few words.",
        ),
        fields=_VERDICT_FIELDS,
        columns=("label", "verdict", "reason"),
        groups=tuple(groups),
        inputs={"version": version},
    )


def _one_take(takes: Sequence[TakeRecord]) -> TakeRecord:
    """The take a pack plays for one line, state and teacher: the selected one, else the first."""
    selected = [t for t in takes if t.selected]
    return min(selected or takes, key=lambda t: (t.round, t.seed, t.take_id))


def draft_bakeoff(root: Root, run: str, seed: int) -> Draft:
    """Per state and line, every teacher's take, rated natural, feeling and preferred."""
    takes = read_jsonl(_read_path(root, f"teachers/bakeoff/{run}/takes.jsonl"), TakeRecord)
    by_key: dict[tuple[str, str, str], list[TakeRecord]] = {}
    for row in takes:
        by_key.setdefault((row.state, row.line_id, row.teacher), []).append(row)
    rng = random.Random(f"{seed}:order")
    groups = []
    seen_lines: set[str] = set()
    for state in SPEAKING_STATES:
        lines = sorted({line for s, line, _ in by_key if s == state})
        rng.shuffle(lines)
        for line in lines:
            if line in seen_lines:
                raise InputError("refused: a bake-off line id appears in two states")
            seen_lines.add(line)
            chosen = [
                _one_take(rows)
                for (s, ln, _), rows in sorted(by_key.items())
                if (s, ln) == (state, line)
            ]
            rng.shuffle(chosen)
            items = tuple(
                DraftItem(
                    clips=((t.audio, ""),),
                    fields=("natural", "feeling", "preferred"),
                    key={
                        "take_id": t.take_id,
                        "teacher": t.teacher,
                        "state": t.state,
                        "line_id": t.line_id,
                    },
                )
                for t in chosen
            )
            groups.append(DraftGroup(id=line, heading=f"State: {state}", items=items))
    if not groups:
        raise InputError(f"refused: bake-off run {run} has no takes")
    return Draft(
        kind="bakeoff",
        title="Bake-off",
        intro=(
            "Each block is one line in one state, read by several voices in a shuffled order.",
            "Rate natural (1 not at all, 5 completely) and feeling: how well it carries the "
            "state (1 not at all, 5 exactly). Then pick your favourite for the line.",
        ),
        fields=_RATING_FIELDS,
        columns=RATINGS_COLUMNS,
        groups=tuple(groups),
        inputs={"run": run},
        names=frozenset(row.teacher for row in takes),
    )


def draft_spotcheck(root: Root, takes_rel: str, seed: int, per_state: int) -> Draft:
    """`per_state` random selected takes per state of one batch, each accepted or rejected."""
    path = _read_path(root, takes_rel)
    takes = [t for t in read_jsonl(path, TakeRecord) if t.selected]
    rng = random.Random(f"{seed}:order")
    groups = []
    for state in SPEAKING_STATES:
        pool = sorted((t for t in takes if t.state == state), key=lambda t: t.take_id)
        if not pool:
            continue
        chosen = rng.sample(pool, min(per_state, len(pool)))
        items = tuple(
            DraftItem(
                clips=((t.audio, ""),),
                fields=("verdict", "reason"),
                key={"take_id": t.take_id, "state": state},
            )
            for t in chosen
        )
        groups.append(DraftGroup(id=state, heading=f"State: {state}", items=items))
    if not groups:
        raise InputError("refused: the batch has no selected takes")
    return Draft(
        kind="spotcheck",
        title="Spot check",
        intro=(
            "Listen to each take for its state. Accept it, or reject it and say why in a few "
            "words.",
        ),
        fields=_VERDICT_FIELDS,
        columns=("label", "verdict", "reason"),
        groups=tuple(groups),
        inputs={"takes": root.rel(path)},
        names=frozenset(t.teacher for t in takes),
    )


def draft_abx(root: Root, manifest_rel: str, seed: int) -> Draft:
    """Per line, every system's clip rated, and an ABX trial for every pair of systems.

    The manifest is `{"lines": [{"line_id", "state", "clips": {system: root-relative FLAC}}]}`,
    where a system is `seed`, `teacher:<name>` or `student:<id>`. X is a separate copy of A or
    B with its own label, chosen by the seed.
    """
    path = _read_path(root, manifest_rel)
    doc = read_json(path)
    lines = doc.get("lines") if isinstance(doc, dict) else None
    if not isinstance(lines, list) or not lines:
        raise InputError("refused: the ABX manifest has no lines")
    rng = random.Random(f"{seed}:order")
    groups = []
    names: set[str] = set()
    for number, line in enumerate(lines, start=1):
        clips = line.get("clips") if isinstance(line, dict) else None
        if (
            not isinstance(clips, dict)
            or len(clips) < 2
            or not isinstance(line.get("line_id"), str)
            or line.get("state") not in SPEAKING_STATES
            or not all(isinstance(v, str) for v in clips.values())
        ):
            raise InputError("refused: an ABX manifest line is malformed")
        line_id, state = line["line_id"], line["state"]
        names.update(system.split(":", 1)[1] for system in clips if ":" in system)
        systems = sorted(clips)
        ratings = [
            DraftItem(
                clips=((clips[system], ""),),
                fields=("natural", "feeling", "preferred"),
                key={"type": "rating", "system": system, "line_id": line_id, "state": state},
            )
            for system in systems
        ]
        trials = []
        for first, second in combinations(systems, 2):
            a, b = (second, first) if rng.random() < 0.5 else (first, second)
            x = rng.choice("AB")
            trials.append(
                DraftItem(
                    clips=((clips[a], "A"), (clips[b], "B"), (clips[a if x == "A" else b], "X")),
                    fields=("x_is",),
                    key={"type": "abx", "A": a, "B": b, "X": x, "line_id": line_id, "state": state},
                )
            )
        rng.shuffle(ratings)
        rng.shuffle(trials)
        groups.append(
            DraftGroup(
                id=line_id,
                heading=f"Line {number} of {len(lines)}, state: {state}",
                items=(*ratings, *trials),
            )
        )
    return Draft(
        kind="abx",
        title="ABX",
        intro=(
            "Each block is one line. Rate each single clip natural and feeling (1-5) and pick "
            "your favourite.",
            "In each row of three, X is A or B: say which.",
        ),
        fields={
            **_RATING_FIELDS,
            "x_is": {"type": "choice", "title": "X is", "options": ["A", "B"]},
        },
        columns=("label", "line_id", "natural", "feeling", "preferred", "x_is"),
        groups=tuple(groups),
        inputs={"manifest": root.rel(path)},
        names=frozenset(names),
    )


# ---------------------------------------------------------------------------
# Building a pack
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BuiltPack:
    """What `build_pack` wrote, for the command's output: ids and counts only."""

    pack_id: str
    kind: str
    items: int
    clips: int


def draft_for(root: Root, kind: str, options: Mapping[str, str], seed: int) -> Draft:
    """Build the draft of one pack kind from its command options.

    Raises:
        InputError: An unknown kind, or an input the kind's builder refuses.
    """
    gates = gates_mod.load(root.path)
    if kind == "labels":
        return draft_labels(root, options["run"], seed)
    if kind == "clean-ab":
        return draft_clean_ab(root, options["before"], options["after"], seed)
    if kind == "text":
        return draft_text(root, options["version"], seed, int(gates.value("text.sample_n")))
    if kind == "bakeoff":
        return draft_bakeoff(root, options["run"], seed)
    if kind == "spotcheck":
        per_state = int(gates.value("spotcheck.sample_n"))
        return draft_spotcheck(root, options["takes"], seed, per_state)
    if kind == "abx":
        return draft_abx(root, options["manifest"], seed)
    raise InputError(f"refused: unknown pack kind; use one of {', '.join(KINDS)}")


def build_pack(
    root: Root,
    kind: str,
    options: Mapping[str, str],
    seed: int,
    *,
    template_path: Path = TEMPLATE_PATH,
) -> BuiltPack:
    """Write a pack: the key first, then the clips, then the page.

    Everything that can refuse runs before the first write: the inputs, the clips' formats and
    the blind check. The page is written last, so a failure leaves no page to open.

    Raises:
        InputError: The pack exists, an input is refused, the clips' formats differ or carry
            tags, or the page would name a forbidden name.
        RootError: The root refused a write.
    """
    pid = pack_id(seed)
    page_dir = root.path / PACKS_DIR / pid
    key_dir = root.path / PACKS_DIR / f"{pid}.key"
    if page_dir.exists() or key_dir.exists():
        raise InputError(f"refused: pack {pid} already exists; a pack is built once")
    draft = draft_for(root, kind, options, seed)
    data, key_items, clips = _finalise(draft, seed, pid)
    sources = {label: _read_path(root, rel) for label, rel in clips.items()}
    check_formats(sources.values())
    page = render(load_template(template_path), data, forbidden_names(root, draft.names))
    key = {
        "schema": KEY_SCHEMA,
        "pack": pid,
        "kind": kind,
        "seed": seed,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "inputs": dict(draft.inputs),
        "clips": {
            label: {"src": root.rel(path), "sha256": sha256_file(path)}
            for label, path in sources.items()
        },
        "items": {label: dict(info) for label, info in key_items.items()},
    }
    write_json(root, key_dir / KEY_FILE, key)
    for label, path in sources.items():
        root.write_bytes(page_dir / CLIPS_DIR / f"{label}.flac", path.read_bytes())
    root.write_bytes(page_dir / PAGE_FILE, page.encode("utf-8"))
    return BuiltPack(pack_id=pid, kind=kind, items=len(key_items), clips=len(clips))


# ---------------------------------------------------------------------------
# Ratings and the key
# ---------------------------------------------------------------------------


def _valid(spec: Mapping[str, object], value: str, row: Mapping[str, str]) -> bool:
    kind = spec["type"]
    if kind == "scale":
        bounds = (spec["min"], spec["max"])
        return value.isdigit() and int(bounds[0]) <= int(value) <= int(bounds[1])  # type: ignore[arg-type]
    if kind == "choice":
        return value in spec["options"]  # type: ignore[operator]
    if kind in ("flag", "pick"):
        return value in ("0", "1")
    if kind == "text":
        rule = spec.get("required_if")
        if isinstance(rule, list) and row.get(rule[0], "").strip() == rule[1]:
            return bool(value)
        return True
    return False


def check_ratings(data: Mapping[str, object], text: str) -> dict[str, dict[str, str]]:
    """Check an exported `ratings.csv` against the page and return each item's values.

    Complete means the page's columns, every item exactly once, each item's fields valid for
    their type, a reason wherever a verdict is reject, and exactly one favourite per block.

    Raises:
        InputError: The ratings are incomplete; the message gives counts only.
    """
    columns = list(data["columns"])  # type: ignore[call-overload]
    rows = list(csv.reader(io.StringIO(text)))
    if not rows or [cell.strip() for cell in rows[0]] != columns:
        raise InputError(
            "refused: the ratings file does not have this pack's columns; the key stays sealed"
        )
    items = {
        item["label"]: (group, item)
        for group in data["groups"]  # type: ignore[attr-defined]
        for item in group["items"]
    }
    fields: Mapping[str, Mapping[str, object]] = data["fields"]  # type: ignore[assignment]
    got: dict[str, dict[str, str]] = {}
    problems = 0
    for row in rows[1:]:
        if not any(cell.strip() for cell in row):
            continue
        values = {c: v.strip() for c, v in zip(columns, row, strict=False)}
        label = values.get("label", "")
        if len(row) != len(columns) or label not in items or label in got:
            problems += 1
            continue
        group, item = items[label]
        if "line_id" in values and values["line_id"] != group["id"]:
            problems += 1
        got[label] = values
    missing = len(items) - len(got)
    for label, values in got.items():
        _, item = items[label]
        problems += sum(
            not _valid(fields[name], values.get(name, ""), values) for name in item["fields"]
        )
    for group in data["groups"]:  # type: ignore[attr-defined]
        picks = {n for i in group["items"] for n in i["fields"] if fields[n]["type"] == "pick"}
        for name in picks:
            ones = [
                i["label"]
                for i in group["items"]
                if name in i["fields"] and got.get(i["label"], {}).get(name) == "1"
            ]
            problems += len(ones) != 1
    if missing or problems:
        raise InputError(
            f"refused: the ratings are incomplete ({missing} items missing, {problems} values "
            "missing or invalid); the key stays sealed"
        )
    return {
        label: {name: values.get(name, "") for name in items[label][1]["fields"]}
        for label, values in got.items()
    }


def read_key(root: Root, pid: str) -> dict:
    """Open a pack's key; `import_pack` calls it only after the ratings are complete.

    Raises:
        InputError: The key is missing or not a schema 1 key.
    """
    key = read_json(root.path / PACKS_DIR / f"{pid}.key" / KEY_FILE)
    if not isinstance(key, dict) or key.get("schema") != KEY_SCHEMA or key.get("kind") not in KINDS:
        raise InputError("refused: the pack key is not a schema 1 key")
    return key


Emit = Callable[[str], None]


def import_pack(root: Root, pid: str, ratings_path: Path, *, emit: Emit = print) -> None:
    """Check the ratings, then unseal the key and write the kind's decisions.

    Raises:
        InputError: An unknown pack, an unreadable or incomplete ratings file, or a decision
            input (such as `gains.json`) that is missing or malformed.
        RootError: The root refused a write.
    """
    if not _PACK_ID_RE.fullmatch(pid):
        raise InputError("refused: a pack id is 16 lower-case hex digits")
    page_path = root.path / PACKS_DIR / pid / PAGE_FILE
    if not page_path.is_file():
        raise InputError(f"refused: there is no pack {pid}")
    data = page_data(page_path.read_text(encoding="utf-8"))
    try:
        text = ratings_path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise InputError("refused: the ratings file is missing or unreadable") from exc
    ratings = check_ratings(data, text)
    key = read_key(root, pid)
    if key["kind"] != data.get("kind") or set(key["items"]) != set(ratings):
        raise InputError("refused: the key does not match the pack's page")
    gates = gates_mod.load(root.path)
    _IMPORTERS[key["kind"]](root, key, ratings, gates, emit)


def _import_labels(
    root: Root, key: dict, ratings: dict, gates: gates_mod.Gates, emit: Emit
) -> None:
    items: dict[str, dict] = key["items"]
    marked = {label for label in items if ratings[label]["not_her"] == "1"}
    dropped = {items[label]["flag_group"] for label in marked if items[label]["flag_group"]}
    flag_groups = {info["flag_group"] for info in items.values() if info["flag_group"]}
    rows = []
    with_group = 0
    for label, info in items.items():
        if label in marked:
            continue
        if info["flag_group"] in dropped:
            with_group += 1
            continue
        rows.append((info["segment_id"], ratings[label]["state"]))
    text = "segment_id,state\n" + "".join(f"{sid},{state}\n" for sid, state in sorted(rows))
    root.write_bytes(LABELS_REL, text.encode("utf-8"))
    emit(
        f"labels: {len(rows)} segments written to {LABELS_REL}; {len(marked)} marked not her, "
        f"{with_group} more dropped with {len(dropped)} flag groups; "
        f"{len(flag_groups - dropped)} flag groups confirmed"
    )


def _import_clean_ab(
    root: Root, key: dict, ratings: dict, gates: gates_mod.Gates, emit: Emit
) -> None:
    rows = sorted(
        (
            {
                "segment_id": info["segment_id"],
                "preferred": info[ratings[label]["preferred"]],
                "voice_changed": ratings[label]["voice_changed"] == "1",
            }
            for label, info in key["items"].items()
        ),
        key=lambda row: row["segment_id"],
    )
    counts = {
        "pairs": len(rows),
        "after_preferred": sum(row["preferred"] == "after" for row in rows),
        "before_preferred": sum(row["preferred"] == "before" for row in rows),
        "voice_changed": sum(row["voice_changed"] for row in rows),
    }
    after = key["inputs"]["after"]
    rel = f"seed/runs/{after}/clean_review.json"
    doc = {
        "schema": 1,
        "pack": key["pack"],
        "before": key["inputs"]["before"],
        "after": after,
        "counts": counts,
        "segments": rows,
    }
    write_json(root, rel, doc)
    emit(
        f"clean-ab: {counts['pairs']} pairs, after preferred {counts['after_preferred']}, "
        f"voice changed {counts['voice_changed']}; written to {rel}"
    )


def _verdicts(key: dict, ratings: dict, id_field: str) -> tuple[dict[str, dict], list[dict]]:
    states: dict[str, dict] = {}
    rejected = []
    for label, info in key["items"].items():
        entry = states.setdefault(info["state"], {"n": 0, "rejected": 0})
        entry["n"] += 1
        if ratings[label]["verdict"] == "reject":
            entry["rejected"] += 1
            reason = ratings[label]["reason"]
            rejected.append({id_field: info[id_field], "state": info["state"], "reason": reason})
    ordered = {s: states[s] for s in SPEAKING_STATES if s in states}
    rejected.sort(key=lambda row: (row["state"], row[id_field]))
    return ordered, rejected


def _import_text(root: Root, key: dict, ratings: dict, gates: gates_mod.Gates, emit: Emit) -> None:
    reject_max = gates.value("text.reject_max")
    states, rejected = _verdicts(key, ratings, "line_id")
    for entry in states.values():
        entry["pass"] = entry["rejected"] <= reject_max
    version = key["inputs"]["version"]
    rel = f"text/{version}/review.json"
    doc = {
        "schema": 1,
        "pack": key["pack"],
        "version": version,
        "thresholds": {
            "text.sample_n": gates.value("text.sample_n"),
            "text.reject_max": reject_max,
        },
        "states": states,
        "rejected": rejected,
    }
    write_json(root, rel, doc)
    for state, entry in states.items():
        verdict = "pass" if entry["pass"] else "fail"
        emit(f"text {state}: {entry['rejected']} of {entry['n']} rejected, {verdict}")
    emit(f"text: review written to {rel}")


def _import_spotcheck(
    root: Root, key: dict, ratings: dict, gates: gates_mod.Gates, emit: Emit
) -> None:
    reject_max = gates.value("spotcheck.reject_max")
    states, rejected = _verdicts(key, ratings, "take_id")
    for entry in states.values():
        entry["accepted"] = entry["rejected"] <= reject_max
    takes_rel = key["inputs"]["takes"]
    rel = PurePosixPath(takes_rel).with_name("spotcheck.json").as_posix()
    doc = {
        "schema": 1,
        "pack": key["pack"],
        "takes": takes_rel,
        "thresholds": {
            "spotcheck.sample_n": gates.value("spotcheck.sample_n"),
            "spotcheck.reject_max": reject_max,
        },
        "accepted": all(entry["accepted"] for entry in states.values()),
        "states": states,
        "rejected": rejected,
    }
    write_json(root, rel, doc)
    for state, entry in states.items():
        verdict = "accepted" if entry["accepted"] else "not accepted"
        emit(f"spotcheck {state}: {entry['rejected']} of {entry['n']} rejected, {verdict}")
    emit(f"spotcheck: batch {'accepted' if doc['accepted'] else 'not accepted'}; written to {rel}")


def _read_gains(root: Root, rel: str) -> dict[str, float]:
    gains = read_json(_read_path(root, rel))
    if not isinstance(gains, dict) or not all(
        isinstance(v, int | float) and not isinstance(v, bool) for v in gains.values()
    ):
        raise InputError(f"refused: {rel} must map each teacher to its gain")
    return gains


def _import_bakeoff(
    root: Root, key: dict, ratings: dict, gates: gates_mod.Gates, emit: Emit
) -> None:
    base = f"teachers/bakeoff/{key['inputs']['run']}"
    gains = _read_gains(root, f"{base}/gains.json")
    takes = read_jsonl(_read_path(root, f"{base}/takes.jsonl"), TakeRecord)
    weak_below = gates.value("bakeoff.weak_rating_below")
    pass_min = gates.value("bakeoff.pass_fraction_min")
    scores: dict[tuple[str, str], dict[str, list[int]]] = {}
    for label, info in key["items"].items():
        entry = scores.setdefault(
            (info["state"], info["teacher"]), {"natural": [], "feeling": [], "preferred": []}
        )
        for name in entry:
            entry[name].append(int(ratings[label][name]))
    states: dict[str, dict] = {}
    for state in SPEAKING_STATES:
        teachers = sorted(t for s, t in scores if s == state)
        if not teachers:
            continue

        def rank(teacher: str, state: str = state) -> tuple[float, float, str]:
            entry = scores[(state, teacher)]
            preferred = statistics.mean(entry["preferred"])
            return (-preferred, -statistics.mean(entry["natural"]), teacher)

        chosen = min(teachers, key=rank)
        if chosen not in gains:
            raise InputError(f"refused: {base}/gains.json has no gain for teacher {chosen}")
        entry = scores[(state, chosen)]
        own = [t for t in takes if t.state == state and t.teacher == chosen]
        keep_rate = sum(t.kept for t in own) / len(own) if own else 0.0
        feeling = statistics.mean(entry["feeling"])
        reasons = []
        if feeling < weak_below:
            reasons.append("rating")
        if keep_rate < pass_min:
            reasons.append("keep-rate")
        states[state] = {
            "teacher": chosen,
            "gain": gains[chosen],
            "mean_preferred": statistics.mean(entry["preferred"]),
            "mean_natural": statistics.mean(entry["natural"]),
            "mean_feeling": feeling,
            "keep_rate": keep_rate,
            "weak": bool(reasons),
            "weak_reasons": reasons,
        }
    rel = f"{base}/choice.json"
    doc = {
        "schema": 1,
        "pack": key["pack"],
        "run": key["inputs"]["run"],
        "thresholds": {
            "bakeoff.weak_rating_below": weak_below,
            "bakeoff.pass_fraction_min": pass_min,
        },
        "states": states,
    }
    write_json(root, rel, doc)
    for state, entry in states.items():
        emit(
            f"choice {state}: teacher {entry['teacher']}, "
            f"mean feeling {entry['mean_feeling']:.2f}, keep rate {entry['keep_rate']:.2f}, "
            f"weak {'yes' if entry['weak'] else 'no'}"
        )
    emit(f"bakeoff: choice written to {rel}")


def _import_abx(root: Root, key: dict, ratings: dict, gates: gates_mod.Gates, emit: Emit) -> None:
    pairs: dict[tuple[str, str], dict[str, int]] = {}
    systems: dict[str, dict[str, list[int]]] = {}
    for label, info in key["items"].items():
        values = ratings[label]
        if info["type"] == "abx":
            a, b = sorted((info["A"], info["B"]))
            entry = pairs.setdefault((a, b), {"trials": 0, "correct": 0})
            entry["trials"] += 1
            entry["correct"] += values["x_is"] == info["X"]
        else:
            entry = systems.setdefault(
                info["system"], {"natural": [], "feeling": [], "preferred": []}
            )
            for name in entry:
                entry[name].append(int(values[name]))
    abx = [{"a": a, "b": b, **counts} for (a, b), counts in sorted(pairs.items())]
    tallies = {
        system: {
            "n": len(entry["natural"]),
            "natural_mean": statistics.mean(entry["natural"]),
            "feeling_mean": statistics.mean(entry["feeling"]),
            "preferred": sum(entry["preferred"]),
        }
        for system, entry in sorted(systems.items())
    }
    manifest = key["inputs"]["manifest"]
    rel = PurePosixPath(manifest).with_name("abx_tallies.json").as_posix()
    doc = {"schema": 1, "pack": key["pack"], "manifest": manifest, "abx": abx, "ratings": tallies}
    write_json(root, rel, doc)
    for row in abx:
        emit(f"abx {row['a']} vs {row['b']}: {row['correct']} of {row['trials']} correct")
    emit(f"abx: tallies written to {rel}")


_IMPORTERS: dict[str, Callable[[Root, dict, dict, gates_mod.Gates, Emit], None]] = {
    "labels": _import_labels,
    "clean-ab": _import_clean_ab,
    "text": _import_text,
    "bakeoff": _import_bakeoff,
    "spotcheck": _import_spotcheck,
    "abx": _import_abx,
}
