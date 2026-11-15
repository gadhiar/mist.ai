"""SDH subtitle cues of the target, the spans made from them, and the recall check on them.

Runs in the base image (family `score`, which holds ffmpeg) as
`/opt/venv-score/bin/python -I -m scripts.voice_data.stages.sdh --data /data --step scan|score`.

- `scan --source S`: with `seed/raw/`, `seed/inventory.jsonl` and `config/sdh.json` mounted
  read-only, find the source's SDH stream, convert it to SRT in memory with ffmpeg, and write the
  times and flags of the target's cues to `seed/sdh/<S>.jsonl`, with the counts beside it.
- `score --run R --min-s A --max-s B`: with `seed/sdh/` and the run's `segments.jsonl` mounted
  read-only, write `seed/runs/<R>/sdh-score.json`: per source and overall, the share of the
  held-out spans' seconds that the run's kept segments cover, primary and flagged apart.

No cue text and no label leaves this process: the SRT text, the labels and each stream's title
are held in memory and compared, and every file and printed line holds ids, counts, times and
shares. ffmpeg's stderr can echo the input's name, so it is discarded and a failure names the
source id.

`source_spans` is the one definition of the spans: `extract` imports it for its SDH anchors and
`score` for the held-out spans, so the two agree on which spans steered retrieval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..commands import extract as extract_contract
from ..commands import sdh as contract
from ..errors import InputError
from ..records import SourceRecord, read_jsonl

SCHEMA = 1
# Her cues closer than this are one span (Interfaces, SDH spans).
MERGE_GAP_S = 0.5
# Every fifth kept span of a source, in time order, is held out of the anchors for `score`.
HOLD_OUT_EVERY = 5
EPS = 1e-6
SOURCE_ID = re.compile(r"src-[0-9]{4,}")
# Bitmap subtitle codecs: ffmpeg cannot turn them into text, so an SDH stream in one is skipped.
IMAGE_CODECS = frozenset(
    {"hdmv_pgs_subtitle", "pgssub", "dvd_subtitle", "dvdsub", "dvb_subtitle", "dvbsub", "xsub"}
)
# An untagged or undetermined stream is taken as English, as a single-language release tags none.
ENGLISH_TAGS = frozenset({"", "und", "eng", "en"})
SDH_TITLE = re.compile(r"(?<![a-z0-9])sdh(?![a-z0-9])", re.IGNORECASE)
TIMING = re.compile(
    r"(\d+):(\d{1,2}):(\d{1,2})[,.](\d{1,3})\s*-->\s*(\d+):(\d{1,2}):(\d{1,2})[,.](\d{1,3})"
)
# Markup inside a cue: `<i>` and `</i>` set italics; other HTML-like tags and `{...}` overrides
# carry no text.
MARKUP = re.compile(r"(?P<italic><\s*(?P<close>/)?\s*i\s*>)|<[^>]*>|\{[^}]*\}", re.IGNORECASE)
LEAD = re.compile(r"\s*-?\s*")


class StageError(RuntimeError):
    """The step cannot proceed; the message names ids and counts only."""


# Stream choice.


@dataclass(frozen=True, slots=True)
class StreamChoice:
    """The SDH stream taken, if any, and how many SDH streams there were and were skipped."""

    index: int | None
    candidates: int
    image_skipped: int


def _tag(stream: Mapping[str, object], key: str) -> str | None:
    tags = stream.get("tags")
    if not isinstance(tags, Mapping):
        return None
    value = next((v for k, v in tags.items() if str(k).lower() == key), None)
    return value if isinstance(value, str) else None


def is_sdh(stream: Mapping[str, object]) -> bool:
    """Return whether an English subtitle stream is SDH: by its title, or its disposition."""
    if stream.get("codec_type") != "subtitle":
        return False
    if (_tag(stream, "language") or "").strip().lower() not in ENGLISH_TAGS:
        return False
    disposition = stream.get("disposition")
    if isinstance(disposition, Mapping) and disposition.get("hearing_impaired") == 1:
        return True
    title = _tag(stream, "title")
    return bool(title and SDH_TITLE.search(title))


def choose_stream(streams: Iterable[Mapping[str, object]]) -> StreamChoice:
    """Return the lowest-index text SDH stream; image SDH streams are skipped and counted."""
    found = sorted((s for s in streams if is_sdh(s)), key=lambda s: int(s["index"]))  # type: ignore[call-overload]
    text = [s for s in found if str(s.get("codec_name") or "").lower() not in IMAGE_CODECS]
    index = int(text[0]["index"]) if text else None  # type: ignore[call-overload]
    return StreamChoice(index=index, candidates=len(found), image_skipped=len(found) - len(text))


# Cues.


@dataclass(frozen=True, slots=True)
class Cue:
    """One of her cues as `seed/sdh/<source_id>.jsonl` holds it: times and two flags."""

    start_s: float
    end_s: float
    multi: bool
    offscreen: bool

    def to_json(self) -> dict[str, object]:
        """Return the cue's line of the cue file."""
        return {
            "start_s": self.start_s,
            "end_s": self.end_s,
            "multi": self.multi,
            "offscreen": self.offscreen,
        }

    @classmethod
    def from_json(cls, data: object) -> Cue:
        """Read one line of a cue file.

        Raises:
            StageError: The line is not a cue.
        """
        if not isinstance(data, Mapping):
            raise StageError("a cue line is not an object")
        start, end = data.get("start_s"), data.get("end_s")
        multi, offscreen = data.get("multi"), data.get("offscreen")
        numbers = all(isinstance(v, int | float) and not isinstance(v, bool) for v in (start, end))
        if not numbers or not isinstance(multi, bool) or not isinstance(offscreen, bool):
            raise StageError("a cue line lacks its times or flags")
        return cls(float(start), float(end), multi, offscreen)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class SrtCue:
    """One SRT cue, held in memory only."""

    start_s: float
    end_s: float
    lines: tuple[str, ...]


def _seconds(hours: str, minutes: str, seconds: str, millis: str) -> float:
    total = int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis.ljust(3, "0")) / 1000
    return round(total, 3)


def parse_srt(text: str) -> list[SrtCue]:
    """Return the cues of an SRT text; a block without a timing line is skipped."""
    text = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    cues = []
    for block in re.split(r"\n[ \t]*\n", text):
        lines = block.split("\n")
        for at, line in enumerate(lines):
            match = TIMING.search(line)
            if match is None:
                continue
            groups = match.groups()
            cues.append(
                SrtCue(
                    start_s=_seconds(*groups[:4]),
                    end_s=_seconds(*groups[4:]),
                    lines=tuple(x for x in lines[at + 1 :] if x.strip()),
                )
            )
            break
    return cues


def label_patterns(labels: Iterable[str]) -> list[re.Pattern[str]]:
    """Return a pattern per label: `[<label>]`, any case, with or without a colon."""
    return [
        re.compile(r"\[\s*" + re.escape(label.strip()) + r"\s*:?\s*\]:?", re.IGNORECASE)
        for label in labels
        if label.strip()
    ]


def _label_end(plain: str, patterns: Sequence[re.Pattern[str]]) -> int | None:
    # A line is hers when, after an optional dialogue dash, it starts with one of her labels.
    lead = LEAD.match(plain).end()  # type: ignore[union-attr]
    for pattern in patterns:
        match = pattern.match(plain, lead)
        if match is not None:
            return match.end()
    return None


def _styled_lines(lines: Sequence[str]) -> list[list[tuple[str, bool]]]:
    """Return each line's characters with whether `<i>` holds them; markup is dropped.

    Italics are tracked across the cue's lines, as a cue may open `<i>` on one line and close it
    on the next.
    """
    out: list[list[tuple[str, bool]]] = []
    depth = 0
    for line in lines:
        chars: list[tuple[str, bool]] = []
        pos = 0
        for match in MARKUP.finditer(line):
            chars.extend((c, depth > 0) for c in line[pos : match.start()])
            if match.group("italic"):
                depth = max(0, depth - 1) if match.group("close") else depth + 1
            pos = match.end()
        chars.extend((c, depth > 0) for c in line[pos:])
        out.append(chars)
    return out


def classify(lines: Sequence[str], patterns: Sequence[re.Pattern[str]]) -> tuple[bool, bool] | None:
    """Return `(multi, offscreen)` when the cue is hers, else `None`.

    `multi`: another line starts a speaker's line, with a dash or a bracket. `offscreen`: every
    character she speaks, from after her label to the next speaker's line, is in italics.
    """
    styled = _styled_lines(lines)
    plains = ["".join(c for c, _ in chars) for chars in styled]
    ends = [_label_end(plain, patterns) for plain in plains]
    her = next((i for i, end in enumerate(ends) if end is not None), None)
    if her is None:
        return None
    speaker = [plain.lstrip().startswith(("-", "[")) for plain in plains]
    multi = any(speaker[j] and ends[j] is None for j in range(len(plains)) if j != her)
    stop = next((j for j in range(her + 1, len(plains)) if speaker[j]), len(plains))
    spoken = styled[her][ends[her] :] + [c for chars in styled[her + 1 : stop] for c in chars]
    marks = [italic for c, italic in spoken if not c.isspace()]
    return multi, bool(marks) and all(marks)


def her_cues(text: str, labels: Iterable[str]) -> tuple[list[Cue], int]:
    """Return her cues in an SRT text, and how many cues it holds."""
    patterns = label_patterns(labels)
    cues = parse_srt(text)
    hers = []
    for cue in cues:
        flags = classify(cue.lines, patterns)
        if flags is not None:
            hers.append(Cue(cue.start_s, cue.end_s, *flags))
    return hers, len(cues)


# Spans.


@dataclass(frozen=True, slots=True)
class SdhSpan:
    """Her merged cues: a span is `offscreen` when any of its cues is."""

    start_s: float
    end_s: float
    offscreen: bool
    cues: int


@dataclass(frozen=True, slots=True)
class SourceSpans:
    """One source's cues and the spans made from them.

    Attributes:
        found: Her cues; `found_offscreen` of them offscreen, `multi` of them with a second
            speaker (never merged into a span).
        merged: Her non-multi cues merged across gaps of at most `MERGE_GAP_S`.
        kept: The merged spans of `min_s`-`max_s`, in time order.
        held_out: Every `HOLD_OUT_EVERY`-th kept span: measured by `score`, never an anchor.
        anchors: The other kept spans.
    """

    found: int
    found_offscreen: int
    multi: int
    merged: tuple[SdhSpan, ...]
    kept: tuple[SdhSpan, ...]
    held_out: tuple[SdhSpan, ...]
    anchors: tuple[SdhSpan, ...]


def source_spans(cues: Sequence[Cue], min_s: float, max_s: float) -> SourceSpans:
    """Return a source's spans: merged, kept by length, and split into held out and anchors.

    A multi cue holds another speaker, so it is never merged and ends the run it interrupts.
    """
    ordered = sorted(cues, key=lambda c: (c.start_s, c.end_s))
    runs: list[tuple[float, float, bool, int]] = []
    joinable = False
    for cue in ordered:
        if cue.multi:
            joinable = False
            continue
        if joinable and cue.start_s - runs[-1][1] <= MERGE_GAP_S + EPS:
            start, end, offscreen, count = runs[-1]
            runs[-1] = (start, max(end, cue.end_s), offscreen or cue.offscreen, count + 1)
        else:
            runs.append((cue.start_s, cue.end_s, cue.offscreen, 1))
        joinable = True
    merged = tuple(SdhSpan(round(s, 3), round(e, 3), o, n) for s, e, o, n in runs)
    kept = tuple(s for s in merged if min_s - EPS <= s.end_s - s.start_s <= max_s + EPS)
    return SourceSpans(
        found=len(ordered),
        found_offscreen=sum(c.offscreen for c in ordered),
        multi=sum(c.multi for c in ordered),
        merged=merged,
        kept=kept,
        held_out=tuple(s for n, s in enumerate(kept, 1) if n % HOLD_OUT_EVERY == 0),
        anchors=tuple(s for n, s in enumerate(kept, 1) if n % HOLD_OUT_EVERY != 0),
    )


def read_cues(path: Path) -> list[Cue]:
    """Read a source's cue file.

    Raises:
        StageError: The file is unreadable or a line is not a cue; the message names the file.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        return [Cue.from_json(json.loads(line)) for line in lines if line.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, StageError) as exc:
        raise StageError(f"{contract.SDH_REL}/{path.name} is not a cue file") from exc


def read_spans(
    data: Path, source_ids: Iterable[str], min_s: float, max_s: float
) -> dict[str, SourceSpans]:
    """Return the spans of each listed source that has a cue file; `{}` when no scan has run."""
    folder = data / contract.SDH_REL
    if not folder.is_dir():
        return {}
    found = {}
    for sid in source_ids:
        path = folder / contract.cues_name(sid)
        if path.is_file():
            found[sid] = source_spans(read_cues(path), min_s, max_s)
    return found


# Coverage.


def union(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return the intervals merged where they overlap or touch, in order."""
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def covered_s(span: SdhSpan, cover: Sequence[tuple[float, float]]) -> float:
    """Return the seconds of `span` inside the union `cover`."""
    return sum(max(0.0, min(end, span.end_s) - max(start, span.start_s)) for start, end in cover)


# The steps.


def _run(argv: Sequence[str]) -> subprocess.CompletedProcess[bytes]:
    """Run a tool with both streams captured; tests replace this function."""
    return subprocess.run(list(argv), capture_output=True, check=False)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(text, encoding="utf-8")
    os.replace(part, path)


def _write_json(path: Path, payload: object) -> None:
    _write_text(path, json.dumps(payload, indent=2) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _labels(data: Path) -> list[str]:
    try:
        config = json.loads((data / contract.CONFIG_REL).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StageError(f"{contract.CONFIG_REL} is missing or not JSON") from exc
    labels = config.get("labels") if isinstance(config, dict) else None
    if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
        raise StageError(f"{contract.CONFIG_REL} holds no list of labels")
    if not label_patterns(labels):
        raise StageError(f"{contract.CONFIG_REL} holds no labels")
    return labels


def _source_path(data: Path, source_id: str) -> Path:
    try:
        records = read_jsonl(data / contract.INVENTORY_REL, SourceRecord)
    except InputError as exc:
        raise StageError(f"{contract.INVENTORY_REL} is unreadable") from exc
    record = next((r for r in records if r.source_id == source_id), None)
    if record is None:
        raise StageError(f"{source_id} is not in the inventory")
    path = (data / record.path).resolve()
    if not path.is_relative_to((data / contract.RAW_REL).resolve()):
        raise StageError(f"{source_id}'s file is not under {contract.RAW_REL}")
    return path


def _streams(path: Path, source_id: str) -> list[Mapping[str, object]]:
    proc = _run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)])
    if proc.returncode != 0:
        raise StageError(f"{source_id}: ffprobe exited {proc.returncode}")
    try:
        data = json.loads(proc.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StageError(f"{source_id}: ffprobe output is not JSON") from exc
    streams = data.get("streams") if isinstance(data, dict) else None
    if not isinstance(streams, list):
        raise StageError(f"{source_id}: ffprobe found no streams")
    return [s for s in streams if isinstance(s, Mapping) and isinstance(s.get("index"), int)]


def _subtitle_text(path: Path, index: int, source_id: str) -> str:
    argv = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", f"0:{index}"]
    proc = _run([*argv, "-f", "srt", "-"])
    if proc.returncode != 0:
        raise StageError(f"{source_id}: conversion of stream {index} exited {proc.returncode}")
    return proc.stdout.decode("utf-8", errors="replace")


def scan(data: Path, source_id: str) -> list[str]:
    """Write one source's cue file and counts; return the root-relative files written."""
    if not SOURCE_ID.fullmatch(source_id):
        raise StageError("the scan needs a source id")
    labels = _labels(data)
    path = _source_path(data, source_id)
    choice = choose_stream(_streams(path, source_id))
    cues: list[Cue] = []
    total = 0
    if choice.index is not None:
        cues, total = her_cues(_subtitle_text(path, choice.index, source_id), labels)
    cues_rel = f"{contract.SDH_REL}/{contract.cues_name(source_id)}"
    _write_text(data / cues_rel, "".join(json.dumps(c.to_json()) + "\n" for c in cues))
    counts_rel = f"{contract.SDH_REL}/{contract.scan_name(source_id)}"
    counts = {
        "schema": SCHEMA,
        "source_id": source_id,
        "stream": choice.index,
        "sdh_streams": choice.candidates,
        "image_skipped": choice.image_skipped,
        "cues": total,
        "hers": len(cues),
        "multi": sum(c.multi for c in cues),
        "offscreen": sum(c.offscreen for c in cues),
    }
    _write_json(data / counts_rel, counts)
    stream = "none" if choice.index is None else str(choice.index)
    print(f"sdh scan: {source_id} stream {stream}, {total} cues, {len(cues)} hers")
    return [cues_rel, counts_rel]


def _segments(path: Path) -> list[tuple[str, float, float, str]]:
    """Return the kept segments of a run as (source id, start, end, cluster)."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise StageError("the run's segments are unreadable") from exc
    kept = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if row["kept"] is True:
                kept.append(
                    (str(row["source_id"]), float(row["start_s"]), float(row["end_s"]),
                     str(row["cluster"]))
                )  # fmt: skip
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise StageError(f"segments line {number} is not a segment") from exc
    return kept


def _score_row(spans: Sequence[SdhSpan], primary: float, flagged: float) -> dict[str, object]:
    seconds = sum(s.end_s - s.start_s for s in spans)
    return {
        "held_out_spans": len(spans),
        "held_out_s": round(seconds, 3),
        "primary_share": round(primary / seconds, 4) if seconds > 0 else None,
        "flagged_share": round(flagged / seconds, 4) if seconds > 0 else None,
    }


def score(data: Path, run_id: str, min_s: float, max_s: float) -> list[str]:
    """Write the run's `sdh-score.json`; return the root-relative file written."""
    if not extract_contract.RUN_ID.fullmatch(run_id) or ".." in run_id:
        raise StageError("a run id is one folder name under seed/runs")
    run_rel = extract_contract.run_rel(run_id)
    kept = _segments(data / run_rel / extract_contract.SEGMENTS_NAME)
    folder = data / contract.SDH_REL
    sids = sorted(
        p.name.removesuffix(".jsonl")
        for p in folder.glob("src-*.jsonl")
        if SOURCE_ID.fullmatch(p.name.removesuffix(".jsonl"))
    )
    rows = []
    all_spans: list[SdhSpan] = []
    totals = [0.0, 0.0]
    for sid in sids:
        held = source_spans(read_cues(folder / contract.cues_name(sid)), min_s, max_s).held_out
        covered = []
        for cluster in ("primary", "flagged"):
            cover = union((s, e) for k_sid, s, e, c in kept if k_sid == sid and c == cluster)
            covered.append(sum(covered_s(span, cover) for span in held))
        rows.append({"source_id": sid, **_score_row(held, covered[0], covered[1])})
        all_spans.extend(held)
        totals = [totals[0] + covered[0], totals[1] + covered[1]]
    result = {
        "schema": SCHEMA,
        "run": run_id,
        "rule": {
            "min_s": min_s,
            "max_s": max_s,
            "merge_gap_s": MERGE_GAP_S,
            "hold_out_every": HOLD_OUT_EVERY,
        },
        "overall": _score_row(all_spans, totals[0], totals[1]),
        "sources": rows,
    }
    score_rel = f"{run_rel}/{contract.SCORE_NAME}"
    _write_json(data / score_rel, result)
    print(f"sdh score: {len(sids)} sources, {len(all_spans)} held-out spans")
    return [score_rel]


def main(argv: Sequence[str] | None = None) -> int:
    """Run one step; write `outputs.json` for the executor to verify."""
    parser = argparse.ArgumentParser(prog="scripts.voice_data.stages.sdh")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--step", required=True, choices=contract.STEPS)
    parser.add_argument("--source")
    parser.add_argument("--run")
    parser.add_argument("--min-s", type=float)
    parser.add_argument("--max-s", type=float)
    args = parser.parse_args(argv)
    try:
        if args.step == "scan":
            written = scan(args.data, args.source or "")
            outputs_dir = args.data / contract.SDH_REL
        else:
            if args.run is None or args.min_s is None or args.max_s is None:
                raise StageError("score needs --run, --min-s and --max-s")
            written = score(args.data, args.run, args.min_s, args.max_s)
            outputs_dir = args.data / extract_contract.run_rel(args.run)
    except StageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    files = [{"rel": rel, "sha256": _sha256(args.data / rel)} for rel in written]
    _write_json(outputs_dir / "outputs.json", {"schema": SCHEMA, "files": files})
    return 0


if __name__ == "__main__":
    sys.exit(main())
