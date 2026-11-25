"""The training text the teacher speaks: public transcripts plus invented assistant lines.

The 26 ear-test lines in `scripts/model_bench/speech/lines.json` judge the trained voice, so no
training line may teach it those words: a line that equals one of them, or shares a run of
`NGRAM` words with one, after normalisation, is held out. Every line is neutral, because stage 1
trains the neutral state only.

Each text version holds only new lines: a line whose id or normalised text an earlier version of
the root holds is dropped, so a second teacher run speaks nothing the first one spoke. A line is
kept only from the `text.line_chars_min` gate to `MAX_CHARS` characters after `clean`, because the
first teacher run kept a take far more often for long lines than for short ones; `MIN_CHARS` stays
the floor any gate value must respect.

The public sources are read from the folder a rehearsal root keeps them in (`public/`): one
LibriTTS-R test-clean utterance per `.normalized.txt` file, and Expresso's
`read_transcriptions.txt` (`<utterance id><TAB><text>` per line). Either may be absent; the build
counts it as 0 and refuses only when both are. The source texts are public, but they are still
never printed, so the command's output stays counts and root-relative paths alone.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .errors import InputError
from .gates import Gates
from .states import NEUTRAL_STATE

ASSISTANT_LINES_PATH = Path(__file__).with_name("text_lines.jsonl")
EAR_TEST_PATH = Path(__file__).resolve().parents[1] / "model_bench" / "speech" / "lines.json"
LIBRITTSR_DIR = ("librittsr", "LibriTTS_R", "test-clean")
LIBRITTSR_SUFFIX = ".normalized.txt"
EXPRESSO_FILE = ("expresso", "expresso", "read_transcriptions.txt")

ASSISTANT = "assistant"
LIBRITTSR = "librittsr"
EXPRESSO = "expresso"
# Earlier sources win a duplicate, so the invented lines, written for this register, come first.
SOURCE_ORDER = (ASSISTANT, LIBRITTSR, EXPRESSO)
PUBLIC_SOURCES = (LIBRITTSR, EXPRESSO)

MIN_CHARS = 20
MAX_CHARS = 200
LINE_CHARS_GATE = "text.line_chars_min"
NGRAM = 8
VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
TEXT_DIR = "text"
LINES_NAME = "lines.jsonl"
SUMMARY_NAME = "summary.json"

# Bracketed spans are non-speech marks (a laugh, a breath); a teacher would read them aloud.
_TAG = re.compile(r"\[[^\]]*\]|<[^>]*>|\{[^}]*\}")
_SPACE = re.compile(r"\s+")
_APOSTROPHE = re.compile(r"['’]")
_NON_WORD = re.compile(r"[\W_]+")
_ID_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]")


@dataclass(frozen=True, slots=True)
class SourceLine:
    """One candidate line: its source, its id in the output, and its raw text."""

    source: str
    line_id: str
    text: str


@dataclass(frozen=True, slots=True)
class Corpus:
    """The kept lines and the counts of what the rules dropped.

    Each drop count is of its own rule, in the order the rules run.

    Attributes:
        lines: Kept lines, cleaned, in source order.
        line_chars_min: The shortest cleaned line kept, from the `text.line_chars_min` gate.
        earlier_versions: The root's earlier text versions, sorted.
        held_out: Lines dropped for overlapping an ear-test line.
        dropped_earlier: Lines dropped because an earlier version holds their id or text.
        dropped_length: Lines dropped for a cleaned length outside `line_chars_min`-`MAX_CHARS`.
        dropped_duplicates: Lines dropped as a normalised duplicate of an earlier kept line.
        sources: Kept lines per source; a missing source counts 0.
        missing: The sources that were absent.
    """

    lines: tuple[SourceLine, ...]
    line_chars_min: int
    earlier_versions: tuple[str, ...]
    held_out: int
    dropped_earlier: int
    dropped_length: int
    dropped_duplicates: int
    sources: Mapping[str, int]
    missing: tuple[str, ...]

    def rows(self) -> list[dict[str, str]]:
        """Return the `lines.jsonl` rows, the shape `packs.draft_text` reads."""
        return [
            {"line_id": line.line_id, "state": NEUTRAL_STATE, "text": line.text}
            for line in self.lines
        ]

    def summary(self) -> dict[str, object]:
        """Return `summary.json`: counts only, never a line's text or a source path."""
        return {
            "lines": len(self.lines),
            "chars": sum(len(line.text) for line in self.lines),
            "line_chars_min": self.line_chars_min,
            "line_chars_max": MAX_CHARS,
            "earlier_versions": list(self.earlier_versions),
            "held_out": self.held_out,
            "dropped_earlier": self.dropped_earlier,
            "dropped_length": self.dropped_length,
            "dropped_duplicates": self.dropped_duplicates,
            "sources": dict(self.sources),
        }

    def report_lines(self) -> list[str]:
        """Return the printed summary: counts, version names and source names only."""
        summary = self.summary()
        out = [
            f"text: {summary['lines']} lines, {summary['chars']} chars",
            f"earlier versions: {', '.join(self.earlier_versions) or 'none'}",
            f"held out (ear-test overlap): {self.held_out}",
            f"dropped: {self.dropped_earlier} in an earlier version, {self.dropped_length} "
            f"outside {self.line_chars_min}-{MAX_CHARS} chars, {self.dropped_duplicates} "
            "duplicates",
        ]
        for name in SOURCE_ORDER:
            state = "missing, 0 lines" if name in self.missing else f"{self.sources[name]} lines"
            out.append(f"source {name}: {state}")
        return out


def clean(text: str) -> str:
    """Return the text a teacher reads: non-speech tags removed, whitespace collapsed."""
    return _SPACE.sub(" ", _TAG.sub(" ", text)).strip()


def normalise(text: str) -> str:
    """Return the comparison form: case, punctuation and whitespace ignored.

    Apostrophes are removed rather than spaced, so "it's" and "its" compare equal and a
    contraction stays one word.
    """
    folded = unicodedata.normalize("NFKC", text).casefold()
    return _SPACE.sub(" ", _NON_WORD.sub(" ", _APOSTROPHE.sub("", folded))).strip()


def _words(text: str) -> tuple[str, ...]:
    return tuple(normalise(clean(text)).split())


def _grams(words: Sequence[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


class HoldOut:
    """Lines no training line may equal or share an `n`-word run with, after normalisation."""

    def __init__(self, lines: Iterable[str], n: int = NGRAM) -> None:
        self._n = n
        self._exact: set[tuple[str, ...]] = set()
        self._grams: set[tuple[str, ...]] = set()
        for line in lines:
            words = _words(line)
            if words:
                self._exact.add(words)
                self._grams |= _grams(words, n)

    def holds(self, text: str) -> bool:
        """Return whether `text` equals a held-out line or shares an `n`-word run with one."""
        words = _words(text)
        return words in self._exact or not self._grams.isdisjoint(_grams(words, self._n))


def load_ear_test(path: Path = EAR_TEST_PATH) -> list[str]:
    """Return every ear-test text: each entry's `text`, and its `turbo_text` where it has one.

    The ear test plays the file's `lines` and its `acknowledgements` (20 and 6, the 26 the
    `voice_ab.lines` gate counts), so both are held out. An acknowledgement is shorter than
    `NGRAM` words, so it holds out only a line that equals it.

    Raises:
        InputError: The file is missing or not JSON, it has no `lines`, a list is malformed, or
            an entry has no text.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputError("refused: the ear-test lines file is missing or not JSON") from exc
    if not isinstance(data, dict) or not isinstance(data.get("lines"), list) or not data["lines"]:
        raise InputError("refused: the ear-test lines file has no lines")
    acknowledgements = data.get("acknowledgements", [])
    if not isinstance(acknowledgements, list):
        raise InputError("refused: the ear-test acknowledgements are not a list")
    texts: list[str] = []
    for line in [*data["lines"], *acknowledgements]:
        if not isinstance(line, dict) or not isinstance(line.get("text"), str):
            raise InputError("refused: an ear-test line has no text")
        texts.append(line["text"])
        if isinstance(line.get("turbo_text"), str):
            texts.append(line["turbo_text"])
    return texts


def _read_text(path: Path, what: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise InputError(f"refused: a {what} is unreadable") from exc


def _source_id(source: str, raw: str) -> str:
    return f"{source}-{_ID_UNSAFE.sub('_', raw)}"


def read_librittsr(sources: Path) -> list[SourceLine] | None:
    """Return LibriTTS-R test-clean's utterances in path order, or None when absent.

    Raises:
        InputError: A transcript is unreadable.
    """
    base = sources.joinpath(*LIBRITTSR_DIR)
    if not base.is_dir():
        return None
    return [
        SourceLine(
            source=LIBRITTSR,
            line_id=_source_id(LIBRITTSR, path.name[: -len(LIBRITTSR_SUFFIX)]),
            text=_read_text(path, "LibriTTS-R transcript"),
        )
        for path in sorted(base.glob(f"*/*/*{LIBRITTSR_SUFFIX}"))
    ]


def read_expresso(sources: Path) -> list[SourceLine] | None:
    """Return Expresso's read transcriptions in file order, or None when absent.

    Raises:
        InputError: The file is unreadable, or a non-blank line has no tab.
    """
    path = sources.joinpath(*EXPRESSO_FILE)
    if not path.is_file():
        return None
    out: list[SourceLine] = []
    for number, raw in enumerate(_read_text(path, "Expresso transcript file").splitlines(), 1):
        if not raw.strip():
            continue
        utt, sep, text = raw.partition("\t")
        if not sep or not utt.strip():
            raise InputError(f"refused: Expresso transcript line {number} has no id and tab")
        out.append(SourceLine(EXPRESSO, _source_id(EXPRESSO, utt.strip()), text))
    return out


def read_assistant(path: Path = ASSISTANT_LINES_PATH) -> list[SourceLine]:
    """Return the invented assistant-register lines, one JSON object per line.

    Raises:
        InputError: The file is unreadable, or a line is not an object with string `line_id`
            and `text`.
    """
    out: list[SourceLine] = []
    for number, raw in enumerate(_read_text(path, "assistant lines file").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise InputError(f"refused: assistant lines line {number} is not JSON") from exc
        if not (
            isinstance(row, dict)
            and isinstance(row.get("line_id"), str)
            and isinstance(row.get("text"), str)
        ):
            raise InputError(f"refused: assistant lines line {number} lacks a line_id or text")
        out.append(SourceLine(ASSISTANT, _source_id(ASSISTANT, row["line_id"]), row["text"]))
    return out


def line_chars_gate(gates: Gates) -> int:
    """Return the `text.line_chars_min` gate: the shortest cleaned line a build keeps.

    Raises:
        InputError: The value is not a whole number from `MIN_CHARS` to `MAX_CHARS`.
    """
    value = gates.value(LINE_CHARS_GATE)
    if float(value).is_integer() and MIN_CHARS <= value <= MAX_CHARS:
        return int(value)
    raise InputError(
        f"refused: gate {LINE_CHARS_GATE} must be a whole number from {MIN_CHARS} to {MAX_CHARS}"
    )


@dataclass(frozen=True, slots=True)
class EarlierVersions:
    """What the root's earlier text versions hold: their names, line ids and normalised texts."""

    versions: tuple[str, ...]
    line_ids: frozenset[str]
    texts: frozenset[str]

    def holds(self, line_id: str, text: str) -> bool:
        """Return whether an earlier version holds this line id or this text, normalised."""
        return line_id in self.line_ids or normalise(text) in self.texts


NO_EARLIER = EarlierVersions(versions=(), line_ids=frozenset(), texts=frozenset())


def read_earlier(root_path: Path) -> EarlierVersions:
    """Return what every folder under the root's `text/` holds, each read from its lines file.

    A folder without a readable lines file is refused rather than skipped: skipping it would let
    a new version repeat the lines a half-written or damaged version once held.

    Raises:
        InputError: A folder has no readable `lines.jsonl`, or a row of one is not an object with
            a string `line_id` and `text`. Errors name the folder and a row number, never text.
    """
    text_dir = root_path / TEXT_DIR
    if not text_dir.is_dir():
        return NO_EARLIER
    versions = sorted(entry.name for entry in text_dir.iterdir() if entry.is_dir())
    line_ids: set[str] = set()
    texts: set[str] = set()
    for version in versions:
        rel = f"{TEXT_DIR}/{version}"
        try:
            rows = (text_dir / version / LINES_NAME).read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            raise InputError(f"refused: {rel} has no readable {LINES_NAME}") from exc
        for number, raw in enumerate(rows, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise InputError(f"refused: {rel}/{LINES_NAME} row {number} is not JSON") from exc
            if not (
                isinstance(row, dict)
                and isinstance(row.get("line_id"), str)
                and isinstance(row.get("text"), str)
            ):
                raise InputError(
                    f"refused: {rel}/{LINES_NAME} row {number} lacks a line_id or text"
                )
            line_ids.add(row["line_id"])
            texts.add(normalise(clean(row["text"])))
    return EarlierVersions(tuple(versions), frozenset(line_ids), frozenset(texts))


def build(
    sources: Mapping[str, Sequence[SourceLine] | None],
    hold_out: HoldOut,
    earlier: EarlierVersions,
    *,
    line_chars_min: int,
) -> Corpus:
    """Apply the rules in order: hold-out, earlier versions, length, then duplicates.

    The hold-out runs first so `held_out` counts every overlapping line, whatever its length or
    version; the earlier versions run before the length so `dropped_earlier` counts every line a
    version already holds.

    Args:
        sources: Each source's lines, or None for a missing source.
        hold_out: The ear-test lines.
        earlier: What the root's earlier text versions hold.
        line_chars_min: The shortest cleaned line kept, from `line_chars_gate`.

    Raises:
        InputError: Both public sources are missing, `line_chars_min` is outside
            `MIN_CHARS`-`MAX_CHARS`, or two kept lines share an id.
    """
    if not MIN_CHARS <= line_chars_min <= MAX_CHARS:
        raise InputError(f"refused: the shortest line must be {MIN_CHARS}-{MAX_CHARS} chars")
    if all(sources.get(name) is None for name in PUBLIC_SOURCES):
        raise InputError("refused: neither LibriTTS-R nor Expresso is under --sources")
    kept: list[SourceLine] = []
    counts = {name: 0 for name in SOURCE_ORDER}
    missing: list[str] = []
    seen_text: set[str] = set()
    seen_ids: set[str] = set()
    held = in_earlier = duplicates = length = 0
    for name in SOURCE_ORDER:
        lines = sources.get(name)
        if lines is None:
            missing.append(name)
            continue
        for line in lines:
            text = clean(line.text)
            if hold_out.holds(text):
                held += 1
                continue
            if earlier.holds(line.line_id, text):
                in_earlier += 1
                continue
            if not line_chars_min <= len(text) <= MAX_CHARS:
                length += 1
                continue
            key = normalise(text)
            if key in seen_text:
                duplicates += 1
                continue
            if line.line_id in seen_ids:
                raise InputError(f"refused: two {name} lines share one id")
            seen_text.add(key)
            seen_ids.add(line.line_id)
            kept.append(SourceLine(line.source, line.line_id, text))
            counts[name] += 1
    return Corpus(
        lines=tuple(kept),
        line_chars_min=line_chars_min,
        earlier_versions=earlier.versions,
        held_out=held,
        dropped_earlier=in_earlier,
        dropped_length=length,
        dropped_duplicates=duplicates,
        sources=counts,
        missing=tuple(missing),
    )


def read_sources(sources: Path, assistant_lines: Path) -> dict[str, list[SourceLine] | None]:
    """Read the three sources; a missing public source is None, a missing assistant file refused.

    Raises:
        InputError: A source is unreadable or malformed.
    """
    return {
        ASSISTANT: read_assistant(assistant_lines),
        LIBRITTSR: read_librittsr(sources),
        EXPRESSO: read_expresso(sources),
    }
