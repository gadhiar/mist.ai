"""Tests for `text build`: the training text the teacher speaks, with the ear-test lines held out.

The public sources are fake trees under the test's tmp directory, laid out as a rehearsal root's
`public/` holds them; no test reads a real rehearsal or private root.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

import pytest

from scripts.voice_data import cli, packs, text_corpus
from scripts.voice_data.errors import InputError
from scripts.voice_data.root import Root

REPO = Path(__file__).resolve().parents[3]
EAR_TEST = REPO / "scripts" / "model_bench" / "speech" / "lines.json"
COMMITTED = REPO / "scripts" / "voice_data" / "text_lines.jsonl"

# Capitalised words a committed assistant line may use mid-sentence: days, months, units, "I".
ALLOWED_CAPITALS = frozenset(
    {
        "I",
        "I'm",
        "I'll",
        "I've",
        "I'd",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
        "Celsius",
        "Fahrenheit",
    }
)


def _ear_texts() -> list[str]:
    """Return the 26 ear-test texts: the 20 lines first, then the 6 acknowledgements."""
    data = json.loads(EAR_TEST.read_text(encoding="utf-8"))
    return [line["text"] for line in [*data["lines"], *data["acknowledgements"]]]


def _librittsr(sources: Path, lines: dict[str, str]) -> None:
    for utt, text in lines.items():
        speaker, chapter = utt.split("_")[:2]
        folder = sources / "librittsr" / "LibriTTS_R" / "test-clean" / speaker / chapter
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{utt}.normalized.txt").write_text(text, encoding="utf-8")


def _expresso(sources: Path, lines: dict[str, str]) -> None:
    folder = sources / "expresso" / "expresso"
    folder.mkdir(parents=True, exist_ok=True)
    body = "".join(f"{utt}\t{text}\n" for utt, text in lines.items())
    (folder / "read_transcriptions.txt").write_text(body, encoding="utf-8")


def _assistant(path: Path, lines: dict[str, str]) -> Path:
    body = "".join(json.dumps({"line_id": i, "text": t}) + "\n" for i, t in lines.items())
    path.write_text(body, encoding="utf-8")
    return path


def _gate(root: Root, value: object) -> None:
    """Set the root's `text.line_chars_min` overlay.

    A fixture whose lines are 20-79 chars sets v1's 20, so the test keeps the meaning it had
    before the gate existed.
    """
    path = root.path / "config" / "gates.json"
    path.write_text(json.dumps({"text.line_chars_min": value}), encoding="utf-8")


def _text(n: int, tag: str) -> str:
    """Return a line of exactly `n` chars whose normalised form is distinct per `tag` and `n`."""
    text = (f"Line {tag} reads " + "x" * n)[:n]
    assert len(text) == n
    return text


def _earlier(root: Root, version: str, lines: dict[str, str]) -> None:
    """Plant an earlier text version's `lines.jsonl` in the root, as `text build` writes it."""
    folder = root.path / "text" / version
    folder.mkdir(parents=True)
    rows = [{"line_id": i, "state": "neutral", "text": t} for i, t in lines.items()]
    body = "".join(json.dumps(row) + "\n" for row in rows)
    (folder / "lines.jsonl").write_text(body, encoding="utf-8")


def _build(root: Root, sources: Path, *extra: str, version: str = "v1") -> int:
    argv = ["--root", str(root.path), "text", "build", "--version", version]
    return cli.main([*argv, "--sources", str(sources), *extra])


def _rows(root: Root, version: str = "v1") -> list[dict]:
    text = (root.path / "text" / version / "lines.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _summary(root: Root, version: str = "v1") -> dict:
    return json.loads((root.path / "text" / version / "summary.json").read_text(encoding="utf-8"))


# --- normalisation and the hold-out -----------------------------------------------------------


def test_normalise_ignores_case_punctuation_and_whitespace() -> None:
    assert text_corpus.normalise("  It's   TEN, past-six!  ") == "its ten past six"
    assert text_corpus.normalise("It’s ten.") == text_corpus.normalise("its TEN")


def test_clean_strips_tags_and_collapses_whitespace() -> None:
    assert text_corpus.clean(" Well [laugh]  that <breath> went\n fine. ") == (
        "Well that went fine."
    )


def test_hold_out_matches_an_equal_line_after_normalisation() -> None:
    hold = text_corpus.HoldOut(["Your timer is set for ten minutes."])
    assert hold.holds("your TIMER is set -- for ten minutes!!")
    assert not hold.holds("Your timer is set for nine minutes.")


def test_hold_out_matches_a_shared_eight_word_sequence() -> None:
    hold = text_corpus.HoldOut(["one two three four five six seven eight nine ten"])
    assert hold.holds("Before that, Three four five six seven eight nine ten, and after.")
    assert not hold.holds("Before that, three four five six seven eight nine, and after.")


def test_hold_out_reads_every_ear_test_line() -> None:
    held = text_corpus.load_ear_test()
    assert len(held) >= len(_ear_texts()) == 26
    hold = text_corpus.HoldOut(held)
    assert all(hold.holds(text) for text in _ear_texts())


def test_hold_out_includes_the_acknowledgements(tmp_path: Path) -> None:
    path = tmp_path / "lines.json"
    entries = {
        "lines": [{"text": "A line of the ear test."}],
        "acknowledgements": [{"text": "On it."}],
    }
    path.write_text(json.dumps(entries), encoding="utf-8")
    hold = text_corpus.HoldOut(text_corpus.load_ear_test(path))
    assert hold.holds("on it!")
    assert not hold.holds("On it, then.")


def test_malformed_acknowledgements_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "lines.json"
    path.write_text(json.dumps({"lines": [{"text": "A line."}], "acknowledgements": {}}), "utf-8")
    with pytest.raises(InputError):
        text_corpus.load_ear_test(path)


# --- the command, on planted sources ----------------------------------------------------------


@pytest.fixture
def planted(tmp_path: Path, canaries, private_root: Root) -> Path:
    """A fake `public/` folder whose lines exercise every rule once, plus a canary line.

    Its kept lines are 20-79 chars, so the root's gate overlay sets v1's 20.
    """
    _gate(private_root, 20)
    ear = _ear_texts()
    first8 = " ".join(ear[1].split()[:8])
    first7 = " ".join(ear[1].split()[:7])
    sources = tmp_path / "public"
    _librittsr(
        sources,
        {
            "1089_134686_000001_000001": "The lamp stood on the table beside the open window.",
            "1089_134686_000002_000000": ear[0].upper().replace(",", " ;"),
            "1089_134686_000003_000000": f"Somebody wrote that {first8} later on.",
            "1188_133604_000001_000000": f"Somebody wrote that {first7} quietly later.",
            "1188_133604_000002_000000": "Too short.",
            "1188_133604_000003_000000": "word " * 50,
            "1188_133604_000004_000000": f"{canaries.transcript}, said the reader.",
        },
    )
    _expresso(
        sources,
        {
            "ex01_default_00001": "The river ran [laugh] low through the dry valley that year.",
            "ex02_default_00001": "THE LAMP STOOD ON THE TABLE, BESIDE THE OPEN WINDOW!",
        },
    )
    return sources


def test_build_merges_sources_and_holds_out_the_ear_test_lines(
    private_root: Root,
    planted: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    assert_no_canary: Callable[[str], None],
) -> None:
    assistant = _assistant(
        tmp_path / "assistant.jsonl",
        {"a0001": "Your next reminder is at half past four this afternoon."},
    )
    assert _build(private_root, planted, "--assistant-lines", str(assistant)) == 0
    rows = _rows(private_root)
    texts = [row["text"] for row in rows]
    assert texts == [
        "Your next reminder is at half past four this afternoon.",
        "The lamp stood on the table beside the open window.",
        f"Somebody wrote that {' '.join(_ear_texts()[1].split()[:7])} quietly later.",
        texts[3],
        "The river ran low through the dry valley that year.",
    ]
    assert all(set(row) == {"line_id", "state", "text"} for row in rows)
    assert all(row["state"] == "neutral" for row in rows)
    assert len({row["line_id"] for row in rows}) == len(rows)
    assert _summary(private_root) == {
        "lines": 5,
        "chars": sum(len(t) for t in texts),
        "line_chars_min": 20,
        "line_chars_max": 200,
        "earlier_versions": [],
        "held_out": 2,
        "dropped_earlier": 0,
        "dropped_length": 2,
        "dropped_duplicates": 1,
        "sources": {"assistant": 1, "librittsr": 3, "expresso": 1},
    }
    out = capsys.readouterr()
    printed = out.out + out.err
    assert_no_canary(printed)
    assert str(planted) not in printed
    assert str(tmp_path) not in printed
    for text in texts:
        assert text not in printed


def test_no_built_line_overlaps_an_ear_test_line(
    private_root: Root, planted: Path, tmp_path: Path
) -> None:
    assistant = _assistant(tmp_path / "a.jsonl", {"a1": "Your alarm is set for seven tomorrow."})
    assert _build(private_root, planted, "--assistant-lines", str(assistant)) == 0
    ear = [text_corpus.normalise(t).split() for t in _ear_texts()]
    grams = {tuple(w[i : i + 8]) for w in ear for i in range(len(w) - 7)}
    for row in _rows(private_root):
        words = text_corpus.normalise(row["text"]).split()
        assert words not in ear
        assert not {tuple(words[i : i + 8]) for i in range(len(words) - 7)} & grams


def test_the_text_pack_drafts_from_the_built_lines(
    private_root: Root, planted: Path, tmp_path: Path
) -> None:
    assert _build(private_root, planted) == 0
    draft = packs.draft_text(private_root, "v1", seed=7, per_state=30)
    assert [group.id for group in draft.groups] == ["neutral"]
    assert len(draft.groups[0].items) == 30


def test_default_assistant_lines_are_the_committed_file(private_root: Root, planted: Path) -> None:
    assert _build(private_root, planted) == 0
    committed = [line for line in COMMITTED.read_text(encoding="utf-8").splitlines() if line]
    assert _summary(private_root)["sources"]["assistant"] == len(committed)


def test_a_missing_source_counts_zero_and_is_reported(
    private_root: Root, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _gate(private_root, 20)
    sources = tmp_path / "public"
    _expresso(sources, {"ex01_default_00001": "The river ran low through the dry valley."})
    assert _build(private_root, sources) == 0
    summary = _summary(private_root)
    assert summary["sources"]["librittsr"] == 0
    assert summary["sources"]["expresso"] == 1
    assert "librittsr: missing" in capsys.readouterr().out


def test_both_public_sources_missing_is_refused(private_root: Root, tmp_path: Path) -> None:
    sources = tmp_path / "public"
    sources.mkdir()
    assert _build(private_root, sources) == 2
    assert not (private_root.path / "text" / "v1").exists()


def test_sources_that_are_not_a_folder_are_refused(private_root: Root, tmp_path: Path) -> None:
    assert _build(private_root, tmp_path / "absent") == 2


def test_an_existing_version_is_refused(private_root: Root, planted: Path) -> None:
    assert _build(private_root, planted) == 0
    before = (private_root.path / "text" / "v1" / "lines.jsonl").read_bytes()
    assert _build(private_root, planted) == 2
    assert (private_root.path / "text" / "v1" / "lines.jsonl").read_bytes() == before


@pytest.mark.parametrize("version", ["..", "a/b", "../v1", ".hidden", ""])
def test_a_version_that_is_not_one_folder_name_is_refused(
    private_root: Root, planted: Path, version: str
) -> None:
    assert _build(private_root, planted, version=version) == 2


def test_a_malformed_assistant_file_is_refused(
    private_root: Root, planted: Path, tmp_path: Path
) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"line_id": "a1"}\n', encoding="utf-8")
    assert _build(private_root, planted, "--assistant-lines", str(bad)) == 2
    assert _build(private_root, planted, "--assistant-lines", str(tmp_path / "none.jsonl")) == 2


def test_an_expresso_line_without_a_tab_is_refused(private_root: Root, tmp_path: Path) -> None:
    folder = tmp_path / "public" / "expresso" / "expresso"
    folder.mkdir(parents=True)
    (folder / "read_transcriptions.txt").write_text("no tab on this line\n", encoding="utf-8")
    assert _build(private_root, tmp_path / "public") == 2


def test_without_a_root_the_build_is_refused(planted: Path, monkeypatch) -> None:
    monkeypatch.delenv("MIST_VOICE_PRIVATE_ROOT", raising=False)
    assert cli.main(["text", "build", "--version", "v1", "--sources", str(planted)]) == 2


# --- the length gate --------------------------------------------------------------------------


def test_the_default_gate_keeps_lines_of_80_to_200_chars_after_cleaning(
    private_root: Root, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sources = tmp_path / "public"
    _librittsr(
        sources,
        {
            "1089_134686_000001_000000": _text(79, "a"),
            "1089_134686_000002_000000": _text(80, "b"),
            "1089_134686_000003_000000": _text(200, "c"),
            "1089_134686_000004_000000": _text(201, "d"),
            # 86 raw chars, 75 once the tag is cleaned away: the gate reads the cleaned length.
            "1089_134686_000005_000000": _text(75, "e") + " [laugh]   ",
        },
    )
    assistant = _assistant(tmp_path / "a.jsonl", {"a1": _text(79, "f"), "a2": _text(120, "g")})
    assert _build(private_root, sources, "--assistant-lines", str(assistant)) == 0
    assert [len(row["text"]) for row in _rows(private_root)] == [120, 80, 200]
    summary = _summary(private_root)
    assert (summary["line_chars_min"], summary["line_chars_max"]) == (80, 200)
    assert summary["dropped_length"] == 4
    assert "4 outside 80-200 chars" in capsys.readouterr().out


def test_the_root_overlay_sets_the_shortest_kept_line(private_root: Root, tmp_path: Path) -> None:
    _gate(private_root, 50)
    sources = tmp_path / "public"
    _expresso(sources, {"ex01_default_00001": _text(49, "a"), "ex01_default_00002": _text(50, "b")})
    assistant = _assistant(tmp_path / "a.jsonl", {})
    assert _build(private_root, sources, "--assistant-lines", str(assistant)) == 0
    assert [row["text"] for row in _rows(private_root)] == [_text(50, "b")]
    assert _summary(private_root)["line_chars_min"] == 50
    assert _summary(private_root)["dropped_length"] == 1


@pytest.mark.parametrize("value", [19, 201, 80.5, -80])
def test_a_gate_that_is_not_a_whole_number_from_20_to_200_is_refused(
    private_root: Root, tmp_path: Path, value: float, capsys: pytest.CaptureFixture[str]
) -> None:
    _gate(private_root, value)
    sources = tmp_path / "public"
    _expresso(sources, {"ex01_default_00001": _text(100, "a")})
    assert _build(private_root, sources) == 2
    assert "text.line_chars_min" in capsys.readouterr().err
    assert not (private_root.path / "text" / "v1").exists()


@pytest.mark.parametrize("value", [20, 200])
def test_the_gate_bounds_are_accepted(private_root: Root, tmp_path: Path, value: int) -> None:
    _gate(private_root, value)
    sources = tmp_path / "public"
    _expresso(sources, {"ex01_default_00001": _text(200, "a")})
    assert _build(private_root, sources) == 0
    assert _summary(private_root)["line_chars_min"] == value


# --- earlier text versions --------------------------------------------------------------------


def test_lines_an_earlier_version_holds_are_dropped_by_id_and_by_text(
    private_root: Root, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before = _text(110, "kept, before")
    _earlier(
        private_root,
        "v0",
        {"librittsr-1089_134686_000001_000000": _text(100, "old"), "assistant-a7": before},
    )
    sources = tmp_path / "public"
    new = _text(120, "new")
    by_id = _text(100, "same id, new words")
    by_text = before.upper().replace(",", "")
    _librittsr(
        sources,
        {
            "1089_134686_000001_000000": by_id,
            "1089_134686_000002_000000": by_text,
            "1089_134686_000003_000000": new,
        },
    )
    assistant = _assistant(tmp_path / "a.jsonl", {})
    assert _build(private_root, sources, "--assistant-lines", str(assistant)) == 0
    assert [(row["line_id"], row["text"]) for row in _rows(private_root)] == [
        ("librittsr-1089_134686_000003_000000", new)
    ]
    summary = _summary(private_root)
    assert summary["earlier_versions"] == ["v0"]
    assert summary["dropped_earlier"] == 2
    printed = capsys.readouterr().out
    assert "earlier versions: v0" in printed
    assert "2 in an earlier version" in printed
    for text in (before, new, by_id, by_text):
        assert text not in printed


def test_a_second_build_of_the_same_sources_keeps_nothing_new(
    private_root: Root, tmp_path: Path
) -> None:
    sources = tmp_path / "public"
    _librittsr(sources, {"1089_134686_000001_000000": _text(90, "a")})
    _expresso(sources, {"ex01_default_00001": _text(95, "b")})
    assistant = _assistant(tmp_path / "a.jsonl", {"a1": _text(85, "c")})
    assert _build(private_root, sources, "--assistant-lines", str(assistant)) == 0
    assert _build(private_root, sources, "--assistant-lines", str(assistant), version="v2") == 0
    summary = _summary(private_root, "v2")
    assert _rows(private_root, "v2") == []
    assert summary["earlier_versions"] == ["v1"]
    assert summary["dropped_earlier"] == 3


def test_earlier_versions_are_every_text_folder_sorted(private_root: Root, tmp_path: Path) -> None:
    _earlier(private_root, "v2", {"assistant-a1": _text(90, "a")})
    _earlier(private_root, "v10", {"assistant-a2": _text(90, "b")})
    (private_root.path / "text" / "notes.txt").write_text("not a version", encoding="utf-8")
    sources = tmp_path / "public"
    _expresso(sources, {"ex01_default_00001": _text(90, "c")})
    assert _build(private_root, sources, version="v3") == 0
    assert _summary(private_root, "v3")["earlier_versions"] == ["v10", "v2"]


@pytest.mark.parametrize(
    "body",
    [None, b"not json\n", b'{"line_id": "assistant-a1"}\n', b"\xff\xfe\n", b"[1, 2]\n"],
    ids=["missing", "not-json", "no-text", "not-utf8", "not-an-object"],
)
def test_a_text_folder_without_readable_lines_is_refused(
    private_root: Root, tmp_path: Path, body: bytes | None, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = private_root.path / "text" / "v0"
    folder.mkdir()
    if body is not None:
        (folder / "lines.jsonl").write_bytes(body)
    sources = tmp_path / "public"
    _expresso(sources, {"ex01_default_00001": _text(90, "a")})
    assert _build(private_root, sources) == 2
    assert "text/v0" in capsys.readouterr().err
    assert not (private_root.path / "text" / "v1").exists()


def test_the_rules_run_in_order_and_each_drop_counts_once(
    private_root: Root, tmp_path: Path
) -> None:
    ear = _ear_texts()[0]
    long = _text(150, "kept")
    comma = "Line p, reads " + "x" * 66
    assert len(comma) == 80
    _earlier(
        private_root,
        "v0",
        {
            "librittsr-1089_134686_000001_000000": _text(90, "old one"),
            "librittsr-1089_134686_000002_000000": _text(90, "old two"),
        },
    )
    sources = tmp_path / "public"
    _librittsr(
        sources,
        {
            # Held out and in an earlier version: the hold-out runs first.
            "1089_134686_000001_000000": ear,
            # In an earlier version and too short: the earlier versions run before the length.
            "1089_134686_000002_000000": _text(30, "short, earlier"),
            "1089_134686_000003_000000": _text(30, "short"),
            "1089_134686_000004_000000": long,
            "1089_134686_000005_000000": comma,
            "1089_134686_000006_000000": long.upper(),
            # A duplicate of a kept line, 79 chars without the comma: the length runs first.
            "1089_134686_000007_000000": comma.replace(",", ""),
        },
    )
    assistant = _assistant(tmp_path / "a.jsonl", {})
    assert _build(private_root, sources, "--assistant-lines", str(assistant)) == 0
    assert [row["text"] for row in _rows(private_root)] == [long, comma]
    summary = _summary(private_root)
    drops = {k: v for k, v in summary.items() if k == "held_out" or k.startswith("dropped_")}
    assert list(drops.items()) == [
        ("held_out", 1),
        ("dropped_earlier", 1),
        ("dropped_length", 2),
        ("dropped_duplicates", 1),
    ]


# --- the committed assistant-register lines ---------------------------------------------------


def _committed_rows() -> list[dict]:
    lines = COMMITTED.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def capitalised_mid_sentence(text: str) -> list[str]:
    """Return the capitalised words that neither start a sentence nor are allowed."""
    found: list[str] = []
    start = True
    for token in text.split():
        word = token.strip("\"'(),;:.?!")
        if word and word[0].isupper() and not start and word not in ALLOWED_CAPITALS:
            found.append(word)
        start = token.rstrip("\"')").endswith((".", "?", "!"))
    return found


def test_capital_check_flags_a_name_and_allows_the_list() -> None:
    assert capitalised_mid_sentence("Call home at noon. Then rest.") == []
    assert capitalised_mid_sentence("I will call Xyzzy at noon.") == ["Xyzzy"]
    assert capitalised_mid_sentence("On Monday I'll check. It's in May, at ten Celsius.") == []


def test_committed_lines_have_ids_and_text() -> None:
    rows = _committed_rows()
    assert 950 <= len(rows) <= 1100
    assert all(set(row) == {"line_id", "text"} for row in rows)
    assert all(re.fullmatch(r"a\d{4}", row["line_id"]) for row in rows)
    assert len({row["line_id"] for row in rows}) == len(rows)


def test_committed_lines_capitalise_only_sentence_starts_and_the_allowlist() -> None:
    flagged = {row["line_id"]: capitalised_mid_sentence(row["text"]) for row in _committed_rows()}
    assert {k: v for k, v in flagged.items() if v} == {}


def test_committed_lines_pass_every_build_rule() -> None:
    hold = text_corpus.HoldOut(text_corpus.load_ear_test())
    seen: set[str] = set()
    for row in _committed_rows():
        text = row["text"]
        assert text_corpus.clean(text) == text, row["line_id"]
        assert text_corpus.MIN_CHARS <= len(text) <= text_corpus.MAX_CHARS, row["line_id"]
        assert not hold.holds(text), row["line_id"]
        key = text_corpus.normalise(text)
        assert key not in seen, row["line_id"]
        seen.add(key)
