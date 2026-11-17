"""Tests for `rehearsal anchors`: one timestamp row per non-neutral state, from the ground truth.

The root is a rehearsal root with a hand-written `rehearsal/truth.jsonl` and `seed/inventory.jsonl`
for four episodes; no build job runs, since the action reads only those two files and the gates.
The truth is laid out so that each part of the row rule decides one row:

- calm: ep01, its own episode, holds two qualifying calm lines; the first in file order wins.
- awake: ep01 holds a qualifying awake line, but the search starts at ep02, where an off-centre,
  an overlapped, a too-short and a too-long line come before the one that qualifies.
- sleepy: its own episode, ep03, has none; the search moves on to ep04, not back to ep01.
- excited: its own episode, ep04, has none; the search wraps to ep01 (none) and then ep02.
- neutral: ep01 holds a qualifying neutral line, which gets no row; the solo clip covers it.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from scripts.voice_data import cli, gates
from scripts.voice_data import rehearsal as pkg
from scripts.voice_data import root as root_mod
from scripts.voice_data.commands import extract as extract_cmd
from scripts.voice_data.commands import rehearsal as rehearsal_cmd
from scripts.voice_data.errors import InputError
from scripts.voice_data.records import SourceRecord, read_jsonl, write_jsonl

from .conftest import Canaries

EPISODES = ("ep01", "ep02", "ep03", "ep04")
SHA = {ep: str(n) * 64 for n, ep in enumerate(EPISODES, start=1)}
INVENTORY_REL = "seed/inventory.jsonl"


def line(
    episode: str,
    n: int,
    state: str | None,
    start: float,
    end: float,
    geometry: str | None = "centre",
    overlap: bool = False,
) -> pkg.TruthLine:
    """A truth line; a line with no state is another speaker's."""
    target = state is not None
    return pkg.TruthLine(
        episode=episode,
        episode_sha256=SHA[episode],
        line=n,
        speaker="target" if target else "other",
        speaker_id="" if target else "121",
        start_s=start,
        end_s=end,
        state=state,
        style="default" if target else None,
        geometry=geometry if target else None,
        clip=f"expresso/clip-{episode}-{n}.wav",
        overlap=overlap,
    )


TRUTH = [
    line("ep01", 1, "neutral", 1.0, 6.0),
    line("ep01", 2, "awake", 7.0, 12.0),
    line("ep01", 3, None, 12.5, 14.0),
    line("ep01", 4, "calm", 14.5, 19.75),  # 5.25 s: the calm row
    line("ep01", 5, "calm", 20.0, 27.5),  # 7.5 s: the calm row once anchors.min_s is 6
    line("ep01", 6, "sleepy", 28.0, 33.0),
    line("ep02", 1, "awake", 1.0, 6.0, geometry="off-centre"),
    line("ep02", 2, "awake", 7.0, 12.0, overlap=True),
    line("ep02", 3, "awake", 13.0, 15.0),  # 2 s: shorter than anchors.min_s
    line("ep02", 4, "awake", 16.0, 56.0),  # 40 s: longer than anchors.max_s
    line("ep02", 5, "awake", 57.0, 61.5),  # the awake row
    line("ep02", 6, "excited", 62.0, 66.25),  # the excited row, after the wrap
    line("ep03", 1, "calm", 1.0, 5.0),
    line("ep03", 2, "excited", 6.0, 10.0),
    line("ep03", 3, "sleepy", 11.0, 13.0, geometry="fallback"),
    line("ep04", 1, "sleepy", 2.0, 8.125),  # the sleepy row
    line("ep04", 2, "neutral", 9.0, 14.0),
]

EXPECTED = [
    ("calm", "src-0001", "rehearsal/ep01.mkv", 14.5, 19.75),
    ("awake", "src-0002", "rehearsal/ep02.mkv", 57.0, 61.5),
    ("sleepy", "src-0004", "rehearsal/ep04.mkv", 2.0, 8.125),
    ("excited", "src-0002", "rehearsal/ep02.mkv", 62.0, 66.25),
]


def record(n: int, episode: str) -> SourceRecord:
    return SourceRecord(
        f"src-{n:04d}", f"seed/raw/rehearsal/{episode}.mkv", "matroska", 1, 0, (), 1, 0, None,
        "eac3", None, 1, "packets", 48000, 6, "5.1", 60.0, False, 1, SHA[episode], "t", "FC",
    )  # fmt: skip


INVENTORY = [record(n, ep) for n, ep in enumerate(EPISODES, start=1)]


def run_cli(argv: Sequence[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


def write_truth(root: root_mod.Root, lines: Sequence[pkg.TruthLine]) -> None:
    text = "".join(json.dumps(ln.to_json()) + "\n" for ln in lines)
    root.write_bytes(pkg.TRUTH_REL, text.encode("utf-8"))


@pytest.fixture
def stand_in(tmp_path: Path) -> root_mod.Root:
    """A rehearsal root holding the truth and the inventory of the four episodes."""
    root = root_mod.init_root(tmp_path / "stand-in", "rehearsal")
    write_truth(root, TRUTH)
    write_jsonl(root, INVENTORY_REL, INVENTORY)
    return root


def anchors(root: root_mod.Root) -> Callable[[], tuple[int, str, str]]:
    return lambda: run_cli(["--root", str(root.path), "rehearsal", "anchors"])


# ---------------------------------------------------------------------------
# The rows
# ---------------------------------------------------------------------------


def test_writes_one_row_per_non_neutral_state_in_order(stand_in: root_mod.Root) -> None:
    code, out, err = anchors(stand_in)()
    assert code == 0, err
    text = (stand_in.path / extract_cmd.ANCHORS_REL).read_text(encoding="utf-8")
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == ["source", "start_s", "end_s"]
    assert [(r[0], float(r[1]), float(r[2])) for r in rows[1:]] == [e[2:] for e in EXPECTED]


def test_prints_one_line_per_row_and_no_file_name(stand_in: root_mod.Root) -> None:
    code, out, err = anchors(stand_in)()
    assert code == 0, err
    assert out.splitlines() == [
        f"time-{n} {state} {sid}" for n, (state, sid, *_) in enumerate(EXPECTED, start=1)
    ]
    for name in ("ep01", "ep02", "ep04", ".mkv", "rehearsal/", "anchors.csv", "expresso"):
        assert name not in out + err


def test_the_written_file_round_trips_through_read_anchors(stand_in: root_mod.Root) -> None:
    assert anchors(stand_in)()[0] == 0
    g = gates.load(stand_in.path)
    read = extract_cmd.read_anchors(
        stand_in, read_jsonl(stand_in.path / INVENTORY_REL, SourceRecord),
        g.value("anchors.min_s"), g.value("anchors.max_s"),
    )  # fmt: skip
    assert read == [{"source_id": e[1], "start_s": e[3], "end_s": e[4]} for e in EXPECTED]


def test_anchor_rows_returns_line_record_and_source() -> None:
    rows = rehearsal_cmd.anchor_rows(TRUTH, INVENTORY, 3.0, 30.0)
    got = [(ln.state, rec.source_id, src, ln.start_s, ln.end_s) for ln, rec, src in rows]
    assert got == EXPECTED
    for ln, rec, _ in rows:
        assert ln.speaker == "target" and rec.sha256 == ln.episode_sha256


def test_the_length_gates_in_force_decide(stand_in: root_mod.Root) -> None:
    # The root's overlay lowers anchors.max_s to 5 s. Both calm lines of ep01 are now too long,
    # so calm moves on to ep03; ep04's 6.125 s sleepy line is too long, so sleepy wraps to ep01.
    stand_in.write_bytes(gates.OVERLAY_REL, b'{"anchors.max_s": 5}')
    code, out, err = anchors(stand_in)()
    assert code == 0, err
    text = (stand_in.path / extract_cmd.ANCHORS_REL).read_text(encoding="utf-8")
    assert text.splitlines()[1:] == [
        "rehearsal/ep03.mkv,1.0,5.0",
        "rehearsal/ep02.mkv,57.0,61.5",
        "rehearsal/ep01.mkv,28.0,33.0",
        "rehearsal/ep02.mkv,62.0,66.25",
    ]
    assert out.splitlines()[0] == "time-1 calm src-0003"
    read = extract_cmd.read_anchors(
        stand_in, read_jsonl(stand_in.path / INVENTORY_REL, SourceRecord), 3.0, 5.0
    )
    assert [a["source_id"] for a in read] == ["src-0003", "src-0002", "src-0001", "src-0002"]


def test_anchor_rows_refuses_rather_than_returning_fewer_rows() -> None:
    with pytest.raises(InputError, match="sleepy"):
        rehearsal_cmd.anchor_rows([ln for ln in TRUTH if ln.state != "sleepy"], INVENTORY, 3, 30)


def test_excited_wraps_past_the_last_episode_before_ep02() -> None:
    # With ep02's excited line gone, the wrap reaches ep03's.
    truth = [ln for ln in TRUTH if not (ln.episode == "ep02" and ln.state == "excited")]
    rows = rehearsal_cmd.anchor_rows(truth, INVENTORY, 3.0, 30.0)
    assert (rows[3][0].episode, rows[3][0].line) == ("ep03", 2)


def test_fewer_episodes_than_states_wraps_the_start() -> None:
    # Three episodes: excited's own episode, ep04, does not exist; its search starts at ep01.
    truth = [ln for ln in TRUTH if ln.episode != "ep04"] + [line("ep01", 7, "excited", 40.0, 45.0)]
    truth = [replace(ln, state="sleepy") if (ln.episode, ln.line) == ("ep03", 1) else ln
             for ln in truth]  # fmt: skip
    rows = rehearsal_cmd.anchor_rows(truth, INVENTORY[:3], 3.0, 30.0)
    assert [(ln.episode, ln.line) for ln, _, _ in rows] == [
        ("ep01", 4),
        ("ep02", 5),
        ("ep03", 1),
        ("ep01", 7),
    ]


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_refused_unless_the_root_is_a_rehearsal_root(private_root: root_mod.Root) -> None:
    write_truth(private_root, TRUTH)
    write_jsonl(private_root, INVENTORY_REL, INVENTORY)
    code, out, err = anchors(private_root)()
    assert code == 2 and "kind rehearsal" in err and out == ""
    assert not (private_root.path / extract_cmd.ANCHORS_REL).exists()


def test_refused_when_anchors_csv_exists(stand_in: root_mod.Root) -> None:
    stand_in.write_bytes(extract_cmd.ANCHORS_REL, b"source,start_s,end_s\n")
    code, out, err = anchors(stand_in)()
    assert code == 2 and "exists" in err and out == ""
    assert (stand_in.path / extract_cmd.ANCHORS_REL).read_bytes() == b"source,start_s,end_s\n"


def test_refused_when_the_truth_is_missing(stand_in: root_mod.Root) -> None:
    (stand_in.path / pkg.TRUTH_REL).unlink()
    code, out, err = anchors(stand_in)()
    assert code == 2 and "truth.jsonl" in err and out == ""
    assert not (stand_in.path / extract_cmd.ANCHORS_REL).exists()


def test_refused_when_the_inventory_is_missing(stand_in: root_mod.Root) -> None:
    (stand_in.path / INVENTORY_REL).unlink()
    code, out, err = anchors(stand_in)()
    assert code == 2 and "inventory.jsonl" in err and out == ""
    assert not (stand_in.path / extract_cmd.ANCHORS_REL).exists()


def test_refused_when_an_episode_is_not_in_the_inventory(stand_in: root_mod.Root) -> None:
    # ep03 gives no row, yet its absence from the inventory still refuses.
    write_jsonl(stand_in, INVENTORY_REL, [r for r in INVENTORY if r.sha256 != SHA["ep03"]])
    code, out, err = anchors(stand_in)()
    assert code == 2 and "ep03" in err and "sha256" in err and out == ""
    assert not (stand_in.path / extract_cmd.ANCHORS_REL).exists()


@pytest.mark.parametrize("state", ["calm", "awake", "sleepy", "excited"])
def test_refused_when_a_state_has_no_qualifying_line(stand_in: root_mod.Root, state: str) -> None:
    # Every line of the state stays, but none qualifies: each is overlapped.
    write_truth(stand_in, [replace(ln, overlap=True) if ln.state == state else ln for ln in TRUTH])
    code, out, err = anchors(stand_in)()
    assert code == 2 and state in err and out == ""
    assert not (stand_in.path / extract_cmd.ANCHORS_REL).exists()


def test_no_canary_or_source_name_reaches_the_output(tmp_path: Path) -> None:
    root = root_mod.init_root(tmp_path / "stand-in", "rehearsal")
    write_truth(root, TRUTH)
    name = Canaries().source_name
    write_jsonl(root, INVENTORY_REL, [replace(r, path=f"seed/raw/{name}") for r in INVENTORY])
    code, out, err = anchors(root)()
    assert code == 0, err
    assert name not in out + err
    text = (root.path / extract_cmd.ANCHORS_REL).read_text(encoding="utf-8")
    assert text.splitlines()[1].startswith(f"{name},")


def test_the_module_docstring_lists_the_action() -> None:
    doc = rehearsal_cmd.__doc__ or ""
    assert "anchors" in doc.splitlines()[0] and "- `anchors`" in doc
