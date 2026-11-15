"""Tests for the blind listening packs (T9): six kinds, the sealed key and `pack import`.

Records the packs read that later tasks produce (bake-off takes, `gains.json`, spot-check takes,
the ABX manifest) are built here from the Interfaces shapes with synthetic data. Clips are small
synthetic FLAC files: a real STREAMINFO header and an opaque payload unique to each clip, so the
byte-identity check is meaningful and no audio processing is needed.
"""

from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from scripts.voice_data import cli, packs
from scripts.voice_data import root as root_mod
from scripts.voice_data.errors import InputError
from scripts.voice_data.states import SPEAKING_STATES

SEED = 7


# --- synthetic inputs --------------------------------------------------------------------------


def flac_bytes(
    payload: bytes,
    *,
    rate: int = 24000,
    channels: int = 1,
    bits: int = 16,
    comments: tuple[str, ...] | None = None,
    picture: bool = False,
) -> bytes:
    """A FLAC stream header (STREAMINFO, optional VORBIS_COMMENT and PICTURE) and a payload."""
    info = bytearray(34)
    info[0:2] = (4096).to_bytes(2, "big")
    info[2:4] = (4096).to_bytes(2, "big")
    packed = (rate << 44) | ((channels - 1) << 41) | ((bits - 1) << 36) | 24000
    info[10:18] = packed.to_bytes(8, "big")
    blocks: list[tuple[int, bytes]] = [(0, bytes(info))]
    if comments is not None:
        vendor = b"synthetic"
        body = len(vendor).to_bytes(4, "little") + vendor + len(comments).to_bytes(4, "little")
        for comment in comments:
            raw = comment.encode("utf-8")
            body += len(raw).to_bytes(4, "little") + raw
        blocks.append((4, body))
    if picture:
        blocks.append((6, b"\x00" * 32))
    out = bytearray(b"fLaC")
    for index, (kind, body) in enumerate(blocks):
        last = 0x80 if index == len(blocks) - 1 else 0
        out.append(last | kind)
        out += len(body).to_bytes(3, "big")
        out += body
    return bytes(out) + b"\xff\xf8" + payload


def write_clip(root: root_mod.Root, rel: str, **kwargs: object) -> Path:
    path = root.path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(flac_bytes(rel.encode("utf-8") * 3, **kwargs))  # type: ignore[arg-type]
    return path


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def segment(
    number: int,
    *,
    run: str = "r1",
    source: str = "src-0001",
    diar: str | None = "spk-01",
    cluster: str = "primary",
    flag_group: str | None = None,
    state: str = "calm",
    kept: bool = True,
    transcript: str = "a plain line of speech",
) -> dict:
    sid = f"seg-{number:06d}"
    return {
        "segment_id": sid,
        "source_id": source,
        "start_s": 1.0,
        "end_s": 5.0,
        "channel": "FC",
        "channel_reason": None,
        "separator": "none",
        "audio": f"seed/runs/{run}/audio/{sid}.flac",
        "diar_cluster": diar,
        "cluster": cluster,
        "flag_group": flag_group,
        "sim": {"wavlm-sv": 0.8, "ecapa": 0.7},
        "overlap_ms": 0,
        "snr_db": 20.0,
        "c50_db": 30.0,
        "transcripts": {"whisperx": transcript, "crisper": transcript},
        "asr_agreement_cer": 0.0,
        "prosody": {"f0_mean_hz": 200.0, "f0_std_hz": 20.0, "rate_sps": 4.0, "energy_db": -20.0},
        "arousal": 0.4,
        "state_proposed": state,
        "state": None,
        "set": "short",
        "kept": kept,
        "reasons": [],
    }


def write_run(root: root_mod.Root, run: str, segments: list[dict]) -> None:
    write_jsonl(root.path / "seed" / "runs" / run / "segments.jsonl", segments)
    for seg in segments:
        write_clip(root, seg["audio"])


def take(
    line_id: str,
    state: str,
    teacher: str,
    seed: int,
    *,
    base: str,
    kept: bool = True,
    selected: bool = False,
) -> dict:
    take_id = root_mod.take_id(line_id, state, teacher, seed)
    return {
        "take_id": take_id,
        "line_id": line_id,
        "state": state,
        "teacher": teacher,
        "seed": seed,
        "round": 1,
        "audio": f"{base}/{teacher}/{take_id}.flac",
        "sim": {"wavlm-sv": 0.8, "ecapa": 0.7},
        "cer": 0.01,
        "prosody": {"f0_mean_hz": 200.0, "f0_std_hz": 20.0, "rate_sps": 4.0, "energy_db": -20.0},
        "arousal": 0.4,
        "kept": kept,
        "selected": selected,
        "reasons": [],
    }


def write_takes(root: root_mod.Root, rel: str, takes: list[dict]) -> None:
    write_jsonl(root.path / rel, takes)
    for row in takes:
        write_clip(root, row["audio"])


def run_cli(root: root_mod.Root, *args: str) -> int:
    return cli.main(["--root", str(root.path), *args])


def pack_dir(root: root_mod.Root, seed: int = SEED) -> Path:
    return root.path / "packs" / packs.pack_id(seed)


def page_of(root: root_mod.Root, seed: int = SEED) -> dict:
    return packs.page_data((pack_dir(root, seed) / "index.html").read_text(encoding="utf-8"))


def key_of(root: root_mod.Root, seed: int = SEED) -> dict:
    path = root.path / "packs" / f"{packs.pack_id(seed)}.key" / "key.json"
    return json.loads(path.read_text(encoding="utf-8"))


def items_of(data: dict) -> list[tuple[dict, dict]]:
    return [(group, item) for group in data["groups"] for item in group["items"]]


def answer(data: dict, fill: Callable[[dict, dict], dict[str, object]]) -> str:
    """The CSV the page exports, with each item's values from `fill(group, item)`."""
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(data["columns"])
    for group, item in items_of(data):
        values: dict[str, object] = {"label": item["label"], "line_id": group["id"]}
        values.update(fill(group, item))
        writer.writerow([values.get(column, "") for column in data["columns"]])
    return out.getvalue()


def import_ratings(root: root_mod.Root, text: str, tmp_path: Path, seed: int = SEED) -> int:
    ratings = tmp_path / "downloads" / "ratings.csv"
    ratings.parent.mkdir(parents=True, exist_ok=True)
    ratings.write_text(text, encoding="utf-8")
    return run_cli(root, "pack", "import", "--pack", packs.pack_id(seed), "--ratings", str(ratings))


# --- the copies of listening.py's page helpers -------------------------------------------------


def test_copies_match_listening_originals() -> None:
    pytest.importorskip("yaml")
    from scripts.model_bench.speech import listening

    for seed in (0, 1, 7, 2**31 - 1):
        assert packs.pack_id(seed) == listening.pack_id(seed)
    assert packs.RATINGS_COLUMNS == listening.RATINGS_COLUMNS
    assert packs.PLACEHOLDER == listening.PLACEHOLDER
    block = f'<script id="pack-data" type="application/json">{packs.PLACEHOLDER}</script>'
    template = f"<p>x</p>{block}"
    data = {"pack": "p", "groups": [{"text": "a </script> b", "n": [1, 2]}]}
    ours = packs.render_page(template, data)
    assert ours == listening.render_page(template, data)
    assert packs.page_data(ours) == listening.page_data(ours) == data


def test_the_cli_with_pack_loads_without_pyyaml() -> None:
    code = (
        "import sys; sys.modules['yaml'] = None; "
        "from scripts.voice_data import cli; cli.build_parser(); "
        "assert 'scripts.model_bench.speech.listening' not in sys.modules"
    )
    repo = Path(__file__).resolve().parents[3]
    done = subprocess.run(
        [sys.executable, "-c", code], cwd=repo, capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr[-400:]


def test_page_data_without_a_block_is_an_input_error() -> None:
    with pytest.raises(InputError):
        packs.page_data("<html></html>")


def test_template_holds_the_placeholder_once() -> None:
    template = packs.load_template()
    assert template.count(packs.PLACEHOLDER) == 1
    assert 'id="pack-data"' in template


# --- FLAC headers -------------------------------------------------------------------------------


def test_flac_info_reads_a_real_flac_header(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    soundfile = pytest.importorskip("soundfile")
    path = tmp_path / "tone.flac"
    tone = (0.1 * np.sin(np.arange(2400) / 10.0)).astype("float32")
    soundfile.write(str(path), tone, 24000, subtype="PCM_16", format="FLAC")
    info = packs.flac_info(path)
    assert (info.sample_rate, info.channels, info.bits_per_sample) == (24000, 1, 16)
    assert info.tagged is False


def test_flac_info_sees_tags_and_pictures(tmp_path: Path) -> None:
    plain = tmp_path / "plain.flac"
    plain.write_bytes(flac_bytes(b"x", comments=()))
    tagged = tmp_path / "tagged.flac"
    tagged.write_bytes(flac_bytes(b"x", comments=("ENCODER=someone",)))
    pictured = tmp_path / "pictured.flac"
    pictured.write_bytes(flac_bytes(b"x", picture=True))
    assert packs.flac_info(plain).tagged is False
    assert packs.flac_info(tagged).tagged is True
    assert packs.flac_info(pictured).tagged is True
    info = packs.flac_info(tmp_path / "plain.flac")
    assert (info.sample_rate, info.channels, info.bits_per_sample) == (24000, 1, 16)


def test_flac_info_refuses_a_file_that_is_not_flac(tmp_path: Path) -> None:
    path = tmp_path / "x.flac"
    path.write_bytes(b"RIFF....WAVEfmt ")
    with pytest.raises(InputError):
        packs.flac_info(path)


@pytest.mark.parametrize(
    "odd",
    [{"rate": 44100}, {"channels": 2}, {"bits": 24}, {"comments": ("TITLE=x",)}],
    ids=["44.1kHz", "stereo", "24-bit", "tagged"],
)
def test_render_refuses_a_pack_whose_clips_differ_or_carry_tags(
    private_root: root_mod.Root, odd: dict, capsys: pytest.CaptureFixture[str]
) -> None:
    segs = [segment(n) for n in range(1, 5)]
    write_run(private_root, "r1", segs)
    write_clip(private_root, segs[2]["audio"], **odd)
    assert run_cli(private_root, "pack", "labels", "--run", "r1", "--seed", str(SEED)) == 2
    err = capsys.readouterr().err
    assert ("formats" in err) or ("carry tags" in err)
    assert list((private_root.path / "packs").iterdir()) == []


# --- the blind check ------------------------------------------------------------------------


def _plant(root: root_mod.Root, where: str) -> tuple[str, tuple[str, ...]]:
    """Plant one name where `forbidden_names` finds it; return it and the extra names."""
    if where == "teacher":
        return "indextts2", ("indextts2",)
    if where == "student":
        (root.path / "students" / "pocket-ft-s7").mkdir(parents=True)
        return "pocket-ft-s7", ()
    if where == "family":
        return "styletts2", ()
    if where == "source":
        row = {"source_id": "src-0001", "path": "seed/raw/Some Show/Episode Nine zz.mkv"}
        write_jsonl(root.path / "seed" / "inventory.jsonl", [row])
        return "Episode Nine zz", ()
    (root.path / "guard").mkdir(exist_ok=True)
    (root.path / "guard" / "terms.txt").write_text("# names\nzorblatt\n", encoding="utf-8")
    return "Zorblatt", ()


@pytest.mark.parametrize("where", ["teacher", "student", "family", "source", "term"])
def test_render_refuses_a_page_naming_a_planted_name(
    private_root: root_mod.Root, where: str
) -> None:
    name, extra = _plant(private_root, where)
    forbidden = packs.forbidden_names(private_root, extra)
    template = packs.load_template()
    clean = {"pack": "p", "kind": "labels", "groups": [{"id": "g1", "heading": "Set 1"}]}
    packs.render(template, clean, forbidden)
    planted = {**clean, "groups": [{"id": "g1", "heading": f"Set 1 {name}"}]}
    with pytest.raises(InputError) as caught:
        packs.render(template, planted, forbidden)
    assert name.lower() not in str(caught.value).lower()


# --- labels -----------------------------------------------------------------------------------


def _labels_segments() -> list[dict]:
    segs = [segment(n, state=SPEAKING_STATES[n % 5]) for n in range(1, 9)]
    group_a = "src-0001/spk-03"
    segs += [segment(n, cluster="flagged", diar="spk-03", flag_group=group_a) for n in range(9, 14)]
    group_b = "src-0002/spk-03"
    segs += [
        segment(n, source="src-0002", cluster="flagged", diar="spk-03", flag_group=group_b)
        for n in range(14, 16)
    ]
    # An off-centre segment: no diarized cluster, its own id as its flag group.
    segs.append(segment(16, diar=None, cluster="flagged", flag_group="seg-000016"))
    segs.append(segment(17, kept=False))
    return segs


def test_labels_pack_layout_order_and_byte_identical_clips(
    private_root: root_mod.Root,
) -> None:
    segs = _labels_segments()
    write_run(private_root, "r1", segs)
    assert run_cli(private_root, "pack", "labels", "--run", "r1", "--seed", str(SEED)) == 0
    directory = pack_dir(private_root)
    assert (directory / "index.html").is_file()
    assert (private_root.path / "packs" / f"{packs.pack_id(SEED)}.key" / "key.json").is_file()
    assert not (directory / "key.json").exists()
    data, key = page_of(private_root), key_of(private_root)
    assert data["columns"] == ["label", "state", "not_her"]
    kept = [s for s in segs if s["kept"]]
    pairs = items_of(data)
    assert len(pairs) == len(kept)
    labels = sorted(item["label"] for _, item in pairs)
    assert labels == [f"c{n:02d}" for n in range(1, len(kept) + 1)]
    assert labels != [item["label"] for _, item in pairs]  # seeded, not in label order
    by_id = {s["segment_id"]: s for s in kept}
    for _, item in pairs:
        info = key["items"][item["label"]]
        seg = by_id[info["segment_id"]]
        assert item["defaults"]["state"] == seg["state_proposed"]
        clip = directory / "clips" / f"{item['label']}.flac"
        assert clip.read_bytes() == (private_root.path / seg["audio"]).read_bytes()
    # Three segments of each multi-segment flag group lead, one page group per flag group.
    leads = data["groups"][:2]
    lead_groups = [{key["items"][i["label"]]["flag_group"] for i in g["items"]} for g in leads]
    assert sorted(len(g["items"]) for g in leads) == [2, 3]
    assert sorted(next(iter(g)) for g in lead_groups) == ["src-0001/spk-03", "src-0002/spk-03"]
    assert all(len(g) == 1 for g in lead_groups)
    page = (directory / "index.html").read_text(encoding="utf-8")
    assert "seg-0000" not in page and "src-000" not in page


def test_labels_import_drops_groups_and_marked_segments(
    private_root: root_mod.Root, tmp_path: Path
) -> None:
    segs = _labels_segments()
    write_run(private_root, "r1", segs)
    assert run_cli(private_root, "pack", "labels", "--run", "r1", "--seed", str(SEED)) == 0
    data, key = page_of(private_root), key_of(private_root)
    marked = {"seg-000002", "seg-000010", "seg-000016"}  # a primary, group A, off-centre

    def fill(group: dict, item: dict) -> dict:
        sid = key["items"][item["label"]]["segment_id"]
        state = "excited" if sid == "seg-000001" else item["defaults"]["state"]
        return {"state": state, "not_her": 1 if sid in marked else 0}

    assert import_ratings(private_root, answer(data, fill), tmp_path) == 0
    text = (private_root.path / "seed" / "labels.csv").read_text(encoding="utf-8")
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == ["segment_id", "state"]
    got = dict(rows[1:])
    expected_ids = {f"seg-{n:06d}" for n in (1, 3, 4, 5, 6, 7, 8, 14, 15)}
    # Group A (seg 9-13) goes with one mark; group B, which shares the cluster label spk-03 in
    # another episode, stays; the off-centre segment's drop removes only itself.
    assert set(got) == expected_ids
    assert got["seg-000001"] == "excited"
    assert got["seg-000003"] == SPEAKING_STATES[3]


def test_labels_group_with_no_mark_is_confirmed(
    private_root: root_mod.Root, tmp_path: Path
) -> None:
    segs = _labels_segments()
    write_run(private_root, "r1", segs)
    assert run_cli(private_root, "pack", "labels", "--run", "r1", "--seed", str(SEED)) == 0
    data = page_of(private_root)
    text = answer(data, lambda g, i: {"state": i["defaults"]["state"], "not_her": 0})
    assert import_ratings(private_root, text, tmp_path) == 0
    rows = (private_root.path / "seed" / "labels.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1 + len([s for s in segs if s["kept"]])


# --- the seal -----------------------------------------------------------------------------------


def _sealed(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args: object, **_kwargs: object) -> dict:
        raise AssertionError("the key was read before the ratings were complete")

    monkeypatch.setattr(packs, "read_key", refuse)


@pytest.mark.parametrize("damage", ["missing-row", "bad-state", "bad-header", "empty-flag"])
def test_incomplete_ratings_exit_2_with_the_key_sealed(
    private_root: root_mod.Root,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    write_run(private_root, "r1", [segment(n) for n in range(1, 5)])
    assert run_cli(private_root, "pack", "labels", "--run", "r1", "--seed", str(SEED)) == 0
    data = page_of(private_root)
    text = answer(data, lambda g, i: {"state": "calm", "not_her": 0})
    lines = text.splitlines()
    if damage == "missing-row":
        text = "\n".join(lines[:-1]) + "\n"
    elif damage == "bad-state":
        text = text.replace(",calm,", ",grumpy,", 1)
    elif damage == "bad-header":
        text = "label,state\n" + "\n".join(lines[1:]) + "\n"
    else:
        text = "\n".join([lines[0], lines[1].rsplit(",", 1)[0] + ",", *lines[2:]]) + "\n"
    _sealed(monkeypatch)
    assert import_ratings(private_root, text, tmp_path) == 2
    assert not (private_root.path / "seed" / "labels.csv").exists()
    _, err = capsys.readouterr()
    assert "the key stays sealed" in err


def test_import_refuses_an_unknown_pack(private_root: root_mod.Root, tmp_path: Path) -> None:
    assert import_ratings(private_root, "label\n", tmp_path, seed=99) == 2


def test_a_pack_is_built_once(private_root: root_mod.Root) -> None:
    write_run(private_root, "r1", [segment(n) for n in range(1, 3)])
    assert run_cli(private_root, "pack", "labels", "--run", "r1", "--seed", str(SEED)) == 0
    assert run_cli(private_root, "pack", "labels", "--run", "r1", "--seed", str(SEED)) == 2


# --- clean-ab ---------------------------------------------------------------------------------


def test_clean_ab_pairs_in_seeded_order_and_import(
    private_root: root_mod.Root, tmp_path: Path
) -> None:
    before = [segment(n, run="r1") for n in range(1, 13)]
    after = [segment(n, run="c1") for n in range(1, 13)]
    after[0]["kept"] = False
    write_run(private_root, "r1", before)
    write_run(private_root, "c1", after)
    assert (
        run_cli(private_root, "pack", "clean-ab", "--before", "r1", "--after", "c1", "--seed", "3")
        == 0
    )
    data, key = page_of(private_root, 3), key_of(private_root, 3)
    pairs = items_of(data)
    assert len(pairs) == 11
    assert data["columns"] == ["label", "preferred", "voice_changed"]
    orders = set()
    for _, item in pairs:
        assert [c["name"] for c in item["clips"]] == ["A", "B"]
        info = key["items"][item["label"]]
        orders.add(info["A"])
        for clip in item["clips"]:
            side = info[clip["name"]]
            run = "r1" if side == "before" else "c1"
            src = private_root.path / "seed" / "runs" / run / "audio" / f"{info['segment_id']}.flac"
            packed = pack_dir(private_root, 3) / "clips" / f"{clip['label']}.flac"
            assert packed.read_bytes() == src.read_bytes()
    assert orders == {"before", "after"}

    def fill(group: dict, item: dict) -> dict:
        info = key["items"][item["label"]]
        after_side = "A" if info["A"] == "after" else "B"
        changed = 1 if info["segment_id"] == "seg-000002" else 0
        return {"preferred": after_side, "voice_changed": changed}

    assert import_ratings(private_root, answer(data, fill), tmp_path, seed=3) == 0
    review = json.loads(
        (private_root.path / "seed" / "runs" / "c1" / "clean_review.json").read_text()
    )
    assert review["counts"] == {
        "pairs": 11,
        "after_preferred": 11,
        "before_preferred": 0,
        "voice_changed": 1,
    }
    changed = [row for row in review["segments"] if row["voice_changed"]]
    assert [row["segment_id"] for row in changed] == ["seg-000002"]


# --- text ---------------------------------------------------------------------------------------


def _lines(per_state: int) -> list[dict]:
    return [
        {"line_id": f"{state}-{n:03d}", "state": state, "text": f"Line {n} for the {state} set."}
        for state in SPEAKING_STATES
        for n in range(per_state)
    ]


def test_text_pack_shows_30_lines_per_state_and_import_applies_the_gate(
    private_root: root_mod.Root, tmp_path: Path
) -> None:
    write_jsonl(private_root.path / "text" / "v1" / "lines.jsonl", _lines(35))
    assert run_cli(private_root, "pack", "text", "--version", "v1", "--seed", str(SEED)) == 0
    data, key = page_of(private_root), key_of(private_root)
    assert data["columns"] == ["label", "verdict", "reason"]
    assert not (pack_dir(private_root) / "clips").exists() or not any(
        (pack_dir(private_root) / "clips").iterdir()
    )
    per_state: dict[str, int] = {}
    for _, item in items_of(data):
        info = key["items"][item["label"]]
        per_state[info["state"]] = per_state.get(info["state"], 0) + 1
        assert item["text"].startswith("Line ")
        assert item["clips"] == []
    assert per_state == {state: 30 for state in SPEAKING_STATES}
    rejected = {"calm": 3, "sleepy": 4}
    seen: dict[str, int] = {}

    def fill(group: dict, item: dict) -> dict:
        state = key["items"][item["label"]]["state"]
        seen[state] = seen.get(state, 0) + 1
        if seen[state] <= rejected.get(state, 0):
            return {"verdict": "reject", "reason": "awkward, stiff"}
        return {"verdict": "accept"}

    assert import_ratings(private_root, answer(data, fill), tmp_path) == 0
    review = json.loads((private_root.path / "text" / "v1" / "review.json").read_text())
    assert review["states"]["calm"] == {"n": 30, "rejected": 3, "pass": True}
    assert review["states"]["sleepy"] == {"n": 30, "rejected": 4, "pass": False}
    assert review["states"]["neutral"]["pass"] is True
    assert review["thresholds"]["text.reject_max"] == 3


def test_text_reject_without_a_reason_is_incomplete(
    private_root: root_mod.Root, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_jsonl(private_root.path / "text" / "v1" / "lines.jsonl", _lines(2))
    assert run_cli(private_root, "pack", "text", "--version", "v1", "--seed", str(SEED)) == 0
    text = answer(page_of(private_root), lambda g, i: {"verdict": "reject"})
    _sealed(monkeypatch)
    assert import_ratings(private_root, text, tmp_path) == 2


# --- bake-off -----------------------------------------------------------------------------------

BAKE = "teachers/bakeoff/b1"
TEACHERS = ("indextts2", "cosyvoice3")
BAKE_STATES = ("neutral", "calm", "awake", "sleepy")


def _bakeoff_takes() -> list[dict]:
    rows = []
    for state in BAKE_STATES:
        for line in (f"{state}-L1", f"{state}-L2"):
            for teacher in TEACHERS:
                for seed in range(3):
                    # neutral: indextts2 keeps only 2 of its 6 takes (weak by keep rate).
                    kept = not (state == "neutral" and teacher == "indextts2" and seed > 0)
                    if state == "neutral" and teacher == "indextts2" and line.endswith("L1"):
                        kept = kept and seed == 0
                    rows.append(
                        take(line, state, teacher, seed, base=BAKE, kept=kept, selected=seed == 1)
                    )
    return rows


def test_bakeoff_pack_rates_every_teachers_selected_take(private_root: root_mod.Root) -> None:
    takes = _bakeoff_takes()
    write_takes(private_root, f"{BAKE}/takes.jsonl", takes)
    assert run_cli(private_root, "pack", "bakeoff", "--run", "b1", "--seed", str(SEED)) == 0
    data, key = page_of(private_root), key_of(private_root)
    assert tuple(data["columns"]) == packs.RATINGS_COLUMNS
    assert data["fields"]["natural"]["type"] == "scale"
    assert len(data["groups"]) == len(BAKE_STATES) * 2
    selected = {t["take_id"]: t for t in takes if t["selected"]}
    for group in data["groups"]:
        infos = [key["items"][i["label"]] for i in group["items"]]
        assert sorted(info["teacher"] for info in infos) == sorted(TEACHERS)
        assert {info["line_id"] for info in infos} == {group["id"]}
        for info, item in zip(infos, group["items"], strict=True):
            assert info["take_id"] in selected
            src = private_root.path / selected[info["take_id"]]["audio"]
            packed = pack_dir(private_root) / "clips" / f"{item['label']}.flac"
            assert packed.read_bytes() == src.read_bytes()
    page = (pack_dir(private_root) / "index.html").read_text(encoding="utf-8")
    for teacher in TEACHERS:
        assert teacher not in page


def test_bakeoff_import_writes_choice_with_weak_states_and_gains(
    private_root: root_mod.Root, tmp_path: Path
) -> None:
    write_takes(private_root, f"{BAKE}/takes.jsonl", _bakeoff_takes())
    gains = {"indextts2": 1.25, "cosyvoice3": 0.8}
    (private_root.path / BAKE / "gains.json").write_text(json.dumps(gains), encoding="utf-8")
    assert run_cli(private_root, "pack", "bakeoff", "--run", "b1", "--seed", str(SEED)) == 0
    data, key = page_of(private_root), key_of(private_root)

    def fill(group: dict, item: dict) -> dict:
        info = key["items"][item["label"]]
        state, teacher = info["state"], info["teacher"]
        line2 = info["line_id"].endswith("L2")
        if state in ("neutral", "calm"):
            winner = "indextts2"
            feeling = 4
        elif state == "sleepy":
            winner = "cosyvoice3"
            feeling = 3
        else:  # awake: one line each, cosyvoice3 rated more natural
            winner = "indextts2" if line2 else "cosyvoice3"
            feeling = 4
        natural = 5 if (state == "awake" and teacher == "cosyvoice3") else 3
        return {
            "natural": natural,
            "feeling": feeling,
            "preferred": 1 if teacher == winner else 0,
        }

    assert import_ratings(private_root, answer(data, fill), tmp_path) == 0
    choice = json.loads((private_root.path / BAKE / "choice.json").read_text())
    states = choice["states"]
    assert states["calm"]["teacher"] == "indextts2"
    assert states["calm"]["weak"] is False
    assert states["calm"]["gain"] == 1.25
    assert states["sleepy"]["teacher"] == "cosyvoice3"
    assert states["sleepy"]["weak"] is True  # by rating: mean feeling 3 < 3.5
    assert states["sleepy"]["gain"] == 0.8
    assert states["neutral"]["teacher"] == "indextts2"
    assert states["neutral"]["weak"] is True  # by keep rate: 2 of 6 takes pass
    assert states["neutral"]["keep_rate"] == pytest.approx(2 / 6)
    assert states["awake"]["teacher"] == "cosyvoice3"  # a tie on preferred goes to natural
    assert choice["thresholds"]["bakeoff.weak_rating_below"] == 3.5


def test_bakeoff_import_needs_gains(private_root: root_mod.Root, tmp_path: Path) -> None:
    write_takes(private_root, f"{BAKE}/takes.jsonl", _bakeoff_takes())
    assert run_cli(private_root, "pack", "bakeoff", "--run", "b1", "--seed", str(SEED)) == 0
    data = page_of(private_root)

    def fill(group: dict, item: dict) -> dict:
        first = group["items"][0]["label"] == item["label"]
        return {"natural": 3, "feeling": 4, "preferred": 1 if first else 0}

    assert import_ratings(private_root, answer(data, fill), tmp_path) == 2
    assert not (private_root.path / BAKE / "choice.json").exists()


# --- spot check -------------------------------------------------------------------------------

BATCH = "corpus/v1/batches/b3"


def test_spotcheck_samples_50_selected_takes_per_state_and_import(
    private_root: root_mod.Root, tmp_path: Path
) -> None:
    takes = []
    for state in ("calm", "sleepy"):
        for n in range(60):
            takes.append(take(f"{state}-{n}", state, "indextts2", 0, base=BATCH, selected=True))
        takes.append(take(f"{state}-x", state, "indextts2", 1, base=BATCH, selected=False))
    write_takes(private_root, f"{BATCH}/takes.jsonl", takes)
    assert (
        run_cli(private_root, "pack", "spotcheck", "--takes", f"{BATCH}/takes.jsonl", "--seed", "5")
        == 0
    )
    data, key = page_of(private_root, 5), key_of(private_root, 5)
    assert data["columns"] == ["label", "verdict", "reason"]
    infos = [key["items"][item["label"]] for _, item in items_of(data)]
    assert sum(info["state"] == "calm" for info in infos) == 50
    assert sum(info["state"] == "sleepy" for info in infos) == 50
    selected = {t["take_id"] for t in takes if t["selected"]}
    assert {info["take_id"] for info in infos} <= selected
    rejects = {"calm": 5, "sleepy": 6}
    seen: dict[str, int] = {}

    def fill(group: dict, item: dict) -> dict:
        state = key["items"][item["label"]]["state"]
        seen[state] = seen.get(state, 0) + 1
        if seen[state] <= rejects[state]:
            return {"verdict": "reject", "reason": "buzz"}
        return {"verdict": "accept"}

    assert import_ratings(private_root, answer(data, fill), tmp_path, seed=5) == 0
    verdicts = json.loads((private_root.path / BATCH / "spotcheck.json").read_text())
    assert verdicts["states"]["calm"] == {"n": 50, "rejected": 5, "accepted": True}
    assert verdicts["states"]["sleepy"] == {"n": 50, "rejected": 6, "accepted": False}
    assert verdicts["accepted"] is False
    assert len(verdicts["rejected"]) == 11


# --- ABX ----------------------------------------------------------------------------------------


def test_abx_pack_and_tallies(private_root: root_mod.Root, tmp_path: Path) -> None:
    base = "bench/voice-data/run1"
    systems = ("seed", "teacher:indextts2", "student:pocket-ft")
    lines = []
    for line in ("L1", "L2"):
        clips = {}
        for system in systems:
            rel = f"{base}/clips/{line}-{system.replace(':', '-')}.flac"
            write_clip(private_root, rel)
            clips[system] = rel
        lines.append({"line_id": line, "state": "calm", "clips": clips})
    manifest = private_root.path / base / "abx.json"
    manifest.write_text(json.dumps({"lines": lines}), encoding="utf-8")
    argv = ["pack", "abx", "--manifest", f"{base}/abx.json", "--seed", "4"]
    assert run_cli(private_root, *argv) == 0
    data, key = page_of(private_root, 4), key_of(private_root, 4)
    assert data["columns"] == ["label", "line_id", "natural", "feeling", "preferred", "x_is"]
    trials = [(g, i) for g, i in items_of(data) if key["items"][i["label"]]["type"] == "abx"]
    ratings = [(g, i) for g, i in items_of(data) if key["items"][i["label"]]["type"] == "rating"]
    assert len(trials) == 6 and len(ratings) == 6
    for _, item in trials:
        assert [c["name"] for c in item["clips"]] == ["A", "B", "X"]
        info = key["items"][item["label"]]
        clip_dir = pack_dir(private_root, 4) / "clips"
        x_bytes = (clip_dir / f"{item['clips'][2]['label']}.flac").read_bytes()
        same = item["clips"][0 if info["X"] == "A" else 1]["label"]
        assert x_bytes == (clip_dir / f"{same}.flac").read_bytes()
    page = (pack_dir(private_root, 4) / "index.html").read_text(encoding="utf-8")
    assert "pocket-ft" not in page and "indextts2" not in page
    wrong = trials[0][1]["label"]

    def fill(group: dict, item: dict) -> dict:
        info = key["items"][item["label"]]
        if info["type"] == "abx":
            right = info["X"]
            return {"x_is": ("B" if right == "A" else "A") if item["label"] == wrong else right}
        preferred = 1 if info["system"] == "seed" else 0
        return {"natural": 4, "feeling": 3, "preferred": preferred}

    assert import_ratings(private_root, answer(data, fill), tmp_path, seed=4) == 0
    tallies = json.loads((private_root.path / base / "abx_tallies.json").read_text())
    assert sum(row["trials"] for row in tallies["abx"]) == 6
    assert sum(row["correct"] for row in tallies["abx"]) == 5
    assert tallies["ratings"]["seed"]["preferred"] == 2
    assert tallies["ratings"]["student:pocket-ft"]["natural_mean"] == 4


# --- privacy ------------------------------------------------------------------------------------


def _snapshot(base: Path) -> dict[Path, bytes]:
    return {p: p.read_bytes() for p in base.rglob("*") if p.is_file()}


def test_pack_commands_leak_no_canary(
    private_root: root_mod.Root,
    tmp_path: Path,
    canaries,  # the conftest fixture's Canaries
    assert_no_canary: Callable[[str], None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    segs = [segment(n, transcript=canaries.transcript) for n in range(1, 6)]
    write_run(private_root, "r1", segs)
    row = {"source_id": "src-0001", "path": f"seed/raw/Show/{canaries.source_name}"}
    write_jsonl(private_root.path / "seed" / "inventory.jsonl", [row])
    terms = private_root.path / "guard" / "terms.txt"
    terms.write_text(canaries.term + "\n", encoding="utf-8")
    write_jsonl(private_root.path / "text" / "v1" / "lines.jsonl", _lines(3))
    before = _snapshot(tmp_path)
    argvs = [
        ["pack", "labels", "--run", "r1", "--seed", str(SEED)],
        ["pack", "text", "--version", "v1", "--seed", "8"],
    ]
    for argv in argvs:
        assert run_cli(private_root, *argv) == 0
    data = page_of(private_root)
    ratings = tmp_path / "ratings.csv"
    ratings.write_text(answer(data, lambda g, i: {"state": "calm", "not_her": 0}), encoding="utf-8")
    import_argv = ["pack", "import", "--pack", packs.pack_id(SEED), "--ratings", str(ratings)]
    assert run_cli(private_root, *import_argv) == 0
    out, err = capsys.readouterr()
    assert_no_canary(out)
    assert_no_canary(err)
    for argv in [*argvs, import_argv]:
        assert_no_canary(" ".join(argv))
    after = _snapshot(tmp_path)
    written = [p for p, content in after.items() if before.get(p) != content]
    assert any(p.name == "index.html" for p in written)
    assert any(p.name == "labels.csv" for p in written)
    for path in written:
        assert_no_canary(path.read_bytes().decode("utf-8", errors="replace"))


def test_text_pack_refuses_a_line_holding_a_private_term(
    private_root: root_mod.Root,
    canaries,  # the conftest fixture's Canaries
    assert_no_canary: Callable[[str], None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    term = canaries.term
    (private_root.path / "guard" / "terms.txt").write_text(term + "\n", encoding="utf-8")
    lines = _lines(2)
    lines[0]["text"] = f"Say hello to {term} today."
    write_jsonl(private_root.path / "text" / "v1" / "lines.jsonl", lines)
    assert run_cli(private_root, "pack", "text", "--version", "v1", "--seed", str(SEED)) == 2
    out, err = capsys.readouterr()
    assert "would name 1 " in err
    assert_no_canary(out + err)
    assert list((private_root.path / "packs").iterdir()) == []


@pytest.mark.parametrize("as_file", [False, True], ids=["git-dir", "git-file"])
def test_every_pack_writer_refuses_a_root_in_a_git_work_tree(
    private_root: root_mod.Root,
    tmp_path: Path,
    as_file: bool,
    capsys: pytest.CaptureFixture[str],
) -> None:
    write_run(private_root, "r1", [segment(n) for n in range(1, 4)])
    assert run_cli(private_root, "pack", "labels", "--run", "r1", "--seed", "1") == 0
    data = page_of(private_root, 1)
    text = answer(data, lambda g, i: {"state": "calm", "not_her": 0})
    git = tmp_path / ".git"
    if as_file:
        git.write_text("gitdir: elsewhere\n", encoding="utf-8")
    else:
        git.mkdir()
    write_jsonl(private_root.path / "text" / "v1" / "lines.jsonl", _lines(2))
    assert run_cli(private_root, "pack", "labels", "--run", "r1", "--seed", "2") == 2
    assert run_cli(private_root, "pack", "text", "--version", "v1", "--seed", "3") == 2
    assert not pack_dir(private_root, 2).exists() and not pack_dir(private_root, 3).exists()
    ratings = tmp_path / "r.csv"
    ratings.write_text(text, encoding="utf-8")
    argv = ["pack", "import", "--pack", packs.pack_id(1), "--ratings", str(ratings)]
    assert run_cli(private_root, *argv) == 2
    assert not (private_root.path / "seed" / "labels.csv").exists()
    assert capsys.readouterr().err.count("holds .git") == 3


def test_a_clip_outside_the_root_is_refused(private_root: root_mod.Root) -> None:
    segs = [segment(n) for n in range(1, 3)]
    segs[1]["audio"] = "../elsewhere.flac"
    write_jsonl(private_root.path / "seed" / "runs" / "r1" / "segments.jsonl", segs)
    write_clip(private_root, segs[0]["audio"])
    (private_root.path.parent / "elsewhere.flac").write_bytes(flac_bytes(b"x"))
    assert run_cli(private_root, "pack", "labels", "--run", "r1", "--seed", str(SEED)) == 2
    assert not os.listdir(private_root.path / "packs")
