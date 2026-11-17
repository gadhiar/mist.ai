"""Tests for `sdh scan` and `sdh score`: SDH stream choice, her cues, spans and the recall check.

Every probe result and subtitle track here is synthetic and built in this file: ffprobe JSON as
dicts and SRT as strings, served by a fake of the stage's one subprocess call. No subtitle file
is committed (the `no-voice-data-records` hook refuses one), and the labels are made up. The jobs
run through the real `LocalDocker` with a fake `DockerOps` that runs the stage module in this
process with `--data` pointed at the root, so the argv and the mounts are the real ones.
"""

from __future__ import annotations

import ast
import contextlib
import importlib
import io
import json
import subprocess
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from scripts.voice_data import cli
from scripts.voice_data import root as root_mod
from scripts.voice_data.commands import sdh as sdh_cmd
from scripts.voice_data.records import Prosody, SegmentRecord, SourceRecord, write_jsonl
from scripts.voice_data.stages import sdh as sdh_stage
from scripts.voice_data.stages.sdh import Cue

from .conftest import Canaries

IMAGE = "mist-voice-base:0123456789ab"
LABEL = "SPEAKER-A"
OTHER = "SPEAKER-B"


# Synthetic probe results and subtitle tracks.


def stream(
    index: int,
    codec: str = "subrip",
    *,
    kind: str = "subtitle",
    title: str | None = None,
    language: str | None = "eng",
    hearing_impaired: int = 0,
) -> dict[str, object]:
    """One stream object as `ffprobe -show_streams -of json` gives it."""
    tags: dict[str, str] = {}
    if language is not None:
        tags["language"] = language
    if title is not None:
        tags["title"] = title
    return {
        "index": index,
        "codec_type": kind,
        "codec_name": codec,
        "disposition": {"default": 0, "hearing_impaired": hearing_impaired},
        "tags": tags,
    }


def stamp(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    hours, rest = divmod(millis, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def srt(cues: Sequence[tuple[float, float, str]]) -> str:
    """An SRT track from (start, end, text) triples; text lines are separated by newlines."""
    blocks = [
        f"{n}\n{stamp(start)} --> {stamp(end)}\n{text}\n"
        for n, (start, end, text) in enumerate(cues, 1)
    ]
    return "\n".join(blocks)


@dataclass
class Media:
    """What the fake tools report for one source file."""

    streams: list[dict[str, object]]
    subtitles: dict[int, str] = field(default_factory=dict)


class FakeTools:
    """The stage's `_run`: ffprobe returns the media's streams, ffmpeg the mapped stream as SRT."""

    def __init__(self, media: Mapping[str, Media]) -> None:
        self.media = media
        self.calls: list[list[str]] = []

    def __call__(self, argv: Sequence[str]) -> subprocess.CompletedProcess[bytes]:
        argv = list(argv)
        self.calls.append(argv)
        if argv[0] == "ffprobe":
            media = self.media[Path(argv[-1]).name]
            out = json.dumps({"streams": media.streams}).encode("utf-8")
        elif argv[0] == "ffmpeg":
            media = self.media[Path(argv[argv.index("-i") + 1]).name]
            index = int(argv[argv.index("-map") + 1].split(":")[1])
            assert argv[argv.index("-f") + 1] == "srt" and argv[-1] == "-"
            out = media.subtitles[index].encode("utf-8")
        else:
            raise AssertionError(f"unexpected tool {argv[0]}")
        return subprocess.CompletedProcess(argv, 0, stdout=out, stderr=b"")


# Stream choice.


def test_the_stream_titled_sdh_is_chosen_case_insensitively() -> None:
    streams = [stream(0, "hevc", kind="video"), stream(1), stream(2, title="sdh")]
    choice = sdh_stage.choose_stream(streams)
    assert (choice.index, choice.image_skipped) == (2, 0)


def test_the_hearing_impaired_disposition_chooses_a_stream() -> None:
    choice = sdh_stage.choose_stream([stream(3), stream(4, hearing_impaired=1)])
    assert choice.index == 4


def test_the_lowest_index_wins_among_several() -> None:
    streams = [stream(7, title="SDH"), stream(5, hearing_impaired=1), stream(6, title="Sdh")]
    assert sdh_stage.choose_stream(streams).index == 5


def test_an_image_stream_is_skipped_and_counted() -> None:
    streams = [stream(2, "hdmv_pgs_subtitle", title="SDH"), stream(3, "dvd_subtitle", title="SDH")]
    choice = sdh_stage.choose_stream(streams)
    assert (choice.index, choice.image_skipped) == (None, 2)
    streams.append(stream(4, "ass", hearing_impaired=1))
    choice = sdh_stage.choose_stream(streams)
    assert (choice.index, choice.image_skipped) == (4, 2)


def test_no_sdh_stream_is_none_not_an_error() -> None:
    streams = [stream(0, "aac", kind="audio"), stream(1, title="Forced"), stream(2)]
    choice = sdh_stage.choose_stream(streams)
    assert (choice.index, choice.image_skipped) == (None, 0)


def test_a_stream_in_another_language_is_not_taken() -> None:
    streams = [stream(1, title="SDH", language="spa"), stream(2, title="SDH", language=None)]
    assert sdh_stage.choose_stream(streams).index == 2


# Cue parsing.


def _hers(text: str, labels: Sequence[str] = (LABEL,)) -> tuple[list[Cue], int]:
    return sdh_stage.her_cues(text, labels)


def test_srt_cues_are_parsed_with_their_times() -> None:
    text = srt([(812.4, 815.1, f"[{LABEL}] One."), (3725.25, 3727.0, f"[{LABEL}] Two.")])
    cues, total = _hers(text)
    assert total == 2
    assert [(c.start_s, c.end_s) for c in cues] == [(812.4, 815.1), (3725.25, 3727.0)]


def test_the_label_matches_with_any_case_and_with_or_without_a_colon() -> None:
    text = srt([
        (1.0, 2.0, "[speaker-a] plain."),
        (3.0, 4.0, "[Speaker-A]: with a colon."),
        (5.0, 6.0, "-[SPEAKER-A] after a dash."),
        (7.0, 8.0, f"[{OTHER}] someone else."),
        (9.0, 10.0, "[SPEAKER-AB] a longer label."),
        (11.0, 12.0, "No label at all."),
        (13.0, 14.0, "She says [SPEAKER-A] mid-line."),
    ])  # fmt: skip
    cues, total = _hers(text)
    assert total == 7
    assert [c.start_s for c in cues] == [1.0, 3.0, 5.0]


def test_any_of_several_labels_matches() -> None:
    text = srt([(1.0, 2.0, f"[{OTHER}] one."), (3.0, 4.0, f"[{LABEL}] two.")])
    assert len(_hers(text, (LABEL, OTHER))[0]) == 2


def test_multi_marks_a_second_speaker_line() -> None:
    text = srt([
        (1.0, 2.0, f"-[{LABEL}] Mine.\n- Yours."),
        (3.0, 4.0, f"[{OTHER}] Yours.\n[{LABEL}] Mine."),
        (5.0, 6.0, f"[{LABEL}] Mine, wrapped\nover two lines."),
        (7.0, 8.0, f"[{LABEL}] Mine.\n[sighs]"),
    ])  # fmt: skip
    cues, _ = _hers(text)
    assert [c.multi for c in cues] == [True, True, False, True]


def test_offscreen_marks_her_line_wholly_in_italics() -> None:
    text = srt([
        (1.0, 2.0, f"<i>[{LABEL}] On the phone.</i>"),
        (3.0, 4.0, f"[{LABEL}] <i>Label outside, speech inside.</i>"),
        (5.0, 6.0, f"<i>[{LABEL}] Wrapped over</i>\n<i>two lines.</i>"),
        (7.0, 8.0, f"[{LABEL}] <i>Only</i> partly."),
        (9.0, 10.0, f"[{LABEL}] Not at all."),
        (11.0, 12.0, f"[{LABEL}] Here.\n- <i>[{OTHER}] There.</i>"),
    ])  # fmt: skip
    cues, _ = _hers(text)
    assert [c.offscreen for c in cues] == [True, True, True, False, False, False]


# Spans: the merge, the length window and the held-out split.


def cue(start: float, end: float, multi: bool = False, offscreen: bool = False) -> Cue:
    return Cue(start_s=start, end_s=end, multi=multi, offscreen=offscreen)


def test_cues_with_gaps_of_at_most_half_a_second_merge() -> None:
    cues = [cue(10.0, 12.0), cue(12.5, 14.0), cue(14.6, 18.0), cue(18.0, 19.0, offscreen=True)]
    spans = sdh_stage.source_spans(cues, 3.0, 30.0)
    assert [(s.start_s, s.end_s) for s in spans.merged] == [(10.0, 14.0), (14.6, 19.0)]
    assert [s.offscreen for s in spans.merged] == [False, True]
    assert spans.found == 4 and spans.found_offscreen == 1


def test_a_multi_cue_is_not_merged_and_breaks_a_run() -> None:
    cues = [cue(10.0, 12.0), cue(12.1, 12.3, multi=True), cue(12.4, 15.0)]
    spans = sdh_stage.source_spans(cues, 3.0, 30.0)
    assert [(s.start_s, s.end_s) for s in spans.merged] == [(10.0, 12.0), (12.4, 15.0)]
    assert spans.multi == 1


def test_merged_spans_of_3_to_30_s_are_kept() -> None:
    cues = [cue(0.0, 2.9), cue(10.0, 13.0), cue(20.0, 50.0), cue(60.0, 90.1)]
    spans = sdh_stage.source_spans(cues, 3.0, 30.0)
    assert [(s.start_s, s.end_s) for s in spans.kept] == [(10.0, 13.0), (20.0, 50.0)]


def test_every_fifth_kept_span_is_held_out_and_never_an_anchor() -> None:
    cues = [cue(10.0 * k, 10.0 * k + 4.0) for k in range(12)]
    cues.insert(3, cue(35.0, 35.5))  # a merged span too short to keep, so not counted
    spans = sdh_stage.source_spans(cues, 3.0, 30.0)
    assert len(spans.kept) == 12
    assert [s.start_s for s in spans.held_out] == [40.0, 90.0]
    assert [s.start_s for s in spans.anchors] == [10.0 * k for k in range(12) if k not in (4, 9)]


def test_a_source_with_fewer_than_five_spans_holds_none_out() -> None:
    spans = sdh_stage.source_spans([cue(10.0 * k, 10.0 * k + 4.0) for k in range(4)], 3.0, 30.0)
    assert spans.held_out == () and len(spans.anchors) == 4


def test_spans_are_taken_in_time_order() -> None:
    cues = [cue(10.0 * k, 10.0 * k + 4.0) for k in range(5)][::-1]
    assert sdh_stage.source_spans(cues, 3.0, 30.0).held_out[0].start_s == 40.0


# The commands, through the fake container boundary.


def make_root(base: Path, sources: Sequence[str], labels: Sequence[str] | None) -> root_mod.Root:
    root = root_mod.init_root(base / "voice-root", "private")
    (root.path / "seed" / "raw" / "show").mkdir(parents=True)
    records = []
    for number, name in enumerate(sources, 1):
        (root.path / "seed" / "raw" / "show" / name).write_bytes(b"")
        records.append(
            SourceRecord(
                source_id=f"src-{number:04d}", path=f"seed/raw/show/{name}", container="matroska",
                size_bytes=1, mtime_ns=0, streams=(), stream_index=1, audio_index=0,
                language="eng", codec="eac3", profile=None, bitrate_bps=640000,
                bitrate_from="packets", sample_rate=48000, channels=6, layout="5.1",
                duration_s=1800.0, lossless=False, effective_bandwidth_hz=16000,
                sha256=f"{number:064d}", track=f"seed/tracks/src-{number:04d}.mka",
                primary_channel="FC",
            )  # fmt: skip
        )
    write_jsonl(root, "seed/inventory.jsonl", records)
    if labels is not None:
        config = json.dumps({"labels": list(labels)})
        (root.path / "config" / "sdh.json").write_text(config, encoding="utf-8")
    return root


class StageDocker:
    """A `DockerOps` whose `run_logged` runs the stage module in this process."""

    def __init__(self, root: root_mod.Root, tools: FakeTools | None = None) -> None:
        self.root = root
        self.tools = tools
        self.argvs: list[list[str]] = []

    def capture(self, argv: Sequence[str]) -> str:
        raise AssertionError("an sdh job runs no GPU preflight")

    def check(self, argv: Sequence[str]) -> None:
        raise AssertionError("sdh builds nothing")

    def run_logged(self, argv: Sequence[str], log: Path) -> int:
        self.argvs.append(list(argv))
        at = list(argv).index("-m")
        module, rest = argv[at + 1], list(argv[at + 2 :])
        assert rest[:2] == ["--data", "/data"]
        rest[1] = str(self.root.path)
        stage = importlib.import_module(module)
        with (
            pytest.MonkeyPatch.context() as mp,
            log.open("w", encoding="utf-8") as handle,
            contextlib.redirect_stdout(handle),
            contextlib.redirect_stderr(handle),
        ):
            if self.tools is not None:
                mp.setattr(stage, "_run", self.tools)
            return stage.main(rest)


@contextlib.contextmanager
def _patched(docker: StageDocker) -> Iterator[None]:
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sdh_cmd, "make_docker", lambda: docker)
        mp.setattr(sdh_cmd, "image_tag", lambda family: IMAGE)
        yield


def sdh(root: root_mod.Root, docker: StageDocker, *args: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with _patched(docker), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(["--root", str(root.path), "sdh", *args])
    return code, out.getvalue(), err.getvalue()


def two_sources(text: str) -> dict[str, Media]:
    """Source a has a regular track, an image SDH track and a text SDH track; b has no SDH."""
    a = Media(
        streams=[
            stream(0, "hevc", kind="video"),
            stream(1, "eac3", kind="audio"),
            stream(2),
            stream(3, "hdmv_pgs_subtitle", title="SDH"),
            stream(4, title="SDH"),
        ],
        subtitles={2: srt([(1.0, 2.0, f"[{LABEL}] not the SDH track.")]), 4: text},
    )
    b = Media(streams=[stream(0, "eac3", kind="audio"), stream(1, title="Forced")])
    return {"a.mkv": a, "b.mkv": b}


SCAN_TRACK = srt([
    (10.0, 12.0, f"[{LABEL}] One."),
    (12.4, 14.0, f"[{LABEL}]: two."),
    (20.0, 24.0, f"<i>[{LABEL}] Three.</i>"),
    (30.0, 33.0, f"[{OTHER}] Not hers."),
    (40.0, 42.0, f"-[{LABEL}] Four.\n- Someone else."),
    (50.0, 51.0, "[door closes]"),
])  # fmt: skip


@pytest.fixture
def scanned(tmp_path: Path) -> tuple[root_mod.Root, StageDocker, int, str, str]:
    root = make_root(tmp_path, ["a.mkv", "b.mkv"], [LABEL])
    docker = StageDocker(root, FakeTools(two_sources(SCAN_TRACK)))
    code, out, err = sdh(root, docker, "scan")
    return root, docker, code, out, err


def read_cues(root: root_mod.Root, sid: str) -> list[dict[str, object]]:
    text = (root.path / "seed" / "sdh" / f"{sid}.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_scan_writes_her_cue_times_per_source(scanned) -> None:
    root, _, code, _, err = scanned
    assert code == 0, err
    assert read_cues(root, "src-0001") == [
        {"start_s": 10.0, "end_s": 12.0, "multi": False, "offscreen": False},
        {"start_s": 12.4, "end_s": 14.0, "multi": False, "offscreen": False},
        {"start_s": 20.0, "end_s": 24.0, "multi": False, "offscreen": True},
        {"start_s": 40.0, "end_s": 42.0, "multi": True, "offscreen": False},
    ]
    assert read_cues(root, "src-0002") == []


def test_scan_prints_counts_only(scanned) -> None:
    _, _, code, out, _ = scanned
    assert code == 0
    assert (
        "src-0001: SDH stream 4, 1 image stream skipped, 6 cues, 4 hers (multi 1, offscreen 1)"
        in out
    )
    assert "src-0002: no SDH stream" in out
    assert "sdh scan: 2 sources, 1 with an SDH stream, 1 without, 1 image stream skipped" in out


def test_scan_builds_one_local_score_job_per_source_without_network_or_gpu(scanned) -> None:
    root, docker, code, _, _ = scanned
    assert code == 0
    assert len(docker.argvs) == 2
    for argv, sid in zip(docker.argvs, ("src-0001", "src-0002"), strict=True):
        assert argv[:2] == ["docker", "run"]
        assert "--network" in argv and argv[argv.index("--network") + 1] == "none"
        assert "--gpus" not in argv
        assert IMAGE in argv and "/opt/venv-score/bin/python" in argv
        assert argv[-4:] == ["--step", "scan", "--source", sid]
        for rel in ("seed/raw", "seed/inventory.jsonl", "config/sdh.json"):
            assert f"{root.path / rel}:/data/{rel}:ro" in argv
        assert f"{root.path / 'seed' / 'sdh'}:/data/seed/sdh" in argv
    jobs_ = [json.loads(p.read_text()) for p in sorted((root.path / "runs").glob("job-*/job.json"))]
    assert [j["family"] for j in jobs_] == ["score", "score"]
    assert all(j["gpu"] is False and j["network"] is False for j in jobs_)


def test_a_missing_config_exits_2_naming_the_file(tmp_path: Path) -> None:
    root = make_root(tmp_path, ["a.mkv"], None)
    docker = StageDocker(root, FakeTools(two_sources(SCAN_TRACK)))
    code, _, err = sdh(root, docker, "scan")
    assert code == 2
    assert "config/sdh.json" in err
    assert docker.argvs == []


@pytest.mark.parametrize(
    "content", ['{"labels": []}', '{"names": ["SPEAKER-QZV"]}', "SPEAKER-QZV not json"]
)
def test_a_malformed_config_exits_2_without_its_content(tmp_path: Path, content: str) -> None:
    root = make_root(tmp_path, ["a.mkv"], None)
    (root.path / "config" / "sdh.json").write_text(content, encoding="utf-8")
    docker = StageDocker(root, FakeTools(two_sources(SCAN_TRACK)))
    code, out, err = sdh(root, docker, "scan")
    assert code == 2
    assert "config/sdh.json" in err
    assert "QZV" not in out + err
    assert docker.argvs == []


def test_scan_without_an_inventory_exits_2(tmp_path: Path) -> None:
    root = make_root(tmp_path, ["a.mkv"], [LABEL])
    (root.path / "seed" / "inventory.jsonl").unlink()
    code, _, err = sdh(root, StageDocker(root), "scan")
    assert code == 2 and "ingest" in err


# The recall check.


def segment(n: int, sid: str, start: float, end: float, cluster: str, kept: bool = True):
    return SegmentRecord(
        segment_id=f"seg-{n:06d}", source_id=sid, start_s=start, end_s=end, channel="FC",
        channel_reason=None, separator="none", audio="", diar_cluster="spk-01", cluster=cluster,
        flag_group=None, sim={}, overlap_ms=0, snr_db=30.0, c50_db=40.0, transcripts={},
        asr_agreement_cer=0.0, prosody=Prosody(0.0, 0.0, 0.0, 0.0), arousal=0.0,
        state_proposed="", state=None, set="short", kept=kept, reasons=() if kept else ("snr",),
    )  # fmt: skip


def write_cues(root: root_mod.Root, sid: str, cues: Sequence[Cue]) -> None:
    write_jsonl(root, f"seed/sdh/{sid}.jsonl", [c.to_json() for c in cues])


def scored_root(tmp_path: Path) -> root_mod.Root:
    root = make_root(tmp_path, ["a.mkv", "b.mkv"], [LABEL])
    # src-0001: ten 4 s spans; the 5th (40-44) and the 10th (90-94) are held out.
    write_cues(root, "src-0001", [cue(10.0 * k, 10.0 * k + 4.0) for k in range(10)])
    # src-0002: three spans, so none is held out.
    write_cues(root, "src-0002", [cue(10.0 * k, 10.0 * k + 4.0) for k in range(3)])
    write_jsonl(
        root,
        "seed/runs/r1/segments.jsonl",
        [
            segment(1, "src-0001", 0.0, 4.0, "primary"),  # a span that steered retrieval
            segment(2, "src-0001", 40.0, 42.0, "primary"),
            segment(3, "src-0001", 41.0, 43.0, "primary"),  # overlaps seg 2: 40-43 counts once
            segment(4, "src-0001", 42.0, 44.0, "primary", kept=False),
            segment(5, "src-0001", 92.0, 94.0, "flagged"),
            segment(6, "src-0002", 0.0, 4.0, "primary"),
        ],
    )
    return root


def test_score_measures_the_held_out_spans_only(tmp_path: Path) -> None:
    root = scored_root(tmp_path)
    docker = StageDocker(root)
    code, out, err = sdh(root, docker, "score", "--run", "r1")
    assert code == 0, err
    score = json.loads((root.path / "seed/runs/r1/sdh-score.json").read_text())
    assert score["overall"] == {
        "held_out_spans": 2, "held_out_s": 8.0, "primary_share": 0.375, "flagged_share": 0.25,
    }  # fmt: skip
    by_id = {s["source_id"]: s for s in score["sources"]}
    assert by_id["src-0001"]["primary_share"] == 0.375
    assert by_id["src-0002"] == {
        "source_id": "src-0002", "held_out_spans": 0, "held_out_s": 0.0,
        "primary_share": None, "flagged_share": None,
    }  # fmt: skip
    assert "sdh score r1: 2 held-out spans, 8.0 s; primary 0.375, flagged 0.250" in out
    assert "src-0002: 0 held-out spans, 0.0 s; primary none, flagged none" in out


def test_score_builds_one_local_score_job(tmp_path: Path) -> None:
    root = scored_root(tmp_path)
    docker = StageDocker(root)
    assert sdh(root, docker, "score", "--run", "r1")[0] == 0
    (argv,) = docker.argvs
    assert "--network" in argv and "--gpus" not in argv
    assert f"{root.path / 'seed' / 'sdh'}:/data/seed/sdh:ro" in argv
    segs = "seed/runs/r1/segments.jsonl"
    assert f"{root.path / segs}:/data/{segs}:ro" in argv
    assert f"{root.path / 'seed' / 'runs' / 'r1'}:/data/seed/runs/r1" in argv
    assert argv[-8:] == ["--step", "score", "--run", "r1", "--min-s", "3", "--max-s", "30"]
    job = json.loads(next((root.path / "runs").glob("job-*/job.json")).read_text())
    assert job["family"] == "score" and job["gpu"] is False and job["network"] is False


def test_score_without_a_scan_or_a_run_exits_2(tmp_path: Path) -> None:
    root = make_root(tmp_path, ["a.mkv"], [LABEL])
    docker = StageDocker(root)
    code, _, err = sdh(root, docker, "score", "--run", "r1")
    assert code == 2 and "sdh scan" in err
    write_cues(root, "src-0001", [cue(0.0, 4.0)])
    code, _, err = sdh(root, docker, "score", "--run", "r1")
    assert code == 2 and "segments.jsonl" in err
    assert sdh(root, docker, "score", "--run", "../x")[0] == 2
    assert docker.argvs == []


# Output discipline and host purity.


def test_no_cue_text_or_label_leaves_either_command(tmp_path: Path) -> None:
    canaries = Canaries()
    label = "SPEAKER-QZV"
    track = srt([
        (10.0 + 5.0 * k, 14.0 + 5.0 * k, f"[{label}]: {canaries.transcript} {k}")
        for k in range(6)
    ])  # fmt: skip
    root = make_root(tmp_path, ["a.mkv"], [label])
    media = {"a.mkv": Media(streams=[stream(2, title="SDH")], subtitles={2: track})}
    docker = StageDocker(root, FakeTools(media))
    code, out, err = sdh(root, docker, "scan")
    assert code == 0, err
    write_jsonl(
        root, "seed/runs/r1/segments.jsonl", [segment(1, "src-0001", 30.0, 34.0, "primary")]
    )
    code2, out2, err2 = sdh(root, docker, "score", "--run", "r1")
    assert code2 == 0, err2
    assert len(read_cues(root, "src-0001")) == 6
    planted = {root.path / "config" / "sdh.json"}
    texts = [out, err, out2, err2, *("\n".join(a) for a in docker.argvs)]
    checked = 0
    for path in tmp_path.rglob("*"):
        if path.is_file() and path not in planted:
            texts += [path.relative_to(tmp_path).as_posix(), path.read_bytes().decode("latin-1")]
            checked += 1
    assert checked >= 8
    for text in texts:
        lowered = text.lower()
        assert label.lower() not in lowered
        assert canaries.transcript not in lowered


def test_the_host_command_imports_nothing_from_the_stage() -> None:
    source = Path(sdh_cmd.__file__).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert "stages" not in module.split(".")
        elif isinstance(node, ast.Import):
            assert all("stages" not in a.name.split(".") for a in node.names)
