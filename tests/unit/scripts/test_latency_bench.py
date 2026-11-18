"""Tests for the voice latency bench, `scripts.voice_bench.latency_bench`.

The analysis runs over recorded `turn_metrics` lines, built from the protocol contract's own
`turn_metrics` vector, so a line the bench reads here is one the server can send. The driver runs
against `FakeServer`, a scripted socket, with a fake clock whose `sleep` returns at once, so no
backend, network or real time is involved.
"""

import asyncio
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from backend.realtime.budget import BUDGET_PRESETS, STAGE_FIELDS
from backend.realtime.protocol import (
    Direction,
    Frame,
    MsgType,
    decode_frame,
    encode_frame,
)
from backend.realtime.speech import VoiceInfo
from scripts.voice_bench import latency_bench as bench
from scripts.voice_fixtures import generate_fixtures as gen

REPO = Path(__file__).resolve().parents[3]
VECTORS = json.loads(
    (REPO / "tests" / "fixtures" / "realtime_protocol_v2" / "vectors.json").read_text("utf-8")
)
BASE = next(
    v["message"] for v in VECTORS["control"]["server_to_client"] if v["name"] == "turn_metrics"
)
FLOOR = BUDGET_PRESETS["floor"]


def full_budget(preset: str = "floor") -> dict:
    budget = BUDGET_PRESETS[preset]
    return {"preset": preset, **{f: getattr(budget, f) for f in bench.BUDGET_ROWS}}


def metrics(**overrides) -> dict:
    message = copy.deepcopy(BASE)
    message["budget"] = full_budget()
    for key, value in overrides.items():
        message[key] = value
    return message


def line(message: dict) -> str:
    return json.dumps(message)


def table_row(lines: list[str], header: str, row: str) -> list[str]:
    """The cells of `row` in the table under the group header that starts with `header`."""
    start = next(i for i, text in enumerate(lines) if text.startswith(header))
    for text in lines[start + 1 :]:
        if text.startswith("=="):
            break
        cells = text.split()
        if cells and cells[0] == row:
            return cells
    raise AssertionError(f"no row {row!r} under {header!r}")


# -- Percentiles and rows ---------------------------------------------------------------------


class TestPercentile:
    def test_nearest_rank_over_ten_values(self):
        values = [100, 10, 90, 20, 80, 30, 70, 40, 60, 50]

        assert bench.percentile(values, 50) == 50
        assert bench.percentile(values, 90) == 90

    def test_one_value_is_every_percentile(self):
        assert bench.percentile([7], 50) == 7
        assert bench.percentile([7], 90) == 7

    def test_no_values_has_no_percentile(self):
        with pytest.raises(ValueError):
            bench.percentile([], 50)


class TestTurnRows:
    def test_each_stage_is_its_timestamp_less_the_previous_one(self):
        rows = bench.turn_rows(bench.parse_metrics_line(line(metrics())))

        assert rows == {
            "end_of_turn_ms": 240,
            "stt_final_ms": 280,
            "context_build_ms": 10,
            "llm_first_token_ms": 250,
            "first_segment_ms": 115,
            "tts_first_chunk_ms": 535,
            "transport_playback_ms": 100,
            "total_ms": 1530,
            "caption_lag_ms": 310,
        }

    def test_stages_over_budget_are_the_ones_the_server_judged(self):
        message = bench.parse_metrics_line(line(metrics()))

        rows = bench.turn_rows(message)

        over = [stage for stage in STAGE_FIELDS if rows[stage] > getattr(FLOOR, stage)]
        assert over == [s for s in message.over_budget if s in STAGE_FIELDS]

    def test_a_missing_stage_is_skipped_and_the_next_spans_the_gap(self):
        timings = dict(BASE["timings"])
        del timings["context_built"]

        rows = bench.turn_rows(bench.parse_metrics_line(line(metrics(timings=timings))))

        assert "context_build_ms" not in rows
        assert rows["llm_first_token_ms"] == 780 - 520

    def test_a_tool_turn_has_its_tool_call_row(self):
        timings = {**BASE["timings"], "tool_call_parsed": 1100}

        rows = bench.turn_rows(
            bench.parse_metrics_line(line(metrics(kind="tool", timings=timings)))
        )

        assert rows["tool_call_ms"] == 1100 - 780

    def test_a_plain_turn_has_no_tool_call_row(self):
        timings = {**BASE["timings"], "tool_call_parsed": 1100}

        rows = bench.turn_rows(bench.parse_metrics_line(line(metrics(timings=timings))))

        assert "tool_call_ms" not in rows

    def test_screen_and_wake_timings_pass_through_when_present(self):
        timings = {
            **BASE["timings"],
            "wake_screen_ms": 180,
            "screen_command_ms": 420,
            "scene_anchor_lead_ms": 90,
        }

        rows = bench.turn_rows(bench.parse_metrics_line(line(metrics(timings=timings))))

        assert rows["wake_screen_ms"] == 180
        assert rows["screen_command_ms"] == 420
        assert rows["scene_anchor_lead_ms"] == 90


class TestParseMetricsLine:
    @pytest.mark.parametrize(
        "text",
        [
            pytest.param("", id="blank"),
            pytest.param("== power_level=full kind=plain ==", id="a-table-line"),
            pytest.param('{"type": "stream_start", "turn_id": "t"}', id="another-message"),
            pytest.param("{not json", id="broken-json"),
        ],
    )
    def test_other_lines_are_not_metrics(self, text):
        assert bench.parse_metrics_line(text) is None

    def test_an_invalid_turn_metrics_line_is_refused_naming_the_field(self):
        bad = metrics()
        bad["power_level"] = "maximum"

        with pytest.raises(bench.BenchError, match="power_level"):
            bench.parse_metrics_line(line(bad))


# -- The report -------------------------------------------------------------------------------


def report_for(messages: list[dict], scene: list[str] | None = None) -> list[str]:
    samples = [bench.sample_from(bench.parse_metrics_line(line(m))) for m in messages]
    pixels = bench.read_scene_log(scene or [])
    return bench.report(samples, pixels)


class TestReport:
    def test_groups_by_power_level_then_kind(self):
        tool_timings = {**BASE["timings"], "tool_call_parsed": 1100}
        lines = report_for(
            [
                metrics(turn_id="a"),
                metrics(turn_id="b"),
                metrics(turn_id="c", power_level="low", kind="tool", timings=tool_timings),
            ]
        )

        headers = [text for text in lines if text.startswith("==")]
        assert headers == [
            "== power_level=full kind=plain turns=2 preset=floor ==",
            "== power_level=low kind=tool turns=1 preset=floor ==",
        ]

    def test_each_row_has_p50_p90_and_its_budget_beside_the_memo_gates(self):
        late = copy.deepcopy(BASE["timings"])
        late["first_played"] = 2600
        lines = report_for([metrics(turn_id=str(n)) for n in range(9)] + [metrics(timings=late)])

        total = table_row(lines, "== power_level=full", "total_ms")
        eot = table_row(lines, "== power_level=full", "end_of_turn_ms")

        assert total == ["total_ms", "10", "1530", "1530", "1470/2100", "850/1200"]
        assert eot == ["end_of_turn_ms", "10", "240", "240", "250", "250"]

    def test_p90_takes_the_slow_tail(self):
        timings = [copy.deepcopy(BASE["timings"]) for _ in range(10)]
        for n, item in enumerate(timings):
            item["first_played"] = 1000 + 100 * n
        lines = report_for([metrics(turn_id=str(n), timings=t) for n, t in enumerate(timings)])

        total = table_row(lines, "== power_level=full", "total_ms")

        assert total[2:4] == ["1400", "1800"]

    def test_caption_and_wake_rows_sit_beside_their_budget_rows(self):
        timings = {**BASE["timings"], "wake_screen_ms": 180}
        lines = report_for([metrics(timings=timings)])

        caption = table_row(lines, "== power_level=full", "caption_lag_ms")
        wake = table_row(lines, "== power_level=full", "wake_screen_ms")

        assert caption[4:] == [
            str(FLOOR.caption_lag_ms),
            str(BUDGET_PRESETS["memo"].caption_lag_ms),
        ]
        assert wake[4:] == [str(FLOOR.wake_screen_ms), str(BUDGET_PRESETS["memo"].wake_screen_ms)]

    def test_stage_2_rows_appear_only_when_present(self):
        plain = report_for([metrics()])
        timings = {**BASE["timings"], "screen_command_ms": 420, "scene_anchor_lead_ms": 90}
        staged = report_for([metrics(timings=timings)])

        assert not any(text.startswith("screen_command_ms") for text in plain)
        assert table_row(staged, "== power_level=full", "screen_command_ms")[4] == str(
            FLOOR.screen_command_ms
        )
        assert table_row(staged, "== power_level=full", "scene_anchor_lead_ms")[4] == "-"

    def test_the_budget_column_is_the_turns_active_preset(self):
        memo = metrics(budget=full_budget("memo"))
        lines = report_for([memo])

        assert lines[0] == "== power_level=full kind=plain turns=1 preset=memo =="
        total = table_row(lines, "== power_level=full", "total_ms")
        assert total[4] == "850/1200"

    def test_cached_token_share_is_the_mean_over_turns_with_usage(self):
        def prompt(prompt_tokens, cached_tokens):
            return {
                **BASE["prompt"],
                "prompt_tokens": prompt_tokens,
                "cached_tokens": cached_tokens,
            }

        lines = report_for(
            [
                metrics(prompt=prompt(1000, 800)),
                metrics(prompt=prompt(1000, 600)),
                metrics(prompt=prompt(None, None)),
            ]
        )

        assert "cached_tokens share: mean 0.700 over 2 turns (1 without usage)" in lines

    def test_numbers_over_budget_change_nothing_but_the_figures(self):
        slow = {key: value * 10 for key, value in BASE["timings"].items()}

        lines = report_for([metrics(timings=slow)])

        assert table_row(lines, "== power_level=full", "total_ms")[2] == "15300"


class TestSceneLog:
    def test_reads_json_and_key_value_lines_and_ignores_the_rest(self):
        pixels = bench.read_scene_log(
            [
                '{"turn_id": "turn-1", "event_to_pixel_ms": 31}',
                "2026-11-18 10:00:00 INFO backend.realtime.scene: event_to_pixel_ms=44 turn_id=turn-2",
                "INFO backend.realtime.scene: event_to_pixel_ms=12",
                "INFO backend.realtime.scene: scene op sent",
                "",
            ]
        )

        assert pixels == [
            bench.PixelReport("turn-1", 31),
            bench.PixelReport("turn-2", 44),
            bench.PixelReport(None, 12),
        ]

    def test_event_to_pixel_joins_its_turns_group_beside_its_budget_row(self):
        lines = report_for(
            [metrics(turn_id="turn-1")],
            ['{"turn_id": "turn-1", "event_to_pixel_ms": 31}', "event_to_pixel_ms=60"],
        )

        row = table_row(lines, "== power_level=full", "event_to_pixel_ms")
        assert row == ["event_to_pixel_ms", "1", "31", "31", "50", "50"]
        assert "scene log: 2 event_to_pixel reports, 1 matched to a turn" in lines
        assert "event_to_pixel_ms over every report: n=2 p50=31 p90=60" in lines


# -- The command line over recorded lines -----------------------------------------------------


class TestReplay:
    def test_replay_prints_the_tables_and_exits_0(self, tmp_path, capsys):
        recorded = tmp_path / "latency_bench.jsonl"
        recorded.write_text(
            "\n".join(
                [line(metrics(turn_id="a")), "== a table line ==", line(metrics(turn_id="b"))]
            )
            + "\n",
            encoding="utf-8",
        )

        code = bench.main(["--replay", str(recorded)])

        out = capsys.readouterr().out.splitlines()
        assert code == 0
        assert "== power_level=full kind=plain turns=2 preset=floor ==" in out

    def test_replay_reads_the_scene_log(self, tmp_path, capsys):
        recorded = tmp_path / "latency_bench.jsonl"
        recorded.write_text(line(metrics(turn_id="turn-1")) + "\n", encoding="utf-8")
        scene = tmp_path / "scene.log"
        scene.write_text("event_to_pixel_ms=40 turn_id=turn-1\n", encoding="utf-8")

        code = bench.main(["--replay", str(recorded), "--scene-log", str(scene)])

        out = capsys.readouterr().out.splitlines()
        assert code == 0
        assert "scene log: 1 event_to_pixel reports, 1 matched to a turn" in out

    def test_replay_skips_an_invalid_line_with_a_warning(self, tmp_path, capsys):
        bad = metrics(turn_id="bad")
        bad["kind"] = "other"
        recorded = tmp_path / "latency_bench.jsonl"
        recorded.write_text(line(bad) + "\n" + line(metrics()) + "\n", encoding="utf-8")

        code = bench.main(["--replay", str(recorded)])

        captured = capsys.readouterr()
        assert code == 0
        assert "kind" in captured.err
        assert "turns=1" in captured.out

    def test_url_or_replay_is_required(self):
        with pytest.raises(SystemExit) as excinfo:
            bench.main([])

        assert excinfo.value.code == 2


# -- The driver -------------------------------------------------------------------------------


class TestUplinkFrames:
    def test_cuts_audio_into_20_ms_pcm16_frames_with_rising_seq(self):
        audio = np.full(16000 // 50 * 2 + 100, 0.25, dtype=np.float32)

        frames = bench.uplink_frames(audio, first_seq=5)

        decoded = [decode_frame(frame, direction=Direction.UP) for frame in frames]
        assert [frame.seq for frame in decoded] == [5, 6, 7]
        assert all(frame.msg_type == MsgType.MIC_AUDIO for frame in decoded)
        assert all(len(frame.payload) == 640 for frame in decoded)
        last = np.frombuffer(decoded[-1].payload, dtype="<i2")
        assert np.all(last[:100] == round(0.25 * 32767))
        assert not np.any(last[100:])


class TestPlaybackTracker:
    def test_answers_a_finished_stream_with_its_played_length(self):
        tracker = bench.PlaybackTracker()
        tracker.stream_start({"type": "stream_start", "turn_id": "t1", "audio_tag": 3})
        chunk = Frame(MsgType.AUDIO_CHUNK, 3, 0, 24000, b"\x00\x00" * 2400)

        first = tracker.frame(chunk)
        second = tracker.frame(Frame(MsgType.AUDIO_CHUNK, 3, 1, 24000, b"\x00\x00" * 2400))
        done = tracker.frame(Frame(MsgType.AUDIO_COMPLETE, 3, 2, 24000))

        assert first == [{"type": "playback_progress", "turn_id": "t1", "played_ms": 100}]
        assert second == [{"type": "playback_progress", "turn_id": "t1", "played_ms": 200}]
        assert done == [{"type": "playback_done", "turn_id": "t1", "played_ms": 200}]

    def test_frames_of_an_unknown_tag_are_ignored(self):
        tracker = bench.PlaybackTracker()

        assert tracker.frame(Frame(MsgType.AUDIO_COMPLETE, 9, 0, 24000)) == []


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += max(seconds, 0.0)
        await asyncio.sleep(0)


class FakeServer:
    """A `/ws` peer: answers voice_start, then plays one turn per wake and reports its metrics.

    A turn is two 100 ms chunks and a complete marker; its `turn_metrics` comes only after the
    bench sends `playback_done`, as the real server commits a spoken turn.
    """

    def __init__(self, *, voice: bool = True, refuse: str | None = None) -> None:
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.sent: list[object] = []
        self.turns = 0
        capabilities = {
            "tts_enabled": True,
            "vad_enabled": True,
            "voice": {"spotters": ["endpoint"]} if voice else None,
        }
        self.inbox.put_nowait(json.dumps({"type": "status", "message": "hello"}))
        self.inbox.put_nowait(
            json.dumps({"type": "session_started", "session_id": "s", "capabilities": capabilities})
        )
        self._refuse = refuse

    async def send(self, data: str | bytes) -> None:
        self.sent.append(data)
        if isinstance(data, bytes):
            return
        message = json.loads(data)
        if message["type"] == "voice_start":
            self.inbox.put_nowait(
                json.dumps(
                    {
                        "type": "voice_state",
                        "cause": "answer",
                        "req": message["req"],
                        "active": self._refuse is None,
                        "refused": self._refuse,
                    }
                )
            )
        elif message["type"] == "wake":
            self.turns += 1
            turn_id = f"turn-{self.turns}"
            self.inbox.put_nowait(
                json.dumps({"type": "stream_start", "turn_id": turn_id, "audio_tag": self.turns})
            )
            for seq in range(2):
                frame = Frame(MsgType.AUDIO_CHUNK, self.turns, seq, 24000, b"\x01\x00" * 2400)
                self.inbox.put_nowait(encode_frame(frame))
            self.inbox.put_nowait(encode_frame(Frame(MsgType.AUDIO_COMPLETE, self.turns, 2, 24000)))
        elif message["type"] == "playback_done":
            self.inbox.put_nowait(line(metrics(turn_id=message["turn_id"])))

    async def recv(self) -> str | bytes:
        return await self.inbox.get()

    def sent_json(self, kind: str) -> list[dict]:
        return [
            json.loads(item)
            for item in self.sent
            if isinstance(item, str) and json.loads(item)["type"] == kind
        ]


def wake_fixture(fixture_id: str, seconds: float = 2.0) -> bench.BenchFixture:
    return bench.BenchFixture(
        id=fixture_id, audio=np.zeros(int(16000 * seconds), np.float32), backlog_ms=1600
    )


def run_drive(server: FakeServer, fixtures, *, rounds: int = 1, emitted: list | None = None):
    clock = FakeClock()
    out = [] if emitted is None else emitted
    return asyncio.run(
        bench.drive(
            server,
            fixtures,
            rounds=rounds,
            turn_timeout_s=30.0,
            emit=out.append,
            warn=lambda text: None,
            sleep=clock.sleep,
            clock=clock,
        )
    )


class TestDrive:
    def test_one_turn_per_fixture_and_round_each_printed_as_a_json_line(self):
        server = FakeServer()
        emitted: list[str] = []

        results = run_drive(
            server, [wake_fixture("w1"), wake_fixture("w2")], rounds=2, emitted=emitted
        )

        assert [message.turn_id for message in results] == ["turn-1", "turn-2", "turn-3", "turn-4"]
        assert [json.loads(text)["turn_id"] for text in emitted] == [
            "turn-1",
            "turn-2",
            "turn-3",
            "turn-4",
        ]

    def test_starts_voice_as_an_endpoint_spotter_and_sends_each_wake_with_its_backlog(self):
        server = FakeServer()

        run_drive(server, [wake_fixture("w1")])

        (start,) = server.sent_json("voice_start")
        assert start["mode"] == "wake"
        assert start["spotter"] == "endpoint"
        assert start["sample_rate"] == 16000
        assert start["frame_ms"] == 20
        (wake,) = server.sent_json("wake")
        assert wake["backlog_ms"] == 1600
        assert wake["keyword"] == "mist"

    def test_answers_the_turns_audio_with_playback_done(self):
        server = FakeServer()

        run_drive(server, [wake_fixture("w1")])

        assert server.sent_json("playback_done") == [
            {"type": "playback_done", "turn_id": "turn-1", "played_ms": 200}
        ]

    def test_streams_the_whole_fixture_after_its_wake(self):
        server = FakeServer()

        run_drive(server, [wake_fixture("w1", seconds=2.0)])

        wake_index = next(
            i for i, item in enumerate(server.sent) if isinstance(item, str) and '"wake"' in item
        )
        frames = [item for item in server.sent[wake_index:] if isinstance(item, bytes)]
        assert len(frames) >= 2.0 * 1000 / 20

    def test_a_server_without_voice_is_a_bench_error(self):
        with pytest.raises(bench.BenchError, match="voice"):
            run_drive(FakeServer(voice=False), [wake_fixture("w1")])

    def test_a_refused_voice_start_is_a_bench_error_naming_the_reason(self):
        with pytest.raises(bench.BenchError, match="power_off"):
            run_drive(FakeServer(refuse="power_off"), [wake_fixture("w1")])


class TestBenchFixtures:
    def test_only_wake_fixtures_are_driven(self, tmp_path):
        tone = np.full(16000, 0.1, np.float32)
        fixtures = [
            gen.complete_fixture("complete-01", "Hello there.", tone),
            gen.wake_fixture("wake-01", "Hey Mist,", "hello.", "plain", tone[:8000], tone),
        ]
        gen.write_fixtures(tmp_path, fixtures, VoiceInfo("fake-tts", "fake-voice", "mist"))

        loaded = bench.load_bench_fixtures(tmp_path)

        (only,) = loaded
        assert only.id == "wake-01"
        assert only.backlog_ms == fixtures[1].entry.wake.name_end_ms
        assert len(only.audio) == len(fixtures[1].audio)
